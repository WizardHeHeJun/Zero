"""Graphiti 语义记忆侧信道测试：工厂回退（本机直跑）+ fake-store 接线（确定性）+ 实机 smoke。

设计见 PRP/计划「引入 Graphiti（深度集成）」。Graphiti 真路径需 Neo4j + LLM，本机无法验证，
故实机用例 importorskip + 连接/LLM env 探测优雅 skip；深度集成的接线（recall→recalled_context
→语言检索）用一个内存版 FakeSemanticStore 在本机做**确定性**单测，无任何外部依赖。
"""

from __future__ import annotations

import builtins
from datetime import UTC, datetime, timedelta

import pytest

from src.agents.language import LanguageAgent
from src.memory.client import MemoryClient
from src.memory.types import Scope
from src.orchestration.memory_recall import MemoryRecallAgent
from src.orchestration.state import AffectState, Stimulus
from src.storage import graph_store as gs
from src.storage.graph_store import StoredFact, build_semantic_store


class FakeSemanticStore:
    """内存版 SemanticStore：记录 episode、search 按 scope/key 返回（不做真语义检索）。

    用于在本机确定性地验证「写 episode → 召回 → 进语言检索」的接线，不依赖 Graphiti/LLM。
    """

    def __init__(self) -> None:
        self.episodes: list[tuple[str, str, str, datetime]] = []

    async def add_episode(self, *, scope: str, key: str, content: str, valid_at: datetime) -> None:
        self.episodes.append((scope, key, content, valid_at))

    async def search(
        self,
        query: str,
        *,
        scope: str,
        key: str | None = None,
        at: datetime | None = None,
        limit: int = 5,
    ) -> list[StoredFact]:
        return [
            StoredFact(scope=s, key=k, content=c, valid_at=v)
            for (s, k, c, v) in self.episodes
            if s == scope and (key is None or k == key)
        ][:limit]


# --------------------------- 工厂回退（本机直跑） --------------------------- #


def test_factory_default_returns_none() -> None:
    """未设 ZERO_SEMANTIC_BACKEND → 无语义后端（None），严格零回归。"""
    assert build_semantic_store("") is None


def test_factory_graphiti_falls_back_without_driver(monkeypatch) -> None:
    """选 graphiti 但驱动不可用（_graphiti_store 返回 None）→ 工厂返回 None，不抛。"""
    monkeypatch.setattr(gs, "_graphiti_store", lambda: None)
    monkeypatch.setenv("ZERO_SEMANTIC_BACKEND", "graphiti")
    assert build_semantic_store() is None


def test_graphiti_store_helper_returns_none_when_import_fails(monkeypatch) -> None:
    """缺 graphiti-core 包时 _graphiti_store 探测失败返回 None（触发工厂回退）。"""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "graphiti_core" or name.startswith("graphiti_core."):
            raise ImportError("simulated missing graphiti-core")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert gs._graphiti_store() is None


# ----------------- MemoryClient 语义 API（fake-store / 无后端） ----------------- #


async def test_write_episode_and_recall_route_to_semantic() -> None:
    fake = FakeSemanticStore()
    mem = MemoryClient(semantic=fake)
    await mem.write_episode("用户喜欢猫", scope=Scope.USER, key="u1")
    assert fake.episodes and fake.episodes[0][2] == "用户喜欢猫"
    facts = await mem.recall("猫", scope=Scope.USER, key="u1")
    assert [f.content for f in facts] == ["用户喜欢猫"]
    assert all(f.scope is Scope.USER for f in facts)


async def test_semantic_api_noop_without_backend() -> None:
    """无语义后端：write_episode no-op、recall 返回 []（零回归）。"""
    mem = MemoryClient()  # semantic=None
    await mem.write_episode("x", scope=Scope.USER, key="u1")  # 不抛
    assert await mem.recall("x", scope=Scope.USER, key="u1") == []


async def test_semantic_api_requires_explicit_scope() -> None:
    mem = MemoryClient(semantic=FakeSemanticStore())
    with pytest.raises(ValueError):
        await mem.write_episode("x", scope=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await mem.recall("x", scope=None)  # type: ignore[arg-type]


# ------------------ 深度集成接线：recall → recalled_context → 语言 ------------------ #


async def test_memory_recall_populates_context_from_semantic() -> None:
    mem = MemoryClient(semantic=FakeSemanticStore())
    await mem.write_episode("用户对刺激「loss」表现出 sad 情绪", scope=Scope.USER, key="u1")
    state = AffectState(recall_enabled=True, user_id="u1", stimulus=Stimulus(name="loss"))
    out = await MemoryRecallAgent(mem)(state)
    assert out["recalled_context"] == ["用户对刺激「loss」表现出 sad 情绪"]


async def test_memory_recall_no_context_without_semantic() -> None:
    """无语义后端 → recalled_context 不出现（零回归，退化为仅确定性 disposition）。"""
    state = AffectState(recall_enabled=True, user_id="u1", stimulus=Stimulus(name="loss"))
    out = await MemoryRecallAgent(MemoryClient())(state)
    assert "recalled_context" not in out


async def test_language_retrieved_includes_recalled_context() -> None:
    """语义召回事实进入 LanguageAgent 检索串 → 影响生成（深度集成落点）。"""
    state = AffectState(
        language_enabled=True,
        affect_sample=(0.5, 0.3),
        recalled_context=["用户喜欢猫"],
        stimulus=Stimulus(name="topic"),
    )
    out = await LanguageAgent()(state)
    assert "用户喜欢猫" in out["language_text"]


async def test_language_unaffected_without_context() -> None:
    """无召回上下文时语言生成与现状一致（不含 recall 片段）。"""
    state = AffectState(
        language_enabled=True, affect_sample=(0.5, 0.3), stimulus=Stimulus(name="topic")
    )
    out = await LanguageAgent()(state)
    assert "recall:" not in out["language_text"]


# --------------------- 实机 smoke（importorskip + env 探测 → skip） --------------------- #


def _live_graphiti() -> gs.GraphitiGraphStore:
    """连真实 Graphiti（Neo4j + LLM）；缺驱动/实例/LLM 配置时 skip。"""
    import os

    pytest.importorskip("graphiti_core")
    if not (os.getenv("ZERO_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")):
        pytest.skip("无 OpenAI 兼容 LLM 配置，跳过 Graphiti 实机用例")
    uri = os.getenv("ZERO_NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("ZERO_NEO4J_USER", "neo4j")
    password = os.getenv("ZERO_NEO4J_PASSWORD", "password")
    try:
        store = gs.GraphitiGraphStore(uri, user, password)
    except Exception:
        pytest.skip("无可用 Graphiti/Neo4j 实例，跳过")
    return store


async def test_graphiti_roundtrip_smoke() -> None:
    """add_episode → search 往返；抽取/失效为 LLM/矛盾驱动，故只验返回列表、不强断言内容。"""
    store = _live_graphiti()
    try:
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        await store.add_episode(scope="user", key="u1", content="Alice loves cats.", valid_at=t0)
        facts = await store.search(
            "What does Alice love?", scope="user", key="u1", at=t0 + timedelta(hours=1)
        )
        assert isinstance(facts, list)
    finally:
        await store.close()
