"""D3+D5：召回三维重排 + first_contact 单测（PR-3）。

覆盖：
  - _parse_importance：解析 precision=、缺失/畸形回退 0.5。
  - _rank_episodes：recency / sim / importance 三维各自单维排序正确（关其余权重）；
    first_contact ×1.2；空列表返回空；Δt=0 clamp 不溢出；arousal 调制默认关。
  - MemoryRecallAgent 输出 recalled_facts（已重排的 Fact 列表）。
  - SupervisorAgent first_contact：同一 user 仅首条 episode 打标。
  - AffectState.recalled_facts 默认空（零回归）。

纯数值/fake store，不调真 embedding/LLM。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.memory.client import MemoryClient
from src.memory.types import Fact, Scope
from src.orchestration.memory_recall import (
    MemoryRecallAgent,
    _rank_episodes,
    parse_importance,
)
from src.orchestration.state import AffectState, Stimulus
from src.orchestration.supervisor import SupervisorAgent
from src.storage.backends.deterministic import StoredFact

NOW = datetime(2026, 6, 1, tzinfo=UTC)


def _fact(content: str, *, sim: float = 0.0, days_ago: float = 0.0) -> Fact:
    return Fact(
        content=content,
        scope=Scope.USER,
        valid_at=NOW - timedelta(days=days_ago),
        key="u1",
        sim=sim,
    )


# --------------------------------------------------------------------------- #
# _parse_importance
# --------------------------------------------------------------------------- #


def test_parse_importance_extracts_precision() -> None:
    assert abs(parse_importance("gist | precision=0.90 | streams=[]") - 0.90) < 1e-9


def test_parse_importance_missing_defaults_half() -> None:
    assert parse_importance("没有精度字段的旧 episode") == 0.5


def test_parse_importance_malformed_defaults_half() -> None:
    assert parse_importance("precision=abc") == 0.5


def test_parse_importance_ignores_user_text_injection() -> None:
    """用户原话含 precision=0.99 时仍取尾部真实元数据字段（取最后匹配，WARN-1）。"""
    content = (
        "你说：我的 precision=0.99 很高 | 情绪=平静(0.1,0.1)"
        " | precision=0.50 | streams=[] | value=0.30"
    )
    assert abs(parse_importance(content) - 0.50) < 1e-9


# --------------------------------------------------------------------------- #
# _rank_episodes 三维
# --------------------------------------------------------------------------- #


def test_rank_empty_returns_empty() -> None:
    assert _rank_episodes([], NOW) == []


def test_rank_recency_orders_recent_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """α=1,β=γ=0：近的（Δt 小）排前。"""
    monkeypatch.setenv("ZERO_RECALL_ALPHA", "1.0")
    monkeypatch.setenv("ZERO_RECALL_BETA", "0.0")
    monkeypatch.setenv("ZERO_RECALL_GAMMA", "0.0")
    old = _fact("precision=0.50", days_ago=10)
    recent = _fact("precision=0.50", days_ago=0)
    ranked = _rank_episodes([old, recent], NOW)
    assert ranked[0] is recent, "近因应排前"


def test_rank_sim_orders_higher_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """β=1,α=γ=0：sim 高的排前。"""
    monkeypatch.setenv("ZERO_RECALL_ALPHA", "0.0")
    monkeypatch.setenv("ZERO_RECALL_BETA", "1.0")
    monkeypatch.setenv("ZERO_RECALL_GAMMA", "0.0")
    low = _fact("precision=0.50", sim=0.2, days_ago=1)
    high = _fact("precision=0.50", sim=0.9, days_ago=1)
    ranked = _rank_episodes([low, high], NOW)
    assert ranked[0] is high, "高相关度应排前"


def test_rank_importance_orders_higher_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """γ=1,α=β=0：importance（写入 precision）高的排前。"""
    monkeypatch.setenv("ZERO_RECALL_ALPHA", "0.0")
    monkeypatch.setenv("ZERO_RECALL_BETA", "0.0")
    monkeypatch.setenv("ZERO_RECALL_GAMMA", "1.0")
    low = _fact("precision=0.10", days_ago=1)
    high = _fact("precision=0.90", days_ago=1)
    ranked = _rank_episodes([low, high], NOW)
    assert ranked[0] is high, "高显著度应排前"


def test_rank_first_contact_boost(monkeypatch: pytest.MonkeyPatch) -> None:
    """γ=1：first_contact 命中 importance ×1.2，可压过同等 precision 的非首因。"""
    monkeypatch.setenv("ZERO_RECALL_ALPHA", "0.0")
    monkeypatch.setenv("ZERO_RECALL_BETA", "0.0")
    monkeypatch.setenv("ZERO_RECALL_GAMMA", "1.0")
    plain = _fact("precision=0.50", days_ago=1)
    first = _fact("precision=0.50 | first_contact=True", days_ago=1)
    ranked = _rank_episodes([plain, first], NOW)
    assert ranked[0] is first, "first_contact ×1.2 应排前"


def test_rank_delta_t_clamp_no_overflow() -> None:
    """valid_at=now（Δt=0）→ clamp 到 1 天，不抛 ZeroDivision/Overflow，分数有限。"""
    fresh = _fact("precision=0.50", days_ago=0)
    ranked = _rank_episodes([fresh], NOW)
    assert ranked == [fresh]


def test_rank_arousal_mod_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认 ZERO_RECALL_AROUSAL_MOD 关：传 arousal 不改变 gamma（重排与 arousal=0 一致）。"""
    monkeypatch.delenv("ZERO_RECALL_AROUSAL_MOD", raising=False)
    facts = [
        _fact("precision=0.90", sim=0.1, days_ago=20),
        _fact("precision=0.10", sim=0.9, days_ago=0),
    ]
    assert [f.content for f in _rank_episodes(facts, NOW, arousal=1.0)] == [
        f.content for f in _rank_episodes(facts, NOW, arousal=0.0)
    ]


def test_rank_arousal_mod_on_boosts_importance(monkeypatch: pytest.MonkeyPatch) -> None:
    """开 AROUSAL_MOD + 高 arousal：放大 importance 维，可使高显著旧 episode 反超。"""
    monkeypatch.setenv("ZERO_RECALL_AROUSAL_MOD", "1")
    monkeypatch.setenv("ZERO_RECALL_ALPHA", "0.34")
    monkeypatch.setenv("ZERO_RECALL_BETA", "0.0")
    monkeypatch.setenv("ZERO_RECALL_GAMMA", "0.30")
    # 旧但高显著（Δt=100 天 → recency≈0.1）vs 新但平淡（Δt clamp 1 → recency=1）
    important_old = _fact("precision=0.95", days_ago=100)
    fresh_dull = _fact("precision=0.10", days_ago=0)
    off = _rank_episodes([important_old, fresh_dull], NOW, arousal=0.0)
    on = _rank_episodes([important_old, fresh_dull], NOW, arousal=1.0)
    assert off[0] is fresh_dull, "低唤醒下近因占优"
    assert on[0] is important_old, "高唤醒放大 importance，旧但显著者反超"


# --------------------------------------------------------------------------- #
# MemoryRecallAgent 输出 recalled_facts
# --------------------------------------------------------------------------- #


class _RankRecallStore:
    """search 返回两条 StoredFact（一新高显著、一旧低显著），验证重排后顺序。"""

    async def add_episode(self, *, scope: str, key: str, content: str, valid_at: datetime) -> None:
        return None

    async def search(
        self,
        query: str,
        *,
        scope: str,
        key: str | None = None,
        at: datetime | None = None,
        limit: int = 5,
        sim_threshold: float | None = None,
    ) -> list[StoredFact]:
        now = datetime.now(UTC)
        return [
            StoredFact(
                scope=scope,
                key=key or "u1",
                content="旧 | precision=0.10",
                valid_at=now - timedelta(days=30),
                sim=0.1,
            ),
            StoredFact(
                scope=scope,
                key=key or "u1",
                content="新 | precision=0.90",
                valid_at=now,
                sim=0.9,
            ),
        ]


async def test_memory_recall_outputs_ranked_facts() -> None:
    """recall_enabled=True → out 含 recalled_facts（Fact 列表），高显著新 episode 排前。"""
    mem = MemoryClient(semantic=_RankRecallStore())
    state = AffectState(
        recall_enabled=True,
        user_id="u1",
        stimulus=Stimulus(name="s", goal_congruence=0.0, intensity=0.5),
    )
    out = await MemoryRecallAgent(mem)(state)
    facts = out.get("recalled_facts")
    assert facts is not None and len(facts) == 2
    assert all(isinstance(f, Fact) for f in facts)
    assert "precision=0.90" in facts[0].content, "新且高显著应排第一"
    # recalled_context 与 recalled_facts 顺序一致
    assert out["recalled_context"][0] == facts[0].content


# --------------------------------------------------------------------------- #
# SupervisorAgent first_contact 标签
# --------------------------------------------------------------------------- #


class _EpisodeRecorder:
    def __init__(self) -> None:
        self.contents: list[str] = []

    async def add_episode(self, *, scope: str, key: str, content: str, valid_at: datetime) -> None:
        self.contents.append(content)

    async def search(self, query: str, **kwargs: object) -> list[StoredFact]:
        return []


def _high_salience_state(user_id: str) -> AffectState:
    return AffectState(
        stimulus=Stimulus(name="s", text="某事", goal_congruence=0.6, intensity=0.8),
        affect_sample=(0.4, 0.5),
        affect_precision=0.9,
        rpe=0.8,
        value_estimate=0.3,
        user_id=user_id,
    )


async def test_supervisor_first_contact_tagged_once() -> None:
    """同一 SupervisorAgent 实例、同 user：首条 episode 含 first_contact=True，第二条不含。"""
    store = _EpisodeRecorder()
    sup = SupervisorAgent(MemoryClient(semantic=store))
    await sup(_high_salience_state("u1"))
    await sup(_high_salience_state("u1"))
    assert len(store.contents) == 2
    assert "first_contact=True" in store.contents[0], "首条应打 first_contact"
    assert "first_contact=True" not in store.contents[1], "第二条不应再打"


async def test_supervisor_first_contact_per_user() -> None:
    """不同 user 各自有自己的首因标签。"""
    store = _EpisodeRecorder()
    sup = SupervisorAgent(MemoryClient(semantic=store))
    await sup(_high_salience_state("u1"))
    await sup(_high_salience_state("u2"))
    assert all("first_contact=True" in c for c in store.contents), "两个 user 各自首条都应打标"


# --------------------------------------------------------------------------- #
# 零回归
# --------------------------------------------------------------------------- #


def test_affect_state_recalled_facts_default_empty() -> None:
    assert AffectState().recalled_facts == []


async def test_memory_recall_no_facts_without_semantic() -> None:
    """无语义后端 → recalled_facts 不在 output（零回归）。"""
    mem = MemoryClient()
    state = AffectState(
        recall_enabled=True,
        user_id="u1",
        stimulus=Stimulus(name="s", goal_congruence=0.0, intensity=0.5),
    )
    out = await MemoryRecallAgent(mem)(state)
    assert out.get("recalled_facts", []) == []
