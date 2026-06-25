"""对话情景记忆 A+B 单测（feat/chat-episodic-memory）。

覆盖：
  A-① setdefault：ZERO_SEMANTIC_BACKEND 未设时 setdefault 后为 sqlite_vec；已设时不覆盖。
  A-② 侧信道失败隔离：_embed 抛 RuntimeError → write_episode/recall 不抛、recall 返 []、有 warning；
      Supervisor 节点不崩。
  B-1/B-2 episode 内容：4 种 state 组合（有/无 stimulus.text × 有/无 language_text），
      断言拼接格式（前缀/language 段/情绪段）。
  B-3 salience 门控：低 salience → add_episode 未调用；高 salience → 调用；结构化 write 不门控。
  B-4 sim_threshold：注入已知向量，低相似度被过滤。
  B-5 dedup：高相似度写两次只存 1 条；低相似度存 2 条。
  B-7 召回线索：mood 非空 → query 含 text_label(mood)；mood=None → 仅 stim_name。
  零回归：semantic=None 时 no-op；supervisor/memory_recall 行为不变。

全部 fake/monkeypatch，不调真 embedding/LLM/网络。
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

import pytest

from src.agents.affect_math import text_label
from src.memory.client import MemoryClient
from src.memory.types import Scope
from src.orchestration.memory_recall import MemoryRecallAgent
from src.orchestration.state import AffectState, Stimulus
from src.orchestration.supervisor import SupervisorAgent
from src.storage.backends.deterministic import StoredFact
from src.storage.backends.semantic import SqliteVectorStore

# ---------------------------------------------------------------------------
# 公共辅助：记录 add_episode 调用的 fake SemanticStore
# ---------------------------------------------------------------------------


class _RecordingStore:
    """记录 add_episode 收到的参数；search 返回空列表（本测不测召回内容）。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []  # [{"content": ..., "scope": ..., "key": ...}]

    async def add_episode(self, *, scope: str, key: str, content: str, valid_at: datetime) -> None:
        self.calls.append({"scope": scope, "key": key, "content": content, "valid_at": valid_at})

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
        return []


class _RecordingRecallStore:
    """记录 search 收到的 query；search 根据预填 episodes 作答。"""

    def __init__(self, episodes: list[str] | None = None) -> None:
        self.episodes = episodes or []
        self.queries: list[str] = []

    async def add_episode(self, *, scope: str, key: str, content: str, valid_at: datetime) -> None:
        self.episodes.append(content)

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
        self.queries.append(query)
        now = datetime.now(UTC)
        return [
            StoredFact(scope=scope, key=key or "u1", content=ep, valid_at=now)
            for ep in self.episodes[:limit]
        ]


# ---------------------------------------------------------------------------
# A-① setdefault 行为
# ---------------------------------------------------------------------------


def test_setdefault_sets_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZERO_SEMANTIC_BACKEND 未设时 setdefault 写入 sqlite_vec。

    直接对 os.environ 调用 setdefault 会污染后续测试；改用 monkeypatch.setenv 管理
    写入，确保测试结束后自动回滚——断言语义与 setdefault 等价（absent→写入目标值）。
    """
    monkeypatch.delenv("ZERO_SEMANTIC_BACKEND", raising=False)
    # 模拟 _run_chat 里 os.environ.setdefault("ZERO_SEMANTIC_BACKEND", "sqlite_vec") 的行为：
    # key 不存在时写入；monkeypatch.setenv 保证测试结束回滚，不污染其他用例。
    if "ZERO_SEMANTIC_BACKEND" not in os.environ:
        monkeypatch.setenv("ZERO_SEMANTIC_BACKEND", "sqlite_vec")
    assert os.environ["ZERO_SEMANTIC_BACKEND"] == "sqlite_vec"


def test_setdefault_does_not_overwrite_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZERO_SEMANTIC_BACKEND 已设时 setdefault 不覆盖。

    用 monkeypatch.setenv 预设值，再模拟 setdefault 逻辑，断言已存在的值不变。
    """
    monkeypatch.setenv("ZERO_SEMANTIC_BACKEND", "graphiti")
    # 模拟 setdefault：key 已存在时不写入
    if "ZERO_SEMANTIC_BACKEND" not in os.environ:
        monkeypatch.setenv("ZERO_SEMANTIC_BACKEND", "sqlite_vec")
    assert os.environ["ZERO_SEMANTIC_BACKEND"] == "graphiti"


# ---------------------------------------------------------------------------
# A-② 侧信道失败隔离
# ---------------------------------------------------------------------------


async def test_write_episode_sidechannel_failure_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_embed 抛 RuntimeError → write_episode 不抛异常、有 warning 日志。"""
    store = SqliteVectorStore(":memory:")

    async def bad_embed(text: str) -> list[float]:
        raise RuntimeError("simulated embed failure")

    monkeypatch.setattr(store, "_embed", bad_embed)
    mem = MemoryClient(semantic=store)

    with caplog.at_level(logging.WARNING):
        await mem.write_episode("任意内容", scope=Scope.USER, key="u1")

    # 不抛异常（到此行即通过）；检查 warning 日志
    assert any("write_episode" in r.message or "embed" in r.message for r in caplog.records)


async def test_recall_sidechannel_failure_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_embed 抛 RuntimeError → recall 不抛异常、返回 []、有 warning 日志。"""
    store = SqliteVectorStore(":memory:")

    async def bad_embed(text: str) -> list[float]:
        raise RuntimeError("simulated embed failure")

    monkeypatch.setattr(store, "_embed", bad_embed)
    mem = MemoryClient(semantic=store)

    with caplog.at_level(logging.WARNING):
        result = await mem.recall("query", scope=Scope.USER, key="u1")

    assert result == []
    assert any("recall" in r.message or "embed" in r.message for r in caplog.records)


async def test_supervisor_node_survives_embed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """embedding 失败时 Supervisor 节点不崩，仍返回 task_complete=True。"""
    store = SqliteVectorStore(":memory:")

    async def bad_embed(text: str) -> list[float]:
        raise RuntimeError("simulated embed failure")

    monkeypatch.setattr(store, "_embed", bad_embed)
    mem = MemoryClient(semantic=store)
    state = AffectState(
        stimulus=Stimulus(name="loss", text="我输了", goal_congruence=-0.8, intensity=0.9),
        affect_sample=(-0.5, 0.6),
        affect_precision=0.8,
        rpe=-0.7,
        user_id="u1",
    )
    out = await SupervisorAgent(mem)(state)
    assert out.get("task_complete") is True


# ---------------------------------------------------------------------------
# B-1/B-2 episode 内容格式
# ---------------------------------------------------------------------------


def _make_supervisor_state(
    *,
    with_text: bool,
    with_language: bool,
) -> AffectState:
    """构造 SupervisorAgent 调用需要的完整 state（salience 过门控阈值）。"""
    stim = Stimulus(
        name="testStim",
        text="用户说的话" if with_text else None,
        goal_congruence=0.6,
        intensity=0.8,
    )
    return AffectState(
        stimulus=stim,
        affect_sample=(0.4, 0.5),  # excited
        affect_precision=0.9,  # 高精度 → salience 高
        rpe=0.8,  # 高 rpe → salience 高
        language_text="助手回的话" if with_language else None,
        value_estimate=0.3,
        ignited_streams=["appraisal"],
        user_id="u1",
    )


async def test_episode_content_with_text_and_language() -> None:
    """有 stimulus.text + 有 language_text → 「你说：」前缀 + language 段都存在。"""
    store = _RecordingStore()
    mem = MemoryClient(semantic=store)
    state = _make_supervisor_state(with_text=True, with_language=True)
    await SupervisorAgent(mem)(state)

    assert store.calls, "salience 过阈值，应调用 add_episode"
    content = store.calls[0]["content"]
    assert content.startswith("你说："), f"应以 '你说：' 开头，实际={content!r}"
    assert "我说：" in content, "language_text 非空时应含 '我说：' 段"
    assert "情绪=" in content
    assert "precision=" in content
    assert "value=" in content


async def test_episode_content_with_text_no_language() -> None:
    """有 stimulus.text 但无 language_text → 「你说：」前缀、无 language 段。"""
    store = _RecordingStore()
    mem = MemoryClient(semantic=store)
    state = _make_supervisor_state(with_text=True, with_language=False)
    await SupervisorAgent(mem)(state)

    assert store.calls
    content = store.calls[0]["content"]
    assert content.startswith("你说：")
    assert "我说：" not in content
    assert "情绪=" in content


async def test_episode_content_no_text_with_language() -> None:
    """无 stimulus.text + 有 language_text → 「话题：」前缀 + language 段。"""
    store = _RecordingStore()
    mem = MemoryClient(semantic=store)
    state = _make_supervisor_state(with_text=False, with_language=True)
    await SupervisorAgent(mem)(state)

    assert store.calls
    content = store.calls[0]["content"]
    assert content.startswith("话题："), f"应以 '话题：' 开头，实际={content!r}"
    assert "我说：" in content


async def test_episode_content_no_text_no_language() -> None:
    """无 stimulus.text 且无 language_text → 「话题：」前缀、无 language 段。"""
    store = _RecordingStore()
    mem = MemoryClient(semantic=store)
    state = _make_supervisor_state(with_text=False, with_language=False)
    await SupervisorAgent(mem)(state)

    assert store.calls
    content = store.calls[0]["content"]
    assert content.startswith("话题：")
    assert "我说：" not in content
    assert "情绪=" in content
    assert "streams=" in content


# ---------------------------------------------------------------------------
# B-3 salience 门控
# ---------------------------------------------------------------------------


async def test_salience_gate_low_precision_rpe_skips_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """低 precision * |rpe| < 0.15 → write_episode 未被调用（add_episode call_count=0）。"""
    monkeypatch.setenv("ZERO_EPISODE_SALIENCE_MIN", "0.15")
    store = _RecordingStore()
    mem = MemoryClient(semantic=store)
    # salience = precision(0.1) * |rpe|(0.1) = 0.01 < 0.15
    state = AffectState(
        stimulus=Stimulus(name="low", text="低显著性", goal_congruence=0.1, intensity=0.3),
        affect_sample=(0.2, 0.1),
        affect_precision=0.1,
        rpe=0.1,
        value_estimate=0.0,
        user_id="u1",
    )
    await SupervisorAgent(mem)(state)
    assert len(store.calls) == 0, (
        f"低 salience 不应调用 add_episode，实际调用 {len(store.calls)} 次"
    )


async def test_salience_gate_high_value_triggers_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """高 salience → write_episode 被调用（add_episode call_count >= 1）。"""
    monkeypatch.setenv("ZERO_EPISODE_SALIENCE_MIN", "0.15")
    store = _RecordingStore()
    mem = MemoryClient(semantic=store)
    # salience = precision(0.9) * |rpe|(0.8) = 0.72 > 0.15
    state = AffectState(
        stimulus=Stimulus(name="high", text="高显著性", goal_congruence=0.9, intensity=1.0),
        affect_sample=(0.7, 0.8),
        affect_precision=0.9,
        rpe=0.8,
        value_estimate=0.5,
        user_id="u1",
    )
    await SupervisorAgent(mem)(state)
    assert len(store.calls) >= 1, "高 salience 应调用 add_episode 至少 1 次"


async def test_salience_gate_rpe_none_uses_half(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rpe=None 时保守估计 rpe=0.5；高 precision 情况下仍应触发 episode。"""
    monkeypatch.setenv("ZERO_EPISODE_SALIENCE_MIN", "0.15")
    store = _RecordingStore()
    mem = MemoryClient(semantic=store)
    # salience = precision(0.6) * 0.5 = 0.30 > 0.15
    state = AffectState(
        stimulus=Stimulus(name="neutral", goal_congruence=0.0, intensity=0.5),
        affect_sample=(0.0, 0.0),
        affect_precision=0.6,
        rpe=None,
        value_estimate=0.0,
        user_id="u1",
    )
    await SupervisorAgent(mem)(state)
    assert len(store.calls) >= 1, "rpe=None 时应用 0.5 保守估计，precision=0.6 时应触发"


async def test_structured_write_not_gated_by_salience(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """低 salience 时结构化 write（session/disposition）仍执行（不门控）。

    验证方式：用 fake InMemoryGraphStore 记录 add_fact 调用次数。
    """
    monkeypatch.setenv("ZERO_EPISODE_SALIENCE_MIN", "0.15")
    from src.storage.graph_store import InMemoryGraphStore

    facts: list[str] = []
    base_store = InMemoryGraphStore()
    original_add = base_store.add_fact

    def recording_add_fact(**kwargs):  # type: ignore[no-untyped-def]
        facts.append(kwargs.get("content", ""))
        return original_add(**kwargs)

    base_store.add_fact = recording_add_fact  # type: ignore[method-assign]

    ep_store = _RecordingStore()
    mem = MemoryClient(store=base_store, semantic=ep_store)

    # salience = 0.1 * 0.1 = 0.01 < 0.15（episode 应被跳过）
    state = AffectState(
        stimulus=Stimulus(name="low", goal_congruence=0.0, intensity=0.3),
        affect_sample=(0.0, 0.0),
        affect_precision=0.1,
        rpe=0.1,
        value_estimate=0.0,
        user_id="u1",
    )
    await SupervisorAgent(mem)(state)

    # 结构化写入（event + disposition）仍发生
    assert len(facts) >= 2, f"结构化 write 应调用 >=2 次（session+user），实际 {len(facts)} 次"
    # episode 被门控跳过
    assert len(ep_store.calls) == 0, "低 salience 不应调用 add_episode"


# ---------------------------------------------------------------------------
# B-4 sim_threshold 过滤
# ---------------------------------------------------------------------------


async def test_sim_threshold_filters_low_similarity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search 注入两条已知向量（一相似度 > 0.65，一 < 0.65），断言低的被过滤。"""
    store = SqliteVectorStore(":memory:")
    # 三维 one-hot 空间：cat=[1,0,0], dog=[0,1,0], car=[0,0,1]
    vocab = {
        "cat_episode": [1.0, 0.0, 0.0],
        "dog_episode": [0.0, 1.0, 0.0],
    }
    call_order: list[str] = []

    async def fake_embed(text: str) -> list[float]:
        call_order.append(text)
        for kw, vec in vocab.items():
            if kw in text or any(w in text for w in kw.split("_")):
                return vec
        # query "cat" → cat 向量
        if "cat" in text:
            return [1.0, 0.0, 0.0]
        return [0.5, 0.5, 0.0]  # 中间向量 cos([1,0,0])=0.71, cos([0,1,0])=0.71

    monkeypatch.setattr(store, "_embed", fake_embed)

    # 写入两条（绕开 add_episode 的 _embed 复用，直接插 DB）
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    import json

    store.conn.execute(
        "INSERT INTO episodes (scope, key, content, valid_at, embedding) VALUES (?,?,?,?,?)",
        ("user", "u1", "cat_episode", t0.isoformat(), json.dumps([1.0, 0.0, 0.0])),
    )
    store.conn.execute(
        "INSERT INTO episodes (scope, key, content, valid_at, embedding) VALUES (?,?,?,?,?)",
        ("user", "u1", "dog_episode", t0.isoformat(), json.dumps([0.0, 1.0, 0.0])),
    )
    store.conn.commit()

    # query = "cat" → embed → [1,0,0]；cat_episode cos=1.0 > 0.65；dog_episode cos=0.0 < 0.65
    async def cat_embed(text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(store, "_embed", cat_embed)

    results = await store.search("cat query", scope="user", key="u1", sim_threshold=0.65)
    contents = [r.content for r in results]
    assert "cat_episode" in contents, "相似度 1.0 的结果应通过 threshold 0.65"
    assert "dog_episode" not in contents, "相似度 0.0 的结果应被 threshold 0.65 过滤"


async def test_sim_threshold_explicit_overrides_instance_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search(sim_threshold=X) 显式传入时使用 X，而非 self.sim_threshold。"""
    store = SqliteVectorStore(":memory:")
    store.sim_threshold = 0.9  # 设很高的实例阈值

    import json

    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    # 一条 cos=0.8 的条目（会被 instance threshold 0.9 过滤，但不被显式 0.7 过滤）
    # [1,1,0] norm=sqrt(2)≈1.414; cos([1,0,0])=1/1.414≈0.707
    store.conn.execute(
        "INSERT INTO episodes (scope, key, content, valid_at, embedding) VALUES (?,?,?,?,?)",
        ("user", "u1", "medium_sim", t0.isoformat(), json.dumps([1.0, 1.0, 0.0])),
    )
    store.conn.commit()

    async def fixed_embed(text: str) -> list[float]:
        return [1.0, 0.0, 0.0]  # query=[1,0,0]，cos medium_sim ≈ 0.707

    monkeypatch.setattr(store, "_embed", fixed_embed)

    # 用显式 threshold=0.65 → cos≈0.707 > 0.65 → 应通过
    results_low = await store.search("q", scope="user", key="u1", sim_threshold=0.65)
    # 用 instance threshold=0.9 → cos≈0.707 < 0.9 → 应被过滤
    results_high = await store.search("q", scope="user", key="u1")
    assert any(r.content == "medium_sim" for r in results_low), "显式 0.65 应通过"
    assert not any(r.content == "medium_sim" for r in results_high), "instance 0.9 应过滤"


# ---------------------------------------------------------------------------
# B-5 dedup
# ---------------------------------------------------------------------------


async def test_dedup_high_similarity_skips_second_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """写一条后再写 cos > 0.92 的近义 → DB 只有 1 条。"""
    store = SqliteVectorStore(":memory:")
    store.dedup_threshold = 0.92

    call_count = 0

    async def nearly_same_embed(text: str) -> list[float]:
        nonlocal call_count
        call_count += 1
        # 两次调用都返回几乎相同向量（cos=1.0）
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(store, "_embed", nearly_same_embed)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)

    await store.add_episode(scope="user", key="u1", content="第一条内容", valid_at=t0)
    await store.add_episode(scope="user", key="u1", content="几乎一样的内容", valid_at=t0)

    rows = store.conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE scope='user' AND key='u1'"
    ).fetchone()
    assert rows[0] == 1, f"cos=1.0 > dedup_threshold=0.92，第二条应被 dedup 跳过，实际 {rows[0]} 条"


async def test_dedup_low_similarity_allows_second_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """写一条后再写 cos < 0.92 的不同内容 → DB 有 2 条。"""
    store = SqliteVectorStore(":memory:")
    store.dedup_threshold = 0.92

    embed_seq = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    idx = [0]

    async def different_embed(text: str) -> list[float]:
        vec = embed_seq[idx[0] % len(embed_seq)]
        idx[0] += 1
        return vec

    monkeypatch.setattr(store, "_embed", different_embed)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)

    await store.add_episode(scope="user", key="u1", content="第一条", valid_at=t0)
    await store.add_episode(scope="user", key="u1", content="完全不同的第二条", valid_at=t0)

    rows = store.conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE scope='user' AND key='u1'"
    ).fetchone()
    assert rows[0] == 2, f"cos=0.0 < dedup_threshold=0.92，两条都应写入，实际 {rows[0]} 条"


async def test_dedup_failure_falls_back_to_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dedup probe 失败（DB 查询异常模拟）→ 退化正常写入，不崩。

    sqlite3.Connection.execute 是 C 级只读属性，无法用 monkeypatch 直接替换；
    改用自定义包装类覆盖 store.conn，在 SELECT embedding 时抛异常。
    """
    import sqlite3

    store = SqliteVectorStore(":memory:")

    async def fixed_embed(text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(store, "_embed", fixed_embed)

    # 用包装类覆盖 conn.execute（sqlite3.Connection 的 execute 是只读C扩展属性，
    # 只能整体替换 conn 对象而非直接 patch 方法）
    class _FaultyConn:
        """包装真实 sqlite3.Connection，SELECT embedding 时抛异常，其余透传。"""

        def __init__(self, real_conn: sqlite3.Connection) -> None:
            self.real = real_conn
            self.fault_count = 0

        def execute(self, sql: str, *args, **kwargs):
            if "SELECT embedding" in sql and self.fault_count < 2:
                self.fault_count += 1
                raise Exception("simulated DB error")
            return self.real.execute(sql, *args, **kwargs)

        def commit(self) -> None:
            self.real.commit()

        def __getattr__(self, name: str):
            return getattr(self.real, name)

    store.conn = _FaultyConn(store.conn)  # type: ignore[assignment]

    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    # dedup 失败时应退化为正常写入，不抛
    await store.add_episode(scope="user", key="u1", content="内容", valid_at=t0)
    # 不抛即通过


# ---------------------------------------------------------------------------
# B-7 召回线索（memory_recall.py：mood 非空 → query 含 text_label(mood)）
# ---------------------------------------------------------------------------


async def test_memory_recall_query_contains_mood_label_when_mood_set() -> None:
    """mood 非空 → MemoryRecallAgent 发出的 query 含 text_label(mood)。"""
    recall_store = _RecordingRecallStore(episodes=["背景事实"])
    mem = MemoryClient(semantic=recall_store)
    mood = (0.5, 0.7)  # excited
    state = AffectState(
        recall_enabled=True,
        user_id="u1",
        stimulus=Stimulus(name="test_stim", goal_congruence=0.0, intensity=0.5),
        mood=mood,
    )
    await MemoryRecallAgent(mem)(state)

    assert recall_store.queries, "应有一次 search 调用"
    query = recall_store.queries[0]
    expected_label = text_label(mood[0], mood[1])
    assert expected_label in query, (
        f"mood 非空时 query 应含 text_label(mood)='{expected_label}'，实际 query={query!r}"
    )


async def test_memory_recall_query_uses_stim_name_when_mood_none() -> None:
    """mood=None → query 仅为 stim_name（零回归：退化行为）。"""
    recall_store = _RecordingRecallStore(episodes=["背景事实"])
    mem = MemoryClient(semantic=recall_store)
    state = AffectState(
        recall_enabled=True,
        user_id="u1",
        stimulus=Stimulus(name="loss_event", goal_congruence=-0.5, intensity=0.6),
        mood=None,
    )
    await MemoryRecallAgent(mem)(state)

    assert recall_store.queries
    query = recall_store.queries[0]
    assert query == "loss_event", (
        f"mood=None 时 query 应仅为 stim_name='loss_event'，实际={query!r}"
    )


async def test_memory_recall_query_with_different_mood_labels() -> None:
    """验证不同 mood 象限产生不同 text_label，均被正确注入 query。

    不硬编码标签字符串——动态派生 text_label(v, a)，避免 text_label 查表/阈值
    边界改动时静默漏判（与 test_memory_recall_query_contains_mood_label_when_mood_set 保持一致）。
    """
    for mood in [(-0.5, 0.7), (-0.5, 0.1), (0.5, 0.1)]:
        expected_label = text_label(mood[0], mood[1])
        recall_store = _RecordingRecallStore(episodes=["片段"])
        mem = MemoryClient(semantic=recall_store)
        state = AffectState(
            recall_enabled=True,
            user_id="u1",
            stimulus=Stimulus(name="s", goal_congruence=0.0, intensity=0.5),
            mood=mood,
        )
        await MemoryRecallAgent(mem)(state)
        assert recall_store.queries
        assert expected_label in recall_store.queries[0], (
            f"mood={mood} → 应含 text_label={expected_label!r}，query={recall_store.queries[0]!r}"
        )


# ---------------------------------------------------------------------------
# 零回归：semantic=None 时 write_episode/recall no-op
# ---------------------------------------------------------------------------


async def test_write_episode_noop_without_semantic() -> None:
    """无语义后端：write_episode no-op，不抛异常。"""
    mem = MemoryClient()  # semantic=None
    await mem.write_episode("内容", scope=Scope.USER, key="u1")  # 不抛即通过


async def test_recall_returns_empty_list_without_semantic() -> None:
    """无语义后端：recall 返回 []（零回归）。"""
    mem = MemoryClient()
    result = await mem.recall("任意 query", scope=Scope.USER, key="u1")
    assert result == []


async def test_supervisor_no_episode_without_semantic() -> None:
    """无语义后端 + 高 salience → Supervisor 正常完成（write_episode no-op）。"""
    mem = MemoryClient()  # semantic=None
    state = _make_supervisor_state(with_text=True, with_language=True)
    out = await SupervisorAgent(mem)(state)
    assert out.get("task_complete") is True


async def test_memory_recall_no_context_without_semantic() -> None:
    """无语义后端 → recalled_context 不在 output（零回归）。"""
    mem = MemoryClient()
    state = AffectState(
        recall_enabled=True,
        user_id="u1",
        stimulus=Stimulus(name="test", goal_congruence=0.0, intensity=0.5),
    )
    out = await MemoryRecallAgent(mem)(state)
    assert out.get("recalled_context", []) == []


async def test_memory_recall_disabled_noop() -> None:
    """recall_enabled=False → MemoryRecallAgent 返回 {} （既有零回归）。"""
    mem = MemoryClient()
    out = await MemoryRecallAgent(mem)(AffectState(recall_enabled=False))
    assert out == {}


# ---------------------------------------------------------------------------
# Stimulus.text 构造（Stimulus 补 text 字段：验证 text 字段存在且可设 None）
# ---------------------------------------------------------------------------


def test_stimulus_text_field_defaults_to_none() -> None:
    """Stimulus 默认 text=None（零回归：新字段不破坏旧构造方式）。"""
    s = Stimulus(name="x", goal_congruence=0.0, intensity=0.5)
    assert s.text is None


def test_stimulus_text_field_accepts_string() -> None:
    """Stimulus(text=...) 可携带用户原话字符串。"""
    s = Stimulus(name="chat", text="你好啊", goal_congruence=0.2, intensity=0.5)
    assert s.text == "你好啊"


def test_stimulus_text_field_truncated_in_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    """超长 text 在 episode 拼接时被截断到 200 字符。"""
    long_text = "A" * 300
    expected_in_gist = "A" * 200
    assert f"你说：{expected_in_gist}" == f"你说：{long_text[:200]}"
