"""D3+D5：召回三维重排 + first_contact 单测（PR-3）。

覆盖：
  - parse_importance（现居 src/memory/utils）：解析 precision=、缺失/畸形回退 0.5。
  - _rank_episodes：recency / sim / importance 三维各自单维排序正确（关其余权重）；
    first_contact ×1.2；空列表返回空；Δt=0 clamp 不溢出；arousal 调制默认关。
  - 召回侧 salience 衰减（recency = I^κ·Δt^(-d)·`ZERO_RECALL_SALIENCE_DECAY` 默认关）：
    门关逐字等价 / 门开高显著旧 episode 反超 / κ=0 精确退化 / ACT-R 优先 / recency 不破 (0,1]。
    五条均经变异验证可转红（去掉 I^κ、优先级写反、漏 Hill 归一）。
  - MemoryRecallAgent 输出 recalled_facts（已重排的 Fact 列表）。
  - SupervisorAgent first_contact：同一 user 仅首条 episode 打标。
  - AffectState.recalled_facts 默认空（零回归）。

纯数值/fake store，不调真 embedding/LLM。
旋钮参数迁构造期后，_rank_episodes 测试改为直接传参（无 monkeypatch）。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.memory.client import MemoryClient
from src.memory.types import Fact, Scope
from src.memory.utils import parse_importance
from src.orchestration.memory_recall import (
    MemoryRecallAgent,
    _rank_episodes,
    normalized_importance,
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
# _rank_episodes 三维（直接传参，不依赖 monkeypatch env）
# --------------------------------------------------------------------------- #


def test_rank_empty_returns_empty() -> None:
    assert _rank_episodes([], NOW) == []


def test_rank_recency_orders_recent_first() -> None:
    """alpha=1,beta=gamma=0：近的（Δt 小）排前。"""
    old = _fact("precision=0.50", days_ago=10)
    recent = _fact("precision=0.50", days_ago=0)
    ranked = _rank_episodes([old, recent], NOW, alpha=1.0, beta=0.0, gamma=0.0)
    assert ranked[0] is recent, "近因应排前"


def test_rank_sim_orders_higher_first() -> None:
    """beta=1,alpha=gamma=0：sim 高的排前。"""
    low = _fact("precision=0.50", sim=0.2, days_ago=1)
    high = _fact("precision=0.50", sim=0.9, days_ago=1)
    ranked = _rank_episodes([low, high], NOW, alpha=0.0, beta=1.0, gamma=0.0)
    assert ranked[0] is high, "高相关度应排前"


def test_rank_importance_orders_higher_first() -> None:
    """gamma=1,alpha=beta=0：importance（写入 precision）高的排前。"""
    low = _fact("precision=0.10", days_ago=1)
    high = _fact("precision=0.90", days_ago=1)
    ranked = _rank_episodes([low, high], NOW, alpha=0.0, beta=0.0, gamma=1.0)
    assert ranked[0] is high, "高显著度应排前"


def test_rank_first_contact_boost() -> None:
    """gamma=1：first_contact 命中 importance ×1.2，可压过同等 precision 的非首因。"""
    plain = _fact("precision=0.50", days_ago=1)
    first = _fact("precision=0.50 | first_contact=True", days_ago=1)
    ranked = _rank_episodes([plain, first], NOW, alpha=0.0, beta=0.0, gamma=1.0)
    assert ranked[0] is first, "first_contact ×1.2 应排前"


def test_rank_delta_t_clamp_no_overflow() -> None:
    """valid_at=now（Δt=0）→ clamp 到 1 天，不抛 ZeroDivision/Overflow，分数有限。"""
    fresh = _fact("precision=0.50", days_ago=0)
    ranked = _rank_episodes([fresh], NOW)
    assert ranked == [fresh]


def test_rank_arousal_mod_off_by_default() -> None:
    """默认 arousal_mod=False：传 arousal 不改变 gamma（重排与 arousal=0 一致）。"""
    facts = [
        _fact("precision=0.90", sim=0.1, days_ago=20),
        _fact("precision=0.10", sim=0.9, days_ago=0),
    ]
    assert [f.content for f in _rank_episodes(facts, NOW, arousal=1.0)] == [
        f.content for f in _rank_episodes(facts, NOW, arousal=0.0)
    ]


def test_rank_arousal_mod_on_boosts_importance() -> None:
    """arousal_mod=True + 高 arousal：放大 importance 维，可使高显著旧 episode 反超。"""
    # 旧但高显著（Δt=100 天 → recency≈0.1，precision=72 → 归一 0.71）vs 新但平淡
    # （Δt clamp 1 → recency=1，precision=1 → 归一 0.03）。用真实量级 precision（归一后才成立）。
    important_old = _fact("precision=72.00", days_ago=100)
    fresh_dull = _fact("precision=1.00", days_ago=0)
    off = _rank_episodes(
        [important_old, fresh_dull],
        NOW,
        arousal=0.0,
        alpha=0.35,
        beta=0.0,
        gamma=0.40,
        arousal_mod=False,
        importance_scale=30.0,
    )
    on = _rank_episodes(
        [important_old, fresh_dull],
        NOW,
        arousal=1.0,
        alpha=0.35,
        beta=0.0,
        gamma=0.40,
        arousal_mod=True,
        importance_scale=30.0,
    )
    assert off[0] is fresh_dull, "低唤醒下近因占优"
    assert on[0] is important_old, "高唤醒放大 importance，旧但显著者反超"


# --------------------------------------------------------------------------- #
# 召回侧 salience 衰减（recency = I^κ·Δt^(-d)）：默认关，开启后高显著者更耐遗忘
# --------------------------------------------------------------------------- #

# 共用样本：旧但显著 vs 新但平淡。alpha=1 关掉 sim/importance 两维，
# 使断言只咬 recency 维本身（否则 γ 维的 importance 会混进来，测不出衰减调制）。
_SALIENCE_OLD = _fact("precision=72.00", days_ago=30)
_SALIENCE_FRESH = _fact("precision=1.00", days_ago=1)
_RECENCY_ONLY = {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "importance_scale": 30.0}


def test_rank_salience_decay_off_is_byte_identical() -> None:
    """门关（默认）→ 排序与不传该参数逐字一致（零回归断言）。"""
    facts = [
        _fact("precision=72.00", sim=0.3, days_ago=30),
        _fact("precision=1.00", sim=0.7, days_ago=1),
        _fact("precision=40.00 | first_contact=True", sim=0.5, days_ago=10),
    ]
    baseline = [f.content for f in _rank_episodes(facts, NOW)]
    gated_off = [f.content for f in _rank_episodes(facts, NOW, salience_decay_enabled=False)]
    assert gated_off == baseline


def test_rank_salience_decay_on_lets_salient_old_win() -> None:
    """门开 → 高显著旧 episode 的 recency 反超低显著新 episode（衰减被 I^κ 调制）。"""
    pair = [_SALIENCE_OLD, _SALIENCE_FRESH]
    off = _rank_episodes(pair, NOW, salience_decay_enabled=False, **_RECENCY_ONLY)
    on = _rank_episodes(pair, NOW, salience_decay_enabled=True, salience_kappa=3.0, **_RECENCY_ONLY)
    assert off[0] is _SALIENCE_FRESH, "门关时纯幂律：新的必胜"
    assert on[0] is _SALIENCE_OLD, "门开时高显著者衰减更慢，反超"


def test_rank_salience_decay_kappa_zero_degenerates() -> None:
    """变异测试靶子：κ=0 → I^0=1，必须精确退化回原幂律排序。

    若本断言在 κ=0 下仍能区分两者，说明衰减调制混入了 κ 之外的通路（实现跑偏）。
    与 test_rank_salience_decay_on_lets_salient_old_win 成对：一个证明门开有效果，
    一个证明该效果**完全**由 κ 承载——绿灯因此可证伪（green-light-must-prove-it-can-go-red）。
    """
    pair = [_SALIENCE_OLD, _SALIENCE_FRESH]
    baseline = [f.content for f in _rank_episodes(pair, NOW, **_RECENCY_ONLY)]
    kappa_zero = [
        f.content
        for f in _rank_episodes(
            pair, NOW, salience_decay_enabled=True, salience_kappa=0.0, **_RECENCY_ONLY
        )
    ]
    assert kappa_zero == baseline


def test_rank_salience_decay_yields_to_actr() -> None:
    """两门同开 → ACT-R 优先：常访问但平淡者靠 Petrov B 压过高显著者。

    visited（precision=1，κ=3 下 salience recency≈3e-5）若被 salience 衰减接管必输给
    plain（precision=72 → 0.35）；实测排前 ⇒ 走的是 ACT-R（≈0.72）。互斥优先级写反即红。
    """
    visited = Fact(
        content="precision=1.00",
        scope=Scope.USER,
        valid_at=NOW - timedelta(days=1),
        key="u1",
        episode_id="1",
        access_count=9,
    )
    plain = _fact("precision=72.00", days_ago=1)  # 无 episode_id → 只能走 salience 衰减
    ranked = _rank_episodes(
        [plain, visited],
        NOW,
        salience_decay_enabled=True,
        salience_kappa=3.0,
        actr_enabled=True,
        **_RECENCY_ONLY,
    )
    assert ranked[0] is visited, "ACT-R 应优先于 salience 衰减接管 recency 维"


def test_rank_salience_decay_stays_in_unit_range() -> None:
    """量纲守恒：I∈(0,1) × Δt^(-d)∈(0,1] ⇒ recency 仍 ≤1，不破坏三维等权。

    靶子：sim=1.0 的极旧无关 episode（recency≈0.007）总分≈1.007，必须压过
    recency 最大化的那条（precision=72 + Δt clamp=1 → 0.84）。若实现漏了 Hill 归一、
    让原始 precision 进 I^κ（72^0.5=8.5），recency 破 1、顺序翻转 → 本测试转红。
    """
    max_recency = _fact("precision=72.00", days_ago=0)  # recency 取到上界
    sim_anchor = _fact("precision=0.50", sim=1.0, days_ago=365)  # recency≈0，靠 sim 拿≈1.0
    ranked = _rank_episodes(
        [max_recency, sim_anchor],
        NOW,
        salience_decay_enabled=True,
        salience_kappa=0.5,
        alpha=1.0,
        beta=1.0,
        gamma=0.0,
        importance_scale=30.0,
    )
    assert ranked[0] is sim_anchor, "recency 越界（>1）才会让 max_recency 反超"


# --------------------------------------------------------------------------- #
# D8：importance 归一化（Hill 饱和，修 dogfood 暴露的量纲碾压）
# --------------------------------------------------------------------------- #


def test_normalized_importance_bounded_and_monotonic() -> None:
    """Hill 饱和 p/(p+C)：有界 (0,1) 且单调；C=30 时 5→0.143, 72→0.706。"""
    lo = normalized_importance("precision=5.00", scale=30.0)
    hi = normalized_importance("precision=72.00", scale=30.0)
    assert 0.0 < lo < hi < 1.0
    assert abs(lo - 5 / 35) < 1e-9
    assert abs(hi - 72 / 102) < 1e-9


def test_rank_normalization_lets_recency_sim_matter() -> None:
    """归一后 importance 不再碾压：近且相关的中精度 episode 胜过旧而无关的高精度（默认等权）。

    未归一时 importance=72 的 γ·importance≈24 会独占评分（dogfood 暴露的失真）；归一后
    三维同量纲，recency+sim 能压过单纯高 precision。
    """
    old_high_prec = _fact("precision=72.00", sim=0.1, days_ago=100)  # 旧·无关·高精度
    recent_relevant = _fact("precision=40.00", sim=0.9, days_ago=0)  # 新·相关·中精度
    ranked = _rank_episodes(
        [old_high_prec, recent_relevant],
        NOW,
        alpha=0.33,
        beta=0.34,
        gamma=0.33,
        importance_scale=30.0,
    )
    assert ranked[0] is recent_relevant, (
        "归一后 recency+sim 能压过单纯高 precision（修 domination）"
    )


# --------------------------------------------------------------------------- #
# MemoryRecallAgent 输出 recalled_facts
# --------------------------------------------------------------------------- #


class _RankRecallStore:
    """search 返回两条 StoredFact（一新高显著、一旧低显著），验证重排后顺序。"""

    async def add_episode(
        self,
        *,
        scope: str,
        key: str,
        content: str,
        valid_at: datetime,
        embed_text: str | None = None,
    ) -> None:
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

    async def add_episode(
        self,
        *,
        scope: str,
        key: str,
        content: str,
        valid_at: datetime,
        embed_text: str | None = None,
    ) -> None:
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
