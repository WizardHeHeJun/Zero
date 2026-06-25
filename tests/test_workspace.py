"""v3 显著度门控全局工作空间：并行流竞争 + ignition + 精度加权再入。

纯函数（fast_survival_prior/stream_salience/ignite/precision_reconcile）单测
+ affect_core 工作空间分支 + 端到端管线 + 默认关零回归。torch/API-free。
"""

from __future__ import annotations

from src.agents.affect_core import AffectCoreAgent
from src.agents.affect_math import (
    LANG_BASE_PRECISION,
    SURVIVAL_PRECISION,
    fast_survival_prior,
    ignite,
    precision_reconcile,
    reconcile_affect,
    stream_salience,
)
from src.orchestration.runner import run
from src.orchestration.state import AffectState, Stimulus

# ---------- 纯函数 ----------


def test_fast_survival_prior_coarse_low_precision() -> None:
    # 效价随目标符号；唤醒随强度上升；精度为低固定值（粗快=不确定）
    pos_mu, pos_prec = fast_survival_prior([0.8, 0.0, 0.0, 1.0])
    neg_mu, _ = fast_survival_prior([-0.8, 0.0, 0.0, 1.0])
    low_arousal_mu, _ = fast_survival_prior([0.8, 0.0, 0.0, 0.1])
    assert pos_mu[0] > 0 > neg_mu[0]
    assert pos_mu[1] > low_arousal_mu[1]  # 强度高 → 唤醒高
    assert pos_prec == (SURVIVAL_PRECISION, SURVIVAL_PRECISION)


def test_stream_salience_monotonic() -> None:
    # 偏离中性越远、精度越高 → salience 越大
    base = stream_salience((0.3, 0.3), (1.0, 1.0))
    assert stream_salience((0.6, 0.6), (1.0, 1.0)) > base  # |μ| 更大
    assert stream_salience((0.3, 0.3), (2.0, 2.0)) > base  # Π 更大
    assert stream_salience((0.0, 0.0), (5.0, 5.0)) == 0.0  # 中性零显著


def test_ignite_gates_weak_keeps_strong() -> None:
    streams = [
        ("strong", (0.8, 0.8), (1.5, 1.5)),
        ("weak", (0.02, 0.02), (0.3, 0.3)),
    ]
    terms, names = ignite(streams)
    assert "strong" in names
    assert "weak" not in names  # 弱流被门控
    assert len(terms) == len(names)


def test_ignite_never_empty() -> None:
    # 全弱刺激：无流过阈也要保留最显著者（不空播）
    streams = [
        ("a", (0.01, 0.01), (0.2, 0.2)),
        ("b", (0.03, 0.0), (0.2, 0.2)),
    ]
    _, names = ignite(streams)
    assert names == ["b"]  # 较显著的一条


def test_precision_reconcile_high_kernel_resists() -> None:
    e_star = (0.6, 0.5)
    lang = (-0.6, -0.5)
    # 高内核精度 → 贴近 e*；高语言精度 → 贴近语言
    near_kernel = precision_reconcile(e_star, 10.0, lang, lang_precision=1.0)
    near_lang = precision_reconcile(e_star, 0.1, lang, lang_precision=10.0)
    assert abs(near_kernel[0] - e_star[0]) < abs(near_lang[0] - e_star[0])
    # 等精度 ≡ 固定中点（reconcile_affect weight=0.5）
    mid = precision_reconcile(e_star, LANG_BASE_PRECISION, lang, lang_precision=LANG_BASE_PRECISION)
    ref = reconcile_affect(e_star, lang)
    assert abs(mid[0] - ref[0]) < 1e-9 and abs(mid[1] - ref[1]) < 1e-9


# ---------- affect_core 节点 ----------


def _core_state(**kw: object) -> AffectState:
    return AffectState(
        stimulus=Stimulus(name="t", goal_congruence=0.5, intensity=0.8),
        features=[0.5, 0.0, 0.0, 0.8],
        prior_mu=(0.3, 0.5),
        prior_sigma=(0.2, 0.2),
        reward=0.5,
        rpe=0.2,
        precision=0.6,
        rng_seed=7,
        **kw,
    )


def test_affect_core_workspace_ignites_and_reports_precision() -> None:
    out = AffectCoreAgent()(_core_state(workspace_enabled=True))
    assert out["ignited_streams"]  # 非空：至少一条流点燃
    assert out["affect_precision"] > 0
    assert out["affect_sample"] is not None


def test_affect_core_zero_regression_keys() -> None:
    # 默认关：返回 dict 不含工作空间新键，走原 gaussian_fuse 路径
    out = AffectCoreAgent()(_core_state(workspace_enabled=False))
    assert "ignited_streams" not in out
    assert "affect_precision" not in out
    assert set(out) == {"post_mu", "post_sigma", "affect_sample", "trace"}


# ---------- 端到端管线 ----------


async def test_pipeline_workspace_survival_dominates_on_arousal() -> None:
    # 高唤醒、目标中性（突发巨响）→ 快生存流应点燃
    traj = await run(
        [Stimulus(name="bang", goal_congruence=0.0, intensity=1.0)],
        thread_id="ws-survival",
        workspace_enabled=True,
        rng_seed=7,
    )
    step = traj[0]
    assert "survival" in step["ignited_streams"]
    assert step["affect_precision"] is not None


async def test_pipeline_workspace_off_zero_regression() -> None:
    # 同 rng_seed 下 workspace 关 == 既有默认路径：affect 轨迹一致、无点燃流
    stim = Stimulus(name="evt", goal_congruence=0.6, intensity=0.7)
    on = await run([stim], thread_id="ws-a", workspace_enabled=False, rng_seed=7)
    ref = await run([stim], thread_id="ws-b", rng_seed=7)
    assert on[0]["valence_arousal"] == ref[0]["valence_arousal"]
    assert on[0]["ignited_streams"] == []
