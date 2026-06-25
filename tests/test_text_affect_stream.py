"""文本情感流（text_affect stream）专项测试——议会约束落地验证。

分组：
A. 常量约束：TEXT_AFFECT_PRECISION < SURVIVAL_PRECISION 且 >= MIN_PRECISION。
B. 默认关零回归：workspace_enabled=True，text_affect=None → 与无文本流逐字一致；
   workspace_enabled=False + text_affect 有值 → 也不产生文本流。
C. 文本流有效接入：workspace_enabled=True, 强正向 text_affect → "text" 出现在
   ignited_streams；post_mu 相对无文本流向正向拉拢；极弱信号验证显著度门控。
D. BUG 修复回归：fake regressor 返回 valence=-0.9，goal_congruence=0.8 → features[0]
   ==0.8（非 -0.9）；fast_survival_prior 的 mu valence 分量为正（不被 valence 污染）。
E. 端到端快照：固定 rng_seed + fake regressor，perception→affect_core；text_affect 非
   None → affect_sample 在 (-1,1)^2；同参数 text_affect=None 时结果不同（流确实有效）。
F. occ_prior 入口不变：AppraisalAgent 产出无 text_affect 污染、签名不变。
"""

from __future__ import annotations

import pytest

from src.agents.affect_core import AffectCoreAgent
from src.agents.affect_math import (
    MIN_PRECISION,
    SURVIVAL_PRECISION,
    TEXT_AFFECT_PRECISION,
    fast_survival_prior,
)
from src.agents.appraisal import AppraisalAgent
from src.agents.perception import PerceptionAgent
from src.orchestration.state import AffectState, Stimulus

# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


class FakeTextRegressor:
    """鸭子类型 fake：predict_affect 返回固定 (valence, arousal)。"""

    def __init__(self, valence: float = 0.3, arousal: float = -0.1) -> None:
        self.valence = valence
        self.arousal = arousal

    def predict_affect(self, _text: str) -> tuple[float, float]:
        return (self.valence, self.arousal)


def _base_core_state(**overrides: object) -> AffectState:
    """构造 AffectCoreAgent 前置条件齐备的 state。

    默认 workspace_enabled=True，可通过 overrides 覆盖任意字段（包括 workspace_enabled=False）。
    """
    defaults: dict[str, object] = {
        "stimulus": Stimulus(name="t", goal_congruence=0.5, intensity=0.8),
        "features": [0.5, 0.0, 0.0, 0.8],
        "prior_mu": (0.3, 0.5),
        "prior_sigma": (0.2, 0.2),
        "reward": 0.5,
        "rpe": 0.2,
        "precision": 0.6,
        "rng_seed": 42,
        "workspace_enabled": True,
    }
    defaults.update(overrides)
    return AffectState(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# A. 常量约束
# ---------------------------------------------------------------------------


def test_text_affect_precision_lt_survival_precision() -> None:
    """TEXT_AFFECT_PRECISION 必须严格小于 SURVIVAL_PRECISION（议会约束：文本流精度下限）。"""
    assert TEXT_AFFECT_PRECISION < SURVIVAL_PRECISION, (
        f"应满足 {TEXT_AFFECT_PRECISION} < {SURVIVAL_PRECISION}"
    )


def test_text_affect_precision_ge_min_precision() -> None:
    """TEXT_AFFECT_PRECISION 必须 >= MIN_PRECISION（防止精度退化为零导致数值不稳定）。"""
    assert TEXT_AFFECT_PRECISION >= MIN_PRECISION, (
        f"TEXT_AFFECT_PRECISION={TEXT_AFFECT_PRECISION} 应 >= MIN_PRECISION={MIN_PRECISION}"
    )


# ---------------------------------------------------------------------------
# B. 默认关零回归
# ---------------------------------------------------------------------------


def test_workspace_text_affect_none_zero_regression() -> None:
    """workspace_enabled=True，text_affect=None → post_mu/post_sigma/affect_sample 与
    等价 state（无 text_affect 字段）逐字一致，ignited_streams 不含 "text"。
    """
    state_with_none = _base_core_state(text_affect=None)
    state_ref = _base_core_state()  # text_affect 字段默认 None

    agent = AffectCoreAgent()
    out_none = agent(state_with_none)
    out_ref = agent(state_ref)

    # 逐字一致（同 rng_seed 下完全可复现）
    assert out_none["post_mu"] == out_ref["post_mu"]
    assert out_none["post_sigma"] == out_ref["post_sigma"]
    assert out_none["affect_sample"] == out_ref["affect_sample"]

    # 不含 "text" 流
    assert "text" not in out_none["ignited_streams"]
    assert "text" not in out_ref["ignited_streams"]


def test_workspace_off_text_affect_present_no_text_stream() -> None:
    """workspace_enabled=False 时，即便 text_affect 有值，也不走工作空间分支，
    不产生 ignited_streams 字段，行为与 v1 基线一致（零回归）。
    """
    state = _base_core_state(workspace_enabled=False, text_affect=(0.7, 0.5))
    out = AffectCoreAgent()(state)

    # 非 workspace 分支：不含工作空间键
    assert "ignited_streams" not in out
    assert "affect_precision" not in out
    # 仍有标准输出字段
    assert "post_mu" in out
    assert "affect_sample" in out


def test_workspace_off_zero_regression_same_as_baseline() -> None:
    """workspace_enabled=False + text_affect=None：与纯 v1 gaussian_fuse 路径完全一致。"""
    state_off = _base_core_state(workspace_enabled=False, text_affect=None)
    state_base = _base_core_state(workspace_enabled=False)

    agent = AffectCoreAgent()
    out_off = agent(state_off)
    out_base = agent(state_base)

    assert out_off["post_mu"] == out_base["post_mu"]
    assert out_off["post_sigma"] == out_base["post_sigma"]
    assert out_off["affect_sample"] == out_base["affect_sample"]
    assert set(out_off) == set(out_base)


# ---------------------------------------------------------------------------
# C. 文本流有效接入
# ---------------------------------------------------------------------------


def test_strong_positive_text_affect_ignites_text_stream() -> None:
    """workspace_enabled=True, text_affect=(0.9,0.9) 强正向 → "text" 出现在 ignited_streams。"""
    state = _base_core_state(text_affect=(0.9, 0.9))
    out = AffectCoreAgent()(state)

    assert "text" in out["ignited_streams"], (
        f"强正向 text_affect 应点燃文本流，实际 ignited_streams={out['ignited_streams']}"
    )


def test_text_affect_pulls_post_mu_positive() -> None:
    """有强正向 text_affect 时，post_mu valence 应相比无文本流更靠近正向。

    对比：同 state 下 text_affect=(0.9,0.9) vs text_affect=None，前者的 post_mu[0] 更大。
    """
    state_with = _base_core_state(text_affect=(0.9, 0.9))
    state_without = _base_core_state(text_affect=None)

    agent = AffectCoreAgent()
    out_with = agent(state_with)
    out_without = agent(state_without)

    # 强正向文本流应使 post_mu valence 分量更偏正
    assert out_with["post_mu"][0] >= out_without["post_mu"][0], (
        f"有强正向文本流的 post_mu[0]={out_with['post_mu'][0]} "
        f"应 >= 无文本流的 {out_without['post_mu'][0]}"
    )


def test_weak_text_affect_may_not_ignite() -> None:
    """极弱 text_affect=(0.001, 0.001) → salience 极低，验证门控行为。

    文本流 salience = |mu| * precision ≈ sqrt(2)*0.001 * 0.3 ≈ 0.0004 << SALIENCE_THRESHOLD=0.18，
    所以文本流不应出现在 ignited_streams（被门控在局部）。
    """
    state = _base_core_state(text_affect=(0.001, 0.001))
    out = AffectCoreAgent()(state)

    # 极弱信号被门控，不应点燃文本流
    assert "text" not in out["ignited_streams"], (
        f"极弱 text_affect 不应点燃文本流，实际 ignited_streams={out['ignited_streams']}"
    )


# ---------------------------------------------------------------------------
# D. BUG 修复回归：features 不被 regressor valence 污染
# ---------------------------------------------------------------------------


def test_bug_fix_features_not_polluted_by_regressor_valence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fake regressor 返回 valence=-0.9，stim.goal_congruence=0.8。

    修复后：PerceptionAgent 产出 features[0]==0.8（来自 goal_congruence），
    而不是 -0.9（回归器 valence）。
    """
    monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
    monkeypatch.delenv("ZERO_TEXT_AFFECT_MODEL_PATH", raising=False)

    agent = PerceptionAgent()
    # fake regressor 返回截然相反的 valence
    agent.text_regressor = FakeTextRegressor(valence=-0.9, arousal=0.2)  # type: ignore[assignment]

    state = AffectState(
        stimulus=Stimulus(
            name="bug_fix_test",
            text="some text",
            goal_congruence=0.8,
            standard_compliance=0.3,
            attitude_appeal=0.1,
            intensity=0.9,
        )
    )

    out = agent(state)
    features = out["features"]

    # features[0] 应为 goal_congruence=0.8，不是 regressor valence=-0.9
    assert features[0] == pytest.approx(0.8), (
        f"features[0]={features[0]} 应为 goal_congruence=0.8（BUG：被 valence=-0.9 污染）"
    )


def test_bug_fix_survival_prior_mu_valence_positive() -> None:
    """features 是 OCC 布局时，fast_survival_prior 的 valence 分量应为正（goal>0）。

    BUG 修复前：features[0]=valence=-0.9 → survival mu valence 为负（被污染）。
    BUG 修复后：features[0]=goal_congruence=0.8 → survival mu valence 为正（0.6*0.8=0.48）。
    """
    # 构造修复后的 OCC 布局 features（goal_congruence=0.8 在 index 0）
    features_occ_layout = [0.8, 0.3, 0.1, 0.9]
    surv_mu, surv_prec = fast_survival_prior(features_occ_layout)

    # survival mu valence 应为正（来自 goal_congruence=0.8，不被 regressor valence=-0.9 污染）
    assert surv_mu[0] > 0.0, (
        f"fast_survival_prior mu valence={surv_mu[0]} 应 > 0（goal_congruence=0.8 驱动）"
    )
    assert surv_prec == (SURVIVAL_PRECISION, SURVIVAL_PRECISION)


# ---------------------------------------------------------------------------
# E. 端到端快照：perception → affect_core（fake regressor，无需 torch）
# ---------------------------------------------------------------------------


def test_e2e_text_affect_changes_fusion_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """端到端：同 rng_seed，有 text_affect vs 无 text_affect → affect_sample 不同。

    证明文本流真实参与了融合，而非空接（double-count 验证：只通过 text_affect 字段
    注入，不重复写入 features，两次 perception 产出的 features 相同，差异仅来自 text stream）。
    """
    monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
    monkeypatch.delenv("ZERO_TEXT_AFFECT_MODEL_PATH", raising=False)

    # 1. 有文本流：fake regressor 注入强正向 (0.9, 0.9)
    perception_with = PerceptionAgent()
    perception_with.text_regressor = FakeTextRegressor(valence=0.9, arousal=0.9)  # type: ignore[assignment]

    stim = Stimulus(
        name="e2e_test",
        text="happy",
        goal_congruence=0.4,
        standard_compliance=0.1,
        attitude_appeal=0.2,
        intensity=0.7,
    )

    state_with = AffectState(stimulus=stim)
    perc_out_with = perception_with(state_with)

    # 合并 perception 输出到 state（模拟图执行）
    state_after_perc_with = AffectState(
        stimulus=stim,
        features=perc_out_with["features"],
        text_affect=perc_out_with.get("text_affect"),
        prior_mu=(0.3, 0.4),
        prior_sigma=(0.2, 0.2),
        reward=0.4,
        rpe=0.1,
        precision=0.5,
        rng_seed=99,
        workspace_enabled=True,
        trace=perc_out_with.get("trace", []),
    )

    out_with = AffectCoreAgent()(state_after_perc_with)

    # 2. 无文本流：perception 不注入 regressor
    perception_without = PerceptionAgent()
    assert perception_without.text_regressor is None

    state_without = AffectState(stimulus=stim)
    perc_out_without = perception_without(state_without)

    state_after_perc_without = AffectState(
        stimulus=stim,
        features=perc_out_without["features"],
        text_affect=None,  # 明确无文本流
        prior_mu=(0.3, 0.4),
        prior_sigma=(0.2, 0.2),
        reward=0.4,
        rpe=0.1,
        precision=0.5,
        rng_seed=99,
        workspace_enabled=True,
        trace=perc_out_without.get("trace", []),
    )

    out_without = AffectCoreAgent()(state_after_perc_without)

    # affect_sample 在 (-1,1)^2
    sample_with = out_with["affect_sample"]
    assert -1.0 <= sample_with[0] <= 1.0
    assert -1.0 <= sample_with[1] <= 1.0

    # 文本流改变了融合（features 相同，差异来自文本流）
    assert out_with["post_mu"] != out_without["post_mu"], (
        "有文本流与无文本流的 post_mu 应不同（文本流真实参与融合）"
    )

    # 两次的 features 相同（没有 double-count）：文本 (v,a) 只走 text_affect，不进 features
    assert perc_out_with["features"] == perc_out_without["features"], (
        "features 应相同（文本 v/a 不写入 features，无 double-count）"
    )

    # 有文本流时 "text" 应在 ignited_streams（强正向信号）
    assert "text" in out_with["ignited_streams"]


def test_e2e_affect_sample_bounded() -> None:
    """端到端：fake regressor 注入 text_affect，affect_sample 值域在 (-1,1)^2。"""
    stim = Stimulus(
        name="bounded_test",
        text="extreme",
        goal_congruence=1.0,
        standard_compliance=1.0,
        attitude_appeal=1.0,
        intensity=1.0,
    )

    perception = PerceptionAgent()
    perception.text_regressor = FakeTextRegressor(valence=0.9, arousal=0.9)  # type: ignore[assignment]

    state_in = AffectState(stimulus=stim)
    perc_out = perception(state_in)

    state_core = AffectState(
        stimulus=stim,
        features=perc_out["features"],
        text_affect=perc_out.get("text_affect"),
        prior_mu=(0.8, 0.8),
        prior_sigma=(0.1, 0.1),
        reward=1.0,
        rpe=0.5,
        precision=1.0,
        rng_seed=7,
        workspace_enabled=True,
    )

    out = AffectCoreAgent()(state_core)
    sample = out["affect_sample"]

    assert sample is not None
    assert -1.0 <= sample[0] <= 1.0
    assert -1.0 <= sample[1] <= 1.0


# ---------------------------------------------------------------------------
# F. occ_prior 入口不变：AppraisalAgent 产出无 text_affect 污染
# ---------------------------------------------------------------------------


def test_appraisal_agent_output_no_text_affect_field() -> None:
    """AppraisalAgent 产出字段不含 text_affect 相关内容，签名不变。"""
    state = AffectState(
        stimulus=Stimulus(
            name="appraisal_test",
            goal_congruence=0.6,
            standard_compliance=0.3,
            attitude_appeal=0.2,
            intensity=0.8,
        ),
        text_affect=(0.5, 0.5),  # 即便 state 有 text_affect，AppraisalAgent 也不应读它
    )

    out = AppraisalAgent()(state)

    # 标准产出字段
    assert "prior_mu" in out
    assert "prior_sigma" in out
    assert "reward" in out
    assert "appraisal" in out

    # 无 text_affect 相关污染
    assert "text_affect" not in out
    assert "text" not in str(out.get("appraisal", {}))


def test_appraisal_agent_ignores_text_affect_in_state() -> None:
    """AppraisalAgent 产出的先验只取决于 OCC 维度，text_affect 不影响其结果。"""
    stim = Stimulus(
        name="appraisal_isolation",
        goal_congruence=0.5,
        standard_compliance=0.2,
        attitude_appeal=0.3,
        intensity=0.7,
    )

    state_no_text = AffectState(stimulus=stim, text_affect=None)
    state_with_text = AffectState(stimulus=stim, text_affect=(0.9, 0.9))

    agent = AppraisalAgent()
    out_no_text = agent(state_no_text)
    out_with_text = agent(state_with_text)

    # 产出完全相同，text_affect 对 appraisal 无影响
    assert out_no_text["prior_mu"] == out_with_text["prior_mu"]
    assert out_no_text["prior_sigma"] == out_with_text["prior_sigma"]
    assert out_no_text["reward"] == out_with_text["reward"]
