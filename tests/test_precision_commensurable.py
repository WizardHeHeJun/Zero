"""精度量纲齐次化（议会 2026-07-28 第四轮）：零回归 + 效果验收 + 锚点。

背景：`fuse_terms` 的 Π 是**逆方差**（比值尺度），而 survival/mood/text/value 四条流的 Π
是**裸常数/裸 sigmoid**（只验证过序，Stevens 1946 的定序尺度）。把它们反解成 σ 得
1.00~1.83——全部宽于 [-1,1] 值域的半宽，语义上等于「这条流什么都不知道」，与它们实际占的
权重自相矛盾。本模块验证 `precision_commensurable` 门把四条流放到同一把尺子上。

⚠ 只统一量纲，**不等于校准正确**——实证校准（从回归器残差估计 σ）是独立后续项目。

分区：
1. 零回归：门关时逐字旧行为（对照**写死的旧公式**，不是对照自身）。
2. 贯通：env → SessionConfig → AffectState → 节点，防静默失效。
3. fail-fast：混标度（门开 + 旧标度旋钮）在配置期被拒。
4. 效果验收 + 锚点：可比性 / 域穷举 / N_eff / **唤醒否决**（线 A 解冻判据）。
5. 变异测试：每条锚点演示它在什么 bug 下会红，**以及它抓不到什么**。
"""

from __future__ import annotations

import itertools
import math

import pytest

from src.agents.affect_core import AffectCoreAgent
from src.agents.affect_math import (
    AROUSAL_GAIN,
    MIN_PRECISION,
    MIN_SIGMA,
    MOOD_PRECISION,
    SURVIVAL_PRECISION,
    TEXT_AFFECT_PRECISION,
    VALUE_PRECISION_CEILING,
    fast_survival_prior,
    fuse_terms,
    mood_precision,
    occ_prior,
    precision,
    precision_da,
    sigmoid,
    text_affect_precision,
)
from src.agents.value import ValueAgent
from src.orchestration.runner import SessionConfig
from src.orchestration.state import AffectState, Stimulus

# `.env.example:45` 出厂即生效的 deactivation 基线——唤醒否决失效模式的触发前提之一。
AROUSAL_BASELINE = -0.08


# ---------------------------------------------------------------------------
# 1. 零回归：门关时逐字旧行为
#
# 关键：对照**写死在测试里的旧公式**，不是对照被测代码自身——后者只能证明「代码等于
# 它自己」。旧公式抄自本次改动前的 affect_math.py。
# ---------------------------------------------------------------------------


def _legacy_precision(delta: float, value_estimate: float) -> float:
    """改动前 `precision(δ,V)`：max(MIN_PRECISION, σ(α|δ| + β·V))。"""
    return max(1e-3, sigmoid(1.0 * abs(delta) + 0.5 * value_estimate))


def _legacy_precision_da(delta: float) -> float:
    """改动前 `precision_da(δ)`：max(MIN_PRECISION, σ(α|δ|))。"""
    return max(1e-3, sigmoid(1.0 * abs(delta)))


@pytest.mark.parametrize("inten", [-1.0, -0.5, 0.0, 0.3, 0.7, 1.0])
def test_gate_off_survival_precision_matches_legacy_constant(inten: float) -> None:
    """门关：survival 精度逐字等于旧裸常数 0.4（两维皆是）。"""
    _, prec = fast_survival_prior([0.5, 0.0, 0.0, inten], commensurable=False)
    assert prec == (0.4, 0.4)
    assert SURVIVAL_PRECISION == 0.4, "常量本身不得被改动（防『改默认值当零回归』）"


@pytest.mark.parametrize("delta", [-2.0, -0.5, 0.0, 0.5, 2.0])
@pytest.mark.parametrize("value_estimate", [-1.0, 0.0, 1.0])
def test_gate_off_precision_matches_legacy(delta: float, value_estimate: float) -> None:
    """门关：`precision(δ,V)` 逐字等于旧式（默认融合路径的证据精度来源）。"""
    assert precision(delta, value_estimate, commensurable=False) == _legacy_precision(
        delta, value_estimate
    )


@pytest.mark.parametrize("delta", [-2.0, -0.5, 0.0, 0.5, 2.0])
def test_gate_off_precision_da_matches_legacy(delta: float) -> None:
    """门关：`precision_da(δ)` 逐字等于旧式。"""
    assert precision_da(delta, commensurable=False) == _legacy_precision_da(delta)


def test_gate_off_mood_and_text_precision_match_legacy_constants() -> None:
    """门关：mood/text 精度逐字等于旧裸常数 0.8 / 0.3。"""
    assert mood_precision(commensurable=False) == 0.8 == MOOD_PRECISION
    assert text_affect_precision(commensurable=False) == 0.3 == TEXT_AFFECT_PRECISION


def _core_state(**kw: object) -> AffectState:
    """构造一个能走完 AffectCoreAgent 的 state（affect_readout=map 消采样随机性）。"""
    base: dict = {
        "stimulus": Stimulus(name="s", goal_congruence=0.4, intensity=0.6),
        "prior_mu": (0.2, 0.3),
        "prior_sigma": (0.25, 0.25),
        "reward": 0.4,
        "rpe": 0.35,
        "precision": 0.6,
        "value_table": {"s": 0.3},
        "rng_seed": 20260729,
        "affect_readout": "map",
    }
    base.update(kw)
    return AffectState(**base)


def _run_value_then_core(**kw: object) -> dict:
    """跑 ValueAgent → AffectCoreAgent 的真实顺序。

    默认融合分支（`gaussian_fuse`）的证据精度来自 `state.precision`，而它由 **ValueAgent**
    产出（`value.py:22`）——单独调 AffectCoreAgent 看不到该分支的门效果。这一点本身是
    接线事实：**默认路径的门只在有 reward（ValueAgent 真跑）的回合生效**。
    """
    st = _core_state(**kw)
    delta = ValueAgent()(st)
    if delta:
        st = st.model_copy(update=delta)
    return AffectCoreAgent()(st)


_BRANCHES = [
    pytest.param({}, id="branch3-default-gaussian_fuse"),
    pytest.param({"mood_enabled": True, "mood": (0.1, -0.2)}, id="branch2-mood_enabled"),
    pytest.param(
        {"workspace_enabled": True, "mood_enabled": True, "mood": (0.1, -0.2)},
        id="branch1-workspace",
    ),
]


@pytest.mark.parametrize("branch_kw", _BRANCHES)
def test_gate_off_all_three_fusion_branches_unchanged(branch_kw: dict) -> None:
    """门关：三条融合分支的后验与显式 `precision_commensurable=False` 一致。

    ⚠ 本用例只证「默认值 == 显式关」；「等于改动前」由上面对照写死旧公式的用例承担。
    两者合起来才是完整的零回归证据。
    """
    out_default = _run_value_then_core(**branch_kw)
    out_explicit = _run_value_then_core(precision_commensurable=False, **branch_kw)
    assert out_default["post_mu"] == out_explicit["post_mu"]
    assert out_default["post_sigma"] == out_explicit["post_sigma"]


@pytest.mark.parametrize("branch_kw", _BRANCHES)
def test_gate_on_actually_changes_each_branch(branch_kw: dict) -> None:
    """门开：三条分支的后验**都**变化——防某条分支漏接线而静默 no-op。

    这是 zero-link physiology BLOCK-1 同款防护：那次 chat 工厂漏传 SessionConfig，
    门开了但路径完全没生效，测试全绿却什么都没发生。
    """
    off = _run_value_then_core(**branch_kw)
    on = _run_value_then_core(precision_commensurable=True, **branch_kw)
    assert off["post_mu"] != on["post_mu"], f"门开后 post_mu 未变化，该分支可能漏接线：{branch_kw}"


def test_no_window_where_core_runs_without_value_agent() -> None:
    """接线完备性：默认分支的门经 `state.precision` 生效，而它由 ValueAgent 产出。

    若存在「affect_core 跑了但 ValueAgent 没跑」的回合，该回合就会拿旧标度的 precision
    去做齐次化后的融合——半边标度的静默混用。实测两者**共用同一个 `reward is None` 守卫**
    （`value.py:18` 与 `affect_core.py:40`），故该窗口不存在。
    """
    st = _core_state(reward=None, rpe=None)
    assert ValueAgent()(st) == {}, "ValueAgent 应在无 reward 时返回空增量"
    assert AffectCoreAgent()(st) == {}, "affect_core 也应在无 reward 时返回空增量"
    # 有 reward 时两者都跑，precision 被 ValueAgent 覆写（不会用到 state 里的旧值）
    st2 = _core_state(precision=999.0, precision_commensurable=True)
    assert ValueAgent()(st2)["precision"] != 999.0


def test_value_agent_honors_gate() -> None:
    """ValueAgent 的 pi 随门切换——它是默认融合路径唯一的证据精度来源。"""
    agent = ValueAgent()
    out_off = agent(_core_state())
    out_on = agent(_core_state(precision_commensurable=True))
    assert out_off["precision"] == _legacy_precision(out_off["rpe"], out_off["value_estimate"])
    assert out_on["precision"] > out_off["precision"], "门开后 value 流精度应重标定到更高量级"
    assert out_on["precision"] <= VALUE_PRECISION_CEILING


# ---------------------------------------------------------------------------
# 2. 贯通：env → SessionConfig → to_state_flags → AffectState
# ---------------------------------------------------------------------------


def test_session_config_flag_reaches_state_flags() -> None:
    """SessionConfig.precision_commensurable 经 to_state_flags 贯通到 state 初值。"""
    assert SessionConfig().to_state_flags()["precision_commensurable"] is False
    on = SessionConfig(precision_commensurable=True).to_state_flags()
    assert on["precision_commensurable"] is True
    # 该键必须是 AffectState 的合法字段，否则 ainvoke 时被 pydantic 拒
    assert "precision_commensurable" in AffectState.model_fields


def test_env_end_to_end_wiring() -> None:
    """ZERO_PRECISION_COMMENSURABLE 在 chat 工厂里既被读、也被传进 SessionConfig。

    两者缺一即静默失效（门开了但没接上），正是 BLOCK-1 的形状。
    """
    from src.orchestration import chat_driver

    with open(chat_driver.__file__, encoding="utf-8") as fh:
        text = fh.read()
    assert 'os.getenv("ZERO_PRECISION_COMMENSURABLE"' in text
    assert "precision_commensurable=precision_commensurable," in text


def test_mcp_boundary_flag_is_governance_gated() -> None:
    """MCP 面：该门列入治理白名单——client override 不得旁路（改的是默认融合路径）。"""
    from src.mcp_server.server import _MCP_GOVERNANCE_GATED_FLAGS, _build_session_config

    assert "precision_commensurable" in _MCP_GOVERNANCE_GATED_FLAGS
    cfg = _build_session_config({"precision_commensurable": True})
    assert cfg.precision_commensurable is False, "client override 应被静默忽略"


# ---------------------------------------------------------------------------
# 3. fail-fast：混标度在配置期被拒
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({"mood_precision": 1.2}, id="mood-knob-stale"),
        pytest.param({"text_affect_precision": 0.5}, id="text-knob-stale"),
        pytest.param({"mood_precision": 0.9, "text_affect_precision": 0.4}, id="both-stale"),
    ],
)
def test_mixed_scale_rejected(kw: dict) -> None:
    """门开 + 旧标度旋钮被显式改过 → fail-fast（不静默覆盖用户配置）。"""
    with pytest.raises(ValueError, match="不可混用"):
        SessionConfig(precision_commensurable=True, **kw)


def test_default_knobs_allowed_with_gate_on() -> None:
    """旋钮留在默认值时门开合法（否则门等于开不了）。"""
    assert SessionConfig(precision_commensurable=True).precision_commensurable is True


def test_stale_knobs_allowed_with_gate_off() -> None:
    """门关时旧标度旋钮照常可调（不误伤既有用法）。"""
    assert SessionConfig(mood_precision=1.2, text_affect_precision=0.5).mood_precision == 1.2


# ---------------------------------------------------------------------------
# 4. 效果验收 + 锚点
#
# 全部数字取自五流装配（survival/appraisal/value/mood/text）+ arousal_gain +
# deactivation 基线 -0.08，与 `affect_core` workspace 分支一致。换装配数字会变，
# 阈值随之失去意义——故所有阈值都注明实测值与余量。
# ---------------------------------------------------------------------------

_GRID = [-1.0, -0.5, 0.0, 0.5, 1.0]


def _build_streams(
    gc: float,
    sc: float,
    aa: float,
    inten: float,
    *,
    comm: bool,
    mut_surv: bool | None = None,
    mut_val: bool | None = None,
    mut_mood: bool | None = None,
    mut_text: bool | None = None,
) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    """复刻 `affect_core` workspace 分支的五流装配。

    `mut_*` 供变异测试单独关掉某条流的齐次化，模拟「漏接线」这类现实 bug。
    """
    c_surv = comm if mut_surv is None else mut_surv
    c_val = comm if mut_val is None else mut_val
    c_mood = comm if mut_mood is None else mut_mood
    c_text = comm if mut_text is None else mut_text
    mu_o, sig_o, _ = occ_prior(gc, sc, aa, inten, arousal_baseline=AROUSAL_BASELINE)
    gain = 1.0 + AROUSAL_GAIN * max(0.0, mu_o[1])
    app_prec = (gain / max(MIN_SIGMA, sig_o[0]) ** 2, gain / max(MIN_SIGMA, sig_o[1]) ** 2)
    surv_mu, surv_prec = fast_survival_prior([gc, sc, aa, inten], commensurable=c_surv)
    pi_da = precision_da(0.0, commensurable=c_val) * gain
    mp = mood_precision(commensurable=c_mood)
    tp = text_affect_precision(commensurable=c_text)
    return [
        ("survival", surv_mu, surv_prec),
        ("appraisal", mu_o, app_prec),
        ("value", (0.0, 0.0), (MIN_PRECISION, pi_da)),
        ("mood", (0.1, 0.1), (mp, mp)),
        ("text", (0.2, 0.2), (tp, tp)),
    ]


def _share(streams: list, name: str, dim: int = 1) -> float:
    ps = [max(MIN_PRECISION, s[2][dim]) for s in streams]
    me = next(max(MIN_PRECISION, s[2][dim]) for s in streams if s[0] == name)
    return me / sum(ps)


def _n_eff(streams: list, dim: int = 1) -> float:
    """Kish 有效样本量 N_eff = (ΣΠ)²/ΣΠ²。**可观测量**，不做「越大越对」的价值判断。"""
    ps = [max(MIN_PRECISION, s[2][dim]) for s in streams]
    return sum(ps) ** 2 / sum(p * p for p in ps)


def _sweep(comm: bool, **mut: object) -> tuple[list[float], list[float]]:
    shares, neffs = [], []
    for gc, sc, aa, inten in itertools.product(_GRID, _GRID, _GRID, _GRID):
        st = _build_streams(gc, sc, aa, inten, comm=comm, **mut)  # type: ignore[arg-type]
        shares.append(_share(st, "appraisal"))
        neffs.append(_n_eff(st))
    return shares, neffs


def _veto_at(gc: float, inten: float, comm: bool, **mut: object) -> tuple[float, float]:
    """返回（appraisal 占比, 抑制倍数）；抑制 = |排除 appraisal 的 post_a / 含它的 post_a|。"""
    st = _build_streams(gc, 0.0, 0.0, inten, comm=comm, **mut)  # type: ignore[arg-type]
    post_open, _ = fuse_terms([(s[1], s[2]) for s in st])
    post_gated, _ = fuse_terms([(s[1], s[2]) for s in st if s[0] != "appraisal"])
    if abs(post_open[1]) < 1e-12:
        return _share(st, "appraisal"), math.inf
    return _share(st, "appraisal"), abs(post_gated[1] / post_open[1])


# ── 锚点 ①：可比性不变量（输入层判准，不看结果）─────────────────────────────
# 数学席 A1：每条流的 Π 必须能写成 1/σ²，且 σ 表达在被估计量的值域上。[-1,1] 半宽是 1.0；
# σ > 1.0 意味着「该流的不确定性宽于整个可能取值范围」，那它就不该拿到可观权重。


@pytest.mark.parametrize("inten", _GRID)
def test_anchor1_all_streams_sigma_within_domain_half_width(inten: float) -> None:
    """锚点①：门开时每条流反解出的 σ ≤ 1.0（值域半宽）。"""
    for name, _, prec in _build_streams(0.3, 0.0, 0.0, inten, comm=True):
        for dim, p in enumerate(prec):
            if p <= MIN_PRECISION:
                continue  # 有意置极小（value 流 valence 维的独立性修正），不在判准内
            sigma = 1.0 / math.sqrt(p)
            assert sigma <= 1.0, f"{name} 维{dim} 反解 σ={sigma:.3f} > 1.0（宽于值域半宽）"


def test_anchor1_legacy_streams_all_violate() -> None:
    """🔴 判别力证明：门关（现状）时四条非评价流**全部**违反判准。

    这既是本 PRP 的立项依据，也证明锚点①不是恒绿的摆设。
    """
    violations = {}
    for name, _, prec in _build_streams(0.3, 0.0, 0.0, 0.5, comm=False):
        p = max(MIN_PRECISION, prec[1])
        sigma = 1.0 / math.sqrt(p)
        if sigma > 1.0:
            violations[name] = round(sigma, 3)
    assert set(violations) == {"survival", "value", "mood", "text"}, violations
    assert violations["survival"] == pytest.approx(1.581, abs=1e-3)
    assert violations["text"] == pytest.approx(1.826, abs=1e-3)


@pytest.mark.parametrize(
    "mut",
    [
        pytest.param({"mut_surv": False}, id="漏-survival"),
        pytest.param({"mut_val": False}, id="漏-value"),
        pytest.param({"mut_mood": False}, id="漏-mood"),
        pytest.param({"mut_text": False}, id="漏-text"),
    ],
)
def test_anchor1_goes_red_on_any_single_missed_stream(mut: dict) -> None:
    """🔴 变异测试：**漏接任意一条流**，锚点①都变红——逐流敏感。"""
    bad = [
        name
        for name, _, prec in _build_streams(0.3, 0.0, 0.0, 0.5, comm=True, **mut)
        for p in (max(MIN_PRECISION, prec[1]),)
        if p > MIN_PRECISION and 1.0 / math.sqrt(p) > 1.0
    ]
    assert bad, f"变异 {mut} 未被锚点①抓到——该锚点对此 bug 无判别力"


# ── 锚点 ②：单流不再压倒性支配 ──────────────────────────────────────────────
# ⚠ 刻意**不**写「appraisal 占比恒 < X%」——高强度刺激下它占 91.76% 是合法的理想观察者
# 行为（Ernst & Banks 2002：噪声趋零时该模态权重应趋 1）。硬上界会在正确实现上假红，
# 然后被调松，判别力一起调没（线 A ①④ 的教训）。改为断言**分布位移**。
#
# 实测（五流）：门关均值 92.70% / 区间 [80.32%, 98.73%]；门开均值 69.97% / [38.46%, 91.76%]。
# 阈值 0.72 的余量：完整齐次化 69.97%，最接近的变异体「漏 value」74.47%。


def test_anchor2_appraisal_dominance_relaxes() -> None:
    """锚点②：全域穷举下 appraisal 的权重**分布**显著左移。"""
    off_shares, _ = _sweep(False)
    on_shares, _ = _sweep(True)
    off_mean = sum(off_shares) / len(off_shares)
    on_mean = sum(on_shares) / len(on_shares)
    assert off_mean > 0.90, f"门关基线应压倒性支配，实测均值 {off_mean:.2%}"
    assert on_mean < 0.72, f"门开后均值应显著下降，实测 {on_mean:.2%}"
    assert min(on_shares) < 0.50, f"门开后下界应跌破半数，实测 {min(on_shares):.2%}"
    # 上界**允许**仍然很高——那是合法的高确定性场景，不是缺陷
    assert max(on_shares) < max(off_shares)


@pytest.mark.parametrize(
    "mut",
    [
        pytest.param({"mut_surv": False}, id="漏-survival"),
        pytest.param({"mut_val": False}, id="漏-value"),
        pytest.param({"mut_mood": False, "mut_text": False}, id="漏-mood+text"),
    ],
)
def test_anchor2_goes_red_on_partial_homogenization(mut: dict) -> None:
    """🔴 变异测试：漏接任一条流，锚点②的均值判据都变红。

    实测均值：漏 survival 76.98% / 漏 value 74.47% / 漏 mood+text 75.21%，均 ≥ 0.72。
    """
    shares, _ = _sweep(True, **mut)
    mean = sum(shares) / len(shares)
    assert mean >= 0.72, f"变异 {mut} 未被锚点②抓到（实测均值 {mean:.2%}）"


# ── 锚点 ③：N_eff 不再钉在 1 ─────────────────────────────────────────────────
# 实测：门关均值 1.175；门开 2.117。阈值 1.95 的余量：最接近的变异体「漏 value」1.839。


def test_anchor3_effective_sample_size_rises() -> None:
    """锚点③：N_eff 从「几乎只有一条流在说话」抬升。"""
    _, off_neffs = _sweep(False)
    _, on_neffs = _sweep(True)
    off_mean = sum(off_neffs) / len(off_neffs)
    on_mean = sum(on_neffs) / len(on_neffs)
    assert off_mean < 1.30, f"门关基线 N_eff 应接近 1，实测 {off_mean:.3f}"
    assert on_mean > 1.95, f"门开后 N_eff 应显著抬升，实测 {on_mean:.3f}"


@pytest.mark.parametrize(
    "mut",
    [
        pytest.param({"mut_surv": False}, id="漏-survival"),
        pytest.param({"mut_val": False}, id="漏-value"),
        pytest.param({"mut_mood": False, "mut_text": False}, id="漏-mood+text"),
    ],
)
def test_anchor3_goes_red_on_partial_homogenization(mut: dict) -> None:
    """🔴 变异测试：漏接任一条流，N_eff 回落到阈值以下。

    实测：漏 survival 1.738 / 漏 value 1.839 / 漏 mood+text 1.784，均 ≤ 1.95。
    """
    _, neffs = _sweep(True, **mut)
    mean = sum(neffs) / len(neffs)
    assert mean <= 1.95, f"变异 {mut} 未被锚点③抓到（实测 N_eff {mean:.3f}）"


# ── 锚点 ④：唤醒否决失效模式消失（🛑 线 A 解冻判据）──────────────────────────
# 第五次推翻：`ZERO_AROUSAL_BASELINE=-0.08`（`.env.example:45` 出厂即生效）下，
# valence=0 且 0.4|I|=0.08 时 occ arousal **归零**，但 Π=11.11 使它占 84.75% 权重，
# 于是「零偏离的高精度流」把别的流的 arousal 一起按到近零。
# 见 notes/2026-07-29-arousal-veto-fifth-overturn.md。


def test_veto_point_produces_zero_arousal_deviation() -> None:
    """前提核验：gc=sc=aa=0 且 I=0.2 时 occ arousal 归零（否则整条锚点打空）。

    ⚠ 不是**精确** 0：`0.4*0.2 == 0.08000000000000002`，减 0.08 余 1.39e-17。
    纪要里的「精确 0.000000」是格式化输出的理想化。1e-17 比其它流的 μ（O(0.1)）低 16 个
    数量级，机制上等同零偏离，但断言必须写成 `< 1e-15` 而不是 `== 0.0`。
    """
    mu, _, _ = occ_prior(0.0, 0.0, 0.0, 0.2, arousal_baseline=AROUSAL_BASELINE)
    assert abs(mu[1]) < 1e-15, f"否决点前提不成立，μ_a={mu[1]!r}"
    assert mu[1] != 0.0, "若真变成精确 0，说明公式改了，本用例的浮点说明需同步更新"
    # 且它此时精度并不低——这才是「否决」的机制
    conf = 0.3 + 0.5 * 0.2
    assert 1.0 / max(MIN_SIGMA, 0.5 * (1 - conf)) ** 2 == pytest.approx(11.111, abs=1e-3)


def test_anchor4_arousal_veto_disappears() -> None:
    """锚点④（🛑 线 A 解冻判据）：零偏离高精度流对 arousal 的压制大幅收敛。

    实测：占比 84.75% → 45.53%；抑制 6.6× → 1.8×。
    """
    share_off, sup_off = _veto_at(0.0, 0.2, comm=False)
    share_on, sup_on = _veto_at(0.0, 0.2, comm=True)
    assert share_off > 0.80, f"门关基线应压倒性支配，实测 {share_off:.2%}"
    assert share_on < 0.60, f"门开后占比应显著下降，实测 {share_on:.2%}"
    assert sup_off > 5.0, f"门关基线应有强抑制，实测 {sup_off:.1f}×"
    assert sup_on < 3.0, f"门开后抑制应收敛，实测 {sup_on:.1f}×"


def test_anchor4_worst_case_suppression_and_sign_flips() -> None:
    """锚点④补充：全网格最坏抑制倍数 + arousal 符号翻转率。

    符号翻转是唤醒否决最严重的下游后果——后验唤醒的**方向**被一条零偏离流反转。
    实测：最坏抑制 110.4× → 2.0×；符号翻转 0.773% → 0.000%。
    """
    n = 41
    results = {}
    for comm in (False, True):
        worst, flips, total = 0.0, 0, 0
        for i in range(n):
            gc = -1.0 + 2.0 * i / (n - 1)
            for j in range(n):
                inten = -1.0 + 2.0 * j / (n - 1)
                st = _build_streams(gc, 0.0, 0.0, inten, comm=comm)
                po, _ = fuse_terms([(s[1], s[2]) for s in st])
                pg, _ = fuse_terms([(s[1], s[2]) for s in st if s[0] != "appraisal"])
                total += 1
                if po[1] * pg[1] < 0:
                    flips += 1
                if abs(po[1]) > 1e-9:
                    worst = max(worst, abs(pg[1] / po[1]))
        results[comm] = (worst, flips / total)
    (worst_off, flip_off), (worst_on, flip_on) = results[False], results[True]
    assert worst_off > 50.0, f"门关基线最坏抑制应很大，实测 {worst_off:.1f}×"
    assert worst_on < 3.0, f"门开后最坏抑制应收敛，实测 {worst_on:.1f}×"
    assert flip_off > 0.005, f"门关基线应存在符号翻转，实测 {flip_off:.3%}"
    assert flip_on == 0.0, f"门开后符号翻转应清零，实测 {flip_on:.3%}"


def test_anchor4_is_insensitive_to_single_missed_stream() -> None:
    """🔴 **锚点④抓不到什么**——如实记录其判别力边界，不假装它覆盖了单流漏接。

    实测否决点抑制倍数：完整 1.8× / 漏 survival 2.3× / 漏 value 2.0× / 漏 mood+text 2.2×，
    **全部低于 3.0 阈值**。即：锚点④只判别「有没有做齐次化」，不判别「做全了没」。
    逐流覆盖由锚点①②③承担（见上面三组变异测试）。

    写成用例而非注释，是为了让这条边界随代码一起演进——若哪天变异体真被抓到了，
    本用例会红，提示回来更新判别力说明。
    """
    for mut in ({"mut_surv": False}, {"mut_val": False}, {"mut_mood": False, "mut_text": False}):
        _, sup = _veto_at(0.0, 0.2, comm=True, **mut)
        assert sup < 3.0, f"锚点④意外抓到了 {mut}（{sup:.1f}×）——请更新判别力边界说明"


def test_anchor4_goes_red_without_homogenization() -> None:
    """🔴 变异测试：完全不做齐次化时锚点④变红（6.6× ≥ 3.0）。"""
    _, sup = _veto_at(0.0, 0.2, comm=False)
    assert sup >= 3.0, "变异未生效——锚点④无判别力"


# ---------------------------------------------------------------------------
# 5. 边界与不变量
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delta", [-5.0, -1.0, 0.0, 1.0, 5.0])
def test_commensurable_value_precision_within_ceiling(delta: float) -> None:
    """门开时 value 流精度恒在 [MIN_PRECISION, VALUE_PRECISION_CEILING) 内。"""
    p = precision_da(delta, commensurable=True)
    assert MIN_PRECISION <= p < VALUE_PRECISION_CEILING


@pytest.mark.parametrize("inten", [-1.0, -0.3, 0.0, 0.3, 1.0])
def test_commensurable_survival_stays_weaker_than_appraisal(inten: float) -> None:
    """齐次化后 survival 仍恒弱于 appraisal——「粗快=不确定」的设计意图在同一把尺子上成立。"""
    _, surv = fast_survival_prior([0.5, 0.0, 0.0, inten], commensurable=True)
    _, sig_o, _ = occ_prior(0.5, 0.0, 0.0, inten)
    app = 1.0 / max(MIN_SIGMA, sig_o[0]) ** 2
    assert surv[0] < app, f"I={inten}: survival {surv[0]:.3f} 应弱于 appraisal {app:.3f}"


def test_commensurable_preserves_monotonicity() -> None:
    """齐次化是**尺度重标定**：单调性/序关系必须原样保留（否则就不只是换尺子了）。"""
    deltas = [0.0, 0.2, 0.5, 1.0, 2.0]
    assert [precision_da(d, commensurable=False) for d in deltas] == sorted(
        precision_da(d, commensurable=False) for d in deltas
    )
    assert [precision_da(d, commensurable=True) for d in deltas] == sorted(
        precision_da(d, commensurable=True) for d in deltas
    )
    intens = [0.0, 0.25, 0.5, 0.75, 1.0]
    surv = [fast_survival_prior([0, 0, 0, i], commensurable=True)[1][0] for i in intens]
    assert surv == sorted(surv)
