"""P3 · 1-A 层级预测编码 hierarchical_fuse 单测 + 贯通零回归。

覆盖：
  1. 退化 A（layers=1）→ 逐字等价 fuse_terms(all)，<1e-9
  2. 退化 B（layers=2, coupling=0）→ 同上
  3. coupling>0 语义：输出 = L1 核心后验；L0 经 w²·π_L0 影响 L1
  4. 单调性：w↑ 时 L0 对 μ_core 的拉动增强（L0 与 L1 均值异号时可测）
  5. 有界性网格扫描（μ∈{-1,0,1}²×精度极端）：无越界/无 NaN
  6. 稳定性/硬拒：coupling>1.0 → ValueError；边界 0.0/1.0 不报错
  7. 边界：L0 空 → fallback fuse_terms(all)；L1 空 → fallback fuse_terms(all)
  8. 贯通零回归：SessionConfig 默认 + ZERO_HPC_* env → build_chat_driver → session.config
  9. to_state_flags() 含两键且与旧 flags 集合一致

期望值全部从方程 / fuse_terms 派生，不凭记忆硬编码。
"""

from __future__ import annotations

import math

import pytest

from src.agents.affect_math import (
    MAX_SAMPLE_SIGMA,
    MIN_PRECISION,
    MIN_SIGMA,
    fuse_terms,
    hierarchical_fuse,
)
from src.orchestration.runner import SessionConfig

# ---------------------------------------------------------------------------
# 测试用 named_terms 构造辅助
# ---------------------------------------------------------------------------

# 「混合」集：含 survival + mood(L0) + appraisal + value(L1)
_NAMED_MIXED: list[tuple[str, tuple[float, float], tuple[float, float]]] = [
    ("survival", (0.3, 0.2), (0.4, 0.4)),
    ("mood", (-0.1, 0.15), (0.8, 0.8)),
    ("appraisal", (0.5, 0.6), (1.2, 1.2)),
    ("value", (0.1, 0.4), (0.7, 0.7)),
]

# 「对比」集：L0 明显负效价，L1 明显正效价（用于单调性断言）
_NAMED_CONTRAST: list[tuple[str, tuple[float, float], tuple[float, float]]] = [
    ("survival", (-0.8, 0.5), (0.4, 0.4)),  # L0: 负效价
    ("mood", (-0.6, 0.3), (0.8, 0.8)),  # L0: 负效价
    ("appraisal", (0.7, 0.6), (1.5, 1.5)),  # L1: 正效价
    ("value", (0.5, 0.4), (0.9, 0.9)),  # L1: 正效价
]

# 仅 L1（无 survival/mood），用于 L0 空边界
_NAMED_NO_L0: list[tuple[str, tuple[float, float], tuple[float, float]]] = [
    ("appraisal", (0.5, 0.6), (1.2, 1.2)),
    ("value", (0.1, 0.4), (0.7, 0.7)),
]

# 仅 L0（无 appraisal/value），用于 L1 空边界
_NAMED_NO_L1: list[tuple[str, tuple[float, float], tuple[float, float]]] = [
    ("survival", (-0.8, 0.5), (0.4, 0.4)),
    ("mood", (-0.6, 0.3), (0.8, 0.8)),
]


def _plain(
    named: list[tuple[str, tuple[float, float], tuple[float, float]]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """剥 name → fuse_terms 格式。"""
    return [(mu, prec) for _, mu, prec in named]


# ---------------------------------------------------------------------------
# 1. 退化 A：layers=1 → 逐字等价 fuse_terms(all)
# ---------------------------------------------------------------------------


class TestDegenerateA:
    """layers=1 → 内部直接走 fuse_terms(all)，<1e-9 浮点零误差。"""

    def test_layers_1_equals_fuse_terms_mixed(self) -> None:
        """混合 terms：layers=1 逐维 <1e-9。"""
        ft = fuse_terms(_plain(_NAMED_MIXED))
        hf = hierarchical_fuse(_NAMED_MIXED, layers=1)
        assert abs(ft[0][0] - hf[0][0]) < 1e-9, "valence 不等"
        assert abs(ft[0][1] - hf[0][1]) < 1e-9, "arousal 不等"
        assert abs(ft[1][0] - hf[1][0]) < 1e-9, "sigma_v 不等"
        assert abs(ft[1][1] - hf[1][1]) < 1e-9, "sigma_a 不等"

    def test_layers_1_equals_fuse_terms_contrast(self) -> None:
        """对比 terms（L0 负, L1 正）：layers=1 逐维 <1e-9。"""
        ft = fuse_terms(_plain(_NAMED_CONTRAST))
        hf = hierarchical_fuse(_NAMED_CONTRAST, layers=1)
        assert abs(ft[0][0] - hf[0][0]) < 1e-9
        assert abs(ft[0][1] - hf[0][1]) < 1e-9

    def test_layers_1_default_param_is_one(self) -> None:
        """不传 layers 时默认=1，等价 layers=1 调用。"""
        hf_default = hierarchical_fuse(_NAMED_MIXED)
        hf_explicit = hierarchical_fuse(_NAMED_MIXED, layers=1)
        assert hf_default == hf_explicit


# ---------------------------------------------------------------------------
# 2. 退化 B：layers=2, coupling=0.0 → 逐字等价 fuse_terms(all)
# ---------------------------------------------------------------------------


class TestDegenerateB:
    """coupling=0.0 时关闭层级（flatten 旁路）：同一 fuse_terms 代码路径，<1e-9。"""

    def test_coupling_zero_equals_fuse_terms_mixed(self) -> None:
        """coupling=0 混合 terms：逐维 <1e-9。"""
        ft = fuse_terms(_plain(_NAMED_MIXED))
        hf = hierarchical_fuse(_NAMED_MIXED, layers=2, coupling=0.0)
        assert abs(ft[0][0] - hf[0][0]) < 1e-9
        assert abs(ft[0][1] - hf[0][1]) < 1e-9
        assert abs(ft[1][0] - hf[1][0]) < 1e-9
        assert abs(ft[1][1] - hf[1][1]) < 1e-9

    def test_coupling_zero_equals_fuse_terms_contrast(self) -> None:
        """coupling=0 对比 terms：逐维 <1e-9。"""
        ft = fuse_terms(_plain(_NAMED_CONTRAST))
        hf = hierarchical_fuse(_NAMED_CONTRAST, layers=2, coupling=0.0)
        assert abs(ft[0][0] - hf[0][0]) < 1e-9
        assert abs(ft[0][1] - hf[0][1]) < 1e-9

    def test_coupling_zero_default_param_is_zero(self) -> None:
        """不传 coupling 时默认=0.0，等价 coupling=0.0 调用。"""
        hf_default = hierarchical_fuse(_NAMED_MIXED, layers=2)
        hf_zero = hierarchical_fuse(_NAMED_MIXED, layers=2, coupling=0.0)
        assert hf_default == hf_zero


# ---------------------------------------------------------------------------
# 3. coupling>0 语义：L0 经 w²·π_L0 影响 L1；输出 ≠ 退化值
# ---------------------------------------------------------------------------


class TestCouplingSemantics:
    """coupling∈(0,1] 时：输出取 L1 核心后验，且确实不同于 flat fuse_terms(all)。"""

    def test_output_differs_from_flat_fuse_terms(self) -> None:
        """coupling=0.5 时输出 μ 与 fuse_terms(all) 不同（L0 影响了 L1）。"""
        ft = fuse_terms(_plain(_NAMED_CONTRAST))
        hf = hierarchical_fuse(_NAMED_CONTRAST, layers=2, coupling=0.5)
        # 至少一维不同（L0 拉动了后验）
        diff_v = abs(hf[0][0] - ft[0][0])
        diff_a = abs(hf[0][1] - ft[0][1])
        assert diff_v > 1e-9 or diff_a > 1e-9, (
            "coupling=0.5 时输出应与 flat fuse_terms(all) 不同，实际两者相等"
        )

    def test_output_matches_manual_five_step_formula(self) -> None:
        """coupling=0.5 对比 terms：逐维手算五步公式核验。

        手算方程（design.md 六-bis）：
          ① π_L0 = Σπ_i (L0); μ_L0 = Σπ_i·μ_i / π_L0
          ② π_L1e = Σπ_j (L1); μ_L1e = Σπ_j·μ_j / π_L1e
          ⑤ π_core = π_L1e + w²·π_L0
             μ_core = (π_L1e·μ_L1e + w²·π_L0·μ_L0) / π_core
             σ_core = sqrt(1/π_core) → clamp(MIN_SIGMA, MAX_SAMPLE_SIGMA)
             μ_core → clamp(-1, 1)
        """
        w = 0.5
        w2 = w * w

        for d in range(2):  # valence / arousal
            # ① L0
            num_l0, den_l0 = 0.0, 0.0
            for name, mu, prec in _NAMED_CONTRAST:
                if name in {"survival", "mood"}:
                    p = max(MIN_PRECISION, prec[d])
                    num_l0 += p * mu[d]
                    den_l0 += p
            pi_l0 = max(MIN_PRECISION, den_l0)
            mu_l0 = num_l0 / pi_l0

            # ② L1
            num_l1, den_l1 = 0.0, 0.0
            for name, mu, prec in _NAMED_CONTRAST:
                if name not in {"survival", "mood"}:
                    p = max(MIN_PRECISION, prec[d])
                    num_l1 += p * mu[d]
                    den_l1 += p
            pi_l1e = max(MIN_PRECISION, den_l1)
            mu_l1e = num_l1 / pi_l1e

            # ⑤
            pi_core = max(MIN_PRECISION, pi_l1e + w2 * pi_l0)
            mu_core_raw = (pi_l1e * mu_l1e + w2 * pi_l0 * mu_l0) / pi_core
            mu_core_exp = max(-1.0, min(1.0, mu_core_raw))
            sig_core_exp = max(MIN_SIGMA, min(MAX_SAMPLE_SIGMA, math.sqrt(1.0 / pi_core)))

            hf = hierarchical_fuse(_NAMED_CONTRAST, layers=2, coupling=w)
            assert abs(hf[0][d] - mu_core_exp) < 1e-9, (
                f"dim={d}: μ_core 手算={mu_core_exp:.9f} 函数={hf[0][d]:.9f}"
            )
            assert abs(hf[1][d] - sig_core_exp) < 1e-9, (
                f"dim={d}: σ_core 手算={sig_core_exp:.9f} 函数={hf[1][d]:.9f}"
            )


# ---------------------------------------------------------------------------
# 4. 单调性：coupling 从 0.2→0.8，L0 拉动增强
# ---------------------------------------------------------------------------


class TestMonotonicity:
    """L0 负效价, L1 正效价：w↑ → μ_core 向 L0（负向）单调移动。"""

    def test_valence_monotone_decreasing_with_coupling(self) -> None:
        """coupling 从 0.2→0.5→0.8，valence 单调递减（L0 负向拉力增强）。"""
        hf_02 = hierarchical_fuse(_NAMED_CONTRAST, layers=2, coupling=0.2)
        hf_05 = hierarchical_fuse(_NAMED_CONTRAST, layers=2, coupling=0.5)
        hf_08 = hierarchical_fuse(_NAMED_CONTRAST, layers=2, coupling=0.8)
        assert hf_02[0][0] > hf_05[0][0] > hf_08[0][0], (
            f"期望 valence 单调递减: w=0.2:{hf_02[0][0]:.6f} "
            f"w=0.5:{hf_05[0][0]:.6f} w=0.8:{hf_08[0][0]:.6f}"
        )

    def test_coupling_1_more_l0_influence_than_half(self) -> None:
        """coupling=1.0（最大合法值）比 coupling=0.5 更偏向 L0（效价更小）。"""
        hf_05 = hierarchical_fuse(_NAMED_CONTRAST, layers=2, coupling=0.5)
        hf_10 = hierarchical_fuse(_NAMED_CONTRAST, layers=2, coupling=1.0)
        # L0 负，L1 正 → coupling 更大时 μ_core 更偏负
        assert hf_10[0][0] < hf_05[0][0], (
            f"coupling=1.0 应使 valence 更低（更偏 L0）: "
            f"1.0:{hf_10[0][0]:.6f} 0.5:{hf_05[0][0]:.6f}"
        )

    def test_coupling_zero_is_flat_not_l1_only(self) -> None:
        """coupling=0 = fuse_terms(all)，而非仅 L1（验证 flatten 旁路语义正确）。"""
        hf_zero = hierarchical_fuse(_NAMED_CONTRAST, layers=2, coupling=0.0)
        ft_all = fuse_terms(_plain(_NAMED_CONTRAST))
        ft_l1_only = fuse_terms(
            [(mu, prec) for name, mu, prec in _NAMED_CONTRAST if name not in {"survival", "mood"}]
        )
        assert abs(hf_zero[0][0] - ft_all[0][0]) < 1e-9, "coupling=0 应等于 fuse_terms(all)"
        # fuse_terms(all) 和 fuse_terms(L1_only) 应不同（L0 项存在影响了均值）
        assert abs(ft_all[0][0] - ft_l1_only[0][0]) > 1e-9, (
            "fuse_terms(all) 与 fuse_terms(L1_only) 应不同（否则 L0 项无效）"
        )


# ---------------------------------------------------------------------------
# 5. 有界性网格扫描
# ---------------------------------------------------------------------------


class TestBoundedness:
    """μ∈{-1,0,1}² × 精度∈{MIN_PRECISION,1.0,50.0} × coupling∈{0.1,0.5,1.0}：
    μ_core∈[-1,1]、σ_core∈[MIN_SIGMA, MAX_SAMPLE_SIGMA]、无 NaN。
    """

    def test_grid_scan_no_violations(self) -> None:
        """全网格无越界、无 NaN。"""
        mu_vals = [-1.0, 0.0, 1.0]
        prec_vals = [MIN_PRECISION, 1.0, 50.0]
        coupling_vals = [0.1, 0.5, 1.0]

        violations = []
        for mv in mu_vals:
            for ma in mu_vals:
                for pv in prec_vals:
                    for pa in prec_vals:
                        terms = [
                            ("survival", (mv, ma), (pv, pa)),
                            ("mood", (mv * 0.5, ma * 0.5), (pv * 0.5, pa * 0.5)),
                            ("appraisal", (-mv * 0.8, -ma * 0.8), (pv, pa)),
                            ("value", (mv * 0.3, ma * 0.3), (pv * 0.3, pa * 0.3)),
                        ]
                        for w in coupling_vals:
                            result = hierarchical_fuse(terms, layers=2, coupling=w)
                            mu_c, sig_c = result[0], result[1]
                            for dim, (mc, sc) in enumerate(zip(mu_c, sig_c, strict=True)):
                                if math.isnan(mc) or math.isnan(sc):
                                    violations.append(
                                        f"NaN: dim={dim} mv={mv} ma={ma} pv={pv} pa={pa} w={w}"
                                    )
                                elif not (-1.0 <= mc <= 1.0):
                                    violations.append(f"mu d{dim}={mc:.3f} w={w} in=({mv},{ma})")
                                elif not (MIN_SIGMA <= sc <= MAX_SAMPLE_SIGMA):
                                    violations.append(f"sig d{dim}={sc:.3f} w={w}")

        assert not violations, f"有界性违反 {len(violations)} 处:\n" + "\n".join(violations[:5])

    def test_extreme_precision_no_divide_by_zero(self) -> None:
        """极端精度（很小/很大）不触发除零或 NaN。"""
        terms_min = [
            ("survival", (0.5, 0.5), (MIN_PRECISION, MIN_PRECISION)),
            ("appraisal", (0.3, 0.3), (MIN_PRECISION, MIN_PRECISION)),
        ]
        result = hierarchical_fuse(terms_min, layers=2, coupling=0.5)
        assert not any(
            math.isnan(x) for x in [result[0][0], result[0][1], result[1][0], result[1][1]]
        )

        terms_huge = [
            ("survival", (0.5, 0.5), (100.0, 100.0)),
            ("appraisal", (0.3, 0.3), (100.0, 100.0)),
        ]
        result2 = hierarchical_fuse(terms_huge, layers=2, coupling=0.5)
        assert not any(
            math.isnan(x) for x in [result2[0][0], result2[0][1], result2[1][0], result2[1][1]]
        )


# ---------------------------------------------------------------------------
# 6. 稳定性/硬拒
# ---------------------------------------------------------------------------


class TestCouplingValidation:
    """coupling>1.0 → ValueError（硬拒不 clamp）；边界 0.0/1.0 不报错。"""

    def test_coupling_gt_1_raises_value_error(self) -> None:
        """coupling=1.01 → ValueError。"""
        with pytest.raises(ValueError, match="coupling"):
            hierarchical_fuse(_NAMED_MIXED, layers=2, coupling=1.01)

    def test_coupling_2_raises_value_error(self) -> None:
        """coupling=2.0 → ValueError。"""
        with pytest.raises(ValueError):
            hierarchical_fuse(_NAMED_MIXED, layers=2, coupling=2.0)

    def test_coupling_negative_raises_value_error(self) -> None:
        """coupling<0 → ValueError（design w∈(0,1]，负值反向 top-down；WARN-2）。"""
        with pytest.raises(ValueError, match="coupling"):
            hierarchical_fuse(_NAMED_MIXED, layers=2, coupling=-0.5)

    def test_coupling_0_boundary_ok(self) -> None:
        """coupling=0.0（边界）不报错，走退化旁路。"""
        result = hierarchical_fuse(_NAMED_MIXED, layers=2, coupling=0.0)
        assert result is not None

    def test_coupling_1_boundary_ok(self) -> None:
        """coupling=1.0（边界最大合法值）不报错、输出合法。

        数学注：w=1 时 π_core = π_L1e + π_L0 = π_total，μ_core 恒等于 fuse_terms(all)
        ——这是正确的数学性质（凸组合权重和=1），不是 bug。边界测试只验证「不抛异常、
        输出在合法范围内」；差异测试由 coupling=0.5 覆盖（TestMonotonicity）。
        """
        result = hierarchical_fuse(_NAMED_CONTRAST, layers=2, coupling=1.0)
        assert result is not None
        mu_c, sig_c = result
        assert -1.0 <= mu_c[0] <= 1.0, f"valence 越界: {mu_c[0]}"
        assert -1.0 <= mu_c[1] <= 1.0, f"arousal 越界: {mu_c[1]}"
        assert MIN_SIGMA <= sig_c[0] <= MAX_SAMPLE_SIGMA, f"sigma_v 越界: {sig_c[0]}"
        assert MIN_SIGMA <= sig_c[1] <= MAX_SAMPLE_SIGMA, f"sigma_a 越界: {sig_c[1]}"


# ---------------------------------------------------------------------------
# 7. 边界：L0 空 / L1 空 → fallback fuse_terms(all)
# ---------------------------------------------------------------------------


class TestBoundaryEmptyLayers:
    """L0 空（无 survival/mood）或 L1 空（只有 survival/mood）→ fallback fuse_terms(all)。"""

    def test_l0_empty_equals_fuse_terms_all(self) -> None:
        """无 L0 项：coupling=0.5 → fallback = fuse_terms(all)，逐维 <1e-9。"""
        ft = fuse_terms(_plain(_NAMED_NO_L0))
        hf = hierarchical_fuse(_NAMED_NO_L0, layers=2, coupling=0.5)
        assert abs(ft[0][0] - hf[0][0]) < 1e-9, f"L0空 valence: ft={ft[0][0]:.9f} hf={hf[0][0]:.9f}"
        assert abs(ft[0][1] - hf[0][1]) < 1e-9, f"L0空 arousal: ft={ft[0][1]:.9f} hf={hf[0][1]:.9f}"
        assert abs(ft[1][0] - hf[1][0]) < 1e-9, "L0空 sigma_v 不等"
        assert abs(ft[1][1] - hf[1][1]) < 1e-9, "L0空 sigma_a 不等"

    def test_l1_empty_equals_fuse_terms_all(self) -> None:
        """无 L1 项：coupling=0.5 → fallback = fuse_terms(all)，逐维 <1e-9。"""
        ft = fuse_terms(_plain(_NAMED_NO_L1))
        hf = hierarchical_fuse(_NAMED_NO_L1, layers=2, coupling=0.5)
        assert abs(ft[0][0] - hf[0][0]) < 1e-9, f"L1空 valence: ft={ft[0][0]:.9f} hf={hf[0][0]:.9f}"
        assert abs(ft[0][1] - hf[0][1]) < 1e-9, f"L1空 arousal: ft={ft[0][1]:.9f} hf={hf[0][1]:.9f}"

    def test_l0_empty_layers1_also_equals_fuse_terms(self) -> None:
        """L0 空 + layers=1：两条退化路径结果相同（均等价 fuse_terms(all)）。"""
        hf_l1 = hierarchical_fuse(_NAMED_NO_L0, layers=1)
        hf_c0 = hierarchical_fuse(_NAMED_NO_L0, layers=2, coupling=0.5)
        assert abs(hf_l1[0][0] - hf_c0[0][0]) < 1e-9

    def test_single_item_no_crash(self) -> None:
        """单项（既是 L0 又是唯一 → L1 空）：不崩溃。"""
        terms_single = [("survival", (0.3, 0.4), (0.5, 0.5))]
        result = hierarchical_fuse(terms_single, layers=2, coupling=0.5)
        # 应 fallback fuse_terms(all)
        ft = fuse_terms(_plain(terms_single))
        assert abs(result[0][0] - ft[0][0]) < 1e-9


# ---------------------------------------------------------------------------
# 8 & 9. 贯通零回归：SessionConfig + build_chat_driver env
# ---------------------------------------------------------------------------


class TestHPCPassthrough:
    """P3 HPC 旋钮贯通零回归（仿 test_b8_integration.py TestSessionConfig 范式）。"""

    # ── 8a：SessionConfig 默认值 ──

    def test_session_config_default_hierarchical_layers(self) -> None:
        """SessionConfig() 默认 hierarchical_layers == 1（平层零回归）。"""
        cfg = SessionConfig()
        assert cfg.hierarchical_layers == 1

    def test_session_config_default_hierarchical_coupling(self) -> None:
        """SessionConfig() 默认 hierarchical_coupling == 0.0（关层级零回归）。"""
        cfg = SessionConfig()
        assert cfg.hierarchical_coupling == pytest.approx(0.0)

    # ── 8b：to_state_flags 含两键且默认值正确 ──

    def test_to_state_flags_contains_hierarchical_layers(self) -> None:
        """to_state_flags() 包含 'hierarchical_layers' 键，值为 1。"""
        flags = SessionConfig().to_state_flags()
        assert "hierarchical_layers" in flags
        assert flags["hierarchical_layers"] == 1

    def test_to_state_flags_contains_hierarchical_coupling(self) -> None:
        """to_state_flags() 包含 'hierarchical_coupling' 键，值为 0.0。"""
        flags = SessionConfig().to_state_flags()
        assert "hierarchical_coupling" in flags
        assert flags["hierarchical_coupling"] == pytest.approx(0.0)

    # ── 8c：零回归断言——旧 flags 键不受影响 ──

    def test_zero_regression_old_flags_unchanged(self) -> None:
        """不设新 env 时，to_state_flags() 旧键与预期逐字一致（零回归）。"""
        cfg = SessionConfig()
        flags = cfg.to_state_flags()
        old_defaults = {
            "regulation_enabled": False,
            "regulation_strategy": "suppression",
            "mood_enabled": False,
            "recall_enabled": False,
            "language_enabled": False,
            "workspace_enabled": False,
            "appraisal_conditioning_enabled": False,
            "language_max_iters": 3,
            "rng_seed": None,
            "sample_sigma_cap": None,
            "affect_readout": "sample",
            "arousal_baseline": 0.0,
            "arousal_gain_cap": None,
            "precision_split": False,
            "fuse_independence_correct": False,
            "ignition_survival_fallback": False,
        }
        for k, v in old_defaults.items():
            assert flags[k] == v, f"零回归失败: {k}={flags[k]!r} 期望={v!r}"

    # ── 8d：env 设值 → build_chat_driver → session.config ──

    def test_env_hpc_layers_2_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_HPC_LAYERS=2 → build_chat_driver → session.config.hierarchical_layers==2。"""
        monkeypatch.setenv("ZERO_HPC_LAYERS", "2")
        monkeypatch.setenv("ZERO_HPC_COUPLING", "0.5")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        from src.orchestration.chat_driver import build_chat_driver

        driver = build_chat_driver(thread="test-hpc-layers")
        assert driver.session.config.hierarchical_layers == 2

    def test_env_hpc_coupling_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_HPC_COUPLING=0.5 → build_chat_driver → session.config.hierarchical_coupling==0.5"""
        monkeypatch.setenv("ZERO_HPC_LAYERS", "2")
        monkeypatch.setenv("ZERO_HPC_COUPLING", "0.5")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        from src.orchestration.chat_driver import build_chat_driver

        driver = build_chat_driver(thread="test-hpc-coupling")
        assert driver.session.config.hierarchical_coupling == pytest.approx(0.5)

    def test_no_env_hpc_default_layers_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """未设 ZERO_HPC_LAYERS → session.config.hierarchical_layers==1（零回归）。"""
        monkeypatch.delenv("ZERO_HPC_LAYERS", raising=False)
        monkeypatch.delenv("ZERO_HPC_COUPLING", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        from src.orchestration.chat_driver import build_chat_driver

        driver = build_chat_driver(thread="test-hpc-default")
        assert driver.session.config.hierarchical_layers == 1
        assert driver.session.config.hierarchical_coupling == pytest.approx(0.0)

    # ── 8e：ConversationSession 旧展开参数路径 ──

    def test_conversation_session_legacy_params(self) -> None:
        """旧展开参数传 hierarchical_layers/coupling → session.config 字段正确。"""
        from src.orchestration.runner import ConversationSession

        session = ConversationSession(
            thread_id="t",
            hierarchical_layers=2,
            hierarchical_coupling=0.6,
        )
        assert session.config.hierarchical_layers == 2
        assert session.config.hierarchical_coupling == pytest.approx(0.6)

    def test_conversation_session_config_priority(self) -> None:
        """传 config= 时 hierarchical_layers/coupling 优先于旧展开参数。"""
        from src.orchestration.runner import ConversationSession

        cfg = SessionConfig(hierarchical_layers=2, hierarchical_coupling=0.7)
        session = ConversationSession(
            thread_id="t",
            config=cfg,
            hierarchical_layers=1,  # 应被忽略
            hierarchical_coupling=0.0,  # 应被忽略
        )
        assert session.config.hierarchical_layers == 2
        assert session.config.hierarchical_coupling == pytest.approx(0.7)

    def test_to_state_flags_hpc_in_model_dump(self) -> None:
        """SessionConfig 显式设 HPC 旋钮，model_dump() 展开后含正确值。"""
        cfg = SessionConfig(hierarchical_layers=2, hierarchical_coupling=0.8)
        flags = cfg.to_state_flags()
        assert flags["hierarchical_layers"] == 2
        assert flags["hierarchical_coupling"] == pytest.approx(0.8)
