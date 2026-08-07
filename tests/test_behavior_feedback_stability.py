"""行为反馈流 × mood 耦合稳定性判据测试（T3 证明层；议会必改 #1/#3/#8）。

「先证明后实现」的绿灯本体：本文件全绿是 T4 流接线的前置条件。判据数值系议会钉死
（K=10 / m0 步长 0.01 / 盆边界位移阈 0.03），变异三类证明判据能红（绿灯先证能红）。
仿真只走 affect_math 纯函数层（CS 席约束），全确定无 RNG。
"""

from __future__ import annotations

import pytest

from scripts.calibrate_sigma_bhv import calibrate
from scripts.simulate_behavior_feedback_coupling import (
    BASIN_SHIFT_TOL,
    PI_B_LEVELS,
    run_grid,
    simulate,
)
from src.agents.affect_math import (
    MIN_PRECISION,
    MOOD_DRIVE,
    MOOD_INERTIA,
    W_MAX_BEHAVIOR,
)

_VARIED = [((t * 0.6180339887) % 1.0) * 1.6 - 0.8 for t in range(500)]


@pytest.fixture(scope="module")
def grid_report() -> dict[str, object]:
    """全网格只跑一次，各判据共享（运行预算见 test_runtime_budget）。"""
    return run_grid()


# ── 判据 1：无新增锁定 ────────────────────────────────────────────────────────


def test_no_new_lockins(grid_report: dict[str, object]) -> None:
    """基线（π_b=0）不锁定的配置，开行为流后不得新增锁定（|m|≥0.95 驻留 ≥K=10 且
    撤刺激不退出）。mood 自身双稳是特性；判据只看**新增**。"""
    assert grid_report["lock_violations"] == []


# ── 判据 2：盆边界位移有界 ────────────────────────────────────────────────────


def test_basin_boundary_shift_within_tol(grid_report: dict[str, object]) -> None:
    """零输入与恒偏置两组、其余流三档：开流前后吸引盆分界点位移 ≤ 0.03（3 个扫描步，
    防把网格分辨率误判为位移）。"""
    shifts = grid_report["boundary_shifts"]
    assert isinstance(shifts, dict)
    assert shifts, "盆边界扫描不得为空"
    for key, shift in shifts.items():
        assert shift <= BASIN_SHIFT_TOL, f"{key}: 位移 {shift} 超阈 {BASIN_SHIFT_TOL}"


# ── 判据 3：w_b 后置封顶不变量（全网格实测）──────────────────────────────────


def test_w_b_capped_across_grid(grid_report: dict[str, object]) -> None:
    w_b_max = grid_report["w_b_max"]
    assert isinstance(w_b_max, float)
    assert 0.0 < w_b_max <= W_MAX_BEHAVIOR + 1e-9


def test_runtime_budget(grid_report: dict[str, object]) -> None:
    """CS 席回归锁：网格膨胀静默拖垮 CI 时此处先红（实测 ~5s，预算留一倍余量）。"""
    elapsed = grid_report["elapsed_s"]
    assert isinstance(elapsed, float)
    assert elapsed < 10.0


def test_magnitude_check_documented() -> None:
    """解析式降格后的量级检验（议会必改 #1；非严格证明——严格结论在上面三条判据）：
    一次往返环增益量级 W_MAX·drive 应远小于 1−inertia。"""
    assert W_MAX_BEHAVIOR * MOOD_DRIVE < 0.25 * (1.0 - MOOD_INERTIA)


# ── 变异三类（绿灯先证能红）──────────────────────────────────────────────────


def test_mutation_cap_disabled_breaks_weight_invariant() -> None:
    """变异①：绕过后置封顶 → 全沉默环境下 w_b 失控（数学席实测的最坏情形），
    判据 3 的断言应声变红。"""
    res = simulate(
        _VARIED,
        pi_b=PI_B_LEVELS["legacy"],
        pi_other=MIN_PRECISION,
        pi_m=MIN_PRECISION,  # mood 也沉默——真·全沉默才是 π_b 失控的情形
        kappa=0.5,
        cap_enabled=False,
    )
    assert res.w_b_max > 0.9  # 未封顶 w_b→0.97：正确实现下不可能出现
    capped = simulate(
        _VARIED,
        pi_b=PI_B_LEVELS["legacy"],
        pi_other=MIN_PRECISION,
        pi_m=MIN_PRECISION,
        kappa=0.5,
        cap_enabled=True,
    )
    assert capped.w_b_max <= W_MAX_BEHAVIOR + 1e-9  # 同配置封顶后回到界内（剔除→0 亦满足）


def test_mutation_delta_as_mu_is_category_error() -> None:
    """变异②：把位移量 δ 当位置 μ 喂（design §1.2 判定的范畴错误）。恒正输入 +
    suppression（κ=0.5）下 δ<0——变异体把「压低了」误宣称为「唤醒为负」，
    系统性把 arousal 拉到显著低于正确实现。"""
    gated = simulate([0.8] * 300, pi_b=PI_B_LEVELS["legacy"], pi_other=0.4, kappa=0.5)
    mutant = simulate(
        [0.8] * 300,
        pi_b=PI_B_LEVELS["legacy"],
        pi_other=0.4,
        kappa=0.5,
        evidence_mode="delta_as_mu",
    )
    assert gated.a_traj[-1] > 0.0
    assert mutant.a_traj[-1] < gated.a_traj[-1] - 0.05


def test_mutation_always_present_breaks_default_zero_regression() -> None:
    """变异③：拆在场门恒在场。κ=1（生产默认 regulation 关）时正确实现 = 流缺席 =
    与基线逐位相同；变异体注入 mood 已携带信息的重复（double counting），轨迹偏离。"""
    baseline = simulate([0.6] * 200, pi_b=0.0, pi_other=0.4, kappa=1.0)
    gated = simulate([0.6] * 200, pi_b=PI_B_LEVELS["legacy"], pi_other=0.4, kappa=1.0)
    assert gated.a_traj == baseline.a_traj  # 正确实现：门开但流缺席=零回归
    mutant = simulate(
        [0.6] * 200,
        pi_b=PI_B_LEVELS["legacy"],
        pi_other=0.4,
        kappa=1.0,
        evidence_mode="always",
    )
    assert mutant.a_traj != baseline.a_traj  # 变异体：double counting 可测


# ── σ 校准三口径（议会必改 #4）────────────────────────────────────────────────


@pytest.fixture(scope="module")
def calibration_report() -> dict[str, float]:
    return calibrate()


def test_sigma_bhv_is_conservative_upper_bound(calibration_report: dict[str, float]) -> None:
    """① P95 本身作保守上界（刻意高估 σ）：SIGMA_BHV 须 ≥ 开环候选。
    ⚠ 模型内代理只证「未低估」这一必要条件，充分校准待真实日志（脚本 docstring）。"""
    assert calibration_report["sigma_bhv_current"] >= calibration_report["open_loop_p95"]
    assert calibration_report["open_loop_p95"] > 0.1  # 校准数据非退化（varied regime 生效）


def test_closed_loop_consistency(calibration_report: dict[str, float]) -> None:
    """② 自指校准检验：开环统计的 σ 接回闭环后 |Δa| P95 漂移须有界，否则开环标定不可迁移。"""
    assert calibration_report["closed_shift"] < 0.1


def test_regulation_offset_measured_separately(calibration_report: dict[str, float]) -> None:
    """③ 调节偏移是回合内量、|Δa| 覆盖不到——单独口径测得非零值（分量分开报告）。"""
    assert calibration_report["regulation_offset_p95"] > 0.0
