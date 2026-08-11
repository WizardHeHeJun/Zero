"""σ_bhv 校准脚本（行为反馈环第二步·T3；议会必改 #4 的三处统计口径落实）。

产出 `affect_math.SIGMA_BHV` 的实测依据（当前值 1.3 为保守占位，本脚本核验/回填）：

① **分位数 ≠ σ**：取回合间 |Δa| 分布的 **95 分位本身**作 σ 的**保守上界**（非无偏估计——
   若 Δa ~ N(0,σ)，P95(|Δa|) ≈ 1.96σ，直接当 σ 用会高估 σ、低估精度，方向安全，刻意为之）。
② **自指校准检验**：开环（π_b=0）统计出的 σ 用于闭环后，|Δa| 分布会变——闭环重测做
   一致性检验（P95 漂移应有界），否则开环标定不可迁移。
③ **调节偏移单独测**：|a_expr − a_felt| 是回合内量，|Δa|（跨回合漂移）覆盖不到，
   分开报告两个分量——当前 σ 候选只覆盖漂移分量，如实标注。

数据源说明：本脚本用耦合仿真（simulate_behavior_feedback_coupling 的确定性输入网格，
含 "varied" 低差异序列提供有量级的逐轮漂移）作 |Δa| 来源——**模型内代理**：它证明的是
SIGMA_BHV 未低估（必要条件），不构成充分校准；将来有真实对话日志时按同口径重测、
以日志为准。纯函数层、确定性、无 RNG。
"""

from __future__ import annotations

import argparse

from scripts.simulate_behavior_feedback_coupling import (
    INPUT_REGIMES,
    OTHER_STREAMS,
    PI_B_LEVELS,
    simulate,
)
from src.agents.affect_math import SIGMA_BHV


def _percentile(sorted_values: list[float], q: float) -> float:
    """最近秩法分位数（无插值，保守取整到上界侧）。"""
    if not sorted_values:
        return 0.0
    rank = min(len(sorted_values) - 1, max(0, int(q * len(sorted_values) + 0.5) - 1))
    return sorted_values[rank]


def _abs_deltas(*, pi_b: float, kappa: float) -> list[float]:
    """跨全网格收集回合间 |Δa|（跳过前 10 回合暂态）。"""
    deltas: list[float] = []
    for inputs in INPUT_REGIMES.values():
        for pi_other in OTHER_STREAMS.values():
            res = simulate(inputs, pi_b=pi_b, pi_other=pi_other, kappa=kappa, settle_turns=0)
            traj = res.a_traj[10:]
            deltas.extend(abs(b - a) for a, b in zip(traj, traj[1:], strict=False))
    return sorted(deltas)


def _regulation_offsets(kappa: float) -> list[float]:
    """回合内调节偏移 |a_expr − a_felt| = (1−κ)·|a|（分量③，单独口径）。"""
    offsets: list[float] = []
    for inputs in INPUT_REGIMES.values():
        for pi_other in OTHER_STREAMS.values():
            res = simulate(inputs, pi_b=0.0, pi_other=pi_other, kappa=kappa, settle_turns=0)
            offsets.extend((1.0 - kappa) * abs(a) for a in res.a_traj[10:])
    return sorted(offsets)


def calibrate() -> dict[str, float]:
    """跑三段校准，返回观测量 dict（tests 消费）。"""
    open_loop = _abs_deltas(pi_b=0.0, kappa=1.0)
    sigma_candidate = _percentile(open_loop, 0.95)  # ① P95 本身 = 保守上界
    closed_loop = _abs_deltas(pi_b=PI_B_LEVELS["commensurable"], kappa=0.5)
    closed_p95 = _percentile(closed_loop, 0.95)  # ② 闭环一致性
    reg_offsets = _regulation_offsets(0.5)
    offset_p95 = _percentile(reg_offsets, 0.95)  # ③ 调节偏移分量
    return {
        "open_loop_p95": sigma_candidate,
        "closed_loop_p95": closed_p95,
        "closed_shift": abs(closed_p95 - sigma_candidate),
        "regulation_offset_p95": offset_p95,
        "sigma_bhv_current": SIGMA_BHV,
        "sigma_conservative": float(SIGMA_BHV >= sigma_candidate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    report = calibrate()
    print(f"① 开环 |Δa| P95（σ 保守上界候选）：{report['open_loop_p95']:.4f}")
    print(f"② 闭环 |Δa| P95：{report['closed_loop_p95']:.4f}（漂移 {report['closed_shift']:.4f}）")
    print(f"③ 调节偏移 |a_expr−a_felt| P95（独立分量）：{report['regulation_offset_p95']:.4f}")
    print(
        f"当前 SIGMA_BHV={report['sigma_bhv_current']} "
        f"{'≥ 候选（保守成立）' if report['sigma_conservative'] else '< 候选（须上调！）'}"
    )


if __name__ == "__main__":
    main()
