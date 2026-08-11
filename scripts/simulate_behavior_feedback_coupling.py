"""行为反馈流 × mood 双稳环耦合稳定性仿真（行为反馈环第二步·T3 证明层）。

「先证明后实现」的证明主体（议会必改 #1/#3/#8，纪要
notes/2026-08-07-behavior-feedback-council.md）：解析线性化已被数学席判失真降格——
正确的偏导 `∂m/∂a = MOOD_DRIVE = 0.2`，一次往返环增益量级 ≈ `w_m·w_b·drive`，应远小于
`1−inertia = 0.4`——**此式仅作量级检验、非严格证明**（双稳 + clamp 双区域非线性系统
不受单一线性不等式刻画）；严格结论由本脚本的数值判据给出。

实现约束（CS 席）：**只走 `affect_math` 纯函数层**（`mood_step` / `fuse_terms` /
`cap_stream_weight`），禁经图 ainvoke；无 RNG、无 wall-clock，输入序列全确定 ⇒ 逐位可复现。

建模口径：按**生效组合**（`gate_fusion=False` 的全流原生精度加权语义）建模——默认硬门下
行为流恒被 salience 滤除（design.md §三 ⚠），耦合风险只存在于生效组合，仿真最坏情形即它。
arousal 维标量系统（valence 维行为流恒 MIN_PRECISION，不参与耦合）：

    a_t   = fuse([其余流(u_t, π_u), mood(m_{t-1}, π_m), behavior(a_expr_{t-1}, π_b)])
    m_t   = mood_step(m_{t-1}, a_t)          # 双稳：0.6m + 0.5tanh(2m) + 0.2a
    a_expr_t = κ·a_t                          # κ=1 无调节（最坏耦合）；κ<1 suppression 压幅

变异模式（tests 用，绿灯先证能红）：
    evidence_mode="gated"（正确·κ=1 时流缺席=absent-cue）/ "always"（变异：拆在场门恒在场）
    / "delta_as_mu"（变异：把位移量 δ 当位置 μ 喂——design §1.2 判定的范畴错误）；
    cap_enabled=False（变异：绕过 w_b 后置封顶）。
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

from src.agents.affect_math import (
    MIN_PRECISION,
    MOOD_DRIVE,
    behavior_precision,
    cap_stream_weight,
    fuse_terms,
    mood_step,
)

# ── 网格（design §2.3 四维）───────────────────────────────────────────────────

# 维 1：输入 regime（u_t 序列生成器；全确定）。
# "varied" 用黄金分割低差异序列（无 RNG 的逐轮变化）——zero/constant/alternating 大段恒定，
# 回合间 |Δa| 会塌到 0，σ 校准（calibrate_sigma_bhv）需要有真实量级的漂移分布。
INPUT_REGIMES: dict[str, list[float]] = {
    "zero": [0.0] * 500,
    "constant": [0.8] * 500,
    "alternating": [0.8 if (t // 25) % 2 == 0 else -0.8 for t in range(500)],
    "varied": [((t * 0.6180339887) % 1.0) * 1.6 - 0.8 for t in range(500)],
}
# 维 2：π_b 档（0=基线关；legacy 档；齐次档）
PI_B_LEVELS: dict[str, float] = {
    "off": 0.0,
    "legacy": behavior_precision(),
    "commensurable": behavior_precision(commensurable=True),
}
# 维 3：mood 参数（drive, self_gain）——默认组 + 加深组（灵敏性）
MOOD_PARAMS: list[tuple[float, float]] = [(0.2, 0.5), (0.3, 0.6)]
# 维 4：其余流集体强度（数学席点名补维：π_b 失控恰在全沉默处）
OTHER_STREAMS: dict[str, float] = {
    "all_silent": MIN_PRECISION,  # 全沉默：其余流触底
    "survival_only": 0.4,  # 仅 survival 地板（SURVIVAL_PRECISION）
    "all_ignited": 8.4,  # 全点燃：appraisal(≈8) + survival(0.4) 量级
}

LOCK_LEVEL = 0.95  # 锁定电平（clamp 边界驻留判定）
LOCK_K = 10  # 锁定驻留回合数（议会钉死 K=10）
SETTLE_TURNS = 100  # 撤刺激后的观察窗（判「不退出」）
BASIN_STEP = 0.01  # 盆边界 m0 扫描步长（议会钉死）
BASIN_SHIFT_TOL = 0.03  # 盆边界位移阈 = 3 个扫描步（议会钉死，防把分辨率当位移）


@dataclass
class SimResult:
    """单配置仿真轨迹与观测量。"""

    a_traj: list[float] = field(default_factory=list)
    m_traj: list[float] = field(default_factory=list)
    w_b_max: float = 0.0  # 本次仿真实际观测到的行为流最大权重（封顶不变量的证据）
    locked: bool = False  # 锁定 = 驻留 ≥K 且撤刺激后仍不退出


def _behavior_term(
    a_expr_prev: float,
    a_felt_prev: float,
    pi_b: float,
    evidence_mode: str,
    kappa: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """按模式产出行为流项；None=流缺席。gated 模式复刻 behavior_feedback_evidence 的
    在场门语义（κ=1 ⇔ voluntary 与 spontaneous 相等 ⇔ 第一步副本 voluntary=None）。"""
    if pi_b <= 0.0:
        return None
    if evidence_mode == "gated":
        if kappa == 1.0:
            return None  # δ≡0 ⇒ absent-cue（正确实现）
        return ((0.0, a_expr_prev), (MIN_PRECISION, pi_b))
    if evidence_mode == "always":
        return ((0.0, a_expr_prev), (MIN_PRECISION, pi_b))  # 变异：拆在场门
    if evidence_mode == "delta_as_mu":
        delta = a_expr_prev - a_felt_prev
        return ((0.0, delta), (MIN_PRECISION, pi_b))  # 变异：位移当位置（范畴错误）
    raise ValueError(f"未知 evidence_mode={evidence_mode!r}")


def simulate(
    inputs: list[float],
    *,
    pi_b: float,
    pi_other: float,
    drive: float = MOOD_DRIVE,
    self_gain: float = 0.5,
    pi_m: float = 1.0,
    kappa: float = 1.0,
    m0: float = 0.0,
    evidence_mode: str = "gated",
    cap_enabled: bool = True,
    settle_turns: int = SETTLE_TURNS,
) -> SimResult:
    """跑一条合系统轨迹：len(inputs) 回合带输入 + settle_turns 回合零输入（判退出）。"""
    result = SimResult()
    a_prev = 0.0
    m_prev = m0
    a_expr_prev = 0.0
    a_felt_prev = 0.0
    for u in [*inputs, *([0.0] * settle_turns)]:
        terms: list[tuple[tuple[float, float], tuple[float, float]]] = [
            ((0.0, u), (pi_other, pi_other)),
            ((0.0, m_prev), (pi_m, pi_m)),
        ]
        names = ["others", "mood"]
        bterm = _behavior_term(a_expr_prev, a_felt_prev, pi_b, evidence_mode, kappa)
        if bterm is not None:
            terms.append(bterm)
            names.append("behavior")
        if cap_enabled:
            terms, names = cap_stream_weight(terms, names, target="behavior")
        if "behavior" in names:
            idx = names.index("behavior")
            total = sum(max(MIN_PRECISION, prec[1]) for _, prec in terms)
            result.w_b_max = max(result.w_b_max, max(MIN_PRECISION, terms[idx][1][1]) / total)
        (_, a_t), _ = fuse_terms(terms)
        m_t = mood_step((0.0, m_prev), (0.0, a_t), self_gain=self_gain, drive=drive)[1]
        a_felt_prev = a_t
        a_expr_prev = kappa * a_t
        a_prev = a_t
        result.a_traj.append(a_t)
        result.m_traj.append(m_t)
        m_prev = m_t
    del a_prev
    # 锁定判定：输入期内驻留 ≥K，且撤刺激观察窗末仍在电平之上（不退出）
    n_in = len(inputs)
    in_phase = result.m_traj[:n_in]
    run = best = 0
    for m in in_phase:
        run = run + 1 if abs(m) >= LOCK_LEVEL else 0
        best = max(best, run)
    tail = result.m_traj[-LOCK_K:]
    result.locked = best >= LOCK_K and all(abs(m) >= LOCK_LEVEL for m in tail)
    return result


def basin_boundary(
    *,
    pi_b: float,
    pi_other: float,
    kappa: float = 0.5,
    evidence_mode: str = "gated",
    cap_enabled: bool = True,
    bias: float = 0.0,
    turns: int = 500,
) -> float:
    """零输入（或恒偏置 bias）下扫 m0 ∈ [-1,1]（步长 BASIN_STEP）定吸引盆分界点。

    返回「终态落正盆的最小 m0」；对称零输入下应≈0，位移与基线比较（阈 BASIN_SHIFT_TOL）。
    """
    boundary = 1.0 + BASIN_STEP  # 全负盆时的哨兵（不在扫描域内）
    steps = int(round(2.0 / BASIN_STEP)) + 1
    for i in range(steps):
        m0 = -1.0 + i * BASIN_STEP
        res = simulate(
            [bias] * turns,
            pi_b=pi_b,
            pi_other=pi_other,
            kappa=kappa,
            m0=m0,
            evidence_mode=evidence_mode,
            cap_enabled=cap_enabled,
            settle_turns=0,
        )
        if res.m_traj[-1] > 0.0:
            boundary = m0
            break
    return boundary


def run_grid() -> dict[str, object]:
    """全网格扫描（design §2.3 四维），返回汇总报告 dict（tests 消费）。"""
    started = time.perf_counter()
    lock_violations: list[str] = []
    w_b_overall = 0.0
    for regime, inputs in INPUT_REGIMES.items():
        for other_name, pi_other in OTHER_STREAMS.items():
            for drive, self_gain in MOOD_PARAMS:
                for kappa in (1.0, 0.5):  # 无调节（最坏耦合门关）/ suppression 压半
                    base = simulate(
                        inputs,
                        pi_b=0.0,
                        pi_other=pi_other,
                        drive=drive,
                        self_gain=self_gain,
                        kappa=kappa,
                    )
                    for level_name, pi_b in PI_B_LEVELS.items():
                        if pi_b == 0.0:
                            continue
                        res = simulate(
                            inputs,
                            pi_b=pi_b,
                            pi_other=pi_other,
                            drive=drive,
                            self_gain=self_gain,
                            kappa=kappa,
                        )
                        w_b_overall = max(w_b_overall, res.w_b_max)
                        if res.locked and not base.locked:
                            lock_violations.append(
                                f"{regime}/{other_name}/drive={drive}/sg={self_gain}"
                                f"/kappa={kappa}/{level_name}"
                            )
    # 盆边界：零输入 + 恒偏置两组，基线 vs 开流（suppression κ=0.5 才有流在场）
    boundary_shifts: dict[str, float] = {}
    for bias_name, bias in (("zero_input", 0.0), ("biased_input", 0.3)):
        for other_name, pi_other in OTHER_STREAMS.items():
            b0 = basin_boundary(pi_b=0.0, pi_other=pi_other, bias=bias)
            b1 = basin_boundary(pi_b=PI_B_LEVELS["commensurable"], pi_other=pi_other, bias=bias)
            boundary_shifts[f"{bias_name}/{other_name}"] = abs(b1 - b0)
    elapsed = time.perf_counter() - started
    return {
        "lock_violations": lock_violations,
        "w_b_max": w_b_overall,
        "boundary_shifts": boundary_shifts,
        "elapsed_s": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    report = run_grid()
    print(f"耗时 {report['elapsed_s']:.2f}s")
    print(f"w_b 实测最大值：{report['w_b_max']:.4f}")
    print(f"新增锁定配置：{report['lock_violations'] or '无'}")
    print("盆边界位移：")
    shifts = report["boundary_shifts"]
    assert isinstance(shifts, dict)
    for key, shift in shifts.items():
        print(f"  {key}: {shift:.4f}")


if __name__ == "__main__":
    main()
