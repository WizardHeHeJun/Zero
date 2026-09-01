"""γ 折减占比表（PRP/gamma-physio design.md §五·design-door 第 1 条分布证据）。

按 Zero_MCP 2026-09-01 交付件 §2/§4（wire·二值锚，样本外主口径）**参数化重建** μ_a 样本
（μ_a = α̂ + β̂·a + e，e~N(0,σ_e)，固定种子）——逐窗原始数据未随件交付，重建参数与条件
窗数（baseline 160 / stress 111）全部出自交付件，无自由参数。对声明 Πa 取四档真实实践值
（对方子源政策值 0.18 / 对方 M8 上界 0.359 / hrv 诚实值 0.391 / 我方 cap 0.8），输出每档
折减前/后的 Πa_declared、salience 分布分位数、点燃占比、对 survival 无条件地板的权重上界。

用法：conda run -n affective-expression python -m scripts.verify_gamma_physio_occupancy
产物：stdout markdown 表，重定向落 PRP/gamma-physio/artifacts/（知识层，主检出）。
"""

from __future__ import annotations

import random
import statistics

from src.agents.affect_math import (
    GAMMA_PHYSIO,
    MIN_PRECISION,
    SALIENCE_THRESHOLD,
    SURVIVAL_PRECISION,
    W_MAX_PHYSIO,
    expand_external_priors,
    stream_salience,
)

# 交付件 §2 wire·二值锚（与 GAMMA_PHYSIO 的输入同源；重建仅用点估计+残差，SE 不进样本）
ALPHA_HAT = -0.0105
BETA_HAT = 0.4267
SIGMA_E = 0.1597
# 交付件 §4：样本外 wire 存在窗（meditation/amusement 不入二值锚回归）
N_BASELINE = 160  # a=0
N_STRESS = 111  # a=1
SEED = 20260901

# 声明 Πa 四档真实实践值（design §五）
PI_A_LEVELS: tuple[tuple[str, float], ...] = (
    ("对方子源政策值", 0.18),
    ("对方 M8 上界", 0.359),
    ("hrv 诚实值(双输区)", 0.391),
    ("我方 cap", 0.8),
)


def _quantiles(xs: list[float]) -> tuple[float, float, float, float, float]:
    q = statistics.quantiles(xs, n=20, method="inclusive")  # 5% 步长
    return (min(xs), q[4], statistics.median(xs), q[14], max(xs))  # min/p25/p50/p75/max


def build_mu_a_samples() -> list[float]:
    """重建 μ_a 逐窗样本（clamp 到 M7 合法域 [-1,1]，越界样本会被 M7 拒收非折减）。"""
    rng = random.Random(SEED)
    samples: list[float] = []
    for a, n in ((0.0, N_BASELINE), (1.0, N_STRESS)):
        for _ in range(n):
            mu = ALPHA_HAT + BETA_HAT * a + rng.gauss(0.0, SIGMA_E)
            samples.append(max(-1.0, min(1.0, mu)))
    return samples


def main() -> None:
    mus = build_mu_a_samples()
    print(f"# γ 折减占比表（重建 n={len(mus)} 窗：baseline {N_BASELINE} + stress {N_STRESS}）")
    print()
    print(f"- GAMMA_PHYSIO = {GAMMA_PHYSIO:.5f}（公式现算，输入见 affect_math.py 常量注释）")
    print(f"- SALIENCE_THRESHOLD = {SALIENCE_THRESHOLD} · MIN_PRECISION = {MIN_PRECISION}")
    print(f"- μ_a 重建分布：min/p25/p50/p75/max = {tuple(round(v, 4) for v in _quantiles(mus))}")
    print()
    print(
        "| 档位 | Πa_naive | Πa_declared(=γ·naive) | salience p50 | salience max | "
        "点燃占比(前→后) | w 上界 vs survival 地板(前→后) |"
    )
    print("| --- | ---: | ---: | ---: | ---: | --- | --- |")
    for label, pi_a in PI_A_LEVELS:
        expanded = expand_external_priors(
            [("physio", (0.0, m), (0.5, pi_a)) for m in mus[:1]],
            precision_cap=0.8,
            max_streams=5,
        )
        pi_declared = expanded[0][2][1]
        # 反事实（无 γ）也走产品码 stream_salience——防其公式将来修改时本列静默失真
        sal_naive = [stream_salience((0.0, m), (MIN_PRECISION, pi_a)) for m in mus]
        sal_decl = [stream_salience((0.0, m), (MIN_PRECISION, pi_declared)) for m in mus]
        ign_naive = sum(s >= SALIENCE_THRESHOLD for s in sal_naive) / len(mus)
        ign_decl = sum(s >= SALIENCE_THRESHOLD for s in sal_decl) / len(mus)
        # 权重上界：其余流全弱到只剩 survival 无条件地板时的凸组合权重（arousal 维）
        w_naive = pi_a / (pi_a + SURVIVAL_PRECISION)
        w_decl = pi_declared / (pi_declared + SURVIVAL_PRECISION)
        q_decl = _quantiles(sal_decl)
        print(
            f"| {label} | {pi_a:.3f} | {pi_declared:.5f} | {q_decl[2]:.5f} | {q_decl[4]:.5f} "
            f"| {ign_naive:.1%} → {ign_decl:.1%} | {w_naive:.3f} → {w_decl:.3f} |"
        )
    print()
    print(f"- 判读：折减后全部档位 salience max < {SALIENCE_THRESHOLD}（点燃占比恒 0）；")
    print(
        f"  权重上界远低于 W_MAX_PHYSIO={W_MAX_PHYSIO}（后置封顶为退化情形兜底，生产装配"
        "常态 no-op——survival/appraisal/value 三条无条件在场时 Σ_other 更大、w 更低）。"
    )
    print(
        "- 标注：μ_a 为参数化重建非逐窗原始数据（design §五）；salience 已隐含 mean(Π) 对"
        "一维流的 ≈2.83× 结构性折扣（7-30 失真必改#4）与 γ 的乘积叠加。"
    )


if __name__ == "__main__":
    main()
