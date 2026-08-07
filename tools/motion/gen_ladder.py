"""幅度梯度载荷：一次播若干档递增幅度，让眼睛直接挑一个档位。

**为什么不是 A/B**：「幅度该多大」是**量级判断**，不是"哪个更自然"的二选一。
二选一每轮只给 1 bit，挑量级要挑很多轮；递增梯度一次就能定位（心理物理学的极限法）。
故本脚本**如实按序递增**、不打乱——这里没有"哪版是新的"这种会污染判断的信息，
不需要盲法（盲法是为了防偏好偏置，不是仪式）。

🛑 **clamp 是硬上限**：皮套角度参数 ±`ANGLE_RANGE_DEG`（30°），超了会被削平顶部
（= 波形失真 + 满幅摇头，2026-08-06 踩过）。

⚠ **安全性必须按 `arousal=1.0` 判，不是按你观看时的唤醒档**（2026-08-07 本脚本第一版
就是按 0.8 判的，漏放了一个在激动时会削平的档位）。判据对齐守卫
`test_clamp_never_actually_fires`：a=1.0 下**零触顶**且**峰值 < 0.9×30=27°**。

不安全的档**照样生成、但如实标注**（`safety` 字段 + 播放时报出），不偷偷剔除——
你要看就得看得到，但不能不知情地选中它。

跑：
    python gen_ladder.py                    # 默认梯度
    python gen_ladder.py --arousal 0.8      # 换唤醒档位看（高唤醒最容易触顶）
"""

from __future__ import annotations

import argparse
import json

import _paths as P

P.use_zero()

import src.agents.motion_synth as ms  # noqa: E402

DURATION_MS = 10000.0
SEED = 20260807
LABELS = ("一", "二", "三", "四", "五")
# 相对**当前默认**的倍数，刻意跨过默认值：0.74 = 2026-08-07 调大前的旧值，1.0 = 当前默认，
# 往上到 clamp 前的余量。上界由脚本对每档实测 clamp 率把关，不靠这里写死。
FACTORS = (0.74, 1.0, 1.15, 1.25)


# 特征值（sd / clamp 率）用**长跑多种子**测，别用 10 秒载荷本身量：
# 10 秒只含约 4 个姿态周期，sd 会系统性偏低（实测 4.88° vs 长跑 7.37°），
# clamp 那种低频事件更是根本抽不到——这与本轮定耦合常数时踩的是同一个坑。
PROFILE_SECONDS = 120.0
PROFILE_SEEDS = (11, 31, 77, 101, 233)


def profile(scale: float, arousal: float) -> tuple[float, float, float]:
    """长跑多种子测该 scale 的真实 yaw sd、峰值、clamp 触发率。"""
    sds: list[float] = []
    peak = 0.0
    clamped = 0
    total = 0
    for seed in PROFILE_SEEDS:
        frames, _ = ms.generate(
            0.0,
            arousal,
            PROFILE_SECONDS * 1000.0,
            ms.PhaseState(noise_seed=seed),
            amplitude_scale=scale,
        )
        values = [float(f["params"][ms.PARAM_ANGLE_X]) for f in frames]
        mean = sum(values) / len(values)
        sds.append((sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5)
        for frame in frames:
            for key in (ms.PARAM_ANGLE_X, ms.PARAM_ANGLE_Y, ms.PARAM_ANGLE_Z):
                total += 1
                value = abs(float(frame["params"][key]))
                peak = max(peak, value)
                if value >= ms.ANGLE_RANGE_DEG - 1e-6:
                    clamped += 1
    return sum(sds) / len(sds), peak, clamped / max(total, 1)


def build(scale: float, arousal: float) -> list[dict]:
    """产出实际要播的那 10 秒载荷。"""
    phase = ms.PhaseState(noise_seed=SEED, next_blink_ms=ms.initial_blink_ms(SEED))
    heads, _ = ms.generate_dual((-0.3, arousal), None, DURATION_MS, phase, amplitude_scale=scale)
    return heads["voluntary"]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arousal", type=float, default=0.45, help="唤醒档位（高唤醒最容易触顶）")
args = parser.parse_args()

segments = []
default_rung = LABELS[FACTORS.index(1.0)] if 1.0 in FACTORS else "（不在梯度内）"
print(
    f"唤醒 {args.arousal:+.2f} · 每档 {DURATION_MS / 1000:.0f} 秒 · "
    f"当前默认 = 第 {default_rung} 档（倍数 1.00）\n"
)
# 🛑 安全性按 **arousal=1.0** 判（不是按观看档位），阈值对齐守卫：零触顶 且 峰值 < 0.9×30。
SAFETY_LIMIT = ms.ANGLE_RANGE_DEG * 0.9
print(
    f"{'档':>4} {'倍数':>6} {'基幅等效':>9} {'yaw sd':>9} "
    f"{'a=1.0 峰值':>11} {'a=1.0 触顶':>11}  安全性"
)
for label, factor in zip(LABELS, FACTORS, strict=False):
    scale = ms.DEFAULT_AMPLITUDE_SCALE * factor
    sd, _, _ = profile(scale, args.arousal)
    _, peak_max, clamp_rate = profile(scale, 1.0)
    if clamp_rate > 0:
        safety, note = "clips", "❌ 激动时削平波形"
    elif peak_max >= SAFETY_LIMIT:
        safety, note = (
            "no_margin",
            f"⚠ 不削平但 margin 归零（吃满 {peak_max / ms.ANGLE_RANGE_DEG:.0%} 量程）",
        )
    else:
        safety, note = "ok", "✅"
    print(
        f"{label:>4} {factor:>6.2f} {ms.NOISE_AMPLITUDE_DEG * factor:>9.1f} {sd:>8.2f}° "
        f"{peak_max:>10.2f}° {clamp_rate:>10.2%}  {note}"
    )
    segments.append(
        {
            "label": label,
            "keyframes": build(scale, args.arousal),
            "duration_s": DURATION_MS / 1000.0,
            "factor": factor,
            "safety": safety,
        }
    )

with open(P.AB_PAYLOAD, "w", encoding="utf-8") as fh:
    json.dump({"segments": segments}, fh, ensure_ascii=False)
print(f"\n{len(segments)} 档载荷 → {P.AB_PAYLOAD}")
print("参照：同域真人待机 yaw sd = 21.3°（但那是「在街上等人」可随意张望，非对话场景）")
print("播放：cd d:\\Zero_MCP 后跑 play_ladder.py")
