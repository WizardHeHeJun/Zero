"""盲测 A/B 载荷：当前实现 vs 某一项改动前的状态，随机排序，答案单独落 `ab_key.json`。

议会要求：「真正的验收锚点是用户最初抱怨的『不自然』，是主观感知问题，
统计距离缩小不保证观感变好」——故必须盲法。

⚠ 数学席：单次 2AFC 只是 n=1 伯努利观测。要多轮 + 二项检验（如 ≥9/10 为通过线）。
**每轮换一个 `--seed`** 即可得到独立试次（噪声场、姿态目标序列、甲乙先后全都变）；
每轮播完记下用户的选择，跑完再一次性对答案。

用法：

    python gen_ab.py                       # 默认变体，默认种子
    python gen_ab.py --variant coupling    # 指定对比哪一项
    python gen_ab.py --variant coupling --seed 7  # 换种子 = 换一个独立试次
    python gen_ab.py --list                # 看有哪些变体
"""

from __future__ import annotations

import argparse
import json
import zlib
from collections.abc import Callable

import _paths as P

P.use_zero()

import src.agents.motion_synth as ms  # noqa: E402

DEFAULT_SEED = 860613
# 取契约上限 10s（TRAJECTORY_MAX_SEGMENT_MS=10000，超了对面会 rejected）。
# 8 秒只够眨 2 次，判断眨眼频率样本太少；10 秒能眨 2~3 次，且播放脚本会连播两遍。
DURATION_MS = 10000.0
AFFECT = (-0.3, 0.45)  # 中等唤醒，两版差异最能看出来


def _restore_axis_ratio() -> Callable[[], None]:
    """旧版 = 三轴等幅（2026-08-06 那轮改动前）。"""
    original = ms.AXIS_AMPLITUDE_RATIO
    ms.AXIS_AMPLITUDE_RATIO = (1.0, 1.0, 1.0)
    return lambda: setattr(ms, "AXIS_AMPLITUDE_RATIO", original)


def _restore_coupling() -> Callable[[], None]:
    """旧版 = 改动前的**原式** `roll = 0.6·roll_own + (−0.125)·yaw`。

    ⚠ 必须复现原式本身，不能只把 r 换成旧的隐含值 −0.74：原式除了相关更强，还把 roll
    幅度压到三轴比例的 91%，两项差异都要在盲测里出现，否则测的不是"改动前 vs 改动后"。

    改后 r=−0.45 取自免泄漏重测（三数据集一致，见
    `notes/2026-08-07-motion-idle-constants-criteria.md`）。两版**符号相同**（都是对侧）
    ——本轮验的是量级：符号若反会一眼看出、不需要盲测。
    """
    original = ms._pose_cycle

    def old_form(t: float, seed: int, mod: ms.Modulation) -> tuple[list[float], list[float], float]:
        pose, eye, frac = original(t, seed, mod)
        # 反解出耦合前的两个分量，再按原式重组
        coupled = ms.YAW_ROLL_COUPLING * ms.AXIS_AMPLITUDE_RATIO[2] * pose[0]
        own = (pose[2] - coupled) / (1.0 - ms.YAW_ROLL_COUPLING**2) ** 0.5
        pose[2] = 0.6 * own + (-0.125) * pose[0]
        return pose, eye, frac

    ms._pose_cycle = old_form  # type: ignore[assignment]
    return lambda: setattr(ms, "_pose_cycle", original)


def _restore_sway_hz() -> Callable[[], None]:
    """旧版 = 借用的 `SWAY_HZ=0.07`（周期 14.3s）。

    改后 0.04（周期 25s）取自同域实测：漂移带主峰在两个数据集上**边界稳定**
    （0.048 / 0.035Hz），与呼吸带那项恰好相反 —— 后者的"峰"会跟着频带边界跑，故被否决。
    """
    original = ms.SWAY_HZ
    ms.SWAY_HZ = 0.07
    return lambda: setattr(ms, "SWAY_HZ", original)


def _restore_kinematics() -> Callable[[], None]:
    """旧版 = 2026-08-07 运动学校准前：微颤**叠加式** 0.12 + 转移 0.45s。

    ⚠ **必须精确复现原式**，不能只把 ratio 设回 0.12：当前实现已改为**混合式**
    `gain*((1-r)*pose + r*micro)`，而原式是**叠加式** `gain*(pose + r*micro)`，
    后者包络是前者的 (1+r) 倍。只改 ratio 会得到一个幅度偏小的假"旧版"，
    测出来的差异里混进本不属于本项改动的幅度差（耦合那一版已踩过同样的坑）。

    代数等价：`gain*(pose + 0.12*micro)` = `gain*1.12*((1-r')*pose + r'*micro)`，
    其中 r' = 0.12/1.12。故同时把 ratio 设为 r'、把幅度乘 1.12。

    改后：微颤混合式 0.20 + 转移 0.60s，靶子是真人的**角速度分布**（无阈值形状量）。
    改前高尾过冲 p90/p95/p99 = 1.93/2.32/2.23 倍真人（=「过快」），低尾又太静（=「僵硬」）；
    改后降到 1.06/1.06/0.95。代价：同 clamp 安全线下 yaw sd 从 11.5° 降到 8.9°
    ——**幅度换分布形状**，这一版就是拿这两样让眼睛选。
    """
    original_ratio = ms.MICRO_TREMOR_RATIO
    original_rise = ms.POSE_RISE_S
    additive = 0.12
    ms.MICRO_TREMOR_RATIO = additive / (1.0 + additive)
    ms.POSE_RISE_S = 0.45
    _OVERRIDES["amplitude_scale"] = ms.DEFAULT_AMPLITUDE_SCALE * (1.0 + additive)

    def undo() -> None:
        ms.MICRO_TREMOR_RATIO = original_ratio
        ms.POSE_RISE_S = original_rise
        _OVERRIDES["amplitude_scale"] = None

    return undo


VARIANTS: dict[str, tuple[str, Callable[[], Callable[[], None]]]] = {
    "axis_ratio": ("三轴幅度比（旧＝三轴等幅）", _restore_axis_ratio),
    "coupling": ("yaw-roll 耦合（旧＝原式 0.6·roll + (−0.125)·yaw）", _restore_coupling),
    "kinematics": ("运动学校准（旧＝叠加微颤 0.12 + 转移 0.45s）", _restore_kinematics),
    "sway_hz": ("低频漂移频率（旧＝借用值 0.07Hz，新＝实测 0.04Hz）", _restore_sway_hz),
}


# 变体可覆盖的合成参数。⚠ 不能靠改 `ms.DEFAULT_AMPLITUDE_SCALE` 模块全局——
# `generate_dual` 的默认形参在**函数定义时**就绑定了那个值，改全局不生效（Python 语义）。
_OVERRIDES: dict[str, float | None] = {"amplitude_scale": None}


def build(seed: int) -> list[dict]:
    phase = ms.PhaseState(noise_seed=seed, next_blink_ms=ms.initial_blink_ms(seed))
    scale = _OVERRIDES["amplitude_scale"]
    heads, _ = ms.generate_dual(
        AFFECT,
        None,
        DURATION_MS,
        phase,
        amplitude_scale=scale if scale is not None else ms.DEFAULT_AMPLITUDE_SCALE,
    )
    return heads["voluntary"]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--variant", default="coupling", choices=sorted(VARIANTS))
parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="换种子 = 换一个独立试次")
parser.add_argument("--list", action="store_true", help="列出可用变体后退出")
args = parser.parse_args()

if args.list:
    for name, (desc, _) in sorted(VARIANTS.items()):
        print(f"  {name:12s} {desc}")
    raise SystemExit(0)

description, mutate = VARIANTS[args.variant]

new_frames = build(args.seed)
undo = mutate()
old_frames = build(args.seed)
undo()

# 确定性"随机"排序：由种子与变体名共同决定谁先播，主程也不预告顺序
first_is_new = bool(zlib.crc32(f"{args.seed}:{args.variant}".encode()) % 2)
ordered = (
    [("新", new_frames), ("旧", old_frames)]
    if first_is_new
    else [("旧", old_frames), ("新", new_frames)]
)

payload = {
    "segments": [
        {"label": label, "keyframes": frames, "duration_s": DURATION_MS / 1000.0}
        for label, (_, frames) in zip(("甲", "乙"), ordered, strict=True)
    ]
}
with open(P.AB_PAYLOAD, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, ensure_ascii=False)
with open(P.AB_KEY, "w", encoding="utf-8") as fh:
    json.dump(
        {"甲": ordered[0][0], "乙": ordered[1][0], "variant": args.variant, "seed": args.seed},
        fh,
        ensure_ascii=False,
    )

print(f"对比项：{description}")
print(f"种子 {args.seed} · 甲/乙 各 {len(new_frames)} 帧、各 {DURATION_MS / 1000:.0f} 秒")
print(f"载荷 → {P.AB_PAYLOAD}")
print("答案已单独写入 ab_key.json（播放脚本不读它）")
