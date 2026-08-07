"""真人 vs 本合成器的**角速度分布**对比：把「僵硬」「过快」量成数字。

**为什么用速度分布而不是移动/驻留分割**：分割要定阈，而阈值定义了你看到什么
（`idle_criteria.py` 判据 B 已实证：真数据与合成数据**都不存在阈值平台**）。
角速度分布是**无阈值**的——直接比两条分布，形状和位置都看得见：

- 「**过快**」= 我们的分布相对真人整体右移（高速档占比更大）。
- 「**僵硬**」= 我们的分布**双峰**（驻留期贴近 0 + 转移期一个高速峰），
  而真人若是单峰连续，就说明真实头动没有"定住—猛动"这种二分。

⚠ 两侧必须**同帧率**再比：速度是差分算的，30fps 与 20fps 的差分尺度不同。
统一重采样到 20fps（合成器的输出帧率）后再差分。

跑：`python kinematics_gap.py`
"""

from __future__ import annotations

import statistics
import sys

import _paths as P
import numpy as np
from anatomy import detect_anatomy, head_angles, parse_bvh

sys.path.insert(0, str(P.REPO))
from src.agents.motion_synth import (  # noqa: E402
    PARAM_ANGLE_X,
    PARAM_ANGLE_Y,
    PARAM_ANGLE_Z,
    PhaseState,
    generate,
)

TARGET_FPS = 20.0
MIN_FRAMES = 300
PERCENTILES = (10, 25, 50, 75, 90, 99)


def resample(series: np.ndarray, source_fps: float) -> np.ndarray:
    """线性重采样到 TARGET_FPS——不同帧率下的差分不可直接比。"""
    if abs(source_fps - TARGET_FPS) < 1e-9:
        return series
    duration = len(series) / source_fps
    count = int(duration * TARGET_FPS)
    return np.interp(np.arange(count) / TARGET_FPS, np.arange(len(series)) / source_fps, series)


def angular_speed(yaw: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """合成角速度（度/秒）。"""
    return np.sqrt(np.diff(yaw) ** 2 + np.diff(pitch) ** 2 + np.diff(roll) ** 2) * TARGET_FPS


def real_speeds() -> np.ndarray:
    out: list[np.ndarray] = []
    for path in sorted((P.STAYSTILL / "idle").glob("*.bvh")):
        skeleton = parse_bvh(path)
        if len(skeleton.frames) < MIN_FRAMES:
            continue
        angles = head_angles(skeleton, detect_anatomy(skeleton))
        axes = [resample(angles[k], 30.0) for k in ("yaw", "pitch", "roll")]
        out.append(angular_speed(*axes))
    return np.concatenate(out)


def synth_speeds(arousal: float) -> np.ndarray:
    out: list[np.ndarray] = []
    for seed in (11, 31, 77, 101, 233):
        frames, _ = generate(0.0, arousal, 120_000.0, PhaseState(noise_seed=seed), fps=TARGET_FPS)
        axes = [
            np.array([float(f["params"][key]) for f in frames])
            for key in (PARAM_ANGLE_X, PARAM_ANGLE_Y, PARAM_ANGLE_Z)
        ]
        out.append(angular_speed(*axes))
    return np.concatenate(out)


def describe(label: str, speeds: np.ndarray) -> dict[str, float]:
    values = {f"p{p}": float(np.percentile(speeds, p)) for p in PERCENTILES}
    print(
        f"   {label:22s} "
        + "  ".join(f"{k}={v:6.2f}" for k, v in values.items())
        + f"   均值={float(np.mean(speeds)):6.2f}"
    )
    return values


print("=" * 96)
print("① 角速度分布（度/秒，统一 20fps 后差分）—— 位置右移 = 更快")
print("=" * 96)
real = real_speeds()
stats_real = describe("真人 StayStill 待机", real)
stats_synth: dict[float, dict[str, float]] = {}
for arousal in (-0.5, 0.0, 0.45):
    stats_synth[arousal] = describe(f"本合成器 a={arousal:+.2f}", synth_speeds(arousal))

print("\n   ⇒ 与真人的倍数（>1 = 我们更快）")
for arousal, values in stats_synth.items():
    ratios = "  ".join(
        f"p{p}={values[f'p{p}'] / max(stats_real[f'p{p}'], 1e-9):5.2f}×" for p in PERCENTILES
    )
    print(f"   a={arousal:+.2f}  {ratios}")

print()
print("=" * 96)
print("② 分布形状：真人是单峰连续，还是和我们一样「贴 0 + 高速峰」的双峰？")
print("=" * 96)
print("   判据：把速度按其自身中位数归一后看直方图形状。")
print("   双峰（大量样本贴近 0 + 另一簇在高位）= 定住—猛动的二分结构 = 观感上的「僵硬」。")


def shape(label: str, speeds: np.ndarray) -> None:
    normalized = speeds / max(float(np.median(speeds)), 1e-9)
    near_zero = float(np.mean(normalized < 0.2))
    mid = float(np.mean((normalized >= 0.2) & (normalized < 1.5)))
    high = float(np.mean(normalized >= 1.5))
    print(
        f"   {label:22s} 近零(<0.2×中位) {near_zero:6.1%} · 中段 {mid:6.1%} · "
        f"高速(≥1.5×) {high:6.1%}"
    )


shape("真人 StayStill 待机", real)
for arousal in (-0.5, 0.0, 0.45):
    shape(f"本合成器 a={arousal:+.2f}", synth_speeds(arousal))

print()
print("=" * 96)
print("③ 结论用量")
print("=" * 96)
median_ratio = stats_synth[0.45]["p50"] / max(stats_real["p50"], 1e-9)
p90_ratio = stats_synth[0.45]["p90"] / max(stats_real["p90"], 1e-9)
print(f"   中等唤醒下：中位速度是真人的 {median_ratio:.2f}×，p90 是 {p90_ratio:.2f}×")
print(f"   真人速度中位 {statistics.median(real.tolist()):.2f}°/s")
