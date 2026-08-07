"""从 StayStill 待机数据提取合成器各程序化常数的同域实测值（**概览用**）。

🛑 **本脚本只给"数据长什么样"的概览，不作为定值依据**——2026-08-07 起，
每个常数该不该改由 `coupling_measure.py` / `idle_criteria.py` 的**判据**决定：

| 本脚本会打印的 | 判据结论 |
| --- | --- |
| 三轴幅度比 | ✅ 已采纳（1 : 0.33 : 0.19），留出集复现（sd 比 0.144~0.224） |
| yaw-roll 耦合 +0.415 | ❌ **几何伪影**，已被消融对照臂证伪 → 用 `coupling_measure.py` 的 −0.45 |
| 呼吸带主峰 0.178Hz | ❌ **取带边界的产物**，非真峰 → `BREATH_HZ` 保留文献值 0.27 |
| 转移/驻留时长 | ❌ 绝对秒数不可采信（无阈值平台）→ 只有"占空比约为合成器 2.7 倍"可用 |

角度提取自 2026-08-07 起统一走 `anatomy.py`（坐标系实测判定 + 免泄漏 roll）。

⚠ 分割参数沿用数学席的处方：平滑 + 迟滞双阈 + 最短段过滤，且阈值用**峰值百分比**
（生物力学惯例）而非分布分位数（后者会按构造抹平占空比，已在 RAVDESS 上栽过）。
"""

import statistics
from collections import defaultdict
from pathlib import Path

import _paths as P  # 转正后统一取路径（原为 scratchpad 绝对路径）
import numpy as np
from anatomy import detect_anatomy, head_angles, parse_bvh

ROOT = Path(P.STAYSTILL / "idle")
FPS = 30.0


def series(path):
    """逐帧头部 yaw/pitch/roll（度）——走 `anatomy.py`：坐标系由骨架实测判定、roll 免泄漏。

    ⚠ 旧版在此手写提取，把关节局部 +Y 当"朝前"（实际朝后）、roll 用 atan2(up_x, up_z)
    （pitch≠0 时混入 yaw）。两处错都不驱红，只会让下游常数悄悄错掉。
    """
    skeleton = parse_bvh(Path(path))
    if len(skeleton.frames) == 0:
        return None
    angles = head_angles(skeleton, detect_anatomy(skeleton))
    return angles["yaw"], angles["pitch"], angles["roll"]


files = sorted(ROOT.glob("*.bvh"))
print(f"待机片段 {len(files)} 条\n")

amp = defaultdict(list)
moves, dwells = [], []
peak_freqs = defaultdict(list)
coupling = []

for f in files:
    s = series(f)
    if s is None or len(s[0]) < 300:
        continue
    yaw, pitch, roll = s
    for k, v in (("yaw", yaw), ("pitch", pitch), ("roll", roll)):
        amp[k].append(float(np.std(v)))

    # ① 频谱主峰（对应呼吸/漂移带）：对去趋势后的 pitch 做 FFT
    sig = pitch - np.mean(pitch)
    n = len(sig)
    spec = np.abs(np.fft.rfft(sig * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1 / FPS)
    for band, (lo, hi) in (
        ("低频漂移 0.02-0.15Hz", (0.02, 0.15)),
        ("呼吸带 0.15-0.5Hz", (0.15, 0.5)),
    ):
        sel = (freqs >= lo) & (freqs <= hi)
        if sel.any():
            peak_freqs[band].append(float(freqs[sel][np.argmax(spec[sel])]))

    # ② 运动/驻留分割（峰值百分比阈 + 迟滞 + 最短段）
    speed = np.sqrt(np.diff(yaw) ** 2 + np.diff(pitch) ** 2 + np.diff(roll) ** 2) * FPS
    k = 5
    sm = np.convolve(speed, np.ones(k) / k, mode="same")
    pk = np.percentile(sm, 95)
    hi_thr, lo_thr = pk * 0.20, pk * 0.08
    moving = sm[0] > hi_thr
    start = 0
    for i in range(1, len(sm)):
        now = (sm[i] > hi_thr) if not moving else (sm[i] > lo_thr)
        if now != moving:
            dur = (i - start) / FPS
            if dur >= 0.15:
                (moves if moving else dwells).append(dur)
                start = i
            moving = now

    # ③ yaw-roll 耦合
    if np.std(yaw) > 1e-6 and np.std(roll) > 1e-6:
        coupling.append(float(np.corrcoef(yaw, roll)[0, 1]))

print("① 三轴幅度（sd，度）")
for k in ("yaw", "pitch", "roll"):
    v = sorted(amp[k])
    q1, q3 = v[len(v) // 4], v[3 * len(v) // 4]
    print(f"   {k:6s} 中位={statistics.median(v):6.2f}  四分位 {q1:.2f}~{q3:.2f}")
base = statistics.median(amp["yaw"])
print(
    f"   ⇒ 相对 yaw 的比例： yaw 1.00 · pitch {statistics.median(amp['pitch']) / base:.2f} "
    f"· roll {statistics.median(amp['roll']) / base:.2f}"
)

print("\n② 频谱主峰（Hz）—— 对应呼吸带与低频漂移带")
for band, v in peak_freqs.items():
    print(
        f"   {band:22s} 中位={statistics.median(v):.3f}Hz  (周期 {1 / statistics.median(v):.1f}s)"
    )

print(f"\n③ 运动/驻留分割（n_move={len(moves)}, n_dwell={len(dwells)}）")
print(
    f"   转移时长 中位={statistics.median(moves):.3f}s  四分位 "
    f"{np.percentile(moves, 25):.3f}~{np.percentile(moves, 75):.3f}s"
)
print(
    f"   驻留时长 中位={statistics.median(dwells):.3f}s  四分位 "
    f"{np.percentile(dwells, 25):.3f}~{np.percentile(dwells, 75):.3f}s"
)
print(f"   ⇒ 周期（转移+驻留）中位 ≈ {statistics.median(moves) + statistics.median(dwells):.2f}s")

print(f"\n④ yaw-roll 相关（n={len(coupling)}）")
print(f"   中位={statistics.median(coupling):+.3f}  均值={statistics.fmean(coupling):+.3f}")

print("\n" + "=" * 64)
print("对照当前 motion_synth.py 的值 —— ⚠ 差异不等于该改，见本文件顶部的判据结论表")
print("=" * 64)
print(
    f"  三轴幅度      当前 14.0° 等幅        → 实测 {statistics.median(amp['yaw']):.1f}/"
    f"{statistics.median(amp['pitch']):.1f}/{statistics.median(amp['roll']):.1f}（不等幅）"
)
breath = statistics.median(peak_freqs["呼吸带 0.15-0.5Hz"])
sway = statistics.median(peak_freqs["低频漂移 0.02-0.15Hz"])
print(f"  BREATH_HZ     当前 0.27              → 带内主峰 {breath:.3f} ❌ 边界产物，不改")
print(f"  SWAY_HZ       当前 0.07              → 实测 {sway:.3f}")
print(f"  POSE_RISE_S   当前 0.45              → 分割 {statistics.median(moves):.3f} ❌ 无阈值平台")
cycle = statistics.median(moves) + statistics.median(dwells)
print(f"  POSE_CYCLE_S  当前 2.4               → 分割 {cycle:.2f} ❌ 同上，绝对值不可采信")
print(
    f"  YAW_ROLL_COUPLING 当前 -0.45         → 本脚本原序列相关 {statistics.median(coupling):+.3f}"
)
print("     （原序列相关含慢漂移；定值用的是免泄漏**增量**相关 −0.458，见 coupling_measure.py）")
