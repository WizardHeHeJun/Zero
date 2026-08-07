"""从 StayStill 待机数据提取合成器全部程序化常数的同域实测值。

对应 motion_synth.py 里当前的手写/借用文献值：
  NOISE_AMPLITUDE_DEG=14（三轴等幅）· BREATH_HZ=0.27 · SWAY_HZ=0.07
  BREATH_AMPLITUDE_DEG=0.2 · SWAY_AMPLITUDE_DEG=0.6
  POSE_CYCLE_S=2.4 · POSE_RISE_S=0.45 · YAW_ROLL_COUPLING=-0.125

⚠ 分割参数沿用数学席的处方：平滑 + 迟滞双阈 + 最短段过滤，且阈值用**峰值百分比**
（生物力学惯例）而非分布分位数（后者会按构造抹平占空比，已在 RAVDESS 上栽过）。
"""

import math
import statistics
from collections import defaultdict
from pathlib import Path

import _paths as P  # 转正后统一取路径（原为 scratchpad 绝对路径）
import numpy as np

ROOT = Path(P.STAYSTILL / "idle")
FPS = 30.0


def parse_bvh(path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    joints, channels, cur = [], {}, None
    for i, raw in enumerate(lines):
        s = raw.strip()
        if s.startswith(("ROOT ", "JOINT ")):
            cur = s.split(None, 1)[1]
            joints.append(cur)
        elif s.startswith("CHANNELS") and cur:
            channels[cur] = s.split()[2:]
        elif s.startswith("MOTION"):
            n = int(lines[i + 1].split(":")[1])
            return (
                joints,
                channels,
                [[float(v) for v in ln.split()] for ln in lines[i + 3 : i + 3 + n] if ln.strip()],
            )
    return joints, channels, []


def col(j, c, joint, ch):
    idx = 0
    for jj in j:
        for cc in c.get(jj, []):
            if jj == joint and cc == ch:
                return idx
            idx += 1
    return None


def rot(a, d):
    r = math.radians(d)
    co, si = math.cos(r), math.sin(r)
    return (
        [[1, 0, 0], [0, co, -si], [0, si, co]]
        if a == "X"
        else [[co, 0, si], [0, 1, 0], [-si, 0, co]]
        if a == "Y"
        else [[co, -si, 0], [si, co, 0], [0, 0, 1]]
    )


def mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def jm(j, c, joint, row):
    m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for ch in c.get(joint, []):
        if ch.endswith("rotation"):
            m = mm(m, rot(ch[0], row[col(j, c, joint, ch)]))
    return m


def series(path):
    j, c, data = parse_bvh(path)
    if not data:
        return None
    ys, ps, rs = [], [], []
    for row in data:
        m = mm(jm(j, c, "neck", row), jm(j, c, "face", row))
        axis = [m[0][2], m[1][2], m[2][2]]
        fwd = [m[0][1], m[1][1], m[2][1]]
        ys.append(math.degrees(math.atan2(fwd[0], fwd[1])))
        ps.append(math.degrees(math.asin(max(-1.0, min(1.0, fwd[2])))))
        rs.append(math.degrees(math.atan2(axis[0], axis[2])))
    unwrap = lambda v: np.unwrap(np.radians(v)) * 180 / math.pi  # noqa: E731
    return unwrap(ys), np.array(ps), unwrap(rs)


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
print("对照当前 motion_synth.py 手写值")
print("=" * 64)
print(
    f"  三轴幅度      当前 14.0° 等幅        → 实测 {statistics.median(amp['yaw']):.1f}/"
    f"{statistics.median(amp['pitch']):.1f}/{statistics.median(amp['roll']):.1f}（不等幅）"
)
breath = statistics.median(peak_freqs["呼吸带 0.15-0.5Hz"])
sway = statistics.median(peak_freqs["低频漂移 0.02-0.15Hz"])
print(f"  BREATH_HZ     当前 0.27              → 实测 {breath:.3f}")
print(f"  SWAY_HZ       当前 0.07              → 实测 {sway:.3f}")
print(f"  POSE_RISE_S   当前 0.45              → 实测 {statistics.median(moves):.3f}")
cycle = statistics.median(moves) + statistics.median(dwells)
print(f"  POSE_CYCLE_S  当前 2.4               → 实测 {cycle:.2f}")
print(f"  YAW_ROLL_COUPLING 当前 -0.125        → 实测 {statistics.median(coupling):+.3f}")
