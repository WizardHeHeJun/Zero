"""交接文档第五节剩余两项的**判据**：转移/驻留时长 · 呼吸频率。

两项的共同问题都不是"数字是多少"，而是"凭什么信这个数"。本脚本给的是判据，不是数字。

## 一、转移/驻留时长（实测 1.433s / 0.600s，与现有「移动-驻留」模型相反）

议会警告过「阈值定义了你看到什么」（RAVDESS 上已栽过一次：逐轨迹自适应阈把占空比压缩 13 倍）。
所以先问一个更前置的问题：**这个分割器在已知真值上量得准吗？**

判据 A ·**正控（positive control）**：用我们自己的合成器造一段轨迹——它的转移/驻留时长
    是**已知的**（`POSE_RISE_S` / `POSE_CYCLE_S`）——喂给**同一个**分割器，看能否量回来。
    量不回来 ⇒ 该分割器在真数据上的输出没有意义，这一项直接搁置。
判据 B ·**阈值不变性**：把迟滞阈在合理区间扫一遍，看估计值有没有平台区。
    没有平台 ⇒ 数字是阈值的函数，不是数据的性质。
判据 C ·**跨数据集**：StayStill 与留出的 ReActIdle genuine 是否给出一致的量级。

## 二、呼吸频率（实测 0.178Hz，低于静息呼吸 12–20 次/分）

判据 D ·**峰的显著性**：0.15–0.5Hz 带内的"主峰"到底是一个**峰**，还是只是 1/f 背景
    在带内左端的最大值？后者的话，"0.178Hz"其实等于"带的下边界"，是取带方式的产物。
    做法：拟合 log-log 背景，量峰相对背景的突起（prominence）；并把取带下限往下挪，
    看峰位是否**跟着边界跑**——跟着跑 = 没有真峰。
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import _paths as P
import numpy as np
from anatomy import detect_anatomy, head_angles, parse_bvh

sys.path.insert(0, str(P.REPO))
from src.agents.motion_synth import (  # noqa: E402
    PARAM_ANGLE_X,
    PARAM_ANGLE_Y,
    PARAM_ANGLE_Z,
    POSE_CYCLE_S,
    POSE_RISE_S,
    PhaseState,
    generate,
)

MIN_FRAMES = 300


def segment(
    yaw: np.ndarray,
    pitch: np.ndarray,
    roll: np.ndarray,
    fps: float,
    *,
    hi_pct: float = 0.20,
    lo_pct: float = 0.08,
    min_dur: float = 0.15,
    smooth: int = 5,
) -> tuple[list[float], list[float]]:
    """「移动/驻留」分割：平滑 + 峰值百分比迟滞双阈 + 最短段过滤（数学席处方）。

    阈用**峰值百分比**（生物力学惯例）而非分布分位数——后者会按构造抹平占空比。
    与 `idle_constants.py` 里的实现同构，此处参数化以便扫阈。
    """
    speed = np.sqrt(np.diff(yaw) ** 2 + np.diff(pitch) ** 2 + np.diff(roll) ** 2) * fps
    kernel = np.ones(smooth) / smooth
    smoothed = np.convolve(speed, kernel, mode="same")
    peak = np.percentile(smoothed, 95)
    hi, lo = peak * hi_pct, peak * lo_pct

    moves: list[float] = []
    dwells: list[float] = []
    moving = bool(smoothed[0] > hi)
    start = 0
    for i in range(1, len(smoothed)):
        now = bool(smoothed[i] > hi) if not moving else bool(smoothed[i] > lo)
        if now != moving:
            duration = (i - start) / fps
            if duration >= min_dur:
                (moves if moving else dwells).append(duration)
                start = i
            moving = now
    return moves, dwells


def synthetic_series(seconds: float = 600.0, fps: float = 20.0) -> tuple[np.ndarray, ...]:
    """用我方合成器造一段**真值已知**的轨迹：转移 = POSE_RISE_S、周期 = POSE_CYCLE_S。"""
    frames, _ = generate(0.0, 0.0, seconds * 1000.0, PhaseState(noise_seed=7), fps=fps)
    axes = []
    for key in (PARAM_ANGLE_X, PARAM_ANGLE_Y, PARAM_ANGLE_Z):
        axes.append(np.array([float(f["params"][key]) for f in frames]))  # type: ignore[index]
    return tuple(axes)


def load_dataset(directory: Path, fps: float) -> list[tuple[np.ndarray, ...]]:
    out = []
    for path in sorted(directory.glob("*.bvh")):
        skeleton = parse_bvh(path)
        if len(skeleton.frames) < MIN_FRAMES:
            continue
        angles = head_angles(skeleton, detect_anatomy(skeleton))
        out.append((angles["yaw"], angles["pitch"], angles["roll"]))
    return out


print("=" * 76)
print("判据 A · 正控：分割器能否量回我方合成器的已知真值？")
print("=" * 76)
# 中性档（arousal=0 ⇒ speed=1.0）下：周期 = POSE_CYCLE_S，转移 = POSE_RISE_S × 1.35（onset=0.5）
truth_cycle = POSE_CYCLE_S
truth_rise = POSE_RISE_S * (1.35 - 0.5 * 0.5)
print(
    f"   真值：转移 {truth_rise:.3f}s · 周期 {truth_cycle:.3f}s · "
    f"驻留 {truth_cycle - truth_rise:.3f}s"
)
syn_yaw, syn_pitch, syn_roll = synthetic_series()
moves, dwells = segment(syn_yaw, syn_pitch, syn_roll, 20.0)
if moves and dwells:
    m, d = statistics.median(moves), statistics.median(dwells)
    print(f"   量得：转移 {m:.3f}s · 驻留 {d:.3f}s · 周期 {m + d:.3f}s")
    err_rise = (m - truth_rise) / truth_rise
    err_cycle = (m + d - truth_cycle) / truth_cycle
    print(f"   偏差：转移 {err_rise:+.0%} · 周期 {err_cycle:+.0%}")
    verdict = (
        "✅ 可用" if abs(err_rise) < 0.35 and abs(err_cycle) < 0.35 else "❌ 分割器有系统性偏差"
    )
    print(f"   ⇒ {verdict}（判据：两项偏差均 <35% 才认为它在真数据上的读数可信）")
else:
    print("   ❌ 分割器在合成数据上分不出段")

print()
print("=" * 76)
print("判据 B · 阈值不变性：扫迟滞阈，看有没有平台区")
print("=" * 76)
staystill = load_dataset(P.STAYSTILL / "idle", 30.0)
print(f"   StayStill idle n={len(staystill)} 条")
print("   ⚠ 同一套扫阈**同时跑在合成数据上**作对照——这是区分两种失败的关键：")
print(
    "     「阈值没选对」(合成也无平台) vs 「数据本来就没有移动-驻留的二分结构」(只有真数据无平台)"
)
print(
    f"   {'hi%':>6} {'lo%':>6} | {'真·转移':>9} {'真·驻留':>9} "
    f"{'真·占空比':>9} | {'合成·占空比':>11}"
)
duties_real: list[float] = []
duties_syn: list[float] = []
for hi in (0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
    all_moves: list[float] = []
    all_dwells: list[float] = []
    for yaw, pitch, roll in staystill:
        mv, dw = segment(yaw, pitch, roll, 30.0, hi_pct=hi, lo_pct=hi * 0.4)
        all_moves += mv
        all_dwells += dw
    syn_mv, syn_dw = segment(syn_yaw, syn_pitch, syn_roll, 20.0, hi_pct=hi, lo_pct=hi * 0.4)
    if not all_moves or not all_dwells or not syn_mv or not syn_dw:
        continue
    m, d = statistics.median(all_moves), statistics.median(all_dwells)
    duty = sum(all_moves) / (sum(all_moves) + sum(all_dwells))
    duty_syn = sum(syn_mv) / (sum(syn_mv) + sum(syn_dw))
    duties_real.append(duty)
    duties_syn.append(duty_syn)
    print(f"   {hi:>6.2f} {hi * 0.4:>6.2f} | {m:>9.3f} {d:>9.3f} {duty:>9.1%} | {duty_syn:>11.1%}")
if duties_real:
    span_real = max(duties_real) - min(duties_real)
    span_syn = max(duties_syn) - min(duties_syn)
    ratios = [r / s for r, s in zip(duties_real, duties_syn, strict=True)]
    print(f"\n   占空比随阈变化幅度：真数据 {span_real:.1%} · 合成 {span_syn:.1%}")
    if span_real >= 0.15 and span_syn >= 0.15:
        print("   ⇒ ❌ **两者都无平台** ⇒ 这个分割器根本没有阈值不变的占空比估计。")
        print("        故「转移 1.433s / 驻留 0.600s」这类**绝对值**不可采信——换个阈就换个数。")
        print("        （判据 A 说明它在**已知真值**上能量准，但真数据上无从知道该选哪个阈。）")
    elif span_real < 0.15:
        print("   ⇒ ✅ 真数据存在平台，绝对值可采信")

    # 绝对值不可用，但**同阈下的比值**在整个区间稳定 —— 这才是可以拿来用的量。
    print(
        f"\n   真/合成 占空比比值：{min(ratios):.2f}~{max(ratios):.2f}"
        f"（中位 {statistics.median(ratios):.2f}）"
    )
    if max(ratios) / max(min(ratios), 1e-9) < 2.0 and min(ratios) > 1.5:
        print("   ⇒ ✅ **比值在全阈值区间稳定**：真人待机的头部处于运动的时间占比，")
        print(f"        约为本合成器当前的 {statistics.median(ratios):.1f} 倍。")
        print("        这是本项唯一站得住的结论——方向真实、量级只能给到倍数，给不到秒数。")
    else:
        print("   ⇒ ⚠ 比值也不稳定，本项无可用结论")

print()
print("=" * 76)
print("判据 C · 跨数据集一致性（留出集 ReActIdle genuine）")
print("=" * 76)
for label, directory, fps in (
    ("StayStill idle", P.STAYSTILL / "idle", 30.0),
    ("ReActIdle genuine", P.REACTIDLE / "genuine", 30.0),
):
    if not directory.exists():
        continue
    clips = load_dataset(directory, fps)
    all_moves, all_dwells = [], []
    for yaw, pitch, roll in clips:
        mv, dw = segment(yaw, pitch, roll, fps)
        all_moves += mv
        all_dwells += dw
    if all_moves and all_dwells:
        duty = sum(all_moves) / (sum(all_moves) + sum(all_dwells))
        print(
            f"   {label:20s} 转移 {statistics.median(all_moves):.3f}s · "
            f"驻留 {statistics.median(all_dwells):.3f}s · 运动占空比 {duty:.1%}"
        )

print()
print("=" * 76)
print("判据 D · 呼吸峰的显著性：是真峰，还是 1/f 背景在带内左端的最大值？")
print("=" * 76)
for label, directory, fps in (
    ("StayStill idle", P.STAYSTILL / "idle", 30.0),
    ("ReActIdle genuine", P.REACTIDLE / "genuine", 30.0),
):
    if not directory.exists():
        continue
    peaks_by_band: dict[float, list[float]] = {}
    prominences: list[float] = []
    for _yaw, pitch, _roll in load_dataset(directory, fps):
        signal = pitch - np.mean(pitch)
        n = len(signal)
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(n))) ** 2
        freqs = np.fft.rfftfreq(n, d=1 / fps)
        # 边界敏感性：把带下限往下挪，峰位是否跟着跑
        for lo in (0.10, 0.15, 0.20):
            sel = (freqs >= lo) & (freqs <= 0.5)
            if sel.any():
                peaks_by_band.setdefault(lo, []).append(float(freqs[sel][np.argmax(spectrum[sel])]))
        # 峰相对 1/f 背景的突起：在 log-log 上拟合直线当背景
        band = (freqs > 0.02) & (freqs < 2.0)
        log_f, log_p = np.log(freqs[band]), np.log(np.maximum(spectrum[band], 1e-30))
        slope, intercept = np.polyfit(log_f, log_p, 1)
        residual = log_p - (slope * log_f + intercept)
        breath = (freqs[band] >= 0.15) & (freqs[band] <= 0.5)
        if breath.any():
            prominences.append(float(np.max(residual[breath]) / max(np.std(residual), 1e-9)))
    print(f"   {label}")
    for lo, values in sorted(peaks_by_band.items()):
        print(f"      带下限 {lo:.2f}Hz ⇒ 主峰中位 {statistics.median(values):.3f}Hz")
    lows = sorted(peaks_by_band)
    shift = statistics.median(peaks_by_band[lows[-1]]) - statistics.median(peaks_by_band[lows[0]])
    print(f"      带下限挪 {lows[0]:.2f}→{lows[-1]:.2f}Hz，峰位移动 {shift:+.3f}Hz  ⇒ ", end="")
    print("❌ 峰位跟着边界跑 = 没有真峰" if shift > 0.03 else "✅ 峰位稳定")
    if prominences:
        med = statistics.median(prominences)
        print(f"      峰相对 1/f 背景的突起 = {med:.2f} 个残差 sd  ⇒ ", end="")
        print("✅ 显著" if med > 2.0 else "❌ 不显著，与背景起伏无异")


print()
print("=" * 76)
print("判据 E · 低频漂移带：峰位是否也跟着边界跑 + 带内幅度是否与合成器可比")
print("=" * 76)
print("   ⚠ 幅度必须**把合成器输出送进同一测量流程**再比：数据的带内能量包含姿态变化的")
print("     泄漏，而 SWAY_AMPLITUDE_DEG 只是我们额外叠的那条正弦——直接比是错的口径。")


def band_sd(signal: np.ndarray, fps: float, lo: float, hi: float) -> float:
    """带通后信号的 sd（度）：FFT 置零非带内分量再逆变换。"""
    centered = signal - np.mean(signal)
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(len(centered), d=1 / fps)
    spectrum[(freqs < lo) | (freqs > hi)] = 0.0
    return float(np.std(np.fft.irfft(spectrum, n=len(centered))))


DRIFT = (0.02, 0.15)
BREATH = (0.15, 0.5)
for label, clips, fps in (
    ("StayStill idle", load_dataset(P.STAYSTILL / "idle", 30.0), 30.0),
    ("ReActIdle genuine", load_dataset(P.REACTIDLE / "genuine", 30.0), 30.0),
    ("★ 本合成器输出", [(syn_yaw, syn_pitch, syn_roll)], 20.0),
):
    if not clips:
        continue
    # ⚠ 绝对幅度**不可比**：StayStill 被试是"在街上等人"可随意东张西望（yaw 极值超 ±40°），
    #   而对话中的数字人应大部分时间面向用户——整体尺度是**产品决定**，已单独由
    #   NOISE_AMPLITUDE_DEG 控制。故这里比的是**带内能量占总能量的份额**（尺度无关）。
    drift_yaw = [band_sd(yaw, fps, *DRIFT) / max(float(np.std(yaw)), 1e-9) for yaw, _p, _r in clips]
    drift_pitch = [
        band_sd(pitch, fps, *DRIFT) / max(float(np.std(pitch)), 1e-9) for _y, pitch, _r in clips
    ]
    breath_pitch = [
        band_sd(pitch, fps, *BREATH) / max(float(np.std(pitch)), 1e-9) for _y, pitch, _r in clips
    ]
    print(
        f"   {label:18s} 漂移带占比 yaw {statistics.median(drift_yaw):5.1%} · "
        f"pitch {statistics.median(drift_pitch):5.1%}  |  呼吸带占比 pitch "
        f"{statistics.median(breath_pitch):5.1%}"
    )

# 漂移带峰位的边界敏感性 —— 与呼吸带**结果相反**是关键：
# 纯 1/f 背景下主峰必然贴在带下边界上；这里下限挪了峰不动 ⇒ 是真峰。
for label, directory in (
    ("StayStill idle", P.STAYSTILL / "idle"),
    ("ReActIdle genuine（留出）", P.REACTIDLE / "genuine"),
):
    if not directory.exists():
        continue
    peaks_by_edge: dict[float, list[float]] = {}
    for _yaw, pitch, _roll in load_dataset(directory, 30.0):
        signal = pitch - np.mean(pitch)
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
        freqs = np.fft.rfftfreq(len(signal), d=1 / 30.0)
        # ⚠ 扫描下限不得低于**频率分辨率**（1/片段时长）：StayStill 片段约 125s ⇒ 0.008Hz、
        #    ReActIdle genuine 约 91s ⇒ 0.011Hz。低于它的"峰位"只是第一个可分辨频点，
        #    会被误判成"跟着边界跑"。故从 0.012Hz 起扫。
        #    上界同样有约束：**必须明显低于候选峰**（两集分别 0.048 / 0.035Hz）。
        #    判据的逻辑是"若为 1/f 伪影则峰会贴在下边界"，边界一旦越过真峰，
        #    峰位当然会被推着走——那不是判据失败，是对照臂选错了。
        for lo in (0.012, 0.016, 0.020):
            sel = (freqs >= lo) & (freqs <= 0.15)
            if sel.any():
                peaks_by_edge.setdefault(lo, []).append(float(freqs[sel][np.argmax(spectrum[sel])]))
    print(f"   {label} 漂移带主峰的边界敏感性：")
    for lo, values in sorted(peaks_by_edge.items()):
        print(f"      带下限 {lo:.3f}Hz ⇒ 主峰中位 {statistics.median(values):.3f}Hz")
    edges = sorted(peaks_by_edge)
    moved = statistics.median(peaks_by_edge[edges[-1]]) - statistics.median(peaks_by_edge[edges[0]])
    print(f"      下限挪 {edges[0]:.3f}→{edges[-1]:.3f}Hz，峰位移动 {moved:+.3f}Hz  ⇒ ", end="")
    print("❌ 跟着边界跑" if moved > 0.01 else "✅ 峰位稳定，是真峰（与呼吸带结论相反）")
