"""自采管线的**正控**：用已知常数合成数据 → 走完整管线 → 看能否量回原值。

🛑 **采数据之前先跑这个。** 这一程最有用的一步就是它：三个"同域实测"的常数里有两个是
测量伪影（呼吸峰是取带边界的产物、yaw-roll +0.415 是几何泄漏），而它们在数值上
全都像"实测优于借用文献值"。**没过正控的测量管线，采再多数据也只是把噪声量得更精确。**

本自检覆盖：
1. **规范层往返**：落盘再读回，数值与元数据不变、约定不匹配会报错。
2. **质量核查会红**：短片段/低采样率/丢帧/未动，各自触发对应的 issue。
3. **测量正控**：合成器按已知常数产数据 → 标定器是否量回那些常数。
4. **判别力（该红的要红）**：把真值改掉，量出来的必须跟着变——否则测量与真值无关。

跑：`python capture_selftest.py`（不需要任何真实数据）
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import _paths as P
import numpy as np
from capture_calibrate import (
    ANALYSIS_FPS,
    measure_axis_ratio,
    measure_band_peak,
    measure_coupling,
    measure_speed_profile,
)
from capture_schema import HeadPoseCapture

sys.path.insert(0, str(P.REPO))
from src.agents import motion_synth as ms  # noqa: E402

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"   {'✅' if ok else '❌'} {label}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def synth_capture(subject: str, seconds: float = 300.0, seed: int = 11) -> HeadPoseCapture:
    """用合成器造一段**真值已知**的采集（真值 = 当前模块常数）。"""
    frames, _ = ms.generate(
        0.0, 0.45, seconds * 1000.0, ms.PhaseState(noise_seed=seed), fps=ANALYSIS_FPS
    )
    axes = {
        "yaw": ms.PARAM_ANGLE_X,
        "pitch": ms.PARAM_ANGLE_Y,
        "roll": ms.PARAM_ANGLE_Z,
    }
    series = {k: np.array([float(f["params"][v]) for f in frames]) for k, v in axes.items()}
    return HeadPoseCapture(
        t_s=np.arange(len(frames)) / ANALYSIS_FPS,
        source="synthetic",
        subject=subject,
        **series,
    )


print("① 规范层往返")
capture = synth_capture("s01")
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "s01.npz"
    capture.to_npz(path)
    loaded = HeadPoseCapture.from_npz(path)
    check("落盘再读回，角度逐值相同", bool(np.allclose(capture.yaw, loaded.yaw)))
    check("元数据往返", loaded.subject == "s01" and loaded.source == "synthetic")
    meta = path.with_suffix(".json")
    meta.write_text(meta.read_text(encoding="utf-8").replace("anatomical-v1", "bogus-v9"), "utf-8")
    try:
        HeadPoseCapture.from_npz(path)
        check("约定不匹配应报错", False, "读进来了 —— 危险：错约定的数据会静默污染标定")
    except ValueError:
        check("约定不匹配应报错", True)

print("\n② 质量核查（每条都该红）")
short = synth_capture("s01", seconds=30.0)
check("时长不足被抓", any("时长" in i for i in short.quality()["issues"]))
still = HeadPoseCapture(
    t_s=np.arange(6000) / ANALYSIS_FPS,
    yaw=np.zeros(6000),
    pitch=np.zeros(6000),
    roll=np.zeros(6000),
    source="synthetic",
    subject="s01",
)
check("头没动被抓", any("疑似跟踪未生效" in i for i in still.quality()["issues"]))
gappy = synth_capture("s01")
gappy.t_s = gappy.t_s + np.concatenate([np.zeros(3000), np.full(len(gappy.t_s) - 3000, 5.0)])
check("丢帧被抓", gappy.quality()["dropouts"] > 0)
check(
    "subject 为空被抓",
    any(
        "subject" in i
        for i in HeadPoseCapture(
            t_s=capture.t_s,
            yaw=capture.yaw,
            pitch=capture.pitch,
            roll=capture.roll,
            source="x",
            subject="",
        ).quality()["issues"]
    ),
)

print("\n③ 测量正控：能否量回合成器的已知常数")
captures = [synth_capture(f"s{i:02d}", seed=seed) for i, seed in enumerate((11, 31, 77, 101, 233))]
axis = measure_axis_ratio(captures)
truth_pitch = ms.AXIS_AMPLITUDE_RATIO[1]
check(
    "三轴比例",
    abs(axis.measured - truth_pitch) < 0.10,
    f"量得 {axis.measured:.3f} · 真值 {truth_pitch}",
)
coupling = measure_coupling(captures)
check(
    "yaw-roll 耦合符号",
    (coupling.measured < 0) == (ms.YAW_ROLL_COUPLING < 0),
    f"量得 {coupling.measured:+.3f} · 真值 {ms.YAW_ROLL_COUPLING:+.2f}",
)

print("\n④ 判别力：改掉真值，量出来必须跟着变（否则测量与真值无关）")
original_ratio = ms.AXIS_AMPLITUDE_RATIO
original_coupling = ms.YAW_ROLL_COUPLING
try:
    ms.AXIS_AMPLITUDE_RATIO = (1.0, 0.80, 0.19)
    mutated = measure_axis_ratio(
        [synth_capture(f"s{i:02d}", seed=s) for i, s in enumerate((11, 31, 77))]
    )
    check(
        "三轴比例跟着真值走",
        mutated.measured > axis.measured * 1.5,
        f"真值 0.33→0.80 时量得 {axis.measured:.3f}→{mutated.measured:.3f}",
    )
    ms.AXIS_AMPLITUDE_RATIO = original_ratio
    ms.YAW_ROLL_COUPLING = +0.45  # 符号翻转
    flipped = measure_coupling(
        [synth_capture(f"s{i:02d}", seed=s) for i, s in enumerate((11, 31, 77))]
    )
    check(
        "耦合符号跟着真值翻",
        flipped.measured > 0,
        f"真值 −0.45→+0.45 时量得 {coupling.measured:+.3f}→{flipped.measured:+.3f}",
    )
finally:
    ms.AXIS_AMPLITUDE_RATIO = original_ratio
    ms.YAW_ROLL_COUPLING = original_coupling

print("\n⑤ 频带判据不得误杀真峰")
# ⚠ 这一条是 2026-08-07 正控实测抓出来的：合成数据 SWAY 真值就是 0.04，
#   而边界敏感性判据把它判成"跟着边界跑"——因为位移 0.003Hz 其实**不到一个频点**
#   （300s 片段的 FFT 分辨率就是 0.0033Hz）。容差没考虑分辨率 ⇒ 真峰被误杀。
sway = measure_band_peak(captures, "SWAY_HZ", ms.SWAY_HZ, (0.012, 0.016, 0.020), 0.15)
check(
    "已知真峰不被误杀",
    sway.verdict == "采信",
    f"真值 {ms.SWAY_HZ}Hz · 量得 {sway.measured:.4f}Hz · {sway.verdict}",
)
check(
    "量得的峰位接近真值",
    abs(sway.measured - ms.SWAY_HZ) < 0.015,
    f"偏差 {abs(sway.measured - ms.SWAY_HZ):.4f}Hz",
)

print("\n⑥ 速度分布靶子可算")
profile = measure_speed_profile(captures)
check(
    "归一分位单调递增",
    all(profile[f"p{a}"] <= profile[f"p{b}"] for a, b in ((10, 25), (25, 50), (50, 75), (75, 90))),
    f"p50={profile['p50']:.2f} p90={profile['p90']:.2f} p99={profile['p99']:.2f}",
)

print(
    "\n"
    + (
        "全部通过 —— 测量管线可用于真实采集"
        if not failures
        else f"❌ {len(failures)} 项失败：{failures}"
    )
)
raise SystemExit(1 if failures else 0)
