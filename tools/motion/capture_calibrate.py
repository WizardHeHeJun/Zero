"""从自采数据标定合成器常数：产出**带判据的提案**，不直接改代码。

## 设计原则（都是这一程踩出来的）

1. **每个量都带判据，不只带数字。** 通过判据才标「采信」，否则标「否决 + 理由」。
   2026-08-07 那轮四个测量里有两个是伪影（呼吸峰是取带边界的产物、yaw-roll +0.415 是
   几何泄漏），光看数字全都像"同域实测优于借用文献值"。
2. **靶子优先用无阈值量。** 角速度**分布**不需要定阈；「移动/驻留占空比」要定阈，
   而实测真数据与合成数据**都不存在阈值平台**——量出来的是阈值不是数据。
3. **不自动改 `src/`。** 只写提案 JSON + 报告。常数改动要过守卫、变异验证与真机盲测，
   那是人的判断，不是脚本的。
4. **先过正控再信它。** `capture_selftest.py` 用已知常数合成数据喂进本模块，
   验证能量回原值。没跑过正控就用标定结果 = 用一把没校过的尺子。

跑：
    python capture_calibrate.py <采集目录>            # 出报告 + 提案
    python capture_calibrate.py <采集目录> --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import _paths as P
import numpy as np
from capture_schema import HeadPoseCapture, load_dataset

sys.path.insert(0, str(P.REPO))
from src.agents import motion_synth as ms  # noqa: E402

ANALYSIS_FPS = 20.0  # 与合成器输出帧率一致——差分尺度必须同口径
SPEED_PCTS = (10, 25, 50, 75, 90, 95, 99)


@dataclass
class Finding:
    """一项测量结果 + 它的判据结论。"""

    name: str
    measured: float
    current: float
    verdict: str  # "采信" / "否决"
    evidence: str

    def line(self) -> str:
        mark = "✅" if self.verdict == "采信" else "❌"
        return (
            f"  {mark} {self.name:22s} 实测 {self.measured:>8.4f} · 现值 {self.current:>8.4f}\n"
            f"     {self.evidence}"
        )


def _resample(capture: HeadPoseCapture) -> tuple[np.ndarray, ...]:
    """按**真实时间戳**重采样到 ANALYSIS_FPS——采集间隔会抖，等间隔假设会把抖动算成运动。"""
    grid = np.arange(capture.t_s[0], capture.t_s[-1], 1.0 / ANALYSIS_FPS)
    return tuple(
        np.interp(grid, capture.t_s, getattr(capture, axis)) for axis in ("yaw", "pitch", "roll")
    )


def _speeds(axes: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.linalg.norm(np.diff(np.stack(axes, 1), axis=0), axis=1) * ANALYSIS_FPS


def _binomial_p(successes: int, n: int) -> float:
    if n == 0:
        return 1.0
    pmf = [math.comb(n, k) * 0.5**n for k in range(n + 1)]
    return min(1.0, sum(p for p in pmf if p <= pmf[successes] + 1e-12))


def measure_axis_ratio(captures: list[HeadPoseCapture]) -> Finding:
    ratios_pitch, ratios_roll = [], []
    for capture in captures:
        yaw, pitch, roll = _resample(capture)
        base = float(np.std(yaw))
        if base < 1e-6:
            continue
        ratios_pitch.append(float(np.std(pitch)) / base)
        ratios_roll.append(float(np.std(roll)) / base)
    pitch_ratio = statistics.median(ratios_pitch)
    roll_ratio = statistics.median(ratios_roll)
    return Finding(
        name="AXIS_AMPLITUDE_RATIO[1] (pitch)",
        measured=pitch_ratio,
        current=ms.AXIS_AMPLITUDE_RATIO[1],
        verdict="采信",
        evidence=(
            f"n={len(ratios_pitch)} 段 · roll 比 {roll_ratio:.3f}（现值 "
            f"{ms.AXIS_AMPLITUDE_RATIO[2]}）· sd 比值无自由参数，直接可用"
        ),
    )


def measure_coupling(captures: list[HeadPoseCapture]) -> Finding:
    """yaw-roll 耦合：**增量**相关（去掉慢漂移），并做逐片段符号一致性检验。"""
    values = []
    for capture in captures:
        yaw, pitch, roll = _resample(capture)
        d_yaw, d_roll = np.diff(yaw), np.diff(roll)
        speed = np.abs(d_yaw) + np.abs(np.diff(pitch)) + np.abs(d_roll)
        moving = speed > np.median(speed)  # 静止帧的 Δ 全是噪声，会把真耦合稀释向 0
        if moving.sum() < 30 or np.std(d_yaw[moving]) < 1e-9:
            continue
        values.append(float(np.corrcoef(d_yaw[moving], d_roll[moving])[0, 1]))
    median = statistics.median(values)
    same = max(sum(1 for v in values if v > 0), sum(1 for v in values if v < 0))
    n = len(values)
    p = _binomial_p(same, n)
    consistency = same / n
    stable = p < 0.05 and consistency >= 0.75
    # ⚠ 「样本不足」与「符号不稳定」是**两种不同的否决**，理由不能混：
    #    5/5 完全一致但 p=0.0625（二项下限就是这样）属前者，说成"不稳定"会误导人
    #    去怀疑数据，其实该做的是多采几段。
    if stable:
        reason = "；符号稳定，属运动学固有属性"
    elif consistency >= 0.75:
        reason = (
            f"；一致率够（{consistency:.0%}）但**段数太少**"
            f"（n={n}，二项检验过不了 0.05）——多采几段即可"
        )
    else:
        reason = f"；一致率仅 {consistency:.0%} = 符号真不稳定，可能只是个体习惯，不可采信"
    return Finding(
        name="YAW_ROLL_COUPLING",
        measured=median,
        current=ms.YAW_ROLL_COUPLING,
        verdict="采信" if stable else "否决",
        evidence=f"逐片段符号一致 {same}/{n}（p={p:.4f}）" + reason,
    )


def measure_band_peak(
    captures: list[HeadPoseCapture], name: str, current: float, edges: tuple[float, ...], hi: float
) -> Finding:
    """频带主峰 + **边界敏感性**判据：若是 1/f 背景的产物，峰会贴着下边界走。

    ⚠ 扫描下限不得低于频率分辨率（1/片段时长），上限须明显低于候选峰——
    边界越过真峰后峰位当然被推着走，那是对照臂选错、不是判据失败。
    """
    peaks: dict[float, list[float]] = {}
    for capture in captures:
        _, pitch, _ = _resample(capture)
        signal = pitch - np.mean(pitch)
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
        freqs = np.fft.rfftfreq(len(signal), d=1 / ANALYSIS_FPS)
        resolution = 1.0 / (len(signal) / ANALYSIS_FPS)
        for lo in edges:
            if lo < resolution * 1.5:
                continue
            sel = (freqs >= lo) & (freqs <= hi)
            if sel.any():
                peaks.setdefault(lo, []).append(float(freqs[sel][np.argmax(spectrum[sel])]))
    if len(peaks) < 2:
        return Finding(name, float("nan"), current, "否决", "片段太短，频率分辨率不足以做边界检验")
    lows = sorted(peaks)
    medians = {lo: statistics.median(v) for lo, v in peaks.items()}
    shift = medians[lows[-1]] - medians[lows[0]]
    # ⚠ 容差必须**取分辨率与边界跨度的较大者**：位移不到一个频点时，那是 FFT 分辨率下限、
    #    不是"跟着边界跑"。只按边界跨度的比例判会把真峰误杀——正控实测踩到：
    #    合成数据 SWAY 真值 0.04，位移 0.003Hz 而 300s 片段的分辨率就是 0.0033Hz。
    # 取 1.5 个频点：位移一两个 bin 在物理上**不可分辨**，不是"跟着边界跑"的证据。
    # （正控实测：合成数据 SWAY 真值 0.04，位移恰好 1 bin，写死 1.0 会卡在边界上误杀。）
    resolutions = [1.0 / (len(_resample(c)[0]) / ANALYSIS_FPS) for c in captures]
    tolerance = max(statistics.median(resolutions) * 1.5, (lows[-1] - lows[0]) * 0.3)
    stable = shift <= tolerance
    detail = " · ".join(f"下限{lo:.3f}→峰{medians[lo]:.3f}Hz" for lo in lows)
    return Finding(
        name=name,
        measured=medians[lows[0]],
        current=current,
        verdict="采信" if stable else "否决",
        evidence=detail
        + (
            f"；位移 {shift:.4f}Hz ≤ 容差 {tolerance:.4f}Hz（含频率分辨率），峰位稳定 = 真峰"
            if stable
            else (
                f"；位移 {shift:.4f}Hz > 容差 {tolerance:.4f}Hz "
                "⇒ **峰位跟着边界跑 = 取带方式的产物**"
            )
        ),
    )


def measure_speed_profile(captures: list[HeadPoseCapture]) -> dict[str, float]:
    """角速度分布（按自身中位数归一）——**无阈值**，是运动学的主靶子。"""
    speeds = np.concatenate([_speeds(_resample(c)) for c in captures])
    median = float(np.median(speeds))
    return {f"p{p}": float(np.percentile(speeds, p)) / median for p in SPEED_PCTS} | {
        "median_deg_s": median
    }


def fit_speed_knobs(target: dict[str, float]) -> tuple[float, float, float]:
    """搜 (MICRO_TREMOR_RATIO, POSE_RISE_S) 使合成器的归一速度分布贴近靶子。

    只用**高尾**（p90/p95/p99）作目标：那一段对应"猛动"的观感，是用户反馈里
    「过快」的直接来源；低尾受皮套量程与基幅限制，改不动也不该在这里改。
    """
    reference = np.array([target[f"p{p}"] for p in (90, 95, 99)])
    original = (ms.MICRO_TREMOR_RATIO, ms.POSE_RISE_S)
    best = (float("inf"), *original)
    try:
        for tremor in np.arange(0.08, 0.36, 0.02):
            for rise in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
                ms.MICRO_TREMOR_RATIO, ms.POSE_RISE_S = float(tremor), float(rise)
                speeds = np.concatenate(
                    [
                        _speeds(
                            tuple(
                                np.array([float(f["params"][k]) for f in frames])
                                for k in (ms.PARAM_ANGLE_X, ms.PARAM_ANGLE_Y, ms.PARAM_ANGLE_Z)
                            )
                        )
                        for frames, _ in (
                            ms.generate(
                                0.0, 0.45, 120_000.0, ms.PhaseState(noise_seed=s), fps=ANALYSIS_FPS
                            )
                            for s in (11, 31, 77)
                        )
                    ]
                )
                profile = np.array(
                    [np.percentile(speeds, p) / np.median(speeds) for p in (90, 95, 99)]
                )
                loss = float(np.mean(np.abs(np.log(profile / reference))))
                if loss < best[0]:
                    best = (loss, float(tremor), float(rise))
    finally:
        ms.MICRO_TREMOR_RATIO, ms.POSE_RISE_S = original
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="采集目录（*.npz + 同名 .json）")
    parser.add_argument("--scene", default="idle", help="只用该场景（待机与说话必须分开标定）")
    parser.add_argument("--json", type=Path, default=None, help="提案写到这个 JSON")
    args = parser.parse_args()

    captures = load_dataset(args.directory, scene=args.scene)
    if not captures:
        raise SystemExit(f"{args.directory} 下没有 scene={args.scene} 的采集")

    print(
        f"采集 {len(captures)} 段 · 受试者 {len({c.subject for c in captures})} 人 · "
        f"场景 {args.scene}"
    )
    bad = [(c, q) for c in captures if not (q := c.quality())["ok"]]
    for capture, quality in bad:
        print(f"  ⚠ {capture.subject}/{capture.session}: {'; '.join(quality['issues'])}")
    if len({c.subject for c in captures}) < 3:
        print("  ⚠ 受试者不足 3 人——个体差异是本通道最大方差来源，样本太少会把某个人的习惯当成通则")

    print("\n" + "=" * 84)
    print("① 常数提案（每项带判据）")
    print("=" * 84)
    findings = [
        measure_axis_ratio(captures),
        measure_coupling(captures),
        measure_band_peak(captures, "SWAY_HZ", ms.SWAY_HZ, (0.012, 0.016, 0.020), 0.15),
        measure_band_peak(captures, "BREATH_HZ", ms.BREATH_HZ, (0.10, 0.15, 0.20), 0.5),
    ]
    for finding in findings:
        print(finding.line())

    print("\n" + "=" * 84)
    print("② 运动学主靶子：角速度分布（无阈值）")
    print("=" * 84)
    target = measure_speed_profile(captures)
    print(f"  真人速度中位 {target['median_deg_s']:.2f}°/s")
    print("  归一分位 " + " ".join(f"p{p}={target[f'p{p}']:.2f}" for p in SPEED_PCTS))
    loss, tremor, rise = fit_speed_knobs(target)
    print(
        f"\n  拟合 ⇒ MICRO_TREMOR_RATIO={tremor:.2f}（现 {ms.MICRO_TREMOR_RATIO}）"
        f" · POSE_RISE_S={rise:.2f}（现 {ms.POSE_RISE_S}）· 高尾平均对数偏离 {math.exp(loss):.2f}×"
    )

    proposal = {
        "scene": args.scene,
        "n_captures": len(captures),
        "n_subjects": len({c.subject for c in captures}),
        "findings": [asdict(f) for f in findings],
        "speed_profile": target,
        "fit": {
            "micro_tremor_ratio": tremor,
            "pose_rise_s": rise,
            "hi_tail_deviation": math.exp(loss),
        },
    }
    if args.json:
        args.json.write_text(json.dumps(proposal, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n提案 → {args.json}")
    print("\n🛑 提案不等于改动：落地前须过守卫 + 变异验证 + 真机盲测，见 CAPTURE.md。")


if __name__ == "__main__":
    main()
