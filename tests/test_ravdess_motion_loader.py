"""RAVDESS 头姿 loader 与运动学统计量估计器的单测（合成 fixture，不需真数据）。

覆盖议会设计门定下的几条硬约束：丢帧必须被过滤（否则跟踪失败的突跳会被当成真实头动）、
演员 id 必须可解析（leave-actor-out 的前提）、有效帧不足时返回 None 而非全零。
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.agents.datasets.ravdess_motion import (  # noqa: E402
    MIN_VALID_FRAMES,
    MOTION_STAT_KEYS,
    affect_anchor,
    blink_intervals,
    extract_motion_stats,
    load_ravdess_motion,
    parse_actor_id,
)

_COLUMNS = [
    "frame",
    "timestamp",
    "confidence",
    "success",
    "pose_Rx",
    "pose_Ry",
    "pose_Rz",
    "AU45_c",
]


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)  # type: ignore[arg-type]


def _rows(
    n: int,
    *,
    amplitude: float = 0.1,
    fps: float = 30.0,
    success: int = 1,
    confidence: float = 0.98,
    blink_every: int | None = None,
) -> list[dict[str, object]]:
    """造一段正弦晃头：幅度可调，用于验证统计量随幅度单调。"""
    import math

    rows: list[dict[str, object]] = []
    for i in range(n):
        t = i / fps
        rows.append(
            {
                "frame": i + 1,
                "timestamp": round(t, 4),
                "confidence": confidence,
                "success": success,
                "pose_Rx": amplitude * math.sin(2 * math.pi * 0.5 * t),
                "pose_Ry": 0.0,
                "pose_Rz": 0.0,
                "AU45_c": 1 if blink_every and i % blink_every == 0 else 0,
            }
        )
    return rows


def _series(rows: list[dict[str, object]]) -> tuple[list[list[float]], list[float]]:
    frames = [[float(r["pose_Rx"]), float(r["pose_Ry"]), float(r["pose_Rz"])] for r in rows]
    stamps = [float(r["timestamp"]) for r in rows]
    return frames, stamps


def test_stats_keys_align_with_returned_length() -> None:
    """统计量名字与返回值逐位对应——错位会让下游按名字取到别的维。"""
    frames, stamps = _series(_rows(120))
    stats = extract_motion_stats(frames, stamps)
    assert stats is not None
    assert len(stats) == len(MOTION_STAT_KEYS)


def test_amplitude_and_speed_increase_with_motion() -> None:
    """幅度更大的同频晃动 → 幅度 RMS 与角速度均值都更大（唤醒→幅度/速度的可学前提）。"""
    small = extract_motion_stats(*_series(_rows(120, amplitude=0.05)))
    large = extract_motion_stats(*_series(_rows(120, amplitude=0.25)))
    assert small is not None and large is not None
    idx_amp = MOTION_STAT_KEYS.index("amplitude_rms")
    idx_speed = MOTION_STAT_KEYS.index("angular_speed_mean")
    assert large[idx_amp] > small[idx_amp]
    assert large[idx_speed] > small[idx_speed]


def test_constant_series_has_zero_amplitude_and_speed() -> None:
    """完全不动的序列 → 幅度与速度为 0（去均值后无残留），不是某个非零噪声。"""
    stats = extract_motion_stats(*_series(_rows(120, amplitude=0.0)))
    assert stats is not None
    assert stats[MOTION_STAT_KEYS.index("amplitude_rms")] == pytest.approx(0.0, abs=1e-9)
    assert stats[MOTION_STAT_KEYS.index("angular_speed_mean")] == pytest.approx(0.0, abs=1e-9)


def test_insufficient_frames_returns_none_not_zeros() -> None:
    """有效帧不足 → None。返回全零会被当成"这个情绪下头不动"的真实观测混进训练集。"""
    assert extract_motion_stats(*_series(_rows(MIN_VALID_FRAMES - 1))) is None


def test_parse_actor_id() -> None:
    """演员 id = 文件名第 7 段；leave-actor-out 全靠它。"""
    assert parse_actor_id("01-01-05-01-01-01-13.csv") == "13"
    assert parse_actor_id("garbage.csv") is None


def test_blink_intervals_counts_rising_edges() -> None:
    """眨眼间隔取上升沿之差：每 30 帧眨一次 @30fps → 间隔约 1.0s。"""
    rows = _rows(180, blink_every=30)
    intervals = blink_intervals([{k: str(v) for k, v in r.items()} for r in rows])
    assert len(intervals) == 5  # 6 次上升沿 → 5 个间隔
    assert all(i == pytest.approx(1.0, abs=1e-3) for i in intervals)


def test_loader_filters_failed_tracking(tmp_path: Path) -> None:
    """success==0 的帧必须被丢弃——跟踪失败帧的姿态突跳会污染幅度/jerk 统计量。

    构造：一条全程 success==1 的正常轨迹，一条帧数相同但 success==0 的轨迹。
    后者有效帧为 0 → 应整条跳过，而不是产出一行垃圾统计量。
    """
    _write_csv(tmp_path / "01-01-05-01-01-01-01.csv", _rows(120))
    _write_csv(tmp_path / "01-01-05-01-01-01-02.csv", _rows(120, success=0))
    x, y, groups = load_ravdess_motion(tmp_path, return_groups=True)
    assert x.shape[0] == 1  # 只有正常那条进来了
    assert groups == ["01"]
    assert y.shape[1] == len(MOTION_STAT_KEYS)


def test_loader_filters_low_confidence(tmp_path: Path) -> None:
    """置信度低于阈的帧同样丢弃（与 success 双判据，缺一不可）。"""
    _write_csv(tmp_path / "01-01-05-01-01-01-01.csv", _rows(120, confidence=0.1))
    with pytest.raises(FileNotFoundError):
        load_ravdess_motion(tmp_path)


def test_loader_reuses_shared_valence_anchor(tmp_path: Path) -> None:
    """valence 分量取自共享的 `EMOTION_CODE_TO_VA`，不另起一套映射（强度只动 arousal）。"""
    from src.agents.datasets.ravdess import EMOTION_CODE_TO_VA

    _write_csv(tmp_path / "01-01-05-01-01-01-07.csv", _rows(120))  # 05 = 愤怒
    x, _ = load_ravdess_motion(tmp_path)
    assert x[0][0].item() == pytest.approx(EMOTION_CODE_TO_VA["05"][0])


def test_intensity_scales_arousal_not_valence() -> None:
    """strong 的 arousal 幅度 > normal，valence 不受影响。

    实测依据：同演员×同情绪配对下 strong 的幅度/速度/jerk 分别在 66%/70%/65% 的配对中
    更大；而按情绪码映射的 arousal 与幅度几乎无关（r=0.04）。撤掉缩放这条会红。
    """
    v_normal, a_normal = affect_anchor("05", "01")  # 愤怒 normal
    v_strong, a_strong = affect_anchor("05", "02")  # 愤怒 strong
    assert v_normal == v_strong  # 效价不动
    assert abs(a_strong) > abs(a_normal)  # 唤醒幅度变大


def test_missing_intensity_falls_back_to_base_anchor() -> None:
    """强度码缺失/未知 → 退回基锚点，不缩放（中性只有 normal 一档，不能因此丢数据）。"""
    from src.agents.datasets.ravdess import EMOTION_CODE_TO_VA

    assert affect_anchor("01", None) == pytest.approx(EMOTION_CODE_TO_VA["01"])
    assert affect_anchor("03", "99") == pytest.approx(EMOTION_CODE_TO_VA["03"])


def test_arousal_stays_in_range() -> None:
    """缩放后仍钳在 [-1,1]——越界会让下游模型收到定义域外的标签。"""
    for code in ("01", "02", "03", "04", "05", "06", "07", "08"):
        for intensity in ("01", "02", None):
            _, arousal = affect_anchor(code, intensity)
            assert -1.0 <= arousal <= 1.0


def test_loader_return_groups_default_off(tmp_path: Path) -> None:
    """默认不返回分组 → 与其它 loader 同形（零回归）。"""
    _write_csv(tmp_path / "01-01-05-01-01-01-01.csv", _rows(120))
    result = load_ravdess_motion(tmp_path)
    assert len(result) == 2
