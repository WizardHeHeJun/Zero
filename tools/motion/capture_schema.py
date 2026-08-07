"""自采动作数据的**规范格式**：一切采集源都先落成这个形状，再进分析/标定。

## 为什么要有规范层

这一程在轴映射上错了三次，根因都是「约定被当成常识」：`freemocap` 是 Z-up 面朝 −Y、
`ReActIdle` 是 Y-up 面朝 +Z 而**关节名完全相同**；OpenFace 的「右」是画面的右还是人的右
文档不写；Live2D 的 `ParamAngleX` 说的是**画面**的右。每接一个新源就要重踩一遍。

规范层把这件事**一次性钉死**：任何采集源都必须在**入口**声明并转换到下面这套解剖约定，
之后所有分析代码只认这一套。转换对不对由 `capture_selftest.py` 的正控把关。

## 解剖约定（与 `anatomy.py` 完全一致，不要另立一套）

| 量 | 正方向 | 单位 |
| --- | --- | --- |
| `yaw` | 头转向**受试者自己的右侧** | 度 |
| `pitch` | **抬头** | 度 |
| `roll` | **头顶**倒向自己的右侧（右耳靠近右肩） | 度 |

⚠ `roll` 必须是 **swing-twist**（绕当前视线轴的扭转），不是 `atan2(up_x, up_z)`——
后者在 pitch≠0 时混入正比于 yaw 的分量，会凭空造出 yaw-roll 相关（已被消融证伪）。

## 时间

`t_s` 存**真实时间戳**，不假设等间隔。采集端（尤其轮询式的 VTS 跟踪）实际间隔会抖，
写死 fps 再差分会把抖动算成运动。分析侧统一重采样后再算速度。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CONVENTION = (
    "anatomical-v1: +yaw=turn to subject's right, +pitch=up, +roll=crown to subject's right"
)

# 质量下限。依据都是这一程实测踩到的坑，不是拍的：
# - 时长：单段 200 秒才有约 83 个姿态周期；更短时 sd 有 ~9% 波动、clamp 这类尾部事件抽不到。
# - 帧率：低于 15Hz 时 0.6s 量级的转移只剩个位数采样点，转移时长无从谈起。
# - 间隔抖动：中位间隔的 3 倍以上视为丢帧，须计入而非静默插值。
MIN_DURATION_S = 200.0
MIN_FPS = 15.0
DROPOUT_FACTOR = 3.0


@dataclass
class HeadPoseCapture:
    """一段头姿采集。角度单位为度，约定见模块 docstring。

    Attributes:
        t_s: 真实时间戳（秒，单调递增，**不假设等间隔**）。
        yaw/pitch/roll: 逐帧角度（度）。
        source: 采集源标识（如 `vts-tracking` / `bvh:freemocap` / `csv:<设备名>`）。
        subject: 受试者标识（**必填**——个体差异是本通道最大的方差来源，
            按人分组留出是硬要求，混在一起会让"泛化"变成"记住了这个人"）。
        session: 同一受试者的第几次采集（同人不同天的姿势习惯也会漂）。
        scene: 场景标签，如 `idle` / `speaking`。🛑 **待机与说话必须分开采**——
            说话时约 80% 的头动是言语驱动的，混进待机集会把整套常数带偏。
        notes: 自由备注（设备、坐姿、环境等）。
    """

    t_s: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    roll: np.ndarray
    source: str
    subject: str
    session: str = "1"
    scene: str = "idle"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return float(self.t_s[-1] - self.t_s[0]) if len(self.t_s) > 1 else 0.0

    @property
    def median_fps(self) -> float:
        if len(self.t_s) < 2:
            return 0.0
        return 1.0 / float(np.median(np.diff(self.t_s)))

    def quality(self) -> dict[str, Any]:
        """质量核查。**不自动修复**——发现问题就报出来由人决定，静默插值会把补出来的
        值当成真实运动进统计量（同 `ravdess_motion` loader 的既定处置）。"""
        issues: list[str] = []
        if len(self.t_s) < 2:
            return {"ok": False, "issues": ["采集为空或只有一帧"]}
        intervals = np.diff(self.t_s)
        if np.any(intervals <= 0):
            issues.append("时间戳非单调递增")
        median_interval = float(np.median(intervals))
        dropouts = int(np.sum(intervals > median_interval * DROPOUT_FACTOR))
        if self.duration_s < MIN_DURATION_S:
            issues.append(f"时长 {self.duration_s:.0f}s < 下限 {MIN_DURATION_S:.0f}s")
        if self.median_fps < MIN_FPS:
            issues.append(f"采样率 {self.median_fps:.1f}Hz < 下限 {MIN_FPS:.0f}Hz")
        if dropouts:
            issues.append(f"{dropouts} 处丢帧（间隔 > {DROPOUT_FACTOR}× 中位）")
        if not self.subject:
            issues.append("subject 为空——按人分组留出无从做起")
        spread = float(np.std(self.yaw))
        if spread < 0.5:
            issues.append(f"yaw 标准差仅 {spread:.2f}°，疑似跟踪未生效/头未动")
        return {
            "ok": not issues,
            "issues": issues,
            "duration_s": round(self.duration_s, 1),
            "median_fps": round(self.median_fps, 2),
            "dropouts": dropouts,
            "frames": len(self.t_s),
        }

    def to_npz(self, path: Path) -> dict[str, Any]:
        """落盘 `.npz` + 同名 `.json` provenance 边车（对齐本仓权重 sidecar 惯例）。

        Returns:
            写出的 provenance 字典（也已写进边车文件）。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, t_s=self.t_s, yaw=self.yaw, pitch=self.pitch, roll=self.roll)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        provenance = {
            "schema_version": 1,
            "convention": CONVENTION,
            "artifact": path.name,
            "artifact_sha256": digest,
            "source": self.source,
            "subject": self.subject,
            "session": self.session,
            "scene": self.scene,
            "notes": self.notes,
            "quality": self.quality(),
            "extra": self.extra,
        }
        path.with_suffix(".json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return provenance

    @classmethod
    def from_npz(cls, path: Path) -> HeadPoseCapture:
        path = Path(path)
        data = np.load(path)
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if meta and meta.get("convention") != CONVENTION:
            raise ValueError(
                f"{path.name} 的约定是 {meta.get('convention')!r}，与当前 {CONVENTION!r} 不符"
                "——**不要**直接改这行放行，先确认转换是否需要重做"
            )
        return cls(
            t_s=data["t_s"],
            yaw=data["yaw"],
            pitch=data["pitch"],
            roll=data["roll"],
            source=meta.get("source", "unknown"),
            subject=meta.get("subject", ""),
            session=meta.get("session", "1"),
            scene=meta.get("scene", "idle"),
            notes=meta.get("notes", ""),
            extra=meta.get("extra", {}),
        )


def load_dataset(directory: Path, *, scene: str | None = "idle") -> list[HeadPoseCapture]:
    """加载一个目录下的全部采集；`scene` 非 None 时只取该场景。

    🛑 默认只取 `idle`：待机与说话的运动学**是两个分布**，混着标定等于两边都不对。
    """
    out: list[HeadPoseCapture] = []
    for path in sorted(Path(directory).glob("*.npz")):
        capture = HeadPoseCapture.from_npz(path)
        if scene is not None and capture.scene != scene:
            continue
        out.append(capture)
    return out
