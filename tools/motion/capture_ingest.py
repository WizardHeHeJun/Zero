"""把各种采集源转成 `capture_schema.HeadPoseCapture`。

🛑 **每个适配器都必须显式声明源的约定，不许猜。** 这一程在轴映射上错三次，全是
「看着像 Y-up 就当 Y-up」。所以：

- BVH：约定由 `anatomy.detect_anatomy()` **从骨架左右对称关节实测判定**，不读死轴号。
- 通用 CSV：调用方**必须**传 `axis_signs` 明确每轴正方向对应哪个解剖方向；
  不传就报错，**不给默认值**——默认值就是猜。

新增一个源时，配套在 `capture_selftest.py` 里加一条"合成已知运动 → 过该适配器 →
是否量回原值"的正控。没有正控的适配器不要用。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from anatomy import detect_anatomy, head_angles, parse_bvh
from capture_schema import HeadPoseCapture


def from_bvh(
    path: Path, *, subject: str, fps: float, scene: str = "idle", session: str = "1"
) -> HeadPoseCapture:
    """BVH mocap → 规范格式。坐标系由骨架 rest OFFSET **实测判定**（见 `anatomy.py`）。

    Args:
        fps: BVH 的采样率。⚠ 用文件里 `Frame Time` 的倒数更稳，但不少导出器写错，
            故要求调用方显式给——错了会让所有速度量按比例失真且**不驱红**。
    """
    skeleton = parse_bvh(Path(path))
    angles = head_angles(skeleton, detect_anatomy(skeleton))
    count = len(angles["yaw"])
    return HeadPoseCapture(
        t_s=np.arange(count) / fps,
        yaw=angles["yaw"],
        pitch=angles["pitch"],
        roll=angles["roll"],
        source=f"bvh:{Path(path).parent.name}",
        subject=subject,
        session=session,
        scene=scene,
        notes=f"轴系实测判定：{detect_anatomy(skeleton).describe()}",
    )


def from_csv(
    path: Path,
    *,
    subject: str,
    columns: dict[str, str],
    axis_signs: dict[str, int],
    degrees: bool = True,
    time_column: str | None = None,
    fps: float | None = None,
    scene: str = "idle",
    session: str = "1",
    source: str = "csv",
) -> HeadPoseCapture:
    """通用 CSV → 规范格式（给自采设备用）。

    Args:
        columns: 解剖量 → CSV 列名，如 `{"yaw": "head_yaw", "pitch": "...", "roll": "..."}`。
        axis_signs: 每个量的**符号修正** `+1`/`-1`——即该列的正方向是否与本仓解剖约定同向
            （+yaw=转向自己的右、+pitch=抬头、+roll=头顶倒向自己的右）。
            🛑 **必填且必须先实测确认**：录一段"只向右转头"看该列是正是负，别读文档猜。
        degrees: 源是否已是度；False 则按弧度换算。
        time_column / fps: 二选一。有真实时间戳就用它（推荐）；只有等间隔假设时给 fps。

    Raises:
        ValueError: 缺 `axis_signs` 的某一轴、或时间信息缺失/冲突。
    """
    missing = [k for k in ("yaw", "pitch", "roll") if k not in axis_signs]
    if missing:
        raise ValueError(
            f"axis_signs 缺 {missing}——必须显式声明每轴正方向（录一段单向转头实测确认），"
            "本函数刻意不给默认值：默认值就是猜"
        )
    if (time_column is None) == (fps is None):
        raise ValueError("time_column 与 fps 必须且只能给一个")

    rows = list(csv.DictReader(Path(path).open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"{path} 无数据行")

    def column(name: str) -> np.ndarray:
        values = np.array([float(r[columns[name]]) for r in rows])
        if not degrees:
            values = np.degrees(values)
        return values * axis_signs[name]

    stamps = (
        np.array([float(r[time_column]) for r in rows])
        if time_column
        else np.arange(len(rows)) / float(fps)  # type: ignore[arg-type]
    )
    return HeadPoseCapture(
        t_s=stamps,
        yaw=column("yaw"),
        pitch=column("pitch"),
        roll=column("roll"),
        source=source,
        subject=subject,
        session=session,
        scene=scene,
        notes=f"axis_signs={axis_signs} degrees={degrees}",
    )


def from_vts_parameters(
    t_s: np.ndarray,
    face_angle_x: np.ndarray,
    face_angle_y: np.ndarray,
    face_angle_z: np.ndarray,
    *,
    subject: str,
    scene: str = "idle",
    session: str = "1",
    notes: str = "",
) -> HeadPoseCapture:
    """VTS 跟踪参数 → 规范格式。

    符号换算（**官方文档现场核验**，非推断）：Live2D 标准参数表对 `ParamAngleX` 与
    `ParamAngleZ` 用逐字相同的措辞「+で画面の右を向く / Turn to the right of the screen
    when +」，`ParamAngleY` 是「+で画面の上を向く / Turn face up when +」
    （https://docs.live2d.com/en/cubism-editor-manual/standard-parameter-list/）。
    「画面的右」= 面对镜头的人**自己的左** ⇒ X 与 Z 都要取反，Y 同向。

    ⚠ VTS 里每个模型可把 OUTPUT 区间反号（社区处理摄像头镜像的常规做法）。但本函数吃的是
    **INPUT 侧的跟踪值**，不过模型输出映射，故不受其影响；若哪天改从输出侧取，须重验。
    """
    return HeadPoseCapture(
        t_s=np.asarray(t_s, dtype=float),
        yaw=-np.asarray(face_angle_x, dtype=float),
        pitch=np.asarray(face_angle_y, dtype=float),
        roll=-np.asarray(face_angle_z, dtype=float),
        source="vts-tracking",
        subject=subject,
        session=session,
        scene=scene,
        notes=notes or "FaceAngleX/Z 取反（画面右 = 受试者左）；Y 同向",
    )
