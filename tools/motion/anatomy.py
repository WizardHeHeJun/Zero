"""BVH 解剖坐标系的**自动判定** + 免泄漏的头部 yaw/pitch/roll 提取。

## 为什么要有这个模块

动作层此前三次在轴映射上出错，根因是**约定被当成常识**：

1. `freemocap`(StayStill) 是 Z-up，`ReActIdle` 是 Y-up —— 两者**关节名完全相同**
   （`neck`/`face`/`upper_arm.L`…），只看名字读通道必错。
2. 旧提取把关节局部 +Y 当"朝前"。按左右对称关节实测：StayStill 的 `upper_arm.L`
   OFFSET 在 +X ⇒ 左=+X、上=+Z ⇒ **面朝 −Y**，局部 +Y 其实是**朝后**。
   （ReActIdle 则是 左=+X、上=+Y ⇒ 面朝 +Z。）
3. 旧 roll 用 `atan2(up_x, up_z)`，在 pitch≠0 时会**混入 yaw**：
   先 pitch α 再 yaw θ ⇒ up 的横向分量 = sinα·sinθ。待机数据 pitch·yaw 都不小
   （sd 约 7° / 21°），该伪分量量级与真 roll 的 sd 同阶 —— 它**正比于 yaw**，
   于是凭空造出一个 yaw-roll 相关，符号由「这个人平时低头还是抬头」决定。
   这足以解释"待机 +0.415 / 说话 −0.125 符号相反"的谜。

## 判定方法（不依赖任何文档约定，全部从数据本身取）

- **左右**：`upper_arm.L` / `upper_arm.R` 的 rest OFFSET 之差（肩→上臂是纯侧向）。
- **上**：脊柱链 `neck → face` 的 rest OFFSET（头顶方向）。
- **前**：`up × right`，符号由「人体三元组 (right, forward, up) 右手系」定；
  若该 BVH 是左手系，`_handedness` 会测出来并翻转，故对两种手性都成立。

## 角度定义（解剖语义，与坐标轴无关）

以 rest 姿态的 (right₀, forward₀, up₀) 为参考系，头部朝向矩阵 `M = R_neck ∘ R_face`
（相对**躯干**，正是颈椎耦合该测的量）：

- `yaw`   > 0 ⇒ 头转向**受试者自己的右侧**
- `pitch` > 0 ⇒ **抬头**
- `roll`  > 0 ⇒ **头顶倒向自己的右侧**（右耳靠近右肩，即"向右歪头"）

`roll` 用 **swing-twist**：先把世界上方投影到与当前视线垂直的平面得到"零 roll 参考上方"
`u_ref`，再量头顶相对它的扭转角。纯 yaw / 纯 pitch / 二者复合都恒给 roll=0，
**不会有轴间泄漏**——这是与旧公式的唯一实质差别。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# rest 姿态下用来定侧向的一对左右对称关节（按优先级尝试）
LATERAL_JOINT_PAIRS = (("upper_arm.L", "upper_arm.R"), ("thigh.L", "thigh.R"))
# rest 姿态下用来定头顶方向的关节（其 OFFSET 即 neck→face 的向量）
HEAD_UP_JOINT = "face"


@dataclass(frozen=True)
class Skeleton:
    """一份 BVH 的静态结构（层级 + 通道 + rest OFFSET）与逐帧数值。"""

    joints: list[str]
    channels: dict[str, list[str]]
    offsets: dict[str, np.ndarray]
    frames: np.ndarray  # (n_frames, n_channels)
    frame_time: float

    def channel_index(self, joint: str, channel: str) -> int | None:
        index = 0
        for name in self.joints:
            for chan in self.channels.get(name, []):
                if name == joint and chan == channel:
                    return index
                index += 1
        return None


@dataclass(frozen=True)
class AnatomyFrame:
    """rest 姿态下的解剖三轴（在躯干局部坐标里的单位向量）。"""

    right: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    handedness: float  # cross(forward, up) 与 right 同向为 +1，反向为 −1

    def describe(self) -> str:
        def axis_name(v: np.ndarray) -> str:
            i = int(np.argmax(np.abs(v)))
            return f"{'-' if v[i] < 0 else '+'}{'XYZ'[i]}"

        return (
            f"右={axis_name(self.right)} 前={axis_name(self.forward)} "
            f"上={axis_name(self.up)} (手性 {self.handedness:+.0f})"
        )


def parse_bvh(path: Path) -> Skeleton:
    """解析 BVH：层级、通道、rest OFFSET、MOTION 数值。"""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    joints: list[str] = []
    channels: dict[str, list[str]] = {}
    offsets: dict[str, np.ndarray] = {}
    current: str | None = None
    frames: np.ndarray = np.zeros((0, 0))
    frame_time = 1.0 / 30.0

    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith(("ROOT ", "JOINT ")):
            current = stripped.split(None, 1)[1]
            joints.append(current)
        elif stripped.startswith("OFFSET") and current is not None and current not in offsets:
            offsets[current] = np.array([float(v) for v in stripped.split()[1:4]])
        elif stripped.startswith("CHANNELS") and current is not None:
            channels[current] = stripped.split()[2:]
        elif stripped.startswith("MOTION"):
            count = int(lines[i + 1].split(":")[1])
            frame_time = float(lines[i + 2].split(":")[1])
            rows = [ln.split() for ln in lines[i + 3 : i + 3 + count] if ln.strip()]
            frames = np.array(rows, dtype=float)
            break
    return Skeleton(joints, channels, offsets, frames, frame_time)


def _handedness(right: np.ndarray, forward: np.ndarray, up: np.ndarray) -> float:
    return float(np.sign(np.dot(np.cross(forward, up), right)))


def detect_anatomy(skeleton: Skeleton) -> AnatomyFrame:
    """从 rest OFFSET 判定解剖三轴。**不看关节名以外的任何约定**。

    Raises:
        ValueError: 骨架缺少左右对称关节或头部关节，无法判定（宁可报错也不猜）。
    """
    left = None
    for left_name, right_name in LATERAL_JOINT_PAIRS:
        if left_name in skeleton.offsets and right_name in skeleton.offsets:
            left = skeleton.offsets[left_name] - skeleton.offsets[right_name]
            if np.linalg.norm(left) > 1e-6:
                break
            left = None
    if left is None:
        raise ValueError("骨架缺少可用的左右对称关节，无法判定侧向轴")
    if HEAD_UP_JOINT not in skeleton.offsets:
        raise ValueError(f"骨架缺少 {HEAD_UP_JOINT} 关节，无法判定头顶方向")

    up = skeleton.offsets[HEAD_UP_JOINT].astype(float)
    if np.linalg.norm(up) <= 1e-6:
        raise ValueError(f"{HEAD_UP_JOINT} 的 OFFSET 为零向量，无法判定头顶方向")
    up = up / np.linalg.norm(up)
    right = -left / np.linalg.norm(left)  # 受试者的右 = 左的反向
    right = right - up * np.dot(right, up)  # 正交化（防 OFFSET 略带纵向分量）
    right = right / np.linalg.norm(right)

    # 人体三元组 (right, forward, up) 取右手系 ⇒ forward = up × right；
    # 若该 BVH 实为左手系，下面的 handedness 会测出 −1，后续 roll 公式据此翻转。
    forward = np.cross(up, right)
    forward = forward / np.linalg.norm(forward)
    return AnatomyFrame(right, forward, up, _handedness(right, forward, up))


def _axis_rotations(degrees: np.ndarray, axis: str) -> np.ndarray:
    """(n,) 角度 → (n,3,3) 绕单轴旋转矩阵。"""
    rad = np.radians(degrees)
    cos, sin = np.cos(rad), np.sin(rad)
    one, zero = np.ones_like(cos), np.zeros_like(cos)
    if axis == "X":
        rows = [[one, zero, zero], [zero, cos, -sin], [zero, sin, cos]]
    elif axis == "Y":
        rows = [[cos, zero, sin], [zero, one, zero], [-sin, zero, cos]]
    else:
        rows = [[cos, -sin, zero], [sin, cos, zero], [zero, zero, one]]
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def joint_matrices(skeleton: Skeleton, joint: str) -> np.ndarray:
    """某关节的逐帧局部旋转矩阵 (n,3,3)，按 BVH 通道顺序左乘。"""
    n = len(skeleton.frames)
    result = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    for channel in skeleton.channels.get(joint, []):
        if not channel.endswith("rotation"):
            continue
        index = skeleton.channel_index(joint, channel)
        if index is None:
            continue
        result = result @ _axis_rotations(skeleton.frames[:, index], channel[0])
    return result


def head_angles(skeleton: Skeleton, anatomy: AnatomyFrame) -> dict[str, np.ndarray]:
    """逐帧头部朝向 → 解剖语义的 yaw/pitch/roll（度）。见 `angles_from_matrices`。"""
    matrices = joint_matrices(skeleton, "neck") @ joint_matrices(skeleton, "face")
    return angles_from_matrices(matrices, anatomy)


def angles_from_matrices(matrices: np.ndarray, anatomy: AnatomyFrame) -> dict[str, np.ndarray]:
    """头部朝向矩阵 (n,3,3) → 解剖语义的 yaw/pitch/roll（度）。

    `roll_naive` 一并返回，仅供与旧公式对照（它带 pitch×yaw 泄漏，**不要用于定值**）。

    Returns:
        yaw: >0 转向自己的右侧 · pitch: >0 抬头 · roll: >0 头顶倒向自己的右侧。
    """
    forward = matrices @ anatomy.forward
    up_head = matrices @ anatomy.up

    right0, forward0, up0 = anatomy.right, anatomy.forward, anatomy.up
    yaw = np.degrees(np.arctan2(forward @ right0, forward @ forward0))
    pitch = np.degrees(np.arcsin(np.clip(forward @ up0, -1.0, 1.0)))

    # swing-twist：把世界上方投影到与当前视线垂直的平面 ⇒ 该视线下的"零 roll 参考上方"
    projection = up0[None, :] - forward * (forward @ up0)[:, None]
    norms = np.linalg.norm(projection, axis=1, keepdims=True)
    up_ref = projection / np.maximum(norms, 1e-9)
    right_ref = anatomy.handedness * np.cross(forward, up_ref)
    roll = np.degrees(
        np.arctan2(
            np.einsum("ij,ij->i", up_head, right_ref), np.einsum("ij,ij->i", up_head, up_ref)
        )
    )

    naive = np.degrees(np.arctan2(up_head @ right0, up_head @ up0))
    return {
        "yaw": np.unwrap(np.radians(yaw)) * 180 / math.pi,
        "pitch": pitch,
        "roll": np.unwrap(np.radians(roll)) * 180 / math.pi,
        "roll_naive": np.unwrap(np.radians(naive)) * 180 / math.pi,
    }
