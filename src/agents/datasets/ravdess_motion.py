"""RAVDESS 头部运动 DataLoader：(v,a) ← 文件名情绪码，Y ← OpenFace 逐帧头姿的运动学统计量。

获取数据（无 EULA，Zenodo 开放直下）：
  1. 下载 https://zenodo.org/records/3255102 的 `FacialTracking_Actors_01-24.zip`（417MB）
  2. 解压到 `data/ravdess_motion/`（已 gitignore）。目录下是 Actor_xx/*.csv（2452 个试验）。
  3. 训练：python -m scripts.train_motion --root data/ravdess_motion

CSV 由 OpenFace 2.1.0 产出，712 列；本模块只取头姿旋转与眨眼两组：
  - `pose_Rx/Ry/Rz`：绕 X/Y/Z 轴旋转，**弧度**。用旋转而非 `pose_T*`（位置，mm），
    因为对面 Live2D 的驱动参数 `FaceAngleX/Y/Z` 是角度语义。
  - `AU45_c`：眨眼存在性（0/1），供实测眨眼间隔分布（合成器 idle 基线用）。
  - `confidence`：跟踪质量。`success` 列**在本数据集中不存在**（该版导出省略），故按
    「存在才判、缺失即视为成功」处理——**写成硬判据会让整份数据一条都读不出来**（2026-08-05
    实测踩到）。

⚠ **质量过滤在本数据集上近乎惰性，别把它当成已做过数据清洗**（实测 2452 条抽样）：
`confidence` 全库取值只有 {0.77, 0.82, 0.88, 0.93, 0.98} 五档，**最低档也高于**默认阈
`MIN_CONFIDENCE`；`frame` 序号连续无跳号 ⇒ 作者未剔除失败帧、也没有失败帧留下的空洞。
过滤逻辑保留是为了兼容**其它** OpenFace 导出（那里 success/低置信是真会出现的），
在本数据集上它实际不淘汰任何帧。相应的单测用的是合成低置信数据——测的是机制可用，
**不代表真数据里存在被它拦下的坏帧**。

文件名与音频版同构（`modality-vocalChannel-emotion-intensity-statement-repetition-actor`），
故情绪码解析与 (v,a) 锚点**复用** `ravdess` 模块的 `parse_emotion_code` / `EMOTION_CODE_TO_VA`，
不复制常量。第 7 段是 actor id，供 **leave-actor-out** 留出——见 `load_ravdess_motion`
的 `return_groups`。

⚠ 数据性质限界：RAVDESS 是**演员摆拍（posed）**、坐姿念固定台词。原作者自陈表演型表达
较自发表达夸大，故由此学到的「唤醒→幅度」斜率上界偏陡，须在下游做幅度衰减校准。
议会设计门判定与文献见 `notes/2026-08-05-motion-council-design-gate.md`。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Literal, overload

import torch

from src.agents.datasets.ravdess import EMOTION_CODE_TO_VA, parse_emotion_code

# OpenFace 列名（现场核验自官方 Output-Format 文档）
POSE_ROTATION_COLUMNS = ("pose_Rx", "pose_Ry", "pose_Rz")
BLINK_COLUMN = "AU45_c"
SUCCESS_COLUMN = "success"
CONFIDENCE_COLUMN = "confidence"
TIMESTAMP_COLUMN = "timestamp"

# 跟踪质量阈：低于此置信度的帧连同 success==0（若该列存在）一并丢弃。
# ⚠ 本数据集实测 confidence ∈ {0.77, 0.82, 0.88, 0.93, 0.98} ⇒ 此阈**从不淘汰任何帧**。
# 保留它是为兼容其它 OpenFace 导出；调高会真的开始丢数据，须先量化收益再动。
MIN_CONFIDENCE = 0.75

# 一条轨迹至少要这么多有效帧才纳入（少于此则统计量不可靠，整条跳过）
MIN_VALID_FRAMES = 30

# 运动学统计量的名字，与 `extract_motion_stats` 返回值逐位对应。
# ⚠ 议会裁定：**不含方向/朝向维**——「效价→趋近-回避朝向」被判构念错误（愤怒是负效价但
# 趋近取向），改用 coping_potential 门控方向则头部运动学证据不足（专测实证为零结果）。
# 两条候选机制都不够 ⇒ v1 整维移出。详见议会纪要。
#
# K=3：原第四维 `nonstationarity` 已按 T1.2 实测**合并删除**——它与 `max_jerk` 在全库
# 2452 条上 r=+0.817（议会定的合并阈是 |r|>0.8），保留等于把同一信息计两次、白占容量
# （8 锚点下容量本就紧张）。留 `max_jerk` 而非 `nonstationarity`，因前者有直接文献支撑
# （Pollick 等的 jerkiness 构念），后者只是它的相关量。
MOTION_STAT_KEYS = ("amplitude_rms", "angular_speed_mean", "max_jerk")

# 强度码（文件名第 4 段）→ arousal 幅度缩放。
#
# ⚠ **这是 2026-08-05 真数据实测后的修正，不是原设计**。原设计只用情绪码→(v,a) 锚点做监督，
# 实测发现该 arousal 坐标与运动学**几乎无关**（amplitude_rms r=+0.040、speed r=+0.282），
# 而 RAVDESS 自带的 normal/strong 强度操纵效应清晰（全库 语音 normal→strong：
# amp 0.0607→0.0787、speed 0.291→0.380、jerk 1219→1830；**同演员×同情绪配对** n=168 中
# strong 更大的占比 amp 66.1% / speed 70.2% / jerk 64.9%，50% 为无效应基线）。
# 另中性 vs 有情绪差 2 倍以上（amp 0.0301 vs 0.0697）。
#
# 解释：情绪码→(v,a) 把 happy/surprised/fearful/angry 的 arousal 挤在 +0.5~+0.7 一小段，
# 却完全丢掉了「同一情绪演得多用力」这一真实变异源——而后者正是演员实际改变头动的维度。
#
# 处理：**不改共享的 `EMOTION_CODE_TO_VA`**（韵律通道在用，改了就是跨通道回归），
# 只在本模块按强度对 arousal 分量做缩放；valence 不动（强度是唤醒操纵，非效价操纵）。
INTENSITY_AROUSAL_SCALE: dict[str, float] = {
    "01": 0.75,  # normal
    "02": 1.25,  # strong
}


def _rotation_series(rows: list[dict[str, str]]) -> tuple[list[list[float]], list[float]]:
    """抽出通过质量过滤的头姿旋转序列与对应时间戳（秒）。

    Returns:
        (frames, timestamps)：frames[i] 是第 i 个有效帧的 [Rx, Ry, Rz]（弧度）。
        丢帧不做插值补齐——补出来的值会被当成真实运动进统计量；宁可段变短。
    """
    frames: list[list[float]] = []
    timestamps: list[float] = []
    for row in rows:
        try:
            # success 缺失即视为成功——本数据集根本没有这列，写成硬判据会丢光全部帧。
            success = row.get(SUCCESS_COLUMN)
            if success is not None and int(float(success)) != 1:
                continue
            if float(row[CONFIDENCE_COLUMN]) < MIN_CONFIDENCE:
                continue
            frames.append([float(row[col]) for col in POSE_ROTATION_COLUMNS])
            timestamps.append(float(row[TIMESTAMP_COLUMN]))
        except (KeyError, ValueError):
            continue  # 列缺失或非数值 → 该帧不可用，跳过（不让坏行终止整条轨迹）
    return frames, timestamps


def _derivative(series: list[float], timestamps: list[float]) -> list[float]:
    """按实际时间戳做一阶前向差分（不假设等间隔——丢帧过滤后间隔不均匀）。"""
    out: list[float] = []
    for i in range(1, len(series)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0.0:
            continue
        out.append((series[i] - series[i - 1]) / dt)
    return out


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def extract_motion_stats(frames: list[list[float]], timestamps: list[float]) -> list[float] | None:
    """把一条头姿旋转序列压成 K 维运动学统计量，与 `MOTION_STAT_KEYS` 逐位对应。

    有效帧不足 `MIN_VALID_FRAMES` 时返回 None（调用方跳过该条），不返回全零——
    全零会被当成"这个情绪下头不动"的真实观测混进训练集。

    三个统计量（议会收敛 + T1.2 实测降维后的结果，每维有现场核验的文献支撑）：
    - `amplitude_rms`：三轴去均值后的合成幅度 RMS —— 唤醒→幅度。
    - `angular_speed_mean`：合成角速度均值 —— 唤醒→速度。
    - `max_jerk`：三阶导数峰值 —— Pollick 等指出的 jerkiness 构念，捕捉起始动力学；
      纯平均量会把这层瞬时信息平均掉。
    ~~`nonstationarity`~~ 已删：与 `max_jerk` 实测 r=+0.817（>议会合并阈 0.8），见
    `MOTION_STAT_KEYS` 注释。
    """
    if len(frames) < MIN_VALID_FRAMES:
        return None

    axes = list(zip(*frames, strict=True))  # 3 × N，按轴拆开
    means = [sum(axis) / len(axis) for axis in axes]
    centered = [[v - mean for v in axis] for axis, mean in zip(axes, means, strict=True)]

    # 合成幅度：逐帧三轴欧氏范数的 RMS（不是三轴各自 RMS 再平均——那会低估斜向运动）
    magnitudes = [
        math.sqrt(sum(centered[ax][i] ** 2 for ax in range(3))) for i in range(len(frames))
    ]
    amplitude_rms = _rms(magnitudes)

    velocities = [_derivative(list(axis), timestamps) for axis in axes]
    if not velocities[0]:
        return None
    speed = [
        math.sqrt(sum(velocities[ax][i] ** 2 for ax in range(3))) for i in range(len(velocities[0]))
    ]
    angular_speed_mean = sum(speed) / len(speed)

    # jerk：对合成角速度再求两阶导（速度→加速度→jerk），取绝对值峰值
    speed_timestamps = timestamps[1:]
    accel = _derivative(speed, speed_timestamps)
    jerk = _derivative(accel, speed_timestamps[1:]) if accel else []
    max_jerk = max((abs(j) for j in jerk), default=0.0)

    return [amplitude_rms, angular_speed_mean, max_jerk]


def blink_intervals(rows: list[dict[str, str]]) -> list[float]:
    """从 `AU45_c` 抽眨眼间隔（秒），供实测 IBI 分布。

    取每段连续 AU45_c==1 的**起始**时刻作为一次眨眼，相邻起始之差即间隔。
    文献称自发眨眼**非泊松**（泊松要求均值=方差，实证不满足）；本函数让合成器的 IBI
    参数可由数据定，而不是只取文献区间。
    ⚠ 本数据是**说话中**的录制，眨眼受言语产出调制，与静息 IBI 不可直接等同。
    """
    intervals: list[float] = []
    previous_start: float | None = None
    was_closed = False
    for row in rows:
        try:
            closed = int(float(row[BLINK_COLUMN])) == 1
            stamp = float(row[TIMESTAMP_COLUMN])
        except (KeyError, ValueError):
            continue
        if closed and not was_closed:  # 上升沿 = 一次眨眼开始
            if previous_start is not None:
                intervals.append(stamp - previous_start)
            previous_start = stamp
        was_closed = closed
    return intervals


def parse_actor_id(filename: str) -> str | None:
    """从 RAVDESS 文件名解析演员 id（第 7 段）；非法格式返回 None。

    leave-actor-out 留出必需：X 只有 8 个不同取值（`EMOTION_CODE_TO_VA` 的 8 个锚点），
    从 X 反推不出这一行来自哪个演员。而头姿的个体差异（习惯性倾角、幅度、视线模式）很大
    程度上由解剖与人格决定——按行随机切分会让同一演员跨切分泄漏，val 失去意义。
    同源教训：本仓韵律通道实测 61% 方差来自说话人身份。
    """
    parts = Path(filename).stem.split("-")
    if len(parts) < 7:
        return None
    return parts[6]


def parse_intensity_code(filename: str) -> str | None:
    """从文件名解析强度码（第 4 段：01 normal / 02 strong）；非法或未知返回 None。"""
    parts = Path(filename).stem.split("-")
    if len(parts) < 4:
        return None
    code = parts[3]
    return code if code in INTENSITY_AROUSAL_SCALE else None


def affect_anchor(emotion_code: str, intensity_code: str | None) -> tuple[float, float]:
    """情绪码 + 强度码 → (valence, arousal)。

    基锚点取自共享的 `EMOTION_CODE_TO_VA`（**不改它**——韵律通道在用）；arousal 分量按
    强度缩放并钳回 [-1,1]。缩放**只作用于 arousal**：RAVDESS 的 normal/strong 是「演得多
    用力」的唤醒操纵，不是效价操纵。

    强度码缺失（如中性只有 normal 一档，或文件名异常）→ 不缩放，退回基锚点。
    实测依据见 `INTENSITY_AROUSAL_SCALE` 注释。
    """
    valence, arousal = EMOTION_CODE_TO_VA[emotion_code]
    scale = INTENSITY_AROUSAL_SCALE.get(intensity_code or "", 1.0)
    return valence, max(-1.0, min(1.0, arousal * scale))


@overload
def load_ravdess_motion(
    root: str | Path,
    *,
    limit: int | None = ...,
    return_groups: Literal[False] = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def load_ravdess_motion(
    root: str | Path,
    *,
    limit: int | None = ...,
    return_groups: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]: ...


def load_ravdess_motion(
    root: str | Path,
    *,
    limit: int | None = None,
    return_groups: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[str]]:
    """递归加载 OpenFace CSV，返回 (X=(v,a), Y=运动学统计量) float32 张量。

    X ∈ [-1,1]^2，Y 为 `MOTION_STAT_KEYS` 对应的 K 维原始量纲统计量（**未归一化**——
    归一化交训练脚本，避免 loader 与训练两处各自缩放导致口径分叉）。
    无可解析 CSV 时抛 FileNotFoundError。

    `return_groups=True` 时额外返回每行的演员 id（长度 == X 行数），供 leave-actor-out。
    默认 False，返回值逐字保持与其它 loader 同形（零回归）。
    """
    files = sorted(Path(root).rglob("*.csv"))
    if limit is not None:
        files = files[:limit]

    xs: list[list[float]] = []
    ys: list[list[float]] = []
    groups: list[str] = []
    for path in files:
        code = parse_emotion_code(path.name)
        actor = parse_actor_id(path.name)
        if code is None or actor is None:
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(fh)]
        frames, timestamps = _rotation_series(rows)
        stats = extract_motion_stats(frames, timestamps)
        if stats is None:
            continue  # 有效帧不足，整条跳过（不补零）
        xs.append(list(affect_anchor(code, parse_intensity_code(path.name))))
        ys.append(stats)
        groups.append(actor)

    if not xs:
        raise FileNotFoundError(f"未在 {root} 找到可解析的 RAVDESS 头姿 CSV（检查解压路径与列名）")
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    if return_groups:
        return x, y, groups
    return x, y
