"""口型合成 v1：wav 能量包络 → 嘴部关键帧轨迹（纯函数、无 I/O、确定性）。

**这是 v1 的刻意简化**：嘴开合幅度只跟语音能量走，不区分音素（"啊"与"呜"同形）。
v2（Bert-VITS2 隐变量 z → 音素级口型映射头）落地前必须过科学家议会门 + 独立 PRP
（见 `PRP/speech-output/prp.md` A7），别在本文件上"顺手"加音素逻辑。

## 融合规则（锚点测试钉死，改动须过评审）

`MOUTH_PARAMS` 是**唯一允许语音流写入**的渲染参数集合——嘴归语音、其余归情绪：

- 本模块输出的关键帧只含 `MOUTH_PARAMS` 内的键（`tests/test_lipsync_fusion_anchor.py`）；
- 反向：`agents.motion_synth` 的情绪动作流永不写 `MOUTH_PARAMS`（同一测试文件钉反向）。

关键帧形状与 `motion_synth` 输出同构（渲染端 `params_animate` 现行解析）：
`[{"t_ms": int, "params": {参数名: 值}}]`，`t_ms` 从 0 起算 = 音频首采样时刻。
"""

from __future__ import annotations

import contextlib
import io
import math
import wave
from array import array

# 唯一允许语音流写入的嘴部参数集合。
# ⚠ 命名空间是 **VTS 输入参数**（params_list 实测 127 个，嘴部开合 = `MouthOpen`∈[0,1]），
# 不是 Live2D 输出参数（`ParamMouthOpenY`）——2026-08-14 首次真机联调实测：后者被渲染端
# `[vtsb:invalid_params] 参数缺席` 拒收，音频照播、嘴不动。
# v1 只驱动开合一维；若加 `MouthSmile` 须同步更新融合锚点测试与跨仓规范。
MOUTH_PARAMS: tuple[str, ...] = ("MouthOpen",)

# 包络平滑：单极点 attack/release 不对称——嘴张开快（跟上爆破音）、闭合稍慢（避免抖动）。
# 系数按 20fps 帧节奏取值；确定性常数，非可调 env（观感常数，改动走真机盲测口径）。
# 2026-08-14 三轮真机盲测定档：一轮「一味一张一闭」⇒ 加快跟随；二轮「频率快、机械」⇒
# 回落 0.55/0.35；三轮「张闭仍快、回落太快、僵硬」⇒ **0.35/0.18**。能这样一路调慢而不
# 回到一轮的「恒张」，是因为起伏**深度**已交给对比度拉伸（FLOOR_PERCENTILE 重标定每句
# 的谷底）、平滑系数只管起伏**速度**——两轴解耦，速度可独立压慢。
ATTACK_ALPHA = 0.35
RELEASE_ALPHA = 0.18

# 低能量忽略门：低于峰值标度这一比例的能量一律视为静音（嘴渐合）。初值 0.06 只滤底噪；
# 五轮盲测（用户提议「忽略一些低能量的部分」）提到 0.20；六轮「去掉的有点狠」回落
# 0.15；七轮「再降低一点」定 0.11——弱音节尾/气声仍忽略，次强段驱动完整保留。
NOISE_GATE_RATIO = 0.11

# 归一标度取包络的该分位数（而非 max）：单个爆破峰不该把整句其余部分压瘪。
NORM_PERCENTILE = 0.95

# 对比度拉伸的"闭合参考"分位（有声帧内取）：低于它的有声帧映射到近闭合。
# 动机（一轮盲测）：中文语流能量连续，直接 v/scale 会把大半有声帧顶到高位
# （实测有声均值 0.78）——音节间的浅谷要拉伸成可见的闭合动作才有"说话感"。
# 二轮盲测「机械」后 0.30 → 0.20：拉伸变浅，浅谷不再砸到底。
FLOOR_PERCENTILE = 0.20

# 拉伸后的伽马（<1 提亮中段）：避免拉伸把中等能量压得太闭、观感变"嘴瓢"。
OPENING_GAMMA = 0.75

# 有声段的最小开度：**说话中不完全闭嘴**（人语流的唇闭只在爆破音瞬间，砸到 0 是方波感
# 「机械」的主因，二轮盲测定）；只有过不了噪声门的真静音帧才归零闭嘴。
VOICED_MIN_OPENING = 0.12

# 开度限速（四轮盲测「嘴部变化太快」定）：每帧开度变化的硬上界——平滑系数只能压
# 平均速度，能量突跳时单帧仍可蹦 0.3+；限速给速度一个**无条件上界**（20fps 下
# 闭→全张至少 0.5s），起张与回落对称生效，静音收口也顺此渐合而非弹回。
MAX_DELTA_PER_FRAME = 0.10


def energy_envelope(wav_bytes: bytes, fps: float) -> list[float]:
    """按 `fps` 帧节奏计算 wav 的 RMS 能量包络（未平滑、未归一）。

    只接受 PCM 16-bit；单声道直接用，双声道取左声道（确定性，不做混音）。
    其余格式抛 `ValueError`——调用方（表现层 sink）自行捕获降级，本函数保持纯粹。
    """
    if fps <= 0:
        raise ValueError(f"fps 必须为正：{fps}")
    with contextlib.closing(wave.open(io.BytesIO(wav_bytes), "rb")) as wf:
        sampwidth = wf.getsampwidth()
        channels = wf.getnchannels()
        rate = wf.getframerate()
        if sampwidth != 2:
            raise ValueError(f"只支持 PCM 16-bit wav，收到 sampwidth={sampwidth}")
        if channels not in (1, 2):
            raise ValueError(f"只支持单/双声道，收到 channels={channels}")
        raw = wf.readframes(wf.getnframes())
    samples = array("h")
    samples.frombytes(raw)
    if channels == 2:
        samples = samples[::2]
    if not samples:
        return []
    window = max(1, int(round(rate / fps)))
    envelope: list[float] = []
    for start in range(0, len(samples), window):
        chunk = samples[start : start + window]
        acc = 0.0
        for value in chunk:
            acc += float(value) * float(value)
        envelope.append(math.sqrt(acc / len(chunk)) / 32768.0)
    return envelope


def smooth_envelope(envelope: list[float]) -> list[float]:
    """attack/release 不对称单极点平滑（张嘴快、闭嘴稍慢）。"""
    smoothed: list[float] = []
    level = 0.0
    for value in envelope:
        alpha = ATTACK_ALPHA if value > level else RELEASE_ALPHA
        level += alpha * (value - level)
        smoothed.append(level)
    return smoothed


def _percentile(values: list[float], q: float) -> float:
    """确定性分位数（最近秩法）；空表返回 0。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))
    return ordered[rank]


def envelope_to_mouth_track(envelope: list[float], fps: float) -> list[dict[str, object]]:
    """平滑+归一+对比度拉伸后的包络 → `MOUTH_PARAMS` 关键帧轨迹（形状同 `motion_synth`）。

    归一按整句 `NORM_PERCENTILE` 分位标度（轻声句也张得开嘴）；有声帧再做**对比度
    拉伸**（`FLOOR_PERCENTILE` 低分位为闭合参考 + `OPENING_GAMMA`）——否则中文语流
    的连续能量会把嘴顶成"整句恒张、句尾才闭"（2026-08-14 真机反馈）。噪声门作用在
    **原始包络**上：静音帧必闭嘴——release 平滑只塑造有声段内的衰减，不许外溢到
    静音段拖出"余音张嘴"。全静音（标度 0）输出全零轨迹而非除零。
    """
    smoothed = smooth_envelope(envelope)
    scale = _percentile(smoothed, NORM_PERCENTILE)
    gate = scale * NOISE_GATE_RATIO
    voiced = [smoothed[i] for i in range(len(smoothed)) if scale > 0.0 and envelope[i] >= gate]
    floor = _percentile(voiced, FLOOR_PERCENTILE)
    span = _percentile(voiced, NORM_PERCENTILE) - floor
    track: list[dict[str, object]] = []
    limited = 0.0
    for i, value in enumerate(smoothed):
        if scale <= 0.0 or envelope[i] < gate:
            opening = 0.0
        elif span <= 0.0:
            opening = min(1.0, value / scale)  # 退化：能量近恒定时回落旧口径
        else:
            stretched = min(1.0, max(0.0, (value - floor) / span))
            opening = VOICED_MIN_OPENING + (1.0 - VOICED_MIN_OPENING) * stretched**OPENING_GAMMA
        delta = max(-MAX_DELTA_PER_FRAME, min(MAX_DELTA_PER_FRAME, opening - limited))
        limited += delta
        track.append(
            {
                "t_ms": int(round(i * 1000.0 / fps)),
                "params": {MOUTH_PARAMS[0]: round(limited, 4)},
            }
        )
    return track


def mouth_track_from_wav(wav_bytes: bytes, fps: float) -> list[dict[str, object]]:
    """wav → 口型关键帧轨迹（`energy_envelope` + `envelope_to_mouth_track` 的组合便捷口）。"""
    return envelope_to_mouth_track(energy_envelope(wav_bytes, fps), fps)
