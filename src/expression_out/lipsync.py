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

# 唯一允许语音流写入的嘴部参数集合（Live2D 标准参数：+ 为张嘴）。
# v1 只驱动嘴部开合一维；若加 ParamMouthForm 须同步更新融合锚点测试与跨仓规范。
MOUTH_PARAMS: tuple[str, ...] = ("ParamMouthOpenY",)

# 包络平滑：单极点 attack/release 不对称——嘴张开快（跟上爆破音）、闭合稍慢（避免抖动）。
# 系数按 20fps 帧节奏取值；确定性常数，非可调 env（观感常数，改动走真机盲测口径）。
ATTACK_ALPHA = 0.6
RELEASE_ALPHA = 0.25

# 噪声门：低于峰值标度这一比例的能量视为静音（嘴闭合），滤掉底噪让静音段真正闭嘴。
NOISE_GATE_RATIO = 0.06

# 归一标度取包络的该分位数（而非 max）：单个爆破峰不该把整句其余部分压瘪。
NORM_PERCENTILE = 0.95


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
    """平滑+归一后的包络 → `MOUTH_PARAMS` 关键帧轨迹（形状同 `motion_synth`）。

    归一按整句 `NORM_PERCENTILE` 分位标度（轻声句也张得开嘴）。噪声门作用在
    **原始包络**上：静音帧必闭嘴——release 平滑只塑造有声段内的衰减，不许外溢到
    静音段拖出"余音张嘴"（实测拖尾 0.32，闭嘴要 6+ 帧）。全静音（标度 0）输出
    全零轨迹而非除零。
    """
    smoothed = smooth_envelope(envelope)
    scale = _percentile(smoothed, NORM_PERCENTILE)
    gate = scale * NOISE_GATE_RATIO
    track: list[dict[str, object]] = []
    for i, value in enumerate(smoothed):
        if scale <= 0.0 or envelope[i] < gate:
            opening = 0.0
        else:
            opening = min(1.0, value / scale)
        track.append(
            {
                "t_ms": int(round(i * 1000.0 / fps)),
                "params": {MOUTH_PARAMS[0]: round(opening, 4)},
            }
        )
    return track


def mouth_track_from_wav(wav_bytes: bytes, fps: float) -> list[dict[str, object]]:
    """wav → 口型关键帧轨迹（`energy_envelope` + `envelope_to_mouth_track` 的组合便捷口）。"""
    return envelope_to_mouth_track(energy_envelope(wav_bytes, fps), fps)
