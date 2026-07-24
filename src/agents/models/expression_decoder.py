"""ExpressionDecoder：把 (valence, arousal) → 表达通道 的解析占位蒸馏成可训练 torch 网络。

向量布局（11 维，全部归一化到 [0,1]，与 affect_math 的解析映射一致）：
  0 AU04 · 1 AU06 · 2 AU12 · 3 AU15 · 4 au_intensity
  5 hr_n · 6 gsr · 7 pupil_n（门关 legacy）/ temperature_n（门开 canonical）
  8 speech_rate_n · 9 pitch_n · 10 energy

idx7 双语义（canonical_physiology 门控）：
  门关（默认）= pupil_n：归一化瞳孔，反归一化 pupil_mm=3+2·vec[7]（legacy 占位域）。
  门开 = temperature_n：归一化温度，目标值 (36−3·clamp(|a|)−30)/10，反归一化
    temperature_c=30+10·vec[7]（与 PhysiologyDecoder.predict_physiology 同域 [30,40]）。
  ⚠ 旧权重 expression_decoder.pt 在 canonical 路径复用 idx7 会被误解为 temperature；
  canonical 路径须重训生成新权重（scripts/train_expression.py 说明）。

`affect_to_vector` 是蒸馏目标（解析"真值"）；`vector_to_channels` 把网络输出反归一化回
与 `affect_math.decode_channels` 同构的通道字典（不含 text_label，由调用方补）。
hr(idx5)/sc(idx6) 门开时对齐 canonical 量纲（hr=50+70·vec[5]，sc=20·vec[6]）；
门关时逐字 legacy（hr=70+40·vec[5]，sc=vec[6]，pupil_mm=3+2·vec[7]）。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.agents.affect_math import clamp, text_label

CHANNEL_DIM = 11


def affect_to_vector(
    valence: float,
    arousal: float,
    *,
    canonical_physiology: bool = False,
) -> list[float]:
    """解析"真值"：(v,a) → 11 维归一化通道向量（蒸馏目标）。

    canonical_physiology=False（默认）：逐字 legacy 目标，旧权重兼容默认路径，零回归。
    canonical_physiology=True：canonical 目标，idx5/6/7 改为 canonical 口径（须重训权重）：
      idx5 hr_n   = clamp(0.5*(1+arousal))        → 反归一化 50+70·vec[5] ∈[50,120]
      idx6 gsr_n  = clamp(|arousal|)              → 反归一化 20·vec[6] ∈[0,20]μS
      idx7 temp_n = (36−3·clamp(|arousal|)−30)/10 → 反归一化 30+10·vec[7] ∈[33,36]⊂[30,40]
    与 decode_channels(canonical_physiology=True) 保持源码一致（guide:140 三处同步）。
    """
    if canonical_physiology:
        # canonical 归一目标：与 decode_channels 门开公式方向/域对齐
        # idx5 hr_n：全域覆盖（Ekman/Levenson 1983 负高唤醒同样升 HR）
        hr_n = clamp(0.5 * (1.0 + arousal), 0.0, 1.0)
        # idx6 gsr_n：EDA 与 |arousal| 正相关（B-2 精神），归一到 [0,1]（反归一化 20× 得 μS）
        gsr_n = clamp(abs(arousal), 0.0, 1.0)
        # idx7 temp_n：温度归一，公式 (36−3·|arousal|−30)/10 使反归一化 30+10·vec 得 [33,36]°C
        # 与真 PhysiologyDecoder.predict_physiology 同域（sigmoid→[30,40]·高唤醒→降温方向对齐）
        temp_n = clamp((36.0 - 3.0 * clamp(abs(arousal), 0.0, 1.0) - 30.0) / 10.0, 0.0, 1.0)
        return [
            clamp(-0.6 * valence, 0.0, 1.0) if valence < 0 else 0.0,  # AU04
            clamp(0.6 * valence, 0.0, 1.0) if valence >= 0 else 0.0,  # AU06
            clamp(valence, 0.0, 1.0) if valence >= 0 else 0.0,  # AU12
            clamp(-valence, 0.0, 1.0) if valence < 0 else 0.0,  # AU15
            clamp(abs(arousal), 0.0, 1.0),  # au_intensity
            hr_n,  # idx5 hr_n（canonical）
            gsr_n,  # idx6 gsr_n（canonical，μS 归一）
            temp_n,  # idx7 temperature_n（canonical；门关为 pupil_n）
            (clamp(arousal, -1.0, 1.0) + 1.0) / 2.0,  # speech_rate_n
            (clamp(arousal, -1.0, 1.0) + 1.0) / 2.0,  # pitch_n（B-8/A-P0-C：F0 随唤醒）
            clamp(0.5 + 0.5 * arousal, 0.0, 1.0),  # energy
        ]
    # 门关（默认）：逐字 legacy 目标，零回归
    return [
        clamp(-0.6 * valence, 0.0, 1.0) if valence < 0 else 0.0,  # AU04
        clamp(0.6 * valence, 0.0, 1.0) if valence >= 0 else 0.0,  # AU06
        clamp(valence, 0.0, 1.0) if valence >= 0 else 0.0,  # AU12
        clamp(-valence, 0.0, 1.0) if valence < 0 else 0.0,  # AU15
        clamp(abs(arousal), 0.0, 1.0),  # au_intensity
        clamp(arousal, 0.0, 1.0),  # hr_n（legacy）
        clamp(
            abs(arousal), 0.0, 1.0
        ),  # gsr（议会 B-2/A-P0-D：EDA 随 |arousal|，与 decode_channels 一致）
        clamp(arousal, 0.0, 1.0),  # pupil_n（legacy；门开为 temperature_n）
        (clamp(arousal, -1.0, 1.0) + 1.0) / 2.0,  # speech_rate_n
        (clamp(arousal, -1.0, 1.0) + 1.0) / 2.0,  # pitch_n（议会 B-8/A-P0-C：F0 随唤醒非效价）
        clamp(0.5 + 0.5 * arousal, 0.0, 1.0),  # energy
    ]


def vector_to_channels(
    vec: list[float],
    *,
    canonical_physiology: bool = False,
) -> dict[str, Any]:
    """把 11 维归一化向量反归一化为通道字典（结构对齐 decode_channels，不含 text_label）。

    canonical_physiology=False（默认）：逐字 legacy 反归一化，零回归。
    canonical_physiology=True：physiology 块改为 canonical 口径，与 decode_channels 门开一致：
      heart_rate_bpm = 50 + 70·vec[5]  ∈ [50,120] bpm
      skin_conductance = 20·vec[6]      ∈ [0,20] μS
      temperature_c = 30 + 10·vec[7]   ∈ [30,40] °C（目标 [33,36]⊂[30,40]）
      pupil_mm 删除（canonical 路径·WESAD 无此信号）。
    两路径 physiology 键集与 decode_channels 同步（guide:140 三处同步·CS NEEDS-CHANGES）。
    """
    if canonical_physiology:
        physiology: dict[str, float] = {
            "heart_rate_bpm": 50.0 + 70.0 * vec[5],  # canonical：全域覆盖 [50,120]
            "skin_conductance": 20.0 * vec[6],  # canonical：μS 量纲 [0,20]
            "temperature_c": 30.0 + 10.0 * vec[7],  # canonical：[30,40]°C（目标 [33,36]）
        }
    else:
        physiology = {
            "heart_rate_bpm": 70.0 + 40.0 * vec[5],  # legacy：[70,110]
            "skin_conductance": vec[6],  # legacy：[0,1]（无量纲）
            "pupil_mm": 3.0 + 2.0 * vec[7],  # legacy：[3,5] mm
        }
    return {
        "facs_au": {
            "AU04": vec[0],
            "AU06": vec[1],
            "AU12": vec[2],
            "AU15": vec[3],
            "intensity": vec[4],
        },
        "physiology": physiology,
        "prosody": {
            "speech_rate": 1.0 + 0.5 * (2.0 * vec[8] - 1.0),
            "pitch": 1.0 + 0.3 * (2.0 * vec[9] - 1.0),
            "energy": vec[10],
        },
        # 整向量通路把内部 [0,1] 反归一化回**倍率口径**（见上 speech_rate/pitch 的 1.0±… 映射）
        # → "ratio"（zero-link Q1 拍板 2026-07-14）。与解析占位同量纲；仅专用 ProsodyDecoder
        # 出 normalized。兄弟键，不入 prosody 子 dict（保通道纯 3 值·零回归）。
        "prosody_scale": "ratio",
    }


class ExpressionDecoder(nn.Module):
    """(v,a) → 11 维通道向量的 MLP，输出经 sigmoid 落在 [0,1]。"""

    def __init__(self, hidden: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, CHANNEL_DIM), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_channels(
        self, valence: float, arousal: float, canonical_physiology: bool = False
    ) -> dict[str, Any]:
        """单点推理，返回与 decode_channels 同构的通道字典（含 text_label）。

        `canonical_physiology`（默认 False=legacy 口径·与旧权重兼容·零回归）透传给
        `vector_to_channels`；canonical 路径须同时传 True **且**权重为 canonical 重训版
        （旧 legacy 权重的 idx7 是 pupil_n，接 canonical 反归一化会被误解为 temperature）。
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor([[valence, arousal]], dtype=torch.float32)
            vec = self(x)[0].tolist()
        channels = vector_to_channels(vec, canonical_physiology=canonical_physiology)
        channels["text_label"] = text_label(valence, arousal)
        return channels


def load_decoder(path: str, hidden: int = 32, num_layers: int = 2) -> ExpressionDecoder:
    """从权重文件加载已训练的解码器。"""
    model = ExpressionDecoder(hidden=hidden, num_layers=num_layers)
    # weights_only=True：只反序列化张量/state_dict，不执行任意 pickle（对齐 load_facs_decoder；
    # PyTorch ≥2.4 将默认此值）。加载的是 state_dict，安全无副作用。
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model
