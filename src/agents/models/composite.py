"""CompositeChannelDecoder：逐通道渐进真网络化。

在 affect_math 解析占位之上，用已注入的"真通道模型"覆盖对应通道；未注入的通道回退占位。
满足 ExpressionAgent 的 ChannelDecoder 协议（predict_channels）。每接一个真实数据集
（RAVDESS→韵律、WESAD→生理、AffectNet/DISFA→表情），加一个对应模型即可，契约不变。
"""

from __future__ import annotations

from typing import Any, Protocol

from src.agents.affect_math import decode_channels, text_label


class ProsodyModel(Protocol):
    def predict_prosody(self, valence: float, arousal: float) -> dict[str, float]: ...


class PhysiologyModel(Protocol):
    def predict_physiology(self, valence: float, arousal: float) -> dict[str, float]: ...


class CompositeChannelDecoder:
    """注入若干真通道模型；predict_channels 用真模型覆盖对应通道，其余回退解析占位。"""

    def __init__(
        self,
        *,
        prosody_model: ProsodyModel | None = None,
        physiology_model: PhysiologyModel | None = None,
    ) -> None:
        self.prosody_model = prosody_model
        self.physiology_model = physiology_model

    def predict_channels(self, valence: float, arousal: float) -> dict[str, Any]:
        channels = decode_channels((valence, arousal))  # 解析占位提供全部 4 通道
        if self.prosody_model is not None:
            channels["prosody"] = self.prosody_model.predict_prosody(valence, arousal)
        if self.physiology_model is not None:
            channels["physiology"] = self.physiology_model.predict_physiology(valence, arousal)
        channels["text_label"] = text_label(valence, arousal)
        return channels
