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


class FacsModel(Protocol):
    def predict_facs(self, valence: float, arousal: float) -> dict[str, float]: ...


class CompositeChannelDecoder:
    """注入若干真通道模型；predict_channels 用真模型覆盖对应通道，其余回退解析占位。

    `coping_potential`、`facs_extended`、`k_arousal`、`k_coping` 经 `__init__` 注入，
    透传给 `decode_channels`；`predict_channels(v, a)` 公开签名不变（CS 席约束 #4）。

    系数注入（W4 config-only-via-env）：
        k_arousal: ⚖ AU05/07 arousal 增益（默认 1.5=现值，零回归）。
        k_coping:  ⚖ 区分性 AU coping 增益（默认 1.2=现值，零回归）。
        方向由议会定，不可改符号/单调；仅幅度可调（via ZERO_FACS_K_AROUSAL/K_COPING env）。
    """

    def __init__(
        self,
        *,
        prosody_model: ProsodyModel | None = None,
        physiology_model: PhysiologyModel | None = None,
        facs_model: FacsModel | None = None,
        coping_potential: float = 0.0,
        facs_extended: bool = False,
        k_arousal: float = 1.5,
        k_coping: float = 1.2,
    ) -> None:
        self.prosody_model = prosody_model
        self.physiology_model = physiology_model
        self.facs_model = facs_model
        self.coping_potential = coping_potential
        self.facs_extended = facs_extended
        self.k_arousal = k_arousal
        self.k_coping = k_coping

    def predict_channels(self, valence: float, arousal: float) -> dict[str, Any]:
        channels = decode_channels(
            (valence, arousal),
            coping_potential=self.coping_potential,
            facs_extended=self.facs_extended,
            k_arousal=self.k_arousal,
            k_coping=self.k_coping,
        )  # 解析占位提供全部 4 通道
        if self.prosody_model is not None:
            channels["prosody"] = self.prosody_model.predict_prosody(valence, arousal)
        if self.physiology_model is not None:
            channels["physiology"] = self.physiology_model.predict_physiology(valence, arousal)
        if self.facs_model is not None:
            # W2 TODO：真 FacsDecoder 路径当前不接 coping，待 facs_decoder_ext.pt 就绪时
            # 为 FacsModel 协议加 coping_potential 参（breaking·下一轮工程门）。
            channels["facs_au"] = self.facs_model.predict_facs(valence, arousal)
        channels["text_label"] = text_label(valence, arousal)
        return channels
