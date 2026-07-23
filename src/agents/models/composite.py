"""CompositeChannelDecoder：逐通道渐进真网络化。

在 affect_math 解析占位之上，用已注入的"真通道模型"覆盖对应通道；未注入的通道回退占位。
满足 ExpressionAgent 的 ChannelDecoder 协议（predict_channels）。每接一个真实数据集
（RAVDESS→韵律、WESAD→生理、AffectNet/DISFA→表情），加一个对应模型即可，契约不变。
"""

from __future__ import annotations

from typing import Any, Protocol

from src.agents.affect_math import decode_channels, text_label

# coping 判别性 AU（议会定方向）：愤怒 AU23、恐惧 AU01/02/20。真 FacsDecoder 的
# predict_facs(v,a) 不吃 coping、学不到这四个 AU 的愤怒/恐惧分野（同 (v,a) 同点），
# 故注入真模型时这四个 AU 由解析占位的 coping 增量承担（C2 residual 叠加）。
_COPING_DRIVEN_AUS = ("AU23", "AU01", "AU02", "AU20")


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

    C2 residual 叠加（议会 C2 设计门 2026-07-14）：注入真 FacsModel 且 facs_extended 时，
    真模型出通用 AU 基准、coping 判别 AU（AU23/01/02/20）按 residual_alpha 与占位 coping 增量
    混合（真模型不吃 coping、学不到愤怒/恐惧分野）。默认 α=1.0 → 判别 AU 纯占位保分野。
    未注入 facs_model 时此路径不触发 = 零回归。（via ZERO_FACS_RESIDUAL_ALPHA env）

    ⚠ 调用方契约：
    - 注入的 `facs_model` 键集须与 `facs_extended` 对齐——`facs_extended=True` 注入
      `FacsDecoder(extended=True)`（13 键，任务 D 起含 AU17/AU26）、`facs_extended=False` 注入
      5 键旧模型。否则 legacy 路径会挂上 13 键 base，破坏下游对旧 5-AU 布局的假设。
    - 构造时 `facs_extended` 须与运行时 `state.facs_extended` **同源**（本项目从同一 `.env` →
      SessionConfig → state 流入）。否则 `predict_channels(v,a)`（用构造固定值）与经
      `predict_channels_coping`（用当轮 state 值）两路径会静默分叉。
    """

    def __init__(
        self,
        *,
        prosody_model: ProsodyModel | None = None,
        physiology_model: PhysiologyModel | None = None,
        facs_model: FacsModel | None = None,
        coping_potential: float = 0.0,
        facs_extended: bool = False,
        canonical_physiology: bool = False,
        k_arousal: float = 1.5,
        k_coping: float = 1.2,
        residual_alpha: float = 1.0,
    ) -> None:
        self.prosody_model = prosody_model
        self.physiology_model = physiology_model
        self.facs_model = facs_model
        self.coping_potential = coping_potential
        self.facs_extended = facs_extended
        # canonical_physiology：physiology 占位口径门控（ZERO_PHYSIOLOGY_CANONICAL_PLACEHOLDER）。
        # False（默认）→ legacy {hr[70,110]/sc[0,1]/pupil_mm[3,5]}，零回归；
        # True → canonical 议会 2026-07-23 公式 {hr[50,120]/sc μS[0,20]/temp_c[33,36]}。
        # physiology_model=None 时（回退占位）生效；注入真模型时 flag 无意义。
        # 与 state.canonical_physiology 同源（同一 env·构造期固定·非 per-turn）。
        self.canonical_physiology = canonical_physiology
        self.k_arousal = k_arousal
        self.k_coping = k_coping
        # residual_alpha：C2 residual 叠加系数（议会 C2 设计门 2026-07-14）∈[0,1]。
        # 注入真 FacsModel 且 facs_extended 时，coping 判别 AU = base*(1-α)+placeholder_coping*α。
        # 默认 1.0 → 判别 AU 纯占位（保 coping 分野）、其余 AU 用真模型基准；0 → 全用真模型
        # （coping 分野丢，退化为旧 W2 覆盖）。方向议会定、幅度工程可动。
        # env: ZERO_FACS_RESIDUAL_ALPHA。越界会让混合外推、AU 越 [0,1] → fail-fast。
        if not 0.0 <= residual_alpha <= 1.0:
            raise ValueError(
                f"residual_alpha 须 ∈[0,1]（避混合外推使 AU 越界），实为 {residual_alpha}"
            )
        self.residual_alpha = residual_alpha

    def predict_channels(self, valence: float, arousal: float) -> dict[str, Any]:
        """公开签名不变（CS 席约束 #4）：用构造时固定的 coping/facs_extended。"""
        return self.predict_channels_coping(
            valence, arousal, self.coping_potential, self.facs_extended
        )

    def predict_channels_coping(
        self,
        valence: float,
        arousal: float,
        coping_potential: float,
        facs_extended: bool,
    ) -> dict[str, Any]:
        """可选扩展（议会遗留 2 设计门 2026-07-14·方案 b）：接 **per-turn** coping/facs_extended。

        `ExpressionAgent._decode` 用 `hasattr` 探测此方法——有则传入当轮 `coping_potential_state`、
        无则回退 `predict_channels(v,a)`（旧 decoder 零改动）。**additive 非 breaking**：不改
        `ChannelDecoder`/`FacsModel` Protocol、不改 `predict_channels(v,a)` 签名（与 C2 当时 BLOCK
        的「改 predict_facs Protocol 签名」本质不同——那是 Protocol 层参数加法，所有实现被迫跟改）。
        注入真 facs_model 后 C2 residual 才拿到正确当轮 coping（否则固定 coping 使 α>0 无意义）。
        """
        channels = decode_channels(
            (valence, arousal),
            coping_potential=coping_potential,
            facs_extended=facs_extended,
            canonical_physiology=self.canonical_physiology,
            k_arousal=self.k_arousal,
            k_coping=self.k_coping,
        )  # 解析占位提供全部 4 通道
        if self.prosody_model is not None:
            prosody = self.prosody_model.predict_prosody(valence, arousal)
            # 防御性 [0,1] 校验（zero-link T4·2026-07-22）：normalized 契约要求三值 ∈[0,1]，
            # 供 MCP mapper 线性映射不外插。sigmoid 的 ProsodyDecoder 本满足；对任意注入的
            # ProsodyModel 实现越界即 fail-fast（同上 residual_alpha 越界 raise 口径 :81-84）。
            if any(not 0.0 <= v <= 1.0 for v in prosody.values()):
                raise ValueError(f"prosody_scale=normalized 要求三值 ∈[0,1]，实为 {prosody}")
            channels["prosody"] = prosody
            # 专用 ProsodyDecoder 出归一 [0,1]（sigmoid）→ 覆盖 decode_channels 占位的 "ratio"
            # 量纲标记为 "normalized"（zero-link Q1 拍板 2026-07-14·canonical 目标口径）。
            channels["prosody_scale"] = "normalized"
        if self.physiology_model is not None:
            channels["physiology"] = self.physiology_model.predict_physiology(valence, arousal)
        if self.facs_model is not None:
            # C2 residual 叠加（议会 C2 设计门 2026-07-14；替代旧 W2 TODO 的整通道覆盖）：
            # 真模型出通用 AU 基准（喜/悲/惊…从真脸学）；coping 判别 AU（AU23/01/02/20）——
            # 真模型 predict_facs(v,a) 不吃 coping、学不到愤怒/恐惧分野——按 residual_alpha 与
            # 解析占位的 coping 增量混合（占位值在上面 channels["facs_au"] 里，已带 coping 算出）。
            base = self.facs_model.predict_facs(valence, arousal)
            if facs_extended:
                placeholder = channels["facs_au"]
                alpha = self.residual_alpha
                for au in _COPING_DRIVEN_AUS:
                    if au in base and au in placeholder:
                        base[au] = base[au] * (1.0 - alpha) + placeholder[au] * alpha
            channels["facs_au"] = base
        channels["text_label"] = text_label(valence, arousal)
        return channels
