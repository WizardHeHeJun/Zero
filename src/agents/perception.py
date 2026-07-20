"""PerceptionAgent：感知节点，由 stimulus 提取特征向量。

对应感觉皮层 + 杏仁核早期。节点契约：(state) -> dict，只返回增量。

文本路径（ZERO_TEXT_AFFECT_BACKEND=st）：用句向量回归器预测 (valence, arousal)；
OCC 占位路径（默认关）：直接取 OCC 评价维度；两路径输出接口完全兼容。

text_coping 独立标量流（W1·议会 2026-07-20·默认关=零回归）：
  ZERO_DIRECTION_HEAD_MODEL_PATH 未设 → direction_head=None → text_coping_prior 恒 None。
  state.text_coping_enabled=False（默认）→ 归零（不读 env·节点看 state）。
  两条件均满足时产出 tanh(logit) ∈ [-1, 1]，写入 text_coping_prior。
  弃权门 τ 读 ZERO_ANGER_ABSTAIN_LOGIT_THRESHOLD（默认 0.0=不弃权）。
"""

from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING

from src.orchestration.state import AffectState

if TYPE_CHECKING:
    from src.agents.models.direction_head import DirectionHead
    from src.agents.models.text_affect_regressor_st import STTextAffectRegressor

logger = logging.getLogger(__name__)


def _build_text_affect_regressor() -> STTextAffectRegressor | None:
    """工厂函数：按环境变量决定是否加载句向量文本情感回归器。

    - ZERO_TEXT_AFFECT_BACKEND 未设置或为空 → 返回 None（默认关，不 import 重依赖）。
    - 设为 "st" 但缺 ZERO_TEXT_AFFECT_MODEL_PATH → warning + 返回 None（回退 OCC）。
    - 设为 "st" 且有路径 → 延迟 import 加载；任何异常 → warning + 返回 None（fail-soft）。
    """
    backend = os.getenv("ZERO_TEXT_AFFECT_BACKEND", "")
    if not backend:
        return None

    if backend != "st":
        logger.warning("未知 ZERO_TEXT_AFFECT_BACKEND=%r，回退 OCC 路径", backend)
        return None

    model_path = os.getenv("ZERO_TEXT_AFFECT_MODEL_PATH", "")
    if not model_path:
        logger.warning(
            "ZERO_TEXT_AFFECT_BACKEND=st 但未设置 ZERO_TEXT_AFFECT_MODEL_PATH，回退 OCC 路径"
        )
        return None

    try:
        from src.agents.models.text_affect_regressor_st import load_st_text_affect_regressor

        regressor = load_st_text_affect_regressor(model_path)
        logger.info("已加载 STTextAffectRegressor，权重路径=%s", model_path)
        return regressor
    except Exception as exc:  # noqa: BLE001
        logger.warning("加载 STTextAffectRegressor 失败（%s），回退 OCC 路径", exc, exc_info=True)
        return None


def _build_direction_head() -> DirectionHead | None:
    """工厂函数：按环境变量决定是否加载 DirectionHead（motivational_direction_prior）。

    - ZERO_DIRECTION_HEAD_MODEL_PATH 未设置或为空 → 返回 None（默认关·不引 torch）。
    - 设了路径 → 延迟 import src.agents.models.direction_head 加载；
      任何异常 → warning + 返回 None（fail-soft）。
    """
    model_path = os.getenv("ZERO_DIRECTION_HEAD_MODEL_PATH", "")
    if not model_path:
        return None

    try:
        from src.agents.models.direction_head import load_direction_head

        head = load_direction_head(model_path)
        logger.info("已加载 DirectionHead，权重路径=%s", model_path)
        return head
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "加载 DirectionHead 失败（%s），text_coping_prior 将恒 None", exc, exc_info=True
        )
        return None


def _domain_direction_accepts(domain: str, direction_sign: float) -> bool:
    """域×方向匹配谓词：仅在信号的 home 域接受该方向。

    confrontational → direction_sign > 0（anger home 域）。
    survival_narrative → direction_sign < 0（fear home 域）。
    neutral（或其余未知值）→ False（显式两不属·弃权）。

    重要（A-W1·议会 2026-07-20 WARN-1）：
      domain 参数必须非 None——None 旁路由调用点 `if domain is not None` 处理。
      本谓词的 `return False`（neutral 分支）是显式弃权，语义上≠ None 旁路；
      若将 None 传入此谓词会产生 False，导致 text_coping_prior 被置 None，
      与「None=整门旁路、prior 逐字不变」的零回归语义分叉——严禁混用。

    热路径：纯枚举+符号比较，LLM-free，无判别器（域由调用方注入）。
    """
    if domain == "confrontational":
        return direction_sign > 0.0  # anger home
    if domain == "survival_narrative":
        return direction_sign < 0.0  # fear home
    return False  # neutral → 显式两不属 → 弃权（≠ None 旁路）


class PerceptionAgent:
    """从 stimulus 抽取特征向量。

    默认（OCC 占位路径）：直接取 OCC 评价维度 [goal_congruence, standard_compliance,
    attitude_appeal, intensity]，不引入任何重依赖。

    文本路径（ZERO_TEXT_AFFECT_BACKEND=st）：用句向量回归器预测 (valence, arousal)，
    features 为 [valence, arousal, intensity, 0.0]。

    text_coping 独立标量流（W1·默认关=零回归）：
      ZERO_DIRECTION_HEAD_MODEL_PATH 设了 + state.text_coping_enabled=True 时，
      产出 text_coping_prior = tanh(logit) ∈ [-1, 1]（+1≈anger趋近·-1≈fear回避）。
      弃权门 τ（ZERO_ANGER_ABSTAIN_LOGIT_THRESHOLD，默认 0.0=不弃权）：
        abs(logit) < τ → text_coping_prior=None（弃权·B1 科学决策·当前 τ=0 inert）。

    B2 域门（A1 热路径·议会 2026-07-20）：
      _compute_text_coping 在 τ 弃权门之后施 A1 域×方向匹配门：
        domain=None（默认）→ 整门旁路，prior 逐字不变（零回归，保 1061 passed）。
        domain 非 None → 调用 _domain_direction_accepts(domain, prior) 检查方向：
          失配（off-domain）→ prior=None（硬弃·off-domain π_t 近似为 0）。
          匹配 → prior 逐字传出。
      域门只置 text_coping_prior=None，绝不进 fuse_terms/occ_prior/其余流（来源正交）。
    """

    def __init__(self) -> None:
        self.text_regressor: STTextAffectRegressor | None = _build_text_affect_regressor()
        self.direction_head: DirectionHead | None = _build_direction_head()
        # 弃权门阈值：工厂/init 期读，默认 0.0（inert·旋钮 inert·B1 科学决策后再调）
        self.abstain_threshold: float = float(
            os.getenv("ZERO_ANGER_ABSTAIN_LOGIT_THRESHOLD", "0.0")
        )

    def __call__(self, state: AffectState) -> dict:
        stim = state.stimulus
        if stim is None:
            # 无 stimulus 也显式归零 text_coping_prior，防 LastValue 残留（节点契约·WARN-1）
            return {"text_coping_prior": None}

        # ── text_coping 独立标量流（W1·节点契约：每条路径都显式写键，防 LastValue 残留）──
        text_coping_delta = self._compute_text_coping(state)

        if self.text_regressor is not None and stim.text is not None:
            # 文本路径：句向量回归器预测 (valence, arousal)
            valence, arousal = self.text_regressor.predict_affect(stim.text)
            # features 恢复 OCC 布局（与 OCC 路径同下标），让 fast_survival_prior 的下标假设安全
            features = [
                stim.goal_congruence,
                stim.standard_compliance,
                stim.attitude_appeal,
                stim.intensity,
            ]
            backend_tag = "st_text"
            entry = {
                "node": "perception",
                "features": features,
                "backend": backend_tag,
                "text_affect": (valence, arousal),
                **text_coping_delta,
            }
            return {
                "features": features,
                "text_affect": (valence, arousal),
                "trace": [entry],
                **text_coping_delta,
            }
        else:
            # OCC 占位路径（默认）
            features = [
                stim.goal_congruence,
                stim.standard_compliance,
                stim.attitude_appeal,
                stim.intensity,
            ]
            backend_tag = "occ_placeholder"

        entry = {
            "node": "perception",
            "features": features,
            "backend": backend_tag,
            **text_coping_delta,
        }
        # 显式归零 text_affect：防止多轮同 thread 时上轮文本路径的值残留进 AffectCore
        return {
            "features": features,
            "text_affect": None,
            "trace": [entry],
            **text_coping_delta,
        }

    def _compute_text_coping(self, state: AffectState) -> dict:
        """产出 text_coping_prior 增量（节点契约：每条路径都返回该键，防 LastValue 残留）。

        text_coping_source（bool）是 AppraisalAgent 的输出 flag，PerceptionAgent 不写。
        PerceptionAgent 只负责产出 text_coping_prior（float | None）。

        门控逻辑（短路，按优先级）：
          1. state.text_coping_enabled is False（默认）→ 归零（零回归·分支 1/3）。
          2. self.direction_head is None → 归零（未加载权重）。
          3. stim 或 stim.text 缺失 → 归零。
          4. 产出路径：encode_texts → logit → 弃权门 τ → A1 域门 → tanh(logit) 或 None。

        A1 域门（B2·议会 2026-07-20·热路径单门）：
          τ 弃权门之后，在 return 之前施 A1 域×方向匹配门（_domain_direction_accepts）：
            stim.domain=None（默认）→ 整门旁路，prior 逐字不变（零回归，保 1061 passed）。
            stim.domain 非 None 且 prior 非 None → 调用谓词：
              匹配 → prior 不变；失配 → prior=None（硬弃·off-domain π_t 近似为 0）。
          域门只置 text_coping_prior=None，绝不触及 fuse_terms/occ_prior 等流。

        节点读 state.text_coping_enabled，不读 ZERO_TEXT_COPING_ENABLED env
        （env 在驱动层注入 state·守节点契约）。
        """
        _null: dict = {"text_coping_prior": None}

        # 门控 1：总开关（默认 False=零回归）
        if not state.text_coping_enabled:
            return _null

        # 门控 2：权重未加载
        if self.direction_head is None:
            return _null

        # 门控 3：stim/text 缺失
        stim = state.stimulus
        if stim is None or stim.text is None:
            return _null

        # 产出路径：延迟 import encode_texts（torch 重依赖，与 direction_head 同批）
        import torch

        from src.agents.models.text_affect_regressor_st import encode_texts

        with torch.no_grad():
            emb = encode_texts([stim.text])  # (1, dim)
            logit_t = self.direction_head(emb)  # (1,) or scalar
            logit: float = float(logit_t[0] if logit_t.dim() >= 1 else logit_t)

        # 弃权门（τ 默认 0.0=不弃权·B1 科学决策后再定激活值）
        if abs(logit) < self.abstain_threshold:
            prior: float | None = None
        else:
            prior = math.tanh(logit)

        logger.debug(
            "text_coping logit=%.4f abstain_τ=%.4f prior=%s",
            logit,
            self.abstain_threshold,
            prior,
        )

        # A1 域门（B2·议会 2026-07-20·热路径·域×方向匹配）
        # A-W1：domain=None → 整门旁路，prior 逐字不变（零回归）；
        #        domain 非 None → 谓词只处理非 None domain（None 旁路由此 if 包裹）。
        # prior is not None 防止 None 入谓词（符号比较对 None 无意义）。
        domain = stim.domain
        # 注意：prior is None（τ 弃权门已置 None）时整个域门条件短路为 False → 整门跳过，
        # prior 不再被置 None，直接透传（与下方 domain is None 旁路分支外观对称，
        # 但语义不同——防后续维护者误改条件顺序将「弃权」混同「域失配置 None」）。
        if (
            domain is not None
            and prior is not None
            and not _domain_direction_accepts(domain, prior)
        ):
            prior = None  # 域失配 → 硬弃（off-domain π_t 近似为 0）
            logger.debug("text_coping 域失配·硬弃 domain=%r prior→None", domain)
        # domain is None → 整门旁路·prior 逐字不变（零回归）

        return {"text_coping_prior": prior}
