"""PerceptionAgent：感知节点，由 stimulus 提取特征向量。

对应感觉皮层 + 杏仁核早期。节点契约：(state) -> dict，只返回增量。

文本路径（ZERO_TEXT_AFFECT_BACKEND=st）：用句向量回归器预测 (valence, arousal)；
OCC 占位路径（默认关）：直接取 OCC 评价维度；两路径输出接口完全兼容。
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from src.orchestration.state import AffectState

if TYPE_CHECKING:
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


class PerceptionAgent:
    """从 stimulus 抽取特征向量。

    默认（OCC 占位路径）：直接取 OCC 评价维度 [goal_congruence, standard_compliance,
    attitude_appeal, intensity]，不引入任何重依赖。

    文本路径（ZERO_TEXT_AFFECT_BACKEND=st）：用句向量回归器预测 (valence, arousal)，
    features 为 [valence, arousal, intensity, 0.0]。
    """

    def __init__(self) -> None:
        self.text_regressor: STTextAffectRegressor | None = _build_text_affect_regressor()

    def __call__(self, state: AffectState) -> dict:
        stim = state.stimulus
        if stim is None:
            return {}

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
            }
            return {"features": features, "text_affect": (valence, arousal), "trace": [entry]}
        else:
            # OCC 占位路径（默认）
            features = [
                stim.goal_congruence,
                stim.standard_compliance,
                stim.attitude_appeal,
                stim.intensity,
            ]
            backend_tag = "occ_placeholder"

        entry = {"node": "perception", "features": features, "backend": backend_tag}
        # 显式归零 text_affect：防止多轮同 thread 时上轮文本路径的值残留进 AffectCore
        return {"features": features, "text_affect": None, "trace": [entry]}
