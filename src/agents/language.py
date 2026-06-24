"""LanguageAgent：语言生成 + affect↔language 双向收敛回路的语言侧。

由「情感 e* + 当前上下文 + 检索信息」生成语言内容，并反推该语言表达出的情感，
供条件边判断与内核情感是否一致。回路重写时先按双向互调把 e* 向上一轮语言情感
拉拢（reconcile_affect），使情感与语言相互判断、向中间收敛（情感不再是纯固定内核）。

语言模型可注入（鸭子类型 `LanguageModel` 协议，结构同 expression.ChannelDecoder）：
未注入则用占位 `_TemplateLanguageModel`（复用 text_label，torch-free / API-free）。
真模型（如 OpenAI 兼容 adapter）走网络 I/O，故 generate 为 async；节点 __call__ 亦 async。
节点契约：(state) -> dict，只返回增量。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.agents.affect_math import affect_distance, reconcile_affect, text_label
from src.orchestration.state import AffectState


@dataclass
class LanguageDraft:
    """一次语言生成的产物：文本 + 该文本反推出的情感。"""

    text: str
    affect: tuple[float, float]


class LanguageModel(Protocol):
    """语言模型协议（鸭子类型）；真 OpenAI 兼容 adapter 结构上满足即可注入。"""

    async def generate(
        self,
        *,
        affect: tuple[float, float],
        context: str,
        retrieved: str,
        feedback: str | None,
    ) -> LanguageDraft: ...


class _TemplateLanguageModel:
    """占位语言模型：按情感象限套模板生成文本，affect 回传目标情感（默认一致）。

    复用 `text_label` 的离散情绪词；不依赖 torch / 外部 API，保证默认零依赖、可单测。
    真模型（如 OpenAI 兼容 adapter）注入后替换本占位，LanguageAgent 契约不变。
    """

    async def generate(
        self,
        *,
        affect: tuple[float, float],
        context: str,
        retrieved: str,
        feedback: str | None,
    ) -> LanguageDraft:
        label = text_label(affect[0], affect[1])
        topic = context or "this"
        text = f"[{label}] 关于 {topic}"
        if retrieved:
            text += f"（recall: {retrieved}）"
        return LanguageDraft(text=text, affect=affect)


class LanguageAgent:
    """语言节点：生成语言并反推情感；回路重写前按双向互调微调内核 e*。"""

    def __init__(self, model: LanguageModel | None = None) -> None:
        self.model: LanguageModel = model if model is not None else _TemplateLanguageModel()

    async def __call__(self, state: AffectState) -> dict:
        if not state.language_enabled or state.affect_sample is None:
            return {}
        affect = state.affect_sample
        out: dict = {}
        # 回路重写：先把内核 e* 向上一轮语言情感拉拢（双向互调落点，写回增量）
        if state.language_iter > 0 and state.language_affect is not None:
            affect = reconcile_affect(affect, state.language_affect)
            out["affect_sample"] = affect

        context = state.stimulus.name if state.stimulus is not None else ""
        retrieved = (
            f"disposition={state.recalled_disposition:.2f}"
            if state.recalled_disposition is not None
            else ""
        )
        feedback = (
            f"prev_dist={state.language_consistency:.3f}"
            if state.language_iter > 0 and state.language_consistency is not None
            else None
        )
        draft = await self.model.generate(
            affect=affect, context=context, retrieved=retrieved, feedback=feedback
        )
        consistency = affect_distance(draft.affect, affect)
        entry = {
            "node": "language",
            "language_affect": draft.affect,
            "language_consistency": consistency,
            "iter": state.language_iter + 1,
        }
        out.update(
            {
                "language_text": draft.text,
                "language_affect": draft.affect,
                "language_consistency": consistency,
                "language_iter": state.language_iter + 1,
                "trace": [entry],
            }
        )
        return out
