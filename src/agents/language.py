"""LanguageAgent：语言生成 + affect↔language 双向收敛回路的语言侧。

由「情感 e* + 当前上下文 + 检索信息 + （可选）OCC 评价结构」生成语言内容，并反推该语言
表达出的情感，供条件边判断与内核情感是否一致。回路重写时先按双向互调把 e* 向上一轮语言
情感拉拢（reconcile_affect），使情感与语言相互判断、向中间收敛（情感不再是纯固定内核）。

评价条件化（`appraisal_conditioning_enabled`，默认关）：把 OCC 评价维度而非仅最终 (v,a)
并入生成（CPM/EMA/APTNESS——评价驱动 NLG），给更高粒度、更得体的情绪表达。

语言模型可注入（鸭子类型 `LanguageModel` 协议，结构同 expression.ChannelDecoder）：
未注入则用占位 `_TemplateLanguageModel`（torch-free / API-free）。
真模型（如 OpenAI 兼容 adapter）走网络 I/O，故 generate 为 async；节点 __call__ 亦 async。
节点契约：(state) -> dict，只返回增量。

职责分离：`LanguageModel` 是图内节点（StateGraph 编排路径）的生成协议；`ConversationModel`
是图外 REPL（--chat 交互对话）的对话协议——两者均可由同一实现类（如 OpenAILanguageModel）
结构上同时满足，无需显式声明继承。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.agents.affect_math import (
    LANG_BASE_PRECISION,
    affect_distance,
    precision_reconcile,
    reconcile_affect,
)
from src.agents.emotion_lexicon import affect_descriptor, motivational_system, suggest_affect_words
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
        appraisal: str = "",
    ) -> LanguageDraft: ...


@runtime_checkable
class ConversationModel(Protocol):
    """对话路径协议（图外 REPL）：带历史的自然对话 + 用户输入情感评价。

    与 LanguageModel（图内节点 generate）职责分离——converse 接对话历史(OpenAI messages 格式)
    + 当前情绪坐标返回回应文本；appraise_text 把文本评价成 (valence, arousal) 作评价桥。
    实现者 OpenAILanguageModel 同时满足两协议（结构匹配即可，无需显式声明）。
    """

    async def converse(
        self,
        history: list[dict[str, str]],
        affect: tuple[float, float],
        retrieved: str = "",
        *,
        push: bool = False,
    ) -> str:
        """生成一轮对话回应。

        history: OpenAI messages 格式的对话历史（末条应为用户最新发言）。
        affect: 当前情绪坐标 (valence, arousal)。
        retrieved: 召回的背景上下文（空串时不注入，行为与改前一致）。
        push: 皮层下不随意通路——情绪经用词倾向漏进输出而非显式指令。
        """
        ...

    async def appraise_text(self, text: str) -> tuple[float, float]: ...


class _TemplateLanguageModel:
    """占位语言模型：按情感套模板生成文本，affect 回传目标情感（默认一致）。

    用 `emotion_lexicon` 的细粒度情绪词（circumplex 8 扇区 × 强度）+ Panksepp 动机色彩，
    比旧 4 档 `text_label` 更有粒度；并附 1 个与 e* 最对齐的情绪词样例（词典桥）。
    不依赖 torch / 外部 API，保证默认零依赖、可单测。真模型注入后替换本占位，契约不变。
    """

    async def generate(
        self,
        *,
        affect: tuple[float, float],
        context: str,
        retrieved: str,
        feedback: str | None,
        appraisal: str = "",
    ) -> LanguageDraft:
        descriptor = affect_descriptor(affect[0], affect[1])
        topic = context or "this"
        cue = suggest_affect_words(affect[0], affect[1], k=1)
        text = f"[{descriptor}] 关于 {topic}"
        if cue:
            text += f"，{cue[0]}"
        if appraisal:
            text += f"（appraisal: {appraisal}）"
        if retrieved:
            text += f"（recall: {retrieved}）"
        return LanguageDraft(text=text, affect=affect)


def _appraisal_summary(state: AffectState) -> str:
    """从 stimulus 的 OCC 评价维度 + 动机系统组装一行评价摘要（评价驱动 NLG 的条件）。

    无 stimulus 时返回空串（退化为不条件化）。仅取标量，不放大对象。
    """
    stim = state.stimulus
    if stim is None:
        return ""
    v = state.appraisal.get("valence")
    a = state.appraisal.get("arousal")
    system = motivational_system(v, a) if v is not None and a is not None else "?"
    return (
        f"目标一致性={stim.goal_congruence:+.2f}, 标准契合={stim.standard_compliance:+.2f}, "
        f"对象喜好={stim.attitude_appeal:+.2f}, 强度={stim.intensity:.2f}; 动机系统={system}"
    )


class LanguageAgent:
    """语言节点：生成语言并反推情感；回路重写前按双向互调微调内核 e*。"""

    def __init__(self, model: LanguageModel | None = None) -> None:
        self.model: LanguageModel = model if model is not None else _TemplateLanguageModel()

    async def __call__(self, state: AffectState) -> dict:
        if not state.language_enabled or state.affect_sample is None:
            return {}
        affect = state.affect_sample
        out: dict = {}
        # 回路重写：先把内核 e* 向上一轮语言情感拉拢（双向互调落点，写回增量）。
        # workspace 开启时走精度加权再入（高精度内核抗拉拢）；否则固定中点拉拢（v1）。
        if state.language_iter > 0 and state.language_affect is not None:
            if state.workspace_enabled:
                e_prec = (
                    state.affect_precision
                    if state.affect_precision is not None
                    else LANG_BASE_PRECISION
                )
                affect = precision_reconcile(affect, e_prec, state.language_affect)
            else:
                affect = reconcile_affect(affect, state.language_affect)
            out["affect_sample"] = affect

        context = state.stimulus.name if state.stimulus is not None else ""
        # 检索信息 = 确定性 disposition + 语义召回事实（Graphiti 等）；后者为空时退化为现状
        retrieved_parts: list[str] = []
        if state.recalled_disposition is not None:
            retrieved_parts.append(f"disposition={state.recalled_disposition:.2f}")
        if state.recalled_context:
            retrieved_parts.append("; ".join(state.recalled_context))
        retrieved = " | ".join(retrieved_parts)
        feedback = (
            f"prev_dist={state.language_consistency:.3f}"
            if state.language_iter > 0 and state.language_consistency is not None
            else None
        )
        # 评价条件化（默认关）：仅在开启且有评价摘要时才传 appraisal，
        # 使默认路径调用签名与改前一致——对未感知 appraisal 的注入模型零回归。
        appraisal = _appraisal_summary(state) if state.appraisal_conditioning_enabled else ""
        if appraisal:
            draft = await self.model.generate(
                affect=affect,
                context=context,
                retrieved=retrieved,
                feedback=feedback,
                appraisal=appraisal,
            )
        else:
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
