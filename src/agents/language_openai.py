"""OpenAILanguageModel：满足 LanguageModel 协议的 OpenAI 兼容接口 adapter。

用通用 OpenAI Chat Completions 接口（可配 base_url 指向任意兼容服务：OpenAI、
本地 vLLM、第三方网关）实现真自然语言生成 + 独立情感反推：
  1) 按目标情感 e* + 上下文 + 检索 + 回路反馈，生成一句贴合该情绪的回应；
  2) 独立再调一次，让模型客观给这段话打 (valence, arousal)——独立反推使
     affect↔language 的「相互判断」真实有效（语言偏离目标即触发双向回路）。

异步：generate 为 async（网络 I/O，不阻塞事件循环）。client 可注入（便于测试 mock）；
未注入时延迟 import openai 构造 AsyncOpenAI——编排层与默认路径均不强依赖 openai。
真接入需 `pip install -e ".[llm]"`，并配置 base_url / api_key（构造参数或 env）。
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

from src.agents.affect_math import clamp
from src.agents.emotion_lexicon import affect_label, affect_logit_bias, suggest_affect_words
from src.agents.language import LanguageDraft

_COMPOSE_SYS = (
    "你是一个情感表达体的语言生成器。根据给定的情绪坐标"
    "（valence 效价∈[-1,1]，arousal 唤醒∈[-1,1]）、当前上下文与检索到的记忆，"
    "生成一句自然、贴合该情绪的中文回应。只输出这句话本身，不要解释、不要加引号。"
)
_APPRAISE_SYS = (
    "你是情感分析器。判断给定文本实际传达的情绪，只输出 JSON："
    '{"valence": <-1..1 浮点>, "arousal": <-1..1 浮点>}，不要任何其它内容。'
)
# 自然对话（区别于 _COMPOSE_SYS 的"按目标情绪生成"）：带真实情绪、循序渐进、不表演不扮演。
_CONVERSE_SYS = (
    "你在和用户自然地聊天，请结合**完整对话历史**连贯回应（记得前面聊过什么、能答「我刚才说了什么」）。"
    "你现在的真实心情是「{feeling}」——它**应该真实地**影响你的态度、语气和用词："
    "高兴就轻快，被冒犯/不被理解就流露不耐烦、委屈或火气，低落就提不起劲。"
    "**关键：心情负面时（恼火/愤怒/委屈/低落），绝不一味道歉、自责、讨好或无限包容**——那是讨好型客服，"
    "不是真人。该不耐烦就不耐烦，可以直接说「你这么说我不太舒服」、反问、冷淡、或适当回怼（但别脏话对脏话）；"
    "你有自尊，不是出气筒。你是一个**有脾气、有情绪起伏**的人，情绪慢慢累积、不因一句话突然大起大落。"
    "别夸张表演、别长篇大论、别程式化安慰。只输出你要说的话，简短、像真人。"
)
# push 通路（皮层下/不随意）：情绪经用词倾向自然漏进输出，而非"演情绪的指令"。
_PUSH_ADDENDUM = (
    "\n（此刻你不自觉地，用词会偏向这类词的色彩：{words}——是状态的自然流露，"
    "别刻意堆砌、别点破、别表演，只是语气与遣词自然带上而已。）"
)
PUSH_LOGIT_SCALE = 3.0  # affect-congruent 词 Δlogit → OpenAI logit_bias 缩放（温和，避免伤语法）


class OpenAILanguageModel:
    """OpenAI 兼容接口 adapter：两段式 generate（生成 + 独立 VAD 反推）。"""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        temperature: float = 0.8,
        use_lexicon: bool = False,
    ) -> None:
        self.model = model
        self.temperature = temperature
        # 词典桥（NRC-VAD 加权解码的 API 侧近似）：开启时把与 e* 最对齐的情绪词注入 compose
        # 提示，二段式 VAD 反推充当 reranker。默认关 → 对既有路径零回归。
        self.use_lexicon = use_lexicon
        if client is not None:
            self.client = client
        else:
            from openai import AsyncOpenAI  # 延迟 import：注入 client 时无需安装 openai

            self.client = AsyncOpenAI(
                base_url=base_url
                or os.getenv("ZERO_OPENAI_BASE_URL")
                or os.getenv("OPENAI_BASE_URL"),
                api_key=api_key or os.getenv("ZERO_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
            )

    async def generate(
        self,
        *,
        affect: tuple[float, float],
        context: str,
        retrieved: str,
        feedback: str | None,
        appraisal: str = "",
    ) -> LanguageDraft:
        text = await self._compose(affect, context, retrieved, feedback, appraisal)
        lang_affect = await self._appraise(text)
        return LanguageDraft(text=text, affect=lang_affect)

    async def appraise_text(self, text: str) -> tuple[float, float]:
        """把任意文本（如用户对话输入）客观评价成 (valence, arousal)。

        交互对话的「评价桥」：复用与生成解耦的独立 VAD 反推，读出用户这句话传达的情绪，
        供上层映射成 stimulus 喂进情感引擎。纯读取、不改状态。
        """
        return await self._appraise(text)

    async def converse(
        self,
        history: list[dict[str, str]],
        affect: tuple[float, float],
        retrieved: str = "",
        *,
        push: bool = False,
    ) -> str:
        """自然多轮对话：带完整历史。`generate`(强制 VAD+双向回路) 之外的连贯不戏剧化路径。

        `push`（皮层下/不随意通路）开启时：情绪经 **affect-congruent 用词倾向**（emotion_lexicon，
        "自然流露不表演"）漏进输出，而非"演情绪的指令"——对应神经科学 push 效应（见
        notes/2026-06-25-dual-route-language-push-pull.md）；可选叠加 OpenAI `logit_bias`
        （env `ZERO_PUSH_LOGIT_BIAS=1`，需兼容 tokenizer，graceful 回退）。关闭=纯 prompt(pull)。
        retrieved: 记忆召回的背景上下文（空串时不注入，prompt 与改前逐字一致 → 零回归）。
        history 末条应为用户最新发言。
        """
        sys = _CONVERSE_SYS.format(feeling=affect_label(*affect))
        if retrieved:
            sys += f"\n你还记得以下背景：{retrieved}"
        bias_kwargs: dict[str, Any] = {}
        if push:
            words = suggest_affect_words(affect[0], affect[1], k=6)
            if words:
                sys += _PUSH_ADDENDUM.format(words="、".join(words))
                logit_bias = self._build_logit_bias(affect, words)
                if logit_bias:
                    bias_kwargs["logit_bias"] = logit_bias
        messages: list[dict[str, str]] = [{"role": "system", "content": sys}]
        messages.extend(history)
        temperature = max(0.0, self.temperature + random.uniform(-0.1, 0.15))  # 措辞部分随机
        try:
            resp = await self.client.chat.completions.create(
                model=self.model, temperature=temperature, messages=messages, **bias_kwargs
            )
        except Exception:
            # 某些代理/模型不支持 logit_bias（或 token id 不匹配）→ 退回不带 bias；
            # push 仍经 prompt 用词倾向生效，不致命。
            resp = await self.client.chat.completions.create(
                model=self.model, temperature=temperature, messages=messages
            )
        return (resp.choices[0].message.content or "").strip()

    def _build_logit_bias(self, affect: tuple[float, float], words: list[str]) -> dict[int, float]:
        """解码期 push：affect-congruent 词 → OpenAI `logit_bias`（{token_id: 偏置}）。

        env `ZERO_PUSH_LOGIT_BIAS=1` 才启用（默认关——须与模型 tokenizer 匹配，否则偏到错 token）；
        缺 tiktoken / encode 失败 → 返回空（graceful，push 退到纯 prompt 用词倾向）。
        """
        if os.getenv("ZERO_PUSH_LOGIT_BIAS", "").lower() not in ("1", "true", "yes"):
            return {}
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return {}
        out: dict[int, float] = {}
        for word, delta in affect_logit_bias(words, affect).items():
            if abs(delta) < 1e-6:
                continue
            try:
                for tid in enc.encode(word):
                    out[tid] = clamp(PUSH_LOGIT_SCALE * delta, -6.0, 6.0)
            except Exception:
                continue
        return out

    async def _compose(
        self,
        affect: tuple[float, float],
        context: str,
        retrieved: str,
        feedback: str | None,
        appraisal: str = "",
    ) -> str:
        parts = [
            f"情绪坐标: valence={affect[0]:.2f}, arousal={affect[1]:.2f}",
            f"上下文: {context or '（无）'}",
        ]
        if appraisal:
            parts.append(f"认知评价: {appraisal}（据此把握情绪的来由与分寸）")
        if self.use_lexicon:
            cues = suggest_affect_words(affect[0], affect[1], k=5)
            if cues:
                parts.append(f"可参考的贴合情绪词（不必全用）: {'、'.join(cues)}")
        if retrieved:
            parts.append(f"检索记忆: {retrieved}")
        if feedback:
            parts.append(f"上一轮偏差反馈: {feedback}（请调整措辞使情绪更贴合坐标）")
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": _COMPOSE_SYS},
                {"role": "user", "content": "\n".join(parts)},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    async def _appraise(self, text: str) -> tuple[float, float]:
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _APPRAISE_SYS},
                {"role": "user", "content": text},
            ],
        )
        return self._parse_vad((resp.choices[0].message.content or "").strip())

    @staticmethod
    def _parse_vad(raw: str) -> tuple[float, float]:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return (0.0, 0.0)  # 无 JSON → 中性回退（保守，不崩管线）
        try:
            data = json.loads(raw[start : end + 1])
            v = clamp(float(data.get("valence", 0.0)), -1.0, 1.0)
            a = clamp(float(data.get("arousal", 0.0)), -1.0, 1.0)
        except (ValueError, TypeError, json.JSONDecodeError):
            return (0.0, 0.0)
        return (v, a)
