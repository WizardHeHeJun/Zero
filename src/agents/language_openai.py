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
from typing import Any

from src.agents.affect_math import clamp
from src.agents.emotion_lexicon import suggest_affect_words
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
