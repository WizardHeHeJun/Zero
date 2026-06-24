"""OpenAILanguageModel adapter 单测：用 fake async client，不依赖 openai 包。

验证两段式：compose 文本 + 独立 appraise VAD；以及 VAD 解析的容错、回退与钳制。
"""

from __future__ import annotations

import pytest

from src.agents.language_openai import OpenAILanguageModel


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, queue: list[str]) -> None:
        self.queue = queue

    async def create(self, **kwargs: object) -> _Resp:
        return _Resp(self.queue.pop(0))


class _Chat:
    def __init__(self, queue: list[str]) -> None:
        self.completions = _Completions(queue)


class _FakeClient:
    """按调用顺序返回：第 1 次 compose 文本，第 2 次 appraise 的 VAD JSON。"""

    def __init__(self, *, text: str, vad: str) -> None:
        self.chat = _Chat([text, vad])


async def test_generates_text_and_independent_vad() -> None:
    client = _FakeClient(text="我好开心！", vad='{"valence": 0.8, "arousal": 0.6}')
    model = OpenAILanguageModel(client=client, model="x")
    draft = await model.generate(affect=(0.5, 0.4), context="收到礼物", retrieved="", feedback=None)
    assert draft.text == "我好开心！"
    assert draft.affect == (0.8, 0.6)


async def test_parses_vad_embedded_in_noise() -> None:
    client = _FakeClient(text="还行吧", vad='结果是 {"valence": -0.2, "arousal": 0.1} 。')
    model = OpenAILanguageModel(client=client, model="x")
    draft = await model.generate(affect=(0.0, 0.0), context="", retrieved="", feedback=None)
    assert draft.affect[0] == pytest.approx(-0.2)
    assert draft.affect[1] == pytest.approx(0.1)


async def test_falls_back_to_neutral_on_bad_vad() -> None:
    client = _FakeClient(text="嗯", vad="not json at all")
    model = OpenAILanguageModel(client=client, model="x")
    draft = await model.generate(affect=(0.0, 0.0), context="", retrieved="", feedback=None)
    assert draft.affect == (0.0, 0.0)


async def test_clamps_out_of_range_vad() -> None:
    client = _FakeClient(text="爆炸开心", vad='{"valence": 5, "arousal": -9}')
    model = OpenAILanguageModel(client=client, model="x")
    draft = await model.generate(affect=(0.0, 0.0), context="", retrieved="", feedback=None)
    assert draft.affect == (1.0, -1.0)
