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


# ---------- P3：appraise 分级标定校准（ZERO_APPRAISE_CALIBRATE 门控，默认关零回归） ----------


class _CapturingClient:
    """记录每次 create 的 kwargs（看 appraise 的 system prompt），返回固定 VAD。"""

    def __init__(self, vad: str = '{"valence": -0.3, "arousal": 0.2}') -> None:
        self.calls: list[dict] = []

        class _C:
            async def create(_self, **kwargs: object) -> _Resp:  # noqa: N805
                self.calls.append(dict(kwargs))
                return _Resp(vad)

        self.chat = type("_Chat", (), {"completions": _C()})()


async def test_appraise_calibration_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZERO_APPRAISE_CALIBRATE 未设 → appraise prompt 与 _APPRAISE_SYS 逐字相等（零回归）。"""
    from src.agents.language_openai import _APPRAISE_SYS

    monkeypatch.delenv("ZERO_APPRAISE_CALIBRATE", raising=False)
    client = _CapturingClient()
    await OpenAILanguageModel(client=client, model="x").appraise_text("你真没用")
    assert client.calls[0]["messages"][0]["content"] == _APPRAISE_SYS


async def test_appraise_calibration_on_injects_graded_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ZERO_APPRAISE_CALIBRATE=1 → system prompt 附分级校准锚（含极端攻击 -0.95 锚）。"""
    monkeypatch.setenv("ZERO_APPRAISE_CALIBRATE", "1")
    client = _CapturingClient()
    await OpenAILanguageModel(client=client, model="x").appraise_text("你真没用")
    sys = client.calls[0]["messages"][0]["content"]
    assert "参照标定示例" in sys and "valence≈-0.95" in sys
