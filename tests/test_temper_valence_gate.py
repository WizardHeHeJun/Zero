"""脾气段的 valence 门控（ZERO_TEMPER_VALENCE_GATE）。

背景：一次 100 轮实跑实测语气强度与引擎情绪**反相关**——情绪「平静」的 49 轮里 49%
的回复带命令/反问/贬抑语气，而情绪明确为负的 30 轮里只有 17%。根因是
`_TEMPER_ADDENDUM`（「负面时别讨好、该不耐烦就不耐烦」）被**无条件**注入，
中性话题没有情绪素材时 LLM 便用「性格」填补空白。

本门控让脾气段只在 e* 的 valence 确实为负且够强时注入，使语气强度真正由引擎驱动。
⚠ 默认未设 = 无条件注入 = 改前逐字行为（零回归），便于跑 A/B 对照。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("openai")

from src.agents.language_openai import (  # noqa: E402
    _TEMPER_ADDENDUM,
    OpenAILanguageModel,
)

HISTORY = [{"role": "user", "content": "外面还在下雨吗"}]


class _CapturingClient:
    """捕获 chat.completions.create 收到的 messages，不发网络请求。"""

    def __init__(self) -> None:
        self.captured: dict[str, Any] = {}
        outer = self

        class _Completions:
            async def create(self, **kwargs: Any) -> Any:
                outer.captured = kwargs
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
                )

        self.chat = SimpleNamespace(completions=_Completions())


async def _system_prompt_for(
    affect: tuple[float, float], monkeypatch: pytest.MonkeyPatch, gate: str | None
) -> str:
    if gate is None:
        monkeypatch.delenv("ZERO_TEMPER_VALENCE_GATE", raising=False)
    else:
        monkeypatch.setenv("ZERO_TEMPER_VALENCE_GATE", gate)
    client = _CapturingClient()
    lm = OpenAILanguageModel(model="test-model", client=client)
    await lm.converse(HISTORY, affect)
    return client.captured["messages"][0]["content"]


async def test_unset_gate_always_injects_temper(monkeypatch: pytest.MonkeyPatch) -> None:
    """未设阈值 → 无条件注入脾气段（改前行为，零回归）。"""
    for affect in [(0.09, -0.03), (-0.30, 0.10), (0.0, 0.0)]:
        sys_prompt = await _system_prompt_for(affect, monkeypatch, None)
        assert _TEMPER_ADDENDUM in sys_prompt, f"未设阈值时 affect={affect} 也应注入"


async def test_gate_suppresses_temper_on_calm_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """设了阈值 → 中性/正面情绪的轮次**不再**注入脾气段。

    取的是实跑里真实出现过的「平静」轮 e* 值。
    """
    for affect in [(0.09, -0.03), (-0.03, -0.02), (0.10, -0.03), (0.0, 0.0)]:
        sys_prompt = await _system_prompt_for(affect, monkeypatch, "-0.15")
        assert _TEMPER_ADDENDUM not in sys_prompt, f"affect={affect} 不该注入脾气段"


async def test_gate_still_injects_when_genuinely_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """情绪确实为负且够强 → 仍注入，阶段 15–17「负面时别讨好」的裁定不被推翻。"""
    for affect in [(-0.30, 0.10), (-0.47, -0.05), (-0.15, 0.0)]:
        sys_prompt = await _system_prompt_for(affect, monkeypatch, "-0.15")
        assert _TEMPER_ADDENDUM in sys_prompt, f"affect={affect} 应注入脾气段"


async def test_base_prompt_always_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """基础段（诚实优先 / 情绪影响语气 / 简短像真人）无论门控如何都在。"""
    for gate in (None, "-0.15"):
        sys_prompt = await _system_prompt_for((0.09, -0.03), monkeypatch, gate)
        assert "诚实优先" in sys_prompt
        assert "你现在的真实心情是" in sys_prompt
        assert "简短、像真人" in sys_prompt
        assert "情绪慢慢累积、不因一句话突然大起大落" in sys_prompt


async def test_gate_boundary_is_inclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    """边界取 <=：valence 恰等于阈值时注入（避免边界处行为不确定）。"""
    assert _TEMPER_ADDENDUM in await _system_prompt_for((-0.15, 0.0), monkeypatch, "-0.15")
    assert _TEMPER_ADDENDUM not in await _system_prompt_for((-0.1499, 0.0), monkeypatch, "-0.15")


async def test_mutation_gate_must_actually_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """**证明这些断言能红**：若门控失效（阈值被当成恒真），中性轮的用例必须失败。

    这里直接对照两种配置下同一 affect 的产物——若两者相同，说明门根本没起作用。
    """
    calm = (0.09, -0.03)
    without_gate = await _system_prompt_for(calm, monkeypatch, None)
    with_gate = await _system_prompt_for(calm, monkeypatch, "-0.15")
    assert without_gate != with_gate, "两种配置产出同一 prompt，说明门控没有生效"
    assert len(with_gate) < len(without_gate), "门控生效时 prompt 应更短（少了脾气段）"
