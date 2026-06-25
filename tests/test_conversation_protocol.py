"""ConversationModel 协议测试：协议契约、结构满足性、SteeringLanguageModel 新方法、零回归。

ConversationModel 未标 @runtime_checkable，用鸭子类型验证：
  - 构造满足协议的 fake 实例，断言方法可 await、返回类型正确。
  - 用 hasattr 检查现有实现是否有 converse/appraise_text。
torch/openai 依赖用 importorskip 优雅跳过。
"""

from __future__ import annotations

import inspect

import pytest

from src.agents.language import ConversationModel, LanguageModel
from src.agents.language_steering import SteeringLanguageModel

# ---------------------------------------------------------------------------
# 1. FakeConversationModel：鸭子验证协议契约
# ---------------------------------------------------------------------------


class FakeConversationModel:
    """满足 ConversationModel 协议的轻量 fake，零外部依赖。"""

    async def converse(
        self,
        history: list[dict[str, str]],
        affect: tuple[float, float],
        *,
        push: bool = False,
    ) -> str:
        return "fake-reply"

    async def appraise_text(self, text: str) -> tuple[float, float]:
        return (0.5, 0.3)


async def test_fake_converse_returns_str() -> None:
    """converse 可 await，返回 str。"""
    lm = FakeConversationModel()
    history = [{"role": "user", "content": "你好"}]
    result = await lm.converse(history, (0.1, 0.2))
    assert isinstance(result, str)
    assert result == "fake-reply"


async def test_fake_converse_push_kwarg_accepted() -> None:
    """push 关键字可选传入（协议签名兼容性）。"""
    lm = FakeConversationModel()
    result = await lm.converse([{"role": "user", "content": "hi"}], (0.0, 0.0), push=True)
    assert isinstance(result, str)


async def test_fake_appraise_text_returns_float_tuple() -> None:
    """appraise_text 可 await，返回 (float, float) 元组。"""
    lm = FakeConversationModel()
    result = await lm.appraise_text("我感到高兴")
    assert isinstance(result, tuple)
    assert len(result) == 2
    v, a = result
    assert isinstance(v, float)
    assert isinstance(a, float)
    assert result == (0.5, 0.3)


def test_fake_converse_is_coroutine_function() -> None:
    """converse 必须是协程函数（async def）。"""
    assert inspect.iscoroutinefunction(FakeConversationModel.converse)


def test_fake_appraise_text_is_coroutine_function() -> None:
    """appraise_text 必须是协程函数（async def）。"""
    assert inspect.iscoroutinefunction(FakeConversationModel.appraise_text)


def test_fake_duck_satisfies_conversation_model_structurally() -> None:
    """结构匹配验证：有两个必要方法且均为协程函数——鸭子 == 协议满足。"""
    lm = FakeConversationModel()
    assert hasattr(lm, "converse")
    assert hasattr(lm, "appraise_text")
    assert inspect.iscoroutinefunction(lm.converse)
    assert inspect.iscoroutinefunction(lm.appraise_text)


# ---------------------------------------------------------------------------
# 2. OpenAILanguageModel 结构满足 ConversationModel（不实例化真 client）
# ---------------------------------------------------------------------------


def test_openai_language_model_has_conversation_model_methods() -> None:
    """OpenAILanguageModel 有 converse + appraise_text 且均为协程函数。

    只做静态/类级属性检查，不实例化真 openai client（torch/网络-free）。
    """
    pytest.importorskip("src.agents.language_openai")
    from src.agents.language_openai import OpenAILanguageModel

    assert hasattr(OpenAILanguageModel, "converse")
    assert hasattr(OpenAILanguageModel, "appraise_text")
    assert hasattr(OpenAILanguageModel, "generate")
    assert inspect.iscoroutinefunction(OpenAILanguageModel.converse)
    assert inspect.iscoroutinefunction(OpenAILanguageModel.appraise_text)


def test_openai_language_model_satisfies_language_model_protocol_structurally() -> None:
    """OpenAILanguageModel 同时满足 LanguageModel（generate）协议——两协议并存。"""
    pytest.importorskip("src.agents.language_openai")
    from src.agents.language_openai import OpenAILanguageModel

    assert hasattr(OpenAILanguageModel, "generate")
    assert inspect.iscoroutinefunction(OpenAILanguageModel.generate)


# ---------------------------------------------------------------------------
# 3. SteeringLanguageModel.converse + appraise_text 单测（注入 fake backend）
# ---------------------------------------------------------------------------


class _FakeBackend:
    """捕获 delta、返回固定文本的 fake SteerBackend，torch-free。"""

    def __init__(self, text: str = "steered-text") -> None:
        self.text = text
        self.last_prompt: str | None = None
        self.last_delta: list[float] | None = None

    def generate_steered(self, prompt: str, delta: list[float]) -> str:
        self.last_prompt = prompt
        self.last_delta = delta
        return self.text


async def test_steering_converse_returns_nonempty_str() -> None:
    """SteeringLanguageModel.converse 返回非空 str，不调用真 torch/transformers。"""
    backend = _FakeBackend(text="steering-reply")
    model = SteeringLanguageModel(
        w_valence=[1.0, 0.0],
        w_arousal=[0.0, 1.0],
        alpha=0.5,
        backend=backend,
    )
    history = [{"role": "user", "content": "hi"}]
    result = await model.converse(history, (0.3, 0.2))
    assert isinstance(result, str)
    assert result == "steering-reply"


async def test_steering_converse_uses_last_user_message_as_context() -> None:
    """converse 取最后一条 user 消息作 context（best-effort）。"""
    backend = _FakeBackend(text="ok")
    model = SteeringLanguageModel(
        w_valence=[1.0, 0.0],
        w_arousal=[0.0, 1.0],
        alpha=0.5,
        backend=backend,
    )
    history = [
        {"role": "user", "content": "first message"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second message"},
    ]
    await model.converse(history, (0.1, 0.1))
    # prompt 包含最后一条 user 的内容
    assert backend.last_prompt is not None
    assert "second message" in backend.last_prompt


async def test_steering_converse_push_ignored_no_error() -> None:
    """push 参数被接受但无错误（steering 路径天然隐状态注情感，push 被忽略）。"""
    backend = _FakeBackend(text="ok")
    model = SteeringLanguageModel(
        w_valence=[1.0, 0.0],
        w_arousal=[0.0, 1.0],
        backend=backend,
    )
    result = await model.converse([], (0.0, 0.0), push=True)
    assert isinstance(result, str)


async def test_steering_converse_calls_backend_with_delta() -> None:
    """converse 把 affect 转为 steering delta 并传给 backend。"""
    from src.agents.language_steering import steering_delta

    backend = _FakeBackend()
    model = SteeringLanguageModel(
        w_valence=[1.0, 0.0],
        w_arousal=[0.0, 1.0],
        alpha=0.5,
        backend=backend,
    )
    affect = (0.3, 0.2)
    await model.converse([{"role": "user", "content": "test"}], affect)
    # delta 应由 steering_delta 计算（w_valence/w_arousal 经单位化，这里已是单位向量）
    expected = steering_delta(affect[0], affect[1], model.w_valence, model.w_arousal, alpha=0.5)
    assert backend.last_delta == expected


async def test_steering_appraise_text_returns_tuple() -> None:
    """SteeringLanguageModel.appraise_text 用词典法返回 (float, float)。"""
    backend = _FakeBackend()
    model = SteeringLanguageModel(
        w_valence=[1.0, 0.0],
        w_arousal=[0.0, 1.0],
        backend=backend,
    )
    result = await model.appraise_text("我感到愤怒")
    assert isinstance(result, tuple)
    assert len(result) == 2
    v, a = result
    assert isinstance(v, float)
    assert isinstance(a, float)
    assert -1.0 <= v <= 1.0
    assert -1.0 <= a <= 1.0


async def test_steering_appraise_text_uses_injected_appraiser() -> None:
    """appraise_text 使用注入的 appraiser callable（fake），可隔离词典依赖。"""
    backend = _FakeBackend()
    fixed_result = (0.7, -0.3)

    def fake_appraiser(text: str) -> tuple[float, float]:
        return fixed_result

    model = SteeringLanguageModel(
        w_valence=[1.0, 0.0],
        w_arousal=[0.0, 1.0],
        backend=backend,
        appraiser=fake_appraiser,
    )
    result = await model.appraise_text("任意文本")
    assert result == fixed_result


def test_steering_satisfies_conversation_model_structurally() -> None:
    """SteeringLanguageModel 同时满足 ConversationModel 协议（鸭子结构检查）。"""
    backend = _FakeBackend()
    model = SteeringLanguageModel(
        w_valence=[1.0, 0.0],
        w_arousal=[0.0, 1.0],
        backend=backend,
    )
    assert hasattr(model, "converse")
    assert hasattr(model, "appraise_text")
    assert inspect.iscoroutinefunction(model.converse)
    assert inspect.iscoroutinefunction(model.appraise_text)


def test_steering_still_satisfies_language_model_protocol() -> None:
    """SteeringLanguageModel 仍满足原 LanguageModel 协议（generate 不变——零回归）。"""
    backend = _FakeBackend()
    model: LanguageModel = SteeringLanguageModel(
        w_valence=[1.0, 0.0],
        w_arousal=[0.0, 1.0],
        backend=backend,
    )
    assert hasattr(model, "generate")
    assert inspect.iscoroutinefunction(model.generate)


# ---------------------------------------------------------------------------
# 4. 协议签名零回归：LanguageModel.generate 签名未变
# ---------------------------------------------------------------------------


def test_language_model_generate_signature_unchanged() -> None:
    """LanguageModel.generate 签名须与既有测试期望一致（纯增量，零回归）。"""
    sig = inspect.signature(LanguageModel.generate)
    params = list(sig.parameters.keys())
    # 必须包含原协议的所有参数（self + 关键字参数）
    assert "self" in params
    assert "affect" in params
    assert "context" in params
    assert "retrieved" in params
    assert "feedback" in params


def test_conversation_model_converse_signature() -> None:
    """ConversationModel.converse 签名：history, affect, *, push=False。"""
    sig = inspect.signature(ConversationModel.converse)
    params = sig.parameters
    assert "self" in params
    assert "history" in params
    assert "affect" in params
    assert "push" in params
    # push 有默认值 False
    assert params["push"].default is False


def test_conversation_model_appraise_text_signature() -> None:
    """ConversationModel.appraise_text 签名：text 参数。"""
    sig = inspect.signature(ConversationModel.appraise_text)
    params = sig.parameters
    assert "self" in params
    assert "text" in params
