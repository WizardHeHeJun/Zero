"""阶段15：recall 回灌对话 — converse(retrieved=...) 集成测试。

覆盖：
  1. retrieved 注入 OpenAI converse：mock 断言 retrieved 进 prompt；retrieved="" 不追加（零回归）。
  2. recall 回灌管道（集成）：fake SemanticStore 有命中 → recalled_context 非空。
  3. 无记忆 no-op / 零回归：空 MemoryClient → recalled_context=[]；recall_enabled=False 同。
  4. steering converse retrieved：注入 fake backend + fake appraiser，retrieved 进 _build_prompt。
  5. 协议结构：runtime_checkable ConversationModel 被满足（FakeConversationModel + retrieved）。

全部 mock/fake，不打真 LLM / torch / 网络。
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

# ---------------------------------------------------------------------------
# 公共 fake client helpers（同 test_language_openai.py 风格）
# ---------------------------------------------------------------------------


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


# ---------------------------------------------------------------------------
# Task 1：retrieved 注入 OpenAI converse
# ---------------------------------------------------------------------------


class _CapturingCompletions:
    """捕获每次 create 调用的 kwargs，返回固定回应。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs: object) -> _Resp:
        self.calls.append(dict(kwargs))
        return _Resp("回应文本")


class _CapturingChat:
    def __init__(self) -> None:
        self.completions = _CapturingCompletions()


class _CapturingClient:
    def __init__(self) -> None:
        self.chat = _CapturingChat()


async def test_openai_converse_injects_retrieved_into_system_prompt() -> None:
    """retrieved 非空时，system prompt 应追加「你还记得以下背景：…」。"""
    from src.agents.language_openai import OpenAILanguageModel

    client = _CapturingClient()
    lm = OpenAILanguageModel(client=client, model="x")
    history = [{"role": "user", "content": "你好"}]
    await lm.converse(history, (0.2, 0.1), retrieved="背景X")

    assert len(client.chat.completions.calls) == 1
    messages = client.chat.completions.calls[0]["messages"]
    system_content = messages[0]["content"]
    assert messages[0]["role"] == "system"
    assert "背景X" in system_content
    assert "你还记得以下背景" in system_content


async def test_openai_converse_no_retrieved_system_prompt_unchanged() -> None:
    """retrieved="" 时，system prompt 不追加记忆背景行（零回归）。"""
    from src.agents.language_openai import OpenAILanguageModel

    client = _CapturingClient()
    lm = OpenAILanguageModel(client=client, model="x")
    history = [{"role": "user", "content": "你好"}]
    await lm.converse(history, (0.2, 0.1))  # retrieved 默认空串

    messages = client.chat.completions.calls[0]["messages"]
    system_content = messages[0]["content"]
    assert "你还记得以下背景" not in system_content


async def test_openai_converse_retrieved_in_second_positional_position() -> None:
    """retrieved 以第三位置参数传入也能正确注入（调用方式与 main.py 一致）。"""
    from src.agents.language_openai import OpenAILanguageModel

    client = _CapturingClient()
    lm = OpenAILanguageModel(client=client, model="x")
    history = [{"role": "user", "content": "hi"}]
    # main.py 里调用方式：lm.converse(history[-20:], emotion, recalled_str, push=True)
    await lm.converse(history, (0.0, 0.0), "某段背景", push=False)

    messages = client.chat.completions.calls[0]["messages"]
    assert "某段背景" in messages[0]["content"]


async def test_openai_converse_history_fully_passed_with_retrieved() -> None:
    """retrieved 注入时，完整 history 仍被传入（messages[1:] == history）。"""
    from src.agents.language_openai import OpenAILanguageModel

    client = _CapturingClient()
    lm = OpenAILanguageModel(client=client, model="x")
    history = [
        {"role": "user", "content": "我叫小明"},
        {"role": "assistant", "content": "你好小明"},
        {"role": "user", "content": "我刚说我叫什么？"},
    ]
    await lm.converse(history, (-0.1, 0.2), retrieved="一段背景")

    messages = client.chat.completions.calls[0]["messages"]
    assert messages[1:] == history  # 完整历史原样带入


# ---------------------------------------------------------------------------
# Task 2：recall 回灌管道（集成）—— fake SemanticStore 有命中
# ---------------------------------------------------------------------------


class _FakeSemanticStore:
    """内存 SemanticStore：预填一条可被召回的 episode，search 返回它。"""

    def __init__(self, episodes: list[str] | None = None) -> None:
        self.episodes: list[str] = episodes or []

    async def add_episode(self, *, scope: str, key: str, content: str, valid_at: datetime) -> None:
        self.episodes.append(content)

    async def search(
        self,
        query: str,
        *,
        scope: str,
        key: str | None = None,
        at: datetime | None = None,
        limit: int = 5,
    ) -> list:
        from src.storage.backends.deterministic import StoredFact

        return [
            StoredFact(
                scope=scope,
                key=key or "u1",
                content=ep,
                valid_at=datetime.now(UTC),
            )
            for ep in self.episodes[:limit]
        ]


async def test_recall_pipeline_populates_recalled_context() -> None:
    """recall_enabled=True + fake SemanticStore 有 episode → recalled_context 非空。"""
    from src.memory.client import MemoryClient
    from src.orchestration.memory_recall import MemoryRecallAgent
    from src.orchestration.state import AffectState, Stimulus

    semantic = _FakeSemanticStore(episodes=["用户之前提到自己喜欢音乐"])
    mem = MemoryClient(semantic=semantic)
    state = AffectState(
        recall_enabled=True,
        user_id="u1",
        stimulus=Stimulus(name="query", goal_congruence=0.1, intensity=0.5),
    )
    out = await MemoryRecallAgent(mem)(state)
    assert "recalled_context" in out
    assert len(out["recalled_context"]) > 0
    assert "音乐" in out["recalled_context"][0]


async def test_recall_pipeline_recalled_context_in_session_step() -> None:
    """ConversationSession(recall_enabled=True) + fake semantic → step 返回 recalled_context。

    验证 recall 节点的输出能被 _state_to_entry 收入 step dict。
    """
    from src.memory.client import MemoryClient
    from src.orchestration.runner import ConversationSession
    from src.orchestration.state import Stimulus

    semantic = _FakeSemanticStore(episodes=["用户喜欢古典音乐", "上次说过喜欢贝多芬"])
    mem = MemoryClient(semantic=semantic)
    session = ConversationSession(
        thread_id="t-recall-int",
        memory=mem,
        recall_enabled=True,
        rng_seed=42,
    )
    stim = Stimulus(name="音乐", goal_congruence=0.5, intensity=0.5)
    step = await session.step(stim)
    assert step["recalled_context"] is not None
    assert len(step["recalled_context"]) > 0


# ---------------------------------------------------------------------------
# Task 3：无记忆 no-op / 零回归
# ---------------------------------------------------------------------------


async def test_recall_noop_empty_memory_recalled_context_empty() -> None:
    """recall_enabled=True 但空 MemoryClient（无 semantic）→ recalled_context 为 []。"""
    from src.memory.client import MemoryClient
    from src.orchestration.memory_recall import MemoryRecallAgent
    from src.orchestration.state import AffectState, Stimulus

    mem = MemoryClient()  # 无 semantic store
    state = AffectState(
        recall_enabled=True,
        user_id="u1",
        stimulus=Stimulus(name="x", goal_congruence=0.0, intensity=0.3),
    )
    out = await MemoryRecallAgent(mem)(state)
    # 空 MemoryClient：recalled 为 [] → out 里不会有 recalled_context 键（或为空）
    assert out.get("recalled_context", []) == []


async def test_recall_disabled_step_returns_empty_recalled_context() -> None:
    """recall_enabled=False（默认）→ step recalled_context 为 []，行为与改前一致（零回归）。"""
    from src.orchestration.runner import ConversationSession
    from src.orchestration.state import Stimulus

    session = ConversationSession(
        thread_id="t-recall-off",
        recall_enabled=False,
        rng_seed=1,
    )
    stim = Stimulus(name="test", goal_congruence=0.3, intensity=0.5)
    step = await session.step(stim)
    assert step["recalled_context"] == [] or step["recalled_context"] is None


async def test_recall_disabled_converse_gets_empty_retrieved() -> None:
    """recall_enabled=False 时，main.py _run_chat 等效路径：recalled_str="" → 不注入背景。

    用 CapturingClient 直接验证 converse(retrieved="") 时 system prompt 不含背景。
    """
    from src.agents.language_openai import OpenAILanguageModel

    client = _CapturingClient()
    lm = OpenAILanguageModel(client=client, model="x")
    # 模拟 _run_chat 里 recalled_str="" 的路径
    await lm.converse([{"role": "user", "content": "hi"}], (0.0, 0.0), "")
    system_content = client.chat.completions.calls[0]["messages"][0]["content"]
    assert "你还记得以下背景" not in system_content


# ---------------------------------------------------------------------------
# Task 4：SteeringLanguageModel.converse retrieved 进入 _build_prompt
# ---------------------------------------------------------------------------


class _CapturingSteerBackend:
    """捕获 prompt + delta 的 fake SteerBackend。"""

    def __init__(self, reply: str = "steering-reply") -> None:
        self.reply = reply
        self.last_prompt: str | None = None
        self.last_delta: list[float] | None = None

    def generate_steered(self, prompt: str, delta: list[float]) -> str:
        self.last_prompt = prompt
        self.last_delta = delta
        return self.reply


async def test_steering_converse_retrieved_enters_prompt() -> None:
    """SteeringLanguageModel.converse(retrieved='bg') → retrieved 出现在 backend prompt。"""
    from src.agents.language_steering import SteeringLanguageModel

    backend = _CapturingSteerBackend(reply="ok")
    model = SteeringLanguageModel(
        w_valence=[1.0, 0.0],
        w_arousal=[0.0, 1.0],
        alpha=0.5,
        backend=backend,
    )
    history = [{"role": "user", "content": "test context"}]
    result = await model.converse(history, (0.3, 0.2), retrieved="bg_background")
    assert isinstance(result, str)
    assert result == "ok"
    assert backend.last_prompt is not None
    assert "bg_background" in backend.last_prompt


async def test_steering_converse_empty_retrieved_no_memory_line() -> None:
    """retrieved="" 时，_build_prompt 不追加记忆行（零回归：prompt 仅有上下文）。"""
    from src.agents.language_steering import SteeringLanguageModel

    backend = _CapturingSteerBackend()
    model = SteeringLanguageModel(
        w_valence=[1.0, 0.0],
        w_arousal=[0.0, 1.0],
        backend=backend,
    )
    history = [{"role": "user", "content": "hello"}]
    await model.converse(history, (0.0, 0.0))  # retrieved 默认 ""
    assert backend.last_prompt is not None
    assert "记忆" not in backend.last_prompt


async def test_steering_converse_retrieved_returns_nonempty_str() -> None:
    """steering converse 带 retrieved 时仍返回非空 str（基本契约）。"""
    from src.agents.language_steering import SteeringLanguageModel

    backend = _CapturingSteerBackend(reply="non-empty")
    model = SteeringLanguageModel(
        w_valence=[1.0, 0.0],
        w_arousal=[0.0, 1.0],
        backend=backend,
    )
    result = await model.converse(
        [{"role": "user", "content": "hi"}], (0.1, 0.1), retrieved="some bg"
    )
    assert isinstance(result, str)
    assert result != ""


async def test_steering_converse_with_fake_appraiser_and_retrieved() -> None:
    """fake appraiser + retrieved 注入：appraise_text 仍按注入 appraiser 运行（zero coupling）。"""
    from src.agents.language_steering import SteeringLanguageModel

    backend = _CapturingSteerBackend(reply="text")
    fixed = (0.4, -0.2)

    def fake_appraiser(text: str) -> tuple[float, float]:
        return fixed

    model = SteeringLanguageModel(
        w_valence=[1.0, 0.0],
        w_arousal=[0.0, 1.0],
        backend=backend,
        appraiser=fake_appraiser,
    )
    result = await model.converse(
        [{"role": "user", "content": "ctx"}], (0.3, 0.1), retrieved="retrieved_bg"
    )
    assert result == "text"
    # retrieved 进了 prompt
    assert "retrieved_bg" in (backend.last_prompt or "")


# ---------------------------------------------------------------------------
# Task 5：FakeConversationModel 满足 runtime_checkable ConversationModel 协议
# ---------------------------------------------------------------------------


class _FakeConversationModelWithRetrieved:
    """满足新签名（含 retrieved）的轻量 fake。"""

    async def converse(
        self,
        history: list[dict[str, str]],
        affect: tuple[float, float],
        retrieved: str = "",
        *,
        push: bool = False,
    ) -> str:
        return f"reply-retrieved={retrieved}"

    async def appraise_text(self, text: str) -> tuple[float, float]:
        return (0.1, 0.2)


def test_fake_with_retrieved_satisfies_conversation_model_protocol() -> None:
    """含 retrieved 参数的 fake 结构满足 runtime_checkable ConversationModel。"""
    from src.agents.language import ConversationModel

    lm = _FakeConversationModelWithRetrieved()
    assert isinstance(lm, ConversationModel)
    assert hasattr(lm, "converse")
    assert hasattr(lm, "appraise_text")
    assert inspect.iscoroutinefunction(lm.converse)
    assert inspect.iscoroutinefunction(lm.appraise_text)


async def test_fake_converse_retrieved_forwarded() -> None:
    """fake.converse(retrieved='X') 把 retrieved 正确传到回应里（参数转发验证）。"""
    lm = _FakeConversationModelWithRetrieved()
    result = await lm.converse([{"role": "user", "content": "hi"}], (0.0, 0.0), "X")
    assert "X" in result


# ---------------------------------------------------------------------------
# 额外零回归：openai converse push 行为在 retrieved="" 时与改前逐字一致
# ---------------------------------------------------------------------------


async def test_openai_converse_push_still_works_with_empty_retrieved() -> None:
    """push=True + retrieved="" 时，「用词会偏向」仍被注入（push 通路零回归）。"""
    from src.agents.language_openai import OpenAILanguageModel

    client = _CapturingClient()
    lm = OpenAILanguageModel(client=client, model="x")
    hist = [{"role": "user", "content": "hi"}]
    await lm.converse(hist, (-0.6, 0.5), "", push=True)
    system_content = client.chat.completions.calls[0]["messages"][0]["content"]
    assert "用词会偏向" in system_content


async def test_openai_converse_push_and_retrieved_both_injected() -> None:
    """push=True + retrieved 非空：两条注入都在 system prompt 里（互不排斥）。"""
    from src.agents.language_openai import OpenAILanguageModel

    client = _CapturingClient()
    lm = OpenAILanguageModel(client=client, model="x")
    hist = [{"role": "user", "content": "hi"}]
    await lm.converse(hist, (-0.6, 0.5), "some_context", push=True)
    sys_content = client.chat.completions.calls[0]["messages"][0]["content"]
    assert "用词会偏向" in sys_content
    assert "some_context" in sys_content
    assert "你还记得以下背景" in sys_content
