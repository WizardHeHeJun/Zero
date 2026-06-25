"""交互对话基元：ConversationSession 跨轮持久（mood/情绪连续）+ 评价桥 appraise_text。

ConversationSession 用模板语言模型（torch/API-free），不依赖真 LLM；appraise_text 用
fake async client（同 test_language_openai），不依赖 openai 包。
"""

from __future__ import annotations

from src.agents.language_openai import OpenAILanguageModel
from src.orchestration.runner import ConversationSession
from src.orchestration.state import Stimulus


async def test_session_persists_mood_across_turns() -> None:
    """同一会话连灌负面刺激：mood 跨轮累积变负（运行态跨 step 持久 = 情绪连续/A.7 滞后）。"""
    session = ConversationSession(
        thread_id="t-conv-neg", mood_enabled=True, workspace_enabled=True, rng_seed=7
    )
    neg = Stimulus(name="被否决", goal_congruence=-0.8, standard_compliance=-0.6, intensity=0.8)
    moods = [(await session.step(neg))["mood"] for _ in range(4)]
    assert all(m is not None for m in moods)
    assert moods[0][0] < 0  # 首轮已偏负
    assert moods[-1][0] < moods[0][0]  # 跨轮累积更负（持久 + 双稳滞后），证明运行态跨轮留存


async def test_session_emits_language_and_ignited_streams() -> None:
    """单轮：workspace+language 开启 → 有生成语言 + 非空点燃流。"""
    session = ConversationSession(
        thread_id="t-conv-pos", language_enabled=True, workspace_enabled=True, rng_seed=7
    )
    step = await session.step(Stimulus(name="拿到 offer", goal_congruence=0.8, intensity=0.6))
    assert step["language_text"] is not None
    assert step["ignited_streams"]  # 工作空间点燃流非空
    assert step["valence_arousal"] is not None


async def test_session_value_table_persists_across_turns() -> None:
    """同一 stimulus 反复出现：在线 TD 的 value_estimate 跨轮变化（value_table 持久）。"""
    session = ConversationSession(thread_id="t-conv-td", rng_seed=7)
    stim = Stimulus(name="repeat", goal_congruence=0.7, intensity=0.5)
    first = (await session.step(stim))["value_estimate"]
    for _ in range(3):
        last = (await session.step(stim))["value_estimate"]
    assert first is not None and last is not None
    assert last != first  # 价值随重复在线学习（跨轮 checkpoint 持久）


# ---------- 评价桥 appraise_text（fake client，不依赖 openai） ----------


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
    def __init__(self, content: str) -> None:
        self.content = content

    async def create(self, **kwargs: object) -> _Resp:
        return _Resp(self.content)


class _Chat:
    def __init__(self, content: str) -> None:
        self.completions = _Completions(content)


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = _Chat(content)


async def test_appraise_text_reads_user_affect() -> None:
    """评价桥：把用户文本客观读成 (v,a)（复用独立 VAD 反推）。"""
    lm = OpenAILanguageModel(client=_FakeClient('{"valence": -0.7, "arousal": 0.6}'), model="x")
    v, a = await lm.appraise_text("你怎么能这样对我！")
    assert v == -0.7
    assert a == 0.6


async def test_converse_passes_history_and_returns_reply() -> None:
    """自然对话：converse 带历史调用、返回回应文本（不强制 VAD，不戏剧化）。"""
    captured: dict[str, object] = {}

    class _CapturingCompletions:
        async def create(self, **kwargs: object) -> _Resp:
            captured.update(kwargs)
            return _Resp("嗯，我记得你刚说的。")

    class _CapturingChat:
        def __init__(self) -> None:
            self.completions = _CapturingCompletions()

    class _CapturingClient:
        def __init__(self) -> None:
            self.chat = _CapturingChat()

    lm = OpenAILanguageModel(client=_CapturingClient(), model="x")
    history = [
        {"role": "user", "content": "我叫小明"},
        {"role": "assistant", "content": "你好小明"},
        {"role": "user", "content": "我刚说我叫什么？"},
    ]
    reply = await lm.converse(history, (-0.2, 0.1))
    assert reply == "嗯，我记得你刚说的。"
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"  # 自然对话系统提示
    assert messages[1:] == history  # 完整历史原样带入（连贯性）


def test_leaky_feeling_responds_not_latches() -> None:
    """chat 情绪积累用泄漏积分（mood_step self_gain=0）：持续负面 → 逐步变负（响应输入），
    而非 A.7 双稳从正盆自锁、被骂也不动（后者正是 chat 不能用它的原因）。
    """
    from src.agents.affect_math import mood_step

    feeling = (0.2, 0.1)  # 起步略正（开场寒暄）
    neg = (-0.6, 0.5)
    one = mood_step(feeling, neg, inertia=0.7, self_gain=0.0, drive=0.3)
    assert one[0] < feeling[0]  # 单步即向负移动（响应），但渐进
    leaky = feeling
    for _ in range(8):
        leaky = mood_step(leaky, neg, inertia=0.7, self_gain=0.0, drive=0.3)
    assert leaky[0] < -0.4  # 持续负面累积到明显负（会动怒/受伤）

    bistable = (0.5, 0.1)
    for _ in range(8):
        bistable = mood_step(bistable, neg)  # 默认双稳（A.7）
    assert bistable[0] > leaky[0]  # 双稳黏正盆、不响应 → 实证 chat 改用泄漏积分的必要


def test_conversation_log_roundtrip_and_feeling() -> None:
    """对话 transcript + 累积情绪落库/重载（跨重启记忆 + 情绪续上的本地存储基元）。"""
    from main import ConversationLog

    log = ConversationLog(path=":memory:")
    log.append("t1", "user", "你好")
    log.append("t1", "assistant", "嗨")
    log.append("t2", "user", "别的线程")
    assert log.recent("t1", limit=10) == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "嗨"},
    ]  # 按时间正序、按 thread 隔离
    assert log.load_feeling("t1") == (0.0, 0.0)  # 无记录 → 平静起步
    log.save_feeling("t1", (-0.4, 0.3))
    log.save_feeling("t1", (-0.5, 0.2))  # upsert 覆盖
    assert log.load_feeling("t1") == (-0.5, 0.2)
