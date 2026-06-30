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


def test_session_threads_sample_sigma_cap_into_flags() -> None:
    """sample_sigma_cap 经 ConversationSession 进 flags；默认 None（防抖旋钮零回归）。"""
    capped = ConversationSession(thread_id="t-sigma", sample_sigma_cap=0.1)
    assert capped.flags["sample_sigma_cap"] == 0.1
    default = ConversationSession(thread_id="t-sigma-def")
    assert default.flags["sample_sigma_cap"] is None


def test_session_threads_affect_readout_into_flags() -> None:
    """affect_readout 经 ConversationSession 进 flags；默认 'sample'（P4 读出旋钮零回归）。"""
    mapped = ConversationSession(thread_id="t-readout", affect_readout="map")
    assert mapped.flags["affect_readout"] == "map"
    default = ConversationSession(thread_id="t-readout-def")
    assert default.flags["affect_readout"] == "sample"


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


async def test_converse_push_injects_word_tendency_not_pull() -> None:
    """push（皮层下/不随意）：affect-congruent 用词倾向注入系统提示——情绪经用词漏出，非'演情绪'。
    关闭 push（pull）则不注入。
    """
    captured: dict[str, object] = {}

    class _Cap:
        async def create(self, **kwargs: object) -> _Resp:
            captured.update(kwargs)
            return _Resp("嗯。")

    class _Chat2:
        def __init__(self) -> None:
            self.completions = _Cap()

    class _Client2:
        def __init__(self) -> None:
            self.chat = _Chat2()

    lm = OpenAILanguageModel(client=_Client2(), model="x")
    hist = [{"role": "user", "content": "hi"}]
    await lm.converse(hist, (-0.6, 0.5), push=True)
    assert "用词会偏向" in captured["messages"][0]["content"]  # push 词倾向注入
    await lm.converse(hist, (-0.6, 0.5), push=False)
    assert "用词会偏向" not in captured["messages"][0]["content"]  # pull 不注入


def test_build_logit_bias_gated_off_by_default(monkeypatch) -> None:
    """解码期 logit_bias 默认关（需 env + 兼容 tokenizer）；不设 env → 空（graceful）。"""
    monkeypatch.delenv("ZERO_PUSH_LOGIT_BIAS", raising=False)
    lm = OpenAILanguageModel(client=_FakeClient("x"), model="x")
    assert lm._build_logit_bias((-0.6, 0.5), ["愤怒", "烦躁"]) == {}


def test_emotion_is_short_lived_decays_to_baseline() -> None:
    """快变情绪：单刺激飙起，刺激停止后几轮内衰退回基线（情绪短时、不长期累积）。"""
    from src.agents.affect_math import emotion_decay_step

    base = (0.0, 0.0)
    spike = emotion_decay_step(base, base, (-0.7, 0.5))
    assert spike[0] < -0.2  # 单刺激即飙起（情绪反应）
    emotion = spike
    for _ in range(4):
        emotion = emotion_decay_step(emotion, base, (0.0, 0.0))  # 刺激停止
    assert abs(emotion[0]) < 0.1 and abs(emotion[1]) < 0.1  # 回到基线附近（短时）


def test_emotion_recovers_to_attitude_not_absolute_neutral() -> None:
    """怒火退去后回到「对此人的态度」基线，而非绝对中性（态度=情绪衰退的落点）。"""
    from src.agents.affect_math import emotion_decay_step

    soured = (-0.4, 0.0)  # 对此人态度已变冷
    emotion = (-0.7, 0.3)  # 当前怒火
    for _ in range(6):
        emotion = emotion_decay_step(emotion, soured, (0.0, 0.0))
    assert abs(emotion[0] - soured[0]) < 0.08  # 收敛到态度基线，不是 0


def test_attitude_accumulates_slowly_object_bound() -> None:
    """慢变态度：单步几乎不动（长期印象不因一句话定型）；持续负面多轮才慢慢变冷。"""
    from src.agents.affect_math import attitude_step

    one = attitude_step((0.0, 0.0), (-0.8, 0.0))
    assert one[0] > -0.15  # 单步只挪一点点
    attitude = (0.0, 0.0)
    for _ in range(20):
        attitude = attitude_step(attitude, (-0.8, 0.0))
    assert attitude[0] < -0.5  # 多轮持续负面 → 态度才成形变冷


def test_attitude_reverts_toward_setpoint_bounds_ratchet() -> None:
    """议会 B：态度含向 setpoint 的弱回归——持续同向刺激稳态被钳在 |s| 内（防单调棘轮漂移到极端）。

    reversion=0 退化为旧纯 EWMA（稳态趋近 s）；默认含 reversion 时稳态 a*=rate·s/(rate+reversion)，
    小于 s，验证回归项确实压低了累积上限（affective homeostasis）。
    """
    from src.agents.affect_math import attitude_step

    s = (0.8, 0.0)
    a_rev = (0.0, 0.0)
    a_pure = (0.0, 0.0)
    for _ in range(200):
        a_rev = attitude_step(a_rev, s)  # 默认含 reversion
        a_pure = attitude_step(a_pure, s, reversion=0.0)  # 旧纯 EWMA（零回归对照）
    assert a_pure[0] > 0.78  # 纯 EWMA 稳态趋近 s=0.8
    assert a_rev[0] < a_pure[0]  # 回归项压低稳态（防棘轮）
    assert a_rev[0] < 0.8  # 始终被钳在 |s| 内，不无限漂移


def test_conversation_log_roundtrip_and_feeling() -> None:
    """对话 transcript + 累积情绪落库/重载（跨重启记忆 + 情绪续上的本地存储基元）。"""
    from src.storage.conversation_log import ConversationLog

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
