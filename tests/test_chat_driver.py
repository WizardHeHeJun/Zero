"""ChatDriver（对话核心从临时入口 main.py 迁出到编排层）单测：一轮 step 串联 + 两时间尺度 + 落盘。

用假 session 注入固定 e*、lm=None 走词典回退，把噪声 gauss monkeypatch 成 0 求确定性，
验证「评价→引擎→两时间尺度情绪→生成→落盘」一轮链路与状态推进（行为零回归的护栏）。

旋钮参数迁构造期后，测试策略：直接传参（ChatDriver(..., noise_std=0.05)）验证旋钮有效，
不再依赖构造后 setenv（monkeypatch 对已固化的 self.* 无效）。
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from src.orchestration.chat_driver import ChatDriver, ChatTurn
from src.storage.conversation_log import ConversationLog


class _FakeSession:
    """假 ConversationSession：step 返回固定 e*，并记录被喂入的 stim.text（gist 依赖）。"""

    def __init__(self, ev: tuple[float, float]) -> None:
        self.ev = ev
        self.seen_text: str | None = None

    async def step(self, stim: Any) -> dict[str, Any]:
        self.seen_text = stim.text
        return {"valence_arousal": self.ev, "recalled_context": []}


def _make_driver(
    log: ConversationLog,
    session: Any,
    *,
    lm: Any = None,
    attitude: tuple[float, float] = (0.0, 0.0),
    noise_std: float = 0.0,  # 默认关噪声使测试确定性；需测噪声时显式传值
) -> ChatDriver:
    return ChatDriver(
        thread="t",
        lm=lm,
        log=log,
        session=session,
        history=[],
        attitude=attitude,
        mode="test",
        noise_std=noise_std,
    )


async def test_step_returns_turn_and_persists(monkeypatch: pytest.MonkeyPatch) -> None:
    """一轮 step：返回 ChatTurn、词典回退回复含情绪标签、user+assistant+态度全部落盘。"""
    monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
    log = ConversationLog(":memory:")
    session = _FakeSession((0.6, 0.4))
    driver = _make_driver(log, session)
    turn = await driver.step("我很开心")
    assert isinstance(turn, ChatTurn)
    assert turn.emotion_label in turn.reply  # 词典回退模板带情绪标签
    assert session.seen_text == "我很开心"  # 本句 text 喂进引擎（gist 依赖）
    assert [r["content"] for r in log.recent("t", 10)] == ["我很开心", turn.reply]
    assert log.load_feeling("t") == pytest.approx(turn.attitude)  # 只持久化态度
    log.close()


async def test_two_timescale_both_move(monkeypatch: pytest.MonkeyPatch) -> None:
    """正向 e* 下，慢变 attitude 与快变 emotion 都从初值发生更新（两时间尺度都动）。"""
    monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
    log = ConversationLog(":memory:")
    driver = _make_driver(log, _FakeSession((0.8, 0.3)))
    turn = await driver.step("好棒")
    assert turn.attitude != (0.0, 0.0)
    assert turn.emotion != (0.0, 0.0)
    assert turn.attitude == driver.attitude  # 持有状态与返回同步
    log.close()


async def test_attitude_prior_is_pre_step_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """attitude_prior 是进入本轮前的快照（attitude_step 在 step 之后才更新）。"""
    monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
    log = ConversationLog(":memory:")
    driver = _make_driver(log, _FakeSession((0.8, 0.3)), attitude=(0.25, 0.1))
    turn = await driver.step("hi")
    assert turn.attitude_prior == 0.25
    assert turn.attitude[0] != 0.25  # 已被 attitude_step 推进
    log.close()


async def test_emotion_noise_std_default_005() -> None:
    """构造期 noise_std 默认 0.05：self.noise_std 值断言（构造层传参验证，替代原 monkeypatch）。"""
    log = ConversationLog(":memory:")
    session = _FakeSession((0.6, 0.4))
    driver = ChatDriver(
        thread="t",
        lm=None,
        log=log,
        session=session,
        history=[],
        attitude=(0.0, 0.0),
        mode="test",
        # 不传 noise_std → 使用默认值
    )
    assert driver.noise_std == 0.05  # 默认 0.05（逐字旧行为）
    log.close()


async def test_emotion_noise_std_override_zero() -> None:
    """noise_std=0 → 构造后 self.noise_std 为 0，step 中 gauss 以 sigma=0 调用（旋钮真生效）。"""
    sigmas: list[float] = []

    import src.orchestration.chat_driver as _mod

    original_gauss = _mod.random.gauss

    def capture_gauss(mu: float, sigma: float) -> float:
        sigmas.append(sigma)
        return 0.0

    _mod.random.gauss = capture_gauss  # type: ignore[assignment]
    try:
        log = ConversationLog(":memory:")
        driver = ChatDriver(
            thread="t",
            lm=None,
            log=log,
            session=_FakeSession((0.6, 0.4)),
            history=[],
            attitude=(0.0, 0.0),
            mode="test",
            noise_std=0.0,
        )
        await driver.step("x")
        assert sigmas and all(s == 0.0 for s in sigmas)
        log.close()
    finally:
        _mod.random.gauss = original_gauss  # type: ignore[assignment]


async def test_emotion_noise_reproducible_with_rng_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """P5：同一 rng_seed → 情绪噪声逐轮可复现（两 driver 的情绪序列逐字相等）。"""
    monkeypatch.setenv("ZERO_EMOTION_NOISE_STD", "0.05")  # 钉死噪声开（不受跑测环境 env 污染）
    log_a, log_b = ConversationLog(":memory:"), ConversationLog(":memory:")
    da = ChatDriver(
        thread="t",
        lm=None,
        log=log_a,
        session=_FakeSession((0.6, 0.4)),
        history=[],
        attitude=(0.0, 0.0),
        mode="test",
        rng_seed=123,
        noise_std=0.05,
    )
    db = ChatDriver(
        thread="t",
        lm=None,
        log=log_b,
        session=_FakeSession((0.6, 0.4)),
        history=[],
        attitude=(0.0, 0.0),
        mode="test",
        rng_seed=123,
        noise_std=0.05,
    )
    seq_a = [(await da.step("x")).emotion for _ in range(5)]
    seq_b = [(await db.step("x")).emotion for _ in range(5)]
    assert seq_a == seq_b  # 同 seed 逐字复现
    log_a.close()
    log_b.close()


async def test_step_emits_conversation_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """每轮 step 向 zero.conversation 发一条含 user 原文 + reply 的可读记录（对话内容日志）。"""
    monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
    records: list[logging.LogRecord] = []
    conv = logging.getLogger("zero.conversation")
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    conv.addHandler(handler)
    saved_level = conv.level
    conv.setLevel(logging.INFO)  # 具名 logger 默认 NOTSET → 显式置 INFO 才放行本记录
    try:
        log = ConversationLog(":memory:")
        driver = _make_driver(log, _FakeSession((0.6, 0.4)))
        turn = await driver.step("我很开心")
        log.close()
    finally:
        conv.removeHandler(handler)
        conv.setLevel(saved_level)
    assert len(records) == 1
    message = records[0].getMessage()
    assert "我很开心" in message  # user 原文
    assert turn.reply in message  # Zero 回复原文
    assert "第1轮" in message  # 轮次标注


async def test_build_chat_driver_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """工厂正确从 env 读取旋钮并透传给 ChatDriver.self.*（构造前 setenv → 构造 → 断言 self.*）。"""
    from src.orchestration.chat_driver import build_chat_driver

    monkeypatch.setenv("ZERO_EMOTION_NOISE_STD", "0.01")
    monkeypatch.setenv("ZERO_INTENSITY_FLOOR", "0.1")
    monkeypatch.setenv("ZERO_HABITUATION_TAU", "7")
    monkeypatch.setenv("ZERO_HISTORY_PRIMACY_K", "3")
    monkeypatch.setenv("ZERO_HISTORY_WINDOW", "20")
    monkeypatch.setenv("ZERO_RECALL_INJECT_MIN", "0.6")
    monkeypatch.setenv("ZERO_RECALL_IMPORTANCE_SCALE", "50")
    monkeypatch.setenv("ZERO_ATTITUDE_RATE_DECAY_K", "0.5")
    monkeypatch.setenv("ZERO_FAMILIARITY_TAU", "15")
    monkeypatch.setenv("ZERO_EMOTION_BASELINE_ATTITUDE_W", "0.8")
    # 不设 ZERO_OPENAI_API_KEY / ZERO_OPENAI_MODEL → lm=None（无网络）

    driver = build_chat_driver(thread="test-env-read")
    assert driver.noise_std == pytest.approx(0.01)
    assert driver.intensity_floor == pytest.approx(0.1)
    assert driver.hab_tau == pytest.approx(7.0)
    assert driver.primacy_k == 3
    assert driver.window_n == 20
    assert driver.inject_min == pytest.approx(0.6)
    assert driver.importance_scale == pytest.approx(50.0)
    assert driver.decay_k == pytest.approx(0.5)
    assert driver.fam_tau == pytest.approx(15.0)
    assert driver.baseline_w == pytest.approx(0.8)
