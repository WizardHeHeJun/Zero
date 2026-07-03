"""B6：standard_compliance 评价桥 + chat_driver 接线测试。

覆盖：
1. appraise_standard_compliance 纯函数（辱骂→负、致谢/道歉→正、中性→0、范围∈[-1,1]）。
2. 自指过滤（议会 2026-07-02 Item 1）：「我太蠢了」不应误判为攻击（net≥0）。
3. 三级加权语义：强烈脏话权重 > 标准辱骂 > 轻度贬义。
4. 零回归断言：门控关（standard_compliance_enabled=False）时 Stimulus.standard_compliance==0.0。
5. 端到端小验证：开启时辱骂输入 → standard_compliance<0 → occ_prior valence 更负。
"""

from __future__ import annotations

from typing import Any

import pytest

from src.agents.affect_math import occ_prior
from src.agents.emotion_lexicon import appraise_standard_compliance
from src.orchestration.chat_driver import ChatDriver
from src.orchestration.state import Stimulus
from src.storage.conversation_log import ConversationLog

# ---------------------------------------------------------------------------
# 1. appraise_standard_compliance 纯函数测试
# ---------------------------------------------------------------------------


def test_violation_text_returns_negative() -> None:
    """辱骂/脏话文本 → 负值（规范违反）。"""
    score = appraise_standard_compliance("你这个白痴，给我滚！")
    assert score < 0.0


def test_compliance_text_returns_positive() -> None:
    """致谢文本 → 正值（规范遵从）。"""
    score = appraise_standard_compliance("非常感谢你的帮助！")
    assert score > 0.0


def test_apology_returns_positive() -> None:
    """道歉文本 → 正值。"""
    score = appraise_standard_compliance("对不起，我之前说错了，抱歉。")
    assert score > 0.0


def test_neutral_text_returns_zero() -> None:
    """中性文本（无信号词）→ 0.0。"""
    score = appraise_standard_compliance("今天天气怎么样？")
    assert score == 0.0


def test_output_in_range() -> None:
    """输出始终在 [-1, 1]。"""
    cases = [
        "你这个废物垃圾混蛋滚开",  # 多违反词
        "谢谢谢谢谢谢谢谢谢谢谢",  # 多遵从词（堆词）
        "",  # 空串
        "hello world",
    ]
    for text in cases:
        s = appraise_standard_compliance(text)
        assert -1.0 <= s <= 1.0, f"out of range for text={text!r}: {s}"


def test_deterministic() -> None:
    """相同输入多次调用结果一致（确定性）。"""
    text = "你好，请问一下，谢谢！"
    assert appraise_standard_compliance(text) == appraise_standard_compliance(text)


def test_mixed_signal_net_direction() -> None:
    """同时有违反和遵从词时，净方向取决于哪类更多。"""
    # 纯违反
    v_only = appraise_standard_compliance("你真是个废物，给我滚")
    # 纯遵从
    c_only = appraise_standard_compliance("非常感谢，太棒了，你真好")
    assert v_only < 0.0
    assert c_only > 0.0


# ---------------------------------------------------------------------------
# 2. 零回归断言：门控关时 Stimulus.standard_compliance == 0.0
# ---------------------------------------------------------------------------


class _FakeSessionCapture:
    """假 session：捕获被喂入的 Stimulus 对象。"""

    def __init__(self) -> None:
        self.captured_stim: Stimulus | None = None

    async def step(self, stim: Any) -> dict[str, Any]:
        self.captured_stim = stim
        return {"valence_arousal": (0.0, 0.0), "recalled_context": []}


async def test_zero_regression_gate_off() -> None:
    """门控关（默认）时，Stimulus.standard_compliance 恒为默认 0.0（逐字零回归）。"""
    log = ConversationLog(":memory:")
    fake = _FakeSessionCapture()
    driver = ChatDriver(
        thread="t",
        lm=None,
        log=log,
        session=fake,  # type: ignore[arg-type]
        history=[],
        attitude=(0.0, 0.0),
        mode="test",
        noise_std=0.0,
        # standard_compliance_enabled 不传 → 默认 False
    )
    await driver.step("你这个白痴给我滚！")
    assert fake.captured_stim is not None
    # 门控关：无论输入多恶劣，standard_compliance 应为默认 0.0
    assert fake.captured_stim.standard_compliance == pytest.approx(0.0)
    log.close()


async def test_gate_off_is_default() -> None:
    """ChatDriver 构造时 standard_compliance_enabled 默认为 False。"""
    log = ConversationLog(":memory:")
    driver = ChatDriver(
        thread="t",
        lm=None,
        log=log,
        session=_FakeSessionCapture(),  # type: ignore[arg-type]
        history=[],
        attitude=(0.0, 0.0),
        mode="test",
    )
    assert driver.standard_compliance_enabled is False
    log.close()


# ---------------------------------------------------------------------------
# 3. 开启时接线生效：辱骂输入 → standard_compliance<0 → OCC valence 更负
# ---------------------------------------------------------------------------


async def test_gate_on_fills_negative_for_abuse() -> None:
    """开启后辱骂输入 → Stimulus.standard_compliance < 0。"""
    log = ConversationLog(":memory:")
    fake = _FakeSessionCapture()
    driver = ChatDriver(
        thread="t",
        lm=None,
        log=log,
        session=fake,  # type: ignore[arg-type]
        history=[],
        attitude=(0.0, 0.0),
        mode="test",
        noise_std=0.0,
        standard_compliance_enabled=True,
    )
    await driver.step("你这个混蛋废物，给我滚！")
    assert fake.captured_stim is not None
    assert fake.captured_stim.standard_compliance < 0.0


async def test_gate_on_fills_positive_for_thanks() -> None:
    """开启后致谢输入 → Stimulus.standard_compliance > 0。"""
    log = ConversationLog(":memory:")
    fake = _FakeSessionCapture()
    driver = ChatDriver(
        thread="t",
        lm=None,
        log=log,
        session=fake,  # type: ignore[arg-type]
        history=[],
        attitude=(0.0, 0.0),
        mode="test",
        noise_std=0.0,
        standard_compliance_enabled=True,
    )
    await driver.step("非常感谢你！谢谢！")
    assert fake.captured_stim is not None
    assert fake.captured_stim.standard_compliance > 0.0


def test_occ_prior_valence_more_negative_with_violation() -> None:
    """端到端：standard_compliance<0 经 occ_prior 使 prior_mu valence 更负（OCC 分支 B 通电）。

    occ_prior 文档：standard_compliance 以 0.3 权重进 valence。
    负的 standard_compliance → prior_mu[0] 比 0 时更负。
    """
    # 基准：standard_compliance=0（门控关的等价状态）
    mu_baseline, _, _ = occ_prior(
        goal_congruence=0.0,
        standard_compliance=0.0,
        attitude_appeal=0.0,
        intensity=1.0,
    )
    # 辱骂场景：standard_compliance 取典型负值 -0.5
    sc_neg = appraise_standard_compliance("你这个废物，给我滚！")
    assert sc_neg < 0.0  # 先验证评价桥本身
    mu_abuse, _, _ = occ_prior(
        goal_congruence=0.0,
        standard_compliance=sc_neg,
        attitude_appeal=0.0,
        intensity=1.0,
    )
    # 辱骂场景的 prior valence 应比基准更负
    assert mu_abuse[0] < mu_baseline[0], (
        f"expected valence({mu_abuse[0]:.3f}) < baseline({mu_baseline[0]:.3f})"
    )


def test_build_chat_driver_reads_zero_standard_compliance_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """工厂从 ZERO_STANDARD_COMPLIANCE env 读取并透传给 ChatDriver.standard_compliance_enabled。"""
    from src.orchestration.chat_driver import build_chat_driver

    monkeypatch.setenv("ZERO_STANDARD_COMPLIANCE", "1")
    driver = build_chat_driver(thread="test-sc-env")
    assert driver.standard_compliance_enabled is True


def test_build_chat_driver_default_gate_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """工厂未设 ZERO_STANDARD_COMPLIANCE → standard_compliance_enabled=False（默认关）。"""
    from src.orchestration.chat_driver import build_chat_driver

    monkeypatch.delenv("ZERO_STANDARD_COMPLIANCE", raising=False)
    driver = build_chat_driver(thread="test-sc-default")
    assert driver.standard_compliance_enabled is False


# ---------------------------------------------------------------------------
# 议会 2026-07-02 Item 1 精化：自指过滤 + 三级加权
# ---------------------------------------------------------------------------


def test_self_referential_shame_not_penalized() -> None:
    """「我太蠢了」含自指 → 不误判为对他人攻击，net≥0（OCC shame 非 reproach）。

    议会 Item 1：violation 词前窗口含「我」→ 自指过滤，不计负分。
    """
    score = appraise_standard_compliance("我太蠢了，怎么又做错了")
    assert score >= 0.0, f"自指自责不应被判为规范违反，得分={score:.3f}"


def test_self_ref_with_ziji_not_penalized() -> None:
    """「自己」前缀自指→ 不计 violation 分。"""
    score = appraise_standard_compliance("自己真笨，记不住")
    assert score >= 0.0, f"自指自责不应被判为规范违反，得分={score:.3f}"


def test_non_self_ref_abuse_still_penalized() -> None:
    """无自指前缀的辱骂词 → 仍计负分（自指过滤不误伤真实攻击）。"""
    score = appraise_standard_compliance("你真是太蠢了")
    assert score < 0.0, f"对他人辱骂应计负分，得分={score:.3f}"


def test_weighted_strong_violation_more_negative_than_mild() -> None:
    """强烈脏话（权重 1.5）比轻度贬义（权重 0.5）产生更大负值。

    验证三级加权语义（议会 Item 1）：权重差异体现在得分差异上。
    """
    mild_score = appraise_standard_compliance("你真无聊")  # 轻度 0.5
    strong_score = appraise_standard_compliance("去你的fuck")  # 强烈 1.5
    assert mild_score > strong_score, f"轻度({mild_score:.3f}) 应比强烈({strong_score:.3f})负值小"


def test_weighted_strong_compliance_more_positive_than_mild() -> None:
    """强烈赞许（权重 1.5）比轻度礼貌（权重 0.5）产生更大正值。"""
    mild_score = appraise_standard_compliance("请")  # 轻度 0.5
    strong_score = appraise_standard_compliance("非常感谢")  # 强烈 1.5
    assert strong_score > mild_score, f"强烈({strong_score:.3f}) 应比轻度({mild_score:.3f})正值大"


def test_stacking_violation_capped_at_max_signals() -> None:
    """堆叠多个 violation 词权重之和超过 _MAX_SIGNALS 时被钳制 → 仍返回 -1.0。"""
    # 去死(1.5)+fuck(1.5)+nmsl(1.5)+草(1.5) = 6.0 > _MAX_SIGNALS=4 → v_score=1.0 → net=-1.0
    score = appraise_standard_compliance("去死 fuck nmsl 草")
    assert score == pytest.approx(-1.0)


def test_stacking_compliance_capped_at_max_signals() -> None:
    """堆叠多个 compliance 词权重之和超过 _MAX_SIGNALS 时被钳制 → 仍返回 1.0。"""
    # 非常感谢(1.5)+amazing(1.5)+well done(1.5)+佩服(1.5) = 6.0 > 4 → c_score=1.0 → net=1.0
    score = appraise_standard_compliance("非常感谢 amazing well done 佩服")
    assert score == pytest.approx(1.0)
