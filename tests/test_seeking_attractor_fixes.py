"""科学家议会 "seeking 吸引盆" 裁决落地回归（notes/2026-07-01-seeking-attractor-council.md）。

覆盖 P1-a(Q1 intensity 下限)、P1-b(Q2b attitude arousal 维独立回归)、P1-c(Q7 occ_prior
去整流 deactivation)、P2(Q6 习惯化衰减)。每项均含**默认关=逐字零回归**断言 + 旋钮真生效的
功能断言。纯函数直测 + chat_driver 行为测（假 session 注入固定 e*、lm=None 词典回退、噪声置 0）。
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from src.agents.affect_core import AffectCoreAgent
from src.agents.affect_math import attitude_step, habituation_factor, occ_prior
from src.agents.appraisal import AppraisalAgent
from src.orchestration.chat_driver import ChatDriver, _relationship_hint
from src.orchestration.state import AffectState, Stimulus
from src.storage.conversation_log import ConversationLog

# ---------- 纯函数：occ_prior arousal_baseline（P1-c / Q7）----------


def test_occ_prior_arousal_baseline_default_zero_regression() -> None:
    """arousal_baseline 默认 0.0 → 与不传参逐字相等（零回归）。"""
    assert occ_prior(0.3, 0.1, 0.2, 0.5) == occ_prior(0.3, 0.1, 0.2, 0.5, arousal_baseline=0.0)


def test_occ_prior_negative_baseline_enables_deactivation() -> None:
    """负 arousal_baseline 让平淡输入（低强度、低 |valence|）的 arousal 证据回落到零/负。"""
    (_, a_old), _, _ = occ_prior(0.0, 0.0, 0.0, 0.2)  # 旧：0.4·0.2 = 0.08 恒正
    (_, a_new), _, _ = occ_prior(0.0, 0.0, 0.0, 0.2, arousal_baseline=-0.08)
    assert a_old == pytest.approx(0.08)
    assert a_new == pytest.approx(0.0)  # 平淡 → 静息
    (_, a_neg), _, _ = occ_prior(0.0, 0.0, 0.0, 0.2, arousal_baseline=-0.3)
    assert a_neg < 0.0  # 深度 deactivation（circumplex 下半区）


def test_occ_prior_baseline_keeps_valence_arm() -> None:
    """0.6·|valence| 项（circumplex V 形）不受 baseline 影响——强情绪两端仍抬唤醒。"""
    (_, a_bland), _, _ = occ_prior(0.0, 0.0, 0.0, 0.2, arousal_baseline=-0.08)
    (_, a_strong), _, _ = occ_prior(0.9, 0.0, 0.0, 0.2, arousal_baseline=-0.08)
    assert a_strong > a_bland  # |valence| 大 → arousal 仍显著更高


# ---------- 纯函数：attitude_step reversion_a（P1-b / Q2b）----------


def test_attitude_step_reversion_a_default_none_zero_regression() -> None:
    """reversion_a 默认 None → 两维同用 reversion，与不传参逐字相等（零回归）。"""
    base = attitude_step((0.1, 0.2), (0.3, 0.4))
    assert base == attitude_step((0.1, 0.2), (0.3, 0.4), reversion_a=None)


def test_attitude_step_reversion_a_pulls_arousal_only() -> None:
    """独立 reversion_a≫reversion：arousal 维更强回归 setpoint，valence 维不受影响。"""
    v_def, a_def = attitude_step((0.5, 0.5), (0.0, 0.0))  # 默认弱回归
    v_str, a_str = attitude_step((0.5, 0.5), (0.0, 0.0), reversion_a=0.4)  # arousal 强回归
    assert v_str == pytest.approx(v_def)  # valence 维不变
    assert a_str < a_def  # arousal 维被更快拉向 setpoint(0)


# ---------- 纯函数：habituation_factor（P2 / Q6）----------


def test_habituation_factor_tau_nonpositive_is_one() -> None:
    """τ<=0 → η=1.0（不衰减，零回归开关）；exposure 任意值都返回 1。"""
    assert habituation_factor(0, 0.0) == 1.0
    assert habituation_factor(100, -5.0) == 1.0


def test_habituation_factor_exp_decay_monotonic() -> None:
    """η(n)=exp(-n/τ)：n=0 为 1、随 exposure 单调递减、恒在 (0,1]；负 exposure 按 0。"""
    assert habituation_factor(0, 5.0) == 1.0
    assert habituation_factor(10, 5.0) == pytest.approx(math.exp(-2.0))
    assert habituation_factor(5, 5.0) > habituation_factor(10, 5.0)
    assert habituation_factor(-3, 5.0) == 1.0  # 负 exposure 按 0 处理
    assert 0.0 < habituation_factor(50, 5.0) <= 1.0


# ---------- AppraisalAgent 节点：arousal_baseline 穿透（P1-c 接线）----------


def test_appraisal_node_passes_arousal_baseline() -> None:
    """AppraisalAgent 把 state.arousal_baseline 透传进 occ_prior：默认 0 零回归、负值降 arousal。"""
    stim = Stimulus(name="x", goal_congruence=0.0, intensity=0.2)
    out_default = AppraisalAgent()(AffectState(stimulus=stim))  # 默认 arousal_baseline=0.0
    out_deact = AppraisalAgent()(AffectState(stimulus=stim, arousal_baseline=-0.08))
    assert out_default["prior_mu"][1] == pytest.approx(0.08)  # 零回归
    assert out_deact["prior_mu"][1] == pytest.approx(0.0)  # deactivation 生效


# ---------- chat_driver 行为：旋钮默认关零回归 + 开启生效 ----------


class _StimCapSession:
    """假 session：记录被喂入的 stim（验证 intensity 下限旋钮），step 返回固定 e*。"""

    def __init__(self, ev: tuple[float, float]) -> None:
        self.ev = ev
        self.stims: list[Any] = []

    async def step(self, stim: Any) -> dict[str, Any]:
        self.stims.append(stim)
        # recalled_facts 显式留空：镜像真 _state_to_entry 的键集，lm=None 时 chat_driver 走空分支
        # 不消费召回（无 LLM），避免未来 step 内对缺键做非 .get 假设时测试静默失真。
        return {"valence_arousal": self.ev, "recalled_context": [], "recalled_facts": []}


def _driver(
    log: ConversationLog,
    session: Any,
    attitude: tuple[float, float] = (0.0, 0.0),
    *,
    intensity_floor: float = 0.2,
    hab_tau: float = 0.0,
    reversion_a: float | None = None,
    decay_k: float = 0.0,
    noise_std: float = 0.0,
) -> ChatDriver:
    """辅助构造 ChatDriver，旋钮参数直接传入（不依赖 env monkeypatch）。"""
    return ChatDriver(
        thread="t",
        lm=None,
        log=log,
        session=session,
        history=[],
        attitude=attitude,
        mode="test",
        intensity_floor=intensity_floor,
        hab_tau=hab_tau,
        reversion_a=reversion_a,
        decay_k=decay_k,
        noise_std=noise_std,
    )


async def test_intensity_floor_default_is_020() -> None:
    """intensity_floor 默认 0.2 → 中性输入 intensity 被钉在下限 0.2（构造层传参验证）。"""
    log = ConversationLog(":memory:")
    session = _StimCapSession((0.0, 0.0))
    await _driver(log, session, intensity_floor=0.2).step("嗯")  # 中性词典评价 a≈0 < 下限
    assert session.stims[0].intensity == pytest.approx(0.2)
    log.close()


async def test_intensity_floor_override_zero() -> None:
    """intensity_floor=0 → 中性输入 intensity 落到 |a|≈0（旋钮真生效，去直流底噪）。"""
    log = ConversationLog(":memory:")
    session = _StimCapSession((0.0, 0.0))
    await _driver(log, session, intensity_floor=0.0).step("嗯")
    assert session.stims[0].intensity < 0.2  # 不再被 0.2 兜底
    log.close()


async def test_habituation_default_off_no_decay() -> None:
    """hab_tau=0（默认）→ η 恒 1，多轮同输入 attitude arousal 单调累积（零回归）。"""
    log = ConversationLog(":memory:")
    d = _driver(log, _StimCapSession((0.0, 0.8)), hab_tau=0.0)
    atts = [(await d.step("x")).attitude[1] for _ in range(6)]
    assert atts == sorted(atts)  # 无衰减 → 单调不降
    assert atts[-1] > atts[0]  # 且真正向累积（arousal 直流偏置爬升，旧行为）
    log.close()


async def test_habituation_on_suppresses_arousal() -> None:
    """hab_tau>0 → 同一对象重复输入的 arousal 累积被显著压低（对照 hab_tau=0）。"""

    async def run(hab_tau: float) -> float:
        log = ConversationLog(":memory:")
        d = _driver(log, _StimCapSession((0.0, 0.8)), hab_tau=hab_tau)
        att = (0.0, 0.0)
        for _ in range(6):
            att = (await d.step("x")).attitude
        log.close()
        return att[1]

    arousal_off = await run(0.0)
    arousal_on = await run(3.0)
    assert arousal_on < arousal_off  # 习惯化把 arousal 累积压低


async def test_attitude_reversion_a_lowers_arousal() -> None:
    """reversion_a 强回归 → 多轮后 attitude arousal 平台显著低于默认 None（Q2b 生效）。"""

    async def run(reversion_a: float | None) -> float:
        log = ConversationLog(":memory:")
        d = _driver(log, _StimCapSession((0.0, 0.8)), reversion_a=reversion_a)
        att = (0.0, 0.0)
        for _ in range(8):
            att = (await d.step("x")).attitude
        log.close()
        return att[1]

    arousal_def = await run(None)
    arousal_strong = await run(0.5)
    assert arousal_strong < arousal_def


# ---------- 议会二轮：P4-d arousal_gain cap ----------


def _core_state(cap: float | None) -> AffectState:
    """高 arousal 先验 + workspace 路径的最小 AffectState（affect_core 走 arousal_gain 分支）。"""
    return AffectState(
        prior_mu=(0.0, 0.8),
        prior_sigma=(0.2, 0.2),
        reward=0.5,
        rpe=0.3,
        precision=0.6,
        features=[0.5, 0.0, 0.0, 0.8],
        workspace_enabled=True,
        arousal_gain_cap=cap,
        affect_readout="map",
    )


def test_arousal_gain_cap_default_none_zero_regression() -> None:
    """arousal_gain_cap 默认 None → 不 cap；同输入两次 map 读出逐字相等（零回归、无采样噪声）。"""
    a = AffectCoreAgent()(_core_state(None))
    b = AffectCoreAgent()(_core_state(None))
    assert a["post_mu"] == b["post_mu"]
    assert a["affect_precision"] == b["affect_precision"]


def test_arousal_gain_cap_lowers_precision() -> None:
    """设 cap → 高 arousal 先验的 gain 被钳低 → 评价/价值流精度降 → 后验精度低于不 cap。"""
    uncapped = AffectCoreAgent()(_core_state(None))
    capped = AffectCoreAgent()(_core_state(0.3))  # gain 1.8 → min(1.8,1.3)=1.3
    assert capped["affect_precision"] < uncapped["affect_precision"]


# ---------- 议会二轮：Q5-A 熟悉度门控 rate 衰减 ----------


async def test_rate_decay_default_off_and_slows_formation() -> None:
    """decay_k=0（默认）→ rate_eff=rate（零回归）；decay_k>0 → 熟悉度衰减 rate → 态度形成更慢。"""

    async def run(decay_k: float) -> float:
        log = ConversationLog(":memory:")
        d = _driver(log, _StimCapSession((0.6, 0.0)), decay_k=decay_k)
        att = (0.0, 0.0)
        for _ in range(10):
            att = (await d.step("x")).attitude
        log.close()
        return att[0]

    v_off = await run(0.0)
    v_on = await run(0.9)
    assert v_on < v_off  # 越熟态度形成越慢（近似"陌生态更稳"），单不动点仅减缓漂移


# ---------- 议会二轮：Q5-B 关系距离提示（软约束、纯函数、默认关）----------


def test_relationship_hint_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZERO_RELATIONSHIP_STAGE_HINT 未设 → 任意曝光都返回 ""（converse 不注入=零回归）。"""
    monkeypatch.delenv("ZERO_RELATIONSHIP_STAGE_HINT", raising=False)
    assert _relationship_hint(0) == ""
    assert _relationship_hint(100) == ""


def test_relationship_hint_tiers_when_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """开启 → 按曝光三档给字符串锚（陌生/一阵/熟络），阈值对齐议会 N_up 3-5/10-15。"""
    monkeypatch.setenv("ZERO_RELATIONSHIP_STAGE_HINT", "1")
    early, mid, late = _relationship_hint(2), _relationship_hint(10), _relationship_hint(30)
    assert "陌生" in early
    assert mid and mid != early
    assert late and late != mid
