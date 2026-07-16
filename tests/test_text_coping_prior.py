"""text_coping_prior 独立标量流单测（议会 2026-07-16 B3 相 2 落地）。

覆盖范围（对应任务 B-G）：
  B. AppraisalAgent B3 四分支数值：分支1-4 cp 值 + text_coping_source flag。
  C. motivational_system 中间带哑火：text_coping_source=True/False 行为区分。
  D. 每轮归零 spy（R6·关键）：step() base dict text_coping_prior=None / text_coping_source=False。
  E. CS 白名单静态断言（R4）：B3 区块不含 fuse_terms/occ_prior 等消费者。
  F. AffectState / SessionConfig 字段存在性。
  G. SessionConfig.text_coping_precision 精度上限 fail-fast（le=0.10）。
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from src.agents.appraisal import AppraisalAgent
from src.agents.emotion_lexicon import motivational_system
from src.orchestration.runner import SessionConfig
from src.orchestration.state import AffectState, Stimulus

# ─────────────────────────────────────────────────────────────────
# B. AppraisalAgent B3 四分支数值测试
# ─────────────────────────────────────────────────────────────────


def _make_state_b3(
    *,
    control_appraisal: float | None,
    text_coping_prior: float | None,
    text_coping_precision: float = 0.08,
    coping_potential_enabled: bool = True,
    text_coping_enabled: bool = True,
) -> AffectState:
    """B3 测试专用 helper：构造含 control_appraisal（float|None）的 AffectState。

    text_coping_enabled 默认 True：B3 分支逻辑测试需要门开才能让 text 参与分支2/4。
    gate=False 的零回归测试由 TestTextCopingGate 覆盖（使用 _make_state_b3_with_gate）。
    """
    stim = Stimulus(
        name="b3_test", goal_congruence=0.0, intensity=0.5, control_appraisal=control_appraisal
    )
    return AffectState(
        stimulus=stim,
        coping_potential_enabled=coping_potential_enabled,
        text_coping_enabled=text_coping_enabled,
        text_coping_prior=text_coping_prior,
        text_coping_precision=text_coping_precision,
    )


class TestAppraisalAgentB3Branches:
    """AppraisalAgent B3 四分支数值 + text_coping_source flag。"""

    # ──────────── 分支1：ctrl=None & text=None → cp=0.0, src_flag=False ────────────

    def test_branch1_both_none_cp_is_zero(self) -> None:
        """分支1：ctrl=None & text=None → coping_potential_state=0.0。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(control_appraisal=None, text_coping_prior=None)
        out = agent(state)
        # 分支1 仍写 coping_updates
        assert out["coping_potential_state"] == pytest.approx(0.0)

    def test_branch1_both_none_src_flag_false(self) -> None:
        """分支1：ctrl=None & text=None → text_coping_source=False。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(control_appraisal=None, text_coping_prior=None)
        out = agent(state)
        assert out["text_coping_source"] is False

    # ──────────── 分支2：ctrl=None & text=v → cp=clamp(text), src_flag=True ────────────

    def test_branch2_text_only_cp_equals_clamped_text(self) -> None:
        """分支2：ctrl=None, text=0.3 → coping_potential_state≈0.3（在范围内无 clamp）。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(control_appraisal=None, text_coping_prior=0.3)
        out = agent(state)
        assert out["coping_potential_state"] == pytest.approx(0.3)

    def test_branch2_text_only_negative(self) -> None:
        """分支2：ctrl=None, text=-0.5 → coping_potential_state≈-0.5。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(control_appraisal=None, text_coping_prior=-0.5)
        out = agent(state)
        assert out["coping_potential_state"] == pytest.approx(-0.5)

    def test_branch2_text_only_clamped_above(self) -> None:
        """分支2：ctrl=None, text=2.0 → clamp 到 1.0。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(control_appraisal=None, text_coping_prior=2.0)
        out = agent(state)
        assert out["coping_potential_state"] == pytest.approx(1.0)

    def test_branch2_text_only_clamped_below(self) -> None:
        """分支2：ctrl=None, text=-2.0 → clamp 到 -1.0。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(control_appraisal=None, text_coping_prior=-2.0)
        out = agent(state)
        assert out["coping_potential_state"] == pytest.approx(-1.0)

    def test_branch2_src_flag_true(self) -> None:
        """分支2：ctrl=None, text=v → text_coping_source=True。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(control_appraisal=None, text_coping_prior=0.3)
        out = agent(state)
        assert out["text_coping_source"] is True

    # ──────────── 分支3：ctrl=v & text=None → cp=clamp(ctrl), src_flag=False（旧行为） ────────────

    def test_branch3_ctrl_only_cp_equals_ctrl(self) -> None:
        """分支3：ctrl=0.4, text=None → coping_potential_state≈0.4（旧行为零回归）。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(control_appraisal=0.4, text_coping_prior=None)
        out = agent(state)
        assert out["coping_potential_state"] == pytest.approx(0.4)

    def test_branch3_ctrl_only_negative(self) -> None:
        """分支3：ctrl=-0.7, text=None → coping_potential_state≈-0.7（旧行为零回归）。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(control_appraisal=-0.7, text_coping_prior=None)
        out = agent(state)
        assert out["coping_potential_state"] == pytest.approx(-0.7)

    def test_branch3_ctrl_clamped_above(self) -> None:
        """分支3：ctrl=2.0, text=None → clamp 到 1.0（旧行为等价）。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(control_appraisal=2.0, text_coping_prior=None)
        out = agent(state)
        assert out["coping_potential_state"] == pytest.approx(1.0)

    def test_branch3_ctrl_clamped_below(self) -> None:
        """分支3：ctrl=-2.0, text=None → clamp 到 -1.0（旧行为等价）。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(control_appraisal=-2.0, text_coping_prior=None)
        out = agent(state)
        assert out["coping_potential_state"] == pytest.approx(-1.0)

    def test_branch3_src_flag_false(self) -> None:
        """分支3：ctrl=v, text=None → text_coping_source=False（旧行为零回归）。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(control_appraisal=0.4, text_coping_prior=None)
        out = agent(state)
        assert out["text_coping_source"] is False

    # ──────────── 分支4：ctrl=c & text=t → MLE 精度加权 ────────────

    def test_branch4_both_present_precision_weighted(self) -> None:
        """分支4：ctrl=0.6, text=0.2, pi_t=0.08 → (0.6+0.08*0.2)/1.08 ≈ 0.570。

        计算：π_ctrl=1.0, π_t=0.08
          ctrl_c=clamp(0.6)=0.6, text_c=clamp(0.2)=0.2
          cp = clamp((1.0*0.6 + 0.08*0.2) / (1.0+0.08)) = clamp(0.616/1.08) ≈ clamp(0.5704) ≈ 0.5704
        """
        pi_t = 0.08
        ctrl_v = 0.6
        text_v = 0.2
        expected = (1.0 * ctrl_v + pi_t * text_v) / (1.0 + pi_t)
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(
            control_appraisal=ctrl_v,
            text_coping_prior=text_v,
            text_coping_precision=pi_t,
        )
        out = agent(state)
        assert out["coping_potential_state"] == pytest.approx(expected, rel=1e-5)

    def test_branch4_both_present_negative_ctrl(self) -> None:
        """分支4：ctrl=-0.8, text=0.2, pi_t=0.08 → 精度加权后拉向负端。"""
        pi_t = 0.08
        ctrl_v = -0.8
        text_v = 0.2
        expected = (1.0 * ctrl_v + pi_t * text_v) / (1.0 + pi_t)
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(
            control_appraisal=ctrl_v,
            text_coping_prior=text_v,
            text_coping_precision=pi_t,
        )
        out = agent(state)
        assert out["coping_potential_state"] == pytest.approx(expected, rel=1e-5)

    def test_branch4_pi_t_low_ctrl_dominates(self) -> None:
        """分支4 pi_t 极小时 ctrl 主导：ctrl=0.8, text=0.0, pi_t=0.01 → cp ≈ ctrl。"""
        pi_t = 0.01
        ctrl_v = 0.8
        text_v = 0.0
        expected = (1.0 * ctrl_v + pi_t * text_v) / (1.0 + pi_t)
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(
            control_appraisal=ctrl_v,
            text_coping_prior=text_v,
            text_coping_precision=pi_t,
        )
        out = agent(state)
        # 精度极低的 text 几乎不改变 ctrl 主导的结果
        assert out["coping_potential_state"] == pytest.approx(expected, rel=1e-5)
        # ctrl 主导：cp 应非常接近 ctrl_v
        assert abs(out["coping_potential_state"] - ctrl_v) < 0.01

    def test_branch4_both_present_clamped(self) -> None:
        """分支4：ctrl=0.9, text=0.9, pi_t=0.08 → 精度加权后 clamp 到 ≤1.0。"""
        pi_t = 0.08
        ctrl_v = 0.9
        text_v = 0.9
        raw = (1.0 * ctrl_v + pi_t * text_v) / (1.0 + pi_t)
        expected = min(raw, 1.0)  # clamp[-1,1]
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(
            control_appraisal=ctrl_v,
            text_coping_prior=text_v,
            text_coping_precision=pi_t,
        )
        out = agent(state)
        assert out["coping_potential_state"] == pytest.approx(expected, rel=1e-5)
        assert out["coping_potential_state"] <= 1.0

    def test_branch4_src_flag_true(self) -> None:
        """分支4：ctrl=c, text=t → text_coping_source=True（text 参与融合）。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(
            control_appraisal=0.6,
            text_coping_prior=0.2,
            text_coping_precision=0.08,
        )
        out = agent(state)
        assert out["text_coping_source"] is True

    # ──────────── enabled=False 门控关 → 不产 coping_potential_state，归零 text_coping_source ────

    def test_disabled_text_coping_source_is_false(self) -> None:
        """enabled=False → text_coping_source=False（WARN-2 归零，防调用方残留上轮 True）。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(
            control_appraisal=None, text_coping_prior=0.3, coping_potential_enabled=False
        )
        out = agent(state)
        assert out.get("text_coping_source") is False

    def test_disabled_does_not_produce_coping_state(self) -> None:
        """enabled=False → 返回 dict 不含 coping_potential_state（零回归）。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3(
            control_appraisal=None, text_coping_prior=0.3, coping_potential_enabled=False
        )
        out = agent(state)
        assert "coping_potential_state" not in out


# ─────────────────────────────────────────────────────────────────
# C. motivational_system 中间带哑火测试
# ─────────────────────────────────────────────────────────────────

# TEXT_COPING_MIDDLE_BAND = 0.15（函数体内局部常量，对应议会约束3）
# COPING_RAGE_THRESHOLD = 0.3
# COPING_FEAR_THRESHOLD = -0.3
_MIDDLE_BAND = 0.15
_RAGE_THRESHOLD = 0.3
_FEAR_THRESHOLD = -0.3


class TestMotivationalSystemMiddleBandFirewall:
    """text_coping_source=True 中间带 [-0.15,+0.15] 哑火：防文本低信度先验翻转 rage/fear。

    测试顺序关键：guard 在 COPING_RAGE_THRESHOLD 判断之前（R3）。
    """

    # ── 中间带哑火（text_coping_source=True） ──

    def test_text_source_middle_band_positive_returns_rage(self) -> None:
        """(-v,+a), text_coping_source=True, coping=0.1（中间带正侧）→ rage（哑火保守默认）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=0.1, text_coping_source=True)
        assert result == "rage"

    def test_text_source_middle_band_negative_returns_rage(self) -> None:
        """(-v,+a), text_coping_source=True, coping=-0.1（中间带负侧）→ rage（哑火，非 fear）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=-0.1, text_coping_source=True)
        assert result == "rage"

    def test_text_source_middle_band_zero_returns_rage(self) -> None:
        """(-v,+a), text_coping_source=True, coping=0.0（中间带零点）→ rage（哑火）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=0.0, text_coping_source=True)
        assert result == "rage"

    def test_text_source_exactly_at_band_boundary_returns_rage(self) -> None:
        """(-v,+a), text_coping_source=True, coping=0.15（边界等于 BAND）→ rage（哑火边界相等）。"""
        result = motivational_system(
            -0.6, 0.6, coping_potential=_MIDDLE_BAND, text_coping_source=True
        )
        assert result == "rage"

    def test_text_source_exactly_at_neg_band_boundary_returns_rage(self) -> None:
        """(-v,+a), text_coping_source=True, coping=-0.15 → rage（负边界等于 BAND，哑火）。"""
        result = motivational_system(
            -0.6, 0.6, coping_potential=-_MIDDLE_BAND, text_coping_source=True
        )
        assert result == "rage"

    # ── guard 在 RAGE_THRESHOLD 之前：coping>0.15 但 <=0.3 → 放行走正常逻辑 ──

    def test_text_source_above_middle_band_below_rage_threshold_passthrough(self) -> None:
        """(-v,+a), text_coping_source=True, coping=0.2（>0.15 但 <0.3 RAGE 阈值）
        → 放行正常逻辑，中间段 [−0.3,+0.3] 保守 rage（guard 在 RAGE_THRESHOLD 前、只拦 ≤0.15）。
        """
        # coping=0.2 超出中间带（>0.15），guard 不触发；但仍在 [−0.3,0.3] 中间段 → rage
        result = motivational_system(-0.6, 0.6, coping_potential=0.2, text_coping_source=True)
        assert result == "rage"

    def test_text_source_coping_just_above_band_is_not_firewall_rage(self) -> None:
        """coping=0.16（刚超 0.15 BAND 上界）→ 不被中间带哑火拦（走正常 coping 路径）。

        coping=0.16 仍在 [−0.3,+0.3] 中间段 → 正常路径返回 rage（中间段保守默认）。
        测试目的：验证 guard 用 abs(cp)<=0.15 严格边界、不扩大拦截范围。
        """
        result = motivational_system(-0.6, 0.6, coping_potential=0.16, text_coping_source=True)
        assert result == "rage"  # 正常路径中间段保守 rage

    # ── 极端段放行：text_coping_source=True，coping 超出 RAGE/FEAR 阈值 ──

    def test_text_source_above_rage_threshold_returns_rage(self) -> None:
        """(-v,+a), text_coping_source=True, coping=0.4（>0.3 RAGE 阈值）→ rage（极端段放行）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=0.4, text_coping_source=True)
        assert result == "rage"

    def test_text_source_below_fear_threshold_returns_fear(self) -> None:
        """(-v,+a), text_coping_source=True, coping=-0.5（<-0.3 FEAR 阈值）→ fear（极端段放行）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=-0.5, text_coping_source=True)
        assert result == "fear"

    def test_text_source_coping_at_fear_boundary_passthrough(self) -> None:
        """(-v,+a), text_coping_source=True, coping=-0.31（刚过 -0.3 FEAR 阈值）→ fear（放行）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=-0.31, text_coping_source=True)
        assert result == "fear"

    # ── text_coping_source=False（旧调用方）任意 coping → 逐字旧行为（零回归） ──

    def test_old_caller_text_source_false_middle_band_still_rage(self) -> None:
        """text_coping_source=False（默认），coping=0.0 → rage（旧行为零回归，不触发哑火）。"""
        result_new = motivational_system(-0.6, 0.6, coping_potential=0.0, text_coping_source=False)
        result_old = motivational_system(-0.6, 0.6, coping_potential=0.0)
        assert result_new == "rage"
        assert result_new == result_old

    def test_old_caller_text_source_false_low_coping_not_fear(self) -> None:
        """text_coping_source=False, coping=-0.1 → rage（旧行为：中间段保守 rage，不触发哑火）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=-0.1, text_coping_source=False)
        assert result == "rage"

    def test_old_caller_text_source_false_below_fear_threshold(self) -> None:
        """text_coping_source=False, coping=-0.5 → fear（旧行为：极端段 < FEAR_THRESHOLD）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=-0.5, text_coping_source=False)
        assert result == "fear"

    def test_old_caller_no_text_source_arg_any_coping_unchanged(self) -> None:
        """不传 text_coping_source 参数 → 等价 False → 旧路径逐字零回归（未改签名默认值）。"""
        for cp in [0.0, 0.1, -0.1, 0.5, -0.5]:
            result_explicit = motivational_system(
                -0.6, 0.6, coping_potential=cp, text_coping_source=False
            )
            result_default = motivational_system(-0.6, 0.6, coping_potential=cp)
            assert result_default == result_explicit, (
                f"text_coping_source 默认=False 时行为应与显式传 False 逐字一致，cp={cp}"
            )

    # ── 非 (-v,+a) 象限不受 text_coping_source 影响 ──

    def test_non_rage_fear_quadrant_unaffected_by_text_source(self) -> None:
        """text_coping_source=True 不影响其余三个象限（只影响 (-v,+a) 中间带哑火）。"""
        # (+v,+a) → seeking（不管 text_coping_source）
        assert (
            motivational_system(0.6, 0.6, coping_potential=0.1, text_coping_source=True)
            == "seeking"
        )
        # (+v,-a) → care
        assert (
            motivational_system(0.6, -0.6, coping_potential=-0.1, text_coping_source=True) == "care"
        )
        # (-v,-a) → panic_grief
        assert (
            motivational_system(-0.6, -0.6, coping_potential=0.1, text_coping_source=True)
            == "panic_grief"
        )
        # r < NEUTRAL_RADIUS → neutral
        assert (
            motivational_system(0.0, 0.0, coping_potential=0.1, text_coping_source=True)
            == "neutral"
        )


# ─────────────────────────────────────────────────────────────────
# D. 每轮归零 spy（R6·关键）
# ─────────────────────────────────────────────────────────────────


async def test_text_coping_prior_zeroed_each_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """每轮归零 spy（B3·R6 关键）：step() 在 base dict 中显式归零 text_coping_prior=None
    和 text_coping_source=False，防 LangGraph LastValue checkpoint 残留。

    spy 逻辑（仿 test_external_priors.py::test_external_priors_no_residual_across_turns）：
      - 第1轮：state_overrides={"text_coping_prior": 0.5} → base["text_coping_prior"]==0.5。
      - 第2轮：state_overrides=None（不注入）→ base["text_coping_prior"] 必须为 None
                且 base["text_coping_source"] 必须为 False（step() 的归零基准）。
    """
    from src.orchestration.runner import ConversationSession, SessionConfig

    cfg = SessionConfig(coping_potential_enabled=True, rng_seed=42, affect_readout="map")
    stim = Stimulus(name="test_zero", goal_congruence=0.2, intensity=0.5)
    session = ConversationSession(thread_id="t-text-coping-zero", config=cfg)

    captured: list[dict] = []
    orig_ainvoke = session.graph.ainvoke

    async def _spy(base: dict, *args: object, **kwargs: object) -> object:
        captured.append(dict(base))
        return await orig_ainvoke(base, *args, **kwargs)

    monkeypatch.setattr(session.graph, "ainvoke", _spy)

    # 第1轮：注入 text_coping_prior
    await session.step(stim, state_overrides={"text_coping_prior": 0.5})
    assert captured[-1]["text_coping_prior"] == pytest.approx(0.5), (
        "第1轮 state_overrides 的 text_coping_prior=0.5 应写入 ainvoke input"
    )

    # 第2轮：不注入 → base 必须显式归零
    await session.step(stim, state_overrides=None)
    assert captured[-1]["text_coping_prior"] is None, (
        "第2轮不传 state_overrides 时 text_coping_prior 必须显式归零为 None；"
        "非 None = LastValue checkpoint 残留，违反 B3 约束5"
    )
    assert captured[-1]["text_coping_source"] is False, (
        "第2轮 text_coping_source 必须显式归零为 False（防残留）"
    )


async def test_text_coping_source_zeroed_when_override_has_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """text_coping_source 即使在有 text_coping_prior 注入时也先被 base 归零，
    再被 state_overrides 覆盖（验证 base 归零不被 state_overrides 静默跳过）。
    """
    from src.orchestration.runner import ConversationSession, SessionConfig

    cfg = SessionConfig(coping_potential_enabled=True, rng_seed=7, affect_readout="map")
    stim = Stimulus(name="test_src", goal_congruence=0.0, intensity=0.3)
    session = ConversationSession(thread_id="t-text-src-zero", config=cfg)

    captured: list[dict] = []
    orig_ainvoke = session.graph.ainvoke

    async def _spy(base: dict, *args: object, **kwargs: object) -> object:
        captured.append(dict(base))
        return await orig_ainvoke(base, *args, **kwargs)

    monkeypatch.setattr(session.graph, "ainvoke", _spy)

    # 注入 text_coping_prior 但不显式传 text_coping_source → 应从 base 归零 False
    await session.step(stim, state_overrides={"text_coping_prior": 0.3})
    # text_coping_source 应为 False（base 归零基准；AppraisalAgent 才写 True 进 state，
    # 但 ainvoke input 这个 base 只是初始输入，不是节点输出）
    assert captured[-1]["text_coping_source"] is False, (
        "text_coping_source 在 ainvoke input 的 base 里应始终为 False（图外归零基准）"
    )
    assert captured[-1]["text_coping_prior"] == pytest.approx(0.3)


# ─────────────────────────────────────────────────────────────────
# E. CS 白名单静态断言（R4）
# ─────────────────────────────────────────────────────────────────


class TestAppraisalB3SourceWhitelist:
    """R4 约束：B3 区块不得使用 fuse_terms/occ_prior/fast_survival_prior/hierarchical_fuse。

    text_coping_prior 的唯一合法消费者是 AppraisalAgent 的 B3 区块，
    且该区块不得反向依赖 fuse/occ 融合函数（守来源正交红线）。
    """

    def _get_b3_executable_source(self) -> str:
        """提取 AppraisalAgent.__call__ B3 区块的可执行源码行（去掉注释行）。

        `coping_updates: dict = {}` 是实际代码入口，注释在其之前。
        只检测非注释行，避免注释中出现白名单词（如"不复用 fuse_terms / hierarchical_fuse"）
        误触发断言。
        """
        source = inspect.getsource(AppraisalAgent.__call__)
        # 从 coping_updates: dict 开始（B3 实际执行代码入口，注释区在此之前）
        idx = source.find("coping_updates: dict = {}")
        assert idx >= 0, "源码应含 'coping_updates: dict = {}' 作为 B3 代码入口"
        code_block = source[idx:]
        # 过滤注释行（以 # 开头的行，去掉前导空白后），只留非注释代码行
        non_comment_lines = [
            line
            for line in code_block.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return "\n".join(non_comment_lines)

    def test_b3_block_not_contain_fuse_terms(self) -> None:
        """B3 可执行代码不含 'fuse_terms'（来源正交：text_coping_prior 不入 fuse_terms）。"""
        block = self._get_b3_executable_source()
        assert "fuse_terms" not in block, (
            "B3 可执行代码不得调用 fuse_terms（text_coping_prior 白名单·来源正交红线）"
        )

    def test_b3_block_not_contain_occ_prior(self) -> None:
        """B3 可执行代码不含 'occ_prior'（text_coping_prior 不进 OCC VA 路径）。"""
        block = self._get_b3_executable_source()
        assert "occ_prior" not in block, (
            "B3 融合区块可执行代码不得调用 occ_prior（来源须正交于 OCC VA 路径）"
        )

    def test_b3_block_not_contain_fast_survival_prior(self) -> None:
        """B3 可执行代码不含 'fast_survival_prior'（text_coping_prior 不进生存先验路径）。"""
        block = self._get_b3_executable_source()
        assert "fast_survival_prior" not in block, (
            "B3 融合区块可执行代码不得调用 fast_survival_prior（来源正交·守白名单）"
        )

    def test_b3_block_not_contain_hierarchical_fuse(self) -> None:
        """B3 可执行代码不含 'hierarchical_fuse'（text_coping_prior 不进层级融合）。"""
        block = self._get_b3_executable_source()
        assert "hierarchical_fuse" not in block, (
            "B3 融合区块可执行代码不得调用 hierarchical_fuse（来源正交·守白名单）"
        )

    def test_b3_block_not_contain_goal_congruence(self) -> None:
        """B3 可执行代码不含 'goal_congruence'（T2 裁决：coping 来源正交于 OCC VA 路径）。"""
        block = self._get_b3_executable_source()
        assert "goal_congruence" not in block, (
            "B3 融合区块可执行代码不得引用 goal_congruence（T2 裁决：来源须正交于 OCC VA 路径）"
        )


# ─────────────────────────────────────────────────────────────────
# F. 字段存在性
# ─────────────────────────────────────────────────────────────────


class TestFieldExistence:
    """AffectState / SessionConfig / Stimulus 字段存在性断言。"""

    def test_affect_state_has_text_coping_prior(self) -> None:
        """AffectState.model_fields 含 'text_coping_prior'。"""
        assert "text_coping_prior" in AffectState.model_fields

    def test_affect_state_has_text_coping_source(self) -> None:
        """AffectState.model_fields 含 'text_coping_source'。"""
        assert "text_coping_source" in AffectState.model_fields

    def test_affect_state_has_text_coping_precision(self) -> None:
        """AffectState.model_fields 含 'text_coping_precision'。"""
        assert "text_coping_precision" in AffectState.model_fields

    def test_affect_state_text_coping_prior_default_none(self) -> None:
        """AffectState() 默认 text_coping_prior=None（每轮归零的 None 基准）。"""
        state = AffectState()
        assert state.text_coping_prior is None

    def test_affect_state_text_coping_source_default_false(self) -> None:
        """AffectState() 默认 text_coping_source=False。"""
        state = AffectState()
        assert state.text_coping_source is False

    def test_affect_state_text_coping_precision_default_008(self) -> None:
        """AffectState() 默认 text_coping_precision=0.08。"""
        state = AffectState()
        assert state.text_coping_precision == pytest.approx(0.08)

    def test_session_config_has_text_coping_enabled(self) -> None:
        """SessionConfig.model_fields 含 'text_coping_enabled'。"""
        assert "text_coping_enabled" in SessionConfig.model_fields

    def test_session_config_has_text_coping_precision(self) -> None:
        """SessionConfig.model_fields 含 'text_coping_precision'。"""
        assert "text_coping_precision" in SessionConfig.model_fields

    def test_session_config_text_coping_enabled_default_false(self) -> None:
        """SessionConfig() 默认 text_coping_enabled=False（零回归门关）。"""
        cfg = SessionConfig()
        assert cfg.text_coping_enabled is False

    def test_session_config_text_coping_precision_default_008(self) -> None:
        """SessionConfig() 默认 text_coping_precision=0.08。"""
        cfg = SessionConfig()
        assert cfg.text_coping_precision == pytest.approx(0.08)

    def test_stimulus_control_appraisal_default_none(self) -> None:
        """Stimulus 默认 control_appraisal=None（B3 前置：absent cue，精度趋零）。"""
        stim = Stimulus(name="test")
        assert stim.control_appraisal is None

    def test_stimulus_control_appraisal_field_in_model_dump(self) -> None:
        """Stimulus.model_dump() 含 'control_appraisal' 键（序列化不丢失）。"""
        stim = Stimulus(name="test")
        assert "control_appraisal" in stim.model_dump()

    def test_stimulus_control_appraisal_none_in_model_dump(self) -> None:
        """Stimulus() 默认时 model_dump()['control_appraisal'] == None。"""
        stim = Stimulus(name="test")
        assert stim.model_dump()["control_appraisal"] is None

    def test_stimulus_explicit_float_roundtrips(self) -> None:
        """Stimulus(control_appraisal=0.6) → model_dump 含正确值（float 仍可传入）。"""
        stim = Stimulus(name="test", control_appraisal=0.6)
        assert stim.model_dump()["control_appraisal"] == pytest.approx(0.6)

    def test_session_config_to_state_flags_contains_text_coping_enabled(self) -> None:
        """to_state_flags() 含 'text_coping_enabled' 键。"""
        flags = SessionConfig().to_state_flags()
        assert "text_coping_enabled" in flags

    def test_session_config_to_state_flags_contains_text_coping_precision(self) -> None:
        """to_state_flags() 含 'text_coping_precision' 键。"""
        flags = SessionConfig().to_state_flags()
        assert "text_coping_precision" in flags

    def test_session_config_to_state_flags_text_coping_enabled_default_false(self) -> None:
        """to_state_flags()['text_coping_enabled'] 默认为 False。"""
        flags = SessionConfig().to_state_flags()
        assert flags["text_coping_enabled"] is False

    def test_session_config_to_state_flags_text_coping_precision_default(self) -> None:
        """to_state_flags()['text_coping_precision'] 默认为 0.08。"""
        flags = SessionConfig().to_state_flags()
        assert flags["text_coping_precision"] == pytest.approx(0.08)

    def test_affect_state_has_text_coping_enabled(self) -> None:
        """AffectState.model_fields 含 'text_coping_enabled'（BLOCK-1 落点）。"""
        assert "text_coping_enabled" in AffectState.model_fields

    def test_affect_state_text_coping_enabled_default_false(self) -> None:
        """AffectState() 默认 text_coping_enabled=False（零回归门关）。"""
        state = AffectState()
        assert state.text_coping_enabled is False


# ─────────────────────────────────────────────────────────────────
# G. SessionConfig 精度上限 fail-fast
# ─────────────────────────────────────────────────────────────────


class TestSessionConfigPrecisionFailFast:
    """SessionConfig.text_coping_precision 精度上限 fail-fast（Field le=0.10）。

    议会：π_t≤0.10（.env 缺省 0.08）；超出 fail-fast 由 SessionConfig 层承担
    （AffectState 层不加 le 防 checkpoint 反序列化 fail）。
    """

    def test_precision_above_limit_raises(self) -> None:
        """SessionConfig(text_coping_precision=0.2)（>0.10）应 raise ValidationError。"""
        with pytest.raises(ValidationError):
            SessionConfig(text_coping_precision=0.2)

    def test_precision_exactly_at_limit_passes(self) -> None:
        """SessionConfig(text_coping_precision=0.10)（边界等于上限）应通过。"""
        cfg = SessionConfig(text_coping_precision=0.10)
        assert cfg.text_coping_precision == pytest.approx(0.10)

    def test_precision_just_above_limit_raises(self) -> None:
        """SessionConfig(text_coping_precision=0.101)（刚超上限）应 raise ValidationError。"""
        with pytest.raises(ValidationError):
            SessionConfig(text_coping_precision=0.101)

    def test_precision_zero_or_negative_raises(self) -> None:
        """SessionConfig(text_coping_precision=0.0)（gt=0 违反）应 raise ValidationError。"""
        with pytest.raises(ValidationError):
            SessionConfig(text_coping_precision=0.0)

    def test_precision_negative_raises(self) -> None:
        """SessionConfig(text_coping_precision=-0.01)（<0）应 raise ValidationError。"""
        with pytest.raises(ValidationError):
            SessionConfig(text_coping_precision=-0.01)

    def test_default_precision_is_valid(self) -> None:
        """SessionConfig() 默认 text_coping_precision=0.08（在 (0,0.10] 范围内，正常构造）。"""
        cfg = SessionConfig()
        assert 0 < cfg.text_coping_precision <= 0.10

    def test_small_valid_precision_passes(self) -> None:
        """SessionConfig(text_coping_precision=0.01)（小正值，在范围内）应通过。"""
        cfg = SessionConfig(text_coping_precision=0.01)
        assert cfg.text_coping_precision == pytest.approx(0.01)

    def test_affect_state_precision_field_accepts_above_limit(self) -> None:
        """AffectState 层 text_coping_precision 字段不加 le 约束——接受 >0.10 的值不报错。

        这是设计意图：AffectState.text_coping_precision 无 le 约束，防 checkpoint
        反序列化 fail（只在 SessionConfig 层 fail-fast，不在 state 层双重校验）。
        """
        # 构造 AffectState 传入 >0.10 的值不应报错
        state = AffectState(text_coping_precision=0.5)
        assert state.text_coping_precision == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────
# H. BLOCK-1 门控开/关行为测试
# ─────────────────────────────────────────────────────────────────


def _make_state_b3_with_gate(
    *,
    control_appraisal: float | None,
    text_coping_prior: float | None,
    text_coping_enabled: bool,
    text_coping_precision: float = 0.08,
) -> AffectState:
    """门控开/关测试 helper：同时设 text_coping_enabled + text_coping_prior。"""
    stim = Stimulus(
        name="gate_test", goal_congruence=0.0, intensity=0.5, control_appraisal=control_appraisal
    )
    return AffectState(
        stimulus=stim,
        coping_potential_enabled=True,
        text_coping_enabled=text_coping_enabled,
        text_coping_prior=text_coping_prior,
        text_coping_precision=text_coping_precision,
    )


class TestTextCopingGate:
    """BLOCK-1：text_coping_enabled 门控语义测试。

    门控关（False，默认）时：text_coping_prior 被强制视为 None，
    只走分支1/3（纯 ctrl 路径），coping_potential_state 不受 text 影响。
    门控开（True）时：text_coping_prior=0.5 正常进分支2/4。
    """

    # ── 门控关（默认）── text_coping_prior 有值但被忽略

    def test_gate_off_text_ignored_ctrl_none_gives_zero(self) -> None:
        """gate=False, ctrl=None, text=0.5 → 分支1（text 被忽略）→ cp=0.0, src=False。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3_with_gate(
            control_appraisal=None, text_coping_prior=0.5, text_coping_enabled=False
        )
        out = agent(state)
        # text 被门控丢弃 → 两皆 None → 分支1 → cp=0.0
        assert out["coping_potential_state"] == pytest.approx(0.0)
        assert out["text_coping_source"] is False

    def test_gate_off_text_ignored_ctrl_present_gives_ctrl(self) -> None:
        """gate=False, ctrl=0.4, text=0.5 → 分支3（text 被忽略）→ cp=0.4, src=False。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3_with_gate(
            control_appraisal=0.4, text_coping_prior=0.5, text_coping_enabled=False
        )
        out = agent(state)
        # text 被门控丢弃 → ctrl=0.4, text=None → 分支3
        assert out["coping_potential_state"] == pytest.approx(0.4)
        assert out["text_coping_source"] is False

    def test_gate_off_coping_state_unaffected_by_text(self) -> None:
        """gate=False 时 text_coping_prior=0.9 不影响 coping_potential_state（零回归）。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        # ctrl=None, text=0.9, gate=False → 与 text=None 行为完全一致（cp=0.0）
        state_with_text = _make_state_b3_with_gate(
            control_appraisal=None, text_coping_prior=0.9, text_coping_enabled=False
        )
        state_no_text = _make_state_b3_with_gate(
            control_appraisal=None, text_coping_prior=None, text_coping_enabled=False
        )
        out_with = agent(state_with_text)
        out_none = agent(state_no_text)
        assert out_with["coping_potential_state"] == pytest.approx(
            out_none["coping_potential_state"]
        ), "gate=False 时 text_coping_prior 有值与 None 结果应完全一致"

    # ── 门控开 ── text_coping_prior 正常参与分支2/4

    def test_gate_on_text_only_enters_branch2(self) -> None:
        """gate=True, ctrl=None, text=0.5 → 分支2 → cp=0.5, src=True。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state_b3_with_gate(
            control_appraisal=None, text_coping_prior=0.5, text_coping_enabled=True
        )
        out = agent(state)
        assert out["coping_potential_state"] == pytest.approx(0.5)
        assert out["text_coping_source"] is True

    def test_gate_on_both_present_enters_branch4(self) -> None:
        """gate=True, ctrl=0.6, text=0.5, pi_t=0.08 → 分支4 精度加权, src=True。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        pi_t = 0.08
        ctrl_v = 0.6
        text_v = 0.5
        expected = (1.0 * ctrl_v + pi_t * text_v) / (1.0 + pi_t)
        state = _make_state_b3_with_gate(
            control_appraisal=ctrl_v,
            text_coping_prior=text_v,
            text_coping_enabled=True,
            text_coping_precision=pi_t,
        )
        out = agent(state)
        assert out["coping_potential_state"] == pytest.approx(expected, rel=1e-5)
        assert out["text_coping_source"] is True
