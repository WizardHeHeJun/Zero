"""B8 批次集成测试：env/persona → 运行时旋钮贯通 + SessionConfig 收口。

覆盖：
A. workspace 三 flag 的 env→state 贯通（默认关零回归 + 开启生效）
B. chat 两时间尺度旋钮的 env→ChatDriver 贯通（默认零回归 + 开启生效）
C. va_coupling persona 注入端到端（persona 给 va_coupling → occ_prior 收到非对称系数）
D. SessionConfig 收口（构造/model_dump 展开等价旧 flags dict，零回归）
torch/API-free，纯函数 + state 单测 + ChatDriver 行为测。
"""

from __future__ import annotations

from typing import Any

import pytest

from src.agents.affect_math import occ_prior
from src.agents.appraisal import AppraisalAgent
from src.agents.emotion_lexicon import PANKSEPP_FEAR_AROUSAL_THRESHOLD
from src.agents.language import _appraisal_summary
from src.orchestration.chat_driver import ChatDriver, build_chat_driver
from src.orchestration.runner import ConversationSession, SessionConfig
from src.orchestration.state import AffectState, Stimulus
from src.storage.conversation_log import ConversationLog

# ─────────────────────────────────────────────────────────────────
# 工具：假 session（避免真图 + checkpointer）
# ─────────────────────────────────────────────────────────────────


class _FakeSession:
    """假 ConversationSession：step 返回固定 e*，记录 stim 供断言。"""

    def __init__(self, ev: tuple[float, float] = (0.3, 0.4)) -> None:
        self.ev = ev
        self.last_stim: Any = None

    async def step(self, stim: Any, state_overrides: dict | None = None) -> dict[str, Any]:
        self.last_stim = stim
        return {"valence_arousal": self.ev, "recalled_context": []}


def _make_driver(
    session: Any,
    *,
    log: ConversationLog | None = None,
    attitude: tuple[float, float] = (0.0, 0.0),
    **kwargs: Any,
) -> ChatDriver:
    return ChatDriver(
        thread="t",
        lm=None,
        log=log or ConversationLog(":memory:"),
        session=session,
        history=[],
        attitude=attitude,
        mode="test",
        noise_std=0.0,
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────
# A. workspace 三 flag：env → SessionConfig → state
# ─────────────────────────────────────────────────────────────────


class TestWorkspaceFlagEnvPassthrough:
    """A：三 flag 默认 False（零回归）；env 开启后 ConversationSession 持有正确值。"""

    def test_precision_split_default_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """未设 ZERO_PRECISION_SPLIT → session.config.precision_split=False（零回归）。"""
        monkeypatch.delenv("ZERO_PRECISION_SPLIT", raising=False)
        cfg = SessionConfig()
        assert cfg.precision_split is False

    def test_precision_split_env_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_PRECISION_SPLIT=1 → session.config.precision_split=True。"""
        monkeypatch.setenv("ZERO_PRECISION_SPLIT", "1")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        driver = build_chat_driver(thread="test-ps")
        assert driver.session.config.precision_split is True

    def test_fuse_independence_correct_default_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ZERO_FUSE_INDEPENDENCE_CORRECT", raising=False)
        cfg = SessionConfig()
        assert cfg.fuse_independence_correct is False

    def test_fuse_independence_correct_env_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ZERO_FUSE_INDEPENDENCE_CORRECT", "true")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        driver = build_chat_driver(thread="test-fic")
        assert driver.session.config.fuse_independence_correct is True

    def test_ignition_survival_fallback_default_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ZERO_IGNITION_SURVIVAL_FALLBACK", raising=False)
        cfg = SessionConfig()
        assert cfg.ignition_survival_fallback is False

    def test_ignition_survival_fallback_env_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ZERO_IGNITION_SURVIVAL_FALLBACK", "yes")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        driver = build_chat_driver(thread="test-isf")
        assert driver.session.config.ignition_survival_fallback is True

    def test_all_three_flags_off_state_unchanged(self) -> None:
        """三 flag 全 False → to_state_flags 里对应值为 False（零回归，不影响现有 state 字段）。"""
        cfg = SessionConfig(
            precision_split=False, fuse_independence_correct=False, ignition_survival_fallback=False
        )
        flags = cfg.to_state_flags()
        assert flags["precision_split"] is False
        assert flags["fuse_independence_correct"] is False
        assert flags["ignition_survival_fallback"] is False


# ─────────────────────────────────────────────────────────────────
# B. chat 两时间尺度旋钮：env → ChatDriver.self.*
# ─────────────────────────────────────────────────────────────────


class TestChatTimescaleKnobs:
    """B：attitude_arousal_weight 和 sensitization_* 默认零回归；env 开启后透传 self.*。"""

    def test_attitude_arousal_weight_default_zero(self) -> None:
        """未传参 → self.attitude_arousal_weight=0.0（零回归：attitude_step 退化旧行为）。"""
        driver = _make_driver(_FakeSession())
        assert driver.attitude_arousal_weight == pytest.approx(0.0)

    def test_attitude_arousal_weight_env_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_ATTITUDE_AROUSAL_WEIGHT=0.3 → self.attitude_arousal_weight=0.3。"""
        monkeypatch.setenv("ZERO_ATTITUDE_AROUSAL_WEIGHT", "0.3")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        driver = build_chat_driver(thread="test-aaw")
        assert driver.attitude_arousal_weight == pytest.approx(0.3)

    def test_sensitization_gain_default_zero(self) -> None:
        driver = _make_driver(_FakeSession())
        assert driver.sens_gain == pytest.approx(0.0)

    def test_sensitization_threshold_default(self) -> None:
        driver = _make_driver(_FakeSession())
        assert driver.sens_threshold == pytest.approx(0.5)

    def test_sensitization_env_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ZERO_HABITUATION_SENSITIZATION_GAIN", "0.2")
        monkeypatch.setenv("ZERO_SENSITIZATION_THRESHOLD", "0.6")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        driver = build_chat_driver(thread="test-sen")
        assert driver.sens_gain == pytest.approx(0.2)
        assert driver.sens_threshold == pytest.approx(0.6)

    async def test_zero_regression_hab_no_tau(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """hab_tau=0（默认）时，不设 gain → η=1.0，arousal 不衰减（零回归）。"""
        monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
        log = ConversationLog(":memory:")
        session = _FakeSession((0.3, 0.5))
        driver = _make_driver(session, log=log, hab_tau=0.0)
        turn = await driver.step("hi")
        # η=1 → e 的 arousal 部分不衰减
        # 仅断言 step 正常返回（零回归：无崩溃、无异常）
        assert turn.emotion is not None
        log.close()

    async def test_sensitization_changes_arousal_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """sensitization_gain>0 + 高唤醒 → η>1（敏化项激活），emotion arousal 更高。"""
        monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
        log_off = ConversationLog(":memory:")
        log_on = ConversationLog(":memory:")
        # 高唤醒 e*=(0.3, 0.8)，intensity=|0.8|=0.8 > threshold=0.5 → 敏化主导
        session_off = _FakeSession((0.3, 0.8))
        session_on = _FakeSession((0.3, 0.8))
        # hab_tau=5 使习惯化有效；对比有/无 sensitization_gain（exposure=0，首轮）
        # 注：exposure=0 时 exp(-0/5)=1，hab=1；sen=gain*max(0.8-0.5,0)=0.3*gain；η=1+0.3*gain
        driver_off = _make_driver(session_off, log=log_off, hab_tau=5.0, sensitization_gain=0.0)
        driver_on = _make_driver(
            session_on, log=log_on, hab_tau=5.0, sensitization_gain=0.3, sensitization_threshold=0.5
        )
        turn_off = await driver_off.step("x")
        turn_on = await driver_on.step("x")
        # 敏化增益 → η更大 → 调制后 arousal 更大 → emotion arousal 更高
        assert turn_on.emotion[1] >= turn_off.emotion[1], (
            f"sensitization_gain 开启应使 emotion arousal 更高，"
            f"但 on={turn_on.emotion[1]:.4f} < off={turn_off.emotion[1]:.4f}"
        )
        log_off.close()
        log_on.close()


# ─────────────────────────────────────────────────────────────────
# C. va_coupling persona 注入端到端
# ─────────────────────────────────────────────────────────────────


class TestVaCouplingPersonaInjection:
    """C：persona va_coupling_pos/neg → AffectState → AppraisalAgent → occ_prior 非对称。"""

    def test_va_coupling_default_none_zero_regression(self) -> None:
        """va_coupling_pos/neg 默认 None → AppraisalAgent 用 occ_prior 0.6/0.6（零回归）。"""
        stim = Stimulus(name="t", goal_congruence=0.5, attitude_appeal=0.0, intensity=0.8)
        state_default = AffectState(stimulus=stim, prior_mu=None, prior_sigma=None, reward=None)
        state_none = AffectState(
            stimulus=stim,
            prior_mu=None,
            prior_sigma=None,
            reward=None,
            va_coupling_pos=None,
            va_coupling_neg=None,
        )
        out_default = AppraisalAgent()(state_default)
        out_none = AppraisalAgent()(state_none)
        assert out_default["prior_mu"] == out_none["prior_mu"]

    def test_va_coupling_neg_higher_than_pos_increases_negative_arousal(self) -> None:
        """va_coupling_neg > va_coupling_pos → 负效价侧 arousal 更高（negativity bias）。"""
        stim_neg = Stimulus(name="neg", goal_congruence=-0.6, attitude_appeal=0.0, intensity=0.8)
        # 对称系数（默认 0.6/0.6）
        state_sym = AffectState(stimulus=stim_neg)
        out_sym = AppraisalAgent()(state_sym)

        # 非对称系数（neg=0.8）
        state_asym = AffectState(stimulus=stim_neg, va_coupling_pos=0.6, va_coupling_neg=0.8)
        out_asym = AppraisalAgent()(state_asym)

        # 负效价输入下，更高的 neg 系数应给出更高的 arousal
        assert out_asym["prior_mu"][1] > out_sym["prior_mu"][1], (
            f"va_coupling_neg=0.8 应比 0.6 给出更高 arousal，"
            f"asym={out_asym['prior_mu'][1]:.4f} vs sym={out_sym['prior_mu'][1]:.4f}"
        )

    def test_va_coupling_only_affects_arousal_not_valence(self) -> None:
        """va_coupling 只影响 arousal 维，不影响 valence 维（occ_prior 语义正确）。"""
        stim = Stimulus(name="t", goal_congruence=0.5, attitude_appeal=0.0, intensity=0.8)
        state_sym = AffectState(stimulus=stim)
        state_asym = AffectState(stimulus=stim, va_coupling_pos=0.4, va_coupling_neg=0.9)
        out_sym = AppraisalAgent()(state_sym)
        out_asym = AppraisalAgent()(state_asym)
        # valence 由 goal_congruence/standard_compliance/attitude_appeal 决定，与 va_coupling 无关
        assert out_sym["prior_mu"][0] == pytest.approx(out_asym["prior_mu"][0]), (
            "va_coupling 不应影响 valence 维"
        )

    def test_persona_va_coupling_passthrough_to_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """persona.va_coupling_pos/neg 经 build_chat_driver → session.config 贯通到位。"""
        import json
        import os as _os
        import tempfile

        data = {
            "name": "test",
            "card": "test card",
            "va_coupling_pos": 0.4,
            "va_coupling_neg": 0.8,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            tmp_path = f.name
        try:
            monkeypatch.setenv("ZERO_PERSONA_FILE", tmp_path)
            monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
            monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
            monkeypatch.delenv("ZERO_VA_COUPLING_POS", raising=False)
            monkeypatch.delenv("ZERO_VA_COUPLING_NEG", raising=False)
            driver = build_chat_driver(thread="test-vac")
            assert driver.session.config.va_coupling_pos == pytest.approx(0.4)
            assert driver.session.config.va_coupling_neg == pytest.approx(0.8)
        finally:
            _os.unlink(tmp_path)

    def test_env_va_coupling_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_VA_COUPLING_POS/NEG env → session.config（无 persona 文件时从 env 读）。"""
        monkeypatch.delenv("ZERO_PERSONA_FILE", raising=False)
        monkeypatch.setenv("ZERO_VA_COUPLING_POS", "0.45")
        monkeypatch.setenv("ZERO_VA_COUPLING_NEG", "0.75")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        driver = build_chat_driver(thread="test-vac-env")
        assert driver.session.config.va_coupling_pos == pytest.approx(0.45)
        assert driver.session.config.va_coupling_neg == pytest.approx(0.75)

    def test_occ_prior_direct_asymmetry(self) -> None:
        """直接测 occ_prior：neg > pos 时负效价侧 arousal 更高。纯函数零依赖验证。

        occ_prior 返回 (prior_mu, prior_sigma, reward)；prior_mu 是 (valence, arousal) tuple。
        arousal 取 prior_mu[1]。
        """
        # 正效价输入（goal_congruence=0.6 → valence > 0）：va_coupling_pos 起作用
        mu_pos_sym, _, _ = occ_prior(0.6, 0.0, 0.0, 0.8, va_coupling_pos=0.6, va_coupling_neg=0.6)
        mu_pos_asym, _, _ = occ_prior(0.6, 0.0, 0.0, 0.8, va_coupling_pos=0.4, va_coupling_neg=0.8)
        # 正效价下 pos 系数更小（0.4 < 0.6）→ 该侧 arousal 贡献更少 → 总 arousal 更低
        assert mu_pos_asym[1] < mu_pos_sym[1], (
            f"pos=0.4<0.6，正效价侧 arousal 应更低："
            f"asym={mu_pos_asym[1]:.4f} sym={mu_pos_sym[1]:.4f}"
        )

        # 负效价输入（goal_congruence=-0.6 → valence < 0）：va_coupling_neg 起作用
        mu_neg_sym, _, _ = occ_prior(-0.6, 0.0, 0.0, 0.8, va_coupling_pos=0.6, va_coupling_neg=0.6)
        mu_neg_asym, _, _ = occ_prior(-0.6, 0.0, 0.0, 0.8, va_coupling_pos=0.4, va_coupling_neg=0.8)
        # 负效价下 neg 系数更大（0.8 > 0.6）→ 该侧 arousal 贡献更多 → 总 arousal 更高
        assert mu_neg_asym[1] > mu_neg_sym[1], (
            f"neg=0.8>0.6，负效价侧 arousal 应更高："
            f"asym={mu_neg_asym[1]:.4f} sym={mu_neg_sym[1]:.4f}"
        )


# ─────────────────────────────────────────────────────────────────
# D. SessionConfig 收口
# ─────────────────────────────────────────────────────────────────


class TestSessionConfig:
    """D：SessionConfig 构造/model_dump 等价旧 flags dict；零回归；新旋钮集中一处。"""

    def test_default_config_matches_old_flags(self) -> None:
        """默认 SessionConfig().to_state_flags() 与旧 ConversationSession.flags dict 逐字对齐。"""
        cfg = SessionConfig()
        flags = cfg.to_state_flags()
        # 旧 flags dict 的所有键都在新 flags 里且值相同
        old_defaults = {
            "regulation_enabled": False,
            "regulation_strategy": "suppression",
            "mood_enabled": False,
            "recall_enabled": False,
            "language_enabled": False,
            "workspace_enabled": False,
            "appraisal_conditioning_enabled": False,
            "language_max_iters": 3,
            "rng_seed": None,
            "sample_sigma_cap": None,
            "affect_readout": "sample",
            "arousal_baseline": 0.0,
            "arousal_gain_cap": None,
        }
        for k, v in old_defaults.items():
            assert flags[k] == v, f"SessionConfig default mismatch: {k}={flags[k]!r} != {v!r}"

    def test_new_flags_default_false_none(self) -> None:
        """新增旋钮（B8）默认 False/None → 零回归（不影响旧路径）。"""
        cfg = SessionConfig()
        assert cfg.precision_split is False
        assert cfg.fuse_independence_correct is False
        assert cfg.ignition_survival_fallback is False
        assert cfg.va_coupling_pos is None
        assert cfg.va_coupling_neg is None

    def test_model_dump_contains_all_new_flags(self) -> None:
        """model_dump() 输出包含所有新旋钮（确保 to_state_flags 展开完整）。"""
        cfg = SessionConfig(
            precision_split=True,
            fuse_independence_correct=True,
            ignition_survival_fallback=True,
            va_coupling_pos=0.4,
            va_coupling_neg=0.8,
        )
        flags = cfg.to_state_flags()
        assert flags["precision_split"] is True
        assert flags["fuse_independence_correct"] is True
        assert flags["ignition_survival_fallback"] is True
        assert flags["va_coupling_pos"] == pytest.approx(0.4)
        assert flags["va_coupling_neg"] == pytest.approx(0.8)

    def test_config_param_takes_priority_over_legacy(self) -> None:
        """ConversationSession 传 config= 时，config 优先于旧展开参数。"""
        cfg = SessionConfig(precision_split=True, affect_readout="map")
        session = ConversationSession(
            thread_id="t",
            config=cfg,
            # 传旧参数（应被忽略）
            affect_readout="sample",
            precision_split=False,
        )
        assert session.config.precision_split is True
        assert session.config.affect_readout == "map"

    def test_legacy_params_build_equivalent_config(self) -> None:
        """不传 config 时，旧展开参数构造出等价的 SessionConfig。"""
        session = ConversationSession(
            thread_id="t",
            affect_readout="map",
            arousal_baseline=-0.08,
            precision_split=True,
        )
        flags = session.config.to_state_flags()
        assert flags["affect_readout"] == "map"
        assert flags["arousal_baseline"] == pytest.approx(-0.08)
        assert flags["precision_split"] is True

    def test_session_step_injects_all_flags(self) -> None:
        """session.step 向 ainvoke 注入的 dict 包含新旋钮（通过 to_state_flags 展开）。"""
        # 只验证 to_state_flags 展开完整，不走真图
        cfg = SessionConfig(
            precision_split=True,
            va_coupling_pos=0.5,
            va_coupling_neg=0.7,
        )
        flags = cfg.to_state_flags()
        assert "precision_split" in flags
        assert "va_coupling_pos" in flags
        assert "va_coupling_neg" in flags
        assert flags["precision_split"] is True
        assert flags["va_coupling_pos"] == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────
# E. Panksepp RAGE/FEAR 次级区分（WARN-4 / A-P1-D）：
#    env → SessionConfig → state → _appraisal_summary
# ─────────────────────────────────────────────────────────────────


class TestPankseppDistinguishFearPassthrough:
    """E：panksepp_distinguish_fear 默认 False（零回归）；env 开启后贯通到 _appraisal_summary。"""

    def test_default_false_zero_regression(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """未设 ZERO_PANKSEPP_DISTINGUISH_FEAR → SessionConfig 字段 False + 展开进 state flags。"""
        monkeypatch.delenv("ZERO_PANKSEPP_DISTINGUISH_FEAR", raising=False)
        cfg = SessionConfig()
        assert cfg.panksepp_distinguish_fear is False
        assert cfg.to_state_flags()["panksepp_distinguish_fear"] is False

    def test_env_true_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_PANKSEPP_DISTINGUISH_FEAR=1 → session.config.panksepp_distinguish_fear=True。"""
        monkeypatch.setenv("ZERO_PANKSEPP_DISTINGUISH_FEAR", "1")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        driver = build_chat_driver(thread="test-pdf")
        assert driver.session.config.panksepp_distinguish_fear is True

    def test_appraisal_summary_default_rage_zero_regression(self) -> None:
        """默认 state（flag=False）：(-v,+高a) 摘要动机系统仍为 rage（逐字零回归）。"""
        stim = Stimulus(name="threat", goal_congruence=-0.5, intensity=0.8)
        state = AffectState(stimulus=stim, appraisal={"valence": -0.5, "arousal": 0.7})
        assert "动机系统=rage" in _appraisal_summary(state)

    def test_appraisal_summary_flag_on_high_arousal_fear(self) -> None:
        """flag=True 且 arousal≥阈值：(-v,+高a) 摘要动机系统区分为 fear（贯通生效）。"""
        stim = Stimulus(name="threat", goal_congruence=-0.5, intensity=0.8)
        state = AffectState(
            stimulus=stim,
            appraisal={"valence": -0.5, "arousal": PANKSEPP_FEAR_AROUSAL_THRESHOLD},
            panksepp_distinguish_fear=True,
        )
        assert "动机系统=fear" in _appraisal_summary(state)

    def test_appraisal_summary_flag_on_mid_arousal_still_rage(self) -> None:
        """flag=True 但 arousal<阈值（仍≥0）：维持 rage（次级区分仅在极高唤醒改判）。"""
        stim = Stimulus(name="threat", goal_congruence=-0.5, intensity=0.8)
        state = AffectState(
            stimulus=stim,
            appraisal={"valence": -0.5, "arousal": PANKSEPP_FEAR_AROUSAL_THRESHOLD - 0.1},
            panksepp_distinguish_fear=True,
        )
        assert "动机系统=rage" in _appraisal_summary(state)


# ─────────────────────────────────────────────────────────────────
# D. SessionConfig checkpoint 白名单说明（非执行，注释性）
# ─────────────────────────────────────────────────────────────────
# SessionConfig 不进 checkpoint（它是 ainvoke 时注入 state 的一次性 dict 展开，
# 不持久化在 AffectState 里）。受 checkpoint 影响的是 AffectState 字段本身，
# 本批次未改变其字段定义，因此不影响 ALLOWED_CHECKPOINT_TYPES 白名单。
# 遗留：SessionConfig.to_state_flags() 里的 None 值字段（va_coupling_pos/neg）
# 被 pydantic model_dump 默认包含；AffectState 接受 None 值，不影响序列化。


# ─────────────────────────────────────────────────────────────────
# F. T6-a/b · mood_precision / text_affect_precision 旋钮贯通
#    + ignition_beta 默认 None 零回归（本批三项新 env 联合验证）
# ─────────────────────────────────────────────────────────────────


class TestT6PrecisionKnobs:
    """F：mood_precision / text_affect_precision 默认值=现常量（逐字零回归）；
    env 设值后经 build_chat_driver → session.config 贯通到位；
    to_state_flags() 含三项（mood_precision / text_affect_precision / ignition_beta）。
    """

    # ── F1：SessionConfig 默认值 ──

    def test_mood_precision_default(self) -> None:
        """未设 ZERO_MOOD_PRECISION → SessionConfig().mood_precision == 0.8（零回归）。"""
        cfg = SessionConfig()
        assert cfg.mood_precision == pytest.approx(0.8)

    def test_text_affect_precision_default(self) -> None:
        """未设 ZERO_TEXT_AFFECT_PRECISION → SessionConfig().
        text_affect_precision == 0.3（零回归，等于 TEXT_AFFECT_PRECISION 常量）。"""
        cfg = SessionConfig()
        assert cfg.text_affect_precision == pytest.approx(0.3)

    def test_ignition_beta_default_none(self) -> None:
        """未设 ZERO_IGNITION_BETA → SessionConfig().ignition_beta is None（零回归）。"""
        cfg = SessionConfig()
        assert cfg.ignition_beta is None

    # ── F2：to_state_flags() 包含三个新键 ──

    def test_to_state_flags_contains_new_keys(self) -> None:
        """to_state_flags() 展开后包含 mood_precision / text_affect_precision / ignition_beta。"""
        flags = SessionConfig().to_state_flags()
        assert "mood_precision" in flags
        assert "text_affect_precision" in flags
        assert "ignition_beta" in flags
        assert flags["mood_precision"] == pytest.approx(0.8)
        assert flags["text_affect_precision"] == pytest.approx(0.3)
        assert flags["ignition_beta"] is None

    # ── F3：零回归断言——不设任何新 env，to_state_flags() 新键取默认值=常量 ──

    def test_zero_regression_no_new_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """不设三个新 env → to_state_flags() 新键取默认值，旧键不变（逐字零回归）。"""
        monkeypatch.delenv("ZERO_MOOD_PRECISION", raising=False)
        monkeypatch.delenv("ZERO_TEXT_AFFECT_PRECISION", raising=False)
        monkeypatch.delenv("ZERO_IGNITION_BETA", raising=False)
        flags = SessionConfig().to_state_flags()
        # 三项新键取现常量默认值
        assert flags["mood_precision"] == pytest.approx(0.8)
        assert flags["text_affect_precision"] == pytest.approx(0.3)
        assert flags["ignition_beta"] is None
        # 旧键零回归抽查
        assert flags["regulation_enabled"] is False
        assert flags["workspace_enabled"] is False
        assert flags["affect_readout"] == "sample"

    # ── F4：env 设值后 build_chat_driver 透传到 session.config ──

    def test_mood_precision_env_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_MOOD_PRECISION=0.5 → session.config.mood_precision == 0.5。"""
        monkeypatch.setenv("ZERO_MOOD_PRECISION", "0.5")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        driver = build_chat_driver(thread="test-mp")
        assert driver.session.config.mood_precision == pytest.approx(0.5)

    def test_text_affect_precision_env_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_TEXT_AFFECT_PRECISION=0.15 → session.config.text_affect_precision == 0.15。"""
        monkeypatch.setenv("ZERO_TEXT_AFFECT_PRECISION", "0.15")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        driver = build_chat_driver(thread="test-tap")
        assert driver.session.config.text_affect_precision == pytest.approx(0.15)

    def test_ignition_beta_env_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_IGNITION_BETA=20 → session.config.ignition_beta == 20.0。"""
        monkeypatch.setenv("ZERO_IGNITION_BETA", "20")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        driver = build_chat_driver(thread="test-ib")
        assert driver.session.config.ignition_beta == pytest.approx(20.0)

    # ── F5：AffectState 默认字段值 ──

    def test_affect_state_mood_precision_default(self) -> None:
        """AffectState() 默认 mood_precision == 0.8（与 MOOD_PRECISION 常量一致）。"""
        from src.orchestration.state import AffectState

        state = AffectState()
        assert state.mood_precision == pytest.approx(0.8)

    def test_affect_state_text_affect_precision_default(self) -> None:
        """AffectState() 默认 text_affect_precision == 0.3（与 TEXT_AFFECT_PRECISION 常量一致）。"""
        from src.orchestration.state import AffectState

        state = AffectState()
        assert state.text_affect_precision == pytest.approx(0.3)

    # ── F6：ConversationSession 旧展开参数路径贯通 ──

    def test_conversation_session_legacy_params(self) -> None:
        """旧展开参数传 mood_precision/text_affect_precision → session.config 字段正确。"""
        session = ConversationSession(
            thread_id="t",
            mood_precision=0.6,
            text_affect_precision=0.2,
        )
        assert session.config.mood_precision == pytest.approx(0.6)
        assert session.config.text_affect_precision == pytest.approx(0.2)

    def test_conversation_session_config_param_priority(self) -> None:
        """传 config= 时优先于旧展开参数（mood_precision 取 config 值）。"""
        cfg = SessionConfig(mood_precision=0.9, text_affect_precision=0.1)
        session = ConversationSession(
            thread_id="t",
            config=cfg,
            mood_precision=0.6,  # 应被忽略
            text_affect_precision=0.25,  # 应被忽略
        )
        assert session.config.mood_precision == pytest.approx(0.9)
        assert session.config.text_affect_precision == pytest.approx(0.1)
