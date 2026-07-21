"""A1 WARN-3 fear 专属门单测（B1 BLOCK 前置·议会 2026-07-21·Task 10-11）。

覆盖六 class：
  1. perception 路径一门：_compute_text_coping A1 fear 专属门行为。
  2. motivational_system 路径二门：coping<COPING_FEAR_THRESHOLD 分支门控。
  3. control_appraisal<0 直注路径：经 _appraisal_summary 消费，验证输出门堵住直注泄漏。
  4. anger 不误伤：confrontational 域 / 高 coping 路径完全不受 fear 门约束。
  5. 零回归：AffectState/SessionConfig 默认 fear_domain_enabled=False，coping=0.0 门关=rage。
  6. MCP 治理：fear_domain_enabled 在 _MCP_GOVERNANCE_GATED_FLAGS 中，client override 被忽略。

重依赖策略：
  - DirectionHead 以 FakeDirectionHead（鸭子类型）注入，encode_texts 以 monkeypatch 替换。
  - torch 不可用时相关用例 importorskip 跳过。
  - MCP 测试 mcp 不可用时 importorskip 跳过。
  - 其余路径（motivational_system / state 字段 / SessionConfig）均无重依赖，无 skip。
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from src.agents.emotion_lexicon import motivational_system
from src.agents.perception import PerceptionAgent
from src.orchestration.runner import SessionConfig
from src.orchestration.state import AffectState, Stimulus

# ---------------------------------------------------------------------------
# 共用 Fake 辅助（仿 test_b2_fear_domain.py 风格）
# ---------------------------------------------------------------------------


class _FakeTensor:
    """最小 torch.Tensor 接口：dim / __getitem__ / __float__。"""

    def __init__(self, value: float) -> None:
        self.value = value

    def dim(self) -> int:
        return 1

    def __getitem__(self, _idx: int) -> _FakeTensor:
        return self

    def __float__(self) -> float:
        return self.value


class FakeDirectionHead:
    """鸭子类型 fake：__call__(emb) → _FakeTensor(fixed_logit)，不引 torch。"""

    def __init__(self, logit: float = 1.0) -> None:
        self.logit = logit

    def __call__(self, emb: Any) -> _FakeTensor:
        return _FakeTensor(self.logit)


def _make_agent_with_fake_head(
    *,
    logit: float = 1.0,
    abstain_threshold: float = 0.0,
) -> PerceptionAgent:
    """构造 PerceptionAgent 并注入 FakeDirectionHead。"""
    agent = PerceptionAgent()
    agent.direction_head = FakeDirectionHead(logit=logit)  # type: ignore[assignment]
    agent.abstain_threshold = abstain_threshold
    return agent


def _patch_encode_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    """把 encode_texts 替换成不引真 torch 推理的 fake（返回零张量）。

    若 torch 不可用则 importorskip 跳过。
    """
    torch = pytest.importorskip("torch")

    def _fake_encode_texts(texts: list[str]) -> Any:
        return torch.zeros(len(texts), 384)

    import src.agents.models.text_affect_regressor_st as _st_mod

    monkeypatch.setattr(_st_mod, "encode_texts", _fake_encode_texts)


def _make_perception_state(
    *,
    text: str | None = "test input",
    text_coping_enabled: bool = True,
    domain: str | None = None,
    control_appraisal: float | None = None,
    fear_domain_enabled: bool = False,
) -> AffectState:
    """构造最小 AffectState，供 PerceptionAgent 路径一测试用。"""
    stim = Stimulus(
        name="a1_test",
        text=text,
        goal_congruence=0.3,
        intensity=0.6,
        control_appraisal=control_appraisal,
        domain=domain,  # type: ignore[arg-type]
    )
    return AffectState(
        stimulus=stim,
        text_coping_enabled=text_coping_enabled,
        fear_domain_enabled=fear_domain_enabled,
    )


def _make_appraisal_state(
    *,
    coping_potential_state: float,
    fear_domain_enabled: bool = False,
    coping_potential_enabled: bool = True,
) -> AffectState:
    """构造最小 AffectState，供 _appraisal_summary 路径三测试用。"""
    stim = Stimulus(name="t", goal_congruence=-0.5, intensity=0.6)
    return AffectState(
        stimulus=stim,
        appraisal={"valence": -0.6, "arousal": 0.6},  # (-v,+a) 象限
        coping_potential_state=coping_potential_state,
        coping_potential_enabled=coping_potential_enabled,
        fear_domain_enabled=fear_domain_enabled,
    )


# ===========================================================================
# Class 1. perception 路径一门：_compute_text_coping A1 fear 专属门
# ===========================================================================


class TestPerceptionFearGate:
    """perception._compute_text_coping 中 A1 fear 专属门行为（路径一）。

    门只约束 survival_narrative 域，confrontational 域不受影响。
    """

    def test_survival_fear_logit_gate_closed_prior_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """survival_narrative + fear logit(−) + 门关(默认) → prior=None（硬弃·流卫生）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=-1.0, abstain_threshold=0.0)
        state = _make_perception_state(
            domain="survival_narrative", text_coping_enabled=True, fear_domain_enabled=False
        )
        out = agent(state)
        assert out["text_coping_prior"] is None

    def test_survival_fear_logit_gate_open_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """survival_narrative + fear logit(−) + 门开 → prior≈tanh(-1.0)（放行）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=-1.0, abstain_threshold=0.0)
        state = _make_perception_state(
            domain="survival_narrative", text_coping_enabled=True, fear_domain_enabled=True
        )
        out = agent(state)
        assert out["text_coping_prior"] == pytest.approx(math.tanh(-1.0), abs=1e-5)

    def test_confrontational_anger_gate_closed_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """confrontational + anger logit(+) + 门关 → prior≈tanh(+1.0)（anger 不受 fear 门）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=1.0, abstain_threshold=0.0)
        state = _make_perception_state(
            domain="confrontational", text_coping_enabled=True, fear_domain_enabled=False
        )
        out = agent(state)
        assert out["text_coping_prior"] == pytest.approx(math.tanh(1.0), abs=1e-5)

    def test_confrontational_fear_logit_gate_closed_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """confrontational + fear logit(−) + 门关 → prior=None（off-domain 方向门·非 fear 专属门）。

        注意：这里是原 B2 域方向门（off-domain 硬弃），不是 A1 fear 专属门；
        两个机制结果相同，但语义不同——confrontational 域失配属域方向匹配失败，
        不属 fear 专属门保护范畴（A1 只约束 survival_narrative）。
        """
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=-1.0, abstain_threshold=0.0)
        state = _make_perception_state(
            domain="confrontational", text_coping_enabled=True, fear_domain_enabled=False
        )
        out = agent(state)
        assert out["text_coping_prior"] is None

    def test_domain_none_gate_closed_prior_not_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """domain=None + 门关 → prior≈tanh(logit)（域门整体旁路·逐字不变·零回归）。

        A-W1 旁路：domain=None 时 fear 门不介入，prior 值与改前完全一致。
        """
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=-1.0, abstain_threshold=0.0)
        state = _make_perception_state(
            domain=None, text_coping_enabled=True, fear_domain_enabled=False
        )
        out = agent(state)
        assert out["text_coping_prior"] == pytest.approx(math.tanh(-1.0), abs=1e-5)


# ===========================================================================
# Class 2. motivational_system 路径二门：coping<-0.3 分支 fear_domain_enabled 门控
# ===========================================================================


class TestMotivationalSystemFearGate:
    """emotion_lexicon.motivational_system 中 fear_domain_enabled 路径二门控。

    coping_potential<COPING_FEAR_THRESHOLD(-0.3) → fear_domain_enabled=False→rage/True→fear。
    其他象限与 coping 段不受影响。
    """

    def test_coping_minus_half_gate_closed_is_rage(self) -> None:
        """coping=-0.5 + 门关(默认) → rage（WARN-3 保守回退）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=-0.5)
        assert result == "rage"

    def test_coping_minus_half_gate_open_is_fear(self) -> None:
        """coping=-0.5 + 门开 → fear（路径二·单点完整·门开放行）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=-0.5, fear_domain_enabled=True)
        assert result == "fear"

    def test_coping_minus_0_31_gate_closed_is_rage(self) -> None:
        """coping=-0.31(刚过 -0.3) + 门关 → rage（边界·WARN-3）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=-0.31)
        assert result == "rage"

    def test_coping_zero_gate_closed_is_rage(self) -> None:
        """coping=0.0 + 门关 → rage（零回归·中间段·不受 fear 门影响）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=0.0)
        assert result == "rage"

    def test_coping_half_gate_closed_is_rage(self) -> None:
        """coping=+0.5 + 门关 → rage（高 coping 趋近段·不受 fear 门影响）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=0.5)
        assert result == "rage"

    def test_seeking_gate_closed_unchanged(self) -> None:
        """(+v,+a) + coping=-0.5 + 门关 → seeking（其他象限不受 fear 门）。"""
        result = motivational_system(0.6, 0.6, coping_potential=-0.5)
        assert result == "seeking"

    def test_seeking_gate_open_unchanged(self) -> None:
        """(+v,+a) + coping=-0.5 + 门开 → seeking（其他象限不受 fear 门，门开也不误伤）。"""
        result = motivational_system(0.6, 0.6, coping_potential=-0.5, fear_domain_enabled=True)
        assert result == "seeking"

    def test_care_gate_closed_unchanged(self) -> None:
        """(+v,-a) + coping=-0.5 + 门关 → care（其他象限不受 fear 门）。"""
        result = motivational_system(0.6, -0.6, coping_potential=-0.5)
        assert result == "care"

    def test_care_gate_open_unchanged(self) -> None:
        """(+v,-a) + coping=-0.5 + 门开 → care（门开不误伤 care 象限）。"""
        result = motivational_system(0.6, -0.6, coping_potential=-0.5, fear_domain_enabled=True)
        assert result == "care"

    def test_panic_grief_gate_closed_unchanged(self) -> None:
        """(-v,-a) + coping=-0.5 + 门关 → panic_grief（其他象限不受 fear 门）。"""
        result = motivational_system(-0.6, -0.6, coping_potential=-0.5)
        assert result == "panic_grief"

    def test_panic_grief_gate_open_unchanged(self) -> None:
        """(-v,-a) + coping=-0.5 + 门开 → panic_grief（门开不误伤 panic_grief 象限）。"""
        result = motivational_system(-0.6, -0.6, coping_potential=-0.5, fear_domain_enabled=True)
        assert result == "panic_grief"


# ===========================================================================
# Class 3. control_appraisal<0 直注路径（_appraisal_summary 消费）
# ===========================================================================


class TestControlAppraisalDirectInjectGate:
    """control_appraisal<0 经 _appraisal_summary → motivational_system 路径二门控。

    验证：perception 路径一门拦不住 control_appraisal 直注的场景，
    被输出门（路径二 motivational_system fear_domain_enabled）正确堵住。
    """

    @staticmethod
    def _summary_from_state(state: AffectState) -> str:
        from src.agents.language import _appraisal_summary

        return _appraisal_summary(state)

    def test_ctrl_negative_gate_closed_shows_rage(self) -> None:
        """ctrl=-0.5 + coping_potential_state=-0.5 + 门关(默认) → 摘要动机系统=rage（输出门堵住）。

        这条验证 perception 路径一门拦不住的「control_appraisal 直注」泄漏路径
        （直注不经 text_coping 流·不经 survival_narrative 域门），被路径二输出门堵住。
        """
        state = _make_appraisal_state(coping_potential_state=-0.5, fear_domain_enabled=False)
        summary = self._summary_from_state(state)
        assert "动机系统=rage" in summary

    def test_ctrl_negative_gate_open_shows_fear(self) -> None:
        """ctrl=-0.5 + coping_potential_state=-0.5 + 门开 → 摘要动机系统=fear（门开放行）。"""
        state = _make_appraisal_state(coping_potential_state=-0.5, fear_domain_enabled=True)
        summary = self._summary_from_state(state)
        assert "动机系统=fear" in summary

    def test_ctrl_positive_gate_closed_shows_rage(self) -> None:
        """ctrl=+0.5 + coping_potential_state=+0.5 + 门关 → rage（高 coping·无论门状态）。"""
        state = _make_appraisal_state(coping_potential_state=0.5, fear_domain_enabled=False)
        summary = self._summary_from_state(state)
        assert "动机系统=rage" in summary

    def test_ctrl_zero_gate_closed_shows_rage(self) -> None:
        """coping_potential_state=0.0 + 门关 → rage（中间段保守默认·零回归）。"""
        state = _make_appraisal_state(coping_potential_state=0.0, fear_domain_enabled=False)
        summary = self._summary_from_state(state)
        assert "动机系统=rage" in summary


# ===========================================================================
# Class 4. anger 不误伤：confrontational 域 / 高 coping 不受 fear 门约束
# ===========================================================================


class TestAngerNotAffected:
    """fear 专属门不误伤 anger 路径（confrontational 域 / coping>0.3）。

    守「anger confrontational 路径完全不受此门约束」的架构语义边界。
    """

    def test_confrontational_high_ctrl_gate_closed_coping_rage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """confrontational + ctrl=+0.5 + 门关 → prior≈tanh(+1.0)（anger 放行）。

        confrontational 域正向 logit 即使 fear 门关也能透传。
        """
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=1.0, abstain_threshold=0.0)
        state = _make_perception_state(
            domain="confrontational",
            text_coping_enabled=True,
            fear_domain_enabled=False,
            control_appraisal=0.5,
        )
        out = agent(state)
        assert out["text_coping_prior"] == pytest.approx(math.tanh(1.0), abs=1e-5)

    def test_motivational_high_coping_gate_closed_is_rage(self) -> None:
        """(-v,+a) + coping=+0.5 + 门关 → rage（高 coping·不受 fear 门）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=0.5, fear_domain_enabled=False)
        assert result == "rage"

    def test_motivational_high_coping_gate_open_still_rage(self) -> None:
        """(-v,+a) coping=+0.5 门开 → rage（高 coping 走 rage·fear 门只管 coping<-0.3）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=0.5, fear_domain_enabled=True)
        assert result == "rage"

    def test_motivational_mid_coping_gate_open_still_rage(self) -> None:
        """(-v,+a) + coping=0.0(中间段) + 门开 → rage（中间段保守默认·fear 门不介入中间段）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=0.0, fear_domain_enabled=True)
        assert result == "rage"


# ===========================================================================
# Class 5. 零回归：默认 fear_domain_enabled=False，与改前行为逐字一致
# ===========================================================================


class TestZeroRegressionFearDomain:
    """fear 专属门默认关时，所有现有路径行为与改前逐字一致。"""

    def test_affect_state_default_fear_domain_enabled_false(self) -> None:
        """AffectState 默认 fear_domain_enabled=False（零回归字段存在）。"""
        state = AffectState(stimulus=None)
        assert state.fear_domain_enabled is False

    def test_session_config_default_fear_domain_enabled_false(self) -> None:
        """SessionConfig 默认 fear_domain_enabled=False（零回归字段存在）。"""
        cfg = SessionConfig()
        assert cfg.fear_domain_enabled is False

    def test_session_config_to_state_flags_contains_key(self) -> None:
        """to_state_flags() 含 'fear_domain_enabled' 键（完整透传）。"""
        cfg = SessionConfig()
        flags = cfg.to_state_flags()
        assert "fear_domain_enabled" in flags
        assert flags["fear_domain_enabled"] is False

    def test_session_config_fear_domain_enabled_true_propagates(self) -> None:
        """SessionConfig(fear_domain_enabled=True) 正确透传到 flags。"""
        cfg = SessionConfig(fear_domain_enabled=True)
        assert cfg.fear_domain_enabled is True
        flags = cfg.to_state_flags()
        assert flags["fear_domain_enabled"] is True

    def test_coping_zero_gate_closed_rage_zero_regression(self) -> None:
        """coping=0.0 + 门关(默认) → rage（与改前 (-v,+a)=rage 逐字一致）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=0.0)
        assert result == "rage"

    def test_coping_zero_no_arg_gate_closed_rage_zero_regression(self) -> None:
        """不传 fear_domain_enabled/coping_potential → 默认 rage（向后兼容）。"""
        result = motivational_system(-0.6, 0.6)
        assert result == "rage"

    def test_seeking_quadrant_zero_regression(self) -> None:
        """(+v,+a) → seeking（fear 门默认关·其他象限零回归）。"""
        result = motivational_system(0.6, 0.6)
        assert result == "seeking"

    def test_care_quadrant_zero_regression(self) -> None:
        """(+v,-a) → care（fear 门默认关·其他象限零回归）。"""
        result = motivational_system(0.6, -0.6)
        assert result == "care"

    def test_panic_grief_quadrant_zero_regression(self) -> None:
        """(-v,-a) → panic_grief（fear 门默认关·其他象限零回归）。"""
        result = motivational_system(-0.6, -0.6)
        assert result == "panic_grief"


# ===========================================================================
# Class 6. MCP 治理：fear_domain_enabled 在 _MCP_GOVERNANCE_GATED_FLAGS 中
# ===========================================================================

pytest.importorskip("mcp")


class TestMCPGovernanceFearDomain:
    """MCP 治理门：fear_domain_enabled 受 _MCP_GOVERNANCE_GATED_FLAGS 保护。

    仿 test_mcp_server.py 中 governance 系列测试风格，
    显式 delenv 防 .env 预设 env 造成假绿。
    """

    def test_fear_domain_in_gated_flags(self) -> None:
        """_MCP_GOVERNANCE_GATED_FLAGS 含 'fear_domain_enabled'（静态断言·治理完整性）。"""
        from src.mcp_server.server import _MCP_GOVERNANCE_GATED_FLAGS

        assert "fear_domain_enabled" in _MCP_GOVERNANCE_GATED_FLAGS

    def test_governance_default_fear_domain_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """无 env 无 override → fear_domain_enabled=False（零回归·MCP 默认保守）。

        显式 delenv 防 shell 预设 env 时假绿。
        """
        from src.mcp_server.server import _build_session_config

        monkeypatch.delenv("ZERO_MCP_FEAR_DOMAIN_ENABLED", raising=False)
        cfg = _build_session_config(None)
        assert cfg.fear_domain_enabled is False

    def test_governance_client_override_fear_domain_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """client override fear_domain_enabled=True → 被治理静默忽略，config 仍 False。

        「生产关·MCP 开」旁路防御：client 不得经 config 覆写议会门控字段。
        """
        from src.mcp_server.server import _build_session_config

        monkeypatch.delenv("ZERO_MCP_FEAR_DOMAIN_ENABLED", raising=False)
        cfg = _build_session_config({"fear_domain_enabled": True})
        assert cfg.fear_domain_enabled is False

    def test_governance_env_fear_domain_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_MCP_FEAR_DOMAIN_ENABLED=true → fear_domain_enabled=True（env 治理正路）。"""
        from src.mcp_server.server import _build_session_config

        monkeypatch.setenv("ZERO_MCP_FEAR_DOMAIN_ENABLED", "true")
        cfg = _build_session_config(None)
        assert cfg.fear_domain_enabled is True

    def test_governance_env_fear_domain_false_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_MCP_FEAR_DOMAIN_ENABLED=false → fear_domain_enabled=False（显式 false）。"""
        from src.mcp_server.server import _build_session_config

        monkeypatch.setenv("ZERO_MCP_FEAR_DOMAIN_ENABLED", "false")
        cfg = _build_session_config(None)
        assert cfg.fear_domain_enabled is False

    def test_governance_fear_domain_override_does_not_affect_other_flags(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """client 同时 override fear_domain=True + contagion_alpha=0.1 → fear 忽略，contagion 生效。

        验证治理过滤仅针对门控字段，不误伤普通可覆写字段。
        """
        from src.mcp_server.server import _build_session_config

        monkeypatch.delenv("ZERO_MCP_FEAR_DOMAIN_ENABLED", raising=False)
        cfg = _build_session_config({"fear_domain_enabled": True, "contagion_alpha": 0.1})
        assert cfg.fear_domain_enabled is False  # 门控字段被过滤
        assert cfg.contagion_alpha == pytest.approx(0.1)  # 非门控字段正常覆写
