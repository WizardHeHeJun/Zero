"""B2 fear 域激活条件单测（议会 2026-07-20·Task 5）。

覆盖七大落点：
  1. 谓词四象限：_domain_direction_accepts 四种域×方向组合。
  2. neutral 两弃：neutral 域无论正负 sign 均返回 False。
  3. A-W1 旁路（零回归核心）：domain=None 时 text_coping_prior 逐字不变。
  4. 热路径域门集成：_compute_text_coping A1 域门行为（五路径）。
  5. 边界层 ValueError：model_validator _check_domain_ctrl_sign 六组合。
  6. mapping A-map 路径：stimulus_from_payload domain 提取/校验/不一致注入。
  7. 节点契约：_compute_text_coping 每条路径均返回含 "text_coping_prior" 键。

重依赖策略：
  - direction_head 以 FakeDirectionHead（鸭子类型）注入，不真引 torch 权重。
  - torch 依赖路径（encode_texts）走 monkeypatch 替换；torch 不可用时 importorskip 跳过。
  - 其余路径（谓词/validator/mapping）均无重依赖，无 skip。
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from src.agents.perception import PerceptionAgent, _domain_direction_accepts
from src.mcp_server.mapping import stimulus_from_payload
from src.orchestration.state import AffectState, Stimulus

# ---------------------------------------------------------------------------
# 共用 Fake 辅助（仿 test_perception_direction_head.py 风格）
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


def _make_state(
    *,
    text: str | None = "test input",
    text_coping_enabled: bool = True,
    domain: str | None = None,
    control_appraisal: float | None = None,
    fear_domain_enabled: bool = False,
) -> AffectState:
    """构造最小 AffectState，域轴可注入。"""
    stim = Stimulus(
        name="b2_test",
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


# ===========================================================================
# 1. 谓词四象限：_domain_direction_accepts 四组合
# ===========================================================================


class TestPredicateFourQuadrants:
    """_domain_direction_accepts 的四象限行为。"""

    def test_confrontational_positive_sign_true(self) -> None:
        """confrontational × 正方向 → True（anger home 域接受 anger 信号）。"""
        assert _domain_direction_accepts("confrontational", +0.5) is True

    def test_confrontational_negative_sign_false(self) -> None:
        """confrontational × 负方向 → False（anger 域拒绝 fear 信号）。"""
        assert _domain_direction_accepts("confrontational", -0.5) is False

    def test_survival_narrative_negative_sign_true(self) -> None:
        """survival_narrative × 负方向 → True（fear home 域接受 fear 信号）。"""
        assert _domain_direction_accepts("survival_narrative", -0.5) is True

    def test_survival_narrative_positive_sign_false(self) -> None:
        """survival_narrative × 正方向 → False（fear 域拒绝 anger 信号）。"""
        assert _domain_direction_accepts("survival_narrative", +0.5) is False

    def test_confrontational_large_positive(self) -> None:
        """confrontational × 大正值（0.9）→ True（边界充分）。"""
        assert _domain_direction_accepts("confrontational", +0.9) is True

    def test_survival_narrative_large_negative(self) -> None:
        """survival_narrative × 大负值（-0.9）→ True（边界充分）。"""
        assert _domain_direction_accepts("survival_narrative", -0.9) is True


# ===========================================================================
# 2. neutral 两弃：neutral 域正负 sign 均 False
# ===========================================================================


class TestNeutralAlwaysFalse:
    """neutral 是显式两不属 → 无论方向一律 False（≠ None 旁路）。"""

    def test_neutral_positive_sign_false(self) -> None:
        """neutral × 正方向 → False。"""
        assert _domain_direction_accepts("neutral", +0.5) is False

    def test_neutral_negative_sign_false(self) -> None:
        """neutral × 负方向 → False。"""
        assert _domain_direction_accepts("neutral", -0.5) is False

    def test_neutral_zero_sign_false(self) -> None:
        """neutral × 零 → False（零不属于任一 home）。"""
        assert _domain_direction_accepts("neutral", 0.0) is False

    def test_unknown_domain_also_false(self) -> None:
        """未知 domain 字串 → False（谓词 else 分支）。"""
        assert _domain_direction_accepts("unknown", +0.5) is False


# ===========================================================================
# 3. A-W1 旁路（零回归核心）：domain=None 时 text 信号不被域门截断
# ===========================================================================


class TestAW1DomainNoneBypass:
    """A-W1：domain=None 时域门整体旁路，prior 逐字等于 tanh(logit)。

    这是最重要的零回归断言——enabled=True 且 domain=None 时行为与旧代码完全一致。
    """

    def test_domain_none_positive_logit_prior_not_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """domain=None + logit=+1.0 + enabled=True → prior ≈ tanh(1.0)（不被域门截断）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=1.0, abstain_threshold=0.0)
        state = _make_state(domain=None, text_coping_enabled=True)
        out = agent(state)
        expected = math.tanh(1.0)
        assert out["text_coping_prior"] == pytest.approx(expected, abs=1e-5)

    def test_domain_none_negative_logit_prior_not_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """domain=None + logit=-1.0 + enabled=True → prior ≈ tanh(-1.0)（不被域门截断）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=-1.0, abstain_threshold=0.0)
        state = _make_state(domain=None, text_coping_enabled=True)
        out = agent(state)
        expected = math.tanh(-1.0)
        assert out["text_coping_prior"] == pytest.approx(expected, abs=1e-5)

    def test_domain_none_enabled_false_prior_is_none(self) -> None:
        """domain=None + enabled=False → prior=None（零回归·开关关闭路径）。"""
        agent = _make_agent_with_fake_head(logit=1.0, abstain_threshold=0.0)
        state = _make_state(domain=None, text_coping_enabled=False)
        out = agent(state)
        assert out["text_coping_prior"] is None

    def test_domain_none_no_domain_gate_effect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """domain=None 时无论 logit 正负，prior 均不被域门截断（逐字旧行为）。"""
        _patch_encode_texts(monkeypatch)
        for logit in [1.5, -1.5, 0.1, -0.1]:
            agent = _make_agent_with_fake_head(logit=logit, abstain_threshold=0.0)
            state = _make_state(domain=None, text_coping_enabled=True)
            out = agent(state)
            expected = math.tanh(logit)
            assert out["text_coping_prior"] == pytest.approx(expected, abs=1e-5), (
                f"domain=None 时 logit={logit} 应旁路域门，prior 应≈tanh({logit})"
            )


# ===========================================================================
# 4. 热路径域门集成：_compute_text_coping A1 域门行为
# ===========================================================================


class TestDomainGateIntegration:
    """_compute_text_coping 内 A1 域门的集成行为（五典型路径）。"""

    def test_confrontational_anger_logit_positive_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """confrontational + anger logit(+) → prior≈tanh(logit)（域匹配·透传）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=1.0, abstain_threshold=0.0)
        state = _make_state(domain="confrontational", text_coping_enabled=True)
        out = agent(state)
        assert out["text_coping_prior"] == pytest.approx(math.tanh(1.0), abs=1e-5)

    def test_confrontational_fear_logit_negative_blocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """confrontational + fear logit(−) → prior=None（域失配·硬弃）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=-1.0, abstain_threshold=0.0)
        state = _make_state(domain="confrontational", text_coping_enabled=True)
        out = agent(state)
        assert out["text_coping_prior"] is None

    def test_survival_narrative_fear_logit_negative_gate_closed_prior_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """survival_narrative + fear logit(−) + fear_domain_enabled=False(默认) → prior=None。

        WARN-3 fear 专属门（A1 路径一）：门关时 survival_narrative 域信号一律硬弃。
        """
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=-1.0, abstain_threshold=0.0)
        state = _make_state(
            domain="survival_narrative", text_coping_enabled=True, fear_domain_enabled=False
        )
        out = agent(state)
        assert out["text_coping_prior"] is None

    def test_survival_narrative_fear_logit_negative_gate_open_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """survival_narrative + fear logit(−) + fear_domain_enabled=True → prior≈tanh(-1.0)。

        WARN-3 fear 专属门开启后，fear home 域方向信号正常透传（A1 路径一·门开放行）。
        """
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=-1.0, abstain_threshold=0.0)
        state = _make_state(
            domain="survival_narrative", text_coping_enabled=True, fear_domain_enabled=True
        )
        out = agent(state)
        assert out["text_coping_prior"] == pytest.approx(math.tanh(-1.0), abs=1e-5)

    def test_survival_narrative_anger_logit_positive_blocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """survival_narrative + anger logit(+) → prior=None（域失配·硬弃）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=1.0, abstain_threshold=0.0)
        state = _make_state(domain="survival_narrative", text_coping_enabled=True)
        out = agent(state)
        assert out["text_coping_prior"] is None

    def test_neutral_positive_sign_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """neutral + logit(+) → prior=None（显式弃权·非 None 旁路）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=1.0, abstain_threshold=0.0)
        state = _make_state(domain="neutral", text_coping_enabled=True)
        out = agent(state)
        assert out["text_coping_prior"] is None

    def test_neutral_negative_sign_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """neutral + logit(−) → prior=None（显式弃权·非 None 旁路）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=-1.0, abstain_threshold=0.0)
        state = _make_state(domain="neutral", text_coping_enabled=True)
        out = agent(state)
        assert out["text_coping_prior"] is None


# ===========================================================================
# 5. 边界层 ValueError：model_validator _check_domain_ctrl_sign
# ===========================================================================


class TestModelValidatorBoundary:
    """Stimulus model_validator _check_domain_ctrl_sign 的六种合法/非法组合。"""

    # ── confrontational ────────────────────────────────────────────────────

    def test_confrontational_ctrl_positive_ok(self) -> None:
        """confrontational + ctrl=+0.5 → 合法（anger 符号匹配）。"""
        stim = Stimulus(
            name="v1", goal_congruence=0.0, domain="confrontational", control_appraisal=0.5
        )
        assert stim.control_appraisal == pytest.approx(0.5)

    def test_confrontational_ctrl_zero_ok(self) -> None:
        """confrontational + ctrl=0.0 → 合法（genuine-zero 不违反 >= 0）。"""
        stim = Stimulus(
            name="v2", goal_congruence=0.0, domain="confrontational", control_appraisal=0.0
        )
        assert stim.control_appraisal == pytest.approx(0.0)

    def test_confrontational_ctrl_negative_raises(self) -> None:
        """confrontational + ctrl=-0.1 → ValueError（fear 符号注入 anger 域拒绝）。"""
        with pytest.raises(ValueError, match="confrontational"):
            Stimulus(
                name="v3",
                goal_congruence=0.0,
                domain="confrontational",
                control_appraisal=-0.1,
            )

    # ── survival_narrative ─────────────────────────────────────────────────

    def test_survival_narrative_ctrl_negative_ok(self) -> None:
        """survival_narrative + ctrl=-0.5 → 合法（fear 符号匹配）。"""
        stim = Stimulus(
            name="v4",
            goal_congruence=0.0,
            domain="survival_narrative",
            control_appraisal=-0.5,
        )
        assert stim.control_appraisal == pytest.approx(-0.5)

    def test_survival_narrative_ctrl_zero_ok(self) -> None:
        """survival_narrative + ctrl=0.0 → 合法（genuine-zero 不违反 <= 0）。"""
        stim = Stimulus(
            name="v5",
            goal_congruence=0.0,
            domain="survival_narrative",
            control_appraisal=0.0,
        )
        assert stim.control_appraisal == pytest.approx(0.0)

    def test_survival_narrative_ctrl_positive_raises(self) -> None:
        """survival_narrative + ctrl=+0.1 → ValueError（anger 符号注入 fear 域拒绝）。"""
        with pytest.raises(ValueError, match="survival_narrative"):
            Stimulus(
                name="v6",
                goal_congruence=0.0,
                domain="survival_narrative",
                control_appraisal=0.1,
            )

    # ── neutral / domain=None → no-op ────────────────────────────────────

    def test_neutral_any_ctrl_ok(self) -> None:
        """neutral + 任意 ctrl → 合法（validator no-op）。"""
        for ctrl in [0.5, -0.5, 0.0, 1.0, -1.0]:
            stim = Stimulus(
                name="v7", goal_congruence=0.0, domain="neutral", control_appraisal=ctrl
            )
            assert stim.control_appraisal == pytest.approx(ctrl), (
                f"neutral + ctrl={ctrl} 应通过 validator"
            )

    def test_domain_none_any_ctrl_ok(self) -> None:
        """domain=None + 任意 ctrl → 合法（validator no-op，旁路）。"""
        for ctrl in [0.5, -0.5, 0.0, None]:
            stim = Stimulus(name="v8", goal_congruence=0.0, domain=None, control_appraisal=ctrl)
            assert stim.control_appraisal == ctrl

    def test_ctrl_none_any_domain_ok(self) -> None:
        """ctrl=None + 任意 domain → 合法（absent cue，validator no-op）。"""
        for domain in ["confrontational", "survival_narrative", "neutral", None]:
            stim = Stimulus(
                name="v9",
                goal_congruence=0.0,
                domain=domain,  # type: ignore[arg-type]
                control_appraisal=None,
            )
            assert stim.control_appraisal is None


# ===========================================================================
# 6. mapping A-map 路径：stimulus_from_payload domain 提取/校验/不一致注入
# ===========================================================================


class TestMappingAMap:
    """mapping.stimulus_from_payload B2 domain 字段提取与边界校验。"""

    def test_domain_confrontational_extracted(self) -> None:
        """payload 含 domain='confrontational' → Stimulus.domain='confrontational'。"""
        stim = stimulus_from_payload({"valence": 0.5, "arousal": 0.3, "domain": "confrontational"})
        assert stim.domain == "confrontational"

    def test_domain_survival_narrative_extracted(self) -> None:
        """payload 含 domain='survival_narrative' → Stimulus.domain='survival_narrative'。"""
        stim = stimulus_from_payload(
            {"valence": -0.3, "arousal": 0.4, "domain": "survival_narrative"}
        )
        assert stim.domain == "survival_narrative"

    def test_domain_neutral_extracted(self) -> None:
        """payload 含 domain='neutral' → Stimulus.domain='neutral'。"""
        stim = stimulus_from_payload({"valence": 0.0, "arousal": 0.1, "domain": "neutral"})
        assert stim.domain == "neutral"

    def test_domain_absent_defaults_none(self) -> None:
        """payload 无 domain 键 → Stimulus.domain=None（零回归旁路）。"""
        stim = stimulus_from_payload({"valence": 0.1, "arousal": 0.2})
        assert stim.domain is None

    def test_domain_integer_raises(self) -> None:
        """domain=123（非 str）→ ValueError（类型校验）。"""
        with pytest.raises(ValueError, match="domain"):
            stimulus_from_payload({"valence": 0.0, "arousal": 0.0, "domain": 123})

    def test_domain_invalid_string_raises(self) -> None:
        """domain='unknown'（非枚举值）→ ValueError。"""
        with pytest.raises(ValueError, match="domain"):
            stimulus_from_payload({"valence": 0.0, "arousal": 0.0, "domain": "unknown"})

    def test_inconsistent_domain_ctrl_sign_raises(self) -> None:
        """confrontational + coping=-0.3（fear 符号）→ ValueError（model_validator 触发）。"""
        with pytest.raises(ValueError):
            stimulus_from_payload(
                {
                    "valence": 0.5,
                    "arousal": 0.3,
                    "coping_potential": -0.3,
                    "domain": "confrontational",
                }
            )

    def test_consistent_domain_ctrl_sign_ok(self) -> None:
        """confrontational + coping=+0.4（anger 符号）→ 合法。"""
        stim = stimulus_from_payload(
            {
                "valence": 0.5,
                "arousal": 0.3,
                "coping_potential": 0.4,
                "domain": "confrontational",
            }
        )
        assert stim.domain == "confrontational"
        assert stim.control_appraisal == pytest.approx(0.4)

    def test_survival_narrative_consistent_ctrl_ok(self) -> None:
        """survival_narrative + coping=-0.5 → 合法。"""
        stim = stimulus_from_payload(
            {
                "valence": -0.3,
                "arousal": 0.4,
                "coping_potential": -0.5,
                "domain": "survival_narrative",
            }
        )
        assert stim.domain == "survival_narrative"
        assert stim.control_appraisal == pytest.approx(-0.5)


# ===========================================================================
# 7. 节点契约：_compute_text_coping 每条路径均含 "text_coping_prior" 键
# ===========================================================================


class TestNodeContractB2:
    """节点契约：所有执行路径返回 dict 均含 "text_coping_prior" 键（防 LastValue 残留）。"""

    def test_enabled_false_key_present(self) -> None:
        """enabled=False → dict 含 text_coping_prior 键（值为 None）。"""
        agent = _make_agent_with_fake_head(logit=1.0)
        state = _make_state(domain="confrontational", text_coping_enabled=False)
        out = agent(state)
        assert "text_coping_prior" in out
        assert out["text_coping_prior"] is None

    def test_domain_none_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """domain=None + enabled=True + head 注入 → 含键（域门旁路路径）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=1.0)
        state = _make_state(domain=None, text_coping_enabled=True)
        out = agent(state)
        assert "text_coping_prior" in out

    def test_domain_match_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """domain=confrontational + logit(+) → 含键（域匹配透传路径）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=1.0)
        state = _make_state(domain="confrontational", text_coping_enabled=True)
        out = agent(state)
        assert "text_coping_prior" in out

    def test_domain_mismatch_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """domain=confrontational + logit(−) → 含键（域失配 None 路径）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=-1.0)
        state = _make_state(domain="confrontational", text_coping_enabled=True)
        out = agent(state)
        assert "text_coping_prior" in out
        assert out["text_coping_prior"] is None

    def test_neutral_domain_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """domain=neutral → 含键（neutral 弃权路径）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=1.0)
        state = _make_state(domain="neutral", text_coping_enabled=True)
        out = agent(state)
        assert "text_coping_prior" in out
        assert out["text_coping_prior"] is None

    def test_no_head_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """direction_head=None + enabled=True → 含键（门控 2 归零路径）。"""
        monkeypatch.delenv("ZERO_DIRECTION_HEAD_MODEL_PATH", raising=False)
        monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
        agent = PerceptionAgent()
        assert agent.direction_head is None
        state = _make_state(domain="confrontational", text_coping_enabled=True)
        out = agent(state)
        assert "text_coping_prior" in out
        assert out["text_coping_prior"] is None

    def test_no_text_in_stim_key_present(self) -> None:
        """stim.text=None + enabled=True → 含键（门控 3 归零路径）。"""
        agent = _make_agent_with_fake_head(logit=1.0)
        state = _make_state(text=None, domain="confrontational", text_coping_enabled=True)
        out = agent(state)
        assert "text_coping_prior" in out
        assert out["text_coping_prior"] is None

    def test_all_returned_keys_valid_state_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """返回 dict 所有键均在 AffectState.model_fields 中（无越界字段）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=1.0)
        state = _make_state(domain="confrontational", text_coping_enabled=True)
        out = agent(state)
        extra = set(out.keys()) - set(AffectState.model_fields)
        assert not extra, f"返回 dict 含非 AffectState 字段：{extra}"
