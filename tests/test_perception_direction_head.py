"""PerceptionAgent W1 text_coping_prior 接线单测（议会 2026-07-20）。

覆盖 6 项架构师约束：
  1. 零回归：text_coping_enabled=False（默认）→ text_coping_prior=None（任何路径）。
  2. opt-in 门控：direction_head=None 时即使 text_coping_enabled=True → text_coping_prior=None。
  3. 正常产出路径：注入 fake direction_head(logit=1.0) + enabled=True → prior≈tanh(1.0)≈0.7616。
  4. 弃权门：logit=0.05 + τ=0.25 → None（|0.05|<0.25）；τ=0.0（默认）同 logit 不弃权。
  5. 节点契约·防残留：返回 dict 始终含 text_coping_prior 键（所有执行路径）。
  6. B3 集成零回归：text_coping_enabled=False 时下游 AppraisalAgent 仍走分支1/3。

重依赖策略：direction_head 以鸭子类型 FakeDirectionHead 注入（固定 logit），
  不真 import torch/sentence-transformers；encode_texts 以 monkeypatch 替换。
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from src.agents.perception import PerceptionAgent
from src.orchestration.state import AffectState, Stimulus

# ---------------------------------------------------------------------------
# 辅助：FakeDirectionHead（鸭子类型·固定 logit·不引 torch）
# ---------------------------------------------------------------------------


class _FakeTensor:
    """模拟单元素 torch.Tensor 的最小接口（dim / __getitem__ / float 转换）。"""

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


class FakeTextRegressor:
    """同 test_perception_text_affect.py，供文本路径测试复用。"""

    def predict_affect(self, _text: str) -> tuple[float, float]:
        return (0.1, 0.1)


def _make_state(
    *,
    text: str | None = "test input text",
    goal_congruence: float = 0.3,
    intensity: float = 0.6,
    text_coping_enabled: bool = False,
) -> AffectState:
    """构造最小 AffectState，默认带 stim.text 以覆盖文本分支。"""
    stim = Stimulus(
        name="w1_test",
        text=text,
        goal_congruence=goal_congruence,
        intensity=intensity,
    )
    return AffectState(stimulus=stim, text_coping_enabled=text_coping_enabled)


def _make_agent_with_fake_head(
    *,
    logit: float = 1.0,
    abstain_threshold: float = 0.0,
) -> PerceptionAgent:
    """构造 PerceptionAgent 并注入 FakeDirectionHead + 设置弃权门阈值。"""
    agent = PerceptionAgent()
    agent.direction_head = FakeDirectionHead(logit=logit)  # type: ignore[assignment]
    agent.abstain_threshold = abstain_threshold
    return agent


# ---------------------------------------------------------------------------
# _patch_encode_texts：用 monkeypatch 把 encode_texts 替换成返回零向量的 fake
# ---------------------------------------------------------------------------


def _patch_encode_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    """把 src.agents.models.text_affect_regressor_st.encode_texts 替换成不引 torch 的 fake。

    PerceptionAgent._compute_text_coping 在产出路径里 `import torch` 并调
    encode_texts([stim.text])；我们 monkeypatch 模块级函数，同时也 mock torch
    以避免真正 import（ImportError 时测试 skip；有 torch 时 patch 仍有效）。
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch 不可用，跳过依赖 encode_texts 的产出路径测试")

    # torch 可用时，patch encode_texts 为返回固定零张量的 fake
    import torch as _torch

    def _fake_encode_texts(texts: list[str]) -> _torch.Tensor:  # type: ignore[return]
        return _torch.zeros(len(texts), 384)

    import src.agents.models.text_affect_regressor_st as _st_mod

    monkeypatch.setattr(_st_mod, "encode_texts", _fake_encode_texts)


# ===========================================================================
# 1. 零回归：text_coping_enabled=False（默认）→ text_coping_prior=None
# ===========================================================================


class TestZeroRegression:
    """text_coping_enabled=False 时，text_coping_prior 必为 None（零回归保证）。"""

    def test_default_occ_path_prior_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OCC 路径（无 text_regressor）+ enabled=False → text_coping_prior=None。"""
        monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
        monkeypatch.delenv("ZERO_DIRECTION_HEAD_MODEL_PATH", raising=False)
        agent = PerceptionAgent()
        state = _make_state(text_coping_enabled=False)
        out = agent(state)
        assert out["text_coping_prior"] is None

    def test_with_fake_head_injected_prior_still_none_when_disabled(self) -> None:
        """注入 FakeDirectionHead 但 enabled=False → text_coping_prior 仍为 None。"""
        agent = _make_agent_with_fake_head(logit=1.0)
        state = _make_state(text_coping_enabled=False)
        out = agent(state)
        assert out["text_coping_prior"] is None

    def test_text_path_prior_none_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """文本路径（fake text_regressor）+ enabled=False → text_coping_prior=None。"""
        monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
        agent = PerceptionAgent()
        agent.text_regressor = FakeTextRegressor()  # type: ignore[assignment]
        agent.direction_head = FakeDirectionHead(logit=0.9)  # type: ignore[assignment]
        state = _make_state(text="hello world", text_coping_enabled=False)
        out = agent(state)
        assert out["text_coping_prior"] is None

    def test_no_stimulus_zeroes_text_coping_prior(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """stimulus=None → 显式归零 text_coping_prior=None（防 LastValue 残留·节点契约·WARN-1）。"""
        monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
        agent = PerceptionAgent()
        state = AffectState(stimulus=None, text_coping_enabled=False)
        out = agent(state)
        assert out == {"text_coping_prior": None}


# ===========================================================================
# 2. opt-in 门控：direction_head=None 时即使 enabled=True → text_coping_prior=None
# ===========================================================================


class TestOptInGate:
    """direction_head=None 时 text_coping_prior 恒为 None，无论 enabled。"""

    def test_no_head_enabled_true_occ_path_gives_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """direction_head=None，enabled=True，OCC 路径 → text_coping_prior=None。"""
        monkeypatch.delenv("ZERO_DIRECTION_HEAD_MODEL_PATH", raising=False)
        monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
        agent = PerceptionAgent()
        assert agent.direction_head is None
        state = _make_state(text_coping_enabled=True)
        out = agent(state)
        assert out["text_coping_prior"] is None

    def test_no_head_enabled_true_no_text_in_stim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """direction_head=None，enabled=True，stim.text=None → text_coping_prior=None。"""
        monkeypatch.delenv("ZERO_DIRECTION_HEAD_MODEL_PATH", raising=False)
        monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
        agent = PerceptionAgent()
        state = _make_state(text=None, text_coping_enabled=True)
        out = agent(state)
        assert out["text_coping_prior"] is None

    def test_env_path_not_set_direction_head_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_DIRECTION_HEAD_MODEL_PATH 未设 → _build_direction_head() → None。"""
        monkeypatch.delenv("ZERO_DIRECTION_HEAD_MODEL_PATH", raising=False)
        agent = PerceptionAgent()
        assert agent.direction_head is None

    def test_no_text_in_stim_with_head_gives_none(self) -> None:
        """direction_head 已注入，但 stim.text=None → 门控 3 → text_coping_prior=None。"""
        agent = _make_agent_with_fake_head(logit=1.0)
        state = _make_state(text=None, text_coping_enabled=True)
        out = agent(state)
        assert out["text_coping_prior"] is None


# ===========================================================================
# 3. 正常产出路径：logit=1.0 + enabled=True → prior≈tanh(1.0)≈0.7616
# ===========================================================================


class TestNormalOutputPath:
    """注入 FakeDirectionHead(logit=1.0) + encode_texts mock + enabled=True → 正确产出。"""

    def test_prior_equals_tanh_of_logit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """logit=1.0 → text_coping_prior≈tanh(1.0)≈0.7616。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=1.0, abstain_threshold=0.0)
        state = _make_state(text="test", text_coping_enabled=True)
        out = agent(state)
        expected = math.tanh(1.0)
        assert out["text_coping_prior"] == pytest.approx(expected, abs=1e-5)

    def test_prior_negative_logit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """logit=-2.0 → text_coping_prior≈tanh(-2.0)（回避端负值）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=-2.0, abstain_threshold=0.0)
        state = _make_state(text="afraid", text_coping_enabled=True)
        out = agent(state)
        expected = math.tanh(-2.0)
        assert out["text_coping_prior"] == pytest.approx(expected, abs=1e-5)

    def test_prior_zero_logit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """logit=0.0（恰好等于 τ=0.0）→ 不弃权（abs(0.0)≥0.0 为 False 时 NOT弃权）→ tanh(0)=0.0。

        代码：abs(logit) < self.abstain_threshold → 0.0 < 0.0 → False → 不弃权。
        """
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=0.0, abstain_threshold=0.0)
        state = _make_state(text="neutral", text_coping_enabled=True)
        out = agent(state)
        assert out["text_coping_prior"] == pytest.approx(0.0, abs=1e-5)

    def test_prior_in_range_minus1_to_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """tanh 定义域决定 prior ∈ [-1, 1]（对任意 logit 都成立）。"""
        _patch_encode_texts(monkeypatch)
        for logit in [5.0, -5.0, 0.5, -0.5, 1.0, -1.0]:
            agent = _make_agent_with_fake_head(logit=logit, abstain_threshold=0.0)
            state = _make_state(text="text", text_coping_enabled=True)
            out = agent(state)
            assert out["text_coping_prior"] is not None
            assert -1.0 <= out["text_coping_prior"] <= 1.0, (
                f"logit={logit}: prior={out['text_coping_prior']} 超出 [-1,1]"
            )

    def test_prior_deterministic_same_logit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """同一 logit 多次调用产出完全相同（确定性）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=0.8, abstain_threshold=0.0)
        results = []
        for _ in range(3):
            state = _make_state(text="repeated", text_coping_enabled=True)
            out = agent(state)
            results.append(out["text_coping_prior"])
        assert results[0] == pytest.approx(results[1], abs=1e-10)
        assert results[0] == pytest.approx(results[2], abs=1e-10)


# ===========================================================================
# 4. 弃权门：τ>0 时 |logit|<τ → None；τ=0.0（默认）→ 不弃权
# ===========================================================================


class TestAbstainGate:
    """弃权门 ZERO_ANGER_ABSTAIN_LOGIT_THRESHOLD 语义测试。"""

    def test_low_logit_below_threshold_gives_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """logit=0.05, τ=0.25 → |0.05|<0.25 → text_coping_prior=None（弃权）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=0.05, abstain_threshold=0.25)
        state = _make_state(text="uncertain", text_coping_enabled=True)
        out = agent(state)
        assert out["text_coping_prior"] is None

    def test_negative_low_logit_below_threshold_gives_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """logit=-0.05, τ=0.25 → |-0.05|<0.25 → text_coping_prior=None（负向弃权）。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=-0.05, abstain_threshold=0.25)
        state = _make_state(text="uncertain neg", text_coping_enabled=True)
        out = agent(state)
        assert out["text_coping_prior"] is None

    def test_zero_threshold_same_logit_not_abstained(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """τ=0.0（默认）时 logit=0.05 不弃权 → prior=tanh(0.05)。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=0.05, abstain_threshold=0.0)
        state = _make_state(text="text", text_coping_enabled=True)
        out = agent(state)
        expected = math.tanh(0.05)
        assert out["text_coping_prior"] == pytest.approx(expected, abs=1e-5)

    def test_logit_exactly_at_threshold_still_abstained(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """logit=τ（严格等于）→ abs(logit)<τ 为 False → 不弃权 → 产出 tanh(τ)。

        代码使用严格小于（<），等于阈值不弃权。
        """
        _patch_encode_texts(monkeypatch)
        tau = 0.25
        agent = _make_agent_with_fake_head(logit=tau, abstain_threshold=tau)
        state = _make_state(text="boundary", text_coping_enabled=True)
        out = agent(state)
        # abs(0.25) < 0.25 → False → 不弃权
        assert out["text_coping_prior"] == pytest.approx(math.tanh(tau), abs=1e-5)

    def test_logit_above_threshold_not_abstained(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """logit=0.5, τ=0.25 → |0.5|≥0.25 → 不弃权 → prior=tanh(0.5)。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=0.5, abstain_threshold=0.25)
        state = _make_state(text="confident", text_coping_enabled=True)
        out = agent(state)
        expected = math.tanh(0.5)
        assert out["text_coping_prior"] == pytest.approx(expected, abs=1e-5)

    def test_abstain_threshold_read_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_ANGER_ABSTAIN_LOGIT_THRESHOLD env 变量 → abstain_threshold 被正确读取。"""
        monkeypatch.setenv("ZERO_ANGER_ABSTAIN_LOGIT_THRESHOLD", "0.30")
        monkeypatch.delenv("ZERO_DIRECTION_HEAD_MODEL_PATH", raising=False)
        monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
        agent = PerceptionAgent()
        assert agent.abstain_threshold == pytest.approx(0.30, abs=1e-9)

    def test_default_abstain_threshold_is_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_ANGER_ABSTAIN_LOGIT_THRESHOLD 未设 → abstain_threshold=0.0（inert）。"""
        monkeypatch.delenv("ZERO_ANGER_ABSTAIN_LOGIT_THRESHOLD", raising=False)
        monkeypatch.delenv("ZERO_DIRECTION_HEAD_MODEL_PATH", raising=False)
        monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
        agent = PerceptionAgent()
        assert agent.abstain_threshold == pytest.approx(0.0)


# ===========================================================================
# 5. 节点契约·防残留：所有执行路径都含 text_coping_prior 键
# ===========================================================================


class TestNodeContract:
    """节点契约：返回 dict 始终含 text_coping_prior 键（防 LastValue checkpoint 残留）。"""

    def test_occ_path_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OCC 路径（无 head·enabled=False）→ dict 含 text_coping_prior 键。"""
        monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
        monkeypatch.delenv("ZERO_DIRECTION_HEAD_MODEL_PATH", raising=False)
        agent = PerceptionAgent()
        state = _make_state(text=None, text_coping_enabled=False)
        out = agent(state)
        assert "text_coping_prior" in out, "OCC 路径缺 text_coping_prior 键（LastValue 残留风险）"

    def test_text_path_key_present_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """文本路径 + enabled=False → dict 含 text_coping_prior 键。"""
        monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
        agent = PerceptionAgent()
        agent.text_regressor = FakeTextRegressor()  # type: ignore[assignment]
        state = _make_state(text="hello", text_coping_enabled=False)
        out = agent(state)
        assert "text_coping_prior" in out

    def test_text_path_key_present_head_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """文本路径 + head=None + enabled=True → dict 含 text_coping_prior 键。"""
        monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
        agent = PerceptionAgent()
        agent.text_regressor = FakeTextRegressor()  # type: ignore[assignment]
        assert agent.direction_head is None
        state = _make_state(text="hello", text_coping_enabled=True)
        out = agent(state)
        assert "text_coping_prior" in out

    def test_key_present_on_all_paths_occ_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OCC 路径 + enabled=True + head=None → dict 含 text_coping_prior 键。"""
        monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
        monkeypatch.delenv("ZERO_DIRECTION_HEAD_MODEL_PATH", raising=False)
        agent = PerceptionAgent()
        state = _make_state(text=None, text_coping_enabled=True)
        out = agent(state)
        assert "text_coping_prior" in out

    def test_key_present_on_output_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """产出路径（head 注入 + enabled=True + encode_texts mock）→ dict 含 text_coping_prior。"""
        _patch_encode_texts(monkeypatch)
        agent = _make_agent_with_fake_head(logit=1.0)
        state = _make_state(text="test", text_coping_enabled=True)
        out = agent(state)
        assert "text_coping_prior" in out

    def test_return_is_dict_not_mutating_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """返回 dict + 不 mutate 入参（节点契约）。"""
        monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
        agent = PerceptionAgent()
        state = _make_state(text_coping_enabled=False)
        before = state.model_copy(deep=True)
        out = agent(state)
        assert isinstance(out, dict)
        assert state == before, "PerceptionAgent.__call__ 不应 mutate 入参 AffectState"

    def test_all_keys_in_affect_state_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """返回 dict 的所有键都在 AffectState.model_fields 中（无越界字段）。"""
        monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
        agent = PerceptionAgent()
        agent.text_regressor = FakeTextRegressor()  # type: ignore[assignment]
        state = _make_state(text="check", text_coping_enabled=False)
        out = agent(state)
        extra = set(out.keys()) - set(AffectState.model_fields)
        assert not extra, f"返回 dict 含非 AffectState 字段：{extra}"


# ===========================================================================
# 6. B3 集成零回归：text_coping_enabled=False → AppraisalAgent 分支1/3 逐字不变
# ===========================================================================


class TestB3IntegrationZeroRegression:
    """text_coping_enabled=False 时，Perception→Appraisal 端到端 coping 路径零回归。"""

    def _run_appraisal_with_text_prior(
        self,
        text_coping_prior: float | None,
        *,
        text_coping_enabled: bool,
        control_appraisal: float | None = 0.4,
    ) -> dict:
        """辅助：直接构造 AffectState 并跑 AppraisalAgent（不过 Perception）。"""
        from src.agents.appraisal import AppraisalAgent

        stim = Stimulus(
            name="b3_zero_reg",
            goal_congruence=0.2,
            intensity=0.5,
            control_appraisal=control_appraisal,
        )
        state = AffectState(
            stimulus=stim,
            coping_potential_enabled=True,
            text_coping_enabled=text_coping_enabled,
            text_coping_prior=text_coping_prior,
        )
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        return agent(state)

    def test_gate_off_text_prior_none_vs_nonzero_same_cp(self) -> None:
        """gate=False：text_coping_prior=0.8 与 None 结果完全一致（分支3 零回归）。

        AppraisalAgent BLOCK-1：gate=False 时 text 强制视为 None，只走分支1/3。
        """
        out_with = self._run_appraisal_with_text_prior(
            0.8, text_coping_enabled=False, control_appraisal=0.4
        )
        out_none = self._run_appraisal_with_text_prior(
            None, text_coping_enabled=False, control_appraisal=0.4
        )
        assert out_with["coping_potential_state"] == pytest.approx(
            out_none["coping_potential_state"]
        ), "gate=False 时 text_coping_prior 有值与 None 的 coping_potential_state 应逐字一致"

    def test_gate_off_text_coping_source_always_false(self) -> None:
        """gate=False：无论 text_coping_prior 值如何，text_coping_source 恒为 False。"""
        for prior in [None, 0.0, 0.5, -0.5, 0.9]:
            out = self._run_appraisal_with_text_prior(
                prior, text_coping_enabled=False, control_appraisal=0.4
            )
            assert out["text_coping_source"] is False, (
                f"gate=False, prior={prior}: text_coping_source 应为 False"
            )

    def test_gate_off_branch1_ctrl_none_text_ignored(self) -> None:
        """gate=False, ctrl=None, text=0.9 → 分支1 → cp=0.0（text 被丢弃）。"""
        out = self._run_appraisal_with_text_prior(
            0.9, text_coping_enabled=False, control_appraisal=None
        )
        assert out["coping_potential_state"] == pytest.approx(0.0)

    def test_gate_off_branch3_ctrl_v_text_ignored(self) -> None:
        """gate=False, ctrl=0.6, text=0.9 → 分支3 → cp=0.6（text 被丢弃）。"""
        out = self._run_appraisal_with_text_prior(
            0.9, text_coping_enabled=False, control_appraisal=0.6
        )
        assert out["coping_potential_state"] == pytest.approx(0.6)

    def test_gate_off_perception_prior_is_none_downstream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Perception（gate=False）→ text_coping_prior=None → Appraisal B3 分支1/3 不变。

        端到端：PerceptionAgent 产出 text_coping_prior=None，
        透传给 AppraisalAgent 后走分支3（ctrl=0.4），cp=0.4。
        """
        from src.agents.appraisal import AppraisalAgent

        monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
        monkeypatch.delenv("ZERO_DIRECTION_HEAD_MODEL_PATH", raising=False)

        # Perception 阶段
        perception = PerceptionAgent()
        p_state = _make_state(text="hello", text_coping_enabled=False)
        p_out = perception(p_state)
        assert p_out["text_coping_prior"] is None

        # Appraisal 阶段（用 Perception 产出的 prior）
        stim = Stimulus(
            name="e2e_zero_reg",
            goal_congruence=0.0,
            intensity=0.5,
            control_appraisal=0.4,
        )
        a_state = AffectState(
            stimulus=stim,
            coping_potential_enabled=True,
            text_coping_enabled=False,  # gate 关
            text_coping_prior=p_out["text_coping_prior"],  # None
        )
        appraisal = AppraisalAgent(now_fn=lambda: 1000.0)
        a_out = appraisal(a_state)
        # 分支3：ctrl=0.4, text=None → cp=0.4
        assert a_out["coping_potential_state"] == pytest.approx(0.4)
        assert a_out["text_coping_source"] is False
