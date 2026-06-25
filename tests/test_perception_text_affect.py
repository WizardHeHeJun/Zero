"""PerceptionAgent 文本路径（ZERO_TEXT_AFFECT_BACKEND=st）单测。

覆盖：
- 默认关（env 未设置）→ text_regressor is None，走 OCC 路径
- 默认关时 sentence_transformers 不被 import
- 设 backend=st 但无 MODEL_PATH → fail-soft 回退 OCC
- 节点契约：返回 dict、key 合法、不 mutate 入参
- 文本路径：fake regressor 注入 → features/backend 正确
- OCC 回退：text_regressor 非 None 但 stim.text is None → OCC
- torch/ST smoke：STTextAffectRegressor.predict_affect 输出形状与值域
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from src.agents.perception import PerceptionAgent
from src.orchestration.state import AffectState, Stimulus

AFFECT_STATE_FIELDS = set(AffectState.model_fields)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


class FakeTextRegressor:
    """鸭子类型 fake：predict_affect 返回固定 (valence, arousal)。"""

    def __init__(self, valence: float = 0.3, arousal: float = -0.1) -> None:
        self.valence = valence
        self.arousal = arousal

    def predict_affect(self, _text: str) -> tuple[float, float]:
        return (self.valence, self.arousal)


def _make_state(
    *,
    with_text: str | None = None,
    goal_congruence: float = 0.5,
    standard_compliance: float = 0.2,
    attitude_appeal: float = 0.3,
    intensity: float = 0.8,
) -> AffectState:
    return AffectState(
        stimulus=Stimulus(
            name="test_stim",
            text=with_text,
            goal_congruence=goal_congruence,
            standard_compliance=standard_compliance,
            attitude_appeal=attitude_appeal,
            intensity=intensity,
        )
    )


# ---------------------------------------------------------------------------
# 1. 默认关：env 未设置 → text_regressor is None，走 OCC
# ---------------------------------------------------------------------------


def test_default_off_no_text_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """env 未设 ZERO_TEXT_AFFECT_BACKEND → text_regressor is None；返回 OCC 路径。"""
    monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
    monkeypatch.delenv("ZERO_TEXT_AFFECT_MODEL_PATH", raising=False)

    agent = PerceptionAgent()
    assert agent.text_regressor is None

    state = _make_state(
        goal_congruence=0.6, standard_compliance=0.2, attitude_appeal=0.4, intensity=0.9
    )
    out = agent(state)

    assert isinstance(out, dict)
    assert out  # 非空增量
    features = out["features"]
    assert len(features) == 4
    # OCC 路径：features = [goal_congruence, standard_compliance, attitude_appeal, intensity]
    assert features[0] == pytest.approx(0.6)
    assert features[1] == pytest.approx(0.2)
    assert features[2] == pytest.approx(0.4)
    assert features[3] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 2. 默认关时 sentence_transformers 不应被 import（验证等价命题：构造不触发 import）
# ---------------------------------------------------------------------------


def test_default_off_does_not_import_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认关时，PerceptionAgent 构造不会触发 sentence_transformers 的 import。

    策略：在 sys.modules 中将 sentence_transformers 屏蔽（置 None/删除后注入哨兵），
    然后构造 PerceptionAgent，确认构造成功且 text_regressor 仍为 None。
    若 _build_text_affect_regressor 在 env='' 时偷偷 import 了重依赖，哨兵会
    触发 ImportError 并让测试失败。
    """
    monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
    monkeypatch.delenv("ZERO_TEXT_AFFECT_MODEL_PATH", raising=False)

    # 把 sentence_transformers 注入一个会爆的 sentinel，若真的被 import 则测试失败
    original = sys.modules.get("sentence_transformers", None)

    class _BlockedModule:
        """任何属性访问都抛 ImportError，模拟库不可用。"""

        def __getattr__(self, item: str) -> Any:
            raise ImportError(f"sentence_transformers 不应在默认关时被访问：{item}")

    # 若已被前序测试 import，暂时替换为 sentinel
    monkeypatch.setitem(sys.modules, "sentence_transformers", _BlockedModule())  # type: ignore[arg-type]

    try:
        agent = PerceptionAgent()
        # 构造成功，不触发 sentinel 里的 ImportError
        assert agent.text_regressor is None
    finally:
        # monkeypatch 自动恢复，但显式防御
        if original is None:
            sys.modules.pop("sentence_transformers", None)


# ---------------------------------------------------------------------------
# 3. backend=st 但无 MODEL_PATH → fail-soft 回退 OCC（无异常）
# ---------------------------------------------------------------------------


def test_backend_st_without_model_path_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZERO_TEXT_AFFECT_BACKEND=st 但未设 MODEL_PATH → text_regressor is None，无异常。"""
    monkeypatch.setenv("ZERO_TEXT_AFFECT_BACKEND", "st")
    monkeypatch.delenv("ZERO_TEXT_AFFECT_MODEL_PATH", raising=False)

    agent = PerceptionAgent()
    assert agent.text_regressor is None  # fail-soft 回退


# ---------------------------------------------------------------------------
# 4. 节点契约（默认 OCC 路径）
# ---------------------------------------------------------------------------


def test_node_contract_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认 PerceptionAgent：返回 dict、key 在 AffectState 字段内、不 mutate 入参。"""
    monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
    monkeypatch.delenv("ZERO_TEXT_AFFECT_MODEL_PATH", raising=False)

    agent = PerceptionAgent()
    state = _make_state()
    before = state.model_copy(deep=True)

    out = agent(state)

    assert isinstance(out, dict)
    assert out  # 非空
    assert set(out).issubset(AFFECT_STATE_FIELDS)
    assert state == before  # 未 mutate 入参


# ---------------------------------------------------------------------------
# 5. 文本路径：fake regressor → features 与 backend 正确
# ---------------------------------------------------------------------------


def test_node_contract_with_text_stimulus(monkeypatch: pytest.MonkeyPatch) -> None:
    """fake regressor(0.3,-0.1) + stim.text='hello'.

    BUG 修复后行为：文本路径 features 是 OCC 布局（取 stim 的 OCC 维度），
    而不是 [valence, arousal, intensity, 0.0]（旧 BUG 行为）。
    回归器的 (v,a) 写入增量 text_affect，不污染 features。
    """
    monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
    monkeypatch.delenv("ZERO_TEXT_AFFECT_MODEL_PATH", raising=False)

    agent = PerceptionAgent()
    # 用鸭子类型 fake 注入，绕过真实模型加载
    agent.text_regressor = FakeTextRegressor(valence=0.3, arousal=-0.1)  # type: ignore[assignment]

    goal_congruence = 0.5
    standard_compliance = 0.2
    attitude_appeal = 0.3
    intensity = 0.75
    state = _make_state(
        with_text="hello",
        goal_congruence=goal_congruence,
        standard_compliance=standard_compliance,
        attitude_appeal=attitude_appeal,
        intensity=intensity,
    )
    before = state.model_copy(deep=True)

    out = agent(state)

    assert isinstance(out, dict)
    features = out["features"]
    assert len(features) == 4
    # 修复后：features = OCC 布局（不再是 [valence, arousal, intensity, 0.0]）
    assert features[0] == pytest.approx(goal_congruence)  # goal_congruence（非 valence）
    assert features[1] == pytest.approx(standard_compliance)  # standard_compliance（非 arousal）
    assert features[2] == pytest.approx(attitude_appeal)  # attitude_appeal（非 intensity）
    assert features[3] == pytest.approx(intensity)  # intensity

    # 回归器的 (v,a) 写入 text_affect，不进 features
    assert out["text_affect"] == pytest.approx((0.3, -0.1))

    trace = out["trace"]
    assert len(trace) == 1
    assert trace[0]["backend"] == "st_text"
    assert trace[0]["node"] == "perception"
    assert trace[0]["text_affect"] == pytest.approx((0.3, -0.1))

    # 不 mutate 入参
    assert state == before


# ---------------------------------------------------------------------------
# 5b. 文本路径 features 是 OCC 布局（坐实不被 valence 污染）
# ---------------------------------------------------------------------------


def test_text_path_features_are_occ_layout_not_valence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """文本路径下 features[0] 是 goal_congruence，不是 fake regressor 返回的 valence。

    构造非零 OCC 维度（goal_congruence=0.8）+ fake regressor 返回与 OCC 明显不同的 (v,a)
    （valence=-0.9），断言 features[0]==0.8（非 -0.9），确保生存流不被 valence 污染。
    """
    monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
    monkeypatch.delenv("ZERO_TEXT_AFFECT_MODEL_PATH", raising=False)

    agent = PerceptionAgent()
    # fake regressor 返回与 OCC 截然不同的 valence（-0.9 vs goal_congruence=0.8）
    agent.text_regressor = FakeTextRegressor(valence=-0.9, arousal=0.5)  # type: ignore[assignment]

    state = _make_state(
        with_text="test text",
        goal_congruence=0.8,
        standard_compliance=0.3,
        attitude_appeal=0.1,
        intensity=0.9,
    )

    out = agent(state)
    features = out["features"]

    # OCC 布局：features[0] 是 goal_congruence=0.8，不是 valence=-0.9
    assert features[0] == pytest.approx(0.8), (
        f"features[0] 应为 goal_congruence=0.8，实际={features[0]}（被 valence 污染了）"
    )
    assert features[0] != pytest.approx(-0.9)  # 明确排除旧 BUG 行为

    # text_affect 包含回归器的 (v,a)
    assert out["text_affect"] == pytest.approx((-0.9, 0.5))


# ---------------------------------------------------------------------------
# 6. text_regressor 非 None 但 stim.text is None → 回退 OCC
# ---------------------------------------------------------------------------


def test_occ_fallback_when_no_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """text_regressor 已注入但 stim.text is None → 走 OCC，backend==occ_placeholder。"""
    monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)

    agent = PerceptionAgent()
    agent.text_regressor = FakeTextRegressor()  # type: ignore[assignment]

    # 不传 text（stim.text is None）
    state = _make_state(
        with_text=None,
        goal_congruence=0.4,
        standard_compliance=0.1,
        attitude_appeal=0.2,
        intensity=0.6,
    )

    out = agent(state)

    trace = out["trace"]
    assert trace[0]["backend"] == "occ_placeholder"

    # OCC 路径 features 验证
    features = out["features"]
    assert features[0] == pytest.approx(0.4)
    assert features[1] == pytest.approx(0.1)
    assert features[2] == pytest.approx(0.2)
    assert features[3] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# W1. OCC 路径显式归零 text_affect 的回归测试
#     实现工程师修复：OCC 路径 return 补了 "text_affect": None，
#     防多轮同 thread 时上轮文本流经 Checkpointer 跨轮泄漏。
# ---------------------------------------------------------------------------


def test_occ_path_returns_text_affect_none_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认关（text_regressor=None）走 OCC 路径时，增量 dict 中 text_affect 键显式存在且为 None。

    验证：键存在（非缺键）且值为 None——Checkpointer 合并时才会把 state 字段重置为 None，
    缺键不会覆盖上轮残留的 text_affect 值。
    """
    monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
    monkeypatch.delenv("ZERO_TEXT_AFFECT_MODEL_PATH", raising=False)

    agent = PerceptionAgent()
    assert agent.text_regressor is None  # 确认走 OCC 路径

    state = _make_state(
        goal_congruence=0.5, standard_compliance=0.2, attitude_appeal=0.3, intensity=0.8
    )
    out = agent(state)

    # 键必须存在（不是缺键），且值显式为 None
    assert "text_affect" in out, (
        "OCC 路径应在增量 dict 中显式返回 text_affect 键（防 Checkpointer 残留）"
    )
    assert out["text_affect"] is None, f"OCC 路径 text_affect 应为 None，实际={out['text_affect']}"


def test_occ_path_resets_text_affect_after_text_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """跨轮语义：上轮走文本路径（text_affect=(0.9,0.9)），本轮走 OCC 路径。

    验证 OCC 路径的增量把 text_affect 显式置 None，Checkpointer 合并后不会残留上轮的值。
    模拟方式：同一 agent 先跑文本路径、再跑 OCC 路径，确认后者的增量含显式 None。
    """
    monkeypatch.delenv("ZERO_TEXT_AFFECT_BACKEND", raising=False)
    monkeypatch.delenv("ZERO_TEXT_AFFECT_MODEL_PATH", raising=False)

    agent = PerceptionAgent()

    # 第 1 轮：注入 fake regressor，文本路径，产出 text_affect=(0.9,0.9)
    agent.text_regressor = FakeTextRegressor(valence=0.9, arousal=0.9)  # type: ignore[assignment]
    state_round1 = _make_state(with_text="happy", goal_congruence=0.5, intensity=0.8)
    out_round1 = agent(state_round1)
    assert out_round1.get("text_affect") == pytest.approx((0.9, 0.9))  # 确认第 1 轮产出

    # 第 2 轮：移除 regressor，本轮走 OCC 路径
    agent.text_regressor = None
    state_round2 = _make_state(
        with_text=None,  # 无文本，走 OCC
        goal_congruence=0.4,
        standard_compliance=0.1,
        attitude_appeal=0.2,
        intensity=0.6,
    )
    out_round2 = agent(state_round2)

    # OCC 路径增量必须显式含 text_affect=None，才能覆盖 Checkpointer 中上轮残留的 (0.9,0.9)
    assert "text_affect" in out_round2, (
        "OCC 路径增量必须含 text_affect 键，否则 Checkpointer 会残留上轮的 (0.9,0.9)"
    )
    assert out_round2["text_affect"] is None, (
        f"OCC 路径 text_affect 应为 None 以清除上轮残留，实际={out_round2['text_affect']}"
    )


# ---------------------------------------------------------------------------
# 7. torch/ST smoke：STTextAffectRegressor.predict_affect 形状与值域
#    缺 torch 或 sentence_transformers 则仅此测试 skip，前 6 个不受影响
# ---------------------------------------------------------------------------


def test_st_regressor_predict_affect_shape() -> None:
    """STTextAffectRegressor().predict_affect 返回 2-tuple，值在 [-1,1]。

    缺 torch 或 sentence_transformers 时 skip（重依赖 importorskip 在函数体内，
    不影响模块顶层的其余 6 个纯 Python 测试）。
    """
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    from src.agents.models.text_affect_regressor_st import STTextAffectRegressor

    model = STTextAffectRegressor()
    result = model.predict_affect("happy day")

    assert isinstance(result, tuple)
    assert len(result) == 2
    v, a = result
    assert isinstance(v, float)
    assert isinstance(a, float)
    assert -1.0 <= v <= 1.0, f"valence={v} 超出 [-1,1]"
    assert -1.0 <= a <= 1.0, f"arousal={a} 超出 [-1,1]"
