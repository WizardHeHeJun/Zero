"""行为反馈流集成收口测试（T6；design.md §五 验收 + §三 ⚠ 硬门交互锚点）。

覆盖：门控贯通、门关/门开-regulation-关双重零回归、硬门滤除锚点、生效组合下的
有界影响、MCP 治理（overrides 静默忽略 + env 生效）。
纯函数层判据与变异见 test_behavior_feedback_stability / test_behavior_feedback_math。
"""

from __future__ import annotations

import pytest

from src.agents.affect_math import W_MAX_BEHAVIOR
from src.mcp_server.registry import SessionRegistry
from src.mcp_server.server import build_server
from src.orchestration.runner import ConversationSession
from src.orchestration.state import AffectState, Stimulus

_STIMS = [
    Stimulus(name=f"s{i}", goal_congruence=g, intensity=0.8) for i, g in enumerate((0.7, -0.5, 0.4))
]


def _session(thread_id: str, **kwargs: object) -> ConversationSession:
    """行为反馈生效组合的会话底座：efference 副本链 + regulation 触发 + workspace +
    非默认硬门（gate_fusion=False，全流原生精度加权——design §三 ⚠ 生效组合）。"""
    defaults: dict[str, object] = {
        "thread_id": thread_id,
        "motion_backend": "efference",
        "regulation_enabled": True,
        "workspace_enabled": True,
        "gate_fusion": False,
        "rng_seed": 23,
    }
    defaults.update(kwargs)
    return ConversationSession(**defaults)  # type: ignore[arg-type]


def test_default_flag_off() -> None:
    """门控默认关（零回归的第一重保证）。"""
    assert AffectState().behavior_feedback_enabled is False
    session = ConversationSession(thread_id="bf-default")
    assert session.config.behavior_feedback_enabled is False


async def test_flag_on_but_regulation_off_is_zero_regression() -> None:
    """门开但 regulation 关（生产默认）：voluntary 恒 None ⇒ 流恒缺席 ⇒ 与门关逐字相同。
    这是 absent-cue 在场门的集成级验证（G1 第二重保证）。"""
    plain = ConversationSession(
        thread_id="bf-reg-off-a",
        motion_backend="efference",
        workspace_enabled=True,
        gate_fusion=False,
        rng_seed=23,
    )
    flagged = ConversationSession(
        thread_id="bf-reg-off-b",
        motion_backend="efference",
        workspace_enabled=True,
        gate_fusion=False,
        behavior_feedback_enabled=True,
        rng_seed=23,
    )
    for stim in _STIMS:
        entry_plain = await plain.step(stim)
        entry_flagged = await flagged.step(stim)
        assert entry_flagged["valence_arousal"] == entry_plain["valence_arousal"]
        assert entry_flagged["prior_mu"] == entry_plain["prior_mu"]


async def test_default_hard_gate_filters_behavior_stream() -> None:
    """硬门交互锚点（design §三 ⚠·validate-prp 阶段补）：门开 + 流在场 +
    gate_fusion=True（默认硬门）→ 行为流 salience≈0.075·|a_expr| < 阈值 0.18 恒被滤除，
    post_mu 与门关逐字相同。特征化既有硬门语义，防将来误判为 bug。"""
    plain = _session("bf-hardgate-a", gate_fusion=True)
    flagged = _session("bf-hardgate-b", gate_fusion=True, behavior_feedback_enabled=True)
    for stim in _STIMS:
        entry_plain = await plain.step(stim)
        entry_flagged = await flagged.step(stim)
        assert entry_flagged["valence_arousal"] == entry_plain["valence_arousal"]


async def test_bounded_influence_in_effective_config() -> None:
    """有界影响（G2）：生效组合（efference + regulation 真触发 + workspace + 门关硬门）下，
    行为流对后验有**非零且有界**的影响。上界 = 凸组合位移界 w_b·|μ_b − post| ≤ 2·W_MAX
    （粗上界；W_MAX 封顶保证）。轮 1 无副本可用（流缺席），断言从轮 2 起。"""
    plain = _session("bf-bounded-a")
    flagged = _session("bf-bounded-b", behavior_feedback_enabled=True)
    diffs: list[float] = []
    for stim in _STIMS:
        entry_plain = await plain.step(stim)
        entry_flagged = await flagged.step(stim)
        va_plain = entry_plain["valence_arousal"]
        va_flagged = entry_flagged["valence_arousal"]
        assert va_plain is not None and va_flagged is not None
        diffs.append(abs(va_flagged[1] - va_plain[1]))
        # 有界：任何一轮的 arousal 位移不得超过凸组合粗上界
        assert diffs[-1] <= 2.0 * W_MAX_BEHAVIOR + 1e-9
    # 非零：regulation 每轮真触发（suppression 默认策略改变 affect）⇒ 副本 voluntary
    # 非 None ⇒ 从轮 2 起流在场，整个轨迹不应全程无影响
    assert any(d > 0.0 for d in diffs), "生效组合下行为流应产生可测影响（全零=流从未在场）"


async def test_efference_copy_voluntary_present_with_regulation() -> None:
    """前置链自检：regulation 开 + efference 档 ⇒ 副本 voluntary 路非 None（在场门底座）。
    若此断言红，上面有界影响测试的「非零」就失去了机制前提（防按错误前提解读绿灯）。"""
    session = _session("bf-chain-check")
    await session.step(_STIMS[0])
    copy = session.last_motion_efference
    assert copy is not None
    assert copy["voluntary"] is not None  # 调节确实改变了表达 ⇒ δ≠0


# ── MCP 治理（议会必改 #7）───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mcp_overrides_cannot_enable_behavior_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """client 经 config 传 behavior_feedback_enabled/motion_backend 均被静默忽略
    （治理白名单）；仅部署端 env 可开。"""
    monkeypatch.delenv("ZERO_MCP_BEHAVIOR_FEEDBACK", raising=False)
    monkeypatch.delenv("ZERO_MCP_MOTION_BACKEND", raising=False)
    registry = SessionRegistry()
    server = build_server(registry)
    await server.call_tool(
        "zero.open_session",
        {
            "session_id": "bf-gov-1",
            "config": {"behavior_feedback_enabled": True, "motion_backend": "efference"},
        },
    )
    session = await registry.get("bf-gov-1")
    assert session is not None
    assert session.config.behavior_feedback_enabled is False  # override 被忽略
    assert session.config.motion_backend == "synth"  # override 被忽略
    await server.call_tool("zero.close_session", {"session_id": "bf-gov-1"})


@pytest.mark.asyncio
async def test_mcp_env_enables_behavior_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    """成对 env 入口生效（design D6 教训：进白名单必须配 env，否则永久取默认）。"""
    monkeypatch.setenv("ZERO_MCP_BEHAVIOR_FEEDBACK", "true")
    monkeypatch.setenv("ZERO_MCP_MOTION_BACKEND", "efference")
    registry = SessionRegistry()
    server = build_server(registry)
    await server.call_tool("zero.open_session", {"session_id": "bf-gov-2"})
    session = await registry.get("bf-gov-2")
    assert session is not None
    assert session.config.behavior_feedback_enabled is True
    assert session.config.motion_backend == "efference"
    await server.call_tool("zero.close_session", {"session_id": "bf-gov-2"})


@pytest.mark.asyncio
async def test_mcp_motion_backend_env_invalid_falls_back_to_synth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非法 env 值回落 synth（零回归档）而非崩会话——部署端配置错不扳倒会话开启。"""
    monkeypatch.setenv("ZERO_MCP_MOTION_BACKEND", "turbo")
    registry = SessionRegistry()
    server = build_server(registry)
    await server.call_tool("zero.open_session", {"session_id": "bf-gov-3"})
    session = await registry.get("bf-gov-3")
    assert session is not None
    assert session.config.motion_backend == "synth"
    await server.call_tool("zero.close_session", {"session_id": "bf-gov-3"})
