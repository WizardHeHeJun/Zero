"""`zero.motion` 工具面：门控、契约不变量、只读性、相位续接。

契约不变量按对面 `params_animate` 的拒收条件逐条钉——这些不是学术担忧，
违反了对面会当场 rejected。
"""

from __future__ import annotations

import re

import pytest

from src.mcp_server.registry import SessionRegistry
from src.mcp_server.server import (
    MOTION_DEFAULT_FPS,
    MOTION_MAX_KEYFRAMES,
    MOTION_MAX_SEGMENT_MS,
    build_server,
)

# 消费方提取机读码用的正则（与 server.py 头部文档同源）——用它做断言才算证明"可提取"。
_CODE_RE = re.compile(r"\[zero:([a-z][a-z0-9-]*)\]")


async def _call(server, name: str, **kwargs):
    return await server.call_tool(name, kwargs)


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZERO_MOTION_ENABLED", "true")


@pytest.mark.asyncio
async def test_disabled_by_default_yields_extractable_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """未开门控 → motion-disabled，且**消费方正则真能提取出来**。

    ⚠ 这条专防下划线写法（`motion_disabled`）——那样写正则匹配不到，
    消费方拿到 None，等于错误码形同虚设。
    """
    monkeypatch.delenv("ZERO_MOTION_ENABLED", raising=False)
    server = build_server(SessionRegistry())
    with pytest.raises(Exception) as excinfo:
        await _call(server, "zero.motion", session_id="whatever")
    code = _CODE_RE.search(str(excinfo.value))
    assert code is not None, f"机读码提取失败：{excinfo.value}"
    assert code.group(1) == "motion-disabled"


@pytest.mark.asyncio
async def test_unknown_session(enabled: None) -> None:
    server = build_server(SessionRegistry())
    with pytest.raises(Exception) as excinfo:
        await _call(server, "zero.motion", session_id="nope")
    code = _CODE_RE.search(str(excinfo.value))
    assert code is not None and code.group(1) == "unknown-session"


@pytest.mark.asyncio
async def test_contract_invariants(enabled: None) -> None:
    """t_ms 从 0 起算且严格升序、同段键集一致、帧数不越上限。"""
    registry = SessionRegistry()
    server = build_server(registry)
    await _call(server, "zero.open_session", session_id="m1")
    try:
        result = await _call(server, "zero.motion", session_id="m1", duration_ms=3000)
        payload = _payload(result)
        frames = payload["keyframes"]
        stamps = [f["t_ms"] for f in frames]
        assert stamps[0] == 0
        assert all(b > a for a, b in zip(stamps, stamps[1:], strict=False))
        assert len({frozenset(f["params"]) for f in frames}) == 1
        assert len(frames) <= MOTION_MAX_KEYFRAMES
    finally:
        await _call(server, "zero.close_session", session_id="m1")


@pytest.mark.asyncio
async def test_oversized_duration_is_clamped_not_rejected(enabled: None) -> None:
    """超长段 clamp 而非报错，且帧数仍不越 600（显式算术保证，非"建议 fps"）。"""
    registry = SessionRegistry()
    server = build_server(registry)
    await _call(server, "zero.open_session", session_id="m2")
    try:
        result = await _call(server, "zero.motion", session_id="m2", duration_ms=999_999)
        payload = _payload(result)
        assert payload["phase_ms"] <= MOTION_MAX_SEGMENT_MS
        assert len(payload["keyframes"]) <= MOTION_MAX_KEYFRAMES
    finally:
        await _call(server, "zero.close_session", session_id="m2")


@pytest.mark.asyncio
async def test_phase_continues_across_pulls(enabled: None) -> None:
    """跨两次拉取相位累积推进——不累积就会每段都从头开始，拼接点跳变。"""
    registry = SessionRegistry()
    server = build_server(registry)
    await _call(server, "zero.open_session", session_id="m3")
    try:
        first = _payload(await _call(server, "zero.motion", session_id="m3", duration_ms=1000))
        second = _payload(await _call(server, "zero.motion", session_id="m3", duration_ms=1000))
        assert second["phase_ms"] > first["phase_ms"]
        assert second["keyframes"][0]["t_ms"] == 0  # 但段内仍从 0 起算
    finally:
        await _call(server, "zero.close_session", session_id="m3")


@pytest.mark.asyncio
async def test_events_from_reply_text(enabled: None) -> None:
    """传回复文本 → 出离散行为意图（第 ③ 层）；不传则只有轨迹。"""
    registry = SessionRegistry()
    server = build_server(registry)
    await _call(server, "zero.open_session", session_id="m4")
    try:
        silent = _payload(await _call(server, "zero.motion", session_id="m4", duration_ms=500))
        assert silent["events"] == []
        spoken = _payload(
            await _call(
                server,
                "zero.motion",
                session_id="m4",
                duration_ms=500,
                reply_text="（点了点头）嗯，是这样。",
            )
        )
        assert "nod" in [e["name"] for e in spoken["events"]]
    finally:
        await _call(server, "zero.close_session", session_id="m4")


@pytest.mark.asyncio
async def test_motion_does_not_advance_engine(enabled: None) -> None:
    """⚠ 只读：反复拉取动作**不推进内核**——step 前后的引擎状态不因 motion 而变。"""
    registry = SessionRegistry()
    server = build_server(registry)
    await _call(server, "zero.open_session", session_id="m5")
    try:
        session = await registry.get("m5")
        assert session is not None
        for _ in range(5):
            await _call(server, "zero.motion", session_id="m5", duration_ms=500)
        assert session.last_affect()[0] is None  # 从未 step 过 → 仍是 None
    finally:
        await _call(server, "zero.close_session", session_id="m5")


def _payload(result) -> dict:
    """从 FastMCP 返回值里取 JSON 载荷（工具全是 structured_output=False）。"""
    import json

    content = result.content if hasattr(result, "content") else result[0]
    text = content[0].text if isinstance(content, list) else content.text
    return json.loads(text)


# ── motion_backend="directive" 消费闭环（PRP/motion/design-agent.md §1/§4）──────────


@pytest.mark.asyncio
async def test_directive_modulation_actually_used(enabled: None) -> None:
    """directive 里的调制系数确实被用上：构造一个与解析回退明显不同的 directive，
    断言输出轨迹随之改变（同 session_id/duration_ms ⇒ 同噪声种子/相位起点，
    唯一变量是是否注入了 motion_directive）——不满足于"没报错"。
    """
    session_id = "directive-diff"
    duration_ms = 2000

    # 基线：从未注入 motion_directive（motion_backend 默认 synth，session.last_motion_directive
    # 恒 None）⇒ 走现有解析回退 modulation_from_affect((0.0, 0.0))。
    baseline_registry = SessionRegistry()
    baseline_server = build_server(baseline_registry)
    await _call(baseline_server, "zero.open_session", session_id=session_id)
    try:
        baseline = _payload(
            await _call(
                baseline_server, "zero.motion", session_id=session_id, duration_ms=duration_ms
            )
        )
    finally:
        await _call(baseline_server, "zero.close_session", session_id=session_id)

    # 对照：手工把 session.last_motion_directive 设成幅度/速度都明显放大的值（独立
    # registry/server ⇒ 相位从同一起点重新算，唯一差异就是这一份 directive）。
    directive_registry = SessionRegistry()
    directive_server = build_server(directive_registry)
    await _call(directive_server, "zero.open_session", session_id=session_id)
    try:
        directive_session = await directive_registry.get(session_id)
        assert directive_session is not None
        directive_session.last_motion_directive = {
            "amplitude": 3.0,
            "speed": 3.0,
            "onset": 1.0,
            "scene": "idle",
            "events": [],
            "prosody_ref": None,
        }
        directed = _payload(
            await _call(
                directive_server, "zero.motion", session_id=session_id, duration_ms=duration_ms
            )
        )
    finally:
        await _call(directive_server, "zero.close_session", session_id=session_id)

    base_x = [f["params"]["FaceAngleX"] for f in baseline["keyframes"]]
    dir_x = [f["params"]["FaceAngleX"] for f in directed["keyframes"]]
    assert base_x != dir_x  # 轨迹确实随注入的调制系数改变
    assert max(abs(v) for v in dir_x) > max(abs(v) for v in base_x)  # 幅度确实放大


@pytest.mark.asyncio
async def test_directive_events_take_priority_over_reply_text(enabled: None) -> None:
    """directive 里已有的 events 优先于 reply_text 现场解析——reply_text 那一路本身不变。"""
    session_id = "directive-events"
    registry = SessionRegistry()
    server = build_server(registry)
    await _call(server, "zero.open_session", session_id=session_id)
    try:
        session = await registry.get(session_id)
        assert session is not None
        session.last_motion_directive = {
            "amplitude": 1.0,
            "speed": 1.0,
            "onset": 0.5,
            "scene": "idle",
            "events": [{"name": "shake", "intensity": 0.6, "direction": None, "source": "lexical"}],
            "prosody_ref": None,
        }
        # reply_text 会命中 nod（肯定句），但 directive 里已有 shake ⇒ 优先取 directive。
        result = _payload(
            await _call(
                server,
                "zero.motion",
                session_id=session_id,
                duration_ms=500,
                reply_text="没错，就是这样。",
            )
        )
        names = [e["name"] for e in result["events"]]
        assert names == ["shake"]
        assert "nod" not in names
    finally:
        await _call(server, "zero.close_session", session_id=session_id)


@pytest.mark.asyncio
async def test_directive_none_falls_back_without_error(enabled: None) -> None:
    """last_motion_directive 为 None（未 step / 门控没开 / 跨重启 resume）优雅退回，不抛异常。

    用「先产出一份 directive，再关会话重开（resume，新 ConversationSession 对象）」
    模拟跨重启：该只读实例属性不持久化，新对象上恒为 None。
    """
    session_id = "directive-resume"
    registry = SessionRegistry()
    server = build_server(registry)
    await _call(
        server, "zero.open_session", session_id=session_id, config={"motion_backend": "directive"}
    )
    try:
        session = await registry.get(session_id)
        assert session is not None
        await _call(
            server,
            "zero.step",
            session_id=session_id,
            stim={"valence": 0.5, "arousal": 0.6},
        )
        assert session.last_motion_directive is not None  # directive 模式下 step 后确实产出
    finally:
        await _call(server, "zero.close_session", session_id=session_id)

    # resume：同 session_id 重开，新对象的 last_motion_directive 未持久化 → 恒 None。
    await _call(
        server, "zero.open_session", session_id=session_id, config={"motion_backend": "directive"}
    )
    try:
        resumed = await registry.get(session_id)
        assert resumed is not None
        assert resumed.last_motion_directive is None
        result = _payload(
            await _call(server, "zero.motion", session_id=session_id, duration_ms=500)
        )
        assert len(result["keyframes"]) > 0  # 没抛异常，优雅退回现有解析路径
    finally:
        await _call(server, "zero.close_session", session_id=session_id)


@pytest.mark.asyncio
async def test_synth_default_matches_manual_generate_dual(enabled: None) -> None:
    """motion_backend 默认 synth ⇒ zero.motion 输出与手工调用
    `motion_synth.generate_dual(..., modulation=None)` 逐字相同——本次 directive 消费改动
    对默认门控**零可见影响**（不是"看起来差不多"，是逐帧比对）。
    """
    import zlib

    from src.agents.motion_synth import PhaseState, generate_dual, initial_blink_ms

    session_id = "synth-zero-regression"
    duration_ms = 1500
    registry = SessionRegistry()
    server = build_server(registry)
    await _call(server, "zero.open_session", session_id=session_id)
    try:
        result = _payload(
            await _call(server, "zero.motion", session_id=session_id, duration_ms=duration_ms)
        )
    finally:
        await _call(server, "zero.close_session", session_id=session_id)

    span = max(1, min(int(duration_ms), MOTION_MAX_SEGMENT_MS))
    fps = min(MOTION_DEFAULT_FPS, (MOTION_MAX_KEYFRAMES - 1) * 1000.0 / span)
    seed = zlib.crc32(session_id.encode("utf-8"))
    phase_in = PhaseState(noise_seed=seed, next_blink_ms=initial_blink_ms(seed))
    heads, phase_out = generate_dual(
        (0.0, 0.0), None, float(span), phase_in, voluntary_leak=1.0, fps=fps, modulation=None
    )
    assert result["keyframes"] == heads["voluntary"]
    assert result["spontaneous"] == heads["spontaneous"]
    assert result["phase_ms"] == phase_out.elapsed_ms


@pytest.mark.asyncio
async def test_directive_voluntary_matches_synth_end_to_end(enabled: None) -> None:
    """完整闭环回归（2026-08-07 修复验证）：真实 step 一轮（regulation_enabled=True +
    非默认 voluntary_coping_leak）后，directive 模式的 voluntary 轨迹须与「同一
    (affect, regulated, leak, seed) 手工现算 synth 路径」逐字相同——不是手工构造的
    合成 directive，是端到端跑出来的真实 affect_sample/regulated_affect。
    """
    import zlib

    from src.agents.motion_synth import PhaseState, generate_dual, initial_blink_ms

    session_id = "directive-vs-synth-e2e"
    duration_ms = 3000
    registry = SessionRegistry()
    server = build_server(registry)
    await _call(
        server,
        "zero.open_session",
        session_id=session_id,
        config={
            "motion_backend": "directive",
            "regulation_enabled": True,
            "voluntary_coping_leak": 0.6,
        },
    )
    try:
        await _call(
            server, "zero.step", session_id=session_id, stim={"valence": -0.6, "arousal": 0.8}
        )
        session = await registry.get(session_id)
        assert session is not None
        affect, regulated, leak = session.last_affect()
        assert regulated is not None and regulated != affect  # 调节确实生效，非平凡场景
        assert affect is not None  # 已 step 过，只为给 mypy 窄化 Optional
        directed = _payload(
            await _call(server, "zero.motion", session_id=session_id, duration_ms=duration_ms)
        )
    finally:
        await _call(server, "zero.close_session", session_id=session_id)

    span = min(duration_ms, MOTION_MAX_SEGMENT_MS)
    seed = zlib.crc32(session_id.encode("utf-8"))
    phase_in = PhaseState(noise_seed=seed, next_blink_ms=initial_blink_ms(seed))
    synth_heads, _ = generate_dual(
        affect, regulated, float(span), phase_in, voluntary_leak=leak, fps=MOTION_DEFAULT_FPS
    )
    assert directed["keyframes"] == synth_heads["voluntary"]  # 修复前这条会红


@pytest.mark.asyncio
async def test_directive_empty_events_falls_back_to_reply_text(enabled: None) -> None:
    """WARN-2 回落边界：directive 里 events 为空（language_text 为空的默认情形）时，
    zero.motion 必须回落到 reply_text 现场解析路径，不是静默丢弃 reply_text 传的行为意图。
    """
    session_id = "directive-empty-events-fallback"
    registry = SessionRegistry()
    server = build_server(registry)
    await _call(
        server, "zero.open_session", session_id=session_id, config={"motion_backend": "directive"}
    )
    try:
        session = await registry.get(session_id)
        assert session is not None
        # 模拟 MotionAgent 在 language_text 为空时的真实输出：events 为空列表。
        session.last_motion_directive = {
            "amplitude": 1.0,
            "speed": 1.0,
            "onset": 0.5,
            "regulated": None,
            "scene": "idle",
            "events": [],
            "prosody_ref": None,
        }
        result = _payload(
            await _call(
                server,
                "zero.motion",
                session_id=session_id,
                duration_ms=500,
                reply_text="（点了点头）嗯，是这样。",
            )
        )
        assert "nod" in [e["name"] for e in result["events"]]
    finally:
        await _call(server, "zero.close_session", session_id=session_id)


async def _open_step_motion(server, registry, session_id: str, backend: str):
    """open(指定 backend + 固定 rng_seed) → step → motion，返回 (payload, 副本快照)。"""
    await _call(
        server,
        "zero.open_session",
        session_id=session_id,
        config={"motion_backend": backend, "rng_seed": 42},
    )
    try:
        await _call(
            server, "zero.step", session_id=session_id, stim={"valence": 0.5, "arousal": 0.6}
        )
        payload = _payload(
            await _call(server, "zero.motion", session_id=session_id, duration_ms=1500)
        )
        session = await registry.get(session_id)
        assert session is not None
        return payload, session.last_motion_efference
    finally:
        await _call(server, "zero.close_session", session_id=session_id)


@pytest.mark.asyncio
async def test_efference_full_link_matches_directive_and_no_leak(enabled: None) -> None:
    """efference 档 MCP 全链路回归锁（code-reviewer WARN-2 2026-08-07）：
    `open_session(config={"motion_backend":"efference"}) → step → zero.motion` 的输出
    须与 "directive" 档**逐字一致**（副本纯增量，拉取侧判据不区分 backend 值），
    且 `motion_efference` 不得泄漏进任何 MCP 返回体——此前这两条只靠人工走读，无回归锁。

    同一 session_id 先后两开（轨迹种子 = crc32(session_id) 须相同）+ 固定 rng_seed
    （内核采样确定），两档输出才可逐字比对。
    """
    session_id = "efference-vs-directive-e2e"
    registry = SessionRegistry()
    server = build_server(registry)
    directive_payload, directive_copy = await _open_step_motion(
        server, registry, session_id, "directive"
    )
    assert directive_copy is None  # directive 档不写副本（零回归）
    efference_payload, efference_copy = await _open_step_motion(
        server, registry, session_id, "efference"
    )
    assert efference_payload == directive_payload  # 逐字一致，副本纯增量
    assert efference_copy is not None  # efference 档经 MCP 全链路确实写了副本
    assert "motion_efference" not in repr(efference_payload)  # 副本不泄漏进 MCP 返回体
