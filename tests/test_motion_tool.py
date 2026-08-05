"""`zero.motion` 工具面：门控、契约不变量、只读性、相位续接。

契约不变量按对面 `params_animate` 的拒收条件逐条钉——这些不是学术担忧，
违反了对面会当场 rejected。
"""

from __future__ import annotations

import re

import pytest

from src.mcp_server.registry import SessionRegistry
from src.mcp_server.server import (
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
