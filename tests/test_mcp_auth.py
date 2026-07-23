"""zero-link MCP HTTP Bearer 鉴权（T5）单测：门策略纯函数 + 纯 ASGI 中间件 + 真栈 e2e。

auth.py 只依赖标准库 → 纯函数/中间件测试无需 mcp（不 importorskip）。中间件行为用
httpx.ASGITransport 直接驱动（无需 uvicorn/lifespan）；真栈 401/200 用 uvicorn 起 wrapped
真 MCP app（镜像 test_mcp_server.test_http_transport_roundtrip 的 socket-prebind）。
缺 httpx/uvicorn/mcp 时 importorskip 优雅跳过（CI 轻量环境）。
"""

from __future__ import annotations

import pytest

from src.mcp_server.auth import (
    BearerAuthMiddleware,
    _bearer_from_scope,
    _is_loopback,
    resolve_enforced_token,
    wrap_with_auth,
)


async def _noop_app(scope, receive, send) -> None:  # 假下游（wrap_with_auth 只查身份/包裹，不调）
    return None


# ── 门策略纯函数 ─────────────────────────────────────────────────────────


def test_is_loopback() -> None:
    assert _is_loopback("127.0.0.1") is True
    assert _is_loopback("::1") is True
    assert _is_loopback("localhost") is True
    assert _is_loopback("127.0.0.5") is True  # 整个 127.0.0.0/8
    assert _is_loopback("0.0.0.0") is False
    assert _is_loopback("192.168.1.9") is False
    assert _is_loopback("example.com") is False  # 未知主机名 → 保守非 loopback


def test_resolve_enforced_token() -> None:
    # token 非空 → 强制（即便 loopback）
    assert resolve_enforced_token("127.0.0.1", "secret") == "secret"
    assert resolve_enforced_token("0.0.0.0", "secret") == "secret"
    # token 空/None + loopback → None（免鉴权）
    assert resolve_enforced_token("127.0.0.1", None) is None
    assert resolve_enforced_token("localhost", "") is None
    # token 空 + 非 loopback → fail-fast 指向 env
    with pytest.raises(ValueError, match="ZERO_MCP_HTTP_TOKEN"):
        resolve_enforced_token("0.0.0.0", None)
    # W3：纯空白 token 视同未设（strip 后）；前后空白被 strip
    assert resolve_enforced_token("127.0.0.1", "   ") is None
    with pytest.raises(ValueError, match="ZERO_MCP_HTTP_TOKEN"):
        resolve_enforced_token("0.0.0.0", "   ")
    assert resolve_enforced_token("127.0.0.1", "  secret  ") == "secret"
    # W2：非 ASCII token → fail-fast（HTTP 头编码歧义会恒 401）
    with pytest.raises(ValueError, match="ASCII"):
        resolve_enforced_token("127.0.0.1", "tökén")


def test_wrap_with_auth() -> None:
    # loopback 无 token → 原 app 对象（免鉴权·零回归）
    assert wrap_with_auth(_noop_app, "127.0.0.1", None) is _noop_app
    # 有 token → BearerAuthMiddleware 包裹
    assert isinstance(wrap_with_auth(_noop_app, "127.0.0.1", "secret"), BearerAuthMiddleware)
    # 非 loopback 无 token → fail-fast
    with pytest.raises(ValueError):
        wrap_with_auth(_noop_app, "0.0.0.0", None)


def test_bearer_from_scope() -> None:
    def scope(headers: list[tuple[bytes, bytes]]) -> dict[str, object]:
        return {"type": "http", "headers": headers}

    assert _bearer_from_scope(scope([(b"authorization", b"Bearer abc")])) == "abc"
    assert _bearer_from_scope(scope([(b"Authorization", b"bearer abc")])) == "abc"  # 大小写不敏感
    assert _bearer_from_scope(scope([(b"authorization", b"Basic abc")])) is None
    assert _bearer_from_scope(scope([])) is None
    # W1：多 authorization 头时跳过非 Bearer（Basic）继续扫到 Bearer
    two = scope([(b"authorization", b"Basic x"), (b"authorization", b"Bearer abc")])
    assert _bearer_from_scope(two) == "abc"


# ── 中间件行为（httpx.ASGITransport·无需 uvicorn/lifespan）──────────────────


async def test_middleware_correct_token_passes_through() -> None:
    httpx = pytest.importorskip("httpx")
    hits = {"n": 0}

    async def downstream(scope, receive, send) -> None:
        hits["n"] += 1
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw = BearerAuthMiddleware(downstream, "secret")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mw), base_url="http://t") as hc:
        r = await hc.post("/mcp", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200 and r.text == "ok" and hits["n"] == 1


async def test_middleware_missing_token_401() -> None:
    httpx = pytest.importorskip("httpx")
    hits = {"n": 0}

    async def downstream(scope, receive, send) -> None:
        hits["n"] += 1

    mw = BearerAuthMiddleware(downstream, "secret")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mw), base_url="http://t") as hc:
        r = await hc.post("/mcp")  # 无 Authorization
    assert r.status_code == 401
    assert r.json()["error"] == "invalid_token"
    assert "www-authenticate" in {k.lower() for k in r.headers}
    assert hits["n"] == 0  # 未触下游


async def test_middleware_wrong_token_401() -> None:
    httpx = pytest.importorskip("httpx")

    async def downstream(scope, receive, send) -> None:
        raise AssertionError("错 token 不应到达下游")

    mw = BearerAuthMiddleware(downstream, "secret")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mw), base_url="http://t") as hc:
        r = await hc.post("/mcp", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


# ── 真栈 e2e（uvicorn 起 wrapped 真 MCP app）──────────────────────────────


async def _serve(app, sock):
    """在 ephemeral 端口起 uvicorn 服务 app；返回 (server, task)，调用方负责收尾。"""
    import asyncio

    import uvicorn

    uv = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    task = asyncio.create_task(uv.serve(sockets=[sock]))
    for _ in range(100):
        if uv.started:
            break
        await asyncio.sleep(0.05)
    assert uv.started, "uvicorn 未在 5s 内启动"
    return uv, task


async def _shutdown(uv, task) -> None:
    import asyncio

    uv.should_exit = True
    try:
        await asyncio.wait_for(task, timeout=5)
    except TimeoutError:
        task.cancel()


async def test_http_server_rejects_without_token_401() -> None:
    """真栈：wrapped 真 MCP app 经 uvicorn 起，raw POST /mcp 无 token → 401。"""
    pytest.importorskip("mcp")
    pytest.importorskip("uvicorn")
    httpx = pytest.importorskip("httpx")
    import socket

    from src.mcp_server.server import build_server

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    app = wrap_with_auth(build_server().streamable_http_app(), "127.0.0.1", "secret")
    uv, task = await _serve(app, sock)
    try:
        async with httpx.AsyncClient() as hc:
            r = await hc.post(
                f"http://127.0.0.1:{port}/mcp",
                json={"jsonrpc": "2.0", "method": "x", "id": 1},
            )
        assert r.status_code == 401
        assert r.json()["error"] == "invalid_token"
    finally:
        await _shutdown(uv, task)


async def test_http_server_accepts_correct_token_roundtrip() -> None:
    """真栈：带 Bearer 的 client 全链路 open→step→close 走通（鉴权不破坏 SSE）。"""
    pytest.importorskip("mcp")
    pytest.importorskip("uvicorn")
    httpx = pytest.importorskip("httpx")
    import json
    import socket

    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    from src.mcp_server.server import build_server

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    app = wrap_with_auth(build_server().streamable_http_app(), "127.0.0.1", "secret")
    uv, task = await _serve(app, sock)
    try:
        async with httpx.AsyncClient(headers={"Authorization": "Bearer secret"}) as authed:
            async with streamable_http_client(
                f"http://127.0.0.1:{port}/mcp", http_client=authed
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    sid = json.loads(
                        (await session.call_tool("zero.open_session", {})).content[0].text
                    )["session_id"]
                    r = await session.call_tool(
                        "zero.step",
                        {"session_id": sid, "stim": {"valence": 0.4, "arousal": 0.5}},
                    )
                    assert r.isError is False
                    c = await session.call_tool("zero.close_session", {"session_id": sid})
                    assert json.loads(c.content[0].text) == {"ok": True}
    finally:
        await _shutdown(uv, task)
