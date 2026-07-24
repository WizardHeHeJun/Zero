"""zero-link MCP HTTP 传输的 Bearer 鉴权（边界层·纯 ASGI + 标准库·不依赖内核）。

streamable-http 走 SSE（`EventSourceResponse`·keep-alive 长连接），故用**纯 ASGI 中间件**
（非 Starlette `BaseHTTPMiddleware`——后者缓冲响应体会破坏 SSE 流）。校验
`Authorization: Bearer <token>`，缺/错回 401 结构化错误（镜像 mcp SDK
`RequireAuthMiddleware._send_auth_error` 形状）。

门策略（`resolve_enforced_token`）：设了 `ZERO_MCP_HTTP_TOKEN` 即强制（即便 loopback）；未设 +
loopback 免鉴权（本机·零回归）；未设 + 非 loopback（如 0.0.0.0 对外）启动 fail-fast，不静默开无
鉴权裸端口。secrets 走 env（不硬编码），比对用 `hmac.compare_digest` 常量时间防计时旁路。
"""

from __future__ import annotations

import hmac
import ipaddress
import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

# ASGI 类型别名（不引 starlette；纯 ASGI callable 三元签名）。scope/message 用 MutableMapping
# 对齐 ASGI 规范与 Starlette.__call__，使 wrap_with_auth 可直接收 streamable_http_app() 的返回。
Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

_LOOPBACK_NAMES = frozenset({"localhost"})


def _is_loopback(host: str) -> bool:
    """host 是否 loopback（127.0.0.0/8、::1、localhost）；未知主机名 → False（保守要鉴权）。"""
    h = host.strip().lower()
    if h in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def resolve_enforced_token(host: str, token: str | None) -> str | None:
    """决定是否强制鉴权 + 用哪个 token（可测纯函数）。

    - token 非空（strip 后非纯空白）→ 返回它（强制·即便 loopback，显式配置即启用）。须纯 ASCII——
      非 ASCII 密钥经 HTTP 头 latin-1/UTF-8 编码歧义会恒 401，故此处 fail-fast 而非静默拒。
    - token 空/None/纯空白 + loopback host → None（免鉴权·本机零回归）。
    - token 空/None/纯空白 + 非 loopback host → fail-fast `ValueError`（拒绝对外开无鉴权裸端口）。
    """
    stripped = token.strip() if token else ""
    if stripped:
        if not stripped.isascii():
            raise ValueError(
                "ZERO_MCP_HTTP_TOKEN 须为纯 ASCII（hex/UUID/base64 等）；"
                "非 ASCII 密钥经 HTTP 头编码歧义会恒 401"
            )
        return stripped
    if _is_loopback(host):
        return None
    raise ValueError(
        f"ZERO_MCP_HTTP_HOST={host!r} 非 loopback 却未设 ZERO_MCP_HTTP_TOKEN——"
        "拒绝对外开无鉴权裸端口；请设 token 或绑回 127.0.0.1"
    )


def _bearer_from_scope(scope: Scope) -> str | None:
    """从 ASGI scope 取首个 `Authorization: Bearer <token>` 的 token；无 Bearer 头 → None。

    多个 authorization 头时**跳过**非 Bearer 头继续扫（RFC 7235 §4.1 不禁多头）——防代理/网关
    插入的非 Bearer 头（如 `Basic …`）挡掉后续正确的 Bearer 头致恒 401。
    """
    for name, value in scope.get("headers", []):
        if name.lower() == b"authorization":
            raw = value.decode("latin-1")
            if raw[:7].lower() == "bearer ":
                return raw[7:]
    return None


async def _send_401(send: Send) -> None:
    """回 401 结构化错误（镜像 mcp SDK 形状：JSON body + `WWW-Authenticate: Bearer`）。"""
    body = json.dumps(
        {"error": "invalid_token", "error_description": "缺少或无效的 Bearer token"}
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (
                    b"www-authenticate",
                    b'Bearer error="invalid_token", error_description="authentication required"',
                ),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


class BearerAuthMiddleware:
    """纯 ASGI 中间件：对 http 请求强制 `Authorization: Bearer <token>`；缺/错回 401。

    非 http scope（lifespan/websocket）直接放行——否则 uvicorn 的 lifespan 握手会被拦。
    比对常量时间（`hmac.compare_digest`），避免计时旁路。
    """

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.token_bytes = token.encode("utf-8")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        provided = _bearer_from_scope(scope)
        if provided is None or not hmac.compare_digest(provided.encode("utf-8"), self.token_bytes):
            await _send_401(send)
            return
        await self.app(scope, receive, send)


def wrap_with_auth(app: ASGIApp, host: str, token: str | None) -> ASGIApp:
    """按门策略包裹 ASGI app：需鉴权 → `BearerAuthMiddleware`；免鉴权 → 原 app。

    非 loopback 且无 token → `resolve_enforced_token` 抛 `ValueError`（启动期 fail-fast）。
    """
    enforced = resolve_enforced_token(host, token)
    return app if enforced is None else BearerAuthMiddleware(app, enforced)
