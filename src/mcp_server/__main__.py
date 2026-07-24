"""`python -m src.mcp_server` 入口：从 env 装配 server + 起传输。

与 client 占位对齐（边界回执 §三）：client 拉起子进程 = `command`(conda env python) +
`args=["-m","src.mcp_server"]` + `cwd=D:\\Zero`，env 传完整副本（保证子进程读到 ZERO_* 配置）。
传输默认 stdio（本地优先）；`ZERO_MCP_TRANSPORT=http`（或 streamable-http）走 Streamable HTTP。
"""

from __future__ import annotations

import logging
import os

from src.mcp_server.server import build_server


def main() -> None:
    """装配 server 并按 `ZERO_MCP_TRANSPORT` 起传输（stdio 默认 / streamable-http 备选）。"""
    logging.basicConfig(level=os.getenv("ZERO_LOG_LEVEL", "INFO"))
    server = build_server()
    transport = os.getenv("ZERO_MCP_TRANSPORT", "stdio").lower()
    if transport in ("http", "streamable-http"):
        import uvicorn

        from src.mcp_server.auth import wrap_with_auth

        # host/port 复用 build_server 已校验值（非法已 fail-fast）；wrap_with_auth 按
        # ZERO_MCP_HTTP_TOKEN + host 决定强制/免鉴权/对外无 token 启动 fail-fast（见 auth.py）。
        app = wrap_with_auth(
            server.streamable_http_app(),
            server.settings.host,
            os.getenv("ZERO_MCP_HTTP_TOKEN"),
        )
        uvicorn.run(
            app,
            host=server.settings.host,
            port=server.settings.port,
            log_level=os.getenv("ZERO_LOG_LEVEL", "info").lower(),
        )
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
