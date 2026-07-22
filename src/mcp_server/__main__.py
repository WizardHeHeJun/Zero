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
        server.run(transport="streamable-http")
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
