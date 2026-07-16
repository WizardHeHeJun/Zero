"""zero-link MCP server：把 Zero 情感引擎的 `ConversationSession` 包成 MCP 三工具。

边界适配层（三层架构之外）：不反向依赖内核内部、不进 affect 热路径、默认不入 core 依赖
（`mcp` 走 optional-deps 组）。对齐 MCP 侧 client 契约（open/step/close + 会话身份）。
入口 `python -m src.mcp_server`（见 `__main__`）。
"""

from __future__ import annotations

from src.mcp_server.registry import SessionRegistry
from src.mcp_server.server import build_server

__all__ = ["SessionRegistry", "build_server"]
