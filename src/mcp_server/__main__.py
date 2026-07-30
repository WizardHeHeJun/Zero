"""`python -m src.mcp_server` 入口：从 env 装配 server + 起传输。

与 client 占位对齐（边界回执 §三）：client 拉起子进程 = `command`(conda env python) +
`args=["-m","src.mcp_server"]` + `cwd=D:\\Zero`，env 传完整副本（保证子进程读到 ZERO_* 配置）。
传输默认 stdio（本地优先）；`ZERO_MCP_TRANSPORT=http`（或 streamable-http）走 Streamable HTTP。
"""

from __future__ import annotations

import logging
import os

from src.mcp_server.server import TRANSPORT_STREAMABLE_HTTP, build_server, resolve_transport


def main() -> None:
    """装配 server 并按 `ZERO_MCP_TRANSPORT` 起传输（stdio 默认 / streamable-http 备选）。

    🛑 传输解析**必须**走 `server.resolve_transport`，不得在此重抄一遍 env 判别：
    `zero.describe_config` 用同一符号回报「生效传输」，两处各写一份就等于造一个漂移源
    —— 回读面会宣称一个部署端根本没起的传输，而消费方拿它当真（同 R11 的失效型）。
    行为与抽取前逐字一致（默认 stdio · 大小写不敏感 · `http`/`streamable-http` 两别名都认 ·
    其余值回落 stdio），同源与行为两条回归锁在 `tests/test_mcp_server.py`
    （`test_transport_resolution_is_single_sourced` /
    `test_resolve_transport_matches_pre_extraction_expression`）。
    """
    # 🛑 走 `setup_logging()` 落**文件**，不是只 `basicConfig` 进 stderr（2026-07-30，兑现 R10）。
    # 为什么必须落文件：stdio 传输下本进程是**配套项目拉起的子进程**，stderr 全灌进它的
    # `.stderr` 字段 ⇒ **我方本地零留痕**。而跨仓反复在争「某个数是在哪套门控下产生的」，
    # 事后却没有任何一侧的产物能裁定 —— 观察期的可裁定性正卡在这里。
    # 与 chat 面（`main.py`）/ CLI（`scripts/cli_modes.py`）同源，不另造一套。
    # ⚠ 失败不得挡启动：日志目录不可写（只读容器 / 权限）时回落 `basicConfig`，
    # 宁可少一份文件也不能让 server 起不来。
    try:
        from src.observability import setup_logging

        log_path = setup_logging()
        logging.getLogger(__name__).info("mcp_server 日志文件：%s", log_path)
    except Exception:
        # 🛑 这里**必须真调** `basicConfig`：上面那句注释与下面那条 warning
        # 都宣称「回落 basicConfig」，
        # 而 `setup_logging` 失败时 root logger 是**未配置**状态（默认 WARNING、无 handler）
        # ⇒ 不调它的话，warning 靠 lastResort 勉强出 stderr，而**所有 INFO 全丢**——
        # 包括 `build_server.open_session` 刚加的那条门控快照，即 R10 想要的可裁定性
        # 恰在最该生效的场景（只读容器 / 目录不可写）失效，且相对改动前是**回退**
        # （改前恒 `basicConfig(level=ZERO_LOG_LEVEL)`）。
        # 回归锁：`tests/test_mcp_server.py::test_setup_logging_failure_falls_back_to_basicconfig`。
        logging.basicConfig(level=os.getenv("ZERO_LOG_LEVEL", "INFO"))
        logging.getLogger(__name__).warning(
            "setup_logging 失败，已回落 basicConfig（仅 stderr）；"
            "stdio 传输下这意味着我方本地无留痕",
            exc_info=True,
        )
    server = build_server()
    if resolve_transport() == TRANSPORT_STREAMABLE_HTTP:
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
