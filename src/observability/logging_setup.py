"""统一日志初始化：入口无关的可观测性设施。

任何启动入口（当前的**临时** `main.py`，或将来迁移后的正式主入口）只需在启动早期
调用一次 `setup_logging()`，即可让全项目 `logging.getLogger(__name__)` 的输出落到
**本次启动专属**的日志文件（每次启动一个新文件），并可选同时回显到控制台（stderr）。

设计要点（为「删除临时 main.py 后做主入口迁移」服务）：
- 配置逻辑全在本模块、**不依赖任何启动脚本或业务层** → 换入口零改动，import 本函数即可。
- 业务模块只管 `logging.getLogger(__name__)`，**不** import 本模块，故不引入跨层依赖。
- 幂等：重复调用不重复挂 handler（多入口 / 测试 / 被反复 import 都安全）。
- 控制台 handler 走 stderr，不污染靠 stdout 输出 JSON 的批处理模式（如 `--trace`）。

配置经环境变量（无则用合理默认，符合「运行配置走 env」）：
- ZERO_LOG_DIR           日志目录（默认 ``logs``）
- ZERO_LOG_LEVEL         文件与项目 logger 级别（默认 ``INFO``）
- ZERO_LOG_CONSOLE       是否同时输出到控制台（默认 ``1``；设 ``0`` 关闭）
- ZERO_LOG_CONSOLE_PLAIN 控制台是否用极简 message-only 格式（默认 ``1``，不污染交互 UI）
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

# 纯第三方库噪声：任何入口都不想要它们刷屏。项目自身 logger 的按需压制留给各入口处理。
NOISY_THIRD_PARTY = ("httpx", "httpcore", "openai", "urllib3", "sentence_transformers")

FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
CONSOLE_PLAIN_FORMAT = "%(message)s"

# 挂在本模块所建 handler 上的标记，用于幂等探测（区别于入口自带的其它 handler）。
HANDLER_FLAG = "_zero_observability_managed"


def _resolve_level(raw: str | None, default: int = logging.INFO) -> int:
    """把 env 里的级别名（``INFO`` / ``DEBUG`` …）解析为 logging 级别整数；无效则回退默认。"""
    if not raw:
        return default
    level = logging.getLevelName(raw.strip().upper())
    return level if isinstance(level, int) else default


def _managed_file_handler(root: logging.Logger) -> logging.FileHandler | None:
    """若 root 已挂过本模块建的 FileHandler，返回它（幂等复用）；否则 None。"""
    for handler in root.handlers:
        if getattr(handler, HANDLER_FLAG, False) and isinstance(handler, logging.FileHandler):
            return handler
    return None


def setup_logging(*, log_dir: str | None = None, level: int | None = None) -> Path:
    """初始化全局日志：每次启动新建一个日志文件，返回该文件路径。

    入口无关、可重复调用（幂等：同一进程内反复调只建一次文件、不重复挂 handler）。
    显式参数优先于环境变量，二者皆缺用默认。不同进程启动会得到不同文件名
    （时间戳 + pid），从而实现「每次启动一份新日志」。
    """
    root = logging.getLogger()

    existing = _managed_file_handler(root)
    if existing is not None:
        return Path(existing.baseFilename)

    resolved_level = level if level is not None else _resolve_level(os.getenv("ZERO_LOG_LEVEL"))
    directory = Path(log_dir or os.getenv("ZERO_LOG_DIR") or "logs")
    directory.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = directory / f"zero-{stamp}-{os.getpid()}.log"

    root.setLevel(resolved_level)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
    setattr(file_handler, HANDLER_FLAG, True)
    root.addHandler(file_handler)

    if (os.getenv("ZERO_LOG_CONSOLE") or "1") != "0":
        console = logging.StreamHandler()  # 默认 stderr，不抢 stdout
        console.setLevel(resolved_level)
        plain = (os.getenv("ZERO_LOG_CONSOLE_PLAIN") or "1") != "0"
        console.setFormatter(logging.Formatter(CONSOLE_PLAIN_FORMAT if plain else FILE_FORMAT))
        setattr(console, HANDLER_FLAG, True)
        root.addHandler(console)

    for noisy in NOISY_THIRD_PARTY:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "日志初始化完成 → %s（level=%s）",
        log_path,
        logging.getLevelName(resolved_level),
    )
    return log_path
