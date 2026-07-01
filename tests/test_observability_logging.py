"""src/observability 统一日志设施单测：落盘、幂等、env 覆盖，及「入口/层无关」约束守护。

入口无关是本设施的设计目的（main.py 是临时入口，将来会删并迁移正式主入口）——
其中 test_logging_setup_is_entrypoint_and_layer_agnostic 把该约束钉成回归断言。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.observability import setup_conversation_log, setup_logging
from src.observability.logging_setup import (
    CONVERSATION_LOGGER_NAME,
    HANDLER_FLAG,
    _resolve_level,
)


@pytest.fixture
def isolated_root() -> Iterator[logging.Logger]:
    """给每个测试一个干净的 root logger；结束后关闭本测试建的 handler 并还原原状态。

    setup_logging 改的是全局 root，必须隔离，否则会污染其它测试；FileHandler 还持有
    打开的文件句柄，Windows 下不 close 会锁住 tmp_path 导致清理失败。
    """
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    root.handlers.clear()
    try:
        yield root
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers.clear()
        root.handlers.extend(saved_handlers)
        root.setLevel(saved_level)


def test_creates_new_log_file(
    isolated_root: logging.Logger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """启动即在目标目录建出一个 .log 文件，并返回其路径。"""
    monkeypatch.setenv("ZERO_LOG_CONSOLE", "0")
    path = setup_logging(log_dir=str(tmp_path))
    assert path.exists()
    assert path.parent == tmp_path
    assert path.suffix == ".log"


def test_messages_land_in_file(
    isolated_root: logging.Logger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """初始化后，任意模块 logger 的输出都落进本次启动的日志文件。"""
    monkeypatch.setenv("ZERO_LOG_CONSOLE", "0")
    path = setup_logging(log_dir=str(tmp_path))
    logging.getLogger("zero.test").info("hello-埋点")
    for handler in isolated_root.handlers:
        handler.flush()
    assert "hello-埋点" in path.read_text(encoding="utf-8")


def test_idempotent_same_process(
    isolated_root: logging.Logger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同进程内重复调用：复用同一文件、不重复挂 FileHandler（多入口/被反复 import 安全）。"""
    monkeypatch.setenv("ZERO_LOG_CONSOLE", "0")
    first = setup_logging(log_dir=str(tmp_path))
    second = setup_logging(log_dir=str(tmp_path))
    assert first == second
    managed_file_handlers = [
        h
        for h in isolated_root.handlers
        if getattr(h, HANDLER_FLAG, False) and isinstance(h, logging.FileHandler)
    ]
    assert len(managed_file_handlers) == 1


def test_env_overrides(
    isolated_root: logging.Logger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ZERO_LOG_DIR / ZERO_LOG_LEVEL 生效（配置走 env，代码不写死）。"""
    monkeypatch.setenv("ZERO_LOG_CONSOLE", "0")
    monkeypatch.setenv("ZERO_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("ZERO_LOG_LEVEL", "DEBUG")
    path = setup_logging()
    assert path.parent == tmp_path
    assert isolated_root.level == logging.DEBUG


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("DEBUG", logging.DEBUG),
        ("info", logging.INFO),
        ("WARNING", logging.WARNING),
        (None, logging.INFO),
        ("", logging.INFO),
        ("not-a-level", logging.INFO),
    ],
)
def test_resolve_level(raw: str | None, expected: int) -> None:
    """级别名解析：合法名→对应级别；空/无效→回退 INFO（不因配置笔误而崩）。"""
    assert _resolve_level(raw) == expected


def test_logging_setup_is_entrypoint_and_layer_agnostic() -> None:
    """核心约束回归：日志设施不依赖临时入口 main、不 import 任何业务层。

    这样删掉 main.py 做主入口迁移时，本设施毫不牵连，新入口 import setup_logging 即用。
    """
    source = (
        Path(__file__).resolve().parents[1] / "src" / "observability" / "logging_setup.py"
    ).read_text(encoding="utf-8")
    assert "import main" not in source
    for layer in ("src.agents", "src.orchestration", "src.memory", "src.storage"):
        assert layer not in source, f"日志设施不应 import 业务层 {layer}"


@pytest.fixture
def isolated_conversation_logger() -> Iterator[logging.Logger]:
    """隔离 ``zero.conversation`` logger：保存/还原 handlers·level·propagate。

    setup_conversation_log 改的是这个具名 logger 的全局状态（挂 handler、置 propagate=False），
    必须隔离否则串到其它测试；FileHandler 还持有打开的文件句柄，Windows 下不 close 会锁 tmp_path。
    """
    conv = logging.getLogger(CONVERSATION_LOGGER_NAME)
    saved_handlers = conv.handlers[:]
    saved_level = conv.level
    saved_propagate = conv.propagate
    conv.handlers.clear()
    try:
        yield conv
    finally:
        for handler in conv.handlers:
            handler.close()
        conv.handlers.clear()
        conv.handlers.extend(saved_handlers)
        conv.setLevel(saved_level)
        conv.propagate = saved_propagate


def test_conversation_log_on_writes_content(
    isolated_conversation_logger: logging.Logger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认开：建出含 conversation 的 .log，logger 内容落文件、propagate 关（不灌主日志）。"""
    monkeypatch.delenv("ZERO_CONVERSATION_LOG", raising=False)
    path = setup_conversation_log(log_dir=str(tmp_path))
    assert path is not None
    assert path.parent == tmp_path
    assert "conversation" in path.name and path.suffix == ".log"
    assert isolated_conversation_logger.propagate is False
    isolated_conversation_logger.info("第1轮\n  你 > 你好-埋点")
    for handler in isolated_conversation_logger.handlers:
        handler.flush()
    assert "你好-埋点" in path.read_text(encoding="utf-8")


def test_conversation_log_off_returns_none_and_suppresses(
    isolated_conversation_logger: logging.Logger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ZERO_CONVERSATION_LOG=0：返回 None、无 FileHandler（内容彻底不落日志文件）。"""
    monkeypatch.setenv("ZERO_CONVERSATION_LOG", "0")
    path = setup_conversation_log(log_dir=str(tmp_path))
    assert path is None
    file_handlers = [
        h for h in isolated_conversation_logger.handlers if isinstance(h, logging.FileHandler)
    ]
    assert not file_handlers
    assert not list(tmp_path.glob("conversation-*.log"))


def test_conversation_log_idempotent(
    isolated_conversation_logger: logging.Logger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同进程重复调：复用同一文件、只挂一个 FileHandler。"""
    monkeypatch.delenv("ZERO_CONVERSATION_LOG", raising=False)
    first = setup_conversation_log(log_dir=str(tmp_path))
    second = setup_conversation_log(log_dir=str(tmp_path))
    assert first == second
    managed_file_handlers = [
        h
        for h in isolated_conversation_logger.handlers
        if getattr(h, HANDLER_FLAG, False) and isinstance(h, logging.FileHandler)
    ]
    assert len(managed_file_handlers) == 1
