"""ConversationLog（从临时入口 main.py 迁出到存储层）单测：落盘 / 正序 / 态度持久化 / 隔离。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.storage.conversation_log import ConversationLog


def test_append_and_recent_chronological(tmp_path: Path) -> None:
    """append 的消息按时间正序取回（喂 LLM 的对话历史顺序）。"""
    log = ConversationLog(str(tmp_path / "c.sqlite3"))
    log.append("t", "user", "hi")
    log.append("t", "assistant", "yo")
    rows = log.recent("t", 10)
    assert [r["content"] for r in rows] == ["hi", "yo"]
    assert [r["role"] for r in rows] == ["user", "assistant"]
    log.close()


def test_recent_limit_keeps_latest(tmp_path: Path) -> None:
    """limit 只取最近 N 条，且仍按正序返回。"""
    log = ConversationLog(str(tmp_path / "c.sqlite3"))
    for i in range(5):
        log.append("t", "user", f"m{i}")
    assert [r["content"] for r in log.recent("t", 2)] == ["m3", "m4"]
    log.close()


def test_save_load_feeling_roundtrip(tmp_path: Path) -> None:
    """态度落盘 + 读回 + upsert 覆盖；无记录则平静起步 (0,0)。"""
    log = ConversationLog(str(tmp_path / "c.sqlite3"))
    assert log.load_feeling("t") == (0.0, 0.0)
    log.save_feeling("t", (0.3, -0.2))
    assert log.load_feeling("t") == pytest.approx((0.3, -0.2))
    log.save_feeling("t", (0.9, 0.1))
    assert log.load_feeling("t") == pytest.approx((0.9, 0.1))
    log.close()


def test_threads_isolated(tmp_path: Path) -> None:
    """不同 thread 的 transcript 与态度互不串味。"""
    log = ConversationLog(str(tmp_path / "c.sqlite3"))
    log.append("a", "user", "xa")
    log.append("b", "user", "xb")
    log.save_feeling("a", (0.5, 0.5))
    assert [r["content"] for r in log.recent("a", 10)] == ["xa"]
    assert [r["content"] for r in log.recent("b", 10)] == ["xb"]
    assert log.load_feeling("a") == pytest.approx((0.5, 0.5))
    assert log.load_feeling("b") == (0.0, 0.0)
    log.close()


def test_persists_across_reopen(tmp_path: Path) -> None:
    """重开库 = 模拟重启：transcript 与态度都续上（跨重启记忆）。"""
    path = str(tmp_path / "c.sqlite3")
    log = ConversationLog(path)
    log.append("t", "user", "remember")
    log.save_feeling("t", (0.4, 0.4))
    log.close()
    reopened = ConversationLog(path)
    assert [r["content"] for r in reopened.recent("t", 10)] == ["remember"]
    assert reopened.load_feeling("t") == pytest.approx((0.4, 0.4))
    reopened.close()
