"""tools/reset_db：清空/删除本地库（架构测试用）的纯函数单测，全用临时库、不碰 data/。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tools.reset_db import (
    SQLITE_SIDECARS,
    _remove_sidecars,
    delete_db_files,
    truncate_chat_thread,
    truncate_sqlite,
)


def test_truncate_sqlite_empties_tables_keeps_file(tmp_path: Path) -> None:
    db = tmp_path / "x.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t (a TEXT)")
    conn.execute("INSERT INTO t VALUES ('hi')")
    conn.commit()
    conn.close()
    assert "清空" in truncate_sqlite(db)
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT count(*) FROM t").fetchone()[0] == 0  # 内容清空
    conn.close()
    assert db.exists()  # 库文件与表结构保留


def test_truncate_sqlite_wal_mode_cleans_sidecars(tmp_path: Path) -> None:
    """WAL 模式库 truncate 后不留 -wal/-shm 残留（这是新版「清得干净」的核心）。"""
    db = tmp_path / "wal.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (a TEXT)")
    conn.execute("INSERT INTO t VALUES ('hi')")
    conn.commit()
    conn.close()
    truncate_sqlite(db)
    assert not (tmp_path / "wal.sqlite3-wal").exists()  # WAL 边车清干净
    assert not (tmp_path / "wal.sqlite3-shm").exists()
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT count(*) FROM t").fetchone()[0] == 0
    conn.close()


def test_truncate_chat_thread_only_target(tmp_path: Path) -> None:
    db = tmp_path / "chat.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE turns (thread TEXT, ts TEXT, role TEXT, content TEXT)")
    conn.execute("CREATE TABLE meta (thread TEXT PRIMARY KEY, feeling_v REAL, feeling_a REAL)")
    conn.executemany(
        "INSERT INTO turns VALUES (?,?,?,?)",
        [("a", "t", "user", "x"), ("a", "t", "assistant", "y"), ("b", "t", "user", "z")],
    )
    conn.execute("INSERT INTO meta VALUES ('a', 0.1, 0.2)")
    conn.commit()
    conn.close()
    truncate_chat_thread(db, "a")
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT count(*) FROM turns WHERE thread='a'").fetchone()[0] == 0
    assert (
        conn.execute("SELECT count(*) FROM turns WHERE thread='b'").fetchone()[0] == 1
    )  # 别的会话保留
    assert conn.execute("SELECT count(*) FROM meta WHERE thread='a'").fetchone()[0] == 0
    conn.close()


def test_delete_db_files_removes_main_and_sidecars(tmp_path: Path) -> None:
    """整库删除连 -wal/-shm 一并删（旧脚本只删主库、留孤儿 WAL，正是「清不干净」的根因）。"""
    db = tmp_path / "x.sqlite3"
    db.write_text("main")
    wal = tmp_path / "x.sqlite3-wal"
    shm = tmp_path / "x.sqlite3-shm"
    wal.write_text("wal")
    shm.write_text("shm")
    msg = delete_db_files(db, SQLITE_SIDECARS)
    assert "已删除" in msg
    assert not db.exists()
    assert not wal.exists()  # 边车不留孤儿
    assert not shm.exists()


def test_delete_db_files_missing_is_graceful(tmp_path: Path) -> None:
    assert "不存在" in delete_db_files(tmp_path / "nope.sqlite3", SQLITE_SIDECARS)  # 优雅跳过


def test_remove_sidecars_best_effort(tmp_path: Path) -> None:
    """边车清理返回 (已删, 锁定)；正常情况全删、无锁定。"""
    db = tmp_path / "x.sqlite3"
    db.write_text("m")
    (tmp_path / "x.sqlite3-wal").write_text("w")
    (tmp_path / "x.sqlite3-shm").write_text("s")
    removed, locked = _remove_sidecars(db, SQLITE_SIDECARS)
    assert set(removed) == {"-wal", "-shm"}
    assert locked == []
    assert not (tmp_path / "x.sqlite3-wal").exists()
    assert not (tmp_path / "x.sqlite3-shm").exists()
