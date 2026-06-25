"""tools/reset_db：清空/删除本地库（架构测试用）的纯函数单测，全用临时库、不碰 data/。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tools.reset_db import delete_path, truncate_chat_thread, truncate_sqlite


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


def test_delete_path_and_missing_is_graceful(tmp_path: Path) -> None:
    f = tmp_path / "f.sqlite3"
    f.write_text("x")
    assert "已删除" in delete_path(f)
    assert not f.exists()
    assert "不存在" in delete_path(tmp_path / "nope.sqlite3")  # 不存在优雅跳过
