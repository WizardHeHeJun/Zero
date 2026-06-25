"""清空本地数据库内容（架构测试用）。

系统在 `data/` 下落盘的本地库会随测试累积（对话历史会污染 LLM 回合、态度/价值跨重启续上），
本脚本一键清空，回到干净状态。覆盖：

| key | 库 | 内容 |
| --- | --- | --- |
| `chat` | `data/chat_history.sqlite3` | ConversationLog：对话 transcript + 累积态度 |
| `graph` | `ZERO_GRAPH_DB`(默认 `data/graph.sqlite3`) | 长期情绪倾向图谱（时序失效） |
| `semantic` | `ZERO_SEMANTIC_DB`(默认 `data/semantic.sqlite3`) | 语义记忆 episode + 向量 |
| `checkpoints` | `ZERO_CHECKPOINT_DB`(默认 `data/checkpoints.sqlite3`) | 运行态 Checkpointer |
| `kuzu` | `ZERO_KUZU_PATH`(默认 `data/graphiti.kuzu`) | Graphiti kuzu 图库（整体删除） |

默认 **truncate**（清空各表、保留库文件，下次运行直接复用）；`--delete-files` 整库删除
（下次运行各 store 的 `CREATE TABLE IF NOT EXISTS` 自动重建）。不碰容器后端（Postgres/Neo4j）——
那是部署态，本地测试用不到，需要时直接重建容器卷。

用法：
  python -m tools.reset_db                  # 清空全部本地库（确认后）
  python -m tools.reset_db --yes            # 跳过确认
  python -m tools.reset_db --chat           # 只清对话库
  python -m tools.reset_db --chat --thread chat   # 只清对话库里 thread=chat 的记录（保留别的会话）
  python -m tools.reset_db --delete-files --yes   # 整库删除而非清表
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
from pathlib import Path

# 仓库根（tools/ 的父目录）——使脚本与运行目录无关
ROOT = Path(__file__).resolve().parent.parent

# 各本地库的默认路径（与运行时同款 env，缺省即默认）
SQLITE_DBS: dict[str, str] = {
    "chat": "data/chat_history.sqlite3",
    "graph": os.getenv("ZERO_GRAPH_DB", "data/graph.sqlite3"),
    "semantic": os.getenv("ZERO_SEMANTIC_DB", "data/semantic.sqlite3"),
    "checkpoints": os.getenv("ZERO_CHECKPOINT_DB", "data/checkpoints.sqlite3"),
}
KUZU_PATH: str = os.getenv("ZERO_KUZU_PATH", "data/graphiti.kuzu")


def _resolve(path: str) -> Path:
    """相对路径按仓库根解析；绝对路径原样。"""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def truncate_sqlite(path: Path) -> str:
    """清空一个 SQLite 库的所有用户表（保留库文件与表结构）。"""
    if not path.exists():
        return "不存在，跳过"
    conn = sqlite3.connect(path)
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            conn.execute(f'DELETE FROM "{table}"')
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()
    return f"已清空 {len(tables)} 张表"


def truncate_chat_thread(path: Path, thread: str) -> str:
    """只清对话库里某个 thread 的记录（turns + meta），保留其它会话。"""
    if not path.exists():
        return "不存在，跳过"
    conn = sqlite3.connect(path)
    try:
        removed = 0
        for table in ("turns", "meta"):
            try:
                cur = conn.execute(f"DELETE FROM {table} WHERE thread = ?", (thread,))
                removed += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            except sqlite3.OperationalError:
                pass  # 表不存在则跳过
        conn.commit()
    finally:
        conn.close()
    return f"已删 thread={thread} 的 {removed} 行"


def delete_path(path: Path) -> str:
    """整体删除文件或目录（kuzu 为目录/文件，sqlite --delete-files 时用）。"""
    if not path.exists():
        return "不存在，跳过"
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return "已删除"


def _selected_keys(args: argparse.Namespace) -> list[str]:
    """按 flag 选目标；未指定任一具体库则选全部。"""
    explicit = [k for k in ("chat", "graph", "semantic", "checkpoints", "kuzu") if getattr(args, k)]
    return explicit or ["chat", "graph", "semantic", "checkpoints", "kuzu"]


def main() -> None:
    parser = argparse.ArgumentParser(description="清空本地数据库内容（架构测试用）")
    parser.add_argument("--chat", action="store_true", help="对话库 chat_history")
    parser.add_argument("--graph", action="store_true", help="长期倾向图谱 graph")
    parser.add_argument("--semantic", action="store_true", help="语义记忆 semantic")
    parser.add_argument("--checkpoints", action="store_true", help="运行态 checkpointer")
    parser.add_argument("--kuzu", action="store_true", help="Graphiti kuzu 图库（整体删除）")
    parser.add_argument("--thread", help="只清对话库里该 thread 的记录（仅对 chat 生效）")
    parser.add_argument(
        "--delete-files", action="store_true", help="整库删除而非清表（下次运行自动重建）"
    )
    parser.add_argument("--yes", action="store_true", help="跳过确认")
    args = parser.parse_args()

    keys = _selected_keys(args)
    # 预览：列出将要操作的库及其存在性
    print("将清空以下本地库（", "整库删除" if args.delete_files else "清空内容", "）：", sep="")
    targets: list[tuple[str, Path]] = []
    for key in keys:
        if key == "kuzu":
            path = _resolve(KUZU_PATH)
        else:
            path = _resolve(SQLITE_DBS[key])
        exists = "存在" if path.exists() else "不存在"
        scope = f"（仅 thread={args.thread}）" if (key == "chat" and args.thread) else ""
        print(f"  · {key:<12} {path}  [{exists}]{scope}")
        targets.append((key, path))

    if not args.yes:
        reply = input("\n确认清空？(yes/N) ").strip().lower()
        if reply not in ("y", "yes"):
            print("已取消。")
            return

    print()
    for key, path in targets:
        if key == "chat" and args.thread:
            result = truncate_chat_thread(path, args.thread)
        elif key == "kuzu":
            result = delete_path(path)  # kuzu 不支持清表，整体删除
        elif args.delete_files:
            result = delete_path(path)
        else:
            result = truncate_sqlite(path)
        print(f"  ✓ {key:<12} {result}")
    print("\n完成。下次运行各 store 会自动重建空库/空表。")


if __name__ == "__main__":
    main()
