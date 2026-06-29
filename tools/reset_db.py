"""清空本地数据库内容（架构测试用）。

系统在 `data/` 下落盘的本地库会随测试累积（对话历史污染 LLM 回合、态度/价值跨重启续上），
本脚本一键清空回到干净状态。路径用运行时同款 env，**自动加载根目录 `.env`** 对齐实际配置。覆盖：

- `chat`：固定 `data/chat_history.sqlite3` — 对话 transcript + 累积 attitude
- `graph`：`ZERO_GRAPH_DB`（`data/graph.sqlite3`）— 确定性倾向图谱（时序失效）
- `semantic`：`ZERO_SEMANTIC_DB`（`data/semantic.sqlite3`）— 语义 episode+向量（默认语义后端）
- `checkpoints`：`ZERO_CHECKPOINT_DB`（`data/checkpoints.sqlite3`）— 运行态（**WAL 模式**）
- `kuzu`：`ZERO_KUZU_PATH`（`data/graphiti.kuzu`）— **已弃用**，默认不清、需 `--kuzu` 整删

**为什么"清得干净"**：SQLite WAL 模式落 `-wal`/`-shm` 边车（kuzu 落 `.wal`/`.lock`）。truncate
先 `wal_checkpoint(TRUNCATE)` 折回主库再 `VACUUM`、关连接后清残留边车；`--delete-files` 连边车一并
删除——避免旧脚本"只删主库、留孤儿 WAL 被重放"的不干净。边车被活进程占用时跳过告警、不中断。

默认 **truncate**（清空各表、保留库文件下次直接复用）；`--delete-files` 整库删除（下次运行各 store
的 `CREATE TABLE IF NOT EXISTS` 自动重建）。不碰容器后端（Postgres/Neo4j）——那是部署态、需要时重建
容器卷。数据集文件（ravdess/wesad/emobank 等）不在清单内，绝不触碰。

用法：
  python -m tools.reset_db                  # 清空全部本地库（确认后）
  python -m tools.reset_db --yes            # 跳过确认
  python -m tools.reset_db --chat           # 只清对话库
  python -m tools.reset_db --chat --thread chat   # 只清对话库里 thread=chat 的记录（保留别的会话）
  python -m tools.reset_db --delete-files --yes   # 整库删除（连边车）而非清表
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
from pathlib import Path

# 仓库根（tools/ 的父目录）——使脚本与运行目录无关
ROOT = Path(__file__).resolve().parent.parent

# SQLite 在 WAL/回滚模式下的边车文件后缀（清/删主库时须一并处理，否则留孤儿 WAL 可被重放）
SQLITE_SIDECARS = ("-wal", "-shm", "-journal")
# kuzu 嵌入式库的边车（不同版本可能为目录或文件 + 这些同名旁文件）
KUZU_SIDECARS = (".wal", ".shadow", ".lock", ".tmp")

SQLITE_KEYS = ("chat", "graph", "semantic", "checkpoints")
ALL_KEYS = (*SQLITE_KEYS, "kuzu")


def _load_dotenv() -> None:
    """装了 python-dotenv 且有根目录 `.env` 就加载，使清库路径与运行时实际配置一致；未装静默跳过。

    与各正式入口（main.py / scripts）同款：库代码不依赖 dotenv，仅入口/工具脚本加载 `.env`。
    必须在读取 `ZERO_*` env 之前调用（故路径表在 `_db_paths` 里惰性读取，不在 import 时定死）。
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def _db_paths() -> dict[str, str]:
    """各本地库路径（env 覆盖默认；须在 `_load_dotenv` 之后调用）。默认镜像 src 各 store 工厂。"""
    return {
        # chat 在框架内无 env 覆盖：固定 conversation_log.DEFAULT_CHAT_HISTORY_PATH
        "chat": "data/chat_history.sqlite3",
        "graph": os.getenv("ZERO_GRAPH_DB", "data/graph.sqlite3"),
        "semantic": os.getenv("ZERO_SEMANTIC_DB", "data/semantic.sqlite3"),
        "checkpoints": os.getenv("ZERO_CHECKPOINT_DB", "data/checkpoints.sqlite3"),
    }


def _kuzu_path() -> str:
    return os.getenv("ZERO_KUZU_PATH", "data/graphiti.kuzu")


def _resolve(path: str) -> Path:
    """相对路径按仓库根解析；绝对路径原样。"""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _sidecars(path: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """主库路径对应的边车文件（按后缀拼接，存在与否不在此判定）。"""
    return [path.with_name(path.name + suffix) for suffix in suffixes]


def _path_size(path: Path) -> int:
    """文件/目录占用字节（目录递归求和）；不存在记 0。"""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _footprint(key: str, path: Path) -> int:
    """该 key 的磁盘总占用（主库 + 所有边车），用于预览展示真实清理量。"""
    suffixes = KUZU_SIDECARS if key == "kuzu" else SQLITE_SIDECARS
    return _path_size(path) + sum(_path_size(s) for s in _sidecars(path, suffixes))


def _human(num: int) -> str:
    """字节数转人类可读（B/KB/MB/GB）。"""
    size = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _rm(path: Path) -> None:
    """删一个文件或目录。"""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _remove_sidecars(path: Path, suffixes: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """删除存在的边车文件/目录，返回 (已删后缀, 锁定跳过后缀)。

    best-effort：边车被占用（Windows 上活进程持 -wal）时**不抛**，记录跳过、继续——主表已清，
    删不掉的边车不该让整次清库以 traceback 中断（旧脚本正是在此崩，见会话教训）。
    """
    removed: list[str] = []
    locked: list[str] = []
    for side in _sidecars(path, suffixes):
        if not side.exists():
            continue
        suffix = side.name[len(path.name) :]
        try:
            _rm(side)
            removed.append(suffix)
        except OSError:
            locked.append(suffix)
    return removed, locked


def truncate_sqlite(path: Path) -> str:
    """清空一个 SQLite 库的所有用户表，折回并截断 WAL、VACUUM 回收，再清残留边车。"""
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
        # 把 WAL 折回主库并截断 -wal（AsyncSqliteSaver 用 WAL 模式，否则旧 WAL 残留 4MB+）
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError:
            pass  # 非 WAL 模式无此 pragma，忽略
        conn.execute("VACUUM")
    finally:
        conn.close()
    removed, locked = _remove_sidecars(path, SQLITE_SIDECARS)  # 关连接后清残留 -wal/-shm/-journal
    extra = f"，清残留 {'/'.join(removed)}" if removed else ""
    warn = f"（⚠ {'/'.join(locked)} 被占用未删，关掉在用进程后重跑）" if locked else ""
    return f"已清空 {len(tables)} 张表{extra}{warn}"


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
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # 折回 WAL，别让删的行还堆在 -wal
        except sqlite3.OperationalError:
            pass
    finally:
        conn.close()
    return f"已删 thread={thread} 的 {removed} 行"


def delete_db_files(path: Path, suffixes: tuple[str, ...]) -> str:
    """整体删除主库 + 所有边车（文件或目录），best-effort。下次运行各 store 自动重建。

    被占用（活进程持文件）时不抛，记录跳过——避免整次清库以 traceback 中断（见会话教训）。
    """
    existing = [p for p in (path, *_sidecars(path, suffixes)) if p.exists()]
    if not existing:
        return "不存在，跳过"
    deleted: list[str] = []
    locked: list[str] = []
    for p in existing:
        try:
            _rm(p)
            deleted.append(p.name)
        except OSError:
            locked.append(p.name)
    msg = f"已删除 {len(deleted)} 项（{'、'.join(deleted)}）" if deleted else "未删除任何项"
    if locked:
        msg += f"（⚠ {'、'.join(locked)} 被占用未删，关掉在用进程后重跑）"
    return msg


def _selected_keys(args: argparse.Namespace) -> list[str]:
    """按 flag 选目标；未指定任一具体库则选默认全集（SQLITE_KEYS，**不含 kuzu**）。

    kuzu 已弃用、默认从不产数据（默认语义后端是 sqlite_vec，非 graphiti/kuzu），故不进默认全清，
    避免每次清库刷一行无意义的 `[不存在]`；仍可经 `--kuzu` 显式清理（真用过 kuzu 落了盘时）。
    """
    explicit = [k for k in ALL_KEYS if getattr(args, k)]
    return explicit or list(SQLITE_KEYS)


def main() -> None:
    parser = argparse.ArgumentParser(description="清空本地数据库内容（架构测试用）")
    parser.add_argument("--chat", action="store_true", help="对话库 chat_history")
    parser.add_argument("--graph", action="store_true", help="长期倾向图谱 graph")
    parser.add_argument("--semantic", action="store_true", help="语义记忆 semantic")
    parser.add_argument("--checkpoints", action="store_true", help="运行态 checkpointer")
    parser.add_argument(
        "--kuzu",
        action="store_true",
        help="Graphiti kuzu 图库（已弃用，默认不清，需显式指定，整体删除）",
    )
    parser.add_argument("--thread", help="只清对话库里该 thread 的记录（仅对 chat 生效）")
    parser.add_argument(
        "--delete-files",
        action="store_true",
        help="整库删除（连 WAL/边车）而非清表（下次运行自动重建）",
    )
    parser.add_argument("--yes", action="store_true", help="跳过确认")
    args = parser.parse_args()

    _load_dotenv()  # 入口负责加载 .env，使路径与运行时实际配置一致（库代码不读 .env）
    db_paths = _db_paths()
    kuzu_path = _kuzu_path()

    keys = _selected_keys(args)
    # 预览：列出将要操作的库、存在性与真实占用（含边车），让"清了多少"一目了然
    action = "整库删除（连边车）" if args.delete_files else "清空内容"
    print(f"将对以下本地库执行【{action}】：")
    targets: list[tuple[str, Path]] = []
    for key in keys:
        path = _resolve(kuzu_path if key == "kuzu" else db_paths[key])
        if path.exists():
            state = f"存在 {_human(_footprint(key, path))}"
        else:
            state = "不存在"
        scope = f"（仅 thread={args.thread}）" if (key == "chat" and args.thread) else ""
        print(f"  · {key:<12} {path}  [{state}]{scope}")
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
            result = delete_db_files(path, KUZU_SIDECARS)  # kuzu 不支持清表，整体删除
        elif args.delete_files:
            result = delete_db_files(path, SQLITE_SIDECARS)
        else:
            result = truncate_sqlite(path)
        print(f"  ✓ {key:<12} {result}")
    print("\n完成。下次运行各 store 会自动重建空库/空表。")


if __name__ == "__main__":
    main()
