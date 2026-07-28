"""本地命令行验证语义记忆闭环（无需 Docker / 无需图库服务）。

两种后端都支持，由 `ZERO_SEMANTIC_BACKEND` 选：
  sqlite_vec（推荐先跑这条）：只需 OpenAI 兼容 key，无需图库服务。
  graphiti：另需 `pip install -e ".[graphiti]"` + 图库服务（`ZERO_GRAPHITI_DB`）。

配 env——二选一：填进根目录 `.env`（本脚本自动加载、填一次复用）；或 shell 导出：
  ZERO_SEMANTIC_BACKEND / ZERO_OPENAI_API_KEY / ZERO_OPENAI_BASE_URL / ZERO_GRAPHITI_EMBED_MODEL

跑：python -m scripts.verify_graphiti_local

验证逻辑：同一个 user 连跑两次刺激（共享一个 MemoryClient / 同一后端连接）。
第 1 次任务完成时 Supervisor 写富 episode（须过显著度门，故本脚本开 workspace 让
`affect_precision` 有值——salience=precision×|rpe|，不开则恒 0、永远写不进）；
第 2 次 MemoryRecall 按 stimulus.name 语义召回上一条 → recalled_context 非空 →
进 LanguageAgent 检索。看到 recalled_context 非空即"闭环跑通"。
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from src.memory.client import MemoryClient
from src.orchestration.runner import run
from src.orchestration.state import Stimulus
from src.storage.graph_store import build_graph_store, build_semantic_store

logger = logging.getLogger(__name__)

# 自查用独立库：不写用户的生产语义库（默认 data/semantic.sqlite3，含真实对话记忆）。
# 每次跑前清空，保证结果可复现——否则上一轮的同文本 episode 会被 dedup 跳过。
# 想改在真实库上验证，设 ZERO_VERIFY_SEMANTIC_DB 指向它。
VERIFY_DB = os.getenv("ZERO_VERIFY_SEMANTIC_DB", "data/verify_semantic.sqlite3")


def _load_dotenv() -> None:
    """若装了 python-dotenv 且存在 .env，则加载（env 填进 .env 一次复用，免每次 shell 导出）。

    可选：未装 python-dotenv 时静默跳过——仍可用 shell 导出 env。仅本便捷脚本加载 .env，
    不影响库代码（库一律 os.getenv，由真实环境/容器注入）。
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


async def verify() -> None:
    _load_dotenv()
    # 隔离到自查专用库并清空（在 build_semantic_store 之前设，工厂构造期才读得到）
    if VERIFY_DB != ":memory:":
        Path(VERIFY_DB).unlink(missing_ok=True)
    os.environ["ZERO_SEMANTIC_DB"] = VERIFY_DB
    print(f"自查库（每次清空、不动生产库）：{VERIFY_DB}\n")
    semantic = build_semantic_store()
    if semantic is None:
        print(
            "✗ 未启用语义后端。请设 ZERO_SEMANTIC_BACKEND=sqlite_vec（+ ZERO_OPENAI_API_KEY /"
            " ZERO_OPENAI_BASE_URL / ZERO_GRAPHITI_EMBED_MODEL）；走 graphiti 则另需 .[graphiti]。"
        )
        return

    # 复用同一个 MemoryClient（同一后端连接），避免两次重开嵌入式库
    client = MemoryClient(build_graph_store(), semantic=semantic)
    user, thread = "verify-user", "verify-thread"

    # workspace_enabled=True：让 affect_core 写出 affect_precision。Supervisor 的
    # 显著度门 salience=precision×|rpe| 靠它才有值，否则恒 0、episode 永远写不进去。
    common = dict(
        thread_id=thread,
        user_id=user,
        memory=client,
        recall_enabled=True,
        language_enabled=True,
        workspace_enabled=True,
        rng_seed=0,
    )

    print("① 第 1 次刺激（负面）→ 任务完成写富 episode 到语义记忆 …")
    await run(
        [
            Stimulus(
                name="项目失败",
                text="我负责的项目失败了，半年的心血全白费",
                goal_congruence=-0.9,
                intensity=1.0,
            )
        ],
        **common,
    )

    print("② 第 2 次刺激 → MemoryRecall 语义召回上一条 episode …")
    traj = await run(
        [Stimulus(name="项目失败之后的心情", goal_congruence=0.1, intensity=0.6)],
        **common,
    )

    ctx = traj[-1]["recalled_context"]
    print(f"\nrecalled_context（语义召回）: {ctx}")
    print(f"language_text（含召回的语言）: {traj[-1]['language_text']}")
    if ctx:
        print("\n✓ 闭环跑通")
    else:
        print(
            "\n⚠ recalled_context 为空。依次查："
            "\n   1) episode 是否写进去了——显著度门 ZERO_EPISODE_SALIENCE_MIN（默认 0.15）"
            "\n   2) 召回相似度是否够——ZERO_RECALL_SIM_MIN（默认 0.65）"
            "\n   3) embedding 模型是否可用——ZERO_GRAPHITI_EMBED_MODEL 须是 key 有权限的嵌入模型"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(verify())


if __name__ == "__main__":
    main()
