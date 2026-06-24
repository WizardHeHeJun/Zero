"""本地命令行验证 Graphiti 语义记忆闭环（无需 Docker / 无需服务）。

前置：
  pip install -e ".[graphiti]"        # 装 graphiti-core + kuzu（嵌入式图库）
  设 env（PowerShell 用 $env:NAME="..."）：
    ZERO_SEMANTIC_BACKEND=graphiti
    ZERO_GRAPHITI_DB=kuzu   # 嵌入式图库、本地无服务（⚠ kuzu 已 deprecated，仅作本地 smoke）
    ZERO_KUZU_PATH=data/graphiti.kuzu   # 落盘路径（默认值，可不设）
    ZERO_OPENAI_BASE_URL / ZERO_OPENAI_API_KEY / ZERO_GRAPHITI_MODEL   # OpenAI 兼容 LLM

跑：python -m scripts.verify_graphiti_local

验证逻辑：同一个 user 跑两次刺激（共享一个 MemoryClient，复用同一图库连接）。
第 1 次任务完成时 Supervisor 写富 episode → Graphiti 抽实体/关系入图；
第 2 次 MemoryRecall 语义召回上一条 → recalled_context 非空 → 进 LanguageAgent 检索。
看到 recalled_context 非空即"流程跑通"。
"""

from __future__ import annotations

import asyncio
import logging

from src.memory.client import MemoryClient
from src.orchestration.runner import run
from src.orchestration.state import Stimulus
from src.storage.graph_store import build_graph_store, build_semantic_store

logger = logging.getLogger(__name__)


async def verify() -> None:
    semantic = build_semantic_store()
    if semantic is None:
        print(
            "✗ 未启用语义后端。请设 ZERO_SEMANTIC_BACKEND=graphiti"
            "（+ ZERO_GRAPHITI_DB=kuzu + ZERO_OPENAI_* + ZERO_GRAPHITI_MODEL），并装 .[graphiti]。"
        )
        return

    # 复用同一个 MemoryClient（同一图库连接），避免两次重开嵌入式 kuzu 库
    client = MemoryClient(build_graph_store(), semantic=semantic)
    user, thread = "verify-user", "verify-thread"

    print("① 第 1 次刺激（负面）→ 任务完成写富 episode 到 Graphiti …")
    await run(
        [Stimulus(name="项目失败", goal_congruence=-0.9, intensity=1.0)],
        thread_id=thread,
        user_id=user,
        memory=client,
        recall_enabled=True,
        language_enabled=True,
        rng_seed=0,
    )

    print("② 第 2 次刺激 → MemoryRecall 语义召回上一条 episode …")
    traj = await run(
        [Stimulus(name="新的挑战", goal_congruence=0.1, intensity=0.6)],
        thread_id=thread,
        user_id=user,
        memory=client,
        recall_enabled=True,
        language_enabled=True,
        rng_seed=0,
    )

    ctx = traj[-1]["recalled_context"]
    print(f"\nrecalled_context（语义召回）: {ctx}")
    print(f"language_text（含召回的语言）: {traj[-1]['language_text']}")
    print("\n✓ 闭环跑通" if ctx else "\n⚠ recalled_context 为空——查 LLM/Graphiti 是否真的抽取入图")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(verify())


if __name__ == "__main__":
    main()
