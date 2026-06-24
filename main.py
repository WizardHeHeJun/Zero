"""【临时脚本】方便本地验证的启动入口——非项目正式入口，验证完可删。

跑情感表达多 Agent 管线，打印 (v,a) 轨迹与关键中间量。
默认零可选依赖（不需要 ml/graphiti），总能起。用法：
  python main.py                       # 跑一组示例刺激（纯核心管线）
  python main.py --mood --language     # 开启慢变心境 / 语言层双向回路
  python main.py --recall              # 开启长期倾向记忆回灌（多轮才显效）

更专项的入口：
  python -m scripts.demo_pipeline          # 端到端真网络化 demo（合成训练 → 注入 → 跑）
  python -m scripts.verify_graphiti_local  # Graphiti 语义召回本地验证（需 .[graphiti] + LLM key）
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from src.orchestration.runner import dump_trajectory, run
from src.orchestration.state import Stimulus

DEFAULT_STIMULI = [
    Stimulus(name="win", goal_congruence=0.9, intensity=0.9),
    Stimulus(name="loss", goal_congruence=-0.8, intensity=0.7),
    Stimulus(name="neutral", goal_congruence=0.0, intensity=0.4),
]


async def _run(args: argparse.Namespace) -> None:
    trajectory = await run(
        DEFAULT_STIMULI,
        thread_id="main",
        rng_seed=7,
        mood_enabled=bool(args.mood),
        language_enabled=bool(args.language),
        recall_enabled=bool(args.recall),
    )
    print(dump_trajectory(trajectory))


def main() -> None:
    parser = argparse.ArgumentParser(description="情感表达多 Agent 系统 — 直接启动")
    parser.add_argument("--mood", action="store_true", help="开启慢变心境（A.7 滞后）")
    parser.add_argument("--language", action="store_true", help="开启语言层 affect↔language 回路")
    parser.add_argument("--recall", action="store_true", help="开启长期倾向记忆回灌")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
