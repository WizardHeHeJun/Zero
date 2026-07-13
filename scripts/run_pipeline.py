"""端到端管线：合成训练 ExpressionDecoder → 注入 LangGraph 管线 → 跑刺激序列。

无需外部数据集（用合成数据 bootstrap 训练）。展示训练好的真网络如何驱动表达输出。
用法：python -m scripts.run_pipeline
"""

from __future__ import annotations

import asyncio
import logging

from scripts.train_expression import train
from src.agents.models.expression_decoder import load_decoder
from src.orchestration.runner import dump_trajectory, run
from src.orchestration.state import Stimulus

logger = logging.getLogger(__name__)


async def run_pipeline() -> None:
    # 1. 合成数据训练表达解码器并保存
    out = "artifacts/expression_decoder.pt"
    train(epochs=300, n=2048, out=out)
    decoder = load_decoder(out)

    # 2. 把训练好的解码器注入管线，跑一组对比刺激
    stimuli = [
        Stimulus(name="win", goal_congruence=0.9, intensity=0.9),
        Stimulus(name="loss", goal_congruence=-0.8, intensity=0.7),
    ]
    trajectory = await run(stimuli, thread_id="pipeline", rng_seed=7, expression_decoder=decoder)
    print(dump_trajectory(trajectory))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
