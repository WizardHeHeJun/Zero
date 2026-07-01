"""本地命令行验证 P4 情绪读出修复（无需 LLM / 无需服务，确定性）。

议会三轮判定：`--chat` 连续敌意时「情绪标签逐轮翻号」（敌意却采到正 valence）的根因是
**先验量级弱**（软 appraise 给 goal_congruence≈-0.2，post_mu valence≈-0.1、贴 0 边界）
**+ 逐轮单样本采样的大方差**。两条修复互补：
  - P3 `ZERO_APPRAISE_CALIBRATE`：把敌意标定拉到 -0.7~-0.9（先验远离 0 边界）。
  - P4 `ZERO_AFFECT_READOUT=map`：e*=post_mu（MMSE 点估计，去掉采样方差）。

本脚本绕过 LLM、直接给引擎喂 goal_congruence<0 的 stimulus，统计每轮 e* 的 valence 符号，
对比 sample（默认）vs map（P4）在三档先验强度下的翻号率——量化坐实两条修复各自的作用。

跑：python -m scripts.verify_affect_readout

期望（典型）：
  goal=-0.2（软 appraise 现场）：sample 翻号 ~20% / map 0%   → P4 在弱先验下消翻号
  goal=-0.7（P3 标定后）       ：sample 0%      / map 0%   → P3 拉强先验也独立消翻号
  各档 sample 与 map 的均值 valence 几乎一致 → map 只杀方差、不偏置信号（MMSE 性质）。
"""

from __future__ import annotations

import asyncio
import os

from src.orchestration.runner import run
from src.orchestration.state import Stimulus

N = 40  # 每档刺激轮数


async def _flip_rate(readout: str, goal: float) -> tuple[int, float]:
    """跑 N 轮 goal_congruence=goal 的敌意刺激，返回 (valence>0 的翻号轮数, 均值 valence)。

    sample 用 rng_seed=None（每轮新随机，显出翻号率）；map 确定性、与 seed 无关。
    """
    stims = [Stimulus(name=f"h{i}", goal_congruence=goal, intensity=0.7) for i in range(N)]
    traj = await run(
        stims,
        thread_id=f"verify-readout-{readout}-{goal}",
        workspace_enabled=True,  # 与 --chat 同款配置
        recall_enabled=False,
        affect_readout=readout,
    )
    vals = [t["valence_arousal"][0] for t in traj]
    return sum(1 for v in vals if v > 0), sum(vals) / len(vals)


async def main() -> None:
    print(f"连续 {N} 轮敌意刺激，统计引擎 e* 的 valence>0 翻号轮数（sample 每轮新随机）：\n")
    print(f"{'goal_congruence':>16} | {'sample 翻号(默认)':>22} | {'map 翻号(P4)':>18}")
    print("-" * 64)
    for goal in (-0.2, -0.35, -0.7):
        s_pos, s_mean = await _flip_rate("sample", goal)
        m_pos, m_mean = await _flip_rate("map", goal)
        print(
            f"{goal:>16.2f} | {s_pos:>3}/{N} ({s_pos / N:>4.0%}) 均值{s_mean:+.2f}"
            f"   | {m_pos:>3}/{N} ({m_pos / N:>4.0%}) 均值{m_mean:+.2f}"
        )


if __name__ == "__main__":
    # 入口策略（非库逻辑）：内存 checkpointer，避免 sqlite 落盘/teardown 噪声。
    os.environ.setdefault("ZERO_CHECKPOINT_BACKEND", "memory")
    asyncio.run(main())
