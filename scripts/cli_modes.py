"""验证 demo 场景（从临时入口 main.py 迁出）：核心轨迹 / 工作空间 / LLM 文本输出。

这些是「验证脚手架」而非产品核心逻辑，独立成可单独运行的脚本入口：
  python -m scripts.demo_modes --trace [--mood --language --recall]   # (v,a) 轨迹 JSON
  python -m scripts.demo_modes --workspace                            # 并行流点燃对比
  python -m scripts.demo_modes --llm                                  # LLM 文本输出情绪验证
"""

from __future__ import annotations

import argparse
import asyncio
import os

from src.observability import setup_logging
from src.orchestration.runner import dump_trajectory, run
from src.orchestration.state import Stimulus

DEFAULT_STIMULI = [
    Stimulus(name="win", goal_congruence=0.9, intensity=0.9),
    Stimulus(name="loss", goal_congruence=-0.8, intensity=0.7),
    Stimulus(name="neutral", goal_congruence=0.0, intensity=0.4),
]

# --workspace 用：对比哪些并行流点燃（突发高唤醒中性 vs 目标显著 vs 弱刺激）
WORKSPACE_STIMULI = [
    Stimulus(name="突如其来的巨响", goal_congruence=0.0, intensity=1.0),  # 生存流主导
    Stimulus(
        name="拿到梦寐以求的 offer", goal_congruence=0.9, attitude_appeal=0.7, intensity=0.8
    ),  # 评价/价值流主导
    Stimulus(name="无关紧要的小事", goal_congruence=0.05, intensity=0.2),  # 弱刺激，少数流点燃
]

# --llm 用：情绪鲜明的场景（OCC 评价维度），覆盖正/负 × 高/低强度
EMOTION_STIMULI = [
    Stimulus(name="收到心仪已久的礼物", goal_congruence=0.9, attitude_appeal=0.8, intensity=0.8),
    Stimulus(
        name="精心准备的方案被当众否决",
        goal_congruence=-0.9,
        standard_compliance=-0.6,
        intensity=0.9,
    ),
    Stimulus(name="久别重逢的老友", goal_congruence=0.6, attitude_appeal=0.7, intensity=0.5),
    Stimulus(
        name="被误解还受了委屈", goal_congruence=-0.7, standard_compliance=-0.8, intensity=0.7
    ),
]


def _load_dotenv() -> None:
    """装了 python-dotenv 且有 .env 就加载；未装静默跳过。仅脚本加载，库代码不依赖。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


async def run_core(args: argparse.Namespace) -> None:
    """纯核心管线（默认，零可选依赖）：打印 (v,a) 轨迹 JSON。"""
    trajectory = await run(
        DEFAULT_STIMULI,
        thread_id="main",
        rng_seed=7,
        mood_enabled=bool(args.mood),
        language_enabled=bool(args.language),
        recall_enabled=bool(args.recall),
    )
    print(dump_trajectory(trajectory))


async def run_workspace() -> None:
    """显著度门控全局工作空间：跑刺激，打印每个刺激点燃了哪些并行流 + e*。

    预期：高唤醒中性刺激（巨响）survival 流单独点燃；目标显著刺激 appraisal/value 主导；
    弱刺激只点燃最显著的一条（不空播）。纯核心管线，零可选依赖。
    """
    print("显著度门控全局工作空间｜并行流: survival(快) / appraisal(OCC) / value(RPE)\n")
    for i, stim in enumerate(WORKSPACE_STIMULI):
        traj = await run(
            [stim],
            thread_id=f"workspace-{i}",
            workspace_enabled=True,
            rng_seed=7,
        )
        s = traj[-1]
        ev = s["valence_arousal"]
        print(f"【{stim.name}】")
        print(f"  点燃的流   = {s['ignited_streams']}")
        print(
            f"  内核 e*    = ({ev[0]:+.2f}, {ev[1]:+.2f})  后验精度={s['affect_precision']:.2f}\n"
        )


async def run_llm() -> None:
    """接真 LLM 验证文本输出情绪：内核 e* → 生成语言 → 独立 VAD 反推 → 一致性。"""
    _load_dotenv()
    # 全部配置只来自 .env——代码不写死默认，缺哪项就明确报错
    api_key = os.getenv("ZERO_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("ZERO_OPENAI_MODEL")
    if not api_key or not model_name:
        print(
            "✗ 缺少配置。请在 .env 设 ZERO_OPENAI_API_KEY 与 ZERO_OPENAI_MODEL"
            "（模型 id 取你的 key 实际可访问的那个；可选 ZERO_OPENAI_BASE_URL）。"
        )
        return
    # 延迟 import：仅 --llm 路径需要 openai
    from src.agents.emotion_lexicon import affect_descriptor
    from src.agents.language_openai import OpenAILanguageModel

    lm = OpenAILanguageModel(model=model_name, use_lexicon=True)
    print(f"模型={model_name}｜use_lexicon｜appraisal_conditioning｜双向回路\n")
    for i, stim in enumerate(EMOTION_STIMULI):
        traj = await run(
            [stim],
            thread_id=f"llm-emo-{i}",
            language_enabled=True,
            appraisal_conditioning_enabled=True,
            language_model=lm,
            rng_seed=7,
        )
        s = traj[-1]
        ev = s["valence_arousal"]
        la = s["language_affect"]
        cons = s["language_consistency"]
        ev_d = affect_descriptor(ev[0], ev[1])
        print(f"【{stim.name}】")
        print(f"  内核 e*      = ({ev[0]:+.2f}, {ev[1]:+.2f}) → {ev_d}")
        print(f"  生成语言     = {s['language_text']}")
        if la is not None:
            la_d = affect_descriptor(la[0], la[1])
            print(f"  语言反推 VAD = ({la[0]:+.2f}, {la[1]:+.2f}) → {la_d}")
        print(f"  一致性距离   = {cons:.3f}（越小越贴合）  回路迭代={s['language_iter']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="情感系统验证 demo 场景（核心 / 工作空间 / LLM）")
    parser.add_argument("--trace", action="store_true", help="跑核心管线打印 (v,a) 轨迹 JSON")
    parser.add_argument("--mood", action="store_true", help="[--trace] 慢变心境（A.7 滞后）")
    parser.add_argument("--language", action="store_true", help="[--trace] 语言层双向回路")
    parser.add_argument("--recall", action="store_true", help="[--trace] 长期倾向记忆回灌")
    parser.add_argument("--workspace", action="store_true", help="并行流 + 显著度门控全局工作空间")
    parser.add_argument("--llm", action="store_true", help="接真模型做文本输出情绪验证（需 key）")
    args = parser.parse_args()
    setup_logging()
    if args.workspace:
        asyncio.run(run_workspace())
    elif args.llm:
        asyncio.run(run_llm())
    else:
        asyncio.run(run_core(args))


if __name__ == "__main__":
    main()
