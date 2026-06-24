"""【临时脚本】方便本地验证的启动入口——非项目正式入口，验证完可删。

跑情感表达多 Agent 管线，打印 (v,a) 轨迹与关键中间量。
默认零可选依赖（不需要 ml/graphiti），总能起。用法：
  python main.py                       # 跑一组示例刺激（纯核心管线）
  python main.py --mood --language     # 开启慢变心境 / 语言层双向回路
  python main.py --recall              # 开启长期倾向记忆回灌（多轮才显效）
  python main.py --llm                 # 接 OpenAI 兼容真模型做「文本输出情绪验证」

--llm 需 `.[llm]`（已装 openai 即可）+ 在 .env 配（库代码不读 .env，本脚本 _load_dotenv 读）：
  ZERO_OPENAI_API_KEY=sk-...                       # 必填
  ZERO_OPENAI_MODEL=<你的 key 可访问的模型 id>      # 必填（无代码默认，避免发到无权限模型）
  ZERO_OPENAI_BASE_URL=https://.../v1              # 可选（OpenAI/vLLM/第三方网关）

更专项的入口：
  python -m scripts.demo_pipeline          # 端到端真网络化 demo（合成训练 → 注入 → 跑）
  python -m scripts.verify_graphiti_local  # Graphiti 语义召回本地验证（需 .[graphiti] + LLM key）
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from src.orchestration.runner import dump_trajectory, run
from src.orchestration.state import Stimulus

DEFAULT_STIMULI = [
    Stimulus(name="win", goal_congruence=0.9, intensity=0.9),
    Stimulus(name="loss", goal_congruence=-0.8, intensity=0.7),
    Stimulus(name="neutral", goal_congruence=0.0, intensity=0.4),
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


async def _run_core(args: argparse.Namespace) -> None:
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


async def _run_llm() -> None:
    """接真 LLM 验证文本输出情绪：内核 e* → 生成语言 → 独立 VAD 反推 → 一致性。"""
    _load_dotenv()
    # 全部配置只来自 .env（本脚本 _load_dotenv 加载）——代码不写死默认，缺哪项就明确报错
    api_key = os.getenv("ZERO_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("ZERO_OPENAI_MODEL")
    if not api_key or not model_name:
        print(
            "✗ 缺少配置。请在 .env 设 ZERO_OPENAI_API_KEY 与 ZERO_OPENAI_MODEL"
            "（模型 id 取你的 key 实际可访问的那个；可选 ZERO_OPENAI_BASE_URL）。"
        )
        return
    # 延迟 import：仅 --llm 路径需要 openai，默认路径零可选依赖
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
    parser = argparse.ArgumentParser(description="情感表达多 Agent 系统 — 直接启动")
    parser.add_argument("--mood", action="store_true", help="开启慢变心境（A.7 滞后）")
    parser.add_argument("--language", action="store_true", help="开启语言层 affect↔language 回路")
    parser.add_argument("--recall", action="store_true", help="开启长期倾向记忆回灌")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="接 OpenAI 兼容真模型做文本输出情绪验证（需 .[llm] + key）",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(_run_llm() if args.llm else _run_core(args))


if __name__ == "__main__":
    main()
