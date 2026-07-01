"""本地命令行验证 P3 评价标定（**需 LLM key**：读 .env 的 ZERO_OPENAI_*）。

议会二轮判定：`_APPRAISE_SYS` 无标定锚时，LLM 有系统性正向偏置（positivity bias），把明确
敌意只读成 goal_congruence≈-0.2~-0.3（贴 0 边界，是情绪标签翻号的先验侧根因之一）。P3
`ZERO_APPRAISE_CALIBRATE=1` 注入**分级**标定锚，应把敌意拉到 OCC anger 应有的 -0.7~-0.9，
而**不误伤**"语气直但中性"的输入。

本脚本对同一组分级样本，各调一次 appraise（标定关 vs 开），并排打印读数——实测标定是否
把强负样本拉强、又没把中性/轻负样本一并拉过头。这是议会规定 **P1(b) 落地的前置验证**。

跑：配好 .env(ZERO_OPENAI_API_KEY + ZERO_OPENAI_MODEL) 后
    `python -m scripts.verify_appraise_calibration`

期望（典型）：
  完全中性/轻微批评：开关两列都接近 0 / 轻负（标定没误伤）。
  明确敌意/极端攻击：开列显著更负（→ -0.7~-0.9 档），关列偏软（~-0.2~-0.4）。
"""

from __future__ import annotations

import asyncio
import os

# (标签, 文本) —— 与 _APPRAISE_CALIBRATION 的语义锚同源的分级样本；末样本不在锚里，测泛化。
SAMPLES: list[tuple[str, str]] = [
    ("完全中性", "你好，今天过得怎么样？"),
    ("语气直但中性", "把时间地点直接发我"),
    ("轻微批评", "这个回答不太对"),
    ("明确敌意", "你真没用，根本解决不了我的问题"),
    ("极端攻击", "滚开，你这个废物"),
]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


async def _appraise_all(calibrate: bool) -> dict[str, tuple[float, float]]:
    """按 calibrate 开/关设 env，对每个样本调一次真 LLM appraise，返回 {标签: (v,a)}。"""
    os.environ["ZERO_APPRAISE_CALIBRATE"] = "1" if calibrate else "0"
    from src.agents.language_openai import OpenAILanguageModel

    model = OpenAILanguageModel(model=os.environ["ZERO_OPENAI_MODEL"])  # key/base_url 从 env 读
    out: dict[str, tuple[float, float]] = {}
    for label, text in SAMPLES:
        out[label] = await model.appraise_text(text)
    return out


async def main() -> None:
    _load_dotenv()
    if not (os.getenv("ZERO_OPENAI_API_KEY") and os.getenv("ZERO_OPENAI_MODEL")):
        print("需先在 .env 配 ZERO_OPENAI_API_KEY + ZERO_OPENAI_MODEL（本 eval 需真 LLM）。")
        return
    off = await _appraise_all(calibrate=False)
    on = await _appraise_all(calibrate=True)
    print(f"模型={os.environ['ZERO_OPENAI_MODEL']}｜列为 appraise 读出 (valence, arousal)\n")
    print(f"{'输入类型':<14} | {'标定关(默认)':>16} | {'标定开(P3)':>16} | Δvalence")
    print("-" * 66)
    for label, _ in SAMPLES:
        vo, vn = off[label][0], on[label][0]
        print(
            f"{label:<14} | {str((round(off[label][0], 2), round(off[label][1], 2))):>16}"
            f" | {str((round(on[label][0], 2), round(on[label][1], 2))):>16} | {vn - vo:+.2f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
