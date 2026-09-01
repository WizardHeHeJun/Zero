"""c-behavior-tempo 文本层占比表 v2：T1×T2 2×2 析因实跑（需 LLM API，design.md 裁决 T2-8）。

议会裁定：参数层单测是**必要非充分**——温度→熵单调是数学事实，但熵→「审慎语义」的
第二跳未证（DPI 预警：文本层可分性上界=参数层 TV），必须以生产路径实跑的**输出文本统计**
验收。本脚本按 2×2 析因（T1 风格句 × T2 温度偏置）各跑 N 轮。

**指标 v2（2026-08-31 数学席复裁后修订，与 v1 报告不可混比）**：

  - 平均句长（字符/句，终止符切分）——**T1 预注册主端点，单侧**（高C 风格句 → 句长增；
    v1 实测 t≈2.0 临界显著，本轮 n=60/臂 复验）
  - distinct-2（字符 bigram 型例比）与字符 TTR——**T2 预注册主端点，单侧**（降温 → token
    熵降 → 词汇/搭配多样性降；依据=softmax 熵单调性的直接推论，Hinton et al. 2015
    arXiv:1503.02531。⚠ v1 曾引 arXiv:2509.10179 佐证「温度→多样性」，数学席现场核验
    该文不支持此论点，已撤；实证引文待补核，机制依据如实标注为数学推论）
  - hedge（模糊限制语）密度、分段数——探索性次端点（分段数 v1 显**正向超可加交互**
    信号：T1 边际 0.33/0.67 随 T2 状态翻倍，须交互项检验后才能归入锚点，本轮随附
    双重差分）
  - ~~结构化连接词密度~~——**废弃**（v1 实测 μ≤0.17 且有臂 σ=0，退化点质量，不满足
    design-door「分布不塌陷」）

⚠ 词表/切分是**离线测量工具**（分析侧操作化），不进生产路径、不受 text-predicate-admission
管辖；改动=改口径，重跑前后不可混比。**每轮原始回复文本随报告落盘**（v1 未存原文导致
无法免费重算新指标——教训：占比表必须留语料）。

生效组合与生产一致：走 `persona_prompt_text` + `persona_converse_temperature_offset`
两个生产装配函数构造 `OpenAILanguageModel`，固定中性情绪与固定输入集（隔离引擎/记忆
噪声——本表只归因 T1/T2 两机制）。

跑：python -m scripts.verify_c_tempo_distribution [每臂轮数，默认 60]
产出：PRP/c-behavior-tempo/verification-run-<日期>-v2.md
判读：T2 若在 distinct-2 上仍贴噪声地板 → 按数学席复裁先按新指标效应量做功效核算再谈
K=0.25→0.4，不得沿用句长口径的 n、不得只调 K 不复验。
"""

from __future__ import annotations

import asyncio
import os
import re
import statistics
import sys
from datetime import date
from pathlib import Path


def _load_dotenv() -> None:
    """与 main.py 同款：可选 dotenv，未装则静默跳过（secrets 由环境注入，不入库）。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


# ── 预注册指标操作化 v2（离线测量词表；改动=改口径，前后不可混比）────────────
_SENT_SPLIT = re.compile(r"[。！？!?；;\n]+")
_HEDGES = ("可能", "大概", "或许", "也许", "不太确定", "我觉得", "应该是", "差不多")

_USER_PROMPTS = [
    "最近想换个工作，你怎么看？",
    "周末打算出去玩，有什么建议吗？",
    "我在学做饭，先从什么练起好？",
    "帮我想想怎么安排下个月的预算。",
    "朋友约我临时去旅行，去不去？",
    "我总是睡得很晚，怎么调整？",
    "想养只猫，需要准备些什么？",
    "工作和学习时间冲突了，怎么办？",
    "最近有点提不起劲，聊聊？",
    "我想学门新语言，从哪开始？",
]

# 自定义卡（不在预设闭集 → T1 可注入）
_HIGH_C_CARD = "你是一个稳重的对话伙伴，和这位用户是初次对话。"

_METRIC_NAMES = ("句长", "TTR", "distinct2", "hedge", "分段")


def _metrics(text: str) -> tuple[float, float, float, int, int]:
    """(平均句长, 字符TTR, distinct-2, hedge 命中, 分段数)。空回复按零处理。"""
    sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    n_seg = len(sents)
    avg_len = (sum(len(s) for s in sents) / n_seg) if n_seg else 0.0
    chars = [ch for ch in text if not ch.isspace()]
    ttr = (len(set(chars)) / len(chars)) if chars else 0.0
    bigrams = [text[i : i + 2] for i in range(len(text) - 1)]
    distinct2 = (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    n_hedge = sum(text.count(w) for w in _HEDGES)
    return avg_len, ttr, distinct2, n_hedge, n_seg


async def _run_arm(t1: bool, t2: bool, rounds: int) -> tuple[list[tuple[float, ...]], list[str]]:
    """一个析因臂：按生产装配函数构造 lm，逐轮 converse 固定输入集，返回 (指标, 原文)。"""
    from src.agents.language_openai import OpenAILanguageModel
    from src.agents.persona import _persona_from_dict
    from src.orchestration.chat_driver import (
        persona_converse_temperature_offset,
        persona_prompt_text,
    )

    os.environ["ZERO_PERSONA_C_STYLE"] = "true" if t1 else "false"
    os.environ["ZERO_PERSONA_C_TEMPO"] = "true" if t2 else "false"
    persona = _persona_from_dict({"card": _HIGH_C_CARD, "big_five": {"conscientiousness": 0.6}})
    offset = persona_converse_temperature_offset(persona)
    if offset is not None:
        lm = OpenAILanguageModel(
            model=os.environ["ZERO_OPENAI_MODEL"],
            persona=persona_prompt_text(persona),
            converse_temperature_offset=offset,
        )
    else:
        lm = OpenAILanguageModel(
            model=os.environ["ZERO_OPENAI_MODEL"], persona=persona_prompt_text(persona)
        )
    rows: list[tuple[float, ...]] = []
    texts: list[str] = []
    for i in range(rounds):
        user = _USER_PROMPTS[i % len(_USER_PROMPTS)]
        reply = await lm.converse([{"role": "user", "content": user}], (0.0, 0.0))
        rows.append(_metrics(reply))
        texts.append(reply)
    return rows, texts


def _arm_summary(rows: list[tuple[float, ...]]) -> str:
    cols = list(zip(*rows, strict=True))
    parts = []
    for name, vals in zip(_METRIC_NAMES, cols, strict=True):
        fv = [float(v) for v in vals]
        parts.append(f"{name} μ={statistics.mean(fv):.3f} σ={statistics.pstdev(fv):.3f}")
    return " · ".join(parts)


def _cell_means(rows: list[tuple[float, ...]]) -> list[float]:
    cols = list(zip(*rows, strict=True))
    return [statistics.mean(float(v) for v in vals) for vals in cols]


async def main() -> None:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    _load_dotenv()
    if not (
        (os.getenv("ZERO_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))
        and os.getenv("ZERO_OPENAI_MODEL")
    ):
        raise SystemExit(
            "缺 LLM 配置（ZERO_OPENAI_API_KEY/OPENAI_API_KEY + ZERO_OPENAI_MODEL）——"
            "本脚本必须真跑 LLM 产出文本层占比表，无降级路径（fail-fast）。"
        )
    arms: dict[str, list[tuple[float, ...]]] = {}
    corpora: dict[str, list[str]] = {}
    for t1 in (False, True):
        for t2 in (False, True):
            key = f"T1={'开' if t1 else '关'}×T2={'开' if t2 else '关'}"
            print(f"跑臂 {key}（{rounds} 轮）…", flush=True)
            arms[key], corpora[key] = await _run_arm(t1, t2, rounds)

    keys = list(arms)  # 顺序：关关 / 关开 / 开关 / 开开
    m00, m01, m10, m11 = (_cell_means(arms[k]) for k in keys)
    lines = [
        f"# c-behavior-tempo 文本层占比表 v2（{date.today().isoformat()} · 每臂 {rounds} 轮）",
        "",
        "> 指标 v2（连接词已废弃、增 TTR/distinct-2；与 v1 不可混比）；预注册主端点：",
        "> T1→句长（单侧增）、T2→distinct-2（单侧降）。判读规则见 design.md 裁决 T2-8",
        "> 与数学席 2026-08-31 复裁（功效核算先于重跑/调 K）。",
        "",
        "| 臂 | 指标摘要 |",
        "| --- | --- |",
    ]
    for key in keys:
        lines.append(f"| {key} | {_arm_summary(arms[key])} |")
    lines += ["", "## 边际效应与交互（双重差分）", ""]
    lines.append("| 指标 | T1边际(T2关/开) | T2边际(T1关/开) | 交互(双重差分) |")
    lines.append("| --- | --- | --- | --- |")
    for i, name in enumerate(_METRIC_NAMES):
        t1_eff = (m10[i] - m00[i], m11[i] - m01[i])
        t2_eff = (m01[i] - m00[i], m11[i] - m10[i])
        inter = (m11[i] - m10[i]) - (m01[i] - m00[i])
        lines.append(
            f"| {name} | {t1_eff[0]:+.3f} / {t1_eff[1]:+.3f}"
            f" | {t2_eff[0]:+.3f} / {t2_eff[1]:+.3f} | {inter:+.3f} |"
        )
    lines += ["", "## 逐轮原始指标（句长/TTR/distinct2/hedge/分段）", ""]
    for key in keys:
        vals = " ".join(
            f"({r[0]:.0f},{r[1]:.2f},{r[2]:.2f},{r[3]:.0f},{r[4]:.0f})" for r in arms[key]
        )
        lines.append(f"- **{key}**：{vals}")
    lines += ["", "## 原始语料（供复算新指标，v1 未存原文的教训）", ""]
    for key in keys:
        lines.append(f"### {key}")
        lines.extend(f"{i + 1}. {t}".replace("\n", "␤") for i, t in enumerate(corpora[key]))
        lines.append("")
    out_path = Path("PRP/c-behavior-tempo") / f"verification-run-{date.today().isoformat()}-v2.md"
    # 先写临时文件再替换（pitfalls「open(w) 先截断再写」加固；同日重跑覆盖属预期）
    tmp_path = out_path.with_suffix(".tmp")
    tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp_path.replace(out_path)
    print(f"\n已写 {out_path}")
    for key in keys:
        print(f"  {key}: {_arm_summary(arms[key])}")


if __name__ == "__main__":
    asyncio.run(main())
