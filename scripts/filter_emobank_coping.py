"""EmoBank 社会支配子集筛选（P-block · 议会 2026-07-15 修正1）。

议会裁定「文本→coping_potential」监督须**剔除社会支配噪声句**：文本主体持制度性权力
（君主/上级/法官）或命令句中施令者 D 高但被令方无情境选择——这类句子高 writer-D 来自
**社会地位/权力**，而非「主体对情境的可控评价」（coping potential 语义）。用原始全量 D
训练会把社会支配学进 coping_potential，污染 rage/fear 分离（心理席修正1 + 神经席联合背书）。

实现（工程范围·Q1 首版）：**关键词/命令句启发式**——保守宁漏勿滥（只剔高置信社会权力句，
不误伤普通高控制感句）。无金标「社会支配」标注集，故不报 F1，改报**过滤比例 + 被剔/保留句
writer-D 均值差**作可测代理（若假设成立，被剔句应系统性高 D）。首版启发式，可后续升级为
轻量分类器（tasks.md Q1）。

用法：
  python -m scripts.filter_emobank_coping --in data/emobank_writer.csv \
    --out data/emobank_writer_filtered.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# 制度性权力/社会地位名词（高置信社会支配来源；小写匹配词边界）。
POWER_NOUNS: frozenset[str] = frozenset(
    {
        "king",
        "queen",
        "emperor",
        "empress",
        "monarch",
        "president",
        "prime minister",
        "judge",
        "court",
        "jury",
        "magistrate",
        "governor",
        "senator",
        "official",
        "boss",
        "master",
        "mistress",
        "lord",
        "ruler",
        "commander",
        "general",
        "colonel",
        "officer",
        "chief",
        "captain",
        "sergeant",
        "dictator",
        "tyrant",
        "majesty",
        "authority",
        "authorities",
        "regime",
        "throne",
        "crown",
        "government",
    }
)

# 命令/裁决动词（施令者对局面单向支配；被令方无情境选择）。
COMMAND_VERBS: frozenset[str] = frozenset(
    {
        "order",
        "ordered",
        "orders",
        "command",
        "commanded",
        "commands",
        "decree",
        "decreed",
        "dictate",
        "dictated",
        "sentence",
        "sentenced",
        "condemn",
        "condemned",
        "forbid",
        "forbade",
        "forbidden",
        "prohibit",
        "prohibited",
        "rule",
        "ruled",
        "govern",
        "governed",
        "reign",
        "reigned",
        "enslave",
        "enslaved",
        "conquer",
        "conquered",
        "subjugate",
        "subjugated",
    }
)

_WORD_RE = re.compile(r"[a-z']+")


def is_social_dominance(text: str) -> bool:
    """启发式判定：文本是否为高置信社会支配/制度性权力句（应剔除）。

    命中条件（任一）：① 含制度性权力名词；② 含命令/裁决动词。保守取并集但词表精挑，
    宁漏勿滥——普通「我能掌控」类高控制感句不含这些词、不被误剔。
    """
    lowered = text.lower()
    # 多词短语单独检测（如 "prime minister"）。
    for phrase in ("prime minister",):
        if phrase in lowered:
            return True
    tokens = set(_WORD_RE.findall(lowered))
    if tokens & POWER_NOUNS:
        return True
    if tokens & COMMAND_VERBS:
        return True
    return False


def filter_rows(in_path: str, out_path: str) -> dict[str, float]:
    """读 writer CSV，剔社会支配句，写净化 CSV。返回统计（比例 + 被剔/保留 D 均值）。"""
    kept_d: list[float] = []
    dropped_d: list[float] = []
    total = 0
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(in_path, newline="", encoding="utf-8") as fin,
        open(out, "w", newline="", encoding="utf-8") as fout,
    ):
        reader = csv.DictReader(fin)
        assert reader.fieldnames is not None
        writer = csv.DictWriter(fout, fieldnames=list(reader.fieldnames))
        writer.writeheader()
        for row in reader:
            total += 1
            d_norm = (float(row["D"]) - 3.0) / 2.0  # 同 read_emobank_rows 归一，仅供统计
            if is_social_dominance(row["text"]):
                dropped_d.append(d_norm)
            else:
                kept_d.append(d_norm)
                writer.writerow(row)

    if total == 0:
        raise ValueError(f"{in_path} 无数据行")
    n_drop = len(dropped_d)
    ratio = n_drop / total
    kept_mean = sum(kept_d) / len(kept_d) if kept_d else float("nan")
    drop_mean = sum(dropped_d) / n_drop if n_drop else float("nan")
    return {
        "total": float(total),
        "dropped": float(n_drop),
        "ratio": ratio,
        "kept_d_mean": kept_mean,
        "dropped_d_mean": drop_mean,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="EmoBank 社会支配子集筛选（修正1）")
    parser.add_argument("--in", dest="in_path", default="data/emobank_writer.csv")
    parser.add_argument("--out", dest="out_path", default="data/emobank_writer_filtered.csv")
    args = parser.parse_args()

    stats = filter_rows(args.in_path, args.out_path)
    logger.info(
        "过滤：总 %d，剔除 %d（%.2f%%），保留 %d",
        int(stats["total"]),
        int(stats["dropped"]),
        stats["ratio"] * 100,
        int(stats["total"] - stats["dropped"]),
    )
    logger.info(
        "被剔句 writer-D 均值 %.3f vs 保留句 %.3f（差>0 印证社会支配句系统性高 D 的假设）",
        stats["dropped_d_mean"],
        stats["kept_d_mean"],
    )
    print(
        f"done: {args.out_path} kept={int(stats['total'] - stats['dropped'])} "
        f"dropped={int(stats['dropped'])} ({stats['ratio'] * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
