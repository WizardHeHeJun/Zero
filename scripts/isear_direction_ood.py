"""ISEAR 跨源 OOD 方向门（B-fe 第二生存源·2026-07-21/22·仅验证不训练）。

承接 B-fe 设计门议会裁决（notes/2026-07-21-fear-b-fe-source-selection-council.md·总判 PASS）：
fear 生产解锁须第二生存叙事/防御威胁体裁的独立源·fear Wilson LB≥0.80（现仅 ED 单源 0.90
=工程近似·数学席 R 条件：d_HΔH(ED, survival 全分布) 单源不可估）。四席裁 **ISEAR 单主源**
（Scherer & Wallbott 1994·原生 fear 1095/anger 1096·第一人称情绪前因情境·CC BY-NC-SA 3.0·
**仅验证不训练·license 传染禁·gitignore·Research-Only**）。

**本脚本 = A 类选项 α：跑全 fear/anger baseline Wilson（不提纯）。**
数学席单调性引理：全集 LB≥0.80 → 物理威胁提纯子集 LB≥全集（sustained anxiety 移除后只升不降）。
⚠ **全集 LB 是保守上界·非 pure fear 可靠性**：心理席实测 ISEAR fear ~48% 即时物理威胁
（immediate-concrete-physical·Lazarus 1991 fear 核心关系主题）+ ~44% 预期弥漫焦虑
（uncertain-existential·Lazarus anxiety·功能上 sustained 型·与社交域 SemEval 0.709 构念重叠）
+ ~8% 模糊。全集含 ~44% sustained anxiety 掺入 → LB 偏低（保守方向）；提纯版 LB 为真值参考区间。

⚠ **禁解剖标签**（神经席硬约束·Didier/Grogans 2026 SCAN·CeA/BNST 人类 fMRI 无法稳定二分·
BF₁₀=0.07）：注释以**功能性时间维度**（phasic 即时可终止 fear vs sustained 持续弥漫 anxiety·
Davis/Walker 2010 功能尺度）描述·不声称对应 CeA/BNST 解剖区。

⚠ **验证 ≠ 解锁**：fear 维持默认关（ZERO_FEAR_DOMAIN_ENABLED=false）·本脚本仅出方向可靠性数据
回议会（B-fe-unlock 正式解锁须单独议会门：两源 LB 均≥0.80·R 条件披露·π_t(fear)=0.08 确认）。
提纯判据词表属议会定（B-2·四维锚·工程不私拍）·本脚本不做构念提纯（仅去无效占位行=数据清洗）。

标签（ISEAR Field1 文本·EMOT 数字 2=fear/3=anger）：anger→期望 logit>0(+)·fear→期望 logit<0(−)。
full-coverage 分侧 Wilson LB（fear B-fe 门槛 0.80·anger 对照门槛 0.70·与前源同口径）。
权重 motivational_direction_prior_m.pt（GoEmotions+crowd-enVENT·ISEAR 是真 OOD）。
数据 data/raw/isear/isear.csv（`|` 分隔·latin-1 编码·本地 gitignore·仅验证不训练）。

用法：PYTHONPATH=d:/Zero HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CONDA_NO_PLUGINS=true \
  "E:/anaconda/Scripts/conda.exe" run -n affective-expression --no-capture-output \
  python -m scripts.isear_direction_ood --weights artifacts/motivational_direction_prior_m.pt
"""

from __future__ import annotations

import argparse
import csv
import logging

import torch

from scripts.direction_gate_text_coping import wilson_lower_bound
from scripts.train_direction_head import DirectionHead
from src.agents.models.text_affect_regressor_st import DEFAULT_ENCODER, ST_FEATURE_DIM, encode_texts

logger = logging.getLogger(__name__)

_ANGER = "anger"
_FEAR = "fear"
# 无效占位（ISEAR 部分被试未作答/空情境）——数据清洗去无效行，非构念提纯（提纯判据属议会 B-2）。
_INVALID_MARKERS = ("no response", "[ no response", "never felt", "not applicable", "blank")


def _read_isear(csv_path: str) -> tuple[list[str], list[str]]:
    """读 ISEAR（`|` 分隔·latin-1）：Field1=anger→'+'、fear→'-'；SIT 为情境文本。

    跳过 SIT 为空或含无效占位标记的行（诚实丢弃·非静默截断·非构念提纯）。
    """
    texts: list[str] = []
    pol: list[str] = []
    skipped = 0
    with open(csv_path, encoding="latin-1") as f:
        for row in csv.DictReader(f, delimiter="|"):
            emo = (row.get("Field1") or "").strip().lower()
            if emo not in (_ANGER, _FEAR):
                continue
            sit = (row.get("SIT") or "").strip()
            low = sit.lower()
            if not sit or any(m in low for m in _INVALID_MARKERS):
                skipped += 1
                continue
            pol.append("+" if emo == _ANGER else "-")
            texts.append(sit)
    if skipped:
        logger.warning("ISEAR 无效占位行整行跳过 %d 行（SIT 空或 no-response 类）", skipped)
    return texts, pol


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(
        description="ISEAR 跨源 OOD 方向门（B-fe 第二生存源·选项 α 全 baseline·仅验证）"
    )
    p.add_argument("--csv", default="data/raw/isear/isear.csv")
    p.add_argument("--weights", default="artifacts/motivational_direction_prior_m.pt")
    args = p.parse_args()

    texts, pol = _read_isear(args.csv)
    model = DirectionHead(ST_FEATURE_DIM)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logit = model(encode_texts(texts, encoder=DEFAULT_ENCODER)).tolist()

    a_pred = [logit[i] for i in range(len(pol)) if pol[i] == "+"]
    f_pred = [logit[i] for i in range(len(pol)) if pol[i] == "-"]
    a_ok = sum(1 for v in a_pred if v > 0)
    f_ok = sum(1 for v in f_pred if v < 0)
    a_lb = wilson_lower_bound(a_ok, len(a_pred)) if a_pred else 0.0
    f_lb = wilson_lower_bound(f_ok, len(f_pred)) if f_pred else 0.0

    logger.info("─" * 70)
    logger.info("ISEAR 跨源 OOD 方向门（第一人称情绪前因情境·原生金标·仅验证·B-fe 第二生存源）:")
    logger.info(
        "  anger  %d/%d=%.1f%%  Wilson 下界 %.4f",
        a_ok,
        len(a_pred),
        (a_ok / len(a_pred) * 100) if a_pred else 0.0,
        a_lb,
    )
    logger.info(
        "  fear   %d/%d=%.1f%%  Wilson 下界 %.4f",
        f_ok,
        len(f_pred),
        (f_ok / len(f_pred) * 100) if f_pred else 0.0,
        f_lb,
    )
    logger.info("─" * 70)
    logger.info(
        "⚠ 全集 baseline（选项 α·不提纯）：全 ISEAR fear 含 ~44%% 预期弥漫焦虑（Lazarus anxiety·"
    )
    logger.info(
        "  功能上 sustained 型·近社交域）掺入 → fear LB 为**保守上界**（数学席单调性引理：提纯物理"
    )
    logger.info(
        "  威胁子集[~48%%·phasic 即时可终止]后 LB 只升不降）。提纯版 LB 为真值参考·词表属议会 B-2。"
    )
    logger.info(
        "对照四源 fear：ED 生存叙事 LB0.90 > SemEval 社交 0.709 > DailyDialog 自然对话 0.582 >"
    )
    logger.info(
        "  EmoryNLP 表演对话 0.264(崩)。fear B-fe 门槛 LB≥0.80（第二生存源）；anger 对照门槛 0.70。"
    )
    # 全集 baseline 三区间诊断（心理席）：≥0.80 第二源初步成立(提纯只升)；0.70-0.80 焦虑噪声压低
    # =提纯必要实证；<0.70 须换源或追加 Emotion-Stimulus。fear 解锁终裁属议会门（B-5·不在此裁）。
    f_verdict = (
        "fear≥0.80(全集即过·第二源初步成立)"
        if f_lb >= 0.80
        else "fear 0.70-0.80(焦虑噪声压低·提纯必要)"
        if f_lb >= 0.70
        else "fear<0.70(须换源/追加交叉源)"
    )
    logger.info("ISEAR 第二生存源全 baseline（仅验证·非解锁裁决·回议会 B-1/B-5）：%s", f_verdict)
    print(
        f"ISEAR-OOD: anger {a_ok}/{len(a_pred)} lb={a_lb:.3f} | "
        f"fear {f_ok}/{len(f_pred)} lb={f_lb:.3f} => {f_verdict}"
    )


if __name__ == "__main__":
    main()
