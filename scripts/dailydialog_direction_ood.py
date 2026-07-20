"""DailyDialog 跨源 OOD 方向门（议会 B2 T-3·第三源·2026-07-20·仅验证）。

承接 fear 域脆弱重估（notes/2026-07-20-fear-domain-fragility-council.md）与 B2 设计门裁决
（notes/2026-07-20-fear-domain-activation-b2-council.md·T-3）：fear 三源 ED 生存叙事 0.90 →
SemEval 社交焦虑 0.709 → EmoryNLP 表演对话 0.264 崩。**EmoryNLP 是表演式对话（Friends 演员台词）**，
其 fear 崩不能排除 performative 偏置。B2 council T-3 把 DailyDialog 第三源从「可选加分」升为
**fear 生产解锁前置的域稳定性验证**。

本脚本用 **DailyDialog**（人写**自然日常对话**·非表演·CC BY-NC-SA·**仅验证不训练**·license 传染禁）
作第三 OOD 源，回答两问：
  ① fear 对话域崩塌是 EmoryNLP 表演偏置的 artifact，还是自然对话域 fear 本就不可靠？
     —— DailyDialog fear 若也低 → 坐实「对话域 fear 不可靠·非表演 artifact」→ 支持 fear 域条件化
        （对话域默认关·仅 survival_narrative 生存叙事域解锁）；DailyDialog fear 若高 → EmoryNLP 崩是
        表演 artifact·须重估。
  ② 量化对话域 π_t(fear)（B-pi·B3 阶段用）。

⚠ 关键：DailyDialog 是**对话域**（≈日常社交·近社交焦虑型 fear），**非 survival_narrative 生存叙事
域**。它验证的是「对话域 fear 不可靠这一域条件化前提是否稳」，**不新增 survival_narrative 源**
（ED 仍是 survival 域唯一源·单源 OOD 局限见数学席 §4）。fear 生产解锁最终仍属议会定。

标签（DailyDialog 官方·0-6）：1=anger→期望 logit>0(+)、3=fear→期望 logit<0(−)；其余跳过。
full-coverage 分侧 Wilson LB（bar：两侧 n≥50 且 LB≥0.70·与 SemEval/EmoryNLP 同口径）。
权重 motivational_direction_prior_m.pt（GoEmotions+crowd-enVENT·DailyDialog 是真 OOD）。
数据 data/raw/dailydialog/<split>/dialogues[_emotion]_<split>.txt（本地 gitignore·仅验证不训练）。

用法：PYTHONPATH=d:/Zero HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "E:/anaconda/Scripts/conda.exe" run -n affective-expression --no-capture-output \
  python -m scripts.dailydialog_direction_ood --weights artifacts/motivational_direction_prior_m.pt
"""

from __future__ import annotations

import argparse
import logging

import torch

from scripts.direction_gate_text_coping import wilson_lower_bound
from scripts.train_direction_head import DirectionHead
from src.agents.models.text_affect_regressor_st import DEFAULT_ENCODER, ST_FEATURE_DIM, encode_texts

logger = logging.getLogger(__name__)

# DailyDialog 官方情绪标签编码（0=no emotion·1=anger·2=disgust·3=fear·
# 4=happiness·5=sadness·6=surprise）
_ANGER = "1"
_FEAR = "3"


def _read_dailydialog(
    text_paths: list[str], emotion_paths: list[str]
) -> tuple[list[str], list[str]]:
    """读 DailyDialog：文本行按 __eou__ 分句，与 emotion 行逐 utterance 对齐；anger→'+'、fear→'-'。

    对齐异常行（utterance 数≠标签数）整行跳过，防错位污染（诚实丢弃·非静默截断）。
    """
    texts: list[str] = []
    pol: list[str] = []
    skipped = 0
    for tpath, epath in zip(text_paths, emotion_paths, strict=True):
        with open(tpath, encoding="utf-8") as tf, open(epath, encoding="utf-8") as ef:
            for tline, eline in zip(tf, ef, strict=False):
                utts = [u.strip() for u in tline.strip().split("__eou__") if u.strip()]
                emos = eline.strip().split()
                if len(utts) != len(emos):
                    skipped += 1
                    continue
                for utt, emo in zip(utts, emos, strict=True):
                    if emo == _ANGER:
                        pol.append("+")
                        texts.append(utt)
                    elif emo == _FEAR:
                        pol.append("-")
                        texts.append(utt)
    if skipped:
        logger.warning("DailyDialog 对齐异常整行跳过 %d 行（utterance 数≠标签数）", skipped)
    return texts, pol


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="DailyDialog 跨源 OOD 方向门（B2 T-3 第三源·仅验证）")
    p.add_argument(
        "--text",
        nargs="+",
        default=[
            "data/raw/dailydialog/train/dialogues_train.txt",
            "data/raw/dailydialog/validation/dialogues_validation.txt",
            "data/raw/dailydialog/test/dialogues_test.txt",
        ],
    )
    p.add_argument(
        "--emotion",
        nargs="+",
        default=[
            "data/raw/dailydialog/train/dialogues_emotion_train.txt",
            "data/raw/dailydialog/validation/dialogues_emotion_validation.txt",
            "data/raw/dailydialog/test/dialogues_emotion_test.txt",
        ],
    )
    p.add_argument("--weights", default="artifacts/motivational_direction_prior_m.pt")
    args = p.parse_args()

    texts, pol = _read_dailydialog(args.text, args.emotion)
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
    logger.info("DailyDialog 跨源 OOD 方向门（人写自然日常对话·非表演·原生金标·仅验证·第三源）:")
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
    passed = len(a_pred) >= 50 and len(f_pred) >= 50 and a_lb >= 0.70 and f_lb >= 0.70
    verdict = "PASS(两侧 n≥50 且 LB≥0.70)" if passed else "FAIL(某侧 n<50 或 LB<0.70)"
    logger.info("─" * 70)
    logger.info(
        "对照三源：ED 生存叙事 anger 74%/fear 93%(LB0.90)；SemEval Twitter anger 88%/0.857·"
    )
    logger.info("  fear 76%/0.709；EmoryNLP 表演对话 anger 80%/0.776·fear 28%/0.264(崩)。")
    logger.info(
        "解读：DailyDialog=自然对话域。fear 若低 → 对话域 fear 不可靠非表演 artifact（支持域条件化·"
    )
    logger.info(
        "  对话域默认关）；fear 若高 → EmoryNLP 崩系表演 artifact 须重估。fear 生产解锁仍属议会定。"
    )
    logger.info("DailyDialog 第三源（仅验证·非解锁裁决）：%s", verdict)
    print(
        f"DAILYDIALOG-OOD: anger {a_ok}/{len(a_pred)} lb={a_lb:.3f} | "
        f"fear {f_ok}/{len(f_pred)} lb={f_lb:.3f} => {verdict}"
    )


if __name__ == "__main__":
    main()
