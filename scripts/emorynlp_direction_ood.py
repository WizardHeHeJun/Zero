"""EmoryNLP 跨源 OOD 方向门（议会 B5·第二 confrontational 源·2026-07-20·仅验证）。

议会 anger 解锁裁决（notes/2026-07-20-anger-unlock-decision-council.md）与数学/神经席点出
**单源局限**：SemEval anger LB=0.857 是单一 OOD 源（Twitter 微博），须再一个 confrontational
源交叉、排除 Twitter 媒体/performative-anger 混淆、确认「可行动性在线→anger 底物在线」跨媒介稳。

本脚本用 **EmoryNLP**（Friends 情景喜剧·**口语对话体裁**·与 Twitter 微博不同媒介）作第二 OOD 源：
原生情绪金标 Mad(≈anger·1332)/Scared(≈fear·1645)、Apache-2.0（license 干净·可训练·本处仅验证）。
诚实局限：EmoryNLP 是**表演式**对话（演员台词），与 Twitter outrage-culture 同属「表演型」——
它测「不同媒介（口语 TV 对话 vs 书面微博）」是否复现 anger 可靠性，但**不完全排除** performative；
真正排除 performative 需自然对话语料（如 DailyDialog·下轮）。

Mad→期望 logit>0(+)、Scared→期望 logit<0(−)（单标签·无重叠）。full-coverage 分侧 Wilson LB。
权重 motivational_direction_prior_m.pt（GoEmotions+crowd-enVENT·EmoryNLP 是真 OOD）。数据
data/raw/emorynlp/*.json（本地 gitignore·Apache·仅验证不训练）。

用法：PYTHONPATH=d:/Zero HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "E:/anaconda/Scripts/conda.exe" run -n affective-expression --no-capture-output \
  python -m scripts.emorynlp_direction_ood --weights artifacts/motivational_direction_prior_m.pt
"""

from __future__ import annotations

import argparse
import json
import logging

import torch

from scripts.direction_gate_text_coping import wilson_lower_bound
from scripts.train_direction_head import DirectionHead
from src.agents.models.text_affect_regressor_st import DEFAULT_ENCODER, ST_FEATURE_DIM, encode_texts

logger = logging.getLogger(__name__)


def _read_emorynlp(paths: list[str]) -> tuple[list[str], list[str]]:
    """读 EmoryNLP JSON（episodes→scenes→utterances）：Mad→'+'、Scared→'-'（单标签）。"""
    texts: list[str] = []
    pol: list[str] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for ep in data.get("episodes", []):
            for scene in ep.get("scenes", []):
                for utt in scene.get("utterances", []):
                    emo = utt.get("emotion")
                    if emo == "Mad":
                        pol.append("+")
                    elif emo == "Scared":
                        pol.append("-")
                    else:
                        continue
                    texts.append(utt["transcript"])
    return texts, pol


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="EmoryNLP 跨源 OOD 方向门（B5 第二源·仅验证）")
    p.add_argument(
        "--data",
        nargs="+",
        default=[
            "data/raw/emorynlp/emotion-detection-trn.json",
            "data/raw/emorynlp/emotion-detection-dev.json",
            "data/raw/emorynlp/emotion-detection-tst.json",
        ],
    )
    p.add_argument("--weights", default="artifacts/motivational_direction_prior_m.pt")
    args = p.parse_args()

    texts, pol = _read_emorynlp(args.data)
    model = DirectionHead(ST_FEATURE_DIM)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logit = model(encode_texts(texts, encoder=DEFAULT_ENCODER)).tolist()

    a_pred = [logit[i] for i in range(len(pol)) if pol[i] == "+"]
    f_pred = [logit[i] for i in range(len(pol)) if pol[i] == "-"]
    a_ok = sum(1 for v in a_pred if v > 0)
    f_ok = sum(1 for v in f_pred if v < 0)
    a_lb, f_lb = wilson_lower_bound(a_ok, len(a_pred)), wilson_lower_bound(f_ok, len(f_pred))

    logger.info("─" * 70)
    logger.info(
        "EmoryNLP 跨源 OOD 方向门（Friends 口语对话·原生金标·仅验证·第二 confrontational 源）:"
    )
    logger.info(
        "  anger(Mad)   %d/%d=%.1f%%  Wilson 下界 %.4f",
        a_ok,
        len(a_pred),
        a_ok / len(a_pred) * 100,
        a_lb,
    )
    logger.info(
        "  fear(Scared) %d/%d=%.1f%%  Wilson 下界 %.4f",
        f_ok,
        len(f_pred),
        f_ok / len(f_pred) * 100,
        f_lb,
    )
    passed = len(a_pred) >= 50 and len(f_pred) >= 50 and a_lb >= 0.70 and f_lb >= 0.70
    verdict = "PASS(两侧 n≥50 且 LB≥0.70)" if passed else "FAIL(某侧 n<50 或 LB<0.70)"
    logger.info("─" * 70)
    logger.info(
        "对照：SemEval Twitter anger 88%/LB0.857·fear 76%/0.709；ED 叙事 anger 74%·fear 93%。"
    )
    logger.info("EmoryNLP 第二源（仅验证·非解锁裁决·anger 解锁属议会定）：%s", verdict)
    print(
        f"EMORYNLP-OOD: anger {a_ok}/{len(a_pred)} lb={a_lb:.3f} | "
        f"fear {f_ok}/{len(f_pred)} lb={f_lb:.3f} => {verdict}"
    )


if __name__ == "__main__":
    main()
