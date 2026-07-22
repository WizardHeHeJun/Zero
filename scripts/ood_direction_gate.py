"""跨源 OOD 方向门（议会三轮 Q6）：GoEmotions 训的 direction 头在别源上验方向泛化。

议会 Q6 裁：只在 in-domain 过关不够，须**跨源 OOD**（防学到数据集捷径）——模型在训练源
之外的 anger/fear 上仍须**分侧** Wilson CI 下界≥0.70（每侧 n≥50），任一源/任一侧未过=门 FAIL。

本脚本对 EmpatheticDialogues（对话域·最贴 Zero live-chat）做 OOD 验证：
  anger 族(angry/furious/annoyed)→期望 logit>0；fear 族(afraid/terrified/anxious/apprehensive)
  →期望 logit<0。prompt 情境文本作输入（`_comma_` 还原逗号），按 conv_id 去重。

诚实说明：ED 是 CC BY-NC（仅作**验证**、不训练；验证用不产权重、无 license 传染），与训练用的
GoEmotions(Apache-2.0) 分离。crowd-enVENT 作第三源可另跑。

用法：python -m scripts.ood_direction_gate --ed data/raw/ed/empatheticdialogues/test.csv \
  --weights artifacts/motivational_direction_prior.pt
"""

from __future__ import annotations

import argparse
import csv
import logging
import math

import torch

from scripts.train_direction_head import DirectionHead
from src.agents.models.text_affect_regressor_st import DEFAULT_ENCODER, ST_FEATURE_DIM, encode_texts

logger = logging.getLogger(__name__)

ANGER_FAMILY = {"angry", "furious", "annoyed"}
FEAR_FAMILY = {"afraid", "terrified", "anxious", "apprehensive"}


def _wilson_lb(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - margin) / denom


def _read_ed(path: str) -> tuple[list[str], list[str]]:
    """读 ED test.csv，按 conv_id 去重，返回 (prompt 情境文本, +/−) 极性（仅 anger/fear 族）。"""
    seen: set[str] = set()
    texts: list[str] = []
    polarity: list[str] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cid = row["conv_id"]
            if cid in seen:
                continue
            emo = row["context"].strip().lower()
            if emo in ANGER_FAMILY:
                pol = "+"
            elif emo in FEAR_FAMILY:
                pol = "-"
            else:
                continue
            seen.add(cid)
            texts.append(row["prompt"].replace("_comma_", ","))
            polarity.append(pol)
    return texts, polarity


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="EmpatheticDialogues OOD 方向门（分侧 Wilson≥0.70）")
    p.add_argument("--ed", default="data/raw/ed/empatheticdialogues/test.csv")
    p.add_argument("--weights", default="artifacts/motivational_direction_prior.pt")
    args = p.parse_args()

    texts, pol = _read_ed(args.ed)
    model = DirectionHead(ST_FEATURE_DIM)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logit = model(encode_texts(texts, encoder=DEFAULT_ENCODER)).tolist()

    a_pred = [logit[i] for i in range(len(pol)) if pol[i] == "+"]
    f_pred = [logit[i] for i in range(len(pol)) if pol[i] == "-"]
    a_ok = sum(1 for v in a_pred if v > 0)
    f_ok = sum(1 for v in f_pred if v < 0)
    a_lb, f_lb = _wilson_lb(a_ok, len(a_pred)), _wilson_lb(f_ok, len(f_pred))

    logger.info("─" * 62)
    logger.info("EmpatheticDialogues OOD 方向门（对话域·跨源；每侧 n≥50 且 Wilson 下界≥0.70）:")
    logger.info(
        "  anger 族 %d/%d=%.1f%%  Wilson 下界 %.4f",
        a_ok,
        len(a_pred),
        a_ok / len(a_pred) * 100,
        a_lb,
    )
    logger.info(
        "  fear  族 %d/%d=%.1f%%  Wilson 下界 %.4f",
        f_ok,
        len(f_pred),
        f_ok / len(f_pred) * 100,
        f_lb,
    )
    passed = len(a_pred) >= 50 and len(f_pred) >= 50 and a_lb >= 0.70 and f_lb >= 0.70
    verdict = "PASS(OOD 两侧下界≥0.70)" if passed else "FAIL(某侧下界<0.70)"
    logger.info("─" * 62)
    logger.info("EmpatheticDialogues 跨源 OOD：%s", verdict)
    print(f"OOD-GATE ed: anger_lb={a_lb:.3f} fear_lb={f_lb:.3f} => {verdict}")


if __name__ == "__main__":
    main()
