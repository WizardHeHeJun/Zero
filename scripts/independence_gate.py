"""独立性门 r(V_pred, D_pred)≤0.50（议会 coping 原始红线 · text_coping 解锁前置之二）。

议会（coping_potential 原始设计门 + 符号监督重训）定：text→方向先验作**独立低精度标量流**，
绝不与内核 VA 共线（重蹈 PAD-D 被批评的症结）。解锁 anger 热路径的 gate 顺序＝
方向门（分源 Wilson≥0.70）→ **独立性门 r(V_pred,D_pred)≤0.50** → 解锁 `ZERO_TEXT_COPING_ENABLED`。

本脚本在一批多样文本上算 valence 预测（STTextAffectRegressor·output_dim=2）与方向预测
（DirectionHead 符号监督头·raw logit / tanh）的 Pearson r，|r|≤0.50 即门通过（0.45–0.50 精度减半·
>0.50 落选项 C）。默认用 SemEval E-c 全测试集（3259 条·覆盖 11 类情绪·非仅 anger/fear，更能测
全域正交性）。

原理：anger 与 fear 同处负价（v<0），方向头只按趋近/回避**符号**分离二者——故应对 valence
近乎无信息、r≈0，正是独立标量流所需。这与旧 SAM-D 路线相 1 的 r=0.18 强通过一脉相承。

用法：PYTHONPATH=d:/Zero HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "E:/anaconda/Scripts/conda.exe" run -n affective-expression --no-capture-output \
  python -m scripts.independence_gate --weights artifacts/motivational_direction_prior_m.pt
"""

from __future__ import annotations

import argparse
import logging
import statistics

import torch

from scripts.train_direction_head import DirectionHead
from src.agents.models.text_affect_regressor_st import (
    DEFAULT_ENCODER,
    ST_FEATURE_DIM,
    encode_texts,
    load_st_text_affect_regressor,
)

logger = logging.getLogger(__name__)

INDEP_BAR = 0.50


def _read_tweets(path: str) -> list[str]:
    """读 SemEval E-c TSV 的 Tweet 列（全类·作多样文本集测正交性）。"""
    texts: list[str] = []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        ti = header.index("Tweet")
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) > ti:
                texts.append(c[ti])
    return texts


def _pearson(x: list[float], y: list[float]) -> float:
    mx, my = statistics.mean(x), statistics.mean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="独立性门 r(V_pred,D_pred)≤0.50（解锁前置之二）")
    p.add_argument("--data", default="data/raw/semeval2018_task1/E-c-En-test-gold.txt")
    p.add_argument("--va-weights", default="artifacts/text_affect_regressor_st.pt")
    p.add_argument("--weights", default="artifacts/motivational_direction_prior_m.pt")
    args = p.parse_args()

    texts = _read_tweets(args.data)
    feats = encode_texts(texts, encoder=DEFAULT_ENCODER)

    va = load_st_text_affect_regressor(args.va_weights, output_dim=2)
    with torch.no_grad():
        va_out = va(feats)
    v_pred = va_out[:, 0].tolist()

    dh = DirectionHead(ST_FEATURE_DIM)
    dh.load_state_dict(torch.load(args.weights, map_location="cpu"))
    dh.eval()
    with torch.no_grad():
        logit = dh(feats).tolist()
    d_tanh = torch.tanh(torch.tensor(logit)).tolist()

    r_logit = _pearson(v_pred, logit)
    r_tanh = _pearson(v_pred, d_tanh)
    worst = max(abs(r_logit), abs(r_tanh))

    logger.info("─" * 62)
    logger.info("独立性门 r(V_pred, D_pred)（%d 条多样文本；|r|≤%.2f 过）:", len(texts), INDEP_BAR)
    logger.info("  r(V_pred, D_logit) = %+.4f", r_logit)
    logger.info("  r(V_pred, D_tanh)  = %+.4f", r_tanh)
    passed = worst <= INDEP_BAR
    verdict = (
        f"PASS(|r|={worst:.4f}≤{INDEP_BAR:.2f}·正交)"
        if passed
        else f"FAIL(|r|={worst:.4f}>{INDEP_BAR:.2f})"
    )
    logger.info("─" * 62)
    logger.info("独立性门（独立低精度标量流不与内核 VA 共线）：%s", verdict)
    print(f"INDEP-GATE: r(V,D_logit)={r_logit:+.3f} r(V,D_tanh)={r_tanh:+.3f} => {verdict}")


if __name__ == "__main__":
    main()
