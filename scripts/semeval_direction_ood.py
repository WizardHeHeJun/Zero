"""SemEval-2018 Task1 E-c 跨源 OOD 方向门（议会 B5 · 2026-07-20 · **仅验证**）。

议会 B5 综合裁定（`notes/2026-07-20-anger-b5-proxy-mapping-council.md`）：SemEval-2018 作
**validation-only** 第二 OOD 源（research-only license · **不训练** · 同 ED 定位 · 零 license
入训包袱）。这是数学席点出的**真杠杆**——E-c English 测试集 signed anger≈962，full-coverage
Wilson LB≥0.70 的功效≈100%（远超 ED 的 n=242、24% 稳健率）；且**原生 anger/fear 金标·零代理
噪声**（区别于 TRAC-1 aggression 代理，后者被议会 BLOCK-1 禁作验证）。

直接实测「更对抗的 Twitter 语料是否给出比 ED（叙事/倾诉体裁·anger 天花板 ~74%/LB0.68）更高、
更可稳健确认的 anger 方向真值」。signed 子集（仿 `train_direction_head._read_goemotions_signed`）：
anger=1&fear=0→期望 logit>0（+）；fear=1&anger=0→期望 logit<0（−）；both/neither 剔除。
full-coverage 分侧报 Wilson CI 下界（每侧 n≥50 且 LB≥0.70 过；不合并、防高分侧掩盖低分侧）。

数据 `data/raw/semeval2018_task1/E-c-En-test-gold.txt`（本地 gitignore · research-only 仅验证）。
诚实说明：SemEval license research-only（同 ED CC BY-NC）——只作验证不训练、不产权重、无传染。

用法：PYTHONPATH=d:/Zero HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "E:/anaconda/Scripts/conda.exe" run -n affective-expression --no-capture-output \
  python -m scripts.semeval_direction_ood --weights artifacts/motivational_direction_prior_m.pt
"""

from __future__ import annotations

import argparse
import logging

import torch

from scripts.direction_gate_text_coping import wilson_lower_bound
from scripts.train_direction_head import DirectionHead
from src.agents.models.text_affect_regressor_st import DEFAULT_ENCODER, ST_FEATURE_DIM, encode_texts

logger = logging.getLogger(__name__)


def _read_semeval_ec(path: str) -> tuple[list[str], list[str]]:
    """读 SemEval E-c test-gold TSV：anger=1&fear=0→'+'、fear=1&anger=0→'-'（signed 子集）。

    E-c 是多标签情绪分类（11 类含 anger/fear 布尔金标）。取互斥的 anger/fear 单标样本构成
    双极方向验证集（both/neither 剔除，与 _read_goemotions_signed 同口径）。
    """
    texts: list[str] = []
    pol: list[str] = []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        ai, fi, ti = header.index("anger"), header.index("fear"), header.index("Tweet")
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) <= max(ai, fi, ti):
                continue
            a, f = c[ai] == "1", c[fi] == "1"
            if a and not f:
                pol.append("+")
            elif f and not a:
                pol.append("-")
            else:
                continue
            texts.append(c[ti])
    return texts, pol


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="SemEval-2018 E-c 跨源 OOD 方向门（B5·仅验证）")
    p.add_argument("--data", default="data/raw/semeval2018_task1/E-c-En-test-gold.txt")
    p.add_argument("--weights", default="artifacts/motivational_direction_prior_m.pt")
    args = p.parse_args()

    texts, pol = _read_semeval_ec(args.data)
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

    logger.info("─" * 68)
    logger.info("SemEval-2018 E-c 跨源 OOD 方向门（Twitter·当下时态·原生金标·仅验证）:")
    logger.info(
        "  anger 侧 %d/%d=%.1f%%  Wilson 下界 %.4f",
        a_ok,
        len(a_pred),
        a_ok / len(a_pred) * 100,
        a_lb,
    )
    logger.info(
        "  fear  侧 %d/%d=%.1f%%  Wilson 下界 %.4f",
        f_ok,
        len(f_pred),
        f_ok / len(f_pred) * 100,
        f_lb,
    )
    passed = len(a_pred) >= 50 and len(f_pred) >= 50 and a_lb >= 0.70 and f_lb >= 0.70
    verdict = "PASS(两侧 n≥50 且 LB≥0.70)" if passed else "FAIL(某侧 n<50 或 LB<0.70)"
    logger.info("─" * 68)
    logger.info("对照 ED（叙事/倾诉体裁）：anger ~74%/LB≈0.68（撞构念天花板）、fear ~93%/LB≈0.90。")
    logger.info("SemEval 跨源 OOD（仅验证·非解锁裁决·anger 解锁仍属议会定）：%s", verdict)
    print(
        f"SEMEVAL-OOD: anger {a_ok}/{len(a_pred)} lb={a_lb:.3f} | "
        f"fear {f_ok}/{len(f_pred)} lb={f_lb:.3f} => {verdict}"
    )


if __name__ == "__main__":
    main()
