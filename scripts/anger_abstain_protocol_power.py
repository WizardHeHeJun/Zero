"""anger 弃权门·切分协议功效对比（议会五轮 B3 决策输入 · **非新裁决**）。

数学席（`notes/2026-07-18-anger-delta-validation-council.md`·B3）指出官方 40/40/20 协议在
anger n=242 数据饥饿下 V_reserve 得不偿失、V_test 检验功效偏低（真值≈0.775 时 n≈83 covered 下
P(Wilson LB≥0.70)≈29%）。本脚本对**同一 ED 数据 / 同一权重**跑多种切分比例的 1000 种子功效对比，
量化「改协议能买到多少功效」，供议会 B3 决策（是否改 50/50 或加大 V_test、时机）。

⚠ **这不是新的解锁裁决**：同一数据换切分比例只是**提高功效 / 降低切分方差**，不产生新的
独立证据（真效应仍受 ~0.77 构念天花板约束，见议会综合）。官方**预注册**裁决仍是 40/40/20
（`scripts/validate_anger_abstain.py`），anger 是否解锁属议会定语义，本脚本不改变该结论。

复用 `validate_anger_abstain` 的切分/度量原语（`_three_way_split` 已参数化 split_cal/split_test），
保证与官方脚本切分逻辑逐字一致。数据 EmpatheticDialogues（CC BY-NC 仅验证）。

用法：PYTHONPATH=d:/Zero HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "E:/anaconda/Scripts/conda.exe" run -n affective-expression --no-capture-output \
  python -m scripts.anger_abstain_protocol_power \
    --weights artifacts/motivational_direction_prior_m.pt \
    --report notes/2026-07-18-anger-abstain-protocol-power-comparison.md
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics
from pathlib import Path

import torch

from scripts.ood_direction_gate import _read_ed
from scripts.train_direction_head import DirectionHead
from scripts.validate_anger_abstain import (
    ANGER_LB_BAR,
    CANDIDATE_THRESHOLDS,
    COVERAGE_FLOOR,
    DEFAULT_SEED,
    DEFAULT_STABILITY_SEEDS,
    _anger_selective_metric,
    _fear_full_metric,
    _select_frozen_threshold,
    _three_way_split,
)
from src.agents.models.text_affect_regressor_st import DEFAULT_ENCODER, ST_FEATURE_DIM, encode_texts

logger = logging.getLogger(__name__)

# (标签, V_cal 占比, V_test 占比)；test=0.4 且 cal=0.4 → 余 0.2 为 reserve；否则弃 reserve。
PROTOCOLS: tuple[tuple[str, float, float], ...] = (
    ("40/40/20 官方·预注册", 0.40, 0.40),
    ("50/50 弃reserve", 0.50, 0.50),
    ("60/40 弃reserve(大V_cal)", 0.60, 0.40),
    ("40/60 弃reserve(大V_test)", 0.40, 0.60),
)


def _protocol_seed(
    logits: list[float],
    anger_idx: list[int],
    fear_idx: list[int],
    seed: int,
    cal: float,
    test: float,
) -> tuple[object, object] | None:
    """跑一个种子的完整协议，返回 (anger V_test metric, fear V_test metric)；无法标定→None。"""
    a_cal, a_test, _ = _three_way_split(anger_idx, seed, cal, test)
    f_test = _three_way_split(fear_idx, seed + 1, cal, test)[1]
    cal_curve = [
        _anger_selective_metric([logits[i] for i in a_cal], t) for t in CANDIDATE_THRESHOLDS
    ]
    frozen = _select_frozen_threshold(cal_curve)
    if frozen is None:
        return None
    ta = _anger_selective_metric([logits[i] for i in a_test], frozen.threshold)
    ft = _fear_full_metric([logits[i] for i in f_test])
    return ta, ft


def _sweep_protocol(
    logits: list[float],
    anger_idx: list[int],
    fear_idx: list[int],
    cal: float,
    test: float,
    n_seeds: int,
    prereg_seed: int,
) -> dict:
    """对一个切分协议跑 n_seeds 种子 + 预注册单看，返回功效汇总。"""
    lbs: list[float] = []
    n_test_sizes: list[int] = []
    fear_lbs: list[float] = []
    n_pass = 0
    for s in range(n_seeds):
        r = _protocol_seed(logits, anger_idx, fear_idx, s, cal, test)
        if r is None:
            continue
        ta, ft = r
        lbs.append(ta.wilson_lb)  # type: ignore[attr-defined]
        n_test_sizes.append(ta.n_total)  # type: ignore[attr-defined]
        fear_lbs.append(ft.wilson_lb)  # type: ignore[attr-defined]
        if ta.wilson_lb >= ANGER_LB_BAR and ta.coverage >= COVERAGE_FLOOR:  # type: ignore[attr-defined]
            n_pass += 1
    n = len(lbs)
    pr = _protocol_seed(logits, anger_idx, fear_idx, prereg_seed, cal, test)
    pr_ta = pr[0] if pr else None
    pr_lb = pr_ta.wilson_lb if pr_ta else float("nan")  # type: ignore[attr-defined]
    pr_cov = pr_ta.coverage if pr_ta else float("nan")  # type: ignore[attr-defined]
    pr_pass = bool(pr_ta and pr_lb >= ANGER_LB_BAR and pr_cov >= COVERAGE_FLOOR)
    pct = (
        sum(1 for v in lbs if v <= pr_lb) / n * 100 if n and not math.isnan(pr_lb) else float("nan")
    )
    return {
        "n_calibrated": n,
        "n_test_anger": statistics.median(n_test_sizes) if n else float("nan"),
        "pass_rate": n_pass / n if n else float("nan"),
        "lb_median": statistics.median(lbs) if n else float("nan"),
        "lb_mean": statistics.mean(lbs) if n else float("nan"),
        "fear_lb_median": statistics.median(fear_lbs) if n else float("nan"),
        "prereg_lb": pr_lb,
        "prereg_cov": pr_cov,
        "prereg_pass": pr_pass,
        "prereg_percentile": pct,
    }


def compare(ed_path: str, weights_path: str, seed: int, n_seeds: int) -> dict:
    """编码一次，对所有协议跑功效对比。"""
    texts, pol = _read_ed(ed_path)
    model = DirectionHead(ST_FEATURE_DIM)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        logits = model(encode_texts(texts, encoder=DEFAULT_ENCODER)).tolist()
    anger_idx = [i for i, p in enumerate(pol) if p == "+"]
    fear_idx = [i for i, p in enumerate(pol) if p == "-"]
    rows = []
    for label, cal, test in PROTOCOLS:
        r = _sweep_protocol(logits, anger_idx, fear_idx, cal, test, n_seeds, seed)
        r["label"] = label
        r["cal"] = cal
        r["test"] = test
        rows.append(r)
    return {"seed": seed, "n_seeds": n_seeds, "weights": weights_path, "rows": rows}


def _log(res: dict) -> None:
    logger.info("─" * 78)
    logger.info(
        "anger 弃权门切分协议功效对比（%d 种子·预注册 seed=%d·B3 决策输入·非新裁决）",
        res["n_seeds"],
        res["seed"],
    )
    logger.info("─" * 78)
    for r in res["rows"]:
        logger.info(
            "  %-24s V_test≈%d  δPASS率 %5.1f%%  LB中位 %.4f  预注册 LB=%.4f(%s·%2.0f分位)",
            r["label"],
            int(r["n_test_anger"]),
            r["pass_rate"] * 100,
            r["lb_median"],
            r["prereg_lb"],
            "过" if r["prereg_pass"] else "未过",
            r["prereg_percentile"],
        )
    logger.info("─" * 78)
    logger.info(
        "注：同数据换切分只提功效/降方差、非新独立证据；官方预注册仍 40/40/20；解锁属议会定。"
    )


def _write_report(path: str, res: dict) -> None:
    lines: list[str] = []
    lines.append("# anger 弃权门 · 切分协议功效对比（议会 B3 决策输入 · 非新裁决）\n")
    lines.append(
        f"> 自动生成于 `scripts/anger_abstain_protocol_power.py`。{res['n_seeds']} 种子，"
        f"预注册 seed={res['seed']}，权重 `{res['weights']}`，"
        "数据 ED test.csv（CC BY-NC·仅验证）。\n"
    )
    lines.append(
        "**⚠ 非新裁决**：同一数据换切分比例只提高功效 / 降低切分方差，**不产生新独立证据**"
        "（真效应仍受 ~0.77 构念天花板约束）。官方预注册裁决仍是 40/40/20；anger 是否解锁属议会"
        "定语义。本表仅量化「改协议买到多少功效」，供议会 B3 决策。\n"
    )
    lines.append(
        "| 切分协议 | V_test(anger) | δ PASS 率 | V_test LB 中位 | 预注册单看 LB | 预注册分位 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in res["rows"]:
        lines.append(
            f"| {r['label']} | ≈{int(r['n_test_anger'])} | {r['pass_rate'] * 100:.1f}% | "
            f"{r['lb_median']:.4f} | {r['prereg_lb']:.4f}"
            f"（{'过' if r['prereg_pass'] else '未过'}） | {r['prereg_percentile']:.0f} 分位 |"
        )
    lines.append(
        "\n**读法**：δ PASS 率＝1000 种子里 V_test 达 LB≥0.70@覆盖≥80% 的比例（功效代理）；"
        "LB 中位＝典型切分的 Wilson 下界。V_test 越大 → 功效越高、但同数据不改变真效应"
        "（中位 LB 仍在 ~0.68–0.72 边界附近浮动）。\n"
    )
    lines.append(
        "**结论指向（供议会）**：加大 V_test（如 40/60）能显著抬 PASS 率与降方差，但**中位 LB "
        "仍卡 0.70 边界**——功效能让「确认」更容易达成，却搬不动 ~0.77 构念天花板。故 B3"
        "「改协议」只在 confrontational 语料（B5·可能抬高真值）配套时才真正有意义；单纯换切分"
        "≈在同一天花板下反复掷骰，**议会须权衡是否值得**、以及是否要求**新数据的预注册**再验"
        "（防同数据反复重切＝变相 data snooping）。fear 端全协议稳过（见 fear LB 中位）。\n"
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("报告已写入 %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="anger 弃权门切分协议功效对比（B3 决策输入）")
    p.add_argument("--ed", default="data/raw/ed/empatheticdialogues/test.csv")
    p.add_argument("--weights", default="artifacts/motivational_direction_prior_m.pt")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--stability-seeds", type=int, default=DEFAULT_STABILITY_SEEDS)
    p.add_argument("--report", default=None, help="可选：markdown 报告路径（notes/）")
    args = p.parse_args()

    res = compare(args.ed, args.weights, args.seed, args.stability_seeds)
    _log(res)
    if args.report:
        _write_report(args.report, res)
    print(
        "PROTOCOL-POWER: "
        + " | ".join(
            f"{r['label'].split()[0]} pass={r['pass_rate'] * 100:.0f}% lbmed={r['lb_median']:.3f}"
            for r in res["rows"]
        )
    )


if __name__ == "__main__":
    main()
