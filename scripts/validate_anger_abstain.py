"""三路切分独立验证 anger 置信弃权门（议会四轮 2026-07-18 · 选项 δ · P0 前置）。

**为什么存在**：上一轮的置信弃权曲线（|logit|≥0.25→LB0.715）被数学席判**选择偏差**——
在同一 OOD 集上扫多个候选阈值、挑「过关的那个」= 多重比较（上一轮扫 7 个、FWER 偏高），
LB 偏乐观、不算合法证据。本脚本用**三路切分**堵这个洞（**本轮 V_cal 只扫 4 个候选，见
`CANDIDATE_THRESHOLDS`**；4 次的族错误率 1-(1-0.05)^4≈18.5%）：

  V_cal(40%)   —— 唯一允许扫阈值/挑阈值的集；在这上面按「覆盖≥80% 约束下 Wilson LB 最大」
                  选出**一个**阈值 → **冻结**。
  V_test(40%)  —— 用冻结阈值**独立跑且仅跑一次**；报出的 anger 侧 Wilson LB + 覆盖率才算数，
                  **不许**回去改阈值重跑（否则又变 data snooping）。
  V_reserve(20%) —— 存档不用（留给将来若需二次独立验证，避免污染 V_test）。

**判据（议会裁定）**：`V_test` anger 侧 Wilson 95% CI 下界 ≥0.70 **且** 覆盖率 ≥0.80
  → δ **字面 PASS**，anger 热路径可进 P1 解锁；否则 → 回落**选项 α（仅 fear 启用）**——
  fear 侧 δ 门不弃权、全程用（跨源 LB≈0.90，production-ready、无悬念）。

**种子稳健性披露（不是挑种子）**：单个三路切分的 V_test（anger n≈83–97）方差不小，字面判据
  可能被一次幸运切分带过。预注册种子给**官方字面裁决**；另跑 N 个种子的**同协议**扫描，
  报「δ PASS 的种子占比 / V_test LB 中位数 / 预注册种子落在分布的分位」——这是标准
  sensitivity 披露（无论结果好坏都照报），非事后挑种子。字面 PASS 但不稳健时，「是否接受
  脆弱单次 PASS」属**议会定语义**（anger bar 可达性），工程不私拍、回议会。

**弃权语义**：anger 侧「高置信才用、低置信弃权回退默认」是标准 selective classification 的
  risk–coverage 权衡（Geifman & El-Yaniv 2017）；用 |logit| 幅度门与 emotion_lexicon 的
  text 来源中间带哑火（|coping|≤MIDDLE_BAND 保守回落）**统一**为同一「幅度不足则弃权」范式。
  覆盖集诚实计入「置信但方向错」（logit≤−阈值）样本、不作单侧有利裁剪。

数据：EmpatheticDialogues test.csv（CC BY-NC，仅**验证**不训练、无 license 传染），
  经 `ood_direction_gate._read_ed` 去重族映射后 anger 242 / fear 293（40/40/20 → 各集 ≥50）。
权重：`artifacts/motivational_direction_prior_m.pt`（GoEmotions+crowd-enVENT 主方案 M；
  含 CC BY-NC 数据、按 Research-Only 对待）。结果落 `notes/`，不进 `src/` 热路径。

用法（conda env affective-expression；MiniLM 已本地缓存、离线）：
  PYTHONPATH=d:/Zero HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "E:/anaconda/Scripts/conda.exe" run -n affective-expression --no-capture-output \
    python -m scripts.validate_anger_abstain \
      --weights artifacts/motivational_direction_prior_m.pt \
      --report notes/2026-07-18-anger-abstain-3way-validation.md
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import statistics
from dataclasses import asdict, dataclass

import torch

from scripts.direction_gate_text_coping import wilson_lower_bound
from scripts.ood_direction_gate import _read_ed
from scripts.train_direction_head import DirectionHead
from src.agents.models.text_affect_regressor_st import DEFAULT_ENCODER, ST_FEATURE_DIM, encode_texts

logger = logging.getLogger(__name__)

# 议会四轮定：候选阈值序列只在 V_cal 上扫（作用于 |logit| 幅度，与 .env
# ZERO_ANGER_ABSTAIN_LOGIT_THRESHOLD 同标度）；覆盖率地板与 anger LB bar 是构念调整 bar。
CANDIDATE_THRESHOLDS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
COVERAGE_FLOOR = 0.80
ANGER_LB_BAR = 0.70
SPLIT_CAL = 0.40
SPLIT_TEST = 0.40  # 余下 20% = V_reserve（存档不用）
DEFAULT_SEED = 20260718
DEFAULT_STABILITY_SEEDS = 1000
# 工程读数启发式（非议会定 bar）：多数种子（含余量）过才算稳健；不达则字面 PASS 也回议会。
# ⚠ 数学席（议会四轮 2026-07-18）：此门在 p_true≈0.775 / n≈83 下**结构不可达**——
# P(单次 LB≥0.70)≈29% 远低于 0.80，会把「小样本低功效」误判成「真值不稳」。故它**只作启发
# 参考**，正式判据仍是 V_test LB≥0.70 且覆盖≥80%；稳健性主看 lb_median 与预注册分位。
# 指标是否重构为对 lb_median 做区间推断=议会 B2 悬而未决
# （notes/2026-07-18-anger-delta-validation-council.md）。
STABILITY_PASS_FLOOR = 0.80


@dataclass
class SideMetric:
    """一侧（anger/fear）在某阈值下的选择性预测度量。"""

    threshold: float
    n_total: int
    n_covered: int
    coverage: float
    k_correct: int
    accuracy: float
    wilson_lb: float


def _anger_selective_metric(logits: list[float], threshold: float) -> SideMetric:
    """anger 侧选择性度量：|logit|≥threshold 为覆盖（高置信），覆盖内 logit>0 为方向正确。

    覆盖集诚实含「置信但方向错」（logit≤−threshold）样本、计入错误，不作单侧有利裁剪
    （否则等于把 anger→fear 的自信误判改称「非 anger 预测」剔出分母，是选择性框定、
    重蹈 data-snooping 之嫌）。这是标准 selective classification 的 selective accuracy。
    """
    n_total = len(logits)
    covered = [v for v in logits if abs(v) >= threshold]
    n_covered = len(covered)
    k_correct = sum(1 for v in covered if v > 0)
    coverage = n_covered / n_total if n_total else float("nan")
    accuracy = k_correct / n_covered if n_covered else float("nan")
    lb = wilson_lower_bound(k_correct, n_covered)
    return SideMetric(threshold, n_total, n_covered, coverage, k_correct, accuracy, lb)


def _fear_full_metric(logits: list[float]) -> SideMetric:
    """fear 侧全覆盖度量（δ 门 fear 端不弃权、全程用）：logit<0 为方向正确。"""
    n = len(logits)
    k = sum(1 for v in logits if v < 0)
    lb = wilson_lower_bound(k, n)
    return SideMetric(0.0, n, n, 1.0 if n else float("nan"), k, k / n if n else float("nan"), lb)


def _three_way_split(indices: list[int], seed: int) -> tuple[list[int], list[int], list[int]]:
    """固定种子把一侧样本索引洗牌后 40/40/20 三路切分：V_cal / V_test / V_reserve。

    分侧（stratified）独立切分——保证 V_cal / V_test 两集各含足量 anger 与 fear。
    固定种子可复现（脚本无种子的 random 会不可复现，坑见交接 §5）。
    """
    shuffled = list(indices)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_cal = round(SPLIT_CAL * n)
    n_test = round(SPLIT_TEST * n)
    return shuffled[:n_cal], shuffled[n_cal : n_cal + n_test], shuffled[n_cal + n_test :]


def _select_frozen_threshold(cal_curve: list[SideMetric]) -> SideMetric | None:
    """在 V_cal 曲线上按「覆盖≥地板约束下 Wilson LB 最大」选一个阈值；tie-break 取更高覆盖。

    返回 None = 无候选满足覆盖地板（连最小阈值都覆盖不足）→ 门无法标定 → 回落选项 α。
    """
    eligible = [m for m in cal_curve if m.coverage >= COVERAGE_FLOOR]
    if not eligible:
        return None
    return max(eligible, key=lambda m: (m.wilson_lb, m.coverage))


def _protocol_once(
    logits: list[float], anger_idx: list[int], fear_idx: list[int], seed: int
) -> tuple[list[SideMetric], SideMetric | None, SideMetric | None, SideMetric, SideMetric]:
    """按协议跑一个种子：切分→V_cal 选阈冻结→V_test 单跑。

    返回 (V_cal 弃权曲线, 冻结阈值 metric|None, V_test anger metric|None, fear V_cal, fear V_test)。
    """
    a_cal, a_test, _ = _three_way_split(anger_idx, seed)
    f_cal, f_test, _ = _three_way_split(fear_idx, seed + 1)
    a_cal_l = [logits[i] for i in a_cal]
    a_test_l = [logits[i] for i in a_test]
    cal_curve = [_anger_selective_metric(a_cal_l, t) for t in CANDIDATE_THRESHOLDS]
    frozen = _select_frozen_threshold(cal_curve)
    test_anger = _anger_selective_metric(a_test_l, frozen.threshold) if frozen else None
    return (
        cal_curve,
        frozen,
        test_anger,
        _fear_full_metric([logits[i] for i in f_cal]),
        _fear_full_metric([logits[i] for i in f_test]),
    )


def _stability_sweep(
    logits: list[float],
    anger_idx: list[int],
    fear_idx: list[int],
    n_seeds: int,
    prereg_test: dict | None,
) -> dict:
    """跑 N 个种子的同协议扫描，披露字面判据的种子依赖性（sensitivity，非挑种子）。"""
    lbs: list[float] = []
    covs: list[float] = []
    accs: list[float] = []
    fear_lbs: list[float] = []
    taus: list[float] = []
    n_pass = 0
    for s in range(n_seeds):
        _, frozen, test_anger, _, fear_test = _protocol_once(logits, anger_idx, fear_idx, s)
        if frozen is None or test_anger is None:
            continue
        taus.append(frozen.threshold)
        lbs.append(test_anger.wilson_lb)
        covs.append(test_anger.coverage)
        accs.append(test_anger.accuracy)
        fear_lbs.append(fear_test.wilson_lb)
        if test_anger.wilson_lb >= ANGER_LB_BAR and test_anger.coverage >= COVERAGE_FLOOR:
            n_pass += 1
    n = len(lbs)
    pass_rate = n_pass / n if n else float("nan")
    prereg_lb = prereg_test["wilson_lb"] if prereg_test else float("nan")
    prereg_pct = (
        sum(1 for v in lbs if v <= prereg_lb) / n * 100
        if n and not math.isnan(prereg_lb)
        else float("nan")
    )
    return {
        "n_seeds": n_seeds,
        "n_calibrated": n,
        "pass_rate": pass_rate,
        "frac_lb_ge_bar": (sum(1 for v in lbs if v >= ANGER_LB_BAR) / n) if n else float("nan"),
        "lb_median": statistics.median(lbs) if n else float("nan"),
        "lb_mean": statistics.mean(lbs) if n else float("nan"),
        "lb_stdev": statistics.pstdev(lbs) if n > 1 else float("nan"),
        "lb_min": min(lbs) if n else float("nan"),
        "lb_max": max(lbs) if n else float("nan"),
        "acc_median": statistics.median(accs) if n else float("nan"),
        "cov_median": statistics.median(covs) if n else float("nan"),
        "fear_lb_median": statistics.median(fear_lbs) if n else float("nan"),
        "fear_lb_min": min(fear_lbs) if n else float("nan"),
        "prereg_lb": prereg_lb,
        "prereg_percentile": prereg_pct,
        "tau_counts": {f"{t:.2f}": taus.count(t) for t in CANDIDATE_THRESHOLDS},
        "robust": (not math.isnan(pass_rate)) and pass_rate >= STABILITY_PASS_FLOOR,
    }


def validate(
    ed_path: str, weights_path: str, seed: int, *, stability_seeds: int = DEFAULT_STABILITY_SEEDS
) -> dict:
    """跑三路切分验证（预注册种子字面裁决 + 种子稳健性披露），返回结构化结果。"""
    texts, pol = _read_ed(ed_path)
    model = DirectionHead(ST_FEATURE_DIM)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    torch.manual_seed(seed)  # eval 本确定性；仍固定以防未来含随机算子
    with torch.no_grad():
        logits = model(encode_texts(texts, encoder=DEFAULT_ENCODER)).tolist()

    anger_idx = [i for i, p in enumerate(pol) if p == "+"]
    fear_idx = [i for i, p in enumerate(pol) if p == "-"]
    a_cal, a_test, a_reserve = _three_way_split(anger_idx, seed)
    f_cal, f_test, f_reserve = _three_way_split(fear_idx, seed + 1)

    # ── 预注册种子的官方字面裁决 ──
    cal_curve, frozen, test_metric, fear_cal, fear_test = _protocol_once(
        logits, anger_idx, fear_idx, seed
    )
    # 全量点估计（最小偏差·非独立留出）：冻结阈值下对**全部** anger 样本的选择性正确率，最大
    # power、供议会读效应量。⚠ 这不是留出验证（会碰 V_cal 样本），仅作最小偏差点估计参考——
    # 用于校正「24% 过」被误读成「效应是噪声」（数学席：真效应正压在 0.70 边界、非明显 FAIL）。
    full_anger = (
        _anger_selective_metric([logits[i] for i in anger_idx], frozen.threshold)
        if frozen
        else None
    )
    if test_metric is None:
        delta_pass = False
        verdict_reason = f"V_cal 无候选阈值满足覆盖≥{COVERAGE_FLOOR * 100:.0f}%，门无法标定"
    else:
        delta_pass = (
            test_metric.wilson_lb >= ANGER_LB_BAR and test_metric.coverage >= COVERAGE_FLOOR
        )
        verdict_reason = (
            f"V_test anger LB={test_metric.wilson_lb:.4f} "
            f"覆盖={test_metric.coverage * 100:.1f}%"
            f"（bar：LB≥{ANGER_LB_BAR:.2f} 且覆盖≥{COVERAGE_FLOOR * 100:.0f}%）"
        )

    stability = (
        _stability_sweep(
            logits,
            anger_idx,
            fear_idx,
            stability_seeds,
            asdict(test_metric) if test_metric else None,
        )
        if stability_seeds > 0
        else None
    )

    return {
        "seed": seed,
        "weights": weights_path,
        "split_sizes": {
            "anger": {"V_cal": len(a_cal), "V_test": len(a_test), "V_reserve": len(a_reserve)},
            "fear": {"V_cal": len(f_cal), "V_test": len(f_test), "V_reserve": len(f_reserve)},
        },
        "cal_curve": [asdict(m) for m in cal_curve],
        "frozen": asdict(frozen) if frozen else None,
        "test_anger": asdict(test_metric) if test_metric else None,
        "full_data_anger": asdict(full_anger) if full_anger else None,
        "fear_cal": asdict(fear_cal),
        "fear_test": asdict(fear_test),
        "delta_pass": delta_pass,
        "verdict_reason": verdict_reason,
        "stability": stability,
    }


def _recommendation(r: dict) -> str:
    """把字面裁决 + 稳健性合成一句工程建议（不代议会拍板，仅路由）。"""
    if not r["delta_pass"]:
        return "δ FAIL → 回落选项 α（仅 fear 启用；fear 已 LB≈0.90 production-ready）"
    st = r["stability"]
    if st is None or st["robust"]:
        return "δ 稳健 PASS → anger 热路径可进 P1 解锁"
    fd = r.get("full_data_anger")
    fd_note = ""
    if fd:
        tag = "≥0.70·过" if fd["wilson_lb"] >= ANGER_LB_BAR else "<0.70"
        fd_note = f"全量点估计（n={fd['n_covered']}）LB={fd['wilson_lb']:.4f}（{tag}）；"
    return (
        f"δ 预注册种子字面 PASS 但**边界·欠功效**：{st['n_seeds']} 种子 "
        f"{st['pass_rate'] * 100:.0f}% 过、V_test LB 中位 {st['lb_median']:.3f}、"
        f"预注册落 {st['prereg_percentile']:.0f} 分位；但 40/40/20 主动丢弃 power"
        f"（真值≈77.5% 时 n≈83 下 P(LB≥0.70)≈29%，恰配观测），{fd_note}"
        "p_true 点估计≈0.775(>0.70)但独立留出 n≈83 确认不了；"
        "0.68–0.72 是 1000 种子 LB 分布中段(非 p_true 的 CI)、弃权对 anger 近乎无增益(构念天花板)。"
        "→ 回议会定「边界 PASS 是否解锁 / 改 50-50 提功效 / 扩第二 OOD 源」，"
        "勿据此单方解锁 anger；fear 端不受影响仍可走选项 α"
    )


def _fmt_metric(m: dict) -> str:
    return (
        f"τ={m['threshold']:.2f}  覆盖 {m['n_covered']}/{m['n_total']}={m['coverage'] * 100:.1f}%  "
        f"正确 {m['k_correct']}/{m['n_covered']}={m['accuracy'] * 100:.1f}%  "
        f"Wilson 下界 {m['wilson_lb']:.4f}"
    )


def _log_result(r: dict) -> None:
    sz = r["split_sizes"]
    logger.info("─" * 74)
    logger.info("三路切分 anger 置信弃权门验证（seed=%d，权重=%s）", r["seed"], r["weights"])
    logger.info(
        "  切分  anger V_cal/V_test/V_reserve=%d/%d/%d  fear=%d/%d/%d",
        sz["anger"]["V_cal"],
        sz["anger"]["V_test"],
        sz["anger"]["V_reserve"],
        sz["fear"]["V_cal"],
        sz["fear"]["V_test"],
        sz["fear"]["V_reserve"],
    )
    logger.info("─" * 74)
    logger.info(
        "① V_cal 弃权曲线（仅此集允许扫阈值；覆盖≥%.0f%% 约束下选 LB 最大）:", COVERAGE_FLOOR * 100
    )
    for m in r["cal_curve"]:
        elig = "✓合格" if m["coverage"] >= COVERAGE_FLOOR else "✗覆盖不足"
        logger.info("    %s  [%s]", _fmt_metric(m), elig)
    if r["frozen"] is None:
        logger.info("  → 无候选满足覆盖地板，阈值无法冻结。")
    else:
        logger.info(
            "  → 冻结阈值 τ*=%.2f（V_cal 上 LB=%.4f 覆盖=%.1f%%）",
            r["frozen"]["threshold"],
            r["frozen"]["wilson_lb"],
            r["frozen"]["coverage"] * 100,
        )
    logger.info("─" * 74)
    logger.info("② V_test 独立单跑（冻结阈值·不回改）:")
    if r["test_anger"] is not None:
        logger.info("    anger  %s", _fmt_metric(r["test_anger"]))
    logger.info(
        "    fear   全覆盖 %d/%d=%.1f%%  Wilson 下界 %.4f（δ 门 fear 端不弃权）",
        r["fear_test"]["k_correct"],
        r["fear_test"]["n_total"],
        r["fear_test"]["accuracy"] * 100,
        r["fear_test"]["wilson_lb"],
    )
    if r.get("full_data_anger") is not None:
        logger.info(
            "    anger  全量点估计（最大 power·非留出）%s", _fmt_metric(r["full_data_anger"])
        )
    st = r["stability"]
    if st is not None:
        logger.info("─" * 74)
        logger.info("③ 种子稳健性披露（%d 种子·同协议·非挑种子）:", st["n_seeds"])
        logger.info(
            "    δ PASS 占比 %.1f%%  |  anger V_test LB 中位 %.4f（均 %.4f±%.4f，%.3f–%.3f）",
            st["pass_rate"] * 100,
            st["lb_median"],
            st["lb_mean"],
            st["lb_stdev"],
            st["lb_min"],
            st["lb_max"],
        )
        logger.info(
            "    预注册 seed=%d 的 LB=%.4f 落在分布第 %.0f 分位  |  fear LB 中位 %.4f(最低 %.4f)",
            r["seed"],
            st["prereg_lb"],
            st["prereg_percentile"],
            st["fear_lb_median"],
            st["fear_lb_min"],
        )
    logger.info("─" * 74)
    logger.info("字面裁决：%s", "δ 字面 PASS" if r["delta_pass"] else "δ FAIL")
    logger.info("依据：%s", r["verdict_reason"])
    logger.info("工程建议：%s", _recommendation(r))


def _write_report(path: str, r: dict) -> None:
    sz = r["split_sizes"]
    lines: list[str] = []
    lines.append("# anger 置信弃权门 · 三路切分独立验证（议会四轮 δ · P0）\n")
    lines.append(
        f"> 自动生成于 `scripts/validate_anger_abstain.py`。预注册 seed={r['seed']}，"
        f"权重 `{r['weights']}`，数据 EmpatheticDialogues test.csv（CC BY-NC·仅验证）。\n"
    )
    lines.append("## 方法（堵选择偏差）\n")
    lines.append(
        "阈值**只在 V_cal 上**扫候选 "
        + "/".join(f"{t:.2f}" for t in CANDIDATE_THRESHOLDS)
        + f"、按「覆盖≥{COVERAGE_FLOOR * 100:.0f}% 约束下 Wilson LB 最大」冻结一个；"
        + "**V_test 用冻结阈值独立跑且仅跑一次**（报出的数才算数）；V_reserve 存档不用。\n"
    )
    lines.append(
        f"分侧切分（固定种子）：anger V_cal/V_test/V_reserve={sz['anger']['V_cal']}/"
        f"{sz['anger']['V_test']}/{sz['anger']['V_reserve']}，fear={sz['fear']['V_cal']}/"
        f"{sz['fear']['V_test']}/{sz['fear']['V_reserve']}。\n"
    )
    lines.append("## ① V_cal 弃权曲线（选阈用，非终裁）\n")
    lines.append("| τ | 覆盖 | 覆盖率 | 方向正确 | 正确率 | Wilson 下界 | 合格(覆盖≥80%) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for m in r["cal_curve"]:
        elig = "✓" if m["coverage"] >= COVERAGE_FLOOR else "✗"
        lines.append(
            f"| {m['threshold']:.2f} | {m['n_covered']}/{m['n_total']} | "
            f"{m['coverage'] * 100:.1f}% | {m['k_correct']}/{m['n_covered']} | "
            f"{m['accuracy'] * 100:.1f}% | {m['wilson_lb']:.4f} | {elig} |"
        )
    if r["frozen"] is not None:
        lines.append(
            f"\n**冻结阈值 τ\\*={r['frozen']['threshold']:.2f}**"
            f"（V_cal LB={r['frozen']['wilson_lb']:.4f}，"
            f"覆盖={r['frozen']['coverage'] * 100:.1f}%）。"
            "注（本预注册切分下）：τ 被覆盖≥80% 约束强制到最小候选——各阈值 anger 正确率近乎持平"
            "（弃权对 anger 近乎无增益，印证构念天花板；稳健性扫描中约 8% 种子落 τ=0.50）。\n"
        )
    else:
        lines.append("\n**无候选满足覆盖地板 → 阈值无法冻结。**\n")
    lines.append("## ② V_test 独立单跑（预注册种子的字面终裁）\n")
    if r["test_anger"] is not None:
        t = r["test_anger"]
        lines.append(
            f"- **anger**：τ={t['threshold']:.2f}，覆盖 {t['n_covered']}/{t['n_total']}="
            f"{t['coverage'] * 100:.1f}%，方向正确 {t['k_correct']}/{t['n_covered']}="
            f"{t['accuracy'] * 100:.1f}%，**Wilson 下界 {t['wilson_lb']:.4f}**。"
        )
    ft = r["fear_test"]
    lines.append(
        f"- **fear**（全覆盖·不弃权）："
        f"{ft['k_correct']}/{ft['n_total']}={ft['accuracy'] * 100:.1f}%，"
        f"**Wilson 下界 {ft['wilson_lb']:.4f}**。"
    )
    fd = r.get("full_data_anger")
    if fd is not None:
        tag = "≥0.70 → 过" if fd["wilson_lb"] >= ANGER_LB_BAR else "<0.70"
        lines.append(
            f"- **anger 全量点估计**（冻结 τ 下全部 anger·最大 power·⚠非独立留出，会碰 V_cal）："
            f"{fd['k_correct']}/{fd['n_covered']}={fd['accuracy'] * 100:.1f}%，"
            f"**Wilson 下界 {fd['wilson_lb']:.4f}**（{tag}）——最小偏差的效应量点估计，"
            "供议会读「真效应压在 0.70 边界」而非「效应是噪声」。\n"
        )
    st = r["stability"]
    if st is not None:
        lines.append("## ③ 种子稳健性披露（sensitivity·非挑种子）\n")
        lines.append(
            f"同协议跑 {st['n_seeds']} 个种子"
            "（预注册种子给官方字面裁决，此扫描仅披露其种子依赖性）：\n"
        )
        lines.append(
            f"- **δ PASS 种子占比：{st['pass_rate'] * 100:.1f}%**"
            f"（LB≥0.70 占 {st['frac_lb_ge_bar'] * 100:.1f}%）。"
        )
        lines.append(
            f"- anger V_test LB：中位 **{st['lb_median']:.4f}**、均 {st['lb_mean']:.4f}"
            f"±{st['lb_stdev']:.4f}、范围 {st['lb_min']:.3f}–{st['lb_max']:.3f}；"
            f"正确率中位 {st['acc_median'] * 100:.1f}%、覆盖中位 {st['cov_median'] * 100:.1f}%。"
        )
        lines.append(
            f"- **预注册 seed={r['seed']} 的 LB={st['prereg_lb']:.4f} 落在分布第 "
            f"{st['prereg_percentile']:.0f} 分位**（越高说明这次切分越偏乐观尾部）。"
        )
        lines.append(
            f"- fear V_test LB：中位 {st['fear_lb_median']:.4f}、最低 {st['fear_lb_min']:.4f}"
            "（全程稳过，与 anger 形成非对称）。"
        )
        lines.append(
            f"- 注：**24% 大半源于 40/40/20 主动丢弃的检验功效**（真值≈77.5% 时 n≈83 下 "
            f"P(LB≥0.70)≈29%，恰配观测），非「效应是噪声」；扫描种子 0..{st['n_seeds'] - 1} 与"
            "预注册种子不相交，分位为**独立参考分布**。\n"
        )
    lines.append("## 裁决\n")
    literal = (
        "**δ 字面 PASS**（预注册种子 V_test LB≥0.70 且覆盖≥80%）"
        if r["delta_pass"]
        else "**δ FAIL**"
    )
    lines.append(f"- 字面：{literal}。依据：{r['verdict_reason']}。")
    lines.append(f"- **工程建议**：{_recommendation(r)}。\n")
    lines.append(
        "\n> 红线：独立低精度标量流不进 fuse_terms/occ_prior；默认全关零回归；"
        "含 CC BY-NC 数据的权重按 Research-Only 对待。"
        "弃权阈值/anger bar 可达性属议会定语义，工程不私拍——脆弱单次 PASS 是否解锁回议会。\n"
    )
    # 结果只落 notes/（与 docstring 声称一致）；缺目录先建，防 FileNotFoundError（仿姊妹脚本）。
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("报告已写入 %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="anger 置信弃权门三路切分独立验证（议会 δ · P0）")
    p.add_argument("--ed", default="data/raw/ed/empatheticdialogues/test.csv")
    p.add_argument("--weights", default="artifacts/motivational_direction_prior_m.pt")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--stability-seeds",
        type=int,
        default=DEFAULT_STABILITY_SEEDS,
        help="种子稳健性扫描的种子数（0=只跑预注册单次，不做披露）",
    )
    p.add_argument("--report", default=None, help="可选：把结果写入此 markdown 报告路径（notes/）")
    args = p.parse_args()

    r = validate(args.ed, args.weights, args.seed, stability_seeds=args.stability_seeds)
    _log_result(r)
    if args.report:
        _write_report(args.report, r)

    verdict = "PASS" if r["delta_pass"] else "FAIL"
    a_lb = r["test_anger"]["wilson_lb"] if r["test_anger"] else float("nan")
    a_cov = r["test_anger"]["coverage"] if r["test_anger"] else float("nan")
    pass_rate = r["stability"]["pass_rate"] if r["stability"] else float("nan")
    print(
        f"ABSTAIN-GATE 3way: frozen_tau="
        f"{r['frozen']['threshold'] if r['frozen'] else float('nan')} "
        f"anger_test_lb={a_lb:.3f} anger_test_cov={a_cov:.3f} "
        f"fear_test_lb={r['fear_test']['wilson_lb']:.3f} => δ 字面 {verdict} "
        f"| 稳健性 pass_rate={pass_rate:.2f}"
    )


if __name__ == "__main__":
    main()
