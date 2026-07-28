"""轮数 / 切分的可重跑扫描：把「训多少轮更好」从单次实验变成可复核的表。

**为什么需要它**：本仓历史上的轮数结论都出自单次切分 + 少数种子，而实测显示**切分方差远大于
初始化方差**——拿单次结果比较 1000 轮与 2000 轮，比的其实是噪声。本脚本把两种方差**分开扫**：
切分维度用穷举（韵律 C(8,2)=28 折 / 生理 15 折 LOSO）或多次随机类级留出，初始化维度用多个
`--init-seeds`，然后报 mean±SD、**同切分配对差 ± SE**、以及各轮数的胜出切分数。

同切分配对差是关键：跨切分直接比均值会被切分难度淹没，而同一切分下两个轮数的差值消掉了
切分效应，才是「多训 1000 轮到底值不值」的正确问法。

**效率**：数据只加载一次（韵律提特征极慢，重复加载是之前一次扫描跑 25 分钟的主因）；同一
(切分, init_seed) 下**只训练一次**到最大轮数，在网格点上快照评估，而不是每个轮数重训。

用法：
  python -m scripts.sweep_epochs --channel prosody --root data/ravdess \\
      --epochs-grid 300,1000,2000 --init-seeds 0,1 --report notes/sweep-prosody.md
"""

from __future__ import annotations

import argparse
import itertools
import logging
import random
import statistics
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from scripts._train_common import MIN_GROUP_SIZE, evaluate_with_baselines

logger = logging.getLogger(__name__)

CHANNELS = ("prosody", "physiology", "facs")


def load_channel(channel: str, source: str) -> tuple[Any, Any, list[Hashable], Any]:
    """加载一个通道的全量数据，返回 `(x, y, groups, model_factory)`。**整个扫描只调一次。**"""
    if channel == "prosody":
        from src.agents.datasets.ravdess import load_ravdess
        from src.agents.models.prosody_decoder import ProsodyDecoder

        x, y = load_ravdess(source)
        return x, y, [tuple(r.tolist()) for r in x], ProsodyDecoder
    if channel == "physiology":
        from src.agents.datasets.wesad import load_wesad
        from src.agents.models.physiology_decoder import PhysiologyDecoder

        x, y, subjects = load_wesad(source, return_groups=True)
        return x, y, list(subjects), PhysiologyDecoder
    if channel == "facs":
        from src.agents.datasets.facs_csv import load_facs_csv_ext
        from src.agents.models.facs_decoder import FacsDecoder

        x, y = load_facs_csv_ext(source)
        return x, y, [tuple(r.tolist()) for r in x], lambda: FacsDecoder(extended=True)
    raise ValueError(f"--channel 须为 {CHANNELS} 之一，得到 {channel!r}")


def enumerate_splits(
    groups: Sequence[Hashable], *, mode: str, n_val_groups: int, max_splits: int, seed: int
) -> list[tuple[list[int], list[int], str]]:
    """枚举切分，返回 `[(train_idx, val_idx, label)]`。

    `mode="exhaustive"`：穷举「留 `n_val_groups` 组出去」的**全部** C(G,k) 种组合——组数少时
    （韵律 8 组 → 28 折、生理 15 人留 1 → 15 折）这是唯一能消除切分抽样偏差的做法。
    `mode="random"`：组数多时穷举不现实（FACS 38 组留 8 组是天文数字），退回多次随机类级留出。
    """
    keys = sorted({g for g in groups}, key=str)
    if len(keys) < MIN_GROUP_SIZE + 1:
        raise ValueError(f"组数 {len(keys)} 太少，无法做留出扫描")

    combos: list[tuple[Hashable, ...]]
    if mode == "exhaustive":
        combos = list(itertools.combinations(keys, n_val_groups))
        if len(combos) > max_splits:
            logger.warning(
                "穷举得 %d 折，超过 --max-splits %d；将**均匀抽样**而非截断前 N 个"
                "（截断会系统性偏向字典序靠前的组）",
                len(combos),
                max_splits,
            )
            rng = random.Random(seed)
            combos = sorted(rng.sample(combos, max_splits), key=str)
    else:
        rng = random.Random(seed)
        seen: set[tuple[Hashable, ...]] = set()
        combos = []
        while len(combos) < max_splits:
            pick = tuple(sorted(rng.sample(keys, n_val_groups), key=str))
            if pick in seen:  # 去重：重复切分会让「独立折数」虚高
                continue
            seen.add(pick)
            combos.append(pick)

    # 标签用组序号而非组键本身：FACS/韵律的键是 (v,a) 浮点元组，直接展开会让表格完全没法读
    key_index = {k: i for i, k in enumerate(keys)}
    out: list[tuple[list[int], list[int], str]] = []
    for combo in combos:
        val_keys = set(combo)
        tr = [i for i, g in enumerate(groups) if g not in val_keys]
        va = [i for i, g in enumerate(groups) if g in val_keys]
        if tr and va:
            out.append((tr, va, "+".join(f"g{key_index[k]}" for k in combo)))
    return out


def group_index_table(groups: Sequence[Hashable], *, limit: int = 20) -> list[str]:
    """组序号 → 组键的对照表，供逐折明细回溯。组数过多时省略（否则报告被它淹没）。"""
    keys = sorted({g for g in groups}, key=str)
    if len(keys) > limit:
        return [
            "",
            f"（共 {len(keys)} 组，超过 {limit} 不列对照表；折标签中的 gN 为组的字典序序号）",
        ]
    lines = ["", "## 组序号对照", "", "| 序号 | 组键 | 行数 |", "| --- | --- | --- |"]
    counts = {k: sum(1 for g in groups if g == k) for k in keys}
    for i, k in enumerate(keys):
        lines.append(f"| g{i} | `{k}` | {counts[k]} |")
    return lines


def fit_and_snapshot(
    model_factory: Any,
    x_tr: Any,
    y_tr: Any,
    x_val: Any,
    y_val: Any,
    val_groups: Sequence[Hashable],
    *,
    epochs_grid: Sequence[int],
    init_seed: int,
    lr: float,
    columns: Sequence[int] | None,
) -> dict[int, dict[str, float]]:
    """训练**一次**到最大轮数，在网格点上快照评估——不为每个轮数重训。

    这样同一 (切分, init_seed) 下不同轮数的结果共享完全相同的初始化与数据顺序，
    轮数之间的差值才是纯粹的「多训了几轮」，不掺初始化差异。
    """
    torch.manual_seed(init_seed)
    model = model_factory()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_mean = y_tr.mean(dim=0, keepdim=True)
    checkpoints = sorted(set(epochs_grid))
    snapshots: dict[int, dict[str, float]] = {}

    model.train()
    for step in range(1, max(checkpoints) + 1):
        optimizer.zero_grad()
        loss = loss_fn(model(x_tr), y_tr)
        loss.backward()
        optimizer.step()
        if step in set(checkpoints):
            train_loss = float(loss.item())
            model.eval()
            with torch.no_grad():
                pred = model(x_val)
            model.train()
            metrics = evaluate_with_baselines(
                y_val, pred, val_groups, train_mean=train_mean, columns=columns
            )
            metrics["train_loss"] = train_loss
            snapshots[step] = metrics
    return snapshots


def _fmt(value: float, digits: int = 5) -> str:
    return f"{value:.{digits}f}"


def render_report(
    rows: list[dict[str, Any]],
    *,
    channel: str,
    epochs_grid: Sequence[int],
    meta: dict[str, Any],
    groups: Sequence[Hashable] | None = None,
) -> str:
    """渲染 md 表：mean±SD / 同切分配对差±SE / 胜出切分数。"""
    grid = sorted(set(epochs_grid))
    lines = [
        f"# 轮数扫描：{channel}",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        *(f"| {k} | {v} |" for k, v in meta.items()),
        "",
        "## 各轮数汇总（val 面）",
        "",
        "| epochs | 技能分 mean±SD | val MSE mean±SD | 组等权 MSE mean | 可学空间 mean |",
        "| --- | --- | --- | --- | --- |",
    ]
    by_epoch: dict[int, list[dict[str, float]]] = {e: [] for e in grid}
    for row in rows:
        by_epoch[row["epochs"]].append(row["metrics"])

    for e in grid:
        skills = [m["skill_score"] for m in by_epoch[e]]
        mses = [m["mse"] for m in by_epoch[e]]
        cb = [m["mse_class_balanced"] for m in by_epoch[e]]
        hr = [m["headroom_used"] for m in by_epoch[e]]
        sd = statistics.stdev(skills) if len(skills) > 1 else 0.0
        sd_mse = statistics.stdev(mses) if len(mses) > 1 else 0.0
        lines.append(
            f"| {e} | {statistics.mean(skills):+.4f} ± {sd:.4f} "
            f"| {_fmt(statistics.mean(mses))} ± {_fmt(sd_mse)} "
            f"| {_fmt(statistics.mean(cb))} | {statistics.mean(hr) * 100:+.1f}% |"
        )

    # 同切分配对差：消掉切分难度，才是「多训 N 轮值不值」的正确问法
    lines += [
        "",
        "## 同切分配对差（后者 − 前者的 val MSE；负=多训更好）",
        "",
        "| 对比 | 配对差 mean | SE | 后者更优的切分数 |",
        "| --- | --- | --- | --- |",
    ]
    paired: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        paired.setdefault((row["split"], row["init_seed"]), [])
    for a, b in itertools.combinations(grid, 2):
        diffs = []
        for row_a in rows:
            if row_a["epochs"] != a:
                continue
            match = [
                r
                for r in rows
                if r["epochs"] == b
                and r["split"] == row_a["split"]
                and r["init_seed"] == row_a["init_seed"]
            ]
            if match:
                diffs.append(match[0]["metrics"]["mse"] - row_a["metrics"]["mse"])
        if not diffs:
            continue
        mean = statistics.mean(diffs)
        se = (statistics.stdev(diffs) / len(diffs) ** 0.5) if len(diffs) > 1 else 0.0
        wins = sum(1 for d in diffs if d < 0)
        lines.append(f"| {a} → {b} | {mean:+.6f} | {se:.6f} | {wins}/{len(diffs)} |")

    lines += [
        "",
        "## 逐折明细",
        "",
        "| 切分 | init_seed | epochs | 技能分 | val MSE | 常数基线 | 组内下界 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| {row['label']} | {row['init_seed']} | {row['epochs']} "
            f"| {m['skill_score']:+.4f} | {_fmt(m['mse'])} "
            f"| {_fmt(m['mse_constant'])} | {_fmt(m['within_group_floor'])} |"
        )
    if groups is not None:
        lines += group_index_table(groups)
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="轮数 / 切分扫描，切分方差与初始化方差分开报。")
    p.add_argument("--channel", choices=list(CHANNELS), required=True)
    p.add_argument("--root", required=True, help="数据源（韵律/生理为目录，FACS 为 CSV 路径）")
    p.add_argument("--epochs-grid", default="300,1000,2000", help="逗号分隔的轮数网格")
    p.add_argument("--init-seeds", default="0", help="逗号分隔的模型初始化种子")
    p.add_argument("--split-mode", choices=["exhaustive", "random"], default="exhaustive")
    p.add_argument("--n-val-groups", type=int, default=2, help="每折留出的组数")
    p.add_argument("--max-splits", type=int, default=28, help="折数上限（超出则均匀抽样）")
    p.add_argument("--split-seed", type=int, default=0, help="随机/抽样切分的种子")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--report", default=None, help="md 报告输出路径；不给则只打印摘要")
    args = p.parse_args()

    grid = [int(v) for v in args.epochs_grid.split(",")]
    seeds = [int(v) for v in args.init_seeds.split(",")]

    logger.info("加载 %s 数据（整个扫描只加载一次）...", args.channel)
    x, y, groups, factory = load_channel(args.channel, args.root)
    columns = None
    if args.channel == "facs":
        from scripts.train_facs import _RUNTIME_CONSUMED_COLUMNS

        columns = _RUNTIME_CONSUMED_COLUMNS

    splits = enumerate_splits(
        groups,
        mode=args.split_mode,
        n_val_groups=args.n_val_groups,
        max_splits=args.max_splits,
        seed=args.split_seed,
    )
    logger.info(
        "样本 %d 行 / %d 组 → %d 折 × %d 个 init 种子（每折训 1 次到 %d 轮，网格点快照）",
        len(groups),
        len({str(g) for g in groups}),
        len(splits),
        len(seeds),
        max(grid),
    )

    rows: list[dict[str, Any]] = []
    for split_idx, (tr, va, label) in enumerate(splits):
        val_groups = [groups[i] for i in va]
        for init_seed in seeds:
            snaps = fit_and_snapshot(
                factory,
                x[tr],
                y[tr],
                x[va],
                y[va],
                val_groups,
                epochs_grid=grid,
                init_seed=init_seed,
                lr=args.lr,
                columns=columns,
            )
            for epochs, metrics in snaps.items():
                rows.append(
                    {
                        "split": split_idx,
                        "label": label,
                        "init_seed": init_seed,
                        "epochs": epochs,
                        "metrics": metrics,
                    }
                )
        logger.info("  折 %d/%d 完成（val 组：%s）", split_idx + 1, len(splits), label)

    meta = {
        "通道": args.channel,
        "数据源": args.root,
        "切分方式": f"{args.split_mode}（每折留 {args.n_val_groups} 组）",
        "折数": len(splits),
        "init 种子": ",".join(str(s) for s in seeds),
        "轮数网格": ",".join(str(e) for e in grid),
        "lr": args.lr,
        "评分维度": "运行时消费的 9 维" if columns else "全部输出维",
    }
    report = render_report(rows, channel=args.channel, epochs_grid=grid, meta=meta, groups=groups)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report, encoding="utf-8")
        logger.info("报告 -> %s", args.report)
    print(report)


if __name__ == "__main__":
    main()
