"""生成记忆遗忘曲线图（docs/v2/consolidation-forgetting.png）。

数据不是手绘示意：直接调用 `src.memory.consolidation.EbbinghausDecay`（分层幂律衰减，
全默认参数）逐日模拟，改遗忘指数后重跑本脚本即可同步图与实现。

⚠ 2026-08-13 复裁（PRP/sleep-consolidation-verdict/design.md，路 B·退役）：本脚本原有
「SESSION→USER 睡眠巩固升迁」面板依赖已删除的 `SleepConsolidation` 类，该策略从未在生产
生效，已随之退役——本脚本收窄为单一衰减曲线面板，不再画升迁时间线。

跑：python -m tools.plot_consolidation
依赖：matplotlib（仅本脚本的文档生成用，不在项目依赖里；任意 Python 环境
`pip install matplotlib` 即可，consolidation 本身纯标准库）。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.memory.consolidation import EbbinghausDecay

OUTPUT = Path("docs/v2/consolidation-forgetting.png")
BIRTH = datetime(2026, 1, 1, tzinfo=UTC)
DAYS = list(range(1, 61))

COLOR_SESSION = "#D25D5A"
COLOR_USER = "#5178C6"
COLOR_SALIENT = "#8569CB"


def _episode(scope: str, salience: float) -> dict[str, Any]:
    return {
        "episode_id": f"{scope}-{salience}",
        "scope": scope,
        "valid_at": BIRTH,
        "salience": salience,
    }


def decay_curve(scope: str, salience: float) -> list[float]:
    """逐日取真实现算出的 decay_weight（每天一次 compute，取唯一一条的结果）。"""
    decay = EbbinghausDecay()
    weights: list[float] = []
    for day in DAYS:
        updates, _ = decay.compute([_episode(scope, salience)], now=BIRTH + timedelta(days=day))
        weights.append(updates[0][0])
    return weights


def plot_decay(ax: plt.Axes) -> None:
    ax.plot(
        DAYS,
        decay_curve("session", 0.5),
        color=COLOR_SESSION,
        lw=2.2,
        label="SESSION 短期（d=0.8 快衰）",
    )
    ax.plot(
        DAYS,
        decay_curve("user", 0.5),
        color=COLOR_USER,
        lw=2.2,
        label="USER 长期（d=0.3 慢衰）",
    )
    ax.plot(
        DAYS,
        decay_curve("session", 0.9),
        color=COLOR_SALIENT,
        lw=1.6,
        ls="--",
        label="SESSION 但高显著（salience 0.9）",
    )
    ax.set_title("分层幂律遗忘：同一条记忆，短期忘得快、长期忘得慢", fontsize=11)
    ax.set_xlabel("距写入的天数 Δt")
    ax.set_ylabel("decay_weight（召回权重）")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax1 = plt.subplots(1, 1, figsize=(6.2, 4.0))
    plot_decay(ax1)
    fig.suptitle(
        "记忆遗忘（consolidation 真实现·默认参数·该机制默认关闭）",
        fontsize=12,
    )
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"written {OUTPUT}")


if __name__ == "__main__":
    main()
