"""生成记忆巩固与遗忘曲线图（docs/v2/consolidation-forgetting.png）。

数据不是手绘示意：直接调用 `src.memory.consolidation` 的真实现
（`EbbinghausDecay` 分层幂律衰减 / `SleepConsolidation` 双准则升迁，全默认参数）
逐日模拟，改遗忘指数或升迁门槛后重跑本脚本即可同步图与实现。

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

from src.memory.consolidation import EbbinghausDecay, SleepConsolidation

OUTPUT = Path("docs/v2/consolidation-forgetting.png")
BIRTH = datetime(2026, 1, 1, tzinfo=UTC)
DAYS = list(range(1, 61))

COLOR_SESSION = "#D25D5A"
COLOR_USER = "#5178C6"
COLOR_SALIENT = "#8569CB"
COLOR_MARK = "#D4B45B"


def _episode(scope: str, salience: float, count: int = 0) -> dict[str, Any]:
    return {
        "episode_id": f"{scope}-{salience}",
        "scope": scope,
        "valid_at": BIRTH,
        "salience": salience,
        "consolidation_count": count,
    }


def decay_curve(scope: str, salience: float) -> list[float]:
    """逐日取真实现算出的 decay_weight（每天一次 compute，取唯一一条的结果）。"""
    decay = EbbinghausDecay()
    weights: list[float] = []
    for day in DAYS:
        updates, _ = decay.compute([_episode(scope, salience)], now=BIRTH + timedelta(days=day))
        weights.append(updates[0][0])
    return weights


def promotion_timeline() -> tuple[list[float], int | None]:
    """一条被反复强化的 SESSION 记忆：每 3 天强化一次，满足双准则即升迁 USER。"""
    decay = EbbinghausDecay()
    sleep = SleepConsolidation()
    salience = 0.30
    scope = "session"
    promoted_on: int | None = None
    weights: list[float] = []
    for day in DAYS:
        now = BIRTH + timedelta(days=day)
        count = day // 3  # 每 3 天被重新提及一次，consolidation_count 随之累加
        ep = _episode(scope, salience, count)
        if scope == "session":
            _, to_promote = sleep.compute([ep], now=now)
            if to_promote:
                scope = "user"
                promoted_on = day
                ep = _episode(scope, salience, count)
        updates, _ = decay.compute([ep], now=now)
        weights.append(updates[0][0])
    return weights, promoted_on


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


def plot_promotion(ax: plt.Axes) -> None:
    weights, promoted_on = promotion_timeline()
    ax.plot(DAYS, weights, color=COLOR_SESSION, lw=2.2, label="被反复提及的一条记忆")
    if promoted_on is not None:
        ax.axvline(promoted_on, color=COLOR_MARK, lw=1.4, ls=":")
        ax.annotate(
            f"第 {promoted_on} 天：显著度与强化次数双双达标\nSESSION → USER，之后换慢衰曲线",
            xy=(promoted_on, weights[promoted_on - 1]),
            xytext=(promoted_on + 6, 0.72),
            fontsize=9,
            color="#5C6370",
            arrowprops={"arrowstyle": "->", "color": COLOR_MARK, "lw": 1.0},
        )
    ax.set_title("睡眠巩固：只有反复被强化的显著记忆才升为长期", fontsize=11)
    ax.set_xlabel("距写入的天数 Δt")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    plot_decay(ax1)
    plot_promotion(ax2)
    fig.suptitle(
        "记忆巩固与遗忘（consolidation 真实现·默认参数·该机制默认关闭）",
        fontsize=12,
    )
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"written {OUTPUT}")


if __name__ == "__main__":
    main()
