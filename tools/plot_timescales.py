"""生成三时间尺度冲击-响应曲线图（docs/v2/timescales-dynamics.png）。

数据不是手绘示意：直接调用 `src.agents.affect_math` 的真方程
（`attitude_step` / `emotion_decay_step`，全默认参数）逐轮模拟，
改动力学参数后重跑本脚本即可同步图与实现。

跑：python -m tools.plot_timescales
依赖：matplotlib（仅本脚本的文档生成用，不在项目依赖里；任意 Python 环境
`pip install matplotlib` 即可，affect_math 本身纯标准库）。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.agents.affect_math import attitude_step, emotion_decay_step

OUTPUT = Path("docs/v2/timescales-dynamics.png")
ROUNDS = 13
IMPULSE = (-0.8, 0.7)  # 负面刺激 (v, a)
NEUTRAL = (0.0, 0.0)

COLOR_ESTAR = "#868E96"
COLOR_EMOTION = "#D25D5A"
COLOR_ATTITUDE = "#8569CB"


def simulate(stimulus_rounds: set[int]) -> tuple[list[float], list[float], list[float]]:
    """按轮模拟：指定轮次给负面刺激、其余中性，返回 (e*, emotion, attitude) 的效价轨迹。"""
    estar: list[float] = [0.0]
    emotion_v: list[float] = [0.0]
    attitude_v: list[float] = [0.0]
    emotion = (0.0, 0.0)
    attitude = (0.0, 0.0)
    for t in range(1, ROUNDS):
        stim = IMPULSE if t in stimulus_rounds else NEUTRAL
        attitude = attitude_step(attitude, stim)
        emotion = emotion_decay_step(emotion, attitude, stim)
        estar.append(stim[0])
        emotion_v.append(emotion[0])
        attitude_v.append(attitude[0])
    return estar, emotion_v, attitude_v


def plot_panel(ax: plt.Axes, stimulus_rounds: set[int], title: str) -> None:
    estar, emotion_v, attitude_v = simulate(stimulus_rounds)
    x = range(ROUNDS)
    ax.plot(
        x, estar, color=COLOR_ESTAR, lw=1.2, ls=":", marker="o", ms=3, label="瞬时 e*（当轮冲击）"
    )
    ax.plot(
        x,
        emotion_v,
        color=COLOR_EMOTION,
        lw=2.2,
        marker="o",
        ms=3.5,
        label="快变 emotion（数轮衰退）",
    )
    ax.plot(
        x,
        attitude_v,
        color=COLOR_ATTITUDE,
        lw=2.2,
        marker="s",
        ms=3.5,
        label="慢变 attitude（缓慢沉淀·回弹）",
    )
    ax.axhline(0.0, color="#CED4DA", lw=0.8, zorder=0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("轮次")
    ax.set_ylim(-1.05, 0.15)
    ax.set_xticks(list(x))
    ax.grid(alpha=0.25, lw=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    plot_panel(ax1, {1}, "单次负面冲击：情绪激起又退去，态度几乎不动（不记恨）")
    plot_panel(ax2, set(range(1, 7)), "反复负面刺激(第1-6轮)：态度才逐步沉淀，停止后缓慢回弹")
    ax1.set_ylabel("效价 v")
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.suptitle("三时间尺度冲击-响应（affect_math 真方程轨迹·默认参数）", fontsize=12)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"written {OUTPUT}")


if __name__ == "__main__":
    main()
