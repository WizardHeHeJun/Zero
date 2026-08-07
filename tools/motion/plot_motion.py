"""把合成轨迹画出来，直观看效果。"""

import _paths as P  # 转正后统一取路径（原为 scratchpad 绝对路径）
import matplotlib

P.use_zero()  # 必须在 import src.* 之前
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.agents.motion_synth import (  # noqa: E402
    PARAM_ANGLE_X,
    PARAM_ANGLE_Y,
    PARAM_EYE_OPEN_L,
    PhaseState,
    generate,
    generate_dual,
    initial_blink_ms,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

DUR = 12000.0
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

# ① 三种唤醒水平的头部左右摆
ax = axes[0]
for arousal, label, color in (
    (-0.8, "平静 a=-0.8", "#4C9F70"),
    (0.0, "中性 a=0.0", "#5B8FF9"),
    (0.8, "激动 a=+0.8", "#E8684A"),
):
    seed = 5
    frames, _ = generate(
        0.0, arousal, DUR, PhaseState(noise_seed=seed, next_blink_ms=initial_blink_ms(seed))
    )
    t = [f["t_ms"] / 1000 for f in frames]
    y = [f["params"][PARAM_ANGLE_X] for f in frames]
    ax.plot(t, y, label=label, color=color, lw=1.4)
ax.set_ylabel("FaceAngleX（度）")
ax.set_title("① 情绪直驱：唤醒越高，头部摆动越大越快（同种子，仅 arousal 不同）")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.25)
ax.axhline(0, color="#999", lw=0.6)

# ② 意志调控双通路
ax = axes[1]
seed = 9
heads, _ = generate_dual(
    (-0.7, 0.8),
    (-0.7, 0.8),
    DUR,
    PhaseState(noise_seed=seed, next_blink_ms=initial_blink_ms(seed)),
    voluntary_leak=0.3,
)
t = [f["t_ms"] / 1000 for f in heads["spontaneous"]]
ax.plot(
    t,
    [f["params"][PARAM_ANGLE_X] for f in heads["spontaneous"]],
    label="非随意（情绪原样泄漏）",
    color="#E8684A",
    lw=1.4,
)
ax.plot(
    t,
    [f["params"][PARAM_ANGLE_X] for f in heads["voluntary"]],
    label="随意（压着点，leak=0.3）",
    color="#5B8FF9",
    lw=1.6,
)
ax.set_ylabel("FaceAngleX（度）")
ax.set_title("② 意志调控：同一情绪，压制后幅度变小但压不平（不归零）")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.25)
ax.axhline(0, color="#999", lw=0.6)

# ③ 呼吸与眨眼
ax = axes[2]
seed = 3
frames, _ = generate(
    0.0, -0.6, DUR, PhaseState(noise_seed=seed, next_blink_ms=initial_blink_ms(seed))
)
t = [f["t_ms"] / 1000 for f in frames]
ax.plot(
    t,
    [f["params"][PARAM_ANGLE_Y] for f in frames],
    color="#4C9F70",
    lw=1.4,
    label="FaceAngleY（含呼吸 0.27Hz）",
)
ax2 = ax.twinx()
ax2.fill_between(
    t,
    [f["params"][PARAM_EYE_OPEN_L] for f in frames],
    1.0,
    color="#8B5CF6",
    alpha=0.35,
    label="眨眼",
)
ax2.set_ylim(-0.05, 1.6)
ax2.set_yticks([0, 1])
ax2.set_ylabel("眼睑开合")
ax.set_xlabel("时间（秒）")
ax.set_ylabel("FaceAngleY（度）")
ax.set_title("③ idle 基线：呼吸带上下微动 + 非泊松眨眼（紫色为闭眼）")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.25)

plt.tight_layout()
out = str(P.PLOT_PNG)  # 目录已由 _paths 建好，无需再 makedirs
plt.savefig(out, dpi=130)
print(f"saved -> {out}")
