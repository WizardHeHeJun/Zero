"""`anatomy.py` 的自检：在**已知的合成旋转**上验证解剖符号与免泄漏性。

为什么必须有：轴映射这块已连错三次，而错了**不驱红**——数值上完全看不出来。
本自检是判别式的（含"应该红"的变异项），不是"跑通即绿"。

跑：`python selftest_anatomy.py`（无需数据集）
"""

from __future__ import annotations

import numpy as np
from anatomy import AnatomyFrame, _handedness, angles_from_matrices


def rodrigues(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """绕任意轴的右手法则旋转矩阵。"""
    n = axis / np.linalg.norm(axis)
    rad = np.radians(angle_deg)
    cross = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.cos(rad) * np.eye(3) + np.sin(rad) * cross + (1 - np.cos(rad)) * np.outer(n, n)


def make_anatomy(right: list[float], forward: list[float], up: list[float]) -> AnatomyFrame:
    r, f, u = (np.array(v, dtype=float) for v in (right, forward, up))
    return AnatomyFrame(r, f, u, _handedness(r, f, u))


# 两套真实数据集各自的解剖系（由 detect_anatomy 从 rest OFFSET 实测得出）
FRAMES = {
    "StayStill(Z-up, 面朝 −Y)": make_anatomy([-1, 0, 0], [0, -1, 0], [0, 0, 1]),
    "ReActIdle(Y-up, 面朝 +Z)": make_anatomy([-1, 0, 0], [0, 0, 1], [0, 1, 0]),
}

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "✅" if condition else "❌"
    print(f"   {mark} {label}{('  ' + detail) if detail else ''}")
    if not condition:
        failures.append(label)


for name, anatomy in FRAMES.items():
    print(f"\n【{name}】{anatomy.describe()}")

    # ① 纯 yaw：绕解剖上轴把视线转向自己的右侧 ⇒ yaw>0，pitch/roll 恒 0
    #    右手法则绕 up 转 −20° 才是"转向右"（forward×up=right ⇒ up×forward=−right）
    m = rodrigues(anatomy.up, -20.0)[None]
    a = angles_from_matrices(m, anatomy)
    check("转向右侧 ⇒ yaw>0", a["yaw"][0] > 0, f"yaw={a['yaw'][0]:+.1f}°")
    check("纯 yaw ⇒ roll≈0", abs(a["roll"][0]) < 1e-6, f"roll={a['roll'][0]:+.3f}°")

    # ② 纯 roll：绕视线轴 ⇒ 头顶倒向右侧为正
    m = rodrigues(anatomy.forward, 15.0)[None]
    a = angles_from_matrices(m, anatomy)
    check("头顶倒向右 ⇒ roll>0", a["roll"][0] > 0, f"roll={a['roll'][0]:+.1f}°")
    check("纯 roll ⇒ yaw≈0", abs(a["yaw"][0]) < 1e-6, f"yaw={a['yaw'][0]:+.3f}°")

    # ③ 纯 pitch：抬头为正。绕**右**轴按右手法则转正角 ⇒ 视线抬起
    #    （right×forward=up，故 R(right,+φ)·forward = cosφ·forward + sinφ·up）
    m = rodrigues(anatomy.right, 12.0)[None]
    a = angles_from_matrices(m, anatomy)
    check("抬头 ⇒ pitch>0", a["pitch"][0] > 0, f"pitch={a['pitch'][0]:+.1f}°")
    check("纯 pitch ⇒ roll≈0", abs(a["roll"][0]) < 1e-6, f"roll={a['roll'][0]:+.3f}°")

    # ④ 关键判别项：pitch 与 yaw 复合时，真 roll 必须仍为 0，
    #    而旧公式 roll_naive 必须**不为 0** —— 这正是伪耦合的来源。
    #    若 roll_naive 也是 0，说明这套自检没有判别力（会被当成"两个公式都对"）。
    pitch_first = rodrigues(anatomy.right, -10.0)  # 低头/抬头
    yaw_after = rodrigues(anatomy.up, -25.0)  # 再转向右
    a = angles_from_matrices((yaw_after @ pitch_first)[None], anatomy)
    check("pitch×yaw 复合 ⇒ 真 roll≈0", abs(a["roll"][0]) < 1e-6, f"roll={a['roll'][0]:+.3f}°")
    check(
        "pitch×yaw 复合 ⇒ 旧公式 roll_naive 显著偏离 0（伪耦合）",
        abs(a["roll_naive"][0]) > 1.0,
        f"roll_naive={a['roll_naive'][0]:+.2f}°",
    )

# ⑤ 伪耦合的**符号翻转**演示：同样的转头，低头者与抬头者的旧 roll 符号相反。
#    这是"待机 +0.415 / 说话 −0.125 符号相反"最可能的机械解释。
#    解析式：up_head·right₀ = −sin(pitch)·sin(yaw) ⇒ **抬头(pitch>0) 给负相关**、低头给正相关。
#    ⚠ 别把这两行标反——真数据的集内归因 corr(平均 pitch, 旧公式耦合) 必须是**负**的才算对上。
print("\n【伪耦合演示：旧公式下，同样转头，低头 vs 抬头给出相反的 yaw-roll 相关】")
anatomy = FRAMES["StayStill(Z-up, 面朝 −Y)"]
for label, pitch_deg in (("持续低头 −10°", -10.0), ("持续抬头 +10°", +10.0)):
    yaws = np.linspace(-25, 25, 51)
    mats = np.stack([rodrigues(anatomy.up, -y) @ rodrigues(anatomy.right, pitch_deg) for y in yaws])
    a = angles_from_matrices(mats, anatomy)
    r_true = np.corrcoef(a["yaw"], a["roll"])[0, 1] if np.std(a["roll"]) > 1e-9 else 0.0
    r_naive = np.corrcoef(a["yaw"], a["roll_naive"])[0, 1]
    print(f"   {label}: 真 roll 相关 {r_true:+.3f}（无耦合）· 旧公式相关 {r_naive:+.3f}")

print("\n" + ("全部通过" if not failures else f"❌ {len(failures)} 项失败: {failures}"))
raise SystemExit(1 if failures else 0)
