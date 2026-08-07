"""yaw-roll 耦合的**判别式**测定：解剖语义 + 免泄漏 roll + 跨数据集符号一致性。

交接文档第五节列了三项"量到但没敢用"的数字，其中耦合符号最危险（侧倾反向一眼可见）。
本脚本给的不是一个数，是一套**判据**：

| 判据 | 通过条件 | 不通过意味着 |
| --- | --- | --- |
| ① 解剖锚定 | 三轴由 rest OFFSET 实测判定，非假设 | 符号无解剖含义，不可跨数据集比较 |
| ② 免泄漏 | roll 用 swing-twist，pitch×yaw 复合仍给 0 | 测到的"耦合"可能纯是几何伪影 |
| ③ 符号一致性 | 逐片段符号一致率显著高于 50%（二项检验） | 只是个体习惯，非运动学固有属性 |
| ④ 跨数据集复现 | StayStill / ReActIdle(留出) 同号且量级可比 | 域特异，不可外推到数字人 |
| ⑤ 伪影归因 | 旧公式的逐片段耦合应可由「该片段平均 pitch」预测 | 泄漏解释不成立，另有原因 |

跑：`python coupling_measure.py`
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path

import _paths as P
import numpy as np
from anatomy import angles_from_matrices, detect_anatomy, head_angles, parse_bvh

MIN_FRAMES = 300
# 增量相关只取"确实在动"的帧：静止帧的 Δ 全是噪声，纳入会把真耦合稀释向 0。
# 阈取该片段合成角速度的中位数（非参数、无自由参数），避免再引入一个手调阈。
DATASETS: tuple[tuple[str, Path], ...] = (
    ("StayStill idle（主）", P.STAYSTILL / "idle"),
    ("ReActIdle genuine（留出·真自发）", P.REACTIDLE / "genuine"),
    ("ReActIdle acted（留出·表演）", P.REACTIDLE / "acted"),
)


def binomial_p_two_sided(successes: int, n: int) -> float:
    """符号检验的双侧 p（p=0.5 的精确二项）。不引 scipy —— 本仓工具链保持轻。"""
    if n == 0:
        return 1.0

    def pmf(k: int) -> float:
        return math.comb(n, k) * 0.5**n

    observed = pmf(successes)
    return min(1.0, sum(pmf(k) for k in range(n + 1) if pmf(k) <= observed + 1e-12))


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 10 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _rodrigues(axis: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """(n,) 角度 → 绕定轴的 (n,3,3) 右手法则旋转矩阵。"""
    unit = axis / np.linalg.norm(axis)
    rad = np.radians(angles_deg)
    cross = np.array([[0, -unit[2], unit[1]], [unit[2], 0, -unit[0]], [-unit[1], unit[0], 0]])
    cos, sin = np.cos(rad)[:, None, None], np.sin(rad)[:, None, None]
    return cos * np.eye(3) + sin * cross + (1 - cos) * np.outer(unit, unit)


KEYS = ("raw", "delta", "naive", "naive_ablated", "mean_pitch", "sd_ratio")


def measure(directory: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {k: [] for k in KEYS}
    frames_desc: list[str] = []
    for path in sorted(directory.glob("*.bvh")):
        skeleton = parse_bvh(path)
        if len(skeleton.frames) < MIN_FRAMES:
            continue
        anatomy = detect_anatomy(skeleton)
        if not frames_desc:
            frames_desc.append(anatomy.describe())
        angles = head_angles(skeleton, anatomy)
        yaw, pitch, roll, naive = (angles[k] for k in ("yaw", "pitch", "roll", "roll_naive"))

        raw = corr(yaw, roll)
        nai = corr(yaw, naive)
        d_yaw, d_roll = np.diff(yaw), np.diff(roll)
        speed = np.abs(np.diff(yaw)) + np.abs(np.diff(pitch)) + np.abs(np.diff(roll))
        moving = speed > np.median(speed)
        delta = corr(d_yaw[moving], d_roll[moving])

        # ⑥ 直接消融：保留实测 yaw/pitch，把真实 roll **置零**再重算旧公式。
        # 得到的即"纯几何泄漏"分量——不需要任何关于泄漏机制的假设，是构造出来的对照臂。
        ablated = _rodrigues(anatomy.up, -yaw) @ _rodrigues(anatomy.right, pitch)
        ablated_angles = angles_from_matrices(ablated, anatomy)
        assert np.max(np.abs(ablated_angles["roll"])) < 1e-6, "消融构造本身带 roll，对照臂不成立"
        leak = corr(yaw, ablated_angles["roll_naive"])

        if raw is None or nai is None or delta is None or leak is None:
            continue
        out["raw"].append(raw)
        out["delta"].append(delta)
        out["naive"].append(nai)
        out["naive_ablated"].append(leak)
        out["mean_pitch"].append(float(np.mean(pitch)))
        out["sd_ratio"].append(float(np.std(roll) / max(float(np.std(yaw)), 1e-9)))
    out["_desc"] = frames_desc  # type: ignore[assignment]
    return out


def report(label: str, values: list[float]) -> None:
    n = len(values)
    positive = sum(1 for v in values if v > 0)
    p = binomial_p_two_sided(max(positive, n - positive), n)
    med = statistics.median(values)
    q1, q3 = np.percentile(values, 25), np.percentile(values, 75)
    consistent = max(positive, n - positive) / n if n else 0.0
    verdict = "✅ 符号稳定" if p < 0.05 and consistent >= 0.75 else "⚠ 符号不稳定"
    print(
        f"   {label:12s} 中位 {med:+.3f}  四分位 {q1:+.3f}~{q3:+.3f}  "
        f"同号 {max(positive, n - positive)}/{n} ({consistent:.0%})  p={p:.4f}  {verdict}"
    )


all_delta: list[float] = []
all_ratio: list[float] = []

for name, directory in DATASETS:
    if not directory.exists():
        print(f"\n【{name}】目录不存在，跳过：{directory}")
        continue
    data = measure(directory)
    n = len(data["raw"])
    desc = data["_desc"][0] if data["_desc"] else "?"  # type: ignore[index]
    print(f"\n【{name}】n={n} 条 · 解剖系 {desc}")
    print("   —— 正号 = 同侧（转右伴随头顶倒向右）；负号 = 对侧 ——")
    report("解剖·原序列", data["raw"])
    report("解剖·增量", data["delta"])
    report("旧公式(泄漏)", data["naive"])
    report("↑消融对照臂", data["naive_ablated"])
    # ⑤ 归因必须**集内**做：两数据集 rest 姿态基准不同，pitch 均值不可跨集比较
    #    （合并算会得到由数据集间差异驱动的假相关——上一版就栽在这）。
    attribution = corr(np.array(data["mean_pitch"]), np.array(data["naive"]))
    # 泄漏解析式 up_head·right₀ = −sin(pitch)·sin(yaw) ⇒ 抬头者旧公式偏负、低头者偏正，
    # 故这个相关**应为负**才与泄漏机制一致（见 selftest_anatomy.py 的合成演示）。
    ok = "✅ 与泄漏机制一致" if attribution is not None and attribution < -0.2 else "⚠ 不一致"
    print(f"   ⑤ 集内归因 corr(片段平均 pitch, 旧公式耦合) = {attribution:+.3f}  {ok}")
    print(f"   sd(roll)/sd(yaw) 中位 = {statistics.median(data['sd_ratio']):.3f}")
    all_delta += data["delta"]
    all_ratio += data["sd_ratio"]

print("\n" + "=" * 72)
print("⑥ 消融对照臂读法：真实 roll 置零后旧公式仍给出的相关 = 纯几何泄漏")
print("   若「消融对照臂」与「旧公式」量级相当 ⇒ 旧公式测到的主要是伪影，其符号不可用。")
print("=" * 72)
print(
    f"\n结论用量（三集合并 · 解剖·增量）：中位 r = {statistics.median(all_delta):+.3f}"
    f"  (n={len(all_delta)})"
)
print(f"三集合并 sd(roll)/sd(yaw) 中位 = {statistics.median(all_ratio):.3f}")
