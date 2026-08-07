"""核对盲测两版的**实际**数值差异，防「以为改了 A、其实改了 B」。

每个变体的预期不同（`ab_key.json` 里记了是哪个变体），故预期文案按变体给——
写死一套会在下一轮悄悄误导。
"""

import json
import statistics

import _paths as P  # 转正后统一取路径（原为 scratchpad 绝对路径）

KEYS = ("FaceAngleX", "FaceAngleY", "FaceAngleZ")
NAMES = {"FaceAngleX": "yaw(左右转头)", "FaceAngleY": "pitch(低头抬头)", "FaceAngleZ": "roll(侧倾)"}

payload = json.loads(P.AB_PAYLOAD.read_text(encoding="utf-8"))
segs = {s["label"]: s for s in payload["segments"]}
# ⚠ 本脚本是**事后核对**用，读答案是正当的；播放脚本绝不可读它（盲测前提）。
key = json.loads(P.AB_KEY.read_text(encoding="utf-8"))

stats = {}
for label, seg in segs.items():
    version = key[label]
    stats[version] = {}
    for k in KEYS:
        vals = [f["params"][k] for f in seg["keyframes"]]
        stats[version][k] = (statistics.pstdev(vals), max(abs(v) for v in vals))

print(f"{'轴':16s} {'旧 sd':>8s} {'新 sd':>8s} {'新/旧':>7s}   {'旧峰值':>8s} {'新峰值':>8s}")
for k in KEYS:
    o_sd, o_pk = stats["旧"][k]
    n_sd, n_pk = stats["新"][k]
    print(f"{NAMES[k]:16s} {o_sd:8.2f} {n_sd:8.2f} {n_sd / o_sd:7.2f}   {o_pk:8.2f} {n_pk:8.2f}")

variant = key.get("variant", "axis_ratio")
ratio_yaw = stats["新"]["FaceAngleX"][0] / stats["旧"]["FaceAngleX"][0]
ratio_roll = stats["新"]["FaceAngleZ"][0] / stats["旧"]["FaceAngleZ"][0]


# 顺带核对**相关系数**——耦合变体的差异主要在这里，只看 sd 会漏掉
def _corr(label: str) -> float:
    seg = next(s for s in payload["segments"] if key[s["label"]] == label)
    xs = [f["params"]["FaceAngleX"] for f in seg["keyframes"]]
    zs = [f["params"]["FaceAngleZ"] for f in seg["keyframes"]]
    mx, mz = statistics.fmean(xs), statistics.fmean(zs)
    dx = [x - mx for x in xs]
    dz = [z - mz for z in zs]
    denominator = (sum(a * a for a in dx) * sum(b * b for b in dz)) ** 0.5
    return sum(a * b for a, b in zip(dx, dz, strict=True)) / denominator if denominator else 0.0


print(f"\nyaw-roll 相关：旧 {_corr('旧'):+.3f} → 新 {_corr('新'):+.3f}")

if variant == "sway_hz":
    print("预期（sway_hz）：三轴 sd 几乎不变——改的是低频漂移的**频率**（0.07→0.04Hz），不是幅度。")
    print("🛑 **本变体在 10 秒片段上基本不可测**：0.04Hz 一个周期 25 秒，片段里只走 0.4 个周期，")
    print("   两版差异退化成「慢漂移的相位/走向不同」。要盲测它必须用更长的连续播放")
    print("   （`loop_vts.py` 连播），或干脆承认这一项只有数据依据、无主观依据。")
elif variant == "coupling":
    print("预期（coupling）：yaw/pitch 逐字不变 · roll 幅度约 +13%（旧式把它压到比例的 91%）")
    print("             · yaw-roll 相关的**绝对值**下降（−0.74 隐含值 → −0.45 实测值）")
    # ⚠ 单个 10 秒片段只含约 4 个姿态周期，片段级相关系数噪声极大——40 种子实测：
    #    r=−0.74 时片段相关四分位 −0.768~−0.327、r=−0.45 时 −0.588~−0.013，**分布重叠**。
    #    所以下面不对单片段的相关值设阈；同种子配对的**次序**才稳（实测 40/40 一致）。
    if abs(ratio_yaw - 1.0) > 0.02:
        print(f"⚠ yaw 比值 {ratio_yaw:.2f} 不该变 —— 改到了不该改的轴，需查")
    else:
        print(f"✓ yaw 未被波及（比值 {ratio_yaw:.2f}）；roll 比值 {ratio_roll:.2f}")
        print("  片段级相关值仅供参考，勿据单片段判定实现对错（样本量不足，见上方注释）")
else:
    print("预期（axis_ratio）：yaw 比值 ≈1.00（不变）· pitch ≈0.33 · roll ≈0.19")
    if abs(ratio_yaw - 1.0) > 0.15:
        print(f"⚠ yaw 比值 {ratio_yaw:.2f} 偏离 1.00 —— 实现可能有误，需查")
    else:
        print(f"✓ yaw 比值 {ratio_yaw:.2f}，未被缩放；「整体变小」的观感来自 pitch/roll")
