"""核对盲测两版的实际差异：yaw 应**不变**，仅 pitch/roll 缩小。

用户观感是「幅度小了很多」，需排除「yaw 也被缩了」这种实现错误。
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

print("\n预期：yaw 比值 ≈1.00（不变）· pitch ≈0.33 · roll ≈0.19")
ratio_yaw = stats["新"]["FaceAngleX"][0] / stats["旧"]["FaceAngleX"][0]
if abs(ratio_yaw - 1.0) > 0.15:
    print(f"⚠ yaw 比值 {ratio_yaw:.2f} 偏离 1.00 —— 实现可能有误，需查")
else:
    print(f"✓ yaw 比值 {ratio_yaw:.2f}，未被缩放；「整体变小」的观感来自 pitch/roll")
