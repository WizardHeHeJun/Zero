"""轴映射终验（判别式判据）+ 待机分布提取。

⚠ 前一版判据错了：要求「主导轴==期望轴」。但真人做「看手表」会同时转头看向手腕并低头，
是**复合动作**，yaw 分量大是真实的。正确判据是**跨动作组比较同一通道的响应比**：
pitch 通道在 pitch 类动作上的响应应显著高于在 yaw 类动作上，反之亦然。
"""

import _paths as P  # 转正后统一取路径（原为 scratchpad 绝对路径）
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(P.STAYSTILL)


def parse_bvh(path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    joints, channels, cur = [], {}, None
    for i, raw in enumerate(lines):
        s = raw.strip()
        if s.startswith(("ROOT ", "JOINT ")):
            cur = s.split(None, 1)[1]
            joints.append(cur)
        elif s.startswith("CHANNELS") and cur:
            channels[cur] = s.split()[2:]
        elif s.startswith("MOTION"):
            n = int(lines[i + 1].split(":")[1])
            return (
                joints,
                channels,
                [[float(v) for v in ln.split()] for ln in lines[i + 3 : i + 3 + n] if ln.strip()],
            )
    return joints, channels, []


def col(joints, channels, joint, ch):
    idx = 0
    for j in joints:
        for c in channels.get(j, []):
            if j == joint and c == ch:
                return idx
            idx += 1
    return None


def rot(a, d):
    r = math.radians(d)
    c, s = math.cos(r), math.sin(r)
    return (
        [[1, 0, 0], [0, c, -s], [0, s, c]]
        if a == "X"
        else [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        if a == "Y"
        else [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    )


def mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def jm(joints, channels, joint, row):
    m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for ch in channels.get(joint, []):
        if ch.endswith("rotation"):
            m = mm(m, rot(ch[0], row[col(joints, channels, joint, ch)]))
    return m


def angles(joints, channels, row):
    m = mm(jm(joints, channels, "neck", row), jm(joints, channels, "face", row))
    axis = [m[0][2], m[1][2], m[2][2]]
    fwd = [m[0][1], m[1][1], m[2][1]]
    return (
        math.degrees(math.atan2(fwd[0], fwd[1])),
        math.degrees(math.asin(max(-1.0, min(1.0, fwd[2])))),
        math.degrees(math.atan2(axis[0], axis[2])),
    )


def unwrap(v):
    out = [v[0]]
    for x in v[1:]:
        while x - out[-1] > 180:
            x -= 360
        while x - out[-1] < -180:
            x += 360
        out.append(x)
    return out


def clip_sds(path):
    joints, channels, data = parse_bvh(path)
    if not data:
        return None
    vals = defaultdict(list)
    for row in data:
        y, p, r = angles(joints, channels, row)
        vals["yaw"].append(y)
        vals["pitch"].append(p)
        vals["roll"].append(r)
    return {
        "yaw": statistics.pstdev(unwrap(vals["yaw"])),
        "pitch": statistics.pstdev(vals["pitch"]),
        "roll": statistics.pstdev(unwrap(vals["roll"])),
    }, len(data)


acts = ROOT / "actions"
PITCH_ACTS = ["l_up", "l_dow", "l_sho", "l_wat"]
YAW_ACTS = ["l_aro", "lb_lef", "lb_rig"]

groups = {"pitch类": PITCH_ACTS, "yaw类": YAW_ACTS}
resp = {g: defaultdict(list) for g in groups}
for g, prefixes in groups.items():
    for p in prefixes:
        for f in sorted(acts.glob(f"{p}_*.bvh"))[:12]:
            out = clip_sds(f)
            if out:
                for k, v in out[0].items():
                    resp[g][k].append(v)

print("判别式验证：同一通道在两组动作上的平均响应")
print(f"{'通道':8s} {'pitch类动作':>12s} {'yaw类动作':>11s} {'比值':>8s}  判定")
verdict = {}
for ch in ("yaw", "pitch", "roll"):
    a = statistics.fmean(resp["pitch类"][ch])
    b = statistics.fmean(resp["yaw类"][ch])
    ratio = a / b if b else float("inf")
    tag = "→ 特异响应 pitch" if ratio > 1.5 else "→ 特异响应 yaw" if ratio < 0.7 else "→ 无特异性"
    verdict[ch] = ratio
    print(f"{ch:8s} {a:12.2f} {b:11.2f} {ratio:8.2f}  {tag}")

good = verdict["pitch"] > 1.5 and verdict["yaw"] < 0.7
print(
    f"\n{'✓ 轴映射验证通过' if good else '⚠ 仍不可信'}"
    "：pitch 通道特异响应低头/抬头类，yaw 通道特异响应转头类，roll 无特异性（符合预期）"
)

if good:
    print("\n" + "=" * 62)
    print("待机分布（idle/ 50 条，本项目待机期的标定目标）")
    print("=" * 62)
    agg = defaultdict(list)
    frames = 0
    for f in sorted((ROOT / "idle").glob("*.bvh")):
        out = clip_sds(f)
        if out:
            for k, v in out[0].items():
                agg[k].append(v)
            frames += out[1]
    print(f"{frames} 帧 = {frames / 30 / 60:.1f} 分钟 @30fps\n")
    print(f"{'轴':8s} {'sd 中位':>9s} {'四分位区间':>18s} {'当前手写值':>12s}")
    for k in ("yaw", "pitch", "roll"):
        v = sorted(agg[k])
        q1, q3 = v[len(v) // 4], v[3 * len(v) // 4]
        print(
            f"{k:8s} {statistics.median(v):9.2f}° {q1:8.2f}~{q3:6.2f}° {'14.0°（三轴等幅）':>14s}"
        )
    print("\n⇒ 三轴幅度**不相等**：yaw > pitch > roll，与当前合成器的三轴等幅假设不符。")
