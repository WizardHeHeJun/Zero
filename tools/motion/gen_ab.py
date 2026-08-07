"""盲测 A/B 载荷：旧版（改动前参数）vs 新版（议会常数 + 眼头协同），随机排序。

议会要求：「真正的验收锚点是用户最初抱怨的『不自然』，是主观感知问题，
统计距离缩小不保证观感变好」——故必须盲法。
"""

import _paths as P  # 转正后统一取路径（原为 scratchpad 绝对路径）
import json
import sys
import zlib

P.use_zero()

import src.agents.motion_synth as ms  # noqa: E402

OUT = str(P.AB_PAYLOAD)
KEY = str(P.AB_KEY)

SEED = 860613
# 取契约上限 10s（TRAJECTORY_MAX_SEGMENT_MS=10000，超了对面会 rejected）。
# 8 秒只够眨 2 次，判断眨眼频率样本太少；10 秒能眨 2~3 次，且播放脚本会连播两遍。
DUR = 10000.0
AFFECT = (-0.3, 0.45)  # 中等唤醒，两版差异最能看出来


def build(label: str) -> list[dict]:
    phase = ms.PhaseState(noise_seed=SEED, next_blink_ms=ms.initial_blink_ms(SEED))
    heads, _ = ms.generate_dual(AFFECT, None, DUR, phase)
    return heads["voluntary"]


# 新版：当前代码（POSE_RISE_S=0.45, 耦合 -0.125, 眼头协同开）
new_frames = build("new")

# 旧版 = 本轮改动前（三轴等幅）。其余（转移时长/耦合/眼头协同）两版**相同**，
# 以便单独检验「三轴幅度比」这一项的观感贡献——上一轮盲测已确认那批改动有效。
orig_ratio = ms.AXIS_AMPLITUDE_RATIO
ms.AXIS_AMPLITUDE_RATIO = (1.0, 1.0, 1.0)
old_frames = build("old")
ms.AXIS_AMPLITUDE_RATIO = orig_ratio

# 确定性"随机"排序：用种子决定谁先播，主程也不预告顺序
first_is_new = bool(zlib.crc32(str(SEED).encode()) % 2)
order = (
    [("新", new_frames), ("旧", old_frames)]
    if first_is_new
    else [
        ("旧", old_frames),
        ("新", new_frames),
    ]
)

payload = {
    "segments": [
        {"label": name, "keyframes": frames, "duration_s": DUR / 1000.0}
        for name, frames in [("甲", order[0][1]), ("乙", order[1][1])]
    ]
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False)
with open(KEY, "w", encoding="utf-8") as f:
    json.dump({"甲": order[0][0], "乙": order[1][0]}, f, ensure_ascii=False)

print(f"甲/乙 各 {len(new_frames)} 帧，各 {DUR / 1000:.0f} 秒")
print(f"载荷 → {OUT}")
print("答案已单独写入 ab_key.json（播放脚本不读它）")
