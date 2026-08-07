"""第一步（仅 Zero 侧）：生成轨迹 + 行为意图，落 JSON。

两仓都用 `src.` 作包根，同时进 sys.path 会互相覆盖 —— 故拆两步，各自干净。
"""

import json
import statistics

import _paths as P  # 转正后统一取路径（原为 scratchpad 绝对路径）

P.use_zero()

from src.agents.behavior_intent import (  # noqa: E402
    lexical_intents,
    merge_intents,
    stage_direction_intents,
)
from src.agents.language_openai import strip_stage_directions_with_segments  # noqa: E402
from src.agents.motion_synth import (  # noqa: E402
    PARAM_ANGLE_X,
    PhaseState,
    generate_dual,
    initial_blink_ms,
)

OUT = str(P.MOTION_PAYLOAD)

seed = 20260806
phase = PhaseState(noise_seed=seed, next_blink_ms=initial_blink_ms(seed))
segments = []
for label, affect in (("平静", (0.2, -0.75)), ("激动", (-0.6, 0.85))):
    heads, phase = generate_dual(affect, None, 6000.0, phase)
    frames = heads["voluntary"]
    xs = [f["params"][PARAM_ANGLE_X] for f in frames]
    segments.append(
        {
            "label": label,
            "keyframes": frames,
            "sd": round(statistics.pstdev(xs), 2),
            "peak": round(max(abs(v) for v in xs), 2),
        }
    )
    print(f"{label}: {len(frames)} 帧 sd={segments[-1]['sd']}° 峰值={segments[-1]['peak']}°")

replies = [
    "（点了点头）嗯，我明白了。",
    "不是吧，我没说过。",
    "（皱了皱眉）你说的是哪一次？",
    "（我帮你把灯关了）现在好点没？",
]
events = []
for reply in replies:
    _, segs = strip_stage_directions_with_segments(reply)
    intents = merge_intents(lexical_intents(reply), stage_direction_intents(segs))
    events.append(
        {
            "reply": reply,
            "acts": [
                {"name": i.name, "intensity": i.intensity, "direction": i.direction}
                for i in intents
            ],
        }
    )
    names = ", ".join(i.name for i in intents) or "（无——闭集挡下）"
    print(f"「{reply}」→ {names}")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"segments": segments, "events": events}, f, ensure_ascii=False)
print(f"\n已写入 {OUT}")
