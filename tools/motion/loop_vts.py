"""持续循环驱动 VTS：情绪在 平静→中性→激动 之间缓慢巡回，一直跑到被停止。

不再一次性播完 —— 用户多次错过观看时机，改为常驻，随时可看。
轨迹由 Zero 侧 motion_synth 预生成成 JSON（避免两仓 `src.` 包根冲突）。
"""

import _paths as P  # 转正后统一取路径（原为 scratchpad 绝对路径）
import asyncio
import json
import os
import sys

os.environ["VTS_BEHAVIOR_ENABLED"] = "true"
os.environ["VTS_TOKEN_FILE"] = str(P.VTS_TOKEN)
P.use_zero_mcp()

LOOP_PAYLOAD = str(P.LOOP_PAYLOAD)


async def main() -> None:
    from src.agents.models.vts_behavior import (
        BehaviorRequest,
        TrajectoryKeyframe,
        TrajectoryRequest,
    )
    from src.mcp.behavior.service import BehaviorService

    with open(LOOP_PAYLOAD, encoding="utf-8") as f:
        payload = json.load(f)
    segments = payload["segments"]
    gestures = payload["gestures"]

    service = BehaviorService()
    st = await service.connect()
    print(f"连接 VTS：connected={st.connected} healthy={st.healthy}", flush=True)
    print("▶ 开始循环播放，随时可看。停止：告诉我一声即可。\n", flush=True)

    round_no = 0
    try:
        while True:
            round_no += 1
            for seg in segments:
                req = TrajectoryRequest(
                    keyframes=[
                        TrajectoryKeyframe(t_ms=k["t_ms"], params=k["params"])
                        for k in seg["keyframes"]
                    ],
                    mode="absolute",
                    append=True,
                )
                r = service.animate(req)
                print(
                    f"[第{round_no}轮] {seg['label']:6s} 峰值={seg['peak']:5.2f}° → {r.status}"
                    + (f" [{r.code}]" if r.code else ""),
                    flush=True,
                )
                # 队列满就退避，别把 queue 打爆（对面 MAX_QUEUE=5）
                while r.queue_depth is not None and r.queue_depth >= 3:
                    await asyncio.sleep(1.5)
                    break
                await asyncio.sleep(seg["duration_s"] * 0.9)

            # 每轮末尾穿插一个离散动作，好对比「轨迹」与「行为词」的差别
            g = gestures[round_no % len(gestures)]
            rec = await service.trigger(BehaviorRequest(name=g, intensity=0.65))
            print(f"[第{round_no}轮] 离散行为 {g} → {rec.status}", flush=True)
            await asyncio.sleep(2.0)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await service.disconnect()
        print("已断开，VTS 收回参数控制权。", flush=True)


asyncio.run(main())
