"""只播盲测里的某一段（默认甲），可重复播放。不读答案文件。

用法：python play_one.py [甲|乙] [重复次数]
"""

import _paths as P  # 转正后统一取路径（原为 scratchpad 绝对路径）
import asyncio
import json
import os
import sys

os.environ["VTS_BEHAVIOR_ENABLED"] = "true"
os.environ["VTS_TOKEN_FILE"] = str(P.VTS_TOKEN)
P.use_zero_mcp()

PAYLOAD = str(P.AB_PAYLOAD)
WANT = sys.argv[1] if len(sys.argv) > 1 else "甲"
REPEAT = int(sys.argv[2]) if len(sys.argv) > 2 else 1


async def main() -> None:
    from src.agents.models.vts_behavior import TrajectoryKeyframe, TrajectoryRequest
    from src.mcp.behavior.service import BehaviorService

    with open(PAYLOAD, encoding="utf-8") as f:
        payload = json.load(f)
    seg = next(s for s in payload["segments"] if s["label"] == WANT)

    service = BehaviorService()
    st = await service.connect()
    print(f"连接 VTS：healthy={st.healthy}", flush=True)

    for i in range(10, 0, -1):
        print(f"  {i} 秒后播放【{WANT}】，请切到 VTube Studio …", flush=True)
        await asyncio.sleep(1.0)

    for k in range(REPEAT):
        req = TrajectoryRequest(
            keyframes=[
                TrajectoryKeyframe(t_ms=f["t_ms"], params=f["params"]) for f in seg["keyframes"]
            ],
            mode="absolute",
            append=False,
        )
        r = service.animate(req)
        print(f"\n▶▶ 【{WANT}】第 {k + 1}/{REPEAT} 遍（8 秒） → {r.status}", flush=True)
        await asyncio.sleep(seg["duration_s"] + 0.8)
        if k + 1 < REPEAT:
            service.clear_params()
            await asyncio.sleep(1.5)

    await service.disconnect()
    print(f"\n【{WANT}】播完。", flush=True)


asyncio.run(main())
