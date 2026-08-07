"""甲乙交替对比：甲→乙→甲→乙…，每段间报出当前是哪个，便于边看边判断。仍不读答案文件。"""

import _paths as P  # 转正后统一取路径（原为 scratchpad 绝对路径）
import asyncio
import json
import os
import sys

os.environ["VTS_BEHAVIOR_ENABLED"] = "true"
os.environ["VTS_TOKEN_FILE"] = str(P.VTS_TOKEN)
P.use_zero_mcp()

PAYLOAD = str(P.AB_PAYLOAD)
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3


async def main() -> None:
    from src.agents.models.vts_behavior import TrajectoryKeyframe, TrajectoryRequest
    from src.mcp.behavior.service import BehaviorService

    with open(PAYLOAD, encoding="utf-8") as f:
        payload = json.load(f)
    segs = {s["label"]: s for s in payload["segments"]}

    service = BehaviorService()
    st = await service.connect()
    print(f"连接 VTS：healthy={st.healthy}", flush=True)
    for i in range(10, 0, -1):
        print(f"  {i} 秒后开始，请切到 VTube Studio …", flush=True)
        await asyncio.sleep(1.0)

    for r in range(ROUNDS):
        for label in ("甲", "乙"):
            seg = segs[label]
            req = TrajectoryRequest(
                keyframes=[
                    TrajectoryKeyframe(t_ms=f["t_ms"], params=f["params"]) for f in seg["keyframes"]
                ],
                mode="absolute",
                append=False,
            )
            rec = service.animate(req)
            print(f"▶▶ 第{r + 1}轮 【{label}】 → {rec.status}", flush=True)
            await asyncio.sleep(seg["duration_s"] + 0.6)
            service.clear_params()
            await asyncio.sleep(1.2)  # 短暂归位，避免两段首尾粘连

    await service.disconnect()
    print("\n交替对比结束。", flush=True)


asyncio.run(main())
