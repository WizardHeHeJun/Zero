"""按序播放幅度梯度（`gen_ladder.py` 产出），每段报出档位号，看完直接说要第几档。

与 `play_alt.py` 的区别：那个是甲乙二选一（盲测「哪个更自然」），这个是按序递增
让眼睛定量级——故**如实报档位号**，不藏。

在 `d:\\Zero_MCP` 目录下跑（两仓不能同时进 sys.path）：
    python d:\\Zero\\tools\\motion\\play_ladder.py [轮数]
"""

import asyncio
import json
import os
import sys

import _paths as P

os.environ["VTS_BEHAVIOR_ENABLED"] = "true"
os.environ["VTS_TOKEN_FILE"] = str(P.VTS_TOKEN)
P.use_zero_mcp()

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 2


async def main() -> None:
    from src.agents.models.vts_behavior import TrajectoryKeyframe, TrajectoryRequest
    from src.mcp.behavior.service import BehaviorService

    with open(P.AB_PAYLOAD, encoding="utf-8") as fh:
        payload = json.load(fh)
    segments = payload["segments"]

    service = BehaviorService()
    status = await service.connect()
    print(f"连接 VTS：healthy={status.healthy}", flush=True)
    for i in range(10, 0, -1):
        print(f"  {i} 秒后开始，请切到 VTube Studio …", flush=True)
        await asyncio.sleep(1.0)

    for round_index in range(ROUNDS):
        for segment in segments:
            request = TrajectoryRequest(
                keyframes=[
                    TrajectoryKeyframe(t_ms=f["t_ms"], params=f["params"])
                    for f in segment["keyframes"]
                ],
                mode="absolute",
                append=False,
            )
            receipt = service.animate(request)
            # 安全标注**必须报出来**：不安全的档照样播（你要看得到），但不能不知情地选中
            note = {
                "ok": "",
                "no_margin": "  ⚠ 该档 margin 归零（激动时贴满量程）",
                "clips": "  ❌ 该档激动时会削平波形",
            }.get(segment.get("safety", "ok"), "")
            print(
                f"▶▶ 第{round_index + 1}轮 【第 {segment['label']} 档】"
                f"（{segment['factor']:.2f}×）→ {receipt.status}{note}",
                flush=True,
            )
            await asyncio.sleep(segment["duration_s"] + 0.6)
            service.clear_params()
            await asyncio.sleep(1.2)  # 短暂归位，避免两段首尾粘连

    await service.disconnect()
    print("\n梯度播放结束——直接说要第几档（或说都不够大，我再扩上界）。", flush=True)


asyncio.run(main())
