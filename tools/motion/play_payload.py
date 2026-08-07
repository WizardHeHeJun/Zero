"""第二步（仅 Zero_MCP 侧）：读 JSON 载荷，经 BehaviorService 驱动 VTube Studio。"""

import _paths as P  # 转正后统一取路径（原为 scratchpad 绝对路径）
import asyncio
import json
import os
import sys

os.environ["VTS_BEHAVIOR_ENABLED"] = "true"
os.environ["VTS_TOKEN_FILE"] = str(P.VTS_TOKEN)
P.use_zero_mcp()

PAYLOAD = str(P.MOTION_PAYLOAD)


async def main() -> None:
    from src.agents.models.vts_behavior import (
        BehaviorRequest,
        TrajectoryKeyframe,
        TrajectoryRequest,
    )
    from src.mcp.behavior.service import BehaviorService

    with open(PAYLOAD, encoding="utf-8") as f:
        payload = json.load(f)

    service = BehaviorService()
    print("[1/4] 连接 VTS …", flush=True)
    st = await service.connect()
    print(
        f"      connected={st.connected} healthy={st.healthy} model_id={st.model_id} "
        f"热键={st.hotkey_count} 缺席参数={st.unavailable_params}",
        flush=True,
    )

    cat = service.list_params()
    names = {p.name for p in (cat.params or [])}
    print(f"      皮套暴露 {len(names)} 个输入参数", flush=True)
    for want in ("FaceAngleX", "FaceAngleY", "FaceAngleZ", "EyeOpenLeft"):
        print(f"        {want:12s} {'✓ 可驱动' if want in names else '✗ 缺席'}", flush=True)

    # ⚠ 倒计时必须在 animate() **之前**：轨迹在投喂那一刻就开始播，
    # 放在之后等于倒计时期间动作已经放完了（上一版就是这个错）。
    print("\n[2/4] 准备投喂轨迹", flush=True)
    for i in (8, 7, 6, 5, 4, 3, 2, 1):
        print(f"      ▶ {i} 秒后开始，请切到 VTube Studio 窗口 …", flush=True)
        await asyncio.sleep(1.0)

    for seg in payload["segments"]:
        req = TrajectoryRequest(
            keyframes=[
                TrajectoryKeyframe(t_ms=k["t_ms"], params=k["params"]) for k in seg["keyframes"]
            ],
            mode="absolute",
            append=True,
        )
        r = service.animate(req)
        print(
            f"      {seg['label']}  {len(seg['keyframes'])}帧 sd={seg['sd']}° 峰值={seg['peak']}°"
            f"  → {r.status} queue={r.queue_depth}"
            + (f" code={r.code}" if r.code else "")
            + (f" 丢弃={r.dropped_params}" if r.dropped_params else ""),
            flush=True,
        )

    print("      ▶▶ 开始：前 6 秒【平静】——几乎不动，只有极轻微摆动和眨眼", flush=True)
    await asyncio.sleep(6.2)
    print("      ▶▶ 现在：后 6 秒【激动】——摆动明显变大变快", flush=True)
    await asyncio.sleep(6.3)
    print("      ▶▶ 轨迹播完", flush=True)

    print("\n[3/4] 离散行为（真实回复文本推导）", flush=True)
    for ev in payload["events"]:
        if not ev["acts"]:
            print(f"      「{ev['reply']}」→ 无动作（物理世界宣称被闭集挡下）", flush=True)
            continue
        for act in ev["acts"]:
            kw = {"name": act["name"], "intensity": act["intensity"]}
            if act["direction"]:
                kw["direction"] = act["direction"]
            r = await service.trigger(BehaviorRequest(**kw))
            print(
                f"      「{ev['reply'][:11]}…」→ {act['name']:12s} {r.status}"
                + (f" [{r.code}]" if r.code else ""),
                flush=True,
            )
            await asyncio.sleep(1.8)

    print("\n[4/4] 收尾", flush=True)
    st = service.status()
    print(f"      healthy={st.healthy} 活跃行为={len(st.active)} 冷却中={st.cooldowns}", flush=True)
    await service.disconnect()
    print("      已断开", flush=True)


asyncio.run(main())
