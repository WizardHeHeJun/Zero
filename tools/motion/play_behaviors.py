"""离散行为巡演：把 12 词闭集逐个放到皮套上看一遍。

与轨迹类脚本（`play_alt` / `play_ladder`）**是两条不同的通路**：
- 轨迹通路：Zero 侧 `motion_synth` 产出连续关键帧 → `BehaviorService.animate`。走的是「怎么动」。
- 本脚本：离散行为词 → `BehaviorService.trigger`。走的是「做什么动作」（点头/摇头/歪头…）。
  运行时由 `behavior_intent`（词法 + 舞台说明路由）从回复文本里判出来，
  ⚠ 且**由我方转投** `behavior_trigger`——对方不解析我方返回体（跨仓契约已明确）。

有方向的词会把每个方向都放一遍（`head_tilt` 左右、`glance` 四向）。
不可用/降级的词会如实报出来（对方 `resolve_degradation` 的判定），不静默跳过。

在 `d:\\Zero_MCP` 目录下跑（两仓不能同时进 sys.path）：
    python d:\\Zero\\tools\\motion\\play_behaviors.py [每词间隔秒]
"""

import asyncio
import os
import sys

import _paths as P

os.environ["VTS_BEHAVIOR_ENABLED"] = "true"
os.environ["VTS_TOKEN_FILE"] = str(P.VTS_TOKEN)
P.use_zero_mcp()

GAP_S = float(sys.argv[1]) if len(sys.argv) > 1 else 2.5

# 与 Zero 侧 `behavior_intent.BEHAVIOR_VOCABULARY` 同一份 12 词（此处按「先看头部大动作、
# 再看细微表情」的观看顺序排，不是字母序）。方向词的候选方向由对方 spec 决定，
# 本脚本不写死——写死会在对方加方向时静默漏播。
VIEW_ORDER = (
    "nod",
    "shake",
    "head_tilt",
    "glance",
    "lean_in",
    "lean_back",
    "body_sway",
    "brow_raise",
    "brow_furrow",
    "eyes_widen",
    "smile",
    "blink",
)


async def main() -> None:
    from src.agents.models.vts_behavior import BehaviorRequest
    from src.mcp.behavior.service import BehaviorService
    from src.mcp.zero.sinks.behavior_overlay import VOCABULARY

    service = BehaviorService()
    status = await service.connect()
    print(f"连接 VTS：healthy={status.healthy}", flush=True)
    for i in range(10, 0, -1):
        print(f"  {i} 秒后开始，请切到 VTube Studio …", flush=True)
        await asyncio.sleep(1.0)

    for name in VIEW_ORDER:
        # 方向集**从对方词表实测取**（`behavior_overlay.VOCABULARY`，非 service），
        # 不写死——写死会在对方加方向时静默漏播。
        # ⚠ 也**不包 try/except 静默退化**：拿不到就是接口对不上，该炸出来让人去核；
        #   悄悄退成「无方向单播」等于漏看一半动作还以为看全了。
        spec = VOCABULARY[name]
        directions: tuple[str | None, ...] = (
            spec.directions if spec.directions is not None else (None,)
        )

        for direction in directions:
            request = BehaviorRequest(name=name, direction=direction)
            receipt = await service.trigger(request)
            label = f"{name}" + (f"（{direction}）" if direction else "")
            detail = ""
            degraded = getattr(receipt, "degraded_channels", None)
            if degraded:
                detail = f"  ⚠ 降级通道 {degraded}"
            print(f"▶▶ {label:24s} → {receipt.status}{detail}", flush=True)
            await asyncio.sleep(GAP_S)

    await service.disconnect()
    print("\n巡演结束。哪个动作想调（幅度/时长/方向），直接说名字。", flush=True)


asyncio.run(main())
