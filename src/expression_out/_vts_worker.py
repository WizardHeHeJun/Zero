"""皮套渲染 worker：在**配套项目 Zero_MCP 的 sys.path 下**跑，进程内直连 VTS。

## 为什么是独立子进程（而不是本进程直接连）

两仓都用 `src.` 作包根、不能同进一个 `sys.path`——渲染实现（`BehaviorService` /
`VtsExpressionSink`）在对面仓，本进程（Zero）装着对话与情感引擎，只能隔进程调。

## 为什么不用对方的 MCP server（2026-08-11 实测发现 → 对方已修，本接法待切回）

对方有现成的 `vts_behavior_mcp_server`（暴露 `vts_connect`/`params_animate`/
`behavior_trigger`），首选本该是它。但实测：**经 MCP stdio 调 `vts_connect` 25 秒不返回**，
而**同一份代码在普通进程内直连秒通**。

⚠ **根因订正**（对方回执查穿，我方原猜的「anyio 上下文差异」**被证伪**）：真因是
**FastMCP stdio server 进入事件循环后，工具体里首次 `import numpy`（或传递性拉 numpy 的
包）会无限期卡在扩展模块加载**——对方 `_get_service()` 的延迟 import 正踩此坑；而进程内
直连之所以好，是因为 import 发生在事件循环**之前**（计时窗口的差别，不是上下文的差别）。

对方已修（预热 import 提到 `mcp.run()` 之前）。**切回条件已满足**：拉到对方含该修复的
main 后，把 `VtsSink._rpc` 换回 `ClientSession.call_tool`，本 worker 即可退役；在此之前
它走已验证可靠的进程内直连，只把 IPC 降级成一条极简的行分隔 JSON 协议：

    stdin  ← {"op": "connect"} / {"op": "animate", "keyframes": [...]}
             / {"op": "behavior", "name": "nod", "intensity": 0.6, "direction": null}
             / {"op": "close"}
    stdout → {"ok": true, ...} 每条一行（父进程只读不解析细节，失败只记日志）

⚠ 本文件**不得 import 任何 Zero 侧模块**——它运行在对面的 sys.path 下。
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


async def main() -> None:
    from src.agents.models.vts_behavior import (
        BehaviorRequest,
        TrajectoryKeyframe,
        TrajectoryRequest,
    )
    from src.mcp.behavior.service import BehaviorService

    service = BehaviorService()
    connected = False
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:  # 父进程关了 stdin ⇒ 收工
            break
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _emit({"ok": False, "error": "bad-json"})
            continue
        op = msg.get("op")
        try:
            if op == "connect":
                status = await service.connect()
                connected = True
                _emit({"ok": True, "healthy": status.healthy, "model_id": status.model_id})
            elif op == "animate":
                request = TrajectoryRequest(
                    keyframes=[
                        TrajectoryKeyframe(t_ms=f["t_ms"], params=f["params"])
                        for f in msg["keyframes"]
                    ],
                    mode=msg.get("mode", "absolute"),
                    append=msg.get("append", True),
                )
                receipt = service.animate(request)
                _emit({"ok": True, "status": receipt.status})
            elif op == "behavior":
                receipt = await service.trigger(
                    BehaviorRequest(
                        name=msg["name"],
                        intensity=msg.get("intensity", 0.5),
                        direction=msg.get("direction"),
                    )
                )
                _emit({"ok": True, "status": receipt.status})
            elif op == "close":
                if connected:
                    await service.disconnect()
                _emit({"ok": True})
                break
            else:
                _emit({"ok": False, "error": f"unknown-op:{op}"})
        except Exception as exc:  # 渲染故障不该杀掉 worker——回执给父进程，继续收指令
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    asyncio.run(main())
