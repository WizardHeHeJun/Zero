"""皮套渲染 worker：在**配套项目 Zero_MCP 的 sys.path 下**跑，进程内直连 VTS。

## 为什么是独立子进程（而不是本进程直接连）

两仓都用 `src.` 作包根、不能同进一个 `sys.path`——渲染实现（`BehaviorService` /
`VtsExpressionSink`）在对面仓，本进程（Zero）装着对话与情感引擎，只能隔进程调。

## 为什么不用对方的 MCP server（2026-08-11 实测决定）

对方有现成的 `vts_behavior_mcp_server`（暴露 `vts_connect`/`params_animate`/
`behavior_trigger`），首选本该是它。但实测：**经 MCP stdio 调 `vts_connect` 25 秒不返回**
（卡在 `BehaviorService.connect()` 里的 VTS 握手/枚举阶段），而**同一份代码在普通进程内
直连秒通**（`healthy=True`、轨迹 `accepted`）——对照实验两侧只差"是否跑在 MCP server 的
anyio 上下文里"。这是对方仓的问题，已跨仓通报；在它修好前，本 worker 走已验证可靠的
进程内直连，只把 IPC 降级成一条极简的行分隔 JSON 协议：

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
