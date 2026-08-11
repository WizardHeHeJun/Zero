"""把动作层**接入实跑**：MCP 客户端拉 `zero.motion`，直驱 VTube Studio。

## 为什么是这个形状

设计定的接口就是 `zero.motion`——**独立拉取、不推进引擎**（`PRP/motion/design-agent.md`）。
两仓都用 `src.` 作包根不能同进 sys.path，而 MCP 走 **stdio 跨进程**，天然绕开这个冲突：
本脚本跑在 Zero_MCP 侧（有 VTS），Zero 的 MCP server 是它 spawn 的**子进程**。
这正是当初选 MCP 的理由，不是绕路。

✅ **2026-08-07 实测：Zero 自己的 MCP server 的 `call_tool` 不挂起**（每次秒回）。
交接文档第五节记的「call_tool 挂起」是**对方**那个 server 的问题，别把它当成本侧的阻塞——
我按那条记录本来差点绕开 MCP 去做文件桥。

## 节拍

- **动作按帧连续**（20fps，段长 1~10s）：本脚本自己的时钟循环，每段拉一次。
- **情绪按回合更新**：`--stim` 给的刺激推进引擎；不给就纯待机。
- 相位由 server 侧 per-session 保管（`phase_ms`），**跨段续接不跳变**——所以必须
  一直用同一个 `session_id`，换 id 会让轨迹在拼接点跳一下。

在 `d:\\Zero_MCP` 目录下跑（VTS 要开着）：

    python d:\\Zero\\tools\\motion\\live_bridge.py --seconds 60
    python d:\\Zero\\tools\\motion\\live_bridge.py --seconds 60 --stim=-0.6,0.8  # 带情绪

⚠ `--stim` 的负 valence **必须用 `=` 形式**（`--stim=-0.6,0.8`）：argparse 会把以 `-`
开头的值当成选项名，空格形式只在 valence 为正时能用。

不连 VTS 只验链路（这条可以在 Zero 侧跑）：

    python live_bridge.py --dry-run --seconds 20 --stim=-0.6,0.8

dry-run 会打印每段的 `phase_ms` 与**拼接差**——跨段续接正常时应为 0.000°；
非零说明相位没接上（换过 `session_id`，或 server 中途重启过）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import _paths as P

os.environ.setdefault("VTS_BEHAVIOR_ENABLED", "true")
os.environ.setdefault("VTS_TOKEN_FILE", str(P.VTS_TOKEN))

SEGMENT_MS = 4000.0  # 契约上限 10s；取 4s 让情绪变化的响应不至于迟钝
TOOL_PREFIX = "zero."


def server_env(feedback: bool = False) -> dict[str, str]:
    """Zero MCP server 子进程的环境。

    🛑 门控必须显式打开——它们默认关是**故意的**（零回归），不是遗漏：
    - `ZERO_MOTION_ENABLED`：`zero.motion` 工具的总开关，关着会回 `motion-disabled`。
    - `ZERO_MCP_MOTION_BACKEND`：让图内 MotionAgent 产出 `motion_directive`
      （`efference` 档额外留 `motion_efference` 副本）。`synth`（默认）也能出轨迹，
      只是走拉取侧解析回退。

    ⚠ **2026-08-11 修的真缺陷**：此前这里设的是 `ZERO_MOTION_BACKEND`——那是
    `chat_driver` 读的 env，**MCP server 侧根本不读它**（`_build_session_config` 只认
    `ZERO_MCP_*` 治理 env）。实测本脚本历来跑的都是 `motion_backend="synth"`，
    即 `directive` 档在这条真机路径上**从未真正被验证过**（PROGRESS/交接文档记的
    「端到端跑通」成立，但走的是拉取侧现算那条路）。

    `feedback=True` 时另开行为反馈流的**生效组合**（行为反馈环第二步）：
    副本档 `efference` + 总门 + `gate_fusion=false`——默认硬门下行为流的 salience
    恒低于阈值、必被滤除（`PRP/behavior-feedback-loop/design.md` §三），
    不关硬门就等于没开。调节侧（`regulation_enabled`）不是治理旗标，经 open_session
    的 config overrides 传（见 `run()`）。
    """
    env = dict(os.environ)
    backend = os.getenv("ZERO_MCP_MOTION_BACKEND", "efference" if feedback else "directive")
    env.update(
        {
            "ZERO_MOTION_ENABLED": "true",
            "ZERO_MCP_MOTION_BACKEND": backend,
            "PYTHONPATH": str(P.REPO),
            "PYTHONIOENCODING": "utf-8",
        }
    )
    if feedback:
        env.update(
            {
                "ZERO_MCP_BEHAVIOR_FEEDBACK": "true",
                "ZERO_MCP_IGNITION_GATE_FUSION": "false",  # 关硬门，否则行为流恒被滤除
            }
        )
    return env


async def run(
    seconds: float,
    session_id: str,
    stim: tuple[float, float] | None,
    dry_run: bool,
    lead: float = 5.0,
    verify: bool = False,
    feedback: bool = False,
    stim_every: float = 0.0,
) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    service = None
    if not dry_run:
        # ⚠ 只在真驱动时才进 Zero_MCP 的 sys.path——dry-run 下不碰它，
        #   于是这条路径在 Zero 侧也能跑（用来验 MCP 拉取与相位续接，不需要 VTS）。
        P.use_zero_mcp()
        from src.mcp.behavior.service import BehaviorService

        service = BehaviorService()
        status = await service.connect()
        print(f"连接 VTS：healthy={status.healthy}", flush=True)
    else:
        print("dry-run：只走 MCP 拉取，不连 VTS", flush=True)

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server"],
        env=server_env(feedback),
        cwd=str(P.REPO),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as mcp:
            await mcp.initialize()
            open_args: dict[str, object] = {"session_id": session_id}
            if feedback:
                # regulation 不是治理旗标，经 config overrides 传；行为反馈流的在场门是
                # 「调节真改变了表达」（副本 voluntary 非 None），不开调节则流恒缺席。
                open_args["config"] = {"regulation_enabled": True}
            await mcp.call_tool(f"{TOOL_PREFIX}open_session", open_args)
            # 自报实际生效档位：env 名写错时静默退回默认档是本脚本踩过的真坑
            # （2026-08-11：设 ZERO_MOTION_BACKEND 却被 MCP 侧忽略，一直跑 synth）。
            gates = server_env(feedback)
            print(
                f"引擎会话 {session_id} 已开 · backend={gates['ZERO_MCP_MOTION_BACKEND']}"
                f" · 行为反馈={'on' if feedback else 'off'}",
                flush=True,
            )

            async def push_stim(tag: str) -> None:
                assert stim is not None
                out = await mcp.call_tool(
                    f"{TOOL_PREFIX}step",
                    {"session_id": session_id, "stim": {"valence": stim[0], "arousal": stim[1]}},
                )
                body = json.loads(getattr(out.content[0], "text", "{}"))
                print(f"{tag} {stim} ⇒ e*={body.get('valence_arousal')}", flush=True)

            if stim is not None:
                await push_stim("注入刺激")

            if service is not None:
                for i in range(int(lead), 0, -1):
                    print(f"  {i} 秒后开始，请切到 VTube Studio …", flush=True)
                    await asyncio.sleep(1.0)

            elapsed = 0.0
            last_tail: dict[str, float] | None = None
            next_stim_at = stim_every if (stim is not None and stim_every > 0) else None
            while elapsed < seconds:
                # 多轮推进（行为反馈环第二步验收所需）：副本是**上一回合**的运动指令，
                # 单轮 step 下行为反馈流必然缺席（无副本可读）——要看它真的在场，
                # 引擎至少要推进两轮。这也更贴近真实对话（每轮回复推进一次引擎）。
                if next_stim_at is not None and elapsed >= next_stim_at:
                    await push_stim(f"↻ {elapsed:5.1f}s 重注刺激")
                    next_stim_at += stim_every
                result = await mcp.call_tool(
                    f"{TOOL_PREFIX}motion",
                    {"session_id": session_id, "duration_ms": SEGMENT_MS},
                )
                if result.isError:
                    text = getattr(result.content[0], "text", "")
                    raise RuntimeError(f"zero.motion 失败：{text}")
                payload = json.loads(getattr(result.content[0], "text", "{}"))
                frames = payload.get("keyframes") or []
                # 相位续接自查：上一段末帧与本段首帧的角度差应很小——
                # 跳变说明 phase 没接上（换过 session_id / server 重启过）。
                seam = None
                if last_tail is not None and frames:
                    head = frames[0]["params"]
                    seam = max(abs(head[k] - last_tail[k]) for k in ("FaceAngleX", "FaceAngleY"))
                if frames:
                    last_tail = frames[-1]["params"]

                events = payload.get("events") or []
                if service is None:
                    print(
                        f"▶ {elapsed:5.1f}s  {len(frames)} 帧 · phase_ms={payload.get('phase_ms')}"
                        + (f" · 拼接差 {seam:.3f}°" if seam is not None else ""),
                        flush=True,
                    )
                    await asyncio.sleep(0.05)  # dry-run 不等真实时长
                    elapsed += SEGMENT_MS / 1000.0
                    continue

                from src.agents.models.vts_behavior import TrajectoryKeyframe, TrajectoryRequest

                request = TrajectoryRequest(
                    keyframes=[
                        TrajectoryKeyframe(t_ms=f["t_ms"], params=f["params"]) for f in frames
                    ],
                    mode="absolute",
                    append=False,
                )
                receipt = service.animate(request)
                # ⚠ events **由我方转投** `behavior_trigger`——对方不解析我方返回体
                #   （跨仓契约已明确：决策在我方、执行在你方）。
                for event in events:
                    from src.agents.models.vts_behavior import BehaviorRequest

                    await service.trigger(BehaviorRequest(**event))
                print(
                    f"▶ {elapsed:5.1f}s  {len(frames)} 帧 → {receipt.status}"
                    + (f"  events={[e['name'] for e in events]}" if events else ""),
                    flush=True,
                )
                if verify:
                    # ⚠ 回读**必须在段中途**取：段末参数已走到末帧，且注入需 ≥1Hz 重发才不回落。
                    #   读回值与我们发的关键帧对不上 ⇒ 注入没落到模型上（不是"用户没看到"）。
                    await asyncio.sleep(SEGMENT_MS / 2000.0)
                    live = await service.sink.api.request("InputParameterListRequest")
                    table = {
                        p["name"]: float(p.get("value", 0.0))
                        for p in [
                            *live.get("defaultParameters", []),
                            *live.get("customParameters", []),
                        ]
                    }
                    mid = frames[len(frames) // 2]["params"]
                    diffs = {
                        k: (mid[k], table.get(k))
                        for k in ("FaceAngleX", "FaceAngleY", "FaceAngleZ")
                    }
                    parts = []
                    for key, (sent, read) in diffs.items():
                        got = read if read is not None else float("nan")
                        parts.append(f"{key}发{sent:+.2f}/读{got:+.2f}")
                    print("    回读：" + " · ".join(parts), flush=True)
                    await asyncio.sleep(SEGMENT_MS / 2000.0)
                else:
                    await asyncio.sleep(SEGMENT_MS / 1000.0)
                elapsed += SEGMENT_MS / 1000.0

            await mcp.call_tool(f"{TOOL_PREFIX}close_session", {"session_id": session_id})

    if service is not None:
        await service.disconnect()
    print("\n结束。", flush=True)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seconds", type=float, default=60.0, help="驱动时长")
parser.add_argument("--session", default="live", help="引擎会话 id（换 id 会让相位跳变）")
parser.add_argument(
    "--stim",
    default=None,
    help="注入一条刺激，格式 valence,arousal。⚠ 负值须用 = 形式：--stim=-0.6,0.8",
)
parser.add_argument("--dry-run", action="store_true", help="不连 VTS，只验 MCP 拉取与跨段相位续接")
parser.add_argument("--lead", type=float, default=5.0, help="开播前的倒计时秒数（够你切窗口）")
parser.add_argument(
    "--verify", action="store_true", help="播放中途回读 VTS 实际参数值，证明注入真的落到模型上"
)
parser.add_argument(
    "--feedback",
    action="store_true",
    help="开行为反馈流的生效组合（efference 副本档 + 总门 + 关硬门 + regulation）",
)
parser.add_argument(
    "--stim-every",
    type=float,
    default=0.0,
    dest="stim_every",
    help="每隔 N 秒重注一次 --stim（多轮推进引擎；行为反馈流需 ≥2 轮才在场）",
)
args = parser.parse_args()

stimulus: tuple[float, float] | None = None
if args.stim:
    parts = args.stim.split(",")
    if len(parts) != 2:
        raise SystemExit("--stim 格式为 valence,arousal，如 -0.6,0.8")
    stimulus = (float(parts[0]), float(parts[1]))

asyncio.run(
    run(
        args.seconds,
        args.session,
        stimulus,
        args.dry_run,
        args.lead,
        args.verify,
        args.feedback,
        args.stim_every,
    )
)
