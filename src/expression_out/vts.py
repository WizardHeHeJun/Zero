"""皮套表现：把情绪表现成 Live2D 形象的连续动作 + 离散行为。

## 形状：为什么是「持续动 + 每轮叠加」

动作有两种时间结构，本 sink 同时承担：

- **连续轨迹（①情绪直驱）**：形象一直在动（呼吸/漂移/微颤/眨眼），幅度与速度随当前情绪
  变化——不是"说一句动一下"。由后台循环按段（默认 2s）持续投递，相位跨段续接。
- **离散行为（③语义判断）**：每轮回复里"认同/否定/疑问"这类判断，映射成 12 词闭集里的
  点头/摇头/歪头等。由 `emit` 在对话轮次上触发。

轨迹合成复用 `agents.motion_synth`（纯函数、给定 seed 确定），行为抽取复用
`agents.behavior_intent`（12 词闭集守卫在内）——本模块只做**投递**，不含任何动作决策。

## 为什么要跨进程（不是绕路）

渲染在配套项目 Zero_MCP（`VtsExpressionSink` 持 VTS WebSocket），而两仓都用 `src.` 作包根、
不能同进一个 `sys.path`。MCP stdio 天然跨进程，正是为此选的：本进程算，对面渲染，
决策在我方、执行在对方（zero-link 既有契约）。

连接管理（spawn/握手/超时分档/errorID 51 语义）在 `transport.VtsTransport`——因为
⚠ **同一时刻只能一个进程连 VTS**（参数注入独占），语音口型（`speech.py`）与本 sink
必须共享同一条连接（2026-08-14 speech-output T2 抽出，语义零变化）。跑本 sink 前先停掉
`tools/motion/` 下的 live_bridge / loop_vts 等脚本，否则对面会拒连。

## 📎 一段已结案的历史（2026-08-11，别再照着旧结论排查）

本 sink 曾短暂改走「薄 worker 子进程 + 行分隔 JSON」，因为经对方 MCP 调 `vts_connect`
**25 秒挂起**。当时我方猜「anyio 上下文差异」——**被对方实测证伪**。真因是：**FastMCP
stdio server 进入事件循环之后，在工具体里首次 `import numpy`（或任何传递性拉 numpy 的
包）会无限期卡在扩展模块加载**（Windows loader），既不返回也不抛错；进程内直连之所以
好，是因为 import 发生在事件循环**之前**——是**计时窗口**的差别，不是上下文的差别。
判据是「是否首次触达 numpy」而非「是否原生扩展」，`asyncio.to_thread` 包起来也救不了。

对方已修并合 main（预热 import 提到 `mcp.run()` 之前）；我方实测 `vts_connect` 秒回
（`healthy=true`）后**已切回标准 `ClientSession.call_tool`，worker 退役**。

⇒ 我方 server 自查结论：**不踩此雷**（拉 torch/numpy 的解码器构造在 `build_server()` 里、
即 `server.run()` 之前），且已配结构守卫 `tests/test_mcp_native_import_guard.py` 防将来
有人把重依赖 import 挪进工具体（该守卫做过变异验证：埋入即红）。

env：`ZERO_VTS_SINK=true` 开启（默认关=零回归）；`ZERO_VTS_MCP_REPO` 指向 Zero_MCP
（默认取 Zero 仓的兄弟目录）；`ZERO_VTS_TOKEN_FILE` 授权 token 落盘位置。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from pathlib import Path
from typing import Any

from src.agents.behavior_intent import (
    BehaviorIntent,
    lexical_intents,
    merge_intents,
    stage_direction_intents,
)
from src.agents.language_openai import strip_stage_directions_with_segments
from src.agents.motion_synth import (
    PhaseState,
    generate_dual,
    initial_blink_ms,
    modulation_from_affect,
)
from src.expression_out.base import ExpressionFrame

# 常量再导出：超时分档语义属连接层（transport），但既有消费方（测试/文档）从本模块取。
from src.expression_out.transport import (
    CONNECT_TIMEOUT_FIRST_AUTH_S as CONNECT_TIMEOUT_FIRST_AUTH_S,
)
from src.expression_out.transport import (
    CONNECT_TIMEOUT_S as CONNECT_TIMEOUT_S,
)
from src.expression_out.transport import VtsTransport

logger = logging.getLogger(__name__)

DEFAULT_SEGMENT_MS = 2000.0  # 段长：短些让情绪切换跟手（对面契约上限 10s）
DEFAULT_FPS = 20.0


class VtsSink:
    """把 `ExpressionFrame` 表现成皮套动作。实现 `ExpressionSink` 协议。

    未 `connect()` 或连接失败时全程 no-op（`emit` 静默返回）——表现端故障不扳倒对话。
    连接经注入的 `VtsTransport`（可与语音 sink 共享）；不传则自建一条。
    """

    def __init__(
        self,
        *,
        transport: VtsTransport | None = None,
        mcp_repo: Path | None = None,
        token_file: Path | None = None,
        segment_ms: float = DEFAULT_SEGMENT_MS,
        fps: float = DEFAULT_FPS,
        rng_seed: int = 20260811,
    ) -> None:
        # transport 与 mcp_repo/token_file 同传时以 transport 为准（工厂共享装配场景
        # 不会同传；直构场景两参数只是自建 transport 的便捷入口）。
        self.transport = transport or VtsTransport(mcp_repo=mcp_repo, token_file=token_file)
        self.segment_ms = segment_ms
        self.fps = fps
        self.emotion: tuple[float, float] = (0.0, 0.0)
        self.regulated: tuple[float, float] | None = None
        self.phase = PhaseState(noise_seed=rng_seed, next_blink_ms=initial_blink_ms(rng_seed))
        self.loop_task: asyncio.Task[None] | None = None
        self.stop = asyncio.Event()

    def _connect_timeout(self) -> float:
        """委托 transport 的超时分档（保留在此供既有测试/调用方直接钉住）。"""
        return self.transport.connect_timeout()

    async def connect(self) -> bool:
        """经共享 transport 连渲染端并开动作循环；失败返回 False（不抛，可继续纯对话）。

        transport.connect 幂等：语音 sink 先连过的话，这里直接复用同一条会话。
        """
        if not await self.transport.connect():
            return False
        self.stop = asyncio.Event()
        self.loop_task = asyncio.create_task(self._drive())
        return True

    async def emit(self, frame: ExpressionFrame) -> None:
        """接一帧表现内容：更新连续轨迹的驱动情绪 + 触发本轮的离散行为。"""
        self.emotion = frame.emotion
        self.regulated = frame.regulated
        if self.transport.session is None or not frame.reply:
            return
        try:
            intents = self._intents(frame.reply)
            if intents:
                logger.debug("离散行为投递：%s", [intent.name for intent in intents])
            for intent in intents:
                args: dict[str, Any] = {"name": intent.name, "intensity": intent.intensity}
                if intent.direction:
                    args["direction"] = intent.direction
                await self.transport.call_tool("behavior_trigger", args)
        except Exception as exc:
            logger.warning("离散行为投递失败（对话不受影响）：%s", exc)

    async def aclose(self) -> None:
        """停循环、断连接（幂等）。

        ⚠ transport 可能与语音 sink 共享：aclose 在会话收尾时对全部 sink 逐个调用，
        transport.aclose 幂等，后关的一方是 no-op。
        """
        self.stop.set()
        if self.loop_task is not None:
            self.loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.loop_task
            self.loop_task = None
        await self.transport.aclose()

    def _intents(self, reply: str) -> list[BehaviorIntent]:
        """③层离散行为：与 `MotionAgent._events` 同一套判定（含 12 词闭集守卫）。"""
        _, segments = strip_stage_directions_with_segments(reply)
        return merge_intents(lexical_intents(reply), stage_direction_intents(segments))

    async def _drive(self) -> None:
        """后台动作循环：按段持续投轨迹，幅度/速度随**当前**情绪走。

        不等对话轮次——形象一直在动（待机也有呼吸/漂移/眨眼），这是①层情绪直驱的本意。
        投递失败只记日志并退避重试，不终止循环、更不影响对话。
        """
        while not self.stop.is_set():
            try:
                await self._push_segment()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("轨迹投递失败，1 秒后重试：%s", exc)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.stop.wait(), timeout=1.0)
                continue
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.stop.wait(), timeout=self.segment_ms / 1000.0)

    async def _push_segment(self) -> None:
        """合成一段轨迹并投给渲染端。相位由 `self.phase` 跨段续接，接缝不跳变。"""
        if self.transport.session is None:
            return
        modulation = modulation_from_affect(*self.emotion)
        heads, self.phase = generate_dual(
            self.emotion,
            self.regulated,
            self.segment_ms,
            self.phase,
            voluntary_leak=1.0,
            fps=self.fps,
            modulation=modulation,
        )
        await self.transport.call_tool(
            "params_animate",
            {"keyframes": heads["voluntary"], "mode": "absolute", "append": True},
        )
        logger.debug(
            "轨迹段已投递：%.0f ms × %d 帧，emotion=(%.2f, %.2f)%s",
            self.segment_ms,
            len(heads["voluntary"]),
            self.emotion[0],
            self.emotion[1],
            "（含调节通路）" if self.regulated is not None else "",
        )


def build_vts_sink(transport: VtsTransport | None = None) -> VtsSink | None:
    """按 env 装配皮套 sink；`ZERO_VTS_SINK` 未开则返回 None（默认关=零回归）。"""
    raw = os.getenv("ZERO_VTS_SINK", "false").strip().lower()
    if raw not in ("1", "true", "yes", "on"):
        return None
    return VtsSink(transport=transport)
