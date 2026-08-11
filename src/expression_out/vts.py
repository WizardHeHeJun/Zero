"""皮套表现：把情绪表现成 Live2D 形象的连续动作 + 离散行为。

## 形状：为什么是「持续动 + 每轮叠加」

动作有两种时间结构，本 sink 同时承担：

- **连续轨迹（①情绪直驱）**：形象一直在动（呼吸/漂移/微颤/眨眼），幅度与速度随当前情绪
  变化——不是"说一句动一下"。由后台循环按段（默认 2s）持续投递，相位跨段续接。
- **离散行为（③语义判断）**：每轮回复里"认同/否定/疑问"这类判断，映射成 12 词闭集里的
  点头/摇头/歪头等。由 `emit` 在对话轮次上触发。

轨迹合成复用 `agents.motion_synth`（纯函数、给定 seed 确定），行为抽取复用
`agents.behavior_intent`（12 词闭集守卫在内）——本模块只做**投递**，不含任何动作决策。

## ⚠ 根因已由对方查穿（2026-08-11 回执，订正下方我方原猜测）

我方原以为是「anyio 上下文 vs 普通上下文」的差别——**被证伪**。真因是：**FastMCP stdio
server 进入事件循环之后，在工具体里首次 `import numpy`（或任何传递性拉 numpy 的包）会
无限期卡在扩展模块加载**（Windows loader），既不返回也不抛错。对方 `_get_service()` 的
延迟 import 链路正好首次拉起 numpy。判据是「**是否首次触达 numpy**」而非「是否原生扩展」，
且 `asyncio.to_thread` 包起来也救不了。对方已修（预热 import 提到 `mcp.run()` 之前）。

⇒ **切回标准 MCP 的条件已满足**：拉到对方含该修复的 main 后，把下面 `_rpc` 换回
`ClientSession.call_tool` 即可，上层协议不动。
⇒ 我方 server 自查结论：**不踩此雷**（拉 torch/numpy 的解码器构造在 `build_server()` 里、
即 `server.run()` 之前），且已配结构守卫 `tests/test_mcp_native_import_guard.py` 防将来
有人把重依赖 import 挪进工具体（该守卫做过变异验证：埋入即红）。

## 为什么要跨进程（不是绕路）

渲染在配套项目 Zero_MCP（`VtsExpressionSink` 持 VTS WebSocket），而两仓都用 `src.` 作包根、
不能同进一个 `sys.path`。MCP stdio 天然跨进程，正是为此选的：本进程算，对面渲染，
决策在我方、执行在对方（zero-link 既有契约）。

⚠ **同一时刻只能一个进程连 VTS**（参数注入独占）——跑本 sink 前先停掉
`tools/motion/` 下的 live_bridge / loop_vts 等脚本，否则对面会拒连。

env：`ZERO_VTS_SINK=true` 开启（默认关=零回归）；`ZERO_VTS_MCP_REPO` 指向 Zero_MCP
（默认取 Zero 仓的兄弟目录）；`ZERO_VTS_TOKEN_FILE` 授权 token 落盘位置。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
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

logger = logging.getLogger(__name__)

DEFAULT_SEGMENT_MS = 2000.0  # 段长：短些让情绪切换跟手（对面契约上限 10s）
DEFAULT_FPS = 20.0


class VtsSink:
    """把 `ExpressionFrame` 表现成皮套动作。实现 `ExpressionSink` 协议。

    未 `connect()` 或连接失败时全程 no-op（`emit` 静默返回）——表现端故障不扳倒对话。
    """

    def __init__(
        self,
        *,
        mcp_repo: Path | None = None,
        token_file: Path | None = None,
        segment_ms: float = DEFAULT_SEGMENT_MS,
        fps: float = DEFAULT_FPS,
        rng_seed: int = 20260811,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        self.mcp_repo = mcp_repo or Path(
            os.getenv("ZERO_VTS_MCP_REPO", str(repo_root.parent / "Zero_MCP"))
        )
        self.token_file = token_file or Path(
            os.getenv(
                "ZERO_VTS_TOKEN_FILE", str(repo_root / "data" / "steering" / "motion" / "vts_token")
            )
        )
        self.segment_ms = segment_ms
        self.fps = fps
        self.emotion: tuple[float, float] = (0.0, 0.0)
        self.regulated: tuple[float, float] | None = None
        self.phase = PhaseState(noise_seed=rng_seed, next_blink_ms=initial_blink_ms(rng_seed))
        self.proc: asyncio.subprocess.Process | None = None
        self.io_lock = asyncio.Lock()  # worker 是行分隔请求-响应，必须串行化
        self.loop_task: asyncio.Task[None] | None = None
        self.stop = asyncio.Event()

    async def connect(self) -> bool:
        """起渲染 worker 子进程、连 VTS、开动作循环；失败返回 False（不抛，可继续纯对话）。

        worker 见 `_vts_worker`：它在对面仓的 sys.path 下进程内直连 VTS——这条路已实测可靠，
        而经对方 MCP server 的 `vts_connect` 会卡死（同文件 docstring 记了对照实验）。
        """
        env = dict(os.environ)
        env.update(
            {
                "VTS_BEHAVIOR_ENABLED": "true",
                "VTS_TOKEN_FILE": str(self.token_file),
                "PYTHONPATH": str(self.mcp_repo),
                "PYTHONIOENCODING": "utf-8",
            }
        )
        worker_src = Path(__file__).with_name("_vts_worker.py")
        try:
            self.proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(worker_src),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self.mcp_repo),
            )
            reply = await asyncio.wait_for(self._rpc({"op": "connect"}), timeout=30.0)
        except Exception as exc:  # 连不上就退化成纯对话，这是表现层该有的姿态
            logger.warning("皮套连接失败，退化为纯对话：%s", exc)
            await self._kill_worker()
            return False
        except BaseException:
            # ⚠ CancelledError 是 BaseException 不是 Exception（code-reviewer 2026-08-11）：
            # 子进程已起、握手中途被外部取消时，若不在这里清理，句柄丢失而进程仍在——
            # 单进程 REPL 下靠父进程退出兜底，但本层文档明写供多 sink/长生命周期宿主复用，
            # 那种场景会堆积子进程。清理后原样抛出，不吞取消。
            await self._kill_worker()
            raise
        if not reply.get("ok"):
            logger.warning("皮套连接被拒，退化为纯对话：%s", reply.get("error"))
            await self._kill_worker()
            return False
        self.stop = asyncio.Event()
        self.loop_task = asyncio.create_task(self._drive())
        logger.info("皮套已连接：healthy=%s model=%s", reply.get("healthy"), reply.get("model_id"))
        return True

    async def _rpc(self, msg: dict[str, Any]) -> dict[str, Any]:
        """向 worker 发一条指令并读回执（行分隔 JSON）。"""
        proc = self.proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError("worker 未启动")
        async with self.io_lock:  # 串行化：回执按发送顺序回，多协程并发写会错配
            proc.stdin.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
            await proc.stdin.drain()
            line = await proc.stdout.readline()
        if not line:
            raise RuntimeError("worker 已退出")
        return dict(json.loads(line.decode("utf-8")))

    async def _kill_worker(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()

    async def emit(self, frame: ExpressionFrame) -> None:
        """接一帧表现内容：更新连续轨迹的驱动情绪 + 触发本轮的离散行为。"""
        self.emotion = frame.emotion
        self.regulated = frame.regulated
        if self.proc is None or not frame.reply:
            return
        try:
            for intent in self._intents(frame.reply):
                await self._rpc(
                    {
                        "op": "behavior",
                        "name": intent.name,
                        "intensity": intent.intensity,
                        "direction": intent.direction,
                    }
                )
        except Exception as exc:
            logger.warning("离散行为投递失败（对话不受影响）：%s", exc)

    async def aclose(self) -> None:
        """停循环、断连接（幂等）。"""
        self.stop.set()
        if self.loop_task is not None:
            self.loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.loop_task
            self.loop_task = None
        if self.proc is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._rpc({"op": "close"}), timeout=5.0)
            await self._kill_worker()

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
        if self.proc is None:
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
        await self._rpc(
            {
                "op": "animate",
                "keyframes": heads["voluntary"],
                "mode": "absolute",
                "append": True,
            }
        )


def build_vts_sink() -> VtsSink | None:
    """按 env 装配皮套 sink；`ZERO_VTS_SINK` 未开则返回 None（默认关=零回归）。"""
    raw = os.getenv("ZERO_VTS_SINK", "false").strip().lower()
    if raw not in ("1", "true", "yes", "on"):
        return None
    return VtsSink()
