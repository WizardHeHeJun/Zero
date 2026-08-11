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

# ── vts_connect 的超时预算：按「等人」还是「等机器」拆（配套项目 2026-08-11 实测）──
# 对面实测：`vts_connect` 墙钟 26.91s，其中**几乎全部**是等人点 VTS 的授权弹窗——
# token 落盘之后到「行为层已连接」只有 **77ms**。⇒ 单一常量会把人的反应时间算进机器预算。
# 🛑 而超时太短是**自我延续**的：客户端取消后，VTS **不会**收掉那个授权弹窗，此后每次
# 连接都在 `AuthenticationTokenRequest` 上被 errorID 51 拒（"authentication is currently
# ongoing"），直到有人手动点掉 ⇒ 越短越容易给下一次尝试埋一个 51，观测上像"问题变严重了"。
# 故按 token 文件是否已落盘分档：首次授权留够人点击，之后走机器档。
CONNECT_TIMEOUT_S = 15.0  # 已授权过：纯机器路径（对面实测 0.0x s 级），留两个数量级余量
CONNECT_TIMEOUT_FIRST_AUTH_S = 180.0  # 首次授权：含人去 VTS 点「允许插件」的时间
# 🛑 人档按**纯人因**取值，不必给对面留余量（对面自查交底，2026-08-11）：其
# `AuthenticationTokenRequest` 是全仓**唯一**显式传 `timeout=None` 的请求（就是它在等人点弹窗），
# 其余请求走它自己的 10s，且 `vts_connect` 工具体不套超时 ⇒ **我方这个数是唯一的那把闸**，
# 不存在两侧超时打架、也不会被对面抢先超时。调它的判据只有一条：**人多久会注意到那个弹窗**。


def _text_of(result: Any) -> str:
    """取 MCP 工具返回体的文本载荷（对面全部走 unstructured `content[0].text`）。"""
    content = getattr(result, "content", None)
    if not content:
        return ""
    return str(getattr(content[0], "text", "") or "")


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
        self.session: Any = None  # mcp.ClientSession；None = 未连接
        self.stack: contextlib.AsyncExitStack | None = None
        self.loop_task: asyncio.Task[None] | None = None
        self.stop = asyncio.Event()

    def _connect_timeout(self) -> float:
        """按「这次要不要等人」选超时档（判据见模块常量处）。

        判据取在 **token 文件是否已落盘** 上，而不是一个固定余量：token 在 = 对面不弹窗、
        全程机器路径；token 不在 = 必然弹窗、必须等人。写成方法而非内联三元，是为了它可被
        单测直接钉住——这一档选错的代价不是慢，是**在 VTS 里埋下一个挂起授权窗**。
        """
        return CONNECT_TIMEOUT_S if self.token_file.exists() else CONNECT_TIMEOUT_FIRST_AUTH_S

    async def connect(self) -> bool:
        """spawn 渲染端 MCP server、连 VTS、开动作循环；失败返回 False（不抛，可继续纯对话）。

        ⚠ **`vts_connect` 不返回时的两种成因，判别点在 stderr**（配套项目 2026-08-11）：
        有「等待用户在 VTube Studio 弹窗中允许插件」那句 = **在等人**（正常，见超时分档）；
        没有那句 = 踩了 numpy import 死锁（对面已修，见模块 docstring 那段结案历史）。
        另注：`Get-NetTCPConnection` 查 8001 无 ESTABLISHED **查不出**挂起授权窗——
        那是 VTS 进程里的 UI 状态，连接确实已断。两条都要进排查 checklist。

        ⚠ 撞上挂起授权窗时（`errorID 51`），对面文案自 2026-08-11 起**自解释**（含「去 VTS
        点掉那个窗再重试」+ 那条查不出的提示），我方 `reply.isError` 分支原样透传即可。
        将来若要**自动**判别这一支，判据取对面结构化的 `error_id`，**别正则抠人读文案**
        （中英混排、会随文案改动静默失效——同我方「判据不取在名字/文本上」那条纪律）。
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            logger.warning("未安装 mcp，皮套表现不可用（对话照常）")
            return False
        env = dict(os.environ)
        env.update(
            {
                "VTS_BEHAVIOR_ENABLED": "true",
                "VTS_TOKEN_FILE": str(self.token_file),
                "PYTHONPATH": str(self.mcp_repo),
                "PYTHONIOENCODING": "utf-8",
            }
        )
        # ⚠ 在 try 之前算：`except TimeoutError` 分支要用它，而 try 内早期失败也可能抛
        # TimeoutError（stdio spawn 阶段），那时若 timeout 还未绑定就是 NameError 盖住真异常。
        timeout = self._connect_timeout()
        stack = contextlib.AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=sys.executable,
                        args=["-m", "src.mcp.vts_behavior_mcp_server"],
                        env=env,
                        cwd=str(self.mcp_repo),
                    )
                )
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            if timeout == CONNECT_TIMEOUT_FIRST_AUTH_S:
                logger.info(
                    "VTS 尚未授权过（%s 不存在）：请在 VTube Studio 的弹窗里点「允许插件」，"
                    "最多等 %.0fs；授权一次后 token 落盘，此后不再弹窗",
                    self.token_file,
                    timeout,
                )
            reply = await asyncio.wait_for(session.call_tool("vts_connect", {}), timeout=timeout)
        except TimeoutError as exc:
            # 单列一支不是为了措辞好看：超时**留下的东西**与其它失败不同（见模块常量处）。
            logger.warning(
                "皮套连接超时（%.0fs），退化为纯对话；⚠ 若此刻 VTS 里有授权弹窗，它**不会**随本次"
                "取消而关闭 —— 下次连接会被 errorID 51 拒到有人手动点掉它：%s",
                timeout,
                exc,
            )
            await self._close_stack(stack)
            return False
        except Exception as exc:  # 连不上就退化成纯对话，这是表现层该有的姿态
            logger.warning("皮套连接失败，退化为纯对话：%s", exc)
            await self._close_stack(stack)
            return False
        except BaseException:
            # ⚠ CancelledError 是 BaseException 不是 Exception（code-reviewer 2026-08-11）：
            # server 子进程已起、握手中途被外部取消时，不在这里收就会漏掉它。
            # 单进程 REPL 下靠父进程退出兜底，但本层供多 sink/长生命周期宿主复用，
            # 那种场景会堆积子进程。清理后原样抛出，不吞取消。
            await self._close_stack(stack)
            raise
        if reply.isError:
            logger.warning("皮套连接被拒，退化为纯对话：%s", _text_of(reply))
            await self._close_stack(stack)
            return False
        body = json.loads(_text_of(reply) or "{}")
        self.stack = stack
        self.session = session
        self.stop = asyncio.Event()
        self.loop_task = asyncio.create_task(self._drive())
        logger.info("皮套已连接：healthy=%s model=%s", body.get("healthy"), body.get("model_id"))
        return True

    async def _close_stack(self, stack: contextlib.AsyncExitStack) -> None:
        """关掉 MCP 会话与 server 子进程（幂等；关不掉也不抛）。"""
        with contextlib.suppress(Exception):
            await stack.aclose()
        self.stack = None
        self.session = None

    async def emit(self, frame: ExpressionFrame) -> None:
        """接一帧表现内容：更新连续轨迹的驱动情绪 + 触发本轮的离散行为。"""
        self.emotion = frame.emotion
        self.regulated = frame.regulated
        if self.session is None or not frame.reply:
            return
        try:
            for intent in self._intents(frame.reply):
                args: dict[str, Any] = {"name": intent.name, "intensity": intent.intensity}
                if intent.direction:
                    args["direction"] = intent.direction
                await self.session.call_tool("behavior_trigger", args)
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
        if self.stack is not None:
            await self._close_stack(self.stack)

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
        if self.session is None:
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
        await self.session.call_tool(
            "params_animate",
            {"keyframes": heads["voluntary"], "mode": "absolute", "append": True},
        )


def build_vts_sink() -> VtsSink | None:
    """按 env 装配皮套 sink；`ZERO_VTS_SINK` 未开则返回 None（默认关=零回归）。"""
    raw = os.getenv("ZERO_VTS_SINK", "false").strip().lower()
    if raw not in ("1", "true", "yes", "on"):
        return None
    return VtsSink()
