"""渲染端 MCP 连接的共享传输层（自 `vts.py` 抽出，2026-08-14 speech-output T2）。

**为什么单独成类**：⚠ 同一时刻只能一个进程连 VTS（参数注入独占）。语音口型
（`speech.py`）与皮套动作（`vts.py`）都要往渲染端投参数，若各自 spawn 一条 MCP
子进程连接，第二条会被 VTS 拒连。故把「spawn 子进程 + 握手 + 超时分档 + 收尾」
抽成本类，多个 sink 注入**同一实例**共享一条连接；`connect()` 幂等（已连直接复用），
`aclose()` 幂等（重复关不报错）。

连接语义、超时分档与全部前科注释**原样自 `vts.py` 搬入**，行为零变化。
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

logger = logging.getLogger(__name__)

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


def text_of(result: Any) -> str:
    """取 MCP 工具返回体的文本载荷（对面全部走 unstructured `content[0].text`）。

    包内共享工具（vts/speech 都消费），公开命名——本仓约定下划线仅私有。
    """
    content = getattr(result, "content", None)
    if not content:
        return ""
    return str(getattr(content[0], "text", "") or "")


class VtsTransport:
    """持一条到渲染端（Zero_MCP）的 MCP 会话，供多个表现 sink 共享调用。

    未连接/连接失败时 `session` 保持 None——各 sink 据此短路自己的投递。
    本类只管**连接与调用**，不含任何动作/口型决策。
    """

    def __init__(self, *, mcp_repo: Path | None = None, token_file: Path | None = None) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        self.mcp_repo = mcp_repo or Path(
            os.getenv("ZERO_VTS_MCP_REPO", str(repo_root.parent / "Zero_MCP"))
        )
        self.token_file = token_file or Path(
            os.getenv(
                "ZERO_VTS_TOKEN_FILE", str(repo_root / "data" / "steering" / "motion" / "vts_token")
            )
        )
        self.session: Any = None  # mcp.ClientSession；None = 未连接
        self.stack: contextlib.AsyncExitStack | None = None

    def connect_timeout(self) -> float:
        """按「这次要不要等人」选超时档（判据见模块常量处）。

        判据取在 **token 文件是否已落盘** 上，而不是一个固定余量：token 在 = 对面不弹窗、
        全程机器路径；token 不在 = 必然弹窗、必须等人。写成方法而非内联三元，是为了它可被
        单测直接钉住——这一档选错的代价不是慢，是**在 VTS 里埋下一个挂起授权窗**。
        """
        return CONNECT_TIMEOUT_S if self.token_file.exists() else CONNECT_TIMEOUT_FIRST_AUTH_S

    async def connect(self) -> bool:
        """spawn 渲染端 MCP server 并连 VTS；失败返回 False（不抛，可继续纯对话）。

        **幂等**：已连接时直接返回 True——多 sink 共享同一实例，入口对每个 sink 各
        调一次 `connect()`，第二次起为 no-op 复用。

        ⚠ **`vts_connect` 不返回时的两种成因，判别点在 stderr**（配套项目 2026-08-11）：
        有「等待用户在 VTube Studio 弹窗中允许插件」那句 = **在等人**（正常，见超时分档）；
        没有那句 = 踩了 numpy import 死锁（对面已修，见 `vts.py` docstring 那段结案历史）。
        另注：`Get-NetTCPConnection` 查 8001 无 ESTABLISHED **查不出**挂起授权窗——
        那是 VTS 进程里的 UI 状态，连接确实已断。两条都要进排查 checklist。

        ⚠ 撞上挂起授权窗时（`errorID 51`），对面文案自 2026-08-11 起**自解释**（含「去 VTS
        点掉那个窗再重试」+ 那条查不出的提示），我方 `reply.isError` 分支原样透传即可。
        将来若要**自动**判别这一支，判据取对面结构化的 `error_id`，**别正则抠人读文案**
        （中英混排、会随文案改动静默失效——同我方「判据不取在名字/文本上」那条纪律）。
        """
        if self.session is not None:
            return True
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            logger.warning("未安装 mcp，渲染端表现不可用（对话照常）")
            return False
        env = dict(os.environ)
        env.update(
            {
                "VTS_BEHAVIOR_ENABLED": "true",
                # ⚠ server 是**我方 spawn 的**——对方部署侧开的 flag 对这条链路不生效，
                # 环境由这里注入（2026-08-14 首次真机联调实测：漏传 ⇒ speech_play 吃
                # [vtsb:speech_disabled]）。speech 工具开着但无人调时惰性无副作用，故恒开。
                "VTS_SPEECH_ENABLED": "true",
                "VTS_TOKEN_FILE": str(self.token_file),
                "PYTHONPATH": str(self.mcp_repo),
                "PYTHONIOENCODING": "utf-8",
            }
        )
        # ⚠ 在 try 之前算：`except TimeoutError` 分支要用它，而 try 内早期失败也可能抛
        # TimeoutError（stdio spawn 阶段），那时若 timeout 还未绑定就是 NameError 盖住真异常。
        timeout = self.connect_timeout()
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
            logger.warning("皮套连接被拒，退化为纯对话：%s", text_of(reply))
            await self._close_stack(stack)
            return False
        body = json.loads(text_of(reply) or "{}")
        self.stack = stack
        self.session = session
        logger.info("皮套已连接：healthy=%s model=%s", body.get("healthy"), body.get("model_id"))
        return True

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """经共享会话调渲染端工具。未连接时抛 RuntimeError——调用方应先按 `session` 短路。"""
        if self.session is None:
            raise RuntimeError("渲染端未连接（session=None），调用方应先短路")
        return await self.session.call_tool(name, args)

    async def aclose(self) -> None:
        """断连接、收子进程（幂等：未连接/重复关都安全）。"""
        if self.stack is not None:
            await self._close_stack(self.stack)

    async def _close_stack(self, stack: contextlib.AsyncExitStack) -> None:
        """关掉 MCP 会话与 server 子进程（幂等；关不掉也不抛）。"""
        with contextlib.suppress(Exception):
            await stack.aclose()
        self.stack = None
        self.session = None
