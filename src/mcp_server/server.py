"""zero-link MCP **server**：把 `ConversationSession` 包成三工具（边界适配层，三层之外）。

对齐 MCP 侧 client 契约（`D:\\Zero_MCP\\notes\\2026-07-15-zero-link-client-ready-to-zero.md`）：
- `zero.open_session(persona?, config?) -> {"session_id": str}`
- `zero.step(session_id, stim, external_priors?) -> <expression 子 dict>`
- `zero.close_session(session_id) -> {"ok": true}`
错误一律 `ToolError`（FastMCP 转成 `CallToolResult.isError=true`，消息进 `content[0].text`）。

红线守则：本模块**不反向依赖**内核内部（只经 `ConversationSession` 公开 `step`）、
**不进 affect 热路径**（映射是纯数据搬运）、secrets 走 env。`mcp` 走 optional-deps 不入 core。

会话门控默认（`ZERO_MCP_*` env 可调）：`workspace_enabled=True`（否则 external_priors 被整段
跳过，`affect_core.py:44,100-107`——external_priors 是 client 契约核心、独立低精度流已经完整
PRP + code-reviewer 批准，故默认开）；`coping_potential_enabled=False`（**与生产/chat 路径一致·
零回归**——原默认 True 被议会四轮 2026-07-18 判为「生产关·MCP 开」治理旁路：anger 方向先验尚未
经议会解锁，不得经 MCP 面静默生效。MCP 边界是否统一/anger 侧是否受同一弃权门约束=议会 B1 悬而
未决，见 notes/2026-07-18-anger-delta-validation-council.md；解锁前维持最保守默认关）；
`text_coping_enabled=False`（**与生产/chat 路径一致·零回归**——text_coping 需议会解锁，
MCP 面不得旁路生效，仅 `ZERO_MCP_TEXT_COPING_ENABLED` env 治理，见 A5·2026-07-21）。

**议会解锁门治理原则（A5·A6·2026-07-21）**：`text_coping_enabled`/`coping_potential_enabled`
是议会门控字段，**只受 `ZERO_MCP_*` env 治理，client 经 config overrides 传入的同名字段被
静默忽略**（`_MCP_GOVERNANCE_GATED_FLAGS` 过滤）——防「生产关·MCP 开」旁路。
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from src.agents.expression import ChannelDecoder
from src.mcp_server.mapping import external_priors_from_payload, stimulus_from_payload
from src.mcp_server.registry import SessionRegistry
from src.orchestration.runner import ConversationSession, SessionConfig

logger = logging.getLogger(__name__)

# 议会解锁门：这两个字段**只受 ZERO_MCP_* env 治理**，client config overrides 中的同名键
# 被 _build_session_config 静默忽略（A5·A6·2026-07-21）——防「生产关·MCP 开」旁路。
_MCP_GOVERNANCE_GATED_FLAGS: frozenset[str] = frozenset(
    {"coping_potential_enabled", "text_coping_enabled", "fear_domain_enabled"}
)

# step 未知/过期 session_id 的机读错误前缀（zero-link T6）：MCP graceful_step 按此前缀判定 →
# 用同 id resume 重开重试（区别于其它 ToolError）。消息仍含 "session_id" 子串保既有断言。
_UNKNOWN_SESSION_MARKER = "unknown-session"


def _env_flag(name: str, default: bool) -> bool:
    """读布尔 env；未设 → default。真值集与 chat_driver 一致（1/true/yes/on）。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _build_session_config(overrides: dict[str, Any] | None) -> SessionConfig:
    """从 env 装配 MCP 边界的会话默认门控；`overrides` 覆写 SessionConfig 已声明字段。

    覆写只挑 SessionConfig 已有字段（防注入未知键），并经 SessionConfig 构造**重新校验**
    （如共情系数 L1 上界 `runner.py:131-140`），坏值 fail-fast → 上层转 ToolError。

    **议会解锁门（A5·A6·2026-07-21）**：`_MCP_GOVERNANCE_GATED_FLAGS` 中的字段
    （`text_coping_enabled`/`coping_potential_enabled`）**不受 overrides 覆写**，
    只由 `ZERO_MCP_TEXT_COPING_ENABLED`/`ZERO_MCP_COPING_ENABLED` env 治理——防 client
    经 config 旁路生产门控（「生产关·MCP 开」治理漏洞）。非门控字段（如 `contagion_alpha`）
    仍可正常 override。
    """
    base: dict[str, Any] = {
        "workspace_enabled": _env_flag("ZERO_MCP_WORKSPACE_ENABLED", True),
        # 默认 False：与生产/chat 零回归一致（议会四轮 2026-07-18·B1 治理旁路整改）——
        # anger 方向先验未经议会解锁，MCP 面不得旁路生效；解锁后按议会裁定更新。
        "coping_potential_enabled": _env_flag("ZERO_MCP_COPING_ENABLED", False),
        # 默认 False：与生产/chat 零回归一致（A5·2026-07-21）——text_coping 需议会解锁，
        # 仅 ZERO_MCP_TEXT_COPING_ENABLED env 治理；client override 被 gated_flags 静默忽略。
        "text_coping_enabled": _env_flag("ZERO_MCP_TEXT_COPING_ENABLED", False),
        # 默认 False：WARN-3 fear 专属门（B1 BLOCK 前置·议会 2026-07-21·A1）——任何路径不产
        # fear 域激活；仅 ZERO_MCP_FEAR_DOMAIN_ENABLED env 治理，client override 被 gated_flags
        # 静默忽略（与 text_coping_enabled 同一治理模式）。
        "fear_domain_enabled": _env_flag("ZERO_MCP_FEAR_DOMAIN_ENABLED", False),
        "external_prior_precision_cap": float(
            os.getenv("ZERO_EXTERNAL_PRIOR_PRECISION_CAP", "0.8")
        ),
        "max_external_streams": int(os.getenv("ZERO_MAX_EXTERNAL_STREAMS", "5")),
    }
    if overrides:
        base.update(
            {
                k: v
                for k, v in overrides.items()
                if k in SessionConfig.model_fields and k not in _MCP_GOVERNANCE_GATED_FLAGS
            }
        )
    return SessionConfig(**base)


def _maybe_expression_decoder() -> ChannelDecoder | None:
    """env 门控注入真通道解码器：`ZERO_FACS_MODEL_PATH` / `ZERO_PROSODY_MODEL_PATH` 皆未设 → None
    （占位路径，无需 torch）。两通道**独立门控**：可单独设 prosody（facs 仍占位·prosody_scale
    翻 normalized），反之亦然；只设 FACS 与改前逐字一致（零回归）。

    仅设了对应权重路径才**延迟 import** models 层公开 `load_facs_decoder` / `load_prosody_decoder`
    （该链引 torch 侧）；默认（都未设）路径完全不 import → server 保持轻依赖（仅 langgraph+
    pydantic+mcp）。**只依赖 models 公开类**（非 chat_driver 私有工厂），边界层不耦合编排层内部；
    构造口径与 `chat_driver._build_expression_decoder` 刻意保持一致（真 13-AU 权重 + 真韵律可跑进
    MCP 输出）。系数方向议会定、幅度工程可动——env 缺省=构造默认（1.5/1.2/1.0）；文件不可读
    fail-fast（不静默降级），形状不配对的 RuntimeError 原样穿透（含 size mismatch）。
    """
    facs_path = os.getenv("ZERO_FACS_MODEL_PATH")
    prosody_path = os.getenv("ZERO_PROSODY_MODEL_PATH")
    if not facs_path and not prosody_path:
        return None
    from src.agents.models.composite import CompositeChannelDecoder

    facs_extended = _env_flag("ZERO_FACS_EXTENDED", False)
    kwargs: dict[str, Any] = {
        "facs_extended": facs_extended,
        "k_arousal": float(os.getenv("ZERO_FACS_K_AROUSAL", "1.5")),
        "k_coping": float(os.getenv("ZERO_FACS_K_COPING", "1.2")),
        "residual_alpha": float(os.getenv("ZERO_FACS_RESIDUAL_ALPHA", "1.0")),
    }
    if facs_path:
        from src.agents.models.facs_decoder import load_facs_decoder

        try:
            kwargs["facs_model"] = load_facs_decoder(facs_path, extended=facs_extended)
        except (
            OSError
        ) as e:  # 文件不可读 fail-fast；形状不配对 RuntimeError 原样穿透（size mismatch）。
            # ⚠ build_server ← __main__.main：此 RuntimeError 意味进程不可启动的配置 fail-fast，
            # 非工具级错误——不应在 build_server 处被 ToolError 吞（启动硬失败胜于启动后静默出错）。
            raise RuntimeError(
                f"ZERO_FACS_MODEL_PATH={facs_path!r} 指向的权重文件不可读，请检查配置"
            ) from e
    if prosody_path:
        from src.agents.models.prosody_decoder import load_prosody_decoder

        try:
            kwargs["prosody_model"] = load_prosody_decoder(prosody_path)
        except OSError as e:  # 同 FACS：文件不可读 fail-fast；形状不配对 RuntimeError 原样穿透。
            raise RuntimeError(
                f"ZERO_PROSODY_MODEL_PATH={prosody_path!r} 指向的权重文件不可读，请检查配置"
            ) from e
    return CompositeChannelDecoder(**kwargs)


def build_server(registry: SessionRegistry | None = None) -> FastMCP:
    """装配并返回挂好三工具的 FastMCP server（传输无关：入口再选 stdio/http）。

    `registry` 可注入（测试用）；缺省新建。表情解码器在此按 env 构造一次、注入每个会话
    （无状态、可共享）。工具全部 `structured_output=False`：只发 unstructured `content[0].text`
    （JSON），与 client 读 `content[0].text` 的口径完全一致（不逼 client 改读 structuredContent）。
    """
    registry = registry if registry is not None else SessionRegistry()
    decoder = _maybe_expression_decoder()
    # HTTP 传输的监听配置（stdio 下无害·仅 streamable-http 生效）：host/port/path 走 env，
    # 便于给 MCP 侧稳定 endpoint（默认 127.0.0.1:8000/mcp，与 FastMCP 默认一致）。
    # port 非法值 fail-fast 指向 env 名（与 _build_session_config 的 float/int 处理风格一致）。
    try:
        http_port = int(os.getenv("ZERO_MCP_HTTP_PORT", "8000"))
    except ValueError as e:
        raise ValueError(
            f"ZERO_MCP_HTTP_PORT 须为整数，当前值={os.getenv('ZERO_MCP_HTTP_PORT')!r}"
        ) from e
    mcp: FastMCP = FastMCP(
        "zero",
        host=os.getenv("ZERO_MCP_HTTP_HOST", "127.0.0.1"),
        port=http_port,
        streamable_http_path=os.getenv("ZERO_MCP_HTTP_PATH", "/mcp"),
    )

    @mcp.tool(
        name="zero.open_session",
        description=(
            "建/重开一个 Zero 情感引擎会话（建图/checkpointer 一次），返回 {session_id}。"
            "传 session_id 则按旧 id 重开（跨 server 重启续会话·须持久后端）；不传则新铸。"
        ),
        structured_output=False,
    )
    async def open_session(
        persona: str | None = None,
        config: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> dict[str, str]:
        """建会话；传 `session_id` 走 resume（zero-link T6）。

        `config` 经唯一治理入口 `_build_session_config`（议会 env-only 门控 resume 亦保持）。
        `session_id` 不传 → 新铸 uuid4；传了 → 用作 thread_id 重开：已在 registry 活跃则幂等返回
        （不重建/不重开连接/不覆盖运行态），否则新建绑该 id。运行态是否真续取决于持久后端
        （`ZERO_CHECKPOINT_BACKEND=sqlite` 才跨重启恢复；memory 后端重开=全新会话）。
        SessionConfig 不进 checkpoint，resume 须再供同一 config。`persona` 预留入参（暂不生效）。
        """
        try:
            cfg = _build_session_config(config)
        except (ValueError, TypeError) as e:
            raise ToolError(f"config 不合法：{e}") from e
        if session_id is not None:
            if not isinstance(session_id, str) or not session_id.strip():
                raise ToolError(f"session_id 须为非空字符串，实际为 {session_id!r}")
            if await registry.get(session_id) is not None:
                # 同进程内仍活跃：幂等返回（不重建·不重开 aiosqlite 连接·不覆盖在飞运行态）。
                # 会话门控构造时固定、贯穿整会话，故此处不重应用 cfg：client 对活跃会话传的新
                # config 静默不生效（SessionConfig 不可变；跨重启 resume 重建时才用新 cfg）。
                logger.info("zero.open_session resume(active) sid=%s", session_id)
                return {"session_id": session_id}
            # resume：新建绑同 thread_id，持久后端 ainvoke 时自动从 checkpoint 恢复。
            sid = session_id
        else:
            sid = uuid.uuid4().hex
        # thread_id=user_id=sid：会话态经 checkpointer 按 thread_id 跨轮持久 + 记忆作用域天然按
        # session 隔离（防多会话共享 default-user 记忆串味，同 chat_driver 口径）。
        session = ConversationSession(
            thread_id=sid,
            user_id=sid,
            config=cfg,
            expression_decoder=decoder,
        )
        await registry.open(sid, session)
        logger.info(
            "zero.open_session sid=%s persona=%s resume=%s active=%d",
            sid,
            persona,
            session_id is not None,
            await registry.count(),
        )
        return {"session_id": sid}

    @mcp.tool(
        name="zero.step",
        description=(
            "喂一条刺激 {valence, arousal, coping_potential?} + 可选 external_priors，"
            "推进一轮，返回 expression 子 dict（valence_arousal/spontaneous/voluntary/…）。"
        ),
        structured_output=False,
    )
    async def step(
        session_id: str,
        stim: dict[str, Any],
        external_priors: list[Any] | None = None,
    ) -> dict[str, Any]:
        session, lock = await registry.acquire(session_id)
        if session is None or lock is None:
            raise ToolError(
                f"{_UNKNOWN_SESSION_MARKER}: 未知 session_id={session_id!r}；"
                "请先调 zero.open_session（可用同 id resume 续会话）"
            )
        try:
            stimulus = stimulus_from_payload(stim)
            priors = external_priors_from_payload(external_priors)
        except (ValueError, TypeError) as e:
            raise ToolError(f"stim/external_priors 载荷不合法：{e}") from e
        # 同会话串行化：LangGraph checkpointer 读-改-写非原子，并发 ainvoke(同 thread_id) 会竞态
        # （http 传输下 client 可能并发）；stdio 顺序则此锁无争用、零成本。
        async with lock:
            try:
                step_out = await session.step(stimulus, state_overrides={"external_priors": priors})
            except ValueError as e:
                # expand_external_priors 的 M3/M6 fail-fast（精度>0/≤cap、流数≤max、形状良构）。
                raise ToolError(f"external_priors 校验失败（指向 MCP 传参）：{e}") from e
        expression = step_out.get("expression") or {}
        return expression

    @mcp.tool(
        name="zero.close_session",
        description="释放一个会话；未知 id 幂等。返回 {'ok': true}（client 忽略返回值）。",
        structured_output=False,
    )
    async def close_session(session_id: str) -> dict[str, bool]:
        session, lock = await registry.acquire(session_id)
        if session is None or lock is None:
            return {"ok": True}  # 幂等：未知/已关 id 不报错（client 忽略返回值）
        # lock=每会话 step 锁；registry.close() 持 registry 表级锁（两把不同 asyncio.Lock·无死锁）。
        # 持锁再关串行化在飞 step，避免 aclose 关连接与 ainvoke 竞态（http 并发）。
        async with lock:
            await registry.close(session_id)  # 表内移除 → 后续 step 同 id 回 unknown-session
            await session.aclose()  # 关运行态 aiosqlite 连接（sqlite）/InMemory no-op·幂等
        logger.info("zero.close_session sid=%s active=%d", session_id, await registry.count())
        return {"ok": True}

    return mcp
