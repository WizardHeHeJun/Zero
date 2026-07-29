"""zero-link MCP **server**：把 `ConversationSession` 包成三工具（边界适配层，三层之外）。

对齐 MCP 侧 client 契约（`D:\\Zero_MCP\\notes\\2026-07-15-zero-link-client-ready-to-zero.md`）：
- `zero.open_session(persona?, config?) -> {"session_id": str}`
- `zero.step(session_id, stim, external_priors?) -> <expression 子 dict>`
- `zero.close_session(session_id) -> {"ok": true}`
错误一律 `ToolError`（FastMCP 转成 `CallToolResult.isError=true`，消息进 `content[0].text`）。

红线守则：本模块**不反向依赖**内核内部（只经 `ConversationSession` 公开 `step`）、
**不进 affect 热路径**（映射是纯数据搬运）、secrets 走 env。`mcp` 走 optional-deps 不入 core。

会话门控默认（`ZERO_MCP_*` env 可调）：`workspace_enabled=True`（否则 external_priors 被整段
跳过，见 `affect_core` 的 `if state.workspace_enabled:` 分支——external_priors 是 client
契约核心、独立低精度流已经完整
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
from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from src.agents.expression import ChannelDecoder
from src.mcp_server.mapping import external_priors_from_payload, stimulus_from_payload
from src.mcp_server.registry import SessionRegistry
from src.orchestration.external_prior import ExternalPriorError
from src.orchestration.runner import ConversationSession, SessionConfig

logger = logging.getLogger(__name__)

# 议会解锁门：这两个字段**只受 ZERO_MCP_* env 治理**，client config overrides 中的同名键
# 被 _build_session_config 静默忽略（A5·A6·2026-07-21）——防「生产关·MCP 开」旁路。
_MCP_GOVERNANCE_GATED_FLAGS: frozenset[str] = frozenset(
    {
        "coping_potential_enabled",
        "text_coping_enabled",
        "fear_domain_enabled",
        # precision_commensurable（议会 2026-07-28 第四轮）：改的是**默认融合路径**的证据加权，
        # 比上面三个门更深——client 经 config 打开它等于在 MCP 面单方面换掉内核的精度标度。
        # 同治理模式：只受 ZERO_MCP_PRECISION_COMMENSURABLE env 管，overrides 静默忽略。
        "precision_commensurable",
        # gate_fusion（议会第三轮 D1）：它决定**数值后验怎么算**（硬门是否参与数值通路），
        # 比上面几个门更深；且 physio 排除是对 Zero_MCP 的跨仓承诺，不容 client 单边解除。
        "gate_fusion",
        "exclude_physio_fusion",
    }
)

# step 未知/过期 session_id 的机读错误前缀（zero-link T6）：MCP graceful_step 按此前缀判定 →
# 用同 id resume 重开重试（区别于其它 ToolError）。消息仍含 "session_id" 子串保既有断言。
_UNKNOWN_SESSION_MARKER = "unknown-session"


class ServerEnvError(RuntimeError):
    """**部署端** env 值不合法（不是 client 传参问题）。

    刻意**不继承** `ValueError`/`TypeError`：`open_session` 用
    `except (ValueError, TypeError) -> ToolError("config 不合法")` 兜 SessionConfig 的构造校验，
    若本类落进那一支，部署端把 `ZERO_EXTERNAL_PRIOR_PRECISION_CAP` 写成 `"0.8x"` 这种错，
    报出来会指向 **client 的 config** —— client 照着改 config 永远改不好。
    ⚠ stdio 传输下 client 进程环境**就是** server 进程环境（其 `_build_subprocess_env` 全量拷贝
    `os.environ`），所以这条归责错位真的会落到对方头上，不是理论风险。
    """


def _env_flag(name: str, default: bool) -> bool:
    """读布尔 env：真值集 → True，假值集 → False，**其余一律回落 `default`**。

    🛑 **旧实现只判真值集**（`raw.lower() in ("1","true","yes","on")`），未识别值一律 False。
    那对「默认 False」的旗标恰好等于 default，看不出问题；但用在**默认 True** 的旗标上
    （`workspace_enabled` / `gate_fusion` / `exclude_physio_fusion`）**失败方向就反了**：
    空串 / 带空格的 `"true "` / `"enabled"` 这类值会静默把门**打开**，
    而 `chat_driver` 侧同样取值判的是假值集、结论是门**关**——两侧语义直接冲突。
    后果实测：`ZERO_MCP_EXCLUDE_PHYSIO_FUSION=""` 时反号 physio 回到数值通路，
    arousal 后验被抬高 150%+，等于单边解除对 Zero_MCP 的 D7 跨仓承诺。
    （终审工作流抓出，两名独立验证者复现。）

    现改为**方向无关**：未识别值回落 `default`，并 `strip()` 掉首尾空白。
    对三个既有的「默认 False」旗标**逐字零回归**（未识别值 → default → False，与旧实现同值）；
    对「默认 True」的旗标则把失败方向从「打开未评审的新架构」改回「保持默认」。
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default  # 未识别（含空串）→ 回落默认，**不再一律 False**


def _env_number[T: (int, float)](name: str, default: str, caster: Callable[[str], T]) -> T:
    """读数值 env；坏值抛 `ServerEnvError`（指名 env），**不**抛 ValueError。

    与 `ZERO_MCP_HTTP_PORT` 的处理同口径。区别在于本函数用于 `_build_session_config`，
    而后者被 `open_session` 的 `except (ValueError, TypeError)` 包着——裸 `float()` 的
    ValueError 落进那一支就会被贴成「config 不合法」，把部署端的锅甩给 client。
    """
    raw = os.getenv(name, default)
    try:
        return caster(raw)
    except (ValueError, TypeError) as e:
        raise ServerEnvError(
            f"{name} 须为 {caster.__name__}，当前值={raw!r}；这是**部署端 env** 的问题，"
            "改 client 传的 config 无效"
        ) from e


def _build_session_config(overrides: dict[str, Any] | None) -> SessionConfig:
    """从 env 装配 MCP 边界的会话默认门控；`overrides` 覆写 SessionConfig 已声明字段。

    覆写只挑 SessionConfig 已有字段（防注入未知键），并经 SessionConfig 构造**重新校验**
    （如共情系数 L1 上界 `SessionConfig._check_empathy_l1`），坏值 fail-fast → 上层转 ToolError。

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
        # canonical_physiology：physiology 占位口径门控（议会 2026-07-23；默认关=零回归）。
        # 与 _maybe_expression_decoder 同读 ZERO_PHYSIOLOGY_CANONICAL_PLACEHOLDER，保证
        # state.canonical_physiology（经 SessionConfig）与 decoder 构造期固定值同源。
        # MCP 收窄前提（CS 席约束·design.md）：消费方须确认部署端已开此 env 才能依赖
        # canonical 键集（{hr,sc(μS),temperature_c}）；默认关=legacy 占位（含 pupil_mm）。
        "canonical_physiology": _env_flag("ZERO_PHYSIOLOGY_CANONICAL_PLACEHOLDER", False),
        # facs_extended：AU 扩展集合门控。**必须与 _maybe_expression_decoder 同读同一 env**——
        # 该函数按此 env 决定 load_facs_decoder(extended=...)（13 键 vs 5 键），而 expression 节点
        # 按 state.facs_extended 走。此前 base 漏了这一行：设 ZERO_FACS_EXTENDED=true 且给了
        # ZERO_FACS_MODEL_PATH 时，decoder 载入 13 键真模型、state 却取字段默认 False
        # → C2 residual 的 coping 分野（AU23/AU01/AU02/AU20）被静默跳过。同 canonical_physiology
        # 的同源契约（composite.py 的键集对齐要求）。回归锁：
        # tests/test_mcp_server.py::test_mcp_facs_extended_env_seeds_session_config
        # （撤掉本行即红，已实证）。
        "facs_extended": _env_flag("ZERO_FACS_EXTENDED", False),
        # 默认 False：与生产/chat 零回归一致（议会 2026-07-28 第四轮）——齐次化改的是每轮
        # 无条件执行的默认融合路径，MCP 面不得旁路生效；仅 ZERO_MCP_PRECISION_COMMENSURABLE
        # env 治理，client override 被 gated_flags 静默忽略。
        "precision_commensurable": _env_flag("ZERO_MCP_PRECISION_COMMENSURABLE", False),
        # ⚠ 默认 **True**（=门关=与生产/chat 零回归一致）。方向与上面几个相反：
        # 漏了这一行会使 gate_fusion 在 MCP 路径**永久 True**——新架构永远开不出来
        # （不是永远开着）。写双向用例时别按其它旗标的方向照抄。
        "gate_fusion": _env_flag("ZERO_MCP_IGNITION_GATE_FUSION", True),
        # physio 排除默认 True（D7 跨仓承诺·由我方单边可控）。
        "exclude_physio_fusion": _env_flag("ZERO_MCP_EXCLUDE_PHYSIO_FUSION", True),
        # ⚠ 这两个走 _env_number：裸 float()/int() 抛的 ValueError 会被 open_session 的
        # `except (ValueError, TypeError)` 贴成「config 不合法」——**部署端 env 写错却指向
        # client 传参**（见 ServerEnvError 的说明）。
        "external_prior_precision_cap": _env_number(
            "ZERO_EXTERNAL_PRIOR_PRECISION_CAP", "0.8", float
        ),
        "max_external_streams": _env_number("ZERO_MAX_EXTERNAL_STREAMS", "5", int),
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
    """env 门控注入真通道解码器：`ZERO_FACS_MODEL_PATH` / `ZERO_PROSODY_MODEL_PATH` /
    `ZERO_PHYSIOLOGY_MODEL_PATH` 皆未设 → None（占位路径，无需 torch）。三通道**独立门控**：
    可单独设任一路（如只设 physiology → facs/prosody 仍占位、physiology 出真 WESAD
    {hr,sc(μS),temperature_c}），互不影响；只设 FACS 与改前逐字一致（零回归）。

    仅设了对应权重路径才**延迟 import** models 层公开 `load_facs_decoder` / `load_prosody_decoder`
    （该链引 torch 侧）；默认（都未设）路径完全不 import → server 保持轻依赖（仅 langgraph+
    pydantic+mcp）。**只依赖 models 公开类**（非 chat_driver 私有工厂），边界层不耦合编排层内部；
    构造口径与 `chat_driver._build_expression_decoder` 刻意保持一致（真 13-AU 权重 + 真韵律可跑进
    MCP 输出）。系数方向议会定、幅度工程可动——env 缺省=构造默认（1.5/1.2/1.0）；文件不可读
    fail-fast（不静默降级），形状不配对的 RuntimeError 原样穿透（含 size mismatch）。
    """
    facs_path = os.getenv("ZERO_FACS_MODEL_PATH")
    prosody_path = os.getenv("ZERO_PROSODY_MODEL_PATH")
    physiology_path = os.getenv("ZERO_PHYSIOLOGY_MODEL_PATH")
    if not facs_path and not prosody_path and not physiology_path:
        return None
    from src.agents.models.composite import CompositeChannelDecoder

    facs_extended = _env_flag("ZERO_FACS_EXTENDED", False)
    # canonical_physiology：physiology 占位口径门控（议会 2026-07-23）。
    # 与 facs_extended 同双读同一 env 模式（构造期固定·非 per-turn）。
    # MCP 收窄前提：此 flag 须与 state.canonical_physiology 同源（同一 env）——
    # MCP 路径下 state 由 SessionConfig.to_state_flags() 贯通，canonical_physiology
    # 在 _build_session_config 中注入（见下方），与 decoder 构造期固定同源。
    canonical_physiology = _env_flag("ZERO_PHYSIOLOGY_CANONICAL_PLACEHOLDER", False)
    kwargs: dict[str, Any] = {
        "facs_extended": facs_extended,
        "canonical_physiology": canonical_physiology,
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
    if physiology_path:
        from src.agents.models.physiology_decoder import load_physiology_decoder

        try:
            kwargs["physiology_model"] = load_physiology_decoder(physiology_path)
        except OSError as e:  # 同 FACS：文件不可读 fail-fast；形状不配对 RuntimeError 原样穿透。
            raise RuntimeError(
                f"ZERO_PHYSIOLOGY_MODEL_PATH={physiology_path!r} 指向的权重文件不可读，请检查配置"
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
    # port 非法值 fail-fast 指向 env 名（与 _build_session_config 的 _env_number 同口径）。
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
        except ServerEnvError as e:
            # 部署端 env 坏值：必须与下面的 client-config 分支分开报，否则 client 照着改
            # config 永远改不好（同 zero.step 里 ExternalPriorError / ValueError 的分治理由）。
            raise ToolError(f"服务端 env 配置不合法（**非** client config 问题）：{e}") from e
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
            except ExternalPriorError as e:
                # expand_external_priors 的 M3/M6/M7 fail-fast（精度>0/≤cap、流数≤max、
                # 形状良构、μ∈[-1,1]）——**确实**指向 client 传参，改传参就能好。
                raise ToolError(f"external_priors 校验失败（指向 MCP 传参）：{e}") from e
            except ValueError as e:
                # 内核其它位置抛的 ValueError（如 HPC coupling 越界、未来的配置互斥 fail-fast）。
                # ⚠ 这里**必须**与上一分支分开报（议会 2026-07-29 第五轮校验 §四-5）：
                # 原先一个 except 包住整个 step（全图执行），把内核任何异常都贴成
                # 「external_priors 校验失败（指向 MCP 传参）」→ 误导性甩锅。client 照着改
                # 传参永远改不好，而活跃会话的 config 不可变（见 open_session 的 config 语义）
                # → 无法自救，表现为 open 成功、**每 step 崩**。
                logger.exception("zero.step 内核执行失败 sid=%s", session_id)
                raise ToolError(
                    f"内核执行失败（**非** external_priors 传参问题，改传参无效）：{e}；"
                    "多为会话级配置组合不兼容，须以新配置重开会话"
                ) from e
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
