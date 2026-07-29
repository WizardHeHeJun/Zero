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

import asyncio
import logging
import math
import os
import uuid
from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from src.agents.expression import ChannelDecoder
from src.mcp_server.mapping import external_priors_from_payload, stimulus_from_payload
from src.mcp_server.registry import SessionRegistry
from src.orchestration.external_prior import (
    EXTERNAL_PRIOR_SCHEMA_VERSION,
    ExternalPriorError,
)
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
        # ignition_beta（2026-07-29 跨仓）：非 None 即走**软门**分支，而软门下全部流（含 physio）
        # 一律进 fuse_terms、无阈值筛除、也不施 D7 排除 ⇒ client 经 config 传它即可让 physio
        # 进入数值后验，等于单边解除对 Zero_MCP 的 D7 承诺。深度等同 gate_fusion，故入白名单。
        # ⚠ 配套项目已确认其生产码从不传该字段，但那是**当前调用点的事实、不是结构保证**
        # （其 client 的 config 是无白名单透传），故我方按结构收紧、不依赖对方不传。
        "ignition_beta",
        # canonical_physiology（2026-07-29 跨仓）：它决定 physiology 载荷的**量纲与键集**
        # （legacy 归一 [0,1] + pupil_mm ↔ canonical μS[0,20] + temperature_c），
        # 而消费侧据此选 skin_conductance_max_us=1.0 还是 20.0——选错即 20× 欠/过标度且不报错。
        # 🛑 入白名单前它**不在**治理内，而配套项目 client 对 config 是无白名单透传
        # ⇒ 两侧 env 全未设时，client 经 config 置真即可让载荷变成 canonical μS
        # （对方实测 sc=16.0），消费侧却按 env 推断成 legacy。
        # 收紧理由与 ignition_beta 逐字同型：它改的是**对外可见的载荷语义**，不容 client 单边翻转。
        # base 里已由 ZERO_PHYSIOLOGY_CANONICAL_PLACEHOLDER 播种，故入白名单是零额外接线。
        "canonical_physiology",
    }
)

# ── 机读错误码（zero-link 跨仓契约）────────────────────────────────────────
# 🛑 **必须是位置不敏感的方括号令牌，不能用位置 0 的裸前缀**。
# 原因（2026-07-29 两侧实证）：FastMCP 在工具层把 ToolError 加壳成
#   "Error executing tool <tool_name>: <原文>"
# ⇒ 任何写在位置 0 的前缀，在 wire 上**永远不在位置 0**。
# 旧实现 `_UNKNOWN_SESSION_MARKER = "unknown-session"` 正是裸前缀：配套项目按
# `text.lstrip().startswith(...)` 判定 ⇒ 对真 server 文本**恒 False**，
# T6·④ 的 resume 重试通路是**生产死码**；而两侧单测都用子串/未加壳夹具，故长期全绿。
# 典型「检查比消费方宽松 ⇒ 绿灯从没能红」。
#
# 现格式：`[zero:<code>]`，ASCII kebab-case，**全文恰出现一次**，位置不限。
# 消费方按 `re.search(r"\[zero:([a-z][a-z0-9-]*)\]")` 提取查表，无需与 SDK 抢位置。
# ⚠ 码值请按**符号名** pin（下面的常量），不要 pin 字面量。
ZERO_ERROR_CODE_UNKNOWN_SESSION = "unknown-session"
ZERO_ERROR_CODE_CONFIG_INCOMPATIBLE = "config-incompatible"
ZERO_ERROR_CODE_EXTERNAL_PRIOR_INVALID = "external-prior-invalid"
ZERO_ERROR_CODE_PAYLOAD_INVALID = "payload-invalid"
ZERO_ERROR_CODE_CONFIG_INVALID = "config-invalid"
ZERO_ERROR_CODE_DEPLOY_ENV_INVALID = "deploy-env-invalid"
# ── 超时：**两个码，不是一个**（Zero_MCP 2026-07-29 建议，我方采纳）──
# 二者**重试语义相反**，用同一个码等于把判别推回人读文案：
#   · timeout-lock：等锁超时，本轮**未进入内核**、运行态未改动 ⇒ 可退避后原样重试；
#   · timeout-step：内核执行超时，取消 ainvoke 会在 checkpointer 留**半截运行态**
#     （已实证：LangGraph 每个 super-step 写一次 checkpoint，我方图线性 10 节点）
#     ⇒ **不可原样重试**，重试会让已跑完的节点重跑、reducer 通道双重累加。
# ⚠ timeout-step 目前**只登记不产出**——执行超时尚未实现（选型见跨仓件，倾向 shield 或节点内降级）。
# 先登记是为了让消费方的分类表一次到位，不必等我方落地再改一轮。
ZERO_ERROR_CODE_TIMEOUT_LOCK = "timeout-lock"
ZERO_ERROR_CODE_TIMEOUT_STEP = "timeout-step"

ZERO_ERROR_CODES: frozenset[str] = frozenset(
    {
        ZERO_ERROR_CODE_UNKNOWN_SESSION,
        ZERO_ERROR_CODE_CONFIG_INCOMPATIBLE,
        ZERO_ERROR_CODE_EXTERNAL_PRIOR_INVALID,
        ZERO_ERROR_CODE_PAYLOAD_INVALID,
        ZERO_ERROR_CODE_CONFIG_INVALID,
        ZERO_ERROR_CODE_DEPLOY_ENV_INVALID,
        ZERO_ERROR_CODE_TIMEOUT_LOCK,
        ZERO_ERROR_CODE_TIMEOUT_STEP,
    }
)

# 兼容别名：旧名仍导出，值不变（配套项目现有断言含 "unknown-session" 子串者不受影响）。
_UNKNOWN_SESSION_MARKER = ZERO_ERROR_CODE_UNKNOWN_SESSION

# describe_config 的字段集版本：**增删任何键都要 bump**。
# 没有它，消费方在字段集演进后会静默少读（对方 desktop 面踩过这个坑）。
DESCRIBE_CONFIG_VERSION = 1


async def _purge_thread_state(thread_id: str) -> bool:
    """删掉某 thread_id 的全部持久 checkpoint；返回是否确有删除动作。

    走 LangGraph checkpointer 的公开 `adelete_thread`（存在则用）。老版本 saver 没有该方法时
    返回 False 并记 WARNING —— **不自己拼 SQL**：那会绑死 sqlite 的表结构，
    后端一换（postgres）就是静默失效或删错表。
    """
    from src.storage.checkpointer import build_checkpointer

    saver = build_checkpointer()
    deleter = getattr(saver, "adelete_thread", None)
    if deleter is None:
        logger.warning(
            "checkpointer %s 无 adelete_thread，purge 无法删除持久态 thread_id=%s",
            type(saver).__name__,
            thread_id,
        )
        return False
    await deleter(thread_id)
    return True


def _tool_error(code: str, message: str) -> ToolError:
    """构造带机读令牌的 `ToolError`：`[zero:<code>] <人读文案>`。

    令牌前置只为人读顺眼；消费方按 `search` 匹配，**不依赖位置**——
    FastMCP 加壳后它会落在文案中部，这正是本设计要容忍的。
    """
    if code not in ZERO_ERROR_CODES:  # 防手抖引入未登记码，消费方查表会漏
        raise ValueError(f"未登记的错误码 {code!r}；请先加进 ZERO_ERROR_CODES")
    # 净化人读文案里的同形字面量：坏载荷会被回显进文案（如 session_id={...!r}），
    # 若其中含 "[zero:" 就会出现第二个令牌，破坏「全文恰一次」契约、让消费方取到歧义结果。
    # 换开括号即可——保留可读性，且不引入零宽字符这类看不见的处理。
    return ToolError(f"[zero:{code}] {message.replace('[zero:', '(zero:')}")


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


def _env_optional_float(name: str, *, positive_finite: bool = False) -> float | None:
    """读「可选 float」env：未设 / 空串 → `None`（保留「未设即关」语义）；坏值抛 `ServerEnvError`。

    不能用 `_env_number`：那个必须给字符串默认值，而 `ignition_beta` 的 `None` 与任何浮点数
    语义不同（`None` = 硬 step 门，任何 float = 软门，含 `0.0`）。

    `positive_finite=True` 时**额外**要求解析结果是正的有限数。
    🛑 该判据必须 **opt-in**，绝不能做成本函数的无条件行为——两个使用者的合法域不同：
      · `ZERO_MCP_STEP_LOCK_TIMEOUT`：`≤0` 会让 `asyncio.wait_for` 走快路径、协程根本没机会
        执行 ⇒ **锁空闲时也无条件超时**（`nan` 效果等同 0），`inf` 则与未设等价；
        三类都让「上一轮 step 仍在执行…可原样重试」这句文案变成假话，故读取即拒。
      · `ZERO_MCP_IGNITION_BETA`：`0.0` 是**语义合法**的软门陡度（gate ≡ 0.5 均匀融合，
        退化但可运行），与未设（硬门 `None`）语义不同 ⇒ 套上同一判据会当场误伤。
    ⚠ 判据打在**解析后的 float** 上（`math.isfinite`），不做字符串黑名单：
    `float()` 接受 `"nan"` / `"inf"` / `"Infinity"` / `"1e400"`(→inf) / `"1_0"` 等多种写法。
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError as e:
        raise ServerEnvError(
            f"{name} 须为浮点数或留空，当前值={raw!r}；这是**部署端 env** 的问题，"
            "改 client 传的 config 无效"
        ) from e
    if positive_finite and not (value > 0 and math.isfinite(value)):
        raise ServerEnvError(
            f"{name} 须为正的有限秒数（>0，不接受 0/负数/nan/inf），当前值={raw!r}；"
            "留空=无限等待。这是**部署端 env** 的问题，改 client 传的 config 无效"
        )
    return value


async def _acquire_with_timeout(lock: asyncio.Lock, session_id: str) -> None:
    """按 `ZERO_MCP_STEP_LOCK_TIMEOUT`（秒，未设=无限）获取会话锁；超时抛 `[zero:timeout-lock]`。

    ⚠ **只超时「获取」，不超时「执行」**——见调用点注释。超时的是排队中的请求，
    正在跑的那一轮不受影响，故本函数**不会**引起运行态半落盘。
    （两个超时码 2026-07-29 已拆分：`timeout-lock` 可原样重试 / `timeout-step` 不可，
    后者只登记不产出；本函数只可能产出前者。）

    **值域**：该 env 须为正的有限秒数。`≤0` / `nan` 会让 `asyncio.wait_for` 在锁空闲时也
    无条件超时，`inf` 与未设等价 —— 三类一律读取即拒，出口是 `[zero:deploy-env-invalid]`
    （**部署端**归责，与 `open_session` 的同款分支同一句式，便于消费方分类表复用）。
    要「不设超时」请留空而不是写 `0`。

    ⚠ `asyncio.wait_for` 取消 `lock.acquire()` 时，CPython 的 `asyncio.Lock` 会在
    `CancelledError` 分支里把「取消瞬间恰好抢到锁」这种竞态还回去（唤醒下一个等待者）。
    该性质由 `test_lock_timeout_does_not_leak_the_lock` 实证锁住——它是本函数正确性的地基，
    不是可有可无的边界用例。
    """
    # 🛑 必须就地转成带码 ToolError：`ServerEnvError` 只在 `build_server.open_session` 被捕获转码，
    # 而本函数在 `zero.step` 里是**裸 await**（不在任何 try 内）——直接上抛会裸奔到 FastMCP、
    # 被加壳成 "Error executing tool zero.step: …"，wire 上**没有 `[zero:...]` 令牌**，
    # 消费方 `re.search` 取到 None。这正是 `_UNKNOWN_SESSION_MARKER` 死码事故的同型
    # （全过程见上方机读错误码注释段）。
    try:
        timeout = _env_optional_float("ZERO_MCP_STEP_LOCK_TIMEOUT", positive_finite=True)
    except ServerEnvError as e:
        raise _tool_error(
            ZERO_ERROR_CODE_DEPLOY_ENV_INVALID,
            f"服务端 env 配置不合法（**非** client config 问题）：{e}",
        ) from e
    if timeout is None:
        await lock.acquire()
        return
    try:
        await asyncio.wait_for(lock.acquire(), timeout)
    except TimeoutError as e:  # py3.11+ asyncio.TimeoutError 即 builtin TimeoutError
        raise _tool_error(
            ZERO_ERROR_CODE_TIMEOUT_LOCK,
            f"等待会话锁超时（{timeout}s）：sid={session_id!r} 上一轮 step 仍在执行。"
            "本轮未进入内核、运行态未改动，可原样重试",
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
        # ⚠ **成对要求**：进了 _MCP_GOVERNANCE_GATED_FLAGS 就必须在 base 里给 env 入口，
        # 否则该字段在 MCP 路径**永久取默认值**（对 ignition_beta 即永久 None）——design D6 的教训。
        "ignition_beta": _env_optional_float("ZERO_MCP_IGNITION_BETA"),
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
            "建/重开一个 Zero 情感引擎会话（建图/checkpointer 一次）。"
            "返回 {session_id, resumed}；resume 且探测到上一轮被中途取消时另带 "
            "{interrupted_at: [待执行节点名]}——该会话运行态停在 super-step 边界，"
            "续跑会从此处继续而非重跑整轮。传 session_id 则按旧 id 重开"
            "（跨 server 重启续会话·须持久后端）；不传则新铸。"
        ),
        structured_output=False,
    )
    async def open_session(
        persona: str | None = None,
        config: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:  # resumed 是 bool、interrupted_at 是 list[str]，故不能收窄成 dict[str,str]
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
            raise _tool_error(
                ZERO_ERROR_CODE_DEPLOY_ENV_INVALID,
                f"服务端 env 配置不合法（**非** client config 问题）：{e}",
            ) from e
        except (ValueError, TypeError) as e:
            raise _tool_error(ZERO_ERROR_CODE_CONFIG_INVALID, f"config 不合法：{e}") from e
        if session_id is not None:
            if not isinstance(session_id, str) or not session_id.strip():
                raise _tool_error(
                    ZERO_ERROR_CODE_PAYLOAD_INVALID,
                    f"session_id 须为非空字符串，实际为 {session_id!r}",
                )
            if await registry.get(session_id) is not None:
                # 同进程内仍活跃：幂等返回（不重建·不重开 aiosqlite 连接·不覆盖在飞运行态）。
                # 会话门控构造时固定、贯穿整会话，故此处不重应用 cfg：client 对活跃会话传的新
                # config 静默不生效（SessionConfig 不可变；跨重启 resume 重建时才用新 cfg）。
                logger.info("zero.open_session resume(active) sid=%s", session_id)
                # ⚠ 本分支**不得**回显刚算出的 `cfg`（Zero_MCP R11 实现警告）：活跃会话的门控
                # 构造时固定，回显 cfg = 回显「本可生效但实际未生效」的值，**比不回显更危险**
                # （消费方会拿它当真、比对通过、实则语义已分叉）。要回显只能从 session 对象取。
                return {"session_id": session_id, "resumed": True}
            # resume：新建绑同 thread_id，持久后端 ainvoke 时自动从 checkpoint 恢复。
            sid = session_id
            resuming = True
        else:
            sid = uuid.uuid4().hex
            resuming = False
        # thread_id=user_id=sid：会话态经 checkpointer 按 thread_id 跨轮持久 + 记忆作用域天然按
        # session 隔离（防多会话共享 default-user 记忆串味，同 chat_driver 口径）。
        session = ConversationSession(
            thread_id=sid,
            user_id=sid,
            config=cfg,
            expression_decoder=decoder,
        )
        await registry.open(sid, session)
        # ── resume 时先看一眼上一轮是否被中途取消（2026-07-29 跨仓实证）──
        # 我方默认 stdio 传输，client 关 stdin 即让在飞的 step 在 +0.015s 被取消，
        # LangGraph 每个 super-step 都已落盘 ⇒ 半截运行态跨重启保留。
        # 🛑 半截 checkpoint 存在本身不致命，**被当成有效状态续跑才致命** —— 此处至少让它可见。
        # ⚠ 只报告不回滚：回滚是对外可见的行为变更，须单独决策。
        interrupted: tuple[str, ...] | None = None
        if resuming:
            try:
                interrupted = await session.interrupted_at()
            except Exception:
                # 探测失败不得挡住 resume 本身（宁可少一条观测量，也不把会话打不开）。
                logger.exception("zero.open_session 中断探测失败 sid=%s", sid)
            if interrupted is not None:
                logger.warning(
                    "zero.open_session resume 发现上一轮被中断 sid=%s 待执行节点=%s；"
                    "该会话的运行态停在 super-step 边界，续跑会从此处继续而非重跑整轮",
                    sid,
                    interrupted,
                )
        logger.info(
            "zero.open_session sid=%s persona=%s resume=%s active=%d",
            sid,
            persona,
            session_id is not None,
            await registry.count(),
        )
        # 返回体**只增不改**：配套项目按「除 session_id 外容忍并收下额外键、缺键即回落」解析，
        # 故新增键对现网零回归。`interrupted` 缺席 = 未探测（新建会话）或探测失败。
        out: dict[str, Any] = {"session_id": sid, "resumed": resuming}
        if interrupted is not None:
            out["interrupted_at"] = list(interrupted)
        return out

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
            raise _tool_error(
                ZERO_ERROR_CODE_UNKNOWN_SESSION,
                f"未知 session_id={session_id!r}；"
                "请先调 zero.open_session（可用同 id resume 续会话）",
            )
        try:
            stimulus = stimulus_from_payload(stim)
            priors = external_priors_from_payload(external_priors)
        except (ValueError, TypeError) as e:
            raise _tool_error(
                ZERO_ERROR_CODE_PAYLOAD_INVALID, f"stim/external_priors 载荷不合法：{e}"
            ) from e
        # 同会话串行化：LangGraph checkpointer 读-改-写非原子，并发 ainvoke(同 thread_id) 会竞态
        # （http 传输下 client 可能并发）；stdio 顺序则此锁无争用、零成本。
        # ── 锁**获取**超时（R12 上半，2026-07-29 跨仓契约）──
        # ⚠ 这里刻意只给「获取」加超时，**不给 step 执行加超时**：
        #   · 获取超时只让**排队者**失败，不碰正在跑的 ainvoke ⇒ 无状态风险；
        #   · 执行超时要取消 ainvoke，而「LangGraph 被取消时 checkpointer 会不会落半截运行态」
        #     今天**未经证实**（已向对方标 UNVERIFIED），在答案出来前取消 ainvoke 可能留下脏态、
        #     被下一次同 thread_id 的 resume 读到。宁可不做，也不做成一个会静默污染运行态的超时。
        # 默认 None = 无限等待 = 逐字旧行为（零回归）；秒数按对方 R12「先测再钉」，暂不预设。
        await _acquire_with_timeout(lock, session_id)
        try:
            # 🛑 拿到锁后**复查在册**（TOCTOU）：`acquire()` 到真正拿到锁之间，
            # `close_session` 可能已把该会话摘牌并关掉 aiosqlite 连接。此时若照常 step，
            # 会撞 `ValueError("Connection closed")` 并被下面的 except ValueError 贴成
            # **`config-incompatible`** —— 归责完全错（既不是配置不兼容、也不是传参问题，
            # 而是会话已被关闭），且该码语义是「须以新配置重开」，会把 client 引到错误的自救动作。
            # 正确语义是 `unknown-session`：会话没了，同 id 重开即可。
            if await registry.get(session_id) is None:
                raise _tool_error(
                    ZERO_ERROR_CODE_UNKNOWN_SESSION,
                    f"会话 {session_id!r} 在本轮排队等待期间已被关闭；"
                    "本轮未进入内核，可用同 id 重开后重试",
                )
            try:
                step_out = await session.step(stimulus, state_overrides={"external_priors": priors})
            except ExternalPriorError as e:
                # expand_external_priors 的 M3/M6/M7 fail-fast（精度>0/≤cap、流数≤max、
                # 形状良构、μ∈[-1,1]）——**确实**指向 client 传参，改传参就能好。
                raise _tool_error(
                    ZERO_ERROR_CODE_EXTERNAL_PRIOR_INVALID,
                    f"external_priors 校验失败（指向 MCP 传参）：{e}",
                ) from e
            except ValueError as e:
                # 内核其它位置抛的 ValueError（如 HPC coupling 越界、未来的配置互斥 fail-fast）。
                # ⚠ 这里**必须**与上一分支分开报（议会 2026-07-29 第五轮校验 §四-5）：
                # 原先一个 except 包住整个 step（全图执行），把内核任何异常都贴成
                # 「external_priors 校验失败（指向 MCP 传参）」→ 误导性甩锅。client 照着改
                # 传参永远改不好，而活跃会话的 config 不可变（见 open_session 的 config 语义）
                # → 无法自救，表现为 open 成功、**每 step 崩**。
                logger.exception("zero.step 内核执行失败 sid=%s", session_id)
                raise _tool_error(
                    ZERO_ERROR_CODE_CONFIG_INCOMPATIBLE,
                    f"内核执行失败（**非** external_priors 传参问题，改传参无效）：{e}；"
                    "多为会话级配置组合不兼容，须以新配置重开会话",
                ) from e
        finally:
            lock.release()
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
        # ⚠ 顺序是**先摘牌、再取锁**（2026-07-29 改；此前是裸 `async with lock:` 包住两步）。
        # 旧顺序的问题：一次挂起的 step **永久持有会话锁** ⇒ close 无限等待、连关会话都做不到
        # （配套项目原话「这条我方无法自救」——client 侧超时不发取消通知，锁只能我方自解）。
        # 新顺序：
        #   ① 先 `registry.close()` 摘牌 —— 立刻止住新活（后续 step 拿不到会话），且它只持
        #      registry 表级锁、不碰会话锁，**不会被在飞 step 卡住**；
        #   ② 再带超时取会话锁做 `aclose()` —— 拿到锁说明在飞 step 已结束，关连接无竞态。
        # 🛑 为什么 aclose 一定要等锁：摘牌后仍可能有 step **已经取到 session/lock 引用**在排队，
        #    此时提前关连接会让它们撞 `ValueError("Connection closed")`。
        #    该竞态的另一半由 `zero.step` 的「拿到锁后复查在册」兜住
        #    （报 unknown-session 而非被误贴成 config-incompatible）。
        await registry.close(session_id)  # 表内移除 → 后续 step 同 id 回 unknown-session
        try:
            await _acquire_with_timeout(lock, session_id)
        except ToolError:
            # ⚠ 已知遗留（本轮接受·不修）：这一支也会吞掉 `_acquire_with_timeout` 新增的
            # `deploy-env-invalid`（ZERO_MCP_STEP_LOCK_TIMEOUT 值域不合法），并误记成下面那条
            # 「等锁超时」WARNING。收窄本 except 属 close 降级路径的语义变更，另走门；
            # 对 client 可见行为不变（仍 `{ok: True}`），只是日志归因错。
            # 超时：在飞 step 仍未结束。**不上抛**——`close_session` 的契约是幂等 `{ok:true}`，
            # 为「连接没来得及关」而破坏它得不偿失（会话已摘牌，不会再接新活）。
            # 连接随对象回收/进程退出释放；长驻 HTTP server 下这是一条可观测的泄漏，故记 WARNING。
            logger.warning(
                "zero.close_session 等锁超时 sid=%s：会话已摘牌但 aiosqlite 连接未关"
                "（在飞 step 仍在执行）。连接将随对象回收释放",
                session_id,
            )
            return {"ok": True}
        try:
            await session.aclose()  # 关运行态 aiosqlite 连接（sqlite）/InMemory no-op·幂等
        finally:
            lock.release()
        logger.info("zero.close_session sid=%s active=%d", session_id, await registry.count())
        return {"ok": True}

    @mcp.tool(
        name="zero.describe_config",
        description=(
            "只读回读面：不传 session_id 返回**部署端默认**（env + caps + versions，"
            "供在 open_session 之前决定是否发某类流）；传 session_id 返回**该会话真实生效**的值。"
            "未知 session_id 视同不传。字段集演进看 describe_config_version。"
        ),
        structured_output=False,
    )
    async def describe_config(session_id: str | None = None) -> dict[str, Any]:
        """回读生效门控 —— 配套项目在**运行期**确认「这个会话到底拿到哪几个门」的唯一入口。

        为什么必须有（Zero_MCP 2026-07-29）：我方源码注释把「确认部署端已开某 env」的义务
        派给了消费方，却只挂 open/step/close 三个工具、`open_session` 只回 `{session_id}`
        ⇒ 消费方**无手段可确认**。跨仓 env 对照表因此只能当文档、不能当校验。
        且 HTTP 传输下两进程**不共享 env**，「同名 env 对齐」这条机制结构上不成立。

        形制按对方三条要求（它们各自对应其 desktop 面踩过的一个坑）：
        1. **带版本号** —— 否则字段集增删后旧 client 静默少读；
        2. **按字段名显式取值，不得用类型过滤器** —— 其 desktop client 用
           `{k: bool(v) for … if isinstance(v, bool)}` 过滤，实测让一个非 bool 字段无声蒸发；
        3. **不可知项显式回 `null`，不省略键、不回 `False`** —— 否则「探测失败」与
           「该能力不可用」不可区分。

        🛑 取值源：**从 `session.config` 取，不从 `_build_session_config` 刚算出的 cfg 取**
        （对方 R11 实现警告）。活跃会话的门控构造时固定，回显刚算的 cfg 等于回显
        「本可生效但实际未生效」的值 —— 比不回显更危险：消费方会拿它当真、比对通过、
        实则语义已分叉。
        """
        session = await registry.get(session_id) if session_id else None
        cfg = session.config if session is not None else _build_session_config(None)
        return {
            "describe_config_version": DESCRIBE_CONFIG_VERSION,
            "session_id": session_id,
            "resolved_for_session": session is not None,
            "workspace_enabled": cfg.workspace_enabled,
            "gate_fusion": cfg.gate_fusion,
            "exclude_physio_fusion": cfg.exclude_physio_fusion,
            "precision_commensurable": cfg.precision_commensurable,
            "ignition_beta": cfg.ignition_beta,
            "coping_potential_enabled": cfg.coping_potential_enabled,
            "text_coping_enabled": cfg.text_coping_enabled,
            "fear_domain_enabled": cfg.fear_domain_enabled,
            "canonical_physiology": cfg.canonical_physiology,
            "facs_extended": cfg.facs_extended,
            "external_prior_precision_cap": cfg.external_prior_precision_cap,
            "max_external_streams": cfg.max_external_streams,
            "external_prior_schema_version": EXTERNAL_PRIOR_SCHEMA_VERSION,
            "governance_gated_flags": sorted(_MCP_GOVERNANCE_GATED_FLAGS),
            "error_codes": sorted(ZERO_ERROR_CODES),
            # 第二批（对方已接受后置）：显式回 null，**不省略键**——省略会让「未实现」
            # 与「探测失败」不可区分，正是对方第 3 条形制要求要避开的。
            "sample_sigma_cap": cfg.sample_sigma_cap,
            "affect_readout": cfg.affect_readout,
            "weights_version": None,  # sidecar 今天无读取方，见跨仓件
        }

    @mcp.tool(
        name="zero.purge_session",
        description=(
            "删除一个会话的**全部持久运行态**（按 thread_id 清 checkpoint）；未知 id 幂等。"
            "返回 {'ok': true, 'purged': bool}。⚠ 不可逆，与 close_session 语义不同："
            "close 只释放连接与登记，purge 才真正删数据。"
        ),
        structured_output=False,
    )
    async def purge_session(session_id: str) -> dict[str, Any]:
        """跨仓数据删除入口（Zero_MCP §4.5 三项提案之一）。

        ⚠ **保留期天数与「哪侧是数据控制方」仍待双方各自的人拍板**，本工具不预设策略、
        不自动清理 —— 它只提供「被要求删除时能删掉」的能力。双方已确认该能力的落地
        **不依赖**那两项定性。

        实现：先 `close_session` 走一遍（释放连接、摘牌，幂等），再删 checkpoint。
        """
        await close_session(session_id)
        purged = await _purge_thread_state(session_id)
        logger.info("zero.purge_session sid=%s purged=%s", session_id, purged)
        return {"ok": True, "purged": purged}

    return mcp
