"""zero-link MCP server 契约测试：映射纯函数 + FastMCP in-memory client 往返 + 错误路径。

`mcp` 走 optional-deps（默认不入 core），无 mcp 的机器整包不可导入 → `importorskip` 跳过（零回归）。
断言对齐 MCP 侧 client 期望契约（open→{session_id} / step→expression 子 dict / close→{ok}；
错误→isError=true）。
"""

from __future__ import annotations

import json
from pathlib import Path  # _facs_v2_weights() 用

import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import (  # noqa: E402
    create_connected_server_and_client_session as connect,
)

from src.mcp_server.mapping import (  # noqa: E402
    external_priors_from_payload,
    stimulus_from_payload,
)
from src.mcp_server.server import build_server  # noqa: E402

# ── 映射纯函数（无 mcp 交互）─────────────────────────────────────────────


def test_stimulus_maps_valence_arousal_coping() -> None:
    stim = stimulus_from_payload({"valence": 0.6, "arousal": 0.5, "coping_potential": 0.3})
    assert stim.goal_congruence == 0.6  # valence → goal_congruence
    assert stim.intensity == pytest.approx(0.5)  # |arousal| → intensity
    assert stim.control_appraisal == pytest.approx(0.3)  # coping → control_appraisal
    assert stim.attitude_appeal == 0.0  # 会话边界不承载 chat 层 attitude
    assert stim.text is None  # 不跑我方文本回归器


def test_stimulus_omits_coping_is_absent_cue() -> None:
    # 省略 coping → control_appraisal 默认 None=absent cue（B3 新语义 state.py:36）；
    # 与 client model_dump(exclude_none=True) 省略 None 天然对齐。显式 0.0 才是 genuine-zero。
    stim = stimulus_from_payload({"valence": -0.2, "arousal": 0.1})
    assert stim.control_appraisal is None
    stim0 = stimulus_from_payload({"valence": 0.0, "arousal": 0.0, "coping_potential": 0.0})
    assert stim0.control_appraisal == 0.0  # 显式 0.0 → genuine-zero（参与融合）


def test_stimulus_intensity_is_abs_and_clamped() -> None:
    assert stimulus_from_payload({"valence": 0.0, "arousal": -0.8}).intensity == pytest.approx(0.8)
    assert stimulus_from_payload({"valence": 0.0, "arousal": 1.5}).intensity == 1.0  # clamp ≤1


def test_stimulus_out_of_domain_valence_raises() -> None:
    """越域 valence 在边界 fail-fast（议会 2026-07-28 第四轮 A4 落地后的**新契约**）。

    ⚠ 与 arousal 的处置**有意不对称**，见 `mapping.py` docstring：
      - `intensity = min(1.0, max(floor, |arousal|))` 是**语义映射**（幅度→强度），
        `min(1.0, ...)` 顺带把越域 arousal 静默钳到 1.0；
      - `goal_congruence = valence` 是**恒等透传**，越域即 client 违约 → 拒绝，
        与 M3/M6/M7 一致（fail-fast 指向 MCP 传参）。
    该拒绝不会裸崩：`server.py:288-292` 的 `except (ValueError, TypeError)` 转 ToolError。
    """
    for bad in (1.5, -1.5, 1.0001):
        with pytest.raises(ValueError):  # pydantic ValidationError 是 ValueError 子类
            stimulus_from_payload({"valence": bad, "arousal": 0.1})
    # 边界值合法，不得误伤
    for ok in (-1.0, 1.0, 0.0):
        assert stimulus_from_payload({"valence": ok, "arousal": 0.1}).goal_congruence == ok
    # 对照：同样越域的 arousal 仍被静默钳制（不对称是有意的，改它须与 Zero_MCP 协调）
    assert stimulus_from_payload({"valence": 0.0, "arousal": 99.0}).intensity == 1.0


def test_stimulus_missing_va_raises() -> None:
    with pytest.raises(ValueError):
        stimulus_from_payload({"valence": 0.5})  # 缺 arousal


def test_external_priors_array_to_tuple() -> None:
    out = external_priors_from_payload([["face", [-0.3, 0.6], [0.2, 0.12]]])
    assert out == [("face", (-0.3, 0.6), (0.2, 0.12))]
    assert isinstance(out[0], tuple) and isinstance(out[0][1], tuple)  # 真 tuple（expand_ 要求）


def test_external_priors_empty() -> None:
    assert external_priors_from_payload(None) == []
    assert external_priors_from_payload([]) == []


def test_external_priors_bad_shape_raises() -> None:
    with pytest.raises(ValueError):
        external_priors_from_payload([["face", [0.1], [0.2, 0.1]]])  # μ 非 2 元


# ── FastMCP in-memory 往返 ───────────────────────────────────────────────


async def test_open_step_close_roundtrip() -> None:
    async with connect(build_server()) as client:
        await client.initialize()
        names = {t.name for t in (await client.list_tools()).tools}
        assert names == {"zero.open_session", "zero.step", "zero.close_session"}  # 点名工具

        r = await client.call_tool("zero.open_session", {})
        assert r.isError is False
        session_id = json.loads(r.content[0].text)["session_id"]
        assert isinstance(session_id, str) and session_id

        r = await client.call_tool(
            "zero.step",
            {
                "session_id": session_id,
                "stim": {"valence": 0.6, "arousal": 0.5, "coping_potential": 0.3},
            },
        )
        assert r.isError is False
        expr = json.loads(r.content[0].text)  # content[0].text 是 JSON（client 读法）
        assert {"valence_arousal", "spontaneous", "voluntary"} <= set(expr)
        assert (
            isinstance(expr["valence_arousal"], list) and len(expr["valence_arousal"]) == 2
        )  # array
        for head in ("spontaneous", "voluntary"):
            assert {"facs_au", "text_label", "physiology", "prosody"} <= set(expr[head])

        r = await client.call_tool("zero.close_session", {"session_id": session_id})
        assert r.isError is False
        assert json.loads(r.content[0].text) == {"ok": True}


async def test_step_unknown_session_is_error() -> None:
    async with connect(build_server()) as client:
        await client.initialize()
        r = await client.call_tool(
            "zero.step", {"session_id": "does-not-exist", "stim": {"valence": 0.0, "arousal": 0.0}}
        )
        assert r.isError is True
        assert "session_id" in r.content[0].text
        assert "unknown-session" in r.content[0].text  # 机读标记（T6·MCP graceful_step 据此降级）


async def test_step_bad_external_prior_is_error() -> None:
    """精度 9.9 > cap 0.8 → expand_external_priors M3 fail-fast → ToolError → isError=true。"""
    async with connect(build_server()) as client:
        await client.initialize()
        sid = json.loads((await client.call_tool("zero.open_session", {})).content[0].text)[
            "session_id"
        ]
        r = await client.call_tool(
            "zero.step",
            {
                "session_id": sid,
                "stim": {"valence": 0.0, "arousal": 0.0},
                "external_priors": [["face", [0.0, 0.0], [9.9, 0.1]]],
            },
        )
        assert r.isError is True


async def test_step_with_valid_external_prior_ok() -> None:
    """workspace_enabled 默认 True → 一条合法 face 先验被消费，step 不崩、expression 良构。"""
    async with connect(build_server()) as client:
        await client.initialize()
        sid = json.loads((await client.call_tool("zero.open_session", {})).content[0].text)[
            "session_id"
        ]
        r = await client.call_tool(
            "zero.step",
            {
                "session_id": sid,
                "stim": {"valence": -0.4, "arousal": 0.7},
                "external_priors": [["face", [-0.3, 0.6], [0.2, 0.12]]],
            },
        )
        assert r.isError is False
        assert "valence_arousal" in json.loads(r.content[0].text)


async def test_session_state_accumulates_across_steps() -> None:
    """同一 session_id 跨 step 复用同一 thread_id → 值表跨轮持久（有状态 actor）。"""
    async with connect(build_server()) as client:
        await client.initialize()
        sid = json.loads((await client.call_tool("zero.open_session", {})).content[0].text)[
            "session_id"
        ]
        for _ in range(3):
            r = await client.call_tool(
                "zero.step", {"session_id": sid, "stim": {"valence": 0.8, "arousal": 0.6}}
            )
            assert r.isError is False  # 多轮不崩、会话句柄稳定


async def test_concurrent_steps_same_session_serialized() -> None:
    """并发向同一 session_id 发多个 step：per-session 锁串行化，不竞态 checkpointer（W4）。"""
    import asyncio

    async with connect(build_server()) as client:
        await client.initialize()
        sid = json.loads((await client.call_tool("zero.open_session", {})).content[0].text)[
            "session_id"
        ]
        results = await asyncio.gather(
            *(
                client.call_tool(
                    "zero.step", {"session_id": sid, "stim": {"valence": 0.3, "arousal": 0.4}}
                )
                for _ in range(5)
            )
        )
        assert all(r.isError is False for r in results)
        assert all("valence_arousal" in json.loads(r.content[0].text) for r in results)


# ── 真 13-AU 权重路径（需 torch + 权重文件；权重走 gitignore/Release，缺则跳过）──────────


def _facs_v2_weights() -> Path | None:
    p = Path(__file__).resolve().parents[1] / "artifacts" / "facs_decoder_ext_v2.pt"
    return p if p.exists() else None


async def test_real_facs_decoder_13au_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """设 ZERO_FACS_MODEL_PATH → 注入真 13-AU 解码器：出全 13 键 facs_au、值为 python float
    （无 numpy 泄漏、JSON 可序列化过边界）。权重/torch 缺则优雅跳过（CI 零依赖）。
    """
    pytest.importorskip("torch")
    weights = _facs_v2_weights()
    if weights is None:
        pytest.skip("facs_decoder_ext_v2.pt 权重不在（gitignore/Release 分发）")
    monkeypatch.setenv("ZERO_FACS_MODEL_PATH", str(weights))
    monkeypatch.setenv("ZERO_FACS_EXTENDED", "true")
    async with connect(build_server()) as client:
        await client.initialize()
        sid = json.loads((await client.call_tool("zero.open_session", {})).content[0].text)[
            "session_id"
        ]
        r = await client.call_tool(
            "zero.step",
            {"session_id": sid, "stim": {"valence": -0.5, "arousal": 0.7, "coping_potential": 0.8}},
        )
        assert r.isError is False
        from src.agents.models.facs_decoder import FACS_KEYS_EXT

        spontaneous = json.loads(r.content[0].text)["spontaneous"]
        assert set(spontaneous["facs_au"]) == set(FACS_KEYS_EXT)  # 键名精确对齐（非仅数量）
        assert all(isinstance(v, float) for v in spontaneous["facs_au"].values())  # JSON-safe
        assert all(0.0 <= v <= 1.0 for v in spontaneous["facs_au"].values())  # 值域守界 [0,1]


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [("true", True), ("1", True), ("false", False), (None, False)],
)
def test_mcp_facs_extended_env_seeds_session_config(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None, expected: bool
) -> None:
    """ZERO_FACS_EXTENDED 必须同时播种 state.facs_extended，而不只喂 decoder。

    缺口（本用例即为其回归锁）：`_build_session_config` 的 base 曾漏掉这一键，而
    `_maybe_expression_decoder` 读了它 → 同设 ZERO_FACS_EXTENDED=true 与 ZERO_FACS_MODEL_PATH 时，
    decoder 载入 13 键真模型、state 却取字段默认 False，C2 residual 的 coping 分野
    （AU23/AU01/AU02/AU20）被静默跳过。修复前 env_value="true" 这格返回 False → 本用例变红。
    """
    from src.mcp_server.server import _build_session_config

    if env_value is None:
        monkeypatch.delenv("ZERO_FACS_EXTENDED", raising=False)
    else:
        monkeypatch.setenv("ZERO_FACS_EXTENDED", env_value)
    assert _build_session_config(None).facs_extended is expected


def test_mcp_facs_extended_same_env_as_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """同源契约：session config 与 decoder 构造读的是**同一个** env 名。

    不比对字面量（那只会锁住注释），而是改 env 后断言两侧**一起翻**——任一侧改读别的 env 即红。
    """
    pytest.importorskip("torch")  # _maybe_expression_decoder 的 facs 分支延迟 import models 层
    import src.agents.models.facs_decoder as facs_mod
    import src.mcp_server.server as srv

    captured: dict[str, bool] = {}

    def _fake_load(path: str, extended: bool) -> object:
        captured["extended"] = extended
        return object()

    monkeypatch.setattr(facs_mod, "load_facs_decoder", _fake_load)
    monkeypatch.setenv("ZERO_FACS_MODEL_PATH", "dummy.pt")
    monkeypatch.delenv("ZERO_PROSODY_MODEL_PATH", raising=False)
    monkeypatch.delenv("ZERO_PHYSIOLOGY_MODEL_PATH", raising=False)
    for value, expected in (("true", True), ("false", False)):
        captured.clear()
        monkeypatch.setenv("ZERO_FACS_EXTENDED", value)
        srv._maybe_expression_decoder()
        assert captured["extended"] is expected, "decoder 侧未跟随 env"
        assert srv._build_session_config(None).facs_extended is expected, "state 侧未跟随 env"


# ── 机读错误码：位置不敏感令牌 [zero:<code>] ─────────────────────────────


def _wire_code(text: str) -> str | None:
    """按**配套项目实际用的**正则从 wire 文本里提取码——不用我方内部知识取巧。"""
    import re

    m = re.search(r"\[zero:([a-z][a-z0-9-]*)\]", text)
    return m.group(1) if m else None


async def test_error_token_survives_fastmcp_wrapping() -> None:
    """🛑 本组的核心判别力：令牌必须在 **FastMCP 加壳后**仍可被提取。

    2026-07-29 两侧实证：FastMCP 把 ToolError 加壳成
    `Error executing tool <name>: <原文>` ⇒ **位置 0 的裸前缀在 wire 上永远不在位置 0**。
    旧实现 `_UNKNOWN_SESSION_MARKER` 正是裸前缀，配套项目按 `startswith` 判定 ⇒ 恒 False，
    其 resume 重试通路是**生产死码**；而两侧旧用例都用子串/未加壳夹具，故长期全绿。

    本用例**刻意同时断言两件事**：① 加壳确实发生（否则本用例没在测东西）；
    ② `startswith` 确实不成立——把这条钉死，防日后有人「顺手」改回裸前缀。
    """
    from src.mcp_server.server import ZERO_ERROR_CODE_UNKNOWN_SESSION

    async with connect(build_server()) as client:
        await client.initialize()
        r = await client.call_tool(
            "zero.step", {"session_id": "nope", "stim": {"valence": 0.0, "arousal": 0.0}}
        )
        assert r.isError is True
        text = getattr(r.content[0], "text", "")
        assert text.startswith("Error executing tool"), "FastMCP 未加壳 → 本用例失去判别力，须重写"
        assert not text.lstrip().startswith(ZERO_ERROR_CODE_UNKNOWN_SESSION), (
            "裸前缀在 wire 上不可能成立；若这条通过，说明有人改回了位置敏感格式"
        )
        assert _wire_code(text) == ZERO_ERROR_CODE_UNKNOWN_SESSION


@pytest.mark.parametrize(
    ("call", "expected_code"),
    [
        (
            ("zero.step", {"session_id": "nope", "stim": {"valence": 0.0, "arousal": 0.0}}),
            "unknown-session",
        ),
        (
            ("zero.open_session", {"session_id": "   "}),
            "payload-invalid",
        ),
    ],
)
async def test_error_codes_are_extractable(call: tuple[str, dict], expected_code: str) -> None:
    """各错误出口都带得上令牌，且码值取自登记表。"""
    from src.mcp_server.server import ZERO_ERROR_CODES

    name, args = call
    async with connect(build_server()) as client:
        await client.initialize()
        r = await client.call_tool(name, args)
        assert r.isError is True
        code = _wire_code(getattr(r.content[0], "text", ""))
        assert code == expected_code
        assert code in ZERO_ERROR_CODES


async def test_deploy_env_error_carries_its_own_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """部署端 env 坏值的码与 client-config 坏值**必须不同**——归责靠它区分。"""
    monkeypatch.setenv("ZERO_EXTERNAL_PRIOR_PRECISION_CAP", "0.8x")
    async with connect(build_server()) as client:
        await client.initialize()
        r = await client.call_tool("zero.open_session", {})
        assert _wire_code(getattr(r.content[0], "text", "")) == "deploy-env-invalid"


def test_tool_error_rejects_unregistered_code() -> None:
    """未登记的码构造即抛——防手抖引入消费方查不到的码。"""
    from src.mcp_server.server import _tool_error

    with pytest.raises(ValueError, match="未登记的错误码"):
        _tool_error("made-up-code", "x")


def test_error_token_appears_exactly_once() -> None:
    """契约要求「全文恰出现一次」——多于一次会让消费方的 search 取到歧义结果。"""
    from src.mcp_server.server import ZERO_ERROR_CODE_CONFIG_INVALID, _tool_error

    text = str(_tool_error(ZERO_ERROR_CODE_CONFIG_INVALID, "内含 [zero:] 字样的用户输入"))
    assert text.count("[zero:") == 1


# ── 结构性守卫 + 锁获取超时 ───────────────────────────────────────────────


def test_no_toolerror_raised_outside_the_helper() -> None:
    """🛑 AST 级：`server.py` 里**不得**有绕过 `_tool_error` 的 `raise ToolError(...)`。

    配套项目 2026-07-29 指出：它按 `re.search` 取**首个**令牌，若某条 raise 绕过 `_tool_error`，
    ① 该错误没有码、② 回显载荷里的 `[zero:` 不会被净化 ⇒ 它会取到用户输入伪造的令牌而不报歧义。
    我方原有的 `test_error_token_appears_exactly_once` 只覆盖 `_tool_error` **自身**，
    挡不住「新增一处裸 raise」——本用例补的正是那条缝。
    """
    import ast
    import inspect

    import src.mcp_server.server as srv

    tree = ast.parse(inspect.getsource(srv))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "ToolError"
    ]
    assert not offenders, (
        f"server.py 第 {offenders} 行直接 raise 了 ToolError；请改用 _tool_error(code, msg)——"
        "裸 ToolError 既无机读码、也不会净化回显载荷里的 [zero: 字面量"
    )


async def test_lock_timeout_does_not_leak_the_lock() -> None:
    """🛑 本组的地基：`wait_for` 取消 `Lock.acquire()` **不得**把锁留在「已占用但无人持有」态。

    若 CPython 的 `asyncio.Lock` 在取消瞬间恰好抢到锁却不还回去，超时一次就会**永久锁死该会话**
    ——那比不加超时更糟。这条不是边界用例，是 `_acquire_with_timeout` 能不能用的前提。
    """
    import asyncio

    lock = asyncio.Lock()
    await lock.acquire()  # 持有者
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(lock.acquire(), 0.05)  # 排队者超时
    assert lock.locked(), "持有者仍应持有"
    lock.release()
    assert not lock.locked(), "释放后必须真正可用——否则超时泄漏了锁"
    await asyncio.wait_for(lock.acquire(), 0.05)  # 后续等待者拿得到
    lock.release()


async def test_step_lock_timeout_yields_timeout_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """锁被占住时，排队的 step 超时 → `[zero:timeout]`，且**未进入内核**。"""
    import asyncio

    from src.mcp_server.registry import SessionRegistry

    monkeypatch.setenv("ZERO_MCP_STEP_LOCK_TIMEOUT", "0.05")
    registry = SessionRegistry()
    async with connect(build_server(registry=registry)) as client:
        await client.initialize()
        # getattr 而非 .text：见本文件既有说明（不给 mypy tests 基线添 union-attr 增量）
        opened = await client.call_tool("zero.open_session", {})
        sid = json.loads(getattr(opened.content[0], "text", ""))["session_id"]
        _, lock = await registry.acquire(sid)
        assert lock is not None
        await lock.acquire()  # 模拟「上一轮 step 仍在执行」
        try:
            r = await client.call_tool(
                "zero.step", {"session_id": sid, "stim": {"valence": 0.0, "arousal": 0.0}}
            )
            assert r.isError is True
            text = getattr(r.content[0], "text", "")
            assert _wire_code(text) == "timeout"
            assert "可原样重试" in text, "须告知调用方本轮未改动运行态"
        finally:
            lock.release()

        # 锁释放后照常可用——超时那一轮没留下任何副作用
        r2 = await client.call_tool(
            "zero.step", {"session_id": sid, "stim": {"valence": 0.1, "arousal": 0.1}}
        )
        assert r2.isError is False
        await asyncio.sleep(0)


async def test_no_lock_timeout_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """未设 env = 无限等待 = 逐字旧行为（零回归）；空串同样回落无限等待。"""
    from src.mcp_server.server import _env_optional_float

    for value in (None, "", "  "):
        if value is None:
            monkeypatch.delenv("ZERO_MCP_STEP_LOCK_TIMEOUT", raising=False)
        else:
            monkeypatch.setenv("ZERO_MCP_STEP_LOCK_TIMEOUT", value)
        assert _env_optional_float("ZERO_MCP_STEP_LOCK_TIMEOUT") is None


def test_server_env_error_is_not_valueerror() -> None:
    """判别力自证：`ServerEnvError` **不得**继承 ValueError/TypeError。

    整个归责修复就靠这一点——一旦它落进 `open_session` 的
    `except (ValueError, TypeError)`，部署端 env 写错又会被贴成「config 不合法」，
    修复静默失效而所有用例照样绿。若日后有人改继承链，本用例即红。
    """
    from src.mcp_server.server import ServerEnvError

    assert not issubclass(ServerEnvError, (ValueError, TypeError))


@pytest.mark.parametrize(
    ("env_name", "bad_value"),
    [("ZERO_EXTERNAL_PRIOR_PRECISION_CAP", "0.8x"), ("ZERO_MAX_EXTERNAL_STREAMS", "many")],
)
def test_numeric_env_bad_value_points_at_env(
    monkeypatch: pytest.MonkeyPatch, env_name: str, bad_value: str
) -> None:
    """部署端数值 env 写坏 → `ServerEnvError` 且消息指名 env（不是 client 的 config）。"""
    from src.mcp_server.server import ServerEnvError, _build_session_config

    monkeypatch.setenv(env_name, bad_value)
    with pytest.raises(ServerEnvError, match=env_name):
        _build_session_config(None)


async def test_open_session_env_error_message_does_not_blame_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """端到端：坏 env 经 zero.open_session 返回的错误文案须点明「非 client config 问题」。

    修复前该错误被贴成「config 不合法」——而 stdio 下 client 进程环境就是 server 进程环境，
    对方会照着改自己传的 config，永远改不好。
    """
    monkeypatch.setenv("ZERO_EXTERNAL_PRIOR_PRECISION_CAP", "0.8x")
    async with connect(build_server()) as client:
        await client.initialize()
        r = await client.call_tool("zero.open_session", {})
        assert r.isError is True
        # getattr 而非 `.text`：content 是 TextContent|ImageContent|… 联合类型，直接取 .text
        # 会给 `mypy tests` 添 4 条 union-attr（本仓已有 28 处同款噪声，不再增量）。
        text = getattr(r.content[0], "text", "")
        assert "ZERO_EXTERNAL_PRIOR_PRECISION_CAP" in text, "错误未指名 env"
        assert "非" in text and "client" in text, "错误未澄清归责方"


def test_ignition_beta_is_governance_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ignition_beta` 入治理白名单：client 经 config 传它被静默忽略。

    为什么它必须入白名单：非 None 即走**软门**分支，而软门下全部流（含 physio）一律进
    `fuse_terms`、无阈值筛除、也不施 D7 排除 ⇒ client 传它即可单边解除 D7 跨仓承诺。
    ⚠ 配套项目已确认其生产码从不传该字段，但那是「当前调用点的事实、不是结构保证」，
    故我方按结构收紧、**不依赖对方不传**。
    """
    from src.mcp_server.server import _MCP_GOVERNANCE_GATED_FLAGS, _build_session_config

    monkeypatch.delenv("ZERO_MCP_IGNITION_BETA", raising=False)
    assert "ignition_beta" in _MCP_GOVERNANCE_GATED_FLAGS
    assert _build_session_config({"ignition_beta": 20.0}).ignition_beta is None


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [(None, None), ("", None), ("   ", None), ("20", 20.0), ("0", 0.0)],
)
def test_ignition_beta_env_seeding(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None, expected: float | None
) -> None:
    """成对要求：进白名单**必须**同时给 env 入口，否则该字段在 MCP 路径永久取默认值（None）。

    撤掉 base 里那一行 ⇒ `"20"` 那格返回 None，本用例即红。
    ⚠ `"0"` 与未设**语义不同**（0.0 是软门、None 是硬门），故空串回落 None 而非 0.0。
    """
    from src.mcp_server.server import _build_session_config

    if env_value is None:
        monkeypatch.delenv("ZERO_MCP_IGNITION_BETA", raising=False)
    else:
        monkeypatch.setenv("ZERO_MCP_IGNITION_BETA", env_value)
    assert _build_session_config(None).ignition_beta == expected


def test_ignition_beta_bad_env_points_at_deploy_side(monkeypatch: pytest.MonkeyPatch) -> None:
    """坏值走 `ServerEnvError`（部署端归责），不被贴成「config 不合法」。"""
    from src.mcp_server.server import ServerEnvError, _build_session_config

    monkeypatch.setenv("ZERO_MCP_IGNITION_BETA", "soft")
    with pytest.raises(ServerEnvError, match="ZERO_MCP_IGNITION_BETA"):
        _build_session_config(None)


def test_mcp_facs_extended_not_governance_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    """特征化（**非**期望行为）：facs_extended 不在治理白名单，client 可经 config 单边覆写。

    已在 2026-07-29 跨仓回执 §4.1 向 Zero_MCP 声明「治理保证只覆盖 6 项、不含本字段」。
    若日后把它纳入 `_MCP_GOVERNANCE_GATED_FLAGS`（须先确认不摘除对方已用能力），本用例应变红，
    届时改成断言 override 被忽略——**不要靠放宽断言让它变绿**。
    """
    from src.mcp_server.server import _MCP_GOVERNANCE_GATED_FLAGS, _build_session_config

    monkeypatch.delenv("ZERO_FACS_EXTENDED", raising=False)
    assert "facs_extended" not in _MCP_GOVERNANCE_GATED_FLAGS
    assert _build_session_config({"facs_extended": True}).facs_extended is True


def test_maybe_expression_decoder_wires_prosody(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MCP 工厂 _maybe_expression_decoder：设 ZERO_PROSODY_MODEL_PATH → 注入真 ProsodyDecoder，
    prosody_scale 翻 normalized（与 chat 工厂同口径·独立于 FACS 门控）。torch 缺则跳过。
    """
    torch = pytest.importorskip("torch")
    from src.agents.models.composite import CompositeChannelDecoder
    from src.agents.models.prosody_decoder import ProsodyDecoder
    from src.mcp_server.server import _maybe_expression_decoder

    monkeypatch.delenv("ZERO_FACS_MODEL_PATH", raising=False)
    wpath = tmp_path / "prosody_decoder.pt"
    torch.save(ProsodyDecoder().state_dict(), wpath)
    monkeypatch.setenv("ZERO_PROSODY_MODEL_PATH", str(wpath))
    decoder = _maybe_expression_decoder()
    assert isinstance(decoder, CompositeChannelDecoder)
    assert isinstance(decoder.prosody_model, ProsodyDecoder)
    assert decoder.facs_model is None  # 独立门控：未设 FACS → facs 仍占位
    channels = decoder.predict_channels(0.5, 0.5)
    assert channels["prosody_scale"] == "normalized"
    assert all(0.0 <= v <= 1.0 for v in channels["prosody"].values())


async def test_step_returns_normalized_prosody_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """e2e：设 ZERO_PROSODY_MODEL_PATH → zero.step 返回 expression.prosody_scale == normalized、
    prosody 三值 ∈[0,1]（tag 经工具边界原样透传·T4）。torch 缺则跳过。
    """
    torch = pytest.importorskip("torch")
    from src.agents.models.prosody_decoder import ProsodyDecoder

    monkeypatch.delenv("ZERO_FACS_MODEL_PATH", raising=False)
    wpath = tmp_path / "prosody_decoder.pt"
    torch.save(ProsodyDecoder().state_dict(), wpath)
    monkeypatch.setenv("ZERO_PROSODY_MODEL_PATH", str(wpath))
    async with connect(build_server()) as client:
        await client.initialize()
        sid = json.loads((await client.call_tool("zero.open_session", {})).content[0].text)[
            "session_id"
        ]
        r = await client.call_tool(
            "zero.step",
            {"session_id": sid, "stim": {"valence": 0.4, "arousal": 0.5}},
        )
        assert r.isError is False
        expr = json.loads(r.content[0].text)
        assert expr["prosody_scale"] == "normalized"  # 顶层 hoist 经工具边界原样透传
        assert all(0.0 <= v <= 1.0 for v in expr["spontaneous"]["prosody"].values())


def test_maybe_expression_decoder_wires_physiology(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MCP 工厂 _maybe_expression_decoder：设 ZERO_PHYSIOLOGY_MODEL_PATH → 注入真
    PhysiologyDecoder，physiology 出 WESAD canonical {hr,sc(μS),temperature_c}（与 chat 工厂
    同口径·独立于 FACS/prosody 门控）。torch 缺则跳过。
    """
    torch = pytest.importorskip("torch")
    from src.agents.models.composite import CompositeChannelDecoder
    from src.agents.models.physiology_decoder import PhysiologyDecoder
    from src.mcp_server.server import _maybe_expression_decoder

    monkeypatch.delenv("ZERO_FACS_MODEL_PATH", raising=False)
    monkeypatch.delenv("ZERO_PROSODY_MODEL_PATH", raising=False)
    wpath = tmp_path / "physiology_decoder.pt"
    torch.save(PhysiologyDecoder().state_dict(), wpath)
    monkeypatch.setenv("ZERO_PHYSIOLOGY_MODEL_PATH", str(wpath))
    decoder = _maybe_expression_decoder()
    assert isinstance(decoder, CompositeChannelDecoder)
    assert isinstance(decoder.physiology_model, PhysiologyDecoder)
    assert decoder.facs_model is None  # 独立门控：未设 FACS → facs 仍占位
    physio = decoder.predict_channels(0.5, 0.5)["physiology"]
    assert set(physio) == {"heart_rate_bpm", "skin_conductance", "temperature_c"}
    assert 50.0 <= physio["heart_rate_bpm"] <= 120.0
    assert 0.0 <= physio["skin_conductance"] <= 20.0  # μS
    assert 30.0 <= physio["temperature_c"] <= 40.0  # °C


async def test_step_returns_wesad_physiology(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """e2e：设 ZERO_PHYSIOLOGY_MODEL_PATH → zero.step 返回 physiology 三值为 WESAD 真信号量纲
    （含 temperature_c、无 pupil_mm；经工具边界原样透传）。torch 缺则跳过。
    """
    torch = pytest.importorskip("torch")
    from src.agents.models.physiology_decoder import PhysiologyDecoder

    monkeypatch.delenv("ZERO_FACS_MODEL_PATH", raising=False)
    monkeypatch.delenv("ZERO_PROSODY_MODEL_PATH", raising=False)
    wpath = tmp_path / "physiology_decoder.pt"
    torch.save(PhysiologyDecoder().state_dict(), wpath)
    monkeypatch.setenv("ZERO_PHYSIOLOGY_MODEL_PATH", str(wpath))
    async with connect(build_server()) as client:
        await client.initialize()
        sid = json.loads((await client.call_tool("zero.open_session", {})).content[0].text)[
            "session_id"
        ]
        r = await client.call_tool(
            "zero.step",
            {"session_id": sid, "stim": {"valence": 0.4, "arousal": 0.5}},
        )
        assert r.isError is False
        physio = json.loads(r.content[0].text)["spontaneous"]["physiology"]
        assert set(physio) == {"heart_rate_bpm", "skin_conductance", "temperature_c"}
        assert 30.0 <= physio["temperature_c"] <= 40.0  # °C（真 WESAD 信号·非 pupil_mm 占位）


# ── HTTP（streamable-http）传输真端口往返（需 uvicorn；缺则跳过）──────────────────────


async def test_http_transport_roundtrip() -> None:
    """起 streamable-http server 于临时端口，streamablehttp_client 跑 open→step→close 全绿。

    验证 HTTP 传输真可服务（非仅骨架）；与 stdio 同一 build_server、同契约。uvicorn 缺则跳过。
    """
    pytest.importorskip("uvicorn")
    import asyncio
    import socket

    import uvicorn
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    # 持有已 bind 的 socket 直接交给 uvicorn（消除 bind→close→rebind 之间的端口竞态窗，W1）。
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

    app = build_server().streamable_http_app()
    uv = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    task = asyncio.create_task(uv.serve(sockets=[sock]))
    try:
        for _ in range(100):
            if uv.started:
                break
            await asyncio.sleep(0.05)
        assert uv.started, "uvicorn 未在 5s 内启动"
        async with streamable_http_client(f"http://127.0.0.1:{port}/mcp") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = {t.name for t in (await session.list_tools()).tools}
                assert names == {"zero.open_session", "zero.step", "zero.close_session"}
                sid = json.loads(
                    (await session.call_tool("zero.open_session", {})).content[0].text
                )["session_id"]
                r = await session.call_tool(
                    "zero.step", {"session_id": sid, "stim": {"valence": 0.6, "arousal": 0.5}}
                )
                assert r.isError is False
                assert "valence_arousal" in json.loads(r.content[0].text)
                r = await session.call_tool("zero.close_session", {"session_id": sid})
                assert json.loads(r.content[0].text) == {"ok": True}
    finally:
        uv.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5)  # tear-down 有界，CI 不因收尾卡死（W2）
        except TimeoutError:
            task.cancel()


# ── 议会解锁门治理旁路测试（A5·A6·2026-07-21）────────────────────────────────────────────
#
# 验证 _MCP_GOVERNANCE_GATED_FLAGS 堵住「生产关·MCP 开」旁路：
#   - client 经 config overrides 传 text_coping_enabled/coping_potential_enabled=True → 静默忽略
#   - env ZERO_MCP_TEXT_COPING_ENABLED=true → text_coping_enabled 生效（env 治理正路）
#   - 非门控字段 contagion_alpha override → 仍正常覆写（不误伤）
#   - 默认（无 env 无 override）→ 两门均 False（零回归）


def test_governance_default_both_gates_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """无 env 无 override → text_coping_enabled=False·coping_potential_enabled=False（零回归）。

    显式 delenv 两门 env，防 shell 预设 env 时假绿（pitfalls 第7条：不自动加载 .env）。
    """
    from src.mcp_server.server import _build_session_config

    monkeypatch.delenv("ZERO_MCP_COPING_ENABLED", raising=False)
    monkeypatch.delenv("ZERO_MCP_TEXT_COPING_ENABLED", raising=False)
    cfg = _build_session_config(None)
    assert cfg.text_coping_enabled is False
    assert cfg.coping_potential_enabled is False


def test_governance_override_text_coping_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """client override text_coping_enabled=True → 被治理静默忽略，config 仍 False（A5）。"""
    from src.mcp_server.server import _build_session_config

    # 确保 env 未设（不受已有 .env 污染）
    monkeypatch.delenv("ZERO_MCP_TEXT_COPING_ENABLED", raising=False)
    cfg = _build_session_config({"text_coping_enabled": True})
    assert cfg.text_coping_enabled is False


def test_governance_override_coping_potential_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """client override coping_potential_enabled=True → 被治理静默忽略，仍 False（A6 残洞）。"""
    from src.mcp_server.server import _build_session_config

    monkeypatch.delenv("ZERO_MCP_COPING_ENABLED", raising=False)
    cfg = _build_session_config({"coping_potential_enabled": True})
    assert cfg.coping_potential_enabled is False


def test_governance_env_text_coping_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """env ZERO_MCP_TEXT_COPING_ENABLED=true → text_coping_enabled=True（env 治理正路生效）。"""
    from src.mcp_server.server import _build_session_config

    monkeypatch.setenv("ZERO_MCP_TEXT_COPING_ENABLED", "true")
    cfg = _build_session_config(None)
    assert cfg.text_coping_enabled is True


def test_governance_env_coping_potential_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZERO_MCP_COPING_ENABLED=true → coping_potential_enabled=True。

    env 治理正路·与 test_governance_env_text_coping_enabled 对称；
    delenv text 门防串扰，确保结果仅由 ZERO_MCP_COPING_ENABLED 驱动。
    """
    from src.mcp_server.server import _build_session_config

    monkeypatch.setenv("ZERO_MCP_COPING_ENABLED", "true")
    monkeypatch.delenv("ZERO_MCP_TEXT_COPING_ENABLED", raising=False)
    cfg = _build_session_config(None)
    assert cfg.coping_potential_enabled is True


def test_governance_non_gated_override_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """非门控字段 contagion_alpha override → 正常覆写（治理过滤不误伤普通字段）。"""
    from src.mcp_server.server import _build_session_config

    monkeypatch.delenv("ZERO_MCP_TEXT_COPING_ENABLED", raising=False)
    monkeypatch.delenv("ZERO_MCP_COPING_ENABLED", raising=False)
    cfg = _build_session_config({"contagion_alpha": 0.15})
    assert cfg.contagion_alpha == pytest.approx(0.15)
    # 门控字段维持 False（未被 override 带进来）
    assert cfg.text_coping_enabled is False
    assert cfg.coping_potential_enabled is False


# ── T6 会话跨重启持久：resume-by-id / 幂等 / 机读标记 / close aclose ──────────────────


async def test_open_session_honors_client_session_id() -> None:
    """传 session_id → open_session 用作会话 id（不新铸 uuid），step 立即可用（T6 resume 入口）。"""
    async with connect(build_server()) as client:
        await client.initialize()
        r = await client.call_tool("zero.open_session", {"session_id": "cli-sid-1"})
        assert json.loads(r.content[0].text)["session_id"] == "cli-sid-1"
        rr = await client.call_tool(
            "zero.step", {"session_id": "cli-sid-1", "stim": {"valence": 0.3, "arousal": 0.4}}
        )
        assert rr.isError is False


async def test_open_session_empty_session_id_is_error() -> None:
    """传空/空白 session_id → 结构化 ToolError（非空字符串校验）。"""
    async with connect(build_server()) as client:
        await client.initialize()
        r = await client.call_tool("zero.open_session", {"session_id": "  "})
        assert r.isError is True
        assert "session_id" in r.content[0].text


async def test_resume_is_idempotent_for_active_session() -> None:
    """同进程内 open(session_id=X) 两次 → 返回同 id、活跃会话数不翻倍（幂等 re-attach）。"""
    from src.mcp_server.registry import SessionRegistry

    reg = SessionRegistry()
    async with connect(build_server(reg)) as client:
        await client.initialize()
        a = json.loads(
            (await client.call_tool("zero.open_session", {"session_id": "dup"})).content[0].text
        )["session_id"]
        b = json.loads(
            (await client.call_tool("zero.open_session", {"session_id": "dup"})).content[0].text
        )["session_id"]
        assert a == b == "dup"
        assert await reg.count() == 1  # 未重复登记（活跃期内断言·不依赖 transport 断开行为）


async def test_resume_preserves_governance_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """resume 也经 _build_session_config：client 传 gated 门控 override 被静默忽略。"""
    from src.mcp_server.registry import SessionRegistry

    monkeypatch.delenv("ZERO_MCP_COPING_ENABLED", raising=False)
    reg = SessionRegistry()
    async with connect(build_server(reg)) as client:
        await client.initialize()
        await client.call_tool(
            "zero.open_session",
            {"session_id": "gov", "config": {"coping_potential_enabled": True}},
        )
    session = await reg.get("gov")
    assert session is not None
    assert session.config.coping_potential_enabled is False  # gated：override 被忽略


async def test_close_session_aclose_no_error_both_backends(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """close_session 幂等关会话（memory/sqlite 后端）；close 后同 id step 回 unknown-session。"""
    for backend in ("memory", "sqlite"):
        if backend == "sqlite":
            pytest.importorskip("aiosqlite")
            monkeypatch.setenv("ZERO_CHECKPOINT_DB", str(tmp_path / f"{backend}.sqlite3"))
        monkeypatch.setenv("ZERO_CHECKPOINT_BACKEND", backend)
        async with connect(build_server()) as client:
            await client.initialize()
            sid = json.loads((await client.call_tool("zero.open_session", {})).content[0].text)[
                "session_id"
            ]
            c = await client.call_tool("zero.close_session", {"session_id": sid})
            assert json.loads(c.content[0].text) == {"ok": True}
            r = await client.call_tool(
                "zero.step", {"session_id": sid, "stim": {"valence": 0.0, "arousal": 0.0}}
            )
            assert r.isError is True and "unknown-session" in r.content[0].text


async def test_resume_persists_run_state_across_server_instances(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """sqlite 后端 + 同 session_id 跨 server 实例 resume（模拟重启·T6 最关键）。

    server1 step 落盘 → 新 checkpointer 同 DB 读回 → server2 resume 续 step 不报错。
    """
    pytest.importorskip("langgraph.checkpoint.sqlite.aio")
    pytest.importorskip("aiosqlite")
    from src.memory.client import MemoryClient
    from src.orchestration.graph import build_graph
    from src.orchestration.runner import ALLOWED_CHECKPOINT_TYPES
    from src.storage.checkpointer import build_checkpointer

    monkeypatch.setenv("ZERO_CHECKPOINT_BACKEND", "sqlite")
    monkeypatch.setenv("ZERO_CHECKPOINT_DB", str(tmp_path / "resume.sqlite3"))
    sid = "resume-persist"
    stim = {"valence": -0.5, "arousal": 0.6}

    # server1：绑 sid 建会话、step 数轮 → 运行态按 thread_id=sid 落 sqlite
    async with connect(build_server()) as c1:
        await c1.initialize()
        await c1.call_tool("zero.open_session", {"session_id": sid})
        for _ in range(3):
            rr = await c1.call_tool("zero.step", {"session_id": sid, "stim": stim})
            assert rr.isError is False
        await c1.call_tool("zero.close_session", {"session_id": sid})

    # 落盘校验：新 checkpointer 同 DB 能按 thread_id 读回 sid 的 checkpoint（运行态真持久）
    saver = build_checkpointer(ALLOWED_CHECKPOINT_TYPES)
    state = await build_graph(saver, MemoryClient()).aget_state(
        {"configurable": {"thread_id": sid}}
    )
    conn = getattr(saver, "conn", None)
    if conn is not None:
        await conn.close()  # 关检查连接（避免泄漏 warning）
    assert state is not None and state.values, "zero.step 应把运行态按 client session_id 落 sqlite"
    ve = state.values.get("value_estimate")
    assert isinstance(ve, float) and ve != 0.0, (
        f"3 步 loss 应让 value_estimate 累积漂移（证明续的是落盘运行态·非全新会话），实际 {ve}"
    )

    # server2（新 registry·同 DB·模拟重启）：resume 同 sid → 续 step 不报错
    async with connect(build_server()) as c2:
        await c2.initialize()
        r2 = await c2.call_tool("zero.open_session", {"session_id": sid})
        assert json.loads(r2.content[0].text)["session_id"] == sid
        rr2 = await c2.call_tool("zero.step", {"session_id": sid, "stim": stim})
        assert rr2.isError is False
        await c2.call_tool("zero.close_session", {"session_id": sid})  # 关连接（aclose·免泄漏）
