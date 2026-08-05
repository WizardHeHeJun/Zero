"""zero-link MCP server 契约测试：映射纯函数 + FastMCP in-memory client 往返 + 错误路径。

`mcp` 走 optional-deps（默认不入 core），无 mcp 的机器整包不可导入 → `importorskip` 跳过（零回归）。
断言对齐 MCP 侧 client 期望契约（open→{session_id} / step→expression 子 dict / close→{ok}；
错误→isError=true）。
"""

from __future__ import annotations

import ast  # 结构性守卫（ToolError / cfg 回显 / stateless_http 双重 pin / 传输同源）用
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
        # 全等断言（非子集）：新增或删除工具都会红，逼迫改动方来这里确认对外面变了。
        assert names == {
            "zero.open_session",
            "zero.step",
            "zero.close_session",
            "zero.describe_config",
            "zero.purge_session",
            "zero.motion",  # 动作通道（2026-08-05）：独立拉取，不参与 step 节奏
        }

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
    """未知 sid → `[zero:unknown-session]`（T6·MCP graceful_step 据此降级重开）。

    ⚠ 收紧记录：原先断的是 `"unknown-session" in text` 子串 + `"session_id" in text`。
    前者对「令牌退回裸前缀」这种真 bug 恒绿（那正是 7632cc0 修掉的事故）；后者更糟——
    工具入参名本身就叫 session_id，几乎不可能红，是一条不可证伪的断言。
    要保留「文案指明是哪个 id」的语义，就改断**被回显的那个 id**。
    """
    from src.mcp_server.server import ZERO_ERROR_CODE_UNKNOWN_SESSION

    async with connect(build_server()) as client:
        await client.initialize()
        r = await client.call_tool(
            "zero.step", {"session_id": "does-not-exist", "stim": {"valence": 0.0, "arousal": 0.0}}
        )
        assert r.isError is True
        text = getattr(r.content[0], "text", "")
        assert _wire_code(text) == ZERO_ERROR_CODE_UNKNOWN_SESSION
        assert "does-not-exist" in text, "文案须回显是哪个 id 未知"


async def test_step_bad_external_prior_is_error() -> None:
    """精度 9.9 > cap 0.8 → M3 fail-fast → `[zero:external-prior-invalid]`。

    这是 `external-prior-invalid` 在 wire 层的**唯一**覆盖点，且它压住 `build_server.step` 里
    `except ExternalPriorError` / `except ValueError` 的分治——锁的是**归责语义**
    （指向 client 传参 vs 指向会话配置组合），不是实现细节。
    ⚠ 原先函数体只有 `assert r.isError is True`：载荷映射崩、会话没开、内核崩、归责错位
    一律通过。日后若这条变红，说明归责口径变了，须回设计门确认，**不要靠放宽断言让它变绿**。
    """
    from src.mcp_server.server import ZERO_ERROR_CODE_EXTERNAL_PRIOR_INVALID

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
        code = _wire_code(getattr(r.content[0], "text", ""))
        assert code == ZERO_ERROR_CODE_EXTERNAL_PRIOR_INVALID, f"归责错位：实得 {code}"


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
        (
            # contagion_alpha 有 `Field(ge=0.0, le=0.3)` 且**不在** _MCP_GOVERNANCE_GATED_FLAGS
            # （在名单里的字段会被 _build_session_config 静默丢弃，根本触发不了构造期校验）。
            ("zero.open_session", {"config": {"contagion_alpha": 9.9}}),
            "config-invalid",
        ),
    ],
)
async def test_error_codes_are_extractable(call: tuple[str, dict], expected_code: str) -> None:
    """各错误出口都带得上令牌、码值取自登记表，且令牌在 **wire 文本**上全文恰一次。

    参数表只收「一次 `call_tool` 即可触发」的出口；需要先开会话、先 monkeypatch env、
    先预置 registry 或先关连接的出口各自独立成例——塞进这个纯 `(call, expected_code)` 壳子
    会把它变成带一堆可选 setup 钩子的怪物。验收按**出口**清点（wire 上 8 个可达出口），
    **不是**按 8 个码：`payload-invalid` 有两个出口、`unknown-session` 也有两个。

    与 `::test_every_registered_code_is_extractable_by_consumer_regex` **分工不同、别删其一**：
    那条在**未加壳**文本上逐码断（表驱动，是 `timeout-step` 这类端到端够不着的码的唯一覆盖），
    本条在 **wire** 文本上逐出口断（覆盖 FastMCP 加壳后的真实形态）。
    ⚠ `count == 1` 只在该出口**回显含毒载荷**时才有判别力；本表三格都不回显，
    这条的判别力由 `::test_wire_token_stays_exactly_once_with_poisoned_payload` 承担。
    """
    from src.mcp_server.server import ZERO_ERROR_CODES

    name, args = call
    async with connect(build_server()) as client:
        await client.initialize()
        r = await client.call_tool(name, args)
        assert r.isError is True
        text = getattr(r.content[0], "text", "")
        assert text.count("[zero:") == 1, f"wire 文本上令牌须全文恰一次，实得 {text!r}"
        code = _wire_code(text)
        assert code == expected_code
        assert code in ZERO_ERROR_CODES


async def test_deploy_env_error_carries_its_own_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """部署端 env 坏值的码与 client-config 坏值**必须不同**——归责靠它区分。"""
    monkeypatch.setenv("ZERO_EXTERNAL_PRIOR_PRECISION_CAP", "0.8x")
    async with connect(build_server()) as client:
        await client.initialize()
        r = await client.call_tool("zero.open_session", {})
        text = getattr(r.content[0], "text", "")
        assert text.count("[zero:") == 1, f"wire 文本上令牌须全文恰一次，实得 {text!r}"
        assert _wire_code(text) == "deploy-env-invalid"


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


def test_every_registered_code_is_extractable_by_consumer_regex() -> None:
    """🛑 表驱动：`ZERO_ERROR_CODES` 里**每一个**码都必须能被消费方正则原样提取。

    今天全仓没有任何用例遍历这张表——往表里加 `"Config-Invalid"` / `"config_invalid"`
    这种大写/下划线码，1300+ 用例照样全绿，而消费方的 `re.search` 对它取不到，
    或更坏：`"foo]bar"` 这种码 `search` 会成功但 `group(1)=="foo"`，
    消费方拿**错的码**去查表比拿不到更坏。本条补的就是那条缝。

    与 `::test_error_codes_are_extractable` **分工不同、不得删其一**：本条断的是
    `str(_tool_error(...))` 的**未加壳**文本，是 `timeout-step` 这类「只登记不产出」、
    端到端根本够不着的码的**唯一**可能覆盖；那条断 wire 上加壳后的真实形态、按出口清点。

    三条形制约束：① 复用 `_wire_code`，不另抄一份正则（两处真源必然漂移）；
    ② 断 `== code` 而非 `is not None`（防截断式假绿）；
    ③ `sorted()` 不是可选——frozenset 迭代序随 PYTHONHASHSEED 变，失败信息不可复现。
    """
    from src.mcp_server.server import ZERO_ERROR_CODES, _tool_error

    # helper 自证：防日后有人给 `_wire_code` 加 IGNORECASE / 放宽字符集，把本条 +
    # ::test_error_token_survives_fastmcp_wrapping + ::test_error_codes_are_extractable +
    # ::test_step_lock_timeout_yields_timeout_code 的判别力一次性悄悄抽走。
    assert _wire_code("[zero:Foo] x") is None, "_wire_code 已放宽到接受大写码 → 判别力被抽走"
    assert _wire_code("[zero:a_b] x") is None, "_wire_code 已放宽到接受下划线 → 判别力被抽走"

    offenders: list[tuple[str, str | None, int]] = []
    for code in sorted(ZERO_ERROR_CODES):
        text = str(_tool_error(code, "占位文案"))
        extracted = _wire_code(text)
        token_count = text.count("[zero:")
        if extracted != code or token_count != 1:
            offenders.append((code, extracted, token_count))
    assert not offenders, (
        f"这些码经 _tool_error 产出后无法被消费方正则原样提取"
        f"（码, 提取结果, 令牌数）：{offenders}；"
        "码值须是 ASCII kebab-case（首字符小写字母、其后 [a-z0-9-]），且令牌全文恰出现一次"
    )


async def test_wire_token_stays_exactly_once_with_poisoned_payload() -> None:
    """🛑 「wire 全文恰一次」的**可证伪性**由本条承担：真喂一个含 `[zero:` 的载荷。

    含毒载荷确实会被回显：`mapping.stimulus_from_payload` 的缺键分支把 `sorted(stim)`
    原样写进 ValueError 文案 ⇒ 若 `_tool_error` 不做净化，wire 上会出现**两个** `[zero:`，
    消费方 `re.search` 会取到 client 自己伪造的那个、且不报歧义。
    净化把回显里的 `[zero:` 换成 `(zero:`，故 wire 上仍恰一次。

    ⚠ 别改用 `open_session(session_id="[zero:fake]")`：该值 `.strip()` 后非空、过得了校验、
    根本不进 payload-invalid 出口——那条路上本用例会变成不可证伪的绿灯。
    """
    from src.mcp_server.server import ZERO_ERROR_CODE_PAYLOAD_INVALID

    async with connect(build_server()) as client:
        await client.initialize()
        sid = json.loads(
            getattr((await client.call_tool("zero.open_session", {})).content[0], "text", "")
        )["session_id"]
        r = await client.call_tool("zero.step", {"session_id": sid, "stim": {"[zero:fake]": 1}})
        assert r.isError is True
        # 把「content[0] 就是全文」这个前提显式化：今天由 MCP SDK 的错误结果构造硬编码成
        # 单元素列表保证，SDK 改成多段就红——那时 count 断言必须改成跨段清点。
        assert len(r.content) == 1
        text = getattr(r.content[0], "text", "")
        # 先断契约、再断判别力自证：顺序反了的话，撤掉净化时报的会是「载荷未被回显」，
        # 把「净化没了」误诊成「用例失效」。
        assert text.count("[zero:") == 1, f"净化失效：wire 上出现了第二个令牌——{text!r}"
        assert "(zero:fake" in text, "含毒载荷未被回显 → 本用例失去判别力，须换一条真回显的出口"
        assert _wire_code(text) == ZERO_ERROR_CODE_PAYLOAD_INVALID


# ── 结构性守卫 + 锁获取超时 ───────────────────────────────────────────────


def _toolerror_bound_names(source: str) -> set[str]:
    """解析出 `ToolError` 在这段源码里的**全部绑定名**（含 `as` 别名）。

    `from ... import ToolError as TE` 之后 `raise TE(...)` 一样是裸构造——按字面量
    `"ToolError"` 匹配的守卫会整条放过它。故绑定名必须从模块自身解析，不能写死。
    """
    import ast

    return {
        alias.asname or alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name == "ToolError"
    }


def _toolerror_sites(source: str, filename: str) -> list[int]:
    """扫一段源码，返回**构造** `ToolError` 的行号（三处合法位除外）。

    判据是**构造点**而非 `raise` 点：任何 `raise ToolError(...)` 也是一次构造，故严格覆盖
    旧的 raise 点判据，同时堵住旧判据四个 `and` 各自漏掉的那种写法：
      · `raise TE(...)`（别名，旧判据比的是 `func.id == "ToolError"`）；
      · `raise exceptions.ToolError(...)`（旧判据要求 `func` 是 `ast.Name`，实为 `ast.Attribute`）；
      · `err = ToolError(...); raise err`（旧判据要求 `node.exc` 是 `ast.Call`）；
      · `raise _mk_err()`（构造点在别处，旧判据只看 raise 处）。

    白名单**只三处**：`_tool_error` 的整个 FunctionDef 子树（一并覆盖 `-> ToolError` 注解与
    `return ToolError(...)`，且**逐文件**判——别的模块里出现同名函数不该白拿豁免）；
    `ast.ExceptHandler.type` 子树（`close_session` 的 `except ToolError:` 需要它，
    🛑 只放 `type` 不放 handler body，否则 `except ToolError as e: raise ToolError(...)` 会溜走）；
    import 语句本身（`ast.alias` 不产生 Name/Attribute 节点，天然不落网）。

    ⚠ 动态构造（`getattr(exceptions, "ToolError")(...)` / `type(e)(...)`）**明确不堵**：
    结构性测试打不过存心绕过的人，它的目标是无心之失。把边界写清楚比留半吊子检测更有价值，
    也防后人「顺手加强」成一个自己都说不清覆盖面的东西。
    """
    import ast

    bound = _toolerror_bound_names(source)
    tree = ast.parse(source, filename=filename)
    exempt: set[int] = set()
    for node in ast.walk(tree):
        subtree: ast.AST | None = None
        if isinstance(node, ast.FunctionDef) and node.name == "_tool_error":
            subtree = node if filename == "server.py" else None
        elif isinstance(node, ast.ExceptHandler) and node.type is not None:
            subtree = node.type
        if subtree is not None:
            exempt.update(id(n) for n in ast.walk(subtree))
    return sorted(
        {
            node.lineno
            for node in ast.walk(tree)
            if id(node) not in exempt
            and (
                (isinstance(node, ast.Name) and node.id in bound)
                or (isinstance(node, ast.Attribute) and node.attr == "ToolError")
            )
        }
    )


def test_toolerror_guard_catches_the_four_bypasses() -> None:
    """守卫自身的判别力自证：喂合成源码，四种绕行必须全抓、三处合法位必须不抓。

    这让判别力不依赖任何手工变异步骤——旧守卫的四个 `and` 各是一道缝，与这四种绕行一一对应。
    """
    bypasses = (
        "from mcp.server.fastmcp.exceptions import ToolError\n"
        "from mcp.server.fastmcp.exceptions import ToolError as TE\n"
        "from mcp.server.fastmcp import exceptions\n"
        "\n"
        "def _mk_err():\n"
        "    return ToolError('d')\n"
        "\n"
        "def a():\n"
        "    raise TE('a')\n"
        "\n"
        "def b():\n"
        "    raise exceptions.ToolError('b')\n"
        "\n"
        "def c():\n"
        "    err = ToolError('c')\n"
        "    raise err\n"
        "\n"
        "def d():\n"
        "    raise _mk_err()\n"
    )
    lines = bypasses.splitlines()
    caught = {lines[n - 1].strip() for n in _toolerror_sites(bypasses, "server.py")}
    assert caught == {
        "return ToolError('d')",  # ④ raise _mk_err() 的构造点
        "raise TE('a')",  # ① 别名
        "raise exceptions.ToolError('b')",  # ② 属性访问
        "err = ToolError('c')",  # ③ 先构造后 raise
    }, f"有绕行未被抓住，实得 {caught}"

    legal = (
        "from mcp.server.fastmcp.exceptions import ToolError\n"
        "\n"
        "def _tool_error(code: str, message: str) -> ToolError:\n"
        "    return ToolError(f'[zero:{code}] {message}')\n"
        "\n"
        "def close():\n"
        "    try:\n"
        "        pass\n"
        "    except ToolError:\n"
        "        pass\n"
    )
    assert _toolerror_sites(legal, "server.py") == [], "三处合法位被误报"
    # 豁免逐文件判：同一段源码换个文件名，`_tool_error` 不再被豁免（注解 + 构造两行）；
    # 而 `except` 子句的 type 在任何文件里都合法。
    assert _toolerror_sites(legal, "tools.py") == [3, 4], "_tool_error 豁免不该跨文件生效"
    reraise = (
        "from mcp.server.fastmcp.exceptions import ToolError\n"
        "\n"
        "def close():\n"
        "    try:\n"
        "        pass\n"
        "    except ToolError as e:\n"
        "        raise ToolError(str(e))\n"
    )
    assert _toolerror_sites(reraise, "server.py") == [7], (
        "白名单放行了 handler body —— 只能放 handler.type"
    )


def test_toolerror_is_constructed_only_inside_the_helper() -> None:
    """🛑 AST 级：`src/mcp_server/` 里**不得**有绕过 `_tool_error` 的 `ToolError(...)` 构造。

    配套项目 2026-07-29 指出：它按 `re.search` 取**首个**令牌，若某条 raise 绕过 `_tool_error`，
    ① 该错误没有码、② 回显载荷里的 `[zero:` 不会被净化 ⇒ 它会取到用户输入伪造的令牌而不报歧义。
    `::test_error_token_appears_exactly_once` 只覆盖 `_tool_error` **自身**，挡不住新增的裸构造。

    本条**替换**（而非并存于）旧的「raise 点」守卫：构造点判据严格覆盖旧断言，并存只会留下
    两处要同步维护的匹配逻辑。判据本身的判别力由
    `::test_toolerror_guard_catches_the_four_bypasses` 自证。

    扫描面是整个 `src/mcp_server/`（非递归 glob，天然不碰 `__pycache__`）而非单个 `server.py`：
    今天扩面抓不到新东西，收益 100% 在防未来漂移——`server.py` 已 700+ 行 5 个工具，拆出
    `tools.py` 是可预期的；`mapping.py` 现抛裸 `ValueError` 且文案回显载荷，若被「就地升级」成
    直接抛 ToolError，会同时丢掉机读码登记与 `[zero:` 净化，正是本守卫存在的全部理由。
    """
    import src.mcp_server.server as srv

    files = sorted(Path(srv.__file__).parent.glob("*.py"))
    assert files, "扫描面为空 → 守卫恒真空绿，glob 写错了"
    assert any(p.name == "server.py" for p in files), "server.py 不在扫描面内 → 守卫失去主要抓手"

    # 自检：上游一改 import，offenders 恒空 ⇒ 永久真空绿（仓内「绿灯必须先证明能红」的同型）。
    # 改成 `import ToolError as TE` 时 `alias.name` 仍是 ToolError，故本条仍绿、别名的构造点照抓。
    assert _toolerror_bound_names(Path(srv.__file__).read_text(encoding="utf-8")), (
        "server.py 已不再 from … import ToolError → 本守卫失去抓手，须重写而不是默认绿"
    )

    offenders: list[str] = []
    for path in files:
        # 🛑 必须显式 utf-8：本仓在 Windows 上默认 cp936，满是中文注释的文件会 UnicodeDecodeError，
        # 表现为「测试报错」而非「报 offender」，极易被误判成守卫写坏而放弃扩面。
        source = path.read_text(encoding="utf-8")
        offenders += [f"{path.name}:{line}" for line in _toolerror_sites(source, path.name)]
    assert not offenders, (
        f"这些位置直接构造了 ToolError：{offenders}；请改用 _tool_error(code, msg)——"
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
    """锁被占住时，排队的 step 超时 → `[zero:timeout-lock]`，且**未进入内核**。

    ⚠ 码按**符号名** pin（`server.py` 机读错误码段注释的明文要求），不 pin 字面量。
    阴性对照义务：本条的值 `0.05` 是合法值，必须仍报 `timeout-lock` 而**非**
    `deploy-env-invalid` —— 它同时是 `::test_step_lock_timeout_rejects_non_positive_or_nonfinite`
    没有误伤合法区间的证据。
    """
    import asyncio

    from src.mcp_server.registry import SessionRegistry
    from src.mcp_server.server import ZERO_ERROR_CODE_TIMEOUT_LOCK

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
            assert text.count("[zero:") == 1, f"wire 文本上令牌须全文恰一次，实得 {text!r}"
            assert _wire_code(text) == ZERO_ERROR_CODE_TIMEOUT_LOCK
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
    """未设 env = 无限等待 = 逐字旧行为（零回归）；空串同样回落无限等待。

    ⚠ 必须带 `positive_finite=True` 调用——那是 `_acquire_with_timeout` 的**生产调用形态**。
    留着单参调用等于指着一个生产不再那样用的形态断言，绿灯从此没有意义
    （仓内「锚点要绑生产调用点」的同型教训）。
    """
    from src.mcp_server.server import _env_optional_float

    for value in (None, "", "  "):
        if value is None:
            monkeypatch.delenv("ZERO_MCP_STEP_LOCK_TIMEOUT", raising=False)
        else:
            monkeypatch.setenv("ZERO_MCP_STEP_LOCK_TIMEOUT", value)
        assert _env_optional_float("ZERO_MCP_STEP_LOCK_TIMEOUT", positive_finite=True) is None


@pytest.mark.parametrize(
    "bad_value", ["0", "-1", "-0.5", "nan", "NaN", "inf", "-inf", "Infinity", "1e400"]
)
async def test_step_lock_timeout_rejects_non_positive_or_nonfinite(
    monkeypatch: pytest.MonkeyPatch, bad_value: str
) -> None:
    """`ZERO_MCP_STEP_LOCK_TIMEOUT` 的非正/非有限值必须**读取即拒**，且带部署端归责码。

    为什么不能放过：`asyncio.wait_for(coro, timeout)` 在 `timeout <= 0` 时走快路径，
    协程根本没机会执行 ⇒ **锁空闲时也照样超时**，即每一次 `zero.step` 都无条件返回
    `timeout-lock`，而那条文案写的是「上一轮 step 仍在执行……可原样重试」——两句都是假的。
    `nan` 效果等同 0（到期判据 `nan >= end_time` 恒 False）；`inf` 与未设等价，
    多留一个常驻 TimerHandle，同样不该是一个有意配置。要「不设超时」请留空。

    ⚠ 判据打在**解析后的 float** 上，故 `"Infinity"` / `"1e400"`（→inf）这类写法也必须被拒——
    字符串黑名单挡不住它们。
    ⚠ 用 `_wire_code` 而**不是**子串：裸子串对「令牌丢了但文案还在」的退化恒绿。
    """
    from src.mcp_server.server import ZERO_ERROR_CODE_DEPLOY_ENV_INVALID

    monkeypatch.setenv("ZERO_MCP_STEP_LOCK_TIMEOUT", bad_value)
    async with connect(build_server()) as client:
        await client.initialize()
        sid = json.loads(
            getattr((await client.call_tool("zero.open_session", {})).content[0], "text", "")
        )["session_id"]
        r = await client.call_tool(
            "zero.step", {"session_id": sid, "stim": {"valence": 0.0, "arousal": 0.0}}
        )
        assert r.isError is True
        text = getattr(r.content[0], "text", "")
        assert _wire_code(text) == ZERO_ERROR_CODE_DEPLOY_ENV_INVALID, (
            f"坏 env 未在 step 通路上转成带码 ToolError，wire 文本={text!r}"
        )
        assert "ZERO_MCP_STEP_LOCK_TIMEOUT" in text, "错误未指名是哪个 env"


def test_step_lock_and_ignition_beta_have_different_legal_domains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """🛑 同一个函数、同一个输入 `"0"`、两种结论——把「值域校验不得误伤」固化成测试。

    `_env_optional_float` 有两个使用者，合法域**不同**：
      · `ZERO_MCP_IGNITION_BETA`：`0.0` 是语义合法的软门陡度（gate ≡ 0.5 均匀融合，
        退化但可运行），与未设（硬门 `None`）语义不同；
      · `ZERO_MCP_STEP_LOCK_TIMEOUT`：`0` 会让锁空闲时也无条件超时。
    故值域判据必须 **opt-in**。若日后有人把它改成本函数的无条件行为，本条与
    `::test_ignition_beta_env_seeding[0-0.0]` 会同时红。
    """
    from src.mcp_server.server import ServerEnvError, _env_optional_float

    monkeypatch.setenv("ZERO_MCP_IGNITION_BETA", "0")
    monkeypatch.setenv("ZERO_MCP_STEP_LOCK_TIMEOUT", "0")
    assert _env_optional_float("ZERO_MCP_IGNITION_BETA") == 0.0
    with pytest.raises(ServerEnvError, match="ZERO_MCP_STEP_LOCK_TIMEOUT"):
        _env_optional_float("ZERO_MCP_STEP_LOCK_TIMEOUT", positive_finite=True)


# ── 取消留下的半截运行态：检测与回显 ─────────────────────────────────────


async def test_interrupted_at_is_none_on_clean_session() -> None:
    """跑完整轮后 `next` 恒为空 ⇒ `interrupted_at()` 返回 None（负对照）。

    没有这条，下面那条阳性用例无法排除「该方法恒返回非 None」。
    """
    from src.orchestration.runner import ConversationSession, SessionConfig
    from src.orchestration.state import Stimulus

    session = ConversationSession(thread_id="clean-t1", user_id="clean-t1", config=SessionConfig())
    try:
        assert await session.interrupted_at() is None, "全新会话不该被判为中断"
        await session.step(Stimulus(name="s", goal_congruence=0.4, intensity=0.6))
        assert await session.interrupted_at() is None, "跑完整轮后 next 应为空"
    finally:
        await session.aclose()


async def test_interrupted_at_detects_a_real_half_turn() -> None:
    """🛑 判别力核心：**真造一个半截态**，确认检测得出来。

    造法不是伪造 checkpoint，而是**真的取消一次在飞的 ainvoke**——这正是生产里
    stdio 关 stdin 会发生的事（2026-07-29 实证：+0.015s 取消）。
    取消点选在图跑到一半时，此时已完成 super-step 的 checkpoint 已落盘、`next` 非空。

    若 `interrupted_at()` 改成恒 None、或判据从 `next` 换成别的东西，本用例即红。
    """
    import asyncio

    from src.orchestration.runner import ConversationSession, SessionConfig
    from src.orchestration.state import Stimulus

    session = ConversationSession(thread_id="half-t1", user_id="half-t1", config=SessionConfig())
    try:
        task = asyncio.ensure_future(
            session.step(Stimulus(name="s", goal_congruence=0.4, intensity=0.6))
        )
        # 让图跑起来但不跑完：让出若干次事件循环后取消。
        for _ in range(3):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        nxt = await session.interrupted_at()
        if nxt is None:
            pytest.skip(
                "本机上该图跑得太快、取消未落在 super-step 之间——本用例此次未构造出目标态。"
                "⚠ 这不是「没有半截态」的证据（跨仓件已实证它存在），只是本次没抓到"
            )
        assert isinstance(nxt, tuple) and nxt, "检测到中断时应返回非空的待执行节点元组"
    finally:
        await session.aclose()


async def test_interrupt_probe_is_explicit_four_state() -> None:
    """🛑 缺席不得承载多义：`interrupt_probe` **恒存在**且四态可区分。

    旧口径把「未探测」「探测成功且干净」「探测失败」**都表示成 interrupted_at 缺席**，
    消费方只能一律解释成「可以安全续跑」。要害（配套项目 2026-07-30 §5-1 指出）：
    **「探测失败」与要防的半截态是故障相关的** —— 探测读的正是那份可能半写的 checkpoint
    ⇒ 越是真出事的时候越可能探测失败 ⇒ **止血在最该生效时静默失效，且无可区分信号**。
    """
    async with connect(build_server()) as client:
        await client.initialize()
        fresh = json.loads(
            getattr((await client.call_tool("zero.open_session", {})).content[0], "text", "")
        )
        assert fresh["interrupt_probe"] == "not_probed", "新建会话未探测"

        sid = fresh["session_id"]
        active = json.loads(
            getattr(
                (await client.call_tool("zero.open_session", {"session_id": sid})).content[0],
                "text",
                "",
            )
        )
        # 活跃幂等重开也是一条缺席路径——对方指出旧注释漏了它
        assert active["interrupt_probe"] == "not_probed"
        assert active["resumed"] is True


async def test_probe_failure_is_distinguishable_from_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """🛑 本组判别力核心：探测**抛异常**时必须回 `probe_failed`，**不得**与 `clean` 同形。

    造法：让 `interrupted_at()` 抛，再走 resume-不活跃分支。
    若实现退回旧口径（宽 except 后什么都不标），这一格会与 clean 一样表现为「无 interrupted_at」
    ⇒ 本用例即红。这正是「最该报警时不报警」那条缝。
    """
    from src.mcp_server.registry import SessionRegistry
    from src.orchestration.runner import ConversationSession

    async def _boom(self: object) -> None:
        raise RuntimeError("checkpoint 半写，读不出来")

    monkeypatch.setattr(ConversationSession, "interrupted_at", _boom)
    registry = SessionRegistry()
    async with connect(build_server(registry=registry)) as client:
        await client.initialize()
        # 直接用一个未在册的 id 走 resume 分支（不活跃 ⇒ 会探测）
        r = json.loads(
            getattr(
                (await client.call_tool("zero.open_session", {"session_id": "resume-me"})).content[
                    0
                ],
                "text",
                "",
            )
        )
        assert r["resumed"] is True
        assert r["interrupt_probe"] == "probe_failed", (
            "探测失败被当成了 clean —— 消费方会据此认为可以安全续跑"
        )
        assert "interrupted_at" not in r, "探测失败时不得伪造 interrupted_at"


async def test_describe_config_version_bumped_with_contract_change() -> None:
    """契约版本随本轮变化 bump —— 纪律覆盖值域/语义变化，不只增删键。"""
    from src.mcp_server.server import DESCRIBE_CONFIG_VERSION

    # ⚠ 下界必须随每轮 bump 一起收紧，否则本断言不是 tripwire、只是注释：
    # 实测把 DESCRIBE_CONFIG_VERSION 改回 2 时，旧的 `>= 2` 下界让**全套 1850 条全绿**。
    assert DESCRIBE_CONFIG_VERSION >= 4, (
        "本轮 describe_config 新增 memory_store_impl / semantic_store_impl / checkpointer_impl "
        "三键（契约变化），须 bump 到 ≥4；上一轮的理由是新增 transport / stateless_http 两键"
    )
    async with connect(build_server()) as client:
        await client.initialize()
        d = json.loads(
            getattr((await client.call_tool("zero.describe_config", {})).content[0], "text", "")
        )
        assert d["describe_config_version"] == DESCRIBE_CONFIG_VERSION


async def test_open_session_reports_resumed_flag() -> None:
    """`open_session` 回显 `resumed`：新建 False、按 id 重开 True。

    该键是配套项目在观察期唯一能事后判断「本次是否走了 resume 分支」的观测量。
    返回体只增不改——其解析按「容忍额外键、缺键即回落」，故对现网零回归。
    """
    async with connect(build_server()) as client:
        await client.initialize()
        fresh = json.loads(
            getattr((await client.call_tool("zero.open_session", {})).content[0], "text", "")
        )
        assert fresh["resumed"] is False
        again = json.loads(
            getattr(
                (
                    await client.call_tool("zero.open_session", {"session_id": fresh["session_id"]})
                ).content[0],
                "text",
                "",
            )
        )
        assert again["session_id"] == fresh["session_id"]
        assert again["resumed"] is True


def test_active_resume_branch_does_not_echo_freshly_built_cfg() -> None:
    """🛑 AST：活跃 resume 分支的 return 里**不得**出现 `cfg`。

    配套项目 R11 的实现警告：该分支的门控是构造时固定的，回显刚算出的 `cfg` 等于回显
    「本可生效但实际未生效」的值——**比不回显更危险**（消费方会拿它当真、比对通过、
    实则语义已分叉）。要回显只能从 session 对象取。
    本用例钉住这条，防日后「顺手把 cfg 加进去」。
    """
    import ast
    import inspect

    import src.mcp_server.server as srv

    tree = ast.parse(inspect.getsource(srv))
    # ⚠ 必须**收窄到 open_session**：`describe_config` 的 return 里出现 `cfg` 是合法的——
    # 那里的 cfg 取自 `session.config`（会话真实生效值）或部署端默认，正是它该回的东西。
    # 第一版守卫扫全模块，加了 describe_config 后当场假红——判据要盯**语义位置**而非文本出现。
    targets = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "open_session"
    ]
    assert targets, "未找到 open_session——本守卫已失去锚点，请重写而不是删掉"
    for node in ast.walk(targets[0]):
        if not (isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)):
            continue
        names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
        assert "cfg" not in names, (
            "open_session 的 return 字典里出现了 cfg——活跃 resume 分支回显它会造成语义分叉"
        )


# ── close 饿死 与 归责错位（两条同批·修前者会制造后者）────────────────────


async def test_close_session_not_starved_by_inflight_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """🛑 一次挂起的 step 持锁时，`close_session` **不得**被无限饿死。

    配套项目原话：「这条我方无法自救」——client 侧超时不发取消通知，锁只能我方自解。
    修法是**先摘牌、再带超时取锁**：摘牌只持 registry 表级锁，不会被在飞 step 卡住。
    改回旧的裸 `async with lock:` 包住两步，本用例会挂到 wait_for 超时而失败。
    """
    import asyncio

    from src.mcp_server.registry import SessionRegistry

    monkeypatch.setenv("ZERO_MCP_STEP_LOCK_TIMEOUT", "0.05")
    registry = SessionRegistry()
    async with connect(build_server(registry=registry)) as client:
        await client.initialize()
        opened = await client.call_tool("zero.open_session", {})
        sid = json.loads(getattr(opened.content[0], "text", ""))["session_id"]
        _, lock = await registry.acquire(sid)
        assert lock is not None
        await lock.acquire()  # 模拟「一次挂起的 step 永久持锁」
        try:
            r = await asyncio.wait_for(
                client.call_tool("zero.close_session", {"session_id": sid}), timeout=5.0
            )
            # 契约是幂等 {ok:true}——为「连接没来得及关」而破坏它得不偿失
            assert r.isError is False
            assert json.loads(getattr(r.content[0], "text", "")) == {"ok": True}
            assert await registry.get(sid) is None, "close 必须先摘牌，才能止住新活"
        finally:
            lock.release()


async def test_queued_step_after_close_reports_unknown_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """🛑 归责：close 摘牌后，排队中的 step 必须报 `unknown-session`，不得报 config-incompatible。

    竞态成因：`registry.acquire()` 到真正拿到锁之间，会话可能已被 close 摘牌并关掉 aiosqlite。
    此时若照常 step，会撞 `ValueError("Connection closed")` 落进 `except ValueError`
    → 被贴成 config-incompatible（语义是「须以新配置重开」）——归责完全错，
    且会把 client 引到错误的自救动作。正确语义是 unknown-session（同 id 重开即可）。

    **与上一条必须同批**：修了 close 的饿死，才会真正出现「摘牌后仍有 step 在排队」。
    """
    from src.mcp_server.registry import SessionRegistry

    monkeypatch.delenv("ZERO_MCP_STEP_LOCK_TIMEOUT", raising=False)
    registry = SessionRegistry()
    async with connect(build_server(registry=registry)) as client:
        await client.initialize()
        opened = await client.call_tool("zero.open_session", {})
        sid = json.loads(getattr(opened.content[0], "text", ""))["session_id"]
        await registry.close(sid)  # 模拟 close 抢先摘牌
        r = await client.call_tool(
            "zero.step", {"session_id": sid, "stim": {"valence": 0.0, "arousal": 0.0}}
        )
        assert r.isError is True
        text = getattr(r.content[0], "text", "")
        assert text.count("[zero:") == 1, f"wire 文本上令牌须全文恰一次，实得 {text!r}"
        from src.mcp_server.server import ZERO_ERROR_CODE_UNKNOWN_SESSION

        code = _wire_code(text)
        assert code == ZERO_ERROR_CODE_UNKNOWN_SESSION, (
            f"归责错位：期望 {ZERO_ERROR_CODE_UNKNOWN_SESSION}，实得 {code}"
        )


async def test_step_on_closed_connection_reports_config_incompatible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`config-incompatible` 在 wire 层的**唯一**覆盖：内核抛裸 `ValueError` 时的归责出口。

    造法**不打桩**（判别力最高）：sqlite 后端开会话 → 直接 `await session.aclose()` 关掉
    aiosqlite 连接，但**绝不** `reg.close(sid)` —— 会话仍在册，故 `zero.step` 过得了
    「未知 sid」与拿锁后的 TOCTOU 复查，一路撞进 `session.step` 的
    `ValueError("Connection closed")`，落进 `build_server.step` 的 `except ValueError` 分支。

    🛑 三个前提缺一不可，动了任何一个本用例都会**照绿、没人发现**：
      ① 会话必须**留在册**——顺手补一行 `reg.close(sid)`，码就滑成 `unknown-session`；
      ② 后端必须是 sqlite——memory 后端的 `aclose` 是 no-op，step 会照常成功；
      ③ `except ExternalPriorError` 必须排在 `except ValueError` **之前**（子类在前），
         两支互换顺序码就滑成 `external-prior-invalid`。

    ⚠ 废案备忘：`config={"hierarchical_coupling": 1.5}` → 每 step 崩在 `hierarchical_fuse`
    的那条天然路径已失效——该字段现有 `Field(ge=0.0, le=1.0)`，越界被前移到构造期，出
    `config-invalid`。别再照那条路重写本用例。
    ⚠ 本条只补 wire 覆盖，**不动**该分治的归类口径（那是设计门定的语义）。
    """
    pytest.importorskip("aiosqlite")
    from src.mcp_server.registry import SessionRegistry
    from src.mcp_server.server import ZERO_ERROR_CODE_CONFIG_INCOMPATIBLE

    monkeypatch.setenv("ZERO_CHECKPOINT_BACKEND", "sqlite")
    monkeypatch.setenv("ZERO_CHECKPOINT_DB", str(tmp_path / "closed-conn.sqlite3"))
    reg = SessionRegistry()
    async with connect(build_server(registry=reg)) as client:
        await client.initialize()
        sid = json.loads(
            getattr((await client.call_tool("zero.open_session", {})).content[0], "text", "")
        )["session_id"]
        # 🛑 必须先真跑一轮：`aiosqlite.connect()` 是**惰性**的，而 AsyncSqliteSaver.setup()
        # 在 `not conn.is_alive()` 时会 `await self.conn` 重新起一条连接 ⇒ 若在从未 step 过的
        # 会话上直接 aclose，下一次 step 会**若无其事地重连并成功**，本用例当场变成假绿
        # （本轮实测踩过：isError=False，返回了完整 expression）。
        warm = await client.call_tool(
            "zero.step", {"session_id": sid, "stim": {"valence": 0.1, "arousal": 0.1}}
        )
        assert warm.isError is False, "预热轮就失败 → 后面测的不是「连接被关」这件事"

        session, _ = await reg.acquire(sid)
        assert session is not None
        await session.aclose()  # 只关连接，**不**摘牌
        assert await reg.get(sid) is not None, "会话必须留在册，否则测到的是 unknown-session"

        r = await client.call_tool(
            "zero.step", {"session_id": sid, "stim": {"valence": 0.0, "arousal": 0.0}}
        )
        assert r.isError is True
        text = getattr(r.content[0], "text", "")
        assert text.count("[zero:") == 1, f"wire 文本上令牌须全文恰一次，实得 {text!r}"
        code = _wire_code(text)
        assert code == ZERO_ERROR_CODE_CONFIG_INCOMPATIBLE, f"归责错位：实得 {code}"


@pytest.mark.parametrize(
    "kw",
    [
        {"hierarchical_layers": 2, "hierarchical_coupling": 1.5},
        {"hierarchical_layers": 0},
        {"hierarchical_coupling": -0.1},
    ],
)
def test_hpc_knobs_rejected_at_construction(kw: dict) -> None:
    """HPC 两个旋钮的越界值必须在**构造期**被拒。

    此前两者均无 Field 约束：`coupling=1.5` 能构造成功 ⇒ `open_session` 通过、
    **每一步 step 崩**在 `hierarchical_fuse` 的运行期 raise 上；而活跃会话的 config 不可变
    ⇒ client 无法自救，且错误被贴成「内核执行失败」而非「配置越界」。
    撤掉 Field 约束本组即红（实测撤掉后三格全部构造成功）。
    """
    from src.orchestration.runner import SessionConfig

    with pytest.raises(ValueError):
        SessionConfig(**kw)


def test_hpc_knobs_accept_legal_range() -> None:
    """不得误伤合法区间（layers≥1 · coupling∈[0,1] 含端点）。"""
    from src.orchestration.runner import SessionConfig

    cfg = SessionConfig(hierarchical_layers=2, hierarchical_coupling=1.0)
    assert cfg.hierarchical_coupling == 1.0
    assert SessionConfig(hierarchical_layers=1, hierarchical_coupling=0.0).hierarchical_layers == 1


# ── describe_config / purge_session ──────────────────────────────────────


async def test_describe_config_reflects_session_not_current_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """🛑 本组最要紧的一条：带 sid 时回的必须是**该会话真实生效**的值，不是当下 env 重算的。

    造法：开会话时 env 为 A，开完后把 env 改成 B，再 describe。
    会话门控构造时固定 ⇒ 必须仍回 A。若实现改成从 `_build_session_config(None)` 重算，
    就会回 B —— 那正是配套项目 R11 警告的「回显本可生效但实际未生效的值」，
    消费方会拿它当真、比对通过、实则语义已分叉，**比不回显更危险**。
    """
    monkeypatch.setenv("ZERO_MCP_PRECISION_COMMENSURABLE", "true")
    async with connect(build_server()) as client:
        await client.initialize()
        sid = json.loads(
            getattr((await client.call_tool("zero.open_session", {})).content[0], "text", "")
        )["session_id"]
        # 会话已建；此刻翻转 env
        monkeypatch.setenv("ZERO_MCP_PRECISION_COMMENSURABLE", "false")

        with_sid = json.loads(
            getattr(
                (await client.call_tool("zero.describe_config", {"session_id": sid})).content[0],
                "text",
                "",
            )
        )
        assert with_sid["resolved_for_session"] is True
        assert with_sid["precision_commensurable"] is True, (
            "带 sid 时回的是当下 env 重算值而非会话生效值——语义分叉"
        )

        # 不带 sid = 部署端默认，应跟随当下 env
        no_sid = json.loads(
            getattr((await client.call_tool("zero.describe_config", {})).content[0], "text", "")
        )
        assert no_sid["resolved_for_session"] is False
        assert no_sid["precision_commensurable"] is False


async def test_describe_config_reports_actual_backend_impls() -> None:
    """带 sid 时回报**实际构造出的后端类名**（默认部署 = 全内存）。

    存在理由（Zero_MCP 2026-07-30 §E.13）：对方要在观察期确认「这个部署是不是全内存」，
    原方案是显式设 `ZERO_MEMORY_BACKEND` / `ZERO_SEMANTIC_BACKEND` 再在件里报所设的值。
    那证明不了事实 —— 见下一条测试。
    """
    async with connect(build_server()) as client:
        await client.initialize()
        sid = json.loads(
            getattr((await client.call_tool("zero.open_session", {})).content[0], "text", "")
        )["session_id"]
        d = json.loads(
            getattr(
                (await client.call_tool("zero.describe_config", {"session_id": sid})).content[0],
                "text",
                "",
            )
        )
        assert d["memory_store_impl"] == "InMemoryGraphStore"
        assert d["checkpointer_impl"] == "InMemorySaver"
        # 三态而非两态：语义后端默认关闭，「关闭」必须与「不可知」可区分（形制第 3 条）。
        assert d["semantic_store_impl"] == "disabled", (
            "语义关闭回了 null —— 与「无 sid 不可知」不可区分，正是形制第 3 条要避开的"
        )


async def test_describe_config_backend_impl_is_not_env_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """🛑 本组最要紧的一条：判据取在**实际构造出的对象**上，不是 env 字面量。

    造法：把 `ZERO_MEMORY_BACKEND` 设成一个无法识别的值。三个工厂对无法识别的值
    **一律静默回退内存档**（`build_graph_store` 末尾无条件 `return InMemoryGraphStore()`），
    故实际后端仍是 InMemory，而 env 字面量是那个垃圾值。
    ⇒ 若实现改成回显 env（或按 env 名映射类名），本条立刻红。

    这与 `_purge_detached_thread` 改判据是同一条纪律：**判据不得取在名字/文本上**。
    真实部署里的对应形态是 `ZERO_MEMORY_BACKEND=neo4j` 缺驱动、
    `ZERO_CHECKPOINT_BACKEND=sqlite` 缺 db extra —— 都会静默回退，
    此处用「无法识别的值」造同一效果，是为了不依赖测试环境装没装驱动。
    """
    monkeypatch.setenv("ZERO_MEMORY_BACKEND", "definitely-not-a-real-backend")
    async with connect(build_server()) as client:
        await client.initialize()
        sid = json.loads(
            getattr((await client.call_tool("zero.open_session", {})).content[0], "text", "")
        )["session_id"]
        d = json.loads(
            getattr(
                (await client.call_tool("zero.describe_config", {"session_id": sid})).content[0],
                "text",
                "",
            )
        )
        assert d["memory_store_impl"] == "InMemoryGraphStore", (
            f"回了 {d['memory_store_impl']!r} —— 疑似回显 env 字面量而非实际构造出的类"
        )


async def test_describe_config_backend_impls_null_without_sid() -> None:
    """不带 sid 时三键**在场且为 null**：进程级没有「那个实例」，且现构造有副作用。

    🛑 不得回 env 字面量充数 —— 那会让消费方以为拿到了事实。
    也不得省略键（形制第 2 条：省略会让「未实现」与「探测失败」不可区分）。
    """
    async with connect(build_server()) as client:
        await client.initialize()
        d = json.loads(
            getattr((await client.call_tool("zero.describe_config", {})).content[0], "text", "")
        )
        for key in ("memory_store_impl", "semantic_store_impl", "checkpointer_impl"):
            assert key in d, f"{key} 被省略了 —— 形制第 2 条要求不省略键"
            assert d[key] is None, f"{key} 无 sid 时回了 {d[key]!r}，应为 null（不可知）"


async def test_describe_config_shape_meets_consumer_requirements() -> None:
    """形制三条（各对应配套项目 desktop 面踩过的一个坑）：

    ① 带版本号——否则字段集增删后旧 client 静默少读；
    ② 不可知项显式回 `null`、**不省略键**——否则「未实现」与「探测失败」不可区分；
    ③ 值不得被类型过滤器吞掉——故此处断言含**非 bool** 字段（float/int/str/list）。
    """
    async with connect(build_server()) as client:
        await client.initialize()
        d = json.loads(
            getattr((await client.call_tool("zero.describe_config", {})).content[0], "text", "")
        )
        assert isinstance(d["describe_config_version"], int)
        assert "weights_version" in d and d["weights_version"] is None, "不可知项须显式 null"
        # 非 bool 字段必须在——它们正是类型过滤器会吞掉的那批
        assert isinstance(d["external_prior_precision_cap"], float)
        assert isinstance(d["max_external_streams"], int)
        assert isinstance(d["external_prior_schema_version"], int)
        assert isinstance(d["governance_gated_flags"], list) and d["governance_gated_flags"]
        assert isinstance(d["error_codes"], list)


async def test_describe_config_unknown_sid_falls_back_to_defaults() -> None:
    """未知 sid 视同不传（不报错）——否则消费方探测一个已关会话会拿到工具错误。"""
    async with connect(build_server()) as client:
        await client.initialize()
        d = json.loads(
            getattr(
                (await client.call_tool("zero.describe_config", {"session_id": "nope"})).content[0],
                "text",
                "",
            )
        )
        assert d["resolved_for_session"] is False


async def test_describe_config_governance_list_matches_source() -> None:
    """回显的治理白名单必须与源常量同源——两处各写一份迟早分叉。"""
    from src.mcp_server.server import _MCP_GOVERNANCE_GATED_FLAGS

    async with connect(build_server()) as client:
        await client.initialize()
        d = json.loads(
            getattr((await client.call_tool("zero.describe_config", {})).content[0], "text", "")
        )
        assert set(d["governance_gated_flags"]) == set(_MCP_GOVERNANCE_GATED_FLAGS)


async def test_purge_session_is_idempotent_on_unknown_id() -> None:
    """未知 id 幂等——删除入口不得因「已经没了」而报错。"""
    async with connect(build_server()) as client:
        await client.initialize()
        r = await client.call_tool("zero.purge_session", {"session_id": "never-existed"})
        assert r.isError is False
        assert json.loads(getattr(r.content[0], "text", ""))["ok"] is True


async def test_purge_memory_backend_reports_purged_false_honestly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """🛑 memory 后端 + 会话不在册 ⇒ `purged=False`，**不得回 True**。

    初版走 `build_checkpointer()` 新建 saver 再删——而它每次返回**新的空 InMemorySaver**
    （实测两次 build 非同一实例），删它等于什么都没删，却回 `purged=True`：
    **数据删除 API 的假成功**，而 memory 正是默认后端。
    这条由配套项目只读核出、我方实证确认。改回旧实现本用例即红。
    """
    monkeypatch.delenv("ZERO_CHECKPOINT_BACKEND", raising=False)
    async with connect(build_server()) as client:
        await client.initialize()
        r = await client.call_tool("zero.purge_session", {"session_id": "never-existed"})
        d = json.loads(getattr(r.content[0], "text", ""))
        assert d["ok"] is True, "ok 表示请求被正确处理"
        assert d["purged"] is False, "memory 后端无持久副本可删，回 True 即假成功"
        assert d["backend"] == "memory"
        assert "如实回报" in d["detail"]


async def test_purge_judges_by_actual_saver_not_env_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """🛑 `ZERO_CHECKPOINT_BACKEND=sqlite` 但驱动缺失回退 InMemory ⇒ 必须仍回 `purged=False`。

    `build_checkpointer` 在 sqlite **缺 db extra** 时会**告警回退 InMemorySaver**。
    若按 **env 名**判后端，这种部署会走「持久后端」分支 ⇒ 在回退出的**空** saver 上
    `adelete_thread` ⇒ 又回 `purged=True`
    —— 正是刚修掉的那个「假成功」**换了个门回来**。
    判据必须取在**实际构造出的 saver 类型**上。改回按 env 名判，本用例即红。
    """
    from langgraph.checkpoint.memory import InMemorySaver

    import src.storage.checkpointer as ck

    monkeypatch.setenv("ZERO_CHECKPOINT_BACKEND", "sqlite")
    # 模拟「缺 db extra」：_sqlite_saver 返回 None ⇒ build_checkpointer 回退 InMemorySaver
    monkeypatch.setattr(ck, "_sqlite_saver", lambda serde: None)
    assert isinstance(ck.build_checkpointer(), InMemorySaver), (
        "前提不成立：未发生回退，本用例失去判别力"
    )

    async with connect(build_server()) as client:
        await client.initialize()
        r = await client.call_tool("zero.purge_session", {"session_id": "never-existed"})
        d = json.loads(getattr(r.content[0], "text", ""))
        assert d["purged"] is False, (
            "env 写 sqlite 但实际回退 InMemory 时回了 purged=True —— 假成功"
        )
        assert "回退" in d["detail"], "须点明这是驱动缺失回退，否则运维看不出为什么没删成"


async def test_describe_config_bad_env_carries_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不带 sid 时部署端 env 坏值必须带机读码，不得裸穿。

    该路径会跑 `_build_session_config(None)`，坏 env 抛的是 `ServerEnvError`；
    其它三个入口都做了转码，**只有本工具漏过** —— 与 `_UNKNOWN_SESSION_MARKER` 那条死码同型。
    """
    monkeypatch.setenv("ZERO_EXTERNAL_PRIOR_PRECISION_CAP", "0.8x")
    async with connect(build_server()) as client:
        await client.initialize()
        r = await client.call_tool("zero.describe_config", {})
        assert r.isError is True
        assert _wire_code(getattr(r.content[0], "text", "")) == "deploy-env-invalid"


async def test_open_session_description_documents_interrupt_probe() -> None:
    """对外元数据必须与返回体一致：`list_tools` 的 description 要提 `interrupt_probe`。

    它是 client 能读到的契约。代码改了、description 没改 = 契约两张皮；
    更糟的是旧文案宣传的正是「靠 interrupted_at 缺席判断」这个已被消灭的口径。
    """
    async with connect(build_server()) as client:
        await client.initialize()
        tools = {t.name: (t.description or "") for t in (await client.list_tools()).tools}
        desc = tools["zero.open_session"]
        assert "interrupt_probe" in desc
        for state in ("not_probed", "clean", "interrupted", "probe_failed"):
            assert state in desc, f"四态未在对外 description 中列全：缺 {state}"
        assert "不要靠 interrupted_at 是否缺席" in desc


async def test_purge_unsupported_backend_carries_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未接线后端（postgres）必须带机读码，不得裸穿 `NotImplementedError`。

    实证：`build_checkpointer()` 在 postgres 下构造期 raise，且**不是** ToolError
    ⇒ 无 `[zero:<code>]` 令牌，消费方查表落空。
    """
    monkeypatch.setenv("ZERO_CHECKPOINT_BACKEND", "postgres")
    async with connect(build_server()) as client:
        await client.initialize()
        r = await client.call_tool("zero.purge_session", {"session_id": "x"})
        assert r.isError is True
        assert _wire_code(getattr(r.content[0], "text", "")) == "deploy-env-invalid"


async def test_purge_uses_the_sessions_own_saver_not_a_fresh_one() -> None:
    """在册会话的删除必须落在**该会话自己的** checkpointer 上。

    钉法：换掉 session 自身 saver 的 `adelete_thread`，断言它**确实被调用**且拿到对的 thread_id。
    若实现改回「新建一个 saver 再删」，本用例即红——那正是假成功的成因。
    """
    from src.mcp_server.registry import SessionRegistry

    registry = SessionRegistry()
    async with connect(build_server(registry=registry)) as client:
        await client.initialize()
        sid = json.loads(
            getattr((await client.call_tool("zero.open_session", {})).content[0], "text", "")
        )["session_id"]
        session = await registry.get(sid)
        assert session is not None
        seen: list[str] = []

        async def _fake_delete(thread_id: str) -> None:
            seen.append(thread_id)

        session.checkpointer.adelete_thread = _fake_delete  # type: ignore[attr-defined]
        r = await client.call_tool("zero.purge_session", {"session_id": sid})
        d = json.loads(getattr(r.content[0], "text", ""))
        assert d["purged"] is True
        assert seen == [sid], f"未落在会话自身 saver 上（实收 {seen}）"


async def test_purge_session_closes_then_purges() -> None:
    """purge 先走 close（摘牌），之后同 id step 报 unknown-session。"""
    from src.mcp_server.registry import SessionRegistry

    registry = SessionRegistry()
    async with connect(build_server(registry=registry)) as client:
        await client.initialize()
        sid = json.loads(
            getattr((await client.call_tool("zero.open_session", {})).content[0], "text", "")
        )["session_id"]
        r = await client.call_tool("zero.purge_session", {"session_id": sid})
        assert r.isError is False
        assert await registry.get(sid) is None, "purge 必须先摘牌"
        after = await client.call_tool(
            "zero.step", {"session_id": sid, "stim": {"valence": 0.0, "arousal": 0.0}}
        )
        assert _wire_code(getattr(after.content[0], "text", "")) == "unknown-session"


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
        # ⚠ 原先是 `"非" in text and "client" in text`：单个「非」字在中文文案里几乎必然出现
        # （「非法配置」也含它）⇒ 实际不可证伪。改断源码里那句归责原文的整段。
        assert "非** client config 问题" in text, "错误未澄清归责方"


def test_canonical_physiology_is_governance_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    """`canonical_physiology` 入治理白名单：client 经 config 传它被静默忽略。

    它决定 physiology 载荷的**量纲与键集**（legacy 归一 `[0,1]` + `pupil_mm` ↔
    canonical μS`[0,20]` + `temperature_c`），消费侧据此选 `skin_conductance_max_us`
    是 1.0 还是 20.0——选错即 **20× 欠/过标度且不报错**。
    入白名单前配套项目实测：两侧 env 全未设时，client 经 config 置真即可让载荷变成
    canonical μS（sc=16.0），而消费侧按 env 推断成 legacy。
    """
    from src.mcp_server.server import _MCP_GOVERNANCE_GATED_FLAGS, _build_session_config

    monkeypatch.delenv("ZERO_PHYSIOLOGY_CANONICAL_PLACEHOLDER", raising=False)
    assert "canonical_physiology" in _MCP_GOVERNANCE_GATED_FLAGS
    assert _build_session_config({"canonical_physiology": True}).canonical_physiology is False


def test_canonical_physiology_env_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """入白名单不得把它钉死——env 仍是唯一且有效的入口（成对要求）。"""
    from src.mcp_server.server import _build_session_config

    monkeypatch.setenv("ZERO_PHYSIOLOGY_CANONICAL_PLACEHOLDER", "true")
    assert _build_session_config(None).canonical_physiology is True
    assert _build_session_config({"canonical_physiology": False}).canonical_physiology is True


def test_timeout_codes_are_split_by_retry_semantics() -> None:
    """两个超时码必须分开：重试语义相反，一个码承载两种等于把判别推回人读文案。

    · `timeout-lock`：本轮未进内核 ⇒ **可**退避重试；
    · `timeout-step`：取消 ainvoke 会留半截运行态 ⇒ **不可**原样重试。
    ⚠ `timeout-step` 目前只登记不产出（执行超时未实现），本用例同时钉住这个事实——
    若它开始被产出而语义未定，这条会提醒补齐。
    """
    from src.mcp_server.server import (
        ZERO_ERROR_CODE_TIMEOUT_LOCK,
        ZERO_ERROR_CODE_TIMEOUT_STEP,
        ZERO_ERROR_CODES,
    )

    assert ZERO_ERROR_CODE_TIMEOUT_LOCK != ZERO_ERROR_CODE_TIMEOUT_STEP
    assert {ZERO_ERROR_CODE_TIMEOUT_LOCK, ZERO_ERROR_CODE_TIMEOUT_STEP} <= ZERO_ERROR_CODES
    # 只登记不产出：AST 查所有 _tool_error(...) 调用点的第一个实参，不得有 TIMEOUT_STEP。
    # （不能用字符串搜——登记表 ZERO_ERROR_CODES 自身就含这个名字，会假红。第一版就栽在这。）
    import ast
    import inspect

    import src.mcp_server.server as srv

    tree = ast.parse(inspect.getsource(srv))
    emitted = {
        node.args[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_tool_error"
        and node.args
        and isinstance(node.args[0], ast.Name)
    }
    assert "ZERO_ERROR_CODE_TIMEOUT_STEP" not in emitted, (
        "timeout-step 已被产出，但执行超时的半截运行态语义尚未落地——请同批补齐"
    )
    assert "ZERO_ERROR_CODE_TIMEOUT_LOCK" in emitted, (
        "timeout-lock 应当有产出点，否则本组没在测东西"
    )


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
                assert names == {
                    "zero.open_session",
                    "zero.step",
                    "zero.close_session",
                    "zero.describe_config",
                    "zero.purge_session",
                    "zero.motion",  # 动作通道（2026-08-05）
                }
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
    """传空/空白 session_id → `[zero:payload-invalid]`（非空字符串校验）。

    与 `::test_error_codes_are_extractable` 的空白格分工：那格按**出口**清点 wire 令牌形态，
    本条钉的是 `.strip()` 判据本身——纯空串与纯空白**两者都**要被拒。
    ⚠ 原先断的是 `"session_id" in text`：工具入参名本身就叫 session_id，几乎不可能红。
    """
    from src.mcp_server.server import ZERO_ERROR_CODE_PAYLOAD_INVALID

    async with connect(build_server()) as client:
        await client.initialize()
        for bad in ("", "  "):
            r = await client.call_tool("zero.open_session", {"session_id": bad})
            assert r.isError is True, f"session_id={bad!r} 未被拒"
            code = _wire_code(getattr(r.content[0], "text", ""))
            assert code == ZERO_ERROR_CODE_PAYLOAD_INVALID, f"session_id={bad!r} 实得码 {code}"


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
    """close_session 幂等关会话（memory/sqlite 后端）；close 后同 id step 回 unknown-session。

    ⚠ 断言消息必须带 `backend`：两个后端跑在同一个循环里，不带就看不出是哪格红。
    ⚠ 原先断的是 `"unknown-session" in text` 子串——对「令牌退回裸前缀」恒绿，已收成 `_wire_code`。
    """
    from src.mcp_server.server import ZERO_ERROR_CODE_UNKNOWN_SESSION

    for backend in ("memory", "sqlite"):
        if backend == "sqlite":
            pytest.importorskip("aiosqlite")
            monkeypatch.setenv("ZERO_CHECKPOINT_DB", str(tmp_path / f"{backend}.sqlite3"))
        monkeypatch.setenv("ZERO_CHECKPOINT_BACKEND", backend)
        async with connect(build_server()) as client:
            await client.initialize()
            sid = json.loads(
                getattr((await client.call_tool("zero.open_session", {})).content[0], "text", "")
            )["session_id"]
            c = await client.call_tool("zero.close_session", {"session_id": sid})
            assert json.loads(getattr(c.content[0], "text", "")) == {"ok": True}
            r = await client.call_tool(
                "zero.step", {"session_id": sid, "stim": {"valence": 0.0, "arousal": 0.0}}
            )
            assert r.isError is True, f"backend={backend}：close 后 step 应报错"
            code = _wire_code(getattr(r.content[0], "text", ""))
            assert code == ZERO_ERROR_CODE_UNKNOWN_SESSION, f"backend={backend}：实得码 {code}"


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


# ── 生效传输回读 + stateless_http 双重 pin（Zero_MCP 2026-07-30 §3.3）────────────────────
#
# 两件事同批，动机同源：
# ① **运行期回读传输**——我方自己写在 `describe_config` docstring 里的那句「HTTP 传输下两进程
#    **不共享 env**，『同名 env 对齐』这条机制结构上不成立」对**传输模式本身**逐字适用 ⇒
#    源码 pin 只能保证「源码里是 stateful」，只有 `describe_config` 能让消费方在运行期、
#    对着真正连上的那个部署确认。
# ② **stateless_http 的双重 pin**——对方原话：「pin 要覆盖两件事：`FastMCP(...)` 构造不传
#    `stateless_http`，**且不存在读该值的 env**。只钉构造实参挡不住『以后加一个 env』——
#    而**部署期翻转**恰恰是我方唯一在意的失败形态。」
# ⚠ 全仓今天 `stateless_http` 零命中 ⇒ 两条守卫**必然绿**，故判别力不能靠「今天是绿的」自证：
#   `::test_stateless_guards_catch_the_flip_vectors` 喂合成源码把三种翻转与两处合法读一起钉住
#   （仓内「绿灯必须先证明能红」）。


def _fastmcp_calls_in(source: str, filename: str, func_name: str) -> list[list[str | None]]:
    """返回 `func_name` 里每次 `FastMCP(...)` 调用的关键字实参名列表（`**d` 展开记作 `None`）。

    按名字匹配 `FastMCP`（`ast.Name` 或 `ast.Attribute`）。若上游改成
    `from … import FastMCP as FM`，本函数会回空表 —— 调用方据此**报失锚**（安全方向：
    宁可红一次让人重写守卫，也不要静默变成真空绿）。
    """
    tree = ast.parse(source, filename=filename)
    calls: list[list[str | None]] = []
    for target in ast.walk(tree):
        if not (
            isinstance(target, ast.FunctionDef | ast.AsyncFunctionDef) and target.name == func_name
        ):
            continue
        for node in ast.walk(target):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if called == "FastMCP":
                calls.append([kw.arg for kw in node.keywords])
    return calls


# 读 env 的**间接层**：这些函数体内 env 名是形参（变量），由调用方传字面量 ⇒ 不算「不可判」。
# 新增同类助手时加进来，否则 `_stateless_flip_sites` 会把它报成不可判（刻意的：那正是
# 「以后偷偷加一个 env」最容易藏进去的地方）。
_ENV_NAME_INDIRECTION: frozenset[str] = frozenset(
    {"_env_flag", "_env_number", "_env_optional_float"}
)
# 直接读 env 的写法（`_ENV_NAME_INDIRECTION` 里的助手也在内：它们第一个实参就是 env 名）。
_ENV_READ_FUNCS: frozenset[str] = frozenset({"getenv"}) | _ENV_NAME_INDIRECTION


def _is_os_environ(node: ast.AST) -> bool:
    """`node` 是否 `os.environ` / 裸 `environ`。"""
    return (isinstance(node, ast.Attribute) and node.attr == "environ") or (
        isinstance(node, ast.Name) and node.id == "environ"
    )


def _env_read_name(node: ast.AST) -> tuple[bool, str | None]:
    """`(该节点是否一次 env 读取, env 名)`；名不是字符串字面量则 env 名回 `None`（=不可判）。

    覆盖 `os.getenv(...)` / 裸 `getenv(...)`（`from os import getenv`）/ `os.environ[...]` /
    `os.environ.get(...)` / `os.environ.setdefault(...)`，以及 `_ENV_NAME_INDIRECTION` 里的助手。
    """
    if isinstance(node, ast.Subscript) and _is_os_environ(node.value):
        key = node.slice
        return True, key.value if isinstance(key, ast.Constant) and isinstance(
            key.value, str
        ) else None
    if not isinstance(node, ast.Call):
        return False, None
    func = node.func
    hit = (isinstance(func, ast.Name) and func.id in _ENV_READ_FUNCS) or (
        isinstance(func, ast.Attribute)
        and (
            func.attr in _ENV_READ_FUNCS
            or (func.attr in ("get", "setdefault") and _is_os_environ(func.value))
        )
    )
    if not hit:
        return False, None
    first = node.args[0] if node.args else None
    return True, (
        first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None
    )


def _env_read_sites(source: str, filename: str) -> list[tuple[int, str | None, str]]:
    """扫一段源码，返回全部 env 读取点：`(行号, env 名或 None, 最内层函数名)`。

    最内层函数名用于**豁免间接层**与判定「某个 env 只在某一个函数里被读」（同源守卫要用）。
    """
    sites: list[tuple[int, str | None, str]] = []

    def visit(node: ast.AST, enclosing: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = (
                child.name
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                else enclosing
            )
            is_read, env_name = _env_read_name(child)
            if is_read:
                sites.append((getattr(child, "lineno", 0), env_name, enclosing))
            visit(child, inner)

    visit(ast.parse(source, filename=filename), "<module>")
    return sites


def _stateless_flip_sites(source: str, filename: str) -> list[tuple[int, str]]:
    """扫一段源码，返回能让 `stateless_http` **在部署期被翻转**的位置：`(行号, 原因)`。

    两条判据（缺一不可，正是对方要的「双重」）：
    **写入侧（与 env 名无关）**——任何调用里的 `stateless_http=` 关键字实参；对 `.stateless_http`
    属性的**赋值/删除**。这一条**不看 env 叫什么**，故「以后加一个叫别的名字但语义相同的 env」
    只要经这两种写法落地就会被抓。
    **env 名侧**——env 名归一化（去掉非字母数字后大写）后含 `STATELESS`；以及**非豁免函数**里
    env 名不是字面量（记「不可判」并同样报红）。

    🛑 **只读不算**：`{"stateless_http": mcp.settings.stateless_http}` 这种 Load 上下文的属性读
    是 `describe_config` 的生产写法，必须放行 —— 判据要分清「读出来如实回报」与「写进去翻转」。
    """
    import re

    tree = ast.parse(source, filename=filename)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "stateless_http":
            found.append((node.value.lineno, "关键字实参 stateless_http=… 传进了某次调用"))
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == "stateless_http"
            and isinstance(node.ctx, ast.Store | ast.Del)
        ):
            found.append((node.lineno, "对 .stateless_http 属性赋值/删除（部署期翻转的写入侧）"))
    for lineno, env_name, enclosing in _env_read_sites(source, filename):
        if env_name is None:
            if enclosing not in _ENV_NAME_INDIRECTION:
                found.append(
                    (
                        lineno,
                        f"{enclosing}() 里 env 名不是字面量 ⇒ 守卫**不可判**"
                        "（改成字面量，或把新助手加进 _ENV_NAME_INDIRECTION）",
                    )
                )
        elif "STATELESS" in re.sub(r"[^A-Za-z0-9]", "", env_name).upper():
            found.append((lineno, f"env {env_name!r} 的名字命中 stateless 语义"))
    return sorted(found)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "stdio"),  # 未设 = 默认
        ("stdio", "stdio"),
        ("http", "streamable-http"),  # 短别名 → **规范化**成 SDK 正名
        ("HTTP", "streamable-http"),  # 大小写不敏感
        ("Streamable-HTTP", "streamable-http"),
        ("streamable-http", "streamable-http"),
        ("", "stdio"),  # 空串：未识别 → 与 __main__ 的 else 同向回落
        ("sse", "stdio"),  # 拼错/别的传输名 → 同上
        (" http", "stdio"),  # 带空白：抽取前不 strip，回读面也不得 strip
    ],
)
async def test_describe_config_reports_effective_transport(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: str
) -> None:
    """`describe_config` 回**规范化后的生效传输值**，未识别值的回落方向与 `__main__` 一致。

    对方核过我方返回体里此前**没有任何传输相关字段**（daecce1）⇒ 消费方无手段在运行期确认
    「连上的这个部署到底起的是哪种传输」，而它整套取消模型的适用边界就挂在这上面。
    ⚠ 断言的是**规范化值**（`http` → `streamable-http`），不是原始 env 字符串：回原始串等于把
    别名/大小写的解析责任又推回消费方，两边各解析一次就会分叉。
    """
    from src.mcp_server.server import TRANSPORT_STDIO, TRANSPORT_STREAMABLE_HTTP

    if raw is None:
        monkeypatch.delenv("ZERO_MCP_TRANSPORT", raising=False)
    else:
        monkeypatch.setenv("ZERO_MCP_TRANSPORT", raw)
    async with connect(build_server()) as client:
        await client.initialize()
        d = json.loads(
            getattr((await client.call_tool("zero.describe_config", {})).content[0], "text", "")
        )
    assert d["transport"] == expected, f"ZERO_MCP_TRANSPORT={raw!r} 的生效传输回错了"
    assert d["transport"] in (TRANSPORT_STDIO, TRANSPORT_STREAMABLE_HTTP), (
        "值域必须是规范化后的两态之一——按符号名 pin，勿 pin 字面量"
    )


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        " ",
        "stdio",
        "STDIO",
        "http",
        "Http",
        "HTTP",
        "streamable-http",
        "STREAMABLE-HTTP",
        " http",
        "http ",
        "streamable_http",
        "sse",
        "1",
        "true",
    ],
)
def test_resolve_transport_matches_pre_extraction_expression(
    monkeypatch: pytest.MonkeyPatch, raw: str | None
) -> None:
    """抽取出的 `resolve_transport` 与抽取前写在 `__main__.main` 里那两行**逐字同判**。

    oracle 就是抽取前的原表达式（在测试里重写一份是刻意的：它是「行为逐字不变」唯一可执行的
    判据）。含 `" http"` / `"http "` 两条 —— 抽取前没有 `strip()`，加上就是**行为变更**而非重构。
    """
    import os

    from src.mcp_server.server import TRANSPORT_STREAMABLE_HTTP, resolve_transport

    if raw is None:
        monkeypatch.delenv("ZERO_MCP_TRANSPORT", raising=False)
    else:
        monkeypatch.setenv("ZERO_MCP_TRANSPORT", raw)
    # ⚠ 抽取前的原样表达式，勿「顺手改进」
    legacy_is_http = os.getenv("ZERO_MCP_TRANSPORT", "stdio").lower() in (
        "http",
        "streamable-http",
    )
    assert (resolve_transport() == TRANSPORT_STREAMABLE_HTTP) is legacy_is_http, (
        f"ZERO_MCP_TRANSPORT={raw!r}：抽取后的判别与抽取前不一致 —— 这是行为变更，不是重构"
    )


def test_transport_resolution_is_single_sourced() -> None:
    """🛑 AST：`ZERO_MCP_TRANSPORT` 在 `src/mcp_server/` 里**只能**被 `resolve_transport` 读。

    件一的关键设计约束：传输模式此前只在 `__main__.main` 里解析，而 `describe_config` 在
    `server.py`。若在回读面里**重抄一遍**解析，就是亲手造一个新的漂移源 —— 回读面会宣称一个
    部署端根本没起的传输，且消费方会拿它当真（同 R11「回显本可生效但实际未生效的值」的失效型，
    比不报更糟）。故解析必须是**同一个符号**。

    ⚠ 挡不住：把 env 名先塞进变量再读（`n = "ZERO_MCP_TRANSPORT"; os.getenv(n)`）——
    那属存心绕过；本守卫的目标是无心之失（同 `_toolerror_sites` 的边界声明）。
    """
    import src.mcp_server.server as srv

    files = sorted(Path(srv.__file__).parent.glob("*.py"))
    assert {p.name for p in files} >= {"server.py", "__main__.py"}, (
        "扫描面缺 server.py/__main__.py —— 守卫失去两个抓手，glob 写错了"
    )
    readers: list[tuple[str, str, int]] = []
    for path in files:
        # 显式 utf-8：本仓 Windows 默认 cp936，中文注释会 UnicodeDecodeError（表现为报错而非报红）
        source = path.read_text(encoding="utf-8")
        readers += [
            (path.name, enclosing, line)
            for line, env_name, enclosing in _env_read_sites(source, path.name)
            if env_name == "ZERO_MCP_TRANSPORT"
        ]
    assert readers, "全包内没有一处读 ZERO_MCP_TRANSPORT —— 守卫已失锚（env 名改了？），请重写"
    assert [(f, fn) for f, fn, _ in readers] == [("server.py", "resolve_transport")], (
        f"ZERO_MCP_TRANSPORT 被多处/别处读取：{readers}；解析必须只在 resolve_transport 里，"
        "__main__ 与 describe_config 共用它 —— 两处各写一份必然漂移"
    )


async def test_describe_config_reports_stateless_http() -> None:
    """一次调用就能确认「stateful + 何种传输」：`stateless_http` 与 `transport` 同时在。

    ⚠ 值取自**本进程 FastMCP 实例的 settings**，不是写死的 `False` —— 哪天真开了 stateless，
    这里会**如实**变 `True` 而不是继续说谎；「不会被悄悄开」由下面两条结构守卫钉住。
    """
    from mcp.server.fastmcp import FastMCP

    async with connect(build_server()) as client:
        await client.initialize()
        d = json.loads(
            getattr((await client.call_tool("zero.describe_config", {})).content[0], "text", "")
        )
    assert d["stateless_http"] is False, "我方从不传该实参 ⇒ 今天恒 False"
    assert "transport" in d, "两键须成对：只知 stateful 不知传输，消费方仍要再问一次"
    assert FastMCP("probe").settings.stateless_http is False, (
        "SDK 的 stateless_http 默认已不是 False —— 「不传即 stateful」这句前提没了，"
        "两侧取消模型的适用边界须重测（对方 §3.3 限定：反号结论来自最小 FastMCP 变体）"
    )


def test_fastmcp_construction_does_not_pass_stateless_http() -> None:
    """🛑 AST（双重 pin 之一）：`build_server` 里那次 `FastMCP(...)` 不得传 `stateless_http`。

    对方挂在「我方是 stateful」这个前提上的是**整套取消模型的适用边界**（断连不取消 / 关传输
    才取消 / `__aexit__` 发的 DELETE 才是真风险口），而该结论此前只以 notes 文字形式存在
    —— 产品码 0 处、守卫 0 处，**没有任何东西会因为对端翻转而变红**。这正是「该 pin」的
    定义性情形。stateless 下取消语义**反号**（对方最小 FastMCP 变体实测）。
    """
    import src.mcp_server.server as srv

    source = Path(srv.__file__).read_text(encoding="utf-8")
    calls = _fastmcp_calls_in(source, "server.py", "build_server")
    assert len(calls) == 1, (
        f"build_server 里的 FastMCP 构造点不是恰好一处（实得 {len(calls)}）—— 守卫失锚"
        "（改名/别名 import/挪了位置？），请重写而不是删掉"
    )
    assert None not in calls[0], (
        "FastMCP(...) 里出现了 `**` 展开 ⇒ 本守卫看不见键名、已失效：请把实参写成显式关键字"
    )
    assert "stateless_http" not in calls[0], (
        "FastMCP(...) 传了 stateless_http —— 跨仓前提是「不传该实参、取 SDK 默认 stateful」；"
        "真要开 stateless，两侧的取消语义都要重测（对方原话：不能直接搬用）"
    )


def test_no_env_can_flip_stateless_http() -> None:
    """🛑 AST（双重 pin 之二）：`src/mcp_server/` 下不存在能让 `stateless_http` 被翻转的位置。

    对方原话：「只钉构造实参挡不住『以后加一个 env』——而**部署期翻转**恰恰是我方唯一在意的
    失败形态。」

    **能挡**（判据在 `_stateless_flip_sites`，判别力由
    `::test_stateless_guards_catch_the_flip_vectors` 自证）：
      · 任何调用里的 `stateless_http=` 关键字实参 —— 含 `FastMCP` 之外的调用、含未来新增文件
        （与上一条守卫**故意重叠**：上一条锚在当前构造点、归因精确；本条扫全包、防未来新构造点）；
      · 对 `.stateless_http` 的**赋值/删除**（`mcp.settings.stateless_http = …`）——
        **与 env 叫什么无关**，故「以后加一个叫别的名字但语义相同的 env」只要经这种写法落地
        就会被抓（这是本条相对「只查 env 名」的真正增量）；
      · env **名**归一化后含 `STATELESS`（去掉非字母数字再比，故 `ZERO_MCP_STATE_LESS_HTTP`
        这种靠下划线拆词的写法躲不过）；
      · 非豁免函数里 env 名不是字面量 → 记「不可判」同样报红。

    **挡不住**（明说，不声称挡住了实际挡不住的）：
      · `setattr(mcp.settings, "stateless_" + "http", True)` 一类动态写入；
      · 别的文件里新写 `FastMCP(**d)` 且 `d` 运行期拼出（当前构造点由上一条守卫的 `**` 检查兜住）；
      · 把 `FastMCP` 构造整个搬到 `src/mcp_server/` **之外**（扫描面只有本包）；
      · env 的**值**（如 `ZERO_MCP_HTTP_MODE=stateless`）：只查名、不查值；
      · SDK 自己的 `FASTMCP_*` env 前缀 —— 当前版本对 `stateless_http` **无效**（已实测：
        `FastMCP.__init__` 把它作显式形参传给 Settings，故 `FASTMCP_STATELESS_HTTP` 不起作用），
        但那属源码 pin 结构上摸不到的部署期向量，只能靠 `describe_config` 的运行期回读兜住
        —— 这正是对方 §3.3-2 把回读面称作「更值的形态」的理由。
    """
    import src.mcp_server.server as srv

    files = sorted(Path(srv.__file__).parent.glob("*.py"))
    assert files, "扫描面为空 → 守卫恒真空绿，glob 写错了"
    assert any(p.name == "server.py" for p in files), "server.py 不在扫描面内 → 守卫失去主要抓手"
    offenders: list[str] = []
    for path in files:
        source = path.read_text(encoding="utf-8")  # 显式 utf-8：见上一条守卫的同款说明
        offenders += [
            f"{path.name}:{line}: {why}" for line, why in _stateless_flip_sites(source, path.name)
        ]
    assert not offenders, (
        f"出现了能翻转 stateless_http 的位置：{offenders}；跨仓前提是「不传该实参、"
        "**且不存在读该值的 env**」——部署期翻转会让对方整套取消模型的适用边界反号"
    )


def test_stateless_guards_catch_the_flip_vectors() -> None:
    """判别力自证：喂合成源码，三种翻转必须全抓、两处合法写法必须不抓。

    ⚠ 必须有这一条：全仓今天 `stateless_http` 零命中 ⇒ 上面两条守卫**必然绿**，
    「今天绿」证明不了它们会红（仓内「绿灯必须先证明能红」）。这里把判别力做成常驻断言，
    不依赖任何一次性的手工变异。
    """
    flips = (
        "import os\n"
        "from mcp.server.fastmcp import FastMCP\n"
        "\n"
        "def build_server():\n"
        "    mcp = FastMCP('zero', stateless_http=True)\n"
        "    mcp.settings.stateless_http = os.getenv('ZERO_MCP_STATELESS_HTTP') == '1'\n"
        "    mcp.settings.stateless_http = _env_flag('ZERO_MCP_FAST_MODE', False)\n"
        "    return mcp\n"
    )
    # ① 构造实参：两条守卫都该抓（重叠是故意的，见 test_no_env_can_flip_stateless_http）
    assert _fastmcp_calls_in(flips, "server.py", "build_server") == [["stateless_http"]]
    by_line: dict[int, str] = {}
    for line, why in _stateless_flip_sites(flips, "server.py"):
        by_line[line] = by_line.get(line, "") + " | " + why
    assert set(by_line) == {5, 6, 7}, f"有翻转向量未被抓住，实得 {by_line}"
    assert "关键字实参" in by_line[5]  # ① FastMCP(..., stateless_http=True)
    assert "命中 stateless 语义" in by_line[6] and "属性赋值" in by_line[6]  # ② 同名 env + 写入
    assert "属性赋值" in by_line[7], (  # ③ **换个名字**的 env + 写入 —— 只有写入侧判据能抓
        "换名 env 只被写入侧判据兜住；这条一旦漏，「以后加一个叫别的名字的 env」就穿了"
    )

    legal = (
        "import os\n"
        "from mcp.server.fastmcp import FastMCP\n"
        "\n"
        "def build_server():\n"
        "    mcp = FastMCP('zero', host=os.getenv('ZERO_MCP_HTTP_HOST', '127.0.0.1'))\n"
        "    return {'stateless_http': mcp.settings.stateless_http}\n"
        "\n"
        "def _env_flag(name, default):\n"
        "    return os.getenv(name)\n"
    )
    assert _stateless_flip_sites(legal, "server.py") == [], (
        "误报：`{'stateless_http': mcp.settings.stateless_http}` 是 describe_config 的生产写法"
        "（只读回显·必须放行），`_env_flag(name)` 是既有间接层"
    )
    opaque = "import os\n\ndef f(n):\n    return os.getenv(n)\n"
    assert [why for _, why in _stateless_flip_sites(opaque, "server.py")] == [
        "f() 里 env 名不是字面量 ⇒ 守卫**不可判**"
        "（改成字面量，或把新助手加进 _ENV_NAME_INDIRECTION）"
    ], "非豁免函数里的变量 env 名必须报「不可判」——那是「偷偷加一个 env」最好的藏身处"


# ── R10 日志回落：setup_logging 失败时必须真调 basicConfig ────────────────────
# 🛑 为什么要这条锁：`__main__.main` 的注释与 warning 都宣称「回落 basicConfig」，
# 而 2026-07-30 实测代码里**根本没有那个调用** —— setup_logging 失败时 root logger
# 保持未配置（默认 WARNING、无 handler）⇒ warning 靠 lastResort 勉强出 stderr，
# 而**所有 INFO 全丢**，包括 `open_session` 的门控快照。即 R10 想要的可裁定性
# 恰在最该生效的场景（只读容器 / 目录不可写）失效，且相对改动前是**回退**。
# 一句宣称「已回落」的注释 + 没有该调用的代码 = 一个永远不会响的警报。
#
# ⚠ 本文件此前**没有任何测试 import 过 `src.mcp_server.__main__`** ⇒ `main()` 整块零运行时覆盖。
# 本条是该模块的第一条运行时锁，故刻意把三处外部依赖全 patch 掉、只观测日志配置这一件事。
def test_setup_logging_failure_falls_back_to_basicconfig(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """setup_logging 抛异常时，`main()` 必须真调 `logging.basicConfig` 且带 ZERO_LOG_LEVEL。"""
    import logging as _logging

    from src.mcp_server import __main__ as entry

    calls: list[dict[str, object]] = []

    def _spy_basic_config(**kwargs: object) -> None:
        calls.append(kwargs)

    def _boom() -> str:
        raise OSError("日志目录不可写（模拟只读容器）")

    class _DummyServer:
        def run(self, transport: str) -> None:
            self.ran_with = transport

    dummy = _DummyServer()
    # setup_logging 在 main() 内部**函数体里** import，故要 patch 它的来源模块。
    monkeypatch.setattr("src.observability.setup_logging", _boom)
    monkeypatch.setattr(entry, "build_server", lambda: dummy)
    monkeypatch.setattr(_logging, "basicConfig", _spy_basic_config)
    monkeypatch.setenv("ZERO_MCP_TRANSPORT", "stdio")  # 走 else 支，不起 uvicorn
    monkeypatch.setenv("ZERO_LOG_LEVEL", "DEBUG")  # 非默认值，用来证明它真被读

    entry.main()

    assert calls, (
        "setup_logging 失败时未调 logging.basicConfig —— root logger 保持未配置 ⇒ INFO 全丢"
        "（含 open_session 的门控快照）。注释宣称『回落 basicConfig』就必须真有该调用"
    )
    assert calls[0].get("level") == "DEBUG", (
        f"回落时未按 ZERO_LOG_LEVEL 配级别，实得 {calls[0]!r}；"
        "改动前该路径恒 basicConfig(level=ZERO_LOG_LEVEL)，丢掉它是回退"
    )
    assert getattr(dummy, "ran_with", None) == "stdio", (
        "日志回落不得挡启动 —— server 仍须照常起 stdio 传输"
    )
