"""zero-link MCP server 契约测试：映射纯函数 + FastMCP in-memory client 往返 + 错误路径。

`mcp` 走 optional-deps（默认不入 core），无 mcp 的机器整包不可导入 → `importorskip` 跳过（零回归）。
断言对齐 MCP 侧 client 期望契约（open→{session_id} / step→expression 子 dict / close→{ok}；
错误→isError=true）。
"""

from __future__ import annotations

import json

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
