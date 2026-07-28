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
