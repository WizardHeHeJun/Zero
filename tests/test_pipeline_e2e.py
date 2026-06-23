"""端到端集成：训练好的解码器经 build_graph 注入，真表达流过完整 LangGraph 管线。

torch 缺失则整文件跳过（注入路径需要 torch 模型；核心管线本身不依赖 torch）。
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from src.agents.models.expression_decoder import ExpressionDecoder  # noqa: E402
from src.memory.client import MemoryClient  # noqa: E402
from src.orchestration.graph import build_graph  # noqa: E402
from src.orchestration.runner import ALLOWED_CHECKPOINT_TYPES  # noqa: E402
from src.orchestration.state import Stimulus  # noqa: E402
from src.storage.checkpointer import build_checkpointer  # noqa: E402

FULL_AU = {"AU04", "AU06", "AU12", "AU15", "intensity"}


def make_graph(decoder=None):
    return build_graph(
        build_checkpointer(ALLOWED_CHECKPOINT_TYPES),
        MemoryClient(),
        expression_decoder=decoder,
    )


async def test_injected_decoder_flows_through_pipeline() -> None:
    graph = make_graph(ExpressionDecoder())  # 未训练也满足契约
    result = await graph.ainvoke(
        {"stimulus": Stimulus(name="x", goal_congruence=0.6, intensity=0.8), "rng_seed": 1},
        config={"configurable": {"thread_id": "e2e-model"}},
    )
    facs = result["expression"]["spontaneous"]["facs_au"]
    # ExpressionDecoder 产出全部 5 个 AU 键；解析占位仅 2~3 个 → 证明模型在管线中生效
    assert set(facs) == FULL_AU


async def test_no_decoder_uses_placeholder() -> None:
    graph = make_graph()
    result = await graph.ainvoke(
        {"stimulus": Stimulus(name="x", goal_congruence=0.6, intensity=0.8), "rng_seed": 1},
        config={"configurable": {"thread_id": "e2e-placeholder"}},
    )
    facs = result["expression"]["spontaneous"]["facs_au"]
    # 正效价占位分支：含 AU12，不含 AU04 → 与真模型可区分
    assert "AU12" in facs
    assert "AU04" not in facs
