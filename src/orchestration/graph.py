"""StateGraph 装配：Worker（含 MemoryRecall/Mood/Language）+ Supervisor + Checkpointer。

依赖注入 memory/checkpointer，编排层经 memory API 访问记忆、经 checkpointer
持久化运行态。条件边路由函数 route_after_mood / route_after_language 独立、可单测。
入口统一从编译后的 graph 进，不直接调内部节点。
"""

from __future__ import annotations

import time
from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.affect_core import AffectCoreAgent
from src.agents.affect_math import LANGUAGE_TOLERANCE
from src.agents.appraisal import AppraisalAgent
from src.agents.expression import ChannelDecoder, ExpressionAgent
from src.agents.language import LanguageAgent, LanguageModel
from src.agents.mood import MoodAgent
from src.agents.perception import PerceptionAgent
from src.agents.regulation import RegulationAgent
from src.agents.value import ValueAgent
from src.memory.client import MemoryClient
from src.orchestration.memory_recall import MemoryRecallAgent
from src.orchestration.state import AffectState
from src.orchestration.supervisor import SupervisorAgent


def route_after_mood(state: AffectState) -> str:
    """mood 后路由：优先进语言回路，其次调控，否则直达表达。可独立单测。

    language_enabled=False 时退化为原 regulation/expression 二选一（零回归）。
    """
    if state.language_enabled:
        return "language"
    return "regulation" if state.regulation_enabled else "expression"


def route_after_language(state: AffectState) -> str:
    """语言回路路由：不一致且未达上限则回 language，否则进下游。可独立单测。

    收敛（dist<=τ 或无观测）或达 language_max_iters 上限 → 进下游（regulation/expression）。
    """
    dist = state.language_consistency
    converged = dist is None or dist <= LANGUAGE_TOLERANCE
    if converged or state.language_iter >= state.language_max_iters:
        return "regulation" if state.regulation_enabled else "expression"
    return "language"


def build_graph(
    checkpointer: BaseCheckpointSaver,
    memory: MemoryClient,
    *,
    expression_decoder: ChannelDecoder | None = None,
    language_model: LanguageModel | None = None,
    now_fn: Callable[[], float] = time.time,
) -> CompiledStateGraph:
    """装配并编译情感表达 StateGraph。

    管线：memory_recall → perception → appraisal → value → affect_core → mood
    →（条件边）language ⇄ regulation/expression → supervisor。其中 memory_recall /
    mood / language 分别由 `recall_enabled` / `mood_enabled` / `language_enabled` 门控，
    关闭时为 no-op（默认 v1 行为不变）。language 开启时进入 affect↔language 双向收敛
    回路（带 `language_max_iters` 终止上限）。
    expression_decoder / language_model：可选注入的真通道解码器/语言模型（鸭子类型，
    编排层不依赖 torch / anthropic）。
    now_fn：时钟注入（默认 time.time）。供 AppraisalAgent 的 HPA 皮质醇更新节点读取
    wall-clock，纯函数 cortisol_step 不触碰时钟（P3 1-B CS 红线）；测试可注入确定性时钟。
    """
    graph: StateGraph = StateGraph(AffectState)
    graph.add_node("memory_recall", MemoryRecallAgent(memory))
    graph.add_node("perception", PerceptionAgent())
    graph.add_node("appraisal", AppraisalAgent(now_fn=now_fn))
    graph.add_node("value", ValueAgent())
    graph.add_node("affect_core", AffectCoreAgent())
    graph.add_node("mood", MoodAgent())
    graph.add_node("language", LanguageAgent(model=language_model))
    graph.add_node("regulation", RegulationAgent())
    graph.add_node("expression", ExpressionAgent(decoder=expression_decoder))
    graph.add_node("supervisor", SupervisorAgent(memory))

    graph.add_edge(START, "memory_recall")
    graph.add_edge("memory_recall", "perception")
    graph.add_edge("perception", "appraisal")
    graph.add_edge("appraisal", "value")
    graph.add_edge("value", "affect_core")
    graph.add_edge("affect_core", "mood")
    # 门控分支：language（双向回路）/ regulation（掩饰）/ expression（直达）三选一；
    # language 节点自身再经 route_after_language 决定回路或进下游。两处用 inline 字面量，
    # 让 mypy 按参数期望类型推断为 dict[Hashable, str]（dict 键 invariant，提取变量会失配）。
    graph.add_conditional_edges(
        "mood",
        route_after_mood,
        {"language": "language", "regulation": "regulation", "expression": "expression"},
    )
    graph.add_conditional_edges(
        "language",
        route_after_language,
        {"language": "language", "regulation": "regulation", "expression": "expression"},
    )
    graph.add_edge("regulation", "expression")
    graph.add_edge("expression", "supervisor")
    graph.add_edge("supervisor", END)

    return graph.compile(checkpointer=checkpointer)
