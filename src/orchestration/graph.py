"""StateGraph 装配：Worker（含 MemoryRecall/Mood）+ Supervisor + Checkpointer。

依赖注入 memory/checkpointer，编排层经 memory API 访问记忆、经 checkpointer
持久化运行态。条件边路由函数 route_after_affect_core 独立、可单测。
入口统一从编译后的 graph 进，不直接调内部节点。
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.affect_core import AffectCoreAgent
from src.agents.appraisal import AppraisalAgent
from src.agents.expression import ChannelDecoder, ExpressionAgent
from src.agents.mood import MoodAgent
from src.agents.perception import PerceptionAgent
from src.agents.regulation import RegulationAgent
from src.agents.value import ValueAgent
from src.memory.client import MemoryClient
from src.orchestration.memory_recall import MemoryRecallAgent
from src.orchestration.state import AffectState
from src.orchestration.supervisor import SupervisorAgent


def route_after_affect_core(state: AffectState) -> str:
    """条件边路由：开启调控走 regulation，否则直达 expression。可独立单测。"""
    return "regulation" if state.regulation_enabled else "expression"


def build_graph(
    checkpointer: BaseCheckpointSaver,
    memory: MemoryClient,
    *,
    expression_decoder: ChannelDecoder | None = None,
) -> CompiledStateGraph:
    """装配并编译情感表达 StateGraph。

    管线：memory_recall → perception → appraisal → value → affect_core → mood
    →（条件边）regulation/expression → supervisor。其中 memory_recall / mood 分别由
    `recall_enabled` / `mood_enabled` 门控，关闭时为 no-op（默认 v1 行为不变）。
    expression_decoder：可选注入的真通道解码器（鸭子类型，编排层不依赖 torch）。
    """
    graph: StateGraph = StateGraph(AffectState)
    graph.add_node("memory_recall", MemoryRecallAgent(memory))
    graph.add_node("perception", PerceptionAgent())
    graph.add_node("appraisal", AppraisalAgent())
    graph.add_node("value", ValueAgent())
    graph.add_node("affect_core", AffectCoreAgent())
    graph.add_node("mood", MoodAgent())
    graph.add_node("regulation", RegulationAgent())
    graph.add_node("expression", ExpressionAgent(decoder=expression_decoder))
    graph.add_node("supervisor", SupervisorAgent(memory))

    graph.add_edge(START, "memory_recall")
    graph.add_edge("memory_recall", "perception")
    graph.add_edge("perception", "appraisal")
    graph.add_edge("appraisal", "value")
    graph.add_edge("value", "affect_core")
    graph.add_edge("affect_core", "mood")
    graph.add_conditional_edges(
        "mood",
        route_after_affect_core,
        {"regulation": "regulation", "expression": "expression"},
    )
    graph.add_edge("regulation", "expression")
    graph.add_edge("expression", "supervisor")
    graph.add_edge("supervisor", END)

    return graph.compile(checkpointer=checkpointer)
