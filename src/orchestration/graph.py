"""StateGraph 装配：6 个 Worker + Supervisor + Checkpointer。

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
from src.agents.expression import ExpressionAgent
from src.agents.perception import PerceptionAgent
from src.agents.regulation import RegulationAgent
from src.agents.value import ValueAgent
from src.memory.client import MemoryClient
from src.orchestration.state import AffectState
from src.orchestration.supervisor import SupervisorAgent


def route_after_affect_core(state: AffectState) -> str:
    """条件边路由：开启调控走 regulation，否则直达 expression。可独立单测。"""
    return "regulation" if state.regulation_enabled else "expression"


def build_graph(checkpointer: BaseCheckpointSaver, memory: MemoryClient) -> CompiledStateGraph:
    """装配并编译情感表达 StateGraph。"""
    graph: StateGraph = StateGraph(AffectState)
    graph.add_node("perception", PerceptionAgent())
    graph.add_node("appraisal", AppraisalAgent())
    graph.add_node("value", ValueAgent())
    graph.add_node("affect_core", AffectCoreAgent())
    graph.add_node("regulation", RegulationAgent())
    graph.add_node("expression", ExpressionAgent())
    graph.add_node("supervisor", SupervisorAgent(memory))

    graph.add_edge(START, "perception")
    graph.add_edge("perception", "appraisal")
    graph.add_edge("appraisal", "value")
    graph.add_edge("value", "affect_core")
    graph.add_conditional_edges(
        "affect_core",
        route_after_affect_core,
        {"regulation": "regulation", "expression": "expression"},
    )
    graph.add_edge("regulation", "expression")
    graph.add_edge("expression", "supervisor")
    graph.add_edge("supervisor", END)

    return graph.compile(checkpointer=checkpointer)
