"""研究原型运行入口：跑 stimulus 序列，产出 (v,a) 轨迹与逐步中间量。

带 thread_id 走 Checkpointer，运行态（含 value_table）跨刺激持久化，
从而可观测 ValueAgent 的在线学习。轨迹可 dump 成 JSON 供观测。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from src.agents.expression import ChannelDecoder
from src.agents.language import LanguageModel
from src.memory.client import MemoryClient
from src.orchestration.graph import build_graph
from src.orchestration.state import AffectState, Stimulus
from src.storage.checkpointer import build_checkpointer
from src.storage.graph_store import build_graph_store, build_semantic_store

logger = logging.getLogger(__name__)

# 编排层声明自己放进运行态、需从 checkpoint 恢复的自定义类型（存储层据此白名单）。
# Fact：AffectState.recalled_facts 携带（D1）；非 InMemory checkpointer（sqlite/postgres）
# 反序列化须白名单，否则还原成 dict 致下游 f.sim/f.content AttributeError。
ALLOWED_CHECKPOINT_TYPES = [
    ("src.orchestration.state", "Stimulus"),
    ("src.memory.types", "Fact"),
    ("src.memory.types", "Scope"),  # Fact.scope 是 Scope 枚举，随 Fact 一同反序列化需白名单
]


def _state_to_entry(stim_name: str, state: AffectState) -> dict[str, Any]:
    """把一次运行后的 AffectState 抽成可观测轨迹条目（run 与 ConversationSession 共用）。"""
    return {
        "stimulus": stim_name,
        "valence_arousal": state.affect_sample,
        "ignited_streams": state.ignited_streams,
        "affect_precision": state.affect_precision,
        "prior_mu": state.prior_mu,
        "prior_sigma": state.prior_sigma,
        "reward": state.reward,
        "rpe": state.rpe,
        "precision": state.precision,
        "value_estimate": state.value_estimate,
        "mood": state.mood,
        "recalled_context": state.recalled_context,
        "recalled_facts": state.recalled_facts,  # D1：供 ChatDriver 按 importance 注入 history
        "language_text": state.language_text,
        "language_affect": state.language_affect,
        "language_consistency": state.language_consistency,
        "language_iter": state.language_iter,
        "expression": state.expression,
    }


async def run(
    stimuli: Sequence[Stimulus],
    *,
    thread_id: str,
    memory: MemoryClient | None = None,
    session_id: str | None = None,
    user_id: str = "default-user",
    group_id: str = "default-group",
    regulation_enabled: bool = False,
    regulation_strategy: str = "suppression",
    mood_enabled: bool = False,
    recall_enabled: bool = False,
    language_enabled: bool = False,
    workspace_enabled: bool = False,
    appraisal_conditioning_enabled: bool = False,
    language_max_iters: int = 3,
    rng_seed: int | None = None,
    sample_sigma_cap: float | None = None,
    expression_decoder: ChannelDecoder | None = None,
    language_model: LanguageModel | None = None,
) -> list[dict[str, Any]]:
    """按序跑 stimuli，返回每个刺激的情绪向量与关键中间量。

    session_id 默认绑定到 thread_id，使不同线程的会话记忆天然隔离（防串味）；
    user_id/group_id 应由调用方按真实身份显式传入。
    expression_decoder：可选注入训练好的真通道解码器，走真网络表达。
    language_model：可选注入的语言模型（鸭子类型），开启 language_enabled 后驱动
    affect↔language 双向收敛回路；未注入则用占位模板模型。
    """
    client = (
        memory
        if memory is not None
        else MemoryClient(build_graph_store(), semantic=build_semantic_store())
    )
    session = session_id if session_id is not None else thread_id
    checkpointer = build_checkpointer(ALLOWED_CHECKPOINT_TYPES)
    graph = build_graph(
        checkpointer=checkpointer,
        memory=client,
        expression_decoder=expression_decoder,
        language_model=language_model,
    )

    trajectory: list[dict[str, Any]] = []
    for stim in stimuli:
        result = await graph.ainvoke(
            {
                "stimulus": stim,
                "session_id": session,
                "user_id": user_id,
                "group_id": group_id,
                "regulation_enabled": regulation_enabled,
                "regulation_strategy": regulation_strategy,
                "mood_enabled": mood_enabled,
                "recall_enabled": recall_enabled,
                "language_enabled": language_enabled,
                "workspace_enabled": workspace_enabled,
                "appraisal_conditioning_enabled": appraisal_conditioning_enabled,
                "language_max_iters": language_max_iters,
                "rng_seed": rng_seed,
                "sample_sigma_cap": sample_sigma_cap,
                "task_complete": False,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        # 本 langgraph 版本对 pydantic schema 的 ainvoke 返回 dict；AffectState(**result)
        # 会触发 pydantic 校验（字段缺失/类型不符即报错），不会静默掩盖问题。
        state = result if isinstance(result, AffectState) else AffectState(**result)
        logger.debug(
            "runner stim=%s e*=%s precision=%s",
            stim.name,
            state.affect_sample,
            getattr(state, "affect_precision", None),
        )
        trajectory.append(_state_to_entry(stim.name, state))
    return trajectory


def dump_trajectory(trajectory: list[dict[str, Any]]) -> str:
    """把轨迹序列化为 JSON 文本（研究观测用）。"""
    return json.dumps(trajectory, ensure_ascii=False, indent=2, default=str)


class ConversationSession:
    """多轮交互会话：建图/checkpointer **一次**，每轮 `step` 喂一个 stimulus。

    与 `run`（一次性跑完整序列）不同，本类把同一 checkpointer 留在内存中，逐轮 `ainvoke`
    复用同一 `thread_id` —— 故运行态（mood 心境、value_table 价值记忆）**跨轮持久**，
    情绪轨迹形成历史依赖（A.7 滞后在对话里显现）。供交互式对话入口（`main.py --chat`）使用：
    把"用户每句话"评价成 stimulus 喂进来，引擎演化的 e* 驱动语言层生成带情绪的回应。

    门控开关在构造时固定、贯穿整个会话；记忆/语言模型可注入（鸭子类型，编排层不依赖 openai）。
    """

    def __init__(
        self,
        *,
        thread_id: str,
        memory: MemoryClient | None = None,
        session_id: str | None = None,
        user_id: str = "default-user",
        group_id: str = "default-group",
        regulation_enabled: bool = False,
        regulation_strategy: str = "suppression",
        mood_enabled: bool = False,
        recall_enabled: bool = False,
        language_enabled: bool = False,
        workspace_enabled: bool = False,
        appraisal_conditioning_enabled: bool = False,
        language_max_iters: int = 3,
        rng_seed: int | None = None,
        sample_sigma_cap: float | None = None,
        expression_decoder: ChannelDecoder | None = None,
        language_model: LanguageModel | None = None,
    ) -> None:
        client = (
            memory
            if memory is not None
            else MemoryClient(build_graph_store(), semantic=build_semantic_store())
        )
        self.thread_id = thread_id
        self.session_id = session_id if session_id is not None else thread_id
        self.user_id = user_id
        self.group_id = group_id
        self.checkpointer = build_checkpointer(ALLOWED_CHECKPOINT_TYPES)
        self.graph = build_graph(
            checkpointer=self.checkpointer,
            memory=client,
            expression_decoder=expression_decoder,
            language_model=language_model,
        )
        self.flags = {
            "regulation_enabled": regulation_enabled,
            "regulation_strategy": regulation_strategy,
            "mood_enabled": mood_enabled,
            "recall_enabled": recall_enabled,
            "language_enabled": language_enabled,
            "workspace_enabled": workspace_enabled,
            "appraisal_conditioning_enabled": appraisal_conditioning_enabled,
            "language_max_iters": language_max_iters,
            "rng_seed": rng_seed,
            "sample_sigma_cap": sample_sigma_cap,
        }

    async def step(self, stim: Stimulus) -> dict[str, Any]:
        """跑一轮：喂一个 stimulus，返回该轮的可观测轨迹条目（含 e*、mood、生成语言）。"""
        result = await self.graph.ainvoke(
            {
                "stimulus": stim,
                "session_id": self.session_id,
                "user_id": self.user_id,
                "group_id": self.group_id,
                "task_complete": False,
                **self.flags,
            },
            config={"configurable": {"thread_id": self.thread_id}},
        )
        state = result if isinstance(result, AffectState) else AffectState(**result)
        return _state_to_entry(stim.name, state)
