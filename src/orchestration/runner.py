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
from src.memory.client import MemoryClient
from src.orchestration.graph import build_graph
from src.orchestration.state import AffectState, Stimulus
from src.storage.checkpointer import build_checkpointer

logger = logging.getLogger(__name__)

# 编排层声明自己放进运行态、需从 checkpoint 恢复的自定义类型（存储层据此白名单）
ALLOWED_CHECKPOINT_TYPES = [("src.orchestration.state", "Stimulus")]


async def run(
    stimuli: Sequence[Stimulus],
    *,
    thread_id: str,
    memory: MemoryClient | None = None,
    session_id: str | None = None,
    user_id: str = "default-user",
    group_id: str = "default-group",
    regulation_enabled: bool = False,
    rng_seed: int | None = None,
    expression_decoder: ChannelDecoder | None = None,
) -> list[dict[str, Any]]:
    """按序跑 stimuli，返回每个刺激的情绪向量与关键中间量。

    session_id 默认绑定到 thread_id，使不同线程的会话记忆天然隔离（防串味）；
    user_id/group_id 应由调用方按真实身份显式传入。
    expression_decoder：可选注入训练好的真通道解码器，走真网络表达。
    """
    client = memory if memory is not None else MemoryClient()
    session = session_id if session_id is not None else thread_id
    checkpointer = build_checkpointer(ALLOWED_CHECKPOINT_TYPES)
    graph = build_graph(
        checkpointer=checkpointer, memory=client, expression_decoder=expression_decoder
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
                "rng_seed": rng_seed,
                "task_complete": False,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        # 本 langgraph 版本对 pydantic schema 的 ainvoke 返回 dict；AffectState(**result)
        # 会触发 pydantic 校验（字段缺失/类型不符即报错），不会静默掩盖问题。
        state = result if isinstance(result, AffectState) else AffectState(**result)
        trajectory.append(
            {
                "stimulus": stim.name,
                "valence_arousal": state.affect_sample,
                "prior_mu": state.prior_mu,
                "prior_sigma": state.prior_sigma,
                "reward": state.reward,
                "rpe": state.rpe,
                "precision": state.precision,
                "value_estimate": state.value_estimate,
                "expression": state.expression,
            }
        )
    return trajectory


def dump_trajectory(trajectory: list[dict[str, Any]]) -> str:
    """把轨迹序列化为 JSON 文本（研究观测用）。"""
    return json.dumps(trajectory, ensure_ascii=False, indent=2, default=str)
