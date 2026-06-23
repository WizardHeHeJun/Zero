"""编排层共享状态：Stimulus 输入与 AffectState。

AffectState 用 pydantic 定义结构化 state；节点只返回增量字段（见 orchestration-rules.md）。
state 不放大对象（向量/文档）；trace 仅存标量中间量。运行态字段（value_table、
后验、采样点）由 Checkpointer 持久化，不写入长期记忆图谱。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


class Stimulus(BaseModel):
    """一个待评价的事件（OCC 评价输入）。各评价维度取值约定在 [-1, 1]。

    用 pydantic 模型以便被 LangGraph Checkpointer 原生序列化。构造用关键字参数。
    """

    name: str
    goal_congruence: float = 0.0  # 与目标的一致性（事件维度）
    standard_compliance: float = 0.0  # 与标准的契合（行为维度）
    attitude_appeal: float = 0.0  # 对象的喜好（吸引力维度）
    intensity: float = 1.0  # 事件显著度/强度


class AffectState(BaseModel):
    """情感流水线全程共享的状态。节点只返回增量字段。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 输入
    stimulus: Stimulus | None = None

    # Perception
    features: list[float] = Field(default_factory=list)

    # Appraisal（OCC 理性先验）
    appraisal: dict[str, float] = Field(default_factory=dict)
    prior_mu: tuple[float, float] | None = None
    prior_sigma: tuple[float, float] | None = None
    reward: float | None = None

    # Value（在线 TD，价值/精度辅佐）—— value_table 是运行态
    value_table: dict[str, float] = Field(default_factory=dict)
    value_estimate: float | None = None
    rpe: float | None = None
    precision: float | None = None

    # AffectCore（主动推断，后验 + 采样）
    post_mu: tuple[float, float] | None = None
    post_sigma: tuple[float, float] | None = None
    affect_sample: tuple[float, float] | None = None  # e*（随机性来源）

    # Regulation / Expression（双通路·多通道）
    regulated_affect: tuple[float, float] | None = None
    expression: dict[str, Any] = Field(default_factory=dict)

    # 观测与作用域：trace 用 reducer 累加，节点只需返回自己的 [entry]（避免每步全量拷贝）
    trace: Annotated[list[dict[str, Any]], operator.add] = Field(default_factory=list)
    session_id: str = "default-session"
    user_id: str = "default-user"
    group_id: str = "default-group"

    # 控制开关
    regulation_enabled: bool = False  # 开启掩饰/再评价（双通路对比）
    rng_seed: int | None = None  # 采样可控（测试用）
    task_complete: bool = False
