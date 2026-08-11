"""表现层出口的协议与数据结构（纯定义，无 I/O、无外部依赖）。

放在 `orchestration` 之外是有意的：对话层只依赖本模块的**协议**，具体实现
（VTS/TTS/…）各自带自己的外部依赖，谁都不进对话核心的 import 图。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ExpressionFrame:
    """一轮对话产生的**待表现内容**：情绪状态 + 说了什么。

    刻意只带表现所需的量，不搬运整份对话状态（history/记忆/态度轨迹都不进来）——
    表现端不该看到、也不需要看到那些。

    Attributes:
        emotion: 本轮快变情绪 (valence, arousal)，各维 ∈ [-1, 1]。表现的主驱动量。
        emotion_label: 情绪词（"欣喜"/"平静"…），供文本类表现形式直接用。
        reply: 本轮回复原文。动作层从它抽第③层离散行为（点头/摇头等语义驱动动作）；
            语音类表现形式则拿它去合成。
        regulated: 经调节后的表达侧情绪（未开调节时 None）。双通路表现用：
            自发通路走 `emotion`，随意通路走本字段（Rinn 1984 双通路）。
        channels: `ExpressionAgent` 解码出的通道值（FACS AU / 韵律 / 生理…），
            没有则空 dict。表现端按自己支持的键取用，不认识的忽略。
    """

    emotion: tuple[float, float]
    emotion_label: str
    reply: str
    regulated: tuple[float, float] | None = None
    channels: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ExpressionSink(Protocol):
    """表现层出口协议。实现方负责把一帧表现内容落到某种具体形式上。

    契约：
    - `emit` **不得抛异常**打断对话——表现失败自行降级（记日志/静默丢弃）。
      对话与情绪是主链路，表现是下游，下游故障不回灌上游。
    - `emit` 应尽快返回；需要长时间播放的（如一段动作轨迹）应投递后即返回，
      由实现自己的后台循环消化。
    - `aclose` 幂等：重复调用、未连接时调用都不报错。

    ## 🛑 什么该进这个协议，什么不该（2026-08-11 定，接设备控制前必读）

    **该进**：皮套动作、语音（TTS）、灯光氛围、纯文字情绪标签——它们是**同一份情绪
    状态的不同表现形式**，共同特征是「失败了就当没表现，世界没有被改变」。

    **不该进**：开关灯、发消息、点鼠标这类**设备/环境控制**。它们看起来也是"数字人做了
    动作"，但性质不同——那是**改变外部世界**，失败不能静默吞（用户以为灯关了但没关，
    比没有这个功能更糟），通常还要回执、重试、权限确认。将来接这类能力时另立
    `ActionSink`（行动出口）：语义是「意图执行」，契约要求返回回执、允许抛错、
    可能需要确认环节。

    判据一句话：**表现失败可以静默降级，行动失败必须让人知道**——两者契约天生相反，
    塞进同一个协议会逼着实现方在"吞掉异常"和"打断对话"之间二选一，两边都错。
    """

    async def connect(self) -> bool:
        """建立与表现端的连接并启动后台任务；返回是否可用。

        **必须实现，即使不需要连接**——不需要外部连接的表现形式（纯文字标签、写文件）
        直接 `return True` 即可。放进协议是因为入口要对每个 sink 统一调用它：
        写成"VtsSink 独有的扩展方法"会让入口只能靠鸭子类型硬调，加一个没有该方法的
        实现就 `AttributeError`（code-reviewer 2026-08-11 实证，`mypy main.py` 可复现）。

        约定与 `emit` 一致：**不抛异常**，连不上就返回 False 并自行降级（调用方据此
        决定是否继续无表现地跑）。
        """
        ...

    async def emit(self, frame: ExpressionFrame) -> None:
        """把一帧表现内容送出去。"""
        ...

    async def aclose(self) -> None:
        """释放连接/后台任务（幂等）。"""
        ...
