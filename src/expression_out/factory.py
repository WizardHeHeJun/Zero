"""按 env 装配表现层出口。

**这是全仓唯一把「协议」与「具体表现形式」绑在一起的地方**——`ChatDriver` 只认协议，
加新表现形式（灯光/…）时只需在这里多一个分支，对话核心一行不动。

全部默认关：未配任何 `ZERO_*_SINK` 时返回空列表 ⇒ 不表现 ⇒ 逐字零回归。

**连接共享**：皮套与语音都往渲染端投参数，而 VTS 参数注入单进程独占——两个 sink
必须共用一条 `VtsTransport`（`transport.connect` 幂等，谁先 connect 谁真建链）。
"""

from __future__ import annotations

import logging

from src.expression_out.base import ExpressionSink

logger = logging.getLogger(__name__)


def build_expression_sinks() -> list[ExpressionSink]:
    """从 env 装配全部已开启的表现出口（默认空=不表现）。

    ⚠ 只**构造**不连接：真实连接属 async I/O，由入口在事件循环里逐个
    `await sink.connect()`（构造期做 I/O 会让工厂无法在同步上下文用）。
    ⚠ 具体实现走函数内局部 import：模块加载期不拉 `src.agents.*` 依赖链。
    """
    sinks: list[ExpressionSink] = []
    from src.expression_out.speech import build_speech_sink
    from src.expression_out.transport import VtsTransport
    from src.expression_out.vts import build_vts_sink

    # 共享一条渲染端连接（构造零 I/O，任何 sink 都没开时它只是个惰性壳）。
    transport = VtsTransport()
    vts = build_vts_sink(transport)
    if vts is not None:
        sinks.append(vts)
        logger.info("表现层：皮套出口已装配（ZERO_VTS_SINK）")
    speech = build_speech_sink(transport)
    if speech is not None:
        sinks.append(speech)
        logger.info("表现层：语音出口已装配（ZERO_TTS_SINK，与皮套共享渲染端连接）")
    return sinks
