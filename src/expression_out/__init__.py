"""表现层出口（边界适配层，三层之外——同 `mcp_server` 的定位）。

## 这一层解决什么

对话产生情绪是主链路；**把情绪表现出来**有很多种形式——皮套动作、语音、灯光、
纯文字标签——它们都是同一份情绪状态的不同**表现**，不是不同的情绪。所以这里定义
一个统一出口协议 `ExpressionSink`，`ChatDriver` 每轮把「本轮情绪 + 回复」交给它，
具体表现形式由注入的实现决定。

    对话（ChatDriver：LLM ⊗ 情感引擎）
        └── emit(ExpressionFrame) ──▶ ExpressionSink
                                       ├── VtsSink   皮套（连续轨迹 + 离散行为）
                                       └── ...       将来的其它表现形式

## 边界纪律

- **`ChatDriver` 只认协议**（鸭子类型注入），不 import 任何具体实现——换表现形式、
  加表现形式都不动对话核心，也不让 `orchestration` 依赖 MCP/WebSocket 这些外部 I/O。
- **表现失败不得扳倒对话**：渲染端断线、超时、拒收都只降级记日志（见各实现的 `emit`）。
  情绪与记忆是主链路，表现是下游。
- **不在情感热路径**：`emit` 发生在 `ChatDriver.step` 末尾（一轮对话完成后），
  不在 `affect_core` 的确定性数值通路里，不影响给定 seed 的可复现性。
"""

from src.expression_out.base import ExpressionFrame, ExpressionSink

__all__ = ["ExpressionFrame", "ExpressionSink"]
