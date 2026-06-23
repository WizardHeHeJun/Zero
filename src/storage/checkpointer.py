"""运行态持久化：第一版用 LangGraph 内存 Saver，接口对齐未来 Postgres。

运行态（V(s)、当前后验、AffectState）经此持久化；长期记忆走图谱后端，
二者分离，不互混（见 memory-rules.md #3）。

存储层为最底层、对上层无感知：从 checkpoint 恢复的自定义类型白名单由
上层（编排层）通过 allowed_types 传入，本模块不硬编码任何上层类型名。
"""

from __future__ import annotations

from collections.abc import Iterable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


def build_checkpointer(
    allowed_types: Iterable[tuple[str, ...]] | None = None,
) -> BaseCheckpointSaver:
    """构造运行态 Checkpointer。

    allowed_types：白名单从 checkpoint 反序列化的自定义类型 (module, qualname)，
    由编排层提供。第一版返回内存 Saver（占位）；后续替换 Postgres saver 签名不变。
    """
    if allowed_types is None:
        return InMemorySaver()
    serde = JsonPlusSerializer(allowed_msgpack_modules=list(allowed_types))
    return InMemorySaver(serde=serde)
