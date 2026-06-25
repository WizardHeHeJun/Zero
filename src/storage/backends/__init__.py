"""存储层后端实现子包：确定性图谱（deterministic）+ 语义侧信道（semantic）。

`src/storage/graph_store.py` 是稳定门面：再导出本子包的 store 类 + 协议，并持有工厂
（`build_graph_store`/`build_semantic_store` 及其 `_neo4j_store`/`_graphiti_store` 探测）。
上层（记忆层/编排层）一律从 `graph_store` 导入，不直接依赖本子包路径——便于后续增删后端。
"""
