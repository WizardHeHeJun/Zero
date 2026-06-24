# 情感表达多 Agent 系统 — 容器镜像（后续服务器部署用）。
# 本机无 Docker、未在本地构建；compose 提供 postgres/neo4j，app 经 env 接真后端。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 运行依赖（核心 + db 真后端驱动）。ml extra（torch 等）按需在镜像里再装。
RUN pip install --upgrade pip && pip install \
    "langgraph>=0.2" "pydantic>=2" \
    "langgraph-checkpoint-sqlite" "langgraph-checkpoint-postgres" "psycopg[binary]" "neo4j"

COPY src/ ./src/
COPY scripts/ ./scripts/
RUN mkdir -p /app/data

# 默认本地落盘后端；compose 里覆盖为 postgres
ENV ZERO_CHECKPOINT_BACKEND=sqlite \
    ZERO_CHECKPOINT_DB=/app/data/checkpoints.sqlite3 \
    ZERO_MEMORY_BACKEND=sqlite \
    ZERO_GRAPH_DB=/app/data/graph.sqlite3

# 占位入口：跑核心管线冒烟（不依赖 torch）；按需替换为真实服务入口。
CMD ["python", "-c", "import asyncio; from src.orchestration.runner import run; from src.orchestration.state import Stimulus; print(asyncio.run(run([Stimulus(name='boot', goal_congruence=0.5, intensity=0.7)], thread_id='boot', rng_seed=0)))"]
