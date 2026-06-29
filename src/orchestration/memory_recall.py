"""MemoryRecallAgent：长期情绪倾向回灌（记忆读闭环）。

在管线开头读 `user` 作用域的长期情绪倾向（由 Supervisor 在任务完成时写入），
解析出标量 disposition 放进 state，供 AppraisalAgent 偏置先验——让记忆层真正被
『用上』（此前仅写不读，闭合 读↔写 回路）。注入 MemoryClient，不直连图谱。
`recall_enabled` 关闭或无记忆时为 no-op（严格零回归）。
节点契约：(state) -> dict，只返回增量。
"""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime

from src.agents.affect_math import text_label
from src.memory.client import MemoryClient
from src.memory.types import Fact, Scope
from src.orchestration.state import AffectState

logger = logging.getLogger(__name__)


def _parse_disposition(facts: list[Fact]) -> float | None:
    """从最新一条 disposition 事实解析 value 标量（写入格式见 SupervisorAgent）。"""
    if not facts:
        return None
    content = facts[-1].content  # "disposition stimulus=<name> value=<float>"
    marker = "value="
    idx = content.rfind(marker)
    if idx < 0:
        return None
    try:
        return float(content[idx + len(marker) :].split()[0])
    except ValueError:
        logger.debug("disposition value 解析失败，content=%.80s", content)
        return None


def parse_importance(content: str) -> float:
    """从 episode 文本解析写入时显著度 `precision=`（importance 维度）；缺失返回 0.5 保守默认。

    对应 SupervisorAgent 固化的写入格式 `gist | ... | precision=<float> | ...`。取**最后一个**
    precision= 匹配——结构化元数据字段在尾部、用户原话/语言在前段，避免用户输入里的
    `precision=0.99` 污染评分（WARN-1）。纯正则、无 LLM（守确定性热路径 BLOCK-1）。
    返回 0 会让 γ 维恒失效，故缺失取 0.5 中性值。
    """
    matches = re.findall(r"precision=([0-9]+(?:\.[0-9]+)?)", content)
    if not matches:
        return 0.5
    try:
        return float(matches[-1])
    except ValueError:
        return 0.5


def normalized_importance(content: str) -> float:
    """把无界的写入精度（affect_precision=方差倒数，实测 ~28–72）归一到 (0,1)，供三维打分/注入门。

    用 Hill 饱和 `p/(p+C)`（与 Kalman 增益/逆方差加权同构、单调有界、边际递减；数学席 D8）：
    与 `sim∈[0,1]`、`recency=Δt^(-d)∈(0,1]` 同量纲，α/β/γ 等权才恢复语义；否则原始 precision
    几十倍碾压另两维（dogfood 实测）。C 走 env `ZERO_RECALL_IMPORTANCE_SCALE`（默认 30，匹配实测
    量级使 INJECT_MIN=0.5 成为「高质量门」）。纯数值、无 LLM。C 固定（非自适应集合统计）以守
    可复现/确定性（CS 席 D8 红线）。precision 字段缺失时 parse 返 0.5 → 归一后 ~低位（未知不优先）。
    """
    p = parse_importance(content)
    c = float(os.getenv("ZERO_RECALL_IMPORTANCE_SCALE", "30"))
    return p / (p + c) if (p + c) > 0 else 0.0


def _rank_episodes(facts: list[Fact], now: datetime, *, arousal: float = 0.0) -> list[Fact]:
    """三维加权和重排（D3）：`score = α·Δt^(-d) + β·sim + γ_eff·importance`。

    - recency：`Δt = max(1.0, (now-valid_at)/天)`，幂律 `Δt^(-d)`（Wixted&Ebbesen 1991，
      非指数——指数会系统性高估久远记忆可及性）。
    - relevance：`fact.sim`（D4 透传的余弦；确定性后端为 0.0，该维自然退化）。
    - importance：写入时 precision 显著度（`_parse_importance`）；命中 `first_contact=True`
      时 ×1.2（D5 首因加权，对应系列位置效应 primacy）。
    加权和而非乘积（任一维为 0 不归零）；**禁 rpe 当第四维**（salience 已含 |rpe|，防 double
    counting）。`γ_eff = γ·(1+0.5·clamp(arousal,0,1))` 仅当 `ZERO_RECALL_AROUSAL_MOD` 开启
    （默认关，唤醒做 NE 调制代理）。纯数值无 LLM；返回按 score 降序的新列表，不改元素。
    """
    if not facts:
        return []
    d = float(os.getenv("ZERO_RECALL_DECAY_D", "0.5"))
    alpha = float(os.getenv("ZERO_RECALL_ALPHA", "0.33"))
    beta = float(os.getenv("ZERO_RECALL_BETA", "0.34"))
    gamma = float(os.getenv("ZERO_RECALL_GAMMA", "0.33"))
    if os.getenv("ZERO_RECALL_AROUSAL_MOD", "0").lower() not in ("0", "", "false"):
        gamma = gamma * (1.0 + 0.5 * max(0.0, min(1.0, arousal)))

    def score(fact: Fact) -> float:
        valid_at = fact.valid_at if fact.valid_at.tzinfo else fact.valid_at.replace(tzinfo=UTC)
        delta_days = max(1.0, (now - valid_at).total_seconds() / 86400.0)
        recency = delta_days ** (-d)
        importance = normalized_importance(fact.content)  # D8：Hill 归一到 (0,1)，量纲对齐
        if "first_contact=True" in fact.content:
            importance *= 1.2
        return alpha * recency + beta * fact.sim + gamma * importance

    return sorted(facts, key=score, reverse=True)


class MemoryRecallAgent:
    """读 user 长期倾向 → recalled_disposition（偏置 appraisal）。注入 client，不直连图谱。"""

    def __init__(self, memory: MemoryClient) -> None:
        self.memory = memory

    async def __call__(self, state: AffectState) -> dict:
        if not state.recall_enabled:
            return {}
        out: dict = {}
        entry: dict = {"node": "memory_recall"}

        # 确定性标量 disposition（偏置 appraisal）——不变
        facts = await self.memory.query("disposition", scope=Scope.USER, key=state.user_id)
        disposition = _parse_disposition(facts)
        if disposition is not None:
            out["recalled_disposition"] = disposition
            entry["recalled_disposition"] = disposition

        # 语义召回（语义记忆侧信道）→ recalled_context 喂语言层检索；
        # 无语义后端时 recall 返回 []，整段自动 no-op（严格零回归）。
        # B-7：mood 非 None 时拼情绪线索，提升召回相关性；否则退化为 stimulus.name（零回归）。
        stim_name = state.stimulus.name if state.stimulus is not None else "disposition"
        if state.mood is not None:
            query = f"{stim_name} {text_label(state.mood[0], state.mood[1])}"
        else:
            query = stim_name
        recalled = await self.memory.recall(query, scope=Scope.USER, key=state.user_id)
        if recalled:
            # D3 三维重排：recency×sim×importance 加权和。arousal 取已携带的 mood 唤醒——
            # 召回节点在 affect_core 之前，本轮 affect_sample 尚未算出，故用 mood[1] 作 NE 代理。
            arousal = state.mood[1] if state.mood is not None else 0.0
            ranked = _rank_episodes(recalled, datetime.now(UTC), arousal=arousal)
            out["recalled_context"] = [f.content for f in ranked]
            out["recalled_facts"] = ranked  # D1：供 chat_driver 按 importance 注入 history
            entry["recalled_context_n"] = len(ranked)

        if not out:
            return {}
        out["trace"] = [entry]
        return out
