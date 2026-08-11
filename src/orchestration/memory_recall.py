"""MemoryRecallAgent：长期情绪倾向回灌（记忆读闭环）。

在管线开头读 `user` 作用域的长期情绪倾向（由 Supervisor 在任务完成时写入），
解析出标量 disposition 放进 state，供 AppraisalAgent 偏置先验——让记忆层真正被
『用上』（此前仅写不读，闭合 读↔写 回路）。注入 MemoryClient，不直连图谱。
`recall_enabled` 关闭或无记忆时为 no-op（严格零回归）。
节点契约：(state) -> dict，只返回增量。
"""

from __future__ import annotations

import logging
import math
import os
from datetime import UTC, datetime

from src.agents.affect_math import text_label
from src.memory.client import MemoryClient
from src.memory.types import Fact, Scope
from src.memory.utils import normalize_precision, parse_importance
from src.orchestration.state import AffectState

logger = logging.getLogger(__name__)

# parse_importance 已上提至 src/memory/utils.py（原先本模块与 consolidation 各持一份副本）。
# 此处 import 仅供 normalized_importance 内部使用，不做兼容再导出——实测全仓无任何调用方
# 按 memory_recall.parse_importance 引用，留个二次入口只会让同一函数有两条导入路径。


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


def normalized_importance(content: str, scale: float = 30.0) -> float:
    """把无界的写入精度（affect_precision=方差倒数，实测 ~28–72）归一到 (0,1)，供三维打分/注入门。

    用 Hill 饱和 `p/(p+C)`（与 Kalman 增益/逆方差加权同构、单调有界、边际递减；数学席 D8）：
    与 `sim∈[0,1]`、`recency=Δt^(-d)∈(0,1]` 同量纲，α/β/γ 等权才恢复语义；否则原始 precision
    几十倍碾压另两维（dogfood 实测）。scale（=C）默认 30，匹配实测量级使 INJECT_MIN=0.5 成为
    「高质量门」；由调用方（MemoryRecallAgent）从 env 读取一次后传入——纯函数、可直接单测。
    C 固定（非自适应集合统计）以守可复现/确定性（CS 席 D8 红线）。
    precision 字段缺失时 parse 返 0.5 → 归一后 ~低位（未知不优先）。
    """
    return normalize_precision(parse_importance(content), scale)


def _petrov_b_norm(
    access_count: int,
    l_days: float,
    d: float,
    b_scale: float,
) -> float:
    """Petrov 2006 B 近似 + sigmoid 归一到 (0, 1]。模块级纯函数，可单测。

    B ≈ ln(n / (1 − d)) − d · ln(L)；sigmoid(B/b_scale) ∈ (0,1]。
    替换 _rank_episodes 的 recency 项（actr_enabled=True 时生效·零回归）。
    access_count=0 或 l_days<=0 → 返回极小正值（不奖励未访问 episode）。
    d ∈ (0,1)；d=1 时分母趋零→钳到 0.999 防 log 爆炸。
    """
    n = max(1, access_count)
    l_val = max(1.0, l_days)
    _d = min(0.999, max(0.001, d))
    try:
        b = math.log(n / (1.0 - _d)) - _d * math.log(l_val)
        return 1.0 / (1.0 + math.exp(-b / b_scale))
    except (ValueError, OverflowError):
        return 1e-6


def _rank_episodes(
    facts: list[Fact],
    now: datetime,
    *,
    arousal: float = 0.0,
    decay_d: float = 0.5,
    alpha: float = 0.33,
    beta: float = 0.34,
    gamma: float = 0.33,
    arousal_mod: bool = False,
    importance_scale: float = 30.0,
    actr_enabled: bool = False,
    actr_b_scale: float = 3.0,
    salience_decay_enabled: bool = False,
    salience_kappa: float = 0.5,
) -> list[Fact]:
    """三维加权和重排（D3）：`score = α·recency + β·sim + γ_eff·importance`。

    - recency：默认 `Δt^(-d)`（Wixted&Ebbesen 1991 幂律，非指数）；
      actr_enabled=True 且 fact.episode_id 非空且 access_count>0 时，用 Petrov B 近似
      替换（ACT-R 频率·归一到 (0,1]·与 sim/importance 同量纲）——零回归。
      salience_decay_enabled=True 时改用 `I^κ · Δt^(-d)`（I=normalized_importance∈(0,1)）：
      显著度调制**衰减速率**而非基线权重，对应 McGaugh 唤醒调制巩固强度，与
      `consolidation.EbbinghausDecay` 的 `a_eff = a·salience^κ` 同构（该处衰减只作用于
      `_trim_capacity` 驱逐顺序，召回排序此前完全不受衰减影响——本参数补上这条回路）。
      **两门互斥：ACT-R 优先**（二者都在改写 recency 维，同开会二次污染）。
      **已知耦合（有界·非 bug）**：I 同时出现在 γ 维基线与本处衰减调制中；κ 控制第二处
      强度，κ=0 → `I^0=1` 精确退化回原式（变异测试靶子 test_rank_salience_decay_kappa_zero）。
      ⚠ **调 κ 前必读的实测代价**（23 条真实 episode·Δt 铺 1–90 天·sim 固定·唯一变量为本旋钮）：
      κ=2 时 20/23 条位次变化、方向符合预期（高显著者上移），但 `affect_precision` 的上游
      估计噪声被**同步放大**——观测到低信息量却精度估高的 episode（「刚整理了一下桌子」
      I=0.374）压过高信息量却精度估低的（「我最近状态不太好」I=0.268）。κ 越大放大越狠。
      根因在 affect_precision 估计环节、非本旋钮引入，但选值须知情：`.env.example` 的 κ
      推荐值取保守侧（0.5）正是为此。
      量纲：I∈(0,1)、Δt^(-d)∈(0,1] ⇒ 乘积仍 ∈(0,1]，与 sim/importance 同量纲不破坏等权。
      ⚠ 范围收窄：MemoryRecallAgent 的 recall 只查 Scope.USER，故 EbbinghausDecay 的
      SESSION/USER 分层 d 在召回路径用不上，实际只有 d_user 一档生效。
    - relevance：`fact.sim`（D4 透传余弦；确定性后端为 0.0，该维自然退化）。
    - importance：写入时 precision 显著度；first_contact=True 时 ×1.2（D5 首因加权）。
    加权和而非乘积；`γ_eff = γ·(1+0.5·clamp(arousal,0,1))` 仅 arousal_mod=True 生效。
    纯数值无 LLM，所有旋钮由调用方传入；返回按 score 降序的新列表，不改元素。
    """
    if not facts:
        return []
    gamma_eff = gamma * (1.0 + 0.5 * max(0.0, min(1.0, arousal))) if arousal_mod else gamma

    def score(fact: Fact) -> float:
        valid_at = fact.valid_at if fact.valid_at.tzinfo else fact.valid_at.replace(tzinfo=UTC)
        delta_days = max(1.0, (now - valid_at).total_seconds() / 86400.0)
        importance = normalized_importance(fact.content, importance_scale)  # D8：Hill 归一
        # recency 维三选一，优先级 ACT-R > salience 衰减 > 原始幂律（后两者默认关=零回归）。
        # ACT-R 门控：有 episode_id 且 access_count>0 才替换 recency。
        if actr_enabled and fact.episode_id and fact.access_count > 0:
            recency = _petrov_b_norm(fact.access_count, delta_days, decay_d, actr_b_scale)
        elif salience_decay_enabled:
            recency = (importance**salience_kappa) * (delta_days ** (-decay_d))
        else:
            recency = delta_days ** (-decay_d)
        # ×1.2 首因加权在 salience 衰减调制**之后**施加：首因是检索优先权（D5），
        # 不是巩固强度，不该进衰减速率；且 boost 后可 >1（0.9×1.2=1.08），
        # 若入 I^κ 会把 recency 顶出 (0,1] 量纲。
        if "first_contact=True" in fact.content:
            importance *= 1.2
        return alpha * recency + beta * fact.sim + gamma_eff * importance

    return sorted(facts, key=score, reverse=True)


class MemoryRecallAgent:
    """读 user 长期倾向 → recalled_disposition（偏置 appraisal）。注入 client，不直连图谱。

    旋钮参数在构造期一次从 env 读取，存 self.*，不在每次 __call__ 热路径重读。
    """

    def __init__(self, memory: MemoryClient) -> None:
        self.memory = memory
        # 召回三维权重旋钮（构造期一次解析，默认值与旧 getenv 逐字一致）
        self.decay_d = float(os.getenv("ZERO_RECALL_DECAY_D", "0.5"))
        self.alpha = float(os.getenv("ZERO_RECALL_ALPHA", "0.33"))
        self.beta = float(os.getenv("ZERO_RECALL_BETA", "0.34"))
        self.gamma = float(os.getenv("ZERO_RECALL_GAMMA", "0.33"))
        self.arousal_mod = os.getenv("ZERO_RECALL_AROUSAL_MOD", "0").lower() not in (
            "0",
            "",
            "false",
        )
        self.importance_scale = float(os.getenv("ZERO_RECALL_IMPORTANCE_SCALE", "30"))
        # ACT-R 频率门控（默认关=零回归）：开启时用 Petrov B 近似替换 recency 项。
        # ZERO_ACTR_ENABLED 未设/0/false → False → _rank_episodes 用原幂律 Δt^(-d)。
        self.actr_enabled = os.getenv("ZERO_ACTR_ENABLED", "0").lower() not in (
            "0",
            "",
            "false",
        )
        # Petrov B sigmoid 归一 scale（越小越激进·默认 3.0·仅 actr_enabled=True 时生效）
        self.actr_b_scale = float(os.getenv("ZERO_ACTR_B_SCALE", "3.0"))
        # 召回侧 salience 衰减门控（默认关=零回归）：开启时 recency 改用 I^κ·Δt^(-d)，
        # 让显著度调制衰减速率（高显著旧 episode 更耐遗忘）。与 ACT-R 互斥、ACT-R 优先。
        self.salience_decay_enabled = os.getenv("ZERO_RECALL_SALIENCE_DECAY", "0").lower() not in (
            "0",
            "",
            "false",
        )
        # 衰减调制指数 κ（默认 0.5，与 consolidation.EbbinghausDecay.kappa 对齐；
        # κ=0 精确退化为原幂律·κ 越大显著度对留存的影响越强）。
        self.salience_kappa = float(os.getenv("ZERO_RECALL_SALIENCE_KAPPA", "0.5"))
        if self.salience_decay_enabled and self.actr_enabled:
            # 构造期一次告警（非热路径）：两门都改写 recency 维，同开时 salience 衰减被 ACT-R 覆盖。
            logger.warning(
                "ZERO_RECALL_SALIENCE_DECAY 与 ZERO_ACTR_ENABLED 同时开启；"
                "二者均改写 recency 维，实际生效的是 ACT-R（salience 衰减仅对 "
                "access_count=0 或无 episode_id 的 Fact 生效）"
            )

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
            ranked = _rank_episodes(
                recalled,
                datetime.now(UTC),
                arousal=arousal,
                decay_d=self.decay_d,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                arousal_mod=self.arousal_mod,
                importance_scale=self.importance_scale,
                actr_enabled=self.actr_enabled,
                actr_b_scale=self.actr_b_scale,
                salience_decay_enabled=self.salience_decay_enabled,
                salience_kappa=self.salience_kappa,
            )
            out["recalled_context"] = [f.content for f in ranked]
            out["recalled_facts"] = ranked  # D1：供 chat_driver 按 importance 注入 history
            entry["recalled_context_n"] = len(ranked)
            # B 类：recalled_episode_ids 供 Supervisor 任务完成节点节流更新 access_count。
            # 不在此节点写 access_count（CS BLOCK：召回时更新污染当轮排序自一致性）。
            episode_ids = [f.episode_id for f in ranked if f.episode_id]
            if episode_ids:
                out["recalled_episode_ids"] = episode_ids

        if not out:
            return {}
        out["trace"] = [entry]
        return out
