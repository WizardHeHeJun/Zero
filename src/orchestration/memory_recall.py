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
from src.memory.utils import (
    DEFAULT_TAG_WEIGHTS,
    combine_importance_with_precision,
    importance_excess,
    importance_signal,
    normalize_precision,
    parse_importance,
    parse_importance_tags,
    tag_excess,
)
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
    tag_importance_enabled: bool = False,
) -> list[Fact]:
    """三维加权和重排（D3）：`score = α·recency + β·sim + γ_eff·importance`。

    - recency：默认 `Δt^(-d)`（Wixted&Ebbesen 1991 幂律，非指数）；
      actr_enabled=True 且 fact.episode_id 非空且 access_count>0 时，用 Petrov B 近似
      替换（ACT-R 频率·归一到 (0,1]·与 sim/importance 同量纲）——零回归。
      salience_decay_enabled=True 时改用 `Δt^(−d_eff)`，`d_eff = d/(1+κ·u)`、
      `u = importance_excess(I)`：显著度调制**衰减速率本身**（即指数 d），对应 McGaugh
      唤醒调制巩固强度。`decay_weight` 此前只作用于 `_trim_capacity` 驱逐顺序，召回排序
      完全不受衰减影响——本参数补上这条回路。
      **两门互斥：ACT-R 优先**（二者都在改写 recency 维，同开会二次污染）。
      κ=0 时 `d_eff = d` 精确退化回原式；**u=0（中性/无 tag）时对任意 κ 亦恒等于原式**
      ——正交性由参数化本身保证，不靠注释约束。
      ⚠ **2026-08-12 判定订正（原「已知耦合·有界·非 bug」作废，改判失真）**：旧式
      `I^κ·Δt^(−d)` 有**确定性基线漂移**——`b0^κ ≠ 1` ⇒ 即便无任何 tag 的普通 episode，
      开门后 recency 也被系统性压低。实测 Δt=4 天、κ=2 时闲聊典型压低 **94%**，三维占比
      由 `recency 31.8%/sim 52.4%/imp 15.9%` 变成 `2.8%/74.6%/22.6%`——加权和被悄悄变成
      「几乎只看 sim」。此漂移与 I 有无噪声**无关**（原注释只归因于噪声放大，是不完整的）。
      2026-08-11 离线 A/B 所报「高显著者上移 ⇒ 机制生效」为**误读**：真实成因是 recency
      维被整体压扁，当时 sim 固定同值、剩下起作用的只有 importance。
      （数学席判失真 + 主程复算证实，见 `PRP/importance-signal/design.md`。）
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
        # importance 维信号来源二选一（tag_importance_enabled 默认关 = 零回归）：
        # - 开：tag 派生的 noisy-OR 信号，**三个 tag 统一计入**（修心理席指出的 2/3 缺口：
        #   此前 identity 靠覆写 precision、first_contact 靠 ×1.2，而 commitment 只决定
        #   写不写、对排序毫无影响）。此路**不读 precision=**（PRD 目标 G2）。
        # - 关：原 Hill 归一的 affect_precision 代理（D8）。
        if tag_importance_enabled:
            # fold-in noisy-OR（议会二轮张力 1）：tag 证据在 precision 基线上**放大**，
            # 而非替换。无 tag 时 w_p=1.0 使其精确退化为 precision 基线 ⇒
            # 保留全部 171 种取值，不再塌成 2 种（上一轮「替代」实测把该维压平 6.5 倍）。
            # precision 基线读 `precision_raw=`（floor 前原始读数，缺失回退 precision=），
            # 使 IDENTITY_MEMORY_PRECISION 覆写不经此路二次扩散——见
            # utils.combine_importance_with_precision / parse_raw_precision。
            tags = parse_importance_tags(fact.content)
            importance = combine_importance_with_precision(
                importance_signal(tags), fact.content, scale=importance_scale
            )
        else:
            importance = normalized_importance(fact.content, importance_scale)  # D8：Hill 归一
        # recency 维三选一，优先级 ACT-R > salience 衰减 > 原始幂律（后两者默认关=零回归）。
        # ACT-R 门控：有 episode_id 且 access_count>0 才替换 recency。
        if actr_enabled and fact.episode_id and fact.access_count > 0:
            recency = _petrov_b_norm(fact.access_count, delta_days, decay_d, actr_b_scale)
        elif salience_decay_enabled:
            # 调**衰减指数**而非乘结果：「显著度调制遗忘速率」，遗忘速率就是 d。
            #   d_eff = d/(1+κ·u)，recency = Δt^(−d_eff)
            # - u=0（无 tag/中性）→ d_eff = d ⇒ **与门关逐字相同**，对任意 κ 成立（正交）
            # - u>0 → d_eff < d ⇒ 衰减更慢（方向合 McGaugh「只增强不弱化」）
            # - Δt^(−d_eff) ∈ (0,1] ⇒ **量纲天然保持**，与 sim∈[0,1] 可等权相加
            # ⚠ 不能用乘子式 `(1+κ·u)·Δt^(−d)`（议会数学席原式）：其上界为 1+κ，会把
            #   recency 顶出 (0,1]、悄悄放大 α 维实权——该式只论证了 ≥1，未给上界。
            #   test_rank_salience_decay_stays_in_unit_range 实测抓到。
            # ⚠ 也不能用旧式 `I^κ`（已判失真）——见 utils.importance_excess 的失真说明。
            # 语义后果（非缺陷）：Δt=1 天时 recency 恒为 1（上界），故高显著旧 episode
            # 在 recency 维**不可能超过刚发生的**——「更耐遗忘」不等于「比刚发生的还新」。
            # 旧式能做到只是因为它把新 episode 压低了。反超与否由三维合力决定。
            # u 的来源分两种口径（议会二轮数学席点名的隐蔽污染）：
            # - tag 模式：`u` 必须**只依赖 tag**。若沿用 importance_excess(importance) 从
            #   合并后的 I 反解，而 I 现已混入 Ĩ_prec ⇒ precision 的正常波动会被误当成
            #   「重要性超出基线」，污染「u=0 时对任意 κ 恒等于门关」这条正交性保证。
            # - 门关模式：维持原口径（从 Hill 归一的 importance 反解），零回归。
            u = (
                tag_excess(parse_importance_tags(fact.content))
                if tag_importance_enabled
                else importance_excess(importance)
            )
            d_eff = decay_d / (1.0 + salience_kappa * u)
            recency = delta_days ** (-d_eff)
        else:
            recency = delta_days ** (-decay_d)
        # ×1.2 首因加权在 salience 衰减调制**之后**施加：首因是检索优先权（D5），
        # 不是巩固强度，不该进衰减速率；且 boost 后可 >1（0.9×1.2=1.08），
        # 若入 I^κ 会把 recency 顶出 (0,1] 量纲。
        # 判定走 parse_importance_tags 的**位置锚定**，不用裸子串 `in`——后者会把用户原话里
        # 的字面串 first_contact=True 当成系统 tag，用户自称即可白拿 ×1.2（PRP 执行期实证：
        # importance 0.2500→0.3000，与真首因加成完全相同并反超排序）。
        # ⚠ 仅门**关**路径施加（code-reviewer 2026-08-13 WARN）：tag 模式下 first_contact
        # 已作为 noisy-OR 的一项证据计入 importance（w=0.2 即「精确复现 ×1.2」的锚定），
        # 再乘一次 = 同一证据双重计权（实测 ×1.15 再 ×1.2 ≈ ×1.38）。门关路径的
        # normalized_importance 不含任何 tag 证据，×1.2 是其唯一的首因通道，原样保留。
        if not tag_importance_enabled and parse_importance_tags(fact.content)["first_contact"]:
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
        # 衰减调制强度 κ（默认 1.0）。语义：满 tag（u→1）时**有效衰减指数缩到 d/(1+κ)**
        # ——κ=1 即「遗忘速率减半」。κ=0 精确退化为原幂律。
        # ⚠ 该默认值的依据只在**除法式** d_eff=d/(1+κu) 下成立；早期草案里「recency 加成
        # 比例恰等于 importance 维加成比例」的说法是针对**乘子式** (1+κu)·Δt^(−d) 推的，
        # 除法式下比值 Δt^(d·κu/(1+κu)) 依赖 Δt、不是常数，那条依据已作废（code-reviewer
        # WARN 指出）。亦**不与** consolidation.EbbinghausDecay.kappa「对齐」——两处公式
        # 不同构（此处调指数、那处调振幅），各有独立依据。
        self.salience_kappa = float(os.getenv("ZERO_RECALL_SALIENCE_KAPPA", "1.0"))
        # importance 维改吃 tag 派生信号（默认关=零回归）。开启后该维**不再读 precision=**，
        # 三个 tag 统一进 noisy-OR（含此前对排序毫无影响的 commitment）。
        self.tag_importance_enabled = os.getenv("ZERO_TAG_IMPORTANCE", "0").lower() not in (
            "0",
            "",
            "false",
        )
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
                tag_importance_enabled=self.tag_importance_enabled,
            )
            # 排障用（CS 席 Q5）：口径记在 trace，**不加第 4 个 content tag**——后者会扩大
            # 文本解析面。① inject_min 仍读 precision=、②③ 走 fold-in（tag×precision_raw），
            # 两套并存是 D-B 的有意后果，此字段用于区分某轮排序到底走了哪套。
            # importance_zeroed_tags：权重为 0 而被临时移出信号的 tag（当前 commitment，
            # 议会二轮张力 3）——留痕防「信号还在、只是权重为零」这件事被遗忘。
            if self.tag_importance_enabled:
                entry["importance_source"] = "tag_foldin"
                zeroed = sorted(k for k, w in DEFAULT_TAG_WEIGHTS.items() if w == 0.0)
                if zeroed:
                    entry["importance_zeroed_tags"] = zeroed
            else:
                entry["importance_source"] = "precision"
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
