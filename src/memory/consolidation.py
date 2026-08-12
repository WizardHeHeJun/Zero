"""记忆巩固与遗忘：Ebbinghaus 分层幂律衰减 + 睡眠巩固迁移 + ACT-R 频率归一。

本模块属**记忆层**（src/memory/），不 import 编排层（src/orchestration/）。
所有算法确定性、无 LLM、无 torch——守 affect 热路径红线（确定性可复现）。
三策略均由 run_consolidation_batch 统一调用，默认全部门关（零回归）。

生物学注释（工程近似声明）：
① salience = normalize_precision(precision) × |rpe| 是 BLA-NE 情绪唤醒调制的**工程代理**，
  非直接测量（基底外侧杏仁核-去甲肾上腺素轴的唤醒调制由 McGaugh 2004 综述，DOI 见下）。
  - salience 计算用 Hill 归一 p/(p+30) 把无界 precision（实测 ~28–72）归一到 (0, 0.5]；
    scale=30 与 memory_recall.py normalized_importance(scale=30) 对齐（议会 WARN-3b）。
  - rpe=0.5 为常数代理（事后不可得）。**机制限制（神经席 WARN-3b）**：
    此代理不区分高/低 RPE episode 的意外度差异——所有 episode 获同等意外度权重，
    丢失了 BLA-NE 轴中因突发/意外事件引起的差异化巩固效应。
② 会话结束触发巩固 = **生物睡眠周期工程近似**（生物实际跨天至数周；
  此处以对话结束作周期近似，Davis & Zhong 2017 DOI 10.1016/j.neuron.2017.05.039）。
③ 分层幂律 d_session/d_user 对应系统巩固快/慢双阶段
  （McClelland et al. 1995 互补学习系统；MCM 4 参数模型留为理论参照，
  本实现用更轻量的幂律近似而非完整 MCM）。
④ ACT-R 频率近似：Petrov 2006 B≈ln(n/(1−d))−d·ln(L) 简化近似
  （完整 ACT-R 需 chunk activation 历史，此处仅用 access_count+age 近似）。

Davis & Zhong 2017: https://doi.org/10.1016/j.neuron.2017.05.039
McGaugh 2004 (Annu Rev Neurosci): https://doi.org/10.1146/annurev.neuro.27.070203.144157
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from src.memory.utils import (
    importance_excess,
    importance_signal,
    normalize_precision,
    parse_importance,
    parse_importance_tags,
)

logger = logging.getLogger(__name__)


# ── 协议：策略接口 ──────────────────────────────────────────────────────────────


@runtime_checkable
class ConsolidationStrategy(Protocol):
    """记忆巩固策略协议：接收 episode 元数据列表，返回操作指令。

    返回值约定（均为 episode_id str 列表）：
      decay_updates: list[tuple[float, str]]  — (new_decay_weight, episode_id)
      consolidate_ids: list[str]              — 升迁到 user scope 的 episode_id
    """

    def compute(
        self,
        episodes: list[dict[str, Any]],
        *,
        now: datetime,
    ) -> tuple[list[tuple[float, str]], list[str]]: ...


# ── 策略一：Ebbinghaus 分层幂律衰减 ─────────────────────────────────────────────


class EbbinghausDecay:
    """分层幂律遗忘衰减（SESSION 快衰 d_s / USER 慢衰 d_u）。

    decay_weight = a_eff × Δt^(−d)，结果钳到 (0, 1]。
    重要性调制有效振幅：`a_eff = a × (1 + κ·u)`，`u = importance_excess(importance)`
    （重要性高→振幅大→衰减慢，仿 McGaugh 唤醒调制方向）。

    ⚠ 2026-08-12 两处改动（PRP importance-signal）：
    ① 信号来源由 `affect_precision` 派生的 salience 换成 **tag 派生的 importance**——
       后验精度衡量「情绪判断有多确定」，与「内容有多重要」方向常相反。
    ② 参数化由 `salience^κ` 换成 `(1+κ·u)` 混合式——旧式在中性点 `b0^κ ≠ 1`，
       对**无 tag 的普通 episode 也施加压低**（确定性基线漂移，与噪声无关）。
       混合式在 `u=0` 时对任意 κ 恒为 `a`，正交性由参数化保证而非靠注释约束。
    两项必须同批改：旧 salience 值域 `(0,0.5]`，若只换参数化而不换信号来源，
    `u` 会**恒为 0**、调制静默失效。

    注意：**非 MCM 4 参数模型**（MCM 留理论参照）；用更轻量的幂律近似。
    分层：SESSION scope 用 d_s（快衰），USER scope 用 d_u（慢衰），
    对应系统巩固理论的快速海马存储与慢速新皮层整合两阶段。
    """

    def __init__(
        self,
        d_session: float = 0.8,
        d_user: float = 0.3,
        a: float = 1.0,
        kappa: float = 0.5,
        tag_importance_enabled: bool = False,
    ) -> None:
        self.d_session = d_session
        self.d_user = d_user
        self.a = a
        self.kappa = kappa
        # 默认 False = 走旧口径（salience^κ），与改前逐字等价。env 由编排层读后传入，
        # 本层不 getenv（守三层：记忆层不读部署配置）。
        self.tag_importance_enabled = tag_importance_enabled

    def compute(
        self,
        episodes: list[dict[str, Any]],
        *,
        now: datetime,
    ) -> tuple[list[tuple[float, str]], list[str]]:
        """计算每个 episode 的新 decay_weight，不触发巩固迁移。"""
        updates: list[tuple[float, str]] = []
        for ep in episodes:
            eid = ep.get("episode_id")
            if not eid:
                continue
            scope = ep.get("scope", "session")
            d = self.d_session if scope == "session" else self.d_user
            valid_at = ep.get("valid_at")
            if valid_at is None:
                continue
            if not valid_at.tzinfo:
                valid_at = valid_at.replace(tzinfo=UTC)
            delta_days = max(1.0, (now - valid_at).total_seconds() / 86400.0)
            if self.tag_importance_enabled:
                # 振幅调制走 tag 重要性（值域 [b0,1)），不再走 affect_precision 派生的
                # salience：后者衡量情绪后验确定性，与内容重要性方向常相反（见
                # utils.importance_signal）。混合式 a_eff = a·(1+κ·u)：u=0（无 tag）时对
                # 任意 κ 恒为 a ⇒ 不触碰基线。此处调**振幅** a 而非 _rank_episodes 的
                # **指数** d——decay_weight 有 min(1.0,…) 钳制、不要求与 sim 同量纲，
                # 故振幅式在这里安全（两处是不同的正确形式）。
                a_eff = self.a * (1.0 + self.kappa * importance_excess(ep.get("importance", 0.5)))
            else:
                # 旧口径（默认）：affect_precision 派生的 salience 直接作幂函数底数。
                # ⚠ 已知失真但**保留为默认**以守零回归：salience^κ 在中性点 ≠1，对无 tag
                # 的普通 episode 也施加压低（同 _rank_episodes 旧式的基线漂移）。
                a_eff = self.a * (max(0.0, ep.get("salience", 0.5)) ** self.kappa)
            new_dw = min(1.0, max(1e-6, a_eff * (delta_days ** (-d))))
            updates.append((new_dw, eid))
        return updates, []


# ── 策略二：睡眠巩固迁移 ────────────────────────────────────────────────────────


class SleepConsolidation:
    """双准则 SESSION→USER 巩固迁移（工程近似生物睡眠周期·Davis & Zhong 2017）。

    升迁条件（AND）：salience ≥ salience_threshold AND consolidation_count ≥ min_count。
    两条件同时满足才升迁，防止「低 consolidation_count 高 salience」的偶发突出事件
    在未经足够强化前被误升迁为长期记忆（对应生物记忆巩固需要重复激活的再激活机制）。
    升迁后原 SESSION 行 invalid_at 软删（由 consolidate_session_to_user 执行）。
    """

    def __init__(
        self,
        salience_threshold: float = 0.25,
        consolidation_count_min: int = 3,
    ) -> None:
        self.salience_threshold = salience_threshold
        self.consolidation_count_min = consolidation_count_min

    def compute(
        self,
        episodes: list[dict[str, Any]],
        *,
        now: datetime,
    ) -> tuple[list[tuple[float, str]], list[str]]:
        """筛选满足双准则的 SESSION episode id，供升迁到 USER scope。"""
        consolidate_ids: list[str] = []
        for ep in episodes:
            eid = ep.get("episode_id")
            if not eid:
                continue
            if ep.get("scope", "") != "session":
                continue
            salience = ep.get("salience", 0.0)
            cc = ep.get("consolidation_count", 0) or 0
            if salience >= self.salience_threshold and cc >= self.consolidation_count_min:
                consolidate_ids.append(eid)
        return [], consolidate_ids


# ── 策略三：ACT-R 频率归一 ──────────────────────────────────────────────────────


class ACTRFrequency:
    """ACT-R Petrov 2006 B 近似 + sigmoid 归一到 (0, 1]。

    B ≈ ln(n / (1 − d)) − d · ln(L)  （Petrov 2006 简化近似；
    完整 ACT-R 需全历史 chunk activation，此处仅用 access_count(n) + age(L) 近似）。
    sigmoid(B / b_scale) ∈ (0, 1]，与 recency/sim 同量纲，供 _rank_episodes 替换 recency 维。
    d = decay_d（幂律指数，与 EbbinghausDecay 分层参数对应）。
    L = age in days（max 1）；n = access_count（需 >0 才有意义）。
    """

    def __init__(self, d: float = 0.5, b_scale: float = 3.0) -> None:
        self.d = d
        self.b_scale = b_scale

    def compute(
        self,
        episodes: list[dict[str, Any]],
        *,
        now: datetime,
    ) -> tuple[list[tuple[float, str]], list[str]]:
        """本策略不产生 decay_weight 更新或巩固迁移，仅作归一化计算接口。"""
        return [], []

    def compute_b_norm(
        self,
        access_count: int,
        l_days: float,
        d: float | None = None,
        b_scale: float | None = None,
    ) -> float:
        """Petrov B 近似 + sigmoid 归一到 (0, 1]。

        access_count=0 或 l_days<=0 → 返回最小正值（不奖励未访问 episode）。
        d 和 b_scale 优先用传入值，缺省用构造参数（便于 _rank_episodes 复用）。
        """
        _d = d if d is not None else self.d
        _bs = b_scale if b_scale is not None else self.b_scale
        return _petrov_b_norm(access_count, l_days, _d, _bs)


# ── 纯函数：Petrov B 归一（编排层 memory_recall.py 有独立同逻辑副本·两处同步维护）────


def _petrov_b_norm(
    access_count: int,
    l_days: float,
    d: float,
    b_scale: float,
) -> float:
    """Petrov 2006 B 近似 + sigmoid 归一到 (0, 1]。模块级纯函数，可单测。

    B ≈ ln(n / (1 − d)) − d · ln(L)
    result = 1 / (1 + exp(−B / b_scale))  ∈ (0, 1]

    access_count(n)=0 或 l_days<=0 → 返回极小正值（避免零但不奖励未访问）。
    d ∈ (0, 1)；d=1 时分母趋零→回退 d=0.999 防 log 爆炸。
    """
    n = max(1, access_count)  # n=0 时取 1 避免 ln(0)；下游排序中未访问 episode 得分极低
    l_val = max(1.0, l_days)
    _d = min(0.999, max(0.001, d))
    try:
        b = math.log(n / (1.0 - _d)) - _d * math.log(l_val)
        return 1.0 / (1.0 + math.exp(-b / b_scale))
    except (ValueError, OverflowError):
        return 1e-6


# ── 统一批处理入口 ───────────────────────────────────────────────────────────────


async def run_consolidation_batch(
    semantic: Any,
    *,
    scope_session: str,
    scope_user: str,
    key: str,
    consolidation_enabled: bool = False,
    d_session: float = 0.8,
    d_user: float = 0.3,
    salience_threshold: float = 0.25,
    consolidation_count_min: int = 3,
    actr_b_scale: float = 3.0,
    tag_importance_enabled: bool = False,
) -> None:
    """记忆巩固批处理：Ebbinghaus 衰减 + 睡眠巩固迁移（会话结束触发）。

    参数：
      semantic            — SemanticStore 实例（duck-type，需具备 episode 查询/更新方法）
      scope_session/user  — 显式 scope 字符串（记忆层规则 #2：禁止默认 scope）
      key                 — user/session key（通常 = user_id）
      consolidation_enabled — 主门：False → 整体 no-op（零回归）
      d_session/d_user    — 分层幂律指数（SESSION 快衰 / USER 慢衰）
      salience_threshold  — 睡眠巩固 salience 门（≥ 此值才候选）
      consolidation_count_min — 睡眠巩固 consolidation_count 门（≥ 此值才升迁）
      actr_b_scale        — Petrov B sigmoid 归一 scale（越小越激进）

    整体门控：consolidation_enabled=False → 直接 return（不查 DB、不写任何内容）。
    此函数不 import 编排层（守三层单向），由 MemoryClient.run_consolidation_batch 调用。

    注意事项（工程近似声明）：
    - Ebbinghaus 衰减走 decay_weight 字段，不物理删除（_trim_capacity 按 decay_weight 驱逐）。
    - 睡眠巩固迁移：SESSION→USER 复制行，原行 invalid_at 软删（search 路径自动跳过）。
    - 本函数确定性、无 LLM、无 torch（守 affect 热路径红线）。
    - ACT-R 频率归一（ACTRFrequency）仅控召回侧 _rank_episodes recency 替换，巩固批处理不消费。
    """
    if not consolidation_enabled:
        return

    now = datetime.now(UTC)

    # ── 1. 读取当前 SESSION/USER scope episode 元数据 ────────────────────────
    # 经存储层协议方法（fetch_episodes_for_consolidation）读取，不穿透内部 conn/db_lock。
    # 无此能力（如 GraphitiGraphStore）直接降级跳过（不崩管线）。
    if not hasattr(semantic, "fetch_episodes_for_consolidation"):
        logger.warning(
            "run_consolidation_batch: semantic 后端不支持 fetch_episodes_for_consolidation，跳过"
        )
        return

    rows_dicts: list[dict[str, Any]] = await semantic.fetch_episodes_for_consolidation(
        scope_session, scope_user, key
    )

    if not rows_dicts:
        return

    # ── 2. 构造 episode 元数据列表（含 salience 代理） ───────────────────────
    # precision 解析走同层 utils.parse_importance（原为本函数内联的 _parse_importance_local，
    # 与 memory_recall 各持一份同逻辑副本；已上提消 DRY·仍不 import 编排层·守三层单向红线）。
    episodes: list[dict] = []
    for row in rows_dicts:
        valid_at_str = row.get("valid_at")
        if not isinstance(valid_at_str, str):
            continue
        try:
            valid_at = datetime.fromisoformat(valid_at_str)
        except ValueError:
            continue
        if not valid_at.tzinfo:
            valid_at = valid_at.replace(tzinfo=UTC)
        # salience 代理：Hill 归一 p/(p+30) × 0.5（议会 WARN-3b·量纲修正）
        # - Hill 归一把无界 precision（实测 ~28–72）归一到 (0, 1)，scale=30 与
        #   memory_recall.py normalized_importance(scale=30) 对齐，消除量纲失真。
        # - × 0.5 = rpe 常数代理（事后不可得，保守用 0.5·丢失意外度个体差异·见模块头注释①）
        # 归一后 salience∈(0, 0.5]：precision=0.5(fallback)→~0.008；28→~0.241；30→0.25；72→~0.353
        content = row.get("content", "")
        precision = parse_importance(content)
        salience = normalize_precision(precision, scale=30.0) * 0.5
        # importance：tag 派生的内容重要性（值域 [0.5,1)），供 EbbinghausDecay 的振幅调制。
        # 与上面的 salience 并存而非替代——salience 仍供 SleepConsolidation 的升迁门使用
        # （那是「情绪显著度」语义，对应 McGaugh 唤醒调制，本 PRP 不改；D-B 只解耦 ②③）。
        importance = importance_signal(parse_importance_tags(content))
        episodes.append(
            {
                "episode_id": row["episode_id"],
                "scope": row.get("scope", ""),
                "key": row.get("key", ""),
                "content": content,
                "valid_at": valid_at,
                "access_count": row.get("access_count", 0),
                "consolidation_count": row.get("consolidation_count", 0),
                "decay_weight": row.get("decay_weight", 1.0),
                "salience": salience,
                "importance": importance,
            }
        )

    # ── 3. Ebbinghaus 分层幂律衰减（decay_weight 更新） ──────────────────────
    decay_strategy = EbbinghausDecay(
        d_session=d_session, d_user=d_user, tag_importance_enabled=tag_importance_enabled
    )
    decay_updates, _ = decay_strategy.compute(episodes, now=now)
    if decay_updates and hasattr(semantic, "apply_decay_weights"):
        try:
            await semantic.apply_decay_weights(decay_updates)
            logger.debug("consolidation decay_updates n=%d", len(decay_updates))
        except Exception as exc:
            logger.warning("apply_decay_weights failed: %s", exc, exc_info=True)

    # ── 4. 睡眠巩固迁移：SESSION→USER（双准则门） ────────────────────────────
    sleep_strategy = SleepConsolidation(
        salience_threshold=salience_threshold,
        consolidation_count_min=consolidation_count_min,
    )
    _, consolidate_ids = sleep_strategy.compute(episodes, now=now)
    if consolidate_ids and hasattr(semantic, "consolidate_session_to_user"):
        try:
            await semantic.consolidate_session_to_user(scope_session, scope_user, consolidate_ids)
            logger.info("consolidation session→user key=%s n=%d", key, len(consolidate_ids))
        except Exception as exc:
            logger.warning("consolidate_session_to_user failed: %s", exc, exc_info=True)
