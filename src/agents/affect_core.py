"""AffectCoreAgent：主动推断生成核（随机性来源）。

对应岛叶 + 内感受预测层级。把 OCC 先验与 RPE 驱动的证据按精度做高斯积融合，
再从后验**采样**出最终情绪 e*=(valence, arousal)——随机性在此进入。
`mood_enabled` 时把「已在的心境」并入为第三个精度加权先验（**读**）；心境的**更新**
由本节点之后独立的 MoodAgent 完成（A.7）。默认行为与 v1 一致。
节点契约：(state) -> dict，只返回增量。
"""

from __future__ import annotations

import logging
import random
from typing import TypedDict

from src.agents.affect_math import (
    AROUSAL_GAIN,
    MIN_PRECISION,
    MIN_SIGMA,
    SALIENCE_THRESHOLD,
    behavior_feedback_evidence,
    cap_stream_weight,
    effective_stream_count,
    evidence_from_value,
    expand_external_priors,
    fast_survival_prior,
    fuse_terms,
    gaussian_fuse,
    hierarchical_fuse,
    ignite,
    mood_precision,
    precision_da,
    report_ignited,
    sample_affect,
    text_affect_precision,
)
from src.orchestration.state import AffectState

logger = logging.getLogger(__name__)


class _GateCriteria(TypedDict):
    """`ignite()` 与 `report_ignited()` **必须共享**的判据参数。

    用 TypedDict 而非裸 dict：`**kwargs` 展开裸 dict 会丢类型信息（mypy 报 arg-type），
    而用 TypedDict 既保住「一处定义、两处消费」的防漂移结构，又不牺牲静态检查。
    """

    survival_fallback: bool
    soft_beta: float | None
    # `threshold` 也纳入——**即使当前两处调用都用签名默认值 `SALIENCE_THRESHOLD`**。
    # 不纳入的话它就退回「两边各自吃同一个默认值」这种**隐性**对齐，正是本结构要消灭的模式，
    # 只是收窄到一个字段上：将来若有人加 `ignition_threshold` 旋钮要传非默认阈值，
    # TypedDict 不会提醒他两处都改。纳入后两处引用同一个键，改阈值会被这个结构自然架住。
    threshold: float


class AffectCoreAgent:
    """高斯积融合先验与证据 → 后验 → 采样 e*；可选并入慢变心境（读，A.7）。"""

    def __call__(self, state: AffectState) -> dict:
        if state.prior_mu is None or state.prior_sigma is None or state.reward is None:
            return {}
        delta = state.rpe if state.rpe is not None else 0.0
        pi = state.precision if state.precision is not None else 1.0
        evidence = evidence_from_value(state.reward, delta)
        ignited: list[str] = []
        if state.workspace_enabled:
            # 显著度门控全局工作空间（v3）：并行流 (name, μ, Π) 竞争 → ignition 广播。
            # NE/唤醒增益：唤醒越高，评价·价值流的精度（投票权）越大（精度=神经调质增益）。
            arousal_gain = 1.0 + AROUSAL_GAIN * max(0.0, state.prior_mu[1])
            # P4-d（议会二轮·廉价 cap 防御）：默认 None → 不 cap（线性无界，零回归）；设 cap 则钳
            # arousal_gain≤1+cap，防高唤醒段 LC-NE 正反馈无界（完整倒 U 立项排后）。纯标量。
            if state.arousal_gain_cap is not None:
                arousal_gain = min(arousal_gain, 1.0 + state.arousal_gain_cap)
            prior_prec = (
                arousal_gain / max(MIN_SIGMA, state.prior_sigma[0]) ** 2,
                arousal_gain / max(MIN_SIGMA, state.prior_sigma[1]) ** 2,
            )
            # precision_commensurable（议会 2026-07-28 第四轮）：门开时 survival 流的精度
            # 由裸常数 0.4 改为 1/σ²（conf→σ→Π 链路，同构 occ_prior 但截距/斜率远低）。
            comm = state.precision_commensurable
            # arousal_floor_fix 与 gate_fusion **共用同一开关**（议会 D5 强制）：
            # 杜绝「只修地板不改架构」这种未评审的中间态。门关时逐字旧行为（带 0.5 地板）。
            surv_mu, surv_prec = fast_survival_prior(
                state.features, commensurable=comm, arousal_floor_fix=not state.gate_fusion
            )

            # A-P1-A（议会裁决·神经席 M5+数学席 M6）：precision_split 门控。
            # True → value 流证据精度用 precision_da(rpe)（仅 DA 路径 |δ|，消 β·V 混同）；
            # False → 逐字旧行为 pi*arousal_gain（零回归）。
            # 注：else 分支的 pi 来自 state.precision（value.py），门开时**已在那里齐次化**，
            # 故此处不重复处理——两条支路的量纲始终一致。
            rpe_for_da = state.rpe if state.rpe is not None else 0.0
            if state.precision_split:
                pi_da = precision_da(rpe_for_da, commensurable=comm) * arousal_gain
            else:
                pi_da = pi * arousal_gain

            # A-P0-B（议会裁决·数学席 M1）：fuse_independence_correct 门控。
            # value 流 valence 维与 appraisal/survival 共线（均依赖 goal_congruence）→
            # 条件独立假设失立 → 精度相加过度自信。
            # True → valence 维精度置 MIN_PRECISION（仅保留 arousal 维 π_DA，独立信号）；
            # False → 逐字旧行为（两维同精度，零回归）。
            # 评价流（appraisal）精度 = prior_prec，即 arousal_gain 加成后的 NE 路径 pi_ne。
            if state.fuse_independence_correct:
                value_prec: tuple[float, float] = (MIN_PRECISION, pi_da)  # valence 维极小
            else:
                value_prec = (pi_da, pi_da)

            streams: list[tuple[str, tuple[float, float], tuple[float, float]]] = [
                ("survival", surv_mu, surv_prec),  # 快生存流（低精度、可单独点燃）
                (
                    "appraisal",
                    state.prior_mu,
                    prior_prec,
                ),  # 慢评价流（OCC）；精度=arousal_gain/σ²，即 NE 路径 pi_ne
                ("value", evidence, value_prec),  # 价值流（RPE）
            ]
            # mood/text 两条流的精度是 env 可调旋钮（ZERO_MOOD_PRECISION / ZERO_TEXT_AFFECT_
            # PRECISION），默认停在旧标度 0.8/0.3。门开时**无条件**改用齐次值 1/σ²，
            # 即丢弃 state 里的旋钮值——这在热路径里是有意为之（节点不做校验、不抛异常）。
            # 防线在上游：SessionConfig._check_precision_scale_consistency 在配置期就拒绝
            # 「门开 + 旋钮被显式改过」的组合，使这里永远不会真的丢弃用户的显式配置。
            # AffectState 层刻意不设对等校验——它要能从 checkpoint 原样反序列化（同 :284 的理由）。
            # 直接构造 AffectState 绕过 SessionConfig 的调用点（测试）会静默忽略旋钮，非数值错误。
            mood_prec = mood_precision(commensurable=True) if comm else state.mood_precision
            text_prec = (
                text_affect_precision(commensurable=True) if comm else state.text_affect_precision
            )
            if state.mood_enabled and state.mood is not None:
                streams.append(("mood", state.mood, (mood_prec, mood_prec)))
            if state.text_affect is not None:
                streams.append(("text", state.text_affect, (text_prec, text_prec)))
            # 外部多模态先验流注入（T4·议会 2026-07-15 M1；design.md 受约束方案 c）。
            # 仅读 state，只 extend 局部 streams，不写任何 state 字段（节点契约）。
            # 不进 occ_prior/survival 入口（守 text 流先例：独立低精度竞争流，非底噪）。
            if state.external_priors:
                streams.extend(
                    expand_external_priors(
                        state.external_priors,
                        precision_cap=state.external_prior_precision_cap,
                        max_streams=state.max_external_streams,
                    )
                )
            # 行为反馈流（行为反馈环第二步·议会设计门 NEEDS-CHANGES 落地，
            # notes/2026-08-07-behavior-feedback-council.md；默认关=零回归）。
            # 三重在场门：总门开 ∧ 副本非 None（staleness 修正后=恰好上一回合）∧
            # 其 voluntary 非 None（调节差 δ≠0）——后两重在 behavior_feedback_evidence
            # 内判定，缺一即 None=流缺席（absent-cue，不注入 0 值假证据）。
            # 生产默认 regulation 关 ⇒ voluntary 恒 None ⇒ 门开也零回归。
            # ⚠ 默认硬门（gate_fusion=True）下本流 salience≈0.075·|a_expr|<阈值 0.18
            # 恒被滤除——生效组合须 gate_fusion=False 或软门（design.md §三，非缺陷）。
            if state.behavior_feedback_enabled and state.motion_efference is not None:
                behavior_term = behavior_feedback_evidence(
                    state.motion_efference, commensurable=comm
                )
                if behavior_term is not None:
                    streams.append(("behavior", *behavior_term))
            # 数值通路与报告通路**分离**（议会第三轮 D1 + 实现期 D13）：
            #   `ignite()` → 供 fuse_terms 的 (μ,Π) + **与之对齐**的流名（BLOCK 1 的实质保证：
            #     二者同一次筛选产出、恒等长，下面的 zip(strict=True) 永不失配）；
            #   `report_ignited()` → 「哪些流变得可报告」，**不影响任何数值**。
            # gate_fusion=True（默认）时两者返回逐值相等 → out["ignited_streams"] 逐字不变。
            # 两个函数**共享同一份判据参数**，写成一个 dict 传——不是风格偏好：
            # 分开手写两遍时，「参数保持同步」是一条只靠人维护的隐性契约，一旦漂移
            # （比如只给一边改了 soft_beta）两边都不会报错，`ignited_streams` 会悄悄变成
            # 与实际融合流集合不一致的标签。长度失配有 zip(strict=True) 响亮兜底，
            # **参数漂移没有任何兜底** → 用共享 dict 从结构上消除这种可能。
            gate_criteria: _GateCriteria = {
                "survival_fallback": state.ignition_survival_fallback,
                "soft_beta": state.ignition_beta,
                # 当前恒为签名默认值；显式写出来是为了让「改阈值」这件事必须动这一处，
                # 而不是分头改两个调用点（见 _GateCriteria docstring）。
                "threshold": SALIENCE_THRESHOLD,
            }
            terms, fusion_names = ignite(
                streams,
                gate_fusion=state.gate_fusion,
                exclude_physio_fusion=state.exclude_physio_fusion,
                **gate_criteria,
            )
            # w_b 后置封顶（议会必改 #2·数学席失真裁定的修法）：对**本轮真正进入融合**的
            # terms 现算行为流权重并封顶 ≤ W_MAX——前置 cap π_b 在其余流全弱时约束不了
            # 凸组合权重（w_b→0.97）。terms/names 成对返回保持对齐（BLOCK 1 先例）；
            # 重标定值跌破 MIN_PRECISION 的退化情形整条剔除（cap_stream_weight docstring）。
            # 仅总门开时调用（"behavior" 不在 names 时原样返回，但默认路径连查找都不做）。
            if state.behavior_feedback_enabled:
                terms, fusion_names = cap_stream_weight(terms, fusion_names, target="behavior")
            ignited = report_ignited(streams, **gate_criteria)
            # P3 层级预测编码（HPC v1）：门控关（默认 layers=1 或 coupling=0.0）→
            # 走现 fuse_terms 路径逐字不变（hierarchical_fuse 内部退化旁路保证零回归）。
            # 开启（layers>=2 且 coupling>0）→ 重建带 name 的流列表传给 hierarchical_fuse。
            # v1 在 ignite 之后做层级融合（保守取向；神经席「各流先层级、再进 GNW 竞争」的精确拓扑
            # 留 v2）。named_terms 用 ignite 输出的 terms（已含 soft_beta gate 调制后的精度，
            # soft_beta=None 时即原始精度）与 ignited 名单同序对齐重建——与软门控叠加时
            # 精度语义自洽（gate 先于 HPC）；soft_beta=None（默认）时精度不变，零回归。
            if state.hierarchical_layers >= 2 and state.hierarchical_coupling > 0.0:
                named_terms: list[tuple[str, tuple[float, float], tuple[float, float]]] = [
                    # 用 fusion_names（与 terms 同源对齐）而非 ignited（报告子集）——
                    # 门开时后者是前者的子集、长度不等，用它会当场 ValueError（BLOCK 1）。
                    (name, mu, prec)
                    for name, (mu, prec) in zip(fusion_names, terms, strict=True)
                ]
                if named_terms:
                    post_mu, post_sigma = hierarchical_fuse(
                        named_terms,
                        layers=state.hierarchical_layers,
                        coupling=state.hierarchical_coupling,
                    )
                else:
                    post_mu, post_sigma = fuse_terms(terms)
            else:
                post_mu, post_sigma = fuse_terms(terms)
        elif state.mood_enabled:
            # 把「已在的心境」作为第三个精度加权先验并入融合（当前被过去弯折）
            prev_mood = state.mood if state.mood is not None else (0.0, 0.0)
            prior_prec = (
                1.0 / max(MIN_SIGMA, state.prior_sigma[0]) ** 2,
                1.0 / max(MIN_SIGMA, state.prior_sigma[1]) ** 2,
            )
            # precision_commensurable：同 workspace 分支的 mood 处理。evidence 的 pi 来自
            # value.py，门开时已在那里齐次化，此处不重复处理。
            mood_prec_nb = (
                mood_precision(commensurable=True)
                if state.precision_commensurable
                else state.mood_precision
            )
            post_mu, post_sigma = fuse_terms(
                [
                    (state.prior_mu, prior_prec),
                    (evidence, (pi, pi)),
                    (prev_mood, (mood_prec_nb, mood_prec_nb)),
                ]
            )
        else:
            post_mu, post_sigma = gaussian_fuse(state.prior_mu, state.prior_sigma, evidence, pi)
        # rng_seed 为空时每次调用都重新随机（有意：生产情绪表达的随机性），
        # 非漏传 seed；测试需可复现时显式传 rng_seed。
        rng = random.Random(state.rng_seed) if state.rng_seed is not None else None
        # P4（议会 α，数学+神经一致）：map 读出取后验均值 e*=post_mu（MMSE 最优点估计，消单样本大
        # 方差致的逐轮翻号，时序连续性交既有 emotion_decay_step 的 AR1≈0.4 承担）；默认 sample=逐轮
        # 后验采样（逐字旧行为，零回归）。sample_affect 保留——可供表达层做简并可变性。
        if state.affect_readout == "map":
            e_star = post_mu
        else:
            e_star = sample_affect(post_mu, post_sigma, rng=rng, sigma_cap=state.sample_sigma_cap)
        entry: dict = {
            "node": "affect_core",
            "post_mu": post_mu,
            "post_sigma": post_sigma,
            "affect_sample": e_star,
        }
        out: dict = {
            "post_mu": post_mu,
            "post_sigma": post_sigma,
            "affect_sample": e_star,
            "trace": [entry],
        }
        # 工作空间额外产出：点燃的流名 + 后验精度（供语言层精度加权再入）。
        # 默认关时返回 dict 与 trace 与 v1/v2 逐字一致（零回归）。
        if state.workspace_enabled:
            entry["ignited_streams"] = ignited
            out["ignited_streams"] = ignited
            out["affect_precision"] = 0.5 * (1.0 / post_sigma[0] ** 2 + 1.0 / post_sigma[1] ** 2)
            # Kish 有效流数（议会 2026-07-28 第四轮 D5）：**纯观测量**。读数 →1 = 后验实际
            # 由单流决定。⚠ 不得据此下硬断言——单流主导在合法校准下也会发生（见函数 docstring）。
            #
            # ⚠ 措辞要准：`entry` **就是** `out["trace"][0]` 的同一个对象（列表存引用），
            # 故 n_eff **确实随 trace 进 state 并被 Checkpointer 持久化**。它不是"游离于
            # state 之外"。成立的不变式只有两条：① 不是独立的顶层 out 字段；
            # ② 全仓无任何下游读 `trace[...]["n_eff"]`，故不参与计算、不做门。
            # 体量上是一对 float，属 `state.py` 模块 docstring
            # 「trace 仅存标量中间量」的允许范围。
            #
            # 只在 comm 门开时写：`workspace_enabled` 是早于本项就已上线的**独立**旗标，
            # 本项承诺的「默认关=零回归」只覆盖 precision_commensurable。若不加这层判断，
            # 已开 workspace 但没开本门的用户 trace 会静默多一个键——那是本项范围外溢。
            if comm:
                entry["n_eff"] = effective_stream_count(terms)
        logger.debug(
            "affect_core e*=%s post_mu=%s post_sigma=%s ignited=%s",
            e_star,
            post_mu,
            post_sigma,
            ignited,
        )
        return out
