"""AppraisalAgent：OCC 评价节点（理性先验），产出先验分布与 reward。

对应 mPFC/ACC + 杏仁核效价标记。产出的 reward = 目标一致性，供 ValueAgent
做 TD（闭合评价 2 ↔ 价值 3）。若 `state.recalled_disposition` 存在（记忆读闭环），
按 RECALL_BIAS_WEIGHT 把长期情绪倾向叠加到先验 valence——当前评价被长期人格弯折。
reward 不受回灌影响（保持目标一致性），故 TD/价值通路不变。

P3 1-B HPA 皮质醇慢回路：本节点兼负 cortisol 更新（cortisol_enabled=True 时）。
时钟注入：构造时传入 `now_fn`（默认 time.time），在节点内调用、算好 delta_t 后
传给纯函数 cortisol_step——纯函数体内绝不触碰时钟（CS 红线）。

节点契约：(state) -> dict，只返回增量。
"""

from __future__ import annotations

import time
from collections.abc import Callable

from src.agents.affect_math import (
    CORTISOL_CAP,
    CORTISOL_IMPULSE,
    CORTISOL_TAU_DECAY,
    CORTISOL_THETA_GOAL,
    CORTISOL_THETA_INTENSITY,
    clamp,
    cortisol_step,
    cortisol_trigger,
    occ_prior,
)
from src.orchestration.state import AffectState

RECALL_BIAS_WEIGHT = 0.3  # 长期倾向对先验 valence 的偏置强度


class AppraisalAgent:
    """规则化 OCC 评价 → valence-arousal 先验 N(mu, sigma) + reward，可选回灌偏置。

    P3 1-B：当 state.cortisol_enabled=True 时，同步更新皮质醇慢回路：
    1. 时钟调用（编排层）：delta_t = now_fn() - state.cortisol_updated_at
    2. 触发判据（触发解耦，只读 appraisal 输入）：
       impulse = cortisol_trigger(goal_congruence, intensity)
    3. 状态更新（纯函数，收算好的 delta_t）：cortisol_new = cortisol_step(cortisol, delta_t, ...)
    4. 消费 cortisol→arousal：cortisol_arousal_gate 开时，arousal_baseline 叠加偏置

    now_fn 注入（对齐 build_graph 现有 memory/decoder 注入模式）：
    - 生产：默认 time.time（编排层节点自然可调时钟）
    - 测试：注入确定性时钟函数（如 lambda: 1000.0）可复现验证纯函数行为
    """

    def __init__(self, now_fn: Callable[[], float] = time.time) -> None:
        self.now_fn = now_fn

    def __call__(self, state: AffectState) -> dict:
        stim = state.stimulus
        if stim is None:
            return {}

        # ── HPA 皮质醇更新（cortisol_enabled 总门，默认关=零回归）──
        cortisol_updates: dict = {}
        if state.cortisol_enabled:
            # 时钟注入：wall-clock 只在此编排层节点读，纯函数只收 delta_t（CS 红线）
            now = self.now_fn()
            prev_t = state.cortisol_updated_at
            delta_t = now - prev_t if prev_t is not None else 0.0
            # 触发解耦：只读 appraisal 输入（goal_congruence/intensity）
            # 绝不读 arousal/emotion 状态（防 runaway，∂I/∂c≡0）
            # 动力学常数：state 字段 None → 回退 affect_math 议会推荐常量（零回归）；
            # 非 None（ZERO_CORTISOL_TAU/IMPULSE/THETA_* env 注入）→ 覆盖。
            tau = state.cortisol_tau if state.cortisol_tau is not None else CORTISOL_TAU_DECAY
            imp = state.cortisol_impulse if state.cortisol_impulse is not None else CORTISOL_IMPULSE
            tg = (
                state.cortisol_theta_goal
                if state.cortisol_theta_goal is not None
                else CORTISOL_THETA_GOAL
            )
            ti = (
                state.cortisol_theta_intensity
                if state.cortisol_theta_intensity is not None
                else CORTISOL_THETA_INTENSITY
            )
            impulse = cortisol_trigger(
                stim.goal_congruence,
                stim.intensity,
                theta_goal=tg,
                theta_intensity=ti,
                impulse=imp,
            )
            cortisol_new = cortisol_step(
                state.cortisol_state,
                delta_t,
                tau_decay=tau,
                impulse=impulse,
                cap=CORTISOL_CAP,
            )
            cortisol_updates = {
                "cortisol_state": cortisol_new,
                "cortisol_updated_at": now,
            }
        else:
            cortisol_new = state.cortisol_state

        # ── cortisol → arousal 基线抬升（cortisol_arousal_gate 门，默认关=零回归）──
        # 只抬 arousal 不动 valence（cortisol 不决定效价，valence 由 OCC 目标评价定）
        arousal_offset = (
            state.cortisol_arousal_alpha * cortisol_new
            if (state.cortisol_enabled and state.cortisol_arousal_gate)
            else 0.0
        )
        # 叠加 state.arousal_baseline（P1-c Q7 基准平移）
        effective_arousal_baseline = state.arousal_baseline + arousal_offset

        # ── C（A-P2-A）：va_coupling 非对称。state 字段 None → 不传关键字参数，
        # occ_prior 用默认 0.6/0.6（零回归）；非 None（persona/env 注入）→ 传入，
        # 启用 Kuppens 2013 negativity bias 非对称系数。
        coupling_kwargs: dict[str, float] = {}
        if state.va_coupling_pos is not None:
            coupling_kwargs["va_coupling_pos"] = state.va_coupling_pos
        if state.va_coupling_neg is not None:
            coupling_kwargs["va_coupling_neg"] = state.va_coupling_neg
        prior_mu, prior_sigma, reward = occ_prior(
            stim.goal_congruence,
            stim.standard_compliance,
            stim.attitude_appeal,
            stim.intensity,
            arousal_baseline=effective_arousal_baseline,
            **coupling_kwargs,
        )
        if state.recalled_disposition is not None:
            biased_v = clamp(
                prior_mu[0] + RECALL_BIAS_WEIGHT * state.recalled_disposition, -1.0, 1.0
            )
            prior_mu = (biased_v, prior_mu[1])

        # ── P3 1-C ToM / 社会情绪：共情偏置（图外 interlocutor_affect 标量注入后消费）──
        # 消费侧纯 math·无 LLM/torch；None/各 w=0 → 零回归（逐字不影响 prior_mu）。
        # 叠加顺序：① 传染（常开·差分双维）→ ② CARE(v_i<0) / ③ 替代喜悦(v_i>threshold)，
        # ②③互斥（v<0 走②·v>threshold 走③；v∈[0,threshold] 两者均不触发）→ 末尾 clamp。
        if state.interlocutor_affect is not None:
            v_i, a_i = state.interlocutor_affect
            v0, a0 = prior_mu

            # ① 传染（常开·差分双维；w_c=1 全同步、w_c=0 不变、凸组合天然有界）
            w_c = state.contagion_alpha
            v0 += w_c * (v_i - v0)
            a0 += w_c * (a_i - a0)

            # ② CARE（对方 v_i<0 触发；relu(−v_i)；抬高自身 valence/CARE 先验）
            # 触发器：对方 v_i<0（对方痛苦）；非对方 motivational_system=="care" 标签（那是满足态）
            if v_i < 0:
                care_bias = state.care_bias_alpha * max(-v_i, 0.0)
                v0 += care_bias

            # ③ 替代喜悦（对方 v_i>threshold 且 a_i>0；v1 非竞争默认·只替代喜悦）
            # ②③互斥：v_i<0 走②·v_i>threshold(>0) 走③；v∈[0,threshold] 均不触发
            elif v_i > state.vicarious_threshold and a_i > 0.0:
                vic_bias = state.vicarious_alpha * v_i
                v0 += vic_bias

            prior_mu = (clamp(v0, -1.0, 1.0), clamp(a0, -1.0, 1.0))

        appraisal = {"valence": prior_mu[0], "arousal": prior_mu[1]}
        entry = {
            "node": "appraisal",
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
            "reward": reward,
        }

        # ── coping_potential 独立标量流（议会 2026-07-13；默认关=零回归）──
        # 来源独立于 occ_prior VA 路径；绝不读 goal_congruence（T2 裁决：来源须正交）。
        # enabled=False（默认）→ 不改任何字段，逐字零回归。
        #
        # B3 四分支融合（议会 2026-07-16 Q3=B3）：
        #   ctrl = stim.control_appraisal（float|None；None=absent cue，精度趋零不参与融合）
        #   text = state.text_coping_prior（float|None；None=门关=零回归）
        #
        #   分支1：两皆 None → cp=0.0, src_flag=False（数值等价旧行为，R1 逐字零回归）
        #   分支2：仅 text（ctrl None）→ cp=clamp(text), src_flag=True（纯文本先验）
        #   分支3：仅 ctrl（text None）→ cp=clamp(ctrl), src_flag=False（旧行为零回归，R4）
        #   分支4：两皆有 → MLE 精度加权（Ernst & Banks 2002）：
        #            π_ctrl=1.0（固定·议会定；过程量位阶高于感受代理）
        #            π_t=state.text_coping_precision（ZERO_TEXT_COPING_PRECISION env，≤0.10）
        #            cp = clamp((π_ctrl·clamp(ctrl) + π_t·clamp(text)) / (π_ctrl+π_t))
        #            src_flag=True（text 参与融合，供中间带哑火消费）
        #   不复用 fuse_terms / hierarchical_fuse（守来源正交红线）。
        #
        # ⚠ 分支可达性按入口而异（2026-07-30 议会 D5a，三席现场核验）：
        #   `orchestration.chat_driver` 从不注入 Stimulus.control_appraisal ⇒ ctrl 恒 None ⇒
        #   chat 路径只走分支1/2，**分支4 从未被触发**，ZERO_TEXT_COPING_PRECISION(π_t) 在该路径下
        #   是 inert 参数（分支2 是 cp=clamp(text) 直通覆盖，不经精度加权、无跨轮平滑）。
        #   仅 `mcp_server.mapping` 会注入 control_appraisal 使分支4 可达。
        #   ⇒ 任何「低精度先验阻尼」类表述必须限定 scope，勿跨入口套用。
        #
        # ⚠ 已知局限（神经席约束7；不粉饰）：
        #   text_coping_prior 学自 EmoBank SAM Dominance，为感受状态代理（主观量），
        #   而 control_appraisal 是次级评价过程量（Smith & Ellsworth 1985 controllability）；
        #   测量层次差异在 writer-D IAA≈0.54 天花板下筛选后仍残留，靠 π_t≤0.10 对冲。
        #
        # ── 非对称可靠性与命名边界（议会 2026-07-20·SemEval OOD 后订正·域条件化解锁）──
        # ── A2：anger 三源 R 披露（数学席·议会 2026-07-21·可采纳）──
        # anger 三源 confrontational 域 Wilson LB：SemEval 0.857 / EmoryNLP 0.776 / DailyDialog 0.730。  # noqa: E501
        # P2 域散度：range=极差 0.127 < 0.20 → d_HΔH(confrontational 各体裁) 可估·泛化误差上界
        #   ≤ 0.127/2 = 0.064（Ben-David et al. 2010 DOI:10.1007/s10994-009-5152-4）。
        # R 披露（须明注）：
        #   (a) ED 叙事域 anger LB≈0.68 < bar(0.70) → 不解锁（域失配·行动路径关闭·Kelley 2013）。
        #   (b) E-c 多标签（SemEval emotion-c）宽于纯 approach-anger → LB=0.857 对纯 approach-anger
        #       可能轻微高估（宽定义·残余不确定性·数学席）。
        #   (c) τ 弃权门当前无统计增益（<2pp·FWER≈18.5%·见 perception.py A4 注释）。
        # 比 fear 单源（d_HΔH 未量化·单源幻觉历史）信息论保证更强——见 A6 注释。
        #
        # ① 非对称来源与**双向域依赖**（可靠性是域特异的·非本质属性）：fear 回避锚皮层下
        #    PAG/杏仁核生存回路、anger 趋近依赖前额叶可行动性评价。
        #    ── A5：Harmon-Jones 2003 双引区分（议会 2026-07-21）──
        #    Harmon-Jones et al. (2003) EEG 实验直证（anger coping & frontal cortex·
        #      Cognition & Emotion 17(1):1-24·PubMed:29715737）→ anger 前额叶激活·趋近动机直证。
        #    Harmon-Jones, E. (2003) BAS 理论综述（anger & behavioral approach system·
        #      Pers. Individ. Differ. 35:995-1005·DOI:10.1016/S0191-8869(02)00313-6）→ BAS 理论框架。  # noqa: E501
        #    两文作者部分重叠但体裁不同（Cog&Emo=EEG 实验·PAID=理论综述）；勿混引。
        #    实测（SemEval-2018 Twitter vs EmpatheticDialogues）——anger/fear **各有弱域**：
        #      anger：confrontational 域（Twitter·当下·可行动性在线）LB=0.857 恢复；叙事/倾诉域
        #        （ED·行动路径关闭·Kelley 2013 沉思型右额叶）LB≈0.68。~74% 是 ED 域下限、非普遍。
        #      fear：ED 生存/防御型（CeA-PAG·情境无关）LB≈0.90；Twitter 社交焦虑/apprehension
        #        （BNST-皮层依赖·情境依赖·Davis&Walker 2009）LB=0.709（仅过 bar）——「fear 全域稳」
        #        是 ED 单源幻觉。
        #    「情境无关」限皮层下生存回路的**防御行为触发**层面（LeDoux&Brown 2017），非意识 fear
        #    感受本身（Barrett 2017）。E-c 多标签 anger 宽于纯 approach-anger·LB=0.857 对后者可能
        #    轻微高估（残余不确定性·数学席·见 A2 R 披露(b)）。详见 notes/2026-07-20-anger-unlock-decision-council。  # noqa: E501
        # ② 与 coping_potential 的边界（议会三轮正名）：本 text 先验的**训练方案**已定为符号
        #    监督 motivational_direction_prior，是 anger/fear 类别的**趋近-回避方向符号先验**，
        #    非 Lazarus/Scherer 意义的应对评价连续量；上方 SAM-D 段描述的是被取代的旧回归
        #    路线（留档解释 π_t 对冲的由来）。**W1 已接线（2026-07-20）**：PerceptionAgent 经
        #    DirectionHead（ZERO_DIRECTION_HEAD_MODEL_PATH 门控·opt-in）产出 text_coping_prior；
        #    未设权重 / text_coping_enabled=False（默认）则恒 None、只走分支1/3（零回归）。
        # ③ 解锁状态（议会 2026-07-20·**域条件化解锁·取代 δ 全域弃权**）：方向门(SemEval anger
        #    LB=0.857)+独立性门(r=−0.031)均过→**confrontational 域内解锁依据充分**（非全域无条件·
        #    ED 域仍 FAIL）。工程＝opt-in（ZERO_TEXT_COPING_ENABLED 默认仍 False·零回归）+ 域判别由
        #    **调用方注入 control_appraisal**（confrontational 注入·narrative 保 None 回退分支1/3·
        #    热路径不建域判别器）；真正生效前置=②的 W1 接线。anger 高置信才用、低置信约 12–20% 弃权
        #    （无力型/沉思型 anger 噪声）；调用者不得假设 anger 信号总在。
        # ④ fear 对称域条件化（议会 2026-07-20·EmoryNLP 第二源后·选项 α 重构）：δ「fear 全域稳」
        #    证伪——fear 三源 ED 生存域 LB≈0.90→SemEval 社交焦虑 0.709→EmoryNLP 表演对话 0.264 崩
        #    （anger 反双源稳 0.857/0.776）。fear 也域依赖·home 域与 anger 相反（fear=生存叙事·
        #    anger=confrontational）。
        #    【CeA/BNST 标签保守化·机制假说非确立事实】：CeA phasic 生存 fear / BNST sustained
        #    社交焦虑的解剖映射是**机制假说**（Davis&Walker 2009 DOI:10.1038/npp.2009.109），
        #    Grogans et al. (2023 SCAN) meta 分析显示 CeA/BNST 无显著区域×确定性解离（≈随机），
        #    人类实证效度不足——故域定义以体裁/情境为锚，不以解剖结构为一级依据。
        #    【B2 已落地·2026-07-20】：Stimulus.domain 正交字段（state.py）+ perception.py
        #    _domain_direction_accepts 热路径域门均已实现；off-domain text_coping_prior 硬弃
        #    ≡ 该域 π_t 近似为 0（硬弃非无理论工程约定·Ernst&Banks 2002 MLE）。
        #    fear 生产解锁默认关（B-fe）：议会 2026-07-21（DailyDialog 第三源后）裁**维持默认关**。
        #    【DailyDialog 第三源·四源梯度】ED 0.90>SemEval 0.709>DailyDialog 0.582>EmoryNLP 0.264。
        #    对话域 fear sub-bar 有**两层原因**（心理/神经席·勿混为一）：(a) 构念异质——live-chat
        #    「害怕」多为 Lazarus anxiety（不确定）非 fear（即时具体威胁·Lazarus 1991），方向信号
        #    不稳（主因）；(b) 体裁放大——表演台词夸张高唤醒→方向混淆（EmoryNLP 0.264·DailyDialog
        #    自然对话 0.582 高出 +32pp 证明两层各有贡献·非纯表演 artifact；Bossuyt 2014 fear 方向
        #    目标依赖可转趋近·对话域非零信号）。
        #    【B-pi·π_t(fear) 推荐 0.08】闭式映射 π_t(fear)=π_t(anger 0.07)×(LB_fear 0.90/LB_anger
        #    均值 0.821)≈0.077→取 0.08（Ben-David 2010·不超 0.10）；fear 默认关**不 set .env**·
        #    待解锁裁；对话域域门硬弃→π_t 不生效（仅 survival 域才可能生效）。
        #    【R 条件·单源 OOD 残余风险】ED 体裁⊂survival 全分布·非等于（d_HΔH 未量化·单源解锁属
        #    工程近似非数学充分·「单源幻觉」历史）；残余风险 ε≤p_mis+(1−LB_ED)≈p_mis+0.10。
        #    须第二生存叙事源（LB≥0.80）才升为可解锁——见
        #    notes/2026-07-21-dailydialog-fear-unlock-council.md。
        # ── A6：anger 三源信息论保证（神经席·议会 2026-07-21）──
        # anger 三体裁（SemEval Twitter / EmoryNLP / DailyDialog confrontational）交叉收敛效度：
        #   三源 d_HΔH 可估（Ben-David 2010）·极差 0.127·泛化误差上界 ≤0.064。
        #   Harmon-Jones & Gable 2018（Psychophysiology DOI:10.1111/psyp.12879）跨 30+ 实验室
        #   确认 confrontational anger 左前额叶 BAS·体裁无关的神经机制收敛。
        #   三体裁交叉收敛效度比 fear 单源（d_HΔH 未量化·单源幻觉·「ED 0.90」跨域即崩）
        #   信息论保证**更强**——这是议会 2026-07-21 四席 PASS anger 的核心依据。
        #   fear 解锁须第二生存叙事源 LB≥0.80（单源 ED 幻觉历史·d_HΔH 须量化）；
        #   当前 fear 维持默认关（B-fe·ZERO_FEAR_DOMAIN_ENABLED 默认 False）。
        coping_updates: dict = {}
        if state.coping_potential_enabled and stim is not None:
            ctrl = stim.control_appraisal  # float | None
            # ── text_coping_enabled 门控（BLOCK-1）──
            # False（默认）→ text 强制视为 None，只走分支1/3（纯 ctrl 路径，零回归）；
            # True → 允许 text_coping_prior 参与 B3 融合（分支2/4）。
            text = state.text_coping_prior if state.text_coping_enabled else None
            if ctrl is None and text is None:
                # 分支1：两皆 None → 强制 cp=0.0（每轮覆盖，非保持上轮值）；src_flag=False
                # ⚠ 0.0 是契约值（2026-08-31 Dominance 议会再确认，≥5 处锚点测试锁死）。
                # 未来任何 trait→state 静息基线（不限 Dominance）须走解耦通道（新增可选
                # 字段、None 回退此字面量），不得覆写。fear 跨层不一致已复裁为**有意的
                # 「面部泄漏」行为**（同日轻量门；锚点 test_fear_crosslayer_leakage）——
                # 且注意：分支3（ctrl 直接注入）从不经过任何 fear 域门，负 ctrl 点亮
                # fear-AU 今天即可达，这是 mcp_server 契约层该向调用方披露的行为。
                cp: float = 0.0
                src_flag: bool = False
            elif ctrl is None:
                # 分支2：仅 text（B3：absent ctrl 精度趋零，不参与融合）
                assert text is not None  # 分支1 已排除两皆 None；此处 text 必非 None
                cp = clamp(text, -1.0, 1.0)
                src_flag = True
            elif text is None:
                # 分支3：仅 ctrl（旧行为路径，R4 零回归）
                cp = clamp(ctrl, -1.0, 1.0)
                src_flag = False
            else:
                # 分支4：两皆有 → MLE 精度加权（Ernst & Banks 2002）
                pi_ctrl: float = 1.0  # 固定·议会定（过程量位阶高于感受代理）
                pi_t: float = state.text_coping_precision  # ≤0.10（SessionConfig 层 fail-fast）
                ctrl_c = clamp(ctrl, -1.0, 1.0)
                text_c = clamp(text, -1.0, 1.0)
                cp = clamp(
                    (pi_ctrl * ctrl_c + pi_t * text_c) / (pi_ctrl + pi_t),
                    -1.0,
                    1.0,
                )
                src_flag = True
            coping_updates = {"coping_potential_state": cp, "text_coping_source": src_flag}
        else:
            # coping_potential_enabled=False（默认）→ 不改 coping_potential_state；
            # text_coping_source 归零为 False，防绕过 step() 的调用路径残留上轮 True（WARN-2）。
            coping_updates = {"text_coping_source": False}

        return {
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
            "reward": reward,
            "appraisal": appraisal,
            "trace": [entry],
            **cortisol_updates,
            **coping_updates,
        }
