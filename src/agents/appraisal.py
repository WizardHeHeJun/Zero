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
        coping_updates: dict = {}
        if state.coping_potential_enabled and stim is not None:
            cp = clamp(stim.control_appraisal, -1.0, 1.0)
            coping_updates = {"coping_potential_state": cp}

        return {
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
            "reward": reward,
            "appraisal": appraisal,
            "trace": [entry],
            **cortisol_updates,
            **coping_updates,
        }
