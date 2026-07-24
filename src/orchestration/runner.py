"""研究原型运行入口：跑 stimulus 序列，产出 (v,a) 轨迹与逐步中间量。

带 thread_id 走 Checkpointer，运行态（含 value_table）跨刺激持久化，
从而可观测 ValueAgent 的在线学习。轨迹可 dump 成 JSON 供观测。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field, model_validator

from src.agents.expression import ChannelDecoder
from src.agents.language import LanguageModel
from src.memory.client import MemoryClient
from src.orchestration.graph import build_graph
from src.orchestration.state import AffectState, Stimulus
from src.storage.checkpointer import build_checkpointer
from src.storage.graph_store import build_graph_store, build_semantic_store

logger = logging.getLogger(__name__)


class SessionConfig(BaseModel):
    """会话级门控旋钮的收口模型（D·架构师 E-P1-C）。

    收拢 ConversationSession.__init__ 和 run() 里散落的所有会话级 flags，
    使新增旋钮只需在此处一处加字段。**不改 AffectState 字段定义本身**（避免破坏
    checkpoint 序列化），字段在 ainvoke 时经 model_dump() 展开注入 state 初值。

    默认值与旧 ConversationSession.__init__ 参数默认逐字对齐，确保零回归：
    不传 SessionConfig / 传 None → 行为与改动前完全一致。
    """

    # ── 基础门控 ──
    regulation_enabled: bool = False
    regulation_strategy: str = "suppression"
    mood_enabled: bool = False
    recall_enabled: bool = False
    language_enabled: bool = False
    workspace_enabled: bool = False
    appraisal_conditioning_enabled: bool = False
    language_max_iters: int = 3
    rng_seed: int | None = None
    sample_sigma_cap: float | None = None
    affect_readout: str = "sample"
    arousal_baseline: float = 0.0
    arousal_gain_cap: float | None = None
    # ── A. workspace 三 flag（B4/B5；默认 False = 零回归）──
    # 不设对应 env 时行为与改前完全一致。
    precision_split: bool = False
    fuse_independence_correct: bool = False
    ignition_survival_fallback: bool = False
    # ── C. va_coupling 非对称系数（A-P2-A；默认 None = occ_prior 用 0.6/0.6 = 零回归）──
    va_coupling_pos: float | None = None
    va_coupling_neg: float | None = None
    # ── E. Panksepp RAGE/FEAR 次级区分（WARN-4 / A-P1-D；默认 False = 零回归）──
    # True 时 motivational_system 在 (-v,+a) 象限按 arousal 阈值分 fear/rage；此处仅贯通
    # on/off 开关，精确阈值/坐标待议会 P1-D。
    panksepp_distinguish_fear: bool = False
    # ── ignition_beta（议会 2026-07-02 Item 2；默认 None = 硬 step 零回归）──
    # 非 None → ignite() logistic 软门控；推荐区间 [20,50]，典型值 20。
    ignition_beta: float | None = None
    # ── T6-a · mood 流精度旋钮（默认 0.8 = MOOD_PRECISION 常量，逐字零回归）──
    # 由 ZERO_MOOD_PRECISION env 注入；未设 → 0.8 → 行为与改前完全一致。
    mood_precision: float = 0.8
    # ── T6-b · 文本语义流精度旋钮（默认 0.3 = TEXT_AFFECT_PRECISION 常量，逐字零回归）──
    # 由 ZERO_TEXT_AFFECT_PRECISION env 注入；未设 → 0.3 → 行为与改前完全一致。
    text_affect_precision: float = 0.3
    # ── P3 · HPC 层级预测编码旋钮（默认 1/0.0 = 平层零回归，逐字等价现 fuse_terms）──
    # hierarchical_layers=1 或 hierarchical_coupling=0.0 → 退化 fuse_terms(all)（双退化）；
    # layers=2, coupling∈(0,1] → 启用 2 层 HPC（L0=survival+mood / L1=其余）；
    # coupling>1 → affect_math.hierarchical_fuse raise ValueError（硬拒不 clamp）。
    # 由 ZERO_HPC_LAYERS / ZERO_HPC_COUPLING env → chat_driver → ConversationSession 透传。
    hierarchical_layers: int = 1
    hierarchical_coupling: float = 0.0
    # ── P3 1-B · HPA 皮质醇慢回路旋钮（默认全关=零回归；config-only-via-env）──
    # cortisol_enabled=False 总门 → 现路径逐字不变。
    # cortisol_arousal_gate/attitude_gate：各作用子门（默认关=零回归）。
    # 常数建议值：tau≈5400s / impulse 0.6-0.8 / arousal_alpha 0.1-0.3 / attitude_alpha≤1(f_max≤2)
    # 触发解耦：cortisol_trigger 只读 appraisal 输入，不读 arousal/emotion（防 runaway）。
    # cortisol 只进 Checkpointer，绝不入图谱（CS 红线；memory.write 不含 cortisol）。
    cortisol_enabled: bool = False
    cortisol_arousal_gate: bool = False
    cortisol_attitude_gate: bool = False
    cortisol_arousal_alpha: float = 0.0  # 默认 0 → offset=0 → 零回归
    cortisol_attitude_alpha: float = 0.0  # 默认 0 → rate_eff=ATTITUDE_RATE×1 → 零回归
    # 动力学常数（None → AppraisalAgent 回退 affect_math 常量=零回归；env 覆盖）
    cortisol_tau: float | None = None
    cortisol_impulse: float | None = None
    cortisol_theta_goal: float | None = None
    cortisol_theta_intensity: float | None = None
    # ── coping_potential 独立标量流（议会 2026-07-13；默认关=零回归）──
    # coping_potential_enabled=False → AppraisalAgent 不产 coping_potential_state → 词典层零回归。
    # coping_potential 只进 Checkpointer，绝不写入图谱（CS 红线）。
    coping_potential_enabled: bool = False
    # ── text_coping 接线旋钮（议会 2026-07-16 B3；默认关=零回归）──
    # text_coping_enabled=False → AppraisalAgent B3 走分支1/3（仅 ctrl 路径）→ 词典层零回归。
    # text_coping_source 是 AppraisalAgent 输出 flag，不是会话级输入，不进 SessionConfig。
    text_coping_enabled: bool = False
    # ── fear_domain_enabled：WARN-3 fear 专属门（B1 BLOCK 前置·议会 2026-07-21·A1）──
    # False（默认）→ 两条泄漏路径均硬弃 fear 域激活·与生产/chat 零回归一致；
    # True → 解除硬弃，须 env 显式开（ZERO_FEAR_DOMAIN_ENABLED）。
    # anger confrontational 路径完全不受此门（仅 survival_narrative 域关）。
    fear_domain_enabled: bool = False
    # π_t 精度上界（le=0.10 在此层 fail-fast；AffectState 层不加 le 防 checkpoint 反序列化 fail）。
    # 单源 EmoBank·议会 2026-07-16·π_t≤0.10·方向判据过后 CI 下界≥0.70 可升 0.15。
    text_coping_precision: float = Field(default=0.08, gt=0.0, le=0.10)
    # ── facs_extended：AU 扩展集合门控（设计门 PASS·路径 b；默认关=零回归）──
    # True → ExpressionAgent 占位路径把 coping_potential_state 透传给 decode_channels，
    # 启用 11-AU 扩展集合（FACS_KEYS_EXT）；False=旧 5-AU 逐字行为（零回归）。
    # 占位路径（decoder=None）经 ExpressionAgent 消费；注入 decoder 若实现 predict_channels_coping
    # 也拿到 per-turn coping（议会遗留 2·方案 b 已落地），否则回退 predict_channels(v,a)。
    facs_extended: bool = False
    # ── canonical_physiology：physiology 占位口径门控（议会 2026-07-23；默认关=零回归）──
    # True → 占位出 canonical {hr[50,120]/sc μS[0,20]/temperature_c[33,36]}；
    # False（默认）→ legacy {hr[70,110]/sc[0,1]/pupil_mm[3,5]}，逐字零回归。
    # 经 ZERO_PHYSIOLOGY_CANONICAL_PLACEHOLDER → chat_driver → SessionConfig → to_state_flags。
    canonical_physiology: bool = False
    # ── voluntary_coping_leak：双通路差异化（议会 C1 设计门 2026-07-14）∈[0,1]──
    # 自发头全量传 coping、随意头传 coping×leak（意志部分压制 coping-driven AU）。
    # 默认 1.0=两头等值=零回归；推荐 0.3。仅 facs_extended=True 时对 facs_au 生效。
    voluntary_coping_leak: float = Field(default=1.0, ge=0.0, le=1.0)

    # ── 外部多模态先验流注入口（议会 2026-07-15 M3/M6；config-only-via-env）──
    # external_priors 本身是每轮 state_overrides 内容（同 interlocutor_affect），不在此收口。
    # 此处只存会话级固定的校验参数（precision_cap / max_streams）。
    # ZERO_EXTERNAL_PRIOR_PRECISION_CAP 默认 0.8；ZERO_MAX_EXTERNAL_STREAMS 默认 5。
    external_prior_precision_cap: float = Field(default=0.8, gt=0.0)
    max_external_streams: int = Field(default=5, ge=0)

    # ── P3 1-C · ToM / 社会情绪共情旋钮（默认全 0/0.3 = 零回归；config-only-via-env）──
    # interlocutor_affect 是每轮可变标量（依赖 user_text），不在此收口；
    # 此处只存会话级固定系数（contagion/care/vicarious alpha + threshold）。
    # 默认全 0 → 共情偏置为 0 → 零回归（prior_mu 逐字不变）。
    # 安全区间：contagion≤0.3·推荐[0.05,0.25] / care≤0.4 / vicarious≤0.2 / L1≤0.6。
    # 数学席证的稳定性上界（W3/W4）：各系数硬上界 + L1 和 ≤0.6（防总偏置破 mood 双稳）。
    contagion_alpha: float = Field(default=0.0, ge=0.0, le=0.3)
    care_bias_alpha: float = Field(default=0.0, ge=0.0, le=0.4)
    vicarious_alpha: float = Field(default=0.0, ge=0.0, le=0.2)
    vicarious_threshold: float = 0.3

    model_config = {"frozen": False}  # 允许工厂侧构造后修改（不需 model_copy）

    @model_validator(mode="after")
    def _check_empathy_l1(self) -> SessionConfig:
        """P3 1-C：三通道共情系数 L1 和 ≤0.6（数学席证：总偏置过大破 mood 双稳/attitude 收敛）。"""
        l1 = self.contagion_alpha + self.care_bias_alpha + self.vicarious_alpha
        if l1 > 0.6:
            raise ValueError(
                f"共情系数 L1 和 {l1:.3f} >0.6（contagion+care+vicarious）；"
                f"数学席证会破 mood 双稳/attitude 收敛，请调低"
            )
        return self

    def to_state_flags(self) -> dict[str, Any]:
        """把 SessionConfig 展开成可直接传入 ainvoke 的 state 初值 dict。

        等价于旧 ConversationSession.flags dict + 新增旋钮，确保接线完整性。
        None 值也显式传入（AffectState 字段接受 None，不过滤）。
        """
        return self.model_dump()


# 编排层声明自己放进运行态、需从 checkpoint 恢复的自定义类型（存储层据此白名单）。
# Fact：AffectState.recalled_facts 携带（D1）；非 InMemory checkpointer（sqlite/postgres）
# 反序列化须白名单，否则还原成 dict 致下游 f.sim/f.content AttributeError。
ALLOWED_CHECKPOINT_TYPES = [
    ("src.orchestration.state", "Stimulus"),
    ("src.memory.types", "Fact"),
    ("src.memory.types", "Scope"),  # Fact.scope 是 Scope 枚举，随 Fact 一同反序列化需白名单
]


def _state_to_entry(stim_name: str, state: AffectState) -> dict[str, Any]:
    """把一次运行后的 AffectState 抽成可观测轨迹条目（run 与 ConversationSession 共用）。"""
    return {
        "stimulus": stim_name,
        "valence_arousal": state.affect_sample,
        "ignited_streams": state.ignited_streams,
        "affect_precision": state.affect_precision,
        "prior_mu": state.prior_mu,
        "prior_sigma": state.prior_sigma,
        "reward": state.reward,
        "rpe": state.rpe,
        "precision": state.precision,
        "value_estimate": state.value_estimate,
        "mood": state.mood,
        "recalled_context": state.recalled_context,
        "recalled_facts": state.recalled_facts,  # D1：供 ChatDriver 按 importance 注入 history
        "language_text": state.language_text,
        "language_affect": state.language_affect,
        "language_consistency": state.language_consistency,
        "language_iter": state.language_iter,
        "expression": state.expression,
        # P3 1-B：皮质醇慢变量（供 chat_driver 消费 cortisol→ATTITUDE_RATE；仅标量观测，不入图谱）
        "cortisol_state": state.cortisol_state,
        # coping_potential：情境控制感标量（可观测，供校准/观测；仅标量、不入图谱，同 cortisol）
        "coping_potential_state": state.coping_potential_state,
    }


async def run(
    stimuli: Sequence[Stimulus],
    *,
    thread_id: str,
    memory: MemoryClient | None = None,
    session_id: str | None = None,
    user_id: str = "default-user",
    group_id: str = "default-group",
    regulation_enabled: bool = False,
    regulation_strategy: str = "suppression",
    mood_enabled: bool = False,
    recall_enabled: bool = False,
    language_enabled: bool = False,
    workspace_enabled: bool = False,
    appraisal_conditioning_enabled: bool = False,
    language_max_iters: int = 3,
    rng_seed: int | None = None,
    sample_sigma_cap: float | None = None,
    affect_readout: str = "sample",
    arousal_baseline: float = 0.0,
    arousal_gain_cap: float | None = None,
    precision_split: bool = False,
    fuse_independence_correct: bool = False,
    ignition_survival_fallback: bool = False,
    va_coupling_pos: float | None = None,
    va_coupling_neg: float | None = None,
    panksepp_distinguish_fear: bool = False,
    ignition_beta: float | None = None,
    mood_precision: float = 0.8,
    text_affect_precision: float = 0.3,
    hierarchical_layers: int = 1,
    hierarchical_coupling: float = 0.0,
    # P3 1-B：HPA 皮质醇慢回路旋钮（默认全关=零回归）
    cortisol_enabled: bool = False,
    cortisol_arousal_gate: bool = False,
    cortisol_attitude_gate: bool = False,
    cortisol_arousal_alpha: float = 0.0,
    cortisol_attitude_alpha: float = 0.0,
    cortisol_tau: float | None = None,
    cortisol_impulse: float | None = None,
    cortisol_theta_goal: float | None = None,
    cortisol_theta_intensity: float | None = None,
    # P3 1-C：ToM 共情旋钮（默认 0/0/0/0.3 = 零回归）
    contagion_alpha: float = 0.0,
    care_bias_alpha: float = 0.0,
    vicarious_alpha: float = 0.0,
    vicarious_threshold: float = 0.3,
    # coping_potential 独立标量流（议会 2026-07-13；默认关=零回归）
    coping_potential_enabled: bool = False,
    # text_coping 接线旋钮（议会 2026-07-16 B3；默认关=零回归）
    text_coping_enabled: bool = False,
    text_coping_precision: float = 0.08,
    # fear_domain_enabled：WARN-3 fear 专属门（B1 BLOCK 前置·议会 2026-07-21；默认关=零回归）
    fear_domain_enabled: bool = False,
    # facs_extended：AU 扩展集合门控（设计门 PASS·路径 b；默认关=零回归）
    facs_extended: bool = False,
    # canonical_physiology：physiology 占位口径门控（议会 2026-07-23；默认关=零回归）
    canonical_physiology: bool = False,
    # voluntary_coping_leak：双通路差异化（议会 C1 设计门 2026-07-14；默认 1.0=零回归）
    voluntary_coping_leak: float = 1.0,
    # 外部多模态先验流注入口（议会 2026-07-15 M3/M6；默认=零回归）
    external_prior_precision_cap: float = 0.8,
    max_external_streams: int = 5,
    expression_decoder: ChannelDecoder | None = None,
    language_model: LanguageModel | None = None,
) -> list[dict[str, Any]]:
    """按序跑 stimuli，返回每个刺激的情绪向量与关键中间量。

    session_id 默认绑定到 thread_id，使不同线程的会话记忆天然隔离（防串味）；
    user_id/group_id 应由调用方按真实身份显式传入。
    expression_decoder：可选注入训练好的真通道解码器，走真网络表达。
    language_model：可选注入的语言模型（鸭子类型），开启 language_enabled 后驱动
    affect↔language 双向收敛回路；未注入则用占位模板模型。

    external_prior_precision_cap / max_external_streams 是外部先验流的**校验**参数（M3/M6）；
    external_priors 本身（每轮的 (name,(μv,μa),(Πv,Πa)) 数据）**不经 run() 注入**——它是每轮
    可变量，经 `ConversationSession.step(stim, state_overrides={"external_priors": [...]})` 注入
    （同 interlocutor_affect）。run() 批量接口每条 stimulus 不携带外部先验（code-reviewer W5）。
    """
    client = (
        memory
        if memory is not None
        else MemoryClient(build_graph_store(), semantic=build_semantic_store())
    )
    session = session_id if session_id is not None else thread_id
    checkpointer = build_checkpointer(ALLOWED_CHECKPOINT_TYPES)
    graph = build_graph(
        checkpointer=checkpointer,
        memory=client,
        expression_decoder=expression_decoder,
        language_model=language_model,
    )

    trajectory: list[dict[str, Any]] = []
    for stim in stimuli:
        result = await graph.ainvoke(
            {
                "stimulus": stim,
                "session_id": session,
                "user_id": user_id,
                "group_id": group_id,
                "regulation_enabled": regulation_enabled,
                "regulation_strategy": regulation_strategy,
                "mood_enabled": mood_enabled,
                "recall_enabled": recall_enabled,
                "language_enabled": language_enabled,
                "workspace_enabled": workspace_enabled,
                "appraisal_conditioning_enabled": appraisal_conditioning_enabled,
                "language_max_iters": language_max_iters,
                "rng_seed": rng_seed,
                "sample_sigma_cap": sample_sigma_cap,
                "affect_readout": affect_readout,
                "arousal_baseline": arousal_baseline,
                "arousal_gain_cap": arousal_gain_cap,
                # B8 新增 5 旋钮（默认 False/None=零回归；对齐 ConversationSession/SessionConfig）
                "precision_split": precision_split,
                "fuse_independence_correct": fuse_independence_correct,
                "ignition_survival_fallback": ignition_survival_fallback,
                "va_coupling_pos": va_coupling_pos,
                "va_coupling_neg": va_coupling_neg,
                "panksepp_distinguish_fear": panksepp_distinguish_fear,
                # 议会 2026-07-02 Item 2：ignite 软门控陡度（None=硬 step 零回归）
                "ignition_beta": ignition_beta,
                # T6-a/b：mood/text 流精度旋钮（默认=现常量，逐字零回归）
                "mood_precision": mood_precision,
                "text_affect_precision": text_affect_precision,
                # P3 HPC 旋钮（默认 1/0.0=平层零回归）
                "hierarchical_layers": hierarchical_layers,
                "hierarchical_coupling": hierarchical_coupling,
                # P3 1-B：HPA 皮质醇慢回路旋钮（默认全关=零回归）
                "cortisol_enabled": cortisol_enabled,
                "cortisol_arousal_gate": cortisol_arousal_gate,
                "cortisol_attitude_gate": cortisol_attitude_gate,
                "cortisol_arousal_alpha": cortisol_arousal_alpha,
                "cortisol_attitude_alpha": cortisol_attitude_alpha,
                "cortisol_tau": cortisol_tau,
                "cortisol_impulse": cortisol_impulse,
                "cortisol_theta_goal": cortisol_theta_goal,
                "cortisol_theta_intensity": cortisol_theta_intensity,
                # P3 1-C：ToM 共情旋钮（默认 0/0/0/0.3=零回归）
                "contagion_alpha": contagion_alpha,
                "care_bias_alpha": care_bias_alpha,
                "vicarious_alpha": vicarious_alpha,
                "vicarious_threshold": vicarious_threshold,
                # coping_potential 独立标量流（议会 2026-07-13；默认关=零回归）
                "coping_potential_enabled": coping_potential_enabled,
                # text_coping 接线旋钮（议会 2026-07-16 B3；默认关=零回归）
                "text_coping_enabled": text_coping_enabled,
                "text_coping_precision": text_coping_precision,
                # fear_domain_enabled：WARN-3 fear 专属门（B1 BLOCK 前置·议会 2026-07-21；默认关）
                "fear_domain_enabled": fear_domain_enabled,
                # text_coping 每轮防御归零（INFO-2·与 ConversationSession.step() 基准一致）：
                # 批跑不逐轮注入 text_coping_prior，但显式归零防 checkpoint 残留（一致性+防御）。
                "text_coping_prior": None,
                "text_coping_source": False,
                # facs_extended：AU 扩展集合门控（默认关=零回归）
                "facs_extended": facs_extended,
                # canonical_physiology：physiology 占位口径门控（议会 2026-07-23；默认关=零回归）
                "canonical_physiology": canonical_physiology,
                # voluntary_coping_leak：双通路差异化（C1 设计门 2026-07-14；默认 1.0=零回归）
                "voluntary_coping_leak": voluntary_coping_leak,
                # 外部多模态先验流注入口（议会 2026-07-15 M3/M6；默认=零回归）
                "external_prior_precision_cap": external_prior_precision_cap,
                "max_external_streams": max_external_streams,
                "task_complete": False,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        # 本 langgraph 版本对 pydantic schema 的 ainvoke 返回 dict；AffectState(**result)
        # 会触发 pydantic 校验（字段缺失/类型不符即报错），不会静默掩盖问题。
        state = result if isinstance(result, AffectState) else AffectState(**result)
        logger.debug(
            "runner stim=%s e*=%s precision=%s",
            stim.name,
            state.affect_sample,
            getattr(state, "affect_precision", None),
        )
        trajectory.append(_state_to_entry(stim.name, state))
    return trajectory


def dump_trajectory(trajectory: list[dict[str, Any]]) -> str:
    """把轨迹序列化为 JSON 文本（研究观测用）。"""
    return json.dumps(trajectory, ensure_ascii=False, indent=2, default=str)


class ConversationSession:
    """多轮交互会话：建图/checkpointer **一次**，每轮 `step` 喂一个 stimulus。

    与 `run`（一次性跑完整序列）不同，本类把同一 checkpointer 留在内存中，逐轮 `ainvoke`
    复用同一 `thread_id` —— 故运行态（mood 心境、value_table 价值记忆）**跨轮持久**，
    情绪轨迹形成历史依赖（A.7 滞后在对话里显现）。供交互式对话入口（`main.py --chat`）使用：
    把"用户每句话"评价成 stimulus 喂进来，引擎演化的 e* 驱动语言层生成带情绪的回应。

    门控开关在构造时固定、贯穿整个会话。两种接法（向后兼容）：
    1. 传 `config=SessionConfig(...)` —— 新首选，新增旋钮只在 SessionConfig 一处加。
    2. 传旧展开参数（regulation_enabled/mood_enabled/...）—— 旧调用方零改动。
       两者同传时 `config` 优先（旧参数被忽略）。

    记忆/语言模型可注入（鸭子类型，编排层不依赖 openai）。
    """

    def __init__(
        self,
        *,
        thread_id: str,
        memory: MemoryClient | None = None,
        session_id: str | None = None,
        user_id: str = "default-user",
        group_id: str = "default-group",
        # ── 新首选接法：传 SessionConfig ──
        config: SessionConfig | None = None,
        # ── 旧展开参数（向后兼容；config=None 时生效）──
        regulation_enabled: bool = False,
        regulation_strategy: str = "suppression",
        mood_enabled: bool = False,
        recall_enabled: bool = False,
        language_enabled: bool = False,
        workspace_enabled: bool = False,
        appraisal_conditioning_enabled: bool = False,
        language_max_iters: int = 3,
        rng_seed: int | None = None,
        sample_sigma_cap: float | None = None,
        affect_readout: str = "sample",
        arousal_baseline: float = 0.0,
        arousal_gain_cap: float | None = None,
        precision_split: bool = False,
        fuse_independence_correct: bool = False,
        ignition_survival_fallback: bool = False,
        va_coupling_pos: float | None = None,
        va_coupling_neg: float | None = None,
        panksepp_distinguish_fear: bool = False,
        ignition_beta: float | None = None,
        mood_precision: float = 0.8,
        text_affect_precision: float = 0.3,
        hierarchical_layers: int = 1,
        hierarchical_coupling: float = 0.0,
        # P3 1-B：HPA 皮质醇慢回路旋钮（默认全关=零回归）
        cortisol_enabled: bool = False,
        cortisol_arousal_gate: bool = False,
        cortisol_attitude_gate: bool = False,
        cortisol_arousal_alpha: float = 0.0,
        cortisol_attitude_alpha: float = 0.0,
        cortisol_tau: float | None = None,
        cortisol_impulse: float | None = None,
        cortisol_theta_goal: float | None = None,
        cortisol_theta_intensity: float | None = None,
        # P3 1-C：ToM 共情旋钮（默认 0/0/0/0.3 = 零回归）
        contagion_alpha: float = 0.0,
        care_bias_alpha: float = 0.0,
        vicarious_alpha: float = 0.0,
        vicarious_threshold: float = 0.3,
        # coping_potential 独立标量流（议会 2026-07-13；默认关=零回归）
        coping_potential_enabled: bool = False,
        # text_coping 接线旋钮（议会 2026-07-16 B3；默认关=零回归）
        text_coping_enabled: bool = False,
        text_coping_precision: float = 0.08,
        # fear_domain_enabled：WARN-3 fear 专属门（B1 BLOCK 前置·议会 2026-07-21；默认关=零回归）
        fear_domain_enabled: bool = False,
        # facs_extended：AU 扩展集合门控（设计门 PASS·路径 b；默认关=零回归）
        facs_extended: bool = False,
        # canonical_physiology：physiology 占位口径门控（议会 2026-07-23；默认关=零回归）
        canonical_physiology: bool = False,
        # voluntary_coping_leak：双通路差异化（议会 C1 设计门 2026-07-14；默认 1.0=零回归）
        voluntary_coping_leak: float = 1.0,
        # 外部多模态先验流注入口（议会 2026-07-15 M3/M6；默认=零回归）
        external_prior_precision_cap: float = 0.8,
        max_external_streams: int = 5,
        expression_decoder: ChannelDecoder | None = None,
        language_model: LanguageModel | None = None,
    ) -> None:
        client = (
            memory
            if memory is not None
            else MemoryClient(build_graph_store(), semantic=build_semantic_store())
        )
        self.thread_id = thread_id
        self.session_id = session_id if session_id is not None else thread_id
        self.user_id = user_id
        self.group_id = group_id
        self.checkpointer = build_checkpointer(ALLOWED_CHECKPOINT_TYPES)
        self.graph = build_graph(
            checkpointer=self.checkpointer,
            memory=client,
            expression_decoder=expression_decoder,
            language_model=language_model,
        )
        # D：SessionConfig 收口。config 优先；无 config 时从旧展开参数构造（零回归）。
        if config is not None:
            self.config = config
        else:
            self.config = SessionConfig(
                regulation_enabled=regulation_enabled,
                regulation_strategy=regulation_strategy,
                mood_enabled=mood_enabled,
                recall_enabled=recall_enabled,
                language_enabled=language_enabled,
                workspace_enabled=workspace_enabled,
                appraisal_conditioning_enabled=appraisal_conditioning_enabled,
                language_max_iters=language_max_iters,
                rng_seed=rng_seed,
                sample_sigma_cap=sample_sigma_cap,
                affect_readout=affect_readout,
                arousal_baseline=arousal_baseline,
                arousal_gain_cap=arousal_gain_cap,
                precision_split=precision_split,
                fuse_independence_correct=fuse_independence_correct,
                ignition_survival_fallback=ignition_survival_fallback,
                va_coupling_pos=va_coupling_pos,
                va_coupling_neg=va_coupling_neg,
                panksepp_distinguish_fear=panksepp_distinguish_fear,
                ignition_beta=ignition_beta,
                mood_precision=mood_precision,
                text_affect_precision=text_affect_precision,
                hierarchical_layers=hierarchical_layers,
                hierarchical_coupling=hierarchical_coupling,
                # P3 1-B：HPA 皮质醇慢回路旋钮
                cortisol_enabled=cortisol_enabled,
                cortisol_arousal_gate=cortisol_arousal_gate,
                cortisol_attitude_gate=cortisol_attitude_gate,
                cortisol_arousal_alpha=cortisol_arousal_alpha,
                cortisol_attitude_alpha=cortisol_attitude_alpha,
                cortisol_tau=cortisol_tau,
                cortisol_impulse=cortisol_impulse,
                cortisol_theta_goal=cortisol_theta_goal,
                cortisol_theta_intensity=cortisol_theta_intensity,
                # P3 1-C：ToM 共情旋钮
                contagion_alpha=contagion_alpha,
                care_bias_alpha=care_bias_alpha,
                vicarious_alpha=vicarious_alpha,
                vicarious_threshold=vicarious_threshold,
                # coping_potential 独立标量流（议会 2026-07-13；默认关=零回归）
                coping_potential_enabled=coping_potential_enabled,
                # text_coping 接线旋钮（议会 2026-07-16 B3；默认关=零回归）
                text_coping_enabled=text_coping_enabled,
                text_coping_precision=text_coping_precision,
                # fear_domain_enabled：WARN-3 fear 专属门（B1 BLOCK 前置·议会 2026-07-21；默认关=零回归）  # noqa: E501
                fear_domain_enabled=fear_domain_enabled,
                # facs_extended：AU 扩展集合门控（默认关=零回归）
                facs_extended=facs_extended,
                # canonical_physiology：physiology 占位口径门控（议会 2026-07-23；默认关=零回归）
                canonical_physiology=canonical_physiology,
                # voluntary_coping_leak：双通路差异化（C1 设计门 2026-07-14；默认 1.0=零回归）
                voluntary_coping_leak=voluntary_coping_leak,
                # 外部多模态先验流注入口（议会 2026-07-15 M3/M6；默认=零回归）
                external_prior_precision_cap=external_prior_precision_cap,
                max_external_streams=max_external_streams,
            )

    @property
    def flags(self) -> dict[str, object]:
        """向后兼容属性：代理到 self.config.to_state_flags()。

        旧代码访问 session.flags["sample_sigma_cap"] 等继续工作；
        新代码直接用 session.config.*。
        """
        return self.config.to_state_flags()

    async def step(
        self,
        stim: Stimulus,
        state_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """跑一轮：喂一个 stimulus，返回该轮的可观测轨迹条目（含 e*、mood、生成语言）。

        state_overrides：可选的每轮 state 字段覆盖（优先级高于 config.to_state_flags()）。
        用于注入每轮可变标量（如 interlocutor_affect）而不污染会话级固定配置。
        """
        base: dict[str, Any] = {
            "stimulus": stim,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "group_id": self.group_id,
            "task_complete": False,
            **self.config.to_state_flags(),
            # external_priors 每轮显式归零（M4·code-reviewer B1 2026-07-15）：它是 LastValue
            # channel，若不在本轮 ainvoke input 里显式给值，会从 checkpoint 恢复上一轮注入的
            # 非空 list → 跨轮残留（同 text_affect 被 PerceptionAgent 每轮显式归零的教训）。
            # state_overrides 若含 external_priors 会覆盖此空基准（下方 base.update）。
            "external_priors": [],
            # text_coping 每轮显式归零（B3·议会 2026-07-16 约束5）：两者皆是 LastValue channel，
            # 不归零会从 checkpoint 恢复上轮值造成残留（仿 external_priors 归零先例）。
            # state_overrides 若含 text_coping_prior 会覆盖此 None 基准（下方 base.update）。
            "text_coping_prior": None,
            "text_coping_source": False,
            # recalled_episode_ids 每轮显式归零（B 类·记忆巩固·2026-07-22）：
            # LastValue channel，不归零会跨轮残留上一轮召回的 episode id，
            # 导致 Supervisor 对已过期 episode 重复更新 access_count。
            # 仿 external_priors 归零先例。
            "recalled_episode_ids": [],
        }
        if state_overrides is not None:
            base.update(state_overrides)
        result = await self.graph.ainvoke(
            base,
            config={"configurable": {"thread_id": self.thread_id}},
        )
        state = result if isinstance(result, AffectState) else AffectState(**result)
        return _state_to_entry(stim.name, state)

    async def aclose(self) -> None:
        """释放运行态 checkpointer 的底层 aiosqlite 连接（sqlite 后端）；InMemory 无连接 = no-op。

        幂等：连接已关 / 事件循环已关不上抛。供边界层（如 MCP `close_session`）显式释放，避免长驻
        HTTP server 每会话泄漏一条 aiosqlite 连接（否则进程退出报 `Event loop is closed`）。
        记忆层随对象回收、不在此关（与本方法无关）。
        """
        conn = getattr(self.checkpointer, "conn", None)
        if conn is None:
            return
        try:
            await conn.close()
        except Exception as e:
            # best-effort 幂等释放：关连接的任何异常（含 sqlite3.Error/事件循环已关）都吞掉不上抛，
            # 否则破坏 close_session 的幂等 {ok:true} 语义（清理路径故意宽捕获）。
            logger.debug("ConversationSession.aclose 关连接忽略异常：%s", e)
