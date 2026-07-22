"""情感引擎 ⊗ LLM 的交互对话驱动器（IO 无关的对话核心逻辑）。

原本整段写死在**临时入口** `main.py` 的 `_run_chat` 里；抽出到编排层，使「删除 main.py
做主入口迁移」后，对话驱动（两时间尺度情绪 + 评价桥 + 召回回灌 + 落盘）不随入口消亡。

职责划分（关键）：
- **本模块只管「一轮对话怎么算」**：评价→引擎 step→两时间尺度情绪更新→生成回复→落盘，
  以及启动初始化（env 默认 / lm·log·session 构造 / 历史·态度载入）。**不读 input、不 print**。
- **IO（读用户输入 / 打印 / REPL 循环）由入口提供**——新入口换一套 IO 即可复用本驱动器。

两时间尺度（affective chronometry，ALMA/WASABI 文献）：
- 快变 `emotion`：向 `attitude` 基线衰退恢复 + 被 e* 冲击 + 噪声（怒火几轮回落，不长期累积）。
- 慢变 `attitude`：按 e* 缓慢累积（evaluative conditioning），多轮成形，是 emotion 衰退的基线。
  **只持久化 attitude**（情绪短时、重启归基线）。
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass
from datetime import UTC, datetime

from src.agents.affect_math import (
    ATTITUDE_RATE,
    attitude_step,
    clamp,
    emotion_decay_step,
    habituation_factor,
)
from src.agents.emotion_lexicon import (
    affect_label,
    appraise_standard_compliance,
    infer_domain,
)
from src.agents.emotion_lexicon import appraise_text as lexicon_appraise
from src.agents.language import ConversationModel
from src.agents.models.composite import CompositeChannelDecoder
from src.agents.persona import Persona, load_persona
from src.memory.client import MemoryClient
from src.memory.types import Fact, Scope
from src.orchestration.memory_recall import normalized_importance
from src.orchestration.runner import ConversationSession
from src.orchestration.state import Stimulus
from src.storage.conversation_log import ConversationLog
from src.storage.graph_store import build_graph_store, build_semantic_store

logger = logging.getLogger(__name__)
# 「带对话内容」的专用 logger：每轮 step 末尾发一条可读块（含 user/reply 原文 + 引擎 trace），
# 由 src.observability.setup_conversation_log 路由到独立 logs/conversation-*.log（入口接线）。
# 未接线时（如测试/其它入口）默认 propagate → 落主日志或被吞，均不影响对话本身（零回归）。
_conversation_logger = logging.getLogger("zero.conversation")

# 中性人格单例：作 ChatDriver.persona 的默认值（避免在参数默认里调用 Persona()，ruff B008）。
_NEUTRAL_PERSONA = Persona()

# L3 种子记忆的写入显著度：取高于召回注入门（ZERO_RECALL_INJECT_MIN 默认 0.5 → Hill 归一需
# precision≳30）的量级，使「预置共同记忆」首轮即能被召回并升入对话注意力预算（对应 supervisor
# 富 episode 的 affect_precision ~28–72 区间）。纯常量、与 ATTITUDE_RATE 等同属引擎参数。
SEED_MEMORY_PRECISION = 40.0


@dataclass
class ChatTurn:
    """一轮对话的结果 + 供入口展示的 trace 量（纯数据，无 IO）。"""

    reply: str
    appraised: tuple[float, float]  # 这句话评出的 (v, a)
    attitude_prior: float  # 进入本轮时的 attitude[0]（喂 occ_prior 的先验快照）
    emotion: tuple[float, float]  # 本轮快变情绪（表达取它）
    emotion_label: str
    attitude: tuple[float, float]  # 本轮后对此人的慢变态度


def _u_shape_history(history: list[dict[str, str]], k: int, n: int) -> list[dict[str, str]]:
    """U 形工作记忆窗（D2）：保留首 k 条（首因 primacy）+ 尾 (n-k) 条（近因 recency）。

    人类系列位置效应是 U 形——首尾皆强（Murdock 1962）；单纯尾窗 history[-n:] 漏掉首因
    （「第一次见面说的话」）。k=0 或 k>=n 或 len<=n 时安全退化为 history[-n:]（逐字节等价
    原行为，K=0 即零回归开关）。纯切片、无副作用、可独立单测。
    """
    if k <= 0 or k >= n or len(history) <= n:
        return history[-n:]
    return history[:k] + history[-(n - k) :]


def _inject_recalled_as_system(
    window: list[dict[str, str]],
    recalled_facts: list[Fact],
    inject_min: float,
    importance_scale: float = 30.0,
) -> list[dict[str, str]]:
    """把高 importance 召回 episode 以 system 条目插入 window 头部，进 LLM 注意力预算竞争（D1）。

    现状召回仅作旁路字符串回灌、不与近期 history 在 attention 预算里竞争；本函数让
    importance(=写入 precision) ≥ inject_min 的 episode 作为 {"role":"system",...} 与 history
    同窗竞争（对应皮层 reinstatement）。recalled_facts 已按三维 score 降序（_rank_episodes）；
    空列表原样返回 window（BLOCK-2 fallback，零回归）。纯数据装配、无 LLM（守 BLOCK-1）。
    importance_scale 由调用方（ChatDriver）从构造期参数传入，与 MemoryRecallAgent 保持一致。
    """
    if not recalled_facts:
        return window
    system_entries: list[dict[str, str]] = [
        {"role": "system", "content": f"（记忆片段）{f.content}"}
        for f in recalled_facts
        if normalized_importance(f.content, importance_scale) >= inject_min
    ]
    return system_entries + window


def _relationship_hint(exposure: int) -> str:
    """Q5-B：按曝光轮次映射关系距离标签（供 converse 软约束），env 门控默认关。

    `ZERO_RELATIONSHIP_STAGE_HINT` 未设/为假 → 返回 ""（converse 不注入=逐字零回归）。开启时按
    曝光三档（阈值对齐议会 N_up 陌生→初识 3-5 / 初识→朋友 10-15）给字符串锚。此为"止血"软提示，
    非真关系状态机（Q5-C1 立项）——只给 LLM 分寸参考，不进 affect 热路径、不反馈引擎数值。纯函数。
    """
    if os.getenv("ZERO_RELATIONSHIP_STAGE_HINT", "").lower() not in ("1", "true", "yes", "on"):
        return ""
    if exposure < 5:
        return "初次接触、刚认识不久，还很陌生"
    if exposure < 15:
        return "认识了一阵、但还称不上熟"
    return "已经比较熟络"


def _build_expression_decoder(facs_extended: bool) -> CompositeChannelDecoder | None:
    """env 门控构造真表情解码器：`ZERO_FACS_MODEL_PATH` 未设/空 → None（占位路径，零回归）。

    设了权重路径才延迟 import torch 侧 `load_facs_decoder`（同 OpenAILanguageModel 先例：
    默认路径不引重依赖）。`extended` 与运行时 `state.facs_extended` **同源**——调用方传入
    同一 `ZERO_FACS_EXTENDED` 解析值（守 CompositeChannelDecoder 的键集对齐契约）；权重形状
    与 extended 不配对时 `load_state_dict` fail-fast，不静默回退占位（config-only-via-env）。
    加载失败**不学 perception.py 文本通道的 fail-soft**（那是辅助流，缺模型可降级 OCC 路径）：
    FACS 配了权重路径即声明真模型为表情主路径，静默降级占位会掩盖配置错误，故一律 fail-fast；
    文件缺失/不可读时翻译成指向本 env 的 RuntimeError（不裸抛 torch 堆栈）。
    系数 k_arousal/k_coping/residual_alpha：⚖ 方向议会定、幅度工程可动——构造期一次读 env，
    默认=构造函数默认（1.5/1.2/1.0）=零回归；residual_alpha 越界由 CompositeChannelDecoder
    构造抛 ValueError（fail-fast）。
    """
    model_path = os.getenv("ZERO_FACS_MODEL_PATH", "")
    if not model_path:
        return None
    from src.agents.models.facs_decoder import load_facs_decoder  # 延迟：设了权重才需 torch

    try:
        facs_model = load_facs_decoder(model_path, extended=facs_extended)
    except OSError as e:  # 文件缺失/不可读（FileNotFoundError ⊂ OSError）；形状不配对的
        # RuntimeError 原样穿透（本身已含 size mismatch 详情）。
        raise RuntimeError(
            f"ZERO_FACS_MODEL_PATH={model_path!r} 指向的权重文件不可读，请检查配置"
        ) from e
    return CompositeChannelDecoder(
        facs_model=facs_model,
        facs_extended=facs_extended,
        k_arousal=float(os.getenv("ZERO_FACS_K_AROUSAL", "1.5")),
        k_coping=float(os.getenv("ZERO_FACS_K_COPING", "1.2")),
        residual_alpha=float(os.getenv("ZERO_FACS_RESIDUAL_ALPHA", "1.0")),
    )


class ChatDriver:
    """对话驱动：持有跨轮状态（history / emotion / attitude），每调一次 `step` 推进一轮。

    通过依赖注入（lm / log / session）构造，便于测试与换入口；用 `build_chat_driver()`
    工厂从 env 装配生产实例。

    旋钮参数（均在构造期一次解析，不在 step() 热路径读 env）：
    - intensity_floor: ZERO_INTENSITY_FLOOR 默认 0.2
    - hab_tau: ZERO_HABITUATION_TAU 默认 0.0（<=0 不衰减）
    - reversion_a: ZERO_ATTITUDE_REVERSION_A 默认 None（与 reversion 同维）
    - setpoint_a: ZERO_ATTITUDE_SETPOINT_A 默认 None（取 persona 气质 arousal 基线）
    - decay_k: ZERO_ATTITUDE_RATE_DECAY_K 默认 0.0
    - fam_tau: ZERO_FAMILIARITY_TAU 默认 20.0
    - baseline_w: ZERO_EMOTION_BASELINE_ATTITUDE_W 默认 0.6
    - noise_std: ZERO_EMOTION_NOISE_STD 默认 0.05
    - primacy_k: ZERO_HISTORY_PRIMACY_K 默认 5
    - window_n: ZERO_HISTORY_WINDOW 默认 40
    - inject_min: ZERO_RECALL_INJECT_MIN 默认 0.5
    - sample_sigma_cap: ZERO_SAMPLE_SIGMA_MAX 默认 None（未设=用引擎常量 MAX_SAMPLE_SIGMA）
    - standard_compliance_enabled: ZERO_STANDARD_COMPLIANCE 默认 False（门控关=零回归）
    - text_domain_enabled: ZERO_TEXT_DOMAIN_ENABLED 默认 False（B opt-in·默认关=零回归）
    - attitude_arousal_weight: ZERO_ATTITUDE_AROUSAL_WEIGHT 默认 0.0（零回归）
    - sensitization_gain: ZERO_HABITUATION_SENSITIZATION_GAIN 默认 0.0（零回归）
    - sensitization_threshold: ZERO_SENSITIZATION_THRESHOLD 默认 0.5（零回归）
    - cortisol_attitude_gate: ZERO_CORTISOL_ATTITUDE_GATE 默认 False（零回归）
    - cortisol_attitude_alpha: ZERO_CORTISOL_ATTITUDE_ALPHA 默认 0.0（零回归）
    - contagion_alpha: ZERO_CONTAGION_ALPHA 默认 0.0（零回归）
    - care_bias_alpha: ZERO_CARE_BIAS_ALPHA 默认 0.0（零回归）
    - vicarious_alpha: ZERO_VICARIOUS_ALPHA 默认 0.0（零回归）
    - vicarious_threshold: ZERO_VICARIOUS_THRESHOLD 默认 0.3
    - consolidation_enabled: ZERO_CONSOLIDATION_ENABLED 默认 False（B 类·零回归）
    - actr_enabled: ZERO_ACTR_ENABLED 默认 False（B 类·零回归）
    - d_session: ZERO_CONSOLIDATION_D_SESSION 默认 0.8
    - d_user: ZERO_CONSOLIDATION_D_USER 默认 0.3
    - consolidation_count_min: ZERO_CONSOLIDATION_COUNT_MIN 默认 3
    - consolidation_salience_threshold: ZERO_CONSOLIDATION_SALIENCE_THRESHOLD 默认 0.25
    - actr_b_scale: ZERO_ACTR_B_SCALE 默认 3.0
    - consolidation_timeout: ZERO_CONSOLIDATION_TIMEOUT 默认 30.0
    """

    def __init__(
        self,
        *,
        thread: str,
        lm: ConversationModel | None,
        log: ConversationLog,
        session: ConversationSession,
        history: list[dict[str, str]],
        attitude: tuple[float, float],
        mode: str,
        persona: Persona = _NEUTRAL_PERSONA,
        memory: MemoryClient | None = None,
        seed_key: str = "",
        first_contact: bool = False,
        rng_seed: int | None = None,
        # 旋钮参数：构造期一次解析；工厂从 env 读，测试直接传值
        intensity_floor: float = 0.2,
        hab_tau: float = 0.0,
        reversion_a: float | None = None,
        setpoint_a: float | None = None,
        decay_k: float = 0.0,
        fam_tau: float = 20.0,
        baseline_w: float = 0.6,
        noise_std: float = 0.05,
        primacy_k: int = 5,
        window_n: int = 40,
        inject_min: float = 0.5,
        importance_scale: float = 30.0,
        standard_compliance_enabled: bool = False,
        text_domain_enabled: bool = False,
        # B（B7a·两时间尺度旋钮）：默认值均为旧行为（零回归）
        attitude_arousal_weight: float = 0.0,
        sensitization_gain: float = 0.0,
        sensitization_threshold: float = 0.5,
        # P3 1-B：cortisol→ATTITUDE_RATE 消费旋钮（默认关=零回归）
        # cortisol_attitude_gate=True 时 rate_eff=ATTITUDE_RATE*(1+alpha*cortisol)（f_max≤2）
        cortisol_attitude_gate: bool = False,
        cortisol_attitude_alpha: float = 0.0,
        # P3 1-C：ToM 共情系数（默认全 0/0.3=零回归；session.config 同步持有）
        # 这些系数供 step() 判断门控状态：任一 alpha>0 → 注入 interlocutor_affect；关则不注入。
        contagion_alpha: float = 0.0,
        care_bias_alpha: float = 0.0,
        vicarious_alpha: float = 0.0,
        vicarious_threshold: float = 0.3,
        # B 类·记忆巩固旋钮（默认全关=零回归；aclose 时触发·不在 step 热路径）
        # ZERO_CONSOLIDATION_ENABLED=0 → aclose no-op；开启后会话结束时跑 Ebbinghaus+睡眠巩固。
        consolidation_enabled: bool = False,
        actr_enabled: bool = False,  # ACT-R recency 替换门（ZERO_ACTR_ENABLED）
        d_session: float = 0.8,  # SESSION 幂律衰减指数（ZERO_CONSOLIDATION_D_SESSION）
        d_user: float = 0.3,  # USER 幂律衰减指数（ZERO_CONSOLIDATION_D_USER）
        consolidation_count_min: int = 3,  # 睡眠巩固最低强化次数门（ZERO_CONSOLIDATION_COUNT_MIN）
        actr_b_scale: float = 3.0,  # Petrov B sigmoid scale（ZERO_ACTR_B_SCALE）
        consolidation_timeout: float = 30.0,  # aclose 巩固超时秒（ZERO_CONSOLIDATION_TIMEOUT）
        # salience 升迁门（Hill 归一后·默认 0.25·议会 2026-07-22）
        consolidation_salience_threshold: float = 0.25,
    ) -> None:
        self.thread = thread
        self.lm = lm
        self.log = log
        self.session = session
        self.history = history
        self.attitude = attitude
        self.emotion = attitude  # 情绪短时：启动即回到「对此人的态度」基线，不带旧情绪
        self.mode = mode
        # L2 气质底色 + L3 预置关系：默认中性 Persona() / 无记忆句柄 → 逐字现有行为（零回归）。
        self.persona = persona
        self.memory = memory
        self.seed_key = seed_key  # 种子记忆写入的 user 作用域 key（= user_id = thread）
        self.first_contact = first_contact  # 首次接触此人（空 transcript）才播种关系
        self.seeded = False  # L3 种子记忆只在首轮写一次的守卫
        self.exposure = (
            0  # P2（Q6）：对此对话对象的累计曝光轮次，喂 habituation_factor 衰减 arousal
        )
        # 情绪噪声 RNG（P5 可复现，议会建议）：给了 seed → 专属 Random（跨轮推进、跨进程可复现）；
        # None → 用模块级 random（逐字旧行为、零回归，且保留既有 monkeypatch 测试可控）。
        self.rng = random.Random(rng_seed) if rng_seed is not None else None
        # 构造期固化旋钮（env 只在工厂读一次）
        self.intensity_floor = intensity_floor
        self.hab_tau = hab_tau
        self.reversion_a = reversion_a
        self.setpoint_a = setpoint_a
        self.decay_k = decay_k
        self.fam_tau = fam_tau
        self.baseline_w = baseline_w
        self.noise_std = noise_std
        self.primacy_k = primacy_k
        self.window_n = window_n
        self.inject_min = inject_min
        self.importance_scale = importance_scale
        # B6：OCC 分支 B 通电开关。默认 False → step 构造 Stimulus 时不传 standard_compliance
        # （保持默认 0.0，逐字零回归）；True 时调确定性评价桥 appraise_standard_compliance 填充。
        self.standard_compliance_enabled = standard_compliance_enabled
        # B opt-in：开→step 调 infer_domain 注入 domain·关→domain=None 零回归
        self.text_domain_enabled = text_domain_enabled
        # B（B7a·两时间尺度旋钮）：构造期固化，step 热路径读 self.* 不重读 env。
        # A-P2-E：attitude_step 唤醒加权累积率（ZERO_ATTITUDE_AROUSAL_WEIGHT 默认 0.0，零回归）；
        # 高唤醒 stimulus 使态度累积加速（McGaugh 2004 唤醒调制记忆巩固）。
        self.attitude_arousal_weight = attitude_arousal_weight
        # A-P3-D：habituation_factor 敏化项——强刺激（intensity > sens_threshold）时
        # 敏化主导 η 可 >1（Groves & Thompson 1970 双过程）。默认 gain=0.0 → 纯习惯化（零回归）。
        # intensity 取本轮 e 的 |arousal|（唤醒作为刺激强度指标，语义：强情绪刺激触发敏化）。
        self.sens_gain = sensitization_gain
        self.sens_threshold = sensitization_threshold
        # P3 1-B：cortisol→ATTITUDE_RATE 放大（allostatic load 阈值重塑）
        # cortisol_attitude_gate=False（默认）→ rate_eff 不变，零回归。
        # cortisol_state 从每轮 step_out 读（图内 Checkpointer 维护，不入图谱）。
        self.cortisol_attitude_gate = cortisol_attitude_gate
        self.cortisol_attitude_alpha = cortisol_attitude_alpha
        # P3 1-C：ToM 共情系数（构造期固化；step 热路径读 self.* 不重读 env）。
        # 门控判断：任一 alpha>0 → step 注入 interlocutor_affect；全 0 → 不注入（零回归）。
        self.contagion_alpha = contagion_alpha
        self.care_bias_alpha = care_bias_alpha
        self.vicarious_alpha = vicarious_alpha
        self.vicarious_threshold = vicarious_threshold
        # B 类·记忆巩固旋钮（构造期固化；aclose 时消费；不在 step 热路径）
        self.consolidation_enabled = consolidation_enabled
        self.actr_enabled = actr_enabled
        self.d_session = d_session
        self.d_user = d_user
        self.consolidation_count_min = consolidation_count_min
        self.actr_b_scale = actr_b_scale
        self.consolidation_timeout = consolidation_timeout
        self.consolidation_salience_threshold = consolidation_salience_threshold

    async def step(self, user_text: str) -> ChatTurn:
        """推进一轮：评价→引擎→两时间尺度情绪→生成回复→落盘，返回本轮结果。"""
        # L3：首次接触此人时，先把预置的共同记忆灌进语义库（只一次），使本轮起即可召回。
        await self._maybe_seed_memories()
        # 评价桥：读这句话情绪（真 LLM 或词典回退）
        if self.lm is not None:
            v, a = await self.lm.appraise_text(user_text)
        else:
            v, a = lexicon_appraise(user_text)
        # 快照：进入本轮时的态度（attitude_step 在 step 之后才更新）
        attitude_prior = self.attitude[0]
        # P1-a（议会 Q1·失真必改）：intensity 下限旋钮。旧硬编码 0.2 是与内容无关的 arousal 直流
        # 底噪源（占 occ_prior arousal 的 0.4·|intensity|=0.08 常量），中性对话也被抬唤醒。默认 0.2
        # =逐字旧行为（零回归）；纪要推荐 0 走 .env.example，下限越低中性越能回落静息。
        # B6：standard_compliance_enabled 开启时调确定性评价桥填充 Stimulus；关闭时不传
        # （Stimulus 默认 0.0，逐字零回归——门控关时 Stimulus 构造与改动前逐字相同）。
        stim_kwargs: dict = dict(
            name=user_text[:40],
            text=user_text,
            goal_congruence=v,
            attitude_appeal=self.attitude[0],
            intensity=min(1.0, max(self.intensity_floor, abs(a))),
        )
        if self.standard_compliance_enabled:
            stim_kwargs["standard_compliance"] = appraise_standard_compliance(user_text)
        # B opt-in（ZERO_TEXT_DOMAIN_ENABLED·议会 2026-07-21）：
        # 开→infer_domain 词典桥每轮二值判 confrontational/None 注入 domain；
        # 关→不传→Stimulus.domain 默认 None 旁路零回归。
        # 不传 control_appraisal（ctrl=None·_check_domain_ctrl_sign no-op 合法）。
        if self.text_domain_enabled:
            stim_kwargs["domain"] = infer_domain(user_text)
        stim = Stimulus(**stim_kwargs)
        # P3 1-C：任一共情 alpha>0（门开）把 interlocutor_va 注入每轮 state_overrides；
        # 全关（默认）→ 不传 state_overrides → session.step(stim) 调用签名与改前逐字一致（零回归）。
        # 走 state_overrides 不污染会话级固定的 session.config。
        tom_gate_open = (
            self.contagion_alpha > 0.0 or self.care_bias_alpha > 0.0 or self.vicarious_alpha > 0.0
        )
        if tom_gate_open:
            # 图外独立估计对方情绪 VAD（确定性词典法）。⚠ 语义边界：lm.appraise_text 是
            # 「对我目标一致性」(自身 OCC)，此处是「对方情绪 VAD」——语义不同、字段独立、
            # 绝不复用同一 (v,a)。仅门开时计算；估计在图外、不进 StateGraph 确定性节点。
            interlocutor_va = lexicon_appraise(user_text)
            step_out = await self.session.step(
                stim, state_overrides={"interlocutor_affect": interlocutor_va}
            )
        else:
            step_out = await self.session.step(stim)
        e = step_out["valence_arousal"] or (0.0, 0.0)
        # B（A-P3-D）：双过程习惯化+敏化。intensity 取本轮 e 的 |arousal|（唤醒=刺激强度指标）。
        # 默认 sens_gain=0.0 → 退化为纯习惯化 exp(-n/τ)（零回归）；sens_gain>0 时强刺激敏化主导。
        # P2（议会 Q6·失真必改）：τ via ZERO_HABITUATION_TAU；未设/<=0 → η=1 不衰减（零回归）。
        # self.exposure = 本轮前累计曝光数 n（于 step 末尾统一自增）：habituation(P2)/familiarity
        # (Q5-A)/relationship_hint(Q5-B) 三处同读 n，首轮 n=0=不习惯化+完全陌生，语义对齐（W1）。
        e_arousal_intensity = abs(e[1])
        hab_eta = habituation_factor(
            self.exposure,
            self.hab_tau,
            intensity=e_arousal_intensity,
            sensitization_gain=self.sens_gain,
            sensitization_threshold=self.sens_threshold,
        )
        e = (e[0], e[1] * hab_eta)
        # 语义召回上下文（recall_enabled 时图内 memory_recall 节点填充；无语义后端则空）
        recalled: list[str] = step_out.get("recalled_context") or []
        recalled_str = " | ".join(r[:120] for r in recalled) if recalled else ""
        # D1：召回的原始 Fact（已三维重排），高 importance 者升入 history 注意力预算竞争
        recalled_facts: list[Fact] = step_out.get("recalled_facts") or []
        # 慢：态度按 e* 缓慢累积 + 向个体 setpoint 弱回归（attitude_step 内含 reversion 防棘轮）。
        # setpoint = persona 气质基线（L2，默认中性 → 与改前逐字一致）。
        # P1-b（议会 Q2b）：arousal 维独立回归率 reversion_a（未设 → None → 同 reversion，零回归）
        # + 独立 setpoint_a（未设 → 取 persona 气质 arousal 基线）。设 reversion_a≫reversion（推荐
        # 0.3–0.5）令 attitude 不累积 arousal 直流偏置（心理席：态度是 valence 维评价）。
        setpoint = (
            self.persona.setpoint
            if self.setpoint_a is None
            else (self.persona.setpoint[0], self.setpoint_a)
        )
        # Q5-A（议会二轮·止血，非真多稳态）：熟悉度门控 rate 衰减——越熟态度形成越慢（近似"陌生态
        # 更稳"）。familiarity=1−exp(−exposure/τ_f)；rate_eff=rate·(1−k·familiarity)。K=0（默认）→
        # rate_eff=rate 逐字零回归。单不动点仅减缓漂移，非真多稳态（真关系态见 C1 立项）。
        familiarity = 1.0 - habituation_factor(self.exposure, self.fam_tau)
        rate_eff = ATTITUDE_RATE * (1.0 - self.decay_k * familiarity)
        # P3 1-B：cortisol→ATTITUDE_RATE 放大（allostatic load 阈值重塑，McEwen & Gianaros 2011）。
        # cortisol_attitude_gate=True 时 rate_eff *= (1 + alpha·cortisol)（f_max≤2，alpha≤1）。
        # 门控关（默认）→ rate_eff 不变，零回归。cortisol_state 由图内 Checkpointer 维护，
        # 不入图谱（CS 红线）；从 step_out 读（_state_to_entry 已暴露 cortisol_state 字段）。
        if self.cortisol_attitude_gate:
            cortisol_state: float = step_out.get("cortisol_state") or 0.0
            rate_eff = rate_eff * (1.0 + self.cortisol_attitude_alpha * cortisol_state)
        # B（A-P2-E）：attitude_arousal_weight 默认 0.0 → rate_eff 不变（零回归）；
        # >0 时高唤醒 stimulus 放大有效累积率（McGaugh 2004 唤醒调制记忆巩固/态度形成）。
        self.attitude = attitude_step(
            self.attitude,
            e,
            rate=rate_eff,
            setpoint=setpoint,
            reversion_a=self.reversion_a,
            arousal_weight=self.attitude_arousal_weight,
        )
        # 快：情绪向「基线」衰退恢复 + 当前 e* 冲击 + 噪声（短时）。
        # 议会 B（必改）：基线不是纯 attitude——它被持续正刺激推高后，纯 attitude 基线会让情绪
        # 永远停在高位（emotion 的家随 attitude 上漂、无中性拉力）。改为 attitude 与个体 setpoint
        # （persona 气质基线，默认中性）的混合 `w·attitude + (1-w)·setpoint`，给情绪回归力
        # （affective homeostasis；生物席必改 #2 / Russell 2003）。w=1 退化为旧纯 attitude 基线。
        # recovery/reactivity 亦取自 persona（L2 气质：默认=引擎常量，神经质↑可调高反应性/慢恢复）。
        baseline = (
            self.baseline_w * self.attitude[0] + (1.0 - self.baseline_w) * self.persona.setpoint[0],
            self.baseline_w * self.attitude[1] + (1.0 - self.baseline_w) * self.persona.setpoint[1],
        )
        self.emotion = emotion_decay_step(
            self.emotion,
            baseline,
            e,
            recovery=self.persona.recovery,
            reactivity=self.persona.reactivity,
        )
        # 情绪噪声标准差（「防抖」旋钮）：默认 0.05（逐字旧行为），调小=更稳，0=关该噪声源。
        # P5：有 seed 用专属 Random 复现；无 seed 走模块级 random（零回归、可 monkeypatch）。
        gauss = self.rng.gauss if self.rng is not None else random.gauss
        self.emotion = (
            clamp(self.emotion[0] + gauss(0.0, self.noise_std), -1.0, 1.0),
            clamp(self.emotion[1] + gauss(0.0, self.noise_std), -1.0, 1.0),
        )
        word = affect_label(*self.emotion)
        self.history.append({"role": "user", "content": user_text})
        if self.lm is not None:
            # D2 U 形窗（首因+近因）→ D1 注入高 importance 召回为 system 条目，进同一注意力预算竞争
            # 议会 A：20 条(≈10 轮)对长对话保持率过低——本现场「下午两点」第 19 轮即被挤出尾窗、
            # 第 22 轮追问时已不在上下文（Murdock 1962 中央位置遗忘）。提到 40(≈20 轮)、primacy 5
            # （覆盖最初契约性陈述）。仍是 env 可调的纯切片，不改 U 形算法。
            window = _u_shape_history(self.history, self.primacy_k, self.window_n)
            window = _inject_recalled_as_system(
                window, recalled_facts, self.inject_min, self.importance_scale
            )
            # Q5-B（议会二轮·止血）：关系距离标签（按曝光轮次三档，对齐议会 N_up 3-5/10-15）注入
            # LLM 软约束别越距离。ZERO_RELATIONSHIP_STAGE_HINT 默认关 → 空串 → converse 零回归。
            relationship_hint = _relationship_hint(self.exposure)
            # push 通路：情绪经用词倾向自然漏进输出；retrieved 回灌召回背景（与 system 条目并存）
            reply = await self.lm.converse(
                window, self.emotion, recalled_str, push=True, relationship_hint=relationship_hint
            )
        else:
            reply = f"（{word}）嗯，我在听，你接着说。"
        self.history.append({"role": "assistant", "content": reply})
        self.log.append(self.thread, "user", user_text)
        self.log.append(self.thread, "assistant", reply)
        self.log.save_feeling(self.thread, self.attitude)  # 只持久化「态度」
        logger.debug(
            "chat step thread=%s appraise=%s e*=%s emotion=%s attitude=%s",
            self.thread,
            (v, a),
            e,
            self.emotion,
            self.attitude,
        )
        # 「带对话内容」的可读日志：本轮 = 一条记录。无条件发往 zero.conversation，开关/落盘/隔离
        # 由入口的 setup_conversation_log 兜底（ZERO_CONVERSATION_LOG=0 时被 NullHandler 吞）。
        _conversation_logger.info(
            "thread=%s 第%d轮\n  你   > %s\n  Zero > %s\n"
            "  ├ 评价(v,a)=(%+.2f,%+.2f)  先验=%+.2f\n"
            "  ├ 情绪=%s (%+.2f,%+.2f)\n"
            "  └ 对你的态度=(%+.2f,%+.2f)",
            self.thread,
            len(self.history) // 2,
            user_text,
            reply,
            v,
            a,
            attitude_prior,
            word,
            self.emotion[0],
            self.emotion[1],
            self.attitude[0],
            self.attitude[1],
        )
        self.exposure += 1  # 本轮末尾统一自增（见上：三处 exposure 用法同读本轮前计数 n，W1）
        return ChatTurn(
            reply=reply,
            appraised=(v, a),
            attitude_prior=attitude_prior,
            emotion=self.emotion,
            emotion_label=word,
            attitude=self.attitude,
        )

    async def aclose(self) -> None:
        """会话结束清理：触发记忆巩固批处理，关闭语义后端连接。

        B 类·记忆巩固（2026-07-22）：
        - consolidation_enabled=False（默认）→ 巩固段 no-op，直接关连接（零回归）。
        - 开启后：asyncio.wait_for 包裹巩固，超时（consolidation_timeout 秒）降级 warning，
          不崩对话退出流程。巩固经 MemoryClient.run_consolidation_batch（守三层单向）。
        - 末尾调 self.memory.aclose() 关闭语义后端（SqliteVectorStore/Graphiti 连接）。
        - 无记忆句柄时整体 no-op。
        延迟 import consolidation 模块（aclose 路径·非热路径；仅门开时实际触发）。
        """
        if self.memory is None:
            return
        if self.consolidation_enabled:
            try:
                await asyncio.wait_for(
                    self.memory.run_consolidation_batch(
                        scope_session="session",
                        scope_user="user",
                        key=self.seed_key or self.thread,
                        consolidation_enabled=self.consolidation_enabled,
                        d_session=self.d_session,
                        d_user=self.d_user,
                        salience_threshold=self.consolidation_salience_threshold,
                        consolidation_count_min=self.consolidation_count_min,
                        actr_b_scale=self.actr_b_scale,
                    ),
                    timeout=self.consolidation_timeout,
                )
            except TimeoutError:
                logger.warning(
                    "aclose consolidation 超时 (%.1fs) thread=%s，降级跳过",
                    self.consolidation_timeout,
                    self.thread,
                )
            except Exception as exc:
                logger.warning(
                    "aclose consolidation 失败 thread=%s: %s，降级跳过",
                    self.thread,
                    exc,
                    exc_info=True,
                )
        await self.memory.aclose()

    async def _maybe_seed_memories(self) -> None:
        """L3：首次接触此人时把 persona 预置的共同记忆幂等写入 user 作用域语义库（只一次）。

        守卫 `seeded` 保证一会话只写一次；`first_contact`（空 transcript）门控避免覆盖已有关系；
        无记忆句柄 / 无种子 → no-op（零回归）。镜像 supervisor 富 episode：embed_text 喂纯文本检索
        语义、content 附 precision/first_contact 元数据供召回三维重排与注入门。写入走
        MemoryClient.write_episode（内含失败隔离 try/except），绝不崩对话。**一次性 init 写入、
        非每条消息——不违 memory-rules#1 节流（不在每步刷图谱）。**
        """
        if self.seeded:
            return
        self.seeded = True
        if not (self.first_contact and self.persona.seed_memories and self.memory is not None):
            return
        now = datetime.now(UTC)
        for text in self.persona.seed_memories:
            content = (
                f"{text} | precision={SEED_MEMORY_PRECISION:.2f} | seed=True | first_contact=True"
            )
            await self.memory.write_episode(
                content, scope=Scope.USER, key=self.seed_key, valid_at=now, embed_text=text
            )
        logger.info(
            "persona 预置关系：thread=%s 播种 %d 条共同记忆",
            self.thread,
            len(self.persona.seed_memories),
        )


def build_chat_driver(thread: str | None = None) -> ChatDriver:
    """从环境装配生产用对话驱动器：读 env 构造 lm/log/session + 载入历史/态度。

    缺 LLM key → lm=None（词典评价 + 模板回退，仍跑两时间尺度情绪）。**纯装配、无全局副作用**：
    后端选择（ZERO_*_BACKEND 默认）、噪声日志压制等「入口/部署策略」由调用方（入口）在此
    之前设好；本工厂只读已就绪的 env，不写 os.environ、不改全局 logger 级别。
    人格经 `load_persona()` 读入（未配置 → 中性 Persona()，逐字现有行为）：L1 人设卡入 lm、
    L2 气质/L3 种子透传给 ChatDriver。记忆后端显式构造一次、同时注入 session 与 ChatDriver，
    使种子记忆落在召回会查的同一 user/key 下。表情通道经 `_build_expression_decoder` 装配：
    `ZERO_FACS_MODEL_PATH` 门控注入真解码器（未设 → None → 占位路径零回归）。
    """
    resolved_thread = thread or os.getenv("ZERO_CHAT_THREAD") or "chat"
    persona = load_persona()  # 人格定义（env/JSON 文件；未配置 → 中性 Persona()，零回归）
    api_key = os.getenv("ZERO_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("ZERO_OPENAI_MODEL")
    lm: ConversationModel | None = None
    mode = "无 key 回退（词典评价 + 模板）"
    if api_key and model_name:
        from src.agents.language_openai import OpenAILanguageModel  # 延迟：仅有 key 才需 openai

        # L1：人设卡注入对话 system prompt（空卡 → 与改前逐字一致）
        lm = OpenAILanguageModel(model=model_name, persona=persona.card)
        mode = f"真 LLM（{model_name}）"

    log = ConversationLog()
    history = log.recent(resolved_thread, 20)  # 重载历史 → 跨重启记忆
    first_contact = not history  # 空 transcript = 首次接触此人（决定是否播种预置关系）
    # L3：首次接触且 persona 给了初始态度 → 以它起步（带着对此人的预置情感）；否则续上持久化态度。
    if first_contact and persona.initial_attitude is not None:
        attitude = persona.initial_attitude
    else:
        attitude = log.load_feeling(resolved_thread)  # 续上对此人的长期态度（持久化的慢变量）
    # 显式构造记忆后端并注入 session：使「种子记忆写入」与「图内召回」共用同一后端（种子落在召回
    # 会查的 user/key 下）。与 runner/ConversationSession 默认装配等价（默认后端由 env 决定）。
    memory = MemoryClient(build_graph_store(), semantic=build_semantic_store())
    # 后验采样 sigma 上限（情绪「防抖」旋钮）：未设 → None → 用引擎常量 MAX_SAMPLE_SIGMA（零回归）。
    sigma_cap_env = os.getenv("ZERO_SAMPLE_SIGMA_MAX")
    sample_sigma_cap = float(sigma_cap_env) if sigma_cap_env else None
    # P5（可复现，议会建议）：单个种子贯穿 --chat 两处随机源——引擎后验采样（session.rng_seed）+
    # chat 层情绪噪声（ChatDriver.rng_seed）。未设 → None → 两处走旧随机（零回归；eval 设此复现）。
    seed_env = os.getenv("ZERO_CHAT_RNG_SEED")
    rng_seed = int(seed_env) if seed_env else None
    # 情绪读出模式（P4 议会 α）：'map'=后验均值 e*=post_mu（消逐轮翻号）；默认 'sample'=旧行为。
    affect_readout = os.getenv("ZERO_AFFECT_READOUT", "sample")
    # arousal 证据基准平移（P1-c 议会 Q7）：默认 0.0=旧整流行为（零回归）；负值启 deactivation
    # （平淡输入把 arousal 拉向静息/负）。经 session→state→AppraisalAgent→occ_prior。
    arousal_baseline = float(os.getenv("ZERO_AROUSAL_BASELINE", "0"))
    # arousal_gain 增益上限（P4-d 议会二轮·廉价 cap）：未设 → None → 不 cap（零回归）；设 [0.3,0.6]
    # 则 arousal_gain 钳到 1+cap，防高唤醒正反馈无界。经 session→state→AffectCoreAgent。
    gain_cap_env = os.getenv("ZERO_AROUSAL_GAIN_CAP")
    arousal_gain_cap = float(gain_cap_env) if gain_cap_env else None
    # A. workspace 三 flag（B4/B5）：默认未设 → False → 零回归（对应 state 字段同名 flag）。
    # 经 session(SessionConfig) → ainvoke state 初值 → AffectCoreAgent 消费。
    precision_split = os.getenv("ZERO_PRECISION_SPLIT", "").lower() in ("1", "true", "yes", "on")
    fuse_independence_correct = os.getenv("ZERO_FUSE_INDEPENDENCE_CORRECT", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    ignition_survival_fallback = os.getenv("ZERO_IGNITION_SURVIVAL_FALLBACK", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # E（WARN-4 / A-P1-D）：Panksepp RAGE/FEAR 次级区分。默认未设 → False → 零回归。
    # True 时 motivational_system 在 (-v,+a) 象限按 arousal 阈值分 fear/rage（阈值待议会 P1-D）。
    # 经 session(SessionConfig) → state → _appraisal_summary 贯通
    # （仅 appraisal_conditioning 开时经该摘要可见）。
    panksepp_distinguish_fear = os.getenv("ZERO_PANKSEPP_DISTINGUISH_FEAR", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # C（A-P2-A）：va_coupling 系数。优先取 persona 字段；未配则读 env；都无 → None（零回归）。
    # 路径：persona/env → SessionConfig → state → AppraisalAgent → occ_prior。
    va_coupling_pos: float | None = persona.va_coupling_pos
    va_coupling_neg: float | None = persona.va_coupling_neg
    if va_coupling_pos is None:
        _pos_env = os.getenv("ZERO_VA_COUPLING_POS")
        va_coupling_pos = float(_pos_env) if _pos_env else None
    if va_coupling_neg is None:
        _neg_env = os.getenv("ZERO_VA_COUPLING_NEG")
        va_coupling_neg = float(_neg_env) if _neg_env else None
    # 旋钮参数：在工厂一次性从 env 读取，透传给 ChatDriver 构造期固化（不在 step 热路径重读）。
    # P1-a：intensity 下限（ZERO_INTENSITY_FLOOR 默认 0.2，逐字旧行为）
    intensity_floor = float(os.getenv("ZERO_INTENSITY_FLOOR", "0.2"))
    # P2：习惯化衰减 τ（ZERO_HABITUATION_TAU 默认 0，<=0 不衰减，零回归）
    hab_tau = float(os.getenv("ZERO_HABITUATION_TAU", "0"))
    # P1-b：arousal 独立回归率（ZERO_ATTITUDE_REVERSION_A 未设 → None → 与 reversion 同维，零回归）
    rev_a_env = os.getenv("ZERO_ATTITUDE_REVERSION_A")
    reversion_a = float(rev_a_env) if rev_a_env else None
    # P1-b：arousal 独立 setpoint（ZERO_ATTITUDE_SETPOINT_A 未设 → None → 取 persona 气质基线）
    setpoint_a_env = os.getenv("ZERO_ATTITUDE_SETPOINT_A")
    setpoint_a = float(setpoint_a_env) if setpoint_a_env else None
    # Q5-A：熟悉度门控 rate 衰减系数（ZERO_ATTITUDE_RATE_DECAY_K 默认 0，零回归）
    decay_k = float(os.getenv("ZERO_ATTITUDE_RATE_DECAY_K", "0"))
    # Q5-A：熟悉度 τ（ZERO_FAMILIARITY_TAU 默认 20）
    fam_tau = float(os.getenv("ZERO_FAMILIARITY_TAU", "20"))
    # 议会 B：情绪基线混合权重（ZERO_EMOTION_BASELINE_ATTITUDE_W 默认 0.6，逐字旧行为）
    baseline_w = float(os.getenv("ZERO_EMOTION_BASELINE_ATTITUDE_W", "0.6"))
    # 情绪噪声标准差（ZERO_EMOTION_NOISE_STD 默认 0.05，逐字旧行为）
    noise_std = float(os.getenv("ZERO_EMOTION_NOISE_STD", "0.05"))
    # D2 U 形历史窗 primacy（ZERO_HISTORY_PRIMACY_K 默认 5）
    primacy_k = int(os.getenv("ZERO_HISTORY_PRIMACY_K", "5"))
    # D2 U 形历史窗大小（ZERO_HISTORY_WINDOW 默认 40）
    window_n = int(os.getenv("ZERO_HISTORY_WINDOW", "40"))
    # D1 召回注入门（ZERO_RECALL_INJECT_MIN 默认 0.5）
    inject_min = float(os.getenv("ZERO_RECALL_INJECT_MIN", "0.5"))
    # D8 Hill 归一 scale（ZERO_RECALL_IMPORTANCE_SCALE 默认 30，与 MemoryRecallAgent 保持一致）
    importance_scale = float(os.getenv("ZERO_RECALL_IMPORTANCE_SCALE", "30"))
    # B6：OCC 分支 B 通电开关（ZERO_STANDARD_COMPLIANCE；默认关=零回归）
    standard_compliance_enabled = os.getenv("ZERO_STANDARD_COMPLIANCE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # B opt-in：live-chat 域注入开关（ZERO_TEXT_DOMAIN_ENABLED·议会 2026-07-21·默认关=零回归）。
    # 关→domain=None 旁路；翻 ZERO_TEXT_COPING_ENABLED 不会全域生效（域条件化守住）。
    text_domain_enabled = os.getenv("ZERO_TEXT_DOMAIN_ENABLED", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # B（B7a·两时间尺度旋钮）：默认未设 → 旧默认值 → 零回归。
    # A-P2-E：attitude_step 唤醒加权累积率（ZERO_ATTITUDE_AROUSAL_WEIGHT 默认 0.0，零回归）
    attitude_arousal_weight = float(os.getenv("ZERO_ATTITUDE_AROUSAL_WEIGHT", "0"))
    # A-P3-D：习惯化+敏化双过程旋钮（默认均为零回归值）
    sensitization_gain = float(os.getenv("ZERO_HABITUATION_SENSITIZATION_GAIN", "0"))
    sensitization_threshold = float(os.getenv("ZERO_SENSITIZATION_THRESHOLD", "0.5"))
    # 议会 2026-07-02 Item 2：ignite 软门控陡度（ZERO_IGNITION_BETA）
    # 未设 → None → 硬 step 零回归；β∈[20,50]，典型值 20。
    # 经 session → state.ignition_beta → ignite(soft_beta)。
    _ignition_beta_env = os.getenv("ZERO_IGNITION_BETA")
    ignition_beta: float | None = float(_ignition_beta_env) if _ignition_beta_env else None
    # T6-a：mood 流精度旋钮（ZERO_MOOD_PRECISION；默认 0.8 = MOOD_PRECISION 常量，逐字零回归）
    # 未设 → 0.8 → 行为与改前完全一致；调小降低 mood 流投票权。
    mood_precision = float(os.getenv("ZERO_MOOD_PRECISION", "0.8"))
    # T6-b：文本语义流精度旋钮（ZERO_TEXT_AFFECT_PRECISION；默认 0.3=TEXT_AFFECT_PRECISION，
    # 逐字零回归）。未设 → 0.3；固定低精度（Friston 2009 初始固定精度合理起点）。
    text_affect_precision = float(os.getenv("ZERO_TEXT_AFFECT_PRECISION", "0.3"))
    # P3 HPC 旋钮（ZERO_HPC_LAYERS / ZERO_HPC_COUPLING；默认 1/0.0=平层零回归）。
    # layers=1 或 coupling=0.0 → 退化 fuse_terms(all)，逐字等价现行为；
    # layers=2, coupling∈(0,1] → 启用 2 层层级预测编码（design.md 六-bis 定稿）；
    # coupling>1 → affect_math.hierarchical_fuse raise ValueError（硬拒不 clamp）。
    hpc_layers = int(os.getenv("ZERO_HPC_LAYERS", "1"))
    hpc_coupling = float(os.getenv("ZERO_HPC_COUPLING", "0"))
    # P3 1-B：HPA 皮质醇慢回路旋钮（ZERO_CORTISOL_*；默认全关=零回归）。
    # cortisol 只进 Checkpointer（经 session→state），绝不写入图谱（CS 红线）。
    # 触发解耦：cortisol_trigger 只读 appraisal 输入 goal_congruence/intensity（防 runaway）。
    # 安全区间：τ≈5400s / cap=1 / attitude_alpha≤1(f_max≤2) / arousal_alpha 0.1-0.3。
    cortisol_enabled = os.getenv("ZERO_CORTISOL_ENABLED", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    cortisol_arousal_gate = os.getenv("ZERO_CORTISOL_AROUSAL_GATE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    cortisol_attitude_gate = os.getenv("ZERO_CORTISOL_ATTITUDE_GATE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    _cortisol_tau_env = os.getenv("ZERO_CORTISOL_TAU")
    cortisol_tau = float(_cortisol_tau_env) if _cortisol_tau_env else None
    _cortisol_impulse_env = os.getenv("ZERO_CORTISOL_IMPULSE")
    cortisol_impulse = float(_cortisol_impulse_env) if _cortisol_impulse_env else None
    cortisol_arousal_alpha = float(os.getenv("ZERO_CORTISOL_AROUSAL_ALPHA", "0"))
    cortisol_attitude_alpha = float(os.getenv("ZERO_CORTISOL_ATTITUDE_ALPHA", "0"))
    _theta_goal_env = os.getenv("ZERO_CORTISOL_THETA_GOAL")
    cortisol_theta_goal = float(_theta_goal_env) if _theta_goal_env else None
    _theta_intensity_env = os.getenv("ZERO_CORTISOL_THETA_INTENSITY")
    cortisol_theta_intensity = float(_theta_intensity_env) if _theta_intensity_env else None
    # P3 1-C：ToM 共情旋钮（ZERO_CONTAGION_ALPHA / ZERO_CARE_BIAS_ALPHA / ZERO_VICARIOUS_ALPHA /
    # ZERO_VICARIOUS_THRESHOLD；默认全 0/0.3=零回归）。
    # 安全区间：contagion≤0.3·推荐[0.05,0.25] / care≤0.4 / vicarious≤0.2 / L1≤0.6。
    # 估计图外：interlocutor_affect 在 step() 由 lexicon_appraise 独立产出，不进 StateGraph。
    contagion_alpha = float(os.getenv("ZERO_CONTAGION_ALPHA", "0"))
    care_bias_alpha = float(os.getenv("ZERO_CARE_BIAS_ALPHA", "0"))
    vicarious_alpha = float(os.getenv("ZERO_VICARIOUS_ALPHA", "0"))
    vicarious_threshold = float(os.getenv("ZERO_VICARIOUS_THRESHOLD", "0.3"))
    # coping_potential 独立标量流（议会 2026-07-13；默认关=零回归）。
    # step() 构造 Stimulus 不传 control_appraisal（默认 None），ChatDriver 不持有该旋钮。
    coping_potential_enabled = os.getenv("ZERO_COPING_POTENTIAL_ENABLED", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # text_coping 接线旋钮（议会 2026-07-16 B3；默认关=零回归）。
    # ZERO_TEXT_COPING_ENABLED 未设/false → False → AppraisalAgent B3 走仅 ctrl/两皆 None 分支。
    # ZERO_TEXT_COPING_PRECISION：π_t 上限 ≤0.10（SessionConfig 层 fail-fast）；缺省 0.08。
    text_coping_enabled = os.getenv("ZERO_TEXT_COPING_ENABLED", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    text_coping_precision = float(os.getenv("ZERO_TEXT_COPING_PRECISION", "0.08"))
    # fear_domain_enabled：WARN-3 fear 专属门（B1 BLOCK 前置·议会 2026-07-21·A1；默认关=零回归）。
    # False → 任何路径不产 fear 域激活（两泄漏路径均硬弃/回退）；True 须 env 显式开。
    # anger confrontational 路径完全不受此门。
    fear_domain_enabled = os.getenv("ZERO_FEAR_DOMAIN_ENABLED", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # facs_extended：AU 扩展集合门控（设计门 PASS·路径 b；默认关=零回归）。
    # True → ExpressionAgent 占位路径用 coping_potential_state 驱动 11-AU 扩展集合。
    # 占位路径（decoder=None）经 ExpressionAgent 消费；注入 decoder 若实现 predict_channels_coping
    # 也拿到 per-turn coping（议会遗留 2·方案 b 已落地）。
    facs_extended = os.getenv("ZERO_FACS_EXTENDED", "").lower() in ("1", "true", "yes", "on")
    # voluntary_coping_leak：双通路差异化（议会 C1 设计门 2026-07-14）∈[0,1]。默认 1.0=两头
    # 等值=零回归；推荐 0.3（随意头仅保留自发头 30% coping-driven 强度）。仅 facs_extended 时生效。
    voluntary_coping_leak = float(os.getenv("ZERO_VOLUNTARY_COPING_LEAK", "1.0"))
    # 外部多模态先验流注入口（议会 2026-07-15 M3/M6；config-only-via-env）。
    # external_priors 本身每轮由 state_overrides 注入（MCP 侧），不在此读取。
    # 此处只读会话级固定的校验参数：精度上界 + 流数上界。
    external_prior_precision_cap = float(os.getenv("ZERO_EXTERNAL_PRIOR_PRECISION_CAP", "0.8"))
    max_external_streams = int(os.getenv("ZERO_MAX_EXTERNAL_STREAMS", "5"))
    # 真表情解码器注入（composite 工厂接线）：ZERO_FACS_MODEL_PATH 未设 → None → ExpressionAgent
    # 走解析占位路径（逐字零回归）；设了 → 加载真权重构 CompositeChannelDecoder，训好的真模型在
    # 跑图里生效（per-turn coping 经可选 predict_channels_coping 已可达，议会遗留 2·方案 b）。
    # k_arousal/k_coping：⚖ 方向议会定、幅度工程可动——占位路径用 decode_channels 内置默认
    # （1.5/1.2，不为系数拉 state 字段）；composite 路径经 ZERO_FACS_K_AROUSAL/K_COPING 构造期读入。
    expression_decoder = _build_expression_decoder(facs_extended)
    # B 类·记忆巩固旋钮（仿 cortisol_tau 模式：env 读一次传构造，默认全关=零回归）
    # ZERO_CONSOLIDATION_ENABLED：主门，默认 0=关；开启后 aclose 触发 Ebbinghaus+睡眠巩固。
    consolidation_enabled = os.getenv("ZERO_CONSOLIDATION_ENABLED", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # ZERO_ACTR_ENABLED：ACT-R recency 替换门，默认 0=关（_rank_episodes 用幂律 Δt^(-d)）。
    actr_enabled = os.getenv("ZERO_ACTR_ENABLED", "").lower() in ("1", "true", "yes", "on")
    # SESSION 幂律衰减指数（ZERO_CONSOLIDATION_D_SESSION 默认 0.8；快衰对应海马快速存储）
    d_session = float(os.getenv("ZERO_CONSOLIDATION_D_SESSION", "0.8"))
    # USER 幂律衰减指数（ZERO_CONSOLIDATION_D_USER 默认 0.3；慢衰对应新皮层慢整合）
    d_user = float(os.getenv("ZERO_CONSOLIDATION_D_USER", "0.3"))
    # 睡眠巩固最低强化次数门（ZERO_CONSOLIDATION_COUNT_MIN 默认 3）
    consolidation_count_min = int(os.getenv("ZERO_CONSOLIDATION_COUNT_MIN", "3"))
    # Petrov B sigmoid 归一 scale（ZERO_ACTR_B_SCALE 默认 3.0；越小频率效应越激进）
    actr_b_scale = float(os.getenv("ZERO_ACTR_B_SCALE", "3.0"))
    # aclose 巩固超时秒（ZERO_CONSOLIDATION_TIMEOUT 默认 30.0；超时降级 warning 不崩）
    _consolidation_timeout_env = os.getenv("ZERO_CONSOLIDATION_TIMEOUT")
    consolidation_timeout = (
        float(_consolidation_timeout_env) if _consolidation_timeout_env else 30.0
    )
    # salience 升迁门（Hill 归一后·默认 0.25·议会 2026-07-22）
    consolidation_salience_threshold = float(
        os.getenv("ZERO_CONSOLIDATION_SALIENCE_THRESHOLD", "0.25")
    )
    # user_id=thread：让 disposition/episode 的 user scope 与 ConversationLog 的 thread 对齐，
    # 避免切 ZERO_CHAT_THREAD 时共享 "default-user" 记忆造成串味。
    # cortisol 动力学常数：env → SessionConfig → state → AppraisalAgent（None→回退常量=零回归）。
    session = ConversationSession(
        thread_id=resolved_thread,
        memory=memory,
        user_id=resolved_thread,
        mood_enabled=False,  # A.7 双稳会自锁；chat 用 emotion(快衰退)+attitude(慢累积)
        workspace_enabled=True,
        recall_enabled=True,  # 开语义召回，把 recalled_context 回灌 converse
        sample_sigma_cap=sample_sigma_cap,  # 「防抖」旋钮（ZERO_SAMPLE_SIGMA_MAX；None=零回归）
        rng_seed=rng_seed,  # P5：引擎采样可复现（ZERO_CHAT_RNG_SEED；None=零回归）
        affect_readout=affect_readout,  # P4：'map' 均值读出（ZERO_AFFECT_READOUT；默认 sample）
        arousal_baseline=arousal_baseline,  # P1-c(Q7)：ZERO_AROUSAL_BASELINE；0=零回归
        arousal_gain_cap=arousal_gain_cap,  # P4-d：ZERO_AROUSAL_GAIN_CAP；None=不 cap 零回归
        # A. workspace 三 flag（B4/B5）：默认 False=零回归；env 开启后经 SessionConfig 贯通。
        precision_split=precision_split,
        fuse_independence_correct=fuse_independence_correct,
        ignition_survival_fallback=ignition_survival_fallback,
        # C（A-P2-A）：va_coupling 非对称系数；None=occ_prior 用 0.6/0.6（零回归）。
        va_coupling_pos=va_coupling_pos,
        va_coupling_neg=va_coupling_neg,
        # E（WARN-4 / A-P1-D）：Panksepp fear/rage 次级区分开关；默认 False=零回归。
        panksepp_distinguish_fear=panksepp_distinguish_fear,
        # 议会 2026-07-02 Item 2：ignite 软门控陡度；None=硬 step 零回归。
        ignition_beta=ignition_beta,
        # T6-a/b：mood/text 流精度旋钮（默认=现常量，逐字零回归）。
        mood_precision=mood_precision,
        text_affect_precision=text_affect_precision,
        # P3 HPC 旋钮（默认 1/0.0=平层零回归）。
        hierarchical_layers=hpc_layers,
        hierarchical_coupling=hpc_coupling,
        # P3 1-B：HPA 皮质醇慢回路旋钮（默认全关=零回归）。
        cortisol_enabled=cortisol_enabled,
        cortisol_arousal_gate=cortisol_arousal_gate,
        cortisol_attitude_gate=cortisol_attitude_gate,
        cortisol_arousal_alpha=cortisol_arousal_alpha,
        cortisol_attitude_alpha=cortisol_attitude_alpha,
        # 动力学常数（None→AppraisalAgent 回退 affect_math 议会推荐常量=零回归；env 覆盖）。
        cortisol_tau=cortisol_tau,
        cortisol_impulse=cortisol_impulse,
        cortisol_theta_goal=cortisol_theta_goal,
        cortisol_theta_intensity=cortisol_theta_intensity,
        # P3 1-C：ToM 共情系数（默认 0/0/0/0.3=零回归；经 SessionConfig → to_state_flags 贯通）。
        contagion_alpha=contagion_alpha,
        care_bias_alpha=care_bias_alpha,
        vicarious_alpha=vicarious_alpha,
        vicarious_threshold=vicarious_threshold,
        # coping_potential 独立标量流（议会 2026-07-13；默认关=零回归）。
        coping_potential_enabled=coping_potential_enabled,
        # text_coping 接线旋钮（议会 2026-07-16 B3；默认关=零回归）。
        text_coping_enabled=text_coping_enabled,
        text_coping_precision=text_coping_precision,
        # fear_domain_enabled：WARN-3 fear 专属门（B1 BLOCK 前置·议会 2026-07-21；默认关=零回归）。
        fear_domain_enabled=fear_domain_enabled,
        # facs_extended：AU 扩展集合门控（默认关=零回归）。
        facs_extended=facs_extended,
        # voluntary_coping_leak：双通路差异化（C1 设计门 2026-07-14；默认 1.0=零回归）。
        voluntary_coping_leak=voluntary_coping_leak,
        # 外部多模态先验流注入口（议会 2026-07-15 M3/M6；默认=零回归）。
        external_prior_precision_cap=external_prior_precision_cap,
        max_external_streams=max_external_streams,
        # 真表情解码器（ZERO_FACS_MODEL_PATH 门控；None=占位路径零回归）。
        expression_decoder=expression_decoder,
    )
    return ChatDriver(
        thread=resolved_thread,
        lm=lm,
        log=log,
        session=session,
        history=history,
        attitude=attitude,
        mode=mode,
        persona=persona,
        memory=memory,
        seed_key=resolved_thread,  # 种子记忆 user 作用域 key（= user_id = thread）
        first_contact=first_contact,
        rng_seed=rng_seed,  # P5：chat 层情绪噪声可复现（同一 ZERO_CHAT_RNG_SEED；None=零回归）
        intensity_floor=intensity_floor,
        hab_tau=hab_tau,
        reversion_a=reversion_a,
        setpoint_a=setpoint_a,
        decay_k=decay_k,
        fam_tau=fam_tau,
        baseline_w=baseline_w,
        noise_std=noise_std,
        primacy_k=primacy_k,
        window_n=window_n,
        inject_min=inject_min,
        importance_scale=importance_scale,
        standard_compliance_enabled=standard_compliance_enabled,
        text_domain_enabled=text_domain_enabled,
        # B（B7a）：两时间尺度旋钮透传（默认=旧行为，零回归）
        attitude_arousal_weight=attitude_arousal_weight,
        sensitization_gain=sensitization_gain,
        sensitization_threshold=sensitization_threshold,
        # P3 1-B：cortisol→ATTITUDE_RATE 消费旋钮（默认关=零回归）
        cortisol_attitude_gate=cortisol_attitude_gate,
        cortisol_attitude_alpha=cortisol_attitude_alpha,
        # P3 1-C：ToM 共情系数（默认全 0/0.3=零回归）
        contagion_alpha=contagion_alpha,
        care_bias_alpha=care_bias_alpha,
        vicarious_alpha=vicarious_alpha,
        vicarious_threshold=vicarious_threshold,
        # B 类·记忆巩固旋钮透传（默认全关=零回归）
        consolidation_enabled=consolidation_enabled,
        actr_enabled=actr_enabled,
        d_session=d_session,
        d_user=d_user,
        consolidation_count_min=consolidation_count_min,
        actr_b_scale=actr_b_scale,
        consolidation_timeout=consolidation_timeout,
        consolidation_salience_threshold=consolidation_salience_threshold,
    )
