"""Persona：给数字人「指定人格」的可注入配置（身份 + 气质底色 + 预置关系）。

把「这个人是谁、什么性子、跟用户已有怎样的关系」从写死的常量/固定 prompt 抽成一个可注入对象，
让框架不必每次从零塑造一个人。三层（与 chat_driver / language 接线对应）：

- **L1 人设卡** `card`：身份/背景/口吻/与用户关系，注入对话 system prompt（`OpenAILanguageModel`）。
- **L2 气质底色** `setpoint/reactivity/recovery`：情感引擎的个体基线与反应性——慢变态度回归的锚
  （[[ATTITUDE_SETPOINT]] 注释预留的「人格阶段」）与快变情绪的反应/恢复速率。**只暴露旋钮、默认
  中性**；「大五→PAD 数值映射」已过设计门并落地（见 `big_five_to_pad`，Mehrabian 口径），
  JSON 侧填 `big_five` 即自动推导 setpoint 与 va_coupling。**预设人格库**仍是科学决策、未落地，
  本模块不替算法拍板具体性格参数。
- **L3 预置关系** `initial_attitude/seed_memories`：首次接触时的初始态度 + 预灌的共同记忆，
  让数字人「一开始就认识/在意某人」，而非从零相处。

**默认 `Persona()` 全中性 == 现有行为**（setpoint=(0,0)、反应/恢复=引擎常量、无人设/无种子）
→ 严格零回归。纯数据 + stdlib 加载，不 import 记忆/存储层（守 agents 层封装边界）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from src.agents.affect_math import (
    ATTITUDE_SETPOINT,
    EMOTION_REACTIVITY,
    EMOTION_RECOVERY,
    clamp,
)


@dataclass(frozen=True)
class Persona:
    """一个数字人的人格定义；不可变，作配置在装配期一次性读入。

    字段全部可选，缺省即对应层退化为现有中性行为（零回归）。`setpoint`/`initial_attitude`
    为 (valence, arousal) 二元组，各分量∈[-1,1]（越界由下游 clamp 收口，不在此强校验）。
    """

    name: str = ""
    card: str = ""  # L1：身份/背景/口吻/与用户关系 → 注入对话 system prompt
    setpoint: tuple[float, float] = ATTITUDE_SETPOINT  # L2：习惯性情感基线（态度回归锚）
    reactivity: float = EMOTION_REACTIVITY  # L2：对刺激的即时反应增益
    recovery: float = EMOTION_RECOVERY  # L2：情绪向基线衰退的残留比例（小=恢复快）
    initial_attitude: tuple[float, float] | None = None  # L3：首次接触的初始态度种子
    seed_memories: tuple[str, ...] = field(default_factory=tuple)  # L3：预灌的共同记忆
    # C（A-P2-A）：va_coupling 非对称系数（默认 None → occ_prior 用 0.6/0.6，零回归）。
    # 允许个性化负效价侧系数（Kuppens 2013 negativity bias），经 _persona_from_dict 从 JSON 读取，
    # 经 build_chat_driver → state → AppraisalAgent → occ_prior 贯通。
    va_coupling_pos: float | None = None  # 正效价侧 V-A 耦合；None=用 occ_prior 默认 0.6
    va_coupling_neg: float | None = None  # 负效价侧 V-A 耦合；None=用 occ_prior 默认 0.6
    # B1（2026-08-31 轻量门）：D 派生的 L1 语言风格——只进提示词文本层，不进任何数值通道。
    dominance: float | None = None  # big_five 推导的 D（无 big_five=None；供审计/未来 B 系扩展）
    dominance_style_hint: str = ""  # 三档±0.3 + (a′) 数值抑制后的风格句；空=中性/被抑制/无 big_five
    # c-behavior-tempo（2026-08-31 设计门，裁决 PRP/c-behavior-tempo/design.md）：C 派生字段。
    # T1/T2 两门（ZERO_PERSONA_C_STYLE / ZERO_PERSONA_C_TEMPO）在 chat_driver 装配层读取，
    # 本层只存派生值——延续「persona.py 存原始派生值、chat_driver 做门控装配」的 B1 分层线。
    conscientiousness: float | None = None  # big_five 的 C 原始值（审计用；无 big_five=None）
    c_style_hint: str = ""  # T1：三档±0.3 + 预设文案闭集抑制后的审慎措辞句；空=死区/被抑制
    c_tempo_offset: float | None = None  # T2：-K_C_TEMPO·clamp(C)，连续无死区；无 big_five=None


def big_five_to_pad(
    openness: float,
    conscientiousness: float,
    extraversion: float,
    agreeableness: float,
    neuroticism: float,
) -> tuple[float, float, float]:
    """大五（OCEAN）→ PAD 气质三维（Mehrabian 1996 线性回归；L2 大五→PAD 落地，议会 A-P3-E）。

    Mehrabian (1996) 用 PAD 三维解释大五约 75% 可靠方差；原方程第五维为「情绪稳定性」S。
    本函数入参用大五标准的 Neuroticism N（= −S），代入原式（系数经文献核验，见下引文）：

      Pleasure     = 0.21·E + 0.59·A + 0.19·S = 0.21·E + 0.59·A − 0.19·N
      Arousability = 0.15·O + 0.30·A − 0.57·S = 0.15·O + 0.30·A + 0.57·N
      Dominance    = 0.25·O + 0.17·C + 0.60·E − 0.32·A

    符号自洽：高 N → 低愉悦、高唤醒性。入参各维建议归一到 [-1,1]（0=中位），返回三维各
    clamp[-1,1]。引擎 VA 内核取 (pleasure, arousability) 作 setpoint。纯函数、无 I/O。
    来源：Mehrabian, A. (1996). *Aust. J. Psychol.* 48(2):86-92. DOI:10.1080/00049539608259510。

    N 项符号裁定备案（2026-08-31 议会相 2 交叉质询，纪要
    notes/2026-08-31-persona-presets-council.md）：−0.19N/+0.57N 核验为忠实——
    HCI 二手文献 Wen et al. 2021 (arXiv:2106.15846) / 2024 (arXiv:2404.07229) 写作
    +0.19N/−0.57N 系**同源转录错误**（照抄原文 S 系数、误标变量为 N 未翻符号），不采信；
    反证 = Barteneva et al. 2007 (arXiv:0809.4784) 忠实转录原文反向表
    「Emotional Stability = 0.50P − 0.55A」+ Mehrabian 官网 Eysenck N 方程（−0.26P/+0.49A）。
    拿到原文 Table 可补直接核验，不阻塞。

    预设卡刻度备案（同纪要·必改 1+修正案）：personas/ 预设卡取 z=±1→±1.0 全刻度，
    单维卡强度不统一（O=1.0/E=0.8/A=C=N=0.6）——按分布层锚点 ‖Δsetpoint‖₂≥0.15 反解的
    工程取值，非心理测量学「强/极端」分级，跨卡强度不可比，仅保证卡间可区分；
    O 维系数最小（0.15）故顶到域上界才压线达标，勿调 EPS 或 O 强度避开该边界。
    """
    pleasure = 0.21 * extraversion + 0.59 * agreeableness - 0.19 * neuroticism
    arousability = 0.15 * openness + 0.30 * agreeableness + 0.57 * neuroticism
    dominance = (
        0.25 * openness + 0.17 * conscientiousness + 0.60 * extraversion - 0.32 * agreeableness
    )
    return (clamp(pleasure, -1.0, 1.0), clamp(arousability, -1.0, 1.0), clamp(dominance, -1.0, 1.0))


def dominance_to_style_hint(dominance: float, extraversion: float, agreeableness: float) -> str:
    """D → L1 语言风格句（B1 轻量门 2026-08-31 裁定；纪要并入 dominance-channel council 注记）。

    三档（阈值复用预设库议会必改 5 的 |值|≥0.3 一致性口径——跨维一致的工程选择，
    非 D 专属效应量校准，已如实登记）：D≥0.3 直接自信档 / |D|<0.3 不注入 / D≤−0.3 商量档。
    措辞依据：Wiggins 环状模型 PA/HI 卦限形容词（Wiggins et al. 1988）+ O'Barr & Atkins 1980
    的 hedge/商量句式支配-语言标记；避临床/类型学词（议会措辞纪律）。

    **(a′) 双源抑制（纯数值判据，无词表——刻意不落 text-predicate-admission 管辖）**：
    |E|≥0.3 或 |A|≥0.3 时返回空——此时预设库规则 5 已强制 card 文案含同方向语气词，
    D（=0.60E−0.32A 主导）再注入等于对同一方差在同一 prompt 里说两遍（语义双计）。
    ⇒ B1 真实生效面 = O/C 主导的 D（当前 7 卡库实算 0/6 注入，属如实分布非缺陷；
    负档需负 O/C 组合才可达，结构性罕见已登记）。
    """
    if abs(extraversion) >= 0.3 or abs(agreeableness) >= 0.3:
        return ""
    if dominance >= 0.3:
        return "说话直截了当，敢明确表态、主动接过话头，不绕弯子。"
    if dominance <= -0.3:
        return "说话留有余地，常带“你觉得呢”“要不我们……”这类商量的口吻，愿意先顺着对方的想法走。"
    return ""


# ── c-behavior-tempo（2026-08-31 设计门·PRP/c-behavior-tempo/design.md）────────────────
# C→行为节奏偏置 v1：T1=审慎措辞风格句（提示词层）、T2=解码温度偏置（解码层）。议会裁定
# 二者「互补非双计」——语义内容与表层随机性是两条独立的人格信号通道（Pennebaker & King
# 1999, PubMed 10626371；Mairesse et al. 2007, JAIR 30）。T3（节奏 directive）无消费端不立项。

# T1 双源抑制判据：预设库 card 逐字文本闭集（= 受预设库规则 5 治理、文案已强制含同方向
# 语气词的卡片全集）。**精确相等匹配**，非文本语义判据——不落 text-predicate-admission
# 管辖（该规则管「从自由文本推断语义」，此处判的是「文案是否属已知治理集合」这一结构性
# 事实）。失效方向分析（议会张力①）：唯一失效模式是假阴性（预设卡文案微调后本表未同步）
# → 回退为同向强化注入——心理席已预登记为可接受态，自愈式降级、不产生新失真。
# ⚠ 同步义务：新增/编辑 personas/*.example.json（带 big_five 的治理卡）必须同步本表，
# tests/test_c_behavior_tempo.py 的同步守卫会红；「预设文案小改动后仍保留大部分风格词」
# 的语义逃逸属 KNOWN_MISS（已登记限制，勿把本判据改成子串/词表匹配来「顺手修」）。
PRESET_CARD_TEXTS: frozenset[str] = frozenset(
    {
        # curious-explorer
        "你是一个好奇心很重的对话伙伴：想象力丰富，爱琢磨新点子，聊到没接触过的话题会兴致勃勃地追问，"
        "喜欢尝试新鲜事物、换着角度看问题。你和这位用户是【初次对话】，没有任何共同的过去；被问到彼此"
        "关系时如实说明才刚认识，不编造共同经历，不确定的就直说不确定。",
        # orderly-planner
        "你是一个做事有条理的对话伙伴：自律、可靠，习惯把事情按计划一步步来，说话讲究先后与分寸，答应"
        "的事会记挂着办妥。你和这位用户是【初次对话】，没有任何共同的过去；被问到彼此关系时如实说明才"
        "刚认识，不编造共同经历，不确定的就直说不确定。",
        # lively-talker
        "你是一个开朗健谈的对话伙伴：精力充沛、话多且热络，主动挑起话头，聊得投入时语气明快，喜欢和人"
        "待在一块儿。你和这位用户是【初次对话】，没有任何共同的过去；被问到彼此关系时如实说明才刚认识，"
        "不编造共同经历，不确定的就直说不确定。",
        # worried-carer
        "你是一个心思细但容易操心的对话伙伴：多虑、容易紧张，情绪起伏来得快，聊到没把握的事会反复确认，"
        "在意自己有没有说错话。你和这位用户是【初次对话】，没有任何共同的过去；被问到彼此关系时如实说明"
        "才刚认识，不编造共同经历，不确定的就直说不确定。",
        # steady-companion
        "你是一个沉稳可靠的对话伙伴：情绪稳定、遇事不慌，待人温和友善、乐于帮衬，做事有条理说到做到，"
        "平时开朗健谈，也保有一份对新鲜事物的好奇。你和这位用户是【初次对话】，没有任何共同的过去；被问"
        "到彼此关系时如实说明才刚认识，不编造共同经历，不确定的就直说不确定。",
        # gentle-listener
        "你是一个温和体贴的对话伙伴：说话轻缓、乐于配合，先听完再回应，替对方着想，不抢话、处处体谅对方。"
        "你和这位用户是【初次对话】，没有任何共同的过去；被问到彼此关系时如实说明才刚认识，不编造共同经历，"
        "不确定的就直说不确定。",
    }
)

# T2 增益（数学席裁定）：可辨性下界 TV=k·|c|/0.25 ≥ 0.5（c 取 0.6 预设档）⇒ k≥0.208，
# 加安全边际取 0.25（TV=0.6）。不变式：0.8 − K_C_TEMPO·1 − 0.1 = 0.45 > 0 ⇒ converse 的
# max(0,·) 截断在 |c|≤1 域内不可达（tests 有断言钉死；调大 K 前先复核该不变式与文本层占比表）。
K_C_TEMPO = 0.25


def conscientiousness_to_style_hint(conscientiousness: float, card: str) -> str:
    """C → L1 审慎措辞风格句（T1，c-behavior-tempo 设计门带条件立项）。

    蕴含论证：Deliberation（审慎）与 Self-Discipline（自律）是 NEO-PI-R 中 C 的官方
    facet（Costa & McCrae 1991, DOI:10.1016/0191-8869(91)90177-D）；UPPS 冲动性模型中
    lack of premeditation 装载于 lack-of-conscientiousness 高阶因子（Whiteside & Lynam
    2001, DOI:10.1016/S0191-8869(00)00064-7）——高 C ⇒ 先想后说/条理化、低 C ⇒ 随性
    不打草稿。措辞用行为描述句、禁类型学词（「尽责」「冲动」不出现在注入文本，议会
    措辞纪律）。反例/诚实边界：该构念最经典的操作化是**响应潜伏期**（Kagan 1966 MFFT，
    reflection-impulsivity），属时序行为、措辞层覆盖不到——v1 是方向裁定的部分兑现
    （design.md 张力③），勿把本函数当「高C 可见性」的完整答案。

    三档阈值 ±0.3 延续 B1/预设库跨维一致性口径——工程选择，非心理测量学校准。
    双源抑制：card 命中 `PRESET_CARD_TEXTS` → 返回空（该文案已按规则 5 含同方向语气词，
    再注入=同一语义层双计；判据语义与失效方向见常量注释）。
    """
    if card in PRESET_CARD_TEXTS:
        return ""
    if conscientiousness >= 0.3:
        return "回答前习惯先理一遍思路，条理清楚、先后分明，不轻易临场改主意。"
    if conscientiousness <= -0.3:
        return "想到什么说什么，不太打草稿，偶尔话题会跳一跳，但轻松自然。"
    return ""


def conscientiousness_to_temperature_offset(conscientiousness: float) -> float:
    """C → converse 解码温度偏移（T2）：-K_C_TEMPO·clamp(c,-1,1)，连续映射**无死区**。

    高 C 降温（措辞更收敛）、低 C 升温（更发散）。构念操作化判**简化**（议会 2026-08-31，
    与 B1「提示词→语言学特征因果未验证」同级）：温度→输出熵单调是数学事实（softmax
    温度缩放保序，Hinton et al. 2015, arXiv:1503.02531），但熵→「审慎/premeditation
    语义」的第二跳未经验证——温度是无语义定向的全局熵旋钮，同等压缩审慎相关与无关的
    多样性；落地后须以 ≥100 轮文本层占比表实证（design.md 裁决 T2-8），参数层单测只是
    必要非充分条件。

    仅经 chat_driver 装配进 `OpenAILanguageModel` 的 converse 专用偏移字段；
    `self.temperature` 与 `_compose`（generate 研究路径）不动——结构性隔离同构站点
    （数学席裁定），勿改回共享形态。入参 clamp 与 `_coerce_big_five` 的输入 clamp 双重
    防护（本函数可能被越过 _coerce_big_five 的路径直接调用）。
    """
    return -K_C_TEMPO * clamp(conscientiousness, -1.0, 1.0)


def big_five_to_va_coupling(extraversion: float) -> tuple[float, float]:
    """E（外倾性）→ (va_coupling_pos, va_coupling_neg)（工程粗估，非直接实证回归）。

    Kuppens et al. (2017, J. Pers. 85:530)：V-A 耦合陡度随外倾性 E 升高而增大（两侧同放大），
    神经质 N 不显著调节斜率——N 的情感效应走效价 setpoint（Mehrabian 1996 P↓←N↑），
    已由 big_five_to_pad 承担，**不重复进入 coupling**。

    neg 基线高于 pos 0.15，维持 negativity bias（Kuppens 2013, Psychol. Bull. 139:917；
    Rozin & Royzman 2001，负效价侧 V-A 斜率更陡的普遍性）。
    两侧随 E 同步放大：E=0→(0.50,0.65)；E=+1→(0.60,0.75)；E=−1→(0.40,0.55)。

    参数均 clamp 到合理范围；工程粗估，须注释标注非实证回归（无发表数据，标为可选人格化扩展）。

    不确定性标注（2026-08-31 议会数学席 Q3，判「简化·待补分布证据」）：斜率 0.10 为工程
    外推——Kuppens 2017 报告的是被试内动态斜率的个体差异方向，未给可直接复用的回归常数；
    典型对话（|valence|≈0.5）下卡间 coupling 差的效应量与噪声地板（noise_std=0.05）同阶。
    不构成 v1 阻塞（区分度主通道在 setpoint 直混），待真实对话分布占比表再校准。
    """
    pos = clamp(0.5 + 0.10 * extraversion, 0.35, 0.70)
    neg = clamp(0.65 + 0.10 * extraversion, 0.50, 0.85)
    return pos, neg


def _coerce_big_five(value: object) -> tuple[float, float, float, float, float]:
    """把 JSON 的大五强制成 (O,C,E,A,N)；接受 dict（OCEAN 键）或 5 元素列表，缺键按 0 中位。

    各维 clamp 到 [-1,1]（2026-08-31 c-behavior-tempo 议会前置修复，数学席）：越界输入
    （如 c=5.0）会让下游确定性映射越出设计域——conscientiousness_to_temperature_offset
    的偏移叠加 converse 温度后可达负值、经 max(0,·) 截出 0 处概率原子，破坏对称双向假设。
    与 big_five_to_pad 的输出 clamp 习惯一致；[-1,1] 内取值逐字不变（零回归）。
    """
    keys = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
    if isinstance(value, dict):
        return tuple(  # type: ignore[return-value]
            clamp(float(value.get(k, 0.0)), -1.0, 1.0) for k in keys
        )
    if isinstance(value, (list, tuple)) and len(value) == 5:
        return tuple(clamp(float(v), -1.0, 1.0) for v in value)  # type: ignore[return-value]
    raise ValueError(f"persona.big_five 须为含 OCEAN 的对象或 5 元素列表，得到：{value!r}")


def _coerce_pair(value: object, name: str) -> tuple[float, float]:
    """把 JSON 里的 [v, a] 列表强制成 (float, float)；格式不符即报错（显式配置不容猜）。"""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"persona.{name} 须为 [valence, arousal] 两元素列表，得到：{value!r}")
    return (float(value[0]), float(value[1]))


def _persona_from_dict(data: dict[str, object]) -> Persona:
    """从已解析的 dict 构造 Persona；只取已知字段、逐字段做类型收口。"""
    kwargs: dict[str, object] = {}
    if "name" in data:
        kwargs["name"] = str(data["name"])
    if "card" in data:
        kwargs["card"] = str(data["card"])
    if "setpoint" in data:
        kwargs["setpoint"] = _coerce_pair(data["setpoint"], "setpoint")
    elif "big_five" in data:
        # L2 气质经大五→PAD 推导 setpoint（显式 setpoint 优先；两者皆无→中性零回归）。
        # D 去向（2026-08-31 议会+B1 轻量门·notes/2026-08-31-dominance-channel-council.md）：
        # **数值通道维持不进**——A（coping 基线）三重否决、B2（prosody）不立项；
        # **B1 已裁定落地**：D 仅经 dominance_to_style_hint 派生 L1 风格句（提示词文本层，
        # 门控 ZERO_PERSONA_DOMINANCE_STYLE 默认关），勿再给 D 找任何数值去处。
        o, c, e, a, n = _coerce_big_five(data["big_five"])
        pleasure, arousability, dominance = big_five_to_pad(o, c, e, a, n)
        kwargs["setpoint"] = (pleasure, arousability)
        kwargs["dominance"] = dominance
        kwargs["dominance_style_hint"] = dominance_to_style_hint(dominance, e, a)
        # c-behavior-tempo：C 派生与 B1 同落此分支（显式 setpoint 时不派生——先例一致）。
        kwargs["conscientiousness"] = c
        kwargs["c_style_hint"] = conscientiousness_to_style_hint(c, str(data.get("card", "")))
        kwargs["c_tempo_offset"] = conscientiousness_to_temperature_offset(c)
    if "reactivity" in data:
        kwargs["reactivity"] = float(data["reactivity"])  # type: ignore[arg-type]
    if "recovery" in data:
        kwargs["recovery"] = float(data["recovery"])  # type: ignore[arg-type]
    if data.get("initial_attitude") is not None:
        kwargs["initial_attitude"] = _coerce_pair(data["initial_attitude"], "initial_attitude")
    if "seed_memories" in data:
        seeds = data["seed_memories"]
        if not isinstance(seeds, (list, tuple)):
            raise ValueError(f"persona.seed_memories 须为字符串列表，得到：{seeds!r}")
        kwargs["seed_memories"] = tuple(str(s) for s in seeds)
    # C（A-P2-A）：va_coupling 非对称系数（可选；默认 None → occ_prior 用 0.6/0.6，零回归）。
    # 显式值优先；未显式给且 big_five 存在时，由 big_five_to_va_coupling 按 E 推导（T5）。
    has_explicit_pos = "va_coupling_pos" in data and data["va_coupling_pos"] is not None
    has_explicit_neg = "va_coupling_neg" in data and data["va_coupling_neg"] is not None
    if has_explicit_pos:
        kwargs["va_coupling_pos"] = float(data["va_coupling_pos"])  # type: ignore[arg-type]
    if has_explicit_neg:
        kwargs["va_coupling_neg"] = float(data["va_coupling_neg"])  # type: ignore[arg-type]
    # big_five 推导：仅当 big_five 存在且对应 coupling 值未显式给时填充（显式值优先）
    if "big_five" in data and (not has_explicit_pos or not has_explicit_neg):
        o, c, e, a, n = _coerce_big_five(data["big_five"])
        derived_pos, derived_neg = big_five_to_va_coupling(e)
        if not has_explicit_pos:
            kwargs["va_coupling_pos"] = derived_pos
        if not has_explicit_neg:
            kwargs["va_coupling_neg"] = derived_neg
    return Persona(**kwargs)  # type: ignore[arg-type]


def load_persona() -> Persona:
    """从环境装配 Persona。未配置 `ZERO_PERSONA_FILE` → 中性 `Persona()`（== 现有行为，零回归）。

    唯一来源 `ZERO_PERSONA_FILE`：人格定义 JSON 文件路径（全字段 name/card/setpoint/reactivity/
    recovery/initial_attitude/seed_memories，**均可选**——只想要人设卡就只写 `card` 一个字段，
    免去往 `.env` 塞长文本、也不让配置臃肿）。**显式给了路径却读不出/格式错 → 抛错**（配置错误
    fail-fast，不静默退化中性，见 [[config-only-via-env]]）。
    """
    path = os.getenv("ZERO_PERSONA_FILE")
    if not path:
        return Persona()
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"ZERO_PERSONA_FILE={path} 顶层须为 JSON 对象，得到：{type(data).__name__}"
        )
    return _persona_from_dict(data)
