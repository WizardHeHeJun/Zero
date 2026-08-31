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
    """把 JSON 的大五强制成 (O,C,E,A,N)；接受 dict（OCEAN 键）或 5 元素列表，缺键按 0 中位。"""
    keys = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
    if isinstance(value, dict):
        return tuple(float(value.get(k, 0.0)) for k in keys)  # type: ignore[return-value]
    if isinstance(value, (list, tuple)) and len(value) == 5:
        return tuple(float(v) for v in value)  # type: ignore[return-value]
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
