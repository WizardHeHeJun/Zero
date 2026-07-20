"""情绪词典层：把内核 e*=(valence, arousal) 落到「有粒度、像人」的文本表达。

承接 notes/2026-06-24-text-output-emotion.md 的文献调研，补三件事（纯函数、确定性、
torch-free / API-free，不动 affect_math.text_label 以保零回归）：

1. 粒度升级（构建论 / Russell circumplex）：`affect_label` 把 VA 平面按极坐标分 8 扇区 ×
   强度分级，给出比 4 档（excited/content/angry/sad）细得多的情绪词。
2. 动机色彩（Panksepp 七大情绪系统）：`motivational_system` 在 VA 之上叠一层动机标签
   （seeking/care/rage/panic_grief…）。注意 VA 二维欠定离散系统（RAGE 与 FEAR 同处 -v+a 象限），
   故该层是「主导系统」近似，不是判别——见 docstring 注记。
3. 词典桥 / 加权解码（NRC-VAD + Affect-LM β）：`affect_logit_bias` 给候选词按
   Δlogit = β·⟨φ(w), e*⟩ 加偏置；`suggest_affect_words` 取与 e* 最对齐的词供提示注入。
   内置 `SEED_VAD_LEXICON` 为小规模种子表，生产可整体替换为 NRC-VAD v2（55k+ 词）。
4. 情绪时间包络（ECM internal memory）：`intensity_envelope` 给一句话内的情绪强度一条
   句首满、句尾衰减的 sigmoid 包络。

本模块只依赖标准库 math 与同层 affect_math.clamp。
"""

from __future__ import annotations

import math

from src.agents.affect_math import clamp

# 情绪强度的「中性死区」半径：r < NEUTRAL_RADIUS 视为平静（避免给微弱情感强行贴词）
NEUTRAL_RADIUS = 0.15
# 强度分级阈值（极坐标半径 r=hypot(v,a)）：弱 / 中 / 强
INTENSITY_MILD = 0.45
INTENSITY_STRONG = 0.8
# 加权解码默认强度旋钮 β（Affect-LM 式；越大越贴情感、过大伤语法，故设默认温和值）
DEFAULT_BIAS_BETA = 1.0
# 情绪时间包络的 sigmoid 陡度
ENVELOPE_STEEPNESS = 6.0
# ⚠ TOMBSTONE（议会 2026-07-02 判失真）：纯 arousal 阈值在 VA 二维区分 RAGE/FEAR
# 无神经生理依据——二者均处 (-v,+a) 象限、arousal 相当；0.6 无文献支撑、方向可能相反
# （高 arousal 的 FEAR=逃跑；RAGE 的 arousal 可低于冻结态 FEAR）。真正分野在动机方向
# （趋近/回避）+ PAD Dominance 维（RAGE 高支配/FEAR 顺从），超出当前纯 VA 管线。
# 神经+生物两席共识：二者同处 (-v,+a)、arousal 相当，真正分野在动机方向/回避-趋近 +
# PAD Dominance；0.6 无文献支撑、方向可能相反。
# 引文：Panksepp 1998/2011（Ch.10 RAGE/Ch.11 FEAR 解剖与神经化学分离）；
#        Carver & Harmon-Jones 2009（Psychol. Bull. 135:183，RAGE=趋近左额叶/FEAR=回避右额叶）；
#        Wacker et al. 2003（Emotion 3:167，EEG 直接证明 anger/fear 动机方向神经可分）。
# 生产不建议启用 distinguish_fear=True；未来接入 Dominance/趋近-回避维度可替换此纯 arousal 逻辑。
# 仅在 motivational_system(distinguish_fear=True) 时生效；默认门控关闭，零回归。
PANKSEPP_FEAR_AROUSAL_THRESHOLD = 0.6

# VA 极坐标 8 扇区（角度从 +valence 轴逆时针，y=arousal）→ 三档强度词（弱/中/强）。
# 词为中文（与语言层中文生成一致）。扇区中心角：0,45,…,315。
_WHEEL: tuple[tuple[float, tuple[str, str, str]], ...] = (
    (0.0, ("愉悦", "高兴", "喜悦")),  # v+ a≈0
    (45.0, ("欣喜", "兴奋", "狂喜")),  # v+ a+
    (90.0, ("专注", "警觉", "激动")),  # a+ v≈0
    (135.0, ("恼火", "愤怒", "暴怒")),  # v- a+（含焦虑/恐惧族，见 motivational_system 注记）
    (180.0, ("不悦", "沮丧", "痛苦")),  # v- a≈0
    (225.0, ("低落", "忧伤", "抑郁")),  # v- a-
    (270.0, ("倦怠", "疲惫", "麻木")),  # a- v≈0
    (315.0, ("放松", "安宁", "满足")),  # v+ a-
)


def _polar(valence: float, arousal: float) -> tuple[float, float]:
    """返回 (半径 r, 角度 deg∈[0,360))；r 为强度，角度定情绪类别。"""
    r = math.hypot(valence, arousal)
    deg = math.degrees(math.atan2(arousal, valence)) % 360.0
    return r, deg


def _intensity_tier(r: float) -> int:
    """半径 → 强度档位：0 弱 / 1 中 / 2 强。"""
    if r >= INTENSITY_STRONG:
        return 2
    if r >= INTENSITY_MILD:
        return 1
    return 0


def affect_label(valence: float, arousal: float) -> str:
    """VA → 细粒度情绪词（circumplex 8 扇区 × 强度分级）。

    r < NEUTRAL_RADIUS 返回「平静」；否则取最近扇区中心对应的强度档词。
    纯函数、确定性；不替代 affect_math.text_label（后者仍作 4 档通道值）。
    """
    r, deg = _polar(valence, arousal)
    if r < NEUTRAL_RADIUS:
        return "平静"
    # 取角度最近的扇区（每 45° 一格）
    idx = int(round(deg / 45.0)) % len(_WHEEL)
    words = _WHEEL[idx][1]
    return words[_intensity_tier(r)]


def motivational_system(
    valence: float,
    arousal: float,
    *,
    distinguish_fear: bool = False,
    coping_potential: float = 0.0,
    text_coping_source: bool = False,
) -> str:
    """VA → Panksepp 主导情绪系统标签（动机色彩）。

    映射（r<NEUTRAL_RADIUS 为 neutral）：
      (+v,+a) → seeking（多巴胺能动机/探索/期待；注意 SEEKING 是动机系统而非情感结果，
                VA 二维对其欠定，Panksepp 2011；此映射为工程近似）
      (+v,-a) → care（照护/亲和/满足）
      (-v,+a) → rage（愤怒/威胁）  ← 默认，**动机系统主导近似、非 RAGE/FEAR 神经判别器**。
                  RAGE 与 FEAR 均处 (-v,+a) 象限，VA 二维无法判别——这是架构上已知的欠定，
                  而非实现缺陷。真正分野在动机方向（趋近/回避）+ PAD Dominance 维（RAGE
                  高支配/FEAR 顺从），超出当前纯 VA 管线（Carver & Harmon-Jones 2009；
                  Wacker et al. 2003）。取 rage 为默认仅因 RAGE 在 (-v,+a) 中更常见/显著，
                  **非基于 arousal 判别 FEAR**。
                  distinguish_fear=True 时按 arousal 阈值次级区分（实验旋钮，议会 2026-07-02
                  判无可辩护阈值，见 PANKSEPP_FEAR_AROUSAL_THRESHOLD tombstone）。
      (-v,-a) → panic_grief（失落/悲伤/分离）

    注记（议会 2026-07-02 · 神经+生物席共识）：RAGE 与 FEAR 是独立皮层下回路
    （Panksepp 1998 Ch.10/11），二者均处 (-v,+a) 象限、arousal 相当；纯 arousal 阈值
    无神经生理依据区分二者（Barrett 2017：fear/anger 共享 core affect，无单一神经指纹）。
    此函数的 (-v,+a) → rage 是**动机系统主导近似**，不声称为 RAGE/FEAR 神经判别器。

    参数
    ----
    distinguish_fear : bool
        默认 False——保持旧行为（(-v,+a) 一律→ rage），**零回归**。
        True 时按 arousal 阈值（PANKSEPP_FEAR_AROUSAL_THRESHOLD）次级区分 fear/rage。
        ⚠ 议会 2026-07-02 判该阈值无可辩护文献依据，生产不建议开启。

    coping_potential : float
        情境控制感 ∈ [-1,1]（默认 0.0=零回归，(-v,+a) 一律→ rage，与旧行为逐字兼容）。
        非零时在 (-v,+a) 象限用 control_appraisal 分离 rage/fear（议会 2026-07-13 P1-D）：
        >COPING_RAGE_THRESHOLD → rage（高控制/趋近）；<COPING_FEAR_THRESHOLD → fear（低控制/回避）；
        中间带哑火（text 来源·B3 约束3）见 text_coping_source 参数。
        由 AppraisalAgent 产出 coping_potential_state；经 language._appraisal_summary 读 state
        透传至此（仅 appraisal_conditioning 开时经该摘要消费，与 distinguish_fear 同一路径）。

        非对称可靠性与命名边界（议会 2026-07-20·SemEval OOD 后订正）：可靠性**域特异·各有弱域**
        ——fear 回避锚皮层下 PAG/杏仁核生存回路（ED 叙事域 Wilson LB≈0.90 稳；但 Twitter 社交焦虑域
        降至 LB=0.709·Davis&Walker 2009 BNST——「fear 全域稳」是 ED 单源幻觉；「情境无关」限防御行为
        触发层面 LeDoux&Brown 2017，非意识 fear 感受本身 Barrett 2017）、anger 趋近依赖前额叶可行动
        性评价（叙事/倾诉域 ~74%·confrontational 域恢复 LB=0.857·非本质下限）。
        故 text 来源 coping 是 anger/fear 的**趋近-回避方向符号先验**（训练方案=符号监督
        motivational_direction_prior；W1 已接线 2026-07-20：PerceptionAgent 经 DirectionHead
        opt-in 产出 text_coping_prior，见 appraisal.py 同段注释），
        非 Lazarus/Scherer 应对评价连续量；anger 侧约 12–20% 低置信弃权回退默认
        （三路切分验证见 scripts/validate_anger_abstain.py），调用者不得假设 anger 信号总在。

    text_coping_source : bool
        默认 False=旧调用方零回归（不触发中间带哑火）。
        True=本轮 coping_potential 值来自文本先验（B3 分支2/4），在 (-v,+a) 象限且
        |coping_potential|≤TEXT_COPING_MIDDLE_BAND 时哑火返回 rage（保守默认），
        防止低信度文本先验在中间带错误翻转 rage/fear（议会 2026-07-16 神经席约束3）。
        仅影响 (-v,+a) 象限的 rage/fear 判断，其余象限无影响。

    env 贯通状态：已接 ZERO_PANKSEPP_DISTINGUISH_FEAR（默认关=零回归）。经
    chat_driver/runner → SessionConfig → AffectState.panksepp_distinguish_fear →
    language.py::_appraisal_summary 读 state 传入（仅 appraisal_conditioning 开时经该摘要可见）。
    on/off 开关已贯通；精确阈值/坐标（PANKSEPP_FEAR_AROUSAL_THRESHOLD）仍待议会 P1-D 定论。

    返回小写枚举串，便于下游条件化与单测。
    """
    # text 来源 coping 中间带哑火边界（议会 2026-07-16 神经席；Wacker et al. 2003）：
    # 极端段方可分 rage/fear；[-0.15,+0.15] 方向可靠度不足，保守回落 rage。
    # 暂定·待方向判据（留出集 Wilson CI 下界≥0.70）后回工程升/降，
    # 届时仿 ignition_beta 走 env 注入（新增 state 字段 + ZERO_TEXT_COPING_MIDDLE_BAND）。
    TEXT_COPING_MIDDLE_BAND = 0.15
    r, _ = _polar(valence, arousal)
    if r < NEUTRAL_RADIUS:
        return "neutral"
    if valence >= 0.0:
        return "seeking" if arousal >= 0.0 else "care"
    # (-v, +a) 象限
    if arousal >= 0.0:
        # ── 中间带哑火 guard（B3 约束3）：text 来源且 |coping| 在中间带 → 保守 rage ──
        # 必须在 COPING_RAGE_THRESHOLD/COPING_FEAR_THRESHOLD 判断之前执行（R3）。
        # 与旧 (-v,+a)→rage 保守默认逐字兼容（中间带 → rage，非 fear）。
        if text_coping_source and abs(coping_potential) <= TEXT_COPING_MIDDLE_BAND:
            return "rage"
        # coping_potential 分离路径（议会 2026-07-13 P1-D）。阈值为函数体内局部变量：
        # ⚖ 议会初值(Smith & Ellsworth 1985 control 维)，待 P1 观测校准，工程不私拍。
        # P1 校准出新值后走 env 注入（仿 ignition_beta 新增 state 透传），届时改此为参数。
        COPING_RAGE_THRESHOLD = 0.3
        COPING_FEAR_THRESHOLD = -0.3
        if coping_potential > COPING_RAGE_THRESHOLD:
            return "rage"
        if coping_potential < COPING_FEAR_THRESHOLD:
            return "fear"
        # 中间段 [-0.3, 0.3]：保守默认 rage（与旧 (-v,+a)→rage 逐字兼容）
        # 旧 distinguish_fear 路径（tombstone 标注；coping_potential=0 时走此回退）
        if distinguish_fear and arousal >= PANKSEPP_FEAR_AROUSAL_THRESHOLD:
            return "fear"
        return "rage"
    return "panic_grief"


def affect_descriptor(valence: float, arousal: float) -> str:
    """组合「细粒度词·动机系统」一行描述，供模板文本 / 提示注入复用。"""
    return f"{affect_label(valence, arousal)}·{motivational_system(valence, arousal)}"


# 种子 VAD 词典：中文情绪词 → (valence, arousal)，约定取值 [-1,1]。
# 小规模、覆盖一圈，仅作词典桥/加权解码的可单测内核；生产可整体替换为 NRC-VAD v2。
SEED_VAD_LEXICON: dict[str, tuple[float, float]] = {
    "狂喜": (0.9, 0.9),
    "兴奋": (0.7, 0.8),
    "高兴": (0.8, 0.5),
    "期待": (0.5, 0.4),
    "感激": (0.7, 0.2),
    "温暖": (0.7, -0.1),
    "满足": (0.7, -0.2),
    "放松": (0.5, -0.6),
    "安宁": (0.6, -0.7),
    "平静": (0.0, -0.4),
    "无聊": (-0.3, -0.5),
    "疲惫": (-0.2, -0.6),
    "失望": (-0.5, -0.2),
    "沮丧": (-0.5, -0.3),
    "忧伤": (-0.7, -0.3),
    "抑郁": (-0.7, -0.5),
    "焦虑": (-0.6, 0.6),
    "紧张": (-0.4, 0.6),
    "恐惧": (-0.7, 0.7),
    "厌恶": (-0.7, 0.4),
    "恼火": (-0.6, 0.5),
    "愤怒": (-0.8, 0.7),
    "暴怒": (-0.9, 0.9),
    # 惊讶效价中性化（A-P2-D）：Russell/Scherer 指惊讶效价不稳定，改为中性 (0.0, 0.7)。
    # 细分词：正向惊讶("惊喜") / 负向惊讶("惊吓") 独立词条。
    "惊讶": (0.0, 0.7),
    # 惊喜：Russell 1980 astonished ~69.8°（v+），0.6 偏强 → 精化为 0.5（议会 2026-07-02）。
    "惊喜": (0.5, 0.7),
    # 惊吓：Russell 1980 alarmed ~96.5°，与恐惧 (-0.7,0.7) 拉开距离；负效价弱、唤醒略升
    #       (-0.3, 0.8)（议会 2026-07-02 精化；旧 (-0.6,0.7) 与恐惧仅差 0.1 区分度不足）。
    "惊吓": (-0.3, 0.8),
}


def _affect_dot(word_vad: tuple[float, float], e_star: tuple[float, float]) -> float:
    """⟨φ(w), e*⟩：词的 VA 向量与目标情感的内积（对齐度）。"""
    return word_vad[0] * e_star[0] + word_vad[1] * e_star[1]


def affect_logit_bias(
    words: list[str],
    e_star: tuple[float, float],
    *,
    beta: float = DEFAULT_BIAS_BETA,
    lexicon: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """加权解码偏置：Δlogit(w) = β·⟨φ(w), e*⟩（Affect-LM / NRC-VAD 词典桥）。

    与 e* 同向的词得正偏置、反向得负偏置；词典外的词偏置为 0（不干预）。
    beta=0 时全 0（关闭）。纯函数、确定性，不依赖具体 LM。
    """
    table = lexicon if lexicon is not None else SEED_VAD_LEXICON
    out: dict[str, float] = {}
    for w in words:
        vad = table.get(w)
        out[w] = beta * _affect_dot(vad, e_star) if vad is not None else 0.0
    return out


def suggest_affect_words(
    valence: float,
    arousal: float,
    *,
    k: int = 5,
    lexicon: dict[str, tuple[float, float]] | None = None,
) -> list[str]:
    """取与 e*=(valence, arousal) 最对齐的前 k 个情绪词（供提示注入 / 重排）。

    按 ⟨φ(w), e*⟩ 降序；同分时按词稳定排序保证确定性。
    """
    table = lexicon if lexicon is not None else SEED_VAD_LEXICON
    e_star = (valence, arousal)
    ranked = sorted(table.items(), key=lambda kv: (-_affect_dot(kv[1], e_star), kv[0]))
    return [w for w, _ in ranked[: max(0, k)]]


def appraise_text(
    text: str,
    *,
    lexicon: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """词典法反推文本情感：取文本中命中的情绪词 VAD 均值（路径 1 的反向）。

    无命中词 → 中性 (0.0, 0.0)。确定性、无外部依赖；供无 LLM 时的轻量情感反推
    （如 SteeringLanguageModel 的 appraisal-back）。生产可换 STTextAffectRegressor。
    """
    table = lexicon if lexicon is not None else SEED_VAD_LEXICON
    hits = [vad for w, vad in table.items() if w in text]
    if not hits:
        return (0.0, 0.0)
    n = len(hits)
    return (sum(v for v, _ in hits) / n, sum(a for _, a in hits) / n)


# ---------------------------------------------------------------------------
# 社会规范遵从评价桥（B6·OCC 分支 B 通电）
# ---------------------------------------------------------------------------
# 议会 2026-07-02 精化（Item 1）：词表升级为三级加权 dict + 自指过滤。
# 纯确定性词典/规则，绝不调 LLM/embedding（守热路径红线）。

# 规范违反词：三级权重（轻度 0.5 / 标准 1.0 / 强烈 1.5）
# 议会依据：OCC praiseworthiness 有强度差（Ortony et al. 1988 Ch.4）；轻度贬义 ≠ 脏话威胁。
_VIOLATION_WEIGHTS: dict[str, float] = {
    # 轻度消极描述（0.5）
    "蠢": 0.5,
    "笨": 0.5,
    "丑": 0.5,
    "无聊": 0.5,
    # 标准辱骂/命令式威胁（1.0）
    "白痴": 1.0,
    "废物": 1.0,
    "垃圾": 1.0,
    "滚": 1.0,
    "闭嘴": 1.0,
    "混蛋": 1.0,
    "bitch": 1.0,
    "asshole": 1.0,
    # 强烈脏话/死亡威胁（1.5）
    "去死": 1.5,
    "死": 1.5,
    "他妈的": 1.5,
    "fuck": 1.5,
    "nmsl": 1.5,
    "草": 1.5,
}

# 规范遵从词：三级权重（轻度礼貌 0.5 / 标准致谢道歉 1.0 / 强烈赞许钦佩 1.5）
_COMPLIANCE_WEIGHTS: dict[str, float] = {
    # 轻度礼貌（0.5）
    "请": 0.5,
    "麻烦": 0.5,
    "劳烦": 0.5,
    "please": 0.5,
    "kindly": 0.5,
    # 标准致谢/道歉（1.0）
    "谢谢": 1.0,
    "感谢": 1.0,
    "对不起": 1.0,
    "抱歉": 1.0,
    "sorry": 1.0,
    "thank you": 1.0,
    "thanks": 1.0,
    "my apologies": 1.0,
    # 强烈赞许/钦佩（1.5）
    "太感谢": 1.5,
    "非常感谢": 1.5,
    "佩服": 1.5,
    "钦佩": 1.5,
    "excellent": 1.5,
    "amazing": 1.5,
    "well done": 1.5,
}

# 信号强度上限（防堆词刷分），非 OCC praiseworthiness 理论值（Ortony et al. 1988 Ch.4）
_MAX_SIGNALS = 4

# 自指上下文窗口宽度（命中词前的字符数）：覆盖「我太蠢了」「自己真笨」等短句
_SELF_REF_WINDOW = 5


def _is_self_referential(text: str, word: str) -> bool:
    """命中 violation 词前 ~_SELF_REF_WINDOW 字窗口内含自指标记 → 判自指，不计为攻击。

    避免把「我太蠢了」（OCC shame/self-reproach）误判为对他人 reproach/攻击。
    纯正则/字符串扫描，确定性，无外部依赖。

    自指标记：「我」「自己」「本人」或正则「我.*?(蠢|笨|白痴|废物)」跨词匹配。
    """
    import re  # stdlib，仅本函数用，lazy import 避免模块级副作用

    pos = text.find(word)
    if pos < 0:
        return False
    # 窗口：命中词前最多 _SELF_REF_WINDOW 字
    window_start = max(0, pos - _SELF_REF_WINDOW)
    window = text[window_start:pos]
    if any(marker in window for marker in ("我", "自己", "本人")):
        return True
    # 宽松跨词匹配：「我……蠢/笨/白痴/废物」句式
    if re.search(r"我.{0,10}(蠢|笨|白痴|废物)", text[: pos + len(word)]):
        return True
    return False


def appraise_standard_compliance(text: str) -> float:
    """从用户文本确定性估计社会规范遵从度，返回 ∈[-1, 1]。

    规范违反（辱骂/命令式威胁/贬低/脏话）→ 负；
    规范遵从（致谢/道歉/礼貌/赞许）→ 正；
    无信号 → 0.0。

    算法（议会 2026-07-02 精化）：
    1. 自指过滤：violation 词命中且上下文含自指（我/自己/本人）→ 不计分
       （防止「我太蠢了」OCC shame 被误判为对他人 reproach）。
    2. 三级加权求和：v_raw = Σ weight(violation 命中且非自指)；
                      c_raw = Σ weight(compliance 命中)。
    3. 钳制归一：v_score = min(v_raw, _MAX_SIGNALS) / _MAX_SIGNALS ∈ [0,1]；同理 c_score。
    4. net = clamp(c_score - v_score, -1, 1)。

    纯词典/规则，确定性，torch/LLM-free（守红线）。
    """
    lower = text.lower()
    v_raw = sum(
        w
        for word, w in _VIOLATION_WEIGHTS.items()
        if word in lower and not _is_self_referential(lower, word)
    )
    c_raw = sum(w for word, w in _COMPLIANCE_WEIGHTS.items() if word in lower)
    v_score = min(v_raw, _MAX_SIGNALS) / _MAX_SIGNALS  # [0, 1]
    c_score = min(c_raw, _MAX_SIGNALS) / _MAX_SIGNALS  # [0, 1]
    net = c_score - v_score  # ∈ [-1, 1]
    return float(clamp(net, -1.0, 1.0))


def intensity_envelope(
    position: int,
    length: int,
    *,
    floor: float = 0.0,
    steepness: float = ENVELOPE_STEEPNESS,
) -> float:
    """一句话内情绪强度的时间包络（ECM internal memory：句首满、句尾衰减到 floor）。

    返回 [floor, 1] 的标量：position=0 → 1，position=length-1 → floor，单调递减（sigmoid 形）。
    length<=1 时返回 1.0。纯函数、确定性、有界。
    """
    if length <= 1:
        return 1.0
    frac = clamp(position / (length - 1), 0.0, 1.0)  # 0..1
    hi = _sigmoid(0.5 * steepness)
    lo = _sigmoid(-0.5 * steepness)
    raw = _sigmoid(steepness * (0.5 - frac))
    env01 = (raw - lo) / (hi - lo)  # 归一到 [0,1]，frac=0→1、frac=1→0
    return clamp(floor + (1.0 - floor) * env01, floor, 1.0)


def _sigmoid(x: float) -> float:
    """数值稳定 logistic（本模块自用，避免反向依赖 affect_math.sigmoid 的语义耦合）。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)
