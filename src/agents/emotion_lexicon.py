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


def motivational_system(valence: float, arousal: float) -> str:
    """VA → Panksepp 主导情绪系统标签（动机色彩）。

    映射（r<NEUTRAL_RADIUS 为 neutral）：
      (+v,+a)→seeking（探索/渴求/玩乐） (+v,-a)→care（照护/亲和/满足）
      (-v,+a)→rage（愤怒/威胁；FEAR 同处该象限，VA 欠定，取 rage 为主导近似）
      (-v,-a)→panic_grief（失落/悲伤/分离）
    返回小写枚举串，便于下游条件化与单测。
    """
    r, _ = _polar(valence, arousal)
    if r < NEUTRAL_RADIUS:
        return "neutral"
    if valence >= 0.0:
        return "seeking" if arousal >= 0.0 else "care"
    return "rage" if arousal >= 0.0 else "panic_grief"


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
    "惊讶": (0.1, 0.7),
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
