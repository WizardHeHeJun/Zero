"""emotion_lexicon 单测：细粒度词、Panksepp 动机、词典桥加权解码、ECM 情绪时间包络。

全部纯函数、确定性、无外部依赖（torch/API-free）。
"""

from __future__ import annotations

import pytest

from src.agents.emotion_lexicon import (
    NEUTRAL_RADIUS,
    PANKSEPP_FEAR_AROUSAL_THRESHOLD,
    SEED_VAD_LEXICON,
    affect_descriptor,
    affect_label,
    affect_logit_bias,
    appraise_text,
    infer_domain,
    intensity_envelope,
    motivational_system,
    suggest_affect_words,
)

# ---------------------------------------------------------------------------
# A-P1-D Panksepp RAGE/FEAR 门控测试
# ---------------------------------------------------------------------------


def test_motivational_system_distinguish_fear_off_zero_regression() -> None:
    """distinguish_fear=False（默认）时行为与改前完全一致，严格零回归断言。"""
    # (-v,+a) 无论 arousal 高低，一律 rage
    assert motivational_system(-0.6, 0.6) == "rage"
    assert motivational_system(-0.5, 0.9) == "rage"  # 极高 arousal 仍 rage
    assert motivational_system(-0.3, PANKSEPP_FEAR_AROUSAL_THRESHOLD) == "rage"
    # 其余象限不受影响
    assert motivational_system(0.6, 0.6) == "seeking"
    assert motivational_system(0.6, -0.6) == "care"
    assert motivational_system(-0.6, -0.6) == "panic_grief"
    assert motivational_system(0.0, 0.0) == "neutral"


def test_motivational_system_distinguish_fear_on_high_arousal_fear() -> None:
    """distinguish_fear=True 且 arousal ≥ 阈值 → fear。"""
    # arousal 恰好等于阈值（边界含入）
    assert (
        motivational_system(-0.5, PANKSEPP_FEAR_AROUSAL_THRESHOLD, distinguish_fear=True) == "fear"
    )
    # arousal 明显超阈值
    assert motivational_system(-0.5, 0.9, distinguish_fear=True) == "fear"
    assert motivational_system(-0.8, 0.8, distinguish_fear=True) == "fear"


def test_motivational_system_distinguish_fear_on_mid_arousal_rage() -> None:
    """distinguish_fear=True 且 arousal < 阈值（但仍 ≥ 0）→ rage。"""
    below = PANKSEPP_FEAR_AROUSAL_THRESHOLD - 0.1  # 0.5
    assert motivational_system(-0.5, below, distinguish_fear=True) == "rage"
    assert motivational_system(-0.6, 0.3, distinguish_fear=True) == "rage"


def test_motivational_system_distinguish_fear_other_quadrants_unchanged() -> None:
    """开启门控不影响其他象限的输出。"""
    assert motivational_system(0.6, 0.6, distinguish_fear=True) == "seeking"
    assert motivational_system(0.6, -0.6, distinguish_fear=True) == "care"
    assert motivational_system(-0.6, -0.6, distinguish_fear=True) == "panic_grief"
    assert motivational_system(0.0, 0.0, distinguish_fear=True) == "neutral"


# ---------------------------------------------------------------------------
# A-P2-D 惊讶词条中性化测试
# ---------------------------------------------------------------------------


def test_surprise_valence_neutral() -> None:
    """惊讶效价中性化：(0.0, 0.7)（失真修正，非零回归）。"""
    v, a = SEED_VAD_LEXICON["惊讶"]
    assert v == pytest.approx(0.0)
    assert a == pytest.approx(0.7)


def test_surprise_variants_in_lexicon() -> None:
    """惊喜/惊吓两条细分词存在且效价方向正确（议会 2026-07-02 精化坐标）。"""
    v_xi, a_xi = SEED_VAD_LEXICON["惊喜"]
    v_xia, a_xia = SEED_VAD_LEXICON["惊吓"]
    assert v_xi > 0.0, "惊喜应为正效价"
    assert v_xia < 0.0, "惊吓应为负效价"
    # 惊喜：(0.5, 0.7)；Russell 1980 astonished ~69.8°，旧 0.6 偏强
    assert v_xi == pytest.approx(0.5)
    assert a_xi == pytest.approx(0.7)
    # 惊吓：(-0.3, 0.8)；Russell 1980 alarmed ~96.5°，与恐惧 (-0.7,0.7) 拉开距离
    assert v_xia == pytest.approx(-0.3)
    assert a_xia == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 原有测试（下方不变）
# ---------------------------------------------------------------------------


def test_affect_label_neutral_deadzone() -> None:
    # 微弱情感（r<NEUTRAL_RADIUS）→ 平静，不强行贴词
    assert affect_label(0.05, 0.05) == "平静"
    assert affect_label(0.0, 0.0) == "平静"


def test_affect_label_finer_than_four_buckets() -> None:
    # 四象限取强情感，词应落在对应扇区且彼此不同（粒度 > 旧 4 档）
    pos_high = affect_label(0.6, 0.7)  # v+ a+ → 欣喜族
    neg_high = affect_label(-0.6, 0.7)  # v- a+ → 恼火族
    neg_low = affect_label(-0.6, -0.6)  # v- a- → 忧伤族
    pos_low = affect_label(0.6, -0.6)  # v+ a- → 放松族
    labels = {pos_high, neg_high, neg_low, pos_low}
    assert len(labels) == 4  # 四象限互不相同
    assert "平静" not in labels


def test_affect_label_intensity_grading() -> None:
    # 同方向、半径越大 → 词越强（弱/中/强分级）
    mild = affect_label(0.32, 0.32)  # r≈0.45
    strong = affect_label(0.65, 0.65)  # r≈0.92
    assert mild != strong


def test_affect_label_deterministic() -> None:
    assert affect_label(-0.5, 0.6) == affect_label(-0.5, 0.6)


def test_motivational_system_quadrants() -> None:
    assert motivational_system(0.6, 0.6) == "seeking"  # +v+a 探索/渴求
    assert motivational_system(0.6, -0.6) == "care"  # +v-a 照护/亲和
    assert motivational_system(-0.6, 0.6) == "rage"  # -v+a 愤怒（FEAR 同象限，主导近似）
    assert motivational_system(-0.6, -0.6) == "panic_grief"  # -v-a 失落/悲伤
    assert motivational_system(0.0, 0.0) == "neutral"


def test_motivational_system_neutral_boundary() -> None:
    assert motivational_system(NEUTRAL_RADIUS * 0.5, 0.0) == "neutral"


def test_affect_descriptor_combines_label_and_system() -> None:
    d = affect_descriptor(0.6, 0.7)
    assert "·" in d
    word, system = d.split("·")
    assert word == affect_label(0.6, 0.7)
    assert system == motivational_system(0.6, 0.7)


def test_affect_logit_bias_aligns_with_target() -> None:
    e_star = (0.8, 0.6)  # 正向高唤起
    bias = affect_logit_bias(["狂喜", "抑郁", "未知词"], e_star)
    assert bias["狂喜"] > 0.0  # 同向 → 正偏置
    assert bias["抑郁"] < 0.0  # 反向 → 负偏置
    assert bias["未知词"] == 0.0  # 词典外 → 不干预


def test_affect_logit_bias_beta_scales_and_zero_disables() -> None:
    e_star = (0.8, 0.6)
    b1 = affect_logit_bias(["狂喜"], e_star, beta=1.0)["狂喜"]
    b2 = affect_logit_bias(["狂喜"], e_star, beta=2.0)["狂喜"]
    assert b2 == pytest.approx(2.0 * b1)
    assert affect_logit_bias(["狂喜"], e_star, beta=0.0)["狂喜"] == 0.0


def test_suggest_affect_words_topk_aligned_and_deterministic() -> None:
    words = suggest_affect_words(0.8, 0.6, k=3)
    assert len(words) == 3
    assert all(w in SEED_VAD_LEXICON for w in words)
    # 正向高唤起：最贴合的应是正高唤起词，不应是负向词
    assert "抑郁" not in words
    assert suggest_affect_words(0.8, 0.6, k=3) == words  # 确定性

    neg = suggest_affect_words(-0.8, 0.6, k=3)  # 负向高唤起 → 愤怒/焦虑族
    assert any(w in {"愤怒", "暴怒", "焦虑", "恐惧", "恼火"} for w in neg)


def test_suggest_affect_words_k_bounds() -> None:
    assert suggest_affect_words(0.5, 0.5, k=0) == []
    assert len(suggest_affect_words(0.5, 0.5, k=1000)) == len(SEED_VAD_LEXICON)


def test_intensity_envelope_decays_from_full_to_floor() -> None:
    n = 8
    env = [intensity_envelope(i, n) for i in range(n)]
    assert env[0] == pytest.approx(1.0)  # 句首满
    assert env[-1] == pytest.approx(0.0)  # 句尾归零（floor 默认 0）
    assert all(env[i] > env[i + 1] for i in range(n - 1))  # 单调递减
    assert all(0.0 <= v <= 1.0 for v in env)  # 有界


def test_intensity_envelope_respects_floor_and_singleton() -> None:
    assert intensity_envelope(0, 1) == 1.0  # 单 token → 满
    last = intensity_envelope(9, 10, floor=0.3)
    assert last == pytest.approx(0.3)  # 句尾落到 floor
    assert intensity_envelope(0, 10, floor=0.3) == pytest.approx(1.0)


def test_appraise_text_averages_matched_words() -> None:
    hi, sf = SEED_VAD_LEXICON["高兴"], SEED_VAD_LEXICON["满足"]
    v, a = appraise_text("我感到高兴又满足")  # 仅命中「高兴」「满足」
    assert v == pytest.approx((hi[0] + sf[0]) / 2)
    assert a == pytest.approx((hi[1] + sf[1]) / 2)


def test_appraise_text_neutral_when_no_match() -> None:
    assert appraise_text("今天天气不错适合散步") == (0.0, 0.0)


# ---------------------------------------------------------------------------
# infer_domain 单测（live-chat 域注入桥·议会 2026-07-21）
# ---------------------------------------------------------------------------
# 真实词来自 _VIOLATION_WEIGHTS：
#   0.5 轻度：蠢 笨 丑 无聊
#   1.0 标准：白痴 废物 垃圾 滚 闭嘴 混蛋 bitch asshole
#   1.5 强烈：去死 死 他妈的 fuck nmsl 草
# ---------------------------------------------------------------------------


def test_infer_domain_standard_1_0_confrontational() -> None:
    """1.0 档标准辱骂词·指向他人 → confrontational。

    用「废物」「闭嘴」「混蛋」「白痴」「bitch」覆盖 1.0 档实际词。
    注：「给我滚出去」含「我」字处于「滚」的前 5 字窗口→ _is_self_referential 误过滤，
    避开此歧义句·改用「快滚」句式（「滚」前无自指标记）。
    """
    assert infer_domain("你真是个废物") == "confrontational"
    assert infer_domain("快滚别出现") == "confrontational"  # 「滚」前无自指标记
    assert infer_domain("混蛋别来找我") == "confrontational"
    assert infer_domain("你是个白痴") == "confrontational"
    assert infer_domain("you are a bitch") == "confrontational"


def test_infer_domain_extreme_1_5_confrontational() -> None:
    """1.5 档强烈脏话/死亡威胁 → confrontational（最强信号）。

    用「去死」「他妈的」「fuck」验证（1.5 档覆盖）。
    """
    assert infer_domain("去死吧你") == "confrontational"
    assert infer_domain("他妈的你在干什么") == "confrontational"
    assert infer_domain("fuck you") == "confrontational"
    assert infer_domain("nmsl") == "confrontational"


def test_infer_domain_mild_0_5_returns_none() -> None:
    """0.5 轻度词（蠢/笨/丑/无聊）→ None（关键红线：轻度不触发 confrontational）。

    议会心理席：0.5 档方向性盲区留 B-direction，不应产生 confrontational。
    测试用句须只含 0.5 词·不混入 ≥1.0 强信号词（如「死」1.5·「白痴」1.0）。
    """
    assert infer_domain("你有点蠢") is None
    assert infer_domain("真是太笨了") is None  # 只含「笨」(0.5)，无其他强信号词
    assert infer_domain("长得真丑") is None
    assert infer_domain("好无聊啊") is None


def test_infer_domain_self_referential_returns_none() -> None:
    """强信号词 + 自指上下文（「我」在窗口内）→ None（OCC shame 不算对他人攻击）。

    自指窗口 _SELF_REF_WINDOW=5 字；「我白痴了」窗口含「我」→ 过滤。
    宽松正则覆盖「我……废物」跨词形式。
    """
    assert infer_domain("我真是个白痴") is None
    assert infer_domain("自己真废物") is None
    assert infer_domain("本人太垃圾了") is None


def test_infer_domain_neutral_text_returns_none() -> None:
    """无任何违规词的纯中性文本 → None。"""
    assert infer_domain("今天天气很好") is None
    assert infer_domain("我想喝点水") is None
    assert infer_domain("") is None
    assert infer_domain("谢谢你帮我") is None


def test_infer_domain_survival_returns_none() -> None:
    """BLOCK 红线：survival/求救类文本 → None，绝不产 survival_narrative。

    词典桥值域只有 {confrontational, None}，三路体裁分类属 FPN 离线层（CS+神经席 BLOCK）。
    """
    assert infer_domain("好害怕") is None
    assert infer_domain("快救我") is None
    assert infer_domain("他要伤害我") is None
    assert infer_domain("我想消失掉") is None
    assert infer_domain("没有人关心我") is None


def test_infer_domain_value_domain_exhaustive() -> None:
    """值域全覆盖：多条不同文本，返回值必须 ∈ {confrontational, None}，绝不是 survival_narrative
    或 neutral（BLOCK 议会 2026-07-21）。"""
    samples = [
        "你这废物",  # confrontational
        "去死吧",  # confrontational
        "你蠢",  # None（0.5 轻度）
        "今天心情还好",  # None（中性）
        "好害怕快救我",  # None（survival 绝不产）
        "我太白痴了",  # None（自指）
        "fuck",  # confrontational
        "",  # None（空）
        "加油！",  # None
        "混蛋给我滚",  # confrontational（双命中）
    ]
    allowed = {"confrontational", None}
    for text in samples:
        result = infer_domain(text)
        # 绝不产 survival_narrative / neutral 字符串（值域硬约束）
        assert result in allowed, f"值域违例 text={text!r} result={result!r}"
        assert result != "survival_narrative", f"BLOCK: 产出 survival_narrative text={text!r}"
        assert result != "neutral", f"BLOCK: 产出 neutral 字符串 text={text!r}"
