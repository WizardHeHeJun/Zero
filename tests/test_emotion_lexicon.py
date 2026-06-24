"""emotion_lexicon 单测：细粒度词、Panksepp 动机、词典桥加权解码、ECM 情绪时间包络。

全部纯函数、确定性、无外部依赖（torch/API-free）。
"""

from __future__ import annotations

import pytest

from src.agents.emotion_lexicon import (
    NEUTRAL_RADIUS,
    SEED_VAD_LEXICON,
    affect_descriptor,
    affect_label,
    affect_logit_bias,
    appraise_text,
    intensity_envelope,
    motivational_system,
    suggest_affect_words,
)


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
