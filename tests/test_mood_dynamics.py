"""Mood 双稳动力学（A.7）单测：双稳、滞后捕获、有界、多项融合等价。"""

from __future__ import annotations

from src.agents.affect_math import (
    MAX_SAMPLE_SIGMA,
    MIN_SIGMA,
    fuse_terms,
    gaussian_fuse,
    mood_step,
)


def _settle(
    initial: tuple[float, float], drive: tuple[float, float], steps: int
) -> tuple[float, float]:
    m = initial
    for _ in range(steps):
        m = mood_step(m, drive)
    return m


def test_mood_bistable_two_basins() -> None:
    # 零 drive：小正初值收敛到正吸引子、小负初值收敛到负吸引子（pitchfork 双稳）
    pos = _settle((0.1, 0.0), (0.0, 0.0), 200)
    neg = _settle((-0.1, 0.0), (0.0, 0.0), 200)
    assert pos[0] > 0.3
    assert neg[0] < -0.3


def test_mood_hysteresis_capture() -> None:
    # 持续负向把心境推入负盆
    stuck = _settle((0.0, 0.0), (-0.8, 0.0), 30)
    assert stuck[0] < -0.3
    # 之后给轻微正向，仍困在负盆（回不去）
    still_neg = _settle(stuck, (0.2, 0.0), 30)
    assert still_neg[0] < 0.0
    # 对照：从中性出发同样轻微正向 → 能进正盆，证明是历史决定了走向
    fresh = _settle((0.0, 0.0), (0.2, 0.0), 30)
    assert fresh[0] > still_neg[0]


def test_mood_bounded_under_extremes() -> None:
    m = _settle((2.0, -2.0), (2.0, -2.0), 50)
    assert -1.0 <= m[0] <= 1.0
    assert -1.0 <= m[1] <= 1.0


def test_fuse_terms_matches_gaussian_fuse_two_terms() -> None:
    prior_mu = (0.2, -0.1)
    prior_sigma = (0.2, 0.3)
    evidence = (0.8, 0.6)
    pi = 2.0
    g_mu, g_sigma = gaussian_fuse(prior_mu, prior_sigma, evidence, pi)
    prior_prec = (1.0 / prior_sigma[0] ** 2, 1.0 / prior_sigma[1] ** 2)
    f_mu, f_sigma = fuse_terms([(prior_mu, prior_prec), (evidence, (pi, pi))])
    assert abs(f_mu[0] - g_mu[0]) < 1e-9
    assert abs(f_mu[1] - g_mu[1]) < 1e-9
    assert abs(f_sigma[0] - g_sigma[0]) < 1e-9
    assert abs(f_sigma[1] - g_sigma[1]) < 1e-9
    for s in (*f_sigma, *g_sigma):
        assert MIN_SIGMA <= s <= MAX_SAMPLE_SIGMA
