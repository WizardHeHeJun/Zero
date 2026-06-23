"""数学内核直接单测：先验/TD/精度/高斯融合/采样的取值与边界钳制。"""

from __future__ import annotations

import random

from src.agents.affect_math import (
    MAX_SAMPLE_SIGMA,
    MIN_PRECISION,
    MIN_SIGMA,
    evidence_from_value,
    gaussian_fuse,
    occ_prior,
    precision,
    sample_affect,
    td_update,
)


def test_occ_prior_clamped_and_reward_is_goal_congruence() -> None:
    (mu_v, mu_a), (sig_v, sig_a), reward = occ_prior(2.0, 2.0, 2.0, 2.0)  # 越界输入
    assert -1.0 <= mu_v <= 1.0
    assert -1.0 <= mu_a <= 1.0
    assert sig_v >= MIN_SIGMA and sig_a >= MIN_SIGMA
    assert reward == 1.0  # goal_congruence 钳制到 [-1,1]


def test_td_update_does_not_mutate_input_and_moves_toward_reward() -> None:
    table = {"s": 0.0}
    delta, new_value, updated = td_update(table, "s", 1.0)
    assert table == {"s": 0.0}  # 入参未被原地修改
    assert delta == 1.0
    assert new_value > 0.0  # 朝奖励方向移动
    assert updated["s"] == new_value


def test_precision_floor_and_monotonic_in_delta() -> None:
    assert precision(0.0, 0.0) >= MIN_PRECISION
    assert precision(2.0, 0.0) > precision(0.1, 0.0)  # |δ| 越大精度越高


def test_gaussian_fuse_bounds_and_pull_toward_evidence() -> None:
    prior_mu = (0.0, 0.0)
    prior_sigma = (0.2, 0.2)
    evidence = evidence_from_value(1.0, 1.0)  # (1.0, 1.0)
    low = gaussian_fuse(prior_mu, prior_sigma, evidence, pi=0.1)
    high = gaussian_fuse(prior_mu, prior_sigma, evidence, pi=10.0)
    # 精度越高，后验越被证据拉离先验
    assert high[0][0] > low[0][0]
    for sig in (*low[1], *high[1]):
        assert MIN_SIGMA <= sig <= MAX_SAMPLE_SIGMA
    for mu in (*low[0], *high[0]):
        assert -1.0 <= mu <= 1.0


def test_sample_affect_seeded_is_reproducible_and_bounded() -> None:
    post_mu = (0.5, 0.5)
    post_sigma = (0.3, 0.3)
    a = sample_affect(post_mu, post_sigma, rng=random.Random(42))
    b = sample_affect(post_mu, post_sigma, rng=random.Random(42))
    assert a == b  # 同种子可复现
    assert -1.0 <= a[0] <= 1.0 and -1.0 <= a[1] <= 1.0
