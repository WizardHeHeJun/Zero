"""硬门摘出数值通路（议会第三轮 D1，五轮评审）：零回归 + BLOCK + 锚点 A/B/C。

**核心改动**：`ignite()` 的硬门原先同时决定「谁进 `fuse_terms`」与「谁可报告」。
神经席裁定 GNW ignition = 「什么内容变得**可报告**」，不是「谁计算数值」；且
**阈下不点燃 ≠ 阈下零影响**。`gate_fusion=False` 时数值后验走**全流原生 (μ, Π)**，
「哪些流点燃」改由 `report_ignited()` 单独回答。

⚠ **本门方向与仓内其它旋钮相反**：`gate_fusion` 默认 **True = 门关 = 逐字旧行为**。
漏接线会使它永久 True——**新架构永远开不出来**（不是永远开着）。用例按此方向写。

分区：
1. 零回归（门关）：`ignite` 与 `report_ignited` 逐值相等 + 三条既有分支不变。
2. BLOCK 1：两个返回值恒对齐 → `zip(strict=True)` 永不失配。
3. BLOCK 2：`gate_fusion=False` × HPC 在 **SessionConfig 构造期** fail-fast。
4. D12：空 `fusion_terms` 前置条件 + `fuse_terms` 空输入。
5. D7：physio 排除（跨仓承诺）。
6. 锚点 A/B/C + **各自的变异测试**（先证正确实现恒绿，再证接回历史 bug 会红）。
"""

from __future__ import annotations

import itertools
import math

import pytest

from src.agents.affect_core import AffectCoreAgent
from src.agents.affect_math import (
    MIN_PRECISION,
    SALIENCE_THRESHOLD,
    fast_survival_prior,
    fuse_terms,
    ignite,
    report_ignited,
    stream_salience,
)
from src.orchestration.runner import SessionConfig
from src.orchestration.state import AffectState, Stimulus

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

Stream = tuple[str, tuple[float, float], tuple[float, float]]


def _streams(*specs: tuple[str, float, float, float, float]) -> list[Stream]:
    return [(n, (mv, ma), (pv, pa)) for n, mv, ma, pv, pa in specs]


_CORE: list[Stream] = _streams(
    ("survival", 0.30, 0.60, 0.40, 0.40),
    ("appraisal", 0.15, 0.09, 11.11, 11.11),
    ("value", 0.00, 0.30, 0.001, 0.62),
)


def _weighted_ref(terms: list[tuple[tuple[float, float], tuple[float, float]]], d: int) -> float:
    """锚点 B 的参考量：`Σ p_i μ_i / Σ p_i`，**`p_i = max(MIN_PRECISION, Π_i)`**。

    🛑 **必须用有效精度而非原生 Π**：`fuse_terms` 内部就是 `p = max(MIN_PRECISION, prec[d])`。
    用原生 Π 当参考量，在验收要求的采样域（Π 对数均匀跨 [1e-9, cap]）下**10 万组里
    84427 组正确实现会假红**，最坏偏差 2.885e-04（比 1e-9 容差高 5 个数量级）。
    第四轮就栽在这里。
    """
    ps = [max(MIN_PRECISION, prec[d]) for _, prec in terms]
    return sum(p * mu[d] for (mu, _), p in zip(terms, ps, strict=True)) / sum(ps)


# ---------------------------------------------------------------------------
# 1. 零回归（门关 = 默认）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("soft_beta", [None, 20.0])
@pytest.mark.parametrize("survival_fallback", [False, True])
def test_gate_off_ignite_and_report_agree_exactly(
    soft_beta: float | None, survival_fallback: bool
) -> None:
    """门关：`ignite()` 的流名与 `report_ignited()` **逐值相等**。

    这是零回归的必要条件——`affect_core` 用前者做 zip、用后者填 `ignited_streams`，
    二者不等就意味着 `out["ignited_streams"]` 变了。覆盖软门/硬门 × 兜底开关四种组合。
    """
    _, fusion_names = ignite(
        _CORE, survival_fallback=survival_fallback, soft_beta=soft_beta, gate_fusion=True
    )
    ignited = report_ignited(_CORE, survival_fallback=survival_fallback, soft_beta=soft_beta)
    assert fusion_names == ignited


def test_gate_off_all_weak_fallback_still_agrees() -> None:
    """门关 · 全弱刺激（无流过阈）：兜底分支下二者仍逐值相等。"""
    weak = _streams(("survival", 0.01, 0.01, 0.01, 0.01), ("appraisal", 0.02, 0.01, 0.02, 0.02))
    assert all(stream_salience(mu, p) < SALIENCE_THRESHOLD for _, mu, p in weak)
    for fb in (False, True):
        _, names = ignite(weak, survival_fallback=fb, soft_beta=None, gate_fusion=True)
        assert names == report_ignited(weak, survival_fallback=fb, soft_beta=None)
        assert len(names) == 1, "兜底应只保留一条"


def test_gate_off_default_is_true_not_false() -> None:
    """⚠ 方向锁定：不传 `gate_fusion` == 传 `True`（**不是** False）。

    本门方向与仓内其它旗标相反，最容易写反的就是这一条。
    """
    assert ignite(_CORE) == ignite(_CORE, gate_fusion=True)
    assert SessionConfig().gate_fusion is True
    assert AffectState().gate_fusion is True
    assert SessionConfig().to_state_flags()["gate_fusion"] is True


def _core_state(**kw: object) -> AffectState:
    base: dict = {
        "stimulus": Stimulus(name="s", goal_congruence=0.4, intensity=0.6),
        "prior_mu": (0.2, 0.3),
        "prior_sigma": (0.25, 0.25),
        "reward": 0.4,
        "rpe": 0.35,
        "precision": 0.6,
        "workspace_enabled": True,
        "rng_seed": 20260729,
        "affect_readout": "map",
    }
    base.update(kw)
    return AffectState(**base)


def test_gate_off_affect_core_unchanged() -> None:
    """门关：`AffectCoreAgent` 的后验与 `ignited_streams` 与显式关一致。"""
    agent = AffectCoreAgent()
    a = agent(_core_state())
    b = agent(_core_state(gate_fusion=True))
    assert a["post_mu"] == b["post_mu"]
    assert a["ignited_streams"] == b["ignited_streams"]


def test_gate_on_actually_changes_posterior() -> None:
    """门开：后验确实变了——防漏接线导致静默 no-op（BLOCK-1 同款防护）。"""
    agent = AffectCoreAgent()
    off = agent(_core_state())
    on = agent(_core_state(gate_fusion=False))
    assert off["post_mu"] != on["post_mu"], "门开后 post_mu 未变化，接线可能没通"


# ---------------------------------------------------------------------------
# 2. BLOCK 1 —— 两个返回值恒对齐
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gate_fusion", [True, False])
@pytest.mark.parametrize("soft_beta", [None, 20.0])
def test_block1_fusion_terms_and_names_always_aligned(
    gate_fusion: bool, soft_beta: float | None
) -> None:
    """BLOCK 1 的实质保证：`ignite()` 两个返回值**恒等长**，`zip(strict=True)` 永不失配。

    第四轮方案是把 `ignite()` 改三元组；实现期改判为**拆两个函数**（design D13）——
    对齐由「同一次筛选产出」在构造上保证，不需要第三个返回值，
    且 24 个既有调用点的返回元数一个不动。
    """
    terms, names = ignite(_CORE, soft_beta=soft_beta, gate_fusion=gate_fusion)
    assert len(terms) == len(names)
    list(zip(names, terms, strict=True))  # 不抛即通过


def test_block1_report_subset_would_have_broken_the_old_zip() -> None:
    """反向证明 BLOCK 1 真实存在：门开时报告集**真的**比融合集短。

    若 `affect_core` 仍用 `zip(ignited, terms, strict=True)`，这里就会 ValueError。
    """
    terms, names = ignite(_CORE, soft_beta=None, gate_fusion=False)
    ignited = report_ignited(_CORE, soft_beta=None)
    assert len(ignited) < len(names), "该场景下报告集应真子集于融合集，否则本用例没在测东西"
    with pytest.raises(ValueError):
        list(zip(ignited, terms, strict=True))


# ---------------------------------------------------------------------------
# 3. BLOCK 2 —— 与 HPC 显式互斥，在 SessionConfig 构造期 fail-fast
# ---------------------------------------------------------------------------


def test_block2_gate_fusion_x_hpc_rejected_at_config_time() -> None:
    """`gate_fusion=False` × `layers≥2` × `coupling>0` → 构造期抛错。

    **位置很关键**（第五轮订正）：两个纯函数各自都拿不到三字段全集；且若在热路径抛，
    MCP 面就是永久 DoS（活跃会话 config 不可变，client 无法自救），
    而触发组合 `coupling∈[0.3,0.8]` 恰是 README 推荐值、不是坏输入。
    """
    with pytest.raises(ValueError, match="联合语义未定义"):
        SessionConfig(gate_fusion=False, hierarchical_layers=2, hierarchical_coupling=0.5)


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({"gate_fusion": False, "hierarchical_layers": 1}, id="HPC-层数退化"),
        pytest.param(
            {"gate_fusion": False, "hierarchical_layers": 2, "hierarchical_coupling": 0.0},
            id="HPC-耦合退化",
        ),
        pytest.param(
            {"gate_fusion": True, "hierarchical_layers": 2, "hierarchical_coupling": 0.5},
            id="门关+HPC（旧路径须照常可用）",
        ),
    ],
)
def test_block2_does_not_over_reject(kw: dict) -> None:
    """互斥只收缩**两个开关的乘积空间**，不得误伤任一单独开启的路径。"""
    assert SessionConfig(**kw) is not None


# ---------------------------------------------------------------------------
# 4. D12 —— 空 fusion_terms 的前置条件
# ---------------------------------------------------------------------------


def test_d12_all_physio_streams_raise_named_error() -> None:
    """传入的流全是被排除流 → 指名道姓的报错，不是 `ZeroDivisionError`。"""
    only_physio = _streams(("eda_tonic", 0.0, 0.8, 0.001, 0.175))
    with pytest.raises(ValueError, match="全部命中 physio 排除前缀"):
        ignite(only_physio, gate_fusion=False, exclude_physio_fusion=True)
    # 显式关掉排除即可正常返回（逃生舱有效）
    terms, _ = ignite(only_physio, gate_fusion=False, exclude_physio_fusion=False)
    assert len(terms) == 1


def test_d12_fuse_terms_empty_raises_informative_error() -> None:
    """`fuse_terms([])` 从 `ZeroDivisionError` 改成指名道姓的 `ValueError`。"""
    with pytest.raises(ValueError, match="收到空的 terms"):
        fuse_terms([])
    # **非空输入逐位不变**——纯错误信息改善，非行为变更
    got = fuse_terms([((0.2, 0.3), (1.0, 2.0)), ((0.4, 0.1), (3.0, 1.0))])
    assert got[0][0] == pytest.approx((1.0 * 0.2 + 3.0 * 0.4) / 4.0)


def test_d12_empty_input_is_not_a_precondition_violation() -> None:
    """`ignite([])` 本身不报错——空输入是空输出，不是「全被排除」。"""
    assert ignite([], gate_fusion=False) == ([], [])


# ---------------------------------------------------------------------------
# 5. D7 —— physio 排除（跨仓承诺）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["physio", "eda_tonic", "hrv_rmssd", "pupil_d", "scr_amp"])
def test_d7_physio_excluded_from_fusion_by_default(name: str) -> None:
    """五个 physio 前缀在门开时默认**不进数值通路**。

    Zero_MCP 用 WESAD 真被试验证其 EDA arousal 与唤醒**系统性反号**，明确请求
    「宁可继续门掉——『暂时不参与融合』优于『以反号参与』」。由我方单边可控。
    """
    with_physio = [*_CORE, (name, (0.0, 0.85), (MIN_PRECISION, 0.175))]
    _, names = ignite(with_physio, gate_fusion=False)
    assert name not in names
    _, names_off = ignite(with_physio, gate_fusion=False, exclude_physio_fusion=False)
    assert name in names_off


def test_d7_physio_still_reportable_even_when_excluded_from_fusion() -> None:
    """**反向语义须被验收**：physio 可以「点燃（可报告）」却**不参与数值**。

    这不是 bug，是新架构的直接推论——报告通路与数值通路分离。
    但该列表经 supervisor 落进 USER 作用域 episode 并注入 LLM 上下文，
    故这个反直觉组合必须被显式测到、而不是等人在生产里读到才发现。
    """
    strong_physio = [*_CORE, ("physio", (0.0, 0.95), (MIN_PRECISION, 5.0))]
    _, fusion_names = ignite(strong_physio, gate_fusion=False, soft_beta=None)
    ignited = report_ignited(strong_physio, soft_beta=None)
    assert "physio" in ignited, "高显著 physio 应可报告"
    assert "physio" not in fusion_names, "但不得进数值通路"


# ---------------------------------------------------------------------------
# 6. 锚点 A/B/C + 变异测试
#
# 🛑 每条锚点两件事都要做到：(a) 接回历史 bug 会红；(b) **先证明在正确实现上恒绿**。
# (b) 比 (a) 更要紧——第四轮的①④各自在正确实现上也会红，而
# **「在正确实现上会红的断言」比「不可证伪的绿灯」更危险**：它会在落地时被调松，
# 而调松方向恰好把判别力一起调没。
# ---------------------------------------------------------------------------

_GRID = [-1.0, -0.4, 0.0, 0.4, 1.0]


def _random_streams(seed: int) -> list[Stream]:
    import random

    rnd = random.Random(seed)
    n = rnd.randint(2, 5)
    return [
        (
            f"s{i}",
            (rnd.uniform(-1, 1), rnd.uniform(-1, 1)),
            (10 ** rnd.uniform(-9, 0.5), 10 ** rnd.uniform(-9, 0.5)),
        )
        for i in range(n)
    ]


# ── 锚点 A：后验落在各流 μ 的凸包内（退化轴保护）──


def _assert_anchor_a(
    terms: list[tuple[tuple[float, float], tuple[float, float]]],
    post: tuple[float, float] | None = None,
) -> None:
    """`post` 可注入——供变异测试**真的驱动本断言变红**，而不是在旁边算个恒真的不等式。"""
    if post is None:
        post, _ = fuse_terms(terms)
    for d in (0, 1):
        mus = [mu[d] for mu, _ in terms]
        lo, hi = min(mus), max(mus)
        if hi > lo:
            # 非退化轴：精度加权平均必**严格**落在开区间内（各权重均 >0）
            assert lo < post[d] < hi, f"维{d}: post={post[d]} 不在开区间 ({lo}, {hi})"
        else:
            # 🛑 **退化轴保护**：全流同 μ 时开区间为空，严格断言会在**正确实现上假红**。
            # 中性轮正是这种输入（Stimulus 三个 appraisal 字段默认 0.0 → 全流 μ_v ≡ 0）。
            assert post[d] == pytest.approx(lo), f"维{d}: 退化轴上 post 应等于该值"


@pytest.mark.parametrize("seed", range(30))
def test_anchor_a_holds_on_correct_implementation(seed: int) -> None:
    """锚点 A · (b) 恒绿证明：30 组随机装配 + 门开门关，正确实现下永不假红。"""
    streams = _random_streams(seed)
    for gate in (True, False):
        terms, _ = ignite(streams, gate_fusion=gate, soft_beta=None)
        _assert_anchor_a(terms)


def test_anchor_a_degenerate_axis_is_the_neutral_turn() -> None:
    """锚点 A · 退化轴保护的**必要性**：中性轮就是退化输入，没保护就假红。"""
    neutral = _streams(
        ("survival", 0.0, 0.60, 0.40, 0.40),
        ("appraisal", 0.0, 0.09, 11.11, 11.11),
        ("value", 0.0, 0.30, 0.001, 0.62),
    )
    terms, _ = ignite(neutral, gate_fusion=False, soft_beta=None)
    assert len({mu[0] for mu, _ in terms}) == 1, "valence 轴应退化（全流 μ_v ≡ 0）"
    _assert_anchor_a(terms)  # 有保护 → 绿
    # 无保护（严格开区间）会红——这正是第四轮①的假红形态
    post, _ = fuse_terms(terms)
    mus = [mu[0] for mu, _ in terms]
    assert not (min(mus) < post[0] < max(mus)), "开区间为空，严格断言必假——故保护是必须的"


def test_anchor_a_goes_red_on_dominant_mu_bug() -> None:
    """锚点 A · (a) 变异测试：注入越出凸包的后验 → **断言真的抛 AssertionError**。

    🛑 本用例的**上一版是错的**，如实记录：它只断言 `max(mus)+0.05 不在 (min,max) 内`——
    这对任意 `mus` **数学上恒真**，从头到尾没调过 `_assert_anchor_a`，
    既不依赖 `ignite`/`fuse_terms` 的任何真实行为，也不会在断言被改弱（`<` 改 `<=`）时变红。
    那正是本 PRP 反复强调的反模式的一个变体：**看起来在测，其实什么都没测到**。
    现改为把 bogus post 注入 `_assert_anchor_a` 并要求它抛。
    """
    terms, _ = ignite(_CORE, gate_fusion=False, soft_beta=None)
    mus = [mu[1] for mu, _ in terms]
    good_post, _ = fuse_terms(terms)
    _assert_anchor_a(terms, post=good_post)  # 正确后验 → 绿
    for bogus_a in (max(mus) + 0.05, min(mus) - 0.05, max(mus)):  # 越上界 / 越下界 / 恰在边界
        with pytest.raises(AssertionError):
            _assert_anchor_a(terms, post=(good_post[0], bogus_a))


def test_anchor_b_goes_red_on_injected_wrong_posterior() -> None:
    """锚点 B · 补一条同款：注入偏离参考量的后验 → 断言真的抛。

    与 `test_anchor_b_goes_red_on_hard_gate_collapse` 互补：那条测「真实历史 bug」，
    这条测「断言本身的驱红能力」。
    """
    terms, _ = ignite(_CORE, gate_fusion=False, soft_beta=None)
    good_post, _ = fuse_terms(terms)
    _assert_anchor_b(terms, post=good_post)
    for delta in (1e-8, -1e-8, 0.05):  # 刚过容差 / 反向 / 明显偏离
        with pytest.raises(AssertionError):
            _assert_anchor_b(terms, post=(good_post[0], good_post[1] + delta))


# ── 锚点 B：硬契约（用有效精度作参考量）──


def _assert_anchor_b(
    terms: list[tuple[tuple[float, float], tuple[float, float]]],
    post: tuple[float, float] | None = None,
) -> None:
    """`post` 可注入——同 `_assert_anchor_a`，供变异测试真的驱红本断言。"""
    if post is None:
        post, _ = fuse_terms(terms)
    for d in (0, 1):
        ref = _weighted_ref(terms, d)
        if abs(ref) <= 1.0:  # clamp 未生效的区间才谈 1e-9 契约
            assert abs(post[d] - ref) < 1e-9, f"维{d}: post={post[d]} vs ref={ref}"


@pytest.mark.parametrize("seed", range(50))
def test_anchor_b_holds_on_correct_implementation(seed: int) -> None:
    """锚点 B · (b) 恒绿证明：50 组随机装配，**Π 对数均匀跨 [1e-9, ~3]**。

    这正是第四轮②假红的采样域（10 万组里 84427 组红）——用有效精度作参考量后恒绿。
    """
    streams = _random_streams(seed)
    for gate in (True, False):
        terms, _ = ignite(streams, gate_fusion=gate, soft_beta=None)
        _assert_anchor_b(terms)


def test_anchor_b_would_false_red_with_raw_precision_reference() -> None:
    """🛑 证明「必须用有效精度」不是空话：用**原生 Π** 当参考量会在正确实现上假红。"""
    tiny = _streams(("a", 0.9, 0.9, 1e-9, 1e-9), ("b", -0.9, -0.9, 1.0, 1.0))
    terms, _ = ignite(tiny, gate_fusion=False, soft_beta=None)
    post, _ = fuse_terms(terms)
    raw_ps = [prec[1] for _, prec in terms]
    raw_ref = sum(p * mu[1] for (mu, _), p in zip(terms, raw_ps, strict=True)) / sum(raw_ps)
    assert abs(post[1] - raw_ref) > 1e-9, "原生 Π 参考量应产生可观偏差（第四轮②的假红来源）"
    _assert_anchor_b(terms)  # 有效精度参考量 → 绿


def test_anchor_b_goes_red_on_hard_gate_collapse() -> None:
    """锚点 B · (a) 变异测试：接回硬门塌缩（`gate_fusion=True`）→ 相对全流参考量变红。

    这是 4 个历史变体里 B 命中的三个之一。
    """
    all_terms, _ = ignite(_CORE, gate_fusion=False, soft_beta=None)
    gated_terms, _ = ignite(_CORE, gate_fusion=True, soft_beta=None)
    assert len(gated_terms) < len(all_terms), "该场景硬门须真的排除掉流，否则没在测东西"
    post_gated, _ = fuse_terms(gated_terms)
    ref_all = _weighted_ref(all_terms, 1)
    assert abs(post_gated[1] - ref_all) > 1e-9, "变异未生效——锚点 B 对硬门塌缩无判别力"


# ── 锚点 C：流级（承接 floor bug）──
#
# 🛑 **为什么必须有一条不看 post 的锚点**：第四轮四条锚点全部漏掉 floor bug，
# 根因是**结构性的**——四条都是 `(streams → post)` 的函数，而 floor 污染的是
# **装配阶段的 μ**，锚点会从**已被污染的流列表**重新推导参照 → 按构造恒绿。


@pytest.mark.parametrize("goal", _GRID)
def test_anchor_c_no_signal_no_arousal_claim(goal: float) -> None:
    """锚点 C · 修复后：零强度输入下 survival 不再断言「确定的中等唤醒」。"""
    mu, _ = fast_survival_prior([goal, 0.0, 0.0, 0.0], arousal_floor_fix=True)
    assert mu[1] == 0.0, f"I=0 时 μ_a 应为 0，实际 {mu[1]}"


def test_anchor_c_goes_red_on_floor_bug() -> None:
    """锚点 C · (a) 变异测试：接回地板 → 立刻变红。"""
    mu, _ = fast_survival_prior([0.0, 0.0, 0.0, 0.0], arousal_floor_fix=False)
    assert mu[1] == 0.5, "变异未生效——锚点 C 对 floor bug 无判别力"


def test_anchor_c_catches_what_post_shaped_anchors_structurally_cannot() -> None:
    """🛑 锚点 C 的**存在理由**：证明 A/B 这类 `(streams→post)` 锚点确实抓不到 floor bug。

    做法：拿带地板与不带地板的两套流各自算后验——**两个后验都各自满足 A 与 B**
    （因为锚点从各自已污染/未污染的流重新推导参照），而两者数值明显不同。
    即：A/B 全绿，但后验被改变了。
    """
    feats = [0.0, 0.0, 0.0, 0.0]
    posts = []
    for fix in (False, True):
        surv_mu, surv_prec = fast_survival_prior(feats, arousal_floor_fix=fix)
        streams: list[Stream] = [("survival", surv_mu, surv_prec), *_CORE[1:]]
        terms, _ = ignite(streams, gate_fusion=False, soft_beta=None)
        _assert_anchor_a(terms)  # 两边都绿
        _assert_anchor_b(terms)  # 两边都绿
        posts.append(fuse_terms(terms)[0][1])
    assert abs(posts[0] - posts[1]) > 1e-6, (
        f"地板确实改变了后验（{posts[0]} vs {posts[1]}），而 A/B 两边全绿——"
        "这就是为什么必须有流级锚点 C"
    )


def test_anchor_c_is_bound_to_the_same_switch_as_gate_fusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D5 强制：地板修复与 `gate_fusion` **共用同一开关**，杜绝未评审的中间态。

    🛑 **本用例的上一版是同义反复，如实记录**（终审工作流抓出，同一反模式第三次）：
    它 `agent(st)` 后把返回值整个丢弃，再自己用 `arousal_floor_fix=not st.gate_fusion`
    **重算一遍期望**——断言的是测试代码自己，与生产调用点 `affect_core.py` 零耦合。
    实测把那行的门方向写反（**在默认门关位直接破零回归**，`post_mu`/`ignited_streams` 都变），
    全套 1687 条**无一变红**。

    现改为**监视生产调用点**：spy 掉 `affect_core` 命名空间里的 `fast_survival_prior`，
    断言它**实际收到**的 `arousal_floor_fix` 与 `gate_fusion` 反相。门方向写反即红。
    """
    import src.agents.affect_core as core_mod

    real = core_mod.fast_survival_prior
    captured: list[bool | None] = []

    def spy(features: list[float], **kw: object) -> tuple:
        captured.append(kw.get("arousal_floor_fix"))  # type: ignore[arg-type]
        return real(features, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(core_mod, "fast_survival_prior", spy)

    for gate_fusion, expect_floor_fix in ((True, False), (False, True)):
        captured.clear()
        AffectCoreAgent()(_core_state(gate_fusion=gate_fusion, features=[0.0, 0.0, 0.0, 0.0]))
        assert captured == [expect_floor_fix], (
            f"gate_fusion={gate_fusion} 时生产调用点传的 arousal_floor_fix 应为 "
            f"{expect_floor_fix}，实际 {captured}"
        )


# ---------------------------------------------------------------------------
# 7. 治理 + 场景矩阵（精简版：在场流集合 × 门 × 软门）
# ---------------------------------------------------------------------------


def test_governance_flags_are_paired_and_direction_is_right() -> None:
    """治理白名单与 base dict **成对**；且默认方向是 True（最容易写反的一条）。"""
    from src.mcp_server.server import _MCP_GOVERNANCE_GATED_FLAGS, _build_session_config

    assert {"gate_fusion", "exclude_physio_fusion"} <= _MCP_GOVERNANCE_GATED_FLAGS
    cfg = _build_session_config(None)
    assert cfg.gate_fusion is True, "漏 base dict 会使其永久 True——但那样这条也过，看下一条"
    assert cfg.exclude_physio_fusion is True
    # client override 不得旁路（跨仓承诺不容单边解除）
    hacked = _build_session_config({"gate_fusion": False, "exclude_physio_fusion": False})
    assert hacked.gate_fusion is True
    assert hacked.exclude_physio_fusion is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(None, True, id="未设→True(门关)"),
        pytest.param("", True, id="空串→True"),
        pytest.param("false", False, id="false→开"),
        pytest.param("FALSE", False, id="大小写不敏感"),
        pytest.param("0", False, id="0→开"),
        pytest.param("no", False, id="no→开"),
        pytest.param("off", False, id="off→开"),
        pytest.param("true", True, id="true→关"),
        pytest.param("yes", True, id="其它值一律按关（保守）"),
    ],
)
def test_env_parsing_is_false_set_and_defaults_to_gate_closed(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: bool
) -> None:
    """🛑 env 解析必须判**假值集**，且未设时默认 True（门关）。

    这条改成了**真行为测试**（实跑 `build_chat_driver`），不再用源码文本匹配——
    文本匹配只能防「把 `not in` 整体换成 `in`」这种字面改法，**防不住假值集里漏写一个值**
    （如漏掉 `"off"`），而这恰恰是这个「方向最容易写反」的开关的最高风险点。
    用弱测试守最高风险处不对称。
    """
    from src.orchestration.chat_driver import build_chat_driver

    monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)  # 不走网络
    if raw is None:
        monkeypatch.delenv("ZERO_IGNITION_GATE_FUSION", raising=False)
    else:
        monkeypatch.setenv("ZERO_IGNITION_GATE_FUSION", raw)
    driver = build_chat_driver(thread=f"env-gate-{raw!r}")
    assert driver.session.config.gate_fusion is expected


def test_env_physio_exclusion_defaults_to_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    """physio 排除同为「默认 True + 判假值集」——跨仓承诺不能因未设 env 而失效。"""
    from src.orchestration.chat_driver import build_chat_driver

    monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ZERO_EXCLUDE_PHYSIO_FUSION", raising=False)
    assert build_chat_driver(thread="env-physio-unset").session.config.exclude_physio_fusion is True
    monkeypatch.setenv("ZERO_EXCLUDE_PHYSIO_FUSION", "off")
    assert build_chat_driver(thread="env-physio-off").session.config.exclude_physio_fusion is False


@pytest.mark.parametrize(
    "present",
    [
        pytest.param([], id="核心3"),
        pytest.param([("text", (0.2, 0.2), (0.3, 0.3))], id="+text"),
        pytest.param([("mood", (0.1, -0.1), (0.8, 0.8))], id="+mood"),
        pytest.param(
            [("text", (0.2, 0.2), (0.3, 0.3)), ("mood", (0.1, -0.1), (0.8, 0.8))],
            id="+text+mood",
        ),
    ],
)
@pytest.mark.parametrize("n_external", [0, 1, 2])
@pytest.mark.parametrize("soft_beta", [None, 20.0])
def test_scenario_matrix(present: list[Stream], n_external: int, soft_beta: float | None) -> None:
    """场景矩阵：在场流集合 × external 注入数 × 软门 × 门开关。

    每格断言：融合集与流名对齐 · 报告集 ⊆ 全流名 · 锚点 A/B 成立 · physio 不在融合集。
    """
    ext: list[Stream] = [(f"face_{i}", (0.3, 0.2), (0.20, 0.12)) for i in range(n_external)]
    # 🛑 矩阵里**必须**有一条 physio 前缀的流，否则下面那条「physio 不在融合集」的断言
    # 按构造恒真（没有可被排除的东西可测），是零判别力的摆设。上一版就漏了这条。
    physio: list[Stream] = [("eda_tonic", (0.0, 0.7), (MIN_PRECISION, 0.175))]
    streams: list[Stream] = [*_CORE, *present, *ext, *physio]
    for gate in (True, False):
        terms, names = ignite(streams, gate_fusion=gate, soft_beta=soft_beta)
        assert len(terms) == len(names)
        ignited = report_ignited(streams, soft_beta=soft_beta)
        assert set(ignited) <= {n for n, _, _ in streams}
        _assert_anchor_a(terms)
        _assert_anchor_b(terms)
        if not gate:
            assert not any(
                n.lower().startswith(("physio", "eda", "hrv", "pupil", "scr")) for n in names
            )


def test_domain_sweep_gate_on_never_violates_hard_contract() -> None:
    """域穷举：门开时硬契约在整个评价域上成立（这是新架构的定义性质）。"""
    for gc, inten in itertools.product(_GRID, _GRID):
        surv_mu, surv_prec = fast_survival_prior([gc, 0.0, 0.0, inten], arousal_floor_fix=True)
        streams: list[Stream] = [("survival", surv_mu, surv_prec), *_CORE[1:]]
        terms, _ = ignite(streams, gate_fusion=False, soft_beta=None)
        post, _ = fuse_terms(terms)
        for d in (0, 1):
            ref = _weighted_ref(terms, d)
            if abs(ref) <= 1.0:
                assert math.isclose(post[d], ref, abs_tol=1e-9)


def test_survival_fallback_scope_narrows_but_does_not_vanish_when_gate_opens() -> None:
    """`ignition_survival_fallback` 在门开后**作用域收窄、不是失效**。

    摘门后它不再影响数值通路（融合集恒为全流），但仍经 `report_ignited()` 决定
    全弱刺激时报告的是 survival 还是 top-1 —— 即从「管数值+管报告」收窄为「只管报告」。
    这条写成用例，是因为 `tasks.md` 的独立待办里曾把它列为「设了却不生效也不报错」，
    实测**不成立**；若哪天真的失效了，本用例会红。
    """
    weak = _streams(
        ("survival", 0.01, 0.01, 0.01, 0.01),
        ("appraisal", 0.02, 0.02, 0.02, 0.02),
    )
    assert all(stream_salience(mu, p) < SALIENCE_THRESHOLD for _, mu, p in weak)
    # 报告通路：两种取值给出不同结果 → 仍有作用
    assert report_ignited(weak, survival_fallback=True, soft_beta=None) == ["survival"]
    assert report_ignited(weak, survival_fallback=False, soft_beta=None) == ["appraisal"]
    # 数值通路：门开后恒为全流，不受该 flag 影响 → 作用域确已收窄
    for fb in (True, False):
        _, names = ignite(weak, survival_fallback=fb, soft_beta=None, gate_fusion=False)
        assert names == ["survival", "appraisal"]


# ---------------------------------------------------------------------------
# 8. `run()` 入口的护栏覆盖（BLOCK-1 回归锁）
#
# 🛑 `run()` 是一条**不经 ConversationSession 的公开入口**（scripts/run_pipeline.py 等
# 直接调），它手拼 ainvoke 初值 dict、从不构造 SessionConfig ——写在 SessionConfig
# 校验器里的护栏在这条路径上曾经**等于不存在**（code-reviewer 实测：同参数经
# ConversationSession fail-fast、经 run() 静默产出错误数值）。
#
# 修复本身此前**没有回归测试**：没有任何东西能防止它被移位/删除/被 try-except 吞掉。
# 这条 PRP 全篇强调「断言要真的能变红」，这个位置不该是例外。
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_entrypoint_enforces_block2_mutual_exclusion() -> None:
    """`run()` 必须触发 BLOCK 2 互斥校验（护栏不能只在 ConversationSession 上生效）。"""
    from src.orchestration.runner import run

    with pytest.raises(ValueError, match="联合语义未定义"):
        await run(
            [Stimulus(name="s", goal_congruence=0.4, intensity=0.6)],
            thread_id="block2-via-run",
            gate_fusion=False,
            hierarchical_layers=2,
            hierarchical_coupling=0.5,
        )


@pytest.mark.asyncio
async def test_run_entrypoint_enforces_all_session_config_validators() -> None:
    """不止 BLOCK 2——`run()` 补的这一道会触发 SessionConfig 的**全部**跨字段校验。

    这正是用 `locals()` 按 `model_fields` 过滤（而非手写字段清单）换来的：
    线 B 的混标度校验、共情系数 L1 上界这些**早于本轮就存在**的护栏，
    在 `run()` 上此前同样是缺口，一并被补上了。
    """
    from src.orchestration.runner import run

    stim = [Stimulus(name="s", goal_congruence=0.4, intensity=0.6)]
    # 线 B：齐次化门开 + 旧标度旋钮 → 混标度
    with pytest.raises(ValueError, match="不可混用"):
        await run(
            stim, thread_id="mixscale-via-run", precision_commensurable=True, mood_precision=1.2
        )
    # 更早就存在的：共情系数 L1 和 > 0.6
    with pytest.raises(ValueError, match="共情系数"):
        await run(stim, thread_id="empathy-via-run", contagion_alpha=0.3, care_bias_alpha=0.4)


@pytest.mark.asyncio
async def test_run_entrypoint_does_not_over_reject() -> None:
    """护栏不得误伤——合法参数组合经 `run()` 照常跑完。"""
    from src.orchestration.runner import run

    traj = await run(
        [Stimulus(name="s", goal_congruence=0.4, intensity=0.6)],
        thread_id="ok-via-run",
        workspace_enabled=True,
        gate_fusion=False,  # 新架构单独开（不配 HPC）→ 合法
    )
    assert len(traj) == 1


# ---------------------------------------------------------------------------
# 9. MCP 侧与 chat 侧的 env 语义必须一致（终审工作流抓出的第二条）
#
# 🛑 `_env_flag` 旧实现只判**真值集**，未识别值一律 False。那对「默认 False」的旗标恰好
# 等于 default、看不出问题；用在**默认 True** 的旗标上失败方向就反了——空串 /「true 」/
# 「enabled」会静默把门**打开**，而 chat 侧同样取值判假值集、结论是门**关**。
# 后果：`ZERO_MCP_EXCLUDE_PHYSIO_FUSION=""` 时反号 physio 回到数值通路、arousal 抬高 150%+，
# 等于单边解除对 Zero_MCP 的 D7 跨仓承诺。
# ---------------------------------------------------------------------------

_ENV_VALUES = [
    None,
    "",
    " ",
    "true",
    "TRUE",
    "true ",
    " true",
    "1",
    "yes",
    "on",
    "false",
    "FALSE",
    "0",
    "no",
    "off",
    "enabled",
    "disabled",
    "y",
    "n",
    "t",
    "f",
]


@pytest.mark.parametrize("raw", _ENV_VALUES)
def test_mcp_and_chat_agree_on_every_env_value(
    monkeypatch: pytest.MonkeyPatch, raw: str | None
) -> None:
    """同一 env 取值，MCP 边界与 chat 工厂必须给出**相同**的 gate_fusion 结论。

    两侧各测各的（本文件早先那组 chat 侧用例）**测不出跨侧分歧**——必须直接断言一致性。
    """
    from src.mcp_server.server import _build_session_config
    from src.orchestration.chat_driver import build_chat_driver

    monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
    for name in ("ZERO_IGNITION_GATE_FUSION", "ZERO_MCP_IGNITION_GATE_FUSION"):
        if raw is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, raw)
    chat = build_chat_driver(thread=f"agree-{raw!r}").session.config.gate_fusion
    mcp = _build_session_config(None).gate_fusion
    assert chat == mcp, f"取值 {raw!r}：chat={chat} 但 mcp={mcp} —— 两侧语义分歧"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, True), ("", True), (" ", True), ("enabled", True), ("true ", True), ("off", False)],
)
def test_env_flag_falls_back_to_default_not_to_false(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: bool
) -> None:
    """未识别值（含空串）回落 `default`，**不是**一律 False——这是失败方向的关键。"""
    from src.mcp_server.server import _env_flag

    if raw is None:
        monkeypatch.delenv("ZERO_TEST_FLAG_DIRECTION", raising=False)
    else:
        monkeypatch.setenv("ZERO_TEST_FLAG_DIRECTION", raw)
    assert _env_flag("ZERO_TEST_FLAG_DIRECTION", True) is expected


@pytest.mark.parametrize("raw", ["", " ", "enabled", "y", "t"])
def test_env_flag_zero_regression_for_default_false_flags(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """对三个既有的「默认 False」旗标**逐字零回归**：未识别值仍得 False（旧实现同值）。"""
    from src.mcp_server.server import _env_flag

    monkeypatch.setenv("ZERO_TEST_FLAG_DIRECTION", raw)
    assert _env_flag("ZERO_TEST_FLAG_DIRECTION", False) is False


def test_physio_exclusion_survives_empty_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """🛑 跨仓承诺不得因一个空 env 值被单边解除。"""
    from src.mcp_server.server import _build_session_config

    monkeypatch.setenv("ZERO_MCP_EXCLUDE_PHYSIO_FUSION", "")
    assert _build_session_config(None).exclude_physio_fusion is True
