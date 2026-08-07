"""行为反馈流纯函数层单测（行为反馈环第二步·T2）。

对象：affect_math.behavior_feedback_evidence / cap_stream_weight / behavior_precision。
设计权威 PRP/behavior-feedback-loop/design.md §1/§2；议会裁定
notes/2026-08-07-behavior-feedback-council.md（C 主干 + 在场门 + 位置空间口径 + w_b 后置封顶）。
"""

from __future__ import annotations

import pytest

from src.agents.affect_math import (
    BEHAVIOR_PRECISION,
    MIN_PRECISION,
    SIGMA_BHV,
    W_MAX_BEHAVIOR,
    behavior_feedback_evidence,
    behavior_precision,
    cap_stream_weight,
)
from src.agents.motion_synth import modulation_from_affect


def _copy(
    *,
    voluntary_arousal: float | None = None,
    scene: str = "idle",
) -> dict:
    """按第一步副本口径构造 motion_efference（voluntary 由真实仿射映射产出，保证逆映射
    测试测的是与生产同一条数学，不是测试自造的镜像）。"""
    spont = modulation_from_affect(0.0, 0.2)
    copy: dict = {
        "spontaneous": {
            "amplitude": spont.amplitude,
            "speed": spont.speed,
            "onset": spont.onset_sharpness,
        },
        "voluntary": None,
        "scene": scene,
        "events": [],
    }
    if voluntary_arousal is not None:
        vol = modulation_from_affect(0.0, voluntary_arousal)
        copy["voluntary"] = {
            "amplitude": vol.amplitude,
            "speed": vol.speed,
            "onset": vol.onset_sharpness,
        }
    return copy


# ── 在场门（absent-cue）────────────────────────────────────────────────────────


def test_absent_when_voluntary_none() -> None:
    """voluntary=None ⇔ δ≡0 ⇒ 流缺席（None），不注入 0 值假证据——生产默认
    regulation 关时的唯一路径（G1 零回归的机制保证）。"""
    assert behavior_feedback_evidence(_copy()) is None


def test_present_when_voluntary_exists() -> None:
    out = behavior_feedback_evidence(_copy(voluntary_arousal=-0.4))
    assert out is not None


# ── 位置空间口径与仿射逆精确性 ────────────────────────────────────────────────


@pytest.mark.parametrize("a_reg", [-1.0, -0.4, 0.0, 0.3, 0.75, 1.0])
def test_inverse_recovers_regulated_arousal_exactly(a_reg: float) -> None:
    """a_expr = 2·onset − 1 是 modulation_from_affect 的精确仿射逆（无增益依赖）。
    ⚠ 单射性前提：本断言绑定解析回退实现；真模型替换后不得沿用（议会必改 #14）。"""
    out = behavior_feedback_evidence(_copy(voluntary_arousal=a_reg))
    assert out is not None
    (mu_v, mu_a), _ = out
    assert mu_v == 0.0
    assert mu_a == pytest.approx(a_reg, abs=1e-12)


def test_valence_dim_floored() -> None:
    out = behavior_feedback_evidence(_copy(voluntary_arousal=0.5))
    assert out is not None
    _, (prec_v, prec_a) = out
    assert prec_v == MIN_PRECISION
    assert prec_a == BEHAVIOR_PRECISION


def test_precision_two_regimes() -> None:
    """精度两档都有定义（阶段 55 教训）：legacy=常数，齐次档=1/σ²。"""
    assert behavior_precision() == BEHAVIOR_PRECISION
    assert behavior_precision(commensurable=True) == pytest.approx(1.0 / SIGMA_BHV**2)
    out = behavior_feedback_evidence(_copy(voluntary_arousal=0.5), commensurable=True)
    assert out is not None
    assert out[1][1] == pytest.approx(1.0 / SIGMA_BHV**2)


def test_speaking_scene_blocked() -> None:
    """speaking 预留结构（PRD D3）：显式阻塞非空壳——有写入者前不给数值。"""
    with pytest.raises(NotImplementedError, match="speaking"):
        behavior_feedback_evidence(_copy(voluntary_arousal=0.5, scene="speaking"))


# ── w_b 后置封顶（议会必改 #2）────────────────────────────────────────────────


def _w(terms: list, idx: int, dim: int = 1) -> float:
    total = sum(max(MIN_PRECISION, prec[dim]) for _, prec in terms)
    return max(MIN_PRECISION, terms[idx][1][dim]) / total


def test_cap_noop_when_target_absent() -> None:
    terms = [((0.1, 0.2), (1.0, 1.0))]
    assert cap_stream_weight(terms, ["appraisal"], target="behavior") == (terms, ["appraisal"])


def test_cap_noop_when_below_wmax() -> None:
    """强流环境下 w_b 本就 ≤ W_MAX ⇒ 原样返回（不动无辜配置）。"""
    terms = [
        ((0.1, 0.2), (8.0, 8.0)),  # appraisal 强
        ((0.0, 0.3), (MIN_PRECISION, BEHAVIOR_PRECISION)),
    ]
    names = ["appraisal", "behavior"]
    assert cap_stream_weight(terms, names, target="behavior") == (terms, names)
    assert _w(terms, 1) <= W_MAX_BEHAVIOR


def test_cap_prunes_when_others_all_silent() -> None:
    """数学席点名的最坏情形（前置 cap π_b 失效处）：其余流全部触底 MIN_PRECISION。
    此时重标定值跌破 MIN_PRECISION、fuse_terms 地板会使封顶静默失效（只能拿到均分票
    1/(n+1)≈0.33）⇒ 封顶不可达，整条流剔除（absent 语义）——全沉默环境里 lag-1 行为
    回声不该获得均分投票权。terms/names 成对返回保持对齐（BLOCK 1 先例）。"""
    terms = [
        ((0.0, 0.0), (MIN_PRECISION, MIN_PRECISION)),
        ((0.0, 0.0), (MIN_PRECISION, MIN_PRECISION)),
        ((0.0, 0.5), (MIN_PRECISION, BEHAVIOR_PRECISION)),
    ]
    names = ["survival", "appraisal", "behavior"]
    assert _w(terms, 2) > 0.9  # 未封顶确实失控（先证明有病，再证明药有效）
    capped_terms, capped_names = cap_stream_weight(terms, names, target="behavior")
    assert capped_names == ["survival", "appraisal"]  # behavior 被剔除
    assert capped_terms == terms[:2]
    # 剔除后 w_b = 0 ≤ W_MAX 平凡成立；且 fuse_terms 输入非空（不触发 D12 raise）
    assert len(capped_terms) == 2


def test_cap_survival_floor_worst_case_rescales() -> None:
    """含 survival 地板（Π=0.4）的最坏情形：数学席实测未封顶 w_b≈0.27 > 0.15。
    此处重标定值 0.15/0.85·0.4 ≈ 0.071 ≥ MIN_PRECISION ⇒ 走重标定路（非剔除），
    封顶后权重恰为 W_MAX。"""
    terms = [
        ((0.0, 0.5), (0.4, 0.4)),  # survival 地板
        ((0.0, 0.5), (MIN_PRECISION, BEHAVIOR_PRECISION)),
    ]
    names = ["survival", "behavior"]
    assert _w(terms, 1) > W_MAX_BEHAVIOR  # ≈0.27，提案 A 的声称上界被击穿
    capped_terms, capped_names = cap_stream_weight(terms, names, target="behavior")
    assert capped_names == names  # 未剔除
    assert _w(capped_terms, 1) == pytest.approx(W_MAX_BEHAVIOR)
    # 其它流与目标流的 μ/valence 维不动，仅目标流 arousal 维精度被重标定
    assert capped_terms[0] == terms[0]
    assert capped_terms[1][0] == terms[1][0]
    assert capped_terms[1][1][0] == terms[1][1][0]


def test_cap_does_not_mutate_input() -> None:
    terms = [
        ((0.0, 0.5), (0.4, 0.4)),
        ((0.0, 0.5), (MIN_PRECISION, BEHAVIOR_PRECISION)),
    ]
    snapshot = [(mu, prec) for mu, prec in terms]
    names = ["survival", "behavior"]
    names_snapshot = list(names)
    cap_stream_weight(terms, names, target="behavior")
    assert terms == snapshot
    assert names == names_snapshot


def test_cap_length_mismatch_fails_fast() -> None:
    with pytest.raises(ValueError, match="长度不一致"):
        cap_stream_weight([((0.0, 0.0), (1.0, 1.0))], ["a", "b"], target="a")
