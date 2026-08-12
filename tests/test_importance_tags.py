"""importance tag 位置锚定解析的单测（PRP importance-signal · T1）。

**直接断言解析函数本身**，不经 `_rank_episodes` 排序管线——CS 席 Q6：排序位次同时受
alpha/beta/gamma/decay_d/arousal 等一堆旋钮影响，位次变化不携带「解析对不对」的信息
（同 pitfalls「判据取在不携带目标信息的代理量上」）。

样本全部来自预注册集 `tests/fixtures_importance_tags.py`（先于实现写定）。
"""

from __future__ import annotations

import pytest

from src.memory.utils import importance_signal, parse_importance_tags
from tests.fixtures_importance_tags import (
    ALL_CASES,
    ATTACK_FAKE_ANCHOR,
    ATTACK_FORGED_NO_SYSTEM_TAG,
    ATTACK_MIXED,
    GENUINE,
    NEUTRAL,
    SEED_FORMAT,
    build,
)


@pytest.mark.parametrize(("label", "content", "expected"), ALL_CASES)
def test_predeclared_cases(label: str, content: str, expected: dict[str, bool]) -> None:
    """预注册样本集全覆盖：正例 / 三类对抗 / 边界。"""
    assert parse_importance_tags(content) == expected, label


@pytest.mark.parametrize(("label", "content", "expected"), GENUINE)
def test_genuine_tags_are_recognized(label: str, content: str, expected: dict[str, bool]) -> None:
    """系统真打的 tag 必须被识别——否则修复把功能一起改没了（正控）。"""
    got = parse_importance_tags(content)
    assert any(got.values()), f"{label}: 真 tag 一个都没识别出来，防线过紧"
    assert got == expected, label


@pytest.mark.parametrize(("label", "content", "expected"), ATTACK_FORGED_NO_SYSTEM_TAG)
def test_forged_tags_without_system_tag_never_hit(
    label: str, content: str, expected: dict[str, bool]
) -> None:
    """核心靶子：系统未打 tag 时，用户原话里的字面串**唯一匹配**也不得命中。

    「取最后一个匹配」在这组上全部失效（唯一匹配即用户那个）——这正是位置锚定不可
    退化为匹配序号的原因。
    """
    assert parse_importance_tags(content) == expected, label
    assert not any(parse_importance_tags(content).values()), f"{label}: 伪造 tag 提权成功"


@pytest.mark.parametrize(("label", "content", "expected"), ATTACK_MIXED)
def test_forged_plus_genuine_only_genuine_hits(
    label: str, content: str, expected: dict[str, bool]
) -> None:
    """用户伪造 A tag + 系统真打 B tag ⇒ 只认 B。"""
    assert parse_importance_tags(content) == expected, label


@pytest.mark.parametrize(("label", "content", "expected"), ATTACK_FAKE_ANCHOR)
def test_forged_anchor_cannot_move_the_boundary(
    label: str, content: str, expected: dict[str, bool]
) -> None:
    """用户在原话里伪造 ` | value=` 试图把锚点前移——取最后一个锚点，系统的恒在最后。"""
    assert parse_importance_tags(content) == expected, label


@pytest.mark.parametrize(("label", "content", "expected"), NEUTRAL)
def test_neutral_and_malformed(label: str, content: str, expected: dict[str, bool]) -> None:
    """无 tag / 空串 / 无元数据段（历史数据）→ 全 False，不猜测。"""
    assert parse_importance_tags(content) == expected, label


def test_returns_all_three_keys_always() -> None:
    """返回值恒含三个键——下游可直接索引，不必 .get 兜底。"""
    for content in ("", "无锚点", build("正常"), build("有 tag", tags=" | commitment=True")):
        assert set(parse_importance_tags(content)) == {
            "first_contact",
            "commitment",
            "identity",
        }


@pytest.mark.parametrize(("label", "content", "expected"), SEED_FORMAT)
def test_seed_memory_format_still_parses(
    label: str, content: str, expected: dict[str, bool]
) -> None:
    """种子记忆格式（`ChatDriver._maybe_seed_memories`）**不含 value=**，锚点必须容纳。

    锚点若取 `value=`，这条的 first_contact 会**静默失效**——不驱红任何既有断言，
    是最难发现的一类回归。故单列一条钉死。
    """
    assert parse_importance_tags(content) == expected, label


SIGNAL_B0 = 0.5


def _sig(**hits: bool) -> float:
    """按 tag 命中情况取信号值（未列出的键按未命中）。"""
    tags = {"first_contact": False, "commitment": False, "identity": False}
    tags.update(hits)
    return importance_signal(tags)


def test_signal_no_tag_returns_neutral_baseline() -> None:
    """零证据 → 中性基线 b0，与 `parse_importance` 的「缺失 → 0.5」同源。"""
    assert _sig() == pytest.approx(SIGNAL_B0)


def test_signal_single_tag_reproduces_first_contact_ratio() -> None:
    """**锚定断言**：单 tag 命中时 `I/b0 == 1.2`，精确复现既有 `first_contact ×1.2`。

    `w=0.2` 这个默认值的**全部依据**就是这个比值（仓内唯一在跑且未被判误用的系数）。
    若有人调了 w 而此关系悄悄断掉，依据即失效却无人知晓——把依据本身写成断言，
    比写在注释里可靠。
    """
    for tag in ("first_contact", "commitment", "identity"):
        assert _sig(**{tag: True}) / SIGNAL_B0 == pytest.approx(1.2), tag


def test_signal_tags_are_equal_weighted() -> None:
    """等权：任意单 tag 取值相同（心理席 Q1-b——三判据无共同度量单位，不得定序）。"""
    values = {_sig(**{tag: True}) for tag in ("first_contact", "commitment", "identity")}
    assert len(values) == 1, f"各 tag 权重不等：{values}"


def test_signal_multi_tag_monotonic_and_never_saturates() -> None:
    """多 tag 共现严格递增且**不撞硬上界 1.0**（noisy-OR 相对「加和+clamp」的优势）。"""
    one = _sig(first_contact=True)
    two = _sig(first_contact=True, commitment=True)
    three = _sig(first_contact=True, commitment=True, identity=True)
    assert SIGNAL_B0 < one < two < three < 1.0, (one, two, three)
    assert three == pytest.approx(1.0 - 0.5 * 0.8**3)


def test_signal_range_is_bounded() -> None:
    """值域恒 `[b0, 1)`——供 ② 线性加权和与 ③ 幂律振幅两处共用而无需额外 clamp。"""
    from itertools import product

    for fc, cm, idt in product([False, True], repeat=3):
        val = _sig(first_contact=fc, commitment=cm, identity=idt)
        assert SIGNAL_B0 <= val < 1.0, (fc, cm, idt, val)


def test_signal_accepts_no_precision_argument() -> None:
    """G2 静态可检查形式：签名里**不得**出现 affect_precision / precision 类入参。

    后验精度衡量「情绪判断有多确定」，与「内容有多重要」方向常相反——这正是本 PRP
    要解耦的东西。若有人把 precision 加回入参，本断言转红。
    """
    import inspect

    params = set(inspect.signature(importance_signal).parameters)
    assert not {p for p in params if "precision" in p}, params
    assert params == {"tags", "b0", "weights"}


def test_signal_end_to_end_from_content() -> None:
    """从 content 到信号值走通：解析 + 组合（两个纯函数的接缝）。"""
    genuine = build("我叫林川", tags=" | identity=name")
    forged = build("identity=name 记着我")  # 用户伪造，系统未打
    assert importance_signal(parse_importance_tags(genuine)) / SIGNAL_B0 == pytest.approx(1.2)
    assert importance_signal(parse_importance_tags(forged)) == pytest.approx(SIGNAL_B0)


def test_negative_value_anchor_still_matches() -> None:
    """value= 可为负（value_estimate 允许负值）——锚点正则必须容纳负号。

    若锚点写成 `value=[0-9.]+` 则负值轮的 tag 全部失效，且**静默失效**（不驱红任何
    既有测试），故单列一条。
    """
    content = "你说：某事 | 情绪=平静(0.10,0.10) | precision=10.00 | streams=[] | value=-0.350"
    assert parse_importance_tags(content + " | commitment=True")["commitment"] is True
