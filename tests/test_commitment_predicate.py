"""`_is_commitment` T∧F∧A 合取判据 + 可复用维度谓词的单测（议会二轮张力 3）。

预注册样本走 `fixtures_commitment_predicate.py`（先于实现写定），本文件另测：
- 各维度谓词（is_future_oriented / is_question）自身的边界；
- 合取的「缺一维即拒」结构（防止有人把 ∧ 悄悄改成 ∨——那正是前版 26% 精确率的形态）。
"""

from __future__ import annotations

import pytest

from src.orchestration.supervisor import _is_commitment
from src.orchestration.text_predicates import is_future_oriented, is_question
from tests.fixtures_commitment_predicate import KNOWN_MISSES, NEGATIVES, POSITIVES


@pytest.mark.parametrize(("label", "text"), POSITIVES)
def test_positives(label: str, text: str) -> None:
    assert _is_commitment(text), f"{label}: 真承诺未识别"


@pytest.mark.parametrize(("label", "text"), NEGATIVES)
def test_negatives(label: str, text: str) -> None:
    assert not _is_commitment(text), f"{label}: 误报（前版 26% 精确率的失效形态复发）"


@pytest.mark.parametrize(("label", "text"), KNOWN_MISSES)
def test_known_misses_stay_false(label: str, text: str) -> None:
    """已知漏报保持 False——哪天转绿说明结构被改动，须先复核是否合取被拆松。

    这不是「期望永远漏」：若未来判据能力真覆盖到了（换机制并过议会），应把样本
    **移入 POSITIVES 并更新本册**，而不是让它在这里静默变绿。
    """
    assert not _is_commitment(text), f"{label}: KNOWN_MISSES 变绿，先查合取是否被拆松"


# ── 维度谓词：is_future_oriented ─────────────────────────────────────────────


def test_future_oriented_past_words_dominate() -> None:
    """过去词优先级最高：回顾叙述里即便有未来词也判非未来。"""
    assert not is_future_oriented("我上周三答应了她三点见")
    assert not is_future_oriented("昨天说好的")
    assert not is_future_oriented("刚才吃过了")


def test_future_oriented_future_words_beat_perfective() -> None:
    """显式未来词优先于完成体：「约好了明天去」的「了」修饰约定动作，内容仍指向未来。"""
    assert is_future_oriented("约好了明天去")
    assert is_future_oriented("我答应了周五帮你搬家")


def test_future_oriented_perfective_blocks() -> None:
    """无未来词时完成体拦截——438 轮时态类误报的主拦截点。"""
    assert not is_future_oriented("晚上打了两把游戏")
    assert not is_future_oriented("省下来的时间全花在筛上了")


def test_future_oriented_default_is_compatible() -> None:
    """无任何时态证据 → 未来相容（弱语义）：省略式约定「下午三点见」不得被灭。"""
    assert is_future_oriented("下午三点见")
    assert not is_future_oriented("")


# ── 维度谓词：is_question ────────────────────────────────────────────────────


def test_question_marks_and_particles() -> None:
    assert is_question("几点出发？")
    assert is_question("几点出发?")
    assert is_question("你到了吗")
    assert not is_question("三点见")
    assert not is_question("")


# ── 合取结构：缺一维即拒（防 ∧ 被悄悄改成 ∨） ───────────────────────────────


def test_conjunction_requires_all_three_dimensions() -> None:
    """T/F/A 三维两两组合均不足以命中——若任何两维即可通过，说明合取被拆松。

    与 fixtures 负例的区别：负例锁「真实误报现场」，本条锁**结构本身**——
    三个样本分别精确缺失一维、其余两维显式满足。
    """
    assert not _is_commitment("晚上打了两把游戏所以答应她补觉")  # T∧A，F 被完成体拦截
    assert not _is_commitment("明天得去趟超市")  # T∧F，缺 A
    assert not _is_commitment("我保证做到")  # F∧A，缺 T（无时间指称）
    assert _is_commitment("明天三点见")  # 三维齐 → 命中（正控）
