"""确定性文本判据的注册表门禁（`.claude/rules/text-predicate-admission.md` 的机械层）。

背景：同一失效形态（非锚定子串搜索 + 判据词与目标概念不蕴含）在本仓已出现两次
（identity 初版 13/13 假阳、commitment 时间词 26% 精确率），两次都是靠人工/实跑才发现。
本门禁把「预注册样本先于实现、且全部通过」变成 CI 强制：新增 `_is_xxx` 类判据须在
`_REGISTRY` 登记；不登记 = 审查期 BLOCK（见规则文件）。

样本精确、非统计阈值：预注册集上**任何一条**翻车即红——这些样本是实测现场，
不是随机抽样，没有「统计上可接受的错误率」一说。
"""

from __future__ import annotations

from collections.abc import Callable

from src.orchestration.supervisor import _is_commitment
from tests import fixtures_commitment_predicate as commitment_fixtures

# 注册表：判据 → (正例, 负例, 已知漏报)。
# `_is_identity_disclosure` 先于本机制、有专属测试套（test_identity_fact_bypass 等），
# 视为已满足准入；迁入注册表可选（迁入时其样本册须补「字面命中但语义不符」分类标注）。
_REGISTRY: dict[str, tuple[Callable[[str], bool], list, list, list]] = {
    "_is_commitment": (
        _is_commitment,
        commitment_fixtures.POSITIVES,
        commitment_fixtures.NEGATIVES,
        commitment_fixtures.KNOWN_MISSES,
    ),
}

# 准入标准的样本量下限（规则文件「CI 门禁」节）：
_MIN_POSITIVES = 5
_MIN_NEGATIVES = 8
_MIN_LITERAL_HIT_NEGATIVES = 3  # 「字面命中判据词但语义不符」类——两次历史事故的共同形态


def test_registry_sample_floors() -> None:
    """每个登记判据的样本量须达下限；负例中「字面命中」类须足量。

    「字面命中」按**含时间/施为字面词**近似判定（对 commitment 而言）：负例文本含
    任一 T/A 词却仍须判 False，正是词表 OR 结构会翻车、合取结构该拦住的那类。
    """
    import re

    literal_re = re.compile(r"点|时间|明天|下午|晚上|星期|周|答应|说好|约")
    for name, (_fn, pos, neg, _miss) in _REGISTRY.items():
        assert len(pos) >= _MIN_POSITIVES, f"{name}: 正例不足 {_MIN_POSITIVES}"
        assert len(neg) >= _MIN_NEGATIVES, f"{name}: 负例不足 {_MIN_NEGATIVES}"
        literal_hits = [t for _label, t in neg if literal_re.search(t)]
        assert len(literal_hits) >= _MIN_LITERAL_HIT_NEGATIVES, (
            f"{name}: 「字面命中但语义不符」负例不足 {_MIN_LITERAL_HIT_NEGATIVES}——"
            "没有这类样本，词表 OR 结构的判据也能全绿，门禁失去区分力"
        )


def test_registry_all_samples_pass() -> None:
    """预注册集逐条通过：正例 True、负例 False、已知漏报维持 False。"""
    failures: list[str] = []
    for name, (fn, pos, neg, miss) in _REGISTRY.items():
        failures += [f"{name} 正例翻车: {label}" for label, t in pos if not fn(t)]
        failures += [f"{name} 负例误报: {label}" for label, t in neg if fn(t)]
        failures += [f"{name} KNOWN_MISS 变绿(先查合取): {label}" for label, t in miss if fn(t)]
    assert not failures, "\n".join(failures)


def test_registry_covers_all_predicates_in_supervisor() -> None:
    """supervisor 里每个 `_is_xxx(text)` 形态的文本判据都须登记（防静默新增绕过门禁）。

    以「模块内以 `_is_` 开头、接受单个 str 参数的可调用」为发现口径；
    `_is_identity_disclosure` 按上述豁免显式列入白名单。
    """
    import inspect

    from src.orchestration import supervisor

    exempt = {"_is_identity_disclosure"}  # 先于本机制的既有专属测试套
    found = {
        name
        for name, obj in vars(supervisor).items()
        if name.startswith("_is_") and inspect.isfunction(obj)
    }
    unregistered = found - set(_REGISTRY) - exempt
    assert not unregistered, (
        f"发现未登记的文本判据 {sorted(unregistered)}——按 text-predicate-admission.md "
        "补预注册样本并登记 _REGISTRY，或加入豁免白名单并说明理由"
    )
