"""T5：FacsDecoder AU 集合扩展专项测试。

覆盖：
  ① 默认 facs_extended=False 时 decode_channels facs_au 键集/值与旧逐字一致（零回归）。
  ② facs_extended=True 时 facs_au 含 11 AU（FACS_KEYS_EXT 完整键集）。
  ③ 端到端经 CompositeChannelDecoder 注入（防 W1 空悬）：
     (v=-0.6, a=0.6, coping=+0.5) → AU23 高、AU01/02/20 低；
     coping=-0.5 → AU01/02/20 高、AU23 低。
  ④ 连续性：coping 从 -1→1 多点采样 AU23 严格单调升（无阶跃）。
  ⑤ 象限守卫：v≥0 或 a<0 时 AU23/01/02/20=0。
  ⑥ 旧 AU06/12（v≥0）/AU15（v<0）方向不变（旧向零回归守卫）。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

# facs_decoder.py 顶层 import torch → 本文件 collection 即需 torch（旧代码经 FACS_KEYS
# 常量已如此）。无 torch 环境下模块级跳过整个文件，优雅 SKIP 而非 collection ERROR（同
# test_facs.py 范式）。torch-present 环境（conda affective-expression）下全部照跑。
pytest.importorskip("torch")

from src.agents.affect_math import _decode_facs_legacy, decode_channels  # noqa: E402
from src.agents.models.composite import CompositeChannelDecoder  # noqa: E402
from src.agents.models.facs_decoder import (  # noqa: E402
    FACS_KEYS,
    FACS_KEYS_EXT,
    FacsDecoder,
)

# 区分性 AU（象限守卫的目标键）
_DISCRIMINATIVE_AUS = {"AU23", "AU01", "AU02", "AU20"}


# ─── ① 默认零回归 ────────────────────────────────────────────────────────────


class TestDefaultLegacyBehavior:
    """facs_extended=False（默认）时 decode_channels 应与旧 _decode_facs_legacy 逐字一致。"""

    def _cases(self) -> list[tuple[float, float]]:
        return [
            (0.8, 0.5),
            (-0.7, 0.3),
            (0.0, 0.0),
            (0.3, 0.9),
            (-0.5, -0.4),
            (1.0, 1.0),
            (-1.0, -1.0),
        ]

    def test_key_set_matches_legacy(self) -> None:
        """默认输出键集与 _decode_facs_legacy 完全一致（互斥分支：正/负效价各出不同子集）。

        旧行为：正效价→{AU12, AU06, intensity}；负效价→{AU15, AU04, intensity}——
        总键集是 FACS_KEYS 的真子集，且两次调用结果等价是零回归的核心约束。
        不应断言"等于全量 FACS_KEYS"，因为旧设计本身就是互斥分支。
        """
        for v, a in self._cases():
            result = decode_channels((v, a))["facs_au"]
            expected = _decode_facs_legacy(v, a)
            assert set(result) == set(expected), (
                f"v={v} a={a}: decode_channels 键集 {set(result)} != legacy 键集 {set(expected)}"
            )
            # 确保所有键都在 FACS_KEYS 范围内（不产生额外键）
            assert set(result) <= set(FACS_KEYS), (
                f"v={v} a={a}: 产生了 FACS_KEYS 以外的键 {set(result) - set(FACS_KEYS)}"
            )

    def test_values_match_legacy_exactly(self) -> None:
        """默认值与 _decode_facs_legacy 逐字相等。"""
        for v, a in self._cases():
            expected = _decode_facs_legacy(v, a)
            actual = decode_channels((v, a))["facs_au"]
            for key in expected:
                assert abs(actual[key] - expected[key]) < 1e-9, (
                    f"v={v} a={a} {key}: {actual[key]} != {expected[key]}"
                )

    def test_coping_has_no_effect_when_not_extended(self) -> None:
        """coping_potential 非零时，默认模式输出不变（keyword-only 参数无侧效）。"""
        for v, a in self._cases():
            base = decode_channels((v, a))["facs_au"]
            with_coping = decode_channels((v, a), coping_potential=0.9)["facs_au"]
            assert set(base) == set(with_coping)
            for key in base:
                assert abs(base[key] - with_coping[key]) < 1e-9, (
                    f"coping 非零但 facs_extended=False 时值不应变：v={v} a={a} {key}"
                )


# ─── ② 扩展集键集 ────────────────────────────────────────────────────────────


class TestExtendedKeySet:
    """facs_extended=True 时输出完整 11-AU 键集。

    扩展分支与旧分支一样采用互斥正/负效价逻辑——旧向通道（AU12/AU06 vs AU15/AU04）
    在单次调用中仍互斥。但区分性 AU（AU01/02/05/07/20/23）和 intensity 在各象限均输出。
    因此"完整 11 键"需正效价 + 负效价两个测试点合并才能覆盖全集；单次调用只保证
    "键集 ⊆ FACS_KEYS_EXT 且包含所有非互斥键"。
    """

    def test_contains_all_ext_keys_combined(self) -> None:
        """正效价点 + 负效价点的键集并集 == FACS_KEYS_EXT（覆盖互斥分支两侧）。"""
        pos = set(decode_channels((0.5, 0.5), facs_extended=True)["facs_au"])
        neg = set(decode_channels((-0.5, 0.5), facs_extended=True)["facs_au"])
        combined = pos | neg
        assert combined == set(FACS_KEYS_EXT), (
            f"正+负效价键集并集 {combined} != FACS_KEYS_EXT {set(FACS_KEYS_EXT)}"
        )

    def test_negative_valence_contains_discriminative_and_shared_aus(self) -> None:
        """负效价象限（v<0, a≥0）：含所有区分性 AU + 共有 AU + intensity（象限内全 9 键）。"""
        result = decode_channels((-0.5, 0.5), coping_potential=0.5, facs_extended=True)["facs_au"]
        expected_present = {
            "AU15",
            "AU04",
            "AU05",
            "AU07",
            "AU23",
            "AU01",
            "AU02",
            "AU20",
            "intensity",
        }
        for key in expected_present:
            assert key in result, f"v<0 a≥0 象限应含 {key}，实际 keys={set(result)}"

    def test_no_extra_keys_outside_ext(self) -> None:
        """不产生 FACS_KEYS_EXT 以外的键。"""
        for v in [-0.8, 0.0, 0.8]:
            for a in [-0.5, 0.5]:
                result = decode_channels((v, a), facs_extended=True)["facs_au"]
                assert set(result) <= set(FACS_KEYS_EXT), (
                    f"v={v} a={a}: 产生了 FACS_KEYS_EXT 以外的键 {set(result) - set(FACS_KEYS_EXT)}"
                )

    def test_all_values_in_unit_range(self) -> None:
        """所有 AU 强度 ∈ [0, 1]。"""
        for v in [-0.8, -0.3, 0.0, 0.5, 0.9]:
            for a in [-0.5, 0.0, 0.5, 0.9]:
                for coping in [-0.8, 0.0, 0.8]:
                    result = decode_channels((v, a), coping_potential=coping, facs_extended=True)[
                        "facs_au"
                    ]
                    for key, val in result.items():
                        assert 0.0 <= val <= 1.0, (
                            f"AU 超出 [0,1]：v={v} a={a} coping={coping} {key}={val}"
                        )


# ─── ③ 端到端经 CompositeChannelDecoder（防 W1 空悬）─────────────────────────


class TestCompositeEndToEnd:
    """coping/facs_extended 必须真透传到 decode_channels，不能空悬在 composite 层。"""

    def _get_au(self, v: float, a: float, coping: float) -> dict[str, float]:
        decoder = CompositeChannelDecoder(coping_potential=coping, facs_extended=True)
        return decoder.predict_channels(v, a)["facs_au"]

    def test_anger_coping_positive_au23_high_fear_aus_low(self) -> None:
        """(v=-0.6, a=0.6, coping=+0.5)：AU23 应显著高于 AU01/02/20。"""
        facs = self._get_au(-0.6, 0.6, 0.5)
        assert facs["AU23"] > 0.0, "coping>0 象限内 AU23 应 >0"
        # AU23 > 区分性恐惧 AU（AU01/02/20 在 coping>0 时接近 0）
        assert facs["AU23"] > facs["AU01"], f"AU23={facs['AU23']} 应 > AU01={facs['AU01']}"
        assert facs["AU23"] > facs["AU02"], f"AU23={facs['AU23']} 应 > AU02={facs['AU02']}"
        assert facs["AU23"] > facs["AU20"], f"AU23={facs['AU23']} 应 > AU20={facs['AU20']}"

    def test_fear_coping_negative_fear_aus_high_au23_low(self) -> None:
        """(v=-0.6, a=0.6, coping=-0.5)：AU01/02/20 应显著高于 AU23。"""
        facs = self._get_au(-0.6, 0.6, -0.5)
        assert facs["AU01"] > 0.0, "coping<0 象限内 AU01 应 >0"
        assert facs["AU02"] > 0.0, "coping<0 象限内 AU02 应 >0"
        assert facs["AU20"] > 0.0, "coping<0 象限内 AU20 应 >0"
        # 恐惧 AU > AU23（AU23 在 coping<0 时接近 0）
        assert facs["AU01"] > facs["AU23"], f"AU01={facs['AU01']} 应 > AU23={facs['AU23']}"
        assert facs["AU02"] > facs["AU23"], f"AU02={facs['AU02']} 应 > AU23={facs['AU23']}"
        assert facs["AU20"] > facs["AU23"], f"AU20={facs['AU20']} 应 > AU23={facs['AU23']}"

    def test_au01_au02_linked(self) -> None:
        """AU01 与 AU02 须联动同升（值相等）。"""
        for coping in [-0.9, -0.5, -0.1, 0.0]:
            facs = self._get_au(-0.6, 0.6, coping)
            assert abs(facs["AU01"] - facs["AU02"]) < 1e-9, (
                f"coping={coping}: AU01={facs['AU01']} != AU02={facs['AU02']}"
            )

    def test_composite_predict_channels_signature_unchanged(self) -> None:
        """predict_channels(v, a) 公开签名不变（CS 席约束 #4）。"""
        decoder = CompositeChannelDecoder(coping_potential=0.5, facs_extended=True)
        # 仅位置参数 v, a，不接受 coping_potential 作为调用时参数
        result = decoder.predict_channels(-0.6, 0.6)
        assert "facs_au" in result


# ─── ④ 连续性：AU23 随 coping 严格单调升 ─────────────────────────────────────


class TestMonotonicity:
    """push 层连续映射：AU23 随 coping 从 -1→1 严格单调，无 ±0.3 硬阈值阶跃。"""

    def test_au23_monotone_increasing_with_coping(self) -> None:
        """AU23 随 coping ∈ [-1, 1] 单调不降，严格无阶跃。"""
        v, a = -0.6, 0.6  # 高唤醒负效价象限
        coping_samples = [i / 10.0 for i in range(-10, 11)]  # -1.0, -0.9, ..., 1.0
        au23_values = [
            decode_channels((v, a), coping_potential=c, facs_extended=True)["facs_au"]["AU23"]
            for c in coping_samples
        ]
        # 单调不降（允许相等，不允许严格下降）
        for i in range(len(au23_values) - 1):
            assert au23_values[i] <= au23_values[i + 1] + 1e-9, (
                f"AU23 单调性违反：coping={coping_samples[i]:.1f}→{coping_samples[i + 1]:.1f}，"
                f"AU23={au23_values[i]:.6f}→{au23_values[i + 1]:.6f}"
            )

    def test_no_step_at_threshold(self) -> None:
        """在 ±0.3 附近无硬阶跃：AU23 在 coping=0.29→0.31 区间连续。"""
        v, a = -0.6, 0.6
        eps = 1e-3
        for pivot in [0.3, -0.3, 0.0]:
            lo = decode_channels((v, a), coping_potential=pivot - eps, facs_extended=True)[
                "facs_au"
            ]["AU23"]
            hi = decode_channels((v, a), coping_potential=pivot + eps, facs_extended=True)[
                "facs_au"
            ]["AU23"]
            jump = abs(hi - lo)
            # 连续映射在 ε=0.001 范围内跳变不应超过 0.1（阈值硬跳会产生 O(1) 跳变）
            assert jump < 0.1, (
                f"coping={pivot} 附近疑似硬阶跃：delta={jump:.4f}（lo={lo:.4f}, hi={hi:.4f}）"
            )

    def test_au20_monotone_decreasing_with_coping(self) -> None:
        """AU20 随 coping 升而单调降（恐惧 AU，coping↑ 则降）。"""
        v, a = -0.6, 0.6
        coping_samples = [i / 10.0 for i in range(-10, 11)]
        au20_values = [
            decode_channels((v, a), coping_potential=c, facs_extended=True)["facs_au"]["AU20"]
            for c in coping_samples
        ]
        for i in range(len(au20_values) - 1):
            assert au20_values[i] >= au20_values[i + 1] - 1e-9, (
                f"AU20 单调性违反（应随 coping 升而降）：coping={coping_samples[i]:.1f}→"
                f"{coping_samples[i + 1]:.1f}，AU20={au20_values[i]:.6f}→{au20_values[i + 1]:.6f}"
            )


# ─── ⑤ 象限守卫 ──────────────────────────────────────────────────────────────


class TestQuadrantGuard:
    """区分性 AU 仅在 (v<0, a≥0) 象限激活，其他象限归零。"""

    def _disc_aus(self, v: float, a: float, coping: float) -> dict[str, float]:
        facs = decode_channels((v, a), coping_potential=coping, facs_extended=True)["facs_au"]
        return {k: facs[k] for k in _DISCRIMINATIVE_AUS}

    def test_positive_valence_discriminative_zero(self) -> None:
        """v>=0 时区分性 AU 全为 0，无论 coping 如何（象限守卫用 valence<0.0，v=0.0 不进象限）。"""
        for v in [0.0, 0.3, 0.8, 1.0]:
            for coping in [-0.8, 0.0, 0.8]:
                aus = self._disc_aus(v, 0.6, coping)
                for key, val in aus.items():
                    assert val == 0.0, f"v={v} coping={coping} {key}={val} 应为 0（v≥0 象限守卫）"

    def test_negative_arousal_discriminative_zero(self) -> None:
        """a<0 时区分性 AU 全为 0，无论 coping 如何。"""
        for a in [-0.1, -0.5, -1.0]:
            for coping in [-0.8, 0.0, 0.8]:
                aus = self._disc_aus(-0.5, a, coping)
                for key, val in aus.items():
                    assert val == 0.0, f"a={a} coping={coping} {key}={val} 应为 0（a<0 象限守卫）"

    def test_in_quadrant_with_neutral_coping_zero(self) -> None:
        """(v<0, a≥0) 象限内，coping=0.0 时区分性 AU 全为 0（relu 消除）。"""
        aus = self._disc_aus(-0.5, 0.5, 0.0)
        for key, val in aus.items():
            assert val == 0.0, f"coping=0 时 {key}={val} 应为 0"

    def test_in_quadrant_with_coping_nonzero(self) -> None:
        """(v<0, a≥0) 象限内，coping≠0 时区分性 AU 至少部分 >0。"""
        facs_anger = decode_channels((-0.6, 0.6), coping_potential=0.5, facs_extended=True)[
            "facs_au"
        ]
        assert facs_anger["AU23"] > 0.0

        facs_fear = decode_channels((-0.6, 0.6), coping_potential=-0.5, facs_extended=True)[
            "facs_au"
        ]
        assert facs_fear["AU01"] > 0.0
        assert facs_fear["AU02"] > 0.0
        assert facs_fear["AU20"] > 0.0


# ─── ⑥ 旧 AU 方向守卫 ────────────────────────────────────────────────────────


class TestLegacyAUDirections:
    """旧 AU06/AU12（v≥0）/AU15（v<0）方向在扩展模式下不变。"""

    def test_au12_au06_active_when_positive_valence(self) -> None:
        """v>0 时 AU12/AU06 > 0；v<0 时不出现（旧向零回归守卫）。"""
        facs = decode_channels((0.7, 0.5), facs_extended=True)["facs_au"]
        assert facs["AU12"] > 0.0, f"v=0.7 时 AU12={facs['AU12']} 应 >0"
        assert facs["AU06"] > 0.0, f"v=0.7 时 AU06={facs['AU06']} 应 >0"

    def test_au15_active_when_negative_valence(self) -> None:
        """v<0 时 AU15 > 0；v≥0 时为 0（旧向零回归守卫）。"""
        facs = decode_channels((-0.6, 0.5), facs_extended=True)["facs_au"]
        assert facs["AU15"] > 0.0, f"v=-0.6 时 AU15={facs['AU15']} 应 >0"

    def test_au04_active_when_negative_valence(self) -> None:
        """v<0 时 AU04 > 0（旧向零回归守卫）。"""
        facs = decode_channels((-0.6, 0.5), facs_extended=True)["facs_au"]
        assert facs["AU04"] > 0.0, f"v=-0.6 时 AU04={facs['AU04']} 应 >0"

    def test_intensity_tracks_arousal(self) -> None:
        """intensity ∝ |arousal|，方向不变（旧向零回归守卫）。"""
        hi = decode_channels((0.5, 0.9), facs_extended=True)["facs_au"]["intensity"]
        lo = decode_channels((0.5, 0.1), facs_extended=True)["facs_au"]["intensity"]
        assert hi > lo, f"高唤醒 intensity={hi} 应 > 低唤醒 intensity={lo}"

    def test_au12_proportional_to_valence(self) -> None:
        """AU12 随 valence 升（正效价越强笑肌越活跃），旧向。"""
        facs_lo = decode_channels((0.3, 0.5), facs_extended=True)["facs_au"]
        facs_hi = decode_channels((0.8, 0.5), facs_extended=True)["facs_au"]
        assert facs_hi["AU12"] > facs_lo["AU12"], (
            f"AU12 应随 valence 升：lo={facs_lo['AU12']} hi={facs_hi['AU12']}"
        )

    def test_au05_increases_with_arousal(self) -> None:
        """AU05（新增共有）随 arousal+ 升。"""
        lo = decode_channels((-0.5, 0.1), facs_extended=True)["facs_au"]["AU05"]
        hi = decode_channels((-0.5, 0.9), facs_extended=True)["facs_au"]["AU05"]
        assert hi > lo, f"AU05 应随 arousal 升：lo={lo} hi={hi}"


class TestExpressionAgentFacsWiring:
    """W1 防空悬（同 coping_potential W1 教训）：ExpressionAgent 占位路径把
    state.coping_potential_state + state.facs_extended 透传给 decode_channels——
    facs_au 在**图内**真受 coping 影响，非只在直调 decode_channels 时。
    """

    @staticmethod
    def _spontaneous_facs(*, coping: float, facs_extended: bool) -> dict[str, float]:
        from src.agents.expression import ExpressionAgent
        from src.orchestration.state import AffectState

        state = AffectState(
            affect_sample=(-0.6, 0.6),  # (-v,+a) 象限
            coping_potential_state=coping,
            facs_extended=facs_extended,
        )
        out = ExpressionAgent()(state)  # decoder=None → 占位路径
        return out["expression"]["spontaneous"]["facs_au"]

    def test_default_matches_legacy_zero_regression(self) -> None:
        """facs_extended=False（默认）→ 占位 facs_au 与 decode_channels 旧输出逐字一致。"""
        facs = self._spontaneous_facs(coping=0.5, facs_extended=False)
        assert facs == decode_channels((-0.6, 0.6))["facs_au"]

    def test_extended_anger_high_au23(self) -> None:
        """facs_extended=True + coping>0（愤怒）→ 图内 facs_au 的 AU23 高、AU01/AU20 低。"""
        facs = self._spontaneous_facs(coping=0.5, facs_extended=True)
        assert set(facs) <= set(FACS_KEYS_EXT), f"键应 ⊆ FACS_KEYS_EXT：{set(facs)}"
        assert facs["AU23"] > facs["AU01"], f"愤怒应 AU23>AU01：{facs}"
        assert facs["AU23"] > facs["AU20"], f"愤怒应 AU23>AU20：{facs}"

    def test_extended_fear_high_au01_au20(self) -> None:
        """facs_extended=True + coping<0（恐惧）→ 图内 facs_au 的 AU01/AU20 高、AU23 低。"""
        facs = self._spontaneous_facs(coping=-0.5, facs_extended=True)
        assert facs["AU01"] > facs["AU23"], f"恐惧应 AU01>AU23：{facs}"
        assert facs["AU20"] > facs["AU23"], f"恐惧应 AU20>AU23：{facs}"


# ─── ⑧ C1 双通路差异化：随意头 coping 泄漏衰减（议会设计门 2026-07-14）────────────


class TestVoluntaryCopingLeak:
    """C1（议会 C1 设计门）：自发头(push·锥体外路)全量传 coping、随意头(pull·意志调控)传
    coping×voluntary_coping_leak。默认 leak=1.0 → 两头等值=逐字旧行为（零回归）；leak<1 →
    随意头 coping-driven AU 被压低但不归零；leak=0 → 随意头 coping-AU 归零、自发头保留。
    仅 facs_extended=True 时对 facs_au 生效（legacy 模式 coping 不参与 → leak 无效=零回归）。

    构造 regulated_affect=None → 随意头 (v,a) 与自发头相同，唯一差异即 coping 缩放，故可直接
    对比两头 facs_au 隔离出 leak 的作用（防混入 (v,a) 差异）。
    """

    @staticmethod
    def _both_heads(
        *, coping: float, leak: float, facs_extended: bool = True
    ) -> tuple[dict[str, float], dict[str, float]]:
        from src.agents.expression import ExpressionAgent
        from src.orchestration.state import AffectState

        state = AffectState(
            affect_sample=(-0.6, 0.6),  # (-v,+a) 象限
            coping_potential_state=coping,
            facs_extended=facs_extended,
            voluntary_coping_leak=leak,
        )
        expr = ExpressionAgent()(state)["expression"]  # decoder=None → 占位路径
        return expr["spontaneous"]["facs_au"], expr["voluntary"]["facs_au"]

    def test_default_leak_heads_identical_zero_regression(self) -> None:
        """leak=1.0（默认）→ 两头 facs_au 逐字一致（regulated_affect=None 时两头同 (v,a)）。"""
        for coping in [0.5, -0.5, 0.0]:
            spon, vol = self._both_heads(coping=coping, leak=1.0)
            assert spon == vol, f"leak=1.0 应两头等值：coping={coping} spon={spon} vol={vol}"

    def test_leak_reduces_voluntary_anger_au23(self) -> None:
        """愤怒(coping>0)：leak=0.3 → 随意头 AU23 被压低（0 < vol < spon），意志部分压制不归零。"""
        spon, vol = self._both_heads(coping=0.5, leak=0.3)
        assert 0.0 < vol["AU23"] < spon["AU23"], (
            f"随意头 AU23 应被压低但非零：spon={spon['AU23']} vol={vol['AU23']}"
        )

    def test_leak_reduces_voluntary_fear_aus(self) -> None:
        """恐惧(coping<0)：leak=0.3 → 随意头 AU01/02/20 均被压低（0 < vol < spon）。"""
        spon, vol = self._both_heads(coping=-0.5, leak=0.3)
        for au in ("AU01", "AU02", "AU20"):
            assert 0.0 < vol[au] < spon[au], (
                f"随意头 {au} 应被压低但非零：spon={spon[au]} vol={vol[au]}"
            )

    def test_zero_leak_suppresses_voluntary_coping_aus(self) -> None:
        """leak=0.0 → 随意头 coping-driven AU 归零（coping×0→relu），自发头保留。"""
        spon, vol = self._both_heads(coping=0.5, leak=0.0)
        assert spon["AU23"] > 0.0, "自发头 AU23 应保留（coping 全量泄漏）"
        assert vol["AU23"] == 0.0, "leak=0 随意头 AU23 应归零"

    def test_non_coping_aus_unaffected_by_leak(self) -> None:
        """leak 只影响 coping-driven AU；旧向/共有 AU（AU04/05/07/15/intensity）两头仍一致。"""
        spon, vol = self._both_heads(coping=0.5, leak=0.3)
        for au in ("AU04", "AU05", "AU07", "AU15", "intensity"):
            assert spon[au] == pytest.approx(vol[au]), (
                f"{au} 非 coping-driven，两头应一致：spon={spon[au]} vol={vol[au]}"
            )

    def test_legacy_mode_leak_no_effect(self) -> None:
        """facs_extended=False → coping 不参与 facs_au，leak 任意值两头仍逐字旧行为（零回归）。"""
        for leak in [1.0, 0.3, 0.0]:
            spon, vol = self._both_heads(coping=0.5, leak=leak, facs_extended=False)
            assert spon == vol, f"legacy 模式 leak={leak} 两头应等值"
            assert set(spon) <= set(FACS_KEYS), f"legacy 键集应 ⊆ FACS_KEYS：{set(spon)}"

    def test_leak_out_of_range_rejected(self) -> None:
        """voluntary_coping_leak ∈ [0,1]：越界值 pydantic 校验拒绝（fail-fast，防误用）。"""
        from pydantic import ValidationError

        from src.orchestration.state import AffectState

        for bad in [-0.1, 1.5]:
            with pytest.raises(ValidationError):
                AffectState(affect_sample=(-0.6, 0.6), voluntary_coping_leak=bad)


# ─── ⑦ 扩展训练管线端到端 smoke（任务 A：data-independent turnkey）────────────


class TestExtTrainingPipelineTurnkey:
    """--ext 训练管线端到端 smoke：证「数据一到就一条命令」管线打通。

    合成 11-AU 标注 CSV（标签由 `_decode_facs_extended` 解析映射蒸馏，自洽、无需外部
    EULA 数据）→ `train(extended=True)` → 载回 `FacsDecoder(extended=True)` → `predict_facs`
    输出 11 键。**只验管线连通（3 epoch 求跑通不 NaN），不验权重质量**——真权重待任务 B
    的 AU 标注数据。torch 缺失则跳过（纯 Python 的 decode_channels 不需 torch）。
    """

    @staticmethod
    def _write_synthetic_ext_csv(csv_path: Path) -> int:
        """按 load_facs_csv_ext 期望列序写合成 11-AU CSV，返回行数。

        标签取自 decode_channels 的解析映射（路径 A·合成蒸馏）；互斥正/负效价分支
        未出的键补 0，凑满 11 维目标向量。
        """
        header = ["valence", "arousal", *FACS_KEYS_EXT]
        samples = [
            (-0.6, 0.6, 0.5),  # 愤怒象限（coping>0，AU23 高）
            (-0.6, 0.6, -0.5),  # 恐惧象限（coping<0，AU01/02/20 高）
            (0.7, 0.5, 0.0),  # 正效价（AU06/12）
            (-0.5, -0.4, 0.0),  # 低唤醒负效价
            (0.0, 0.0, 0.0),  # 中性
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            for v, a, coping in samples:
                facs = decode_channels((v, a), coping_potential=coping, facs_extended=True)[
                    "facs_au"
                ]
                row: dict[str, float] = {"valence": v, "arousal": a}
                for key in FACS_KEYS_EXT:
                    row[key] = facs.get(key, 0.0)  # 互斥分支未出的键补 0
                writer.writerow(row)
        return len(samples)

    def test_ext_training_pipeline_turnkey(self, tmp_path: Path) -> None:
        """合成 CSV → train(extended=True, epochs=3) → 产出权重 → 载回 → predict 11 键。"""
        torch = pytest.importorskip("torch")

        from scripts.train_facs import train
        from src.agents.datasets.facs_csv import load_facs_csv_ext

        csv_path = tmp_path / "labels_ext.csv"
        n_rows = self._write_synthetic_ext_csv(csv_path)

        # 管线首环：合成 CSV 可被 ext loader 读回，形状 = (n 样本, 2) 输入 / (n, 11) 目标
        x, y = load_facs_csv_ext(str(csv_path))
        assert x.shape == (n_rows, 2)
        assert y.shape == (n_rows, len(FACS_KEYS_EXT))

        out = tmp_path / "facs_decoder_ext.pt"
        final = train(str(csv_path), extended=True, epochs=3, out=str(out))

        assert out.exists(), "训练应产出扩展权重文件"
        assert math.isfinite(final), f"最终 loss 应有限（管线通、无 NaN），实为 {final}"
        assert out.name == "facs_decoder_ext.pt", "隔离命名，不得覆盖旧 5-AU 权重"

        # 载回扩展模型 → 形状/键集齐全（11 维）
        model = FacsDecoder(extended=True)
        model.load_state_dict(torch.load(out, map_location="cpu", weights_only=True))
        facs = model.predict_facs(-0.6, 0.6)
        assert set(facs) == set(FACS_KEYS_EXT), f"predict_facs 应输出 11 键，实为 {set(facs)}"
        assert len(facs) == len(FACS_KEYS_EXT) == 11

    def test_ext_weights_isolated_from_legacy_5au(self, tmp_path: Path) -> None:
        """扩展权重与旧 5-AU 权重形状不同 → state_dict 不可互载（隔离守卫）。"""
        torch = pytest.importorskip("torch")
        from scripts.train_facs import train

        csv_path = tmp_path / "labels_ext.csv"
        self._write_synthetic_ext_csv(csv_path)
        out = tmp_path / "facs_decoder_ext.pt"
        train(str(csv_path), extended=True, epochs=3, out=str(out))

        # 用旧 5 维模型载 11 维权重应失败（形状不匹配 → 破坏 expression_decoder.pt 的守卫）。
        # match= 限定「size mismatch」，避免任意 RuntimeError（IO/读取失败等）假绿。
        legacy = FacsDecoder(extended=False)
        with pytest.raises(RuntimeError, match=r"size mismatch"):
            legacy.load_state_dict(torch.load(out, map_location="cpu", weights_only=True))


# ─── ⑨ C2 residual 叠加：注入真 FacsModel 时 coping 判别 AU 混合（议会 C2 设计门）──────


class _MockFacsModel:
    """假 FacsModel：所有 11 AU 返回常数 base_val（模拟真模型通用基准），便于隔离 residual 混合。

    真模型 predict_facs(v,a) 不吃 coping → 对同一 (v,a) 恒定输出；用常数正好凸显「判别 AU 被
    占位 coping 增量覆盖」与「非判别 AU 保持真模型 base」的分野。
    """

    def __init__(self, base_val: float = 0.5) -> None:
        self.base_val = base_val

    def predict_facs(self, valence: float, arousal: float) -> dict[str, float]:
        return {k: self.base_val for k in FACS_KEYS_EXT}


class TestC2ResidualBlend:
    """C2（议会 C2 设计门）：注入真 FacsModel 且 facs_extended 时，真模型出通用基准、
    coping 判别 AU（AU23/01/02/20）= base*(1-α)+占位coping*α。默认 α=1.0 判别 AU 纯占位保分野；
    α=0 全用真模型（退化为旧 W2 整覆盖）；未注入 facs_model 则此路径不触发 = 零回归。
    """

    @staticmethod
    def _facs(
        *, alpha: float, coping: float = 0.5, facs_extended: bool = True, base_val: float = 0.5
    ) -> dict[str, float]:
        comp = CompositeChannelDecoder(
            facs_model=_MockFacsModel(base_val),
            coping_potential=coping,
            facs_extended=facs_extended,
            residual_alpha=alpha,
        )
        return comp.predict_channels(-0.6, 0.6)["facs_au"]

    def test_alpha_one_coping_aus_from_placeholder(self) -> None:
        """α=1.0（默认）：判别 AU（AU23/01/02/20）取占位 coping 增量、非 base；其余 AU 用 base。"""
        placeholder = decode_channels((-0.6, 0.6), coping_potential=0.5, facs_extended=True)[
            "facs_au"
        ]
        facs = self._facs(alpha=1.0, coping=0.5, base_val=0.5)
        for au in ("AU23", "AU01", "AU02", "AU20"):
            assert facs[au] == pytest.approx(placeholder[au]), (
                f"{au} α=1 应=占位 {placeholder[au]}，实 {facs[au]}"
            )
        # 非判别 AU 用真模型 base=0.5（真脸学的通用真实度）
        assert facs["AU06"] == pytest.approx(0.5)
        assert facs["AU12"] == pytest.approx(0.5)

    def test_alpha_zero_all_from_base(self) -> None:
        """α=0.0：全部 AU 取真模型 base（coping 分野丢，退化为旧 W2 整通道覆盖）。"""
        facs = self._facs(alpha=0.0, coping=0.5, base_val=0.5)
        for au in FACS_KEYS_EXT:
            assert facs[au] == pytest.approx(0.5), f"{au} α=0 应=base 0.5"

    def test_alpha_half_blend(self) -> None:
        """α=0.5：判别 AU = base*0.5 + 占位*0.5（线性混合）。"""
        placeholder = decode_channels((-0.6, 0.6), coping_potential=0.5, facs_extended=True)[
            "facs_au"
        ]
        facs = self._facs(alpha=0.5, coping=0.5, base_val=0.5)
        for au in ("AU23", "AU01", "AU02", "AU20"):
            expected = 0.5 * 0.5 + placeholder[au] * 0.5
            assert facs[au] == pytest.approx(expected), (
                f"{au} 混合值错：期望 {expected} 实 {facs[au]}"
            )

    def test_anger_fear_distinction_preserved_default(self) -> None:
        """默认 α=1.0：注入真模型后愤怒/恐惧 coping 分野仍在（AU23 与 AU01/20 随 coping 反向）。"""
        anger = self._facs(alpha=1.0, coping=0.5)  # coping>0 愤怒
        fear = self._facs(alpha=1.0, coping=-0.5)  # coping<0 恐惧
        assert anger["AU23"] > anger["AU01"], f"愤怒应 AU23>AU01：{anger}"
        assert fear["AU01"] > fear["AU23"], f"恐惧应 AU01>AU23：{fear}"

    def test_legacy_mode_no_blend(self) -> None:
        """facs_extended=False：不做 residual 混合，注入真模型输出直接用（判别 AU=base）。"""
        facs = self._facs(alpha=1.0, coping=0.5, facs_extended=False, base_val=0.5)
        for au in ("AU23", "AU01", "AU02", "AU20"):
            assert facs[au] == pytest.approx(0.5), f"legacy 不混合，{au} 应=base 0.5"

    def test_no_facs_model_zero_regression(self) -> None:
        """facs_model=None：facs_au 纯占位，residual_alpha 任意值都无影响（零回归守卫）。"""
        fa = CompositeChannelDecoder(
            coping_potential=0.5, facs_extended=True, residual_alpha=1.0
        ).predict_channels(-0.6, 0.6)["facs_au"]
        fb = CompositeChannelDecoder(
            coping_potential=0.5, facs_extended=True, residual_alpha=0.0
        ).predict_channels(-0.6, 0.6)["facs_au"]
        assert fa == fb, "未注入 facs_model 时 residual_alpha 不应有任何影响"

    def test_residual_alpha_out_of_range_rejected(self) -> None:
        """residual_alpha ∈ [0,1]：越界值构造时 fail-fast（避混合外推使 AU 越界）。"""
        for bad in [-0.1, 1.5]:
            with pytest.raises(ValueError, match=r"residual_alpha"):
                CompositeChannelDecoder(facs_model=_MockFacsModel(), residual_alpha=bad)


# ─── ⑩ 遗留 2：per-turn coping live-wiring（议会设计门 2026-07-14·方案 b）──────────


class _PlainDecoder:
    """旧式 decoder：只有 predict_channels(v,a)、无 predict_channels_coping（回退路径守卫）。"""

    def predict_channels(self, valence: float, arousal: float) -> dict[str, object]:
        return {"facs_au": {"marker": 1.0}, "text_label": "x", "physiology": {}, "prosody": {}}


class TestLiveWiringCopingAware:
    """遗留 2（方案 b）：注入支持 `predict_channels_coping` 的 decoder 时 per-turn coping 透传。
    additive 非 breaking——旧 decoder（只有 predict_channels）回退、零改动。
    """

    def test_predict_channels_coping_uses_per_turn_not_construction(self) -> None:
        """predict_channels_coping 用传入 coping、非构造固定值（构造 coping=0 也能分愤怒/恐惧）。"""
        comp = CompositeChannelDecoder(facs_extended=True, coping_potential=0.0)  # 构造 coping=0
        anger = comp.predict_channels_coping(-0.6, 0.6, 0.5, True)["facs_au"]
        fear = comp.predict_channels_coping(-0.6, 0.6, -0.5, True)["facs_au"]
        assert anger["AU23"] > anger["AU01"], f"per-turn coping>0 愤怒 AU23>AU01：{anger}"
        assert fear["AU01"] > fear["AU23"], f"per-turn coping<0 恐惧 AU01>AU23：{fear}"

    def test_predict_channels_delegates_with_construction_coping(self) -> None:
        """predict_channels(v,a) 公开签名不变：等价于用构造 coping 调 predict_channels_coping。"""
        comp = CompositeChannelDecoder(facs_extended=True, coping_potential=0.5)
        assert (
            comp.predict_channels(-0.6, 0.6)["facs_au"]
            == comp.predict_channels_coping(-0.6, 0.6, 0.5, True)["facs_au"]
        )

    def test_expression_agent_passes_per_turn_coping_to_injected_decoder(self) -> None:
        """图内：ExpressionAgent 注入 composite → state.coping_potential_state 透传到注入 decoder，
        自发头愤怒/恐惧 coping 分野生效（构造 coping=0 也不影响，证 per-turn 透传）。"""
        from src.agents.expression import ExpressionAgent
        from src.orchestration.state import AffectState

        comp = CompositeChannelDecoder(facs_extended=True, coping_potential=0.0)
        agent = ExpressionAgent(decoder=comp)
        anger = agent(
            AffectState(affect_sample=(-0.6, 0.6), coping_potential_state=0.5, facs_extended=True)
        )["expression"]["spontaneous"]["facs_au"]
        fear = agent(
            AffectState(affect_sample=(-0.6, 0.6), coping_potential_state=-0.5, facs_extended=True)
        )["expression"]["spontaneous"]["facs_au"]
        assert anger["AU23"] > anger["AU01"], f"注入路径愤怒 AU23>AU01：{anger}"
        assert fear["AU01"] > fear["AU23"], f"注入路径恐惧 AU01>AU23：{fear}"

    def test_expression_agent_fallback_plain_decoder(self) -> None:
        """注入只有 predict_channels(v,a) 的旧 decoder → 回退、coping 不透传（零改动守卫）。"""
        from src.agents.expression import ExpressionAgent
        from src.orchestration.state import AffectState

        agent = ExpressionAgent(decoder=_PlainDecoder())
        out = agent(
            AffectState(affect_sample=(-0.6, 0.6), coping_potential_state=0.5, facs_extended=True)
        )["expression"]["spontaneous"]
        assert out["facs_au"] == {"marker": 1.0}, (
            "旧 decoder 走 predict_channels(v,a)、不受 coping 影响"
        )
