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

from src.agents.affect_math import _decode_facs_legacy, decode_channels
from src.agents.models.composite import CompositeChannelDecoder
from src.agents.models.facs_decoder import FACS_KEYS, FACS_KEYS_EXT

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
