"""T8：physiology 占位迁 canonical 专项测试（zero-link 任务②·PRP 验证清单）。

覆盖：
  1. 门关零回归四锚点（默认·不设 env）：
     - ① decode_channels physiology 键集 == {heart_rate_bpm, skin_conductance, pupil_mm}
     - ② 三值逐字数值（70+40·clamp / clamp(|a|) / 3+2·clamp）
     - ③ vector_to_channels(affect_to_vector(v,a)) 含 pupil_mm、无 temperature_c
     - ④ CompositeChannelDecoder()（无注入·门关）回退 legacy
  2. 门开 canonical（canonical_physiology=True 直传）：
     - 键集 == {heart_rate_bpm, skin_conductance, temperature_c}、无 pupil_mm
     - 三值域 [50,120]/[0,20]/[33,36]
     - 温度随 |arousal| 单调降（arousal=0→36、|arousal|=1→33）
     - hr 双向（arousal=-1→50、0→85、1→120）
     - sc == 20·clamp(|arousal|)
  3. 两路径同步（decode_channels vs vector_to_channels 键集一致·门开）
  4. 两 fallback 路径一致（路径 A decoder=None vs 路径 B CompositeChannelDecoder 无模型）
  5. 真模型优先（importorskip torch·physiology 走真模型不受 canonical_physiology 影响）

torch 在 5 组时才用，其余组 torch-free。
"""

from __future__ import annotations

import pytest

from src.agents.affect_math import clamp, decode_channels
from src.agents.expression import ExpressionAgent
from src.agents.models.composite import CompositeChannelDecoder
from src.orchestration.state import AffectState

# ── 测试锚点用例（多组覆盖正/负/零象限） ──────────────────────────────────────
_VA_CASES: list[tuple[float, float]] = [
    (0.0, 0.0),
    (0.8, 0.5),
    (-0.7, 0.6),
    (0.3, -0.4),
    (-0.5, -0.5),
    (1.0, 1.0),
    (-1.0, -1.0),
    (0.0, 1.0),
    (0.0, -1.0),
]

_LEGACY_KEYS = {"heart_rate_bpm", "skin_conductance", "pupil_mm"}
_CANONICAL_KEYS = {"heart_rate_bpm", "skin_conductance", "temperature_c"}


# ── 1.① 门关键集零回归 ────────────────────────────────────────────────────────


class TestLegacyKeySet:
    """门关（默认 canonical_physiology=False）physiology 键集逐字 legacy。"""

    def test_default_physiology_key_set(self) -> None:
        for v, a in _VA_CASES:
            physio = decode_channels((v, a))["physiology"]
            assert set(physio) == _LEGACY_KEYS, f"v={v} a={a}: 键集 {set(physio)} != {_LEGACY_KEYS}"

    def test_explicit_false_physiology_key_set(self) -> None:
        for v, a in _VA_CASES:
            physio = decode_channels((v, a), canonical_physiology=False)["physiology"]
            assert set(physio) == _LEGACY_KEYS, f"v={v} a={a}: 键集 {set(physio)} != {_LEGACY_KEYS}"

    def test_no_temperature_c_in_legacy(self) -> None:
        for v, a in _VA_CASES:
            physio = decode_channels((v, a))["physiology"]
            assert "temperature_c" not in physio, f"v={v} a={a}: legacy 不应含 temperature_c"


# ── 1.② 门关数值逐字零回归 ────────────────────────────────────────────────────


class TestLegacyValues:
    """门关三值与 legacy 公式逐字一致（pytest.approx）。"""

    def test_heart_rate_legacy_formula(self) -> None:
        for v, a in _VA_CASES:
            physio = decode_channels((v, a))["physiology"]
            expected_hr = 70.0 + 40.0 * clamp(a, 0.0, 1.0)
            assert physio["heart_rate_bpm"] == pytest.approx(expected_hr, abs=1e-9), (
                f"v={v} a={a}: hr={physio['heart_rate_bpm']} expected={expected_hr}"
            )

    def test_skin_conductance_legacy_formula(self) -> None:
        for v, a in _VA_CASES:
            physio = decode_channels((v, a))["physiology"]
            expected_sc = clamp(abs(a), 0.0, 1.0)
            assert physio["skin_conductance"] == pytest.approx(expected_sc, abs=1e-9), (
                f"v={v} a={a}: sc={physio['skin_conductance']} expected={expected_sc}"
            )

    def test_pupil_mm_legacy_formula(self) -> None:
        for v, a in _VA_CASES:
            physio = decode_channels((v, a))["physiology"]
            expected_pupil = 3.0 + 2.0 * clamp(a, 0.0, 1.0)
            assert physio["pupil_mm"] == pytest.approx(expected_pupil, abs=1e-9), (
                f"v={v} a={a}: pupil_mm={physio['pupil_mm']} expected={expected_pupil}"
            )

    def test_legacy_hr_range(self) -> None:
        """legacy hr 值域 [70, 110]（arousal 只取正半）。"""
        for v, a in _VA_CASES:
            hr = decode_channels((v, a))["physiology"]["heart_rate_bpm"]
            assert 70.0 <= hr <= 110.0, f"v={v} a={a}: hr={hr} 超出 [70,110]"

    def test_legacy_sc_range(self) -> None:
        """legacy sc 值域 [0, 1]（无量纲）。"""
        for v, a in _VA_CASES:
            sc = decode_channels((v, a))["physiology"]["skin_conductance"]
            assert 0.0 <= sc <= 1.0, f"v={v} a={a}: sc={sc} 超出 [0,1]"

    def test_legacy_pupil_range(self) -> None:
        """legacy pupil_mm 值域 [3, 5] mm。"""
        for v, a in _VA_CASES:
            p = decode_channels((v, a))["physiology"]["pupil_mm"]
            assert 3.0 <= p <= 5.0, f"v={v} a={a}: pupil_mm={p} 超出 [3,5]"


# ── 1.③ 蒸馏向量路径门关零回归 ──────────────────────────────────────────────


class TestVectorPathLegacy:
    """vector_to_channels(affect_to_vector(v,a)) 门关时含 pupil_mm、无 temperature_c。"""

    def test_vector_path_legacy_key_set(self) -> None:
        torch = pytest.importorskip("torch")  # noqa: F841
        from src.agents.models.expression_decoder import affect_to_vector, vector_to_channels

        for v, a in _VA_CASES:
            physio = vector_to_channels(affect_to_vector(v, a))["physiology"]
            assert set(physio) == _LEGACY_KEYS, (
                f"v={v} a={a}: 向量路径门关键集 {set(physio)} != {_LEGACY_KEYS}"
            )

    def test_vector_path_legacy_no_temperature_c(self) -> None:
        pytest.importorskip("torch")
        from src.agents.models.expression_decoder import affect_to_vector, vector_to_channels

        for v, a in _VA_CASES:
            physio = vector_to_channels(affect_to_vector(v, a))["physiology"]
            assert "temperature_c" not in physio

    def test_vector_path_values_match_analytic_legacy(self) -> None:
        """向量路径与解析占位数值一致（legacy）——验证蒸馏旁路源码一致（guide:140）。"""
        pytest.importorskip("torch")
        from src.agents.models.expression_decoder import affect_to_vector, vector_to_channels

        for v, a in _VA_CASES:
            analytic = decode_channels((v, a))["physiology"]
            vec_physio = vector_to_channels(affect_to_vector(v, a))["physiology"]
            for key in analytic:
                assert vec_physio[key] == pytest.approx(analytic[key], abs=1e-5), (
                    f"v={v} a={a} key={key}: vec={vec_physio[key]} analytic={analytic[key]}"
                )


# ── 1.④ CompositeChannelDecoder 无注入门关回退 legacy ──────────────────────


class TestCompositeDefaultLegacy:
    """CompositeChannelDecoder()（无 physiology_model·flag 默认 False）回退 legacy。"""

    def test_composite_default_physiology_key_set(self) -> None:
        decoder = CompositeChannelDecoder()
        for v, a in _VA_CASES:
            physio = decoder.predict_channels(v, a)["physiology"]
            assert set(physio) == _LEGACY_KEYS, (
                f"v={v} a={a}: Composite 默认键集 {set(physio)} != {_LEGACY_KEYS}"
            )

    def test_composite_default_legacy_values(self) -> None:
        decoder = CompositeChannelDecoder()
        for v, a in _VA_CASES:
            physio = decoder.predict_channels(v, a)["physiology"]
            analytic = decode_channels((v, a))["physiology"]
            for key in analytic:
                assert physio[key] == pytest.approx(analytic[key], abs=1e-9), (
                    f"v={v} a={a} key={key}: Composite={physio[key]} analytic={analytic[key]}"
                )

    def test_composite_explicit_false_legacy(self) -> None:
        decoder = CompositeChannelDecoder(canonical_physiology=False)
        for v, a in _VA_CASES:
            physio = decoder.predict_channels(v, a)["physiology"]
            assert set(physio) == _LEGACY_KEYS


# ── 2. 门开 canonical ─────────────────────────────────────────────────────────


class TestCanonicalKeySet:
    """门开（canonical_physiology=True）键集 == {heart_rate_bpm, skin_conductance, temperature_c}。"""  # noqa: E501

    def test_canonical_key_set(self) -> None:
        for v, a in _VA_CASES:
            physio = decode_channels((v, a), canonical_physiology=True)["physiology"]
            assert set(physio) == _CANONICAL_KEYS, (
                f"v={v} a={a}: canonical 键集 {set(physio)} != {_CANONICAL_KEYS}"
            )

    def test_canonical_no_pupil_mm(self) -> None:
        for v, a in _VA_CASES:
            physio = decode_channels((v, a), canonical_physiology=True)["physiology"]
            assert "pupil_mm" not in physio, f"v={v} a={a}: canonical 不应含 pupil_mm"


class TestCanonicalValues:
    """门开 canonical 三值域 + 公式正确性。"""

    def test_canonical_hr_range(self) -> None:
        """hr 值域 [50, 120] bpm（全域·含负高唤醒）。"""
        for v, a in _VA_CASES:
            hr = decode_channels((v, a), canonical_physiology=True)["physiology"]["heart_rate_bpm"]
            assert 50.0 <= hr <= 120.0, f"v={v} a={a}: hr={hr} 超出 [50,120]"

    def test_canonical_sc_range(self) -> None:
        """sc 值域 [0, 20] μS。"""
        for v, a in _VA_CASES:
            sc = decode_channels((v, a), canonical_physiology=True)["physiology"][
                "skin_conductance"
            ]
            assert 0.0 <= sc <= 20.0, f"v={v} a={a}: sc={sc} 超出 [0,20]μS"

    def test_canonical_temperature_range(self) -> None:
        """temperature_c 值域 [33, 36] °C。"""
        for v, a in _VA_CASES:
            temp = decode_channels((v, a), canonical_physiology=True)["physiology"]["temperature_c"]
            assert 33.0 <= temp <= 36.0, f"v={v} a={a}: temp={temp} 超出 [33,36]°C"

    def test_canonical_hr_formula(self) -> None:
        """hr == 50 + 70·clamp(0.5·(1+arousal), 0, 1)（逐字公式）。"""
        for v, a in _VA_CASES:
            hr = decode_channels((v, a), canonical_physiology=True)["physiology"]["heart_rate_bpm"]
            expected = 50.0 + 70.0 * clamp(0.5 * (1.0 + a), 0.0, 1.0)
            assert hr == pytest.approx(expected, abs=1e-9), (
                f"v={v} a={a}: hr={hr} expected={expected}"
            )

    def test_canonical_sc_formula(self) -> None:
        """sc == 20·clamp(|arousal|, 0, 1)。"""
        for v, a in _VA_CASES:
            sc = decode_channels((v, a), canonical_physiology=True)["physiology"][
                "skin_conductance"
            ]
            expected = 20.0 * clamp(abs(a), 0.0, 1.0)
            assert sc == pytest.approx(expected, abs=1e-9), (
                f"v={v} a={a}: sc={sc} expected={expected}"
            )

    def test_canonical_temperature_formula(self) -> None:
        """temperature_c == 36 - 3·clamp(|arousal|, 0, 1)。"""
        for v, a in _VA_CASES:
            temp = decode_channels((v, a), canonical_physiology=True)["physiology"]["temperature_c"]
            expected = 36.0 - 3.0 * clamp(abs(a), 0.0, 1.0)
            assert temp == pytest.approx(expected, abs=1e-9), (
                f"v={v} a={a}: temp={temp} expected={expected}"
            )


class TestCanonicalDirectionality:
    """温度随 |arousal| 单调降；hr 双向；sc 绝对值对称。"""

    def test_temperature_monotone_decreasing_with_abs_arousal(self) -> None:
        """arousal=0 → temp=36；|arousal|=1 → temp=33；中间严格单调降。"""
        v = 0.0
        prev_temp = None
        for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
            temp = decode_channels((v, a), canonical_physiology=True)["physiology"]["temperature_c"]
            if prev_temp is not None:
                assert temp <= prev_temp, (
                    f"arousal={a}: temp={temp} 未单调降（上一值 prev={prev_temp}）"
                )
            prev_temp = temp

    def test_temperature_rest_36(self) -> None:
        """arousal=0 → temperature_c=36（静息值）。"""
        temp = decode_channels((0.0, 0.0), canonical_physiology=True)["physiology"]["temperature_c"]
        assert temp == pytest.approx(36.0, abs=1e-9)

    def test_temperature_max_arousal_33(self) -> None:
        """|arousal|=1 → temperature_c=33（应激极值）。"""
        temp_pos = decode_channels((0.0, 1.0), canonical_physiology=True)["physiology"][
            "temperature_c"
        ]
        temp_neg = decode_channels((0.0, -1.0), canonical_physiology=True)["physiology"][
            "temperature_c"
        ]
        assert temp_pos == pytest.approx(33.0, abs=1e-9)
        assert temp_neg == pytest.approx(33.0, abs=1e-9)

    def test_temperature_symmetric_on_abs_arousal(self) -> None:
        """temperature_c 仅依赖 |arousal|，正负对称（无 valence 分野·议会裁决）。"""
        cases = [(0.3, 0.6), (0.0, 0.4), (-0.8, 0.9)]
        for v, a in cases:
            temp_pos = decode_channels((v, a), canonical_physiology=True)["physiology"][
                "temperature_c"
            ]
            temp_neg = decode_channels((v, -a), canonical_physiology=True)["physiology"][
                "temperature_c"
            ]
            assert temp_pos == pytest.approx(temp_neg, abs=1e-9), f"v={v}: arousal=±{a} 温度不对称"

    def test_hr_bidirectional(self) -> None:
        """hr 双向全域覆盖：arousal=-1→50、0→85、1→120（Ekman/Levenson 1983）。"""
        hr_neg1 = decode_channels((0.0, -1.0), canonical_physiology=True)["physiology"][
            "heart_rate_bpm"
        ]
        hr_zero = decode_channels((0.0, 0.0), canonical_physiology=True)["physiology"][
            "heart_rate_bpm"
        ]
        hr_pos1 = decode_channels((0.0, 1.0), canonical_physiology=True)["physiology"][
            "heart_rate_bpm"
        ]
        assert hr_neg1 == pytest.approx(50.0, abs=1e-9)
        assert hr_zero == pytest.approx(85.0, abs=1e-9)
        assert hr_pos1 == pytest.approx(120.0, abs=1e-9)

    def test_sc_abs_arousal_symmetric(self) -> None:
        """sc 依赖 |arousal|：正负 arousal 等 sc（B-2 精神）。"""
        for a in [0.3, 0.6, 1.0]:
            sc_pos = decode_channels((0.0, a), canonical_physiology=True)["physiology"][
                "skin_conductance"
            ]
            sc_neg = decode_channels((0.0, -a), canonical_physiology=True)["physiology"][
                "skin_conductance"
            ]
            assert sc_pos == pytest.approx(sc_neg, abs=1e-9), f"arousal=±{a} sc 不对称"

    def test_sc_rest_zero(self) -> None:
        """arousal=0 → sc=0（占位中立无偏置·注释说明系统性低于真 decoder ~10μS）。"""
        sc = decode_channels((0.0, 0.0), canonical_physiology=True)["physiology"][
            "skin_conductance"
        ]
        assert sc == pytest.approx(0.0, abs=1e-9)


# ── 3. 两路径同步（guide:140·CS NEEDS-CHANGES） ──────────────────────────────


class TestTwoPathSync:
    """门开时 decode_channels 与 vector_to_channels(affect_to_vector(...)) 同步。"""

    def test_canonical_key_set_sync(self) -> None:
        """门开两路径 physiology 键集一致。"""
        pytest.importorskip("torch")
        from src.agents.models.expression_decoder import affect_to_vector, vector_to_channels

        for v, a in _VA_CASES:
            analytic_keys = set(decode_channels((v, a), canonical_physiology=True)["physiology"])
            vec_keys = set(
                vector_to_channels(
                    affect_to_vector(v, a, canonical_physiology=True), canonical_physiology=True
                )["physiology"]
            )
            assert analytic_keys == vec_keys, (
                f"v={v} a={a}: analytic={analytic_keys} vec={vec_keys}"
            )

    def test_canonical_temperature_domain_sync(self) -> None:
        """门开两路径 temperature_c 同域、同方向。"""
        pytest.importorskip("torch")
        from src.agents.models.expression_decoder import affect_to_vector, vector_to_channels

        for v, a in _VA_CASES:
            analytic_temp = decode_channels((v, a), canonical_physiology=True)["physiology"][
                "temperature_c"
            ]
            vec_temp = vector_to_channels(
                affect_to_vector(v, a, canonical_physiology=True), canonical_physiology=True
            )["physiology"]["temperature_c"]
            # 两路径 temperature_c 在 [33,36]
            assert 33.0 <= analytic_temp <= 36.0
            assert 33.0 <= vec_temp <= 36.0
            # 两路径代数恒等（vector 归一 (36−3|a|−30)/10 → 反归一 30+10·vec 消回 36−3|a|），
            # 纯 Python float64 往返 → 默认 approx(rel=1e-6) 即足；松容差会漏掉真失同步（WARN-2）。
            assert analytic_temp == pytest.approx(vec_temp), (
                f"v={v} a={a}: analytic_temp={analytic_temp} vec_temp={vec_temp}"
            )

    def test_legacy_key_set_sync(self) -> None:
        """门关两路径 physiology 键集一致。"""
        pytest.importorskip("torch")
        from src.agents.models.expression_decoder import affect_to_vector, vector_to_channels

        for v, a in _VA_CASES:
            analytic_keys = set(decode_channels((v, a))["physiology"])
            vec_keys = set(vector_to_channels(affect_to_vector(v, a))["physiology"])
            assert analytic_keys == vec_keys, (
                f"v={v} a={a}: 门关 analytic={analytic_keys} vec={vec_keys}"
            )


# ── 4. 两 fallback 路径一致（路径 A vs 路径 B） ──────────────────────────────


class TestTwoFallbackPathsConsistency:
    """路径 A（ExpressionAgent decoder=None）与路径 B（CompositeChannelDecoder 无 physiology_model）
    门开时输出同键集/同量纲。"""

    def test_canonical_path_b_key_set(self) -> None:
        """路径 B：CompositeChannelDecoder(canonical=True) 无 physiology_model → canonical。"""
        decoder = CompositeChannelDecoder(canonical_physiology=True)
        for v, a in _VA_CASES:
            physio = decoder.predict_channels(v, a)["physiology"]
            assert set(physio) == _CANONICAL_KEYS, (
                f"v={v} a={a}: 路径B 键集 {set(physio)} != {_CANONICAL_KEYS}"
            )

    def test_canonical_path_b_values(self) -> None:
        """路径 B canonical 三值与 decode_channels 一致。"""
        decoder = CompositeChannelDecoder(canonical_physiology=True)
        for v, a in _VA_CASES:
            physio_b = decoder.predict_channels(v, a)["physiology"]
            analytic = decode_channels((v, a), canonical_physiology=True)["physiology"]
            for key in analytic:
                assert physio_b[key] == pytest.approx(analytic[key], abs=1e-9), (
                    f"v={v} a={a} key={key}: 路径B={physio_b[key]} analytic={analytic[key]}"
                )

    def test_canonical_path_a_key_set(self) -> None:
        """路径 A：ExpressionAgent(decoder=None) + state flag=True → canonical 键集。"""
        agent = ExpressionAgent()  # decoder=None 占位路径
        for v, a in _VA_CASES:
            state = AffectState(affect_sample=(v, a), canonical_physiology=True)
            expr = agent(state)
            physio = expr["expression"]["spontaneous"]["physiology"]
            assert set(physio) == _CANONICAL_KEYS, (
                f"v={v} a={a}: 路径A 键集 {set(physio)} != {_CANONICAL_KEYS}"
            )

    def test_canonical_path_a_values(self) -> None:
        """路径 A canonical 三值与 decode_channels 一致。"""
        agent = ExpressionAgent()
        for v, a in _VA_CASES:
            state = AffectState(affect_sample=(v, a), canonical_physiology=True)
            physio_a = agent(state)["expression"]["spontaneous"]["physiology"]
            analytic = decode_channels((v, a), canonical_physiology=True)["physiology"]
            for key in analytic:
                assert physio_a[key] == pytest.approx(analytic[key], abs=1e-9), (
                    f"v={v} a={a} key={key}: 路径A={physio_a[key]} analytic={analytic[key]}"
                )

    def test_path_a_and_b_key_set_consistent(self) -> None:
        """路径 A 与路径 B 门开时键集相同（同源不变式）。"""
        agent = ExpressionAgent()
        decoder = CompositeChannelDecoder(canonical_physiology=True)
        for v, a in _VA_CASES:
            state = AffectState(affect_sample=(v, a), canonical_physiology=True)
            keys_a = set(agent(state)["expression"]["spontaneous"]["physiology"])
            keys_b = set(decoder.predict_channels(v, a)["physiology"])
            assert keys_a == keys_b, f"v={v} a={a}: 路径A键集={keys_a} 路径B键集={keys_b}"

    def test_path_a_and_b_values_consistent(self) -> None:
        """路径 A 与路径 B 门开时数值一致（同源不变式）。"""
        agent = ExpressionAgent()
        decoder = CompositeChannelDecoder(canonical_physiology=True)
        for v, a in _VA_CASES:
            state = AffectState(affect_sample=(v, a), canonical_physiology=True)
            physio_a = agent(state)["expression"]["spontaneous"]["physiology"]
            physio_b = decoder.predict_channels(v, a)["physiology"]
            for key in physio_a:
                assert physio_a[key] == pytest.approx(physio_b[key], abs=1e-9), (
                    f"v={v} a={a} key={key}: 路径A={physio_a[key]} 路径B={physio_b[key]}"
                )

    def test_legacy_path_a_key_set(self) -> None:
        """路径 A 门关（默认）→ legacy 键集（零回归）。"""
        agent = ExpressionAgent()
        for v, a in _VA_CASES:
            state = AffectState(affect_sample=(v, a))  # canonical_physiology=False 默认
            physio = agent(state)["expression"]["spontaneous"]["physiology"]
            assert set(physio) == _LEGACY_KEYS, (
                f"v={v} a={a}: 路径A 门关键集 {set(physio)} != {_LEGACY_KEYS}"
            )

    def test_legacy_path_b_key_set(self) -> None:
        """路径 B 门关（默认）→ legacy 键集（零回归）。"""
        decoder = CompositeChannelDecoder()
        for v, a in _VA_CASES:
            physio = decoder.predict_channels(v, a)["physiology"]
            assert set(physio) == _LEGACY_KEYS, (
                f"v={v} a={a}: 路径B 门关键集 {set(physio)} != {_LEGACY_KEYS}"
            )


# ── 5. 真模型优先（importorskip torch） ──────────────────────────────────────


class TestRealModelPriority:
    """注入真 PhysiologyDecoder 时 physiology 走真模型（canonical 口径），
    不受 canonical_physiology 占位 flag 影响——flag 只管占位回退路径。"""

    def test_real_physiology_model_takes_priority(self) -> None:
        """注入真 PhysiologyDecoder → physiology 出真 canonical 键集（非占位 legacy）。

        canonical_physiology 默认 False 时注入真模型也应出 canonical 量纲（flag 只管占位路径）。
        """
        torch = pytest.importorskip("torch")  # noqa: F841
        from src.agents.models.physiology_decoder import PhysiologyDecoder

        real_model = PhysiologyDecoder()
        decoder = CompositeChannelDecoder(physiology_model=real_model, canonical_physiology=False)
        physio = decoder.predict_channels(0.5, 0.5)["physiology"]
        # 真 PhysiologyDecoder 出 {hr, sc(μS), temperature_c}——canonical 量纲（WESAD 训练域）
        assert set(physio) == _CANONICAL_KEYS, (
            f"真模型注入时键集 {set(physio)} 应 == {_CANONICAL_KEYS}"
        )
        # 值域检查（sigmoid 输出反归一化域）
        assert 50.0 <= physio["heart_rate_bpm"] <= 120.0
        assert 0.0 <= physio["skin_conductance"] <= 20.0
        assert 30.0 <= physio["temperature_c"] <= 40.0

    def test_real_model_canonical_flag_no_effect(self) -> None:
        """canonical=True 时注入真模型 → 输出仍由真模型决定（flag 对真模型路径无意义）。"""
        pytest.importorskip("torch")
        from src.agents.models.physiology_decoder import PhysiologyDecoder

        real_model = PhysiologyDecoder()
        # 两种 flag 下输出相同（真模型路径绕过占位）
        decoder_f = CompositeChannelDecoder(physiology_model=real_model, canonical_physiology=False)
        decoder_t = CompositeChannelDecoder(physiology_model=real_model, canonical_physiology=True)
        physio_f = decoder_f.predict_channels(0.3, 0.4)["physiology"]
        physio_t = decoder_t.predict_channels(0.3, 0.4)["physiology"]
        assert set(physio_f) == set(physio_t) == _CANONICAL_KEYS
        # 同一模型同一输入 → 数值相同
        for key in physio_f:
            assert physio_f[key] == pytest.approx(physio_t[key], abs=1e-9)
