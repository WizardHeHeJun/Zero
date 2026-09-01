"""T7 外部多模态先验流注入口：单测 + 集成回归（M1–M6 · PRP 2026-07-15）。

覆盖范围：
  1. expand_external_priors 纯函数直测
     - 空列表 → []
     - M2：physio 类 name Πv 强制 MIN_PRECISION；非生理原样透传
     - M3：精度 <=0 / >cap / 形状不合法 → ValueError；边界值通过
     - M7：μ 域校验（议会 2026-07-28）——越界/NaN → ValueError；[-1,1] 边界通过
     - M6：len>max_streams → ValueError；==max_streams 通过
  2. AffectCore 节点集成
     - 零回归：external_priors=[]（默认）→ 返回 dict 与无该字段时 key/集合一致
     - M1：注入非冲突外部流 → 后验受影响、arousal 受高精度流拉动
     - 不污染：注入 external_priors 不改 prior_mu/reward/features 字段
     - 契约：AffectCoreAgent 返回 dict 不含 external_priors key
     - 默认关无 workspace：external_priors 在非 workspace 路径不被展开（结果与空一致）
  3. 跨轮不残留（ConversationSession + state_overrides）
     - 第一轮注入外部流、第二轮不注入 → 第二轮结果与完全不注入对照相同
  4. 风险 #125 seeking 回归（连续多轮高 arousal 外部流）
     - 连续 K=10 轮注入高 arousal 外部流，断言 emotion/attitude 不单调锁定 seeking 象限
  5. ⚠ 点燃门可达性特征化（议会 2026-07-28）
     - 用 design.md §五 **推荐默认精度**（非夸大值）驱动 ignite，锁定当前已知缺陷行为：
       physio 结构性不可点燃、face/audio 真实最强样本仍亚阈、单维饱和三模态均不可达
     - 🛑 **该「应变红」预告已作废（2026-07-29）**：按轴加权马氏距离 + θ'=0.28 那套方案
       **从未落地**（全仓 grep `theta_prime`/`mahalanobis` 在 `src/` 零命中，
       `SALIENCE_THRESHOLD` 仍是 0.18）。PR #46 落的是另外两个方案——线 A（硬门从数值通路
       摘出，`ZERO_IGNITION_GATE_FUSION`）+ 线 B（精度量纲齐次化），二者默认关。
       故本组**不会**因该修复变红；真正会动它的是「翻默认」或「改 SALIENCE_THRESHOLD」
     - 反面（同一主题的另一半）：Πa 抬到**现算**的界即可在生产默认路径上自行过门
       —— 曝露面而非期望行为，见 TestPhysioSelfIgniteExposureOnDefaultPath
  6. AffectState 默认值断言
     - external_prior_precision_cap==0.8、max_external_streams==5、external_priors==[]
  7. _PHYSIO_PREFIXES 派生的前缀覆盖抽样
"""

from __future__ import annotations

import pytest

from src.agents.affect_core import AffectCoreAgent
from src.agents.affect_math import (
    _PHYSIO_PREFIXES,
    GAMMA_PHYSIO,
    MIN_PRECISION,
    SALIENCE_THRESHOLD,
    expand_external_priors,
    hierarchical_fuse,
    stream_salience,
)
from src.orchestration.external_prior import ExternalPriorError
from src.orchestration.state import AffectState, Stimulus

# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------


def _base_state(
    *,
    prior_mu: tuple[float, float] = (0.3, 0.5),
    prior_sigma: tuple[float, float] = (0.2, 0.2),
    **kw: object,
) -> AffectState:
    """构造带前置条件的 AffectState（workspace_enabled=True，AffectCore 需要的字段齐备）。

    prior_mu/prior_sigma 作为具名参数单独提出，避免 **kw 与硬编码默认重复键冲突。
    其余覆盖通过 **kw 注入（external_priors、features 等）。
    """
    return AffectState(
        stimulus=Stimulus(name="t", goal_congruence=0.5, intensity=0.8),
        features=[0.5, 0.0, 0.0, 0.8],
        prior_mu=prior_mu,
        prior_sigma=prior_sigma,
        reward=0.5,
        rpe=0.2,
        precision=0.6,
        rng_seed=42,
        workspace_enabled=True,
        affect_readout="map",  # MAP 读出让断言确定、无随机噪声
        **kw,
    )


# --------------------------------------------------------------------------
# 1. expand_external_priors 纯函数直测
# --------------------------------------------------------------------------


class TestExpandExternalPriorsEmpty:
    """空列表 → 返回 []。"""

    def test_empty_list_returns_empty(self) -> None:
        result = expand_external_priors([], precision_cap=0.8, max_streams=5)
        assert result == []

    def test_empty_list_is_list(self) -> None:
        result = expand_external_priors([], precision_cap=0.8, max_streams=5)
        assert isinstance(result, list)


class TestExpandExternalPriorsM2Physio:
    """M2：physio 类 name Πv 强制 MIN_PRECISION；非生理流 Πv 原样透传。"""

    @pytest.mark.parametrize(
        "name",
        [
            "physio_eda",
            "physio_hrv",
            "physio_anything",
            "eda",
            "eda_signal",
            "hrv",
            "hrv_mean",
            "pupil",
            "pupil_size",
            "scr",
            "scr_amplitude",
        ],
    )
    def test_physio_prefix_forces_pi_v_to_min_precision(self, name: str) -> None:
        """name 以 _PHYSIO_PREFIXES 任一前缀开头 → Πv 被强制覆写为 MIN_PRECISION。"""
        prior = (name, (0.1, 0.3), (0.5, 0.5))  # Πv=0.5（原始值，非 MIN_PRECISION）
        result = expand_external_priors([prior], precision_cap=0.8, max_streams=5)
        assert len(result) == 1
        assert result[0][2][0] == pytest.approx(MIN_PRECISION), (
            f"physio 流 '{name}' 的 Πv 应被强制为 MIN_PRECISION={MIN_PRECISION}，"
            f"实际为 {result[0][2][0]}"
        )

    @pytest.mark.parametrize(
        "name",
        [
            "physio_eda",
            "eda",
            "hrv",
            "pupil",
            "scr",
        ],
    )
    def test_physio_pi_a_preserved(self, name: str) -> None:
        """M2 只覆写 Πv、不碰 Πa；Πa 由 **γ 层**按 GAMMA_PHYSIO 折减（预期翻转，2026-09-01
        议会设计门·PRP/gamma-physio）。

        两段断言：① Πa 输出 == GAMMA_PHYSIO·Πa_naive（γ 恰好乘一次，不叠乘）；
        ② γ ∈ (0,1)（是折减不是放大/清零）。禁手抄 0.069——从常量导入现算。
        """
        pi_a = 0.6
        prior = (name, (0.1, 0.3), (0.3, pi_a))
        result = expand_external_priors([prior], precision_cap=0.8, max_streams=5)
        assert result[0][2][1] == pytest.approx(GAMMA_PHYSIO * pi_a), (
            f"physio 流 Πa 应为 γ·{pi_a}={GAMMA_PHYSIO * pi_a}，实际为 {result[0][2][1]}"
        )
        assert 0.0 < GAMMA_PHYSIO < 1.0, "γ 必须是 (0,1) 内的折减乘子"

    @pytest.mark.parametrize(
        "name",
        ["face", "audio", "video", "vision", "fusion", "custom_stream"],
    )
    def test_non_physio_pi_v_preserved(self, name: str) -> None:
        """非生理流 Πv 原样透传（不被覆写）。"""
        pi_v = 0.5
        prior = (name, (0.2, 0.4), (pi_v, 0.6))
        result = expand_external_priors([prior], precision_cap=0.8, max_streams=5)
        assert result[0][2][0] == pytest.approx(pi_v), (
            f"非 physio 流 '{name}' 的 Πv 应透传 {pi_v}，实际为 {result[0][2][0]}"
        )

    def test_physio_prefix_case_insensitive(self) -> None:
        """physio 前缀匹配用小写比较（大写 name 也应被覆写）。"""
        # _PHYSIO_PREFIXES 匹配用 name.lower()，确认大写生效
        result = expand_external_priors(
            [("PHYSIO_EDA", (0.1, 0.2), (0.5, 0.5))],
            precision_cap=0.8,
            max_streams=5,
        )
        assert result[0][2][0] == pytest.approx(MIN_PRECISION)

    def test_physio_all_prefixes_covered(self) -> None:
        """_PHYSIO_PREFIXES 全部前缀抽样覆盖：逐个验证 M2 强制。"""
        for prefix in _PHYSIO_PREFIXES:
            name = f"{prefix}_test_signal"
            result = expand_external_priors(
                [(name, (0.0, 0.5), (0.6, 0.6))],
                precision_cap=0.8,
                max_streams=5,
            )
            assert result[0][2][0] == pytest.approx(MIN_PRECISION), (
                f"前缀 '{prefix}' 应触发 M2 强制，实际 Πv={result[0][2][0]}"
            )


class TestExpandExternalPriorsM2MuValenceZeroing:
    """M2 补完（2026-07-29 跨仓议定）：physio 流的 μv **一并归零**，不只覆写 Πv。

    背景：此前 M2 只覆写 Πv、从不碰 μ，而配套项目把「physio 的 μv 恒 0」误归因给「Zero M2」
    并据此推出其自律上界 0.359。归零后该归因才成为真的。
    """

    @pytest.mark.parametrize("name", ["physio", "eda_tonic", "hrv_rmssd", "pupil_d", "scr_amp"])
    def test_physio_mu_valence_forced_to_zero(self, name: str) -> None:
        """五个前缀的 μv 一律归零；μa **原样透传**（只对效价盲，不动唤醒）。"""
        out = expand_external_priors(
            [(name, (0.9, 0.6), (0.5, 0.5))], precision_cap=0.8, max_streams=5
        )
        assert out[0][1] == (0.0, 0.6), "μv 应归零、μa 应透传"

    @pytest.mark.parametrize("name", ["face", "audio", "text"])
    def test_non_physio_mu_untouched(self, name: str) -> None:
        """非生理流的 μ 一个字节都不动——归零只是 M2 的作用域，不是全局。"""
        out = expand_external_priors(
            [(name, (0.9, 0.6), (0.5, 0.5))], precision_cap=0.8, max_streams=5
        )
        assert out[0][1] == (0.9, 0.6)

    def test_zeroing_happens_after_m7_not_instead_of_it(self) -> None:
        """越域 μv 仍被 M7 **拒绝**，不是「静默接受再抹掉」。

        顺序契约：M7（校验 client 发的东西）→ M2（归一化）。若把归零提到 M7 之前，
        越域载荷会被静默洗白，本用例即红。
        """
        with pytest.raises(ValueError, match="μ 须在"):
            expand_external_priors(
                [("physio", (1.5, 0.6), (0.5, 0.5))], precision_cap=0.8, max_streams=5
            )

    def test_self_ignite_bound_is_computed_not_hardcoded(self) -> None:
        """归零把 physio 的自点燃上界从「√2 保守界」收回到 |μa|≤1 那一档。

        两个界都**现算**（禁止手抄）：不归零时 hypot(μ) 可达 √2 ⇒ 界 = 2T/√2 − MIN_PRECISION；
        归零后 hypot(μ)=|μa| ≤ 1 ⇒ 界 = 2T − MIN_PRECISION。前者更紧，正是我方此前告知对方的那个数。
        """
        import math

        loose = 2 * SALIENCE_THRESHOLD - MIN_PRECISION  # 归零后（|μa|≤1）
        tight = 2 * SALIENCE_THRESHOLD / math.sqrt(2) - MIN_PRECISION  # 不归零（hypot≤√2）
        assert tight < loose, "√2 那一档必须更紧，否则本用例的前提说反了"
        # 归零已生效 ⇒ 一条 μv=0.9 的 physio 流的 salience 只由 μa 决定。
        # γ 层（2026-09-01）后 Πa 输出 = γ·loose：把 γ 除回去即还原「取等过门」恒等式——
        # 本用例锁的是 μv 归零的几何（hypot=|μa|），γ 的折减语义由曝露面类独立锁定。
        out = expand_external_priors(
            [("physio", (0.9, 1.0), (0.5, loose))], precision_cap=0.8, max_streams=5
        )
        name, mu, prec = out[0]
        assert mu == (0.0, 1.0), "μv 归零是本用例前提"
        degamma_prec = (prec[0], prec[1] / GAMMA_PHYSIO)
        assert stream_salience(mu, degamma_prec) == pytest.approx(SALIENCE_THRESHOLD, abs=1e-9)


class TestExpandExternalPriorsM3:
    """M3：精度校验（<=0 / >cap / 形状不合法）→ ValueError；边界值通过。"""

    def test_pi_v_zero_raises(self) -> None:
        """Πv=0 → raise ValueError。"""
        with pytest.raises(ValueError, match="精度须 >0"):
            expand_external_priors(
                [("x", (0.1, 0.2), (0.0, 0.5))], precision_cap=0.8, max_streams=5
            )

    def test_pi_a_zero_raises(self) -> None:
        """Πa=0 → raise ValueError。"""
        with pytest.raises(ValueError, match="精度须 >0"):
            expand_external_priors(
                [("x", (0.1, 0.2), (0.5, 0.0))], precision_cap=0.8, max_streams=5
            )

    def test_pi_v_negative_raises(self) -> None:
        """Πv<0 → raise ValueError。"""
        with pytest.raises(ValueError, match="精度须 >0"):
            expand_external_priors(
                [("x", (0.1, 0.2), (-0.1, 0.5))], precision_cap=0.8, max_streams=5
            )

    def test_pi_a_nan_raises(self) -> None:
        """Πa=NaN → raise。

        **修复前双侧漏检**：NaN 与任何数比较恒 False，故 `pi_a <= 0.0` 与
        `pi_a > precision_cap` 都放行 → NaN 一路进 fuse_terms 产出 NaN 后验且无报错。
        撤掉 `expand_external_priors` 里的 M3′ 有限性校验，本用例即红。
        """
        with pytest.raises(ValueError, match="精度须为有限数"):
            expand_external_priors(
                [("x", (0.1, 0.2), (0.5, float("nan")))], precision_cap=0.8, max_streams=5
            )

    def test_pi_v_nan_raises(self) -> None:
        """Πv=NaN（非 physio 名，M2 不覆写）→ raise。"""
        with pytest.raises(ValueError, match="精度须为有限数"):
            expand_external_priors(
                [("x", (0.1, 0.2), (float("nan"), 0.5))], precision_cap=0.8, max_streams=5
            )

    def test_old_comparisons_would_have_passed_nan(self) -> None:
        """判别力自证：NaN **确实**穿得过原来那两条判据——否则上面两条用例是零判别力的摆设。"""
        nan = float("nan")
        assert not (nan <= 0.0), "NaN 未被正值判据拦下（这正是漏检成因）"
        assert not (nan > 0.8), "NaN 未被上界判据拦下（这正是漏检成因）"

    def test_physio_nan_pi_v_is_overwritten_not_raised(self) -> None:
        """顺序契约：physio 流的 NaN Πv 先被 M2 覆写为 MIN_PRECISION，**不**触发有限性报错。

        锁住 M2 在 M3′ 之前这一顺序（与 M3 同理由）：physio 的 Πv 无条件覆写，
        不因 MCP 误传而误报。若把 M3′ 提到 M2 之前，本用例即红。
        """
        out = expand_external_priors(
            [("physio", (0.0, 0.5), (float("nan"), 0.175))], precision_cap=0.8, max_streams=5
        )
        assert out[0][2][0] == MIN_PRECISION

    def test_pi_inf_raises(self) -> None:
        """Πa=+inf → raise（inf 本就会被上界判据接住，本条只是把报错口径统一到有限性）。"""
        with pytest.raises(ValueError, match="精度须为有限数"):
            expand_external_priors(
                [("x", (0.1, 0.2), (0.5, float("inf")))], precision_cap=0.8, max_streams=5
            )

    def test_nan_string_through_mcp_mapping_raises(self) -> None:
        """端到端：MCP 载荷里的字符串 "nan" 经 `float()` 变成 NaN 送进内核 → 被拦。

        这是对方 client 侧守卫**保护不到**的入口（chat 面与测试夹具同理）。
        """
        from src.mcp_server.mapping import external_priors_from_payload

        priors = external_priors_from_payload([["x", [0.1, 0.2], [0.5, "nan"]]])
        with pytest.raises(ValueError, match="精度须为有限数"):
            expand_external_priors(priors, precision_cap=0.8, max_streams=5)

    def test_pi_v_exceeds_cap_raises(self) -> None:
        """Πv > precision_cap → raise ValueError。"""
        with pytest.raises(ValueError, match="精度超过上界"):
            expand_external_priors(
                [("x", (0.1, 0.2), (0.9, 0.5))], precision_cap=0.8, max_streams=5
            )

    def test_pi_a_exceeds_cap_raises(self) -> None:
        """Πa > precision_cap → raise ValueError。"""
        with pytest.raises(ValueError, match="精度超过上界"):
            expand_external_priors(
                [("x", (0.1, 0.2), (0.5, 0.9))], precision_cap=0.8, max_streams=5
            )

    def test_pi_v_equals_cap_passes(self) -> None:
        """Πv == precision_cap（边界等于）→ 通过（不 raise）。"""
        cap = 0.8
        result = expand_external_priors(
            [("x", (0.1, 0.2), (cap, 0.5))], precision_cap=cap, max_streams=5
        )
        assert len(result) == 1
        assert result[0][2][0] == pytest.approx(cap)

    def test_tiny_positive_precision_passes(self) -> None:
        """极小正精度（如 1e-6，>0 且 <=cap）→ 通过。"""
        result = expand_external_priors(
            [("x", (0.0, 0.0), (1e-6, 1e-6))], precision_cap=0.8, max_streams=5
        )
        assert len(result) == 1

    def test_scalar_precision_raises(self) -> None:
        """标量精度（如 ("x",(0.1,0.2),0.3)）形状不合法 → raise ValueError。"""
        with pytest.raises(ValueError, match="形状不合法"):
            expand_external_priors(
                [("x", (0.1, 0.2), 0.3)],  # type: ignore[list-item]
                precision_cap=0.8,
                max_streams=5,
            )

    def test_wrong_arity_mu_raises(self) -> None:
        """mu 为 3 元组 → 形状不合法 → raise ValueError。"""
        with pytest.raises(ValueError, match="形状不合法"):
            expand_external_priors(
                [("x", (0.1, 0.2, 0.3), (0.5, 0.5))],  # type: ignore[list-item]
                precision_cap=0.8,
                max_streams=5,
            )

    def test_name_not_str_raises(self) -> None:
        """name 非 str → 形状不合法 → raise ValueError。"""
        with pytest.raises(ValueError, match="形状不合法"):
            expand_external_priors(
                [(42, (0.1, 0.2), (0.5, 0.5))],  # type: ignore[list-item]
                precision_cap=0.8,
                max_streams=5,
            )

    def test_two_element_tuple_raises(self) -> None:
        """只有 2 元素的 tuple（缺 prec）→ 形状不合法 → raise ValueError。"""
        with pytest.raises(ValueError, match="形状不合法"):
            expand_external_priors(
                [("x", (0.1, 0.2))],  # type: ignore[list-item]
                precision_cap=0.8,
                max_streams=5,
            )


class TestExpandExternalPriorsM7MuDomain:
    """M7：μ 域校验（议会 2026-07-28）。

    契约 external_prior.py 声明 μv/μa ∈[-1,1]，此前只是注释、运行期不校验——
    越界 μ 会直接抬高 stream_salience 买到本不该有的点燃资格。纯边界收紧：
    合法输入行为逐字不变。
    """

    def test_mu_valence_above_one_raises(self) -> None:
        """μv > 1 → raise ValueError。"""
        with pytest.raises(ValueError, match=r"μ 须在 \[-1, 1\] 内"):
            expand_external_priors(
                [("face", (1.5, 0.2), (0.5, 0.5))], precision_cap=0.8, max_streams=5
            )

    def test_mu_arousal_above_one_raises(self) -> None:
        """μa > 1（此前实测可越过 SALIENCE_THRESHOLD 点燃）→ raise ValueError。"""
        with pytest.raises(ValueError, match=r"μ 须在 \[-1, 1\] 内"):
            expand_external_priors(
                [("eda_sc", (0.0, 2.0), (0.5, 0.18))], precision_cap=0.8, max_streams=5
            )

    def test_mu_below_negative_one_raises(self) -> None:
        """μ < -1 → raise ValueError。"""
        with pytest.raises(ValueError, match=r"μ 须在 \[-1, 1\] 内"):
            expand_external_priors(
                [("audio", (-1.01, 0.0), (0.1, 0.25))], precision_cap=0.8, max_streams=5
            )

    def test_mu_nan_raises(self) -> None:
        """μ 为 NaN → 比较恒 False → 由同一条校验拦下。"""
        with pytest.raises(ValueError, match=r"μ 须在 \[-1, 1\] 内"):
            expand_external_priors(
                [("face", (float("nan"), 0.0), (0.2, 0.12))],
                precision_cap=0.8,
                max_streams=5,
            )

    @pytest.mark.parametrize("mu", [(-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (0.0, 0.0)])
    def test_mu_domain_boundary_passes(self, mu: tuple[float, float]) -> None:
        """μ 恰在闭区间边界与内部 → 通过（合法输入零回归）。"""
        result = expand_external_priors(
            [("face", mu, (0.2, 0.12))], precision_cap=0.8, max_streams=5
        )
        assert result[0][1] == mu

    def test_mu_check_precedes_m2_physio_override(self) -> None:
        """physio 流越界 μ 同样被拦（M7 早于 M2 覆写，不因 Πv 归零而漏检）。"""
        with pytest.raises(ValueError, match=r"μ 须在 \[-1, 1\] 内"):
            expand_external_priors(
                [("physio_eda", (0.0, 1.2), (0.9, 0.18))],
                precision_cap=0.8,
                max_streams=5,
            )


class TestExpandExternalPriorsM6:
    """M6：流数上界校验。"""

    def test_len_exceeds_max_streams_raises(self) -> None:
        """len(external_priors) > max_streams → raise ValueError。"""
        priors = [("face", (0.1, 0.2), (0.5, 0.5))] * 6  # 6 条
        with pytest.raises(ValueError, match="超过 max_external_streams"):
            expand_external_priors(priors, precision_cap=0.8, max_streams=5)

    def test_len_equals_max_streams_passes(self) -> None:
        """len == max_streams（边界等于）→ 通过。"""
        # 正好 5 条；流名须互异（唯一性校验 2026-09-01，与本测点 M6 无关）
        priors = [(f"face_{i}", (0.1, 0.2), (0.5, 0.5)) for i in range(5)]
        result = expand_external_priors(priors, precision_cap=0.8, max_streams=5)
        assert len(result) == 5

    def test_len_one_under_max_streams_passes(self) -> None:
        """len == max_streams - 1 → 通过。"""
        priors = [(f"audio_{i}", (0.2, 0.3), (0.4, 0.4)) for i in range(4)]
        result = expand_external_priors(priors, precision_cap=0.8, max_streams=5)
        assert len(result) == 4

    def test_max_streams_zero_empty_list_passes(self) -> None:
        """max_streams=0、空列表 → 通过（len=0 不超过 0）。"""
        result = expand_external_priors([], precision_cap=0.8, max_streams=0)
        assert result == []

    def test_max_streams_zero_any_stream_raises(self) -> None:
        """max_streams=0、有任意 1 条 → raise ValueError。"""
        with pytest.raises(ValueError, match="超过 max_external_streams"):
            expand_external_priors(
                [("x", (0.1, 0.2), (0.5, 0.5))], precision_cap=0.8, max_streams=0
            )


class TestExpandExternalPriorsOutputShape:
    """返回列表的每条形状与 affect_core.py:77 streams 类型完全一致。"""

    def test_output_item_is_name_mu_prec_tuple(self) -> None:
        """返回条目为 (str, (float,float), (float,float)) 类型。"""
        result = expand_external_priors(
            [("face", (0.2, 0.4), (0.5, 0.6))], precision_cap=0.8, max_streams=5
        )
        name, mu, prec = result[0]
        assert isinstance(name, str)
        assert isinstance(mu, tuple) and len(mu) == 2
        assert isinstance(prec, tuple) and len(prec) == 2
        assert all(isinstance(x, float) for x in mu)
        assert all(isinstance(x, float) for x in prec)

    def test_multi_stream_output_length(self) -> None:
        """多条输入 → 输出条数相同。"""
        priors = [
            ("face", (0.2, 0.4), (0.5, 0.6)),
            ("audio", (-0.1, 0.3), (0.4, 0.7)),
            ("hrv", (0.0, 0.5), (0.6, 0.5)),
        ]
        result = expand_external_priors(priors, precision_cap=0.8, max_streams=5)
        assert len(result) == 3


# --------------------------------------------------------------------------
# 2. AffectCore 节点集成
# --------------------------------------------------------------------------


FIELDS_AFFECT_CORE = {"post_mu", "post_sigma", "affect_sample", "trace"}
FIELDS_AFFECT_CORE_WS = FIELDS_AFFECT_CORE | {"ignited_streams", "affect_precision"}


class TestAffectCoreZeroRegression:
    """external_priors=[]（默认）→ 节点 key 集合与旧路径一致（零回归）。"""

    def test_empty_external_priors_key_set_equals_baseline(self) -> None:
        """external_priors=[] 显式传入 → 返回 key 集合与不设该字段时相同。"""
        state_no_field = _base_state()
        state_empty = _base_state(external_priors=[])
        out_no = AffectCoreAgent()(state_no_field)
        out_empty = AffectCoreAgent()(state_empty)
        assert set(out_no) == set(out_empty), (
            f"external_priors=[] 与不设该字段 key 集应相同；差异：{set(out_no) ^ set(out_empty)}"
        )

    def test_empty_external_priors_post_mu_equals_baseline(self) -> None:
        """external_priors=[] → post_mu 与不设该字段时逐字一致（MAP 读出确定）。"""
        state_no_field = _base_state()
        state_empty = _base_state(external_priors=[])
        out_no = AffectCoreAgent()(state_no_field)
        out_empty = AffectCoreAgent()(state_empty)
        assert out_no["post_mu"] == pytest.approx(out_empty["post_mu"], abs=1e-12), (
            "external_priors=[] 的 post_mu 应与基线逐字一致"
        )

    def test_workspace_disabled_external_priors_ignored(self) -> None:
        """workspace_enabled=False（默认非 workspace 路径）→ external_priors 不展开，
        post_mu 与空列表无 workspace 时一致（非 workspace 路径 expand_external_priors 不被调用）。
        """
        state_no_ws = AffectState(
            stimulus=Stimulus(name="t", goal_congruence=0.5, intensity=0.8),
            prior_mu=(0.3, 0.5),
            prior_sigma=(0.2, 0.2),
            reward=0.5,
            rpe=0.2,
            precision=0.6,
            rng_seed=42,
            workspace_enabled=False,
            affect_readout="map",
            external_priors=[("face", (0.9, 0.9), (0.7, 0.7))],
        )
        state_baseline = AffectState(
            stimulus=Stimulus(name="t", goal_congruence=0.5, intensity=0.8),
            prior_mu=(0.3, 0.5),
            prior_sigma=(0.2, 0.2),
            reward=0.5,
            rpe=0.2,
            precision=0.6,
            rng_seed=42,
            workspace_enabled=False,
            affect_readout="map",
            external_priors=[],
        )
        out_with = AffectCoreAgent()(state_no_ws)
        out_base = AffectCoreAgent()(state_baseline)
        assert out_with["post_mu"] == pytest.approx(out_base["post_mu"], abs=1e-12), (
            "非 workspace 路径 external_priors 不参与融合，结果应与空列表相同"
        )


class TestAffectCoreM1ExternalPriorsInfluence:
    """M1：注入非冲突外部流 → 后验受影响（高精度 arousal 流拉动后验 arousal 更强）。"""

    def test_high_arousal_external_prior_shifts_arousal_upward(self) -> None:
        """注入高 μa 高 Πa 的外部流 → 后验 arousal 比不注入时更高（朝高 arousal 拉动）。"""
        state_no_ext = _base_state(external_priors=[])
        state_with_ext = _base_state(
            external_priors=[
                ("face", (0.3, 0.9), (0.1, 0.7)),  # 高 μa + 高 Πa
            ]
        )
        out_no = AffectCoreAgent()(state_no_ext)
        out_with = AffectCoreAgent()(state_with_ext)
        assert out_with["post_mu"][1] > out_no["post_mu"][1], (
            "高 arousal 外部流应拉动后验 arousal 更高"
        )

    def test_high_valence_external_prior_shifts_valence(self) -> None:
        """注入高 μv 高 Πv 的非 physio 外部流 → 后验 valence 比不注入时更高。"""
        state_no_ext = _base_state(
            prior_mu=(0.0, 0.3),  # 中性先验
            external_priors=[],
        )
        state_with_ext = _base_state(
            prior_mu=(0.0, 0.3),
            external_priors=[
                ("video", (0.8, 0.3), (0.6, 0.4)),  # 高 μv + 高 Πv
            ],
        )
        out_no = AffectCoreAgent()(state_no_ext)
        out_with = AffectCoreAgent()(state_with_ext)
        assert out_with["post_mu"][0] > out_no["post_mu"][0], (
            "高 valence 外部流应使后验 valence 升高"
        )

    def test_precision_weighted_higher_pi_stronger_pull(self) -> None:
        """高 Πa 流比低 Πa 流对后验 arousal 的拉动更强（精度越高投票权越大）。"""
        state_low_pi = _base_state(
            prior_mu=(0.0, 0.0),
            external_priors=[("face", (0.3, 0.9), (0.1, 0.1))],  # 低 Πa
        )
        state_high_pi = _base_state(
            prior_mu=(0.0, 0.0),
            external_priors=[("face", (0.3, 0.9), (0.1, 0.7))],  # 高 Πa
        )
        out_low = AffectCoreAgent()(state_low_pi)
        out_high = AffectCoreAgent()(state_high_pi)
        # 高 Πa 流更靠近 μa=0.9 → 后验 arousal 更高
        assert out_high["post_mu"][1] > out_low["post_mu"][1], (
            "Πa 越高的外部流应对后验 arousal 拉动更强"
        )

    def test_external_priors_streams_included_in_ignited(self) -> None:
        """注入的外部流名应出现在 ignited_streams（若其 salience 过阈）。"""
        state = _base_state(
            prior_mu=(0.0, 0.0),
            external_priors=[
                ("face", (0.8, 0.8), (0.7, 0.7)),  # 高 μ + 高 Π → 高 salience
            ],
        )
        out = AffectCoreAgent()(state)
        assert "face" in out["ignited_streams"], (
            "高 salience 外部流 'face' 应出现在 ignited_streams 中"
        )

    def test_multiple_external_priors_combine(self) -> None:
        """注入 2-3 条非冲突外部流 → 后验比只注入 1 条时更偏向注入方向（融合效果）。"""
        state_one = _base_state(
            prior_mu=(0.0, 0.0),
            external_priors=[("face", (0.3, 0.8), (0.1, 0.5))],
        )
        state_two = _base_state(
            prior_mu=(0.0, 0.0),
            external_priors=[
                ("face", (0.3, 0.8), (0.1, 0.5)),
                ("audio", (0.2, 0.7), (0.1, 0.5)),
            ],
        )
        out_one = AffectCoreAgent()(state_one)
        out_two = AffectCoreAgent()(state_two)
        out_base = AffectCoreAgent()(_base_state(prior_mu=(0.0, 0.0), external_priors=[]))
        # 单条已把 arousal 拉离基线；两条非冲突高 arousal 流累积精度更大 → 比单条更强
        assert out_one["post_mu"][1] > out_base["post_mu"][1], "单条外部流应把 arousal 拉离基线"
        assert out_two["post_mu"][1] > out_one["post_mu"][1], (
            "两条非冲突外部流融合应比单条有更强的 arousal 偏移（累积精度）"
        )


class TestIgnitionReachabilityAtRecommendedPrecision:
    """⚠ 特征化用例：用 design.md §五 **推荐默认精度**驱动点燃门（议会 2026-07-28）。

    这些断言锁定的是**当前已知缺陷行为**，不是期望行为——记录真实状态而非掩盖。
    此前唯一断言 external 点燃的用例用 Π=(0.7,0.7)（推荐值的 4~700 倍），
    绿灯具误导性，缺陷因此静默存活到配套项目 Zero_MCP 实测才暴露。

    现行 stream_salience = hypot(μv,μa)·(Πv+Πa)/2 ≥ SALIENCE_THRESHOLD=0.18 下：
      face  mean(Π)=0.1600 → 需 |μ|≥1.1250（>1.0，单维不可达）
      audio mean(Π)=0.1750 → 需 |μ|≥1.0286（>1.0，单维不可达）
      physio mean(Π)=0.0905 → 需 |μ|≥1.9890 > √2 → **数学上恒不可点燃**

    🛑 **原「应变红」预告已作废（2026-07-29）**：按轴加权马氏距离 + θ'=0.28 那套方案
    **从未落地**，`stream_salience` 与 `SALIENCE_THRESHOLD=0.18` 逐字未动。
    PR #46 落的是线 A（硬门摘出数值通路）+ 线 B（精度齐次化），**二者默认关**，
    故本类在默认配置下逐值不变。留着旧预告等于挂一个**永远不会响的警报**——
    与我方在跨仓件里批评对方的是同一族问题（见 2026-07-29 回执 §R8 的自曝）。
    本类真正会红的时机：翻默认（`ZERO_IGNITION_GATE_FUSION=false`）或改 `SALIENCE_THRESHOLD`。
    """

    # design.md §五 推荐默认（physio Πv 由 M2 强制覆写为 MIN_PRECISION）
    FACE_PREC = (0.20, 0.12)
    AUDIO_PREC = (0.10, 0.25)
    PHYSIO_PREC = (MIN_PRECISION, 0.18)

    def test_physio_at_recommended_precision_never_ignites(self) -> None:
        """physio 取合法域内最强读数（μa=1.0）仍不点燃——结构性不可达。"""
        state = _base_state(
            external_priors=[("physio", (0.0, 1.0), self.PHYSIO_PREC)],
        )
        out = AffectCoreAgent()(state)
        assert "physio" not in out["ignited_streams"], (
            "特征化：physio 在推荐精度下 salience 上限 0.0905 < 0.18，恒不点燃"
        )

    def test_physio_expanded_but_gated(self) -> None:
        """physio 确实被展开成合法流（M1-M3 通过），只是被点燃门挡在融合外。"""
        expanded = expand_external_priors(
            [("physio", (0.0, 0.91), self.PHYSIO_PREC)],
            precision_cap=0.8,
            max_streams=5,
        )
        assert expanded == [("physio", (0.0, 0.91), (MIN_PRECISION, GAMMA_PHYSIO * 0.18))], (
            "展开层正常（Πa 经 γ 层折减，2026-09-01）：点燃门判定在下游 ignite，不在展开/校验"
        )

    def test_face_at_recommended_precision_real_strong_sample_drops(self) -> None:
        """真人脸最强样本（EmotiEffLib Anger |μ|=0.9037）在推荐精度下仍不点燃。"""
        state = _base_state(
            external_priors=[("face", (-0.7177, 0.5492), self.FACE_PREC)],
        )
        out = AffectCoreAgent()(state)
        assert "face" not in out["ignited_streams"], (
            "特征化：真人脸最强表情 salience 0.1446 < 0.18，差门槛约 24%"
        )

    def test_audio_at_recommended_precision_real_strong_sample_drops(self) -> None:
        """真语音最强样本（audeering w2v2 尖叫 |μ|=0.4729）在推荐精度下不点燃。"""
        state = _base_state(
            external_priors=[("audio", (0.0, 0.4729), self.AUDIO_PREC)],
        )
        out = AffectCoreAgent()(state)
        assert "audio" not in out["ignited_streams"], (
            "特征化：真语音最强样本 salience 0.0828 << 0.18"
        )

    def test_single_axis_saturation_unreachable_for_all_modalities(self) -> None:
        """三模态门槛均 >1.0 → 单维饱和（|μ|=1.0）一律不可达。

        环状模型里纯效价轴/纯唤醒轴（如「惊讶」「平静满足」）是合法基本方位，
        当前判据在几何上把这一整类情绪判了死刑（心理席，Russell 1980）。
        """
        for name, prec, mu in [
            ("face", self.FACE_PREC, (1.0, 0.0)),
            ("face", self.FACE_PREC, (0.0, 1.0)),
            ("audio", self.AUDIO_PREC, (1.0, 0.0)),
            ("audio", self.AUDIO_PREC, (0.0, 1.0)),
        ]:
            state = _base_state(external_priors=[(name, mu, prec)])
            out = AffectCoreAgent()(state)
            assert name not in out["ignited_streams"], (
                f"特征化：{name} 单维饱和 μ={mu} 在推荐精度下仍不可点燃"
            )

    def test_merged_physio_at_omega_half_still_gated_under_old_formula(self) -> None:
        """MCP 侧 ω=0.5 预合并（Π_merged=0.175）后，旧公式下仍不点燃。

        与配套项目既有特征化断言自洽——预合并解决的是 Σπ 虚增（重复计数），
        不解决点燃门的构造缺陷，两者是不同层面的问题。
        """
        # eda μa=0.76 / hrv μa=0.91，ω=0.5 → μ_merged=0.845714, Π_merged=0.175
        state = _base_state(
            external_priors=[("physio", (0.0, 0.845714), (MIN_PRECISION, 0.175))],
        )
        out = AffectCoreAgent()(state)
        assert "physio" not in out["ignited_streams"], "特征化：合并后 salience≈0.0744 仍 < 0.18"


class TestPhysioSelfIgniteExposureOnDefaultPath:
    """✅ **曝露面已关闭**（γ 层，议会设计门 2026-09-01·PRP/gamma-physio）：本类原为「physio
    流把 Πa 抬到现算界即可在生产默认配置上自行过点燃门」的曝露面记录，其 docstring 预告的
    「加了结构上界那天本类会红——届时改写成关闭锁、不要放宽」已兑现——现在锁的是**关闭**：
    γ 折减使任意合法 payload 的 physio salience 恒 < SALIENCE_THRESHOLD（代数界：M3 先钳
    Πa≤cap，γ 后乘 ⇒ Πa_declared ≤ γ·cap）。变异判别力：下面的用例同时断言「若无 γ 层
    该载荷本应过门」（撤 γ 即红），关闭归因于 γ 而非别处漂移。

    历史背景（保留供追溯）：physio 流只要把 Πa 抬到某个界，就能在**生产默认配置**
    （`gate_fusion=True` 门关 + `soft_beta=None` 硬门）上**自行过点燃门**。
    我方侧对此没有任何结构约束。

    此前这个曝露面只写在 `ignite()` docstring 的散文里（D7 缺口那段），**零可执行断言**——
    注释会烂、常量会漂。本类把它钉成回归锁。

    界**现算**（`SALIENCE_THRESHOLD` / `MIN_PRECISION` 一动即跟着动，**禁止手抄 0.359**）：
    M2 把 physio 的 Πv 覆写为 MIN_PRECISION、μv 归零 ⇒ hypot(μ)=|μa| ≤ 1
    ⇒ 最坏 salience = |μa|·(MIN_PRECISION + Πa)/2；取 |μa|=1 令其 ≥ SALIENCE_THRESHOLD
    即得 Πa ≥ 2·SALIENCE_THRESHOLD − MIN_PRECISION。判据（`_select_fired` 硬门分支）是
    `s >= threshold`——**取等即过门**，故该界本身就在门内。

    对侧现状：Zero_MCP 已在其出网收口点落了 M8 运行期守卫（最坏 salience 达阈即 raise），
    但那是**对方仓内**的单边守卫——不经该收口点的入口（我方 chat 面、测试夹具、其它 client）
    照样能把这样的载荷送进来，故本曝露面在我方侧依然成立。

    ✅ 上面预告的那天已到（2026-09-01 γ 层）：本类已按预告改写成关闭锁——「载荷被折减
    到亚阈」而非「载荷被拒」（γ 是声明前折减不是 raise，方向裁定见 design.md Q2）。
    残余开口（γ 触及不到）：`_select_fired` 全场亚阈时的 top-1 兜底不检阈值，已另案登记
    （notes/2026-09-01-gamma-physio-council.md §四），勿在本类里追杀。
    """

    @staticmethod
    def _self_ignite_pi_a_bound() -> float:
        """现算 physio 在生产默认路径上自行过门所需的 Πa 下界（禁止手抄常量）。"""
        return 2 * SALIENCE_THRESHOLD - MIN_PRECISION

    def test_computed_bound_is_injectable_at_all(self) -> None:
        """前置条件：该界本身必须落在默认 precision_cap 内，否则下面两条测的不是点燃门。"""
        assert self._self_ignite_pi_a_bound() <= AffectState().external_prior_precision_cap, (
            "现算的自点燃界已超出默认 precision_cap ⇒ M3 会先 raise，"
            "本类的两条断言不再刻画点燃门；若这是有意收紧，请连带重写本类而非删本条"
        )

    def test_physio_at_computed_bound_no_longer_ignites_after_gamma(self) -> None:
        """✅ 关闭锁（原曝露面用例的改写，按其自身预告执行）：Πa 取现算界时，physio 在生产
        默认路径上**不再**过门——γ 层把声明 Πa 折减为 γ·Πa，salience 跌到阈值零头。

        变异判别力（绿灯先证能红）：同时断言「无 γ 时该载荷本应过门」——把 γ 除回去的
        salience ≥ 阈值。撤掉 γ 层（γ=1.0）时下面第三条断言立刻红。
        故意带 μv=0.9 与 Πv=0.5 注入：M2 归零/覆写两条前提照旧一并锁定。
        """
        pi_a = self._self_ignite_pi_a_bound()
        prior = ("physio", (0.9, 1.0), (0.5, pi_a))
        expanded = expand_external_priors(
            [prior],
            precision_cap=AffectState().external_prior_precision_cap,
            max_streams=5,
        )
        assert expanded[0][1] == (0.0, 1.0), "前提①：M2 应把 μv 归零、μa 原样透传"
        assert expanded[0][2] == pytest.approx((MIN_PRECISION, GAMMA_PHYSIO * pi_a)), (
            "前提②：M2 覆写 Πv、γ 层折减 Πa（各恰一次）"
        )
        # 反事实（判别力）：无 γ 时该界恰好取等过门——关闭确实归因于 γ
        degamma = (expanded[0][2][0], expanded[0][2][1] / GAMMA_PHYSIO)
        assert stream_salience(expanded[0][1], degamma) >= SALIENCE_THRESHOLD, (
            "反事实前提变了：现算界在无 γ 语义下本应取等过门"
        )
        assert stream_salience(expanded[0][1], expanded[0][2]) < SALIENCE_THRESHOLD, (
            "γ 折减后现算界载荷必须亚阈——本断言红=γ 层被撤/失效（双输区重新打开）"
        )
        out = AffectCoreAgent()(_base_state(external_priors=[prior]))
        assert "physio" not in out["ignited_streams"], (
            f"γ 层落地后 Πa={pi_a} 的 physio 流不得在生产默认路径（门关+硬门）自行过门；"
            "本断言红=负边际双输区 [0.359,1.0) 重新曝露，先查 expand_external_priors 的 γ 层"
        )
        assert set(out["ignited_streams"]) >= {"survival", "appraisal", "value"}, (
            "内核流须已点燃 ⇒ physio 的落选来自阈值判据，而非全员亚阈的 max 兜底分支"
        )

    def test_physio_just_below_computed_bound_does_not_ignite(self) -> None:
        """反向（防恒真）：Πa 取该界**略下方**时不过门——过门资格确实由这个界划定。"""
        pi_a = self._self_ignite_pi_a_bound() - 1e-6
        prior = ("physio", (0.9, 1.0), (0.5, pi_a))
        expanded = expand_external_priors(
            [prior],
            precision_cap=AffectState().external_prior_precision_cap,
            max_streams=5,
        )
        assert stream_salience(expanded[0][1], expanded[0][2]) < SALIENCE_THRESHOLD
        out = AffectCoreAgent()(_base_state(external_priors=[prior]))
        assert out["ignited_streams"], "本用例前提：至少有内核流点燃"
        assert "physio" not in out["ignited_streams"], (
            f"界下方（Πa={pi_a}）的 physio 不应过门；若它过了，说明上一条的绿灯是恒真的，"
            "两条一起失去刻画能力"
        )
        assert set(out["ignited_streams"]) >= {"survival", "appraisal", "value"}, (
            "内核流须已点燃 ⇒ physio 的落选来自阈值判据，而非 `_select_fired` "
            "全员亚阈时的 max 兜底分支（兜底会把 physio 选回来，让本用例变成假绿）"
        )


class TestGammaPhysioClosureAnchors:
    """γ 层新锚点（PRP/gamma-physio design.md §四·方向层 + γ 可复现性）。

    分布层锚点 = 占比表脚本 `scripts/verify_gamma_physio_occupancy.py`（design §五，
    产物随 PR 落 PRP/gamma-physio/artifacts/），本类只锁方向与推导可复现。
    """

    def test_strongest_legal_payload_cannot_ignite(self) -> None:
        """方向层：合法域最强载荷（Πa=cap、|μa|=1）折减后 salience < 阈值 ⇒ 不点燃。

        代数界（数学席证明）：M3 先钳 Πa≤cap、γ 后乘 ⇒ Πa_declared ≤ γ·cap 恒成立。
        """
        cap = AffectState().external_prior_precision_cap
        prior = ("physio", (0.0, 1.0), (0.5, cap))
        expanded = expand_external_priors([prior], precision_cap=cap, max_streams=5)
        assert stream_salience(expanded[0][1], expanded[0][2]) < SALIENCE_THRESHOLD, (
            "γ·cap 档载荷必须亚阈——红=代数界前提（M3 钳制在 γ 前）被改动"
        )
        out = AffectCoreAgent()(_base_state(external_priors=[prior]))
        assert "physio" not in out["ignited_streams"]
        assert set(out["ignited_streams"]) >= {"survival", "appraisal", "value"}, (
            "内核流在场 ⇒ physio 落选出自阈值判据而非全员亚阈兜底"
        )

    def test_double_loss_zone_representative_closed(self) -> None:
        """方向层：双输区代表点（hrv 诚实 Πa=0.391 ∈ [0.359,1.0)，08-31 数学席新事实）
        折减后不点燃——「不合格但能点燃」区间在本侧关死。"""
        prior = ("hrv_rmssd", (0.0, 1.0), (0.5, 0.391))
        expanded = expand_external_priors([prior], precision_cap=0.8, max_streams=5)
        assert stream_salience(expanded[0][1], expanded[0][2]) < SALIENCE_THRESHOLD
        out = AffectCoreAgent()(_base_state(external_priors=[prior]))
        assert "hrv_rmssd" not in out["ignited_streams"], (
            "双输区代表点重新点燃=γ 层失效，见 notes/2026-09-01-gamma-physio-council.md"
        )

    def test_gamma_reproducible_from_delivery_inputs(self) -> None:
        """γ 可复现性：按 2026-09-01 交付件 §2（wire·二值锚 + ×1.3 保守档）现算，须与
        GAMMA_PHYSIO 逐位一致；并落在口径 C 阈值反解上界内（上界仅作边界检查，禁作输入）。
        """
        alpha_hat, beta_hat = -0.0105, 0.4267
        sigma_e = 0.1597
        se_alpha, se_beta = 0.0126 * 1.3, 0.0197 * 1.3
        expected = sigma_e**2 / (
            (alpha_hat + beta_hat - 1.0) ** 2 + sigma_e**2 + se_alpha**2 + se_beta**2
        )
        assert GAMMA_PHYSIO == pytest.approx(expected, rel=1e-12), (
            "GAMMA_PHYSIO 必须是交付输入的公式现算值——字面数漂移=输入被改而推导没跟上"
        )
        cap = AffectState().external_prior_precision_cap
        bound_c = (2 * SALIENCE_THRESHOLD - MIN_PRECISION) / cap
        assert GAMMA_PHYSIO < bound_c, "γ 超出口径 C 边界=G2 结构性质失守"


class TestExpandExternalPriorsNameUniqueness:
    """流名唯一性（code-reviewer WARN 2026-09-01·γ 落地门）：重名破坏按名寻址语义
    （cap_stream_weight 首匹配 ⇒ 逐流封顶只压第一条），隐性前提改硬约束。"""

    def test_duplicate_name_raises(self) -> None:
        with pytest.raises(ExternalPriorError, match="重复"):
            expand_external_priors(
                [
                    ("physio", (0.0, 0.5), (0.5, 0.3)),
                    ("physio", (0.0, 0.6), (0.5, 0.3)),
                ],
                precision_cap=0.8,
                max_streams=5,
            )

    def test_duplicate_non_physio_name_also_raises(self) -> None:
        """唯一性对全部外部流生效（报告/排除语义同样按名寻址），非 physio 特例。"""
        with pytest.raises(ExternalPriorError, match="重复"):
            expand_external_priors(
                [("face", (0.1, 0.2), (0.3, 0.3)), ("face", (0.2, 0.1), (0.3, 0.3))],
                precision_cap=0.8,
                max_streams=5,
            )

    def test_distinct_names_pass(self) -> None:
        result = expand_external_priors(
            [("eda_sc", (0.0, 0.5), (0.5, 0.3)), ("hrv_rmssd", (0.0, 0.5), (0.5, 0.3))],
            precision_cap=0.8,
            max_streams=5,
        )
        assert len(result) == 2


class TestAffectCoreNoPollution:
    """注入 external_priors 不污染 prior_mu/reward/features 等上游字段。"""

    def test_prior_mu_not_changed_by_external_priors(self) -> None:
        """注入 external_priors 前后 state.prior_mu 原值不变（节点无副作用）。"""
        state = _base_state(
            external_priors=[("face", (0.9, 0.9), (0.6, 0.6))],
        )
        prior_mu_before = state.prior_mu
        AffectCoreAgent()(state)
        assert state.prior_mu == prior_mu_before, "AffectCoreAgent 不应原地改变 state.prior_mu"

    def test_features_not_changed_by_external_priors(self) -> None:
        """注入 external_priors 后 state.features 原值不变（节点无副作用）。"""
        state = _base_state(
            external_priors=[("audio", (0.5, 0.8), (0.5, 0.6))],
        )
        features_before = list(state.features)
        AffectCoreAgent()(state)
        assert list(state.features) == features_before

    def test_prior_mu_same_with_and_without_external_priors(self) -> None:
        """注入 external_priors 不影响 prior_mu（由上游 Appraisal 决定·AffectCore 不重写）。"""
        state_with = _base_state(
            prior_mu=(0.2, 0.4),
            external_priors=[("face", (0.9, 0.9), (0.6, 0.6))],
        )
        state_without = _base_state(
            prior_mu=(0.2, 0.4),
            external_priors=[],
        )
        out_with = AffectCoreAgent()(state_with)
        out_without = AffectCoreAgent()(state_without)
        # prior_mu 不在 AffectCoreAgent 输出 dict 中（不重写上游字段）
        assert "prior_mu" not in out_with
        assert "prior_mu" not in out_without


class TestAffectCoreNodeContract:
    """节点契约：返回 dict 不含 external_priors key；key 集合是 AffectState.model_fields 子集。"""

    def test_return_dict_does_not_contain_external_priors(self) -> None:
        """AffectCoreAgent 返回 dict 不含 external_priors key（节点只增量、不回写注入字段）。"""
        state = _base_state(external_priors=[("face", (0.3, 0.5), (0.5, 0.5))])
        out = AffectCoreAgent()(state)
        assert "external_priors" not in out, (
            "AffectCoreAgent 不应在返回 dict 中写 external_priors（节点契约：不回写外部注入字段）"
        )

    def test_return_dict_no_external_priors_when_empty(self) -> None:
        """external_priors=[] 时返回 dict 同样不含 external_priors key。"""
        state = _base_state(external_priors=[])
        out = AffectCoreAgent()(state)
        assert "external_priors" not in out

    def test_return_keys_subset_of_affect_state_fields(self) -> None:
        """AffectCoreAgent 返回 dict 的 key 集合是 AffectState.model_fields 的子集（节点契约）。"""
        all_fields = set(AffectState.model_fields)
        state = _base_state(external_priors=[("face", (0.3, 0.5), (0.5, 0.5))])
        out = AffectCoreAgent()(state)
        assert set(out).issubset(all_fields), (
            f"返回 key 超出 AffectState.model_fields：{set(out) - all_fields}"
        )


# --------------------------------------------------------------------------
# 3. 默认值断言
# --------------------------------------------------------------------------


class TestAffectStateDefaults:
    """AffectState 默认字段值断言。"""

    def test_external_prior_precision_cap_default(self) -> None:
        """AffectState() 默认 external_prior_precision_cap==0.8。"""
        state = AffectState()
        assert state.external_prior_precision_cap == pytest.approx(0.8)

    def test_max_external_streams_default(self) -> None:
        """AffectState() 默认 max_external_streams==5。"""
        state = AffectState()
        assert state.max_external_streams == 5

    def test_external_priors_default_empty_list(self) -> None:
        """AffectState() 默认 external_priors==[]。"""
        state = AffectState()
        assert state.external_priors == []

    def test_external_priors_default_is_independent(self) -> None:
        """两个 AffectState() 实例的 external_priors 列表是独立对象（default_factory 隔离）。"""
        s1 = AffectState()
        s2 = AffectState()
        s1.external_priors.append(("face", (0.1, 0.2), (0.3, 0.4)))
        assert s2.external_priors == [], "两实例的 external_priors 应互相独立"

    def test_session_config_defaults(self) -> None:
        """SessionConfig() 默认 external_prior_precision_cap==0.8，max_external_streams==5。"""
        from src.orchestration.runner import SessionConfig

        cfg = SessionConfig()
        assert cfg.external_prior_precision_cap == pytest.approx(0.8)
        assert cfg.max_external_streams == 5

    def test_session_config_to_state_flags_contains_ep_fields(self) -> None:
        """to_state_flags() 含 external prior 的两个会话级校验字段（cap + max_streams）。"""
        from src.orchestration.runner import SessionConfig

        flags = SessionConfig().to_state_flags()
        assert "external_prior_precision_cap" in flags
        assert "max_external_streams" in flags
        assert flags["external_prior_precision_cap"] == pytest.approx(0.8)
        assert flags["max_external_streams"] == 5


# --------------------------------------------------------------------------
# 4. 跨轮不残留（ConversationSession + state_overrides）
# --------------------------------------------------------------------------


async def test_external_priors_no_residual_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """跨轮不残留（B1/B2·code-reviewer 2026-07-15）：external_priors 是 LangGraph LastValue
    channel——若某轮 ainvoke input 里不显式给它，channel 会从 checkpoint 恢复上一轮注入的
    非空 list（残留），违反 M4。step() 必须每轮在 base 里显式归零 external_priors=[]。

    直接 spy 每轮传给 graph.ainvoke 的 base dict（框架层归零，不用行为间接推断——后者无法
    区分「正确归零」与「残留但效果相等」，是 B2 指出的旧测试盲点）：
      - 注入轮：base["external_priors"] == 注入的 list；
      - 不注入轮：base["external_priors"] == []（上一轮非空 list=残留·KeyError=未归零）。
    """
    from src.orchestration.runner import ConversationSession, SessionConfig

    cfg = SessionConfig(workspace_enabled=True, rng_seed=7, affect_readout="map")
    stim = Stimulus(name="test", goal_congruence=0.4, intensity=0.7)
    session = ConversationSession(thread_id="t-ep-residual", config=cfg)

    captured: list[dict[str, object]] = []
    orig_ainvoke = session.graph.ainvoke

    async def _spy(base: dict[str, object], *args: object, **kwargs: object) -> object:
        captured.append(dict(base))
        return await orig_ainvoke(base, *args, **kwargs)

    monkeypatch.setattr(session.graph, "ainvoke", _spy)

    injected = [("face", (0.8, 0.9), (0.5, 0.6))]
    # 第一轮：注入 → ainvoke input 含该 list
    r1 = await session.step(stim, state_overrides={"external_priors": injected})
    assert captured[-1]["external_priors"] == injected, (
        "第一轮 state_overrides 的 external_priors 应写入 ainvoke input"
    )
    assert r1["valence_arousal"] is not None

    # 第二轮：不注入 → base 必须显式归零 external_priors=[]（B1 核心断言）；
    # 若无 B1 修复：base 无此键 → KeyError；若残留：为上一轮 injected → 断言失败。
    await session.step(stim, state_overrides=None)
    assert captured[-1]["external_priors"] == [], (
        "第二轮不传 state_overrides 时 external_priors 必须显式归零为 []（B1）；"
        "为上一轮非空 list = LastValue checkpoint 残留、违反 M4"
    )

    # 第三轮：注入不同流 → 覆盖生效（归零基准被 state_overrides 覆盖）
    injected3 = [("audio", (-0.2, 0.6), (0.1, 0.3))]
    await session.step(stim, state_overrides={"external_priors": injected3})
    assert captured[-1]["external_priors"] == injected3


# --------------------------------------------------------------------------
# 5. 风险 #125 seeking 回归
# --------------------------------------------------------------------------


async def test_seeking_no_lock_under_sustained_high_arousal_external_priors() -> None:
    """风险 #125：连续 K=10 轮注入「中性效价 + 高 arousal」外部流，valence 不被非法拉向正区锁定。

    测试逻辑：
      - 外部流 μv≈0（效价中性），μa=0.8（高唤醒）。
      - 用**负效价 stimulus**（goal_congruence=-0.5）提供负效价先验基线。
      - 若 external_priors 正确独立注入（不污染 appraisal/survival 基线），
        valence 应保持在负区附近（由 stimulus 主导）——外部流只拉动 arousal。
      - 若 external_priors 错误地把 μv=0 拉向正区（如替代了 appraisal 基线），
        valence 会被意外拉正，产生 seeking 锁定。
      - 断言：K 轮中 valence 不全部 >0（外部流不应凌驾于负效价 stimulus 之上）。

    注：arousal 维被高 arousal 外部流拉高是预期行为（M1 合规）；只断言 valence 不锁定。
    """
    from src.orchestration.runner import ConversationSession, SessionConfig

    K = 10
    cfg = SessionConfig(
        workspace_enabled=True,
        rng_seed=99,
        affect_readout="map",  # MAP 读出排除随机性，让效价方向断言更稳定
    )
    # 外部流：μv=0（效价中性），μa=0.8（高唤醒），Πa 高（接近 cap），Πv 低（不影响效价）
    neutral_valence_high_arousal_priors = [
        ("face", (0.0, 0.8), (0.05, 0.75)),  # μv=0 → 不拉 valence；Πa=0.75 → 强拉 arousal
        ("audio", (0.0, 0.75), (0.05, 0.70)),  # 同上
    ]
    # 负效价 stimulus：goal_congruence=-0.5 → 先验偏负 valence
    neg_stim = Stimulus(name="neg_event", goal_congruence=-0.5, intensity=0.7)

    session = ConversationSession(thread_id="t-seeking-regr", config=cfg)
    valence_positive_count = 0
    for _ in range(K):
        result = await session.step(
            neg_stim,
            state_overrides={"external_priors": neutral_valence_high_arousal_priors},
        )
        va = result["valence_arousal"]
        if va is not None and va[0] > 0:
            valence_positive_count += 1

    # 关键断言：中性效价外部流不应凌驾负效价 stimulus，使 valence 全轮锁定正区
    assert valence_positive_count < K, (
        f"连续 {K} 轮注入中性效价（μv=0）外部流后，所有 {K} 轮 valence 都被拉正（>0）；"
        f"这说明 external_priors 对 valence 的影响超出了其精度权重应有的份额，"
        f"可能正在替代/覆写 appraisal 基线（风险 #125 seeking 锁定变体）。"
        f"如果此测试红了请回报实现工程师检查 expand_external_priors 注入路径。"
    )


async def test_seeking_no_lock_neutral_stimulus_high_arousal() -> None:
    """风险 #125 补强（code-reviewer W3）：**中性 stimulus**（goal_congruence≈0，无强负效价锚）
    + 连续高 arousal 中性效价外部流——最敏感场景。此时 appraisal 不提供强 valence 锚点，
    若 external_priors 错误地进 occ_prior 入口或污染 survival（而非独立低精度流竞争融合），
    μv=0 的外部流叠加高 arousal 会把 valence 推离中性、形成 seeking 直流偏置锁定。

    断言：(1) valence 不单调锁定正区；(2) valence 幅度保持有界（中性效价外部流不应把
    valence 推离中性太远）——直接检出「高 arousal 外部流泄漏进 valence」的实现错误。
    """
    from src.orchestration.runner import ConversationSession, SessionConfig

    K = 12
    cfg = SessionConfig(workspace_enabled=True, rng_seed=123, affect_readout="map")
    # 中性效价 + 高 arousal 外部流（μv=0·Πa 高·Πv 低）
    priors = [
        ("face", (0.0, 0.85), (0.05, 0.75)),
        ("audio", (0.0, 0.80), (0.05, 0.70)),
    ]
    neutral_stim = Stimulus(name="neutral", goal_congruence=0.0, intensity=0.5)

    session = ConversationSession(thread_id="t-seeking-neutral", config=cfg)
    valences: list[float] = []
    for _ in range(K):
        result = await session.step(neutral_stim, state_overrides={"external_priors": priors})
        va = result["valence_arousal"]
        assert va is not None
        valences.append(va[0])

    # (1) 不单调锁定正区
    assert not all(v > 0.1 for v in valences), (
        f"中性 stimulus + 中性效价高 arousal 外部流连续 {K} 轮，valence 全 >0.1（正区锁定）；"
        f"外部流不应把中性效价推成持续正区。valences={[round(v, 3) for v in valences]}"
    )
    # (2) valence 幅度有界——高 arousal 外部流不应泄漏进 valence 维（μv=0 → valence 应近中性）
    assert max(abs(v) for v in valences) < 0.35, (
        f"中性效价外部流（μv=0）把 valence 推离中性超过 0.35——疑似高 arousal 泄漏进 valence 维"
        f"或误进 occ_prior/survival 入口。valences={[round(v, 3) for v in valences]}"
    )


class TestExternalPriorErrorAttribution:
    """归责可辨：external_priors 校验失败抛专属类型，内核其它错不得被贴成它。

    背景（议会 2026-07-29 第五轮校验 §四-5）：边界层 `server.py` 原先用**一个**
    `except ValueError` 包住整个 `session.step()`（全图执行），把内核任何位置抛出的
    `ValueError` 一律贴成「external_priors 校验失败（指向 MCP 传参）」。后果是误导性甩锅——
    client 照着改传参永远改不好，而活跃会话的 config 不可变 → 无法自救，
    表现为 open 成功、**每 step 崩**。
    """

    def test_all_expand_failures_raise_dedicated_type(self) -> None:
        """M3/M6/M7 与形状校验**全部**抛 ExternalPriorError（漏一处就恢复成误导性甩锅）。"""
        cases = [
            [("a", (0.0, 0.0), (0.1, 0.1))] * 99,  # M6 流数超上界
            [("a", (0.0, 0.0), (0.0, 0.1))],  # M3 精度非正
            [("a", (0.0, 0.0), (9.9, 0.1))],  # M3 超 cap
            [("a", (2.0, 0.0), (0.1, 0.1))],  # M7 μ 越域
            ["not-a-tuple"],  # 形状不良构
        ]
        for priors in cases:
            with pytest.raises(ExternalPriorError):
                expand_external_priors(priors, precision_cap=0.8, max_streams=5)  # type: ignore[arg-type]

    def test_dedicated_type_stays_backward_compatible(self) -> None:
        """仍是 ValueError 子类——既有 `except ValueError` 的调用方/用例不受影响。"""
        assert issubclass(ExternalPriorError, ValueError)
        with pytest.raises(ValueError):
            expand_external_priors(
                [("a", (0.0, 0.0), (0.0, 0.1))], precision_cap=0.8, max_streams=5
            )

    def test_kernel_errors_are_not_external_prior_errors(self) -> None:
        """反向：内核其它 fail-fast **不得**是 ExternalPriorError，否则归责又混回去了。

        取 `hierarchical_fuse` 的 coupling 越界作代表——它是「会话级配置不兼容」那一类，
        改 external_priors 传参对它毫无用处。
        """
        with pytest.raises(ValueError) as ei:
            hierarchical_fuse(
                [("appraisal", (0.1, 0.1), (1.0, 1.0)), ("survival", (0.2, 0.2), (0.4, 0.4))],
                low_names=frozenset({"survival"}),
                layers=2,
                coupling=1.5,
            )
        assert not isinstance(ei.value, ExternalPriorError), (
            "内核配置错被判成 external_priors 传参错——归责混淆已回归"
        )

    def test_boundary_layer_separates_the_two_attributions(self) -> None:
        """边界层源码里两条 except 分支必须并存，且内核那条明说「改传参无效」。

        结构性检查而非跑 server——起 MCP server 需要额外依赖，而这里要锁的是
        「有没有把两类错分开报」这件事本身。
        """
        from src.mcp_server import server

        with open(server.__file__, encoding="utf-8") as fh:
            text = fh.read()
        # 只看 step 调用点之后的片段——server.py 别处还有一个无关的 except ValueError
        # （ZERO_MCP_HTTP_PORT 的 int() 解析），全局字符串比较会误判。
        tail = text[text.index("await session.step(") :]
        assert "except ExternalPriorError as e:" in tail
        assert "except ValueError as e:" in tail
        assert tail.index("except ExternalPriorError as e:") < tail.index(
            "except ValueError as e:"
        ), "专属类型的 except 必须在裸 ValueError **之前**，否则永远进不去（子类被父类先捕获）"
        assert "非** external_priors 传参问题，改传参无效" in tail
