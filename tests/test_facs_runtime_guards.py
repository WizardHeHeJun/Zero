r"""FACS 解码器**运行时守卫**：与训练轮数解耦的行为约束。

**为什么与 `test_facs_au_expansion.py` 分开**：那边测的是**解析占位**（`decode_channels`，
纯 Python 映射，行为由代码写死）；这里测的是**训出来的权重**在运行时的行为。两者是
`CompositeChannelDecoder` residual 混合的两侧，坏法完全不同——占位坏在映射公式，
权重坏在拟合方向/饱和，占位的单调性测试一条也抓不住权重的问题。

**为什么这些断言与轮数解耦**：整改 P1-1 把停机判据从 `epochs=300` 魔数换成了 plateau，
P1-5 又证明轮数会显著改变模型（FACS 300→1500 技能分 +0.019→+0.189）。轮数一变，
任何写死 loss 数值的断言都会失效，而「不饱和 / AU12 随效价升 / AU04 随效价降」这类
**方向与值域性质**在任何合理轮数下都该成立——守卫要守的是这些。

四层：
  ① 合成 CSV 训练 → 方向学得对（data-independent，任何环境都跑）
  ② 真权重（`artifacts/facs_decoder_ext_v2.pt`，缺则跳过）→ 不饱和 / 角点区间 /
     **运行时消费的 9 维逐维**不退化成常数 / 强信号维方向在**每个 arousal 档**都正确 /
     无结构性大幅逆行
  ③ **经生产工厂 + composite 混合**的端到端 → 13 键合法性 / 方向在混合后仍成立 /
     coping 分野零回归。②③ 的区别：②直接 `predict_facs`，③走 `--chat` 真实跑的那条链
  ④ **旧缺陷的防复发闸**：AU15 方向（2026-07-28 重训已修）+ 复现该缺陷的因果测试

**覆盖边界（如实标注，别以为 9 维都被方向守卫罩住了）**：

| 维 | 不退化成常数 | 方向守卫 | 说明 |
| --- | --- | --- | --- |
| AU04 / AU06 / AU07 / AU12 | ✅ | ✅ 数据 \|r\| ≥ 0.5 | 21 个 arousal 档端点符号全对 |
| AU15 / AU17 | ✅ | ✅ | AU15 曾方向反、已修，另有专门的防复发类 |
| AU05 / AU26 / intensity | ✅ | ❌ **无** | 数据里 \|r\| 仅 0.240/0.089/0.390，无方向可断 |

即 9 维全部有「不退化成常数」保护，但**方向**只守得住数据里信号确实强的 6 维。

⚠ **这个缺口的边界是实测过的**：把 AU26 改成 `0.15 − 0.05·valence`（方向反了但没塌成常数），
本文件全部测试**照常全绿**。弱信号维的方向 bug 目前确实无人看守——要补只能等数据侧变强
（换数据源/换表征，属 P3），加断言硬凑只会造出一个测不准的假守卫。

⚠ **不要求逐点严格单调**：旧权重（欠训练 300–400 步）近似线性、处处单调，那是**伪影**；
重训到 5000 步学到 valence×arousal 交互后会出现小幅逆行。守的是方向 + 逆行幅度有界。
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.agents.models.facs_decoder import FACS_KEYS_EXT

REAL_EXT_WEIGHTS = Path("artifacts/facs_decoder_ext_v2.pt")
FACS_EXT_CSV = "data/facs/labels_ext.csv"

# 饱和阈：sigmoid 输出贴到这两端就意味着梯度消失、该维实际已失去对 (v,a) 的响应能力
SATURATION_LOW = 0.02
SATURATION_HIGH = 0.98

# 扫描网格覆盖整个 [-1,1]^2，**含训练锚点包络之外**（实测锚点仅落在
# valence [-0.75, 0.9] × arousal [-0.75, 0.78]）——运行时 (v,a) 来自 AffectCore，
# 不受锚点包络约束，包络外的外推行为必须一并守住。
GRID = [i / 10.0 for i in range(-10, 11)]

# 每一维的全网格极差下限。**必须逐维查**：只抽查几维时，把某一维钉成常数不会被发现——
# 全局「13 维极差」之类的聚合判据在其余维照常变化时依旧成立（code-reviewer 用把 AU17
# 钉成 0.15 的最小 bug 实证了这一点，当时 9 条测试无一变红）。
# 实测各维全网格极差（2026-07-28 重训后）：intensity 最小 0.081、AU05 最大 0.808，
# 取 0.02 对最紧的一维仍有 4 倍余量——够松，不会锁死重训；够紧，能抓住塌成常数。
MIN_GRID_SPAN = 0.02

# 沿 valence 允许的单步逆行上限，占该 arousal 档总跨度的比例。
# 实测重训后逐档取比值再取最大：**AU04 11.99%（最松，@a=+1.0）** > AU15 10.73% >
# AU17 8.16% > AU07 1.55% > AU06 1.27% > AU12 0.00%。取 20% 留 **1.67 倍**余量。
# **不要求逐点严格单调**——理由见 `test_no_gross_monotonicity_reversal` 的 docstring
# （旧权重的严格单调是欠训练伪影）。
MAX_REVERSAL_FRACTION = 0.20

# 训练数据（`data/facs/labels_ext.csv`，1634 行 / 38 锚点）里各 AU 与 valence 的**锚点级**
# 相关系数，2026-07-28 实测。只对 |r| ≥ 0.5 的维断言方向——数据里信号本就微弱的维
# （AU05 −0.240 / AU26 +0.089 / intensity +0.390）没有可断言的方向，硬断只会制造假守卫。
#
# ⚠ 断的是**净变化的符号**而非逐点单调——理由见 `test_no_gross_monotonicity_reversal`。
STRONG_VALENCE_SIGNAL: dict[str, float] = {
    "AU04": -0.701,
    "AU06": +0.679,
    "AU07": +0.616,
    "AU12": +0.739,
    "AU15": -0.526,  # 旧权重曾学成 +0.115（方向反），2026-07-28 重训后 −0.047
    "AU17": -0.538,
}

requires_real_weights = pytest.mark.skipif(
    not REAL_EXT_WEIGHTS.exists(),
    reason=f"{REAL_EXT_WEIGHTS} 不存在（artifacts/ 随 gitignore，权重走 Release）",
)


def _write_synthetic_ext_csv(path: Path, *, n_anchors: int = 12, rows_per_anchor: int = 4) -> None:
    """写一份方向明确的合成 13-AU CSV：AU12/AU06 随 valence 升，AU04/AU15 随 valence 降。

    刻意不用 `decode_channels` 蒸馏（那会把占位的映射公式重新测一遍）——这里要的是一份
    **方向由构造决定**的数据，用来验「训练链路能把数据里的方向学出来」。
    """
    header = ["valence", "arousal", *FACS_KEYS_EXT]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for i in range(n_anchors):
            v = round(-0.9 + 1.8 * i / (n_anchors - 1), 3)
            a = round(0.6 * v, 3)
            for r in range(rows_per_anchor):
                jitter = 0.01 * r  # 组内方差，避免锚点退化成单点
                pos, neg = max(0.0, v), max(0.0, -v)
                # 填充值取 0.15 而非贴近地板的 0.05：真实数据各 AU 均值在 0.075–0.304，
                # 用 0.05 做目标会让「不饱和」断言测的是构造缺陷（拟合 0.05 的模型外推到
                # 角点自然掉出 [0.02, 0.98]），而不是模型行为。
                row: dict[str, float] = dict.fromkeys(header, 0.15)
                row.update(
                    {
                        "valence": v,
                        "arousal": a,
                        "AU06": round(0.15 + 0.7 * pos + jitter, 4),
                        "AU12": round(0.15 + 0.7 * pos + jitter, 4),
                        "AU04": round(0.15 + 0.7 * neg + jitter, 4),
                        "AU15": round(0.15 + 0.7 * neg + jitter, 4),
                        "intensity": round(0.15 + 0.4 * abs(v), 4),
                    }
                )
                writer.writerow(row)


def _sweep(model: object, au: str, *, arousal: float) -> list[float]:
    """沿 valence 扫一档 arousal，返回该 AU 的输出序列。"""
    return [model.predict_facs(v, arousal)[au] for v in GRID]  # type: ignore[attr-defined]


def _net_delta(model: object, au: str, *, arousal: float = 0.0) -> float:
    """valence 从 −1 到 +1 的净变化量（正=随效价升）。"""
    seq = _sweep(model, au, arousal=arousal)
    return seq[-1] - seq[0]


# ─── ① 合成 CSV：训练链路能把数据里的方向学出来（data-independent） ──────────────


class TestSyntheticDirectionsAreLearned:
    """构造方向明确的数据 → 训练 → 方向必须被学出来。

    这层不依赖任何本地权重或真实数据集，任何环境都跑得到；它守的是「(v,a)→AU 回归 +
    训练循环」这条链没坏，而不是某份权重的质量。

    ⚠ **已实测的盲区**（注入 bug 验证时发现，如实记下）：本层对**训练与推理对称**的缺陷
    无能为力。把 `FacsDecoder.forward` 改成 `self.net(x * [[-1, 1]])`（valence 符号反转）
    后，本层三条测试**全部照常通过**——因为模型是带着这个 bug 训练的，权重会学出相反的
    符号把它补偿回来，端到端方向仍然正确。这类缺陷只能由下面用**既有权重**的
    `TestRealWeightsRuntimeGuards` 抓到（它当时确实红了）。两层缺一不可。
    """

    @staticmethod
    def _train_synthetic(tmp_path: Path) -> object:
        pytest.importorskip("torch")
        from scripts.train_facs import train
        from src.agents.models.facs_decoder import load_facs_decoder

        csv_path = tmp_path / "synthetic_ext.csv"
        _write_synthetic_ext_csv(csv_path)
        out = tmp_path / "facs_syn.pt"
        # 必须显式 stop="fixed"：train() 默认 stop="plateau"，只传 epochs 会被静默
        # 按 plateau 跑到 max_epochs（整改 P1-1 起的语义）。
        train(str(csv_path), extended=True, epochs=1200, stop="fixed", seed=0, out=str(out))
        return load_facs_decoder(str(out), extended=True)

    def test_au12_au06_increase_with_valence(self, tmp_path: Path) -> None:
        model = self._train_synthetic(tmp_path)
        for au in ("AU12", "AU06"):
            delta = _net_delta(model, au)
            assert delta > 0.05, (
                f"{au} 应随 valence 升（合成数据如此构造），实测净变化 {delta:+.4f}"
            )

    def test_au04_au15_decrease_with_valence(self, tmp_path: Path) -> None:
        model = self._train_synthetic(tmp_path)
        for au in ("AU04", "AU15"):
            delta = _net_delta(model, au)
            assert delta < -0.05, (
                f"{au} 应随 valence 降（合成数据如此构造），实测净变化 {delta:+.4f}"
            )

    def test_no_saturation_after_training(self, tmp_path: Path) -> None:
        """训完的模型在整个网格（含包络外）不得贴到 sigmoid 两端。"""
        model = self._train_synthetic(tmp_path)
        for v in GRID:
            for a in GRID:
                out = model.predict_facs(v, a)  # type: ignore[attr-defined]
                for key, value in out.items():
                    assert SATURATION_LOW <= value <= SATURATION_HIGH, (
                        f"({v:+.1f},{a:+.1f}) 的 {key}={value:.5f} 已饱和"
                        f"（须落在 [{SATURATION_LOW}, {SATURATION_HIGH}]）"
                    )


# ─── ② 真权重：运行时实际用的那份 ────────────────────────────────────────────────


@requires_real_weights
class TestRealWeightsRuntimeGuards:
    """`facs_decoder_ext_v2.pt`——`--chat` 运行时真正加载的那份权重。

    artifacts/ 随 gitignore，缺文件即跳过（权重走 Release，见 `WEIGHTS.md`）。
    """

    @staticmethod
    def _load() -> object:
        pytest.importorskip("torch")
        from src.agents.models.facs_decoder import load_facs_decoder

        return load_facs_decoder(str(REAL_EXT_WEIGHTS), extended=True)

    def test_no_saturation_anywhere_including_outside_envelope(self) -> None:
        """整个 [-1,1]^2 网格 × 13 键都不得饱和——**包括训练锚点包络之外**。

        锚点只覆盖 valence [-0.75, 0.9] × arousal [-0.75, 0.78]，而运行时 (v,a) 由
        AffectCore 给出、可以走到角点。外推处饱和会让表情在情绪最强时反而失去分辨率。
        """
        model = self._load()
        for v in GRID:
            for a in GRID:
                out = model.predict_facs(v, a)  # type: ignore[attr-defined]
                for key, value in out.items():
                    assert SATURATION_LOW <= value <= SATURATION_HIGH, (
                        f"({v:+.1f},{a:+.1f}) 的 {key}={value:.5f} 已饱和"
                    )

    def test_no_runtime_dim_degenerates_to_constant(self) -> None:
        """**运行时消费的每一维**都必须对 (v,a) 有真实响应，不能退化成常数输出器。

        文本词袋通道就栽在这里（干净口径下几乎是常数输出器，技能分 0.028）——
        裸 loss 看不出来，得直接查动态范围。

        **逐维查、不抽查**：本测试初版只查了 AU12/AU04/AU06 三维，code-reviewer 把 AU17
        钉成常数 0.15 后 9 条测试无一变红——单维塌陷在聚合判据下完全隐形。维集取
        `_RUNTIME_CONSUMED_COLUMNS`（剔除默认 α=1.0 下被解析占位整个替换的
        AU23/01/02/20），与 `train_facs.py` 的评分口径同源、不另抄一份。
        """
        from scripts.train_facs import _RUNTIME_CONSUMED_COLUMNS

        model = self._load()
        runtime_keys = [FACS_KEYS_EXT[i] for i in _RUNTIME_CONSUMED_COLUMNS]
        assert len(runtime_keys) == 9, f"运行时消费维数应为 9，实为 {len(runtime_keys)}"

        for au in runtime_keys:
            values = [
                model.predict_facs(v, a)[au]  # type: ignore[attr-defined]
                for v in GRID
                for a in GRID
            ]
            span = max(values) - min(values)
            assert span > MIN_GRID_SPAN, (
                f"{au} 在整个 (v,a) 网格上的极差仅 {span:.5f}（阈值 {MIN_GRID_SPAN}），"
                f"该维已近似常数——运行时它对情绪不再有任何响应"
            )

    def test_direction_agrees_with_training_data_correlation(self) -> None:
        """信号强的维（|r| ≥ 0.5），模型的净变化方向必须与训练数据的相关号一致。

        「学出来的方向不该和数据相反」对每一个强信号维都该成立。**AU15 曾栽在这条上**
        （旧权重 −0.526 的数据学成 +0.115），2026-07-28 重训后已修（−0.047），
        现与其余 5 维一并断言、不再排除；防复发另见 `TestAU15DirectionRegression`。

        重训后 6 维的余量都不算紧（Δ：AU12 +0.660 / AU04 −0.296 / AU17 −0.070 /
        AU15 −0.047）——最紧的是 AU15，翻号即红。
        """
        model = self._load()
        for au, r in STRONG_VALENCE_SIGNAL.items():
            delta = _net_delta(model, au)
            assert delta * r > 0, (
                f"{au} 的方向与训练数据相反：数据锚点级 r={r:+.3f}，"
                f"模型 Δ(v:−1→+1)={delta:+.4f}（应同号）"
            )

    def test_corner_outputs_in_plausible_range(self) -> None:
        """四角点 + 原点：全部输出须在 (0,1) 内且非退化。

        不断言具体数值——那会把某一次训练的结果锁死；只断言「载入后能出合法的 13 键、
        值域合理」，这是权重换代后仍应成立的性质。
        """
        model = self._load()
        for v, a in [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0), (0.0, 0.0)]:
            out = model.predict_facs(v, a)  # type: ignore[attr-defined]
            assert set(out) == set(FACS_KEYS_EXT), f"({v},{a}) 键集不全：{sorted(out)}"
            assert all(0.0 < x < 1.0 for x in out.values()), f"({v},{a}) 有越界值：{out}"
            assert max(out.values()) - min(out.values()) > 0.02, (
                f"({v},{a}) 的 13 维几乎相等（{out}），表情将无分辨率"
            )

    def test_direction_holds_at_every_arousal_level(self) -> None:
        """**每个 arousal 档**上，valence 端点差的符号都要与训练数据的相关号一致。

        这比「只看 a=0 的净变化」强得多（21 个 arousal 档 × 6 维），也比「逐点严格单调」
        更站得住——见下条测试对逐点单调为何不能要求的说明。实测 2026-07-28 重训后
        **21/21 档全部正确**，6 个方向维无一例外：

        | AU | 期望 | 端点差范围（21 档） |
        | --- | --- | --- |
        | AU12 / AU06 / AU07 | 升 | +0.375~+0.760 / +0.422~+0.622 / +0.195~+0.450 |
        | AU04 / AU15 / AU17 | 降 | −0.331~−0.070 / −0.062~−0.036 / −0.088~−0.063 |
        """
        model = self._load()
        for au, r in STRONG_VALENCE_SIGNAL.items():
            for a in GRID:
                seq = _sweep(model, au, arousal=a)
                endpoint_diff = seq[-1] - seq[0]
                assert endpoint_diff * r > 0, (
                    f"{au} 在 arousal={a:+.1f} 档上方向与数据相反："
                    f"数据 r={r:+.3f}，端点差 f(v=+1)−f(v=−1)={endpoint_diff:+.5f}（应同号）"
                )

    def test_no_gross_monotonicity_reversal(self) -> None:
        """沿 valence 允许小幅逆行，但**单步逆行不得超过该档总跨度的 20%**。

        **为什么不要求逐点严格单调**：初版这条测的是「AU12 升 / AU04 降逐点无逆行」，
        对**旧权重**成立——但那是**欠训练的伪影**：训了 300–400 步的网络还接近线性初始化，
        自然处处单调。2026-07-28 重训到 5000 步后模型学到了 valence×arousal 的交互，
        AU04 出现 19/420 步的小幅逆行、AU06 出现 78/420——**模型变好了，旧断言反而红了**。
        把它改成「严格单调」的更弱版本是错的方向；正确做法是断言真正该守的东西：
        方向（上一条，逐档端点符号）+ 不出现结构性的大幅反转（本条）。

        实测最大单步逆行占该档跨度比（逐档取比值后取最大）：**AU04 11.99%（最松）** >
        AU15 10.73% > AU17 8.16% > AU07 1.55% > AU06 1.27% > AU12 **0.00%**。
        阈值 20% 对最松的 AU04 仍有 **1.67 倍**余量。
        """
        model = self._load()
        for au, r in STRONG_VALENCE_SIGNAL.items():
            for a in GRID:
                seq = _sweep(model, au, arousal=a)
                span = max(seq) - min(seq)
                if span <= 0:
                    continue
                for i in range(len(seq) - 1):
                    reversal = -(seq[i + 1] - seq[i]) * (1 if r > 0 else -1)
                    assert reversal <= MAX_REVERSAL_FRACTION * span, (
                        f"{au} 在 arousal={a:+.1f}、v={GRID[i]:+.1f}→{GRID[i + 1]:+.1f} 处"
                        f"大幅逆行 {reversal:.5f}，达该档跨度 {span:.5f} 的 {reversal / span:.1%}"
                        f"（上限 {MAX_REVERSAL_FRACTION:.0%}）"
                    )


# ─── ③ 经生产工厂 + composite 混合的端到端 ─────────────────────────────────────


@requires_real_weights
class TestRealWeightsThroughCompositeEndToEnd:
    """真权重经**生产工厂 + composite 混合**后的行为——比直接 `predict_facs` 多一层。

    **为什么单独测这一层**（code-reviewer 指出的缺口）：本文件其余测试都绕过
    `CompositeChannelDecoder` 直接调 `load_facs_decoder().predict_facs()`；而
    `tests/test_chat_driver_expression_decoder.py` 虽然走工厂，用的却是**随机初始化**的
    权重（现场存一份），只验接线契约。两边都没覆盖「**真**权重 + composite 的
    `k_arousal`/`k_coping`/`residual_alpha` 混合 + `--chat` 生产工厂」这个组合——
    而这正是运行时实际跑的那条路径。

    整改计划「明确排除」条款要求权重重训后重跑运行时核验（13 键 `[0,1]` / AU 方向 /
    `--chat` 端到端），本类固化第三项，使其不再依赖某次一次性实跑。
    """

    @staticmethod
    def _decoder(monkeypatch: pytest.MonkeyPatch) -> object:
        """装配生产 decoder。**先清空全部相关 env 再设**——否则跑测环境里一个悬空的
        `ZERO_PROSODY_MODEL_PATH` 就能让这三条测试因与 FACS 无关的原因全部报错
        （code-reviewer 实测过）。`_FACS_ENVS` 直接从姊妹文件 import，不另抄一份：
        将来加了新通道门控，两处会一起生效而不是悄悄分叉。
        """
        pytest.importorskip("torch")
        from src.orchestration.chat_driver import _build_expression_decoder
        from tests.test_chat_driver_expression_decoder import _FACS_ENVS

        for name in _FACS_ENVS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("ZERO_FACS_MODEL_PATH", str(REAL_EXT_WEIGHTS))
        monkeypatch.setenv("ZERO_FACS_EXTENDED", "1")
        decoder = _build_expression_decoder(facs_extended=True)
        assert decoder is not None, "设了 ZERO_FACS_MODEL_PATH 就该构造真 decoder，不该回落占位"
        return decoder

    def test_all_keys_legal_through_composite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """经 composite 混合后 13 键仍齐全、值域 [0,1]、且是纯 python float（zero-link 契约）。"""
        decoder = self._decoder(monkeypatch)
        probes = [(-0.6, 0.6, 0.5), (-0.6, 0.6, -0.5), (0.7, 0.5, 0.0), (-0.5, -0.4, 0.0)]
        probes += [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (-1.0, -1.0, 0.0)]
        for v, a, coping in probes:
            facs = decoder.predict_channels_coping(  # type: ignore[attr-defined]
                v, a, coping_potential=coping, facs_extended=True
            )["facs_au"]
            assert set(facs) == set(FACS_KEYS_EXT), f"({v},{a},c={coping}) 键集不全"
            for key, value in facs.items():
                assert isinstance(value, float), f"{key} 不是 python float（JSON 不安全）"
                assert 0.0 <= value <= 1.0, f"({v},{a},c={coping}) 的 {key}={value} 越界"

    def test_valence_directions_survive_composite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """方向在混合后仍然成立——**AU15 是本轮重训修好的那一维，重点看它**。

        实测（2026-07-28 重训后）：AU12 喜悦 0.367 vs 悲伤 0.067；
        **AU15 悲伤 0.110 vs 喜悦 0.064**（旧权重下这一条是反的）；AU04 悲伤 0.350 vs 喜悦 0.163。
        """
        decoder = self._decoder(monkeypatch)
        joy = decoder.predict_channels_coping(  # type: ignore[attr-defined]
            0.7, 0.5, coping_potential=0.0, facs_extended=True
        )["facs_au"]
        sad = decoder.predict_channels_coping(  # type: ignore[attr-defined]
            -0.5, -0.4, coping_potential=0.0, facs_extended=True
        )["facs_au"]
        assert joy["AU12"] > sad["AU12"], (
            f"AU12 喜悦应高于悲伤：{joy['AU12']:.3f} vs {sad['AU12']:.3f}"
        )
        assert sad["AU15"] > joy["AU15"], (
            f"AU15 悲伤应高于喜悦（这正是本轮重训修好的方向）："
            f"{sad['AU15']:.3f} vs {joy['AU15']:.3f}"
        )
        assert sad["AU04"] > joy["AU04"], (
            f"AU04 悲伤应高于喜悦：{sad['AU04']:.3f} vs {joy['AU04']:.3f}"
        )

    def test_coping_split_unaffected_by_retrain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """愤怒/恐惧的 coping 分野不受重训影响——那几维走解析占位，默认 α=1.0 下整个替换。

        这条是**零回归断言**：重训只该改通用 AU，不该动 coping 判别 AU 的行为。
        """
        decoder = self._decoder(monkeypatch)
        angry = decoder.predict_channels_coping(  # type: ignore[attr-defined]
            -0.6, 0.6, coping_potential=0.8, facs_extended=True
        )["facs_au"]
        fear = decoder.predict_channels_coping(  # type: ignore[attr-defined]
            -0.6, 0.6, coping_potential=-0.8, facs_extended=True
        )["facs_au"]
        assert angry["AU23"] > fear["AU23"], "AU23（唇紧·愤怒）应随 coping>0 高于 coping<0"
        assert fear["AU20"] > angry["AU20"], "AU20（唇横拉·恐惧）应随 coping<0 高于 coping>0"


# ─── ④ 旧缺陷的防复发闸 ──────────────────────────────────────────────────────────


@requires_real_weights
class TestAU15DirectionRegression:
    """**曾经的缺陷，已修**：AU15 沿 valence 的方向一度与训练数据的相关号相反。

    这个类保留下来是为了**防止它再犯**——成因是可以重现的，不是一次性事故。

    ## 事故与修复

    | 量 | 旧权重（≈300–400 步） | 新权重（5000 步·seed 0） |
    | --- | --- | --- |
    | Δ(AU15) @ a=0 | **+0.115（方向反）** | **−0.047（正确）** |
    | 强信号维方向正确数（\\|r\\|≥0.5，共 6 维） | 5/6 | **6/6** |
    | 全网格 13 键输出上界 | 0.377（从不出高强度 AU） | 0.850 |
    | train MSE | 0.0343805 | 0.0294785 |

    数据里 AU15（压嘴角·悲伤 AU）与效价锚点级相关 **r = −0.526**，旧权重却学成正相关。
    AU15 **不在** `_COPING_DRIVEN_AUS` 内，composite 默认 α=1.0 下不被解析占位覆盖，
    所以那条错误方向**真实进入过运行时表情**。

    ## 根因：全局欠训练（不是数据、不是损失、不是容量）

    旧权重训于 **300–400 步**（行为指纹推定：其 train MSE 0.0343805，新训练 500 步时
    10/10 已低于该值；它无 provenance sidecar，训于 P0-2 之前，且 `torch.manual_seed`
    当时还没加进 `train_facs.py`，**无法精确复现**）。实测方向正确的 seed 数：
    300 步 4/10 → 1000 步 9/10 → **1500/2000/3000 步 10/10**。≤500 步时符号很大程度上
    是初始化残留（init 符号与 300 步符号一致 7/10）。

    机制：AU15 的 μ=0.096 离 sigmoid 初始输出 0.5 最远、早期梯度几乎全用于压截距，
    而它的斜率信号（between_sd 全场最小）要等截距到位才接管——把输出层 bias 初始化成
    `logit(μ_j)` 后 @300 步方向 **8/8 全对**，这是机制的决定性验证。

    ⚠ 曾写在此处的推测（「AU15 最稀疏 → 梯度被大方差维淹没 + hidden=16 容量不足」）
    **已被证伪且前提方向相反**：AU15 的 init 梯度范数 0.02529 是 13 维**全场最大**
    （AU04 才最小 0.00792）；`hidden=16/1 层` 训到 1500 步即 10/10 正确，非表征瓶颈。

    ## 为什么这个类还留着

    修法之所以「免费」，是因为 `train_facs` 默认 `stop="plateau"` + `max_epochs=5000`
    而 FACS 通道 `rel_tol=1e-4` 从不触发、必定跑满。**一旦有人把默认改回定轮数、
    或把 `max_epochs` 调到 1500 以下，这个缺陷会原样复发**——本类就是那道闸。

    ⚠ 别把它读成「预测变准了」：AU15 在留出面上**没有任何配置打赢常数基线**
    （最好 −0.0097）。修它的价值是移除一个方向错误的运行时信号。
    完整纪要见 `notes/2026-07-28-au15-direction-root-cause.md`。
    """

    def test_au15_decreases_with_valence(self) -> None:
        """当年 strict xfail 的那条断言，现在应当真实通过。"""
        pytest.importorskip("torch")
        from src.agents.models.facs_decoder import load_facs_decoder

        model = load_facs_decoder(str(REAL_EXT_WEIGHTS), extended=True)
        delta = _net_delta(model, "AU15")
        assert delta < 0.0, (
            f"AU15 方向缺陷复发：应随 valence 降（数据锚点级 r=−0.526），实测 Δ={delta:+.4f}。"
            "首要怀疑对象是训练轮数被调回 1500 步以下——见本类 docstring。"
        )

    def test_undertraining_reproduces_the_defect(self) -> None:
        """**证明这道闸有用**：把轮数压回 300，缺陷立刻重现。

        这不是回归测试而是**因果测试**——它锁住的是「轮数不足 → AU15 翻号」这条因果链，
        所以后人若想把训练预算调小，会在这里看到代价，而不是在运行时表情上看到。
        用合成路径之外的**真实数据**跑，但只训 300 步、且不写任何权重文件。
        """
        torch = pytest.importorskip("torch")
        from torch import nn

        from src.agents.datasets.facs_csv import load_facs_csv_ext
        from src.agents.models.facs_decoder import FACS_KEYS_EXT, FacsDecoder

        x, y = load_facs_csv_ext(FACS_EXT_CSV)
        au15 = FACS_KEYS_EXT.index("AU15")
        wrong = 0
        seeds = range(6)
        for seed in seeds:
            torch.manual_seed(seed)
            model = FacsDecoder(hidden=16, num_layers=1, extended=True)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()
            model.train()
            for _ in range(300):
                opt.zero_grad()
                loss_fn(model(x), y).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                out = model(torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
            if float(out[1, au15] - out[0, au15]) > 0:
                wrong += 1
        assert wrong > 0, (
            f"300 步下 {len(list(seeds))} 个 seed 竟全部方向正确——与实测（约 6/10 错向）不符。"
            "若这是训练配方的真实改进（如输出层 bias 初始化到 logit(μ)），"
            "请更新本测试与 notes/2026-07-28-au15-direction-root-cause.md；"
            "若不是，说明这条因果链的复现条件变了，需要重新查。"
        )
