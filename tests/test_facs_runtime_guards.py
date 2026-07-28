"""FACS 解码器**运行时守卫**：与训练轮数解耦的行为约束。

**为什么与 `test_facs_au_expansion.py` 分开**：那边测的是**解析占位**（`decode_channels`，
纯 Python 映射，行为由代码写死）；这里测的是**训出来的权重**在运行时的行为。两者是
`CompositeChannelDecoder` residual 混合的两侧，坏法完全不同——占位坏在映射公式，
权重坏在拟合方向/饱和，占位的单调性测试一条也抓不住权重的问题。

**为什么这些断言与轮数解耦**：整改 P1-1 把停机判据从 `epochs=300` 魔数换成了 plateau，
P1-5 又证明轮数会显著改变模型（FACS 300→1500 技能分 +0.019→+0.189）。轮数一变，
任何写死 loss 数值的断言都会失效，而「不饱和 / AU12 随效价升 / AU04 随效价降」这类
**方向与值域性质**在任何合理轮数下都该成立——守卫要守的是这些。

三层：
  ① 合成 CSV 训练 → 方向学得对（data-independent，任何环境都跑）
  ② 真权重（`artifacts/facs_decoder_ext_v2.pt`，缺则跳过）→ 不饱和 / 角点区间 /
     **运行时消费的 9 维逐维**不退化成常数 / 强信号维方向与训练数据一致
  ③ **已知缺陷固化**：AU15 沿 valence 的方向与训练数据的相关号相反（strict xfail）

**覆盖边界（如实标注，别以为 9 维都被方向守卫罩住了）**：

| 维 | 不退化成常数 | 方向守卫 | 说明 |
| --- | --- | --- | --- |
| AU04 / AU06 / AU07 / AU12 | ✅ | ✅ 数据 \|r\| ≥ 0.5 | 逐点严格单调的只有 AU04/AU06/AU12 |
| AU17 | ✅ | ✅ 但余量极小（Δ=−0.019） | 翻号即红，属真回归 |
| AU15 | ✅ | ⚠ strict xfail | **已知缺陷**，方向反了 |
| AU05 / AU26 / intensity | ✅ | ❌ **无** | 数据里 \|r\| 仅 0.240/0.089/0.390，无方向可断 |

即 9 维全部有「不退化成常数」保护，但**方向**只守得住数据里信号确实强的 6 维。

⚠ **这个缺口的边界是实测过的**：把 AU26 改成 `0.15 − 0.05·valence`（方向反了但没塌成常数），
本文件全部测试**照常全绿**。弱信号维的方向 bug 目前确实无人看守——要补只能等数据侧变强
（换数据源/换表征，属 P3），加断言硬凑只会造出一个测不准的假守卫。
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.agents.models.facs_decoder import FACS_KEYS_EXT

REAL_EXT_WEIGHTS = Path("artifacts/facs_decoder_ext_v2.pt")

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
# 实测各维全网格极差（2026-07-28，`facs_decoder_ext_v2.pt`）：AU26 最小 0.093、AU06 最大 0.258，
# 取 0.02 对最紧的一维仍有 4.7 倍余量——够松，不会锁死重训；够紧，能抓住塌成常数。
MIN_GRID_SPAN = 0.02

# 训练数据（`data/facs/labels_ext.csv`，1634 行 / 38 锚点）里各 AU 与 valence 的**锚点级**
# 相关系数，2026-07-28 实测。只对 |r| ≥ 0.5 的维断言方向——数据里信号本就微弱的维
# （AU05 −0.240 / AU26 +0.089 / intensity +0.390）没有可断言的方向，硬断只会制造假守卫。
#
# ⚠ 逐点严格单调**只有 3 维成立**（AU04 严格降、AU06/AU12 严格升），其余 6 维在
# 420 对采样里都是升降混杂，所以这里断的是**净变化的符号**而非逐点单调。
STRONG_VALENCE_SIGNAL: dict[str, float] = {
    "AU04": -0.701,
    "AU06": +0.679,
    "AU07": +0.616,
    "AU12": +0.739,
    "AU15": -0.526,  # ⚠ 模型实际 +0.115，方向反了 → 见 TestKnownDefectAU15Direction
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

        这是比「逐点单调」更能推广的判据：逐点严格单调实测只有 AU04/AU06/AU12 三维成立，
        而「学出来的方向不该和数据相反」对每一个强信号维都该成立——AU15 正是栽在这条上
        （数据 −0.526、模型 +0.115），故在此排除、由 `TestKnownDefectAU15Direction` 专门记录。

        ⚠ AU17 满足但**余量极小**（Δ=−0.019）：它现在是绿的，重训后若翻号这条会先红，
        那说明 AU17 步了 AU15 的后尘，属真回归而非测试过敏。
        """
        model = self._load()
        for au, r in STRONG_VALENCE_SIGNAL.items():
            if au == "AU15":
                continue  # 已知缺陷，单列 xfail 记录，不在这里重复报红
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

    def test_au12_increases_and_au04_decreases_with_valence(self) -> None:
        """两条最有共识的方向：AU12（颧大肌·笑）随效价升，AU04（皱眉）随效价降。

        实测每个 arousal 档上逐点严格单调，无逆行——这两维是真权重学得最扎实的部分
        （训练数据锚点级相关 r = +0.739 / −0.701）。
        """
        model = self._load()
        for a in GRID:
            au12 = _sweep(model, "AU12", arousal=a)
            au04 = _sweep(model, "AU04", arousal=a)
            for i in range(len(au12) - 1):
                assert au12[i] <= au12[i + 1] + 1e-9, (
                    f"AU12 在 arousal={a:+.1f} 处逆行：v={GRID[i]:+.1f}→{GRID[i + 1]:+.1f}，"
                    f"{au12[i]:.6f}→{au12[i + 1]:.6f}"
                )
                assert au04[i] >= au04[i + 1] - 1e-9, (
                    f"AU04 在 arousal={a:+.1f} 处逆行：v={GRID[i]:+.1f}→{GRID[i + 1]:+.1f}，"
                    f"{au04[i]:.6f}→{au04[i + 1]:.6f}"
                )


# ─── ③ 已知缺陷固化 ──────────────────────────────────────────────────────────────


@requires_real_weights
class TestKnownDefectAU15Direction:
    """**已知缺陷**：AU15 在真权重上沿 valence 的方向与训练数据的相关号相反。

    实测（2026-07-28，整改 P2-2）：

    | 量 | AU04 | AU12 | **AU15** | AU17 |
    | --- | --- | --- | --- | --- |
    | 数据锚点级 r(AU, valence) | −0.701 | +0.739 | **−0.526** | −0.538 |
    | 模型 Δ(v: −1→+1) @ a=0 | −0.135 | +0.199 | **+0.115** | −0.019 |

    即数据里 AU15（压嘴角·悲伤 AU）与效价**负**相关，模型却学成了**正**相关，
    全网格 358/420 对逆行。AU15 **不在** `_COPING_DRIVEN_AUS` 内，composite 默认
    α=1.0 下不会被解析占位覆盖，因此这条错误方向会**真实进入运行时表情**。

    成因推测（未验证，不作结论）：AU15 是训练数据里最稀疏的通用维之一
    （mean 0.096 / sd 0.129，远小于 AU04 的 0.265/0.240），全批量 MSE 下它的梯度
    被大方差维淹没；hidden=16 单隐层的容量也可能不足以同时拟合方向相反的两组维。

    **为什么用 strict xfail 而不是放宽断言**：这是权重的真实缺陷，不是断言写错了。
    放宽会把缺陷洗白；直接断言又会让全量测试常红。strict xfail 让它现在标记为
    "预期失败"，**一旦重训修好就会 XPASS 报错**，强制回来更新这段记录。

    修法属**重训**范畴——整改计划「明确排除」条款规定权重重训与重发 Release 单列一批，
    故本轮不擅自动权重。见 `notes/2026-07-27-training-pipeline-remediation-plan.md`。
    """

    @pytest.mark.xfail(
        strict=True,
        reason="真权重 AU15 沿 valence 方向反了（数据 r=−0.526，模型 Δ=+0.115）；修法属重训批次",
    )
    def test_au15_should_decrease_with_valence(self) -> None:
        pytest.importorskip("torch")
        from src.agents.models.facs_decoder import load_facs_decoder

        model = load_facs_decoder(str(REAL_EXT_WEIGHTS), extended=True)
        delta = _net_delta(model, "AU15")
        assert delta < 0.0, f"AU15 应随 valence 降（数据锚点级 r=−0.526），实测 {delta:+.4f}"

    def test_defect_is_bounded_and_recorded(self) -> None:
        """缺陷幅度的看门测试：AU15 反向的幅度不得再扩大。

        与上面的 xfail 配对——xfail 记录「方向错了」，这条记录「错得有多大」。
        重训若让它变本加厉，这条会先失败。
        """
        pytest.importorskip("torch")
        from src.agents.models.facs_decoder import load_facs_decoder

        model = load_facs_decoder(str(REAL_EXT_WEIGHTS), extended=True)
        delta = _net_delta(model, "AU15")
        assert delta < 0.20, f"AU15 反向幅度已从 +0.115 扩大到 {delta:+.4f}，缺陷在恶化"
