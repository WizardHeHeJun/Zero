"""P1-4 评估口径：裸 MSE 无意义，必须连同常数基线与组内下界一起报。

一个 MSE 0.031 是好是坏，取决于它落在 `[组内下界, 常数基线]` 窗口的哪个位置——FACS 的窗口是
`[0.0259, 0.0361]`（只走完可学空间一半），换个通道同样的 0.031 可能已经触顶。这个文件锁住
四条性质：常数基线的定义、组内下界确实是下界、行加权与组等权会分家、以及 `columns` 裁剪。
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts._train_common import evaluate_with_baselines  # noqa: E402


def _make(groups: list[str], values: list[list[float]]) -> torch.Tensor:
    assert len(groups) == len(values)
    return torch.tensor(values, dtype=torch.float32)


class TestBaselines:
    def test_constant_predictor_scores_zero_skill(self) -> None:
        """定义自洽：预测恒等于训练集均值时，技能分必须正好是 0。"""
        y = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        groups = ["a", "a", "b", "b", "c", "c"]
        mean = y.mean(dim=0, keepdim=True)
        m = evaluate_with_baselines(y, mean.expand_as(y), groups, train_mean=mean)
        assert m["skill_score"] == pytest.approx(0.0, abs=1e-6)
        assert m["mse"] == pytest.approx(m["mse_constant"], rel=1e-6)

    def test_perfect_predictor_scores_one(self) -> None:
        y = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        groups = ["a", "a", "b", "b"]
        m = evaluate_with_baselines(y, y.clone(), groups, train_mean=y.mean(dim=0, keepdim=True))
        assert m["skill_score"] == pytest.approx(1.0)
        assert m["mse"] == pytest.approx(0.0, abs=1e-9)

    def test_worse_than_constant_gives_negative_skill(self) -> None:
        """打不赢常数基线要给出负分，而不是悄悄显示一个「还行」的小 MSE。"""
        y = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        groups = ["a", "a", "b", "b"]
        bad = torch.full_like(y, 99.0)
        m = evaluate_with_baselines(y, bad, groups, train_mean=y.mean(dim=0, keepdim=True))
        assert m["skill_score"] < 0


class TestWithinGroupFloor:
    def test_floor_is_group_variance(self) -> None:
        """组内下界 = 用组内均值预测的误差，等于组内方差（有偏）。"""
        # a 组 {0,2} 均值 1、组内 MSE 1.0；b 组 {10,10} 均值 10、组内 MSE 0
        y = torch.tensor([[0.0], [2.0], [10.0], [10.0]])
        groups = ["a", "a", "b", "b"]
        m = evaluate_with_baselines(y, y.clone(), groups, train_mean=y.mean(dim=0, keepdim=True))
        assert m["within_group_floor"] == pytest.approx((1.0 * 2 + 0.0 * 2) / 4)

    def test_no_predictor_beats_the_floor(self) -> None:
        """下界名副其实：同组样本共享输入，任何 (v,a) 回归器都下不去组内方差。

        用组内均值作预测（这就是 (v,a)→y 回归的最优解），其 MSE 应当正好等于下界。
        """
        y = torch.tensor([[0.0], [2.0], [4.0], [10.0], [12.0], [20.0]])
        groups = ["a", "a", "a", "b", "b", "c"]
        best = y.clone()
        for g in {"a", "b", "c"}:
            idx = [i for i, gg in enumerate(groups) if gg == g]
            best[idx] = y[idx].mean(dim=0, keepdim=True).expand_as(y[idx])
        m = evaluate_with_baselines(y, best, groups, train_mean=y.mean(dim=0, keepdim=True))
        assert m["mse"] == pytest.approx(m["within_group_floor"], rel=1e-6)
        assert m["learnable_excess"] == pytest.approx(0.0, abs=1e-6)
        assert m["headroom_used"] == pytest.approx(1.0, rel=1e-6)


class TestClassBalancedDiverges:
    def test_row_weighted_and_class_balanced_disagree_on_unbalanced_groups(self) -> None:
        """大组落在哪侧会让行加权 MSE 漂移——这正是必须同时报组等权的理由。

        构造：大组（100 行）预测得准、小组（2 行）预测得差。行加权被大组主导显得很好，
        组等权则如实暴露小组被牺牲了。
        """
        big = [[0.0]] * 100
        small = [[10.0]] * 2
        y = torch.tensor(big + small, dtype=torch.float32)
        groups = ["big"] * 100 + ["small"] * 2
        pred = torch.tensor(big + [[0.0]] * 2, dtype=torch.float32)  # 小组预测全错

        m = evaluate_with_baselines(y, pred, groups, train_mean=y.mean(dim=0, keepdim=True))
        assert m["mse"] < m["mse_class_balanced"], "行加权被大组稀释，必然低于组等权"
        assert m["mse_class_balanced"] == pytest.approx(50.0), "两组 MSE(0, 100) 的等权平均"


class TestColumnSubset:
    def test_columns_restrict_evaluation(self) -> None:
        """FACS 用它剔掉运行时不消费的 4 维——把从不使用的维算进指标是自欺。"""
        y = torch.tensor([[0.0, 100.0], [1.0, 200.0], [2.0, 300.0], [3.0, 400.0]])
        pred = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        groups = ["a", "a", "b", "b"]
        mean = y.mean(dim=0, keepdim=True)

        full = evaluate_with_baselines(y, pred, groups, train_mean=mean)
        col0 = evaluate_with_baselines(y, pred, groups, train_mean=mean, columns=[0])

        assert col0["mse"] == pytest.approx(0.0, abs=1e-9), "第 0 维预测完美"
        assert full["mse"] > 1000, "算上第 1 维就被它的巨大误差主导"
        assert col0["skill_score"] == pytest.approx(1.0)

    def test_n_groups_reported(self) -> None:
        y = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        groups = ["a", "a", "b", "b", "c", "c"]
        m = evaluate_with_baselines(y, y.clone(), groups, train_mean=y.mean(dim=0, keepdim=True))
        assert m["n_groups"] == 3


class TestFacsScoresOnlyConsumedDims:
    """FACS 判据只算运行时真正消费的 9 维——把从不被读取的维算进指标是自欺。

    默认 `residual_alpha=1.0` 时 `composite.py` 对 coping 判别 AU 执行
    `base*(1-α) + placeholder*α`，即真模型这 4 维的输出被解析占位**整个替换**。
    """

    def test_consumed_columns_exclude_coping_driven_aus(self) -> None:
        from scripts.train_facs import _RUNTIME_CONSUMED_COLUMNS
        from src.agents.models.composite import _COPING_DRIVEN_AUS
        from src.agents.models.facs_decoder import FACS_KEYS_EXT

        consumed = [FACS_KEYS_EXT[i] for i in _RUNTIME_CONSUMED_COLUMNS]
        assert len(consumed) == 9, f"13 键减 4 个 coping 判别 AU 应余 9，实为 {len(consumed)}"
        assert set(consumed).isdisjoint(_COPING_DRIVEN_AUS)
        assert set(consumed) | set(_COPING_DRIVEN_AUS) == set(FACS_KEYS_EXT), "两者须恰好覆盖全键集"

    def test_drift_guard_between_two_sources(self) -> None:
        """漂移守卫：列索引从 `FACS_KEYS_EXT` 与 `_COPING_DRIVEN_AUS` 现算，不是抄下来的常量。

        任一侧增删键，这里都会跟着变；如果哪天有人把它改成硬编码列表，本测试会失败。
        """
        from scripts.train_facs import _RUNTIME_CONSUMED_COLUMNS
        from src.agents.models.composite import _COPING_DRIVEN_AUS
        from src.agents.models.facs_decoder import FACS_KEYS_EXT

        recomputed = [i for i, k in enumerate(FACS_KEYS_EXT) if k not in _COPING_DRIVEN_AUS]
        assert _RUNTIME_CONSUMED_COLUMNS == recomputed

    def test_metrics_land_in_sidecar(self, tmp_path: Path) -> None:
        """端到端：开留出后 sidecar 的 `val` 段必须带齐技能分口径，而不只是一个裸 MSE。"""
        import csv
        import json

        from scripts._train_common import provenance_path
        from scripts.train_facs import train

        csv_path = tmp_path / "labels.csv"
        fields = ["valence", "arousal", "AU04", "AU06", "AU12", "AU15", "intensity"]
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for a in range(8):
                v = round(-1.0 + 2.0 * a / 7, 3)
                for r in range(3):
                    w.writerow(
                        {
                            "valence": v,
                            "arousal": round(0.5 * v, 3),
                            "AU04": round(max(0.0, -v) + 0.01 * r, 3),
                            "AU06": round(max(0.0, v), 3),
                            "AU12": round(max(0.0, v) * 0.8, 3),
                            "AU15": round(max(0.0, -v) * 0.7, 3),
                            "intensity": abs(v),
                        }
                    )

        out = tmp_path / "facs.pt"
        train(str(csv_path), epochs=50, stop="fixed", val_split="class", out=str(out))
        val = json.loads(provenance_path(out).read_text(encoding="utf-8"))["metrics"]["val"]

        for key in (
            "skill_score",
            "mse",
            "mse_class_balanced",
            "mse_constant",
            "within_group_floor",
            "learnable_excess",
            "headroom_used",
        ):
            assert key in val, f"sidecar 缺少 {key}——裸 MSE 单独出现正是本轮要消灭的"
        assert val["within_group_floor"] <= val["mse"] + 1e-9, "下界不该高于实测 MSE"
        assert val["scored_dims"] == "全部输出维", "非 --ext 模式不做 9 维裁剪"
