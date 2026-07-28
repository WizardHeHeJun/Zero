"""P1-5 轮数/切分扫描脚本：切分枚举、快照评估、报告渲染。

这个脚本产出的是**用来下结论的证据表**，所以它自己的正确性比一般工具更要紧——枚举漏折、
配对差配错、快照取错轮数，都会让读表的人得出反的结论而毫无察觉。
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from scripts.sweep_epochs import (
    enumerate_splits,
    group_index_table,
    render_report,
)


def _groups(spec: dict[str, int]) -> list[str]:
    out: list[str] = []
    for name, count in spec.items():
        out.extend([name] * count)
    return out


class TestEnumerateSplits:
    def test_exhaustive_covers_all_combinations(self) -> None:
        """穷举就该是 C(G,k) 折，一折不多一折不少——少一折就不叫穷举。"""
        groups = _groups({f"g{i}": 3 for i in range(8)})
        splits = enumerate_splits(groups, mode="exhaustive", n_val_groups=2, max_splits=100, seed=0)
        assert len(splits) == len(list(itertools.combinations(range(8), 2))) == 28

    def test_exhaustive_splits_are_distinct(self) -> None:
        groups = _groups({f"g{i}": 3 for i in range(6)})
        splits = enumerate_splits(groups, mode="exhaustive", n_val_groups=2, max_splits=99, seed=0)
        labels = [label for _, _, label in splits]
        assert len(set(labels)) == len(labels) == 15

    def test_every_split_partitions_the_rows(self) -> None:
        groups = _groups({f"g{i}": 4 for i in range(5)})
        splits = enumerate_splits(groups, mode="exhaustive", n_val_groups=2, max_splits=99, seed=0)
        for tr, va, _ in splits:
            assert sorted(tr + va) == list(range(len(groups)))
            assert set(tr) & set(va) == set()
            # 整组进一侧
            assert {groups[i] for i in tr} & {groups[i] for i in va} == set()

    def test_random_mode_deduplicates(self) -> None:
        """随机模式必须去重——重复折会让「独立折数」虚高，直接夸大证据强度。"""
        groups = _groups({f"g{i}": 3 for i in range(6)})
        splits = enumerate_splits(groups, mode="random", n_val_groups=2, max_splits=10, seed=1)
        labels = [label for _, _, label in splits]
        assert len(set(labels)) == len(labels) == 10

    def test_oversized_exhaustive_is_sampled_not_truncated(self) -> None:
        """超上限时**均匀抽样**而非截断前 N 个——截断会系统性偏向字典序靠前的组。

        断言必须能区分「抽样」与「截断」本身：直接检查结果是否触及了
        `combos[:max_splits]` 之外的组合。早先版本断言的是「覆盖组数 > 4」，而
        `combinations` 自然序的前 8 个组合本就覆盖 9 个组，截断版照样通过——那条防线是摆设。
        """
        n_groups, max_splits = 10, 8
        groups = _groups({f"g{i:02d}": 2 for i in range(n_groups)})
        splits = enumerate_splits(
            groups, mode="exhaustive", n_val_groups=2, max_splits=max_splits, seed=3
        )
        assert len(splits) == max_splits

        picked = {tuple(int(tok[1:]) for tok in label.split("+")) for _, _, label in splits}
        truncated = set(list(itertools.combinations(range(n_groups), 2))[:max_splits])
        assert picked - truncated, (
            "抽样结果必须触及「前 N 个组合」之外；若完全落在其中，与截断没有区别"
        )

    def test_labels_use_short_indices(self) -> None:
        """标签用组序号：真实分组键是 (v,a) 浮点元组，直接展开会让报告无法阅读。"""
        groups = [(0.1, 0.2)] * 3 + [(0.3, 0.4)] * 3 + [(0.5, 0.6)] * 3 + [(0.7, 0.8)] * 3
        splits = enumerate_splits(groups, mode="exhaustive", n_val_groups=2, max_splits=99, seed=0)
        for _, _, label in splits:
            assert all(tok.startswith("g") and tok[1:].isdigit() for tok in label.split("+"))

    def test_too_few_groups_fails_fast(self) -> None:
        with pytest.raises(ValueError, match="太少"):
            enumerate_splits(
                _groups({"a": 5}), mode="exhaustive", n_val_groups=1, max_splits=9, seed=0
            )


class TestGroupIndexTable:
    def test_lists_mapping_for_small_group_count(self) -> None:
        lines = group_index_table(_groups({"a": 2, "b": 3}), limit=20)
        text = "\n".join(lines)
        assert "g0" in text and "g1" in text
        assert "| 3 |" in text, "应带每组行数"

    def test_omits_for_large_group_count(self) -> None:
        lines = group_index_table(_groups({f"g{i}": 2 for i in range(30)}), limit=20)
        text = "\n".join(lines)
        assert "对照表" in text and "| g0 |" not in text


class TestRenderReport:
    @staticmethod
    def _row(split: int, seed: int, epochs: int, mse: float) -> dict[str, Any]:
        return {
            "split": split,
            "label": f"g{split}+g{split + 1}",
            "init_seed": seed,
            "epochs": epochs,
            "metrics": {
                "mse": mse,
                "mse_class_balanced": mse,
                "mse_constant": 0.05,
                "within_group_floor": 0.01,
                "skill_score": 1.0 - mse / 0.05,
                "headroom_used": (0.05 - mse) / 0.04,
                "learnable_excess": mse - 0.01,
                "n_groups": 2,
                "train_loss": mse,
            },
        }

    def test_paired_diff_uses_same_split_and_seed(self) -> None:
        """配对差必须在**同切分同种子**内做——跨切分直接比均值会被切分难度淹没。

        构造：切分 0 上多训有效（0.04→0.02），切分 1 上多训反而更差（0.02→0.03）。
        配对差应为 (+(-0.02) + (+0.01))/2 = -0.005，胜出折数 1/2。
        """
        rows = [
            self._row(0, 0, 100, 0.04),
            self._row(0, 0, 500, 0.02),
            self._row(1, 0, 100, 0.02),
            self._row(1, 0, 500, 0.03),
        ]
        report = render_report(rows, channel="t", epochs_grid=[100, 500], meta={})
        assert "-0.005000" in report
        assert "| 1/2 |" in report

    def test_summary_has_one_row_per_epoch(self) -> None:
        rows = [self._row(s, 0, e, 0.03) for s in range(3) for e in (100, 500)]
        report = render_report(rows, channel="t", epochs_grid=[100, 500], meta={"折数": 3})
        assert report.count("| 100 |") >= 1 and report.count("| 500 |") >= 1
        assert "折数" in report

    def test_includes_group_table_when_groups_given(self) -> None:
        rows = [self._row(0, 0, 100, 0.03)]
        report = render_report(
            rows, channel="t", epochs_grid=[100], meta={}, groups=_groups({"a": 2, "b": 2})
        )
        assert "组序号对照" in report


class TestSnapshotSemantics:
    def test_snapshots_hit_exact_grid_points(self) -> None:
        """快照必须落在网格点上，且同一次训练产出全部网格点——不为每个轮数重训。"""
        torch = pytest.importorskip("torch")
        from torch import nn

        from scripts.sweep_epochs import fit_and_snapshot

        x = torch.randn(24, 2)
        y = torch.rand(24, 3)
        groups = ["a"] * 8 + ["b"] * 8 + ["c"] * 8
        snaps = fit_and_snapshot(
            lambda: nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 3)),
            x[:16],
            y[:16],
            x[16:],
            y[16:],
            groups[16:],
            epochs_grid=[5, 20],
            init_seed=0,
            lr=1e-2,
            columns=None,
        )
        assert set(snaps) == {5, 20}
        assert all("skill_score" in m and "train_loss" in m for m in snaps.values())

    def test_same_init_seed_reproduces(self) -> None:
        torch = pytest.importorskip("torch")
        from torch import nn

        from scripts.sweep_epochs import fit_and_snapshot

        torch.manual_seed(99)
        x, y = torch.randn(24, 2), torch.rand(24, 3)
        groups = ["a"] * 8 + ["b"] * 8 + ["c"] * 8

        def run() -> dict[int, dict[str, float]]:
            return fit_and_snapshot(
                lambda: nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 3)),
                x[:16],
                y[:16],
                x[16:],
                y[16:],
                groups[16:],
                epochs_grid=[10],
                init_seed=7,
                lr=1e-2,
                columns=None,
            )

        assert run()[10]["mse"] == pytest.approx(run()[10]["mse"])
