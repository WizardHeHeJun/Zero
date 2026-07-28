"""P1-2 分组留出切分：整组进 train 或 val，禁止按行随机切。

**为什么这件事非做不可**：本仓这几个数据集里同一个 `(v,a)` 锚点或同一个被试对应几十上百行。
按行随机切会让同组样本同时出现在两边——val 退化成「组均值下界」，模型只要记住组均值就能拿
高分，于是给出「永远训更久更好」的错误信号。这正是整个训练管线整改要消灭的那类假信号。
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts._train_common import (
    MIN_GROUP_SIZE,
    MIN_HOLDOUT_GROUPS,
    group_holdout,
)


def _groups(spec: dict[str, int]) -> list[str]:
    """{组名: 行数} → 展开成逐行的组标签。"""
    out: list[str] = []
    for name, count in spec.items():
        out.extend([name] * count)
    return out


class TestNoLeakAcrossSplit:
    def test_every_group_lands_entirely_on_one_side(self) -> None:
        groups = _groups({"a": 5, "b": 3, "c": 7, "d": 2, "e": 4})
        train_idx, val_idx = group_holdout(groups, val_seed=0)

        train_groups = {groups[i] for i in train_idx}
        val_groups = {groups[i] for i in val_idx}
        assert train_groups & val_groups == set(), "同一组不得同时出现在两边——这就是泄漏"
        assert train_groups | val_groups == set(groups), "不得有组被整个丢掉"

    def test_partition_is_complete_and_disjoint(self) -> None:
        groups = _groups({"a": 5, "b": 3, "c": 7, "d": 2, "e": 4})
        train_idx, val_idx = group_holdout(groups, val_seed=3)
        assert sorted(train_idx + val_idx) == list(range(len(groups)))
        assert set(train_idx) & set(val_idx) == set()

    def test_both_sides_non_empty(self) -> None:
        """哪怕组数刚好到下限，也必须两边都有东西。"""
        groups = _groups({f"g{i}": 2 for i in range(MIN_HOLDOUT_GROUPS)})
        train_idx, val_idx = group_holdout(groups, val_seed=0)
        assert train_idx and val_idx


class TestDeterminism:
    def test_same_seed_same_split(self) -> None:
        groups = _groups({f"g{i}": 3 for i in range(10)})
        assert group_holdout(groups, val_seed=7) == group_holdout(groups, val_seed=7)

    def test_different_seed_can_differ(self) -> None:
        groups = _groups({f"g{i}": 3 for i in range(20)})
        splits = {tuple(group_holdout(groups, val_seed=s)[1]) for s in range(6)}
        assert len(splits) > 1, "换 seed 应能得到不同切分，否则 --val-seed 形同虚设"

    def test_group_order_does_not_matter(self) -> None:
        """先定序再洗牌：同一批组、同一 seed，喂入顺序不同也得到同一组划分。"""
        a = _groups({"x": 2, "y": 2, "z": 2, "w": 2})
        b = _groups({"w": 2, "z": 2, "y": 2, "x": 2})
        val_a = {a[i] for i in group_holdout(a, val_seed=1)[1]}
        val_b = {b[i] for i in group_holdout(b, val_seed=1)[1]}
        assert val_a == val_b

    def test_does_not_disturb_torch_rng(self) -> None:
        """切分用独立 RNG——否则开不开留出集会连带改变模型初始化，两次实验不可比。"""
        torch = pytest.importorskip("torch")

        torch.manual_seed(123)
        expected = torch.randn(4)

        torch.manual_seed(123)
        group_holdout(_groups({f"g{i}": 3 for i in range(8)}), val_seed=99)
        actual = torch.randn(4)

        assert torch.equal(expected, actual), "group_holdout 不得消耗 torch 全局 RNG"


class TestFailFast:
    def test_too_few_groups(self) -> None:
        groups = _groups({f"g{i}": 5 for i in range(MIN_HOLDOUT_GROUPS - 1)})
        with pytest.raises(ValueError, match="至少"):
            group_holdout(groups)

    def test_undersized_group(self) -> None:
        groups = _groups({"a": 5, "b": 5, "c": 5, "lonely": MIN_GROUP_SIZE - 1})
        with pytest.raises(ValueError, match="lonely"):
            group_holdout(groups)

    @pytest.mark.parametrize("frac", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_val_frac(self, frac: float) -> None:
        with pytest.raises(ValueError, match="val_frac"):
            group_holdout(_groups({f"g{i}": 3 for i in range(8)}), val_frac=frac)

    def test_error_names_the_offending_groups(self) -> None:
        """报错要能直接指出是哪几组出问题，否则用户面对上千组无从下手。"""
        groups = _groups({"ok1": 5, "ok2": 5, "ok3": 5, "bad_a": 1, "bad_b": 1})
        with pytest.raises(ValueError) as exc:
            group_holdout(groups)
        assert "bad_a" in str(exc.value) and "bad_b" in str(exc.value)


class TestWiredIntoTrainingScripts:
    """接线：`--val-split` 开了要真切、真评、真落账；组数不够要 fail-fast 而非静默跑一个假 val。"""

    @staticmethod
    def _write_facs_csv(path: Path, *, n_anchors: int, rows_per_anchor: int) -> None:
        fields = ["valence", "arousal", "AU04", "AU06", "AU12", "AU15", "intensity"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for a in range(n_anchors):
                v = round(-1.0 + 2.0 * a / max(n_anchors - 1, 1), 3)
                for r in range(rows_per_anchor):
                    jitter = 0.01 * r  # 同锚点内轻微抖动，模拟真实数据的组内方差
                    writer.writerow(
                        {
                            "valence": v,
                            "arousal": round(0.5 * v, 3),
                            "AU04": round(max(0.0, -v) + jitter, 3),
                            "AU06": round(max(0.0, v), 3),
                            "AU12": round(max(0.0, v) * 0.8, 3),
                            "AU15": round(max(0.0, -v) * 0.7, 3),
                            "intensity": abs(v),
                        }
                    )

    def test_holdout_recorded_in_sidecar(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        import json

        from scripts._train_common import provenance_path
        from scripts.train_facs import train

        csv_path = tmp_path / "labels.csv"
        self._write_facs_csv(csv_path, n_anchors=8, rows_per_anchor=3)
        out = tmp_path / "facs.pt"
        train(str(csv_path), epochs=20, stop="fixed", val_split="class", out=str(out))

        rec = json.loads(provenance_path(out).read_text(encoding="utf-8"))
        val = rec["metrics"]["val"]
        assert val is not None, "开了 --val-split 就必须报 val 指标"
        assert val["split"] == "group-holdout:class"
        assert val["n_val_groups"] >= 1 and val["n_train_groups"] >= 1
        assert val["n_train_groups"] + val["n_val_groups"] == 8
        assert val["mse"] >= 0.0
        assert rec["training"]["val_split"] == "class"
        # 训练样本数应当只算 train 侧——把 val 也算进去就等于没切
        assert rec["training"]["n_samples"] == val["n_train_samples"]
        assert val["n_train_samples"] + val["n_samples"] == 8 * 3

    def test_default_none_is_zero_regression(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        import json

        from scripts._train_common import provenance_path
        from scripts.train_facs import train

        csv_path = tmp_path / "labels.csv"
        self._write_facs_csv(csv_path, n_anchors=8, rows_per_anchor=3)
        out = tmp_path / "facs.pt"
        train(str(csv_path), epochs=20, stop="fixed", out=str(out))

        rec = json.loads(provenance_path(out).read_text(encoding="utf-8"))
        assert rec["training"]["val_split"] == "none"
        assert rec["metrics"]["val"] is None
        assert rec["training"]["n_samples"] == 8 * 3, "不切时全部样本都该参与训练"

    def test_too_few_anchors_fails_fast(self, tmp_path: Path) -> None:
        """锚点不够时报错，而不是切出一个没有统计意义的 val 继续跑。"""
        pytest.importorskip("torch")
        from scripts.train_facs import train

        csv_path = tmp_path / "labels.csv"
        self._write_facs_csv(csv_path, n_anchors=3, rows_per_anchor=4)
        with pytest.raises(ValueError, match="至少"):
            train(
                str(csv_path), epochs=5, stop="fixed", val_split="class", out=str(tmp_path / "x.pt")
            )

    def test_val_seed_changes_split_but_not_weights_path(self, tmp_path: Path) -> None:
        """`--val-seed` 只影响切分；模型初始化仍由 `--seed` 决定，两者互不干扰。"""
        pytest.importorskip("torch")
        import json

        from scripts._train_common import provenance_path
        from scripts.train_facs import train

        csv_path = tmp_path / "labels.csv"
        self._write_facs_csv(csv_path, n_anchors=10, rows_per_anchor=3)
        recs = []
        for vs in (0, 3):
            out = tmp_path / f"facs_{vs}.pt"
            train(
                str(csv_path),
                epochs=20,
                stop="fixed",
                val_split="class",
                val_seed=vs,
                seed=5,
                out=str(out),
            )
            recs.append(json.loads(provenance_path(out).read_text(encoding="utf-8")))
        assert recs[0]["metrics"]["val"]["val_seed"] == 0
        assert recs[1]["metrics"]["val"]["val_seed"] == 3
        assert all(r["training"]["seed"] == 5 for r in recs), "模型种子不受 val_seed 影响"


class TestWhyNotNaiveSplit:
    def test_group_split_holds_out_unseen_groups(self) -> None:
        """对照说明：分组切分下，val 里的组在 train 中**一次都没出现过**。

        这正是它与按行随机切的本质区别——后者会让 val 的每个组在 train 里都有兄弟样本。
        """
        groups = _groups({f"anchor{i}": 30 for i in range(10)})
        train_idx, val_idx = group_holdout(groups, val_seed=5)
        seen_in_train = {groups[i] for i in train_idx}
        for i in val_idx:
            assert groups[i] not in seen_in_train
