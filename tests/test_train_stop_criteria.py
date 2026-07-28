"""P1-1 停机判据：用「训练 loss 进入平台」取代 `epochs=300` 这个魔数。

**为什么要换**：固定轮数是拿单次实验凑出来的，换个学习率就静默失效——实测
`lr=3e-3 @300 步` ≈ `lr=1e-3 @1000 步`，真正被调的是 `lr × steps` 的乘积。相对下降判据
对 lr 变化免疫：lr 大就早停、lr 小就多跑，停在同一个平台上。

`stop="fixed"` 保留供既有调用方逐字零回归——4 个 smoke 测试 + `run_pipeline.py` 都显式传它。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from scripts._train_common import (
    DEFAULT_MAX_EPOCHS,
    PlateauStopper,
    add_stop_arguments,
    provenance_path,
    resolve_cli_epochs,
    resolve_epoch_budget,
)


class TestPlateauStopper:
    def test_only_checks_on_window_boundary(self) -> None:
        s = PlateauStopper(window=10)
        for step in range(1, 10):
            assert s.should_stop(step, 1.0) is False, f"step {step} 不是窗口边界，不该判停"

    def test_first_window_never_stops(self) -> None:
        """第一个窗口只记录参照——没有前一个窗口就无从比较相对下降。"""
        s = PlateauStopper(window=10)
        assert s.should_stop(10, 0.5) is False

    def test_stops_when_relative_drop_below_tolerance(self) -> None:
        s = PlateauStopper(window=10, rel_tol=1e-4)
        s.should_stop(10, 1.0)  # 建立参照
        # 下降 5e-5 相对量，低于 1e-4 → 平台
        assert s.should_stop(20, 1.0 - 5e-5) is True

    def test_keeps_going_while_still_learning(self) -> None:
        s = PlateauStopper(window=10, rel_tol=1e-4)
        s.should_stop(10, 1.0)
        assert s.should_stop(20, 0.9) is False, "还在明显下降，不该停"

    def test_rising_loss_counts_as_plateau(self) -> None:
        """loss 反弹时相对下降为负、必然低于阈值——继续跑没有意义，停。"""
        s = PlateauStopper(window=10, rel_tol=1e-4)
        s.should_stop(10, 0.5)
        assert s.should_stop(20, 0.7) is True

    def test_reference_advances_each_window(self) -> None:
        """参照点是「上一个窗口末尾」，不是起始 loss——否则长跑后永远判不出平台。"""
        s = PlateauStopper(window=10, rel_tol=1e-4)
        s.should_stop(10, 1.0)
        assert s.should_stop(20, 0.5) is False  # 相对上窗降一半
        assert s.should_stop(30, 0.5) is True  # 相对上窗（0.5）没动 → 平台

    @pytest.mark.parametrize(("window", "rel_tol"), [(0, 1e-4), (-1, 1e-4), (10, -0.1)])
    def test_rejects_invalid_config(self, window: int, rel_tol: float) -> None:
        with pytest.raises(ValueError):
            PlateauStopper(window=window, rel_tol=rel_tol)


class TestResolveEpochBudget:
    def test_fixed_keeps_old_behaviour(self) -> None:
        budget, stopper = resolve_epoch_budget(stop="fixed", epochs=300, max_epochs=5000)
        assert budget == 300
        assert stopper is None, "fixed 模式不得带平台检测，否则不是逐字旧行为"

    def test_plateau_uses_max_epochs_as_ceiling(self) -> None:
        budget, stopper = resolve_epoch_budget(stop="plateau", epochs=300, max_epochs=5000)
        assert budget == 5000, "plateau 模式下 epochs 不参与，上限由 max_epochs 定"
        assert isinstance(stopper, PlateauStopper)

    def test_unknown_mode_fails_fast(self) -> None:
        """打错的 stop 值必须报错，不能静默当成某个默认值跑掉。"""
        with pytest.raises(ValueError, match="plateau"):
            resolve_epoch_budget(stop="platau", epochs=300, max_epochs=5000)


class TestCliWiring:
    @staticmethod
    def _parse(argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        add_stop_arguments(parser)
        return parser.parse_args(argv)

    def test_defaults_to_plateau(self) -> None:
        args = self._parse([])
        assert args.stop == "plateau"
        assert args.max_epochs == DEFAULT_MAX_EPOCHS
        assert args.epochs is None, "--epochs 须默认 None，才能区分「用户传了」与「没传」"
        assert resolve_cli_epochs(args) == 300

    def test_explicit_epochs_under_plateau_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """plateau 下 --epochs 不生效；静默忽略会让人以为自己控制住了轮数。"""
        args = self._parse(["--epochs", "1234"])
        with caplog.at_level("WARNING"):
            assert resolve_cli_epochs(args) == 1234
        assert "不生效" in caplog.text
        assert "--stop fixed" in caplog.text

    def test_explicit_epochs_under_fixed_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        args = self._parse(["--stop", "fixed", "--epochs", "1234"])
        with caplog.at_level("WARNING"):
            assert resolve_cli_epochs(args) == 1234
        assert caplog.text == ""


class TestPlateauInRealTraining:
    """判据接进真训练循环：上限生效、停机事实落进 sidecar、break 确实接通。

    ⚠ **实测事实**（`notes/2026-07-27-training-pipeline-remediation-plan.md` 已据此订正）：
    默认 `rel_tol=1e-4` 在本仓四个通道里**只有生理通道会触发**（真实 WESAD 上约 2100 步）；
    FACS / 韵律 / expression 到 5000 步仍有 3.8e-4 / 1.6e-4 / 1.8e-2 的相对下降，会跑满上限。
    这些小 MLP 在小数据上是**幂律式持续缓降**，并没有明显的平台拐点。所以这套判据的实际作用
    是「防欠收敛的上限保障」，不是「精确找到最优停机点」——测试按这个真实行为写，不假装
    每个通道都会早停。
    """

    def test_plateau_ceiling_and_fields(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        import json

        from scripts.train_expression import train

        out = tmp_path / "expr.pt"
        train(n=64, out=str(out), stop="plateau", max_epochs=300)

        rec: dict[str, Any] = json.loads(provenance_path(out).read_text(encoding="utf-8"))
        assert rec["training"]["stop"] == "plateau"
        assert rec["training"]["epochs_requested"] == 300, "plateau 下上限来自 max_epochs"
        assert rec["training"]["epochs_ran"] <= 300

    def test_plateau_break_is_actually_wired(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """换上必停的替身 stopper：验证训练循环真的会因判据而 break。

        判据本身的算术由 `TestPlateauStopper` 覆盖；这里单测「接线通没通」——真实数据下
        默认阈值极少触发，光靠真训练测不到这条 break 路径，回归时会静默失守。
        """
        pytest.importorskip("torch")
        import json

        from scripts import _train_common
        from scripts.train_expression import train

        class _AlwaysStop:
            window = 100
            rel_tol = 1.0

            def should_stop(self, step: int, loss: float) -> bool:
                return step >= 100

        monkeypatch.setattr(_train_common, "PlateauStopper", lambda **_kw: _AlwaysStop())

        out = tmp_path / "expr_stop.pt"
        train(n=64, out=str(out), stop="plateau", max_epochs=5000)

        rec = json.loads(provenance_path(out).read_text(encoding="utf-8"))
        assert rec["training"]["epochs_ran"] == 100, "应在替身判停的第 100 步 break"
        assert rec["training"]["stopped_early"] is True

    def test_fixed_runs_exactly_the_requested_epochs(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        import json

        from scripts.train_expression import train

        out = tmp_path / "expr_fixed.pt"
        train(epochs=37, stop="fixed", n=64, out=str(out))

        rec = json.loads(provenance_path(out).read_text(encoding="utf-8"))
        assert rec["training"]["stop"] == "fixed"
        assert rec["training"]["epochs_ran"] == 37
        assert rec["training"]["stopped_early"] is False
