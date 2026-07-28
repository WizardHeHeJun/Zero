"""P0-2 权重 provenance sidecar：旁挂 `<out>.pt.json` 记配方，`.pt` 内容格式一个字节不动。

**为什么有这一整个文件**：权重落盘后与它的配方（轮数/lr/种子/数据/commit）脱钩，是「实验
结论不可追溯」的根因。sidecar 补上这层账；而这层账**必须旁挂**——`src/agents/models/` 下 7 个
loader 全部是 `model.load_state_dict(torch.load(path, ...))`，读的是裸 `state_dict`（facs /
prosody / physiology / expression 4 个还显式传了 `weights_only=True`）。一旦把 `.pt` 改成 dict
checkpoint 就会一次性破掉全部 loader、并让已发布的 `weights-v0.1` 与新代码互不兼容。
`TestPtFormatUnchanged` 就是给这条红线上的锁。

torch 相关用例在函数内 `importorskip`；`_train_common` 自身不依赖 torch，其单元测试在无
torch 环境照跑。
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import _train_common
from scripts._train_common import SCHEMA_VERSION, provenance_path, write_provenance

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeModel:
    """替身模型：`write_provenance` 只读 `type` 与 `parameters()`，不需要真 Module。"""

    def parameters(self) -> list[Any]:
        return []


def _make_facs_csv(path: Path) -> int:
    rows = [
        {"valence": 0.8, "arousal": 0.5, "AU04": 0.0, "AU06": 0.5, "AU12": 0.8, "AU15": 0.0},
        {"valence": -0.7, "arousal": 0.3, "AU04": 0.4, "AU06": 0.0, "AU12": 0.0, "AU15": 0.7},
        {"valence": 0.0, "arousal": 0.0, "AU04": 0.0, "AU06": 0.0, "AU12": 0.0, "AU15": 0.0},
        {"valence": 0.3, "arousal": 0.9, "AU04": 0.1, "AU06": 0.3, "AU12": 0.4, "AU15": 0.1},
    ]
    fields = ["valence", "arousal", "AU04", "AU06", "AU12", "AU15", "intensity"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "intensity": abs(row["arousal"])})
    return len(rows)


def _make_emobank_csv(path: Path, *, with_split: bool) -> None:
    """最小 EmoBank 风格 CSV；`with_split=False` 用于测「官方切分不可用→降级」这条路径。"""
    rows = [
        ("1", "train", "4.5", "3.8", "what a wonderful joyful day"),
        ("2", "train", "1.5", "4.2", "terrible angry awful news"),
        ("3", "train", "3.0", "3.0", "the meeting is at noon"),
        ("4", "train", "2.0", "1.8", "feeling sad and tired quietly"),
        ("5", "dev", "4.2", "3.5", "such a lovely bright morning"),
        ("6", "dev", "1.8", "4.0", "awful dreadful terrible mess"),
    ]
    fields = ["id", "split", "V", "A", "D", "text"] if with_split else ["id", "V", "A", "D", "text"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for rid, split, v, a, text in rows:
            writer.writerow(
                [rid, split, v, a, "3.0", text] if with_split else [rid, v, a, "3.0", text]
            )


def _read_sidecar(out: Path) -> dict[str, Any]:
    return json.loads(provenance_path(out).read_text(encoding="utf-8"))


class TestPtFormatUnchanged:
    """红线：`.pt` 仍是裸 state_dict，`weights_only=True` 的 loader 全部照常工作。"""

    def test_facs_pt_loads_as_bare_state_dict(self, tmp_path: Path) -> None:
        torch = pytest.importorskip("torch")
        from scripts.train_facs import train
        from src.agents.models.facs_decoder import FacsDecoder, load_facs_decoder

        csv_path = tmp_path / "labels.csv"
        _make_facs_csv(csv_path)
        out = tmp_path / "facs.pt"
        train(str(csv_path), epochs=5, stop="fixed", out=str(out))

        raw = torch.load(out, weights_only=True)
        assert isinstance(raw, dict), "顶层必须是裸 state_dict，不能是 {'state_dict':…} 包装"
        assert set(raw) == set(FacsDecoder().state_dict()), "键集必须与模型 state_dict 逐一对应"
        assert all(isinstance(v, torch.Tensor) for v in raw.values()), "值必须全是张量"
        # 真 loader（weights_only=True）能载回 —— 这才是「没破掉 loader」的实证
        assert load_facs_decoder(str(out)) is not None

    def test_expression_pt_loads_via_real_loader(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        from scripts.train_expression import train
        from src.agents.models.expression_decoder import load_decoder

        out = tmp_path / "expr.pt"
        train(epochs=3, stop="fixed", n=64, out=str(out))
        assert load_decoder(str(out)) is not None

    def test_sidecar_is_a_separate_file(self, tmp_path: Path) -> None:
        """sidecar 与权重是两个文件；删掉 sidecar 不影响权重可载入。"""
        pytest.importorskip("torch")
        from scripts.train_expression import train
        from src.agents.models.expression_decoder import load_decoder

        out = tmp_path / "expr.pt"
        train(epochs=3, stop="fixed", n=64, out=str(out))
        sidecar = provenance_path(out)
        assert sidecar.exists() and sidecar != out
        sidecar.unlink()
        assert load_decoder(str(out)) is not None, "sidecar 是元数据，缺了不该影响权重加载"


class TestSidecarRecordsRecipe:
    def test_facs_sidecar_full_recipe(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        from scripts.train_facs import train

        csv_path = tmp_path / "labels.csv"
        n_rows = _make_facs_csv(csv_path)
        out = tmp_path / "facs.pt"
        train(str(csv_path), epochs=7, lr=2e-3, hidden=8, stop="fixed", out=str(out), seed=3)

        rec = _read_sidecar(out)
        assert rec["schema_version"] == SCHEMA_VERSION
        assert rec["script"] == "scripts/train_facs.py"
        assert rec["model"]["class"] == "FacsDecoder"
        assert rec["model"]["hidden"] == 8
        assert rec["model"]["extended"] is False
        assert rec["model"]["param_count"] > 0
        # 精确字典比较：新增/漏记字段都会在这里失败，比逐个 in 检查更能防漂移
        assert rec["training"] == {
            "seed": 3,
            "lr": 2e-3,
            "epochs_requested": 7,
            "epochs_ran": 7,
            "stopped_early": False,
            "n_samples": n_rows,
            "stop": "fixed",
            "val_split": "none",
        }
        assert rec["data"]["kind"] == "file"
        assert rec["data"]["lines"] == n_rows + 1  # 含表头
        assert rec["data"]["sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
        assert rec["env"]["python"] and rec["env"]["torch"]
        assert "commit" in rec["git"] and "dirty" in rec["git"]

    def test_artifact_sha256_pairs_sidecar_to_weights(self, tmp_path: Path) -> None:
        """sidecar 记的文件哈希必须等于磁盘上那份——用于发现「.pt 被换过、sidecar 还是旧的」。"""
        pytest.importorskip("torch")
        from scripts.train_expression import train

        out = tmp_path / "expr.pt"
        train(epochs=3, stop="fixed", n=64, out=str(out))
        rec = _read_sidecar(out)
        assert rec["artifact_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
        assert rec["artifact_bytes"] == out.stat().st_size

    def test_state_dict_hash_is_immune_to_filename(self, tmp_path: Path) -> None:
        """权重数值哈希不随输出文件名变——文件哈希会变，所以它才是真正的「同一份权重」凭证。

        `torch.save` 把输出文件名写进 zip 条目前缀，同一份 state_dict 存成两个文件名就是两个
        文件 sha256。只用 `artifact_sha256` 的话，把权重改个名就会误报「权重被替换」。
        """
        torch = pytest.importorskip("torch")
        from scripts.train_expression import train

        out_a = tmp_path / "alpha.pt"
        out_b = tmp_path / "beta.pt"
        train(epochs=3, stop="fixed", n=64, out=str(out_a), seed=11)
        train(epochs=3, stop="fixed", n=64, out=str(out_b), seed=11)

        rec_a, rec_b = _read_sidecar(out_a), _read_sidecar(out_b)
        # 同种子同配置 → 数值必然一致
        state_a = torch.load(out_a, weights_only=True)
        state_b = torch.load(out_b, weights_only=True)
        assert all(torch.equal(state_a[k], state_b[k]) for k in state_a), "同种子应产出相同张量"

        assert rec_a["model"]["state_dict_sha256"] == rec_b["model"]["state_dict_sha256"]
        assert rec_a["artifact_sha256"] != rec_b["artifact_sha256"], (
            "文件哈希受文件名影响而不同——正是它不能单独当配对凭证的原因"
        )

    def test_synthetic_source_and_no_holdout_note(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        from scripts.train_expression import train

        out = tmp_path / "expr.pt"
        train(epochs=3, stop="fixed", n=64, out=str(out), canonical_physiology=True)
        rec = _read_sidecar(out)
        assert rec["data"] == {"kind": "synthetic", "path": None}
        # canonical 与 legacy 的 idx7 语义不同、权重不可互换，必须随权重落账
        assert rec["training"]["canonical_physiology"] is True
        assert rec["metrics"]["val"] is None
        assert "泛化" in rec["metrics"]["note"], "无留出集时必须显式标注 loss 不是泛化指标"

    def test_directory_source_recorded_without_hashing(self, tmp_path: Path) -> None:
        """RAVDESS/WESAD 根是目录（可达数 GB）：只记路径/mtime，不递归、不哈希。"""
        info = _train_common._describe_source(str(tmp_path))
        assert info["kind"] == "directory"
        assert info["path"] == str(tmp_path)
        assert info["mtime_utc"]
        assert "sha256" not in info

    def test_missing_source_marked(self, tmp_path: Path) -> None:
        info = _train_common._describe_source(str(tmp_path / "nope.csv"))
        assert info["kind"] == "missing"


class TestSentenceVectorChannelNoLongerContaminated:
    """句向量通道的训练集污染修复守卫。

    `load_emobank_embeddings` 此前**根本没有 `split` 参数**，`train_text_affect_st` 与
    `train_text_affect_d` 因此一直读全量——官方 dev/test 一并训进去。已发布的
    `text_affect_regressor_st.pt` 即产自那条路径。这里用「训练样本数 == train 行数（4）而非
    全量（6）」直接钉死：dev 那两行没进训练集。
    """

    TRAIN_ROWS = 4
    DEV_ROWS = 2

    def test_st_uses_official_train_only(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("sentence_transformers")
        from scripts.train_text_affect_st import train

        csv_path = tmp_path / "emobank.csv"
        _make_emobank_csv(csv_path, with_split=True)
        out = tmp_path / "st.pt"
        train(str(csv_path), epochs=3, out=str(out))

        rec = _read_sidecar(out)
        assert rec["training"]["official_split_used"] is True
        assert rec["training"]["n_samples"] == self.TRAIN_ROWS, "dev 行不得进训练集"
        val = rec["metrics"]["val"]
        assert val["split"] == "emobank-official-dev"
        assert val["n_samples"] == self.DEV_ROWS

    def test_st_full_data_opt_out_still_available(self, tmp_path: Path) -> None:
        """`--full-data` 显式旧路径仍可用（无 split 列的数据要靠它），但会在 sidecar 里留痕。"""
        pytest.importorskip("torch")
        pytest.importorskip("sentence_transformers")
        from scripts.train_text_affect_st import train

        csv_path = tmp_path / "emobank.csv"
        _make_emobank_csv(csv_path, with_split=True)
        out = tmp_path / "st_full.pt"
        train(str(csv_path), epochs=3, out=str(out), official_split=False)

        rec = _read_sidecar(out)
        assert rec["training"]["official_split_requested"] is False
        assert rec["training"]["official_split_used"] is False
        assert rec["training"]["n_samples"] == self.TRAIN_ROWS + self.DEV_ROWS
        assert rec["metrics"]["val"] is None

    def test_d_target_recipe_and_split(self, tmp_path: Path) -> None:
        """`train_text_affect_d.py` 此前无任何测试覆盖；target 切列直接决定 output_dim。"""
        pytest.importorskip("torch")
        pytest.importorskip("sentence_transformers")
        from scripts.train_text_affect_d import train

        csv_path = tmp_path / "emobank.csv"
        _make_emobank_csv(csv_path, with_split=True)
        out = tmp_path / "d.pt"
        train(str(csv_path), target="d", epochs=3, out=str(out))

        rec = _read_sidecar(out)
        assert rec["script"] == "scripts/train_text_affect_d.py"
        # target 决定 output_dim（va→2 / d→1 / vad→3）：权重形状因它而异，落错账即载不回
        assert rec["training"]["target"] == "d"
        assert rec["model"]["output_dim"] == 1
        assert rec["data"]["kind"] == "file"
        # D 维单源 EmoBank、无跨数据集可交叉验证，更不能把 dev 训进去
        assert rec["training"]["official_split_used"] is True
        assert rec["training"]["n_samples"] == self.TRAIN_ROWS
        assert rec["metrics"]["val"]["n_samples"] == self.DEV_ROWS

    def test_missing_split_column_degrades_visibly(self, tmp_path: Path) -> None:
        """无 split 列的 CSV 仍能训（降级读全量），但 requested≠used 必须在 sidecar 里看得见。"""
        pytest.importorskip("torch")
        pytest.importorskip("sentence_transformers")
        from scripts.train_text_affect_st import train

        csv_path = tmp_path / "emobank_nosplit.csv"
        _make_emobank_csv(csv_path, with_split=False)
        out = tmp_path / "st_nosplit.pt"
        train(str(csv_path), epochs=3, out=str(out))

        rec = _read_sidecar(out)
        assert rec["training"]["official_split_requested"] is True
        assert rec["training"]["official_split_used"] is False
        assert rec["metrics"]["val"] is None

    def test_d_script_branches_match_st(self, tmp_path: Path) -> None:
        """D 脚本的 `--full-data` 与「无 split 列降级」两分支须与 ST 表现一致。

        两个脚本是同款 try/except 结构的姊妹实现，只测一个的话，将来重构让它们悄悄分叉不会被抓住。
        """
        pytest.importorskip("torch")
        pytest.importorskip("sentence_transformers")
        from scripts.train_text_affect_d import train

        with_split = tmp_path / "emobank.csv"
        _make_emobank_csv(with_split, with_split=True)
        full_out = tmp_path / "d_full.pt"
        train(str(with_split), target="d", epochs=3, out=str(full_out), official_split=False)
        rec = _read_sidecar(full_out)
        assert rec["training"]["official_split_used"] is False
        assert rec["training"]["n_samples"] == self.TRAIN_ROWS + self.DEV_ROWS
        assert rec["metrics"]["val"] is None

        no_split = tmp_path / "emobank_nosplit.csv"
        _make_emobank_csv(no_split, with_split=False)
        degraded_out = tmp_path / "d_nosplit.pt"
        train(str(no_split), target="d", epochs=3, out=str(degraded_out))
        rec = _read_sidecar(degraded_out)
        assert rec["training"]["official_split_requested"] is True
        assert rec["training"]["official_split_used"] is False
        assert rec["metrics"]["val"] is None


class TestEarlyStopAndValidation:
    def test_holdout_metrics_recorded(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        from scripts.train_text_affect import train

        csv_path = tmp_path / "emobank.csv"
        _make_emobank_csv(csv_path, with_split=True)
        out = tmp_path / "text.pt"
        # lr 大 + patience=1：dev 迅速不再改善，必定早停（实测 2/300 轮）
        returned = train(str(csv_path), epochs=300, lr=0.5, out=str(out), patience=1)

        rec = _read_sidecar(out)
        assert rec["training"]["official_split_requested"] is True
        assert rec["training"]["official_split_used"] is True
        val = rec["metrics"]["val"]
        assert val["split"] == "emobank-official-dev"
        assert val["n_samples"] == 2
        assert val["mse"] >= 0.0
        assert val["best_epoch"] >= 0
        # 早停真的发生了，且 stopped_early 与轮数一致（不允许两个真相源打架）
        ran, req = rec["training"]["epochs_ran"], rec["training"]["epochs_requested"]
        assert 1 <= ran < req, "本配置下应当早停；没早停则下面两条关于早停的断言都成了空转"
        assert rec["training"]["stopped_early"] is True
        # train() 返回的是 dev MSE（早停时 final_loss 被改写），而 sidecar 的 final_train_loss
        # 必须仍是**训练集**末轮 loss——两者混为一谈正是本计划要消灭的口径污染。
        assert returned == pytest.approx(val["mse"])
        assert rec["metrics"]["final_train_loss"] != pytest.approx(val["mse"])
        assert val["train_loss_at_best"] >= 0.0

    def test_split_downgrade_is_visible(self, tmp_path: Path) -> None:
        """CSV 无 split 列 → 降级读全量。requested≠used 必须在 sidecar 里看得见。

        否则报出来的分数是「记忆」还是「泛化」事后无从分辨——正是本计划要消灭的那类不可追溯。
        """
        pytest.importorskip("torch")
        from scripts.train_text_affect import train

        csv_path = tmp_path / "emobank_nosplit.csv"
        _make_emobank_csv(csv_path, with_split=False)
        out = tmp_path / "text.pt"
        train(str(csv_path), epochs=5, out=str(out))

        rec = _read_sidecar(out)
        assert rec["training"]["official_split_requested"] is True
        assert rec["training"]["official_split_used"] is False
        assert rec["metrics"]["val"] is None


class TestDegradesInsteadOfBreakingTraining:
    """sidecar 是记账，不是训练产物：任何一环失败都只降级，绝不让跑完的训练白跑。"""

    def test_write_failure_returns_none_and_does_not_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "w.pt"
        out.write_bytes(b"weights")

        def boom(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", boom)
        assert self._call(out) is None, "写不动 sidecar 只能返回 None，不能把异常抛给训练脚本"
        assert out.read_bytes() == b"weights", "权重本身不受影响"

    def test_git_unavailable_degrades_to_null(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "w.pt"
        out.write_bytes(b"weights")
        monkeypatch.setattr(_train_common, "_git", lambda _args: None)
        self._call(out)
        git = _read_sidecar(out)["git"]
        assert git == {"commit": None, "branch": None, "dirty": None}

    def test_dirty_none_is_not_clean(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """`dirty=None`（查不到）与 `dirty=False`（确认干净）语义不同，不可混为一谈。"""
        out = tmp_path / "w.pt"
        out.write_bytes(b"weights")
        monkeypatch.setattr(_train_common, "_git", lambda _args: "")
        self._call(out)
        assert _read_sidecar(out)["git"]["dirty"] is False

    def test_hostile_model_degrades_one_field_not_whole_sidecar(self, tmp_path: Path) -> None:
        """`.parameters()` 抛意料之外的异常时，丢掉的应是一个字段，不是整份 sidecar。"""

        class _HostileModel:
            def parameters(self) -> list[Any]:
                raise RuntimeError("模型对象不配合")

        out = tmp_path / "w.pt"
        out.write_bytes(b"weights")
        assert self._call(out, model=_HostileModel()) is not None
        rec = _read_sidecar(out)
        assert rec["model"]["param_count"] is None
        assert rec["model"]["class"] == "_HostileModel"
        assert rec["training"]["seed"] == 0, "其余字段照常记录"

    def test_unexpected_metadata_failure_never_escapes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """兜底层：采集环节冒出没预料到的异常，也只能变成 warning + None。

        调用点在 `torch.save` **之后**，权重已经落盘——记账失败绝不能反过来把跑完的训练
        变成一次非零退出。
        """
        out = tmp_path / "w.pt"
        out.write_bytes(b"weights")

        def boom() -> dict[str, Any]:
            raise RuntimeError("git 子系统炸了")

        monkeypatch.setattr(_train_common, "_git_info", boom)
        assert self._call(out) is None
        assert out.read_bytes() == b"weights"

    def test_reserved_keys_are_not_overridden(self, tmp_path: Path) -> None:
        """脚本传来的同名键不得顶掉核心配方——顶掉了 sidecar 就失去存在意义。"""
        out = tmp_path / "w.pt"
        out.write_bytes(b"weights")
        self._call(out, model_config={"class": "Forged"}, data_config={"seed": 999})
        rec = _read_sidecar(out)
        assert rec["model"]["class"] == "_FakeModel"
        assert rec["training"]["seed"] == 0

    def test_stopped_early_flag_is_derived(self, tmp_path: Path) -> None:
        out = tmp_path / "w.pt"
        out.write_bytes(b"weights")
        self._call(out, epochs_requested=100, epochs_ran=37)
        rec = _read_sidecar(out)
        assert rec["training"]["stopped_early"] is True
        assert rec["training"]["epochs_ran"] == 37

    @staticmethod
    def _call(out: Path, **overrides: Any) -> Path | None:
        kwargs: dict[str, Any] = {
            "script": "tests/test_train_provenance.py",
            "model": _FakeModel(),
            "model_config": {},
            "data_config": {},
            "data_source": None,
            "n_samples": 1,
            "seed": 0,
            "lr": 1e-3,
            "epochs_requested": 1,
            "epochs_ran": 1,
            "final_train_loss": 0.5,
        }
        kwargs.update(overrides)
        return write_provenance(out, **kwargs)


def test_provenance_path_appends_suffix() -> None:
    """`foo.pt` → `foo.pt.json`：追加而非替换后缀，对应关系自明、不与同目录 json 撞名。"""
    assert provenance_path("artifacts/facs_decoder.pt") == Path("artifacts/facs_decoder.pt.json")


def test_every_training_script_writes_provenance() -> None:
    """漂移守卫：新增/改写训练脚本时忘了落账，在这里被抓住。

    曾豁免过 `train_direction_head.py`，2026-07-27 补齐后**豁免集合清空**——每个产权重的脚本
    都必须落账，没有例外。
    """
    scripts = sorted((REPO_ROOT / "scripts").glob("train_*.py"))
    assert scripts, "没扫到任何训练脚本，守卫本身失效了"
    missing = [p.name for p in scripts if "write_provenance(" not in p.read_text(encoding="utf-8")]
    assert not missing, f"这些训练脚本保存权重却没写 provenance sidecar：{missing}"


def test_every_training_script_has_seed() -> None:
    """同款守卫，针对 P0-1 的 `--seed`。

    P0-1 当初正是因为**没有**这道守卫才漏掉 `train_direction_head.py`——commit message 写着
    「8 个脚本无一调用 manual_seed」，实际只改了 7 个。守卫补上，同样的漏不会再发生第二次。
    """
    scripts = sorted((REPO_ROOT / "scripts").glob("train_*.py"))
    assert scripts, "没扫到任何训练脚本，守卫本身失效了"
    missing = []
    for path in scripts:
        src = path.read_text(encoding="utf-8")
        if "--seed" not in src or "torch.manual_seed(" not in src:
            missing.append(path.name)
    assert not missing, f"这些训练脚本缺 --seed 或没调 torch.manual_seed：{missing}"
