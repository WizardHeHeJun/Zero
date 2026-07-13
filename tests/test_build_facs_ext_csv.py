"""scripts/build_facs_ext_csv.py 单测（OpenFace + VA → labels_ext.csv 转换器）。

纯 stdlib（转换器 torch-free）；覆盖：AU 归一(/5)、VA 归一(/scale, clamp)、置信度/success 过滤、
按图 id 内连接、intensity=均值、类别→(v,a)+coping 映射、列对齐 FACS_KEYS_EXT 漂移守卫。
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.build_facs_ext_csv import AU_TARGETS, OUT_COLUMNS, build


def _write_of_csv(path: Path, values: dict[str, str], *, leading_space: bool = False) -> None:
    """写一个 OpenFace 风格单行 csv（可选给表头加前导空格，模拟 OpenFace 真实表头）。"""
    prefix = " " if leading_space else ""
    header = ",".join(f"{prefix}{k}" for k in values)
    line = ",".join(values.values())
    path.write_text(f"{header}\n{line}\n", encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture()
def openface_dir(tmp_path: Path) -> Path:
    """四图：img_a(高置信 AU01/AU23)、img_b(前导空格表头)、img_low(低置信)、img_fail(success=0)。"""
    d = tmp_path / "of_out"
    d.mkdir()
    _write_of_csv(
        d / "img_a.csv",
        {"confidence": "0.95", "success": "1", "AU01_r": "1.0", "AU23_r": "5.0", "AU09_r": "3.0"},
    )
    _write_of_csv(
        d / "img_b.csv",
        {"confidence": "0.90", "success": "1", "AU12_r": "2.5"},
        leading_space=True,  # 模拟 OpenFace 表头前导空格
    )
    _write_of_csv(d / "img_low.csv", {"confidence": "0.50", "success": "1", "AU01_r": "5.0"})
    _write_of_csv(d / "img_fail.csv", {"confidence": "0.99", "success": "0", "AU01_r": "5.0"})
    return d


def _write_va(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class TestNativeVA:
    """原生 VA 源（--va-csv，valence/arousal 列）。"""

    def test_basic_join_and_normalization(self, openface_dir: Path, tmp_path: Path) -> None:
        va = tmp_path / "va.csv"
        _write_va(
            va,
            [
                {"image": "img_a", "valence": "0.8", "arousal": "0.5"},
                {"image": "img_b", "valence": "-0.6", "arousal": "0.6"},
                {"image": "img_low", "valence": "0.0", "arousal": "0.0"},  # OpenFace 侧被置信过滤
                {"image": "img_unmatched", "valence": "0.1", "arousal": "0.1"},  # 无 OpenFace
            ],
        )
        out = tmp_path / "labels_ext.csv"
        n = build(openface_dir, va, out, va_id_col="image")
        assert n == 2, (
            "只有 img_a/img_b 两侧都在（img_low 低置信、img_fail success=0、unmatched 无脸）"
        )

        rows = {r["valence"] + "," + r["arousal"]: r for r in _read_rows(out)}
        a = rows["0.8,0.5"]
        assert list(_read_rows(out)[0].keys()) == OUT_COLUMNS, "列顺序须对齐 OUT_COLUMNS"
        assert float(a["AU01"]) == pytest.approx(0.2), "AU01_r=1.0 → /5 = 0.2"
        assert float(a["AU23"]) == pytest.approx(1.0), "AU23_r=5.0 → /5 = 1.0"
        # intensity = 10 个目标 AU 均值 = (0.2 + 1.0)/10 = 0.12（AU09 非目标，不计）
        assert float(a["intensity"]) == pytest.approx(0.12)

    def test_va_scale(self, openface_dir: Path, tmp_path: Path) -> None:
        """VA∈[-10,10] 时 --va-scale=10 归一到 [-1,1]。"""
        va = tmp_path / "va.csv"
        _write_va(va, [{"image": "img_a", "valence": "-6.0", "arousal": "6.0"}])
        out = tmp_path / "labels_ext.csv"
        build(openface_dir, va, out, va_id_col="image", va_scale=10.0)
        r = _read_rows(out)[0]
        assert float(r["valence"]) == pytest.approx(-0.6)
        assert float(r["arousal"]) == pytest.approx(0.6)

    def test_va_clamped(self, openface_dir: Path, tmp_path: Path) -> None:
        """超界 VA 被 clamp 到 [-1,1]。"""
        va = tmp_path / "va.csv"
        _write_va(va, [{"image": "img_a", "valence": "2.0", "arousal": "-2.0"}])
        out = tmp_path / "labels_ext.csv"
        build(openface_dir, va, out, va_id_col="image")
        r = _read_rows(out)[0]
        assert float(r["valence"]) == pytest.approx(1.0)
        assert float(r["arousal"]) == pytest.approx(-1.0)


class TestCategoryVA:
    """类别 → (v,a) + coping 映射（--category-col）。"""

    def test_category_maps_to_va_and_coping(self, openface_dir: Path, tmp_path: Path) -> None:
        va = tmp_path / "va.csv"
        _write_va(
            va,
            [
                {"image": "img_a", "emotion": "anger"},
                {"image": "img_b", "emotion": "fear"},
            ],
        )
        out = tmp_path / "labels_ext.csv"
        build(openface_dir, va, out, va_id_col="image", category_col="emotion", with_coping=True)
        rows = {r["coping"]: r for r in _read_rows(out)}
        # anger 与 fear 同 (v,a)=(-0.6,0.6)，靠 coping 分野
        anger = rows["0.7"]
        fear = rows["-0.7"]
        assert float(anger["valence"]) == pytest.approx(-0.6)
        assert float(anger["arousal"]) == pytest.approx(0.6)
        assert float(fear["valence"]) == pytest.approx(-0.6)
        assert float(fear["arousal"]) == pytest.approx(0.6)
        assert "coping" in _read_rows(out)[0], "with_coping=True 应输出 coping 列"

    def test_unknown_category_skipped(self, openface_dir: Path, tmp_path: Path) -> None:
        va = tmp_path / "va.csv"
        _write_va(
            va,
            [
                {"image": "img_a", "emotion": "anger"},
                {"image": "img_b", "emotion": "bogus"},  # 未知类别 → 跳过
            ],
        )
        out = tmp_path / "labels_ext.csv"
        n = build(openface_dir, va, out, va_id_col="image", category_col="emotion")
        assert n == 1, "未知类别行应被跳过"


def test_no_match_raises(openface_dir: Path, tmp_path: Path) -> None:
    """OpenFace 与 VA 无一图对上 → 明确报错（防静默产空表）。"""
    va = tmp_path / "va.csv"
    _write_va(va, [{"image": "totally_other", "valence": "0.1", "arousal": "0.1"}])
    out = tmp_path / "labels_ext.csv"
    with pytest.raises(ValueError, match="无一图 id 对上"):
        build(openface_dir, va, out, va_id_col="image")


def test_columns_match_facs_keys_ext() -> None:
    """漂移守卫：OUT_COLUMNS 的 AU 段 + intensity 必须逐字对齐 FACS_KEYS_EXT。"""
    pytest.importorskip("torch")
    from src.agents.models.facs_decoder import FACS_KEYS_EXT

    assert [*AU_TARGETS, "intensity"] == list(FACS_KEYS_EXT), (
        "转换器 AU 列/顺序与 FacsDecoder FACS_KEYS_EXT 漂移了"
    )
    assert OUT_COLUMNS[2:] == list(FACS_KEYS_EXT)
