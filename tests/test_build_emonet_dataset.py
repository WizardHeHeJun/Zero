"""scripts/build_emonet_dataset.py 单测（emonet parquet → 均衡子集图 + VA 表）。

需 pyarrow/PIL（.[data] extra）；缺则整文件跳过。合成微型 parquet（含 emonet 的
emotion ClassLabel HF metadata + path struct 图 bytes）驱动，无需真数据集。
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("PIL")

import io  # noqa: E402

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402

from scripts.build_emonet_dataset import (  # noqa: E402
    EMONET_VA,
    _short_prefix,
    build_emonet_dataset,
)


def _tiny_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (128, 64, 32)).save(buf, "JPEG")
    return buf.getvalue()


def _make_emonet_parquet(path: Path, emotions_present: list[str], per: int) -> None:
    """合成 emonet 结构 parquet：path struct{bytes,path} + emotion int + HF ClassLabel 元数据。"""
    names = sorted(EMONET_VA)  # 全 40 类作 ClassLabel 名（与真数据集同构）
    jpg = _tiny_jpeg()
    paths, emos = [], []
    idx = 0
    for e in emotions_present:
        for _ in range(per):
            paths.append({"bytes": jpg, "path": f"{idx:05d}.jpg"})
            emos.append(names.index(e))
            idx += 1
    path_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    tbl = pa.table({"path": pa.array(paths, type=path_type), "emotion": pa.array(emos, pa.int64())})
    hf_meta = {
        "info": {
            "features": {
                "emotion": {"_type": "ClassLabel", "names": names},
                "path": {"_type": "Image"},
            }
        }
    }
    tbl = tbl.replace_schema_metadata({b"huggingface": json.dumps(hf_meta).encode()})
    pq.write_table(tbl, path)


def test_emonet_va_map_valid() -> None:
    """EMONET_VA：40 类、值均 ∈[-1,1]（近似 circumplex 坐标健全性）。"""
    assert len(EMONET_VA) == 40
    for emo, (v, a) in EMONET_VA.items():
        assert -1.0 <= v <= 1.0, f"{emo} valence 越界：{v}"
        assert -1.0 <= a <= 1.0, f"{emo} arousal 越界：{a}"
    # 愤怒与恐惧有意同 (v,a)（分野属 coping 维、不在 VA 上）
    assert EMONET_VA["Anger"] == EMONET_VA["Fear"]


def test_emonet_va_no_unendorsed_collision() -> None:
    """几何守卫：除议会背书的同点外，任意两类 (v,a) 不得重合。

    重合等于给 FacsDecoder 矛盾监督（同一输入两组 AU 目标，只能学到平均）。
    2026-07-25 议会复审曾因逐类改坐标、未复核全局几何而新造 3 处零距离碰撞，
    经二轮几何仲裁修复；本用例把该教训固化为回归守卫。

    ⚠ **限度**：本用例只拦截**精确重合**，**不校验 0.10 工作阈值**——把某坐标改到与
    邻居只差 0.02，本用例照样通过。「测试绿」≠「几何仍安全」；改坐标后仍须另跑全表
    距离验算（见 notes/2026-07-25-emonet-va-geometry-round3-council.md）。
    """
    intentional = {"Anger", "Fear", "Distress"}  # 分野属 coping 维、不在 VA 上（议会背书）
    for (n1, p1), (n2, p2) in itertools.combinations(EMONET_VA.items(), 2):
        if n1 in intentional and n2 in intentional:
            continue
        assert p1 != p2, f"{n1} 与 {n2} 坐标重合 {p1}——未经议会背书的同点会制造矛盾监督"


def test_short_prefix() -> None:
    assert _short_prefix("Amusement") == "Amuse"[:6] or _short_prefix("Amusement") == "Amusem"
    assert _short_prefix("Astonishment/Surprise") == "Astoni"  # 取首词截 6
    assert " " not in _short_prefix("Sexual Lust")
    assert "&" not in _short_prefix("Jealousy & Envy")


def test_build_balanced_subset(tmp_path: Path) -> None:
    """两类各 5 张、n_per=3 → 抽 6 张（每类 3），VA 表列/值正确、图落盘。"""
    pqdir = tmp_path / "pq"
    pqdir.mkdir()
    _make_emonet_parquet(pqdir / "train-00000-of-00001.parquet", ["Amusement", "Fear"], per=5)

    img_out = tmp_path / "images"
    va_out = tmp_path / "va.csv"
    n = build_emonet_dataset(pqdir, img_out, va_out, n_per=3)

    assert n == 6, "两类各 3 张（n_per 均衡截断）"
    assert len(list(img_out.glob("*.jpg"))) == 6

    with open(va_out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert set(rows[0].keys()) == {"image", "valence", "arousal"}
    by_emo: dict[tuple[str, str], int] = {}
    for r in rows:
        by_emo[(r["valence"], r["arousal"])] = by_emo.get((r["valence"], r["arousal"]), 0) + 1
    # Amusement 经 2026-07-25 议会复审下调 arousal 至 0.2；Fear (-0.6,0.6) 不动。
    assert by_emo[("0.8", "0.2")] == 3
    assert by_emo[("-0.6", "0.6")] == 3


def test_clean_wipes_prior_batch(tmp_path: Path) -> None:
    """clean=True（默认）先清空 img_out，避免混入旧批。"""
    pqdir = tmp_path / "pq"
    pqdir.mkdir()
    _make_emonet_parquet(pqdir / "train-00000-of-00001.parquet", ["Anger"], per=4)
    img_out = tmp_path / "images"
    img_out.mkdir()
    (img_out / "stale.jpg").write_bytes(b"old")

    build_emonet_dataset(pqdir, img_out, tmp_path / "va.csv", n_per=2)
    assert not (img_out / "stale.jpg").exists(), "旧批应被清空"
    assert len(list(img_out.glob("*.jpg"))) == 2


def test_missing_parquet_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_emonet_dataset(tmp_path, tmp_path / "img", tmp_path / "va.csv")
