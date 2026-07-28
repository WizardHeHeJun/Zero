"""任务 B「一条龙」端到端集成：OpenFace 格式输入 → build_facs_ext_csv 转换器 →
load_facs_csv_ext → train_facs --ext → 载回 predict。

证明用户将来跑的**完整数据管线（转换器 → 训练）就绪**——缺的仅是真 OpenFace 二进制 + 真
人脸数据集（DiffusionFER/AFEW-VA，见 PRP/facs-au-expansion/HANDOFF.md 任务 B）。本测试用
合成 OpenFace 格式 csv + 合成 VA 表替代那两者，链路其余部分（转换器 `build` / `load_facs_csv_ext`
/ `train`）全是真代码。数据到位后把这两个合成输入换成真 OpenFace 输出即 turnkey。

⚠ 预期管理：可训练的 FacsDecoder 是 predict_facs(v,a)、不吃 coping，故此链学的是通用 AU
真实度；愤怒/恐惧的 coping 分野仍由解析占位承担（真分野走任务 C，见 HANDOFF）。torch 缺失
则整文件跳过（train 需 torch）。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

pytest.importorskip("torch")


def _write_openface_csv(path: Path, *, confidence: float, aus: dict[str, str]) -> None:
    """写一个 OpenFace 风格单行 csv（confidence/success + 若干 AUxx_r∈[0,5]）。

    表头故意带前导空格模拟 OpenFace 真实输出（转换器 read_openface 会 strip）。
    """
    fields = {"confidence": str(confidence), "success": "1", **aus}
    header = ",".join(f" {k}" for k in fields)  # 前导空格 = OpenFace 真实表头习惯
    line = ",".join(fields.values())
    path.write_text(f"{header}\n{line}\n", encoding="utf-8")


def _write_va(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _first_row_keys(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(next(csv.DictReader(fh)).keys())


def test_converter_to_train_one_shot(tmp_path: Path) -> None:
    """合成 OpenFace 输出 + 原生 VA 表 → 真 build() → train --ext → 权重 → predict 13 键。

    覆盖多象限（正效价/负高唤醒/中性）；愤怒与恐惧图共 (v,a)=(-0.6,0.6)（VA 上同点，分野属
    coping 维，见模块 docstring 与转换器 EMOTION_TO_VA）。
    """
    torch = pytest.importorskip("torch")

    from scripts.build_facs_ext_csv import OUT_COLUMNS, build
    from scripts.train_facs import train
    from src.agents.datasets.facs_csv import load_facs_csv_ext
    from src.agents.models.facs_decoder import FACS_KEYS_EXT, FacsDecoder

    # 1) 合成 OpenFace 输出目录（每图一 csv，AUxx_r∈[0,5]）——替代真 OpenFace 二进制
    of_dir = tmp_path / "of_out"
    of_dir.mkdir()
    of_specs: dict[str, dict[str, str]] = {
        "img_happy": {"AU06_r": "3.0", "AU12_r": "4.5"},  # 正效价笑肌
        # 愤怒判别 AU23 + AU17（angrily-disgusted 复合含颏肌·Du 2014，验证 AU17 抽取）
        "img_anger": {"AU04_r": "3.0", "AU07_r": "2.0", "AU17_r": "2.5", "AU23_r": "4.0"},
        # 恐惧 + AU26 下颌落（恐惧原型含 jaw drop·Ekman 1978，验证 AU26 抽取）
        "img_fear": {
            "AU01_r": "3.5",
            "AU02_r": "3.5",
            "AU05_r": "2.5",
            "AU20_r": "3.0",
            "AU26_r": "3.0",
        },
        "img_neutral": {},  # 中性：无 AU 列 → 转换器填 0
    }
    for name, aus in of_specs.items():
        _write_openface_csv(of_dir / f"{name}.csv", confidence=0.95, aus=aus)

    # 2) 合成原生 VA 表（DiffusionFER 风格：每图 valence/arousal）——替代真数据集标注
    va_csv = tmp_path / "va.csv"
    _write_va(
        va_csv,
        [
            {"image": "img_happy", "valence": "0.8", "arousal": "0.5"},
            {"image": "img_anger", "valence": "-0.6", "arousal": "0.6"},
            {"image": "img_fear", "valence": "-0.6", "arousal": "0.6"},
            {"image": "img_neutral", "valence": "0.0", "arousal": "0.0"},
        ],
    )

    # 3) 真转换器 build() → labels_ext.csv（按图名内连接、AU /5、intensity=均值）
    labels = tmp_path / "labels_ext.csv"
    n_rows = build(of_dir, va_csv, labels, va_id_col="image")
    assert n_rows == 4, "四图两侧都在，应产 4 行"
    assert _first_row_keys(labels) == OUT_COLUMNS, "转换器输出列须对齐 OUT_COLUMNS"
    # 任务 D 防假绿：断言转换器正确抽取新增 AU17/AU26 列并 /5 归一（否则抽取路径 bug 仍全绿）。
    # labels 行不含图名、img_anger/img_fear 的 (v,a) 同为 (-0.6,0.6) 无法按行区分——用跨行最大值：
    # 仅 img_anger 有 AU17_r=2.5、仅 img_fear 有 AU26_r=3.0，故 max(AU17)=0.5、max(AU26)=0.6。
    with open(labels, newline="", encoding="utf-8") as fh:
        label_rows = list(csv.DictReader(fh))
    assert max(float(r["AU17"]) for r in label_rows) == pytest.approx(0.5), "AU17_r=2.5 → /5=0.5"
    assert max(float(r["AU26"]) for r in label_rows) == pytest.approx(0.6), "AU26_r=3.0 → /5=0.6"

    # 4) load_facs_csv_ext 读回 → 形状对（管线中段：转换器输出可被训练 loader 消费）
    x, y = load_facs_csv_ext(str(labels))
    assert x.shape == (n_rows, 2)
    assert y.shape == (n_rows, len(FACS_KEYS_EXT))

    # 5) train --ext → 隔离命名权重（管线末段：真训练循环消费转换器产物）
    out = tmp_path / "facs_decoder_ext.pt"
    final = train(str(labels), extended=True, epochs=5, stop="fixed", out=str(out))
    assert out.exists(), "端到端应产出扩展权重文件"
    assert math.isfinite(final), f"最终 loss 应有限（整链通、无 NaN），实为 {final}"
    assert out.name == "facs_decoder_ext.pt", "隔离命名，不覆盖旧 5-AU 权重"

    # 6) 载回扩展模型 → predict 13 键（证权重可用）
    model = FacsDecoder(extended=True)
    model.load_state_dict(torch.load(out, map_location="cpu", weights_only=True))
    facs = model.predict_facs(-0.6, 0.6)
    assert set(facs) == set(FACS_KEYS_EXT), f"predict_facs 应输出 13 键，实为 {set(facs)}"


def test_category_source_one_shot(tmp_path: Path) -> None:
    """仅类别标注（RAVDESS/ExpW 风格）走 --category-col → 转换器映射 (v,a) → train --ext。

    证明「只有情绪类别、无原生 VA」的数据集也能一条龙（类别→Russell circumplex→(v,a)）。
    """
    torch = pytest.importorskip("torch")

    from scripts.build_facs_ext_csv import build
    from scripts.train_facs import train
    from src.agents.models.facs_decoder import FACS_KEYS_EXT, FacsDecoder

    of_dir = tmp_path / "of_out"
    of_dir.mkdir()
    _write_openface_csv(
        of_dir / "clip_happy.csv", confidence=0.9, aus={"AU06_r": "3.0", "AU12_r": "4.0"}
    )
    _write_openface_csv(
        of_dir / "clip_anger.csv", confidence=0.9, aus={"AU04_r": "3.0", "AU23_r": "4.0"}
    )

    va_csv = tmp_path / "labels.csv"
    _write_va(
        va_csv,
        [
            {"image": "clip_happy", "emotion": "happy"},
            {"image": "clip_anger", "emotion": "anger"},
        ],
    )

    labels = tmp_path / "labels_ext.csv"
    n_rows = build(of_dir, va_csv, labels, va_id_col="image", category_col="emotion")
    assert n_rows == 2

    out = tmp_path / "facs_decoder_ext.pt"
    final = train(str(labels), extended=True, epochs=5, stop="fixed", out=str(out))
    assert out.exists(), "category 路径应产出扩展权重"
    assert math.isfinite(final), f"category 路径 loss 应有限，实为 {final}"

    model = FacsDecoder(extended=True)
    model.load_state_dict(torch.load(out, map_location="cpu", weights_only=True))
    assert set(model.predict_facs(0.8, 0.5)) == set(FACS_KEYS_EXT)
