"""OpenFace AU 输出 + VA 标注 → FacsDecoder 扩展训练集 `labels_ext.csv` 转换器。

个人无挂靠训练路径的「合成」这一步（见 `PRP/facs-au-expansion/HANDOFF.md` 任务 B）：

  1. 对人脸图跑 **OpenFace**（`FaceLandmarkImg -fdir <图目录> -aus -out_dir <of_out>` 每图一个 csv；
     或 `FeatureExtraction`），得每图 `AUxx_r`(0-5) + `confidence`/`success`。
  2. 本脚本把 OpenFace 的 AU 与每图的 (valence, arousal) 按图名拼、归一化，产 `labels_ext.csv`。
  3. `python -m scripts.train_facs --csv <labels_ext.csv> --ext` → `facs_decoder_ext.pt`。

VA 来源二选一：
  `--va-csv PATH`        原生 VA：CSV 每行含 图标识 + valence + arousal（如 DiffusionFER sheet）。
                         `--va-scale`：VA∈[-10,10] 传 10、[-1,1] 传 1（归一到 [-1,1]）。
  `--category-col NAME`  只有情绪类别时：从 VA-csv 的类别列按 Russell circumplex 映射到 (v,a)
                         （给 RAVDESS/ExpW/EmoNet 等只有类别的集用）。

纯 stdlib、torch-free（数据预处理，装不装 ml extra 都能跑）。输出列顺序对齐 `FACS_KEYS_EXT`。

⚠ **关于愤怒/恐惧区分（重要预期管理）**：当前可训练的 `FacsDecoder` 是 `predict_facs(v, a)`、
  **不吃 `coping_potential`**（协议限制，HANDOFF 任务 C）。而愤怒与恐惧在 (v,a) 上是同一点，
  故**真权重训练改善的是通用 AU 真实度（喜/悲/惊/中性…），愤怒/恐惧的 coping 分野仍由解析占位
  `_decode_facs_extended` 承担**。若要把该分野也从数据学，需任务 C（把模型扩成 (v,a,coping)→AU）
  + 类别→coping 标签——本脚本 `--coping-from-category` 会额外输出一列 `coping`（愤怒→+/恐惧→−）
  先把该列备好（当前 `load_facs_csv_ext` 会忽略多余列，不影响现训练）。
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# OpenFace 的 _r 强度列（0-5）对应的 12 个目标 AU；顺序须对齐 facs_decoder.FACS_KEYS_EXT
# （本脚本 torch-free，不 import 该常量以免拉 torch；tests 有漂移守卫断言二者一致）。
# 任务 D：加 AU17(颏肌·厌恶/悲伤)/AU26(下颌落·恐惧/惊讶) 两个通用 AU（Ekman 编号序插入）。
AU_TARGETS = [
    "AU01",
    "AU02",
    "AU04",
    "AU05",
    "AU06",
    "AU07",
    "AU12",
    "AU15",
    "AU17",
    "AU20",
    "AU23",
    "AU26",
]
OUT_COLUMNS = ["valence", "arousal", *AU_TARGETS, "intensity"]

# Russell circumplex：常见情绪标签 → (v,a) 锚点 ∈[-1,1]（仿本项目 ravdess.EMOTION_CODE_TO_VA）。
# 注意：anger 与 fear 有意映到同一 (v,a)——二者的分野属 coping 维、不在 VA 上（见模块 docstring）。
EMOTION_TO_VA: dict[str, tuple[float, float]] = {
    "neutral": (0.0, 0.0),
    "calm": (0.3, -0.4),
    "happy": (0.8, 0.5),
    "happiness": (0.8, 0.5),
    "joy": (0.8, 0.5),
    "sad": (-0.6, -0.4),
    "sadness": (-0.6, -0.4),
    "angry": (-0.6, 0.6),
    "anger": (-0.6, 0.6),
    "fear": (-0.6, 0.6),
    "fearful": (-0.6, 0.6),
    "afraid": (-0.6, 0.6),
    "disgust": (-0.6, 0.2),
    "disgusted": (-0.6, 0.2),
    "surprise": (0.4, 0.7),
    "surprised": (0.4, 0.7),
    "contempt": (-0.4, 0.1),
}
# 类别 → coping（愤怒=高控制/趋近、恐惧=低控制/回避；供任务 C 的 (v,a,coping) 训练备列）。
EMOTION_TO_COPING: dict[str, float] = {
    "angry": 0.7,
    "anger": 0.7,
    "fear": -0.7,
    "fearful": -0.7,
    "afraid": -0.7,
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm_id(raw: str) -> str:
    """图标识归一：取无扩展名的 basename、小写，供 OpenFace 输出与 VA 表稳健对拼。"""
    return Path(str(raw).strip().replace("\\", "/")).stem.lower()


def read_openface(
    openface_dir: str | Path, *, id_col: str | None = None, conf_min: float = 0.8
) -> dict[str, dict[str, float]]:
    """读 OpenFace 输出（目录下所有 .csv，每图一个 csv 的常见情形），返回 {图id: {AU: 归一强度}}。

    - OpenFace 表头常带前导空格（如 ` AU01_r`）——统一 strip。
    - 仅保留 `success==1` 且 `confidence>=conf_min` 的行；同一 id 多行（视频帧）取置信度最高者。
    - AU 归一：`AUxx_r / 5 → [0,1]`；缺失的目标 AU 列填 0 并告警一次。
    - id 来源：给了 `id_col` 且存在则用该列，否则用 csv 文件名 stem（每图一 csv 约定）。
    """
    out: dict[str, dict[str, float]] = {}
    best_conf: dict[str, float] = {}
    warned_missing = False
    paths = sorted(Path(openface_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"{openface_dir} 下无 .csv（OpenFace 输出）")
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldmap = {(k or "").strip(): k for k in (reader.fieldnames or [])}
            for row in reader:
                clean = {k.strip(): v for k, v in row.items()}
                if clean.get("success", "1").strip() not in ("1", "1.0"):
                    continue
                try:
                    conf = float(clean.get("confidence", "1"))
                except ValueError:
                    conf = 1.0
                if conf < conf_min:
                    continue
                if id_col and id_col in clean:
                    fid = _norm_id(clean[id_col])
                else:
                    fid = path.stem.lower()
                if fid in best_conf and best_conf[fid] >= conf:
                    continue
                aus: dict[str, float] = {}
                for au in AU_TARGETS:
                    col = f"{au}_r"
                    if col in clean:
                        try:
                            aus[au] = _clamp(float(clean[col]) / 5.0, 0.0, 1.0)
                        except ValueError:
                            aus[au] = 0.0
                    else:
                        aus[au] = 0.0
                        if not warned_missing:
                            logger.warning(
                                "OpenFace 缺 %s_r 列（%s）——填 0；确认跑 OpenFace 带了 -aus",
                                au,
                                list(fieldmap),
                            )
                            warned_missing = True
                out[fid] = aus
                best_conf[fid] = conf
    return out


def read_va(
    va_csv: str | Path,
    *,
    id_col: str,
    valence_col: str = "valence",
    arousal_col: str = "arousal",
    category_col: str | None = None,
    va_scale: float = 1.0,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """读 VA 标注表，返回 ({图id: (v,a)}, {图id: coping})。

    - 原生 VA：读 `valence_col`/`arousal_col`，`/va_scale` 后 clamp 到 [-1,1]。
    - 类别 VA：给 `category_col` 时，按 EMOTION_TO_VA 映射类别→(v,a)、EMOTION_TO_COPING→coping。
    - 无法解析/未知类别的行跳过并告警。
    """
    va: dict[str, tuple[float, float]] = {}
    coping: dict[str, float] = {}
    with open(va_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            clean = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            if id_col not in clean:
                raise KeyError(f"VA csv 缺 id 列 {id_col!r}；实际列={list(clean)}")
            fid = _norm_id(clean[id_col])
            if category_col:
                cat = clean.get(category_col, "").lower()
                if cat not in EMOTION_TO_VA:
                    logger.warning("未知情绪类别 %r（id=%s）跳过", cat, fid)
                    continue
                va[fid] = EMOTION_TO_VA[cat]
                coping[fid] = EMOTION_TO_COPING.get(cat, 0.0)
            else:
                try:
                    v = _clamp(float(clean[valence_col]) / va_scale, -1.0, 1.0)
                    a = _clamp(float(clean[arousal_col]) / va_scale, -1.0, 1.0)
                except (KeyError, ValueError):
                    logger.warning("VA 行无法解析（id=%s）跳过", fid)
                    continue
                va[fid] = (v, a)
    return va, coping


def build(
    openface_dir: str | Path,
    va_csv: str | Path,
    out: str | Path,
    *,
    id_col: str | None = None,
    va_id_col: str = "image",
    category_col: str | None = None,
    va_scale: float = 1.0,
    conf_min: float = 0.8,
    with_coping: bool = False,
) -> int:
    """拼 OpenFace AU 与 VA，写 labels_ext.csv，返回写出的行数。

    intensity = 10 个 AU 归一强度的均值（∈[0,1]，整体面部激活代理）。
    仅对两侧都存在（按归一 id 内连接）的图写行。`with_coping` 额外写 coping 列（任务 C 备用）。
    """
    au_by_id = read_openface(openface_dir, id_col=id_col, conf_min=conf_min)
    va_by_id, coping_by_id = read_va(
        va_csv, id_col=va_id_col, category_col=category_col, va_scale=va_scale
    )
    matched = sorted(set(au_by_id) & set(va_by_id))
    if not matched:
        raise ValueError(
            f"OpenFace({len(au_by_id)}) 与 VA({len(va_by_id)}) 无一图 id 对上——"
            f"核对 --id-col/--va-id-col 与图名是否一致"
        )
    columns = [*OUT_COLUMNS, "coping"] if with_coping else OUT_COLUMNS
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for fid in matched:
            aus = au_by_id[fid]
            v, a = va_by_id[fid]
            intensity = sum(aus[au] for au in AU_TARGETS) / len(AU_TARGETS)
            record = {"valence": v, "arousal": a, "intensity": intensity, **aus}
            if with_coping:
                record["coping"] = coping_by_id.get(fid, 0.0)
            writer.writerow(record)
    logger.info(
        "写出 %d 行 → %s（OpenFace %d 图 ∩ VA %d 图）",
        len(matched),
        out_path,
        len(au_by_id),
        len(va_by_id),
    )
    return len(matched)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="OpenFace AU + VA → labels_ext.csv")
    parser.add_argument("--openface", required=True, help="OpenFace 输出 .csv 所在目录")
    parser.add_argument("--va-csv", required=True, help="VA/情绪标注 CSV")
    parser.add_argument(
        "--out", default="data/facs/labels_ext.csv", help="输出 labels_ext.csv 路径"
    )
    parser.add_argument(
        "--id-col", default=None, help="OpenFace csv 里的图标识列（默认用 csv 文件名 stem）"
    )
    parser.add_argument("--va-id-col", default="image", help="VA csv 里的图标识列")
    parser.add_argument(
        "--category-col", default=None, help="VA csv 里的情绪类别列（给则走类别→VA 映射）"
    )
    parser.add_argument(
        "--va-scale", type=float, default=1.0, help="原生 VA 归一除数（[-10,10] 传 10）"
    )
    parser.add_argument("--conf-min", type=float, default=0.8, help="OpenFace 置信度下限")
    parser.add_argument(
        "--coping-from-category",
        action="store_true",
        help="额外输出 coping 列（愤怒+/恐惧−，供任务 C 的 (v,a,coping) 训练）",
    )
    args = parser.parse_args()
    n = build(
        args.openface,
        args.va_csv,
        args.out,
        id_col=args.id_col,
        va_id_col=args.va_id_col,
        category_col=args.category_col,
        va_scale=args.va_scale,
        conf_min=args.conf_min,
        with_coping=args.coping_from_category,
    )
    print(f"done, {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
