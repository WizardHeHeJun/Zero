"""emonet-face-binary（HuggingFace·CC-BY-4.0）parquet → 抽均衡子集图 + VA 表。

真权重训练管线的「数据准备」这一步（见 PRP/facs-au-expansion/HANDOFF.md 任务 B）：

  1. 下数据集：`hf download laion/emonet-face-binary --repo-type dataset`（需 HF 登录）。
  2. **本脚本**：从 parquet 抽均衡子集图（每类 N 张）到 `--img-out`，并按 40 类→(v,a) 映射
     生成 `--va-out` 的 VA 表（列 image,valence,arousal，供 build_facs_ext_csv --va-csv）。
  3. 对 `--img-out` 跑 OpenFace（`FaceLandmarkImg -fdir <img-out> -aus -out_dir of_out`）。
  4. `python -m scripts.build_facs_ext_csv --openface of_out --va-csv <va-out> --va-id-col image
     --out data/facs/labels_ext.csv` → `python -m scripts.train_facs --csv ... --ext`。

依赖 `pip install -e .[data]`（pyarrow/pillow/huggingface-hub）。图内嵌 parquet、须 pyarrow 读。

⚠ **40 类→(v,a) 映射的忠实性（重要）**：EMONET_VA 是**近似 circumplex 坐标**（Russell 1980
环状模型结构 + 情感常模惯例），用作训练标签——同本项目对 RAVDESS/WESAD 的 category→VA 既有
做法（`ravdess.EMOTION_CODE_TO_VA`）。**坐标为工程近似、可校准，是科学家议会 fidelity 复审的
候选**（尤其 Awe/Longing/Intoxication 等模糊类）。愤怒/恐惧有意映到同一 (v,a)——二者分野属
coping 维、不在 VA 上（FacsDecoder predict_facs(v,a) 学通用 AU，coping 分野由占位/C2 承担）。
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# emonet 40 类 → (valence, arousal) ∈[-1,1]：近似 circumplex 坐标·可校准（见模块 docstring）。
EMONET_VA: dict[str, tuple[float, float]] = {
    "Affection": (0.7, 0.1),
    "Amusement": (0.8, 0.4),
    "Anger": (-0.6, 0.6),
    "Astonishment/Surprise": (0.15, 0.75),
    "Awe": (0.4, 0.5),
    "Bitterness": (-0.5, 0.1),
    "Concentration": (0.05, 0.25),
    "Confusion": (-0.25, 0.25),
    "Contemplation": (0.05, -0.15),
    "Contempt": (-0.45, 0.15),
    "Contentment": (0.7, -0.35),
    "Disappointment": (-0.5, -0.15),
    "Disgust": (-0.6, 0.35),
    "Distress": (-0.6, 0.6),
    "Doubt": (-0.25, 0.05),
    "Elation": (0.8, 0.7),
    "Embarrassment": (-0.3, 0.45),
    "Emotional Numbness": (-0.2, -0.6),
    "Fatigue/Exhaustion": (-0.3, -0.75),
    "Fear": (-0.6, 0.6),  # 与 Anger/Distress 同 (v,a)：愤怒/恐惧分野属 coping 维、不在 VA 上
    "Hope/Enthusiasm/Optimism": (0.6, 0.45),
    "Hopelessness": (-0.7, -0.1),
    "Impatience and Irritability": (-0.4, 0.5),
    "Infatuation": (0.6, 0.5),
    "Interest": (0.5, 0.35),
    "Intoxication/Altered States of Consciousness": (0.1, -0.1),
    "Jealousy & Envy": (-0.5, 0.45),
    "Longing": (-0.15, 0.15),
    "Malevolence/Malice": (-0.6, 0.4),
    "Pain": (-0.7, 0.55),
    "Pleasure/Ecstasy": (0.9, 0.6),
    "Pride": (0.65, 0.45),
    "Relief": (0.5, -0.3),
    "Sadness": (-0.6, -0.4),
    "Sexual Lust": (0.5, 0.75),
    "Shame": (-0.55, 0.2),
    "Sourness": (-0.4, -0.05),
    "Teasing": (0.4, 0.4),
    "Thankfulness/Gratitude": (0.7, 0.1),
    "Triumph": (0.75, 0.65),
}


def _emotion_names(pf: object) -> list[str]:
    """从 parquet 的 HF schema metadata 取 emotion ClassLabel 名（int 索引→名）。"""
    meta = json.loads(pf.schema_arrow.metadata[b"huggingface"])  # type: ignore[attr-defined]
    return meta["info"]["features"]["emotion"]["names"]


def _short_prefix(emotion: str) -> str:
    """类前缀（防跨类图名撞车）：取首词、去空格/&、截 6 字符。"""
    return emotion.split("/")[0].replace(" ", "").replace("&", "")[:6]


def build_emonet_dataset(
    parquet_dir: str | Path,
    img_out: str | Path,
    va_out: str | Path,
    *,
    n_per: int = 50,
    clean: bool = True,
) -> int:
    """抽 emonet 均衡子集图 + 写 VA 表，返回写出的图数。

    parquet_dir：含 train-*-of-*.parquet 的目录（HF 缓存快照目录或 --local-dir 下载目录）。
    n_per：每类抽多少张（均衡）。clean=True 时先清空 img_out（避免混入旧批）。
    """
    import pyarrow.parquet as pq
    from PIL import Image

    img_out = Path(img_out)
    va_out = Path(va_out)
    if clean and img_out.is_dir():
        shutil.rmtree(img_out)
    img_out.mkdir(parents=True, exist_ok=True)
    va_out.parent.mkdir(parents=True, exist_ok=True)

    pqs = sorted(glob.glob(os.path.join(str(parquet_dir), "**", "train-*.parquet"), recursive=True))
    if not pqs:
        raise FileNotFoundError(f"{parquet_dir} 下无 train-*.parquet")

    per: dict[str, int] = defaultdict(int)
    rows_va: list[tuple[str, float, float]] = []
    emo_names: list[str] | None = None
    for pqpath in pqs:
        pf = pq.ParquetFile(pqpath)
        if emo_names is None:
            emo_names = _emotion_names(pf)
            missing = set(emo_names) - set(EMONET_VA)
            if missing:
                raise ValueError(f"EMONET_VA 缺类：{missing}（数据集 emotion 类名与映射漂移）")
        for rg in range(pf.num_row_groups):
            if all(per[e] >= n_per for e in emo_names):
                break
            for r in pf.read_row_group(rg).to_pylist():
                emo = emo_names[r["emotion"]]
                if per[emo] >= n_per:
                    continue
                stem = Path(r["path"]["path"]).stem
                fid = f"{_short_prefix(emo)}_{stem}"
                try:
                    im = Image.open(io.BytesIO(r["path"]["bytes"])).convert("RGB")
                except Exception as exc:  # noqa: BLE001 — 单张解码失败跳过不中断全批
                    logger.warning("解码失败跳过 %s：%s", stem, exc)
                    continue
                im.save(img_out / f"{fid}.jpg", "JPEG", quality=92)
                v, a = EMONET_VA[emo]
                rows_va.append((fid, v, a))
                per[emo] += 1

    with open(va_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "valence", "arousal"])
        w.writerows(rows_va)

    short = {e: per[e] for e in (emo_names or []) if per[e] < n_per}
    logger.info(
        "写出 %d 图 → %s；VA 表 → %s；不足 n_per 的类：%s",
        len(rows_va),
        img_out,
        va_out,
        short or "无",
    )
    return len(rows_va)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    default_cache = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--laion--emonet-face-binary"
    )
    parser = argparse.ArgumentParser(description="emonet-face-binary → 均衡子集图 + VA 表")
    parser.add_argument(
        "--parquet-dir", default=default_cache, help="含 train-*.parquet 的目录（默认 HF 缓存）"
    )
    parser.add_argument("--img-out", default="data/emonet/images", help="抽出的图目录")
    parser.add_argument("--va-out", default="data/emonet/emonet_va.csv", help="VA 表输出")
    parser.add_argument("--n-per", type=int, default=50, help="每类抽多少张（均衡）")
    args = parser.parse_args()
    n = build_emonet_dataset(args.parquet_dir, args.img_out, args.va_out, n_per=args.n_per)
    print(f"done, {n} images -> {args.img_out}")


if __name__ == "__main__":
    main()
