"""构建 EmoBank writer-perspective 训练数据（P-block · 议会 2026-07-15 修正2）。

议会裁定「文本→coping_potential」监督须选 **writer perspective** D（非 reader，
非双视角聚合），因 writer 视角更近文本产出者的自评状态、更近 control appraisal 本质。
仓库自带的 `data/emobank.csv`（列 id,split,V,A,D,text）是**双视角加权聚合**、无法分视角。

本脚本把 EmoBank 官方 `corpus/writer.csv`（writer 视角逐句 V/A/D，无 text）按 id 内连接
到 `data/emobank.csv` 的 split/text 上，产出 `data/emobank_writer.csv`（列同 emobank.csv：
id,split,V,A,D,text，但 V/A/D 为 **writer 视角**）。

关键：产出列名与 emobank.csv 一致 → `read_emobank_rows(include_d=True)` **无需改代码**即可读，
「选 writer perspective」靠喂不同 CSV 实现，不靠改 reader（降 churn）。

用法：
  1. 取官方 writer.csv（github.com/JULIELab/EmoBank 的 corpus/writer.csv）：
     curl -sL <raw.githubusercontent.com 的该文件 URL> -o <writer_raw>
  2. python -m scripts.build_emobank_writer --writer <writer_raw> --emobank data/emobank.csv \
       --out data/emobank_writer.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)


def _read_writer_vad(path: str | Path) -> dict[str, tuple[str, str, str]]:
    """读 EmoBank writer.csv，返回 {id: (V, A, D)}（字符串原样，1–5 量表）。"""
    out: dict[str, tuple[str, str, str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["id"]] = (row["V"], row["A"], row["D"])
    if not out:
        raise ValueError(f"{path} 无 writer VAD 数据行（需含 id,V,A,D 列）")
    return out


def _pearson(xs: list[float], ys: list[float]) -> float:
    """样本皮尔逊相关系数（供标注层 r(V,D) 参照，不依赖 numpy）。"""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(sxx * syy)
    return sxy / denom if denom > 0 else float("nan")


def build(writer_path: str, emobank_path: str, out_path: str) -> tuple[int, int, float]:
    """内连接 writer VAD 与 emobank split/text，写 out。返回 (匹配数, 未匹配数, r(V,D))。"""
    writer_vad = _read_writer_vad(writer_path)

    matched = 0
    unmatched = 0
    vs: list[float] = []
    ds: list[float] = []
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(emobank_path, newline="", encoding="utf-8") as fin,
        open(out, "w", newline="", encoding="utf-8") as fout,
    ):
        writer_out = csv.DictWriter(fout, fieldnames=["id", "split", "V", "A", "D", "text"])
        writer_out.writeheader()
        for row in csv.DictReader(fin):
            wid = row["id"]
            if wid not in writer_vad:
                unmatched += 1
                continue
            wv, wa, wd = writer_vad[wid]
            writer_out.writerow(
                {"id": wid, "split": row["split"], "V": wv, "A": wa, "D": wd, "text": row["text"]}
            )
            vs.append(float(wv))
            ds.append(float(wd))
            matched += 1

    if matched == 0:
        raise ValueError("writer 与 emobank 无 id 交集——检查两文件是否同一 EmoBank 版本")
    r_vd = _pearson(vs, ds)
    return matched, unmatched, r_vd


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="构建 EmoBank writer-perspective 训练数据（P-block）"
    )
    parser.add_argument("--writer", required=True, help="EmoBank 官方 corpus/writer.csv 路径")
    parser.add_argument(
        "--emobank", default="data/emobank.csv", help="仓库 emobank.csv（取 split/text）"
    )
    parser.add_argument(
        "--out", default="data/emobank_writer.csv", help="输出 writer-perspective CSV"
    )
    args = parser.parse_args()

    matched, unmatched, r_vd = build(args.writer, args.emobank, args.out)
    logger.info("写出 %s：matched=%d, emobank 侧未匹配=%d", args.out, matched, unmatched)
    logger.info("writer 视角标注层 r(V,D)=%.4f（对照聚合层 0.38；供独立性 GATE 参照）", r_vd)
    print(f"done: {args.out} matched={matched} r(V,D)_writer={r_vd:.4f}")


if __name__ == "__main__":
    main()
