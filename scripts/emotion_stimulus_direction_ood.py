"""Emotion-Stimulus 跨源 OOD 方向门（B-4 独立交叉源·2026-07-22·仅验证不训练）。

承接 B-fe 设计门议会裁决（notes/2026-07-21-fear-b-fe-source-selection-council.md·B-4）+
用户 2026-07-22 拍板「先 Emotion-Stimulus 交叉加固再回 B-5」：ISEAR 全 baseline fear LB=0.918
（第二生存源初步成立），加第三独立源交叉验证方向一致性更稳（数学席不强制三源但更稳·fear 敏感）。

**Emotion-Stimulus**（Ghazi, Inkpen & Szpakowicz 2015·research-only·**仅验证不训练·gitignore**）：
句级情绪句 + 情绪 cause 标注（XML tag：`<emotion>text<cause>span<\\cause>text<\\emotion>`）。
两文件：Emotion Cause.txt(820·含 cause) + No Cause.txt(1594·无 cause)·合并跑全 baseline。

⚠ **体裁边界**：Emotion-Stimulus 是**句级混合体裁**（新闻/一般句·非生存叙事·非对话）→ 作
**独立第三方交叉验证源**（不同体裁·验证 fear 方向可靠性跨源一致），**非第三个 survival_narrative
源**（ED/ISEAR 是生存叙事域·此源体裁不同）。cause 标注可定位威胁因（提纯工具·B-2 议会定·本脚本
不提纯）。全 baseline fear LB 解读须结合体裁：句级混合体裁 fear 可能含部分社交 fear。

⚠ **禁解剖标签**（神经席·Didier/Grogans 2026）·**验证≠解锁**（fear 维持默认关·B-5 议会终裁）·
**不做构念提纯**（仅 tag 清洗=数据解析·提纯判据词表属议会 B-2·工程不私拍）。

标签（XML tag·anger→期望 logit>0(+)·fear→期望 logit<0(−)）；full-coverage 分侧 Wilson LB
（fear 交叉门槛参照 0.80·anger 对照 0.70）。权重 motivational_direction_prior_m.pt
（GoEmotions+crowd-enVENT·Emotion-Stimulus 是真 OOD）。合计 anger 483/fear 423（均>数学 n≥185）。
数据 data/raw/emotion_stimulus/Dataset/*.txt（本地 gitignore·仅验证不训练）。

用法：PYTHONPATH=d:/Zero HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CONDA_NO_PLUGINS=true \
  "E:/anaconda/Scripts/conda.exe" run -n affective-expression --no-capture-output \
  python -m scripts.emotion_stimulus_direction_ood
"""

from __future__ import annotations

import argparse
import logging
import re

import torch

from scripts.direction_gate_text_coping import wilson_lower_bound
from scripts.train_direction_head import DirectionHead
from src.agents.models.text_affect_regressor_st import DEFAULT_ENCODER, ST_FEATURE_DIM, encode_texts

logger = logging.getLogger(__name__)

# XML tag 清洗：匹配 <anger> / <\anger>（反斜杠闭合）/ <cause> / <\cause> 等，去 tag 留纯句。
_TAG_RE = re.compile(r"<\\?[a-z]+>")


def _read_emotion_stimulus(paths: list[str]) -> tuple[list[str], list[str]]:
    """读 Emotion-Stimulus：行首 <anger>→'+'、<fear>→'-'；去所有 XML tag 留纯句。其余情绪跳过。"""
    texts: list[str] = []
    pol: list[str] = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if s.startswith("<anger>"):
                    p = "+"
                elif s.startswith("<fear>"):
                    p = "-"
                else:
                    continue
                sent = _TAG_RE.sub("", s).strip()
                if sent:
                    pol.append(p)
                    texts.append(sent)
    return texts, pol


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(
        description="Emotion-Stimulus 跨源 OOD 方向门（B-4 独立交叉源·仅验证）"
    )
    p.add_argument(
        "--files",
        nargs="+",
        default=[
            "data/raw/emotion_stimulus/Dataset/Emotion Cause.txt",
            "data/raw/emotion_stimulus/Dataset/No Cause.txt",
        ],
    )
    p.add_argument("--weights", default="artifacts/motivational_direction_prior_m.pt")
    args = p.parse_args()

    texts, pol = _read_emotion_stimulus(args.files)
    model = DirectionHead(ST_FEATURE_DIM)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logit = model(encode_texts(texts, encoder=DEFAULT_ENCODER)).tolist()

    a_pred = [logit[i] for i in range(len(pol)) if pol[i] == "+"]
    f_pred = [logit[i] for i in range(len(pol)) if pol[i] == "-"]
    a_ok = sum(1 for v in a_pred if v > 0)
    f_ok = sum(1 for v in f_pred if v < 0)
    a_lb = wilson_lower_bound(a_ok, len(a_pred)) if a_pred else 0.0
    f_lb = wilson_lower_bound(f_ok, len(f_pred)) if f_pred else 0.0

    logger.info("─" * 70)
    logger.info("Emotion-Stimulus 跨源 OOD 方向门（句级混合体裁·原生金标·仅验证·B-4 独立交叉源）:")
    logger.info(
        "  anger  %d/%d=%.1f%%  Wilson 下界 %.4f",
        a_ok,
        len(a_pred),
        (a_ok / len(a_pred) * 100) if a_pred else 0.0,
        a_lb,
    )
    logger.info(
        "  fear   %d/%d=%.1f%%  Wilson 下界 %.4f",
        f_ok,
        len(f_pred),
        (f_ok / len(f_pred) * 100) if f_pred else 0.0,
        f_lb,
    )
    logger.info("─" * 70)
    logger.info(
        "⚠ 体裁边界：Emotion-Stimulus=句级混合体裁（新闻/一般句·非生存叙事·非对话）→独立交叉源"
    )
    logger.info(
        "  （验证 fear 方向跨源一致性）·非第三 survival 源。对照生存源 fear：ED 0.90/ISEAR 0.918。"
    )
    logger.info(
        "解读：fear 交叉≥0.80→三源方向一致加固第二源；0.70-0.80→混合体裁部分社交 fear 掺入；"
    )
    logger.info("  <0.70→体裁差异大。fear 解锁终裁属议会门（B-5·验证≠解锁·fear 维持默认关）。")
    f_verdict = (
        "fear≥0.80(交叉一致·加固)"
        if f_lb >= 0.80
        else "fear 0.70-0.80(混合体裁掺入)"
        if f_lb >= 0.70
        else "fear<0.70(体裁差异大)"
    )
    logger.info("Emotion-Stimulus 独立交叉源（仅验证·非解锁裁决·回议会 B-5）：%s", f_verdict)
    print(
        f"EMOSTIM-OOD: anger {a_ok}/{len(a_pred)} lb={a_lb:.3f} | "
        f"fear {f_ok}/{len(f_pred)} lb={f_lb:.3f} => {f_verdict}"
    )


if __name__ == "__main__":
    main()
