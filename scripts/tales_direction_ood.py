"""Tales-Emotion 跨源 OOD 方向门（B-fe-unlock 前置·第三 survival 体裁源·2026-07-22·仅验证）。

承接 B-5 fear 解锁终裁议会（notes/2026-07-22-fear-b5-unlock-final-council.md·科学门 PASS·翻默认
NOT YET）：心理席指出体裁缺口——两生存源 ED（书面回忆对话叙事）+ ISEAR（第一人称情绪前因问卷）
**同属书面回忆叙事单体裁**（极差 0.018 小正因体裁同质），对 live-chat 真实 survival 触发分布散度
未量化；须第三个**非书面回忆**体裁源补 diversity（对称 anger 三媒介 SemEval/EmoryNLP/DailyDialog）。

**Tales-Emotion**（Alm 2008·童话叙事 Potter/Andersen/Grimm·GPLv3 底本 Gutenberg 公版·
**仅验证不训练·gitignore**）：情绪标注句级金标（4 标注者全同高一致版·数字码 2=anger/3=fear/
4=happy/6=Su+/7=Su−·neutral 已剔）。**体裁最补 diversity**：虚构第三人称文学叙事·视角/时态/
来源机制全正交于 ED/ISEAR 第一人称书面回忆。

⚠ **诚实局限（须标注）**：Tales 是**虚构童话叙事**·补**体裁多样性维度**（打破书面回忆同质）·
**非补真实即时 survival 威胁维度**（虚构≠真实用户 survival 触发）。三角冲突（真实即时威胁×原生
分类金标×license 无单源全占·shortlist §3）下 Tales 是唯一原生标+可下载+license 净的 diversity 源。

⚠ **禁解剖标签**（神经席·Didier/Grogans 2026）·**验证≠解锁**（fear 维持默认关·翻默认=B-fe-unlock
产品定·工程不私翻）·**不做构念提纯**（仅解析·提纯词表属议会 B-2）。

标签（Alm 数字码·2=anger→期望 logit>0(+)·3=fear→期望 logit<0(−)）；full-coverage 分侧 Wilson LB
（fear 门槛参照 0.80·anger 对照 0.70）。fear n≈166/anger n≈218（三册合计·n≥60 最低线上·fear
略低 n≥185 故 CI 偏宽·诚实报告）。权重 motivational_direction_prior_m.pt（Tales 是真 OOD）。
数据 data/raw/tales/{Potter,HCAndersen,Grimms}.txt（本地 gitignore·仅验证不训练）。

用法：PYTHONPATH=d:/Zero HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CONDA_NO_PLUGINS=true \
  "E:/anaconda/Scripts/conda.exe" run -n affective-expression --no-capture-output \
  python -m scripts.tales_direction_ood
"""

from __future__ import annotations

import argparse
import logging

import torch

from scripts.direction_gate_text_coping import wilson_lower_bound
from scripts.train_direction_head import DirectionHead
from src.agents.models.text_affect_regressor_st import DEFAULT_ENCODER, ST_FEATURE_DIM, encode_texts

logger = logging.getLogger(__name__)

_ANGER = "2"  # Alm 数字码：2=angry(+disgust 合并)
_FEAR = "3"  # 3=fearful


def _read_tales(paths: list[str]) -> tuple[list[str], list[str]]:
    """读 Tales All4labsagree：跳 `.agree` 分节行；`SentID@code@sentence` 中 2→'+'、3→'-'。"""
    texts: list[str] = []
    pol: list[str] = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s or s.endswith(".agree"):
                    continue  # 分节行（文件名）
                parts = s.split("@", 2)
                if len(parts) != 3:
                    continue
                code, sent = parts[1], parts[2].strip()
                if code == _ANGER:
                    p = "+"
                elif code == _FEAR:
                    p = "-"
                else:
                    continue
                if sent:
                    pol.append(p)
                    texts.append(sent)
    return texts, pol


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(
        description="Tales-Emotion 跨源 OOD 方向门（B-fe-unlock 前置·第三体裁源·仅验证）"
    )
    p.add_argument(
        "--files",
        nargs="+",
        default=[
            "data/raw/tales/Potter.txt",
            "data/raw/tales/HCAndersen.txt",
            "data/raw/tales/Grimms.txt",
        ],
    )
    p.add_argument("--weights", default="artifacts/motivational_direction_prior_m.pt")
    args = p.parse_args()

    texts, pol = _read_tales(args.files)
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
    logger.info("Tales-Emotion 跨源 OOD 方向门（虚构童话叙事·原生金标·仅验证·第三体裁源）:")
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
        "⚠ 体裁边界：Tales=虚构第三人称童话叙事→补**体裁多样性维度**（打破 ED/ISEAR 书面回忆同质）·"
    )
    logger.info(
        "  **非补真实即时 survival 威胁维度**（虚构≠真实用户 survival 触发·补 diversity 维）。"
    )
    logger.info(
        "解读：fear≥0.80→第三体裁源方向稳·体裁 diversity 补缺口（三体裁书面回忆+问卷+虚构叙事）；"
    )
    logger.info(
        "  0.70-0.80→虚构叙事 fear 略弱；<0.70→虚构体裁 gap 大。B-fe-unlock 翻默认仍属议会/产品定。"
    )
    f_verdict = (
        "fear≥0.80(第三体裁源稳·diversity 补上)"
        if f_lb >= 0.80
        else "fear 0.70-0.80(虚构叙事略弱)"
        if f_lb >= 0.70
        else "fear<0.70(虚构体裁 gap 大)"
    )
    logger.info("Tales 第三体裁源（仅验证·非解锁裁决·回议会 B-fe-unlock）：%s", f_verdict)
    print(
        f"TALES-OOD: anger {a_ok}/{len(a_pred)} lb={a_lb:.3f} | "
        f"fear {f_ok}/{len(f_pred)} lb={f_lb:.3f} => {f_verdict}"
    )


if __name__ == "__main__":
    main()
