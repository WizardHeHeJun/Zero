"""RAVDESS 语音韵律 DataLoader：(v,a) ← 文件名情绪标签，Y ← librosa 提取的真实韵律特征。

获取数据（无 EULA，需 Kaggle 账号）：
  1. 下载 https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
  2. 解压到本地，例如 `data/ravdess/`（已 gitignore）。目录下是 Actor_xx/*.wav。
  3. 训练：python -m scripts.train_prosody --root data/ravdess

RAVDESS 文件名：modality-vocalChannel-emotion-intensity-statement-repetition-actor.wav
第 3 段为情绪码：01 中性 02 平静 03 高兴 04 悲伤 05 愤怒 06 恐惧 07 厌恶 08 惊讶。
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch

from src.agents.affect_math import clamp

# 情绪码 → (valence, arousal)，约定取值 [-1, 1]（近似映射）
EMOTION_CODE_TO_VA: dict[str, tuple[float, float]] = {
    "01": (0.0, 0.0),  # neutral
    "02": (0.3, -0.5),  # calm
    "03": (0.7, 0.5),  # happy
    "04": (-0.6, -0.4),  # sad
    "05": (-0.6, 0.7),  # angry
    "06": (-0.5, 0.6),  # fearful
    "07": (-0.6, 0.2),  # disgust
    "08": (0.3, 0.6),  # surprised
}


# TODO(D-dim): RAVDESS 无逐样本连续 D（Dominance）标注。
# 当前仅用文件名离散情绪码映射 (v,a)；若要扩展 D 维监督，需数据采集 + 研究决策：
#   - 离散情绪码 → D 的映射须经议会操作化对齐（见议会 notes 2026-07-13 #2）；
#   - RAVDESS 本身未提供连续效价/唤醒/优势度评分，扩 D 属数据采集阻塞；
#   - 在此之前不要把 D 列加入 load_ravdess 的输出张量。
def parse_emotion_code(filename: str) -> str | None:
    """从 RAVDESS 文件名解析情绪码；非法格式返回 None。"""
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    code = parts[2]
    return code if code in EMOTION_CODE_TO_VA else None


def extract_prosody(path: str | Path, *, sr: int = 16000) -> list[float]:
    """从单条音频提取归一化韵律特征 [speech_rate, pitch, energy]，均落在 [0,1]。"""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    f0, _, _ = librosa.pyin(y, fmin=65.0, fmax=400.0, sr=sr)
    f0_mean = float(np.nanmean(f0)) if np.any(~np.isnan(f0)) else 0.0
    rms_mean = float(np.mean(librosa.feature.rms(y=y)))
    zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    speech_rate = clamp((zcr_mean - 0.02) / 0.18, 0.0, 1.0)  # 过零率作发音速率代理
    pitch = clamp((f0_mean - 80.0) / (400.0 - 80.0), 0.0, 1.0)
    energy = clamp(rms_mean * 5.0, 0.0, 1.0)
    return [speech_rate, pitch, energy]


def load_ravdess(
    root: str | Path, *, limit: int | None = None, sr: int = 16000
) -> tuple[torch.Tensor, torch.Tensor]:
    """递归加载 RAVDESS，返回 (X=(v,a), Y=韵律3维) float32 张量。

    X ∈ [-1,1]^2，Y ∈ [0,1]^3。无可解析 wav 时抛 FileNotFoundError。
    """
    files = sorted(Path(root).rglob("*.wav"))
    if limit is not None:
        files = files[:limit]

    xs: list[list[float]] = []
    ys: list[list[float]] = []
    for path in files:
        code = parse_emotion_code(path.name)
        if code is None:
            continue
        xs.append(list(EMOTION_CODE_TO_VA[code]))
        ys.append(extract_prosody(path, sr=sr))

    if not xs:
        raise FileNotFoundError(f"未在 {root} 找到可解析的 RAVDESS wav")
    return (
        torch.tensor(xs, dtype=torch.float32),
        torch.tensor(ys, dtype=torch.float32),
    )
