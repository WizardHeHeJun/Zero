"""WESAD 生理 DataLoader：condition→(v,a)，Y←胸带信号特征(HR/EDA/Temp)。

获取数据（开放下载，需同意条款）：
  1. 下载 https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection
  2. 解压到本地，例如 `data/wesad/`（已 gitignore）。目录下是 Sxx/Sxx.pkl。
  3. 训练：python -m scripts.train_physiology --root data/wesad

WESAD 胸带 (RespiBAN) 采样率 700Hz；label：1 baseline / 2 stress / 3 amusement / 4 meditation
（0 及 5~7 为过渡/忽略）。pkl 为 Python2 序列化，用 latin1 读。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

from src.agents.affect_math import clamp

CHEST_FS = 700  # 胸带采样率 Hz

# condition → (valence, arousal)，约定取值 [-1, 1]（近似映射）
CONDITION_TO_VA: dict[int, tuple[float, float]] = {
    1: (0.0, 0.0),  # baseline
    2: (-0.6, 0.7),  # stress
    3: (0.6, 0.5),  # amusement
    4: (0.3, -0.6),  # meditation
}


def estimate_heart_rate(ecg: np.ndarray, *, fs: int = CHEST_FS) -> float:
    """从 ECG 段经 R 波检测估计心率（bpm）；不足两拍返回 0。"""
    signal = np.asarray(ecg, dtype=float).reshape(-1)
    std = float(signal.std()) or 1.0
    peaks, _ = find_peaks((signal - signal.mean()) / std, height=1.0, distance=int(0.4 * fs))
    if peaks.size < 2:
        return 0.0
    rr_seconds = np.diff(peaks) / fs
    return float(60.0 / np.mean(rr_seconds))


def extract_physiology(
    ecg: np.ndarray, eda: np.ndarray, temp: np.ndarray, *, fs: int = CHEST_FS
) -> list[float]:
    """从一段胸带信号提取归一化生理特征 [hr, eda, temp]，均落在 [0,1]。"""
    hr = estimate_heart_rate(ecg, fs=fs)
    eda_mean = float(np.mean(eda))
    temp_mean = float(np.mean(temp))
    return [
        clamp((hr - 50.0) / 70.0, 0.0, 1.0),
        clamp(eda_mean / 20.0, 0.0, 1.0),
        clamp((temp_mean - 30.0) / 10.0, 0.0, 1.0),
    ]


def load_wesad(
    root: str | Path, *, window_seconds: int = 30, limit: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """递归加载 WESAD pkl，按 condition 切窗，返回 (X=(v,a), Y=生理3维) float32 张量。

    X ∈ [-1,1]^2，Y ∈ [0,1]^3。无可用段时抛 FileNotFoundError。
    """
    files = sorted(Path(root).rglob("*.pkl"))
    if limit is not None:
        files = files[:limit]

    win = int(window_seconds * CHEST_FS)
    xs: list[list[float]] = []
    ys: list[list[float]] = []
    for path in files:
        with open(path, "rb") as fh:
            data = pickle.load(fh, encoding="latin1")
        chest = data["signal"]["chest"]
        ecg = np.asarray(chest["ECG"]).reshape(-1)
        eda = np.asarray(chest["EDA"]).reshape(-1)
        temp = np.asarray(chest["Temp"]).reshape(-1)
        labels = np.asarray(data["label"]).reshape(-1)

        for condition, va in CONDITION_TO_VA.items():
            idx = np.where(labels == condition)[0]
            if idx.size < win:
                continue
            for start in range(0, idx.size - win + 1, win):
                sel = idx[start : start + win]
                xs.append(list(va))
                ys.append(extract_physiology(ecg[sel], eda[sel], temp[sel]))

    if not xs:
        raise FileNotFoundError(f"未在 {root} 找到可用 WESAD 段（检查 window_seconds 是否过大）")
    return (
        torch.tensor(xs, dtype=torch.float32),
        torch.tensor(ys, dtype=torch.float32),
    )
