"""动作层工具脚本的共用路径（转正自会话 scratchpad，2026-08-07）。

原脚本写死了会话临时目录的绝对路径，转正后会失效——统一从这里取。
产物（盲测载荷/答案/图）落 `data/steering/motion/`（`data/` 已 gitignore，不入库）。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # d:\Zero
MCP_REPO = REPO.parent / "Zero_MCP"

# 产物目录（临时/可重建，不入版本库）
OUT = REPO / "data" / "steering" / "motion"
OUT.mkdir(parents=True, exist_ok=True)

AB_PAYLOAD = OUT / "ab_payload.json"
AB_KEY = OUT / "ab_key.json"  # ⚠ 播放脚本**不得**读它——盲测的前提
LOOP_PAYLOAD = OUT / "loop_payload.json"
MOTION_PAYLOAD = OUT / "motion_payload.json"
PLOT_PNG = OUT / "motion_demo.png"

# VTS 授权 token（首次连接时由 VTube Studio 弹窗授权后写入）
VTS_TOKEN = OUT / "vts_token"

# 数据集
STAYSTILL = REPO / "data" / "staystill" / "freemocap"
REACTIDLE = REPO / "data" / "reactidle"
RAVDESS_MOTION = REPO / "data" / "ravdess_motion"
RAVDESS_AUDIO = REPO / "data" / "ravdess"


def use_zero() -> None:
    """把 Zero 仓放进 sys.path。

    ⚠ 两仓都用 `src.` 作包根，**不能同时进 sys.path**（会互相覆盖）——
    故生成载荷（Zero 侧）与驱动 VTS（Zero_MCP 侧）必须拆成两个进程。
    """
    sys.path.insert(0, str(REPO))


def use_zero_mcp() -> None:
    """把 Zero_MCP 仓放进 sys.path（同上，不可与 use_zero 混用于同一进程）。"""
    sys.path.insert(0, str(MCP_REPO))
