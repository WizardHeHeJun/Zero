"""外部多模态先验流协议 schema（编排层·编排-MCP 协议镜像）。

形状：每条 ExternalPrior = (name, (μv, μa), (Πv, Πa))
  - name:         流标识符（字符串）
  - (μv, μa):     效价/唤醒均值 ∈ [-1, 1]
  - (Πv, Πa):     逐维精度（非负实数；MCP 提供，Zero 校验+防御性覆写）

与 MCP `as_zero_streams()` 输出对齐（M1 逐维精度形状）：
  list[(name, (μv, μa), (Πv, Πa))]
与 affect_core.py:77 的 streams 类型完全一致，可直接 extend。

重要注意事项（议会 design.md M1–M6）：
  - physio 流（前缀 physio/eda/hrv/pupil/scr）：Πv 被 Zero 强制覆写 MIN_PRECISION（M2）；
    EDA/HRV/瞳孔对效价盲（Kreibig 2010），给 valence 精度=注入偏差。
  - MCP 不 import Zero；经本 schema 版本化后跨仓协议由版本号兜底漂移。
  - 跨仓子进程回归断言 EXTERNAL_PRIOR_SCHEMA_VERSION 一致（M5）。
"""

from __future__ import annotations

# 协议 schema 版本（跨仓对接锚点）：MCP as_zero_streams() 与本文件对齐时须同版本（M5）
EXTERNAL_PRIOR_SCHEMA_VERSION: int = 1

# ExternalPrior: (name, (μv, μa), (Πv, Πa))
# 与 affect_core.py streams 类型 list[tuple[str, tuple[float, float], tuple[float, float]]]
# 原生一致，无需引入新机制——直接 extend（M1 议会三席强收敛）。
ExternalPrior = tuple[str, tuple[float, float], tuple[float, float]]


class ExternalPriorError(ValueError):
    """external_priors 载荷违约（M3/M6/M7 校验失败）——**确实指向 MCP 传参**。

    存在的理由是**归责可辨**（议会 2026-07-29 第五轮校验 §四-5）：边界层 `server.py` 原先
    用一个 `except ValueError` 包住**整个** `session.step()`（全图执行），把内核任何位置抛出的
    `ValueError` 一律贴成「external_priors 校验失败（指向 MCP 传参）」。后果是**误导性甩锅**：
    client 照着改传参永远改不好，而活跃会话的 config 不可变（`server.py:230-234`）→ 无法自救，
    表现为 open 成功、每 step 崩。

    继承 `ValueError` 以保持向后兼容（既有 `except ValueError` 仍能捕获），
    但让边界层能把「真的是你传的参不对」与「内核自己出错了」分开报。
    """
