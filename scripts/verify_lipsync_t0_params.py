"""T0 探针：真机核验圆唇维 VTS 输入参数真名（PRP/lipsync-v2 design.md M6/T0）。

连渲染端（经 `VtsTransport`，与皮套/语音同一条链路）→ `params_list` → 过滤嘴部相关
输入参数，打印「参数名 + 取值范围 + 默认值」供定夺 `ZERO_TTS_MOUTH_FORM_PARAM`。

⚠ 命名空间是 **VTS 输入参数**（8-14 教训：`MouthOpen` 而非 Live2D `ParamMouthOpenY`）
——本脚本列的就是输入参数面，照抄进 .env 即可。

用法（VTube Studio 须开着、API 已授权）：
  conda run -n affective-expression python -m scripts.verify_lipsync_t0_params
token 不在默认位时：设 ZERO_VTS_TOKEN_FILE（与 --chat 同一份）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from src.expression_out.transport import VtsTransport, text_of

# 嘴部候选关键词（小写包含匹配；宁多列人筛，勿漏）
_MOUTH_RE = re.compile(r"mouth|lip|pucker|smile|form", re.IGNORECASE)


def _iter_param_dicts(data: Any) -> list[dict[str, Any]]:
    """从未知形状的回包里挖参数字典列表（对面回包形状不锚死，宁全打印勿猜错）。"""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("parameters", "params", "items", "list"):
            inner = data.get(key)
            if isinstance(inner, list):
                return [x for x in inner if isinstance(x, dict)]
    return []


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    transport = VtsTransport()
    if not await transport.connect():
        print("❌ 渲染端/VTS 连接失败——VTube Studio 开着吗？（详见上方 warning）")
        return 1
    try:
        reply = await transport.call_tool("params_list", {})
        body = text_of(reply)
        if getattr(reply, "isError", False):
            print(f"❌ params_list 被拒：{body}")
            return 1
        try:
            data = json.loads(body or "{}")
        except json.JSONDecodeError:
            print(body)
            return 0
        params = _iter_param_dicts(data)
        if not params:
            print("⚠ 回包里没挖到参数列表，原文如下（人工看形状）：")
            print(body)
            return 0
        hits = [p for p in params if _MOUTH_RE.search(str(p.get("name", "")))]
        print(f"共 {len(params)} 个输入参数；嘴部相关 {len(hits)} 个：\n")
        for p in hits:
            print(
                f"  {p.get('name')}  范围 [{p.get('min')}, {p.get('max')}]"
                f"  默认 {p.get('default', p.get('defaultValue'))}  当前 {p.get('value')}"
            )
        print(
            "\n→ 圆唇维选型：挑「张合之外、控制嘴形圆/扁」的那个（常见 MouthForm/"
            "MouthPucker/MouthSmile，随皮套而异）；定名后填 .env 的 "
            "ZERO_TTS_MOUTH_FORM_PARAM，并按 PRP/lipsync-v2/tasks.md 顶部提醒"
            "把真名并入 lipsync.MOUTH_PARAMS 上界 + 复查融合锚点②。"
        )
        return 0
    finally:
        await transport.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
