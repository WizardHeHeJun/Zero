"""跨仓 `speech_play` 联调验收脚本（speech-output v1，规范§五承诺件）。

固定 wav + 已知口型轨迹，验收跨仓规范的两条行为保证：

1. **同步**：三段 440ms 蜂鸣分别始于 0.0s / 0.8s / 1.6s——录屏逐帧核对「嘴张开」
   与「听到蜂鸣」的对齐（≤80ms，约 ±2 帧 @24fps 录屏）。
2. **嘴部独占**：播放期间蜂鸣间隙嘴应完全闭合（轨迹静音段=0），若被 idle/行为
   注入顶开即违约。

用法（渲染端 Zero_MCP 的 `speech_play` 就绪后）：
    conda run -n affective-expression --no-capture-output python -m scripts.verify_speech_play
    # 只生成 wav+轨迹、不连渲染端（自检脚本本体）：加 --dry-run

可选 env：`ZERO_VTS_MCP_REPO` / `ZERO_VTS_TOKEN_FILE`（同 VtsSink，默认兄弟目录）。
工具未上线时报「渲染端拒收/未知工具」并以退出码 1 结束——这属预期，不是本脚本故障。
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import sys
import tempfile
import wave
from array import array
from pathlib import Path

from src.expression_out.lipsync import MOUTH_PARAMS, mouth_track_from_wav
from src.expression_out.transport import VtsTransport, text_of

FPS = 20.0
RATE = 44100
BEEP_S = 0.44
GAP_S = 0.36
BEEPS = 3  # 蜂鸣起点：0.0 / 0.8 / 1.6 s


def build_test_wav() -> bytes:
    """确定性验收音频：BEEPS 段 220Hz 蜂鸣，段间静音（无随机源，跨机器逐字节一致）。"""
    samples = array("h")
    for _ in range(BEEPS):
        for i in range(int(BEEP_S * RATE)):
            samples.append(int(0.7 * 32000 * math.sin(2 * math.pi * 220.0 * i / RATE)))
        samples.extend([0] * int(GAP_S * RATE))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def describe(track: list[dict[str, object]]) -> str:
    key = MOUTH_PARAMS[0]
    values = [float(f["params"][key]) for f in track]  # type: ignore[index, arg-type]
    opens = sum(1 for v in values if v > 0)
    return f"帧数 {len(track)} · 开口帧 {opens} · 峰值 {max(values):.2f}（键只含 {key}）"


async def run(dry_run: bool) -> int:
    wav_bytes = build_test_wav()
    track = mouth_track_from_wav(wav_bytes, FPS)
    wav_path = Path(tempfile.gettempdir()) / "zero_speech_play_verify.wav"
    wav_path.write_bytes(wav_bytes)
    duration_ms = BEEPS * (BEEP_S + GAP_S) * 1000.0
    print(f"验收 wav：{wav_path}（{duration_ms:.0f}ms，蜂鸣起点 0.0/0.8/1.6s）")
    print(f"口型轨迹：{describe(track)}")
    if dry_run:
        print("--dry-run：不连渲染端，脚本本体自检完成")
        return 0

    transport = VtsTransport()
    if not await transport.connect():
        print("✗ 渲染端连接失败（详见上方日志）——先确认 Zero_MCP/VTS 可用", file=sys.stderr)
        return 1
    try:
        reply = await transport.call_tool(
            "speech_play",
            {"wav_path": str(wav_path), "mouth_track": track, "fps": FPS},
        )
        payload = text_of(reply)
        if getattr(reply, "isError", False):
            print(f"✗ speech_play 拒收（工具未上线属预期）：{payload}", file=sys.stderr)
            return 1
        body = json.loads(payload or "{}")
        print(f"✓ speech_play 已受理：{body}")
        got = float(body.get("duration_ms") or 0.0)
        if abs(got - duration_ms) > 100.0:
            print(
                f"⚠ duration_ms 偏差 {got - duration_ms:+.0f}ms（回包 {got:.0f}）", file=sys.stderr
            )
        wait_s = (got if got > 0 else duration_ms) / 1000.0 + 0.5
        print(f"等待播放结束（{wait_s:.1f}s）…期间请录屏核对：蜂鸣↔张嘴对齐、间隙闭嘴不被顶开")
        await asyncio.sleep(wait_s)
        print("✓ 验收调用完成——同步与独占两条以录屏逐帧核对为准（规范§五）")
        return 0
    finally:
        await transport.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="speech_play 跨仓联调验收")
    parser.add_argument("--dry-run", action="store_true", help="只生成 wav+轨迹，不连渲染端")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.dry_run)))


if __name__ == "__main__":
    main()
