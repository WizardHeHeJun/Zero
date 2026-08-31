"""语音输入：`--chat` 的 push-to-talk 麦克风转写（2026-08-31 计划③，用户拍板「做」）。

## 交互（v1 = push-to-talk，用户手控起止即天然半双工）

```text
你 > ⏎（空回车）→ ● 录音中…再按回车结束 → ⏎ → faster-whisper 本地转写 → 当打字输入用
```

- 转写文本走与打字**完全同构**的输入路径（`ChatDriver.step(text)`），内核零改动；
- 全部阻塞操作（录音等待回车、模型推理）由入口经 `asyncio.to_thread` 调用，
  事件循环不被冻结（裸 `input()` 冻结事件循环的坑见 main.py 注释与 pitfalls）；
- v1 不做 VAD 自动断句（留扩展位：`_capture_until_enter` 换成 VAD 采集器即可，
  换时须补半双工协调——push-to-talk 下用户自己不会在数字人说话时按录音，
  VAD 会，建议届时戴耳机或接入 TtsSpeechSink 忙碌态互斥）。

env：`ZERO_ASR_INPUT=true` 开启（默认关=零回归）；`ZERO_ASR_MODEL` 必填
（faster-whisper 模型名/路径，如 `small`/`large-v3-turbo`——模型选型是部署质量决策，
缺失构造期 fail-fast，不设代码默认）；`ZERO_ASR_DEVICE`（默认 auto）/
`ZERO_ASR_COMPUTE_TYPE`（默认 default）/`ZERO_ASR_LANGUAGE`(默认 zh) 是引擎枚举参数、
非部署差异项，给默认属 A6 例外（对照 `config-only-via-env` 纪律以此处为准）。
依赖走可选 extra `asr`（faster-whisper + sounddevice），门开未装构造期 fail-fast。
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # faster-whisper 期望采样率（协议常数，非配置）


class VoiceInput:
    """push-to-talk 采集 + 本地转写。所有方法**同步阻塞**，须经 `asyncio.to_thread` 调用。

    `model` 可注入替身（tests）；生产由 `build_voice_input` 装配 faster-whisper。
    """

    def __init__(self, *, model: Any, language: str = "zh") -> None:
        self.model = model
        self.language = language

    def record_until_enter(self) -> str:
        """录音到用户再按回车，转写返回文本；空音频/无语音返回空串（调用方按空输入处理）。"""
        audio = self._capture_until_enter()
        if audio is None or len(audio) == 0:
            return ""
        return self.transcribe(audio)

    def _capture_until_enter(self) -> Any:
        """麦克风采集直到回车（硬件层，独立成方法便于将来换 VAD 采集器）。"""
        import numpy as np
        import sounddevice as sd

        chunks: list[Any] = []

        def _on_block(indata: Any, _frames: int, _time: Any, _status: Any) -> None:
            chunks.append(indata.copy())

        # 「录音中…」提示由调用方（入口层）打印——本层只管采集，不带 UI（审查 WARN）。
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=_on_block
        ):
            input()
        if not chunks:
            return None
        return np.concatenate(chunks).reshape(-1)

    def transcribe(self, audio: Any) -> str:
        """float32 单声道 16kHz 波形 → 文本（faster-whisper segments 拼接）。"""
        segments, _info = self.model.transcribe(audio, language=self.language)
        return "".join(segment.text for segment in segments).strip()


def build_voice_input() -> VoiceInput | None:
    """按 env 装配语音输入；`ZERO_ASR_INPUT` 未开则返回 None（默认关=零回归）。

    门开但配置/依赖缺失 ⇒ 构造期 fail-fast（照 build_speech_sink 先例）：
    「开了却按回车没反应」比启动报错更糟。模型在此加载（首次会触发下载，显式可见）。
    """
    raw = os.getenv("ZERO_ASR_INPUT", "false").strip().lower()
    if raw not in ("1", "true", "yes", "on"):
        return None
    model_name = os.getenv("ZERO_ASR_MODEL")
    if not model_name:
        raise RuntimeError(
            "ZERO_ASR_INPUT 已开启但未配 ZERO_ASR_MODEL（faster-whisper 模型名/路径，"
            "如 small / large-v3-turbo）——配置只走 .env，不设代码默认"
        )
    try:
        import numpy  # noqa: F401  # 采集拼接用；随 faster-whisper 传递安装，仍显式探测
        import sounddevice  # noqa: F401  # 探测采集依赖；真正使用在 _capture_until_enter
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "ZERO_ASR_INPUT 已开启但缺依赖——安装可选 extra：`uv sync --extra asr` "
            "或 `pip install -e .[asr]`"
        ) from exc
    device = os.getenv("ZERO_ASR_DEVICE", "auto")
    compute_type = os.getenv("ZERO_ASR_COMPUTE_TYPE", "default")
    logger.info("加载 ASR 模型 %s（device=%s）…首次运行会下载权重", model_name, device)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    return VoiceInput(model=model, language=os.getenv("ZERO_ASR_LANGUAGE", "zh"))
