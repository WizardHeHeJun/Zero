"""语音表现：把每轮回复合成为语音，交渲染端播放并同步口型。

## 数据流（v1，跨仓规范 `notes/2026-08-14-cross-repo-speech-play-spec.md`）

```text
emit(frame) ──投队列即返── worker（串行）:
    strip 舞台说明 → HTTP 调本地 Bert-VITS2 → wav
    → lipsync 能量包络 → 口型关键帧（只含 MOUTH_PARAMS，锚点钉死）
    → transport.call_tool("speech_play", {wav_path, mouth_track, fps})
    → 按回包 duration_ms 节流（播完才取下一条，不打断在播语音）
```

- **emit 尽快返回**：整句合成耗时数秒，全部搬进后台 worker；对话轮次只付一次
  `queue.put_nowait` 的代价。
- **播放在渲染端**（用户 2026-08-14 拍板）：音频与口型在对方单侧时钟内对齐，
  我方不碰声卡；`speech_play` 未上线/失败时静默降级只记日志（表现层契约）。
  失败码（对方 2026-08-14 回执定稿八令牌，含队满 `[vtsb:throttled]`=稍后可原样重试）
  ——我方按对方明示**统一静默降级不特殊化**，令牌随载荷原文进 warning 日志可辨识；
  将来若加重试逻辑，只对 `throttled` 开。
- **舞台说明不出声**：合成前复用 `strip_stage_directions_with_segments` 取纯文本，
  「（笑）」类括注不会被朗读（与 `VtsSink._intents`、factual 模式同源机械层）。
- **韵律流留口**（PRD G4）：每句合成后 `prosody_frames()` 可取能量帧序列，供将来
  动作层说话分支消费（议会裁定④：说话分支只接实时 TTS 韵律流）；本期无消费者。

env：`ZERO_TTS_SINK=true` 开启（默认关=零回归）；`ZERO_TTS_SERVER_URL`/`ZERO_TTS_SPEAKER`
必填（配置只走 .env，缺失构造期 fail-fast，不设代码默认）；`ZERO_TTS_LANGUAGE`（默认 ZH）
与 `ZERO_TTS_MODEL_ID`（默认 0）是 **PRP A6 拍板的例外**：二者是 hiyoriUI 协议枚举/序号、
非部署差异项，给默认不属「猜配置」——对照 `config-only-via-env` 纪律时以此处为准。
HTTP 依赖 `httpx` 走可选 extra `tts`——门开未装同样构造期 fail-fast（静默没声音比报错糟）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agents.language_openai import strip_stage_directions_with_segments
from src.expression_out.base import ExpressionFrame
from src.expression_out.lipsync import energy_envelope, envelope_to_mouth_track
from src.expression_out.transport import VtsTransport, text_of

logger = logging.getLogger(__name__)

DEFAULT_FPS = 20.0  # 与连续动作流同节奏（motion_synth DEFAULT_FPS）
SYNTH_TIMEOUT_S = 60.0  # 整句合成的 HTTP 超时：GPU 数秒量级，留一个数量级余量

# 按句切分（ZERO_TTS_SENTENCE_SPLIT，2026-08-31 计划④）：短句合并下限。
# 12 字 ≈ 1.5-2s 音频——再短则合成开销占比过高、句间颗粒停顿明显；非心理学量，
# 纯工程取值，观感不对可调（与 lipsync 常数同款调参纪律：改前留注释轨迹）。
SENTENCE_MIN_CHARS = 12
_SENTENCE_TERMINATORS = "。！？!?…"
# 括号/引号保护：内部的终止符不切（「他说“走。”然后…」不应从引号中间断开）。
_BRACKET_CLOSERS = {"（": "）", "(": ")", "「": "」", "『": "』", "“": "”", "《": "》"}


def _is_only_terminators(segment: str) -> bool:
    """段落是否只含终止符/空白（「……」的第二个 … 这类残段，应并回上一段）。"""
    return all(ch in _SENTENCE_TERMINATORS or ch.isspace() for ch in segment)


def split_sentences(text: str, min_chars: int = SENTENCE_MIN_CHARS) -> list[str]:
    """把整段回复切成可逐句合成的段落（纯函数；`ZERO_TTS_SENTENCE_SPLIT` 门开才被消费）。

    规则：
    - 在**括号/引号深度为 0** 的 `。！？!?…`（及换行）处切分，终止符归前段；
      连续终止符（「……」「！？」）合并进同一段。
    - 短段贪心**向后合并**到 ≥ min_chars 字（避免「嗯。」级碎片的合成开销与颗粒停顿）；
      收尾残段不足 min_chars 时并回前一段。
    - 已知不切（有意）：英文句点 `.`（小数/URL/缩写误伤率高，中文对话主链路用不上）；
      分号 `；`（切了句间韵律更碎）。已知误伤：未闭合括号会使其后整段不再切分
      （深度回不到 0）——宁不切勿错切，退化为整段合成，无正确性损失。
    """
    segments: list[str] = []
    buf: list[str] = []
    stack: list[str] = []
    for ch in text:
        closer = _BRACKET_CLOSERS.get(ch)
        if closer is not None:
            stack.append(closer)
            buf.append(ch)
            continue
        if stack and ch == stack[-1]:
            stack.pop()
            buf.append(ch)
            continue
        buf.append(ch)
        if not stack and (ch in _SENTENCE_TERMINATORS or ch == "\n"):
            segment = "".join(buf).strip()
            buf = []
            if not segment:
                continue
            if segments and _is_only_terminators(segment):
                segments[-1] += segment
            else:
                segments.append(segment)
    tail = "".join(buf).strip()
    if tail:
        if segments and _is_only_terminators(tail):
            segments[-1] += tail
        else:
            segments.append(tail)
    # 短段贪心向后合并
    merged: list[str] = []
    for segment in segments:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] += segment
        else:
            merged.append(segment)
    if len(merged) >= 2 and len(merged[-1]) < min_chars:
        merged[-2] += merged[-1]
        merged.pop()
    return merged


# 「整句均为括号段」判定（无嵌套）：配合 strip 的「全剥空回退原文」不变式用，见 _speak。
# 蕴含论证（text-predicate-admission）：strip 后文本仍整体为括号段 ⇒ 本轮无括号外正文可读
# ——判据是**结构**（括号包裹）非语义词表，命中即「没有裸露正文」为真，蕴含成立；
# 由它推出「跳过朗读」则是产品选择：括号内容读出来必然带出括号语气词问题。
# 已知误伤（KNOWN_MISS，测试钉住）：「（好的）」类全括号普通答话也被跳过——对语音这是
# 可接受方向（读出括号更怪）。已知漏报：嵌套括号「（他说（笑））」不命中 ⇒ 会带括号朗读。
_PURE_BRACKET_RE = re.compile(r"(?:\s*[（(][^（）()]*[）)])+\s*")


@dataclass(frozen=True)
class ProsodyFrame:
    """一帧韵律信息（动作层说话分支的预留消费单元）。

    Attributes:
        t_ms: 自音频首采样起算的时刻（与口型轨迹同一时基）。
        energy: RMS 能量 ∈ [0, 1]（v1 唯一维度；v2 扩 f0 等须过议会门）。
    """

    t_ms: int
    energy: float


class TtsSpeechSink:
    """把 `ExpressionFrame.reply` 表现成语音（渲染端播放）。实现 `ExpressionSink` 协议。

    未 `connect()` 或连接失败时全程 no-op；worker 内任何失败只记日志——
    表现端故障不扳倒对话。连接经注入的 `VtsTransport`（与皮套 sink 共享同一条）。
    """

    def __init__(
        self,
        *,
        transport: VtsTransport,
        server_url: str,
        speaker: str,
        language: str = "ZH",
        model_id: int = 0,
        fps: float = DEFAULT_FPS,
        wav_dir: Path | None = None,
        sentence_split: bool = False,
    ) -> None:
        self.transport = transport
        self.server_url = server_url
        self.speaker = speaker
        self.language = language
        self.model_id = model_id
        self.fps = fps
        # 按句切分流水（默认关=零回归：整句合成逐字旧行为）。开=首包延迟≈首段合成时长。
        self.sentence_split = sentence_split
        self.wav_dir = wav_dir or Path(tempfile.gettempdir()) / "zero_tts"
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.worker_task: asyncio.Task[None] | None = None
        self.last_prosody: list[ProsodyFrame] = []
        self.utterance_seq = 0
        self.last_wav: Path | None = None
        self.http_transport: Any = None  # 测试注入 httpx.MockTransport；None=真网络

    async def connect(self) -> bool:
        """经共享 transport 连渲染端并起后台合成 worker；失败返回 False（不抛）。"""
        if not await self.transport.connect():
            return False
        self.worker_task = asyncio.create_task(self._worker())
        return True

    async def emit(self, frame: ExpressionFrame) -> None:
        """投队列即返（协议要求 emit 尽快返回；合成/播放全在 worker 串行消化）。"""
        if self.worker_task is None or not frame.reply:
            return
        self.queue.put_nowait(frame.reply)
        backlog = self.queue.qsize()
        if backlog > 1:
            logger.debug("语音队列积压 %d 句（串行合成播放，按序消化）", backlog)

    async def aclose(self) -> None:
        """停 worker、清最后一句 wav、断连接（幂等）。

        ⚠ 已知良性竞态（code-reviewer 2026-08-14）：transport 与皮套共享且皮套先
        aclose 时，worker 若正 await `speech_play` 会撞上 `call_tool` 的 RuntimeError
        ——被 `_worker` 的兜底捕获只记 warning，不影响收尾；语音自身 aclose 先 cancel
        worker，无此窗口。
        """
        if self.worker_task is not None:
            self.worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.worker_task
            self.worker_task = None
        if self.last_wav is not None:
            with contextlib.suppress(OSError):
                self.last_wav.unlink()
            self.last_wav = None
        await self.transport.aclose()

    def prosody_frames(self) -> list[ProsodyFrame]:
        """最近一句已合成语音的韵律帧（副本）。本期无消费者，契约测试钉形状。"""
        return list(self.last_prosody)

    async def _worker(self) -> None:
        """串行消化队列：一句失败不影响下一句，更不影响对话。"""
        while True:
            reply = await self.queue.get()
            try:
                await self._speak(reply)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("语音表现失败（对话不受影响）：%s", exc)

    async def _speak(self, reply: str) -> None:
        text, _ = strip_stage_directions_with_segments(reply)
        text = text.strip()
        # ⚠ strip 的不变式是「全剥空回退原文」（文本回复绝不产出空串）——语音侧语义相反：
        # 全括注轮次（「（点了点头）」）该**闭嘴**而不是把括注朗读出来。回退情形与
        # 「本来就没有括注」在返回值上不可分（两者都 cleaned==原文、segments==[]），
        # 故这里独立判「整句均为括号段」跳过。代价是「（好的）」这类全括号短语也不出声
        # ——对语音这是可接受方向（读括号本身更怪）。
        if not text or _PURE_BRACKET_RE.fullmatch(text):
            logger.debug("语音：本轮无可朗读正文（空/全括注），跳过合成")
            return
        segments = split_sentences(text) if self.sentence_split else [text]
        if len(segments) <= 1:
            wav_bytes, synth_s = await self._synth_timed(text)
            await self._deliver(text, wav_bytes, synth_s)
            return
        # 按句流水（门开且多段）：预取一段——合成第 i+1 段与播放第 i 段重叠，
        # 首包延迟 ≈ 首段合成时长；wav 落盘仍发生在上一段播完（sleep）之后，
        # 「串行 ⇒ 删上一句文件必已播完」的 W5 不变式保持不变。
        logger.debug("语音按句切分：%d 字 → %d 段", len(text), len(segments))
        synth_task: asyncio.Task[tuple[bytes, float]] = asyncio.create_task(
            self._synth_timed(segments[0])
        )
        try:
            for i, segment in enumerate(segments):
                try:
                    synth_result: tuple[bytes, float] | None = await synth_task
                except Exception as exc:
                    logger.warning("语音分段合成失败（跳过该段，后续继续）：%s", exc)
                    synth_result = None
                if i + 1 < len(segments):
                    synth_task = asyncio.create_task(self._synth_timed(segments[i + 1]))
                if synth_result is not None:
                    await self._deliver(segment, synth_result[0], synth_result[1])
        finally:
            # 不丢弃在飞的预取句柄（python-code：不裸 create_task 丢句柄）：
            # 中途异常退出时取消并等待；已完成未消费的取回异常防「never retrieved」告警。
            if not synth_task.done():
                synth_task.cancel()
                # 一并吞普通 Exception（审查 WARN 2026-08-31）：cancel 生效前任务恰好以真实
                # 异常完成的 TOCTOU 窄窗里，await 会把该异常在 finally 中抛出、顶替正在传播的
                # 外层 CancelledError ⇒ _worker 的取消分支被绕过、aclose 挂住。清理路径故意宽捕获。
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await synth_task
            elif not synth_task.cancelled():
                with contextlib.suppress(Exception):
                    synth_task.exception()

    async def _synth_timed(self, text: str) -> tuple[bytes, float]:
        """合成一段并计时（供整句路径与分段流水共用）。"""
        started = time.monotonic()
        wav_bytes = await self._synthesize(text)
        return wav_bytes, time.monotonic() - started

    async def _deliver(self, text: str, wav_bytes: bytes, synth_s: float) -> None:
        """一段音频的投递全链：包络→口型→落盘→speech_play→按时长节流。

        整句路径（sentence_split 关）下与 v1 行为逐字一致；分段路径下逐段调用，
        `last_prosody` 为**最后一段**的韵律帧（本期无消费者，split 语义在此成文）。
        """
        envelope = energy_envelope(wav_bytes, self.fps)
        logger.debug(
            "TTS 合成完成：%d 字 → %.1fs 音频，耗时 %.2fs（wav %.0f KB）",
            len(text),
            len(envelope) / self.fps,
            synth_s,
            len(wav_bytes) / 1024,
        )
        frame_ms = 1000.0 / self.fps
        self.last_prosody = [
            ProsodyFrame(t_ms=int(round(i * frame_ms)), energy=value)
            for i, value in enumerate(envelope)
        ]
        track = envelope_to_mouth_track(envelope, self.fps)
        wav_path = self._write_wav(wav_bytes)
        result = await self.transport.call_tool(
            "speech_play",
            {"wav_path": str(wav_path), "mouth_track": track, "fps": self.fps},
        )
        if getattr(result, "isError", False):
            logger.warning("speech_play 被拒（对话不受影响）：%s", text_of(result))
            return
        body = json.loads(text_of(result) or "{}")
        duration_ms = float(body.get("duration_ms") or 0.0)
        if duration_ms <= 0:
            # 回包缺/异常 duration 时用本地已知音频时长兜底——完全不节流会让下一句的
            # `_write_wav` 在播放未结束前删掉在播文件（code-reviewer W5）。
            duration_ms = len(envelope) * 1000.0 / self.fps
        logger.debug(
            "speech_play 已受理：时长 %.0f ms · 口型帧 %d（播完再取下一句）",
            duration_ms,
            len(track),
        )
        if duration_ms > 0:
            # 串行节流（跨仓规范§二并发语义）：播完再取下一条 ⇒ 对方正常收不到重叠调用。
            await asyncio.sleep(duration_ms / 1000.0)

    async def _synthesize(self, text: str) -> bytes:
        """HTTP 调本地 Bert-VITS2 拿整句 wav。

        请求形状集中在此一处：不同 fork 的 server 接口略有差异，部署期（T8）若需
        适配只改这里。延迟 import httpx：无 tts extra 的环境仍可 import 本模块
        （fake 单测走不到这）。
        """
        import httpx

        # hiyoriUI /voice 契约（部署实核 2026-08-14）：model_id 必填、说话人键叫 speaker_name；
        # 成功回 wav 字节，业务错误回 200+JSON({status,detail}) ⇒ 不能只看 HTTP 码。
        params: dict[str, Any] = {
            "text": text,
            "model_id": self.model_id,
            "speaker_name": self.speaker,
            "language": self.language,
        }
        async with httpx.AsyncClient(
            timeout=SYNTH_TIMEOUT_S, transport=self.http_transport
        ) as client:
            resp = await client.get(self.server_url, params=params)
            resp.raise_for_status()
            body = bytes(resp.content)
        if not body.startswith(b"RIFF"):
            snippet = body[:200].decode("utf-8", errors="replace")
            raise RuntimeError(f"TTS 服务未返回 wav（业务错误直传）：{snippet}")
        return body

    def _write_wav(self, wav_bytes: bytes) -> Path:
        """wav 落盘给渲染端读（同机路径契约）；顺手清上一句的文件（串行 ⇒ 必已播完）。"""
        self.wav_dir.mkdir(parents=True, exist_ok=True)
        if self.last_wav is not None:
            with contextlib.suppress(OSError):
                self.last_wav.unlink()
        self.utterance_seq += 1
        path = self.wav_dir / f"utt_{os.getpid()}_{self.utterance_seq:05d}.wav"
        path.write_bytes(wav_bytes)
        self.last_wav = path
        return path


def build_speech_sink(transport: VtsTransport | None = None) -> TtsSpeechSink | None:
    """按 env 装配语音 sink；`ZERO_TTS_SINK` 未开则返回 None（默认关=零回归）。

    门开但配置/依赖缺失 ⇒ **构造期 fail-fast**（照 AsyncPostgresSaver 先例）：
    部署上"开了却悄悄没声音"比启动报错更糟，表现层的静默降级只给**运行时**故障。
    """
    raw = os.getenv("ZERO_TTS_SINK", "false").strip().lower()
    if raw not in ("1", "true", "yes", "on"):
        return None
    server_url = os.getenv("ZERO_TTS_SERVER_URL")
    if not server_url:
        raise RuntimeError(
            "ZERO_TTS_SINK 已开启但未配 ZERO_TTS_SERVER_URL（本地 Bert-VITS2 服务地址）"
            "——配置只走 .env，不设代码默认"
        )
    speaker = os.getenv("ZERO_TTS_SPEAKER")
    if not speaker:
        raise RuntimeError(
            "ZERO_TTS_SINK 已开启但未配 ZERO_TTS_SPEAKER（说话人名，取决于所装底模）"
            "——配置只走 .env，不设代码默认"
        )
    try:
        import httpx  # noqa: F401  # 只探测可用性；真正使用在 _synthesize
    except ImportError as exc:
        raise RuntimeError(
            "ZERO_TTS_SINK 已开启但未安装 httpx——安装可选 extra：`uv sync --extra tts` "
            "或 `pip install -e .[tts]`"
        ) from exc
    sentence_split = os.getenv("ZERO_TTS_SENTENCE_SPLIT", "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    return TtsSpeechSink(
        transport=transport or VtsTransport(),
        server_url=server_url,
        speaker=speaker,
        language=os.getenv("ZERO_TTS_LANGUAGE", "ZH"),
        model_id=int(os.getenv("ZERO_TTS_MODEL_ID", "0")),
        sentence_split=sentence_split,
    )
