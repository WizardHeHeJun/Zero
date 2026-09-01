"""口型合成 v2：音素级同步（纯函数、无 I/O、torch-free、确定性）。

设计权威 `PRP/lipsync-v2/design.md`（M1-M9），任务清单 `PRP/lipsync-v2/tasks.md`（T2）。
v1（`lipsync.py` 的能量包络口型）零改动——本文件不导入其私有符号，只借公开的
`smooth_envelope`（架构决策 #3：`_percentile` 自实现，重复优于跨文件耦合私有工具）。

## 数据来源

Bert-VITS2 补丁（`tools/tts_patch/`）在 `/voice` 响应头 `X-Phoneme-Durations` 里带回
`{"phones": [...], "durations": [帧数...]}`——`durations` 单位是模型侧的帧数（w_ceil，
hop_length=512 架构假设），非毫秒；本模块负责换算、校验、重排、合成。

## 管线（tasks.md T2 定的顺序，不重排）

```
resolve_phoneme_durations（帧数×frame_ms 换算 + M8 容差校验/等比缩放）
  → build_phoneme_keyframes：
      τ 仿射重排（M1，Σ 不变）→ 音素边界 t_ms（floor，M4）
      → 主元音查表 + b/p/m 闭唇 + 其余声母目标=后接韵母（M2）
      → SLEW_RATE 逐帧限幅（M5，作用在音素形状轨迹，非最终乘积）
      → 关键帧时间点插值包络（smooth_envelope + 自算百分位定标，不做 v1 对比度拉伸）
      → M3 能量混合（地板只施加在包络因子）
```
"""

from __future__ import annotations

import math

from src.expression_out.lipsync import smooth_envelope

# Bert-VITS2 架构假设（`net_g.infer` 返回 attn 对齐到的帧率）：换 fork 需人工核对同步。
HOP_LENGTH_SAMPLES = 512

# τ（最小驻留）：候选 {60, 80, 100}ms 中点，占位值——design.md §四，G4 盲测（≤7 轮）
# ∩ M9 双预算数值门定值，通过的最小 τ 定值；三候选全败须回议会复裁，不得私放宽预算。
TAU_MS = 80.0

# β（能量混合基底）：候选 [0.3, 0.5] 中点，占位值——M3 公式
# `factor = BETA + (1-BETA)*max(floor, env_norm)`，G4 盲测定值（防 E[XY]≤min 期望收缩）。
BETA = 0.4

# 连续滑移率（ms^-1，非 v1 的 per-frame 定值）：候选 0.002，占位值——G4 盲测；
# 非均匀关键帧下 `max_delta = SLEW_RATE * dt_ms`（M5），依赖渲染端线性插值语义
# （08-31 跨仓回执已终答，语义变更需重新论证）。
SLEW_RATE = 0.002

# 有声段最小开度（M3 地板，只施加在包络因子，不作用在乘积——见 `_mix_energy`）；
# 与 v1 的 VOICED_MIN_OPENING 同款动机、独立常量（v2 合成链路不沿用 v1 值，防误耦合）。
VOICED_MIN_OPENING_V2 = 0.12

# M8 时长恒等校验容差：`max(DURATION_TOLERANCE_FRAMES*frame_ms, DURATION_TOLERANCE_RATIO*wav_ms)`。
DURATION_TOLERANCE_FRAMES = 2
DURATION_TOLERANCE_RATIO = 0.01

# 双唇塞音——M2 强制闭唇（open=0，与后接韵母无关，物理约束：发音瞬间双唇必须接触）。
BILABIAL_CONSONANTS = frozenset({"b", "p", "m"})

# 自算归一定标分位（v1 `NORM_PERCENTILE` 同值、独立实现——只做响度定标，
# **不做** v1 的 FLOOR_PERCENTILE 对比度拉伸/OPENING_GAMMA（架构决策 #4：拉伸会二次
# 压缩音素查表已给出的形状差异，违反 G4「动态范围不收缩」验收项）。
_NORM_PERCENTILE = 0.95

# 主元音音段查表：键=`text/chinese.py`（经 opencpop-strict.txt）输出的真实音素符号，
# 值=(open, form) ∈ [0,1]²——open=下颌开合度，form=圆唇/pucker 度（T0 未核验真机参数
# 名前，`build_phoneme_keyframes` 的 mouth_form_param=None 时 form 不写出）。
#
# **查表键=主元音音段（M2）**，不按四呼标签分类——`iong`/`iu` 虽传统归类"齐齿呼"
# （i 起头），主元音实际是 `ong`/`ou` 的圆唇后元音，form 必须高；`o` 零声母、无韵头，
# 按"有无 u/v 韵头"分类会误判成不圆唇。o/iong 是 design.md M2 明确点名的反例，
# 逐一核对 opencpop-strict.txt 后 iu 同款同理一并订正（design 未点名但同一错误模式）。
#
# 覆盖面核对方式：tasks.md 给出的 35 符号清单 + 核对 `text/symbols.py` 的 `zh_symbols`
# 后确认还存在 4 个清单遗漏但确会在真实合成中出现的符号——`i0`/`ir`（z/c/s、zh/ch/sh/r
# 后的舌尖元音，如"zi""chi"）、`E`/`En`（零声母 y 类音节"ye""yan"的特殊韵母，
# 不写作"ie"/"ian"）——均已按"读 chinese.py 确认清单，不凭 tasks.md 记忆"的要求补全，
# 不属于自由发挥（未改变 M2 规则本身，只是把同一规则应用到验证后发现的完整符号集）。
#
# f/v 无唇齿维、ü(v)/u 趋同（design §三简化 #1/#2，勿加第三维）：v 系列 form 与
# 对应 u 系列同档、open 略低（ü 比 u 略靠前偏闭）。
_VOWEL_SHAPE_TABLE: dict[str, tuple[float, float]] = {
    # 开口呼（主元音 a，开度大、不圆唇）
    "a": (0.90, 0.10),
    "ai": (0.85, 0.10),
    "an": (0.85, 0.10),
    "ang": (0.85, 0.15),
    "ao": (0.80, 0.25),  # 收尾 u 滑音带一点圆唇
    "ia": (0.85, 0.10),
    "ian": (0.60, 0.10),
    "iang": (0.80, 0.10),
    "iao": (0.75, 0.20),
    "ua": (0.85, 0.20),
    "uai": (0.80, 0.20),
    "uan": (0.80, 0.20),
    "uang": (0.80, 0.20),
    # 央/半开元音（e/en 系，中等开度、不圆唇）
    "e": (0.50, 0.15),
    "ei": (0.50, 0.10),
    "en": (0.40, 0.15),
    "eng": (0.40, 0.15),
    "er": (0.55, 0.15),
    "ie": (0.45, 0.10),
    "E": (0.50, 0.10),  # "ye" 专用韵母（非 "ie"）
    "En": (0.45, 0.10),  # "yan" 专用韵母（非 "ian"）
    # 闭口呼（主元音 i，开度小、不圆唇）
    "i": (0.15, 0.05),
    "in": (0.20, 0.05),
    "ing": (0.20, 0.05),
    "i0": (0.10, 0.05),  # z/c/s 后舌尖元音（"zi""ci""si"），近闭唇延续
    "ir": (0.10, 0.05),  # zh/ch/sh/r 后卷舌元音（"zhi""chi""shi""ri"），同上
    # 圆唇（o/u/ü 系与 iong/iu 反例——M2 点名/同款订正，不得按韵头误判）
    "o": (0.45, 0.90),
    "ong": (0.35, 0.85),
    "iong": (0.35, 0.85),  # M2 反例：齐齿呼标签下的圆唇主元音
    "ou": (0.40, 0.70),
    "iu": (0.35, 0.75),  # 同上错误模式：主元音是圆唇的 "ou"，非 i
    "u": (0.15, 0.95),
    "ui": (0.40, 0.40),
    "un": (0.35, 0.40),
    "uo": (0.45, 0.85),
    "v": (0.15, 0.90),  # ü，唇形与 u 趋同（design §三简化 #2）
    "van": (0.50, 0.60),
    "ve": (0.45, 0.60),
    "vn": (0.20, 0.70),
}


def frame_ms_from_sample_rate(sample_rate: float, hop_length: int = HOP_LENGTH_SAMPLES) -> float:
    """TTS 侧一帧（w_ceil 计数单位）对应的毫秒数：`hop_length / sample_rate * 1000`。"""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate 必须为正：{sample_rate}")
    return hop_length / sample_rate * 1000.0


def resolve_phoneme_durations(
    phones: list[str],
    durations: list[int],
    tts_frame_ms: float,
    wav_duration_ms: float,
) -> list[float] | None:
    """帧数 durations → 毫秒列表；M8 判据：结构检查 + 时长恒等容差校验。

    结构检查失败（非空/等长/逐项>0）或超出容差 → `None`（调用方降级 v1）；
    带内则按比例缩放（不截断）——TTS 侧 `w_ceil` 累计与 wav 采样点存在取整误差，
    等比缩放吸收该误差而不破坏各音素的相对时长比例。
    """
    if not phones or not durations or len(phones) != len(durations):
        return None
    if any(d <= 0 for d in durations):
        return None
    durations_ms = [d * tts_frame_ms for d in durations]
    total_ms = sum(durations_ms)
    if total_ms <= 0.0:
        return None
    tolerance_ms = max(
        DURATION_TOLERANCE_FRAMES * tts_frame_ms, DURATION_TOLERANCE_RATIO * wav_duration_ms
    )
    if abs(total_ms - wav_duration_ms) > tolerance_ms:
        return None
    scale = wav_duration_ms / total_ms
    return [d_ms * scale for d_ms in durations_ms]


def _tau_affine_reorder(durations_ms: list[float]) -> list[float]:
    """M1 τ 仿射重排：短音素（<TAU_MS）拉到 TAU_MS，柔性段（其余音素）按比例收缩吸收
    超额，保证 Σ 不变。

    退化情形（Σ_short_forced 超总时长，含无柔性段可收缩的全短句）：柔性段吸收不完
    超额时，比例收缩会让某段变负——确定性回退为**放弃本句重排、原样返回**（Σ 不变
    式仍然成立，只是失去 τ 保护；测试钉住这一分支，不允许静默产出负时长）。
    """
    total = sum(durations_ms)
    short_mask = [d < TAU_MS for d in durations_ms]
    if not any(short_mask):
        return list(durations_ms)
    target = [
        TAU_MS if is_short else d for is_short, d in zip(short_mask, durations_ms, strict=True)
    ]
    overage = sum(target) - total
    if overage <= 0.0:
        return target
    flex_sum = sum(d for is_short, d in zip(short_mask, durations_ms, strict=True) if not is_short)
    if flex_sum <= overage:
        return list(durations_ms)
    flex_scale = (flex_sum - overage) / flex_sum
    return [
        TAU_MS if is_short else d * flex_scale
        for is_short, d in zip(short_mask, durations_ms, strict=True)
    ]


def _phone_boundaries_ms(durations_ms: list[float]) -> list[int]:
    """音素边界 t_ms：累计和向早取整（floor，M4）——不得改 round()/ceil()。"""
    boundaries: list[int] = []
    cumulative = 0.0
    for d in durations_ms:
        boundaries.append(int(math.floor(cumulative)))
        cumulative += d
    return boundaries


def _shape_for_phones(phones: list[str]) -> list[tuple[float, float]]:
    """逐音素查 (open, form) 目标：b/p/m 强制闭唇；主元音查表；其余（含标点，M2 未
    单独登记标点分支，已知简化留观察）目标=**后接的下一个主元音**；句尾无后续目标
    时退化为闭合（design §三简化 #4，非新增行为）。
    """
    shapes: list[tuple[float, float]] = []
    for i, phone in enumerate(phones):
        if phone in BILABIAL_CONSONANTS:
            shapes.append((0.0, 0.0))
            continue
        vowel_shape = _VOWEL_SHAPE_TABLE.get(phone)
        if vowel_shape is not None:
            shapes.append(vowel_shape)
            continue
        target = (0.0, 0.0)
        for j in range(i + 1, len(phones)):
            next_shape = _VOWEL_SHAPE_TABLE.get(phones[j])
            if next_shape is not None:
                target = next_shape
                break
        shapes.append(target)
    return shapes


def _slew_limit_shapes(
    shapes: list[tuple[float, float]], boundaries: list[int]
) -> list[tuple[float, float]]:
    """M5 连续滑移率限幅：作用在音素形状轨迹（非最终能量乘积），
    `max_delta = SLEW_RATE * dt_ms`——非均匀关键帧下 v1 的 per-frame 定值失效。

    起点假定为静音闭合（open=form=0，与 v1 `envelope_to_mouth_track` 的
    `limited=0.0` 起点同款）；第一个音素若边界恰在 t_ms=0，dt=0 ⇒ 首帧强制闭合
    ——这与"折叠进首音素的前导 blank 时长即模型的起音预热"是一致的，非 bug。
    """
    limited: list[tuple[float, float]] = []
    prev_open, prev_form = 0.0, 0.0
    prev_t = 0
    for (target_open, target_form), t_ms in zip(shapes, boundaries, strict=True):
        dt_ms = max(0, t_ms - prev_t)
        max_delta = SLEW_RATE * dt_ms
        prev_open += max(-max_delta, min(max_delta, target_open - prev_open))
        prev_form += max(-max_delta, min(max_delta, target_form - prev_form))
        limited.append((prev_open, prev_form))
        prev_t = t_ms
    return limited


def _percentile(values: list[float], q: float) -> float:
    """确定性分位数（最近秩法，与 v1 同算法、独立实现）；空表返回 0。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))
    return ordered[rank]


def _interpolate_envelope(smoothed: list[float], frame_ms: float, t_ms: int) -> float:
    """在非均匀关键帧时间点上线性插值等间隔包络序列（越界夹到端点）。"""
    if not smoothed:
        return 0.0
    if len(smoothed) == 1 or frame_ms <= 0.0:
        return smoothed[0]
    idx_f = t_ms / frame_ms
    idx0 = max(0, min(len(smoothed) - 1, int(math.floor(idx_f))))
    idx1 = min(len(smoothed) - 1, idx0 + 1)
    frac = min(1.0, max(0.0, idx_f - idx0))
    return smoothed[idx0] * (1.0 - frac) + smoothed[idx1] * frac


def _mix_energy(shape_open: float, env_norm: float) -> float:
    """M3 能量混合：地板只施加在包络因子（乘前），绝不作用在乘积上——
    `shape_open=0`（闭唇音）⇒ 乘积恒为 0，即使 `env_norm` 拉满（架构决策 #5）。
    """
    factor = BETA + (1.0 - BETA) * max(VOICED_MIN_OPENING_V2, env_norm)
    return shape_open * factor


def build_phoneme_keyframes(
    phones: list[str],
    durations_ms: list[float],
    envelope: list[float],
    envelope_frame_ms: float,
    mouth_form_param: str | None,
) -> list[dict[str, object]]:
    """音素序列 + 已解析毫秒时长 + 能量包络 → 口型关键帧（形状同 `lipsync.py` v1）。

    `mouth_form_param is None` 时每帧只含 `MouthOpen`（M6 单键退化，T0 未核验圆唇
    参数真名前的默认状态）；否则每帧同时含两键（同一遍历同一时间戳序列构造，
    架构决策 #7：无需事后并集，键集天然逐帧统一）。
    """
    if not phones or not durations_ms:
        return []
    reordered = _tau_affine_reorder(durations_ms)
    boundaries = _phone_boundaries_ms(reordered)
    raw_shapes = _shape_for_phones(phones)
    slewed_shapes = _slew_limit_shapes(raw_shapes, boundaries)
    smoothed_envelope = smooth_envelope(envelope)
    scale = _percentile(smoothed_envelope, _NORM_PERCENTILE)
    frames: list[dict[str, object]] = []
    for t_ms, (shape_open, shape_form) in zip(boundaries, slewed_shapes, strict=True):
        env_value = _interpolate_envelope(smoothed_envelope, envelope_frame_ms, t_ms)
        env_norm = 0.0 if scale <= 0.0 else min(1.0, env_value / scale)
        open_value = _mix_energy(shape_open, env_norm)
        params: dict[str, object] = {"MouthOpen": round(open_value, 4)}
        if mouth_form_param is not None:
            params[mouth_form_param] = round(shape_form, 4)
        frames.append({"t_ms": t_ms, "params": params})
    return frames
