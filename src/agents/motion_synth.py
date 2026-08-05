r"""动作合成器：把 (v,a) 调制系数 + 相位状态合成为 Live2D 参数关键帧轨迹。

纯函数、torch-free（与 `emotion_lexicon` 同族），给定 seed 与相位输入**完全确定性**。
结构 = 程序化 idle 基线 + 情感调制的噪声振荡：

    trajectory(t) = idle_baseline(t) + modulation(v, a) ⊙ noise_basis(t)

设计决策全部来自议会设计门（`notes/2026-08-05-motion-council-design-gate.md`），几条**不要改回去**
的裁定：

1. **无方向/朝向维**。「效价→趋近-回避朝向」是构念错误——愤怒是负效价但**趋近**取向，按
   valence 裸符号定向会让生气时后撤；改用 coping_potential 门控则头部运动学证据不足
   （专测姿态运动学的实证为零结果，且测的是观察者而非表达者）。两条候选机制都不够 ⇒
   v1 整维移出，只留 arousal 驱动的幅度/速度/onset。
2. **噪声用 value-noise 插值（Perlin 式）直接作位置**。OU 过程样本路径几乎必然处处不可导，
   直接当位置会让「角速度」在理论上未定义、帧率越高越不稳，与速度/加速度上限冲突；
   若改用 OU 必须驱动**速度**再积分。本实现走可导路线，C0/C1 平凡成立。
3. **眨眼非泊松、且不接 arousal**。泊松要求均值=方差，实证不满足；IBI 正偏态且有阵发成串。
   眨眼率主要受认知负荷/注意（多巴胺能）调制，与呼吸/心率的自主神经通路不是一套机制。
4. **与生理/韵律通道去同步**。共享同一因果起点（都只读 e\* 的 arousal），但不字面共享单一
   标量——两个通道帧级锁相会比真人「更整齐」。本模块用独立种子与更短的响应时间常数。
5. **相位功能式线程**：`generate` 收 `phase_in` 返回 `phase_out`，模块内**无可变状态**，
   故解码器可作进程级单例共享（与 `ChannelDecoder` 族的无状态前提一致）；per-session 相位
   由调用方（mcp_server 层）保管。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

# ── idle 基线参数（生物学取值，议会核验）────────────────────────────────────────
# 静息呼吸 12–20 次/分 = 0.2–0.33Hz。取中位，**与低频姿态漂移分开建模**——两者共用一套
# 参数会让呼吸感消失或把漂移变成假呼吸。
BREATH_HZ = 0.27
BREATH_AMPLITUDE = 0.006  # 呼吸对头部的机械耦合幅度很小（文献称 "weak head movements"）

# 低频姿态漂移（postural sway）：远慢于呼吸的独立带
SWAY_HZ = 0.07
SWAY_AMPLITUDE = 0.02

# 生理性震颤 3–10Hz 是**非目标**频段（虚拟形象尺度不可见）；噪声基频上限刻意留在其下，
# 防合成器噪声误入该带。
NOISE_BASE_HZ = 0.9

# 眨眼：静息 15–20 次/分 ⇒ IBI 均值 3–4s。正偏态 + 阵发成串（非泊松）。
BLINK_IBI_MEAN_S = 3.5
BLINK_IBI_SIGMA = 0.55  # log-normal 的形状参数，制造正偏态长尾
BLINK_BURST_PROB = 0.18  # 成串概率：一次眨眼后紧跟第二次
BLINK_BURST_IBI_S = 0.35
BLINK_CLOSE_MS = 120.0

# 调制映射：arousal ∈ [-1,1] → 幅度/频率缩放。未接训练模型时的解析回退（等价于既有各通道
# 「未注入真模型则走解析占位」的模式）。真模型注入后由其输出覆盖。
AROUSAL_AMPLITUDE_GAIN = 0.75
AROUSAL_SPEED_GAIN = 0.6

# posed→spontaneous 域校准：RAVDESS 是演员摆拍，原作者自陈较自发表达夸大 ⇒ 学到的
# 「唤醒→幅度」斜率上界偏陡。默认整体衰减；可经 .env 调。
DEFAULT_AMPLITUDE_SCALE = 0.7

# 输出的 Live2D 参数名（对面 `params_list` 的标准输入参数）。用角度语义参数，
# 与训练信号 `pose_R*`（旋转弧度）同构。
PARAM_ANGLE_X = "FaceAngleX"
PARAM_ANGLE_Y = "FaceAngleY"
PARAM_ANGLE_Z = "FaceAngleZ"
PARAM_EYE_OPEN_L = "EyeOpenLeft"
PARAM_EYE_OPEN_R = "EyeOpenRight"

# VTS 角度参数惯例值域（度）；合成结果按此 clamp，防越界被对面拒收。
ANGLE_RANGE_DEG = 30.0


@dataclass(frozen=True)
class PhaseState:
    """跨拉取续接所需的全部状态——**钉到字段级**，不用「振荡器状态」这种不可核验的说法。

    Attributes:
        elapsed_ms: 会话内累计经过时间。呼吸/漂移相位由它算，**不按段重置**——按段重置
            会让每次拉取的拼接点出现真实跳变（破坏 C0）。
        noise_seed: 噪声场种子。固定即噪声场固定，配合绝对时间坐标使续接天然连续。
        next_blink_ms: 下一次眨眼的绝对时刻（会话时间轴）。跨段保留才不会每段都在开头眨眼。
        blink_burst_pending: 上一次眨眼是否触发了成串，决定下一间隔取短值。
    """

    elapsed_ms: float = 0.0
    noise_seed: int = 0
    next_blink_ms: float = BLINK_IBI_MEAN_S * 1000.0
    blink_burst_pending: bool = False


@dataclass(frozen=True)
class Modulation:
    """情感调制系数（训练模型的输出；未注入模型时由 `modulation_from_affect` 解析给出）。

    ⚠ **没有方向/朝向项**——见模块 docstring 决策 1。
    """

    amplitude: float  # 幅度缩放（× 基线）
    speed: float  # 频率缩放（× 基线）
    onset_sharpness: float  # 起始锐度：越大越"猛地一动"，对应 jerk 构念


def _hash01(seed: int, index: int) -> float:
    """确定性伪随机 [0,1)：整数散列，不用全局 RNG（避免隐藏可变状态、保证可复现）。"""
    x = (seed * 0x9E3779B1 + index * 0x85EBCA77) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x2545F491) & 0xFFFFFFFF
    x ^= x >> 13
    return (x & 0xFFFFFF) / float(0x1000000)


def _smoothstep(t: float) -> float:
    """Perlin 的五次平滑插值 6t⁵−15t⁴+10t³：一阶、二阶导在节点处均为 0 ⇒ 结果处处可导。

    这正是选它而非 OU 的理由——位置轨迹可导，「角速度」才有良定义。
    """
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def value_noise(t_seconds: float, *, seed: int, frequency: float) -> float:
    """一维 value noise ∈ [-1,1]：格点随机值 + 五次平滑插值。处处可导（C1）。

    以**绝对时间**为坐标 ⇒ 跨段续接只要 seed 与时间原点不变即天然连续，无需额外桥接。
    """
    x = t_seconds * frequency
    i = math.floor(x)
    frac = x - i
    a = _hash01(seed, int(i)) * 2.0 - 1.0
    b = _hash01(seed, int(i) + 1) * 2.0 - 1.0
    return a + (b - a) * _smoothstep(frac)


def modulation_from_affect(valence: float, arousal: float) -> Modulation:
    """解析回退：(v,a) → 调制系数。真模型注入后由其输出替代本函数。

    只用 arousal（幅度/速度/锐度三条均由唤醒驱动，文献支撑扎实）；**valence 不参与**——
    它在文献里对应的是肢体节段间相位关系与趋近-回避方向，前者本模型不建模、后者已按
    议会裁定移出（见模块 docstring 决策 1）。保留形参是为了签名与其它通道一致、
    且将来若有 valence 的可靠运动学证据可就地接入。
    """
    del valence  # 刻意不用：见 docstring
    level = (arousal + 1.0) / 2.0  # [-1,1] → [0,1]
    return Modulation(
        amplitude=1.0 + AROUSAL_AMPLITUDE_GAIN * (level - 0.5) * 2.0,
        speed=1.0 + AROUSAL_SPEED_GAIN * (level - 0.5) * 2.0,
        onset_sharpness=level,
    )


def _blink_openness(now_ms: float, phase: PhaseState) -> tuple[float, PhaseState]:
    """算当前眼睑开合 ∈ [0,1]，并在越过预定时刻后排下一次眨眼。

    IBI 走 log-normal（正偏态长尾）+ 成串分支，**不是泊松**——泊松要求均值=方差，实证不满足。
    **不读 arousal**：眨眼率主要受认知负荷/注意调制，本仓当前无该信号 ⇒ 保持解耦。
    """
    state = phase
    if now_ms >= state.next_blink_ms + BLINK_CLOSE_MS:
        index = int(state.next_blink_ms)
        if state.blink_burst_pending:
            interval_s = BLINK_BURST_IBI_S
            burst_next = False
        else:
            u = _hash01(state.noise_seed ^ 0x5EED, index)
            # log-normal：exp(μ + σz)，用 hash 反演一个近似正态（Box-Muller 的简化替代）
            z = math.sqrt(-2.0 * math.log(max(u, 1e-9))) * math.cos(
                2.0 * math.pi * _hash01(state.noise_seed ^ 0xB11B, index)
            )
            interval_s = BLINK_IBI_MEAN_S * math.exp(BLINK_IBI_SIGMA * z - BLINK_IBI_SIGMA**2 / 2)
            burst_next = _hash01(state.noise_seed ^ 0xBADA, index) < BLINK_BURST_PROB
        state = replace(
            state,
            next_blink_ms=state.next_blink_ms + BLINK_CLOSE_MS + interval_s * 1000.0,
            blink_burst_pending=burst_next,
        )

    delta = now_ms - state.next_blink_ms
    if 0.0 <= delta <= BLINK_CLOSE_MS:
        # 半个余弦：闭合再睁开，端点导数为 0（不跳变）
        return (1.0 - math.sin(math.pi * delta / BLINK_CLOSE_MS)), state
    return 1.0, state


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def generate(
    valence: float,
    arousal: float,
    duration_ms: float,
    phase_in: PhaseState,
    *,
    fps: float = 20.0,
    modulation: Modulation | None = None,
    amplitude_scale: float = DEFAULT_AMPLITUDE_SCALE,
) -> tuple[list[dict[str, object]], PhaseState]:
    """合成一段关键帧轨迹，返回 (keyframes, phase_out)。

    **功能式线程相位**：不改 `phase_in`，把推进后的状态作为第二个返回值交回调用方保管。
    模块内无可变状态 ⇒ 可作进程级单例共享。

    Args:
        valence: 效价 ∈ [-1,1]（当前仅透传给调制函数，见 `modulation_from_affect`）。
        arousal: 唤醒 ∈ [-1,1]，驱动幅度/速度/锐度。
        duration_ms: 本段时长。调用方须已按契约上限 clamp（本函数不重复裁剪策略，
            但会按 fps 保证帧数不超 `TRAJECTORY_MAX_KEYFRAMES` 的算术前提）。
        phase_in: 上一段结束时的相位；首段传 `PhaseState(noise_seed=...)`。
        fps: 关键帧率。20fps × 10s = 200 帧，远低于对面 600 帧上限。
        modulation: 训练模型给的调制系数；None → 用解析回退。
        amplitude_scale: posed→spontaneous 域校准的整体衰减系数。

    Returns:
        keyframes: `[{"t_ms": int, "params": {参数名: 值}}]`，`t_ms` **从 0 起算**
            （segment-relative，契约要求），严格升序，同段键集一致。
        phase_out: 供下一段续接。
    """
    mod = modulation or modulation_from_affect(valence, arousal)
    step_ms = 1000.0 / fps
    count = max(2, int(duration_ms / step_ms) + 1)

    keyframes: list[dict[str, object]] = []
    phase = phase_in
    for i in range(count):
        local_ms = min(i * step_ms, duration_ms)
        absolute_ms = phase_in.elapsed_ms + local_ms
        t = absolute_ms / 1000.0

        # 呼吸带与漂移带分开：两者共用参数会让呼吸感消失
        breath = BREATH_AMPLITUDE * math.sin(2.0 * math.pi * BREATH_HZ * t)
        sway = SWAY_AMPLITUDE * math.sin(2.0 * math.pi * SWAY_HZ * t)

        # 三轴各取一路独立噪声（种子偏移不同），频率随 speed 调制、幅度随 amplitude 调制。
        # onset_sharpness 经一个更高频的次谐波体现"起始更猛"，对应 jerk 构念。
        freq = NOISE_BASE_HZ * mod.speed
        gain = mod.amplitude * amplitude_scale
        axis_values: list[float] = []
        for axis in range(3):
            base = value_noise(t, seed=phase.noise_seed + axis * 101, frequency=freq)
            sharp = value_noise(t, seed=phase.noise_seed + axis * 101 + 7, frequency=freq * 2.5)
            axis_values.append(gain * (base + mod.onset_sharpness * 0.35 * sharp))

        openness, phase = _blink_openness(absolute_ms, phase)

        keyframes.append(
            {
                "t_ms": int(round(local_ms)),
                "params": {
                    PARAM_ANGLE_X: _clamp(
                        (axis_values[0] + sway) * ANGLE_RANGE_DEG, -ANGLE_RANGE_DEG, ANGLE_RANGE_DEG
                    ),
                    PARAM_ANGLE_Y: _clamp(
                        (axis_values[1] + breath) * ANGLE_RANGE_DEG,
                        -ANGLE_RANGE_DEG,
                        ANGLE_RANGE_DEG,
                    ),
                    PARAM_ANGLE_Z: _clamp(
                        axis_values[2] * ANGLE_RANGE_DEG, -ANGLE_RANGE_DEG, ANGLE_RANGE_DEG
                    ),
                    PARAM_EYE_OPEN_L: openness,
                    PARAM_EYE_OPEN_R: openness,
                },
            }
        )

    phase_out = replace(phase, elapsed_ms=phase_in.elapsed_ms + duration_ms)
    return keyframes, phase_out


# 意志对动作的压制上限：意志可部分压制但**不归零**（Rinn 1984 的双通路面部神经支配结论，
# 本仓 `ExpressionAgent` 的 voluntary_coping_leak 同源）。留残余是关键——完全压平等于
# 「情绪塌陷」，那是既定裁定明确反对的。
MIN_VOLUNTARY_LEAK = 0.25


def generate_dual(
    affect: tuple[float, float],
    regulated_affect: tuple[float, float] | None,
    duration_ms: float,
    phase_in: PhaseState,
    *,
    voluntary_leak: float = 1.0,
    fps: float = 20.0,
    modulation: Modulation | None = None,
    amplitude_scale: float = DEFAULT_AMPLITUDE_SCALE,
) -> tuple[dict[str, list[dict[str, object]]], PhaseState]:
    """产出**双通路**轨迹，镜像 `ExpressionAgent` 对每个表情通道的既有做法。

    - `spontaneous`（第 ① 层·非随意）：由 `affect` 直驱，锥体外路/皮层下——情绪原样泄漏。
    - `voluntary`（第 ② 层·随意）：由 `regulated_affect` 驱动、且情绪成分按 `voluntary_leak`
      衰减，锥体束意志调控——「压着点动作」的那一路。

    两路**共用同一相位输入**（同一具身体，呼吸/眨眼节律不该分叉），故只推进一次相位；
    返回的 `phase_out` 对两路都适用。

    `regulated_affect=None`（未开调节）或 `voluntary_leak=1.0` 时，`voluntary` 与
    `spontaneous` **逐字相同**——与 `ExpressionAgent` 默认 leak=1.0 两头等值的零回归语义一致。

    Args:
        affect: 未调节的 e\\*。
        regulated_affect: 经 Regulation 的 e\\*；None → 退回 `affect`（未开调节）。
        voluntary_leak: ∈[0,1]，随意头保留多少情绪驱动。会被钳到 `MIN_VOLUNTARY_LEAK`
            以上——**意志压不平情绪**，压平即情绪塌陷。

    Returns:
        ({"spontaneous": [...], "voluntary": [...]}, phase_out)。
        渲染端在调节开启时应取 `voluntary`（那才是观察者看到的），`spontaneous` 作为
        「若不压制会是什么样」的对照保留，用途与既有表情双通路一致。
    """
    spontaneous, phase_out = generate(
        affect[0],
        affect[1],
        duration_ms,
        phase_in,
        fps=fps,
        modulation=modulation,
        amplitude_scale=amplitude_scale,
    )
    leak = max(MIN_VOLUNTARY_LEAK, min(1.0, voluntary_leak))
    source = regulated_affect if regulated_affect is not None else affect
    if source == affect and leak >= 1.0:
        # 两头等值：直接复用，避免同一输入算两遍（也保证逐字相同而非"数值相近"）
        return {"spontaneous": spontaneous, "voluntary": spontaneous}, phase_out
    voluntary, _ = generate(
        source[0],
        source[1],
        duration_ms,
        phase_in,  # 同一相位输入：同一具身体，节律不分叉
        fps=fps,
        modulation=modulation,
        amplitude_scale=amplitude_scale * leak,
    )
    return {"spontaneous": spontaneous, "voluntary": voluntary}, phase_out
