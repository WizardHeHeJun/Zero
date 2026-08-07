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
# 呼吸对头部的机械耦合幅度很小（文献称 "weak head movements"，被用作非接触呼吸监测信号）
BREATH_AMPLITUDE_DEG = 0.2

# 低频姿态漂移（postural sway）：远慢于呼吸的独立带
SWAY_HZ = 0.07
SWAY_AMPLITUDE_DEG = 0.6

# 生理性震颤 3–10Hz 是**非目标**频段（虚拟形象尺度不可见）；噪声基频上限刻意留在其下，
# 防合成器噪声误入该带。
NOISE_BASE_HZ = 0.9

# 眨眼：静息 15–20 次/分 ⇒ IBI 均值 3–4s。正偏态 + 阵发成串（非泊松）。
# ⚠ 2026-08-06 真机观感修正（用户反馈「眨眼频率有点快」，实测坐实）：
# 改前 3.5s + 18% 成串 ⇒ 实测眨眼率 **19.7 次/分**，卡在文献区间 15~20 的**上沿**；
# 且 log-normal 减 σ²/2 修正后**中位数只有 3.05s**（比设定的均值短）——我当初按均值设参、
# 未验证实际落点，这是两个独立错误叠加。8 秒片段内期望眨 2.6 次、每 6 次含 1 次连眨，
# 短片段里几乎必然被看到。
# 改后取 4.4s：中位数落到约 3.9s、率约 15 次/分（区间下沿，安静待机更贴切）。
BLINK_IBI_MEAN_S = 4.4
BLINK_IBI_SIGMA = 0.55  # log-normal 的形状参数，制造正偏态长尾
# 成串概率由 0.18 降至 0.05：文献只称「存在阵发成串」，**从未给出频率**——18% 是我拍脑袋
# 定的，实测每 6 次眨眼就有 1 次连眨，明显过密。5% 保留该现象但不再显眼。
BLINK_BURST_PROB = 0.05
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

# 眼球注视方向（皮套实测值域 ±1，governed=False ⇒ 表情通路未占用，可自由注入）
PARAM_EYE_L_X = "EyeLeftX"
PARAM_EYE_L_Y = "EyeLeftY"
PARAM_EYE_R_X = "EyeRightX"
PARAM_EYE_R_Y = "EyeRightY"

# ── 眼-头协同（2026-08-06 议会复核新增；此前眼球方向**完全静止**，被判为可信的不自然来源）
# 机制：真实注视转移是**眼先到位、头随后追赶**，头追上的过程中眼球相对头部反向回正
# （前庭眼反射）。实现即：眼睛用远快的过渡曲线抵达同一目标，眼球相对头部的偏移
# = 目标 − 头部当前位置，随头部追近而自然衰减到 0。
#
# ⚠ 两个常数的**来源不同，必须分别标注**（议会要求）：
# - 提前量 25~40ms：取**文献值**（Freedman 2008：眼球扫视比头动早 25–40ms 开始）。
#   本数据集**未能验证**——30fps 下该量仅 0.75~1.2 帧，且 OpenFace 的 gaze 估计自带
#   数度噪声，两个限制叠加使实测滞后恒为 0 帧。
# - 幅度：取**数据实测**——RAVDESS 全库眼球相对头部活动幅度中位 8.3°。
#   按议会建议当作**上界**处理并加保守缩放（估计值里可能混有测量噪声）。
EYE_LEAD_S = 0.03
EYE_RISE_S = 0.10  # 眼跳本身是几十~百余毫秒量级的弹道运动，远快于头部 0.45s
EYE_AMPLITUDE_DEG = 8.3  # 实测中位（上界估计）
EYE_AMPLITUDE_SCALE = 0.7  # 保守缩放，同 DEFAULT_AMPLITUDE_SCALE 的处置精神
EYE_PARAM_PER_DEG = 1.0 / 15.0  # 皮套参数 ±1 ≈ ±15° 视觉注视量程（工程估计，非实测）

# 情感调制噪声的**基幅**（度，对应调制系数 =1.0 即中性）。
# ⚠ 2026-08-06 端到端实测修正：此前直接用 `ANGLE_RANGE_DEG` 当基幅是**错的**——
# value_noise∈[-1,1] × gain(0.25~1.75) × 30° 最大到 ±52°，被 clamp 削平顶部，
# 实测 arousal≥+0.5 时数值恰好等于 ±30.00 ⇒ **clamp 真的在触发 = 波形失真 + 满幅摇头**。
# 参照量级：静息姿态摆动/呼吸耦合约 ±1~3°，说话时强调性头动约 ±5~15°。
# 取 8° 使：深度平静 ≈ ±1°（几乎只有呼吸）· 中性 ≈ ±5° · 高唤醒 ≈ ±8°，clamp 不再触发。
NOISE_AMPLITUDE_DEG = 14.0

# 三轴幅度比（yaw : pitch : roll）。**同域实测**，非手调：
# StayStill 待机数据 50 条 / 104.6 分钟，三轴 sd 中位 21.29° / 6.99° / 4.08° ⇒ 1 : 0.33 : 0.19。
# 改前三轴**等幅**，直接导致「低头幅度与转头一样大」——真人待机时低头只有转头的三分之一，
# 这正是真机观感反馈「低头比较不自然」的来源（2026-08-06/07）。
#
# ⚠ 只搬**比例**不搬**绝对幅度**：StayStill 被试是「在街上等人」可随意东张西望
# （实测 yaw 极值超 ±40°），而对话中的数字人应大部分时间面向用户 ⇒ 整体尺度由
# `NOISE_AMPLITUDE_DEG` 单独控制，属产品决定而非数据决定。
# 索引对应 axis 0/1/2 = yaw(FaceAngleX) / pitch(FaceAngleY) / roll(FaceAngleZ)。
AXIS_AMPLITUDE_RATIO = (1.0, 0.33, 0.19)

# 「移动-驻留」的标称周期（秒，调制系数=1 时）：一次姿态转移 + 驻留的总时长。
# 真人对话中的头部姿态调整大致每 1.5~3 秒一次；随 speed 调制缩短。
POSE_CYCLE_S = 2.4

# 单次姿态转移的标称时长（秒）。真人头部重定向 0.2~0.4s 完成后停住；
# 用绝对时长而非周期比例，否则慢周期下转移被拉长成"飘"。
# 2026-08-06 议会复核上调 0.3 → 0.45：0.3 被判**低估**，双文献支撑——
# Zangemeister et al. 1981 头动 main-sequence（头部惯性大，35° gaze shift 后头部还要续转
# 约 250ms）；Hendicott et al. 2002 在**非言语**日常任务实测头动幅度 6.5°/均速 11°/s
# ⇒ 幅度÷均速 ≈ 0.59s，比 RAVDESS 实测（0.43~0.50s）还长。
POSE_RISE_S = 0.45

# 驻留期叠加的"活体微颤"占比：让停住的时候不像定格画面，但不破坏「移动-驻留」结构。
MICRO_TREMOR_RATIO = 0.12

# yaw→roll 耦合系数（转头带侧倾）。取自 RAVDESS 全库实测（三档情绪一致 ⇒ 常数）。
# ⚠ 刻意在此**重新声明**而非从 `datasets.ravdess_motion` import：合成器是运行时热路径，
# 不该为一个标量把数据加载模块拉进依赖（该模块引 csv/torch 侧）。两处数值须一致，
# 改动时同步——数据侧的同名常量带完整实测出处。
YAW_ROLL_COUPLING = -0.125

# VTS 角度参数的**安全上限**（度）：只作 clamp 防越界被对面拒收。
# 🛑 正常运行**不应触发**——触发即意味着波形顶部被削平（见上）。有测试守这条。
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
    # ⚠ 默认值是**固定**的 3.5s ⇒ 每个新会话开头都恰好 3.5 秒不眨眼（2026-08-06 演示中
    # 3 秒片段一次都没眨到才发现）。构造时应传 `initial_blink_ms(seed)` 让它随会话分散；
    # 保留固定默认是为了测试可复现。
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


def initial_blink_ms(seed: int) -> float:
    """按种子分散首次眨眼时刻 ∈ (0, IBI 均值]，避免每个新会话开头都恰好静止同样久。

    确定性（同 seed 同结果），故不破坏可复现性。
    """
    return BLINK_IBI_MEAN_S * 1000.0 * (0.15 + 0.85 * _hash01(seed ^ 0xB1B1, 0))


def _smoothstep(t: float) -> float:
    """Perlin 的五次平滑插值 6t⁵−15t⁴+10t³：一阶、二阶导在节点处均为 0 ⇒ 结果处处可导。

    这正是选它而非 OU 的理由——位置轨迹可导，「角速度」才有良定义。
    也用作「移动-驻留」的最小 jerk 过渡曲线：两端速度、加速度均为 0，
    起落自然、无突兀启停。
    """
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _pose_cycle(t: float, seed: int, mod: Modulation) -> tuple[list[float], list[float], float]:
    """「移动-驻留」姿态序列：返回 (头部姿态, 眼球相对头部偏移, 本周期进度)。

    第二个返回值是**眼-头协同**（2026-08-06 议会新增）：眼球用远快的过渡曲线 + 提前量
    抵达同一目标，故 `eye_offset = 眼球世界位置 − 头部当前位置`——头部还差多少没走完，
    就是眼球相对头部偏出去多少，随头部追近自然衰减到 0（前庭眼反射的粗近似）。
    此前眼球方向完全静止，被议会判为可信的不自然来源。

    ⚠ 2026-08-06 真机观感修正：此前三轴各自跑一路连续噪声，头会**持续漂移**——
    像被水流推着，而不是在"看向某处、停住、再看别处"。真实头动是**弹道式**的：
    快速移到一个姿态 → 驻留 → 再移动（gaze shift + hold）。本函数即此结构。

    每个周期：前 `rise` 比例做最小 jerk 过渡（`_smoothstep`，两端速度/加速度为 0），
    其余时间驻留在目标姿态。周期长度随 `mod.speed` 缩短（越激动动得越频繁）。
    周期边界处位置连续、速度为零 ⇒ C0/C1 均连续，跨段续接不跳变。

    **轴间耦合**（生物力学）：真人转头（yaw）必然伴随轻微侧倾（roll），不是三轴各飘各的。
    此处 roll 取自身分量 + 一部分 yaw 的反向耦合。
    """
    # ⚠ 上限 3.2s 不可去：`POSE_CYCLE_S / speed` 在低唤醒下会拉到 6 秒以上，采样点几乎
    # 全落在驻留平台 → 平静态看着像**完全静止**（2026-08-06 真机实测 0.57° 峰值）。
    # 平静不等于石化：真人安静时仍有每 2~3 秒一次的小幅姿态调整。
    cycle_s = min(3.2, max(0.45, POSE_CYCLE_S / max(mod.speed, 0.2)))
    index = int(t // cycle_s)
    frac = (t - index * cycle_s) / cycle_s
    # ⚠ 过渡时长用**绝对秒数**，不是周期比例（2026-08-06 真机观感修正）。
    # 原写法 rise=0.25~0.5×周期，中性档周期 2.4s ⇒ 单次转移要 0.9 秒；而真人的头部
    # 重定向是 0.2~0.4 秒完成后停住。慢 2~3 倍就会显出「飘」的质感，即便有停顿结构也一样。
    rise_s = POSE_RISE_S * (1.35 - 0.5 * mod.onset_sharpness)  # 越锐利越快
    rise = min(0.8, rise_s / cycle_s)  # 转成本周期内的比例；留至少 20% 驻留
    progress = _smoothstep(min(1.0, frac / rise)) if frac < rise else 1.0

    def target(idx: int, axis: int) -> float:
        """姿态目标，按 `AXIS_AMPLITUDE_RATIO` 分轴缩放（同域实测比例，见该常量注释）。"""
        return (_hash01(seed + axis * 977, idx) * 2.0 - 1.0) * AXIS_AMPLITUDE_RATIO[axis]

    # 眼球：同一目标，但提前 EYE_LEAD_S 起步、用远短的 EYE_RISE_S 走完 ⇒ 几乎瞬间到位。
    eye_frac = (t + EYE_LEAD_S - index * cycle_s) / cycle_s
    eye_rise = min(0.9, EYE_RISE_S / cycle_s)
    eye_progress = (
        _smoothstep(min(1.0, max(0.0, eye_frac) / eye_rise)) if eye_frac < eye_rise else 1.0
    )

    pose: list[float] = []
    eye_offset: list[float] = []
    for axis in range(3):
        start = target(index - 1, axis) if index > 0 else 0.0
        end = target(index, axis)
        head = start + (end - start) * progress
        pose.append(head)
        # 眼球世界位置减头部当前位置 = 眼球相对头部的偏转（头追上则归零）
        eye_world = start + (end - start) * eye_progress
        eye_offset.append(eye_world - head)
    # roll ← 自身 + yaw 反向耦合（转头带侧倾）。
    # 系数 2026-08-06 议会复核由 −0.35 下修至实测值：−0.35 是**大幅主动转头**的量级
    # （Guo et al. 2021 实测全幅旋转 55.5° 时上/下颈段各约 18~21° 补偿性侧屈），
    # 而会话级小角度头动的耦合应更弱，RAVDESS 全库实测 −0.125（三档情绪完全一致
    # ⇒ 属运动学固有属性、非情绪调制，故为常数不进模型；Ceylan et al. 2000 的 Fick 型
    # 头部约束支持这一机制归属）。
    pose[2] = 0.6 * pose[2] + YAW_ROLL_COUPLING * pose[0]
    return pose, eye_offset, frac


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

        # 呼吸带与漂移带分开：两者共用参数会让呼吸感消失。两者均以**度**为单位直接给出，
        # 不随情感调制缩放——呼吸和体态摆动是持续的生理背景，不因情绪激动而成倍放大。
        breath_deg = BREATH_AMPLITUDE_DEG * math.sin(2.0 * math.pi * BREATH_HZ * t)
        sway_deg = SWAY_AMPLITUDE_DEG * math.sin(2.0 * math.pi * SWAY_HZ * t)

        # 主结构 =「移动-驻留」姿态序列（弹道式，非持续漂移）；
        # 再叠一层很轻的连续噪声当"活体微颤"，避免驻留期像定格画面。
        gain_deg = mod.amplitude * amplitude_scale * NOISE_AMPLITUDE_DEG
        pose, eye_offset, _frac = _pose_cycle(t, phase.noise_seed, mod)
        micro_freq = NOISE_BASE_HZ * mod.speed
        axis_deg: list[float] = []
        for axis in range(3):
            micro = value_noise(t, seed=phase.noise_seed + axis * 101 + 7, frequency=micro_freq)
            axis_deg.append(gain_deg * (pose[axis] + MICRO_TREMOR_RATIO * micro))

        openness, phase = _blink_openness(absolute_ms, phase)

        # 眼球注视：偏移量（归一化）→ 度 → 皮套参数（±1）。两眼同步（不做斜视/辐辏）。
        eye_gain = EYE_AMPLITUDE_DEG * EYE_AMPLITUDE_SCALE * EYE_PARAM_PER_DEG
        eye_x = _clamp(eye_offset[0] * eye_gain, -1.0, 1.0)
        eye_y = _clamp(eye_offset[1] * eye_gain, -1.0, 1.0)

        keyframes.append(
            {
                "t_ms": int(round(local_ms)),
                "params": {
                    PARAM_ANGLE_X: _clamp(
                        axis_deg[0] + sway_deg, -ANGLE_RANGE_DEG, ANGLE_RANGE_DEG
                    ),
                    PARAM_ANGLE_Y: _clamp(
                        axis_deg[1] + breath_deg, -ANGLE_RANGE_DEG, ANGLE_RANGE_DEG
                    ),
                    PARAM_ANGLE_Z: _clamp(axis_deg[2], -ANGLE_RANGE_DEG, ANGLE_RANGE_DEG),
                    PARAM_EYE_OPEN_L: openness,
                    PARAM_EYE_OPEN_R: openness,
                    PARAM_EYE_L_X: eye_x,
                    PARAM_EYE_L_Y: eye_y,
                    PARAM_EYE_R_X: eye_x,
                    PARAM_EYE_R_Y: eye_y,
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
