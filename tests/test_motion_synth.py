"""动作合成器单测：连续性、确定性、契约不变量、议会裁定的几条硬约束。

每条测试对应一条「撤掉实现就该红」的性质，不是覆盖率填充。
"""

from __future__ import annotations

import math
import statistics

import pytest

from src.agents.motion_synth import (
    ANGLE_RANGE_DEG,
    PARAM_ANGLE_X,
    PARAM_ANGLE_Y,
    PARAM_ANGLE_Z,
    PARAM_EYE_OPEN_L,
    PARAM_EYE_OPEN_R,
    Modulation,
    PhaseState,
    generate,
    initial_blink_ms,
    modulation_from_affect,
    value_noise,
)

_ANGLE_KEYS = (PARAM_ANGLE_X, PARAM_ANGLE_Y, PARAM_ANGLE_Z)


def _params(frame: dict[str, object]) -> dict[str, float]:
    params = frame["params"]
    assert isinstance(params, dict)
    return {k: float(v) for k, v in params.items()}


def _angles(frames: list[dict[str, object]], key: str) -> list[float]:
    return [_params(f)[key] for f in frames]


# ── 契约不变量（对面 params_animate 会按这些拒收）────────────────────────────


def test_t_ms_starts_at_zero_and_is_strictly_increasing() -> None:
    """t_ms 每次调用从 0 起算（segment-relative）且严格升序——契约硬要求。"""
    frames, _ = generate(0.0, 0.0, 2000.0, PhaseState(noise_seed=7))
    stamps = [int(f["t_ms"]) for f in frames]  # type: ignore[call-overload]
    assert stamps[0] == 0
    assert all(b > a for a, b in zip(stamps, stamps[1:], strict=False))


def test_second_segment_also_starts_at_zero() -> None:
    """续接段的 t_ms 仍从 0 起算，不是会话累计绝对时间（易错点，专测）。"""
    _, phase = generate(0.0, 0.0, 2000.0, PhaseState(noise_seed=7))
    frames, _ = generate(0.0, 0.0, 2000.0, phase)
    assert int(frames[0]["t_ms"]) == 0  # type: ignore[call-overload]


def test_all_frames_share_identical_key_set() -> None:
    """同段所有帧键集必须一致，否则对面整段 rejected。"""
    frames, _ = generate(0.3, 0.5, 3000.0, PhaseState(noise_seed=3))
    key_sets = {frozenset(_params(f)) for f in frames}
    assert len(key_sets) == 1


def test_frame_count_stays_under_contract_limit() -> None:
    """10s @20fps = 201 帧，远低于对面 600 帧上限（帧数由显式算术保证，非"建议"）。"""
    frames, _ = generate(0.0, 0.9, 10000.0, PhaseState(noise_seed=1))
    assert len(frames) <= 600


def test_angles_stay_within_range() -> None:
    """角度不得越界——越界会被对面 clamp 或拒收，且观感抽搐。"""
    frames, _ = generate(-0.9, 1.0, 5000.0, PhaseState(noise_seed=11))
    for frame in frames:
        for key in _ANGLE_KEYS:
            assert abs(_params(frame)[key]) <= ANGLE_RANGE_DEG


def test_clamp_never_actually_fires() -> None:
    """🛑 clamp 是**安全上限**，正常运行不该触发——触发即波形顶部被削平（失真+满幅）。

    2026-08-06 端到端实测踩到：此前噪声基幅误用 ANGLE_RANGE_DEG，arousal≥+0.5 时输出
    恰好等于 ±30.00，即 clamp 在削波。「不越界」那条测试**抓不到这个**——被削平的
    波形当然不越界。这条查的是「有没有真的贴到上限」。

    ⚠ 2026-08-07 加强：原实现只跑 **5 秒、单种子**（约 2 个姿态周期），采样量根本抽不到
    尾部峰值——实测把基幅调到会削平 2% 帧的 28° 时，它**依然全绿**。削波是低频尾部事件，
    必须长跑多种子才测得到。现在直接数触顶帧数并要求**恰好为 0**（比"峰值离上限多远"更硬）。
    """
    for arousal in (-1.0, -0.5, 0.0, 0.5, 1.0):
        clamped = 0
        peak = 0.0
        for seed in (11, 31, 77, 101, 233):
            frames, _ = generate(0.0, arousal, 60_000.0, PhaseState(noise_seed=seed))
            for frame in frames:
                for key in _ANGLE_KEYS:
                    value = abs(_params(frame)[key])
                    peak = max(peak, value)
                    if value >= ANGLE_RANGE_DEG - 1e-6:
                        clamped += 1
        # 🛑 **这一条是真正的缺陷判据**：触顶即削平波形，必须恰好为 0。
        assert clamped == 0, f"arousal={arousal} 有 {clamped} 帧触顶被削平（峰值 {peak:.2f}°）"
        # 余量阈：原为 0.9（留 3° 余量）。2026-08-07 用户按真机观感选定第三档幅度
        # （基幅 21.8），实测 a=1.0 峰值 29.38°、**21 条 180 秒轨迹 22.7 万采样零触顶**，
        # 但余量只剩 0.62°（吃满量程 97.9%）。余量是**被有意花掉的**，不是漏检——
        # ⚠ 故今后任何往幅度通路上再加量的改动（新叠加项 / 提高基幅 / 提高唤醒增益 /
        # 接入输出幅度更大的调制模型）都必须重跑本条：它已经没有缓冲了。
        assert peak < ANGLE_RANGE_DEG * 0.99, f"arousal={arousal} 峰值 {peak:.2f}° 已贴死上限"


def _peak_over_runs(arousal: float, *, seconds: float = 60.0) -> float:
    """长跑多种子的峰值。短样本抽不到尾部——见下方两条守卫的教训注释。"""
    peak = 0.0
    for seed in (11, 31, 77, 101, 233):
        frames, _ = generate(0.0, arousal, seconds * 1000.0, PhaseState(noise_seed=seed))
        peak = max(peak, max(abs(_params(f)[k]) for f in frames for k in _ANGLE_KEYS))
    return peak


def test_calm_is_visibly_calmer_than_excited() -> None:
    """平静态必须**看起来明显比激动态小**——这才是「幅度合不合理」的可锚定判据。

    ⚠ 2026-08-07 重写。原实现断言「平静峰值 < 4.0°」，引的是文献「静息姿态摆动 ±1~3°」，
    有**两处问题**：
    1. **口径错配**：那个 ±1~3° 说的是姿态摆动（对应本模块的 `SWAY_AMPLITUDE_DEG=0.6`，
       本就在范围内），不是头部**总**运动。同域真人待机（StayStill，只是在等人）
       yaw sd 就有 21.3°，峰值远超 40° —— 拿全身摆动的数值卡头部总峰值，标准本身是错的。
    2. **采样不足**：只跑 5 秒单种子。实测同一配置短样本 2.73° / 长跑真值 3.95°，
       整体幅度从 14 抬到 19 时长跑值已到 5.16°、**守卫却仍全绿**——假绿灯。

    改为**比值**判据：平静/激动的峰值比。它不依赖任何绝对度数（故不受整体尺度调整影响），
    直接编码真正要守的性质——情绪得在幅度上看得出来。整体尺度另由 clamp 那条守。
    """
    calm_peak = _peak_over_runs(-0.9)
    excited_peak = _peak_over_runs(0.9)
    ratio = calm_peak / excited_peak
    assert ratio < 0.45, (
        f"平静/激动峰值比 {ratio:.2f} 过高（平静 {calm_peak:.2f}° vs 激动 {excited_peak:.2f}°）"
        "——情绪不再改变动作幅度"
    )
    assert excited_peak > calm_peak * 1.5, "激动态幅度未显著大于平静态"


# ── 连续性（G2 / G3）────────────────────────────────────────────────────────


def test_noise_is_continuous_across_segment_boundary() -> None:
    """跨段拼接点值差极小（C0）——绝对时间坐标 + 固定种子使续接天然连续。

    这是「相位必须按会话绝对时间、不按段重置」那条设计的直接检验。
    """
    seg1, phase = generate(0.0, 0.4, 1000.0, PhaseState(noise_seed=5))
    seg2, _ = generate(0.0, 0.4, 1000.0, phase)
    for key in _ANGLE_KEYS:
        tail = _params(seg1[-1])[key]
        head = _params(seg2[0])[key]
        assert abs(head - tail) < 0.5  # 度；相邻帧本就有的正常位移量级


def test_phase_reset_would_break_continuity() -> None:
    """反证：若第二段从零相位重开（模拟"按段重置"的错误实现），拼接点会真的跳变。

    没有这条，上面那条连续性测试可能只是"数值都很小"的假绿。
    """
    seg1, phase = generate(0.0, 0.4, 1000.0, PhaseState(noise_seed=5))
    correct, _ = generate(0.0, 0.4, 1000.0, phase)
    naive, _ = generate(0.0, 0.4, 1000.0, PhaseState(noise_seed=5))  # 相位归零 = 错误实现
    gaps_correct = [abs(_params(correct[0])[k] - _params(seg1[-1])[k]) for k in _ANGLE_KEYS]
    gaps_naive = [abs(_params(naive[0])[k] - _params(seg1[-1])[k]) for k in _ANGLE_KEYS]
    assert max(gaps_naive) > max(gaps_correct)


def test_velocity_is_bounded() -> None:
    """帧间速度有界（G2 无跳变）——可导噪声的直接收益，OU 直接作位置则做不到。"""
    frames, _ = generate(0.0, 1.0, 5000.0, PhaseState(noise_seed=9))
    for key in _ANGLE_KEYS:
        series = _angles(frames, key)
        speeds = [abs(b - a) for a, b in zip(series, series[1:], strict=False)]
        assert max(speeds) < ANGLE_RANGE_DEG / 2.0


def test_value_noise_is_smooth() -> None:
    """value noise 处处可导：细分采样下二阶差分有界（五次平滑插值的性质）。"""
    samples = [value_noise(i / 200.0, seed=4, frequency=1.0) for i in range(400)]
    second = [
        abs(samples[i + 1] - 2 * samples[i] + samples[i - 1]) for i in range(1, len(samples) - 1)
    ]
    assert max(second) < 0.05


# ── idle 与情感调制（G1 / G2）───────────────────────────────────────────────


def test_calm_state_is_not_frozen() -> None:
    """平静态仍有呼吸/微摆基线——「不静止」是 G2 的一半（另一半是不抽搐）。"""
    frames, _ = generate(0.0, -0.9, 4000.0, PhaseState(noise_seed=2))
    spread = max(_angles(frames, PARAM_ANGLE_Y)) - min(_angles(frames, PARAM_ANGLE_Y))
    assert spread > 0.05


def test_calm_state_is_visibly_moving() -> None:
    """⚠ 平静 ≠ 石化：低唤醒下也须有**肉眼可见**的姿态调整（真机实测标定）。

    2026-08-06 踩到：改「移动-驻留」结构后，低唤醒的周期被拉到 6 秒以上，采样点几乎全落
    在驻留平台，实测峰值仅 0.57° —— 皮套上看着完全静止。上面那条 `spread > 0.05` 的
    松阈值**抓不到**（0.57° 远大于 0.05）。本条按可见性设阈。
    """
    frames, _ = generate(0.0, -0.85, 8000.0, PhaseState(noise_seed=2))
    peak = max(abs(_params(f)[k]) for f in frames for k in _ANGLE_KEYS)
    assert peak > 1.2, f"平静态峰值仅 {peak:.2f}°，皮套上看着像静止"


def test_high_arousal_moves_more_than_calm() -> None:
    """高唤醒幅度显著大于平静（G1 的核心可分性，同种子排除噪声差异）。"""
    calm, _ = generate(0.0, -0.8, 5000.0, PhaseState(noise_seed=6))
    excited, _ = generate(0.0, 0.8, 5000.0, PhaseState(noise_seed=6))
    calm_rms = statistics.pstdev(_angles(calm, PARAM_ANGLE_X))
    excited_rms = statistics.pstdev(_angles(excited, PARAM_ANGLE_X))
    assert excited_rms > calm_rms * 1.5


def test_valence_does_not_drive_direction() -> None:
    """⚠ 议会裁定：v1 无方向维。同 arousal 下改变 valence **不得**改变输出。

    这条是防「后来者好心把 valence→朝向加回去」的守卫——那是构念错误（愤怒是负效价
    但趋近取向，按裸符号定向会让生气时后撤）。
    """
    negative, _ = generate(-0.9, 0.5, 3000.0, PhaseState(noise_seed=8))
    positive, _ = generate(0.9, 0.5, 3000.0, PhaseState(noise_seed=8))
    assert _angles(negative, PARAM_ANGLE_X) == _angles(positive, PARAM_ANGLE_X)
    assert _angles(negative, PARAM_ANGLE_Z) == _angles(positive, PARAM_ANGLE_Z)


def test_modulation_has_no_direction_field() -> None:
    """`Modulation` 不含方向项——结构层面钉死议会裁定。"""
    mod = modulation_from_affect(-0.5, 0.5)
    assert set(vars(mod)) == {"amplitude", "speed", "onset_sharpness"}


def test_amplitude_scale_attenuates() -> None:
    """posed→spontaneous 域校准系数真的在起作用（不是写了不接）。"""
    full, _ = generate(0.0, 0.8, 3000.0, PhaseState(noise_seed=12), amplitude_scale=1.0)
    damped, _ = generate(0.0, 0.8, 3000.0, PhaseState(noise_seed=12), amplitude_scale=0.5)
    assert statistics.pstdev(_angles(damped, PARAM_ANGLE_X)) < statistics.pstdev(
        _angles(full, PARAM_ANGLE_X)
    )


def test_injected_modulation_overrides_analytic_fallback() -> None:
    """注入训练模型的调制系数时，解析回退不再生效（真模型能接上的前提）。"""
    weak = Modulation(amplitude=0.1, speed=1.0, onset_sharpness=0.0)
    strong = Modulation(amplitude=2.0, speed=1.0, onset_sharpness=0.0)
    a, _ = generate(0.0, 0.0, 3000.0, PhaseState(noise_seed=13), modulation=weak)
    b, _ = generate(0.0, 0.0, 3000.0, PhaseState(noise_seed=13), modulation=strong)
    assert statistics.pstdev(_angles(b, PARAM_ANGLE_X)) > statistics.pstdev(
        _angles(a, PARAM_ANGLE_X)
    )


# ── 眨眼（生物学：非泊松、不接 arousal）──────────────────────────────────────


def test_blink_openness_within_unit_range() -> None:
    frames, _ = generate(0.0, 0.0, 8000.0, PhaseState(noise_seed=21))
    for frame in frames:
        assert 0.0 <= _params(frame)[PARAM_EYE_OPEN_L] <= 1.0
        assert _params(frame)[PARAM_EYE_OPEN_L] == _params(frame)[PARAM_EYE_OPEN_R]


def test_initial_blink_is_seed_dispersed() -> None:
    """首次眨眼时刻随种子分散——固定默认会让每个新会话开头都恰好静止同样久。

    确定性不变：同 seed 同结果。
    """
    from src.agents.motion_synth import BLINK_IBI_MEAN_S, initial_blink_ms

    values = {initial_blink_ms(s) for s in range(20)}
    assert len(values) > 15  # 真的分散了，不是常数
    assert all(0.0 < v <= BLINK_IBI_MEAN_S * 1000.0 for v in values)
    assert initial_blink_ms(7) == initial_blink_ms(7)  # 同种子可复现


def test_blink_actually_happens() -> None:
    """长段里必须真的眨过眼（否则眼睑通道等于没接）。"""
    frames, _ = generate(0.0, 0.0, 20000.0, PhaseState(noise_seed=22))
    assert min(_params(f)[PARAM_EYE_OPEN_L] for f in frames) < 0.5


def test_blink_rate_is_calm_not_nervous() -> None:
    """⚠ 眨眼率须落在安静待机的合理区间——真机观感反馈「频率有点快」后加的守卫。

    2026-08-06 实测踩到两个叠加的实现错误：
    (1) log-normal 减 σ²/2 修正后**中位数比设定均值短**（设 3.5s 实得 3.05s），
        我当初按均值设参、未验证实际落点；
    (2) 成串概率 18% 是拍脑袋定的（文献只称「存在阵发成串」，未给频率），
        实测每 6 次眨眼含 1 次连眨，8 秒片段里几乎必然被看到。
    合计眨眼率 19.7 次/分，卡在文献 15~20 的上沿。本条按**实际率**设阈，
    不再只断言参数值——参数对不代表行为对。
    """
    phase = PhaseState(noise_seed=77, next_blink_ms=initial_blink_ms(77))
    times: list[float] = []
    for _ in range(12):
        frames, phase = generate(0.0, 0.0, 8000.0, phase)
        base = phase.elapsed_ms - 8000.0
        closed = False
        for frame in frames:
            openness = _params(frame)[PARAM_EYE_OPEN_L]
            if openness < 0.5 and not closed:
                times.append((base + float(frame["t_ms"])) / 1000.0)  # type: ignore[call-overload]
                closed = True
            elif openness >= 0.5:
                closed = False
    assert len(times) > 10
    span = times[-1] - times[0]
    rate = (len(times) - 1) / span * 60.0
    assert 10.0 < rate < 17.0, f"眨眼率 {rate:.1f} 次/分不在安静待机区间"

    intervals = [b - a for a, b in zip(times, times[1:], strict=False)]
    burst_ratio = sum(1 for v in intervals if v < 1.0) / len(intervals)
    assert burst_ratio < 0.12, f"成串占比 {burst_ratio:.0%} 过高（每几次就连眨一次会很显眼）"


def test_blink_intervals_are_not_poisson() -> None:
    """IBI 的方差显著偏离均值²（泊松要求两者相等）——议会点名的生物学约束。

    实现方式：log-normal + 成串分支。若有人把它改回齐次泊松/均匀随机，这条会红。
    """
    intervals: list[float] = []
    phase = PhaseState(noise_seed=31)
    previous: float | None = None
    for _ in range(40):
        frames, phase = generate(0.0, 0.0, 5000.0, phase)
        for frame in frames:
            if _params(frame)[PARAM_EYE_OPEN_L] < 0.5:
                now = phase.elapsed_ms - 5000.0 + float(int(frame["t_ms"]))  # type: ignore[call-overload]
                if previous is None or now - previous > 200.0:
                    if previous is not None:
                        intervals.append((now - previous) / 1000.0)
                    previous = now
    assert len(intervals) > 10
    mean = statistics.fmean(intervals)
    variance = statistics.pvariance(intervals)
    # 指数分布（泊松间隔）满足 variance == mean²；正偏态+成串会让它明显偏离
    assert not math.isclose(variance, mean**2, rel_tol=0.35)


def test_blink_rate_independent_of_arousal() -> None:
    """⚠ 眨眼率**不由 arousal 驱动**（受认知负荷/注意的多巴胺能通路调制，非自主唤醒）。

    防「顺手把眨眼接到 arousal 上」——那是把两条不同的生理通路混为一谈。
    """
    calm, _ = generate(0.0, -0.9, 12000.0, PhaseState(noise_seed=41))
    tense, _ = generate(0.0, 0.9, 12000.0, PhaseState(noise_seed=41))
    calm_blinks = [_params(f)[PARAM_EYE_OPEN_L] for f in calm]
    tense_blinks = [_params(f)[PARAM_EYE_OPEN_L] for f in tense]
    assert calm_blinks == tense_blinks


# ── 确定性（G5）─────────────────────────────────────────────────────────────


def test_same_seed_same_output() -> None:
    """给定 (状态, seed) 输出可复现——热路径确定性纪律。"""
    a, pa = generate(0.2, 0.3, 3000.0, PhaseState(noise_seed=99))
    b, pb = generate(0.2, 0.3, 3000.0, PhaseState(noise_seed=99))
    assert a == b
    assert pa == pb


def test_generate_does_not_mutate_input_phase() -> None:
    """功能式线程相位：不改入参，推进后的状态经返回值交回（模块无可变状态的前提）。"""
    phase_in = PhaseState(noise_seed=77)
    _, phase_out = generate(0.0, 0.0, 3000.0, phase_in)
    assert phase_in.elapsed_ms == 0.0
    assert phase_out.elapsed_ms == pytest.approx(3000.0)
    assert phase_out is not phase_in
