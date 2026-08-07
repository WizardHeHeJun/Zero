"""yaw-roll 耦合：符号（对侧）与量级由三个数据集的免泄漏实测钉死。

**为什么要单独一组守卫**：耦合符号错了会让侧倾反向——皮套上一眼可见，数值里完全看不出。
而恰好有一个很容易被误采信的数字：旧提取公式在 StayStill 上给出的 **+0.415**。
它是几何伪影（`atan2(up_x, up_z)` 在 pitch≠0 时混入 yaw，伪分量正比于 yaw），
把真实 roll 置零做消融后该公式仍报 +0.50~+0.96 —— 量到的主要是伪影本身。

免泄漏（swing-twist）重测：StayStill −0.474 · ReActIdle genuine −0.424 · acted −0.397，
逐片段符号一致率 96%/100%/93%。测定脚本与五条判据见 `tools/motion/coupling_measure.py`，
坐标系自动判定与合成自检见 `tools/motion/anatomy.py` / `selftest_anatomy.py`。

皮套侧：Live2D 标准参数表对 ParamAngleX 与 ParamAngleZ 用逐字相同的「+ = 画面右」措辞
⇒ 两者正方向同侧 ⇒ 解剖系换算时 yaw 与 roll 同时翻转 ⇒ **相关系数符号不变**。

本组按**行为**断言（实际输出序列的相关系数），不是断言参数值——参数对不代表行为对
（眨眼那次的教训：均值设 3.5s、行为中位只有 3.05s）。
"""

from __future__ import annotations

import statistics

from src.agents.motion_synth import (
    AXIS_AMPLITUDE_RATIO,
    PARAM_ANGLE_X,
    PARAM_ANGLE_Z,
    YAW_ROLL_COUPLING,
    PhaseState,
    _pose_cycle,
    generate,
    initial_blink_ms,
    modulation_from_affect,
)

# 三集合并实测（解剖·增量口径）。守卫容差取 ±0.15：够宽以容纳合成器叠加的呼吸/漂移/微颤，
# 够窄以在符号翻转或量级失控时驱红。
MEASURED_R = -0.45


def _series(frames: list[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for frame in frames:
        params = frame["params"]
        assert isinstance(params, dict)
        values.append(float(params[key]))
    return values


def _pearson(xs: list[float], ys: list[float]) -> float:
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    cov = sum(a * b for a, b in zip(dx, dy, strict=True))
    norm = (sum(a * a for a in dx) * sum(b * b for b in dy)) ** 0.5
    return cov / norm if norm > 0 else 0.0


def _long_run(seed: int) -> list[dict[str, object]]:
    """跑够长（80 秒）才让随机目标的分布稳定——单段的目标数太少，相关系数噪声极大。"""
    phase = PhaseState(noise_seed=seed, next_blink_ms=initial_blink_ms(seed))
    frames: list[dict[str, object]] = []
    for _ in range(8):
        segment, phase = generate(0.0, 0.0, 10000.0, phase)
        frames.extend(segment)
    return frames


def test_coupling_constant_is_the_measured_correlation() -> None:
    """常量语义 = 实测相关系数本身（可与数据/文献直接对照），不是内部混合系数。

    🛑 **不得改成 +0.415**——那是被消融证伪的几何伪影，改回去等于让侧倾反向。
    """
    assert YAW_ROLL_COUPLING == -0.45


def test_coupling_is_contralateral() -> None:
    """符号：转头与侧倾**反向**（对侧）。这是三个数据集一致、判别力最强的一条。"""
    for seed in (11, 31, 77):
        frames = _long_run(seed)
        r = _pearson(_series(frames, PARAM_ANGLE_X), _series(frames, PARAM_ANGLE_Z))
        assert r < -0.15, f"seed={seed} 的 yaw-roll 相关 {r:+.3f} 不是对侧耦合"


def test_coupling_magnitude_matches_measurement() -> None:
    """量级：输出序列的实际相关应落在实测值附近，不能只是"符号对了"。"""
    values = []
    for seed in (11, 31, 77, 101, 233):
        frames = _long_run(seed)
        values.append(_pearson(_series(frames, PARAM_ANGLE_X), _series(frames, PARAM_ANGLE_Z)))
    median = statistics.median(values)
    assert abs(median - MEASURED_R) < 0.15, (
        f"实际相关中位 {median:+.3f} 偏离实测 {MEASURED_R:+.2f} 过多（各 seed: "
        + ", ".join(f"{v:+.3f}" for v in values)
        + "）"
    )


def test_coupling_preserves_axis_amplitude_ratio() -> None:
    """耦合项**不得**改变 roll 的幅度——方差配分（√(1−r²) 与 r 作权重）就是为这条。

    在 `_pose_cycle` 这一层测：输出层的 FaceAngleZ 还叠了微颤（等绝对量加到三轴上，
    对小幅度的 roll 相对放大更多，实测把比值从 0.19 抬到 0.219），
    在那测量不出耦合项自身有没有破坏比例——判别力会被淹掉。

    ⚠ 必须**跨 seed 汇池**：单 seed 200 秒只产出约 83 个姿态目标，两条 hash 流的经验 sd
    有 ~9% 波动（r=0 时实测 0.173 而非 0.19），量到的会是采样噪声而不是代数。
    汇池 24 个 seed 后比值稳定在 0.192~0.194 且**与 r 无关**（代数预期如此）。

    旧写法 `0.6*roll + (−0.125)*yaw` 在本层给 0.172（比例值的 91%），会被本测试驱红。
    """
    mod = modulation_from_affect(0.0, 0.0)
    yaws, rolls = [], []
    for seed in range(1, 25):
        for step in range(4000):
            pose, _eye, _frac = _pose_cycle(step * 0.05, seed, mod)
            yaws.append(pose[0])
            rolls.append(pose[2])
    ratio = statistics.pstdev(rolls) / statistics.pstdev(yaws)
    expected = AXIS_AMPLITUDE_RATIO[2]
    assert abs(ratio - expected) < 0.012, (
        f"pose 层 roll/yaw 幅度比 {ratio:.4f} 偏离三轴比例 {expected}——耦合项破坏了幅度配分"
    )
