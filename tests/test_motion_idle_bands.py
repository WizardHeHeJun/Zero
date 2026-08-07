"""idle 基线两条正弦（呼吸 / 低频漂移）的频率守卫。

**为什么需要**：`SWAY_HZ` 2026-08-07 由借用值 0.07 改为同域实测 0.04，`BREATH_HZ` 则是
**量过之后决定不改**（0.178Hz 那个"峰"被判为取带边界的产物）。两者都没有守卫，手滑改回去
不会被任何测试发现。判据与数据见 `notes/2026-08-07-motion-idle-constants-criteria.md`。

**为什么用消融隔离而不是直接对输出做频谱**：这两条正弦幅度很小（0.6° / 0.2°），同轴上还叠着
幅度大一个量级的「移动-驻留」姿态运动。实测：对合成后的 `FaceAngleX` 做频谱，`SWAY_HZ`
取 0.04 与 0.07 得到的带内主峰**完全相同**（0.0383Hz，那其实是姿态运动的结构）——
在合成信号上根本分辨不出来，那样的测试会恒绿。
故把对应的幅度常数置零再作差，得到的就是**纯正弦本身**，再量它的频率。
这与本轮定这些常数时用的消融对照臂是同一套办法。
"""

from __future__ import annotations

import pytest

import src.agents.motion_synth as motion_synth
from src.agents.motion_synth import (
    BREATH_HZ,
    PARAM_ANGLE_X,
    PARAM_ANGLE_Y,
    SWAY_HZ,
    PhaseState,
    generate,
)

FPS = 20.0
# 跑够长才量得准：漂移一个周期就 25 秒，样本里要有足够多个周期。
DURATION_S = 600.0


def _series(param: str) -> list[float]:
    frames, _ = generate(0.0, 0.0, DURATION_S * 1000.0, PhaseState(noise_seed=5), fps=FPS)
    return [float(f["params"][param]) for f in frames]  # type: ignore[index]


def _isolated_frequency(monkeypatch: pytest.MonkeyPatch, param: str, amplitude_attr: str) -> float:
    """把某条正弦的幅度常数置零再作差，隔离出它本身，用过零点数反推频率（Hz）。"""
    baseline = _series(param)
    monkeypatch.setattr(motion_synth, amplitude_attr, 0.0)
    ablated = _series(param)
    wave = [a - b for a, b in zip(baseline, ablated, strict=True)]

    assert max(abs(v) for v in wave) > 1e-9, f"{amplitude_attr} 置零后无差异——该正弦根本没在输出里"

    crossings = sum(
        1
        for i in range(1, len(wave))
        if (wave[i - 1] <= 0.0 < wave[i]) or (wave[i - 1] >= 0.0 > wave[i])
    )
    return crossings / (2.0 * DURATION_S)  # 一个周期两次过零


def test_sway_frequency_matches_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    """低频漂移 = 同域实测 0.04Hz（周期 25s）。

    StayStill 0.048 / ReActIdle genuine（留出）0.035 ⇒ 取中。该峰**边界稳定**（真峰），
    与呼吸带那项相反。🛑 别改回借用值 0.07。
    """
    frequency = _isolated_frequency(monkeypatch, PARAM_ANGLE_X, "SWAY_AMPLITUDE_DEG")
    assert abs(frequency - SWAY_HZ) < 0.005, f"漂移实际频率 {frequency:.4f}Hz 与 SWAY_HZ 不符"
    assert abs(frequency - 0.04) < 0.005, f"漂移实际频率 {frequency:.4f}Hz 偏离实测 0.04Hz"


def test_breath_frequency_stays_at_literature_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """呼吸保留文献值 0.27Hz（静息 12–20 次/分的中位）。

    🛑 **别改成 0.178**：那个"峰"是取带方式的产物——把 0.15–0.5Hz 的下限挪到 0.20，
    "峰"跟着从 0.127 挪到 0.225（几乎一比一跟着边界跑），相对 1/f 背景的突起仅 1.65~1.67
    个残差 sd（不显著），两个数据集一致 ⇒ 待机头动里测不到可归因于呼吸的谱峰。
    """
    frequency = _isolated_frequency(monkeypatch, PARAM_ANGLE_Y, "BREATH_AMPLITUDE_DEG")
    assert abs(frequency - BREATH_HZ) < 0.01, f"呼吸实际频率 {frequency:.4f}Hz 与 BREATH_HZ 不符"
    assert abs(frequency - 0.27) < 0.01, (
        f"呼吸实际频率 {frequency:.4f}Hz 偏离文献值 0.27Hz——若是想改用实测 0.178，见上方 docstring"
    )


def test_two_bands_stay_separated() -> None:
    """呼吸与漂移必须分属两条带：共用一套参数会让呼吸感消失、或把漂移变成假呼吸。

    ⚠ 本条读的是 **import 时的常量快照**，故只对**真改源码**驱红，对运行时
    `monkeypatch.setattr(motion_synth, "SWAY_HZ", ...)` 不敏感——写变异脚本时别据此
    以为它失效了。已实测：源码改 `SWAY_HZ=0.10` 时本条与漂移频率那条一起变红。
    """
    assert BREATH_HZ > SWAY_HZ * 3.0, (
        f"呼吸 {BREATH_HZ}Hz 与漂移 {SWAY_HZ}Hz 靠得太近，两条带失去区分"
    )
