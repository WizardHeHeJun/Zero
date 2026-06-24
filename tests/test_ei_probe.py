"""EmoBench 式情商探针回归：curated 情绪场景 → 端到端管线 → 细粒度情绪正确性。

非 EmoBench 原集（其为 400 题 LLM 推理基准，licensing + 规模不入仓）；这是面向本 VA 管线
的自包含代理探针：场景(OCC 评价维度) → appraisal→value→affect_core → affect_sample，
用 emotion_lexicon 度量「效价方向 + Panksepp 动机系统」的命中率，守住「评价→情绪」映射
与粒度升级的正确性。接真 EmoBench：把 PROBES 换成其 (scenario, label) 并加 VA 映射层即可。

走占位路径（不注入解码器），torch-free；rng_seed 固定 → 确定性可回归。
"""

from __future__ import annotations

from src.agents.emotion_lexicon import affect_label, motivational_system
from src.memory.client import MemoryClient
from src.orchestration.graph import build_graph
from src.orchestration.runner import ALLOWED_CHECKPOINT_TYPES
from src.orchestration.state import Stimulus
from src.storage.checkpointer import build_checkpointer

# (场景名, OCC 维度, 期望效价符号, 期望动机系统集合)
# occ_prior 的 arousal=0.4|intensity|+0.6|valence| 恒 ≥0 → 情绪多落上半平面，
# 故探针聚焦「效价方向 + 高/低唤起强度」这一稳健可判别信号（不强求 care/panic 下半盘）。
PROBES: list[tuple[str, dict[str, float], str, set[str]]] = [
    ("得偿所愿", {"goal_congruence": 0.9, "intensity": 0.9}, "+", {"seeking"}),
    (
        "称心如意",
        {"goal_congruence": 0.7, "attitude_appeal": 0.8, "intensity": 0.6},
        "+",
        {"seeking"},
    ),
    ("小确幸", {"goal_congruence": 0.5, "intensity": 0.3}, "+", {"seeking", "care"}),
    (
        "遭遇不公",
        {"goal_congruence": -0.9, "standard_compliance": -0.8, "intensity": 0.9},
        "-",
        {"rage"},
    ),
    (
        "触犯底线",
        {"goal_congruence": -0.5, "standard_compliance": -0.9, "intensity": 0.8},
        "-",
        {"rage"},
    ),
    ("失望落空", {"goal_congruence": -0.6, "intensity": 0.5}, "-", {"rage", "panic_grief"}),
]


async def _run_affect(stim: Stimulus) -> tuple[float, float]:
    """跑一遍占位管线，取提交的 e*=(valence, arousal)。"""
    graph = build_graph(build_checkpointer(ALLOWED_CHECKPOINT_TYPES), MemoryClient())
    result = await graph.ainvoke(
        {"stimulus": stim, "rng_seed": 7},
        config={"configurable": {"thread_id": f"ei-{stim.name}"}},
    )
    v, a = result["affect_sample"]
    return (float(v), float(a))


async def test_ei_probe_direction_system_and_granularity() -> None:
    correct_sign = 0
    correct_system = 0
    labels: set[str] = set()
    for name, dims, sign, systems in PROBES:
        v, a = await _run_affect(Stimulus(name=name, **dims))
        labels.add(affect_label(v, a))
        if (v >= 0.0) == (sign == "+"):
            correct_sign += 1
        if motivational_system(v, a) in systems:
            correct_system += 1
    n = len(PROBES)
    # 效价方向：清晰场景 + 确定性种子 → 应全对
    assert correct_sign == n, f"效价方向命中 {correct_sign}/{n}"
    # 动机系统：允许个别边界滑动，但整体高命中
    assert correct_system / n >= 0.8, f"动机系统命中率 {correct_system}/{n}"
    # 粒度：产出多样情绪词，未坍缩到单一标签（守住粒度升级）
    assert len(labels) >= 3, f"情绪词多样性不足: {labels}"
