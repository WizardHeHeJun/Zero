"""文本输入路径闭环 demo：STTextAffectRegressor → 完整 6 节点管线 → e* + 情绪词标签。

展示「文本型 stimulus」如何通过 PerceptionAgent 的句向量回归器产出 text_affect=(v,a)，
再经完整管线（appraisal → value → affect_core → mood → expression → supervisor）
演化为最终情绪向量 e*。

用法：
    设环境变量后跑（PowerShell 示例）：
        $env:ZERO_TEXT_AFFECT_BACKEND="st"
        $env:ZERO_TEXT_AFFECT_MODEL_PATH="artifacts/text_affect_regressor_st.pt"
        conda run -n affective-expression --no-capture-output python -m scripts.demo_text_input

    不设 ZERO_TEXT_AFFECT_BACKEND 时（默认关行为演示）：
        conda run -n affective-expression --no-capture-output python -m scripts.demo_text_input

--------------------------------------------------------------------------------
设计定调：文本作独立低精度先验流（TEXT_AFFECT_PRECISION=0.3）
经 fuse_terms 精度加权参与后验，不进 occ_prior 入口、不污染 survival。

诚实标注：文本流精度低（0.3），故对 e* 是"温和拉动"而非主导；
occ_prior（prior_mu）仍由 OCC 字段决定（文本型 stimulus 未填 OCC 字段 → 近中性先验）。
文本影响路径：PerceptionAgent 产出 text_affect=(v,a) → AffectCore 把 text_affect
作独立 term 进 fuse_terms，精度加权（TEXT_AFFECT_PRECISION=0.3）参与后验计算。
开 workspace_enabled=True 时 ignited_streams 中会出现 "text" 流名。

实现要点：features 布局为 OCC 下标顺序（perception.py），
fast_survival_prior 读 features[0]=goal_congruence / features[3]=intensity，
文本信息走 text_affect 字段、不会污染 survival 流。

--------------------------------------------------------------------------------
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

from src.agents.affect_math import text_label
from src.memory.client import MemoryClient
from src.orchestration.graph import build_graph
from src.orchestration.runner import ALLOWED_CHECKPOINT_TYPES
from src.orchestration.state import AffectState, Stimulus
from src.storage.checkpointer import build_checkpointer
from src.storage.graph_store import build_graph_store, build_semantic_store

logger = logging.getLogger(__name__)


def _check_env() -> bool:
    """检查文本路径的 env 配置。

    返回 True 表示后端已配置（文本路径激活），False 表示默认关（OCC 路径）。
    默认关时打印提示并继续运行 OCC 路径演示（不退出，演示默认关行为）。
    """
    backend = os.getenv("ZERO_TEXT_AFFECT_BACKEND", "")
    model_path = os.getenv("ZERO_TEXT_AFFECT_MODEL_PATH", "")

    if not backend:
        print(
            "\n[默认关演示] ZERO_TEXT_AFFECT_BACKEND 未设置。"
            "\n  文本路径处于默认关状态 → PerceptionAgent 走 OCC 占位路径。"
            "\n  text_affect 字段将为 None，不加入 fuse_terms 文本流。"
            "\n  sentence-transformers / STTextAffectRegressor 不会被 import。"
            "\n  要启用文本路径，设置："
            "\n    ZERO_TEXT_AFFECT_BACKEND=st"
            "\n    ZERO_TEXT_AFFECT_MODEL_PATH=artifacts/text_affect_regressor_st.pt"
            "\n  本次演示 OCC 路径行为（stimulus 文本字段存在但被忽略）。\n"
        )
        return False

    if backend != "st":
        print(
            f"\n[配置错误] ZERO_TEXT_AFFECT_BACKEND={backend!r} 为未知后端，仅支持 'st'。"
            "\n  回退 OCC 路径并继续演示。\n"
        )
        return False

    if not model_path:
        print(
            "\n[配置不完整] ZERO_TEXT_AFFECT_BACKEND=st 已设置，"
            "但 ZERO_TEXT_AFFECT_MODEL_PATH 未设置。"
            "\n  回退 OCC 路径并继续演示。\n"
        )
        return False

    if not os.path.isfile(model_path):
        print(
            f"\n[配置错误] ZERO_TEXT_AFFECT_MODEL_PATH={model_path!r} 文件不存在。"
            "\n  请确认权重路径正确（如 artifacts/text_affect_regressor_st.pt）。"
            "\n  回退 OCC 路径并继续演示。\n"
        )
        return False

    print(
        f"\n[文本路径激活] backend=st，权重路径={model_path!r}"
        "\n  PerceptionAgent 将用 STTextAffectRegressor.predict_affect 产出 text_affect=(v,a)。"
        "\n  AffectCore 将把 text_affect 作独立低精度流（TEXT_AFFECT_PRECISION=0.3）"
        "\n  经 fuse_terms 精度加权参与后验（不进 occ_prior 入口）。\n"
    )
    return True


def _perception_from_trace(state: AffectState) -> dict[str, Any]:
    """从 AffectState.trace 中提取 perception 节点的 features 和 backend 标签。

    runner._state_to_entry 不暴露 trace；demo 直接持有 AffectState 对象，
    从 state.trace 里找 perception 条目，以便展示 backend 路径标签。

    AffectState.trace 使用 operator.add reducer：多轮 ainvoke 共用同一 checkpointer 时，
    每轮追加本轮各节点条目——取最后一个 perception 条目（代表本轮），而非第一个（可能是上轮的）。
    """
    last: dict[str, Any] | None = None
    for entry in state.trace:
        if entry.get("node") == "perception":
            last = entry
    if last is not None:
        return {
            "features": last.get("features", state.features),
            "backend": last.get("backend", "unknown"),
        }
    return {"features": state.features, "backend": "unknown"}


def _print_state(stim_text: str, state: AffectState, text_path_active: bool) -> None:
    """格式化打印单次管线运行的关键中间量与最终 e*。

    打印内容：
    - features：OCC 布局（goal_congruence, standard_compliance, attitude_appeal, intensity）
    - text_affect：文本路径产出的独立 (v,a) 先验流（默认关时为 None）
    - ignited_streams：确认 "text" 流是否进入点燃（workspace_enabled=True 时可见）
    - prior_mu：OCC 字段决定的先验均值（文本型 stim 未填 OCC -> 近中性先验）
    - e*=(v,a)：fuse_terms 精度加权后验的采样结果
    """
    perc = _perception_from_trace(state)
    e_star = state.affect_sample
    emotion_label = text_label(e_star[0], e_star[1]) if e_star else "unknown"
    prior_mu = state.prior_mu
    affect_precision = state.affect_precision

    print(f"  刺激文本     : {stim_text!r}")
    print(f"  感知 backend : {perc['backend']}")
    # features 恢复 OCC 布局：[goal_congruence, standard_compliance, attitude_appeal, intensity]
    # 文本型 stim 未填 OCC 字段 -> 均为 0.0（近中性），fast_survival_prior 据此安全读 features[0/3]
    print(f"  感知 features: {[round(f, 4) for f in (perc['features'] or [])]}")
    # text_affect：文本路径（backend=st_text）由回归器产出的独立 (v,a) 先验流
    # 此值经 AffectCore 以 TEXT_AFFECT_PRECISION=0.3 进入 fuse_terms，
    # 不进 occ_prior 入口、不修改 prior_mu。默认关（OCC 路径）时为 None。
    ta_str = (
        str(tuple(round(x, 4) for x in state.text_affect))
        if state.text_affect is not None
        else "None"
    )
    print(f"  text_affect  : {ta_str}  <- 文本独立低精度先验流 (v,a)；None=默认关/OCC 路径")
    # prior_mu：OCC 字段决定（文本型 stim 未填 OCC -> 近中性先验），不受文本流影响
    print(
        f"  prior_mu(OCC): {tuple(round(x, 4) for x in prior_mu) if prior_mu else 'None'}"
        "  <- OCC 字段决定（文本型 stim 未填 OCC -> 近中性先验，文本流不改此入口）"
    )
    # ignited_streams：workspace_enabled=True 时可见 "text" 流（表示文本流通过显著度门控点燃）
    print(f"  ignited 流   : {state.ignited_streams}  <- 含 'text' 表示文本流已点燃参与后验")
    print(
        f"  affect_prec  : {round(affect_precision, 4) if affect_precision is not None else 'None'}"
    )
    print(f"  e* (v,a)     : {tuple(round(x, 4) for x in e_star) if e_star else 'None'}")
    print(f"  情绪词标签   : {emotion_label}")

    if not text_path_active:
        print(
            "  [注] 默认关状态：stim.text 存在但 PerceptionAgent 未加载回归器，"
            "text_affect=None，文本流不进入 fuse_terms。"
            "features 来自 OCC 字段（全零 -> 中性）。"
        )
    elif perc["backend"] == "occ_placeholder":
        print("  [注] 文本路径 env 已设但回归器加载失败，实际走 OCC 路径（见日志 warning）。")
    else:
        # text_path_active=True 且 backend=st_text：确认文本流生效
        if "text" in state.ignited_streams:
            print(
                "  [文本流确认] ignited_streams 含 'text'：文本流通过显著度门控点燃，"
                "以 TEXT_AFFECT_PRECISION=0.3 参与 fuse_terms 后验。"
            )
        else:
            print(
                "  [注意] ignited_streams 不含 'text'：文本流显著度未过 SALIENCE_THRESHOLD，"
                "停留局部（e* 仍由其他流主导，文本流温和但可能未达门控阈值）。"
            )

    print()


async def _run_stimuli(
    stimuli: list[Stimulus],
    *,
    thread_id: str,
    rng_seed: int | None,
    workspace_enabled: bool,
) -> list[AffectState]:
    """用底层 build_graph 跑 stimuli 序列，返回完整 AffectState 列表。

    runner.run() 返回的 _state_to_entry dict 不含 trace；
    demo 需要 state.trace 来提取 perception backend 标签，因此直接用 graph.ainvoke。
    graph.py / runner.py 的签名与接线保持不变——demo 直接构造初始 state dict，
    workspace_enabled=True 在此注入，不需要改 runner 签名。
    """
    client = MemoryClient(build_graph_store(), semantic=build_semantic_store())
    checkpointer = build_checkpointer(ALLOWED_CHECKPOINT_TYPES)
    graph = build_graph(checkpointer=checkpointer, memory=client)

    states: list[AffectState] = []
    for stim in stimuli:
        result = await graph.ainvoke(
            {
                "stimulus": stim,
                "session_id": thread_id,
                "user_id": "demo-user",
                "group_id": "demo-group",
                "regulation_enabled": False,
                "regulation_strategy": "suppression",
                "mood_enabled": False,
                "recall_enabled": False,
                "language_enabled": False,
                "workspace_enabled": workspace_enabled,
                "appraisal_conditioning_enabled": False,
                "language_max_iters": 3,
                "rng_seed": rng_seed,
                "task_complete": False,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        state = AffectState(**result)
        states.append(state)
    return states


async def run_demo(text_path_active: bool) -> None:
    """构造正负文本 stimulus，跑完整管线，打印关键中间量与最终 e*。

    workspace_enabled=True：开启显著度门控全局工作空间（文本流参与融合的前提）。
    AffectCore 将把 text_affect 作独立 term 进 fuse_terms，
    精度加权（TEXT_AFFECT_PRECISION=0.3）参与后验，
    不进 occ_prior 入口、不污染 survival。

    prior_mu 来自 AppraisalAgent 读 Stimulus.goal_congruence 等 OCC 字段。
    文本型 stimulus 未填 OCC 字段 -> prior_mu 退化为近中性 (≈0, low_arousal)。
    文本影响路径：text_affect=(v,a) → AffectCore 独立流 → fuse_terms 精度加权后验。
    文本流精度低（0.3），是"温和拉动"而非主导；occ_prior 仍由 OCC 字段决定。
    """
    # 正向文本：期待高 valence 预测
    pos_text = "What a wonderful, exciting day! I feel great about everything!"
    # 负向文本：期待低 valence 预测
    neg_text = "This is terrible, I'm furious. Everything is going wrong."

    # 文本型 stimulus：OCC 字段保持默认（全零），text 字段携带原始文本
    # OCC 字段全零 -> prior_mu 退化为 (0.0, 约 0.2) 近中性先验（不被文本改变）
    # 文本影响走独立 text_affect 通道 -> fuse_terms 精度加权后验（TEXT_AFFECT_PRECISION=0.3）
    stimuli = [
        Stimulus(name="positive_text", text=pos_text, intensity=0.8),
        Stimulus(name="negative_text", text=neg_text, intensity=0.8),
    ]

    states = await _run_stimuli(
        stimuli,
        thread_id="demo-text-input",
        rng_seed=42,
        # workspace_enabled=True：需开启工作空间，文本流才能进入 fuse_terms
        workspace_enabled=True,
    )

    print("=" * 70)
    print("  文本输入路径闭环 demo 结果（workspace_enabled=True）")
    print("=" * 70)
    print()

    print("[1] 正向文本（positive）")
    _print_state(pos_text, states[0], text_path_active)

    print("[2] 负向文本（negative）")
    _print_state(neg_text, states[1], text_path_active)

    print("=" * 70)
    if text_path_active:
        print("  [文本路径确认]")
        print("    - backend=st_text: 文本路径被走到，text_affect=(v,a) 来自回归器")
        print("    - text_affect 作独立低精度流（TEXT_AFFECT_PRECISION=0.3）进 fuse_terms")
        print("    - prior_mu 仍由 OCC 字段决定（近中性，不改 occ_prior 入口）")
        print("    - 文本流是温和拉动（精度 0.3 低于 occ_prior/survival），非主导")
        print("    - ignited_streams 含 'text' = 文本流通过显著度门控，实际参与后验")
    else:
        print("  [默认关确认] backend=occ_placeholder；text_affect=None；")
        print("    features 来自 OCC 字段（均零->中性）。")
        print("    sentence-transformers / STTextAffectRegressor 未加载。")
        print("    fuse_terms 不加文本流，保持默认行为。")
    print("=" * 70)

    # 完整 JSON 轨迹（supply further observation）—— 用 state 字段序列化
    print("\n[完整轨迹 JSON]")
    trajectory = [
        {
            "stimulus": stim.name,
            "valence_arousal": state.affect_sample,
            "features": state.features,
            "text_affect": state.text_affect,
            "perception_backend": _perception_from_trace(state)["backend"],
            "ignited_streams": state.ignited_streams,
            "affect_precision": state.affect_precision,
            "prior_mu": state.prior_mu,
            "prior_sigma": state.prior_sigma,
            "reward": state.reward,
            "rpe": state.rpe,
            "precision": state.precision,
            "value_estimate": state.value_estimate,
            "mood": state.mood,
            "expression": state.expression,
        }
        for stim, state in zip(stimuli, states, strict=True)
    ]
    print(json.dumps(trajectory, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    text_path_active = _check_env()
    asyncio.run(run_demo(text_path_active))


if __name__ == "__main__":
    main()
