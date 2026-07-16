"""zero-link 边界数据映射（纯函数·无 IO）：MCP 线上 JSON 载荷 ⇄ 内核类型。

跨边界只走 JSON，故：
- client 发的 `stim = {valence, arousal, coping_potential?}`（`AffectStimulus.model_dump`）
  须映射成内核 OCC 结构 `Stimulus`（无 valence/arousal 字段）。
- `external_priors` 线上是 `list[[name, [μv,μa], [Πv,Πa]]]`（JSON array，无 tuple），
  而 `expand_external_priors` 要求 **真 tuple**（`affect_math.py:977-989` isinstance 校验），
  故须 array→tuple。

映射口径镜像既有 canonical 文本→Stimulus 路径 `chat_driver.py:307-316`，不新造语义。
本模块只做形状搬运；精度上界/流数/正性等权威校验交下游 `expand_external_priors`（M3/M6）。
"""

from __future__ import annotations

from typing import Any

from src.orchestration.external_prior import ExternalPrior
from src.orchestration.state import Stimulus


def stimulus_from_payload(
    stim: dict[str, Any],
    *,
    name: str = "mcp-step",
    intensity_floor: float = 0.0,
) -> Stimulus:
    """把 client 的 `{valence, arousal, coping_potential?}` 映射成内核 `Stimulus`。

    镜像 `chat_driver.py:307-316` 的 canonical (v,a)→Stimulus：
    - `goal_congruence = valence`
    - `intensity = min(1.0, max(intensity_floor, |arousal|))`（floor 默认 0：不注入 arousal 底噪）
    - `control_appraisal = coping_potential`（仅非 None 时设；client `model_dump(exclude_none=True)`
      故省略 coping → 不设 → 保持默认 **None=absent cue**，`state.py:36`）
    `attitude_appeal` 保持默认 0.0（会话边界不承载 chat 层 running attitude）；
    `text=None`（client 已 appraise 过 (v,a)，**不**再跑我方文本回归器）。

    coping 是否真正生效由会话侧 `coping_potential_enabled` 门控决定；生效时走 B3 四分支融合
    （`appraisal.py:192-228`）：ctrl=None → absent cue（分支1/2 精度趋零不参与），显式 0.0 →
    genuine-zero（分支3/4 参与）。client 省略 coping 与显式 0.0 语义天然对齐此新契约。
    """
    if "valence" not in stim or "arousal" not in stim:
        raise ValueError(
            f"stim 缺 valence/arousal（AffectStimulus 必有两字段）；实际键={sorted(stim)}"
        )
    valence = float(stim["valence"])
    arousal = float(stim["arousal"])
    kwargs: dict[str, Any] = {
        "name": name,
        "goal_congruence": valence,
        "intensity": min(1.0, max(intensity_floor, abs(arousal))),
    }
    coping = stim.get("coping_potential")
    if coping is not None:
        kwargs["control_appraisal"] = float(coping)
    return Stimulus(**kwargs)


def external_priors_from_payload(payload: list[Any] | None) -> list[ExternalPrior]:
    """把 JSON array 形式的 external_priors 反序列化回内核所需的 3 元 tuple。

    每条线上是 `[name, [μv,μa], [Πv,Πa]]`；`expand_external_priors` 要求真 tuple。
    本函数只做形状良构（3 元 + 两个 2 元子序列）+ array→tuple；精度正性/上界/流数上界/
    physio 覆写等权威校验交 `expand_external_priors`（`affect_math.py:935-1010`，M2/M3/M6）。
    空/None → 空列表（零回归：无外部先验）。
    """
    if not payload:
        return []
    priors: list[ExternalPrior] = []
    for i, item in enumerate(payload):
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 3
            or not isinstance(item[1], (list, tuple))
            or len(item[1]) != 2
            or not isinstance(item[2], (list, tuple))
            or len(item[2]) != 2
        ):
            raise ValueError(
                f"external_priors[{i}] 形状须为 [name, [μv,μa], [Πv,Πa]]，实际为 {item!r}"
            )
        name, mu, prec = item
        priors.append(
            (
                str(name),
                (float(mu[0]), float(mu[1])),
                (float(prec[0]), float(prec[1])),
            )
        )
    return priors
