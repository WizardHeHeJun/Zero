"""Persona：给数字人「指定人格」的可注入配置（身份 + 气质底色 + 预置关系）。

把「这个人是谁、什么性子、跟用户已有怎样的关系」从写死的常量/固定 prompt 抽成一个可注入对象，
让框架不必每次从零塑造一个人。三层（与 chat_driver / language 接线对应）：

- **L1 人设卡** `card`：身份/背景/口吻/与用户关系，注入对话 system prompt（`OpenAILanguageModel`）。
- **L2 气质底色** `setpoint/reactivity/recovery`：情感引擎的个体基线与反应性——慢变态度回归的锚
  （[[ATTITUDE_SETPOINT]] 注释预留的「人格阶段」）与快变情绪的反应/恢复速率。**只暴露旋钮、默认
  中性**；「大五→PAD 的具体数值映射 / 预设人格库」是科学决策，须走 `/science-council` 设计门，
  本模块不替算法拍板。
- **L3 预置关系** `initial_attitude/seed_memories`：首次接触时的初始态度 + 预灌的共同记忆，
  让数字人「一开始就认识/在意某人」，而非从零相处。

**默认 `Persona()` 全中性 == 现有行为**（setpoint=(0,0)、反应/恢复=引擎常量、无人设/无种子）
→ 严格零回归。纯数据 + stdlib 加载，不 import 记忆/存储层（守 agents 层封装边界）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from src.agents.affect_math import ATTITUDE_SETPOINT, EMOTION_REACTIVITY, EMOTION_RECOVERY


@dataclass(frozen=True)
class Persona:
    """一个数字人的人格定义；不可变，作配置在装配期一次性读入。

    字段全部可选，缺省即对应层退化为现有中性行为（零回归）。`setpoint`/`initial_attitude`
    为 (valence, arousal) 二元组，各分量∈[-1,1]（越界由下游 clamp 收口，不在此强校验）。
    """

    name: str = ""
    card: str = ""  # L1：身份/背景/口吻/与用户关系 → 注入对话 system prompt
    setpoint: tuple[float, float] = ATTITUDE_SETPOINT  # L2：习惯性情感基线（态度回归锚）
    reactivity: float = EMOTION_REACTIVITY  # L2：对刺激的即时反应增益
    recovery: float = EMOTION_RECOVERY  # L2：情绪向基线衰退的残留比例（小=恢复快）
    initial_attitude: tuple[float, float] | None = None  # L3：首次接触的初始态度种子
    seed_memories: tuple[str, ...] = field(default_factory=tuple)  # L3：预灌的共同记忆


def _coerce_pair(value: object, name: str) -> tuple[float, float]:
    """把 JSON 里的 [v, a] 列表强制成 (float, float)；格式不符即报错（显式配置不容猜）。"""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"persona.{name} 须为 [valence, arousal] 两元素列表，得到：{value!r}")
    return (float(value[0]), float(value[1]))


def _persona_from_dict(data: dict[str, object]) -> Persona:
    """从已解析的 dict 构造 Persona；只取已知字段、逐字段做类型收口。"""
    kwargs: dict[str, object] = {}
    if "name" in data:
        kwargs["name"] = str(data["name"])
    if "card" in data:
        kwargs["card"] = str(data["card"])
    if "setpoint" in data:
        kwargs["setpoint"] = _coerce_pair(data["setpoint"], "setpoint")
    if "reactivity" in data:
        kwargs["reactivity"] = float(data["reactivity"])  # type: ignore[arg-type]
    if "recovery" in data:
        kwargs["recovery"] = float(data["recovery"])  # type: ignore[arg-type]
    if data.get("initial_attitude") is not None:
        kwargs["initial_attitude"] = _coerce_pair(data["initial_attitude"], "initial_attitude")
    if "seed_memories" in data:
        seeds = data["seed_memories"]
        if not isinstance(seeds, (list, tuple)):
            raise ValueError(f"persona.seed_memories 须为字符串列表，得到：{seeds!r}")
        kwargs["seed_memories"] = tuple(str(s) for s in seeds)
    return Persona(**kwargs)  # type: ignore[arg-type]


def load_persona() -> Persona:
    """从环境装配 Persona。未配置任何项 → 返回中性 `Persona()`（== 现有行为，零回归）。

    两种来源（互补）：
    - `ZERO_PERSONA_FILE`：人格定义 JSON 文件路径（全字段：name/card/setpoint/reactivity/
      recovery/initial_attitude/seed_memories，均可选）。**显式给了路径却读不出/格式错 → 抛错**
      （配置错误 fail-fast，不静默退化中性，见 [[config-only-via-env]]）。
    - `ZERO_PERSONA`：仅人设卡文本的快捷入口（只配 L1 时省去写文件）；与文件并存时文件优先。
    """
    path = os.getenv("ZERO_PERSONA_FILE")
    if path:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(
                f"ZERO_PERSONA_FILE={path} 顶层须为 JSON 对象，得到：{type(data).__name__}"
            )
        return _persona_from_dict(data)
    card = os.getenv("ZERO_PERSONA", "")
    return Persona(card=card)
