r"""MotionAgent：动作层决策进图——按回合产出 motion_directive，供 zero.motion 拉取侧渲染。

设计文档：`PRP/motion/design-agent.md`（已过议会门，见其 §6 裁定表）。核心矛盾（§0）：
LangGraph 图按对话回合推进，动作渲染按帧（20fps）——本 Agent 解决两种节拍的接口问题：
**决策进图、渲染留拉取路径**，与 `ExpressionAgent` 先例同构（它每回合产出 AU 值，
MCP 侧连续渲染；本 Agent 每回合产出 motion_directive，`zero.motion` 侧连续合成关键帧）。

产出的 `motion_directive` 是**小的、可序列化、可核验**的结构（不是不透明张量）：

    amplitude / speed / onset   spontaneous（非随意通路）调制系数，由 state.affect_sample
                                 经解析回退 `motion_synth.modulation_from_affect` 产出
    regulated: {amplitude,      voluntary（随意通路）**专属**调制系数，由 state.regulated_affect
      speed, onset} | None      经同一解析回退产出；regulated_affect 缺失或与 affect_sample
                                 相等时给 None（结构最小，`zero.motion` 据此回退复用顶层系数）。
                                 🛑 2026-08-07 修复：此前 directive 只有一组顶层系数，
                                 `zero.motion` 把它同时喂给 spontaneous/voluntary 两路，
                                 voluntary 那一路会丢掉"被调节后的调制系数"、只剩
                                 amplitude_scale×leak 的整体缩放——意志调控（第②层）
                                 大半失效，是个真缺陷，不是假设。两组系数均由解析回退
                                 产出（该函数已过议会设计门，真模型注入后由其输出替代——
                                 本模块目前无注入口，与 `motion_synth` 现状一致：热路径
                                 无 LLM/meta，给定 seed 完全确定性）
    scene: "idle" | "speaking"  时间结构分支标记，判定权归本 Agent（见 `_determine_scene`）
    events: list[dict]          离散行为意图（12 词闭集，`behavior_intent` 已实现，序列化为
                                 {name, intensity, direction, source}）
    prosody_ref: str | None     说话期韵律流引用（见 `_determine_scene`：目前恒 None）

三个已裁定的未决项（`design-agent.md` §6，不再讨论）：

1. **scene 判定权归本 Agent，不归 Supervisor**——`orchestration-rules.md` 定 Supervisor
   只做分发协调、不实现业务逻辑；scene 判定需读"本回合有无待发文本/语音"，属 Worker 职责。
   本 Agent 位于 `expression` 之后，天然读得到本回合表达结果。
2. **韵律流只接受实时 TTS 韵律**；🛑 不得实现"文本预估韵律"作为降级路径（Munhall
   头动-语音错配实验证明起作用的是与当下语音的精确锁相，文本时长预估是"较温和版本的
   同一种错配"）。TTS 接口未就绪 ⇒ speaking 分支保持未启用，见 `_determine_scene`。
3. **韵律流不进 `AffectState`，走引用**（`prosody_ref: str | None`）——引用本身只是个字符串，
   进 state 无妨；真正体量大的韵律时序数据将来才需要仿 `ConversationSession.last_affect_sample`
   的只读实例属性先例。当前无 TTS、无数据可引用，暂不新增该实例属性——避免留一个
   没有任何写入者的空壳字段。

节点契约：`(state) -> dict`，只返回增量，不原地 mutate（同其它 Worker）。前置字段缺失
（`affect_sample` 未产出）⇒ 返回 `{}`，不抛异常。门控用单一枚举 `state.motion_backend`
（"synth" 默认 / "directive" 启用本 Agent，CS 席要求避免多布尔组合爆炸）；默认值使
`zero.step` 逐字零回归——"synth" 时本 Agent 恒返回 `{}`，`motion_directive` 不出现在
任何返回体。本 Agent 不写记忆（记忆节流由 Supervisor 统一做）。

⚠ **已知未验证组合**（`_events` docstring细说）：`motion_backend="directive"` 与
`language_enabled=True` 同开时，events 依据图内 `LanguageAgent` 合成的文本路由行为意图，
但生产聊天路径实际发给用户的文本另有来源（`chat_driver.converse()`），两者当前对不上。
现状安全仅因为 `language_enabled` 默认关（events 恒空，`zero.motion` 正确回落 `reply_text`）。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from src.agents.behavior_intent import lexical_intents, merge_intents, stage_direction_intents
from src.agents.language_openai import strip_stage_directions_with_segments
from src.agents.motion_synth import Modulation, modulation_from_affect
from src.orchestration.state import AffectState


class MotionAgent:
    """按回合产出 `motion_directive`：调制系数（含 voluntary 专属一组）+ 场景标记 +
    离散行为意图 + 韵律引用。

    图内位置：`expression` 之后、`supervisor` 之前（动作是表达层产物，且 Supervisor
    是任务完成/记忆节流节点，应在最后）。仅在 `state.motion_backend == "directive"` 时
    产出；默认 `"synth"` 时本 Agent 是 no-op——`zero.motion` 拉取侧沿用现有做法：
    现场读 `session.last_affect()` 自算调制系数，不消费 `motion_directive`。
    """

    def __call__(self, state: AffectState) -> dict[str, Any]:
        if state.motion_backend != "directive":
            return {}
        affect_sample = state.affect_sample
        if affect_sample is None:
            return {}
        mod = modulation_from_affect(*affect_sample)
        regulated_mod = self._regulated_modulation(affect_sample, state.regulated_affect)
        scene = self._determine_scene(state)
        events = self._events(state)
        directive: dict[str, Any] = {
            "amplitude": mod.amplitude,
            "speed": mod.speed,
            "onset": mod.onset_sharpness,
            "regulated": self._serialize_modulation(regulated_mod),
            "scene": scene,
            "events": events,
            "prosody_ref": None,  # TTS 未接线；见 `_determine_scene` 阻塞说明
        }
        entry = {"node": "motion", "scene": scene, "n_events": len(events)}
        return {"motion_directive": directive, "trace": [entry]}

    def _regulated_modulation(
        self,
        affect_sample: tuple[float, float],
        regulated_affect: tuple[float, float] | None,
    ) -> Modulation | None:
        """voluntary（随意通路）专属调制系数——由 `regulated_affect` 而非 `affect_sample` 算出。

        🛑 2026-08-07 修复的真缺陷：此前 directive 只给 spontaneous 一组系数，
        `zero.motion` 把它同时喂给两路（`motion_synth.generate_dual` 的
        `modulation` 形参两路共用），voluntary 路因此丢掉"被调节后的调制系数"。
        `regulated_affect` 与调节旋钮（`voluntary_coping_leak`）都已在图内（`AffectState`），
        两条通路的调制系数理应都在图内算完——这正是"决策进图"的本意，不该留到拉取侧再补算。

        `regulated_affect` 缺失（未开调节）或与 `affect_sample` 相等（调节没改变什么）时
        返回 `None`——结构最小，`zero.motion` 据此回退复用顶层系数，与
        `motion_synth.generate_dual` 的 `modulation_voluntary=None` 零回归语义对齐。
        """
        if regulated_affect is None or regulated_affect == affect_sample:
            return None
        return modulation_from_affect(*regulated_affect)

    def _serialize_modulation(self, mod: Modulation | None) -> dict[str, float] | None:
        if mod is None:
            return None
        return {"amplitude": mod.amplitude, "speed": mod.speed, "onset": mod.onset_sharpness}

    def _determine_scene(self, state: AffectState) -> Literal["idle", "speaking"]:
        """判定本回合场景：`idle`（待机时间结构）或 `speaking`（言语驱动时间结构）。

        判定权归本 Agent，不归 Supervisor（议会 2026-08-06 第五轮裁定；`orchestration-rules.md`
        定 Supervisor 只做分发协调、不实现业务逻辑）。本 Agent 位于 `expression` 之后，
        天然读得到本回合有无待发文本（`state.language_text`，由 `LanguageAgent` 写入）。

        🛑 speaking 分支目前**恒不触发**——这是刻意的依赖阻塞，不是遗漏：speaking 场景的
        时间结构必须由**实时 TTS 韵律流**驱动（Munhall 头动-语音错配实验证明起作用的是
        与当下语音的精确锁相；文本预估韵律只是"较温和版本的同一种错配"，议会已明确否决
        作为降级路径，见 `design-agent.md` §2/§3/§6）。本仓当前**没有 TTS 管线接口**，无
        真实韵律流可读——故即使本回合确有待发文本（`has_pending_text` 为真），也不得判
        speaking：套用 idle 的时间结构却贴 speaking 标签，比诚实地留在 idle 更误导。
        TTS 接口就绪后，只需把 `tts_prosody_available` 换成真实可用性判断即可解除阻塞，
        判定权归属不必再动。
        """
        has_pending_text = state.language_text is not None
        tts_prosody_available = False  # 阻塞点：见上，TTS 管线未接线，不得降级
        if has_pending_text and tts_prosody_available:
            return "speaking"
        return "idle"

    def _events(self, state: AffectState) -> list[dict[str, Any]]:
        """③ 层离散行为意图：词法规则 + 舞台说明路由（`behavior_intent`，12 词闭集）。

        消费 `state.language_text`（本回合图内 `LanguageAgent` 产出，仅在
        `state.language_enabled=True` 时非 None）——与生产聊天路径（`chat_driver` 经
        `converse()` 走独立 LLM 调用、不经图内 `language` 节点）是**两条不同的文本来源**。
        `zero.motion` 现有 `reply_text` 参数覆盖后者，与本方法并存、优先级见其调用点注释。

        ⚠ **已知未验证的组合**（未经复核，非空壳提醒）：若某会话同时开
        `motion_backend="directive"` 与 `language_enabled=True`，本方法会依据**图内合成、
        永远不会真正说出口的文本**路由行为意图（可能触发 nod/shake 却对应另一段话）。
        当前之所以安全，纯粹是因为 `language_enabled` 默认 `False` ⇒ `state.language_text`
        恒 `None` ⇒ 本方法恒返回 `[]` ⇒ `zero.motion` 里 `if directive_events:` 判假、
        正确回落到 `reply_text` 那一路（见 `tests/test_motion_tool.py` 的回落边界测试）。
        这条组合一旦被打开就需要重新设计文本来源的合流点，不是本轮改动范围。
        """
        if not state.language_text:
            return []
        _, segments = strip_stage_directions_with_segments(state.language_text)
        lexical = lexical_intents(state.language_text)
        stage = stage_direction_intents(segments)
        merged = merge_intents(lexical, stage)
        return [asdict(intent) for intent in merged]
