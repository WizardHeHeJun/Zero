"""行为意图抽取：从回复文本推导「主管判断直驱」的离散行为（12 词闭集）。

对应设计文档 `PRP/motion/design.md` §0.5 的**第 ③ 层**——与情绪无关的、由判断直接决定的
动作：点头是因为**认同**（不是因为高兴），歪头是因为**疑惑**（不是因为低唤醒）。
第 ①/② 层（情绪直驱 / 意志调控）走连续轨迹，见 `motion_synth`。

两个意图来源（用户 2026-08-05 拍板「按复杂度分流、两者结合」）：

- **③a 确定性词法**：对已生成的回复做正则判定，覆盖高频小动作（肯定/否定/疑问/强调）。
  零新增 LLM 调用、确定性可单测。仓内先例：`supervisor` 的承诺标记与身份自陈判定同为
  「纯正则、无 LLM、可单测」。
- **③b 舞台说明路由**：模型本就自发产出「（点了点头）」这类括号动作（④臂 100 轮实测 33 条），
  阶段 63 起被 `language_openai.strip_stage_directions` **直接删除**。有了 Live2D 身体后，
  这批内容改为**先解析成行为意图**再从文本剥离——可见文本仍然干净，内容转而驱动形象。

🛑 **闭集白名单是安全边界，不是便利设施**：只有能映射进 `BEHAVIOR_VOCABULARY` 的才放行。
阶段 63 删舞台说明的根因是模型在**虚构自己没有的具身能力**；有了皮套，「点头」成为真能力，
但「我帮你关灯了」「我走过去看了看」**依然不是**——它们映射不进闭集，照旧被丢弃。
放宽这个闭集 = 让数字人重新开始宣称它做不到的事。

节流不在本模块：per-behavior 冷却与通道优先级仲裁由 MCP 执行侧负责（其已实现
reactive > deliberate > posture 与 250ms 全局最小间隔），我方**不重复实现**，
连珠炮会收到执行侧回执码，按回执退避即可。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# MCP 侧 12 词行为闭集（契约真相在对面 `vts_behavior.py`；此处按名字引用，勿自造新词）。
BEHAVIOR_VOCABULARY: frozenset[str] = frozenset(
    {
        "nod",
        "shake",
        "head_tilt",
        "glance",
        "blink",
        "brow_raise",
        "brow_furrow",
        "eyes_widen",
        "smile",
        "lean_in",
        "lean_back",
        "body_sway",
    }
)


@dataclass(frozen=True)
class BehaviorIntent:
    """一条待投递的行为意图。字段与 MCP `behavior_trigger` 的入参对齐。

    Attributes:
        name: 12 词之一（保证在 `BEHAVIOR_VOCABULARY` 内）。
        intensity: 0–1 幅度。
        direction: 需要方向的行为（`head_tilt`/`glance`）才有值，否则 None。
        source: `"lexical"`（③a）/ `"stage"`（③b）/ `"deliberate"`（③c 上游直达），
            供观测与调参归因，不影响执行。
    """

    name: str
    intensity: float = 0.5
    direction: str | None = None
    source: str = "lexical"


# ── ③a 词法规则 ────────────────────────────────────────────────────────────
# 取向「宁漏勿误」，与 strip_stage_directions 一致：漏一个点头是瑕疵，
# 在否定句上点头是**语义矛盾**，比不动更糟。故每条规则都要求明确的词形证据。

# 否定优先于肯定：「不是吧」「没有」含「是/有」，若先判肯定会反向。顺序在 _LEXICAL_RULES 里固定。
_NEGATION_RE = re.compile(r"不是|不对|不行|不能|没有|并非|恐怕不|我不|别这么|不至于")
_AFFIRMATION_RE = re.compile(
    r"对(?:啊|呀|的)?[，。！]|没错|是这样|确实|对头|嗯[，。]|好的|明白|懂了|同意"
)
_QUESTION_RE = re.compile(r"[?？]|吗[。？]?$|呢[。？]?$|难道|为什么|怎么会")
_EMPHASIS_RE = re.compile(r"[!！]|真的|一定|绝对|千万|必须")


def lexical_intents(reply: str) -> list[BehaviorIntent]:
    """③a：从回复文本按词法推导行为意图。纯正则、无 LLM、确定性。

    判定顺序有意义——**否定先于肯定**：「不是吧」「没有」等否定串里含「是/有」，
    先判肯定会在否定句上点头，属语义反向错误（比不动更糟）。

    每类最多产出一条，避免一句话里塞满动作；同一回复最多 2 条（`_MAX_LEXICAL`）。
    """
    if not reply.strip():
        return []
    intents: list[BehaviorIntent] = []
    if _NEGATION_RE.search(reply):
        intents.append(BehaviorIntent("shake", intensity=0.55, source="lexical"))
    elif _AFFIRMATION_RE.search(reply):
        intents.append(BehaviorIntent("nod", intensity=0.5, source="lexical"))
    if _QUESTION_RE.search(reply):
        intents.append(
            BehaviorIntent("head_tilt", intensity=0.45, direction="left", source="lexical")
        )
    elif _EMPHASIS_RE.search(reply):
        intents.append(BehaviorIntent("brow_raise", intensity=0.5, source="lexical"))
    return intents[:_MAX_LEXICAL]


_MAX_LEXICAL = 2


# ── ③c 非文本意图源（行为反馈环第一步·缺口 B）────────────────────────────────
# 上游经 state_overrides 直接下达的行为意图（「先决定做个动作」，不从已生成文本反推）。
# 与 ③a/③b 的关键差异：调用方是**代码**不是模型——静默丢弃会藏调用方 bug，
# 故非闭集名在此 fail-fast（external_priors 的 M3/M6 先例：错误指向传参方，改传参就能好）。
# ③b 对模型产出的舞台说明仍保持静默丢弃（那边「丢弃是预期行为」，见 stage_direction_intents）。


def deliberate_behavior_intents(payload: list[dict[str, object]]) -> list[BehaviorIntent]:
    """③c：解析并校验上游下达的行为意图，产出 `source="deliberate"` 的 `BehaviorIntent`。

    Args:
        payload: `[{name, intensity?, direction?}]`（`AffectState.deliberate_intents` 原样）。

    Returns:
        校验通过的意图列表，顺序保持输入序。

    Raises:
        ValueError: name 不在 12 词闭集，或 intensity 出 [0,1]——fail-fast 指向调用方传参
            错误。闭集是安全边界：放行「关灯」「走过去」等物理世界宣称 = 让数字人重新
            宣称它做不到的事（模块 docstring）。
    """
    intents: list[BehaviorIntent] = []
    for item in payload:
        name = item.get("name")
        if not isinstance(name, str) or name not in BEHAVIOR_VOCABULARY:
            raise ValueError(
                f"deliberate 行为意图 name={name!r} 不在 12 词闭集 "
                f"{sorted(BEHAVIOR_VOCABULARY)}（闭集是安全边界，调用方传参错误）"
            )
        intensity_raw = item.get("intensity", 0.5)
        if not isinstance(intensity_raw, (int, float)) or not 0.0 <= float(intensity_raw) <= 1.0:
            raise ValueError(
                f"deliberate 行为意图 {name!r} 的 intensity={intensity_raw!r} 须在 [0,1]"
            )
        direction_raw = item.get("direction")
        if direction_raw is not None and not isinstance(direction_raw, str):
            raise ValueError(
                f"deliberate 行为意图 {name!r} 的 direction={direction_raw!r} 须为 str 或缺省"
            )
        intents.append(
            BehaviorIntent(
                name,
                intensity=float(intensity_raw),
                direction=direction_raw,
                source="deliberate",
            )
        )
    return intents


# ── ③b 舞台说明 → 行为闭集映射 ──────────────────────────────────────────────
# 键是 strip_stage_directions 会剥掉的那类动作/神态词；值是 12 词之一。
# ⚠ 只列**皮套真能做**的。「关灯」「走过去」「递给你」等对物理世界的行动宣称**刻意不列**，
# 映射不中即丢弃 —— 这正是阶段 63 那条边界的执行机制，不是遗漏。
_STAGE_TO_BEHAVIOR: tuple[tuple[re.Pattern[str], str, float], ...] = (
    (re.compile(r"点头|点了点头|颔首"), "nod", 0.6),
    (re.compile(r"摇头|摇了摇头"), "shake", 0.6),
    (re.compile(r"歪头|偏头|歪了歪"), "head_tilt", 0.5),
    (re.compile(r"挑眉|扬眉|挑了挑眉"), "brow_raise", 0.6),
    (re.compile(r"皱眉|蹙眉|皱了皱"), "brow_furrow", 0.6),
    (re.compile(r"瞪大|睁大|愣住|一愣"), "eyes_widen", 0.65),
    (re.compile(r"笑|莞尔|嘴角(?:上扬|一挑)"), "smile", 0.55),
    (re.compile(r"凑近|前倾|靠近|探身"), "lean_in", 0.5),
    (re.compile(r"后仰|往后|退开|靠回"), "lean_back", 0.5),
    (re.compile(r"看向|望向|瞥|移开(?:目光|视线)|转头看"), "glance", 0.45),
    (re.compile(r"眨(?:了)?(?:眨)?眼"), "blink", 0.5),
    (re.compile(r"晃|摇晃|轻晃|摆动"), "body_sway", 0.4),
)


def stage_direction_intents(segments: list[str]) -> list[BehaviorIntent]:
    """③b：把剥离下来的舞台说明片段映射成行为意图；映射不中的**丢弃**。

    Args:
        segments: `strip_stage_directions` 剥掉的括号内文本（不含括号）。

    Returns:
        映射命中的行为意图，按输入顺序、去重（同名只留第一条）。
        ⚠ 丢弃是**预期行为**不是失败：「我帮你关灯了」这类物理世界宣称映射不进闭集，
        必须继续被丢掉——见模块 docstring 的安全边界说明。
    """
    intents: list[BehaviorIntent] = []
    seen: set[str] = set()
    for segment in segments:
        for pattern, name, intensity in _STAGE_TO_BEHAVIOR:
            if pattern.search(segment) and name not in seen:
                seen.add(name)
                direction = "left" if name in ("head_tilt", "glance") else None
                intents.append(
                    BehaviorIntent(name, intensity=intensity, direction=direction, source="stage")
                )
                break  # 一个片段只出一个行为，避免「笑着点头」出两条抢同一通道
    return intents


def merge_intents(
    lexical: list[BehaviorIntent],
    stage: list[BehaviorIntent],
    *,
    deliberate: list[BehaviorIntent] | None = None,
    limit: int = 3,
) -> list[BehaviorIntent]:
    """合并各路意图并去重，优先级 **deliberate > stage > lexical**——上游显式指令最可信，
    其次模型的显式表达意图（舞台说明），词法推断垫底。

    `deliberate` 缺省 None＝旧两路行为逐字不变（向后兼容）。节流交 MCP 执行侧
    （冷却/通道仲裁已在对面实现）；此处只做同名去重与条数上限，防单轮投递过多。
    """
    merged: list[BehaviorIntent] = []
    seen: set[str] = set()
    for intent in [*(deliberate or []), *stage, *lexical]:
        if intent.name in seen:
            continue
        if intent.name not in BEHAVIOR_VOCABULARY:
            continue  # 闭集守卫：非 12 词一律不放行（防有人往映射表里加新词绕过边界）
        seen.add(intent.name)
        merged.append(intent)
    return merged[:limit]
