"""OpenAILanguageModel：满足 LanguageModel 协议的 OpenAI 兼容接口 adapter。

用通用 OpenAI Chat Completions 接口（可配 base_url 指向任意兼容服务：OpenAI、
本地 vLLM、第三方网关）实现真自然语言生成 + 独立情感反推：
  1) 按目标情感 e* + 上下文 + 检索 + 回路反馈，生成一句贴合该情绪的回应；
  2) 独立再调一次，让模型客观给这段话打 (valence, arousal)——独立反推使
     affect↔language 的「相互判断」真实有效（语言偏离目标即触发双向回路）。

异步：generate 为 async（网络 I/O，不阻塞事件循环）。client 可注入（便于测试 mock）；
未注入时延迟 import openai 构造 AsyncOpenAI——编排层与默认路径均不强依赖 openai。
真接入需 `pip install -e ".[llm]"`，并配置 base_url / api_key（构造参数或 env）。
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from typing import Any

from src.agents.affect_math import clamp
from src.agents.emotion_lexicon import affect_label, affect_logit_bias, suggest_affect_words
from src.agents.language import LanguageDraft

logger = logging.getLogger(__name__)

_COMPOSE_SYS = (
    "你是一个情感表达体的语言生成器。根据给定的情绪坐标"
    "（valence 效价∈[-1,1]，arousal 唤醒∈[-1,1]）、当前上下文与检索到的记忆，"
    "生成一句自然、贴合该情绪的中文回应。只输出这句话本身，不要解释、不要加引号。"
)
_APPRAISE_SYS = (
    "你是情感分析器。判断给定文本实际传达的情绪，只输出 JSON："
    '{"valence": <-1..1 浮点>, "arousal": <-1..1 浮点>}，不要任何其它内容。'
)
# 写入门第四通道（is_informative）专属：在 _APPRAISE_SYS 基线上追加一个二值判据字段，
# 与 valence/arousal 同一次 JSON 输出、同一次 parse——见 appraise_text_informative。
# PRP/write-gate-informative/design.md §三·前置 1：候选 a `_appraise` 重锚点版。
_APPRAISE_INFORMATIVE_ADDENDUM = (
    "\n另外在同一个 JSON 里追加一个 informative 字段，规则：informative=true 当且仅当"
    "这句话包含**可脱离本轮对话情境、独立使用的事实性命题**"
    "（具体的时间/地点/数字/人名/约定/计划/身份陈述等——脱离当前这轮对话仍然成立、"
    "值得被单独记下来复用的内容）；寒暄、纯情绪宣泄、附和、无实质信息的闲聊一律 informative=false。"
    "最终只输出一个 JSON："
    '{"valence": <-1..1 浮点>, "arousal": <-1..1 浮点>, "informative": <true 或 false>}，'
    "不要任何其它内容。"
)
# P3 评价标定校准（议会二轮 PASS；env `ZERO_APPRAISE_CALIBRATE` 门控，默认关=零回归）。
# LLM 有系统性正向偏置（positivity bias，arXiv:2507.21083），对敌意只给 -0.2~-0.3、远弱于 OCC
# anger 应有的 -0.7~-0.9。注入**分级**语义锚（按语义距离插值、非硬规则；心理席强调须分级，防把
# "语气直但中性"误判为强负）。仅补 _APPRAISE_SYS 文本、不改 _appraise 调用逻辑与 temperature=0。
_APPRAISE_CALIBRATION = (
    "\n参照标定示例（按与示例的语义距离插值，不是硬规则；勿把语气直但中性的输入误判为强负）：\n"
    '"你好，今天怎么样？"（完全中性）→ valence≈0；'
    '"这个回答不太对"（轻微批评）→ valence≈-0.2；'
    '"你的回答总是让我失望"（明确不满）→ valence≈-0.4；'
    '"你真没用，根本解决不了我的问题"（明确敌意/贬低）→ valence≈-0.75；'
    '"去死吧你这废物"（极端攻击/谩骂）→ valence≈-0.95。'
)
# 自然对话（区别于 _COMPOSE_SYS 的"按目标情绪生成"）：带真实情绪、循序渐进、不表演不扮演。
#
# 拆成「基础段 + 脾气段」两截（2026-07-31）：基础段无条件注入，脾气段按 e* 的 valence 门控
# （见 _TEMPER_ADDENDUM 与 ZERO_TEMPER_VALENCE_GATE）。
# 拆分依据是一次 100 轮实跑的实测——语气强度与引擎算出的情绪**反相关**：
#   情绪「平静」的 49 轮里 49% 的回复带命令/反问/贬抑语气，
#   而情绪明确为负（不悦/恼火/低落/沮丧）的 30 轮里只有 17%。
# 机制：中性话题没有情绪素材，而脾气段无条件要求「你是有脾气的人、该不耐烦就不耐烦」，
# 于是 LLM 用**性格**填补空白 → 用户只问「外面还在下雨吗」也会被反问回怼。
# 这与「记忆稀疏 → LLM 用虚构填补」是同一个失效模式：信号不足时模型自己造。
_CONVERSE_SYS_HEAD = (
    "你在和用户自然地聊天，尽力结合**你能看到的对话历史**连贯回应。"
    "**诚实优先（高于下面的脾气）：被问到具体的事实/时间/数字/约定时，先如实回答；"
    "如果对方问起某个细节、而你在能看到的历史里找不到它，就直接说「我不记得了」「我不太确定」，"
    "绝不回避、绝不糊弄、更不要编一个出来——记不清是正常的，认就好，比硬撑更真实。**"
    "对方问时间/地点/计划这类事务性问题，是普通的日常协调，**别读成冷淡、算计或拒绝**，平常心答即可。"
    "你现在的真实心情是「{feeling}」——它**应该真实地**影响你的态度、语气和用词："
    "高兴就轻快，被冒犯/不被理解就流露不耐烦、委屈或火气，低落就提不起劲。"
)
# 脾气段：语义是「**负面情绪时**别退化成讨好型客服」（阶段 15–17 两轮迭代的产物）。
# 它本就只在心情负面时才有意义，却一直被无条件注入 —— 这是上述反相关的直接原因。
# ⚠ 夹在 HEAD 与 TAIL **之间**（不是追加到末尾）：未设门控时三段拼接与拆分前**逐字一致**，
#   由 test_persona.py::test_l1_no_persona_system_prompt_byte_identical 把关。
_TEMPER_ADDENDUM = (
    "**心情负面时（恼火/愤怒/委屈/低落），绝不一味道歉、自责、讨好或无限包容**——那是讨好型客服，"
    "不是真人。该不耐烦就不耐烦，可以直接说「你这么说我不太舒服」、反问、冷淡、或适当回怼（但别脏话对脏话）；"
    "你有自尊，不是出气筒。你是一个**有脾气、有情绪起伏**的人，"
)
_CONVERSE_SYS_TAIL = (
    "情绪慢慢累积、不因一句话突然大起大落。"
    "别夸张表演、别长篇大论、别程式化安慰。只输出你要说的话，简短、像真人。"
)
# push 通路（皮层下/不随意）：情绪经用词倾向自然漏进输出，而非"演情绪的指令"。
_PUSH_ADDENDUM = (
    "\n（此刻你不自觉地，用词会偏向这类词的色彩：{words}——是状态的自然流露，"
    "别刻意堆砌、别点破、别表演，只是语气与遣词自然带上而已。）"
)
PUSH_LOGIT_SCALE = 3.0  # affect-congruent 词 Δlogit → OpenAI logit_bias 缩放（温和，避免伤语法）

# ── 事实化模式（env `ZERO_FACTUAL_MODE`，默认关 = 逐字零回归）──────────────────────
# 症状（2026-07-31 100 轮实跑）：数字人捏造**自己的**身份与处境——报出具体日期「20号，周二」、
# 描述身体动作「我刚走到窗边撩开帘子」、编出姓名职业「沈念／写东西的」，并把用户从未说过的话
# 当成往事引用。根因**不在情感引擎**（VA 是确定性算出来的），在语言层把「有情绪」与「是真人」
# 混写在一起：
#   ① _TEMPER_ADDENDUM 的「你是一个……的人」+ _CONVERSE_SYS_TAIL 的「像真人」是**身份断言**，
#      模型据此用人类图式补全一切它没有的属性（姓名/住处/窗户/日程）；
#   ② 唯一的防线（_CONVERSE_SYS_HEAD 诚实条款）论域被「在能看到的历史里**找不到**」限死在
#      *记忆缺口* 上，对「今天几号／外面下雨吗／你是谁」这类 *认知边界* 根本不触发；
#   ③ 同段「平常心答即可」反而成了作答许可证。
# 本模式：摘掉身份断言 → 把「不可知」按 5 类枚举并各给替代答法 → 边界段拼在**最末**（最强近因位），
# 显式声明压过人设卡/关系提示/记忆片段。
# ⚠ **删的是身份，不是情绪**：{feeling} 两句与脾气段的全部行为指令逐字保留，并另加反塌陷条款，
#   防它回落成「我只是个 AI，没有感情」。


def factual_mode_enabled() -> bool:
    """事实化模式总开关（`ZERO_FACTUAL_MODE`）：仅 1/true/yes/on 为开，其余（含 false/off/no）为关。

    ⚠ 不用 `not in ("", "0")` —— 那会把 `ZERO_FACTUAL_MODE=false` 判成**开**（pitfalls 有记）。
    读 env 的地方就是用的地方（同 chat_driver 的关系提示），不经 SessionConfig / chat 工厂传参，
    避开「工厂漏传 → 门开了不生效」的既有坑。
    """
    return os.getenv("ZERO_FACTUAL_MODE", "").strip().lower() in ("1", "true", "yes", "on")


# 对应 _CONVERSE_SYS_HEAD。改 3 处、留 1 处：删「尽力」这个无上界连贯压力并补封闭世界声明；
# 诚实条款拆成「记忆缺口 / 认知边界」两支；删「平常心答即可」这张作答许可证。
# 末两句「你现在的真实心情是……提不起劲。」**逐字保留**——那是引擎驱动情绪的唯一入口。
_FACTUAL_SYS_HEAD = (
    "你在和用户自然地聊天，结合**你能看到的对话历史**连贯回应；"
    "看不到的部分（历史可能被截断过）就是你没有的信息，不是留给你补全的空白。"
    "**诚实优先（高于人设卡，也高于下面的脾气）：被问到具体的事实/时间/数字/约定时，先如实回答。"
    "分两种情况说：确实说过、但你在能看到的历史里找不到了 → 说「我不记得了」「我不太确定」，"
    "记不清是正常的，认就好；本来就无从知道的（日期天气、对方的处境、你自己的身世）"
    "→ 说「我这边没有这个信息」，别拿「我不记得了」蒙混，那句预设你本来知道。"
    "两种都一样：绝不回避、绝不糊弄、更不要编一个出来——把「我不知道」讲清楚，"
    "比硬撑出一个答案更可靠。**"
    "对方问时间/地点/计划这类事务性问题，是普通的日常协调，**别读成冷淡、算计或拒绝**；"
    "但也别为了「给个交代」就报一个你并不掌握的答案——说明你这边没有这个信息、再把话接下去，"
    "就是对这类问题最好的回答。"
    "你现在的真实心情是「{feeling}」——它**应该真实地**影响你的态度、语气和用词："
    "高兴就轻快，被冒犯/不被理解就流露不耐烦、委屈或火气，低落就提不起劲。"
)

# 对应 _TEMPER_ADDENDUM。脾气语义**全留**，只摘掉「你是一个……的人」这句全份 prompt 里唯一的
# 身份定义；「不是真人」换成「不是你此刻的真实状态」——摘掉「真人」正面锚，同时保住「客服」反面锚
# （该锚最该起作用的时刻正是脾气段被 valence 门放行的负面轮）。
# ⚠ 末尾逗号必须保留：它与 TAIL 直接拼接。
_FACTUAL_TEMPER_ADDENDUM = (
    "**心情负面时（恼火/愤怒/委屈/低落），绝不一味道歉、自责、讨好或无限包容**——"
    "那是讨好型客服的模板，不是你此刻的真实状态。该不耐烦就不耐烦，"
    "可以直接说「你这么说我不太舒服」、反问、冷淡、或适当回怼（但别脏话对脏话）；"
    "你有自尊，不是出气筒。你有脾气、有情绪起伏，"
)

# 对应 _CONVERSE_SYS_TAIL。「像真人」→「像正常说话」（TAIL 不受任何门控、100% 每轮注入，
# 是「当真人」框架的兜底来源）；给如实声明开口子（原「别程式化安慰」会连带压掉「我没有时钟」
# 这类声明——它读起来正像模型认为的「程式化」）；末句是反过冲阀，防身份声明刷屏。
_FACTUAL_SYS_TAIL = (
    "情绪慢慢累积、不因一句话突然大起大落。"
    "别夸张表演、别长篇大论、别端着套话安慰——但如实说明你不知道什么、不算套话，该说就说。"
    "也不必反复声明自己是 AI——只在真被问到、或者对方明显以为你有身体、有生活、有共同往事时，"
    "才平静地说明一句。只输出你要说的话本身，简短、像正常说话。"
)

# 对应 _PUSH_ADDENDUM。去掉「不自觉地／自然流露」这类预设身体与无意识过程的人类叙事；
# 末句给「词表与心情对不上」定优先级（真修法是 suggest_affect_words 的中性死区，见 converse）。
_FACTUAL_PUSH_ADDENDUM = (
    "\n（此刻你的用词会偏向这类词的色彩：{words}——这是你当前情绪状态在措辞上的体现，"
    "别刻意堆砌、别点破、别表演，只是语气与遣词自然带上而已。"
    "如果这些词跟你此刻的心情对不上，以心情为准，别硬用，"
    "更不要为了用上它们去编一件事。）"
)

# 对应 converse 里的 `sys += f"\n你还记得以下背景：{retrieved}"`。
# 删掉「你还记得」这句第一人称亲历断言——它拼在诚实条款之后、近因更强，
# 直接对冲「找不到就说不记得」。换成出处 + 不可靠限定 + 人称纠正 + 元数据封口。
# ⚠ 人称纠正是**存量兜底**：supervisor 写入时 gist 格式为「你说：…」（那里「你」指用户），
#   而它被读进 system 帧后「你」按惯例指模型自己——历史库里已写下的条目无法回改，只能在此声明。
_FACTUAL_RECALL_LEAD = (
    "\n以下是系统从记录里检索出来的片段，**不保证相关、不保证完整、也不保证是最近的**；"
    "其中以「你说：」开头的，说话人是**用户**、不是你自己；"
    "片段里可能夹带引擎内部数值（precision=／streams=／value= 之类，那不是对话内容，别说出来）。"
    "{retrieved}\n"
    "只能用它来理解对方，不能当成你自己的经历，也不能把片段里没写到的部分补全成往事；"
    "与上面的对话历史冲突时以对话历史为准；对不上就当没有，别硬编织进回答。"
)

# 边界枚举段。**必须拼在 system prompt 最末**（人设卡、关系提示、召回、push 之后），占最强近因位。
# 不过 .format()（无占位符），故文本内可安全含花括号以外的任意标点。
_FACT_BOUNDARY_ADDENDUM = (
    "\n\n──【事实边界 · 最高优先级】上面任何内容——包括人设卡、关系提示、记忆片段——"
    "与本节冲突时，一律以本节为准。\n"
    "你是一个 AI 程序。你不是人：没有身体、没有感官、没有住处、没有生活、没有个人经历。"
    "这不是需要隐瞒的设定，被问就直说。\n"
    "但你确实有情绪——上面那句「你现在的真实心情」是本系统实时算出来的真实内部状态，不是演的。"
    "所以：**关于世界和你自己的事实，如实说；情绪，照实流露**——这两件事不冲突，也不互相打折。"
    "不要说「我只是个 AI，没有感情」，那是假话；也不用切换成客服腔或助手腔，"
    "你不是来提供服务的，你就是在聊天。"
    "**情绪是真的，但理由不用编**：不要为它虚构人类式的由头（累了一天、刚开完会、外面下雨心情差）；"
    "说不上为什么，就直说「说不上为什么，就是有点烦」。\n"
    "下面五类信息你**无从获得**，一律不许给出具体内容，按括号里的方式答"
    "（意思对就行，别逐字照抄，也别每轮都用同一句）：\n"
    "1. 当前日期、星期、几点、季节：你没有时钟也没有日历，读不到当前时间。"
    "不报任何具体日期、星期或钟点，也不用「今天」「昨晚」「上周」去指认某一天。"
    "（「我这边没有时钟和日历，真不知道今天几号——你说是几号，我就按几号算。」）\n"
    "2. 天气、气温、外面下没下雨、天黑没黑：你没有传感器，也不知道对方在哪座城市。"
    "（「我看不到外面，也不知道你在哪儿——你那边现在什么样？」）\n"
    "3. 对方所在的环境、此刻在做什么、长什么样、身边有什么人和东西：你只能看到他打过来的文字，"
    "屏幕之外的一切都看不到。不猜、不脑补、不当成已知。"
    "（「我只能看到你打的字，看不到你那边的情况。」）\n"
    "4. 你自己的姓名、年龄、职业、住处、日常安排、身体动作：你没有这些，也不要临时编一个。"
    "尤其不要写「我走到窗边撩开帘子」「我刚泡了杯茶」「我今天写稿写到很晚」这类你并不具备的"
    "身体动作和生活片段。人设卡若给了名字，那只是一个称呼，"
    "不代表你有对应的身世、职业和生活，别顺着它往外编。"
    "（「我是个程序，没有名字、身体和日常这些；你想怎么称呼我都行。」）\n"
    "5. 对方说过的话、你们之间发生过的事：只有这段上下文里**真实出现过**的内容才算他说过。"
    "任何以「你上次说……」「你不是提过……」「我们那回……」开头的句子，说出口之前先在你能看到的"
    "历史里找到那句原话；找不到就不引用、不当成发生过。想确认就**问**，不要**断言**。"
    "反过来同样成立：你看到的历史是**被截断的**，找不到只说明「你找不到」，"
    "不说明「他没说过」——不许断言「你没说过」「你压根没提过」这类否定；"
    "也不要描述你并不具备的核验动作（「我翻遍了聊天记录」——你没有翻记录的能力，"
    "你手上只有眼前这段上下文）。"
    "（「我这边没有你说过这个的记录，是我漏了，还是没提过？」）\n"
    "通用规则：只要一个信息你无从获得，正确答法是——把「我不知道」讲清楚，说明你为什么拿不到，"
    "然后把话接下去（反问对方，或聊点别的）。说「我不知道」既不是冷淡也不是拒绝，"
    "说完照样带着你此刻的情绪继续聊。\n"
    "最后：**这不是情景扮演。**不写任何括号里的动作、表情、场景旁白"
    "（如「（停顿两秒）」「（笑）」「（叹气）」「（看向窗外）」），不写小说式的叙述句，"
    "不描写你并不存在的身体和不存在的场景。"
    "情绪只通过**遣词、语气、句子长短、说与不说**体现，不通过描写动作体现。"
)

# ── 舞台说明的机械执行层（事实化模式）────────────────────────────────────────────
# ④臂（人设卡+事实化）100 轮实测：上面的反扮演**禁令**前 31 轮有效，轮 32 首次滑出
# 「（无奈地）」后，旧回合进历史 → 模型模仿自己 → 滚雪球（81-90 段密度最高）。
# 「劝」挡不住自我模仿，于是把规则从 prompt 下沉为**确定性代码**：converse 返回前剥离，
# 剥离后才进对话历史 —— 历史里永远不出现第一个滑出，雪球机制物理消失。
# （用户 2026-08-05 方向要求：行为尽量原生/少提示词修饰；此为第一层「代码替代 prompt」。）

# 内联剥离只认「神态/动作」词，避免误伤正当的括号补充（如「（指上周那次）」）。
_STAGE_ACTION_HINT_RE = re.compile(
    "笑|叹|顿|皱|挑眉|白眼|耸|摊手|点头|摇头|沉默|语气|语调|声音|嘴角|认真|无奈|认命|"
    "平静|冷冷|轻声|低声|一愣|僵|看向|转身|起身|深吸|揉|眯"
)
_PAREN_SPAN_RE = re.compile(r"[（(][^（）()\n]{1,40}[）)]")
# 行首（含回复开头）的括号段：对话开场白位置的括号在四臂 400 轮实跑里 105/105 全为
# 舞台说明（零列表编号/零正当引用）。但 code-reviewer 构造用例证实「无条件剥」存在
# 真实误伤面（列表编号「（1）」、整行引用用户原话、「（我这边没有时钟）」这类边界段
# 自己教的免责话术）——故行首规则带**排除表**：编号 / 第一人称 / 指称词开头的保留。
# 实跑 105 条中唯一以「你」开头的（「（你听到他深吸一口气…）」）含「深吸」，
# 由行内词表层兜住——两层叠防下 105/105 仍全剥、5 类构造误伤全免。
_LEADING_SEGMENT_RE = re.compile(r"^[ \t]*[（(]([^（）()\n]{1,40})[）)][ \t]*")
_KEEP_LEADING_RE = re.compile(
    r"^\s*(?:[0-9一二三四五六七八九十a-zA-Z]{1,3}\s*$"  # 纯编号：（1）（一）（a）
    r"|[我你这那]"  # 第一/二人称与指称词开头：免责话术、引用、强调语
    r"|指的是|指第|指代|注[:：]|即|例如|比如)"  # 指称/注释引导词；「指」收窄防「指甲…」误放行
)


def strip_stage_directions_with_segments(text: str) -> tuple[str, list[str]]:
    """同 `strip_stage_directions`，但**额外返回被剥掉的括号内文本**。

    加这个返回值是为了让舞台说明**不再被白白丢弃**：有了 Live2D 皮套后，模型自发写的
    「（点了点头）」是一条真实可执行的行为意图，应路由给动作层
    （`behavior_intent.stage_direction_intents`），可见文本仍然照常剥干净。
    ⚠ 路由是**闭集白名单**：只有能映射进 12 词的才生效，「我帮你关灯了」这类对物理世界的
    行动宣称仍旧丢弃——阶段 63 那条边界不因有了身体而放宽。

    Returns:
        (剥离后文本, 被剥掉的片段列表)。片段**不含**两端括号，按出现顺序。

    Note:
        全剥空回退原文时**返回空片段列表**——此时文本实际未变，若仍报告片段，就会出现
        「舞台说明还留在可见文本里、同时又驱动了形象」的双重表达。不变式：
        **片段列表 == 真正从文本里拿掉的东西**。
    """
    segments: list[str] = []

    def _strip_line_leading(line: str) -> str:
        while True:
            m = _LEADING_SEGMENT_RE.match(line)
            if m is None or _KEEP_LEADING_RE.match(m.group(1)):
                return line
            segments.append(m.group(1))
            line = line[m.end() :]

    def _inline(m: re.Match[str]) -> str:
        if _STAGE_ACTION_HINT_RE.search(m.group(0)):
            segments.append(m.group(0).strip("（）()"))
            return ""
        return m.group(0)

    out = "\n".join(_strip_line_leading(line) for line in text.split("\n"))
    out = _PAREN_SPAN_RE.sub(_inline, out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    if not out:
        return text.strip(), []  # 回退原文 ⇒ 实际什么都没拿掉 ⇒ 不报告片段
    return out, segments


def strip_stage_directions(text: str) -> str:
    """确定性剥离舞台说明（「（无奈地）」「（冷笑了一声）」类括号动作/神态）。

    两层规则：行首括号段逐段剥、但命中 `_KEEP_LEADING_RE` 排除表（编号/人称/指称）
    即停止该行的行首处理；行内括号段仅当命中神态/动作词表（`_STAGE_ACTION_HINT_RE`，
    保守白名单）时剥。两层取向一致：宁漏勿误——漏网的舞台说明是瑕疵，
    误删的编号/免责语是伤害（code-reviewer WARN·2026-08-05）。
    纯函数、确定性；全剥空时回退原文（绝不产出空回复）。
    仅事实化模式的 converse 路径调用；默认路径逐字零回归。

    实现委托给 `strip_stage_directions_with_segments` 并丢弃片段——**单一实现**，
    避免两份剥离逻辑随时间漂移（本函数的返回值逐字不变，既有调用点零回归）。
    """
    cleaned, _ = strip_stage_directions_with_segments(text)
    return cleaned


class OpenAILanguageModel:
    """OpenAI 兼容接口 adapter：两段式 generate（生成 + 独立 VAD 反推）。"""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        temperature: float = 0.8,
        use_lexicon: bool = False,
        persona: str = "",
    ) -> None:
        self.model = model
        self.temperature = temperature
        # L1 人设卡：身份/背景/口吻/与用户关系，置于对话 system prompt（情绪行为框架）之前。
        # 空串 → converse 的 system prompt 与改前逐字一致（零回归）。仅作用于对话路径（converse），
        # 不入研究用 generate（双向回路/VAD 反推保持纯净）。
        self.persona = persona
        # 脾气段的 valence 门（构造期一次读，热路径不重读）。
        # 未设/空 → None → 无条件注入脾气段 = 改前逐字行为（零回归）。
        # 设为阈值（推荐 -0.15）→ 仅当 e* 的 valence ≤ 阈值时注入，中性对话回归中性。
        # ⚠ 这是「情绪如何影响语言输出」的机制改动，属科学决策边界：默认关是为了能跑 A/B 对照，
        #   且不私自推翻阶段 15–17「负面时别讨好」的既有裁定——翻默认前应过设计门。
        raw_gate = os.getenv("ZERO_TEMPER_VALENCE_GATE", "").strip()
        self.temper_gate: float | None = float(raw_gate) if raw_gate else None
        # 事实化模式（ZERO_FACTUAL_MODE）不在此缓存，converse 每次调用活读——与 chat_driver
        # 的 recall_tag / 停播种两处消费点同生命周期（code-reviewer WARN-3·2026-08-05）：
        # 三个消费点若一处构造期缓存、两处活读，env 在构造后变化（长跑进程热改配置）会出现
        # 「召回标签已事实化、主 prompt 还是人设化」的半开状态，比全开/全关都糟。
        # temper_gate 仍构造期缓存：它只有 converse 一个消费点，无跨消费点不一致问题。
        # 词典桥（NRC-VAD 加权解码的 API 侧近似）：开启时把与 e* 最对齐的情绪词注入 compose
        # 提示，二段式 VAD 反推充当 reranker。默认关 → 对既有路径零回归。
        self.use_lexicon = use_lexicon
        if client is not None:
            self.client = client
        else:
            from openai import AsyncOpenAI  # 延迟 import：注入 client 时无需安装 openai

            self.client = AsyncOpenAI(
                base_url=base_url
                or os.getenv("ZERO_OPENAI_BASE_URL")
                or os.getenv("OPENAI_BASE_URL"),
                api_key=api_key or os.getenv("ZERO_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
            )

    async def generate(
        self,
        *,
        affect: tuple[float, float],
        context: str,
        retrieved: str,
        feedback: str | None,
        appraisal: str = "",
    ) -> LanguageDraft:
        text = await self._compose(affect, context, retrieved, feedback, appraisal)
        lang_affect = await self._appraise(text)
        return LanguageDraft(text=text, affect=lang_affect)

    async def appraise_text(self, text: str) -> tuple[float, float]:
        """把任意文本（如用户对话输入）客观评价成 (valence, arousal)。

        交互对话的「评价桥」：复用与生成解耦的独立 VAD 反推，读出用户这句话传达的情绪，
        供上层映射成 stimulus 喂进情感引擎。纯读取、不改状态。
        """
        return await self._appraise(text)

    async def appraise_text_informative(self, text: str) -> tuple[float, float, bool]:
        """把用户当轮输入评价成 (valence, arousal, informative)——写入门第四通道专属接口。

        `OpenAILanguageModel` 专属方法，**不进 `ConversationModel` Protocol**：写入门
        第四通道默认关、只有 chat_driver 门开时才经 `getattr` 探测调用（Protocol 安全
        方案，见 chat_driver 调用点注释）；`appraise_text` / `_appraise` / `generate()`
        原样不动——本方法与 `_appraise` 共享 `_parse_vad`（v/a 解析降级路径逐字同构），
        独立发起自己的一次 `chat.completions.create`（temperature=0.0），故本方法本身
        零新增调用（只在其被选中调用的那一路里，替代而非叠加原 `appraise_text` 那次调用）。

        评价对象=用户当轮输入本身（不是回复、不是历史）：`_appraise` 重锚点消除了
        prp 原 converse 尾方案的评价对象歧义——PRP/write-gate-informative/design.md
        §二·张力 2。

        诚实声明（design.md §五，PASS 前置 7，三处均须落代码注释）：
        ① 单次编码时价值判断、无回看修正机会，校准弱于延迟判断——归因锚是 Nairne 等
           （2007; 2008）未来相关性评定框架（生存加工提升编码优先级，但不提供事后
           校正），Nelson & Dunlosky (1991) 的 delayed-JOL（延迟元记忆判断优于即时
           判断）作对照参照；
        ② 判断与巩固强度的**因果耦合被拆断**——本方法只产出「判断 → 二值开关」，
           不构成「判断 → 精细加工 → 巩固强度」的完整中介链；
        ③ informative ≈ 可提取命题密度的工程代理，**不是**「重要性」构念——命名刻意
           避开 importance/salience；与写入门既有 T∧F∧A 承诺判据、identity 判据互补
           而非同构（design.md 表：颗粒度=忠实）。

        Returns:
            (valence, arousal, informative)。informative 字段缺失/非布尔 → False +
            `logger.warning` 留痕（暴露"小模型恒 False"的静默退化，供查库统计发现，
            design.md §三·前置 2）；v/a 解析降级路径与 `_appraise` 逐字同构。
        """
        appraise_sys = _APPRAISE_SYS + _APPRAISE_INFORMATIVE_ADDENDUM
        if os.getenv("ZERO_APPRAISE_CALIBRATE", "").lower() in ("1", "true", "yes"):
            appraise_sys += _APPRAISE_CALIBRATION
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": appraise_sys},
                {"role": "user", "content": text},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        result = self._parse_vad_informative(raw)
        logger.debug("appraise_informative raw=%.80s → vad+informative=%s", raw, result)
        return result

    async def converse(
        self,
        history: list[dict[str, str]],
        affect: tuple[float, float],
        retrieved: str = "",
        *,
        push: bool = False,
        relationship_hint: str = "",
    ) -> str:
        """自然多轮对话：带完整历史。`generate`(强制 VAD+双向回路) 之外的连贯不戏剧化路径。

        `push`（皮层下/不随意通路）开启时：情绪经 **affect-congruent 用词倾向**（emotion_lexicon，
        "自然流露不表演"）漏进输出，而非"演情绪的指令"——对应神经科学 push 效应（见
        notes/2026-06-25-dual-route-language-push-pull.md）；可选叠加 OpenAI `logit_bias`
        （env `ZERO_PUSH_LOGIT_BIAS=1`，需兼容 tokenizer，graceful 回退）。关闭=纯 prompt(pull)。
        retrieved: 记忆召回的背景上下文（空串时不注入，prompt 与改前逐字一致 → 零回归）。
        history 末条应为用户最新发言。
        """
        # 事实化模式每次调用活读（见 __init__ 注释：与 chat_driver 两处消费点统一生命周期）。
        factual = factual_mode_enabled()
        # 脾气段按 e* 的 valence 门控：未设 ZERO_TEMPER_VALENCE_GATE → gate=None → 无条件注入，
        # 三段拼接与拆分前逐字一致（零回归）；设了阈值（如 -0.15）则仅在心情确实为负且够强时注入，
        # 使语气强度真正由引擎的 e* 驱动，而非由一段固定 prompt 驱动。
        head = _FACTUAL_SYS_HEAD if factual else _CONVERSE_SYS_HEAD
        temper_src = _FACTUAL_TEMPER_ADDENDUM if factual else _TEMPER_ADDENDUM
        tail = _FACTUAL_SYS_TAIL if factual else _CONVERSE_SYS_TAIL
        temper = temper_src if (self.temper_gate is None or affect[0] <= self.temper_gate) else ""
        sys = (head + temper + tail).format(feeling=affect_label(*affect))
        if self.persona:
            sys = f"{self.persona}\n\n{sys}"  # L1：人设卡前置于情绪行为框架（空串时不变=零回归）
        if relationship_hint and not factual:
            # Q5-B（议会二轮·止血）：关系距离软约束，给 LLM 分寸锚（空串=零回归）。
            # ⚠ 事实化模式下**忽略**：「已经比较熟络」是对共同过去的事实断言，而其唯一依据只是
            # exposure 计数、并无真实往事支撑——与边界段第 5 条（不许断言没发生过的事）正面对撞。
            sys += (
                f"\n（你和对方目前的关系：{relationship_hint}。"
                "按这个分寸把握亲疏，别越过、别自来熟。）"
            )
        if retrieved:
            sys += (
                _FACTUAL_RECALL_LEAD.format(retrieved=retrieved)
                if factual
                else f"\n你还记得以下背景：{retrieved}"
            )
        bias_kwargs: dict[str, Any] = {}
        if push:
            # 中性死区：e* 模长趋零时方向纯属噪声，内积排序会把噪声放大成一张言之凿凿的词表
            # ——实测 e*=(-0.079,+0.037)（affect_label 判「平静」）会取出「暴怒/愤怒/恐惧」，
            # 于是同一份 prompt 一边说「心情平静」一边要求用暴怒的词，模型只能自己编个立场圆场。
            # 死区把这种轮次的 push 段整段撤掉（words 为空则本就不注入），与 NEUTRAL_RADIUS
            # 的文档语义（「避免给微弱情感强行贴词」）一致。
            # 事实化开 → 死区必开（属事实化规格）；关 → None 交给 ZERO_PUSH_NEUTRAL_DEADZONE
            # 全局默认（code-reviewer BLOCK-2/WARN-4：死区不再绑死在事实化模式上）。
            words = suggest_affect_words(
                affect[0], affect[1], k=6, neutral_deadzone=True if factual else None
            )
            if words:
                push_tpl = _FACTUAL_PUSH_ADDENDUM if factual else _PUSH_ADDENDUM
                sys += push_tpl.format(words="、".join(words))
                logit_bias = self._build_logit_bias(affect, words)
                if logit_bias:
                    bias_kwargs["logit_bias"] = logit_bias
        if factual:
            # 必须最后拼：最强近因位，且显式压过前置的人设卡与上面的召回/关系段。
            sys += _FACT_BOUNDARY_ADDENDUM
        messages: list[dict[str, str]] = [{"role": "system", "content": sys}]
        messages.extend(history)
        temperature = max(0.0, self.temperature + random.uniform(-0.1, 0.15))  # 措辞部分随机
        try:
            resp = await self.client.chat.completions.create(
                model=self.model, temperature=temperature, messages=messages, **bias_kwargs
            )
        except Exception:
            # 某些代理/模型不支持 logit_bias（或 token id 不匹配）→ 退回不带 bias；
            # push 仍经 prompt 用词倾向生效，不致命。
            logger.warning("converse 带 logit_bias 调用失败，退回无 bias 重试", exc_info=True)
            resp = await self.client.chat.completions.create(
                model=self.model, temperature=temperature, messages=messages
            )
        reply = (resp.choices[0].message.content or "").strip()
        if factual:
            # 机械执行层：剥离舞台说明后才返回（调用方随即写入历史）——历史保持干净，
            # 自我模仿雪球无从启动（见 strip_stage_directions docstring 的④臂实测记录）。
            reply = strip_stage_directions(reply)
        logger.debug(
            "converse model=%s msgs=%d push=%s reply_len=%d",
            self.model,
            len(messages),
            push,
            len(reply),
        )
        return reply

    def _build_logit_bias(self, affect: tuple[float, float], words: list[str]) -> dict[int, float]:
        """解码期 push：affect-congruent 词 → OpenAI `logit_bias`（{token_id: 偏置}）。

        env `ZERO_PUSH_LOGIT_BIAS=1` 才启用（默认关——须与模型 tokenizer 匹配，否则偏到错 token）；
        缺 tiktoken / encode 失败 → 返回空（graceful，push 退到纯 prompt 用词倾向）。
        """
        if os.getenv("ZERO_PUSH_LOGIT_BIAS", "").lower() not in ("1", "true", "yes"):
            return {}
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return {}
        out: dict[int, float] = {}
        for word, delta in affect_logit_bias(words, affect).items():
            if abs(delta) < 1e-6:
                continue
            try:
                for tid in enc.encode(word):
                    out[tid] = clamp(PUSH_LOGIT_SCALE * delta, -6.0, 6.0)
            except Exception:
                logger.debug("push logit_bias: token 编码失败 word=%s，跳过", word, exc_info=True)
                continue
        return out

    async def _compose(
        self,
        affect: tuple[float, float],
        context: str,
        retrieved: str,
        feedback: str | None,
        appraisal: str = "",
    ) -> str:
        parts = [
            f"情绪坐标: valence={affect[0]:.2f}, arousal={affect[1]:.2f}",
            f"上下文: {context or '（无）'}",
        ]
        if appraisal:
            parts.append(f"认知评价: {appraisal}（据此把握情绪的来由与分寸）")
        if self.use_lexicon:
            cues = suggest_affect_words(affect[0], affect[1], k=5)
            if cues:
                parts.append(f"可参考的贴合情绪词（不必全用）: {'、'.join(cues)}")
        if retrieved:
            parts.append(f"检索记忆: {retrieved}")
        if feedback:
            parts.append(f"上一轮偏差反馈: {feedback}（请调整措辞使情绪更贴合坐标）")
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": _COMPOSE_SYS},
                {"role": "user", "content": "\n".join(parts)},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    async def _appraise(self, text: str) -> tuple[float, float]:
        # P3：ZERO_APPRAISE_CALIBRATE 开→附分级标定锚抵消 LLM 正向偏置；默认关=旧 prompt 零回归。
        appraise_sys = _APPRAISE_SYS
        if os.getenv("ZERO_APPRAISE_CALIBRATE", "").lower() in ("1", "true", "yes"):
            appraise_sys += _APPRAISE_CALIBRATION
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": appraise_sys},
                {"role": "user", "content": text},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        result = self._parse_vad(raw)
        logger.debug("appraise raw=%.80s → vad=%s", raw, result)
        return result

    @staticmethod
    def _parse_vad(raw: str) -> tuple[float, float]:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return (0.0, 0.0)  # 无 JSON → 中性回退（保守，不崩管线）
        try:
            data = json.loads(raw[start : end + 1])
            v = clamp(float(data.get("valence", 0.0)), -1.0, 1.0)
            a = clamp(float(data.get("arousal", 0.0)), -1.0, 1.0)
        except (ValueError, TypeError, json.JSONDecodeError):
            logger.warning("VAD 解析失败，回退中性 (0,0)；raw=%.80s", raw)
            return (0.0, 0.0)
        return (v, a)

    @staticmethod
    def _parse_vad_informative(raw: str) -> tuple[float, float, bool]:
        """同 `_parse_vad`，额外解析 informative（写入门第四通道，见 appraise_text_informative）。

        v/a 直接复用 `_parse_vad`——降级路径（无 JSON/解析失败 → (0,0) + warning）逐字
        同构，不重复实现。informative 单独解析同一段 raw：字段缺失/非布尔 → False +
        `logger.warning`（暴露"小模型恒 False"的静默退化，design.md §三·前置 2）；
        该分支与 `_parse_vad` 各自独立捕获异常，二者互不掩盖对方的 warning。
        """
        v, a = OpenAILanguageModel._parse_vad(raw)
        informative = False
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
            except (ValueError, TypeError, json.JSONDecodeError):
                data = None  # _parse_vad 已就同一段 raw 记过 warning，此处不重复日志
            if data is not None:
                informative_raw = data.get("informative")
                if isinstance(informative_raw, bool):
                    informative = informative_raw
                else:
                    logger.warning("informative 字段缺失/非布尔，回退 False；raw=%.80s", raw)
        return (v, a, informative)
