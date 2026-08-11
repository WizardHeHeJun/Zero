"""SupervisorAgent：协调与任务完成节点。

只做协调 + 任务完成判定，**不含业务逻辑**（业务在各 Worker）。记忆写入
**只在此处**发生（节流，见 memory-rules.md #1）：当前情绪事件写 session 作用域，
长期情绪倾向写 user 作用域，均显式 scope。注入 MemoryClient，不直连图谱。

存储边界（与 src/storage 的 ConversationLog 并行、职责不重叠）：
- SupervisorAgent 只写情感事件（write，session scope）、长期 episode（write_episode，
  user scope）、disposition（write，user scope）至 MemoryClient。
- 对话 transcript 与 attitude 短期态属运行态，由 src/storage/conversation_log.py 的
  ConversationLog 管理（turns 表 + meta 表，SQLite，无需 MemoryClient）。
- 两套存储并行运行——不在此处读写对话历史，不在 ConversationLog 写情感记忆。

层归属修正（B 类·2026-07-22）：
- ACT-R access_count 更新经 MemoryClient.batch_update_access_count，不直接访问
  self.memory.semantic（守三层单向：编排层→记忆层 API，不跨到存储层）。
"""

from __future__ import annotations

import os
import re

from src.agents.affect_math import text_label
from src.memory.client import MemoryClient
from src.memory.types import Scope
from src.orchestration.state import AffectState

# 议会 A 语义写入通道：承诺/日程类标记（时间点/钟点/约定/星期日期）。命中即使情感低唤醒、低 salience
# 也强制写 episode——对应 PFC 对「语义重要但情绪平淡」内容的独立巩固调制（生物席「下午两点」案例），
# 不依赖 McGaugh 的唤醒×意外度门控。纯确定性正则，不进 LLM（守 affect 热路径红线）。
_COMMITMENT_RE = re.compile(
    r"几点|[一二三四五六七八九十两零\d]\s*点|\d\s*[:：]\s*\d|时间|约(好|定|会)|说好|答应|"
    r"明天|后天|今晚|上午|中午|下午|晚上|星期[一二三四五六日天]|周[一二三四五六日天]|\d+\s*号"
)


def _is_commitment(text: str) -> bool:
    """文本是否含承诺/日程类语义标记（时间/约定/日期）。纯正则、无 LLM、可单测。"""
    return bool(text) and _COMMITMENT_RE.search(text) is not None


# 身份自陈写入通道（议会 2026-07-30 D1·BLOCK）：与上面承诺/日程旁路**并列**的第三条正交通道。
# 依据：McGaugh 2004 是**调制假说**（调制已发生编码的巩固强度），非存在门——反证见
# Cahill et al. 1995 (Nature 377:295-296)：双侧杏仁核损毁者对故事中性段落记忆与常人无异。
# 而主门 salience=precision×|rpe| 是**乘性 AND 门**，|rpe|=0 是测度为正的整个超平面，
# 使身份自陈必然丢失（100 轮实测：姓名/职业 salience=0.000 被丢，语义库含「林川」0 条）。
# 姓名/职业属「个人语义」（Renoult et al. 2012 DOI:10.1016/j.tics.2012.09.003），
# 其巩固靠自我关联精细加工而非情绪唤醒峰值。纯确定性正则、不进 LLM（守 CS 席红线）。
#
# 判据分四段，缺一不可（宁漏勿误：任一段不确定即放行给主门，回到现状而非制造错误记忆）。
# ⚠ 段 3 **不得**改成「长度 + 字符类」——实测「做后端开发的」与「做什么工作的」在长度与
# 字符类上完全同构，那种写法判别力恒为零（首版 PRP 正是栽在这里）。

# 段 1+2：自指主语 + 身份谓词。不锚 ^（容「对了我叫…」），宽松引入的假阳由段 4 兜。
# 「我」与谓词之间只允许**闭集副词**——用通配会把「我朋友是医生」「我妈是退休教师」收进来。
_SELF_ADVERB = r"(?:平时|现在|目前|一直|最近|其实|本来)?"
_IDENTITY_NAME_RE = re.compile(
    rf"我{_SELF_ADVERB}(?:叫|名字是|的名字是)\s*([^\s，,。.！!？?、；;]{{2,4}})"
)
# 段 3：具名宾语走**闭集**。职业名词表（可单测、可扩，不依赖长度/字符类）。
# ⚠ **只收明确的职业名词**，刻意排除可作动词/泛指的词（测试/开发/设计/培训/咨询/数据/安全/
#   销售/编辑/摄影/翻译）——否则「我在测试你」「我在开发一个新功能」这类**活动陈述**
#   会被判成身份自陈（code-reviewer 实测 13/13 假阳）。代价是漏掉「我是做测试的」，
#   合 D1-Q4 宁漏勿误：漏判退回现状，误判则制造错误记忆。
_OCCUPATIONS: frozenset[str] = frozenset(
    {
        "后端",
        "前端",
        "运维",
        "算法",
        "产品",
        "程序员",
        "工程师",
        "架构师",
        "老师",
        "教师",
        "医生",
        "护士",
        "律师",
        "会计",
        "记者",
        "客服",
        "司机",
        "厨师",
        "警察",
        "公务员",
        "经理",
        "助理",
        "秘书",
        "作家",
        "画家",
    }
)

# 职业词直接编进正则的**交替组**（长词优先），让引擎自己回溯找到落在闭集里的切分。
# ⚠ 不要写成「先用 [一-龥]{2,4} 捕获再查表」——那样「我是做后端开发的」会捕到「做后端开」，
#   一次匹配失败就返回、不回溯，快验实测该写法只命中 7/12 正样本。
_OCCUPATION_ALT = "|".join(sorted(_OCCUPATIONS, key=len, reverse=True))
_IDENTITY_JOB_RE = re.compile(
    rf"我{_SELF_ADVERB}(?:是|在|做)[^，,。.！!？?]{{0,10}}?({_OCCUPATION_ALT})"
)

# 段 3b：宾语排除。三组，缺一不可——「叫」在中文里作使役动词极常见
# （「我叫外卖了」「我叫救护车了」「我叫他来」），只挡代词是远远不够的。
# 停用动词/虚词：出现在宾语**首位**即整句放行（「我叫你别管」「我是说…」）。
_OBJECT_STOPWORDS: tuple[str, ...] = (
    "你",
    "他",
    "她",
    "它",
    "我",
    "说",
    "想",
    "让",
    "叫",
    "怕",
    "觉得",
    "以为",
    "打算",
    "准备",
    "真的",
    "有点",
    "特别",
    "这",
    "那",
    "个",
    "了",
    "过",
)

# 姓名宾语不得以助词/趋向动词收尾——「我叫外卖了」「我叫救护车了」的宾语正是这一形态。
_NAME_TRAILING: tuple[str, ...] = ("了", "来", "去", "上", "下", "过", "吧", "呢", "啊", "的")

# 姓名宾语不得以这些常见非人名名词起头（使役对象而非自我介绍）。
_NON_NAME_NOUNS: tuple[str, ...] = (
    "外卖",
    "快递",
    "救护",
    "出租",
    "滴滴",
    "代驾",
    "大家",
    "咱们",
    "他们",
    "师傅",
    "阿姨",
    "护工",
)

# 职业词后若紧跟宾语标记，说明是「在做某事」而非「我的职业是」——整句放行。
_OBJECT_MARKER_AFTER_JOB: tuple[str, ...] = (
    "你",
    "我",
    "他",
    "她",
    "它",
    "一个",
    "一下",
    "一份",
    "一件",
    "这",
    "那",
    "新",
    "几",
)

# 段 4：疑问排除。挡第一人称疑问（「我是老师吗」「我叫什么名字」）——
# 这类句子段 1-3 会放行（谓词与宾语都合法），只有本段能拦。
# ⚠ 注意区分：100 轮里那条假阳「我是做什么工作的」**不是**靠本段拦住的，
#   真正拦它的是段 3 的职业闭集（"工作"不在表内）；本段对它只是冗余的第二道防线。
#   该结论由 test_identity_fact_bypass 的实现级变异测试实测得出（假设错过一次，红了才发现）。
_QUESTION_RE = re.compile(r"什么|哪|谁|几[个天点位]|多少|吗|呢|[?？]")


def _is_identity_disclosure(text: str) -> tuple[str, str] | None:
    """文本是否为用户对自己的身份自陈；是则返回 (属性类型, 抽出的实体)，否则 None。

    纯正则、无 LLM、可单测。属性类型取值 `name` / `occupation`（居住地本期不收：
    100 轮语料零居住地自陈，而「我在」是误报重灾区）。
    """
    if not text:
        return None
    # 段 4 先行：疑问句一律放行给主门（含「我叫什么名字」「我是做什么工作的」）。
    if _QUESTION_RE.search(text):
        return None
    name_match = _IDENTITY_NAME_RE.search(text)
    if name_match:
        entity = name_match.group(1)
        # 三道排除，任一命中即放行给主门（宁漏勿误）：
        #   ① 首位是代词/虚词（「我叫你别管」）
        #   ② 以助词/趋向动词收尾（「我叫外卖了」「我叫救护车了」）
        #   ③ 是常见使役对象或含职业名词（「我叫外卖师傅上来」「我叫产品经理来开会」）
        if (
            not entity.startswith(_OBJECT_STOPWORDS)
            and not entity.endswith(_NAME_TRAILING)
            and not entity.startswith(_NON_NAME_NOUNS)
            and not any(occ in entity for occ in _OCCUPATIONS)
        ):
            return ("name", entity)
    job_match = _IDENTITY_JOB_RE.search(text)
    if job_match:
        # 捕获组就是职业表的交替分支，命中即已落在闭集内，无需再查表。
        # 但要再看紧跟其后的内容：跟着宾语标记说明是「在做某事」不是「我的职业是」。
        if not text[job_match.end() :].startswith(_OBJECT_MARKER_AFTER_JOB):
            return ("occupation", job_match.group(1))
    return None


# 身份 episode 的写入显著度下限（PRP 架构决策 F）。
# 实测：身份轮 affect_precision=8.56 → Hill 归一 8.56/(8.56+30)=0.222，低于召回注入门
# ZERO_RECALL_INJECT_MIN 默认 0.5 ⇒ 该 episode 永远进不了 LLM 的注意力预算，
# 「写进去了却召不回」。取 40.0 使归一后 0.571 > 0.5，与既有先例
# chat_driver.SEED_MEMORY_PRECISION 同值同源（那条注释亦写明「取高于召回注入门」）。
# ⚠ 这是「人为抬高情感精度以换召回优先级」，属科学决策边界；沿用同源先例而非新发明，
#   若议会要独立裁定该做法的适用范围，改这一个常量即可。
# ⚠ 该常量「一钱多用」（议会 2026-07-31 WARN-2 → D2/D4，已披露非隐藏）：
#   `precision=` 是**共享字段**，被三处消费，而本处只论证了第一处——
#     ① 召回注入门（chat_driver 的 inject_min 判定）：只需落在门限哪一侧，覆写安全；
#     ② 召回排序（memory_recall._rank_episodes 的 importance 维）：数值直接进线性加权和，
#        实测本覆写带来 **+0.1153 固定加成**（占三维总权重 11.5%，等价于 sim 高出 0.34）；
#     ③ 遗忘调制（consolidation.EbbinghausDecay 的 a_eff = a×salience^κ）：数值直接进幂函数，
#        实测 a_eff **×1.604**，且是振幅项、不随 Δt 衰减。
#   ②③ 两处方向与设计意图一致（身份/种子事实本就该更易召回、更慢遗忘），
#   但它们是经由一个语义为「后验方差倒数」的量实现的 —— 是效果导向的**代理**，非机制忠实
#   （数学席判 Stevens 1946 尺度类型误用：只对阈值判定安全的覆写扩散进了要求基数尺度的公式）。
#   正解是独立的 importance 信号（比照本仓 `first_contact=True` 标记 + 独立系数的先例），
#   已 DEFERRED 至独立 PRP（议会 D5）；⚠ 若届时实现，**只能走 content 内文本 tag，
#   不得给 episodes 表加 DB 列**（议会 D1·BLOCK：GraphitiGraphStore 无结构化列钩子，
#   加列会造成两后端能力分叉）。
#   改动本常量会同时改变上述三处行为，回归锁见 tests/test_precision_shared_field_coupling.py。
IDENTITY_MEMORY_PRECISION = 40.0


def _env_flag_directionless(name: str, default: bool) -> bool:
    """方向无关地解析布尔 env：真值集→True / 假值集→False / **其余回落 default**。

    ⚠ 不能用「只判真值集」的写法——那把「未识别」当「假」，对**默认 True** 的旗标
    失败方向是反的（空串/带空格的 "true " 会静默关掉本应默认开的开关）。
    本仓 `mcp_server._env_flag` 是同一模式，但那是依赖编排层的适配层，
    此处不 import 它（编排层不得反向依赖），复刻实现。
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


class SupervisorAgent:
    """任务完成节点：标记完成并节流 flush 记忆。

    salience 相关旋钮在构造期一次从 env 读取，存 self.*，不在每次 __call__ 热路径重读。
    """

    def __init__(self, memory: MemoryClient) -> None:
        self.memory = memory
        # D5：已写过 episode 的 key，判 first_contact（primacy）。
        # TODO(multi-worker): 进程内 set，单进程/单 worker 可接受；
        #   多 worker / 重启下同一 user key 可被多标（first_contact 会重复打）。
        #   当前单进程 --chat 场景多标一次可接受（first_contact ×1.2 对同 key 第二个
        #   episode 少量误伤），仅多 worker 部署前须下沉持久层（如查语义库该 user 是否
        #   已有 episode），不能依赖进程内 set。
        self.seen_episode_keys: set[str] = set()
        # salience 旋钮（构造期一次解析，默认值与旧 getenv 逐字一致）
        # D6：ZERO_EPISODE_SALIENCE_AFFECTIVE_ADD 开启时让 value 量级补偿 salience 门（默认 0=关）
        self.salience_affective_add = os.getenv(
            "ZERO_EPISODE_SALIENCE_AFFECTIVE_ADD", "0"
        ).lower() not in ("0", "", "false")
        # salience 写 episode 最低门（ZERO_EPISODE_SALIENCE_MIN 默认 0.15）
        self.ep_threshold = float(os.getenv("ZERO_EPISODE_SALIENCE_MIN", "0.15"))
        # 身份自陈旁路（议会 D1）：**默认开**——本项修的是已判定为失真的行为，
        # 默认关等于让失真继续存在。保留显式关闭作逃生阀与 A/B 对照臂。
        self.identity_bypass_enabled = _env_flag_directionless("ZERO_IDENTITY_FACT_BYPASS", True)
        # 身份事实去重：(user_id, 属性类型, 实体) 三元组。
        # ⚠ **不按属性类型一次性去重**——那与 memory-rules.md #4「新事实使旧事实失效」冲突：
        #   用户改口/换工作（「我上个月换工作了，现在做算法」）会被静默吞掉，比没记住更糟。
        #   同类型不同实体 = 新事实照写，时序失效交给记忆层。
        # 带 user_id 是对齐 seen_episode_keys 先例，防未来 SupervisorAgent 池化后跨用户串味（#2）。
        # 进程内 set：重启后丢失、同一事实会再写一次（余弦去重不保证拦，见 PRP 架构决策 C.3）。
        self.seen_identity_facts: set[tuple[str, str, str]] = set()

    def _is_first_contact(self, key: str) -> bool:
        """首次为该 key 写 episode 时返回 True（并登记），之后恒 False。

        进程内 set 轻量记录、无额外 IO。**边界**：重启 / 多实例（并发或重开 session 各持独立
        SupervisorAgent）下同 key 可被多标——可接受（多标一次优于漏标，议会 D5 悬而未决 #4）。
        对应系列位置效应的首因端（D5）：「第一次见面说的话」获额外检索权重。写 episode 前调一次。
        """
        if key in self.seen_episode_keys:
            return False
        self.seen_episode_keys.add(key)
        return True

    async def __call__(self, state: AffectState) -> dict:
        affect = state.affect_sample
        if affect is not None:
            stim_name = state.stimulus.name if state.stimulus is not None else "unknown"
            # 当前情绪事件：session 作用域（session 内自然有界，非长期图谱 flood，故每轮写）
            await self.memory.write(
                f"event={stim_name} affect=({affect[0]:.2f},{affect[1]:.2f})",
                scope=Scope.SESSION,
                key=state.session_id,
            )
            value = state.value_estimate if state.value_estimate is not None else 0.0
            # 显著度门 salience=precision×|rpe|（rpe=None→0.5 保守）。两个因子分属不同调制系统，
            # 不共用一条引文：precision 项对应杏仁核-NE 唤醒调制（McGaugh 2004，
            # DOI:10.1146/annurev.neuro.27.070203.144157）；|rpe| 项对应多巴胺价值预测误差
            # （Bromberg-Martin et al. 2010，DOI:10.1016/j.neuron.2010.11.022）。
            # ⚠ 已知失真（2026-07-30 议会 4/5 席，不粉饰）：McGaugh 是**调制假说**——描述唤醒对
            # 已发生编码的巩固强度做连续增益，从不主张无该信号即不编码（反证 Cahill et al. 1995,
            # Nature 377:295-296：杏仁核损毁者对中性段落记忆正常）。而此处把它用作**二元不可逆
            # 写入门**，且 salience 是乘性 AND 门——|rpe|=0 是测度为正的整个超平面，再高的 precision
            # 也被归零，身份自陈类中性事实（姓名/职业/计划）因此必然丢失。同一 salience 在
            # memory.consolidation.EbbinghausDecay 里作连续调制（a_eff=a×salience**κ）才是忠实用法。
            # 修复方向见议会 D1：仿下方 _is_commitment 先例增设确定性身份自陈旁路（须走完整 PRP）。
            salience = (state.affect_precision or 0.0) * (
                abs(state.rpe) if state.rpe is not None else 0.5
            )
            # D6：低唤醒高语义内容 precision 低、rpe≈0 易漏写。可调低阈值或开
            # ZERO_EPISODE_SALIENCE_AFFECTIVE_ADD 让 value 量级补偿门控（默认 0=关，零回归）。
            if self.salience_affective_add:
                salience += 0.3 * abs(value)
            ep_threshold = self.ep_threshold
            user_text = (state.stimulus.text or "") if state.stimulus is not None else ""
            # 议会 A 语义写入通道：承诺/日程内容即便低 salience 也写（独立于 McGaugh 情绪门）
            is_commitment = _is_commitment(user_text)
            salient = salience >= ep_threshold
            # 议会 D1 身份自陈通道：独立于 salience/rpe 的第三条判据（与上两条并列，非替代）。
            # 去重按 (user_id, 类型, 实体) 三元组——同类型不同实体视为新事实照写（守 #4 时序失效）。
            identity_fact = (
                _is_identity_disclosure(user_text) if self.identity_bypass_enabled else None
            )
            identity_hit = (
                identity_fact is not None
                and (state.user_id, identity_fact[0], identity_fact[1])
                not in self.seen_identity_facts
            )

            # 长期情绪倾向：user 作用域。确定性后端 add_fact 按 (scope,key) 时序失效
            # （memory-rules#4），活跃 disposition 恒为最新一条、不胀活跃集；且写在 supervisor
            # 任务完成节点（合 memory-rules#1）——此处「任务」粒度 = 单轮对话（图末端节点），
            # 与「长任务里每步写」的 memory-rules#1 禁止情形不同（后者才是高频刷图谱的坑）。
            # chat 长对话的增长在 episode 侧，由 salience 门 + dedup + 容量上限收口，
            # 不在此误伤 disposition→召回闭环。
            await self.memory.write(
                f"disposition stimulus={stim_name} value={value:.3f}",
                scope=Scope.USER,
                key=state.user_id,
            )

            # 富 episode：情感显著 或 承诺/日程 或 身份自陈 命中即写。
            # ⚠ 三者 **OR 进同一条件**、全轮仍只有一次 write_episode 调用——不新增独立写入分支
            # （否则会重复消费 _is_first_contact 的首因名额，且节流断言失去意义）。
            if salient or is_commitment or identity_hit:
                # B-1 gist：用户原话（stimulus.text）优先，退化到 name[:40]
                gist = f"你说：{user_text[:200]}" if user_text else f"话题：{stim_name[:40]}"
                # B-2 language 段：language_text 非空才拼，否则省略（language_enabled=False 干净）
                lang_seg = ""
                if state.language_text:
                    lang_seg = f" / 我说：{(state.language_text or '')[:200]}"
                # 议会 A：gist_text = 喂向量的检索语义（你说/我说），与下面带元数据的存储全文分离，
                # 元数据数字不再稀释 embedding（encoding specificity）。
                gist_text = f"{gist}{lang_seg}"

                label = text_label(affect[0], affect[1])
                streams = state.ignited_streams or []
                # D5 首因：该 user 首条 episode 打 first_contact 标签（召回重排 ×1.2）
                fc_seg = " | first_contact=True" if self._is_first_contact(state.user_id) else ""
                # 语义重要性 tag（**只写不消费**·当前无任何下游读取，故行为零回归）。
                # 为何现在就写：is_commitment / identity_fact 已在写入门算出但算完即弃，而这两个
                # 判据**只能在写入时算**（依赖 user_text），事后无法回填——不趁现在落进 content，
                # 将来独立 importance 信号落地时，此前所有历史 episode 都将无 tag 可用。
                # 为何走 content 文本而非 DB 列：议会 D1·BLOCK——GraphitiGraphStore 无结构化列
                # 钩子，加列会造成两后端能力分叉。
                # 消费方（未来）注意：这些 tag 位于尾部元数据段，与用户原话同处一个字符串，
                # 解析须取**最后一个**匹配（同 parse_importance 的 WARN-1 防注入口径）。
                commit_seg = " | commitment=True" if is_commitment else ""
                identity_seg = (
                    f" | identity={identity_fact[0]}" if identity_fact is not None else ""
                )

                # 架构决策 F：身份 episode 的 precision 取下限，否则被召回注入门拦住
                # （8.56 → Hill 归一 0.222 < inject_min 0.5，写进去也召不回）。
                ep_precision = state.affect_precision or 0.0
                if identity_fact is not None:
                    ep_precision = max(ep_precision, IDENTITY_MEMORY_PRECISION)
                episode_content = (
                    f"{gist_text}"
                    f" | 情绪={label}({affect[0]:.2f},{affect[1]:.2f})"
                    f" | precision={ep_precision:.2f}"
                    f" | streams={streams}"
                    f" | value={value:.3f}"
                    f"{fc_seg}{commit_seg}{identity_seg}"
                )
                # 无语义后端时 no-op（零回归）；只对 gist_text 嵌入（embed_text），全文仅存储/展示
                await self.memory.write_episode(
                    episode_content,
                    scope=Scope.USER,
                    key=state.user_id,
                    embed_text=gist_text,
                )
                # 登记时机：**无论本轮写入由哪条判据触发**，只要身份自陈命中且确实写了就登记。
                # 若只在 identity_hit 触发时登记，高唤醒身份自陈经 salience 主门写入后不打标，
                # 之后平静复述会经旁路二次写入。
                if identity_fact is not None:
                    self.seen_identity_facts.add(
                        (state.user_id, identity_fact[0], identity_fact[1])
                    )

        # B 类·ACT-R 节流更新：任务完成节点读 recalled_episode_ids，
        # 经 MemoryClient.batch_update_access_count 更新 access_count（不直连 semantic）。
        # 层归属修正：Supervisor→MemoryClient→semantic，守三层单向。
        # 不在召回节点更新（CS BLOCK：避免污染当轮排序自一致性）。
        episode_ids = state.recalled_episode_ids
        if episode_ids and self.memory is not None:
            await self.memory.batch_update_access_count(list(episode_ids))

        entry = {"node": "supervisor", "task_complete": True}
        return {"task_complete": True, "trace": [entry]}
