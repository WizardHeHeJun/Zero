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
            # 显著度门 salience=precision×|rpe|（rpe=None→0.5 保守；McGaugh 2004 唤醒×意外度）
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

            # 富 episode：情感显著 或 承诺/日程语义通道命中 即写
            if salient or is_commitment:
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

                episode_content = (
                    f"{gist_text}"
                    f" | 情绪={label}({affect[0]:.2f},{affect[1]:.2f})"
                    f" | precision={state.affect_precision or 0.0:.2f}"
                    f" | streams={streams}"
                    f" | value={value:.3f}"
                    f"{fc_seg}"
                )
                # 无语义后端时 no-op（零回归）；只对 gist_text 嵌入（embed_text），全文仅存储/展示
                await self.memory.write_episode(
                    episode_content,
                    scope=Scope.USER,
                    key=state.user_id,
                    embed_text=gist_text,
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
