"""情感引擎 ⊗ LLM 的交互对话驱动器（IO 无关的对话核心逻辑）。

原本整段写死在**临时入口** `main.py` 的 `_run_chat` 里；抽出到编排层，使「删除 main.py
做主入口迁移」后，对话驱动（两时间尺度情绪 + 评价桥 + 召回回灌 + 落盘）不随入口消亡。

职责划分（关键）：
- **本模块只管「一轮对话怎么算」**：评价→引擎 step→两时间尺度情绪更新→生成回复→落盘，
  以及启动初始化（env 默认 / lm·log·session 构造 / 历史·态度载入）。**不读 input、不 print**。
- **IO（读用户输入 / 打印 / REPL 循环）由入口提供**——新入口换一套 IO 即可复用本驱动器。

两时间尺度（affective chronometry，ALMA/WASABI 文献）：
- 快变 `emotion`：向 `attitude` 基线衰退恢复 + 被 e* 冲击 + 噪声（怒火几轮回落，不长期累积）。
- 慢变 `attitude`：按 e* 缓慢累积（evaluative conditioning），多轮成形，是 emotion 衰退的基线。
  **只持久化 attitude**（情绪短时、重启归基线）。
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass

from src.agents.affect_math import attitude_step, clamp, emotion_decay_step
from src.agents.emotion_lexicon import affect_label
from src.agents.emotion_lexicon import appraise_text as lexicon_appraise
from src.agents.language import ConversationModel
from src.memory.types import Fact
from src.orchestration.memory_recall import parse_importance
from src.orchestration.runner import ConversationSession
from src.orchestration.state import Stimulus
from src.storage.conversation_log import ConversationLog

logger = logging.getLogger(__name__)


@dataclass
class ChatTurn:
    """一轮对话的结果 + 供入口展示的 trace 量（纯数据，无 IO）。"""

    reply: str
    appraised: tuple[float, float]  # 这句话评出的 (v, a)
    attitude_prior: float  # 进入本轮时的 attitude[0]（喂 occ_prior 的先验快照）
    emotion: tuple[float, float]  # 本轮快变情绪（表达取它）
    emotion_label: str
    attitude: tuple[float, float]  # 本轮后对此人的慢变态度


def _u_shape_history(history: list[dict[str, str]], k: int, n: int) -> list[dict[str, str]]:
    """U 形工作记忆窗（D2）：保留首 k 条（首因 primacy）+ 尾 (n-k) 条（近因 recency）。

    人类系列位置效应是 U 形——首尾皆强（Murdock 1962）；单纯尾窗 history[-n:] 漏掉首因
    （「第一次见面说的话」）。k=0 或 k>=n 或 len<=n 时安全退化为 history[-n:]（逐字节等价
    原行为，K=0 即零回归开关）。纯切片、无副作用、可独立单测。
    """
    if k <= 0 or k >= n or len(history) <= n:
        return history[-n:]
    return history[:k] + history[-(n - k) :]


def _inject_recalled_as_system(
    window: list[dict[str, str]],
    recalled_facts: list[Fact],
    inject_min: float,
) -> list[dict[str, str]]:
    """把高 importance 召回 episode 以 system 条目插入 window 头部，进 LLM 注意力预算竞争（D1）。

    现状召回仅作旁路字符串回灌、不与近期 history 在 attention 预算里竞争；本函数让
    importance(=写入 precision) ≥ inject_min 的 episode 作为 {"role":"system",...} 与 history
    同窗竞争（对应皮层 reinstatement）。recalled_facts 已按三维 score 降序（_rank_episodes）；
    空列表原样返回 window（BLOCK-2 fallback，零回归）。纯数据装配、无 LLM（守 BLOCK-1）。
    """
    if not recalled_facts:
        return window
    system_entries: list[dict[str, str]] = [
        {"role": "system", "content": f"（记忆片段）{f.content}"}
        for f in recalled_facts
        if parse_importance(f.content) >= inject_min
    ]
    return system_entries + window


class ChatDriver:
    """对话驱动：持有跨轮状态（history / emotion / attitude），每调一次 `step` 推进一轮。

    通过依赖注入（lm / log / session）构造，便于测试与换入口；用 `build_chat_driver()`
    工厂从 env 装配生产实例。
    """

    def __init__(
        self,
        *,
        thread: str,
        lm: ConversationModel | None,
        log: ConversationLog,
        session: ConversationSession,
        history: list[dict[str, str]],
        attitude: tuple[float, float],
        mode: str,
    ) -> None:
        self.thread = thread
        self.lm = lm
        self.log = log
        self.session = session
        self.history = history
        self.attitude = attitude
        self.emotion = attitude  # 情绪短时：启动即回到「对此人的态度」基线，不带旧情绪
        self.mode = mode

    async def step(self, user_text: str) -> ChatTurn:
        """推进一轮：评价→引擎→两时间尺度情绪→生成回复→落盘，返回本轮结果。"""
        # 评价桥：读这句话情绪（真 LLM 或词典回退）
        if self.lm is not None:
            v, a = await self.lm.appraise_text(user_text)
        else:
            v, a = lexicon_appraise(user_text)
        # 快照：进入本轮时的态度（attitude_step 在 step 之后才更新）
        attitude_prior = self.attitude[0]
        stim = Stimulus(
            name=user_text[:40],
            text=user_text,
            goal_congruence=v,
            attitude_appeal=self.attitude[0],
            intensity=min(1.0, max(0.2, abs(a))),
        )
        step_out = await self.session.step(stim)
        e = step_out["valence_arousal"] or (0.0, 0.0)
        # 语义召回上下文（recall_enabled 时图内 memory_recall 节点填充；无语义后端则空）
        recalled: list[str] = step_out.get("recalled_context") or []
        recalled_str = " | ".join(r[:120] for r in recalled) if recalled else ""
        # D1：召回的原始 Fact（已三维重排），高 importance 者升入 history 注意力预算竞争
        recalled_facts: list[Fact] = step_out.get("recalled_facts") or []
        # 慢：态度按 e* 缓慢累积（长期印象）
        self.attitude = attitude_step(self.attitude, e)
        # 快：情绪向 attitude 基线衰退恢复 + 当前 e* 冲击 + 噪声（短时）
        self.emotion = emotion_decay_step(self.emotion, self.attitude, e)
        self.emotion = (
            clamp(self.emotion[0] + random.gauss(0.0, 0.05), -1.0, 1.0),
            clamp(self.emotion[1] + random.gauss(0.0, 0.05), -1.0, 1.0),
        )
        word = affect_label(*self.emotion)
        self.history.append({"role": "user", "content": user_text})
        if self.lm is not None:
            # D2 U 形窗（首因+近因）→ D1 注入高 importance 召回为 system 条目，进同一注意力预算竞争
            primacy_k = int(os.getenv("ZERO_HISTORY_PRIMACY_K", "3"))
            window_n = int(os.getenv("ZERO_HISTORY_WINDOW", "20"))
            inject_min = float(os.getenv("ZERO_RECALL_INJECT_MIN", "0.5"))
            window = _u_shape_history(self.history, primacy_k, window_n)
            window = _inject_recalled_as_system(window, recalled_facts, inject_min)
            # push 通路：情绪经用词倾向自然漏进输出；retrieved 回灌召回背景（与 system 条目并存）
            reply = await self.lm.converse(window, self.emotion, recalled_str, push=True)
        else:
            reply = f"（{word}）嗯，我在听，你接着说。"
        self.history.append({"role": "assistant", "content": reply})
        self.log.append(self.thread, "user", user_text)
        self.log.append(self.thread, "assistant", reply)
        self.log.save_feeling(self.thread, self.attitude)  # 只持久化「态度」
        logger.debug(
            "chat step thread=%s appraise=%s e*=%s emotion=%s attitude=%s",
            self.thread,
            (v, a),
            e,
            self.emotion,
            self.attitude,
        )
        return ChatTurn(
            reply=reply,
            appraised=(v, a),
            attitude_prior=attitude_prior,
            emotion=self.emotion,
            emotion_label=word,
            attitude=self.attitude,
        )


def build_chat_driver(thread: str | None = None) -> ChatDriver:
    """从环境装配生产用对话驱动器：读 env 构造 lm/log/session + 载入历史/态度。

    缺 LLM key → lm=None（词典评价 + 模板回退，仍跑两时间尺度情绪）。**纯装配、无全局副作用**：
    后端选择（ZERO_*_BACKEND 默认）、噪声日志压制等「入口/部署策略」由调用方（入口）在此
    之前设好；本工厂只读已就绪的 env，不写 os.environ、不改全局 logger 级别。
    """
    resolved_thread = thread or os.getenv("ZERO_CHAT_THREAD") or "chat"
    api_key = os.getenv("ZERO_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("ZERO_OPENAI_MODEL")
    lm: ConversationModel | None = None
    mode = "无 key 回退（词典评价 + 模板）"
    if api_key and model_name:
        from src.agents.language_openai import OpenAILanguageModel  # 延迟：仅有 key 才需 openai

        lm = OpenAILanguageModel(model=model_name)
        mode = f"真 LLM（{model_name}）"

    log = ConversationLog()
    history = log.recent(resolved_thread, 20)  # 重载历史 → 跨重启记忆
    attitude = log.load_feeling(resolved_thread)  # 续上对此人的长期态度（持久化的慢变量）
    # user_id=thread：让 disposition/episode 的 user scope 与 ConversationLog 的 thread 对齐，
    # 避免切 ZERO_CHAT_THREAD 时共享 "default-user" 记忆造成串味。
    session = ConversationSession(
        thread_id=resolved_thread,
        user_id=resolved_thread,
        mood_enabled=False,  # A.7 双稳会自锁；chat 用 emotion(快衰退)+attitude(慢累积)
        workspace_enabled=True,
        recall_enabled=True,  # 开语义召回，把 recalled_context 回灌 converse
    )
    return ChatDriver(
        thread=resolved_thread,
        lm=lm,
        log=log,
        session=session,
        history=history,
        attitude=attitude,
        mode=mode,
    )
