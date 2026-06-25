"""情感表达多 Agent 系统 · 本地启动 / 验证入口（官方 CLI）。

**直接 `python main.py` 即进交互对话**，验证整条系统流程（评价→引擎→两时间尺度→双路输出）。
默认零可选依赖（缺 LLM key 自动回退词典+模板）。用法：
  python main.py                       # 【默认】交互对话：情感引擎 ⊗ LLM（直接验证整条流程）
  python main.py --workspace           # 并行流 + 显著度门控全局工作空间（看哪些流点燃）
  python main.py --llm                 # 接真模型跑固定情绪场景的文本输出验证（批处理）
  python main.py --trace               # 跑核心管线打印 (v,a) 轨迹 JSON（旧默认）
  python main.py --trace --mood --language --recall   # 轨迹模式叠加门控

--chat / --llm 接真 LLM 需 `.[llm]`（已装 openai 即可）+ 在 .env 配（库代码不读 .env，本脚本读）：
  ZERO_OPENAI_API_KEY=sk-...                       # 必填
  ZERO_OPENAI_MODEL=<你的 key 可访问的模型 id>      # 必填（无代码默认，避免发到无权限模型）
  ZERO_OPENAI_BASE_URL=https://.../v1              # 可选（OpenAI/vLLM/第三方网关）
--chat 缺 key 时自动回退：词典法评价 + 模板语言，仍演示引擎演化与 mood 跨轮滞后（回应较朴素）。

更专项的入口：
  python -m scripts.demo_pipeline          # 端到端真网络化 demo（合成训练 → 注入 → 跑）
  python -m scripts.verify_graphiti_local  # Graphiti 语义召回本地验证（需 .[graphiti] + LLM key）
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random

from src.orchestration.runner import ConversationSession, dump_trajectory, run
from src.orchestration.state import Stimulus

DEFAULT_STIMULI = [
    Stimulus(name="win", goal_congruence=0.9, intensity=0.9),
    Stimulus(name="loss", goal_congruence=-0.8, intensity=0.7),
    Stimulus(name="neutral", goal_congruence=0.0, intensity=0.4),
]

# --workspace 用：对比哪些并行流点燃（突发高唤醒中性 vs 目标显著 vs 弱刺激）
WORKSPACE_STIMULI = [
    Stimulus(name="突如其来的巨响", goal_congruence=0.0, intensity=1.0),  # 生存流主导
    Stimulus(
        name="拿到梦寐以求的 offer", goal_congruence=0.9, attitude_appeal=0.7, intensity=0.8
    ),  # 评价/价值流主导
    Stimulus(name="无关紧要的小事", goal_congruence=0.05, intensity=0.2),  # 弱刺激，少数流点燃
]

# --llm 用：情绪鲜明的场景（OCC 评价维度），覆盖正/负 × 高/低强度
EMOTION_STIMULI = [
    Stimulus(name="收到心仪已久的礼物", goal_congruence=0.9, attitude_appeal=0.8, intensity=0.8),
    Stimulus(
        name="精心准备的方案被当众否决",
        goal_congruence=-0.9,
        standard_compliance=-0.6,
        intensity=0.9,
    ),
    Stimulus(name="久别重逢的老友", goal_congruence=0.6, attitude_appeal=0.7, intensity=0.5),
    Stimulus(
        name="被误解还受了委屈", goal_congruence=-0.7, standard_compliance=-0.8, intensity=0.7
    ),
]


def _load_dotenv() -> None:
    """装了 python-dotenv 且有 .env 就加载；未装静默跳过。仅脚本加载，库代码不依赖。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


async def _run_core(args: argparse.Namespace) -> None:
    """纯核心管线（默认，零可选依赖）：打印 (v,a) 轨迹 JSON。"""
    trajectory = await run(
        DEFAULT_STIMULI,
        thread_id="main",
        rng_seed=7,
        mood_enabled=bool(args.mood),
        language_enabled=bool(args.language),
        recall_enabled=bool(args.recall),
    )
    print(dump_trajectory(trajectory))


async def _run_workspace() -> None:
    """显著度门控全局工作空间：跑刺激，打印每个刺激点燃了哪些并行流 + e*。

    预期：高唤醒中性刺激（巨响）survival 流单独点燃；目标显著刺激 appraisal/value 主导；
    弱刺激只点燃最显著的一条（不空播）。纯核心管线，零可选依赖。
    """
    print("显著度门控全局工作空间｜并行流: survival(快) / appraisal(OCC) / value(RPE)\n")
    for i, stim in enumerate(WORKSPACE_STIMULI):
        traj = await run(
            [stim],
            thread_id=f"workspace-{i}",
            workspace_enabled=True,
            rng_seed=7,
        )
        s = traj[-1]
        ev = s["valence_arousal"]
        print(f"【{stim.name}】")
        print(f"  点燃的流   = {s['ignited_streams']}")
        print(
            f"  内核 e*    = ({ev[0]:+.2f}, {ev[1]:+.2f})  后验精度={s['affect_precision']:.2f}\n"
        )


class ConversationLog:
    """对话 transcript 落本地 SQLite（stdlib，无额外依赖）：逐轮存、重启重载近 N 轮 → 长期记忆。

    存储边界（与 supervisor 并行、职责不重叠）：
    - 本类管对话运行态：transcript turns 表（每轮 user/assistant 消息）+
      跨重启 attitude meta 表（持久化的慢变情绪基线）。
    - 情感事件/长期 episode/disposition 属长期情感记忆，由 SupervisorAgent 经
      MemoryClient 写入（user/session scope，见 supervisor.py）。
    - 两套存储并行运行——不在此处调用 MemoryClient，不在 supervisor 读写 transcript。
    - 本类仅在 --chat REPL 路径使用；图内 StateGraph 路径（--llm 等）不经此类、走
      SupervisorAgent→MemoryClient，两条路径互斥，attitude 不双写。
    """

    def __init__(self, path: str = "data/chat_history.sqlite3") -> None:
        import sqlite3

        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS turns "
            "(thread TEXT NOT NULL, ts TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS meta "
            "(thread TEXT PRIMARY KEY, feeling_v REAL, feeling_a REAL)"
        )
        self.conn.commit()

    def append(self, thread: str, role: str, content: str) -> None:
        from datetime import UTC, datetime

        self.conn.execute(
            "INSERT INTO turns (thread, ts, role, content) VALUES (?, ?, ?, ?)",
            (thread, datetime.now(UTC).isoformat(), role, content),
        )
        self.conn.commit()

    def recent(self, thread: str, limit: int = 20) -> list[dict[str, str]]:
        """取该 thread 最近 limit 条，按时间正序返回 [{role, content}…]（喂 LLM 对话历史）。"""
        rows = self.conn.execute(
            "SELECT role, content FROM turns WHERE thread = ? ORDER BY rowid DESC LIMIT ?",
            (thread, limit),
        ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def save_feeling(self, thread: str, feeling: tuple[float, float]) -> None:
        """落盘累积情绪（跨重启续上「情绪积累」，不每次从平静重来）。"""
        self.conn.execute(
            "INSERT INTO meta (thread, feeling_v, feeling_a) VALUES (?, ?, ?) "
            "ON CONFLICT(thread) DO UPDATE SET feeling_v=excluded.feeling_v, "
            "feeling_a=excluded.feeling_a",
            (thread, feeling[0], feeling[1]),
        )
        self.conn.commit()

    def load_feeling(self, thread: str) -> tuple[float, float]:
        """读回累积情绪；无记录则 (0, 0)（平静起步）。"""
        row = self.conn.execute(
            "SELECT feeling_v, feeling_a FROM meta WHERE thread = ?", (thread,)
        ).fetchone()
        return (float(row[0]), float(row[1])) if row else (0.0, 0.0)


async def _run_chat() -> None:
    """交互对话：情感引擎 ⊗ LLM，**两时间尺度**情绪（快变情绪 + 慢变态度），不表演不扮演。

    每句话评价成 (v,a) 喂引擎得瞬时 e*，分两路（affective chronometry + ALMA/WASABI 文献，
    见 notes/2026-06-25-emotion-decay-…）：
    - 快变 `emotion`（短时）：向 `attitude` 基线衰退恢复 + 被 e* 冲击 + 噪声——怒火飙起后几轮回落，
      情绪**不长期累积**（衰退太慢=emotional inertia 病理）。表达取它。
    - 慢变 `attitude`（对此人的长期印象）：按 e* 缓慢累积（evaluative conditioning），多轮才成形，
      是 emotion 衰退回归的基线 → 持续被骂才变冷。**只持久化 attitude**（情绪短时、重启归基线）。
    transcript + attitude 落本地 SQLite、重启续上。输入 exit/quit 退出。
    """
    _load_dotenv()
    # 本地持久：长期倾向 → sqlite（stdlib 即落盘）；运行态走 sqlite（缺 .[db] 自动回退内存）
    os.environ.setdefault("ZERO_CHECKPOINT_BACKEND", "sqlite")
    os.environ.setdefault("ZERO_MEMORY_BACKEND", "sqlite")
    for noisy in ("httpx", "openai", "src.memory.client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("src.storage").setLevel(logging.ERROR)  # 隐藏 sqlite 缺驱动的回退告警
    from src.agents.affect_math import attitude_step, clamp, emotion_decay_step
    from src.agents.emotion_lexicon import affect_label
    from src.agents.emotion_lexicon import appraise_text as lexicon_appraise
    from src.agents.language import ConversationModel

    api_key = os.getenv("ZERO_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("ZERO_OPENAI_MODEL")
    lm: ConversationModel | None = None
    if api_key and model_name:
        from src.agents.language_openai import OpenAILanguageModel

        lm = OpenAILanguageModel(model=model_name)
        mode = f"真 LLM（{model_name}）"
    else:
        mode = "无 key 回退（词典评价 + 模板）"

    thread = os.getenv("ZERO_CHAT_THREAD", "chat")  # 可切独立会话/重置（默认 chat）
    log = ConversationLog()
    history = log.recent(thread, 20)  # 重载历史 → 跨重启记忆
    attitude = log.load_feeling(thread)  # 续上对此人的长期态度（持久化的慢变量）
    emotion = attitude  # 情绪是短时的：启动即回到「对此人的态度」基线，不带旧情绪
    # mood 关：A.7 双稳会自锁；chat 侧用 emotion(快衰退)+attitude(慢累积) 两时间尺度
    # recall_enabled=True：开语义记忆召回，把 recalled_context 回灌 converse retrieved 参数
    # user_id=thread：让 disposition/episode 的 user scope 与 ConversationLog 的 thread 对齐，
    # 避免切 ZERO_CHAT_THREAD 时共享 "default-user" 记忆造成串味。
    session = ConversationSession(
        thread_id=thread,
        user_id=thread,
        mood_enabled=False,
        workspace_enabled=True,
        recall_enabled=True,
    )
    print(f"情感对话｜{mode}｜情绪快衰退·态度慢积累｜历史 {len(history) // 2} 轮｜exit/quit 退出")
    print(f"对你的态度：{affect_label(*attitude)}｜记忆落 data/chat_history.sqlite3\n")
    while True:
        try:
            user = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit", ":q"}:
            break
        # 评价桥：读这句话情绪（真 LLM 或词典回退）
        v, a = await lm.appraise_text(user) if lm is not None else lexicon_appraise(user)
        # attitude_appeal 承载对此人的长期累积态度（OCC 对象维度），用进入本轮时的 attitude[0]
        # 作为 AppraisalAgent occ_prior 的先验（0.2 权重通路）；当前句即时情感走
        # goal_congruence（不变）；recalled_disposition（TD 价值偏置）走独立加法通路，
        # 三者来源不同、作用点不同，不重复计入。
        # 快照：stim 喂入的先验值（attitude_step 在 step 之后才更新，此处为本轮进入时的态度）
        attitude_for_trace = attitude[0]
        stim = Stimulus(
            name=user[:40],
            goal_congruence=v,
            attitude_appeal=attitude[0],
            intensity=min(1.0, max(0.2, abs(a))),
        )
        step = await session.step(stim)
        e = step["valence_arousal"] or (0.0, 0.0)
        # 取语义召回上下文（recall_enabled=True 时图内 memory_recall 节点填充；无语义后端则空列表）
        recalled: list[str] = step.get("recalled_context") or []
        # 每条截断至 120 字符，防 system prompt 膨胀
        recalled_str = " | ".join(r[:120] for r in recalled) if recalled else ""
        # 慢：对此人的态度按 e* 缓慢累积（evaluative conditioning，长期印象）
        attitude = attitude_step(attitude, e)
        # 快：情绪向 attitude 基线衰退恢复 + 当前 e* 冲击 + 噪声（短时——刺激停几轮就回落）
        emotion = emotion_decay_step(emotion, attitude, e)
        emotion = (
            clamp(emotion[0] + random.gauss(0.0, 0.05), -1.0, 1.0),
            clamp(emotion[1] + random.gauss(0.0, 0.05), -1.0, 1.0),
        )
        word = affect_label(*emotion)
        history.append({"role": "user", "content": user})
        if lm is not None:
            # push 通路：情绪经用词倾向自然漏进输出（不靠"演情绪"指令）；retrieved 回灌召回背景
            reply = await lm.converse(history[-20:], emotion, recalled_str, push=True)
        else:
            reply = f"（{word}）嗯，我在听，你接着说。"
        history.append({"role": "assistant", "content": reply})
        log.append(thread, "user", user)
        log.append(thread, "assistant", reply)
        log.save_feeling(thread, attitude)  # 只持久化「态度」（情绪短时、重启归基线）
        print(f"\n{reply}")
        print(
            f"  └─ 你这句≈({v:+.2f},{a:+.2f}) | attitude_appeal先验={attitude_for_trace:+.2f}"
            f" | 情绪={word} ({emotion[0]:+.2f},{emotion[1]:+.2f})"
            f" | 对你的态度=({attitude[0]:+.2f},{attitude[1]:+.2f})\n"
        )


async def _run_llm() -> None:
    """接真 LLM 验证文本输出情绪：内核 e* → 生成语言 → 独立 VAD 反推 → 一致性。"""
    _load_dotenv()
    # 全部配置只来自 .env（本脚本 _load_dotenv 加载）——代码不写死默认，缺哪项就明确报错
    api_key = os.getenv("ZERO_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("ZERO_OPENAI_MODEL")
    if not api_key or not model_name:
        print(
            "✗ 缺少配置。请在 .env 设 ZERO_OPENAI_API_KEY 与 ZERO_OPENAI_MODEL"
            "（模型 id 取你的 key 实际可访问的那个；可选 ZERO_OPENAI_BASE_URL）。"
        )
        return
    # 延迟 import：仅 --llm 路径需要 openai，默认路径零可选依赖
    from src.agents.emotion_lexicon import affect_descriptor
    from src.agents.language_openai import OpenAILanguageModel

    lm = OpenAILanguageModel(model=model_name, use_lexicon=True)
    print(f"模型={model_name}｜use_lexicon｜appraisal_conditioning｜双向回路\n")
    for i, stim in enumerate(EMOTION_STIMULI):
        traj = await run(
            [stim],
            thread_id=f"llm-emo-{i}",
            language_enabled=True,
            appraisal_conditioning_enabled=True,
            language_model=lm,
            rng_seed=7,
        )
        s = traj[-1]
        ev = s["valence_arousal"]
        la = s["language_affect"]
        cons = s["language_consistency"]
        ev_d = affect_descriptor(ev[0], ev[1])
        print(f"【{stim.name}】")
        print(f"  内核 e*      = ({ev[0]:+.2f}, {ev[1]:+.2f}) → {ev_d}")
        print(f"  生成语言     = {s['language_text']}")
        if la is not None:
            la_d = affect_descriptor(la[0], la[1])
            print(f"  语言反推 VAD = ({la[0]:+.2f}, {la[1]:+.2f}) → {la_d}")
        print(f"  一致性距离   = {cons:.3f}（越小越贴合）  回路迭代={s['language_iter']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="情感表达多 Agent 系统 — 直接启动即进对话，验证整条系统流程"
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="跑核心管线打印 (v,a) 轨迹 JSON（旧默认；可配 --mood/--language/--recall）",
    )
    parser.add_argument("--mood", action="store_true", help="[--trace] 慢变心境（A.7 滞后）")
    parser.add_argument("--language", action="store_true", help="[--trace] 语言层双向回路")
    parser.add_argument("--recall", action="store_true", help="[--trace] 长期倾向记忆回灌")
    parser.add_argument(
        "--workspace",
        action="store_true",
        help="并行流 + 显著度门控全局工作空间（打印每刺激点燃的流）",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="交互对话（**默认**，可省略）：情感引擎 ⊗ LLM，缺 key 回退词典+模板",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="接 OpenAI 兼容真模型做文本输出情绪验证（需 .[llm] + key）",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # 默认（无 flag 或 --chat）：直接进对话，验证整条系统流程；其它模式用显式 flag。
    if args.trace:
        asyncio.run(_run_core(args))
    elif args.llm:
        asyncio.run(_run_llm())
    elif args.workspace:
        asyncio.run(_run_workspace())
    else:
        asyncio.run(_run_chat())


if __name__ == "__main__":
    main()
