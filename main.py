"""情感表达多 Agent 系统 · 本地启动 / 验证入口（官方 CLI）。

跑情感表达管线打印 (v,a) 轨迹与关键中间量；或进入**交互对话**验证「情感引擎 ⊗ LLM 输出」耦合。
默认零可选依赖（不需要 ml/graphiti），总能起。用法：
  python main.py                       # 跑一组示例刺激（纯核心管线）
  python main.py --mood --language     # 开启慢变心境 / 语言层双向回路
  python main.py --recall              # 开启长期倾向记忆回灌（多轮才显效）
  python main.py --workspace           # 并行流 + 显著度门控全局工作空间（看哪些流点燃）
  python main.py --chat                # 交互对话：你说一句→引擎评价演化 e*/mood→LLM 生成带情绪回应
  python main.py --llm                 # 接真模型跑固定情绪场景的文本输出验证（批处理）

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
    """对话 transcript 落本地 SQLite（stdlib，无额外依赖）：逐轮存、重启重载近 N 轮 → 长期记忆。"""

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
    """交互对话：情感引擎 ⊗ LLM，情绪**真积累·响应·多样·部分随机**，不表演不扮演。

    每句话评价成 (v,a) 喂引擎得瞬时 e*；情绪状态 = e* 的**泄漏积分**（`mood_step` 的 self_gain=0
    退化版，非 A.7 双稳——双稳会自锁、被骂也不动）+ 小噪声 → 慢慢累积、被持续输入推动（被骂逐步
    变负、动怒）。映射到**多样情绪词**（affect_label）喂 converse，回应带真脾气、渐进、不戏剧化、
    能记起上文。transcript 与累积情绪都落本地 SQLite、重启续上。输入 exit/quit 退出。
    """
    _load_dotenv()
    # 本地持久：长期倾向 → sqlite（stdlib 即落盘）；运行态走 sqlite（缺 .[db] 自动回退内存）
    os.environ.setdefault("ZERO_CHECKPOINT_BACKEND", "sqlite")
    os.environ.setdefault("ZERO_MEMORY_BACKEND", "sqlite")
    for noisy in ("httpx", "openai", "src.memory.client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("src.storage").setLevel(logging.ERROR)  # 隐藏 sqlite 缺驱动的回退告警
    from src.agents.affect_math import clamp, mood_step
    from src.agents.emotion_lexicon import affect_label
    from src.agents.emotion_lexicon import appraise_text as lexicon_appraise

    api_key = os.getenv("ZERO_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("ZERO_OPENAI_MODEL")
    lm = None
    if api_key and model_name:
        from src.agents.language_openai import OpenAILanguageModel

        lm = OpenAILanguageModel(model=model_name)
        mode = f"真 LLM（{model_name}）"
    else:
        mode = "无 key 回退（词典评价 + 模板）"

    thread = os.getenv("ZERO_CHAT_THREAD", "chat")  # 可切独立会话/重置（默认 chat）
    log = ConversationLog()
    history = log.recent(thread, 20)  # 重载历史 → 跨重启记忆
    feeling = log.load_feeling(thread)  # 续上累积情绪（跨重启）
    # mood 关：A.7 双稳会自锁、不响应输入；改用 chat 侧泄漏积分（响应 + 渐进 + 不锁死）
    session = ConversationSession(thread_id=thread, mood_enabled=False, workspace_enabled=True)
    print(
        f"情感对话｜{mode}｜情绪积累·响应·部分随机｜历史已载 {len(history) // 2} 轮｜exit/quit 退出"
    )
    print(f"当前心情：{affect_label(*feeling)}｜记忆落 data/chat_history.sqlite3\n")
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
        # 轻阻尼喂引擎（保留方向与强度；渐进性交给下面的泄漏积分，而非阉割单句输入）
        stim = Stimulus(
            name=user[:40],
            goal_congruence=v,
            attitude_appeal=0.5 * v,
            intensity=min(1.0, max(0.2, abs(a))),
        )
        step = await session.step(stim)
        e = step["valence_arousal"] or (0.0, 0.0)
        # 泄漏积分（self_gain=0 不自锁；inertia=0.7 渐进、drive=0.3 响应）+ 噪声 → 累积·部分随机
        feeling = mood_step(feeling, e, inertia=0.7, self_gain=0.0, drive=0.3)
        feeling = (
            clamp(feeling[0] + random.gauss(0.0, 0.07), -1.0, 1.0),
            clamp(feeling[1] + random.gauss(0.0, 0.07), -1.0, 1.0),
        )
        word = affect_label(*feeling)
        history.append({"role": "user", "content": user})
        if lm is not None:
            reply = await lm.converse(history[-20:], feeling)
        else:
            reply = f"（{word}）嗯，我在听，你接着说。"
        history.append({"role": "assistant", "content": reply})
        log.append(thread, "user", user)
        log.append(thread, "assistant", reply)
        log.save_feeling(thread, feeling)
        print(f"\n{reply}")
        print(
            f"  └─ 你这句≈({v:+.2f},{a:+.2f}) | 心情={word} ({feeling[0]:+.2f},{feeling[1]:+.2f})\n"
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
    parser = argparse.ArgumentParser(description="情感表达多 Agent 系统 — 直接启动")
    parser.add_argument("--mood", action="store_true", help="开启慢变心境（A.7 滞后）")
    parser.add_argument("--language", action="store_true", help="开启语言层 affect↔language 回路")
    parser.add_argument("--recall", action="store_true", help="开启长期倾向记忆回灌")
    parser.add_argument(
        "--workspace",
        action="store_true",
        help="开启并行流 + 显著度门控全局工作空间（打印每刺激点燃的流）",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="交互对话：情感引擎 ⊗ LLM 输出耦合（缺 key 自动回退词典+模板）",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="接 OpenAI 兼容真模型做文本输出情绪验证（需 .[llm] + key）",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.chat:
        asyncio.run(_run_chat())
    elif args.llm:
        asyncio.run(_run_llm())
    elif args.workspace:
        asyncio.run(_run_workspace())
    else:
        asyncio.run(_run_core(args))


if __name__ == "__main__":
    main()
