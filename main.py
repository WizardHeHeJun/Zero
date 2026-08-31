"""情感表达多 Agent 系统 · 本地启动 / 验证入口。

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

**本文件是临时验证入口，自身不含核心逻辑**——只做参数解析 + 装配，便于将来删除并迁移正式
主入口（届时 import 同样的 src/scripts 单元即可，逻辑不随本文件消亡）：
  - 对话核心 / 历史存储 → src/orchestration/chat_driver.py · src/storage/conversation_log.py
  - CLI 运行模式        → scripts/cli_modes.py（亦可 `python -m scripts.cli_modes ...` 单独跑）
  - 统一日志            → src/observability/

更专项的入口：
  python -m scripts.run_pipeline           # 端到端真网络化管线（合成训练 → 注入 → 跑）
  python -m scripts.verify_graphiti_local  # Graphiti 语义召回本地验证（需 .[graphiti] + LLM key）
"""

from __future__ import annotations

import argparse
import asyncio

from src.observability import setup_conversation_log, setup_logging


def _load_dotenv() -> None:
    """装了 python-dotenv 且有 .env 就加载；未装静默跳过。仅入口加载，库代码不依赖。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


async def _chat_repl() -> None:
    """交互对话 REPL：IO（读输入 / 打印 / 循环）留在入口，一轮对话核心逻辑在 chat_driver。

    本函数只负责把用户输入喂给 `ChatDriver.step` 并展示结果——不含任何情绪算法或存储逻辑，
    故删除 main.py 后，新入口换一套 IO 即可复用同一驱动器。
    """
    import logging
    import os

    _load_dotenv()  # 入口负责加载 .env（库/编排层不读 .env）
    # 「带对话内容」的专用日志：每轮 step 末尾写 logs/conversation-*.log（含 user/reply + 引擎
    # trace）。放 .env 加载后才能读到 ZERO_CONVERSATION_LOG（默认开，设 0 关）；仅对话路径接线，
    # 避免非对话模式产出空文件。
    setup_conversation_log()
    # chat 入口策略（属入口/部署选择，非对话核心）：本地落盘后端开箱即用（缺 .[db] 回退内存）+
    # 压制项目自身噪声日志。放入口而非 build_chat_driver，保编排层工厂纯装配、无全局副作用。
    os.environ.setdefault("ZERO_CHECKPOINT_BACKEND", "sqlite")
    os.environ.setdefault("ZERO_MEMORY_BACKEND", "sqlite")
    os.environ.setdefault("ZERO_SEMANTIC_BACKEND", "sqlite_vec")
    # chat 每轮即一次「任务完成」，长对话会持续写 episode。给语义库一个每-key 容量上限，
    # 配合 salience 门 + dedup 抑制长期增长（确定性 disposition 自带时序失效、不胀，无需另限）。
    os.environ.setdefault("ZERO_EPISODE_MAX_PER_KEY", "300")
    logging.getLogger("src.memory.client").setLevel(logging.WARNING)
    logging.getLogger("src.storage").setLevel(logging.ERROR)  # 隐藏 sqlite 缺驱动的回退告警
    from src.agents.emotion_lexicon import affect_label
    from src.orchestration.chat_driver import build_chat_driver
    from src.orchestration.voice_input import build_voice_input

    driver = build_chat_driver()
    voice = build_voice_input()  # None=未开（默认）；开=空回车进入 push-to-talk
    # 表现层出口（皮套/…）：构造在工厂，**连接在这里**——连接是 async I/O（spawn 子进程 +
    # 握手），工厂保持纯装配。连不上只降级为纯对话（sink 自己 warning），不中断启动。
    for sink in driver.expression_sinks:
        await sink.connect()
    rounds = len(driver.history) // 2
    print(f"情感对话｜{driver.mode}｜情绪快衰退·态度慢积累｜历史 {rounds} 轮｜exit/quit 退出")
    print(f"对你的态度：{affect_label(*driver.attitude)}｜记忆落 data/chat_history.sqlite3")
    if voice is not None:
        print("语音输入已开：空回车开始说话，再回车结束（数字人出声时别录，或戴耳机）")
    print()
    try:
        while True:
            try:
                # ⚠ 必须 to_thread：裸 input() 会阻塞事件循环——等输入期间语音 worker/
                # 后台任务全部饿死，上一轮的 TTS 要拖到下一句进来才播（2026-08-14 真机实测）。
                # 代价：Ctrl+C 恰在提示符时 input 线程仍挂着，进程收尾可能等到你按一次回车；
                # 常规退出走 exit/quit/EOF 不受影响。
                user = (await asyncio.to_thread(input, "你 > ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                if voice is None:
                    continue
                # push-to-talk：录音等待回车与模型推理都是阻塞操作，同样必须 to_thread
                # （理由同上——冻结事件循环会饿死语音 worker）。提示文案归入口层打印
                # （voice_input 只管采集/转写，换非 REPL 前端时不带 UI 泄漏）。
                print("● 录音中…再按回车结束")
                try:
                    user = (await asyncio.to_thread(voice.record_until_enter)).strip()
                except (EOFError, KeyboardInterrupt):
                    # 与打字路径同款优雅退出；代价同上（input 线程可能等一次回车）。
                    print()
                    break
                except Exception as exc:  # noqa: BLE001
                    # 外设/转写故障降级为提示继续打字（2026-08-31 真机实测：默认麦缺失
                    # PortAudioError 曾扳倒整个对话——表现层「降级不断话」纪律对输入侧同样适用）。
                    print(f"  └─ 语音采集失败：{exc}（可继续打字；换设备设 ZERO_ASR_INPUT_DEVICE）")
                    continue
                if not user:
                    print("  └─ （没听清，再试一次）")
                    continue
                print(f"你(语音) > {user}")
            if user.lower() in {"exit", "quit", ":q"}:
                break
            turn = await driver.step(user)
            v, a = turn.appraised
            print(f"\n{turn.reply}")
            print(
                f"  └─ 你这句≈({v:+.2f},{a:+.2f}) | attitude_appeal先验={turn.attitude_prior:+.2f}"
                f" | 情绪={turn.emotion_label} ({turn.emotion[0]:+.2f},{turn.emotion[1]:+.2f})"
                f" | 对你的态度=({turn.attitude[0]:+.2f},{turn.attitude[1]:+.2f})\n"
            )
    finally:
        for sink in driver.expression_sinks:
            # 逐个兜底（code-reviewer 2026-08-11）：清理链后面是记忆巩固与 sqlite 句柄
            # 释放（Windows 文件锁），表现层关不掉不该连累它们。
            try:
                await sink.aclose()  # 停动作循环、断渲染端连接（幂等）
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).warning("表现层关闭失败：%s", exc)
        await driver.aclose()  # B 类·会话结束巩固 + 关语义后端连接（默认关=no-op，零回归）
        driver.log.close()  # 显式释放 sqlite 句柄（Windows 防文件锁；W2）


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
    # 统一日志：每次启动落一份新日志文件（入口无关，实现见 src/observability）。
    # 本入口零核心逻辑：默认对话转发给 chat_driver；
    # --trace/--workspace/--llm 转发给 scripts.cli_modes。
    setup_logging()
    if args.trace or args.workspace or args.llm:
        from scripts.cli_modes import run_core, run_llm, run_workspace

        if args.trace:
            asyncio.run(run_core(args))
        elif args.llm:
            asyncio.run(run_llm())
        else:
            asyncio.run(run_workspace())
    else:
        asyncio.run(_chat_repl())


if __name__ == "__main__":
    main()
