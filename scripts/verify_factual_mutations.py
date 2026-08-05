"""变异验证：逐个改坏阶段 62 相关实现，确认 test_factual_mode.py 的断言**会红**。

「绿灯必须先证明它能红」——不可证伪的绿灯比红灯更危险（会被越调越松）。
每个变异以逐字锚点临时替换源码 → 跑对应测试 → 立刻还原；任何变异后测试仍绿即退出码 1。

覆盖面（16 条）：事实化门控/边界段位置/情绪反塌陷/召回 LastValue 双入口归零/
push 中性死区/召回标签零回归/条款5非对称/舞台说明剥离（调用·行首规则·排除表）/截断事实门。

用法：``python -m scripts.verify_factual_mutations``（在仓根、任意已装 pytest 的解释器下）。

⚠ 锚点纪律：step() 与 run() 的归零行文本相同但缩进不同（12 vs 16 空格）——
短缩进单行锚是深缩进行的**后缀子串**（substring 不认行首），单行锚必须带独特上下文
消歧（见 pitfalls「逐字锚点替换」条）。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MUTATIONS: list[tuple[str, Path, str, str, str]] = [
    (
        "1 门控恒开（converse 局部 factual 强制 True）",
        REPO / "src/agents/language_openai.py",
        "        factual = factual_mode_enabled()",
        "        factual = True",
        "test_gate_shut_prompt_is_byte_identical",
    ),
    (
        "2 factual 开关用 not in ('','0') 解析（方向坑）",
        REPO / "src/agents/language_openai.py",
        '    return os.getenv("ZERO_FACTUAL_MODE", "").strip().lower() in ("1", "true", "yes", "on")',  # noqa: E501 —— 锚点须与源码逐字一致
        '    return os.getenv("ZERO_FACTUAL_MODE", "").strip() not in ("", "0")',
        "test_falsy_values_keep_the_gate_shut",
    ),
    (
        "3 边界段重复注入到 push 之前（count==1 应驱红）",
        REPO / "src/agents/language_openai.py",
        "        bias_kwargs: dict[str, Any] = {}\n        if push:",
        "        if factual:\n"
        "            sys += _FACT_BOUNDARY_ADDENDUM\n"
        "        bias_kwargs: dict[str, Any] = {}\n"
        "        if push:",
        "test_boundary_section_is_last",
    ),
    (
        "4 删掉末位边界段拼接（配合 3 构成真移动）",
        REPO / "src/agents/language_openai.py",
        "        if factual:\n"
        "            # 必须最后拼：最强近因位，且显式压过前置的人设卡与上面的召回/关系段。\n"
        "            sys += _FACT_BOUNDARY_ADDENDUM\n",
        "",
        "test_boundary_section_is_last",
    ),
    (
        "5 为压捏造把情绪条款一起删掉",
        REPO / "src/agents/language_openai.py",
        '    "你现在的真实心情是「{feeling}」——它**应该真实地**影响你的态度、语气和用词："\n'
        '    "高兴就轻快，被冒犯/不被理解就流露不耐烦、委屈或火气，低落就提不起劲。"\n'
        ")\n\n"
        "# 对应 _TEMPER_ADDENDUM。",
        '    "你现在的真实心情是「{feeling}」。"\n)\n\n# 对应 _TEMPER_ADDENDUM。',
        "test_open_gate_keeps_engine_driven_feeling_clause",
    ),
    (
        "6 撤回 step() 的 recalled_context/recalled_facts 归零",
        REPO / "src/orchestration/runner.py",
        '            "recalled_context": [],\n            "recalled_facts": [],\n',
        "",
        "test_runner_step_zeros_recalled_context_and_facts",
    ),
    (
        "7 死区默认改成开（悄悄改变默认行为）",
        REPO / "src/agents/emotion_lexicon.py",
        "    neutral_deadzone: bool | None = None,",
        "    neutral_deadzone: bool | None = True,",
        "test_mutation_push_deadzone_default_is_off",
    ),
    (
        "8 召回标签形参写死成事实化标签（破零回归）",
        REPO / "src/orchestration/chat_driver.py",
        '    recall_tag: str = "（记忆片段）",',
        "    recall_tag: str = FACTUAL_RECALL_TAG,",
        "test_recall_tag_defaults_to_original_label",
    ),
    (
        "9 撤回 run() 的五字段归零",
        REPO / "src/orchestration/runner.py",
        '                "external_priors": [],\n'
        '                "recalled_episode_ids": [],\n'
        '                "recalled_context": [],\n'
        '                "recalled_facts": [],\n'
        '                "recalled_disposition": None,\n',
        "",
        "test_runner_run_zeros_recall_lastvalue_fields",
    ),
    (
        "10 死区 env 用 not in ('','0') 解析（方向坑）",
        REPO / "src/agents/emotion_lexicon.py",
        '    return os.getenv("ZERO_PUSH_NEUTRAL_DEADZONE", "").strip().lower() in ("1", "true", "yes", "on")',  # noqa: E501 —— 锚点须与源码逐字一致
        '    return os.getenv("ZERO_PUSH_NEUTRAL_DEADZONE", "").strip() not in ("", "0")',
        "test_deadzone_env_falsy_keeps_off",
    ),
    (
        "11 撤回 step() 的 recalled_disposition 归零",
        REPO / "src/orchestration/runner.py",
        '            "recalled_disposition": None,\n        }',
        "        }",
        "test_runner_step_zeros_recalled_context_and_facts",
    ),
    (
        "12 撤回条款5的非对称规则（回退成断言否定可行）",
        REPO / "src/agents/language_openai.py",
        '    "反过来同样成立：你看到的历史是**被截断的**，找不到只说明「你找不到」，"\n'
        '    "不说明「他没说过」——不许断言「你没说过」「你压根没提过」这类否定；"\n'
        '    "也不要描述你并不具备的核验动作（「我翻遍了聊天记录」——你没有翻记录的能力，"\n'
        '    "你手上只有眼前这段上下文）。"\n',
        "",
        "test_boundary_forbids_absence_assertion",
    ),
    (
        "13 撤掉 converse 的剥离调用（机械执行层失效）",
        REPO / "src/agents/language_openai.py",
        "        if factual:\n"
        "            # 机械执行层：剥离舞台说明后才返回（调用方随即写入历史）——历史保持干净，\n"
        "            # 自我模仿雪球无从启动（见 strip_stage_directions docstring 的④臂实测记录）。\n"  # noqa: E501 —— 锚点须与源码逐字一致
        "            reply = strip_stage_directions(reply)\n",
        "",
        "test_converse_strips_stage_directions_when_factual",
    ),
    (
        "14 行首剥离退化成永不匹配（丢行首规则）",
        REPO / "src/agents/language_openai.py",
        '_LEADING_SEGMENT_RE = re.compile(r"^[ \\t]*[（(]([^（）()\\n]{1,40})[）)][ \\t]*")',
        '_LEADING_SEGMENT_RE = re.compile(r"(?!x)x")  # 永不匹配',
        "test_strip_stage_directions_line_leading",
    ),
    (
        "15 截断事实撤掉事实化门（模式关也注入=破零回归）",
        REPO / "src/orchestration/chat_driver.py",
        "    if dropped <= 0 or not factual_mode_enabled():",
        "    if dropped <= 0:",
        "test_truncation_fact_injected_only_when_factual_and_truncated",
    ),
    (
        "16 行首排除表掏空（回退成无条件剥=复活 5 类误伤）",
        REPO / "src/agents/language_openai.py",
        '    r"|[我你这那]"  # 第一/二人称与指称词开头：免责话术、引用、强调语\n'
        '    r"|指的是|指第|指代|注[:：]|即|例如|比如)"'
        "  # 指称/注释引导词；「指」收窄防「指甲…」误放行",
        '    r")"',
        "test_strip_stage_directions_keeps_legit_line_leading",
    ),
]


def main() -> int:
    failures: list[str] = []
    for name, path, old, new, test in MUTATIONS:
        original = path.read_text(encoding="utf-8")
        if original.count(old) != 1:
            print(f"[跳过] {name}: 锚点匹配 {original.count(old)} 次")
            failures.append(f"{name}（锚点未命中，变异未生效——源码措辞变了请同步锚点）")
            continue
        path.write_text(original.replace(old, new, 1), encoding="utf-8")
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    f"tests/test_factual_mode.py::{test}",
                    "-q",
                    "--no-header",
                    "-p",
                    "no:cacheprovider",
                ],
                cwd=str(REPO),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        finally:
            path.write_text(original, encoding="utf-8")
        went_red = proc.returncode != 0
        print(f"[{'✓ 会红' if went_red else '✗ 仍绿'}] {name}  →  {test}")
        if not went_red:
            failures.append(f"{name} → {test} 变异后仍然通过（断言不可证伪）")

    print()
    if failures:
        print("以下变异未能驱红：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"全部 {len(MUTATIONS)} 个变异均驱红：断言可证伪。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
