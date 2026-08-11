"""守卫：MCP 工具体内的延迟 import 不得首次触达 numpy（否则 stdio 面静默挂起）。

## 为什么需要这条（配套项目 2026-08-11 实证的坑，我方今天恰好不踩）

**FastMCP stdio server 进入事件循环之后，在工具体里首次 `import numpy`（或任何传递性
拉 numpy 的包：scipy / onnxruntime / torch …）会无限期卡在扩展模块加载**
（Windows loader `LoadLibrary`），既不返回也不抛错。对方用 `faulthandler` 抓到确切栈，
并给了 20 行 stock FastMCP 最小复现；边界实测：

- import 放 `mcp.run()` **之前** → 正常（这就是修法）
- 塞进 `asyncio.to_thread` → **照样死锁**（线程化无效，最反直觉的一条）
- 判据是「**是否首次触达 numpy**」，不是「是否原生扩展」（sqlite3/PIL 都没事）

我方当前**结构性安全**：唯一会拉 torch/numpy 的构造（`_maybe_expression_decoder`）发生在
`build_server()` 里，而入口是 `build_server()` → `server.run()`，即 import 在事件循环之前
就完成了。工具体内那几个延迟 import（motion_synth / behavior_intent / language_openai）
实测都不拉 numpy。

**但这份安全没有守卫**：谁给工具体加一个用 numpy 的功能（很容易——比如给 `zero.motion`
加个统计量），单测会全绿、真 wire 上却挂起。本文件就是那道守卫：

🛑 **失败时不要改这个测试，要么把 import 提到 `build_server()` 里（事件循环之前），
要么确认那条链路真的不碰 numpy。**
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SERVER_SRC = REPO / "src" / "mcp_server" / "server.py"

# 构造期函数：它们在 `build_server()` 内（`server.run()` 之前）执行，拉重依赖是安全的。
_CONSTRUCTION_TIME_FUNCS = frozenset({"_maybe_expression_decoder", "build_server"})


def _deferred_imports_in_tools() -> set[str]:
    """AST 扫出「运行期才执行」的函数体内 import（排除构造期函数）。

    只看 `from X import ...` / `import X` 出现在某个函数体内的情形——模块顶层的 import
    在 server 模块加载时就完成，不在风险面内。
    """
    tree = ast.parse(SERVER_SRC.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name in _CONSTRUCTION_TIME_FUNCS:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.ImportFrom) and inner.module:
                found.add(inner.module)
            elif isinstance(inner, ast.Import):
                found.update(alias.name for alias in inner.names)
    return found


def test_scan_finds_the_known_deferred_imports() -> None:
    """先证明扫描器真能看见东西——否则下面那条会因为「扫出空集」而假绿。"""
    modules = _deferred_imports_in_tools()
    assert modules, "AST 扫不出任何函数体内 import：扫描器坏了，不是代码干净了"
    assert any("motion_synth" in m for m in modules)


@pytest.mark.parametrize("module", sorted(_deferred_imports_in_tools()))
def test_deferred_import_does_not_pull_numpy(module: str) -> None:
    """逐个模块在**干净子进程**里 import，断言它不会首次引入 numpy。

    必须开子进程：本测试进程早被 pytest/其它测试拉起过 numpy，`sys.modules` 已污染，
    在进程内测等于恒真（这正是「绿灯必须先证明能红」要防的那种假绿）。
    """
    code = f"import sys;__import__({module!r});sys.exit(1 if 'numpy' in sys.modules else 0)"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        cwd=str(REPO),
        capture_output=True,
        timeout=180,
    )
    if result.returncode == 1:
        pytest.fail(
            f"{module} 会拉起 numpy，而它是 MCP 工具体内的延迟 import ⇒ stdio 面首次调用"
            f"该工具会**静默挂起**（不报错、不超时）。修法：把 import 提到 build_server() 里"
            f"（server.run() 之前），或改用不碰 numpy 的实现。详见本文件模块 docstring。"
        )
    assert result.returncode == 0, f"{module} 无法导入：{result.stderr.decode('utf-8', 'replace')}"


def test_decoder_construction_stays_before_event_loop() -> None:
    """结构守卫：拉 torch 的解码器构造必须留在 `build_server()` 里。

    把 `_maybe_expression_decoder()` 挪进任何工具体 = 把 torch(→numpy) 的首次 import
    推到事件循环之后 = 复现对方那枚雷。本断言钉住调用点。
    """
    tree = ast.parse(SERVER_SRC.read_text(encoding="utf-8"))
    callers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "_maybe_expression_decoder"
            ):
                callers.add(node.name)
    assert callers == {"build_server"}, (
        f"_maybe_expression_decoder 被 {callers} 调用；它拉 torch→numpy，"
        "只能在 build_server()（server.run() 之前）调用"
    )
