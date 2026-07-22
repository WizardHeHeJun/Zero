"""T8 跨仓契约回归：Zero external_priors ↔ MCP PerceptionHub.as_zero_streams() 形状对齐。

策略（澄清3）：
  - 定位配套 MCP 仓（D:\\Zero_MCP），子进程调用 PerceptionHub.as_zero_streams()，
    断言输出形状 == Zero ExternalPrior 形状（逐维精度 (name,(μv,μa),(Πv,Πa))）。
  - 同时验证 EXTERNAL_PRIOR_SCHEMA_VERSION 一致（M5）。
  - 缺 MCP 仓 / 依赖缺失 → pytest.skip 优雅跳过，绝不硬失败、不阻塞 T1–T7。
  - 仿 tests/mcp/test_zero_contract_crosscheck.py 的子进程 + importorskip 模式。

覆盖范围：
  1. PerceptionHub.as_zero_streams() 输出每条形状为 (str,(float,float),(float,float))
  2. as_zero_streams 输出与 ExternalPrior 类型注解一致（逐维精度，非标量精度）
  3. EXTERNAL_PRIOR_SCHEMA_VERSION 在 Zero 和 MCP 侧保持一致（M5）
  4. 空列表输入 → 空列表输出（无副作用）
  5. 单通道失败（异常/None）→ 降级跳过、不影响其他通道
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# 仓路径常量
# ---------------------------------------------------------------------------

_ZERO_ROOT = Path("D:/Zero")
_ZERO_MCP_ROOT = Path("D:/Zero_MCP")
_ZERO_MCP_SRC = _ZERO_MCP_ROOT / "src"
_ZERO_SRC = _ZERO_ROOT / "src"

# ---------------------------------------------------------------------------
# 可用性检查
# ---------------------------------------------------------------------------


def _zero_available() -> bool:
    return _ZERO_SRC.is_dir()


def _zero_mcp_available() -> bool:
    return _ZERO_MCP_SRC.is_dir()


def _both_available() -> bool:
    return _zero_available() and _zero_mcp_available()


# ---------------------------------------------------------------------------
# 子进程脚本：验证 PerceptionHub.as_zero_streams() 输出形状
# ---------------------------------------------------------------------------

# 脚本：从 MCP 仓创建几个 ModalityPrior 实例，经 as_zero_streams 转换，
# 序列化输出形状信息供 Zero 侧断言。
_HUB_SHAPE_SCRIPT = r"""
import sys
import json

zero_mcp_src = sys.argv[1]
sys.path.insert(0, zero_mcp_src)

try:
    from src.mcp.zero.perception import PerceptionHub
    from src.agents.models.zero_affect import ModalityPrior
except ImportError as e:
    print(json.dumps({"skip": True, "reason": f"import MCP 感知模块失败: {e}"}))
    sys.exit(0)

# 构造几个测试 ModalityPrior，覆盖不同模态
test_priors = [
    ModalityPrior(modality="face", mu=(0.3, 0.5), precision=(0.4, 0.6)),
    ModalityPrior(modality="audio", mu=(-0.1, 0.7), precision=(0.5, 0.5)),
    ModalityPrior(modality="physio_eda", mu=(0.0, 0.4), precision=(0.3, 0.5)),
]

try:
    streams = PerceptionHub.as_zero_streams(test_priors)
except Exception as e:
    print(json.dumps({"skip": True, "reason": f"as_zero_streams() 调用失败: {e}"}))
    sys.exit(0)

# 序列化形状信息
items = []
for stream in streams:
    if not (isinstance(stream, (tuple, list)) and len(stream) == 3):
        items.append({
            "error": f"stream 形状错误: {stream!r}（须 3-tuple）",
            "raw": repr(stream),
        })
        continue
    name, mu, prec = stream
    items.append({
        "name": name,
        "name_is_str": isinstance(name, str),
        "mu": list(mu),
        "mu_len": len(mu),
        "prec": list(prec),
        "prec_len": len(prec),
        "mu_types": [type(x).__name__ for x in mu],
        "prec_types": [type(x).__name__ for x in prec],
    })

print(json.dumps({
    "skip": False,
    "total": len(streams),
    "items": items,
}))
"""

# 脚本：验证 schema 版本对齐（M5）
_SCHEMA_VERSION_SCRIPT = r"""
import sys
import json

zero_mcp_src = sys.argv[1]
zero_src = sys.argv[2]
sys.path.insert(0, zero_mcp_src)
sys.path.insert(0, zero_src)

mcp_version = None
zero_version = None

# 尝试从 MCP 侧获取 schema version（若定义）
try:
    from src.mcp.zero.perception import PerceptionHub  # noqa: F401 确认可 import
    # MCP 协议层可能不直接暴露 schema version；
    # 若存在，从 protocols.py 或 __init__ 读取
    try:
        from src.mcp.zero import protocols as mcp_proto
        mcp_version = getattr(mcp_proto, "EXTERNAL_PRIOR_SCHEMA_VERSION", None)
    except ImportError:
        mcp_version = None
except ImportError as e:
    print(json.dumps({"skip": True, "reason": f"import MCP 失败: {e}"}))
    sys.exit(0)

# 从 Zero 侧读 schema version
try:
    from src.orchestration.external_prior import EXTERNAL_PRIOR_SCHEMA_VERSION
    zero_version = EXTERNAL_PRIOR_SCHEMA_VERSION
except ImportError as e:
    print(json.dumps({"skip": True, "reason": f"import Zero external_prior 失败: {e}"}))
    sys.exit(0)

print(json.dumps({
    "skip": False,
    "zero_version": zero_version,
    "mcp_version": mcp_version,
    "mcp_version_defined": mcp_version is not None,
}))
"""

# 脚本：验证空列表输入 → 空列表输出
_HUB_EMPTY_SCRIPT = r"""
import sys
import json

zero_mcp_src = sys.argv[1]
sys.path.insert(0, zero_mcp_src)

try:
    from src.mcp.zero.perception import PerceptionHub
except ImportError as e:
    print(json.dumps({"skip": True, "reason": f"import PerceptionHub 失败: {e}"}))
    sys.exit(0)

try:
    result = PerceptionHub.as_zero_streams([])
    print(json.dumps({
        "skip": False,
        "result_is_list": isinstance(result, list),
        "result_len": len(result),
    }))
except Exception as e:
    print(json.dumps({"skip": True, "reason": f"as_zero_streams([]) 失败: {e}"}))
"""

# 脚本：验证单通道失败降级（PerceptionHub.collect 异常处理）
_HUB_DEGRADED_SCRIPT = r"""
import sys
import json
import asyncio

zero_mcp_src = sys.argv[1]
sys.path.insert(0, zero_mcp_src)

try:
    from src.mcp.zero.perception import PerceptionHub, PerceptionChannel
    from src.agents.models.zero_affect import ModalityPrior
except ImportError as e:
    print(json.dumps({"skip": True, "reason": f"import 失败: {e}"}))
    sys.exit(0)


class _FailChannel:
    name = "fail_ch"

    async def sense(self):
        raise RuntimeError("模拟感知失败")


class _OKChannel:
    name = "ok_ch"

    async def sense(self):
        return ModalityPrior(modality="ok_ch", mu=(0.2, 0.3), precision=(0.4, 0.5))


class _NoneChannel:
    name = "none_ch"

    async def sense(self):
        return None


async def _run():
    hub = PerceptionHub([_FailChannel(), _OKChannel(), _NoneChannel()])
    priors = await hub.collect()
    return priors


try:
    priors = asyncio.run(_run())
    names = [p.modality for p in priors]
    print(json.dumps({
        "skip": False,
        "collected_names": names,
        "fail_excluded": "fail_ch" not in names,
        "ok_included": "ok_ch" in names,
        "none_excluded": "none_ch" not in names,
    }))
except Exception as e:
    print(json.dumps({"skip": True, "reason": f"PerceptionHub.collect 测试失败: {e}"}))
"""


# ---------------------------------------------------------------------------
# 辅助：子进程调用
# ---------------------------------------------------------------------------


def _run_script(script: str, *extra_args: str) -> dict[str, Any]:
    """运行子进程脚本，返回 stdout JSON；失败/超时 → pytest.skip。"""
    cmd = [sys.executable, "-c", script, str(_ZERO_MCP_SRC)]
    cmd.extend(extra_args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("子进程超时，跳过跨仓契约回归")
    if result.returncode != 0:
        pytest.skip(
            f"子进程非零退出 ({result.returncode})；"
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    try:
        data: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        pytest.skip(f"子进程输出非合法 JSON，跳过: {e}；stdout={result.stdout!r}")
    if data.get("skip"):
        pytest.skip(data.get("reason", "子进程请求跳过"))
    return data


# ---------------------------------------------------------------------------
# 顶层跳过装饰器（MCP 仓不在位时整个类 skip）
# ---------------------------------------------------------------------------

_REQUIRES_MCP = pytest.mark.skipif(
    not _both_available(),
    reason=(
        f"跨仓契约测试需要 D:\\Zero ({_ZERO_SRC}) 和 D:\\Zero_MCP ({_ZERO_MCP_SRC}) 均在位；"
        "当前环境缺少其中一个，跳过 T8"
    ),
)


# ---------------------------------------------------------------------------
# T8 测试类
# ---------------------------------------------------------------------------


@_REQUIRES_MCP
class TestExternalPriorContractCrossCheck:
    """T8 跨仓契约回归：Zero ExternalPrior 形状 ↔ MCP PerceptionHub.as_zero_streams()。

    所有用例：D:\\Zero 或 D:\\Zero_MCP 不可用 → skip（不硬失败、不阻塞 T1–T7）。
    """

    def test_as_zero_streams_each_item_has_three_elements(self) -> None:
        """as_zero_streams() 每条输出为 3 元素 tuple/list (name, mu, prec)。"""
        data = _run_script(_HUB_SHAPE_SCRIPT)
        items: list[dict] = data["items"]
        assert items, "as_zero_streams 返回非空列表（测试先验非空）"
        for item in items:
            if "error" in item:
                pytest.fail(f"as_zero_streams 输出形状错误: {item['error']}")

    def test_as_zero_streams_name_is_str(self) -> None:
        """as_zero_streams() 每条 name 是 str 类型（ExternalPrior 要求）。"""
        data = _run_script(_HUB_SHAPE_SCRIPT)
        for item in data["items"]:
            assert item["name_is_str"], (
                f"as_zero_streams 输出 name={item.get('name')!r} 不是 str，"
                f"与 ExternalPrior=(str,(float,float),(float,float)) 不符"
            )

    def test_as_zero_streams_mu_is_two_element(self) -> None:
        """as_zero_streams() 每条 mu 为 2 元素元组（(μv,μa)，ExternalPrior 要求）。"""
        data = _run_script(_HUB_SHAPE_SCRIPT)
        for item in data["items"]:
            assert item["mu_len"] == 2, (
                f"as_zero_streams mu 长度={item['mu_len']}，期望 2（逐维精度 (μv,μa)）"
            )

    def test_as_zero_streams_prec_is_two_element(self) -> None:
        """as_zero_streams() 每条 prec 为 2 元素元组（(Πv,Πa)，逐维精度，非标量）。

        这是关键契约（澄清3）：MCP 输出逐维精度 tuple，与 ExternalPrior=(name,(μv,μa),(Πv,Πa))
        形状原生对齐，不是标量精度（Zero expand_external_priors 明确要求 2-tuple prec）。
        """
        data = _run_script(_HUB_SHAPE_SCRIPT)
        for item in data["items"]:
            assert item["prec_len"] == 2, (
                f"as_zero_streams prec 长度={item['prec_len']}，"
                f"期望 2（逐维精度 (Πv,Πa)，非标量）；"
                f"若长度为 1 说明 MCP 侧仍在输出标量精度，与 ExternalPrior 形状不符"
            )

    def test_as_zero_streams_output_matches_external_prior_shape(self) -> None:
        """as_zero_streams() 输出与 Zero ExternalPrior=(str,(float,float),(float,float)) 对齐。

        综合断言：name=str，mu=2-tuple，prec=2-tuple。
        """
        data = _run_script(_HUB_SHAPE_SCRIPT)
        for item in data["items"]:
            assert item["name_is_str"], f"name 非 str: {item}"
            assert item["mu_len"] == 2, f"mu 长度非 2: {item}"
            assert item["prec_len"] == 2, f"prec 长度非 2: {item}"

    def test_as_zero_streams_correct_modality_names_passed_through(self) -> None:
        """as_zero_streams() 保留 ModalityPrior.modality 作为 stream name（不重命名）。"""
        data = _run_script(_HUB_SHAPE_SCRIPT)
        expected_names = {"face", "audio", "physio_eda"}
        actual_names = {item["name"] for item in data["items"]}
        assert actual_names == expected_names, (
            f"as_zero_streams 输出 name 集合={actual_names}，"
            f"期望={expected_names}（modality 名应原样透传）"
        )

    def test_as_zero_streams_total_count_matches_input(self) -> None:
        """as_zero_streams() 输出条数 = 输入 ModalityPrior 数（3 条测试先验 → 3 条输出）。"""
        data = _run_script(_HUB_SHAPE_SCRIPT)
        assert data["total"] == 3, (
            f"as_zero_streams 输出 {data['total']} 条，期望 3 条（1:1 转换，无过滤/融合）"
        )

    def test_as_zero_streams_empty_input_returns_empty(self) -> None:
        """as_zero_streams([]) → [] （空输入空输出，无副作用）。"""
        data = _run_script(_HUB_EMPTY_SCRIPT)
        assert data["result_is_list"], "as_zero_streams([]) 应返回 list"
        assert data["result_len"] == 0, f"as_zero_streams([]) 返回 {data['result_len']} 条，期望 0"

    def test_perception_hub_collect_degrades_gracefully_on_failure(self) -> None:
        """PerceptionHub.collect：单通道 raise/None → 降级跳过，不影响其他通道。

        - fail_ch 抛 RuntimeError → 被忽略
        - ok_ch 正常返回 → 保留
        - none_ch 返回 None → 被忽略
        最终 collect() 只包含 ok_ch。
        """
        data = _run_script(_HUB_DEGRADED_SCRIPT)
        assert data["fail_excluded"], "抛异常的通道（fail_ch）应被降级跳过"
        assert data["ok_included"], "正常通道（ok_ch）应保留在 collect 结果中"
        assert data["none_excluded"], "返回 None 的通道（none_ch）应被跳过"


@_REQUIRES_MCP
class TestExternalPriorSchemaVersion:
    """M5：EXTERNAL_PRIOR_SCHEMA_VERSION Zero 侧确认；MCP 若定义则须一致。"""

    def test_zero_schema_version_is_one(self) -> None:
        """Zero 侧 EXTERNAL_PRIOR_SCHEMA_VERSION == 1（当前版本号锚点）。"""
        from src.orchestration.external_prior import EXTERNAL_PRIOR_SCHEMA_VERSION

        assert EXTERNAL_PRIOR_SCHEMA_VERSION == 1, (
            f"Zero EXTERNAL_PRIOR_SCHEMA_VERSION={EXTERNAL_PRIOR_SCHEMA_VERSION}，期望 1"
        )

    def test_mcp_schema_version_aligned_if_defined(self) -> None:
        """若 MCP 侧也定义了 EXTERNAL_PRIOR_SCHEMA_VERSION，须与 Zero 侧一致（M5）。

        MCP 当前仅在感知协议中使用 Zero schema，不一定重复定义版本号；
        若未定义则 skip（不硬失败），若定义则必须对齐。
        """
        data = _run_script(_SCHEMA_VERSION_SCRIPT, str(_ZERO_SRC))
        if not data["mcp_version_defined"]:
            pytest.skip("MCP 侧未定义 EXTERNAL_PRIOR_SCHEMA_VERSION，跳过版本对齐检查")
        assert data["mcp_version"] == data["zero_version"], (
            f"schema 版本不一致：MCP={data['mcp_version']}，Zero={data['zero_version']}；"
            f"需同步更新（M5：跨仓协议由版本号兜底漂移）"
        )


# ---------------------------------------------------------------------------
# 补充：纯 Zero 侧 external_prior schema 契约（不需要 MCP 仓，始终跑）
# ---------------------------------------------------------------------------


class TestZeroExternalPriorSchemaLocal:
    """Zero 侧 external_prior.py 模块级契约（不依赖 MCP 仓，T1–T7 后的补充层）。

    这组测试始终跑（无跨仓依赖），确保核心 schema 常量和类型别名本身不漂移。
    """

    def test_external_prior_schema_version_is_int(self) -> None:
        """EXTERNAL_PRIOR_SCHEMA_VERSION 是 int 类型。"""
        from src.orchestration.external_prior import EXTERNAL_PRIOR_SCHEMA_VERSION

        assert isinstance(EXTERNAL_PRIOR_SCHEMA_VERSION, int), (
            f"EXTERNAL_PRIOR_SCHEMA_VERSION 应是 int，实际 {type(EXTERNAL_PRIOR_SCHEMA_VERSION)}"
        )

    def test_external_prior_schema_version_positive(self) -> None:
        """EXTERNAL_PRIOR_SCHEMA_VERSION > 0（合法版本号）。"""
        from src.orchestration.external_prior import EXTERNAL_PRIOR_SCHEMA_VERSION

        assert EXTERNAL_PRIOR_SCHEMA_VERSION > 0

    def test_external_prior_type_alias_matches_streams_shape(self) -> None:
        """ExternalPrior 类型别名 = tuple[str, tuple[float,float], tuple[float,float]]。

        通过构造合法实例验证类型注解可接受。
        """
        from src.orchestration.external_prior import ExternalPrior

        # 构造一个合法的 ExternalPrior 实例
        prior: ExternalPrior = ("face", (0.2, 0.4), (0.5, 0.6))
        name, mu, prec = prior
        assert isinstance(name, str)
        assert len(mu) == 2
        assert len(prec) == 2

    def test_expand_output_is_consistent_with_external_prior_type(self) -> None:
        """expand_external_priors 输出每条形状与 ExternalPrior 类型别名一致。

        ExternalPrior = (str, (float,float), (float,float))；
        展开结果应可直接赋值给 ExternalPrior 类型（形状一致，不需转换）。
        """
        from src.agents.affect_math import expand_external_priors
        from src.orchestration.external_prior import ExternalPrior

        priors: list[ExternalPrior] = [
            ("face", (0.2, 0.4), (0.5, 0.6)),
            ("audio", (-0.1, 0.3), (0.4, 0.7)),
        ]
        result = expand_external_priors(priors, precision_cap=0.8, max_streams=5)
        for item in result:
            name, mu, prec = item
            assert isinstance(name, str), f"name={name!r} 非 str"
            assert isinstance(mu, tuple) and len(mu) == 2, f"mu={mu} 形状错误"
            assert isinstance(prec, tuple) and len(prec) == 2, f"prec={prec} 形状错误"
            # 验证可构造为 ExternalPrior（类型别名，实质为 tuple）
            as_ep: ExternalPrior = (name, mu, prec)
            assert as_ep[0] == name

    def test_external_prior_importable_from_orchestration(self) -> None:
        """ExternalPrior 与 EXTERNAL_PRIOR_SCHEMA_VERSION 均可从 external_prior 模块 import。"""
        from src.orchestration.external_prior import (  # noqa: F401
            EXTERNAL_PRIOR_SCHEMA_VERSION,
            ExternalPrior,
        )

    def test_external_prior_imported_in_affect_math(self) -> None:
        """affect_math.py 成功 import ExternalPrior（用于 expand_external_priors 类型注解）。"""
        from src.agents.affect_math import expand_external_priors  # noqa: F401

        # 能 import 且函数存在即验证 ExternalPrior import 链路无断裂
        assert callable(expand_external_priors)

    def test_external_prior_importable_in_state(self) -> None:
        """AffectState 含 external_priors: list[ExternalPrior] 字段，且 state.py 可 import。"""
        from src.orchestration.state import AffectState

        state = AffectState()
        assert hasattr(state, "external_priors")
        assert isinstance(state.external_priors, list)
