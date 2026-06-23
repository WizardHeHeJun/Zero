"""T6.7 层依赖：agents 不 import storage/不直连图谱；记忆/存储不上调。（pitfalls 3 / 依赖方向）"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def read_module_sources(package: str) -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in (SRC / package).glob("*.py")}


def imports(text: str, module: str) -> bool:
    """检测真实 import 语句（避免命中字符串/注释里的同名子串）。"""
    pattern = re.compile(rf"^\s*(?:from|import)\s+{re.escape(module)}\b", re.MULTILINE)
    return pattern.search(text) is not None


def test_agents_do_not_import_storage_or_graph_drivers() -> None:
    for name, text in read_module_sources("agents").items():
        assert not imports(text, "src.storage"), name
        # Worker Agent 不直接碰记忆层：记忆写入只由 Supervisor 节流（memory-rules #1）。
        assert not imports(text, "src.memory"), name
        assert "neo4j" not in text.lower(), name


def test_memory_does_not_import_orchestration() -> None:
    for name, text in read_module_sources("memory").items():
        assert not imports(text, "src.orchestration"), name
        assert not imports(text, "src.agents"), name


def test_storage_does_not_import_upper_layers() -> None:
    for name, text in read_module_sources("storage").items():
        assert not imports(text, "src.memory"), name
        assert not imports(text, "src.orchestration"), name
        assert not imports(text, "src.agents"), name
