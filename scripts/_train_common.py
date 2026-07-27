"""训练脚本共用工具：权重 provenance sidecar。

**为什么存在**：此前各 `train_*.py` 只 `torch.save(state_dict)`，权重落盘后就与它的
配方（轮数 / lr / 种子 / 数据 / 代码版本）彻底脱钩。磁盘上那份 `.pt` 到底跑了多少轮、用的
哪版数据、对应哪个 commit，事后谁也说不清——`weights-v0.1` 就是这样一批「训得出、说不清」
的权重（详见 `WEIGHTS.md`）。没有 provenance，任何「A 比 B 好 2%」的结论都不可追溯。

**为什么写成旁挂 `.json` 而不是 dict checkpoint**：`src/agents/models/` 下 7 个 loader 全部是
`model.load_state_dict(torch.load(path, ...))`，读的是**裸 state_dict**（其中 facs / prosody /
physiology / expression 4 个还显式传了 `weights_only=True`，连 pickle 都不执行）。把 `.pt` 改成
`{"state_dict": ..., "meta": ...}` 会一次性破掉全部 loader，也会让已发布的 `weights-v0.1`
与新代码互不兼容。因此 provenance 一律写到 `<out>.pt.json`，**`.pt` 的内容格式一个字节都不动**。

**失败语义**：sidecar 是元数据，不是训练产物。git 不可用、数据文件读不动、json 落盘失败
——一律降级成字段 `null` 或直接放弃这份 sidecar，只留一条 warning。权重此时**已经保存**，
绝不能让记账失败把跑完的训练变成非零退出。
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
"""sidecar 结构版本；字段语义变更时 +1，读取方据此兼容旧文件。"""

REPO_ROOT = Path(__file__).resolve().parents[1]

_MAX_HASH_BYTES = 512 * 1024 * 1024
"""超过此大小的数据文件不算 sha256（避免 GB 级语料把训练脚本卡在收尾）。"""

_GIT_TIMEOUT_SECONDS = 10


def provenance_path(out_path: str | Path) -> Path:
    """权重路径 → 其 sidecar 路径（`foo.pt` → `foo.pt.json`）。

    刻意用**追加**后缀而非替换：`foo.pt` 与 `foo.json` 的对应关系靠约定，
    `foo.pt.json` 则自明，且不会和同目录里别的 `foo.json` 撞名。
    """
    return Path(str(out_path) + ".json")


def write_provenance(
    out_path: str | Path,
    *,
    script: str,
    model: Any,
    model_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    data_source: str | None,
    n_samples: int,
    seed: int,
    lr: float,
    epochs_requested: int,
    epochs_ran: int,
    final_train_loss: float,
    val: Mapping[str, Any] | None = None,
) -> Path | None:
    """在 `out_path` 旁写 `<out_path>.json`，记录这份权重的完整配方。

    Args:
        out_path: 刚保存的权重路径（`.pt`）；sidecar 写到 `<out_path>.json`。
        script: 产出该权重的脚本，如 `"scripts/train_facs.py"`。
        model: 已训练的 `torch.nn.Module`（只读 `type` 与 `parameters()`，本模块不 import torch）。
        model_config: 影响**模型结构**的参数（hidden / num_layers / extended / encoder…）。
        data_config: 影响**训练数据**的参数（limit / window_seconds / target…）。
        data_source: 数据集路径；`None` 表示合成数据（无外部来源）。
        n_samples: 实际参与训练的样本数（张量行数，非文件行数）。
        seed: `torch.manual_seed` 的种子。
        lr: 学习率。
        epochs_requested: 请求的轮数（`--epochs`）。
        epochs_ran: **实际**跑完的轮数；早停时小于 `epochs_requested`。
        final_train_loss: 最后一轮的**训练集** loss（早停恢复最优权重时，见 `val`）。
        val: 验证指标；无留出集时为 `None`——此时 `final_train_loss` 是训练集拟合度、
            **不是泛化指标**，sidecar 会显式标注这一点。

    Returns:
        写成功返回 sidecar 路径，失败返回 `None`（已记 warning，训练结果不受影响）。
    """
    out = Path(out_path)
    sidecar = provenance_path(out)
    try:
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact": out.name,
            "artifact_sha256": _sha256_file(out),
            "artifact_bytes": _size_bytes(out),
            "script": script,
            "created_utc": datetime.now(UTC).isoformat(),
            "model": _merge(
                {"class": type(model).__name__, "param_count": _param_count(model)},
                model_config,
                where="model_config",
            ),
            "training": _merge(
                {
                    "seed": seed,
                    "lr": lr,
                    "epochs_requested": epochs_requested,
                    "epochs_ran": epochs_ran,
                    "stopped_early": epochs_ran < epochs_requested,
                    "n_samples": n_samples,
                },
                data_config,
                where="data_config",
            ),
            "data": _describe_source(data_source),
            "metrics": {
                "final_train_loss": final_train_loss,
                "val": dict(val) if val is not None else None,
            },
            "git": _git_info(),
            "env": _env_info(),
        }
        if val is None:
            record["metrics"]["note"] = (
                "无留出集：final_train_loss 是训练集拟合度，不可当泛化指标读。"
            )
        sidecar.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        # 宽捕获是刻意的：调用点在 `torch.save` 之后，权重**已经落盘**，绝不能让记账环节
        # 把跑完的训练变成一次非零退出。下游各 helper 已按具体异常做字段级降级，这一层只兜
        # 没预料到的那类（例如传进来的 model 在 `.parameters()` 上抛自定义异常）。
        # 注意 `Exception` 不含 KeyboardInterrupt/SystemExit——用户中断照常穿透。
        logger.warning("provenance sidecar 未写出（权重已保存，训练结果不受影响）：%s", exc)
        return None
    logger.info("saved provenance -> %s", sidecar)
    return sidecar


def _merge(base: dict[str, Any], extra: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    """把脚本自定义配置并入固定字段。

    键冲突时**保留固定字段**并告警：核心配方（种子/轮数/样本数…）是 sidecar 的存在理由，
    绝不能被某个脚本随手传的同名键静默顶掉。
    """
    merged = dict(base)
    for key, value in extra.items():
        if key in merged:
            logger.warning(
                "provenance %s 的键 %r 与保留字段冲突，已忽略（值 %r）", where, key, value
            )
            continue
        merged[key] = value
    return merged


def _param_count(model: Any) -> int | None:
    """模型参数总数；拿不到就返回 None。

    宽捕获：`model` 是外部传进来的任意对象，`.parameters()` 可能抛任何东西。丢一个
    `param_count` 字段，好过让整份 sidecar 写不出来。
    """
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception as exc:
        logger.warning("param_count 不可得：%s", exc)
        return None


def _size_bytes(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _sha256_file(path: Path, *, max_bytes: int = _MAX_HASH_BYTES) -> str | None:
    """文件 sha256；不存在 / 过大 / 读不动一律返回 None。

    权重自身的哈希是 sidecar 与 `.pt` 的**配对凭证**——手工覆盖了 `.pt` 却留着旧 sidecar
    时，靠它能立刻发现两者对不上（用户侧 `sha256sum` 即可核对）。
    """
    size = _size_bytes(path)
    if size is None:
        return None
    if size > max_bytes:
        logger.info("跳过 sha256（%s 超过 %d 字节）", path.name, max_bytes)
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        logger.warning("sha256 计算失败（%s）：%s", path, exc)
        return None
    return digest.hexdigest()


def _describe_source(data_source: str | None) -> dict[str, Any]:
    """描述数据来源：合成 / 文件（带哈希行数）/ 目录 / 已不存在。"""
    if data_source is None:
        return {"kind": "synthetic", "path": None}
    path = Path(data_source)
    info: dict[str, Any] = {"path": str(data_source)}
    try:
        is_file, is_dir = path.is_file(), path.is_dir()
    except OSError:
        is_file = is_dir = False
    if is_file:
        info["kind"] = "file"
        info["size_bytes"] = _size_bytes(path)
        info["sha256"] = _sha256_file(path)
        info["lines"] = _count_lines(path)
    elif is_dir:
        # 目录（RAVDESS/WESAD 根）可达数 GB，不递归遍历、不哈希——只记路径与 mtime。
        info["kind"] = "directory"
        info["mtime_utc"] = _mtime_utc(path)
    else:
        info["kind"] = "missing"
    return info


def _count_lines(path: Path, *, max_bytes: int = _MAX_HASH_BYTES) -> int | None:
    """文本行数（CSV 含表头）；二进制 / 过大 / 读不动返回 None。"""
    size = _size_bytes(path)
    if size is None or size > max_bytes:
        return None
    try:
        with path.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError as exc:
        logger.warning("行数统计失败（%s）：%s", path, exc)
        return None


def _mtime_utc(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
    except OSError:
        return None


def _git(args: list[str]) -> str | None:
    """在仓库根跑一条只读 git 命令；git 缺失 / 非仓库 / 超时一律返回 None。"""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("git %s 执行失败：%s", " ".join(args), exc)
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _git_info() -> dict[str, Any]:
    """代码版本。`dirty=True` 时 commit 号并不足以复现——务必同时看它。"""
    commit = _git(["rev-parse", "HEAD"])
    status = _git(["status", "--porcelain"])
    return {
        "commit": commit,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        # 空字符串 = 工作区干净；None = 查不到（区别于「干净」，不可混淆）。
        "dirty": None if status is None else bool(status),
    }


def _env_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "torch": _package_version("torch"),
        "platform": platform.platform(),
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
