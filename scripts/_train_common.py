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

**停机判据**：`PlateauStopper` / `resolve_epoch_budget` 把「跑多少轮」从 `epochs=300` 这个
魔数换成「训练 loss 进入平台或触上限」。⚠ 4 个裸循环脚本的 `train()` **默认 `stop="plateau"`**
——新写调用方（尤其是测试）若想要「精确跑 N 轮」，必须显式传 `stop="fixed"`，否则只传
`epochs=N` 会被静默按 plateau 语义跑到 `max_epochs`。既有调用方均已显式传参。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import random
import subprocess
import sys
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
"""sidecar 结构版本；字段增删或语义变更时 +1，读取方据此兼容旧文件。

v2（2026-07-27）：`model.state_dict_sha256` —— 权重**数值**的哈希。此前只有
`artifact_sha256`（文件哈希），而它会随输出文件名变化（见 `_state_dict_sha256`），
不足以回答「这份权重是不是那次训练产出的」。
"""

REPO_ROOT = Path(__file__).resolve().parents[1]

_MAX_HASH_BYTES = 512 * 1024 * 1024
"""超过此大小的数据文件不算 sha256（避免 GB 级语料把训练脚本卡在收尾）。"""

_GIT_TIMEOUT_SECONDS = 10

DEFAULT_PLATEAU_WINDOW = 100
DEFAULT_PLATEAU_REL_TOL = 1e-4
DEFAULT_MAX_EPOCHS = 5000


class PlateauStopper:
    """训练 loss 平台检测：每 `window` 步比一次相对下降，低于 `rel_tol` 即判定收敛。

    **为什么不用固定轮数**：`epochs=300` 这类魔数是拿单次实验凑出来的，换个学习率就
    静默失效——实测 `lr=3e-3 @300 步` ≈ `lr=1e-3 @1000 步`，真正被调的是 `lr × steps`
    的乘积，而不是任何一个单独的数。相对下降判据对 lr 变化免疫：lr 大就早停、lr 小就多跑，
    停在同一个平台上。

    **为什么是相对而非绝对阈值**：各通道 loss 量级差一个数量级以上（韵律 ~0.017、
    FACS ~0.035），绝对阈值必然要么对这个太松、要么对那个太严。

    判据只看**训练** loss，因此它回答的是「还学不学得动」，不是「泛化到没到最好」。
    有留出集的通道（文本三个）应当用 dev 早停，不要用这个。

    ⚠ **实测行为（2026-07-27，务必按这个理解它的作用）**：默认 `rel_tol=1e-4` 在本仓四个
    通道里**只有生理通道会真正触发**（真实 WESAD 约 2100 步）；FACS / 韵律 / expression 跑到
    5000 步时相对下降仍有 3.8e-4 / 1.6e-4 / 1.8e-2，会一直跑到 `max_epochs` 上限。原因是
    这些小 MLP 在小数据上呈**幂律式持续缓降**，压根没有明显的平台拐点。

    所以这套判据的实际作用是**「防欠收敛的上限保障」**，而不是「精确找到最优停机点」——
    它保证不会停在 300 这种明显欠收敛的地方，代价是多数通道会跑满上限。

    **为什么不放宽到 1e-3**：那样生理通道会停在 1400 步，而实测它 1400 处仍在欠收敛——
    2026-07-28 的**穷举 15 折 LOSO**（× 2 init 种子）显示 300 → 1500 的同切分配对差
    −0.001692（22/30 折支持多训）；反过来 FACS / 韵律跑满 5000 落在「1000 vs 2000 vs 5000
    配对检验全不显著」的区间内，只是慢一点、不损失质量。宁可多跑。

    ⚠ 早先此处引用的是「生理 val 单调降到 3000、argmin 众数 3000」，**该表述已被同一次穷举
    扫描收紧**：1500 → 3000 的配对差仅 −0.000063（19/30 折），即 **1500 之后收益已饱和**，
    3000 并非明显最优。这不改变「保留 1e-4」的结论（依据是 1400 欠收敛，该论据未变），
    但原措辞会让人以为 3000 有明确优势。
    """

    def __init__(
        self,
        *,
        window: int = DEFAULT_PLATEAU_WINDOW,
        rel_tol: float = DEFAULT_PLATEAU_REL_TOL,
    ) -> None:
        if window < 1:
            raise ValueError(f"window 须为正整数，得到 {window}")
        if rel_tol < 0:
            raise ValueError(f"rel_tol 不可为负，得到 {rel_tol}")
        self.window = window
        self.rel_tol = rel_tol
        self.reference = float("inf")  # 上一个窗口末尾的 loss；inf = 尚无参照

    def should_stop(self, step: int, loss: float) -> bool:
        """`step` 是**已完成**的轮数（从 1 起）。非窗口边界一律返回 False。

        第一个窗口只记录参照、不判停——没有前一个窗口就无从比较相对下降。
        """
        if step % self.window != 0:
            return False
        previous, self.reference = self.reference, loss
        if not math.isfinite(previous) or previous <= 0:
            return False
        return (previous - loss) / previous < self.rel_tol


def resolve_epoch_budget(
    *, stop: str, epochs: int, max_epochs: int
) -> tuple[int, PlateauStopper | None]:
    """把 `stop` 模式翻译成「循环上限 + 是否带平台检测」。

    `stop="fixed"` → 跑满 `epochs`、无检测（旧行为，供既有调用方逐字零回归）；
    `stop="plateau"` → 上限 `max_epochs`、带检测。其它值 fail-fast，不静默当默认处理。
    """
    if stop == "fixed":
        return epochs, None
    if stop == "plateau":
        return max_epochs, PlateauStopper()
    raise ValueError(f"stop 须为 'plateau' 或 'fixed'，得到 {stop!r}")


DEFAULT_VAL_FRAC = 0.2
MIN_HOLDOUT_GROUPS = 4
MIN_GROUP_SIZE = 2


def group_holdout(
    groups: Sequence[Hashable],
    *,
    val_seed: int = 0,
    val_frac: float = DEFAULT_VAL_FRAC,
    min_groups: int = MIN_HOLDOUT_GROUPS,
    min_group_size: int = MIN_GROUP_SIZE,
) -> tuple[list[int], list[int]]:
    """按**组**切出留出集，返回 `(train_idx, val_idx)`。整组进 train 或整组进 val，不拆。

    **为什么禁止朴素随机切分**：本仓这几个数据集里，同一个 `(v,a)` 锚点或同一个被试会对应
    几十上百行。按**行**随机切会让同组样本同时出现在两边——val 退化成「组均值下界」，
    模型只要记住组均值就能拿高分，于是给出「永远训更久更好」的错误信号。按组切才能让 val
    真正衡量「没见过的类/人」。

    **fail-fast 而非静默降级**：组数不足或有单样本组时直接抛错。此时切出来的 val 无统计
    意义，静默返回一个「能跑但没用」的切分，比报错危险得多。

    切分用独立的 `random.Random(val_seed)`，**不碰 torch 全局 RNG**——否则开不开留出集会
    连带改变模型初始化，两次实验就不可比了。
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac 须在 (0,1) 内，得到 {val_frac}")
    counts = Counter(groups)
    if len(counts) < min_groups:
        raise ValueError(
            f"留出切分需要至少 {min_groups} 个组，只有 {len(counts)} 个"
            "（组数太少时 val 无统计意义，故不静默降级）"
        )
    undersized = sorted(str(g) for g, c in counts.items() if c < min_group_size)
    if undersized:
        raise ValueError(
            f"这些组样本数 < {min_group_size}，无法构成有意义的留出：{undersized[:5]}"
            f"{' 等' if len(undersized) > 5 else ''}"
        )

    keys = sorted(counts, key=str)  # 先定序再洗牌：同 val_seed 必得同切分
    rng = random.Random(val_seed)
    rng.shuffle(keys)
    n_val = max(1, round(len(keys) * val_frac))
    n_val = min(n_val, len(keys) - 1)  # 至少留一组给 train
    val_keys = set(keys[:n_val])

    train_idx = [i for i, g in enumerate(groups) if g not in val_keys]
    val_idx = [i for i, g in enumerate(groups) if g in val_keys]
    return train_idx, val_idx


def evaluate_with_baselines(
    y_true: Any,
    y_pred: Any,
    groups: Sequence[Hashable],
    *,
    train_mean: Any,
    columns: Sequence[int] | None = None,
) -> dict[str, float]:
    """把裸 MSE 放进它该在的坐标系：常数基线在哪、组内下界在哪、这个模型走了多远。

    **为什么禁止单报裸 MSE**：0.031 这个数字本身毫无信息量——要看它落在
    `[组内下界, 常数基线]` 这个窗口的什么位置。FACS 的窗口是 `[0.0259, 0.0361]`，
    也就是说 0.031 只走完了可学空间的一半；换个通道同样的 0.031 可能已经是天花板。

    各项含义：

    - `mse` / `mse_class_balanced`：行加权 / **组等权**。两者会因大组落在哪侧而分道扬镳
      （FACS 38 个锚点行数 31–120 不均，实测切分间跨度可达 34%），所以两个都报。
    - `mse_constant`：永远预测训练集均值的 MSE。**打不赢它的模型没有价值。**
    - `within_group_floor`：每组用组内均值预测的 MSE。`(v,a) → y` 是多对一，组内方差
      学不掉，这是**任何** `(v,a)` 回归器可达的最低误差。
    - `skill_score` = `1 − mse/mse_constant`：0 ＝与常数预测同水平，1 ＝完美。
    - `learnable_excess` = `mse − within_group_floor`：还剩多少**能学而没学到**的误差。
    - `headroom_used` = `(mse_constant − mse) / (mse_constant − within_group_floor)`：
      在「常数基线 → 组内下界」这段可学空间里走完了几成。这比 skill_score 更公平——
      后者把学不掉的组内方差也算进了分母。
      ⚠ **名字里的「used」不保证落在 [0,100%]**：分母是 `mse_constant − within_group_floor`
      这段**可学空间**，通道间宽窄差很多（韵律实测仅 0.00435）。空间越窄，同样幅度的
      「跑输常数基线」被放大得越夸张——韵律的 −202% 并不代表错得比别人离谱 10 倍，只说明
      它那段可学空间本来就窄。**读这个百分比时必须同时看 `mse_constant` 与
      `within_group_floor` 的绝对差**，不要单独引用。

    `columns` 限定参与计算的输出维（FACS 用它剔掉运行时不消费的 4 维，见调用处）。
    """

    def _sel(t: Any) -> Any:
        return t if columns is None else t[:, list(columns)]

    yt, yp, mean = _sel(y_true), _sel(y_pred), _sel(train_mean)

    def _mse(a: Any, b: Any) -> float:
        return float(((a - b) ** 2).mean().item())

    idx_by_group: dict[Hashable, list[int]] = {}
    for i, g in enumerate(groups):
        idx_by_group.setdefault(g, []).append(i)

    per_group_mse: list[float] = []
    floor_weighted = 0.0
    for idx in idx_by_group.values():
        block = yt[idx]
        per_group_mse.append(_mse(yp[idx], block))
        # 组内均值是该组能达到的最好预测——组内方差是学不掉的部分
        floor_weighted += _mse(block.mean(dim=0, keepdim=True).expand_as(block), block) * len(idx)

    mse = _mse(yp, yt)
    mse_const = _mse(mean.expand_as(yt), yt)
    floor = floor_weighted / len(groups)
    span = mse_const - floor
    return {
        "mse": mse,
        "mse_class_balanced": sum(per_group_mse) / len(per_group_mse),
        "mse_constant": mse_const,
        "within_group_floor": floor,
        "skill_score": 1.0 - mse / mse_const if mse_const > 0 else float("nan"),
        "learnable_excess": mse - floor,
        "headroom_used": (mse_const - mse) / span if span > 0 else float("nan"),
        "n_groups": len(idx_by_group),
    }


def split_by_groups(
    x: Any,
    y: Any,
    groups: Sequence[Hashable],
    *,
    val_seed: int = 0,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> tuple[Any, Any, Any, Any, list[Hashable], dict[str, Any]]:
    """按组切分一对张量，返回 `(x_train, y_train, x_val, y_val, val_groups, info)`。

    `val_groups` 是 val 侧**逐行**的组标签——算组内下界必须知道哪些行属于同一组，而它对
    physiology 无法从 `x` 反推（X 只有 4 个取值），所以只能在切分时一并交出来。

    只用索引，不 import torch——`x[list_of_int]` 对 `torch.Tensor` 与 numpy 数组同样成立。
    `info` 汇总组数与样本数，直接进 sidecar 的 `val` 段：只报 MSE 而不说「基于几个组、
    几个样本」的话，读者无从判断这个数字有多少统计意义。
    """
    if len(groups) != len(x):
        raise ValueError(f"groups 长度 {len(groups)} 与样本数 {len(x)} 不一致")
    train_idx, val_idx = group_holdout(groups, val_seed=val_seed, val_frac=val_frac)
    info = {
        "val_seed": val_seed,
        "val_frac": val_frac,
        "n_train_groups": len({groups[i] for i in train_idx}),
        "n_val_groups": len({groups[i] for i in val_idx}),
        "n_train_samples": len(train_idx),
        "n_samples": len(val_idx),
    }
    val_groups = [groups[i] for i in val_idx]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx], val_groups, info


def add_val_split_arguments(
    parser: argparse.ArgumentParser, *, choices: Sequence[str], group_key: str
) -> None:
    """加 `--val-split` / `--val-seed`。默认 `none` ＝ 不切 ＝ 逐字旧行为。

    `choices` 各脚本自定：每个通道只有一种正确的分组方式（FACS/韵律按 `(v,a)` 锚点、
    生理按受试者），把不适用的选项摆出来只会诱导误用。
    """
    parser.add_argument(
        "--val-split",
        choices=list(choices),
        default="none",
        help=f"留出集切分方式（默认 none＝不切）；开启时按 {group_key} 分组，整组进一侧",
    )
    parser.add_argument(
        "--val-seed", type=int, default=0, help="切分随机种子（与模型 --seed 相互独立）"
    )


def add_stop_arguments(parser: argparse.ArgumentParser, *, epochs_default: int = 300) -> None:
    """给训练脚本加统一的停机判据三件套：`--stop` / `--max-epochs` / `--epochs`。

    `--epochs` 用 `default=None` 是为了区分「用户显式传了」与「没传」——plateau 模式下
    `--epochs` 不起作用，静默忽略会让人以为自己控制住了轮数，必须告警（见
    `resolve_cli_epochs`）。
    """
    parser.add_argument(
        "--stop",
        choices=["plateau", "fixed"],
        default="plateau",
        help="停机判据：plateau=训练 loss 进入平台即停（默认·对 lr 变化免疫）/ fixed=跑满 --epochs",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help=f"plateau 模式的轮数上限（默认 {DEFAULT_MAX_EPOCHS}）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"固定轮数，仅 --stop fixed 生效（默认 {epochs_default}）",
    )


def resolve_cli_epochs(args: argparse.Namespace, *, default: int = 300) -> int:
    """取 `--epochs` 有效值；plateau 模式下用户显式传了就告警，不静默吞掉。"""
    if args.epochs is None:
        return default
    if args.stop == "plateau":
        logger.warning(
            "--stop plateau 下 --epochs=%d 不生效（轮数由平台判据决定、--max-epochs 封顶）；"
            "要跑固定轮数请加 --stop fixed",
            args.epochs,
        )
    return args.epochs


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
                {
                    "class": type(model).__name__,
                    "param_count": _param_count(model),
                    "state_dict_sha256": _state_dict_sha256(model),
                },
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


def _state_dict_sha256(model: Any) -> str | None:
    """权重**数值本身**的哈希——对文件名与容器元数据免疫。

    为什么不能只靠 `artifact_sha256`（文件哈希）：`torch.save` 把**输出文件名**写进 zip
    条目前缀（实测同一个 state_dict 存成 `same_a.pt` / `same_b.pt`，条目分别是
    `same_a/data.pkl` / `same_b/data.pkl`，文件 sha256 因此不同）。所以文件哈希只在
    「输出路径相同」时可比：把权重改个名，它就对不上了，但权重其实一个 bit 都没变。

    要回答「磁盘上这份权重是不是那次训练产出的」，得看这个字段。
    """
    try:
        state = model.state_dict()
        digest = hashlib.sha256()
        for key in sorted(state):
            digest.update(key.encode("utf-8"))
            digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    except Exception as exc:
        logger.warning("state_dict 哈希不可得：%s", exc)
        return None


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
