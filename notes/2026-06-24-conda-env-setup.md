# Conda 环境配置记录 · affective-expression

> 2026-06-24 建立。本项目的 Python 隔离环境用 **conda** 管理。

## 关键事实

| 项 | 值 |
| --- | --- |
| conda 发行版 | Anaconda，装在 `E:\anaconda`（conda 26.1.1，base Python 3.13.9） |
| conda 入口 | `E:\anaconda\Scripts\conda.exe`（**未加入系统 PATH**） |
| 环境名 | `affective-expression`（与 `pyproject.toml` 的 `name` 一致） |
| 环境路径 | `E:\anaconda\envs\affective-expression` |
| Python | 3.12.13（项目要求 `>=3.12,<3.14`，ruff/mypy 均按 py312） |
| 依赖来源 | 主依赖 + dev 工具按 `pyproject.toml`，用环境内 `pip` 安装 |
| 验证 | `pytest` 全绿（27/27） |

## 配置文件

- `environment.yml` — 跨平台可重建，依赖口径对齐 `pyproject.toml`（loose 约束）。
- `environment.lock.yml` — 本机 win-64 实测精确版本快照（记录用途）。

## 常用命令

conda 不在 PATH，命令前用完整路径 `E:\anaconda\Scripts\conda.exe`（或先 `E:\anaconda\condabin\conda.bat init powershell` 把 `conda activate` 接好）。

```powershell
$conda = "E:\anaconda\Scripts\conda.exe"

# 重建环境（已存在时跳过）
& $conda env create -f environment.yml

# 在环境内跑命令（无需先 activate）
& $conda run -n affective-expression --no-capture-output python -m pytest -q
& $conda run -n affective-expression --no-capture-output ruff check .
& $conda run -n affective-expression --no-capture-output mypy .

# 同步依赖变更（改了 environment.yml 后）
& $conda env update -f environment.yml --prune

# 导出当前精确版本（更新 lock）
& $conda env export -n affective-expression --no-builds
```

## 备注

- 仓库本身已标准化用 **uv**（`uv.lock` + `[tool.uv]`）。本 conda 环境是按用户偏好额外提供的等价隔离环境，二者依赖口径都以 `pyproject.toml` 为准；改依赖时优先改 `pyproject.toml`，再同步回 `environment.yml`。
- `langgraph` 不暴露 `__version__` 属性，验证版本请用 `pip show langgraph`。
