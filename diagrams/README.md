# 架构图（diagrams/）

每个 `<时间戳>/` 目录含 `diagram.json`（源，v2 schema）+ `diagram.png`（渲染图）。
**PNG 由项目的图渲染工具从 `diagram.json` 生成**——改 json 后需重渲染该目录的 png。

| 时间戳 | 主题 | 状态 |
| --- | --- | --- |
| `2026-06-23T000000` | harness 通用三层架构（编排/记忆/存储）+ 情感旁路（4 步） | 历史·分层视角 |
| `2026-06-23T010000` | 早期设想：Supervisor → **6 脑区并行** → Integrator(全局工作空间) | ⚠️ **已被取代**：1:1「脑区=模块」不符神经科学（Pessoa 双重竞争/Barrett 简并）；现实落地为 v3 **显著度门控并行工作空间**（功能流并行 + ignition，见 `notes/2026-06-25-parallel-brain-pathways-and-workspace.md`） |
| `2026-06-23T020000` | 情感内核 ⇄ 语言一致性仲裁回路 | 部分有效：语言层已演化为**双路 pull/push**（`notes/2026-06-25-dual-route-language-push-pull.md`） |
| `2026-06-25T000000` | **当前完整结构**：评价桥 → 情感引擎(贝叶斯主动推断+工作空间) → **三时间尺度**(瞬时 e\*/快变 emotion/慢变 attitude) → 双路语言 + 双通路表达 → 记忆/落盘 | ✅ **现状·权威** |

> `2026-06-25T000000/diagram.json` 是当前权威结构源；其 PNG 待用项目图工具渲染。
> 结构详解见 `PROGRESS.md` 阶段 13–17 与对应 `notes/2026-06-25-*`。
