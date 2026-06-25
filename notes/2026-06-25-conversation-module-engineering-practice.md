# 工程实践记录：对话模块扎实化（阶段 18–21）

- 日期：2026-06-25
- 范围：把"AI 人机交流对话"这一现阶段目标做扎实——从盘点未耦合子模块，到经科学家议会评审、软件工程师团队落地，分四个阶段把对话主干打通。
- 性质：**实践/过程复盘**（与 PROGRESS 的逐阶段技术记录、`notes/*-council.md` 的决策记录互补——本文记"怎么走的、为什么这么走、学到什么"）。

## 一、起点：盘点孤立子模块
以"打通 LLM 人机对话"为目标做全 `src/` 可达性分析，发现对话的 LLM I/O、记忆、情绪时间尺度**都跑在 LangGraph 图外**（main.py 缝合），并识别孤岛：`language_steering`（实现 generate 协议从未注入）、`text_affect` 输入通道（接进 PerceptionAgent 但 --chat 从不喂）。结论：对话"能跑"但不是"经系统打通"。

## 二、主回路：science-council ⟂ engineer-team，跑了三轮议会 + 四轮工程
严格按 harness 治理（议会管"建什么/对不对"、工程师管"怎么实现"、算法语义异议回议会、code-reviewer 独立门），本次密集应用：

| 议会 | 议题 | 关键裁决 | 落库 |
| --- | --- | --- | --- |
| 议会①（风险3→受约束c） | 文本测得情感接内核哪一环 | 否决"只进 features/替换 prior_mu"，定**独立低精度流进 fuse_terms**（不进 occ_prior/不污染 survival） | `2026-06-25-text-affect-prior-routing-council.md` |
| 议会②（可行性+架构） | 先对话后多模态可行吗/走哪条架构 | 可行、走 **hybrid**（感知图内/对话 REPL 图外）；多模态 N>2 融合算子**须再开议会** | `...-conversation-module-and-multimodal-roadmap-council.md` |
| 议会③（记忆语义） | 对话经历写什么/何时/scope/召回 | episode **确定性三层拼接**（禁 LLM 摘要）+ 显著性门控写入 + 选择性召回 | `...-conversation-episodic-memory-council.md` |

工程四轮（每轮 架构→实现→测试→集成→code-reviewer→收口，默认关零回归）：
- 阶段 18（PR #27 已合）：text_affect 受约束 c 接内核 + 修 features 布局 BUG。
- 阶段 19（PR #28 已合）：`ConversationModel` 协议 + 双存储边界（hybrid 定调）。
- 阶段 20（PR #29）：recall 回灌对话 + 长期 attitude 进 AppraisalAgent 先验。
- 阶段 21（PR #30）：语义召回通电（sqlite_vec 默认开）+ 显著性门控情景 episode + 侧信道失败隔离。

## 三、做成了什么：对话能力闭环
`python main.py` 现在跑的是一条端到端贯通的仿生人对话：
```
你说一句
 → 评价桥读成 (v,a)（appraise_text，图外 REPL）
 → 图内情感引擎：text_affect 独立流 + OCC 评价(含长期 attitude 先验) + 价值/生存流 → 显著度门控融合 → e*
 → 两时间尺度：快变 emotion(几轮衰退) + 慢变 attitude(慢累积、持久化)
 → push 通路：情绪经用词倾向自然漏进回应（converse，不"演"）
 → 显著性高的轮择要写情景 episode；下轮按相关性+情绪线索选择性召回回灌
```
每一步都"经系统"：协议可注入、记忆走分层、确定性热路径无 LLM/meta。一键恢复出厂 `tools/reset_db.py` 覆盖全部本地落盘点（含语义库）。

## 四、守住的红线（贯穿四轮）
- **确定性热路径**：情绪/记忆数据全由引擎确定性产生；LLM 只在图外 REPL 出语言内容，不进 affect 数学、不替仿生人"编造"记忆（gist 确定性拼接、禁 LLM 摘要）。
- **hybrid 不破**：converse 始终图外、graph 拓扑零改；感知/多模态才进图。
- **记忆纪律**：write_episode 仅任务完成节点（节流）、显式 scope、user_id=thread 隔离、运行态 transcript ⊥ 长期 episode、不直连图谱、语义侧信道失败绝不拖垮主对话。
- **默认关零回归**：每轮新能力默认/无后端时退化，pytest 一路绿（169→276 passed）。

## 五、学到的（已沉淀为跨会话记忆）
- 语义/外部测得情感信号接内核 = 独立低精度流，不进 occ_prior/不污染 survival（`text-affect-prior-routing-decision`）。
- 每阶段交付/合 PR 就地同步 PROGRESS + ai-docs，别攒；README 是对外窗口只放能力（`sync-progress-docs-per-stage`、`readme-outward-progress-local`）。
- 接受工程子代理交付前主程必跑全套 ruff/mypy，别只看 pytest 绿（`verify-lint-before-accepting-deliverables`）。
- 多窗口并发：分支操作前 `git fetch --prune`、对不上立即停手（见 `pitfalls.md`，本次踩过 stale-ref 误推已合并分支）。

## 六、下一道门槛
多模态扩展（FER/SER/HRV 进 fuse_terms）**必须先开议会评融合算子**（N>2 相关流 double counting 超线性放大 / 语言 top-down 调制 / 模态冲突仲裁）——见议会②③。完整记忆巩固与遗忘（McGaugh/Ebbinghaus 离线批处理）亦留后续。
