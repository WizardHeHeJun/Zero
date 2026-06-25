# 软件工程师团队：工程落地的实现臂（设计）

> **日期**：2026-06-25
> **性质**：设计（非实现）。与「科学家议会」配对——议会管研究/算法/学术忠实（建什么、对不对），**软件工程师团队管工程落地（怎么实现、写出来、测出来、接进去）**。
> **缘起**：用户指出——科学家议会用于算法的讨论与研究，**真正的工程落地需要一套软件工程师**。本 note 设计这支实现臂，并接进 harness 主回路（议会 → 工程师团队 → 审查门 → 文档/沉淀）。
> **承接**：`notes/2026-06-25-science-council-design.md`（议会）、`notes/2026-06-25-harness-engineering-plan-and-assessment.md`（harness 工程逻辑与回路）。

---

## 一、分工：科学家议会 vs 软件工程师团队

| | 科学家议会 | 软件工程师团队 |
| --- | --- | --- |
| 管什么 | **建什么 / 对不对**（研究、算法、学术忠实） | **怎么落地**（实现、测试、接线、可靠性） |
| 阶段 | 设计期（评审/选型） | 实现期（把已评审设计变成可跑可测的代码） |
| 写代码？ | **否**（只读顾问，不下场） | **是**（这是本职：写实现 + 测试 + 接线） |
| 输出 | 带引文的设计决策（忠实/简化/失真） | 通过测试、合红线的代码 + 文档 |
| 对数据产生 | 不介入（以结果为分析起点） | **建造**生成机制本身，但绝不把 LLM/meta 塞进确定性热路径 |
| 工具 | Read/Grep/WebSearch（无 Edit/Write） | Read/Edit/Write/Bash（真正动代码） |

首尾相接：议会定"建什么/对不对" → 工程师团队"把它落地" → `code-reviewer` 独立把关 → 文档/沉淀。

---

## 二、团队角色（roster）

| 角色 | 职责 | 工具 |
| --- | --- | --- |
| **架构师** `engineer-architect` | 把已评审的设计落成模块/接口/层归属/数据流；产出文件级 `tasks`；守三层依赖与鸭子类型协议注入 | Read/Glob/Grep（读多写少，产出落点+tasks） |
| **实现工程师** `engineer-implementer` | 按 tasks 写实际代码，匹配同层既有风格，遵红线（确定性热路径/记忆纪律） | Read/Edit/Write/Glob/Grep/Bash |
| **测试工程师** `engineer-test` | 写单测/回归（边界、**零回归断言**、`importorskip` 跳过模式）；跑 `pytest` 直到绿 | Read/Edit/Write/Glob/Grep/Bash |
| **集成/可靠性工程师** `engineer-integration` | 接线（graph 注册/runner 贯通/env 工厂）、错误处理、async 生命周期、默认关零回归验证、性能/成本 | Read/Edit/Write/Glob/Grep/Bash |
| **主程/技术负责人** `engineer-lead` | 协调排序 tasks、把关交付、决定何时交 `code-reviewer`、对"落地了没"负责 | Read/Glob/Grep/Bash + 编排 |

**复用 `code-reviewer` 作独立审查门**（不重造）：职责分离——实现者 ≠ 审查者。

---

## 三、工作流（从已评审设计到落地）

入口 `/engineer <已评审设计 / PRP / 任务>`（与 `/science-council` 对称；PRP 的 `/execute-prp` 即调用本团队角色化落地）：

```text
已评审设计（议会 design.md / PRP tasks）
  ├─ 1 架构师：读 design+tasks → 定文件级落点（哪些文件/接口/复用点）
  ├─ 2 实现工程师：按 tasks 写代码（匹配同层风格、遵红线）
  ├─ 3 测试工程师：写测试 + 跑 pytest（零回归 + 覆盖）
  ├─ 4 集成/可靠性：接线 + env 工厂 + async + 默认关验证
  ├─ 5 code-reviewer（独立门）：BLOCK/WARN/INFO + PASS/NEEDS-CHANGES
  └─ 6 主程收口：未过则回环修，过则交 /generate-doc 同步 + /learn 沉淀
```

---

## 四、与现有 harness 的关系（不重复造轮子）

- **PRP** 是骨架（refine→generate→validate→execute）；软件工程师团队是 `/execute-prp` **实现阶段的角色化落地方式**。
- **`/engineer`** 作独立入口（拿到一份已评审设计就能落地），`/execute-prp` 引用之。
- **`code-reviewer`** 复用为审查门，不重造；测试工程师写测试、reviewer 独立审，职责分离。
- **科学家议会** 在上游（设计门）；工程师团队在下游（实现门）。算法语义有异议**回议会**，工程师不私自改科学决策。

---

## 五、护栏

- **写代码但守红线**：三层单向依赖；确定性热路径 torch/LLM-free、可复现、零回归；记忆写入节流 + 显式 scope；重依赖进 optional extra + 鸭子类型注入 + 默认关。
- **建造而非介入**：团队建造"引擎自己生成数据"的机制，**绝不把 LLM/meta/议会塞进运行时生成回路**（守 `analysis-results-first-no-intervene` 的另一面）。
- **不推翻科学**：议会定的学术决策不私自改；有异议回议会。
- **测试先于"完成"**：零回归 + 覆盖、`pytest` 绿才算落地。
- **并发纪律**：动 `src/` 前核实分支、不撞并行窗口（本仓常多窗口并发）。
- **职责分离**：实现者 ≠ 审查者，`code-reviewer` 独立把关。

---

## 六、落点（待确认后脚手架）

- `.claude/agents/`：5 个定义（`engineer-architect`/`engineer-implementer`/`engineer-test`/`engineer-integration`/`engineer-lead`）。
- `.claude/commands/engineer.md`：`/engineer <设计/任务>` 编排上述工作流；`/execute-prp` 引用。
- 同步 catalog/ai-docs；harness 主回路图更新为 **议会 → 工程师团队 → 审查门 → 文档/沉淀**（已在 harness-plan 回路体现）。

---

## 七、决策记录与下一步

- ✅ 确立分工：议会（研究/算法）⟂ 工程师团队（工程落地），首尾相接、职责分离。
- ⏳ **待确认 roster/协议**（5 角色是否合适、要不要并/拆），确认后脚手架 5 个 agent + `/engineer` 命令。
- 🔗 与议会同源纪律：默认关零回归、红线优先、可追溯。
