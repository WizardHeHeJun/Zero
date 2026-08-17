# docs/ — 框架图（v1 旧版存档 / v2 当前）

- **`v2/`** — **现阶段权威架构图**（共 **12 张**：10 张架构图 + 2 张动力学曲线；与当前实现对齐）。README 内嵌的就是这套。架构图有 `.mmd`（Mermaid 源，**当前 PNG 由它经飞书画板渲染**）+ `.png`（渲染图）；旧四图另有 `.json`（whiteboard-cli DSL 富样式备选，**已不再维护、不保证与 `.mmd` 一致——一律以 `.mmd` 为准**；已知 `framework-current.json` 未收录实验特性旁挂等较新标注）。
- **`v1/`** — 上一版存档（2026-06-25）：仅 `framework-current` / `runtime-flow` 两张的 `.json` + `.png`（whiteboard-cli DSL 渲染）。保留供对照，不再内嵌、不再维护。

## v2 文件

| 图 | 源 | 说明 |
| --- | --- | --- |
| `framework-current` | `.mmd` / `.json` / `.png` | **情绪引擎框架图**（总览）：评价桥 → 引擎 → 三时间尺度(含稳态回弹) → 双路语言 + 表达双通路 → 表现；记忆桥与人格注入旁挂；**+ 四条虚线旁挂（默认关闭）：🧭 情感第三维 coping_potential（可选·区分愤怒/恐惧）/ 🫂 ToM 共情 / 🧪 HPA 皮质醇慢回路 / 🏛️ 层级预测编码（实验特性）** |
| `runtime-flow` | `.mmd` / `.json` / `.png` | **项目运作流程图（LLM ⊗ 情感引擎）**：一轮对话怎么走，标注 LLM 接点 / 确定性引擎 / 跨轮持久(emotion 会话内·attitude 落盘) / 人格接点 / 记忆·态度反哺评价先验两条回边 |
| `memory-architecture` | `.mmd` / `.json` / `.png` | **记忆架构图（注意力↔记忆桥）**：显著性门控写(含写入 dedup) → 情景库(容量遗忘·语义侧信道) → 三维召回重排 → 注入注意力预算；disposition 确定性图谱·时序失效·偏置评价；**+ 可选的巩固与遗忘旁挂（默认关）：分层幂律遗忘 / ACT-R 频率召回** |
| `persona-injection` | `.mmd` / `.json` / `.png` | **人格注入图**：Persona 三层（L1 人设卡→语言 / L2 气质底色·大五→PAD·va_coupling→引擎 / L3 预置关系→记忆），默认中性、行为不变 |
| `chat-persistence-map` | `.mmd` / `.png` | **--chat 数据落点地图**：一轮对话的数据各落哪个存储（对话运行态 / Checkpointer / 确定性图谱 / 语义侧信道 / 日志）、归哪个 env 变量管、chat 默认值 |
| `layered-architecture` | `.mmd` / `.png` | **三层架构图**：编排 → 记忆 → 存储只能向下依赖，observability 横切 |
| `mcp-boundary` | `.mmd` / `.png` | **MCP 接点边界图**：内核内建 MCP server（图示会话主回路 open/step/close 三工具；另有只读回读 describe_config 与清运行态 purge_session 两工具未入图）← MCP 协议 → 配套项目的感知输入 client + 执行操控 client；标出哪些在本仓、哪些在对面 |
| `workspace-ignition` | `.mmd` / `.png` | **工作空间点燃图**：并行流 (μ,π)（OCC 评价 / TD 价值 / 快生存 / 心境 / 文本语义 + 经 MCP 注入的外部多模态先验）→ 显著网络打分 → 过阈点燃广播 → 精度加权融合 → 后验读出 e*；另标出可选的「数值通路 ⊥ 报告通路」分离（`ZERO_IGNITION_GATE_FUSION`）与生理流排除（`ZERO_EXCLUDE_PHYSIO_FUSION`） |
| `coping-third-dimension` | `.mmd` / `.png` | **第三维分岔图**：同一个负效价高唤醒坐标 (v&lt;0,a&gt;0)，由情境控制感 `control_appraisal` 驱动的独立标量流分成愤怒(有掌控)/恐惧(失控)，再下传到判别性表情 AU（愤怒 AU23 / 恐惧 AU01·02·20）；默认关、核心 (v,a) 不受影响。总览图 `framework-current` 只标该能力存在，机制细节看这张 |
| `speech-pipeline` | `.mmd` / `.png` | **语音与口型同步链路图**：对话轮完成 → 语音出口（剥舞台说明→本地 Bert-VITS2 整句合成）→ 口型合成（能量包络→嘴部关键帧，对比度拉伸·开度限速·低能量忽略）→ `speech_play` 交渲染端同一时钟播放+注入；皮套出口并行走 `params_animate`/`behavior_trigger`，两出口共享同一条 MCP 连接；融合规则「嘴归语音（MouthOpen）·其余归情绪」标在边上 |
| `consolidation-forgetting` | `tools/plot_consolidation.py` / `.png` | **分层幂律遗忘曲线**：SESSION d=0.8 快衰 vs USER d=0.3 慢衰，叠高显著度的振幅调制。直接调 `src/memory/consolidation.py` 真实现生成，改参数重跑脚本即同步；该机制默认关（图/脚本已同步收窄为单一衰减曲线：原 SESSION→USER 升迁分支已随该机制移除） |
| `timescales-dynamics` | `tools/plot_timescales.py` / `.png` | **三时间尺度冲击-响应曲线**：单次冲击 vs 反复刺激双面板，`affect_math` 真方程轨迹（matplotlib 生成，改动力学参数后重跑脚本即同步） |

## 结构（情绪引擎框架图 · 左→右流水线）

```text
👤 用户输入 → 🔍 评价桥(文本→v,a) → 🧠 情感引擎(贝叶斯主动推断：OCC评价·TD价值/精度·显著度门控工作空间·后验采样 e*)
            → ⏱️ 三时间尺度(瞬时 e* / 快变 emotion / 慢变 attitude) + 稳态回弹(向 setpoint 弱回归)·习惯化
            → 🗣️ 双路语言(LLM 命题 pull + 情绪 push 泄漏) ┐
            → 🎭 表达双通路(自发/随意 × 4 通道)            ┴→ 💬 最终表现
🎭 人格 Persona(可选) → 人设卡进语言 · 气质 setpoint 进时间尺度 · 预置关系进记忆
情感引擎 ⇄ 🗄️ 记忆/持久(注意力↔记忆桥：显著性写·三维召回·注入预算·时序遗忘)；三时间尺度 → 态度落盘
🧭 情感第三维 coping_potential(可选·默认关) → 情境控制感驱动的独立标量流(与 v,a 正交、不入融合) → 表情判别性 AU · 用词区分愤怒/恐惧
⚗️ 实验特性(虚线·默认关闭·不开启则行为不变)：🫂 ToM 对方情绪→共情偏置进引擎 · 🧪 HPA 皮质醇→抬 arousal 基线/放大态度率 · 🏛️ 层级预测编码→引擎融合层级化
```

## 重新渲染（v2）

**当前 PNG 经飞书画板渲染**（无需本地 `npx`，走 `lark-cli`；见 `lark-whiteboard` skill）：

1. 在某飞书文档建空白画板拿 token：`lark-cli docs +create --api-version v2 --title <标题> --content '<whiteboard type="blank"></whiteboard>' --doc-format markdown --as user`，从 `data.document.new_blocks[].block_token` 取 token。
2. 灌入 Mermaid：`lark-cli whiteboard +update --whiteboard-token <token> --input_format mermaid --source @docs/v2/<图>.mmd --overwrite --as user`
3. 导出图片：`lark-cli whiteboard +query --whiteboard-token <token> --output_as image --output docs/v2/<图>.png --overwrite --as user`
4. 裁白边（飞书导出画布偏大）：用任意图像库按内容包围盒 trim（如 Pillow `ImageChops.difference` + `getbbox`）。

> ⚠ 画板 `+update` 是**异步**的：`+query` 立刻导出会拿到旧画布。批量渲染时导出后比对文件哈希、变化了才收图（轮询 5s×N），否则多张图会撞成同一张。
>
> `timescales-dynamics.png` 不走画板：`python -m tools.plot_timescales`（需任意环境装 matplotlib）直接由 `affect_math` 真方程生成。

**备选（富样式 whiteboard-cli DSL，需本地 Node）**：

```bash
npx -y @larksuite/whiteboard-cli@^0.2.0 -i docs/v2/<图>.json -o docs/v2/<图>.png
```
