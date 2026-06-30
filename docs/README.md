# docs/ — 框架图（v1 旧版存档 / v2 当前）

- **`v2/`** — **现阶段（2026-06-30，含阶段 25–27：注意力↔记忆桥 · 情绪稳态回弹 · 人格注入）权威架构图**。README 内嵌的就是这套。每张图有 `.mmd`（Mermaid 源，**当前 PNG 由它经飞书画板渲染**）+ `.json`（whiteboard-cli DSL，富样式备选）+ `.png`（渲染图）。
- **`v1/`** — 上一版存档（2026-06-25，阶段 13–24）：仅 `framework-current` / `runtime-flow` 两张的 `.json` + `.png`（whiteboard-cli DSL 渲染）。保留供对照，不再内嵌、不再维护。

## v2 文件

| 图 | 源 | 说明 |
| --- | --- | --- |
| `framework-current` | `.mmd` / `.json` / `.png` | **情绪引擎框架图**（总览）：评价桥 → 引擎 → 三时间尺度(含稳态回弹) → 双路语言 + 表达双通路 → 表现；记忆桥与人格注入旁挂 |
| `runtime-flow` | `.mmd` / `.json` / `.png` | **项目运作流程图（LLM ⊗ 情感引擎）**：一轮对话怎么走，标注 LLM 接点 / 确定性引擎 / 跨轮持久 / 人格接点 |
| `memory-architecture` | `.mmd` / `.json` / `.png` | **记忆架构图（注意力↔记忆桥）**：显著性门控写 → 情景库(时序失效/容量遗忘) → 三维召回重排 → 注入注意力预算；disposition 偏置评价 |
| `persona-injection` | `.mmd` / `.json` / `.png` | **人格注入图**：Persona 三层（L1 人设卡→语言 / L2 气质底色→引擎 / L3 预置关系→记忆），默认中性零回归 |

## 结构（情绪引擎框架图 · 左→右流水线）

```text
👤 用户输入 → 🔍 评价桥(文本→v,a) → 🧠 情感引擎(贝叶斯主动推断：OCC评价·TD价值/精度·显著度门控工作空间·后验采样 e*)
            → ⏱️ 三时间尺度(瞬时 e* / 快变 emotion / 慢变 attitude) + 稳态回弹(向 setpoint 弱回归)
            → 🗣️ 双路语言(LLM 命题 pull + 情绪 push 泄漏) ┐
            → 🎭 表达双通路(自发/随意 × 4 通道)            ┴→ 💬 最终表现
🎭 人格 Persona(可选) → 人设卡进语言 · 气质 setpoint 进时间尺度 · 预置关系进记忆
情感引擎 ⇄ 🗄️ 记忆/持久(注意力↔记忆桥：显著性写·三维召回·注入预算·时序遗忘)；三时间尺度 → 态度落盘
```

详解见 `PROGRESS.md` 阶段 13–27 与对应 `notes/2026-06-2*-*`（工作空间 · 记忆桥 · 情绪稳态 · 人格接口）。

## 重新渲染（v2）

**当前 PNG 经飞书画板渲染**（无需本地 `npx`，走 `lark-cli`；见 `lark-whiteboard` skill）：

1. 在某飞书文档建空白画板拿 token：`lark-cli docs +create --api-version v2 --title <标题> --content '<whiteboard type="blank"></whiteboard>' --doc-format markdown --as user`，从 `data.document.new_blocks[].block_token` 取 token。
2. 灌入 Mermaid：`lark-cli whiteboard +update --whiteboard-token <token> --input_format mermaid --source @docs/v2/<图>.mmd --overwrite --as user`
3. 导出图片：`lark-cli whiteboard +query --whiteboard-token <token> --output_as image --output docs/v2/<图>.png --overwrite --as user`
4. 裁白边（飞书导出画布偏大）：用任意图像库按内容包围盒 trim（如 Pillow `ImageChops.difference` + `getbbox`）。

**备选（富样式 whiteboard-cli DSL，需本地 Node）**：

```bash
npx -y @larksuite/whiteboard-cli@^0.2.0 -i docs/v2/<图>.json -o docs/v2/<图>.png
```
