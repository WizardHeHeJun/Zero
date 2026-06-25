# docs/ — 现阶段框架图

本目录存放项目**现阶段（2026-06-25）权威架构图**，并用**飞书画板**渲染托管（可在线协作编辑）。

## 飞书画板

- **在线文档**：<https://www.feishu.cn/docx/DrEydWZzXoW893x9jGKcQRZEnmc>
- 画板 token：`IgarwfJOkhWNyXbhHZhc9EyHn4c`
- 由 `framework-current.json`（whiteboard-cli v2 DSL）经 `whiteboard +update` 上传生成；改 DSL 后重新上传即可同步。

## 文件

| 文件 | 说明 |
| --- | --- |
| `framework-current.json` | 源 DSL（whiteboard-cli v2），现状权威结构，复制自 `diagrams/2026-06-25T000000/diagram.json` |
| `framework-current.png` | 本地 whiteboard-cli 渲染图 |

## 结构（左→右流水线）

```
👤 用户输入 → 🔍 评价桥(文本→v,a) → 🧠 情感引擎(贝叶斯主动推断：OCC评价·TD价值/精度·显著度门控工作空间·后验采样 e*)
            → ⏱️ 三时间尺度(瞬时 e* / 快变 emotion / 慢变 attitude)
            → 🗣️ 双路语言(LLM 命题 pull + 情绪 push 泄漏) ┐
            → 🎭 表达双通路(自发/随意 × 4 通道)            ┴→ 💬 最终表现
情感引擎 ⇄ 🗄️ 记忆/持久(图谱+语义·ConversationLog)；三时间尺度 → 态度落盘
```

详解见 `PROGRESS.md` 阶段 13–17 与对应 `notes/2026-06-25-*`。

## 重新渲染 / 同步飞书

```bash
# 本地渲染 PNG
npx -y @larksuite/whiteboard-cli@^0.2.0 -i docs/framework-current.json -o docs/framework-current.png

# 同步到飞书画板（DSL → openapi → +update）
npx -y @larksuite/whiteboard-cli@^0.2.0 --to openapi -i docs/framework-current.json --format json \
  | lark-cli whiteboard +update --whiteboard-token IgarwfJOkhWNyXbhHZhc9EyHn4c \
      --input_format raw --source - --overwrite --as user
```
