# 真网络化所需数据集清单

> 把 `ExpressionAgent` 各通道的解析占位替换为真实数据训练的模型时所需的公开数据集。
> 获取方式：🟢 Kaggle/开放直下 · 🟡 需申请 EULA（填表授权）。数据放 `data/`（已 gitignore），权重存 `artifacts/`（已 gitignore）。

## 按通道对应（与代码契约一致）

```text
Stimulus(goal/standard/attitude) → (valence,arousal) → {FACS AU, 文本标签, 生理, 韵律}
        AppraisalAgent / 文本输入侧      AffectCore           ExpressionAgent 双通路
```

| 通道 / 组件 | 模型 | 数据集 | 获取 | 训练脚本 |
| --- | --- | --- | --- | --- |
| 全通道 bootstrap | `ExpressionDecoder` | 合成（无需外部数据） | — | `scripts/train_expression.py` |
| 韵律 prosody | `ProsodyDecoder` | **RAVDESS** | 🟢 | `scripts/train_prosody.py --root data/ravdess` |
| 生理 physiology | `PhysiologyDecoder` | **WESAD** | 🟢 | `scripts/train_physiology.py --root data/wesad` |
| 表情 FACS AU（13-AU 扩展·**已实跑**） | `FacsDecoder(extended=True)` | **emonet-face-binary**（CC-BY-4.0）+ OpenFace 抽 AU | 🟢 | `build_emonet_dataset` → OpenFace → `build_facs_ext_csv` → `train_facs.py --csv data/facs/labels_ext.csv --ext` |
| 表情 FACS AU（旧 5-AU·更高保真备选） | `FacsDecoder` | **AffectNet / DISFA / EmotioNet** | 🟡 | `scripts/train_facs.py --csv data/facs/labels.csv` |
| 文本→(v,a) 输入侧 | `TextAffectRegressor` | **EmoBank** | 🟢 | `scripts/train_text_affect.py --csv data/emobank.csv` |

## 数据集详情与链接

### 🟢 开放 / Kaggle 直下（推荐起步）

| 数据集 | 提供 | 规模 | 链接 |
| --- | --- | --- | --- |
| RAVDESS | 演员语音/歌唱，8 类情绪（文件名编码） | 1,440 条 | <https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio> |
| CREMA-D | 多族裔演员语音，6 类情绪（韵律备选） | 7,442 条 | <https://www.kaggle.com/datasets/ejlok1/cremad> |
| WESAD | 胸带/腕带生理（ECG/EDA/Temp…）+ 情绪态 | 15 人 | <https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection> |
| EmoBank | 文本 V-A-D（1–5 量表） | 10,062 句 | <https://github.com/JULIELab/EmoBank> · <https://www.kaggle.com/datasets/jackksoncsie/emobank> |
| NRC-VAD | 词级 V-A-D 词典（文本特征辅助） | 55k+ 词 | <https://saifmohammad.com/WebPages/nrc-vad.html> |
| crowd-enVent | 文本事件 + **评价维度**标注（贴 OCC） | ~6,600 | <https://www.romanklinger.de/data-sets/> |

> **表情通道现状**：13-AU 扩展模型已用 `laion/emonet-face-binary`（CC-BY-4.0，40 类合成脸）+ OpenFace 抽 AU 的 **EULA-free 路径**训出真权重，无需等待下表的 EULA 申请。下表的 AffectNet/DISFA 等仍是**更高保真的逐帧 AU 数据源**（自发表情、真人脸），可作后续升级选项。

### 🟡 需申请 EULA（填表授权，可能数天）

| 数据集 | 提供 | 链接 |
| --- | --- | --- |
| AffectNet | 人脸图 + 连续 V-A + AU + 表情 | <https://arxiv.org/abs/1708.03985>（官网申请） |
| Aff-Wild2 | in-the-wild 视频，逐帧 V-A + 12 AU | <https://ibug.doc.ic.ac.uk/resources/aff-wild2/> |
| DISFA | 自发表情，逐帧 12 AU 强度 | <http://mohammadmahoor.com/disfa/> |
| BP4D / EmotioNet | 自发 AU 标注（大规模） | 见各官网 |
| IEMOCAP / MSP-Podcast | 维度标注语音（V-A） | <https://sail.usc.edu/iemocap/> |
| DEAP | EEG + 外周生理 + V-A-D | <https://www.eecs.qmul.ac.uk/mmv/datasets/deap/> |

## 数据格式约定（loader 入口）

各 loader 已就位，把数据放到约定位置即可训练（详见对应 `src/agents/datasets/*.py` 模块文档）：

- **RAVDESS**：解压到 `data/ravdess/`（含 `Actor_xx/*.wav`）→ `ravdess.py` 按文件名第 3 段解析情绪码。
- **WESAD**：解压到 `data/wesad/`（含 `Sxx/Sxx.pkl`）→ `wesad.py` 按 condition 切窗、scipy 算心率。
- **FACS**：导出标注为 CSV `data/facs/labels.csv`，列 `valence,arousal,AU04,AU06,AU12,AU15,intensity` → `facs_csv.py`。
- **EmoBank**：放 `data/emobank.csv`（列 `id,split,V,A,D,text`）→ `emobank.py`，V/A 由 1–5 归一化到 [-1,1]。

## 许可与隐私

多为**研究用途授权**，注意商用限制与被试隐私条款；不要把原始数据/权重提交进版本库（`data/`、`artifacts/` 已 gitignore）。
