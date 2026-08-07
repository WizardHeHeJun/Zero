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

### 🟢 待机动作（2026-08-06 核验 · 尚未接入训练管线）

数字人**待机期**头部动作的数据源。⚠ RAVDESS 不适用于此场景——实测其头部角速度包络与
音频能量包络相关 |r| 中位 0.416、静默段仅占 19–20%，议会三轮判定「约 80% 是言语驱动头动，
余下是『即将开口』的预备态，没有一段是真正的安静待机」。详见
[notes/2026-08-06-motion-real-kinematics-route-council.md](notes/2026-08-06-motion-real-kinematics-route-council.md)。

| 数据集 | 内容 | 许可 | 规模 | 判定 |
| --- | --- | --- | --- | --- |
| **StayStill** | 50 人 × 3D 待机 BVH，30fps，含头部三轴 | **MIT**（⚠ 论文是 CC BY 4.0，**数据是 MIT**，勿混） | 纯待机 1:41:01 + 头部动作 0:56:54 ≈ **2h38m** | **可用·主选** |
| **ReActIdle** | 同组，**欺骗协议采集的真自发**待机 + 明确禁言 | MIT | genuine 15.2min + acted 对照 30.6min | **可用·作留出验证集** |
| ~~IdlePose~~ | 单目 2D 关键点 | 未知 | 未知 | **不可用**：无任何下载入口，且 2D 反解不出头部三轴 |
| GazeBase | 眼动 1000Hz，含眨眼标签（`lab` 列 −1） | CC BY 4.0 | 322 人 | 仅供**眨眼节律**：被试下巴托固定、无头动、任务刺激驱动 |

- StayStill：[Zenodo 18741736](https://zenodo.org/records/18741736)（368MB，零门槛）·
  [GitHub](https://github.com/Enekoassets/StayStill) · [arXiv:2605.13693](https://arxiv.org/abs/2605.13693)
  （SCA 2026 + Computer Graphics Forum 已接收）
- ReActIdle：[GitHub](https://github.com/Enekoassets/ReActIdle) ·
  [CAVW 2026, DOI:10.1002/cav.70116](https://doi.org/10.1002/cav.70116)（已出版）
- GazeBase：[figshare](https://doi.org/10.6084/m9.figshare.12912257) ·
  [Sci Data](https://doi.org/10.1038/s41597-021-00959-y)

#### ⚠ 接入前必读的四个坑（均为实测发现，论文未写）

1. **头部三轴要复合两个关节**：`freemocap/` 的 `face` 关节**只有 X 轴非零**（Y/Z 恒 0），
   三轴由 `neck` 承载 ⇒ 头部朝向 = `R_neck ∘ R_face`。只读 `face` 会得到单自由度铰链。
2. **`lafan/Head` 三通道恒为 0**（全折进 `Neck`）⇒ 读它会得到常量。用 `freemocap/`（未经重定向）。
3. **ReActIdle 的 BVH 文件头帧率是错的**：写 `Frame Time: 0.041667`（24fps），
   但按帧数反算论文自报时长得 **30fps**。照抄文件头会引入 25% 时基误差——
   动作整体放慢，**肉眼极难发现**。
4. **原始数据未清洗**：Zenodo 是 raw（README 明确要求生成器流程用 raw），含 pose estimation
   失败段；手部关节有 Euler 解缠爆炸（>1400°），**头部链干净**但整身重定向要注意。
   另 ReActIdle 的 `010_genuine`/`015_genuine` 头部几乎不动（std 比其余低一个数量级），
   疑似跟踪失败，占 genuine 总量 22%，**用前必须逐 clip QC**。

#### 已知限界

StayStill 是**表演的**待机（被试知情、被要求"演"等人），非自发。但同组
[CAVW 2026 对照实验](https://doi.org/10.1002/cav.70116)（123+114 名被试）实测：
**用户无法区分真实与表演的待机动画**，而**手工制作的（Mixamo）则明显可被区分**
⇒ 分界线在「录制的 vs 手搓的」，不在「自发 vs 表演」。ReActIdle 的 genuine 片段
可用于在本项目下游指标上独立复核这一结论。
另：三个数据集**均无眼球注视与眨眼**，该通道仍是开放缺口。

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
