# 权重清单与溯源

各通道解码器权重的**出处、结构与校验值**。权重本体不进仓库（`artifacts/` 已 gitignore），
随 [Release `weights-v0.1`](https://github.com/WizardHeHeJun/Zero/releases/tag/weights-v0.1) 分发。

从现在起，每次训练会在权重旁自动写一份 **provenance sidecar**（`<权重>.pt.json`），记录轮数 /
学习率 / 随机种子 / 数据来源与哈希 / 代码 commit / 最终 loss。下面这张表是给**它之前**产出的
权重补的账——那时没有这套机制，所以有些字段是真的没记，如实标注为「未记录」，不做事后推测。

---

## `weights-v0.1` 已发布权重

下表的 sha256 与结构由 Release 附件**逐个下载读出**，不是从本地文件推断。

| 权重文件 | 方向 | 结构（输入→隐层→输出） | 参数量 | 数据集 | 发布时报告的 loss |
| --- | --- | --- | --- | --- | --- |
| `text_affect_regressor.pt` | 文本→(v,a) | 256 → 64 → 2 | 16578 | EmoBank | 0.016 ⚠ |
| `text_affect_regressor_st.pt` | 文本→(v,a) | 384 → 64 → 2 | 24770 | EmoBank（10062 句） | 0.0056 ⚠ |
| `prosody_decoder.pt` | (v,a)→韵律 | 2 → 16 → 3 | 99 | RAVDESS | 0.026 |
| `physiology_decoder.pt` | (v,a)→生理 | 2 → 16 → 3 | 99 | WESAD | 0.024 |
| `expression_decoder.pt` | (v,a)→表情 | 2 → 32 → 32 → 11 | 1515 | 合成（解析占位蒸馏） | demo |

> ⚠ **两个文本权重的 loss 不是泛化指标。** 它们训练时读了 EmoBank 全量——包括官方 dev/test，
> 那两个数字是**训练集拟合度**。下一节给出干净口径下的实测。其余三行的 loss 也都是训练集 loss
> （这些数据集没有官方切分），同样不能当泛化质量读。

### 文本两通道：干净口径实测（2026-07-27）

EmoBank 有官方 8062/1000/1000 切分。下面用同样 300 轮，只改「读全量」与「只读官方 train」，
在 **test 面**（test 全程不参与训练与早停）对比。技能分 = `1 − MSE/MSE_const`，
常数基线（永远预测训练集均值）test MSE = 0.02246：

| 通道 | 口径 | test MSE | 技能分 |
| --- | --- | --- | --- |
| 词袋 `text_affect_regressor.pt` | 污染（已发布权重的口径） | 0.01229 | 0.453 ⚠虚高 |
| 词袋 | 干净（重训后） | 0.02182 | **0.028** |
| 句向量 `text_affect_regressor_st.pt` | 污染（已发布权重的口径） | 0.00674 | 0.700 ⚠虚高 |
| 句向量 | 干净（重训后） | 0.01436 | **0.361** |

两条给使用者的实话：

- **词袋版干净口径的技能分只有 0.028**——比「永远预测均值」好不到 3%，实用价值很低。
  实测它对四句探针的预测全挤在 `|v| ≤ 0.053` 内，「best night of my life」甚至被判成负效价。
- **句向量版是有真实技能的（0.361）**，且与词袋版的差距比原先宣传的更大：不是「降 64%」，
  而是 0.028 对 0.361。同样四句探针它方向全对、幅度合理（`max|v| = 0.531`）。
  运行时置 `ZERO_TEXT_AFFECT_BACKEND=st` 启用句向量版。

## 干净口径重训（2026-07-27，尚未发布）

读取缺陷已修（loader 默认只读官方 train + dev 早停），并已按干净口径重训。**`artifacts/` 下这两份
本地权重已经是新的，与 `weights-v0.1` 的附件不再相同。**

每个通道跑 5 个种子、按 **dev** 选最优（test 只在最后评一次，不参与任何选择）：

| 权重文件 | dev MSE（5 seed mean±SD） | 选中 seed | 早停于 | test MSE | 技能分 | sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| `text_affect_regressor.pt` | 0.02170 ± 0.00003 | 2 | 125/300 | 0.02182 | 0.028 | `2aaa5a01f5ff8e598fd4c1e0d5d035ad2abf67dc67b9df9287edff3aade4586f` |
| `text_affect_regressor_st.pt` | 0.01362 ± 0.00009 | 4 | 177/300 | 0.01436 | 0.361 | `a3dd5ecb5e37b043632c9b3392f124438cd19c9a48063e25b820f6533ee32f4a` |

种子间标准差极小（0.00003 / 0.00009），说明这两个通道对初始化几乎不敏感——与需要多种子才能下结论的
表情通道不同，这里单次结果就是可信的。

> ✅ **provenance 干净，满足发布前置条件**：代码提交后按同样命令重跑，得到的**权重张量与提交前
> 逐一相同**（`torch.equal` 全等）。sidecar 记 `git.commit = 60a18b9`、`git.dirty = false`，
> 即这两份权重可以从该 commit 重建。
>
> 复现命令（在 `affective-expression` 环境内，仓库处于 `60a18b9`）：
>
> ```powershell
> python -m scripts.train_text_affect    --csv data/emobank.csv --epochs 300 --seed 2
> python -m scripts.train_text_affect_st --csv data/emobank.csv --epochs 300 --seed 4
> ```
>
> ⚠ **核对时请用 `state_dict_sha256`，不要用文件 sha256**：`torch.save` 会把**输出文件名**写进
> zip 条目前缀（实测同一个 state_dict 存成 `a.pt` / `b.pt`，条目分别是 `a/data.pkl` /
> `b/data.pkl`，文件哈希因此不同）。所以上表的文件 sha256 **只在输出到同一路径时可比**——
> 存到别处重跑，文件哈希必然不同，但权重一个 bit 都没变。sidecar 从 v2 起记录
> `model.state_dict_sha256`（权重数值的哈希，对文件名与容器元数据免疫），那才是
> 「这份权重是不是那次训练产出的」的判据。
>
> （本节两份权重的 sidecar 仍是 v1 格式、尚无该字段，下次重训时自动补上。）

校验值（`sha256sum <文件>` 可自行核对）：

| 权重文件 | 字节 | sha256 |
| --- | --- | --- |
| `text_affect_regressor.pt` | 68969 | `e4a3f4de8e24cfba17cb19317ae819dfb864b41a034399a3898a95ee4462c862` |
| `text_affect_regressor_st.pt` | 101831 | `7e6e9cf70fab81a95d0073657a6bc3b3e50626dde39827268ac2190370db1c6c` |
| `prosody_decoder.pt` | 2989 | `630e44de26557ebf16fe2d0d1f8450ea987bc98a895cebda4a8886e569fac27a` |
| `physiology_decoder.pt` | 3019 | `9d64c97d2e57784ad9acbb3435dd072cea3d3386058b49468d111007373e8f5b` |
| `expression_decoder.pt` | 9241 | `17c691de9b7b4359225e331f6bdc2932f4d167bcdf91b7633942b63ecd058568` |

结构一列对应各脚本的默认形状（`--hidden` / `--num-layers` 默认值）。**训练更大的网络需从头重训**，
这些权重只兼容默认形状。

### 这批权重没有记录的东西

| 字段 | 状态 |
| --- | --- |
| 训练/评估切分 | **两个文本权重读了全量**（含官方 dev/test），其余三个数据集无官方切分、也没有留出集 |
| 训练轮数 / 学习率 | 未记录。当时只保存 `state_dict`，命令行参数没有随权重留存 |
| 随机种子 | **不存在**。产出这批权重时，训练脚本尚未调用 `torch.manual_seed`，同一条命令每次跑出的权重都不同 |
| 数据快照 | 未记录。只知道数据集名称，不知道具体版本与行数 |
| 代码 commit | 未记录 |

因此**这批权重不可精确复现**——按同样的命令重训只能得到统计上相近、数值上不同的权重。
这正是 sidecar 机制要终结的情况：此后产出的每份权重都带完整配方，重训即可复现。

> `README` 里的 `--epochs 300` 是**示例命令**，不是任何已发布权重的配方；各权重的实际训练参数以本文件为准，
> 而本文件对这批权重的回答是「未记录」。

---

## 未随 Release 分发的本地权重

以下权重由仓内脚本产出但未发布，配方取自当时的工程记录，同样缺少种子等字段。

| 权重文件 | 方向 | 训练配方（已知部分） | 最终 loss |
| --- | --- | --- | --- |
| `facs_decoder_ext.pt` | (v,a)→11 AU | `train_facs --ext --epochs 500`，emonet-face-binary（CC-BY）→ OpenFace AU，1634 行 | 0.032 |
| `facs_decoder_ext_v2.pt` | (v,a)→13 AU | **2026-07-28 重训**：`train_facs --ext --seed 0`（默认 `stop=plateau`，实跑满 5000 步），同上数据管线 | 0.029479 |
| `expression_decoder_canonical.pt` | (v,a)→表情 | `train_expression --canonical-physiology`（idx7 = 体温，与 legacy 口径**不可互换**） | — |

### `facs_decoder_ext_v2.pt` 2026-07-28 重训：修 AU15 方向缺陷

**为什么重训**：旧版（2026-07-25）的 **AU15 沿 valence 方向与训练数据相反**——数据锚点级
相关 r = −0.526，模型却学成 Δ(v:−1→+1) = **+0.115**。AU15 不在 `_COPING_DRIVEN_AUS` 内，
composite 默认 α=1.0 下不被解析占位覆盖，所以那条错误方向**真实进入过运行时表情**。

根因是**全局欠训练**：旧版训于 ≈300–400 步（`epochs=300` 魔数时代的产物），而实测 AU15 的
valence 斜率符号在 ≤500 步时很大程度上是初始化残留，1500 步起才由数据接管。整改 P1-1 落地的
`stop="plateau"` + `max_epochs=5000` 已使默认配方必定跑满 5000 步，**重训一次即自动修好**。
完整根因见 `notes/2026-07-28-au15-direction-root-cause.md`。

| 量 | 旧（≈300–400 步·无种子） | **新（5000 步·seed 0）** |
| --- | --- | --- |
| Δ(AU15) @ a=0 | +0.115（方向反） | **−0.047（正确）** |
| 强信号维方向正确数（\|r\|≥0.5，6 维） | 5/6 | **6/6** |
| 全网格 13 键输出上界 | 0.377（从不出高强度 AU） | **0.850** |
| AU12 / AU06 全网格动态范围 | 0.220 / 0.258 | **0.760 / 0.681** |
| train MSE（单侧参考，非泛化指标） | 0.0343805 | **0.0294785** |
| provenance sidecar | ❌ 无 | ✅ 有 |
| sha256 | `0977777380…f473` | `b7ec88d7be…f12b` |

⚠ **不要读成「预测变准了」**：AU15 在留出面上**没有任何配置打赢常数基线**（最好 −0.0097）。
这次修的是**方向错误**，不是精度。

⚠ **旧权重无法精确复现**：它训于 `torch.manual_seed` 加进 `train_facs.py`（`798f0ef`,
2026-07-27）之前，没有固定种子、也没有 sidecar。文件本身留档于
`artifacts/archive/facs_decoder_ext_v2-300steps-noseed-2026-07-25.pt`，但**配方无法重建**。

复现新权重：

```powershell
python -m scripts.train_facs --csv data/facs/labels_ext.csv --ext `
    --out artifacts/facs_decoder_ext_v2.pt --seed 0
```

---

## 怎么用 sidecar

训练产出的每份权重旁会有一个同名 `.json`：

```text
artifacts/facs_decoder_ext_v2.pt        # 权重本体（裸 state_dict，格式不变）
artifacts/facs_decoder_ext_v2.pt.json   # provenance sidecar
```

sidecar 里的 `artifact_sha256` 是它与权重的**配对凭证**。手工替换过 `.pt` 而 sidecar 还是旧的时，
两边哈希对不上即可发现：

```powershell
sha256sum artifacts/facs_decoder_ext_v2.pt
python -c "import json;print(json.load(open('artifacts/facs_decoder_ext_v2.pt.json'))['artifact_sha256'])"
```

几个读的时候要留意的字段：

- `training.stopped_early` / `epochs_ran` —— 早停时实际跑的轮数，可能远小于 `epochs_requested`。
- `metrics.val` —— 为 `null` 表示**没有留出集**，此时 `final_train_loss` 是训练集拟合度，不是泛化指标。
- `training.official_split_used` —— 文本通道若为 `false`，说明官方切分不可用、已降级读全量数据。
- `git.dirty` —— 为 `true` 时工作区有未提交改动，光凭 `git.commit` 复现不出这份权重。

sidecar 是元数据：删掉它不影响权重加载，写入失败也不会中断训练。

### 发布权重时

`artifacts/` 整个目录不进版本库，sidecar 也一样——它只存在于训练那台机器上。因此**发布新权重时，
请把 `<权重>.pt.json` 与 `.pt` 一并作为 Release 附件上传**，否则下载权重的人拿到的仍是一份
没有配方的二进制，和上面 `weights-v0.1` 的处境没有区别。同时把本文件的表格补上新权重那一行。
