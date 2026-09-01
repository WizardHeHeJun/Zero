# Bert-VITS2 补丁：`/voice` 响应头 `X-Phoneme-Durations`

供 lipsync-v2（音素级口型同步）用；Zero 侧代理没有 `D:\Bert-VITS2` 的写权限，
本补丁须由用户手工应用（见「应用方式」）。设计权威见
`PRP/lipsync-v2/design.md`（M1-M9），任务清单见 `PRP/lipsync-v2/tasks.md`（T1）。

## 契约字段

`/voice` 成功合成 wav 时，若模型是 `latest_version` 且能取到对齐信息，响应头新增：

```
X-Phoneme-Durations: {"phones":["b","o1",...],"durations":[7,12,...]}
```

- `ensure_ascii=True` + 紧凑分隔符（无空格），JSON 解析在 Zero 侧（`speech.py`）完成。
- `phones`：与文本发音顺序一致的**真实音素符号**列表（`text/chinese.py` 的
  `pinyin_to_symbol_map` 输出符号，如声母 `b`/`zh`，韵母 `a`/`ang`/`ong` 等，
  **不含**语言模型内部的 blank 占位符 `_`——见下「blank 折叠方向」）。
- `durations`：与 `phones` 等长，单位=**帧数**（w_ceil，非毫秒；Zero 侧
  `lipsync_phoneme.frame_ms_from_sample_rate` 负责换算为毫秒）。
- **任一段拿不到（模型非 `latest_version` / 取 `attn` 失败 / 任何未预期异常）
  ⇒ 整个响应不发该头**（不做部分成功）。Zero 侧头缺失时天然降级回 v1
  能量包络口型（`envelope_to_mouth_track`），无需特殊处理。

## blank 折叠方向（补丁内部完成，Zero 侧收不到 blank 语义）

Bert-VITS2 的 `get_text()` 在 `hps.data.add_blank=True` 时会用
`commons.intersperse` 在每个音素前后插入 blank 占位符 `_`（`net_g.infer`
返回的 `attn`——形状 `[b,1,t',t]`——对 t' 轴求和后，t 轴上每个位置的帧数
天然包含这些 blank 占位的帧数）。此外中文 g2p（`text/chinese.py::g2p`）
本身也会在整句首尾各加一个边界 `_`，与 intersperse 无关但符号相同。

补丁在**内部**把每一段连续的 blank 帧数折叠进相邻真实音素：

- 折叠方向 = **并入紧随其后的真实音素**（`_fold_blank_durations` 里的默认行为）；
- 唯一例外：整句**末尾**的 blank 游程没有"后一个"真实音素可并，此时**并入
  前一个**（最后一个真实音素）——纯粹是残余时长的归属选择，不代表协同发音
  前移（那是 design.md 已知简化 #3，独立于此）。

这是一个**有意的方向选择**、非中立事实，若后续核验发现某类音素（如爆破音
起始段）观感偏长，可复查是否与该折叠方向有关。

## 应用方式

```powershell
pwsh tools\tts_patch\apply.ps1 -RepoPath D:\Bert-VITS2
```

等价于（脚本是薄封装）：

```
git -C D:\Bert-VITS2 apply tools\tts_patch\0001-add-phoneme-durations-header.patch
```

应用前脚本会比对目标仓 `HEAD` 与 `VERSION` 记录的核验 commit——不匹配只警告，
不阻断（`git apply` 本身若上下文对不上会报错退出，是真正的把关）。

## 改动范围

- `infer.py`：新增 `_fold_blank_durations` 纯函数；`infer()` 新增可选参数
  `return_phoneme_durations`（默认 `False`，不传时行为与改前逐字一致）；
  `net_g.infer(...)` 的调用点保留原返回值 `[0][0,0]` 的取法不变，额外在
  `net_g_out[1]`（`attn`）上求 `phones`/`durations`；`skip_start`/`skip_end`
  会打乱音素-bert 对应关系，二者任一为真时直接跳过提取（不猜裁剪后的映射）。
- `hiyoriUI.py::_voice`：仅当 `loaded_models.models[model_id].version ==
  latest_version` 时才请求 `return_phoneme_durations=True`；多段循环内
  `extend` 累积 `phones`/`durations`（顺序与 `np.concatenate` 音频拼接顺序
  一致）；组 `Response` 前按上述「任一段拿不到⇒整个不发头」判据决定是否加头。
