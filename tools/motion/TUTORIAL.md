# 动作层使用教程

> 按「我想做什么」组织。每节给完整命令、能看到什么、以及**判读结果的注意事项**。
> 工具清单见 [README.md](README.md)，自采协议见 [CAPTURE.md](CAPTURE.md)。

## 0. 一次性前置

```powershell
# conda 环境（本仓固定用它，不在 PATH，用完整路径）
E:\anaconda\Scripts\conda.exe run -n affective-expression --no-capture-output python -V
```

驱动皮套的脚本还需要：**VTube Studio 开着** + API 端口 8001 开启。

🛑 **同一时刻只能有一个脚本连 VTS**（参数注入独占，两个插件同时连会 454）。
上一个脚本没退干净就起下一个 = 连不上。

⚠ **两仓不能同时进 `sys.path`**：Zero 与 Zero_MCP 都用 `src.` 作包根。所以下面凡是
「驱动皮套」的命令都要求 `cd d:\Zero_MCP`，凡是「算数据」的都在 `d:\Zero\tools\motion`。
搞反了会报 import 错，不是脚本坏了。

首次连 VTS 会弹插件授权窗（插件名 `Zero-MCP Expression Bridge`）——**弹窗只在请求挂起
期间显示**，脚本超时退出后就没了。所以要在脚本等待时切过去点「允许」。
token 存 `data/steering/motion/vts_token`，之后不用再点。

---

## 1. 我想看看角色现在动起来什么样

```powershell
cd d:\Zero_MCP
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output `
  python d:\Zero\tools\motion\loop_vts.py        # 情绪巡回，Ctrl+C 停
```

不想连皮套、只想看轨迹长什么样：

```powershell
cd d:\Zero\tools\motion
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output python plot_motion.py
# → data/steering/motion/motion_demo.png（三面板：情绪直驱 / 意志调控 / idle 基线）
```

## 2. 我想看有哪些离散动作

```powershell
cd d:\Zero_MCP
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output `
  python d:\Zero\tools\motion\play_behaviors.py
```

12 词闭集逐个播一遍，含 `head_tilt` 两向、`glance` 四向，共 16 次触发。

⚠ 这条走的是 `BehaviorService.trigger` 通路，与轨迹通路**是两条不同的路**：
轨迹管「怎么动」（连续关键帧），离散词管「做什么动作」。运行时后者由 `behavior_intent`
从回复文本判出来，且**由我方转投** `behavior_trigger`——对方不解析我方返回体。

## 3. 我觉得幅度不对，想调

**别直接改常数**。用幅度梯度，一次把量级定下来：

```powershell
cd d:\Zero\tools\motion
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output python gen_ladder.py

cd d:\Zero_MCP
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output `
  python d:\Zero\tools\motion\play_ladder.py 2
```

四档递增，**不盲**（「该多大」是量级判断，二选一每轮只给 1 bit，梯度快得多）。
每档会报安全性标注：

| 标注 | 含义 |
| --- | --- |
| ✅ | 可用 |
| ⚠ margin 归零 | 不削平，但吃满量程——以后任何再加量的改动都会顶破 |
| ❌ 激动时削平波形 | 平时看不出来，一激动就露 |

🛑 **安全性按 `arousal=1.0` 判，不是按你观看时的唤醒档**。第一版按 0.8 判，
放过了一个激动时会削平的档位。

**天花板**：皮套参数 ±30° 是硬顶。真人待机 yaw sd 21.3° 在本皮套上**不可达**
（sd 21° 配 ±30 上限意味着约 16% 的时间在削平）。要更大只能在皮套侧改模型对
`ParamAngleX` 的视觉响应幅度，不在本仓范围。

## 4. 我想验证某个改动是不是真的更好

真机盲测是**唯一有效的验收**——议会明确：统计距离缩小**不保证**观感变好。

```powershell
cd d:\Zero\tools\motion
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output python gen_ab.py --list
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output `
  python gen_ab.py --variant coupling --seed 1

cd d:\Zero_MCP
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output `
  python d:\Zero\tools\motion\play_alt.py 3
```

**流程纪律**：盲看 → 给完观感 → 才公布 `ab_key.json`。先说答案会污染判断。
事后核对用 `check_ab_delta.py`（它读答案是正当的）。

三个容易翻车的地方：

1. **独立试次靠换 `--seed`**，不是连播同一份载荷 N 遍——后者只是重复曝光同一个样本。
   单次 2AFC 只是 n=1 伯努利观测，要多轮 + 二项检验（如 ≥9/10）。
2. **跑之前先做功效核算**。实例：yaw-roll 耦合改动在 10 秒片段上，两版的片段级相关
   分布**重叠**（40 种子实测），盲测等于掷硬币；稳的只有同种子配对次序（40/40）。
   所以要问"甲乙哪个更自然"，别问"这段侧倾对不对"。
3. **有些项在 10 秒片段上根本不可测**。`sway_hz` 一个周期 25 秒，装不进 10 秒——
   要么长时连播，要么承认它只有数据依据、无主观依据。

⚠ 加新变体时，**旧臂必须精确复现原实现**。踩过两次：只把系数换回旧值，却没复现旧公式
（旧式包络是新式的 1.12 倍），测出来的差异里混进了本不属于该项的幅度差。

## 5. 我想知道现在离真人有多远

```powershell
cd d:\Zero\tools\motion
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output python kinematics_gap.py
```

比的是**角速度分布**（无阈值量）。读法：

- 倍数**随分位单调上升** = 我们的分布太"散"：慢时太慢（观感=僵硬）、快时过冲（观感=过快）。
  这两个词其实是同一个缺陷的两端。
- 看 `p99/p50`（间歇性）：真人 10.3。远高于它 = 定住—猛动的对比度太强；
  远低于 = 会变成持续抖动。

🛑 **别用「移动/驻留占空比」当指标**：它要定阈，而实测真数据与合成数据**都不存在
阈值平台**（占空比随迟滞阈从 89% 连续变到 37%）——量出来的是阈值不是数据。

## 6. 我想用真人数据重新标定常数

完整协议见 **[CAPTURE.md](CAPTURE.md)**。最短路径：

```powershell
cd d:\Zero\tools\motion
# ① 先跑正控——没校过的尺子，采再多数据也只是把噪声量得更精确
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output python capture_selftest.py

# ② 采（用 VTS 摄像头跟踪，不用等设备）
cd d:\Zero_MCP
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output `
  python d:\Zero\tools\motion\capture_vts.py --subject s01 --seconds 300

# ③ 出提案
cd d:\Zero\tools\motion
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output `
  python capture_calibrate.py ..\..\data\steering\motion\captures --json proposal.json
```

**提案 ≠ 改动**。每项带「采信」或「否决 + 理由」，落地前依次过：
守卫测试 → 变异验证（把常数改坏，守卫必须会红）→ 真机盲测。

## 7. 我要接一份新的动作数据（BVH / 设备导出）

🛑 **接任何 BVH 之前先过 `anatomy.py`**，别手写角度提取。这一程在轴映射上错了**三次**：

- `freemocap`(StayStill) 是 Z-up 面朝 −Y，`ReActIdle` 是 Y-up 面朝 +Z，而**关节名完全相同**。
- 旧代码把关节局部 +Y 当"朝前"，实际是**朝后**。
- roll 用 `atan2(up_x, up_z)` 在 pitch≠0 时会混入正比于 yaw 的分量，凭空造出耦合
  （消融证实：把真实 roll 置零后旧公式仍报 +0.50~+0.96）。

**这三个错都不驱红**，数值上完全看不出来。`anatomy.py` 从骨架左右对称关节的 rest OFFSET
**实测**判定坐标系，roll 用 swing-twist 免泄漏，并配了会红的合成自检：

```powershell
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output python selftest_anatomy.py
```

设备导出的 CSV 走 `capture_ingest.from_csv`，其 `axis_signs` **必填、无默认值**——
录一段"只向右转头"实测确认每轴符号，别读文档猜（默认值就是猜）。

## 8. 我改了合成器常数，要怎么确认没搞坏

```powershell
cd d:\Zero
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output python -m pytest -q -k motion
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output ruff check .
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output mypy src
```

⚠ **测试绿不等于没问题**。这一程有**三条守卫是假绿灯**，全靠变异验证才挖出来：

| 守卫 | 假在哪 |
| --- | --- |
| clamp | 5 秒单种子，在真会削平 2% 帧的值上依然全绿 |
| 平静态幅度 | 同样采样不足，且 4.0° 阈值口径错配（引的是全身摆动文献，卡的是头部总峰值） |
| 零回归 | 翻转默认只红了一条**断言参数值**的测试，行为级全绿——顺这条线才挖出双通路调制被压成一路的真缺陷 |

所以改完常数**务必做变异验证**：把它改坏，看守卫会不会红。会红才算有守卫。

🛑 特别注意 **clamp 余量当前只剩 1.72°**。任何往幅度通路上加量的改动
（提高基幅 / 提高唤醒增益 / 新叠加项 / 接输出幅度更大的调制模型）都必须重跑
`test_clamp_never_actually_fires`——它已经没有缓冲了。

## 9. 常见故障

| 现象 | 原因 |
| --- | --- |
| 连不上 VTS | 另一个脚本还占着连接；或 VTS 没开/API 端口没开 |
| 首次连接卡住不动 | 在等你点授权弹窗——切到 VTS 点「允许」 |
| `ModuleNotFoundError: src.*` | 在错的目录跑了：算数据要在 `d:\Zero\tools\motion`，驱动皮套要在 `d:\Zero_MCP` |
| 动作看着"满幅摇头/顶住不动" | clamp 在削平波形。跑 `gen_ladder.py` 看当前档位的安全标注 |
| 盲测两版看不出差别 | 多半是功效不足，不是改动无效。先看 `check_ab_delta.py` 报的实际数值差异 |
