# 自采待机运动学数据：协议与管线

> 2026-08-07 搭建。起因：用户真机反馈「动作僵硬，并且动作过快」，而**现有两条路都实测走不通**
> （见下「为什么要自采」）。用户拍板后续自采，本文是那套框架的入口。

## 为什么要自采（两条已证伪的路，别重走）

| 路子 | 实测结论 |
| --- | --- |
| 已训好的 `artifacts/motion_decoder.pt` | `AnchorInterpolator`、**46 参数**、输入 (v,a) 输出**每段 3 个聚合标量**，训练集是 RAVDESS **演员摆拍说话**数据。provenance 里 `within_group_floor`=1.0586 vs `mse`=1.0907 ⇒ **可学习部分仅 3.2%**。关键不是分数低，是**没有时间结构**——而缺陷在速度分布形状，3 个标量改不了形状 |
| 议会批的二期「轻量线性-高斯状态空间（VAR/Kalman）」 | VAR 探针 p99/p50 只有 **2.7**（真人 10.3）：高斯创新项天生轻尾，复现不了爆发式头动。最大分位偏离 **3.88×**，而程序化合成器是 2.4× ⇒ **按原方案建出来比现状更差** |
| 公开数据（StayStill / ReActIdle） | 可用，且已用于现有常数标定。但**域不对**：被试是「在街上等人」可随意东张西望（yaw sd 21.3°），而对话中的数字人应大部分时间面向用户 |

⇒ 需要的是**对话场景下的待机头动**，公开数据里没有。

## 采什么

### 场景必须分开

🛑 **待机与说话是两个分布，绝不能混采一个集。** RAVDESS 就是栽在这：约 80% 的时长是
言语驱动的头动（实测头部角速度包络 × 音频能量包络 |r| 中位 0.416），拿它标定待机会全歪。

- `--scene idle`：**不说话**，自然放松地坐在你平时用数字人的位置、看着屏幕。
- `--scene speaking`：正常说话。⚠ 说话分支的时间结构必须由**实时 TTS 韵律流**驱动
  （议会裁定，Munhall 错配实验），采来的数据用于学「韵律→头动」映射器，不是回放。

### 量

| 项 | 下限 | 为什么 |
| --- | --- | --- |
| 单段时长 | **200 秒**（脚本默认 300s） | 200s 才有约 83 个姿态周期；更短时 sd 有 ~9% 波动、clamp 这类尾部事件根本抽不到（实测：同一配置短样本量 2.73°、长跑真值 3.95°） |
| 采样率 | **15Hz**（默认 30Hz） | 低于此，0.6s 量级的转移只剩个位数采样点 |
| 受试者 | **≥3 人**，越多越好 | 个体差异是本通道最大方差来源（韵律通道实测 61% 方差来自说话人身份），人太少会把某个人的习惯当成通则 |
| 每人场次 | ≥2 次，不同天 | 同一人不同天的姿势习惯也会漂 |

## 怎么采

### 路线 A：VTS 摄像头跟踪（现在就能用，推荐先跑通）

不用等设备，且**没有轴映射问题**——数据直接落在皮套参数空间，符号约定由 Live2D
官方文档明写。

```powershell
# 先跑正控，确认测量管线本身是准的（不需要任何真实数据）
cd d:\Zero\tools\motion
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output python capture_selftest.py

# 录（在 Zero_MCP 目录下跑；VTS 要开着、摄像头跟踪要生效）
cd d:\Zero_MCP
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output `
  python d:\Zero\tools\motion\capture_vts.py --subject s01 --seconds 300
```

### 路线 B：外接设备

新设备接进来只要写一个适配器，落成同一个规范格式即可：

```python
from capture_ingest import from_csv
capture = from_csv(
    "raw/s01.csv",
    subject="s01",
    columns={"yaw": "head_yaw", "pitch": "head_pitch", "roll": "head_roll"},
    axis_signs={"yaw": +1, "pitch": +1, "roll": -1},   # 🛑 必填，见下
    time_column="timestamp",
)
capture.to_npz("data/steering/motion/captures/s01-idle-1.npz")
```

🛑 **`axis_signs` 必须先实测确认，不许读文档猜**：录一段"只向右转头"，看该列是正是负。
本仓解剖约定是 **+yaw = 转向受试者自己的右、+pitch = 抬头、+roll = 头顶倒向自己的右**。
这一程在轴映射上错了**三次**——`freemocap` 是 Z-up 面朝 −Y、`ReActIdle` 是 Y-up 面朝 +Z
而**关节名完全相同**；OpenFace 文档不写"右"是谁的右；Live2D 的"右"是**画面**的右。
适配器刻意不给默认值：默认值就是猜。

⚠ 新增适配器时，**同步在 `capture_selftest.py` 加一条正控**（合成已知运动 → 过该适配器 →
能否量回原值）。没有正控的适配器不要用。

## 采完怎么用

```powershell
cd d:\Zero\tools\motion
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output `
  python capture_calibrate.py ..\..\data\steering\motion\captures --json proposal.json
```

输出是**带判据的提案**，不是直接改动。每项标「采信」或「否决 + 理由」：

| 常数 | 判据 |
| --- | --- |
| `AXIS_AMPLITUDE_RATIO` | sd 比值，无自由参数，直接可用 |
| `YAW_ROLL_COUPLING` | **增量**相关（去慢漂移）+ 逐片段**符号一致性**二项检验。符号不稳定 ⇒ 只是个体习惯，否决 |
| `SWAY_HZ` / `BREATH_HZ` | **频带边界敏感性**：挪动带下限，峰位跟着跑 ⇒ 是取带方式的产物、不是真峰，否决 |
| `MICRO_TREMOR_RATIO` / `POSE_RISE_S` | 拟合**角速度分布**的高尾（p90/p95/p99）——无阈值靶子 |

### 为什么靶子是角速度分布，不是「移动/驻留占空比」

占空比要定阈，而**实测真数据与合成数据都不存在阈值平台**（占空比随迟滞阈从 89% 连续
变到 37%）——量出来的是阈值不是数据。角速度分布不需要任何阈值。

而且它直接对应用户的两个词：改前我们的速度分位比真人是
p10 0.47× / p50 1.00× / p90 1.93× / p99 2.23×——**慢时太慢（僵硬）、快时过冲（过快）**，
是同一个分布缺陷的两端。

## 提案 → 落地的门

🛑 **提案不等于改动。** 依次过：

1. **守卫测试**：`pytest -k motion`。改幅度通路的必须重跑 `test_clamp_never_actually_fires`
   ——皮套 ±30° 的余量当前只剩 1.72°。
2. **变异验证**：把新常数改坏，守卫必须**会红**。本仓已三次抓到"看起来在测、其实什么都
   没测到"的假绿灯（clamp 守卫 5 秒单种子、平静态守卫口径错配、零回归守卫只断言参数值）。
3. **真机盲测**：`gen_ab.py --variant <项> --seed <每轮换>` + `play_alt.py`。
   ⚠ **跑之前先做功效核算**——耦合那项在 10 秒片段上两版分布重叠，盲测等于掷硬币。
   统计距离缩小**不保证**观感变好，这是议会明确的验收纪律。

## 已知天花板（采多少数据都绕不过）

皮套参数 ±30° 是硬顶：真人待机 yaw sd **21.3°**，在本皮套上**不可达**
（sd 21° 配 ±30 上限意味着约 16% 的时间在削平）。实测无论怎么配，
高唤醒 yaw sd 最多约 15.2°。要更大只能在皮套侧改模型对 `ParamAngleX` 的视觉响应幅度。

## 文件

| 文件 | 作用 |
| --- | --- |
| `capture_schema.py` | 规范格式 + 质量核查 + provenance 边车 |
| `capture_ingest.py` | 各源 → 规范格式（BVH 自动判轴 / 通用 CSV / VTS 跟踪） |
| `capture_vts.py` | 用 VTS 摄像头跟踪录自己的头动 |
| `capture_calibrate.py` | 采集 → 带判据的常数提案 |
| `capture_selftest.py` | **正控**：合成已知常数 → 走管线 → 能否量回（采数据前先跑） |
