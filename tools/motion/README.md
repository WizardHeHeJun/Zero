# tools/motion —— 动作层的验收与标定工具

2026-08-07 从会话 scratchpad 转正（临时目录会被清理，转正前每次都要重建）。
路径统一走 `_paths.py`，产物落 `data/steering/motion/`（已 gitignore）。

## 前置

- conda 环境 `affective-expression`
- 驱动皮套的脚本需 VTube Studio 开着、API 端口 8001 开启
- 🛑 **同一时刻只能有一个脚本连 VTS**（参数注入独占，双插件会 454）

⚠ **两仓不能同时进 `sys.path`**：Zero 与 Zero_MCP 都用 `src.` 作包根，会互相覆盖。
故「生成载荷」（Zero 侧）与「驱动 VTS」（Zero_MCP 侧）必须拆成两个进程——
`_paths.use_zero()` / `_paths.use_zero_mcp()` 二选一，不可混用。

## 分析（Zero 侧，不碰 VTS）

| 脚本 | 用途 |
| --- | --- |
| `anatomy.py` | **坐标系自动判定 + 免泄漏角度提取**（库，非脚本）。从骨架左右对称关节的 rest OFFSET 实测出「左/前/上」，roll 用 swing-twist 消除 pitch×yaw 泄漏。⚠ 接任何 BVH 前先过它——`freemocap` 是 Z-up 面朝 −Y、`ReActIdle` 是 Y-up 面朝 +Z，**关节名完全相同** |
| `selftest_anatomy.py` | `anatomy.py` 的合成自检（无需数据集）。含"应该红"的变异项：零真实 roll 的 pitch×yaw 复合下，旧公式必须给出 ±1.0 的伪相关 |
| `coupling_measure.py` | **yaw-roll 耦合测定**，五条判据（解剖锚定/免泄漏/符号一致性/跨数据集/伪影归因）+ 消融对照臂。结论见 `notes/2026-08-07-motion-idle-constants-criteria.md` |
| `idle_criteria.py` | 转移·驻留时长与呼吸频率的判据检验：分割器**正控**（对已知真值）、阈值不变性扫描（含合成对照）、谱峰边界敏感性 |
| `head_final.py` | **轴映射验证** + 待机三轴分布。判别式判据：比较同一通道在「低头类」vs「转头类」动作上的响应比。⚠ 别用「主导轴须等于期望轴」——真人「看手表」是复合动作。⚠ 其 roll 提取带泄漏，**耦合符号以 `coupling_measure.py` 为准** |
| `idle_constants.py` | 从 StayStill 待机数据提取全部程序化常数的同域实测值，并与当前手写值对照 |
| `plot_motion.py` | 把合成轨迹画成图（三面板：情绪直驱 / 意志调控 / idle 基线） |
| `check_ab_delta.py` | 事后核对盲测两版的**实际**数值差异（防「以为改了 A 实际改了 B」） |
| `play_behaviors.py` | **离散动作巡演**：12 词闭集逐个放到皮套上（含 `head_tilt` 两向、`glance` 四向，共 16 次触发）。走的是 `BehaviorService.trigger` 通路，与轨迹通路是两条不同的路——轨迹管「怎么动」，这个管「做什么动作」 |
| `gen_ladder.py` + `play_ladder.py` | **幅度梯度**：按序播若干档递增幅度，一次定量级。「该多大」是量级判断，二选一每轮只给 1 bit，梯度（心理物理学极限法）快得多，故**不盲**。每档实测 clamp 率、命中即剔除——不给一个"看着更大其实已削平"的选项 |

## 自采数据框架（2026-08-07 新增）

用户拍板后续**自采待机运动学数据**（现有两条路已实测走不通：训好的 `motion_decoder`
无时间结构；议会二期 VAR 比现状还差）。协议与全流程见 **[CAPTURE.md](CAPTURE.md)**。

| 脚本 | 作用 |
| --- | --- |
| `capture_schema.py` | 规范格式（解剖约定一次钉死）+ 质量核查 + provenance 边车 |
| `capture_ingest.py` | 各源 → 规范格式。BVH 自动判轴 / 通用 CSV（`axis_signs` **必填**，不给默认值）/ VTS 跟踪 |
| `capture_vts.py` | 用 VTS 摄像头跟踪录自己的头动——**不用等设备，且天然无轴映射问题** |
| `capture_calibrate.py` | 采集 → **带判据的常数提案**（不自动改 `src/`） |
| `capture_selftest.py` | **正控**：合成已知常数 → 走管线 → 能否量回。⚠ **采数据前先跑这个** |

🛑 采集纪律两条最容易犯的：**待机与说话必须分开采**（RAVDESS 就栽在 80% 是言语驱动头动）；
**`axis_signs` 要录一段单向转头实测确认**，别读文档猜（本仓在轴映射上错过三次）。

## 盲测验收（议会指定的主观锚点）

```powershell
# 1) 生成两版载荷；答案单独落 ab_key.json —— 播放脚本不读它
cd d:\Zero\tools\motion
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output python gen_ab.py --list
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output `
  python gen_ab.py --variant coupling --seed 1      # 换 --seed 即换一个独立试次

# 2) 播放（在 Zero_MCP 目录下跑）
cd d:\Zero_MCP
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output `
  python d:\Zero\tools\motion\play_alt.py 3      # 甲乙交替 3 轮
  # 或 play_one.py 甲 2                          # 只看某一段，连播 2 遍
  # 或 loop_vts.py                               # 情绪巡回持续播放，Ctrl+C 停
```

**流程纪律**：用户盲看 → 给完观感 → 才公布 `ab_key.json`。
先说答案会污染判断。`check_ab_delta.py` 是事后核对用的，读答案正当。

⚠ 数学席提醒：单次 2AFC 只是 n=1 伯努利观测。要多轮 + 二项检验（如 ≥9/10 为通过线），
**并须单独一组盲测问「角色是在说话还是安静待机」**——否则「更自然」可能恰因更像说话而胜出，
反把言语域污染遮盖过去。

**独立试次怎么造**：`--seed` 换一个值即换一整套（噪声场、姿态目标序列、甲乙先后全变）。
连播同一份载荷 N 遍**不是** N 个试次，只是重复曝光同一个样本。

⚠ **先做功效核算再决定值不值得跑**。本轮实例：yaw-roll 耦合改动在 10 秒片段上，
两版的片段级相关分布**重叠**（旧四分位 −0.768~−0.327 / 新 −0.588~−0.013，40 种子实测），
片段太短装不下足够的姿态周期。但同种子**配对次序**稳定（40/40 旧版更负），
所以盲测要用**同种子的甲乙配对**来问，别问"这段的侧倾对不对"。

## 驱动链路（当前绕过对方 MCP server）

Zero 侧 `motion_synth.generate_dual()` → 关键帧 JSON → Zero_MCP 侧
`BehaviorService.animate(TrajectoryRequest)` 直驱 VTS。

**为什么绕过**：对方 MCP server 的 `call_tool` 从 stdio 客户端调用会挂起——
已做对照实验：`initialize`/`list_tools` 正常，但连不碰 VTS 的静态词表 `behavior_list` 也超时，
server 端收到 `CallToolRequest` 后无下文，其日志走 stderr 未污染 JSON-RPC ⇒ 传输层问题。
这条待投给 Zero_MCP。

## VTS 授权

首次连接时 VTube Studio 会弹插件授权窗（插件名 `Zero-MCP Expression Bridge`），
**弹窗只在请求挂起期间显示**——脚本超时退出后弹窗即消失，所以要在脚本等待时去点「允许」。
token 存 `data/steering/motion/vts_token`。
