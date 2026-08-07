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
| `head_final.py` | **轴映射验证** + 待机三轴分布。判别式判据：比较同一通道在「低头类」vs「转头类」动作上的响应比。⚠ 别用「主导轴须等于期望轴」——真人「看手表」是复合动作 |
| `idle_constants.py` | 从 StayStill 待机数据提取全部程序化常数的同域实测值，并与当前手写值对照 |
| `plot_motion.py` | 把合成轨迹画成图（三面板：情绪直驱 / 意志调控 / idle 基线） |
| `check_ab_delta.py` | 事后核对盲测两版的**实际**数值差异（防「以为改了 A 实际改了 B」） |

## 盲测验收（议会指定的主观锚点）

```powershell
# 1) 生成两版载荷；答案单独落 ab_key.json —— 播放脚本不读它
cd d:\Zero\tools\motion
& "E:\anaconda\Scripts\conda.exe" run -n affective-expression --no-capture-output python gen_ab.py

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
