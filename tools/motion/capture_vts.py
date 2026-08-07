"""用 VTube Studio 的**摄像头跟踪**录你自己的头部运动，落成规范采集。

## 为什么这条路值得优先

- **不用等设备**：VTS 已经在跟踪你的脸，`InputParameterListRequest` 回的就是当前跟踪值。
- **没有轴映射问题**：数据直接落在皮套参数空间，`FaceAngleX/Y/Z` 的符号约定由 Live2D
  官方文档明写（见 `capture_ingest.from_vts_parameters` 的现场核验注释）——
  而这一程在 BVH 侧的轴映射上错了三次。
- 设备接上后同一套管线照吃：换个适配器即可（`capture_ingest.from_csv`）。

## 采集纪律（细则见 CAPTURE.md）

🛑 **录待机就别说话**：说话时约 80% 的头动是言语驱动的，混进待机集会把整套常数带偏
（RAVDESS 那批数据就是栽在这）。想要说话分支的数据，单独录、标 `--scene speaking`。

⚠ 本脚本按**真实时间戳**记录，不假设等间隔——轮询式采集的间隔一定会抖，
写死 fps 再差分会把抖动算成运动。

在 `d:\\Zero_MCP` 目录下跑（两仓不能同时进 sys.path）：

    python d:\\Zero\\tools\\motion\\capture_vts.py --subject s01 --seconds 300
    python d:\\Zero\\tools\\motion\\capture_vts.py --subject s01 --scene speaking --seconds 300
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

import _paths as P

os.environ.setdefault("VTS_BEHAVIOR_ENABLED", "true")
os.environ.setdefault("VTS_TOKEN_FILE", str(P.VTS_TOKEN))

TRACKED = ("FaceAngleX", "FaceAngleY", "FaceAngleZ")
DEFAULT_HZ = 30.0


async def record(subject: str, seconds: float, scene: str, session: str, hz: float, notes: str):
    P.use_zero_mcp()
    from src.mcp.behavior.service import BehaviorService

    sys.path.insert(0, str(P.REPO / "tools" / "motion"))
    import numpy as np
    from capture_ingest import from_vts_parameters

    service = BehaviorService()
    status = await service.connect()
    print(f"连接 VTS：healthy={status.healthy}", flush=True)
    api = service.sink.api  # type: ignore[union-attr]

    for i in range(5, 0, -1):
        print(f"  {i} 秒后开始录 {seconds:.0f} 秒（scene={scene}）…", flush=True)
        await asyncio.sleep(1.0)
    if scene == "idle":
        print("  🛑 待机采集：请**不要说话**，自然放松地坐着即可", flush=True)

    stamps: list[float] = []
    values: dict[str, list[float]] = {k: [] for k in TRACKED}
    start = time.perf_counter()
    interval = 1.0 / hz
    next_tick = start
    while True:
        now = time.perf_counter()
        if now - start >= seconds:
            break
        data = await api.request("InputParameterListRequest")
        listed = [*data.get("defaultParameters", []), *data.get("customParameters", [])]
        table = {p["name"]: p for p in listed}
        missing = [k for k in TRACKED if k not in table]
        if missing:
            raise RuntimeError(f"VTS 未回这些参数：{missing}——确认摄像头跟踪已开启")
        # ⚠ 记**采样那一刻**的真实时间，不是理论刻度
        stamps.append(time.perf_counter() - start)
        for key in TRACKED:
            values[key].append(float(table[key].get("value", 0.0)))
        elapsed = time.perf_counter() - start
        if len(stamps) % int(hz * 10) == 0:
            print(f"    已录 {elapsed:.0f}/{seconds:.0f}s（{len(stamps)} 帧）", flush=True)
        next_tick += interval
        await asyncio.sleep(max(0.0, next_tick - time.perf_counter()))

    await service.disconnect()

    capture = from_vts_parameters(
        np.array(stamps),
        np.array(values["FaceAngleX"]),
        np.array(values["FaceAngleY"]),
        np.array(values["FaceAngleZ"]),
        subject=subject,
        scene=scene,
        session=session,
        notes=notes,
    )
    out = P.OUT / "captures" / f"{subject}-{scene}-{session}.npz"
    provenance = capture.to_npz(out)
    quality = provenance["quality"]
    print(f"\n落盘 → {out}")
    print(f"  {quality['frames']} 帧 · {quality['duration_s']}s · 实际 {quality['median_fps']}Hz")
    if quality["ok"]:
        print("  ✅ 质量核查通过")
    else:
        print("  ⚠ 质量问题（**不自动修**，由你决定重录还是接受）：")
        for issue in quality["issues"]:
            print(f"     · {issue}")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--subject", required=True, help="受试者标识（按人分组留出的依据，必填）")
parser.add_argument("--seconds", type=float, default=300.0, help="时长；下限 200s")
parser.add_argument("--scene", default="idle", choices=("idle", "speaking"))
parser.add_argument("--session", default="1", help="同一人的第几次采集")
parser.add_argument("--hz", type=float, default=DEFAULT_HZ, help="目标轮询率（实际以时间戳为准）")
parser.add_argument("--notes", default="", help="设备/坐姿/环境等备注")
args = parser.parse_args()

asyncio.run(record(args.subject, args.seconds, args.scene, args.session, args.hz, args.notes))
