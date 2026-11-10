# TVR 渲染 smoke：通过标准（SMOKE_CRITERIA，提交前冻结）

- 状态：**草稿，等集成者随 vpd/ 一起提交后冻结**。冻结后改阈值只能记进文末的偏差日志，并且必须在 submit 之前。
- 写于 2026-11-08（UTC），WP-TVR；2026-11-08T11:25Z 按评审改过一次（还没冻结，所以不算偏差）：T3 只看启动，T4/T5 要求跑完，T6 要求有 Vulkan 行，T7 的 time_budget 算不过，加了时间预算、中途落盘、残留进程清理和 CVD 处理。依据：`DECISIONS.md`（优先）、`plan.md` §3.7、§5、§7，surveys.json 的 tvr 一节。
- 判定程序：`vpd/tvr/verdict.py`，它读 smoke 输出目录，写 `verdict.json`：`{TVR_GATE, EGL_GATE, reasons, caveats, steps_per_s, facts, next, criteria}`。下面的阈值和 `verdict.py` 顶部的常量是同一份；`vpd/tests/test_tvr.py` 会核对两边一致。
- 简写：`OUT` = `/mnt/umm/users/yinbaiqiao/eval_external_assets/vpd`，`WT` = `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd`。

## 0. 结论先说

一个 sco 作业（1 张 H800，zoetrope，非 root，`timeout 1800`，最多 0.5 卡时，正好是 TVR 的预留额度）回答两件**互相独立**的事：

| 闸门 | 一句话 | 过了意味着 |
|---|---|---|
| **TVR_GATE** | AI2-THOR 5.0.0 CloudRendering 在作业里用拷来的 595.71.05 库走 NVIDIA Vulkan 正常出图，TVRBench 自己的环境代码能载入评测题 | 可以规划 TVR 试点（要另批准和另算卡时） |
| **EGL_GATE**（E0–E5） | EGL 设备平台在作业里能建 NVIDIA 的 GL 上下文，清屏后读回的像素正确 | 只说明 EGL 上下文、清屏和读回能用。清屏只是填色：没有编译着色器，也没有画三角形，**不能**据此说 EGL 类基准可行（2026-11-09 更正，见第 6 节） |
| **EGL_DRAW**（E6，2026-11-09 加） | 在同一个 EGL 上下文里编译、链接 GLSL 330 着色器，开深度测试画一个红三角形，读回的颜色正确 | MuJoCo（`MUJOCO_GL=egl`）、Habitat-sim、Open3D 无头渲染这类基准可以重新考虑 |

三个闸门互不依赖：Vulkan 过、EGL 不过 = TVR 照常推进，Habitat 一类继续关着；EGL_GATE 过、EGL_DRAW 没过或没跑 = Habitat 一类也继续关着。闸门不比较任何模型或起点。

## 1. 冻结阈值（verdict.py 常量；测试逐项核对）

```thresholds
DRIVER = 595.71.05
GPU_NAME = H800
STEPS = 300
STARTUP_MAX_S = 300
DISTINCT_MIN = 0.90
SUCCESS_MIN = 0.95
NPROC = 4
NPROC_STEPS = 150
EGL_RGBA = 51,102,153,255
EGL_TOL = 2
EGL_DRAW_RGB = 255,0,0
EGL_DRAW_TOL = 3
```

## 2. TVR_GATE：以下全部满足才 PASS

| 编号 | 条件 | 数据来源 |
|---|---|---|
| T0 | 登录节点暂存的输入都在：NVGL 库、glvnd、两个 ICD JSON、tvr 环境的 python 和 vulkaninfo、CloudRendering 构建、TVRBench 的 `data/tasks/eval.json` | `staging.json` |
| T1 | 作业节点内核驱动正好是 595.71.05 | `driver_version.txt`（`/proc/driver/nvidia/version`） |
| T2 | `vulkaninfo --summary` 里有一个设备：deviceName 含 H800，deviceType = `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`，deviceUUID 和 `nvidia-smi -L` 的 UUID 相同（summary 没有 UUID 时取完整 vulkaninfo 里同一 GPU 序号的 UUID） | `vulkaninfo_summary.txt`、`vulkaninfo_full.txt`、`nvidia_smi_L.txt` |
| T3 | Controller 启动 < 300 s（只看启动：启动时间测到了就算数，后面某一步报错不影响 T3） | `thor1/result.json` raw.startup_s |
| T4 | 单进程 300 步里坏帧 = 0（坏帧：没有帧、形状不是 360×640×3、或像素标准差 < 2）；300 步没跑完（中途报错或被杀）算不过 | raw.bad_frames |
| T5 | 跑满 300 步；不同帧 ≥ 90%（≥ 270）；lastActionSuccess ≥ 95%（≥ 285） | raw.steps / distinct_frames / action_success |
| T6 | Unity 的 Player.log 里有渲染器/设备行写着 H800，有渲染器/设备行提到 Vulkan（可以不是同一行；Unity 的原文格式**未核实**），且没有任何渲染器/设备行写 llvmpipe、lavapipe、SwiftShader、softpipe | `thor1/Player*.log` |
| T7 | TVRBench ThorEnv 跑 eval.json 里第一道 iTHOR 题（FloorPlan407 easy）和第一道 ProcTHOR 题：都能载入，起点图和目标图都不是坏帧，且两张图不同。因时间不够没跑（`skipped = time_budget`）算不过 | `thor1/result.json` tasks |
| T8 | 同一张卡上 4 个进程各跑 150 步，4 个都没报错、都跑满、坏帧都为 0 | `thor4/result.json` procs |

T7 的唯一例外（plan §3.7、open_q）：ProcTHOR-10k 的 `train.jsonl.gz` 来源**未核实**。如果登录节点暂存时拿不到（`procthor_status.json` 不是 ok），ProcTHOR 那道题记为 `skipped = no_procthor_data`，T7 只看 iTHOR 题；`verdict.json` 的 `caveats` 写明"procthor_not_tested"，ProcTHOR 推到 TVR 试点再测。拿到了但载入失败，T7 不过。

另外只报告、不卡闸门的数：
- 每秒步数：640×360 单进程、1280×720 单进程（50 步）、4 进程合计、lavapipe 640×360（第 6 步，可选）；
- `/dev/nvidia-modeset`、`/dev/dri`、renderD 节点在不在，`CUDA_VISIBLE_DEVICES`、`NVIDIA_DRIVER_CAPABILITIES`；
- 作业本地盘 `/tmp`、`/dev/shm` 剩余字节（给合并权重用，open_q 第 8 条）；
- Vulkan 的 driverInfo 不是 595.71.05 时记进 caveats；
- 1280×720 那 50 步出错（`hd_error`）或 Controller 关闭出错（`stop_error`）只记进 caveats，640×360 的计数照常判；
- 作业给的 `CUDA_VISIBLE_DEVICES` 是 0 以外的数字、而 `nvidia-smi -L` 只列 1 张卡时，脚本把它改成 0 再跑 ai2thor（ai2thor 按 CVD 换算 gpu_device，cuda-vulkan 映射却按容器内的 nvidia-smi 序号），原值和改动记在 `facts.json` 的 `cvd_remap`，也进 caveats。集群文档看 CVD 在作业里从 0 编号，所以多半用不上（推断）。

ProcTHOR 数据的取法和 TVRBench 的 `_load_procthor_house` 一样：有 `train.jsonl.gz` 就用它，没有再用 `train.jsonl`；git-lfs 指针文件不算数据。

## 3. EGL_GATE：以下全部满足才 PASS

| 编号 | 条件 |
|---|---|
| E0 | 驱动是 595.71.05（同 T1） |
| E1 | `eglQueryDevicesEXT` 至少 1 个设备 |
| E2 | 设备 0 上 `eglInitialize` 成功 |
| E3 | EGL_VENDOR 含 NVIDIA |
| E4 | GL_RENDERER 含 H800 |
| E5 | 256×256 pbuffer 清成 (0.2, 0.4, 0.6, 1.0) 后读回第一个像素，每个通道和 (51, 102, 153, 255) 差 ≤ 2 |

数据来源：`egl/result.json`（`egl_probe.py`，只用 ctypes）。

## 3b. EGL_DRAW：E6 满足才 PASS（2026-11-09 加，见第 6 节）

| 编号 | 条件 |
|---|---|
| E6 | 在 E5 的同一个上下文里：GLSL 330 core 的顶点和片元着色器都编译成功、程序链接成功（三份 info log 存进 `egl/result.json` 的 `draw` 和 `egl/shader_info.log`）；打开深度测试（GL_LESS），清颜色 (0.2, 0.4, 0.6) 和深度 1.0，用 VBO/VAO 画一个红三角形（NDC 顶点 (−0.6,−0.6)、(0.6,−0.6)、(0,0.6)，z = 0），glFinish 后 glReadPixels：中心像素 (128,128) 每个通道和 (255, 0, 0) 差 ≤ 3，四个角上的像素每个通道和 (51, 102, 153) 差 ≤ 2，画完和读回后 glGetError 都是 0 |

`verdict.json` 的 `EGL_DRAW` 取 `PASS`、`FAIL`，或 `NOT_RUN`（`egl/result.json` 里没有 `draw`，即 E6 之前的 `egl_probe.py` 跑的，比如 smoke_run1）。`NOT_RUN` 和 `FAIL` 一样，不支持重新考虑 EGL 类基准。EGL_GATE 本身（E0–E5）的定义不变。

## 4. 不过时怎么办（决策树；`verdict.json` 的 `next` 字段写同样的话）

所有重投都要先问用户：TVR 的预留只有 0.5 卡时，一次 smoke 就用完。（2026-11-09 用户取消了卡时和磁盘上限；重投仍要先问用户。）

1. **TVR 过、EGL 过**：TVR 试点可以开始规划（报告 §3.1：Q3B/Q3S/E2 × {0,100} × 2 种子，估 10–20 卡时，要另批）。EGL 类基准只有在 **EGL_DRAW（E6）也过**时才可以重新考虑；E6 不过或没跑，继续关着。
2. **TVR 过、EGL 不过**：TVR 照常推进；Habitat / EGL 类基准继续关着。
3. **T1 不过（驱动不是 595.71.05，脚本退出码 3）**：拷来的库只配 595.71.05，那个节点用不了。问用户：换节点重投，还是放弃。
4. **T0 不过（暂存缺件，退出码 4）**：先把 `stage_tvr_login.sh` 跑完（联网的第 4–7 步要用户点头），再问用户重投。
5. **T2 不过（没有 NVIDIA Vulkan 设备）**：
   - `/dev/nvidia-modeset` 不在：**交用户决定**——请集群管理员给 aigc 作业开 `NVIDIA_DRIVER_CAPABILITIES=graphics`（或 all），还是退到备选方案：lavapipe（第 6 步有测速）、Xvfb + llvmpipe（`ACTIVESPATIAL_USE_X11=1`，最慢）、或报告 §3.1 里的 gsplat-on-ScanNet++ 位姿复现替代。作业里的 root 也补不了设备节点（device cgroup 禁止 mknod）。
   - `/dev/nvidia-modeset` 在：多半是库的问题。看 `ldd.txt` 的 not found 和 `logs/vulkaninfo_*.err`，在登录节点（不碰 GPU）补库，再问用户重投。
6. **T2 过，T3–T8 有不过的**：看 `thor1/Player*.log` 和 `logs/`。TVRBench 作者提过 H200 上 Vulkan 提交路径会卡住、8 张卡只有 3 张能用 CloudRendering（thor_env.py 注释），H800 可能也这样（推断）。问用户：换节点再测一次，还是走第 5 条的备选。
7. 只有 T7 不过：多半是 TVRBench 代码或数据的问题，先在登录节点上查（只用 CPU，不起 Controller），再问用户。

## 5. 怎么跑（submit 和 delete 都是先问再做的事）

```bash
# 登录节点：本地暂存（拷 /usr/lib 里的 595.71.05 库、glvnd、写 ICD JSON、ldd 报告；不联网）
nice -n 19 bash $WT/vpd/tvr/stage_tvr_login.sh --local
# 联网暂存（建 conda 环境、pip、下载 836 MB 构建、克隆 TVRBench、试拿 ProcTHOR）：用户批准后才跑
VPD_TVR_NET_OK=1 nice -n 19 bash $WT/vpd/tvr/stage_tvr_login.sh --net
# 查 OUT/tvr/staging_report.txt：ABSENT 行为空、NOT_FOUND_COUNT=0 后才准备作业
$PY_EVAL $WT/vpd/sco.py prepare --run $OUT/jobs/tvr-smoke-<ts> --name vpd-tvr-smoke --est-gph 0.5 \
    -- timeout 1800 bash $WT/vpd/tvr/tvr_smoke.sh $OUT/tvr/smoke_run1/out
$PY_EVAL $WT/vpd/sco.py submit --run $OUT/jobs/tvr-smoke-<ts>     # 编排者在用户同意后执行
```

**时间预算**（每个内层上限都打满的最坏情况；外层 `timeout 1800`）：

| 步 | 上限 | 说明 |
|---|---|---|
| 0、1 | 约 10 s | 节点信息、ldd |
| 2 | 60 + 60 s | vulkaninfo summary 和完整输出 |
| 3 thor1 | 1020 s | 能装下两次各 300 s 的 Controller 启动（A 段 + B 段 ThorEnv）再加步数和题目；`thor_smoke.py --deadline-s 990`，剩不到 420 s 就不跑 B 段，记 `time_budget` |
| 4 thor4 | 360 s | 4 进程，启动错开 8 s |
| 5 egl | 90 s | |
| 6 lavapipe | 240 s | 只有在 1500 s 前能跑完才跑。只报告速度，不卡闸门。2026-11-09 起：`CUDA_VISIBLE_DEVICES=0`、`--gpu 0`，并预写 `cuda-vulkan-mapping.json = {"0": 0}`，让 ai2thor 给 Unity 传 `-force-device-index 0`；`LP_NUM_THREADS=12`；`thor_smoke.py --fail-fast` 一看到 Player.log 里的 "Caught fatal signal" 或 Unity 进程没了就退出（退出码 3），不再干等 240 s |
| 7 verdict | 约 5 s | |

不跑第 6 步时最坏约 1610 s，跑时最坏约 1510 s，都在 1800 s 以内。`thor_smoke.py` 每过一个阶段都原子地重写 `result.json`（启动后、每 50 步、A 段结束、1280×720 之后、每道题之后、`--nproc` 每个进程结束后），所以内层 `timeout` 杀掉进程时，已经测到的数都还在。每个 thor 步骤结束后，脚本按 `/.ai2thor/releases/<构建名>/<构建名>` 找残留的 Unity 进程，先 TERM，5 s 后还在就 KILL，写进 `logs/reap.log`。Python 都用 `-I -u` 跑，日志不缓冲。

卡时：`--est-gph 0.5` 等于 `timeout 1800`，再加上容器启动和收尾，账本上 TVR 实际卡时会略高于 0.5（推断，几分钟量级）。之后没有计划中的 TVR 作业，这点超出只是记账上的。

作业结束后：读 `$OUT/tvr/smoke_run1/out/verdict.json`，用 `vpd/sco.py status` 确认 SUCCEEDED 或 FAILED，再 `vpd/sco.py delete --run ...` 删任务记录（删之前重查一次）。

**绝不在登录节点上跑** `vulkaninfo`、`egl_probe.py`、`thor_smoke.py` 或任何 Controller。

## 6. 偏差日志

（冻结后若有改动，按"日期 UTC / 改了什么 / 为什么 / 是否在 submit 前"记在这里。）

- **2026-11-09 UTC / EGL 结论说过头，加 E6 和 EGL_DRAW / 在 smoke_run1 之后，下一次 TVR 作业之前**
  - 改了什么：
    - 新增 E6（GLSL 330 着色器编译和链接 + 开深度测试画红三角形 + 读回，第 3b 节），作为单独的 `EGL_DRAW`。
    - 第 0 节的表改成：只有 E6 过，才支持"MuJoCo / Habitat / Open3D 这类 EGL 基准可以重新考虑"；E5 过不再支持这句话。
    - 第 1 节加两个阈值：`EGL_DRAW_RGB = 255,0,0`，`EGL_DRAW_TOL = 3`。
    - `EGL_GATE` 的定义（E0–E5）和阈值都没动。
  - 为什么：
    - E5 只做了 glClear 和读回，没有编译着色器，也没有画任何图元，所以证明不了光栅化能用。Unity 用不用得上它不知道，Habitat 也一样。
    - 11-09 给用户的回复写了"EGL 能用 → Habitat ImageNav 可以重新考虑"，这句说过头了，要撤回，等 E6 跑过再说。
  - smoke_run1 的 `EGL_GATE PASS` 按原定义仍然成立；这次的 `EGL_DRAW` 是 `NOT_RUN`。
- **2026-11-09 UTC / 第 6 步 lavapipe 的测量作废，修了脚本 / 不涉及闸门**
  - smoke_run1 的第 6 步里，`tvr_smoke.sh` unset 了 `CUDA_VISIBLE_DEVICES`，又传了 `--no-gpu-device`，所以 ai2thor 没给 Unity 传 `-force-device-index`。
  - Unity 2020.3 于是记下 `SelectPhysicalDevice requestedDeviceIndex=-1`，接着 `Selected physical device (nil)`，然后 SIGABRT（`home_lvp/.../Player.log`）。脚本又干等了 240 s（`steps.jsonl` 里 rc = 124）。
  - 所以 lavapipe 一步都没测到。"lavapipe 会非常慢"没有依据，撤回。
  - 现在的做法见第 5 节第 6 行：`CUDA_VISIBLE_DEVICES=0`、`--gpu 0`，预写映射文件 `{"0": 0}`，`LP_NUM_THREADS=12`，加 `--fail-fast`。
  - `verdict.json` 新增 `lavapipe` 一块，只报告：跑没跑、步数、速度、出错原因、fatal 行。
  - TVR_GATE FAIL（Vulkan 出黑帧）不受影响。
- **2026-11-10 UTC / TVR 试点改用 CPU 渲染（lavapipe），不再以 TVR_GATE 为前提 / 在任何用 lavapipe 渲染示范或评测的 TVR 作业 submit 之前**
  - 改了什么：
    - TVR 试点（示范重渲染、开发集定档位、冒烟、三个起点的试点）不走 NVIDIA Vulkan，改用 Mesa lavapipe（CPU 上的 Vulkan 实现）出图。配方在 `vpd/tvr/lvp.py`：lavapipe 的 ICD，两种 LD_LIBRARY_PATH 档案（`envlib` / `envlib+orig`，用哪种由探针定，冻结进预注册），角标字体按 sha256 固定（`BADGE_FONT_SHA256` = b184b89e3c10…）。
    - TVR_GATE 的定义、阈值和 smoke_run1 的 FAIL 结论都不动。第 4 节第 5 条里说的"备选方案：lavapipe"就是这次走的路。
    - T2、T6 按定义要求 NVIDIA 设备，lavapipe 下必然不过，所以不再拿来判 TVR 试点。渲染这一块改由新的 `vpd/PREREG_TVR_PILOT.md` 的 G0 机制闸门管：示范和评测用同一套 lvp 档案和同一个字体 sha256、每张 640×360 图 220 个图像 token、自己的运动学环境和 ThorEnv 在每条保留下来的重放上一致等。那份预注册在 TVR 冒烟之前冻结。
    - 第 6 步"只报告"的 lavapipe 测速不变。
  - 为什么：
    - 11-09 smoke_run1：NVIDIA Vulkan 出黑帧（DEVICE_LOST），TVR_GATE FAIL。
    - 11-10 复查（`OUT/plan/DECISIONS.md` G2、G8；两个 vpd-tvr-vkre 作业，zoetrope 和 zoe-cf-render，内核驱动都是 595.71.05，合计 0.027 卡时），两个池结果一样：
      - 旧组合：vkWaitForFences 返回 DEVICE_LOST；
      - 新组合（系统驱动 + 加载库 1.4.357 + 绝对路径 ICD）：vkWaitForFences 返回 VK_TIMEOUT；
      - 不带 glvnd：vkCreateInstance 返回 INCOMPATIBLE_DRIVER；
      - THOR：Player.log 认到 H800 的 Vulkan 渲染器，但 41 帧全黑（RGB 标准差 ≥ 2 的 0/41）；
      - 对照组：GL 三角形和 lavapipe 都通过。
    - 所以本轮不再走 Vulkan。用户 2026-11-10 的第三轮授权（`OUT/plan/ROUND3.md` 第 3、4 节）预先批准了"用 CPU 渲染（lavapipe）往下做，在 SMOKE_CRITERIA.md 的偏差日志和新的预注册里记清楚"，第 4 节"重投先问用户"对这些 TVR 试点作业也由那份授权覆盖。
  - 时间顺序：lavapipe 测速和确定性探针 vpd-tvr-probe（EXPLORATORY，不出任何模型数字，只定并行数 E×L 和档案）在本条之前，于 2026-11-10 15:01Z 提交。它之后的 TVR 作业都在本条之后。
