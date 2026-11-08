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
| **EGL_GATE** | EGL 设备平台在作业里能建 NVIDIA 的 GL 上下文并正确读回像素 | MuJoCo（`MUJOCO_GL=egl`）、Habitat-sim、Open3D 无头渲染这类基准可以重新考虑 |

两个闸门不互相依赖：Vulkan 过、EGL 不过 = TVR 照常推进，Habitat 一类继续关着。闸门不比较任何模型或起点。

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

## 4. 不过时怎么办（决策树；`verdict.json` 的 `next` 字段写同样的话）

所有重投都要先问用户：TVR 的预留只有 0.5 卡时，一次 smoke 就用完。

1. **TVR 过、EGL 过**：TVR 试点可以开始规划（报告 §3.1：Q3B/Q3S/E2 × {0,100} × 2 种子，估 10–20 卡时，要另批）。EGL 类基准可以重新考虑。
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
| 6 lavapipe | 240 s | 只有在 1500 s 前能跑完才跑 |
| 7 verdict | 约 5 s | |

不跑第 6 步时最坏约 1610 s，跑时最坏约 1510 s，都在 1800 s 以内。`thor_smoke.py` 每过一个阶段都原子地重写 `result.json`（启动后、每 50 步、A 段结束、1280×720 之后、每道题之后、`--nproc` 每个进程结束后），所以内层 `timeout` 杀掉进程时，已经测到的数都还在。每个 thor 步骤结束后，脚本按 `/.ai2thor/releases/<构建名>/<构建名>` 找残留的 Unity 进程，先 TERM，5 s 后还在就 KILL，写进 `logs/reap.log`。Python 都用 `-I -u` 跑，日志不缓冲。

卡时：`--est-gph 0.5` 等于 `timeout 1800`，再加上容器启动和收尾，账本上 TVR 实际卡时会略高于 0.5（推断，几分钟量级）。之后没有计划中的 TVR 作业，这点超出只是记账上的。

作业结束后：读 `$OUT/tvr/smoke_run1/out/verdict.json`，用 `vpd/sco.py status` 确认 SUCCEEDED 或 FAILED，再 `vpd/sco.py delete --run ...` 删任务记录（删之前重查一次）。

**绝不在登录节点上跑** `vulkaninfo`、`egl_probe.py`、`thor_smoke.py` 或任何 Controller。

## 6. 偏差日志

（冻结后若有改动，按"日期 UTC / 改了什么 / 为什么 / 是否在 submit 前"记在这里。）
