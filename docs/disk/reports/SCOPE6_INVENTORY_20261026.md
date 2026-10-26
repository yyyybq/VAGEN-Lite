# 新纳入三处的盘点：probe_spatial、SenseNova-U1、Consistency（2026-10-26）

按你 10-26 的决定 6 做，只读。所有时间是 UTC。大小是逐个文件统计的（`disk/catwalk.py`，原始结果在 `disk_runs/raw/scope6_20261026/`）。

## 结论

1. **这 3.6 TB 里，现在能腾出来的只有约 0.29 TB，而且要你批一个 root 任务。**
   - 一半是公开下载的模型和数据集，按你的决定 3 不删：约 1.81 TB。
   - 约 1.68 TB 是 probe_spatial 发布要用的：已验收评估的检查点、冻结代码包、评估产物。
   - 约 0.26 TB 是 root 属主的派生特征，这条线删不了，也没必要删。
2. **唯一明确的候选是 SenseNova-U1 7 月的一次冒烟测试检查点（292.1 GB）**。
   - 8B 模型 it2i 训练的第三次冒烟，87 个文件全是 root 属主。
   - 7 月 26 日以后没动过，没有任何代码、plan 引用这个运行目录。
   - 旁支模型的代码和文档 10-16 已按整理 D4 归档。
   - 要删得先上传 AOSS，再提交一个 1 卡的 root sco 任务去删（10-16 整理窗口、10-21 probe_spatial 都这样做过）。按你的约束，提交 sco 任务要先问你。
3. probe_spatial 10-21 已按"整个 run 没有受保护检查点就删"清过一轮（1.71 TB），所以它剩下的检查点基本都是已验收结果（canonical 804 行、timestep 316 行、C3 180 行）的依据，或是待跑评估、阶段 5 重训要用的。这部分要动只能由那边按发布清单逐项判断。

## 1. 按归属和状态分

| 类别 | 内容 | 大小 | 属主 | 处理 |
| --- | --- | ---: | --- | --- |
| 公开下载（决定 3：不删） | `InsScene-15K/data/`：Hugging Face 数据集 `lifuguan/InsScene-15K`，由 `probe_spatial/data/download_inscene15k.py` 下载 | 1,067.9 GB | 本人 | 留着 |
| 公开下载（决定 3） | `probe_spatial/robocasa365/datasets/v1.0`：RoboCasa365 数据集，下游用其中的 `target/` | 498.5 GB | 本人 | 留着 |
| 公开下载（决定 3） | `Consistency/minimax-h3/models/MiniMax-H3`：MiniMax-H3 权重，probe_spatial 提特征的脚本读它 | 172.5 GB | 本人 | 留着 |
| 公开下载（决定 3） | `.cache/huggingface/hub/` 里 probe_spatial 用的视频和视觉模型（CogVideoX、Wan、Aether、DINOv2、V-JEPA2、VideoMAE、Fast3R） | 74 GB | 本人 | 留着 |
| probe_spatial 发布要用的 | `probe_spatial/logs/inscene15k_streaming/runs/`：其中检查点 1,383 GB，其余是评估表和日志 | 1,554 GB | 大部分是 root | 留着。10-21 的保护清单 1,120 个检查点（1.18 TB）都在这里和 `artifacts/` 里；10-21 以后新增的是阶段 5 重训（`tstep_probe_minimax_bf16_v1_r1`，186.7 GB）和 CogVideoX L20 重评 |
| probe_spatial 发布要用的 | `probe_spatial/artifacts/`（评估产物，含受保护检查点） | 100.0 GB | 大部分 root | 留着 |
| probe_spatial 发布要用的 | `probe_spatial/runtime/`（产生已验收结果的冻结代码包，交接文档说不能用别的代码重跑旧结果） | 29.8 GB | 混合 | 留着 |
| root 属主的派生特征 | `InsScene-15K/` 下的 `FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1`（146.4 GB，21 万个文件里 20.98 万个是 root）、`D_SERIES_*`（49.4 GB）、`CANARY_*`（10.4 GB）等；`probe_spatial/features/`（49.9 GB，全是 root） | 约 260 GB | root | 这条线删不了。其中一部分是 AOSS 上特征的本地镜像，probe_spatial 评估时优先读本地 |
| 候选 | `SenseNova-U1/training/RUN/sensenovau1_8b_it2i_smoke3/`：7 月 26 日的冒烟测试检查点 | 292.1 GB | root（87 个文件） | 见"需要你决定的事"第 1 项 |
| 小项 | `probe_spatial/robocasa365/.uv-cache`（安装包缓存）8.6 GB；`probe_spatial_video/env`、`robocasa365/.venv` 等环境 4–6 GB；SenseNova-U1 其余（代码、日志）1.6 GB | 约 16 GB | 本人 | 暂不动 |

## 2. probe_spatial 的运行目录明细（检查点为主）

| run 组 | 大小 | 其中检查点 | 最近修改 | 说明 |
| --- | ---: | ---: | --- | --- |
| `tstep_probe_seeded_v1` | 253.5 GB | 190.6 GB | 10-25 | 阶段 5 bf16 重训和待跑的 u88 评估 |
| `tstep_probe_opt_v3` | 249.0 GB | 222.7 GB | 09-14 | timestep 已验收的 254 行 |
| `main_v2_probe_v2_vm72_vj96` | 181.8 GB | 172.4 GB | 09-29 | canonical V2 |
| `main_v2_probe_v2_wan_ab112` | 175.0 GB | 168.8 GB | 10-22 | canonical V2 |
| `main_v2_retry_probe74_*`（两组） | 155.1 GB | 151.9 GB | 10-04 | canonical V2 补跑 |
| `main_v2_probe_minimax_all_20261011_r1` | 107.9 GB | 103.2 GB | 10-16 | MiniMax 全层 |
| `tstep_probe_pack_v4_1` | 79.2 GB | 75.6 GB | 09-22 | timestep 已验收的 30 行 |
| 其余 13 组 | 352 GB | 约 290 GB | 09-21 到 10-26 | canonical V2 各模型、单卡补跑、v2 评估 |

这些组的名字都出现在 probe_spatial 的已验收表和发布清单里（`metadata/release_inventory/release_status_20261019.csv`）。

## 需要你决定的事

1. **SenseNova-U1 冒烟检查点（292.1 GB，root 属主）怎么办**。
   - 甲（推荐）：分两步。
     1. 我先以本人身份把它上传 AOSS，逐文件核对大小（root 属主的文件可读）。
     2. 再提交一个 1 卡的 root sco 任务删除，任务只执行按清单逐个删、写记录。
     - 代价：一个单卡任务几分钟。这是这条线第一次提交 sco 任务，按你的约束要你批。
   - 乙：留着。代价：占 0.29 TB。
2. **InsScene-15K 里 root 属主的本地特征镜像（约 0.26 TB）要不要请 probe_spatial 自己清**。
   - 甲（推荐）：不动。probe_spatial 评估时优先读本地镜像，删了会让它重跑变慢（它 10-21 的记录里写过）；它在做发布审核，等它发布后再说。
   - 乙：转告 probe_spatial 会话，请它像 10-21 那样，对 AOSS 上已有、内容一致的部分用 root 任务删本地副本。
