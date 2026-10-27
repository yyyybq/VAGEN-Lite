# 模型线可腾空间清单（2026-10-27，交磁盘线）

按你 10-27 的第 26 条（甲）写，交给磁盘线。本线自己一个文件都不删；删不删、怎么归档，按磁盘线的规矩（先归档、核对引用、逐批经你同意）。时间 UTC；大小用十进制 GB（`du --block-size=1M` 的 MiB × 1.048576 / 1000），18:30 量的。`R` = `/mnt/umm/users/yinbaiqiao/models_runs`。

## 结论

- 能腾的一共约 **1.24 TB**，分五类，都是已经跑完、结果已记下、没有任何在跑或排队的任务读的东西。
- 其中约 **0.63 TB** 是本线自己的（Cambrian 和 Qwen2.5-VL 参照行），本线这边同意即可。
- 另外约 **0.60 TB** 是 Qwen3-VL 的。按第 24 条，Qwen3-VL 的资产已交主线窗口（"主实验"）接手，删这部分还要主线点头。

| 类 | 内容 | 大小 | 归谁 |
| --- | --- | --- | --- |
| 1 | Qwen3-VL 正式 SFT 没选中的 9 个 epoch | 306.1 GB | 主线（Qwen3） |
| 2 | Cambrian 正式 SFT 没选中的 9 个 epoch | 282.1 GB | 本线 |
| 3 | 第二步冒烟的三个 SFT 模型 | 194.6 GB（Qwen3 68.0；Qwen2.5 63.8；Cambrian 62.7） | Qwen3 部分归主线，其余本线 |
| 4 | 第一个种子 4 个 RL 臂最后一个检查点里的续训状态（权重留着） | 403.2 GB（Qwen3 210.4；Cambrian 192.7） | Qwen3 部分归主线，其余本线 |
| 5 | 第二步下游 8 卡最小方案的三个实验目录 | 50.5 GB（Qwen3 17.9；Cambrian 32.6） | Qwen3 部分归主线，其余本线 |
| | 合计 | **约 1,236 GB** | 本线 633.9 GB，主线 602.4 GB（分类小计四舍五入，相加差 0.2 GB） |

## 1. Qwen3-VL 正式 SFT 没选中的 epoch（306.1 GB）

`R/chains/m_qwen3vl8b_formal_sft_20261026/lf_q3formal/model/` 下：

| 路径 | 大小 |
| --- | --- |
| `checkpoint-154`、`-308`、`-462`、`-616`、`-770`、`-924`、`-1078`、`-1386`（第 1–7、9 个 epoch） | 各 34.0 GB，共 272.1 GB |
| `checkpoint-1540`（第 10 个 epoch） | 34.0 GB |

- 为什么用不上：
  - 按验证集选的是第 8 个 epoch，即 `checkpoint-1232`。它在用：种子 20261021 的 RL 两臂和 DS-Q3S 的起点。
  - 其余 epoch 的验证、测试、验收评测都已做完，结果在共用评测集里（`sft_q3formal_e1` … `e10`、`*_greedy`），没有计划或起点登记读它们。
- `checkpoint-1540` 和同目录顶层的 `model-0000*-of-00007.safetensors` 是同一个模型：LLaMA-Factory 最后会把模型再存一遍。
  - 7 个分片大小都相同，第 7 个分片逐字节相同；其余分片没有逐字节比。
  - 删 `checkpoint-1540` 后，第 10 个 epoch 还有顶层这一份。
- 注意：这些检查点按 fp32 存，`config.json` 却写 bf16。

## 2. Cambrian 正式 SFT 没选中的 epoch（282.1 GB）

`R/chains/m_cambrian7b_formal_sft_20261026/lf_cformal/model/` 下：

| 路径 | 大小 |
| --- | --- |
| `checkpoint-154`、`-308`、`-462`、`-616`、`-924`、`-1078`、`-1232`、`-1386`（第 1–4、6–9 个 epoch） | 各 31.3 GB，共 250.7 GB |
| `checkpoint-1540`（第 10 个 epoch；和顶层 `model-*.safetensors` 7 个分片大小都相同） | 31.3 GB |

- 为什么用不上：
  - 选的是第 5 个 epoch，即 `checkpoint-770`。它在用：Cambrian 密集臂重跑 `crd` 和 DS-CSS 的起点。
  - 其余 epoch 的评测和验收都已做完（`sft_cformal_e1` … `e10`）。

## 3. 第二步冒烟的 SFT 模型（194.6 GB）

| 路径 | 大小 | 归谁 |
| --- | --- | --- |
| `R/chains/m_qwen3vl8b_smoke_20261024/lf_q3smoke/` | 68.0 GB | 主线 |
| `R/chains/m_qwen25vl7b_smoke_20261024/lf_q25smoke/` | 63.8 GB | 本线 |
| `R/chains/m_cambrian7b_smoke_20261024/lf_csmoke/` | 62.7 GB | 本线 |

- 内容：每个是 400 条对话、1 个 epoch 的冒烟 SFT，`model/` 顶层和 `checkpoint-7` 各一份同一个模型，各占一半。
- 读过它们的都已结束：第二步冒烟、Qwen3-VL 的 SDPA 对照、第 19 条的两个诊断。结果记在 `docs/models/reports/STEP2_SMOKE_20261025.md` 和 STATUS 第 1c、1d 节。
- 这几条链的 RL 检查点已在第 21 条删过。链目录里的评测结果、逐 token 记录很小，不在清单里，留着。

## 4. 第一个种子 RL 臂的续训状态（403.2 GB；权重留着）

下面每个检查点只删续训用的部分：
- `actor/` 下的 `model_world_size_8_rank_*.pt`、`optim_world_size_8_rank_*.pt`、`extra_state_world_size_8_rank_*.pt`；
- 检查点目录下的 `data.pt`（数据读取位置）。

`actor/huggingface/`（权重）和元数据（`checkpoint_metadata.json`、`COMPLETE`、`resolved_config.yaml`、`fsdp_config.json`）留着。

| 路径（`R/chains/` 下） | 续训部分 | 权重 `actor/huggingface/`（留） | 为什么不再续训 |
| --- | --- | --- | --- |
| `m_qwen3vl8b_formal_rl_20261026/rl_run_q3s0/checkpoints/global_step_150` | 105.2 GB | 35.1 GB | 第 150 次被读不出停止线停下（走样），不续训 |
| `m_qwen3vl8b_formal_rl_20261026/rl_run_q3d/checkpoints/global_step_300` | 105.2 GB | 35.1 GB，是 DS-Q3RD 的起点 | 训满 300 次，已结束 |
| `m_cambrian7b_formal_rl_20261026/rl_run_cs0/checkpoints/global_step_100` | 96.4 GB | 32.1 GB，是 DS-CSRS 的起点 | 第 122 次被读不出停止线停下，不续训 |
| `m_cambrian7b_formal_rl_20261026/rl_run_cd/checkpoints/global_step_50` | 96.4 GB | 32.1 GB | 第 50 次被补题规则停下，已由 `crd` 重跑代替 |

- 这几个臂中间的检查点（第 50、100 次等）本来就只留权重，没有可腾的。
- 下游起点锁只锁 `actor/huggingface/` 里的文件，删同级的续训文件不影响下游。

## 5. 第二步下游 8 卡最小方案（50.5 GB）

| 路径（`R/downstream/experiments/` 下） | 大小 | 归谁 |
| --- | --- | --- |
| `DS-Q3B-smoke-s1` | 17.9 GB | 主线 |
| `DS-CSB-smoke-s1` | 16.3 GB | 本线 |
| `DS-CSB-smoke-vision-s1` | 16.3 GB | 本线 |

- 内容：都是 100 步的冒烟，大头是 `run/final_model/pytorch_model.pt`。
- 每步时间、显存、闭环结果都已记在第二步报告和 STATUS（DS-CSB-smoke-vision 行），没有起点登记或评测读它们。
- `DS-Q3B-100-s1_lockfailed_20261026` 只有约 1 MB（10-26 起点锁失败时留下、改了名的半成品），可删可留。

## 不能动的（在用或要留）

- 正在训的链：`m_qwen3vl8b_formal_rl_20261027_s20261021`、`m_cambrian7b_formal_rl_20261027_dense`。
- 正在训或排队的 7 个下游目录：`DS-{Q3B,CSB,Q3S,Q3RD,CSS,Q3RS,CSRS}-100-s1`。其中的全状态是续训和 15 万、17.5 万步复制要用的。
- 起点：
  - Qwen3 SFT `checkpoint-1232`；
  - Cambrian SFT `checkpoint-770`；
  - `q3d` 第 300 次、`q3s0` 第 100 次、`cs0` 第 100 次的 `actor/huggingface/`；
  - `crd` 跑完后验证集最高的检查点（DS-CSRD 的起点，还没出来）。
- `R/sft_v10_newformat_20261024`（SFT 数据，0.08 GB）和 `R/eval_sets`（4.4 GB，共用评测集和结果）。
- 公开下载的模型（例如 `/mnt/umm/shared_model/vagen-lite-models/` 下的 Cambrian-S-7B）：按磁盘线的规矩永不删。
