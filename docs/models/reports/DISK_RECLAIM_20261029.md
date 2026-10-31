# 模型线可腾空间清单（补充，2026-10-29 14:30 UTC）

结论（10-31 13:00 更新）：第 1d 表 8 项 327.9 GB 你批了，磁盘线 b16 在 12:56 归档 AOSS 后删除；DS-CSRD 训完后还有一份 114 GB 全状态。（10-31 05:10 的结论：）第 1b、1c 表共 307 GB 你 10-31 批了，磁盘线 b11、b12 在 04:58 归档 AOSS 后删除；第 2 表的 17.9 GB 现在也用不上了（决定 30 已做完），没删。（10-29 22:46 的结论：）第 1 表 754.5 GB 已由磁盘线 b10 删掉（先归档 AOSS，记录 `disk_runs/batches/b10_20261029/deleted.tsv`），现在剩第 1b 表 17.9 GB（10-30 新增，用不上了）和第 2 表 17.9 GB（等本线决定 30）。原来：本线下游有 15 项可以腾，共 790.2 GB（0.79 TB）。都属于磁盘线保留规则的 R3（下游全状态，训完、评完就删，除非已计划续训）和 R4（评测出结果后的 `evals_src` 权重副本）。其中 13 项共 754.5 GB 现在就用不上；另外 2 项共 35.7 GB 要等一件事（表 2）。14:35 更新：DS-Q3RS 评完，它的全状态从第 2 表移到第 1 表。15:45 更新：DS-CSRS、DS-CSRD 停训，它们的全状态加进第 2 表。21:35 更新：三个 Cambrian 下游按用户的话续训，三份全状态又在用了，从第 2 表拿掉。本线自己不删，交磁盘线按它的流程处理。

- **起因：** 10-29 14:16 前后，配额读数（`sco afs dir-quota list` 里 `/users/yinbaiqiao` 这一行，只显示整 TB）第一次读到 19 TB。之后 14:17–14:19 连读四次都是 18 TB，所以实际用量在 18.5 TB 上下。磁盘线的监视 14:11 那一遍读的是 18 TB。
- **查过的记录：** 磁盘线 b1–b9 的 `deleted.tsv` 里，本线下游只删过 3 个冒烟实验的 `final_model`（b4、b5），下面这些一项都没动过。
- **大小：** 都是 `du --apparent-size` 量的字节数，单位 GB 按 10^9 算。
- **路径：** 都在 `/mnt/umm/users/yinbaiqiao/models_runs/downstream/experiments/` 下。

## 1. 现在就用不上的（13 项，754.5 GB；已由磁盘线 b10 在 10-29 22:43 归档 AOSS 后删除）

| 路径 | 大小 | 为什么用不上 |
| --- | --- | --- |
| `DS-Q3B-100-s1/run/full_state/steps_175000` | 125.1 GB | 20 万步训完，三点都评完。全状态只用来续训，不会再续 |
| `DS-Q3S-100-s1/run/full_state/steps_175000` | 125.1 GB | 同上 |
| `DS-Q3RD-100-s1/run/full_state/steps_175000` | 125.1 GB | 同上 |
| `DS-CSB-100-s1/run/full_state/steps_175000` | 114.2 GB | 同上 |
| `DS-Q3B-100-s1/evals_src/steps_150000`、`steps_175000` | 2 × 17.9 GB | 中间点的权重副本，两个点都评完，结果在 `evals/steps_*/out` |
| `DS-CSB-100-s1/evals_src/steps_150000`、`steps_175000` | 2 × 16.3 GB | 同上 |
| `DS-Q3RS-100-s1/evals_src/steps_150000`、`steps_175000` | 2 × 17.9 GB | 同上（10-29 12:31、12:41 评完） |
| `DS-Q3S-100-s1/evals_src/steps_150000` | 17.9 GB | 同上 |
| `DS-Q3RD-100-s1/evals_src/steps_150000` | 17.9 GB | 同上 |
| `DS-Q3RS-100-s1/run/full_state/steps_175000` | 125.1 GB | 训完，20 万步任务内评测 14:30 前评完（10-29 14:35 从第 2 表移来） |

## 1b. 10-30 新增、现在就用不上的（1 项，17.9 GB；你 10-31 批了，磁盘线 b11 在 10-31 04:58 归档 AOSS 后删除，记录 `disk_runs/batches/b11_20261030/deleted.tsv`）

| 路径 | 大小 | 为什么用不上 |
| --- | --- | --- |
| `DS-Q3S-100-s1/evals_src/steps_175000` | 17.9 GB | 本线决定 30 原来要用它补跑 OpenDrawer 种子 1012；10-30 下游线按用户的决定把这个回合加进了剔除清单，不用补了（10-30 09:10 从第 2 表移来） |

## 1c. 10-30 21:55 新增、现在就用不上的（3 项，289.1 GB；你 10-31 批了，磁盘线 b12 在 10-31 04:58 归档 AOSS 后删除，75 个文件，`huggingface/` 都留着；记录 `disk_runs/batches/b12_20261030/deleted.tsv`）

三个因走样被停止线停下的 Cambrian RL 臂，最后一个检查点里的续训状态（`actor/` 下的 `model_*`、`optim_*`、`extra_state_*` 等，除 `huggingface/` 以外的文件，各约 96 GB）。这些臂不会再续训；评测只读 `huggingface/`，已经评完或要评的都不受影响。和磁盘线 b9 用的保留规则 R1 同一类（已结束 RL 运行最后一个检查点的续训状态）。路径在 `models_runs/chains/` 下：

| 路径 | 大小 | 为什么用不上 |
| --- | --- | --- |
| `m_cambrian7b_formal_rl_20261030_stab/rl_run_cskl/checkpoints/global_step_100/actor/`（除 `huggingface/`） | 96 GB | cskl 10-30 16:14 第 139 次走样停下；第 50、100 次检查点已评完 |
| `m_cambrian7b_formal_rl_20261030_stab/rl_run_cslr/checkpoints/global_step_250/actor/`（除 `huggingface/`） | 96 GB | cslr 10-30 20:46 第 270 次走样停下；第 50–250 次检查点已评完 |
| `m_cambrian7b_formal_rl_20261030_s20261021/rl_run_cs2s0/checkpoints/global_step_100/actor/`（除 `huggingface/`） | 96 GB | cs2s0 10-30 12:29 第 130 次走样停下。它的检查点评测要等同链的 cs2d 训完，评测只读 `huggingface/` |

## 2. 要等一件事的（1 项，17.9 GB；10-31 起用不上了，10:30 并入第 1d 表）

| 路径 | 大小 | 等什么 |
| --- | --- | --- |
| `DS-Q3RD-100-s1/evals_src/steps_175000` | 17.9 GB | 同上 |

## 1d. 10-31 10:30 新增、现在就用不上的（8 项，约 328 GB；配额读数 20 TB 时列出；你 10-31 批了（决定 42 甲），磁盘线 b16 在 12:56 归档 AOSS 后删除，66 个文件、327.9 GB；记录 `disk_runs/batches/b16_20261031/deleted.tsv`）

都在 `/mnt/umm/users/yinbaiqiao/models_runs/downstream/experiments/` 下，大小用 `du --apparent-size` 量。DS-CSS、DS-CSRS 训到第 20 万步，三个点都评完了，计划里没有续训。按磁盘线的保留规则，这些属于 R3（下游全状态：训完、评完就删，除非已计划续训）和 R4（评测出结果后的 `evals_src` 权重副本）。第 2 表那一项（DS-Q3RD 第 17.5 万步副本，17.9 GB）也一起交。

| 路径 | 大小 | 规则 | 为什么用不上 |
| --- | --- | --- | --- |
| `DS-CSS-100-s1/run/full_state/steps_175000` | 114.2 GB | R3 | 10-31 00:30 三点评完，不续训 |
| `DS-CSRS-100-s1/run/full_state/steps_175000` | 114.2 GB | R3 | 10-31 00:50 三点评完，不续训 |
| `DS-CSS-100-s1/evals_src/steps_175000` | 16.3 GB | R4 | 第 17.5 万步 10-30 22:10 评完 |
| `DS-CSRS-100-s1/evals_src/steps_150000` | 16.3 GB | R4 | 10-30 08:30 评完 |
| `DS-CSRS-100-s1/evals_src/steps_175000` | 16.3 GB | R4 | 10-30 22:30 评完 |
| `DS-CSRD-100-s1/evals_src/steps_150000` | 16.3 GB | R4 | 10-31 01:00 评完 |
| `DS-CSRD-100-s1/evals_src/steps_175000` | 16.3 GB | R4 | 10-31 08:45 评完 |
| `DS-Q3RD-100-s1/evals_src/steps_175000`（原第 2 表） | 17.9 GB | R4 | 决定 30 的补跑 10-30 已做完 |

- **先留着、不列进来的：**
  - `DS-CSS-100-s1/evals_src/steps_150000`（16.3 GB）：决定 39 的 JPEG 评测用过，后续可能还要用它再评（比如换一种压缩）。等第 1zn 节的后续定了再说。
  - `DS-CSRD-100-s1/run/full_state/steps_175000`（114.2 GB）：DS-CSRD 还在训，约 14:20 训完。训完、第 20 万步评完以后同样属于 R3，到时我再补进来。
  - 各实验的 `run/final_model`（第 20 万步权重）：是 MATRIX 和以后比较的依据，不列。

## 1e. 10-31 16:45 新增（1 项，114.2 GB；你 10-31 批了（决定 44 甲），17:50 转告磁盘线执行）

| 路径 | 大小 | 规则 | 为什么用不上 |
| --- | --- | --- | --- |
| `DS-CSRD-100-s1/run/full_state/steps_175000` | 114.2 GB | R3 | DS-CSRD 10-31 16:25 训完，第 20 万步在训练任务里评完，三个点齐了；计划里不续训。训练任务 `pt-x4k6rzax` 已 SUCCEEDED |

- 第 20 万步的最终权重 `run/final_model` 不列：决定 43 的 JPEG 评测 `pt-ep8euu3r` 正要读它。

## 3. 还在涨的（21:35 更新）

三个 Cambrian 下游 10-29 21:17–21:21 续训：每 2.5 万步写一份 114 GB 的全状态，写完新的才删旧的，所以每次暂时多出 114 GB；每个任务另外还留 2 份中间点副本和最终权重，各约 16 GB，三个合计约 0.15 TB。本线决定 33（Cambrian 上游四个新臂）批了的话，还要再加约 1.2 TB，所以请先处理第 1 表。
