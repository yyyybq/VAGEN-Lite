# 模型线状态（唯一的状态文档）

最后更新：2026-10-24 22:40 UTC。

模型线：把 Qwen3-VL-8B-Instruct 和 Cambrian-S（7B 为主）接进统一框架，在主线最高的 setting 上跑主实验和下游实验。报告在 [reports/](reports/)，实验登记表 `MATRIX.md` 第三步建。

## 1. 现在在哪一步

结论：第二步（接入与冒烟）进行中。两个模型的代码（含两个 StarVLA 骨干）都已接上并通过 CPU 核对；两个模型的零样本评测和 1 卡加载都已完成；SFT、RL 和 8 卡下游冒烟都在等空闲机器（Qwen3-VL 的 SFT 从 20:30 起等，两个池一直有主线任务在排队）。

- 第一步（盘点与核实）已完成，你 10-24 确认。报告：[reports/STEP1_INVENTORY_20261024.md](reports/STEP1_INVENTORY_20261024.md)。
- 第二步各块（顺序按你的第 5 节）：

| 块 | 代码 | CPU 核对 | GPU 冒烟 |
| --- | --- | --- | --- |
| Qwen3-VL 接入 | 完成 | 完成：LLaMA-Factory 编码和 Qwen3-VL 自己的模板逐个 token 相同（2 条对话）；只监督回答；每帧 64 个图像 token 和处理器一致；位置编码函数和 Qwen2-VL 对图像相同，不用改 | 零样本完成；SFT 冒烟在等空闲 8 卡机器 |
| Qwen3-VL 的 StarVLA 骨干 | 完成（StarVLA 自带 Qwen3-VL 接口） | — | 1 卡加载通过；8 卡最小方案等 Qwen3-VL SFT 冒烟提交后自动提交 |
| Cambrian-S 代码 | 完成（RL 拼接路径、评测分支、SFT 工具） | 完成：LLaMA-Factory 编码和 RL 训练端展开后的序列逐个 token 相同（2 条对话，2,277 和 5,008 个 token），每帧 756 个位置，像素值和 RL 处理器的相同 | 加载测试通过（HF 和 vLLM 贪心回答相同，逐词对数概率差最大 0.013）；零样本完成；SFT/RL 冒烟链等 Qwen3-VL SFT 冒烟提交后自动启动 |
| Cambrian-S 的 StarVLA 骨干 | 完成（新接口，运行时登记） | 完成：分词器和 Qwen2.5-VL-7B 的完全相同、隐藏维 3584，动作头和动作 token 原样沿用；多相机按 Cambrian 的方式（每个视角 72 个位置）；批量前向和逐条前向相同 | 1 卡加载通过；8 卡最小方案等 Cambrian SFT 冒烟提交后自动提交 |

- 自动接力（`models_runs/watchers/`，每个只在前一个提交后才动；启动训练链前先把计划的提交号设成工作树当时的 HEAD，工作树有未提交改动就不启动）：Qwen3-VL SFT 冒烟提交 → Cambrian 冒烟链启动、DS-Q3B-smoke 提交；Cambrian SFT 冒烟提交 → DS-CSB-smoke 提交；Qwen3-VL RL 冒烟提交 → Qwen2.5-VL 参照冒烟链启动。
- 报告写到 `reports/STEP2_SMOKE_<日期>.md`（冒烟做完再写）。

## 1a. 你的决定（10-24，第一步报告之后）

1. **Cambrian-S 权重**：同意在登录节点经 hf-mirror 下载 Cambrian-S-7B（32.13 GB）和 SigLIP2-so400m-patch14-384（4.58 GB），放到共享模型目录下本线的子目录（`/mnt/umm/shared_model/vagen-lite-models/hub`，HF 缓存的目录结构）；下载后核对 sha256（来源没有就核对大小和分片清单），记录来源地址、修订号、时间。只下这两样，再要下别的先问。
2. **Qwen3-VL 的 SFT**：在现有的 LLaMA-Factory（`ca75f1e`）里用本线新文件登记 Qwen3-VL 的对话模板和多模态插件，不装新包，在 `.venv-qwen3vl-runtime` 里跑。
3. **Qwen3-VL 的 RL 和评测**用 `.venv-qwen3vl-runtime`；主环境一个包都不改。
4. **Cambrian-S 全做**：SFT、两个 RL 臂、评测、下游四个起点。
5. **StarVLA 适配 Cambrian-S**（新文件加开关，用 Cambrian-S 自己的模型类和处理器；估代价不超过 4 天就直接做）。
6. **KL 停止线用主线标准的 0.2**，和 C0、V4 一致（简报里的 1.0 是 A 组的规则，不用）。
7. **V4 的 keep 门槛沿用 0.3318、下限 1.0**。另外用新模型的 SFT 模型回放全失败题，算出它自己的前 20% 门槛，写进报告但不用；逐次记录补进来的题数，到第 50 次平均每次不到 1 道就停下来报告。
8. **Qwen3-VL 权重的 sha256** 放进第二步第一个 sco 任务核对。
9. **之前定的继续有效**：每个模型 RL 两臂（S0 对照、密集为主）；密集变体在提交 RL 时按主线最新汇总取最高（现在是 V4）；SFT 用新格式；下游四个起点都做，原始模型的起点骨干一通就提；和下游同级；Qwen2.5-VL 新格式参照行排最后。
10. **Cambrian-S 每帧 756 个 token 的开销先量再定**：主实验按估算做；下游若一个起点的训练时间超过 Qwen2.5-VL 同样任务（约 42 小时）的 3 倍，停下来报告再提。

10-24 22:45 第二步阶段汇报之后的决定（都选甲）：

11. **StarVLA 环境缺的两个包**（`ezcolorlog`、`open_clip`）：用替身文件，只由 Cambrian 入口加载，不往下游线共用的环境里装东西。
12. **下游多相机**：按 Cambrian 处理多图的方式，每个视角 72 个位置（估每步约为 Qwen2.5-VL 的 1.2 倍）；不用每视角 756 个位置的做法（估约 7 倍）。
13. **补 Qwen2.5-VL-7B 零样本对照行**：同样 705 道、温度 0.8（2 个 1 卡任务，约 0.8 卡·小时），结果名 `m_qwen25vl7b_base_t08`。

执行记录：

- dense-batch2 的 6 个提交按顺序拣进 `models`（10-24 19:41），没有冲突：`aa6451e8→7fe42c0d`、`ec7682ec→f7bce387`、`991cc92d→db55eba1`、`f8ba48c5→cd812b25`、`24f1272e→21294b24`、`418927c4→0fbc32b9`。
- worktree 的 `verl` 子模块从主 checkout 的 verl 本地克隆，检出 `a0ae85c`（没联网）。
- 决定 8：Qwen3-VL 的 4 个分片 sha256 和文件名（blob 名）全部一致（root 任务 `pt-v2kaero2`）。
- 决定 1：下载和核对完成，见第 3 节和第 3a 节。
- GPU 规则（你 10-24 的话："两个池都可用，只用空闲机器"）的执行方式：本线任务名都以 `models-` 开头；8 卡任务等到某个池没有主线任务在排队、且有空机器才提交；1 卡任务等到没有主线 1 卡任务（渲染服务、评测）在排队、且有机器有空卡才提交；暂停文件在时都不提交（`train_chain` 计划 `"pool": "idle"`，`model_matrix/sco_view.py`）。

## 2. 谁在跑

**训练**

| 编号 | 内容 | 状态 | sco 任务 | 更新（UTC） | 卡在哪 |
| --- | --- | --- | --- | --- | --- |
| Q3-SMOKE-SFT | Qwen3-VL-8B SFT 冒烟：新格式 440 条对话（训练 400、验证 40），1 个 epoch | 等卡 | 未提交 | 10-24 22:30 | 两个池都有主线任务在排队（22:29：h800 9 个、zoetrope 3 个），按规则让路；20:30 起 |
| Q3-SMOKE-RL | 从上面的 SFT 跑 RL 10 次更新（V4，带对数概率核对） | 未开始 | — | 10-24 22:30 | 等 SFT 和它的评测 |
| Q25-SMOKE | Qwen2.5-VL-7B 同样的冒烟（对数概率核对的参照） | 未启动 | — | 10-24 22:30 | 接力：Qwen3-VL 的 RL 冒烟提交后自动启动 |
| C-SMOKE | Cambrian-S-7B 同样的冒烟（`m_cambrian7b_smoke_20261024`） | 未启动 | — | 10-24 22:30 | 接力：Qwen3-VL 的 SFT 冒烟提交后自动启动 |
| DS-Q3B-smoke | 下游最小方案：原始 Qwen3-VL-8B 当 StarVLA 骨干，SFT 100 步 + 3 个任务 × 4 个种子闭环 | 未提交 | — | 10-24 22:30 | 接力：Qwen3-VL 的 SFT 冒烟提交后自动提交 |
| DS-CSB-smoke | 同上，原始 Cambrian-S-7B | 未提交 | — | 10-24 22:30 | 实验目录已建；接力：Cambrian 的 SFT 冒烟提交后自动提交 |
| M-qwen3vl8b-sft / -s0 / -d | Qwen3-VL-8B 正式 SFT、RL 两臂 | 未提交 | — | — | 冒烟通过后发任务清单给你确认 |
| M-cambrians7b-sft / -s0 / -d | Cambrian-S-7B 正式 SFT、RL 两臂 | 未提交 | — | — | 同上 |
| M-qwen25vl7b-sft / -s0 / -d | Qwen2.5-VL-7B 参照行（新格式重跑） | 未提交 | — | — | 排在两个新模型之后 |
| DS-Q3B-100-s1 | 下游：原始 Qwen3-VL-8B | 未提交 | — | — | 等 DS-Q3B-smoke 通过，随正式清单一起确认 |
| DS-CSB-100-s1 | 下游：原始 Cambrian-S-7B | 未提交 | — | — | 等 DS-CSB-smoke 通过，随正式清单一起确认 |
| DS-Q3S / Q3RS / Q3RD、DS-CSS / CSRS / CSRD | 下游：两个模型的 SFT、RL 两臂起点 | 未开始 | — | — | 等第三步的检查点 |

**评测**

| 编号 | 内容 | 状态 | sco 任务 | 更新（UTC） | 结果 |
| --- | --- | --- | --- | --- | --- |
| STEP1-LOAD | 1 卡加载测试（Qwen3-VL、Cambrian，HF 和 vLLM） | 已完成 | `pt-kqhed0xd`、`pt-gm3kn6l8` | 10-24 18:35 | 见第一步报告 |
| Q3-SHA256 | Qwen3-VL 分片 sha256 | 已完成 | `pt-v2kaero2` | 10-24 20:28 | 4 个全部一致 |
| Q3B-LOAD | Qwen3-VL-8B 当 StarVLA 骨干的 1 卡加载 | 已完成 | `pt-xh7kmkf2` | 10-24 20:50 | 750/750 个张量相同；LM 隐藏维 4096；前向损失 0.721、峰值 22.96 GB；反向放得下，峰值 35.74 GB；predict_action 输出 [1,16,12] 有限 |
| M-qwen3vl8b-base | Qwen3-VL-8B 零样本，705 道（温度 0.8，top-p 0.92，新格式示例） | 已完成 | `pt-4kvvmez2`（验证）、`pt-fsl9tk8p`（测试） | 10-24 20:57 | 成功率：验证 413 道 0.080，测试 292 道 0.086；不合法回答率 0.000；碰撞率 0.051 / 0.038 |
| C-LOAD-1 | Cambrian-S-7B 真权重 1 卡加载 | 已完成（部分） | `pt-b3ztrs33` | 10-24 20:54 | vLLM 能加载能生成，但图像处理器退到了替代品（下载时缓存缺 `refs/main`，已补）；HF 一级失败于同一原因 |
| C-LOAD-2 | 同上，补 `refs/main` 后重跑 | 已完成（部分） | `pt-ifsus27t` | 10-24 21:04 | vLLM：用 SigLIP2 自己的处理器，加载 155 s、显存 47 GiB，贪心回答 `<action>move_forward\|turn_left\|</action>`；HF：加载 30 s、15.7 GiB，但 Cambrian 源码的带缓存生成在这版 Transformers 上报错（训练不走这条路） |
| C-LOAD-3 | 同上，HF 一级改成不带缓存逐步解码，两级都记下逐词对数概率 | 已完成 | `pt-dmkux9k6` | 10-24 21:47 | 两级都正常；HF 和 vLLM 的贪心回答相同（13 个 token），逐词对数概率差最大 0.013、平均 0.002 |
| M-cambrians7b-base | Cambrian-S-7B 零样本，705 道（温度 0.8，top-p 0.92，新格式示例） | 已完成 | `pt-62zkati2`（验证）、`pt-vwp2bl68`（测试） | 10-24 22:15 | 成功率：验证 413 道 0.065，测试 292 道 0.079；不合法回答率 0.057 / 0.051；碰撞率 0.075 / 0.082；评测用时 1,493 s / 1,142 s |
| CSB-LOAD-1 | Cambrian-S-7B 当 StarVLA 骨干的 1 卡加载 | 失败 | `pt-7bkq0842` | 10-24 22:17 | 权重加载正确（765/765 个张量相同），批量 2 的前向报错：Cambrian 源码的层循环只取批里第一条（见第 5 节）；已改走 Transformers 自己的 Qwen2Model 前向 |
| CSB-LOAD-2 | 同上，改后重跑 | 已完成 | `pt-z2qjr3ew` | 10-24 22:29 | 765/765 个张量相同；隐藏维 3584，动作 token 146663；前向损失 0.627、1.43 s、峰值 31.2 GB；反向放得下，峰值 40.9 GB；predict_action 输出 [1,16,12] 有限 |

现在本线在 sco 上没有任务。

## 3. 权重和数据

| 内容 | 路径 | 说明 |
| --- | --- | --- |
| Qwen3-VL-8B-Instruct | `/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | 只读；分片只有 root 能读（sco 任务里读得到）；sha256 已核对 |
| Cambrian-S-7B | `/mnt/umm/shared_model/vagen-lite-models/hub/models--nyu-visionx--Cambrian-S-7B/snapshots/556e72fded09352f6aa4e9c100c9f39e6fe0fba7` | 本线下载，见 3a |
| SigLIP2-so400m-patch14-384 | 同一缓存下 `models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f` | 本线下载，见 3a；Cambrian 的视觉塔和图像处理器按仓库名从这里读 |
| Cambrian 外部源码 | `/mnt/umm/users/yinbaiqiao/cambrian-s` | 只读 |
| 8 月 RL 训过的 Cambrian 检查点 | `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/models/historical/cambrian_*` | 只在第一步测过代码，不是起点 |
| Qwen2.5-VL-7B 原始 | `E/interiorgs_v10_r11_20261016/inputs/base_7b` | 参照行起点 |
| 新格式 SFT 数据 | `/mnt/umm/users/yinbaiqiao/models_runs/sft_v10_newformat_20261024` | C0 的 11,095 条对话改成新格式（`action_names`）；除格式示例那一行外和 C0 相同（逐条核对过）；冒烟子集 `sft_trajectories_smoke.jsonl`（训练 400、验证 40，种子 20261024） |
| RL 题 | `E/interiorgs_v10_r11_20261016/rl_rows/rows.jsonl`，排除 `E/interiorgs_v10_r10_20261015/depth_check/depth_check_summary.json` | 后备题 10,798 道 |
| 评测集 | `models_runs/eval_sets/eval_val_v2`（413）、`eval_test_v2`（292） | 本线自己的结果目录；`tasks.jsonl` 和 `selection.json` 链接到主线的 `E/interiorgs_v10_r12_20261019/eval_*_v2`，主线的结果目录不写 |
| 密集变体汇总 | `E/interiorgs_v10_r12_20261019/analysis/variants/variants.json` | 提交 RL 时再读一次 |

`E` = `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial`。

### 3a. Cambrian-S 权重的来历

| 项 | Cambrian-S-7B | SigLIP2-so400m-patch14-384 |
| --- | --- | --- |
| 仓库 | `nyu-visionx/Cambrian-S-7B` | `google/siglip2-so400m-patch14-384` |
| 修订号 | `556e72fded09352f6aa4e9c100c9f39e6fe0fba7` | `e8e487298228002f3d8a82e0cd5c8ea9c567f57f` |
| 文件清单和 sha256 的来源 | huggingface.co 的 API（经登录节点代理，10-24 19:41:09） | 同左 |
| 下载来源 | hf-mirror.com（不经代理），`HF_HUB_DISABLE_XET=1` | 同左 |
| 下载时间（UTC） | 10-24 19:41:26 起，用时 267.7 s | 接着下，用时 79.6 s，19:47:13 完 |
| 核对 | 每个文件的大小和清单一致，LFS 文件的 sha256 和清单一致，没有多余文件（`verify.json` 的 all_ok 为真） | 同左 |
| 补记 | 10-24 20:58 给两个仓库写了 `refs/main`（40 字节，内容就是上面的修订号）：按提交号下载时 huggingface_hub 不写它，离线按仓库名加载会找不到快照 | 同左 |

记录文件：`/mnt/umm/users/yinbaiqiao/models_runs/cambrian_weights_20261024/{manifest,download,verify,refs}.json`；脚本 `model_matrix/weights/fetch_cambrian.py`。

Cambrian-S-7B 的配置：语言模型从 Qwen2.5-7B-Instruct 来，隐藏维 3584、28 层、4 个 KV 头、词表 152,064；SigLIP2 视觉塔 26 层，每帧 27×27 = 729 个图像特征，每行末尾加一个换行特征，共 756 个位置；没有下一帧预测头。分词器里没有 `<image>`，本线在 RL 和评测里按 8 月的做法运行时加上（编号 151665），训练端和 SFT 里用 `<|image_pad|>`（151655）标图像位置。

## 4. 实验表

| 编号 | 模型 | 起点 | 内容 | 状态 | 结果 |
| --- | --- | --- | --- | --- | --- |
| M-qwen3vl8b-base | Qwen3-VL-8B | 原始 | 零样本，705 道，温度 0.8 | 完成 | 验证 0.080，测试 0.086 |
| M-qwen3vl8b-sft | Qwen3-VL-8B | 原始 | SFT（C0 的数据来源和配方，新格式） | 未开始 | — |
| M-qwen3vl8b-s0 | Qwen3-VL-8B | 本线 SFT | RL 稀疏臂（`v10_p1_S0`） | 未开始 | — |
| M-qwen3vl8b-d | Qwen3-VL-8B | 本线 SFT | RL 密集臂（提交时按汇总选，现在是 V4） | 未开始 | — |
| M-cambrians7b-base | Cambrian-S-7B | 原始 | 零样本，705 道，温度 0.8 | 完成 | 验证 0.065，测试 0.079 |
| M-cambrians7b-sft / -s0 / -d | Cambrian-S-7B | 同 Qwen3-VL | 同 Qwen3-VL | 未开始 | — |
| M-qwen25vl7b-sft / -s0 / -d | Qwen2.5-VL-7B | 同上 | 同上（参照行） | 未开始 | — |

## 5. 代码和输出

- worktree：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-models`，本地分支 `models`，从 `origin/main` `00363333` 开出，不跟踪远端。代码整理期间不推 main，本地小步提交。
- 改动都是新文件或默认关闭的新开关；不设这些开关时，原有路径的行为不变（每个都有测试）。主要的：
  - 型号登记 `vagen/models/model_types.py`（`qwen2_5_vl` 默认、`qwen3_vl`、`cambrian_qwen`）；训练链计划字段 `model_type`，`rl-config --model-type`。
  - Qwen3-VL：`model_matrix/sft/lf_qwen3vl.py`、`lf_launcher.py`、`accelerate_fsdp_qwen3vl.yaml`、`check_lf_qwen3vl.py`。
  - Cambrian-S：`vagen/agent_loop/cambrian_concat.py`（RL 拼接路径的展开）；`vagen/models/cambrian_register.py` 的开关 `image_token_alias`；`vagen/ray_trainer.py` 的开关 `trainer.agent_loop_manager`；`data_gen/interiorgs_v10/evaluation.py` 的 Cambrian 分支；`model_matrix/sft/lf_cambrian.py`、`lf_launcher_cambrian.py`、`accelerate_fsdp_cambrian.yaml`、`check_lf_cambrian.py`。
  - 对数概率核对：`rl-config --rollout-log-probs`（vLLM 返回采样 token 的对数概率，训练器逐次写对比文件；更新本身不变）。
  - 只用空闲机器：计划 `"pool": "idle"`、`model_matrix/sco_view.py`、`model_matrix/sco_job.py`；任务名前缀计划字段 `job_prefix`。
  - 计划字段 `rl_start_model`（跳过 SFT 从给定模型开始 RL，只给冒烟查管线用；正式实验都是先 SFT 后 RL）。
  - 下游：`model_matrix/downstream/`（只读调用下游线的配方和入口，不改下游线的文件）。Cambrian-S 的 StarVLA 骨干：`starvla_cambrian.py`（接口，运行时登记到 StarVLA 的 `get_vlm_model`，其他起点照旧用 StarVLA 自己的接口）、`cambrian_entry.py`（训练入口、加载测试和策略服务都经它启动）、`cambrian_env.sh`。
  - StarVLA 环境缺 Cambrian 源码在导入时要的两个包（`ezcolorlog` 彩色日志、`open_clip` 只给 CLIP 视觉塔用）。没装任何东西：`model_matrix/downstream/shims/` 放了两个替身文件，只有 `cambrian_entry.py` 把它们放到搜索路径的最后（装了真包就用真包）；SigLIP2 这条路不用它们。
  - Cambrian 源码的语言模型循环取 `layer_outputs[0]`，在 Transformers 4.54 以后（解码层直接返回隐藏状态）取到的是批里的第一条。批量为 1 时没问题（RL 去掉填充后总是 1，SFT 每卡每步 1 条，加载测试 1 条）；批量大于 1 就错。StarVLA 接口改走 Transformers 自己的 `Qwen2Model.forward`（同一组层和位置编码），测试里核对过：单条和 Cambrian 循环相同，左填充的 2 条批和逐条相同（Transformers 4.56.1 和 4.57.0 都测）。SFT 的模型类遇到批量大于 1 直接报错。
- 测试：`data_gen/interiorgs_v10/tests` + `model_matrix/tests`，10-24 21:38 全跑 325 通过、3 跳过；之后 `model_matrix/tests` 22:31 45 通过、3 跳过（跳过的 3 个只在 Qwen3-VL 运行环境里跑，在那里通过）。
- 大文件和任务输出：`/mnt/umm/users/yinbaiqiao/models_runs/`。
- 整理窗口问到下面这些目录，一律回答"不能动"：
  - `VAGEN-Lite-models`、`models_runs/`、`cambrian-s/`；
  - `VAGEN-Lite/.venv-qwen3vl-runtime`；
  - `VAGEN-Lite/models/historical/cambrian_*`；
  - 共享目录里的 Qwen3-VL-8B；`/mnt/umm/shared_model/vagen-lite-models/`。

## 6. 规则

- GPU：bq_tp_mmxbf16 和 bq-ready-* 最高，不动、不问；其次主线的密集奖励实验和 A 组；本线和下游同级，谁先排队谁先上，冲突时记在这里并告诉你；其他扫描最后。本线只用空闲机器（见 1a 执行记录）。
- 暂停文件 `/mnt/umm/users/yinbaiqiao/downstream_runs/PAUSE_SUBMISSIONS` 存在时本线不提交新任务。
- 从集群外拷东西、下载大文件、安装新环境、调用付费接口、删除文件、转换或导出权重产生新副本，都先问你。
