# 外部迁移评测：进度记录

分支 `eval-external`，工作树 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-eval`（从 `origin/main` 的 `a59f9583` 开出）。大文件（数据、模型、克隆的 benchmark 仓库、作业输出）放在 `/mnt/umm/users/yinbaiqiao/eval_external_assets/`，不进 git。

## 当前状态

**第 2 步、第 4 步、第 5 步在跑**：两道关（我们环境复现第十轮、WalkerBench 基座对上 SI）都已通过；165 项作业在一个队列里，状态见 `runs/queue_step2.json`。外部模型在我们环境里的结果已出（见"2026-10-19 第 5 步"一节）。

## 已定的事（2026-10-18 你确认）

1. ESI-Bench 先跳过。
2. ours-SFT 是第十轮最终模型 `interiorgs_v10_train_r10_20261015/lf_formal/model/checkpoint-770`；ours-RL 是第十轮 RL 第 150 次（代号 `ours-RL-150`），加测第 50 次（`ours-RL-50`）。
3. 外部模型：SI-7B、SI-3B，加 5 个 7B 专用模型（SpaceR、Video-R1、ViLASR、VST-7B-RL、SenseNova-SI-1.1）。
4. QA 以 EASI 的 VLMEvalKit 后端为主；SAT 和 VSTI 用 lmms-eval。SAT-Real 的复现标准改成"SI 的数落在我们的 95% 区间外才查"。
5. GPU：单卡作业，放 h800 池，同时最多 4 个。

第 2 步（2026-10-18 你确认）：
6. VSI-Bench 用 VLMEvalKit 后端。
7. 加入第十一轮这一对模型：`ours-SFT-r11`（SFT 第 9 个 epoch，`checkpoint-1386`）和 `ours-RL-r11`（P-S0 第 300 次）。
8. MindCube 完整集保留。
9. VSTI 修正前标签只给 base-3B、SI-3B、SI-7B 跑。
10. 第 4、5 步穿插在第 2 步的空档里跑。

---

## 2026-10-17 第 0 步：摸底

**结论：** QA 迁移的 6 个 benchmark 和 WalkerBench 都能在现有集群上做，不需要付费 API。ESI-Bench 做不了：集群上只有 H800，H800 没有 Isaac Sim 渲染需要的 RT 核（光线追踪单元），这是停下来问你的点。可行性表在 `step0/FEASIBILITY.md`。

**做了什么：**
1. 建了工作树：`git worktree add /mnt/umm/users/yinbaiqiao/VAGEN-Lite-eval -b eval-external origin/main`。主检出 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite` 停在 `94063ca9`，比 `origin/main` 落后 3 个只改文档的提交。我从 `origin/main` 开分支，以后合并时可以快进。主检出里的文件一个没动。
2. 查了已有的东西：
   - EASI v0.2.2 在 `third_party/EASI`，子模块没检出。lmms-eval 在 `third_party/lmms-eval`（`9a33763`），里面有 `sat`、`vsibench`、`mindcube`、`viewspatial`、`mmsi_bench` 这些任务。
   - 以前的 EASI-8 结果在 `easi_results/`，是 8 月用 lmms-eval 跑的。
   - 集群上已经缓存的数据：VSI-Bench 视频 5.4 GB、ViewSpatial 16 GB、EASI 共用数据 21 GB（含 MMSI）、MindCube 原始数据。
3. 用 HF 接口查了 13 个数据集、11 个模型的大小、版本号和是否需要申请，一个都没下载。结果在 `step0/hf_inventory.json`，脚本是 `tools/hf_inventory.py`。
4. 克隆了 ESI-Bench、WalkerBench、VLM-3R（VSTI 的评测代码）和 SAT 的仓库到 `eval_external_assets/repos/`，读了它们的安装要求、数据获取方式和模型接口。
5. 提交了一个单卡 SCO 作业看计算节点（`pt-cs6gjrm7`，跑完已删）。结果：H800，驱动 595.71.05，CUDA 13.2；容器是 Ubuntu 20.04，没挂 NVIDIA 图形库。原始输出在 `step0/gpu_probe_20261017.txt`。
6. 从论文 HTML 里把 Spatial-Interactor 的表 1、2、4、5、7 原样抽了出来，网页摘要工具把列弄乱了。核对后的对照数写进了可行性表。

**结果（要点）：**
- QA：数据新下载约 6.2 GB。SAT-Real 只有 150 题，95% 区间约 ±8 分，"差 1.5 分就查"的阈值在它上面没有意义。
- VSTI-Bench 的"相机位移"标签在 2026-07-13 被作者修正过。SI 没写用的是哪一版，这一项的对照可能不成立。
- WalkerBench：全景图离线包 5.3 GB，运行时只读本地缓存，不调谷歌 API。
- ESI-Bench：重点类里有 8 类可用题不到 100 道（例如长程导航、可通行通道各 60 道），"每类 100 题以上"做不到。SI 的 30 题子集是从双方都跑完的结果里挑的，不是随机抽的。
- 你表里的 ours-SFT 路径是第八轮的模型。按 STATUS.md，第十轮的最终模型是 `interiorgs_v10_train_r10_20261015/lf_formal/model/checkpoint-770`，建议换成它。

**产物：** `eval_external/step0/FEASIBILITY.md`、`step0/hf_inventory.json`、`step0/gpu_probe_20261017.txt`、`tools/hf_inventory.py`。

**下一步：** 等你回复可行性表末尾的 5 个问题。确认后：建 `eval-external` conda 环境，下载 SAT、VSTI、SPBench、SI-7B，进第 1 步。

---

## 2026-10-18 第 1 步：环境、下载、跑通

**做了什么：**
1. **环境**：新建 conda 环境 `eval-external`，从 `vagen-lite` 复制而来（`vagen-lite` 本身一个包都没动）。
   - 核心包和训练环境一样：torch 2.8.0+cu128、transformers 4.56.1、vLLM 0.11.0、datasets 5.0.0。
   - 评测后端：VLMEvalKit 用 EASI v0.2.2 钉住的子模块（`ba60606`）；lmms-eval 用上游 `9a33763`，这是 EASI 钉的 `1516dba` 的后续版本，多了 SAT 任务。两个都不带依赖安装，缺的依赖再按"已装的包一律锁版本"补装。克隆在 `eval_external_assets/repos/`。
2. **注意力实现改成 SDPA**（PyTorch 自带的注意力算子）。原因是官方 flash-attn 预编译包需要 glibc 2.32，作业镜像只有 2.31。VLMEvalKit 加载 Qwen2.5-VL 时把 flash-attn 写死了，所以我在 `qa_transfer/vlmeval_launch.py` 里包了一层 `Qwen2VLChatSDPA`：只改这一个参数，不改 VLMEvalKit 的文件。lmms-eval 直接传 `attn_implementation=sdpa`。我们的训练环境本来也没装 flash-attn。SDPA 和 flash-attn 数值上有极小差别，贪心解码偶尔会翻转个别答案，对总分的影响预计在零点几分以内。
3. **网络**：登录节点走默认代理下 Hugging Face 只有约 0.1 MB/s；不走代理直连 hf-mirror 约 30 MB/s。下载统一用 `tools/login_env.sh`。作业里一律离线（`HF_HUB_OFFLINE=1`），所有数据都在登录节点先下好。
4. **自己的 HF_HOME**：`~/.cache/huggingface` 下的 `datasets/` 和 `vsibench/` 是以前作业以 root 身份留下的，我们写不了。eval_external 用 `eval_external_assets/hf_home`，其中 `hub/` 软链回共用的模型缓存。
5. **lmms-eval 的 VSI-Bench 任务**：原任务要求本地存有 HF 登录令牌（`token: True`），但数据集是公开的。我不把你的令牌发给第三方镜像站，所以加了 `lmms_tasks/vsibench_ee.yaml`，只把这一项改成 False，其余引用原任务。
6. **下载**：数据和 7 个外部模型全部到齐，合计约 125 GB。
   - 镜像站断过一次连接，下载脚本已加重试。
   - Video-R1 的仓库里还带了训练状态文件（`rng_state_*.pth` 等 pickle 格式）。模型一律只下 safetensors 权重、配置和分词器文件，视图里也不链接任何 pickle 文件。
7. **模型视图**（适配层）：每个模型一个只读目录 `eval_external_assets/model_views/<代号>/`。
   - 权重是软链。
   - 配置统一成 bf16。我们的 RL 检查点存的是 fp32。
   - 生成配置统一成贪心解码，`repetition_penalty` 设为 1.0（Qwen 自带的是 1.05）。
   - 每个模型的权重 sha256 缓存在 `model_hashes/`。
   - 我们的 SFT、RL 检查点和基座用的是同一份聊天模板（哈希相同）。
8. **数据核对**：作业运行前，在登录节点用离线模式各加载一遍。
   - SAT-Real 150 题，`nv-njb/SAT@85f0ea8`。
   - VSI-Bench 5,130 题，`nyu-visionx/VSI-Bench@3a6a1cf`。
   - VLMEvalKit 的 VSI 题目表 md5 和它期望的一致。
9. **第一批作业失败了一次**：作业镜像里没有 `git` 命令，写清单时取提交号报错。已改成直接读 `.git` 文件，并重新提交。
10. **作业排队**：`sco/queue.py` 在登录节点后台运行。它每分钟检查一次：有空名额且数据已准备好就提交下一个作业；跑完的作业自动汇总成 `results.json` 后删除；失败的作业保留现场，不自动重交。状态写在 `runs/qa/<阶段>/queue_status.json`。
11. **第 2 步要用的 benchmark 也已准备好**，都在登录节点离线加载过一遍：
    - ViewSpatial 5,712 题。
    - MindCube-Tiny 1,040 题：EASI 的题目表比 MindCube 自己的 jsonl（1,050 题）少 10 道。
    - MindCube 完整集 21,154 题。
    - MMSI 1,000 题，SPBench-MV 319 题。
    - VSTI-Bench 两版：
      - 修正后 5,736 题，`0903eb0`。
      - 修正前 6,042 题，`27ff0a2`。SI 写的"839 道相机位移题"就是修正前的数，所以 SI 的 VSTI 数只能和修正前这版对。
      - 评分代码从 VLM-3R 原样拷贝（Apache 2.0），只加了出处说明。
    - SAT-Syn：用 `array/SAT` 的合成验证集（4,001 题），按 nv-njb 的做法（种子 42、逐行打乱选项）转换。同一做法在 SAT-Real 上 150/150 复现了 nv-njb 的选项顺序。
12. **WalkerBench 接入**（第 4 步用，`closed_loop/walkerbench/`）：每次运行复制一份 WalkerBench，在同一个作业里起 vLLM 服务，用它自带的智能体、回合循环和打分。冒烟测试用一个假的模型服务跑了 5 道题，链路全部打通。过程中发现参考代码有两处问题，都会让模型**一直收不到图片**，而且报错被吞掉：
    - `import requests` 被注释掉了，取图时抛 NameError。
    - 依赖 `py360convert` 缺失时，渲染报错也被吞掉。

    我们的接入直接读本地图片，并补装了 `py360convert`。另外，官方离线全景图包缺 5 道"高度"题的起点图，这 5 道题开局没有画面。
13. **第 5 步脚本**（`diagnostics/`）：
    - `ownenv_eval.py` / `ownenv_job.sh`：用我们环境自己的评测（默认设置，与第十、十一轮相同）跑任意模型，可选把历史打乱或去掉。只在模型调用外面包一层，不改原文件。
    - `failure_breakdown.py`：按每步的 `met` 标记把回合分成四类。已在第十轮现有轨迹上跑过，见下面的结果。

**结果：第 1 步要求的两项（VSI-Bench、SAT-Real）都对上了。** 额外跑的 7 项里，有 4 项对上；另外 3 项差距超阈值，但原因都能指出来，看起来出在 SI 报的数本身，而不是我们的管线。

base-7B 和 Spatial-Interactor（SI）报的基座数对照（汇总表在 `results/qa_step1.md`）：

| benchmark | 我们（95% 区间） | SI 报的 | 差 | 判断 |
|---|---|---|---|---|
| **VSI-Bench**（5,130 题，32 帧，VLMEvalKit） | 37.6 [36.3, 38.9] | 37.9 | −0.3 | 对上 |
| 　路线规划 / 相对距离 | 29.4 / 37.3 | 30.4 / 36.6 | −1.0 / +0.7 | 对上 |
| 　同上，lmms-eval 后端 | 36.8；30.9 / 38.6 | 同上 | −1.1；+0.5 / +2.0 | 也在阈值内 |
| **SAT-Real**（150 题） | 58.0 [50.0, 65.6] | 54.7 | +3.3 | SI 的数在我们的区间内，按约定不查 |
| ViewSpatial（5,712 题） | 37.7 [36.4, 38.9] | 36.3 | +1.4 | 对上（EASI 自测和原论文都是 36.85） |
| MMSI（1,000 题） | 28.0 [25.3, 30.9] | 27.7 | +0.3 | 对上 |
| VSTI-Bench 修正前（6,042 题），5 类平均 | 40.1 [38.9, 41.3] | 39.0 | +1.1 | 对上。评测代码打印的总分是 9 个题型平均，为 47.2，口径不同；SI 和 VLM-3R 论文报的是 5 类平均 |
| 　相机位移（839 题，MRA） | 11.7 | 8.6 | +3.1 | 子项超阈值，原因未查明（可能是抽帧或分辨率差异） |
| SAT-Syn（4,001 题） | 59.1 [57.6, 60.6] | 57.5 | +1.6 | 刚过阈值。SI 没写用的是哪个划分，选项顺序也可能和我们不同 |
| MindCube-Tiny（1,040 题） | 34.2 [31.4, 37.2] | 29.3 | +4.9 | SI 的基座数 29.3（7B）、33.2（3B）恰好等于 MindCube 原论文在**完整集**上报的 29.26、33.21 |
| MindCube 完整集（21,154 题） | 29.4 [28.8, 30.0] | （29.3） | +0.1 | **证实**：SI 表里标为 Tiny 的基座数，其实就是完整集的数 |
| VSTI-Bench 修正后（5,736 题），5 类平均 | 41.5 [40.2, 42.7] | — | — | 修正后标签下相机位移是 9.2（修正前 11.7） |
| SPBench-MV（319 题） | 42.7 [37.4, 48.2] | 34.0 | +8.7 | SI 的 3B 数（36.6）和 SPBench 原论文完全相同，7B 数和原论文（37.3）、EASI 自测（40.7）都对不上。我们比 EASI 高 2.0，在区间内 |

所有已跑完的 benchmark，base-7B 无法解析的回答都在 0.2% 以下。

**VSTI 那次作废的运行**：修正后标签那次实际读到了修正前的数据。两个版本都在 HF 缓存里，离线时 `datasets` 不按版本号区分，只取了其中一份。两版现在都直接读我们钉住的 `test.json`，离线核对分别是 5,736 题和 6,042 题。作废的运行放在 `runs/qa/step1/_invalid/`，正在重跑。教训：每个 benchmark 都要做离线加载核对，VSTI 当时漏了。

**作业耗时**（base-7B，单张 H800）：

| benchmark | 耗时 |
|---|---|
| VSI-Bench（VLMEvalKit） | 4.7 小时 |
| VSI-Bench（lmms-eval） | 1.9 小时 |
| MindCube 完整集 | 约 4.3 小时 |
| VSTI-Bench（每一版） | 约 2 小时 |
| ViewSpatial | 1.2 小时 |
| 其余 5 项 | 合计不到 1 小时 |

我们环境的评测每个模型 30–40 分钟。

我们环境上的失败拆解（第十轮已有轨迹，279 道 test 题）：

| 模型 | 到达 | 提前结束 | 用完步数 |
|---|---|---|---|
| base-7B | 9.0% | 3.2% | 87.8% |
| ours-SFT | 40.1% | 0.4% | 59.5% |
| ours-RL-50 | 40.5% | 0.4% | 59.1% |
| ours-RL-150 | 44.1% | 0.7% | 55.2% |

- 这个环境一满足要求就自动结束回合，所以"到过之后又走偏"按规则为 0。
- 这个环境没有"主动停止"动作，"提前结束"指环境因连续 3 帧低信息画面而结束。
- 失败回合结束时最常见的未满足条件是：目标不在画面内、目标看不清。

**下一步：** 等你确认第 2 步的设置。我的建议：
- VSI 用 VLMEvalKit 后端。
- 加入第十一轮的 SFT 第 9 个 epoch 和 P-S0 第 300 次这一对模型。
- MindCube 完整集保留。

还在跑的有 VSTI-Bench 修正后版本的重跑和 MindCube 完整集。

---

## 2026-10-19 第 2 步启动，两道关都通过

**结论：** 我们环境和 WalkerBench 的接入都对上了已知结果；第 2 步、第 4 步、第 5 步的作业已经全部排进一个队列（163 项），按"两道关 → 我们环境 → WalkerBench → QA"的顺序跑。

**关 1：我们环境**
- 用新接入方式（eval 工作树、`eval-external` 环境、bf16 视图、显式 `repetition_penalty` 1.05）重跑 ours-SFT：成功 112/279 = 40.14%，和第十轮的数完全一样。
- 逐题比，265/279 道题成败相同，208 道每一步的回答都相同。翻转的 14 道正负抵消，这是 vLLM 在不同卡、不同批次下的正常运行间差异，约 5% 的回合。

**关 2：WalkerBench**（base-7B，Standard-100）

| 设置 | 总成功率（95% 区间） | 角度 / 距离 / 高度 / 导航 / 可见性 | 平均步数 |
|---|---|---|---|
| 有图（我们的接入） | 9.0 [4.8, 16.2] | 0 / 0 / 40 / 0 / 5 | 16.5 |
| 不给图（参考智能体的实际行为） | 11.0 [6.3, 18.6] | 0 / 0 / 50 / 0 / 5 | 18.6 |
| SI 报的 | 7.0 | 5 / 0 / 25 / 5 / 0 | 17.71 |

- 两种设置都在 5 分阈值内。
- 每类只有 20 题，1 题就是 5 分。
- 基座模型的成功几乎全来自"高度"题。不给图时高度题反而更高，说明它主要靠常识估高度，图片没帮上忙。

**这一路修掉的问题**（都已提交）：
1. vLLM 服务启动时要现编代码，作业镜像没有 C 编译器 → 改用环境自带的 conda gcc。
2. FlashInfer 采样算子现场编译时缺 `curand.h` → 改用 PyTorch 自带的采样器。贪心解码只取最大值，结果不受影响。
3. vLLM 开启 CUDA graph 后，开跑 2 分钟报"CUDA 非法内存访问"崩溃 → 改用 eager 模式，和我们环境的评测一致。
   - 崩溃后的第一版结果（6.0%）其实是智能体原地空转得出的，已作废。
   - 现在服务挂掉、或有任何调用未应答，这次运行就判无效，结果另存为 `results_invalid.json`，不会被当成正常结果收走。
4. WalkerBench 自带回合循环的一个缺陷：动作执行失败后观测为空，下一步取字段时抛异常，这道题就没有日志。
   - 这类题按失败计分，题数单独报告：有图那次 1 道，不给图那次 6 道。
   - 失败的旧目录在 `runs/walkerbench/step2/_failed/`。

**队列内容**（`logs/queue_step2.sh`，状态在 `runs/queue_step2.json`）：
- 我们环境（第 5 步）：
  - 6 个外部 7B 模型零样本：SI-7B、SpaceR、Video-R1、ViLASR、VST-7B-RL、SenseNova-SI。
  - 两对我们的模型（第十轮 SFT / RL-150、第十一轮 SFT-e9 / P-S0-300），各跑正常 / 打乱历史 / 去掉历史。
- WalkerBench（第 4 步）：其余 13 个模型，加 SI-7B 不给图。
- QA（第 2 步）：13 个模型 × 9 个 benchmark，VSTI 修正前标签另给 base-3B、SI-3B、SI-7B 跑。

## 2026-10-19 第 5 步：外部模型在我们环境里的第一批结果

**结论：** 6 个外部 7B 空间模型在我们环境里没有一个高过基座（base-7B）。5 个和基座持平或略低（6.8%–9.3%，基座 9.0%）；SI-7B 掉到 1.8%，主要是回答格式的问题，不是看不懂空间。

"成功率"指 279 道测试题（第十轮 `eval_test`）里，在 12 轮之内任一轮满足任务条件的题数占比；括号里是 95% Wilson 区间。"无效轮"指回答里解析不出合法动作的轮次占所有轮次的比例。

| 模型 | 成功 / 279 | 成功率（95% 区间） | 无效轮 | 每个动作的碰撞数 |
|---|---|---|---|---|
| base-7B（第十轮评测的数） | 25 | 9.0 [6.1, 12.9] | 0% | — |
| ours-SFT（第十轮最终模型） | 112 | 40.1 [34.6, 46.0] | 0% | 0.12 |
| SI-7B | 5 | 1.8 [0.8, 4.1] | 75.2% | 0.13 |
| SpaceR-7B | 23 | 8.2 [5.6, 12.1] | 0% | 0.05 |
| Video-R1-7B | 20 | 7.2 [4.7, 10.8] | 0% | 0.04 |
| ViLASR-7B | 26 | 9.3 [6.4, 13.3] | 0% | 0.09 |
| VST-7B-RL | 19 | 6.8 [4.4, 10.4] | 0% | 0.01 |
| SenseNova-SI-7B | 25 | 9.0 [6.1, 12.9] | 0% | 0.09 |

- 失败的题几乎全是"预算用完"（12 轮走完还没满足条件），只有 SI-7B 是"提前结束"（244 题，见下）。
- 基座这一行来自第十轮直接跑 `evaluation.py` 的结果。为了让所有模型走同一条代码路径，队列里补了 base-7B 用本工作树的接入方式重跑一遍。

**SI-7B 的格式问题**
- 它给出的动作名是对的（如 "Answer: move_forward|move_right|move_forward"），但 75% 的轮次没有按提示词要求加 `<action>…</action>` 标签。
- 我们环境连续 3 轮解析不出动作就结束这一题，279 题里有 244 题是这样结束的。
- 不是被截断：回答长度中位数 87 个 token，没有一条碰到 384 的上限。
- 正式结果仍按严格格式算，因为提示词写明了格式。另加一个只做诊断的"宽松解析"：回答没有标签、但最后一行是 "Answer: …" 时，把这一行套上标签再交给环境解析，每次改写都记进 `lenient_rewrites.jsonl`。只有 SI-7B 需要，其余 5 个外部模型无效轮都是 0。

**为什么 SpaceR、Video-R1 重跑过**
- 这两个模型（还有 SenseNova-SI）自带的预处理配置把单张图的像素上限设得很低：SpaceR、Video-R1 是 401,408，SenseNova-SI 是 147,456（Qwen2.5-VL 默认 12,845,056）。VLMEvalKit 和 vLLM 加载处理器时不传像素参数，所以只有这几个模型的图会被缩小。Video-R1 的配置还写了 transformers 不认识的处理器名。
- 改成所有模型视图都用同一套像素上下限（3,136 到 12,845,056，Qwen2.5-VL 的默认值），处理器名统一成 `Qwen2VLImageProcessor`，改动记在每个视图的 `view_changes.json`。
- 改之前跑的 SpaceR、Video-R1 结果挪到 `runs/ownenv/_invalid/`，不用。

**历史打乱（第一个结果）**
- ours-SFT 把对话里之前各轮的顺序打乱后，成功 116/279 = 41.6%，和不打乱的 40.1% 差不多。去掉历史那组还在跑。

**QA 的一个早期信号（只有一个模型，待全部跑完再下结论）**
- ours-SFT 在 VSTI-Bench（修正后标签，五类平均）上是 30.1，基座是 41.5。
- 掉得最多的是"相机到物体的距离"：30.1 → 4.4。看回答，ours-SFT 报的距离中位数只有真实值的 0.12 倍（基座是 0.83 倍），86% 的回答不到真实值的一半。
- 也就是说，SFT 之后模型的米制尺度明显偏小。等其余模型跑完，会对所有模型统计同一个比值。

**产物**：`runs/ownenv/eval_test/<模型>/results.json`（成功率、区间、无效轮、失败分类、manifest）；逐题结果在同目录 `results.jsonl`。

**下一步**：队列继续跑（截至 07:00 UTC 完成 20 项，4 项在跑）；我们环境里剩下的是 RL-150 和第十一轮那一对的历史诊断、SI-7B 宽松解析、base-7B 重跑。
