# 外部迁移评测：进度记录

分支 `eval-external`，工作树 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-eval`（从 `origin/main` 的 `a59f9583` 开出）。大文件（数据、模型、克隆的 benchmark 仓库、作业输出）放在 `/mnt/umm/users/yinbaiqiao/eval_external_assets/`，不进 git。

## 当前状态

**第 1 步进行中**：环境、数据、12 个模型的视图都已就绪；base-7B 在全部 QA benchmark 上的复现作业由排队脚本在跑，已出 SAT-Real、SAT-Syn、MindCube-Tiny 三项（见下面 10-18 一节）。第 4、5 步的脚本已写好并测过，等第 1 步通过后再用 GPU。

## 已定的事（2026-10-18 你确认）

1. ESI-Bench 先跳过。
2. ours-SFT 是第十轮最终模型 `interiorgs_v10_train_r10_20261015/lf_formal/model/checkpoint-770`；ours-RL 是第十轮 RL 第 150 次（代号 `ours-RL-150`），加测第 50 次（`ours-RL-50`）。
3. 外部模型：SI-7B、SI-3B，加 5 个 7B 专用模型（SpaceR、Video-R1、ViLASR、VST-7B-RL、SenseNova-SI-1.1）。
4. QA 以 EASI 的 VLMEvalKit 后端为主；SAT 和 VSTI 用 lmms-eval。SAT-Real 的复现标准改成"SI 的数落在我们的 95% 区间外才查"。
5. GPU：单卡作业，放 h800 池，同时最多 4 个。

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

**结果（截至目前）：**

base-7B 和论文报的基座数对照：

| benchmark | 我们（95% 区间） | SI 报的 | 差 | 判断 |
|---|---|---|---|---|
| SAT-Real（150 题） | 58.0 [50.0, 65.6] | 54.7 | +3.3 | SI 的数在我们区间内，按约定不查 |
| SAT-Syn（4,001 题） | 59.1 [57.6, 60.6] | 57.5 | +1.6 | 刚好落在区间外 0.1 分。SI 没写用哪个划分，选项顺序也可能不同 |
| MindCube-Tiny（1,040 题） | 34.2 [31.4, 37.2] | 29.3 | +4.9 | 超阈值。SI 的基座数 29.3（7B）和 33.2（3B）恰好等于 MindCube 原论文在**完整集**上报的 29.26 和 33.21，很可能是直接引用的，不是在 Tiny 上测的。完整集结果出来后核对 |

所有已跑完的 benchmark，base-7B 无法解析的回答都在 0.2% 以下。

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

**下一步：** 等 VSI-Bench 两个后端、ViewSpatial、MMSI、SPBench-MV、VSTI 两版和 MindCube 完整集的 base-7B 结果出齐，判断第 1 步是否通过，以及 VSI 用哪个后端。
