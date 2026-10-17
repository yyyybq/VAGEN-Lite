# 外部迁移评测：进度记录

分支 `eval-external`，工作树 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-eval`（从 `origin/main` 的 `a59f9583` 开出）。大文件（数据、模型、克隆的 benchmark 仓库、作业输出）放在 `/mnt/umm/users/yinbaiqiao/eval_external_assets/`，不进 git。

## 当前状态

**第 0 步做完了，等你确认**（见 `step0/FEASIBILITY.md` 末尾"需要你确认的事"）。还没有装任何环境，也没有下载数据或模型。

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
