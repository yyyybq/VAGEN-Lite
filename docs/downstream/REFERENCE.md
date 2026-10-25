# 下游具身任务线：设计、规则、代码与路径（参考）

最后更新：2026-10-23 20:15 UTC。现状看 [STATUS.md](STATUS.md)，各实验的进度和成功率看 [MATRIX.md](MATRIX.md)（看板每轮重写），评测协议看 [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md)，报告在 [reports/](reports/)。

## 1. 分阶段设计（你 10-22 的指令；同日第二条指令修订）

| 编号 | 阶段 | 内容 | 开始条件 |
| --- | --- | --- | --- |
| E0 | 0 | 评测 D5 停下时保留的 16 个检查点（四个起点 × 2.5 万 / 5 万 / 7.5 万 / 10 万步），顺序：四个起点的 10 万步 → 5 万步 → B 10 万步第二遍（整机，噪声估计）→ 2.5 万和 7.5 万。全部评完后出阶段 0 报告：四条曲线（总体、空间组）、步数规则的判定、空间任务里四起点平均成功率在 10%–60% 的清单（给 E3 选任务）、评测噪声的初步估计（B 10 万步分片版和整机版之差，种子相同） | 主窗口许可 |
| E1 | 1 | DS-{B,S,RS,RD}-100-s1 从头跑 20 万步；全状态续训。**评第 15 万、17.5 万、20 万步**（你 10-23 晚的决定 2）：第 15 万、17.5 万步的权重由看板在全状态被替换前复制到 `EXP/evals_src/steps_<N>/`，整机评测；第 20 万步在任务内评。报告主数字用三点平均，并列三点之间的配对差作为训练末尾的起伏尺度。B 的第二遍评测（repeat_final）已取消 | 10-23 00:02 已提交（你的决定 3，20 万步） |
| E2 | 2 | DS-{B,S,RS,RD}-010-s1 和 DS-{B,S,RS,RD}-030-s1，设置同 E1（20 万步、全状态每 2.5 万步、训练任务自己评第 20 万步） | 010：10-23 00:02 提交；19:17 按你的决定 1 停下；20:22–20:34 按空机规则续训。030：10-23 20:45 按你的决定提交（B、S 在 h800，RS、RD 在 zoetrope；RD 在 zoetrope 没排上，20:54 按你的选择改投 h800），实验目录同时重建（10-21 的旧文件加 `_made_20261021` 存档） |
| E3 | 3 | 四个起点各 1 个 PPO，从各自 E1 的最终检查点出发；任务是阶段 0 定的 CoffeeSetupMug、OpenDrawer、SlideDishwasherRack、TurnOffStove（E1 评完后按同一规则复核，变了先报告）；轮转和训练轮数各 2，32 个环境（第二次实测：64 个要约 34 小时，超出 30 小时），更新 100 次，每个起点约 22 小时；RL 前后用同一套 944 回合评测；最多 2 个 RL 同时跑。方案见 [reports/STEP4_RL_PLAN_20261023.md](reports/STEP4_RL_PLAN_20261023.md) | E1 评完、第二次速度实测之后，按空机规则提交 |
| E4 | — | 种子 2、3 | 暂不跑；E1 结果出来后再定种子 2（你 10-23 晚的决定 3）；已有的目录留着不删 |

- 名额：下游所有 8 卡训练（E1、E2 的训练，RL，测试）共用 4 个，RL 至多占 2 个；1 卡分片最多 8 个；8 卡评测任务不算训练名额。你在和主窗口谈动态名额，谈妥再改。
- 统计：逐任务配对检验（34 个任务为配对单位）+ 评测噪声下限（同一检查点两次评测之差），见 EVAL_PROTOCOL.md 第 3、4 节。配对一律只用两边都完成的回合（逐任务差也是）。
- 阶段 0 报告（10-23，最终版 20:00，[reports/E0_STAGE0_20261023.md](reports/E0_STAGE0_20261023.md)，16 个评测都是 944 回合）：
  - 步数规则判 20 万步，与 E1 一致；
  - 评测噪声下限为 0：B 10 万步评两遍，944 个回合结果全同；
  - 同一起点相邻检查点之差（总体平均 3.3、最大 6.8 个百分点）和同一检查点起点之差（平均 2.9、最大 7.4）一样大，所以单个检查点、单个种子上的起点差要按这个尺度读。因此 E1 改为评三个点。
- 10-22 第一条指令里的"1.5 倍规则"（96 个进程每回合比 24 个进程慢超过 1.5 倍就改评法）已撤销：慢是因为同一台机器被更多进程分用，整机吞吐不变（约每秒 120 个控制步），任务内评测并不低效（你 10-22 第二条指令第 1 条）。

## 2. 评测的运行规则

- **现行规则（你 10-23 晚的决定 1，19:48 起由看板执行）**：
  - 主线（`interiorgs-v10-*`）有任务处于排队状态（STARTING、QUEUEING 等；SUSPENDED 是已停的旧任务，不算）时，什么都不提交；
  - 8 卡任务只投到估计有空闲机器的池，每轮最多 1 个；我们自己有 8 卡任务在那个池排队时不再提交；
  - 按空机提交、还在排队的任务（续训、队列里的任务、8 卡评测），主线一开始排队就撤回，之后再提；已经开始跑的不动；
  - 每轮的顺序是：E1 的 8 卡评测 → 队列里的任务（`downstream_runs/queue/*.json`，比如 RL 速度实测 2）→ 停下的实验续训（`job_released_*`，E2）；
  - `PAUSE_SUBMISSIONS` 存在时一律暂停。
  - 空闲机器数是估计值：池的机器数减去有 aigc 任务的机器数；别的工作区的任务看不到，所以可能偏多。
- 下面是更早的规则，记录在此备查：
- 主窗口 10-22 的条件：同时最多 8 个 1 卡任务，只投 h800 和 zoetrope；有 `interiorgs-v10-*` 在排队时不提交；已经开始的不删；`interiorgs-v10-*` 和 `bq_tp_mmxbf16` 的任务一概不碰；`downstream_runs/PAUSE_SUBMISSIONS` 存在时暂停一切提交。
- 1 卡分片：每个检查点 8 片，每片约 119 回合、12 个仿真进程。原本只投到已有别的任务、还有空卡的机器上（填缝）；10-22 16:30 起 h800 上也可以落在空闲机器上。
- 整机 8 卡评测（10-22 16:30 起，你的要求）：每个检查点一个 8 卡任务，8 个策略服务 × 96 个仿真进程，只投 h800 的空闲机器；每轮最多提交 1 个；我们有 8 卡评测在排队时不再提交；主线有任务排队时，我们还没开始的 8 卡评测自动撤下（记录改名 `job_withdrawn_*`），之后重提。
- 卡住的评测（10-22 第二条指令第 3 条）：运行中的评测 90 分钟没有完成一个回合，看板删掉任务（记录改名 `job_hung_*`）并重提，重提的任务只补缺的回合。
- 同卡数 `bq_tp_mmxbf16` 排队时不提交同卡数的任务（8 卡任务只看同一个池）。
- 每轮读一次配额剩余空间，少于 1 TB 或读不出时只报警、不提交。

## 3. 代码与位置

- **代码与文档**：分支 `downstream`（从 main `59053083` 开出）。只新增 `downstream/` 和 `docs/downstream/`。
- **同步**：本地 worktree `C:\Users\user\Desktop\code\VAGEN-Lite-downstream` 提交 → push → 集群 `git pull --ff-only`。只在集群上生成的文件（如 `starts_lock.json`）在集群上提交、push，再在本地 pull。`docs/downstream/MATRIX.md` 由看板在集群上重写，本地不改它。
- **集群 worktree**：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-downstream`；其中 `third_party`、`playground` 是指向主检出的链接（被 .gitignore 忽略），只读使用。
- **大文件**在 `/mnt/umm/users/yinbaiqiao/downstream_runs/`（不进 git）：`data/target50_{100,030,010}`（训练数据，只建链接）、`experiments/<编号>/`、`checks/`（一次性检查）、`logs/`、看板 `BOARD.md`、看板状态 `WATCH_STATE.json`、报告 `report/`、`report_stage0/`、阶段 0 报告的数字和网页 `report_stage0_20261023/`。
- **看板** `downstream/eval/watch.py`（`downstream/eval/watch.sh start|stop|status`，代码更新后先 stop 再 start；启动参数记在 `downstream_runs/watch.args`）：10-23 20:34 起每 5 分钟一轮（`--every 300`），每 2 分钟一次分片落点检查；按第 2 节的规则提交评测、E2 续训（`--resume-ratios 010 030`）和队列里的任务；在 E1 的第 15 万、17.5 万步复制权重（`--extra-points`、`--extract-python`）；重写 `BOARD.md` 和 `docs/downstream/MATRIX.md`（训练、评测各一张表）；有新评测完成时重建结果报告。按空机提交的任务记在 `WATCH_STATE.json` 的 `idle_sent` 里。用 setsid + nohup 运行，ssh 断开后仍会继续。除了按规则删掉落在空闲机器上、排队中被撤下或卡住的评测任务，不删任何东西。
- **训练**：`downstream/train/make_experiment.py`（建实验；E1 用 `--full-state-interval 25000 --eval-points "" --archive-suffix _stopped_20261022`，复用 DS-*-100-s1 目录，旧的 experiment.json 等改名保留；B 另加 `"repeat_final": true`）、`runs.py make|submit|list`、`train_entry.py`（全状态每 2.5 万步存一次只留最新，启动时找到全状态就续训；`trainer.stop_at` 段末退出）、`full_state.py`、`run_segmented.sh`（训到最后一步 → 任务内评测最终检查点；训练退出码不为 0 时不评）、`run_train.sh`（10-21 的旧流程）。10-25 起：存好检查点和记录后关闭 NCCL 出错，`train_entry.py` 直接以 0 退出（E1 B 遇到过）；训练任务没评就结束的最终检查点，看板按空机规则单独用一个 8 卡任务评，结果写到同一个目录。续训测试：`resume_test.sh`、`resume_kill_test.sh`、`resume_check.py`。
- **评测**：`downstream/eval/run_eval_ckpt.sh CKPT OUT 服务数 进程数 [分片]`、`closed_loop.py`（10-22 起：仿真进程死了或卡住时它的回合重跑；重跑一个评测时只补缺的回合；每 30 秒记一次内存，`OUT/episodes/memory.csv` 和 `summary.json["memory"]`）、计划 `downstream/eval/eval_d6.json`、剔除清单 `downstream/eval/excluded_episodes.json`（6 个回合，944）。10-23 晚起策略服务以 `--idle_timeout -1` 启动，整个评测期间不因空闲关闭，结束时由脚本关。
- **从全状态复制权重**：`downstream/train/extract_weights.py FULL_STATE RUN_DIR DEST [--dry] [--like CKPT]`（StarVLA 环境的 Python：`/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python`）。取出 DeepSpeed `mp_rank_00_model_states.pt` 里的 bf16 模型权重，按 StarVLA 检查点格式写到 `DEST/checkpoints/steps_<N>_pytorch_model.pt`，并复制 `config.yaml`、`dataset_statistics.json`，最后写 `source.json`。看板在 E1 第 15 万、17.5 万步自动调用。
- **报告**：`downstream/report/`（`python -m downstream.report.build --out DIR [--stage0] [--partial]`，vagen-lite 环境；`--partial` 把还没写完的评测按已完成回合计入并标出缺的回合）。主检验逐任务配对 + 噪声下限；E1 有第 15 万、17.5 万、20 万步三点时，主数字是三点平均（`stats.compare_mean`、`build.e1_points`），页面有"E1 三点"一节，并列三点之间的配对差；阶段 0 页含曲线、缺回合的评测、10%–60% 清单（第 10 万步和全部检查点两种算法）、步数规则表、最后两点之差、检查点之间的起伏。网页版（私有 Artifact）：https://claude.ai/artifact/Qa1oNasyCN5pSkjg2LXhBq 。模拟数据在 `downstream_runs/checks/report_sim_20261021/`。
- **第四步（E3）准备**：`downstream/rlinf/fsdp_7b.overrides`（7B 的三个 FSDP 设置）、`site/sitecustomize.py`（RLinf 的环境进程加台面区域顺序修正、分开记演员和价值头的梯度范数）、`run_rlinf.sh`（正式启动脚本）、`export_test.sh`（导回 StarVLA 格式 + 短评测，10-22 测通）、`speed_test.sh`（第一次速度实测，RL-SPEED-1）、`speed_test2.sh`（第二次：64 / 32 个环境，轮转 2、训练 2，RL-SPEED-2，在队列里等空机）。D7 环境：`/mnt/umm/users/yinbaiqiao/envs/rlinf-starvla-rc365`（`source downstream/envs/activate_rlinf.sh <ENV>`）。
- **StarVLA**：`4507931` + 补丁 [downstream/patches/starvla_4507931_working.patch](../../downstream/patches/starvla_4507931_working.patch)；每个实验的 `experiment.json` 记提交号和补丁哈希。
- **剩余空间**：`python3 downstream/quota.py`（读 `sco afs dir-quota list` 整张表里我们这一行）。

## 4. 权重与数据

起点登记在 [downstream/starts.json](../../downstream/starts.json)，锁定值在 [downstream/starts_lock.json](../../downstream/starts_lock.json)；每次建实验和每个训练任务开头都校验，变了就停。路径前缀 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/`。训练时所有起点按 bf16 加载。

| 名字 | 用途 | 路径 | 格式 / 精度 |
| --- | --- | --- | --- |
| B | 原始 Qwen2.5-VL-7B-Instruct（`cc594898`） | `interiorgs_v10_r11_20261016/inputs/base_7b` | 完整 HF，bf16 |
| S | 第十一轮动作 SFT 第 9 个 epoch | `interiorgs_v10_train_r11_20261016/lf_formal/model/checkpoint-1386` | 完整 HF，bf16 + fp32 混合 |
| RS | P-S0 第 300 次 | `interiorgs_v10_train_r11p_20261017/rl_run_ps0/checkpoints/global_step_300/actor/huggingface` | 完整 HF，fp32 |
| RD | P-D 第 150 次 | `interiorgs_v10_train_r11p_20261017/rl_run_pd/checkpoints/global_step_150/actor/huggingface` | 完整 HF，fp32 |
| R10S5（备用） | 第十轮 SFT 第 5 个 epoch | `interiorgs_v10_train_r10_20261015/lf_formal/model/checkpoint-770` | 完整 HF |
| R1（备用） | R1 最终模型 | `R1-clean-Projective-v0/formal/checkpoints/global_step_250/actor/huggingface` | 完整 HF |

| 数据 | 清单 | 规模 |
| --- | --- | --- |
| 第二步 | [step2_smoke.json](../../downstream/demos/manifests/step2_smoke.json) | 3 个任务各前 100 条示教（训练）+ 后 20 条（留出），训练 44,369 帧 |
| 100% | [target50_100.json](../../downstream/demos/manifests/target50_100.json) | 50 个任务，25,307 条示教，14,957,899 帧 |
| 30% | [target50_030.json](../../downstream/demos/manifests/target50_030.json) | 每任务 150 条（官方 `150_demos`），7,500 条，4,442,662 帧 |
| 10% | [target50_010.json](../../downstream/demos/manifests/target50_010.json) | 每任务 50 条（官方 `50_demos`），2,500 条，1,494,106 帧；10% ⊂ 30% ⊂ 100% |
| 原始示教 | `/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365/datasets/v1.0/target/` | 50 个任务齐全，58 GB |

## 5. 已结束的任务

| 任务 | 内容 | 结果 | 目录 |
| --- | --- | --- | --- |
| STEP1-LOAD | 四个起点的加载测试（1 卡） | 通过 | `downstream_runs/step1_load_test_20261020/out/` |
| STEP2-RS-s1 | 第二步训练（8 卡，200 步 434 秒） | 通过（10-21 确认） | `downstream_runs/experiments/STEP2-RS-s1/` |
| `pt-wytuck3x` | RLinf 2 次更新试跑，第四次 | 通过（10-22 10:03–10:19） | `downstream_runs/experiments/RLINF-SMOKE-1/` |
| `pt-cb0kvvf0` | 34 个评测任务的重置扫描 | 原版 89 / 102 对可复现，固定顺序后 102 / 102 | `downstream_runs/checks/reset_sweep_20261021/` |
| 29 个 | 10-21 提交的第三步训练和 D5 评测 | 10-22 09:08 撤下；输出改名 `run_stopped_20261022/`，记录改名 `job_cancelled_20261022/`；D5 的 4 个停在第 113,700–118,800 步，保留了 2.5 万–10 万步检查点（E0） | `downstream_runs/experiments/DS-*/` |
| `pt-i4bzrj4k` | 续训测试第一次 + 96 进程评测检查 + 第四步导出测试（8 卡） | 恢复后逐步完全一致；全状态 109.8 GiB，存 68 秒，读 243–270 秒；"杀"只杀到启动器（弹性启动器的会话），所以重测；96 进程检查 95 回合 39 分钟；导出 78 秒、导出后 24 回合评测正常 | `downstream_runs/checks/resume_test_20261022/` |
| `pt-uv1mh7fe` | 续训重测（8 卡） | 通过：第 150 步杀掉、无残留进程；从第 100 步恢复、段末第 200 步自动退出、再续到 300；第 101–300 步损失、300 个批次哈希、每一步学习率都与不中断的训练相同 | 同上，`RESUME-B2/`、`resume_check_RESUME-B2.json` |
| 测试产物 | 3 份全状态、6 个权重、1 个导出策略，共 440 GB | 10-22 按你的指示删除，留日志、比对 JSON 和评测记录（7.8 MB） | 同上 |

## 6. 决定记录

**10-21 你对第二步报告的决定**：固定台面区域顺序（已执行）；950 回合评测（已执行）；同一检查点评两遍测噪声（10-22 改为 E0 的 B 10 万步两遍、E1 只做 B 的最终检查点两遍）；加强起点校验（已执行）；3 个边界任务单列（已执行）；D5 的中间检查点先留着、剩余空间少于 1 TB 报警（已执行）。

**10-22 你的指令**：
- 撤到只剩连通试跑，其余等密集奖励第一批上机器（已执行）；评测拆成 1 卡、只投已有占用的机器（已执行，后来按 16:30 的要求放宽到 h800 空机）。
- 分阶段设计、先不比种子（第 1 节）。
- 第二条：续训先测通再提交阶段 1（已测通）；任务内评测（已实现）；阶段 1 的提交时机和步数规则；名额 4 个、RL 至多 2 个；闸门；第四步准备（已做）；种子 2、3 的目录留着；第一个分片跑完后报评测时长和预计完成时间（18:05 已报）。
- 16:30：把排队和没提交的迁到 h800 的空机上（已执行）。
- 第三条：1.5 倍规则撤销，E1 只在最终检查点任务内评、B 的最终检查点做第二遍（整机）；删 440 GB 测试文件（已删）；整机评测记内存峰值、超过 80% 降进程数；卡住的整机评测删掉重提只补缺的回合（已实现）；E1 按已定条件提交；看板每轮重写 MATRIX.md、STATUS.md 只留现状；名额先按 4 个；E0 全部评完后出阶段 0 报告。

**10-23 的决定**（执行记录见 STATUS 第 5 节）：
- 00:00：固定剔除清单（5 个，945）；部分结果报告；E1、E2 直接提交（20 万步）、RL 速度实测、E0 剩下的评测；去掉名额上限和让路，只留暂停文件。
- 晚（阶段 0 报告之后）：
  1. GPU：E2 停下，机器让给主线 A 组，有空机再续；E1 继续。
  2. E1 评第 15 万、17.5 万、20 万步，主数字用三点平均；取消 B 的第二遍评测。
  3. 种子 2 先不加，E1 结果出来后再定。
  4. 剔除清单加 LoadDishwasher 1002（944）；修策略服务的 30 分钟空闲关闭。
  5. 第四步方案按实测改：轮转和训练轮数各 2、试 64 个环境、更新 100 次、每个起点不超过 30 小时、任务用阶段 0 的 4 个。
  6. MATRIX 和 STATUS 按原来的行列维护，训练和评测分开。
- 晚 20:34：看板改为每 5 分钟一轮、每轮最多 4 个 8 卡任务（你说"现在有很多空卡"）。
- 晚 20:4x：E2 30% 组四个都上，两个交 zoetrope、两个交 h800，做好新的实验记录（`downstream_runs/logs/make_e2_030_20261023/`，实验目录里的 `experiment.json`）。
