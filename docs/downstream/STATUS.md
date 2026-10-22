# 下游具身任务线状态（唯一的状态文档）

最后更新：2026-10-22 18:20 UTC。实验设计按你 10-22 的指令分阶段（第 1.1 节）；你 10-22 第二条指令定了续训、任务内评测、阶段 1 的提交时机和步数规则、名额和闸门（第 6 节）；16:30 起按你的要求把还没提交的评测和续训测试放到 h800 的空闲机器上。任务书分四步，每步做完停下来等确认。

## 1. 现在在哪一步

- **第一步（盘点与核实）**：已完成并经你确认。报告 [reports/STEP1_INVENTORY_20261020.md](reports/STEP1_INVENTORY_20261020.md)。
- **第二步（连通试跑）**：已完成，10-21 经你确认通过。报告 [reports/STEP2_SMOKE_20261020.md](reports/STEP2_SMOKE_20261020.md)；RLinf 两次更新的试跑 10-22 通过，报告 [reports/STEP2_RLINF_SMOKE_20261022.md](reports/STEP2_RLINF_SMOKE_20261022.md)。
- **第三步：按阶段进行，现在在阶段 0**（下节）；阶段 1 的续训和分段式任务已实现，实测进行中。
- **第四步**：方案初稿 [reports/STEP4_RL_PLAN_DRAFT_20261022.md](reports/STEP4_RL_PLAN_DRAFT_20261022.md)；准备工作已做（第 3 节 `downstream/rlinf/`）。

### 1.1 分阶段设计（你 10-22 的指令）

| 阶段 | 内容 | 开始条件 | 状态 |
| --- | --- | --- | --- |
| 0 | 评测 D5 停下时保留的 16 个检查点（四个起点 × 2.5 万 / 5 万 / 7.5 万 / 10 万步）。顺序：四个起点的 10 万步 → 5 万步 → 2.5 万和 7.5 万。每评完一个起点的 10 万步更新 BOARD | 主窗口许可 | **进行中**（第 2 节）：DS-B 10 万步 8 个 1 卡分片快完（6 片各用 2.88–2.98 小时）；S / RS / RD 10 万步（16:42 起）和 B 5 万步（16:56 起）各一个 8 卡评测在跑，估计各约 3.2 小时；其余 11 个等暂停解除和空闲机器。全部 16 个预计 10-23 06:00–12:00 UTC 完成（按同时能拿到 4 台或 2 台空机器估，未核实） |
| 1 | DS-{B,S,RS,RD}-100-s1 从头跑；全状态续训；分段式任务：训到 5 / 10 / 15 / 20 万步时在任务内评（8 个策略服务 × 12 个仿真进程），再续训；最终检查点的第二次评测走分片 | 续训测通，且阶段 0 评出四个起点的 10 万和 5 万步；最晚 10-23 16:00 UTC | **未提交**：续训第一次实测（`pt-i4bzrj4k`）恢复后逐步完全一致，但"杀"没杀到训练进程；重测 `pt-uv1mh7fe` 18:12 提交、排队中。步数规则：四个起点 5 万→10 万的配对差都在噪声内（p > 0.05 且小于 3 个百分点）用 10 万步，否则 20 万；提交前把步数和依据报给你 |
| 2 | 先 DS-{B,S,RS,RD}-010-s1，再视卡情况 030 | 闸门（任一起点 10 万步检查点在 9 个空间任务上成功率 > 0）过了，且有名额 | 未开始；实验目录要先按阶段 1 的设置重建 |
| 3 | 1 小时实测（含一次满配置更新）→ 四个起点各 1 个 PPO；任务按规则选，所有起点同一批任务；RL 前后用同一套 950 回合评测；最多 2 个 RL 同时跑 | 1 小时实测排在阶段 1 之后、有名额时 | 未开始 |

- 名额：下游所有 8 卡训练（阶段 1、2 的训练，RL，测试）共用 4 个，RL 至多占 2 个；1 卡分片最多 8 个。8 卡评测任务不算训练名额。
- 种子 2、3 暂不跑，不再建它们的实验目录；已有的目录留着不删，看板不读它们。
- 统计：逐任务配对检验（34 个任务为配对单位）+ 评测噪声下限（同一检查点两次评测之差），见 [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md) 第 3、4 节。

### 1.2 评测的运行规则

- 主窗口 10-22 的条件（1 卡任务）：同时最多 8 个 1 卡任务，只投 h800 和 zoetrope；每次提交前查 aigc 工作区里有没有 `interiorgs-v10-*` 在排队，有就不提交；已经开始的不删；`interiorgs-v10-*` 和 `bq_tp_mmxbf16` 的任务一概不碰；建 `downstream_runs/PAUSE_SUBMISSIONS` 即暂停一切提交。
- 原来的填缝规则：1 卡分片只投到已有别的任务、还有空卡的机器上；落在空闲机器上的 2 分钟内删掉、30 分钟后重提。
- **你 10-22 16:30 的要求**（h800 上有好几台空机器，把排队和没提交的迁上去）之后：
  - 阶段 0 还没开始的检查点，每个改用一个 8 卡任务评（8 个策略服务 × 96 个仿真进程，评测计划、种子、台面区域顺序都不变），只投 h800，一个任务占满一台空机器，不把空机器切碎；已经开始分片的（DS-B 10 万步）继续用分片；
  - 每轮最多提交"估计的空闲机器数"个：h800 可用 29 台（共 30 台，1 台异常）减去本工作区有任务在跑的机器数；别的工作区的任务看不到，所以可能估多；我们有 8 卡评测还在排队时不再提交；
  - h800 上的 1 卡分片也可以落在空闲机器上（不再删）；
  - 主线 `interiorgs-v10-*` 排队、暂停文件、同卡数 `bq_tp_mmxbf16` 排队（只看同一个池）时照样不提交。
- 8 卡评测和阶段 1 任务内评测的设置相同，所以它的每回合时间也就是 96 个进程的实测。
## 2. 谁在跑

| sco 任务 | 内容 | 状态 | 目录 |
| --- | --- | --- | --- |
| `pt-rqtlcz9u` `pt-8agh5z4w` `pt-04ifqzpo` `pt-n2vw04v2` `pt-2b6ak1k6` `pt-yqry9u1x` `pt-d34si1tj` `pt-7gjl1h5y` | 阶段 0：DS-B 第 10 万步检查点的分片 1–8（各 1 卡，h800） | 分片 1、3、4、6、7 完成（各 2.88–2.98 小时）。**分片 2、5、8 卡住**：各差 1–2 个回合，17:49–17:58 后不再写结果。原因：某个仿真进程死了（没有报错，最可能是内存被杀），multiprocessing 的 `imap_unordered` 丢了它的回合、主进程一直等。18:43 删掉这 3 个任务（记录改名 `job_hung_*`），由看板重提；`closed_loop.py` 已修好（死掉的进程的回合重跑、卡住的进程先杀掉；重提时只跑缺的回合）。缺的回合：分片 2 StackBowlsCabinet 1013、1021；分片 5 SearingMeat 1044、LoadDishwasher 1002；分片 8 StoreLeftoversInBowl 1003 | `downstream_runs/experiments/DS-B-100-s1/evals_run_stopped_20261022/steps_100000/shards/` |
| `pt-gh0hdj6y` `pt-eauxo4si` `pt-qt5b1gox` | 阶段 0：S、RS、RD 第 10 万步检查点，各一个 8 卡评测（8 个策略服务 × 96 个仿真进程，h800 空闲机器） | 10-22 16:42 UTC 起运行 | `downstream_runs/experiments/DS-{S,RS,RD}-100-s1/evals_run_stopped_20261022/steps_100000/` |
| `pt-vakmljud` | 阶段 0：B 第 5 万步检查点，8 卡评测 | 10-22 16:42 提交，16:56 起运行 | `downstream_runs/experiments/DS-B-100-s1/evals_run_stopped_20261022/steps_50000/` |
| `pt-9mh97gl1` `pt-9hgf4p4h` | 阶段 0：S、RS 第 5 万步检查点，8 卡评测 | 16:58 提交后一直排队；17:53 主窗口因主线 7 个评测排队而暂停我们的提交，我 17:53:54 把这两个（没开始过）撤下，记录改名 `job_withdrawn_*`，等暂停解除后重提 | `downstream_runs/experiments/DS-{S,RS}-100-s1/evals_run_stopped_20261022/steps_50000/` |
| `pt-i4bzrj4k` | 续训测试（8 卡，h800）：A 不中断训 300 步；B 训练后"杀"、从全状态续训到 200（段末）、再续到 300；比对批次哈希、学习率、损失；然后 96 个进程的评测检查（计划的 1/10）和第四步的导出测试 | 10-22 16:36–18:11 UTC，退出码 0。全状态 109.8 GiB / 次，存 68 秒，读 243–270 秒，只留最新生效；B 从第 200 步恢复后第 201–300 步的损失与 A 完全相同、300 个批次哈希和学习率全同；**但"杀"只杀到启动器**（弹性启动器把 8 个训练进程放在各自的会话里），B 没停、训到了 300，所以"杀 → 从第 100 步恢复"和"训到段末退出"没测到。96 进程检查：95 回合 39 分钟，每回合是 24 进程估计的 2.62 倍。导出测试通过（导出 78 秒、16 GB；导出后 24 回合评测正常跑完） | `downstream_runs/checks/resume_test_20261022/`（`resume_check.json`、`export_test/export_test.json`） |
| `pt-uv1mh7fe` | 续训重测（8 卡，h800）：只重跑 B（`RESUME-B2`），按命令行杀掉 B 的全部进程并确认没剩，再续训两次；和第一次的 A 比 | 18:12 提交，排队中；排队期间若主线又有任务排队，会被撤下、之后重提 | `downstream_runs/checks/resume_test_20261022/RESUME-B2/` |
| `pt-wytuck3x` | RLinf 2 次更新试跑，第四次 | **通过**，退出码 0（10-22 10:03–10:19） | `downstream_runs/experiments/RLINF-SMOKE-1/retry3/` |
| 29 个 | 10-21 提交的第三步训练和 D5 评测 | 10-22 09:08 按你的要求撤下；停下的输出改名为 `run_stopped_20261022/`，撤下的任务记录改名为 `job_cancelled_20261022/`，都没删 | `downstream_runs/experiments/DS-*/` |
| `pt-cb0kvvf0` | 34 个评测任务的重置扫描（1 卡） | 已完成：原版 89 / 102 对可复现，固定顺序后 102 / 102；记录已删 | `downstream_runs/checks/reset_sweep_20261021/` |
| 更早的任务 | 第一、二步 | 已结束，记录已删（见第二步报告） | — |

**登录节点上本线的后台进程**：
- **看板** `downstream/eval/watch.py`：10-22 16:41 UTC 起，PID 1686863，代码 `fafa4060`，启动参数 `--stage0 --stage0-cards 8 --idle-ok-pools h800 --hold-until-running pt-ma8luzju pt-zhuu1j9h pt-p6g2le55`（三个是密集奖励第一批的训练任务，都已开始）。
  - 每 15 分钟一轮，每 2 分钟做一次分片落点检查。
  - 按第 1.2 节的规则从优先队列里提交阶段 0 的评测（8 卡任务投 h800 空闲机器，已开始的分片继续）；分段式（阶段 1）实验的检查点由训练任务自己评，看板只显示，最终检查点的第二次评测由它提交分片；阶段 2 按闸门和名额提交。
  - 每轮读一次配额剩余空间：剩余少于 1 TB 或读不出时，只报警、不提交。
  - 有新评测完成时重建结果报告：常规报告 `downstream_runs/report/`，阶段 0 报告 `downstream_runs/report_stage0/`。
  - 看板 `downstream_runs/BOARD.md`，日志 `downstream_runs/logs/watch.log`，状态 `downstream_runs/WATCH_STATE.json`。
  - 用 setsid + nohup 运行，ssh 断开后仍会继续。它除了删掉落在空闲机器上的分片任务，不删任何东西。
  - 启动、停止和查看（在集群 worktree 里；代码更新后要先 stop 再 start）：

```bash
bash downstream/eval/watch.sh start --stage0 --stage0-cards 8 --idle-ok-pools h800 --hold-until-running pt-ma8luzju pt-zhuu1j9h pt-p6g2le55
```

```bash
bash downstream/eval/watch.sh stop
```

```bash
bash downstream/eval/watch.sh status
```

## 3. 代码与位置

- **代码与文档**：分支 `downstream`（从 main `59053083` 开出）。只新增 `downstream/` 和 `docs/downstream/`。
- **同步方式（10-21 起）**：本地 worktree `C:\Users\user\Desktop\code\VAGEN-Lite-downstream` 提交 → push → 集群 `git pull --ff-only`。不再用 tar 覆盖。只在集群上才能生成的文件（如 `starts_lock.json`），在集群上提交、push，再在本地 pull。
- **集群 worktree**：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-downstream`。里面的 `third_party`、`playground` 是指向主检出同名目录的链接（都被 .gitignore 忽略），只读使用。
- **大文件**（实验产物）在 `/mnt/umm/users/yinbaiqiao/downstream_runs/`，不进 git：
  - `data/target50_{100,030,010}`：训练数据目录，只建链接；
  - `experiments/<编号>/`：每个实验；
  - `checks/`：一次性检查；
  - `logs/`：登录节点上的日志；
  - 看板和看板状态。
- **训练任务的建立和提交**：`downstream/train/runs.py make|submit|list`（10-21 建的第三步任务：20 万步，只存最后一个检查点）。分段式实验（experiment.json 里 `"segmented": true`）提交的是 `run_segmented.sh`。
- **阶段 1 的续训和分段式任务**（10-22）：`downstream/train/train_entry.py`（`trainer.full_state_interval` 存全状态、只留最新；`trainer.eval_points` 只在评测点存权重；`trainer.stop_at` 段末退出；启动时找到全状态就续训）、`full_state.py`（评测点、评测方式、1.5 倍规则）、`run_segmented.sh`（训 → 任务内评 → 续训）；建实验用 `make_experiment.py --full-state-interval 25000 --eval-points 50000,100000,150000 --archive-suffix _stopped_20261022`（复用 DS-*-100-s1 目录，旧的 experiment.json 等改名保留）。续训测试：`resume_test.sh`、`resume_check.py`。
- **第四步准备**（10-22）：`downstream/rlinf/fsdp_7b.overrides`、`site/sitecustomize.py`、`run_rlinf.sh`（正式启动脚本）、`export_test.sh`。
- **StarVLA**：`4507931` + 补丁 [downstream/patches/starvla_4507931_working.patch](../../downstream/patches/starvla_4507931_working.patch)（sha256 `b039782d…`，`git apply --reverse --check` 通过）。每个实验的 `experiment.json` 记提交号和补丁哈希。
- **评测**：[EVAL_PROTOCOL.md](EVAL_PROTOCOL.md)，计划文件 [downstream/eval/eval_d6.json](../../downstream/eval/eval_d6.json)，运行脚本 `downstream/eval/run_eval_ckpt.sh CKPT OUT 服务数 进程数[,进程数…]`。
- **剩余空间**：`python3 downstream/quota.py` 读 `sco afs dir-quota list` 整张表里我们这一行（`-d` 参数返回的记录不对，不用它）。
- **结果报告**（10-21 起）：`downstream/report/`。
  - 内容：
    - 收集训练损失和全部评测；
    - 配对统计（10-22 起）：主检验是逐任务配对（34 个任务为配对单位：每个任务一个成功率差，报平均差和对任务重抽样的 95% 区间、Wilcoxon 符号秩检验、符号检验），并和评测噪声下限（同一检查点两次评测之差）比；Wilson 区间和逐回合 McNemar 加 Holm 校正只作参考；
    - 阶段 0 页（`--stage0`）：四个起点成功率随检查点的曲线（全部、空间组分开）、空间任务里四起点平均成功率在 10%–60% 的、每个起点最后两个检查点之差；
    - 8 张图：训练损失、检查点曲线、样本效率曲线、逐任务热力图、逐任务配对差、结果一致性、成功所用步数、评测噪声；
    - 指标说明；
    - 按固定规则挑的范例。范例画面来自每个评测回合记下的 8 帧画面条（`frames/<任务>_s<种子>.jpg`，10-21 起记录）。
  - 生成：看板在有新评测完成时自动重建到 `downstream_runs/report/`，阶段 0 的到 `downstream_runs/report_stage0/`（各有 `index.html`、`results.json`、`figures/`）。手动生成用下面的命令（vagen-lite 环境，登录节点 CPU，几秒钟；阶段 0 加 `--stage0` 并换输出目录）：

```bash
/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python -m downstream.report.build --out /mnt/umm/users/yinbaiqiao/downstream_runs/report
```

  - 网页版（私有 Artifact，我在有新结果时重新发布）：https://claude.ai/artifact/Qa1oNasyCN5pSkjg2LXhBq
  - 版式预览用的模拟数据在 `downstream_runs/checks/report_sim_20261021/`（目录里有 `SIMULATED` 标记文件，页面带"模拟数据"横幅），与真实实验分开。
- **D7 新环境**：`/mnt/umm/users/yinbaiqiao/envs/rlinf-starvla-rc365`。它是叠在 starVLA 环境上的 venv，1.9 GB；包清单在 `downstream/envs/`，激活用 `source downstream/envs/activate_rlinf.sh <ENV>`。

## 4. 权重与数据路径

起点登记在 [downstream/starts.json](../../downstream/starts.json)，锁定值在 [downstream/starts_lock.json](../../downstream/starts_lock.json)。每次建实验和每个训练任务开头都校验一次，变了就停，不换路径。集群路径前缀 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/`。

| 名字 | 用途 | 路径 | 格式 / 精度 |
| --- | --- | --- | --- |
| B | 原始 Qwen2.5-VL-7B-Instruct（`cc594898`） | `interiorgs_v10_r11_20261016/inputs/base_7b` | 完整 HF，bf16 |
| S | 第十一轮动作 SFT 第 9 个 epoch | `interiorgs_v10_train_r11_20261016/lf_formal/model/checkpoint-1386` | 完整 HF，bf16 + fp32 混合 |
| RS | P-S0 第 300 次 | `interiorgs_v10_train_r11p_20261017/rl_run_ps0/checkpoints/global_step_300/actor/huggingface` | 完整 HF，fp32 |
| RD | P-D 第 150 次 | `interiorgs_v10_train_r11p_20261017/rl_run_pd/checkpoints/global_step_150/actor/huggingface` | 完整 HF，fp32 |
| R10S5（备用） | 第十轮 SFT 第 5 个 epoch | `interiorgs_v10_train_r10_20261015/lf_formal/model/checkpoint-770` | 完整 HF |
| R1（备用） | R1 最终模型 | `R1-clean-Projective-v0/formal/checkpoints/global_step_250/actor/huggingface` | 完整 HF |

- 训练时所有起点统一按 bf16 加载（D1）。
- 锁定内容：
  - 分片大小；
  - 索引和 config.json 的 sha256；
  - 10-21 起，还有每个分片的修改时间和首尾各 4 MiB 的 sha256。

| 数据 | 清单 | 规模 |
| --- | --- | --- |
| 第二步 | [downstream/demos/manifests/step2_smoke.json](../../downstream/demos/manifests/step2_smoke.json) | 3 个任务各前 100 条示教（训练）+ 后 20 条（留出），训练 44,369 帧 |
| 第三步 100% | [target50_100.json](../../downstream/demos/manifests/target50_100.json) | 50 个任务，25,307 条示教，14,957,899 帧 |
| 第三步 30% | [target50_030.json](../../downstream/demos/manifests/target50_030.json) | 每任务 150 条（RoboCasa 官方 `150_demos`），7,500 条，4,442,662 帧 |
| 第三步 10% | [target50_010.json](../../downstream/demos/manifests/target50_010.json) | 每任务 50 条（官方 `50_demos`），2,500 条，1,494,106 帧；10% ⊂ 30% ⊂ 100% |
| 原始示教 | `/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365/datasets/v1.0/target/` | 50 个任务齐全，58 GB |

## 5. 实验表

| 编号 | 步骤 / 阶段 | 起点 | 卡数 | 时间 | 状态 | 结果路径 |
| --- | --- | --- | --- | --- | --- | --- |
| STEP1-LOAD | 1 | B / S / RS / RD | 1 | 实测约 5 分钟 | 已完成，四个起点都通过 | `downstream_runs/step1_load_test_20261020/out/` |
| STEP2-RS-s1 | 2 | RS | 8 | 训练实测 434 秒（200 步） | **通过**（10-21 确认） | `downstream_runs/experiments/STEP2-RS-s1/` |
| RLINF-SMOKE-1 | 2（D7） | STEP2-RS-s1 的检查点 | 8 | 第四次 974 秒 | **第四次通过**（`pt-wytuck3x`，2 次更新） | `downstream_runs/experiments/RLINF-SMOKE-1/` |
| D5 停下的四个 | 3 / 阶段 0 | 四个起点 | 评测各 1 卡 × 8 片 | 训练到第 113,700–118,800 步时停下（10-22 09:08） | 16 个检查点按阶段 0 的顺序评测中 | `DS-*-100-s1/run_stopped_20261022/`，评测在 `DS-*-100-s1/evals_run_stopped_20261022/` |
| DS-{B,S,RS,RD}-100-s1 | 阶段 1 | 四个起点 | 各 8 | 20 万步约 42.0 小时（每步 0.75 秒） | 未提交：续训测试在跑；提交时机和步数按第 1.1 节的规则 | `DS-*-100-s1/run/`、`evals/` |
| DS-{B,S,RS,RD}-010-s1，再 030-s1 | 阶段 2 | 四个起点 | 各 8 | 未定 | 实验目录已建好（10-21），等阶段 1 的闸门 | `DS-*-{010,030}-s1/` |
| 每个起点 1 个 PPO | 阶段 3 | 四个起点 | 各 8 | 先实测 1 小时 | 未开始 | — |

## 6. 决定记录与待定

**10-21 你对第二步报告的决定**：
1. 固定台面区域顺序，作为本线所有评测的固定设置。已执行，见 EVAL_PROTOCOL.md。
2. 先提交 100% 数据、种子 2 和 3 的 8 个任务，其余按门槛提交。10-22 撤下；种子 2、3 改为暂不跑。
3. 评测规模改为 950 回合。已执行。
4. 同一检查点评两遍测评测噪声。10-22 改为阶段 1 的最终检查点评两遍（同一批种子）。
5. 加强起点校验。已执行。
6. 3 个边界任务单独列出。看板、报告和协议里已单列。
7. 第三步新任务只存最后一个检查点。阶段 1 改为每 2.5 万步评一次（见待定 2）。
8. D5 的中间检查点先留着；看板每次提交前检查剩余空间，少于 1 TB 只报警不提交。已执行（现剩余至少 7.5 TB）。

**补充要求**：P = 48、96 的并行度测试已不适用（1 卡分片每片只有 14 个 vCPU）；代码同步改用 git；看板断开后仍运行；RLinf 已单独报告，第四步方案初稿已写。

**10-22 你的要求**：
- 下游任务撤到只剩连通试跑需要的一两个，其余等密集奖励第一批上机器后再提。已执行。
- 评测拆成 1 卡，只启动在已经有占用的机器上。已实现（第 1.2 节）。
- 实验设计改为分阶段、先不比种子（第 1.1 节）。阶段 0 已开始；统计已改（报告代码 `c4c5669b`）。

- 你 10-22 的第二条指令（对我第 3 节理解的更正）：
  1. 续训：先实现、测通，再提交阶段 1。DeepSpeed 全状态检查点（bf16 权重、fp32 主权重、AdamW 两组状态、学习率调度位置、数据采样位置、随机数状态），每 2.5 万步存一次只留最新；实测存一次的大小；验证"训 N 步 → 杀 → 续 → 损失曲线和样本编号接得上"。已实现（`downstream/train/train_entry.py`、`full_state.py`），单元测试通过。8 卡实测 `pt-i4bzrj4k`：全状态 109.8 GiB，恢复后逐步完全一致，但杀没杀到训练进程；重测 `pt-uv1mh7fe` 排队中（第 2 节）。
  2. 任务内评测：评测在训练任务自己的分配里跑，分段式任务循环"训到评测点 → 退出 → 8 个策略服务 × 12 个仿真进程跑 950 回合 → 全状态续训"，评测点 5 / 10 / 15 / 20 万步；第一次评测记录 96 个进程的每回合平均时间，比 24 个进程慢超过 1.5 倍则后续只在最终检查点任务内评、中间交分片；阶段 0 和最终检查点的第二次评测仍走分片。已实现（`run_segmented.sh`；24 个进程的基准用第二步报告 3.3 节的实测：每次重置 21.6 秒 + 每控制步 0.196 秒，按同一批回合的步数折算）。
  3. 阶段 1 提交时机和步数：见第 1.1 节；规则已写进阶段 0 报告（"阶段 1 的步数规则"表）。
  4. 名额：见第 1.1 节；闸门改为"任一起点 10 万步在 9 个空间任务上成功率 > 0"，看板已改。
  5. 第四步准备：三个覆盖写进 `downstream/rlinf/fsdp_7b.overrides`；RLinf 的环境进程加台面区域顺序修正、分开演员和价值头的梯度范数（`downstream/rlinf/site/sitecustomize.py`，在登录节点上验证了主进程、spawn 和 fork 出来的子进程都打上了补丁）；导回 StarVLA 格式的测试（`downstream/rlinf/export_test.sh`）排在续训测试任务的最后；1 小时实测排在阶段 1 之后、有名额时。
  6. 种子 2、3 的目录留着，不删。
  7. 第一个分片跑完后报每个检查点的实际评测时长和阶段 0 全部 16 个的预计完成时间：已报（10-22 18:05）。每个检查点约 3 小时（8 个分片各 2.88–2.98 小时；8 卡任务估计约 3.2 小时）；全部 16 个预计 10-23 06:00–12:00 UTC。速度见 EVAL_PROTOCOL.md 第 1 节。
- 你 10-22 16:30：h800 上有好几台空机器，把排队的、还没提交的都迁上去。已执行（第 1.2 节）。
- 主窗口 10-22 17:53：主线 7 个评测排队，建了 `PAUSE_SUBMISSIONS`。已照办：撤下两个排队中的 8 卡评测；看板改为每轮最多提交 1 个 8 卡评测，主线有任务排队时自动撤下我们还没开始的 8 卡评测。主窗口同意没有 `interiorgs-v10-*` 排队时可以自行恢复。

**待定**：
1. 门槛（闸门）不过时会停下来等你决定。
2. 1.5 倍规则：96 个进程每回合时间是 24 个进程估计的 3.9–4.1 倍（阶段 0 的 8 卡评测）和 2.62 倍（续训测试里的检查），都超过 1.5 倍。按规则阶段 1 会在第一次任务内评测（5 万步）后改为只在最终检查点任务内评。我建议从一开始就这样，省掉 5 万步那次任务内评测（每个任务约 3 小时的 8 卡占用），等你决定。
