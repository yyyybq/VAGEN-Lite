# 下游具身任务线状态（唯一的状态文档）

最后更新：2026-10-22 15:20 UTC。实验设计按你 10-22 的指令改为分阶段（第 1.1 节），10-21 的 36 个任务的表作废；阶段 0 已开始。任务书分四步，每步做完停下来等确认。

## 1. 现在在哪一步

- **第一步（盘点与核实）**：已完成并经你确认。报告 [reports/STEP1_INVENTORY_20261020.md](reports/STEP1_INVENTORY_20261020.md)。
- **第二步（连通试跑）**：已完成，10-21 经你确认通过。报告 [reports/STEP2_SMOKE_20261020.md](reports/STEP2_SMOKE_20261020.md)；RLinf 两次更新的试跑 10-22 通过，报告 [reports/STEP2_RLINF_SMOKE_20261022.md](reports/STEP2_RLINF_SMOKE_20261022.md)。
- **第三步：按阶段进行，现在在阶段 0**（下节）。
- **第四步方案初稿**：[reports/STEP4_RL_PLAN_DRAFT_20261022.md](reports/STEP4_RL_PLAN_DRAFT_20261022.md)。

### 1.1 分阶段设计（你 10-22 的指令）

| 阶段 | 内容 | 开始条件 | 状态 |
| --- | --- | --- | --- |
| 0 | 1 卡分片填缝评测 D5 停下时保留的 16 个检查点（四个起点 × 2.5 万 / 5 万 / 7.5 万 / 10 万步）。顺序：四个起点的 10 万步 → 5 万步 → 2.5 万和 7.5 万。每评完一个起点的 10 万步更新 BOARD | 主窗口许可 | **进行中**：10-22 14:55 UTC 起 DS-B 第 10 万步的 7 个分片在 h800 上运行，第 8 个等有空卡的忙机器 |
| 1 | DS-{B,S,RS,RD}-100-s1 从头跑，带续训，任务内每 2.5 万步评一次，最终检查点用同一批种子评两遍；同时最多 4 个 | 触发点（密集奖励第一批上机器；r12k 的训练 10:07 都已开始，已满足） | **未提交**：步数按你看完阶段 0 后的决定（默认 20 万）；"续训"和"任务内评测"的具体做法待你确认（第 6 节） |
| 2 | 先 DS-{B,S,RS,RD}-010-s1，再视卡情况 030 | 阶段 1 的闸门（DS-B-100-s1 第 2.5 万步评测成功回合 > 0） | 未开始 |
| 3 | 1 小时实测（含一次满配置更新）→ 四个起点各 1 个 PPO；任务按规则选，所有起点同一批任务；RL 前后用同一套 950 回合评测；最多 2 个 RL 同时跑 | 阶段 2 之后 | 未开始 |

- 种子 2、3 暂不跑，不再建它们的实验目录。已有的目录（DS-*-100-s{2,3} 里 10-22 停下的输出，以及 10-21 建好的 DS-*-{030,010}-s{2,3}）没有删，看板不再读它们。
- 统计改为：逐任务配对检验（34 个任务为配对单位）+ 评测噪声下限（同一检查点两次评测之差），见 [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md) 第 3、4 节。

### 1.2 阶段 0 的运行规则（主窗口 10-22 的条件）

- 同时最多 8 个 1 卡任务，只投 h800 和 zoetrope，只投到已有别的任务、还有空卡的机器上（填缝）。
- 每次提交新分片前，查 aigc 工作区里有没有显示名以 `interiorgs-v10-` 开头的任务在排队（QUEUEING / STARTING / PENDING），有就不提交。16:20–19:00 UTC 主线密集奖励一批的 36 个 1 卡评测要上机器。
- 已经开始的分片不删。
- `interiorgs-v10-*` 和 `bq_tp_mmxbf16` 的任务一概不碰。
- 主窗口可能要求暂停：在登录节点建文件 `downstream_runs/PAUSE_SUBMISSIONS` 即可，看板见到它就不再提交任何任务，删掉文件后恢复。
- 我另加的一条（按你的卡的优先次序 bq_tp_mmxbf16 > 主线 > 下游）：`bq_tp_mmxbf16` 有同卡数的任务在排队时，不提交同卡数的任务（1 卡分片不让给 8 卡任务，因为 1 卡分片只投到已经有任务的机器，不会占掉一台整机）。
- 每个检查点 8 个 1 卡分片，每片约 119 回合、12 个仿真进程；分片开始后 2 分钟内再查一次落点，如果它所在的机器上只有我们的分片，立刻删掉、30 分钟后重提。局限：只看得到 aigc 工作区的任务。

## 2. 谁在跑

| sco 任务 | 内容 | 状态 | 目录 |
| --- | --- | --- | --- |
| `pt-rqtlcz9u` `pt-8agh5z4w` `pt-04ifqzpo` `pt-n2vw04v2` `pt-2b6ak1k6` `pt-yqry9u1x` `pt-d34si1tj` | 阶段 0：DS-B 第 10 万步检查点的分片 1–7（各 1 卡，h800） | 10-22 14:55 UTC 起运行 | `downstream_runs/experiments/DS-B-100-s1/evals_run_stopped_20261022/steps_100000/shards/` |
| `pt-wytuck3x` | RLinf 2 次更新试跑，第四次 | **通过**，退出码 0（10-22 10:03–10:19） | `downstream_runs/experiments/RLINF-SMOKE-1/retry3/` |
| 29 个 | 10-21 提交的第三步训练和 D5 评测 | 10-22 09:08 按你的要求撤下；停下的输出改名为 `run_stopped_20261022/`，撤下的任务记录改名为 `job_cancelled_20261022/`，都没删 | `downstream_runs/experiments/DS-*/` |
| `pt-cb0kvvf0` | 34 个评测任务的重置扫描（1 卡） | 已完成：原版 89 / 102 对可复现，固定顺序后 102 / 102；记录已删 | `downstream_runs/checks/reset_sweep_20261021/` |
| 更早的任务 | 第一、二步 | 已结束，记录已删（见第二步报告） | — |

**登录节点上本线的后台进程**：
- **看板** `downstream/eval/watch.py`：10-22 15:09 UTC 起，PID 1576483，代码 `c4c5669b`，启动参数 `--stage0 --hold-until-running pt-ma8luzju pt-zhuu1j9h pt-p6g2le55`（三个是密集奖励第一批的训练任务，都已开始）。
  - 每 15 分钟一轮，每 2 分钟做一次分片落点检查。
  - 按第 1.2 节的规则从优先队列里提交阶段 0 的分片。
  - 每轮读一次配额剩余空间：剩余少于 1 TB 或读不出时，只报警、不提交。
  - 有新评测完成时重建结果报告：常规报告 `downstream_runs/report/`，阶段 0 报告 `downstream_runs/report_stage0/`。
  - 看板 `downstream_runs/BOARD.md`，日志 `downstream_runs/logs/watch.log`，状态 `downstream_runs/WATCH_STATE.json`。
  - 用 setsid + nohup 运行，ssh 断开后仍会继续。它除了删掉落在空闲机器上的分片任务，不删任何东西。
  - 启动、停止和查看（在集群 worktree 里；代码更新后要先 stop 再 start）：

```bash
bash downstream/eval/watch.sh start --stage0 --hold-until-running pt-ma8luzju pt-zhuu1j9h pt-p6g2le55
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
- **训练任务的建立和提交**：`downstream/train/runs.py make|submit|list`（10-21 建的第三步任务：20 万步，只存最后一个检查点；阶段 1 的设置见第 6 节待定 1、2）。
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
| DS-{B,S,RS,RD}-100-s1 | 阶段 1 | 四个起点 | 各 8 | 20 万步约 42.0 小时（每步 0.75 秒） | 未提交（第 6 节待定 1、2） | `DS-*-100-s1/run/`、`evals/` |
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

**待定**：
1. **阶段 1 什么时候提交、跑多少步**。学习率按总步数余弦衰减，步数要在开始时定。(a) 现在按默认 20 万步提交（触发点已满足）；(b) 等你看完阶段 0 再定步数后提交。
2. **"其余按上一条指令不变"指的那条指令我没找到**。我目前的理解，请确认：
   - 续训：任务被打断（抢占、节点故障）后从最近的检查点接着训。StarVLA 的 `is_resume` 只恢复权重和步数，不恢复优化器状态，也不接着原来的数据顺序；要完全接上得另存优化器状态（按 AdamW 两份 fp32 状态估算每个检查点多约 56 GB，未核实），未实现；
   - 任务内评测：训练每 2.5 万步存一个检查点（每个 16 GB），看板用 1 卡分片评，规则同阶段 0；
   - 第四步准备：按方案初稿；
   - 并发上限：阶段 1 最多 4 个 8 卡任务，RL 最多 2 个，1 卡分片最多 8 个；阶段 2 的 010 是否要等阶段 1 的 4 个结束、还是和它们一起算在 4 个里，未定。
3. 门槛不过（成功回合 = 0）时会停下来等你决定。
