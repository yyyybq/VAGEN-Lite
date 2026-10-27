# 第一步：交接文件和资产核对（2026-10-27）

> 奖励设计线第一个任务 B（最优密集奖励）的第一步。只读核对，没有用 GPU；新建的只有本线的 worktree 和本文件、STATUS。
> 名词：**状态** = 相机位姿（位置 + 朝向），按 1 cm / 1° 合并成一个键；**粗筛** = `may_succeed`，"有题目物体落在视野锥里"，所有族成功的必要条件，不满足的状态不用判；**判定** = 用环境的成功规则（`release.env_success_oracle`）逐个判断一个状态是否成功；**CPU 小时** = 一个 CPU 核算一小时；**卡时** = 一张 GPU 占一小时。

## 结论

1. **交接文件列的资产全部在、都能读**（第 1 节，31 项）。交接文件本身不在集群上，在本机的主线 worktree `C:\Users\user\Desktop\code\VAGEN-Lite-dense\docs\REWARD_LINE_HANDOFF.md`（10-27 写，未进 git）。
2. **worktree 已建**：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-reward`，分支 `reward-design`，从 `qwen3-main` 的 `d530b546` 开出（含距离表代码和 Qwen3 支持）。交接文件写在合并之前，没写提交号；合并 14:21 UTC 已完成，所以取合并后的最新提交，待你确认（第 7 节第 2 项）。
3. **深度 12 的表要的算力是提示词估算的 2.5–3.3 倍。** 在登录节点实测 4 道题（只建图，判定按抽样计时，第 3 节）：
   - 每道 107–158 万个状态，是旧表（深度 8）中位数 8.1 万的 13–20 倍；不会碰到 200 万的上限。
   - 建图只要 10–17 秒、峰值内存 1.9 GB；时间几乎都花在判定上：每个状态 2.1–8.8 毫秒，每道题 0.4–1.9 CPU 小时。最慢的是投影关系（8.8 毫秒），它恰好是后备题里最多的族（30%）。
   - 按族加权：3,000 道约 **2,800–3,600 CPU 小时**（提示词估 1,100），一台 8 卡机 110 个进程约 **25–33 小时**（估 10 小时）。705 道评测题另加约 660–860 CPU 小时、6–8 小时。
   - 2a 自动接 2b 的门槛是外推 ≤ 1,650 CPU 小时，按这个估算过不了。所以门槛怎么定，现在就请你决定（第 7 节第 1 项）。
4. **磁盘和读表都不是问题**：每张表约 3.4 MB，3,000 道约 10 GB；环境每换一道题读一张表约 0.05 秒，占每次更新的时间不到 1%。
5. **q3s0 和 q3d 的有效配置只差奖励和筛选**（第 2 节），和提示词的假设一致。四个运行要改哪些键已列出。

## 1. 资产表

缩写同交接文件：`E` = `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial`，`R12` = `E/interiorgs_v10_r12_20261019`，`MR` = `/mnt/umm/users/yinbaiqiao/models_runs`，`Q3RL` = `MR/chains/m_qwen3vl8b_formal_rl_20261026`。核对时间 10-27 17:45–18:00 UTC。

| 项 | 路径 | ✓/✗ | 说明 |
| --- | --- | --- | --- |
| 交接文件 | 本机 `VAGEN-Lite-dense/docs/REWARD_LINE_HANDOFF.md` | ✓ | 13.7 KB，未进 git；集群上所有 VAGEN-Lite* 目录里都没有 |
| 代码基点 | `qwen3-main` `d530b546`（主线副本 `VAGEN-Lite-r12dense`，主仓库的 worktree） | ✓ | 含距离表（`cba41b22`）和 models 合并（`10d215ad`）；工作区干净 |
| Qwen3 起点 | `MR/chains/m_qwen3vl8b_formal_sft_20261026/lf_q3formal/model/checkpoint-1232` | ✓ | 32 GB，7 个分片，10-26 07:52；分片权限 600、属主 yinbaiqiao，登录节点和 sco 任务（root）都能读 |
| 原始模型 | `/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd0…` | ✓ | 目录能列；按交接文件，分片只有 root 能读，sco 任务里读得到 |
| SFT 数据（新格式） | `MR/sft_v10_newformat_20261024` | ✓ | plan 的 `sft_dir` |
| Qwen3 运行环境 | `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/.venv-qwen3vl-runtime` | ✓ | Python 3.12.13；训练链按 `model_type: qwen3_vl` 自动用它 |
| 建表用的 Python | `/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python` | ✓ | 3.12.13、numpy 2.2.4；启动脚本 distance-tables 模式默认用它 |
| RL 标准 | `data_gen/interiorgs_v10/configs/rl_standards.json`（`d530b546`） | ✓ | `v10_p1_S0`、`v10_p1_Dcond`、`chain_defaults` |
| Qwen3 两臂的 plan | `Q3RL/plan.json` | ✓ | 照抄的模板；它的 commit 是 models 分支的 `849ae78b` |
| q3s0 训练记录 | `Q3RL/rl_run_q3s0/rollout_data/` | ✓ | 150 次更新（156 MB），每次 192 条（24 题 × 8 条）。记的是回答文本，不含位姿，要按回答重放 |
| q3d 训练记录 | `Q3RL/rl_run_q3d/rollout_data/` | ✓ | 300 次更新（314 MB），格式同上 |
| 两臂的配置报告 | `Q3RL/rl_q3{s0,d}/rl_config_report.json` | ✓ | 第 2 节的差别从这里读出 |
| RL 训练题 | `E/interiorgs_v10_r11_20261016/rl_rows/rows.jsonl` | ✓ | 23,316 行，其中后备训练题 10,798 道（plan 的 `rows` + `task_pool: reserve` + 按深度核对排除） |
| 验证集 v2 | `R12/eval_val_v2/tasks.jsonl`、`selection.json` | ✓ | 413 道 |
| 评测集 v2 | `R12/eval_test_v2/tasks.jsonl`、`selection.json` | ✓ | 292 道 |
| 扩展窗口的评测目录 | `MR/eval_sets/eval_{val,test}_v2/` | ✓ | `tasks.jsonl`、`selection.json` 是指向上两项的链接；本线照这个样子建 `reward_runs/eval_sets/` |
| 旧距离表（后备题） | `R12/analysis/r13_distance/reserve/` | ✓ | 10,798 张，`summary.jsonl` 10,798 行，1.3 GB；起点距离等于 d\* 的 10,798 / 10,798；目录属主 root（只读，正好） |
| 旧距离表（评测题） | `R12/analysis/r13_distance/eval/` | ✓ | 705 张，79 MB |
| 表外位姿抽查 | `R12/analysis/r13_distance/outside_probe.json` | ✓ | 150 个表外位姿重新搜索的结果，按 d\* 分组汇总 |
| 反向扩展表 | `R12/analysis/r13_distance/reserve_goal/` | ✓ | 10,798 张，12 GB |
| 反向扩展抽查 | `R12/analysis/r13_distance/goal_side_probe.json` | ✓ | 有逐题的位姿列表 |
| 回放和诊断脚本 | `R12/scripts/` 下 `dcond_replay_r12.py`、`reward_replay_r13.py`、`reward_replay_r13o.py`、`progress_proxy_a2_r13.py`、`dense_signal_a1_r13.py`、`scenario_compare_r13.py`、`outside_table_probe_r13.py`、`build_goal_tables_r13.py`、`goal_side_probe_r13.py`、`start_chain_dr2.sh` | ✓ | 10 个都在（不在 git 里）。要用时复制到本线的 `reward/` 再改路径，不在原地改 |
| 对比配置样例 | `R12/analysis/r13_compare/cfg/` | ✓ | 11 个 |
| O 下的重放（Qwen2.5） | `R12/analysis/r13_replay/o_p1d/` | ✓ | 主线 O-keep 门槛 1.2754 的依据 |
| keep 门槛的口径 | `cluster_runs/KEEP_THRESHOLD_R12_20261021.md`（集群副本里有）；`KEEP_THRESHOLD_DCOND_R12.md`（只在本机主线 worktree，未进 git） | ✓ | 全失败题 8 条轨迹回报的样本标准差，取第 80 百分位，下限 1.0 |
| 信号诊断 | `cluster_runs/DENSE_SIGNAL_R13.md` | ✓ | 只在本机主线 worktree（未进 git），集群副本里没有 |
| B 的撤回说明 | `E/interiorgs_v10_train_r13b_20261026/HOLD.md` | ✓ | 10-27 补了一句：按用户第 77 条，Qwen2.5 不做 B，表交给本线 |
| 下游的暂停开关 | `/mnt/umm/users/yinbaiqiao/downstream_runs/PAUSE_SUBMISSIONS` | — | 现在没有这个文件（不暂停） |
| 配额 | `sco afs dir-quota list -i 019ec9f9-…` 的 `/users/yinbaiqiao` 行 | ✓ | 22 TB 里用了 17 TB（主线 19 TB 报警） |
| 本线 worktree | `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-reward` | ✓（新建） | 分支 `reward-design` @ `d530b546`；`verl` 子模块 `a0ae85c`，从主 checkout 的 `verl` 本地克隆 |
| 本线输出目录 | `/mnt/umm/users/yinbaiqiao/reward_runs/` | 未建 | 第二步建 |

## 2. q3s0 和 q3d 的差别，四个运行要改的键

**q3s0 和 q3d**（两份 `rl_config_report.json` 逐项比较，只有这几项不同）：

| 项 | q3s0 | q3d |
| --- | --- | --- |
| 标准 / 奖励 | `v10_p1_S0` / S0 | `v10_p1_Dcond` / D_cond |
| 进度奖励 | 无 | 势能差，gamma 0.99，`progress_version: grouped_gated`，系数 10 ÷ 3 = 3.333（`design_variant`） |
| 筛选 | `mixed_outcome`（只要有成有败的题，凑 12 组） | `keep_by_return_std`（门槛 0.3318、下限 1.0，凑 12 组） |
| 池 | h800 | zoetrope |

其余全部相同：
- 数据和采样种子 20261019 / 0；
- GRPO、每题 8 条、24 道候选；
- 学习率 2e-6、KL 系数 0.05、熵系数 0.01、读不出扣 −0.1；
- 历史拼接、格式示例 `action_names`、10,798 道后备题。

**四个运行**：

整份 plan 照抄 `Q3RL/plan.json`，计划级改这些：
- `tr`、`rl_name`；
- `job_prefix: rw`（任务名就成了 `rw-…`）；
- `eval_sets` 指向本线的 `reward_runs/eval_sets/eval_{val,test}_v2`；
- `rl.rows` 换成这 3,000 道的子集文件：做法同主线 C1，`rows` 指向子集，`task_pool` 和 `exclude` 不变；
- `run_pools`。

guards 和 eval 设置不变。各运行在 `rl_runs` 里写：

| 运行 | `rl_runs.<名字>.rl` |
| --- | --- |
| rw_s0 | 不写（= q3s0） |
| rw_g | `standard: v10_p1_Dcond`、`group_filter: keep_by_return_std`、`keep_std_threshold: 0.3318`、`keep_std_floor: 1.0`、`progress_reward_scale: 3.333…`（= q3d） |
| rw_o | `standard: v10_p1_Dcond`、`progress_version: optimal`、`distance_tables: reward_runs/tables/b3000`、`progress_reward_scale: 3.333…` |
| rw_ok | rw_o 的键 + `group_filter: keep_by_return_std`、`keep_std_threshold:`（第三步标定）、`keep_std_floor: 1.0` |

- 主线 10-26 撤回的 B（`opt` / `optk`）就是这么写的，只是没写 `progress_reward_scale`，用的是 D_cond 的原系数 10。
- `job_prefix` 在 `d530b546` 里已经有了：`job_sco.submit` 读 `VAGEN_LITE_JOB_PREFIX`，不写时默认 `interiorgs-v10`。
- q3 的 plan 没写 `retain`，检查点全留。本线留哪些见第 7 节第 5 项。

## 3. 深度 12 的表：大小、上限、耗时、内存（实测）

**做法**：
- 在登录节点（14 核）用本线 worktree 的代码（`reward/step1/graph12_probe.py`），每族挑一道后备训练题，按 (族, d\*) 取最接近的一道。
- 用主线的 `reachable_graph` 建深度 12 的状态图，上限放到 6,000 万，只为数状态。
- 粗筛全部状态；判定只抽 1,500 个通过粗筛的状态计时，再外推到全部通过粗筛的状态。
- 不写任何文件。

| 题 | 族 | d\* | 状态数 | 第 9 / 10 / 11 / 12 层的增长倍数 | 到第 11 层累计 | 粗筛通过 | 判定（毫秒 / 个） | 判定全部（CPU 小时） | 建图（秒） | 峰值内存 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v10_018fbaac… | 投影关系 | 1 | 1,584,140 | 2.63 / 1.93 / 1.41 / 1.16 | 1,049,524 | 48% | 8.77 | 1.85 | 14.8 | 1.87 GB |
| v10_0cb524b7… | 等距 | 3 | 1,383,304 | 2.57 / 1.88 / 1.39 / 1.16 | 922,510 | 72% | 2.16 | 0.60 | 14.2 | 1.87 GB |
| v10_2b69d023… | 遮挡对齐 | 5 | 1,067,849 | 2.47 / 1.75 / 1.23 / 1.02 | 762,560 | 76% | 2.15 | 0.49 | 16.6 | 1.88 GB |
| v10_00224170… | 绝对定位 | 7 | 1,197,562 | 2.52 / 1.86 / 1.38 / 1.18 | 794,686 | 56% | 2.07 | 0.39 | 10.1 | 1.88 GB |

- **增长在第 8 层以后很快变慢**：第 8 层前每层约 3.1–3.4 倍（主线深度 8 的探针也是这样），第 12 层只有 1.0–1.2 倍。这是因为 12 步以内能到的地方是有限的一片，不同走法又会落到同一个位姿。所以按"每层 3.3 倍"外推（会得到 400–800 万个状态）高估了，以实测为准。
- **上限**：`max_states`（默认 200 万）只在扩下一层之前检查。这 4 道到第 11 层累计 76–105 万，第 12 层照常扩，最后 107–158 万，都没碰到上限。更空旷的房间可能更大，所以本线的建表文件会把上限调到 600 万，并记录每道题离上限多远。
- **CPU 小时的外推**：
  - 后备题各族占比（q3d 的配置报告）：投影关系 30.3%、绝对定位 21.2%、占画面 19.5%、等距 14.4%、视野包含 13.1%、遮挡对齐 1.5%。
  - 视野包含和占画面没有实测，按等距的 0.6 CPU 小时算。加权平均每道约 0.93 CPU 小时，3,000 道约 2,800 CPU 小时。
  - 整机满载时单个判定可能更慢：主线整机建表里深度 8 的中位数是 3.44 毫秒，高于这里 4 道中 3 道的约 2.1 毫秒。所以上限多留 1.3 倍的余量；这个倍数是余量，不是测出来的。给区间 **2,800–3,600 CPU 小时**，一台 8 卡机约 **25–33 小时**。
  - 705 道评测题按同样比例约 660–860 CPU 小时、6–8 小时。
  - 每族只有一道实测，投影关系又占了一大半时间，所以误差不小，2a 的 10 道会覆盖全部六族再算一遍。
- **和估算差在哪**：提示词的 1,100 CPU 小时，和主线 STATUS 第十三轮待决第 1 项"扩出的位姿也逐个判成功，约 3,900 CPU 小时"按题数折算的结果一致（3,900 × 3,000 ÷ 10,798 ≈ 1,080）。那个数对应的是反向扩展出的状态，中位数每道 69 万个；深度 12 的正向图每道 107–158 万个，是它的 1.6–2.3 倍。投影关系每个状态 8.8 毫秒，又比平均慢很多。
- **内存**：每个进程峰值 1.9 GB，110 个进程约 210 GB。2a 在登录节点上跑（cgroup 内存上限 240 GiB），开不到 12 个进程；2b 在 8 卡任务里先读 cgroup 的内存上限再定进程数。
- **磁盘**：旧表每个状态 2.6 字节（压缩后），深度 12 的表每张约 3.4 MB，3,000 道约 10 GB，705 道约 2.4 GB。

## 4. 建表命令

- 主线的命令行：`python -m data_gen.interiorgs_v10.distance_table build --rows ROWS --gs-root /mnt/umm/users/yinbaiqiao/InteriorGS_ply --out DIR --task-ids FILE --depth 12 --workers N [--check-prefilter N]`。
  - 不写 `--depth-offset`，深度就固定是 12。
  - 已有的表会跳过，所以能续跑。
  - 每道题的统计写进 `DIR/summary.jsonl`。
- 集群上由启动脚本的 distance-tables 模式包一层：`sco_interiorgs_v10_train.sh distance-tables JOBDIR ARGS`，它会补上 `--gs-root`。主线的 8 卡任务就是这么跑的：
  - 规格 `N4lS.Iq.I80.8`，池 h800，`--workers 110`；
  - 任务名 `interiorgs-v10-distance-tables-job-reserve`，记录在 `R12/analysis/r13_distance/job_reserve/job/`。
- 命令行没有 `max_states` 参数。本线写 `reward/distance12.py`，调主线的 `build(..., max_states=6_000_000)`，表的格式和 `summary.jsonl` 都不变，另外记每层的状态数和进程峰值内存。主线代码不改。

## 5. 环境侧读表的开销

- `TableCache` 在换题时读一张表。实测读 25 张反向扩展表（共 1,764 万个状态、27.7 MB）用 0.59 秒，即每百万个状态 0.034 秒。
- 一张深度 12 的表（约 130 万个状态）读一次约 0.05 秒，占内存约 16 MB（每个状态 12 字节）。
- 每次更新 192 局，读表合计不到 10 秒的 CPU，而且分散在各个环境进程里。q3d 每次更新 67 秒，影响不到 1%。2b 第 4 项和第三步的前 2 次更新会再实测。

## 6. 交接文件和提示词对不上的地方

1. **代码基点**：交接文件说 Qwen3 能用的代码要等合并成 `qwen3-main`，没写提交号；合并已完成，本线取 `d530b546`（第 7 节第 2 项）。
2. **KL 停止线**：交接文件写的用户标准（第 78 条）是 KL 1.0 线；q3d 的 plan 是 `kl_max` 0.2；提示词要求"同 q3d"。q3d 实际 KL 最高 0.027，两条线都不起作用（第 7 节第 3 项）。
3. **建表的算力**：提示词估 1,100 CPU 小时、10 小时，实测外推 2,800–3,600 CPU 小时、25–33 小时（第 3 节，第 7 节第 1 项）。
4. **建表要的题目文件和场景目录交接文件没列**：都从交接文件给的路径里找到了，没有从别的文档猜。
   - 题目文件是 q3 plan 的 `rl.rows`；
   - 场景目录是 `gs_root` = `/mnt/umm/users/yinbaiqiao/InteriorGS_ply`，主线建表任务的 `launch.sh` 也是这个。
5. **编号**：交接文件建议本线用 `RW-<编号>`，本线定为实验 `RW-B-S0`、`RW-B-G`、`RW-B-O`、`RW-B-OK`。运行名和结果名用 `rw_<臂>`、`rw_<臂>_<更新次数>`，和提示词一致。

## 7. 需要你决定的事

1. **距离表的算力**（最重要）
   - **甲（推荐）**：做法不变，仍是 3,000 道、深度 12、全部精确判定，只把 2a 自动接 2b 的门槛从 1,650 改成"外推 ≤ 4,000 CPU 小时"，其余三条（核对全过、内存放得下、磁盘 ≤ 1 TB）不变。
     - 一个 8 卡任务先建 3,000 道（约 25–33 小时），建完就交第二步的报告、等你批准第三步；
     - 705 道评测题在同一个任务里接着建（约 6–8 小时）。它们第四步才用，不耽误提交 RL。
     - 代价：一台 8 卡机占约 31–41 小时，GPU 空着约 250–330 卡时。作为对比，四个 RL 运行合计约 230 卡时。
   - **乙**：同甲，但分成两个 8 卡任务，按题号哈希各算一半。墙钟减半（3,000 道约 13–17 小时），总卡时不变；要同时有两台空机器。
   - **丙**：减到 1,500 道，约 1,400–1,800 CPU 小时、13–17 小时。代价：300 次更新共抽 7,200 次题，每道平均被抽 4.8 次（3,000 道是 2.4 次，主线 10,798 道是 0.7 次），离主线的设定更远。
   - **丁**：先写更快的判定（按族批量计算，和现在的判定逐个核对到完全一致），再建表。可能省一大半时间，但要先花半天到一天写和核对，有判错的风险。
2. **代码基点**
   - **甲（推荐）**：用 `qwen3-main` 的 `d530b546`，worktree 已按这个建好；主线以后修了 bug，再合进来。
   - **乙**：你指定别的提交，我重开 worktree。
3. **KL 停止线**
   - **甲（推荐）**：照提示词，同 q3d 的 `kl_max` 0.2。和 q3d 完全一致便于对照，q3d 实际也碰不到这条线。
   - **乙**：用第 78 条的标准 1.0。
4. **任务名和 GPU 次序**
   - **甲（推荐）**：`job_prefix: rw`，正常排队。下游的看管程序只给 `interiorgs-v10-*` 让路，不会给本线让路。四个 RL 运行合计约 230 卡时，排队就够。
   - **乙**：请下游窗口也给 `rw-` 让路，本线就排在下游前面。
5. **检查点留哪些**（第三步用。这涉及删除，所以先问）
   - **甲（推荐）**：评测完按"留最好和最后"清理。每个运行留最好的 HF 权重（31 GB）和最后的完整状态（131 GB）。峰值约 4 × 290 GB ≈ 1.15 TB，配额会用到约 18.2 TB；清理后剩约 0.65 TB。
   - **乙**：全留，一直占 1.15 TB，配额约 18.2 TB，离主线 19 TB 的报警线只剩约 0.8 TB。

## 依据

- 资产核对：在集群登录节点上用 `stat`、`ls` 和 `wc` 逐项检查；第 2 节比较的是两份 `rl_config_report.json`。
- 第 3 节：`reward/step1/graph12_probe.py`（4 道题深度 12 的状态图），`reward/step1/table_stats.py`（旧表按深度的统计、主线探针的每层增长、每个状态的字节数和读表耗时）。两个都只读，结果抄在本文。
- 主线建表任务的命令和耗时：`R12/analysis/r13_distance/job_reserve/`。
  - 10,798 道，`--workers 110`，计算 14,175 秒；
  - 各题耗时合计 406.6 CPU 小时；
  - 深度 8 的 6,420 道占其中 385.6 小时。
