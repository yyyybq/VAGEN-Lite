# vpd：视角变化下游线（viewpoint downstream）

这条线要检验的是：用动作 SFT/RL 训过的 Qwen3-VL-8B 起点，做视角变化类的下游任务时，是不是比用 QA 训的起点更容易靠微调学会。

现在的阶段是三个闸门：
- WalkerBench 试点；
- TVR 渲染冒烟；
- E3VS 试点。

闸门只看三件事：能不能学会、格式有没有效、有没有饱和。不比较起点之间谁高。

科学方案在报告 v2：`eval_external_assets/downstream_search_20261108/REPORT_downstream_search_v2.md`。实施计划和约束性决定在 `eval_external_assets/vpd/plan/`（DECISIONS.md 优先于 plan.md）。WalkerBench 的事先登记见本目录的 `PREREG_WB_PILOT.md`，E3VS 的见 `PREREG_E3VS_PILOT.md`。

## 1. 位置

- **代码 worktree**：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd`，分支 `viewpoint-downstream`，基线提交是 `bf4b2553`。
  - 新代码只放在 `vpd/` 下。
  - `downstream/` 和 `eval_external/` 只读导入，不改。要改的文件就拷一份到 `vpd/` 里改，文件头写明来源路径和提交号 `bf4b2553`。
- **输出目录**：`/mnt/umm/users/yinbaiqiao/eval_external_assets/vpd/`，下文记作 OUT。
- **运行方式**：每个 vpd 进程的 PYTHONPATH 都设成 worktree 根目录，工作目录也是 worktree 根目录。
  - 不要从 VAGEN-Lite-downstream 或 VAGEN-Lite-eval 导入任何东西：那是别的窗口正在改的 worktree。

## 2. vpd/ 目录

| 路径 | 内容 | 负责的工作包 |
|---|---|---|
| `README.md` | 本文件 | WP-PREREG |
| `PREREG_WB_PILOT.md` | WalkerBench 试点的事先登记。用户确认后提交，提交以后冻结 | WP-PREREG |
| `PREREG_E3VS_PILOT.md` | E3VS 试点的事先登记 | WP-E3VS-A |
| `__init__.py`、`paths.py` | OUT 的目录常量、各解释器路径 | WP-CORE |
| `starts.json`、`starts.py` | 起点表（名字 → 评测 view 代码、checkpoint 路径），加载时断言注意力实现 | WP-CORE |
| `sco.py`、`ledger.py` | 包装 `downstream/sco_job.py` 的 sco 工具，带作业台账和按闸门预留的预算检查 | WP-OPS |
| `jobs/common_env.sh`、`jobs/preflight.sh` | 作业脚本的公共开头（线程数、离线开关、chown 陷阱、节点信息）和开跑前检查 | WP-OPS |
| `jobs/wb_demos.sh` | sco 作业脚本：渲染示范、生成热身集、生成档位 | WP-WBDEMO |
| `jobs/wb_smoke.sh`、`jobs/wb_pilot_start.sh` | sco 作业脚本：WalkerBench smoke 和每个起点的试点 | WP-WBPILOT |
| `jobs/e3vs_prep.sh`、`jobs/e3vs_start.sh` | sco 作业脚本：E3VS 准备和每个起点的试点 | WP-E3VS-B |
| `walkerbench/sandbox.py` | 唯一的 WalkerBench 运行副本 helper：改渲染尺寸、全景目录、任务目录 | WP-CORE |
| `walkerbench/history.py` | 干净的 K 轮历史窗口，训练和评测共用 | WP-CORE |
| `walkerbench/action_regex.py` | 约束解码用的每步正则 | WP-CORE |
| `walkerbench/data/split_city.json` | 冻结的城市切分源清单（sha256 已登记） | WP-WBDATA |
| `walkerbench/split.py`、`tiers.py`、`baselines.py` | 生成 S2 切分；生成嵌套档位顺序；在 179 题上跑 STOP0、FWD、ODO 和专家 | WP-WBDATA |
| `walkerbench/demos.py`、`warmup.py` | RM 专家示范（640x400 真实渲染）；20 段格式热身 | WP-WBDEMO |
| `walkerbench/run_vpd.py`、`smoke_vpd.py` | 评测副本（约束解码、渲染尺寸、K 窗口、合并模型）；假服务器冒烟 | WP-WBEVAL |
| `walkerbench/pilot.py`、`pilot_gate.py` | 试点编排（plan / prepare / submit / status / collect / delete / budget）；闸门判定 | WP-WBPILOT |
| `common/vllm_server_vpd.py` | vLLM 服务副本：可指定模型目录、结构化输出、图像像素上限。WalkerBench 和 E3VS 共用 | WP-WBEVAL |
| `train/n2_train_vpd.py` | 训练脚本，由 `downstream/n/n2_train.py` 改来 | WP-TRAIN |
| `train/parity_check.py`、`train/merge.py`、`train/wb_train_probe.py` | 两个环境的处理器一致性检查；合并到作业本地盘；长序列吞吐探针 | WP-TRAIN |
| `tvr/` | TVR 冒烟：登录节点准备脚本、冒烟作业、Vulkan/EGL 探针、判定脚本、`SMOKE_CRITERIA.md` | WP-TVR |
| `e3vs/` | E3VS 相关代码（几何、BFS 示范、盲下限、下载脚本、`USER_STEPS.md`、渲染、占据栅格、SFT 数据、闭环评测、汇总） | WP-E3VS-A、WP-E3VS-B |
| `tests/` | 每个工作包的 pytest。在登录节点上单核、不用 GPU 就能跑 | 各工作包 |

## 3. OUT 目录

| 路径 | 内容 |
|---|---|
| `plan/` | 计划、决定、审查、调查记录，以及从登录节点 /tmp 拷来的原型（`plan/login_tmp/`） |
| `scratch/<标签>/` | 各工作包自己的临时目录。只能删自己建的东西，绝不用 /tmp |
| `ledger.jsonl` | 每个 sco 作业一行：名字、pt 编号、开始和结束时间、卡时 |
| `jobs/<作业名>-<yyyymmddHHMM>/` | 每个 sco 作业的 RUN 目录（作业文件、worker.log、node.txt、prepared.json、steps.jsonl） |
| `walkerbench/split/` | `split_s2.json`、`expert_splice.json`，以及 `s2test/`、`s2val/`、`s2train/` 三个部分，每个部分下有 `tasks/` 和 `selection.json` |
| `walkerbench/baselines/` | `baselines_s2test.json`（逐题结果、按类型的下限、合并下限 F）、`hard_subsets.json` |
| `walkerbench/demos/s2train_640x400/` | `demos_all.jsonl`、`images/<stem>/turn_XX.jpg`、`manifest.json`、`DONE` |
| `walkerbench/warmup/` | `warmup20.jsonl`、`images/`、`manifest.json` |
| `walkerbench/tiers/` | 每个种子一份 `order_seed<种子>.json`，以及 `seed<种子>/demos_<N>.jsonl`；种子有 1001（试点）和 1、2、3（确认轮） |
| `walkerbench/runs/smoke/` | smoke 作业的输出和 `smoke_verdict.json` |
| `walkerbench/runs/pilot/<起点>/` | 适配器 `adapters/{0W,50,200}/` 和评测结果 `evals/pilot__<起点>__<档>__<分辨率>__<解码>/` |
| `walkerbench/runs/pilot/` | `GATE.json` 和 `GATE.md` 也放这里 |
| `tvr/` | 图形库副本（nvgl_595.71.05）、AI2-THOR 的 HOME、TVRBench 仓库、冒烟输出 |
| `envs/tvr` | TVR 的 conda 环境（要用户批准后才建） |
| `e3vs/` | `repo/`、`data/`、`occ/`、`bfs/`、`sft/`、`parity/`、`runs/` |

合并后的权重**不进** OUT，也不进 /mnt/umm 的任何地方：
- 只写作业本地盘：`/tmp` 剩余 ≥ 40 GB 时用 `/tmp`，否则 `/dev/shm` 剩余 ≥ 40 GB 时用 `/dev/shm`，都不够就报错。
- 评完就删。
- 原因：Q3B 的权重只有 root 能读，合并出来的副本放到共享盘上就变成了普通用户可读的副本。

## 4. 解释器

| 记号 | 路径 | 用途 |
|---|---|---|
| PY_EVAL | `/mnt/umm/users/yinbaiqiao/.conda/envs/eval-external/bin/python` | WalkerBench 环境和渲染、CPU 工具、登录节点上的测试 |
| PY_Q3 | `/mnt/umm/users/yinbaiqiao/eval_external_assets/envs/eval-external-q3/bin/python` | vLLM 0.11.0 + transformers 4.57.6。评测、合并，也是训练的备选环境 |
| PY_TRAIN | `/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python` | torch 2.6 + transformers 4.57.0 + flash_attn。训练，和 N2 一样 |
| SCO | `/mnt/umm/users/yinbaiqiao/.sco/bin/sco` | sco 命令行 |

另外几条：
- 运行读取外部下载文件的脚本时，Python 加 `-I`。
- Windows 本机没有 Python。所有分析都在集群上跑。

## 5. 登录节点规则

规则来自用户的 CLAUDE.md 和 DECISIONS §A。

1. **CPU 配额**：整个账号只有 14 个核，所有窗口共用。
   - 本线的检查每个进程只用 1 个核；单个任务最多 4 个核。
   - 开跑前看一下现在用了几个核：读 `/sys/fs/cgroup/cpu.stat` 里的 usage_usec，隔 5 秒再读一次，两次的差除以 5×10⁶，就是在用的核数。
   - 已经超过 8 个核就等一会儿再跑。任何时候"已用 + 本任务"都不能超过 12 个核。
2. **线程数**：
   - OMP_NUM_THREADS、MKL_NUM_THREADS、OPENBLAS_NUM_THREADS、NUMEXPR_NUM_THREADS 都设成 1，TOKENIZERS_PARALLELISM 设成 false。
   - 导入了 torch 的进程，还要调一次 torch.set_num_threads(1)。
   - 进程池和 worker 数一律显式写数字，不用默认值，也不用 os.cpu_count()。
   - hf 下载加 --max-workers 1。
3. **超过 1 分钟的任务**前面加 nice -n 19。4 个核不够、或者要跑很久的，改成 sco 作业。
4. **登录节点上不用 GPU**，也不加载 8B 模型。
5. **不往 /tmp 放脚本或工作目录**：同名文件会盖掉标准库，而且 /tmp 里的文件可能是别人的。用 `OUT/scratch/<标签>/`。
6. **SSH**：一律直连，并带上 BatchMode=yes、ControlMaster=no、ControlPath=none、ClearAllForwardings=yes 这几个选项。
   - 拷代码：本机打 tar 包，通过 ssh 在 worktree 里解开。
   - 跑脚本：先在本机写好，再从 stdin 传过去。不要写引号很多的 heredoc。
7. **git**：只读命令（show、log、diff）随便用。commit、push、pull、checkout、stash 一律不做，由集成者在用户同意后提交。
8. **不碰正在跑的作业读的文件**，也不改别的 worktree：VAGEN-Lite-downstream、VAGEN-Lite-eval、主线 checkout 等。
9. **必须先停下来问用户的事**（DECISIONS §A3）。代码可以写好，但不执行：
   - 从集群外拷东西、下载大文件、调用付费接口：包括 pip / conda 安装和建环境、下载 AI2-THOR 构建、克隆或下载 TVRBench、E3VS 或任何仓库和数据集、从 HF 下载；
   - 删除文件：自己这次在 scratch 里建的除外；
   - 转换或导出权重产生新副本：包括在真实权重上跑 merge.py；
   - 提交和删除 sco 作业：由编排在用户同意后做。
10. **文档和报告**：用中文写，结论先说，每个数带限定；没核实的标"未核实"，推断标"推断"；时间用 date -u。代码里的注释和 docstring 用英文。

## 6. sco 作业规则

1. **GPU 任务一律用 sco 提交。** 每个作业 1 张 H800，规格 N4lS.Iq.I80.1（14 vCPU、240 GB 内存），镜像 wc-dev:260617。
   - 默认投 zoetrope；zoetrope 的单卡作业开始排队时，可以改投 h800，哪个排队短投哪个。
   - 只通过 `vpd/sco.py` 提交。
2. **命名**：显示名一律是 `downstream-vpd-<作业>`。例如：
   - downstream-vpd-tvr-smoke
   - downstream-vpd-wb-demos
   - downstream-vpd-wb-smoke
   - downstream-vpd-wb-q3b、downstream-vpd-wb-q3s、downstream-vpd-wb-e2
   - downstream-vpd-e3vs-prep、downstream-vpd-e3vs-q3b

   "downstream-" 后面**绝不能以 ev 开头**：下游线正在跑的看守进程会把 downstream-ev* 当成它自己的作业。也不碰 interiorgs-v10-*、bq_tp_mmxbf16、bq-ready-* 这些作业。
3. **并发**：本线同时最多 4 个单卡作业在跑。
4. **以 root 跑**：每个 Qwen3 作业（训练、合并、评测）都以 root 跑，因为 Q3B 的权重只有 root 能读。
   - 作业退出时，把 RUN 目录和自己写过的输出目录 chown 回 20325。
   - 作业脚本不能放在 `RUN/job/launch.sh`，避免自己执行自己。
5. **作业里的线程数**：
   - vLLM 进程 OMP = 4，其他进程 OMP = 2；
   - 训练的 DataLoader 用 3 个 worker，每个 1 线程；
   - WalkerBench 评测用 8 个 worker；示范渲染开 8 个进程，每个 1 线程。
6. **预算**：三个闸门合计 CAP = 23 卡时，不经用户同意不能提高。
   - 按闸门预留：TVR 0.5，E3VS 10.5，WalkerBench 12.0。
   - 提交前检查：已结束作业的实际花费 + 所有在跑和已提交作业的估计上限 + 新作业的估计上限 ≤ 该闸门的预留。
   - 台账写入是原子的（加锁文件）。超预算时，先去掉可选的评测。
7. **跑完就删**：作业跑完、`collect` 核对了预期输出以后，删掉作业记录。删之前再查一次状态。
8. **出处**：每个作业的 prepared.json 都记代码的 git sha 和 PREREG 的提交 sha。WalkerBench 的三个试点作业，要等 smoke 作业判 PASS、PREREG_WB_PILOT.md 已提交之后才能准备。
