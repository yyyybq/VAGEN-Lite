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
  - sco 作业不直接跑 worktree，而是跑提交时的代码快照 `RUN/code`（见第 7 节）。
  - 不要从 VAGEN-Lite-downstream 或 VAGEN-Lite-eval 导入任何东西：那是别的窗口正在改的 worktree。

## 2. vpd/ 目录

| 路径 | 内容 | 负责的工作包 |
|---|---|---|
| `README.md` | 本文件 | WP-PREREG |
| `PREREG_WB_PILOT.md` | WalkerBench 试点的事先登记。用户确认后提交，提交以后冻结 | WP-PREREG |
| `PREREG_E3VS_PILOT.md` | E3VS 试点的事先登记 | WP-E3VS-A |
| `__init__.py`、`paths.py` | OUT 的目录常量、各解释器路径 | WP-CORE |
| `starts.json`、`starts.py` | 起点表（名字 → 评测 view 代码、checkpoint 路径），加载时断言注意力实现。2026-11-13 末尾只加了一条 Q3SG（genc0，TVR 探索起点，DECISIONS G20），训练器指纹因此从 0865dd75f1ee 变成 119da3c84b8d | WP-CORE；第三轮 G20 |
| `sco.py`、`ledger.py` | 包装 `downstream/sco_job.py` 的 sco 工具，带作业台账和按闸门预留的预算检查 | WP-OPS |
| `snapshot.py` | 作业代码快照：prepare 时建 `RUN/code` 和 `SNAPSHOT.json`（三个目录加根目录的 `ply_gaussian_loader.py`）；作业里核对文件（第 7 节） | infra（第三轮） |
| `EXPLORATORY_WB_VARIANTS.md` | WalkerBench 限量探索（DECISIONS G4）的说明：开发集、选择规则、变体清单。探索性质，不是闸门 | wbvar（第三轮） |
| `walkerbench/variants.py`、`walkerbench/odometer.py` | 变体探索：建 s2dev74 和 s2sel 两个题集；V0、V1T、V2、V4、V5 的训练数据；初始化和适配器核对；分析；expect。`odometer.py` 生成里程计那一行（V1、V1T、V2 用） | wbvar（第三轮） |
| `jobs/wb_variants.sh` | sco 作业脚本：`wb_variants.sh VARIANT RUN [BASE]`（作业 vpd-wb-v0v1，先重建 G6 的那一条示范；vpd-wb-v1t、vpd-wb-v2；之后的 V5、V4）。输出只在 `OUT/walkerbench/runs/explore/wbv/` | wbvar（第三轮） |
| `walkerbench/perc.py` | 感知零样本（G5）：建 perc294 题集并核对；评测包装，每个起点参数完全相同；分析；expect。题集在 `OUT/walkerbench/perception/perc294/`，输出在 `OUT/walkerbench/runs/explore/perc/` | perc（第三轮） |
| `jobs/wb_perc.sh` | sco 作业脚本：`wb_perc.sh RUN [START ...]`（作业 vpd-wb-perc，以 root 跑；不给 START 时 Q3B、Q3S、E2 依次评测，给了就只跑列出的起点，顺序仍固定，没列出的记 `eval_<START>:not_listed`） | perc（第三轮） |
| `R/<起点>/evals/perc_eval.lock`（R = `OUT/walkerbench/runs/explore/perc`） | perc eval 的每起点运行目录锁（O_EXCL 建，记 host、pid、utc、RUN、token），评测返回时删掉。别的 RUN 持有、且那个 RUN 既没有 DONE 也没有 `job/exit_code` 时拒绝评测。拿到锁后先看运行目录：已有 results.json（返回 0）或 results_invalid.json（返回 1）且没给 `--overwrite` 时不再跑，explore.json 也不重写（1c） | perc（第三轮）；wave1c |
| 环境变量 `VPD_PERC_TAKEOVER`、`VPD_PERC_TEST_ALLOW_LIVE` | `VPD_PERC_TAKEOVER=1`：接管上面那把锁，只在第一个作业已经终止（`vpd.sco status`）却没留下 DONE 或 `job/exit_code` 时重投用（wb_perc.sh 会传下去）。`VPD_PERC_TEST_ALLOW_LIVE=1`：只给封闭测试用，让 perc eval 在代码快照之外也能跑（登录节点上绝不设） | perc（第三轮） |
| `EXPLORATORY_WB_LEARN483.md` | WalkerBench 可学性探索（用户 2026-11-09 的要求：只用 Q3B，在全部 483 道训练题上微调，不做判断）的说明。探索性质，不是闸门 | wb483（第三轮） |
| `walkerbench/learn483.py` | 483 题探索：建 s2in50 题集、下限、init-check、评测包装（运行时注册 s2in50）、teacher-forced 探针和合并一致性、分析、expect | wb483（第三轮） |
| `jobs/wb_learn483.sh` | sco 作业脚本：`wb_learn483.sh A\|B RUN`（作业 vpd-wb-x483a、vpd-wb-x483b，以 root 跑）。输出只在 `OUT/walkerbench/runs/explore/x483/Q3B/`。步骤表的每一块都经过 `part`，init-check 或训练失败只记下来，后面的步骤照跑（1c） | wb483（第三轮）；wave1c |
| `tests/test_learn483.py` | 483 题探索的测试（解析、探针数字、题集、作业脚本的空跑） | wb483（第三轮） |
| `tvr/lvp.py`、`tvr/probe_lvp.py` | TVR 的 lavapipe（CPU Vulkan）渲染：`lvp.py` 从零建 worker 环境、从只读仓库加载 TVRBench、字体规则、角标遮罩；`probe_lvp.py` 测吞吐网格和确定性，并和公开样本比对（G3） | tvrprobe（第三轮） |
| `python -m vpd.tvr.probe_lvp recommend --out RUN/out` | 只用标准库，从探针的 `summary.json` 重算两个推荐：`recommended`（E×L ≤ 11，TVR 试点用的是这个）和 `recommended_measured`（实测 cgroup 核数 ≤ 11，只记录） | tvrprobe（第三轮） |
| `tvr/pilot_cfg.py` | TVR 试点的冻结常量（动作、正则、渲染、协议、档位、配方、闸门、功效、预算），所有 TVR 模块都从这里取；`missing(job)` 列出某个作业还缺的测量值；py3.10 和 py3.12 都能导入 | prereg（第三轮第二批） |
| `PREREG_TVR_PILOT.md` | TVR 试点的事先登记（中文），里面的 TVR_FROZEN 块必须和 `pilot_cfg.FROZEN` 完全一致 | prereg（第三轮第二批） |
| `common/power_sim.py` | A4 的功效模拟：带种子方差的场景聚类 t 检验 + Holm，输出 `POWER.json` | prereg（第三轮第二批） |
| `tvr/tvr_text.py` | TVRBench 系统提示、每步用户文本、动作解析的逐字副本（不导入 ai2thor），带对固定源码的 ast 核对 | tvrdata（第三轮第二批） |
| `tvr/tasks.py` | 读 sft/eval/rl 任务；按哈希选 dev200；命名题集；公开动作序列；0W 池和嵌套档位；专家中位长度 | tvrdata（第三轮第二批） |
| `tvr/kin_env.py` | 运动学环境：和 ThorEnv 相同的转移、可走集合、合法动作列表和用户文本，不渲染 | tvrdata（第三轮第二批） |
| `tvr/bfs.py` | 在冻结可走集合的位姿图上做 BFS，得到每题最短非 Stop 步数（SPL 的 opt） | tvrdata（第三轮第二批） |
| `tvr/floors.py` | 盲下限 F：STOP0、固定动作 k 次后 Stop、按种子的随机策略，取三者最大 | tvrdata（第三轮第二批） |
| `tvr/reach.py` | ProcTHOR 房子文件；每题起点位姿的可走集合，两次独立加载（A、B），冻结 A；`ReachIndex` | tvrdata（第三轮第二批） |
| `tvr/demos.py` | 从公开动作序列的 THOR 回放生成示范（保留规则、丢弃原因、kin 核对、记录格式 6.5） | tvrdata（第三轮第二批） |
| `tests/tvr_fakes.py` | 共用测试工具：假 ai2thor、假任务和网格、假帧、假 TVR 目录树 | tvrdata（第三轮第二批） |
| `tests/test_tvr_data.py` | tvrdata 各模块的测试 | tvrdata（第三轮第二批） |
| `tvr/thor_worker.py` | THOR worker 进程（TVR_PY）和父进程一侧的池：重置时注入冻结可走集合、走一步、出 PNG | tvreval（第三轮第二批） |
| `tvr/run_tvr.py` | TVR 评测和回放的总控：`eval`（c、u、blind 三个解码臂）、`replay`、`rescore`；断点续跑、截止时间、快速关停 | tvreval（第三轮第二批） |
| `tvr/score_tvr.py` | 打分：SR 和 Wilson 区间、SPL、within1、停止率、场景自助法；`official` 用 eval_va 的 aggregate 对照 | tvreval（第三轮第二批） |
| `tvr/parity_tvr.py` | vLLM 提示 token 数和训练 dry run 的一致性检查（PNG data URL） | tvreval（第三轮第二批） |
| `tvr/fake_server.py` | 测试用的假 OpenAI 接口服务器（按策略回复，确定性的 usage） | tvreval（第三轮第二批） |
| `tvr/render_parity.py` | 渲染一致性按容差判（PSNR ≥ 40 dB 且 SSIM ≥ 0.99，状态、合法动作、用户文本、action_success 必须完全相同）；smoke 和每个评测作业的渲染金丝雀用它 | tvreval（第三轮第二批） |
| `tests/test_tvr_eval.py` | tvreval 各模块的测试 | tvreval（第三轮第二批） |
| `tvr/pilot_tvr.py` | TVR 试点编排（plan / prepare / submit / status / collect / delete / budget / expect / smoke-verdict / cal-decide），含本轮 200 卡时的预算拒绝。2026-11-12 起另有探索模式（DECISIONS G18）：单独的作业表 `XJOBS`（vpd-tvr-xsmoke、vpd-tvr-xtrain-q3rd / -q3rs150、vpd-tvr-xeval-<s>-top / -low，`EXPLORE_STARTS` = Q3RD、Q3RS150），同一配方，输出在 `OUT/tvr/runs/explore/rl/`；守卫另查评测框架指纹、预注册 2026-11-12 那一行、档位文件、适配器的种子和初始化链；本块 25 卡时的预算拒绝；`plan`、`status`、`budget` 加 `--explore` 才列探索作业；`xsmoke-verdict`。试点作业的表、payload、守卫和 expect 不变。2026-11-13 起第三个探索起点 Q3SG（genc0，预注册第 17 节 2026-11-13 那一行，DECISIONS G20）：`EXPLORE_STARTS` = `RL_STARTS`（Q3RD、Q3RS150）+ Q3SG，同一探索根 `runs/explore/rl/`、同一本块预算；作业 vpd-tvr-xsmoke-q3sg（只冒烟 Q3SG，est_high 0.7 = 一个起点 1,815 秒 + 300 + 180 秒向上取到 0.1）、vpd-tvr-xtrain-q3sg、vpd-tvr-xeval-q3sg-top / -low；每个起点各自的登记行（RL 2026-11-12，Q3SG 2026-11-13）、各自的训练器规则（Q3SG：指纹以 119da3c84b8d 开头，且 5 个 .py 的 sha == 试点记录的、starts.json 去掉 Q3SG 后按 indent=1 重写的 sha == 86bb356e；其他作业仍要 0865dd75f1ee，所以在这棵树上拒跑，有意如此）、各自的冒烟判定（Q3SG：runs/explore/rl/smoke 下 starts 含 Q3SG 的最新一份 PASS）；探索适配器按起点登记的训练器代码核对，不按当前树 | tvrjobs（第三轮第二批）；第三轮 G18、G20 |
| `tvr/gate_tvr.py` | TVR 闸门判定 → `GATE.json`、`GATE.md`；从 trajectory.json 独立重数 k/n；不算起点之间的差；G0 第 7 条自 2026-11-11 起只判合并是否正确（`merge_correctness.json` 登记的诊断报告，T−MF ≤ 1e-4），bf16 的 U−M 只报告（预注册第 17 节） | tvrjobs（第三轮第二批）；第三轮 G16 |
| `tvr/compare_tvr.py` | 登记的只报告比较（预注册第 17 节 2026-11-11）：Q3S − Q3B（主报告）、Q3S − E2（补充）每格成对 b、c、精确 p，各起点 k/n 和 Wilson，参照行 Q3B 100 种子 1001 对 1002 → `COMPARE.json`、`COMPARE.md`；GATE.json、POWER.json 写完才算。`eval_dir` / `read_side` 的评测目录前缀默认 pilot（compare_rl 传 xrl） | 第三轮 G16 |
| `tvr/compare_rl.py` | 探索性质的比较（预注册第 17 节 2026-11-12，DECISIONS G18，只报告）：Q3RD − Q3B、Q3RS150 − Q3B（主报告），Q3RD − Q3S、Q3RS150 − Q3S（补充），5 格成对 b（只有前一个起点成功）、c、精确 p 和两边 Wilson，五个起点（Q3B、Q3S、E2、Q3RD、Q3RS150）各档各臂的 k/n；RL 一侧读 `runs/explore/rl`（前缀 xrl），Q3B、Q3S、E2 读已有的试点评测 → `runs/explore/rl/COMPARE_RL.json`、`COMPARE_RL.md`；10 个探索评测都有效之前拒算。`validate`、`side_root`、`crosscheck`、`start_table` 带起点参数（默认本比较的），compare_q3sg 复用；COMPARE_RL 的输出不变 | 第三轮 G18 |
| `tvr/compare_q3sg.py` | 探索性质的比较（预注册第 17 节 2026-11-13，DECISIONS G20，只报告，和 compare_rl 同口径）：Q3SG − Q3B（主：100 c、ALL c）、Q3SG − Q3S（诊断：同两格），compare_rl 的 5 格都照报作描述；每格成对 b、c、精确 p、两边 Wilson；Q3B、Q3S、Q3SG 各档各臂的 k/n；两格各自的预判（Q3SG 的 k ≥ Q3B → 遗忘；Q3SG − Q3S p ≥ 0.05 且 Q3SG − Q3B p < 0.05 → 动作任务本身；其他 → 介于两者之间，不判）→ `runs/explore/rl/COMPARE_Q3SG.json`、`COMPARE_Q3SG.md`；5 个 Q3SG 评测都有效、且每个 Q3SG 适配器的 code_files 满足"5 个 .py 的 sha 和试点 Q3B、Q3S 适配器记录的相同、starts.json 只多 Q3SG 这一条"之前拒算 | 第三轮 G20 |
| `tvr/raw_bf16.py` | 探索冒烟用：按训练器的加载方式（vpd.starts.load_model，float32 → bf16）读原始起点，按 merge.py 的方式存成作业本地盘上的 bf16 目录给 vLLM（RL 起点没有评测视图），并逐位核对几个张量 == checkpoint 转 bf16 | 第三轮 G18 |
| `jobs/tvr_xsmoke.sh` | sco 作业脚本：`tvr_xsmoke.sh RUN`（作业 vpd-tvr-xsmoke，以 root 跑，探索性质，两个 RL 起点一起）：每个起点 bf16 加载、dry run、HF 一致性、原始起点的 vLLM 提示一致性和 smoke8 机制会话；真训练器在 0W 前 16 条上训 1 个 epoch（2 个优化步）、merge.py 合并、合并一致性（只报告）、合并模型的 smoke4 c 臂；`xsmoke-verdict`，不算 SR。起点表参数：`tvr_xsmoke.sh [Q3RD,Q3RS150\|Q3SG] RUN`，不给时是 Q3RD,Q3RS150（作业 vpd-tvr-xsmoke，命令和以前逐字节相同）；Q3SG 是作业 vpd-tvr-xsmoke-q3sg（只冒烟 Q3SG，`xsmoke-verdict --job tvr-xsmoke-q3sg`）；别的表退出 2 | 第三轮 G18、G20 |
| `tvr/recheck_tvr.py` | 独立重算（不导入 tvrdata 的模块）：下限 F 和逐题计数 | tvrjobs（第三轮第二批） |
| `tvr/merge_parity.py` | 合并权重和 LoRA 的 teacher-forced 一致性（平均 \|dCE\| ≤ 0.005，回复 argmax 一致率 ≥ 0.99），每个合并评测都跑 | tvrjobs（第三轮第二批） |
| `jobs/tvr_data.sh` | sco 作业脚本：`tvr_data.sh RUN`（作业 vpd-tvr-data，非 root）：dev200、房子、可走集合、回放、示范、档位、BFS、下限 | tvrjobs（第三轮第二批） |
| `jobs/tvr_cal.sh` | sco 作业脚本：`tvr_cal.sh RUN`（作业 vpd-tvr-cal，以 root 跑，只用 Q3B、只在 dev200 上，探索性质）：机制检查、渲染金丝雀、0W/100/ALL 训练和评测、cal-decide | tvrjobs（第三轮第二批） |
| `jobs/tvr_cal400.sh` | sco 作业脚本：`tvr_cal400.sh RUN`（作业 vpd-tvr-cal400，只在 cal_decision.train_400 时跑）：400 档训练和评测 | tvrjobs（第三轮第二批） |
| `jobs/tvr_mech_smoke.sh` | sco 作业脚本：`tvr_mech_smoke.sh RUN`（作业 vpd-tvr-msmoke，以 root 跑）：三个原始起点的机制冒烟，不算 SR；root 回放的渲染一致性 | tvrjobs（第三轮第二批） |
| `jobs/tvr_train.sh` | sco 作业脚本：`tvr_train.sh START RUN`（作业 vpd-tvr-train-<s>）：0W、100、TOP 档训练（Q3B 还有种子 1002 的复制）。环境变量 `VPD_TVR_ROOT=explore/rl` 时是探索作业 vpd-tvr-xtrain-<s>（起点 Q3RD、Q3RS150、Q3SG，输出在 `runs/explore/rl/<起点>/`）；不设时和以前完全一样 | tvrjobs（第三轮第二批）；第三轮 G18、G20 |
| `jobs/tvr_eval.sh` | sco 作业脚本：`tvr_eval.sh START top\|low RUN`（作业 vpd-tvr-eval-<s>-top、-low）：金丝雀、合并、合并一致性、评测、official。`VPD_TVR_ROOT=explore/rl` 时是探索作业 vpd-tvr-xeval-<s>-<part>（起点 Q3RD、Q3RS150、Q3SG；评测前缀 xrl；合并一致性只报告：本适配器的 json 判 FAILED 时记 `merge_parity_<档>:report_only`）；不设时和以前完全一样 | tvrjobs（第三轮第二批）；第三轮 G18 |
| `tests/test_tvr_jobs.py`、`tests/test_tvr_integration.py` | tvrjobs 的测试（作业脚本命令清单、prepare 守卫、闸门）；全链路集成测试（假 TVR 目录树） | tvrjobs（第三轮第二批） |
| `tests/test_tvr_explore.py` | 探索模式的测试：XJOBS、守卫、本块预算、expect、步骤、xsmoke-verdict（不读成功字段）；三个作业脚本的空跑和假解释器下的非空跑；raw_bf16（小模型）；compare_rl（合成目录）。试点和 RL 作业的测试在试点的训练器指纹下跑（`test_tvr_jobs.use_pilot_trainer`：本树的训练器文件加试点的 starts.json） | 第三轮 G18 |
| `tests/test_tvr_q3sg.py` | 第三个探索起点 Q3SG 的测试：starts.json 只多 Q3SG 一条；训练器指纹（本树 119da3c84b8d，用原来的 starts.json——`git show 216656b9`——重算 0865dd75f1ee）、评测框架指纹 64edd575f36b 不变；Q3SG 作业的表、payload、est_high、本块预算、守卫（指纹前缀和不变量、2026-11-13 那一行、Q3SG 冒烟判定、处理器目录）、步骤、expect、xsmoke-verdict；试点和 RL 作业在本树上拒跑；`tvr_xsmoke.sh` 起点表（默认不变）和 Q3SG 的 train / eval 空跑；compare_q3sg（合成目录：各格、预判、拒算；COMPARE_RL 不变） | 第三轮 G20 |
| `tests/test_tvr_prereg.py`、`tests/test_power_sim.py` | 事先登记块 == `pilot_cfg.FROZEN`、`missing()` 语义；功效模拟 | prereg（第三轮第二批） |
| `tests/test_wave1c.py` | 第三轮 1c 的几处小修：perc 已完成的运行不重写 explore.json、wb_learn483.sh 的 `part`、E3VS `code_meta` 在快照里的 `worktree_head`、强删时复查看到 start_time 不记 0 卡时；本 README 覆盖第二批的每个文件 | wave1c（第三轮第二批） |
| `jobs/tvr_probe.sh` | sco 作业脚本：`tvr_probe.sh RUN`（作业 vpd-tvr-probe，闸门 tvr，非 root，只用 CPU 渲染） | tvrprobe（第三轮） |
| `tvr/vk_recheck.sh`、`tvr/vkre.py` | NVIDIA Vulkan 复查（G2）：作业脚本 `vk_recheck.sh OUTDIR`（作业 vpd-tvr-vkre，系统驱动 + 加载库 1.4.357 + 绝对路径 ICD 的几种变体，加对照），`vkre.py` 汇总并写 expect。不影响冻结的 TVR 冒烟判定 | vkre（第三轮） |
| `jobs/common_env.sh`、`jobs/preflight.sh` | 作业脚本的公共开头（线程数、离线开关、chown 陷阱、节点信息）和开跑前检查 | WP-OPS |
| `jobs/wb_demos.sh` | sco 作业脚本：渲染示范、生成热身集、生成档位 | WP-WBDEMO |
| `jobs/wb_smoke.sh`、`jobs/wb_pilot_start.sh` | sco 作业脚本：WalkerBench smoke 和每个起点的试点 | WP-WBPILOT |
| `jobs/e3vs_prep.sh`、`jobs/e3vs_start.sh` | sco 作业脚本：E3VS 准备和每个起点的试点。2026-11-12 起碰撞模型是官方 mesh：准备作业的 `mesh_val` 一步（原 `occupancy`）验证 99 个 mesh，不过就停（退出码 1 闸门指标不过、2 输入被拒、3 出错没有结果、4 可能是坐标系不一致，都停下问用户）；KEY 是 99 个 mesh 的 `mesh_set_sha256`；新 RUN 里 gsplat_probe、parity 按已有输出跳过；起点作业在 gsplat 之后、训练之前做一次碰撞模型冒烟（`mesh_collision smoke`）；只有要用碰撞模型的 PY_Q3 调用（mesh_val、BFS、下限、冒烟、run.py）才把 `envs/e3vs-o3d` 接在 PYTHONPATH 后面，run.py 不把它传给 vLLM 服务 | WP-E3VS-B；meshcm（第三轮） |
| `walkerbench/sandbox.py` | 唯一的 WalkerBench 运行副本 helper：改渲染尺寸、全景目录、任务目录 | WP-CORE |
| `walkerbench/history.py` | 干净的 K 轮历史窗口，训练和评测共用 | WP-CORE |
| `walkerbench/action_regex.py` | 约束解码用的每步正则 | WP-CORE |
| `walkerbench/data/split_city.json` | 冻结的城市切分源清单（sha256 已登记） | WP-WBDATA |
| `walkerbench/split.py`、`tiers.py`、`baselines.py` | 生成 S2 切分；生成嵌套档位顺序；在 179 题上跑 STOP0、FWD、ODO 和专家。第三轮 B1：`baselines.rm_expert` 把朝向折回 [0, 360)（G6） | WP-WBDATA；wbtrain（第三轮） |
| `walkerbench/demos.py`、`warmup.py` | RM 专家示范（640x400 真实渲染）；20 段格式热身 | WP-WBDEMO |
| `walkerbench/run_vpd.py`、`smoke_vpd.py` | 评测副本（约束解码、渲染尺寸、K 窗口、合并模型）；假服务器冒烟 | WP-WBEVAL |
| `walkerbench/pilot.py`、`pilot_gate.py` | 试点编排（plan / prepare / submit / status / collect / delete / budget）；闸门判定 | WP-WBPILOT |
| `common/vllm_server_vpd.py` | vLLM 服务副本：可指定模型目录、结构化输出、图像像素上限。WalkerBench 和 E3VS 共用 | WP-WBEVAL |
| `train/n2_train_vpd.py` | 训练脚本，由 `downstream/n/n2_train.py` 改来。第三轮加了每条回复可选的 `loss_weight`（G4 的 V5） | WP-TRAIN；wbtrain（第三轮） |
| `train/parity_check.py`、`train/merge.py`、`train/wb_train_probe.py` | 两个环境的处理器一致性检查；合并到作业本地盘；长序列吞吐探针 | WP-TRAIN |
| `tvr/` | TVR 冒烟：登录节点准备脚本、冒烟作业、Vulkan/EGL 探针、判定脚本、`SMOKE_CRITERIA.md` | WP-TVR |
| `e3vs/` | E3VS 相关代码（几何、BFS 示范、盲下限、下载脚本、`USER_STEPS.md`、渲染、碰撞模型、SFT 数据、闭环评测、汇总）。碰撞模型 2026-11-12 起是 `mesh_collision.py`：按文件路径执行固定版本仓库里官方的 `e3vs/eval/metrics.py` `Collisions`（ScanNet++ `mesh_aligned_0.05.ply` + Open3D 光线投射，容差 1e-4 m），BFS 示范、盲走下限、闭环评测共用；`mesh_verify.py` 校验用户下载的 mesh；占据栅格 `occupancy.py` 留作记录，作业不再用 | WP-E3VS-A、WP-E3VS-B；meshcm（第三轮） |
| `tests/test_e3vs_mesh_collision.py` | E3VS 官方 mesh 碰撞模型：官方文件的版本和 sha256、规则 (ii) 的官方语义（两端只有 1e-4 m 容差）、"在障碍里"、验证和拒绝、check、工厂，以及三处共用同一个碰撞对象。open3d 没装时用一个替身跑官方代码；装了以后同样的用例再用真库跑一遍 | meshcm（第三轮） |
| `tests/` | 每个工作包的 pytest。在登录节点上单核、不用 GPU 就能跑 | 各工作包 |
| `tests/test_snapshot.py`、`tests/test_snapshot_facts.py` | 代码快照（建快照、prepare 改写、launch.sh、作业里的核对）；快照里的 git 事实（评测 manifest、训练 meta、E3VS 评测）和 E3VS 记下的加载器 | infra（第三轮） |
| `tests/test_wb_variants.py`、`tests/test_perc.py` | WalkerBench 变体探索；感知零样本 | wbvar、perc（第三轮） |
| `tests/test_tvr_probe.py`、`tests/test_vk_recheck.py` | lavapipe 探针；NVIDIA Vulkan 复查 | tvrprobe、vkre（第三轮） |
| `tests/test_train_weights.py` | 训练脚本的按回复 loss_weight（G4 的 V5） | wbtrain（第三轮） |

## 3. OUT 目录

| 路径 | 内容 |
|---|---|
| `plan/` | 计划、决定、审查、调查记录，以及从登录节点 /tmp 拷来的原型（`plan/login_tmp/`） |
| `scratch/<标签>/` | 各工作包自己的临时目录。只能删自己建的东西，绝不用 /tmp |
| `ledger.jsonl` | 每个 sco 作业一行：名字、pt 编号、开始和结束时间、卡时 |
| `jobs/<作业名>-<yyyymmddHHMM>/` | 每个 sco 作业的 RUN 目录（作业文件、worker.log、node.txt、prepared.json、steps.jsonl），以及代码快照 `code/` |
| `walkerbench/split/` | `split_s2.json`、`expert_splice.json`，以及 `s2test/`、`s2val/`、`s2train/` 三个部分，每个部分下有 `tasks/` 和 `selection.json` |
| `walkerbench/baselines/` | `baselines_s2test.json`（逐题结果、按类型的下限、合并下限 F）、`hard_subsets.json` |
| `walkerbench/demos/s2train_640x400/` | `demos_all.jsonl`、`images/<stem>/turn_XX.jpg`、`manifest.json`、`DONE` |
| `walkerbench/warmup/` | `warmup20.jsonl`、`images/`、`manifest.json` |
| `walkerbench/tiers/` | 每个种子一份 `order_seed<种子>.json`，以及 `seed<种子>/demos_<N>.jsonl`；种子有 1001（试点）和 1、2、3（确认轮） |
| `walkerbench/runs/smoke/` | smoke 作业的输出和 `smoke_verdict.json` |
| `walkerbench/runs/pilot/<起点>/` | 适配器 `adapters/{0W,50,200}/` 和评测结果 `evals/pilot__<起点>__<档>__<分辨率>__<解码>/` |
| `walkerbench/runs/pilot/` | `GATE.json` 和 `GATE.md` 也放这里。探索性的输出一律不放这里 |
| `walkerbench/split/s2dev74/`、`walkerbench/split/s2sel/` | 变体探索的开发集（74 题）和选择集 SEL（s2val + s2dev74 = 131 题）：带 sha 核对的任务副本和 `selection.json`（G4） |
| `walkerbench/runs/explore/wbv/` | WalkerBench 变体探索的输出（探索性质，G4） |
| `walkerbench/demos/s2train_640x400_B1_vis0578/` | G6 重建的那一条示范（wb_variants 作业 A 写；冻结的示范目录不动） |
| `walkerbench/perception/perc294/` | 感知零样本的题集：`tasks/`（294 个 sha 核对过的副本）和 `selection.json`（G5） |
| `walkerbench/runs/explore/perc/` | 感知零样本的输出（探索性质，单独计卡时，不算进 WalkerBench 的 30 卡时） |
| `tvr/` | 图形库副本（nvgl_595.71.05）、AI2-THOR 的 HOME、TVRBench 仓库（`repos/TVRBench`，只读）、公开数据的元文件（`hf_meta/tvr-sft-va/sft_train.json`）、冒烟输出 |
| `tvr/split/dev200/` | `selection.json`：从 rl.json 按哈希选的 200 道开发题（每类 50，每个场景-类别最多 2 道），以及 smoke8、smoke4 |
| `tvr/houses/` | 每个 ProcTHOR 场景一份 `<场景>.json`（train.jsonl 那一行原文）和 `houses.json`（来源和 sha256） |
| `tvr/reach/v1/` | `scenes/<场景>.json`（每题起点位姿的可走集合，A、B 两次加载）、`index.json`（冻结值 = A 的摘要）、`DONE` |
| `tvr/replay/sft1600_lvp/` | 1,600 条公开动作序列的 THOR 回放（run_tvr 的运行目录：protocol、manifest、`tasks/<任务>/trajectory.json`、results.json） |
| `tvr/demos/sft1600_lvp/` | 训练示范：`demos_all.jsonl`、`images/<任务>/step_XX.png` 和 `target.png`、`dropped.json`、`kin_check.json`、`images_sha256.json`（只核对文件完整性）、`manifest.json`、`DONE` |
| `tvr/tiers/v1/` | `warmup0w20.jsonl`、`demos_100.jsonl`、`demos_400.jsonl`、`demos_ALL.jsonl`、`parity3.jsonl`、`order.json`、`manifest.json`（档位嵌套，和 0W 不相交） |
| `tvr/bfs/` | `<题集>_opt.json`：每题在冻结可走集合上的 BFS 最短步数、是否可达 |
| `tvr/floors/` | `floors_eval500.json`、`floors_dev200.json`：三种盲策略的成功率和下限 F |
| `tvr/runs/explore/cal/Q3B/` | 校准作业的输出（探索性质，只用 Q3B、只在 dev200 上）：`dry/`、`shake/`、`canary/`、`adapters/`、`evals/`、`cal_decision.json`；`futility_override.json` 只由编排在用户决定后写 |
| `tvr/runs/smoke/<RUN 名>/` | 机制冒烟的输出和 `smoke_verdict.json`（不算 SR） |
| `tvr/runs/pilot/<起点>/` | 试点的 `adapters/<档>/`、`evals/pilot__<起点>__<档>__<臂>/`、`parity/merge_<档>.json`、`canary/<作业>/render_parity.json` |
| `tvr/runs/pilot/` | `GATE.json`、`GATE.md`（gate_tvr），以及 `POWER.json`（power_sim，只在 G0-G3 通过后）；`merge_correctness.json`（编排写的合并正确性报告登记表，G0 第 7 条读）；`COMPARE.json`、`COMPARE.md`（compare_tvr，在 GATE、POWER 之后） |
| `tvr/runs/explore/rl/<起点>/` | 探索性质的 RL 起点（Q3RD、Q3RS150；DECISIONS G18）和第三个探索起点 Q3SG（DECISIONS G20；"rl" 只是目录名）：`adapters/<档>/`、`evals/xrl__<起点>__<档>__<臂>/`、`parity/merge_<档>.json`（只报告）、`canary/<作业>/render_parity.json`。不进 `runs/pilot/` |
| `tvr/runs/explore/rl/smoke/<RUN 名>/` | 探索冒烟 vpd-tvr-xsmoke、vpd-tvr-xsmoke-q3sg 的输出（每个起点的 `load/`、`dry/`、`evals/`、`adapters/0W/`、`parity/`）和 `smoke_verdict.json`（不算 SR；`starts` 记这次冒烟的起点） |
| `tvr/runs/explore/rl/` | `COMPARE_RL.json`、`COMPARE_RL.md`（compare_rl，10 个探索评测都有效之后）；`COMPARE_Q3SG.json`、`COMPARE_Q3SG.md`（compare_q3sg，5 个 Q3SG 评测都有效、训练器不变量成立之后） |
| `tvr/runs/explore/mpdiag/` | 合并正确性诊断（vpd-tvr-mpdiag，探索性质）的报告，每次一个 `<RUN 名>/`（`mp_diag.json`、`mp_diag.md`） |
| `envs/tvr` | TVR 的 conda 环境（要用户批准后才建） |
| `envs/e3vs-o3d` | E3VS 碰撞模型用的 open3d 0.19.0（官方固定的版本；`pip --target`，用户 2026-11-13 批准后已装）。只有 E3VS 两个作业里要用碰撞模型的 PY_Q3 调用把它接在 PYTHONPATH 后面；`mesh_collision o3d-check` 拒绝其中任何一个 PY_Q3 自己也能导入的顶层模块 |
| `e3vs/` | `repo/`、`data/`、`mesh_dl/`（用户下载的 ScanNet++ mesh）、`mesh_manifest.json`、`mesh_sha256.txt`（mesh_verify）、`mesh_val/validation.json`（碰撞模型验证）、`parity/`、`torch_ext/`、`prep_state/`、`bfs/mesh/`、`floors/floors_mesh.json`、`sft/`、`runs/`；`occ/` 是 2026-11-12 没通过验证的占据栅格，留作记录 |

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
   - `--pool zoe-cf-render`（2026-11-10，DECISIONS G7）只能显式指定，`--pool auto` 永远不会选它；规格同样是 N4lS.Iq.I80.1。有没有配额要投了才知道。
   - 用 `--force-live` 删掉一个从没开始的作业（两次查询都在排队、没有 start_time、worker.log 里没有 `[launch]` 行），台账记 0 卡时，`gph_source` 为 `never_started`。
     - 前提是删除确实生效：`sco_job.delete` 报了 `delete rc=0`，或者删完再查一次，看到没有 start_time 的终止状态或作业列表确认的 NOT_FOUND。
     - 否则（stop 或 delete 失败，作业以后还可能拿到卡；或者复查看到的终止状态带 start_time，说明第二次查询之后它拿到过卡）照旧按估计记（`unknown_charged_est`）。台账的 end 事件里 `delete_check` 记了判断依据，强删的 delete 事件里 `rc` 是从 `sco_job.delete` 输出里解析出的返回码（没有就是 null）。
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

## 7. 代码快照（2026-11-10，DECISIONS G1）

作业跑的是提交时冻结的一份代码，不是 worktree 本身。这样作业在跑的时候，别的线改 worktree 里的文件也影响不到它。

1. **prepare 时**（默认开启）：
   - `git status --porcelain --untracked-files=all -- vpd downstream eval_external ply_gaussian_loader.py` 不为空就拒绝。被忽略的文件（`__pycache__` 等）不算。先提交，再 prepare。
   - 把 HEAD 的这三个目录，加上根目录的 `ply_gaussian_loader.py`（`snapshot.EXTRA_FILES`），用 `git archive` 解到 `RUN/code`，逐个文件核对 git blob。HEAD 里没有这个文件也拒绝。
     - 为什么要带它：`vpd/e3vs/render.py` 要 `from ply_gaussian_loader import PLYGaussianLoader`。以前的作业直接跑 worktree，导入的是 worktree 根目录那一份；快照里没有它的话，作业会退到 eval-external 环境 site-packages 里的另一份（两份只在压缩 PLY 的解码上不同）。
     - E3VS 的渲染记录（`SplatRenderer.meta`、parity 的 `code`）里有 `ply_gaussian_loader`：实际导入的模块路径和 sha256，`in_worktree` 表示是不是快照（worktree）里那一份。
   - 写 `RUN/code/SNAPSHOT.json`：commit、utc、trees、extra_files、n_files、tar_sha256、来源 worktree、每个文件的 blob，以及冻结文档（`PREREG_*.md`、`SMOKE_CRITERIA.md`）最后一次改动的提交。`RUN/job/vpd.json` 和 prepared.json 里也记一份摘要。
     - `extra_files` 是后加的键，格式号仍是 1；以前的快照没有这个键，表示没有额外文件。
2. **payload**：以 worktree 路径开头的参数，一律改写成 `RUN/code` 下的同一路径。所以 prepare 命令里照旧写 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/...`。
   - 只是"包含" worktree 路径的参数（比如 `KEY=/…/VAGEN-Lite-vpd/…`、`bash -c` 字符串）会被拒绝。
   - 改写后的路径在快照里不存在（没提交、被忽略、不在快照里）也拒绝。
   - `--env` 的值里提到 worktree 路径（同样按路径边界判断，比如 `VPD_COMMON_ENV:/…/VAGEN-Lite-vpd/vpd/jobs/common_env.sh`）也拒绝，否则作业会去读 worktree 里的文件。`--no-snapshot` 时不查。
3. **launch.sh**：导出 `VPD_WORKTREE=RUN/code` 和 `PYTHONPATH=RUN/code`，再 cd 进 `RUN/code`。
   - 紧跟在 `exec > >(tee -a worker.log) 2>&1` 那一行后面检查：`RUN/code/SNAPSHOT.json` 必须存在，当前目录必须就是 `RUN/code`。不满足就把原因写进 worker.log，`job/exit_code` 写 97，然后以 97 退出。
   - `vpd.paths.WORKTREE` 跟着 `VPD_WORKTREE` 走；`jobs/common_env.sh` 没有这个变量时，取它自己所在的那棵树。
   - 新的作业脚本按约定这样开头：`source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"`，然后 `cd "$VPD_WORKTREE"`。代码里不再写死 worktree 路径。
   - 例外：非 root 作业的属主包装脚本 `examples/train/active_spatial/sco_run_as_artifact_owner.sh` 不在这三个目录里，仍从 worktree 调用（本线不改它）。
4. **作业里的 git 检查**：快照里没有 `.git`，作业镜像里也没有 git。E3VS 的 PREREG 检查改用 `vpd_snapshot_file`（`python -m vpd.snapshot file <路径>`）：
   - 作业里的副本必须等于快照提交里的文件：先比 blob；有 git 时再和真实 worktree 的 `git show <commit>:<路径>` 比；
   - 输出最后一次改动它的提交，和以前 `git log -1` 的结果一样。
   - 记录用的 git 事实在快照里改从 `SNAPSHOT.json` 取（直接跑 worktree 时不变）：
     - WalkerBench 评测 manifest 的 `extra.vpd.git`：commit 取快照的提交，dirty 为 false，`read` 为 `SNAPSHOT.json`。所以 `smoke_vpd.py` 的 manifest_fields 检查（commit 不能为空）在快照作业里也能过。
     - 训练 meta.json 的 `git_head` 取快照的提交，`vpd_dirty` 为 false，`git_read` 为 `SNAPSHOT.json`。
     - E3VS 评测 manifest 的 `git`：commit 取快照的提交；`prereg_commit` 取作业写的 `$RUN/prereg_commit.txt`，没有就是 null。
     - E3VS `code_meta`（`vpd/e3vs/__init__.py`）的 `worktree_head`：读不到 `.git` 而且是快照时，取快照的提交，并加 `"worktree_head_source": "snapshot"`（第三轮 1c；以前在快照里是 null）。没有哪个 E3VS 汇总拿 `code_meta` 在起点之间比。
5. **其他**：`python -m vpd.snapshot verify` 核对整个快照没有被改动（根目录的 `ply_gaussian_loader.py` 也在内）。`--no-snapshot` 只给测试和应急用，会记在 vpd.json 里（`"no_snapshot": true`）。
   - **E3VS 作业不能用 `--no-snapshot` 跑。** 不在快照里时，prereg 检查（`e3vs_prep.sh` 的 `prereg_guard`、`e3vs_start.sh` 的 `guard`）要用 git，可作业镜像里没有 git，所以检查一定不过，作业会在开头停下。
6. **代码指纹**：快照不改变代码指纹（训练的 CODE_FILES、评测的 code_sha256），因为字节一样。不过改快照机制时改到的文件会让指纹变：
   - 第三轮第一批改了 `vpd/paths.py` 和 `train/n2_train_vpd.py`，都在训练的 CODE_FILES 里，训练指纹从 `63a3a8b92ce7`（试点的适配器）变成 `9b2d44d2a0c6`（提交 ca803a48）。
   - 第三轮 1b 改了 `train/n2_train_vpd.py` 的 `git_head`，训练指纹又变一次，变成 `0865dd75f1ee`（按 `code_fingerprint` 同样的算法算的）。还没有适配器是用新代码训的。
   - 评测的 code_sha256 包括 `walkerbench/run_vpd.py` 本身，所以 1b 之后的评测 manifest 里它的 sha256 也不同。这个值只记录；检查只要求它不为空，没有哪个检查拿它和已有输出比。
