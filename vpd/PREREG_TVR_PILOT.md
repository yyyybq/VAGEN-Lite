# TVR 试点闸门：事先登记（PREREG_TVR_PILOT）

- 状态：**已冻结**（2026-11-11，填 `tiers.n_top` = ALL、确认 `render.eval_envs` = 5 的那次提交，见下面"冻结"一条；`data.*` 在 f393095e 填好）。冻结以后的任何改动只能记进第 17 节的偏差日志。冻结前这里写的是："草稿 — 数据作业之后填 data.*，校准之后填 n_top 和 eval_envs，填完的那次提交即冻结，必须在冒烟（msmoke）之前。"
  - 本文件可以先以草稿提交（例如填 `render.*` 的那次），草稿提交不算冻结：只要第 0.1 节的块里还有"待填"的键，`pilot_cfg.missing()` 就不为空，`pilot_tvr prepare` 就拒绝准备冒烟、训练和评测作业。
  - 冻结 = 填完 `data.*`、`tiers.n_top` 并确认 `render.eval_envs` 的那次提交（git commit，分支 viewpoint-downstream）。冻结以后的任何改动只能记进第 17 节的偏差日志。
- 草稿写于 2026-11-10 16:46 UTC（`date -u`，集群登录节点；wave 2 的 prereg 包）。写的时候还没有任何 TVR 模型输出（DECISIONS G10、G12）。
- wave 2b 修订写于 2026-11-10 20:5x–21:1x UTC（`date -u`，集群登录节点；DECISIONS G13）：确认检验改为终点层面的种子感知检验、A4 的一类错误和功效分别在 ρ = 0 / 1 和 ρ = 0.5 下判、备用设计改为两档种子 1–2、功效模型加任务持续性 λ、校准作业要求全部 `data.*`（附 A：O25）。修订时仍然没有任何 TVR 模型输出（数据作业 tvr-data 在跑，它不读 `power`）。
- wave 2b 第二轮修订写于 2026-11-10 23:4x UTC（`date -u`，集群登录节点；wave-2b 核查之后，冻结之前）：任务持续性 λ 的估计改为不随起点均值变的单因子估计（第 12 节第 7 条；原来的估计其实是起点之间不一致率的一半，起点差 5 个百分点就偏约 ±0.1）；登记 `power.backend` = numpy；第 13 节补登 `gate_tvr` / `pilot_tvr` 在 wave 2b 新加的 G0、冒烟和 A4 条件；第 7 节补 O9 备用值尝试失败后的重判。仍然没有任何 TVR 模型输出。λ 估计的改动要编排在 DECISIONS 里补一条 G13 的修订（G13 原文写"不看起点之间的离散"）。
- wave 2c 修订写于 2026-11-11 01:3x–02:1x UTC（`date -u`，集群登录节点；wave-2b 的独立统计核查之后，冻结之前；DECISIONS G14 说的"核查结果进 G14 补记，实质性的偏松偏差在冻结前加保守的调整"）：λ_used 改为 clip(λ̂ + 0.03, 0, 1)（登记的余量 `power.persistence.margin`，第 12 节第 7 条）；POWER.json 删掉能用闭式还原起点 SR 水平的中间量（以及能从 λ̂ 和 `sigma` 块精确算出、写了只是重复的几个），并写明它给什么、不给什么（第 12 节第 6 条），schema 升到 `vpd.power_sim.tvr/3`；加只报告的设计行 `power.reported_designs`（每个起点 4 个、5 个种子，以及 δ = 0.10）；按核查改了 λ 阈值、功效曲线和第 12 节第 7 条未核实的第 2 条（原来写反了方向）。开始写的时候（01:32Z 的 `ls`）`CAL` 和 `P/<起点>` 都是空目录；02:1x 时校准作业已开始在 `CAL` 下写文件（0W 训练日志、渲染金丝雀、raw 模型的 shake 机制检查），本修订只列过文件名，没有读其中任何结果；这里的设计选择（余量、POWER.json 的白名单、只报告的设计行）在那之前就由编排按合成 p 的核查定了。`P` 下没有试点评测，也没有 POWER.json；核查的数都来自合成 p。这些改动要编排记进 DECISIONS（G14 的补记）。按 wave-2c 审查意见的两轮返修写于 2026-11-11 05:4x–08:3x UTC（`date -u`，集群登录节点；第二轮的改动 08:35Z 装进暂存树）：改了文字、数字的舍入和出处、未核实第 2 条补的一种情形、测试，另给只报告的 4、5 个种子设计加了族错误率行（`reported_designs.type1`，第 12 节第 6 条），余量、白名单和判定规则没变；另写明 A4 不过以后用户选的只报告设计也要满足登记的一类错误界（第 12 节第 6 条、第 13 节 A4）。第二轮的三个评审和整合在 08:3x–11:2x UTC 只读、只测暂存树的副本。出处（编排 2026-11-11 07:30Z 核对，冻结时补正）：第一段（01:3x–02:1x）里校准还没有评测输出；07:30Z 核对时 `CAL/evals` 是空的（第一个校准作业在 train_ALL 时因账号磁盘配额死掉，没到评测；第二个 05:39Z 提交，07:30Z 时还在训 ALL）。第二段的后半，校准评测 07:36Z 起开跑（results.json：ALL 08:07Z、0W 08:35Z、100 09:11Z）；第二轮返修、评审和整合的 5 个代理都没有读过 `OUT/tvr/runs`、校准作业目录或账本（编排扫过它们全部 465 次工具调用的输入，不算每个代理交结果的那一次；它们跑的测试和脚本只用合成根和临时目录，推断），编排 09:16Z 才第一次读校准结果，那时第二轮的改动已经装好，到冻结提交之前没再改（冻结提交又改了其中几处文字：本条的时间和出处、第 12 节第 6、7 条、`power_sim.persistence_hat` 的文档字符串和 `test_power_sim` 的两处注释，依据是合成根、评审结果和代理的工具调用记录，不是校准；冻结提交按校准写的内容只在状态行、下一条、第 0.1、3.2、7、9.3、16 节、O10、O20 和 `pilot_cfg.py` 的 `n_top` 值及两处注释，见下一条）。在那之前读过的模型输出只有 shake 机制检查（raw Q3B，smoke4；编排只读了机制字段，没读成功字段）和第一个校准作业 ALL 训练日志的末尾几行（05:39Z，训练步的 loss、学习率、token 数和耗时，没有成功率）；wave 2c 的全部改动只依据 statcheck 的合成数据（文中另引的 wave-2c 审查和返修的数也都来自合成根）。
- 冻结（本提交；文字写于 2026-11-11 11:3x–13:1x UTC，`date -u`，集群登录节点，12:2x 起的几次改动是按冻结前的两轮审查 wf_4ef19a45-066、wf_3156edf1-709 的补改；校准 tvr-cal pt-js5w1gsc 09:14Z 结束之后、冒烟之前）：`tiers.n_top` 按第 7 节的规则填 ALL（第 0.1 节的块只变了这一个叶子），`render.eval_envs` 按 Q3B dev200 评测的测速确认 5（第 7 节"校准结果"；编排按 2026-11-10 的授权自定，DECISIONS G14a）。本提交新写了第 7 节的"校准结果"一条（三档 k/n、无望停止、测速、O9 的备用值没用、按第 16 节推出的 est_high），并随之改了状态行、第 0.1 节表里 `tiers.n_top`、`render.eval_envs` 两行和表头、第 3.2 节表里 `render.eval_envs` 一行和测速、第 7 节的"待填"、第 9.3 节的服务、第 16 节的预期值、O10、O20，以及 `pilot_cfg.py` 的两处注释（eval_envs、n_top）。同一个提交还改了 wave-2c 第二轮评审和冻结前审查指出的文字问题：上一条的时间、出处和几处措辞（含补记第二轮写明的只报告设计的一类错误界）、第 12 节第 6、7 条的几处说明和出处、`power_sim.persistence_hat` 文档字符串的两处、`test_power_sim` 的两处注释，并让 `test_tvr_prereg` 的断言改判冻结后的状态。没有改任何判定规则、阈值、登记的数或代码逻辑（`.py` 文件里除了 `n_top` 的值只改了注释和文档字符串，测试只有 `test_tvr_prereg` 的断言按冻结后的状态改了）。校准只用了 Q3B 一个起点的绝对数（分离规则）。
- 本文件提交的 git sha 记进每个试点作业的 `prepared.json`，以及 `OUT/tvr/runs/pilot/GATE.json` 的 `prereg_commit`。
- 守门规则（`vpd/tvr/pilot_tvr.py prepare`，契约 5.3.1）：准备 tvr-msmoke、tvr-train-*、tvr-eval-* 之前，必须同时满足：
  1. 本文件已提交，并且没有未提交的改动（`git log -1 -- vpd/PREREG_TVR_PILOT.md` 不为空，`git status --porcelain -- vpd/PREREG_TVR_PILOT.md` 为空；只查本文件）；
  2. `pilot_cfg.check_prereg(<本文件>) == []`：第 0.1 节 `TVR_FROZEN` 块解析出的 JSON 和 `vpd/tvr/pilot_cfg.py` 的 `FROZEN` 逐键相等（同一个解析器，`vpd/tests/test_tvr_prereg.py` 也用它）；
  3. `pilot_cfg.missing(<作业>) == []`；
  4. 训练和评测作业另外要求最新的 `smoke_verdict.json` 为 PASS（第 13 节）；
  5. `tiers.n_top`（手填）等于 `CAL/cal_decision.json` 的 `n_top`（校准规则的输出；N_top = 400 时还要 `with_400` 为真，即 tvr-cal400 之后又跑过 cal-decide；`pilot_tvr.n_top_problems`），并且每个 `data.*` 等于盘上的文件（sha，以及 F、K、剔除数；`pilot_tvr.data_problems`）；
  6. 冒烟和训练作业另外要求 `cal_decision.futility` 为假，或者有用户决定后编排写下的 `CAL/futility_override.json`（第 7 节）。

  作业开头的 guard 步骤（`pilot_tvr guard`）在作业里把第 2–6 条再查一遍；第 1 条在作业里由 `prereg_committed` 步骤对代码快照查（`vpd_snapshot_file`，快照里没有 git）。
- 依据（按优先级）：
  1. 用户第三轮说明 `OUT/plan/ROUND3.md` 第 2–6 节（下称"简报"）；
  2. `OUT/plan/DECISIONS.md` 的 G1–G3、G10（a–j）、G11（探针结果）、G12（采用 wave-2 契约）、G13（wave-2 核查后的决定：确认检验、A4 的 ρ 规则、备用设计、任务持续性）、G14（wave 2b 并入、G13 的 λ 估计修订、`data.*` 填值、tvr-cal；它说独立统计核查的结果进 G14 的补记，wave 2c 据此修订）、G14a（G14 的补记：wave 2c、校准结果和冻结），以及沿用自 WalkerBench 的 B3、B4、B5、B7、B9、B10、B16；
  3. wave-2 接口契约 `OUT/scratch/orch/r3/wave2/contract/CONTRACT_wave2.md` 第 2 版（sha256 前缀 3fa721c558be，下称"契约"）；
  4. 探针作业 vpd-tvr-probe（pt-s9d5txr3）的 `OUT/jobs/tvr-probe-202611101501/out/summary.json`。
- 写法：数字后面括号里是出处。没核实的标"未核实"，推断标"推断"。编排按 2026-11-10 的授权自定的设计选择（简报第 4 节）都注明 DECISIONS 的条目（G10、G12、G13、G14、G14a）。
- 文中路径的简写：
  - `OUT` = `/mnt/umm/users/yinbaiqiao/eval_external_assets/vpd`；
  - `WT` = `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd`（worktree；作业从 `RUN/code` 快照跑，DECISIONS G1）；
  - `REPO` = `OUT/tvr/repos/TVRBench`（只读，commit ad296bb7）；
  - `P` = `OUT/tvr/runs/pilot`（试点根目录，`P/<起点>/{adapters,evals,parity,canary}`）；
  - `CAL` = `OUT/tvr/runs/explore/cal/Q3B`（校准，EXPLORATORY）；`SM` = `OUT/tvr/runs/smoke/<作业名>`（冒烟）。

---

## 0. 结论先说：闸门一览

试点只回答一件事：TVR（目标视角复现，TVRBench）能不能当下游微调任务的尺子。具体是：流程有效、学得会、格式有效、没有饱和、功效够；"用到了图"只报告。**闸门不看起点之间的差。**

| 闸门 | 一句话 | 阈值（详见第 13 节） | 不过时怎么办 |
|---|---|---|---|
| **G0** 机制 | 流程本身是对的 | 第 13 节的 17 条全过：每次评测有效（500 题、0 缺）；训练和评测模板一致；每帧 222 token（220 个 image pad）；渲染档案、字体、L、E 等于冻结值；0W 代码 sha 相同；kin_env 等于 ThorEnv；合并 = LoRA（每次合并评测）；渲染一致（容差规则，不比 sha）；示范重放成功（K ≥ 1,520）；独立复算一致 | 工程问题，不算科学失败。修好后重跑受影响的步骤，记第 17 节 |
| **G1** 学得会 | 至少一个起点最高档学会了 | 同一个起点：最高档（`N_top`）主臂 c 在 500 题上的 Wilson 95% 下界 **>** max(F, 本起点 0W 主臂 c 的成功率) | TVR 判为"按本配方学不会"，不进确认轮；要不要改设计由用户定 |
| **G2** 格式 | 约束解码下格式全对 | 每个起点的每次 c 臂和 blind 臂评测：100% 回复 `re.fullmatch` 正则，没有日志的题为 0 | 先按 G0 查实现；不是实现问题就不进确认轮，交用户 |
| **G3** 没饱和 | 最高档没有贴近专家 | 每个起点最高档 c 臂成功率 **≤ 80%**（≤ 400/500），点估计 | 判为"已饱和"，**不自动下移档位**，交用户 |
| **A3** 用到了图（只报告） | 给图比不给图好 | 同一适配器、同 500 题：最高档 c 臂对 blind 臂，精确双侧二项检验；b > c 且 p < 0.05 记为"给图更好" | 只报告，不卡闸门（简报第 2 节 A3："E3VS 和 TVR 只报告"） |
| **A4** 有量程 | 登记效应量下功效够 | G0–G3 都过以后才跑：`power_sim`（终点层面的种子感知整组 t 检验；模型含任务持续性 λ_used），在 σ_used、λ_used、两档种子偏移相关 ρ = 0.5 下，两个对比 8 个百分点时 Holm 功效 ≥ 0.80，且 ρ = 0 和 ρ = 1 下模拟的族错误率都 ≤ 0.06 | "功效不够"（`power_insufficient`），不进确认轮，交用户。加了任务持续性以后这个结果很可能出现（第 12 节第 7 条） |

其他规则：
- 试点用种子 **1001**（同时管数据顺序和训练）。确认轮用种子 1、2、3。另有 Q3B 的种子重复（seed 1002，0W + 100 档），只用来估种子方差和任务持续性 λ（附 A：O4、O25）。
- A4 的功效模型含任务持续性 λ（同一个起点对同一道题，不同种子往往同样会或同样不会；第 12 节第 7 条），**λ_used = clip(λ̂ + 0.03, 0, 1)**：0.03 是登记的余量（`power.persistence.margin`），抵消 λ̂ 剩下的偏松偏差（第 12 节第 7 条）。独立核查（wave-2b statcheck；合成 p，σ = 0.03，ρ = 0.5，8 个百分点，每格 2,000 次，种子 0）里，Q3S − E2 的功效在 λ = 0 / 0.005 / 0.01 / 0.02 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5 / 0.8 时是 0.813 / 0.799 / 0.800 / 0.793 / 0.781 / 0.767 / 0.748 / 0.720 / 0.662 / 0.597（蒙特卡洛标准误约 0.009）：λ 约 0.005–0.01 起功效就低于 0.80，λ = 0 时的 0.813 也只比 0.80 高约 1.5 个蒙特卡洛标准误（0.013 / 0.0087，`power_curve.json`）；族错误率没有问题（同一组合成 p 上 0.039–0.048）。这些是合成 p 上的数，对真实试点是推断。所以 A4 判 `power_insufficient`、交用户定，是登记时就说明的很可能的结果，不是闸门出错。λ 小时 λ̂ 的标准差约 0.10–0.14，A4 的结果主要由估计噪声决定：按核查，真 λ = 0 时 A4 通过的概率约 0.34–0.44（不加余量时约 0.5；statcheck `tables.txt` 的 OC 表；推断）。为了让用户在 A4 不过时有依据，POWER.json 另报主设计每个起点 4 个、5 个种子和 3 个种子 δ = 0.10 的功效，以及 4 个、5 个种子在 ρ = 0 和 ρ = 1 下的族错误率（`reported_designs`，只报告，不卡闸门；用户在 A4 不过以后选的设计也要满足 ≤ 0.06 的一类错误界；第 12 节第 6 条、第 13 节 A4）。
- 试点数据不进确认分析（pilot data are excluded from the confirmatory analysis）。唯一的例外是第 12、13 节的功效模拟：它用试点的逐题成功率**决定要不要上确认轮**，不当作证据。
- 试点三个起点的数字只列绝对值，不算差，不拿来挑任务（简报第 3 节"调配方和比较起点要分开"）。每个起点的绝对数只在闸门报告（GATE.json / GATE.md 的 k/n）里；POWER.json 不给任何起点的 SR 水平，也不给能只从 POWER.json 用闭式算出起点之间差的值（统计上能部分预测，推断；第 12 节第 6 条）。
- 校准（只用 Q3B、只在 dev200 上）之后有一道**无望停止**：Q3B 最高档 dev200 c 臂的 Wilson 下界 ≤ dev200 的盲下限时，不准备冒烟和训练，先问用户（第 7、13 节；附 A：O21）。
- 所有试过的闸门，过没过都列进最终报告，连同 WalkerBench 和 E3VS 的。

### 0.1 冻结常量块（TVR_FROZEN）

下面这个块是 `vpd/tvr/pilot_cfg.py` 的 `FROZEN`（`pilot_cfg.as_json()` 的输出：`indent=1`、键排序；元组写成列表）。两边必须逐键相等，`vpd/tests/test_tvr_prereg.py` 和 `pilot_tvr prepare` 都用 `pilot_cfg.check_prereg` 检查。每次填值都**在同一个提交里**同时改 `pilot_cfg.py` 和这个块（契约第 3 节 fill rule）。

<!-- TVR_FROZEN_BEGIN -->
```json
{
 "bootstrap": {
  "cluster": "scene",
  "resamples": 10000,
  "seed": 0,
  "use": "descriptive only"
 },
 "budget": {
  "job_est_high_max_h": 5.5,
  "round_baseline_gpuh": 8.8404,
  "round_cap_gpuh": 200.0
 },
 "categories": {
  "LR-easy": [
   "procthor",
   "easy"
  ],
  "LR-hard": [
   "procthor",
   "hard"
  ],
  "SR-easy": [
   "ithor",
   "easy"
  ],
  "SR-hard": [
   "ithor",
   "hard"
  ]
 },
 "data": {
  "F_eval500": 0.0,
  "F_source": "stop0",
  "bfs_dev200_sha256": "82184d170faa38652118990b84a3c8553a999f576047c2c4fd4d0bbc9502efb6",
  "bfs_eval500_sha256": "0e1b0bb865e937bc4a2179f9741742379d403c0784cda92696d1458780f6bb0e",
  "demos_all_sha256": "cf2f702c84bfba042233fc5921ac5a4b5442093735455c46cb945f460b76b897",
  "dev_selection_sha256": "75e64d97e5864d9780c1d925d561e4ccc1e0d126013f946a86679ee87087641e",
  "dropped": 5,
  "floors_eval500_sha256": "5cc85b245c32073b5bd91ff7ddd19342d9491dc6c0502200ca47648f123c4210",
  "houses_sha256": "231089c1b508d0b29ea3f88356639889c3f59f71cfa1477163c4d305c63480ea",
  "kept": 1595,
  "reach_index_sha256": "73dd8ab6d18fd07427109f72c69bfb41301a056fcc0722bcdef16b264adbc5ac",
  "replay_manifest_sha256": "9f826c0421d2d5eb5defc5c2496776baeefcf083b2610d3f253a3afaa880484f",
  "tiers_manifest_sha256": "91f1321db81285a4f8dd7424d3ae4864a6b3961dfc7f3cae8cf837396476122a"
 },
 "dev": {
  "max_per_scene_category": 2,
  "n": 200,
  "per_category": 50,
  "smoke_per_category": 2,
  "source": "rl.json"
 },
 "floors": {
  "fixed_k_max": 39,
  "median": "statistics.median_low of the non-Stop action count per category over the 1600 public sequences",
  "random_seeds": [
   0,
   1,
   2
  ]
 },
 "gates": {
  "futility": "Q3B ALL (or 400) dev200 c Wilson LB <= floors_dev200 F -> stop and ask the user",
  "kin_eval": {
   "task_mismatch_max_frac": 0.01,
   "valid_list_mismatch_max": 0
  },
  "mcnemar_alpha": 0.05,
  "merge_parity": {
   "max_mean_abs_dce": 0.005,
   "min_reply_agreement": 0.99,
   "n_demos": 20,
   "tiers": "every merged pilot eval (0W, 100, TOP, 100s1002)"
  },
  "min_kept": 1520,
  "saturation": 0.8,
  "z": 1.96
 },
 "power": {
  "alpha": 0.05,
  "backend": "numpy",
  "deltas": [
   0.05,
   0.08,
   0.1
  ],
  "design": {
   "contrasts": [
    [
     "Q3S",
     [
      "E2"
     ]
    ],
    [
     "Q3RD",
     [
      "QRL-s1",
      "QRL-s2"
     ]
    ]
   ],
   "seeds": {
    "100": [
     1,
     2,
     3
    ],
    "TOP": [
     1,
     2,
     3
    ]
   },
   "starts": [
    "Q3B",
    "Q3S",
    "E2",
    "Q3RD",
    "QRL-s1",
    "QRL-s2"
   ],
   "tiers": [
    "100",
    "TOP"
   ],
   "weights": [
    0.5,
    0.5
   ]
  },
  "effect_model": "realized SR shift delta per tier, mapped to a pre-clip shift by calibrate_shift",
  "fallback": {
   "rule": "if est_high > 100 GPU-h: fallback iff fallback_pass, else stop and ask",
   "seeds": {
    "100": [
     1,
     2
    ],
    "TOP": [
     1,
     2
    ]
   }
  },
  "holm": true,
  "n_sims": 2000,
  "outcome": {
   "inconclusive": "otherwise",
   "min_effect": 0.05,
   "no_support": "Holm-rejected with the treated start worse, OR the 95% t-CI upper bound < min_effect",
   "overall": "\u652f\u6301 iff >= 1 contrast support and none Holm-rejected the other way; \u4e0d\u652f\u6301 iff both no_support; else \u65e0\u5b9a\u8bba",
   "support": "Holm-rejected with the treated start better"
  },
  "p_source": "per task and tier: mean over the three pilot starts' primary-arm outcomes",
  "persistence": {
   "check_reps": 400,
   "estimator": "D = mean_i (y1_i - y2_i)^2; kappa = E[|r1 - r2| | dbar, S] under r1 - r2 ~ N(0, 2 sigma_used^2), dbar ~ N(r1 - r2, S) (dbar, S, sigma_used as seed_sd): b = 2 sigma_used^2 / (2 sigma_used^2 + S), a = b dbar, v = b S, kappa = sqrt(v) sqrt(2 / pi) exp(-a^2 / (2 v)) + a (1 - 2 Phi(-a / sqrt(v))); m2 = E[(r1 - r2)^2 | dbar, S] = a^2 + v; l_i = (y1_i + y2_i) / 2, L = mean_i l_i; y_a, y_b = the other two pilot starts' rows; C = covariance over tasks (ddof 1); V = the delete-one-scene jackknife (G V - (G - 1) mean_g V_-g) of C(l, y_a) C(l, y_b) / C(y_a, y_b); B = L (1 - L) - V; R = 2 B + m2 / 2; lambda_hat = (R - D) / (R - kappa)",
   "margin": 0.03,
   "range": [
    0.0,
    1.0
   ],
   "rule": "lambda_used = clip(lambda_hat + margin, range) (margin = the registered allowance for the residual liberal bias of lambda_hat); lambda_hat undefined (C(y_a, y_b) <= 0 in the full or a delete-one-scene sample, or R <= kappa) or replicate missing -> range[1]",
   "sensitivity": [
    0.0,
    0.8
   ],
   "source": "the seed replicate (tiers.replicate: seed 1001 vs its replicate seed, same start, replicate tier, c arm, same tasks); the other pilot starts' c-arm outcomes at that tier (seed 1001) only through covariances over tasks, for the spread of the task pattern at the replicate start's scale (one-factor model: every start its own mean and scale; lambda_hat is invariant to start mean SRs; no between-start difference is computed)"
  },
  "power_counts": "Holm-rejected AND treated better, other contrast at effect 0",
  "primary_delta": 0.08,
  "reported_designs": {
   "designs": {
    "main_delta_10pp": {
     "delta": 0.1,
     "seeds": {
      "100": [
       1,
       2,
       3
      ],
      "TOP": [
       1,
       2,
       3
      ]
     }
    },
    "main_seeds_4": {
     "delta": 0.08,
     "seeds": {
      "100": [
       1,
       2,
       3,
       4
      ],
      "TOP": [
       1,
       2,
       3,
       4
      ]
     }
    },
    "main_seeds_5": {
     "delta": 0.08,
     "seeds": {
      "100": [
       1,
       2,
       3,
       4,
       5
      ],
      "TOP": [
       1,
       2,
       3,
       4,
       5
      ]
     }
    }
   },
   "tier_corr": 0.5,
   "type1": {
    "designs": [
     "main_seeds_4",
     "main_seeds_5"
    ],
    "tier_corr": [
     0.0,
     1.0
    ]
   },
   "use": "report only: never gated, the verdicts do not read them"
  },
  "seed": 0,
  "seed_model": "realized-scale shift r_SsT = sigma * (sqrt(rho) z_Ss + sqrt(1 - rho) z_SsT), z iid N(0, 1), per (start, seed, tier); rho = tier_corr (same seed id across tiers); mapped by calibrate_shift; y = 1[W < clip(p_i(T) + c, 0, 1)], W = V_Si (one uniform per start and task, shared by its seeds and both tiers) with probability sqrt(lambda), else a fresh uniform; lambda = persistence lambda_used",
  "seed_sd": {
   "a_priori": 0.03,
   "estimator": "e_i = y1_i - y2_i; d = mean e; S = sum (e_i - d)^2 / (n (n - 1)); sigma_hat = sqrt(max(0, (d^2 - S) / 2)); sigma_used = max(0.03, sigma_hat)",
   "mode": "replicate_max_a_priori",
   "sensitivity": [
    0.05
   ]
  },
  "target_power": 0.8,
  "test": "seed_aware_endpoint_t",
  "test_def": "D = mean_i d_i (per-task endpoint differences; e_Si = sum_T w_T mean_s y_SsTi, seeds paired across tiers); V = CR1 scene-cluster var of mean(d) + k_contrast * sigma2_E_hat / n_seeds; sigma2_E_hat = max(0, mean over starts of [var_s(E_s) - mean_i var_s(e_si) / n]) (ddof 1; e_si = sum_T w_T y_sTi, E_s = mean_i e_si); t reference, df = n_starts * (n_seeds - 1) (12; fallback 6); two-sided; k = sum of squared contrast coefficients (2 for Q3S-E2, 1.5 for Q3RD-mean(QRL))",
  "tier_corr": {
   "power": 0.5,
   "reported": [
    0.0,
    1.0
   ],
   "type1": [
    0.0,
    1.0
   ]
  },
  "type1_max": 0.06
 },
 "protocol": {
  "actions": [
   "MoveAhead",
   "MoveBack",
   "MoveLeft",
   "MoveRight",
   "RotateRight",
   "RotateLeft",
   "LookUp",
   "LookDown",
   "Stop"
  ],
  "blind_concurrency": 16,
  "chat_template_kwargs": null,
  "gpu_memory_utilization": 0.85,
  "history": "full",
  "image_grid": [
   1,
   22,
   40
  ],
  "image_pad_tokens": 220,
  "image_tokens": 222,
  "max_images": 96,
  "max_model_len": 32768,
  "max_model_len_fallback": 128000,
  "max_pixels": 1024000,
  "max_steps": {
   "ithor": 30,
   "procthor": 40
  },
  "max_tokens": {
   "blind": 16,
   "c": 16,
   "u": 64
  },
  "png": "PIL save PNG defaults",
  "reach": "frozen_per_task_start_pose",
  "regex": "Action: (MoveAhead|MoveBack|MoveLeft|MoveRight|RotateRight|RotateLeft|LookUp|LookDown|Stop)",
  "secondary_parser": "eval_va.parse_action",
  "server": {
   "enforce_eager": true,
   "generation_config": "vllm",
   "prefix_caching": "vllm default (on)",
   "seed": 0,
   "structured_outputs_backend": "xgrammar"
  },
  "temperature": 0.0,
  "top_k": -1,
  "top_p": 1.0,
  "versions": {
   "ai2thor": "5.0.0",
   "transformers_serve": "4.57.6",
   "transformers_train": "4.57.0",
   "vllm": "0.11.0",
   "xgrammar": "0.1.25"
  }
 },
 "public": {
  "hf_revision": null,
  "sft_train_json": "OUT/tvr/hf_meta/tvr-sft-va/sft_train.json",
  "sft_train_json_sha256": "cbf6f36d524955f3aaa4f9f6e91f876b50d174c09f8f4a6a69f5c3a85f00e13d"
 },
 "recipe": {
  "accum": 8,
  "alpha": 32.0,
  "attn": "sdpa",
  "dropout": 0.05,
  "epochs": "auto",
  "epochs_rule": "max(3, ceil(240/N))",
  "history_k": 0,
  "lr": 0.0001,
  "max_seq_tokens": 40000,
  "num_workers": 3,
  "rank": 16,
  "threads": 2,
  "trainer_code_sha256_prefix": "0865dd75f1ee",
  "warmup": 0.03
 },
 "render": {
  "anti_aliasing": 4,
  "badge_font_sha256": "b184b89e3c1075f22f6b71575b6fc20d4972b3cfd3b23322ca6fd596dcaef167",
  "build": "thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917",
  "compare_verdict": "same_rendering",
  "consistency_rule": {
   "exact": [
    "states",
    "valid",
    "user_texts",
    "action_success"
   ],
   "frame_cache": "forbidden",
   "mask": "none (same pinned badge font on both sides)",
   "psnr_min_db": 40.0,
   "ssim": "probe_lvp.SSIM_DEFINITION",
   "ssim_min": 0.99
  },
  "deterministic": false,
  "eval_envs": 5,
  "fov": 60,
  "grid_size": 0.25,
  "height": 360,
  "lp_threads": 2,
  "lvp_profile": "envlib",
  "platform": "CloudRendering",
  "probe_run": "OUT/jobs/tvr-probe-202611101501",
  "probe_summary_sha256": "430e34ff2ec6e695c622b5bae135707c3a27bbfb921ec74aab005fcb711a5e5f",
  "quality": "Ultra",
  "recommendation_field": "summary.recommended",
  "recommended_measured": [
   6,
   2
  ],
  "renderer": "lavapipe",
  "replay_envs": 5,
  "rotate_step": 45,
  "ssim_definition_sha256": "9f5517b54436155b3aaec2dfc66b41e3496a37f1ef2e59276739454a1f6fc5eb",
  "width": 640
 },
 "salts": {
  "dev": "tvr-dev-v1|",
  "rand_floor": "tvr-floor-rand-v1|",
  "tier": "tvr-tier-v1|",
  "zero_w": "tvr-0w-v1|"
 },
 "schema": "vpd.tvr.pilot_cfg/1",
 "seed": 1001,
 "starts": [
  "Q3B",
  "Q3S",
  "E2"
 ],
 "tiers": {
  "all_dev_sr_max": 0.7,
  "calibration": [
   "0W",
   "100",
   "ALL"
  ],
  "calibration_extra": "400",
  "n_top": "ALL",
  "replicate": {
   "seed": 1002,
   "start": "Q3B",
   "tier": "100"
  }
 },
 "tvrbench": {
  "commit": "ad296bb7ff880b8a0758187edc3ddc97eff92791",
  "files_sha256": {
   "data/procthor-10k/train.jsonl.gz": "ee3c4aa14b4d8f0895fecfb5fdaca59395427ca1018b2f9aeeedbc61e5824587",
   "data/scene_splits.json": "a36d33acd32a88a984faef7b8f3dd1979972e27822ff72083c593124ede09f44",
   "data/tasks/eval.json": "aff32430551739a3c6c1336e7e7f75a174b73f337e273f5a5a7dee736b7d12c8",
   "data/tasks/rl.json": "4406ec551d9b1d3a389abf2a239220f4a2a75016f82ee8f0ae22b38a1ed42d15",
   "data/tasks/sft.json": "09f4a538974f335505b8721f57afb8d8d35d3cc7a184d8f81ca372e6b29640ac",
   "scripts/eval_va.py": "d27faf8ed8a4ff798eadf5dbe8a95ff17402c04fd26684c1d2e1e545f6b82fb9",
   "tvrbench/env/thor_env.py": "d4ac9688aa7a1c3dca9b42140cc2fc804ca163d7950c7d8ae94ad5c5659349c4",
   "tvrbench/evaluation/metrics.py": "2d91bbc87634bf70d49cd8dba7a1e989a28c49c24bdde80b84f9f1c492171fab",
   "tvrbench/online_rl/thor_view_env.py": "7e3c46c285a0f6344b811a0a5603c206753b3e7ace26a3c66500e29ba6c0072c"
  }
 },
 "zero_w": {
  "expected_epochs": 12,
  "expected_optimizer_steps": 30,
  "n": 20,
  "per_category": 5
 }
}
```
<!-- TVR_FROZEN_END -->

块里各个顶层键在本文件的哪一节解释：

| 键 | 节 | 键 | 节 |
|---|---|---|---|
| `schema` | 0.1（块的版本 `vpd.tvr.pilot_cfg/1`） | `tvrbench` | 2 |
| `public` | 2、5 | `render` | 3 |
| `protocol` | 9 | `starts` | 4 |
| `seed` | 0、8 | `salts` | 5–7、11 |
| `categories` | 1 | `zero_w` | 6 |
| `dev` | 2、7 | `tiers` | 7 |
| `recipe` | 8 | `floors` | 11 |
| `gates` | 13 | `bootstrap` | 12 |
| `power` | 12、15 | `budget` | 16 |
| `data` | 2、5、11 | | |

块里冻结前值为 `null`（Python 的 None）的键，以及冻结前暂定、校准后确认的值（冻结时要填的都已填好；还是 `null` 的只有两个可选键）：

| 键 | 现在 | 谁、什么时候填 | 来源 |
|---|---|---|---|
| `data.dev_selection_sha256` | 已填：`75e64d97e586…`（tvr-data-202611102015） | 编排（契约 7.1 之后） | `OUT/tvr/split/dev200/selection.json` 的 sha256 |
| `data.houses_sha256` | 已填：`231089c1b508…`（tvr-data-202611102015） | 编排 | `OUT/tvr/houses/houses.json` |
| `data.reach_index_sha256` | 已填：`73dd8ab6d18f…`（tvr-data-202611102015） | 编排 | `OUT/tvr/reach/v1/index.json` |
| `data.replay_manifest_sha256` | 已填：`9f826c0421d2…`（tvr-data-202611102015） | 编排 | `OUT/tvr/replay/sft1600_lvp/manifest.json` |
| `data.demos_all_sha256` | 已填：`cf2f702c84bf…`（tvr-data-202611102015） | 编排 | `OUT/tvr/demos/sft1600_lvp/demos_all.jsonl`（= 其 manifest 的 `demos_all_sha256`） |
| `data.kept` | 已填：`1595`（tvr-data-202611102015） | 编排 | demos manifest 的 `kept`（保留的示范条数 K） |
| `data.dropped` | 已填：`5`（tvr-data-202611102015） | 编排 | demos manifest 的 `dropped`（剔除条数；名单在 `dropped.json`） |
| `data.tiers_manifest_sha256` | 已填：`91f1321db812…`（tvr-data-202611102015） | 编排 | `OUT/tvr/tiers/v1/manifest.json` |
| `data.bfs_eval500_sha256` | 已填：`0e1b0bb865e9…`（tvr-data-202611102015） | 编排 | `OUT/tvr/bfs/eval500_opt.json` |
| `data.bfs_dev200_sha256` | 已填：`82184d170faa…`（tvr-data-202611102015） | 编排 | `OUT/tvr/bfs/dev200_opt.json` |
| `data.floors_eval500_sha256` | 已填：`5cc85b245c32…`（tvr-data-202611102015） | 编排 | `OUT/tvr/floors/floors_eval500.json` |
| `data.F_eval500` | 已填：`0.0`（tvr-data-202611102015） | 编排 | `floors_eval500.json` 的 `F.rate` |
| `data.F_source` | 已填：`stop0`（tvr-data-202611102015） | 编排 | `floors_eval500.json` 的 `F.source` |
| `tiers.n_top` | 已填 "ALL"（校准：Q3B ALL 档 dev200 85/200 = 0.425 ≤ 0.70；第 7 节"校准结果"） | 编排（`cal_decision.json`） | 第 7 节的规则 |
| `render.eval_envs` | 已填 5，校准后确认（Q3B dev200 图像臂评测步速最低 5.20 步/秒，高于 4 步/秒；第 7 节"校准结果"） | 编排（只用 Q3B 的校准测速） | 第 3 节 |
| `public.hf_revision` | `null`，未核实；允许保持 `null`（`pilot_cfg.OPTIONAL_NONE`） | — | 第 2 节 |
| `protocol.chat_template_kwargs` | `null`，**按设计**：我们不发送 chat_template_kwargs（`OPTIONAL_NONE`） | — | 第 9 节 |

`render.*` 的其余各项已在探针之后填好（第 3 节）。`data.*` 填写时要用 `sha256sum` 在登录节点复核文件，再提交。13 个 `data.*` 在数据作业之后**一次提交**填完；校准作业 tvr-cal、tvr-cal400 要求全部 13 个都已填（`pilot_cfg.REQUIRED`；G10(a) 要求 K 和剔除名单在任何训练之前登记，而校准就要训 Q3B）。

---

## 1. 目的和范围

1. **试点是什么**：一个闸门试点。它检验 TVR 能不能当"带视角变化的下游微调任务"：示范重放成功、渲染一致，模型在 1,580 条左右的公开专家示范上 LoRA 微调后能学会（高于盲走下限，也高于本起点的格式热身 0W），格式有效，没有饱和，确认轮功效够（简报第 2 节 A）。
2. **任务**：TVRBench（arXiv 2606.01247，repo commit ad296bb7），AI2-THOR 5.0.0 里从起始位姿走到目标视角并停下；成功 = 发出 Stop 且位姿与目标精确一致（REPO `scripts/eval_va.py:202`，`tvrbench/evaluation/metrics.py` 的 is_exact_match：xz 距离 < 0.01 m、转角误差 < 1°、俯仰误差 < 1°）。
3. **类别**（`categories`）：iTHOR 单房间 easy / hard = `SR-easy` / `SR-hard`；ProcTHOR 多房间 easy / hard = `LR-easy` / `LR-hard`（REPO README 的叫法；论文叫 Single / Multi）。评测集每类 125 题（第 2 节）。
4. **闸门不看起点之间的差**：`gate_tvr.py` 不计算、也不打印任何"起点 A − 起点 B"；闸门只用每个起点自己的绝对数字，和同一批题上的下限、本起点的 0W 比（简报第 3 节）。
5. **试点数据不进确认分析**，例外见第 0 节。
6. **选择只用 Q3B 或和起点无关的量**：档位标定、eval_envs 的确认、无望停止都只看 Q3B，并且只在 dev200（rl.json 抽出的开发集）上；eval.json 的 500 题从不用于任何选择（DECISIONS G3(e)、G10(d)）。
7. **报告义务**：所有试过的闸门，过了的和没过的，都列进最终报告。

---

## 2. 冻结的数据

### 2.1 TVRBench（`tvrbench`）

- 仓库：`REPO`，commit `ad296bb7ff880b8a0758187edc3ddc97eff92791`，只读。
- 文件哈希（契约 F9：2026-11-10 15:4x 在登录节点 `sha256sum`；数据作业的 guard 步骤每次重核，不等就停）：

| 文件 | sha256 |
|---|---|
| `tvrbench/env/thor_env.py` | `d4ac9688aa7a1c3dca9b42140cc2fc804ca163d7950c7d8ae94ad5c5659349c4` |
| `tvrbench/online_rl/thor_view_env.py` | `7e3c46c285a0f6344b811a0a5603c206753b3e7ace26a3c66500e29ba6c0072c` |
| `tvrbench/evaluation/metrics.py` | `2d91bbc87634bf70d49cd8dba7a1e989a28c49c24bdde80b84f9f1c492171fab` |
| `scripts/eval_va.py` | `d27faf8ed8a4ff798eadf5dbe8a95ff17402c04fd26684c1d2e1e545f6b82fb9` |
| `data/tasks/eval.json` | `aff32430551739a3c6c1336e7e7f75a174b73f337e273f5a5a7dee736b7d12c8` |
| `data/tasks/sft.json` | `09f4a538974f335505b8721f57afb8d8d35d3cc7a184d8f81ca372e6b29640ac` |
| `data/tasks/rl.json` | `4406ec551d9b1d3a389abf2a239220f4a2a75016f82ee8f0ae22b38a1ed42d15` |
| `data/scene_splits.json` | `a36d33acd32a88a984faef7b8f3dd1979972e27822ff72083c593124ede09f44` |
| `data/procthor-10k/train.jsonl.gz` | `ee3c4aa14b4d8f0895fecfb5fdaca59395427ca1018b2f9aeeedbc61e5824587` |

- 任务数（prereg_confirm 的 tvr_stats.py 对 REPO 任务文件的复算）：
  - eval.json 500 题 = 4 类 × 125，80 个场景（iTHOR 40 + ProcTHOR 40），每场景 2–13 题；
  - sft.json 1,600 题 = 4 × 400，40 个场景；
  - rl.json 4,800 题，120 个场景；
  - 三组场景两两不相交，所以 rl.json 是干净的开发池。
- 没有任何任务的起点就是目标（连 xz 都不同），所以"开局就停"的下限 STOP0 = 0/500（同上）。
- 评测的场景顺序：eval_va 按文件顺序跑（500 题里 484 次换场景，契约 F1）；我们按场景分组跑，因此可达集合必须和任务顺序无关（第 3.4 节）。

### 2.2 公开示范（`public`）

- 文件：`OUT/tvr/hf_meta/tvr-sft-va/sft_train.json`（HF 数据集 TVRBench/tvr-sft-va，经 hf-mirror 下载，简报第 4 节批准），sha256 `cbf6f36d524955f3aaa4f9f6e91f876b50d174c09f8f4a6a69f5c3a85f00e13d`（2026-11-10 16:4x 登录节点 `sha256sum` 复核，= 契约 F9）。
- 内容：1,600 条多轮对话，和 sft.json 一一对应；记录里没有 task_id，task_id 取 `images[0]` 的目录名（契约 F9）。每条 1 个 system（= REPO 的 SYSTEM_PROMPT，942 字符）、每个 user 轮 2 张图（CURRENT + TARGET）、恰好一个 Stop 且在最后；assistant 轮最少 3、中位 12、最多 26，合计 20,735（prereg_confirm、tvr_data_paper）。
- **HF revision：未核实**。`OUT/tvr/hf_meta` 下只有 README.md、dataset_info.json、sft_train.json 和一张 PNG，里面都没有 revision；所以 `public.hf_revision` 保持 `null`。批评者 11-10 经 hf-mirror 看到 README 的 resolve-cache 地址里有 revision 前缀 125d8d7c（critic），它是不是这份 sft_train.json 的 revision 未核实。冻结以 sha256 为准。
- 许可：README 写的是 apache-2.0；ProcTHOR 自己的许可未核实。只作内部研究。
- 我们只用公开文件里的**动作序列**；图像、"Valid actions" 行都在我们的环境里重新渲染、重新计算（第 5 节）。

### 2.3 TVR-dev（`dev`）

- 规则（DECISIONS G10(d)，契约 5.1.2）：从 rl.json 抽 200 题，每类 50 题；每类内按 sha256("tvr-dev-v1|" + task_id) 升序依次取，同一类里同一场景最多 2 题；输出顺序 SR-easy、SR-hard、LR-easy、LR-hard 四块，每块内按哈希序。选中的场景若在 sft 或 eval 里就报错（不会发生，断言）。
- 记录：`OUT/tvr/split/dev200/selection.json`（rl.json 的 sha256、200 个 id、每类 50、场景计数、smoke8 / smoke4）。sha256 → `data.dev_selection_sha256`（已填）。
- 冒烟子集：smoke8 = 每类 dev 顺序的前 2 题；smoke4 = 每类前 1 题（`dev.smoke_per_category` = 2）。sft2 = 探针用过的两条公开样本任务 `ithor_FloorPlan11_easy_t00_s00`、`procthor_procthor_09990_hard_t19_s01`，只用于渲染一致性的金丝雀。
- dev200 只用于校准（第 7 节），**从不用于试点闸门**。

### 2.4 冻结文件哈希表

数据作业（vpd-tvr-data，契约 7.1）跑完以后由编排填进第 0.1 节的块（`data.*`），并用 `sha256sum` 复核；三个起点的试点作业之间这些文件不能变（G0 第 13、14 条检查）。

| 文件 | 键 | sha256 |
|---|---|---|
| `OUT/tvr/split/dev200/selection.json` | `data.dev_selection_sha256` | `75e64d97e5864d9780c1d925d561e4ccc1e0d126013f946a86679ee87087641e` |
| `OUT/tvr/houses/houses.json` | `data.houses_sha256` | `231089c1b508d0b29ea3f88356639889c3f59f71cfa1477163c4d305c63480ea` |
| `OUT/tvr/reach/v1/index.json` | `data.reach_index_sha256` | `73dd8ab6d18fd07427109f72c69bfb41301a056fcc0722bcdef16b264adbc5ac` |
| `OUT/tvr/replay/sft1600_lvp/manifest.json` | `data.replay_manifest_sha256` | `9f826c0421d2d5eb5defc5c2496776baeefcf083b2610d3f253a3afaa880484f` |
| `OUT/tvr/demos/sft1600_lvp/demos_all.jsonl` | `data.demos_all_sha256` | `cf2f702c84bfba042233fc5921ac5a4b5442093735455c46cb945f460b76b897` |
| `OUT/tvr/tiers/v1/manifest.json` | `data.tiers_manifest_sha256` | `91f1321db81285a4f8dd7424d3ae4864a6b3961dfc7f3cae8cf837396476122a` |
| `OUT/tvr/bfs/eval500_opt.json` | `data.bfs_eval500_sha256` | `0e1b0bb865e937bc4a2179f9741742379d403c0784cda92696d1458780f6bb0e` |
| `OUT/tvr/bfs/dev200_opt.json` | `data.bfs_dev200_sha256` | `82184d170faa38652118990b84a3c8553a999f576047c2c4fd4d0bbc9502efb6` |
| `OUT/tvr/floors/floors_eval500.json` | `data.floors_eval500_sha256` | `5cc85b245c32073b5bd91ff7ddd19342d9491dc6c0502200ca47648f123c4210` |

**登记以后不再重新生成这些文件**（数据作业对相同输出是幂等的，对不同输出拒绝覆盖，契约 7.1）。要改就是偏差，记第 17 节。

---

## 3. 环境（`render`）

### 3.1 渲染器和设置

- AI2-THOR 5.0.0，CloudRendering 构建 `thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917`，TVR 环境的 Python 3.10（`OUT/envs/tvr`，`TVR_PY`）。
- **lavapipe（Mesa CPU Vulkan）代替 NVIDIA GPU 渲染**：这是偏差，用户 2026-11-10 批准（简报第 4 节"改用 CPU 渲染的偏差"；DECISIONS G3(a)）。原因：作业容器里 NVIDIA Vulkan 报 DEVICE_LOST；11-10 的 Vulkan 复查在两个池上都没通，本轮不再走 Vulkan（ROUND3 状态；DECISIONS G2）。记进第 17 节。
- 设置和官方评测相同：640×360、quality Ultra、antiAliasing 4、fov 60、gridSize 0.25、转向 45°、俯仰 30°（限 [−30°, 30°]）（REPO thor_env.py、thor_view_env.py；契约第 3 节）。
- 徽标（CURRENT / TARGET 字样）用 TVRBench 自己的 burn_label，字体钉死为 TVR 环境里 matplotlib 自带的 DejaVuSans-Bold.ttf，sha256 `b184b89e3c1075f22f6b71575b6fc20d4972b3cfd3b23322ca6fd596dcaef167`；作业容器里**没有**系统 DejaVu 字体（探针 `font.parent.system_present` = false），所以必须钉死；字体不对 worker 拒绝启动（契约 5.2.1）。

### 3.2 探针冻结的值（探针 pt-s9d5txr3，2026-11-10 15:51Z SUCCEEDED，0.8253 卡时；DECISIONS G11）

| 键 | 值 | 出处 |
|---|---|---|
| `render.probe_run` | `OUT/jobs/tvr-probe-202611101501` | 探针 RUN 目录（只读保留） |
| `render.probe_summary_sha256` | `430e34ff2ec6e695c622b5bae135707c3a27bbfb921ec74aab005fcb711a5e5f` | 2026-11-10 16:4x 登录节点 `sha256sum out/summary.json` |
| `render.lvp_profile` | `envlib`（LD_LIBRARY_PATH 只放 TVR 环境的 lib；第一次就起来，没重试） | `summary.lvp_profile` |
| `render.lp_threads` | 2（每个 Unity 进程的 lavapipe 线程数 L） | `summary.recommended.L` |
| `render.replay_envs` | 5（重放和可达集合导出的并行环境数 E） | `summary.recommended.E` |
| `render.eval_envs` | 5（= replay_envs；校准后确认，第 7 节"校准结果"） | 契约 O10 |
| `render.recommended_measured` | [6, 2]，只记录、不采用 | `summary.recommended_measured` |
| `render.compare_verdict` | `same_rendering` | `summary.compare.verdict` |
| `render.deterministic` | false | `summary.determinism.deterministic` |
| `render.ssim_definition_sha256` | `9f5517b54436155b3aaec2dfc66b41e3496a37f1ef2e59276739454a1f6fc5eb` | sha256(`probe_lvp.SSIM_DEFINITION`)，按只读方式导入 4487309d 的 `vpd/tvr/probe_lvp.py`（文件 sha256 615668d7b119…）算出 |

- **两条推荐**（探针 grid，env-only，每个环境 100 步；编排按 grid_cells 独立复算一致，`scratch/orch/r3/jobs2/probe_facts.md`）：
  - `recommended` = E 5 × L 2（规则 E×L ≤ 11，给 vLLM 和主进程留 ≥ 3 核）：调和平均 9.168 步/秒，实测 cgroup 核数最多 8.584。**采用**（`render.recommendation_field` = `summary.recommended`；run_tvr 的保护也按 E×L ≤ 11）。
  - `recommended_measured` = E 6 × L 2（规则：实测核数 ≤ 11）：11.274 步/秒，10.372 核。只记录。
  - 每步约 0.8–1.0 个核·秒（契约 F4 的表）。有 vLLM 在环路里时，每步多等 0.2–0.4 秒模型，所以评测的总速率预计 5.5–7 步/秒（推断，校准实测）；校准的 dev200 评测实测 5.20–6.48 步/秒（第 7 节"校准结果"）。
- **公开图比对**（DECISIONS G3(d)，规则在看图之前定好）：两条公开样本任务的 27 帧全部通过，最小 PSNR 44.713 dB、最小 SSIM 0.993125（SSIM 定义见 `probe_lvp.SSIM_DEFINITION`，徽标区域 x ≥ 536、y ≤ 31 遮住），判为 `same_rendering`；对照组（位姿错开一步）最大 PSNR 23.2 dB，说明这个度量分得开位姿。**即便判为相同，我们仍然只用自己渲染的图训练**，示范和评测用同一个渲染器（简报第 3 节 TVR）。
- **渲染不是逐位可复现的**（`summary.determinism`，契约 F4）：
  - deterministic = false；跨进程、跨 L、跨动作路径、重载场景之后，帧的 sha256 都可能不同（equal_across_processes / equal_across_L / equal_across_paths / equal_after_reload 全为 false）；只有同一进程、同一位姿、中间不动时重渲一次是逐位相同的（repeat_render_identical = true）。
  - 例：FloorPlan1 的 4 个探针位姿在 32 次渲染里分别有 2 / 31 / 30 / 31 个不同的 sha256；procthor_08101 是 1–3 个。
  - 差多大没有量（探针没存这些帧）：**未核实**。公开图（另一台机器、另一条构建流程）对我们的图都在 44.7 dB 以上，所以我们自己的帧之间的差大概也在这个量级以内（推断）。
  - 后果（规范性）：
    1. 任何一步都不跨进程、跨运行、跨路径比较帧的 sha256；`images_sha256.json` 只是文件完整性记录；
    2. 渲染一致性一律按冻结的容差规则判（`render.consistency_rule`：每帧 PSNR ≥ 40 dB 且 SSIM ≥ 0.99，SSIM = `probe_lvp.SSIM_DEFINITION`，不遮徽标（两边同一字体），另外 states、valid 列表、user 文本、action_success 必须逐项完全相等；`vpd/tvr/render_parity.py`，契约 5.2.6）；
    3. 不建也不用"位姿 → 帧"的缓存（`frame_cache` = forbidden）；每一帧都是 worker 为那一步现渲的；
    4. 重放和评测用同一个 L（2）和同一个档案（envlib）。
- 可达集合在探针里是稳定的：120 个 sft + eval 场景各在两个 worker 里各加载一次，120/120 相同；重载后也相同；GetReachablePositions 每次 ≤ 8 ms（`summary.reach`）。

### 3.3 CPU

- 一个作业 14 个 vCPU。THOR worker 每个 OMP 1、lavapipe 线程 L = 2（`lvp.worker_env`）；vLLM 进程 OMP 4（`VLLMServer.env()`，DECISIONS B9）；作业里其他进程 2（`VPD_THREADS`）；进程池和 loader 数都显式写数字。
- 图像臂 5 个环境 × L 2 = 10 个渲染线程 + vLLM 3–4 + 主进程 0.5–1 ≈ 9–11.5 核（推断，契约第 7 节）；blind 臂不用 THOR，16 个主进程线程（`protocol.blind_concurrency`）。

### 3.4 可达集合：冻结的"按任务、在起点位姿取"的集合（附 A：O1）

- **为什么要冻结**（契约 F1、F2）：
  - ThorEnv 在一个场景加载后第一次需要时才调用 `GetReachablePositions`，并且一直缓存到下次加载场景（REPO thor_env.py:207-218、395-400、437-446）。在 `ThorViewEnv._sync_reset` 里，这次调用发生在把智能体传送到起点之后，所以一个任务若是场景加载后的第一个任务，它用的是**在自己起点位姿上**取的集合，否则用的是前一个任务的集合。
  - `GetReachablePositions` 至少在一个场景里依赖调用时的位姿：sft 任务 `ithor_FloorPlan325_easy_t01_s00` 的起点 (2.5, −0.75) 不在 FloorPlan325 出生位姿取的集合（244 个格子）里（契约 F2）。
  - 我们按场景分组跑，和 eval_va 的文件顺序不同；不冻结的话，示范、评测、kin_env、下限可能对同一题用了不同的集合。
- **定义**：R(任务) = 跑完 TVRBench 自己的 `ThorViewEnv._sync_reset(任务)`（目标已渲染、智能体在起点）、清掉 ThorEnv 缓存之后，`GetReachablePositions` 返回的位置集合；键集合 = {(round(x, 2), round(z, 2))}。对场景加载后的第一个任务，这正是官方的值。
- **怎么取**：数据作业里取一次（`vpd/tvr/reach.py dump`），每个场景在两个不同 worker 上各加载一次、任务顺序相反（A 正序、B 倒序），冻结值取 A；B 只用来记录稳定性。
- **怎么用**：THOR worker 在每次 reset 之后把 ThorEnv 的缓存**注入**为 R(任务)（契约 5.2.1），所以重放、每个评测臂、kin_env、BFS、下限、blind 臂都用同一个集合。worker 仍在每次 reset 时现场调用一次 GetReachablePositions，记录现场集合是否等于冻结集合（只记录，不卡闸门）。
- 冻结集合的摘要（digest）= `"\n".join(f"{x:.2f},{z:.2f}")`（按排序）的 sha256；每个评测任务的 digest 必须等于 `reach/v1/index.json` 里的值（G0 第 11 条）。
- 数据作业之前的核对（契约 F3，feas_reach.py，按探针的出生位姿集合做运动学重放）：1,600 条公开动作序列 0 次移动失败，1,600/1,600 精确到达目标；20,735 个回合里有 142 个（9 个场景的 43 题）公开的 "Valid actions" 行和我们算的不同；500 道评测题的起点和目标格子都在集合里。所以预计 K = 1,600（推断；以真实重放为准）。

### 3.5 成功、步数上限

- 成功 = 发出 Stop **且**停下时位姿精确等于目标（第 1 节第 2 条）。
- 上限（`protocol.max_steps`）：iTHOR 30、ProcTHOR 40 次模型回复（Stop 和无法解析的回复都算一次，eval_va）。到上限没停就算失败。

---

## 4. 起点（`starts`：Q3B、Q3S、E2）

| 名字 | 评测 view 代码 | 路径（只读） | 说明 |
|---|---|---|---|
| **Q3B** | `qwen3vl-8B` | `/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | 原始 Qwen3-VL-8B；权重只有 root 能读 |
| **Q3S** | `c0q3_sft_e8` | `/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_sft_20261026/lf_q3formal/model/checkpoint-1232` | C0 动作 SFT，第 8 轮，1,232 步 |
| **E2** | `sft_m3e2_e10` | `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/E2/model/checkpoint-3010` | QA SFT，第 10 轮，3,010 步 |

- 唯一的权威表是 `vpd/starts.json`（上表抄自它，2026-11-10）。
- 注意力实现：训练和合并都强制 `sdpa`，在 `model.config`、语言模型子配置、视觉子配置三处断言（DECISIONS B10；G0 第 9 条）。
- chat template：三个起点用同一份（sha256 前缀 3636d0f0bd6b，契约 F5）；它没有 thinking 开关，所以**不发送** `chat_template_kwargs`（eval_va 的 `enable_thinking False` 是给 Qwen3.5 的，契约 F8）。
- 训练和评测的 token 化一致（契约 F5，feas_tokens.py）：最长的公开轨迹（26 轮、52 张图）在 n2_train_vpd 的编码下 13,187 token，其中 11,440 个 image pad（每张 640×360 的图 220 个，网格 [1, 22, 40]）；PY_Q3（transformers 4.57.6）和 PY_TRAIN（4.57.0）的 input_ids / 文本 / pixel_values sha256 完全相同；OpenAI 式的消息渲染等于训练渲染。
- 所有模型作业都以 root 跑（附 A：O7；Q3B 必须 root）。

---

## 5. 示范

- **来源**：公开 sft_train.json 的 1,600 条动作序列（第 2.2 节），在我们的环境里用冻结的 lavapipe 档案、L = 2、E = 5、冻结的按任务可达集合重放（`run_tvr replay`，数据作业第 5 步）。每一回合的 CURRENT 图、TARGET 图、"Valid actions" 行都是我们环境的输出（DECISIONS G3(d)、G10(a)）。
- **保留规则**（G10(a)，契约 5.1.7）：一条轨迹保留，当且仅当在我们的环境里：
  1. 每个 THOR 步之后的位姿都等于公开序列在空旷空间里蕴含的位姿（这同时覆盖"每个 Move 都成功"和"每个 Rotate / Look 都给出蕴含的位姿"）；
  2. 最后一个动作是 Stop，并且只有这一个 Stop；
  3. Stop 之前的位姿精确等于目标。
- **剔除原因**（取第一个不满足的，只有语义原因）：`move_failed@k`、`pose_mismatch@k`、`no_stop`、`stop_not_last`、`final_not_target`。每条剔除连同原因和细节写进 `OUT/tvr/demos/sft1600_lvp/dropped.json`。
- **基础设施故障不是剔除原因**：重放目录没有有效的 `results.json`（或少任何一条轨迹）时，`demos build` 拒绝运行、什么都不写。
- **K 的下限**：保留条数 K ≥ `gates.min_kept` = 1,520（1,600 的 95%）。数据作业的 expect 步骤和 G0 第 16 条都查。预计 K = 1,600（第 3.4 节，推断）。
- **K 和剔除名单**（已填，数据作业 tvr-data-202611102015，任何训练之前）：K = 1,595（`data.kept`），剔除 5 条（`data.dropped`），都在 procthor_01401、原因都是 move_failed，名单在 `OUT/tvr/demos/sft1600_lvp/dropped.json`：`procthor_procthor_01401_easy_t02_s00`（move_failed@6）、`procthor_procthor_01401_easy_t02_s01`（@7）、`procthor_procthor_01401_easy_t09_s01`（@7）、`procthor_procthor_01401_hard_t19_s00`（@8）、`procthor_procthor_01401_hard_t19_s01`（@10）。按类别保留 SR-easy 400、SR-hard 400、LR-easy 397、LR-hard 398。独立复算（只读原始文件）给出同样的 K、名单和档位集合。
- **重算的 "Valid actions" 行**：公开数据里有 54 个回合（9 条轨迹，场景 FloorPlan325 和 procthor_01401），专家走的 Move 不在同回合自己的合法动作行里，其中 12 个回合行里一个 Move 都没有（critic，va_check.py）。按出生位姿集合重算，142 个回合的合法动作行和公开的不同（第 3.4 节）。我们的示范一律用我们重算的行，所以训练文本和评测时的文本逐字节相同。
- **kin 检查**（G0 第 6 条，不是剔除原因）：纯运动学环境 kin_env（`vpd/tvr/kin_env.py`，规则逐条照 thor_env.py:131-142、220-333、402-435）用同一个冻结集合、同样的动作，必须每一步给出相同的位姿、action_success、合法动作行和 user 文本；`kin_check.json` 列出所有不一致，`kin_equal_all_kept` 必须为 true。
- **格式**（契约 6.5，n2_train_vpd 的输入）：`{"id", "category", "dataset", "scene", "n_turns", "messages": [system, user, assistant, …], "images": [绝对路径]}`；每个动作一个 user 轮，user 文本 = 重放的 obs_str（含两个 `<image>`），每轮两张图 [step_k, target]；最后一条 assistant 是 `Action: Stop`；没有 loss_weight。
- `demos_all.jsonl` 按 sft.json 的顺序；sha256 → `data.demos_all_sha256`（已填）。

---

## 6. 格式热身 0W（`zero_w`）

- **池**（G10(b)）：从保留的示范里每类取 5 条，共 20 条：每类里 sha256("tvr-0w-v1|" + task_id) 最小的 5 条。档位池 = 保留集合减去这 20 条。
- **怎么训**：和 WalkerBench 试点同一配方（DECISIONS B7、C2）：20 段对话、12 轮、30 个优化步（ceil(20 × 12 / 8)），种子 = 本次运行的种子（试点 1001；种子重复 1002）。
- 每个试点训练作业里都**重新训** 0W，不复用校准或冒烟的（附 A：O8）。0W 的代码 sha 记进 meta，G0 第 5 条断言三个起点相同，并且以 `0865dd75f1ee` 开头。
- 每个档位都从**本起点**的 0W 初始化（种子重复从它自己的 0W_s1002 初始化，附 A：O13）。
- 0W 的评测（pilot__<起点>__0W__c）是 G1 门槛的一部分：最高档的下界必须高于本起点 0W 的点估计。
- 文件：`OUT/tvr/tiers/v1/warmup0w20.jsonl`（20 条，逐字节复制自 demos_all.jsonl 的行）。0W 训练的 20 条示范同时是合并一致性检查的数据（第 13 节 G0 第 7 条）。

---

## 7. 档位（`tiers`）

- **顺序**（G10(c)）：档位池里每类按 sha256("tvr-tier-v1|" + task_id) 升序；N 档取每类前 N/4 条；ALL 档取整个档位池。子集嵌套：100 ⊂ 400 ⊂ ALL，和 0W 池不相交（`tiers/v1/manifest.json` 记 `nested`、`disjoint_from_0w`）。各起点用**完全相同**的子集，所以可以逐题配对。
- 文件：`tiers/v1/{warmup0w20, demos_100, demos_400, demos_ALL, parity3}.jsonl`、`order.json`、`manifest.json`（契约 6.6）。parity3 = 最长的保留示范（平手取最小 id）、档位顺序里第一条 SR-hard、第一条 LR-easy，只用于 dry run。
- **试点档位** = {0W, 100, N_top}（`pilot_cfg.pilot_tiers()`）。
- **校准**（G10(e)；契约 7.2；EXPLORATORY，只用 Q3B、只在 dev200、种子 1001）：
  - 训 0W、100、ALL，在 dev200 上用主协议（c 臂）各评一次；
  - `N_top` = **ALL**，如果 Q3B 的 ALL 档 dev200 成功率 ≤ 70%（`tiers.all_dev_sr_max` = 0.70；**等于 0.70 时取 ALL**）；否则 N_top = **400**，并且另跑校准作业 tvr-cal400（训 400 档、在 dev200 上评）；
  - 结果写进 `CAL/cal_decision.json`，编排把 `tiers.n_top` 填进第 0.1 节：**已填 ALL**（下面"校准结果"）。
  - 校准的其他数字**都不用来改配方**。只有两样用于工程：(1) `render.eval_envs` 可以在冻结前按 Q3B 的测速确认或改（例如带 vLLM 的总速率低于 4 步/秒时）；(2) 训练和评测作业的 est_high 按校准的 Q3B 实测推出（第 16 节）。
  - 校准同时是机制排查：raw Q3B 的 c、blind 臂在 smoke4 上只跑机制（不出成功率），第一次以 root 起 max_model_len 32768 / max_images 96 的 vLLM 服务；起不来就按登记的备用值 128000 重试一次，并在冻结前记第 17 节（附 A：O9）。只有**这一次**尝试里服务确实没起来（run_tvr 退出码 1、这次新写了 `<前缀>__server_error.json`、没有一个臂出结果）才换备用值；拒绝（退出码 2）、超时、TERM 或 THOR 失败都不换。备用值的那次尝试本身没成功（没写下 `CAL/shake/fallback_used.json`）时，校准作业重跑会先把那次尝试留下的 c、blind 臂目录和第一次的 server_error 副本改名为 `<名>.prev-<UTC>`，再从冻结的 max_model_len 起、按同一条规则重新判（`vpd/jobs/tvr_cal.sh` 的 shake）。
- **无望停止**（附 A：O21）：`cal_decision.futility` = Q3B 最高档（ALL；N_top = 400 时在 cal400 之后按 400 档重判）dev200 c 臂的 Wilson 95% 下界 ≤ dev200 的盲下限 F（`floors_dev200.json`）。为真时编排不准备冒烟和训练作业，报告并问用户；`pilot_tvr` 只有在用户决定后由编排写下 `CAL/futility_override.json` 时才放行。这一步只看 Q3B、和起点无关，符合分离规则。
- **校准结果**（tvr-cal pt-js5w1gsc，2026-11-11 09:14Z SUCCEEDED；0W、100 两个适配器和 raw 的 shake 是第一个校准作业 tvr-cal-202611110140（pt-zroa15mv，同一个代码快照 f393095e，02:1x–02:3x；02:52Z 在 train_ALL 时因账号磁盘配额 FAILED）的输出，这次跳过了重训（steps.jsonl：train_0W:skip、train_100:skip）和 shake（steps.jsonl 里 1 秒；worker.log：c、blind 两个臂 complete … skipped，复用已有结果），自己训 ALL、跑三次 dev200 评测；EXPLORATORY，只有 Q3B、dev200、c 臂、种子 1001；`CAL/cal_decision.json` 记的 results_sha256 是 ebdb3db109b6…；编排用独立脚本从 per_task.jsonl 重数过，一致，`OUT/scratch/orch/r3/calcheck/indep_cal.out`）：
  - 0W 1/200、100 12/200、ALL 85/200（0.425，Wilson 95% 0.359–0.494）。ALL ≤ 0.70，所以 N_top = **ALL**，不跑 tvr-cal400；
  - 无望停止为假：ALL 档 Wilson 下界 0.359 > dev200 盲下限 F = 0.005（`floors_dev200.json`，sha256 dc123c1c9059…）；
  - dev200 图像臂评测的步速 6.48 / 5.46 / 5.20 步/秒（0W / 100 / ALL，`results.perf.steps_per_s`），都高于 4 步/秒，所以 `render.eval_envs` 确认 5，不改（raw Q3B 的 shake c 臂只有 smoke4 的 4 题、42.5 秒，2.84 步/秒，题太少（4 题少于 5 个并行环境，测不出稳态速率，推断），不算测速；第 16 节的 R 也只取 dev200 评测）；
  - 服务按冻结的 max_model_len 32768 起来了（第一个作业的 shake 和这次的三次 dev200 评测，`settings.max_model_len_used` 都是 32768；最长提示 20,280 token、每次请求最多 80 张图），没有 `CAL/shake/fallback_used.json`，O9 的备用值没用；
  - 按第 16 节推出的 est_high（`pilot_tvr.est_high`，向上取到 0.001 h；输入：适配器 meta 的 train_seconds + load_seconds 0W 336.7 + 55.7、100 405.5 + 57.7、ALL 6440.3 + 65.4；R = 5.199，blind 臂 16.161（shake 的 blind 臂，smoke4 的 4 题；blind 臂没有 dev200 评测，第 16 节的规则就取它），vllm_ready_s 最大 505.2，merge 186，parity 156；mean_steps_per_task 34.27 / 34.15 / 27.915（0W / 100 / ALL），steps_high 都取上限 17,500）：train Q3S、E2 2.759 h，Q3B 3.068 h；eval top 3.027 h（每个起点），low 2.491 h（Q3B 3.662 h）；都 ≤ 5.5。按这些值第 16 节的试点预留是 35.3 卡时（第 16 节写的预期 33.0；附 A O20 写的"最坏约 3.1 h"，推出的最大是 Q3B low 的 3.662 h）；
  - 这些是 Q3B 一个起点的绝对数，只用于 N_top、eval_envs、无望停止和 est_high（第 16 节），不用来改配方。
- 0 档：试点**不跑** raw 评测（附 A：O11）；曲线的 0 点是 0W。

---

## 8. 训练配方（`recipe`）

训练脚本是冻结的 `vpd/train/n2_train_vpd.py`（代码指纹前缀 `0865dd75f1ee`，覆盖 n2_train_vpd.py、downstream/train/lora.py、vpd/starts.py、vpd/starts.json、vpd/paths.py、vpd/walkerbench/history.py；本轮所有训练都用这一个指纹，ROUND3）。每个作业的 guard 断言 `n2_train_vpd.code_fingerprint()` 以这个前缀开头。

| 项 | 值 |
|---|---|
| 适配器 | LoRA r16 / α32 / dropout 0.05，语言模型 252 个模块（同 WalkerBench 第 6 节）；视觉塔和投影层冻结 |
| 优化 | AdamW，学习率 1e-4；先线性预热 max(1, ceil(0.03·T)) 步（`warmup` 0.03），再余弦降到 0 |
| 批 | 每个 micro-step 一段对话，梯度累积 8 |
| 轮数 | `epochs = max(3, ceil(240/N))`（`--epochs auto`）；0W 为 12 |
| 输入 | 完整多轮历史（`--history-k 0`，默认），每个回复都训练；图像 640×360，`max_pixels` 1,024,000（DECISIONS B16）；`--max-seq-tokens 40000`（TVR 最长约 13.3k token，不会跳过样本，契约 F6） |
| 注意力 | sdpa；梯度检查点（non-reentrant） |
| 线程 | `--threads 2 --num-workers 3`（2 线程 + 3 个 loader，约 5 核） |
| 种子 | 1001（种子重复 1002），同时管数据顺序和训练 |
| 选 checkpoint | 不挑，用最后一步 |
| 环境 | PY_TRAIN（starVLA，transformers 4.57.0）；处理器一致性见第 4 节 |

试点的步数（T = ceil(N·epochs/8)；ALL 档按 K = 1,600 推断，以数据作业的实际 K 为准）：

| 档 | N | 轮数 | 优化步 | 预热步 |
|---|---|---|---|---|
| 0W | 20 | 12 | 30 | 1 |
| 100 | 100 | 3 | 38 | 2 |
| 400（只在 N_top = 400 时） | 400 | 3 | 150 | 5 |
| ALL | K − 20（推断 1,580） | 3 | 593（推断） | 18（推断） |

- 训练命令（契约 7.2 第 3–5 步，7.4）：`n2_train_vpd.py --start S --data tiers/v1/<档文件> --out P/S/adapters/<档> --epochs auto --seed 1001 --phase <0W|tier100|tier400|tierALL> --threads 2 --num-workers 3`，100 / 400 / ALL 加 `--init-adapter P/S/adapters/0W`；种子重复用 `--seed 1002 --phase 0W_s1002 / tier100_s1002`，100 档从 0W_s1002 初始化。
- dry run：`--dry-run --limit 3 --expect-grid 22x40`（parity3），断言每张图网格 [1, 22, 40]。
- 显存：WalkerBench 最长 30,977 token 的样本峰值 35.2–36.0 GiB；TVR 最长 13.2k token，离 80 GB 很远（推断，契约 F6）。
- **合并**：`vpd/train/merge.py`（冻结），合并权重只写作业本地盘（/tmp 剩 ≥ 40 GB，否则 /dev/shm），评测完就删；合并后用 `vpd/tvr/merge_parity.py` 做一致性检查（第 13 节 G0 第 7 条）。

---

## 9. 评测（`protocol`）

### 9.1 回合循环（契约 4.5，等同 eval_va.eval_one_task）

```
messages = [system(SYSTEM_PROMPT 字符串), user(obs_0)]
for t in range(cap):
    reply = model(messages)                     # 各臂的解码见 9.2
    messages.append(assistant(reply))           # 原样的回复文本（u 臂也一样）
    a = parse(reply) or "InvalidAction"         # c / blind：re.fullmatch(REGEX, reply).group(1)；u：eval_va.parse_action
    用 a 走一步（Stop 不调用 Controller）-> obs_{t+1}, done
    if a == "Stop" or done: break
    messages.append(user(obs_{t+1}))            # 图像臂：2 张新图（CURRENT，TARGET 每轮重发）
success = stopped and is_exact(final_pose, target)
```

- 图像臂的 user 内容 = eval_va._obs_to_oai_content：obs_str 按 `<image>` 切开，非空文字段作 `{"type": "text"}`，图按 [CURRENT, TARGET] 的顺序作 `{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}`。
- blind 臂的 user 内容 = 字符串 `obs_str.replace("<image>", "")`（不给图，文字相同；n2_train_vpd --blind 会产生同样的文字）。blind 臂在 kin_env 里走（同一冻结集合），不起 THOR。
- PNG 字节 = PIL `img.save(buf, format="PNG")` 默认参数，和 eval_va._img_to_data_url 以及重放完全一样；训练读同样的文件（无损）。
- 历史：完整多轮历史（`protocol.history` = full），和官方 VA 协议、公开 SFT 数据一致（DECISIONS G3(c)）。
- 用户文本 = TVRBench 的 `build_concat_user_text(valid)` 逐字照抄（`vpd/tvr/tvr_text.py`，用 ast 和 REPO 源码核对）；system = TVRBench 的 SYSTEM_PROMPT（942 字符）。
- 无法解析的回复（只会出现在 u 臂）→ "InvalidAction"：花掉一步，位姿不动，`action_success` 为 None（eval_va）。

### 9.2 解码（契约 4.6；附 A：O18）

请求体（发给 VLLMServer 的 `/v1/chat/completions`）：
`{"model": <起点代码>, "messages": [...], "temperature": 0.0, "top_p": 1.0, "max_tokens": M, "top_k": -1}`，另外：

| 臂 | 用途 | 约束 | max_tokens | 解析 |
|---|---|---|---|---|
| **c**（主） | G1、G2、G3、A3、A4 | `"structured_outputs": {"regex": REGEX}` | 16 | `re.fullmatch(REGEX)` |
| **u**（次要，只报告） | 解析有效率 | 不约束 | 64（eval_va 默认） | eval_va.parse_action |
| **blind**（A3） | 不给图 | 同 c | 16 | 同 c |

- REGEX（`protocol.regex`，DECISIONS G3(b)，只管格式，所有起点相同）：
  `Action: (MoveAhead|MoveBack|MoveLeft|MoveRight|RotateRight|RotateLeft|LookUp|LookDown|Stop)`
  它允许当前不合法的动作（撞墙的 Move 照官方规则花一步、位姿不动），和官方协议一致。
- 不发 chat_template_kwargs、不发 seed、不发 uuid。

### 9.3 服务（vLLM 0.11.0，PY_Q3）

- `VLLMServer(code, max_images=96, max_model_len=32768, gpu_memory_utilization=0.85)`，其余全用默认：xgrammar 结构化输出后端、`--seed 0`、`--generation-config vllm`、`--enforce-eager`、`--mm-processor-kwargs` 带 `max_pixels` 和 `size.longest_edge` = 1,024,000（DECISIONS B16），前缀缓存保持 vLLM 默认（开）（`protocol.server`）。
- 版本（`protocol.versions`）：vllm 0.11.0、xgrammar 0.1.25、transformers 4.57.6（服务）/ 4.57.0（训练）、ai2thor 5.0.0。每次评测的 results.json 记录实际版本，G0 第 10 条核对。
- 最坏的请求：40 个 user 轮、80 张图、20,280 个提示 token（iTHOR 上限 30 时 15,270），在 32,768 以内（契约 F5）。max_images 96 留了 16 张余量（critic：默认的 80 正好卡满）。
- 32768 / 96 张图的服务能不能起来，登记时未核实（低风险；校准里起来了，O9 的备用值没用，第 7 节"校准结果"）；校准第一次起；起不来就用登记的备用值 `protocol.max_model_len_fallback` = 128000，冻结前记第 17 节，之后所有试点评测都用它（附 A：O9）。
- 一次 vLLM 会话服务一个起点一个档位的所有臂；图像臂共用一个 E × L 的 THOR 池；blind 臂在图像臂之后跑。

### 9.4 每个起点的试点评测（契约 7.5；附 A：O6、O20）

评测名 `pilot__<起点>__<档>__<臂>`，目录 `P/<起点>/evals/`。每个起点一个训练作业（tvr-train-<s>）和两个评测作业（tvr-eval-<s>-top、tvr-eval-<s>-low），三个起点完全相同：

| # | 作业 | 模型 | 臂 | 用途 |
|---|---|---|---|---|
| 1 | -low | 0W | c | G1 的 0W 门槛；G0、G2 |
| 2 | -low | 100 档 | c | 曲线描述；功效模拟的 p_i(100)；G0、G2 |
| 3 | -top | N_top 档 | c | **G1、G3、A3、A4**；G0、G2 |
| 4 | -top | N_top 档 | u | 次要：解析有效率（只报告）；G0 |
| 5 | -top | N_top 档 | blind | A3（只报告）；G2 |
| 6（只 Q3B） | -low | 100 档 seed 1002（100s1002） | c | 种子重复：只用来估 σ 和任务持续性 λ（第 12 节第 5、7 条），结果只进 A4；G0、G2 |

- 评测集：eval.json 全部 500 题（DECISIONS G3(f)、G10(h)），不加 `--limit`。
- 每个评测作业开头先跑渲染金丝雀（sft2 重放 + `render_parity`，对照数据作业的重放），不过就不跑这个作业的评测（附 A：O22）。
- 合并评测之前先做合并一致性检查（PY_TRAIN，vLLM 起之前）。
- 评测完对每个评测目录跑 TVRBench 自己的 `eval_va.aggregate()`（`score_tvr official`），必须和我们的成功率相同（G0 第 12 条）。

### 9.5 可复现性说明

逐题结果只能复现到两种噪声以内：(1) vLLM 服务的非确定性（E 个并发的贪心序列不是 batch-invariant 的）；(2) 渲染的非确定性（第 3.2 节）。断点续跑的评测按原样接受。种子重复估出的 sigma 也包含这部分服务噪声（契约 8.4，E-N2）；估出的任务持续性 λ 也一样（这部分噪声让两个种子的逐题结果更不一致，λ 偏低一点；确认轮的种子之间同样有这部分噪声，推断）。

---

## 10. 指标（契约 4.7）

每次评测都报（`results.json`，`score_tvr`）：
- **成功率** SR = k/n，Wilson 95% 区间（z = 1.96，`pilot_gate.wilson_lb/ub`）；每类 k/125。
- **SPL**（次要，G10(j)）：SPL_i = success_i × opt / max(opt, n_nonstop)，opt = 冻结集合的位姿图上起点到目标的 BFS 最少非 Stop 动作数（`vpd/tvr/bfs.py`；不可达则 None，记 0），n_nonstop = Stop 之前的回复数（没停就是全部回复；撞墙、俯仰到头、InvalidAction 都算）。SPL = 500 题平均。公开轨迹的长度和 bfs_distance 差 −9 到 +8，所以不用 bfs_distance（prereg_confirm）。
- **官方口径的 SPL**（只作描述）：标注为"eval_va _spl with our BFS opt (Stop counted in steps)"，即 eval_va._spl(success, 含 Stop 的步数, opt)（eval_va.py:363-368）；最优专家在这个口径下是 opt/(opt+1)。登记的次要指标是上面那个 SPL。
- **差一步成功** within1 = 停下了，并且终点等于目标，或者一个非 Stop 的运动学转移能从终点到目标。
- reached = 轨迹途中某一步到过目标位姿（描述）。
- 停止率 stopped/n；F-stop：fstop_given_stop = (停下数 − 成功数)/停下数（停下数为 0 时为 None），fstop_over_all = (停下数 − 成功数)/n（eval_va 的定义）。
- 无效：u 臂无法解析的回复数；被挡住的动作数（执行了 Move / Look 但 success False）；c / blind 臂的正则不匹配数（G2 要求 0）。
- 终点误差：eval_va 的 final_pos/rot/hor_error 的平均。
- 每题结果写进 `per_task.jsonl`（功效模拟和配对检验用）。
- 每张图的 token：用服务的处理器对保留的 PNG 算 `image_grid_thw`，token 数 = 网格乘积 / 4 + 2 = **222**（220 个 `<|image_pad|>` 加 2 个边界 token；DECISIONS B4 的口径；附 A：O3）。另做 HF 提示 token 数和服务 `usage.prompt_tokens` 的逐项比对。

---

## 11. 盲下限 F（`floors`；G10(f)；附 A：O17）

在 kin_env 里算，用冻结的按任务集合，不用图、不用模型（`vpd/tvr/floors.py`）：
1. **STOP0**：第一次回复就 Stop。eval500 上预计 0（没有任务的起点就是目标）。
2. **FIX**：对 8 个非 Stop 动作 a 和 k = 0..39，策略"a 重复 k 次然后 Stop"；上限为 cap 的题上 k + 1 > cap 的策略永远停不下来（失败）。在 sft1600 上给每个 (a, k) 打分，取 sft 成功最多的；平手取较小的 k，再按 ACTIONS 的顺序；把选中的策略在 eval500（和 dev200）上打分。
3. **RAND**：对种子 s ∈ {0, 1, 2} 和每道题，rng = random.Random(f"tvr-floor-rand-v1|{s}|{task_id}")；m = 该类公开专家非 Stop 动作数的 statistics.median_low（1,600 条公开序列）；走 m 步，每步从当前合法的非 Stop 动作里均匀抽一个；然后 Stop。每个种子一个成功率，取 3 个的平均。
- **F** = max(STOP0, FIX 在 eval500 上, RAND 平均)；`F_source` 记是哪一个；也报等价成功数 ceil(F × 500)。
- 无碰撞的几何估计（EXPLORATORY，不是闸门下限）：长度 ≤ 3 的固定序列里训练集选出的 (MoveRight, LookDown) 在 eval 上 1/500（prereg_confirm 的 tvr_floor_nocoll.py）。所以 F 预计接近 0（推断），这也是 G1 另加"高于本起点 0W"的原因（DECISIONS G10(f)，critic C5）。
- 数值：`data.F_eval500` / `data.F_source`：**已填：0.0 / stop0**（数据作业 tvr-data-202611102015；`recheck_tvr` 独立复算相同）。文件 `OUT/tvr/floors/floors_eval500.json` 的 `warnings` 必须为空列表（G0 第 14 条）。
- dev200 的下限（`floors_dev200.json`）只给校准和无望停止用。
- **独立复算**（简报第 6 节）：`vpd/tvr/recheck_tvr.py floors` 只按本节和第 3.4 节的文字重写转移规则、合法动作行、冻结集合（按 digest 从 reach/v1 场景文件读）、专家中位数和三个下限策略，不导入 kin_env、floors、bfs、tasks、score_tvr；`gate_tvr` 要求它的 F、F 来源和三个分量逐项等于 floors_eval500.json（G0 第 17 条）。

---

## 12. 统计

1. **Wilson 区间**：p̂ = k/n，z = 1.96，LB = ( p̂ + z²/(2n) − z·√( p̂(1−p̂)/n + z²/(4n²) ) ) / ( 1 + z²/n )，UB 把减号换成加号（同 PREREG_WB_PILOT 第 9 节，`pilot_gate.wilson_lb/ub`）。
   - 例（critic 手算）：下限或 0W 为 1/500 时，要 ≥ 3/500 才能过（3/500 的下界 0.00204，2/500 的是 0.0011）；为 0/500 时 1/500 就够。
2. **A3 的配对精确检验**：同一适配器、同 500 题，b = 只有给图时成功的题数，c = 只有不给图时成功的题数；p = min(1, 2·P(X ≤ min(b, c)))，X ~ Bin(b + c, 1/2)；b + c = 0 时 p = 1。"给图更好" = b > c 且 p < 0.05。例：c = 0 要 b ≥ 6，c = 1 要 b ≥ 8，c = 2 要 b ≥ 10（prereg_confirm 的 stats_plan.py）。只报告。
3. **按场景整组的 bootstrap**（`bootstrap`）：把这次运行的场景有放回地重抽 10,000 次（random.Random(0)），SR = 抽中场景的成功数 / 题数，95% 百分位区间；**只作描述**。
4. **终点层面的种子感知场景整组 t 检验**（`power.test` = seed_aware_endpoint_t；`vpd/common/power_sim.py` 的 `seed_aware_endpoint_test` 和 `seed_var_endpoint`；A4 和确认轮共用这一个检验；DECISIONS G13）：
   - 每个起点 S、种子 s、每道题 i：e_{S,s,i} = Σ_T w_T · y_{S,s,T,i}（档 T ∈ {100, TOP}，权重各 0.5；两档按同一个种子配对）；起点的终点 e_{S,i} = mean_s e_{S,s,i}；d_i = 处理起点的终点 − 对照起点（或两个对照的平均）的终点；D = mean_i d_i。
   - V = mean(d) 的 CR1 场景整组方差（G/(G−1) · Σ_g(Σ_{i∈g}(d_i − D))² / n²，eval500 有 80 个场景）**加上**种子分量 k · σ̂_E² / n_s。k = 对比系数的平方和：Q3S − E2 是 2，Q3RD − mean(QRL-s1, QRL-s2) 是 1.5。
   - σ̂_E² = max(0, 各起点 [var_s(E_{S,s}) − mean_i var_s(e_{S,s,i}) / n] 的平均)（ddof 1；E_{S,s} = mean_i e_{S,s,i}），df = 起点数 × (n_s − 1) = 12（6 个起点 × 3 个配对种子；备用设计 6）；σ̂_E² 从（模拟的或真实的）种子里估，从不用真实的 sigma。
   - t = D / √V，参照 t 分布、自由度 df，双侧；两个对比做 Holm（α = 0.05，依次和 0.025、0.05 比）；95% 区间 = D ± t_{0.975, df} √V。自由度不用 Satterthwaite（核对里族错误率 0.05–0.065，偏松）。
   - 每个 (起点, 种子, 档) 的 500 题 c 臂评测都必须在；缺了就补跑，不插补。
   - 为什么要加种子分量（crit-protocol 的必改项，arch-rev 复核）：一个起点在某档某种子上整体偏高或偏低，对所有题都一样，不会出现在按题的残差里；只用场景整组方差的配对检验会严重偏松。arch-rev 的复现（seedaware_sim.py，1,000 次，5 个起点的合成 p）：朴素检验的一类错误在 σ = 0.03 时 0.44–0.46、σ = 0.05 时 0.60；种子感知检验用正态参照 0.059–0.070，用 t 参照（那里 df 20）0.045–0.058；σ̂ 的平均 0.0292（真值 0.03）和 0.0492（真值 0.05）。
   - **为什么用终点层面**（wave-2 核查 stats:a4；DECISIONS G13）：确认轮里同一个 (起点, 种子) 的 100 档和 TOP 档共用 0W_s 初始化、数据顺序和训练种子（第 15 节第 3 条），两档的偏移很可能正相关。原来登记的按 (起点, 档) 格估种子方差的版本（seed_aware_cluster_t，df 24）假设两档独立：相关系数为 ρ 时一个起点终点的种子方差是 σ²/n_s · (w₁² + w₂² + 2ρ·w₁w₂)，ρ = 1 时是它假设值的 2 倍。核对（推断：合成 p，500 题 80 个场景，每格 2,000 次）：旧检验的族错误率在 ρ = 0 / 0.5 / 1 时是 0.044 / 0.111 / 0.174（σ = 0.03），σ = 0.05、ρ = 1 时 0.20；终点层面的检验在三个 ρ 和两个 σ 下都是 0.045–0.051，因为每个起点每个种子的终点里已经含了两档之间的协方差（`seed_var_endpoint` 的恒等式，`test_endpoint_identity`）。代价是 ρ = 0 时约 3 个百分点的功效（df 12 而不是 24）。旧检验作废，只留在 POWER.json 的 `sensitivity_superseded_test` 里（两个设计、ρ = 0 和 1，只报告，不卡闸门）。
   - **未核实**：真实的两档相关有多大，试点里测不出来（试点每档只有 Q3B 的一个种子重复），所以 A4 的一类错误在 ρ = 0 和 ρ = 1 两端都判，功效在中间的 ρ = 0.5 判（第 6 条）。
   - 本包的核对（推断：合成 p，不是试点数据；`power_sim` 在登录节点 1 核跑，每格 2,000 次，λ = 0）：主设计、σ = 0.03 时终点检验的族错误率 0.045（ρ = 0）/ 0.044（ρ = 1）；8 个百分点时功效 Q3S − E2 / Q3RD − mean(QRL) = 0.91 / 0.97（ρ = 0）、0.81 / 0.90（ρ = 0.5）、0.69 / 0.81（ρ = 1）；σ = 0.05、ρ = 0.5 时约 0.39 / 0.51（stats:a4）。玩具设计（5 个起点）上朴素检验的族错误率在 σ = 0.03 时 0.36、σ = 0.05 时 0.62。
5. **种子 SD**（`power.seed_sd`；附 A：O4）：Q3B 100 档 c 臂 seed 1001 对 seed 1002，同 500 题：e_i = y1_i − y2_i，d = mean e，S = Σ(e_i − d)² / (n(n−1))，σ̂ = √max(0, (d² − S)/2)（实现 SR 尺度），**σ_used = max(0.03, σ̂)**。这是同一个起点两个种子之间的差，不是起点之间的差。种子重复是 O4 规定必跑的：`P/Q3B/evals/pilot__Q3B__100s1002__c/per_task.jsonl` 不在、或者它的题目和三个起点汇总的题目不一致时，`power_sim` 拒绝（退出码 2）；只有显式给 `--allow-no-replicate`（`run_tvr(allow_no_replicate=True)`）才用 σ_used = 0.03，这时第 7 条的 λ_used 取登记范围的上端 1（保守），这一选择写进 POWER.json（`registered.allow_no_replicate`、`sigma.replicate_missing`、`persistence.replicate_missing`）。**`gate_tvr` 从不读这样的 POWER.json**：这三个字段任一表示没有种子重复时 A4 记 pending，决定是 `incomplete`（第 13 节 A4）；缺种子重复本来就让 G0 第 1 条判"缺评测"。没有种子重复时还要不要判 A4、怎么判，由用户定，记第 17 节。同一个种子重复也是第 7 条任务持续性 λ 的数据来源（种子之间的不一致率）。
6. **功效模拟**（A4；`power_sim tvr`，契约 8.3）：
   - p_i(T) = 三个试点起点在档 T（100、N_top）上主臂结果的平均（按起点取平均，不算起点差；`power.p_source`）；场景是整组单位。
   - 每次模拟里，对 design.starts 的 6 个起点和每个 (种子 s, 档 T)：抽实现 SR 尺度的种子偏移 r_{S,s,T} = σ(√ρ · z_{S,s} + √(1 − ρ) · z_{S,s,T})（z 独立标准正态；z_{S,s} 是同一个种子两档共用的部分，ρ = 两档偏移的相关，`power.tier_corr`；`power.seed_model`），用 calibrate_shift（[−1, 1] 上 2,001 个格点的 r(c) = mean(clip(p + c, 0, 1)) − mean(p) 的线性插值反函数）换成截断前的偏移 c；被检验对比的处理起点用 calibrate_shift(r + δ)（每档期望的 SR 增益都是 δ）；y_i = 1[W_i < clip(p_i(T) + c, 0, 1)]，W_i 按第 7 条的任务持续性抽（λ = 0 时就是独立的 Bernoulli，和原来逐位相同）。
   - 网格（每个设计、每个 σ ∈ {σ_used, 0.03, 0.05}，都在 λ_used 下；POWER.json 每行的 `role`）：δ = 0 在 ρ = 0 和 ρ = 1（`type1`）；δ ∈ {0.05, 0.08, 0.10}（`power.deltas`）在 ρ = 0.5（`power`）；δ = 0.08 再在 ρ = 0 和 ρ = 1 各算一次（`reported`，只报告）。另有任务持续性的敏感性行（`sensitivity_persistence`：σ_used，λ ∈ {0, 0.8}，δ = 0 在 ρ = 0、1，δ = 0.08 在 ρ = 0.5；只报告）。0.05 的 σ 行也只报告。每格 2,000 次，种子 0，后端 numpy（`power.backend`）；同一格的各块共用随机数（common random numbers）。后端也要登记：`power_sim` 的 python 和 numpy 两个后端抽的随机数流不同，同一个种子只有和后端一起才确定结果（玩具设计上同种子两个后端的功效 0.325 / 0.433 对 0.392 / 0.342，review tvrjobs `rv_check.json`），而 A4 本身常落在离阈值一两个蒙特卡洛标准误的地方（第 7 条：合成 p 上 λ = 0 时 0.813 只比 0.80 高约 1.5 个），换后端就可能翻转判定。`run_tvr` 默认用 `power.backend`，`--backend` 和 `--n-sims` 一样只供测试，实际用的值记进 `registered.backend`；numpy 导入不了时 `run_tvr` 拒绝（ValueError），不悄悄换成 python。每个功效另报蒙特卡洛标准误 √(p(1 − p)/2000)（`mc_se_power`；功效 0.8 附近约 0.009，所以 0.80 上下一个标准误以内的判定是靠运气的，阈值照登记不变）。
   - **功效** = 在 Holm 之后被拒绝、**并且处理起点更好**的模拟所占的比例，另一个对比的效应设为 0（保守；`power.power_counts`）。判的是 σ_used、λ_used、ρ = 0.5、δ = 0.08 那一格；ρ = 0 和 ρ = 1 的功效只报告。
   - **一类错误**（门槛 `power.type1_max` = 0.06，2,000 次时蒙特卡洛标准误约 0.005）= 两个对比效应都为 0 时，Holm 至少拒绝一个（任一方向）的模拟所占比例（族错误率，比单个对比的拒绝率更严；定义由 DECISIONS G13 确认）。**判的值 = σ_used、λ_used 下 ρ = 0 和 ρ = 1 两格里较大的那个**。各对比不校正的拒绝率也报。
   - **为什么功效在 ρ = 0.5 判**（stats:a4）：ρ = 1 时 A4 不可能过（σ 的下限 0.03 上功效已是 0.69）；ρ = 0 是乐观的一端，忽略了促成这次修改的共用 0W_s；终点检验的功效对 ρ 近乎线性（λ = 0 时 0.905、0.864、0.813、0.747、0.694，ρ = 0、0.25、0.5、0.75、1，推断），所以 ρ = 0.5 大约是 ρ 在 [0, 1] 上均匀时的平均功效。
   - **登记的备用设计**（第 15 节第 9 条）= 主设计，但两档都只用种子 1–2（`power.fallback.seeds`；两档配对，所以同一个检验能用，df 6），同样的网格和判定规则，表名 `fallback_seeds_1_2`，判定写进 `fallback_pass`。
   - **只报告的设计行**（`power.reported_designs`，wave 2c；POWER.json 的 `reported_designs` 块，`role` = `reported`，`gated` 为 false；判定从不读它，放在 `tables` 外面，所以 `verdicts`、`pass`、`fallback_pass` 和不加这些行时逐位相同）：给 A4 不过时由用户定方向用。都在 σ_used、λ_used、ρ = 0.5（`power.reported_designs.tier_corr`）下，2,000 次、种子 0，同一个检验，报两个对比的 Holm 功效（同"功效"的数法）和蒙特卡洛标准误：主设计每个起点 4 个种子（两档都用种子 1–4，配对；`main_seeds_4`，df 18）和 5 个种子（种子 1–5；`main_seeds_5`，df 24），δ = 0.08；主设计 3 个种子、δ = 0.10（`main_delta_10pp`，df 12；就是 `tables.main` 里 `power` 行 δ = 0.10 那一格）。df = 起点数 × (n_s − 1)。**一类错误行**（wave 2c 第二轮；`power.reported_designs.type1`，POWER.json 的 `reported_designs.type1`，同样只报告、`gated` 为 false、在 `tables` 外面，判定不读）：`main_seeds_4` 和 `main_seeds_5` 在 δ = 0、ρ = 0 和 ρ = 1（`power.reported_designs.type1.tier_corr`）、σ_used、λ_used 下，2,000 次、种子 0，同一个检验，报两个对比的 Holm 族错误率（同"一类错误"的数法：至少拒绝一个，任一方向）和它的蒙特卡洛标准误 √(f(1 − f)/2000)，另报每个设计两格里较大的那个（`by_design.<设计>.type1`，取法同 A4）和它是否 ≤ `power.type1_max`（`within_type1_max`）。`main_delta_10pp` 不需要一类错误行：δ = 0 时它就是主设计，它的族错误率就是 `tables.main` 的 `type1` 行。A4 不过以后用户要是选了一个只报告的设计，这个设计在这些行里也必须满足登记的一类错误界（两格里较大的 ≤ 0.06）。例（推断：wave-2c fixer 的合成 500 题根，`OUT/scratch/orch/r3/wave2c/fix/scripts/fx_time.py` 写的：80 个场景，Beta(0.6, 1.4) 的 p，Q3S、E2 在 logit 尺度上比 Q3B 高 0.6、低 0.2；λ̂ = 0.020、λ_used = 0.050，σ_used = 0.03）：主设计 Q3S − E2 / Q3RD − mean(QRL) 0.792 / 0.900（A4 不过），4 个种子 0.930 / 0.976，5 个种子 0.972 / 0.995，δ = 0.10 时 0.935 / 0.987（`OUT/scratch/orch/r3/wave2c/fix/out/log_time_new.txt`；本轮在同一个根上重算，数字相同）；族错误率 ρ = 0 / 1 时 4 个种子 0.0465 / 0.0425，5 个种子 0.046 / 0.0595（蒙特卡洛标准误 0.0045–0.0053；5 个种子 ρ = 1 那格离 0.06 不到一个蒙特卡洛标准误；`OUT/scratch/orch/r3/wave2c/refix2/logs/time_new.txt`）。0.0595 大概是偏高的一次抽样（推断）：wave-2c 第二轮统计评审在同一个根上用 20,000 次、另一个种子（20261111）重算，主设计和 4、5 个种子在 ρ = 0 / 1 下都在 0.045–0.048（蒙特卡洛标准误约 0.0015；`OUT/scratch/orch/r3/wave2c/review/stats_r2/out/indep_root_new.json` 的 big 块）。登记的 2,000 次下，一类错误接近 0.06 的设计过不过界会受蒙特卡洛噪声左右（推断）。
   - 输出 `P/POWER.json`（schema `vpd.power_sim.tvr/3`）：输入文件的 sha256、p 的汇总（只按档，不按起点）、σ（Q3B 种子重复的 seed_diff_mean、S、σ̂、σ_used）、λ（第 7 条，`persistence`）、全表（`tables.main`、`tables.fallback_seeds_1_2`，每行带 `tier_corr`、`persistence`、`role`、`mc_se_power`）、判定（`verdicts.<设计>`：`type1`、`type1_by_rho`、`power_at_primary`、`power_tier_corr`、`power_by_rho`、`mc_se_power_at_primary`、`pass`），以及 `sensitivity_superseded_test` 和 `reported_designs`（含一类错误行 `reported_designs.type1`；都只报告）。`registered` 抄录所用的 `test`、`test_def`、`seed_model`、`tier_corr`、`fallback`、`persistence`、`reported_designs`、`n_sims_registered`、`n_sims_used`、`backend`（本次实际用的后端）、`allow_no_replicate`、`type1_definition`。`inputs.pilot_cfg_json_sha256` = sha256(pilot_cfg.as_json(所用 cfg))，`inputs.pilot_cfg_py_sha256` = pilot_cfg.py 文件的 sha256（GATE.json 的 `pilot_cfg_sha256` 是后者）。
   - **POWER.json 给什么、不给什么**（wave 2c；`power_sim` 模块说明的 Separation 一段）：POWER.json 不给任何起点的 SR 水平（任何档、任何种子），也不给任何能只从 POWER.json 用闭式算出起点之间 SR 差的值；给的是 Q3B 种子重复的起点内种子差统计（`sigma` 块的 seed_diff_mean、S、σ̂²、σ̂、σ_used：同一个起点两个种子之差，不是水平；保留，核对 σ_used 要用）和三个起点汇总的 p（`p_summary`，按档）；每个起点的绝对数只在闸门报告（GATE.json / GATE.md 的 k/n）里。wave 2b 的 POWER.json（schema /2）不满足这一条：独立核查（statcheck `diag_leak.json`）用 `persistence` 块的 level（Q3B 那对种子的平均 SR）、`sigma` 块和 `p_summary` 精确还原了 SR(Q3B, 100, s1001) − mean(SR(Q3S, 100), SR(E2, 100)) = 1.5 × (level + seed_diff_mean/2 − p_summary["100"].mean)；只删 level 不够，因为 L(1 − L) = B + V，而 B 能从 λ̂ 和 `sigma` 块（D = (n − 1)·S + d²，κ、m2 由 d、S、σ_used 算出）精确倒推，所以去偏的 V 能精确还原 L（V_plain 只能近似）。闭式之所以被挡住，是因为 level、V 和 V_plain 都不写进 POWER.json。B、R、D、κ、m2 也不写，但不是因为它们能还原水平：它们能从 λ̂ 和 `sigma` 块精确算出，写了只是重复。没有 V 时 L 没有闭式解：V ≥ 0 只给出 L(1 − L) ≥ B，即 L 在 [L_min, 1 − L_min] 里，L_min = (1 − √(1 − 4B))/2；L（< 1/2 时）到下界的距离是 (√((1 − 2L)² + 4V) − (1 − 2L))/2，V 小时约 V/(1 − 2L)，所以这个区间多宽由 V 定（推断）。wave-2c 泄漏审查在两个合成根上算过这个界（`OUT/scratch/orch/r3/wave2c/review/leak/out/result_attack.json` 的 `route3_B_bound_on_L`）：statcheck 的 leak_root 上是 [0.232, 0.768]（真 L = 0.395），审查自己的 root2 上是 [0.188, 0.812]（真 L = 0.428），都宽得没有用处。**统计上的预测（推断）**：这不等于 POWER.json 里没有一点关于起点差的信息。用 λ̂、se、`sigma` 块和 `p_summary` 在合成试点上拟合的模型能在统计上预测 SR(Q3B, 100) − mean(SR(Q3S, 100), SR(E2, 100))：wave-2c 泄漏审查（`OUT/scratch/orch/r3/wave2c/review/leak/out/result_mc_narrow.json`：8,000 个合成试点，七成拟合、三成留出，二次项 OLS）里留出样本的 R² ≈ 0.39，差的绝对值 ≥ 5 个百分点时符号对约 84%（起点偏移更散的先验，同一目录的 `result_mc_wide2.json`：0.36、78%）。这不是闭式解；它给的比闸门报告已经打印的每个起点的绝对数（简报允许）少；而且 POWER.json 只在试点之后（冻结之后）才写。现在 `persistence` 块只有白名单里的键（`power_sim.PERSISTENCE_KEYS`）：lambda_hat、se、lambda_used、margin、n、n_pooled、range、rule、sigma_used、replicate_missing、check（`persistence_check` 在汇总的 p 上的模拟结果，没有按起点的值），λ̂ 识别不了或缺种子重复时另有 note（缺种子重复时没有 n、n_pooled 和 check）；`run_tvr` 写之前检查这个白名单，多一个键就拒绝（ValueError）。其余各块：`tables`、`verdicts`、`sensitivity_superseded_test`、`reported_designs` 都是从汇总的 p 模拟的起点，只依赖 p_i、σ_used、λ_used；`registered` 是登记常量；`inputs` 是路径、文件 sha256 和题数。测试 `test_power_json_gives_no_start_level_and_no_between_start_difference` 在 run_tvr 的真实路径上确认这两条还原路线都算不出来，并且除了模拟块、登记常量、输入哈希和 seed_diff_mean，POWER.json 里没有一个数等于某个起点的 SR、这对种子的水平或起点之间的差。
   - 运行方式（登录节点，1 核；用户的登录节点规则）：`env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 19 PY_EVAL -m vpd.common.power_sim tvr --pilot-root P --out P/POWER.json`。模块被导入时也把这四个变量默认设成 1（调用方已设的不改），numpy 路径不调用 BLAS（整组求和用 reduceat），所以在 gate_tvr 里进程内调用也不会开线程池。合成的 500 题根上整张表（两个设计、2,000 次，含 400 次的 λ 核对）在 1 核上约 40 秒（wave 2b 写的，没留出处；wave 2c 在上面"例"里 `fx_time.py` 的根上量的是 37.1 秒）；wave 2c 加了只报告的设计行以后，同一个根（上面"例"里 `fx_time.py` 的合成 500 题根）上 `run_tvr` 从 37.1 秒变成 42.8 秒（量的是 `run_tvr`，1 核，2,000 次，登录节点；`OUT/scratch/orch/r3/wave2c/fix/out/log_time_base.txt`、`log_time_new.txt`），第二轮加了一类错误行以后是 44.7 秒（同样的量法，`OUT/scratch/orch/r3/wave2c/refix2/logs/time_new.txt`；只差文档字符串的前一版代码另量一次 46.6 秒，同目录 `time_new_draft_code.txt`，数字逐位相同）。`gate_tvr --power` 的路径只在测试的小合成根上量过（进程内调用 `gate_tvr.evaluate(..., power=True)`，不是命令行；`test_tvr_integration` 的 pipe fixture：8 题、2 个场景，2,000 次，1 核）：带 `--power` 的闸门 8.8 秒，不带 4.3 秒，同一个根上直接跑 `run_tvr` 3.8 秒（`OUT/scratch/orch/r3/wave2c/refix2/logs/gate_time.json`）；500 题的根上 `gate_tvr --power` 约等于 `run_tvr` 的时间加闸门本身读写文件的时间（推断）。
7. **任务持续性 λ**（`power.persistence`；`power_sim` 的 `persistence_hat`、`shift_gap`、`shift_m2`、`persistence_check`；附 A：O25；wave-2 核查 stats:a4 的 should 项；估计在 wave-2b 核查后改过，见下）：
   - 问题：如果第 6 条的模型让每个 (起点, 种子, 档, 题) 的结果都在汇总的 p_i 附近独立抽，就等于假设"同一个起点不同种子在同一道题上的结果，和不同起点之间一样不相关"。微调后的起点往往对某道题所有种子都会、或者都不会；这部分持续的差在 3 个种子 × 2 档上平均不掉，全进 CR1，所以不加这一项时 A4 的功效偏乐观（检验本身仍然有效，CR1 量的是真实的 Var(d_i)）。
   - 模型（`power.seed_model`）：每个起点每道题一个均匀数 V_{S,i}，这个起点的所有种子和**两档**共用（只是比"两档各自独立的持续性"保守；同一个种子的两档之间的持续性可能比不同种子之间更强，见下面未核实的第 4 条）；每个结果 y = 1[W < q]，q = clip(p_i(T) + c, 0, 1)，W 以 √λ 的概率取 V_{S,i}，否则取新的均匀数。于是 y 的边际仍是 Bernoulli(q)，同一起点同一道题的两个结果（不同种子或不同档，q 相同）的协方差是 λ·q(1 − q)：λ 就是起点内一道题结果的组内相关。不同起点之间独立。λ = 0 时和原来的抽法逐位相同。
   - 估计（Q3B 种子重复，同 500 题，y1 = seed 1001、y2 = seed 1002；另外两个试点起点 Q3S、E2 的 100 档 c 臂结果 y_a、y_b **只通过逐题结果的协方差**进来；不算、不报任何起点之间的差；下面的中间量 L、V（和不去偏的 V_plain）、B、R、D、κ、m2 只在内存里，不写进 POWER.json：L（level）和 V 能还原 Q3B 的 SR 水平（去偏的 V 精确还原，V_plain 只能近似），所以不写；B、R、D、κ、m2 能从 λ̂ 和 `sigma` 块精确算出，不写只是为了不重复；见第 6 条"POWER.json 给什么、不给什么"）：
     - D = mean_i (y1_i − y2_i)²（两个种子逐题不一致的比例）；
     - 参照：同一个起点两个**没有**持续性的种子在同样的水平和逐题难度分布下的不一致率。设 q_i 是 Q3B 在题 i 上的成功概率，B = mean_i q_i(1 − q_i) = L(1 − L) − V，L = mean_i l_i，l_i = (y1_i + y2_i)/2（这对种子自己的水平），V = var_i(q_i)；
     - V 用单因子模型从协方差取（每个起点对同一个逐题难度模式有自己的均值和尺度，q_{S,i} = a_S + b_S·f_i）：C(l, y_a) = b·b_a·var(f)，C(y_a, y_b) = b_a·b_b·var(f)，所以 V = C(l, y_a)·C(l, y_b) / C(y_a, y_b)（C = 跨题的协方差，ddof 1）；再用按场景去一的 jackknife 去掉这个比值的偏差（V_jk = G·V − (G − 1)·mean_g V_{−g}，G = 80 个场景；不去时 λ = 0 处 λ̂ 偏约 −0.03）。各起点的均值（以及尺度）在这里约掉，所以 λ̂ 不随起点之间 SR 的差变；
     - 模型里，给定这对种子实现的偏移 r1、r2（不截断时），E[D] = (1 − λ)·R + λ·|r1 − r2|，R = 2B + (r1 − r2)²/2（B 取这对种子的平均水平 L，所以没有 O(σ) 的项；两个偏移让所有题朝同一方向动，持续部分的不一致率正好是 |r1 − r2|）；
     - κ = E[|r1 − r2| | d, S]（`shift_gap`）、m2 = E[(r1 − r2)² | d, S] = a² + v（`shift_m2`）：在登记的种子模型 r1 − r2 ~ N(0, 2σ_used²) 和 d | (r1 − r2) ~ N(r1 − r2, S) 下的后验，b = 2σ_used² / (2σ_used² + S)，a = b·d，v = b·S，κ = √v · √(2/π) · exp(−a²/(2v)) + a · (1 − 2Φ(−a/√v))（d、S、σ_used 同第 5 条）；R = 2B + m2/2；
     - **λ̂ = (R − D) / (R − κ)**；**λ_used = clip(λ̂ + 0.03, 0, 1)**（0.03 = `power.persistence.margin`，登记的余量，见下面"余量"；`power.persistence.range`、`rule`；`power_sim.lambda_used_rule`）；C(y_a, y_b) ≤ 0（全样本或去掉任一场景后）或 R ≤ κ 时 λ 识别不了，取 1（保守，不加余量）；种子重复缺失（只在 `--allow-no-replicate` 时允许）也取 1（同以前，取范围上端）。POWER.json 记下 λ̂、`margin` 和 λ_used。
     - 另报按场景整组的 delta 法标准误（`persistence.se`，只算题目抽样）。独立核查里它大致对：登记的模型下是 λ̂ 经验标准差的 0.83–1.18 倍；把 statcheck 的所有行都算上是 0.69–1.50 倍（推断；这些行在登记模型的 p 形状之外还有起点之间的差（均值和尺度）、逐题偏向和人为定的 σ_used，最小的 0.69 来自人为定 σ_used 的一行，最大的 1.50 来自起点均值偏移的一行；statcheck `summary.json`、`est_*.json` 的 mean_delta_se / sd_hat；本轮重算：`OUT/scratch/orch/r3/wave2c/refix2/logs/se_ratio.txt`）。
     - **余量**（wave 2c；编排按独立核查的建议定，DECISIONS G14 的补记）：λ̂ 剩下的偏松偏差（λ̂ 平均偏低，λ_used 偏低，功效被高估，A4 可能误过）在登记的模型下某些 p 形状上是 −0.012 到 −0.020，在最坏的起点偏移情形下到 −0.036（推断，statcheck：合成 p 上的模拟）；这和 A4 的决定窗口一样大（σ_used = 0.03 时 λ 约 0.005–0.01 起功效就低于 0.80，见下面"说清楚"）。所以 λ_used 在截断之前先加 0.03。加了以后，按登记的 σ_used 规则核查过的每一格里调整后估计的平均都 ≥ λ − 0.006（推断，statcheck；最低的一格是有起点偏移的 −0.036 + 0.03）。核查里另有把 σ_used 人为定为 0.01、真 σ = 0.03 的格不满足这一条（到 −0.024，λ = 0.8），这种 σ_used 按登记的 σ_used = max(0.03, σ̂)（第 5 条）不会出现。A4 因此更难过，这要分开看。真 λ = 0 时这是真正的代价：那里的真功效 0.813 ≥ 0.80（`power_curve.json`），A4 通过的概率却从约 0.5 降到约 0.34–0.44。λ = 0.1 / 0.2 时真功效是 0.767 / 0.748 < 0.80（`power_curve.json`），那里 A4 通过就是误过；通过的概率从 0.14–0.24 / 0.03–0.08 降到 0.10–0.18 / 0.02–0.06，正是余量要的效果，不是代价（statcheck `tables.txt` 的 OC 表：P(λ_used ≤ 0.01)，λ_used ≤ 0.01 近似 A4 通过；四种 p 形状，σ = 0.03 和 0.05；推断；本轮重算：`OUT/scratch/orch/r3/wave2c/refix2/logs/se_ratio.txt`，λ = 0 时 0.341–0.435 按四舍五入写成 0.34–0.44，0.435 = 870/2000）。`persistence.check.result` 另报 `margin` 和 `bias_used` = λ_used 的平均 − λ（含余量和截断）。
     - **为什么改**（wave-2b 核查 review stats，`identity.txt`、`maineffect*.json`）：wave 2 登记的估计用 H = m/(m − 1)·mean_i p_i(1 − p_i) 当参照（p_i = 三个起点的平均）。这个 H 恰好等于三对起点之间逐题不一致率平均的一半（恒等式，数值差 0.0），期望里还含三个起点 SR 的方差，而且 D 是在 Q3B 自己的水平上；所以它其实是"起点之间对起点之内"的估计，随起点之间的差变：Q3B 是最好的起点、差 5 个百分点时 λ̂ 偏约 −0.10（λ_used 偏低，功效被高估，A4 可能误过），Q3B 最差时偏 +0.06 到 +0.31。这违反了 G13 "A 类判据不看起点之间的差"的本意，冻结前改掉；新的估计仍然不算起点差，只在协方差里把各起点的均值约掉。
   - 模拟核对（推断：`persistence_check`：数据按登记的模型抽，再走 run_tvr 的同一条估计路径；500 题 80 个场景，3 个试点起点，σ = 0.03；review stats 的同一种合成 p 的 100 档（三分之一的格点，偏向 0 和 1）、连续的 Beta(0.6, 1.4) p 和更居中的 Beta(2, 3) p；这一组是 wave 2b 的数，λ_used 都是不加余量的 clip(λ̂, 0, 1)）：
     - 起点之间没有差时（每格 2,000 次）：λ = 0 / 0.3 / 0.6 / 0.9 的 λ̂ 平均偏差 −0.016 / −0.012 / −0.005 / −0.003（连续 p：−0.015 / −0.006 / −0.002 / −0.002；蒙特卡洛标准误 0.001–0.003），λ̂ 标准差 0.13 / 0.11 / 0.08 / 0.04（连续 p：0.11 / 0.10 / 0.07 / 0.04；wave 2 的估计 0.12 / 0.10 / 0.08 / 0.04）。偏差是比值估计剩下的二阶项，在标准差的 13% 以内；λ = 0 时 λ_used（截在 0）的平均 0.04。σ = 0.05 时偏差 −0.022 到 −0.008（连续 p：−0.017 到 −0.006），λ_used 的平均 0.04 / 0.28 / 0.59 / 0.89；那时 A4 本来就过不了（σ_used ≥ 0.05）。
     - 起点之间有差时（`persistence_check(offsets=…)`，按实现 SR 尺度的偏移经 calibrate_shift 加到各起点，Q3B 最好、居中、最差，±5 和 ±10 个百分点，13 种组合 × 3 种 p，λ = 0 和 0.3，每格 1,000 次，同一组随机数）：新估计的 λ̂ 平均偏差都在 −0.025 到 +0.017 之间（λ = 0.3 时 λ_used 的偏差 −0.010 到 +0.017），wave 2 的估计在同样的数据上 −0.17 到 +0.31。代价是标准差大 5–45%（例如 Beta(2, 3) p 上 0.09 对 0.07，三分之一格点的 p、Q3B 居中 ±10 个百分点时 0.14 对 0.10）。
     - 同一份固定数据上只让一个非重复起点多会 k 道题（随机把 k 道失败改成成功，review stats 的做法）时，新估计的变化是噪声（k = 10 / 25 / 50：+0.05 / +0.02 / −0.08，方向不定），wave 2 的估计一直往上（+0.01 / +0.07 / +0.12）。用同一组均匀数把 E2 的 100 档结果按更高的阈值取（加性的起点差，E2 多会约 6% 的题，2,000 题，Q3B 的两个文件不变；`test_run_tvr_lambda_ignores_a_better_start` 的构造），20 个种子上新估计的变化平均 −0.005、标准差 0.017，wave 2 的估计平均 +0.064（最小 +0.032）。
     - 另一组同样的核对（wave 2，σ = 0.02–0.03）：κ 用 √max(0, d² − S) 时偏差 −0.001 到 −0.009，用 |d| 时 −0.005 到 +0.008；后验均值在 σ_used 等于真 σ 时最准，真 σ 小于 σ_used 时略偏高（保守），所以登记的是后验均值。λ̂ 的标准差约 0.1，对应功效约 ±0.03（推断）。run_tvr 每次也在试点自己的 p_i(100)、λ_used、σ_used 下跑 400 次这个核对（起点之间没有差），写进 `persistence.check`（只报告；从 p̂ 再模拟一次会让方差再缩一次，所以那里的标准差偏大）。
     - 独立核查（wave-2b statcheck，`OUT/scratch/orch/r3/wave2b/statcheck/out`：`summary.json`、`tables.txt`；五种合成 p 形状，每格 2,000 次，有起点偏移时 1,000 次；推断）：登记的模型下 λ̂ 的偏差在约 ±0.02 以内（σ = 0.03 时 −0.013 到 +0.013）；有起点偏移时最偏松的情形 −0.025 到 −0.036；λ 小时 λ̂ 的标准差 0.10–0.14（σ = 0.03）；登记的模型下 delta 法标准误是经验标准差的 0.83–1.18 倍（所有行算上 0.69–1.50 倍，见上面 `persistence.se`）。这些数支持上面的余量。
     - 需要有逐题难度的差别：所有题难度几乎一样时（C(y_a, y_b) 接近 0）λ̂ 识别不了，取 1（保守）。TVR 的四类题难度差得很多（第 13 节 G3 的论文数字 82.4 / 68.8 / 27.2 / 24.8），预计不会碰到（推断）。
   - **对 A4 的影响**（推断：合成 p，σ = 0.03，ρ = 0.5，每格 2,000 次；Q3S − E2 / Q3RD − mean(QRL)）：

     | λ | 0 | 0.05 | 0.1 | 0.2 | 0.4 | 0.6 | 0.8 |
     |---|---|---|---|---|---|---|---|
     | 主设计功效 | 0.81 / 0.90 | 0.78 / 0.89 | 0.77 / 0.88 | 0.75 / 0.86 | 0.70 / 0.82 | 0.64 / 0.78 | 0.60 / 0.74 |
     | 备用设计功效 | 0.50 / 0.64 | — | — | 0.45 / 0.58 | 0.41 / 0.53 | 0.39 / 0.50 | 0.35 / 0.45 |

     族错误率随 λ 下降（主设计，ρ = 0 / ρ = 1：λ = 0 时 0.045 / 0.044，λ = 0.4 时 0.035 / 0.042，λ = 0.8 时 0.025 / 0.026），一类错误仍受控。原因（review stats `vdecomp.json`，零模拟 2,000 次）：V 仍然无偏（Var(D)/E[V] = 0.97–1.05，λ = 0 和 0.8、ρ = 0 和 1），但 CR1 在 V 里的份额随 λ 从 0.23 升到 0.54（ρ = 0）、从 0.13 升到 0.38（ρ = 1），V 的 Satterthwaite 自由度从约 20 升到 47（ρ = 1：16 到 29），而检验固定用 df 12（t 临界值 2.18，对 2.01–2.05）；所以有持续性时检验偏保守（族错误率约 0.03），少一点功效，一类错误仍然有效。不改规则。一个按模型抽的合成试点（真 λ = 0.5，σ = 0.03）上，run_tvr 估出 λ̂ = 0.47（scene 整组 SE 0.10；wave 2 的估计在同一份数据上是 0.56），主设计功效 0.74 / 0.86，判 `power_insufficient`（wave 2b 的数，当时 λ_used = λ̂ = 0.47，不加余量；按现在的规则 λ_used = 0.50）。
   - **说清楚**（wave 2c 按独立核查改过；原来写的阈值"约 0.02"偏高）：独立核查的功效曲线（statcheck `power_curve.json`；合成 p，σ = 0.03，ρ = 0.5，8 个百分点，主设计，每格 2,000 次，种子 0）里，Q3S − E2 的功效在 λ = 0 / 0.005 / 0.01 / 0.02 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5 / 0.8 时是 0.813 / 0.799 / 0.800 / 0.793 / 0.781 / 0.767 / 0.748 / 0.720 / 0.662 / 0.597，蒙特卡洛标准误约 0.009；族错误率 0.039–0.048。所以 σ_used = 0.03 时 λ_used 只要大于约 0.005–0.01，Q3S − E2 在 ρ = 0.5 的功效就低于 0.80，A4 判 `power_insufficient`（第 13 节决策表：不进确认轮，交用户定）；λ = 0 时的 0.813 离 0.80 也只有约 1.5 个蒙特卡洛标准误（0.013 / 0.0087；种子 1 上是 0.796）。这些都是合成 p 上的数，对真实试点是推断。λ 小时 λ̂ 的标准差约 0.10–0.14，所以真 λ 接近 0 时 A4 主要由估计噪声决定；加了余量以后，真 λ = 0 时 A4 通过的概率约 0.34–0.44（上面"余量"；推断）。真实的微调起点 λ 多半明显大于 0（未核实），所以加了这一项以后 A4 很可能过不了；这是功效模型变诚实的结果，不是闸门出错。到时候由用户定的方向（只列出来，推断）：更多种子（例如每个起点 5 个种子：λ = 0.4 时功效约 0.89 / 0.96，λ = 0 时 0.97 / 0.99，df 24；卡时约多 2/3，会超过第 15 节的 100 卡时规则；POWER.json 的 `reported_designs` 在试点自己的 p_i、σ_used、λ_used 下报 4 个和 5 个种子的功效和族错误率，选了的设计也要满足 ≤ 0.06 的一类错误界，第 6 条）、更大的登记效应量（`reported_designs` 也报 δ = 0.10）、或者不把 TVR 当尺子。
   - **未核实**：
     1. λ 只从 Q3B 的 100 档估，假设其他起点、TOP 档和确认轮的起点持续性一样。
     2. 单因子假设：各起点只在同一个逐题难度模式的均值和尺度上不同。均值和尺度的差（包括截断造成的压缩）上面核对过；超出这个模型的、起点特有的逐题偏向，Q3B 自己的那部分按定义就算持续性（λ̂ 照样把它算进去）。Q3S 和 E2 共有、Q3B 没有的偏向只抬高 C(y_a, y_b)，所以 V 偏小、λ̂ 偏高（偏保守）：独立核查（statcheck `tables.txt` 的 (c) 组：两个非重复起点共用一个逐题偏向，尺度 τ = 0.5 / 1.0 / 1.5，σ = 0.03，每格 1,000 次，三种 p 形状；推断）里 λ = 0 时，τ = 0.5 偏约 +0.04（到 +0.05），τ = 1.0 时 +0.10 到 +0.13，τ = 1.5 时 +0.14 到 +0.19，即 +0.04 到 +0.19，λ 越大越小（λ = 0.5 时 +0.02 到 +0.10）。wave 2b 这里对这种偏向写的方向（"V 偏大、λ̂ 偏低（偏松）"）是反的。两个非重复起点共有的偏向只让 A4 更难过。另一种情形偏松：Q3B 只和 Q3S、E2 里的一个共有的逐题偏向 h（例如 Q3S 保留了 Q3B 的一部分逐题习惯、E2 没有；两者都是 SFT 过的模型（第 4 节），可能各自保留不同份额的原始模型的逐题习惯）只抬高 C(l, y_a)，于是（两者尺度相近时）V̂ ≈ b²·var(f) + var(h)，Q3B 自己持续的那部分被当成难度，λ̂ 偏低，大约偏 h 在组内相关里所占的份额，可能抵掉一部分余量（推断，未核实：statcheck 的 (c)、(d)、(e) 组分别是两个非重复起点共有、只有 Q3B 有、各起点各自独立的偏向，没有测这种情形）。真实的偏向有多大仍未核实。
     3. 模型把汇总的 p_i 当真值：p_i 是 3 个结果的平均，mean p_i(1 − p_i) 平均只有真实 Bernoulli 方差的 2/3，p_i 为 0 或 1 的题在模拟里没有方差，这是 `power.p_source` 原有的、偏乐观的一处，本轮没改（影响大小未核实）。
     4. 同一个种子两档之间的持续性：模型让一个起点里任何两个结果（不同种子，或者同一个种子的 100 和 TOP）共用同一个 λ。种子重复（seed 1001 对 1002，各自的 0W）只量得到**不同种子之间**的持续性；确认轮里同一个种子的 100 和 TOP 适配器共用 0W_s 和数据顺序，逐题的一致可能更高，种子重复识别不了。按 review stats 的核对（`crosstier.json`：给同一个种子的两档另加一个共用的均匀数；2,000 次，σ = 0.03，ρ = 0.5），Q3S − E2 的功效偏乐观：不同种子之间 0.2、同种子两档之间 0.2 / 0.4 / 0.5 时 0.755 / 0.737 / 0.724；不同种子之间 0、同种子两档之间 0.3 / 0.6 时从 0.789 降到 0.773 / 0.745。也就是同种子两档多出 0.2–0.6 的持续性约损失 0.02–0.04 的功效；族错误率不受影响（0.037–0.047）。不改规则，也不加敏感性行。
     5. 两个 QRL 起点：第 15 节第 5 条把 QRL-s1 和 QRL-s2 当作两个 RL 种子的重复，但模型给每个起点独立的 V。两个 QRL 起点共有持续的逐题偏向时（review stats `qrl.json`：第 6 个起点用第 5 个的 V），Q3RD − mean(QRL) 的功效从 0.854 降到 0.837（λ = 0.2）、0.809 到 0.780（0.4）、0.725 到 0.667（0.8），即偏乐观 0.02–0.06；现在起约束作用的是 Q3S − E2，所以不影响 A4 的判定；族错误率不变。

---

## 13. 闸门规则（契约 8.1）

所有规则都和结果无关，由 `vpd/tvr/gate_tvr.py` 判定。它读：`P/<起点>/evals/pilot__<起点>__{0W, 100, N_top}__c`、`pilot__<起点>__<N_top>__u`、`pilot__<起点>__<N_top>__blind`（以及 `pilot__Q3B__100s1002__c`：G0、G2 照样检查它，它的结果只进 A4）；适配器的 meta.json；每次合并评测的 `P/<起点>/parity/merge_<档>.json`；`P/<起点>/canary/*/render_parity.json`；最新的 `smoke_verdict.json`（和 `SM/replay2/render_parity.json`）；demos 的 manifest、kin_check、dropped.json；`tiers/v1/manifest.json`；`floors_eval500.json`；`CAL/cal_decision.json`（G0 第 13 条）和 `CAL/shake/fallback_used.json`（有的话，G0 第 10 条）；`P/POWER.json`（A4）；pilot_cfg。写出 `P/GATE.json` 和 `P/GATE.md`：只有绝对数字、Wilson 区间、下限和判定，**不计算起点之间的差**（GATE.json 带 `"note": "no between-start differences are computed"`）。

### G0 机制（全部满足）
1. 每次列出的评测有效（results.json）：500 题，0 缺。
2. 模板：冒烟的 B5 在线 vLLM 比对和 parity_check 每个起点都 PASS；每次图像评测的运行内提示 token 比对相等。
3. 每张图 222 token（网格 [1, 22, 40]）：每次 dry run 和每次图像评测的 token 审计。
4. 渲染器：重放和每次图像评测的 results.json 的 `renderer` 块里，档案、字体 sha、L 等于 `render.*`；重放的 E 等于 `render.replay_envs`，每次图像评测的 E 等于 `render.eval_envs`（各自对照自己的键）。这几个值是框架的请求；另外 `renderer.workers_ok` 必须为真，并且**每个** worker 自己的 ready 回复（`renderer.workers`）：档案 == `render.lvp_profile`、L == `render.lp_threads`、字体 mode 为 pinned 且 sha == `render.badge_font_sha256`、`system_prompt_ok` 和 `tvrbench_files_ok` 为真（系统提示和 TVRBench 文件的 sha 等于冻结值）（`pilot_tvr.worker_problems`）。
5. 代码：三个起点的 0W 共用一个 code_sha256，以 `0865dd75f1ee` 开头；100 档和最高档从本起点的 0W 开始（init sha），它们的 code_sha256 等于本起点 0W 的（种子重复的 0W_s1002 和 100s1002 也对照 Q3B 的 0W）；每次评测的 `results.code_sha256` == `results.code_fingerprint.code_sha256`，所有评测用同一个评测框架 code_sha256。这条"同一个框架"只管评测：数据作业的重放在评测之前跑，可以带更早的 run_tvr 指纹，不参与比较。
6. kin_env = ThorEnv：每条保留的重放上完全一致（`kin_equal_all_kept`，硬性）；图像臂评测的任务上，`valid_list` 类不一致为 0（注入失效），位姿 / 成功类不一致的任务 ≤ 每次评测任务数的 1%（`gates.kin_eval`；附 A：O23）；所有计数和前几个不一致都报。
7. 合并 = LoRA：**每次**合并试点评测（0W、100、最高档，以及 Q3B 的 100s1002）都做 teacher-forced 检查，在 0W 池的 20 条示范上：平均 |ΔCE| ≤ 0.005，回复 argmax 一致率 ≥ 0.99（`gates.merge_parity`；WalkerBench x483 的参考值 0.00072 / 344 of 345；附 A：O2）（2026-11-11 起按第 17 节的偏差改为只检查合并是否正确：服务的权重和合并公式逐位相同、fp32 合并误差 ≤ 1e-4、模块集合一致；这里的 bf16 数照报不判）。
8. 每次合并评测用的是盘上现在的 adapter.pt（sha 相等）。
9. 注意力 sdpa（adapter meta 的 attn_sites），训练 meta 和 vLLM mm kwargs 里 max_pixels = 1,024,000。
10. 评测设置等于 `protocol`（results.settings），包括 blind_concurrency、server 块（enforce_eager、seed 0、generation_config vllm、xgrammar；max_images）和 versions 块（对照 results.server）。results.settings 本身是从 `protocol` 抄的，所以另外按臂判 run_tvr 记下的**本次实际值**：`max_tokens_used` == `protocol.max_tokens[臂]`，`regex_used` == `protocol.regex`（c、blind 臂；u 臂为 null），`blind_concurrency_used` == `protocol.blind_concurrency`（blind 臂；其他为 null），`request_fields` 等于登记的请求字段（c、blind 臂另有 structured_outputs）（`gate_tvr.used_settings`）。max_model_len：所有评测服务的值相同，并且等于 `protocol.max_model_len`，或者在冻结前已登记 O9 偏差（第 17 节保留行里写了 `CAL/shake/fallback_used.json` 的 sha256，`gate_tvr.o9_fallback`）时**每次**评测都等于备用值；没登记却有这个文件、或者登记了却按 32768 跑，都不过；每次评测的 `settings.max_model_len_used` 等于服务实际的值。
11. 每道评测题用的都是冻结集合（每题的 reach digest 等于 index）。
12. TVRBench 自己的 `eval_va.aggregate()` 在每次评测的轨迹上给出和 score_tvr 相同的成功率（official_check 相等；证明轨迹格式对）。
13. 适配器来源（对应 PREREG_WB_PILOT G0 第 6 条）：每个适配器 meta 的起点 == S，种子 == 登记值（1001；种子重复 1002），阶段 == 档位标签，优化步 == 计划步，data_sha256 == tiers/v1/manifest 里那个档文件的 sha，并且那个 manifest 的 sha == `data.tiers_manifest_sha256`；训练和判的最高档就是校准规则的输出：`tiers.n_top` == `CAL/cal_decision.json` 的 `n_top`（N_top = 400 时 `with_400` 为真；`pilot_tvr.n_top_problems`，文件缺了算缺）。
14. 下限文件有效（对应 WalkerBench 第 8 条）：sha == `data.floors_eval500_sha256`，warnings == []，F == `data.F_eval500`。
15. 渲染一致（简报 A1"训练和评测的渲染一致"）：冒烟里以 root 跑的 sft2 重放、每个试点评测作业的金丝雀，都按冻结的容差规则对照数据作业的重放通过（第 3.2 节），并且 states / valid / user 文本 / action_success 完全相等。
16. 示范重放（简报 A1"示范能重放成功"）：重放的 results.json 有效、1,600 条轨迹都在；K ≥ 1,520；每条剔除都有 G10(a) 的原因。
17. 独立复算（简报第 6 节）：gate_tvr 直接从每次被判评测的 `tasks/*/trajectory.json`（`result.success`、`result.stopped`）重数 k/n 和每类，必须等于 results.json；A3 的 b、c 按 task_id 从两次运行的轨迹配对；recheck_tvr 的下限等于 floors_eval500.json（第 11 节）。

只记录、不判：现场集合 == 冻结集合的计数、不稳定的可达集合任务、每个渲染一致性文件里逐帧的 PSNR / SSIM。

G0 不过是工程问题，不算科学失败：修好，重跑受影响的步骤，在第 17 节记一条。

### 冒烟（tvr-msmoke；契约 7.3）
每个起点在 dev 的 smoke8 上跑 raw 模型的 c、u、blind 三臂，**只看机制**：轨迹里的 `result.success` 写成 null，不算也不报成功率（附 A：O14）。`smoke_verdict.json` PASS 当且仅当每个起点：dry run 网格都是 [1, 22, 40]、每图 222 token；parity_check PASS；vllm_parity 相等；每个臂的运行有效、0 缺题，c 和 blind 100% 正则匹配；u 臂跑了（解析率报告，不判）；运行内提示比对相等、token 审计 222；c、u 臂 `renderer` 块的档案、字体 sha、L 等于冻结值，`renderer.workers_ok` 为真，并且每个 worker 自己的 ready 回复满足 G0 第 4 条同样的检查（档案、L、字体 pinned 且 sha 相等、`system_prompt_ok`、`tvrbench_files_ok`；`pilot_tvr.worker_problems`）；并且 `SM/replay2/render_parity.json` 通过（以 root 渲染对照非 root 的数据作业，容差规则，不比 sha）。smoke-verdict 从不读成功字段（有测试）。

### 无望停止（校准之后，只用 Q3B；第 7 节）
`cal_decision.futility` 为真时，编排不准备冒烟和训练作业，报告并问用户。

### G1 学得会（G10(g)）
- **至少一个起点**满足：wilson_lb(k_top_c, 500) **>** max(F, SR_0W_c(本起点))，其中 SR_0W_c 是本起点 0W 主臂评测的点估计。
- 0W 门槛的理由：F 接近 0，只比 F 的话 3/500 就能过（第 11、12 节），说明不了"学会了"（critic C5）。

### G2 格式
每个起点的每次 c 臂和 blind 臂评测：100% 的回复 `re.fullmatch(REGEX)`，没有日志的题为 0。u 臂的解析有效率只作描述，不卡闸门（DECISIONS B3）。

### G3 没饱和
每个起点最高档 c 臂的成功率 ≤ 0.80（≤ 400/500），点估计，没有下移档位条款。参考：论文里 Qwen3.5-9B 全参数 VA-SFT 的总成功率是 50.8%（82.4 / 68.8 / 27.2 / 24.8），LoRA r16 的 8B 模型超过 80% 的可能性小（推断，tvr_data_paper）。

### A3 用到了图（只报告）
每个起点：最高档 c 对最高档 blind，同 500 题，第 12 节第 2 条的精确检验；报 b、c、p 和"images_better_p_lt_0.05"。不卡闸门（简报第 2 节 A3）。文字里的"Valid actions"行含碰撞信息、不含目标信息，所以预计差别大（推断，prereg_confirm）。

### A4 有量程
G0–G3 都过以后才跑第 12 节第 6 条的功效模拟（检验 = 第 12 节第 4 条的终点层面种子感知检验；任务持续性 = 第 7 条的 λ_used）；通过当且仅当：在 σ_used、λ_used 下，ρ = 0.5 时两个对比 8 个百分点的 Holm 功效（只数处理起点更好的拒绝）都 ≥ 0.80，**并且** δ = 0 时 ρ = 0 和 ρ = 1 的模拟族错误率都 ≤ 0.06（取两者中较大的判）。ρ = 0 和 ρ = 1 的功效、σ = 0.05 的行、λ 的敏感性行、作废的旧检验和 `reported_designs` 的设计行（4、5 个种子，δ = 0.10；以及 4、5 个种子在 ρ = 0、1 下的族错误率 `reported_designs.type1`；第 12 节第 6 条）都只报告。A4 不过以后用户要是选了其中一个设计，它在 `reported_designs.type1` 里也必须满足登记的一类错误界（ρ = 0 和 ρ = 1 两格里较大的 ≤ 0.06，`by_design.<设计>.type1`）；δ = 0.10 的设计在 δ = 0 时就是主设计，看 `tables.main` 的 `type1` 行。A4 记 "pending"（决定是 `incomplete`，不判过或不过）的情形：
- G0–G3 没全过；
- `power_sim` 失败（`gate_tvr --power` 时）；
- POWER.json 过期（`gate_tvr.power_stale`）：它记的某个 per_task.jsonl 的 sha256 变了，A4 要的某个 per_task.jsonl（三个起点的 100、N_top 档 c 臂和 100s1002）不在它的输入里，或者 `inputs.pilot_cfg_json_sha256` ≠ sha256(现在的 pilot_cfg.as_json())；
- POWER.json 不是按登记设置算的（`gate_tvr.power_unregistered`）：没有 `registered` 块；`registered.test` ≠ `power.test`；`registered.n_sims_registered` 或 `n_sims_used` ≠ `power.n_sims`（`--n-sims` 只供测试）；`registered.tier_corr`、`fallback`、`persistence` 和 `power.*` 不同；`registered.allow_no_replicate` 不是 false；`sigma.replicate_missing` 或 `persistence.replicate_missing` 不是 false（缺种子重复）；`registered.backend` ≠ `power.backend`（numpy；第 12 节第 6 条）。

  只有不是 pending 的 POWER.json 才按它的 `pass` 判过或不过。登记的备用设计（两档种子 1–2）按同样的规则判，结果是 `fallback_pass`（只在第 15 节第 9 条用）。按第 12 节第 7 条的核对，λ 稍大于 0 时 A4 就会判"功效不够"，交用户定。（2026-11-11 起，A4 不过时的走法由用户预先授权，见第 17 节）

### 决策表

| G0 | G1 | G2 | G3 | A4 | 决定（GATE.json `decision`） |
|---|---|---|---|---|---|
| 不过 | — | — | — | — | `engineering`：修好重跑，不判科学结论 |
| 过 | 任意 | 不过 | 任意 | — | `engineering`：见 G2 的处理；查不出实现问题就不进确认轮，交用户（G0 或 G2 不过时先判 `engineering`，不判科学结论） |
| 过 | 不过 | 过 | 任意 | — | `not_a_ruler`："按本配方学不会"，不进确认轮；改不改设计由用户定 |
| 过 | 过 | 过 | 不过 | — | `not_a_ruler`："最高档已饱和"，不自动下移，交用户 |
| 过 | 过 | 过 | 过 | 不过 | `power_insufficient`：不进确认轮，交用户 |
| 过 | 过 | 过 | 过 | 过 | `ruler`：TVR 是合格的尺子。按第 15、16 节估确认轮卡时，≤ 100 卡时就开跑并通知用户，> 100 就停下问 |
| 缺评测 | | | | | `incomplete`（gate_tvr 退出码 3） |

- 几条同时不过时全部列出。A3 不进决策表。决定里从来没有起点差。第 17 节 2026-11-11 登记的 Q3S − Q3B、Q3S − E2 比较只报告，在闸门和功效之后单独算，也不进决策表。

---

## 14. 试点里允许改的东西

只有这些可以改，改了都记第 17 节：
1. 超时、截止时间、重试次数（第 16 节的规则之内）。渲染的 E、L 和 blind 臂的线程数已冻结在 `render.*` / `protocol.*` 里，冻结后不能改；
2. 纯工程修复（G0 / G2 那类）：修复只动实现、不碰下面"不能改"的任何一项时，可以先修、先重跑，批准人一栏写"待用户事后确认"；用户确认之前，修复后的闸门结果不能用来做任何决定（同 PREREG_WB_PILOT 第 10 节第 4 条）；
3. 登记好的备用值：max_model_len 128000（O9，只能在冻结前定）。

**不能改**：指标、起点、配方、档位（含校准规则和 n_top 的结果）、种子、闸门阈值、评测协议（`protocol` 的全部，含解码、历史、上限）、渲染器和 `render.*`（冻结后 eval_envs 也不能再改）、可达集合的定义、下限的定义、统计方法，以及第 9.4 节的必做评测。碰到其中任何一项就是真正的偏差，必须先经用户批准。

---

## 15. 现在就登记的确认轮条目（冻结；附 A：O5、O24）

只有试点决定为 `ruler` 时才进确认轮，每个任务只跑一次确认轮（简报第 2 节 B）。下面是功效模拟模拟的、也是确认轮要跑的设计（`power.design`）：

1. **起点**：Q3B、Q3S、E2、Q3RD、QRL-s1、QRL-s2（同 WalkerBench）。Q3RD、QRL-s1、QRL-s2 还没有评测视图，能不能直接用现在的训练器训未核实，确认轮之前先查（ROUND3）。
2. **档位**：{100, TOP}，TOP = 试点冻结的 N_top；权重 0.5 / 0.5（`power.design.weights`）。
3. **种子**：两档都用种子 1、2、3。每个 (起点, 种子 s) 先训一个 0W_s（种子 s），作为该种子各档的初始化。只评 0W 种子 1（每起点一次，描述用的锚点，不进终点）。
4. **终点**：每个起点、每道题 Σ_T w_T · mean_s y_{s,i}；500 题 c 臂（主协议，同第 9 节）。
5. **对比**：(a) **Q3S − E2**；(b) **Q3RD − mean(QRL-s1, QRL-s2)**（两个 QRL 种子当作重复）。Holm，一族两个检验，双侧 α = 0.05。
6. **检验**：第 12 节第 4 条的终点层面种子感知场景整组 t 检验（`seed_aware_endpoint_t`）就是**唯一的确认检验**；σ̂_E² 从确认轮自己的 6 个起点 × 3 个配对种子的终点里估（df 12；备用设计 df 6），所以第 3 条让两档共用 0W_s 和种子造成的两档相关已经含在里面（DECISIONS G13）。每个 (起点, 种子, 档) 的 500 题 c 臂评测都必须在，缺了补跑，不插补。场景 bootstrap 只作描述。冻结后不能再改。
7. **结论三选一**（`power.outcome`，min_effect = 0.05 = 登记的最小效应量）：
   - 某个对比"支持" = Holm 拒绝，并且处理起点更好；
   - "不支持" = Holm 拒绝但处理起点更差，**或者** 95% t 区间的上界 < 0.05；
   - 其他 = "无定论"。
   - 总体：**支持** 当且仅当至少一个对比支持、并且没有一个对比被 Holm 反方向拒绝；**不支持** 当且仅当两个对比都不支持；其他为 **无定论**。
   - 实现：`vpd/common/power_sim.py` 的 `confirmatory_outcome`。
8. **卡时估计**（契约 O5，推断）：每个 (起点, 种子) 训练 0W 0.13 + 100 0.15 + ALL 约 2.1 h ≈ 2.4 h，18 个 ≈ 43 卡时；评测 100 c + ALL c 含合并、合并检查、vLLM 启动约 1.0–1.2 h，约 20 卡时；0W 种子 1 的评测约 3。合计中值约 66，预期 55–75，**est_high 约 85**（TOP = ALL）；TOP = 400 时约 35–40。
9. **100 卡时规则**（简报第 4 节"预估不超过 100 卡时就直接开跑，同时发消息告诉我。超过就停下等我"）：确认轮开跑前，用试点实测重算 est_high（第 16 节的公式）。≤ 100 就开跑并通知用户；> 100 时，用登记的**备用设计**——两档都只用种子 1–2（每个起点不训、不评种子 3 的 0W_3、100_3、TOP_3；权重不变；推断约省 21 卡时），检验同第 6 条（df 6），前提是 POWER.json 里备用设计也过了 A4（`fallback_pass`）；备用设计仍 > 100，或者备用设计没过 A4，就停下问用户。按合成 p 的核对，σ ≥ 0.03 时备用设计的功效约 0.50 / 0.64（ρ = 0.5、λ = 0），λ > 0 时更低（第 12 节第 7 条），预计过不了 A4，所以 est_high > 100 实际上就是停下问用户（推断）。
10. 试点数据不进确认分析（第 0 节）；试点的适配器不复用。

---

## 16. 卡时规则（`budget`）

- **每个作业的超时**（契约 5.3.1；照 WalkerBench 的规则，在 pilot_tvr 里重写，用 `vpd.walkerbench.pilot` 的常量）：外层 `timeout` = floor(est_high × 3600) − HEADROOM_S（300 秒）；作业自己的软截止 = 外层 − SOFT_RESERVE_S（180 秒）；kill 窗口 60 秒（数据、冒烟）或 120 秒（其他）；外层 < MIN_OUTER_S（600 秒）就拒绝；est_high > `budget.job_est_high_max_h`（5.5 h）就拒绝，停下问，从不悄悄放大。
- **固定的 est_high**（契约第 7 节）：tvr-data 2.5（外层 8,700 / 软 8,520）；tvr-cal 5.5（19,500 / 19,320）；tvr-cal400 1.5（5,100 / 4,920）；tvr-msmoke 1.0（3,300 / 3,120）。
- **训练和评测的 est_high 由校准的 Q3B 实测推出**（和起点无关）：
  - train-<s>：est_high = 1.3 × (本作业各档的 [校准 meta 的 train_seconds + load_seconds] 之和) / 3600 + 0.1；每档取校准里同一档 Q3B 适配器的值（0W_s1002 / 100_s1002 用校准的 0W / 100；N_top = 400 用 cal400 的 meta）。预计 ALL 约 3.0 h（Q3B 带种子重复约 3.3；校准后推出 2.759 / 3.068，第 7 节"校准结果"），400 约 1.25（Q3B 约 1.55）。
  - eval-<s>-top / -low：est_high = Σ_本作业的评测 [vllm_ready_s_max + steps_high / R + merge_s + parity_s] / 3600 + 0.15；R = 校准里图像臂评测 `results.perf.steps_per_s` 的最小值（blind 臂用冒烟排查里 blind 的步速），merge_s / parity_s = 校准里合并 / 合并检查秒数的最大值，steps_high = min(eval500 的上限总和 17,500, 1.5 × 500 × 校准 dev200 同档的 mean_steps_per_task)（0W → 校准 0W，100 → 校准 100，TOP → 校准 TOP；u 臂用 TOP；100s1002 → 校准 100）。推出来 > 5.5 h 就拒绝，停下问。
- **账本**：每个作业的卡时记进 `OUT/ledger.jsonl`（vpd.sco）。
- **本轮 200 卡时上限**（简报第 2 节 C、第 5 节）：本轮已用 = 账本累计 − `budget.round_baseline_gpuh`（8.8404，本轮开始时的累计）。`pilot_tvr prepare` 在 本轮已用 + 账本里在跑 / 已提交作业的 est_high 之和 + 新作业的 est_high > `budget.round_cap_gpuh`（200）时拒绝（退出码 2），打印这三个数并记进 `RUN/prepared.json["round_budget"]`。原因：`ledger.budget_check` 在 BUDGET_POLICY.json 没设上限时总是放行（契约 P-S6）。到了 200 就停下，给用户总报告。
- **试点预留**（契约 7.6，est_high；训练和评测是预期值）：数据 2.5 + 校准 5.5（+ cal400 1.5）+ 冒烟 1.0 + 训练约 3.0 × 3（+0.3）+ 评测 top 约 2.4 × 3 + low 约 2.2 × 3（Q3B +0.9）≈ 33.0 卡时（N_top = ALL；预期实际 21–27；训练和评测换成校准后按第 16 节推出的 est_high（合计 26.3）时预留是 35.3，见第 7 节"校准结果"）；N_top = 400 约 29。2026-11-10 16:08Z 本轮已用 1.883 卡时，另有在跑作业的 est_high 6.3（ROUND3）。
- **确认轮**：第 15 节第 9 条的 100 卡时规则。

---

## 17. 偏差日志

登记时只有下面几条种子条目。种子条目是冻结**之前**的设计选择：第 1 条由用户批准，第 2–4 条由编排按 2026-11-10 的授权（DECISIONS G12）记下，第 5 条由编排按同一授权（DECISIONS G13）记下，第 6 条只在用了时记。冻结**之后**的每条偏差，批准人只能是用户本人，并且要在本线窗口里确认（同 PREREG_WB_PILOT 第 12 节）；第 14 节第 2 条的纯工程修复例外（先写"待用户事后确认"）。

| 日期（UTC） | 改了什么 | 为什么 | 批准人 |
|---|---|---|---|
| 2026-11-10 | 渲染器用 lavapipe（Mesa CPU Vulkan），不用 NVIDIA GPU Vulkan；示范和评测都用它 | 作业容器里 NVIDIA Vulkan DEVICE_LOST；11-10 的 Vulkan 复查两个池都没通（DECISIONS G2、G3(a)；SMOKE_CRITERIA.md 偏差日志，4487309d） | 用户（2026-11-10，简报第 4 节"改用 CPU 渲染的偏差"） |
| 2026-11-10 | lavapipe 的帧不是逐位可复现的（跨进程、跨 L、跨路径、重载后都可能不同）：渲染一致性改按容差规则判（PSNR ≥ 40 dB 且 SSIM ≥ 0.99，加上 states / valid / user 文本 / action_success 完全相等），帧的 sha256 只作文件完整性记录；不用帧缓存；重放和评测用同一个 L 和档案 | 探针实测 deterministic false（第 3.2 节，契约 F4、P-M1）；按 sha 判会让冒烟按构造失败 | 编排（2026-11-10 的授权，DECISIONS G12）；属于简报第 4 节批准的 CPU 渲染偏差的执行细节 |
| 2026-11-10 | 可达集合：冻结的按任务、在起点位姿取的集合，在数据作业里取一次，注入到重放和每个评测臂（附 A：O1） | 官方的值依赖任务顺序和起点位姿（契约 F1、F2）；G10 没说评测用哪个集合 | 编排（DECISIONS G12） |
| 2026-11-10 | 合并 = LoRA 的检查加到**每次**合并试点评测（0W、100、最高档、100s1002），进 G0（附 A：O2） | 简报 A1 要求"合并后的模型和 LoRA 一致"；G10(g) 的 G0 漏了 | 编排（DECISIONS G12） |
| 2026-11-10 | 确认检验改为终点层面的种子方差（`seed_aware_endpoint_t`，df 12）；A4 的一类错误在 ρ = 0、1 判，功效在 ρ = 0.5 判；备用设计改为两档种子 1–2（df 6）；功效模型加任务持续性 λ（Q3B 种子重复的逐题不一致率，对照别的试点起点逐题结果的协方差给出的、同一水平下没有持续性的不一致率；对起点均值不变，wave 2b 第二轮修订；`power.backend` = numpy 也登记）；校准作业要求全部 `data.*`（附 A：O25）；wave 2c（2026-11-11 写入）：λ_used = clip(λ̂ + 0.03, 0, 1)（登记的余量），POWER.json 不给起点的 SR 水平（schema /3），加只报告的设计行（4、5 个种子，δ = 0.10；4、5 个种子另报族错误率） | wave-2 核查（stats:a4、xcut:protocol）：原检验假设两档偏移独立，ρ = 1 时族错误率 0.17–0.20；原备用设计（只 TOP 用种子 1–2）两档不配对，终点检验用不了；功效模型忽略起点内持续的逐题差，A4 功效偏乐观；校准要训练，K 和剔除名单必须先登记。修订时还没有任何 TVR 模型输出。wave 2c 的依据是 wave-2b 的独立统计核查：λ̂ 剩下的偏松偏差（到 −0.036）和 A4 的决定窗口一样大；wave 2b 的 POWER.json 能还原 Q3B 对另外两个起点的 SR 差；核查时（2026-11-11 00:5x–01:2x UTC，statcheck 输出文件的时间）盘上仍没有 TVR 模型输出，核查只用合成 p | 编排（2026-11-10 的授权，DECISIONS G13；wave 2c 记进 G14 的补记） |
| （备用，只在用了时记） | max_model_len 改用 128000（O9） | 校准排查时 32768 / 96 张图的服务起不来；必须在冻结前记，之后每次试点评测都用 | — |
| 2026-11-11 | G0 第 7 条改为只检查合并是否正确。对每次合并试点评测用到的适配器（0W、100、最高档，以及 Q3B 的 100s1002）：(i) 服务的权重和合并公式逐位相同：用同一个 `vpd/train/merge.py` 重做合并，每个合并模块等于 `n2_train_vpd.merge_adapter` 的结果，适配器以外的张量等于检查点的 bf16，没有多的或少的键（诊断的 served_ok）；(ii) fp32 合并误差 ≤ 1e-4：fp32 下合并的模型 MF 对训练出的函数 T，在 0W 池同样 20 条示范、267 条回复上 teacher-forced 的平均绝对 ΔCE；(iii) 模块集合一致：lora.wrap 包的 = 适配器的 = merge.py 合并的（252 个）。原来的 bf16 比较 U−M（平均绝对 ΔCE ≤ 0.005、回复一致率 ≥ 0.99，`gates.merge_parity`）照报，不判。证据是诊断作业 `vpd-tvr-mpdiag`（EXPLORATORY）的输出：Q3B 0W、Q3S ALL、E2 ALL 在 `OUT/tvr/runs/explore/mpdiag/r3-202611111828`，其余 7 个适配器补跑；`gate_tvr` 的 G0 第 7 条按这三项判。现有评测都保留。`pilot_cfg.py` 不改（它在评测框架的指纹里），1e-4 写在 `gate_tvr.py` | 诊断（pt-6kl8ci5s，0.669 卡时）：合并代码正确（三个适配器 T−MF 为 1e-6 到 2e-6，服务权重逐位等于公式，模块 252 = 252）；训练器自己的 bf16 前向离训练出的函数就有 0.008–0.010（平均绝对 ΔCE），已超过 0.005，原阈值低于 bf16 的数值噪声，没有纯工程修法；ALL 档的合并模型和训练器的前向一样准（Q3S、E2 判 noise），Q3B 0W 判 inconclusive（bf16 只存下约 74% 的小增量，所以 0W 档服务的模型带的增量比训练出的少，每个起点同样服务）。决定时没有人读过任何评测的成功率（DECISIONS G15a、G16） | 用户（2026-11-11，本线窗口） |
| 2026-11-11 | 登记主报告的比较 Q3S − Q3B（只报告，不进闸门、决策表和 A4）：对每一档（0W、100、最高档 ALL）的每个臂（0W、100 档只有 c 臂；ALL 档 c、u、blind 三个臂，共 5 格），两个起点 seed 1001 的同一批 500 道评测题按 task_id 配对，成功取轨迹的 `result.success`（同 G0 第 17 条的口径）；报两边的 k/n 和 Wilson 95% 区间、只有 Q3S 成功的题数 b、只有 Q3B 成功的题数 c、第 12 节第 2 条的双侧精确检验 p。Q3S − E2 同样 5 格照报，作补充（看起作用的是动作训练，还是见过我们的场景）。参照行：Q3B 100 档 seed 1001 对 seed 1002，同法（看同一个起点两个种子之间差多少）。不做多重比较校正。这些 p 只算了题目抽样，没算训练种子之间的波动（每个起点每档只有一个种子；第 12 节第 4 条在合成设计上的核对：只算题目抽样的检验在种子 SD 0.03 时名义 0.05 的一类错误约 0.44–0.46），所以偏乐观，只作描述，不是确认轮的结论（推断）。计算顺序：GATE.json、POWER.json 写完之后，由单独的脚本算，另一个独立脚本复算；GATE.json 仍然不算起点之间的差 | 本线的成功标准是在非我们的任务上验证在我们任务上训过的权重有效；写入时没有人读过任何评测的成功率 | 用户（2026-11-11，本线窗口）；配对检验沿用第 12 节第 2 条，参照行、计算顺序和"偏乐观"的说明是编排按用户的决定补的细节（只报告） |
| 2026-11-11 | A4 不过时的预先授权：闸门判 `power_insufficient`（G0–G3 都过、A4 不过）时不停下问用户，按 POWER.json `reported_designs` 里每个起点 4 个和 5 个种子的设计行（`main_seeds_4`、`main_seeds_5`：σ_used、λ_used、ρ = 0.5、δ = 0.08、Holm、只数处理起点更好的拒绝），选两个对比的功效都 ≥ 0.80、并且 `reported_designs.type1` 里同一设计 ρ = 0 和 ρ = 1 两格较大的族错误率 ≤ 0.06（第 13 节 A4 已要求）的最少种子数；两个都不满足就停下问用户。按第 15、16 节的规则估这个种子数下确认轮的总卡时，≤ 60 卡时就直接开跑并通知用户，> 60 卡时停下问。加种子的做法和冻结块的确认轮一致（起点、档、配方、检验、Holm、结果规则都不变，两档用同一组种子 1–n 配对，df = 6 × (n − 1)），只改种子数；用的时候在本节另记一条 | 用户事先定好 A4 不过时的走法 | 用户（2026-11-11，本线窗口） |
| 2026-11-12 | 探索（EXPLORATORY；不进闸门、决策表、A4 和确认轮）：两个 RL 起点按试点完全相同的配方训练和评测。Q3RD = `c0q3_s1d_300`：从 Q3S（`c0q3_sft_e8`，checkpoint-1232）出发的 C0 动作 RL，奖励 D_cond（稠密 V4：势场塑形，γ 0.99，scale 3.33），GRPO（n 8、lr 2e-6、KL 0.05、熵 0.01），种子 20261019，第 300 次更新。Q3RS = starts.json 的 `Q3RS150` = `c0q3_s1s0_150`：同一 SFT 起点、同一 GRPO 配置，奖励 S0（稀疏），第 150 次更新时因无法解析的回答占比 0.106 > 0.1 自动停止，预登记的检查点就是 150。两者都是 verl 导出的 float32 权重，载入时转 bf16（和 Q3S、E2 一样）；没有评测视图，处理器从检查点目录读，处理器和模板文件和 Q3S 检查点逐字节相同。配方：种子 1001、同一 0W 热身数据和档位数据、训练器指纹 0865dd75f1ee、评测框架指纹 64edd575f36b、同一 eval500 和协议；每个起点 0W / 100 / ALL，评测 top（ALL 的 c、u、blind）和 low（0W、100 的 c）；输出在 `runs/explore/` 下，不进 `runs/pilot/`。主报告的比较：Q3RD − Q3B、Q3RS − Q3B；补充：Q3RD − Q3S、Q3RS − Q3S（看 RL 相对 SFT 修回多少）。5 格同 2026-11-11 登记的比较（0W c、100 c、ALL c、u、blind），同一批 500 题按 task_id 配对，报 b（只有前一个起点成功）、c（只有后一个成功）、第 12 节第 2 条的双侧精确 p、两边 k/n 和 Wilson 95% 区间；Q3B、Q3S 用已有的试点评测，不重跑；不做多重比较校正；p 只算题目抽样，偏乐观（推断）。新适配器按 G0 第 7 条（2026-11-11 的新规则）做合并正确性诊断，只报告。比较由单独的脚本算，另一个独立脚本复算 | 试点里 Q3S 在 TVR 上低于 Q3B；先看动作 RL 起点是否高于 Q3B、RL 相对 SFT 修回多少。写入时这两个起点在 TVR 上还没有任何评测，没有人读过它们的成功率 | 用户（2026-11-12，本线窗口） |
| 2026-11-13 | 探索（EXPLORATORY；不进闸门、决策表、A4 和确认轮）：第三个探索起点 Q3SG 按试点完全相同的配方训练和评测。Q3SG = starts.json 的 `Q3SG` = `sft_q3genc0_e8`（genc0）：基座 Qwen3-VL-8B-Instruct（0c351dd），C0 动作 SFT 的轨迹（9,802 条，和 Q3S 同一文件，sha 511ff945）加 22,000 条通用 VQA 混训，第 8.0 轮 = checkpoint-3976（`VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_q3_20261027/gen3b_sft_20261107/c0g/model/checkpoint-3976`）；处理器、分词器、config 和权重布局和 Q3S 的检查点逐字节相同，上游配方只多改了 image_max_pixels = 65536；没有评测视图，处理器从检查点目录读；载入时转 bf16（和 Q3S 一样）。配方：种子 1001、同一 0W 热身数据和档位数据（同一 sha）、评测框架指纹 64edd575f36b、同一 eval500 和协议；每个起点 0W / 100 / ALL，评测 top（ALL 的 c、u、blind）和 low（0W、100 的 c）；输出在 `runs/explore/rl/Q3SG/` 下（沿用已有的探索根，"rl" 只是目录名），不进 `runs/pilot/`。训练器代码一个字节不改；starts.json 只多 Q3SG 这一条，所以 Q3SG 作业的训练器指纹是 119da3c84b8d（5 个 .py 文件的 sha 和试点、RL 适配器记录的相同；用原来的 starts.json 重算仍是 0865dd75f1ee）；Q3SG 的作业核对新值，比较核对"5 个 .py 文件 sha 不变、starts.json 只多这一条"。登记的比较（主）：Q3SG − Q3B，100 档和 ALL 档的 c 臂；诊断：Q3SG − Q3S，同两格。compare_rl 同口径的 5 格（0W c、100 c、ALL c、u、blind）都照报，作描述。同一批 500 题按 task_id 配对，报 b（只有前一个起点成功）、c（只有后一个成功）、第 12 节第 2 条的双侧精确 p、两边 k/n 和 Wilson 95% 区间；Q3B、Q3S 用已有的试点评测，不重跑；不做多重比较校正；p 只算题目抽样，偏乐观（推断）。预判（读数之前写；100 c、ALL c 两格分别判）：Q3SG 的 k ≥ Q3B 的 k → 缺口主要是遗忘，回放能补；Q3SG − Q3S 不显著（p ≥ 0.05）且 Q3SG − Q3B 显著为负（p < 0.05）→ 缺口来自动作任务本身；其他情况 → 介于两者之间，不判；两格结论不同就分别照报，不合并。新适配器按 G0 第 7 条（2026-11-11 的新规则）做合并正确性诊断，只报告。比较由单独的脚本算，另一个独立脚本复算 | 把"我们的任务本身不利于迁移"和"只训我们的任务导致遗忘"分开（补充 1.2）。写入时 Q3SG 在 TVR 上还没有任何评测，没有人读过它的成功率 | 用户（2026-11-12 补充 1.2；2026-11-13 选 A：starts.json 加 Q3SG；本线窗口） |

---

## 附 A. 和 DECISIONS G10 不同、或 G10 没规定的地方（采用的 O 项）

O1–O24 是 wave-2 契约第 10 节的开放决定，编排按 2026-11-10 的授权照契约的推荐采用（DECISIONS G12，"编排按 2026-11-10 的授权自定"）；O25 是 wave-2 核查之后编排按同一授权做的决定（DECISIONS G13）。都在冻结前可逆：

- **O1** 可达集合：冻结的按任务起点位姿集合，数据作业里取，注入重放和每个评测臂；kin_env、BFS、下限、blind 臂用同一个；现场集合照样记录；G0 加 "kin_env == ThorEnv（评测任务上）"（第 3.4 节、第 13 节 G0 第 6、11 条）。G10 没规定评测用哪个集合，这是缺陷修正。
- **O2** 合并 = LoRA 的 teacher-forced 检查加进 G0，每次合并试点评测都做（G0 第 7 条）。
- **O3** "220 个 image token"：每帧 220 个 `<|image_pad|>` = B4 口径下的 222（prod(grid)/4 + 2）；两个数都登记，检查用 222（第 10 节）。
- **O4** 种子方差：Q3B 种子重复（0W + 100，seed 1002，500 题 c），σ_used = max(0.03, 实现尺度的估计)（第 12 节第 5 条）；同一个重复也估任务持续性 λ（O25）。
- **O5** 确认轮设计现在登记：6 个起点、{100, TOP}、种子 1–3、权重 0.5 / 0.5、0W_s 每种子训、只评 0W 种子 1、种子感知整组 t + Holm、三分结论、备用设计（第 15 节；检验和备用设计按 O25 改过）。
- **O6** 每个起点一个训练作业 + 评测作业（不是一个作业跑完）（第 9.4 节）。
- **O7** 所有模型作业都以 root 跑；渲染一致性按容差规则对照非 root 的数据作业重放（第 13 节 G0 第 15 条）。
- **O8** 校准的 Q3B 适配器**不**复用到试点：试点的所有适配器都由冻结后的作业在同样的守门规则下产出（第 6 节）。
- **O9** vLLM 32768 / 96 张图起不来时的登记备用值 max_model_len 128000，冻结前记偏差（第 9.3 节）。
- **O10** E × L：重放和可达集合导出用 summary.recommended（5 × 2）；图像臂评测用同一个 L，E = `render.eval_envs` = 5（校准后按 Q3B 测速确认，第 7 节"校准结果"；第 3.2 节）。
- **O11** 试点不跑 raw（未微调）评测（第 7 节）。
- **O12** 冻结集合里目标不可达的题留在所有分母里（官方协议），在 bfs 的 n_infeasible 里计数并报告；eval500 上预计 0（第 10 节）。
- **O13** 种子重复训自己的 0W（seed 1002），不复用 seed 1001 的 0W（第 6 节）。
- **O14** 冒烟在 8 道 dev 题上跑 raw 的 Q3S / E2（和 Q3B）：不算也不报成功率，轨迹里 `result.success` 为 null，没有工具读冒烟的成功字段；逐步日志留在盘上，但不汇总（第 13 节冒烟）。
- **O15** wave1c 的 `vpd/e3vs/__init__.py` 修复的落地时机（不影响本文件）。
- **O16** 数据作业只用 CPU，但跑在 GPU 池上（GPU 空闲，约 1.5 卡时记账）；没有已知的纯 CPU 池（第 16 节）。
- **O17** 下限细节：固定策略 k = 0..39、"k + 1 > cap 永远停不下来"；随机策略走类别公开专家长度的 median_low 步后停；平手规则（第 11 节）。
- **O18** 不约束的 u 臂 max_tokens 64（eval_va 默认），c 和 blind 臂 16（第 9.2 节）。
- **O20** 评测**总是**拆成 -top 和 -low 两个作业（一条代码路径；每个作业最坏约 3.1 h，校准后推出的最大值是 Q3B low 的 3.662 h，第 7 节"校准结果"）（第 9.4 节）。
- **O21** 校准之后的无望停止：Q3B 最高档 dev 的 Wilson 下界 ≤ dev 盲下限时先问用户（第 7 节）。
- **O22** 校准和每个试点评测作业开头都跑渲染金丝雀（第 9.4 节；lavapipe 是 CPU JIT，换节点的 CPU 可能改变帧）。
- **O23** kin-on-eval 的容差：valid 列表不一致 0 次（硬性），位姿 / 成功不一致 ≤ 每次评测 1% 的任务（第 13 节 G0 第 6 条）。
- **O24** 确认轮"不支持"的门槛 min_effect = 0.05（最小的登记效应量）（第 15 节第 7 条）。
- **O25**（DECISIONS G13；wave-2 核查 stats:a4 和 xcut:protocol 之后、冻结之前）A4 和确认检验：(a) 唯一的确认检验改为终点层面的种子感知整组 t 检验（`seed_aware_endpoint_t`，df = 起点数 × (n_s − 1) = 12）；原来按 (起点, 档) 格估种子方差的版本（df 24）假设两档偏移独立，ρ = 1 时族错误率 0.17–0.20，作废，只作敏感性报告；(b) A4 的一类错误在 ρ = 0 和 ρ = 1 下判（取较大的 ≤ 0.06），功效在 ρ = 0.5 下判（ρ = 0、1 只报告）；族错误率的定义确认；(c) 备用设计改为两档都只用种子 1–2（配对，同一检验，df 6）；(d) 功效模型加任务持续性 λ：Q3B 种子重复两个种子的逐题不一致率，对照同一水平下没有持续性时的不一致率；后者的逐题难度分散 V 用种子重复的平均行和另外两个试点起点逐题结果的协方差按单因子模型取（每个起点有自己的均值和尺度，按场景去一的 jackknife 去偏），所以 λ̂ 不随起点之间 SR 的差变，也不算、不报起点之间的差（wave-2b 核查 review stats 发现 wave 2 登记的参照 H 等于起点之间不一致率的一半，起点差 5 个百分点时 λ̂ 就偏约 ±0.1；第二轮修订改掉，要编排补一条 G13 修订）；λ_used = clip(λ̂ + 0.03, 0, 1)（wave 2c 加了登记的余量 `power.persistence.margin` = 0.03，抵消 λ̂ 剩下的偏松偏差；识别不了或缺种子重复时仍取 1），λ = 0 和 0.8 的敏感性行只报告；(e) 校准作业 tvr-cal、tvr-cal400 要求全部 13 个 `data.*` 已填；(f) `power.backend` = numpy 登记（两个后端的随机数流不同），POWER.json 的 schema 升到 `vpd.power_sim.tvr/2`（wave 2c 再升到 /3，见 (g)）；(g) wave 2c（wave-2b 的独立统计核查之后、冻结之前；DECISIONS G14 的补记）：(d) 的余量；POWER.json 不给任何起点的 SR 水平、不给能只从 POWER.json 用闭式算出起点之间差的值（`persistence` 块只留白名单里的键：删掉能还原水平的 level、V、V_plain，以及能从 λ̂ 和 `sigma` 块精确算出、写了只是重复的 B、R、D、κ、m2；统计上的部分预测仍可能，推断，第 12 节第 6 条），schema 升到 `vpd.power_sim.tvr/3`；只报告的设计行 `power.reported_designs`（POWER.json 的 `reported_designs`：每个起点 4 个、5 个种子，和 3 个种子 δ = 0.10；第二轮另报 4、5 个种子在 ρ = 0、1 下的族错误率 `reported_designs.type1`，A4 不过后选的设计也要满足 ≤ 0.06）；λ 阈值、功效曲线和未核实第 2 条的方向按核查改正，第 2 条另补一种核查没测、偏松的情形（第 0 节，第 0.1 节，第 12 节第 4–7 条，第 13 节 A4，第 15 节第 6、9 条）。

不在本文件里的：O19（冻结的 WalkerBench 确认轮统计方法没考虑训练种子方差）要改的是已冻结预注册的统计方法，按简报第 5 节由用户定；它不挡 TVR，本文件不涉及 `PREREG_WB_PILOT.md` 的任何统计。

其他和 G10 的关系：G10(a)–(j) 全部照搬；G10(g) 的 G0 在上面 O1、O2 之外，按契约补了渲染一致（容差规则）、示范重放、适配器来源、下限文件、独立复算（第 13 节 G0 第 13–17 条），这些都是简报 A1 和第 6 节已经要求的检查的落实，不改任何阈值。
