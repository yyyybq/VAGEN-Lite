# D2 失败分解和 H3 判定：runbook（读 cell C 结果之前写定）

**2026-11-25 定稿，随 PREREG_Q38 v1.2 提交。** 01:xxZ 初稿，之后按三轮复核改过（美东 11-24 晚）。第 5 节的 R1–R17，用户 11-25 约 06:1xZ 回"按推荐"，全部取推荐项（`DECISIONS_20261124.md` 第 4 节）。11-25 约 21:5xZ 闸门第 3 轮加固跑完（新登录节点，uid 10174），第 3 节、第 4.4 节和第 6 节的 sha256、规则和测试按最终版更新。用户 11-24 规划文档 v2 第 8 节第 4 条，D2 原话："手工：从 VAGEN-Lite-matrix 只读拿，写 runbook，读结果前做"（PREREG_Q38 v1.2 第 3.8 节"失败分解（D2）"一行）。

## 0. 这份文件管什么

**写本文时的状态：**
- cell C（wave C）4 条 RL 都到了第 300 次，评测跑完。
- cell C 的 val、test、探针结果一个都没读：results.jsonl、table、selection、done.json 都没开过（复核时只用 `test -f` 查过存不存在）。读过的只有 tasks.jsonl、model.json、plan.json、链状态，以及 COMPLETE 的时间戳（D8 人工核对用）。链状态 progress.json 里带着训练时的成功率，这是训练遥测，不是评测，看链状态时看到了。
- C0 一边（主线 4 条）为了填稳不稳表，读了 progress.json 的结构、k_v4resume 的停止和重启日志、COMPLETE 的时间戳，没读第 300 次的任何评测结果。
- cell D 没跑。

**本文定两件事：**

| 部分 | 对应 8B 的 | 出什么 | 什么时候跑 |
|---|---|---|---|
| A 失败分解表 | 矩阵链的 `diagnostics` 步，即 `eval.diagnostics.failure_breakdown` | 每格、每题集一份描述表：按族的类别，差哪个条件、哪一组，停在哪；另带换目标重放（wrong target） | 用户批准读该格结果以后 |
| B H3 判定 | `m3_prereg_rest.py` 第 3.4 节，加 SFT 阶段的 3.4 | D − C（主）、D − C0（次）的 ①、② 和"下降"标记 | cell D 评完，且用户批准读结果以后 |

H3 不读 A 的输出，而是像 8B 那样从 results.jsonl 逐题重算。所以 A 只是描述，不进判定。

## 1. 代码从哪来

**来源：** VAGEN-Lite-matrix `f5cd03b6`，branch matrix-v3，工作树干净。用 git archive 只读拷到 `$U/q38/tmp_tests/k_d2/src/matrix_f5cd03b6/`，拷了 `eval`、`data_gen`、`vagen` 和 `docs/transition_qa/matrix_v3/scripts`。拷之前和拷之后，矩阵工作树都是干净的。

| 文件 | sha1sum（拷贝 = 工作树） | 说明 |
|---|---|---|
| `eval/diagnostics/failure_breakdown.py` | 6f3a3dd7（git blob 10ff6df1） | 8B 跑 H3 时矩阵在提交 ec9aeb61，这个文件当时的 git blob 是 036b9c57。之后 50ca6cc9（11-07）只加了 `--wrong-target` 这个开关和它的输出；classify、GATE_GROUP、BUDGET_ENDS、FAMILY_ORDER、breakdown 都没变 |
| `eval/diagnostics/wrong_target.py` | d0d34d45 | 11-07 加的。8B 的 H3 没用它 |
| `docs/transition_qa/matrix_v3/scripts/m3_prereg_rest.py` | beed378c | 和 `exps/matrix_v3/watch/` 里那份逐字相同；8B 的 rest.json（11-03 10:58Z）就是它写的 |
| `exps/matrix_v3/watch/m3_supp_rest.py` | 没进 git，原文拷在 `k_d2/src/m3_supp_rest.py.untracked_copy` | 8B 探索性格子（D_noP0、Cr、B'）用的写法 |

**为什么不在 27B 工作树里跑：** VAGEN-Lite-q38mx 和 VAGEN-Lite-q38 都没有 `eval/diagnostics`，链里也没有 diagnostics 步。q38-mx-c-rl 的计划写的是 `diagnostics: null`，所以 27B 从来没跑过失败分解。

## 2. 和 27B 评测记录合不合（已查，没读 cell C 结果）

- **字段：** failure_breakdown 和 wrong_target 要读的每个字段，27B 记录里都有。用主线 q38s0s1_050 查的：292 行、1,946 个回合，没有 error 行。cell C 的两个 tasks.jsonl（292、382 道）字段齐全，缺 0 个。
- **评测开关：** cell C 的 50 个 model.json（test 25 个、val 25 个）都只开了 `prompt_format_example: action_names`。没有 turn footprint、grounding probe、critic probe，也没有 self marker。
- **代码差异：** 把矩阵 f5cd03b6 和 q38mx a2e069ca 对比，以下几项没有差异会改变 classify 和换目标重放在 cell C 上的结论：
  - 几何模块逐字相同：difficulty、scene、occlusion、occlusion_alignment、target_visibility；
  - check_collision 没变；
  - end_reason 的取值集合相同；
  - 回合记录里 collisions、met、failed_gates、end_reason 的含义相同。
- **兼容测试：** 在主线 q38s0s1_050 的 test 上跑了一次（非判定点，第 50 次，不属于 cell C）。
  - rc 0，用时 36 s，errors 0。
  - 换目标重放：52 个失败的换目标族局里，2 个没过自检（collisions_match、gates_match）。原因是 env 的 CollisionDetector 和生成器的 FastGeometry 对单步移动判得不一样，8B 每个模型也有 0–4 个这种情况。这两局不计入换目标的比例，记作 unchecked。

## 3. A：失败分解表

**脚本：** `$K/scripts/run_fb.sh`，其中 `K=$U/q38/tmp_tests/k_d2`，sha256 2e0fa3d1…（复核后第 3 轮修改版，全值见第 4.4 节）。它对应 8B 矩阵链在 f5cd03b6 的 `diagnostics()`（train_chain.py:2329-2385）。对每个题集跑一次：

```
python -B -m eval.diagnostics.failure_breakdown --eval SET_DIR --names <给的模型里在 SET_DIR 有 results.jsonl 的> --out OUT/failure_<set> --wrong-target
```

- 从 archive 副本跑：cwd 和 PYTHONPATH 都是副本。
- 跑之前先查核数，"已用 + 1 ≤ 12"才跑，并加 nice 19、ionice idle，各线程变量设成 1，TMPDIR 在 k_d2。
- failure_breakdown 会把每个模型的类别计数打到 stdout，这些就是结果。所以 stdout 和 stderr 一律写进 `OUT/logs/failure_<set>.log`，终端上只出现路径和 rc。
- 不带 `--go` 时，脚本在碰结果文件夹、起 python 之前就拒绝（rc 3）。拒绝的情况有两种：题集或某个模型文件夹（按真实路径和字面规范化路径都查）在 `q38/runs/q38-mx-*-rl` 下，或者名字是 27B 的判定点、挑点。
- 第 2、3 轮复核后加的规则（PORT_NOTES 的 "fix round 2""fix round 3" 两节，逐条对应到行号）：
  - 进脚本先用 `env -i` 重起一次，只留白名单里的环境变量（PATH=/usr/bin:/bin、HOME、LANG、各线程变量 = 1 等）；有别的变量、shell 函数或别的 PATH 就拒绝。所以调用时不要设 BASH_ENV，也不要 export K_D2_CLEAN。
  - 查核数挪到最前面（rc 4），闸和起 python 之间不再有等待。
  - 路径逐跳解析（不用 `readlink -f`），任何一跳经过 /proc、/dev、/sys，或链接文字里有控制字符、结尾换行，都拒绝；解析后按 python 实际的 cwd 再核一遍，原路径和解析后路径的 dev:ino 必须相同。python 只拿到核过的绝对路径，用 `-I -B` 跑。
  - 当前进程的挂载必须和普通登录进程一样（uid_map 是全映射，用户目录下没有额外挂载），否则拒绝。
  - `--wrong-target` 之前先跑 `scripts/scene_check.py`：tasks.jsonl 里每个 scene_id 必须是简单名字，场景文件夹和 labels.json、structure.json 不能是链接或硬链接；GS_ROOT 固定为 InteriorGS_ply，VAGEN_GS_ROOT 等变量被清掉。
  - OUT 必须不存在，或者是一个空的、不是链接的文件夹；每次起 python 前再核一遍（recheck）。
  - 以上都只在不带 `--go` 时起作用。带 `--go` 时只多了入口的 `env -i` 一步，后面那段和第 1 轮版本逐字节相同（tests/rerun3_ast/go_block.sh 核过）。

**cell C 的命令（用户批准读 cell C 结果以后才跑）：**

```
U=/mnt/umm/users/yinbaiqiao; K=$U/q38/tmp_tests/k_d2; R=$U/q38/runs/q38-mx-c-rl
NAMES=sft_q38mxc_e9,$(for r in q38mxcs0s1 q38mxcv4s1 q38mxcs0s2 q38mxcv4s2; do for s in 050 100 150 200 250 300; do printf '%s_%s,' $r $s; done; done | sed 's/,$//')
mkdir -p $K/cellC
bash $K/scripts/run_fb.sh --go $K/cellC/diag "$NAMES" test=$R/sets/q38mxrc_eval_test_v2_screened val=$R/sets/q38mxrc_eval_val_v2_screened > $K/cellC/run_fb.out 2>&1
```

（复核发现：没有 `mkdir -p $K/cellC` 时，shell 打不开重定向文件，脚本根本起不来，rc 1。这是安全的失败，但命令照写跑不通，已补上。）

- 一共 25 个名字：SFT 起点，加 4 条 RL × 6 个检查点。
  - 这是矩阵 f5cd03b6 的 `evaluated_models()`（train_chain.py:2284-2303）用在 cell C 两个计划上的结果，只核了名字：s20261019 出 13 个（两条 RL 各 6 个，加 `eval.models` 里的 sft_q38mxc_e9），s20261021 出 12 个，合起来正好是上面 25 个。
  - 这不是 8B 当时实际出的表。8B 的 diag 是 11-03 07:36–08:24Z 跑的，比"只读本链结果"的过滤（9a90c309，11-03 18:06Z）早约 10 小时。所以 8B 每份 failure_{test,val}.json 里都是共享评测文件夹的全部 68 个模型，也没有换目标重放。见 R12。
  - 8B 是两条链（两个种子）各跑一次；这里把两个种子合在一起跑一次。failure_breakdown 每个模型单独算，所以每个模型的数和分开跑相同。
- 输出：`$K/cellC/diag/failure_{test,val}.{json,md}`、`failure_{test,val}_wrong_target.jsonl`、`logs/`。
- 耗时：一个文件夹约 36 s。25 个 × 2 个题集估计 15–30 min，单进程。
- 读结果时报两项自检：
  - 每个模型的 errors 数（failure_breakdown 跳过 error 行，单独计数）；
  - 换目标重放里 unchecked 的局数，即 collisions_match 或 gates_match 没过的。

**这张表只描述，不进判定。** H3 照 8B，从 results.jsonl 逐题重算（第 4 节）。

## 4. B：H3 判定

### 4.1 判定用哪个检查点（PREREG 第 4.2 节；先从链状态定，不看结果）

**规则：**
- RL 读第 300 次。
- 被停止线停下的，用它最后一个检查点；一个检查点都没有的，用它的 SFT 起点。
- 按 val 挑的最好检查点，只作探索性报告。

**cell C 的"稳不稳"表：** 数据来自 `q38-mx-c-rl/s2026101{9,21}/progress.json` 的 jobs、queue 和 rl_runs.last_update，以及 COMPLETE 的时间戳。都没有碰结果。

| 运行 | 交了几次（训练 / 渲染） | 守卫停了没有 | 最后一次更新 | 终点 COMPLETE（UTC） | 训练任务结束状态 | 判定用 |
|---|---|---|---|---|---|---|
| q38mxcs0s1（s1 S0） | 1 / 1 | 没有 | 300 | 11-24 17:38Z 前（s1 最晚一个 17:38:34Z） | SUCCEEDED | q38mxcs0s1_300 |
| q38mxcv4s1（s1 V4） | 1 / 1 | 没有 | 300 | 同上 | SUCCEEDED | q38mxcv4s1_300 |
| q38mxcs0s2（s2 S0） | 1 / 1 | 没有 | 300 | 11-24 17:51:57Z 前（s2 最晚一个） | SUCCEEDED | q38mxcs0s2_300 |
| q38mxcv4s2（s2 V4） | 1 / 1 | 没有 | 300 | 同上 | SUCCEEDED | q38mxcv4s2_300 |

- 24 个检查点（050–300 × 4）都有 COMPLETE，而且每个都写在本链评测开始之前：s1 的评测 17:40:09Z 开始，s2 的 17:54:16Z 开始。
- **D8 (b) 是评测之后补查的：** PREREG 和 DECISIONS 写的是"评测前人工查 COMPLETE"，但链在上面两个时间自动开了评测，人工核对是在 11-25 01Z 前后比的时间戳。比的结果是 24 个 COMPLETE 全部早于评测开始，和评测前查结论一样。这种补查算不算 D8 (b)，见 R15，要用户确认。
- **停止线：** 和 8B 矩阵实际生效的那套相同：kl_max 1.0；invalid_max 0.1，窗口 10；success_drop 0.1，start_updates 5，drop_window 50；entropy_max 0.3，窗口 5；vision_token_crashes 2；NaN 检查。cell C 另加了两条 8B 代码里没有的：host_mem_max_gb 1600 和 gen_slow，都没触发。
- **重试：** 各计划 retries 8。progress.json 里没有 guard_stops、job_retries、startup_retries、stall_warnings 这几个键，queue 里每条运行只有一次训练、一次渲染。
- **update_seen：** q38mxcv4s1、q38mxcs0s2 的 update_seen 是 299，last_update 是 300。这是记账方式造成的，不是少了一次更新：
  - update_seen 只在链每 60 秒轮询一次、而且 sco 还显示训练任务 RUNNING 时才写（q38mx a2e069ca train_chain.py:1100-1126）；
  - last_update 每次保存时从训练日志重算（:314-339、:1334-1346）；
  - 日志里 `step:300` 那一行写在第 300 次的检查点保存之后，进程约 20 秒后退出。这两条运行在 COMPLETE 之后的第一次轮询时，训练任务已经结束，链把它移出了 active，update_seen 就停在 299。
- SFT 起点：sft_q38mxc_e9（`checkpoint-2709`，用户 11-24 定）。

**C0 一边（主线 4 条）的"稳不稳"表：** 数据来自 `q38-main-rl/s2026101{9,21}/progress.json`（只看结构）、`k_v4resume/logs_r2/` 的停止和重启日志，以及 COMPLETE 的时间戳。主线代码是 VAGEN-Lite-q38 c2c6ed47，没有 endpoint_race_fix。

| 运行 | 交了几次（训练 / 渲染） | 停止线停了没有 | 重试 | 最后一次更新 | 第 300 次 COMPLETE（UTC） | 判定用 |
|---|---|---|---|---|---|---|
| q38s0s1（s1 S0） | 2 / 2 | 没有 | 1 次：11-21 18:01:01Z 终点竞态，渲染服务先结束，链删掉还在 RUNNING 的训练任务（进程已正常退出），又交了一对训练 + 渲染；多出来的这一对 19:08Z 按用户"都删"删掉 | 300 | 11-21 18:00:08Z | q38s0s1_300 |
| q38v4s1（s1 V4） | 2 / 2 | 补题守卫（keep_fill）11-21 11:38:41Z 在第 50 次停下；16:14:43Z 按用户"关掉守卫续训"去掉这条守卫，从第 50 次续训 | 没有 | 300 | 11-22 03:58:20Z | q38v4s1_300 |
| q38s0s2（s2 S0） | 2 / 1 | 没有 | 1 次：11-21 14:23:31Z 训练任务 FAILED，链自动重交 | 300 | 11-21 19:55:08Z | q38s0s2_300 |
| q38v4s2（s2 V4） | 2 / 2 | 补题守卫 11-21 11:39:47Z 在第 50 次停下；16:18:01Z 同一决定去掉，从第 50 次续训 | 没有 | 300 | 11-22 11:17:00Z | q38v4s2_300 |

- **COMPLETE 都早于评测：** 每条运行第 300 次的 val、test、new59 评测都在 COMPLETE 之后才交。最早的一个是 val__q38s0s1_300，由 checkpoint_watch 在 COMPLETE 之后 62 秒交（11-21 18:01:10Z）；其余在两个评测步里交（s1 的 11-22 04:36:41Z 开始，s2 的 11:21:41Z 开始）。
- **停止线：** 主线有 kl_max 1.0、invalid 0.1（窗口 10）、success_drop 0.1、keep_fill（两条 V4 在第 50 次被它停下，续训时去掉；两条 S0 没被它停过）、host_mem_max_gb 1600、gen_slow，以及 eval.stop_invalid；没有熵线和图像 token 崩溃线。这些差别 PREREG 第 4.2 节"D − C0"一段已经写明。
- **判定用第 300 次：** 两条 V4 在第 50 次被停下以后续训到了 300，所以不适用"被停下的用最后一个检查点"。PREREG 第 4.2 节写的是"C0 一边：RL 是主线四条运行的第 300 次"。
- update_seen：q38s0s1 是 300，其余三条是 298 或 299（last_update 都是 300）。原因和 cell C 一样；另外三条运行的链在终点前后被手动停过（为了避开终点竞态），轮询更早停了。

cell D 跑完后照 cell C 那张表再填一遍，写进本文以后才读 D 的结果。

### 4.2 怎么算（8B 代码原文，移植没改）

- **逐题分类**：`failure_breakdown.classify`。8B 跑 H3 时用的是提交 ec9aeb61 里的版本，和 f5cd03b6 的副本（sha1sum 6f3a3dd7）这一段相同。规则按顺序判：
  1. 成功，或者任何一个回合 `met` → reached 或 drifted，不算失败。
  2. 否则，end_reason 不是 max_primitive_steps 也不是 max_turns → stopped。
  3. 否则，取各回合 `len(failed_gates)` 的最小值：
     - 最小值为 1 → budget_one_gate，差的条件取第一个这样的回合；
     - 否则 → budget_other。
  4. 起点位姿不算回合。
- **one 和 rel：**
  - one = 这道题是 budget_one_gate；
  - rel = one，并且差的那个条件属于"关系"组；
  - "关系"组：distance、equidistant、occupancy、relation、margin、lateral_within_45deg、target_hidden、target_in_view_direction；
  - 不认识的条件算"其他"。
- **① 差"关系"的比例**（分母是只差一个的失败）：
  - 每组把四个 RL 的 rel 和 one 分别加起来，算 `ztest(ra, na, rb, nb)`；
  - 用的是合并方差的两比例 z 检验，双侧（erfc），不做连续性校正；
  - 判"下降"：`p < 0.05 and d ≤ −10`，d = A − B，单位是百分点。
- **② 差"关系"的只差一个的失败，占全部题的比例**：
  - `paired()`：按列表位置配成 4 对，同种子、同一边，顺序是 [s1 S0, s1 V4, s2 S0, s2 V4]；每对逐题算 `int(a.rel) − int(b.rel)`；
  - 每对的差 = 100 × 和 / n；
  - 种子差 = 该种子两对的平均；合计 = 4 对的平均；
  - 符号检验：把 4 对里不一致的（题，对）合起来，做一次精确二项双侧检验，平局去掉；
  - 95% 置信区间：按题自助法 2,000 次，`random.Random('20261103/{set}/all')`，同一题集的各比较共用一套抽样；
  - 判"下降"：`p < 0.05 and all(种子差 < 0)`；移植另加了 `bool(种子差)` 这一条，见下面的 hunk 4。
- **比较**：(D, C, 主比较)、(D, C0, 次要比较)。8B 的 (E, C0, 探索性) 去掉，因为 27B 没有 E 格。
- **SFT 阶段**：每组一个模型、只配一对，算 ①、②，不校正、不判"下降"。8B 同一节里还有 SFT 阶段的成功率比较（合计和 6 个族，D − C、D − C0），那是 B1 第 2 节的次要比较，不属于 3.4，移植时去掉了；由谁出见 R16。
- **题集**：测试集 292 道为主，验证集 382 道参考，代码相同。

### 4.3 C0 一边怎么拼（27B 才有）

- **题目以本格的 screened tasks.jsonl 为准**：test sha256 b31bfe77…，val sha256 f5596bf3…，和 8B 的 screened 集逐字节相同。
- **C0 的 RL**（主线第 300 次）：
  - test = `q38rl_eval_test_v2` 里属于 screened 的 233 道，加 `q38rl_eval_test_v2_screened_new59` 的 59 道；
  - val = `q38rl_eval_val_v2` 的 413 道里属于 screened 的 382 道。
- **C0 的 SFT**（sft_q38c0_e5）：
  - test = `q38-c0-sft/sets/q38c0_eval_test_v2` 的 233 道，加 new59 的 59 道；
  - val = `q38c0_eval_val_v2` 的 382 道；
  - 只有这一份评测，ls 核过。
- **拼的时候核四件事：**
  1. 来源题集 tasks.jsonl 里每道用到的题，整条记录（全部 17 个字段）必须和本格 screened 的那道逐字段相同。在真实的 tasks.jsonl 上核过，27B 和 8B 的来源都没有任何字段不同，所以不用排除任何字段；
  2. 每个 id 只用一次；
  3. 总数等于 n；
  4. 同一个模型的几个来源文件夹（例如 test 的 233 道和 new59）都有 model.json 时，它们的 `model` 必须指向同一个检查点。主线 4 条的 _300 和 sft_q38c0_e5 都核过，是同一个。
- **error 检查**照 8B，查来源文件的每一行，包括没用到的那 59 行和 31 行，见 R1。

### 4.4 移植脚本和复现测试

| 文件（都在 `$K`） | sha256 | 说明 |
|---|---|---|
| `scripts/h3_q38.py` | ba31a95b287aa24ec9b0e45f75129ba2925601578939a1e5d5b40ee947e63254 | 移植后的脚本（复核后第 3 轮修改版；各轮旧版留作 `.pre_fix1`、`.pre_fix2`、`.pre_fix3`） |
| `scripts/run_fb.sh` | 2e0fa3d1429288b1a89a3b5dfa76ff1676808ee1fc12754ed111cad2353f927d | A 表的跑法（同上） |
| `scripts/scene_check.py` | 10dbf35903765d76568b92ecd12a21c3368a12bda024f32b707f032368f85458 | 第 3 轮新加：run_fb 不带 `--go` 时，`--wrong-target` 之前查 scene_id |
| `scripts/run1.sh` | 5dd6e63febd2737002b525da518102fe3cb1536aecc51c27db15cffa3dc53ef4 | 查核数，nice、ionice，各线程变量设成 1，然后跑一个命令；第 3 轮加了入口的 `env -i` |
| `scripts/PORT.diff` | 25642226db197c085fe9db342db3e541786fd990af5b946abfbb27e25ebfba64 | 8B 原脚本 → 移植，867 行，5 个 hunk |
| `scripts/PORT_NOTES.md` | 0de850f947ff0413fa9d5b3a04d4a4427a1285a53281fbb3312a17bc747077a9 | 每个 hunk 改了什么、为什么；"fix round 1/2/3" 三节，和"威胁模型和剩余问题" |
| `scripts/make_maps.py` | 54ccd2fc5a7e3b76232961c22cb3d0ef5607c14b2ecbf8900e6daea28c482b64 | 生成映射文件（没改） |
| `maps/map_27b_cellCD.json` | bc94f165048f6b826dd2ed137dee64e1c6e0b973a9e9d74ab884a6629bf6601a | 正式判定用的映射，`judgment: true`，没跑过 |
| `maps/map_8b.json` | 4a16621de1aeab53f98a5833b2a2c4fe6e0c0685839d588abf0c1e14f40e90e1 | 8B 复现用 |
| `maps/expect_8b.json` | 8759fa113f77d6d04f2bb29fbbfea1ac733db3e8965fcc66c91205e49d76a4f8 | 8B 复现用 |
| `maps/map_27b_test.json` | 5e3f7e478b580431cf43c36a6af1df6003eed7b90d455dd21fe9af5b0ceda348 | 非判定的测试 |
| `maps/map_27b_test_sft.json` | 049fcf92fe553b8e5b519a2e9aa8c7351ba488e7df61e63db7512d9a94c201a0 | 非判定的测试 |
| `maps/map_27b_guard.json` | 97da4f099c98bb3a3c6d03bf642c33d41ae0084d2c9889cbac8b89976caf86b3 | 闸门测试 |
| `tests/TESTS.md` | e7d10662d89d38e6f4c9476ca084c5f485aa3259fe5396da2f44800633a18932 | T1–T8 的命令和 rc，"fix round 3" 一节 |

各映射和修改前相比只多了一个 `rl_order` 键，其余逐字相同（tests/rerun_maps 核过）；第 2、3 轮都没改映射和 make_maps.py。

**调用方式和第 1 轮比的变化：** h3 不带 `--go` 时必须 `python -I -B`、在普通登录进程的挂载下跑，选项要写全（`--ma` 这种缩写 rc 2，`allow_abbrev=False`，带 `--go` 时也一样，不改任何计算）；OUT 必须不存在或是空文件夹。run_fb.sh 和 run1.sh 见第 3 节。第 3 节、第 4.5 节的两条 `--go` 命令照写可用（复核按新规则逐项核过 argv、环境和 cwd，没有跑）。

**5 个 hunk：**
1. **文件头和常量：** 代码从 archive 副本 import；删掉 qa_probe，以及只有 3.1–3.3 和第 5 节用的常量；RL、SFT、COMPARE 改从映射文件读。
2. **load 改成拼接读取：** 每道题的分类和 8B 逐字相同，8B 的 error 检查保留；新加第 4.3 节那四项核对，代替 8B 的 `len(out) == len(rows) == n`。
3. **新加两道闸：**
   - 闸在 mkdir 之前，也在打开任何结果文件之前；
   - 第一道：`judgment` 不是 JSON 的 false（true、没写、null、0、字符串都算），不带 `--go` 就拒绝；
   - 第二道：`judgment: false`，但有路径落在 `q38-mx-*-rl` 下，或者名字是判定点、挑点，也拒绝。所有文件夹、results.jsonl、model.json 和 tasks 路径都先解析成真实路径（realpath，也看字面规范化后的路径）再判；tasks 路径必须是一个名叫 tasks.jsonl、旁边没有结果的文件；
   - 后面只打开闸门核过的那些解析后的路径；
   - 映射里写了 `rl_order` 时，断言每组 4 个 RL 的顺序是 [s1 S0, s1 V4, s2 S0, s2 V4]（现有映射都写了）；
   - 8B :159-161 写死的 C0 断言改成 `--expect`；
   - ids 和任务族改从 canonical 的 tasks.jsonl 取，有断言保证和 8B 等价；
   - 删掉 3.1、3.2。
4. **down2 加 `bool(seed_diff_pp)`：** 照 8B 的 `m3_supp_rest.py:136`。每组 4 个 RL 时不改变任何判定；只有一组只有一个模型时才起作用。
5. **SFT 阶段：** 只留 3.4 的 ①、② 两行；某组没有 SFT 就跳过，记进 `sft_skipped`。和 8B 的 rest.md 比，表头、分隔行和两行 `| 3.4` 逐字节相同；这一节的标题和说明那一行改了措辞（8B 是"## SFT 阶段（测试集；…）"，移植是"## SFT 阶段的 3.4（测试集；…）"），8B 的 7 行成功率去掉了（见 R16）。输出改名 `rest_h3.{json,md}`，md 不再打到 stdout，只打印路径和 ok。

没改的：sign_test、fp、paired、rate、ztest、KIND_ZH，REPS、SEED、ALPHA 的默认值，以及 3.4 那一段除 down2 以外的部分。

**测试（全过，都没读 cell C；记录在 `tests/TESTS.md`）：** 下面是第一次的结果。复核后改了代码，T1–T5 在新代码上全部重跑（`tests/rerun_t1` … `rerun_t5`），结果和第一次相同：T1 386/386、md 68 行 0 处不同；T2 核对 3,504 和 4,584 次全过；T3 rc 2、rc 3，0 个结果文件被打开；T4 全相同；T5 的输出和第一次逐字节相同。另加了 T6。

- **T1 8B 复现：**
  - 用 `map_8b.json` 在 8B 的数据上跑。这些数据 8B 已经读过、报过。
  - `--expect` 写的是 8B 的 C0 计数（SFT 51/103；RL 41/76、53/105、53/105、58/108），通过。
  - 和 8B `analysis_20261103/rest.json` 比了 386 个叶子值（test、val 各 193 个，值和类型都比），386 个全相同。比的范围：3.4 的每个模型，D−C、D−C0、E−C0 三个比较，SFT 阶段的 4 个键。
  - md 的 3.4 段共 68 行，和 8B 的 rest.md 逐字节相同。
- **T2 27B 拼接读取：**
  - 只用主线的非判定点 050、150、250，题目以 cell C 的 screened tasks.jsonl 为准。
  - test：12 个模型，每个正好 292 道，题目记录核对 3,504 次全过。
  - val：每个 382 道，核对 4,584 次全过。
  - 没有 SFT 的组跳过 SFT 配对，md 里注明了。
- **T3 闸门：**
  - h3_q38.py 带审计钩子跑正式映射、不带 `--go`：rc 2，打开了 0 个 results.jsonl，也没碰 q38/runs 下的任何文件，没有建目录。
  - `judgment: false` 但名字是判定点：第二道闸拒绝。
  - run_fb.sh 跑 cell C、不带 `--go`：rc 3，没起 python，没查核数。
- **T4 函数单测：** 移植的 sign_test、ztest、paired、rate、fp、自助法抽样，和 8B 原函数在合成输入上逐个比，结果全相同。
- **T5 run_fb.sh 正向冒烟：** 主线 new59、q38s0s1_050，rc 0，用时 11 s。第一次跑出了一个相对路径 OUT 的 bug，已修（sha 85d61142 → d467827a），T3c、T3d、T5 都重跑过了。
- **T6 闸门加固（复核后加的）：**
  - 复核用 8B 数据搭了一个假的 cell C 目录，证明旧代码的第二道闸可以被绕过：`//`、`/./`、`/../`、相对路径、符号链接，或者把 tasks 路径指到一个 results.jsonl，`judgment: false` 的映射不带 `--go` 也能读到这个假目录里的结果。正式映射是 `judgment: true`，一直被第一道闸拒绝，不受影响。
  - 新代码上，43 个闸门用例全部符合预期：旧代码放行的 23 个现在都拒绝，4 个对照照常放行；10 次 main() 不带 `--go`，审计钩子在 q38/runs 和假目录下记到 0 次打开。
  - run_fb.sh：模型文件夹是指向假 cell C 的符号链接、`..`、`//` 等 8 种写法，都 rc 3，没起 python，没建目录。
  - 拼接读取：10 种故意弄坏的输入全部中止，包括复核发现旧代码放过的"改了 5 个键以外的字段"。
- **T7 第 2 轮复核加的用例：** probe_h3 113 个、probe_fb 104 个，以及真实路径上的闸门核对，都符合预期，打开判定文件 0 次。
- **第 3 轮（11-25，换登录节点之后，uid 10174）：** 复核又找到 9 类绕过（经 /proc/self/cwd 或 /proc/<pid>/root 的链接、unshare 加 bind mount、scene_id、链接目标末尾的换行、输出是链接、检查和打开之间换文件、BASH_ENV 和 shell 函数、`--g` 缩写），都不带 `--go`。修法见第 3 节和 PORT_NOTES "fix round 3"。新代码上：
  - T1–T7 全部重跑（`tests/rerun3_t1` … `rerun3_t7`）：T1 仍是 386/386，md 0 处不同；T2–T7 的 rc 和计数和第 2 轮相同，T5 的输出逐字节相同。
  - T8（`tests/t8_fix3/node2`）：9 类全部被拒（h3 rc 2、run_fb rc 3），被拒的用例都没有打开判定文件；对照照常通过。T8 的测试脚本本身有 3 个错，已另写用例重测通过，被测脚本没改。
  - AST 比对：3.4 的计算函数和 pre_fix3、和 8B 原脚本都相同；`--go` 那段代码逐字节没变。
  - 正式映射检查（只用 stat、readlink）：cell C 的 25 个来源都齐，cell D 的 10 个还不存在（预期）。
  - 独立复核（`review_fix3/`）：重放了第 2 轮的全部绕过，又试了约 70 个新用例，没有能读到 27B 判定数据的。自己重跑 T1 也是 386/386。
- **剩下的问题（诚实操作者的威胁模型下接受，写在 PORT_NOTES "威胁模型和剩余问题"）：**
  - run_fb 交给 python 之后，failure_breakdown 是没改的 8B 代码，它自己打开文件时没法逐个核 fstat；recheck 之后再换文件，仍能读到。
  - 外层 shell 在 `env -i` 重起之前会先执行 BASH_ENV，所以调用时不能设。
  - 复核另找到两处不在清单里的：① run_fb 没有 h3 那条 model.json 规则，一个从 cell C 拷出去、改了名字的检查点文件夹能过 run_fb 的闸（h3 会拒绝）；这要先手工拷一份判定检查点，第 3 节的命令不会碰到。② h3 写输出时，检查和打开之间把 OUT 换成链接，能在别处新建一个文件（不能覆盖，也不读任何东西）。另有 3 个小问题（conda 里的 .pth 在 `-I` 下仍会执行，两个挂载检查对坏行的处理不一致，scene_id 正则接受末尾换行），都不影响读数。这几处没再改脚本：改了要按第 6 节重跑全部测试，而 cell C 一读，闸门对 cell C 的用处就结束了。cell D 读之前要不要补，到时报用户。

### 4.5 正式跑的命令（cell D 评完、用户批准读结果以后）

1. 照 4.1 给 cell D 填"稳不稳"表，把 `map_27b_cellCD.json` 里 D 格的名字和目录对上实际的（用 make_maps.py 重新生成，不手改）。现在写的是照 C 格格式推的，没核过；对不上时加载会因文件不存在、顺序不符（`rl_order`）或题目记录不符而中止，不会读错数据。
2. 改了映射就记下新的 sha256，然后跑：

```
U=/mnt/umm/users/yinbaiqiao; K=$U/q38/tmp_tests/k_d2; PY=$U/.conda/envs/vagen-lite/bin/python
cd $K && bash scripts/run1.sh $PY -B scripts/h3_q38.py --map maps/map_27b_cellCD.json --go $K/h3_CD > $K/h3_CD.log 2>&1
```

3. 输出是 `$K/h3_CD/rest_h3.{json,md}`。照 R5 报 D−C、D−C0 的 ①、② 四个标记，以及 SFT 阶段的两行。照 B1 第 5 节写明一共比较了几次：每个题集 8 次，test、val 共 16 次。
   - 主 3.4 有 4 个标记：D−C（主）的 ① 和 ②，D−C0（次）的 ① 和 ②；
   - SFT 阶段有 4 个检验：D−C、D−C0 各一个 ①、一个 ②，不判"下降"；
   - test 上的 4 个标记照 R5 读（D−C 主、D−C0 次）；val 是参考，SFT 阶段是次要，都不校正。

## 5. 读结果前要用户定的事（推荐项放第一；用户 11-25 定：全部按推荐）

8B 的代码定了怎么算，但有几处 PREREG 的文字没写清，或者 8B 没碰到这种情况。照冻结规则，下面这些都是"开跑前必须定"的事，也是评测规则，所以都先问用户。

**用户 11-25 约 06:1xZ 回"按推荐"：下表 17 条都取"推荐"一列，"备选"一列不用。** 已写进 PREREG_Q38 v1.2 第 3.8 节 D2、D8 两行和第 4.2 节 H3 的"执行"一段。

| # | 事 | 8B 代码怎么做 | PREREG / 附录的文字 | 推荐 | 备选 |
|---|---|---|---|---|---|
| R1 | 有 error 行；拼接时核对不过 | `assert not any("error" in r ...)`：有就中止，不算。8B 不拼接，没有第二种情况 | 没写 | 照 8B：中止，报用户再定。C0 拼接时 assert 中止（重复 id、题目记录不符、总数不对、同一模型各来源的 model.json 指向不同检查点）也一样：中止，报用户，不手工绕过 | 记成"失败、不是只差一个"，接着算 |
| R2 | ② 的"下降" | `p < 0.05 and all(x < 0 for x in seed_diff_pp)`：两个种子的差都 < 0 | "两个种子的差同号" | 照代码：同号而且都是负的。都为正是"上升"，不算下降 | 照字面"同号"（会把两个都是正的也判成"下降"，和"下降"的意思相反） |
| R3 | ② 的检验单位 | 4 对（同种子、同一边）里不一致的（题，对）合起来，做一次精确二项双侧检验；平局去掉 | "逐题符号检验" | 照代码 | 先在每道题上把 4 对合成一个数，再做符号检验（8B 没这么做） |
| R4 | ① 的检验 | 四个 RL 的失败加在一起，合并方差的两比例 z 检验，双侧，不做连续性校正，每局失败当独立的；"下降"是 A − B ≤ −10 个点（含 −10）且 p < 0.05 | "两比例的 z 检验（四个合起来）"，"下降 ≥ 10 个点" | 照代码。分母为 0（没有只差一个的失败）时中止，报用户 | — |
| R5 | H3 的总判定 | 没写成代码。每个比较各给 ①、② 一个"下降"标记，"H3 不成立"是 8B 手写的 | 4.2 节只写了 ①、② 各自的"下降" | 照 8B：D − C、D − C0 的 ①、② 四个标记分开报。四个都"没下降"就写"H3 不成立"，同 8B 的读法；有任何一个"下降"，报用户定怎么说 | 现在就定：D − C（主）的 ① "下降" 就算 H3 成立，② 和 D − C0 是旁证 |
| R6 | 自助法的种子 | `random.Random(f'{SEED}/{set}/all')`，SEED = 20261103，2,000 次，同一题集的各比较共用一套抽样 | 4.2 节写了 2,000 次，没写种子 | 照 8B 原值 20261103。这是和数据无关的常数，改了没好处 | 换成 20261124 |
| R7 | SFT 阶段的 3.4 | 单列一节，"次要，每组一个模型，不校正"，不判"下降" | 4.2 节："SFT 阶段的比较是次要的" | 照 8B 单报，不判定 | 不报 |
| R8 | A 表里"停在哪"的标签 | `STOP_ZH` 的键是 `collision`，env 记的是 `consecutive_collision`，所以这类停止落进"其他"；`explicit_done_without_success` 也落进"其他"。"stopped"这一类本身不受影响，H3 也不受影响 | 没写 | 照 8B 不改，报告里注一句这两类记在"其他"里 | 在 A 的输出上另加一张改正了标签的表 |
| R9 | A 带不带换目标重放 | 8B 矩阵 HEAD 的计划带（`wrong_target: true`）；11-03 跑 C/D/E 的链时还没有这个开关 | 没写 | 带，只报、不进判定；每个模型报 unchecked 数 | 不带，和 8B 当时的 C/D/E 一样 |
| R10 | 探索性格子（D_noP0、Cr） | `m3_supp_rest.py`（没进 git）：照 3.4 单算，`down2` 多一个 `bool(seed_diff_pp)` 的保护 | 4.2 节："各自单报，不进 H1–H3 的判定" | 照 8B supp 的写法 | — |
| R11 | 什么时候读 cell C | — | D2："读结果前做" | 本文和 R1–R17 定了、PREREG v1.2 提交以后，读 cell C 的结果，跑 A 表；H3 等 cell D | 等 cell D 出来一起读 |
| R12 | A 表跑哪些模型 | f5cd03b6 的 `evaluated_models()`：sft_groups 计划出每个 SFT epoch，start_model 计划出 "start"，再加每 50 次一个、导出了 HF 权重的 RL 检查点，以及计划 `eval.models` 里的名字。8B 实际出的表早于过滤，是共享文件夹里全部 68 个模型 | 没写 | 用 f5cd03b6 这段代码在 cell C 计划上的结果：25 个（第 3 节的命令）。8B 当时实际出的那种"整个文件夹"做法，27B 不照搬 | 只跑判定用的 5 个（第 300 次 × 4，加 SFT 起点） |
| R13 | C、D 两格的 SFT 起点用哪份评测 | 8B 的 SFT 和 RL 在同一个评测文件夹里，同一检查点只有一份评测 | 没写 | 用 RL 格自己评的那份（`q38mxr{c,d}_eval_*` 里的 `sft_q38mx{c,d}_e*`）：和 RL 检查点在同一个文件夹，评测代码同一版，也是同一批评的 | 用 SFT 链挑 epoch 时那份（`q38-mx-{c,d}-sft/sets/q38mx{c,d}_eval_*_v2_screened`） |
| R14 | error 检查查到哪 | 查文件的每一行（8B 的文件正好 n 行，全都用到） | 没写 | 照 8B 代码，查来源文件的每一行，包括拼接时没用到的那 59 行和 31 行；有就中止，报用户 | 只查用到的行 |
| R15 | D8 (b) 补查算不算 | — | PREREG 第 3.8 节 D8、DECISIONS_20261124：cell C 用 (b)，"评测前人工查 COMPLETE" | 算：评测后比时间戳，24 个 COMPLETE 全部早于评测开始，结论和评测前查相同（第 4.1 节）。在 PREREG 的 D8 一行注明"cell C 是评测后补查" | 不算，重评 cell C（要占卡，不推荐） |
| R16 | SFT 阶段的成功率比较由谁出 | 8B 在同一个脚本（m3_prereg_rest.py 的 SFT 阶段一节）里出：合计和 6 个族，D − C、D − C0，逐题符号检验 | 4.2 节："SFT 阶段的比较是次要的"；B1 第 2 节 | 归 H1 那边：和 3.1–3.3 一起移植、一起报，D2 只出 3.4。这样 D2 的闸门和测试不用扩大 | 在 h3_q38.py 里照 8B 加回这几行（T1 能对 8B 逐值比） |
| R17 | "能测出的最小差"用哪个数 | 8B 是开跑前按题数和 RL 数估的（B1 第 3.4 节：①约 9 个百分点，②约 3.5–4.5） | B1 第 3.4 节表，以及"矩阵第三版之后加的"第 6 条："按新的题数重算" | 沿用 B1 第 3.4 节的数：27B 的题数（292 / 382）和每组 RL 数（4 个）都和 8B 一样，不按 27B 的结果重算 | 读结果后按 27B 实际的失败数重算，和 8B 的数并列，标"事后" |

另外两件照 B1 第 5 节写进报告，不用定：
- 次要指标不校正，但写明一共比较了几次（第 4.5 节第 3 步列出）。
- 先按种子、按侧分开报，再合起来。h3_q38.py 的输出照 8B：② 在 json 里有每对的差，md 里有两个种子差和合计；① 照 B1 只算四个合起来。

C0 的 SFT（sft_q38c0_e5）只有一份评测（第 4.3 节），没有 R13 的问题。

**移植 agent 看到的一个数，如实写明：** 查目录时，它在一个 PLANS.json 里看到了 SFT 挑 epoch 那一项的 `"val": "173/382"`。这是 SFT 链挑点用的 val，D9 允许读，PREREG 开头也已经写明读过。它不是 cell C 的 RL 结果，没有用在任何计算里。

**复核 agent 偏离登录节点规则的地方，如实写明：** 几次很小的 python 读 JSON（plan.json、progress.json 的结构，rl_standards.json 的守卫默认值），以及对 archive 副本的几次 find（只看文件名），没有经过 run1.sh、没加 nice；两次 run_fb.sh 拒绝测试直接用 bash 跑（在起 python 之前就退出了）。每次查核数都在 0.4–1.2 / 14 之间。没有读任何判定结果。

## 6. 防误读

- **两个脚本都有闸：** 不带 `--go` 就拒绝判定数据，拒绝在打开任何结果文件之前（T3 已证）。`--go` 只在用户批准读该格结果以后才加。
- **结果只写进文件：** 两个脚本的 stdout 都只有路径和 rc，结果只进 `$K/cellC/`、`$K/h3_CD/` 和日志。报给用户之前，不在终端打印、不 cat、不 grep 结果。
- **改动要再测：** 改 h3_q38.py、run_fb.sh、run1.sh、scene_check.py 或映射中的任何一个，都要重跑 T1–T8，并在本文更新 sha256。
- **定了的事落到 PREREG：** R1–R17 定下来后，写进 PREREG_Q38 v1.2 第 3.8 节"失败分解（D2）"一行，和本文一起提交，然后才读 cell C。（11-25 已写进第 3.8 节 D2、D8 两行和第 4.2 节 H3 的"执行"一段。）
