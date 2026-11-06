# 退步起点评测包：交主线集成（2026-11-06）

奖励线分支 reward-design。结果报告 `docs/reward/reports/PERTURB_EVAL_20261106.md`，预登记 `docs/reward/PREREG_PERTURB.md`。

**一句话：**
- 四个题集，加 5 个脚本（其中 perturb_followup 可选）和 2 个测试文件，全是新文件。
- 加到 qwen3-main 22a0fc62 或 matrix-v3 fba7057a 上都能用。11-06 在两边的临时克隆里都试过：
  - 21 个测试全过；
  - perturbed_starts、turn_metrics、fail_end_steps、compare_perturbed、turn_bins 的 `--help` 都能起；
  - 实际重放了 rw_g_250 在退 2 步验证集上的 374 局，逐局和 11-06 那次重放相同。
- 每个模型评两档约 1.6 卡时。
- 建议矩阵把**每回合出错率**当 G 组的机制指标（第 4 节）。G 的预登记已定稿，这个指标在 G 那里只能算探索性。

用语：
- **退 k 步：** 起点换成离原起点正好 k 个动作、表里剩余步数多 k 的位姿（k = 2 或 4），目标和成功规则不变。
- **d\*：** 原题从原起点出发的最少动作数。
- **余量 = 12 − (d\* + k)：** 一局最多能浪费几个动作，12 是每题的动作上限。
- **每回合出错率：** 一个回合走完，真实剩余步数比回合开始时多，这样的回合占查得到剩余步数的回合的比例。结果报告里叫"逐回合失误率"。
- **找回率：** 出错以后两个回合以内，剩余步数回到出错前或更少的比例。

## 1. 包里有什么

| 东西 | 位置 | 说明 |
| --- | --- | --- |
| 四个题集 | `reward_runs/eval_sets/pert_{val,test}_k{2,4}/tasks.jsonl`（只读） | 由 v2 的 705 道评测题做：退 2 步 val 374 + test 275 = 649 道，退 4 步 339 + 255 = 594 道。sha256 记在预登记里。9 个模型的结果在同一目录下 |
| 造题 | `reward/perturbed_starts.py` | 用距离表的状态图从原起点走 k 步，挑表里剩余步数正好 d\* + k 的位姿，按题号和 k 取种子抽一个；每个新起点都用环境重放核过 |
| 逐回合指标 | `reward/turn_metrics.py` | 每局在评测环境里不渲染地重放，查每回合之后的真实剩余步数：出错率、找回率、停滞率、有 / 无出错的成功率、碰撞率、转身占比、成功局的动作数 ÷ d\*（写出的和执行到成功的各一个） |
| 失败终点 | `reward/fail_end_steps.py` | 重放失败局，查终点的剩余步数，带"精确"标记。qwen3-main 22a0fc62 和 matrix-v3 fba7057a 上已有，和本线逐字节相同 |
| 分档 | `reward/turn_bins.py` | 把 turn_metrics 的逐局记录按 d\* 或余量分档再汇总，不用再重放 |
| 比较 | `reward/compare_perturbed.py` | 本线预登记的比较。模型名和配对是常量；评别的模型，照它的写法另起一份常量 |
| 加算（可选） | `reward/perturb_followup.py` | 这次评测的加算：失败局回没回到原水平、按题自助法区间等。路径和 9 个模型写死在文件里，没有命令行参数；评别的模型要照它另写 |
| 测试 | `reward/tests/test_turn_bins.py`、`test_compare_perturbed.py` | 共 21 个 |
| 距离表 | `reward_runs/tables/eval705`（只读） | 705 道的深度 12 表。筛过的测试集另加 `tables/screened59`，视图是 `tables/eval_test_v2_screened` |

- **依赖：** `data_gen/interiorgs_v10/` 的 `distance_table.py`、`difficulty.py`、`distance_table_near.py`、`scene.py` 和评测环境模板 `configs/rl_env_v10_S0.yaml`，这几个文件在 qwen3-main 22a0fc62、matrix-v3 fba7057a、reward-design 上逐字节相同。
- **环境不完全相同：** `vagen/envs/active_spatial/` 里，三个分支间 env.py、env_config.py、prompt.py 有改动；matrix-v3 另有两个新模块：pointing.py（env.py 开头导入，出提示时会调 target_refs，只有目标段格式才用它的结果）和 occlusion_depth.py（环境不导入，只有 data_gen/transition_qa 用）。在 action_names 格式下这些都不起作用：本线的 optimal_progress_near 和 observe 格式、矩阵 G 的指认奖励和目标段格式都默认不开；E 的 has_think 没有开关，每回合都算，只多记一个字段。
  - 重放会把结果 model.json 里的环境开关（比如格式示例）套进环境。G 的结果用 targets_norm1000，只有 matrix-v3 的环境认，所以 **G 的结果要在 matrix-v3 上重放**，这 7 个文件也要加到 matrix-v3 上。
  - 两边实际重放过的只有 rw_g_250 退 2 步验证集（374 局），逐局和 11-06 的重放相同。本线其余结果和 D 组的两个（sft_m3d_e9、m3ds1d_300）的 model.json 同样只有 no_think + action_names，按上面的环境差别推断两边都能重放、结果相同，没有实测。

## 2. 怎么跑

1. **评测：** 经训练链只评测。
   - 计划照 `reward_runs/runs/rw_pert_eval/plan.json` 写：`eval_only: true`，`eval_sets` 写四个题集目录，`eval.models` 写模型路径。
   - 评法要和这个模型原来的评测逐项相同。这次评的 9 个模型（含矩阵 D 组的两个）都是贪心、no_think、`--format-example action_names`、12 回合、带历史。
   - 每个模型 4 个单卡任务（两档 × val / test）。
2. **重放：** 本线用了两个单卡 sco 任务，一个算 turn_metrics、一个算 fail_end_steps，各自把 36 个结果一起算完，约 3 分钟。sco 任务用主线自己的提交工具；本线的 `reward/sco_job.py` 不在包里。
   - `python -m reward.turn_metrics --result 模型@k2=结果目录 [...] --tables reward_runs/tables/eval705 [--tables reward_runs/tables/screened59] --out 文件.json --workers 12`
   - `python -m reward.fail_end_steps --result 结果目录 [...] --tables 表目录 --out 文件.json --workers 12`
   - turn_metrics 的 `--tables` 可以给几个，按顺序找。fail_end_steps 只收一个目录：筛过的测试集给视图 `tables/eval_test_v2_screened`（见 `FAIL_END_STEPS_HANDOFF.md`）。
3. **分档和比较：**
   - `python -m reward.turn_bins --metrics 文件.json --by margin --bands 0,1-2,3-4,5- --label '*@k2' --label '*@k4' --out 输出`：按余量分档（`--limit` 默认 12）。表的行只来自 `--label 模式`（每个匹配的"模型@k2""模型@k4"一行）或 `--group 名=模式[,模式]`，两样都不给就只有表头。
   - `--by dstar` 按记录里的 d_star 分档：原评测题是 d\*，退步题记的是 d\* + k。要按原题的 d\* 分档，每个 k 单独给标签、档位加 k（例如退 4 步用 `5-7,8-9,10-12`，对应原 d\* 1–3、4–5、6–8）。
   - `python -m reward.compare_perturbed --set val_k2=目录 [...] --original 模型=原 val 目录,原 test 目录 --turns ... --fail-end ... --out ...`
- **代价：**
  - 评测每个模型两档约 1.6 卡时，含装模型；这次 9 个模型共 36 个任务，约 14.7 卡时。
  - 重放 36 个结果约 0.1 卡时。
  - 分档、比较在登录节点，1 个核，几秒。

## 3. 怎么读

**余量分档：**
- **余量 0：** 退 4 步有 26 道，9 个模型全是 0%，不进比较。
- **余量 1–2：** 退 4 步有 190 道，9 个模型只有 0.5%–3.2%，几乎做不成，模型之间的差别读不出来。主比较和次比较用余量 ≥ 1 的题。
- **余量和原题难度分不开：** 同一个 k 里，余量和原 d\* 一一对应（退 4 步：余量 = 8 − d\*），分不开是余量的作用还是原题难度的作用。跨 k 比要按余量对齐，那是不同的题。
- **216 道上界题：** 退 4 步、原 d\* 6–8 的 216 道，也就是余量 0 和 1–2 两档的全部题，d\* + 4 只是上界。表只建到原起点外 12 步，可能有一条走出表外的更短的路，真实余量可能更大。
  - 两边比同一批题，比较不受影响；但读这两档的绝对成功率要带上这一句。
- **题少的档：** 退 2 步余量 1–2 只有 33 道，一道题就是 3 个点。

**失败终点：** 在退步题上，表内的值多数是上界。
- 精确的只有退 2 步 28%–34%、退 4 步 21%–27%（精确的 ÷ 终点在表内的失败局；原起点按同样的口径也只有 22%–56%）。其余的离原起点太远，真实的可能更近。
- 能确定的只有两头：
  - 还差 1–2 步：退 2 步占失败局的 9%–15%，退 4 步 8%–14%；
  - 还差 3 步以上：退 2 步 43%–53%，退 4 步 34%–39%。
- 表内均值（退 2 步 5.5–6.1 步、退 4 步 5.9–6.5 步）不宜单独读：上界让它偏大，表外的终点不算又让它偏小。
- 查不到剩余步数的回合（退 2 步 1.1%、退 4 步 4.5%）不进出错率和找回率。

## 4. 建议：把每回合出错率当 G 组的机制指标

- **有没有出过错几乎决定一局成败：** 原评测题上，没出错的局成功率 86.2%–98.2%，出过错的只有 6.5%–19.3%（`MIX_SCREEN_20261105.md` 第 5.3 节的 56 个检查点、约 3.9 万局里，除走样的 rw_tc1 三个以外的 53 个）。每回合出错率量的就是这种错出得多频繁。
- **被推远以后没有明显变高：** 9 个模型在原起点上 25.1%–33.4%，退 2 步 24.2%–31.3%，退 4 步 24.3%–29.7%；7 个模型两档都降，rw_o 退 2 步高 0.5 个点，m3ds1d 退 2 / 4 步高 0.1 / 1.1 个点。但这不能当作"这个指标不受起点远近影响"的证据：
  - 题集不同：原起点是各模型全部原评测题（705 道，D 组 674 道），退步是余量 ≥ 1 的题；
  - 按原题的 d\* 分档，长题的出错率本来就低：全部 RL 检查点 d\* 1–3 / 4–5 / 6–8 是 34.3% / 29.5% / 25.7%（`MIX_SCREEN_20261105.md` 第 5.4 节），退步等于把题变长了；
  - 走出表外的回合没算出错。
  - 所以比较两个模型要同题配对。
- **RL 降它，奖励的设计没把它分出高低：**
  - 起点 SFT 是 33.0%–33.4%（C0 e8、D e9），本线挑出的 RL 检查点和 m3ds1d 第 300 次是 25.1%–29.3%；
  - 同在第 150 次，稀疏、密集、逐回合三类奖励的 8 个运行都在 26.5%–29.4%。
- **对 G 的意义：** G"先指认再动作"的假设是先看准目标、再走，直接对应的就是每一步走错的比例。
  - 建议 G 对 D 同题配对，报每回合出错率（两个种子，按题自助法给区间），再报找回率；成功率仍按 G 的预登记。
  - G 的预登记第四版 11-06 已定稿（"另加的分析一律标'探索性'"），这一项只能作探索性的机制指标；要进判据，得用户另批并改预登记。
  - D 的参照：D e9 33.0%，m3ds1d 第 300 次 25.1%（原评测题，筛过的 674 道）。
  - 按题自助法的区间要另写代码：compare_perturbed 里的模型和配对是常量；perturb_followup 写死了这次的 9 个模型和路径，只给找回率的区间。
- **算法：** 评测结果带轨迹就能算，不用 GPU。

## 5. 集成

- **只加文件，不改任何现有文件：**

  ```bash
  git checkout fd1eeb09 -- reward/perturbed_starts.py reward/turn_metrics.py reward/turn_bins.py reward/compare_perturbed.py reward/perturb_followup.py reward/tests/test_turn_bins.py reward/tests/test_compare_perturbed.py
  ```

  - 所有 worktree 共用 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/.git`，这个提交直接可见。
  - 提交 fd1eeb09 里 turn_metrics 的效率三项不再先舍到 3 位，测试也加了一个；其余 5 个文件和 2f52b34f 相同。
- 题集、表和 9 个模型的结果都在 `reward_runs/` 下，只读引用即可。
