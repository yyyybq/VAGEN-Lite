# 改动清单（分支 `transition-qa`，相对 main）

本文件列出这个分支改了哪些**现有**文件、为什么，以及训练链（`train_chain`）还需要加什么。新增的目录都只加不改。
分支和 main 保持同步；合并到 main 由用户在冻结期触发（2026-10-19）。

## 1. 改了的现有文件

全部是加选项或加指标，默认行为不变。

| 文件 | 改了什么 | 为什么 |
| --- | --- | --- |
| `data_gen/interiorgs_v10/training_prep.py`（23 行加、1 行改） | `rl-config` 加 `--output-format`（2026-10-18）和 `--max-response-length`（2026-10-19），见 1.1、1.2 | 矩阵 E 组：RL 换 `free_think` 提示（用户 2026-10-18）；回答上限只对 E 升到 6,144（用户 2026-10-19） |
| `vagen/envs/active_spatial/env.py`（6 行加） | 解析结果和每回合的 `turn_metrics` 加 `has_think`：回答里有非空的 `<think>` 块（闭合或被截断都算） | E 组：记录含思考文字的回答比例（用户 2026-10-19） |
| `vagen/agent_loop/gym_agent_loop.py`（10 行加） | 一条序列模式的 agent loop 数每个 episode 里 `has_think` 的回合，写进 `reward_extra_info.think_rate`（占回合数的比例） | 同上；第十一轮的 `invalid_action` 就是这样记的 |
| `vagen/agent_loop/gym_agent_loop_no_concat.py`（1 行加） | 每回合样本的指标加 `think_rate`（0 或 1） | 同上（另一种历史模式） |
| `vagen/ray_trainer.py`（1 行加） | 记 `transition/think_rate/{mean,std,sum,count}` | 同上，和 `transition/invalid_action_rate` 并列 |

### 1.1 `rl-config --output-format {action_only, think_action}`

默认 `action_only`，行为和原来完全一样：

- `think_action` 时，写出的 `train_env.yaml` / `val_env.yaml` 用 `prompt_format: free_think`。
- 系统提示一致性检查仍然对模板做，所以 RL 提示和 SFT 提示只差输出格式那几行。测试把那几行换回去，得到的提示逐字相同。
- 报告 `rl_config_report.json` 加 `output_format`、`rl_prompt_format` 两项。

### 1.2 `rl-config --max-response-length N`

- 不给时，一条序列模式（`--history-mode concat`）的回答上限仍是 `CONCAT_RESPONSE_LENGTH = 4096`。
- 给了就写进 `data.max_response_length`；在其他历史模式下给这个选项会被拒绝（那里的上限是每回合的）。
- 报告加 `max_response_length`。
- 不是标准（`rl_standards.json`）里的项，所以 `--standard` 照常用。
- vLLM 开了分块预填充（`enable_chunked_prefill: true`），`max_num_batched_tokens: 8192` 小于提示加回答的 10,240 也能跑。
  - 没有在 GPU 上试过：一条序列最长从 8,192 个词元（提示 4,096 + 回答 4,096）变成 10,240 个，显存要多一些。
  - 建议 E 正式跑之前先用 `--trial` 跑几次更新。
- 每回合的回答上限（环境的 `response_length_per_turn: 384`、评测的 `max_tokens=384`）没有改。思考文字加动作超过 384 个词元时会被截断，记成 `truncated_before_action`。

### 1.3 含思考文字的回答比例

- 判定：回答里 `<think>` 之后、`</think>`（或回答结尾）之前有非空白文字。
- 所有 RL 都会记。只输出动作的组应接近 0，E 组看它是否涨起来。
- 不改奖励，也不改格式判定。在 `free_think` 提示下只回 `<action>…</action>` 仍然算格式正确。

### 1.4 测试

- `data_gen/interiorgs_v10/tests/test_output_format.py`（新文件，7 个）：
  - 两个选项各自只改该改的地方；
  - 解析器的 `has_think`；
  - agent loop 的计数函数。
- 和第十一轮的 `tests/active_spatial/test_round11_package.py` 一起 11 个，全部通过。

## 2. 输出格式开关的三处

| 地方 | 开关 | 状态 |
| --- | --- | --- |
| SFT 动作样本的导出 | `configs/admission_v2.json` 的 `sft.export.prompt_format`（现在 `no_think`） | 已有，没改。`release.build_sft_record` 只会写只有动作的对话，没有思考文字的来源；按用户决定所有组的 SFT 都只输出动作 |
| RL 的提示词 | 环境的 `prompt_format`（`free_think` / `no_think`）：系统提示（`prompt.py` 的 `FORMAT_CONFIGS`）和每回合的提示（`format_prompt_*`）一起变；生成配置用 `rl-config --output-format`（第 1 节） | 环境已有；rl-config 本次加了选项 |
| 评测 | `evaluation.py run --prompt-format free_think`；SCO 脚本的 `eval` 模式第 6 个参数就是它 | 已有 |
| 动作解析器 | `ActiveSpatialEnv._default_parse_func` → `utils.parse_free_think`：`<think>…</think><action>…</action>` 和 `<action>…</action>` 都读 | 已有，没改 |
| 解析失败率 | 环境每回合的 `turn_metrics`（`missing_action_tag`、`truncated_before_action`、`empty_action_body`、`unknown_action_name` 等）；RL 日志的 `transition/invalid_action_rate`；评测每回合的 `valid` | 已有 |

## 3. 验收：历史轨迹上，两种格式读出的动作是否相同

`python -m eval.diagnostics.format_check`（新文件）：把每个回答分别写成两种格式，用环境自己的解析器读：

- 只输出动作：去掉回答里的 `<think>…</think>`；
- 先想再输出：没有思考块的回答前面加一个。

比较两种写法读出的动作和评测记录下来的动作。

- **覆盖范围：** 47 个结果目录，86,067 个回合。
  - 旧评测集 278 道题 × 10 个模型，其中第一版 R1 模型的 2,296 个回答带思考文字；
  - 评测集 279 道题 × 37 个模型，其中 R1 模型的 2,471 个回答带思考文字；
  - 两份合计 4,767 个带思考文字的回答。
- **结果：** 86,051 个回合（99.98%）两种写法读出的动作序列相同，也和评测记录的相同。
- **16 个不一致：** 全部来自原始 7B 在旧评测集上的回答。它只写了 `move_left|move_forward|…`，没有任何标签：
  - 只输出动作的写法按规则读不出，评测时也确实没执行；
  - 套上思考块后，解析器的兜底规则（思考块之后的裸动作也算）把它读了出来。
  这是现有解析器的规则：先想再输出的格式比只输出动作宽松一点。没有改。
- **读不出动作的回答：** 共 132 个（0.15%），最多的是 R11-PD 第 100 次的 64 个（3.1%）。

结果：`exps/diagnostics_20261018/format_check.{md,json}`。

## 4. E 组的事（2026-10-19 用户已定前两件）

1. **模型可以完全不写思考文字**：在 `free_think` 提示下只回 `<action>…</action>` 也算格式正确，所以 E 的解析失败率可能一直很低，模型却没开始思考。
   - **已定**：记录含思考文字的回答比例（第 1.3 节，`transition/think_rate`）。
   - 不扣分，不改奖励。
2. **回答长度**：一条序列模式下整个 episode 的回答上限是 4,096 个词元。12 个回合每回合约 81 个图像词元、150 个文字词元加动作，已经用掉约 3,100；每回合再加 100–200 个词元的思考文字就会超出。
   - **已定**：只对 E 升到 6,144（第 1.2 节）。
3. **解析器的宽松度不对称（第 3 节）**：建议保持。只有原始模型会写这种回答，改了会改变现有训练对格式的判定。

## 5. 训练链需要加的（矩阵配置 `configs/matrix_v2/` 用；这里不改 `train_chain.py`）

每组 YAML 的 `train_chain_plan` 只用了训练链现在读得了的字段，照第十二轮的 plan 写（`rl.standard`、`rl.seed`、`rl.startup`，
每组 S0 和 D 一对）。

- RL 标准名是变量 `{rl_package}`（`configs/matrix_v2/vars.yaml`），由 `python -m data_gen.sft_mixture.matrix_plan` 填好再交给训练链。
- 有测试核对：`data_gen/sft_mixture/tests/test_matrix_configs.py` 逐个字段去 `train_chain.py` 的源码里找。
- C0 不训练，没有 plan。

读不了的放在 `needs_chain_support`，汇总如下。

| 字段 | 哪几组 | 训练链要做的 | 不改训练链时的办法（各组 `manual_steps`） |
| --- | --- | --- | --- |
| `sft_lf_dirs: {组名: 目录}` | B、C、D | SFT 直接用这个已建好的 LLaMA-Factory 目录（混合配方构建器的输出）里的 `train.yaml`，跳过 `sft-records` 和 `prepare_active_spatial_sft.py`。现在的 `sft_data` 步会对 plan 的 `sft_dir` 重写 `sft_trajectories.jsonl`；矩阵用的是第十一轮的目录，不能让它重跑 | 把混合目录软链到 `TR/lf_mix`，`progress.json` 里把 `render`、`sft_data` 标成 done |
| `start_model` | A、E | 跳过 `render` 到 `pick_sft` 五步，RL 从给定模型开始（A：原始 7B；E：D 组挑出的 SFT 检查点） | `progress.json` 里这五步标 done，写 `rl_model_override`（第十一轮修正包 RL 就是这样起步的） |
| `rl.output_format` | E | `rl_config` 步给 rl-config 传 `--output-format think_action`（rl-config 已支持，第 1.1 节），两个 RL（S0、D）都要 | 手工跑两次 rl-config（`--standard {rl_package}_S0` / `{rl_package}_D`，都加 `--seed 20261019 --output-format think_action --max-response-length 6144`），把 `rl_config` 标成 done |
| `rl.guards.invalid_until` | E | 解析失败率这条只在前 N 次更新里检查（用户要求前 50 次超过 20% 就放弃） | 不加时，`invalid_max: 0.20` 整个训练都适用，比要求严 |
| `eval.rl_prompt_format` | E | `evaluate` 步给 RL 检查点传 `--prompt-format free_think`。现在 RL 检查点固定用默认提示，只有 `eval.models` 里的模型能指定格式 | 手工提交评测任务，SCO 脚本 `eval` 模式的第 6 个参数写 `free_think` |
| `rl_max_response_length` | E | `rl_config` 步给 rl-config 传 `--max-response-length 6144`（rl-config 已支持，第 1.2 节），两个 RL 都要 | 同上一行，手工跑 rl-config 时加上 |
| `diagnostics_after_eval` | 全部 | 评测之后跑失败拆解和 P1 探针 | 手工跑 `eval.diagnostics` |

另外，训练链的 `check_code` 要求主工作树是 plan 里的 commit。矩阵要用的新代码得先合并到 main：
混合配方、诊断、`rl-config --output-format` / `--max-response-length`、`transition/think_rate`。合并由用户在冻结期触发。
各组 YAML 的 `commit` 写的是 `TBD`，`matrix_plan --commit` 可以填。

表里的 7 个字段已经做成补丁（第 6 节）。补丁打上之后就不用手工准备 progress.json 了，各组的 `manual_steps` 只在不打补丁时用。
D_noP0 组（备选）和 D 一样需要 `sft_lf_dirs`、`diagnostics_after_eval`。

## 6. 等冻结期合并的补丁（`docs/transition_qa/patches/`，2026-10-19；2026-10-20 挪到新的 main 上）

两个补丁都改 `data_gen/interiorgs_v10/train_chain.py`。它是主线的现有文件，本分支没有改它：
- 补丁只放在 `docs/transition_qa/patches/`，没有打到分支的代码上；
- 本分支和 main 的这个文件逐字相同，补丁就是对着它做的。

**2026-10-20 重做**：main 新加了 15 个提交（第十二轮批次：`pool: auto`、`retries`、`rl.extend`、`retain`、`sft_records` / `sft_only` 等开关），已合进本分支（`321f63f8`）。`train_chain.py` 改动多，原来的补丁打不上了，按三方合并挪到 main `7e9c4f77` 上：
- 文字冲突 5 处，都是两边各加了开关（模块说明、步骤元组、`sft_data` 开头、`run()` 的跳步、`EVAL_ONLY_STEPS`），两边都保留；
- 另有一处**语义冲突**：main 的 `retain: best_and_last` 在 `evaluate` 一结束就删掉大部分 RL 检查点和 SFT epoch，而补丁 0002 的 `diagnostics` 排在 `evaluate` 之后，要用这些检查点跑探针。改成：plan 里有 `diagnostics` 时，清理放到 `diagnostics` 之后（`retain_after()`）；没有时照旧在 `evaluate` 之后。测试 `test_probes_every_evaluated_model` 核对了顺序。

| 补丁 | 内容 | 改的文件 |
| --- | --- | --- |
| `0001-train-chain-matrix-fields.patch` | 第 5 节的前 6 个字段 | `train_chain.py`（约 80 行），新测试 `tests/test_train_chain_matrix.py`（6 个） |
| `0002-train-chain-diagnostics-step.patch` | 第 7 个字段：新的第 11 步 `diagnostics`，评测之后跑失败拆解和探针（P1；P0 探针到位后同样接入）；有它时 main 的 `retain` 清理挪到它之后 | `train_chain.py`（在 0001 之上约 120 行），`tests/test_train_chain.py`（1 行：步骤多了一个），`tests/test_train_chain_matrix.py`（加 2 个测试） |

### 6.1 每个字段谁写、谁读

所有字段都由 `python -m data_gen.sft_mixture.matrix_plan --group <组> --chain-patch` 从各组 YAML 的 `needs_chain_support` 写进 plan.json；探针路径取自 `configs/matrix_v2/vars.yaml`（`p1_probe`、`p0_probe`）。

| plan 字段 | 哪几组 | 训练链里谁读 | 读了以后写什么 |
| --- | --- | --- | --- |
| `sft_lf_dirs` | B、C、D、D_noP0 | `lf_dir()`：SFT、每个 epoch 的评测、挑 epoch 都用这个目录；`render` 看到它就不渲染；`sft_data` 用 `check_lf_dir()` 核对 train.yaml（`output_dir` 是 `<目录>/model`、epoch 数等于 `sft_epochs`、每个 epoch 存一次），不跑 `sft-records` | progress.json 的 `steps.sft_data.lf_dirs`、`steps.render.note`；LLaMA-Factory 把模型写到 `<目录>/model` |
| `start_model` | A（模型路径）、E（`{from_chain: D 组的 TR}`） | `run()`：`render` 到 `pick_sft` 五步标成 done（note "not planned (start_model)"）；`sft_model()`：RL 起点。E 读 D 组的 `progress.json` 里 `steps.pick_sft.decision.model`，那是 D 组训练链的 `pick_sft` 步写的；D 还没挑出来就停下报错 | `steps.rl_config.start_model`（实际用的起点） |
| `rl.output_format` | E | `rl_config` → `training_prep rl-config --output-format` | RL 的 `train_env.yaml` / `val_env.yaml` 的 `prompt_format: free_think` |
| `rl.max_response_length` | E | `rl_config` → `rl-config --max-response-length` | `ppo.yaml` 的 `data.max_response_length: 6144` |
| `rl.guards.invalid_until` | E | `guard_reason()`：从训练日志的 `transition/invalid_action_rate/mean` 算解析失败率，只在最后一次更新 ≤ 这个值时检查 | 超过门槛时停这个 RL（`guard_stops`） |
| `eval.rl_prompt_format` | E | `evaluate`：RL 检查点的评测任务的第 6 个参数 → `evaluation.py --prompt-format` | 评测结果 `EVAL/<RL 名>_<更新次数>/` |
| `diagnostics` | 全部 | 新的 `diagnostics` 步：`failure_breakdown` 读评测结果（`evaluate` 写的 `EVAL/NAME/results.jsonl`）；探针用 `eval.diagnostics.sco_probe` 提交单卡任务（每个任务 6 个模型），模型是每个 SFT epoch、起点、每个 RL 检查点和 `eval.models`；最后 `qa_probe curve` | `TR/diag/failure_<题集>.{json,md}`、`TR/diag/probe_<名字>/<模型>.json(l)`、`curve.md`、`curve.png`；`steps.diagnostics.job_ids` |

- **用卡**：一个矩阵组约 22 个模型（10 个 SFT epoch、2 个 RL 各 6 个检查点），每个约 3.5 分钟，共约 1.3 卡时，4 个单卡任务。
- **P0 探针**：
  - `p0_probe` 现在是 null，这一项跳过。
  - P0 题库到了以后填路径即可，但 P0 有视频题，`qa_probe.chat` 现在只放图片；要先让它支持视频行。

### 6.2 怎么打（冻结期，用户触发合并之后）

```
git apply --check docs/transition_qa/patches/0001-train-chain-matrix-fields.patch
git apply docs/transition_qa/patches/0001-train-chain-matrix-fields.patch
git apply docs/transition_qa/patches/0002-train-chain-diagnostics-step.patch
python -m pytest -q data_gen/interiorgs_v10/tests/test_train_chain.py data_gen/interiorgs_v10/tests/test_train_chain_matrix.py
python -m data_gen.sft_mixture.matrix_plan --group D --chain-patch --commit <打完补丁的提交> --out <TR>/plan.json
```

- 已核对（2026-10-20，main `7e9c4f77`）：两个补丁按顺序打到 main 的 `train_chain.py` 上没有冲突，结果和测过的文件逐字相同；两个测试文件 33 个测试全部通过（main 原有 25 个、新加 8 个）。main 再改 `train_chain.py` 的话，合并前要重新核对一次。
- 对正在跑的训练链的影响：
  - 打了补丁，旧 plan 没有这些字段，行为不变；
  - 第 11 步 `diagnostics` 记成 "not planned"；
  - 已有的 progress.json 会多一个 `diagnostics` 步，状态 pending。
