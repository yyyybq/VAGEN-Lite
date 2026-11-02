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
  - 2026-10-29 起，main 的格式样例会把 Response format 行里的 `<action>...</action>` 换成真实动作名，测试改成按环境实际显示的那一行换回去。
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

## 6. 等冻结期合并的补丁（`docs/transition_qa/patches/`，2026-10-19；2026-10-20 起几次挪到新的 main 上）

两个补丁都改 `data_gen/interiorgs_v10/train_chain.py`。它是主线的现有文件，本分支没有改它：
- 补丁只放在 `docs/transition_qa/patches/`，没有打到分支的代码上；
- 本分支和 main 的这个文件逐字相同，补丁就是对着它做的。

**2026-10-20 重做**：main 新加了 15 个提交（第十二轮批次：`pool: auto`、`retries`、`rl.extend`、`retain`、`sft_records` / `sft_only` 等开关），已合进本分支（`321f63f8`）。`train_chain.py` 改动多，原来的补丁打不上了，按三方合并挪到 main `7e9c4f77` 上：
- 文字冲突 5 处，都是两边各加了开关（模块说明、步骤元组、`sft_data` 开头、`run()` 的跳步、`EVAL_ONLY_STEPS`），两边都保留；
- 另有一处**语义冲突**：main 的 `retain: best_and_last` 在 `evaluate` 一结束就删掉大部分 RL 检查点和 SFT epoch，而补丁 0002 的 `diagnostics` 排在 `evaluate` 之后，要用这些检查点跑探针。改成：plan 里有 `diagnostics` 时，清理放到 `diagnostics` 之后（`retain_after()`）；没有时照旧在 `evaluate` 之后。测试 `test_probes_every_evaluated_model` 核对了顺序。

**2026-10-21 再挪一次**：main 又加了 8 个提交（`renderer_when_trainer_runs` / `renderer_pool`、`renderer_workers`、被守卫停下的运行重启后保持停下、评测表跳过写坏的最后一行），合进本分支（`ca4d4578`）后补丁 0001 打不上了。同样按三方合并挪到 main `616ae4d8`：只有 1 处文字冲突（模块说明里两边各加的段落），两边都保留；0002 直接打上。

**2026-10-21 第三次**：main 又加了 3 个提交（`029efb97`：rl-config `--entropy-coeff` 和 plan rl `entropy_coeff`；`e8730061`：plan rl `renderer_first`；`cc131696`：只改 STATUS），合进本分支（`3924c144`，没有冲突）后两个补丁都打不上了。按三方合并挪到 main `cc131696`：
- 没有文字冲突：rl-config 的参数两边各加在自己的循环里（main 的 `entropy_coeff` 在优化包那组，补丁的 `output_format`、`max_response_length` 在自己那组），模块说明两段都保留；
- `renderer_first` 只改 RL 任务的提交顺序，和补丁改的步骤（`rl_config` 的参数、守卫、评测参数、`diagnostics`）不重叠；
- 补丁现在写完整的 blob 号（`git diff --full-index`），在主仓库里用 `git apply -3` 也找得到原文件。

**2026-10-21 第四次**：main 又加了 23 个提交（第十二轮批次：`renderer_repack`、启动失败最多重交 5 次、分组过滤 `keep_by_return_std` 及其 rl-config 参数、评测的读不出率等），合进本分支（`a5c5f2fc`，没有冲突；本分支在 `gym_agent_loop.py` 加的 `transition/think_rate` 保留）后两个补丁都打不上了。同样按三方合并挪到 main `2b54b45d`：只有 1 处文字冲突（模块说明里两边各加的段落），两边都保留；0002 直接打上。

**2026-10-27**：合入 main `f8e43d40`（`0efaf894`，没有冲突），两个补丁直接打得上。

**2026-10-29 第五次**：main 又加了 29 个提交，合进本分支（`1c641d63`）：
- 第十二轮冻结期 10-22 的 18 个：格式样例、宽松读取、读不出的罚分、D-cond、标准的训练链默认设置、原始 7B 的格式 SFT、实验矩阵登记等；
- interiorgs_qa 的 11 个。

合并时 `training_prep.py` 的 `rl-config` 有 1 处文字冲突。两边各加了开关（本分支的 `--output-format`，main 的 `--format-example`、`--lenient-parse`、`--invalid-penalty`），写的是 env 配置里不同的键，两边都保留。

合并后补丁 0001 打不上了，因为 main 的 6 个提交改了 `train_chain.py`。同样按三方合并挪到 main `8ab5c4d5`：
- 文字冲突 2 处：
  - 模块说明里两边各加的段落，两边都保留；
  - `evaluate` 给 RL 检查点的评测参数：main 改成每个 RL 自己的 `run_options`（带上它训练时的格式样例和宽松读取），补丁在同一行填 `eval.rl_prompt_format`，合成 `(模型, rl_prompt_format, run_options)`。
- 0002 直接打上。
- **语义上对不上的一处，要改的是矩阵的 plan，补丁不用改**：
  - main 把格式样例（提示里示范动作写法的那一行）的默认值改成真实动作名 `action_names`（`configs/admission_v2.json`，用户 10-22 的 3.1.1）。
  - rl-config 默认用它，并拿 `--sft-trajectories` 给的 SFT 对话核对格式行，对不上就拒。
  - 矩阵各组的 plan 没写 `format_example`，`sft_groups` 指向第十一轮的 SFT 轨迹，那是旧写法（`placeholder`）。
  - 所以合并、打补丁之后，每组的 rl_config 步都会被拒。用 D 组的参数实跑过：退出码 1；加 `--format-example placeholder` 就通过。
  - 怎么改等用户定（`REPORT.md` 第 2 节）。

**2026-11-01 预演 qwen3-main（用户选"两件都先做"）**：main 还停在 `dde3d192`，qwen3-main（`b6606bac`，比 main 多 88 个提交）还没合进来。
在 `/tmp` 的临时克隆里预演，原仓库和各工作树都没动：
- 本分支合入 qwen3-main：只冲突 1 个文件 `vagen/agent_loop/gym_agent_loop.py`，两处，两边在同一位置各加了几行：
  - 本分支加的是思考文字计数 `think_turns`，qwen3-main 加的是 `turn_span`、`invalid_spans`、`progress_return` 和 `end_turn()`；两边都保留。
  - 合并后全部登记的测试：699 个通过、8 个跳过、1 个失败。失败的是要读主工作树 `third_party/starVLA` 的那个，一直如此。
- 补丁挪到 qwen3-main 的 `train_chain.py`（三方合并）：
  - 0001 有 3 处冲突，都是同一件事：模型线 10-24 加了 plan `rl_start_model`（一个路径，跳过 SFT、从它起 RL），和补丁的 `start_model` 重叠。补丁的还收 `{"from_chain": TR}`，E 组靠它从 D 组挑出的检查点起步。
  - 处理：两个键并存，各管各的。`SFT_STEPS` 留一份、注释写上两个键；`sft_model()` 先看 `rl_start_model` 再看 `start_model`；`run()` 两个跳过判断各自保留。
  - 0002 直接打上。两个训练链测试文件 48 个测试全部通过（qwen3-main 40 + 补丁 8）。
  - 挪好的补丁在 `docs/transition_qa/patches/qwen3_main_b6606bac/`。qwen3-main 进 main 时如果训练链没再改，就用这两份。

**2026-11-01 集成分支 matrix-v3（用户决定 B，不等 qwen3-main 进 main）**：
- 分支 `matrix-v3` 从 qwen3-main 的头 `b6606bac` 开出（和预演时同一个提交，不用重新预演），工作树 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix`。
  - verl 子模块原来是空的（训练任务从这个工作树读 `verl/`）：从主仓库本地的模块按同一提交 `a0ae85c7` 克隆进来，没联网，没改主仓库的设置。
- 合入 transition-qa（`19291e73`）：和预演一样，只冲突 `gym_agent_loop.py` 两处，两边都保留。
- 两个补丁按 `patches/qwen3_main_b6606bac/` 原样打上，各一个提交：0001 = `dc81dcc8`，0002 = `e4fafbec`；训练链两个测试文件 48 个通过。
- 之后又合入 transition-qa 的 `f334b6bf`（`af4dc0e0`，`sco_probe` 的两处修正，见第 7 节），加上 `matrix_plan --version 3`、配置和 E 组试跑脚本（`c5ceb851`）。
- 第一次全量测试（`c5ceb851`）：714 个通过、8 个跳过、2 个失败。除了一直就有的 starVLA，多出 1 个：
  - 模型线的 `model_matrix/tests/test_model_types.py::ChainTest::test_rl_start_model_skips_the_sft_steps` 列出 plan 有 `rl_start_model` 时链跑的 5 步，没有补丁 0002 加的第 11 步"诊断"。
  - 行为不变：plan 没写 `diagnostics` 时这一步只记"not planned"。没打补丁的 qwen3-main 上这条测试通过。11-01 预演只跑了没打补丁的合并和训练链两个测试文件，所以没发现。
  - 用户选 1（2026-11-01）：把这一步加进那条测试的预期（`32a3ea6d`），并写进补丁 0002（transition-qa `3027ac92`）。新 0002 在干净的 `b6606bac` 上接着 0001 打，4 个文件和 matrix-v3 逐字节相同。
- 第二次全量测试（`120331b6`，限 4 个核）：715 个通过、8 个跳过、1 个失败（一直就有的 starVLA）。
- 11-02 合入 qwen3-main 的新头 `93dcf25a`（主线在 `b6606bac` 之后的 7 个提交；合并提交 `a9a2ac7b`，matrix-v3 快进到它）：
  - 两边都改的只有 `train_chain.py` 和它的测试文件，git 自动合并，没有冲突；补丁 0001 / 0002 在干净的 `93dcf25a` 上照样能打。
  - 先在临时检出里合并、跑全量测试（限 4 个核）：727 个通过、8 个跳过、1 个失败（一直就有的 starVLA），多出的 12 个通过是主线新加的测试。测试过后 matrix-v3 才快进。
  - 第一批必须有它：C、D 的 164 条 P0 样题里有 22 条带视频（P0_val 17 条里 2 条），每条一段 32 帧；改用的全量 P0 里有 248 条（见下）。
    旧代码按 Qwen2-VL 的写法把一段视频写成一个视觉块；Qwen3-VL 要每两帧一块，块前写时间。主线的 `9e4b054c` 改了这一点。
    （之前说"只有全量 P0 有视频"是错的：样题也有。）
  - CPU 上核对过（`exps/matrix_v3/checks/`）：
    - 24 条视频记录的编码和 Qwen3-VL 自己的处理器逐 token 相同，像素和网格也相同；
    - 用 SFT 的 collator 加一个随机初始化的小 Qwen3-VL（真实的切块和位置编码，窄、两层）跑 bf16 前向、反向，全部通过：单条、两段视频同批、视频和动作记录同批都试了；
    - 位置编码三维齐全，视频 / 图片占位数等于视觉特征数，损失有限，视觉塔有梯度。
    - 真模型在 GPU 上带视频的训练步还没跑过：第一批的 SFT 就是第一次。
  - 主线的两条新停止线（熵、第 91 条）只在 plan 的 `rl.guards` 写了才生效。主线 11-02 转告：用户批了第 9 项（甲），矩阵 RL 也加熵线（`entropy_max` 0.3、`entropy_window` 5）和 `vision_token_crashes` 2。11-02 用户在本分支确认（同时确认密集一侧用 V4），四条链的 plan 都写进了 `rl.guards`（见第 7 节 `matrix_plan --version 3` 一条），预登记写明是 11-02 确认的、在矩阵 RL 开跑前加入。
- 11-02 P0 改用全量（用户改定；transition-qa `22fb9e74`，合入 matrix-v3 为 `03035318`）：
  - 新混合配置 `mixture_matrix_v3_qwen3_20261102`，和 10-30 的只差 P0（`exps/interiorgs_qa_final_20261028/lf_256`）和 P0 的验证条数（val 场景 300 条）。在登录节点上构建（1 个进程、nice 19，6 分钟）。
  - 行数：P0 10,144（去掉边缘场景的 735 条），B′ 10,510、C 19,219、D 28,294、D_noP0 18,150。
  - P1、P2q、P2f 的文件四个组都和样题版逐字节相同，`train.yaml` 只差两条路径；P0 没有 val、test 和被排除场景的行，图片和视频帧都在。
  - 合并只冲突 C、D 链配置里 `sft_lf_dirs` 一行，保留 matrix-v3 的（本工作树里的复制）。复制逐字节核过，样题版的复制挪到 `exps/matrix_v3/sft_p0sample_20261030`（没有训练过）。
  - 全量 P0 有 248 条视频记录（训练 240、验证 8，每段 32 帧，一条最多 10 道题）。按题组抽 24 条，同样在 CPU 上核对：编码 24 条全对，训练步 8 批全过（`exps/matrix_v3/checks/video_check_fullp0.json`）。
  - 第四次全量测试（`03035318`，限 4 个核）：727 个通过、8 个跳过、1 个失败（一直就有的 starVLA）。
- 训练链从这个工作树跑：起链时设 `VAGEN_LITE_WORKTREE=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix`（`check_code` 核对它等于链所在的检出）。
  - 链在跑时，这个工作树里只改 `docs/`；代码要改，等链跑完。

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

- 已核对（2026-10-29，main `8ab5c4d5`）：两个补丁按顺序打到合并了 main 的本分支上没有冲突，结果和三方合并得到的文件逐字相同；本分支的 `train_chain.py` 和 main 的逐字相同。两个测试文件 42 个测试全部通过（main 原有 34 个、新加 8 个）。main 再改 `train_chain.py` 的话，合并前要重新核对一次。
- 要先合并本分支再打补丁（就是上面的顺序）：`diagnostics` 步用到本分支的 `eval/diagnostics`，只有 main 的代码时它的 4 个测试会因为找不到模块而失败。
- 对正在跑的训练链的影响：
  - 打了补丁，旧 plan 没有这些字段，行为不变；
  - 第 11 步 `diagnostics` 记成 "not planned"；
  - 已有的 progress.json 会多一个 `diagnostics` 步，状态 pending。

## 7. 矩阵第三版（Qwen3-VL-8B，2026-10-30）：混合数据构建器和配置

用户 2026-10-30 选 (a)：按主线 10-27 的登记分两步做（qwen3-main `d530b546` 的 `configs/matrix_r12.json`：决定 74、75、78–80）。这一步是 SFT 数据和组的定义，没有改主线的任何现有文件。

- **`data_gen/sft_mixture/build.py`**：
  - 认两个新池 P2q、P2f（`POOLS`、`POOL_ZH`），读法和 P2 一样（`kind: sft_trajectories`），每行的 `pool` 写池自己的名字；
  - 配置里出现不认识的池就停；
  - 池可以写 `format_example`（`action_names` / `placeholder`）：每条对话的系统提示和每个用户回合的格式行，都要和 RL 环境在这个格式样例下显示的逐字相同（`release.protocol_system_prompt` / `_format_block`），否则停。行仍然原样复制，不改写；
  - 配方表的列按配置里有的池排。第二版的配置有 P0、P1、P2，表和以前一样。
- **新配置** `data_gen/sft_mixture/configs/mixture_matrix_v3_qwen3_20261030.yaml`：
  - 内容：B′ = P0 + P2f，C = P0 + P2q，D = P0 + P1 + P2q，E 同 D，D_noP0；
  - 种子同第二版；配方用模型线的 Qwen3 SFT 配方；去掉 128 个边缘场景。
  - 产物在 `exps/sft_mixture_matrix_v3_qwen3_20261030`，条数见 `RECIPES.md`。
- **`configs/matrix_v3/`**：
  - 各组的定义（`vars.yaml`、`A`–`E`、`C0`、`D_noP0`、`README.md`）。A 关闭；B′ 的经过和风险写在 `B.yaml`；C0 是模型线的两个种子。
  - RL 写的是 S0 / V4、两个种子、读不出 10%、KL 1.0、eager。
  - `chain.plan` 是 `null`：训练链的部分等 qwen3-main 合进 main。
- **`configs/matrix_v2/README.md`**：开头注明第二版已关闭。`matrix_plan` 和补丁的测试仍用第二版。
- **测试**：
  - `test_build.py` 加 2 个：新池的文件、标签和配方表的列；格式核对对旧写法和别的题族的系统提示都会停；
  - 新文件 `test_matrix_v3.py` 8 个：各组和 `vars.yaml`、和主线的登记、和混合配置、和集群上建好的数据一致；A 关闭；名字不和已有的重复；C0 的检查点都在。
  - `data_gen/sft_mixture/tests` 共 27 个，全部通过。
- **预登记**：第三版 2026-10-30 定稿，替换了 `PREREG_D_vs_C.md`；旧版（Qwen2.5）在 `PREREG_D_vs_C_v2_qwen25.md`。
- **`eval/diagnostics/qa_probe.py` 的 `compare`**（2026-10-30，按预登记第三版）：
  - 一组可以写两个种子：`GROUP=SEED1:S0,D;SEED2:S0,D`，种子标签可省（seed1、seed2）；
  - 配对是同一种子、同一侧，两组的种子标签要相同；单个结果仍对上另一组的每一对；
  - 每个种子另报两对的平均差、McNemar、自助法的置信区间；合起来的差是所有对的平均，McNemar 用所有"题 × RL"，自助法用同一组抽样；
  - "同号"按两个种子算（每个种子的差是两对的平均）；只有一个种子时照旧按 S0、D 两对算；
  - JSON 多了 `per_seed`、每对的 `seed` / `side` 和 `rules.same_sign_over`；Markdown 多了每个种子的行。
  - 测试：`test_diagnostics.py` 加 3 个（写法、配对规则、两个种子的端到端：一对为负但两个种子同号判"有差异"，两个种子反号判不过），共 23 个通过。
  - 兼容：10-29 的三个 C0 比较用新代码在 /tmp 重跑，旧 JSON 的每一项都相同。
- **`eval/diagnostics/sco_probe.py --python`**（2026-11-01）：用别的 Python 跑探针，比如 Qwen3-VL 的运行时 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/.venv-qwen3vl-runtime/bin/python`（vLLM 0.11.0）。不给时启动脚本和原来逐字相同，有测试核对。
- **`sco_probe` 从 Qwen3 训练链里用**（2026-11-01，`f334b6bf`）：补丁 0002 的诊断步在链的进程里调 `prepare`，传 plan 的 `pool`。
  - qwen3-main 的 plan 池可以是链的模式（`auto` / `queue` / `idle`），不是 sco 的池名，原样传会提交失败；现在换成此刻排队最少的池。
  - 链按 plan 的 `model_type` 设好 `VAGEN_LITE_PYTHON`，现在它就是探针的默认 Python（Qwen3 要用它的运行时）。
  - `--python` 仍优先；两样都没有时启动脚本不变。训练链代码没改。
- **`matrix_plan --version 3`**（matrix-v3 分支，`c5ceb851`）：每组每个种子一个 plan，先做 C、D。
  - 第一个种子（20261019）的链训 SFT（`sft_lf_dirs`），第二个种子（20261021）从它挑出的检查点起步（`start_model: {from_chain}`）；
  - 每条链一对 S0 / 密集，名字 `m3_<组>_s<种子>`、`sft_m3<组>_e<k>`、`m3<组>s<k>s0` / `m3<组>s<k>d`，任务名前缀 `m3`；
  - 密集一侧在用户转达主线的答复前是占位 `PENDING_dense_baseline`：rl-config 不认它，链挑完 SFT 后停在 RL 配置；填上后重写 plan、再跑链，两侧一起开始；
  - 11-02 用户确认 V4、两条新停止线和 zoetrope。matrix-v3 停在 `03035318` 不动（链的任务从它的工作树跑），所以四条链的 plan 由 `exps/matrix_v3/watch/m3_rlplan.py` 写：`--dense registered` 生成，再把池改成 zoetrope、`rl.guards` 加 `entropy_max` 0.3、`entropy_window` 5、`vision_token_crashes` 2，每条链的 `plan_edit.json` 记着。第一个种子的旧 plan 留作 `plan_placeholder_20261102.json`，新旧只差密集一侧和停止线；
  - 两条链的 rl-config 在临时目录里用原始 Qwen3 空跑过（`exps/matrix_v3/dry_rlconfig_20261102/`）：S0 和 V4 只差奖励和补题方式，RL 题逐字节相同（C、D 也相同，10,124 条）；
  - 值守脚本 `exps/matrix_v3/watch/m3_watch.py`（登录节点，nice 19，一个线程，每 5 分钟）：第一个种子的链停在 RL 配置时用新 plan 再跑；它挑出 SFT 后起第二个种子的链；D 挑出后在它上面重跑 E 的试跑。启动前核对 HEAD、暂停文件和核数，别的情况只记下、留给会话；
  - matrix-v3 的 `vars.yaml` 链部分（`pool: zoetrope`、`dense_baseline: registered`）等这批链跑完再改；本分支的 `vars.yaml` 和各组的 `rl.guards` 已加两条线，合并后生成的 plan 自带；
  - 链写的一切都在 matrix 工作树的 `exps/matrix_v3/` 下：C、D 的 SFT 数据和两个评测集是 VAGEN-Lite-data 那份的复制（逐字节相同）。
- **`eval/diagnostics/e_trial.py`**（matrix-v3，`c5ceb851`）：E 组的单卡试跑。评测流程加 free_think 提示、RL 的采样，每回合可生成本局 6,144 个词元里还剩的部分（和 RL 的拼接模式一样），而不是评测的每回合 384；限时 45 分钟。
- 第 5、6 节的训练链字段和补丁第三版照样要用；qwen3-main 合进 main 后，补丁要对合并后的训练链重新核对（2026-11-01 已对 `b6606bac` 预演，见第 6 节）。
