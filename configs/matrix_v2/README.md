# 实验矩阵第二版（A–E 和 C0）

> **已关闭（2026-10-30）**：主线 10-27 的决定 78 规定新实验一律用 Qwen3-VL-8B，Qwen2.5 的矩阵关闭。用户 2026-10-30 选择按主线的登记改做第三版（`configs/matrix_v3`）。本目录留作记录，`matrix_plan` 和补丁的测试仍用它；现行的是第三版。

> **名字提醒**：`v10_p1` / `v10_p2` 是 **RL 修正包**的名字（`data_gen/interiorgs_v10/configs/rl_standards.json` 里的优化设置：GRPO、每次更新抽多少候选题等），和**数据池** P1（转移题）/ P2（第十一轮的 SFT 动作轨迹）**无关**。

每组一个 YAML：`A.yaml` … `E.yaml`、`C0.yaml`。写清数据配方、SFT、RL、评测集。说明见 `docs/transition_qa/PROGRESS.md` 第 7、9、10 节。
**这里只写配置，没有启动任何训练。**

- **RL 标准名是变量**（用户 2026-10-19）：各组写 `{rl_package}_S0` / `{rl_package}_D`，`{rl_package}` 在 `vars.yaml` 里，现在是 `v10_p1`。
  - `{pkg}` 是它的短名（`p1`），用在 RL 的目录名里。
  - 训练链的 plan 由 `python -m data_gen.sft_mixture.matrix_plan --group D [--rl-package v10_p2] [--commit SHA] --out plan.json` 生成：填上变量，再核对两个标准都存在、只差奖励。
  - 第十二轮如果把 p2 定为标准，只改 `vars.yaml` 一行。
- 训练链（`data_gen/interiorgs_v10/train_chain.py`）现在读不了的字段，放在每个文件的 `needs_chain_support` 里；要加什么见 `docs/transition_qa/CHANGES.md` 第 5 节。
- 每个文件的 `manual_steps` 写了不改训练链时怎么手工跑。
- 2026-10-19：
  - 按第十二轮的决定更新：每组的 RL 成对跑（S0 和 D），SFT 10 个 epoch，评测集换成 v2；
  - 加 C0；
  - RL 标准名做成变量；
  - E 组的两件事定了（含思考文字的回答比例、回答上限 6,144）。

## 六组

| 组 | SFT 数据 | 输出格式 | 之后 |
| --- | --- | --- | --- |
| A | 无 | 只输出动作 | 原始 7B 直接 RL |
| B | P0 | 只输出动作 | RL |
| C0 | P2（**主线**：第十一轮正式 SFT 第 9 个 epoch，不新训练） | 只输出动作 | 成对 RL = 第十二轮的复现对，**不新训练**，结果取第十二轮的训练链 |
| C | P0 + P2 | 只输出动作 | RL |
| D | P0 + P1 + P2 | 只输出动作 | RL |
| E | 与 D 相同（共用 D 的 SFT 模型） | SFT 只输出动作；RL 用 `free_think` 提示 | RL（探索性，排在 D 之后；前 50 次更新解析失败率超过 20% 就放弃；回答上限 6,144） |
| D_noP0（备选） | P1 + P2，不含 P0；P1、P2 的行和 D 逐条相同 | 只输出动作 | RL，其余和 D 一致；**只写配置，不排期**（用户 2026-10-19） |

每组的 RL 都是一对：`{pkg}s0`（修正包 + 稀疏奖励 S0）和 `{pkg}d`（修正包 + 密集奖励 D）。

- **P0**：仿 8 个 benchmark 的 QA，现在只有 230 道样题（177 条记录，去掉边缘场景后 164 条）。全量题库可用时（用户通知），改 `data_gen/sft_mixture/configs/mixture_matrix_v2_20261018.yaml` 的 P0 路径，重新构建。
- **P1**：转移题。D 用 9,075 道（2026-10-29 去掉边缘场景后；之前 9,802 道），和 P2 的轨迹条数一样多（用户 2026-10-18 的决定），从 2026-10-20 重建的题库 `exps/transition_qa_20261019b` 里抽（修掉了只看文字就能答的捷径，`docs/transition_qa/DIAG3_PROBE.md`）。
- **P2**：第十一轮的 SFT 动作轨迹，train 9,802 条，去掉边缘场景后 9,075 条，只来自正式题。
- **边缘场景**（用户 2026-10-29）：128 个场景（`configs/scene_screen/scene_exclusions_20261029.json`）从 P0、P1、P2、RL 题和评测集里都去掉了，见 `docs/transition_qa/SCENE_SCREEN.md`。
- 各组的条数见 `docs/transition_qa/RECIPES.md`。

## C0 和矩阵的 SFT 配方（核对于 2026-10-19）

C0 是主线：第十一轮正式 SFT（`interiorgs_v10_train_r11_20261016/lf_formal`，只练 P2，10 个 epoch）挑出的第 9 个 epoch（`checkpoint-1386`），加从它起步的成对 RL。

- **RL 部分**：第十二轮的复现对正是这一对（`interiorgs_v10_train_r12a_20261019`，STATUS 的 R12-P1-S0 / R12-P1-D），C0 不新训练，直接用那条训练链的结果。
- **核对**：`test_matrix_configs.py` 核对第十二轮的 plan 和 D 组的 plan 在这些字段上完全相同：
  - RL：标准、种子、后备题、深度核对排除、停止条件、启动重交；
  - 300 次更新、评测集、评测频率、SFT 目录、原始模型。
- **p2 的情况**：如果改用 `v10_p2`，C0 对应 `plan_r12b.json`（R12-P2-S0 / -D，2026-10-19 还没开始）。

B、C、D 的 `train.yaml` 和主线的 `lf_formal/train.yaml` 逐项比较：

| 项 | 主线（C0） | B / C / D | 影响 |
| --- | --- | --- | --- |
| 模型、全参数、冻结视觉塔、模板 qwen2_vl、cutoff 16,384 | 同 | 同 | 无 |
| 学习率 1e-5 余弦、warmup 3%、每卡 1 条 × 梯度累积 8 × 8 卡 = 每步 64 条、bf16、梯度检查点、种子 42 | 同 | 同 | 无 |
| 10 个 epoch、每个 epoch 存一次 | 同 | 同 | 无 |
| 训练数据 | `trajectory_train`（9,802 条） | 各组的池；P2 是主线那 9,802 条去掉边缘场景的 727 条，其余**逐条相同、顺序相同**（对话和图片） | 组间差别就在这里 |
| 验证集 | `trajectory_val`（全部 1,293 条 val 轨迹） | P0 17 / P1 300 / P2 300，每个池分别算（`eval_on_each_dataset`）；P2 的 300 条是那 1,293 条去掉边缘场景后（1,169 条）的子集 | 只影响验证损失，不影响训练 |
| `image_max_pixels` / `video_max_pixels` | 不写（LLaMA-Factory 默认 768×768 / 256×256） | 65,536（256×256） | 动作帧是 256×256，两边都不缩放；P0 的样题导出也是 256 |
| 输出目录、`val_size`（0 / 不写，默认 0） | — | — | 无 |
| 挑 epoch | 第十一轮按 val400（398 道）挑：e9 41.96% | 按 eval_val_v2_screened（382 道）挑 | 规则相同（成功率最高、打平取更早），题集不同：v2 多了 15 道遮挡题，又去掉了边缘场景 |

- **每个 epoch 的步数随数据量变**：主线 154 步，C 145 步（9,239 条），D 287 步（18,314 条；2026-10-29 去掉边缘场景后）；warmup 按比例，同样是 3%。
- **C 和 C0 的比较**：两者的训练数据差在 P0 的 164 条，以及 2026-10-29 起 C 去掉了 128 个边缘场景（P2 少 727 条，RL 题也去掉这些场景）。所以 C − C0 是 P0 和去掉边缘场景合起来的作用，另含 SFT 重训的随机性和挑 epoch 用的题集不同。

## 所有组相同的设置

- **SFT：**
  - 模型、硬件：Qwen2.5-VL-7B 全参数，冻结视觉塔，8 卡；
  - 超参数：照第十一轮正式 SFT（上表）；
  - 时长：**10 个 epoch**（第十二轮第 6 条）；
  - 挑模型：每个 epoch 在 val 上评测（不采样），挑成功率最高的，打平取更早的；
  - 验证损失：每个池分别算。
- **RL：**
  - 成对：每组两个，`rl.standard` 分别是 `{rl_package}_S0`、`{rl_package}_D`，两者只差奖励（第十二轮第 2 条）；
  - 任务和种子：只用 train 后备题，去掉深度核对标出的题和 128 个边缘场景（`exps/matrix_inputs_20261029/rl_rows`）；种子 20261019，所有组、所有 RL 都一样；
  - 长度和检查点：300 次更新，每 50 次存检查点；
  - 停止条件：同第十一轮修正包，E 组另设；H800 启动失败自动重交，最多 2 次（第十二轮第 5 条）；
  - 回答上限：一条序列模式下整个 episode 4,096 个词元，**只有 E 是 6,144**；
  - 日志：`transition/think_rate` 记含思考文字的回答比例，所有组都记，只输出动作的组应接近 0。
- **评测：**
  - 题集：第十二轮的 v2 题集去掉 128 个边缘场景（用户 2026-10-29）。
    - `val` = `exps/matrix_inputs_20261029/eval_val_v2_screened`，382 道：第十二轮的 eval_val_v2（413 道）去掉边缘场景的 31 道，不补，挑模型用；
    - `test` = `exps/matrix_inputs_20261029/eval_test_v2_screened`，292 道：eval_test_v2 去掉边缘场景的 59 道，从留下的测试场景里同族同难度档补回（回放检查过；遮挡族没有可补的题，13 个名额给了另外五族，遮挡族 29 道），只记录。
  - 频率：SFT 每个 epoch、RL 每 50 次各评一次。
  - 诊断：
    - 失败拆解：`eval/diagnostics/failure_breakdown.py`；
    - P1 探针：`qa_probe.py`，用 `exps/transition_qa_20261019b/p1_probe.jsonl`（838 道，2026-10-29：`goal_now` 176、`goal_after` 171，`occluded_now`、`occluded_after` 33、58，其余每类 30–40）；
    - 基线：失败拆解看 `docs/transition_qa/DIAG_BASELINE.md`，探针看 `DIAG3_PROBE.md`。

## 用卡估计（按第十一轮实测）

- **SFT：** 第十一轮 9,802 条轨迹 10 个 epoch，8 卡训练 1 小时 44 分，算上排队、读写约 4.5 小时。
  - C 组和它同量，约 4.5 小时；
  - D 组样本多一倍、P1 每条很短，估约 6 小时；
  - B 组只有 164 条，十几分钟；
  - C0 不训练。
- **SFT 每个 epoch 的评测：** 10 个 epoch × 2 个题集 = 每组 20 个单卡任务，并行 10 个，约 1 小时。
- **RL：**
  - 每个 300 次约 11.5 小时（1 + 8 卡）；
  - A–E 各一对，共 10 个，约 1,035 卡时；C0 用第十二轮的，不另算；
  - 一对可以同时跑（18 张卡），也可以先后跑；
  - E 的回答更长，每次更新会更慢。
- **合计：** 约 1,300 卡时，RL 占大头。

## 启动前要做的

1. 合并：分支 `transition-qa` 一直和 main 保持同步；合并到 main 由用户在冻结期触发（用户 2026-10-19 的决定）。
   训练链只在主工作树按 plan 的 commit 运行（`check_code`），新代码要在那个 commit 里：混合配方、诊断、`rl-config --output-format` / `--max-response-length`、思考文字比例的指标。
2. 训练链加 `needs_chain_support` 里的字段：
   - 已做成补丁 `docs/transition_qa/patches/0001`、`0002`（`CHANGES.md` 第 6 节）；
   - 合并后打上，再用 `matrix_plan --group <组> --chain-patch --commit <提交>` 生成 plan；
   - 不打补丁就按各文件的 `manual_steps` 手工准备 progress.json。
   - 补丁 0002 的 `diagnostics` 步会给每个检查点自动跑 P1 探针，探针集路径在 `vars.yaml`（`p1_probe`；`p0_probe` 到位后填）。
3. 定 RL 标准：修正包 p2（48 道候选题）还是第十二轮的候选标准。定下来后改 `vars.yaml` 的 `rl_package`。
