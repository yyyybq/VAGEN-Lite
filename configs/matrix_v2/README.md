# 实验矩阵第二版（A–E）

每组一个 YAML：`A.yaml` … `E.yaml`。写清数据配方、SFT、RL、评测集。说明见 `docs/transition_qa/PROGRESS.md` 第 7、9 节。
**这里只写配置，没有启动任何训练。**

- 训练链（`data_gen/interiorgs_v10/train_chain.py`）现在读不了的字段，放在每个文件的 `needs_chain_support` 里；
  要加什么见 `docs/transition_qa/CHANGES.md` 第 5 节。
- 每个文件的 `manual_steps` 写了不改训练链时怎么手工跑。
- 2026-10-19 按第十二轮的决定更新：每组的 RL 成对跑（S0 和 D），SFT 10 个 epoch，评测集换成 v2。

## 五组

| 组 | SFT 数据 | 输出格式 | 之后 |
| --- | --- | --- | --- |
| A | 无 | 只输出动作 | 原始 7B 直接 RL |
| B | P0 | 只输出动作 | RL |
| C | P0 + P2 | 只输出动作 | RL |
| D | P0 + P1 + P2 | 只输出动作 | RL |
| E | 与 D 相同（共用 D 的 SFT 模型） | SFT 只输出动作；RL 用 `free_think` 提示 | RL（探索性，排在 D 之后；前 50 次更新解析失败率超过 20% 就放弃） |

每组的 RL 都是一对：`p1s0`（修正包 p1 + 稀疏奖励 S0）和 `p1d`（修正包 p1 + 密集奖励 D）。

- **P0** 是仿 8 个 benchmark 的 QA，现在只有 230 道样题（177 条记录）。全量题库可用时（用户通知），改
  `data_gen/sft_mixture/configs/mixture_matrix_v2_20261018.yaml` 的 P0 路径，重新构建。
- **P1** 是转移题：D 用 9,802 道，和 P2 的轨迹条数一样多（用户 2026-10-18 的决定）。
- **P2** 是第十一轮的 SFT 动作轨迹，train 9,802 条，只来自正式题。
- 各组的条数见 `docs/transition_qa/RECIPES.md`。

## 所有组相同的设置

- **SFT：**
  - 模型、硬件：Qwen2.5-VL-7B 全参数，冻结视觉塔，8 卡；
  - 超参数：照第十一轮正式 SFT（学习率 1e-5 余弦，64 条一步，cutoff 16,384）；
  - 时长：**10 个 epoch**（第十二轮第 6 条）；
  - 挑模型：每个 epoch 在 val 上评测（不采样），挑成功率最高的，打平取更早的；
  - 验证损失：每个池分别算（`eval_on_each_dataset`）。
- **RL：**
  - 成对：每组两个，`rl.standard` 分别是 `v10_p1_S0`、`v10_p1_D`（`data_gen/interiorgs_v10/configs/rl_standards.json`），
    两者只差奖励（第十二轮第 2 条）；
  - 任务和种子：只用 train 后备题，去掉深度核对标出的题；种子 20261019，所有组、所有 RL 都一样；
  - 长度和检查点：300 次更新，每 50 次存检查点；
  - 停止条件：同第十一轮修正包，E 组另设；H800 启动失败自动重交，最多 2 次（第十二轮第 5 条）。
- **评测：**
  - 题集：第十二轮的 v2 题集。
    - `val` = `interiorgs_v10_r12_20261019/eval_val_v2`，413 道，第十一轮的 val400 加 15 道遮挡题，挑模型用；
    - `test` = `eval_test_v2`，292 道，279 加 13 道遮挡题，只记录。
  - 频率：SFT 每个 epoch、RL 每 50 次各评一次。
  - 诊断：之后跑失败拆解（`eval/diagnostics/failure_breakdown.py`）和 P1 探针（`qa_probe.py`，`p1_probe.jsonl`），
    和 `docs/transition_qa/DIAG_BASELINE.md` 的基线比。基线的失败拆解是在第十一轮的 val400 / test 上算的，v2 多了 28 道遮挡题。

## 用卡估计（按第十一轮实测）

- **SFT：** 第十一轮 9,802 条轨迹 10 个 epoch，8 卡训练 1 小时 44 分，算上排队、读写约 4.5 小时。
  - C 组和它同量，约 4.5 小时；
  - D 组样本多一倍、P1 每条很短，估约 6 小时；
  - B 组只有 177 条，十几分钟。
- **SFT 每个 epoch 的评测：** 10 个 epoch × 2 个题集 = 每组 20 个单卡任务，并行 10 个，约 1 小时。
- **RL：** 每个 300 次约 11.5 小时（1 + 8 卡）；五组各一对，共 10 个，约 1,035 卡时。一对可以同时跑（18 张卡），也可以先后跑。
- **合计：** 约 1,300 卡时，RL 占大头；没成对时估的是约 600 卡时。

## 启动前要做的

1. 第十二轮训练链结束后，把 main 快进到分支 `transition-qa`（用户 2026-10-19 的决定）。训练链只在主工作树按 plan 的 commit 运行（`check_code`）；
   新代码（混合配方、诊断、`rl-config --output-format`）要在那个 commit 里。
2. 训练链加 `needs_chain_support` 里的字段；或者按各文件的 `manual_steps` 手工准备 progress.json。
3. E 组两件待定的事由用户决定（`E.yaml` 的 `pending`）。
4. 修正包 p2（48 道候选题）还是第十二轮的候选标准。如果第十二轮把它定为标准，把各组的 `rl.standard` 换成 `v10_p2_S0` / `v10_p2_D`。
