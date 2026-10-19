# 实验矩阵第二版（A–E）

每组一个 YAML：`A.yaml` … `E.yaml`。写清数据配方、SFT、RL、评测集。说明见 `docs/transition_qa/PROGRESS.md` 第 7 节。
**这里只写配置，没有启动任何训练。** 训练链（`data_gen/interiorgs_v10/train_chain.py`）现在读不了的字段，都放在每个文件的
`needs_chain_support` 里，并在 `docs/transition_qa/CHANGES.md` 第 5 节写了要加什么；
每个文件的 `manual_steps` 写了不改训练链时怎么手工跑。

## 五组

| 组 | SFT 数据 | 输出格式 | 之后 |
| --- | --- | --- | --- |
| A | 无 | 只输出动作 | 原始 7B 直接 RL |
| B | P0 | 只输出动作 | RL |
| C | P0 + P2 | 只输出动作 | RL |
| D | P0 + P1 + P2 | 只输出动作 | RL |
| E | 与 D 相同（共用 D 的 SFT 模型） | SFT 只输出动作；RL 用 `free_think` 提示 | RL（探索性，排在 D 之后；前 50 次更新解析失败率超过 20% 就放弃） |

- **P0** 是仿 8 个 benchmark 的 QA，现在只有 230 道样题（177 条记录）。全量题库可用时（用户通知），改
  `data_gen/sft_mixture/configs/mixture_matrix_v2_20261018.yaml` 的 P0 路径，重新构建。
- **P1** 是转移题：D 用 9,802 道，和 P2 的轨迹条数一样多（用户 2026-10-18 的决定）。
- **P2** 是第十一轮的 SFT 动作轨迹，train 9,802 条，只来自正式题。
- 各组的条数见 `docs/transition_qa/RECIPES.md`。

## 所有组相同的设置

- **SFT：**
  - 模型、硬件：Qwen2.5-VL-7B 全参数，冻结视觉塔，8 卡；
  - 超参数：照第十一轮正式 SFT（学习率 1e-5 余弦，64 条一步，cutoff 16,384）；
  - 时长：**5 个 epoch**（第十轮最终模型就是第 5 个 epoch；第十一轮训到 10 个 epoch 没有提高）；
  - 挑模型：每个 epoch 在 val400 上评测（不采样），挑成功率最高的，打平取更早的；
  - 验证损失：每个池分别算（`eval_on_each_dataset`）。
- **RL：**
  - 任务：只用 train 后备题（SFT 没训过的题，第十一轮第 7 条决定），去掉深度核对标出的题；
  - 算法：修正包 `p1`（GRPO、只训有成有败的题、8 条、学习率 2e-6、KL 0.05、熵 0.01）；
  - 奖励：稀疏 S0（第十一轮稀疏和密集没有差别）；
  - 历史和长度：带历史的一条序列；300 次更新，每 50 次存检查点；
  - 停止条件：同第十一轮修正包，E 组另设。
- **评测：**
  - 题集：val400（398 道 val 题，挑模型）和 test（279 道，只记录）；SFT 每个 epoch、RL 每 50 次各评一次；
  - 诊断：之后跑失败拆解（`eval/diagnostics/failure_breakdown.py`）和 P1 探针（`qa_probe.py`，`p1_probe.jsonl`），
    和 `docs/transition_qa/DIAG_BASELINE.md` 的基线比。

## 用卡估计（按第十一轮实测）

- **SFT：** 第十一轮 9,802 条轨迹每个 epoch 约 10.5 分钟（8 卡）。
  - C 组 5 个 epoch 约 1 小时训练，算上排队、读写约 2 小时；
  - D 组的样本多一倍，但 P1 每条很短，估 2–3 小时；
  - B 组只有 177 条，几分钟。
- **SFT 每个 epoch 的评测：** 5 个 epoch × 2 个题集，每个单卡约 30–60 分钟，并行 10 个。
- **RL：** 300 次约 11.5 小时（1 + 8 卡）。A–E 五组合计约 60 小时的 9 卡时间，能同时跑几组取决于空闲的卡。
- **合计：** 约 600 卡时，RL 占大头。

## 启动前要做的

1. 用户决定把分支 `transition-qa` 合并到 main，且第十一轮训练链已结束。训练链只在主工作树按 plan 的 commit 运行
   （`check_code`）；新代码（混合配方、诊断、`rl-config --output-format`）要在那个 commit 里。
2. 训练链加 `needs_chain_support` 里的字段；或者按各文件的 `manual_steps` 手工准备 progress.json。
3. E 组两件待定的事由用户决定（`E.yaml` 的 `pending`）。
