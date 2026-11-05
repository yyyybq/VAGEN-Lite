# 理解样本掺入 RL（und_mix）

奖励线 2026-11-05 的任务。做法照 NEO-VLA（arXiv 2611.06918）：每隔一个优化步，把一批理解样本的下一词预测损失乘 0.1，加到策略损失上。

开关**默认关**：不写 `und_mix` 的运行，配置文件、报告和每一步训练都和原来逐字相同（有单测）。分支 `reward-mix` 从 matrix-v3 的 03035318 开出，那是矩阵 D 组密集 RL（m3ds1d）跑的提交。

## 用法

**1. 造样本池**（登录节点，一个核，约一分钟）：

```bash
python -m vagen.und_mix prepare --sft-dir <起点 SFT 文件夹> --datasets P0_train,P1_train \
    --probe P1_probe=<p1_probe.jsonl> --questions X0=<X0 questions.jsonl> --questions G0=<G0 questions.jsonl> \
    --out <POOL>
```

- 从起点 SFT 的数据文件里照原样拷出指定的几份。
- 去掉和留出题集共用题号、任务、场景或图片的行，以及没有图也没有视频的行。
- 按去掉的原因记数，写进 `manifest.json`。

**2. 核对 LLaMA-Factory 路径**（CPU）：

```bash
python -m vagen.und_mix check --pool <POOL> --model <起点模型> --rows 1000 --out <FILE>
```

- 覆盖每一条视频行，外加随机 1,000 行。
- 逐行核对：
  - 要算损失的 token 解码出来正好是各回合的回答；
  - 文本位置是 0..n−1；
  - 图像和视频的网格都在。

**3. 计划里写开关**（train_chain 计划的 `rl`，或一个运行的 `rl`）：

```json
"und_mix": {"pool": "<POOL>", "batch_size": 128}
```

可选键及默认值：

| 键 | 默认 | 含义 |
| --- | --- | --- |
| `interval` | 2 | 每几个优化步掺一次 |
| `weight` | 0.1 | 理解损失的权重 |
| `micro_batch_size` | 1 | 每张卡一次前向放几条 |
| `seed` | 20261105 | 抽样的种子 |

- rl-config 把这个对象（`--und-mix`）写成 `actor_rollout_ref.actor.und_mix`，并补上分词器（RL 起点模型）、日志目录（`<运行目录>/und_mix`）和样本池的哈希。

## 训练里做什么

**什么时候掺**
- 第 k 个优化步（k = 已经走过的优化步数，从 AdamW 的状态里读，续训接着数）满足 k mod interval = 0 时掺一批。
- 这批是 batch_size 条不重复的行，由 (seed, k) 决定；每张卡分到同样多。
- 矩阵的配置里每次 RL 更新只有一个优化步，所以 interval 2 就是隔一次更新掺一次，300 次更新共 150 批。

**样本怎么处理**
- 完全照起点 SFT 的 LLaMA-Factory 0.9.3 路径：
  - 模型线的 Qwen3-VL 注册；
  - SFT 配方的模板、截断长度、历史遮罩、图像和视频像素上限；
  - 转换器、`SupervisedDatasetProcessor`、`SFTDataCollatorWith4DAttentionMask`（补齐到 8 的倍数，bf16）。
- 只在回答 token 上算损失。
- 前向用策略损失同一个函数 `_forward_micro_batch`，位置编号是 RL 路径的四行：文本位置，再加整理器算的三行 M-RoPE。
- 每张卡一次前向只放一条，因为训练用的是 eager 注意力，几条拼在一起会互相看见。

**损失和梯度**
- 损失是整批回答 token 上的平均，乘 weight。每张卡按 weight × 卡数 / 全部回答 token 数缩放，FSDP 在卡间取平均之后正好是这个值。
- 先把策略梯度移开，再算理解项的梯度。
- 截断前分别量两项的范数、两项各占总梯度的比例和夹角余弦，然后把两项加起来。
- 之后照常截断、走优化步。

## 记录

每个优化步写进训练日志，`und` 开头的项在没掺的步上记 0：

| 项 | 内容 |
| --- | --- |
| `actor/und_inserted`、`und_samples`、`und_tokens` | 这步掺没掺、几条、几个回答 token |
| `actor/und_loss` | 理解损失（只在掺的步记） |
| `actor/grad_norm_policy`、`grad_norm_und` | 两项的梯度范数（截断前） |
| `actor/grad_share_policy`、`grad_share_und` | 两项各占总梯度范数的比例 |
| `actor/grad_cos_policy_und` | 两项梯度的夹角余弦 |
| `actor/grad_norm_total_preclip` | 截断前的总范数 |
| `actor/und_seconds` | 这步理解部分花的秒数 |

每次更新另记 `actor/und_batches`（掺了几批）和 `actor/und_update_opt_steps`（几个优化步）。

0 号卡还写两种文件：
- `<运行目录>/und_mix/steps.jsonl`：每个优化步一行，掺的步记下抽到的行号和题号。用 `python -m vagen.und_mix counts` 统计每行被抽到几次；续训重复记的步，按最后一次算。
- `first_batch_rank<r>.json`（每张卡一份）：第一批的回答 token 解码结果和应有的回答，逐条比对。

## 问答评测：X0、G0

训练链的诊断步加了一个可选键 `diagnostics.downstream_qa`（`eval/diagnostics/sco_downstream_qa.py`），默认不开。

- 用 1 卡任务，在下游线的检出里只读调用它自己的 `downstream/x/q0_ask.py` 和打分脚本（`x0b_qa.py` / `g0_qa.py analyze`）。
- 评的模型和 P1 探针相同：每个检查点、起点和计划里列的模型。
- 结果写进 `diag/downstream_qa/summary.json`。
