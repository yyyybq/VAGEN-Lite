# C 线阶段 2 记录：训练代码（Cambrian-S 第四阶段 + NFP，Qwen3.8-27B）

2026-11-23。分支 qwen38-s4，代码提交 68fd6581 … a12bad55（相对阶段 0 报告 fba32d53，15 个新文件，不改动别的线的文件）。

## 结论

- 阶段 2 的 9 项都完成了：49 个 CPU 单元测试通过；多机等价测试（2 机 × 2 卡对 1 机 × 4 卡）通过。
- 还没提交正式训练任务。阶段 3 要等阶段 0 剩下的几件事（D2、236 GB 下载、NFP 子集规则、镜像）。

## 用户的决定（2026-11-23，"Use the defaults"）

| # | 问题 | 决定（代码现在的默认值） |
|---|---|---|
| 1 | noNFP 组是否也跑 NFP 样本（这一组里这些样本梯度为 0，约占 1/3 算力） | 照跑：两组的数据、步数和学习率曲线完全一致（launch 的 `DATASETS = s4_qa,s4_nfp`，两组相同） |
| 2 | 留出视频的范围 | 只从 NFP 里排除（data_build `--heldout-scope nfp`）；这 1% 视频的 QA 行照常训练 |
| 3 | 检查点里的 `nfp_head.*` 键 | 到阶段 4 评测时，在 bf16 副本里去掉这些键；训练时不改 |
| 4 | 是否保存优化器状态 | 不存（`save_only_model: true`，与 8B 配方相同）；中断后不能接着训 |
| 5 | 图片的像素上限 | 65,536（`image_max_pixels`，与现有 q38 训练相同） |
| 6 | bf16 副本的位置 | 放在 `<run>/ckpt_bf16/checkpoint-N`（q38_sft.json `bf16_copy: sibling`）；fp32 检查点只留最新一个；每组约 7 份 × 54 GB ≈ 380 GB |

## 代码

| 文件 | 内容 |
|---|---|
| `vagen/q38s4/nfp.py` | NFP 头（2 层 MLP + GELU，5120 → 1152）；目标：冻结编码器 blocks[25] 的输出，按 2×2 合并窗口取平均，detach；第 t 单元预测第 t+1 单元；MSE 和 1 − cos 的求和；复制基线；按全局 token 平均缩放 |
| `vagen/q38s4/lf_nfp.py` | LLaMA-Factory 入口（不改 LF 本身）：collator 把 NFP 样本的标签全部设为忽略；loss = text_loss + 0.1·mse + 0.1·cos；检查编码器梯度为 None；日志 text_loss / nfp_mse / nfp_cos / copy_*；`Q38S4_TOKENIZE_ONLY=1` 时分词完就停 |
| `vagen/q38s4/lf_video.py`、`lf_plugin.py` | D4 取帧：QA 在整段视频上均匀取 256 帧，NFP 取 2 fps 下的一段连续窗口；预处理时定下的 token 数和训练时解码出的帧数保证一致 |
| `vagen/q38s4/data_build.py` | VSI-590K → LF 格式：590,667 行 QA，加 29 万行 NFP（按种子抽取），1% 视频留出 |
| `vagen/q38s4/launch.py`、`run_train.sh` | 多机启动：`prepare` 一次性生成运行目录（代码快照、train.yaml、权重、批次计划），`submit --pool zoetrope\|h800` 不加 `--yes` 时只打印 sco 命令 |
| `vagen/q38s4/tools/multinode_equiv.py` | 多机等价测试：准备数据，逐步比较两次运行 |

批次计划：全局 batch 256 = 机器数 × 每机卡数 × 每卡 batch（1）× 梯度累积。8 机时梯度累积为 4；累积步数不是整数的机器数会被拒绝。按 590,667 + 290,000 行算，一个 epoch 约 3,441 步。

用法（阶段 3 时）：

```
python -m vagen.q38s4.launch prepare --run RUN_TOK --mode tokenize --data DATA --tokenized TOK
python -m vagen.q38s4.launch submit --run RUN_TOK --pool zoetrope            # 看命令，再加 --yes
python -m vagen.q38s4.launch prepare --run RUN_NFP --arm nfp --nodes 8 --data DATA --tokenized TOK
python -m vagen.q38s4.launch prepare --run RUN_NONFP --arm nonfp --nodes 8 --data DATA --tokenized TOK
```

## 测试

- CPU 单元测试共 49 个，内容如下：
  - 形状；
  - 在合成序列上检查第 t 单元对第 t+1 单元的错位对齐，并与真实的 token 排布对照；
  - 冻结确实生效；
  - 两个权重都是 0 时，loss 和梯度与普通 Trainer 逐位一致；
  - 取帧（逐帧解码合成视频）；
  - 数据构建；
  - 批次计划；
  - train.yaml 的键都是 LF 认识的参数；
  - sco 命令；
  - 节点入口脚本的检查。
- 多机等价测试（2026-11-22，zoetrope）：
  - 设置：2 层随机小模型（不共享嵌入权重，与 27B 相同），128 行合成数据，NFP 组，全局 batch 8，16 步。
  - A：1 机 × 4 卡（pt-64g9af2a）；B：2 机 × 2 卡（pt-zgl35az5）。两边的总进程数相同，每一步各进程拿到的样本相同。
  - 第 1 步的 loss 两边完全一样：6.271393。
  - 16 步里的最大相对差（|A − B| / 两者中较大的绝对值）：loss 2.6e-5，text_loss 2.6e-5，nfp_mse 2.3e-5，nfp_cos 1.7e-4，梯度范数 6.4e-4。这点差别来自跨机梯度求和的相加顺序不同。
  - NFP 的配对数、视频数和复制基线，每一步都完全一样。
  - 冻结检查：编码器 27 个参数都没有梯度。
  - 结果文件：`/mnt/umm/users/yinbaiqiao/q38/s4/equiv/run_b2/compare.json`。
- 测试中发现的两件事：
  - 一开始的小模型照搬了 4B 的配置，lm_head 和词嵌入共享同一份权重，在 FSDP 下读到分片后的权重而报错。改成不共享，与 27B 一致；训练链的代码没有改。
  - 两次运行都以退出码 3 结束，原因是 bf16 副本的竞争：第 8 步和第 16 步的保存只隔 3 秒，"只留最新一个检查点"的规则删掉了 checkpoint-8，它的副本还没转完。正式训练每 500 步才存一次（相隔几小时），一次 27B 副本只要几分钟，不会碰到。

## 待办

- 阶段 0 剩下的：D2、236 GB 下载（等用户同意）、NFP 子集规则、镜像。
- 测试留下的文件在 `/mnt/umm/users/yinbaiqiao/q38/s4/equiv/`，约 1.0 GB；删不删等用户决定。
- 阶段 3 要测量的：27B 上的解码速度（现在从第 0 帧顺序解码到所需的最后一帧）和每步耗时。
