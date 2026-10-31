# 给主窗口：RLinf 跑 StarVLA 时画面没缩到训练尺寸（2026-10-31）

> 依据：用户 10-31 决定 2："把这个问题和修法写一页给主窗口"。细节见 [E3_CHECK_20261030.md](E3_CHECK_20261030.md)。

## 结论

1. **问题。** RLinf 的 StarVLA 封装只在两个字段里找"模型训练时的图像尺寸"：`datasets.vla_data.image_size` 和 `default_image_resolution`（`rlinf/models/embodiment/starvla/utils/vlm_preprocess.py` 的 `get_train_image_size`）。StarVLA 检查点把这个尺寸存在第三个字段 `obs_image_size` 里。字段名对不上，RLinf 就不缩图，把环境给的画面原样送进模型。
2. **影响。** 下游线 E3（RoboCasa 上的 PPO）环境出 256×256 的图，模型是按 224×224 训的。我们在自己的评测管线里复现了 RLinf 的做法：
   - 3 个任务 × 10 个种子 × 2 种场景划分，同一个模型：缩到 224×224 时成功 15/60（25%），送 256×256 原图时 2/60（3%）；逐回合配对 14 对 1，p = 0.001（精确 McNemar 检验）。
   - E3 训练回合的成功率只有 0%–2%。只有成功才有奖励，所以 100 次 PPO 更新几乎没有正信号，"E3 没学到东西"就是这么来的。
3. **修法（已做，不动 RLinf 和主线代码）。** 下游线的 RLinf 启动钩子 `downstream/rlinf/site/sitecustomize.py`（downstream 分支，提交 `054556af`）在 RLinf 导入 `vlm_preprocess` 后包一层：原来两个字段都找不到时，退回读 `obs_image_size`。找到原字段时行为不变。设环境变量 `DOWNSTREAM_RL_IMAGE_SIZE_FIX=0` 可以关掉，用来做对照。
4. **验证（在跑）。** 在 RLinf 里用不更新参数的策略，评同样 3 个任务各 10 回合：带修复一次、不带修复一次，看带修复的能不能回到我们评测的 23%–27%。第一次提交卡在 RLinf 建集群那一步（40 分钟没输出，已删），10-31 11:11 UTC 重新排队，加了卡住时自动留诊断。结果出来后补一条消息给你。
5. **不重训 E3**（用户决定）。

## 谁会受影响

- 任何在 RLinf 里跑 StarVLA 检查点、而且环境出图尺寸和训练尺寸不一样的实验。检查办法：看检查点的 `config.yaml` 里 `datasets.vla_data` 下面有没有 `image_size` 或 `default_image_resolution`。只有 `obs_image_size` 的，就会出这个问题。
- 主线 RL 不受影响（主窗口 10-31 在 qwen3-main `b6606bac` 上核对：RLinf / StarVLA 只出现在 `examples/train/robocasa/` 和 `examples/evaluate/robocasa/` 的下游脚本里；Active Spatial RL 的代码 `vagen/`、`data_gen/interiorgs_v10`、`examples/train/active_spatial` 都不引用它们，Qwen3 的运行环境也没装 rlinf 和 starvla）。
- **会碰到的情形**：直接跑主线仓库的 `examples/train/robocasa/run_rlinf_starvla.sh`（不经过我们的钩子）。按用户的规定我们只新增 `downstream/`、`docs/downstream/` 下的文件，不改那个脚本；下游线自己的 RLinf 运行都走 `downstream/rlinf/run_rlinf.sh` / `run_rlinf_eval.sh`，带钩子。
- 要在别处修，可以照我们的钩子做（约 30 行，`patch_vlm_preprocess`），或者给检查点的配置补上 `image_size`。后者会改检查点目录里的文件，我们没用这个办法。

## 名词

- **缩图**：把环境出的图缩放到模型训练时的分辨率（这里是 224×224）再送进模型。我们的评测和 StarVLA 自己的推理服务都会做。
- **pretrain / target**：RoboCasa 的两种场景划分（厨房布局和物体不同）。E3 训练用 pretrain，我们的评测用 target。
