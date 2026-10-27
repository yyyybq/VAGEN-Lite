# 模型线第一步：盘点与核实（2026-10-24）

模型线：把 Qwen3-VL-8B-Instruct 和 Cambrian-S（7B 为主）接进现在的统一框架，在主线现在最高的 setting 上跑主实验和下游实验，作为实验矩阵在模型维度上的扩充。本报告只做第一步：盘点、核实、单卡加载测试、估算。没有训练，没有安装，没有下载权重；唯一的 GPU 用量是两个 1 卡加载测试任务，共约 15 分钟。

时间都是 UTC。路径都在集群 fujian 上；`E` 指 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial`。

## 结论

1. **Qwen3-VL-8B-Instruct 可以接，权重在，单卡加载测试通过。**
   - 权重在共享目录，4 个分片都在，能完整加载（87.7 亿参数，bf16 加载后显存 16.9 GiB）。
   - 用 Transformers 和用 vLLM 加载都能输出，读出来的动作两边一样，都是能执行的动作。
   - 要补三处，估计 2–3 天代码：
     - SFT 工具：本地 LLaMA-Factory 没有 Qwen3-VL 的对话模板；
     - RL 的"一道题一个序列"路径：verl 会把 Qwen3-VL 当成 Qwen2.5-VL 来算位置编码；
     - 运行环境：主环境的 transformers 是 4.56.1，不认 Qwen3-VL；10 月 2 日建过一个只换 transformers 4.57.6 的环境，今天的测试就用它跑通。
2. **Cambrian-S 现在不能开始：集群上没有它的任何一个原始权重，也没有它建模型时必须读的 SigLIP2 视觉编码器文件。**
   - 要从集群外拷进来：Cambrian-S-7B 32.13 GB，SigLIP2 4.58 GB。
   - 8 月调通的代码还在、环境版本没变，vLLM 那条采样路径在今天的代码上能跑（用 8 月 RL 训过的 Cambrian 检查点、替代的图像预处理测的）。
   - 但它当时只接在旧的"每回合一个样本"路径上。主线现在用的是"一道题一个序列"，要另接；SFT 也没有现成工具。
   - Cambrian 每帧画面占 756 个 token，Qwen2.5-VL 是 81 个。一条轨迹估计平均约 9,000、最长约 13,000 个 token，超过主线 RL 的回复上限 4,096；GPU 用量估计是 Qwen 的 3.5–5 倍（估算）。
   - 代码估计 4–5 天。
3. **下游：Qwen3-VL-8B 能作 StarVLA 的骨干，缺口小；Cambrian-S 不能。**
   - StarVLA 按权重 `config.json` 的模型类型选接口，`qwen3_vl` 有现成接口；动作头的输入维度取骨干的隐藏维度，没有写死尺寸。
   - Qwen3-VL 要补的：起点登记、按 bf16 加载、RLinf 里 FSDP 按层切分的层名。约 1 天，都是新文件。
   - Cambrian-S 在 StarVLA 里没有接口。按你的规定，这里停下来等你，不改 StarVLA。
4. **最高 setting 的每一处路径都已核实**（第四节的表）。
   - 密集变体按现在能看到的汇总应选 V4，即 D-cond + keep，进度奖励系数除以 3。
   - V4 在合并 705 道评测题上成功率 48.2%，通过主线挑选规则，但只有一个种子；第二个种子在训。
   - 汇总文件是 10-24 15:04 UTC 的；组合变体 18:18 刚提交。真正的选择在提交 RL 时再读一次。
   - V4 需要的代码不全在 main：要副本分支 dense-batch2 上的 6 个提交，它们互相依赖，只能按顺序一起拣进本线分支。
5. **GPU 估算**（第七节）：
   - Qwen3-VL 主实验约 170 卡·小时（SFT、两个 RL 臂、评测），下游 4 个起点约 1,700 卡·小时；
   - Qwen2.5-VL 参照行的主实验约 160 卡·小时；
   - Cambrian 主实验约 580–850 卡·小时。
   - 只有 Qwen2.5-VL 的那部分是实测换算的，其余是估算。下游是大头。

## 术语

- **SFT**：用示范轨迹做监督微调，教模型照着最短走法输出动作。
- **RL**：在环境里让模型自己走轨迹、按奖励更新模型。本线用主线的标准做法（第十一轮修正包）。
- **采样（rollout）**：RL 训练中让模型在环境里生成轨迹的那一步。现在由 vLLM 执行，vLLM 是专门做快速生成的推理引擎。
- **HF generate**：用 Hugging Face Transformers 库自带的生成函数逐字生成。它慢，但和训练时算概率的代码是同一套。
- **"一道题一个序列"（concat）**：RL 时把一道题的所有回合（每回合的画面、反馈和模型的回答）拼成一个序列，和 SFT 的多回合对话一样。主线第十轮起都用这个。
- **"每回合一个样本"（no_concat）**：每个回合单独作一个训练样本。8 月 Cambrian 和 10 月 2 日 Qwen3-VL 的冒烟都走这条路。
- **S0**：稀疏奖励，只在成功时给 +5。**D**：密集奖励，成功之前每步按离目标的进度给分。
- **D-cond**：D 的一个版本，进度 = 0.25 × 画面组 + 0.25 × 看清组 + 0.5 × 关系组；关系组只在目标进了画面时才算（主线决定 41）。
- **keep**：组内比较时，把 8 条轨迹全失败、但回报差别够大的题也补进更新（主线的 `keep_by_return_std`）。
- **DCK**：D-cond + keep。**V1–V8**：主线在 DCK 上每次只改一处的 8 个设计变体（主线决定 57）。
- **读不出动作**：环境的严格读法从回答里读不出可执行的动作。
- **卡·小时**：占用的 H800 卡数 × 小时数。
- **起点**：下游训练开始时用的视觉语言模型权重。

## 一、权重

### 1.1 找到的

| 模型 | 路径 | 版本 | 大小 | 完整吗 |
| --- | --- | --- | --- | --- |
| Qwen3-VL-8B-Instruct | `/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | HF 修订 `0c351dd0`，8 月 4–5 日下进共享缓存 | 4 个分片共 17,534,339,512 字节（16.3 GiB），bf16 | 4 个分片都在，没有下载中断留下的残件；1 卡测试完整加载，没有缺失或多余的权重键 |
| 8 月 RL 训过的 Cambrian 检查点（4 个） | `VAGEN-Lite/models/historical/cambrian_c8_clean_d0pass_20260824_sco_renderer_r3_full_step{50,100,1000}`、`cambrian_c4_clean_d0pass_20260822_h800_r5_full_step960` | 架构 `cambrian_qwen`，从 Cambrian-S-7B-LFP 出发、按 8 月的旧题和旧奖励训的 RL | 每个 7 个分片 32,189,419,408 字节（按大小是每个参数 4 字节，即 fp32 存储；config 写的计算精度是 bf16） | 完整，但**不是原始权重**，不能当本线的起点；只用来测代码 |
| Cambrian-S 的外部源码 | `/mnt/umm/users/yinbaiqiao/cambrian-s` | 不是 git 仓库，7 月 23 日放进来，`pyproject` 版本 1.0.0 | — | 适配器 `import cambrian` 用它；今天测试能导入 |

- Qwen3-VL-8B 的结构（`config.json`）：语言部分 36 层、隐藏维度 4096、bf16；视觉部分 27 层、图块 16、合并 2，另有 3 层的多层视觉特征注入（deepstack，第 8、16、24 层）。
  - 图像预处理的最小像素数是 65,536，正好是环境画面 256 × 256，所以每帧 16 × 16 个图块、合并后 64 个 token（测试实测也是 64）。
- Qwen3-VL 的 4 个分片和索引文件属主是 root、权限只有属主可读（600）。登录节点上用你的账号读不了；sco 任务以 root 运行，读得了。
  - 所以 sha256 没能在登录节点上核对。HF 缓存的文件名就是 sha256，可以放进第二步的第一个 root 任务里顺手核对，约 3 分钟。
  - 这也意味着以后凡是在登录节点上读这份权重的工具（比如下游的起点检查脚本），都要放进 sco 任务里跑。
- 其他相关但不是本线要的：
  - StarVLA 的 `Qwen3-VL-4B-Instruct-Action`（`~/.cache/huggingface/hub`），是 StarVLA 训过动作的 4B，不是原始模型；
  - 共享目录有 `Qwen3.6-27B`、`Qwen2.5-VL-7B-Instruct`。

### 1.2 找不到的（要你从集群外拷进来）

| 要的东西 | HF 仓库（修订号） | 大小 | 为什么要 |
| --- | --- | --- | --- |
| Cambrian-S-7B | `nyu-visionx/Cambrian-S-7B`（`556e72fd`） | 32.13 GB（7 个分片；按大小是 fp32 存储） | 本线的 Cambrian 起点 |
| SigLIP2 视觉编码器 | `google/siglip2-so400m-patch14-384`（`e8e48729`） | 4.58 GB | Cambrian 的代码建模型时用 `from_pretrained` 读它的配置和权重，图像预处理也读它的 `preprocessor_config.json`；没有它，Transformers 那条路建不起模型（第六节） |
| （可选）Cambrian-S-7B-LFP | `nyu-visionx/Cambrian-S-7B-LFP`（`d04f6984`） | 32.19 GB（单个文件） | 8 月的 RL 从它出发。它多一个"预测下一帧"的辅助头，本线的单张图像任务用不上 |
| （不需要）0.5B / 1.5B / 3B | `nyu-visionx/Cambrian-S-{0.5B,1.5B,3B}` | 3.58 / 7.79 / 13.97 GB | 只在你要小尺寸对照时才要 |

- 这些大小是今天从 HF 的接口读的元数据，没有下载任何权重文件。
- 它们以前都在：`/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP`、`hf_cache/cambrian-s-7b`、`~/.cache/huggingface/hub` 里的两个 SigLIP。现在只剩空目录和 7 月 24 日 root 留下的锁文件。
- AOSS 上的归档（`fj:s3://baiqiao/`）里也没有：只有 `InsScene-15K`、`InteriorGS`、`VAGEN-Lite/exps`、`vagen_lite_archive`、`probe_spatial` 等。谁删的、什么时候删的，没查到记录。
- 共享缓存里 `models--nyu-visionx--Cambrian-S-7B` 只剩一个修订号文件（`556e72fd`，7 月 23 日），没有快照和数据块。
- **集群能不能自己下**：登录节点经代理能连 PyPI、GitHub 和 huggingface.co，不经代理能连 hf-mirror.com（10-18 实测约 30–36 MB/s，约 20 分钟下完 37 GB）。按约束，下载大文件先问你，见"需要你决定的事"第 1 项。

## 二、代码：两个模型以前调通到哪一步

| 模型 | SFT | RL 采样 | 评测 | 下游 | 当时的库 | 现在 main 上 |
| --- | --- | --- | --- | --- | --- | --- |
| Cambrian-S | **从没做过**（7 月的计划里列过"先 SFT 建立格式先验"，没做） | **vLLM**：仓库自己写的 vLLM 模型类 `vagen/models/cambrian_vllm.py` + 插件 `cambrian_plugin.py`（`setup.py` 的 `vllm.general_plugins` 入口，主环境里已注册）；训练端是 `cambrian_register.py` 的适配器 | 训练内验证集；8 月 2–3 日用 EASI 评过原始的 7B-LFP（静态问答）；导航的离线评测缺（记为 missing） | 从没做过 | torch 2.8.0、transformers 4.56.1（检查点的 config 写的）、vLLM 0.11.0（D0 审计记的） | 适配器 5 个文件都在 `vagen/models/`，最后改动 8 月 31 日（`5c582b93`）；`agent_loop_no_concat.py` 里有 Cambrian 分支 |
| Qwen3-VL | **从没做过**（LLaMA-Factory 没有模板，`scripts/prepare_active_spatial_sft.py` 遇到非 Qwen2.5-VL 直接拒绝） | **vLLM**：10 月 2 日 8 卡 1 次更新的冒烟通过（`qwen3vl_active_spatial_smoke_h800_20261002_metricfix`：梯度范数 13.30、策略损失 0.068 有限、检查点完整），走"每回合一个样本"和旧的 v50 环境配置 | 没在 v10 的评测集上评过 | StarVLA 用过 Qwen3-VL-4B-Action（10 月 1 日 NavigateKitchen 训了 10 万步），8B 没试过 | `.venv-qwen3vl-runtime`：主环境加 transformers 4.57.6，其余同主环境 | `agent_loop_no_concat.py` 里有 Qwen3-VL 分支；冒烟脚本和 `scripts/check_qwen3_vl_env.py` 在 |

**8 月 Cambrian 那条路的细节**（`docs/diagnosis/D0_CAMBRIAN_PLUMBING_AUDIT.md`，在未合并的整理分支 `cleanup/20261016` 的 `archive/history/` 下）：

- D0 审计查出过几处训练端和 vLLM 端不一致：视觉编码器的实现、多图特征、注意力的数值精度。都修了，修法见审计的 D0.2–D0.11。
- 剩下的是 vLLM 和 HF 算出的概率有差别，主要在自由思考的文字上。D0.16 的结论是靠 verl 自带的"按采样概率修正"来处理（`rollout_is=token`，阈值 2.0）。
- 之后正式跑了 1000 次更新（`cambrian_c8_clean_d0pass_20260824_sco_renderer_r3_full`）。
- 主线现在的配方**不开**这个修正（`calculate_log_probs: false`，第四节）。
- 动作 token 上的差别：这里比的是 HF 和 vLLM 给同一个 token 的概率之比，看比值偏离 1 超过 5% 的 token 占多少。
  - Qwen2.5-VL：0%（D0.17，动作名 36 个 token）。
  - Cambrian：10%（D0.16，动作名 30 个 token）；思考文字是 44%。
  - 样本都很小。本线是纯动作的回答，没有思考文字，第二步要在纯动作的回答上重测（见第八节）。

**代码整理有没有删掉或移动它们**：

- main 上（`00363333`）什么都没动。
- 整理分支 `cleanup/20261016`（10-16，10 个提交，没合并）把 3 份旁支模型文档、Qwen3-VL 冒烟脚本和 `check_qwen3_vl_env.py` 移进了 `archive/side_models/`。`vagen/models/` 和两个 agent loop 没动。
- main 上没看到整理完成的公告，所以按"代码整理期间"处理：不推 main。
- 你说的 `docs/STATUS.md`、`configs/matrix_r12.json`、`archive/side_models/` 在 main 上不在这些位置：
  - 现在的状态文档是仓库根目录的 `STATUS.md`（10-15 的整理状态）和 `data_gen/interiorgs_v10/STATUS.md`（主线），矩阵登记在 `data_gen/interiorgs_v10/configs/matrix_r12.json`；
  - `archive/side_models/` 只在整理分支上。
  - 本报告按这些位置读的。

**接到主线现在的路径上缺什么**（主线 RL 用 `vagen.agent_loop.gym_agent_loop.GymAgentLoop`，即"一道题一个序列"）：

- GymAgentLoop 本身不分模型，图像和位置编码交给 verl 的 `verl/experimental/agent_loop/agent_loop.py`。
- 那里只按"图像处理器的类名里有没有 `Qwen2VLImageProcessor`"判断，然后一律用 Qwen2-VL 的位置编码函数。
  - Qwen3-VL 的图像处理器叫 `Qwen2VLImageProcessorFast`（今天测出来的），所以会走 Qwen2-VL 的函数。
  - verl 自己的数据集代码（`rl_dataset.py`）对 Qwen3-VL 是分开处理的，agent loop 这里没分。
  - 对单张图像，两个函数给的位置是否逐位相同，要第二步用测试核对；不同就要在模型适配里改，不能假设。
- Cambrian 在这条路径上完全没接：
  - 没有把 `<image>` 展开成 756 个 -200 占位、给训练端的 `pixel_values`；
  - 画面在回复里（后面回合的画面）的掩码也没处理。
  - 这些在 `agent_loop_no_concat.py` 里有现成写法可以照搬。
- 主线评测（`evaluation.py` 的 `VllmPolicy`）用 vLLM 的 `chat` 接口：
  - Qwen3-VL 只要换成新环境的 Python；
  - Cambrian 要 `trust_remote_code` 和插件，`<image>` 占位符在 vLLM 模型类里有定义，没试过。

## 三、环境

| 环境 | Python | torch | transformers | vLLM | 其他 | 现在给谁用 |
| --- | --- | --- | --- | --- | --- | --- |
| `.conda/envs/vagen-lite`（主环境） | 3.12.13 | 2.8.0 | 4.56.1 | 0.11.0 | verl 0.8.0.dev0（子模块 `a0ae85c`）、llamafactory 0.9.6.dev0（可编辑安装，指向主 checkout 的 `third_party/LLaMA-Factory`，提交 `ca75f1e`）、accelerate 1.14.0、peft 0.19.1、datasets 5.0.0、ray 2.56.1、无 flash-attn | 主线 SFT、RL、评测 |
| `VAGEN-Lite/.venv-qwen3vl-runtime` | 3.12.13 | 2.8.0 | **4.57.6** | 0.11.0 | 继承主环境的全部包（`--system-site-packages`），只多一个 transformers | 10-02 的 Qwen3-VL 冒烟；今天的测试 |
| `VAGEN-Lite/.venv-qwen3vl` | 3.12.13 | 2.13.0 | 5.17.0 | 0.30.0 | — | 迁移文档说这版 transformers 和 verl 不兼容，不用 |
| `.conda/envs/starVLA` | 3.10.21 | 2.6.0+cu124 | 4.57.0 | — | flash-attn 2.7.4、deepspeed 0.16.9、starVLA 1.0.1 | 下游 StarVLA 训练和评测 |
| `envs/rlinf-starvla-rc365` | 3.10.21 | 2.6.0+cu124 | 4.57.6 | — | 同上加 RLinf | 下游 RLinf |

**每个模型要什么**：

- **Qwen3-VL**：
  - transformers ≥ 4.57（Qwen3-VL 从 4.57 起才有）；vLLM 0.11.0 今天实测能用。
  - SFT 要有 Qwen3-VL 模板的 LLaMA-Factory：本地 `ca75f1e`（6 月）没有；上游 main 和 PyPI 最新的 0.9.5 有 `qwen3_vl` 模板和插件。
  - 但 0.9.5 要求 transformers ≥4.55 且 ≠4.57.0、accelerate ≤1.11、datasets ≤4.0、peft ≤0.18.1，和主环境冲突，只能单独建环境。
- **Cambrian-S**：
  - 外部源码自己声明的是 transformers 4.37；仓库的适配器 8 月在 4.56.1 上跑通过 1000 次更新，今天在同一环境里 vLLM 那条也跑通（第六节）。
  - 不需要新环境，但缺 SigLIP2 文件。
  - 官方训练代码以 TPU 为主（`cambrian/train/cambrian_trainer.py` 直接 `import torch_xla`），GPU 上的 SFT 要我们自己写。

**集群上 pip 能不能装**：

- 能连：登录节点经默认代理能访问 PyPI（今天测 0.6 秒返回）和 GitHub。
- 以前也装过：下游线 10-20 建了 `rlinf-starvla-rc365`；评测线 `eval_external_assets/wheels/` 里有为 torch 2.8 编好的 flash-attn 2.8.3 轮子。
- 本步没有安装任何东西，也没改任何现有环境。

## 四、最高 setting 的每一处路径

| 项目 | 路径 / 设置 | 在不在 main | 备注 |
| --- | --- | --- | --- |
| C0 的 SFT 数据 | `E/interiorgs_v10_r11_20261016/sft/sft_trajectories.jsonl`：11,095 段正式题对话，train 9,802 / val 1,293，带完整历史，1,169 段插入了"被挡住"回合；帧在同目录 `frames/plain/`（256 × 256） | 数据在 | 旧格式（格式示例是占位符 `action1\|action2\|...`）；矩阵登记的 P2 就是它 |
| 新格式导出工具 | `training_prep sft-records --dir DIR`（`release.build_sft_record`，按 `configs/admission_v2.json`，`format_example` 现在默认 `action_names`，`ea59b987`） | 在 | 它把结果写进 `DIR` 本身，所以要另建一个目录，把 r11 的 `rows.jsonl`、`frames.jsonl`、`frames/` 链接进去再导出，不能在 r11 目录里跑。另一个办法是 `format-sft` 用的逐字替换（`training_prep.format_sft_records` 的 `rewrite`）。第二步两种都做，核对两者逐字相同、和旧数据只差格式文字 |
| SFT 配方 | `scripts/prepare_active_spatial_sft.py` 生成，实例 `E/interiorgs_v10_train_r11_20261016/lf_formal/train.yaml`：全参数，冻结视觉编码器、不冻投影层，模板 `qwen2_vl`，`cutoff_len` 16384，每卡 1 × 梯度累积 8 × 8 卡 = 64，学习率 1e-5，10 个 epoch，余弦，预热 3%，bf16，梯度检查点，sdpa，每个 epoch 存一次；accelerate FSDP 配置 `examples/train/active_spatial/accelerate_fsdp_qwen25vl.yaml` | 在 | 脚本只认 Qwen2.5-VL；FSDP 配置写死 `Qwen2_5_VLDecoderLayer` 等层名。两处都要按模型补新文件 |
| epoch 选择规则 | 只看验证集，评测脚本单卡、不采样，打平取更早（第十一轮决定 2、第十二轮决定 6、10） | 在 | C0 当时用 v1 验证集（398 道）挑出第 9 个 epoch；按决定 10，本线用 v2 验证集（413 道） |
| 标准 RL 配置 | `data_gen/interiorgs_v10/configs/rl_standards.json`：`v10_p1_S0`、`v10_p1_Dcond`（修正包 p1：GRPO、只用有成有败的题、每题 8 条、24 道候选、12 道目标、学习率 2e-6、KL 系数 0.05、熵 0.01、一道题一个序列、后备题、每 50 次存检查点）；`chain_defaults`：32 个渲染进程、渲染先、渲染服务搬家、10% 读不出线、启动重提 5 次 | 在（`2d46d971`、`28f195a4`） | PPO 模板 `configs/ppo_template_v10.yaml`，环境模板 `configs/rl_env_v10_S0.yaml`（12 个动作上限、256 × 256 画面） |
| 停止线 | 标准的默认：KL 0.2、读不出 10%（10 次平均）、成功率下降 0.1（50 次）；你的简报写 KL 1.0 | 在 | C0 和 V4 都是 0.2，KL 实际只到约 0.016。见"需要你决定的事"第 6 项 |
| 后备题 | `E/interiorgs_v10_r11_20261016/rl_rows/rows.jsonl`（train 场景的后备题 10,798 道），排除 `E/interiorgs_v10_r10_20261015/depth_check/depth_check_summary.json` | 在 | `rl-config --task-pool reserve` 会核对和 SFT 对话不重合 |
| 种子 | 任务顺序 20261019，vLLM 采样种子 0（这版 verl 里固定） | — | — |
| S0 臂 | `v10_p1_S0` | 在 | — |
| 密集臂（现在应选 V4） | `v10_p1_Dcond` + `group_filter: keep_by_return_std`（门槛 0.3318、下限 1.0）+ `progress_reward_scale: 3.333`（V4 的 plan，`E/interiorgs_v10_train_r12t_20261023/plan.json` 的 `dv4`） | **不全在**：D-cond 和 keep 在 main；`progress_reward_scale` 这个 rl-config 选项只在 dense-batch2 | 见下面"密集变体" |
| dense-batch2 的提交 | 副本 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-r12dense` 的本地分支，在 `00363333` 上多 6 个：`aa6451e8`（读不出的回合不参与正向更新）、`ec7682ec`（同上的记录方式）、`991cc92d`（V1–V6 的环境开关）、`f8ba48c5`（V8）、`24f1272e`（rl-config 接上设计变体，含 `--progress-reward-scale`）、`418927c4`（弱起点稳定性的三个选项） | 不在 main | 6 个都是默认关的开关。单独拣任何一个都和前面的冲突（逐个 `git apply --check` 过），整段按顺序拣能干净应用。本机副本里同样的 6 个提交号不同（`af007373`…`5d671ead`），内容树完全相同（`d594949c`） |
| 评测集 | `E/interiorgs_v10_r12_20261019/eval_val_v2/tasks.jsonl`（413 道）、`eval_test_v2/tasks.jsonl`（292 道），合并 705 道 | 在 | 结果目录按评测集共用、按结果名区分，同名会被跳过（主线 10-23 的教训）。本线的结果名都带模型前缀，提交前查重名 |
| 终点指标 | `evaluation terminal --eval DIR`（`de293023`）：失败题终点进度分（乘积版）、按族的满足条件比例、碰撞率 | 在 | 新的分组进度分（D-cond 版）在 main 的 `evaluation.py` 里没有，是密集奖励线自己的分析脚本算的（`E/interiorgs_v10_r12_20261019/scripts/variants_results_r12.py`、`c0_paired_r12.py`）。第三步只读地复用 |

**密集变体：现在能看到的最新汇总**

- 文件：`E/interiorgs_v10_r12_20261019/analysis/variants/variants.json`，10-24 15:04 UTC 写的。
- 已有 V1–V6 和 V8，V7 还没进表（它的训练链 `r12t` 的 RL 一步 17:11 完成，链的整体状态 17:49 仍是"进行中"）。
- V4 的第二个种子（`r12y`，种子 20261021）在训；组合变体 18:18 刚提交。

| 变体 | 改的那一处 | 挑出的检查点 | 合并 705 道 | 验证集 / 评测集 | 失败题终点（分组进度分） | 满足条件 | 碰撞率 | 过挑选规则 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **V4** | 进度系数 ÷3 | 第 250 次 | **48.2%** | 49.9% / 45.9% | 0.334 | 79.3% | 10.1% | 是 |
| V1 | 分组权重 0.15/0.15/0.70 | 第 300 次 | 46.0% | 47.9% / 43.2% | 0.360 | 79.5% | 9.8% | 是 |
| V5 | 进度系数 ×3 | 第 300 次 | 45.8% | 47.7% / 43.2% | 0.367 | 79.8% | 12.7% | 是 |
| V8 | 成功和进度分开归一化 | 第 300 次 | 45.8% | 47.7% / 43.2% | 0.354 | 79.1% | 11.4% | 是 |
| V6 | 里程碑 | 第 150 次 | 44.1% | 46.0% / 41.4% | 0.308 | 76.4% | 8.1% | 否 |
| V3 | 软门槛 | 第 100 次 | 43.3% | 45.5% / 40.1% | 0.334 | 77.2% | 10.7% | 否 |
| V2 | 分组权重各 1/3 | 第 200 次 | 42.6% | 45.0% / 39.0% | 0.328 | 76.6% | 13.6% | 否 |
| 对照 DCK（种子 1） | — | — | 44.8% | 45.3% / 44.2% | 0.330 | 78.3% | 8.6% | — |
| 对照第一对 S0 | — | 第 250 次 | 45.8% | 48.4% / 42.1% | 0.299 | 75.9% | 10.0% | — |

- "过挑选规则"是主线的三条：成功率不低于 DCK、失败题终点更近、满足条件更多。
- V4 对 DCK 逐题比较：只有 V4 做成 90 道，只有 DCK 做成 66 道，p = 0.065，没到 0.05。
- 所以现在的选择是 **V4，单种子，汇总日期 10-24 15:04**。提交 RL 时再读一次：那时如果 V4 的第二个种子或组合变体出来了，按"两集合并最高、过规则、有两个种子的优先"重选，写进报告。

## 五、下游

**StarVLA 支持哪些骨干**：

- `starVLA/model/modules/vlm/__init__.py` 先按权重 `config.json` 的 `model_type` 选：`qwen2_5_vl`、`qwen3_vl`。
- 再按名字认：Qwen3.5、Gemma-4、Molmo2、MiniCPM-V、Florence-2、Cosmos-Reason2、EgoVLA（VILA）。
- 本线要用的结构 QwenOFT：在提示后放 16 个占位符，取最后一层的隐藏状态，经 MLP 动作头回归 16 步 × 12 维动作。它只用 Qwen2.5-VL / Qwen3-VL 的接口，动作头的输入维度取骨干的 `hidden_size`。

**下游线的转换和登记**（只读）：

- 下游线说"不需要转换"：起点是完整的 HF 权重，StarVLA 按路径直接读。
- `scripts/prepare_starvla_transfer.py` 的"同族同尺寸"检查，比的是同一次准备里几个起点的 `(model_type, hidden_size)` 是否一致，不和某个固定尺寸比。
- 起点登记在 `downstream/starts.json` 和 `starts_lock.json`，建实验的 `downstream/train/make_experiment.py` 写死从这两个文件读。
- 训练入口 `downstream/train/train_entry.py` 只给 Qwen2.5-VL 打了"按 bf16 加载"的补丁（下游决定 D1）。
- RLinf 的 `downstream/rlinf/fsdp_7b.overrides` 写死按 `Qwen2_5_VLDecoderLayer`、`Qwen2_5_VLVisionBlock` 切分 FSDP。

**对两个模型意味着什么**：

| | Qwen3-VL-8B | Cambrian-S |
| --- | --- | --- |
| StarVLA 接口 | 有（`QWen3.py`，`Qwen3VLForConditionalGeneration`）；starVLA 环境 transformers 4.57.0 能加载 | **没有**。它的视觉编码器（SigLIP2）、投影层、图像换行 token 和 `cambrian_qwen` 结构都不在 StarVLA 里 |
| 同族同尺寸 | 本线的 4 个起点都是 `(qwen3_vl, 4096)`，可以一起准备；和下游的 Qwen2.5-VL（3584）不同，各自分开 | — |
| 起点登记 | 不能改下游线的文件：要一个新文件 `model_matrix/downstream/starts.json` 和一个薄包装，复用它的 `build_config` 等函数 | — |
| bf16 加载 | Qwen3-VL 的 `config.json` 自带 `bfloat16`；本线 SFT / RL 存出的权重要核对，必要时在新的入口里给 `Qwen3VLForConditionalGeneration` 补同样的补丁 | — |
| RLinf | `rlinf/config.py` 已注册 `qwen3_vl`；FSDP 层名要换成 `Qwen3VLTextDecoderLayer`、`Qwen3VLVisionBlock`（transformers 4.57.6 里的类名，已查），新写一个覆盖文件 | 同样没有 |
| 评测、权重复制、报告 | `run_eval_ckpt.sh`、`extract_weights.py`、`downstream/report/` 不分骨干，按路径只读调用 | — |
| 工作量 | 约 1 天，都是新文件，加 1 卡 StarVLA 加载测试（第二步） | 要写一个 StarVLA 骨干接口，估计 3–5 天，还没算 RLinf 那边。按你的规定停在这里，见"需要你决定的事"第 5 项 |

**下游起点代号**（按下游登记表的写法，B / S / RS / RD 前加模型）：

- Qwen3-VL：`Q3B`（原始）、`Q3S`（SFT）、`Q3RS`（RL 稀疏臂）、`Q3RD`（RL 密集臂）；
- Cambrian：`CSB`、`CSS`、`CSRS`、`CSRD`；
- 实验编号 `DS-<起点>-100-s1`，例如 `DS-Q3B-100-s1`；
- Qwen2.5-VL 的参照行沿用下游线 E1 的 DS-{B,S,RS,RD}-100-s1，不重跑，注明它的起点是旧格式训的。

## 六、单卡加载测试

**做法**：

- 一个 1 卡 sco 任务里分 4 段，每段一个新进程（卡从空开始）。
- 任务 `pt-kqhed0xd`，h800，18:19:44–18:28:05。vLLM 两段因为我的测试脚本没设 C 编译器（Triton 启动时要编译），又补跑了一个任务 `pt-gm3kn6l8`，h800，18:29:07–18:34:59。
- 输入是一道 v2 验证集的题（`sft_v10_c269344785dd21218e7a415e`，absolute_positioning，最少 2 步），来自格式 SFT v2 的新格式对话（格式示例是真实动作名）：系统提示 + 第一个用户回合 + 第一帧画面（256 × 256），参考答案 `<action>move_right|</action>`。
- 每个模型出一个贪心回答，再按 RL 的采样方式（温度 0.8、top-p 0.92）采 4 个，都用环境的严格读法读。
- 脚本 `model_matrix/step1/load_test.py`、`run_load_test.sh`；结果在 `/mnt/umm/users/yinbaiqiao/models_runs/step1_load_test_20261024{,_vllm}/`。

| 模型 / 路径 | 结果 | 加载 | 显存 | 提示 token（其中图像） | 贪心回答 | 4 次采样里读得出的 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B，Transformers 4.57.6 | **通过** | 52 秒 | 加载后 16.9 GiB，生成后 17.9 GiB | 672（64） | `<action>move_forward\|</action>` | 4 / 4（都是同一个回答） |
| Qwen3-VL-8B，vLLM 0.11.0（补跑） | **通过** | 183 秒（含 vLLM 预留，`gpu_memory_utilization` 0.5） | 36.7 GiB（预留） | 672（vLLM 自己算的也是 672） | `<action>move_forward\|</action>`，和 Transformers 一样 | 4 / 4 |
| Cambrian（8 月 RL 检查点 c8 第 50 次），Transformers 4.56.1 | **失败**：建模型时 Cambrian 的视觉编码器去读 `google/siglip2-so400m-patch14-384`，缓存里没有 | — | — | 紧凑 607 / 展开 1,362（756） | — | — |
| 同上，vLLM 0.11.0，原样（第一个任务） | **失败**：没有 SigLIP2 的预处理配置时，代码的后备处理器按默认 224 × 224 出图（256 个图块），模型要 729 个，启动试跑报尺寸不符 | — | — | — | — | — |
| 同上，vLLM 0.11.0，用替代的预处理（补跑） | **能输出** | 78 秒 | 47.1 GiB（预留，0.6） | 1,362（756） | `<action>move_forward\|turn_left\|</action>` | 2 / 4（另两个是自由文字和 `move FORWARD 3.0 meters` 这类） |

- **替代的预处理**：画面已先缩放到 384 × 384，SigLIP2 剩下的只有除以 255 和均值、方差都取 0.5 的归一化；`SiglipImageProcessor` 的默认值相同。它只为测通后面的路径，结果里标明了，不能当正式做法。
- Qwen3-VL 原始模型零样本就按格式回答，动作名都在动作表里。这一道答错了方向（参考答案是右移），这里不评成功率。
- Cambrian 用的是 RL 训过的检查点，不是原始权重。它一半的采样读不出，和 7–8 月记录的"Cambrian 格式先验弱"一致；原始 Cambrian-S 的格式表现要等权重进来再测。
- 最大显存（每 2 秒一次的 `nvidia-smi` 记录）：第一个任务 20.2 GiB，补跑 36.7 GiB。
- **读法的一处更正**：任务里的读法没给动作表，编造的名字（如 `look_down`）也会算读得出。补的测试抓到了这一点，现已改成和环境一样按 `rl_env_v10_S0.yaml` 的 `strafe` 动作表读（6 个动作）。用改好的读法把存下的全部回答重读一遍，判断一个都没变：读得出的回答里只有 `move_forward` 和 `turn_left`。

## 七、估算与提交顺序

**依据**：

- **实测**（主线，Qwen2.5-VL-7B）：
  - C0 的 SFT 10 个 epoch 在 8 卡上训练 1 小时 44 分（`train_runtime` 6,248 秒），任务从开始到结束 1.85 小时；
  - 评测 1 卡：验证集 413 道 26–28 分钟、评测集 292 道 19–21 分钟（V4 的结果目录 `done.json`）；
  - RL：V4 300 次更新，每次平均 79.8 秒（采样 24.6 秒、更新 27.4 秒），训练任务 6.96 小时（8 卡）加 1 卡渲染服务；
  - 每回合序列：回复平均 1,668 个 token、最多 2,769 个，提示约 695 个。
- **下游**：每步约 0.75 秒（下游线的估计，它注明未核实），20 万步约 42 小时（8 卡）；一次整机评测约 3 小时（8 卡）。
- **换算到新模型**：
  - Qwen3-VL-8B 参数是 Qwen2.5-VL-7B 的 1.06 倍，每帧图像 token 是 0.79 倍，按 1.0–1.1 倍估。
  - Cambrian 每帧 756 个 token，是 Qwen2.5-VL 的 9.3 倍。SFT 每段对话的 token 约 3 倍（平均 2.6 个回合），RL 每回合序列约 4.5 倍（回复平均约 7,500、最多约 11,700 个 token）。它的配置用 eager 注意力，主环境没有 flash-attn，所以按 3–6 倍估，区间大。

**每个模型**（卡·小时）：

| 项目 | Qwen2.5-VL 参照行 | Qwen3-VL-8B | Cambrian-S-7B | 依据 |
| --- | --- | --- | --- | --- |
| 基座零样本评测（705 道，温度 0.8，1 卡） | 0.8 | 0.8 | 2 | 实测换算 / 估算 |
| SFT（10 个 epoch，8 卡）+ 每个 epoch 评验证集 + 挑出的评评测集 | 19 | 20–22 | 55–70 | 实测 / 估算 / 估算 |
| SFT 验收（200 道训练题 × 4 次，温度 0.8） | 0.9 | 0.9–1 | 2 | 估算 |
| RL 一个臂（300 次，8 + 1 卡，含 6 个检查点各评两集） | 68 | 70–75 | 260–390 | 实测 / 估算 / 估算 |
| RL 两个臂 | 136 | 140–150 | 520–780 | — |
| **主实验合计** | **约 160** | **约 165–175** | **约 580–850** | — |
| 下游一个起点（20 万步 + 3 个评测点） | 不重跑 | 约 430 | 不能做 | 下游线数字换算，估算 |
| 下游 4 个起点 | — | **约 1,700** | — | 估算 |

- Cambrian 那一列等第二步冒烟实测后会大幅收窄。
- 冒烟（第二步）每个模型约 15 卡·小时（估算）：小规模 SFT、10 次 RL、两次评测。

**墙钟时间**（卡够时）：

- Qwen3-VL 主实验约 12–14 小时：SFT 约 2.5 小时、逐 epoch 评测约 1 小时、两个 RL 臂并行约 8 小时、评测约 1 小时。
- 下游一个起点约 45 小时。

**提交顺序**（建议）：

1. 第二步：先 Qwen3-VL 的冒烟和 StarVLA 1 卡加载测试。通过后提下游 `DS-Q3B-100-s1`（原始 Qwen3-VL，不等主实验）。Cambrian 等权重和代码。
2. 第三步：
   - Qwen3-VL：SFT → 验收 → 两个 RL 臂同时 → 评测；
   - Cambrian：同样的顺序，在 Qwen3-VL 之后；
   - Qwen2.5-VL 参照行在两个新模型都提交之后。
3. 第四步：`DS-Q3S`、`DS-Q3RS`、`DS-Q3RD` 在各自检查点出来后提交。

**磁盘**：

- 用户目录配额 22 TB，10-24 读数已用 19 TB，剩约 3 TB。主线守护进程在 19 TB 以上就记警告。
- 主线一个 RL 检查点：只留权重的 31 GB，带优化器状态、能接着训的 124 GB（V4 的第 250、300 次）。一个臂训练中每 50 次存一个，按主线的 `retain: best_and_last` 到结束才删，峰值约 6 × 124 GB ≈ 0.75 TB。
- 所以本线同一时间只跑一个模型的一对 RL（峰值约 1.5 TB），每对结束、评完再按 `best_and_last` 清。Cambrian 和 Qwen3-VL 的检查点大小与此相近（参数量相近）。
- 下游一个起点训练中只留最新一个全状态（约 110 GiB）。

**GPU 规则**：

- bq 前缀的任务最高，不动、不问；主线的密集奖励和 A 组在前；本线和下游同级，谁先排队谁先上；主线的暂停文件 `downstream_runs/PAUSE_SUBMISSIONS` 本线同样遵守。
- 今天两次提交前都查过：暂停文件不在；zoetrope 上主线有 4 个 1 卡渲染服务在等卡，所以只投 h800。h800 上主线等的是 3 个 8 卡训练，要整台空机；1 卡任务在 h800 当场开始，落在已有任务的机器上，不占整机。

## 八、对第二步的影响（计划，等你确认）

1. 模型类型做成配置里的一个字段（默认 Qwen2.5-VL，默认路径输出逐位不变）。差别只放在模型适配里：
   - 图像处理：Qwen3-VL 64 个 token / 帧；Cambrian 756 个，SigLIP2 预处理；
   - 对话模板：Qwen3-VL 用它自己的；Cambrian 是 Qwen2 的 `<|im_start|>` 格式；
   - 停止符：两者都是 `<|im_end|>`；
   - 最大长度：Cambrian 的回复上限要从 4096 提到约 12,000。
2. Qwen3-VL 的位置编码：
   - 写测试核对 Qwen2-VL 和 Qwen3-VL 的位置函数在单张图像输入下是否逐位相同；
   - 不同就在模型适配里按处理器类型选函数（新开关，不改 verl 子模块的默认行为）。
3. Cambrian 的采样按你的第 6 条：
   - 以前的 vLLM 实现在、在今天的代码上能跑（用替代预处理测的），所以用它，不重写；
   - 补的是"一道题一个序列"路径上的 Cambrian 分支（新文件、新开关）。SigLIP2 文件进来后，那个出 224 × 224 的后备处理器就用不上了；它的问题记在这里，不改现有文件；
   - 验证：同一组输入下 vLLM 和 HF generate 的贪心输出逐字一致；动作 token 的概率差，按 D0 审计的口径比，比例差超过 5% 的 token 占多少。
   - 差别大就停下来问你要不要开采样概率修正（会让 Cambrian 的配方和另外两个不同）。
4. 每个功能补测试，全部测试通过再提交到本地分支 `models`。

## 需要你决定的事

1. **Cambrian-S 的权重从哪来**（不定这个，Cambrian 的第二步做不了）。
   - **甲（推荐）**：拷 / 下 Cambrian-S-7B（32.13 GB，修订 `556e72fd`）和 SigLIP2（4.58 GB，修订 `e8e48729`），放到新目录 `/mnt/umm/users/yinbaiqiao/models_ext/`，只读使用。
     - 你从集群外拷；或者你同意后我从登录节点经 hf-mirror 下，约 20 分钟。
     - 用户目录配额 22 TB，今天读数已用 19 TB；多 37 GB 没问题（磁盘的整体约束见第七节）。
   - **乙**：用 Cambrian-S-7B-LFP（32.19 GB）加 SigLIP2，和 8 月的 RL 同一个起点。代价是多一个本线用不上的"预测下一帧"头。
   - **丙**：这一轮只做 Qwen3-VL，Cambrian 不做。省下约 580–850 卡·小时和 4–5 天代码。
2. **Qwen3-VL 的 SFT 用哪个 LLaMA-Factory**。
   - **甲（推荐）**：在现有的 LLaMA-Factory（`ca75f1e`，C0 用的同一份代码）里，用本线自己的新文件在运行时登记 Qwen3-VL 的模板、图像插件和"冻结视觉编码器"规则，在 `.venv-qwen3vl-runtime` 里跑。
     - 不安装任何东西，约 1 天，含测试：LLaMA-Factory 拼出的对话 token 和 Qwen3-VL 自己的对话模板逐位相同。
     - 风险：`ca75f1e` 配 transformers 4.57.6 没人跑过，冒烟时才知道。
   - **乙**：单独新建一个环境装 LLaMA-Factory 0.9.5。约半天，要你同意安装；它的训练代码和 C0 用的那份不同。
3. **Qwen3-VL 的 RL 和评测用哪个 Python**。
   - **甲（推荐）**：用现有的 `.venv-qwen3vl-runtime`（主环境 + transformers 4.57.6；10-02 的 RL 冒烟和今天的测试都用它）。任务脚本加一个选 Python 的新开关。
     - 它在主 checkout 目录里（`VAGEN-Lite/.venv-qwen3vl-runtime`，0.13 GiB），整理任务的盘点里它属于待定的 D4；我会在整理窗口问到时答"不能动"。
   - **乙**：新建一个独立的环境。要安装，约半天。
4. **Cambrian 做到哪一步**（第 1 项选甲或乙时才有）。
   - **甲（推荐）**：主实验全做（SFT + 两个 RL 臂）。约 4–5 天代码（"一道题一个序列"的 Cambrian 分支约 1.5 天、SigLIP2 和长度约半天、自己写 SFT 训练约 2–3 天）加 580–850 卡·小时；下游不做（见第 5 项）。
   - **乙**：只做零样本评测和 SFT，不做 RL。约 2.5–3.5 天代码、约 60–75 卡·小时；缺"密集对稀疏"那一行。
   - **丙**：只做零样本评测。约 1 天、2 卡·小时。
5. **Cambrian 不能作 StarVLA 骨干**（你的规定：写出缺口和代价，停下来等你）。
   - **甲（推荐）**：Cambrian 的下游行留空，报告里注明原因。
   - **乙**：在本线目录里写一个 StarVLA 的 Cambrian 骨干接口，运行时登记、不改 StarVLA 的文件。约 3–5 天，加 RLinf 一侧 1–2 天；4 个起点至少约 1,700 卡·小时（按 Qwen 的数字估算；Cambrian 每帧 token 多，实际可能更高）。
6. **RL 的 KL 停止线用 1.0 还是 0.2**（你的简报写 1.0，主线标准的默认是 0.2）。
   - **甲（推荐）**：照简报，本线所有 RL（含 Qwen2.5-VL 参照行）都用 1.0，组内比较一致。C0 和 V4 用的是 0.2，但它们的 KL 只到约 0.016，实际碰不到。
   - **乙**：用 0.2，和 C0、V4 完全相同。
7. **新模型上 V4 的 keep 门槛**（0.3318 是用 Qwen2.5-VL 第一对 D 的训练记录标定的）。
   - **甲（推荐）**：沿用 0.3318、下限 1.0，逐次记录全失败题的离散度和补进来的题数。到第 50 次平均每次补进不到 1 道，就停下来报告，附该模型自己的离散度分布。这和主线 A 组沿用 C0 门槛的做法相同（主线决定 46、49）。
   - **乙**：每个模型先跑一个 D 再按它自己的记录标定。每个模型多约 70 卡·小时，晚约 10 小时。
8. **Qwen3-VL 权重的 sha256**（分片只有 root 能读）。
   - **甲（推荐）**：放进第二步第一个 root 任务里核对，约 3 分钟，结果写进状态文档。
   - **乙**：不核对，以完整加载为准。
