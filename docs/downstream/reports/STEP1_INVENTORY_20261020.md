# 下游具身任务线 · 第一步：盘点与核实（2026-10-20）

本报告回答任务书第一步的 6 个问题。第一步没有安装、下载或训练任何东西；唯一的 GPU 用量是一次 1 卡 sco 任务（加载测试，见 2.3 节），同一个任务里顺带做了 CPU 渲染检查（1.5 节）。证据文件都在 [step1/](step1/) 目录，集群上的原始产物在 `/mnt/umm/users/yinbaiqiao/downstream_runs/step1_load_test_20261020/`。

## 结论（先看这里）

1. **我们的四个起点权重都能直接当 StarVLA 的视觉语言骨干用，不需要转换、不产生新副本。** 四个都是完整的 Hugging Face 格式，架构相同（Qwen2.5-VL-7B，82.92 亿参数，隐藏维度 3584）。1 卡加载测试（sco 任务 `pt-405nyxoy`）里四个都装进了 StarVLA 的 QwenOFT 模型：加载报告里没有缺失、多余或形状不符的参数键；权重文件里的 729 个张量与加载后的模型逐个完全相等；用假数据走训练前向和推理都输出有限值，动作形状是 16 步 × 12 维。
2. **四个起点存的数值精度不同，要统一。** 原始 7B 存的是 bf16；SFT 第 9 个 epoch 是 bf16 与 fp32 混合（配置写 bf16）；P-S0 第 300 次和 P-D 第 150 次是全 fp32。StarVLA 按配置精度加载（`torch_dtype="auto"`），所以单卡、不经 DeepSpeed 时前两个是 bf16、后两个是 fp32，显存和数值都不同。正式的 8 卡 DeepSpeed 路径会在启动时把整个模型转成 bf16（DeepSpeed 源码已核实），差别在那里消失。建议在我们自己的启动器里统一按 bf16 加载（待你决定，D1）。
3. **StarVLA 的随机种子管不住动作头初始化和数据顺序，要在启动器里补。** 训练脚本在建完模型之后才设种子（`train_starvla.py:476` 建模型，`:144-145` 设种子），数据采样的种子写死为 42（`lerobot_datasets.py:82`，配置里的 seed 传不进去）。第三步要求"种子同时固定动作头初始化和数据顺序"，现在两样都做不到；可以在 `downstream/` 里写一个包装入口解决，不改 StarVLA（D2）。
4. **RoboCasa365 的数据和资产在集群上是齐的。** 目标任务（target，官方评测用的 50 个任务）的人工示教 50/50 套齐全：25,307 条示教、1,496 万帧、58 GB。场景与物体资产 6 类都在。预训练（pretrain）人工示教有 4 套缺失、3 套视频没解压完，第一、二、三步都用不到。
5. **计算节点上只能用 CPU 渲染（OSMesa），GPU 渲染（EGL）不行。** 10 月 2 日、10 月 17 日两次 EGL 都失败，OSMesa 在 10 月 2 日跑完过 2 个完整厨房回合（每个控制步约 0.22 秒，含推理）。今天在加载测试的同一个节点上复测：EGL 能看到的设备数为 0，仍然不可用；OSMesa 能渲染完整的 target 厨房（NavigateKitchen：重置 21.1 秒，之后每步 0.21 秒，3 个 256×256 相机，不含推理）。第二、三步的评测都按 OSMesa 计时。
6. **StarVLA 只跑过与本项目权重无关的东西；计划过的 8 卡连通试跑没有跑过。** 跑过的：10 月 1 日 Qwen3-VL-4B 的 NavigateKitchen 训练 10 万步（8 卡，约 14.8 小时）；10 月 2 日闭环冒烟 2 回合 0 成功；10 月 16 日原始 3B / 7B 单卡各训 2 步。sco 全工作区 2026-09-30 以来 1,500 条任务里，10 月 16 日之后没有任何 StarVLA / RLinf / RoboCasa 任务。
7. **RLinf 现在不能做 StarVLA + RoboCasa365 的 RL。** 官方组合只有 StarVLA + LIBERO + GRPO 和 RoboCasa365 + π0（OpenPI）+ PPO。仓库里已有把两者接起来的补丁和准备脚本，但最后一次冒烟（10 月 17 日）在环境初始化就失败了：RLinf 的 Python 环境里装的是 RoboCasa 0.2.0，没有 RoboCasa365 的任务注册表；这个环境也没装 RLinf 自己列的 StarVLA 依赖（缺 `qwen_vl_utils`，Qwen2.5-VL 骨干的策略在里面加载不起来）。补上要另建或补装环境（要用 pip，需你同意），之后还没验证过渲染、显存和 7B 策略。
8. **按 StarVLA 的默认配方，第三步全量做太贵，建议缩步数、评测用固定子集。** 估算（不是实测）：7B 每步约 1 秒（4B 实测 0.534 秒/步按参数量放大），默认 20 万步 ≈ 56 小时一个 8 卡任务，36 个任务 ≈ 2,000 个 8 卡小时；完整评测 50 任务 × 50 回合 = 2,500 回合，每个模型最多约 452 万控制步，按 10 月 2 日实测每步 0.22 秒、单进程约 276 小时。具体方案见第 6 节，要你选（D5、D6）。

## 术语

- **起点（权重）**：做 StarVLA 训练时视觉语言部分的初始权重。本报告的 B / S / RS / RD 分别是原始 Qwen2.5-VL-7B、第十一轮动作 SFT 第 9 个 epoch、P-S0 第 300 次、P-D 第 150 次。
- **HF 格式**：Hugging Face 的标准模型目录（`config.json`、`model-*.safetensors` 分片与索引、分词器与图像处理器文件），`from_pretrained` 能直接读。与之相对的是训练框架自己的分片检查点（FSDP / DCP，每张卡一个文件）和 LoRA 增量（只存改动参数的小文件）。
- **QwenOFT**：StarVLA 里本项目要用的模型结构：在提示后面放 16 个"🔍"占位符，取视觉语言模型最后一层在这些位置的隐藏状态，经一个新的 MLP 动作头回归出 16 步 × 12 维的连续动作，损失是 L1。
- **动作头**：接在视觉语言模型后面、输出连续动作的小网络（7B 时约 1.29 亿参数），每次训练都从随机初始化开始。
- **bf16 / fp32**：每个参数占 2 字节 / 4 字节的浮点格式。
- **DeepSpeed ZeRO-2**：把优化器状态和梯度切分到各卡上以省显存的训练方式；StarVLA 默认用它跑多卡。
- **target / pretrain**：RoboCasa365 的两类数据与厨房。target 是 50 个官方评测任务的人工示教（每任务约 500 条）和评测厨房；pretrain 是 300 多个任务的预训练示教。
- **atomic_seen / composite_seen / composite_unseen**：target 50 个任务的三组：18 个单步任务、16 个多步任务（预训练里出现过）、16 个多步任务（预训练里没出现过）。
- **任务时限（horizon）**：RoboCasa 1.0.1 给每个任务规定的最大控制步数，单步任务 450–1,050 步，多步任务 1,200–4,350 步；一步 = 一次 12 维动作。
- **EGL / OSMesa**：两种无显示器的离屏渲染方式。EGL 用 GPU 渲染；OSMesa 用 CPU 软件渲染，慢但不依赖显卡驱动的图形部分。
- **sco 任务**：集群的 GPU 作业；任务号形如 `pt-xxxxxxxx`。

## 1. 三个框架的状态

### 1.1 代码

| 框架 | 位置 | 版本 | 本项目的改动 | 相关分支 / 未合并 / 未跟踪 |
| --- | --- | --- | --- | --- |
| StarVLA | 集群主检出里的嵌套仓库 `VAGEN-Lite/third_party/starVLA`（父仓库 `.gitignore` 第 57 行忽略 `third_party/`，不在 git 里） | 分支 `starVLA_dev`，提交 `4507931`（"[docs]: add VLAct benchmark results (#474)"），与 `origin/starVLA_dev` 一致 | 工作区改了 8 个文件，69 行增 12 行删（全文 [step1/starVLA_working.diff](step1/starVLA_working.diff)）：① VLM 按 `config.json` 的 `model_type` 选 Qwen2.5 / Qwen3 接口（支持 `actor/huggingface` 这类目录名）；② 训练配置可直接写 `mixture_spec`（显式的数据集清单）；③ Qwen3 接口不再强制 sdpa、不再 `ignore_mismatched_sizes`；④ RoboCasa365 注册表加 NavigateKitchen pretrain 混合；⑤ 框架模块导入失败时跳过而不是报错；⑥ pytorch3d 改为可选；⑦ 评测脚本模块路径修正。其中 ①②③ 和评测路径这 5 个文件存成了父仓库的 `examples/train/robocasa/patches/active_spatial_transfer.patch`（`git apply --reverse --check` 通过，即与工作区一致）；**④⑤⑥ 3 个文件的改动不在任何补丁文件里** | 未跟踪 2 个：`examples/simBenchmarks/Robocasa_365/train_files/{run_robocasa365_navigatekitchen.sh, starvla_qwenoft_navigatekitchen.yaml}`；另有 `playground/` 下的模型与检查点（见 1.4） |
| RLinf | 嵌套仓库 `VAGEN-Lite/third_party/RLinf` | 分支 `main`，提交 `c70606f`（2026-10-02，"fix(scheduler): avoid aborts in concurrent group creation (#1645)"） | 3 个文件 45 行增（[step1/RLinf_working.diff](step1/RLinf_working.diff)）：RoboCasa365 的 12 维动作全部按 min/max 反归一化（不走 LIBERO 的夹爪二值化）、要求 min/max 统计量、一个单元测试。与父仓库 `examples/train/robocasa/patches/rlinf_robocasa_starvla.patch` 一致（reverse-check 通过） | 未跟踪：`venv_mnt/`（Python 环境，19 GB，10 月 16 日从登录节点本地盘复制过来）、`.venv`（指向它的链接）、`.syslibs/`、`logs/` |
| RoboCasa（两份） | ① `probe_spatial/robocasa365/robocasa`（probe_spatial 仓库里的解压副本，来自 `robocasa-main.tar.gz`，无独立 git） ② `VAGEN-Lite/third_party/robocasa`（RLinf 的 fork，10 月 16 日从登录节点 `/var/tmp` 复制来） | ① **1.0.1**（RoboCasa365，有官方任务注册表 `dataset_registry_utils.py`，要求 mujoco 3.3.1） ② **0.2.0**（旧版 RoboCasa，`main` 分支 `25dfb05`，没有 RoboCasa365 注册表，要求 mujoco 3.2.6） | 无代码改动；② 多了未跟踪的 `fixtures.repo/` 等资产文件，`textures`、`generative_textures` 是指向 ① 的链接 | ② 的远端还有 `rlinf`、`rpent`、`v1.0.1_rlinf` 三个分支（没取下来，内容未核实） |

父仓库（VAGEN-Lite）里与下游有关的代码都已在 main：`scripts/{prepare_starvla_transfer, check_starvla_transfer, eval_starvla_heldout, eval_starvla_mobile, export_starvla_backbone, prepare_rlinf_robocasa, export_rlinf_starvla}.py`、`examples/{train,evaluate}/robocasa/`、`vagen/envs/robocasa/`。相关分支：`active-spatial-pipeline-github-handoff`（10 月 10 日已合并）；`cleanup/20261016` **有 10 个提交没进 main**，其中 `docs/reports/SMOKE_DOWNSTREAM_20261016.md`（10 月 16 日单卡冒烟报告）和 `docs/reports/DECISIONS_20261016.md` 只在这个分支上。整理任务后续的"代码搬动"计划会移动 `scripts/` 和 `examples/` 的文件（同一分支 STATUS 第 1 节），届时本线引用的路径会变。

### 1.2 环境

pip 在登录节点上能用：`pip 26.2.1`，PyPI、阿里云、清华镜像都能连上（HEAD 请求返回 200，经默认代理和不经代理都通）；`~/.cache/pip` 当前用户不可写，会关掉缓存。计算节点没有外网（10 月 16 日冒烟报告），装包只能在登录节点做。下表版本来自各环境的包元数据，导入结果来自逐个模块的单独导入（[step1/env_*.json](step1/)）。

| 环境 | 路径 | 用途 | Python | torch | transformers | flash-attn | 仿真相关 | 能否导入 / 缺什么 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| starVLA | `~/.conda/envs/starVLA` | StarVLA 训练与策略服务 | 3.10.21 | 2.6.0+cu124 | 4.57.0 | 2.7.4.post1 | 无（不需要） | torch、transformers、flash_attn、accelerate 1.5.2、qwen_vl_utils、av、decord、starVLA（指向 `third_party/starVLA`）都能导入；deepspeed 0.16.9 要先设 `CUDA_HOME`（启动脚本 `setup_starvla_env.sh` 会设）；没有 lerobot（StarVLA 用自带的 gr00t 读取器，不需要） |
| 仿真客户端 | `probe_spatial/robocasa365/policies/diffusion_policy/.venv` | StarVLA 闭环评测时跑 RoboCasa365 | 3.10.20 | 2.6.0+cu124 | 5.17.0 | — | mujoco 3.3.1、robosuite 1.5.2、robocasa 1.0.1、gymnasium 1.3.0 | torch、accelerate、mujoco、robosuite、robocasa 1.0.1（含 365 注册表）、gymnasium、hydra、av、starVLA 能导入；transformers 在登录节点上 240 秒内没导入完（文件系统当时很慢，未判定，闭环客户端本身不导入它）；没有 lerobot、decord、qwen_vl_utils（客户端不需要）。10 月 2 日的闭环冒烟就是用这个环境跑通的 |
| robocasa365 | `probe_spatial/robocasa365/.venv` | RoboCasa365 本身（数据工具） | 3.13.14 | 2.9.1+cpu | — | — | mujoco 3.3.1、robosuite 1.5.2、robocasa 1.0.1、lerobot 0.3.3 | 都能导入，含 `robocasa.utils.dataset_registry_utils` |
| RLinf | `third_party/RLinf/.venv` → `venv_mnt` | RLinf 训练 | 3.11.14 | 2.11.0+cu130 | 4.57.6 | 未装 | mujoco **3.2.6**、robosuite 1.5.1、robocasa **0.2.0**、ray 2.59.0、gymnasium 0.29.1、starVLA 1.0.1 | torch、transformers、accelerate 1.15.0、mujoco、robosuite、robocasa 0.2.0、ray、hydra、rlinf、starVLA、lerobot、gymnasium 能导入。缺：① **RoboCasa 1.0.1 的注册表** `robocasa.utils.dataset_registry_utils`（10 月 17 日 RLinf 冒烟就失败在这）；② **`qwen_vl_utils`**——StarVLA 的 Qwen2.5-VL 接口第 10 行就导入它，所以在这个环境里装不起 Qwen2.5-VL 骨干的 StarVLA 策略（10 月的冒烟用的是 Qwen3-VL 策略，碰不到这里）；③ flash_attn、deepspeed、decord。另外 transformers 包目录里的代码是 4.53.2，元数据却写 4.57.6，两者不一致（原因未核实）。RLinf 自己给 StarVLA 的依赖清单 `requirements/embodied/models/starvla.txt`（transformers 4.57.6、accelerate 1.5.2、peft 0.11.1、qwen-vl-utils 等）没有装进这个环境：它是按别的模型装的（accelerate 1.15.0、peft 0.21.2） |

### 1.3 数据与资产

逐套核对了每个数据集的 `info.json` 记的示教数、`episodes.jsonl` 行数、数据文件数、每个相机的视频数是否一致（[step1/datasets_check.tsv](step1/datasets_check.tsv)），对照的是 RoboCasa 1.0.1 自带的官方清单 `box_links_ds.json`（410 套）。数据在 `probe_spatial/robocasa365/datasets/v1.0/`，格式是 LeRobot v2.1（每条示教一个文件），符合 `prepare_starvla_transfer.py` 的要求（只收 v2.0 / v2.1）。

| 部分 | 官方清单 | 齐全 | 缺失 | 不完整 | 示教数 | 帧数 | 大小 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target 单步任务（人工） | 18 | 18 | 0 | 0 | 9,126 | 2,231,347 | target 合计 58 GB |
| target 多步任务（人工） | 32 | 32 | 0 | 0 | 16,181 | 12,726,552 | （同上） |
| pretrain 单步（人工） | 65 | 63 | 2（PickPlaceMicrowaveToCounter、PickPlaceStoveToCounter） | 0 | 7,135 | 1,433,581 | pretrain 合计 412 GB |
| pretrain 多步（人工） | 235 | 230 | 2（DumpLeftovers、SetupBowls） | 3（SoakSponge 视频 54/100、StackBowlsInSink 0/100、StoreDumplings 8/106；目录里还留着没解压完的 tar） | 24,171 | 27,039,320 | （同上） |
| pretrain MimicGen 合成 | 60 | 57 | 3 | 0 | 507,189 | 108,379,328 | （同上） |

- 每个 target 数据集都有 3 个相机（左、右、手腕）的 256×256 视频、16 维状态、12 维动作、`modality.json`。
- **资产**：RoboCasa 1.0.1 下载脚本列的 6 类资产（纹理、生成纹理、Lightwheel 固定件、Objaverse 物体、AI 生成物体、Lightwheel 物体）都在 `probe_spatial/robocasa365/robocasa/robocasa/models/assets/`（各 18、4、20、81、153、69 个条目，合计 23 GB）。没有逐文件核对；10 月 2 日用它跑完过 2 个完整厨房回合。RLinf 用的旧版副本 `third_party/robocasa` 的资产 7.1 GB（纹理链接到上面这份）。
- **StarVLA 的读取器与归一化统计量**：读取器在 StarVLA 里（`examples/simBenchmarks/Robocasa_365/train_files/data_registry/data_config.py`，官方 50 个 target 任务的混合都已注册）。统计量（`dataset_statistics.json`）是训练开始时从所用训练数据现算的，现在只有 10 月 1 日 NavigateKitchen 那次训练留下的一份，不能给新实验用。读取器会往数据目录的 `meta/` 写缓存文件，共享数据要用"可写元数据覆盖层"；`prepare_starvla_transfer.py` 已经按示教建链接和覆盖层。
- **官方 10% / 30% 子集**：RoboCasa 1.0.1 注册表里有 `target_*_10p`、`target_*_30p`，即每任务 50 条 / 150 条示教（`filter_key = 50_demos / 150_demos`，`dataset_registry_utils.py:84-85`）。LeRobot 版数据里这两个 key 对应具体哪几条示教，未核实。

### 1.4 运行记录

查了：sco 全工作区任务列表 3 页共 1,500 条（创建时间 2026-09-30 17:31 至 2026-10-20 11:17 UTC，[step1/sco_jobs_downstream.tsv](step1/sco_jobs_downstream.tsv) 是按 starvla / robocasa / rlinf / downstream / navigatekitchen 过滤后的结果）；集群上 `VAGEN-Lite/{exps, playground}`、`third_party/starVLA/playground`、`third_party/RLinf/logs`、各 worktree、`~/tmp`、`~/sco_jobs`；git 所有分支的提交信息。

| 时间（UTC） | sco 任务 | 权重 | 任务 | 跑到哪 | 结果 | 产物 |
| --- | --- | --- | --- | --- | --- | --- |
| 10-01 02:51–17:40 | `pt-2av31dl7`（8 卡 H800，ZeRO-2） | StarVLA 官方 Qwen3-VL-4B-Instruct-Action（与本项目无关） | NavigateKitchen pretrain 人工示教 503 条，QwenOFT，每卡 batch 8，10 万步 | 完成 10 万步，约 14.8 小时（每 1 万步 89 分钟，0.534 秒/步） | 动作 L1 损失 0.285（第 100 步）→ 0.0012（第 10 万步）；训练批次上的诊断，不是留出集 | `third_party/starVLA/playground/Checkpoints/starvla_qwenoft_NavigateKitchen_20261001_025134/`（11 个 .pt，各 9.8 GB） |
| 10-02 06:17 | 无（登录节点） | 同上 final | 离线推理 3 种输入 | 完成 | 输出 (1, 16, 12)、有限值 | `playground/Checkpoints/starvla_nk_eval/infer_report.json` |
| 10-02 20:43–20:58 | `pt-vsvqp1z7`、`pt-b2dlso6g`、`pt-xkffg3xp` | 同上 | 闭环冒烟的前三次 | 失败 | EGL："Cannot initialize a EGL device display" | `playground/Experiments/starvla_transfer/closed_loop_smoke_20261002T20{4329,4808,5817}Z/` |
| 10-02 21:00–21:04 | `pt-f6ctnhzt`（1 卡） | 同上 final | NavigateKitchen target 厨房，2 回合，每回合上限 200 步，每 8 步重规划，OSMesa 渲染 | 2/2 回合跑完 | **0/2 成功**（"去咖啡机""去水槽"，各 200 步、25 次推理、43.7 / 46.3 秒） | `.../closed_loop_smoke_20261002T210042Z/eval/`（report.json、2 段视频） |
| 10-02 | 无 | 无权重（3B 配置只用于检查） | 两任务数据准备（NavigateKitchen、PickPlaceCounterToCabinet，训练 453 / 452 条，留出 50 / 50 条） | 数据读取与归一化往返检查通过 | — | `playground/Experiments/starvla_transfer/pilot_20261002/` |
| 10-16 06:45–07:01 | `pt-c8ukfb85`、`pt-13kh2el1`、`pt-fsqblfdn`、`pt-udcgghvj`、`pt-tv4rhwuw`、`pt-6wnv9e1u`（均 1 卡） | 原始 Qwen2.5-VL-3B / 7B（HF 缓存） | StarVLA NavigateKitchen 2 步、batch 1 | 前四次失败（单卡 ZeRO-2 显存不够；准备器拒绝已有目录），后两次去掉 DeepSpeed 完成 | 7B：可训练参数 84.21 亿、约 3 秒/步，无评测 | `exps/_cleanup_smoke_20261016/starvla_*`（检查点已按记录删除，`REMOVED_smoke_weights_20261016.tsv`） |
| 10-16 06:45–11:16 | `pt-ignjv4xu`、`pt-nhsxtxqj`、`pt-mabdf9b4`、`pt-nfx51b5p` | 10-01 的 NavigateKitchen 检查点 | RLinf PPO，2 次更新、2 个环境 | 失败 | Python 环境在登录节点本地盘 / 镜像里没有 git / 没有 EGL、OSMesa 库 | `exps/_cleanup_smoke_20261016/rlinf{,2,3,4}/` |
| 10-17 02:05 | `pt-a3vjvtyk`（1 卡） | — | 渲染诊断 | 完成 | NVIDIA EGL 两种接法都失败 | `exps/_cleanup_smoke_20261016/gl_diag/` |
| 10-17 02:05–02:09 | `pt-65515vh8`（1 卡） | 同上 | RLinf 第五次 | 失败（退出码 255） | 环境初始化：`robocasa.utils.dataset_registry_utils` 不存在（RoboCasa 0.2.0）；EGL、OSMesa 两次都停在这里，渲染没测到 | `exps/_cleanup_smoke_20261016/rlinf5/` |
| 10-15 之后计划的 8 卡 StarVLA 连通试跑 | **没找到** | | | | 唯一记录是 `cleanup/20261016` 分支 `DECISIONS_20261016.md`："需要一次 8 卡 StarVLA 训练，属于正式实验，等你安排" | |
| 10-20 12:28 提交，13:30–13:35 运行（本次） | `pt-405nyxoy`（1 卡 H800，h800 池，排队约 62 分钟） | B / S / RS / RD | 加载测试 + 渲染检查 | 完成，退出码 0 | 四个起点都通过；EGL 失败、OSMesa 通过（见 1.5、2.3） | `downstream_runs/step1_load_test_20261020/`（证据副本在 [step1/](step1/)） |

### 1.5 无头渲染

| 时间 | 节点 | EGL | OSMesa | 证据 |
| --- | --- | --- | --- | --- |
| 10-02 | sco 1 卡 | 失败："Cannot initialize a EGL device display ... PLATFORM_DEVICE extension" | 小场景通过；完整厨房 2 回合跑完（每步约 0.22 秒，含推理），需要 `NUMBA_DISABLE_JIT=1`（否则 llvmlite 在摆放物体时崩溃） | `closed_loop_smoke_20261002T210042Z/` |
| 10-17 | sco 1 卡（zoetrope） | 直接用：`eglQueryString` 为空（没有 EGL 调度库）；配 conda 的 glvnd + NVIDIA 厂商文件：建不出渲染上下文。节点没有 `/dev/dri`，`NVIDIA_DRIVER_CAPABILITIES` 未设置，镜像里的 NVIDIA EGL 库是 550.90.07 版 | 未测 | `exps/_cleanup_smoke_20261016/gl_diag/worker.log` |
| 10-20（本次） | sco 1 卡（h800 池），驱动 595.71.05 | 失败：MuJoCo 的 EGL 枚举到 0 个设备（报错 "MUJOCO_EGL_DEVICE_ID ... must be an integer between 0 and -1"）；节点同样没有 `/dev/dri`，`NVIDIA_DRIVER_CAPABILITIES` 未设置 | 通过：小场景通过；target 厨房 NavigateKitchen（种子 0，指令 "Navigate to the stand mixer."）重置 21.1 秒，10 个零动作每步 0.21 秒，3 个相机 256×256 画面都不是常量；需要 `NUMBA_DISABLE_JIT=1`（沿用 10 月 2 日的设置） | [step1/render_egl.json](step1/render_egl.json)、[step1/render_osmesa.json](step1/render_osmesa.json) |

渲染用的是父仓库已有的项目内库覆盖层 `VAGEN-Lite/playground/Runtime/egl/`（10 月 2 日用 `setup_render_runtime.sh` 从 Ubuntu 包解出）。

## 2. 四个起点权重

### 2.1 路径与格式

登记在 [downstream/starts.json](../../../downstream/starts.json)（加一个起点只需加一条路径）。检查只读文件头，不读张量（[step1/hf_export.json](step1/hf_export.json)）。

| 起点 | 路径（集群 `VAGEN-Lite/exps/vagen_active_spatial/` 下） | 格式 | 参数 | 存储精度（文件里） | 配置写的精度（StarVLA 按它加载） |
| --- | --- | --- | ---: | --- | --- |
| B 原始 7B | `interiorgs_v10_r11_20261016/inputs/base_7b` | 完整 HF，5 个分片 16.6 GB | 8,292,166,656 | 全 bf16 | bf16 |
| S SFT 第 9 个 epoch | `interiorgs_v10_train_r11_20261016/lf_formal/model/checkpoint-1386` | 完整 HF，7 个分片 31.9 GB | 同上 | 语言部分与 lm_head fp32；视觉塔 6.32 亿参数 bf16、视觉合并层 0.45 亿 fp32 | bf16 |
| RS P-S0 第 300 次 | `interiorgs_v10_train_r11p_20261017/rl_run_ps0/checkpoints/global_step_300/actor/huggingface` | 完整 HF，7 个分片 33.2 GB（同目录上层另有 FSDP 分片与优化器状态，不用） | 同上 | 全 fp32 | fp32 |
| RD P-D 第 150 次 | `interiorgs_v10_train_r11p_20261017/rl_run_pd/checkpoints/global_step_150/actor/huggingface` | 完整 HF，7 个分片 33.2 GB（FSDP 分片已不在） | 同上 | 全 fp32 | fp32 |
| 备用：第十轮 SFT 第 5 个 epoch | `interiorgs_v10_train_r10_20261015/lf_formal/model/checkpoint-770` | 完整 HF | 同上 | 与 S 相同的混合 | bf16 |
| 备用：R1 最终模型 | `R1-clean-Projective-v0/formal/checkpoints/global_step_250/actor/huggingface` | 完整 HF | 同上 | 全 fp32 | fp32 |

- 六个都齐全：`config.json`、索引里列的每个分片都在且文件头里能找到每个张量、`tokenizer.json`、`tokenizer_config.json`、`preprocessor_config.json`、对话模板；不是 LoRA，也不是只有配置的空目录。嵌入层与 lm_head 形状都是 [152064, 3584]，与配置一致。
- 键名都是旧版布局（`model.layers.*`、`visual.*`），transformers 4.57 加载时会自动改名（2.3 节核对了改名后逐个张量相等）。
- QwenOFT 的动作占位符"🔍"在六个分词器里都是 1 个 token（id 146663），16 个连写得到 16 个 token。
- B 就是 HF 上的 Qwen2.5-VL-7B-Instruct 快照 `cc594898`：第 5 个分片的 sha256 = `0c859795…`，与 HF 缓存里的同名 blob 和 `/mnt/umm/shared_model/Qwen2.5-VL-7B-Instruct` 相同，`config.json` 逐字节相同；它和 `r1_h1_aoss_repair_20260905/.../qwen25vl7b_pretrained_cc594898` 是同一组硬链接（[step1/base7b_shard5_sha256.txt](step1/base7b_shard5_sha256.txt)）。
- **不需要转换，不会产生新副本。**
- 提醒：这些都是训练主线的目录。第十二轮训练链的"只留最好和最后"清理只删它自己目录里的检查点（`ea8c423f`），不会碰第十一轮的目录；但以后整理任务若移动或删除第十一轮检查点，这里的路径会失效。

### 2.2 StarVLA 对骨干的要求

- **支持的模型家族**（`starVLA/model/modules/vlm/__init__.py`）：Qwen2.5-VL、Qwen3-VL、Qwen3.5、Gemma-4、Molmo2、MiniCPM-V、Florence-2、Cosmos-Reason2、EgoVLA。QwenOFT 只用 Qwen2.5-VL / Qwen3-VL 接口。本地补丁让它按 `config.json` 的 `model_type` 选接口，所以 `actor/huggingface` 这种目录名也能认。
- **没有写死 3B 的地方**：动作头的输入维度在建模时取骨干的 `hidden_size`（`QwenOFT.py`：`action_hidden_dim = qwen_vl_interface.model.config.hidden_size`），隐层是它的 2 倍；`MLP_ActionHeader.py:65` 的 `input_dim=2048` 只是默认值，`get_action_model` 会覆盖。出现"3B"的地方只有：`QWen2_5.py:81` 在配置缺 `base_vlm` 时的默认模型名、文档字符串、`QwenDiscreteDiffusion` 等本项目不用的结构的默认值、FAST 分词器的 token 范围（OFT 不用）。视觉塔和图像处理器都随权重目录加载，图像统一缩放到 224×224。
- **`prepare_starvla_transfer.py` 的"同族同尺寸"**：比的是同一次调用里传进来的几个起点之间的 `(model_type, hidden_size)` 是否一致（第 172–173 行），不是和某个固定的 3B 比。四个起点都是 `(qwen2_5_vl, 3584)`，可以一起准备。
- **7B 全参数微调的显存**：
  - 实测：10 月 16 日原始 7B 单卡、不经 DeepSpeed、batch 1 训了 2 步，总参数 84.21 亿全可训练；当时没记录显存峰值。本次加载测试（单卡，按配置精度加载，2 条样本，不含优化器状态）：bf16 的 B、S 加载后 15.97 GB，训练前向峰值 22.37 GB，再做一次反向 33.54 GB；fp32 的 RS、RD 加载后 31.51 GB，前向峰值 52.93 GB，反向 64.55 GB。所以 fp32 起点单卡做全参数训练放不下（权重加梯度已 64.55 GB，AdamW 再要两份 fp32 状态约 67 GB）；bf16 起点单卡能训（10 月 16 日已实测 2 步）。
  - 估算（8 卡 ZeRO-2、bf16，StarVLA 默认路径）：每卡 bf16 权重 16.8 GB（不切分）+ bf16 梯度 16.8 / 8 = 2.1 GB + fp32 主权重与两份 Adam 状态 84.2 亿 × 12 字节 / 8 = 12.6 GB ≈ 31.5 GB，加激活值。按默认 batch 8 / 卡、3 张 224 图（每张约 64 个视觉 token）估计 80 GB 卡放得下，未实测。

### 2.3 加载测试（1 卡 sco 任务 `pt-405nyxoy`）

做法：每个起点单独一个进程，用 StarVLA 自己的建模入口 `build_framework`（与 `train_starvla.py` 相同的配置处理），骨干按路径直接读；截获 StarVLA 内部那次 `from_pretrained` 的加载报告（缺失 / 多余 / 形状不符的键）；再把权重文件里的每个张量和加载后的模型逐个比较（按加载精度比较是否完全相等）；然后在 1 张 H800 上用 2 条假样本（每条 3 张 224×224 随机图、一句指令、16×12 的随机动作）走一次训练前向（L1 损失）和一次 `predict_action`（推理路径）。另外顺带试了一次反向传播，只作参考，不计入成败。配置模板 `examples/train/robocasa/starvla_qwenoft_navigatekitchen_smoke.yaml`（12 维动作、16 步、不用状态）。代码 [downstream/load_test/](../../../downstream/load_test/)，结果 [step1/load_*.json](step1/)。

任务 12:28 UTC 提交，排队约 62 分钟，13:30 开始，四个起点共用 3.6 分钟（13:30:48–13:34:24），整个任务 13:35 结束，退出码 0。节点：H800 80 GB，驱动 595.71.05；starVLA 环境 torch 2.6.0、transformers 4.57.0。实际运行的代码是提交 `7e4fe62e`。`prepared.json` 记的是 `6916aae6`，因为任务开始前我修了 `load_in_starvla.py` 读配置的一行，记录在 `job/code_note.txt`。

| 起点 | 结果 | 加载报告：缺失 / 多余 / 形状不符的键 | 文件里的张量逐个比对 | 加载后的精度 | 参数：骨干 + 动作头 | 加载后显存 | 训练前向（2 条）：损失 / 显存峰值 | 反向（参考）显存峰值 | 推理输出 |
| --- | --- | --- | --- | --- | --- | ---: | --- | ---: | --- |
| B | 通过 | 0 / 0 / 0 | 729 / 729 相等 | bf16 | 8,292,166,656 + 128,608,268 | 15.97 GB | 0.751 / 22.37 GB | 33.54 GB | (1, 16, 12)，有限 |
| S | 通过 | 0 / 0 / 0 | 729 / 729 相等 | bf16 | 同上 | 15.97 GB | 0.627 / 22.37 GB | 33.54 GB | (1, 16, 12)，有限 |
| RS | 通过 | 0 / 0 / 0 | 729 / 729 相等 | fp32 | 同上 | 31.51 GB | 0.608 / 52.93 GB | 64.55 GB | (1, 16, 12)，有限 |
| RD | 通过 | 0 / 0 / 0 | 729 / 729 相等 | fp32 | 同上 | 31.51 GB | 0.622 / 52.93 GB | 64.55 GB | (1, 16, 12)，有限 |

- "逐个比对"是把文件里的张量转成加载后的精度再比。S 的 fp32 部分加载后变成 bf16，所以 S 等于"文件值舍入到 bf16"，不等于原始 fp32 值。
- 损失是随机初始化的动作头对随机目标动作的 L1 误差，只说明能算、是有限值，不代表任何能力。
- 每个起点有梯度的参数是 78.76 亿，比总数少 5.45 亿，正好是 lm_head（输出词表的那一层）。动作损失只用最后一层隐藏状态，不经过 lm_head，所以 StarVLA 训练不会改动它。以后把训练后的骨干导出来测"空间能力保持"时，lm_head 仍是起点的原值。
- fp32 起点的前向峰值比加载后多 21 GB，bf16 起点只多 6 GB。差的部分可能是混合精度计算时临时生成的 bf16 权重副本，未细查。

### 2.4 种子与精度：第三步对照要补的两处

1. **种子**。`train_starvla.py` 先建模型（第 476 行 `build_framework`，动作头在这里随机初始化）、后设种子（第 144–145 行 `set_seed(cfg.seed + rank)`），所以配置里的 seed 管不到动作头初始值；数据采样器的种子写死为 42（`lerobot_datasets.py:82` 默认值，`dataloader/__init__.py` 调用时不传），配置里的 seed 也管不到数据顺序。加载测试在建模前统一设了种子 1234。四个起点在建动作头那一刻的随机数状态哈希相同（`de89fdb4ebc4cb9b`），动作头初始权重的哈希也相同（`e8914bf63f2de953`）。也就是说，只要建模前设同一个种子，装哪个骨干都不影响动作头初值。不在建模前设种子时（现在的训练脚本就是这样），动作头初值由什么决定、会不会随 seed 变，未核实。做法建议：在 `downstream/` 写一个训练入口，建模前 `set_seed(seed)`，并把 seed 传给数据集（D2）。
2. **精度**。按配置精度加载时 B、S 是 bf16，RS、RD 是 fp32。8 卡 DeepSpeed 路径下 DeepSpeed 会在初始化时整体转成 bf16（`deepspeed/runtime/engine.py:1251-1254`，`bf16.enabled=true` 时 `self.module.bfloat16()`；StarVLA 的 `ds_config.yaml` 打开了 bf16），四个起点在训练里是一样的；但单卡或不经 DeepSpeed 时不一样，评测时策略服务用 `--use_bf16`。为了不依赖这个隐含行为，建议启动器统一按 bf16 加载（D1）。

## 3. RLinf 与 RoboCasa365 的兼容性（为第四步铺路，只查不做）

- **当前版本（`c70606f`）官方给出的组合**（`docs/source-en/rst_source/examples/embodied/{starvla, robocasa365}.rst`、`examples/embodiment/config/`）：
  - StarVLA（QwenOFT 等）+ LIBERO-Spatial + GRPO，参考模型是 Qwen2.5-VL-3B 的 OFT 检查点（`libero_spatial_grpo_starvla.yaml`）；
  - RoboCasa365 + OpenPI π0 + PPO，单任务 OpenDrawer（`robocasa365_opendrawer_ppo_openpi.yaml`），另有旧版 RoboCasa + OpenPI；
  - RLinf 的 StarVLA 包装支持 OFT、FAST、Adapter、流匹配四类动作头（`rlinf/models/embodiment/starvla/action_heads/`）；OFT 在 RL 里是"回归均值 + 可学的高斯标准差"（`oft.py` 用 `actor_logstd` 建高斯分布算对数概率）。
- **StarVLA + RoboCasa365 官方不支持。** 本项目已补的部分：RLinf 补丁（12 维全 min/max 反归一化）；`scripts/prepare_rlinf_robocasa.py`（读 StarVLA 检查点的配置和统计量，生成 PPO 配置：GAE、actor-critic、加价值头、学习率 1e-6）；`scripts/export_rlinf_starvla.py`（RL 后导回 StarVLA 格式）。
- **缺口**：
  1. **Python 环境**：现有 RLinf 环境既缺 RoboCasa 1.0.1（RLinf 的 `robocasa365` 环境要它的任务注册表；现装 0.2.0、mujoco 3.2.6），也缺 RLinf 自己给 StarVLA 列的依赖（`qwen-vl-utils` 等，版本还和现装的 accelerate / peft / transformers 冲突）。比较干净的做法是按 RLinf 的安装脚本另建一个"StarVLA + RoboCasa365"环境（`requirements/install.sh embodied --model starvla --env ...`），RoboCasa 1.0.1 用集群上已有的源码；或改用 RoboCasa fork 的 `v1.0.1_rlinf` 分支（要从 GitHub 取）。都要 pip 下载，要你同意。工作量估计 1 天（含重跑冒烟）。
  2. **渲染**：RLinf 的环境进程里还没成功渲染过一帧（10 月 17 日停在上一步）。节点上 EGL 不可用，只能 OSMesa（CPU），每个环境每步约 0.2 秒；RL 要几十个并行环境，CPU 只有每卡 14 核。这是 RL 速度的主要瓶颈，估计 1 天验证。
  3. **模型封装**：RLinf 官方的 StarVLA 例子是 3B；本地 10 月的冒烟用的是 4B 策略，也没走到训练。7B 的 actor（FSDP）、rollout（HF 生成后端）和价值头在 8 卡上的显存未知，要等第二步产出一个 7B 的 StarVLA 检查点后才能测。估计 1–2 天。
  4. **动作接口**：12 维动作顺序、min/max 反归一化、16 步动作块已由补丁和准备脚本处理并有单元测试；底盘/躯干/模式维是否被 RLinf 的 RoboCasa365 包装关掉（官方 OpenPI 配方可以关底盘）需要在冒烟时核对。
- **工作量合计（估计）**：在不出大问题的前提下 3–5 天到"2 次更新的冒烟通过"；正式 RL 的时间取决于 OSMesa 速度（第四步方案里再算）。
- **替代方案（第四步再细化）**：不用 RLinf，改用仓库里已有的 StarVLA 闭环评测客户端加一个简单的在线优化循环；或先只做 SFT 对照（第三步）不做 RL。代价在第四步方案里写。

## 4. 缺什么、需要你提供什么

| 项目 | 第几步要 | 来源 | 大小 | 说明 |
| --- | --- | --- | --- | --- |
| 从集群外下载的东西 | 第二、三步：**无** | — | — | target 50 任务的数据、资产、StarVLA 代码与环境都在集群上 |
| RLinf 的"StarVLA + RoboCasa365"Python 环境（RoboCasa 1.0.1、mujoco 3.3.1、qwen-vl-utils、RLinf 的 StarVLA 依赖清单） | 第四步 | pip（PyPI 或镜像，登录节点能连上），或 RoboCasa fork 的 `v1.0.1_rlinf` 分支 | 未核实（现有 RLinf 环境 19 GB） | 建议另建环境，不动现有的；先问你 |
| 缺失 7 套（人工 4、合成 3）、不完整 3 套的 pretrain 示教 | 只有用 pretrain 数据时才要 | 官方 Box 链接（`probe_spatial/robocasa365/download_robocasa365.sh`） | 未核实（残留的未解压 tar 合计约 4.2 GB） | 第一至三步用不到，不建议现在补 |
| 新实验的输出空间 | 第二、三步 | 共享盘 | 每个 7B StarVLA 检查点约 17 GB（bf16）；第三步 36 个任务只留最后一个检查点约 0.6 TB | 放 `/mnt/umm/users/yinbaiqiao/downstream_runs/`，不进 git |
| 你的决定 | 见文末 | | | |

## 5. 连通试跑的最小方案（第二步，现在不执行）

目的：用一个起点（RS = P-S0 第 300 次）把"加载我们的权重 → StarVLA SFT → RoboCasa365 闭环评测"走通一次，并把配置固化成第三步的模板。

| 环节 | 内容 | 卡数 | 预计时间（估算） |
| --- | --- | --- | --- |
| 数据准备（登录节点 CPU） | 3 个时限最短的单步任务：NavigateKitchen（导航，450 步）、OpenStandMixerHead（450 步）、CloseToasterOvenDoor（450 步）；每任务取前 100 条示教 = 300 条（约 4.3 万帧），另留每任务 20 条做留出集；用 `prepare_starvla_transfer.py` 的按示教链接与覆盖层，示教清单写进实验目录 | 0 | 10 分钟 |
| SFT | 走 StarVLA 的 8 卡 ZeRO-2 路径，经新写的入口（建模前设种子、把种子传给数据），bf16 统一加载；每卡 batch 8、200 步、每 5 步记一次损失、最后存一个检查点 | 8 | 加载约 5 分钟 + 训练 200 × 约 1 秒 ≈ 4 分钟 + 存盘约 3 分钟 |
| 留出集动作误差 | `scripts/eval_starvla_heldout.py`（60 条留出示教上的原始动作 MAE / MSE） | 同一任务里 1 卡 | 5 分钟 |
| 闭环评测 | 策略服务 1 卡（`--use_bf16`）+ 仿真客户端，target 厨房，3 个任务 × 5 回合 = 15 回合，按任务时限 450 步，每 16 步重规划（与动作块长度一致），OSMesa；评测种子 1000–1004 | 同一任务里 1 卡 | 每回合最多 450 × 0.22 ≈ 100 秒；3 个任务并行约 10 分钟 |

- 一个 8 卡任务里串行做完，预计总共 40–60 分钟（含排队前的准备，不含排队）。
- **成功标准**：① 训练损失下降（前 20 步均值 > 后 20 步均值）且全程有限；② 检查点能被策略服务加载、动作形状 (1, 16, 12)；③ 15 个回合全部跑完、算出成功率（不要求高，0/15 也算通过）；④ 实验目录里有最终生效的配置、示教清单、权重路径、代码提交号。
- 跑通后把配置固化为模板：起点只是一个参数（`starts.json` 的名字），其余全部共用。

## 6. 第三步的初步估算

### 6.1 StarVLA 在 RoboCasa365 上的默认配方

来源：`third_party/starVLA/examples/simBenchmarks/Robocasa_365/README.md` 与 `train_files/run_robocasa365_all.sh`。

- 训练数据：target 50 个任务的全部人工示教（25,307 条，1,496 万帧）一起训练；模型 Qwen3-VL-4B + QwenOFT；8 卡，每卡 batch 8；**20 万步**。README 存档的 4 个本地结果用了 12–22 万步、每卡 batch 16–26。
- 评测：50 个任务 × 每任务 50 回合 = 2,500 回合，按任务时限，target 厨房；存档结果总成功率 30.9%–40.6%。README 自己写明其评测的种子没传进环境，不能做配对比较。

### 6.2 时间（都是估算，不是实测）

| 项目 | 依据 | 一次 | 第三步 36 个 |
| --- | --- | --- | --- |
| 7B 每步训练时间 | 4B 实测 0.534 秒/步（10-01，8 卡、每卡 batch 8）× 参数量比（7B 骨干 82.92 亿 + 动作头约 1.29 亿，对 4B 骨干 44.43 亿 + 动作头约 0.66 亿）≈ 1.87 | ≈ 1.0 秒/步 | — |
| 默认 20 万步 | 1.0 × 200,000 | ≈ 56 小时（8 卡） | ≈ 2,000 个 8 卡小时 |
| 3 万步（全局 batch 64，看过约 192 万个样本，约为 target 总帧数的 13%） | 1.0 × 30,000 | ≈ 8.5 小时 | ≈ 300 个 8 卡小时 |
| 完整评测（2,500 回合） | 50 任务时限之和 90,450 步 × 50 回合 = 452 万步上限；10-02 实测每步 0.22 秒（OSMesa、单进程） | 单进程最多约 276 小时；每卡并行 12 个环境进程、8 卡共 96 个时约 3 小时（并行效率未测） | 36 个模型 × 约 3 小时 ≈ 108 个 8 卡小时（未测） |

### 6.3 评测太贵时的固定子集（建议）

| 方案 | 任务 | 每任务回合 | 回合数 | 控制步上限 | 代价 |
| --- | --- | ---: | ---: | ---: | --- |
| A（推荐） | 18 个单步任务全做 + 16 个已见多步任务全做 | 20 | 680 | 98 万 | 去掉未见多步任务（存档里 OFT 只有 11%–15%，地板效应，难分出起点差别）；单步任务每格 20 回合，成功率标准误约 ±11 个百分点，主要靠逐任务配对检验汇总 |
| B | 50 个任务全做 | 20 | 1,000 | 181 万 | 与官方任务集一致，但每任务回合数少于官方 50 |
| C | 18 个单步任务 | 50 | 900 | 59 万 | 与官方单步组完全一致、最便宜，但不含多步任务 |

无论哪个方案，评测种子固定、同一批任务与回合对所有起点相同，按任务类别拆开报告。

## 需要你决定的事

**D1 起点的加载精度**
- (a) **推荐**：启动器统一按 bf16 加载四个起点。与 8 卡 DeepSpeed 实际训练精度一致，显存一致，评测服务也用 bf16。代价：RS、RD、S 的 fp32 部分在加载时舍入到 bf16（DeepSpeed 路径本来也会这样做）。
- (b) 不管，沿用 StarVLA 默认。代价：单卡调试和评测时四个起点精度不同；8 卡训练时仍会被 DeepSpeed 转成 bf16，结论与 (a) 相同但依赖隐含行为。
- (c) 统一 fp32。代价：每卡多约 17 GB 显存，训练更慢，且与 StarVLA 默认配方不一致。

**D2 种子怎么管住动作头和数据顺序**
- (a) **推荐**：在 `downstream/` 写训练入口，建模前设种子、把种子传进数据集，不改 StarVLA。代价：多一层入口，要随 StarVLA 版本核对。
- (b) 改 `third_party/starVLA` 的训练脚本。代价：嵌套仓库不在 git 里，改动要另存补丁，和其他用这个检出的会话共享。
- (c) 不管。代价：第三步的"3 个种子"只变了 GPU 非确定性，动作头和数据顺序三次都一样，误差条会被低估。

**D3 第二步用的任务与规模**
- (a) **推荐**：第 5 节方案（3 个 450 步单步任务、300 条示教、200 步、15 个评测回合，一个 8 卡任务约 1 小时）。
- (b) 更小：1 个任务（NavigateKitchen）、100 条示教、50 步、5 回合。代价：省约一半时间，但看不出多任务混合的读取问题。

**D4 第三步的训练数据范围（"下游训练数据"指什么）**
- (a) **推荐**：target 50 个任务的全部人工示教（与 StarVLA 默认配方、官方 10% / 30% 子集一致；10% / 30% 用官方的每任务 50 / 150 条，天然嵌套，需先核实它们对应哪几条示教）。代价：数据量最大，步数相同时每条示教被看到的次数少。
- (b) 只用 18 个单步任务。代价：与评测方案 C 搭配最便宜，但缺多步任务的结论。

**D5 第三步的训练步数**（所有比例和起点共用一个数）
- (a) **推荐**：先在第二步之后用 B 和 RS 各跑一次 3 万步看成功率是否离开地板，再定。代价：多 2 个 8 卡任务（约 17 小时）。
- (b) 直接 3 万步。代价：36 个任务约 300 个 8 卡小时；若 3 万步时成功率都接近 0，起点之间分不出差别。
- (c) 默认 20 万步。代价：约 2,000 个 8 卡小时。

**D6 第三步的评测规模**
- (a) **推荐**：6.3 节方案 A（34 个已见任务 × 20 回合 = 680 回合）。
- (b) 方案 B（50 任务 × 20 回合）。
- (c) 官方全量（50 × 50 = 2,500 回合）。代价见 6.2 节。

**D7 RLinf 的环境修补（第四步前要做，现在只记下）**
- (a) **推荐**：第三步训练排队期间，按 RLinf 的安装脚本另建一个"StarVLA + RoboCasa365"环境（pip 下载 RLinf 的 StarVLA 依赖、mujoco 3.3.1 等；RoboCasa 1.0.1 用集群上已有源码），现有 RLinf 环境不动，然后重跑 2 次更新的冒烟。代价：要 pip 下载（大小未核实，需你同意），多占一份环境的磁盘（现有一份 19 GB）。
- (b) 往现有 RLinf 环境里补装。代价：省磁盘，但会改动 10 月 16 日的冒烟所用环境，依赖版本互相冲突（accelerate、peft、transformers）。
- (c) 等第四步方案写完再做。代价：第四步开始时多等 3–5 天。

**D8 本线代码和产物位置**（已按此执行，如不同意请告诉我）
- 代码与文档在 `VAGEN-Lite-downstream` 的 `downstream` 分支（只新增 `downstream/`、`docs/downstream/`）；大文件在 `/mnt/umm/users/yinbaiqiao/downstream_runs/`；StarVLA、RLinf、渲染覆盖层只按主检出 `VAGEN-Lite/third_party/`、`VAGEN-Lite/playground/` 的路径读，不改。
