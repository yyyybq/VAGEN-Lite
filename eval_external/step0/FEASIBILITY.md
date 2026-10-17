# 第 0 步：可行性摸底（2026-10-17）

## 结论

1. **QA 迁移的 6 个 benchmark 全部能做。** 数据都在 Hugging Face 上，公开、免申请。新下载约 6.2 GB，另有约 40 GB 已经在集群上。不需要付费 API，单张 H800 就能跑。粗估每个 7B 模型跑完全套约 5–8 卡时（第 1 步实测后更新）。
2. **WalkerBench 能做，不花 API 钱。** 全景图有 5.3 GB 的离线包。运行时只读本地缓存，谷歌地图 API 只在"预取全景图"的接口里用到。打分是规则判定，不调外部模型。模型走 OpenAI 兼容接口接入，我们在 SCO 作业里自己起 vLLM 服务就行。
3. **ESI-Bench 在现有集群上跑不了，这是停下来问你的点。**
   - 卡型：集群上能用的 4 个池（zoetrope、h800、zoe-cf-render、zoe-cci）全是 H800-80G。H800 是 Hopper 架构，没有 RT 核（RT 核是做光线追踪的专用硬件单元）。
   - 官方要求：ESI-Bench 依赖的 Isaac Sim 官方文档写明不支持没有 RT 核的卡（A100、H100）。ESI-Bench 的 README 也只认 20/30/40 系显卡，BEHAVIOR-1K 要求 RTX 2070 以上。
   - 容器：实测计算节点的容器里没有挂 NVIDIA 图形库，系统是 Ubuntu 20.04、glibc 2.31。
   - 需要你在下面三条路里选一条，见"需要你确认的事"第 1 条。
4. **自有环境上的诊断（3c）能做。** 不需要新数据，用现有评测集和打分函数。

## 术语

- **闭环**：模型的动作会改变它下一步看到的画面（走一步、转一下，再看新画面）。和它相对的是**静态问答**，题目和图都是固定的。
- **零样本**：不在这个 benchmark 上做任何训练，直接测。
- **MRA**：数值题的评分方式。把预测值和真值的相对误差，在 0.50、0.55、…、0.95 共 10 个阈值下分别判对错，取平均。VSI-Bench 和 VSTI-Bench 的数值题用它。
- **Wilson 区间**：比例的 95% 置信区间。在题少或比例接近 0、1 时，它比"正态近似"稳。
- **EASI**：一个空间智能评测套件，包装了 VLMEvalKit 和 lmms-eval 两个后端。仓库里 `third_party/EASI` 是 v0.2.2（commit `0c1a41e`），但它的两个子模块没有检出（目录是空的）。以前的 EASI-8 结果用的是 `third_party/lmms-eval`（commit `9a33763`）。
- **VLMEvalKit / lmms-eval**：两套通用的多模态评测框架。同一个 benchmark 在两者里的提示词、抽帧、答案抽取可能不同，所以分数会差几分。
- **卡时**：一张 H800 用一小时。

## 表 1：QA 迁移

"SI 报的数"一列是 Spatial-Interactor 论文里 Qwen2.5-VL-7B 基座的数和他们 7B 模型的数，写成"基座 / SI-7B"。表 1 的列对应关系已经用论文 HTML 里的原表核对过。

| benchmark | 题数 | 评分 | 输入 | 首选工具 | 数据和版本 | 新下载 | 集群上已有 | SI 报的数（基座 / SI-7B） | 注意 |
|---|---|---|---|---|---|---|---|---|---|
| SAT-Real | 150 | 准确率（二选一） | 1–2 张图 | lmms-eval `sat`（EASI 没有） | `nv-njb/SAT@85f0ea8`，是 `array/SAT` test 划分的重存版本，答案顺序用种子 42 固定 | 0.2 GB | 无 | 54.7 / 74.0 | 只有 150 题，1 题就是 0.67 分。在 55% 附近，95% 区间约 ±7.9 分。"差 1.5 分就查原因"的阈值比抽样误差还小，选项顺序换一下就可能差几分。 |
| SAT-Syn | 4,001 | 准确率 | 1–2 张图 | 自己写，照搬 lmms-eval `sat` 的提示词 | `array/SAT@bda5dde` 的 val 划分（合成场景） | 0.3 GB | 无 | 57.5 / 62.5 | SI 没写用的是哪个划分。val 是唯一的合成评测划分，先按它做，你确认。 |
| ViewSpatial-Bench | 5,712 | 准确率 | 单图 | EASI：VLMEvalKit `ViewSpatialBench`，或 lmms-eval `viewspatial` | `lidingm/ViewSpatial-Bench@95d9ea3` | 0 | 有（16 GB 缓存） | 36.3 / 42.4 | EASI 自己测的 7B 基座是 36.85。 |
| MindCube Tiny | 1,050 | 准确率 | 多视角图 | EASI：`MindCubeBench_tiny_raw_qa` / `mindcube_tiny` | `MLL-Lab/MindCube@9c941b4` | 0 | 有（`/mnt/umm/users/yinbaiqiao/MindCube`） | 29.3 / 74.3 | SI 明确写了用的是 Tiny 划分。 |
| MindCube 完整集 | 21,154 | 准确率 | 多视角图 | EASI：`MindCubeBench_raw_qa` / `mindcube_full` | 同上 | 0 | 有 | 无（MindCube 论文的 7B 基座是 29.26，EASI 测的是 31.48） | 题量是 Tiny 的 20 倍，每个模型约 2–3 卡时。 |
| VSI-Bench | 5,130 | 选择题准确率和数值题 MRA，按 8 个子项平均 | 32 帧 | EASI：VLMEvalKit `VSI-Bench_32frame`，或 lmms-eval `vsibench` | `nyu-visionx/VSI-Bench@3a6a1cf`（10 月 8 日还在更新，要钉住版本） | 0 | 有（5.4 GB 视频，版本待核对） | 总分 37.9、路线规划 30.4、相对距离 36.6 / 49.0、34.0、50.6 | EASI 的核对结果：128 帧下，VLMEvalKit 后端和官方代码差 0.2 分，lmms-eval 后端对 7B 高 5.2 分。我们以前的 EASI-8 结果用的是 lmms-eval。 |
| VSTI-Bench | 6,042 | 同 VSI-Bench | 32 帧 | VLM-3R 仓库自带的 lmms-eval 任务 `vstibench`（EASI 没有） | `Journey9ni/vstibench@0903eb0` | 5.5 GB | 无 | 总分 39.0、相机位移 8.6 / 49.5、36.6 | 2026-07-13 作者修正过相机位置的标签，影响"相机位移"等 5 类，修正前后的分数不能比。SI 9 月才发表，但没写用的是哪一版，所以"相机位移"这一项的对照可能不成立。 |
| MMSI-Bench（可选） | 1,000 | 准确率 | 多图 | EASI：`MMSIBench_wo_circular` / `mmsi_bench` | `RunsenXu/MMSI-Bench@ec7c92b` | 0 | 有 | 27.7 / 31.2 | |
| SPBench-MV（建议加） | 319 | 准确率 | 多视角图 | EASI：VLMEvalKit `SPBench-MV` | `hongxingli/SPBench@0361102` | 0.2 GB | 无 | 34.0 / 67.6 | 不在你的清单里，但它是 SI 主表的 4 项之一，加上基本不花钱。 |

**SI 主表里专用模型的数**（基座同为 Qwen2.5-VL，顺序是 VSI 总分 / 路线 / MindCube Tiny / VSTI 总分 / 相机位移）：SpaceR-7B 44.5 / 31.4 / 29.9 / 43.7 / 12.7；Video-R1-7B 33.4 / 31.4 / 30.9 / 36.2 / 5.4；ViLASR-7B 44.6 / 29.9 / 35.1 / 46.2 / 15.7；SpatialLadder-3B 44.8 / 27.3 / 43.5 / 46.2 / 29.7；Spatial-MLLM-4B 46.3 / 33.0 / 32.1 / 38.0 / 16.9。这些模型我们测完可以直接对。

**工具建议：** EASI 覆盖的 4 项（ViewSpatial、MindCube、VSI、MMSI）加上 SPBench，用 EASI v0.2.2 的 VLMEvalKit 后端，因为 EASI 核对过它和官方代码一致。SAT 用 lmms-eval，VSTI 用 VLM-3R 的 lmms-eval 任务。第 1 步先用两个后端各跑一次 base-7B 的 VSI-Bench，看哪个对得上 SI 的 37.9，再定下来。

## 表 2：闭环迁移

| | ESI-Bench | WalkerBench |
|---|---|---|
| 代码 | `ESI-Bench/ESI-Bench@9f6f15e`（2026-08-27） | `lalayang123456-ctrl/WalkerBench@b197f51`（2026-03-10） |
| 题目 | 题目 JSON 共 3,512 个，分 10 大类 29 子类（论文写的是 3,081 个，差在哪还没查）。HF 上的 `ESI-Bench/ESI-Bench@104667c` 只有元数据。 | Standard-100：5 类（角度、距离、高度、导航、可见性）各 20 题。完整集 1,000 题。题目 JSON 在仓库里。 |
| 场景和数据 | OmniGibson + BEHAVIOR-1K 场景资产。资产是加密的 USD 文件，安装要接受 NVIDIA Isaac Sim 的 EULA 和 BEHAVIOR 数据许可。我不会替你点同意。资产大小官方没写。 | 全景图 `lalayg/pano@4156768`，一个 5.33 GB 的 zip。全景图的连通关系在仓库自带的 `data/cache.db`（30 MB）里。 |
| 要不要 API | 不要。自带 `--provider qwen`，可以接本地 OpenAI 兼容服务（默认 `http://127.0.0.1:8000/v1`）。 | 不要。运行时 `observation_generator` 只读本地缓存。谷歌的 Tiles 和 Street View API 只在预取接口里用。打分按距离、误差阈值判定（到达目标 15 米内、高度误差 ±15%、距离误差 ±20%、角度误差 ±30°）。参考智能体 `examples/vln_agent.py` 走 OpenAI 兼容接口，接自己起的 vLLM 服务就行；代码里检查 `API_KEY` 非空，填一个占位值即可，不是真 key。 |
| 显卡 | 渲染要有 RT 核的 RTX 卡。另需一张卡跑模型。 | 环境本身只用 CPU。模型用一张 H800。 |
| 步数上限 | 30（README 示例和你给的设定一致） | 每题自带，Standard 里是 20–28 步。 |
| 能否在现有集群跑 | **不能**（见下面的实测） | **能** |
| SI 报的对照 | 8 类平均：基座 32.9%，SI 41.2%，平均交互轮数 12.34 对 10.55。27 类宏平均：26.3% 对 31.1%。SI 退步的 4 类是倾斜面 16.7→6.7、长程导航 20.0→6.7、连通性 53.3→40.0、可通行通道 40.0→20.0。 | 总成功率：基座 7.0%，SI 14.0%。平均步数 17.71 对 6.52。分项（基座 / SI）：角度 5 / 20，距离 0 / 0，高度 25 / 40，导航 5 / 10，可见性 0 / 0。 |
| 题少的问题 | SI 每类 30 题，而且是"从双方都跑完的结果里挑的"，不是随机抽的。30 题时 33% 的 95% 区间是 [19%, 51%]。8 类合起来 240 题时约 ±6 分，和"差 5 分就查原因"的阈值差不多大。 | 每类 20 题，1 题是 5 分。1/20 的 Wilson 区间是 [0.9%, 23.6%]。 |

**ESI-Bench 每类可用题数**（你要的"每类 100 题以上"有 8 类做不到，最多只能全测）：

| 你列的重点类 | 可用题数 | SI 那 8 类里另外 4 类 | 可用题数 |
|---|---|---|---|
| 视角（View Hallucination） | 437 | 刚体容纳（Rigid Containment） | 80 |
| 部分遮挡（Partial Occlusion） | 95 | 堆叠稳定（Stacking & Stability） | 89 |
| 空间距离（Spatial Distance） | 152 | 未观测变化（Unobserved Change） | 148 |
| 空间关系（镜面大类下的 Spatial Relations） | 113 | 动作顺序（Action Order Inference） | 77 |
| 智能体观测（Agent Observation） | 133 | | |
| 对应（Correspondence） | 88 | | |
| 可通行通道（Traversable Passage） | 60 | | |
| 长程导航（Long-Term Navigation） | 60 | | |

"空间关系"有两个同名的地方：镜面反射大类下的 Spatial Relations 子类（113 题，SI 表 7 里的就是它），和"空间关系"大类（3 个子类共 781 题）。我按 SI 的用法取镜面那个，你确认。

**计算节点实测**（2026-10-17，单卡 SCO 作业 `pt-cs6gjrm7`，已删除；原始输出在 `step0/gpu_probe_20261017.txt`）：

- 显卡和驱动：NVIDIA H800，算力版本 9.0，80 GB；驱动 595.71.05，CUDA 13.2。
- 容器：`NVIDIA_DRIVER_CAPABILITIES` 为空，所以只挂了计算库。没有 `libGLX_nvidia`、`libEGL_nvidia`、`libnvidia-rtcore`，只有一个 Vulkan 加载器。
- 系统：Ubuntu 20.04.3，glibc 2.31。
- 结论：就算换一个带图形库的镜像，卡本身没有 RT 核这一条也绕不过去。

## 表 3：模型

| 代号 | 用哪个 | 大小 | 状态 |
|---|---|---|---|
| base-7B | `Qwen/Qwen2.5-VL-7B-Instruct`（HF 缓存） | 16 GB | 已有 |
| base-3B | `Qwen/Qwen2.5-VL-3B-Instruct`（HF 缓存） | 7.1 GB | 已有 |
| ours-SFT | **建议换成第十轮最终模型** `exps/vagen_active_spatial/interiorgs_v10_train_r10_20261015/lf_formal/model/checkpoint-770`（STATUS 定的最终模型，验证集 47.1%，评测集 40.1%）。你表里写的 `interiorgs_v10_train_20261014/lf/model` 是第八轮的 SFT（3 个 epoch）。 | 16 GB | 已有 |
| ours-RL | 第十轮 RL `interiorgs_v10_train_r10_20261015/rl_run/checkpoints/global_step_{50,75,100,125,150}/actor/huggingface`（fp32，33 GB，加载时转成 bf16）。按验证集挑是第 50 次（46.6%，和起点 SFT 几乎一样）。第 150 次训得最久（验证集 41.4%，评测集 44.1%）。 | 33 GB | 已有，用哪一步要你定 |
| SI-7B | `kagakouko/Spatial-Interactor-Qwen2.5-VL-7B@3e5d2ec` | 16.7 GB | 要下载 |
| SI-3B（可选） | `kagakouko/Spatial-Interactor-Qwen2.5-VL-3B@71c5411` | 8.2 GB | 要下载 |
| 专用模型，底座都是 Qwen2.5-VL-7B，能用同一个加载器 | SpaceR-7B `RUBBISHLIKE/SpaceR`、Video-R1-7B `Video-R1/Video-R1-7B`、ViLASR-7B `AntResearchNLP/ViLASR`、VST-7B-RL `rayruiyang/VST-7B-RL`、SenseNova-SI-1.1 `sensenova/SenseNova-SI-1.1-Qwen2.5-VL-7B` | 各 16.6 GB | 要下载 |
| 专用模型，底座不同（可选） | SpatialLadder-3B `hongxingli/SpatialLadder-3B`（Qwen2.5-VL-3B 底座）、Spatial-MLLM-4B `Diankun/Spatial-MLLM-subset-sft`（要额外的几何编码器和自定义代码） | 15.0 / 10.7 GB | 要下载 |

所有 HF 模型和数据集都是公开的，不需要申请。版本号、大小在 `step0/hf_inventory.json`。

## 表 4：下载量、空间、环境

- 新下载：
  - 数据：SAT 0.5 GB、VSTI 5.5 GB、SPBench 0.2 GB、WalkerBench 全景图 5.3 GB（解压后再加约 5 GB），合计约 11.5 GB。
  - 模型：SI-7B、SI-3B 和 5 个 7B 专用模型，合计约 108 GB。
  - 总量约 120 GB，单项最大 16.7 GB，都在 200 GB / 500 GB 的线以下。如果 ESI-Bench 能做，BEHAVIOR-1K 资产另算，大小未知。
- 放在哪里：`/mnt/umm/users/yinbaiqiao/eval_external_assets/`（不进 git）。`/mnt/umm` 整个文件系统还剩 13 TB（已用 88%）。我没找到按人查配额的命令。
- 环境：新建一个 conda 环境 `eval-external`（约 15–20 GB），里面装 VLMEvalKit、lmms-eval 和 vLLM。不往 `vagen-lite` 里装任何东西，因为第十一轮训练链正在用它。
- 已克隆的仓库（都在 `eval_external_assets/repos/`）：ESI-Bench `9f6f15e`、WalkerBench `b197f51`、VLM-3R `206afd0`（VSTI 的评测代码）、SAT `20ffd44`。BEHAVIOR-1K 仓库太大，300 秒内没克隆完，已删掉，等 ESI-Bench 有着落再说。

## 表 5：GPU 用量粗估（第 1 步实测后更新）

| 项目 | 每个模型 | 模型数 | 合计 |
|---|---|---|---|
| QA 全套（6 项加 SPBench，32 帧） | 5–8 卡时 | 11（2 个基座、我们 2 个、SI 2 个、专用 5 个） | 55–90 卡时 |
| WalkerBench Standard-100 | 约 1 卡时（100 题 × 20–28 步） | 同上，去掉 3B 的 | 约 10 卡时 |
| 自有环境 279 题 × 外部模型 | 按现有评测作业的耗时算，待查 | 6–7 | 待查 |
| 历史打乱诊断 | 279 题 × 2 种打乱方式 | 2 | 待查 |

全部用单卡 SCO 作业。现在第十一轮的 PS0、PD 两条 RL 各占 8+1 张卡在跑，评测作业会和它们抢同一个池。

## 需要你确认的事

1. **ESI-Bench 怎么办**，三选一：
   - (a) 找带 RT 核的卡：RTX 4090、L40S、L20、A6000 都行，每张至少 24 GB 显存，最好同机还有一张卡跑 7B。如果要租云机器，这是要花钱的，你定。
   - (b) 退一步，用 ESI-Bench 官方的离线单视角渲染集 `ESI-Bench/esi-bench-passive-single`（2.76 GB）。这只能当静态问答，回答不了"连续决策"的问题。
   - (c) 先不做 ESI-Bench，后面的步骤照常推进。
2. **ours-SFT 和 ours-RL 用哪两个检查点。** 我的建议是：SFT 用第十轮最终模型 `checkpoint-770`；RL 用第十轮 `global_step_150`，因为要回答的是"RL 训练对迁移有什么影响"，训得最久的那个最能说明问题。另外加测第 50 次作对照。第十一轮的 PS0、PD 出结果后可以再换。
3. **外部专用模型测哪些。** 建议 5 个 7B（SpaceR、Video-R1、ViLASR、VST-7B-RL、SenseNova-SI-1.1），再加 SI-3B。底座不同的两个先不测。
4. **QA 工具。** 按上面的建议：EASI 的 VLMEvalKit 后端为主，SAT 和 VSTI 用 lmms-eval，第 1 步拿 VSI-Bench 定后端。另外 SAT-Real 只有 150 题，"差 1.5 分就查原因"的阈值在它上面没有意义。我建议改成"SI 的数落在我们 95% 区间外才查"。
5. **GPU 并发上限和用哪个池。** 建议最多同时 4 张单卡作业，放在 h800 池，不挤 zoetrope 上的第十一轮训练链。

以上没问题的话，下一步：建 `eval-external` 环境，下载 SAT、VSTI、SPBench 和 SI-7B，然后进第 1 步（base-7B 跑 VSI-Bench 和 SAT-Real）。

## 来源

- Spatial-Interactor：arXiv 2609.23038（表 1、2、4、5、7，已从论文 HTML 原表抽出）
- Spatial-Nav：arXiv 2609.29934（摘要页没有列出它测了哪些专用模型，要看全文）
- ESI-Bench：https://esi-bench.github.io/ ，https://github.com/ESI-Bench/ESI-Bench
- WalkerBench：https://github.com/lalayang123456-ctrl/WalkerBench
- Isaac Sim 硬件要求：https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html
- BEHAVIOR-1K 安装：https://behavior.stanford.edu/getting_started/installation.html
- EASI 核对表：`third_party/EASI/docs/Benchmark_Verification.md`（v0.2.2）
- VSTI-Bench 标签修正：VLM-3R README，2026-07-13 条目
