# 模型线状态（唯一的状态文档）

最后更新：2026-10-25 09:50 UTC。

模型线：把 Qwen3-VL-8B-Instruct 和 Cambrian-S（7B 为主）接进统一框架，在主线最高的 setting 上跑主实验和下游实验。报告在 [reports/](reports/)，实验登记表 `MATRIX.md` 第三步建。

## 1. 现在在哪一步

结论：第二步（接入与冒烟）的 GPU 冒烟全部做完。两个模型的 SFT、RL、评测冒烟和 Qwen2.5-VL 参照冒烟都跑完，两个 StarVLA 骨干的 8 卡下游最小方案都通过。报告 [reports/STEP2_SMOKE_20261025.md](reports/STEP2_SMOKE_20261025.md)，待你决定的 5 件事见第 1b 节，其中有两个模型的正式任务清单。

- 第一步（盘点与核实）已完成，你 10-24 确认。报告：[reports/STEP1_INVENTORY_20261024.md](reports/STEP1_INVENTORY_20261024.md)。
- 第二步各块（顺序按你的第 5 节）：

| 块 | 代码 | CPU 核对 | GPU 冒烟 |
| --- | --- | --- | --- |
| Qwen3-VL 接入 | 完成 | 完成：LLaMA-Factory 编码和 Qwen3-VL 自己的模板逐个 token 相同（2 条对话）；只监督回答；每帧 64 个图像 token 和处理器一致；位置编码函数和 Qwen2-VL 对图像相同，不用改 | 零样本完成；SFT 冒烟排队中 |
| Qwen3-VL 的 StarVLA 骨干 | 完成（StarVLA 自带 Qwen3-VL 接口） | — | 1 卡加载通过；8 卡最小方案排队中 |
| Cambrian-S 代码 | 完成（RL 拼接路径、评测分支、SFT 工具） | 完成：LLaMA-Factory 编码和 RL 训练端展开后的序列逐个 token 相同（2 条对话，2,277 和 5,008 个 token），每帧 756 个位置，像素值和 RL 处理器的相同；RL 配置在 CPU 上按冒烟的选项生成通过 | 加载测试通过（HF 和 vLLM 贪心回答相同，逐词对数概率差最大 0.013）；零样本完成；SFT 冒烟排队中 |
| Cambrian-S 的 StarVLA 骨干 | 完成（新接口，运行时登记） | 完成：分词器和 Qwen2.5-VL-7B 的完全相同、隐藏维 3584，动作头和动作 token 原样沿用；多相机按 Cambrian 的方式（每个视角 72 个位置）；批量前向和逐条前向相同 | 1 卡加载通过；8 卡最小方案排队中 |

- 自动接力（`models_runs/watchers/`）：都已用完或停掉，现在没有在等的。

## 1a. 你的决定（10-24，第一步报告之后）

1. **Cambrian-S 权重**：同意在登录节点经 hf-mirror 下载 Cambrian-S-7B（32.13 GB）和 SigLIP2-so400m-patch14-384（4.58 GB），放到共享模型目录下本线的子目录（`/mnt/umm/shared_model/vagen-lite-models/hub`，HF 缓存的目录结构）；下载后核对 sha256（来源没有就核对大小和分片清单），记录来源地址、修订号、时间。只下这两样，再要下别的先问。
2. **Qwen3-VL 的 SFT**：在现有的 LLaMA-Factory（`ca75f1e`）里用本线新文件登记 Qwen3-VL 的对话模板和多模态插件，不装新包，在 `.venv-qwen3vl-runtime` 里跑。
3. **Qwen3-VL 的 RL 和评测**用 `.venv-qwen3vl-runtime`；主环境一个包都不改。
4. **Cambrian-S 全做**：SFT、两个 RL 臂、评测、下游四个起点。
5. **StarVLA 适配 Cambrian-S**（新文件加开关，用 Cambrian-S 自己的模型类和处理器；估代价不超过 4 天就直接做）。
6. **KL 停止线用主线标准的 0.2**，和 C0、V4 一致（简报里的 1.0 是 A 组的规则，不用）。
7. **V4 的 keep 门槛沿用 0.3318、下限 1.0**。另外用新模型的 SFT 模型回放全失败题，算出它自己的前 20% 门槛，写进报告但不用；逐次记录补进来的题数，到第 50 次平均每次不到 1 道就停下来报告。
8. **Qwen3-VL 权重的 sha256** 放进第二步第一个 sco 任务核对。
9. **之前定的继续有效**：每个模型 RL 两臂（S0 对照、密集为主）；密集变体在提交 RL 时按主线最新汇总取最高（现在是 V4）；SFT 用新格式；下游四个起点都做，原始模型的起点骨干一通就提；和下游同级；Qwen2.5-VL 新格式参照行排最后。
10. **Cambrian-S 每帧 756 个 token 的开销先量再定**：主实验按估算做；下游若一个起点的训练时间超过 Qwen2.5-VL 同样任务（约 42 小时）的 3 倍，停下来报告再提。

10-24 22:45 第二步阶段汇报之后的决定（都选甲）：

11. **StarVLA 环境缺的两个包**（`ezcolorlog`、`open_clip`）：用替身文件，只由 Cambrian 入口加载，不往下游线共用的环境里装东西。
12. **下游多相机**：按 Cambrian 处理多图的方式，每个视角 72 个位置（估每步约为 Qwen2.5-VL 的 1.2 倍）；不用每视角 756 个位置的做法（估约 7 倍）。
13. **补 Qwen2.5-VL-7B 零样本对照行**：同样 705 道、温度 0.8（2 个 1 卡任务，约 0.8 卡·小时），结果名 `m_qwen25vl7b_base_t08`。

执行记录：

- dense-batch2 的 6 个提交按顺序拣进 `models`（10-24 19:41），没有冲突：`aa6451e8→7fe42c0d`、`ec7682ec→f7bce387`、`991cc92d→db55eba1`、`f8ba48c5→cd812b25`、`24f1272e→21294b24`、`418927c4→0fbc32b9`。
- worktree 的 `verl` 子模块从主 checkout 的 verl 本地克隆，检出 `a0ae85c`（没联网）。
- 决定 8：Qwen3-VL 的 4 个分片 sha256 和文件名（blob 名）全部一致（root 任务 `pt-v2kaero2`）。
- 决定 1：下载和核对完成，见第 3 节和第 3a 节。
- GPU 规则（你 10-24 的话："两个池都可用，只用空闲机器"）的执行方式：本线任务名都以 `models-` 开头；8 卡任务等到某个池没有主线任务在排队、且有空机器才提交；1 卡任务等到没有主线 1 卡任务（渲染服务、评测）在排队、且有机器有空卡才提交；暂停文件在时都不提交（`train_chain` 计划 `"pool": "idle"`，`model_matrix/sco_view.py`）。这样 Qwen3-VL 的 SFT 冒烟从 20:30 等到 00:38 也没等到（两个池一直有主线任务排队）。
- **10-25 00:38 你说"开始排队吧"**：本线任务改为直接排队。计划 `"pool": "queue"`、`model_matrix.sco_job --pool queue`：暂停文件不在就提交到排队任务最少的池（和主线的 `"pool": "auto"` 同一个选池规则，另加暂停文件检查）。已在等的 Qwen3-VL 冒烟链没提交过任务，停下后改计划重启（提交号 `07ced152` → `1da365cd`，"核对代码"重做一遍）；还没启动的计划由 `plans.py --recommit` 改池。
- 10-25 00:50 补了一处：Cambrian 的 SFT 任务环境里原来没有 `CAMBRIAN_SRC`，`cambrian_register` 会退到 8 月那台机器的路径，SFT 一开跑就会停。已改为 SFT 注册时先设好本集群的路径，Cambrian 型号的任务环境也带上它（`a5cac843`；排队中的 SFT 开跑时读的是新代码）。评测和 vLLM 采样不受影响（`cambrian_vllm.py` 本来就默认对的路径）。

## 1b. 待你决定的（10-25，第二步报告第 8 节）

14. **Qwen3-VL 的逐词概率差是 Qwen2.5-VL 的约 2 倍**（第 2–10 次更新平均差 0.0030 对 0.0014）。甲（推荐）：训练端改 SDPA 重跑 10 次更新的冒烟做对照，约 8 卡·小时，变小就正式用 SDPA；乙：照主线用 eager 直接上。
15. **正式 RL 加按检查点评测的停止线**。甲（推荐）：每 50 次更新的检查点评测里验证集贪心读不出超过 10% 就停这一臂、报告（新开关，不多花卡）；乙：评测改每 25 次一个，多约 10 卡·小时；丙：不加。
16. **正式 Qwen3-VL 任务清单**：SFT、选 epoch、两臂 RL、评测约 180 卡·小时（拿到卡后约 13 小时），加 DS-Q3B-100-s1 约 430 卡·小时（约 54 小时）。确认后提交。
17. **Cambrian 下游的梯度检查点只给 SigLIP2 做**。甲（推荐）：加开关再跑一次 8 卡最小方案（约 2.5 卡·小时），显存够、更快就用，按计算量估每步省 15–20%；乙：照现在两部分都做（每个起点约 86 小时）。
18. **正式 Cambrian 任务清单**：形式同 16，约 250 卡·小时（约 18 小时），加 DS-CSB-100-s1 约 690 卡·小时（约 86 小时）。RL 存盘只够一个模型的两臂同时跑，建议 Qwen3-VL 的 RL 跑完再开。

## 2. 谁在跑

**训练**

| 编号 | 内容 | 状态 | sco 任务 | 更新（UTC） | 卡在哪 |
| --- | --- | --- | --- | --- | --- |
| Q3-SMOKE-SFT | Qwen3-VL-8B SFT 冒烟：新格式 440 条对话（训练 400、验证 40），1 个 epoch | 完成 | `pt-45vp0qry` 失败；重试 `pt-emouoqg6`（h800） | 10-25 05:49 | 第一次 05:31 开跑就停在导入（Transformers 4.57.6 没有 `is_jieba_available`），已补（`637b72b8`）。重试 05:44 开跑、05:48 结束：7 步 86.8 s（每秒 4.61 条），逐步损失 1.357 → 0.221，验证损失 0.229。这次没有每卡显存记录（作业脚本当时不记，`792106c2` 起可以记）。之后在做两个评测集上的 SFT 评测 |
| Q3-SMOKE-RL | 从上面的 SFT（验证集成功率 0.182）跑 RL 10 次更新（V4，带对数概率核对） | 完成 | 渲染 `pt-o5lc6kte`、训练 `pt-dgsagbbx`（h800） | 10-25 06:51 | 每次更新 60–88 s（第 2–10 次平均 75.8 s；第 1 次 199 s 含预热；主线 Qwen2.5-VL 的 V4 是 79.8 s）；每卡显存峰值已分配 53.0 GB、预留 67.8 GB；回复最长 2,331 个 token（上限 4,096，没有截断）；KL 约 0.002；采样端和训练端逐词概率差平均 0.0025–0.0034、最大约 0.062，相关 0.999；keep 每次补进 0–6 道，平均 3.5；训练时（采样）读不出的回答几乎为 0。第 10 次更新的检查点贪心评测：测试集 292 道成功率 0.253（SFT 冒烟模型 0.164），验证集在出；**但贪心下有约 24% 的回合读不出**（SFT 冒烟模型为 0）：几乎全是把同一个动作重复到超过每步 5 个（`move_backward|move_backward|…`），验证集 413 道 0.242、读不出 25.8%（测试集 23.6%）。Qwen2.5-VL 参照同样的冒烟贪心下读不出只有 0.1% / 0.0%，所以这不是"起点弱"的共性（第 1b 节） |
| Q25-SMOKE | Qwen2.5-VL-7B 同样的冒烟（对数概率核对的参照，带显存记录） | 完成 | SFT `pt-xlg2kake`；渲染 `pt-nx4meznc`、训练 `pt-j4mmm0wa`；评测 `pt-a8wkqopg`、`pt-etemd4o7`（h800） | 10-25 08:55 | 接力条件没触发（进度里只记了渲染服务的任务），07:19 手动启动。SFT：7 步 79.1 s，逐步损失 0.511 → 0.242，验证损失 0.227，每卡显存峰值 44.2 GB；SFT 后验证 / 测试成功率 0.041 / 0.058。RL：每次更新第 2–10 次平均 107.3 s（第 1 次 223 s）；每局平均约 10.2 轮（Qwen3-VL 6.4、Cambrian 5.3）；每卡显存峰值已分配 54.3 GB、预留 67.2 GB；回复最长 2,750 个 token，没有截断；逐词概率差平均 0.0009–0.0034（第 2–10 次平均 0.0014）、最大 0.037–0.087；\|概率比 − 1\| > 5% 的 token 第 2–10 次平均 1.1%（Qwen3-VL 5.0%、Cambrian 2.4%）；KL 第 10 次 0.0095；keep 每次补进 2–6 道，平均 4.2；训练时读不出的比例后几次涨到 5–22%（10 次平均 5.2%）。第 10 次更新的检查点贪心评测：验证 0.252、测试 0.243，读不出 0.1% / 0.0% |
| C-SMOKE-SFT | Cambrian-S-7B 同样的冒烟（`m_cambrian7b_smoke_20261024`）的 SFT | 完成 | `pt-ukg68gka`（h800） | 10-25 05:44 | 05:32 开跑（排队 4 小时 48 分）：7 步 124.4 s（每秒 3.22 条），逐步损失 0.594 → 0.282，验证损失 0.255；同样没有显存记录。SFT 评测：验证集成功率 0.039 |
| C-SMOKE-RL | 从上面的 SFT 跑 RL 10 次更新 | 完成（第 10 次的检查点贪心评测：验证 0.218、测试 0.240，读不出 24.9% / 24.0%） | 第一次：训练 `pt-cfyohm6m`、重试 `pt-jcdq6zxf` 都失败；重跑：渲染 `pt-ntfx5g1f`、训练 `pt-ce8daimr` | 10-25 07:47 | **重跑（SDPA）**：每次更新第 2–10 次平均 93.1 s（第 1 次 216 s），约为主线 V4 的 1.17 倍；每卡显存峰值已分配 67.2 GB；回复最长 10,028 个 token（展开后）；逐词概率差平均约 0.002、最大 0.054–0.083，相关 0.9998，第 1 次更新比 eager 时明显更齐（平均 0.0027 对 0.0078，最大 0.059 对 0.206）；KL 升到 0.012；keep 每次补进 2–9 道，平均 6.2；**训练时读不出的回答比例在涨**：第 7、8、10 次约 18–19%（10 次平均 7.6%，停止线是 10 次平均 10%），读不出的多是编出来的动作名（`left`、`right`、`turn_backward`、`turn.right`、空的 `<action></action>`）。**第一次（eager）**： | 第一次第 1 次更新正常（逐词概率差平均 0.0078、最大 0.206，相关 0.997；最长序列 7.7k 个 token，显存峰值 63.8 GB），第 2 次更新显存不够：训练端用的是主线的 eager 注意力，它存一整张 28 × L × L 的 fp32 矩阵，一局 12 帧约 1.1 万个 token 时就是 13 GiB。改为 Cambrian 型号的任务用 SDPA 注意力（同一个公式，显存随长度线性涨），并打开 verl 的熵分块计算和重算（数值不变，`f9929c3f`），07:16 从 RL 配置一步重跑；第一次的记录在 `rl_attempts/` |
| DS-Q3B-smoke | 下游最小方案：原始 Qwen3-VL-8B 当 StarVLA 骨干，SFT 100 步 + 3 个任务 × 4 个种子闭环 | 完成 | 先 `pt-wdl14ea4`（zoetrope，排了 7.7 小时没拿到卡，删掉）；重交 `pt-d5m9e368`（h800） | 10-25 08:34 | SFT 100 步（全局批量 64）用时 323 s，模型每步 0.76 s（Qwen2.5-VL 参照 DS-B-100-s1 是 0.744 s）；每卡显存最高 74.3 GB（nvidia-smi）；闭环 12 局全部跑完、没有报错，成功 0 局（100 步本来就不期望成功）。外推到 20 万步：模型时间约 42 小时、墙钟约 54 小时（参照同口径约 41 小时、约 52 小时，约 1.02 倍），在决定 10 的 3 倍线以内 |
| DS-CSB-smoke | 同上，原始 Cambrian-S-7B | 完成（第三次通过） | 先 `pt-lqa0g87p`（zoetrope，同样删掉）；`pt-xmljmr8z`、`pt-w2mp26dk` 失败；第三次 `pt-crenzceo`（h800） | 10-25 09:35 | **第三次**：SFT 100 步用时 448 s，模型每步 1.22 s（参照 0.744 s 的 1.64 倍）；梯度检查点在 30 个模块上生效；每卡显存最高 50.2 GB；闭环 12 局全部跑完、没有报错，成功 0 局。外推到 20 万步：模型时间约 68 小时、墙钟约 86 小时，在 3 倍线以内。**前两次**：第一次第一步就显存不够（8 张卡都到 79.2 GiB）：每卡 8 条 × 3 个视角 = 24 张 SigLIP2 图像（每张 729 个块），视觉塔 26 层可训练，中间结果全存着放不下。修法：Cambrian 接口打开梯度检查点（语言模型和 SigLIP2 都用；StarVLA 自己的开关 `enable_gradient_checkpointing`，Cambrian 默认打开，`41313936`），梯度不变，每步慢约三成；批量、步数、可训练部分都不变。第二次同样显存不够：Transformers 只在训练模式下做检查点，StarVLA 从不切到训练模式，开关没生效。现在接口在算梯度时切到训练模式（模型里所有 dropout 都是 0，加载时核对；`cffcef1c`），CPU 上用真模型的类核对过检查点在语言模型各层和 SigLIP2 编码器上都生效。两次的记录在 `attempts/` |
| M-qwen3vl8b-sft / -s0 / -d | Qwen3-VL-8B 正式 SFT、RL 两臂 | 未提交 | — | — | 冒烟通过后发任务清单给你确认 |
| M-cambrians7b-sft / -s0 / -d | Cambrian-S-7B 正式 SFT、RL 两臂 | 未提交 | — | — | 同上 |
| M-qwen25vl7b-sft / -s0 / -d | Qwen2.5-VL-7B 参照行（新格式重跑） | 未提交 | — | — | 排在两个新模型之后 |
| DS-Q3B-100-s1 | 下游：原始 Qwen3-VL-8B | 未提交 | — | — | DS-Q3B-smoke 已通过，随正式清单一起确认 |
| DS-CSB-100-s1 | 下游：原始 Cambrian-S-7B | 未提交 | — | — | DS-CSB-smoke 已通过，随正式清单一起确认（决定 17、18） |
| DS-Q3S / Q3RS / Q3RD、DS-CSS / CSRS / CSRD | 下游：两个模型的 SFT、RL 两臂起点 | 未开始 | — | — | 等第三步的检查点 |

**评测**

| 编号 | 内容 | 状态 | sco 任务 | 更新（UTC） | 结果 |
| --- | --- | --- | --- | --- | --- |
| STEP1-LOAD | 1 卡加载测试（Qwen3-VL、Cambrian，HF 和 vLLM） | 已完成 | `pt-kqhed0xd`、`pt-gm3kn6l8` | 10-24 18:35 | 见第一步报告 |
| Q3-SHA256 | Qwen3-VL 分片 sha256 | 已完成 | `pt-v2kaero2` | 10-24 20:28 | 4 个全部一致 |
| Q3B-LOAD | Qwen3-VL-8B 当 StarVLA 骨干的 1 卡加载 | 已完成 | `pt-xh7kmkf2` | 10-24 20:50 | 750/750 个张量相同；LM 隐藏维 4096；前向损失 0.721、峰值 22.96 GB；反向放得下，峰值 35.74 GB；predict_action 输出 [1,16,12] 有限 |
| M-qwen3vl8b-base | Qwen3-VL-8B 零样本，705 道（温度 0.8，top-p 0.92，新格式示例） | 已完成 | `pt-4kvvmez2`（验证）、`pt-fsl9tk8p`（测试） | 10-24 20:57 | 成功率：验证 413 道 0.080，测试 292 道 0.086；不合法回答率 0.000；碰撞率 0.051 / 0.038 |
| C-LOAD-1 | Cambrian-S-7B 真权重 1 卡加载 | 已完成（部分） | `pt-b3ztrs33` | 10-24 20:54 | vLLM 能加载能生成，但图像处理器退到了替代品（下载时缓存缺 `refs/main`，已补）；HF 一级失败于同一原因 |
| C-LOAD-2 | 同上，补 `refs/main` 后重跑 | 已完成（部分） | `pt-ifsus27t` | 10-24 21:04 | vLLM：用 SigLIP2 自己的处理器，加载 155 s、显存 47 GiB，贪心回答 `<action>move_forward\|turn_left\|</action>`；HF：加载 30 s、15.7 GiB，但 Cambrian 源码的带缓存生成在这版 Transformers 上报错（训练不走这条路） |
| C-LOAD-3 | 同上，HF 一级改成不带缓存逐步解码，两级都记下逐词对数概率 | 已完成 | `pt-dmkux9k6` | 10-24 21:47 | 两级都正常；HF 和 vLLM 的贪心回答相同（13 个 token），逐词对数概率差最大 0.013、平均 0.002 |
| M-cambrians7b-base | Cambrian-S-7B 零样本，705 道（温度 0.8，top-p 0.92，新格式示例） | 已完成 | `pt-62zkati2`（验证）、`pt-vwp2bl68`（测试） | 10-24 22:15 | 成功率：验证 413 道 0.065，测试 292 道 0.079；不合法回答率 0.057 / 0.051；碰撞率 0.075 / 0.082；评测用时 1,493 s / 1,142 s |
| CSB-LOAD-1 | Cambrian-S-7B 当 StarVLA 骨干的 1 卡加载 | 失败 | `pt-7bkq0842` | 10-24 22:17 | 权重加载正确（765/765 个张量相同），批量 2 的前向报错：Cambrian 源码的层循环只取批里第一条（见第 5 节）；已改走 Transformers 自己的 Qwen2Model 前向 |
| CSB-LOAD-2 | 同上，改后重跑 | 已完成 | `pt-z2qjr3ew` | 10-24 22:29 | 765/765 个张量相同；隐藏维 3584，动作 token 146663；前向损失 0.627、1.43 s、峰值 31.2 GB；反向放得下，峰值 40.9 GB；predict_action 输出 [1,16,12] 有限 |

| M-qwen25vl7b-base | Qwen2.5-VL-7B 零样本，705 道（同上设置；决定 13 的对照行） | 已完成 | `pt-cx6ck4vk`（验证）、`pt-xlornswu`（测试） | 10-25 00:23 | 成功率：验证 0.063，测试 0.075；不合法回答率 0.000；碰撞率 0.082 / 0.076；评测用时 1,999 s / 1,406 s |

三个原始模型的零样本都是每题 1 局，验证集成功率的标准误约 0.013，三者之间的差都在 1.5 个标准误以内。

现在本线在 sco 上没有任务；正式任务等你对决定 14–18 的答复。

决定 7 的"新模型自己的前 20% 门槛"（口径同主线：全失败题组 8 局回报的样本标准差，取第 80 百分位）：Qwen3-VL 冒烟模型（400 条对话、1 个 epoch 的 SFT）10 次更新的 240 个题组里，135 个全失败，第 80 百分位 0.500（中位数 0.338，第 90 百分位 0.574）；全失败组里 51.9% 达到 0.3318（主线标定时是 20%）。同样口径，Cambrian 冒烟模型 0.476，Qwen2.5-VL 冒烟模型 0.428。正式数字等正式 SFT 模型的 RL 记录再算，只报告不用。

## 3. 权重和数据

| 内容 | 路径 | 说明 |
| --- | --- | --- |
| Qwen3-VL-8B-Instruct | `/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | 只读；分片只有 root 能读（sco 任务里读得到）；sha256 已核对 |
| Cambrian-S-7B | `/mnt/umm/shared_model/vagen-lite-models/hub/models--nyu-visionx--Cambrian-S-7B/snapshots/556e72fded09352f6aa4e9c100c9f39e6fe0fba7` | 本线下载，见 3a |
| SigLIP2-so400m-patch14-384 | 同一缓存下 `models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f` | 本线下载，见 3a；Cambrian 的视觉塔和图像处理器按仓库名从这里读 |
| Cambrian 外部源码 | `/mnt/umm/users/yinbaiqiao/cambrian-s` | 只读 |
| 8 月 RL 训过的 Cambrian 检查点 | `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/models/historical/cambrian_*` | 只在第一步测过代码，不是起点 |
| Qwen2.5-VL-7B 原始 | `E/interiorgs_v10_r11_20261016/inputs/base_7b` | 参照行起点 |
| 新格式 SFT 数据 | `/mnt/umm/users/yinbaiqiao/models_runs/sft_v10_newformat_20261024` | C0 的 11,095 条对话改成新格式（`action_names`）；除格式示例那一行外和 C0 相同（逐条核对过）；冒烟子集 `sft_trajectories_smoke.jsonl`（训练 400、验证 40，种子 20261024） |
| RL 题 | `E/interiorgs_v10_r11_20261016/rl_rows/rows.jsonl`，排除 `E/interiorgs_v10_r10_20261015/depth_check/depth_check_summary.json` | 后备题 10,798 道 |
| 评测集 | `models_runs/eval_sets/eval_val_v2`（413）、`eval_test_v2`（292） | 本线自己的结果目录；`tasks.jsonl` 和 `selection.json` 链接到主线的 `E/interiorgs_v10_r12_20261019/eval_*_v2`，主线的结果目录不写 |
| 密集变体汇总 | `E/interiorgs_v10_r12_20261019/analysis/variants/variants.json` | 提交 RL 时再读一次 |

`E` = `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial`。

### 3a. Cambrian-S 权重的来历

| 项 | Cambrian-S-7B | SigLIP2-so400m-patch14-384 |
| --- | --- | --- |
| 仓库 | `nyu-visionx/Cambrian-S-7B` | `google/siglip2-so400m-patch14-384` |
| 修订号 | `556e72fded09352f6aa4e9c100c9f39e6fe0fba7` | `e8e487298228002f3d8a82e0cd5c8ea9c567f57f` |
| 文件清单和 sha256 的来源 | huggingface.co 的 API（经登录节点代理，10-24 19:41:09） | 同左 |
| 下载来源 | hf-mirror.com（不经代理），`HF_HUB_DISABLE_XET=1` | 同左 |
| 下载时间（UTC） | 10-24 19:41:26 起，用时 267.7 s | 接着下，用时 79.6 s，19:47:13 完 |
| 核对 | 每个文件的大小和清单一致，LFS 文件的 sha256 和清单一致，没有多余文件（`verify.json` 的 all_ok 为真） | 同左 |
| 补记 | 10-24 20:58 给两个仓库写了 `refs/main`（40 字节，内容就是上面的修订号）：按提交号下载时 huggingface_hub 不写它，离线按仓库名加载会找不到快照 | 同左 |

记录文件：`/mnt/umm/users/yinbaiqiao/models_runs/cambrian_weights_20261024/{manifest,download,verify,refs}.json`；脚本 `model_matrix/weights/fetch_cambrian.py`。

Cambrian-S-7B 的配置：语言模型从 Qwen2.5-7B-Instruct 来，隐藏维 3584、28 层、4 个 KV 头、词表 152,064；SigLIP2 视觉塔 26 层，每帧 27×27 = 729 个图像特征，每行末尾加一个换行特征，共 756 个位置；没有下一帧预测头。分词器里没有 `<image>`，本线在 RL 和评测里按 8 月的做法运行时加上（编号 151665），训练端和 SFT 里用 `<|image_pad|>`（151655）标图像位置。

## 4. 实验表

| 编号 | 模型 | 起点 | 内容 | 状态 | 结果 |
| --- | --- | --- | --- | --- | --- |
| M-qwen3vl8b-base | Qwen3-VL-8B | 原始 | 零样本，705 道，温度 0.8 | 完成 | 验证 0.080，测试 0.086 |
| M-qwen3vl8b-sft | Qwen3-VL-8B | 原始 | SFT（C0 的数据来源和配方，新格式） | 未开始 | — |
| M-qwen3vl8b-s0 | Qwen3-VL-8B | 本线 SFT | RL 稀疏臂（`v10_p1_S0`） | 未开始 | — |
| M-qwen3vl8b-d | Qwen3-VL-8B | 本线 SFT | RL 密集臂（提交时按汇总选，现在是 V4） | 未开始 | — |
| M-cambrians7b-base | Cambrian-S-7B | 原始 | 零样本，705 道，温度 0.8 | 完成 | 验证 0.065，测试 0.079 |
| M-cambrians7b-sft / -s0 / -d | Cambrian-S-7B | 同 Qwen3-VL | 同 Qwen3-VL | 未开始 | — |
| M-qwen25vl7b-base | Qwen2.5-VL-7B | 原始 | 零样本，705 道，温度 0.8（对照行） | 完成 | 验证 0.063，测试 0.075 |
| M-qwen25vl7b-sft / -s0 / -d | Qwen2.5-VL-7B | 同上 | 同上（参照行） | 未开始 | — |

## 5. 代码和输出

- worktree：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite-models`，本地分支 `models`，从 `origin/main` `00363333` 开出，不跟踪远端。代码整理期间不推 main，本地小步提交。
- 改动都是新文件或默认关闭的新开关；不设这些开关时，原有路径的行为不变（每个都有测试）。主要的：
  - 型号登记 `vagen/models/model_types.py`（`qwen2_5_vl` 默认、`qwen3_vl`、`cambrian_qwen`）；训练链计划字段 `model_type`，`rl-config --model-type`。
  - Qwen3-VL：`model_matrix/sft/lf_qwen3vl.py`、`lf_launcher.py`、`accelerate_fsdp_qwen3vl.yaml`、`check_lf_qwen3vl.py`。
  - Cambrian-S：`vagen/agent_loop/cambrian_concat.py`（RL 拼接路径的展开）；`vagen/models/cambrian_register.py` 的开关 `image_token_alias`；`vagen/ray_trainer.py` 的开关 `trainer.agent_loop_manager`；`data_gen/interiorgs_v10/evaluation.py` 的 Cambrian 分支；`model_matrix/sft/lf_cambrian.py`、`lf_launcher_cambrian.py`、`accelerate_fsdp_cambrian.yaml`、`check_lf_cambrian.py`。
  - 对数概率核对：`rl-config --rollout-log-probs`（vLLM 返回采样 token 的对数概率，训练器逐次写对比文件；更新本身不变）。
  - 选池：计划 `"pool": "idle"`（只用空闲机器，`model_matrix/sco_view.py`，10-24 用）、`"pool": "queue"`（直接排队，暂停文件照守，10-25 起用）；单独的任务用 `model_matrix/sco_job.py`（`--pool queue`）；任务名前缀计划字段 `job_prefix`。
  - 计划字段 `rl_start_model`（跳过 SFT 从给定模型开始 RL，只给冒烟查管线用；正式实验都是先 SFT 后 RL）。
  - 下游：`model_matrix/downstream/`（只读调用下游线的配方和入口，不改下游线的文件）。Cambrian-S 的 StarVLA 骨干：`starvla_cambrian.py`（接口，运行时登记到 StarVLA 的 `get_vlm_model`，其他起点照旧用 StarVLA 自己的接口）、`cambrian_entry.py`（训练入口、加载测试和策略服务都经它启动）、`cambrian_env.sh`。
  - StarVLA 环境缺 Cambrian 源码在导入时要的两个包（`ezcolorlog` 彩色日志、`open_clip` 只给 CLIP 视觉塔用）。没装任何东西：`model_matrix/downstream/shims/` 放了两个替身文件，只有 `cambrian_entry.py` 把它们放到搜索路径的最后（装了真包就用真包）；SigLIP2 这条路不用它们。
  - Cambrian 源码的语言模型循环取 `layer_outputs[0]`，在 Transformers 4.54 以后（解码层直接返回隐藏状态）取到的是批里的第一条。批量为 1 时没问题（RL 去掉填充后总是 1，SFT 每卡每步 1 条，加载测试 1 条）；批量大于 1 就错。StarVLA 接口改走 Transformers 自己的 `Qwen2Model.forward`（同一组层和位置编码），测试里核对过：单条和 Cambrian 循环相同，左填充的 2 条批和逐条相同（Transformers 4.56.1 和 4.57.0 都测）。SFT 的模型类遇到批量大于 1 直接报错。
  - StarVLA 的 Cambrian 接口默认打开梯度检查点（StarVLA 自己的开关 `framework.qwenvl.enable_gradient_checkpointing`，它的别的骨干默认关），算梯度时把模型切到训练模式（Transformers 只在训练模式下做检查点，StarVLA 从不切），加载时核对模型里没有非零的 dropout。
  - 训练链停止条件 `keep_fill_min`（默认关；决定 7：keep 臂到第 `keep_fill_at` 次更新（默认 50）平均每次补进不到这个数就停）。
- 测试：`data_gen/interiorgs_v10/tests` + `model_matrix/tests`，10-25 09:00 全跑 348 通过、3 跳过（跳过的 3 个只在 Qwen3-VL 运行环境里跑，在那里通过）；之后改动的文件各自重跑通过（`test_train_chain.py` 35 个，`test_starvla_cambrian.py` 在主环境和 StarVLA 环境各 13 个）。
- 大文件和任务输出：`/mnt/umm/users/yinbaiqiao/models_runs/`。
- 整理窗口问到下面这些目录，一律回答"不能动"：
  - `VAGEN-Lite-models`、`models_runs/`、`cambrian-s/`；
  - `VAGEN-Lite/.venv-qwen3vl-runtime`；
  - `VAGEN-Lite/models/historical/cambrian_*`；
  - 共享目录里的 Qwen3-VL-8B；`/mnt/umm/shared_model/vagen-lite-models/`。

## 6. 规则

- GPU：bq_tp_mmxbf16 和 bq-ready-* 最高，不动、不问；其次主线的密集奖励实验和 A 组；本线和下游同级，谁先排队谁先上，冲突时记在这里并告诉你；其他扫描最后。10-25 00:38 起本线直接排队（见 1a 执行记录），暂停文件照守。
- 暂停文件 `/mnt/umm/users/yinbaiqiao/downstream_runs/PAUSE_SUBMISSIONS` 存在时本线不提交新任务。
- 从集群外拷东西、下载大文件、安装新环境、调用付费接口、删除文件、转换或导出权重产生新副本，都先问你。
