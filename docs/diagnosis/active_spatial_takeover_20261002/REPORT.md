# Active Spatial 接管核验与下一步

日期：2026 年 10 月 2 日。主要产物快照为 **20:51 UTC**；作业状态查询为 **20:38 UTC**；最终 CPU 复测开始于 **21:04 UTC**。这是项目接管审计，不是新实验结果报告。

**判断：项目已具备运行多轮 VLM PPO、canonical 相机与任务验证、离线评估的工程基础；还没有足够证据证明 canonical 空间控制获得稳定提升，更不能声称通用空间 QA 迁移成功。下一步最有价值的是收束当前 R1 实验与固定评估，而不是增加 backbone、任务类型或全量 reward sweep。**

[项目导航](../../active_spatial.md)给出代码入口。[evidence.json](evidence.json)保存 78 个读取来源的 hash、冻结文件核对、逐 episode 汇总及配置摘录；这些证据中包含会继续增长的 train.log，hash 只对应本次读取快照。

## 研究目标与当前主线

长期问题是“主动控制视角是否能让 VLM 学会空间关系，并迁移到静态空间理解”。当前能执行的最小研究问题是：在固定 H1 相机、真实碰撞、六动作、0.3 米平移、20 度转向、12 个 primitive steps 的条件下，模型能否从初始失败状态达到 canonical Projective 成功。

历史多任务训练提供了 proxy 导航指标上的学习现象，但 task success、RGB 相机、初态成功率、可达性和 PPO 数值链路曾分别出过问题。R1 是在修正这些前提；dense-score S0/S1/S5 是尚未开展的因果干预；Act→QA 是另一个尚未通过数据与观测合同的迁移诊断。三条线不应共享一个“已验证有效”的标签。

## 工作区与约束

- 根仓库 HEAD 为 `19e7467102728def945038a9ef4934bbb9ed7633`；比本地 `origin/main` 超前 72 个提交，本轮没有 fetch，不能据此判断远端实时差异。
- `verl` 为 `fd19159313cc9b0ee644668a321e97642394d8ed`，其 `metric_utils.py` 有既有改动。根工作区有大量已修改和未跟踪文件；审计期间还观察到其他改动进入，未把它们归为本轮成果。
- 从仓库根逐级到 `/` 未发现适用的 `AGENTS.md`；`.agents/` 与 `.codex/` 为空。`ele/AGENTS.md` 属于未编辑的嵌套项目。已读取相关 README、交接和 `.cursor/rules/master-results.mdc`，遵守唯一成绩账本约定。
- 本轮未启动 GPU 训练、推理或渲染，未下载权重、生成正式数据、删除产物、提交 commit、改实验协议或更改已有作业生命周期。执行了 CPU 回归、一个既有 batch 的 CPU 重放，以及已有图片的像素比较。

## 真实进展及证据边界

| 工作 | 本轮核实到的事实 | 不能据此推出的结论 |
| --- | --- | --- |
| 历史 D0 修复后 Qwen V46/V47/V48/V50 | 四个运行配置确有 `calculate_log_probs=true`、`use_rollout_log_probs=true`、`rollout_is=token`、`bypass_mode=false`；逐 episode 重算 step700 的 24 个 ID/OOD 结果，与各自 JSON 汇总相符，已补入[主账本](../../../exps/unified_eval_runs/MASTER_RESULTS.md)。 | 这些是历史 task/eval 协议；不能作为新 canonical 成功，也不能混入旧 ID400 列。 |
| 历史评估分布 | 旧 `test.jsonl` 有 7,790 行，排除 delta 后 7,169；相对 v46 的 9,154 条训练参考集，内容重叠为 0，但按当前 ID 定义只有 3,189 条合格。 | “无精确内容泄漏”不等于“全部是 ID”。旧 `id_test` 含训练未见任务等分布差异。 |
| 静态 EASI | 回收的 V46/V47/V48 step700 各有八项；V50 只有两项，因此 Macro 留空。Cambrian C8/C4 的对应离线 summary 仍全为 `missing`。 | 不以训练完成、目录存在或 completion 文件存在代替模型评估完成。 |
| R1 Projective 支持子集 | `frozen_v1/SHA256SUMS` 的 10 项全部匹配；210 个唯一任务、45 场景，left/right=115/95；与 dev32 的 task/scene 及 local-action60 的 source/scene 无交叉。runtime preflight 的 210 条均 PASS，且绑定当前冻结 train hash。 | 这是支持子集，不是历史 9,154 条多任务训练的等规模复现，也没有自动满足另一条 dense-score pilot 的全部数据门禁。 |
| R1 smoke 与正式训练 | smoke 保存了实际权重变化、checkpoint reload 的 PASS 回执；正式训练已记录到 189，完整 marker 与 HF 索引指向的 7 个非空 shard 在 step50/100/150 均存在。 | 本轮未重载这些大模型，也未完整重算权重 hash；marker 与文件完整性不是新的 load 测试。step250 尚无完成证据。 |
| r5 离线 PPO 重放 | 本轮用原始约 647 MB snapshot 在 CPU 重跑 forensic CLI，结果 `REAL_HISTORICAL_REPLAY_PASS`，见[本轮重放记录](r5_replay.json)。 | 没有模型 forward、reward 干预、backward 或 optimizer step；不支持“S1/S5 更好”。 |
| Dense-score pilot | 9 月 29 日 gate 是 `PILOT_BLOCKED_INFRA`，S0/S1/S5 的正式训练、共同 step0 和梯度干预未运行。 | 新 R1 子集的 PASS 不能自动修改这份旧 gate 或代替其 frozen manifest。 |
| Qwen3-VL | `qwen3vl_active_spatial_smoke_h800_20261002_metricfix` 的 step1 有 actor grad norm 13.2962、有限 pg loss 0.06816，且有完整 checkpoint 元数据；日志末尾有 W&B shutdown `BrokenPipeError`。 | 这是一次训练更新的工程证据；本轮没有 reload 或 canonical 基准，不是长程稳定性或模型优越性证据。 |
| Act→QA | 严格配对的 27 条均为 screen occupancy，GT 为 6 Yes / 21 No；直接读取预测发现 Base 与 Act 都是 27 次 `No.`，accuracy=77.78%、balanced accuracy=50%、差值为 0。 | 不能称作迁移成功，也不能据单任务小样本判定 RL 普遍无迁移。公共问题与 GT 可观测性仍有边界问题。 |

历史 `reports/*plus10pp*` 明确是加分情景，不作为实测依据。9 月组会报告对旧协议学习现象的解释也不能直接升级成 canonical/generalization 结论。

## 正在运行的 R1 应当怎样解读

SCO 只读查询确认 trainer `pt-kbvfmyih` 与 renderer `pt-t7bb75de` 在 20:38 UTC 都为 `RUNNING`，两者各占一个 8×H800 worker。提交记录里的 `SUBMITTED_NOT_YET_TRAINED` 已落后于实际 train.log。此前失败重试留下的 STOP 文件不能代表当前版本作业状态。

当前训练来自 `package_v7`，归档明确排除了无关工作区改动。其 `env.py`、`env_config.py`、agent loop、trainer hash 与今天工作区不同；本轮 CPU 测试检验的是今天工作区，不能把新历史窗口、dataset sidecar 或其他修改追认到该训练中。

`r1_freeze_clean_projective_v0.py` 明确保留历史 v46 recipe；实际 `formal/launch_resolved.yaml` 为 `calculate_log_probs=false`、`rollout_is=null`、`bypass_mode=false`。因此这里的 clean 主要描述数据/相机修复，**不等于 D0 修复后的 PPO 控制组**。这是冻结设计的限制，不应中途开 correction 来“修复”正在跑的实验。

scheduler horizon 保持 700；适配器仅把停止步数设为 250；critic warmup=60，日志首次 actor 更新在 step60。step50 不构成已更新 actor 的 RL 结果。

逐行重算 `formal/validation/*.jsonl`：

| Step | 独立 task 数 | 轨迹数 | 成功轨迹 | 有任一次成功的 task 数 | episode 记录的 invalid_action 均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 32 | 128 | 0 | 0 | 0 |
| 50 | 32 | 128 | 0 | 0 | 0 |
| 100 | 32 | 128 | 1 | 1 | 0 |
| 150 | 32 | 128 | 0 | 0 | 26.56% |

每个 task 重复四次；不能把 128 当作独立样本数。该集合用于开发回归，不能作为独立泛化测试。这四个验证文件的 env_exception 和 renderer_failure 记录均为 0。

训练 entropy 从 step1 的 0.834 增到 step150 的 2.464；step188 为 4.202，step189 又回到 2.263。结合 step150 的格式退化，这是需要定位的稳定性信号，不能仅用某一 batch 的训练成功率解释为学会了 canonical 控制，也不能仅凭这些相关性断言 dense reward 是原因。

## 本轮定位的缺陷与已经完成的离线核验

**QA 合同测试曾失败，最终复测已通过。** 初次检查时，无 scene root 的五条合成样本被标为 `coordinate_unconfirmed`，`private_answer=None`；随后 `generate_paired_qa.py` 用 `label_counts[answer] += 1` 访问只有 Yes/No 的字典，触发 `KeyError: None`，测试报告 errors=5。审计期间工作区另有同期修订：只对 Yes/No 计数，并让缺少场景证据的测试明确期望 None。最终独立合成 fixture 核对为保留五条、全部 None、errors=0，完整 CPU 测试也通过。见[复核记录](qa_fixture_reproduction.json)。本轮没有修改生成器、标签或接受标准，不能将同期功能修复计为本轮整理成果；合法 scene fixture 的正向覆盖仍可加强。

**QA v5 renderer worker 的计数检查误判。** `diagnostic_render_worker.py` 把样本 PNG 和 `contact_sheet.png` 都计入输出，导致一个请求被报成两张输出，worker 以 1 退出。已有 manifest 实际只有一条，renderer summary 为 rendered=1/errors=0。未来修复应按 manifest 中 sample_id 与文件映射核验，而不是把任意 PNG 都计为样本；不能直接把旧 done marker 改成 PASS。

本轮利用已有图片独立核对 v5 A 上游路径与 B QA 路径：同一 K、同一 c2w、512×512，文件 SHA256 相同，MAE/P99/max error 均为 0。两张图片的空间通道标准差约 20.882。这个结果只证明**一个既有姿态的两条渲染路径一致**；C 候选姿态、场景合法性和 QA GT 未因此获得验证。见[已有 RGB 核对](qa_saved_rgb_check.json)。这部分无需重开 renderer。

**安全整理与检查。**

- 增加根 README → 项目入口 → 接管报告/主账本的导航，并注册到 MkDocs。
- 增加固定 CPU 检查入口 `scripts/check_active_spatial_cpu.sh`；不调用 launcher，不自动安装依赖。
- 将 QA 中重复的 unittest 方法名改为 `test_missing_scene_context_withholds_label`，恢复原先被覆盖的一项几何检查；未改断言或应用行为。
- `.gitignore` 仅新增两个本地 Qwen3-VL 虚拟环境路径，文件保留在磁盘；没有搬迁脚本、数据、checkpoint 或日志。
- 修正文档中将 success 简化为 `final_score >= threshold` 的旧描述，保持与当前 evaluator 读取环境成功值一致。

检查前 68 passed / 1 failed；恢复被覆盖测试后 69 passed / 1 failed；同期 QA 修订后的最终复测为 **70 passed / 0 failed / 0 skipped**，耗时 24.39 秒。其中 pipeline 30、reward 12、PPO replay 12、R1 5、QA 11。三轮结果与最终源码 hash 均在 [cpu_checks.json](cpu_checks.json)。这些测试不覆盖 GPU kernel、worker 归档复现、真实 RGB 语义标注或长期训练。

新增文档的相对链接、JSON、shell 语法、MkDocs YAML/nav 和本轮编辑的 whitespace 检查均通过。环境未安装 MkDocs，本轮没有安装依赖或构建发布站点。`MASTER_RESULTS.md` 位于被 Git 忽略的本机 `exps/` 下，已更新磁盘文件；本轮未执行暂存或提交，普通 clone 不会自动得到该账本及实验产物。

## 有限的后续安排

以下是下一轮执行顺序；涉及新 GPU 工作或行为修复的内容是计划，本轮没有执行。优先级按“是否能改变主线判断”排列。不要同时展开新的 backbone、reward 矩阵和数据扩展。

| 顺序与预算上限 | 先做什么与为什么 | 具体做法 | 进入下一步的条件 |
| --- | --- | --- | --- |
| 1，半天 CPU 维护 | 关闭剩余 QA worker 计数误判并固化当前代码身份。空标签异常已在同期修订中通过复测，不再重复修复。 | worker 按 manifest 映射计数，用已有 A/B 文件作回归；补合法场景 fixture，明确 unknown/拒收统计，C 仍留未完成。将根仓库 dirty diff、VERL diff、所需未跟踪源码和 frozen/source archive hash 分别留清单。 | 维持当前 70 项全过；合法 scene 和 contact sheet 计数回归通过；没有 GT、动作、reward 或 success gate 改动；已有 A/B 仍逐像素相同。 |
| 2，半天结果收束 | 优先收完现有 R1，确定它实际做了什么。现在已经有足够信号说明不应自动续跑或扩大矩阵。 | 只读跟踪现有 job 至其既定 step250 或终止；保存实际终态和失败原因。按 task 对 step0/100/150/250 的验证作配对汇总；保留数值和格式轨迹。仅在 step250 的 COMPLETE、索引与 shards、验证文件齐备时写“完成”；不重启，不延长，不借更新工作区恢复运行。 | 终态明确；冻结数据、运行归档、scheduler/终点、checkpoint/验证一一对应；区分 dev 指标与 independent 指标，区分数据修复与 PPO recipe。若任务失败，完成失败记录即进入判断，不无限重试。 |
| 3，一次有界评估，需另行授权 GPU | 回答 canonical 动作控制是否真的改善，优先于全 ID/OOD/EASI sweep。 | 使用永久 eval-only 的 local-action60/120 状态，固定 policy 输入、解析、K、碰撞与成功 gate，比较 pinned Base、R1 step100、step250。每模型每状态一次，仅评分解析序列首动作；最多 360 次 decision。每个模型运行前检查 package/protocol hash；保留无效输出在分母中。step250 若不存在，直接转故障诊断，不自动补跑。 | 120/120 状态与训练 source/scene 隔离；报告正确首动作集合命中率、无效比例、parent 级配对和场景分层。预先固定按 parent 重采样的 95% 区间；只有相对 Base 命中率增量的区间下界大于 0、并核清格式变化的贡献后，才讨论扩大；否则保留负结果并进入单一故障诊断。 |
| 4，只选择一条后续干预 | 用前一步结果决定资源去向，避免把未闭合原因叠在一起。 | 若没有 canonical 改善，先以已冻结 batch 检查 reward、GAE 和实际 PPO anchor；r5 重放已通过，不再重复修其排序。若要验证 D0 correction，另立版本的单因素对照，保留当前 run 身份。dense S0/S1/S5 必须重新满足其自身 formal manifest、共同 step0、resolved-config 差异和梯度门禁；QA-SFT 则等公共问题/GT/可观测性合同闭合后再进入。 | 写清一个待检验假设、唯一变量、固定输入、失败退出条件后，才申请一个小试验。未满足 gate 就停在对应缺陷；不直接上三组 pilot，不扩大数据，不新增模型迁移。 |

local-action60 的场景独立于 R1 开发，且本次 train 已避开这些场景；它们对历史 v46 曾是已见场景。它是有界动作回归，不替代完整闭环成功率或广泛 OOD 结论。

本轮交付是可追溯的判断、入口和检查，不是对继续训练的授权。现有作业的后续变化应追加新的观察时间，不覆盖这份快照。

本轮改动归属、同期改动说明及静态检查记录见 [delivery_manifest.json](delivery_manifest.json)，便于在已有脏工作区中单独审阅。
