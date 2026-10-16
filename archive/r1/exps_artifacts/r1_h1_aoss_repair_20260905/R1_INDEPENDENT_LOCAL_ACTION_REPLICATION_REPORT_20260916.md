# R1 独立场景局部动作复验报告（2026-09-16）

## 结论摘要

本轮按冻结协议完成了 60 个 parent source、10 个未参与 R1 开发／调参场景上的局部动作复验。每个 source 构造一个精确 `d*=1` 和一个精确 `d*=2` 状态，共 120 个状态；两个模型各只查询一次并只评分解析序列的第一个动作，共 240 次 policy decision。没有训练、数据再生成、prompt 修改或完整轨迹重跑。

局部低命中在这组 R1 未见场景上方向一致地复现：

| 模型 | d*=1 | d*=2 | 总计 |
|---|---:|---:|---:|
| Qwen2.5-VL-7B-Instruct pretrained | 6/60 (10.00%) | 10/60 (16.67%) | 16/120 (13.33%) |
| v46@250 | 8/60 (13.33%) | 11/60 (18.33%) | 19/120 (15.83%) |
| 六动作均匀随机期望 | 12.67/60 (21.11%) | 13.83/60 (23.06%) | 26.50/120 (22.08%) |

两模型都低于冻结状态集上的均匀随机期望。状态级配对为：双方命中 9、仅 pretrained 命中 7、仅 v46 命中 10、双方未命中 94。按 parent source 汇总两种距离的命中数，v46 胜 8、pretrained 胜 5、平局 47；37/60 个 parent 上两模型均为 0/2。因此本轮支持“局部动作选择本身已经很弱”，而不仅是长程闭环控制失败。

这一结论有明确边界：10 个场景没有参与 R1 selector／阈值调优，但全部属于 v46 训练曾见场景；它们是独立于 R1 开发的回归场景，不是 v46 scene-OOD。状态分布也明显偏向横移 oracle，不能据此给出广泛泛化结论或宣称 v46 与 pretrained 等价。

## 冻结范围与身份

- 有效实验代码：`e12e0ee53fe4468ab5facd085134bbd72ceab087`。
- worker 注入归档 SHA256：`6a61cdc98c7d5bff57cf870fce4a7d848c733e7f61145dbd5e0383008c93de8f`。
- 资源：SCO `pt-tf6lhpce`，`pool=h800`，单 NVIDIA H800 80GB，worker `pt-9ca00a3cdc1b446da2d5963ca8a7e475-worker-0`。
- 两模型在同一 worker 上顺序加载；沿用已验证权重和 `no_concat`、当前 RGB、正式 pose、原任务文本、原解码及动作解析契约。
- 模型权重清单 SHA256：pretrained `46f05ffcc6127a4caa9a3e8c11ddf298b9a5263c8680afe6b5d017ea91702c8b`；v46@250 `1fae9d370ade47ed79970cddbb27edab38c513526a29508bd1d3e7796e23374f`。
- 候选范围：18 个 R1 未见场景、302 个非 train source；最终确定性选出 60 个 parent、10 个场景。
- `r1_development_seen=false`：60/60；`v46_training_scene_seen=true`：60/60；`v46_training_exact_source_seen=false`：60/60。场景见过与 exact source 见过分开保存，未将其误称为 OOD。
- split（parent 分母）：`id_test=52`、`ood_geometry=6`、`ood_category=1`、`ood_template=1`。这一不均衡是本轮最重要的覆盖限制之一。

最终 10 个场景及 parent 数：

| scene | parents | states |
|---|---:|---:|
| 0057_839904 | 4 | 8 |
| 0244_841000 | 6 | 12 |
| 0248_840834 | 4 | 8 |
| 0257_840812 | 2 | 4 |
| 0260_840805 | 11 | 22 |
| 0306_840556 | 10 | 20 |
| 0308_840552 | 4 | 8 |
| 0323_840504 | 7 | 14 |
| 0325_840494 | 2 | 4 |
| 0333_840470 | 10 | 20 |

## 精确距离真值与 Policy 隔离

状态真值由正式 runtime 对六个 primitive actions 完整枚举到深度 2 得到，搜索目标是任意 canonical-success 状态：

- `d*=1`：初态不成功，至少一个一步动作成功；
- `d*=2`：所有一步合法动作均不成功，至少存在一条两步成功路径；
- 保存全部最优首动作集合，不以一条生成证书的首动作代替真值；
- collision/runtime 合法性决定可展开状态，没有用 RGB observability 过滤 runtime 合法中间状态；
- oracle 动作集合、证书、terminal pose 和额外 3D 真值未进入 policy 输入；
- 模型每状态只查询一次，只评分原解析序列的第一个动作；不从后续动作挑选合法或正确答案。

302/302 候选 source 完成 geometry 审计，0 implementation error；101 个 source 至少有一个精确局部状态，84 个同时具有 `d*=1/2`。官方 RGB 资格检查了 63 个 parent，按冻结顺序取得 60 个双距离 parent；最终 120 个状态的初始 RGB 均通过冻结质量门禁。

## 模型结果与逐 Source 配对

### 总体与分层

| 分层 | states | pretrained hit | v46 hit |
|---|---:|---:|---:|
| id_test | 104 | 15 | 18 |
| ood_geometry | 12 | 1 | 0 |
| ood_category | 2 | 0 | 1 |
| ood_template | 2 | 0 | 0 |

每场景命中如下（pretrained / v46）：

| scene | states | pretrained | v46 |
|---|---:|---:|---:|
| 0057_839904 | 8 | 1 | 1 |
| 0244_841000 | 12 | 1 | 3 |
| 0248_840834 | 8 | 1 | 1 |
| 0257_840812 | 4 | 1 | 0 |
| 0260_840805 | 22 | 1 | 3 |
| 0306_840556 | 20 | 4 | 4 |
| 0308_840552 | 8 | 1 | 1 |
| 0323_840504 | 14 | 2 | 3 |
| 0325_840494 | 4 | 0 | 1 |
| 0333_840470 | 20 | 4 | 2 |

v46 比 pretrained 多命中 3/120，但逐 source 的净差和分母都很小；本轮不能据此声称 RL 提升或伤害了局部动作能力。

### 基础设施与非法动作

- 两模型合计 240 次决策全部完成，基础设施错误 0。
- pretrained：非法／空动作 0，collision 0。
- v46：非法动作 1（`id_test:4132` 输出 `move_down`，按冻结解析规则保留为模型错误），collision 0。
- 所有请求保留 raw completion、完整解析列表、首动作、pose、正式 RGB、canonical 前后指标和错误账目。

## 随机／固定动作基线与动作偏置

独立集的 oracle 最优动作 membership（一个状态可有多个最优动作）为：

| action | optimal membership | 固定该动作命中率 |
|---|---:|---:|
| move_backward | 0 | 0.00% |
| move_forward | 14 | 11.67% |
| move_left | 57 | 47.50% |
| move_right | 62 | 51.67% |
| turn_left | 11 | 9.17% |
| turn_right | 15 | 12.50% |

oracle 的 fractional distribution 为：forward 4.083、left 51.000、right 57.250、turn-left 3.083、turn-right 4.583、backward 0。该局部状态构造明显以横移解为主，因此固定 `move_right` 的 62/120 只是一项无需推理的描述性基线，不能包装成可泛化策略。

模型动作分布与 oracle 方向相反地偏向旋转：

- pretrained：turn-left 52、turn-right 51，旋转共 103/120（85.83%）；横移共 13/120。
- v46：turn-left 28、turn-right 61，旋转共 89/120（74.17%）；横移共 22/120；另有 1 个非法动作。

动作偏置能够解释相当一部分低命中：oracle 大多要求横移，而模型大多旋转；两模型均低于均匀随机期望，且远低于固定横移的诊断上限。但由于状态集本身偏横移，这仍是相关性证据，不是对模型内部原因的因果证明。

与旧 Dev32 局部诊断相比，方向一致：旧集上 pretrained 5/64（7.81%）、v46 8/64（12.50%）、随机期望 14.5/64（22.66%）；独立集命中略高，但仍低于随机期望。旧集同样由 `move_left` 固定基线占优（38/64），说明两批开发诊断共享“模型过度旋转、oracle 更偏平移”的现象。

## 动作后的 Relation、入框与 Collision

| 指标 | pretrained | v46 |
|---|---:|---:|
| 一步后 canonical success（即 d*=1 hit） | 6 | 8 |
| relation true：before → after | 68 → 69 | 68 → 71 |
| relation margin improved / worsened | 80 / 40 | 69 / 50（另 1 非法） |
| relation margin mean delta | -3.801 px | -1.814 px |
| relation margin median delta | +0.577 px | +0.417 px |
| inside-frame gate：before → after | 120 → 69 | 120 → 75 |
| 丢失 inside-frame | 51 | 45 |
| min inside fraction mean delta | -0.2312 | -0.1969 |
| collision | 0 | 0 |

模型动作有时改善 relation margin，却频繁损害目标入框；这与过度旋转的动作分布一致。由于 d*=2 首动作未必即时改善 canonical score，单步 margin 变化只用于轨迹诊断，不被当作最优动作真值。

代表性逐状态证据：

- `id_test:772`，`d*=1`，最优集合含 `move_left`：两模型均选 `move_left` 并命中；对应初始和动作后 RGB 清晰、几何变化与横移一致。
- `id_test:4843`，`d*=1`，最优集合含 `move_right`：pretrained 选 `turn_right`，v46 选 `move_right`，为 v46-only。
- `id_test:5507`，`d*=1`，最优集合含 `move_right`：pretrained 选 `move_right`，v46 选 `turn_left`，为 pretrained-only。
- `id_test:2425`，`d*=1`，最优集合含 `move_left`：两模型均选 `turn_right`，双方失败；动作后视野明显向右偏移。

人工抽查这些轨迹确认图片来自正式 renderer，动作前后变化与记录动作一致，没有黑帧、错帧或复用其他 pose。与此同时，`id_test:4843` 和 `id_test:5507` 的画面存在高亮、低纹理白色物体，虽然通过冻结 observability 门禁，目标的人眼辨识质量弱于 `id_test:772`。本轮没有据此修改门禁或挑除样本；该现象作为评估解释限制保留。

## 运行成本与无效尝试隔离

- 有效 v3 worker 阶段约 50 分 21 秒；同一持久任务连同启动调试和两次提前停止的无效尝试，总占用约 2 小时 25 分。
- policy 纯生成耗时合计：pretrained 89.431 秒（均值 0.745 秒／状态，p50 0.687，p95 0.935）；v46 89.305 秒（均值 0.744，p50 0.721，p95 0.980）。总墙钟还包含模型顺序加载、hash、renderer reset 和共享存储 I/O，不能等同于有效推理时间。
- v1 在模型推理前发现 candidate cap 声明为 4 但未实际执行，已标记 `INVALID_ATTEMPT`；v2 在 pretrained smoke 前发现 policy rows 缺 canonical runtime selector metadata，已标记 `INVALID_ATTEMPT`。两次都没有模型决策进入正式统计。
- v3 修复元数据后复用了指纹完全一致的 geometry／RGB checkpoint；重新生成 policy manifest。worker 上 6/6 单元测试通过。
- 本任务 H800 worker 已停止，SCO job 状态为 `SUSPENDED`；未操作其他服务。

## 可回答的问题

### 局部低命中是否跨独立场景复现？

是，按运行前冻结的开发诊断判据已经复现：60 parents、10 个 R1 未见场景、0 infra error；两个模型在 d*=1 和 d*=2 上都不超过 25%，总体均低于六动作随机期望。不过这些并非 v46 scene-OOD，且 split／oracle action 分布不均衡，因此结论限定为“R1 未见场景上的功能性复验”。

### 动作偏置能解释多少？

能解释相当一部分，但不能精确归因全部失败。oracle membership 中 left/right 占 119 次，而模型分别有 85.83% 和 74.17% 输出旋转；固定横移基线显著超过模型。然而固定动作基线利用了该诊断集的分布偏斜，不是可部署策略，也不能证明视觉理解是唯一根因。

### 是否已有依据进入 train-only 首动作监督 pilot？

有依据进入**严格限定的功能性 pilot 设计与实施**，没有依据直接扩大训练或宣称泛化改进。建议下一步仅使用与本 60-source eval 完全隔离的 train-only source：

1. 用相同 formal runtime 枚举构造 `d*=1/2`，标签保存全部最优首动作集合，采用集合值监督而非任意单证书标签；
2. 按 action、distance、scene、category 分层并平衡，避免把本轮横移偏斜学成新的固定动作偏置；
3. 当前 60 parents／120 states 永久保持 evaluation-only，不向 policy 暴露 oracle、证书或 terminal；
4. 用同一模型、prompt、camera、runtime、解码与解析协议评估，预先冻结主要指标：set-valued first-action accuracy、每动作混淆、invalid、collision、入框损失；
5. pilot 只验证首动作监督能否超过随机及固定动作基线，不夹带 LR/KL/aux 扫参，也不把结果包装成完整导航能力提升；
6. 正式泛化结论仍需 scene-disjoint 且 split／action 更均衡的独立评估集。

## 文件与 SHA256

有效根目录：

```text
exps/vagen_active_spatial/r1_h1_aoss_repair_20260905/
  r1_independent_local_action_eval60_v3_20260916/
```

关键文件：

- `frozen_scope.json`：`650b567c05b61ede4a52d3595ce0b7a61df2d37b2e8dcbf879d4c5a28ed507c1`
- `frozen_input/independent_local_action_protocol.json`：`505af81224568c610e707b4e540f517aa08c8572be0727a4d697b812210c02ee`
- `frozen_input/local_policy_rows.jsonl`：`ad7636b13f686e8898826b8ad586fba12a07dd6d3e5c00880829af4f2463454d`
- `frozen_input/local_state_audit.jsonl`：`c6aeab0b122c4f051efa8d8c98ea3b770a5541f97945676efb85422830382a60`
- `paired/summary.json`：`b6dd8666472e81d10332e6c51ac2491641ccac416aac4291b35ddcc644afc401`
- pretrained ledger：`d21a240eba28e9fbca4126d87279e68f10c652272715c4ec1d68434e4e4b9e7a`
- v46 ledger：`fe4df006d533ff193345f02ccac8057bda7eb570ef645871e7172f866a1fc760`
- `SHA256SUMS`：`d1ed5b50b82e76c026c27cd34ffc7a9dbf07c71e8466ecbfbfef343e2fda9aa4`

`SHA256SUMS` 覆盖 763 个文件；完整 `sha256sum -c` 为 763 条 OK、0 FAILED。

## 最终停止状态

本轮独立场景局部动作复验已完成。没有生成训练数据、没有训练、没有重跑 Dev32 完整轨迹、没有修改 generator／prompt，也没有扩大 Medium 生成。下一步应是先冻结一个 action-balanced、train-only 的首动作监督 pilot 方案及其 scene-disjoint 验收集；本报告不自动启动该 pilot。
