# E3VS-Bench 试点闸门：事先登记（PREREG_E3VS_PILOT）

- 状态：**草稿，等用户确认后由集成者提交**（git commit，分支 viewpoint-downstream）。提交以后本文件就冻结，之后的任何改动只能记进第 16 节的偏差日志。
- 草稿写于 2026-11-08 10:43 UTC（`date -u`，集群登录节点）；2026-11-08 13:53 UTC 按跨包审查改过第 5、7、12 节和附 A（auditfix）。
- 第 5 节"起点卡在碰撞余量里"已定为 (ii)（编排 2026-11-09 按用户 2026-11-08 的"其他的按你推荐"选定；用户可在 J5 之前改回 (i)）。E3VS 下载要等磁盘线腾出空间（用户 2026-11-09 的磁盘例外只覆盖 WalkerBench 和 TVR），所以本文件在准备 J5 之前再提交。
- 本文件提交的 git sha 记进 e3vs-prep 和三个 e3vs-start 作业的 `prepared.json`，也记进每次评测的 manifest。
- 守门规则：本文件没提交、或有未提交的改动时，不准备任何 vpd-e3vs-* 作业。检查只看本文件：`git log -1 -- vpd/PREREG_E3VS_PILOT.md` 不为空，且 `git status --porcelain -- vpd/PREREG_E3VS_PILOT.md` 为空。
- 依据（按优先级）：
  1. `eval_external_assets/vpd/plan/DECISIONS.md`，特别是 B2、B3、B7、B8、B12、B14、B16、B18、C2、C4；
  2. `plan.md` §3.8；
  3. 报告 `eval_external_assets/downstream_search_20261108/REPORT_downstream_search_v2.md`（下称"报告 v2"）§3.2、§4.2–§4.8。
- 写法：数字后面括号里是出处。没核实的标"未核实"，推断标"推断"，我补的方案标"我的建议"。
- 路径简写：`OUT` = `/mnt/umm/users/yinbaiqiao/eval_external_assets/vpd`；`WT` = `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd`。

---

## 0. 结论先说：闸门一览

试点只回答一件事：E3VS 能不能当下游微调任务用。三个科学条件：学得会、格式有效、没有饱和。另加一个机制检查。**闸门不看起点之间的差。**

| 闸门 | 一句话 | 阈值（详见第 12 节） | 不过时怎么办 |
|---|---|---|---|
| **E0** 机制 | 流程本身是对的 | 渲染一致性和碰撞模型验证都通过；所有评测有效；0W 的代码 sha 在各起点间相同 | 是工程问题，不算科学失败。修好后重跑受影响的部分，记偏差日志 |
| **E1** 学得会 | 至少一个起点 100 档学会了 | 至少一个起点：100 档主成功率的 Wilson 95% 下界 > 盲走下限（B8 定义，第 10 节） | 判为"≤100 条示范学不会"，不进全量；要不要改设计由用户定 |
| **E2** 格式 | 约束解码下格式全对 | 每个起点的 100 档约束评测：回复 100% 匹配正则，没有日志的 episode 为 0 | 先按 E0 查实现；不是实现问题就不进全量，交用户 |
| **E3** 没饱和 | 100 档没有贴近上限 | 每个起点 100 档主成功率 < 85% | 判为"已饱和"，不按现方案进全量；**不自动改档位**，交用户 |

其他规则：
- 试点用种子 **1001**（同时是数据顺序和训练的种子）。确认轮用种子 1、2、3（DECISIONS C4）。
- 试点数据不进确认分析（pilot data are excluded from the confirmatory analysis）。
- 所有试过的闸门，过没过都列进最终报告（报告 v2 §4.8(1)）。
- E3VS 是近迁移任务（报告 v2 §3.2"权重"），在最终判据里不和 TVR 等权。

---

## 1. 目的和范围

1. **试点是什么**：一个闸门试点。它检验 E3VS 能不能当"带视角变化的下游微调任务"（报告 v2 §3.2、§4.8(1)）。
2. **不是什么**：不检验"动作训练比 QA 训练更好迁移"。试点不比较起点。测试集只有 377 个 episode、21 个场景，按场景整组 bootstrap 的置信区间约 ±5 个百分点（推断，报告 v2 §3.2 不利因素 5），只能当闸门，不能当证据。
3. **规模**：3 个起点 × 0 档（raw 和 0W）+ 100 档。每个起点 4 次评测（第 9 节），共 12 次。

## 2. 数据（冻结）

- **基准**：E3VS-Bench，https://github.com/K0uya/E3VS-BENCH ，固定在 commit `bb7e603ccdcf2dd77fbeedd15660a09310a8a881`（"first commit"，2026-06-04）。代码 MIT，标注 CC BY 4.0。
- **episode 文件**（md5 由 `vpd/e3vs/__init__.py:check_repo` 每次运行时断言）：

| 切分 | 文件 | md5 | episode | 场景 |
|---|---|---|---|---|
| train | `dataset/episode_info_train.jsonl` | `d21ba21fafb1846699bd8d2779f56a40` | 1,406 | 68 |
| val | `dataset/episode_info_val.jsonl` | `6a6df2992f3f9f549738a0b8f0718288` | 231 | 10 |
| test | `dataset/episode_info_test.jsonl` | `c537eb6100ec31b371091258b3533737` | 377 | 21 |

  - 三个切分的场景互不重叠（`vpd/tests/test_e3vs_geometry.py` 核实）。
  - test 的题型：Object Search 116、Counting 100、Object Attribute 55、Object State 44、Spatial Reasoning 32、Context-guided Search 30。
  - 每个起点都是该 episode 的不可答视角之一；所有位姿都没有 roll（测试核实，2,014 个 episode 的全部位姿）。
- **用途**：
  - test：闸门只看 test 的 377 个 episode；
  - val：只做开跑前的合理性检查（BFS 复现、冒烟），不挑模型、不调参；
  - train：示范和热身。
- **splat**：`GaussianWorld/scannetpp_v2_mcmc_3dgs`，每个场景 `<scene>/ckpts/point_cloud_30000.ply` 和 `cfg.yml`，99 个场景共 35.05 GB。
  - 由 `vpd/e3vs/download_splats.sh` 下载（要用户同意，见 `vpd/e3vs/USER_STEPS.md`）。
  - 下载时记下数据集的 revision（`OUT/e3vs/data_revision.txt`）和每个文件的 sha256（`SHA256SUMS`）。每次评测的 manifest 记 PLY 的 sha256。
  - 许可：只限非商业研究；渲染图不对外分发。

## 3. 起点（试点只跑三个）

| 名字 | 评测 view 代码 | 说明 |
|---|---|---|
| **Q3B** | `qwen3vl-8B` | 原始 Qwen3-VL-8B-Instruct |
| **Q3S** | `c0q3_sft_e8` | C0 动作 SFT，第 8 轮，1,232 步 |
| **E2** | `sft_m3e2_e10` | QA SFT（P0+P1），第 10 轮，3,010 步 |

- 路径以 `vpd/starts.json` 为准（WP-CORE），和 `PREREG_WB_PILOT.md` 第 3 节相同。注意力实现、bf16、处理器像素上限的断言也一样（DECISIONS B10、B16）。
- 512x512 的图有 262,144 像素，低于 `max_pixels` = 1,024,000，处理器不缩放。按 Qwen3-VL 每 32x32 像素 1 个 token 算，每张图 16x16 = 256 个 token，加 2 个边界 token（推断，E0 里用 `image_grid_thw` 核实，见第 12 节）。

## 4. 环境（冻结）

- **几何**：完全按官方 `e3vs/environments/camera_geometry.py`。`vpd/e3vs/geometry.py` 是它的纯 numpy 版本；测试在 1,000 条随机动作序列上和官方模块逐步比较，最大绝对误差 < 1e-9。
  - 11 个动作：move_forward / move_backward / move_up / move_down / move_left / move_right / turn_left / turn_right / look_up / look_down / stop。
  - 平移 0.25 m，转动 30°。`action_coordinate = camera`：前后沿完整的三维视线方向；左右沿水平的 planar_forward × up；升降沿世界 Z；turn 绕世界 Z；look 绕相机自身的右轴 c2w[:,0]。
  - 评测里没有俯仰限制，和官方一样。
- **步数**：最多 25 次决策，STOP 也算一次（官方 `max_steps` = 25）。
- **碰撞**：官方 `restrict_collision` 规则。平移的线段撞上障碍时跳过这一步：位姿不变，步数照算，下一条提示写 `Last action result: Collision occurred. You remained at the same position.`。转动永远不算碰撞。障碍用第 5 节的碰撞模型。
- **渲染**（WP-E3VS-B `render.py`）：
  - gsplat 1.5.3；四元数按加载器的 **WXYZ** 原样传入（`gs_data.rotations`），**绝不**用 `get_rotations_xyzw()`；
  - opacity 取 sigmoid，scale 取 exp；SH 阶数由 f_rest 的个数定；`packed=False`；near 0.01、far 1e10；结果截到 [0, 1]；
  - 512x512，焦距 W/2，主点在中心（90° 视场）。
- **图像传递**（我的建议）：评测把渲染结果编码成 PNG（无损，和官方保存的格式一样）发给 vLLM；训练数据用同样的 PNG 字节。

## 5. 碰撞模型（冻结；用前先验证）

- **为什么要自己做**：官方用 ScanNet++ 网格 `mesh_aligned_0.05.ply` 加 Open3D 光线投射。我们没有网格（要另外申请），所以用 splat 深度建占据栅格代替（报告 v2 §3.2）。它比官方松还是严，没有测过（未核实）。
- **怎么建**（WP-E3VS-B `occupancy.py`）：
  - 每个场景大约 180 个位姿：每隔 5 帧取一个 iPhone 位姿，再加每个 episode 的 27 个模板视角；
  - 在这些位姿上渲染期望深度和 alpha；alpha > 0.5 的像素反投影进 5 cm 的体素，记命中次数；
  - 一个体素命中 ≥ 3 次就算占据。
- **iPhone 位姿的相机约定**：建栅格用的 iPhone 位姿按哪种约定（OpenCV 或 OpenGL）解释，由 e3vs-prep 作业里的渲染一致性运行定：取它在 iPhone 位姿上和 iPhone RGB 比较时更接近的那种（`occupancy.py --convention auto`）。一致性运行没给出 iPhone 的结果（文件缺失或 iPhone 部分报错）时，prep 作业在建栅格这一步停下，交用户：重跑一致性检查，或者由用户指定约定。用了哪种、出处是什么，记进 `validation.json` 的 `iphone_convention`。
- **怎么判碰撞**：线段 p0 → p1 + 0.05·u（u 是单位方向）的 1 个体素范围内有任何占据体素，就算碰撞。
- **起点卡在余量里（已定：(ii)；附 A 第 10 条）**：按上面的字面规则，起点离占据体素不到 1 个体素的 episode，往任何方向平移都算碰撞，只能转动。现在改成官方式的跳过：沿线段检查时，离起点不到 1 个体素（`START_SKIP_VOX = 1`，t < 0.05 m）的那一段不检查，只看前方（`occupancy.Grid.collides`，测试 `test_collides_skips_the_start_margin`）。盲走下限、BFS 示范和闭环评测用同一个规则。`validation.json` 照常报 `pooled.starts_by_split.*.near_frac`，只作描述。
- **验证阈值**（必须在任何模型评测之前通过，写进 `OUT/e3vs/occ/validation.json`）：
  - iPhone 轨迹的相邻帧线段里，被判碰撞的比例 < 1%；
  - 模板视角落在占据体素里的比例 < 1%。
- **验证不过**：停下来交用户。阈值（5 cm、3 次、1 个体素、0.05 m）只能在第一次模型评测之前、只按这两个验证指标调一次，调整要记进偏差日志。之后冻结。
- **同一个模型用三处**：BFS 示范、盲走下限、闭环评测。
- **渲染一致性**（我的建议的阈值，E0）：
  - 我们的渲染器和官方 `load_ply_model` + 渲染在相同位姿上比较，PSNR ≥ 40 dB；
  - 再在 iPhone 位姿上和 iPhone RGB 比较，分别试 OpenCV 和 OpenGL 两种相机约定，记下哪种更接近（PSNR/SSIM）。PSNR 数值只作描述；但选出的约定决定碰撞栅格怎么用 iPhone 位姿（见上面"iPhone 位姿的相机约定"），没有这个结果 prep 就停。
  - gsplat 要不要现场编译、用了多久，记进 manifest（DECISIONS B18）。

## 6. 成功定义和指标（冻结）

- **主成功**：最终位姿离任一可答视角在 0.25 m 以内，**并且**视线方向夹角 ≤ 20°（两边都含等号）。最终位姿 = 发 STOP 时的位姿；没发 STOP 的，取第 25 次决策之后的位姿（报告 v2 §3.2）。
  - 实现：`vpd/e3vs/geometry.py:success`。
  - 起点本身永远不算成功：2,014 个 episode 的起点全部不满足（测试核实）。
  - 这个容差下，可答和不可答视角 0% 冲突（报告 v2 §3.2）。
- **次要指标**（都报，都不进闸门）：
  - 要求 STOP 的成功率（只有发了 STOP 的才可能成功）；
  - 停止率；碰撞率（被跳过的平移 / 全部决策）；平均步数；
  - SPL：相对同一碰撞模型下 BFS 最优步数（`bfs_demos.py` 的 `opt_len.json`）；BFS 没解出来的 test episode 不进 SPL 的分母，个数单独报；
  - 到最近可答视角的距离和夹角；
  - 解析有效率：严格（整条回复匹配第 7 节的正则）和宽松（宽松解析器没有退回默认动作）两种。

## 7. 输出格式和解码（冻结；DECISIONS B12、B3）

- **统一的中性格式**：所有起点、训练和评测都一样：

  ```
  THOUGHT: <可以为空>
  ACTION: <动作名>
  ```

  示范和热身的回复是 `THOUGHT: \nACTION: <动作名>`（THOUGHT 为空，和 WalkerBench 的主口径一致，DECISIONS C5）。
- **主评测：不约束解码 + 宽松解析器**（DECISIONS B12；报告 v2 §3.2、§4.5）。
  - 解析器照官方 `agents/base_agent.py:_parse_response_to_action`（非 think 模式）：
    1. 从后 5 行里找 `action\s*:\s*([a-z_]+)`（不分大小写），名字必须是 11 个合法动作之一；
    2. 找不到就在全文找同样的模式；
    3. 还找不到，就看最后 200 个字符：有 `stop` 就取 stop，否则取第一个出现的动作名；
    4. 都没有就退回 `move_forward`，标 `error=True`。
  - 宽松解析率 = 没有退回默认动作（`error=False`）的回复所占的比例。
- **次要评测：约束解码**（THOUGHT 为空）。每一步通过 `extra_body={"structured_outputs": {"regex": R}}` 传入：

  `THOUGHT: \nACTION: (move_forward|move_backward|move_left|move_right|move_up|move_down|turn_left|turn_right|look_up|look_down|stop)`

  它是 E2 闸门的机制检查对象。
- **其他**：贪心（temperature 0）；`max_tokens` 128（官方 Qwen3-VL 配置的 `max_output_tokens`）；每次请求只带当前一张图。
- **vLLM**：0.11.0，`PY_Q3` 环境；`gpu_memory_utilization` 0.80（vLLM 和 gsplat 同在一个作业里）；seed 0；enforce_eager；`--generation-config vllm`；`--mm-processor-kwargs '{"max_pixels": 1024000, "size": {"longest_edge": 1024000, "shortest_edge": <模型自己的值>}}'`（DECISIONS B16）；结构化输出后端 xgrammar；每个请求最多 1 张图。
  - 像素上限要两个键：`size.longest_edge` = 1,024,000 才真正生效，`shortest_edge` 取模型 `preprocessor_config.json` 里的值（Qwen3-VL 是 65,536）。在 vLLM 0.11 / transformers 4.57 里只传 `max_pixels` 不起作用（`from_pretrained(max_pixels=)` 不改 `size.longest_edge`，2026-11-08 探针核实）。`run.py` 用的 VLLMServer 发的就是这两个键，和 `PREREG_WB_PILOT.md` 7.1 节相同。512x512 本来就在上限以下，不受影响。
- **同一个会话**：100 档的不约束评测（#3）和约束评测（#4）在同一个 vLLM 会话里跑（`run.py --mode u,c`，一次服务启动、一遍场景加载）。可选的 raw、0W 约束评测也和各自的不约束评测同会话。
- **动作改名对照**：不在试点里跑，登记给全量（第 14 节第 3 条）。

## 8. 提示词（冻结）

- 用官方 Qwen3-VL-8B-Instruct 的 `single_frame.yaml` 配置，只改输出格式那一段：
  - system：`navigation_system.j2`，`mention_step0_target_centered: true`；动作说明照 `action_instruction.j2`（`use_cot: false`）；
  - user：`input_observation_instruction.j2`，包括问题、`Current step`、`Your current state:` 下的位置（`include_translation_info: true`，`include_rotation_info: false`）、`Last action` 和碰撞那一行；再加当前这一帧图；
  - 没有历史图像，也没有历史动作（`num_past_observations/actions/reasoning` 全为 0）。这样也去掉了"多轮熟悉度"这个混淆（plan §3.8）。
  - 输出格式段的 `action: move_forward` 示例换成第 7 节的中性格式。
- 所有起点、所有档位、训练和评测的提示词逐字相同。具体文本由 WP-E3VS-B 的 `make_sft.py` / `run.py` 共用同一个函数生成，作业开始时把模板的 sha256 记进 manifest。
- 和 plan §3.8 的差别：plan 只写了"问题、步数、上一个动作、碰撞标记"。这里按官方配置**加上位置文字**，因为官方 Qwen 配置就带它（我的建议，附 A 第 5 条）。

## 9. 示范、热身、档位和训练

### 9.1 BFS 示范（`vpd/e3vs/bfs_demos.py`）

- 在 0.25 m / 30° 的动作格点上做广度优先搜索：
  - 状态键：位置取到 1 cm，偏航序号 mod 12，俯仰序号；
  - 俯仰保持在 ±90° 以内（差 1e-6° 也不行，附 A 第 4 条）；
  - 动作按固定顺序展开：move_forward、move_backward、move_left、move_right、move_up、move_down、turn_left、turn_right、look_up、look_down。新状态一生成就判目标，所以找到的是步数最近的可答视角，平局按这个顺序；
  - 路径末尾加 STOP；深度上限 16；状态上限 1.5M（每层结束后检查）。
- 不考虑碰撞时，原型的结果是（val 已用 `bfs_demos.py` 复现，2026-11-08）：
  - val 218/231 解出，13 个超上限，中位数 3 步；
  - test 355/377 解出，22 个超上限；
  - train 1,344/1,406 解出，62 个超上限（调查原型 bfs.py，2026-11-08）。
- 正式示范在 e3vs-prep 作业里用冻结的碰撞模型重跑（`--collides`）。每条路径都用官方语义从起点重放，终点必须成功，且没有一步碰撞。
- **示范池**：train 里在碰撞模型下、上述上限内解出的 episode。超上限或解不出的 episode **排除**，个数报告出来。
- 不可答视角当额外起点（`--extra`）的做法登记给 >900 档，试点不用。

### 9.2 格式热身 0W（我的建议，WP-E3VS-B `make_sft.py` 实现）

- 20 条单步记录，每条一次决策，所有起点用同一份：
  - 取 20 个不同的 train 场景：每个场景里按 md5(`<scene_id>:<episode_index>`) 取最小的那个 episode，各场景再按这个 md5 排序取前 20 个；
  - 图像是该 episode 起点位姿的渲染；
  - 第 i 条（i = 0–19）的动作是官方动作表的第 (i mod 11) 个（顺序见第 4 节）；
  - user 文本照第 8 节的提示词，只把问题换成：`Format practice, not a viewpoint-search task: take the action {name}.`；
  - 回复 `THOUGHT: \nACTION: {name}`。
- 训练：每个起点一个 0W，12 轮，30 个优化步，种子 1001；每个起点作业里都重新训；代码 sha 记进 meta，汇总时断言三个起点相同（DECISIONS B7、C2）。
- 曲线的 0 点是 0W；raw 也报。

### 9.3 档位

- 试点只有 100 档，种子 1001。
- 顺序文件 `OUT/e3vs/sft/order_seed1001.json`，从示范池生成：
  1. 每个（scene_id, question）只留一个 episode：取 md5(`<scene_id>:<episode_index>`) 最小的；
  2. 按题型分组，每组用 `random.Random(1001)` 依次打乱（组按题型名排序）；
  3. 依次往全局顺序里放：每次挑"目标份额 × (已放总数 + 1) − 该题型已放数"最大的题型，平局按题型名，取它组里的下一个；目标份额 = 该题型在去重后的池里的占比。
  - 这样任何前缀都按题型分层，各档嵌套。
- 100 档 = 顺序里的前 100 个 episode，展开成它们的全部决策记录（每一步一条，包括 STOP）。

### 9.4 训练配方

- 和 `PREREG_WB_PILOT.md` 第 6 节相同：LoRA r16 / α32 / dropout 0.05，语言模型全部线性层，视觉塔和投影层冻结；学习率 1e-4，3% 预热加余弦；梯度累积 8；bf16；不挑 checkpoint。
- 轮数：`epochs = max(3, ceil(240/N))`，N = 示范条数。100 档是 3 轮，0W 是 12 轮。
- 100 档从本起点的 0W 初始化，优化器和学习率调度重新开始。
- 记录是单步的，所以每条记录只有一个回复；损失只算回复和它的 `<|im_end|>`。

## 10. 盲走下限（冻结的定义；DECISIONS B8）

- **定义**：在 train 上挑出的固定动作序列，在**冻结的碰撞模型下**的 test 成功率。
  - 候选：10 个非 STOP 动作组成的、长度 0–3 的全部序列，共 1,111 条；执行完再 STOP。
  - 挑法：train 成功率最高；平局取更短的，再平局按枚举顺序（`vpd/e3vs/floors.py`）。val 和 test 不参与挑选。
- **怎么算**：e3vs-prep 作业里运行 `python -m vpd.e3vs.floors --collides <碰撞模型>`，取输出里的 `gate_floor.rate`，记进 manifest。
- **参考值（不考虑碰撞，不进闸门）**：挑出的是 (move_up, look_down)：train 16.1%，val 10.8%，test 14.1%（53/377）。2026-11-08 在登录节点上用 `floors.py` 复现，和调查原型一致。
  - (look_down, move_up) 在 train 上和它并列（两者终点位姿相同），按枚举顺序取前者。
- 出发即停（STOP0）的成功率是 0%（起点永远不算成功）。

## 11. 统计方法

- 每个起点每次评测：成功数 / 377，Wilson 95% 区间。
- 按场景整组重抽的 bootstrap（21 个 test 场景，1 万次，种子 0，百分位区间），只作描述。
- 试点不做起点之间的检验。

## 12. 闸门规则

每个起点的评测（评测名 `pilot__<START>__<tier>__<mode>`，目录 `OUT/e3vs/runs/pilot/<START>/evals/`）：

| # | 模型 | 解码 | 用途 |
|---|---|---|---|
| 1 | raw | 不约束（主） | 0 档描述；E0 |
| 2 | 0W | 不约束（主） | 曲线 0 点；E0 |
| 3 | 100 档 | 不约束（主） | **E1、E3**；E0 |
| 4 | 100 档 | 约束（次） | **E2**；E0 |

必做 4 × 3 = 12 次。预算允许时可加 raw 和 0W 的约束评测，只作描述（第 13 节）。

### E0 机制（DECISIONS B7、B16、B18）

下面各条全部满足（`vpd/e3vs/summary.py gate` 逐条检查）：
- 渲染一致性 PSNR ≥ 40 dB（第 5 节）；碰撞模型两个验证指标都 < 1%。
- 每次评测 377 个 episode 都有日志；没有崩溃。
- 每张图的 token 数：**每次**评测都有 `audit.json`，从处理器的 `image_grid_thw` 算出的值恰好是 [258]（256 + 2）。缺 audit、audit 失败或为空、或不等于 [258]，都算 E0 不过（B4）。258 本身是推断值：不一致时先查原因，记偏差日志，不算科学失败。
- 冻结的碰撞模型：每次评测的 manifest 里 `collides` = `vpd.e3vs.occupancy:load_collider`，每个场景的 `occ_sha256` 等于 `validation.json` 里验证过的栅格。
- 下限的出处：`floors_occ.json` 用的是冻结的碰撞模型，并且 prep 作业的戳 `OUT/e3vs/prep_state/floors` 等于 `validation.json` 的 `grids_sha256`。
- 适配器的出处：
  - 每次 0W、100 档评测 manifest 里的 `adapter_sha256` 等于本起点 `adapters/<档>/adapter.pt` 现在的 sha256；raw 评测不带适配器；
  - 100 档 meta 里的 `init_adapter.sha256` 等于本起点 `adapters/0W/adapter.pt` 的 sha256（从同一个起点的 0W 初始化）。
- 训练数据的出处：
  - prep 作业的戳 `OUT/e3vs/prep_state/sft` 等于 `validation.json` 的 `grids_sha256`（示范是在验证过的栅格上做的）；
  - SFT 数据的 manifest（`warmup_manifest.json` 和 `seed1001/demos_100_manifest.json`）存在，两者的 system 提示词 sha256 相同，并且等于每次评测 manifest 里的 system 提示词 sha256。
- 三个起点 0W 的代码 sha 相同。
- HF 和服务端的 prompt token 数是否一致（audit 的 `prompt_parity`，B5）逐次报出，只作描述。

### E1 学得会（报告 v2 §3.2"闸门"、§4.8(1)；DECISIONS B3、B8）

- 至少一个起点满足：100 档主评测（#3）成功率的 Wilson 95% 下界 > 第 10 节的盲走下限（同一批 377 个 episode 上的点值）。

### E2 格式（DECISIONS B3、B12）

- 每个起点的约束评测（#4）：回复 100% 匹配正则，没有日志的 episode 为 0。
- 不约束评测的解析率（报告 v2 §3.2 写的"≥95%"）只作描述，不挡闸门（DECISIONS B3：不约束 / 宽松的结果在任何闸门里都不起阻断作用）。它在确认轮里用作"曲线从哪一档开始比"的规则（第 14 节第 6 条）。

### E3 没饱和

- 每个起点 100 档主评测（#3）成功率 < 85%（plan §3.8）。

### 决策表

| 情况 | 结论 |
|---|---|
| E0–E3 都过 | E3VS 进全量（第 14 节），要用户确认预算 |
| E0 不过 | 修实现，重跑受影响的部分；不算科学结论 |
| E1 不过 | "≤100 条示范学不会"；交用户 |
| E2 不过（不是实现问题） | 交用户 |
| E3 不过 | "已饱和"；交用户，不自动改档位 |

## 13. 试点里允许改的东西

- 允许：修 bug（记偏差日志，重跑受影响的评测）；预算紧时去掉可选评测。
- 碰撞模型阈值：只能按第 5 节的规则在第一次模型评测之前调一次。
- 不允许：看了任何模型结果以后再改成功定义、下限定义、提示词、格式、档位或闸门阈值。

## 14. 现在就登记的确认轮条目（冻结；报告 v2 §4.2–§4.8）

1. **档位**：{25, 100, 300, 1000}，嵌套，按题型分层，同一种子下各起点用同一个子集。1000 档要用不可答视角当额外起点（只有约 900 个不同的训练 QA 情境）。
2. **种子**：100 档及以下 3 个种子（1、2、3），300 档 2 个（1、2），1000 档 1 个（1）。每个种子重新训 0W_s。
3. **动作改名对照**（DECISIONS B12；报告 v2 §3.2、§4.5）：在 25 和 100 档，把 11 个动作名按官方顺序换成字母代码 A–K（move_forward = A，move_backward = B，move_up = C，move_down = D，move_left = E，move_right = F，turn_left = G，turn_right = H，look_up = I，look_down = J，stop = K），提示词里的动作说明同步改写。改名后优势还在，才不是"认得这些词"。
4. **起点**：核心 Q3B、Q3S、E2；RL 对 Q3RD 和 QRL（QRL-s1、QRL-s2 两个种子当重复）。
5. **主终点**：除 0 档和最高档以外各档主成功率的平均（只有倍增档位时等于对数档位轴上的梯形面积）。两个对比：Q3S − E2，Q3RD − mean(QRL-s1, QRL-s2)，Holm 校正；按场景整组 bootstrap 1 万次，再对种子取平均。
6. **曲线从哪一档开始比**：从所有起点不约束解析率都 ≥ 95% 的那一档开始（报告 v2 §4.5）。
7. **其他**：盲（去图）对照在主对比档位上对 Q3S 和 E2 各做一次（报告 v2 §4.8(5)）；报每个起点离基座的参数距离和上游训练量（C0 第 8 轮 1,232 步，E2 第 10 轮 3,010 步）。
8. **判据权重**：E3VS 的正结果只算近迁移证据，不能顶替 WalkerBench 或 TVR（报告 v2 §4.7）。

## 15. 成本和预算（DECISIONS B2）

- E3VS 预留 10.5 卡时，在三个闸门共用的 23 卡时上限里（DECISIONS B2）。
- 估计（plan §6）：e3vs-prep 1–2 卡时；每个起点 1.5–2.8 卡时（0W + 100 档训练约 0.3–0.5，四次评测各 0.3–0.6）；合计 5.5–10.5。
- 提交前检查：已花的 + 在跑和已提交作业的上限估计 + 新作业的上限估计 ≤ 本闸门的预留。

## 16. 偏差日志

（提交后才开始记。格式：日期 UTC、改了什么、为什么、影响哪些结果。）

- 2026-11-09 16:2x UTC：本文件已在 a4150e06（2026-11-09 10:35，作者 yyyybq）提交。之后编排改了开头一条、第 5 节"起点卡在余量里"和附 A 第 10 条：从"待用户决定 (i)/(ii)"改成"已定 (ii)"（官方式跳过起点 1 个体素，代码 0b926407）。原因：这一条在登记时就写明要在 J5 之前定；编排按用户 11-08 的"其他的按你推荐"选 (ii)，用户可在 J5 之前改回 (i)。影响：盲走下限、BFS 示范、闭环评测的碰撞判定（三处一致）。还没有任何 E3VS 结果。批准人：用户（2026-11-10 在本线窗口确认；2026-11-09 已在本线窗口报告）。
- 2026-11-09 23:52 UTC：用户 2026-11-09 在本线窗口说"接下来没有卡时和磁盘的上限"。所以第 15 节的 10.5 卡时预留（以及三个闸门共用的 23 卡时上限）作废：代码 b898b511 起 BUDGET_POLICY.json 的 gpuh_cap 为 null，只记账不拦截。开头一条里"E3VS 下载要等磁盘线腾出空间"这半句也作废：DISK_EXCEPTION.json 现在含 e3vs，cap_gb 为 null（磁盘线其他 FLAG 和剩余 0.5 TB 的硬下限照常）。下载由用户本人运行（vpd/e3vs/USER_STEPS.md），固定版本的仓库由编排从集群内的副本拷贝（download_splats.sh repo-from-copy；用户 2026-11-09 同意保留并使用之前取到的文件）。影响：只影响提交时的预算和磁盘检查，不影响任何闸门、下限、评测或统计。还没有任何 E3VS 结果。批准人：用户（2026-11-09，本线窗口）。
- 2026-11-10 11:2x UTC：用户 2026-11-10 在本线窗口的第三轮说明（OUT/plan/ROUND3.md 第 4 节）确认碰撞规则 (ii)，并确认上面 16:2x 那一条，所以把它的批准人一栏从"待用户确认"改成"用户（2026-11-10）"。同时批准 J5、J6。闸门、下限、评测、统计都没改。还没有任何 E3VS 结果。批准人：用户（2026-11-10，本线窗口）。
- 2026-11-12 18:3x UTC：登记主报告的比较（只报告，不进第 12 节的闸门和决策表；第 11 节"试点不做起点之间的检验"因此加了这一项例外）。主报告：Q3S − Q3B，在 100 档主评测（第 12 节表里的 #3：100 档、不约束解码、主）；补充：Q3S − E2，同一评测。方法同 TVR 2026-11-11 登记的比较：两个起点在同一批 377 个 test episode 上按 episode 配对，成功按第 6 节；报两边的成功数 / 377 和 Wilson 95% 区间、b（只有前一个起点成功的 episode 数）、c（只有后一个起点成功的数）、双侧精确检验 p = min(1, 2·P(X ≤ min(b, c)))，X ~ Bin(b + c, 1/2)，b + c = 0 时 p = 1；不做多重比较校正；p 只算了 episode 抽样，没算训练种子之间的波动（每个起点只有种子 1001），偏乐观（推断）。计算顺序：闸门（`summary.py gate`）之后，由单独的脚本算，另一个独立脚本复算。读数之后的走法（用户 2026-11-12 定，写在读数之前）：闸门不过 → 报告，E3VS 不当尺子，不再花卡时；闸门过、Q3S − Q3B 在 100 档主评测显著为负（p < 0.05）→ 不跑确认轮，写负结果，报告，等用户定；闸门过、其余情况（为正或无定论）→ 功效模拟 → 确认轮计划 → 预估不超过 100 卡时就开跑（第 14 节，种子 1、2、3；Q3S − E2；Q3RD − mean(QRL-s1, QRL-s2)），Q3S − Q3B 一并报。闸门、下限、评测、统计方法都没改。写入时还没有任何 E3VS 模型结果（J5、J6 都还没交）。批准人：用户（2026-11-12，本线窗口）。

## 附 A. 登记时已知的、和 plan / 报告 v2 / 调查原型不同的地方

1. **主口径是不约束解码 + 宽松解析器**（DECISIONS B12），不是 plan §3.8 写的约束解码为主。约束解码降为次要评测，只用来做 E2 的机制检查。
2. **下限是定义，不是数字**（DECISIONS B8）。plan §3.8 写的 14.1% 是不考虑碰撞的值，只作参考。
3. **不约束解析率不挡闸门**（DECISIONS B3）。报告 v2 §3.2 把"解析率 ≥95%"写进了闸门；这里改为描述指标，并在确认轮里当作曲线起点规则（报告 v2 §4.5）。
4. **BFS 的俯仰限制加了 1e-6° 容差**。+30° 模板视角在文件里存的是 29.99999999999998°，调查原型（严格的 ">= 90"）因此能连抬两次头到正上方。正上方时官方的 planar_forward 是零，左右平移不动，格点模型表示不了。加容差后搜索空间比原型略小。2026-11-08 在 val 上对比（不考虑碰撞）：两种规则都是 218/231 解出、13 个超上限、中位数 3 步，步数分布相同；每条路径、每个位姿也相同，只有 13 个 episode 的已访问状态数变少。最终以 e3vs-prep 作业里带碰撞模型的结果为准。
5. **提示词带位置文字**，因为官方 Qwen3-VL 配置 `include_translation_info: true`。plan §3.8 没提（我的建议）。
6. **饱和阈值是 85%**（plan §3.8），不是 WalkerBench 的 80%。两个任务的上限不同：E3VS 的 BFS 示范池在不考虑碰撞时覆盖约 95% 的 train。
7. **碰撞模型不是官方的**：官方用 ScanNet++ 网格加 Open3D 光线投射，我们用 splat 深度的占据栅格（第 5 节）。和官方数字不可比。
8. **不用官方的 VQA 评分**（GPT-5.1）。成功只看视角（报告 v2 §3.2）。
9. **成功不要求 STOP**：取 STOP 时或第 25 次决策后的位姿。要求 STOP 的版本是次要指标（报告 v2 §3.2、调查 needs_user (a)）。
10. **起点卡在碰撞余量里的处理（已定 (ii)）**：官方式"跳过离起点不到 1 个体素的那段"（编排 2026-11-09 按"其他的按你推荐"选定，用户可在 J5 之前改回 (i)）。代码和测试见第 5 节。
