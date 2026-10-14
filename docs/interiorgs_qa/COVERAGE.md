# InteriorGS Spatial QA：题型覆盖与实施设计

调研日期：2026-10-14。本文只做来源核查、资产审计、CPU 候选普查和实验设计；未渲染、未生成问答数据、未训练、未修改生产代码。研究对象仅为 `VAGEN-Lite` 与 `/mnt/umm/users/yinbaiqiao/InteriorGS_meta`。

现有资产足以启动静态几何、相机关系和部分多视图题；完整覆盖八套还需要语义朝向、功能区、部件或动态资产。下一步的主要工作是冻结任务级GT定义、观测筛选与数据验收，而非继续扩充Yes/No模板。本报告的产量数字都是CPU候选容量；正式benchmark QA仍为0。

## 1. 目标、边界与现状核对

目标是在 InteriorGS 上构建借鉴主流 spatial QA **能力与题型**的数据，分别用于：①从同一个 Qwen 基座出发的 QA-SFT 对照；②保留场景上的同域 QA 评测。无需逐题复制原 benchmark，也不把新数据上的分数称为原 benchmark 成绩。

用户指定 `data_gen/active_spatial_qa/` 的 Active 成功谓词 Yes/No 转换器独立作为 **Read 测试**：同一画面、同一对象，读出当前关系。Read 不计入本表 benchmark 覆盖，也不自动并入 QA-SFT。已有 v10 `qa_states.py` 中 read/predict/act 状态问题也须按此边界另行归类，不能用函数/模板数量替代本表覆盖。

范围取自 [`scripts/easi_eval.py`](../../scripts/easi_eval.py) 的 EASI-8：VSI-Bench、MMSI-Bench、MindCube-Tiny、ViewSpatial、SITE、BLINK、3DSRBench、EmbSpatial。用户要求前三套展开完整题型，因此 MindCube 按完整论文分类审计；现有 EASI 注册名仍是 **MindCube-Tiny**，不把 Tiny 的样本量视为完整覆盖。后五套仅列空间相关任务，排除 BLINK 的艺术风格、取证、反射率等非本研究目标任务。

| 上轮结论 | 本轮核对 |
| --- | --- |
| Active→QA 转换器登记 10 类，统一当前状态 Yes/No | 成立。见 [`contract.py`](../../data_gen/active_spatial_qa/contract.py)、[`generate_paired_qa.py`](../../data_gen/active_spatial_qa/generate_paired_qa.py)。登记能力不等于已验收数据。 |
| 旧生产 3 场景、60 图、左右/同框两类；10 条助手认为可判断；28 条至少一目标完全被墙挡；正式导出 0 | 对 `production_v1_20261012` 成立。10 条只是助手通过、待人工，不能称人工确认的合格图；28 条都已被助手拒绝。 |
| 补充核查：v1之后是否有墙遮挡修复批次 | **有。** 后续 `production_v2_wallfix_20261013` 引入结构墙/开口与对象框预筛：新渲染 24 张，助手通过 14、目标不明确而拒绝 10，人工待审 24，正式导出仍 0。不能把 v1 的 28/60 失效率直接套到 v2 或 v10。 |
| v10 覆盖 1,000 场景并有统一 split | 成立。可核实的完整第五轮产物为 `interiorgs_v10_gen_20261014`：942 场景有题，13,158 主选、14,600 后备；train/val/test 为 800/100/100 场景和 10,477/1,351/1,330 主选。尚不是完成审核的 QA 数据。 |
| v10 有更严格可见规则 | 成立。第五轮冻结规则 `face_visible_v2`：每个必须识别对象至少一个面 25 个采样点中 ≥10 可见，露出面积 ≥画面 0.3%，并检查参照物。工作树已另有 `face_visible_v3`（结构墙面板、门洞、窗帘）与指代 v3。本文以固定 main 提交 `3a964c795f82e4014572d7dc61425cd22d7816f3` 的 v3 可见性为主计数，并保留第五轮快照 v2 的同条件敏感性对照；两轮都固定 r5 split/指代筛选。 |

状态与实现入口：[`v10/STATUS.md`](../../data_gen/interiorgs_v10/STATUS.md)、[`REPORT_R5_20261014.md`](../../data_gen/interiorgs_v10/cluster_runs/REPORT_R5_20261014.md)、[`target_visibility.py`](../../vagen/envs/active_spatial/target_visibility.py)。文档/工作树会继续更新；本文计数以证据文件内 SHA-256、运行快照为准。

本机原始证据（属于 ignored 运行产物，GitHub 不含数据本体）：

- `data_gen/active_spatial_qa/artifacts/production_v1_20261012/website_review_v1_20261013/records/review_summary.json`
- `data_gen/active_spatial_qa/artifacts/production_v1_20261012/wall_occlusion_audit_v1_20261013/FINDINGS.txt`
- `data_gen/active_spatial_qa/artifacts/production_v2_wallfix_20261013/reviewed/sample_manifest.review_summary.json`
- `data_gen/active_spatial_qa/artifacts/production_v2_wallfix_20261013/exports/export_summary.json`
- `exps/vagen_active_spatial/interiorgs_v10_gen_20261014/{split_manifest.json,sft/sft_frames_report.json}`

## 2. 标注能力与判定口径

下表的可实现等级评价的是 **标准答案所需资产**，不是“当前生成器已完成”，也不是“RGB 一定可答”。三个等级严格分开：

- **能按原定义算**：标注和我们控制的相机/变换足以实现同一真值定义；仍需新增代码、观测筛选、题目及图像审核。
- **只能近似**：可构建有意义的 InteriorGS 衍生任务，但框代替真实表面、房间轮廓代替点云地板、可见性代理代替像素等造成定义差异，必须命名和记录差异。
- **需要场景里没有的东西，做不了**：在当前资产边界下缺少语义正面、物体运动、关节、功能区、人体等关键事实。可以另立人工标注/资产项目，不能填成现有覆盖。

官方人工题的标准答案来源要写“人工标注/共识”，不能为了自动生成方便虚构成官方几何公式。对于有人工与几何混合构建的任务，表中分别写明。公共模型输入默认只有题目及 RGB/有序帧；相机外参、世界坐标、框、房间轮廓、标签和答案证据只进入私有审计字段，除非某题原协议或新实验明确提供这些输入。

已实际遍历最新完整 split 中 **1,000/1,000 场景**的元数据，不是仅看一个示例。完整字段频数、异常框、示例和源文件哈希见 [`audit/cpu_audit.json`](audit/cpu_audit.json) 的 `schema`；对应只读脚本见 [`audit/cpu_audit.py`](audit/cpu_audit.py)。

| 资产 | 实查结果 | 能支持 / 不能据此推出 |
| --- | --- | --- |
| 文件清单 | 每场景均有 `labels.json`、`structure.json`、`occupancy.json`、`occupancy.png`，各1,000份 | 元数据目录没有对象mesh、RGB深度、实例mask、运动轨迹；GS渲染资产在独立场景存储，不能把其存在当作已有实例分割 |
| 类别与实例 | 506,061条标签；715种原始类别；字段为`ins_id`、`label`，部分有`bounding_box`、`room_box` | 支持类别/实例清点；原始类别需规范化、漏标/合并框须核验。59,248条无八点框，不可直接计入三维几何任务 |
| 三维框 | 446,813个八顶点框；446,741个可按固定角点次序恢复正交非退化OBB；72个非正交/退化需排除或审核 | `p1-p0`、`p3-p0`、`p4-p0`给三边，检查直角及八角重建后可取长度。不能把无效框静默修成正确GT |
| 框几何方向 | 41,341个框非轴对齐，其中41,329可恢复有效旋转OBB；其余大部分轴对齐 | 可以计算箱体几何轴和无向yaw。现有v10 `parse_labels`取坐标min/max会丢旋转，因此新增尺寸/朝向几何应读原框 |
| 物体语义正面 | 没有`front/semantic_front/heading/orientation/rotation/quaternion`等字段 | **几何主轴不等于“椅子面对哪里”**；轴的正负号、对称物体正面不能从OBB推出。以人体/物体自我视角作问答需额外语义标注或明确修改题设 |
| 房间轮廓 | 6,607个房间，所有记录只有`profile` | 可算几何面积、包含和房间关系；需先校准结构/label坐标的Y符号。房间轮廓并不自动表示通路连通图 |
| 房间名称/类型 | **0个原生room name/type** | v10从床/马桶/家具推断类型，是启发式；不能当GT。可用“含唯一冰箱的房间”指称几何房间，但刷牙区/食物储存区仍没有功能区域标注 |
| 墙/开口 | 31,387面墙（986场景非空），13,658个洞/开口（987场景非空），类型含DOOR/WINDOW/HOLE/OPENING；有位置、厚度、高度/轮廓 | 支持结构遮挡和碰撞近似；14个无墙列表场景需保留fallback标记。框外装饰结构、真实表面孔洞、窗帘透明度仍须实际渲染验证 |
| Occupancy | 1,000份JSON，字段`scale/center/upper/lower/min/max`，另有PNG | 可用作平面空间栅格及坐标映射的基础；本轮只核schema，不把PNG当三维mesh、实例mask或官方最短路GT |
| 相机位姿/时间 | 原始meta中没有本次QA视图；已有r5任务存相机内参、外参和动作证书 | 新采样相机的pose可被精确控制；构建相机运动GT不需要动态物体。动作证书不是完成的视频题 |
| 外观/部件/动态 | 无颜色材质表、部件分割、关节、实例随时间运动或语义功能区真值 | 可以从后续RGB人工补标，或增加可编辑/动态资产；当前不能仅凭类别框自动给这些问题可靠答案 |

**VSI 距离需要精确更正上一轮的表述：** 题干说两物体最近点距离，论文实际构建GT时是在两个框内均匀采样点，取最小欧氏点对距离。因此 InteriorGS 有效OBB足以复用这个**框级GT算法**，并非必须先拿到真实表面mesh；但用中心距或未经说明的AABB距离仍不相同。若改用OBB精确集合距离，应标明是对原采样算法的确定性变体、另定采样/容差。[VSI官方构建说明](https://arxiv.org/html/2412.14171v1#A2.SS1)

本报告把“原始资产能给GT”和“模型观察能支持回答”分开：例如米制框距离可算，但需要检查视觉尺度依据；共线/直角构型可筛，但物体是否认得出仍需RGB；多图相对相机位姿可算，但视图是否有足够共同视觉证据还要验证。


## 3. 复用代码索引

覆盖表中每行的复用字母指向下面的具体文件。它们是可利用的底座，不表示无需新增该子任务的生成器。

| 代号 | 可复用能力与文件 | 复用限制 |
| --- | --- | --- |
| R | [`render_paired_bank.py`](../../data_gen/active_spatial_qa/render_paired_bank.py)、[`interiorgs_v10/pilot.py`](../../data_gen/interiorgs_v10/pilot.py)、[`canonical_camera.py`](../../vagen/envs/active_spatial/canonical_camera.py) | 精确位姿 RGB 与内参一致性；视频时间采样/编码和实例级 mask 还需实现。 |
| C | [`interiorgs_v10/scene.py`](../../data_gen/interiorgs_v10/scene.py)、[`sampling.py`](../../data_gen/interiorgs_v10/sampling.py)、[`endpoints.py`](../../data_gen/interiorgs_v10/endpoints.py)、[`initial_states.py`](../../data_gen/interiorgs_v10/initial_states.py) | 合法位置、相机朝向、现有动作路径；原 benchmark 的轨迹覆盖和多图必要性需新增筛选。 |
| G | [`scene.py`](../../data_gen/interiorgs_v10/scene.py)、[`coordinates.py`](../../data_gen/interiorgs_v10/coordinates.py)、[`fastgeom.py`](../../data_gen/interiorgs_v10/fastgeom.py)、[`objects.py`](../../data_gen/interiorgs_v10/objects.py) | 物体/房间/坐标处理；目前 Obj 转成 AABB，精确 OBB 任务必须回读原八顶点，不能拿 AABB 边长冒充 OBB。 |
| K | [`collision_detector.py`](../../vagen/envs/active_spatial/collision_detector.py)、[`scene.py`](../../data_gen/interiorgs_v10/scene.py) | 相机/动作碰撞；并非完整导航网格，也不等于视频中的可通行路线标注。 |
| V | [`target_visibility.py`](../../vagen/envs/active_spatial/target_visibility.py)、[`interiorgs_v10/occlusion.py`](../../data_gen/interiorgs_v10/occlusion.py) | 框面采样及结构遮挡，版本必须绑定；非真实实例分割、像素可见比例或语义可识别性的证明。局部 QA v2 的 `qa_wall_visibility_v2.py` 尚未跟踪入 Git，复用前应审核，不作为唯一依赖。 |
| F | [`referring.py`](../../data_gen/interiorgs_v10/referring.py)、[`objects.py`](../../data_gen/interiorgs_v10/objects.py)、[`start_context.py`](../../data_gen/interiorgs_v10/start_context.py) | 类别归一、唯一性、参照物/大小、起点上下文。QA 应避免用待求关系本身指代对象造成答案泄漏；家具推断的房间类型不是原生标注。 |
| S | [`scene_identity.py`](../../data_gen/interiorgs_v10/scene_identity.py)、[`make_split.py`](../../data_gen/interiorgs_v10/make_split.py)、[`release.py`](../../data_gen/interiorgs_v10/release.py)、[`admission.py`](../../data_gen/interiorgs_v10/admission.py) | 同源场景分组、split、版本与审核证据；正式 QA 使用既有划分，不重新按问题随机切分。 |

## 4. Benchmark 子任务覆盖表

官方输入模态保持原样列出；后文“单图→多图→视频”分批是实现顺序。把 VSI 内容先做成单图问题，只能记为该能力的单图衍生训练题，不能标为 VSI 视频协议已完成。假设相机绕某物体转向的题，参考系是由题目定义的虚拟相机；它不必依赖物体的语义正面。

### 4.1 三个核心benchmark的一手依据

本次核查日期：2026-10-14。以下是文档采用的**固定快照**，不声称是未来不变的最新版。

| 依据 ID | 固定版本 | 用途与直接链接 |
|---|---|---|
| V-P | arXiv `2412.14171v1` | [VSI 论文，§3、附录 B.1 / Table 3](https://arxiv.org/html/2412.14171v1#A2.SS1)：任务、原标注过程和歧义过滤 |
| V-E | `vision-x-nyu/thinking-in-space@51e089c3ae69b9435e9489058610f5b3964c56a8` | [评分代码](https://github.com/vision-x-nyu/thinking-in-space/blob/51e089c3ae69b9435e9489058610f5b3964c56a8/lmms_eval/tasks/vsibench/utils.py)、[评测配置](https://github.com/vision-x-nyu/thinking-in-space/blob/51e089c3ae69b9435e9489058610f5b3964c56a8/lmms_eval/tasks/vsibench/vsibench.yaml) |
| M-P | arXiv `2505.23764v1` | [MMSI 论文，§3.1–3.2、§4.1、附录 E/F](https://arxiv.org/html/2505.23764v1#S3)：11 类、人工设计、输入及评分 |
| M-D | 官方公开数据卡，本次只读检查 | [MMSI 官方样例](https://huggingface.co/datasets/RunsenXu/MMSI-Bench)：展示参考系、功能区域、测量/外观及运动问题的真实范围；未下载原图/完整数据 |
| M-E | `InternRobotics/MMSI-Bench@13e58a2b8b30d880d7e8a1e4a6aa1c0feda94cac` | [普通及 Circular 评分实现](https://github.com/InternRobotics/MMSI-Bench/blob/13e58a2b8b30d880d7e8a1e4a6aa1c0feda94cac/vlmeval/dataset/mmsi_bench.py) |
| C-P | arXiv `2506.21458v2`，2026-03-31 | [MindCube 论文，附录 B.1–B.2](https://arxiv.org/html/2506.21458v2#A2.SS2)、[评测及认知地图指标，附录 D.2](https://arxiv.org/html/2506.21458v2#A4.SS2) |
| C-E | `mll-lab-nu/MindCube@6d8fd9acdf5b80769373a21cdbfcf068ab356e0f` | [答案评分](https://github.com/mll-lab-nu/MindCube/blob/6d8fd9acdf5b80769373a21cdbfcf068ab356e0f/src/evaluation/evaluator.py)、[答案提取](https://github.com/mll-lab-nu/MindCube/blob/6d8fd9acdf5b80769373a21cdbfcf068ab356e0f/src/evaluation/core/extractors.py)、[汇总过滤](https://github.com/mll-lab-nu/MindCube/blob/6d8fd9acdf5b80769373a21cdbfcf068ab356e0f/src/evaluation/core/base_metrics.py) |
| C-G | 与 C-E 相同 commit | [Rotation 辅助代码](https://github.com/mll-lab-nu/MindCube/blob/6d8fd9acdf5b80769373a21cdbfcf068ab356e0f/src/scaffold_curation/reasoning/rotation_reasoning.py)、[Around 辅助代码](https://github.com/mll-lab-nu/MindCube/blob/6d8fd9acdf5b80769373a21cdbfcf068ab356e0f/src/scaffold_curation/reasoning/around_reasoning.py)、[Among 辅助代码](https://github.com/mll-lab-nu/MindCube/blob/6d8fd9acdf5b80769373a21cdbfcf068ab356e0f/src/scaffold_curation/reasoning/among_reasoning.py)、[Translation 辅助代码](https://github.com/mll-lab-nu/MindCube/blob/6d8fd9acdf5b80769373a21cdbfcf068ab356e0f/src/scaffold_curation/reasoning/translation_reasoning.py) |
| C-D | HF `MLL-Lab/MindCube@9c941b46a6bd65b6914669ef7a579948fc9c8467` | [数据 revision](https://huggingface.co/datasets/MLL-Lab/MindCube/commit/9c941b46a6bd65b6914669ef7a579948fc9c8467)：只固定版本、未下载 `data.zip`。Tiny 仅是诊断子集；接好 Tiny 外部评测不代表完整任务已经迁移 |

### 4.2 VSI-Bench：8 个主任务，方向细分后为 10 行

统一输入 `V`：固定有序 RGB 视频/其采样帧 + 问题；选择题另含选项。相机 pose、物体框、地图只用于离线真值与采样，不能暗中作为普通 QA 模型输入。以下方向任务使用“站在 P、面向 F”构造的**假设观察者坐标系**，不需要物体的语义正面。

统一评分 `V-NA`：数值答案使用 MRA；`V-MC`：选项字母准确率。固定 V-E 代码先对相对方向 easy/medium/hard 三档取平均，再对 8 个主任务取宏平均；不能把方向三档当作三个主任务提高其权重。[评分实现](https://github.com/vision-x-nyu/thinking-in-space/blob/51e089c3ae69b9435e9489058610f5b3964c56a8/lmms_eval/tasks/vsibench/utils.py#L114)

| 稳定 ID / 子任务 | 输入模态 | 参考系 | 答案格式 | 官方 GT 构建 / 评分 | InteriorGS 初评与具体原因 | 可复用模块 | 一手来源 |
|---|---|---|---|---|---|---|---|
| `VSI-count` 物体计数 | V | 场景/题目指定房间；视角无关 | 数值整数 | 统一实例标注计数，人工审核；V-NA | **能按原定义算**：从目标范围内的有效实例计数。需核对类别重映射、成组物体规则和漏标；观察序列须提供计数证据 | C、R、F；新增 scene census 与视频覆盖审核 | [V-P]；[V-E] |
| `VSI-distance-relative` 相对距离 | V | 物体间欧氏距离；视角无关 | 候选物体对应的字母 | 以最近点距离排序；同类多实例候选取最小值；候选距离两两间隔阈值为 30 cm（房间面积 >40 m²）/15 cm（其余），过近剔除；V-MC | **能按原定义算**：原八顶点支持原 benchmark 的框级距离 GT 口径；必须恢复 OBB，当前 AABB 路径对旋转框仅近似。不可用中心距替换 | C、F、R；新增 OBB 最近点/原版采样 oracle 与 margin 过滤 | [V-P]；[V-E] |
| `VSI-direction-easy` 相对方向 easy | V | P→F 定义前向，重力定义上向 | 左/右两选一字母 | 三物体的布局关系和角度边界过滤；V-MC | **能按原定义算**：通过几何位置构造假设相机轴；不使用 P 或 F 的 intrinsic front | C、F、R；新增三元组变换及去歧义 | [V-P]；[V-E] |
| `VSI-direction-medium` 相对方向 medium | V | 同上 | 左/右/后三选一字母 | 需转身至少 135° 才能面向目标时归为后方；V-MC | **能按原定义算**：后方采用转角 ≥135°，并排除角域边界，而非随意四分方向 | C、F、R；新增中档分区 oracle | [V-P]；[V-E] |
| `VSI-direction-hard` 相对方向 hard | V | 同上 | 前左/前右/后左/后右四选一字母 | 假设观察者坐标平面的四象限；V-MC | **能按原定义算**：三元组坐标变换可确定象限，须对轴附近位置留 margin | C、F、R；新增四象限 oracle | [V-P]；[V-E] |
| `VSI-route` 路线转向填空 | V | 路线当前朝向逐步更新 | 一串转向对应的选择题字母 | **人工构建和审核**地标路线，挖去转向；V-MC | **只能近似**：occupancy 可提议路线，但原任务要求可描述、可行、能依视觉地标辨认；需补地标路线生成和人工核验，不能声称原 GT 是最短路公式 | C、R、F、V；新增导航图、地标化、全路线验证 | [V-P]；[V-E] |
| `VSI-size-object` 物体尺寸 | V | 物体 OBB 轴；尺度不依相机 | 数值，厘米 | 唯一实例框的最长边；V-NA | **能按原定义算**：从通过正交性检查的原八顶点恢复边长；当前 AABB 最长边对旋转实例只近似。几何轴足够，语义“正面”不需要 | C、F、R；新增保留原 OBB、尺寸提取与单位校验 | [V-P]；[V-E] |
| `VSI-size-room` 房间面积 | V | 楼层平面；指定被观察空间 | 数值，平方米 | 原版用场景点云的 alpha-shape 估计面积；V-NA | **只能近似**：初步可从 room profile 求地板面积，但需确认它和视频所指空间一致，且没有把自由通行面积/occupancy 格数当房间面积。即使 profile 完整可信，polygon 面积仍是相对于原点云 alpha-shape 算法的不同 GT，应保留近似/衍生标记 | C、R；新增房间范围绑定、polygon 面积和多房间并集 | [V-P]；[V-E] |
| `VSI-distance-absolute` 绝对距离 | V | 物体间欧氏距离；视角无关 | 数值，米 | 在两物体框内采样，以最小点对距离构造 GT；V-NA | **能按原定义算**：恢复原 OBB 后可构造框级 GT；实际表面距离须另有实例几何。不得直接改成 AABB/中心距/导航路程而不标差异 | C、F、R；新增 OBB 距离 oracle、单位校验 | [V-P]；[V-E] |
| `VSI-order` 首次出现顺序 | V，严格保留帧序/时间 | 指定视频时间轴 | 类别排列对应的字母 | 按对象像素数过阈值的首次时刻确定，过近时间剔除；V-MC | **只能近似**：视锥内 bbox/少量射线不等于可见实例像素数。需要实例 mask 或经人工审定的逐帧可见性；必须绑定固定视频，不能把 agent 任意探索顺序称为相同 GT | R、V、F；新增时序 manifest 与像素级首次出现检测 | [V-P]；[V-E] |

VSI 评分实现注意事项：MRA 使用相对准确率阈值 0.50–0.95（步长0.05），等价于50%至5%的相对误差容差；并非模型置信度阈值；固定代码使用 `abs(pred-target)/target <= 1-threshold`。论文公式使用严格 `<`，因此实现版本应记录。原版计数题排除计数 1；新增计数 0 还会触及相对误差除零，不能直接塞进原 evaluator。方向三档之外，数值题虽然数据中可能保存 MC 选项，标准评分仍是数值 MRA。[论文任务约束](https://arxiv.org/html/2412.14171v1#A2.SS1)、[MRA / 类型分派实现](https://github.com/vision-x-nyu/thinking-in-space/blob/51e089c3ae69b9435e9489058610f5b3964c56a8/lmms_eval/tasks/vsibench/utils.py#L82)

### 4.3 MMSI-Bench：11 个任务族

统一输入 `M2`：**两张 RGB 图 + 问题 + 四个选项**；`Mn`：多步任务的多张 RGB 图，原数据最多 10 张。所有题都要求单张图不足以求解；仅把相同单图题复制成双图不算覆盖。`M-H` 表示原版每题的题干、答案、干扰项和参考推理均由人设计/审核，**不存在一个覆盖整类的官方自动 GT 公式**。下表的几何计算或组合程序是 InteriorGS 建议，不能归给原作者。[官方任务/标注说明](https://github.com/InternRobotics/MMSI-Bench/tree/13e58a2b8b30d880d7e8a1e4a6aa1c0feda94cac#introduction)

统一输出 `M-MC`：四选一字母；参考推理可用于训练/审核，但原主分数按答案准确率计。参考系必须按题文本指定保存，原题既有相机坐标，也有由地标定义的“北向”，不能统一假设世界 +X 是前方。[官方样例](https://huggingface.co/datasets/RunsenXu/MMSI-Bench)

| 稳定 ID / 子任务 | 输入模态 | 参考系 | 答案格式 | 官方 GT 构建 / 评分 | InteriorGS 初评与具体原因 | 可复用模块 | 一手来源 |
|---|---|---|---|---|---|---|---|
| `MMSI-position-camera-camera` 相机—相机位置关系 | M2 | 一张图的 camera frame；也可按题指定轴 | M-MC | M-H；选项准确率 | **能按原定义算**：已知两相机 pose 可求相对平移/旋转；需要标清 c2w/w2c、左右手系及相机 forward，不向模型泄露 pose | R、C；新增双视点变换、图像重叠与单图不足过滤 | [M-P]；[M-D]；[M-E] |
| `MMSI-position-camera-object` 相机—物体位置关系 | M2 | 当前/指定相机 ego frame | M-MC | M-H；选项准确率 | **能按原定义算**：位置关系可由 pose + 实例几何判定；应挑查询视角外、另一视角提供证据的物体，而非单图可见方向题 | R、C、V、F；新增跨视图证据选择 | [M-P]；[M-D]；[M-E] |
| `MMSI-position-camera-region` 相机—区域位置关系 | M2 | 相机 ego frame或题设轴 | M-MC | M-H；选项准确率 | **只能近似**：camera–room 几何子集可算，房间可用唯一地标指称并记录 `region_scope=room`；厨房/刷牙区等语义功能区尚缺标签与边界，不能宣称整族覆盖 | C、R、V；新增 semantic region annotation 与指称 | [M-P]；[M-D]；[M-E] |
| `MMSI-position-object-object` 物体—物体位置关系 | M2 | 题设环境轴或地标定义轴；部分题换视角 | M-MC | M-H；选项准确率 | **能按原定义算**：可构造具有明确题设参考轴的位置关系子集；此判定不包括依物体“自身前方”定义的 intrinsic 关系 | C、R、V、F；新增题设轴绑定和跨图关系 oracle | [M-P]；[M-D]；[M-E] |
| `MMSI-position-object-region` 物体—区域位置关系 | M2 | 区域/题设环境轴；可有地标定义北向 | M-MC | M-H；选项准确率 | **只能近似**：object–room 位置/归属子集可算，并用唯一地标指称房间；餐区、休闲区等功能区没有现成 GT，需独立标注 | C、R、F；新增功能区、containment/方向语义规则 | [M-P]；[M-D]；[M-E] |
| `MMSI-position-region-region` 区域—区域位置关系 | M2 | 题设环境轴或假设行进视角 | M-MC | M-H；选项准确率 | **只能近似**：room–room 关系可由轮廓及明确题设轴确定；一般功能区之间的位置/语义对应仍缺标注，不可用房间 adjacency 代表整族 | C、R；新增功能区图、范围和关系标注 | [M-P]；[M-D]；[M-E] |
| `MMSI-attribute-measurement` 几何属性测量 | M2 | 重力/度量世界坐标或明确物体轴 | M-MC，通常比较/量值选项 | M-H；选项准确率 | **能按原定义算**：可生成框高度、长度、大小等有明确定义的比较子集；部件尺寸、非框形状属性仍需额外几何。不得把整族改为单图尺寸题 | C、R、F；新增尺寸定义、误差 margin、跨图测量 | [M-P]；[M-D]；[M-E] |
| `MMSI-attribute-appearance` 外观属性 | M2 | 题设参考视侧或物体固有语义朝向 | M-MC | M-H；选项准确率 | **只能近似**：可做题干明确把图1定义为参考正面后的图2视侧子集，并筛除对称歧义；部件数、真实形状/外观和固有语义正面仍缺 GT，不能把框轴当正面 | R、F；新增属性/部件/语义朝向标注及复核 | [M-P]；[M-D]；[M-E] |
| `MMSI-motion-camera` 相机运动 | 有时间关系的 M2 | 相机初始 frame及旋转轴 | M-MC | M-H；选项准确率 | **能按原定义算**：可控相机序列有相对运动真值，静态 GS 足够。须区分纯平移、纯旋转和组合，且图像支持辨认 | R、C；新增有序 camera-pair sampler 与运动 oracle | [M-P]；[M-D]；[M-E] |
| `MMSI-motion-object` 物体运动 | 两时刻 RGB + Q | 题设相机/世界/物体参考系 | M-MC | M-H；选项准确率 | **需要场景里没有的东西，做不了**：静态 GS 没有对象随时间的位姿/形变。移动相机产生的像素位移不是物体运动；需动态原始序列或可编辑实例及受控动作 | R 仅可复用接口；新增动态实例/时间标注/渲染 | [M-P]；[M-D]；[M-E] |
| `MMSI-multistep` 多步组合推理 | Mn | 按各步骤显式切换、组合 | M-MC | M-H；答案准确率，参考推理用于诊断 | **只能近似**：可先做几何/相机关系的可验证组合子集；完整任务族可组合功能区、外观、动态，受上述缺资产项限制。需记录依赖 DAG 并验证每张图必要性，长题干不等于多步 | R、C、V、F；新增组合程序、证据 DAG、人工审核 | [M-P]；[M-D]；[M-E] |

MMSI 评分需分开两个协议：固定 M-E 的 `MMSIBenchDataset.evaluate` 默认正则提取 A–D 后匹配答案，未提取成功记错，汇总为逐题平均并输出分类准确率；`MMSIBenchCircular` 对四种循环选项顺序分别作答，**同题四次都对**才命中。论文叙述有 LLM extraction fallback，但该 commit 默认路径并未自动启用，Circular 中 LLM 路径也是注释切换。设计文档应分别命名 `MMSI-vanilla` 与 `MMSI-circular`，固定提取器/是否启用 judge，不能混报。[普通评分](https://github.com/InternRobotics/MMSI-Bench/blob/13e58a2b8b30d880d7e8a1e4a6aa1c0feda94cac/vlmeval/dataset/mmsi_bench.py#L121)、[Circular 评分](https://github.com/InternRobotics/MMSI-Bench/blob/13e58a2b8b30d880d7e8a1e4a6aa1c0feda94cac/vlmeval/dataset/mmsi_bench.py#L558)

### 4.4 MindCube：视图设置与题目维度分开记录

MindCube 不是三个互相独立的“方向任务”。覆盖记录至少包含 `view_setting × visual_pattern × relation_query × viewpoint_dynamics × perspective_taking`。后三个题目维度与视图设置可以交叉，但并非全部笛卡尔积都合法；一个样本会在多个维度上同时计数。下表 15 行是**三个设置和四维标签的全部已公布取值**，不能相加声称“15 个互斥任务”。[论文 taxonomy](https://arxiv.org/html/2506.21458v2#A2.SS2.SSS2)

统一输入 `C-MV`：少量 RGB 视图 + 问题/选项；必要时题干包含视点变化说明。原数据以人工选择/标注视图、物体跨图身份、布局、物体语义朝向与遮挡为基础，再按规则产生 QA，以下记作 `C-HR`。它不是拿任意物体框自动生成方向题；许多问题专门询问当前视角不可见的对象。统一答案 `C-MC` 为样本提供选项中的字母，选项数量随题变化，不能全部硬改为四选一。统一主评分 `C-ACC` 为提取答案后的逐题准确率。[标注流程](https://arxiv.org/html/2506.21458v2#A2.SS1)、[官方评分](https://github.com/mll-lab-nu/MindCube/blob/6d8fd9acdf5b80769373a21cdbfcf068ab356e0f/src/evaluation/evaluator.py)

| 稳定 ID / 子任务或维度取值 | 输入模态 | 参考系 | 答案格式 | 官方 GT 构建 / 评分 | InteriorGS 初评与具体原因 | 可复用模块 | 一手来源 |
|---|---|---|---|---|---|---|---|
| `MC-setting-rotation` 原地旋转视图设置 | 2–4 个近正交视图 + Q | 同一位置的不同 camera axes | C-MC | C-HR，已标布局经视角变换；C-ACC | **能按原定义算**：可控 pose 支持正交旋转及查询轴变换；需挑足够场景语义、当前 FOV 外对象，不需要 object intrinsic front | R、C、V、F；新增正交组采样 | [C-P]；[C-G] Rotation；[C-E] |
| `MC-setting-around` 绕视角遮挡设置 | 正/侧等视图 + Q | 视图坐标与统一对象布局 | C-MC | C-HR，侧视遮挡和关系转换；C-ACC | **只能近似**：物体顺序可由框算，但真实遮挡/可识别程度目前不能由少量 bbox 射线精确证明；需 mask/depth/实例几何或人工视图标注 | R、C、V、F；新增共线布局与遮挡组筛选 | [C-P]；[C-G] Around；[C-E] |
| `MC-setting-among` 中心物体及周围物体设置 | 典型为四个正交方位视图 + Q | 中心布局 + 各相机轴 | C-MC | C-HR，中心/外围身份与可见性；C-ACC | **只能近似**：静态资产可尝试筛出构型，尚需验证每个视图的中心/背景关系与真实可见性；包含他者视角的题另缺语义朝向 | R、C、V、F；新增中心—外围布局和联合视图筛选 | [C-P]；[C-G] Among；[C-E] |
| `MC-pattern-linear` 共线构型 | C-MV | 场景布局轴 | 非独立 QA；继承承载题的 C-MC | 人工布局标签；承载题用 C-ACC | **能按原定义算**：对象中心的共线性可作筛选，需预定长度/角度容差；BBox 重心代理与物体实际布局不一致时退回人工审核 | C、R、F；新增构型筛选 | [C-P]；[C-G] Around/Translation |
| `MC-pattern-perpendicular` 直角构型 | C-MV | 场景布局轴 | 非独立 QA；继承 C-MC | 人工布局标签；承载题用 C-ACC | **能按原定义算**：以对象位置筛选相邻连线近直角；不能把所有非共线布局都标成此类 | C、R、F；新增角度 margin | [C-P]；[C-G] Among/Rotation |
| `MC-dynamics-translation` 纯平移 | C-MV；含平移观测/题设 | 朝向固定的相机 frame | C-MC | C-HR；C-ACC，但 translation setting 默认不入官方 overall | **能按原定义算**：相机位置变化及关系传递可由 pose/布局确定；须确保共用锚点和跨图信息必要性 | C、R、V、F；新增平移双视图 | [C-P]；[C-G] Translation；[C-E] |
| `MC-dynamics-rotation` 纯旋转 | C-MV + 旋转题设 | 假设相机轴随旋转更新 | C-MC | C-HR；C-ACC | **能按原定义算**：对已知位置/朝向施加规定旋转；不依赖对象语义正面 | C、R、F；新增旋转程序 oracle | [C-P]；[C-G] Rotation；[C-E] |
| `MC-dynamics-simultaneous` 同时平移和旋转 | C-MV | 起始/终止相机 frame | C-MC | C-HR；C-ACC | **能按原定义算**：记录真实相机变换；若问题问运动过程而不只端点，需保留路径，不能仅以同一个终点变换覆盖所有过程 | C、R、V；新增组合相机运动定义 | [C-P]；[C-G] Around/Among；[C-E] |
| `MC-dynamics-sequential` 先后平移/旋转 | C-MV + 有顺序的假设动作描述 | 每一步更新的 agent frame | C-MC | C-HR；C-ACC | **能按原定义算**：按题目顺序组合变换；静态场景足够，不需要动态物体。但须给移动步长/终止条件使关系唯一 | C、R、F；新增可执行动作 AST 与边界过滤 | [C-P]；[C-G] Around/Among；[C-E] |
| `MC-relation-agent-agent` 观察者—观察者关系 | C-MV | 两个真实/假设视点 | C-MC | C-HR；C-ACC | **能按原定义算**：相机位置、朝向已知；“我如何移动到另一视点”可有可检验答案 | C、R；新增相对位姿任务 | [C-P]；[C-G] Around/Among；[C-E] |
| `MC-relation-agent-object` 观察者—物体关系 | C-MV | 真实或假设 agent frame | C-MC | C-HR；C-ACC | **能按原定义算**：位置/接近远离子集由 pose+位置决定；若承载题另需物体本体朝向，须同时满足 perspective 行的资产要求 | C、R、V、F；新增 ego query | [C-P]；[C-G] Rotation/Around/Among；[C-E] |
| `MC-relation-object-object` 物体—物体关系 | C-MV | 题设视点轴或对象参照轴 | C-MC | C-HR；C-ACC | **能按原定义算**：显式相机/环境轴下的对象关系可算；不能把所有 O-O 都默认视角无关，涉及对象自身前后时另需 intrinsic front | C、R、V、F；新增显式 reference-frame 字段 | [C-P]；[C-G] Around/Among；[C-E] |
| `MC-perspective-self` 自身视角 | C-MV | 当前/给定相机 pose | C-MC | C-HR；C-ACC | **能按原定义算**：参考系由相机直接提供；不可见对象的位置仍须由其他视图提供证据 | C、R、V、F | [C-P]；[C-G]；[C-E] |
| `MC-perspective-other-level1` 他者视角的可见性 | C-MV | 具有语义正面的对象/他者视角 | C-MC | 人工朝向/可见性标注后规则生成；C-ACC | **需要场景里没有的东西，做不了**：缺 object intrinsic front，且真实可见性缺实例级真值。OBB 轴只有几何方向及符号歧义，不能代表沙发等对象“朝哪看” | R、V、F；新增语义正面及视角/可见性标注 | [C-P]；[C-G] Among；[C-E] |
| `MC-perspective-other-level2` 他者视角的相对关系 | C-MV | 以他者语义正面建立的新 frame | C-MC | 人工朝向/布局标注后规则生成；C-ACC | **需要场景里没有的东西，做不了**：核心缺语义正面。若改为“站在 A 面向 B”，会变成显式定义的假设相机任务，应另标，不算替代该子类 | C、R、F；新增 semantic facing 与对象视角变换 | [C-P]；[C-G] Around/Among；[C-E] |

MindCube 的代码可追溯任务族如下。这张表把模板族和上面的交叉标签连接起来，避免“有三种视图采样器”就宣布完成。`q1/q2/...` 是固定 C-G 辅助代码中的分支，**不是论文另行定义的一套互斥 taxonomy，也未验证其覆盖原数据所有 ID**。辅助代码读取 `gt_answer` 后编写推理/地图，并非从渲染图独立产生真值；不能直接作为 InteriorGS label oracle。

| 稳定 ID / 代码可见任务族 | 输入模态 | 参考系 | 答案格式 | 官方 GT / 评分 | InteriorGS 初评与原因 | 可复用模块 | 一手来源 |
|---|---|---|---|---|---|---|---|
| `MC-rotation-current` 当前指定视点的不可见方向查询 | C-MV，2/3/4 视图 | 给定 camera frame | C-MC | C-HR；C-ACC | **能按原定义算**：查询对象在其他图可见、在当前图 FOV 外；无需 semantic front | R、C、V、F | [Rotation helper](https://github.com/mll-lab-nu/MindCube/blob/6d8fd9acdf5b80769373a21cdbfcf068ab356e0f/src/scaffold_curation/reasoning/rotation_reasoning.py)；[C-E] |
| `MC-rotation-mental-turn` 假设转身后的方向查询 | C-MV + 假设旋转 | 更新后的 camera frame | C-MC | C-HR；C-ACC | **能按原定义算**：90°/180°等题设旋转可确定坐标变化 | R、C、F | Rotation helper；[C-E] |
| `MC-around-q1` 跨视点自身移动 | 两图 + Q | 起始 camera frame | C-MC | C-HR；C-ACC | **能按原定义算**：pose 相对变换提供几何 GT，仍需视觉可辨认 | R、C、V | [Around helper](https://github.com/mll-lab-nu/MindCube/blob/6d8fd9acdf5b80769373a21cdbfcf068ab356e0f/src/scaffold_curation/reasoning/around_reasoning.py)；[C-E] |
| `MC-around-q2` 侧视下被挡对象/前后顺序 | 多视图 + Q | 指定侧视 camera frame | C-MC | C-HR；C-ACC | **只能近似**：前后次序可算，原任务需要的真实遮挡状态尚需验证 | R、C、V、F | Around helper；[C-E] |
| `MC-around-q3` 假设转向并前进后的关系 | 多视图 + 有序动作 Q | 逐步更新的 agent frame | C-MC | C-HR；C-ACC | **能按原定义算**：明确动作与步长下可判断接近/远离；Around 观察有效性仍需另一道审核 | R、C、V、F | Around helper；[C-E] |
| `MC-around-q4` 取对象视角判断邻近关系 | 多视图 + Q | 对象 intrinsic frame | C-MC | C-HR；C-ACC | **需要场景里没有的东西，做不了**：需要对象语义正面；bbox 主轴不能代替 | R、C、F；新增 facing | Around helper；[C-E] |
| `MC-among-q1` 中心物体周围的视点移动 | 两图 + Q | 起始 camera frame | C-MC | C-HR；C-ACC | **能按原定义算**：可从受控相机运动得到 GT，需筛到有效 Among 构型 | R、C、V | [Among helper](https://github.com/mll-lab-nu/MindCube/blob/6d8fd9acdf5b80769373a21cdbfcf068ab356e0f/src/scaffold_curation/reasoning/among_reasoning.py)；[C-E] |
| `MC-among-q2-current` 当前自身视点定位周围物体 | 多图 + Q | 给定 camera frame | C-MC | C-HR；C-ACC | **能按原定义算**：位置关系无需对象正面；需跨图身份与有效视图组 | R、C、V、F | Among helper 的 q2/position=1；[C-E] |
| `MC-among-q2-sequence` 假设转向再前进 | 多图 + 动作 Q | 顺序更新的 agent frame | C-MC | C-HR；C-ACC | **能按原定义算**：固定动作含义后可执行求解 | R、C、V、F | Among helper 的 q2/position=2,3；[C-E] |
| `MC-among-q3` 对象视角看到/看不到哪个对象 | 多图 + Q | 中心对象 intrinsic frame | C-MC | C-HR；C-ACC | **需要场景里没有的东西，做不了**：语义正面与可见性真值均待补 | R、V、F；新增 facing/visibility | Among helper；[C-E] |
| `MC-among-q4` 中心对象视角的左右 | 多图 + Q | 中心对象 intrinsic frame | C-MC | C-HR；C-ACC | **需要场景里没有的东西，做不了**：语义正面尚未证实可用 | R、C、F；新增 facing | Among helper；[C-E] |
| `MC-among-q5` 自身视点中的对象—对象关系 | 多图 + Q | 明确的 camera frame | C-MC | C-HR；C-ACC | **能按原定义算**：不必使用对象 intrinsic front，但仍须多图证据 | R、C、V、F | Among helper；[C-E] |
| `MC-among-q6` 周围对象视角中的对象—对象关系 | 多图 + Q | 被指定周围对象 intrinsic frame | C-MC | C-HR；C-ACC | **需要场景里没有的东西，做不了**：必须知道被指定对象自身朝向 | R、C、F；新增 facing | Among helper；[C-E] |
| `MC-translation-transitive` 平移观察下关系传递/反向查询 | 共享锚点的双图 + Q | 保持一致的布局/相机轴 | C-MC | C-HR；分类 C-ACC，默认 overall 排除此 setting | **能按原定义算**：可构造 A–B、B–C 关联并查询 A–C；须区分图像换序、关系求逆与动作时序 | R、C、V、F | [Translation helper](https://github.com/mll-lab-nu/MindCube/blob/6d8fd9acdf5b80769373a21cdbfcf068ab356e0f/src/scaffold_curation/reasoning/translation_reasoning.py)；[C-E] |

MindCube 评分与版本边界：

- 固定代码的基本评分从模型回复提取选项字母，与 `gt_answer` 精确比较；提取失败记错，缺 `gt_answer` 的记录被跳过。实际构建数据时必须提前校验完整答案，不能借跳过机制抬高分数。字母提取器支持 A–E 不代表每题有五个选项。[提取器](https://github.com/mll-lab-nu/MindCube/blob/6d8fd9acdf5b80769373a21cdbfcf068ab356e0f/src/evaluation/core/extractors.py#L15)
- 固定汇总代码跟踪 `translation`，但其 `include_in_overall=False`；`around/rotation/among` 的样本按逐题数量计入 overall，并非三设置宏平均。论文 taxonomy 中的纯平移维度与代码中的 translation setting 应分别记录。新增平移题可单独报告，不能不说明就纳入原 overall。[汇总定义](https://github.com/mll-lab-nu/MindCube/blob/6d8fd9acdf5b80769373a21cdbfcf068ab356e0f/src/evaluation/core/base_metrics.py#L25)
- 普通 QA-SFT 首版只需视觉输入→答案；认知地图、推理链、插值视图属于可选输入/输出配置。若之后训练 map-then-reason，应另评地图有效性、布局/朝向一致性及旋转对齐后的结构指标；不能把地图输入模式与纯视觉 QA 分数混报。[论文输入输出配置及指标](https://arxiv.org/html/2506.21458v2#S3.SS2)
- 本轮没有读取完整 HF 数据内容，因此**不声称已核对全部原始题型 ID、各模板样本量或 Tiny/full 的精确交叉覆盖率**；此处覆盖的是公开 taxonomy 和已读官方辅助代码分支。任何产量/合格率都应由本地候选检查得到，未知处填写“未估”，不能由这些表反推。

#### 核心三套的首版设计含义

建议先落下能够独立验证的几何、相机与有限视图关系：VSI 的计数/框级距离/尺寸/三档方向，MMSI 的 camera-camera、camera-object、object-object、measurement、camera motion 几何子集，以及 MindCube 的 Rotation、自身/明确假设相机参考系与有序视点变换。它们共享场景 oracle 和引用解析，但保留各自的输入、选项和评分 adapter。

涉及 **功能区、对象语义正面、真实遮挡像素、动态对象** 的任务，先列为资产建设项或明确的近似实验。特别是把“对象的视角”改成“站在对象处面向另一个地标”，虽然能得到好用的同域训练题，仍须保留不同 task ID；这样可以利用现有 pose，同时如实记录对原任务的覆盖边界。

上述优先级是基于已核查任务定义和当前资产假设的设计判断，不是已生成数据或测得产量的结论。

<!-- 固定的一手引用：仅用于上述来源列。 -->
[V-P]: https://arxiv.org/html/2412.14171v1#A2.SS1
[V-E]: https://github.com/vision-x-nyu/thinking-in-space/blob/51e089c3ae69b9435e9489058610f5b3964c56a8/lmms_eval/tasks/vsibench/utils.py
[M-P]: https://arxiv.org/html/2505.23764v1#S3
[M-D]: https://huggingface.co/datasets/RunsenXu/MMSI-Bench
[M-E]: https://github.com/InternRobotics/MMSI-Bench/blob/13e58a2b8b30d880d7e8a1e4a6aa1c0feda94cac/vlmeval/dataset/mmsi_bench.py
[C-P]: https://arxiv.org/html/2506.21458v2#A2.SS2
[C-E]: https://github.com/mll-lab-nu/MindCube/blob/6d8fd9acdf5b80769373a21cdbfcf068ab356e0f/src/evaluation/evaluator.py
[C-G]: https://github.com/mll-lab-nu/MindCube/tree/6d8fd9acdf5b80769373a21cdbfcf068ab356e0f/src/scaffold_curation/reasoning
[C-D]: https://huggingface.co/datasets/MLL-Lab/MindCube/commit/9c941b46a6bd65b6914669ef7a579948fc9c8467


### 4.5 ViewSpatial：5 类，不能全部按单图处理

`VS-Eval`：选项字母准确率，按类型和整体报告。官方实现从回答最后一行提取 A–D；不是数值容差评分，也不是所有任务统一四选一。官方 JSON 的人物相对方向题含二选一；场景模拟题可输入多个视角。[官方评分代码][vs-eval]、[官方题目 JSON][vs-data]。

| 稳定 ID | 官方子任务 | 模态 | 参考系 | 答案格式 | 官方 GT 如何构建 | 官方评分 | InteriorGS 初判与真值缺口 | 可复用 | 一手来源 |
|---|---|---|---|---|---|---|---|---|---|
| VS-01 | Cam-Rel. Dir.：相机视角物体相对方位 | 单 RGB | 相机坐标 | 方位词单选，通常 4 项 | ScanNet 物体 3D 坐标与相机变换，角度分箱；人工清理 | VS-Eval | **能按原定义算**；坐标变换可精确；须复现角度区间、复合方位及负选项排斥规则，保留边界余量 | R C V F G | [论文 §3.2][vs-paper] |
| VS-02 | Cam-Obj. Ori.：相机视角头部/注视朝向 | 单 RGB | 人物/动物头部相对相机 | 朝向单选，通常 4 项 | COCO 框与关键点分出头/身体；Orient-Anything 估计角度，人工核验 | VS-Eval | **需要场景里没有的东西，做不了**；缺可用人体/动物头部、身体姿态及注视一致性真值；当前场景标注不能提供此类 GT，须另补合适主体资产及标签 | R C V F | [论文 §3.2][vs-paper] |
| VS-03 | Per-Rel. Dir.：人物视角相对方位 | 单 RGB | 指定人物自身左右/朝向 | 按样本 2 或 4 项；含左右手等题型 | COCO 人物关系由人工标注，并检查指代 | VS-Eval | **需要场景里没有的东西，做不了**；缺人物局部轴、肢体语义和姿态真值；改成“站在 A 面向 B”会变成 VS-05，不能充作此类全覆盖 | R C V F G | [论文][vs-paper]、[题目][vs-data] |
| VS-04 | Per-Obj. Ori.：人物自身视角注视朝向 | 单 RGB | 头部相对身体 | 朝向单选，通常 4 项 | 头部与身体方位角之差映射方向；复杂样本人工处理 | VS-Eval | **需要场景里没有的东西，做不了**；缺头部与身体分别定向的资产和真值；静态物体框无法推导 | R C V F | [论文算法 2][vs-paper] |
| VS-05 | Per-Sce. Sim.：想象站位后的相对方向 | 同场景多帧/多 RGB | 站在 A、面向 B，判断 C | 方位单选，通常 4 项 | ScanNet 唯一实例三元组和覆盖帧；计算指定虚拟站位朝向下的方位 | VS-Eval | **能按原定义算**；显式 A→B 可建立精确参考系，不依赖 A 的语义正面；需多帧覆盖、跨图身份和三元组歧义检查 | R C V F G | [论文][vs-paper]、[多图样本][vs-data] |

### 4.6 EmbSpatial：6 个关系标签

输入是单 RGB；构建 GT 时使用 RGB-D 和 3D 场景信息。`ES-Eval` 为选项准确率，论文分别报告生成答案与选项似然最大值两种协议。其 SFT 中的坐标框输出是辅助训练任务，不应作为第七个 benchmark 子任务。[论文 §2–4][es-paper]、[官方仓库][es-repo]。

官方仅说明利用相机参数和物体 3D 坐标获得 2D 框，没有明确其边界是否来自八角点投影、实例可见轮廓或其他处理；公开仓库未提供生成器。故 ES-01 至 ES-04 当前只能视为投影框关系的近似迁移，不能宣称 GT 完全等价。[ACL 定稿 §2.1][es-final]、[官方仓库][es-repo]。

| 稳定 ID | 官方关系子任务 | 模态 | 参考系 | 答案格式 | 官方 GT 如何构建 | 官方评分 | InteriorGS 初判与真值缺口 | 可复用 | 一手来源 |
|---|---|---|---|---|---|---|---|---|---|
| ES-01 | left | 单 RGB | 图像/相机观察者 | 关系词单选 | 完整 2D 框满足左侧、不重叠关系；框来自场景标注与相机 | ES-Eval | **只能近似**；可计算八点框投影的严格不重叠关系，但官方 2D 框的构建边界未完整公开，未证实与此投影框一致；中心差更不能替代完整框关系 | R C V F G | [论文 §2.1][es-paper] |
| ES-02 | right | 单 RGB | 图像/相机观察者 | 关系词单选 | 完整 2D 框右侧、不重叠 | ES-Eval | **只能近似**；同 ES-01，官方框定义待对齐；不能仅比较中心位置 | R C V F G | [论文 §2.1][es-paper] |
| ES-03 | above | 单 RGB | 图像/相机观察者 | 关系词单选 | 完整 2D 框上方、不重叠 | ES-Eval | **只能近似**；可精确算自定义投影框关系，但与官方框语义的一致性未证实；区别于世界重力轴更高 | R C V F G | [论文 §2.1][es-paper] |
| ES-04 | below | 单 RGB | 图像/相机观察者 | 关系词单选 | 完整 2D 框下方、不重叠 | ES-Eval | **只能近似**；官方框边界定义待核实；倾斜相机下不能用世界 z 替代图像上下关系 | R C V F G | [论文 §2.1][es-paper] |
| ES-05 | close：选择最近对象 | 单 RGB | 相机深度 | 给定对象中选最近者 | 每对象框内平均深度；聚合关系后取最近对象 | ES-Eval | **只能近似**；须可靠度量深度输出，固定有效像素/背景处理；中心欧氏距离或中心 z 不等于框内平均深度 | R C V F G | [论文 §2.1][es-paper] |
| ES-06 | far：选择最远对象 | 单 RGB | 相机深度 | 给定对象中选最远者 | 每对象框内平均深度；聚合关系后取最远对象 | ES-Eval | **只能近似**；同 ES-05，先补深度真值再生成；不能统一改为两物体 Yes/No | R C V F G | [论文 §2.1][es-paper] |

### 4.7 3DSRBench：12 个官方细类

`3D-GT`：真实图像及 HSSD 合成图上的人工标注、二阶段质量检查和人工共识，不能描述为全部由 3D 元数据自动计算。官方未给每个自然语言关系统一的距离或角度阈值；InteriorGS 自动标签需要另行声明几何定义并做人类一致性校准。[论文数据卡][3d-paper]。

`3D-Eval`：单选 2 或 4 项；按题轮换全部选项，**各次都答对才算该题正确（CircularEval）**。FlipEval 是加入水平翻转图并改相应左右标签；不能把“原图和翻转图都答对”自行添加为官方条件。已核官方脚本按去掉选项轮换后缀的题 ID 求乘积，翻转题保持自身 ID。常规/非常规相机合成 split 分别评估，每题仍输入单图。[官方数据说明][3d-data]、[实际评分脚本][3d-eval]。

| 稳定 ID | 官方细类 | 模态 | 参考系 | 答案格式 | 官方 GT 如何构建 | 官方评分 | InteriorGS 初判与真值缺口 | 可复用 | 一手来源 |
|---|---|---|---|---|---|---|---|---|---|
| 3D-01 | height_higher | 单 RGB | 世界重力方向 | 两对象选更高/更低者 | 3D-GT；真实空间高度判断 | 3D-Eval | **只能近似**；可算框高程，须先规定比较中心/底部等哪个量，再校准人工语义；保留俯仰/滚转难例 | R C V F G | [论文][3d-paper]、[原始标签][3d-raw] |
| 3D-02 | location_above | 单 RGB | 世界空间垂直对齐 | Yes/No 二选 | 3D-GT；是否正上/正下方 | 3D-Eval | **只能近似**；可算候选，但“正上方”容差、投影覆盖/距离没有官方通用公式，需标注协议和人工校准 | R C V F G | [论文][3d-paper]、[原始标签][3d-raw] |
| 3D-03 | location_closer_to_camera | 单 RGB | 相机与物体 3D 距离 | 两对象选较近者 | 3D-GT；人工深度/距离判断 | 3D-Eval | **只能近似**；可做明显分离的距离候选；必须声明中心、表面或深度定义，中心值并非官方人工标签公式 | R C V F G | [论文][3d-paper]、[标签][3d-raw] |
| 3D-04 | location_next_to | 单 RGB | 世界 3D 邻近关系 | Yes/No 二选 | 3D-GT；相邻或远离 | 3D-Eval | **只能近似**；框距可候选，缺与物体尺度/场景语义匹配的邻近阈值；不能把任意固定米数说成官方定义 | R C V F G | [论文][3d-paper]、[标签][3d-raw] |
| 3D-05 | orientation_in_front_of | 单 RGB | 参照物自身前后 | Yes/No 二选 | 3D-GT；物体自身朝向下的前后关系 | 3D-Eval | **需要场景里没有的东西，做不了**；缺语义正面；相机前后和框长轴都不能等价替代 | R C V F G | [论文][3d-paper]、[标签][3d-raw] |
| 3D-06 | orientation_on_the_left | 单 RGB | 参照物自身左右 | Yes/No 二选 | 3D-GT；物体局部坐标下左右关系 | 3D-Eval | **需要场景里没有的东西，做不了**；补语义正面/上向/左右轴；不能直接用屏幕横坐标 | R C V F G | [论文][3d-paper]、[标签][3d-raw] |
| 3D-07 | orientation_viewpoint | 单 RGB | 物体自身坐标对相机 | 前/后/左/右四选 | 3D-GT；相机看到物体哪一侧 | 3D-Eval | **需要场景里没有的东西，做不了**；语义正面及轴向对称对象的无效规则缺失 | R C V F G | [原始标签][3d-raw] |
| 3D-08 | multi_object_closer_to | 单 RGB | 世界距离；第三对象作锚点 | 两候选对象中选更近者 | 3D-GT；相对锚点距离比较 | 3D-Eval | **只能近似**；三元组距离可精确算选定代理量；先固定距离定义，排除遮挡和难以指代的实例 | R C V F G | [论文][3d-paper]、[标签][3d-raw] |
| 3D-09 | multi_object_facing | 单 RGB | 主体自身朝向与候选目标 | 候选对象二选 | 3D-GT；朝向目标比较，原始标签含主体面向哪个候选 | 3D-Eval | **需要场景里没有的东西，做不了**；缺主体语义朝向及面向角度准则；仅有位置三元组不足 | R C V F G | [原始标签][3d-raw] |
| 3D-10 | multi_object_viewpoint_towards_object | 单 RGB | 主体局部坐标对另一对象 | 前/后/左/右四选 | 3D-GT；主体哪一侧朝向目标 | 3D-Eval | **需要场景里没有的东西，做不了**；需语义正面和跨对象方位变换；不能用虚拟 A→B 朝向偷换 | R C V F G | [原始标签][3d-raw] |
| 3D-11 | multi_object_parallel | 单 RGB | 对象 3D 方向/轴线比较 | Yes/No 二选 | 3D-GT；两对象是否平行 | 3D-Eval | **只能近似**；框主轴只能给几何近似，需明确物体的比较轴和角度容差；平行不要求同向 | R C V F G | [原始标签][3d-raw] |
| 3D-12 | multi_object_same_direction | 单 RGB | 两对象语义前向比较 | Yes/No 二选 | 3D-GT；是否朝同一方向 | 3D-Eval | **需要场景里没有的东西，做不了**；需带正负方向的语义前向；框无向主轴不能区分同向/反向 | R C V F G | [原始标签][3d-raw] |

### 4.8 BLINK：6 类默认空间核心任务，另列可选相邻能力

BLINK 官方共有 14 类，没有单独固定的“空间子集”。本次默认核心范围采用下表 6 类；这是本次空间迁移的范围选择，不是官方 spatial split。拼图、语义对应、功能对应是可选相邻能力，另表列出；IQ_Test 混合图形规律与空间一致性，不保证每题都属空间推理，故不纳入默认覆盖，保留后文排除说明。Art_Style、Relative_Reflectance、Visual_Similarity、Forensic_Detection 也不纳入本次空间范围；若要求完整 BLINK，则这四类及上述可选/排除类均须另外处理。[官方项目][bl-project]、[数据卡的 14 个配置][bl-data]。

`BL-Eval` 为选项准确率；官方论文用规则与 GPT-3.5 辅助提取选项，选项可为图像。原 BLINK 没有强制 CircularEval。验证集有公开答案，隐藏测试集使用官方评测流程。[论文 §4.1][bl-paper]、[官方仓库][bl-repo]。

| 稳定 ID | 官方任务 | 模态 | 参考系 | 答案格式 | 官方 GT 如何构建 | 官方评分 | InteriorGS 初判与真值缺口 | 可复用 | 一手来源 |
|---|---|---|---|---|---|---|---|---|---|
| BL-01 | Counting | 单 RGB | 当前图像，按问句过滤对象 | 数字作为 4 个选项 | TallyQA 人工计数问答，另采 3 个干扰数字 | BL-Eval | **能按原定义算**；可对有效可见实例计数；必须确认实例标注完整，组合条件须另实现；场景总数不能当画面数量 | R V F G | [论文 §3.2][bl-paper] |
| BL-02 | Relative_Depth | 单 RGB＋两个点标记 | 相机；指定表面像素 | 较近点 A/B 二选 | DIW 人工相对深度标注 | BL-Eval | **只能近似**；需渲染表面深度和标记点有效性；物体中心距离只近似且改变题意 | R C V G | [论文 §3.2][bl-paper] |
| BL-03 | Spatial_Relation | 单 RGB | 随具体 VSR 关系；含观察者/对象关系 | 关系断言 Yes/No 二选 | 继承 VSR 真假标签，GPT 将陈述改写为问题 | BL-Eval | **只能近似**；能先做少数明确方向关系，完整覆盖需逐项 VSR 谓词定义和标注校准；不能全套统一相机左右 | R C V F G | [论文 §3.2][bl-paper] |
| BL-04 | Multi-view_Reasoning | 同对象两个 RGB 视角 | 两相机的相对运动/绕对象方向 | 向左/向右二选 | Wild6D 两帧对应的相对相机运动 | BL-Eval | **能按原定义算**；可记录已知相机轨迹精确标签；须固定左右与环绕方向约定、保证相同目标可辨 | R C V F G | [论文 §3.2][bl-paper] |
| BL-05 | Object_Localization | 单 RGB＋候选框视觉标记 | 图像坐标 | 2 个框选正确框 | LVIS 真值框与加高斯扰动的错误框 | BL-Eval；不是框回归 IoU | **只能近似**；需与可见对象一致的精细 2D 框；3D 框投影不一定等于 LVIS 式框，需人工/实例分割校正 | R C V F G | [论文 §3.2][bl-paper] |
| BL-06 | Visual_Correspondence | 双图＋参考点/候选点 | 跨图同一场景点 | 4 个候选点单选 | HPatches 的真值单应变换给对应点 | BL-Eval | **只能近似**；一般 3D 跨视角需表面点、深度和双向遮挡检查；仅二维单应变换的受限版本可由 RGB 精确制作，但必须标明范围 | R C V G | [论文 §3.2][bl-paper] |

可选相邻能力：下列三行不计入默认空间核心覆盖；原 ID 保留，不重新编号。

| 稳定 ID | 官方任务 | 模态 | 参考系 | 答案格式 | 官方 GT 如何构建 | 官方评分 | InteriorGS 初判与真值缺口 | 可复用 | 一手来源 |
|---|---|---|---|---|---|---|---|---|---|
| BL-07 | Jigsaw | 参考拼块＋两个图像选项 | 2D 图块排列 | 2 个图块选缺失块 | TARA 图像切成 3×3，正确中心块的裁剪索引 | BL-Eval | **能按原定义算**；RGB 裁剪索引可给精确 GT；需新增图像拼接/扰动器，与 3D 空间真值无关 | R | [论文 §3.2][bl-paper] |
| BL-09 | Semantic_Correspondence | 跨实例双图＋点标记 | 对象语义部件对应 | 4 个候选点单选 | SPair-71k 部件关键点对应 | BL-Eval | **需要场景里没有的东西，做不了**；需部件语义、关键点与跨实例对应；同一个实例跨视角跟踪不能替代 | R C V F | [论文 §3.2][bl-paper] |
| BL-10 | Functional_Correspondence | 双对象图＋动作＋点标记 | 动作条件下的功能部位 | 4 个候选点单选 | FunKPoint 的功能对应点标注 | BL-Eval | **需要场景里没有的东西，做不了**；需可供性/动作/功能部件标注；类别相同和几何最近都不提供功能 GT | R V F | [论文 §3.2][bl-paper] |

排除说明（保留 ID `BL-08`）：**IQ_Test 不纳入默认空间覆盖**。官方输入为抽象图形/序列和图像选项，参考系或规律因题而异；答案为图像单选，选项数按样本保留；GT 来自人工收集的题目、解释和核验，评分仍是 BL-Eval。相对当前 InteriorGS，判断为 **需要场景里没有的东西，做不了**：缺抽象图形资产和规则程序；无直接可复用的 3D 模块，仅可复用 QA 格式。若未来只纳入其中空间一致性题，应先逐题筛选，再单独定义子集，不能把全部 IQ_Test 算作空间题。[论文 §3.2][bl-paper]。

### 4.9 SITE：6 个聚合类，以及新增任务的 4 个方向配置

SITE 不是六个有统一公式的生成器。官方汇集 22 个图像来源、8 个视频来源，并在 EgoExo4D 上新增标注；其中包括 BLINK、VSR、3DSRBench、VSI-Bench 等已有题目。它筛选空间样本、转换多选格式并重新分类。因此完整迁移必须保留 `source_dataset`、`source_task`、`original_question_id`、`site_category`；仅生成六类名字对应的少数模板不算复现 SITE。官方没有公布一个能把每个聚合类唯一映射为固定细任务及 GT 公式的规范。[论文表 2、附录 I–II][site-paper]、[官方数据][site-data]。

`SITE-Eval`：每题 2–6 个选项，按样本保留数量；主指标是机会校正准确率：`CAA = (Σ correct_i − Σ 1/k_i) / (N − Σ 1/k_i)`，同时可报告普通准确率。不能统一改成 Yes/No 后比较原分数。论文使用 GPT-4o 提取选项；发布代码使用规则解析，解析失败随机选择，所以必须固定评测版本、解析协议和随机种子。[官方评分实现][site-eval]。

下面 SITE-C01 至 C06 是**官方粗类的覆盖边界**，不是声称已有固定的一项细任务。SITE-N01 至 N04 是另外展开的**两个新增任务各自双向配置**，与粗类重叠，统计覆盖率时不能相加当作十个互斥官方类别。

| 稳定 ID | 官方类别/具体任务范围 | 模态 | 参考系 | 答案格式 | 官方 GT 如何构建 | 官方评分 | InteriorGS 初判与真值缺口 | 可复用 | 一手来源 |
|---|---|---|---|---|---|---|---|---|---|
| SITE-C01 | Counting & Existence：对象出现/缺席、静图或跨帧实例数量 | 单图/视频，按来源 | 当前帧或时间窗口 | 数字/存在性等作为 2–6 选项 | 继承来源人工/程序标签，SITE 筛选和改写；无统一计数公式 | SITE-Eval | **只能近似**；静态可见实例可先构建候选；跨帧规则、动作计数与存在负样本完整性须逐来源确认 | R C V F G | [类别定义][site-paper]、[数据][site-data] |
| SITE-C02 | Spatial Relationship Reasoning：方向、邻近、遮挡、包含等 | 单图/多图/视频，按来源 | 图像、相机或对象自身，按来源 | 关系词或真假等选项；2–6 项 | 继承来源关系 GT；没有统一阈值 | SITE-Eval | **只能近似**；方向可先做；物体局部轴、遮挡表面、包含/接触规则分别缺条件；需按来源关系拆表 | R C V F G | [类别定义][site-paper]、[数据][site-data] |
| SITE-C03 | Object Localization & Positioning：坐标/框/关键点、空间对齐 | 图像或视频，按来源 | 原任务的图像或场景坐标 | SITE 统一为选项；不等于直接坐标回归 | 继承定位标签并转换选项；SITE 不给统一定位误差公式 | SITE-Eval | **只能近似**；可投影框作候选，但精细框、表面关键点和对齐 GT 需补；先确认每个源任务的输出结构 | R C V F G | [类别定义][site-paper]、[数据][site-data] |
| SITE-C04 | 3D Information Understanding：深度、形状、表面、空间尺度及交互 | 单图/多图/视频，按来源 | 物体/相机/世界，按来源 | 2–6 选项，含数值/描述候选 | 继承来源深度/形状等标签；非统一 3D 算法 | SITE-Eval | **只能近似**；框尺度/位置仅覆盖部分；表面深度、细形状、物体交互不由框自动给出 | R C V F G | [类别定义][site-paper]、[数据][site-data] |
| SITE-C05 | Multi-view & Cross-image Reasoning：跨视角对应、隐藏部分和布局整合 | 多图/多视角视频 | 跨相机及对象/场景身份 | 2–6 选项 | 继承源任务 GT，并含新跨视角标注；公式依来源 | SITE-Eval | **只能近似**；静态同场景布局可先构建候选；部件对应、人物跨视角与隐藏形状另需资产/标签 | R C V F G | [类别定义][site-paper]、[数据][site-data] |
| SITE-C06 | Movement Prediction & Navigation：运动轨迹、意图、下一步及路线 | 视频/帧序列，按来源 | 时间、主体、相机/地图，按来源 | 2–6 选项；动作/路线/描述等 | 继承各视频/导航来源标签；官方未规定统一运动公式 | SITE-Eval | **只能近似**；当前仅能覆盖静态场景的相机导航子集；动态物体轨迹/意图所需的时变资产缺失，不能以相机转动覆盖全部 | R C V F G（仅静态导航部分） | [类别定义][site-paper]、[数据][site-data] |
| SITE-N01 | View Association，ego→exo：由第一视角选对应第三视角 | 多图 | 同时刻自我/外部相机、人物身份 | 图像候选单选；按样本选项数 | EgoExo4D 同步视角与时间戳/关键步骤，人工选择和核验 | SITE-Eval | **只能近似**；静态 GS 可做相机/场景匹配近似版；原任务需要人物身份、姿态和同步活动证据 | R C V F G | [新增任务设计][site-paper] |
| SITE-N02 | View Association，exo→ego：由第三视角选对应第一视角 | 多图 | 外部相机与活动者第一视角 | 图像候选单选；按样本选项数 | 同 SITE-N01，方向反转 | SITE-Eval | **只能近似**；须补活动者视角/身份，不可只改变图片顺序就宣称保留原人物任务 | R C V F G | [新增任务设计][site-paper] |
| SITE-N03 | Shuffled Frames Reordering，ego→exo | ego 起止帧＋4 个乱序 exo 帧 | 跨视角共享时间、动态事件 | 排列顺序作为选项；不是自由字符串统一评分 | 同步片段时间顺序与关键动作步骤；人工过滤歧义 | SITE-Eval | **需要场景里没有的东西，做不了**；静态 GS 缺动作进展和时变人物/物体；纯相机路径排序是另一受限任务 | R C（只复用展示/相机基础） | [论文新增任务及图 8][site-paper] |
| SITE-N04 | Shuffled Frames Reordering，exo→ego | exo 起止帧＋4 个乱序 ego 帧 | 同 SITE-N03，方向反转 | 排列顺序选项 | 同 SITE-N03 | SITE-Eval | **需要场景里没有的东西，做不了**；须同步动态资产和可辨事件顺序，静态室内框与墙洞无此 GT | R C（只复用展示/相机基础） | [论文新增任务及图 8][site-paper] |

落到实施清单时，应先把每行收敛成版本化 contract：`source_task`、输入帧规则、参考系、几何量或人工标签定义、无效/歧义条件、选项构造、评分器。共享优先项是多图输入与跨图实例、经过验证的深度/可见性、语义正面和部件标注，以及官方多选/轮换/CAA 评测适配。仅补问题模板无法消除表中 GT 缺口。

[vs-paper]: https://arxiv.org/html/2505.21500v1
[vs-eval]: https://github.com/ZJU-REAL/ViewSpatial-Bench/blob/main/evaluate.py
[vs-data]: https://raw.githubusercontent.com/ZJU-REAL/ViewSpatial-Bench/main/eval/ViewSpatial-Bench.json
[es-paper]: https://arxiv.org/html/2406.05756v1
[es-final]: https://aclanthology.org/2024.acl-short.33.pdf
[es-repo]: https://github.com/mengfeidu/EmbSpatial-Bench
[3d-paper]: https://arxiv.org/html/2412.07825v3
[3d-data]: https://huggingface.co/datasets/ccvl/3DSRBench
[3d-raw]: https://huggingface.co/datasets/ccvl/3DSRBench/blob/main/3dsrbench_v1.csv
[3d-eval]: https://huggingface.co/datasets/ccvl/3DSRBench/blob/main/compute_3drbench_results_circular.py
[bl-paper]: https://arxiv.org/html/2404.12390v1
[bl-project]: https://zeyofu.github.io/blink/
[bl-data]: https://huggingface.co/datasets/BLINK-Benchmark/BLINK/blob/main/README.md
[bl-repo]: https://github.com/zeyofu/BLINK_Benchmark
[site-paper]: https://arxiv.org/html/2505.05456v1
[site-data]: https://huggingface.co/datasets/franky-veteran/SITE-Bench
[site-eval]: https://github.com/wenqi-wang20/SITE-Bench/blob/main/eval_scripts/sitebench/utils.py


## 5. 20 个 train 场景的 CPU 候选普查

用户要求估计每场景每子任务能出多少题。本轮不渲染，因此不能测出 **合格 RGB 题数**：实例可识别、纹理/背面歧义、GS 缺洞、真实遮挡、选项唯一性和多视图必要性都还没有验收。下面给出可复核的候选容量，并把“已执行的几何筛选计数”“结构前置容量”“尚未估计”分开；不拿旧 Read 批次的图审通过率外推新题型。

### 5.1 固定抽样、阈值与能力容量

抽样为 `sorted(random.Random(20261014).sample(sorted(r5 train_scenes), 20))`，没有替换低产场景。主计数使用 main 的冻结提交 `3a964c795f82e4014572d7dc61425cd22d7816f3`、`face_visible_v3`；沿用r5的split、`v10_r5`指代与no-task清单。对照v2使用r5运行自带源码。两轮的目标、参照物文本/几何及467个位置、1,868个视图均一致，代码运行前后哈希不变。

每场景0.8m网格最多检查2,000点，最多24个合法位置×4个yaw（0/90/180/270°），最多24个可唯一指代目标；高度1.5m，墙距≥0.5m、相机半径0.15m、安全余量0.05m。可见性检查目标和指代参照物：框面5×5采样、至少一面≥40%可见且投影面积≥图像0.3%。这些是框几何代理，仍须RGB核验。

关系筛选：左右中心横向投影差≥12px，地面连线与相机横轴夹角≤45°且中心间距≥0.5m；上下投影差≥12px且世界中心高度差≥0.2m；深度中心差≥0.5m且大于横向差；AABB最长边比≥1.2；三物体XY面积≥0.1m²且两两相隔≥0.5m；多图位置基线≥0.5m。类别计数槽要求该类**全部原始带框实例**都被采样并可见，且数量≥2，不把不完整采样子集当全类计数。

主审计约44.5秒CPU完成20/20场景；6,693个原始框中441个通过指代/对象资格，抽样336个。候选数量集中且低产场景真实存在：左右独立对中位9 [0–32]，深度对6 [0–24]，非共线三元组4 [0–48]；多图≥3共享目标的视图对中位仅1.5 [0–94]。按当前保守完整计数条件，总计只有7个类别×视图槽，不能预先承诺每场景每题型均匀配额。

| 容量代码 | 单位/内容 | 层级 | 每场景中位数 [范围] | 20场景合计 |
| --- | --- | --- | ---: | ---: |
| O | 可见独立对象 | 几何筛选 | 10.5 [3–16] | 201 |
| P | 共视无序对象对 | 几何筛选 | 12.5 [1–33] | 296 |
| L | 左右分离无序对象对 | 几何筛选 | 9 [0–32] | 202 |
| D | 中心深度分离无序对象对 | 几何筛选 | 6 [0–24] | 165 |
| S | 可见AABB尺寸差异对 | 几何筛选 | 8 [1–30] | 246 |
| T | 共视非共线三元组 | 几何筛选 | 4 [0–48] | 261 |
| Z | 上下分离对象对×视图槽 | 几何筛选 | 13.5 [0–128] | 648 |
| N | 完整类别计数×视图槽 | 几何筛选 | 0 [0–3] | 7 |
| B | 至少两位置可见的对象 | 多图前置 | 9 [2–14] | 163 |
| M1 | 至少1个共享对象的视图对 | 多图前置 | 99 [16–720] | 3327 |
| M3 | 至少3个共享对象的视图对 | 多图前置 | 1.5 [0–94] | 312 |
| Q | 每处4个固定yaw的合法位置 | 位姿前置 | 24 [14–24] | 467 |
| R | 房间轮廓 | 资产前置 | 7 [1–16] | 145 |
| H | 已有至少2原子动作的证书 | 路径前置 | 13 [13–13] | 260 |
| OB | 有效原始OBB目标 | 资产前置，无视图筛选 | 18.5 [3–24] | 336 |
| OS | 原OBB最长边比≥1.2对象对 | 资产前置，无视图筛选 | 130 [2–241] | 2568 |
| CL | 近共线三元组 | 构型前置，无视图筛选 | 12 [0–20] | 198 |
| RT | 近直角三元组 | 构型前置，无视图筛选 | 231.5 [0–561] | 5001 |

**单位不能混用或相加。** P/L/D/S/T按场景内对象集合去重；Z/N是对象组合×视图槽，同一对象可跨图重复；M1/M3为视图对，Q为位置；R/OB/OS/CL/RT是资产或构型池。很多任务共用同一图和对象，不把这些数加成数据集大小。

### 5.2 每个场景的候选与版本敏感性

下面显示主要单位的20场景明细；全部字段见 [cpu_audit.csv](audit/cpu_audit.csv) 和 [原始JSON](audit/cpu_audit.json)。

|场景|原始框|可唯一指代目标|房间/启发式类型数|合法网格点|采样视图|可见目标|左右独立对|深度独立对|AABB尺寸独立对|非共线三元组|多位置可见目标|≥3共享目标的视图对|现存≥2步路径|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0188_840119|301|3|3/0|120|96|3|1|1|1|0|3|0|13|
|0190_840097|264|9|3/0|27|96|9|12|8|8|7|7|4|13|
|0222_840246|323|58|8/8|84|96|16|12|5|12|4|10|2|13|
|0393_840188|616|7|1/0|166|96|4|2|1|2|1|2|0|13|
|0404_840155|735|5|2/1|234|96|4|2|3|2|1|4|0|13|
|0418_840024|594|26|4/3|212|96|16|17|11|24|19|13|46|13|
|0449_840240|405|18|14/6|65|96|6|0|1|1|0|5|0|13|
|0453_840431|306|27|8/8|54|96|15|5|2|6|0|11|1|13|
|0550_840913|186|22|7/5|42|96|13|11|23|28|34|10|7|13|
|0556_840926|179|16|9/5|28|96|10|23|24|25|31|10|94|13|
|0565_840943|184|17|6/5|21|84|11|15|7|16|12|11|17|13|
|0615_841191|312|52|7/7|90|96|13|24|21|26|36|12|14|13|
|0649_841461|302|33|15/8|120|96|12|3|3|5|1|7|0|13|
|0670_841489|145|19|5/5|104|96|16|32|12|30|46|14|16|13|
|0695_841526|398|19|16/8|130|96|9|2|7|5|4|7|1|13|
|0756_840012|118|9|1/1|70|96|8|16|9|14|16|8|89|13|
|0813_841249|693|53|9/8|190|96|15|7|3|8|0|10|0|13|
|0866_841789|110|14|4/2|14|56|6|2|3|3|1|5|0|13|
|0933_841849|316|19|10/7|81|96|4|1|0|1|0|3|0|13|
|0985_841962|206|15|13/5|54|96|11|15|21|29|48|11|21|13|

原OBB与布局补充只读取同一336目标的原始框，CPU约1秒，**没有套可见性筛选**。两两XY相隔≥0.5m；共线要求三点各自到另两点无限直线的距离最大值≤0.15m、跨度≥1m；直角为某顶点80°–100°。这些是本次探查阈值。有效OBB为336个（46旋转），其尺寸差异对2,568个；近共线198组，近直角5,001组。它们没有证明Around遮挡或Among中心外围配置，也不能与246个已过可见性筛选的AABB尺寸对直接比较。见 [layout_audit.csv](audit/layout_audit.csv)、[计数与输入哈希](audit/layout_audit.json)。

|共享几何能力|v2|v3|变化|变化比例|
|---|---:|---:|---:|---:|
|可见独立目标|213|201|-12|-5.63%|
|左右独立对象对|214|202|-12|-5.61%|
|深度独立对象对|179|165|-14|-7.82%|
|AABB尺寸比较对|264|246|-18|-6.82%|
|非共线三物体组合|276|261|-15|-5.43%|
|≥2位置可见目标|167|163|-4|-2.40%|
|≥3共享目标视图对（结构上界）|323|312|-11|-3.41%|

v3结构墙面/门洞/窗帘改变了遮挡模型，部分场景也增加候选，并非v2集合的单调删减。减少项未经RGB核验，不能直接称旧错误；剩余项也不等于正确题。完整逐场景对照见 [visibility_comparison.csv](audit/visibility_comparison.csv)，保持对象/参照物一致性的校验见 [visibility_comparison.json](audit/visibility_comparison.json)。


### 5.3 逐子任务映射与每场景粗估

下表对覆盖表每个稳定ID给出容量依据；**数值列是前置候选/结构数量，不是已经通过该子任务全部条件的题数**。`中位数 [最小–最大]`按这20场景计算；原始逐场景数见 [task_scene_capacity.csv](audit/task_scene_capacity.csv)（每子任务×每场景一行，含单位、未完成筛选和`not_estimated`）。例如左右对象对不能直接称EmbSpatial整框不重叠题，3个共享对象的视图对也不直接称MMSI合格题。没有伪造“CPU合格→RGB合格”的转换率。

| 子任务稳定ID | 容量代码 | 每场景中位数 [范围] | 还缺什么，如何解释 |
| --- | --- | ---: | --- |
| `VSI-count` | N | 0 [0–3] | 全场景类别≥2的完整共视前置；视频内完整计数未估 |
| `VSI-distance-relative` | P | 12.5 [1–33] | 只数共视对；未筛参照物+四候选、原框内采样距离及选项margin |
| `VSI-direction-easy` | T | 4 [0–48] | 只数三元组；三档角域/边界及视频证据未分别筛选；三行不能相加 |
| `VSI-direction-medium` | T | 4 [0–48] | 只数三元组；三档角域/边界及视频证据未分别筛选；三行不能相加 |
| `VSI-direction-hard` | T | 4 [0–48] | 只数三元组；三档角域/边界及视频证据未分别筛选；三行不能相加 |
| `VSI-route` | H | 13 [13–13] | 已有动作证书条数；不是地标路线候选，视频路线题数未估 |
| `VSI-size-object` | OB | 18.5 [3–24] | 有效原OBB数量；需唯一类别、视觉尺度和视频覆盖筛选 |
| `VSI-size-room` | R | 7 [1–16] | 轮廓面积前置；未形成房间覆盖视频，且不同于点云alpha-shape GT |
| `VSI-distance-absolute` | P | 12.5 [1–33] | 只数共视对象对；未执行原框内采样GT和视频尺度/覆盖验证 |
| `VSI-order` | O | 10.5 [3–16] | 对象池前置；没有时序像素可见性，排列题数未估 |
| `MMSI-position-camera-camera` | M3 | 1.5 [0–94] | 共享锚点视图对；尚未按运动类型筛选，也未证明单图不足 |
| `MMSI-position-camera-object` | B | 9 [2–14] | 同对象跨位置可见；未筛查询视角外对象和互补证据 |
| `MMSI-position-camera-region` | R | 7 [1–16] | 只支持room衍生版的轮廓池；未数相机–房间/物体–房间/房间对；功能区版缺资产 |
| `MMSI-position-object-object` | P | 12.5 [1–33] | 共视对只供前置容量参照；跨图关系题数未估 |
| `MMSI-position-object-region` | R | 7 [1–16] | 只支持room衍生版的轮廓池；未数相机–房间/物体–房间/房间对；功能区版缺资产 |
| `MMSI-position-region-region` | R | 7 [1–16] | 只支持room衍生版的轮廓池；未数相机–房间/物体–房间/房间对；功能区版缺资产 |
| `MMSI-attribute-measurement` | OS | 130 [2–241] | 原OBB尺寸差异对；未筛跨图测量必要性和尺度证据 |
| `MMSI-attribute-appearance` | B | 9 [2–14] | 同对象多位置前置；显式参考视侧变体未筛对称/外观歧义，其他属性缺标签 |
| `MMSI-motion-camera` | M3 | 1.5 [0–94] | 共享锚点视图对；尚未按运动类型筛选，也未证明单图不足 |
| `MMSI-motion-object` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `MMSI-multistep` | M3 | 1.5 [0–94] | 共享锚点对只作前置参照；多步依赖图/删图必要性未估 |
| `MC-setting-rotation` | Q | 24 [14–24] | 合法位置组，每组4yaw；计位置而非视图；未筛查询对象跨图可见且当前FOV外 |
| `MC-setting-around` | CL | 12 [0–20] | 近共线布局池；未采侧视遮挡组、未核真实遮挡及多图必要性 |
| `MC-setting-among` | RT | 231.5 [0–561] | 近直角三元组池；不是已满足中心+外围四视图的Among构型 |
| `MC-pattern-linear` | CL | 12 [0–20] | 近共线布局池；未采侧视遮挡组、未核真实遮挡及多图必要性 |
| `MC-pattern-perpendicular` | RT | 231.5 [0–561] | 近直角三元组池；不是已满足中心+外围四视图的Among构型 |
| `MC-dynamics-translation` | M3 | 1.5 [0–94] | 共享锚点视图对；未细分纯平移/绕物/组合运动及相应构型 |
| `MC-dynamics-rotation` | Q | 24 [14–24] | 合法位置组，每组4yaw；计位置而非视图；未筛查询对象跨图可见且当前FOV外 |
| `MC-dynamics-simultaneous` | M3 | 1.5 [0–94] | 共享锚点视图对；未细分纯平移/绕物/组合运动及相应构型 |
| `MC-dynamics-sequential` | H | 13 [13–13] | 只读已有动作证书容量；未构造对应假设动作、目标关系及有效多图题 |
| `MC-relation-agent-agent` | M3 | 1.5 [0–94] | 共享锚点视图对；未细分纯平移/绕物/组合运动及相应构型 |
| `MC-relation-agent-object` | B | 9 [2–14] | 可跨位置识别的目标前置；查询关系/虚拟视点、多图必要性未估 |
| `MC-relation-object-object` | P | 12.5 [1–33] | 共视对象对前置；未筛Among构型与跨图证据 |
| `MC-perspective-self` | B | 9 [2–14] | 可跨位置识别的目标前置；查询关系/虚拟视点、多图必要性未估 |
| `MC-perspective-other-level1` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `MC-perspective-other-level2` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `MC-rotation-current` | Q | 24 [14–24] | 合法位置组，每组4yaw；计位置而非视图；未筛查询对象跨图可见且当前FOV外 |
| `MC-rotation-mental-turn` | Q | 24 [14–24] | 合法位置组，每组4yaw；计位置而非视图；未筛查询对象跨图可见且当前FOV外 |
| `MC-around-q1` | M3 | 1.5 [0–94] | 共享锚点视图对；未细分纯平移/绕物/组合运动及相应构型 |
| `MC-around-q2` | D | 6 [0–24] | 中心深度差异对；真实遮挡和共线侧视组未筛 |
| `MC-around-q3` | H | 13 [13–13] | 只读已有动作证书容量；未构造对应假设动作、目标关系及有效多图题 |
| `MC-around-q4` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `MC-among-q1` | M3 | 1.5 [0–94] | 共享锚点视图对；未细分纯平移/绕物/组合运动及相应构型 |
| `MC-among-q2-current` | B | 9 [2–14] | 可跨位置识别的目标前置；查询关系/虚拟视点、多图必要性未估 |
| `MC-among-q2-sequence` | H | 13 [13–13] | 只读已有动作证书容量；未构造对应假设动作、目标关系及有效多图题 |
| `MC-among-q3` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `MC-among-q4` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `MC-among-q5` | P | 12.5 [1–33] | 共视对象对前置；未筛Among构型与跨图证据 |
| `MC-among-q6` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `MC-translation-transitive` | M3 | 1.5 [0–94] | 只共享目标前置；未筛A–B/B–C互补图及图间朝向一致性 |
| `VS-01` | L | 9 [0–32] | 相机左右中心分离候选；VS完整角域、ES整框不重叠未筛，左右可交换不能双计 |
| `VS-02` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `VS-03` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `VS-04` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `VS-05` | T | 4 [0–48] | 非共线三元组；未筛多帧覆盖和完整方位边界 |
| `ES-01` | L | 9 [0–32] | 相机左右中心分离候选；VS完整角域、ES整框不重叠未筛，左右可交换不能双计 |
| `ES-02` | L | 9 [0–32] | 相机左右中心分离候选；VS完整角域、ES整框不重叠未筛，左右可交换不能双计 |
| `ES-03` | Z | 13.5 [0–128] | 上下分离对象对×视图槽；ES整框分离/3DSR高度语义与垂直覆盖仍未筛，非独立对象对 |
| `ES-04` | Z | 13.5 [0–128] | 上下分离对象对×视图槽；ES整框分离/3DSR高度语义与垂直覆盖仍未筛，非独立对象对 |
| `ES-05` | D | 6 [0–24] | 中心深度差异对仅代理；框内平均深度及多候选最近/最远排序未做 |
| `ES-06` | D | 6 [0–24] | 中心深度差异对仅代理；框内平均深度及多候选最近/最远排序未做 |
| `3D-01` | Z | 13.5 [0–128] | 上下分离对象对×视图槽；ES整框分离/3DSR高度语义与垂直覆盖仍未筛，非独立对象对 |
| `3D-02` | Z | 13.5 [0–128] | 上下分离对象对×视图槽；ES整框分离/3DSR高度语义与垂直覆盖仍未筛，非独立对象对 |
| `3D-03` | D | 6 [0–24] | 中心深度仅代理；未校准官方人工相机远近语义与真实空间距离 |
| `3D-04` | P | 12.5 [1–33] | 只共视对；邻近语义阈值和困难负例未筛 |
| `3D-05` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `3D-06` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `3D-07` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `3D-08` | T | 4 [0–48] | 三元组前置；未筛锚点距离定义和两候选差距 |
| `3D-09` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `3D-10` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `3D-11` | OB | 18.5 [3–24] | 有效OBB对象池；未筛无向主轴可定义性、角度及成对共视 |
| `3D-12` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `BL-01` | N | 0 [0–3] | 保守的完整类别计数槽；至少2实例，少数类可计；局部画面计数可另放宽但需像素边界 |
| `BL-02` | D | 6 [0–24] | 中心深度仅代理；指定表面像素相对深度、点可见性和标记歧义未做 |
| `BL-03` | L | 9 [0–32] | 仅左右关系子集的候选；其他VSR谓词未估 |
| `BL-04` | B | 9 [2–14] | 同对象多位置前置；未筛绕物左右、固定目标与实际RGB一致性 |
| `BL-05` | O | 10.5 [3–16] | 对象池；可见2D精细框和扰动干扰框尚未构造 |
| `BL-06` | M1 | 99 [16–720] | 视图共享对象对；未求表面点对应、单应关系及双向遮挡 |
| `BL-07` | O | 10.5 [3–16] | 可选拼图：只给可用图像目标池；未渲染图片、未切片，题数未估 |
| `BL-09` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `BL-10` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `SITE-C01` | N | 0 [0–3] | 仅静态计数子集前置；视频/事件出现计数未估 |
| `SITE-C02` | L | 9 [0–32] | 仅相机左右子集前置；须继承各source_task谓词定义 |
| `SITE-C03` | O | 10.5 [3–16] | 只定位对象池；精细框/关键点未筛 |
| `SITE-C04` | D | 6 [0–24] | 仅中心深度代理前置；形状/表面/交互未覆盖 |
| `SITE-C05` | M3 | 1.5 [0–94] | 仅静态多视角前置；EgoExo人物身份/活动证据未覆盖 |
| `SITE-C06` | H | 13 [13–13] | 静态导航前置证书；动态轨迹/意图缺资产，视频题数未估 |
| `SITE-N01` | M3 | 1.5 [0–94] | 仅静态多视角前置；EgoExo人物身份/活动证据未覆盖 |
| `SITE-N02` | M3 | 1.5 [0–94] | 仅静态多视角前置；EgoExo人物身份/活动证据未覆盖 |
| `SITE-N03` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |
| `SITE-N04` | ZERO | 0（当前资产） | 当前资产缺失；见覆盖表所列语义正面/人物/部件/动态依赖 |

复杂多图/视频行的 `未估` 不是 0：表示尚无满足该子任务完整观测条件的 CPU 筛选器。资产缺失行的 `0（当前范围）` 才表示在本轮资产约束下不能出题。表中的计数单位及限额固定；增加视角采样可能增加候选，不可直接乘以 800 场景当发布规模。建议下一阶段先在相同 20 场景渲染少量分层样本，测量各能力族的有效率，再制定全量配额。

## 6. 按单图 → 多图 → 视频分批

以下是 **有相关代码经验的一名工程师的粗估有效人日**（1 人日约 6–8 小时），包含实现与小批核验，不包含等待人工审核、下载新资产、模型训练或全量 GPU 工时。共有底座只计一次；可以并行的模块不应把人日直接当日历天数。工作量与候选规模是两种估算。

| 批次 | 纳入子任务/交付范围 | 主要新工作 | 粗估工作量 |
| --- | --- | --- | --- |
| 0：冻结定义与公共底座 | 固定本表状态、source revision、r5 split、Read 独立身份；选定精确/近似任务边界 | 保留原 OBB、定义私有 GT 证据与单图/多图/视频 schema；统一可见性版本；检查对象指代避免泄漏；公共多选/数值 evaluator；冻结人工审核标准 | 2–4 人日 |
| 1：单图 | EmbSpatial 的六关系（框及深度采用本地代理时另命名）；BLINK Spatial Relation / Relative Depth 及允许的局部 Counting / Object Localization；3DSR 高度/位置/多物体几何子集，排除语义朝向；ViewSpatial 的 scene-centric relation 等可单图非人体子集；SITE 对应静态关系/距离子集 | 单图合法视角与几何间隔筛选；保留足够像素证据；3D 点/框距离和屏幕位置分开；题目和干扰项；单图训练/推理适配与200题分层人工核验 | 5–8 人日；像素深度/mask 方案另加 2–4 人日 |
| 2：多图 | MMSI 相机–相机/相机–物体/物体–物体、相机运动和可支持测量；房间区域衍生题须标近似；MindCube Rotation/Around/Among、视点变换及可支持的视角采择；BLINK Multi-view；ViewSpatial 多图场景想象中可支持部分 | 选择具有基线与互补证据的视图；稳定实例身份；检查单图不能独立解题；构型采样、虚拟相机变换；按各任务原格式生成选项/干扰项（MMSI四选一）和多图模型输入。功能区/语义正面依赖项暂不放行 | 7–12 人日；多步推理的依赖图和删图验证另加 3–5 人日 |
| 3：视频 | VSI 全8主类按视频观测组织（距离遵循框内采样GT；面积和首次出现的代理定义须明示）；相对方向3档；首次出现顺序；地标路线。SITE 只有在不引入真实对象运动的对应视频空间子集 | 与已有动作短轨迹分开的房间巡游/覆盖轨迹；固定帧率/帧数和可达性；每实例可见时间线；路线地标和转弯判据；数值MRA；视频解码/有序帧输入及预算统计 | 6–10 人日；真实表面与实例mask获取另计 |
| 另立资产项目 | MMSI 真实物体运动、功能区/物体部件、3DSR 语义朝向、人/动物、SITE 手物交互/EgoExo动态等 | 新标注或可编辑对象/动态仿真/真人视频；需要先独立验证资产可得性 | 当前不能给可靠完工估计；建议先做2–3人日资产可行性调查，不计入前三批承诺 |

范围选择影响第一批：若只做 VSI/MMSI/MindCube，单图阶段主要是几何 oracle、可见性与模型输入的内部验证，**不会因此完成这三套以视频/多图为主的官方输入协议**。可以保留单图衍生 QA 作为训练数据，但要单列标签与消融，不能在覆盖表中把它填成原协议完成。

每批放行产物应包括：冻结题型配置与版本、同源场景 split、审计 GT、真实输入、可答性/唯一性审核、按子任务与答案统计、官方形式评分和统一研究评分。QA-SFT train 与同域 QA test 必须按现有场景分离；同一物理场景的不同导出、同一图的改写及选项重排不得跨 split。不要为了补齐低产题型而从 val/test 借场景。

## 7. QA-SFT / Act-SFT / RL 预算

### 7.1 当前规模与可比较的计数

第五轮数据快照 `35585b88`，`interiorgs_v10_gen_20261014/sft/` 已有任务及帧清单，正式全量图片/对话尚未完成。只读取清单及证书，在内存中按冻结 prompt 和本地 Qwen2.5-VL-7B tokenizer 计算长度；没有导出训练样本。精确来源/hash和结果见 [`audit/budget_audit.json`](audit/budget_audit.json)，只读复核脚本见 [`audit/budget_audit.py`](audit/budget_audit.py)。

| 项目 | train | val | 性质 |
| --- | ---: | ---: | --- |
| v10 主选轨迹/预计 Act-SFT 对话 | 10,477 | 1,351 | 已存在任务清单；不是审核通过的图像对话 |
| 原子动作 | 48,930 | 6,342 | 读取现有最短路径证书计数 |
| 计划渲染帧（含终点） | 35,156 | 4,590 | render manifest 中的帧事件；train+val 39,746 |
| 预计真正进入 SFT 的图片输入次数 | 24,679 | 3,239 | 当前 `explicit_done=false`，终点帧用于验收，不额外放进模型对话；不等于唯一图片数 |
| assistant 动作回答 turn | 24,679 | 3,239 | 冻结 `action_runs` / 每turn最多5动作的实际 policy；报告中的文字描述若不同，以该快照配置为准 |
| assistant 内容 token | 294,864 | 38,460 | 本地7B tokenizer实数；未含EOS等模板token |
| 预计参与 loss 的 token | 344,222 | 44,938 | assistant 内容 + `<\|im_end\|>` + 换行；`mask_history=false`，不训练prompt |
| 预计总非padding token（文本+视觉+模板） | 10,123,459 | 1,317,721 | 256²输入按Qwen处理为252²，14 patch/2 merge，81视觉token/图；Qwen2-VL ChatML，无截断假设 |
| benchmark题型 QA-SFT 正式数据 | **0** | **0** | 当前还没有本报告定义的新问答集 |
| Read v2 候选/正式数据 | 4 / **0** | 4 / **0** | 另有test候选6；仅助手通过且待人工，不计入benchmark QA-SFT |

视觉 token 的 81 是指定分辨率/处理器下的估算，实际多图/视频分辨率、temporal patch、截断、packing及损失mask会改变数字。源内容token是实算，整条训练序列与loss token是按模板重建的预估，最终须统计训练预处理后的 `input_ids` 与 `labels != -100`。本次估算最长 train 序列2,345 token，未碰16,384上限。没有将伪图或占位路径送入训练。

按当前3 epoch建议配方，Act-SFT 大约消费 **74,037 次图片输入、30.37M 总token、1.033M loss token**。8卡×每卡batch1×累积8为名义全局batch64，约492次optimizer更新（尾批/跨epoch处理以实际loader为准）。不能把“35,156张待渲染图片×3”直接当训练图像曝光预算。

### 7.2 推荐预算对齐方式

建议主对照以 **实际消费的非padding总token（文本+视觉）为训练预算**，固定基座、可训练参数、分辨率与视觉token处理策略、场景划分，并同时报告：唯一视觉输入、累计图片/帧曝光、assistant loss token、optimizer更新、GPU小时。另做一个 **图片/帧曝光对齐** 的小对照，检验收益是否仅来自看了更多场景图。单纯训练步数无法控制不同长度对话的预算。

不建议仅靠重复短答案把 QA 的 loss token 硬凑到 Act：以选择题的单字母答案为例，内容通常1 token、加结束符/换行约3 token；同样24,679次单图回答约74K loss token，低于Act的一遍344K，单按loss匹配会让QA看到约4.65倍图片/重复次数。为了凑长度添加无真值依据的“思维链”也会改变监督内容。数字题长度应实际tokenize，不能全部按1 token计。

RL 与 SFT 还有在线采样、actor/ref/critic forward和环境渲染开销的区别，等optimizer步数或等输入token都不等算力。建议RL同时约束/报告累计观测帧、策略生成token、actor更新token、更新次数及实际GPU小时；若论文主张等计算预算，单独做按实测GPU时间/算力估计匹配的对照，不把SFT模板token视为PPO等价FLOPs。

用户给定的初始化要求已确定：`Qwen base → QA-SFT`、`Qwen base → Act-SFT`、`Qwen base → RL`。现有v10文档的 `Qwen → Act-SFT → RL` 是另一个组合实验，不能冒充同起点的独立RL对照。本轮只记录设计，不启动或改写任何训练。

## 8. 需要用户拍板的事项

| 决策 | 建议 | 影响 |
| --- | --- | --- |
| 范围：8套空间任务还是只做VSI/MMSI/MindCube | **8套作为覆盖清单，首版只承诺静态InteriorGS可支持的能力；共享重复能力，不按benchmark重复堆样本。** 若优先节省工程量，可选三套，但多图/视频成本仍在 | 决定单图阶段是否已有正式交付题型，以及多大范围的GT定义与人工核验 |
| QA-SFT / Act-SFT预算 | 主对照总非padding token对齐；加图片曝光对齐小对照；loss token、步数都记录 | 当前Act每epoch约10.12M总token、24,679图像曝光，QA正式为0。实际QA样本数须在模板和输入帧数确定后反推 |
| 是否渲染视频 | **做一批固定轨迹的视频/有序帧评测**，VSI输入保留序列语义；容器可mp4或固定帧列表，不能把单张图称视频 | 不做时VSI只能标能力衍生单图/多图版；出现顺序、路线及跨视图记忆无法按视频协议验收 |
| 题目措辞和选项格式是否照原benchmark | **训练采用多种独立措辞/随机选项顺序，评测保留“原格式”与“未见改写格式”两组；共享GT和图像** | 原格式可能提高外部QA成绩但混入格式适应。改写组必须按模板族留出，避免只换同义词；字母解析与任务推理分开测 |
| 近似GT能否纳入正式集 | 建议允许，但加明确 `gt_definition` / `approximation`，精确与近似分开报分 | 重点是改变VSI原框内采样距离算法、结构面积vs点云面积、框可见率vs像素、房间区域vs功能区域；保留VSI原框级算法本身不属于定义偏离 |
| 补标注范围 | 第一版不承诺动态对象、语义正面、人体和功能区；若要做再列独立标注/资产预算 | 对MMSI/3DSR/ViewSpatial/SITE部分子任务决定“做不了”是否能转为可做 |
| 公共输入是否给相机位姿 | benchmark样式QA默认不给世界pose；现Act-SFT有`Current camera pose`。如果结论要归因于QA/Act监督形式，须匹配pose输入或做强制pose消融 | 位姿可以直接帮相机运动/参考系题，两组输入信息不一致会混淆监督形式的效果，单独解释不足以消除混淆 |
| 分辨率与每题帧预算 | 初始256²只用于沿用底座；先在小物体/多图题比较更高分辨率，再冻结单图、多图、视频的视觉token上限 | 决定小对象可识别率、MRA误差与训练算力；改变后要重算本节预算 |
| 人工验收与最终规模 | 每批先200题分层人工核验（各子类含正反/边界例），再由实测合格率定配额 | CPU候选不是发布量；目前无法诚实承诺每场景每子任务固定若干合格图 |

Read 单列、使用既有scene split、同一Qwen起点、本轮不生成数据这些已经是用户要求，无需再次决定。待以上范围/预算/视频/格式等选择后，再冻结首版生成规格；本轮工作到文档与研究证据提交为止。

## 9. 审计证据与复现边界

主结果见 [`audit/cpu_audit.json`](audit/cpu_audit.json)，分析源码固定为 VAGEN-Lite 提交 `3a964c795f82e4014572d7dc61425cd22d7816f3`；数据划分和现有轨迹仍来自 `exps/vagen_active_spatial/interiorgs_v10_gen_20261014/`，其记录提交是 `35585b88`。对象选择保留 `v10_r5`，可见性用 `face_visible_v3`。没有使用其他 checkout。

本机复现使用 Python 3.12.13、NumPy 2.2.4、Pillow 12.3.0。CPU几何计算主要依赖NumPy；导入环境基类需要Pillow。使用下列同一环境即可；CUDA已在分析脚本内禁用，BLAS/OMP设为单线程。交付脚本在本目录的 `audit/` 下；以下复制到原审计的临时路径，保持脚本自哈希及脚本间依赖路径可核对。命令会重写同名 `/tmp` 分析结果。

```bash
cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite
cp docs/interiorgs_qa/audit/cpu_audit.py /tmp/interiorgs_qa_cpu_v3.py
cp docs/interiorgs_qa/audit/layout_audit.py /tmp/interiorgs_qa_cpu_layout.py
cp docs/interiorgs_qa/audit/budget_audit.py /tmp/interiorgs_qa_budget_audit.py
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=
mkdir -p /tmp/interiorgs_qa_cpu_v3_snapshot
git archive 3a964c795f82e4014572d7dc61425cd22d7816f3 \
  data_gen/interiorgs_v10 vagen/envs/active_spatial \
  vagen/__init__.py vagen/envs/__init__.py \
  vagen/envs/gym_image_env.py vagen/envs/gym_base_env.py \
  | tar -x -C /tmp/interiorgs_qa_cpu_v3_snapshot
python3 - <<'PY'
from pathlib import Path
from collections import Counter
import json
root = Path('/mnt/umm/users/yinbaiqiao/InteriorGS_meta')
counts = Counter()
for scene in root.iterdir():
    if scene.is_dir() and (scene / 'labels.json').is_file():
        counts.update(p.name for p in scene.iterdir())
Path('/tmp/interiorgs_qa_metadata_files.json').write_text(json.dumps(dict(counts), indent=2))
PY
/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python \
  /tmp/interiorgs_qa_cpu_v3.py
/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python \
  /tmp/interiorgs_qa_cpu_layout.py
/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python \
  /tmp/interiorgs_qa_budget_audit.py
```

需要保留的非代码输入：`InteriorGS_meta` 的1000场景；r5的 `split_manifest.json`、`output/tasks.jsonl`、`code_commit.txt`。元数据目录中的 `occupancy.png` 不参与本次几何计算，只统计其文件存在性。`no_task_objects_20261013.json` 从指定commit的源码快照读取。

主审计已有121项源码/脚本/划分/no-task文件SHA，全部在运行前后核对一致；全量元数据另有有序内容摘要 `79c157277ebb96e60cfa62fd5945bdd33ec87232fc114c7704cd7de7cfa3d7d7`。v3脚本SHA为 `2b6224a10d4122b9bcb90b0b4310d55544376df163b5a2414d877083cc6625e9`。这些足以核对主要几何计算，但原哈希清单不是完整封闭的复现包：原先未哈希现有 `tasks.jsonl`、`code_commit.txt`、独立文件清单JSON，也未保存运行环境版本。

已另存 [`audit/reproduction_inputs.json`](audit/reproduction_inputs.json) 补记上述三个输入的当前SHA和大小；这是运行后补记，不冒充运行开始时已冻结的证据。`tasks.jsonl`只影响现有R5路径计数，不参与新几何视图的筛选。重跑应核对场景列表、采样ID/位置、各计数和元数据摘要；计时、当前工作树HEAD、生成时间等字段不要求逐字节相同。OBB/构型补充脚本自身、主JSON以及20份原始labels都已在补充运行前后核SHA。


预算审计还依赖r5的 `sft/rows.jsonl`、`sft/frames.jsonl`、`job/src_shard1` 和本地Qwen tokenizer。`audit/budget_audit.json` 已记录实际导入的 `qa_states.py`、prompt/export模块、processor/tokenizer配置与LLaMA-Factory模板的哈希；总token和loss仍是指定模板下的重建估算，须在最终预处理后用真实 `input_ids` / `labels` 再核验。

辅助v2复核及版本比较脚本也随附；它们依赖原r5源码快照和对应 `/tmp/interiorgs_qa_*` 文件，不是可脱离数据运行的通用生成器。交付文件仅含调研文字、计数、对象/构型审计证据和只读审计脚本，不含任何新增问答、训练记录或图像。
