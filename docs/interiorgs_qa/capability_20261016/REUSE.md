# 现有代码复用与两图输入验证

真实场景渲染器、场景划分读取和多图导出结构可以复用；旧是非题答案、旧相机采样和旧可见性默认值不能直接用于本轮。两张图的文件解码、训练记录导出、评测请求构造以及 Qwen2.5-VL 处理器均已在 CPU 上实际通过。没有加载模型权重，没有运行训练，也没有据此完成人工审核。当前环境各任务的真实指令、公式和训练数量见 [DESIGN.md](DESIGN.md)，本轮渲染及推理作业状态见 [PROGRESS.md](PROGRESS.md)。

本文的“接口”指供别的程序调用的函数或类。“包围框”是包住标注物体的几何盒子，不是物体真实轮廓；轴对齐框的边平行于世界坐标轴，定向框保留物体自身方向。下面按用途给出可直接定位的文件和函数。

## 旧十类是非题转换器

转换器把原动作任务的完整成功条件改问成“当前是否满足目标”。支持清单在 [active_spatial_qa/contract.py](../../../data_gen/active_spatial_qa/contract.py) 的 `ACTIVE_TASK_TYPES`；实际问句在 [generate_paired_qa.py](../../../data_gen/active_spatial_qa/generate_paired_qa.py) 的 `_question(item)`。支持十类不等于当前训练包含十类。

| 代码支持的任务 | 教模型判断什么 | 当前第十轮正式训练 |
|---|---|---|
| 绝对位置／距离 | 到目标最近侧面的水平距离是否达到要求 | 有 |
| 相对初始状态的位移 | 相对给定起点是否到达指定位置，必须提供参考状态 | 无；本轮不出题 |
| 等距 | 到两个目标的距离是否相等 | 有 |
| 投影左右关系 | 从当前相机看两个对象的左右关系是否成立 | 有 |
| 居中 | 指定对象是否位于要求的中心关系中 | 无；本轮不出题 |
| 遮挡对齐 | 目标是否被指定物体挡住，遮挡物仍可辨认 | 有 |
| 同时入框 | 两个目标是否满足完整入框条件 | 有 |
| 视大小相等 | 两个目标在图中看起来是否一样大 | 无；本轮不出题 |
| 视大小排序 | 指定对象看起来是否达到要求的大小比 | 无；本轮不出题 |
| 占画面高度 | 目标投影框高度是否达到图高的要求比例 | 有；本轮仅保留设计，未启用出题 |

可参考旧转换器保留的父任务、完整目标、相机、状态和轨迹来源字段。不能把旧答案计算直接当作当前环境完整成功判定：`contract.evaluate_state(item,c2w,config=None)` 的规范评分分支只调用基础任务指标，没有合入当前环境的可辨认条件和额外的左右角度限制。旧问句里的“安全余量”“面朝目标”等文字也不一定适用于第十轮，不能照搬。当前完整环境判定应调用 [target_visibility.py](../../../vagen/envs/active_spatial/target_visibility.py) 的 `score_with_visibility(item,c2w,occ)`；能力题只从这份实际测量派生答案，不改原目标。

旧转换器期待的逐步图片字段，与第十轮轨迹实际文件也不同。应读取 `exps/vagen_active_spatial/interiorgs_v10_gen_r10_20261015/sft/frames.jsonl`，按任务编号、步骤和相机连接 `frames/plain/<frame_id>.png`；`sft_trajectories_formal.jsonl` 的 `conversation_image_paths` 只是对话使用的图，不代表每个原始动作后都有图。新入口的绑定检查见 [capability/sources.py](../../../data_gen/interiorgs_qa/capability/sources.py) 的 `validate_trajectory(...)`。

## 渲染、可见性、指代、划分和网页

| 用途 | 现有文件与接口 | 本轮复用办法及边界 |
|---|---|---|
| 真实图像和深度 | [render.py](../../../data_gen/interiorgs_qa/render.py)：`GSRenderer(gs_root,gpu_device=0)`，`set_scene(scene_id)`，`render(c2w,size,K=None)` | 直接复用，返回彩色图、深度和透明度。深度是沿相机前方轴线的距离，单位米；透明度表示像素被场景覆盖的程度。必须传完整相机位姿、焦距、主点及图像尺寸。`ProxyRenderer` 用几何盒子代替真实场景，只能测试，不能生产样题。 |
| 场景与墙 | [scene_ctx.py](../../../data_gen/interiorgs_qa/scene_ctx.py)：`SceneCtx(...)`；[interiorgs_v10/scene.py](../../../data_gen/interiorgs_v10/scene.py)：`load_scene(...)` | 旧上下文默认使用第五轮对象配置和房间边界墙，不能直接当第十轮环境。新入口 [capability/sources.py](../../../data_gen/interiorgs_qa/capability/sources.py) 的 `load_task_scene(task,meta_root)` 显式采用正式题坐标约定、当前排除对象和有厚度的墙。 |
| 相机状态 | [views.py](../../../data_gen/interiorgs_qa/views.py)：`ring_poses(...)`、`frame_candidates(...)` | 不复用旧独立绕物体采样：它可俯仰，相机合法也不代表从任务起点可到达。本轮从实际动作轨迹或经合法动作扩展的水平相机取状态，保留1.5米相机高度。 |
| 环境可辨认条件 | [target_visibility.py](../../../vagen/envs/active_spatial/target_visibility.py)：`FaceGeometry`、`blocked_mask`、`face_visibility` | 直接复用。每个朝向相机的盒面取25个点，至少10个未被挡且在画内，可见面片投影至少占图像面积的0.3%；任一面通过即可。题目借用的描述参照物也要检查。 |
| 真实深度补查 | [interiorgs_v10/depth_check.py](../../../data_gen/interiorgs_v10/depth_check.py)：`depth_hidden(...)`；[visible.py](../../../data_gen/interiorgs_qa/visible.py)：`visible_mask(box,c2w,depth,alpha,K=None)`、`object_view(...)` | 可复用计算函数；前者为环境面采样增加实际深度遮挡，后者估计深度落入定向物体框的像素。后者不是物体分割，不能证明盒内表面一定属于该物体。 |
| 旧“能认出”默认值 | [visible.py](../../../data_gen/interiorgs_qa/visible.py)：`ObjView.recognisable`、`absent`；[views.py](../../../data_gen/interiorgs_qa/views.py)：`framed` | 不直接作为本轮条件。旧判据用整图像素面积、框内可见比例等门槛；`absent` 不能区分画外、遮挡和渲染缺洞；`framed` 默认允许大物体部分入框。与当前环境的“至少一个面”定义不同。 |
| 物体指代 | [common.py](../../../data_gen/interiorgs_qa/common.py)：`GenCtx.name_single`、`name_scene`；正式任务 `target_object.objects` | 旧单图指代允许“仅这张图中唯一”，场景指代还依赖旧配置，因此不照用。保留正式题的 `ref` 描述、`ref_method` 指代办法、`ref_anchor` 描述参照物；新观察中另查同名对象和参照物是否可辨认。 |
| 训练、验证、测试划分 | [scene_ctx.py](../../../data_gen/interiorgs_qa/scene_ctx.py)：`load_split(path)` | 可直接读取第十轮 `split_manifest.json` 中按场景固定的划分。本轮只选其训练场景；不能用旧导出器每隔若干行抽验证样本的测试选项来正式划分数据。 |
| 问法与选项 | [qa.py](../../../data_gen/interiorgs_qa/qa.py)：`Phrasing`、`build_variants`、`Question` | 可参考题目结构和固定种子改写，不能直接继承全部样式；部分旧样式要求标签包装或整段答案。本轮使用完整英文题和单字母回答。距离题选项须按距离升序，旧 `keep_orig_order` 只固定一种问法，不能保证其他问法不重排。 |
| 人工审核页 | [site.py](../../../data_gen/interiorgs_qa/site.py)、[site_template.html](../../../data_gen/interiorgs_qa/site_template.html) | 复用多图、投影框开关、备注和导入导出的交互方式，独立生成能力题页。旧导入没有完整图片与题目绑定，框线还默认按512像素处理，不能不改直接接256像素原图。 |

新页面入口为 [capability/review.py](../../../data_gen/interiorgs_qa/capability/review.py) 的 `build_site(questions_path:Path,out:Path)`，模板为 [capability/review_template.html](../../../data_gen/interiorgs_qa/capability/review_template.html)。原图按字节复制、叠加图单独保存；答案和中文依据默认隐藏。四个审核选项为“答案对／答案错／从图里看不出来／画面质量差”，备注可空，非空备注全部进入分析队列。导出绑定本批清单、每张原图和完整题目的校验值，以及审阅者和审阅时间。保存程序检查结果不会生成人工通过记录。旧空间题和旧 Active Spatial 审核页不覆盖。

## 深度缓存如何读取

旧 [views.py](../../../data_gen/interiorgs_qa/views.py) 的 `ViewBank` 将记录写入 `views.jsonl`，图像和压缩深度位于 `views/<scene>/<view_id>.png`、`<view_id>_256.png`、`<view_id>.depth.npz`。实际读取过旧样题 `0188_840119/8ef2de33dbf56b.depth.npz`：深度为512×512半精度浮点数组，透明度为同尺寸0–255整数，读取时应转浮点并除以255。此旧视角有俯仰，不作为本轮水平相机样题。

`CachedRenderer` 只缓存当前进程，`ViewBank` 没有从磁盘自动恢复的分支；不要为读缓存重新调用渲染。旧 `ViewAnalysis` 固定使用默认焦距和主点，非默认相机应直接调用带 `K` 参数的 `visible_mask`、`object_view`。第十轮完整轨迹目录没有完整的深度与透明度缓存，旧终点检查只为问题帧保存深度，没有透明度；因此本轮选中状态的深度由独立渲染作业补齐，不能冒用旧图的深度。

新缓存由 [capability/render.py](../../../data_gen/interiorgs_qa/capability/render.py) 管理，深度与透明度均按浮点保存，透明度已在0–1范围，**不能再除以255**。主图为256×256像素，裁切检查另存384×384像素加宽图。`cached_ok` 检查图片/深度校验值、数组尺寸、有限数和透明度范围；加宽图只用于检查，不作为模型输入。是否完成真实渲染与同相机对照须看作业产物，文件校验通过本身不代表图片可判断。

## 两图训练格式与评测：已经实际验证到哪一层

监督微调（SFT）是用给定输入学习指定答案。本轮只核对它的输入和导出格式，不更新模型参数。图像处理器是把图片和文字变成模型输入张量的程序；加载它不等于加载模型权重。

| 验证层 | 实际结果 | 尚未代表什么 |
|---|---|---|
| 旧训练导出 | 对真实题 `e66d2dab6398edbe` 调用 [export_lf.py](../../../data_gen/interiorgs_qa/export_lf.py) 的 `build_records(items,res=256,seed,families=('t1',))`，得到1条训练记录、2个图像标记和2条按原顺序保存的图像路径；两张图实际解码均为256×256像素，助手答案为 `D` | 不是训练运行，也不是题目正确性审核 |
| 旧评测请求 | 对同题调用 [evaluate.py](../../../data_gen/interiorgs_qa/evaluate.py) 的 `build_requests(items,families,res)`，图片数量和顺序与训练记录完全一致 | 没有运行旧评测模型 |
| 新双图模型处理器 | 本地 `Qwen2_5_VLProcessor` 实际读取正式轨迹同一任务的初始和下一步两张原图，生成2个图片占位、2行图像网格；图片与文字全部在CPU处理 | 没有加载模型权重，没有验证GPU推理完成 |
| 新纯文字对照 | 同一道问题不附图片，处理器输入没有图片占位或媒体字段，输出也没有像素张量 | 只验证构造与加载，不是无图准确率 |
| 回答监督位置 | [capability/pipeline.py](../../../data_gen/interiorgs_qa/capability/pipeline.py) 的 `training_row(q)` 单独输出用户问题和助手单字母答案；待审训练配置设置不监督用户提示 | 本轮未实际运行完整训练批次装配和逐令牌监督掩码检查，不能称完整SFT加载链已通过 |

处理器验证使用真实任务 `v10_50233748a37f95802cfb4606` 的 `__s0.png`、`__s1.png`。有图输入产生227个文字/图像占位展开后的输入令牌；同一题的纯文字输入产生61个输入令牌。这些数字仅说明该示例的输入长度，不代表整批题的训练量。实际版本为 transformers 4.56.1、vLLM 0.11.0、torch 2.8.0；处理器使用该版本默认的快速预处理。库启动时尝试探测CUDA而提示不可用，最终所有张量都在CPU。

新评测入口 [capability/evaluate.py](../../../data_gen/interiorgs_qa/capability/evaluate.py) 接受 `--questions --model --out`，有图和纯文字请求共用一次模型加载，采用确定性的最高概率回答，严格解析单个选项字母；统计每类随机水平、固定选择最常见字母的成绩和准确率区间。旧入口缺少真正去掉图像标记的纯文字分支，且旧组名排序不兼容新能力名称，因此本轮保留旧评测不动，使用独立新入口。模型结果不会进入审核题旁或用于筛题。

现有多图导出没有结构性障碍；剩余运行验证是本轮实际基座推理及未来训练前的完整批次监督位置检查。训练配方如需导出应显式采用 `exps/vagen_active_spatial/interiorgs_v10_train_r10_20261015/lf_formal/train.yaml`，旧 `export_lf.ACT_RECIPE` 默认指向更早一轮，不能默用。本轮不提交训练。

## 可复核证据

独立保存于 [cpu_evidence/reuse_validation_v1_20261016](../../../exps/interiorgs_capability_qa_20261016/cpu_evidence/reuse_validation_v1_20261016/)，未覆盖已有产物：

- `two_image_export_probe.json`：旧真实双图题的导出、评测请求、图片尺寸及校验值。
- `capability_processor_probe_20261016.json` 与同名 `.py`：本地处理器的有图/纯文字实际CPU输出及复跑脚本。
- `capability_evidence_audit_20261016.md`、`capability_evidence_probe_20261016.json` 与同名 `.py`：4个真实场景的读框、相机与证据数组核查。共853个有效定向框全部读到；控制深度数组只验证接口，不能当真实渲染遮挡证据。
- `capability_reuse_audit_20261016.md`：最初的只读接口核查记录，保留当时的未完成项；最新接入状态以上文和当前代码为准。
- `receipt.json`：上述文件的校验值，并明确记录未加载模型、未训练、没有人工审核。

这些证据只证明对应程序路径和输入能读，不证明样题已经正确或可训练。遮挡像素是否确实来自题目指定物体、原图是否可辨认，仍需要真实图片和用户审核。
