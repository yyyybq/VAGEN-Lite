# 环境能力题：2026-10-16 进度

结论：正在核查当前 v10 任务定义和真实轨迹图片，尚未生成或放行本轮能力题。

## 工作位置与边界

- 独立分支：`capability-qa/20261016`；独立工作目录：`/mnt/umm/users/yinbaiqiao/probe_spatial/.worktrees/VAGEN-Lite-capability-qa-20261016`。
- 整理分支 `cleanup/20261016` 尚未合入 main；不修改共享 VAGEN-Lite 工作目录，不推 main。新分支准备更新到已查询到的远端 main `168d443b`。
- 代码放 `data_gen/interiorgs_qa/capability/`；本轮产物放 `exps/interiorgs_capability_qa_20261016/`。旧 production_v3 和既有 benchmark 样题原样保留。
- GPU 仅允许 zoetrope 的 SCO 单卡作业，用于必要渲染与 Qwen2.5-VL-7B 有图/无图检查；不训练、不以模型答案筛选样题。
- 用户审核前，所有能力题均待人工审核，正式训练放行数量为零。

## 已核实

- 读取根 STATUS、v10 STATUS、interiorgs_qa PROGRESS；它们部分描述是历史状态，以实际路径和数据为准。
- 当前第十轮正式任务和场景划分位于共享目录 `exps/vagen_active_spatial/interiorgs_v10_gen_r10_20261015/`；仅作只读来源。
- 正式任务包含六类；真实成功条件、当前训练去除名单、轨迹帧绑定正在逐项核查。
- 现有 QA 代码支持多图，但默认相机采样、旧物体描述和可见性不能未经适配直接用于本任务。

## 下一步与交付检查

1. 固定数据与代码来源；写六类任务定义、能力题规则、模糊区间和待定事项。
2. 从四个正式 train 场景的可达状态制作样题，保存每个候选的通过/排除理由及几何依据；真实深度不足时不造标签。
3. 新增功能测试通过后发布私有审核页，跑有图/无图基座检查并清理本轮单卡作业；报告图片预算估计，等待用户审核。

本轮用户完整要求：`/mnt/umm/users/yinbaiqiao/.codex/attachments/e5b2d1b0-a19f-4ec5-b513-19dd7799cf2c/pasted-text.txt`。
