# Active Spatial Evaluation

独立的评估系统，用于衡量 VLM agent 在 3D 空间导航任务中的表现，支持 **9 种任务类型分解**、**多 agent 对比** 和 **训练检查点评估**。

## 目录结构

```
evaluation/
├── run_eval.py             # CLI 入口
├── compare.py              # 多 agent 横向对比
├── eval_config.py          # 配置定义
├── eval_runner.py          # 核心评估循环（直接驱动环境，无需 HTTP 服务）
├── metrics.py              # 分任务指标计算
├── agents.py               # Baseline agents（random, heuristic, constant）
├── model_agent.py          # VLM 模型 agent（vLLM / OpenAI / Claude / Gemini）
├── configs/                # 预定义 YAML 配置
│   ├── eval_random.yaml
│   ├── eval_heuristic.yaml
│   ├── eval_frozen_vlm.yaml
│   └── eval_trained_model.yaml
├── scripts/                # Shell 脚本
│   ├── smoke_test.sh           # 快速验证（5 episodes）
│   ├── run_baselines.sh        # Random + Heuristic baselines
│   ├── run_frozen_vlm.sh       # 未训练 VLM zero-shot
│   ├── run_trained_model.sh    # 评估训练检查点
│   └── run_full_eval.sh        # 完整评估流水线
└── outputs/                # 评估结果输出
```

## 快速开始

### 1. 冒烟测试

验证评估管线能否正常运行（3 个 episode，不需要模型 GPU）：

```bash
cd /scratch/by2593/project/Active_Spatial/VAGEN

python evaluation/run_eval.py \
    --agent random \
    --max-episodes 3 \
    --max-turns 3 \
    --render-backend none \
    --output-dir evaluation/outputs/smoke_test
```

### 2. 运行 Baseline 评估

```bash
# Random baseline（下界）
python evaluation/run_eval.py --config evaluation/configs/eval_random.yaml

# Heuristic oracle baseline（使用 ground-truth 几何信息导航，近似上界）
python evaluation/run_eval.py --config evaluation/configs/eval_heuristic.yaml
```

### 3. 评估未训练的 VLM（Zero-Shot）

```bash
python evaluation/run_eval.py --config evaluation/configs/eval_frozen_vlm.yaml
```

### 4. 评估训练后的检查点

```bash
bash evaluation/scripts/run_trained_model.sh /path/to/checkpoint_step_200
```

### 5. 完整评估（所有 agent + 对比）

```bash
bash evaluation/scripts/run_full_eval.sh /path/to/checkpoint
```

## CLI 参数

```
python evaluation/run_eval.py [OPTIONS]

Agent:
  --agent {random,heuristic,constant,model,frozen}   Agent 类型
  --config PATH                                       YAML 配置文件（覆盖其他参数）

Environment:
  --jsonl PATH            JSONL 测试数据路径
  --gs-root PATH          3DGS 数据根目录
  --render-backend {local,client,none}   渲染方式
  --gpu-device INT        渲染 GPU 编号
  --success-threshold F   成功判定阈值（默认 0.85）

Model（agent=model 时）:
  --provider {vllm,openai,claude,gemini}   模型提供方
  --model-name STR        HuggingFace ID 或 API 模型名
  --checkpoint PATH       训练检查点路径
  --tp INT                vLLM tensor parallel size
  --temperature F         采样温度（评估建议 0.1）

Evaluation:
  --max-episodes INT      评估 episode 数（默认全部）
  --max-turns INT         每 episode 最大 LLM 轮数（默认 20）
  --task-types T [T ...]  过滤特定任务类型
  --seed-offset INT       种子偏移

Output:
  --output-dir PATH       结果输出目录
  --eval-name STR         评估运行名称
  --wandb                 启用 WandB 日志
  --save-trajectories     保存完整 episode 轨迹
  --verbose               详细输出
```

## 评估指标

### 总体指标

| 指标 | 含义 |
|------|------|
| `success_rate` | 成功率（final score ≥ threshold） |
| `mean_final_score` | 最终得分均值 [0, 1] |
| `mean_score_improvement` | 最终得分 − 初始得分（正值 = 在改善） |
| `spl` | Success weighted by Path Length（效率） |
| `monotonic_improvement_rate` | score 单调上升的 episode 比例（有方向感 vs 随机抖动） |
| `mean_steps` | 平均执行步数 |
| `mean_collisions` | 平均碰撞次数 |
| `mean_action_validity` | 动作格式正确率 |

### 9 种任务分解

指标按任务类型分别计算，同时按 3 大类聚合：

| 类别 | 任务 |
|------|------|
| **Metric Distance** | absolute_positioning, delta_control, equidistance |
| **Projective Relation** | projective_relations, centering, occlusion_alignment |
| **View Perspective** | fov_inclusion, size_distance_invariance, screen_occupancy |

### 输出示例

```
task                     | n  | success% | final_score | improvement | spl   | steps | collisions | monotonic%
-------------------------+----+----------+-------------+-------------+-------+-------+------------+-----------
ALL                      | 299 | 12.4     | 0.521       | +0.089      | 0.034 | 18.3  | 2.1        | 65.2
-------------------------+----+----------+-------------+-------------+-------+-------+------------+-----------
absolute_positioning     | 80  | 18.8     | 0.612       | +0.142      | 0.051 | 15.1  | 1.8        | 72.5
delta_control            | 40  | 15.0     | 0.580       | +0.110      | 0.042 | 16.8  | 2.0        | 70.0
screen_occupancy         | 80  | 10.0     | 0.489       | +0.065      | 0.028 | 20.2  | 2.3        | 60.0
...
```

## 多 Agent 对比

```bash
python evaluation/compare.py \
    evaluation/outputs/eval_random/results_random.json \
    evaluation/outputs/eval_heuristic/results_heuristic.json \
    evaluation/outputs/eval_frozen/results_model.json \
    evaluation/outputs/eval_trained/results_model.json
```

输出包含：
- 总体指标横向对比表
- 按任务类型的 success rate 对比
- 按类别的 success rate / score 对比

## Baseline Agents

| Agent | 说明 | 用途 |
|-------|------|------|
| `random` | 随机选择合法动作 | **下界**，任何有效模型应显著优于此 |
| `heuristic` | 使用 oracle 几何信息（目标区域坐标）计算最优动作方向 | **近似上界**，衡量任务本身的可解性 |
| `constant` | 每步执行相同动作（如 `move_forward`） | Sanity check |
| `model` (frozen) | 原始 Qwen2.5-VL-3B 无 RL 训练 | 量化 RL 训练带来的增益 |
| `model` (trained) | 经过 PPO 训练的检查点 | 实际评估目标 |

## 输出文件

每次评估生成：

```
evaluation/outputs/<eval_name>/
├── eval_config.yaml              # 评估配置存档
├── results_<agent>.json          # 完整结果（指标 + 每 episode 详情）
└── results_<agent>_summary.txt   # 人类可读摘要
```

`results_*.json` 包含：
- `metrics.overall` — 总体指标
- `metrics.by_task_type` — 9 种任务各自的指标
- `metrics.by_category` — 3 大类的聚合指标
- `episodes` — 每个 episode 的 seed、task_type、score_trajectory、success 等
