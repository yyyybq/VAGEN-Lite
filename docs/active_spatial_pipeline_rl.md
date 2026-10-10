# 第 2 阶段：Active Spatial RL 与权重交接

这一阶段从混合 SFT 的完整 HF 权重继续训练相机导航能力，输出可供 StarVLA 初始化的 HF actor。[上一步：数据和 SFT](active_spatial_pipeline_data.md) · [返回总览](active_spatial_end_to_end.md)。

## 保留 SFT 的验证场景

RL 重新读取的是原始任务 JSONL，不是 SFT conversation。若把 SFT 验证场景又放回 RL 训练集，后续报告就不能继续把它当作未见验证集。因此使用上一步的 scene split 过滤原始任务。

正式数据还必须通过仓库已有的运行时回放准入。若尚无有效证书，在真实渲染环境中执行：

```bash
export PRIVATE_EVIDENCE=/absolute/path/to/task_replay_evidence.jsonl
"$ACTIVE_PY" scripts/verify_active_spatial_dataset.py \
  --manifest "$TASKS" --evidence "$PRIVATE_EVIDENCE" --env-yaml "$ENV_YAML" \
  --output "$RUN/tasks.contract.json"
```

证据 JSONL 每行包含唯一的 `task_id`、逐 turn 的 `actions`（例如 `["turn_left|move_forward", "move_left"]`）和 `initial_rgb` 的绝对路径。使用已审计的成功回放证据及初始图片；所有原始任务都必须有证据，不能用标记字段伪造成功。缺少可达成功轨迹的任务应先修复或从正式任务库中排除，再重新生成数据和划分。冻结目录名、几何审计以及 SFT 成功数量都不能代替这份证书。

```bash
export PPO_TEMPLATE="$ROOT/exps/vagen_active_spatial/R1-gate-aligned-pilot8-v2-20261005/frozen/pilot.yaml"

"$ACTIVE_PY" scripts/prepare_active_spatial_rl.py \
  --template "$PPO_TEMPLATE" --env-yaml "$ENV_YAML" --tasks "$TASKS" \
  --dataset-contract "$RUN/tasks.contract.json" \
  --split-manifest "$RUN/mix_sft/split_manifest.json" \
  --model "$SFT_HF" --steps 150 --out "$RUN/active_rl"
```

准备入口生成新的 `train.yaml`、`train_env.yaml`、`val_env.yaml`、分开的 task JSONL 和 `handoff.json`。它替换 actor/critic 的模型路径、训练步数、optimizer schedule 和输出目录，关闭恢复旧任务，并将环境的 `prompt_format` 设置为混合 SFT 使用的 `no_think`。

入口先校验证书覆盖全部任务且物理执行协议一致，再为两个子集启用 `require_verified_dataset: true`。子集保留原始行和同一证书；改变任务文本、相机、目标或动作步长后必须重新验证。若已经有证书，可直接传入它，不重复回放或覆盖旧证书。

输入 template 必须是完整保存的 PPO 配置；当前示例复制的是 R1 的 `no_concat_gae`、`concat_multi_turn=false` 路线。旧 pilot 的固定终点 wrapper 会断言 700 步 scheduler、限制 endpoint，甚至在集群入口中重设 pretrained 模型路径；新实验直接调用通用 `vagen.main_ppo`。不要同时套用 `sco_r1_gate_aligned_pilot.sh`。

准备后审查 batch、GPU 数、模型大小、rollout 资源与 `save_freq/test_freq`。这几个值从 template 继承，不会按模型规模自动推算。此入口支持单个 `ActiveSpatial` 环境条目以及独立 HTTP 渲染服务；整个环境远程托管的 `RemoteEnv` 走现有 [远程环境说明](active_spatial_remote_env.md)，需要另配环境服务路径。

## 启动 PPO

先在 GPU 节点确认渲染服务能提供真实图像，再在与该模型兼容的 VAGEN 环境运行：

```bash
export PYTHONPATH="$ROOT/verl:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
"$ACTIVE_PY" -m vagen.main_ppo \
  --config-path "$RUN/active_rl" --config-name train
```

应先准备一个独立的新目录，用较小预算验证：真实环境 rollout、actor/critic backward、optimizer 更新、checkpoint 保存/恢复均成功，再启动正式预算。不要把一个只有 forward 的测试记录成 RL 训练完成。若配置仅保存末尾 checkpoint，确认短程测试也实际产生了保存文件。

关注任务成功率、格式错误、primitive action 数、碰撞、截断、原始奖励与 shaping。canonical 成功以布尔门限为准；高 shaping 分数或 loss 下降不能替代成功率。训练集与验证集应分别记录 task/scene 覆盖。

## 导出可加载的 HF actor

```bash
export RL_ACTOR_DIR="$RUN/active_rl/checkpoints/global_step_150/actor"
export SFT_RL_HF="$RUN/active_rl_hf"
PYTHONPATH="$ROOT/verl${PYTHONPATH:+:$PYTHONPATH}" \
  "$ACTIVE_PY" -m verl.model_merger merge \
  --backend fsdp --local_dir "$RL_ACTOR_DIR" --target_dir "$SFT_RL_HF"
```

若 checkpoint 已附带完整 `actor/huggingface` 权重，可直接选择它，无须重复合并。某些 verl 保存格式的 `huggingface/` 只有配置、tokenizer 或 processor；必须检查 safetensors 及 index 指向的全部 shard，不能仅看目录存在。合并使用训练该 checkpoint 的 PyTorch/verl 环境。

| 文件形式 | 可交给下阶段吗 |
| --- | --- |
| 完整 HF config＋权重＋tokenizer＋processor | 可以，下一阶段还会检查 action query token 和尺寸 |
| verl 的 `model_world_size_*_rank_*.pt` | 先用对应 model merger 导出 |
| 只有配置的 `actor/huggingface/` | 不可以 |
| LoRA adapter | 先合并原始 base |
| StarVLA `pytorch_model.pt` | 是后续阶段产物，含动作头，不是此处的 HF actor |

## 在机器人迁移前固定空间能力基线

沿用同一 frozen ID/OOD 套件，在 `base`、`mix_sft`、`mix_sft_rl` 上记录环境成功率与 QA 指标。通用历史 `evaluation/configs/eval_trained_model.yaml` 带有旧动作角度、图像分辨率和分数阈值，不宜直接复用。

可以从新生成的环境 YAML 生成评测配置：

```bash
export EVAL_HF="$SFT_RL_HF"
export EVAL_TASKS="$RUN/active_rl/val.jsonl"
export EVAL_OUT="$RUN/spatial_eval_before_robot"
"$ACTIVE_PY" - <<'PY'
import os
from pathlib import Path
from omegaconf import OmegaConf
root = Path(os.environ['RUN'])
entry = OmegaConf.to_container(OmegaConf.load(root/'active_rl/val_env.yaml'), resolve=True)['envs'][0]
env = entry['config']
env['jsonl_path'] = os.environ['EVAL_TASKS']
cfg = {
    'eval_name': 'spatial_before_robot', 'output_dir': os.environ['EVAL_OUT'],
    'agent_type': 'model', 'max_steps_per_episode': entry['max_turns'],
    'seed_offset': 0, 'save_trajectories': True, 'env': env,
    'model': {'provider': 'vllm', 'model_name': os.environ['BASE_HF'],
              'checkpoint_path': os.environ['EVAL_HF'], 'temperature': 0.0,
              'tensor_parallel_size': 1, 'max_tokens': entry['response_length_per_turn']},
}
OmegaConf.save(OmegaConf.create(cfg), root/'spatial_eval.yaml')
PY
"$ACTIVE_PY" evaluation/run_eval.py --config "$RUN/spatial_eval.yaml"
```

将 `EVAL_TASKS` 换成独立冻结的 ID/OOD 文件时，也要把 `env['dataset_contract_path']` 换成覆盖该文件的证书，并使用新 `EVAL_OUT`，得到正式报告。QA 使用独立 held-out bank 运行 `model_qa_eval.py` 和 `qa_eval.py`，记录 accuracy、Yes/No 分布与格式错误率；训练 bank 的准确率只能作为拟合诊断。

下一步见 [StarVLA → RLinf](active_spatial_pipeline_robot.md)。
