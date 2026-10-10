# 第 1 阶段：任务、轨迹、QA 与混合 SFT

这一阶段得到一个能执行 Active Spatial 动作、同时接受空间 QA 监督的 HF VLM。先冻结任务和协议，再产生两类监督；两类数据必须使用同一个验证划分。[返回总览](active_spatial_end_to_end.md)。

## 固定任务与环境协议

[`data_gen/active_spatial_pipeline/`](../data_gen/active_spatial_pipeline/README.md) 负责从场景生成任务；其 `run_pipeline.py` 输出仍是未验证候选，不能直接当作 R1 正式数据。当前 R1 语料必须满足 canonical task/camera contract，并通过生成、几何和运行时检查。已有可用的冻结输入时优先复用，不在本次流程中修改它。正式任务库应在混合划分前通过[回放证书检查](active_spatial_pipeline_rl.md#保留-sft-的验证场景)，避免训练到中途才发现无法进入 RL。

本机示例配置与任务库：

```bash
export ROOT="$PWD"
export ACTIVE_PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
export STARVLA_PY=/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python
export BASE_HF=/absolute/path/to/Qwen2.5-VL-HF
export ENV_YAML="$ROOT/exps/vagen_active_spatial/R1-gate-aligned-pilot8-v2-20261005/frozen/train.yaml"
export TASKS="$ROOT/exps/vagen_active_spatial/R1-clean-Projective-v0/frozen_v1/train.jsonl"
export GS_ROOT="$ROOT/exps/vagen_active_spatial/R1-clean-Projective-v0/assets/ready"
export R1_RENDER_URL=http://RENDER_HOST:8915/render
export RUN="$ROOT/outputs/spatial_transfer_v1"
mkdir -p "$RUN"
```

这些是本机已有语料的路径示例，不是随 Git 分发的数据。迁移机器时需要恢复任务、标签、结构文件、GS 场景和模型。将外部正式 ID/OOD 测试任务及其保留场景排除在 `$TASKS` 之外；后面的场景划分只负责输入语料内部的 train/val。

不要直接运行带历史 UID、固定模型和固定运行目录的 SCO pilot 包。渲染服务可以单独在 GPU 节点启动：

```bash
bash examples/train/active_spatial/start_gs_render_http_service.sh \
  --gs-root "$GS_ROOT" --host 0.0.0.0 --port 8915 --gpus 0 \
  --max-workers 1 --max-inflight 1 --admit-timeout 900 \
  --conda-env /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
```

## 生成可回放的轨迹监督

```bash
"$ACTIVE_PY" data_gen/active_spatial_sft/run_r1_sft_pipeline.py \
  --env-yaml "$ENV_YAML" --jsonl-path "$TASKS" \
  --output-dir "$RUN/trajectories" --qwen-format jsonl \
  --no-think --beam-width 16
```

入口从环境 YAML 继承动作空间、步长、旋转角、预算、渲染与成功判据；搜索后的轨迹经真实环境回放，才能导出带图片的 SFT。`sft_data.jsonl` 保留 scene/task ID、原始 conversation、分数与回放审计；`qwen_vl_sft.jsonl` 是训练格式。混合准备入口读取前者，因为后者可能已经去掉场景信息。

检查 `generation_manifest.json`、`pipeline_summary.json` 和 `visualization/index.html`。至少确认成功轨迹数量、任务覆盖、实际图片、碰撞与回放是否一致。几何搜索成功或生成了 JSONL 不代表有效的图像监督。审计分数、隐藏目标点和最短路径证书只能留在 metadata 中。

## 从同一任务生成 QA

```bash
"$ACTIVE_PY" -m data_gen.active_spatial_qa.generate_paired_qa \
  --input "$TASKS" --output-dir "$RUN/qa_geometry" --split train \
  --scene-root "$GS_ROOT" --env-yaml "$ENV_YAML"

"$ACTIVE_PY" -m data_gen.active_spatial_qa.runtime_compare \
  --bank "$RUN/qa_geometry/manifest.jsonl" --output "$RUN/qa_runtime_compare.json"

"$ACTIVE_PY" -m data_gen.active_spatial_qa.render_paired_bank \
  --bank "$RUN/qa_geometry/manifest.jsonl" --output-dir "$RUN/qa_rgb" \
  --backend http --renderer-url "$R1_RENDER_URL" --gs-root "$GS_ROOT" \
  --width 256 --height 256
```

`--scene-root` 必须能找到各场景的 `labels.json` 和 `structure.json`。仅有渲染服务不够：没有几何上下文，生成器会保留 `coordinate_unconfirmed` 审计记录并撤回标签。指定了 `--split train` 也不会自动产生有效监督。

QA 的正负标签取自实际状态下的成功判据；`positive_candidate` 不是强制 Yes，`negative_candidate` 不是强制 No。查看 `summary.json`、`errors.jsonl`、contact sheet，以及 Yes/No 比例。历史 bank 可能未保存 canonical camera 信息；应在新目录重新生成和渲染。

修复后的 `runtime_compare` 保留 canonical 版本与评分配置；无有效复核样本、有错误或标签不一致时退出码为 1。渲染会为 canonical 行使用与环境相同的 H1 内参缩放。渲染失败、空图、无标签记录仍不能用于 SFT。

## 用同一场景划分混合

```bash
"$ACTIVE_PY" scripts/prepare_active_spatial_sft.py \
  --trajectory "$RUN/trajectories/sft_data.jsonl" \
  --qa-bank "$RUN/qa_rgb/manifest.jsonl" \
  --model "$BASE_HF" --qa-probability 0.5 --val-fraction 0.1 --seed 42 \
  --out "$RUN/mix_sft"
```

输出包括 `trajectory_train/val.jsonl`、`qa_train/val.jsonl`、`dataset_info.json`、`train.yaml` 和 `split_manifest.json`。训练记录仅包含 `messages`、`images`；所有图片会检查存在性和可读性。无效 QA 被计入排除原因；已标为 test/val 的源 bank 会被拒绝。

划分单位是 scene，因此同一场景的轨迹、QA parent 及其多个 pose 不会跨 train/val。至少需要两个场景，参与训练的每个分支在两侧都要有样本。对于单场景诊断，不应把相邻帧随机切成“独立验证集”。

当前混合器只接受 single-image QA；`delta_control`、带历史的问答、`qa_sft_allowed=false` 的样本被排除。它不会把历史依赖任务静默改成单图问题。`screen_occupancy` 的已有监督合同仍未验证，保持禁止训练。其他历史任务并不会因为能生成 QA 就自动得到 R1 认证。

默认使用 LLaMA-Factory `interleave_over`，两类样本的抽取概率分别为 0.5；一个长轨迹包含多个 assistant turn，因此 **50% 样本不是 50% token/loss 权重**。耗尽较小分支后可能重复采样；训练预算和样本/token 数都要记录。`mask_history: false` 保留轨迹中所有 assistant turn 的监督。

仅轨迹或仅 QA 对照分别使用 `--qa-probability 0` / `1`，并传入同一份 `--split-manifest "$RUN/mix_sft/split_manifest.json"`。保留相同两类输入能保证可比较的场景范围。查看 manifest 的 `qa_label_counts`，不要把单一标签的 bank 当成有辨别力的 QA 训练集。

## 启动 SFT 并保存完整模型

在有 LLaMA-Factory 的环境中执行：

```bash
export SFT_PY=/absolute/path/to/llamafactory/environment/bin/python
PYTHONPATH="$ROOT/third_party/LLaMA-Factory/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$SFT_PY" -m llamafactory.cli train "$RUN/mix_sft/train.yaml"
export SFT_HF="$RUN/mix_sft/model"
```

`train.yaml` 是可审查的起始 recipe：full SFT、冻结 vision tower、单卡 batch 1、梯度累积 8、SDPA。多 GPU 或大模型需要按机器选择 DeepSpeed/FSDP 与 batch；不要把起始配置理解为 7B 全参数单卡必然能放下。应先在训练日志中确认图像 token 预算、最长轨迹是否截断、各数据分支参与训练，随后再扩大预算。

最终目录应包含完整权重、`config.json`、tokenizer 和 processor。若改为 LoRA，先将 adapter 合并到它自己的 base；不能把 adapter-only 目录直接传给下一阶段。下一步见 [Active Spatial RL](active_spatial_pipeline_rl.md)。
