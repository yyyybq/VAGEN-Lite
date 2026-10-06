# GPT-6 Astra：Active Spatial ID / OOD 测评流程

用 OpenAI `gpt-6-astra`（Responses API）跑现有 Active Spatial **ID test** 和五轴 **OOD** 闭环导航评测。协议与 H800 7B unified 矩阵对齐：256×256、success 0.65、strafe、12 turn，便于和 Qwen / Cambrian sweep 对比。

## 前置

- `OPENAI_API_KEY` 已导出（脚本只检查变量是否存在，不会打印密钥）。
- 本机 3DGS 根目录：`/mnt/umm/users/yinbaiqiao/InteriorGS`（可用 `--gs-root` 覆盖）。
- 渲染需要一张 GPU；默认 `gpu_device=0`。
- Python 环境能 `import openai`（当前 conda `vagen-lite` 为 1.99.1，已支持 `responses.create`）。

## 套件

| suite | 含义 | 全量条目（排除 delta_control 后） |
|---|---|---|
| `id_test` | 训练同分布 held-out test | 7169 |
| `ood_v2_centering` | V46/V50 在线验证用的小 OOD | 25 |
| `ood_scene` | 未见过的场景/布局 | 316 |
| `ood_instance` | 已见类别、未见实例 | 200 |
| `ood_category` | 训练中未出现的物体类别 | 338 |
| `ood_template` | 同语义、改写指令 | 355 |
| `ood_geometry` | 几何参数超出训练中心区间 | 335 |
| `smoke` | ID 前 3 条，只用来通管线 | 3 |

全量 ID（7169）对 GPT-6 成本过高。编排脚本按预算做 **按 task_type 分层抽样**，把 slice 写成独立 JSONL，保证 `EvalRunner` 和 `ActiveSpatialEnv.reset(seed)` 扫到同一批题。

## 预算档

| mode | 跑哪些 suite | 规模（约） | 粗估费用（Standard） |
|---|---|---|---|
| `smoke` | smoke × 3 | 3 | 数美元 |
| `canary` | ID 24 + ood_v2 全量 + 其余 OOD 各 8 | ~89 | 数十美元 |
| `standard` | ID 80 + ood_v2 全量 + 其余 OOD 各 40 | ~305 | 数百美元 |
| `full` | ID 200 + 全部 OOD | ~1769 | 很高，先确认额度 |

粗估按每 episode 12 turn、约 $0.4–$1.5。`reasoning_effort=medium` 会偏上限；smoke/调试可改 `low`，或加 `--service-tier flex`。

## 命令

```bash
cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite
export OPENAI_API_KEY=...

# 只写 slice 和 eval_config，不打 API
python scripts/run_gpt6_id_ood_eval.py --mode smoke --dry-run

# 通管线
python scripts/run_gpt6_id_ood_eval.py --mode smoke --run

# 正式小样本 ID + 全部 OOD 轴
python scripts/run_gpt6_id_ood_eval.py --mode canary --run

# 指定轴 / 降低推理强度
python scripts/run_gpt6_id_ood_eval.py \
  --mode canary --suites id_test,ood_scene \
  --reasoning-effort low --run
```

单条 suite 也可以直接走 CLI：

```bash
python evaluation/run_eval.py \
  --config evaluation/configs/eval_gpt6_astra.yaml \
  --jsonl data_gen/active_spatial_pipeline/ood_splits/ood_scene.jsonl
```

`evaluation/configs/eval_gpt6_astra.yaml` 默认是 3 条 smoke。完整矩阵请用编排脚本。

## 输出

```
evaluation/sweeps/active_spatial/gpt6_astra_id_ood_<mode>_<date>/
├── plan.json
├── slices/<suite>.jsonl
├── <suite>/model/eval_config.yaml
├── <suite>/model/results_model.json
├── summary.csv
└── summary.md
```

指标与现有 sweep 相同：`success_rate`、`mean_final_score`、`improvement`、`spl`、按 9 类任务 / 3 大类分解。已有 `results_model.json` 的 suite 会跳过，除非加 `--rerun`。

## 实现要点

- Provider 是 `openai_responses`，走 `client.responses.create`（GPT-6 Astra 官方推荐；Chat Completions 也可用 `provider=openai`）。
- 图像按 Responses 的 `input_image` + PNG data URL 发送；动作格式仍是环境里的 `<think>…</think><action>…</action>`。
- 瞬时 429/5xx 会指数退避重试。
- 环境协议刻意对齐 `test_suites_h800_7b_unified_v1.yaml`，不要和旧的 512px / threshold 0.85 配置混比。
