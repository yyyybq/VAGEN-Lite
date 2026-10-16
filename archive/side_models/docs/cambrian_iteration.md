# Cambrian-S 7B 实验迭代记录

> 创建时间: 2026-06-03  
> 最后更新: 2026-06-03（c3 step49 / c4 step55 / c5 step5 / c6 step69）  
> 范围: c1/c2 诊断实验 → c3~c6 系统消融
> 关联: [v26_v27_analysis.md](v26_v27_analysis.md)（Qwen2.5-VL-3B 实验历史）

---

## 1. 背景与动机

### 1.1 为什么切换到 Cambrian-S 7B

Qwen2.5-VL-3B（v26~v28）系列已验证 PPO on Active Spatial 的基本可行性，峰值 ID_m4=0.607（v26_klhi_lr5e7, step 150）。
切换到 Cambrian-S 7B 的动机：

1. **模型能力**：7B 参数量提供更强的视觉-空间推理能力，理论上应能处理更复杂的几何任务
2. **架构特点**：Cambrian-S 使用多视觉编码器融合（多个 spatial-focused encoder），天然适合空间推理
3. **研究目的**：验证 Active Spatial RL 框架对非 Qwen 架构的泛化能力

### 1.2 已知的 Qwen 3B 训练规律（供对比）

| 现象 | Qwen 3B 行为 |
|------|------------|
| 格式合规 | step 70 时 >95% 响应有 `<action>` 标签 |
| 动作偏好 | move_forward 主导，适度 turn |
| 峰值性能 | ID_m4 = 0.607（v26, step 150） |
| 崩溃模式 | entropy 在 step 150-320 间爆炸（取决于 LR） |

---

## 2. 配置基准（c1 原始设定）

所有 Cambrian 实验共用的固定参数：

```
Model:         /scratch/by2593/hf_cache/cambrian-s-7b
TP_SIZE:       2
GPU_MEM_UTIL:  0.20
N_TRAJECTORY:  4  (GroupAdv)
TRAIN_BATCH:   8
MAX_TURNS:     12
MAX_TRAJ_LEN:  18000
use_remove_padding: False  (必须，NFP 要求)
external_lib:  vagen.models.cambrian_register
ADV_ESTIMATOR: masked_gae
HIGH_LEVEL_GAMMA: 0.95
LAM:           0.95
KL_COEF (gae): 0.001
```

---

## 3. 实验总览

| 实验 | ACTOR_LR | ENTROPY | KL_LOSS | Prompt格式 | CRITIC_WARMUP | 状态 |
|------|----------|---------|---------|-----------|--------------|------|
| c1_groupadv | 1e-6 | 0.005 | 0.20 | free_think | 60 | ✅ 完成 (step ~70 分析) |
| c2_nfp | 1e-6 | 0.005 | 0.20 | free_think | 60 | ✅ 完成 |
| c3_lr5e7 | **5e-7** | 0.005 | 0.20 | free_think | **120** | 🏃 step 49，❌ 无改善 |
| c4_fwdfirst | **5e-7** | 0.005 | 0.20 | **fwd_first** | 60 | 🏃 step 55，✅ **显著改善** |
| c5_entropy_hi | **5e-7** | **0.02** | 0.20 | free_think | 60 | 🏃 step 5，⏳ 过早判断 |
| c6_no_think | **5e-7** | 0.005 | 0.20 | **no_think** | 60 | 🏃 step 69，⚠️ 训练不稳定 |
| **c7_fwdfirst_ehi** | **5e-7** | **0.01** | 0.20 | **fwd_first** | 60 | 📝 脚本已生成，待启动 |

---

## 4. c1 诊断实验（基线）

### 4.1 参数

```
EXPERIMENT_NAME: c1_groupadv_100scenes
ACTOR_LR:        1e-6
ENTROPY_COEFF:   0.005
KL_LOSS_COEF:    0.20
CRITIC_WARMUP:   60
ENV_CONFIG:      env_config_v24_100scenes.yaml (prompt_format: free_think)
TOTAL_STEPS:     2000
```

### 4.2 Rollout 行为分析（step 70）

分析 c1 step 70 的 344 条响应（4 rollout × 86 prompts）：

**格式合规情况：**

| 类型 | 数量 | 占比 |
|------|------|------|
| 有 `<action>` 标签（格式合规） | 287 | 83.4% |
| 无 `<action>` 标签（纯英文散文） | 57 | **16.6%** ← 严重问题 |
| action 名称拼写错误 | ~20 | **~5.8%** |

拼写错误示例：`move Forward`（空格+大写）、`move.forward`（点号）、`strafe_right`（错误名称）、`turn.right`、`look`、`rotate`

**动作分布（有效动作，已正确解析 `<action>...</action>` 后）：**

| 动作 | 次数 | 占比 |
|------|------|------|
| turn_left | 247 | **43.2%** ← 严重偏斜 |
| move_forward | 140 | 24.5% |
| turn_right | 98 | 17.1% |
| move_backward | 16 | 2.8% |
| move_left | 15 | 2.6% |
| move_right | 7 | 1.2% |

**与 Qwen 3B (v28 step 70) 的对比：**

| 指标 | Qwen v28 step 70 | Cambrian c1 step 70 |
|------|-----------------|---------------------|
| 无 action 标签率 | <5% | **16.6%** |
| turn_left 占比 | ~25% | **43.2%** |
| move_forward 占比 | ~40% | **24.5%** |

### 4.3 根本原因分析

**turn_left 43.2% 的成因推断：**

1. **Hint 4**（"Look around if you're unsure"）→ Cambrian 将不确定时的旋转作为默认策略
2. **Hint 5**（"FIRST rotate...THEN translate"）→ "先转后走"的强烈引导，Cambrian 缺乏 Qwen 的 move_forward 先验，因此过度执行
3. **Cambrian 没有导航任务的 SFT 先验** → 默认行为是"看清楚再行动"，即反复旋转

**16.6% 无 action 标签的成因推断：**

1. think 过程过长，模型在生成过程中"迷失"，直接在 think 中写了动作描述
2. 或 Cambrian 将 think 与 action 格式混淆，生成了只有 think 标签的响应

### 4.4 验证指标

| step | ID_b4 | ID_m4 | ID_w4 | OOD_b4 | OOD_m4 | OOD_w4 |
|------|-------|-------|-------|--------|--------|--------|
| *待记录* | | | | | | |

---

## 5. c2 实验

### 5.1 参数（相对 c1 的差异）

> c2 与 c1 在脚本层面参数完全相同（两者都有 `use_remove_padding=False`）。
> c2 最初命名为 NFP（No Remove Padding），但实际上 c1 也已经是 NFP。
> 即 c1 = c2，两者为重复实验，用于确认 c1 的可复现性。

### 5.2 结论

待补充。

---

## 6. c3 实验（LR 诊断）

### 6.1 参数（相对 c1 的差异）

```
★ ACTOR_LR:      1e-6 → 5e-7
★ CRITIC_WARMUP: 60   → 120
ENV_CONFIG:      env_config_v24_100scenes.yaml (prompt_format: free_think, 不变)
```

### 6.2 科学问题

LR=1e-6 是否导致 Cambrian 策略网络震荡，使其无法稳定学习导航策略？
（注：在 Qwen 实验中，LR=1e-6 vs 5e-7 对 entropy 稳定性有显著影响）

### 6.3 预期结果

- **若 c3 改善**：step 100 时 turn_left 占比 < 35%，t_succ > 2% → LR 是根因，5e-7 也是 Cambrian 的正确 LR
- **若 c3 无改善**：turn_left 仍 >40% → LR 不是根因，需要 prompt 或架构层面的修复

### 6.4 训练指标（step 1–49）

| step | entropy | t_succ% | score | resp_len |
|------|---------|---------|-------|----------|
| 1 | 0.886 | 0.82% | 0.397 | 139.5 |
| 10 | 0.882 | 0.28% | 0.112 | 139.3 |
| 25 | 0.879 | 0.54% | 0.259 | 140.1 |
| 49 | 0.876 | 0.80% | 0.389 | 140.7 |

**动作分布（rollout）**

| step | move_forward | turn_left | turn_right | no_tag% |
|------|-------------|-----------|------------|---------|
| 1 | 25.0% | 41.1% | 26.7% | 13.3% |
| 25 | 22.6% | 36.6% | 30.1% | 14.8% |
| 49 | 21.4% | **42.7%** | 27.9% | **19.9%** |

### 6.5 验证指标

尚未到达 step 50（TEST_FREQ=50），暂无验证数据。

### 6.6 实验结论

**c3 无改善，LR 不是根因。**

- turn_left 始终维持在 38–43%，move_forward 维持在 21–25%，与 c1 基线完全一致
- t_succ 在 0.3–1.1% 之间随机震荡，**无上升趋势**
- entropy 稳定在 ~0.877，未出现坍缩或爆炸
- no_tag 率有略微上升趋势（13% → 20%），格式合规没有改善
- **结论**：将 LR 从 1e-6 降至 5e-7、critic_warmup 延长至 120，均无法改变 Cambrian 的 turn_left 先验行为。LR 不是导致行为偏斜的根本原因。

---

## 7. c4 实验（forward-first 提示策略）

### 7.1 参数（相对 c1 的差异）

```
★ ENV_CONFIG:    env_config_v24_100scenes_fwdfirst.yaml
                 prompt_format: free_think → free_think_fwd_first
★ ACTOR_LR:      1e-6 → 5e-7
  VAL_BEFORE_TRAIN: True  (记录 fwd_first 提示下的初始性能)
```

### 7.2 free_think_fwd_first 提示改动

原 Hint 4/5 → 新 Hint 4/5：

**原（free_think）：**
```
4. Look around if you're unsure of the target location
5. Strategy: FIRST rotate (turn_left/turn_right) to face the target along the
   requested view direction, THEN translate (move_forward/strafe) to approach.
```

**新（free_think_fwd_first）：**
```
4. Strategy: DEFAULT to move_forward to close the distance. Only use 
   turn_left/turn_right when you need to change facing direction. After at 
   most 2 consecutive turns, always attempt move_forward.
5. AVOID spinning in circles: if the reward is not improving after repeated 
   turns, switch to move_forward or move_left/move_right to explore a new position.
```

新增：ACTION NAMES 精确拼写提示（针对 5.8% 拼写错误）

### 7.3 科学问题

turn_left 43.2% 的偏好是否主要由提示策略（"先转后走"）引起？

### 7.4 预期结果

- **若 c4 改善**：move_forward 占比 >35%，turn_left < 30% → 提示修复有效
- **若 c4 无改善**：turn_left 仍主导 → Cambrian 的旋转偏好是内在先验（SFT 冷启动是唯一出路）

### 7.5 训练指标（step 1–55）

| step | entropy | t_succ% | score | returns | resp_len |
|------|---------|---------|-------|---------|----------|
| 1 | 0.884 | 6.00% | 3.013 | 16.31 | 138.7 |
| 10 | 0.875 | 2.22% | 1.096 | 6.90 | 132.8 |
| 25 | 0.895 | 3.78% | 1.893 | 11.77 | 139.4 |
| 50 | 0.881 | 4.69% | 2.359 | 13.83 | 132.9 |
| 55 | 0.888 | 4.17% | 2.098 | 12.87 | 129.2 |

**动作分布（rollout，完全逆转）**

| step | move_forward | turn_left | turn_right | no_tag% |
|------|-------------|-----------|------------|---------|
| 1 | **57.4%** | 21.8% | 15.2% | 14.3% |
| 28 | **55.9%** | 17.4% | 18.2% | 18.0% |
| 55 | **53.0%** | 23.1% | 18.4% | 13.1% |

### 7.6 验证指标

| step | ID_b4 | ID_m4 | ID_w4 | OOD_b4 | OOD_m4 | OOD_w4 |
|------|-------|-------|-------|--------|--------|--------|
| 0（训练前基线）| 0.6667 | 0.4286 | 0.0952 | 0.7368 | 0.3158 | 0.0000 |
| **50** | **0.8095** | **0.4048** | **0.0952** | **0.7895** | **0.3816** | **0.0526** |

### 7.7 实验结论

**c4 效果显著，提示策略是 turn_left 偏好的主要根因。✅**

- move_forward 从基线 24.5% → **53–57%**（提升 2.2×），turn_left 从 43.2% → **18–23%**（降低约一半）
- 训练 t_succ 稳定在 2–5%，是 c3 的 3–5 倍
- Val step 50：OOD_m4 从 0.316 → **0.382**（+6.6pp，泛化改善）；ID_b4 从 0.667 → **0.810**（+14pp）
- 注：step 0 基线 ID_m4=0.429 已很高，说明 fwd-first 提示本身（不依赖 RL 训练）对模型行为有显著引导效果
- **结论**：原 Hint 4/5（"先转后走"/"Look around"）是 Cambrian turn_left 偏好的直接诱因。fwd-first 提示策略是目前最有效的干预手段。

---

## 8. c5 实验（高熵正则，直接对抗 turn_left 坍缩）

### 8.1 参数（相对 c1 的差异）

```
★ ENTROPY_COEFF: 0.005 → 0.02  (4x 增加)
  ACTOR_LR:      5e-7  (与 c3/c4 一致)
ENV_CONFIG:      env_config_v24_100scenes.yaml (prompt_format: free_think, 不变)
```

### 8.2 科学问题

turn_left 43.2% 是否是 PPO entropy 坍缩的表现？是否可以通过增大 entropy 正则直接修复？

### 8.3 预期结果

- **若 c5 改善**：action distribution 更均匀，t_succ 上升 → 熵坍缩是根因
- **若 c5 无改善**：entropy 过高导致训练不稳定，或 turn_left 仍主导 → Cambrian 先验过强，无法用正则修复

### 8.4 训练指标（step 1–5，过早）

| step | entropy | t_succ% | score | 每步耗时(s) |
|------|---------|---------|-------|------------|
| 1 | 0.892 | 0.27% | 0.116 | ~1071 |
| 3 | 0.871 | 0.00% | -0.039 | ~1072 |
| 5 | 0.871 | 1.18% | 0.583 | ~1032 |

**动作分布（rollout，步骤 1–5）**

| step | move_forward | turn_left | turn_right | no_tag% |
|------|-------------|-----------|------------|---------|
| 1 | 22.3% | 38.8% | 29.8% | 18.1% |
| 3 | 22.0% | 40.7% | 26.5% | 14.9% |
| 5 | 19.3% | **44.5%** | 27.8% | 18.2% |

### 8.5 验证指标

尚未到达 step 50（TEST_FREQ=50），暂无验证数据。

### 8.6 中间状态评估（step 5，⏳ 过早判断）

- 目前仅运行 5 步，每步耗时约 **1000 秒**（约为 c3/c4 的 2–3 倍），推测受其他任务 GPU 竞争影响
- 早期动作分布与 c1/c3 完全相同，turn_left 仍在 39–45%，暂无改善迹象
- t_succ 步骤 1–4 均为 0%，step 5 才出现 1.2%，但单点无意义
- **需等待 step 50 才能得出结论**

---

## 9. c6 实验（no_think 格式，消除格式合规障碍）

### 9.1 参数（相对 c1 的差异）

```
★ ENV_CONFIG:         env_config_v24_100scenes_no_think.yaml
                      prompt_format: free_think → no_think
★ MAX_RESPONSE_LENGTH: 384 → 128  (no_think 响应极短)
  ACTOR_LR:           5e-7
```

### 9.2 no_think 格式

```
# free_think（c1~c5）要求：
<think>推理过程</think><action>action1|action2|</action>

# no_think（c6）只需要：
<action>action1|action2|</action>
```

### 9.3 科学问题

1. 16.6% 的无 action 标签失败是否由 think 格式的认知负担引起？
2. 简化格式后，格式合规率是否 >95%？
3. no_think 下 turn_left 偏好是否减轻？（若 think 推理在强化旋转策略）

### 9.4 预期结果

- **若 c6 格式合规改善**：合规率 >95%，reward 曲线更快上升 → think 格式是障碍
- **若 c6 格式合规无改善**：Cambrian 对 `<action>` 格式本身也不理解 → 需要 SFT
- **额外发现**：MAX_RESPONSE_LENGTH=128 可大幅提升训练吞吐量（响应从 ~250 tokens 降至 ~20 tokens）

### 9.5 训练指标（step 1–69，含异常段）

| step | entropy | t_succ% | score | resp_len | 备注 |
|------|---------|---------|-------|----------|------|
| 1 | 0.748 | 1.92% | 0.925 | 47.9 | 正常 |
| 10 | 0.765 | 2.63% | 1.269 | 54.3 | 正常 |
| 30 | 0.765 | 3.21% | 1.558 | 48.4 | 正常 |
| 60 | 0.713 | 3.90% | 1.918 | 44.4 | 正常峰值 |
| **61** | **0.572↓** | **6.58%↑** | **3.339** | **17.7↓** | ⚠️ **异常：resp_len 骤降** |
| 62–63 | 0.58–0.61 | 1.5–4.1% | 0.70–2.05 | 16–18 | resp_len 仍异常短 |
| 64 | 0.803 | 1.79% | 0.855 | 31.3 | 恢复中 |
| **65–69** | **0.916→1.16↑** | 0.4–1.3% | 0.1–0.6 | 38–56 | ⚠️ **entropy 爆炸** |

**step 61 异常分析**：resp_len 从 ~48 token 骤降至 17 token，推测模型开始大量输出 `<action>done</action>` 等极短响应，导致 t_succ 虚高（6.6%）。随后 PPO 过度更新，触发 entropy 爆炸（step 66 达 1.147）。

**动作分布演变**

| step | move_forward | turn_left | move_left | no_tag% |
|------|-------------|-----------|-----------|---------|
| 1 | 31.2% | 36.2% | 10.5% | 26.3% |
| 34 | 31.6% | 34.0% | 8.1% | 24.0% |
| **67** | 20.8% | 25.9% | **31.1%** | **29.1%** |

注：step 67 后 move_left 异常主导（31.1%），no_tag 率上升至 29%，行为退化迹象明显。

### 9.6 验证指标

| step | ID_b4 | ID_m4 | ID_w4 | OOD_b4 | OOD_m4 | OOD_w4 |
|------|-------|-------|-------|--------|--------|--------|
| **50** | **0.5714** | **0.2976** | **0.0952** | **0.3158** | **0.1447** | **0.0526** |

与 c4 step 50 对比：

| 实验 | ID_m4 | OOD_m4 |
|------|-------|--------|
| **c4_fwdfirst** | **0.4048** | **0.3816** |
| c6_no_think | 0.2976（−26%） | 0.1447（−62%） |

### 9.7 实验结论

**c6 训练不稳定，no_think 格式未能改善格式合规问题，验证性能显著低于 c4。⚠️**

- no_tag 率始终维持在 **26–29%**（比 c1 的 16.6% 更差），说明 Cambrian 对 `<action>` 格式本身的理解不足，think 格式并非主要障碍
- 验证 OOD_m4=0.145，仅为 c4 的 38%，说明 no_think 格式丢失了重要推理能力
- step 61 触发异常 → entropy 爆炸，step 66 后训练实质失控（entropy=1.15，score<0.6）
- **结论**：no_think 格式不适合 Cambrian。think 推理并非格式合规障碍的根因；Cambrian 需要更直接的格式引导（如 SFT 冷启动）而非去除推理。

---

## 10. c7 实验（fwd-first + 适度 entropy）

### 10.1 参数（相对 c4 的差异）

```
★ ENTROPY_COEFF: 0.005 → 0.01  (2x 适度提升，目标降低 no_tag 率)
★ SAVE_FREQ:     80    → 40    (更密集 checkpoint，追踪 step80/120/160 峰值)
  ENV_CONFIG:    env_config_v24_100scenes_fwdfirst.yaml (fwd-first，与 c4 相同)
  ACTOR_LR:      5e-7  (与 c4 相同)
  VAL_BEFORE_TRAIN: True (记录初始基线)
```

### 10.2 科学问题

1. 在 fwd-first 提示已修复动作偏好的前提下，适度 entropy 正则是否能进一步降低 no_tag 率（c4 残留的 13–18%）？
2. entropy=0.01 是否比 c4（0.005）性能更好，但比 c5（0.02）更稳定？
3. Cambrian + fwd-first 能否在 step 150–200 时超越 Qwen3B 的峰值（ID_m4=0.607）？

### 10.3 预期结果

- **若 c7 改善**：no_tag 率 < 12%，step 150 时 ID_m4 > 0.45 → entropy 与 fwd-first 协同有效
- **若 c7 无改善**：no_tag 率不变，性能持平或低于 c4 → entropy 正则对格式合规无帮助

### 10.4 可选续训方案

c4 在 step 80 保存首个 checkpoint 后，c7 可改为从该 checkpoint 续训：

```bash
RESUME_MODE="resume_path"
trainer.resume_from_path=.../c4_fwdfirst/checkpoints/global_step_80
```

### 10.5 启动条件

建议在 c6 停止（entropy 已爆炸，step 66+ 无改善）后使用空出的 GPU 槽位启动 c7。

### 10.6 验证指标

| step | ID_b4 | ID_m4 | ID_w4 | OOD_b4 | OOD_m4 | OOD_w4 |
|------|-------|-------|-------|--------|--------|--------|
| 0 | | | | | | |
| 50 | | | | | | |
| 100 | | | | | | |

---

## 11. 实验矩阵与假说覆盖

```
                 ┌─────────────────────────────────────────────────────┐
                 │           Cambrian 失败的 4 个可能根因               │
                 │                                                     │
                 │  A. LR 过大 (震荡)     → c3 (LR 5e-7)              │
                 │  B. 提示策略错误        → c4 (fwd-first 提示)       │
                 │  C. Entropy 坍缩       → c5 (entropy 0.005→0.02)   │
                 │  D. 格式合规障碍        → c6 (no_think)             │
                 └─────────────────────────────────────────────────────┘
```

**决策树（已更新，基于 c3 step49 / c4 step55 / c6 step69 实际结果）：**

```
✅ c4 改善（move_forward 55%，val OOD_m4 0.382）
❌ c3 无改善（turn_left 仍 40%，t_succ <1%）
⏳ c5 过早（仅 5步，暂无结论）
⚠️ c6 不稳定（entropy 爆炸，val OOD_m4 仅 0.145）

→ 当前最佳方向：c4（fwd-first 提示）已验证有效
→ 下一步：在 c4 基础上提升，运行 c7（c4 + 更长训练 / entropy 组合）
→ c5 等待 step 50 结果后决定是否继续
→ c6 建议监控 entropy，若 step 80+ 仍 >1.0 则考虑早停
```

---

## 11. 后续实验候选

> 基于当前结果（2026-06-03）更新优先级

| 候选实验 | 优先级 | 前提条件 | 内容 |
|---------|--------|---------|------|
| **c7: c4 延伸** | ⭐⭐⭐ 最高 | c4 已验证有效 | fwd-first 提示 + 训练至 step 150–200，追踪峰值性能 |
| **c8: c4+entropy** | ⭐⭐ 次高 | c5 完成后评估 | fwd-first + entropy=0.02，尝试同时改善多样性 |
| c9: c4+更低LR | ⭐ 待定 | c4 step 100+ 出现不稳定 | fwd-first + LR=2e-7，应对潜在 entropy 爆炸 |
| SFT-cambrian | ⭐ 保底 | 若 c4 最终无法超越 Qwen3B | 先 SFT 建立导航格式先验，再 RL 微调 |
| Qwen2.5-VL-7B | ⭐ 对照 | 有算力余量时 | 确认 c4 改善是 fwd-first 提示泛化性还是 Cambrian 特有 |

---

## 12. 文件索引

| 类型 | 路径 |
|------|------|
| 实验脚本 | `examples/train/active_spatial/experiments/c*.sh` |
| 标准 env config | `examples/train/active_spatial/env_config_v24_100scenes.yaml` |
| fwd-first env config | `examples/train/active_spatial/env_config_v24_100scenes_fwdfirst.yaml` |
| no_think env config | `examples/train/active_spatial/env_config_v24_100scenes_no_think.yaml` |
| 提示定义 | `vagen/envs/active_spatial/prompt.py` |
| Cambrian 注册 | `vagen/models/cambrian_register.py` |
