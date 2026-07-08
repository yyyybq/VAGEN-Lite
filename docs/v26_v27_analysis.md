# v26 / v27 / v28 实验进度分析与 c1/c2 Cambrian 实验

> 生成时间: 2026-05-31 / 更新: 2026-06-16  
> 范围: v25 基线 → v26 LR 消融 → v27 权重恢复实验 → v28 cosine LR 实验（完整分析）→ c1/c2 Cambrian 实验

---

## 1. 背景与问题定义

### 1.1 历史进展简述 (v19→v25)

从 v19 到 v25 的核心迭代轨迹：

| 版本 | 关键改动 | 峰值 ID_m4 | 主要问题 |
|------|---------|-----------|---------|
| v19_thr70_andgate | AND-gate + thr=0.70 | 0.859 | 后期严重崩塌 (step100后→0) |
| v20_winner | KL↑+CRITIC_WARMUP↑+CLIPRANGE↓+entropy↑ | ~0.85 | 首版稳定训练 |
| v24_groupadv | 100 scenes + GroupAdv (N=4) | ~0.60 | 过渡版本 |
| **v25_klhi** | KL=0.20 + entropy=0.005 (基准稳定配置) | **0.500** (@step150) | 峰值后持续缓降到0.238(@step300) |
| v25_cosine | LR cosine decay, 但 warmup 设计缺陷 | — | warmup bug导致无效 |

**v25_klhi 核心发现**:
- KL=0.20 + entropy=0.005 成功阻止了崩塌，是当前稳定基线
- 但 LR=1e-6 恒定导致 step 150 后 mean@4 持续缓降
- 根本假设：**1e-6 LR 对于 step 100+ 偏大**，导致过度优化

### 1.2 v26/v27 的设计动机

基于 v25 分析，三个消融实验同时启动：

| 实验 | 核心改动 | 科学问题 |
|------|---------|---------|
| **v26_klhi_lr5e7** | LR 1e-6 → 5e-7 (恒定) | 减半 LR 是否能延迟/改善峰值退化？ |
| **v26_klhi_lrdecay** | LR cosine 1e-6→1e-7 (无warmup) | 修复 v25_cosine 的 warmup bug 后，decay 是否有效？ |
| **v27_from_v25ckpt** | 从 v25 step240 权重重启，LR=1e-7 | 已退化权重能否通过极低 LR 的 RL 继续恢复？ |

---

## 2. 验证指标曲线

### 2.1 v26_klhi_lr5e7 (LR=5e-7 恒定, 从头训练)

| step | ID_b4 | ID_m4 | ID_w4 | OOD_b4 | OOD_m4 | OOD_w4 |
|------|-------|-------|-------|--------|--------|--------|
| 50   | 0.7143 | 0.4167 | 0.1429 | 0.8421 | 0.4079 | 0.0000 |
| 100  | 0.8095 | 0.4762 | 0.0952 | 0.7895 | 0.4342 | 0.1053 |
| 150  | **0.8571** | **0.6071** | **0.3333** | **0.8421** | **0.5000** | **0.2105** |

→ **三个验证点单调递增**，step 150 的 ID_m4=0.607 **已超过 v25_klhi 的历史峰值 (0.500)**。

### 2.2 v26_klhi_lrdecay (cosine 1e-6→1e-7, 从头训练)

| step | ID_b4 | ID_m4 | ID_w4 | OOD_b4 | OOD_m4 | OOD_w4 |
|------|-------|-------|-------|--------|--------|--------|
| 50   | 0.7143 | 0.4405 | 0.1429 | **0.9474** | 0.4868 | 0.1579 |
| 100  | **0.9048** | **0.5595** | **0.2381** | 0.6316 | 0.3947 | 0.1579 |
| 150  | 0.8095 | 0.5119 | 0.1429 | 0.7368 | 0.4474 | 0.1579 |

→ ID 在 step100 达峰后回落，OOD 在 step50 暴高（actor尚未更新时的噪声）后崩至 0.632。

### 2.3 v27_from_v25ckpt (v25 step240 权重重启, LR=1e-7)

| step | ID_b4 | ID_m4 | OOD_b4 | OOD_m4 |
|------|-------|-------|--------|--------|
| 0 (初始) | 0.7143 | 0.3810 | 0.7368 | 0.3158 |
| 50     | 0.7143 | 0.3810 | 0.6316 | 0.3026 |

→ 50步后指标没有改善（步骤0-59为critic预热期，actor LR=0）。

### 2.4 v28_klhi_lr5e7_cosine (LR cosine 5e-7→5e-8, 从头训练)

| step | ID_b4 | ID_m4 | ID_w4 | OOD_b4 | OOD_m4 | OOD_w4 |
|------|-------|-------|-------|--------|--------|--------|
| 0    | 0.7143 | 0.4167 | 0.0952 | 0.8947 | 0.4079 | 0.1053 |
| 50   | 0.7619 | 0.4524 | 0.0000 | 0.6842 | 0.3553 | 0.1053 |
| 100  | 0.7143 | 0.5238 | 0.2381 | 0.8421 | 0.4342 | 0.2105 |
| 150  | 0.8571 | 0.5119 | 0.1905 | 0.7895 | 0.4737 | 0.1579 |
| **200** | **0.9048** | **0.5595** | 0.0476 | **0.8421** | 0.4868 | 0.1053 |
| 250  | 0.8571 | 0.5595 | 0.2857 | 0.7368 | 0.4605 | 0.2105 |
| 300  | 0.8571 | 0.5357 | 0.0952 | 0.7368 | 0.3289 | 0.0526 |
| 350  | 0.7143 | 0.3929 | 0.0476 | 0.7895 | 0.3289 | 0.0000 |

→ **step 200 达到峰值 ID_b4=0.9048, ID_m4=0.5595**。ID_b4 为所有实验历史最高，但 ID_m4=0.5595 **未超过 v26 的 0.607**。step 200 后开始退化，step 350 时 ID_m4 已降至 0.393，退化轨迹与 v26 晚期相似。

---

## 3. 训练动态分析

### 3.1 v26_klhi_lr5e7 训练指标

| step | entropy | KL loss | resp_clip% | act_entropy |
|------|---------|---------|-----------|-------------|
| 60   | 0.513 | 0.00020 | 19.9% | 0.999 |
| 100  | 0.572 | 0.00113 | 25.6% | 0.827 |
| 150  | 0.762 | 0.00198 | **6.8%** | 0.954 |
| 184  | 0.720 | 0.00239 | 13.1% | 0.607 |

**关键观察**:
- resp_clip% 在 step150 降至 6.8%（学会了短轨迹，不再撞上最大步数）
- policy entropy 缓慢上升（0.51→0.76），KL 保持 <0.003，**完全健康**
- 目前仍在持续改进，**尚未到达峰值**

### 3.2 v26_klhi_lrdecay 训练指标（关键转折点）

| step | entropy | KL loss | pg_clipfrac | LR |
|------|---------|---------|------------|-----|
| 100  | 0.596 | 0.00295 | 0.01392 | 9.99e-7 |
| 130  | 0.831 | 0.00436 | 0.01785 | 9.97e-7 |
| **160** | **1.028** | **0.01047** | **0.02159** | 9.94e-7 |
| 181  | 0.912 | 0.00785 | 0.01496 | 9.92e-7 |

**根本问题**：step181 时 LR 仅从 1e-6 降至 9.92e-7（**衰减不到 1%**），以 1000 步为周期的余弦衰减在前 200 步形同虚设。entropy 在 step160 突破 1.0，KL 急升至 0.01（lr5e7 同期的 5倍）——是 v25_klhi 不稳定模式的完全复现。

### 3.3 v27_from_v25ckpt 训练指标（熵爆炸过程）

| step | entropy | KL | t_succ% |
|------|---------|-----|---------|
| 1-59 | 1.06~1.28 | 0 | ~4% | ← critic预热，actor LR=0 |
| **60** | 1.107 | 0.00017 | 2.97% | ← **actor LR=1e-7 激活** |
| 67   | 2.001 | 0.00141 | 1.51% | 超过2.0 |
| **80** | **2.619** | **0.00280** | 3.17% | **熵爆炸** |

对比：v26两版在step80时 entropy≈0.60，v27已达**2.62（差4倍以上）**。

### 3.4 v28_klhi_lr5e7_cosine 训练指标

| step | entropy | t_succ% | resp_clip% | LR |
|------|---------|---------|-----------|-----|
| 160  | 0.685 | 3.05 | 16.9% | 4.89e-7 |
| 180  | 0.700 | 2.82 | 19.6% | 4.84e-7 |
| 200  | 0.700 | **6.84** | 11.8% | 4.79e-7 |
| 240  | 0.842 | 3.57 | 14.3% | 4.65e-7 |
| 280  | 0.960 | 4.39 | 18.4% | 4.48e-7 |
| 320  | **1.015** | 2.66 | 14.8% | 4.29e-7 |
| 360  | **1.094** | 5.83 | 17.2% | **4.07e-7** |

**关键观察**:
- step 200 时 entropy=0.700（健康），t_succ=6.84%（局部最高），与 val 峰值吻合
- step 320+ entropy 突破 1.0，尽管 LR 已衰减至 4.3e-7，仍出现轻微熵上升
- step 360 时 LR 降至 4.07e-7（起始 5e-7 的 81.4%），cosine 衰减减缓了但未完全抑制后期熵爆炸
- **LR 衰减将 v26 的 step~180 熵爆炸推迟到了 step~320，延迟约 140 步**

---

## 4. 行为多样性监控

新增 `--diversity` 标志到 `scripts/analyze_experiments.py`，解析 `rollout_data/` 目录。

### 4.1 指标说明

| 指标 | 含义 |
|------|------|
| **act_H** | 全轨迹动作分布的香农熵（越低 = 动作越集中，通常移动为主） |
| **1st_H** | 首步动作熵（越低 = 第一步动作越刻板） |
| **uniq3** | 唯一3步前缀数（越低 = 轨迹开头越雷同） |
| **dom%** | 最常用动作的占比（通常为 move_forward） |

### 4.2 三版实验行为多样性对比

**v26_klhi_lr5e7** (act_H 趋势):

| step | act_H | dom% | 解读 |
|------|-------|------|-----|
| 1    | 0.776 | 72%  | 初始状态，较均衡 |
| 60   | **0.999** | 54%  | 峰值多样性（actor LR刚激活，探索最充分）|
| 100  | 0.827 | 68%  | 开始专注 |
| 150  | 0.954 | 63%  | val最佳时仍保持较好多样性 |
| 180  | 0.461 | **87%** | move_forward 集中化，待观察 |

**v26_klhi_lrdecay** (act_H vs policy entropy 对比):

| step | act_H | policy_entropy | 解读 |
|------|-------|----------------|-----|
| 60   | 0.978 | 0.573 | 两者均健康 |
| 100  | 0.687 | 0.596 | 动作集中但policy尚可 |
| 160  | 0.666 | **1.028** | policy爆炸，但动作仍在集中化 |
| 181  | 0.587 | 0.912 | 动作继续集中，不可逆 |

→ policy entropy 爆炸（随机化）与 act entropy 下降（集中化）**同时发生** → 模型在 token 级别变随机，但动作模式却更单一，是典型的不稳定相变。

**v27_from_v25ckpt** (初始状态揭示退化来源):

| step | act_H | dom% | 解读 |
|------|-------|------|-----|
| 1    | 0.824 | 70%  | 已高于v26初始(66-68%)，v25退化的遗留 |
| 40   | 0.584 | 82%  | 快速集中化（critic预热期内也在劣化）|
| 80   | 0.722 | 74%  | entropy爆炸导致些许回升，但仍然集中 |

→ v25 step240 的行为模式**已经高度集中**（move_forward 占 70%），以此为起点的 RL 无法恢复，只会加速劣化。

**v28_klhi_lr5e7_cosine** (act_H 与 v26 对比):

| step | v28 act_H | v26 act_H | v28 move_fwd% | v26 move_fwd% | 解读 |
|------|-----------|-----------|--------------|--------------|------|
| 100  | 0.936 | 0.827 | 61% | 68% | v28 更分散 |
| 120  | 0.813 | 0.737 | 70% | 74% | v28 略优 |
| 140  | 0.702 | 0.858 | 76% | 67% | 短暂反转（v26 已开始稳定化）|
| 160  | **0.921** | 0.677 | 62% | **78%** | v28 大幅反超（cosine LR 效果显现）|
| 220  | 0.539 | — | **86%** | — | act_H 降至警戒线 |
| 300  | 0.685 | — | 78% | — | 集中化维持中 |
| 360  | 0.935 | — | 62% | — | entropy爆炸后多样性异常回升（随机化信号）|

→ v28 在 step 100-160 的 act_H 普遍优于 v26 同期，**cosine LR 确实延缓了 action concentration**。但 step 220 时 act_H 降至 0.539（move_forward 86%），随后 entropy 突破 1.0，最终出现与 v26 相同的 action concentration → entropy 爆炸模式，时序仅延后约 140 步。

---

## 5. 综合结论

### 5.1 已验证的假设

| 假设 | 实验证据 | 结论 |
|------|---------|------|
| LR=5e-7 比 1e-6 更稳定 | v26_lr5e7 step150 ID_m4=0.607 > v25峰值0.500 | ✅ 强烈成立 |
| 余弦衰减（从1e-6）修复v25退化 | lrdecay前200步LR几乎没变，熵爆炸复现 | ❌ 时机太晚，无效 |
| 从退化ckpt低LR恢复 | v27熵在20步内从1.1→2.6，策略急速随机化 | ❌ 反而加速崩溃 |
| RL 持续训练导致输出坍塌 | 熵爆炸方向（策略随机化），而非模式坍塌 | ✅ 确认，方向是熵增 |
| cosine LR (5e-7→5e-8) 能超越 v26 峰值 | v28 峰值 ID_m4=0.5595 < v26 的 0.607 | ❌ **未超越**，ID_b4=0.9048 历史最高但 m4 未达标 |
| cosine LR 能延缓 action concentration | v28 step 160 act_H=0.921 vs v26 同期 0.677 | ✅ **延迟约 140 步**，但最终仍熵爆炸 |

### 5.2 关键机制洞察

1. **过度优化的方向是熵爆炸，非模式坍塌**  
   - 模型并非学会输出单一动作序列（模式坍塌）
   - 而是策略分布随机化（policy entropy超过1.0），导致有效导航信号丢失
   - 同时伴随 pg_clipfrac 飙升和 KL 失控

2. **LR 是稳定性的最核心因子**  
   - 1e-6 在 step ~100 开始不稳定；5e-7 在 step 150+ 仍健康
   - 学习率减半将"失稳步数"从 100 推迟到至少 200（待确认）

3. **行为多样性 act_H 是先导指标**  
   - act_H 低于 0.5 时（move_forward > 80%），后续验证指标有下降风险
   - v26_lr5e7 在 step 180 时 act_H=0.461，需持续监控

4. **从退化ckpt出发时，KL reference 失效**  
   - v27 的 reference model = 同一退化权重，KL 从 0 开始，无法提供稳定锚点
   - 退化恢复需要 SFT 重新锚定，不能靠 RL 自救

---

## 6. 工具更新

### 6.1 `scripts/analyze_experiments.py` — 新增行为多样性分析

```bash
# 行为多样性
python3 scripts/analyze_experiments.py --exps v26_klhi_lr5e7 --diversity

# 结合验证和多样性
python3 scripts/analyze_experiments.py \
    --exps v26_klhi_lr5e7,v26_klhi_lrdecay,v27_from_v25ckpt \
    --val --diversity
```

输出指标：act_H (动作熵), 1st_H (首步动作熵), uniq3 (唯一3步前缀), dominant action 占比

### 6.2 `scripts/probe_spatial_qa.py` — 离线 backbone 能力探测

**用途**: 测试已有 checkpoint 的空间推理 QA 能力是否退化（**不影响训练**，独立在空闲 GPU 上运行）。

**使用方法**:
```bash
# 测试 v25 step240 vs v26 step160 vs base model
python3 scripts/probe_spatial_qa.py \
    --ckpts "base:Qwen/Qwen2.5-VL-3B-Instruct,v25_step240:exps/vagen_active_spatial/v25_groupadv_100scenes_klhi/checkpoints/global_step_240/actor/huggingface,v26_step160:exps/vagen_active_spatial/v26_klhi_lr5e7/checkpoints/global_step_160/actor/huggingface" \
    --n_samples 200 --gpu 4

# 快速测试单个 checkpoint
python3 scripts/probe_spatial_qa.py \
    --ckpts "v26_step160:exps/vagen_active_spatial/v26_klhi_lr5e7/checkpoints/global_step_160/actor/huggingface" \
    --n_samples 200 --gpu 4 --include_base
```

**评测内容**:
- MindCube tinybench 200 samples（多选题，空间推理）
- 5 条文字 sanity-check（检测通用能力退化：方向、逻辑、数学）
- 对比基础模型 Qwen2.5-VL-3B-Instruct 的退化幅度
- 按 type 细分 (1_frame/2_frame/3_frame/three_view/general)

**不影响训练**: 直接从 `actor/huggingface/` 加载，standalone 运行，约 10-15 分钟。

**新增**: `--show_examples N` 参数可打印前 N 条完整模型回答，用于观察输出格式：
```bash
python3 scripts/probe_spatial_qa.py \
    --ckpts "v26_step160:exps/vagen_active_spatial/v26_klhi_lr5e7/checkpoints/global_step_160/actor/huggingface" \
    --n_samples 200 --gpu 0 --include_base --show_examples 10
```

### 6.3 实际运行结果（2026-05-31, n=200, GPU 0, max_new_tokens=256）

| Checkpoint | Sanity | MindCube VQA (200样本) | vs base |
|---|---|---|---|
| **base** (Qwen2.5-VL-3B-Instruct) | 4/5 | 37.5% | — |
| **v25_step240** (val退化期: ID_m4≈0.24) | 4/5 | 39.5% | **+2.0% ≈ 持平** |
| **v26_step160** (当前最佳: ID_m4=0.607) | 4/5 | **41.5%** | **+4.0% ↗ 提升** |

> 注：首版测试 max_new_tokens=64 时准确率偏低（base=32.5%, v26=35.5%），因为模型先推理再给字母，64 tokens 常被截断、无法输出字母，均计为答错。修为 256 后数字更可信。

**按 type 细分对比**:

| Type | N | base | v25_step240 | v26_step160 | 解读 |
|---|---|---|---|---|---|
| `3` (3步推理) | 21 | 38.1% | 47.6% | **52.4%** | v26 最突出，+14.3pp vs base |
| `2` (2步推理) | 24 | 62.5% | 62.5% | **66.7%** | v26 略好 |
| `0_frame` (无图片) | 27 | 48.1% | 44.4% | **51.9%** | v26 超过 base |
| `2_frame` (双帧) | 28 | 39.3% | 42.9% | **46.4%** | v26 最好 |
| `three_view` (四视角) | 38 | 13.2% | **18.4%** | **18.4%** | 两版 RL 均显著改善 |
| `1_frame` (单帧) | 27 | 37.0% | 33.3% | 37.0% | v26 持平 base |
| `3_frame` (三帧) | 28 | 39.3% | 39.3% | 35.7% | 轻微下降，N 小，噪声 |
| `general` | 5 | 40.0% | 40.0% | 40.0% | N 过小，忽略 |

**核心结论：RL 训练没有导致通用空间推理退化**

- v26_step160 整体 **+4.0%** vs base，7 类中有 4 类高于 base
- 最突出：`3`（3步空间推理）v26=52.4% vs base=38.1%，**+14.3pp**，与导航训练任务高度相关
- v25_step240（val已退化）MindCube 仅 +2.0%，说明 val 退化时 backbone 轻微衰退，但量级很小
- Sanity check 全部 4/5（`spatial_left_right` 三个模型一致回答 D，是模型系统性倾向，非退化信号）

---

## 7. 新实验设计：v28_klhi_lr5e7_cosine

### 7.1 动机

- v26_klhi_lrdecay 的失败根本原因：**起始 LR=1e-6 就在不稳定区域**，1000步的余弦周期太长，前200步几乎没有衰减（9.92e-7 vs 1e-6，衰减<1%）
- v26_klhi_lr5e7 的成功表明：**5e-7 是稳定的起始 LR**
- 逻辑延伸：**以 5e-7 为起点做余弦衰减** 是两者的最优结合——既保持稳定学习，又通过衰减延缓 action concentration

### 7.2 核心改动（对比 v26_klhi_lr5e7）

| 参数 | v26_klhi_lr5e7 | **v28_klhi_lr5e7_cosine** | 理由 |
|------|---------------|--------------------------|------|
| ACTOR_LR | 5e-7（恒定）| **5e-7（起点）**  | 保持稳定起始 LR |
| lr_scheduler_type | — (linear default) | **cosine** | 引入衰减 |
| lr_warmup_steps_ratio | — | **0** (无warmup) | 同 lrdecay，直接从max LR起步 |
| min_lr_ratio | — | **0.1** (最小5e-8) | 保留残余学习信号 |
| TOTAL_STEPS | 2000 | **1000** | 余弦在1000步内完成一个周期，衰减更有效 |
| 其他 | KL=0.20, entropy=0.005 | **不变** | 保持稳定配置 |

### 7.3 预期行为

LR 曲线对比：

| step | v25_klhi (1e-6恒定) | v26_lr5e7 (5e-7恒定) | **v28_cosine (5e-7余弦)** |
|------|---------------------|---------------------|--------------------------|
| 50   | 1e-6 | 5e-7 | ~4.9e-7 |
| 100  | 1e-6 | 5e-7 | ~4.5e-7 |
| 200  | 1e-6 | 5e-7 | ~3.5e-7 |
| 300  | 1e-6 | 5e-7 | **2.5e-7** |
| 500  | 1e-6 | 5e-7 | **5e-8** (min) |

- step 100 时 LR≈4.5e-7（轻微衰减，保持强学习信号）
- step 300 时 LR≈2.5e-7（比 v26 慢 2 倍，更保守）
- step 500 时 LR≈5e-8（非常保守，防止集中化）

### 7.4 科学假设

若 v26_lr5e7 在 step 200+ 出现同样的 action concentration (act_H<0.5) 和验证退化，则：
- v28_cosine 应该能通过 LR 衰减**延缓该退化**，可能在 step 300-400 时仍保持较好 ID_m4

### 7.5 启动命令

```bash
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v28_klhi_lr5e7_cosine.sh \
    > v28_klhi_lr5e7_cosine.log 2>&1 &
echo "PID: $!"
```

---

## 8. 实验状态总览（截至 2026-06-03）

### Qwen2.5-VL-3B 系列

| 实验 | 状态 | 当前步数 | 最佳 ID_m4 | 备注 |
|------|------|---------|-----------|------|
| v25_klhi | 已完成 | ~300 | 0.500 (@150) | 基准线，峰值后退化 |
| v26_klhi_lr5e7 | 已完成 | ~241 | **0.607 (@150)** | Qwen 历史最高 m4 |
| v26_klhi_lrdecay | 已终止 | ~181 | 0.560 (@100) | entropy爆炸 |
| v27_from_v25ckpt | 已终止 | ~94 | 0.381 | entropy爆炸(2.62)，恢复失败 |
| v28_klhi_lr5e7_cosine | 已终止 | ~399 | 0.5595 (@200) | ID_b4=0.9048历史最高；cosine延迟collapse140步 |
| **v29_grpo** | **待启动** | — | — | 消融：仅改算法 PPO→GRPO（见§13） |
| **v30_rew_scale** | **待启动** | — | — | 消融：仅改奖励 success 50→5（见§13） |

### Cambrian-S 7B 系列（详见 cambrian_iteration.md）

| 实验 | 状态 | 当前步数 | 最佳 ID_m4 | 备注 |
|------|------|---------|-----------|------|
| c1_groupadv_100scenes | 已终止 | ~79 | 0.071 (@50) | turn_left 偏好，近零成功率 |
| c2_nfp_groupadv | 已终止 | ~77 | 0.048 (@50) | 同上，更差 |
| c3_lr5e7 | 运行中 | ~49 | — | LR=5e-7+WARMUP=120，step 50 待出 |
| c4_fwdfirst | 运行中 | ~55 | **0.405 (@50)** | ✅ fwd-first 提示，**最佳 Cambrian 结果** |
| c5_entropy_hi | 运行中 | ~5 | — | entropy=0.02，速度慢，待数据 |
| c6_no_think | 已终止 | ~69 | 0.298 (@50) | no_think 格式不适合，OOD仅c4的38% |
| c7_fwdfirst_ehi | 待启动 | — | — | fwd-first + entropy=0.01，c4改进版 |

### 当前操作建议（2026-06-03）
1. ~~**终止 c1/c2**~~ → **已完成**（释放 GPU 用于后续实验）
2. **等待 c3/c4/c5 运行至 step 100+**，c4 是当前 Cambrian 最优基线
3. **启动 v29_grpo / v30_rew_scale**（GPU 空出后），验证 v28 collapse 根因假设
4. c7_fwdfirst_ehi 在 c4 达到 step 100 后启动（可从 c4 step80 ckpt 续训）

---

## 9. v28 完整实验分析（step 0-362，截至 2026-06-02）

### 9.1 早期阶段（step 0-150）: v26 vs v28 对比

| 指标 | v26_lr5e7 @150 | v28_cosine @150 | 说明 |
|------|---------------|-----------------|------|
| ID_m4 | **0.607** | 0.512 | v26 高 19%（v28 仍在上升）|
| entropy | 0.762 | 0.685 | v28 更低，更健康 |
| act_H | 0.954 | 0.936 | v28 稍更分散 |
| LR | 5.00e-7 | 4.93e-7 | v28 已开始衰减 |
| OOD_m4 | **0.500** | 0.474 | v26 较好 |

→ 早期 v28 entropy 比 v26 低约 10%，act_H 略高，符合 cosine LR 抑制过度优化的预期。但在 step 150 时 v28 的 ID_m4 仍明显落后于 v26，说明 v28 的峰值到来更晚（step 200）。

### 9.2 step 200 峰值与随后退化

v28 在 step 200 达到历史峰值：
- **ID_b4 = 0.9048**（所有 Qwen 实验历史最高）
- **ID_m4 = 0.5595**（未超越 v26 的 0.607）

随后退化轨迹：

| step | ID_m4 | entropy | LR | act_H | 状态 |
|------|-------|---------|-----|-------|------|
| 200  | **0.5595** | 0.700 | 4.79e-7 | 0.770 | ← 峰值 |
| 250  | 0.5595 | — | 4.65e-7 | 0.792 | 平台期 |
| 300  | 0.5357 | 0.753 | 4.39e-7 | 0.685 | 开始下滑 |
| 320  | — | **1.015** | 4.29e-7 | 0.769 | entropy 突破 1.0 |
| 350  | 0.3929 | — | — | — | 急剧下滑 |
| 360  | — | **1.094** | **4.07e-7** | **0.935** | entropy 爆炸，多样性异常回升 |

### 9.3 v28 vs v26 最终对比

| 指标 | v26_klhi_lr5e7 | v28_cosine | 结论 |
|------|---------------|------------|------|
| 峰值 ID_m4 | **0.607** (@150) | 0.5595 (@200) | v26 仍领先 7.6% |
| 峰值 ID_b4 | 0.857 (@150) | **0.905** (@200) | v28 历史最高 |
| 熵爆炸 step | ~step 180+ | ~step 320+ | v28 **延后约 140 步** |
| 峰值后 m4 退化 | 缓降 | 急降 | v28 退化更快 |

### 9.4 结论

**cosine LR 假设部分成立**：LR 衰减成功将熵爆炸时间点从 step~180 推迟到 step~320，延缓约 140 步。act_H 在 step 160 时显著优于 v26 同期（0.921 vs 0.677）。

**但峰值 m4 未超越 v26**：LR 衰减使 step 100-150 的梯度更新略弱，导致峰值略低于 v26 的恒定 LR=5e-7 配置。

**核心矛盾依然存在**：无论 LR 如何调整，Qwen2.5-VL-3B 的 PPO 训练在 step 200-350 都会出现 entropy 爆炸 → 退化。LR 只能推迟而不能根治。根本解决方案需要从 reward shaping 或 SFT warmstart 角度切入。

---

## 10. c1/c2 Cambrian 实验分析（截至 2026-06-02）

### 10.1 实验设计

| 参数 | c1_groupadv_100scenes | c2_nfp_groupadv |
|------|----------------------|------------------|
| 模型 | cambrian-s-7b | cambrian-s-7b |
| ENV | env_config_v24_100scenes.yaml | 同上 |
| use_remove_padding | True | **False**（NFP）|
| 其他配置 | 与 v26 完全相同（LR=1e-6, KL=0.20, N=4）| 同左 |

### 10.2 训练与验证指标

**c1_groupadv_100scenes**（step ~79）:

| step | ID_b4 | ID_m4 | t_succ% | entropy | dominant action | act_H |
|------|-------|-------|---------|---------|-----------------|-------|
| 50 (val) | 0.2381 | 0.0714 | — | — | — | — |
| 1  | — | — | 1.11% | 0.884 | turn_left (46%) | 1.104 |
| 60 | — | — | 0.56% | 0.881 | turn_left (42%) | 1.116 |
| 80 | — | — | ~0.5% | — | turn_left (47%) | 1.079 |

**c2_nfp_groupadv**（step ~77）:

| step | ID_b4 | ID_m4 | t_succ% | entropy | dominant action | act_H |
|------|-------|-------|---------|---------|-----------------|-------|
| 50 (val) | 0.1905 | 0.0476 | — | — | — | — |
| 1  | — | — | 0.54% | 0.884 | turn_left (47%) | 1.054 |
| 60 | — | — | 0.28% | 0.878 | turn_left (45%) | 1.126 |
| 77 | — | — | ~0.3% | — | turn_left (50%) | 1.115 |

### 10.3 失败模式分析

**与 Qwen 实验的根本差异**：

| 指标 | Qwen v26（step 1）| Cambrian c1（step 1）| 含义 |
|------|------------------|---------------------|------|
| dominant action | move_forward (72%) | **turn_left (46%)** | 行为起点完全不同 |
| act_H | 0.776 | **1.104** | Cambrian 初始随机性更高 |
| t_succ% | ~1-2%（逐步上升）| ~0.5-1.1%（**无改善**）| 无学习信号 |
| val ID_m4 @50 | 0.417 | 0.071 | 性能差 6 倍 |

**失败模式诊断**：
1. **转圈行为（Spinning）**: act_H=1.1 + turn_left 主导，模型在 rollout 中持续旋转而非前进
2. **无学习信号**: t_succ% 在整个训练中维持 0.3-1.1%，未见上升趋势（对比 Qwen v26 同期 ~5-10%）
3. **reward sparsity 恶化**: 转圈行为几乎无法到达目标，GroupAdv 组内全为负信号，RL 无法有效学习
4. **可能原因**: Cambrian 的视觉编码器或指令跟随能力与任务提示格式不匹配；预训练中缺乏第一人称导航数据

### 10.4 建议

| 选项 | 操作 | 科学价值 |
|------|------|---------|
| **终止 c1/c2** | Kill 两个进程，释放 GPU | 避免资源浪费，当前无改善趋势 |
| **调整提示格式** | 修改 agent prompt 适配 Cambrian 指令格式 | 验证是否提示格式导致转圈 |
| **添加方向奖励** | 朝向目标转向给小正奖励 | 解决 reward sparsity |

**短期建议**: 先终止 c1/c2，释放 GPU 用于 Qwen 后续实验。若要继续 Cambrian，需先做 prompt engineering 分析。

---

## 11. 未来方向（中长期，更新于 2026-06-02）

| 实验方向 | 优先级 | 前置条件 | 科学问题 |
|---------|--------|---------|---------|
| **SFT → RL 两阶段** | ★★★ | 生成 SFT 数据 | SFT warmstart 是否完全消除 entropy 爆炸？ |
| **v29: 更激进 LR 衰减** | ★★★ | v28 已完成 | 5e-7→1e-9 或缩短 cosine 周期是否有效？ |
| **entropy 系数提升** | ★★ | — | entropy_coeff 0.005→0.01 是否能显著抑制熵爆炸？ |
| KL 系数消融 | ★★ | v28 已完成 | KL=0.20 是否可提高（0.30/0.40）来增强锚定？ |
| 更多 val seeds (N=8) | ★★ | GPU 空闲 | 当前 val 方差估计是否准确？ |
| backbone probe 定期运行 | ★★ | probe_spatial_qa.py | RL 是否在损害通用空间推理能力？ |
| Cambrian 修复实验 | ★ | 提示格式分析 | 换用适配 Cambrian 的提示是否解锁导航能力？ |

---

## 12. v28 崩塌根因深度诊断（2026-06-03）

基于 v28 完整训练日志（step 160-399）的系统性分析，从四个维度定位 collapse 机因。

### 12.1 诊断数据汇总

| step | entropy | t_succ% | crit_sc | crit_ret | grad_norm | rew_var |
|------|---------|---------|---------|----------|-----------|---------|
| 160  | 0.685 | 3.05 | 1.517 | 7.045  | 4.845 | 82.7  |
| 180  | 0.700 | 2.82 | 1.404 | 6.212  | 4.144 | 96.2  |
| **200** | **0.700** | **6.84** | **3.447** | **16.956** | **5.264** | 175.7 |
| 220  | 0.706 | 2.56 | 1.267 | 6.367  | 4.657 | 74.2  |
| 280  | 0.960 | 4.39 | 2.192 | 9.668  | 4.508 | 130.4 |
| **320** | **1.015** | **2.66** | **1.330** | 6.491  | **9.648** | 75.4  |
| 340  | 0.854 | 7.43 | 3.737 | 17.905 | 4.381 | 195.6 |
| 362  | 1.419 | 4.57 | 2.278 | 11.638 | 5.202 | 119.7 |
| 398  | 1.329 | 2.37 | 1.147 | 5.422  | 4.579 | 68.9  |
| 399  | 1.362 | 4.46 | 2.209 | 8.902  | 5.611 | 139.1 |

### 12.2 Critic 是否崩溃？

**结论：否。Critic 全程健康，但是不稳定性的放大器。**

- `crit_ret / crit_sc` 比值在 4.1–5.2× 间震荡，符合 γ=0.95、平均约 5.8 步的理论预期（`1/(1-0.95)=20`，折算多步 ≈ 4.7×），无趋势性崩溃
- 对比真实 critic 崩溃案例（v22_kl_ablation: critic/score 从 3.142 降至 0.407），v28 完全不同
- **Critic 的不稳定作用**：step 200 时 crit_ret=16.956（过估），actor 更新过猛 → step 220 t_succ 急跌 2.56%；这种"过估→过更新→失误"的震荡在整个训练中周期出现

### 12.3 Collapse 触发机制

**触发点**：step 320 `grad_norm` 尖峰至 **9.648**（正常水平 ~4-5，约 2×），将 entropy 推过 1.0 阈值。

格式输出在 step 398-399 出现明显退化：
- 错误 XML 标签（`<stdaction>`, `<thought>`, `<sink>`）
- 无效动作名（`turn_forward`, `find_target`）
- 中文混入（`动作: move_forward|`）
- 但 **非完全崩溃**：t_succ 仍 2-4%，仍有约 60-70% 的响应格式正确

### 12.4 根因分析

| 根因 | 证据 | 权重 |
|------|------|------|
| **PPO reference policy 过时（主因）** | LR 从 5e-7 降至 4.07e-7（-17%），但 entropy 从 0.700 升至 1.419（+107%）。若 LR 是唯一原因，entropy 应随 LR 下降而下降。事实相反，说明有独立的不稳定源：固定 π_ref 随训练步数逐渐失去 KL 锚定能力 | ★★★ |
| **奖励方差过高（次因）** | `rew_var=75-212` 贯穿全程（`crit/score/max>50`）。根本原因：`success_reward=50.0` vs 失败轨迹 ≈ 0-5，组内方差 ≈ 50²/4≈625，低 SNR 梯度使 step 320 的噪声批次直接引爆 grad_norm | ★★ |
| **3B 模型容量限制（次因）** | b4-m4 gap=34.5%，m4-w4 gap=51%；模型可以成功（b4=90%）但不能稳定复现，反映工作记忆不足以精确计数多步 | ★ |
| Critic 过估震荡（次因）| step 200 过估→step 220 崩跌的周期性震荡削弱了有效训练窗口 | ★ |
| 数据不平衡（微小因素）| centering 任务仅约 54 samples/batch，覆盖不足，但 collapse 是 task-type-general | ☆ |

### 12.5 关键结论

- **v28 collapse 不是 LR 问题**（cosine 已验证）；核心是 PPO 架构对多步 RL 的固有脆弱性
- **两个独立根因**：reference staleness（算法层面）+ rew_var（奖励层面），各自可独立通过 GRPO 或奖励重设计来解决
- **不需要 Constrained Decoding**：格式崩溃是 entropy 爆炸的症状，修好根因即可；强制屏蔽非法 token 会引入 IS 偏差，得不偿失

---

## 13. v29/v30 消融实验设计（2026-06-03）

### 13.1 消融逻辑

基于 §12 的两个独立根因，设计两个单因素消融实验，其余参数与 v28 完全一致：

| 实验 | 算法 | success_reward | potential_scale | near_bonus | 测试假设 |
|------|------|---------|---------|------|------|
| v28 | PPO (no_concat_gae) | 50 | 0.5 | 0.2 | baseline（已 collapse） |
| **v29_grpo** | **GRPO** | 50 | 0.5 | **0.5** | reference staleness 是否是主因？ |
| **v30_rew_scale** | PPO | **5** | **1.0** | **0.5** | rew_var 是否是主因？ |

near_success_bonus 在 v29/v30 均从 0.2→0.5（last-mile 梯度增强，两个实验都做）。

### 13.2 v29_grpo 设计细节

- **算法改动**：`ADV_ESTIMATOR="grpo"`，critic 自动禁用（`need_critic()` 检测到非 GAE 返回 False）
- **GRPO 优势**：advantage 由同组 N=4 轨迹相对奖励归一化，无固定 π_ref，从根本上消除 reference staleness
- **uid 语义已验证**：代码 `group_idx = uid` 确认为 per-trajectory，GRPO within-group normalization 正确
- **等价行为**：dense reward 下 GRPO = REINFORCE++（trajectory sum 后 normalize），对导航任务合理
- **文件**：`experiments/v29_grpo.sh` + `env_config_v24_100scenes_lm.yaml`

### 13.3 v30_rew_scale 设计细节

- **奖励改动**（纯 yaml，无代码修改）：
  - `success_reward: 50.0 → 5.0`（降 10×）
  - `potential_field_reward_scale: 0.5 → 1.0`（升 2×，维持进度信号相对强度）
  - 量级对齐后：成功/失败 advantage 差异从 `~50/1` 缩至 `~5/5`，预期 rew_var 从 75-212 降至 1-8
- **奖励比例不变**：最大进度奖励（`5m × 1.0 = 5.0`）≈ `success_reward=5.0`，比值约 1:1（原来是 1:10）
- **文件**：`experiments/v30_rew_scale.sh` + `env_config_v24_100scenes_rewscale.yaml`

### 13.4 决策树

```
v29 结果分析：
  entropy 在 step 300+ 仍 < 1.0 → reference staleness 是主因 → GRPO 是正确修复
  entropy 仍在 step 200 爆炸 → rew_var 是更主要因素 → 看 v30

v30 结果分析：
  crit_sc 震荡消失，grad_norm 稳定 → rew_var 是主因 → 奖励重设计足够
  仍然 collapse → 两个原因都需要修复 → v31: GRPO + rew_scale 组合
```

### 13.5 未来方向更新（替代 §11）

| 实验方向 | 优先级 | 状态 | 科学问题 |
|---------|--------|------|---------|
| **v29: GRPO 消融** | ★★★ | ✅ 脚本已就绪 | reference staleness 是否为 collapse 主因？ |
| **v30: 奖励重设计消融** | ★★★ | ✅ 脚本已就绪 | rew_var=75-212 是否为 collapse 次因？ |
| **v31: GRPO + rew_scale** | ★★★ | 待 v29/v30 结果 | 组合修复是否能突破 v26 的 0.607 天花板？ |
| **Cambrian c7_fwdfirst_ehi** | ★★ | ✅ 脚本已就绪 | entropy=0.01 是否降低 no_tag 率？ |
| **SFT → RL 两阶段** | ★★ | 需生成数据 | SFT warmstart 是否彻底消除 entropy 爆炸？ |
| backbone probe 定期运行 | ★ | probe_spatial_qa.py 已就绪 | RL 是否损害通用空间推理？ |

---

## 14. v29/v30/v31 最新结果分析（2026-06-16）

本节基于最新日志和评估输出：
- `python3 scripts/analyze_experiments.py --exps v29_grpo,v30_rew_scale,unnamed_0612_1457 --train --val --diversity --every 20`
- 以及对应目录下 `train.log`、`validation/*.jsonl` 的末段记录。

> 重要说明：本轮 v31 配置文件未显式设置 `EXPERIMENT_NAME`，因此运行目录落在 `unnamed_0612_1457/`，不是 `v31_grpo_rewscale/`。后者保留的是更早一轮短跑日志。

### 14.1 三版实验最新状态总览

| 实验 | 实际目录 | latest_step | val 最后步 | peak ID_m4 | peak OOD_m4 | 当前结论 |
|------|----------|------------:|-----------:|-----------:|------------:|----------|
| v29_grpo | exps/vagen_active_spatial/v29_grpo | 244 | 200 | 0.4881 (@0/150) | 0.5658 (@100) | 纯 GRPO 有早中期增益，但后段出现 entropy>1 与动作再集中 |
| v30_rew_scale | exps/vagen_active_spatial/v30_rew_scale | 332 | 300 | 0.5238 (@250) | 0.5526 (@200) | 奖励缩放显著降低 critic 震荡，整体稳定性优于 v29 |
| v31_grpo_rewscale | exps/vagen_active_spatial/unnamed_0612_1457 | 191 | 150 | 0.5357 (@100) | 0.4868 (@100) | 早期最好，但中段出现一次明显不稳尖峰，需继续观察 |

对比历史上限：
- v26 最佳 ID_m4 = **0.607** (@150)
- v28 最佳 ID_m4 = **0.5595** (@200)

当前三版都尚未超过 v26 的 0.607；v31 目前最高 0.5357，尚处中期。

### 14.2 关键参数变化与效果归因

对照关系（单因素 + 组合）：
- v29：仅改算法（PPO/no_concat_gae -> GRPO），奖励仍为高方差版本（success=50）
- v30：仅改奖励（rewscale），算法仍为 PPO/no_concat_gae
- v31：GRPO + rewscale 组合

从结果看：

1. GRPO 单独作用（v29）
- 优点：中早期可维持可用性能（ID_m4 在 0.44-0.49 带内）
- 问题：后段仍出现 entropy 上冲（step 200: 1.081；step 220: 1.167），并在末段出现动作集中化（step 244: act_H=0.587, move_forward=82.4%）
- 结论：仅去 reference staleness 不足以消除后段失稳

2. 奖励缩放单独作用（v30）
- 优点：critic 量级与震荡显著收敛（末段 critic/score 约 0.2-0.5，reward_variance 约 2-3），明显优于 v29（可到 1.5-4.1 与 100+）
- 指标：ID_m4 提升到 0.5238（@250），且 val 延长到 step 300 仍有 0.5119
- 风险：step 320 entropy 仍到 1.219，说明 rewscale 不是唯一根因

3. 组合方案早期表现（v31 = unnamed_0612_1457）
- step 100 达到三版当前最高 ID_m4=0.5357
- 但在 step 180 出现一次不稳信号：entropy=1.060，grad_norm=29.778（显著尖峰）
- step 190/191 又回落到 entropy 0.80/0.75、grad_norm 3-4，说明该尖峰可能是批次级冲击而非不可逆崩溃
- 结论：组合方案存在协同潜力，但证据仍不足（缺 step 200+ val）

### 14.3 是否达到前文预期（基于 §13 假设）

#### 预期 A：v29 验证 reference staleness 为主因
- 观察：v29 仍在 200+ 出现 entropy>1 和动作集中化
- 判断：**部分支持但不充分**。GRPO 有帮助，但不能单独根治 collapse。

#### 预期 B：v30 验证 rew_var 为主因
- 观察：v30 的 critic/reward 震荡显著下降，ID_m4 和稳定性优于 v29
- 判断：**强支持**。奖励方差是重要根因之一。

#### 预期 C：v31 组合修复突破旧上限
- 观察：目前仅到 step 191，peak ID_m4=0.5357 < v28(0.5595) < v26(0.607)
- 判断：**暂未达到**，但尚不能下结论（训练长度与 val 覆盖不足）。

### 14.4 当前阶段结论（2026-06-16）

1. collapse 机制判断更新
- 目前证据支持“双因子”框架：
  - 算法侧（reference staleness）
  - 奖励侧（高方差 rew_var）
- 两者缺一不可；单独改一项只能部分缓解。

2. 最有希望方向
- `GRPO + rewscale` 仍是主线方案（v31），但必须完成 step 300+ 才能验证是否真正跨过历史不稳定区间。

3. 监控先导指标
- 保持对 `entropy`、`act_H`、`dominant_action%`、`grad_norm` 的联动监控
- 特别关注阈值：
  - entropy > 1.0
  - act_H < 0.6 且 move_forward > 80%
  - grad_norm 突增到正常值 2 倍以上

### 14.5 下一步实验建议（可直接执行）

#### 实验优先级 1：把 v31 跑完整（先补证据）
- 目标：至少跑到 step 350（理想到 400）
- 必要输出：补齐 val@200/250/300/350
- 判据：
  - 是否在 200-300 保持 ID_m4 >= 0.54
  - 是否避免连续 entropy>1.0
  - 是否避免 act_H 持续 <0.6

#### 实验优先级 2：v32（保守步长版）
- 基线：复制 v31
- 仅改：`ACTOR_LR 5e-7 -> 3e-7`
- 目的：验证中段尖峰是否主要由步长过大触发

#### 实验优先级 3：v33（后段稳定约束版）
- 基线：复制 v31
- 改动：增强 KL 约束（如提高 `KL_LOSS_COEF` 或分段提高）
- 目的：压制 150+ policy 漂移

#### 实验优先级 4：v34（梯度保险版）
- 基线：复制 v31
- 仅改：`GRAD_CLIP 0.3 -> 0.2`
- 目的：抑制类似 step180 的 grad_norm 尖峰

#### 工程修复（立即执行）
- 在 v31 配置里显式设置：
  - `EXPERIMENT_NAME="v31_grpo_rewscale"`
- 避免后续日志继续落到 `unnamed_*`，影响实验追踪与自动分析。

---

## 15. v31 续训命令 + v32/v33/v34 脚本落地（2026-06-16）

本节记录已完成的工程实现：
- v31 续训脚本（针对当前真实目录 `unnamed_0612_1457`）
- v32/v33/v34 三个新实验脚本
- 启动命令、参数差异、预期判据

### 15.1 v31 如何继续跑？是否需要改脚本？

**结论：需要。**

原因：本轮 v31 运行时未设置 `EXPERIMENT_NAME`，真实训练目录在 `unnamed_0612_1457/`。若直接重跑原 `v31_grpo_rewscale.sh`，会新开一个实验目录，无法续写当前进度。

已新增续训脚本：
- `examples/train/active_spatial/experiments/v31_grpo_rewscale_resume.sh`

关键设置：
- `EXPERIMENT_NAME="unnamed_0612_1457"`
- `RESUME_MODE="auto"`
- `VAL_BEFORE_TRAIN="False"`（避免 resume 时重复 step0 验证）

继续跑命令（可直接执行）：

```bash
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v31_grpo_rewscale_resume.sh \
    > v31_grpo_rewscale_resume.log 2>&1 &
echo "PID: $!"
```

同时，已修复原始脚本：
- `examples/train/active_spatial/experiments/v31_grpo_rewscale.sh`
  - 新增 `EXPERIMENT_NAME="v31_grpo_rewscale"`

这保证后续“新开 v31”不会再落到 `unnamed_*`。

### 15.2 v32/v33/v34 脚本实现（已完成）

已新增 3 个实验脚本：
- `examples/train/active_spatial/experiments/v32_grpo_rewscale_lr3e7.sh`
- `examples/train/active_spatial/experiments/v33_grpo_rewscale_klhi.sh`
- `examples/train/active_spatial/experiments/v34_grpo_rewscale_clip02.sh`

统一基线：均基于 v31（GRPO + rewscale），`NUM_TRAIN_GPUS=4`、`RENDERING_GPU=4`、`TOTAL_STEPS=1000`、cosine LR 保持一致。

参数差异表：

| 实验 | 相对 v31 的改动 | 目的 |
|------|------------------|------|
| v32_grpo_rewscale_lr3e7 | `ACTOR_LR: 5e-7 -> 3e-7` | 验证中段尖峰是否主要由步长过大触发 |
| v33_grpo_rewscale_klhi | `KL_LOSS_COEF: 0.20 -> 0.30`; `KL_COEF: 0.001 -> 0.002` | 增强 KL 约束，抑制 150+ policy 漂移 |
| v34_grpo_rewscale_clip02 | `GRAD_CLIP: 0.3 -> 0.2` | 抑制梯度尖峰导致的瞬时失稳 |

### 15.3 启动命令（v32/v33/v34）

```bash
# v32
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v32_grpo_rewscale_lr3e7.sh \
    > v32_grpo_rewscale_lr3e7.log 2>&1 &

# v33
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v33_grpo_rewscale_klhi.sh \
    > v33_grpo_rewscale_klhi.log 2>&1 &

# v34
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v34_grpo_rewscale_clip02.sh \
    > v34_grpo_rewscale_clip02.log 2>&1 &
```

### 15.4 预期结论与判据（用于对比分析）

统一观测窗口：step 100 / 150 / 200 / 250 / 300 / 350。

核心监控指标：
- 验证：`ID_m4`, `OOD_m4`
- 训练：`actor/entropy`, `actor/grad_norm`, `train/traj_success/mean`
- 多样性：`act_H`, `dominant_action%`

判据阈值（沿用 §14）：
- 风险阈值：`entropy > 1.0`
- 集中化阈值：`act_H < 0.6` 且 `move_forward > 80%`
- 梯度风险：`grad_norm` 达常态两倍以上

各实验预期：

1. v32（低 LR）
- 预期：entropy 上冲推迟或幅度下降，`grad_norm` 尖峰频率降低
- 风险：学习变慢，`ID_m4@150` 可能低于 v31
- 成功信号：`ID_m4@250/300` 不低于 v31 同步，且 entropy 不连续越过 1.0

2. v33（高 KL）
- 预期：policy 漂移减轻，后段曲线更平滑
- 风险：约束过强导致 exploration 不足，`ID_b4` 增长放缓
- 成功信号：`ID_m4` 平台期更长，`OOD_m4` 不出现明显早衰

3. v34（强 clip）
- 预期：抑制批次级突发尖峰（类似 step180）
- 风险：更新幅度受限，可能在中后段出现欠优化
- 成功信号：`grad_norm` 尾部显著收敛，val 曲线波动减小

建议比较基线：
- 主要基线：`unnamed_0612_1457`（当前 v31 实际运行目录）
- 历史上限：v26 (`ID_m4=0.607`) 与 v28 (`ID_m4=0.5595`)

---

## 16. v32/v33/v34 实验结果分析（2026-06-20）

本节基于三版实验的完整训练/验证/多样性数据（截至各自 latest_step）。

### 16.1 关键指标总览

| 实验 | 最新步 | 峰值 ID_m4 | @step | entropy @最新 | act_H @最新 | move_fwd% | rew_var |
|------|--------|-----------|-------|--------------|------------|-----------|---------|
| **v32** (LR=3e-7) | 189 | **0.5833** | 100 | 0.754 | 0.642 ⚠ | 80% ⚠ | ~0.3 |
| **v33** (KL=0.30) | 151 | 0.5595 | 150 ↗ | ~0.62 | **0.863** ✓ | 69% ✓ | ~1.0 |
| **v34** (clip=0.2) | 151 | 0.5476 | 100 | ~0.67 | 0.859 ✓ | 70% ✓ | ~1.0 |

历史对照：v26 = 0.607 @150，v28 = 0.5595 @200，v31 = 0.5357 @100

### 16.2 val 曲线

**v32_grpo_rewscale_lr3e7**:

| step | ID_b4 | ID_m4 | ID_w4 | OOD_b4 | OOD_m4 | OOD_w4 |
|------|-------|-------|-------|--------|--------|--------|
| 0    | 0.6667 | 0.4048 | 0.0952 | 0.8421 | 0.4079 | 0.1053 |
| 50   | 0.8095 | 0.4762 | 0.0476 | 0.8421 | 0.4605 | 0.1053 |
| **100** | 0.8095 | **0.5833** | 0.2857 | **0.9474** | 0.4605 | 0.2105 |
| 150  | 0.8571 | 0.5119 | 0.1905 | 0.8947 | 0.5132 | 0.1579 |

**v33_grpo_rewscale_klhi**:

| step | ID_b4 | ID_m4 | ID_w4 | OOD_b4 | OOD_m4 | OOD_w4 |
|------|-------|-------|-------|--------|--------|--------|
| 0    | 0.8571 | 0.3690 | 0.0000 | 0.9474 | 0.5132 | 0.0526 |
| 50   | 0.6667 | 0.4881 | 0.2381 | 0.8421 | 0.4079 | 0.0000 |
| 100  | **0.9048** | 0.5357 | 0.2381 | 0.8947 | 0.4605 | 0.1053 |
| **150** | 0.8571 | **0.5595** ↗ | 0.1905 | 0.8947 | **0.5000** | 0.1053 |

**v34_grpo_rewscale_clip02**:

| step | ID_b4 | ID_m4 | ID_w4 | OOD_b4 | OOD_m4 | OOD_w4 |
|------|-------|-------|-------|--------|--------|--------|
| 0    | 0.7619 | 0.4405 | 0.0952 | 0.7895 | 0.3684 | 0.0526 |
| 50   | 0.7143 | 0.4167 | 0.1905 | 0.8421 | 0.4737 | 0.0526 |
| **100** | 0.8571 | **0.5476** | 0.0952 | 0.7895 | 0.4211 | 0.1579 |
| 150  | 0.8095 | 0.4881 | 0.1429 | 0.8421 | 0.5132 | 0.2105 |

### 16.3 训练动态对比

| step | v32 entropy | v33 entropy | v34 entropy | v32 act_H | v33 act_H | v34 act_H |
|------|------------|------------|------------|----------|----------|----------|
| 60   | 0.502 | 0.542 | 0.525 | 0.984 | 0.960 | 0.858 |
| 80   | 0.553 | 0.535 | 0.711 | 0.901 | 0.985 | 1.004 |
| 100  | 0.551 | 0.565 | 0.641 | 0.883 | 0.823 | 0.941 |
| 120  | 0.622 | **0.703** | 0.711 | 0.822 | 0.942 | 0.998 |
| 140  | 0.655 | 0.618 | 0.671 | 0.885 | **0.959** | 0.966 |
| 180  | 0.754 | — | — | 0.722 | — | — |
| 189  | — | — | — | 0.642 ⚠ | — | — |

**LR (v32 vs v33/v34)**:
- v32：LR=3e-7（恒定），critic_warmup 结束后直接从 3e-7 开始（cosine 起始值也是 3e-7）
- v33/v34：LR cosine 5e-7 → 5e-8，step 150 约 4.93e-7

**reward_variance**:
- 三版均维持 0.8-2.1（远低于 v28/v31 的 75-212），rewscale 配置（success=5）持续有效。

### 16.4 逐实验结论

#### v32（LR=3e-7）—— 熵健康但动作集中

- 峰值 ID_m4=0.5833 @step100 是三版最高单步值，但 step150 已回落至 0.5119，**峰值已过**
- entropy 全程最低（0.75 @step180），**从未超过 1.0**，LR 降低确实防止了 entropy 爆炸
- **关键问题**：act_H step1→189 从 0.974 持续下降至 0.642，move_forward 升至 **80%**，触及警戒阈值
- val 退化（step100→150）与 action concentration 同步发生，复现了 v26 晚期模式
- 结论：较低 LR 防止了 entropy 爆炸，但**未能阻止 action concentration**，是另一条通往 val 退化的独立路径

#### v33（KL=0.30）—— 当前最优，唯一仍在上升 ⭐

- step150 ID_m4=0.5595 **仍处于上升趋势**，是三版中唯一未出现峰值回落的实验
- entropy 全程最健康（最高 0.703 @step120，随后回落至 0.618），kl_loss 稳定维持 0.0002-0.0004
- act_H 全程 0.82-0.99，move_forward 仅 59-70%，**所有实验中最佳多样性**
- 作用机制：KL=0.30 直接锚定 policy，从根本上阻止 policy 漂移，同时抑制了 entropy 爆炸和 action concentration 两条失稳路径
- **结论：强 KL 约束是目前最有效的单一修改**

#### v34（grad_clip=0.2）—— 多样性好但效用最低

- 峰值 ID_m4=0.5476 @step100，step150 回落至 0.4881，是三版最弱
- act_H 反而最高（0.859-1.004），move_forward 最低（step80 时仅 44.8%）
- **悖论**：探索性最好但 val 性能最差——更强的 clip 限制了有效梯度更新，策略改进信号被削弱
- 结论：单独降低 grad_clip 是负效益改动；多样性高的代价是学习效率下降

### 16.5 机制认知更新

基于三版实验，对 §12 的根因分析补充第三条失稳路径：

```
路径 A（entropy 爆炸）：policy 分布随机化 → 有效导航信号丢失
路径 B（action concentration）：move_forward 集中化 → val 性能退化
路径 C（policy 漂移）：KL 失锚 → 同时触发 A 和 B

修复效果：
  KL=0.30 (v33)  →  切断路径 C  →  A 和 B 均被抑制  ★★★
  LR=3e-7 (v32)  →  延迟路径 A  →  路径 B 仍独立发生  ★★
  clip=0.2 (v34) →  对 A/B 无帮助，反而损害学习效率  ★
```

### 16.6 各修改方式效用排序

1. **KL_coef 提升**（v33）：★★★ — 最根本的稳定化手段，同时解决两条失稳路径
2. **LR 降低**（v32）：★★ — 延迟 entropy 爆炸但不阻止 action concentration，峰值早于预期
3. **grad_clip 降低**（v34）：★ — 负效益，提高探索多样性但降低了有效学习效率

### 16.7 实验状态总览（2026-06-20 更新）

| 实验 | 状态 | 最新步 | 峰值 ID_m4 | 备注 |
|------|------|--------|-----------|------|
| v26_klhi_lr5e7 | 已完成 | ~241 | **0.607** (@150) | Qwen 历史最高 m4 |
| v28_klhi_lr5e7_cosine | 已完成 | ~399 | 0.5595 (@200) | cosine 延迟 collapse 140步 |
| unnamed_0612_1457 (v31) | 运行中/可续 | 191 | 0.5357 (@100) | 最新步 ckpt@150，待续跑 |
| v29_grpo | 已运行 | 244 | 0.4881 (@150) | 仅 GRPO 不足以解决 collapse |
| v30_rew_scale | 已运行 | 332 | 0.5238 (@250) | rewscale 有效但不充分 |
| **v32_grpo_rewscale_lr3e7** | 运行中 | 189 | 0.5833 (@100) | ⚠ act_H 0.642，峰值已过 |
| **v33_grpo_rewscale_klhi** | 运行中/可续 | 151 | 0.5595 (@150) ↗ | ✅ 最优，仍上升，待继续 |
| **v34_grpo_rewscale_clip02** | 运行中 | 151 | 0.5476 (@100) | 峰值已过，低优先级 |

### 16.8 下一步实验建议

#### 立即：继续 v33 到 step 300+

v33 是当前唯一仍在上升且健康的实验，必须跑完 step 200-300 才能验证是否突破 0.607 天花板。

观察判据：
- **成功信号**：step 200-250 时 ID_m4 ≥ 0.57，entropy 持续 < 1.0，act_H > 0.75
- **失败信号**：entropy 突破 1.0，或 act_H 连续 2 个 step 低于 0.65

#### 中期：v35（等 v33 达到 step 200 后设计）

推荐方向：组合 v33 的 KL=0.30 + v32 的 LR=3e-7，构成当前最有理论依据的 belt+suspenders 组合：

| 参数 | v31 | v35 建议 | 理由 |
|------|-----|---------|------|
| KL_LOSS_COEF | 0.20 | **0.30** | v33 验证有效 |
| ACTOR_LR | 5e-7 | **3e-7** | v32 验证 entropy 更健康 |
| GRAD_CLIP | 0.3 | **0.3** | v34 证明降低无益 |

#### 资源策略

- v34 峰值已过，如 GPU 紧张可终止
- v32 act_H 接近警戒线，若 step 200 时 act_H < 0.6 则可终止
- v31 续跑（unnamed_0612_1457）仍有价值，提供 GRPO+rewscale 基线完整曲线

### 16.9 续跑命令

```bash
# v31 续跑（unnamed_0612_1457，从 step 150 ckpt 继续）
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v31_grpo_rewscale_resume.sh \
    > v31_grpo_rewscale_resume.log 2>&1 &
echo "v31 PID: $!"

# v33 续跑（v33_grpo_rewscale_klhi，从 step 150 ckpt 继续）
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v33_grpo_rewscale_klhi_resume.sh \
    > v33_grpo_rewscale_klhi_resume.log 2>&1 &
echo "v33 PID: $!"
```



---

## 17. GRPO 总结性结论与 v35/v36 算法对照实验设计（2026-06-20）

### 17.1 GRPO vs masked_gae 完整历史对比

| 算法 | 实验 | env | 峰值 ID_m4 | @step | 备注 |
|------|------|-----|-----------|-------|------|
| masked_gae | v26_klhi_lr5e7 | 100scenes（旧） | **0.607** | 150 | 历史最高 |
| masked_gae | v28_cosine | 100scenes（旧） | 0.5595 | 200 | cosine LR |
| masked_gae | v30_rew_scale | rewscale | 0.524 | 250 | 引入低方差奖励 |
| GRPO | v29 | 旧 | 0.488 | 150 | 纯 GRPO，基线最差 |
| GRPO | v31 | rewscale | 0.536 | 100 | GRPO+rewscale |
| GRPO | v32 | rewscale | 0.583 | 100↘ | +LR=3e-7，已峰值 |
| GRPO | **v33** | rewscale | **0.560** | 150↗ | +KL=0.30，**仍上升** |

**关键空白**：从未有过 `masked_gae + rewscale + KL=0.30` 的实验。
v30（masked_gae+rewscale）用的是旧 KL=0.20，且结果（0.524）不如 v26（0.607）。

### 17.2 GRPO 总结性结论

**GRPO 引入的动机（v29 时）**：PPO 的 advantage 估计存在 reference staleness，GRPO 通过组内归一化绕开 critic，理论上更稳定。

**实际发现（v29-v34 六个实验后）**：

1. **崩塌机制与算法无关**：entropy 爆炸和 action concentration 在 GRPO 和 masked_gae 中均出现，根因是 KL 散度失锚，不是 advantage 估计方式
2. **GRPO 未能超越 masked_gae 历史最高**：v33（GRPO 最优）= 0.560，低于 v26（masked_gae）= 0.607
3. **GRPO 的探索价值在超参而非算法本身**：rewscale（低方差奖励）和 KL=0.30 是两个关键改进，均可移植回 masked_gae
4. **消融矩阵中存在空白**：`masked_gae + rewscale + KL=0.30` 从未测试，这是最有潜力的未验证组合

**修复效果排序**（跨算法一致）：
```
KL_coef 提升（0.20→0.30）  ★★★  最根本：切断 policy 漂移根因
rewscale（success 50→5）   ★★★  reward variance 降低，critic/GRPO 信号更清晰
LR 降低（5e-7→3e-7）       ★★   延迟 entropy 爆炸，但不阻止 action concentration
grad_clip 降低（0.3→0.2）  ★    负效益：多样性↑ 但有效学习↓
```

### 17.3 未验证的核心问题

> **"masked_gae + rewscale + KL=0.30" 能否超越 v26 (masked_gae + no rewscale) 的 0.607？**

这个问题的答案决定后续方向：

- 若 v35（masked_gae+rewscale+KL=0.30）> 0.607 → **所有超参改进均有效，算法回归 masked_gae**
- 若 v35 ≈ v33（GRPO，0.560 上升中）→ **算法差异不显著，继续 GRPO 也可**
- 若 v35 < v33 → **GRPO 在低方差奖励场景下有独特优势**（需要解释机制）

### 17.4 v35/v36 实验设计

**消融矩阵（更新）**：

```
                    │  KL=0.20     │  KL=0.30
────────────────────┼──────────────┼────────────────────
GRPO + rewscale     │ v31 (0.536)  │ v33 (0.560↗)
masked_gae + rewscale│ v30 (0.524) │ v35 ★ (待测)
masked_gae + rewscale│              │ v36 ★ (LR=3e-7, 待测)
```

#### v35：纯算法对照（v33 的唯一变量：grpo → masked_gae）

| 参数 | v33 (GRPO) | **v35 (masked_gae)** | 理由 |
|------|-----------|---------------------|------|
| ADV_ESTIMATOR | grpo | **masked_gae** | ★ 唯一变量 |
| KL_LOSS_COEF | 0.30 | 0.30 | 对照 v33 |
| ACTOR_LR | 5e-7 cosine | 5e-7 cosine | 对照 v33 |
| ENV_CONFIG | rewscale | rewscale | 对照 v33 |
| 其他全部 | 同 v33 | 同 v33 | — |

科学问题：`masked_gae` vs `grpo`，其余完全对照。

#### v36：belt+suspenders 最优组合

| 参数 | v35 | **v36** | 来源 |
|------|-----|---------|------|
| ADV_ESTIMATOR | masked_gae | masked_gae | — |
| KL_LOSS_COEF | 0.30 | 0.30 | v33 验证 |
| ACTOR_LR | 5e-7 cosine | **3e-7 cosine** | v32 验证 |

科学问题：在 masked_gae 下，LR=3e-7 是否如在 GRPO 下一样有效延缓失稳。

### 17.5 实验优先级与触发条件

```
[现在运行]    v33 续跑（已有 ckpt@150，继续到 step 300+）
                ↓ step 200-250 评估
         ┌──────────────────────────┬──────────────────────────┐
         │ ID_m4 > 0.57，仍上升     │  ID_m4 停滞/回落          │
         ↓                          ↓
[并行启动] v35 + v36               [先启动 v35 快速验证]
         （验证 masked_gae 对照）    然后视 v35 结果决定 v36
```

推荐并行策略：若 GPU 充足，v35 和 v36 可同时启动（不依赖 v33 结果）。
若 GPU 紧张：等 v33@200 后，终止 v34（已峰值），腾出资源给 v35。

### 17.6 启动命令

```bash
# v35：masked_gae + rewscale + KL=0.30（纯算法对照 v33）
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v35_masked_gae_rewscale_kl30.sh \
    > v35_masked_gae_rewscale_kl30.log 2>&1 &
echo "v35 PID: $!"

# v36：masked_gae + rewscale + KL=0.30 + LR=3e-7（belt+suspenders）
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v36_masked_gae_rewscale_kl30_lr3e7.sh \
    > v36_masked_gae_rewscale_kl30_lr3e7.log 2>&1 &
echo "v36 PID: $!"
```

---

## 18. v37/v38 实验分析：fast cosine + 6 task types + centering OOD（2026-07-05）

### 18.1 背景：v35/v36 之后的演进

v35（masked_gae + rewscale + KL=0.30）是截至 v36 的历史最优配置，peak ID_m4=0.5833 @step150，但 step 300 时已衰退到 0.4167。

v37 和 v38 在此基础上做了两项关键改进：

| 实验 | 核心改动 | 科学问题 |
|------|---------|---------|
| **v37** | fast cosine（400步，min_lr=0.05）| cosine LR 能否防止 v35 式的后期 step300 崩跌？ |
| **v38** | v37 基础上移除 centering 训练数据，用 centering 做 OOD 零样本评估 | centering 数据稀少（0.5%）是否影响训练质量？移除后其他任务是否更好？ |

**v37 关键结论**（背景）：fast cosine 成功防止了崩跌，ID_m4 在 step200 达峰 0.583，step250-350 稳定在 0.53-0.57（而 v35 step300 跌至 0.417）。同时发现并修复了 task_type 推断 bug（fov_inclusion 被误判为 absolute_positioning）。

---

### 18.2 v38 实验配置

| 参数 | 值 | 说明 |
|------|-----|-----|
| 算法 | masked_gae | |
| LR | 5e-7 cosine，min_lr_ratio=0.05 | fast cosine，min LR=2.5e-8 |
| KL_LOSS_COEF | 0.30 | 沿用 v33/v35 |
| ENV | env_config_v38_6types.yaml | **6 types，无 centering** |
| 训练集 | 11139 envs | |
| ID val | 18 envs（3/type × 6 types，无 centering）| |
| OOD val | 25 envs（含 centering×6 零样本）| ★ centering 作为 OOD |
| TOTAL_STEPS | 400 | |
| WINDOW_SIZE | 1（no_concat）| 每轮仅看最新图像 |

**训练数据分布（6 类均衡）**：

| task_type | 数量 | 占比 |
|-----------|------|------|
| absolute_positioning | 2003 | 18.0% |
| delta_control | 2003 | 18.0% |
| projective_relations | 2003 | 18.0% |
| occlusion_alignment | 1875 | 16.8% |
| equidistance | 1770 | 15.9% |
| fov_inclusion | 1503 | 13.5% |

---

### 18.3 验证指标曲线（step 0→350，job 到期后停止于 ~step368）

| step | ID_b4 | ID_m4 | ID_w4 | OOD_b4 | OOD_m4 | OOD_w4 |
|------|-------|-------|-------|--------|--------|--------|
| 0    | 0.762 | 0.405 | 0.000 | 0.474 | 0.355 | 0.158 |
| 50   | 0.762 | 0.452 | 0.143 | 0.474 | 0.276 | 0.053 |
| 100  | 0.762 | 0.476 | 0.143 | 0.474 | 0.290 | 0.158 |
| 150  | 0.857 | 0.464 | 0.048 | 0.579 | 0.329 | 0.158 |
| 200  | **0.857** | **0.488** | 0.095 | 0.579 | 0.276 | 0.053 |
| 250  | 0.714 | 0.369 | 0.095 | 0.474 | 0.184 | 0.000 | ← grad_norm 尖峰 |
| 300  | 0.762 | 0.488 | 0.143 | 0.632 | 0.368 | 0.053 |
| 350  | 0.810 | **0.488** | 0.095 | **0.684** | **0.368** | 0.105 |

峰值 ID_m4=**0.488**（step200 起进入平台，未见衰退）；OOD_b4 在 step350 仍在改善（0.684↗），说明模型的 OOD 泛化尚未饱和。

---

### 18.4 训练稳定性

| step | entropy | grad_norm | LR | t_succ% |
|------|---------|-----------|-----|---------|
| 100  | 0.631 | 8.14 | 4.88e-7 | 4.70% |
| 200  | 0.701 | 5.19 | 3.70e-7 | 3.72% |
| **250** | 0.667 | **17.13** ⚠ | 2.81e-7 | 3.07% |
| 300  | 0.637 | 5.05 | 1.89e-7 | 6.42% |
| 350  | 0.735 | 4.89 | 1.08e-7 | 6.37% |

**entropy 全程 < 1.0（最高仅 0.735）**：v38 是迄今运行最稳定的 masked_gae 实验，无熵爆炸。step250 的 `grad_norm=17.1` 是一次批次级冲击（导致 ID_m4 骤降至 0.369），随后恢复，不属于不可逆失稳。cosine LR 至 step350 已降至 1.08e-7，后段梯度持续收敛。

---

### 18.5 行为多样性

act_H 在 0.765–0.971 之间**震荡而非单向下滑**，move_forward 占比 56–72% 交替，**未出现 action concentration**。与 v32/v26 的持续集中化（act_H 跌破 0.6，move_forward >80%）形成对比，说明 fast cosine + KL=0.30 组合有效维持了探索多样性。

---

### 18.6 Per-task 成功率学习曲线

| step | abs_pos | **delta** | equidist | proj_rel | occlusion | fov_incl |
|------|---------|-----------|----------|----------|-----------|---------|
| 0    | 50% | **0%** | 67% | 100% | 100% | 67% |
| 100  | 100% | **0%** | 33% | 100% | 80% | 100% |
| 200  | 100% | **0%** | 67% | 100% | 100% | 100% |
| 250  | 50% | 50% | 33% | 100% | 60% | 100% |
| 350  | 100% | **0%** | 67% | 100% | 80% | 100% |

**delta_control 全程接近 0%**（仅 step250 偶发 50%），是 v38 最主要的短板。

---

### 18.7 delta_control 失败原因深度分析

所有 8 条 delta_control val 轨迹（2 envs × val_n=4）均 succ=0.0，rewards 从 −1.23 到 +0.19。任务示例："Move the camera to the **closer** view of the toilet, about **0.69 meters** away"。

**典型失败轨迹（最优样本，reward=0.186）**：
```
Turn 1: <think>Initially I should first rotate to face the toilet...</think>
        move_right|turn_left|move_forward|turn_left|move_right|
Turn 2: <think>I've rotated... I need to translate forward...</think>
        move_forward|
Turn 3: <think>After several steps, should check...</think>
        turn_left|move_right|move_forward|
Turn 4: <think>I should confirm the toilet is close enough...</think>
        move_forward|
→ 所有 4 turn 用尽，始终未停在 0.69m 处
```

**根本失败原因（从数据实证）**：

| 原因 | 说明 | 权重 |
|------|------|------|
| **公制精度要求（主因）** | 任务要求到达指定米级距离（0.69m / 1.41m）。"front view" 有视觉方向对齐信号，而"0.69 meters away"需要从单张 RGB 估计绝对深度——模型无此能力。| ★★★ |
| **WINDOW_SIZE=1 无历史（主因）** | no_concat 模式每轮仅看当前图像，无法通过步数累计估算已移动多少距离，没有里程计意识，无法判断"现在到了 0.69m，停"。| ★★★ |
| **奖励梯度方向陷阱（次因）** | potential field 对"接近目标"持续给正奖励，但成功要求精确停在目标距离。模型学到"一直前进=更多奖励"，entry 7（reward=−1.23）连续 3 步 move_forward 后超调离目标更远。| ★★ |
| **Val 样本极少（次因）** | ID val 中 delta_control 仅 2 个 env（n=2），梯度信号微弱，随机性大。| ★★ |

**与其他任务的对比**：absolute_positioning / projective_relations / fov_inclusion 的成功标准均可从单张图像直接判断；delta_control 唯一需要公制深度估计，根本上超出了单图 + 无历史的信息界限。

**改进建议**：
1. 在每轮 prompt 中注入"当前相机位置 [tx, ty, tz]"信息（环境已有此数据）
2. 增大 WINDOW_SIZE 以允许推断累积位移
3. 增大 delta_control 的 val 样本数（至少 5 env）

---

### 18.8 失败模式分布（step 350，best-of-4，共 21 环境组）

| Mode | Count | Pct | 说明 |
|------|-------|-----|------|
| success_fast | 7 | 33.3% | 成功，≤2 turn |
| success_std | 10 | 47.6% | 成功，>2 turns |
| near_miss | 2 | 9.5% | 失败但接近成功（delta_control × 1，equidistance × 1）|
| pos_progress | 1 | 4.8% | 失败，方向正确（delta_control）|
| neg_progress | 1 | 4.8% | 失败，远离目标（occlusion_alignment）|
| format_error | 0 | 0.0% | 无格式崩溃 |

**整体成功率 80.9%**（17/21），格式完全健康。

---

### 18.9 v38 vs v35/v37 对比

| 指标 | v35 (rewscale, 5 types) | v37 (fast cosine, 5 types) | **v38 (6 types, OOD centering)** |
|------|------------------------|--------------------------|----------------------------------|
| 峰值 ID_m4 | 0.583 @step100-150 | 0.583 @step200 | **0.488** @step200+ |
| step300 ID_m4 | 0.417（崩跌）| **0.536（稳定）** | **0.488（平台）** |
| entropy 最大 | ~0.8 | ~0.8 | **0.735（最低）** |
| 后期稳定性 | 崩跌 | 稳定 | **最稳定（370步无崩塌）** |
| delta_control | 混入 5-type，不独立 | 混入 5-type | **0%（独立暴露，硬性瓶颈）** |
| OOD_b4 step350 | 0.789 | 0.842（估） | **0.684（低，6-type 分布更难）** |

**结论**：
- v38 绝对 ID_m4 低于 v35/v37，主因是 6-type 分布更难（centering 被移除，delta_control 0% 拖低整体）
- **后期稳定性是历史最优**：370步无崩塌，平台期 ID_m4=0.488
- delta_control 的持续 0% 是独立硬性瓶颈，需要环境信息增强（位置注入/窗口扩大）才能解决
- centering 零样本 OOD 测试结果将由续训 step400 val 给出（job 到期前尚无 OOD_b4 单独统计）

---

### 18.10 v38 续训命令

v38 在 step~368 因 job 到期中止（最新 val checkpoint: step350）。续训脚本已创建：

```bash
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v38_masked_gae_6types_fastcosine_resume.sh \
    > v38_masked_gae_6types_fastcosine_resume.log 2>&1 &
echo "PID: $!"
```

TOTAL_STEPS=400，续训从 step~368 开始仅剩约 32 步。主要目标：
1. 获取 step400 最终 val，确认 OOD_b4 是否继续改善（step350=0.684↗）
2. 观察是否在 step400 前出现后期崩塌

---

### 18.11 实验状态总览（2026-07-05 更新）

| 实验 | 状态 | 最新步 | 峰值 ID_m4 | 备注 |
|------|------|--------|-----------|------|
| v26_klhi_lr5e7 | 已完成 | ~241 | **0.607** (@150) | Qwen ID_m4 历史最高 |
| v28_cosine | 已完成 | ~399 | 0.5595 (@200) | ID_b4 历史最高（0.905） |
| v33 (KL=0.30) | 已完成 | 151 | 0.5595 (@150↗) | GRPO 最优 |
| v35 (masked+kl30) | 已完成 | ~349 | 0.583 (@100-150) | 后期崩跌 |
| v37 (fast cosine) | 已完成 | ~350+ | 0.583 (@200) | fast cosine 防崩跌 ✓ |
| **v38 (6 types, OOD centering)** | **待续训** | ~368 | **0.488** (平台) | 370步最稳定；delta=0% 硬瓶颈 |

---

## 19. v40 进展分析与 v41 评估构成消融设计（2026-07-08）

### 19.1 v40 的两个工程修复状态

v40 的原始目标是继续沿着 v39 的方向验证：

1. `distance_in_obs` 是否真的进入模型观测
2. 增大 `WINDOW_SIZE=3` 是否能缓解 delta_control 的“无里程计”问题

在 2026-07-08 的复查中，确认了两个关键工程事实：

#### 1) distance injection 已真正生效

对 v40 validation JSONL 的逐条检查显示：

- `total=516`
- `with_distance=516`

即每条 validation 轨迹都真实包含 `Distance to target: X.XX m` 文本。

这与 v39 的失败状态形成鲜明对比：

- v39: `with_distance=0`
- v40: `with_distance=516/516`

根因修复点：
- `env.py` 中 `_get_target_pose()` 增强，支持从 `sample_target` / `target_region.sample_point` 等字段恢复目标位姿
- `_build_observation_from_image()` 也补上了 `dist_suffix`，不再只在 `_render()` 路径注入距离信息

#### 2) delta boost 文件生成成功，但消费链路最初有 bug

v40 启动时已正确生成：

- `val_id_delta_boost.jsonl`，共 18 条 ID val 任务
- 其中 `delta_control=6`

但 v40 早期 validation 输出中，实际只出现了 `3` 组 delta（12 entries），而非预期的 `6` 组。

最终定位为：

- 不是 boost 文件内容错
- 而是 **override JSONL 仍通过 `seed: [0, n]` 走随机索引语义**，没有按文件顺序逐条消费

修复方案：

1. `gym_agent_dataset.py`
  - `seed_list` 长度检查从 `> n_envs` 改为 `>= n_envs`
2. `run_experiment.sh`
  - 对 ID override 场景注入显式 `seed_list=[0,1,...,n-1]`
  - 从而让 override JSONL 被按行顺序确定性消费，而不是重新随机采样

这意味着：

- **旧 v40 结果中的 delta 统计仍受该 bug 影响**
- **之后重新启动的 v40 / v41 才是修复后可比较版本**

---

### 19.2 v40 当前实验进展（基于修复前这轮运行的最新结果）

截至最新日志，v40 已训练到：

- `training/global_step = 134`

最新训练指标：

| step | entropy | t_succ% | grad_norm | LR |
|------|---------|---------|-----------|----|
| 134 | 0.597 | 6.08% | 4.73 | 4.87e-7 |

说明：

- 训练**整体健康**，没有 v26/v28 式的 entropy 爆炸
- `grad_norm` 处于正常范围
- `train/traj_success` 与历史稳定实验处于可比区间

#### v40 validation 曲线（修复前这一轮）

| step | ID_b4 | ID_m4 | ID_w4 | OOD_b4 | OOD_m4 | OOD_w4 |
|------|-------|-------|-------|--------|--------|--------|
| 0    | 0.8889 | 0.4722 | 0.1111 | 0.5200 | 0.2800 | 0.0400 |
| 50   | 0.7778 | 0.4583 | 0.2222 | 0.5600 | 0.3000 | 0.0800 |
| 100  | 0.7778 | 0.4444 | 0.2222 | 0.5600 | 0.3100 | 0.0800 |

#### 训练动态

| step | entropy | t_succ% | crit_sc | resp_clip | grad_nrm | lr |
|------|---------|---------|--------|----------|---------|----|
| 60   | 0.511 | 5.22 | 0.380 | 0.176 | 5.404 | 5.00e-7 |
| 80   | 0.528 | 5.32 | 0.363 | 0.199 | 9.255 | 4.99e-7 |
| 100  | 0.546 | 3.10 | 0.227 | 0.200 | 4.170 | 4.96e-7 |
| 120  | 0.532 | 4.95 | 0.332 | 0.221 | 4.733 | 4.91e-7 |

#### 行为多样性

| step | act_H | dom% | dominant |
|------|------|------|----------|
| 1    | 0.842 | 66.9% | move_forward |
| 20   | 0.939 | 60.9% | move_forward |
| 60   | 0.903 | 63.5% | move_forward |
| 100  | 0.729 | 77.1% | move_forward |
| 120  | 0.960 | 59.3% | move_forward |
| 134  | 0.871 | 63.4% | move_forward |

结论：

- v40 **没有出现持续性的 action concentration**
- step100 有一次暂时的 move_forward 集中化（77.1%），但 step120 已恢复
- 因此当前问题不是“策略崩了”，而是 delta_control 仍未稳定学会

---

### 19.3 v40 的核心现象：distance 信号到了，但 delta_control 仍没学稳

按修复前这一轮 v40 的 validation 输出统计：

| step | delta 成功率 |
|------|-------------|
| 0    | 33.3% |
| 50   | 33.3% |
| 100  | 0.0% |

失败模式（step100）显示：

- `success_fast = 12/18`
- `success_std = 2/18`
- 剩余失败中，delta_control 占绝大多数：
  - `pos_progress = 2`
  - `no_progress = 1`

这说明：

1. distance 文本已经进到了模型输入
2. 模型也不是完全瞎走，因为 delta 常表现为 `pos_progress`
3. 但它还没把“距离信号”学成稳定的 stopping / step-count control

换句话说，**distance injection 解决了“看不到距离”的问题，但尚未解决“学会基于距离停下”的问题**。

---

### 19.4 当前认识更新

到 v40 为止，可以把 delta_control 的瓶颈拆成两个层次：

#### 层次 A：观测信息是否足够？

- v38: 不足（无 distance, window=1）
- v39: 配置想要加 distance，但链路未生效
- v40: **distance 已真正可见，window=3 也已给到**

因此“信息完全不够”这个解释已经被部分削弱。

#### 层次 B：模型是否会利用这些信息？

v40 的结果表明：

- 即使信息显式提供，delta_control 仍可能学不会稳定 stopping
- 模型可能更容易把 distance 用作“继续朝目标走”的 shaping 线索，
  但还不会把它转化成“该停了”的决策边界

这意味着后续若继续追 delta，可能需要：

1. 更强的 stopping inductive bias
2. 更明确的任务提示（例如强调“达到目标距离后停止/不要过冲”）
3. 更有针对性的 val /训练消融，而不仅是继续加一般性上下文

---

### 19.5 v41 设计：测试“整体 ID 指标是否主要被 delta_control 拖累”

为了区分：

- “v40 整体变差，是因为所有 task 都学得差”
- 还是“其他 task 正常，只是 delta_control 单项严重拖累 aggregate ID_m4”

设计了 v41：

#### v41 核心思路

- **训练配置与 v40 完全一致**
  - 仍训练 6 types
  - 仍使用 `distance_in_obs=true`
  - 仍使用 `WINDOW_SIZE=3`
  - 仍使用 fast cosine + masked_gae + KL=0.30
- **唯一变化：ID val 排除 delta_control**
  - 用其他 task type 回填到同样的 18 个 ID env

#### 科学问题

若 v41 的 ID_m4 明显高于 v40，则说明：

> v40 的 aggregate ID 指标主要是被 delta_control 没学起来拖累的。

若 v41 仍不高，则说明：

> 问题不止在 delta_control，WINDOW_SIZE=3 / distance 方案本身对整体也没有带来收益。

#### 重要说明

v41 是 **评估构成消融**，不是训练分布消融。

它回答的是：

> “delta_control 有没有在拖累总体指标？”

它**不回答**：

> “如果训练时彻底去掉 delta_control，模型会不会更强？”

要回答后者，需要单独再做一个“train no-delta”实验。

#### v41 脚本

已新增：

- `examples/train/active_spatial/experiments/v41_masked_gae_6types_distinfo_w3_nodeltaidval.sh`

关键设置：

- `ID_VAL_EXCLUDE_TASK_TYPES="delta_control"`

---

### 19.6 当前建议（2026-07-08）

1. **重新启动 v40**
  - 用修复后的 `seed_list` override 链路，拿到“真正 6 个 delta 组”的结果

2. **并行启动 v41**
  - 用来快速验证 aggregate ID 指标是否主要被 delta 拖累

3. **分析时区分两类问题**
  - `v40`: 测 delta 是否真的学起来
  - `v41`: 测 delta 是否在拖累总体指标

这样可以把“任务本身难”与“总体指标是否被单项拉低”两个问题彻底拆开。
