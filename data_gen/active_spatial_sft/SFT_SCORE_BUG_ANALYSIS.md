# SFT数据质量问题根因分析与修复记录

**日期**: 2025年5月  
**作者**: 实验迭代记录  
**背景**: 在 `output_100scenes_5k/` 中生成的 3096 条SFT轨迹里，仅有 54.1% 的score ≥ 0.95（预期应接近100%）。本文档记录了排查过程、发现的所有bug，以及修复方案。

---

## 一、问题描述

### 现象

| 数据文件 | 总条数 | score ≥ 0.95 | 0.5 ≤ score < 0.95 |
|---------|--------|-------------|-------------------|
| sft_data.jsonl (Part1) | 1362 | 744 (54.6%) | 618 (45.4%) |
| sft_data_part2.jsonl (Part2) | 1734 | 931 (53.7%) | 803 (46.3%) |
| **合计** | **3096** | **1675 (54.1%)** | **1421 (45.9%)** |

### 期望

SFT数据使用oracle路径规划器（`path_finder.py`）生成，理论上每一条轨迹都应以满足 score ≥ 0.95 的姿态结束。预期成功率应接近 **100%**，而非 54%。

---

## 二、核心验证方法

首先验证：若将相机直接摆放在 `sample_point` 朝向 `sample_forward`（即pipeline提供的ground truth姿态），scoring函数是否能给出 ≥ 0.95 的分数？

**修复前测试结果（oracle pose scoring）**：

| 任务类型 | 样本数 | ≥ 0.95 比率 |
|---------|--------|------------|
| delta_control | 350 | **0%（无一达标）** |
| fov_inclusion | 266 | 66% |
| size_distance_invariance | 147 | 47% |
| 其余6种任务 | 4237 | ~100% |
| **总计** | **5000** | **~70%** |

这说明：即使直接在答案位置，scoring函数也无法给出高分。根因在scoring函数和path_finder本身，而非轨迹规划能力不足。

---

## 三、发现的Bug（共5个）

### Bug 1：`path_finder.py` — `_extract_target_pose` 忽略了 `sample_forward`

**文件**：`data_gen/active_spatial_sft/path_finder.py`

**问题**：
`_extract_target_pose()` 函数用于提取目标位置和目标朝向，其查找优先级为：
1. 在 `params` 中寻找 `object_center` / `center` / `pivot_point` / `reference_point`
2. ⛔ **直接fallback到"保持初始相机朝向"**

对于 **delta_control、equidistance、projective_relations、occlusion_alignment** 这四类任务，`params` 中不存在 `object_center`，因此直接落入fallback。而这些任务的正确目标朝向存在于 `target_region["sample_forward"]` 中，但代码完全没有读取。

**影响**：Phase 2（`_plan_to_orientation`）尝试对齐的目标方向是初始相机朝向，而非真实目标。初始朝向与正确朝向之间可能相差 150–180°。这导致：
- Phase 2 浪费 ~10 个动作把相机转到错误方向
- Phase 3（beam search）再花 25+ 个动作试图纠正，60步预算耗尽
- 最终分数停在 0.6–0.7 左右

**修复**：在 `for key in ("object_center", ...)` 循环之后、最终fallback之前，添加第二优先级：

```python
# Priority 2: sample_forward stored in target_region — exact ground truth
sample_fwd = target_region.get("sample_forward", None)
if sample_fwd is not None:
    sf = np.array(sample_fwd, dtype=float)
    sf_norm = np.linalg.norm(sf)
    if sf_norm > 1e-6:
        return target_pos, sf / sf_norm
```

使用完整3D向量（含pitch分量），使 `_plan_to_orientation` 能同时修正水平偏角和俯仰角。

---

### Bug 2：`spatial_potential_field.py` — `_score_delta_control` FoV检查使用了 `target_point`

**文件**：`vagen/envs/active_spatial/spatial_potential_field.py`

**问题**：
```python
object_center = np.array(params.get("object_center", target_point))
```
delta_control 的 `params` 中从不存储 `object_center`（只有 `delta`、`start_position` 等），因此始终fallback到 `target_point`（相机目标位置）作为FoV检查对象。

这产生了一个**不可逾越的分数上界**：
- 相机距离目标 0.264m 时，朝向移动方向，score ≈ 0.877（已接近上界）
- 相机再向前移动一步（0.3m）**越过** target_point：target_point 跑到相机身后
- FoV检查触发惩罚底线 `FOV_PENALTY_FLOOR = 0.3`
- score 骤降至 ~0.707

数学验证：
```
pos_w = 0.677 (pos_score=0.768时的动态权重)
ori_w = 0.323
total = 0.677 × 0.768 + 0.323 × 1.0 + 0.03(lookat) = 0.866
```
与观测值 0.8665 吻合。路径规划器的beam search正确地避免了"越过"这一步，但同时也永远无法突破这个局部最优。

**修复**：当 `object_center` 不存在时，在 `sample_forward` 方向前方 5m 处生成一个合成目标点：

```python
object_center = target_point + (fwd / fwd_n) * 5.0
```

该点始终在相机正前方（当相机朝向 `sample_forward` 时），消除了分数屏障。

---

### Bug 3：`spatial_potential_field.py` — `_score_delta_control` 朝向目标方向推断错误

**文件**：`vagen/envs/active_spatial/spatial_potential_field.py`

**问题**：
```python
if delta > 0:
    object_dir = movement_dir  # 面向移动方向
else:
    object_dir = -movement_dir
```

该逻辑假设 `delta > 0` 时相机应朝向移动方向。但数据统计显示：

| 方向关系 | 数量 | 比例 |
|---------|------|------|
| sample_forward 与移动方向相同 | 294 | **84%** |
| sample_forward 与移动方向相反 | 56 | **16%** |

**16%的delta_control任务**（相机需要移动到某位置但朝向物体，而物体在出发点一侧）的朝向目标被推断为完全错误的方向（180°偏差）。

**修复**：直接使用 `target_region["sample_forward"]` 的XY分量作为朝向目标，降级为原始推断作为fallback：

```python
sample_fwd_raw = target_region.get("sample_forward", None)
if sample_fwd_raw is not None:
    sf_xy = normalize(np.array([sf[0], sf[1], 0.0])[:2])
    object_dir = sf_xy
```

---

### Bug 4：`spatial_potential_field.py` — `_score_fov_inclusion` 朝向公式惩罚合理位置

**文件**：`vagen/envs/active_spatial/spatial_potential_field.py`

**问题**：
```python
# 原始公式
orientation_score = 1.0 - (max_angle / fov_rad) * 0.3
```

`fov_inclusion` 任务的 `fov_horizontal = 110°`（半角 55°）。当两个目标物体分别位于 ±25°（完全在视野内），此公式给出：
```
orientation_score = 1.0 - (25/55) × 0.3 = 0.864
```

即使相机完美地把两个物体框在视野中，仍会被扣分。此外，原公式使用**Z=0的2D中点方向**，当相机有俯仰角时，方向与实际forward不一致，进一步降低分数。

**修复**：改为用相机朝向与3D中点方向的夹角余弦来计算朝向分：

```python
midpoint_3d = (center_a + center_b) / 2.0
midpoint_dir = midpoint_3d - camera_position  # 完整3D方向
cos_to_mid = dot(normalize(camera_forward), normalize(midpoint_dir))
orientation_score = (cos_to_mid + 1.0) / 2.0
```

在oracle姿态下，相机朝向3D中点，`cos_to_mid ≈ 1.0`，`orientation_score ≈ 1.0`。

---

### Bug 5：`spatial_potential_field.py` — `_score_size_distance_invariance` 使用3D距离但pipeline用2D距离

**文件**：`vagen/envs/active_spatial/spatial_potential_field.py`

**问题**：
```python
dist_a = max(distance_3d(camera_position, center_a), 0.1)
dist_b = max(distance_3d(camera_position, center_b), 0.1)
```

**验证数据**：
```
使用3D距离检验oracle pose是否满足Apollonius圆约束：53/147 (36%) 通过
使用2D距离检验oracle pose是否满足Apollonius圆约束：147/147 (100%) 通过
```

Pipeline生成Apollonius圆时使用的是2D水平距离（忽略高度差），但scoring函数使用3D距离。当两个物体高度差较大时，两者比值相差可达 2×，导致 `position_score` 仅有 0.6–0.7。

**修复**：改用2D水平距离：

```python
dist_a = max(distance_2d(camera_position, center_a), 0.1)
dist_b = max(distance_2d(camera_position, center_b), 0.1)
```

---

## 四、修复效果验证

验证方法：将5000个oracle pose（`sample_point` + `sample_forward`）直接喂给scoring函数，统计 ≥ 0.95 比例。

**修复后结果**：

| 任务类型 | 样本数 | 均值 | ≥ 0.95 比例 |
|---------|--------|------|------------|
| absolute_positioning | 927 | 1.000 | **100%** |
| centering | 10 | 0.999 | **100%** |
| delta_control | 350 | 1.000 | **100%**（原0%）|
| equidistance | 309 | 0.998 | **100%**（原~100%）|
| fov_inclusion | 266 | 0.994 | **99%**（原66%）|
| occlusion_alignment | 332 | 1.000 | **100%** |
| projective_relations | 601 | 0.993 | **100%** |
| screen_occupancy | 2058 | 0.999 | **100%** |
| size_distance_invariance | 147 | 1.000 | **100%**（原47%）|
| **总计** | **5000** | — | **99.9%** |

剩余4个未达标的条目均为 `fov_inclusion` 任务中pipeline生成了垂直角度极端（60°–78°）的pose，属于pipeline数据质量问题，非scoring bug。

---

## 五、RL训练影响评估

以上修复同时影响RL训练奖励函数（`spatial_potential_field.py` 被RL env直接使用）：

| Bug | RL影响 |
|-----|-------|
| Bug 1 | 仅影响SFT path_finder，RL env不受影响 |
| Bug 2 | 修复后delta_control的奖励梯度更平滑，消除了靠近目标时的"惩罚坑" |
| Bug 3 | 修复了16%的delta_control任务的朝向奖励方向，应为正向影响 |
| Bug 4 | fov_inclusion的朝向奖励更合理（面向中点），梯度更清晰 |
| Bug 5 | size_distance_invariance的position reward现在与Apollonius圆一致 |

**建议**：如当前有RL实验（v21等）正在运行，需评估是否重启以使用修复后的reward函数。修复后reward信号应更准确，对训练有正向影响。

---

## 六、现有SFT数据状态与重新生成建议

### 现有数据（已失效）

- **位置**：`output_100scenes_5k/sft_data.jsonl` + `sft_data_part2.jsonl`
- **问题**：由带bug的版本生成，仅54.1%的轨迹达到score ≥ 0.95
- **结论**：**不宜作为GT trajectory使用**，需重新生成

### 重新生成

使用现有脚本 `run_100scenes_5k.sh` 或其resume版本，输入数据 `sampled_5k.jsonl`（5000条，已固定，无需重新采样）。

```bash
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite
bash data_gen/active_spatial_sft/run_100scenes_5k_v2.sh
```

预期新版本效果：
- 接近100%的轨迹达到score ≥ 0.95
- delta_control成功率：~0% → ~90%+
- fov_inclusion成功率：~30% → ~95%+
- size_distance_invariance成功率：~30% → ~95%+

---

## 七、修改文件汇总

| 文件 | 修改类型 | 对应Bug |
|------|---------|--------|
| `data_gen/active_spatial_sft/path_finder.py` | `_extract_target_pose` 添加sample_forward查找 | Bug 1 |
| `vagen/envs/active_spatial/spatial_potential_field.py` | `_score_delta_control` FoV目标改为合成点 | Bug 2 |
| `vagen/envs/active_spatial/spatial_potential_field.py` | `_score_delta_control` 朝向目标改用sample_forward | Bug 3 |
| `vagen/envs/active_spatial/spatial_potential_field.py` | `_score_fov_inclusion` 朝向公式改为3D中点余弦 | Bug 4 |
| `vagen/envs/active_spatial/spatial_potential_field.py` | `_score_size_distance_invariance` 距离改为2D | Bug 5 |
