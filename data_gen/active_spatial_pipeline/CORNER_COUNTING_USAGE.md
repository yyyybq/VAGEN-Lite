# 可见角点计数 (Visible Corner Counting) 的使用流程详解

## 📋 概述

可见角点计数功能通过统计目标物体的 8 个角点中有多少在图像内可见，来精确判断物体的可见性质量。这个信息在相机位置筛选流程中起到关键的**过滤作用**。

---

## 🔄 完整使用流程

### 1️⃣ 函数调用层级

```
sample_camera_for_pair()           # 相机采样主函数
    ↓
check_visible_corners_count()      # 类方法（封装接口）
    ↓
count_visible_corners()            # 核心函数（独立实现）
    ↓
返回: (has_enough_corners, visible_count)
```

---

## 🎯 核心函数: `count_visible_corners()`

**位置**: `camera_sampler.py`, Line ~552

**功能**: 统计 AABB 的 8 个角点中有多少在图像内可见

**算法流程**:

```python
def count_visible_corners(...) -> int:
    visible_count = 0
    
    # 遍历 8 个角点
    for corner_world in corners:
        # Step 1: 转换到相机坐标系
        pc = world_to_camera(viewmat, corner_world)
        
        # Step 2: 检查是否在相机前方 (z > 0)
        if pc[2] <= 1e-6:
            continue  # 在相机后面，跳过
        
        # Step 3: 投影到图像平面
        u, v, z = project_point_to_image(K, pc)
        
        # Step 4: 检查是否在图像边界内
        if not point_in_image_bounds(u, v, width, height, border=2):
            continue  # 投影在图像外，跳过
        
        # Step 5: (可选) 检查是否被遮挡
        if check_occlusion and occluders is not None:
            if is_point_occluded_by_aabb(cam_pos, corner_world, occluders):
                continue  # 被遮挡，跳过
        
        # 通过所有检查，计数 +1
        visible_count += 1
    
    return visible_count  # 返回 0-8
```

**输出**: 整数 0-8，表示可见角点的数量

---

## 🔌 封装接口: `check_visible_corners_count()`

**位置**: `CameraSampler` 类方法, Line ~1214

**功能**: 封装配置传递，提供高层判断接口

```python
def check_visible_corners_count(self, ..., min_corners=1) -> Tuple[bool, int]:
    # 调用核心函数
    visible_count = count_visible_corners(
        self.intrinsics,  # 自动传递相机内参
        cam_pos, cam_target, target_bmin, target_bmax,
        self.config.image_width,  # 自动传递配置
        self.config.image_height,
        border=2,
        check_occlusion=check_occlusion,
        occluders=occluders,
        target_id=target_id
    )
    
    # 判断是否满足最小角点要求
    has_enough = visible_count >= min_corners
    
    # 返回判断结果 + 实际数量
    return has_enough, visible_count
```

**输出**: 
- `has_enough_corners` (bool): 是否满足最小角点要求
- `visible_count` (int): 实际可见角点数量

---

## 📊 在采样流程中的使用

### 位置: `sample_camera_for_pair()`, Line ~1973

```python
# ========== Enhanced Visibility Checks ==========

# 1. 调用角点检查（物体 A）
has_enough_corners_a, corner_count_a = self.check_visible_corners_count(
    cam_pos, cam_target, a_min, a_max, 
    min_corners=1,          # 要求至少 1 个角点可见
    check_occlusion=False   # 快速检查，暂不考虑遮挡
)

# 2. 调用角点检查（物体 B）
has_enough_corners_b, corner_count_b = self.check_visible_corners_count(
    cam_pos, cam_target, b_min, b_max, 
    min_corners=1,
    check_occlusion=False
)

# 3. 使用返回的信息进行判断
if not has_enough_corners_a:
    # 物体 A 角点不足，记录拒绝原因（包含实际数量）
    rejection_stats[f'obj_a_insufficient_corners_{corner_count_a}'] = \
        rejection_stats.get(f'obj_a_insufficient_corners_{corner_count_a}', 0) + 1

if not has_enough_corners_b:
    # 物体 B 角点不足，记录拒绝原因
    rejection_stats[f'obj_b_insufficient_corners_{corner_count_b}'] = \
        rejection_stats.get(f'obj_b_insufficient_corners_{corner_count_b}', 0) + 1

# 4. 双重判断：两个物体都必须满足角点要求
if not (has_enough_corners_a and has_enough_corners_b):
    continue  # ❌ 拒绝这个相机位置，继续尝试下一个

# 5. ✅ 通过角点检查，继续后续检查（面积、遮挡等）
...
```

---

## 💡 返回信息的两种用途

### 用途 1: 布尔判断 (`has_enough_corners`)

**目的**: 决定是否接受该相机位置

```python
if not (has_enough_corners_a and has_enough_corners_b):
    continue  # 拒绝
```

**逻辑**: AND 操作，两个物体都必须满足最小角点要求

---

### 用途 2: 统计信息 (`corner_count`)

**目的**: 记录详细的拒绝原因，用于调试和统计分析

```python
rejection_stats[f'obj_a_insufficient_corners_{corner_count_a}'] = ...
```

**示例统计输出**:
```
拒绝原因统计:
  obj_a_insufficient_corners_0: 45次  # 物体 A 有 0 个角点可见
  obj_a_insufficient_corners_1: 12次  # 物体 A 有 1 个角点可见（不满足 min=2）
  obj_b_insufficient_corners_0: 38次
  ...
```

**价值**:
- 帮助理解为什么没有找到足够的有效相机位置
- 指导参数调优（如是否需要放宽 `min_corners` 要求）
- 分析场景特性（如某些物体是否太小或位置不佳）

---

## 🔍 详细执行时序图

```
时间 ──────────────────────────────────────────────────────────────►

     ┌─────────────────────────────────────────────────┐
     │ 1. FOV 基础检查 (快速过滤)                        │
     │    - 检查是否有任意角点在图像内                    │
     └─────────────────────────────────────────────────┘
                        ↓ PASS
     ┌─────────────────────────────────────────────────┐
     │ 2. 可见角点计数 (精细判断) ⭐ 我们的功能           │
     │    - 统计具体有几个角点可见 (0-8)                 │
     │    - 判断是否 >= min_corners                      │
     └─────────────────────────────────────────────────┘
                        ↓ PASS
     ┌─────────────────────────────────────────────────┐
     │ 3. 投影面积检查                                   │
     │    - 计算物体投影占图像的面积比例                  │
     └─────────────────────────────────────────────────┘
                        ↓ PASS
     ┌─────────────────────────────────────────────────┐
     │ 4. 2D 遮挡检查                                    │
     │    - 像素级遮挡计算                               │
     └─────────────────────────────────────────────────┘
                        ↓ PASS
     ┌─────────────────────────────────────────────────┐
     │ ✅ 相机位置有效，添加到 valid_poses              │
     └─────────────────────────────────────────────────┘
```

---

## 📐 可见角点数量的含义

| 可见角点数 | 含义 | 视角质量 |
|-----------|------|---------|
| 0 | 物体完全在图像外或在相机后面 | ❌ 不可见 |
| 1 | 仅一个角点可见 | ⚠️ 极差（可能仅边缘可见） |
| 2-3 | 部分角点可见 | 🔶 一般（物体部分可见） |
| 4-5 | 多数角点可见 | ✅ 良好（物体大部分可见） |
| 6-7 | 几乎所有角点可见 | ✅✅ 很好 |
| 8 | 所有角点都可见 | ✅✅✅ 极好（正面完整视角） |

**默认阈值**: `min_corners=1`
- 宽松要求，只要有角点可见即可
- 适合初步筛选

**建议阈值**: `min_corners=2` 或 `min_corners=3`
- 更严格要求，确保物体有足够的覆盖
- 适合高质量数据生成

---

## 🎛️ 配置参数影响

### 参数 1: `min_corners`

**位置**: 函数参数（默认 1）或配置 `config.min_visible_corners`

**影响**:
```python
min_corners=1  # 宽松：至少 1 个角点 → 更多有效位置
min_corners=2  # 平衡：至少 2 个角点 → 合理筛选
min_corners=3  # 严格：至少 3 个角点 → 高质量但位置较少
min_corners=4  # 极严格：至少 4 个角点 → 最高质量但可能找不到位置
```

### 参数 2: `check_occlusion`

**位置**: 函数参数（默认 False）

**影响**:
```python
check_occlusion=False  # 快速检查，不考虑角点是否被遮挡
                       # 用于初步筛选，后续有专门的遮挡检查

check_occlusion=True   # 慢速但精确，每个角点都检查遮挡
                       # 可能导致过度筛选（后续还有 2D 遮挡检查）
```

**建议**: 保持 `False`，让后续的 2D 遮挡检查来处理遮挡问题（更准确）

---

## 🆚 与 FOV 检查的区别

### FOV 检查 (`check_target_in_fov`)
- **检查**: 是否有**任意**角点在图像内
- **返回**: 布尔值（是/否）
- **精度**: 粗略，只知道"可见"或"不可见"

### 角点计数 (`check_visible_corners_count`)
- **检查**: 有**多少个**角点在图像内
- **返回**: 布尔值 + 数量（如 True, 5）
- **精度**: 精细，提供定量的可见性度量

**为什么需要两者？**
1. FOV 检查更快，用于初步过滤
2. 角点计数提供更精确的可见性评估
3. 角点计数可以设置最小要求（如 ≥2 个）

---

## 🔧 实际使用场景

### 场景 1: 默认使用（推荐）

```python
has_enough, count = sampler.check_visible_corners_count(
    cam_pos, cam_target, obj_min, obj_max,
    min_corners=1,          # 至少 1 个角点
    check_occlusion=False   # 不检查遮挡（更快）
)

if not has_enough:
    print(f"拒绝：只有 {count} 个角点可见")
    continue
```

### 场景 2: 高质量模式

```python
has_enough, count = sampler.check_visible_corners_count(
    cam_pos, cam_target, obj_min, obj_max,
    min_corners=3,          # 至少 3 个角点（更严格）
    check_occlusion=True    # 检查角点遮挡
)

if not has_enough:
    print(f"拒绝：只有 {count}/3 个角点可见且未被遮挡")
    continue
```

### 场景 3: 诊断模式（收集统计）

```python
_, count = sampler.check_visible_corners_count(
    cam_pos, cam_target, obj_min, obj_max,
    min_corners=0  # 不设限制，只收集信息
)

corner_distribution[count] = corner_distribution.get(count, 0) + 1
# 统计结果: {0: 45, 1: 23, 2: 67, 3: 89, ...}
```

---

## 📊 与其他检查的协同作用

```
角点计数 → 投影面积 → 2D 遮挡
   ↓           ↓           ↓
 几何可见   尺寸合适    无遮挡
```

**协同效果**:
1. **角点计数**: 确保物体在视野内有足够的几何覆盖
2. **投影面积**: 确保物体投影面积足够大（不太远）
3. **2D 遮挡**: 确保可见部分没有被大量遮挡

**为什么分开检查？**
- 角点可见 ≠ 面积足够（可能太远，角点可见但投影很小）
- 面积足够 ≠ 无遮挡（投影大但可能被前景物体遮挡）
- 三者结合才能保证高质量的相机视角

---

## 💻 调试技巧

### 查看角点分布

```python
corner_counts = []
for pose in candidate_poses:
    _, count = sampler.check_visible_corners_count(
        pose.position, pose.target, obj_min, obj_max
    )
    corner_counts.append(count)

print(f"角点数量分布: {Counter(corner_counts)}")
# 输出: Counter({4: 15, 5: 12, 3: 8, 6: 5, 7: 2})
```

### 可视化角点

```python
from camera_sampler import aabb_corners

corners = aabb_corners(obj_min, obj_max)
for i, corner in enumerate(corners):
    u, v, z = project_and_check(corner)
    if z > 0 and in_bounds(u, v):
        print(f"Corner {i}: VISIBLE at ({u:.1f}, {v:.1f})")
    else:
        print(f"Corner {i}: HIDDEN")
```

---

## 🎯 总结

**可见角点计数的核心价值**:

1. **定量评估**: 提供 0-8 的精确可见性度量
2. **灵活阈值**: 可以根据需求调整 `min_corners`
3. **诊断工具**: 通过 `corner_count` 了解拒绝原因
4. **渐进式过滤**: 在 FOV 检查之后、面积检查之前
5. **高效实现**: 轻量级计算，对性能影响极小

**使用建议**:
- ✅ 始终启用（默认参数即可）
- ✅ 使用 `min_corners=1` 进行快速筛选
- ✅ 高质量数据生成时使用 `min_corners=2-3`
- ✅ 利用 `corner_count` 进行统计分析
- ⚠️ 避免 `check_occlusion=True`（让后续 2D 遮挡处理）
