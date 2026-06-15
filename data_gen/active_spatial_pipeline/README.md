# Active Spatial Perception Dataset Generation Pipeline

This pipeline generates training data for active spatial navigation tasks. It combines object selection, camera pose sampling, and task-based target position generation into a unified workflow.

## Quick Start

### 1. Generate Training Data
```bash
# Generate data for all scenes (default: 'around' move pattern)
cd /scratch/by2593/project/Active_Spatial/VAGEN/data_gen/active_spatial_pipeline
source /scratch/by2593/miniconda3/bin/activate vagen

python run_pipeline.py \
    --scenes_root /scratch/by2593/project/Active_Spatial/InteriorGS \
    --output_dir /scratch/by2593/project/Active_Spatial/VAGEN/data/active_spatial \
    --scene_id 0267_840790 \
    --num_cameras 5 \
    --verbose

# Use 'rotation' pattern: 360° room scanning from room center
python run_pipeline.py \
    --scenes_root /scratch/by2593/project/Active_Spatial/InteriorGS \
    --output_dir /scratch/by2593/project/Active_Spatial/VAGEN/data/active_spatial \
    --scene_id 0267_840790 \
    --move_pattern rotation \
    --rotation_interval 10.0 \
    --verbose

# Use 'linear' pattern: walk past objects (pass_by trajectory)
python run_pipeline.py \
    --scenes_root /scratch/by2593/project/Active_Spatial/InteriorGS \
    --output_dir /scratch/by2593/project/Active_Spatial/VAGEN/data/active_spatial \
    --scene_id 0267_840790 \
    --move_pattern linear \
    --linear_num_steps 7 \
    --linear_move_distance 0.8 \
    --verbose
```

**Move Pattern Options:**
| Pattern | Description |
|---------|-------------|
| `around` | Default. Circle around object horizontally |
| `rotation` | Stand at room center, rotate 360° |
| `linear` | Walk past object (pass_by trajectory) |

### 2. Render Initial and Target Views (NEW)
```bash
# After run_pipeline.py, render the initial camera view and multiple target views
python render_init_target_views.py \
    --jsonl_path /scratch/by2593/project/Active_Spatial/VAGEN/data/active_spatial/train_data.jsonl \
    --output_dir /scratch/by2593/project/Active_Spatial/VAGEN/data/active_spatial/rendered_views \
    --gs_root /scratch/by2593/project/Active_Spatial/InteriorGS \
    --num_target_samples 10



# Render specific scene with limited items (using relative path from active_spatial_pipeline dir)
python render_init_target_views.py \
    --jsonl_path ../../data/active_spatial/train_data_0267_840790.jsonl \
    --output_dir ../../data/active_spatial/rendered_views \
    --gs_root /scratch/by2593/project/Active_Spatial/InteriorGS \
    --scene_id 0267_840790 \
    --max_items 1000 \
    --num_target_samples 10
0251_840828

    
```

### 3. Generate Navigation Paths & Render Images
```bash
# Full test on a single scene (generates paths + renders images)
./run_full_test.sh 0267_840790 2 5
# Arguments: scene_id, samples_per_task, render_every_n_steps

# Or run separately:
# Step 1: Generate paths
python path_generator.py \
    --jsonl_path ../../data/active_spatial/train_data.jsonl \
    --scene_id 0267_840790 \
    --output_path ../../data/active_spatial/paths_0267_840790.jsonl \
    --max_items 10 --verbose

# Step 2: Render images
python render_path_images.py \
    --paths_jsonl ../../data/active_spatial/paths_0267_840790.jsonl \
    --output_dir ../../data/active_spatial/rendered/0267_840790 \
    --gs_root /scratch/by2593/project/Active_Spatial/InteriorGS \
    --render_every_n 5
```

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT                                          │
│  scenes_root/                                                               │
│  ├── scene_001/                                                             │
│  │   ├── labels.json        (物体标注: id, label, bounding_box)             │
│  │   └── structure.json     (房间多边形, 可选)                               │
│  ├── scene_002/                                                             │
│  └── ...                                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Object Selection (object_selector.py)                              │
│  ──────────────────────────────────────────────                             │
│  • 加载 labels.json                                                         │
│  • 过滤黑名单物体 (wall, floor, ceiling, lamp, etc.)                         │
│  • 几何约束过滤 (尺寸、体积、长宽比)                                           │
│  • 房间内唯一性检查                                                          │
│                                                                             │
│  输出: valid_objects = [obj1, obj2, obj3, ...]                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Task-based Object Grouping                                         │
│  ───────────────────────────────────                                        │
│  根据启用的任务类型，自动组织物体:                                             │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ 1-Object Tasks  │  │ 2-Object Tasks  │  │ 3-Object Tasks  │              │
│  │ ─────────────── │  │ ─────────────── │  │ ─────────────── │              │
│  │ • absolute_pos  │  │ • equidistance  │  │ • centering     │              │
│  │ • delta_control │  │ • projective    │  │                 │              │
│  │ • screen_occup  │  │ • occlusion     │  │                 │              │
│  │                 │  │ • fov_inclusion │  │                 │              │
│  │                 │  │ • size_distance │  │                 │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
│           │                    │                    │                       │
│           ▼                    ▼                    ▼                       │
│      [obj1]              [obj1, obj2]        [obj1, obj2, obj3]             │
│      [obj2]              [obj1, obj3]        [obj1, obj2, obj4]             │
│      [obj3]              [obj2, obj3]        ...                            │
│      ...                 ...                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Camera Sampling (camera_sampler.py)                                │
│  ───────────────────────────────────────────                                │
│  对每个物体/物体组:                                                          │
│  • 在物体周围采样 N 个相机位置                                                │
│  • 检查可见性约束                                                            │
│  • 检查遮挡约束                                                              │
│                                                                             │
│  输出: camera_poses = [CameraPose(position, target, yaw), ...]              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Task Generation (task_generator.py)                                │
│  ───────────────────────────────────────────                                │
│  对每个 (物体组, 相机位姿) 组合:                                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  生成 TaskResult:                                                   │    │
│  │  • task_type: 任务类型                                              │    │
│  │  • target_region: 目标区域 (CIRCLE/LINE/RAY/HALF_PLANE/...)         │    │
│  │  • sample_point: 区域内采样的一个有效点                              │    │
│  │  • preset: 空间关系 (front/back/left/right/...)                     │    │
│  │  • object_label: 物体标签                                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Target Region Types:                                                       │
│  ┌──────────┬───────────────────────────────────────────────────────────┐   │
│  │ CIRCLE   │ 以物体为圆心的圆 (absolute_positioning, screen_occupancy) │   │
│  │ POINT    │ 单点 (delta_control)                                      │   │
│  │ LINE     │ 垂直平分线 (equidistance)                                 │   │
│  │ HALF_PLANE│ 半平面 (projective_relations)                            │   │
│  │ RAY      │ 射线 (centering, occlusion_alignment)                     │   │
│  │ ANNULUS  │ 环形区域 (fov_inclusion)                                  │   │
│  │ CURVE    │ Apollonius圆 (size_distance_invariance)                   │   │
│  └──────────┴───────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: Output Generation (pipeline.py)                                    │
│  ───────────────────────────────────────                                    │
│  组装 TrainingDataItem:                                                     │
│  • scene_id                                                                 │
│  • object_label                                                             │
│  • preset                                                                   │
│  • init_camera (intrinsics + extrinsics)                                    │
│  • target_region (完整的目标区域描述)                                        │
│  • sample_target (采样的目标点)                                             │
│  • camera_params (目标朝向)                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT                                         │
│  output_dir/                                                                │
│  ├── train_data.jsonl       (主数据集, 每行一条JSON记录)                     │
│  ├── dataset.json           (JSON格式副本)                                  │
│  ├── metadata.json          (元数据: 统计信息、配置)                         │
│  └── scenes/                (可选: 每个场景的中间结果)                        │
│      ├── scene_001/                                                         │
│      │   └── data.jsonl                                                     │
│      └── ...                                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Overview

The pipeline consists of three main stages:

1. **Object Selection** (`object_selector.py`): Filters and selects suitable objects/pairs from scene labels
2. **Camera Pose Sampling** (`camera_sampler.py`): Samples valid camera positions around objects with enhanced scene boundary validation
3. **Task Generation** (`task_generator.py`): Computes optimal target positions based on 9 spatial navigation tasks

### Supported Tasks

The pipeline supports 9 types of spatial navigation tasks:

#### Metric Distance Tasks
1. **Absolute Positioning**: Move to a position at distance d from target
2. **Delta Control**: Move d meters closer along the view direction
3. **Equidistance**: Find positions equidistant from two objects

#### Relative Position Tasks
4. **Projective Relations**: A appears left/right of B from agent's view
5. **Centering**: Place A between B and C from agent's view
6. **Occlusion Alignment**: Hide A behind B from agent's view

#### View Perspective Tasks
7. **FoV Inclusion**: Both A and B should be visible in the frame
8. **Size-Distance Invariance**: A and B appear same size on screen
9. **Screen Occupancy**: Object A occupies k% of vertical FoV

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy
```

## Usage

### Command Line

```bash
# Basic usage
python run_pipeline.py \
    --scenes_root /scratch/by2593/project/Active_Spatial/InteriorGS \
    --output_dir /scratch/by2593/project/Active_Spatial/VAGEN/data/active_spatial \
    --num_cameras 5

# Process specific scenes
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --scenes 0267_840790 0002_839955

# Enable only specific tasks
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --tasks absolute_positioning equidistance fov_inclusion
```

### Python API

```python
from active_spatial_pipeline import (
    PipelineConfig,
    ActiveSpatialPipeline,
)

# Create configuration
config = PipelineConfig(
    scenes_root='/path/to/InteriorGS',
    output_dir='/path/to/output',
)

# Run pipeline
pipeline = ActiveSpatialPipeline(config)
data_items = pipeline.run()

# Process single scene
data_items = pipeline.run_single_scene('0267_840790')
```

### Using Config File

Create a JSON config file:

```json
{
    "scenes_root": "/path/to/InteriorGS",
    "output_dir": "/path/to/output",
    "object_selection": {
        "min_dim_component": 0.1,
        "max_dim_component": 3.0,
        "min_volume": 0.1
    },
    "camera_sampling": {
        "num_cameras_per_item": 5,
        "move_pattern": "around",
        "camera_heights": [1.0, 1.2, 1.5, 1.8],
        "rotation_interval": 5.0,
        "rotation_camera_height": 1.5,
        "linear_num_steps": 5,
        "linear_move_distance": 0.5
    },
    "task_config": {
        "enabled_tasks": ["absolute_positioning", "equidistance", "fov_inclusion"],
        "fov_horizontal": 90.0,
        "fov_vertical": 60.0
    }
}
```

### Camera Move Patterns

The pipeline supports 3 camera movement patterns:

| Pattern | Description | Use Case |
|---------|-------------|----------|
| `around` | Circle around object horizontally | Default, multi-view of single object |
| `rotation` | Stand at room center, rotate 360° | Room panorama, seeing all objects |
| `linear` | Walk past object (passing motion) | Simulating agent walking past an object |

#### Configuration Parameters

**Common Parameters**:
- `num_cameras_per_item`: Number of camera poses per object (default: 5)
- `max_camera_height`: Maximum camera height (default: 1.6m)
- `min_camera_height`: Minimum camera height (default: 0.8m)

**Rotation Mode** (`move_pattern='rotation'`):
- `rotation_interval`: Degrees between each camera pose (default: 5.0, 360/5 = 72 images)
- `rotation_camera_height`: Fixed camera height (default: 1.5m)

**Linear Mode** (`move_pattern='linear'`):
- `linear_num_steps`: Number of poses along trajectory (default: 5)
- `linear_move_distance`: Total movement distance along trajectory (default: 0.5m)

#### Example: Different Move Patterns

```python
# Default "around" pattern - circle around objects
config = PipelineConfig(
    scenes_root='/path/to/InteriorGS',
    output_dir='/path/to/output',
)
config.camera_sampling.move_pattern = 'around'

# "rotation" pattern - 360° room scan
config.camera_sampling.move_pattern = 'rotation'
config.camera_sampling.rotation_interval = 10.0  # 36 images per room

# "linear" pattern - walk past object
config.camera_sampling.move_pattern = 'linear'
config.camera_sampling.linear_num_steps = 7
config.camera_sampling.linear_move_distance = 0.8
```

Run with config:

```bash
python run_pipeline.py --config config.json
```

### Task-Object Mapping

The pipeline automatically determines how many objects to use based on the enabled tasks:

| Task | Objects Required |
|------|------------------|
| absolute_positioning | 1 object |
| delta_control | 1 object |
| screen_occupancy | 1 object |
| equidistance | 2 objects |
| projective_relations | 2 objects |
| occlusion_alignment | 2 objects |
| fov_inclusion | 2 objects |
| size_distance_invariance | 2 objects |
| centering | 3 objects |

## Output Format

The pipeline generates a `train_data.jsonl` file (JSON Lines format) where each line is a valid JSON object with the following structure:

```json
{
    "scene_id": "0267_840790",
    "object_label": "chair",
    "preset": "front",
    "distance": 2.5,
    "init_camera": {
        "intrinsics": [[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]],
        "extrinsics": [[1.0, 0.0, 0.0, 2.0], ...]
    },
    "target_region": {
        "type": "circle",
        "params": {
            "center": [1.5, 2.0],
            "radius": 2.5,
            "object_center": [1.5, 2.0, 0.5]
        },
        "sample_point": [4.0, 2.0, 1.5],
        "sample_forward": [−1.0, 0.0, 0.0],
        "height": 1.5
    },
    "sample_target": [4.0, 2.0, 1.5],
    "camera_params": {
        "forward": [−1.0, 0.0, 0.0]
    },
    "task_type": "absolute_positioning",
    "task_description": "Move to any position 2.5m from chair"
}
```

### Key Design: Target Region

Most spatial tasks have **multiple valid solutions** (not just a single point). The `target_region` field represents the full solution space:

| Region Type | Description | Tasks Using It |
|-------------|-------------|----------------|
| `point` | Single exact position | delta_control |
| `circle` | Circle at fixed distance from object | absolute_positioning, screen_occupancy |
| `line` | Line segment (perpendicular bisector) | equidistance |
| `ray` | Ray extending from origin in direction | centering, occlusion_alignment |
| `half_plane` | Half of 2D plane | projective_relations |
| `annulus` | Ring region (min/max distance) | fov_inclusion |
| `curve` | Apollonius circle for equal perceived size | size_distance_invariance |

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `scene_id` | string | Scene identifier |
| `object_label` | string | Primary object label (e.g., "chair", "table") |
| `preset` | string | Spatial relation type (front, back, left, right, etc.) |
| `distance` | float | Distance from sample point to target object(s) |
| `init_camera.intrinsics` | 3×3 matrix | Camera intrinsic matrix |
| `init_camera.extrinsics` | 4×4 matrix | Camera pose as transformation matrix |
| `target_region` | object | Full specification of valid target positions |
| `target_region.type` | string | Region type (point, circle, line, ray, etc.) |
| `target_region.params` | object | Region-specific parameters |
| `target_region.sample_point` | [x, y, z] | One sampled valid point |
| `sample_target` | [x, y, z] | Same as target_region.sample_point (convenience) |
| `camera_params.forward` | [x, y, z] | Camera forward direction at sample_target |

### Target Region Parameters by Type

**CIRCLE** (absolute_positioning, screen_occupancy):
```json
{
    "type": "circle",
    "params": {
        "center": [x, y],
        "radius": 2.5,
        "object_center": [x, y, z]
    }
}
```

**LINE** (equidistance):
```json
{
    "type": "line",
    "params": {
        "start": [x1, y1],
        "end": [x2, y2],
        "midpoint": [mx, my],
        "direction": [dx, dy]
    }
}
```

**RAY** (centering, occlusion_alignment):
```json
{
    "type": "ray",
    "params": {
        "origin": [x, y],
        "direction": [dx, dy],
        "min_distance": 0.0,
        "max_distance": 10.0
    }
}
```

**HALF_PLANE** (projective_relations):
```json
{
    "type": "half_plane",
    "params": {
        "boundary_point": [x, y],
        "boundary_direction": [dx, dy],
        "normal": [nx, ny]
    }
}
```

**ANNULUS** (fov_inclusion):
```json
{
    "type": "annulus",
    "params": {
        "center": [x, y],
        "min_radius": 3.0,
        "max_radius": 8.0
    }
}
```

**CURVE** (size_distance_invariance):
```json
{
    "type": "curve",
    "params": {
        "curve_type": "apollonius_circle",
        "center": [x, y],
        "radius": 4.0,
        "points": [[x1, y1, z1], ...]
    }
}
```

## Directory Structure

```
active_spatial_pipeline/
├── __init__.py           # Package exports
├── config.py             # Configuration classes
├── object_selector.py    # Object filtering and selection
├── camera_utils.py       # Camera utility functions (NEW - refactored from camera_sampler.py)
│                         # Contains: SceneBounds, AABB, CameraPose, projection functions,
│                         # FOV checking, occlusion detection, visibility functions
├── camera_sampler.py     # CameraSampler class (uses camera_utils.py)
├── task_generator.py     # Task and target position generation
├── pipeline.py           # Main pipeline orchestration
├── run_pipeline.py       # Command-line interface
├── path_generator.py     # Navigation path generation (greedy search)
├── render_path_images.py # Render images along paths
├── render_init_target_views.py  # Render initial and target views
├── generate_diverse_paths.py    # Generate diverse paths for all task types
├── run_full_test.sh      # Complete test script
├── test_render_scene.sh  # Quick test script
├── test_camera_sampler.py       # Camera sampler unit tests
├── test_camera_validation.py    # Comprehensive camera validation tests
├── test_enhanced_validation.py  # Enhanced visibility validation tests
└── README.md             # This file
```

### Code Architecture (Updated)

The camera-related code is organized into two files for better modularity:

1. **`camera_utils.py`** - Standalone utility functions (reusable)
   - Data classes: `SceneBounds`, `AABB`, `CameraPose`
   - Ray-AABB intersection: `intersects_ray_aabb()`
   - Camera projection: `camtoworld_from_pos_target()`, `world_to_camera()`, `project_point_to_image()`
   - FOV checking: `is_target_in_fov()`, `check_multiple_targets_in_fov()`
   - Occlusion detection: `is_target_occluded()`, `calculate_occlusion_area_2d()`
   - Enhanced visibility: `count_visible_corners()`, `calculate_projected_area_ratio()`

2. **`camera_sampler.py`** - CameraSampler class (business logic)
   - Imports utilities from `camera_utils.py`
   - Handles scene loading, camera sampling, validation
   - Methods: `sample_camera_for_single()`, `sample_camera_for_pair()`, `sample_camera_for_triple()`

## Initial & Target View Rendering (NEW)

The `render_init_target_views.py` script renders images showing the initial camera view and multiple target views from the target region. This is useful for:
- Visualizing what the agent sees at the start
- Visualizing what "success" looks like (target views)
- Data augmentation with multiple valid target views

### Usage

```bash
# Basic usage - render all items
python render_init_target_views.py \
    --jsonl_path ../../data/active_spatial/train_data.jsonl \
    --output_dir ../../data/active_spatial/rendered_views \
    --gs_root /scratch/by2593/project/Active_Spatial/InteriorGS

# Render specific scene with limited items
python render_init_target_views.py \
    --jsonl_path ../../data/active_spatial/train_data.jsonl \
    --output_dir ../../data/active_spatial/rendered_views \
    --gs_root /scratch/by2593/project/Active_Spatial/InteriorGS \
    --scene_id 0267_840790 \
    --max_items 20

# Custom number of target samples (default: 10)
python render_init_target_views.py \
    --jsonl_path ../../data/active_spatial/train_data.jsonl \
    --output_dir ../../data/active_spatial/rendered_views \
    --gs_root /scratch/by2593/project/Active_Spatial/InteriorGS \
    --num_target_samples 5
```

### Output Structure

```
rendered_views/
├── scene_001/
│   ├── item_000001/
│   │   ├── init_view.png           # Initial camera view
│   │   ├── target_view_00.png      # Target view sample 1
│   │   ├── target_view_01.png      # Target view sample 2
│   │   ├── ...
│   │   ├── target_view_09.png      # Target view sample 10
│   │   └── view_info.json          # Metadata
│   ├── item_000002/
│   │   └── ...
│   └── ...
├── scene_002/
│   └── ...
└── render_summary.json              # Overall statistics
```

### view_info.json Format

```json
{
    "scene_id": "0267_840790",
    "task_type": "absolute_positioning",
    "task_description": "Move to any position 2.5m from chair",
    "object_label": "chair",
    "preset": "front",
    "target_region_type": "circle",
    "init_view": "init_view.png",
    "target_views": [
        {
            "filename": "target_view_00.png",
            "position": [3.5, 2.0, 1.5],
            "look_at": [1.5, 2.0, 0.5]
        },
        ...
    ]
}
```

### Target Sampling Strategy by Region Type

| Region Type | Sampling Strategy |
|-------------|-------------------|
| `point` | Single point (only 1 sample) |
| `circle` | Evenly spaced around the circle |
| `line` | Evenly spaced along the line segment |
| `ray` | Evenly spaced along the ray (min to max distance) |
| `half_plane` | Random points in the valid half-plane |
| `annulus` | Random radius within min/max, evenly spaced angles |
| `curve` | Random selection from pre-computed curve points |

### Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--jsonl_path` | Path to train_data.jsonl | Required |
| `--output_dir` | Output directory | Required |
| `--gs_root` | Path to InteriorGS root | Required |
| `--scene_id` | Filter to specific scene | All scenes |
| `--max_items` | Limit number of items | All items |
| `--num_target_samples` | Target views per item | 10 |
| `--render_backend` | 'local' or 'client' | local |
| `--gpu_device` | GPU device ID | Auto |
| `--image_width` | Output image width | 640 |
| `--image_height` | Output image height | 480 |

## Path Generation & Rendering

### Path Generator (`path_generator.py`)

Generates optimal navigation paths from initial camera positions to target positions using:
- **Greedy search**: Evaluates all 6 discrete actions at each step
- **Region-based rewards**: Different reward functions for each target region type
- **Early stopping**: When reward exceeds threshold or no improvement

Action space:
- `move_forward` / `move_backward`: ±0.1m translation
- `turn_left` / `turn_right`: ±5° yaw rotation
- `look_up` / `look_down`: ±5° pitch rotation

### Path Renderer (`render_path_images.py`)

Renders images along generated paths using the Gaussian Splatting renderer:
- Supports both local (GPU) and client (WebSocket) rendering
- Outputs: `step_XXX.png` images + `path_info.json` metadata

### Output Format

Path data (JSONL):
```json
{
  "data_idx": 12345,
  "scene_id": "0267_840790",
  "task_type": "absolute_positioning",
  "task_description": "Move to any position 1.0m from chair",
  "path": [
    {"step_idx": 0, "action": "init", "position": [...], "reward": 0.28},
    {"step_idx": 1, "action": "move_forward", "position": [...], "reward": 0.35},
    ...
    {"step_idx": N, "action": "done", "position": [...], "reward": 0.95}
  ],
  "final_reward": 0.95,
  "success": true,
  "total_steps": 42
}
```

Rendered output:
```
rendered/scene_id/
  path_000001/
    step_000.png  # Initial view
    step_010.png  # After 10 actions
    ...
    step_042.png  # Final view
    path_info.json  # Metadata with reward progression
  path_000002/
    ...
  render_summary.json  # Overall statistics
```

## Configuration Options

### ObjectSelectionConfig
- `blacklist`: Set of object labels to exclude
- `min_dim_component`: Minimum dimension per axis (m)
- `max_dim_component`: Maximum dimension per axis (m)
- `min_volume`: Minimum volume (m³)
- `min_aspect_ratio`: Minimum shortest/longest edge ratio
- `min_pair_dist`: Minimum distance between paired objects
- `max_pair_dist`: Maximum distance between paired objects

### CameraSamplingConfig
- `num_cameras_per_item`: Number of camera poses to sample
- `per_angle`: Number of angles to try per radius
- `max_tries`: Maximum sampling attempts
- `camera_heights`: List of camera heights to try (m)
- `min_visibility_ratio`: Minimum visible area ratio
- `max_occlusion_ratio`: Maximum allowed occlusion

### TaskConfig
- `enabled_tasks`: List of task types to generate
- `absolute_positioning_distances`: Distances for absolute positioning tasks
- `delta_control_deltas`: Movement amounts for delta control tasks
- `projective_relations`: Relation types ['left', 'right']
- `screen_occupancy_ratios`: Screen occupancy targets
- `fov_horizontal`: Horizontal field of view (degrees)
- `fov_vertical`: Vertical field of view (degrees)
- `agent_height`: Default agent eye height (m)

## Camera Sampler - Scene Boundary Validation

The camera sampler (`camera_sampler.py`) includes enhanced validation to ensure cameras are always placed within valid scene boundaries. This is crucial for 3D Gaussian Splatting rendering, which only produces valid images within the trained scene bounds.

### Validation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Camera Position Validation (validate_camera_position)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Scene Bounds Check (from occupancy.json)                                │
│     ├─ Verify camera XY within [min, max] with 10cm safety margin           │
│     └─ Verify camera Z is within reasonable range                           │
│                                                                             │
│  2. Height Sanity Check                                                     │
│     └─ Camera height must be between 0.5m and 3.0m                          │
│                                                                             │
│  3. Room Polygon Check (from structure.json)                                │
│     ├─ Prefer: Camera in same room as target object                         │
│     └─ Fallback: Camera in any valid room                                   │
│                                                                             │
│  4. Visibility Check                                                        │
│     ├─ Object must be in front of camera                                    │
│     └─ Object must be within field of view                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

#### SceneBounds Class
Loads scene boundaries from `occupancy.json`:
```python
@dataclass
class SceneBounds:
    min_point: np.ndarray  # (x_min, y_min, z_min)
    max_point: np.ndarray  # (x_max, y_max, z_max)
    center: np.ndarray     # Scene center
    
    def contains_point_2d(self, x, y, margin=0.0) -> bool:
        """Check if XY position is within scene bounds."""
```

#### Dynamic Radius Limiting
Camera sampling radii are automatically limited based on scene size:
```python
# Prevent cameras from going outside scene
max_radius = min(calculated_radius, scene_size * 0.4)
```

#### Rejection Reason Tracking
When no valid camera poses are found, detailed statistics are logged:
```
[CameraSampler] Warning: No valid camera poses for object chair. 
Rejection reasons: {'outside_scene_bounds_xy': 120, 'not_in_target_room': 45, ...}
```

### Required Scene Files

| File | Purpose | Required |
|------|---------|----------|
| `labels.json` | Object bounding boxes | ✓ Yes |
| `occupancy.json` | Scene 3D boundaries | ✓ Recommended |
| `structure.json` | Room polygons | Optional |

### Testing Camera Validation

```bash
# Run comprehensive validation test
cd /scratch/by2593/project/Active_Spatial/VAGEN/data_gen/active_spatial_pipeline
source /scratch/by2593/miniconda3/bin/activate

# Basic test
python test_camera_sampler.py

# Comprehensive test across multiple scenes
python test_camera_validation.py
```

Expected output:
```
======================================================================
SUMMARY
======================================================================
Scenes tested: 20
Objects tested: 100
Total poses generated: 500
Valid poses (in bounds): 500 (100.0%)
Invalid poses (out of bounds): 0 (0.0%)

✓ No out-of-bounds camera positions detected!
```

## Camera Coordinate System and Orientation

### OpenCV/COLMAP Convention

The pipeline uses the **OpenCV/COLMAP camera coordinate convention**:

```
Camera Coordinate System:
        ↑ Z (forward - camera looks along +Z)
        │
        │
        │
        └──────→ X (right)
       ╱
      ╱
     ↓ Y (down)
```

This convention is used throughout:
- `look_at_matrix()` in `pipeline.py` - generates c2w matrices
- `gs_render_local.py` - expects w2c matrices for gsplat rendering
- `render_path_images.py` - converts c2w → w2c before rendering

### Camera Orientation Strategy

The camera always looks at target object(s) based on object count:

| Objects | Camera Target | Description |
|---------|---------------|-------------|
| 1 object | Object center | Object appears in view center |
| 2 objects | Midpoint | Line between objects' centers |
| 3+ objects | **Second object** | Places the 2nd object in view center |

This strategy ensures:
- Single object tasks: object is clearly centered in frame
- Pair tasks: both objects are symmetrically visible
- Triple tasks: the "reference" object (2nd) is centered, others visible on sides

### Field of View Configuration

Camera FoV is configured in `config.py`:
```python
@dataclass
class CameraSamplingConfig:
    fov_deg: float = 90.0        # Field of view in degrees
    focal_length: float = 277.0  # Focal length for 640x480 at 90° FoV
```

The relationship between FoV and focal length:
```
focal_length = image_width / (2 * tan(fov_deg / 2))

For 90° FoV with 640px width:
focal_length = 640 / (2 * tan(45°)) = 640 / 2 = 320
```

Note: A larger FoV (e.g., 90°) provides wider scene visibility but may introduce more distortion at image edges.

## Input Requirements

Each scene folder should contain:
- `labels.json`: Object annotations with bounding boxes (required)
- `occupancy.json`: Scene 3D boundary information (recommended for camera validation)
- `structure.json` (optional): Room polygon definitions
- `3dgs_compressed.ply`: 3D Gaussian Splatting model (for rendering)

The `labels.json` format:
```json
[
    {
        "ins_id": "123",
        "label": "chair",
        "bounding_box": [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 1.0, "y": 0.0, "z": 0.0},
            ...
        ]
    }
]
```

The `occupancy.json` format:
```json
{
    "scale": 0.05,
    "center": [4.14, -2.39, 0.0],
    "min": [-2.70, -7.42, 0.1],
    "max": [9.98, 2.65, 1.0],
    "lower": [-2.70, -7.42, 0.1],
    "upper": [9.98, 2.65, 1.0]
}
```

## Recent Updates

### 2024-12-30: Code Refactoring

**Camera code refactoring** - The camera-related code has been reorganized for better modularity:

- Created new `camera_utils.py` containing all standalone utility functions
- `camera_sampler.py` now imports from `camera_utils.py`
- **No functional changes** - all existing commands and logic remain the same

**New features added to camera validation:**
- Visible corner counting (`count_visible_corners()`)
- Projected area ratio calculation (`calculate_projected_area_ratio()`)
- 2D image-space occlusion detection (`calculate_occlusion_area_2d()`)

These features improve camera sampling quality by ensuring:
1. Objects have sufficient visible corners in the image
2. Objects are not too small in the image (area ratio check)
3. Objects are not significantly occluded by other objects

**Commands remain unchanged:**
```bash
# Generate training data (same as before)
python run_pipeline.py \
    --scenes_root /scratch/by2593/project/Active_Spatial/InteriorGS \
    --output_dir /scratch/by2593/project/Active_Spatial/VAGEN/data/active_spatial \
    --num_cameras 5 \
    --verbose

# Render views (same as before)
python render_init_target_views.py \
    --jsonl_path ../../data/active_spatial/train_data.jsonl \
    --output_dir ../../data/active_spatial/rendered_views \
    --gs_root /scratch/by2593/project/Active_Spatial/InteriorGS
```

## License

MIT License

---

## Detailed Pipeline Flow: Single Data Item Generation

This section provides a comprehensive, step-by-step explanation of how a single training data item is generated. The flow covers everything from raw scene input to the final JSONL output.

### High-Level Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    SINGLE DATA ITEM GENERATION FLOW                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Scene Input → Object Selection → Object Grouping → Camera Sampling         │
│       │              │                  │                │                   │
│       ▼              ▼                  ▼                ▼                   │
│  labels.json   SceneObject[]      [obj] / [obj,obj]   CameraPose            │
│                                   / [obj,obj,obj]                           │
│       │              │                  │                │                   │
│       └──────────────┴──────────────────┴────────────────┘                   │
│                                   │                                          │
│                                   ▼                                          │
│                          Task Generation                                     │
│                                   │                                          │
│                                   ▼                                          │
│                           TaskResult                                         │
│                   (target_region + sample_point)                             │
│                                   │                                          │
│                                   ▼                                          │
│                    TrainingDataItem Assembly                                 │
│                                   │                                          │
│                                   ▼                                          │
│                         JSONL Output                                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Step 1: Scene Discovery and Loading

**Location**: `pipeline.py` → `get_scene_list()`, `process_scene()`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: SCENE DISCOVERY AND LOADING                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: scenes_root directory                                               │
│         ├── scene_001/                                                       │
│         │   ├── labels.json          ← Object annotations (REQUIRED)        │
│         │   ├── occupancy.json       ← Scene bounds (RECOMMENDED)           │
│         │   ├── structure.json       ← Room polygons (OPTIONAL)             │
│         │   └── 3dgs_compressed.ply  ← 3DGS model (for rendering)           │
│         ├── scene_002/                                                       │
│         └── ...                                                              │
│                                                                              │
│  Process:                                                                    │
│    1. Scan scenes_root for subdirectories                                   │
│    2. Check each subdirectory for labels.json existence                     │
│    3. Build list of valid scene names                                       │
│    4. If --scene_id specified, filter to single scene                       │
│                                                                              │
│  Output: List[str] = ["scene_001", "scene_002", ...]                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Step 2: Object Selection and Filtering

**Location**: `object_selector.py` → `ObjectSelector.select_single_objects()`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: OBJECT SELECTION AND FILTERING                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: labels.json content                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  [                                                                   │   │
│  │    {                                                                 │   │
│  │      "ins_id": "123",                                                │   │
│  │      "label": "chair",                                               │   │
│  │      "bounding_box": [                                               │   │
│  │        {"x": 1.0, "y": 2.0, "z": 0.0},                               │   │
│  │        {"x": 2.0, "y": 2.0, "z": 0.0},                               │   │
│  │        ... (8 corner points)                                         │   │
│  │      ]                                                               │   │
│  │    },                                                                │   │
│  │    ...                                                               │   │
│  │  ]                                                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Filtering Pipeline:                                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Filter 1: Label Blacklist                                             │ │
│  │   - Exclude: wall, floor, ceiling, door, window, lamp, light,         │ │
│  │              curtain, blinds, rug, carpet, mat, room, area, space     │ │
│  │   - Reason: Structural/background objects not suitable for navigation │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                          ↓                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Filter 2: Geometric Constraints                                        │ │
│  │   - min_dim_component: 0.1m (minimum size per axis)                   │ │
│  │   - max_dim_component: 3.0m (maximum size per axis)                   │ │
│  │   - min_volume: 0.1 m³                                                │ │
│  │   - min_aspect_ratio: 0.1 (shortest/longest edge)                     │ │
│  │   - Reason: Filter too small, too large, or degenerate objects        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                          ↓                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Filter 3: Room Uniqueness (Optional)                                   │ │
│  │   - If structure.json exists, prefer objects unique within their room │ │
│  │   - Helps reduce ambiguity in natural language descriptions           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Output: List[SceneObject]                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  SceneObject:                                                        │   │
│  │    - id: str              (unique identifier)                        │   │
│  │    - label: str           ("chair", "table", etc.)                   │   │
│  │    - center: [x, y, z]    (bounding box center)                      │   │
│  │    - dims: [w, d, h]      (width, depth, height)                     │   │
│  │    - aabb_min: [x, y, z]  (axis-aligned bounding box min corner)     │   │
│  │    - aabb_max: [x, y, z]  (axis-aligned bounding box max corner)     │   │
│  │    - room_index: int      (which room the object belongs to)         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Step 3: Task-Based Object Grouping

**Location**: `pipeline.py` → `process_scene()`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: TASK-BASED OBJECT GROUPING                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: List[SceneObject] (valid objects from Step 2)                       │
│                                                                              │
│  Task Type → Object Count Mapping:                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 1-Object Tasks:                     2-Object Tasks:                    │ │
│  │   • absolute_positioning             • equidistance                    │ │
│  │   • delta_control                    • projective_relations            │ │
│  │   • screen_occupancy                 • occlusion_alignment             │ │
│  │                                      • fov_inclusion                   │ │
│  │ 3-Object Tasks:                      • size_distance_invariance        │ │
│  │   • centering                                                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Grouping Process:                                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ For 1-Object Tasks:                                                    │ │
│  │   - Simply iterate over each valid object                             │ │
│  │   - Output: [obj_1], [obj_2], [obj_3], ...                            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ For 2-Object Tasks: (object_selector.select_object_pairs)              │ │
│  │   - Generate all combinations C(n, 2)                                 │ │
│  │   - Filter by distance constraints:                                   │ │
│  │       min_pair_dist: 0.5m                                             │ │
│  │       max_pair_dist: 8.0m                                             │ │
│  │   - Optionally prefer same-room pairs                                 │ │
│  │   - Output: [obj_A, obj_B], [obj_A, obj_C], [obj_B, obj_C], ...       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ For 3-Object Tasks: (object_selector.select_object_triples)            │ │
│  │   - Generate all combinations C(n, 3)                                 │ │
│  │   - Filter by spatial constraints (objects should form valid config)  │ │
│  │   - Output: [obj_A, obj_B, obj_C], [obj_A, obj_B, obj_D], ...         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Output: List of object groups ready for camera sampling                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Step 4: Camera Pose Sampling

**Location**: `camera_sampler.py` → `CameraSampler.sample_cameras()`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: CAMERA POSE SAMPLING                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: Object group [obj_A] or [obj_A, obj_B] or [obj_A, obj_B, obj_C]     │
│                                                                              │
│  Sampling Strategy:                                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Step 4.1: Determine Look-At Target                                     │ │
│  │   • 1 object  → target = obj.center                                   │ │
│  │   • 2 objects → target = midpoint(obj_A.center, obj_B.center)         │ │
│  │   • 3 objects → target = obj_B.center (2nd object = reference)        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                          ↓                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Step 4.2: Sample Camera Positions                                      │ │
│  │   For each of num_cameras_per_item attempts:                          │ │
│  │     1. Sample radius: r ∈ [min_radius, max_radius]                    │ │
│  │     2. Sample angle:  θ ∈ [0, 2π)                                     │ │
│  │     3. Sample height: h ∈ camera_heights (e.g., [1.0, 1.2, 1.5, 1.8]) │ │
│  │     4. Compute position: (x, y, z) = target + (r*cos(θ), r*sin(θ), h) │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                          ↓                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Step 4.3: Validate Camera Position                                     │ │
│  │                                                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ Check 1: Scene Bounds (from occupancy.json)                      │ │ │
│  │  │   - Camera XY must be within scene [min, max] ± 20cm margin      │ │ │
│  │  │   - Camera Z must be in reasonable range [0.5m, 3.0m]            │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ Check 2: Room Polygon (from structure.json, optional)            │ │ │
│  │  │   - Prefer: Camera in same room as target object                 │ │ │
│  │  │   - Fallback: Camera in any valid room polygon                   │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ Check 3: Object Collision (from labels.json)                     │ │ │
│  │  │   - Camera must NOT be inside any furniture/object AABB          │ │ │
│  │  │   - Safety margin: 15cm around each object                       │ │ │
│  │  │   - Loads all object bounding boxes and checks containment       │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ Check 4: Wall Collision (from structure.json)                    │ │ │
│  │  │   - Camera must NOT be inside any wall AABB                      │ │ │
│  │  │   - Safety margin: 20cm from wall surfaces                       │ │ │
│  │  │   - Walls modeled as vertical rectangular prisms from endpoints  │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ Check 5: Wall Distance (from room polygons)                      │ │ │
│  │  │   - Camera must be at least 25-30cm from room boundary edges     │ │ │
│  │  │   - Calculated as 2D point-to-polygon-edge distance              │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ Check 6: Visibility / FOV                                        │ │ │
│  │  │   - Object must be in front of camera (dot product > 0)          │ │ │
│  │  │   - Object center must project within image bounds               │ │ │
│  │  │   - All target objects must be visible in frame                  │ │ │
│  │  │   - Uses full camera intrinsics for precise projection           │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ Check 7: Visible Corners Count                                   │ │ │
│  │  │   - At least 1 corner of object AABB must project to image       │ │ │
│  │  │   - Ensures object is not completely out of frame                │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ Check 8: Projected Area Ratio                                    │ │ │
│  │  │   - Object should not be too small in image                      │ │ │
│  │  │   - min_area_ratio: 5% of image area (default)                   │ │ │
│  │  │   - Ensures object is large enough to be meaningful              │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ Check 9: Occlusion Detection (2D image-space preferred)          │ │ │
│  │  │   - Projects target and occluders to image plane                 │ │ │
│  │  │   - Calculates pixel-level overlap using depth ordering          │ │ │
│  │  │   - max_occlusion_ratio: 70% (default)                           │ │ │
│  │  │   - Fallback: 3D ray-AABB intersection if cv2 unavailable        │ │ │
│  │  │   - Filters out occluders that overlap with target (e.g., items  │ │ │
│  │  │     mounted on the target object)                                │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Output: List[CameraPose]                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  CameraPose:                                                         │   │
│  │    - position: np.ndarray [x, y, z]   (camera position in world)     │   │
│  │    - target: np.ndarray [x, y, z]     (look-at point)                │   │
│  │    - yaw: float                       (horizontal rotation, degrees) │   │
│  │    - pitch: float                     (vertical rotation, degrees)   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Step 5: Task Generation and Target Region Computation

**Location**: `task_generator.py` → `TaskGenerator.generate_*_tasks()`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: TASK GENERATION AND TARGET REGION COMPUTATION                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: (Object group, CameraPose, Task type)                               │
│                                                                              │
│  For each (object_group, camera_pose) combination:                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ 1-Object Tasks: generate_single_object_tasks()                   │ │ │
│  │  ├──────────────────────────────────────────────────────────────────┤ │ │
│  │  │ ABSOLUTE_POSITIONING:                                            │ │ │
│  │  │   - Goal: "Move to position d meters from object"               │ │ │
│  │  │   - Target Region: CIRCLE                                       │ │ │
│  │  │     center = object.center_xy                                    │ │ │
│  │  │     radius = d (from absolute_positioning_distances config)      │ │ │
│  │  │   - Sample: Random point on circle at height h                   │ │ │
│  │  │   - Forward: Direction from sample_point toward object           │ │ │
│  │  ├──────────────────────────────────────────────────────────────────┤ │ │
│  │  │ DELTA_CONTROL:                                                   │ │ │
│  │  │   - Goal: "Move d meters closer along view direction"           │ │ │
│  │  │   - Target Region: POINT                                        │ │ │
│  │  │     position = camera_pos + d * view_direction                   │ │ │
│  │  │   - Sample: The single target point                              │ │ │
│  │  │   - Forward: Same as original camera forward                     │ │ │
│  │  ├──────────────────────────────────────────────────────────────────┤ │ │
│  │  │ SCREEN_OCCUPANCY:                                                │ │ │
│  │  │   - Goal: "Object should occupy k% of vertical FOV"             │ │ │
│  │  │   - Target Region: CIRCLE                                       │ │ │
│  │  │     radius = object_height / (2 * tan(k * fov_vertical / 2))     │ │ │
│  │  │   - Sample: Random point on circle                               │ │ │
│  │  │   - Forward: Direction toward object center                      │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ 2-Object Tasks: generate_two_object_tasks()                      │ │ │
│  │  ├──────────────────────────────────────────────────────────────────┤ │ │
│  │  │ EQUIDISTANCE:                                                    │ │ │
│  │  │   - Goal: "Move to position equidistant from A and B"           │ │ │
│  │  │   - Target Region: LINE (perpendicular bisector)                │ │ │
│  │  │     midpoint = (A.center + B.center) / 2                         │ │ │
│  │  │     direction = perpendicular to (B - A), normalized             │ │ │
│  │  │   - Sample: Random point along the bisector line                 │ │ │
│  │  │   - Forward: Direction toward midpoint of A and B                │ │ │
│  │  ├──────────────────────────────────────────────────────────────────┤ │ │
│  │  │ PROJECTIVE_RELATIONS:                                            │ │ │
│  │  │   - Goal: "A should appear left/right of B from camera view"    │ │ │
│  │  │   - Target Region: HALF_PLANE                                   │ │ │
│  │  │     boundary = line through B perpendicular to (A - B)           │ │ │
│  │  │     normal = direction toward valid half (left or right)         │ │ │
│  │  │   - Sample: Random point in valid half-plane                     │ │ │
│  │  │   - Forward: Direction toward midpoint                           │ │ │
│  │  ├──────────────────────────────────────────────────────────────────┤ │ │
│  │  │ OCCLUSION_ALIGNMENT:                                             │ │ │
│  │  │   - Goal: "A should be hidden behind B"                         │ │ │
│  │  │   - Target Region: RAY                                          │ │ │
│  │  │     origin = A.center                                            │ │ │
│  │  │     direction = normalize(A - B) (away from B through A)         │ │ │
│  │  │   - Sample: Point along ray beyond A                             │ │ │
│  │  │   - Forward: Direction toward B (so A is behind B)               │ │ │
│  │  ├──────────────────────────────────────────────────────────────────┤ │ │
│  │  │ FOV_INCLUSION:                                                   │ │ │
│  │  │   - Goal: "Both A and B should be visible in frame"             │ │ │
│  │  │   - Target Region: ANNULUS                                      │ │ │
│  │  │     center = midpoint(A, B)                                      │ │ │
│  │  │     min_radius = minimum distance for both in FOV                │ │ │
│  │  │     max_radius = maximum useful distance                         │ │ │
│  │  │   - Sample: Random point in annulus region                       │ │ │
│  │  │   - Forward: Direction toward midpoint                           │ │ │
│  │  ├──────────────────────────────────────────────────────────────────┤ │ │
│  │  │ SIZE_DISTANCE_INVARIANCE:                                        │ │ │
│  │  │   - Goal: "A and B should appear same size on screen"           │ │ │
│  │  │   - Target Region: CURVE (Apollonius circle)                    │ │ │
│  │  │     ratio = size_A / size_B                                      │ │ │
│  │  │     Apollonius circle where dist_A / dist_B = ratio              │ │ │
│  │  │   - Sample: Random point on the Apollonius circle                │ │ │
│  │  │   - Forward: Direction toward midpoint                           │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ 3-Object Tasks: generate_three_object_tasks()                    │ │ │
│  │  ├──────────────────────────────────────────────────────────────────┤ │ │
│  │  │ CENTERING:                                                       │ │ │
│  │  │   - Goal: "A should appear between B and C in camera view"      │ │ │
│  │  │   - Target Region: RAY                                          │ │ │
│  │  │     origin = A.center                                            │ │ │
│  │  │     direction = away from midpoint(B, C) through A               │ │ │
│  │  │   - Sample: Point along ray                                      │ │ │
│  │  │   - Forward: Direction toward A (so B, A, C aligned)             │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Output: TaskResult                                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  TaskResult:                                                         │   │
│  │    - task_type: str          ("absolute_positioning", etc.)          │   │
│  │    - object_label: str       ("chair", "table and sofa", etc.)       │   │
│  │    - preset: str             ("front", "left_of", "between", etc.)   │   │
│  │    - description: str        (Natural language task description)     │   │
│  │    - target_region: TargetRegion                                     │   │
│  │        - region_type: RegionType (CIRCLE, LINE, RAY, etc.)           │   │
│  │        - params: Dict         (center, radius, direction, etc.)      │   │
│  │        - sample_point: [x,y,z] (one valid target position)           │   │
│  │        - sample_forward: [x,y,z] (camera forward at sample)          │   │
│  │    - sample_point: np.ndarray (shortcut to target_region.sample)     │   │
│  │    - distance: float          (distance from sample to object)       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Step 6: Training Data Item Assembly

**Location**: `pipeline.py` → `create_training_item()`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: TRAINING DATA ITEM ASSEMBLY                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: (scene_name, CameraPose, TaskResult, objects)                       │
│                                                                              │
│  Assembly Process:                                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 1. Generate Camera Intrinsics                                          │ │
│  │    ┌──────────────────────────────────────────────────────────────┐   │ │
│  │    │ intrinsics = generate_intrinsics(width, height, fov_deg)     │   │ │
│  │    │                                                              │   │ │
│  │    │     [fx,  0, cx]      fx = width / (2 * tan(fov/2))          │   │ │
│  │    │ K = [ 0, fy, cy]      fy = fx (square pixels)                │   │ │
│  │    │     [ 0,  0,  1]      cx, cy = image center                  │   │ │
│  │    └──────────────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 2. Generate Camera Extrinsics (Camera-to-World Matrix)                 │ │
│  │    ┌──────────────────────────────────────────────────────────────┐   │ │
│  │    │ c2w = look_at_matrix(camera_pos, look_at_target)             │   │ │
│  │    │                                                              │   │ │
│  │    │ Uses OpenCV/COLMAP convention:                               │   │ │
│  │    │   X = right, Y = down, Z = forward                           │   │ │
│  │    │                                                              │   │ │
│  │    │     [Rx, Ry, Rz, Tx]                                         │   │ │
│  │    │ T = [Rx, Ry, Rz, Ty]     R = [right | down | forward]        │   │ │
│  │    │     [Rx, Ry, Rz, Tz]     T = camera position                 │   │ │
│  │    │     [ 0,  0,  0,  1]                                         │   │ │
│  │    └──────────────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 3. Extract Target Object Info (for occlusion detection during nav)    │ │
│  │    ┌──────────────────────────────────────────────────────────────┐   │ │
│  │    │ target_object = {                                            │   │ │
│  │    │   "label": "chair",                                          │   │ │
│  │    │   "id": "123",                                               │   │ │
│  │    │   "bbox_min": [x, y, z],                                     │   │ │
│  │    │   "bbox_max": [x, y, z],                                     │   │ │
│  │    │   "center": [x, y, z]                                        │   │ │
│  │    │ }                                                            │   │ │
│  │    └──────────────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 4. Assemble TrainingDataItem                                           │ │
│  │    ┌──────────────────────────────────────────────────────────────┐   │ │
│  │    │ TrainingDataItem:                                            │   │ │
│  │    │   scene_id: "0267_840790"                                    │   │ │
│  │    │   object_label: "chair"                                      │   │ │
│  │    │   preset: "front"                                            │   │ │
│  │    │   distance: 2.5                                              │   │ │
│  │    │   init_camera:                                               │   │ │
│  │    │     intrinsics: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]        │   │ │
│  │    │     extrinsics: [[...], [...], [...], [...]] (4x4 c2w)       │   │ │
│  │    │   target_region:                                             │   │ │
│  │    │     type: "circle"                                           │   │ │
│  │    │     params: {center: [x,y], radius: 2.5, ...}                │   │ │
│  │    │     sample_point: [x, y, z]                                  │   │ │
│  │    │     sample_forward: [dx, dy, dz]                             │   │ │
│  │    │   sample_target: [x, y, z]                                   │   │ │
│  │    │   camera_params: {forward: [dx, dy, dz]}                     │   │ │
│  │    │   task_type: "absolute_positioning"                          │   │ │
│  │    │   task_description: "Move to any position 2.5m from chair"   │   │ │
│  │    │   target_object: {...}                                       │   │ │
│  │    └──────────────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Output: TrainingDataItem (ready for JSON serialization)                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Step 7: Output Serialization

**Location**: `pipeline.py` → `run()`, `run_single_scene()`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: OUTPUT SERIALIZATION                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: List[TrainingDataItem] (all generated items)                        │
│                                                                              │
│  Serialization Process:                                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 1. Convert to JSON-serializable dict                                   │ │
│  │    - TrainingDataItem.to_dict()                                        │ │
│  │    - _convert_to_native(): np.ndarray → list, np.float64 → float      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 2. Write JSONL file (primary format)                                   │ │
│  │    - One JSON object per line                                          │ │
│  │    - File: train_data.jsonl or train_data_{scene_id}.jsonl             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 3. Write JSON file (backup/compatibility)                              │ │
│  │    - Full array format with indentation                                │ │
│  │    - File: dataset.json or dataset_{scene_id}.json                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 4. Write Metadata                                                      │ │
│  │    - metadata.json or metadata_{scene_id}.json                         │ │
│  │    - Contains: timestamp, total_items, scene_stats, config             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 5. Optional: Per-scene intermediate results                            │ │
│  │    - scenes/{scene_id}/data.jsonl                                      │ │
│  │    - Enabled by save_intermediate=True                                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Output Directory Structure:                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  output_dir/                                                         │   │
│  │  ├── train_data.jsonl            ← Main dataset (JSONL format)       │   │
│  │  ├── dataset.json                ← Same data (JSON array format)     │   │
│  │  ├── metadata.json               ← Generation metadata               │   │
│  │  └── scenes/                     ← Per-scene results (optional)      │   │
│  │      ├── scene_001/                                                  │   │
│  │      │   └── data.jsonl                                              │   │
│  │      ├── scene_002/                                                  │   │
│  │      │   └── data.jsonl                                              │   │
│  │      └── ...                                                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Single JSONL Record Example:                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  {                                                                   │   │
│  │    "scene_id": "0267_840790",                                        │   │
│  │    "object_label": "chair",                                          │   │
│  │    "preset": "front",                                                │   │
│  │    "distance": 2.5,                                                  │   │
│  │    "init_camera": {                                                  │   │
│  │      "intrinsics": [[493.7, 0.0, 256.0], ...],                       │   │
│  │      "extrinsics": [[0.98, -0.17, 0.0, 3.5], ...]                    │   │
│  │    },                                                                │   │
│  │    "target_region": {                                                │   │
│  │      "type": "circle",                                               │   │
│  │      "params": {"center": [1.5, 2.0], "radius": 2.5},                │   │
│  │      "sample_point": [4.0, 2.0, 1.5],                                │   │
│  │      "sample_forward": [-0.98, 0.0, -0.17],                          │   │
│  │      "height": 1.5                                                   │   │
│  │    },                                                                │   │
│  │    "sample_target": [4.0, 2.0, 1.5],                                 │   │
│  │    "camera_params": {"forward": [-0.98, 0.0, -0.17]},                │   │
│  │    "task_type": "absolute_positioning",                              │   │
│  │    "task_description": "Move to any position 2.5m from chair",       │   │
│  │    "target_object": {                                                │   │
│  │      "label": "chair",                                               │   │
│  │      "bbox_min": [1.0, 1.5, 0.0],                                    │   │
│  │      "bbox_max": [2.0, 2.5, 1.0],                                    │   │
│  │      "center": [1.5, 2.0, 0.5]                                       │   │
│  │    }                                                                 │   │
│  │  }                                                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Complete Flow Summary

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     COMPLETE PIPELINE EXECUTION FLOW                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  for each scene in scenes:                                                  │
│    │                                                                        │
│    ├─► Step 2: objects = ObjectSelector.select_single_objects(scene)        │
│    │                                                                        │
│    ├─► Step 3: Group objects by task requirements                           │
│    │     • single_objects = [obj1, obj2, obj3, ...]                         │
│    │     • object_pairs = [(obj1,obj2), (obj1,obj3), ...]                   │
│    │     • object_triples = [(obj1,obj2,obj3), ...]                         │
│    │                                                                        │
│    └─► for each object_group:                                               │
│          │                                                                  │
│          ├─► Step 4: camera_poses = CameraSampler.sample_cameras(group)     │
│          │                                                                  │
│          └─► for each camera_pose:                                          │
│                │                                                            │
│                ├─► Step 5: tasks = TaskGenerator.generate_*_tasks(...)      │
│                │                                                            │
│                └─► for each task:                                           │
│                      │                                                      │
│                      ├─► Step 6: item = create_training_item(...)           │
│                      │                                                      │
│                      └─► data_items.append(item)                            │
│                                                                              │
│  Step 7: Write data_items to JSONL/JSON files                               │
│                                                                              │
│  Result: N data items where                                                 │
│    N = Σ(scenes) × Σ(objects/pairs/triples) × cameras × tasks               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Target Regions vs Single Points**: Most spatial navigation tasks have multiple valid solutions. The pipeline generates `target_region` (the full solution space) and `sample_target` (one example point for training).

2. **Task-Object Coupling**: The number of objects determines which tasks are applicable:
   - 1 object → metric distance tasks
   - 2 objects → relative position tasks
   - 3 objects → centering task

3. **Comprehensive Validation Pipeline**: Camera positions undergo 9 validation checks to ensure high-quality training data:
   - **Spatial checks**: Scene bounds, room polygon containment
   - **Collision checks**: Camera must not be inside objects or walls (with safety margins)
   - **Distance checks**: Minimum distance from room boundaries/walls
   - **Visibility checks**: FOV projection, visible corners, projected area ratio
   - **Occlusion checks**: 2D image-space occlusion detection with depth ordering

4. **Modular Architecture**: Each step is encapsulated in a separate module (`object_selector.py`, `camera_sampler.py`, `camera_utils.py`, `task_generator.py`, `pipeline.py`) for maintainability and testing.

5. **OpenCV/COLMAP Convention**: All camera matrices follow the standard computer vision convention (X=right, Y=down, Z=forward) for compatibility with rendering pipelines.

