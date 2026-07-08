#!/usr/bin/env python3
"""
visualize_trajectories.py — 从 validation JSONL 重建轨迹并生成俯视图

工作流程:
  1. 读取 validation/{step}.jsonl (VERL 格式)，从 input 提取初始相机位姿
  2. 从 output 提取动作序列（每 turn 的 <action>...</action>）
  3. 用 ViewManipulator 逐步模拟动作，记录 (tx, ty) 轨迹
  4. 从数据集 JSONL 查找目标区域（target_region）和目标物体 bbox
  5. 用 matplotlib 绘制俯视图：起点 → 轨迹 → 终点 vs 目标位置

用法:
    # 可视化最新 val step 的前 N 条轨迹
    python3 scripts/visualize_trajectories.py \
        --exp v34_grpo_rewscale_clip02 \
        --n_show 12 \
        --out_dir outputs/trajectories

    # 指定 val step
    python3 scripts/visualize_trajectories.py \
        --exp v34_grpo_rewscale_clip02 --val_step 150 --n_show 8

    # 直接指定 val JSONL 路径
    python3 scripts/visualize_trajectories.py \
        --val_jsonl exps/vagen_active_spatial/v34_grpo_rewscale_clip02/validation/150.jsonl \
        --dataset_jsonl data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_7types.jsonl \
        --n_show 12 --out_dir outputs/trajectories
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

# ─────────────────────────── Config ───────────────────────────

DEFAULT_EXPDIR      = "exps/vagen_active_spatial"
DEFAULT_DATASET     = "data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_7types.jsonl"
DEFAULT_STEP_T      = 0.3     # metres
DEFAULT_STEP_R_DEG  = 20.0    # degrees
DEFAULT_VAL_N       = 4
DEFAULT_N_ID        = 21

# Task-type → color for subplot title
TASK_COLORS = {
    "absolute_positioning":      "#2196F3",
    "delta_control":             "#FF9800",
    "equidistance":              "#9C27B0",
    "projective_relations":      "#F44336",
    "centering":                 "#009688",
    "occlusion_alignment":       "#795548",
    "fov_inclusion":             "#607D8B",
    "size_distance_invariance":  "#E91E63",
    "screen_occupancy":          "#CDDC39",
    "unknown":                   "#9E9E9E",
}

TASK_TYPE_RULES = [
    ("projective_relations",     re.compile(r"left_of view|right_of view",          re.I)),
    ("equidistance",             re.compile(r"equidistant view",                    re.I)),
    ("centering",                re.compile(r"center view",                         re.I)),
    ("occlusion_alignment",      re.compile(r"occluded view",                       re.I)),
    ("delta_control",            re.compile(r"closer view|farther view",            re.I)),
    ("size_distance_invariance", re.compile(r"equal_size view",                     re.I)),
    ("screen_occupancy",         re.compile(r"screen occupancy",                    re.I)),
    ("fov_inclusion",            re.compile(r"fov inclusion|field.of.view",         re.I)),
    ("absolute_positioning",     re.compile(r"\b(front|back|left|right|side)\s+view\b", re.I)),
]


def infer_task_type(task_str: str) -> str:
    for ttype, pat in TASK_TYPE_RULES:
        if pat.search(task_str):
            return ttype
    return "unknown"


# ─────────────────────────── 相机位姿解析 ───────────────────────────

_POSE_PAT = re.compile(
    r"Current camera pose: \[tx=([0-9.\-]+), ty=([0-9.\-]+), tz=([0-9.\-]+), "
    r"rx=([0-9.\-]+)°, ry=([0-9.\-]+)°, rz=([0-9.\-]+)°\]"
)


def parse_initial_pose(entry_input: str) -> Optional[Tuple[float, ...]]:
    """解析 entry['input'] 中初始相机位姿 (tx, ty, tz, rx, ry, rz)。"""
    m = _POSE_PAT.search(entry_input)
    if not m:
        return None
    return tuple(float(x) for x in m.groups())


def parse_task_str(entry_input: str) -> str:
    for line in entry_input.split("\n"):
        if line.strip().startswith("Task:"):
            return line.strip()
    return ""


# ─────────────────────────── 动作序列解析 ───────────────────────────

_ACTION_TAG_PAT = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_VALID_ACTIONS  = {
    "move_forward", "move_backward", "move_left", "move_right",
    "turn_left", "turn_right", "look_up", "look_down", "done",
}


def parse_action_sequence(output_str: str) -> List[str]:
    """
    提取所有 <action>...</action> 中的动作，返回展平的动作列表。
    每个 tag 内可能有多个用 '|' 分隔的动作。
    """
    actions = []
    for m in _ACTION_TAG_PAT.finditer(output_str):
        blob = m.group(1).strip()
        parts = [p.strip().lower() for p in blob.split("|") if p.strip()]
        for p in parts:
            if p in _VALID_ACTIONS:
                actions.append(p)
    return actions


# ─────────────────────────── 轨迹模拟 ───────────────────────────

def _c2w_from_pose6(tx, ty, tz, rx_deg, ry_deg, rz_deg) -> np.ndarray:
    """将 (tx,ty,tz,rx,ry,rz) 构建 4×4 camera-to-world 矩阵。"""
    from scipy.spatial.transform import Rotation as R
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()
    c2w[:3, 3]  = [tx, ty, tz]
    return c2w


def simulate_trajectory(
    initial_pose: Tuple[float, ...],
    actions: List[str],
    step_t: float = DEFAULT_STEP_T,
    step_r_deg: float = DEFAULT_STEP_R_DEG,
) -> List[Tuple[float, float, float]]:
    """
    用 ViewManipulator 回放动作序列，记录每步的 (tx, ty, yaw_deg)。
    返回: [(tx, ty, yaw_deg), ...] 包含初始位置在内。
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from vagen.envs.active_spatial.utils import ViewManipulator
    except ImportError:
        # Fallback: approximate flat-floor simulation
        return _simulate_2d_approximate(initial_pose, actions, step_t, step_r_deg)

    tx, ty, tz, rx_deg, ry_deg, rz_deg = initial_pose
    c2w = _c2w_from_pose6(tx, ty, tz, rx_deg, ry_deg, rz_deg)

    vm = ViewManipulator(
        step_translation=step_t,
        step_rotation_deg=step_r_deg,
        world_up_axis="Z",
    )
    vm.reset(c2w)

    # Extract (tx, ty, yaw) at each step
    def _yaw_from_c2w(mat):
        from scipy.spatial.transform import Rotation as R
        r = R.from_matrix(mat[:3, :3])
        return r.as_euler("xyz", degrees=True)[2]

    traj = [(tx, ty, _yaw_from_c2w(c2w))]
    for a in actions:
        if a == "done":
            break
        vm.step(a)
        cur = vm.get_pose()
        traj.append((cur[0, 3], cur[1, 3], _yaw_from_c2w(cur)))

    return traj


def _simulate_2d_approximate(
    initial_pose, actions, step_t, step_r_deg
) -> List[Tuple[float, float, float]]:
    """No-dependency flat-floor approximation (XY plane, Z ignored)."""
    tx, ty, tz, rx_deg, ry_deg, rz_deg = initial_pose
    yaw = math.radians(rz_deg)  # approximate yaw from rz

    traj = [(tx, ty, rz_deg)]
    for a in actions:
        if a == "done":
            break
        if a == "move_forward":
            tx += step_t * math.sin(yaw)
            ty += step_t * math.cos(yaw)
        elif a == "move_backward":
            tx -= step_t * math.sin(yaw)
            ty -= step_t * math.cos(yaw)
        elif a == "move_right":
            tx += step_t * math.cos(yaw)
            ty -= step_t * math.sin(yaw)
        elif a == "move_left":
            tx -= step_t * math.cos(yaw)
            ty += step_t * math.sin(yaw)
        elif a == "turn_left":
            yaw -= math.radians(step_r_deg)
        elif a == "turn_right":
            yaw += math.radians(step_r_deg)
        traj.append((tx, ty, math.degrees(yaw)))
    return traj


# ─────────────────────────── 数据集查找 ───────────────────────────

def build_dataset_index(dataset_jsonl: str) -> Dict[str, Any]:
    """
    加载数据集 JSONL 并建立 task_description → item 的索引。
    用于查找目标区域 (target_region) 和目标物体 bbox。
    """
    index = {}
    if not dataset_jsonl or not os.path.isfile(dataset_jsonl):
        return index
    with open(dataset_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = item.get("task_description", "").strip()
            if key:
                index[key] = item
    return index


def find_dataset_item(
    task_str: str,
    dataset_index: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    根据 val entry 的 task_str 在数据集中查找匹配项。
    task_str 格式: 'Task: Move the camera to the front view of the sofa, about 1.50 meters away.'
    dataset task_description 格式: 'Move to any position 1.5m from sofa'
    Direct match is unlikely; do a fuzzy object-name match.
    """
    # Try direct match with the latter part after 'Task: '
    task_body = task_str.replace("Task:", "").strip()
    if task_body in dataset_index:
        return dataset_index[task_body]
    # No direct match — return None (visualization works without it)
    return None


# ─────────────────────────── 绘图 ───────────────────────────

def _draw_arrow(ax, x, y, yaw_deg, length=0.15, color="black", alpha=0.7, lw=1.0):
    """在 (x, y) 处绘制朝向 yaw_deg 的箭头。"""
    import math
    yaw_rad = math.radians(yaw_deg)
    dx = length * math.sin(yaw_rad)
    dy = length * math.cos(yaw_rad)
    ax.annotate(
        "", xy=(x + dx, y + dy), xytext=(x, y),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, alpha=alpha),
    )


def _draw_bbox_2d(ax, bbox_min, bbox_max, label="", color="gray", alpha=0.3):
    """绘制物体 XY 投影的 2D 边界框。"""
    import matplotlib.patches as mpatches
    x0, y0 = bbox_min[0], bbox_min[1]
    w  = bbox_max[0] - bbox_min[0]
    h  = bbox_max[1] - bbox_min[1]
    rect = mpatches.FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="square,pad=0",
        edgecolor=color, facecolor=color, alpha=alpha, linewidth=1.2,
    )
    ax.add_patch(rect)
    if label:
        cx, cy = x0 + w / 2, y0 + h / 2
        ax.text(cx, cy, label, ha="center", va="center", fontsize=5.5,
                color="black", alpha=0.8, zorder=10)


def _draw_target_region(ax, target_region: dict, color="gold", alpha=0.25):
    """绘制目标区域（circle / point）。"""
    import matplotlib.patches as mpatches
    rtype = target_region.get("type", "point")
    params = target_region.get("params", {})
    sp = target_region.get("sample_point")

    if rtype == "circle":
        center = params.get("center") or params.get("object_center", [0, 0])
        radius = params.get("radius", 0.5)
        cx, cy = float(center[0]), float(center[1])
        circ = mpatches.Circle(
            (cx, cy), radius,
            fill=True, facecolor=color, edgecolor="orange",
            alpha=alpha, linewidth=1.5, linestyle="--", zorder=2,
        )
        ax.add_patch(circ)
    elif rtype == "point" and sp:
        ax.scatter([sp[0]], [sp[1]], c="orange", s=120, marker="*", zorder=5,
                   edgecolors="black", linewidths=0.5, label="target")
    # Sample target point (always shown)
    if sp:
        ax.scatter([sp[0]], [sp[1]], c="gold", s=80, marker="*", zorder=6,
                   edgecolors="darkorange", linewidths=0.8)


def plot_episode(
    ax,
    traj: List[Tuple[float, float, float]],
    success: bool,
    task_type: str,
    task_str: str,
    dataset_item: Optional[Dict],
    ep_idx: int,
):
    """在一个 subplot 上绘制单个 episode 的轨迹。"""
    import matplotlib.pyplot as plt

    if not traj:
        ax.set_visible(False)
        return

    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]

    color = "#4CAF50" if success else "#F44336"
    task_color = TASK_COLORS.get(task_type, "#9E9E9E")

    # ── 绘制轨迹线 ─────────────────────────────────────────────────────
    ax.plot(xs, ys, color=color, linewidth=1.4, alpha=0.85, zorder=3)

    # 每个位置绘制小朝向箭头（最多 20 个，防止太密集）
    step = max(1, len(traj) // 20)
    for i in range(0, len(traj), step):
        _draw_arrow(ax, traj[i][0], traj[i][1], traj[i][2],
                    length=0.12, color=color, alpha=0.5, lw=0.7)

    # 起点（绿色圆圈）
    ax.scatter([xs[0]], [ys[0]], c="limegreen", s=70, marker="o", zorder=7,
               edgecolors="darkgreen", linewidths=0.8, label="start")
    # 终点（红/绿三角）
    end_marker_c = "#4CAF50" if success else "#F44336"
    ax.scatter([xs[-1]], [ys[-1]], c=end_marker_c, s=90, marker="^", zorder=8,
               edgecolors="black", linewidths=0.6, label="end")

    # ── 目标区域 & 物体 bbox ───────────────────────────────────────────
    if dataset_item:
        tr = dataset_item.get("target_region")
        if tr:
            _draw_target_region(ax, tr)
        tobj = dataset_item.get("target_object")
        if tobj:
            bmin = tobj.get("bbox_min")
            bmax = tobj.get("bbox_max")
            if bmin and bmax:
                _draw_bbox_2d(ax, bmin, bmax,
                              label=tobj.get("label", "")[:8],
                              color=task_color, alpha=0.25)

    # ── 标题 & 样式 ────────────────────────────────────────────────────
    result_str = "✓ SUCCESS" if success else "✗ FAIL"
    short_task = task_str.replace("Task: Move the camera to the ", "").replace(" view", "")[:35]
    ax.set_title(
        f"[{ep_idx}] {result_str}\n{short_task}",
        fontsize=6.5, color=task_color if not success else "darkgreen",
        pad=3,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)", fontsize=5)
    ax.set_ylabel("y (m)", fontsize=5)
    ax.tick_params(labelsize=5)
    ax.grid(True, linestyle=":", alpha=0.4, linewidth=0.5)


def visualize_val_file(
    val_jsonl: str,
    dataset_jsonl: str,
    out_dir: str,
    n_show: int = 12,
    val_n: int = DEFAULT_VAL_N,
    n_id: int = DEFAULT_N_ID,
    step_t: float = DEFAULT_STEP_T,
    step_r_deg: float = DEFAULT_STEP_R_DEG,
    select: str = "mixed",          # "mixed" | "success" | "failure"
    n_cols: int = 4,
    show: bool = False,
):
    """
    从单个 val JSONL 文件生成轨迹可视化图。

    select:
        "mixed"   — 成功 & 失败各一半
        "success" — 仅成功 episodes
        "failure" — 仅失败 episodes
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(val_jsonl) as f:
        all_entries = [json.loads(l) for l in f if l.strip()]

    # Only use ID entries (best-of-N: pick the first seed per env for simplicity)
    id_entries = all_entries[: n_id * val_n]
    # Pick one representative per env (highest reward)
    rep_entries = []
    for gi in range(0, len(id_entries), val_n):
        chunk = id_entries[gi : gi + val_n]
        if not chunk:
            continue
        # prefer successful; among successes take shortest action seq
        successes = [e for e in chunk if float(e.get("traj_success", 0)) > 0.5]
        if successes:
            pick = min(successes, key=lambda e: len(parse_action_sequence(e.get("output", ""))))
        else:
            pick = max(chunk, key=lambda e: float(e.get("reward", 0)))
        rep_entries.append(pick)

    # Filter by select mode
    if select == "success":
        pool = [e for e in rep_entries if float(e.get("traj_success", 0)) > 0.5]
    elif select == "failure":
        pool = [e for e in rep_entries if float(e.get("traj_success", 0)) <= 0.5]
    else:
        # mixed: first half success, second half failure
        succ  = [e for e in rep_entries if float(e.get("traj_success", 0)) > 0.5]
        fail  = [e for e in rep_entries if float(e.get("traj_success", 0)) <= 0.5]
        half  = n_show // 2
        pool  = succ[:half] + fail[: n_show - half]

    pool = pool[:n_show]
    if not pool:
        print("[WARN] No episodes to visualize after filtering.", file=sys.stderr)
        return

    # Build dataset index for target info
    ds_index = build_dataset_index(dataset_jsonl)

    # Compute trajectories
    episodes = []
    for ep_i, entry in enumerate(pool):
        pose = parse_initial_pose(entry.get("input", ""))
        if pose is None:
            continue
        actions = parse_action_sequence(entry.get("output", ""))
        traj = simulate_trajectory(pose, actions, step_t=step_t, step_r_deg=step_r_deg)
        task_str  = parse_task_str(entry.get("input", ""))
        task_type = infer_task_type(task_str)
        success   = float(entry.get("traj_success", 0)) > 0.5
        ds_item   = find_dataset_item(task_str, ds_index)
        episodes.append(dict(
            ep_idx=ep_i,
            traj=traj,
            success=success,
            task_type=task_type,
            task_str=task_str,
            ds_item=ds_item,
            n_actions=len(actions),
            reward=float(entry.get("reward", 0)),
        ))

    if not episodes:
        print("[WARN] No episodes with valid poses.", file=sys.stderr)
        return

    n_cols = min(n_cols, len(episodes))
    n_rows = math.ceil(len(episodes) / n_cols)
    fig_w  = n_cols * 3.0
    fig_h  = n_rows * 3.0 + 0.6

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[a] for a in axes]

    val_step = Path(val_jsonl).stem
    exp_name = Path(val_jsonl).parent.parent.name
    succ_count = sum(1 for e in episodes if e["success"])
    fig.suptitle(
        f"Trajectory Visualization  |  {exp_name}  step={val_step}  "
        f"succ={succ_count}/{len(episodes)} ({succ_count/len(episodes)*100:.0f}%)  "
        f"select={select}",
        fontsize=9, y=0.995,
    )

    for i, ep in enumerate(episodes):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        plot_episode(
            ax,
            traj=ep["traj"],
            success=ep["success"],
            task_type=ep["task_type"],
            task_str=ep["task_str"],
            dataset_item=ep["ds_item"],
            ep_idx=ep["ep_idx"],
        )
        # Annotate step count and reward
        ax.text(0.02, 0.02, f"acts={ep['n_actions']}  r={ep['reward']:.2f}",
                transform=ax.transAxes, fontsize=5.5, color="dimgray",
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))

    # Hide unused subplots
    for i in range(len(episodes), n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r][c].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{exp_name}_step{val_step}_{select}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[Saved] {out_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_task_type_summary(
    val_jsonl: str,
    out_dir: str,
    val_n: int = DEFAULT_VAL_N,
    n_id: int = DEFAULT_N_ID,
    step_t: float = DEFAULT_STEP_T,
    step_r_deg: float = DEFAULT_STEP_R_DEG,
):
    """
    生成按 task_type 分组的成功/失败代表性轨迹图（每类最多 2 个）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(val_jsonl) as f:
        all_entries = [json.loads(l) for l in f if l.strip()]

    id_entries = all_entries[: n_id * val_n]

    # Group by task_type
    type_episodes: Dict[str, list] = defaultdict(list)
    for gi in range(0, len(id_entries), val_n):
        chunk = id_entries[gi : gi + val_n]
        if not chunk:
            continue
        task_str  = parse_task_str(chunk[0].get("input", ""))
        task_type = infer_task_type(task_str)
        # Best seed per group
        successes = [e for e in chunk if float(e.get("traj_success", 0)) > 0.5]
        best = (min(successes, key=lambda e: len(parse_action_sequence(e.get("output", ""))))
                if successes else max(chunk, key=lambda e: float(e.get("reward", 0))))
        type_episodes[task_type].append((best, bool(successes)))

    all_types = [t for t in [
        "absolute_positioning", "delta_control", "equidistance",
        "projective_relations", "centering", "occlusion_alignment",
    ] if t in type_episodes]

    if not all_types:
        print("[WARN] No task types found.", file=sys.stderr)
        return

    # Layout: 2 episodes per task type, arranged as a grid
    # Extra left column acts as a row-label strip
    n_show_per_type = 2
    n_ep_cols = n_show_per_type          # episode columns
    n_label_cols = 1                      # task-label strip on the left
    n_cols_total = n_label_cols + n_ep_cols
    n_rows = len(all_types)

    col_ratios = [0.35] + [1.0] * n_ep_cols   # narrow label strip + equal episode cells
    fig, axes = plt.subplots(
        n_rows, n_cols_total,
        figsize=(n_ep_cols * 4.5 + 1.2, n_rows * 4.0 + 0.6),
        gridspec_kw={"width_ratios": col_ratios},
    )
    # Normalise axes to 2D list
    if n_rows == 1:
        axes = [axes]  # axes is now list of 1D arrays

    val_step = Path(val_jsonl).stem
    exp_name = Path(val_jsonl).parent.parent.name
    fig.suptitle(
        f"Per-Task-Type Trajectories  |  {exp_name}  step={val_step}",
        fontsize=11, y=1.00,
    )

    TASK_SHORT = {
        "absolute_positioning":   "Absolute\nPositioning",
        "delta_control":          "Delta\nControl",
        "equidistance":           "Equi-\ndistance",
        "projective_relations":   "Projective\nRelations",
        "centering":              "Centering",
        "occlusion_alignment":    "Occlusion\nAlignment",
    }

    for row_i, task_type in enumerate(all_types):
        eps_for_type = type_episodes[task_type]
        # Show 1 success + 1 failure when possible
        succ_eps = [e for e, s in eps_for_type if s]
        fail_eps = [e for e, s in eps_for_type if not s]
        show_eps = (succ_eps[:1] + fail_eps[:1])[:n_show_per_type]
        while len(show_eps) < n_show_per_type:
            show_eps.append(None)

        # --- label cell (col 0) ---
        ax_label = axes[row_i][0]
        ax_label.set_visible(False)                    # hide axis decorations
        ax_label.set_visible(True)
        ax_label.axis("off")
        ax_label.text(
            0.5, 0.5,
            TASK_SHORT.get(task_type, task_type),
            transform=ax_label.transAxes,
            fontsize=9, ha="center", va="center",
            fontweight="bold",
            color=TASK_COLORS.get(task_type, "#555"),
            wrap=True,
        )

        # --- episode cells (cols 1 and 2) ---
        for col_i in range(n_ep_cols):
            ax = axes[row_i][n_label_cols + col_i]
            entry = show_eps[col_i]
            if entry is None:
                ax.set_visible(False)
                continue
            pose = parse_initial_pose(entry.get("input", ""))
            if pose is None:
                ax.set_visible(False)
                continue
            actions  = parse_action_sequence(entry.get("output", ""))
            traj     = simulate_trajectory(pose, actions, step_t=step_t, step_r_deg=step_r_deg)
            task_str = parse_task_str(entry.get("input", ""))
            success  = float(entry.get("traj_success", 0)) > 0.5

            plot_episode(ax, traj, success, task_type, task_str, None, col_i)
            col_label = "✓ SUCCESS" if success else "✗ FAIL"
            ax.text(0.98, 0.98, col_label,
                    transform=ax.transAxes, fontsize=7,
                    color="#4CAF50" if success else "#F44336",
                    ha="right", va="top", fontweight="bold")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{exp_name}_step{val_step}_per_task.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[Saved] {out_path}")
    plt.close(fig)


# ─────────────────────────── 主程序 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VAGEN-Lite 轨迹俯视图可视化")
    # 数据输入（两种方式二选一）
    parser.add_argument("--exp", default=None,
                        help="实验名（在 --expdir 下查找），自动找最新 val step")
    parser.add_argument("--expdir", default=DEFAULT_EXPDIR)
    parser.add_argument("--val_step", type=int, default=None,
                        help="指定 val step（默认取最新）")
    parser.add_argument("--val_jsonl", default=None,
                        help="直接指定 val JSONL 路径")
    parser.add_argument("--dataset_jsonl", default=DEFAULT_DATASET,
                        help="数据集 JSONL 路径（用于加载目标区域）")
    # 输出控制
    parser.add_argument("--out_dir", default="outputs/trajectories")
    parser.add_argument("--n_show", type=int, default=12,
                        help="要可视化的 episode 数（默认 12）")
    parser.add_argument("--n_cols", type=int, default=4)
    parser.add_argument("--select", default="mixed",
                        choices=["mixed", "success", "failure"],
                        help="选择成功/失败/混合 episodes")
    parser.add_argument("--per_task", action="store_true",
                        help="额外生成 per-task-type 代表性轨迹图")
    parser.add_argument("--show", action="store_true",
                        help="显示交互式绘图窗口")
    # 环境参数
    parser.add_argument("--step_t",     type=float, default=DEFAULT_STEP_T)
    parser.add_argument("--step_r_deg", type=float, default=DEFAULT_STEP_R_DEG)
    parser.add_argument("--val_n",  type=int, default=DEFAULT_VAL_N)
    parser.add_argument("--n_id",   type=int, default=DEFAULT_N_ID)
    args = parser.parse_args()

    # ── Resolve val JSONL ─────────────────────────────────────────────────────
    val_jsonl = args.val_jsonl
    if val_jsonl is None:
        if args.exp is None:
            parser.error("Specify either --exp or --val_jsonl")
        expdir = os.path.join(args.expdir, args.exp)
        val_dir = os.path.join(expdir, "validation")
        if not os.path.isdir(val_dir):
            print(f"ERROR: {val_dir} not found", file=sys.stderr)
            sys.exit(1)
        val_files = sorted(
            [f for f in os.listdir(val_dir) if f.endswith(".jsonl")],
            key=lambda x: int(x[:-6])
        )
        if not val_files:
            print(f"ERROR: No val files in {val_dir}", file=sys.stderr)
            sys.exit(1)
        if args.val_step is not None:
            fname = f"{args.val_step}.jsonl"
            if fname not in val_files:
                print(f"ERROR: step {args.val_step} not found", file=sys.stderr)
                sys.exit(1)
            val_jsonl = os.path.join(val_dir, fname)
        else:
            val_jsonl = os.path.join(val_dir, val_files[-1])
            print(f"Using latest val step: {val_files[-1][:-6]}")

    print(f"Val JSONL: {val_jsonl}")
    print(f"Dataset:   {args.dataset_jsonl}")
    print(f"Select:    {args.select}  n={args.n_show}")

    visualize_val_file(
        val_jsonl=val_jsonl,
        dataset_jsonl=args.dataset_jsonl,
        out_dir=args.out_dir,
        n_show=args.n_show,
        val_n=args.val_n,
        n_id=args.n_id,
        step_t=args.step_t,
        step_r_deg=args.step_r_deg,
        select=args.select,
        n_cols=args.n_cols,
        show=args.show,
    )

    if args.per_task:
        plot_task_type_summary(
            val_jsonl=val_jsonl,
            out_dir=args.out_dir,
            val_n=args.val_n,
            n_id=args.n_id,
            step_t=args.step_t,
            step_r_deg=args.step_r_deg,
        )


if __name__ == "__main__":
    main()
