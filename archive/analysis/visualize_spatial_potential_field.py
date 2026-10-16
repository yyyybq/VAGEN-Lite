#!/usr/bin/env python3
"""Top-down SpatialPotentialField heatmaps for Active Spatial tasks.

For each task, the map is the look-at slice of the potential: at every (x, y)
the camera faces the task's reference object(s), then SpatialPotentialField
returns the geometric score (visual-bbox override off). That is the landscape
the SFT path finder / RL position reward actually climbs.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vagen.envs.active_spatial.spatial_potential_field import (
    create_potential_field,
    signed_distance_to_half_plane,
)

POSE_RE = re.compile(
    r"tx=([-\d.]+), ty=([-\d.]+), tz=([-\d.]+), "
    r"rx=([-\d.]+)°, ry=([-\d.]+)°, rz=([-\d.]+)°"
)

TASK_TITLES = {
    "absolute_positioning": "Absolute positioning",
    "delta_control": "Delta control",
    "equidistance": "Equidistance",
    "projective_relations": "Projective relations",
    "occlusion_alignment": "Occlusion alignment",
    "size_distance_invariance": "Size-distance invariance",
    "fov_inclusion": "FoV inclusion",
    "centering": "Centering",
    "screen_occupancy": "Screen occupancy",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_xy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return None
    return arr[:2]


def lookat_point(item: Dict[str, Any]) -> np.ndarray:
    params = (item.get("target_region") or {}).get("params") or {}
    if params.get("object_center") is not None:
        return np.asarray(params["object_center"], dtype=np.float64)
    a = params.get("object_a_center")
    b = params.get("object_b_center")
    if a is not None and b is not None:
        return 0.5 * (np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64))
    if params.get("midpoint_bc") is not None and params.get("object_a_center") is not None:
        return 0.5 * (
            np.asarray(params["object_a_center"], dtype=np.float64)
            + np.asarray(params["midpoint_bc"], dtype=np.float64)
        )
    target = item.get("target_object") or {}
    if isinstance(target.get("center"), (list, tuple)):
        return np.asarray(target["center"], dtype=np.float64)
    primary = target.get("primary")
    if isinstance(primary, dict) and primary.get("center") is not None:
        return np.asarray(primary["center"], dtype=np.float64)
    sample = (item.get("target_region") or {}).get("sample_point")
    if sample is not None:
        return np.asarray(sample, dtype=np.float64)
    init = np.asarray(item["init_camera"]["extrinsics"], dtype=np.float64)
    return init[:3, 3] + init[:3, 2]


def camera_height(item: Dict[str, Any]) -> float:
    extrinsics = np.asarray(item["init_camera"]["extrinsics"], dtype=np.float64)
    return float(extrinsics[2, 3])


def collect_objects(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    objects: Dict[str, Dict[str, Any]] = {}
    for item in items:
        target = item.get("target_object") or {}
        candidates = []
        if target.get("bbox_min") and target.get("bbox_max"):
            candidates.append(target)
        if isinstance(target.get("objects"), list):
            candidates.extend(target["objects"])
        if isinstance(target.get("primary"), dict):
            candidates.append(target["primary"])
        for obj in candidates:
            if not obj:
                continue
            key = str(obj.get("id") or obj.get("label") or obj.get("center"))
            if key and key not in objects and obj.get("bbox_min") and obj.get("bbox_max"):
                objects[key] = obj
    return list(objects.values())


def scene_bounds(items: Sequence[Dict[str, Any]], pad: float = 1.6) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for item in items:
        extrinsics = np.asarray(item["init_camera"]["extrinsics"], dtype=np.float64)
        xs.append(float(extrinsics[0, 3]))
        ys.append(float(extrinsics[1, 3]))
        params = (item.get("target_region") or {}).get("params") or {}
        for key in (
            "center",
            "object_center",
            "object_a_center",
            "object_b_center",
            "start_position",
            "sample_point",
            "origin",
            "midpoint",
        ):
            xy = as_xy(params.get(key) or (item.get("target_region") or {}).get(key))
            if xy is not None:
                xs.append(float(xy[0]))
                ys.append(float(xy[1]))
        for obj in collect_objects([item]):
            bmin = as_xy(obj.get("bbox_min"))
            bmax = as_xy(obj.get("bbox_max"))
            if bmin is not None and bmax is not None:
                xs.extend([float(bmin[0]), float(bmax[0])])
                ys.extend([float(bmin[1]), float(bmax[1])])
    if not xs:
        return -5.0, 5.0, -5.0, 5.0
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def evaluate_grid(
    field,
    item: Dict[str, Any],
    xs: np.ndarray,
    ys: np.ndarray,
    z: float,
) -> np.ndarray:
    look = lookat_point(item)
    task_type = item["task_type"]
    task_params = {
        "_target_object": item.get("target_object"),
        "_fov_horizontal": 90.0,
        "_fov_vertical": 90.0,
    }
    target_region = item.get("target_region") or {}
    scores = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            pos = np.array([x, y, z], dtype=np.float64)
            if look.size >= 3:
                forward = look[:3] - pos
            else:
                forward = np.array([look[0] - x, look[1] - y, 0.0], dtype=np.float64)
            forward[2] = 0.0
            norm = np.linalg.norm(forward)
            if norm < 1e-6:
                continue
            forward /= norm
            try:
                scores[i, j] = visual_score(
                    field, item, pos, forward, task_type, task_params, target_region
                )
            except Exception:
                continue
    return scores


def visual_score(field, item, pos, forward, task_type, task_params, target_region) -> float:
    """Score used for the heatmap.

    Projective relations are a half-plane: A appears left/right of B iff the
    camera stands on one side of the A-B line while looking at the pair.
    The env total_score uses a soft sigmoid that stays high on the *wrong*
    side near the boundary, which is the wrong picture for this figure.
    """
    if task_type == "projective_relations":
        params = (target_region or {}).get("params") or {}
        boundary = params.get("boundary_point")
        normal = params.get("normal")
        if boundary is None or normal is None:
            result = field.compute_score(pos, forward, task_type, task_params, target_region)
            return float(result.total_score)
        signed = signed_distance_to_half_plane(pos, np.asarray(boundary), np.asarray(normal))
        return 1.0 if signed > 0.0 else 0.0
    result = field.compute_score(pos, forward, task_type, task_params, target_region)
    return float(result.total_score)


def draw_target_region(ax, target_region: Dict[str, Any]) -> None:
    rtype = (target_region or {}).get("type", "")
    params = (target_region or {}).get("params") or {}
    sample = as_xy(target_region.get("sample_point"))

    if rtype == "circle":
        center = as_xy(params.get("center") or params.get("object_center"))
        radius = float(params.get("radius", params.get("sample_distance", 0.5)))
        if center is not None:
            ax.add_patch(
                mpatches.Circle(
                    (center[0], center[1]),
                    radius,
                    fill=False,
                    edgecolor="#f4c430",
                    linewidth=1.6,
                    linestyle="--",
                    zorder=4,
                )
            )
    elif rtype == "annulus":
        center = as_xy(params.get("center"))
        if center is not None:
            for radius, ls in (
                (params.get("min_radius"), ":"),
                (params.get("max_radius"), "--"),
            ):
                if radius is not None:
                    ax.add_patch(
                        mpatches.Circle(
                            (center[0], center[1]),
                            float(radius),
                            fill=False,
                            edgecolor="#f4c430",
                            linewidth=1.4,
                            linestyle=ls,
                            zorder=4,
                        )
                    )
    elif rtype == "point":
        point = as_xy(params.get("start_position"))
        delta = params.get("delta")
        if point is not None:
            ax.scatter([point[0]], [point[1]], c="#f4c430", s=40, marker="o", zorder=5)
            if delta is not None:
                dxy = as_xy(delta)
                if dxy is not None:
                    ax.arrow(
                        point[0],
                        point[1],
                        dxy[0],
                        dxy[1],
                        color="#f4c430",
                        width=0.02,
                        head_width=0.12,
                        length_includes_head=True,
                        zorder=5,
                    )
    elif rtype == "line":
        start = as_xy(params.get("start"))
        end = as_xy(params.get("end"))
        if start is not None and end is not None:
            ax.plot([start[0], end[0]], [start[1], end[1]], color="#f4c430", lw=2.0, zorder=4)
    elif rtype == "ray":
        origin = as_xy(params.get("origin"))
        direction = as_xy(params.get("direction"))
        if origin is not None and direction is not None:
            length = float(params.get("max_distance", 4.0))
            ax.annotate(
                "",
                xy=(origin[0] + direction[0] * length, origin[1] + direction[1] * length),
                xytext=(origin[0], origin[1]),
                arrowprops=dict(arrowstyle="-|>", color="#f4c430", lw=1.8),
                zorder=4,
            )
    elif rtype == "half_plane":
        a = as_xy(params.get("object_a_center"))
        b = as_xy(params.get("object_b_center"))
        point = as_xy(params.get("boundary_point"))
        normal = as_xy(params.get("normal"))
        direction = as_xy(params.get("boundary_direction"))
        if a is not None and b is not None:
            tangent = b - a
        elif direction is not None:
            tangent = direction
        else:
            tangent = None
        if tangent is not None and (a is not None or point is not None):
            origin = a if a is not None else point
            tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
            span = 40.0
            ax.plot(
                [origin[0] - span * tangent[0], origin[0] + span * tangent[0]],
                [origin[1] - span * tangent[1], origin[1] + span * tangent[1]],
                color="#f4c430",
                lw=2.0,
                zorder=4,
            )
        if point is not None and normal is not None:
            nrm = normal / (np.linalg.norm(normal) + 1e-8)
            ax.annotate(
                "valid",
                xy=(point[0] + nrm[0] * 1.1, point[1] + nrm[1] * 1.1),
                fontsize=7,
                color="#f4c430",
                ha="center",
                va="center",
                zorder=6,
            )
    elif rtype == "curve":
        points = params.get("points") or []
        if len(points) >= 2:
            arr = np.asarray(points, dtype=np.float64)
            ax.plot(arr[:, 0], arr[:, 1], color="#f4c430", lw=2.0, zorder=4)

    if sample is not None:
        ax.scatter(
            [sample[0]],
            [sample[1]],
            c="#ffe566",
            s=55,
            marker="*",
            edgecolors="#8a6d00",
            linewidths=0.6,
            zorder=6,
        )


SHORT_LABELS = {
    "smoke lampblack machine": "range hood",
    "bookshelf cabinet": "bookshelf",
    "basin cabinet": "basin cab.",
    "shoe cabinet": "shoe cab.",
    "wall cabinet": "wall cab.",
    "toy_animals": "toys",
    "floor lamp": "lamp",
}


def short_label(label: str) -> str:
    raw = str(label or "").strip()
    return SHORT_LABELS.get(raw, raw.replace("_", " "))


def task_focus_labels(item: Dict[str, Any]) -> set:
    labels = set()
    target = item.get("target_object") or {}
    if target.get("label"):
        labels.add(str(target["label"]))
    for obj in target.get("objects") or []:
        if obj.get("label"):
            labels.add(str(obj["label"]))
    primary = target.get("primary")
    if isinstance(primary, dict) and primary.get("label"):
        labels.add(str(primary["label"]))
    desc = str(item.get("task_description") or "").lower()
    for obj in collect_objects([item]):
        name = str(obj.get("label") or "")
        if name and name.lower() in desc:
            labels.add(name)
    return labels


def _label_anchor(bmin, bmax, used, xmin, xmax, ymin, ymax):
    cx = 0.5 * (bmin[0] + bmax[0])
    cy = 0.5 * (bmin[1] + bmax[1])
    gap_x = max(0.55, 0.35 + 0.15 * (bmax[0] - bmin[0]))
    gap_y = max(0.40, 0.28 + 0.12 * (bmax[1] - bmin[1]))
    candidates = [
        ((cx, bmax[1] + gap_y), "center", "bottom"),
        ((cx, bmin[1] - gap_y), "center", "top"),
        ((bmax[0] + gap_x, cy), "left", "center"),
        ((bmin[0] - gap_x, cy), "right", "center"),
        ((bmax[0] + gap_x, bmax[1] + gap_y), "left", "bottom"),
        ((bmin[0] - gap_x, bmin[1] - gap_y), "right", "top"),
    ]
    best = None
    best_score = -1e9
    for (tx, ty), ha, va in candidates:
        if tx < xmin + 0.15 or tx > xmax - 0.15 or ty < ymin + 0.15 or ty > ymax - 0.15:
            continue
        sep = min((abs(tx - ux) + 0.6 * abs(ty - uy)) for ux, uy in used) if used else 3.0
        inward = -0.15 * abs(tx - cx) + 0.05 * (ty - cy)
        score = sep + inward
        if score > best_score:
            best_score = score
            best = ((tx, ty), ha, va)
    if best is None:
        best = ((cx, bmax[1] + gap_y), "center", "bottom")
    return best


def draw_objects(
    ax,
    objects: Sequence[Dict[str, Any]],
    focus_labels: Optional[set] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    focus_labels = focus_labels or set()
    xmin, xmax, ymin, ymax = bounds if bounds else (-10, 10, -10, 10)
    used_offsets = []
    for obj in objects:
        bmin = as_xy(obj.get("bbox_min"))
        bmax = as_xy(obj.get("bbox_max"))
        if bmin is None or bmax is None:
            continue
        width = float(bmax[0] - bmin[0])
        height = float(bmax[1] - bmin[1])
        name = str(obj.get("label") or "")
        focused = name in focus_labels
        ax.add_patch(
            mpatches.Rectangle(
                (bmin[0], bmin[1]),
                width,
                height,
                facecolor="#f2c14e" if focused else "#cfcfcf",
                edgecolor="#8a6d00" if focused else "#666666",
                linewidth=1.1 if focused else 0.6,
                alpha=0.70 if focused else 0.32,
                zorder=3,
            )
        )
        if not focused or not name:
            continue
        (tx, ty), ha, va = _label_anchor(bmin, bmax, used_offsets, xmin, xmax, ymin, ymax)
        used_offsets.append((tx, ty))
        ax.annotate(
            short_label(name),
            xy=(0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])),
            xytext=(tx, ty),
            textcoords="data",
            ha=ha,
            va=va,
            fontsize=7.5,
            color="#222222",
            arrowprops=dict(arrowstyle="-", color="#8a6d00", lw=0.55, shrinkA=0, shrinkB=2),
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="#dddddd", alpha=0.78),
            zorder=10,
            annotation_clip=True,
        )


def parse_sft_path(record: Dict[str, Any]) -> List[Tuple[float, float]]:
    path: List[Tuple[float, float]] = []
    for turn in record.get("conversations") or []:
        if turn.get("role") != "user":
            continue
        for match in POSE_RE.finditer(turn.get("content") or ""):
            path.append((float(match.group(1)), float(match.group(2))))
    return path


def style_axis(ax, xmin, xmax, ymin, ymax, xlabel: bool = False, ylabel: bool = False) -> None:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)" if xlabel else "", fontsize=8)
    ax.set_ylabel("y (m)" if ylabel else "", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, color="white", alpha=0.22, linewidth=0.35)


def pick_items(dataset: Sequence[Dict[str, Any]], task_types: Sequence[str]) -> "OrderedDict[str, Dict[str, Any]]":
    chosen: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for task_type in task_types:
        for item in dataset:
            if item.get("task_type") == task_type:
                chosen[task_type] = item
                break
    return chosen


def wrap_title(text: str, width: int = 46) -> str:
    words = str(text).split()
    lines, cur = [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if len(trial) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return "\n".join(lines[:3])


def plot_panel(
    ax,
    field,
    item: Dict[str, Any],
    xs: np.ndarray,
    ys: np.ndarray,
    objects: Sequence[Dict[str, Any]],
    path: Optional[Sequence[Tuple[float, float]]] = None,
    title: Optional[str] = None,
):
    z = camera_height(item)
    scores = evaluate_grid(field, item, xs, ys, z)
    mesh = ax.pcolormesh(
        xs,
        ys,
        scores,
        cmap="magma",
        shading="auto",
        vmin=0.0,
        vmax=1.0,
        zorder=1,
    )
    bounds = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
    draw_objects(ax, objects, focus_labels=task_focus_labels(item), bounds=bounds)
    draw_target_region(ax, item.get("target_region") or {})
    init = np.asarray(item["init_camera"]["extrinsics"], dtype=np.float64)
    ax.scatter(
        [init[0, 3]],
        [init[1, 3]],
        c="white",
        s=42,
        marker="o",
        edgecolors="#111111",
        linewidths=0.7,
        zorder=8,
    )
    if path:
        px, py = zip(*path)
        ax.plot(px, py, color="#7fd3ff", lw=1.8, zorder=8)
        ax.scatter([px[-1]], [py[-1]], c="#7fd3ff", s=48, marker="^", edgecolors="#113344", zorder=9)
    ax.set_title(title or TASK_TITLES.get(item["task_type"], item["task_type"]), fontsize=9.5, pad=6)
    return mesh


def make_grid_figure(nrows: int = 2, ncols: int = 3):
    fig = plt.figure(figsize=(16.2, 4.8 * nrows + 1.6))
    gs = fig.add_gridspec(
        nrows,
        ncols + 1,
        width_ratios=[1] * ncols + [0.045],
        left=0.055,
        right=0.96,
        top=0.90 if nrows >= 3 else 0.84,
        bottom=0.08 if nrows >= 3 else 0.12,
        wspace=0.26,
        hspace=0.42 if nrows >= 3 else 0.46,
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]
    cax = fig.add_subplot(gs[:, -1])
    return fig, axes, cax


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.with_suffix('.png')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize SpatialPotentialField landscapes")
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "data_gen/active_spatial_pipeline/output/dataset_0267_840790.json"),
    )
    parser.add_argument(
        "--sft-jsonl",
        default=str(REPO_ROOT / "data_gen/active_spatial_sft/output_0267_v4/sft_data.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "reports/spatial_potential_field_0267"),
    )
    parser.add_argument("--grid", type=int, default=80)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = load_json(Path(args.dataset))
    sft_rows = load_jsonl(Path(args.sft_jsonl)) if Path(args.sft_jsonl).exists() else []
    output_dir = Path(args.output_dir)
    field = create_potential_field({"use_visual_bbox_scoring": False})

    overview_types = [
        "absolute_positioning",
        "delta_control",
        "equidistance",
        "projective_relations",
        "occlusion_alignment",
        "fov_inclusion",
        "size_distance_invariance",
        "centering",
        "screen_occupancy",
    ]
    overview_items = pick_items(dataset, overview_types)
    xmin, xmax, ymin, ymax = scene_bounds(list(overview_items.values()) or dataset[:20])
    xs = np.linspace(xmin, xmax, args.grid)
    ys = np.linspace(ymin, ymax, args.grid)
    objects = collect_objects(dataset)

    fig, axes, cax = make_grid_figure(nrows=3, ncols=3)
    mesh = None
    ncols = 3
    for idx, ax in enumerate(axes):
        if idx >= len(overview_items):
            ax.axis("off")
            continue
        task_type, item = list(overview_items.items())[idx]
        desc = str(item.get("task_description") or "")
        title = f"{TASK_TITLES[task_type]}\n{wrap_title(desc, 42)}"
        mesh = plot_panel(ax, field, item, xs, ys, objects, title=title)
        last_row_start = ncols * ((len(overview_items) - 1) // ncols)
        style_axis(
            ax,
            xmin,
            xmax,
            ymin,
            ymax,
            xlabel=idx >= last_row_start,
            ylabel=idx % ncols == 0,
        )
    if mesh is not None:
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label("Potential score", fontsize=9)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="black", label="init camera"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#ffe566", markeredgecolor="#8a6d00", label="sample target"),
        Line2D([0], [0], color="#f4c430", lw=2, label="target region"),
        mpatches.Patch(facecolor="#f2c14e", edgecolor="#8a6d00", label="task object"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.48, 0.02))
    fig.suptitle(
        "Spatial potential field on scene 0267_840790",
        fontsize=15,
        fontweight="bold",
        y=0.965,
    )
    fig.text(0.5, 0.915, "Look-at heading, geometric score. Gray boxes are other furniture.", ha="center", fontsize=10, color="#444444")
    save_figure(fig, output_dir / "01_task_type_landscapes")

    def init_xy(item):
        extrinsics = item["init_camera"]["extrinsics"]
        return (round(float(extrinsics[0][3]), 4), round(float(extrinsics[1][3]), 4))

    def record_xy(record):
        path = parse_sft_path(record)
        if not path:
            return None
        return (round(path[0][0], 4), round(path[0][1], 4))

    def match_dataset_item(record):
        xy = record_xy(record)
        if xy is None:
            return None
        hits = [
            item
            for item in dataset
            if item.get("task_type") == record.get("task_type")
            and item.get("task_description") == record.get("task_description")
            and init_xy(item) == xy
        ]
        return hits[0] if len(hits) == 1 else None

    preferred_ids = [
        "sft_000000",
        "sft_000098",
        "sft_000104",
        "sft_000108",
        "sft_000110",
        "sft_000137",
    ]
    sft_by_id = {row["id"]: row for row in sft_rows}
    showcase = []
    used_types = set()
    for sft_id in preferred_ids:
        record = sft_by_id.get(sft_id)
        if not record:
            continue
        item = match_dataset_item(record)
        if item is None:
            continue
        showcase.append((record, item))
        used_types.add(record["task_type"])
    for record in sft_rows:
        if record.get("task_type") in used_types:
            continue
        item = match_dataset_item(record)
        if item is None:
            continue
        showcase.append((record, item))
        used_types.add(record["task_type"])
        if len(showcase) >= 6:
            break

    if showcase:
        xmin, xmax, ymin, ymax = scene_bounds([item for _, item in showcase])
        xs = np.linspace(xmin, xmax, args.grid)
        ys = np.linspace(ymin, ymax, args.grid)
        fig, axes, cax = make_grid_figure()
        mesh = None
        for idx, (ax, (record, item)) in enumerate(zip(axes, showcase)):
            path = parse_sft_path(record)
            title = (
                f"{record['id']}  |  {TASK_TITLES.get(record['task_type'], record['task_type'])}\n"
                f"{wrap_title(record.get('task_description', ''), 40)}\n"
                f"score {record.get('initial_score', 0):.2f} -> {record.get('final_score', 0):.2f}"
            )
            mesh = plot_panel(ax, field, item, xs, ys, objects, path=path, title=title)
            style_axis(ax, xmin, xmax, ymin, ymax, xlabel=idx >= 3, ylabel=idx % 3 == 0)
        if mesh is not None:
            cbar = fig.colorbar(mesh, cax=cax)
            cbar.set_label("Potential score", fontsize=9)
        fig.legend(
            handles=[
                Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="black", label="init camera"),
                Line2D([0], [0], color="#7fd3ff", lw=2, label="SFT path"),
                Line2D([0], [0], marker="^", color="none", markerfacecolor="#7fd3ff", markeredgecolor="#113344", label="final pose"),
                Line2D([0], [0], color="#f4c430", lw=2, label="target region"),
            ],
            loc="lower center",
            ncol=4,
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.48, 0.02),
        )
        fig.suptitle(
            "SFT trajectories on the spatial potential field (output_0267_v4)",
            fontsize=15,
            fontweight="bold",
            y=0.965,
        )
        save_figure(fig, output_dir / "02_sft_trajectories_on_field")

        record, item = showcase[0]
        fig, ax = plt.subplots(figsize=(8.8, 7.4))
        path = parse_sft_path(record)
        mesh = plot_panel(
            ax,
            field,
            item,
            xs,
            ys,
            objects,
            path=path,
            title=(
                f"{record['id']}: {wrap_title(item.get('task_description'), 52)}\n"
                f"init {record.get('initial_score', 0):.3f} -> final {record.get('final_score', 0):.3f}"
            ),
        )
        style_axis(ax, xmin, xmax, ymin, ymax, xlabel=True, ylabel=True)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Potential score")
        fig.tight_layout()
        save_figure(fig, output_dir / "03_sft_000000_absolute_positioning")

    copies = [
        (output_dir / "01_task_type_landscapes.png", REPO_ROOT / "data_gen/active_spatial_sft/output_0267_v4/01_task_type_landscapes.png"),
        (output_dir / "02_sft_trajectories_on_field.png", REPO_ROOT / "data_gen/active_spatial_sft/output_0267_v4/02_sft_trajectories_on_field.png"),
        (output_dir / "03_sft_000000_absolute_positioning.png", REPO_ROOT / "data_gen/active_spatial_sft/output_0267_v4/03_sft_000000_absolute_positioning.png"),
        (output_dir / "01_task_type_landscapes.png", REPO_ROOT / "reports/active_spatial_v46_v50_cambrian_20260906_trained_plus10pp/figures/07_spatial_potential_field.png"),
        (output_dir / "01_task_type_landscapes.pdf", REPO_ROOT / "reports/active_spatial_v46_v50_cambrian_20260906_trained_plus10pp/figures/07_spatial_potential_field.pdf"),
    ]
    for src, dst in copies:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"copied {dst}")


if __name__ == "__main__":
    main()
