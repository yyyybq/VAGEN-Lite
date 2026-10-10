#!/usr/bin/env python3
"""Visualize whether the runtime score guides useful Active Spatial paths.

The dashboard deliberately says "score-guided" rather than "optimal": beam
search over a discrete action space is a useful diagnostic, but it is not a
proof of globally shortest-path optimality.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


CANVAS = (1600, 980)
BG = (248, 250, 252)
INK = (25, 32, 45)
MUTED = (95, 105, 120)
BLUE = (38, 110, 210)
GREEN = (25, 155, 90)
RED = (210, 65, 65)
GRID = (215, 221, 230)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size)
    except Exception:
        return ImageFont.load_default()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _score_values(record: Dict[str, Any]) -> List[float]:
    trace = record.get("primitive_trajectory_trace") or record.get("trajectory_trace") or []
    if trace:
        return [float(row["score"]) for row in trace]
    return [float(record.get("initial_score", 0.0)), float(record.get("final_score", 0.0))]


def trajectory_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    trace = record.get("primitive_trajectory_trace") or record.get("trajectory_trace") or []
    scores = _score_values(record)
    deltas = [after - before for before, after in zip(scores, scores[1:])]
    positions = [row.get("position_xyz") for row in trace if row.get("position_xyz")]
    path_length = 0.0
    for before, after in zip(positions, positions[1:]):
        path_length += math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(after, before)))
    displacement = 0.0
    if len(positions) >= 2:
        displacement = math.sqrt(
            sum((float(a) - float(b)) ** 2 for a, b in zip(positions[-1], positions[0]))
        )
    reward_trace = record.get("trajectory_trace") or []
    rewards = [row.get("runtime_reward") for row in reward_trace]
    rewards = [float(value) for value in rewards if value is not None]
    terminal_reward = (
        ((record.get("terminal_reward_trace") or {}).get("accounting") or {}).get(
            "actual_total_reward"
        )
    )
    if terminal_reward is not None:
        rewards.append(float(terminal_reward))
    final_gates = (trace[-1].get("gates") or {}) if trace else {}
    reference = record.get("optimality_reference") or {}
    certified_length = (
        int(reference["certified_lower_bound"])
        if reference.get("shortest_length_certified")
        else None
    )
    action_count = int(record.get("total_actions", 0))
    return {
        "id": record.get("id"),
        "scene_id": record.get("scene_id"),
        "task_type": record.get("task_type"),
        "success": bool(record.get("success")),
        "frames": len(record.get("trajectory_trace") or []) or len(scores),
        "score_points": len(scores),
        "turns": int(record.get("trajectory_steps", max(0, len(scores) - 1))),
        "actions": action_count,
        "initial_score": scores[0],
        "final_score": scores[-1],
        "best_score": max(scores),
        "net_score_gain": scores[-1] - scores[0],
        "largest_score_drop": min(0.0, min(deltas, default=0.0)),
        "monotonic_transition_fraction": (
            sum(delta >= -1e-9 for delta in deltas) / len(deltas) if deltas else 1.0
        ),
        "positive_transition_fraction": (
            sum(delta > 1e-9 for delta in deltas) / len(deltas) if deltas else 0.0
        ),
        "best_score_at_final_frame": scores[-1] >= max(scores) - 1e-9,
        "path_length_m": path_length,
        "straight_line_displacement_m": displacement,
        "path_directness": displacement / path_length if path_length > 1e-9 else 1.0,
        "runtime_reward_total": sum(rewards),
        "final_gate_failures": sorted(key for key, value in final_gates.items() if not value),
        "certified_shortest_action_count": certified_length,
        "action_gap_to_certified_shortest": (
            action_count - certified_length if certified_length is not None else None
        ),
        "matches_certified_shortest_length": (
            action_count == certified_length if certified_length is not None else None
        ),
    }


def _panel(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], title: str) -> None:
    draw.rounded_rectangle(box, radius=14, fill=(255, 255, 255), outline=GRID, width=2)
    draw.text((box[0] + 16, box[1] + 12), title, fill=INK, font=_font(20, bold=True))


def _draw_frames(canvas: Image.Image, record: Dict[str, Any], root: Path) -> None:
    draw = ImageDraw.Draw(canvas)
    box = (30, 135, 1570, 475)
    _panel(draw, box, "Rendered trajectory (selected frames)")
    paths = record.get("primitive_image_paths") or record.get("image_paths") or []
    trace = (
        record.get("primitive_trajectory_trace")
        if record.get("primitive_image_paths")
        else record.get("trajectory_trace")
    ) or []
    if not paths:
        draw.text((50, 200), "No saved RGB frames", fill=RED, font=_font(20))
        return
    count = min(6, len(paths))
    indices = sorted({round(i * (len(paths) - 1) / max(count - 1, 1)) for i in range(count)})
    tile_w, tile_h, gap = 232, 235, 18
    x = 48
    for frame_index in indices:
        path = Path(paths[frame_index])
        path = path if path.is_absolute() else root / path
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((tile_w, 176))
            tile = Image.new("RGB", (tile_w, 176), (235, 238, 243))
            tile.paste(image, ((tile_w - image.width) // 2, (176 - image.height) // 2))
            canvas.paste(tile, (x, 180))
        except Exception:
            draw.rectangle((x, 180, x + tile_w, 356), fill=(245, 225, 225), outline=RED)
            draw.text((x + 10, 250), "missing frame", fill=RED, font=_font(16))
        score = None
        actions: List[str] = []
        success = False
        if frame_index < len(trace):
            score = trace[frame_index].get("score")
            action = trace[frame_index].get("action_from_previous")
            actions = (
                [action]
                if action
                else trace[frame_index].get("actions_from_previous") or []
            )
            success = bool(trace[frame_index].get("canonical_success"))
        label = f"frame {frame_index}  score={float(score):.3f}" if score is not None else f"frame {frame_index}"
        draw.text((x, 365), label, fill=GREEN if success else INK, font=_font(15, bold=success))
        action_text = " <- " + " | ".join(actions) if actions else "initial view"
        if len(action_text) > 31:
            action_text = action_text[:28] + "..."
        draw.text((x, 391), action_text, fill=MUTED, font=_font(13))
        x += tile_w + gap


def _draw_score_curve(draw: ImageDraw.ImageDraw, record: Dict[str, Any]) -> None:
    box = (30, 500, 790, 930)
    _panel(draw, box, "Gate-aligned shaping score by primitive action")
    chart = (85, 570, 750, 855)
    for tick in range(6):
        y = chart[3] - tick * (chart[3] - chart[1]) / 5
        draw.line((chart[0], y, chart[2], y), fill=GRID, width=1)
        draw.text((48, y - 8), f"{tick / 5:.1f}", fill=MUTED, font=_font(12))
    scores = _score_values(record)
    x_for = lambda i: chart[0] + i * (chart[2] - chart[0]) / max(len(scores) - 1, 1)
    y_for = lambda value: chart[3] - max(0.0, min(1.0, value)) * (chart[3] - chart[1])
    points = [(x_for(index), y_for(value)) for index, value in enumerate(scores)]
    if len(points) > 1:
        draw.line(points, fill=BLUE, width=4)
    for index, (x, y) in enumerate(points):
        color = GREEN if index == len(points) - 1 and record.get("success") else BLUE
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline=(255, 255, 255), width=2)
        draw.text((x - 10, chart[3] + 12), str(index), fill=MUTED, font=_font(11))
    metrics = trajectory_metrics(record)
    text = (
        f"gain {metrics['net_score_gain']:+.3f}   "
        f"monotonic {metrics['monotonic_transition_fraction']:.0%}   "
        f"largest drop {metrics['largest_score_drop']:+.3f}"
    )
    draw.text((85, 885), text, fill=INK, font=_font(15))


def _draw_topdown(draw: ImageDraw.ImageDraw, record: Dict[str, Any]) -> None:
    box = (815, 500, 1570, 930)
    _panel(draw, box, "Top-down camera path (XY)")
    trace = record.get("primitive_trajectory_trace") or record.get("trajectory_trace") or []
    poses = [row.get("pose_c2w") for row in trace if row.get("pose_c2w")]
    if not poses:
        draw.text((850, 620), "No pose trace in this record", fill=RED, font=_font(18))
        return
    xs = [float(pose[0][3]) for pose in poses]
    ys = [float(pose[1][3]) for pose in poses]
    margin = 0.5
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin
    if x1 - x0 < 1.0:
        x0 -= 0.5
        x1 += 0.5
    if y1 - y0 < 1.0:
        y0 -= 0.5
        y1 += 0.5
    plot = (865, 570, 1520, 855)
    sx = lambda value: plot[0] + (value - x0) / (x1 - x0) * (plot[2] - plot[0])
    sy = lambda value: plot[3] - (value - y0) / (y1 - y0) * (plot[3] - plot[1])
    points = [(sx(x), sy(y)) for x, y in zip(xs, ys)]
    draw.rectangle(plot, outline=GRID, width=2)
    if len(points) > 1:
        draw.line(points, fill=BLUE, width=4)
    for index, (point, pose) in enumerate(zip(points, poses)):
        color = GREEN if index == len(points) - 1 and record.get("success") else BLUE
        draw.ellipse((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), fill=color)
        forward = (float(pose[0][2]), float(pose[1][2]))
        norm = math.hypot(*forward)
        if norm > 1e-8:
            end = (point[0] + 22 * forward[0] / norm, point[1] - 22 * forward[1] / norm)
            draw.line((point[0], point[1], end[0], end[1]), fill=color, width=3)
        draw.text((point[0] + 7, point[1] - 17), str(index), fill=INK, font=_font(11))
    metrics = trajectory_metrics(record)
    text = (
        f"path {metrics['path_length_m']:.2f} m   displacement "
        f"{metrics['straight_line_displacement_m']:.2f} m   "
        f"directness {metrics['path_directness']:.2f}"
    )
    draw.text((865, 875), text, fill=INK, font=_font(15))
    if metrics["certified_shortest_action_count"] is not None:
        gap = metrics["action_gap_to_certified_shortest"]
        color = GREEN if gap == 0 else RED
        draw.text(
            (865, 905),
            f"certified shortest={metrics['certified_shortest_action_count']} actions; "
            f"generated gap={gap:+d}",
            fill=color,
            font=_font(15, bold=True),
        )


def create_dashboard(record: Dict[str, Any], sft_root: Path, output: Path) -> Path:
    canvas = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(canvas)
    status = "SUCCESS" if record.get("success") else "PARTIAL / FAILED"
    status_color = GREEN if record.get("success") else RED
    draw.text((30, 25), str(record.get("id", "unknown")), fill=INK, font=_font(30, bold=True))
    draw.text((30, 69), str(record.get("task_description", ""))[:150], fill=MUTED, font=_font(18))
    draw.text((1360, 30), status, fill=status_color, font=_font(22, bold=True))
    contract = record.get("score_contract") or {}
    versions = ", ".join(contract.get("search_score_versions") or ["unknown"])
    draw.text((30, 103), f"score: {versions}", fill=MUTED, font=_font(14))
    _draw_frames(canvas, record, sft_root)
    _draw_score_curve(draw, record)
    _draw_topdown(draw, record)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, "PNG")
    return output


def summarize(records: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    metrics = [trajectory_metrics(record) for record in records]
    certified = [row for row in metrics if row["certified_shortest_action_count"] is not None]
    by_task: Dict[str, Counter] = defaultdict(Counter)
    for row in metrics:
        bucket = by_task[str(row.get("task_type") or "unknown")]
        bucket["records"] += 1
        bucket["successes"] += int(row["success"])
    summary = {
        "schema_version": "active_spatial_score_guidance_summary_v1",
        "interpretation": (
            "Diagnostics measure whether discrete score-guided search improves the current "
            "R1 score and reaches canonical gates. Global shortest-length claims are made only "
            "for rows whose audit manifest has a complete, equal lower/upper certificate."
        ),
        "records": len(metrics),
        "successes": sum(row["success"] for row in metrics),
        "success_rate": mean([float(row["success"]) for row in metrics]) if metrics else None,
        "mean_actions": mean([row["actions"] for row in metrics]) if metrics else None,
        "mean_net_score_gain": mean([row["net_score_gain"] for row in metrics]) if metrics else None,
        "mean_monotonic_transition_fraction": (
            mean([row["monotonic_transition_fraction"] for row in metrics]) if metrics else None
        ),
        "best_at_final_rate": (
            mean([float(row["best_score_at_final_frame"]) for row in metrics]) if metrics else None
        ),
        "fully_monotonic_rate": (
            mean([
                float(row["monotonic_transition_fraction"] >= 1.0 - 1e-12)
                for row in metrics
            ])
            if metrics else None
        ),
        "mean_path_directness": mean([row["path_directness"] for row in metrics]) if metrics else None,
        "mean_runtime_reward_total": (
            mean([row["runtime_reward_total"] for row in metrics]) if metrics else None
        ),
        "certified_shortest_comparison": {
            "records": len(certified),
            "matches": sum(bool(row["matches_certified_shortest_length"]) for row in certified),
            "match_rate": (
                mean([float(row["matches_certified_shortest_length"]) for row in certified])
                if certified else None
            ),
            "mean_extra_actions": (
                mean([row["action_gap_to_certified_shortest"] for row in certified])
                if certified else None
            ),
        },
        "by_task": {
            key: {
                **dict(value),
                "success_rate": value["successes"] / value["records"],
            }
            for key, value in sorted(by_task.items())
        },
    }
    return summary, metrics


def visualize(
    sft_jsonl: Path,
    output_dir: Path,
    max_dashboards: int = 50,
) -> Dict[str, Any]:
    summary_path = output_dir / "score_guidance_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite visualization: {summary_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir = output_dir / "dashboards"
    dashboard_dir.mkdir(exist_ok=True)
    records = _read_jsonl(sft_jsonl)
    summary, metrics = summarize(records)
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "trajectory_metrics.jsonl").open("x", encoding="utf-8") as handle:
        for row in metrics:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    dashboard_records = records if max_dashboards < 0 else records[:max_dashboards]
    links = []
    for record in dashboard_records:
        name = f"{record.get('id', 'unknown')}.png"
        create_dashboard(record, sft_jsonl.parent, dashboard_dir / name)
        links.append((record, name, trajectory_metrics(record)))
    with (output_dir / "index.html").open("x", encoding="utf-8") as handle:
        cards = []
        for record, name, metric in links:
            certified = metric["certified_shortest_action_count"] is not None
            search_text = " ".join(
                str(value) for value in (
                    record.get("id", ""), record.get("scene_id", ""),
                    record.get("task_type", ""), record.get("task_description", ""),
                )
            ).lower()
            cards.append(
                "<article class='trajectory' data-search='{search}' data-certified='{certified}'>"
                "<a href='dashboards/{name}'><img loading='lazy' src='dashboards/{name}' "
                "alt='Trajectory dashboard for {ident}'></a>"
                "<div class='meta'><b>{ident}</b><span class='pill {status_class}'>{status}</span>"
                "<span>{actions} actions</span><span>gain {gain:+.3f}</span>"
                "<span>monotonic {mono:.0%}</span><span>{certificate}</span></div></article>".format(
                    name=html.escape(name),
                    ident=html.escape(str(record.get("id", "unknown"))),
                    search=html.escape(search_text, quote=True),
                    certified="yes" if certified else "no",
                    status="success" if record.get("success") else "failed",
                    status_class="ok" if record.get("success") else "bad",
                    actions=metric["actions"],
                    gain=metric["net_score_gain"],
                    mono=metric["monotonic_transition_fraction"],
                    certificate=(
                        "certified shortest"
                        if metric["matches_certified_shortest_length"]
                        else "no complete shortest certificate"
                    ),
                )
            )
        certified = summary["certified_shortest_comparison"]
        handle.write(
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Active Spatial · R1 score-guided trajectories</title>"
            "<style>:root{color-scheme:light;--ink:#19202d;--muted:#667085;--line:#d7dde6;"
            "--bg:#f6f8fb;--card:#fff;--blue:#266ed2;--green:#199b5a}*{box-sizing:border-box}"
            "body{font:15px/1.5 Inter,ui-sans-serif,system-ui,sans-serif;margin:0;background:var(--bg);"
            "color:var(--ink)}main{max-width:1280px;margin:auto;padding:42px 24px 80px}h1{font-size:36px;"
            "margin:0 0 8px}.subtitle{color:var(--muted);max-width:900px}.kpis{display:grid;"
            "grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:28px 0}.kpi{"
            "background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}.kpi b{"
            "display:block;font-size:26px}.kpi span{color:var(--muted)}.tools{position:sticky;top:0;"
            "z-index:2;display:flex;gap:12px;flex-wrap:wrap;background:rgba(246,248,251,.94);"
            "backdrop-filter:blur(8px);padding:14px 0}input,select{font:inherit;padding:10px 12px;"
            "border:1px solid var(--line);border-radius:9px;background:#fff}input{flex:1;min-width:240px}"
            ".trajectory{background:var(--card);border:1px solid var(--line);border-radius:14px;"
            "overflow:hidden;margin:20px 0;box-shadow:0 5px 20px rgba(25,32,45,.05)}img{display:block;"
            "width:100%}.meta{display:flex;gap:13px;align-items:center;flex-wrap:wrap;padding:13px 16px}"
            ".meta span{color:var(--muted)}.pill{padding:2px 8px;border-radius:99px}.pill.ok{"
            "background:#e9f8f0;color:#137a45}.pill.bad{background:#fdecec;color:#b42318}.links a{"
            "color:var(--blue);margin-right:16px}.hidden{display:none}</style></head><body><main>"
            "<h1>Active Spatial · R1 score-guided trajectories</h1>"
            "<p class='subtitle'>Every panel combines primitive RGB views, the gate-aligned shaping "
            "score, and the top-down camera path. "
            f"{html.escape(summary['interpretation'])}</p>"
            "<div class='kpis'>"
            f"<div class='kpi'><b>{summary['records']}</b><span>trajectories</span></div>"
            f"<div class='kpi'><b>{summary['success_rate']:.1%}</b><span>canonical success</span></div>"
            f"<div class='kpi'><b>{summary['mean_actions']:.2f}</b><span>mean actions</span></div>"
            f"<div class='kpi'><b>{summary['mean_net_score_gain']:+.3f}</b><span>mean score gain</span></div>"
            f"<div class='kpi'><b>{summary['fully_monotonic_rate']:.1%}</b><span>fully monotonic</span></div>"
            f"<div class='kpi'><b>{certified['matches']}/{certified['records']}</b>"
            "<span>certified shortest matches</span></div></div>"
            "<p class='links'><a href='score_guidance_summary.json'>Summary JSON</a>"
            "<a href='trajectory_metrics.jsonl'>Trajectory metrics JSONL</a></p>"
            "<div class='tools'><input id='search' type='search' placeholder='Search ID, scene, task…'>"
            "<select id='certificate'><option value='all'>All trajectories</option>"
            "<option value='yes'>Certified shortest only</option>"
            "<option value='no'>Without complete certificate</option></select>"
            "<span id='visible'></span></div><section id='trajectories'>"
            + "".join(cards)
            + "</section></main><script>const q=document.querySelector('#search'),c=document.querySelector('#certificate'),"
            "cards=[...document.querySelectorAll('.trajectory')],v=document.querySelector('#visible');"
            "function filter(){const s=q.value.toLowerCase(),k=c.value;let n=0;cards.forEach(x=>{const show="
            "x.dataset.search.includes(s)&&(k==='all'||x.dataset.certified===k);x.classList.toggle('hidden',!show);"
            "n+=show});v.textContent=n+' shown'}q.addEventListener('input',filter);c.addEventListener('change',filter);"
            "filter()</script></body></html>"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create RGB + score curve + top-down path dashboards from SFT output."
    )
    parser.add_argument("--sft-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-dashboards", type=int, default=50)
    args = parser.parse_args()
    summary = visualize(
        Path(args.sft_jsonl).resolve(),
        Path(args.output_dir).resolve(),
        args.max_dashboards,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
