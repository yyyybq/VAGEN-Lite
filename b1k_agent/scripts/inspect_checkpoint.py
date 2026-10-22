"""Summarise an openpi checkpoint directory without loading it (stdlib only: no JAX, no openpi, no GPU).

    python scripts/inspect_checkpoint.py /nas/baiqiao/checkpoints/pi05_turn_on_the_radio [--out ckpt.json] [--depth 3]

What it reports (read-only):
  tree        directories and files down to --depth, with sizes (directory sizes from os.walk, capped at
              --max-files files per directory)
  norm stats  every assets/<asset_id>/norm_stats.json: per key (state, actions) the length of mean / std /
              q01 / q99 and a short preview. <asset_id> is the --repo-id the executor server needs
              (openpi policy_config.py loads <ckpt>/assets/<repo_id>/norm_stats.json).
  metadata    orbax _CHECKPOINT_METADATA and other small *.json / *.yaml / *.txt files (e.g. wandb_id.txt)
  inferred    train config name (only if some file records it; openpi itself does not store it, so the path
              outputs/checkpoints/<config>/<exp>/<step> is reported as a hint), the --policy.dir to use, the
              number of params files, step directories, and the action / state dims from the norm stats,
              checked against the R1Pro layouts below.

Layouts checked (sources):
  action 23  official R1Pro action (OmniGibson/omnigibson/eval/utils/eval_utils.py ACTION_QPOS_INDICES, copied in
             agent/fields.py ACTION_SLICES): base 0:3 velocity, torso 3:7, left arm 7:14, left gripper 14,
             right arm 15:22, right gripper 22.
  state 23   openpi b1k state (wensi-ai/openpi src/openpi/configs/robots/b1k.py R1Pro.proprio and
             policies/b1k_policy.py extract_state_from_proprio): base_qvel 3 + trunk_qpos 4 + left arm qpos 7 +
             left gripper width 1 (sum of the two finger positions) + right arm qpos 7 + right gripper width 1.
  Norm stats are computed after the data transforms (scripts/compute_norm_stats.py) and LeRobotB1KDataConfig has
  extra_delta_transform=True, so the torso and arm entries of "actions" are DELTAS relative to the current joint
  state, not absolute targets (training/config.py _build_delta_mappings). Normalisation is mean/std
  (use_quantile_norm=False); the model pads state and actions to 32 dims after normalisation.

Status: written and unit-tested on a synthetic checkpoint tree (tests/test_ops_inspect_checkpoint.py); not yet run
on the real checkpoint on 105.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Optional

ACTION_LAYOUT_R1PRO = [("base", 0, 3), ("torso", 3, 7), ("left_arm", 7, 14), ("left_gripper", 14, 15),
                       ("right_arm", 15, 22), ("right_gripper", 22, 23)]
STATE_LAYOUT_OPENPI_B1K = [("base_qvel", 0, 3), ("trunk_qpos", 3, 7), ("left_arm_qpos", 7, 14),
                           ("left_gripper_width", 14, 15), ("right_arm_qpos", 15, 22), ("right_gripper_width", 22, 23)]
PROPRIO_DIM_R1PRO = 61
MODEL_PAD_DIM = 32  # openpi Pi0Config.action_dim
STATS = ("mean", "std", "q01", "q99")
SMALL_FILE_BYTES = 2_000_000
META_SUFFIXES = (".json", ".yaml", ".yml", ".txt")
CONFIG_KEYS = ("config_name", "train_config", "exp_name", "policy_config", "config")


def human(n: Optional[float]) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TiB"


def dir_size(path: str, max_files: int) -> dict:
    """Bytes and file count under ``path`` (symlinks not followed); ``truncated`` when the cap was hit."""
    total, files, truncated = 0, 0, False
    for root, _dirs, names in os.walk(path):
        for name in names:
            if files >= max_files:
                truncated = True
                break
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                continue
            files += 1
        if truncated:
            break
    return {"bytes": total, "files": files, "truncated": truncated}


def tree(root: str, depth: int = 3, max_files: int = 200_000, max_entries: int = 60) -> list[dict]:
    out: list[dict] = []

    def visit(path: str, level: int) -> None:
        try:
            names = sorted(os.listdir(path))
        except OSError as e:
            out.append({"path": os.path.relpath(path, root), "kind": "error", "error": str(e)})
            return
        for i, name in enumerate(names):
            if i >= max_entries:
                out.append({"path": os.path.relpath(os.path.join(path, "..."), root), "kind": "more",
                            "count": len(names) - max_entries})
                break
            p = os.path.join(path, name)
            rel = os.path.relpath(p, root)
            if os.path.islink(p):
                out.append({"path": rel, "kind": "link", "target": os.readlink(p)})
            elif os.path.isdir(p):
                out.append({"path": rel, "kind": "dir", "level": level, **dir_size(p, max_files)})
                if level < depth:
                    visit(p, level + 1)
            else:
                try:
                    size = os.lstat(p).st_size
                except OSError:
                    size = None
                out.append({"path": rel, "kind": "file", "level": level, "bytes": size})

    visit(root, 1)
    return out


def walk_files(root: str, max_depth: int = 8, max_files: int = 200_000):
    """Yield file paths under root down to max_depth (directory levels below root)."""
    seen = 0
    base = root.rstrip(os.sep).count(os.sep)
    for cur, dirs, names in os.walk(root):
        if cur.count(os.sep) - base >= max_depth:
            dirs[:] = []
        for name in names:
            seen += 1
            if seen > max_files:
                return
            yield os.path.join(cur, name)


def _shape(v: Any) -> list[int]:
    shape = []
    while isinstance(v, list):
        shape.append(len(v))
        v = v[0] if v else None
    return shape


def _flat(v: Any) -> list[float]:
    if isinstance(v, list):
        return [x for item in v for x in _flat(item)]
    return [v] if isinstance(v, (int, float)) and not isinstance(v, bool) else []


def summarize_norm_stats(path: str, preview: int = 4) -> dict:
    """Per key of norm_stats.json ({"norm_stats": {key: {mean, std, q01, q99}}}, openpi shared/normalize.py)."""
    out: dict = {"path": path, "keys": {}}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    stats = data.get("norm_stats", data) if isinstance(data, dict) else {}
    if not isinstance(stats, dict):
        out["error"] = "unexpected format (no norm_stats mapping)"
        return out
    for key, entry in sorted(stats.items()):
        if not isinstance(entry, dict):
            continue
        rec = {}
        for stat in STATS:
            v = entry.get(stat)
            if v is None:
                rec[stat] = None
                continue
            flat = _flat(v)
            rec[stat] = {"shape": _shape(v), "len": len(flat),
                         "preview": [round(x, 5) for x in flat[:preview]],
                         "min": round(min(flat), 5) if flat else None, "max": round(max(flat), 5) if flat else None}
        out["keys"][key] = rec
    return out


def stat_dim(ns: dict, key: str) -> Optional[int]:
    rec = ns.get("keys", {}).get(key) or {}
    lens = {s["len"] for s in rec.values() if isinstance(s, dict)}
    return lens.pop() if len(lens) == 1 else None


def layout_note(kind: str, dim: Optional[int]) -> str:
    if dim is None:
        return "unknown (no consistent length in the norm stats)"
    if dim == 23:
        layout = ACTION_LAYOUT_R1PRO if kind == "action" else STATE_LAYOUT_OPENPI_B1K
        return "23 = R1Pro layout: " + ", ".join(f"{n} {a}:{b}" if b - a > 1 else f"{n} {a}" for n, a, b in layout)
    if dim == MODEL_PAD_DIM:
        return f"{dim}: padded to the model dim; the first 23 should follow the R1Pro layout (verify)"
    if kind == "state" and dim == PROPRIO_DIM_R1PRO:
        return "61 = the raw evaluator proprio vector, not the openpi b1k 23-D state (different data config)"
    return f"{dim}: does NOT match the R1Pro 23-D layout; check the train config before serving"


def read_small(path: str) -> dict:
    rec: dict = {"path": path}
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return {**rec, "error": str(e)}
    rec["bytes"] = size
    if size > SMALL_FILE_BYTES:
        rec["skipped"] = "too large"
        return rec
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError) as e:
        return {**rec, "error": f"{type(e).__name__}: {e}"}
    try:
        data = json.loads(text)
        rec["json_keys"] = sorted(data)[:40] if isinstance(data, dict) else None
        rec["data"] = data if len(text) <= 4000 else None
    except ValueError:
        rec["text"] = text[:2000]
    return rec


def find_config_values(obj: Any, out: list, where: str, depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in CONFIG_KEYS and isinstance(v, str) and len(v) <= 200:
                out.append({"key": str(k), "value": v, "source": where})
            find_config_values(v, out, where, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:50]:
            find_config_values(v, out, where, depth + 1)


def path_hint(path: str) -> Optional[dict]:
    """openpi writes outputs/checkpoints/<config>/<exp>/<step>; report that pattern when the path has it.

    The numeric <step> is required: /nas/baiqiao/checkpoints/<name>/<step> is a copied checkpoint whose <name> is not
    a train config, and reading it as one would suggest a --policy.config that does not exist."""
    parts = os.path.abspath(path).split(os.sep)
    if "checkpoints" in parts:
        i = len(parts) - 1 - parts[::-1].index("checkpoints")
        rest = parts[i + 1:]
        if len(rest) >= 3 and rest[2].isdigit():
            return {"config": rest[0], "exp": rest[1], "step": rest[2],
                    "source": "path pattern outputs/checkpoints/<config>/<exp>/<step> (a hint, not a record)"}
    for part in reversed(parts[-3:]):  # the checkpoint dir, or its parent when ``path`` is a step dir
        m = re.search(r"(pi0[a-z0-9_]*)", part)
        if m:
            return {"config": None, "name_fragment": m.group(1),
                    "source": "directory name only (not a train config name)"}
    return None


def inspect(root: str, depth: int = 3, max_files: int = 200_000) -> dict:
    root = os.path.abspath(os.path.expanduser(root))
    summary: dict = {"checkpoint": root, "inspected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "exists": os.path.isdir(root)}
    if not summary["exists"]:
        summary["error"] = f"{root} is not a directory"
        return summary
    summary["total"] = dir_size(root, max_files)
    summary["tree"] = tree(root, depth=depth, max_files=max_files)

    files = list(walk_files(root, max_depth=8, max_files=max_files))
    norm = []
    for p in sorted(f for f in files if os.path.basename(f) == "norm_stats.json"):
        parts = os.path.relpath(p, root).split(os.sep)
        if "assets" in parts:
            i = len(parts) - 1 - parts[::-1].index("assets")
            asset_id, owner = "/".join(parts[i + 1:-1]), os.sep.join(parts[:i]) or "."
        else:
            asset_id, owner = None, os.sep.join(parts[:-1]) or "."
        norm.append({"asset_id": asset_id, "checkpoint_subdir": owner, **summarize_norm_stats(p)})
    summary["norm_stats"] = norm

    meta, configs = [], []
    for p in sorted(files):
        name = os.path.basename(p)
        rel_depth = os.path.relpath(p, root).count(os.sep)
        if name == "norm_stats.json" or rel_depth > 3:
            continue
        if name in ("_CHECKPOINT_METADATA", "_METADATA") or name.endswith(META_SUFFIXES):
            rec = read_small(p)
            rec["path"] = os.path.relpath(p, root)
            find_config_values(rec.get("data"), configs, rec["path"])
            meta.append(rec)
    summary["metadata_files"] = meta

    params = [f for f in files if "params" in os.path.relpath(f, root).split(os.sep)[:-1]]
    safetensors = [os.path.relpath(f, root) for f in files if f.endswith(".safetensors")]
    steps = sorted(int(d) for d in os.listdir(root) if d.isdigit() and os.path.isdir(os.path.join(root, d)))
    policy_dir = None
    if os.path.isdir(os.path.join(root, "params")) or any(os.path.dirname(s) == "" for s in safetensors):
        policy_dir = root
    elif steps:
        last = os.path.join(root, str(steps[-1]))
        if os.path.isdir(os.path.join(last, "params")) or os.path.isfile(os.path.join(last, "model.safetensors")):
            policy_dir = last

    action_dims = sorted({d for d in (stat_dim(n, "actions") for n in norm) if d is not None})
    state_dims = sorted({d for d in (stat_dim(n, "state") for n in norm) if d is not None})
    action_dim = action_dims[0] if len(action_dims) == 1 else None
    state_dim = state_dims[0] if len(state_dims) == 1 else None
    asset_ids = sorted({n["asset_id"] for n in norm if n.get("asset_id")})
    summary["inferred"] = {
        "policy_dir": policy_dir,
        "format": "pytorch (model.safetensors)" if safetensors else ("orbax (params/)" if params else "unknown"),
        "params_files": len(params),
        "safetensors_files": safetensors,
        "step_dirs": steps,
        "asset_ids": asset_ids,
        "repo_id_for_server": asset_ids[0] if len(asset_ids) == 1 else None,
        "train_config_recorded": configs or None,
        "train_config_path_hint": path_hint(policy_dir or root),
        "train_config_note": "openpi does not store the train config name in a checkpoint (checkpoints.py save_state: "
                             "assets, train_state, params); pass it with --policy.config (pi05_b1k for the official "
                             "baseline)",
        "action_dim": action_dim,
        "action_dims_seen": action_dims,
        "action_layout": layout_note("action", action_dim) if norm else "no norm_stats.json found",
        "state_dim": state_dim,
        "state_dims_seen": state_dims,
        "state_layout": layout_note("state", state_dim) if norm else "no norm_stats.json found",
        "actions_are_deltas_for": "torso, left_arm, right_arm (pi05_b1k extra_delta_transform=True)" if action_dim == 23
        else None,
    }
    return summary


def render(summary: dict) -> str:
    lines = [f"# Checkpoint {summary['checkpoint']}", ""]
    if not summary.get("exists"):
        return "\n".join(lines + [f"ERROR: {summary.get('error')}"]) + "\n"
    tot = summary["total"]
    lines.append(f"total {human(tot['bytes'])} in {tot['files']} files{' (capped)' if tot['truncated'] else ''}")
    lines += ["", "## Tree", "```"]
    for e in summary["tree"]:
        indent = "  " * max(e.get("level", 1) - 1, 0)
        if e["kind"] == "dir":
            cap = "+" if e["truncated"] else ""
            lines.append(f"{indent}{os.path.basename(e['path'])}/  ({e['files']}{cap} files, {human(e['bytes'])})")
        elif e["kind"] == "file":
            lines.append(f"{indent}{os.path.basename(e['path'])}  {human(e['bytes'])}")
        elif e["kind"] == "link":
            lines.append(f"{e['path']} -> {e['target']}")
        elif e["kind"] == "more":
            lines.append(f"{e['path']}  ({e['count']} more entries)")
        else:
            lines.append(f"{e['path']}: {e.get('error')}")
    lines += ["```", "", "## Norm stats"]
    if not summary["norm_stats"]:
        lines.append("no norm_stats.json found (the executor server cannot load this checkpoint without one)")
    for n in summary["norm_stats"]:
        lines.append(f"- asset id `{n['asset_id']}` (under `{n['checkpoint_subdir']}`): {n['path']}")
        if n.get("error"):
            lines.append(f"  - ERROR {n['error']}")
        for key, rec in n["keys"].items():
            parts = [f"{s}: len {r['len']} {r['preview']}..." if r else f"{s}: -" for s, r in rec.items()]
            lines.append(f"  - {key}: " + "; ".join(parts))
    lines += ["", "## Metadata files"]
    for m in summary["metadata_files"]:
        what = m.get("error") or m.get("skipped") or (f"keys {m['json_keys']}" if m.get("json_keys") else
                                                       (m.get("text") or "").strip().splitlines()[:1])
        lines.append(f"- {m['path']} ({human(m.get('bytes'))}): {what}")
    if not summary["metadata_files"]:
        lines.append("(none)")
    lines += ["", "## Inferred"]
    for k, v in summary["inferred"].items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="openpi checkpoint dir (the --policy.dir, or the experiment dir with step dirs)")
    ap.add_argument("--out", default=None, help="write the JSON summary here")
    ap.add_argument("--md", default=None, help="also write the text summary here")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--max-files", type=int, default=200_000, help="cap on files counted per directory walk")
    args = ap.parse_args(argv)
    summary = inspect(args.checkpoint, depth=args.depth, max_files=args.max_files)
    text = render(summary)
    # files first: takeover_check.sh pipes stdout through head, which may close the pipe early
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True, default=str)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(text)
    print(text, end="")
    if args.out:
        print(f"wrote {args.out}")
    return 0 if summary.get("exists") else 1


if __name__ == "__main__":
    sys.exit(main())
