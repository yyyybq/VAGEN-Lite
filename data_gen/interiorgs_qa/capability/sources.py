"""Bind capability questions to immutable r10 formal trajectories and reachable states.

This CPU-only command reads existing RGB. It never renders, changes task labels,
or imports a model. Wall clearance is an additional sample filter, explicitly
separate from whether the environment permits the recorded trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from data_gen.interiorgs_v10.admission import chunk_certificate
from data_gen.interiorgs_v10.contract import ACTIONS, CAMERA_HEIGHT_M, apply_action
from data_gen.interiorgs_v10.qa_states import certificate_poses
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.run_scene import ignored_object_ids
from vagen.envs.active_spatial.dataset_contract import row_digest

SCENES = ("0929_841911", "0089_839957", "0948_841871", "0829_841616")
BASE_COMMIT = "168d443b"
VERSION = "capability_sources_r10_v1"
REPO = Path(__file__).resolve().parents[3]
GEN = "exps/vagen_active_spatial/interiorgs_v10_gen_r10_20261015"
DEPTH = "exps/vagen_active_spatial/interiorgs_v10_r10_20261015/depth_check/depth_check_summary.json"


def jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def pose_key(scene_id, c2w, intrinsics):
    """Scene/camera identity, independent of task, rounded only for deduplication."""
    def normalized(a):
        a = np.round(np.asarray(a, dtype=float), 8)
        a[a == 0] = 0  # Normalize negative zero for stable deduplication.
        return a.tolist()
    return _digest([scene_id, normalized(c2w), normalized(intrinsics), [256, 256]])


def load_task_scene(task, meta_root):
    """Load the task's frozen coordinate convention, r10 thick walls and exclusions."""
    sid = task["scene_id"]
    sign = task["collision_convention"]["structure_y_sign"]
    if sign not in (-1, 1):
        raise ValueError("Task has no frozen coordinate sign")
    other = task.get("structure_convention", {}).get("sign", sign)
    if other != sign:
        raise ValueError("Task coordinate conventions disagree")
    no_task = REPO / "data_gen/interiorgs_v10/configs/no_task_objects_20261014.json"
    return load_scene(Path(meta_root) / sid, sid, y_sign_override=float(sign),
                      no_task_ids=ignored_object_ids(str(no_task), sid), wall_model="thick")


def select_tasks(formal, published, train_scenes, flagged, scenes=SCENES):
    """Formal train tasks only; a reserve row cannot become a source by matching an ID."""
    wanted = set(scenes)
    if not wanted <= set(train_scenes):
        raise ValueError("A selected scene is not in the frozen train split")
    pub = {x["task_id"]: x for x in published if x["scene_id"] in wanted}
    if len(pub) != sum(x["scene_id"] in wanted for x in published):
        raise ValueError("Duplicate published task IDs")
    out, rejected = [], Counter()
    for row in formal:
        if row["scene_id"] not in wanted:
            continue
        if row["task_id"] in flagged:
            rejected["depth_excluded"] += 1
            continue
        p = pub.get(row["task_id"])
        if p is None:
            rejected["not_in_sft_published_rows"] += 1
            continue
        if p.get("reserve") is not False or p.get("split") != "train":
            rejected["not_formal_train"] += 1
            continue
        original = {k: v for k, v in p.items() if k not in ("split", "reserve")}
        if original != row:
            raise ValueError("Formal task differs from published trajectory task: " + row["task_id"])
        out.append(p)
    return out, rejected


def validate_trajectory(task, trajectory, frame_rows, rgb_root, max_per_turn):
    """Check actions, grouped turns, every recorded pose and every RGB/metadata binding."""
    tid = task["task_id"]
    if (trajectory["source_task_id"] != tid or trajectory.get("parent_task_id") != tid
            or trajectory["scene_id"] != task["scene_id"] or trajectory.get("reserve") is not False
            or trajectory.get("split") != "train" or trajectory.get("success") is not True):
        raise ValueError("Trajectory is not this successful formal train task: " + tid)
    if trajectory["source_task_sha256"] != row_digest(task):
        raise ValueError("Trajectory task hash mismatch: " + tid)
    cert = list(task["difficulty"]["certificate"])
    if not cert or any(a not in ACTIONS for a in cert):
        raise ValueError("Invalid certificate actions")
    poses = certificate_poses(task)
    trace = trajectory["primitive_trajectory_trace"]
    if len(trace) != len(poses):
        raise ValueError("Primitive trace length mismatch")
    for k, (step, pose) in enumerate(zip(trace, poses)):
        if (step["step"] != k or step["action"] != (cert[k - 1] if k else None)
                or not np.allclose(step["pose_c2w"], pose, atol=1e-9, rtol=0)):
            raise ValueError("Primitive trace pose/action mismatch")
    chunks = chunk_certificate(cert, trajectory["chunking"], max_per_turn)
    expected = []
    boundary = 0
    blocked = trajectory.get("blocked_insert")
    for i, chunk in enumerate(chunks):
        if blocked and blocked["turn"] == i:
            expected.append((boundary, [blocked["move"]], True))
        expected.append((boundary, chunk, False))
        boundary += len(chunk)
    conv = trajectory["conversations"]
    if not conv or conv[0]["role"] != "system" or len(conv) != 1 + 2 * len(expected):
        raise ValueError("Conversation turn structure mismatch")
    input_paths = []
    for i, (k, actions, _) in enumerate(expected):
        user, assistant = conv[1 + i * 2:3 + i * 2]
        target = "<action>" + "".join(a + "|" for a in actions) + "</action>"
        expected_path = Path(rgb_root) / f"{tid}__s{k}.png"
        if (user["role"] != "user" or assistant["role"] != "assistant"
                or assistant["content"].strip() != target
                or Path(user["image_path"]) != expected_path):
            raise ValueError("Grouped actions or conversation image binding mismatch")
        input_paths.append(str(expected_path))
    if input_paths != trajectory["conversation_image_paths"]:
        raise ValueError("Conversation image list mismatch")
    all_paths = input_paths.copy()
    terminal = str(Path(rgb_root) / f"{tid}__s{len(cert)}.png")
    if terminal not in all_paths:
        all_paths.append(terminal)
    if all_paths != trajectory["image_paths"]:
        raise ValueError("Rendered trajectory image list mismatch")
    records = []
    for path in dict.fromkeys(all_paths):
        p = Path(path)
        frame = frame_rows[p.stem]
        k = int(frame["state_id"][1:])
        if (frame["frame_id"] != p.stem or frame["task_id"] != tid
                or frame["scene_id"] != task["scene_id"] or frame["state_id"] != f"s{k}"
                or not np.allclose(frame["c2w"], poses[k], atol=1e-9, rtol=0)
                or frame["camera_item"]["init_camera"] != task["init_camera"]):
            raise ValueError("Rendered frame camera/task binding mismatch")
        with Image.open(p) as im:
            im.load()
            if im.size != (256, 256):
                raise ValueError("RGB resolution differs from Act-SFT")
        records.append((k, p, frame, sha256(p), input_paths.count(path)))
    return poses, records, expected


def check_path(scene, poses):
    """Runtime legality and extra sample clearance are separate, cumulative checks."""
    result, runtime_reasons, sample_reasons = [], [], []
    for k, pose in enumerate(poses):
        p = np.asarray(pose, float)
        reasons = []
        if p.shape != (4, 4) or not np.isfinite(p).all():
            reasons.append("invalid_pose")
        else:
            if abs(p[2, 3] - CAMERA_HEIGHT_M) > 1e-6 or abs(p[2, 2]) > 1e-6:
                reasons.append("noncanonical_camera")
            if scene.room_of(float(p[0, 3]), float(p[1, 3])) is None:
                reasons.append("outside_rooms")
            hit = scene.detector.check_collision(p[:3, 3])
            if hit.has_collision:
                reasons.append("collision_" + str(hit.collision_type))
            if k:
                ok, why = scene.transition_legal(poses[k - 1], p)
                if not ok:
                    reasons.append("transition_" + why)
        runtime_reasons.extend({"step": k, "reason": r} for r in reasons)
        clearance = float(scene.wall_distance(float(p[0, 3]), float(p[1, 3])))
        if clearance < scene.min_wall_distance:
            sample_reasons.append({"step": k, "reason": "sample_wall_clearance", "distance_m": clearance,
                                   "required_m": float(scene.min_wall_distance)})
        result.append({"runtime_legal": not runtime_reasons,
                       "sample_eligible": not runtime_reasons and not sample_reasons,
                       "runtime_rejections": list(runtime_reasons),
                       "sample_rejections": list(sample_reasons),
                       "current_wall_clearance_m": clearance if np.isfinite(clearance) else None,
                       "room_index": scene.room_of(float(p[0, 3]), float(p[1, 3]))})
    return result


def build_task_states(task, trajectory, frame_rows, rgb_root, scene, max_per_turn):
    poses, existing, expected = validate_trajectory(task, trajectory, frame_rows, rgb_root, max_per_turn)
    checks = check_path(scene, poses)
    cert = task["difficulty"]["certificate"]
    tid, sid = task["task_id"], task["scene_id"]
    K = task["init_camera"]["intrinsics"]
    out, used, rejected = [], set(), Counter()
    def state(k, p, actions, kind, frame_id, rgb, digest, checks_for_state, **extra):
        return {"state_id": f"{tid}__{k}", "scene_id": sid, "parent_task_id": tid,
                "c2w": np.asarray(p).tolist(), "actions_from_initial": list(actions),
                "source_kind": kind, "existing_rgb": str(rgb) if rgb else None,
                "frame_id": frame_id, "existing_rgb_sha256": digest,
                "trajectory_parent_id": trajectory["id"], "pose_key": pose_key(sid, p, K),
                "native_intrinsics": K, "split": "train", **checks_for_state, **extra}
    for k, path, frame, digest, count in existing:
        if not checks[k]["runtime_legal"]:
            raise ValueError("Existing trajectory fails runtime path legality: " + tid)
        record = state(f"s{k}", poses[k], cert[:k], "act_sft_input" if count else "act_sft_terminal",
                       frame["frame_id"], path, digest, checks[k], original_input_occurrences=count,
                       certificate_step=k)
        out.append(record)
        used.add(record["pose_key"])
    # Only one layer from the existing rendered trajectory states; never recurse.
    for k, _, _, _, _ in existing:
        for action in ACTIONS:
            candidate = apply_action(poses[k], action)
            key = pose_key(sid, candidate, K)
            if key in used:
                rejected["same_parent_duplicate_pose"] += 1
                continue
            full_path = poses[:k + 1] + [candidate]
            check = check_path(scene, full_path)[-1]
            if not check["runtime_legal"]:
                rejected[check["runtime_rejections"][-1]["reason"]] += 1
                continue
            if not check["sample_eligible"]:
                rejected["sample_wall_clearance_prefix"] += 1
                continue
            used.add(key)
            out.append(state(f"n{k}_{action}", candidate, cert[:k] + [action], "legal_extension",
                             None, None, None, check, extension_from=f"{tid}__s{k}", extension_action=action))
    audit = {"parent_task_id": tid, "trajectory_parent_id": trajectory["id"],
             "certificate_actions": len(cert), "validated_conversation_turns": len(expected),
             "existing_states": len(existing), "extension_attempts": len(existing) * len(ACTIONS),
             "extensions_accepted": sum(s["source_kind"] == "legal_extension" for s in out),
             "extension_rejections": dict(rejected),
             "near_wall_primitive_states": [{"step": k, "distance_m": c["current_wall_clearance_m"]}
                 for k, c in enumerate(checks) if c["current_wall_clearance_m"] is not None
                 and c["current_wall_clearance_m"] < scene.min_wall_distance],
             "existing_states_sample_excluded": sum(not s["sample_eligible"] for s in out)}
    return out, audit


def epoch_budget(path):
    """Count exported rows/image references; this is not a token or training execution count."""
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "available": False}
    rows, inputs, paths = 0, 0, set()
    for row in jsonl(path):
        rows += 1
        inputs += len(row["images"])
        paths.update(row["images"])
    return {"path": str(path), "available": True, "sha256": sha256(path), "trajectory_records": rows,
            "image_input_occurrences": inputs, "unique_image_paths": len(paths),
            "scope": "one traversal of the exported file, before tokenizer/truncation; no training run"}


def build(source_root, out):
    source_root, out = Path(source_root).resolve(), Path(out).resolve()
    targets = [out / name for name in ("tasks.jsonl", "states.jsonl", "source_inventory.json")]
    if any(p.exists() for p in targets):
        raise FileExistsError("Use a fresh output; source artifacts are immutable")
    root = source_root / GEN
    sft = root / "sft"
    paths = {"formal_tasks": root / "output/tasks.jsonl", "split": root / "split_manifest.json",
             "depth_exclusion": source_root / DEPTH, "published_rows": sft / "rows.jsonl",
             "trajectories": sft / "sft_trajectories_formal.jsonl", "frames": sft / "frames.jsonl",
             "render_manifest": sft / "frames/render_manifest.json",
             "r10_export_policy": sft / "job/src/data_gen/interiorgs_v10/configs/admission_v2.json"}
    split = json.loads(paths["split"].read_text())
    flagged = set(json.loads(paths["depth_exclusion"].read_text())["flagged_task_ids"])
    tasks, rejected = select_tasks(list(jsonl(paths["formal_tasks"])), list(jsonl(paths["published_rows"])),
                                   split["train_scenes"], flagged)
    if len(tasks) != 56 or any(sum(t["scene_id"] == s for t in tasks) != 14 for s in SCENES):
        raise ValueError("Frozen four-scene formal source no longer contains 56 tasks")
    tids = {t["task_id"] for t in tasks}
    trajs = [t for t in jsonl(paths["trajectories"]) if t["source_task_id"] in tids]
    trajectories = {t["source_task_id"]: t for t in trajs}
    if len(trajectories) != len(trajs) or set(trajectories) != tids:
        raise ValueError("Missing or duplicate formal source trajectories")
    frs = [f for f in jsonl(paths["frames"]) if f["task_id"] in tids]
    frames = {f["frame_id"]: f for f in frs}
    if len(frames) != len(frs):
        raise ValueError("Duplicate frame metadata")
    manifest = json.loads(paths["render_manifest"].read_text())
    rendered = {x["frame_id"]: x for x in manifest["rendered"]}
    for fid in frames:
        if fid not in rendered or rendered[fid]["file"] != fid + ".png":
            raise ValueError("Frame absent from actual rendering result: " + fid)
    policy = json.loads(paths["r10_export_policy"].read_text())["sft"]["export"]
    meta_root = source_root.parent / "InteriorGS_meta"
    states, audits, assets = [], [], {}
    for sid in SCENES:
        subset = [t for t in tasks if t["scene_id"] == sid]
        scene = load_task_scene(subset[0], meta_root)
        if len({t["collision_convention"]["structure_y_sign"] for t in subset}) != 1:
            raise ValueError("Mixed coordinate conventions in one scene")
        assets[sid] = {name: {"path": str(meta_root / sid / name), "sha256": sha256(meta_root / sid / name)}
                       for name in ("labels.json", "structure.json")}
        for task in subset:
            rows, audit = build_task_states(task, trajectories[task["task_id"]], frames, sft / "frames/plain",
                                           scene, int(policy["max_actions_per_turn"]))
            states.extend(rows)
            audits.append(audit)
        print(json.dumps({"scene_id": sid, "tasks": len(subset),
                          "states": sum(s["scene_id"] == sid for s in states)}, ensure_ascii=False), flush=True)
    if len({s["state_id"] for s in states}) != len(states):
        raise ValueError("Duplicate output state IDs")
    actual_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    base = subprocess.check_output(["git", "rev-parse", BASE_COMMIT], cwd=REPO, text=True).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", base, actual_head], cwd=REPO, check=True)
    code_files = (Path(__file__), REPO / "data_gen/interiorgs_v10/contract.py", REPO / "data_gen/interiorgs_v10/scene.py",
                  REPO / "data_gen/interiorgs_v10/admission.py", REPO / "data_gen/interiorgs_v10/qa_states.py",
                  REPO / "vagen/envs/active_spatial/collision_detector.py")
    inventory = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                 "source_root": str(source_root), "scenes": list(SCENES), "split": "train", "tasks": len(tasks),
                 "source_files": {k: {"path": str(p), "sha256": sha256(p)} for k, p in paths.items()},
                 "source_task_hashes": {t["task_id"]: row_digest(t) for t in tasks}, "assets": assets,
                 "code": {"base_commit": base, "executed_checkout_head": actual_head,
                          "file_sha256": {str(p.relative_to(REPO)): sha256(p) for p in code_files}},
                 "counts": {"states": len(states), "by_source": dict(Counter(s["source_kind"] for s in states)),
                            "sample_eligible_by_source": dict(Counter(s["source_kind"] for s in states if s["sample_eligible"])),
                            "render_pose_keys": len({s["pose_key"] for s in states}),
                            "existing_rgb_states": sum(s["existing_rgb"] is not None for s in states),
                            "near_wall_primitive_states": sum(len(a["near_wall_primitive_states"]) for a in audits),
                            "task_rejections": dict(rejected)},
                 "path_checks": audits, "wall_model": "thick", "sample_wall_clearance_m": 0.5,
                 "clearance_note": "0.5 m is an extra conservative sample filter, not the runtime collision rule. Existing RGB states remain recorded; the pipeline must filter sample_eligible. A prefix with insufficient clearance also fails this extra filter.",
                 "extension_note": "Only one atomic-action layer from each existing rendered trajectory state; full initial action prefix is checked. Different parent tasks retain their own association, with shared pose_key for rendering reuse.",
                 "qa_labels_computed": False, "rgb_rendered": False, "model_run": False,
                 "target_visibility_and_reference_validity": "deferred_to_capability_pipeline",
                 "epoch_budgets": {name: epoch_budget(source_root / f"exps/vagen_active_spatial/interiorgs_v10_train_{tag}/lf_formal/trajectory_train.jsonl")
                                   for name, tag in (("r10", "r10_20261015"), ("r11_if_available", "r11_20261016"))}}
    out.mkdir(parents=True, exist_ok=True)
    for path, rows in ((targets[0], tasks), (targets[1], states)):
        with path.open("x") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    inventory["output_sha256"] = {p.name: sha256(p) for p in targets[:2]}
    with targets[2].open("x") as f:
        json.dump(inventory, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source_root, args.out)
    print(json.dumps(result["counts"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
