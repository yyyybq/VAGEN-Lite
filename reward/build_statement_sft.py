"""Reward line, phase 2 direction 3 (PLAN_DIR3_20261103.md section 2, the user's decisions 1-5 甲 of 2026-11-03):
the statement SFT data from C0's SFT conversations.  CPU only; the C0 folder is only read.

Every conversation is rewritten the way training_prep.format_sft_records rewrites the format example: the system
prompt and each user turn's format text must be exactly the ones the admission policy's SFT export writes
(release.protocol_system_prompt, release._format_block) and are replaced by the same policy's texts with prompt
format "observe" (vagen/envs/active_spatial/prompt.py), so the RL env with prompt_format observe shows the same
text; a conversation whose texts differ is left out and counted.  Each assistant turn gets the statement of the
frame its user turn showed (vagen/statement_truth.py at that frame's c2w from frames.jsonl, the scene's occluders as
the env builds them) in front of its unchanged <action> block.

Checks: every user turn's frame has a pose; each new answer's statement parses back to its truth and its action
block is the old answer's; conversations, turns and splits as in the source.

    python -m reward.build_statement_sft --sft C0_SFT_DIR --out OUT [--gs-root DIR] [--limit N] [--workers 4]

OUT/sft_trajectories.jsonl, OUT/build_report.json, and links to the source's frames, frames.jsonl, rows.jsonl,
split_manifest.json and task_registry.jsonl (the folder works as a chain's sft_dir).
"""

import argparse
import copy
import json
import re
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

GS_ROOT = "/mnt/umm/users/yinbaiqiao/InteriorGS_ply"
LINKS = ("frames", "frames.jsonl", "rows.jsonl", "split_manifest.json", "task_registry.jsonl")
ACTION = re.compile(r"<action>.*?</action>", re.S)


def truths(args):
    """The statement truth of every listed frame of one scene's conversations."""
    gs_root, scene_id, jobs = args
    from vagen.envs.active_spatial.collision_detector import CollisionDetector
    from vagen.envs.active_spatial.target_visibility import OccluderSet
    from vagen.statement_truth import truth

    det = CollisionDetector()
    det.load_scene_from_gs_root(gs_root, scene_id)
    occ = OccluderSet.from_detector(det)
    out = {}
    for row, frames in jobs:
        for fid, c2w in frames:
            try:
                t = truth(row, c2w, occ)
                out[fid] = {k: t[k] for k in ("where", "face", "dist", "dist_m", "missing", "success")}
            except Exception as exc:
                out[fid] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gs-root", default=GS_ROOT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)  # the login node: at most 4 cores a task
    args = ap.parse_args()
    from data_gen.interiorgs_v10.admission import load_policy
    from data_gen.interiorgs_v10.release import _format_block, protocol_system_prompt
    from vagen.statement_truth import DIST_EDGES_M, format_statement, parse_statement

    sft, out = Path(args.sft), Path(args.out)
    if out.resolve() == sft.resolve():
        raise SystemExit("--out must not be the source folder")
    out.mkdir(parents=True, exist_ok=True)
    policy = load_policy()
    new_policy = copy.deepcopy(policy)
    new_policy["sft"]["export"]["prompt_format"] = "observe"
    old_blocks = {first: _format_block(policy, first) for first in (True, False)}
    new_blocks = {first: _format_block(new_policy, first) for first in (True, False)}

    rows = {}
    for line in (sft / "rows.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["task_id"]] = r
    c2w = {}
    with open(sft / "frames.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                c2w[r["frame_id"]] = r["c2w"]
    records = []
    with open(sft / "sft_trajectories.jsonl") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if args.limit and len(records) >= args.limit:
                    break

    def frame_of(m):
        return Path(m["image_path"]).stem

    by_scene = defaultdict(list)
    no_pose = Counter()
    for rec in records:
        frames = [frame_of(m) for m in rec["conversations"] if m.get("role") == "user"]
        no_pose[rec["id"]] = sum(fid not in c2w for fid in frames)
        by_scene[rows[rec["source_task_id"]]["scene_id"]].append(
            (rows[rec["source_task_id"]], [(fid, c2w[fid]) for fid in frames if fid in c2w]))
    jobs = [(args.gs_root, s, js[k:k + 50]) for s, js in sorted(by_scene.items()) for k in range(0, len(js), 50)]
    truth_of = {}
    with Pool(args.workers) as pool:
        for part in pool.imap_unordered(truths, jobs):
            truth_of.update(part)

    skipped, fields = Counter(), defaultdict(Counter)
    written, turns, splits, chars = [], 0, Counter(), []
    for rec in records:
        conv = copy.deepcopy(rec["conversations"])
        t = str(rec.get("task_type") or "")
        if no_pose[rec["id"]]:
            skipped["frame_without_pose"] += 1
            continue
        if conv[0]["role"] != "system" or conv[0]["content"] != protocol_system_prompt(t, policy):
            skipped["system_prompt_differs"] += 1
            continue
        conv[0]["content"] = protocol_system_prompt(t, new_policy)
        ok, seen, k = True, None, 0
        for m in conv[1:]:
            if m["role"] == "user":
                first = k == 0
                k += 1
                if not m["content"].endswith("\n" + old_blocks[first]) or m["content"].count(old_blocks[first]) != 1:
                    skipped["format_text_differs"] += 1
                    ok = False
                    break
                m["content"] = m["content"][: -len(old_blocks[first])] + new_blocks[first]
                seen = truth_of.get(frame_of(m))
            elif m["role"] == "assistant":
                if seen is None or "error" in seen:
                    skipped["no_truth" if seen is None else "truth_error"] += 1
                    ok = False
                    break
                old = m["content"]
                statement = format_statement(seen)
                m["content"] = statement + old
                back = parse_statement(m["content"])
                if back is None or any(back[f] != seen[f] for f in ("where", "face", "dist", "missing")) \
                        or ACTION.findall(m["content"]) != ACTION.findall(old):
                    skipped["statement_round_trip"] += 1
                    ok = False
                    break
                for f in ("dist",):
                    fields[f][seen[f]] += 1
                fields["missing"][",".join(seen["missing"]) or "none"] += 1
                for v in seen["where"].values():
                    fields["where"][v] += 1
                for v in seen["face"].values():
                    fields["face"]["yes" if v else "no"] += 1
                chars.append(len(statement))
                turns += 1
                seen = None  # one statement per user turn
        if not ok:
            continue
        written.append(dict(rec, conversations=conv, prompt_format="observe", statement=True))
        splits[rec.get("split")] += 1
    with open(out / "sft_trajectories.jsonl", "w") as f:
        for rec in written:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    for name in LINKS:
        src, dst = sft / name, out / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())
    report = {"source": str(sft), "source_conversations": len(records), "written": len(written),
              "skipped": dict(skipped), "by_split": dict(splits), "assistant_turns": turns,
              "source_assistant_turns": sum(1 for r in records for m in r["conversations"] if m["role"] == "assistant"),
              "dist_edges_m": list(DIST_EDGES_M), "statement_chars_mean": round(sum(chars) / max(1, len(chars)), 1),
              "fields": {k: dict(v.most_common()) for k, v in fields.items()},
              "by_family": dict(Counter(r.get("task_type") for r in written))}
    (out / "build_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
