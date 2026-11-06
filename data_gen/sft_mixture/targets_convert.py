"""Group G of matrix v3 (the user's instruction of 2026-11-06, part 4.2): the action trajectories P2q in a targets
format, pool P2t.  Every assistant turn gets the targets segment before its action, labelled from the truth on the
frame the turn answers (vagen/envs/active_spatial/pointing.py, the RL env's own: the centre of each clearly visible
target's box, "none" for the others).  The labels are training targets only; nothing is added to what the model is
shown.  The system prompt and each user turn's format block become the targets format's (the RL env's text for that
format: release.protocol_system_prompt and prompt.targets_format_block with the task's objects); the rest of every
message is kept as it is.  "Only the short structure, no free text" (the lesson of group E): the assistant turns
are exactly the targets segment and the action segment.

    python -m data_gen.sft_mixture.targets_convert --rows ROWS --frames FRAMES --in P2q_train.jsonl \\
        --out-dir OUT --split train [--gs-root GS] [--workers 4]

writes OUT/<format>/P2t_<split>.jsonl for both coordinate formats (prompt.TARGET_FORMATS) and
OUT/P2t_<split>_stats.json; a row whose text is not the env's action_names text stops the run."""

from __future__ import annotations

import argparse
import collections
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ACTION_ONLY = re.compile(r"<action>(?:(?:move_forward|move_backward|move_left|move_right|turn_left|turn_right)\|)+</action>")
_INPUTS: Dict[str, Any] = {}


class ConvertError(ValueError):
    pass


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def frame_id(path: str) -> str:
    name = str(path).rsplit("/", 1)[-1]
    return name[:-4] if name.endswith(".png") else name


def texts(family: str, fmt: str) -> Dict[str, Any]:
    """The env's text: system prompts and format blocks of action_names and of ``fmt`` (cached per family)."""
    key = (family, fmt)
    cache = _INPUTS.setdefault("texts", {})
    if key not in cache:
        from data_gen.interiorgs_v10.admission import load_policy
        from data_gen.interiorgs_v10.release import _format_block, protocol_system_prompt
        from data_gen.interiorgs_v10.training_prep import with_format_example

        policy = with_format_example(load_policy(), "action_names")
        new = with_format_example(load_policy(), fmt)
        cache[key] = {"old_system": protocol_system_prompt(family, policy),
                      "new_system": protocol_system_prompt(family, new),
                      "old_block": {first: _format_block(policy, first) for first in (True, False)},
                      "max_actions": int(policy["sft"]["export"]["max_actions_per_turn"])}
    return cache[key]


def parts(row: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], List[str], str]:
    """(message key, messages, the image of each user turn, task id) of an LLaMA-Factory row (messages, images,
    source_id) or of an SFT conversation record (conversations, conversation_image_paths, source_task_id)."""
    if "messages" in row:
        return "messages", row["messages"], list(row.get("images") or []), row["source_id"]
    return "conversations", row["conversations"], list(row.get("conversation_image_paths") or []), row["source_task_id"]


def convert_row(row: Dict[str, Any], item: Dict[str, Any], occ: Any, formats: Sequence[str]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """({format: the converted row}, label statistics) of one P2q row or SFT conversation record."""
    from vagen.envs.active_spatial import pointing as PT
    from vagen.envs.active_spatial.prompt import targets_format_block

    family = item["task_type"]
    key, msgs, images, _ = parts(row)
    users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
    assistants = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    if msgs[0]["role"] != "system" or not (len(users) == len(assistants) == len(images)):
        raise ConvertError(f"{row.get('source_id')}: {len(users)} user, {len(assistants)} assistant turns, {len(images)} images")
    t0 = texts(family, formats[0])
    if msgs[0]["content"] != t0["old_system"]:
        raise ConvertError(f"{row.get('source_id')}: the system prompt is not the env's action_names prompt")
    refs = PT.target_refs(item)
    truths = []
    for k, (ui, ai) in enumerate(zip(users, assistants)):
        block = t0["old_block"][k == 0]
        if not msgs[ui]["content"].endswith("\n" + block) or msgs[ui]["content"].count(block) != 1:
            raise ConvertError(f"{row.get('source_id')}: user turn {k} does not end with the action_names format block")
        if ACTION_ONLY.fullmatch(msgs[ai]["content"]) is None:
            raise ConvertError(f"{row.get('source_id')}: assistant turn {k} is not an action segment")
        fid = frame_id(images[k])
        if fid not in _INPUTS["poses"]:
            raise ConvertError(f"{row.get('source_id')}: no pose for frame {fid}")
        truths.append(PT.truth(item, _INPUTS["poses"][fid], occ))
    out = {}
    for fmt in formats:
        t = texts(family, fmt)
        new = [dict(m) for m in msgs]
        new[0]["content"] = t["new_system"]
        for k, (ui, ai) in enumerate(zip(users, assistants)):
            old_block = t["old_block"][k == 0]
            head = new[ui]["content"][: -len(old_block)]
            new[ui]["content"] = head + targets_format_block(fmt, t["max_actions"], "|", k == 0, refs)
            new[ai]["content"] = PT.label(truths[k], fmt) + msgs[ai]["content"]
        out[fmt] = dict(row, **{key: new})
    stats = {"turns": len(truths), "targets": sum(len(x) for x in truths),
             "in_view": sum(t["in_view"] for x in truths for t in x),
             "turns_all_none": sum(not any(t["in_view"] for t in x) for x in truths)}
    return out, stats


def _scene_job(args: Tuple[str, List[int]]) -> Tuple[str, List[Tuple[int, Dict[str, Dict[str, Any]]]], Dict[str, int], float]:
    from vagen.envs.active_spatial import pointing as PT

    scene, idx = args
    t0 = time.time()
    items = [_INPUTS["rows"][parts(_INPUTS["in"][i])[3]] for i in idx]
    occ = PT.scene_occluders(_INPUTS["gs_root"], scene, _INPUTS["wall_model"], PT.scene_structure_sign(items))
    done, tot = [], collections.Counter()
    for i in idx:
        row = _INPUTS["in"][i]
        item = _INPUTS["rows"][parts(row)[3]]
        conv, st = convert_row(row, item, occ, _INPUTS["formats"])
        done.append((i, conv))
        tot.update(st)
    return scene, done, dict(tot), round(time.time() - t0, 1)


def main(argv: Sequence[str] = None) -> None:
    from vagen.envs.active_spatial.prompt import TARGET_FORMATS

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True, help="the task rows (rows.jsonl of the SFT folder)")
    ap.add_argument("--frames", required=True, help="frames.jsonl of the SFT folder (frame_id -> c2w)")
    ap.add_argument("--in", dest="inp", required=True,
                    help="P2q rows (sft_mixture build output) or SFT conversation records (sft_trajectories.jsonl)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", required=True, help="train / val (the output OUT/<format>/P2t_<split>.jsonl), or a name")
    ap.add_argument("--out-name", default=None, help="the output file name instead of P2t_<split>.jsonl")
    ap.add_argument("--gs-root", default="/mnt/umm/users/yinbaiqiao/InteriorGS_ply")
    ap.add_argument("--wall-model", default="thick")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    formats = list(TARGET_FORMATS)
    _INPUTS.update(rows={r["task_id"]: r for r in read_jsonl(Path(args.rows))},
                   poses={f["frame_id"]: f["c2w"] for f in read_jsonl(Path(args.frames))},
                   gs_root=str(args.gs_root), wall_model=args.wall_model, formats=formats)
    rows = read_jsonl(Path(args.inp))
    _INPUTS["in"] = rows
    missing = sorted({parts(r)[3] for r in rows if parts(r)[3] not in _INPUTS["rows"]})
    if missing:
        raise SystemExit(f"{len(missing)} tasks not in --rows, e.g. {missing[0]}")
    by_scene: Dict[str, List[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_scene[_INPUTS["rows"][parts(r)[3]]["scene_id"]].append(i)
    jobs = sorted(by_scene.items())
    converted: Dict[int, Dict[str, Dict[str, Any]]] = {}
    tot: collections.Counter = collections.Counter()
    t0 = time.time()
    if args.workers > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(args.workers) as pool:
            for k, (scene, done, st, sec) in enumerate(pool.imap_unordered(_scene_job, jobs, chunksize=1), 1):
                converted.update(done)
                tot.update(st)
                print(f"[{k}/{len(jobs)} {time.time() - t0:.0f}s] {scene}: {len(done)} rows, {sec} s", flush=True)
    else:
        for k, job in enumerate(jobs, 1):
            scene, done, st, sec = _scene_job(job)
            converted.update(done)
            tot.update(st)
            print(f"[{k}/{len(jobs)} {time.time() - t0:.0f}s] {scene}: {len(done)} rows, {sec} s", flush=True)
    name = args.out_name or f"P2t_{args.split}.jsonl"
    for fmt in formats:
        (out / fmt).mkdir(parents=True, exist_ok=True)
        with open(out / fmt / name, "w", encoding="utf-8", newline="\n") as fh:
            for i in range(len(rows)):  # the input's order
                fh.write(json.dumps(converted[i][fmt], ensure_ascii=False) + "\n")
    stats = {"input": str(args.inp), "output": name, "rows": len(rows), "scenes": len(jobs), "formats": formats,
             **dict(tot), "in_view_share": round(tot["in_view"] / max(1, tot["targets"]), 4),
             "turns_all_none_share": round(tot["turns_all_none"] / max(1, tot["turns"]), 4),
             "seconds": round(time.time() - t0, 1)}
    (out / f"{Path(name).stem}_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
