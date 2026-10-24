"""C0's SFT data in the new format (the user's decision of 2026-10-24: SFT in the new format, format example
"action_names", the admission policy's default since the round-12 freeze), for every model of the line.

    python -m model_matrix.sft.newformat link --source R11_SFT_DIR --out NEW_DIR
    python -m data_gen.interiorgs_v10.training_prep sft-records --dir NEW_DIR      # the export, with today's policy
    python -m model_matrix.sft.newformat compare --source R11_SFT_DIR --out NEW_DIR
    python -m model_matrix.sft.newformat subset --out NEW_DIR --train 400 --val 40 --name sft_trajectories_smoke.jsonl

``link``: the new folder links the source's inputs (rows, frame list, frames, split manifest, task registry), so the
export writes only into the new folder.  ``compare``: every exported conversation must equal the source's after the
exact rewrite of its format text (training_prep.format_sft_records' rule: the system prompt and each user turn's
format block) -- nothing else may differ.  ``subset``: a seeded, family-proportional draw for the smoke SFT.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import Counter
from pathlib import Path

INPUTS = ("rows.jsonl", "frames.jsonl", "frames", "split_manifest.json", "task_registry.jsonl")
PATH_FIELDS = ("conversation_image_paths", "image_paths", "primitive_image_paths")
SEED = 20261024


def cmd_link(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    for name in INPUTS:
        if not (src / name).exists():
            raise SystemExit(f"{src / name} is missing")
        (out / name).symlink_to((src / name).resolve())
    print(json.dumps({"linked": list(INPUTS), "out": str(out)}))


def _jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def rewrite_rule(policy_old, policy_new):
    from data_gen.interiorgs_v10.release import _format_block, protocol_system_prompt

    blocks = {first: (_format_block(policy_old, first), _format_block(policy_new, first)) for first in (True, False)}

    def rewrite(rec):
        conv = copy.deepcopy(rec["conversations"])
        t = str(rec.get("task_type") or "")
        if conv[0]["role"] != "system" or conv[0]["content"] != protocol_system_prompt(t, policy_old):
            return None, "system_prompt_differs"
        conv[0]["content"] = protocol_system_prompt(t, policy_new)
        users = [m for m in conv if m["role"] == "user"]
        for k, m in enumerate(users):
            old, new = blocks[k == 0]
            if not m["content"].endswith("\n" + old) or m["content"].count(old) != 1:
                return None, "format_text_differs"
            m["content"] = m["content"][: -len(old)] + new
        return conv, None

    return rewrite


def cmd_compare(src: Path, out: Path) -> None:
    from data_gen.interiorgs_v10.admission import load_policy
    from data_gen.interiorgs_v10.release import format_example

    new_policy = load_policy()
    old_policy = copy.deepcopy(new_policy)
    old_policy["sft"]["export"]["format_example"] = "placeholder"
    rewrite = rewrite_rule(old_policy, new_policy)
    old = {r["id"]: r for r in _jsonl(src / "sft_trajectories.jsonl")}
    new = {r["id"]: r for r in _jsonl(out / "sft_trajectories.jsonl")}
    report = {"source": len(old), "export": len(new), "only_in_source": sorted(set(old) - set(new))[:20],
              "only_in_export": sorted(set(new) - set(old))[:20], "format_example": format_example(new_policy),
              "same_after_rewrite": 0, "differ": Counter(), "examples": []}
    def same(key, a, b):  # image paths point through the new folder's links to the same files
        if key in PATH_FIELDS and isinstance(a, list) and isinstance(b, list):
            return [os.path.realpath(x) for x in a] == [os.path.realpath(x) for x in b]
        return a == b

    def real(conv):  # a message's frame, through the links, is the same file
        return [dict(m, image_path=os.path.realpath(m["image_path"])) if m.get("image_path") else m for m in conv]

    for rid in sorted(set(old) & set(new)):
        conv, why = rewrite(old[rid])
        if why is None and real(conv) == real(new[rid]["conversations"]):
            other = {k for k in set(old[rid]) | set(new[rid]) if k not in ("conversations", "format_example")
                     and not same(k, old[rid].get(k), new[rid].get(k))}
            if not other:
                report["same_after_rewrite"] += 1
                continue
            why = "fields:" + ",".join(sorted(other))
        report["differ"][why or "conversation_differs"] += 1
        if len(report["examples"]) < 5:
            report["examples"].append(rid)
    report["differ"] = dict(report["differ"])
    report["ok"] = (report["same_after_rewrite"] == len(old) == len(new))
    (out / "newformat_compare.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


def cmd_subset(out: Path, train: int, val: int, name: str) -> None:
    recs = _jsonl(out / "sft_trajectories.jsonl")
    rng = random.Random(SEED)
    chosen = []
    for split, want in (("train", train), ("val", val)):
        pool = sorted((r for r in recs if r.get("split") == split), key=lambda r: r["id"])
        rng.shuffle(pool)
        fams = sorted({str(r.get("task_type")) for r in pool})
        quota = {f: round(want * sum(1 for r in pool if str(r.get("task_type")) == f) / len(pool)) for f in fams}
        picked = [r for f in fams for r in [x for x in pool if str(x.get("task_type")) == f][:quota[f]]]
        picked += [r for r in pool if r not in picked][: max(0, want - len(picked))]
        chosen += picked[:want]
    (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen))
    print(json.dumps({"file": str(out / name), "seed": SEED, "by_split": dict(Counter(r["split"] for r in chosen)),
                      "by_family": dict(Counter(r["task_type"] for r in chosen)),
                      "turns": dict(Counter(r["trajectory_steps"] for r in chosen))}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("link", "compare"):
        p = sub.add_parser(c)
        p.add_argument("--source", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("subset")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train", type=int, default=400)
    p.add_argument("--val", type=int, default=40)
    p.add_argument("--name", default="sft_trajectories_smoke.jsonl")
    a = ap.parse_args()
    if a.cmd == "link":
        cmd_link(a.source, a.out)
    elif a.cmd == "compare":
        cmd_compare(a.source, a.out)
    else:
        cmd_subset(a.out, a.train, a.val, a.name)


if __name__ == "__main__":
    main()
