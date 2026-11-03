#!/usr/bin/env python3
"""N2 (few-shot; the user's decisions 2-3 of 2026-11-02 and 1 of 2026-11-03): the evaluation instances and the
demonstration datasets.

    python downstream/n/n2_data.py instances --from N1/instances_final.jsonl N1B/instances_final.jsonl --out DIR \
        [--smoke-per-family 10]
    python downstream/n/n2_data.py build --gen GEN --out DIR [--sizes 100 300 1000] [--max-per-turn 3] \
        [--exclude DIR/instances_600.jsonl] [--seed 20261103]

instances: N1's and its second set's instances as one file (DIR/instances_600.jsonl, ids checked unique) and the
  smoke's subset, the first --smoke-per-family of each type in file order (DIR/instances_smoke.jsonl).
build: the demonstrations from GEN (n_gen.py with --path-frames: one instance per type and scene, its shortest path
  with camera A's frame at every pose, cut at the first success).  A demonstration is kept when its last frame
  succeeds, none of its frames is colour noise (n_frames.corrupted: a broken render state of the generation, found
  2026-11-03) and its scene is none of --exclude's (the evaluation's; stops otherwise).  One fixed order (--seed): each
  type's demonstrations shuffled, then the types taken in turn (n_env.FAMILIES order), a type that runs out
  (occlusion) skipped from then on; each size is the first n of that order, so the sizes are nested and every prefix
  is as even over the types as the counts allow.
  The actions are cut into turns as the main line's SFT export does (data_gen/interiorgs_v10/admission.py
  chunk_certificate "action_runs": one turn per run of the same action, a run longer than --max-per-turn split; the
  user's cap of 3 of 2026-11-03).  Each demonstration becomes one multi-turn conversation in the main line's
  sharegpt form (system, then user / assistant per turn), with the text each model generation sees in N1's
  evaluation (n_episode.system_text / user_text with that generation's validation config: qwen25 = n_episode.VAL_ENV,
  qwen3 = VAL_ENV_QWEN3): the first user turn the initial observation with the format example, later ones "Action
  executed." (a shortest path has no blocked move), each with its frame and the pose line; the reply
  "<action>a|b|</action>".
  Writes DIR/order.json (the order, each demonstration's actions and turns), DIR/<qwen25|qwen3>/demos_<n>.jsonl
  ({"id", "task_type", "messages", "images"} per line, image paths absolute) and DIR/summary.json (per size: per type
  counts, scenes, tasks, turns, actions per turn).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import n_env as N  # noqa: E402
import n_frames as F  # noqa: E402

SEED = 20261103
SIZES = (100, 300, 1000)
VARIANTS = ("qwen25", "qwen3")
FEEDBACK = "Action executed."


def read_jsonl(path) -> list:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_jsonl(path: Path, rows) -> int:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)
    return len(rows)


# ------------------------------------------------------------------------------------------- instances
def instances(paths, out: Path, smoke_per_family: int = 10) -> dict:
    rows = [r for p in paths for r in read_jsonl(p)]
    ids = [r["instance_id"] for r in rows]
    dup = sorted(k for k, v in collections.Counter(ids).items() if v > 1)
    if dup:
        raise SystemExit(f"{len(dup)} instance ids occur twice (e.g. {dup[:3]})")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "instances_600.jsonl", rows)
    smoke = [r for t in N.FAMILIES for r in [x for x in rows if x["task_type"] == t][:smoke_per_family]]
    write_jsonl(out / "instances_smoke.jsonl", smoke)
    return {"instances": len(rows), "per_type": dict(collections.Counter(r["task_type"] for r in rows)),
            "smoke": len(smoke), "sources": [str(p) for p in paths]}


# ---------------------------------------------------------------------------------------- demonstrations
def chunk_action_runs(actions, max_per_turn: int = 3) -> list:
    """The main line's "action_runs" chunking (admission.chunk_certificate)."""
    out = []
    for a in actions:
        if out and out[-1][-1] == a and len(out[-1]) < max_per_turn:
            out[-1].append(a)
        else:
            out.append([a])
    return out


def usable(inst: dict) -> bool:
    pf = inst.get("path_frames") or []
    return len(pf) >= 2 and bool(pf[-1]["success"]) and not any(f["success"] for f in pf[:-1])


def demo_order(demos: list, seed: int = SEED) -> list:
    """Each type's demonstrations shuffled (by id first, so the input order does not matter), the types in turn."""
    rng = random.Random(seed)
    pools = {}
    for t in N.FAMILIES:
        pool = sorted((d for d in demos if d["task_type"] == t), key=lambda d: d["instance_id"])
        rng.shuffle(pool)
        pools[t] = pool
    out, i = [], 0
    while any(i < len(p) for p in pools.values()):
        out += [pools[t][i] for t in N.FAMILIES if i < len(pools[t])]
        i += 1
    return out


def conversation(inst: dict, gen: Path, cfg, max_per_turn: int, E) -> dict:
    """One demonstration as a sharegpt conversation in one generation's evaluation text (see the module text)."""
    pf = inst["path_frames"]
    actions = [f["action"] for f in pf[1:]]
    desc = inst["item"]["task_description"]
    msgs = [{"role": "system", "content": E.system_text(cfg, inst["task_type"])}]
    images, k = [], 0
    for t, chunk in enumerate(chunk_action_runs(actions, max_per_turn)):
        frame = pf[k]
        msgs.append({"role": "user", "content": E.user_text(cfg, t == 0, tuple(frame["pose"]), desc,
                                                             "" if t == 0 else FEEDBACK)})
        msgs.append({"role": "assistant", "content": "<action>" + "".join(a + "|" for a in chunk) + "</action>"})
        images.append(str(Path(gen) / frame["image"]))
        k += len(chunk)
    return {"id": inst["instance_id"], "task_type": inst["task_type"], "messages": msgs, "images": images}


def size_summary(chosen: list, max_per_turn: int) -> dict:
    turns = [chunk_action_runs([f["action"] for f in d["path_frames"][1:]], max_per_turn) for d in chosen]
    return {"demos": len(chosen), "per_type": {t: sum(d["task_type"] == t for d in chosen) for t in N.FAMILIES},
            "scenes": len({d["scene_id"] for d in chosen}), "tasks": len({d["task"] for d in chosen}),
            "turns": sum(len(x) for x in turns),
            "turns_per_demo": dict(sorted(collections.Counter(len(x) for x in turns).items())),
            "actions": sum(len(c) for x in turns for c in x),
            "actions_per_turn": dict(sorted(collections.Counter(len(c) for x in turns for c in x).items()))}


def build(gen: Path, out: Path, sizes=SIZES, max_per_turn: int = 3, exclude=(), seed: int = SEED,
          configs=None) -> dict:
    """See the module text.  configs: {variant: validation config path} (default n_episode's two)."""
    import n_episode as E

    gen, out = Path(gen), Path(out)
    src = gen / "instances_final.jsonl"
    found = read_jsonl(src)
    eval_scenes = {r["scene_id"] for p in exclude for r in read_jsonl(p)}
    clash = sorted({d["scene_id"] for d in found} & eval_scenes)
    if clash:
        raise SystemExit(f"{len(clash)} demonstration scenes are evaluation scenes (e.g. {clash[:3]})")
    ok = [d for d in found if usable(d)]
    # 2026-11-03: some renders of the generation are colour noise (n_frames.py); a demonstration with any such
    # frame is left out (its frames and the judge at them came from the same broken process state)
    bad = {d["instance_id"] for d in ok if any(F.corrupted(gen / f["image"]) for f in d["path_frames"])}
    demos = [d for d in ok if d["instance_id"] not in bad]
    order = demo_order(demos, seed)
    if len(order) < max(sizes):
        raise SystemExit(f"only {len(order)} usable demonstrations for the largest size {max(sizes)}")
    configs = configs or {"qwen25": E.VAL_ENV, "qwen3": E.VAL_ENV_QWEN3}
    out.mkdir(parents=True, exist_ok=True)
    (out / "order.json").write_text(json.dumps(
        [{"instance_id": d["instance_id"], "task_type": d["task_type"], "scene_id": d["scene_id"],
          "actions": [f["action"] for f in d["path_frames"][1:]],
          "turns": chunk_action_runs([f["action"] for f in d["path_frames"][1:]], max_per_turn)} for d in order],
        indent=0) + "\n")
    summary = {"gen": str(src), "found": len(found), "path_succeeds": len(ok), "corrupted_frames": len(bad),
               "corrupted_per_type": dict(collections.Counter(d["task_type"] for d in ok
                                                              if d["instance_id"] in bad)),
               "usable": len(demos), "seed": seed, "max_per_turn": max_per_turn,
               "chunking": "action_runs", "excluded_scene_files": [str(p) for p in exclude],
               "found_per_type": dict(collections.Counter(d["task_type"] for d in found)),
               "usable_per_type": dict(collections.Counter(d["task_type"] for d in demos)),
               "configs": {k: str(v) for k, v in configs.items()}, "sizes": {}}
    for variant, path in configs.items():
        cfg, _ = E.upstream_config(Path(path))
        (out / variant).mkdir(exist_ok=True)
        convs = [conversation(d, gen, cfg, max_per_turn, E) for d in order[:max(sizes)]]
        for n in sizes:
            write_jsonl(out / variant / f"demos_{n}.jsonl", convs[:n])
    for n in sizes:
        summary["sizes"][str(n)] = size_summary(order[:n], max_per_turn)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("instances")
    i.add_argument("--from", dest="paths", nargs="+", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--smoke-per-family", type=int, default=10)
    b = sub.add_parser("build")
    b.add_argument("--gen", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    b.add_argument("--max-per-turn", type=int, default=3)
    b.add_argument("--exclude", nargs="*", default=[])
    b.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    if args.cmd == "instances":
        print(json.dumps(instances(args.paths, Path(args.out), args.smoke_per_family)))
    else:
        s = build(Path(args.gen), Path(args.out), args.sizes, args.max_per_turn, args.exclude, args.seed)
        print(json.dumps({k: s[k] for k in ("found", "usable", "usable_per_type")}))
        for n, v in s["sizes"].items():
            print(n, json.dumps(v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
