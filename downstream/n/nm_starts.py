#!/usr/bin/env python3
"""The main line's matrix v3 starts evaluated N1's way (the user, 2026-11-03 evening: the picks of groups C, D and E,
two seeds x S0 / V4 each, plus C's and D's SFT starts; frozen weights, the merged 600 instances; no far transfer).

    python downstream/n/nm_starts.py make --paths P1 P2 ... [--out downstream/n/nm_starts.json]
    python downstream/n/nm_starts.py check [--starts FILE]
    python downstream/n/nm_starts.py val-env NAME [--starts FILE]
    python downstream/n/nm_starts.py submit --out NM --instances FILE [--names ...] [--pool zoetrope] [--go]

make: one entry per weight folder (read by path, never copied), named from where it lies in VAGEN-Lite-matrix:
  * RL: exps/matrix_v3/m3_<g>_s<chain>/rl_run_m3<g>s<k><s0|d>/checkpoints/global_step_<n>/actor/huggingface ->
    "<G>-s<k>-<S0|V4>" (the runs' "d" is the dense V4 reward), its prompt the run's own rl_m3<g>s<k><s0|d>/val_env.yaml,
    its SFT start "<G>-SFT" for C and D and Q3S for E (E's RL grew from the main line's formal SFT epoch 8, N1's Q3S);
  * SFT: exps/matrix_v3/sft/<G>/model/checkpoint-<n> -> "<G>-SFT", its prompt the group's seed-1 S0 run's val_env.
  Every entry: base Q3B (all are Qwen3-VL-8B).  Writes the entries in the order given.
check: every entry's weights (safetensors, config.json, preprocessor_config.json), its val_env, the prompt fields of
  its val_env against its group's others (prompt_format, prompt_format_example, max_actions_per_step, action_space,
  max_episode_steps), and that the SFT it names is an entry or Q3S; exit 1 on any problem.
val-env: the entry's val_env path (run_nm.sh).
submit: one one-card sco job per name (default all) running run_nm.sh NM/<name> INSTANCES <name> 12 4, as the
  container's root (as N1's Qwen3 models); without --go only prints what it would submit.
register(q0_ask): the entries become q0_ask models (Qwen3 family, their own folder), so n_policy_server.py loads
  them by name.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
STARTS = HERE / "nm_starts.json"
MATRIX = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3"
RL_RE = re.compile(r"/m3_(?P<g>[a-z])_s(?P<chain>\d+)/rl_run_m3(?P=g)s(?P<k>\d)(?P<r>s0|d)/checkpoints/"
                   r"global_step_(?P<n>\d+)/actor/huggingface/?$")
SFT_RE = re.compile(r"/sft/(?P<G>[A-Z])/model/checkpoint-(?P<n>\d+)/?$")
REWARD = {"s0": "S0", "d": "V4"}
SFT_OF_GROUP = {"C": "C-SFT", "D": "D-SFT", "E": "Q3S"}
PROMPT_FIELDS = ("prompt_format", "prompt_format_example", "max_actions_per_step", "action_space", "max_episode_steps")


def entry(path: str, matrix: str = MATRIX) -> dict:
    path = str(path).rstrip("/")
    m = RL_RE.search(path)
    if m:
        g, k, r = m["g"], int(m["k"]), m["r"]
        val_env = str(Path(path).parents[3].parent / f"rl_m3{g}s{k}{r}" / "val_env.yaml")
        return {"name": f"{g.upper()}-s{k}-{REWARD[r]}", "path": path, "val_env": val_env, "group": g.upper(),
                "seed": k, "reward": REWARD[r], "step": int(m["n"]), "sft": SFT_OF_GROUP.get(g.upper()),
                "base": "Q3B"}
    m = SFT_RE.search(path)
    if m:
        g = m["G"].lower()
        chains = sorted(Path(matrix).glob(f"m3_{g}_s*"))
        val_env = str((chains[0] if chains else Path(matrix) / f"m3_{g}_s?") / f"rl_m3{g}s1s0" / "val_env.yaml")
        return {"name": f"{m['G']}-SFT", "path": path, "val_env": val_env, "group": m["G"], "seed": None,
                "reward": "SFT", "step": int(m["n"]), "sft": None, "base": "Q3B"}
    raise ValueError(f"not a matrix v3 RL or SFT weight folder: {path}")


def make(paths, matrix: str = MATRIX) -> list:
    out = [entry(p, matrix) for p in paths]
    names = [e["name"] for e in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"two weight folders for {', '.join(dup)}")
    return out


def load(path: Path = STARTS) -> list:
    path = Path(path)
    return json.loads(path.read_text())["starts"] if path.is_file() else []


def by_name(starts) -> dict:
    return {e["name"]: e for e in starts}


def register(q0, path: Path = STARTS) -> list:
    """Add the entries to q0_ask's Qwen3 models (name -> folder); returns their names."""
    names = []
    for e in load(path):
        if e["name"] in q0.MODELS and q0.QWEN3.get(e["name"]) != e["path"]:
            raise ValueError(f"{e['name']} is already a q0_ask model")
        q0.QWEN3[e["name"]] = e["path"]
        names.append(e["name"])
    q0.MODELS = tuple(q0.MODELS) + tuple(n for n in names if n not in q0.MODELS)
    return names


def prompt_fields(val_env: Path) -> dict:
    import yaml

    cfg = yaml.safe_load(Path(val_env).read_text())["envs"][0]["config"]
    return {k: cfg.get(k) for k in PROMPT_FIELDS}


def check(starts) -> list:
    """Problems found (empty: all good)."""
    problems, names = [], by_name(starts)
    fields = {}
    for e in starts:
        p = Path(e["path"])
        if not list(p.glob("*.safetensors")):
            problems.append(f"{e['name']}: no safetensors in {p}")
        for f in ("config.json", "preprocessor_config.json"):
            if not (p / f).is_file():
                problems.append(f"{e['name']}: no {f} in {p}")
        if not Path(e["val_env"]).is_file():
            problems.append(f"{e['name']}: no val_env {e['val_env']}")
            continue
        fields[e["name"]] = prompt_fields(e["val_env"])
        if e.get("sft") and e["sft"] not in names and e["sft"] != "Q3S":
            problems.append(f"{e['name']}: its SFT start {e['sft']} is not evaluated")
    for g in sorted({e["group"] for e in starts}):
        seen = {json.dumps(fields[e["name"]], sort_keys=True) for e in starts
                if e["group"] == g and e["name"] in fields}
        if len(seen) > 1:
            problems.append(f"group {g}: the prompt fields differ: {sorted(seen)}")
    return problems


def submit(starts, out: Path, instances: Path, names=None, pool: str = "zoetrope", go: bool = False,
           workers: int = 12, repair_workers: int = 4) -> list:
    sys.path.insert(0, str(HERE.parent))
    import sco_job

    by = by_name(starts)
    done = []
    for n in names or list(by):
        if n not in by:
            raise KeyError(f"{n} is not in the starts")
        payload = ["bash", str(WT / "downstream/n/run_nm.sh"), str(Path(out) / n), str(instances), n,
                   str(workers), str(repair_workers)]
        run = Path(out) / f"{n}_job"
        row = {"name": n, "run": str(run), "payload": payload}
        if go:
            sco_job.prepare(run, 1, pool, f"nm-{n.lower()}", payload, as_root=True)
            row["job_id"] = sco_job.submit(run)
        done.append(row)
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("make")
    a.add_argument("--paths", nargs="+", required=True)
    a.add_argument("--out", type=Path, default=STARTS)
    a.add_argument("--note", default="", help="the main line's one-line note, kept in the file")
    a = sub.add_parser("check")
    a.add_argument("--starts", type=Path, default=STARTS)
    a = sub.add_parser("val-env")
    a.add_argument("name")
    a.add_argument("--starts", type=Path, default=STARTS)
    a = sub.add_parser("submit")
    a.add_argument("--starts", type=Path, default=STARTS)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--instances", type=Path, required=True)
    a.add_argument("--names", nargs="*", default=None)
    a.add_argument("--pool", default="zoetrope")
    a.add_argument("--go", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "make":
        starts = make(args.paths)
        args.out.write_text(json.dumps({"note": args.note, "starts": starts}, indent=1, ensure_ascii=False) + "\n")
        for e in starts:
            print(f"{e['name']:10s} step {e['step']:>5}  {e['path']}")
        return 0
    starts = load(args.starts)
    if args.cmd == "check":
        problems = check(starts)
        for e in starts:
            print(f"{e['name']:10s} {e['reward']:4s} step {e['step']:>5}  sft {e.get('sft') or '-':6s} {e['path']}")
        for p in problems:
            print("PROBLEM", p)
        return 1 if problems else 0
    if args.cmd == "val-env":
        e = by_name(starts).get(args.name)
        if e is None:
            print(f"{args.name} is not in {args.starts}", file=sys.stderr)
            return 2
        print(e["val_env"])
        return 0
    rows = submit(starts, args.out, args.instances, args.names, args.pool, args.go)
    print(json.dumps(rows, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
