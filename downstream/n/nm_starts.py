#!/usr/bin/env python3
"""The main line's matrix v3 starts evaluated N1's way (the user, 2026-11-03 evening: the picks of groups C, D and E,
two seeds x S0 / V4 each, plus C's and D's SFT starts; frozen weights, the merged 600 instances; no far transfer).

    python downstream/n/nm_starts.py make --paths P1 P2 ... [--out downstream/n/nm_starts.json] [--append]
                                         [--val-env NAME=PATH ...] [--archived NAME=LABEL ...] [--note TEXT]
    python downstream/n/nm_starts.py check [--starts FILE]
    python downstream/n/nm_starts.py val-env NAME [--starts FILE]
    python downstream/n/nm_starts.py submit --out NM --instances FILE [--names ...] [--pool zoetrope] [--go]

make: one entry per weight folder (read by path, never copied), named from where it lies in VAGEN-Lite-matrix:
  * RL: exps/matrix_v3/m3_<g>_s<chain>/rl_run_m3<g>s<k><s0|d>/checkpoints/global_step_<n>/actor/huggingface ->
    "<G>-s<k>-<S0|V4>" (the runs' "d" is the dense V4 reward), its prompt the run's own rl_m3<g>s<k><s0|d>/val_env.yaml,
    its SFT start "<G>-SFT" for C and D and Q3S for E (E's RL grew from the main line's formal SFT epoch 8, N1's Q3S);
  * SFT: exps/matrix_v3/sft/<G>/model/checkpoint-<n> -> "<G>-SFT", its prompt the group's seed-1 S0 run's val_env
    (an RL chain is m3_<g>_s<digits>; the SFT-only chains m3_b_sft, m3_e2_sft, m3_dn_sft are not), or, for a group
    without an RL chain, the main line's Qwen3 RL chain's (QWEN3_VAL_ENV: no_think + action_names, the same text as
    C's and D's; the main line confirmed it for E2, B' and D_noP0 on 2026-11-03).  The matrix's B is the main line's
    B': named "Bp-SFT" (plain "B" is N1's Qwen2.5 base), group "B′".
  Every entry: base Q3B (all are Qwen3-VL-8B); an RL entry also the main line's name of its pick (main_name, e.g.
  m3cs1s0_050).  Writes the entries in the order given; --append adds them to the file's entries (a name twice is an
  error).  --val-env NAME=PATH gives an entry another prompt (2026-11-03: the new E, an SFT on P0 + P1 without RL,
  must not take the old E chain's free_think one); --archived NAME=LABEL marks an entry (new or already in the file)
  as archived, which nm_report.py reports apart under LABEL (2026-11-03: the old E's four RL, "自由思考提示，已废弃").
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
SFT_RE = re.compile(r"/sft/(?P<G>[^/]+)/model/checkpoint-(?P<n>\d+)/?$")
# n_episode.VAL_ENV_QWEN3 (the main line's Qwen3 RL chain; N1's Q3 models): its prompt renders as C's and D's
QWEN3_VAL_ENV = "/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_rl_20261026/rl_q3d/val_env.yaml"
REWARD = {"s0": "S0", "d": "V4"}
SFT_OF_GROUP = {"C": "C-SFT", "D": "D-SFT", "E": "Q3S"}
SFT_NAME = {"B": "Bp"}  # an SFT folder's group -> the start's name stem
GROUP_LABEL = {"B": "B′"}  # -> the group as the main line writes it
PROMPT_FIELDS = ("prompt_format", "prompt_format_example", "max_actions_per_step", "action_space", "max_episode_steps")


def entry(path: str, matrix: str = MATRIX) -> dict:
    path = str(path).rstrip("/")
    m = RL_RE.search(path)
    if m:
        g, k, r = m["g"], int(m["k"]), m["r"]
        val_env = str(Path(path).parents[3].parent / f"rl_m3{g}s{k}{r}" / "val_env.yaml")
        return {"name": f"{g.upper()}-s{k}-{REWARD[r]}", "path": path, "val_env": val_env, "group": g.upper(),
                "seed": k, "reward": REWARD[r], "step": int(m["n"]), "sft": SFT_OF_GROUP.get(g.upper()),
                "base": "Q3B", "main_name": f"m3{g}s{k}{r}_{int(m['n']):03d}"}
    m = SFT_RE.search(path)
    if m:
        G = m["G"]
        g = G.lower()
        chains = [c for c in sorted(Path(matrix).glob(f"m3_{g}_s*")) if re.fullmatch(rf"m3_{re.escape(g)}_s\d+", c.name)]
        val_env = str(chains[0] / f"rl_m3{g}s1s0" / "val_env.yaml") if chains else QWEN3_VAL_ENV
        return {"name": f"{SFT_NAME.get(G, G)}-SFT", "path": path, "val_env": val_env, "group": GROUP_LABEL.get(G, G),
                "seed": None, "reward": "SFT", "step": int(m["n"]), "sft": None, "base": "Q3B"}
    raise ValueError(f"not a matrix v3 RL or SFT weight folder: {path}")


def pairs(items, what: str) -> dict:
    """["NAME=VALUE", ...] -> {NAME: VALUE}."""
    out = {}
    for x in items or ():
        name, sep, value = x.partition("=")
        if not sep or not name or not value:
            raise ValueError(f"{what}: {x!r} is not NAME=VALUE")
        out[name] = value
    return out


def make(paths, matrix: str = MATRIX, existing=(), val_env=None, archived=None) -> list:
    """existing + one entry per path; val_env / archived: {name: prompt path / label} for any of them."""
    out = list(existing) + [entry(p, matrix) for p in paths]
    names = [e["name"] for e in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"two weight folders for {', '.join(dup)}")
    for what, given in (("val_env", val_env or {}), ("archived", archived or {})):
        unknown = sorted(set(given) - set(names))
        if unknown:
            raise KeyError(f"{what} for {', '.join(unknown)}, which are not entries")
        for e in out:
            if e["name"] in given:
                e[what] = given[e["name"]]
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
    for g, arch in sorted({(e["group"], bool(e.get("archived"))) for e in starts}):  # archived ones apart
        seen = {json.dumps(fields[e["name"]], sort_keys=True) for e in starts
                if e["group"] == g and bool(e.get("archived")) == arch and e["name"] in fields}
        if len(seen) > 1:
            problems.append(f"group {g}{' (archived)' if arch else ''}: the prompt fields differ: {sorted(seen)}")
    return problems


def job_name(name: str) -> str:
    """The sco job's name: lower-case letters, digits and hyphens only (D_noP0-SFT -> nm-d-nop0-sft)."""
    return "nm-" + re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")


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
            sco_job.prepare(run, 1, pool, job_name(n), payload, as_root=True)
            row["job_id"] = sco_job.submit(run)
        done.append(row)
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("make")
    a.add_argument("--paths", nargs="*", default=[])
    a.add_argument("--out", type=Path, default=STARTS)
    a.add_argument("--append", action="store_true", help="keep the entries (and note) already in --out")
    a.add_argument("--matrix", default=MATRIX, help="where the RL chains lie (an SFT start's prompt)")
    a.add_argument("--val-env", nargs="*", default=[], metavar="NAME=PATH")
    a.add_argument("--archived", nargs="*", default=[], metavar="NAME=LABEL")
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
        old = json.loads(args.out.read_text()) if args.append and args.out.is_file() else {"note": "", "starts": []}
        starts = make(args.paths, args.matrix, existing=old["starts"], val_env=pairs(args.val_env, "--val-env"),
                      archived=pairs(args.archived, "--archived"))
        note = "\n".join(x for x in (old.get("note", ""), args.note) if x)
        args.out.write_text(json.dumps({"note": note, "starts": starts}, indent=1, ensure_ascii=False) + "\n")
        for e in starts:
            print(f"{e['name']:10s} step {e['step']:>5}  {e['path']}{'  [' + e['archived'] + ']' if e.get('archived') else ''}")
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
