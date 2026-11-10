"""Independent recompute of the TVR pilot numbers (CONTRACT_wave2 5.3.5; brief section 6: every number that goes into
a report is recomputed by an independent script from the raw files).

    PY_EVAL -B -m vpd.tvr.recheck_tvr floors --reach OUT/tvr/reach/v1 --tasks-dir REPO/data/tasks \
        --public OUT/tvr/hf_meta/tvr-sft-va/sft_train.json --out F.json [--task-set eval500]
    PY_EVAL -B -m vpd.tvr.recheck_tvr counts --run D [--out F.json]

INDEPENDENT of the tvrdata package: this module re-implements, from the contract TEXT only (sections 4.2-4.4 and
5.1.5), the pose rounding, the kinematic transitions, the valid-action list, the frozen per-task reachable sets (read
from the reach/v1 scene files by digest, the digest recomputed), the expert medians from the public file and the
three blind floor policies.  It imports vpd.tvr.pilot_cfg ONLY; it must never import vpd.tvr.kin_env, floors, bfs,
tasks or score_tvr (a test asserts these are absent from sys.modules after a run).  Stdlib only.

floors -> {"schema", "task_set", "n", "reach_index_sha256", "stop0", "fixed", "random", "F", "warnings"} in the
shapes of floors_<set>.json (CONTRACT 6.8).  F.source is the name of the component block that gives the maximum
("stop0" | "fixed" | "random"; ties: the first of that order).
counts -> {"run", "n", "k", "rate", "by_category": {cat: {"k", "n"}}, "stopped", "task_ids_sha256"} recounted from
RUN/tasks/*/trajectory.json (result.success is True, result.stopped), never from per_task.jsonl or results.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

from vpd.tvr import pilot_cfg

SCHEMA_FLOORS = "vpd.tvr.floors/1"
GRID = 0.25
HOR_MIN, HOR_MAX = -30, 30
# CONTRACT 4.4: facing angle -> (dx, dz) per grid step (AI2-THOR: rotation 0 = +z, 90 = +x)
DELTAS = {0: (0, 1), 45: (1, 1), 90: (1, 0), 135: (1, -1), 180: (0, -1), 225: (-1, -1), 270: (-1, 0), 315: (-1, 1)}
MOVES = ("MoveAhead", "MoveBack", "MoveLeft", "MoveRight")
FORBIDDEN_IMPORTS = ("vpd.tvr.kin_env", "vpd.tvr.floors", "vpd.tvr.bfs", "vpd.tvr.tasks", "vpd.tvr.score_tvr")


# --- poses and sets (CONTRACT 4.2, 4.3) -----------------------------------------------------------------------

def rnd(v) -> float:
    return round(float(v), 2) + 0.0


def pose(d: dict) -> tuple:
    """(x, z, rot, hor) of {"position": {x, y, z}, "rotation_y", "horizon"} (y ignored)."""
    p = d["position"]
    return (rnd(p["x"]), rnd(p["z"]), int(round(float(d["rotation_y"]))) % 360, int(round(float(d["horizon"]))))


def cells(positions) -> frozenset:
    return frozenset((rnd(p["x"]), rnd(p["z"])) for p in positions)


def digest(positions) -> str:
    return hashlib.sha256("\n".join(f"{x:.2f},{z:.2f}" for (x, z) in sorted(cells(positions))).encode("utf-8")
                          ).hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_reach(reach_dir) -> tuple:
    """({task_id: frozenset of cells}, index sha256, warnings) from reach/v1/index.json + scenes/<scene>.json; each
    frozen set is looked up by the index digest in its scene file's "sets" and its digest recomputed."""
    reach_dir = Path(reach_dir)
    idx_path = reach_dir / "index.json"
    index = json.loads(idx_path.read_text())
    scenes, out, warnings = {}, {}, []
    for tid, ent in index["tasks"].items():
        scene = ent["scene"]
        if scene not in scenes:
            scenes[scene] = json.loads((reach_dir / "scenes" / f"{scene}.json").read_text())
        positions = scenes[scene]["sets"].get(ent["digest"])
        if positions is None:
            raise ValueError(f"{tid}: digest {ent['digest'][:12]} not in scenes/{scene}.json")
        got = digest(positions)
        if got != ent["digest"]:
            raise ValueError(f"{tid}: scene set digest recomputes to {got[:12]}, index says {ent['digest'][:12]}")
        out[tid] = cells(positions)
    return out, sha256_file(idx_path), warnings


# --- transitions (CONTRACT 4.4) -------------------------------------------------------------------------------

def move_cell(p: tuple, action: str) -> tuple:
    x, z, rot, _ = p
    dx, dz = DELTAS[rot]
    if action == "MoveBack":
        dx, dz = -dx, -dz
    elif action == "MoveLeft":
        dx, dz = -dz, dx
    elif action == "MoveRight":
        dx, dz = dz, -dx
    return rnd(x + dx * GRID), rnd(z + dz * GRID)


def valid_list(p: tuple, reach: frozenset) -> list:
    out = [a for a in MOVES if move_cell(p, a) in reach]
    out += ["RotateRight", "RotateLeft"]
    if p[3] > HOR_MIN:
        out.append("LookUp")
    if p[3] < HOR_MAX:
        out.append("LookDown")
    out.append("Stop")
    return out


def transition(p: tuple, action: str, reach: frozenset) -> tuple:
    x, z, rot, hor = p
    if action in MOVES:
        c = move_cell(p, action)
        return (c[0], c[1], rot, hor) if c in reach else p
    if action == "RotateRight":
        return (x, z, (rot + 45) % 360, hor)
    if action == "RotateLeft":
        return (x, z, (rot - 45) % 360, hor)
    if action == "LookUp":
        return p if hor <= HOR_MIN else (x, z, rot, hor - 30)
    if action == "LookDown":
        return p if hor >= HOR_MAX else (x, z, rot, hor + 30)
    raise ValueError(f"not a non-Stop action: {action!r}")


def exact(p: tuple, t: tuple) -> bool:
    """TVRBench is_exact_match: xz distance < 0.01, rotation error < 1, horizon error < 1."""
    d = abs(p[2] - t[2]) % 360
    return math.hypot(p[0] - t[0], p[1] - t[1]) < 0.01 and min(d, 360 - d) < 1 and abs(p[3] - t[3]) < 1


def cap(task: dict) -> int:
    return int(pilot_cfg.FROZEN["protocol"]["max_steps"][task["dataset"]])


def run_sequence(task: dict, reach: frozenset, actions: list) -> bool:
    """Success of 'actions then Stop' (len(actions) + 1 replies; more than the cap never stops: failure)."""
    if len(actions) + 1 > cap(task):
        return False
    p = pose(task["start"])
    for a in actions:
        p = transition(p, a, reach)
    return exact(p, pose(task["target"]))


# --- inputs ---------------------------------------------------------------------------------------------------

def load_split(tasks_dir, name: str) -> list:
    doc = json.loads((Path(tasks_dir) / f"{name}.json").read_text())
    return doc if isinstance(doc, list) else doc["tasks"]


def category(task: dict) -> str:
    return pilot_cfg.CATEGORY_OF[(task["dataset"], task["difficulty"])]


def expert_medians(public_path, sft_by_id: dict) -> dict:
    """{category: median_low of the non-Stop action counts} over the public sequences (task id = the folder of
    images[0]; replies 'Action: <Name>')."""
    recs = json.loads(Path(public_path).read_text())
    per = {c: [] for c in pilot_cfg.CATEGORIES}
    seen = set()
    for r in recs:
        tid = str(r["images"][0]).split("/")[1]
        if tid in seen:
            raise ValueError(f"public file: task {tid} twice")
        seen.add(tid)
        acts = [c["value"].split("Action: ", 1)[1].strip() for c in r["conversations"] if c["from"] == "gpt"]
        per[category(sft_by_id[tid])].append(sum(1 for a in acts if a != "Stop"))
    out = {}
    for c, xs in per.items():
        if not xs:
            raise ValueError(f"public file: no sequence of category {c}")
        xs = sorted(xs)
        out[c] = xs[(len(xs) - 1) // 2]  # median_low
    return out


# --- the three floor policies (CONTRACT 5.1.5) ------------------------------------------------------------------

def rate_block(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": (k / n) if n else None}


def floors(reach_dir, tasks_dir, public_path, task_set: str = "eval500") -> dict:
    split = {"eval500": "eval", "dev200": None}.get(task_set, "eval")
    if split is None:
        raise ValueError("recheck_tvr floors recomputes the eval500 floor only")
    reach, idx_sha, warnings = read_reach(reach_dir)
    sft = load_split(tasks_dir, "sft")
    ev = load_split(tasks_dir, split)
    sft_by_id = {t["task_id"]: t for t in sft}
    for t in sft + ev:
        if t["task_id"] not in reach:
            raise ValueError(f"{t['task_id']}: no frozen reachable set in {reach_dir}")
    # (i) STOP0
    k0 = sum(1 for t in ev if run_sequence(t, reach[t["task_id"]], []))
    # (ii) FIX: a^k + Stop, chosen on sft1600 by successes; ties -> smaller k, then ACTIONS order
    kmax = int(pilot_cfg.FROZEN["floors"]["fixed_k_max"])
    table, best = {}, None
    for k in range(kmax + 1):
        for ai, a in enumerate(pilot_cfg.NON_STOP):
            s = sum(1 for t in sft if run_sequence(t, reach[t["task_id"]], [a] * k))
            table[f"{a}:{k}"] = s
            key = (-s, k, ai)
            if best is None or key < best[0]:
                best = (key, a, k, s)
    _, fa, fk, fs = best
    fe = sum(1 for t in ev if run_sequence(t, reach[t["task_id"]], [fa] * fk))
    # (iii) RAND: uniform valid non-Stop actions for median_low(category) steps, then Stop; 3 seeds
    med = expert_medians(public_path, sft_by_id)
    seeds = list(pilot_cfg.FROZEN["floors"]["random_seeds"])
    salt = pilot_cfg.FROZEN["salts"]["rand_floor"]
    ks = []
    for s in seeds:
        k = 0
        for t in ev:
            rng = random.Random(f"{salt}{s}|{t['task_id']}")
            m = med[category(t)]
            if m + 1 > cap(t):
                continue
            rs = reach[t["task_id"]]
            p = pose(t["start"])
            for _ in range(m):
                nonstop = [v for v in valid_list(p, rs) if v != "Stop"]
                p = transition(p, nonstop[rng.randrange(len(nonstop))], rs)
            k += int(exact(p, pose(t["target"])))
        ks.append(k)
    n = len(ev)
    rates = [k / n for k in ks]
    mean_rate = sum(rates) / len(rates)
    comps = [("stop0", k0 / n), ("fixed", fe / n), ("random", mean_rate)]
    src, F = comps[0]
    for name, r in comps[1:]:
        if r > F:
            src, F = name, r
    return {"schema": SCHEMA_FLOORS, "task_set": task_set, "n": n, "reach_index_sha256": idx_sha,
            "stop0": rate_block(k0, n),
            "fixed": {"chosen": {"action": fa, "k": fk}, "sft": {"k": fs, "n": len(sft)},
                      "eval": rate_block(fe, n), "table_sft": table},
            "random": {"medians": med, "seeds": seeds, "k": ks, "rates": rates, "mean_rate": mean_rate},
            "F": {"rate": F, "source": src, "k_equiv": int(math.ceil(F * n - 1e-12))},
            "warnings": warnings, "recheck": "vpd.tvr.recheck_tvr (independent of tvrdata)"}


RATE_TOL = 1e-12  # rates are k/n floats; the k counts are compared exactly


def compare_floors(mine: dict, theirs: dict) -> list:
    """[] iff F (rate, source) and the three components (k counts exactly, rates within RATE_TOL) agree."""
    bad = []

    def num(a, b, what):
        if a is None or b is None or not isinstance(b, (int, float)) or abs(float(a) - float(b)) > RATE_TOL:
            bad.append(f"{what}: recheck {a!r} != floors file {b!r}")

    try:
        num(mine["F"]["rate"], theirs["F"]["rate"], "F.rate")
        if mine["F"]["source"] != theirs["F"]["source"]:
            bad.append(f"F.source: recheck {mine['F']['source']!r} != floors file {theirs['F']['source']!r}")
        if mine["stop0"]["k"] != theirs["stop0"]["k"]:
            bad.append(f"stop0.k: {mine['stop0']['k']} != {theirs['stop0']['k']}")
        num(mine["stop0"]["rate"], theirs["stop0"]["rate"], "stop0.rate")
        if mine["fixed"]["chosen"] != theirs["fixed"]["chosen"]:
            bad.append(f"fixed.chosen: {mine['fixed']['chosen']} != {theirs['fixed']['chosen']}")
        if mine["fixed"]["eval"]["k"] != theirs["fixed"]["eval"]["k"]:
            bad.append(f"fixed.eval.k: {mine['fixed']['eval']['k']} != {theirs['fixed']['eval']['k']}")
        num(mine["fixed"]["eval"]["rate"], theirs["fixed"]["eval"]["rate"], "fixed.eval.rate")
        if list(mine["random"]["k"]) != list(theirs["random"]["k"]):
            bad.append(f"random.k: {mine['random']['k']} != {theirs['random']['k']}")
        num(mine["random"]["mean_rate"], theirs["random"]["mean_rate"], "random.mean_rate")
        if mine["n"] != theirs.get("n"):
            bad.append(f"n: {mine['n']} != {theirs.get('n')}")
    except (KeyError, TypeError) as e:
        bad.append(f"floors file shape: {e!r}")
    return bad


# --- counts from trajectories ---------------------------------------------------------------------------------

def counts(run_dir) -> dict:
    """k/n (and per category, stopped) recounted from RUN/tasks/<task_id>/trajectory.json files."""
    run_dir = Path(run_dir)
    by = {c: {"k": 0, "n": 0} for c in pilot_cfg.CATEGORIES}
    k = n = stopped = 0
    ids, success = [], {}
    tdir = run_dir / "tasks"
    for p in sorted(tdir.glob("*/trajectory.json")) if tdir.is_dir() else []:
        tr = json.loads(p.read_text())
        res = tr.get("result") or {}
        ok = res.get("success") is True
        cat = pilot_cfg.CATEGORY_OF[(tr["dataset"], tr["difficulty"])]
        by[cat]["n"] += 1
        by[cat]["k"] += int(ok)
        n += 1
        k += int(ok)
        stopped += int(res.get("stopped") is True)
        ids.append(tr["task_id"])
        success[tr["task_id"]] = ok
    return {"run": str(run_dir), "n": n, "k": k, "rate": (k / n) if n else None, "by_category": by,
            "stopped": stopped, "task_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
            "success": success}


def write_json(p, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("floors")
    p.add_argument("--reach", type=Path, required=True)
    p.add_argument("--tasks-dir", type=Path, required=True)
    p.add_argument("--public", type=Path, required=True)
    p.add_argument("--task-set", default="eval500")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--compare", type=Path, default=None, help="a floors_<set>.json to compare with (exit 1 if unequal)")
    p = sub.add_parser("counts")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "floors":
        rep = floors(args.reach, args.tasks_dir, args.public, args.task_set)
        rc = 0
        if args.compare is not None:
            rep["compare"] = {"file": str(args.compare), "problems": compare_floors(rep, json.loads(
                args.compare.read_text()))}
            rc = 0 if not rep["compare"]["problems"] else 1
        write_json(args.out, rep)
        print(json.dumps({"F": rep["F"], "compare": rep.get("compare")}))
        return rc
    rep = counts(args.run)
    rep.pop("success")
    if args.out:
        write_json(args.out, rep)
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
