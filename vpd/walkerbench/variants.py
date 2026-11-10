"""EXPLORATORY WalkerBench limited exploration: Q3B recipe variants (ROUND3 section 3 WalkerBench; DECISIONS G4, G6).
See vpd/EXPLORATORY_WB_VARIANTS.md.

Not part of the pre-registered pilot (PREREG_WB_PILOT.md).  Q3B only; selection on SEL = s2val + s2dev74 (131 tasks);
s2test takes no part (every eval / floors call refuses an s2test task set).  Nothing here writes under
OUT/walkerbench/runs/pilot/ (the pilot's Q3B 0W adapter is read only).  Root X = OUT/walkerbench/runs/explore/wbv/Q3B;
each variant writes only X/<variant>/ (data/, adapter/, evals/), the V1 probes X/V1/evals/, the floors X/floors/,
the B1 check X/b1/, analyze X/analysis/.

Login node (PY_EVAL, 1 core, nice -n 19):
    python -m vpd.walkerbench.variants setup             # split/s2dev74 + split/s2sel (created or verified), X dirs
    python -m vpd.walkerbench.variants data --variant V0|V1T|V2            # X/<V>/data/train.jsonl + manifest.json
    python -m vpd.walkerbench.variants data --variant V5 --base V0|V1T|V2  # + per-reply loss_weight
    python -m vpd.walkerbench.variants data --variant V4 --base V0|V1T|V2|V5   # byte copy (trained 10 epochs)
    python -m vpd.walkerbench.variants analyze           # -> X/analysis/ANALYZE.json, REPORT_EXPLORATORY.md
    python -m vpd.walkerbench.variants expect --variant V [--base B] [--check]   # outputs to verify before delete
In-job (vpd/jobs/wb_variants.sh VARIANT RUN [BASE]):
    PY_EVAL -m ... b1 --stage precheck|before|after|done   # G6 single-demo rebuild checks (job A, first)
    PY_EVAL -m ... floors --out X/floors/floors_s2sel.json --logs X/floors/logs --work RUN/floors_work
    PY_EVAL -m ... init-check [--need-loss-weight]       # pilot 0W adapter.pt sha256 5cdbfd4366a3 (no code equality)
    PY_EVAL -m ... adapter-ok --variant V [--base B]     # exit 0 complete, 1 (re)train, 3 trained with other code
    PY_EVAL -m ... eval-mode --variant V [--base B]      # "CONSTRAINED ODOMETER" of the variant's evals
    PY_Q3   -m ... eval [--odometer off|line] <run_vpd args>   # run_vpd.main + s2dev74/s2sel + odometer hook

Task sets (DECISIONS G4(a)): s2dev74 = continue the frozen md5(city) order of split_s2.json after the 6 s2val cities
and take whole train-rest cities until >= 60 tasks (Yokohama, Johannesburg, Curitiba, Sao Paulo, Chicago: 74 tasks,
36 nav / 38 vis, 0 shared routes / geofences / panos with the remaining 409); s2sel = s2val + s2dev74 (131, 65 / 66).
Both are sha-checked copies of split/s2train/tasks files with a selection.json, registered in run_vpd.TASK_SETS in
this process only (run_vpd.py is not edited).  The exploration training pool is train_rest minus s2dev74 = 409 tasks.

Variants (G4(d)): V0 = the current recipe on the 409 (control); V1 = eval-time odometer line, eval-only probes; V1T =
odometer line in every demo user turn and at eval; V2 = odometer THOUGHT in every demo reply, eval --constrained short;
V5 = the incumbent's data + per-reply loss_weight (STOP 5, ROTATION or MOVE(k != 1) 3, MOVE(1) 1); V4 = the
incumbent's data, --epochs 10.  Every training starts from the pilot's Q3B 0W with seed 1001.  The odometer is
vpd.walkerbench.odometer (one source for the eval line and the demo rewrite).  analyze prints numbers only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import uuid
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

from vpd import paths
from vpd.walkerbench import learn483 as L
from vpd.walkerbench import odometer as O

SCHEMA = "vpd.walkerbench.variants/1"
START = "Q3B"
WB = paths.WB_OUT
X_ROOT = paths.WB_RUNS / "explore" / "wbv" / START
PILOT_ROOT = paths.WB_RUNS / "pilot"
PILOT_Q3B = PILOT_ROOT / START
INIT = PILOT_Q3B / "adapters" / "0W"
INIT_SHA_PREFIX = "5cdbfd4366a3"  # the pilot's Q3B 0W adapter.pt (reused, never retrained here)
SPLIT_S2 = paths.WB_SPLIT / "split_s2.json"
S2DEV_DIR = paths.WB_SPLIT / "s2dev74"
S2SEL_DIR = paths.WB_SPLIT / "s2sel"
S2DEV_TASKS = S2DEV_DIR / "tasks"
S2SEL_TASKS = S2SEL_DIR / "tasks"
TASK_SETS = {"s2dev74": S2DEV_TASKS, "s2sel": S2SEL_TASKS}
EVAL_SETS = ("s2sel", "s2dev74", "s2val")  # the only task sets an exploration eval may use (never s2test)
ORDER = paths.WB_TIERS / "order_seed1001.json"
TIER_ALL = paths.WB_TIERS / "seed1001" / "demos_all.jsonl"  # the 483 train-rest demos in tier order (483A's data)
FROZEN_DEMOS = paths.WB_DEMOS / "s2train_640x400" / "demos_all.jsonl"
FROZEN_DEMOS_SHA256 = "5aea1bfd3d3ec9c44cefb4dbe8ef9749ef85fa613798f80335ca1732932b34b1"
B1_STEM = "vis_0578_target_20260127_1951_33"
B1_OUT = paths.WB_DEMOS / "s2train_640x400_B1_vis0578"
B1_ROTATION = "ROTATION(195.2, 0)"  # rm_expert's step-10 rotation after the heading wrap (-164.8 -> 195.2)

DEV_MIN_TASKS = 60
DEV_RULE = ("continue the frozen md5(city) order of split_s2.json (train_city_md5_order) after the 6 s2val cities and "
            "take whole train-rest cities until there are at least 60 tasks (DECISIONS G4(a))")
DEV_CITIES = ("Yokohama, Japan", "Johannesburg, South Africa", "Curitiba, Brazil", "Sao Paulo, Brazil", "Chicago, USA")
DEV_COUNTS = {"total": 74, "nav": 36, "vis": 38}
SEL_COUNTS = {"total": 131, "nav": 65, "vis": 66}
POOL_N = 409

TRAINED = ("V0", "V1T", "V2", "V5", "V4")
V5_BASES = ("V0", "V1T", "V2")
V4_BASES = ("V0", "V1T", "V2", "V5")
STATIC_MODE = {"V0": ("empty", "off"), "V1T": ("empty", "line"), "V2": ("short", "off")}
EPOCHS = {"V0": 3, "V1T": 3, "V2": 3, "V5": 3, "V4": 10}
V5_WEIGHTS = OrderedDict([("STOP", 5.0), ("TURN", 3.0), ("STRAIGHT", 1.0)])  # TURN = ROTATION or MOVE(k != 1)
RULES = {"V0": "demos_all filtered to the 409 exploration tasks in tier order (seed 1001), lines verbatim",
         "V1T": "V0 + odometer.line(walked before the decision) after the '- Pitch:' line of every user turn",
         "V2": "V0 + odometer.thought(walked before the decision, stated total for nav) as every reply's THOUGHT",
         "V5": "the base variant's data + per assistant message loss_weight: STOP 5, ROTATION or MOVE(k!=1) 3, "
               "MOVE(1) 1",
         "V4": "a byte copy of the base variant's data (trained with --epochs 10)"}
THOUGHT_MAX_CHARS = 32
ALPHA_A3 = 0.05
RATE_MAX_A4 = 0.80

# (owner dir, model, task set, constrained, odometer, blind) per job; V5 / V4 evals follow eval_mode(base)
JOB_EVALS = {
    "V0": (("V0", "V0", "s2sel", "empty", "off", False), ("V0", "V0", "s2sel", "empty", "off", True),
           ("V1", "V0", "s2sel", "empty", "line", False),
           ("V1", "raw", "s2sel", "off", "off", False), ("V1", "raw", "s2sel", "off", "line", False),
           ("V1", "raw", "s2sel", "empty", "off", False)),
    "V1T": (("V1T", "V1T", "s2sel", "empty", "line", False), ("V1T", "V1T", "s2sel", "empty", "line", True)),
    "V2": (("V2", "V2", "s2sel", "short", "off", False), ("V2", "V2", "s2sel", "short", "off", True)),
}
EVAL_NAME_RE = re.compile(r"^wbv__(?P<start>[A-Za-z0-9]+)__(?P<model>[A-Za-z0-9]+)__(?P<set>[a-z0-9]+)__640__"
                          r"(?P<mode>off|empty|short)(?P<odo>__odo)?(?P<blind>__blind)?$")


# ------------------------------------------------------------------------------------------------- utilities
utc, sha256_file, read_json, read_jsonl, write_json, write_text, rate = (L.utc, L.sha256_file, L.read_json,
                                                                         L.read_jsonl, L.write_json, L.write_text,
                                                                         L.rate)


def sha256_text(text: str) -> str:
    return L.sha256_bytes(text.encode("utf-8"))


def is_under(path, root) -> bool:
    return L.is_under(path, root)


def refuse_pilot(path) -> None:
    if is_under(path, PILOT_ROOT):
        raise SystemExit(f"[variants] refused: {path} is under {PILOT_ROOT} (the pilot's outputs are read only)")


def split_doc(split_path=SPLIT_S2) -> dict:
    return json.loads(Path(split_path).read_text())


def s2test_stems(doc: dict | None = None) -> set:
    return set((doc or split_doc())["splits"]["s2test"])


def refuse_s2test(task_set: str | None, tasks_dir=None, doc: dict | None = None) -> None:
    """Exploration never touches s2test (DECISIONS G4(c)): refuse the name, a task dir that is s2test's, any dir holding
    an s2test task, and any task set outside EVAL_SETS."""
    if not task_set:
        raise SystemExit("[variants] refused: give --task-set explicitly (run_vpd's default is s2test)")
    if "s2test" in str(task_set).lower():
        raise SystemExit(f"[variants] refused: task set {task_set!r} (s2test takes no part in the exploration)")
    if task_set not in EVAL_SETS:
        raise SystemExit(f"[variants] refused: task set {task_set!r} is not one of {EVAL_SETS}")
    if tasks_dir is not None:
        tdir = Path(tasks_dir)
        if is_under(tdir, paths.WB_S2TEST_TASKS) or is_under(paths.WB_S2TEST_TASKS, tdir):
            raise SystemExit(f"[variants] refused: {tdir} is the s2test task dir")
        bad = sorted({p.stem for p in tdir.glob("*.json")} & s2test_stems(doc))
        if bad:
            raise SystemExit(f"[variants] refused: {tdir} holds s2test tasks ({bad[:3]})")


def code_state() -> dict:
    out = {}
    for rel in ("vpd/walkerbench/variants.py", "vpd/walkerbench/odometer.py", "vpd/walkerbench/run_vpd.py",
                "vpd/walkerbench/learn483.py", "vpd/train/n2_train_vpd.py"):
        p = paths.WORKTREE / rel
        out[rel] = sha256_file(p) if p.is_file() else None
    return out


def _write_if_new(path: Path, text: str) -> str:
    """'wrote' | 'same'; a file with other content is never replaced (a job may have trained on it)."""
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return "same"
        raise SystemExit(f"[variants] {path} exists with different content (nothing overwritten; remove it by hand "
                         "only if no adapter was trained on it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return "wrote"


# ------------------------------------------------------------------------------------------------- the carve
def task_panos(doc: dict, stem: str) -> set:
    r = doc["routes"][doc["tasks"][stem]["route_id"]]
    return {p for p in [r[0], *r[1], *r[2]] if p}


def leakage(doc: dict, a: list, b: list) -> dict:
    t = doc["tasks"]
    pa = set().union(*(task_panos(doc, s) for s in a)) if a else set()
    pb = set().union(*(task_panos(doc, s) for s in b)) if b else set()
    return {"routes": len({t[s]["route_id"] for s in a} & {t[s]["route_id"] for s in b}),
            "geofences": len({t[s]["geofence"] for s in a} & {t[s]["geofence"] for s in b}),
            "panos": len(pa & pb), "cities": len({t[s]["city"] for s in a} & {t[s]["city"] for s in b})}


def kinds(stems) -> dict:
    stems = list(stems)
    c = Counter(s.split("_", 1)[0] for s in stems)
    return {"total": len(stems), "nav": c.get("nav", 0), "vis": c.get("vis", 0)}


def dev_carve(doc: dict, strict: bool = True) -> dict:
    """D60 of DECISIONS G4(a), with its asserts (strict): 74 tasks, 36 / 38, the five cities, 409 left, and 0 shared
    routes / geofences / panos (spawn, targets, visual_path) / cities with the remaining 409."""
    t, sp = doc["tasks"], doc["splits"]
    rest, val, test = list(sp["train_rest"]), list(sp["s2val"]), set(sp["s2test"])
    order, val_c = list(doc["train_city_md5_order"]), list(doc["val_cities"])
    if order[:len(val_c)] != val_c:
        raise SystemExit("split_s2.json: the md5(city) order does not start with the s2val cities")
    by_city = defaultdict(list)
    for s in rest:
        by_city[t[s]["city"]].append(s)
    cities, n = [], 0
    for c in order[len(val_c):]:
        if n >= DEV_MIN_TASKS:
            break
        if not by_city.get(c):
            continue
        cities.append(c)
        n += len(by_city[c])
    dev = sorted(s for c in cities for s in by_city[c])
    dset = set(dev)
    pool = [s for s in rest if s not in dset]
    leak = leakage(doc, dev, pool)
    out = {"rule": DEV_RULE, "cities": cities, "dev": dev, "pool": pool, "sel": sorted(val + dev),
           "s2val": sorted(val), "counts": {"dev": kinds(dev), "pool": kinds(pool), "sel": kinds(val + dev)},
           "leakage_dev_vs_pool": leak, "per_city": {c: kinds(by_city[c]) for c in cities}}
    probs = []
    if tuple(cities) != DEV_CITIES:
        probs.append(f"dev cities {cities} != {list(DEV_CITIES)}")
    if out["counts"]["dev"] != DEV_COUNTS:
        probs.append(f"dev counts {out['counts']['dev']} != {DEV_COUNTS}")
    if out["counts"]["sel"] != SEL_COUNTS:
        probs.append(f"sel counts {out['counts']['sel']} != {SEL_COUNTS}")
    if len(pool) != POOL_N:
        probs.append(f"pool {len(pool)} != {POOL_N}")
    if any(leak.values()):
        probs.append(f"dev shares {leak} with the remaining pool")
    if (set(dev) | set(val)) & test:
        probs.append("an s2test task in s2dev74 / s2sel")
    if strict and probs:
        raise SystemExit("[variants] dev carve differs from DECISIONS G4(a): " + "; ".join(probs))
    out["problems"] = probs
    return out


def split_plan(name: str, stems: list, doc: dict, split_sha: str, extra: dict) -> dict:
    t = doc["tasks"]
    files = OrderedDict((t[s]["file"], t[s]["sha256"]) for s in sorted(stems))
    return {"split": name, "schema": SCHEMA + ":split", "exploratory": True,
            "purpose": "EXPLORATORY recipe-selection task set of the WalkerBench limited exploration (DECISIONS G4); "
                       "Q3B only, never used for start comparisons; not part of the pilot",
            "split_s2_sha256": split_sha, "ids": sorted(stems), "files": files, "counts": kinds(stems),
            "cities": sorted({t[s]["city"] for s in stems}), **extra}


def split_problems(dest: Path, plan: dict) -> list:
    dest = Path(dest)
    probs = []
    sel = read_json(dest / "selection.json")
    if not sel or sel.get("ids") != plan["ids"] or sel.get("files") != plan["files"]:
        probs.append(f"{dest}/selection.json differs from the plan (ids or files)")
    have = {p.name: p for p in (dest / "tasks").glob("*.json")}
    if set(have) != set(plan["files"]):
        probs.append(f"{dest}/tasks: files differ (missing {sorted(set(plan['files']) - set(have))[:3]}, "
                     f"extra {sorted(set(have) - set(plan['files']))[:3]})")
    for f in sorted(set(have) & set(plan["files"])):
        if sha256_file(have[f]) != plan["files"][f]:
            probs.append(f"{dest}/tasks/{f}: sha256 differs")
    return probs


def build_split(dest: Path, plan: dict, src_tasks: Path) -> dict:
    """Create dest/{tasks/, selection.json} (atomic rename of a sibling temp dir), or verify an existing one."""
    dest, src_tasks = Path(dest), Path(src_tasks)
    refuse_pilot(dest)
    if dest.exists():
        probs = split_problems(dest, plan)
        if probs:
            raise SystemExit(f"[variants] {dest} exists but differs (nothing overwritten): " + "; ".join(probs))
        return {"state": "verified", "dir": str(dest), **plan["counts"]}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    (tmp / "tasks").mkdir(parents=True)
    try:
        for f, sha in plan["files"].items():
            data = (src_tasks / f).read_bytes()
            if L.sha256_bytes(data) != sha:
                raise SystemExit(f"{src_tasks / f}: sha256 differs from split_s2.json")
            (tmp / "tasks" / f).write_bytes(data)
        write_json(tmp / "selection.json", dict(plan, created_utc=utc()))
        try:
            os.rename(tmp, dest)
            state = "created"
        except OSError:
            state = "raced"
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)  # this call's own temp dir
    probs = split_problems(dest, plan)
    if probs:
        raise SystemExit(f"[variants] {dest} check failed: " + "; ".join(probs))
    return {"state": state, "dir": str(dest), **plan["counts"]}


def carve_plans(doc: dict, split_sha: str) -> tuple:
    c = dev_carve(doc)
    dev = split_plan("s2dev74", c["dev"], doc, split_sha,
                     {"rule": DEV_RULE, "dev_cities_in_order": c["cities"], "per_city": c["per_city"],
                      "train_pool": {"rule": "train_rest minus s2dev74 (exploration training pool, G4(b))",
                                     "n": len(c["pool"]), "counts": c["counts"]["pool"],
                                     "ids_sha256": L.sha256_bytes("\n".join(c["pool"]).encode())},
                      "leakage_vs_train_pool": c["leakage_dev_vs_pool"]})
    sel = split_plan("s2sel", c["sel"], doc, split_sha,
                     {"rule": "s2val + s2dev74 (SEL, DECISIONS G4(a))",
                      "members": {"s2val": c["s2val"], "s2dev74": c["dev"]}})
    return c, dev, sel


EXPLORATORY_NOTE = ("EXPLORATORY (DECISIONS G4): WalkerBench limited exploration, Q3B recipe variants on SEL = s2val + "
                    "s2dev74. Not part of the pre-registered pilot; no gate is decided here; s2test is never used. "
                    "See vpd/EXPLORATORY_WB_VARIANTS.md.\n")


def setup(x: Path = X_ROOT, split_path: Path = SPLIT_S2, dev_dest: Path = S2DEV_DIR, sel_dest: Path = S2SEL_DIR,
          src_tasks: Path = paths.WB_S2TRAIN_TASKS) -> dict:
    """s2dev74 and s2sel (created or verified) and the X dirs (mkdir + X/EXPLORATORY.md only)."""
    x = Path(x)
    refuse_pilot(x)
    doc = split_doc(split_path)
    c, dev, sel = carve_plans(doc, sha256_file(split_path))
    rep = {"s2dev74": build_split(dev_dest, dev, src_tasks), "s2sel": build_split(sel_dest, sel, src_tasks),
           "leakage_dev_vs_pool": c["leakage_dev_vs_pool"], "pool": len(c["pool"])}
    for d in (x, x / "floors", x / "analysis"):
        d.mkdir(parents=True, exist_ok=True)
    if not (x / "EXPLORATORY.md").exists():
        write_text(x / "EXPLORATORY.md", EXPLORATORY_NOTE)
    rep["x"] = str(x)
    return rep


def register_task_sets(run_vpd=None):
    """Add 's2dev74' and 's2sel' to run_vpd.TASK_SETS in this process only (run_vpd.py is not edited)."""
    if run_vpd is None:
        from vpd.walkerbench import run_vpd
    for name, tdir in TASK_SETS.items():
        cur = run_vpd.TASK_SETS.get(name)
        if cur is None:
            run_vpd.TASK_SETS[name] = tdir
        elif Path(cur) != Path(tdir):
            raise SystemExit(f"run_vpd.TASK_SETS[{name!r}] is {cur}, expected {tdir}")
    return run_vpd


# ------------------------------------------------------------------------------------------------- data
def vdir(x: Path, variant: str) -> Path:
    return Path(x) / variant


def data_path(x: Path, variant: str) -> Path:
    return vdir(x, variant) / "data" / "train.jsonl"


def pool_lines(order_path: Path = ORDER, tier_all: Path = TIER_ALL, split_path: Path = SPLIT_S2) -> tuple:
    """(ids, verbatim lines) of the 409 exploration tasks in tier order; asserts the tier file is the order."""
    od = json.loads(Path(order_path).read_text())
    lines = [ln for ln in Path(tier_all).read_text(encoding="utf-8").split("\n") if ln.strip()]
    ids = [json.loads(ln)["id"] for ln in lines]
    if ids != list(od["order"]):
        raise SystemExit(f"{tier_all}: ids are not the order of {order_path}")
    c = dev_carve(split_doc(split_path))
    pool = set(c["pool"])
    if pool != set(ids) - set(c["dev"]):
        raise SystemExit("the 409 pool is not the tier order minus s2dev74")
    keep = [(i, ln) for i, ln in zip(ids, lines) if i in pool]
    if len(keep) != POOL_N:
        raise SystemExit(f"{len(keep)} pool demos, expected {POOL_N}")
    return [i for i, _ in keep], [ln for _, ln in keep]


def reply_class(action: str) -> str:
    """STOP | TURN (ROTATION or MOVE(k != 1)) | STRAIGHT (MOVE(1))."""
    if action.startswith("STOP("):
        return "STOP"
    if action.startswith("ROTATION("):
        return "TURN"
    k = O.move_id(action)
    if k is None:
        raise ValueError(f"not a WalkerBench action: {action!r}")
    return "STRAIGHT" if k == 1 else "TURN"


def with_weights(rec: dict, weights=V5_WEIGHTS) -> dict:
    out = json.loads(json.dumps(rec))
    for m in out["messages"]:
        if m["role"] == "assistant":
            m["loss_weight"] = float(weights[reply_class(O.reply_action(m["content"]))])
    return out


def check_record(rec: dict, mode: tuple) -> list:
    """Problems of one rewritten demo: every reply fullmatches its step's eval regex (run_vpd.step_regex of the
    variant's constrained mode; 'off' is checked against 'short'), the odometer's walked equals learn483.turn_table's
    walked_m, V1T lines exactly once per user turn, the '<image>' placeholder still at the end."""
    from vpd.walkerbench import run_vpd as R

    cmode, odo = mode
    probs = []
    turns = O.demo_turns(rec["messages"])
    tt = L.turn_table(rec)
    if len(tt) != len(turns):
        probs.append(f"{rec['id']}: {len(turns)} odometer turns vs {len(tt)} turn_table rows")
    for a, b in zip(turns, tt):
        if abs(a["walked"] - b["walked_m"]) > 1e-6:
            probs.append(f"{rec['id']} turn {b['turn']}: walked {a['walked']} != turn_table {b['walked_m']}")
    for t in turns:
        reply = rec["messages"][t["msg_index"]]["content"]
        rx = R.step_regex(len(t["moves"]), cmode if cmode != "off" else "short", THOUGHT_MAX_CHARS)
        if re.fullmatch(rx, reply) is None:
            probs.append(f"{rec['id']} msg {t['msg_index']}: reply fails the {cmode} regex: {reply[:80]}")
        u = rec["messages"][t["user_index"]]["content"]
        n_lines = sum(1 for ln in u.split("\n") if ln.startswith(O.LINE_PREFIX))
        if n_lines != (1 if odo == "line" else 0):
            probs.append(f"{rec['id']} msg {t['user_index']}: {n_lines} odometer lines")
        if u.count(O.IMAGE) != 1 or not u.endswith(O.IMAGE):
            probs.append(f"{rec['id']} msg {t['user_index']}: '<image>' not once at the end")
    return probs


def base_manifest(x: Path, base: str) -> dict:
    man = read_json(vdir(x, base) / "data" / "manifest.json")
    p = data_path(x, base)
    if not man or not p.is_file():
        raise SystemExit(f"[variants] base {base}: {p} or its manifest.json is missing (build it first)")
    if sha256_file(p) != man.get("sha256"):
        raise SystemExit(f"[variants] base {base}: {p} changed after its manifest was written")
    return man


def eval_mode(variant: str, base: str | None = None, x: Path = X_ROOT) -> tuple:
    """(constrained, odometer) of the variant's evals: V0 empty/off, V1T empty/line, V2 short/off; V5 and V4 inherit
    their base's (from the data manifest once it exists)."""
    if variant in STATIC_MODE:
        if base:
            raise SystemExit(f"[variants] {variant} takes no --base")
        return STATIC_MODE[variant]
    if variant not in ("V5", "V4"):
        raise SystemExit(f"[variants] unknown variant {variant!r}")
    man = read_json(vdir(x, variant) / "data" / "manifest.json")
    if man:
        if base and man.get("base") != base:
            raise SystemExit(f"[variants] {variant} data was built on {man.get('base')}, not {base}")
        return tuple(man["eval"]["constrained_odometer"])
    allowed = V5_BASES if variant == "V5" else V4_BASES
    if base not in allowed:
        raise SystemExit(f"[variants] {variant} needs --base in {allowed}")
    if base in STATIC_MODE:
        return STATIC_MODE[base]
    return eval_mode(base, None, x)  # V4 on V5: V5's manifest (V5 ran first)


def build_data(variant: str, x: Path = X_ROOT, base: str | None = None, order_path: Path = ORDER,
               tier_all: Path = TIER_ALL, split_path: Path = SPLIT_S2) -> dict:
    x = Path(x)
    refuse_pilot(x)
    mode = eval_mode(variant, base, x)
    if variant in ("V0", "V1T", "V2"):
        ids, lines = pool_lines(order_path, tier_all, split_path)
        source = {"file": str(tier_all), "sha256": sha256_file(tier_all), "order": str(order_path),
                  "order_sha256": sha256_file(order_path), "split_s2_sha256": sha256_file(split_path)}
        if variant == "V0":
            out_lines = lines
            for ln in lines:  # verbatim: the JSON round trip would reproduce the line, so V1T/V2 differ only by edits
                if json.dumps(json.loads(ln), ensure_ascii=False) != ln:
                    raise SystemExit("a tier line does not round-trip through json; refusing to rewrite demos")
        else:
            fn = O.with_line if variant == "V1T" else O.with_thought
            out_lines = [json.dumps(fn(json.loads(ln)), ensure_ascii=False) for ln in lines]
    else:
        bman = base_manifest(x, base)
        src = data_path(x, base)
        source = {"file": str(src), "sha256": bman["sha256"], "base_manifest": str(vdir(x, base) / "data" /
                                                                                   "manifest.json")}
        lines = [ln for ln in src.read_text(encoding="utf-8").split("\n") if ln.strip()]
        ids = [json.loads(ln)["id"] for ln in lines]
        if variant == "V5":
            if bman.get("variant") not in V5_BASES:
                raise SystemExit(f"[variants] V5 needs a base in {V5_BASES}")
            out_lines = [json.dumps(with_weights(json.loads(ln)), ensure_ascii=False) for ln in lines]
        else:
            out_lines = lines
    probs, classes = [], Counter()
    for ln in out_lines:
        rec = json.loads(ln)
        probs += check_record(rec, mode)
        for m in rec["messages"]:
            if m["role"] == "assistant":
                classes[reply_class(O.reply_action(m["content"]))] += 1
    if probs:
        raise SystemExit(f"[variants] {variant}: {len(probs)} problems, e.g. {probs[:3]}")
    text = "".join(ln + "\n" for ln in out_lines)
    p = data_path(x, variant)
    st = _write_if_new(p, text)
    man = {"schema": SCHEMA + ":data", "exploratory": True, "variant": variant, "base": base, "rule": RULES[variant],
           "source": source, "file": str(p), "sha256": sha256_text(text), "n": len(out_lines),
           "ids_sha256": L.sha256_bytes("\n".join(ids).encode()), "counts": kinds(ids),
           "reply_classes": dict(sorted(classes.items())),
           "loss_weight": dict(V5_WEIGHTS) if variant == "V5" or (variant == "V4" and base == "V5") else None,
           "epochs": EPOCHS[variant],
           "eval": {"constrained_odometer": list(mode),
                    "args": eval_args(mode)},
           "code": code_state()}
    mp = vdir(x, variant) / "data" / "manifest.json"
    old = read_json(mp)
    if st == "wrote" or not old or old.get("sha256") != man["sha256"]:
        write_json(mp, dict(man, created_utc=utc()))
    return {"state": st, "file": str(p), "n": len(out_lines), "sha256": man["sha256"], "eval": list(mode)}


def eval_args(mode: tuple) -> list:
    cmode, odo = mode
    out = ["--constrained", cmode, "--max-tokens", "1024" if cmode == "off" else "512"]
    if cmode == "short":
        out += ["--thought-max-chars", str(THOUGHT_MAX_CHARS)]
    return out + ["--odometer", odo]


# ------------------------------------------------------------------------------------------------- checks
def init_check(init_dir: Path = INIT, expect_sha: str = INIT_SHA_PREFIX, need_loss_weight: bool = False,
               trainer: Path | None = None) -> tuple:
    """The pilot's Q3B 0W, read only: adapter.pt sha256 prefix, finished, start Q3B.  No trainer-code equality with
    the pilot (the trainer may have changed since: wbtrain's per-reply loss_weight).  With need_loss_weight (V5),
    the trainer must read the per-message 'loss_weight' field (else V5 would silently train unweighted)."""
    init_dir = Path(init_dir)
    probs, facts = [], {}
    pt = init_dir / "adapter.pt"
    facts["init_sha256"] = sha256_file(pt) if pt.is_file() else None
    if not facts["init_sha256"] or not facts["init_sha256"].startswith(expect_sha):
        probs.append(f"{pt}: sha256 {facts['init_sha256']} does not start with {expect_sha}")
    meta = read_json(init_dir / "meta.json") or {}
    facts["init_start"], facts["init_phase"] = meta.get("start"), meta.get("phase")
    if meta.get("start") != START:
        probs.append(f"{init_dir}/meta.json start {meta.get('start')!r} != {START}")
    if not meta.get("optimizer_steps") or meta.get("optimizer_steps") != meta.get("optimizer_steps_planned"):
        probs.append(f"{init_dir}: unfinished ({meta.get('optimizer_steps')} of {meta.get('optimizer_steps_planned')})")
    if need_loss_weight:
        tp = Path(trainer or paths.WORKTREE / "vpd" / "train" / "n2_train_vpd.py")
        src = tp.read_text(encoding="utf-8") if tp.is_file() else ""
        ok = '"loss_weight"' in src or "'loss_weight'" in src
        if ok and trainer is None:  # the real trainer: its reply_weights must return the message's weight
            try:
                from vpd.train import n2_train_vpd as T

                probe = {"id": "probe", "messages": [{"role": "system", "content": "s"},
                                                     {"role": "user", "content": "u"},
                                                     {"role": "assistant", "content": "a", "loss_weight": 5.0}]}
                ok = dict(T.reply_weights(probe)) == {2: 5.0}
            except Exception as e:  # noqa: BLE001 - recorded as the reason
                facts["trainer_probe_error"] = f"{type(e).__name__}: {e}"[:300]
                ok = False
        facts["trainer_reads_loss_weight"] = ok
        if not ok:
            probs.append(f"{tp} does not read the per-message 'loss_weight' field (V5 needs package wbtrain's change)")
    return probs, facts


def adapter_check(variant: str, x: Path = X_ROOT, init: Path = INIT) -> tuple:
    """(0, ok) finished with the current trainer code, from INIT/adapter.pt as now, on X/<V>/data/train.jsonl as now,
    with EPOCHS[V]; (1, why) missing / unfinished / other data, init or epochs: (re)train; (3, why) finished with
    other training code: the job stops, a person decides."""
    from vpd.train import n2_train_vpd as T

    d = vdir(x, variant) / "adapter"
    meta = read_json(d / "meta.json")
    if meta is None or not (d / "adapter.pt").is_file():
        return 1, f"{d}: no adapter.pt + meta.json"
    steps, planned = meta.get("optimizer_steps"), meta.get("optimizer_steps_planned")
    if not steps or (planned is not None and steps != planned):
        return 1, f"{d}: unfinished ({steps} of {planned} optimizer steps)"
    cur = T.code_fingerprint()["code_sha256"]
    if meta.get("code_sha256") != cur:
        return 3, f"{d}: trained with code {meta.get('code_sha256')}, current {cur}"
    dp = data_path(x, variant)
    if not dp.is_file() or meta.get("data_sha256") != sha256_file(dp):
        return 1, f"{d}: trained on data {meta.get('data_sha256')}, {dp} differs or is missing: retrain"
    if int(meta.get("epochs") or 0) != EPOCHS[variant] or meta.get("start") != START:
        return 1, f"{d}: epochs {meta.get('epochs')} / start {meta.get('start')} != {EPOCHS[variant]} / {START}"
    want = sha256_file(Path(init) / "adapter.pt") if (Path(init) / "adapter.pt").is_file() else None
    got = (meta.get("init_adapter") or {}).get("sha256")
    if want is None or got != want or not str(got).startswith(INIT_SHA_PREFIX):
        return 1, f"{d}: started from init {got}, {init}/adapter.pt is {want}: retrain"
    dman = read_json(vdir(x, variant) / "data" / "manifest.json") or {}
    if dman.get("loss_weight") and ((meta.get("loss_weights") or {}).get("mode") != "weighted"):
        return 1, f"{d}: the data carries loss weights but meta.loss_weights is {meta.get('loss_weights')}: retrain"
    return 0, f"{d}: complete ({steps} steps, code {cur[:12]})"


def _fake_rm_case():
    from types import SimpleNamespace

    s = SimpleNamespace(task_config={"visual_path": [{"pano_id": "a"}, {"pano_id": "b"}]},
                        state=SimpleNamespace(pano_id="a", heading=30.0))
    return s, (lambda: [{"pano_id": "b", "heading": -164.8, "id": 1}])


def b1_precheck() -> tuple:
    """The B1 fix is in this code: rm_expert at heading 30 with the next pano at -164.8 rotates to 195.2."""
    from vpd.walkerbench import baselines as B

    s, moves = _fake_rm_case()
    act = B.rm_expert(s, moves, set())
    ok = act.get("type") == "rotation" and act.get("heading") == 195.2
    return ([] if ok else [f"baselines.rm_expert returned {act} (B1 heading wrap not in this code)"]), \
        {"rm_expert_case": act, "baselines_sha256": sha256_file(B.__file__)}


def b1_after(out_dir: Path = B1_OUT, before: dict | None = None, frozen: Path = FROZEN_DEMOS) -> tuple:
    """The rebuilt single demo (G6): 1 record, B1_STEM, no drop, the wrapped rotation present, every ROTATION
    heading in [0, 360), its images on disk; the frozen demos_all.jsonl unchanged (sha256 == the constant and ==
    the 'before' record)."""
    out_dir = Path(out_dir)
    probs, facts = [], {}
    facts["frozen_sha256_after"] = sha256_file(frozen) if Path(frozen).is_file() else None
    if facts["frozen_sha256_after"] != FROZEN_DEMOS_SHA256:
        probs.append(f"{frozen}: sha256 {facts['frozen_sha256_after']} != {FROZEN_DEMOS_SHA256}")
    if before is not None and before.get("frozen_sha256") != facts["frozen_sha256_after"]:
        probs.append(f"{frozen} changed during the rebuild")
    da = out_dir / "demos_all.jsonl"
    recs = read_jsonl(da) if da.is_file() else []
    facts["records"] = [r.get("id") for r in recs]
    if [r.get("id") for r in recs] != [B1_STEM]:
        probs.append(f"{da}: records {facts['records']} != [{B1_STEM}]")
    man = read_json(out_dir / "manifest.json") or {}
    facts["dropped"] = man.get("dropped")
    if man.get("dropped") != []:
        probs.append(f"{out_dir}/manifest.json: dropped {man.get('dropped')!r}")
    facts["baselines_sha256"] = (man.get("code") or {}).get("baselines.py (rm_expert)")
    for r in recs:
        acts = [O.reply_action(m["content"]) for m in r["messages"] if m["role"] == "assistant"]
        heads = [float(h) for h in re.findall(r"ROTATION\((-?\d+(?:\.\d+)?), ", " ".join(acts))]
        facts["rotations"] = [a for a in acts if a.startswith("ROTATION(")]
        if B1_ROTATION not in acts:
            probs.append(f"{B1_STEM}: {B1_ROTATION} not among the replies {facts['rotations']}")
        if any(not 0 <= h < 360 for h in heads):
            probs.append(f"{B1_STEM}: a ROTATION heading outside [0, 360): {heads}")
        missing = [p for p in r.get("images") or [] if not Path(p).is_file()]
        if missing or len(r.get("images") or []) != r.get("n_turns"):
            probs.append(f"{B1_STEM}: {len(r.get('images') or [])} images for {r.get('n_turns')} turns, "
                         f"{len(missing)} missing")
    facts["demos_all_sha256"] = sha256_file(da) if da.is_file() else None
    return probs, facts


# ------------------------------------------------------------------------------------------------- floors
def floors(out: Path, logs: Path, work: Path, tasks_dir: Path = S2SEL_TASKS, split_path: Path = SPLIT_S2,
           stems: list | None = None) -> dict:
    """STOP0 / FWD / ODO / RM on s2sel via learn483.run_floors (rendering stubbed, one sandbox), plus the per-subset
    floors (SEL, s2val, s2dev74) next to it."""
    tasks_dir = Path(tasks_dir)
    refuse_s2test("s2sel", tasks_dir)
    doc = L.run_floors("s2sel", Path(out), Path(logs), Path(work), stems, tasks_dir=tasks_dir)
    sub = floor_subsets(doc, members(tasks_dir.parent))
    write_json(Path(out).with_name(Path(out).stem + "_subsets.json"),
               {"schema": SCHEMA + ":floors_subsets", "exploratory": True, "source": str(out), "subsets": sub})
    return {"floors": doc["floors"], "subsets": sub}


def members(sel_dir: Path = S2SEL_DIR) -> dict:
    sel = read_json(Path(sel_dir) / "selection.json") or {}
    m = sel.get("members") or {}
    out = OrderedDict([("SEL", sorted(sel.get("ids") or []))])
    for k in ("s2val", "s2dev74"):
        out[k] = sorted(m.get(k) or [])
    return out


def floor_subsets(doc: dict, subsets: dict) -> dict:
    out = OrderedDict()
    pols = doc.get("policies") or {}
    for name, stems in subsets.items():
        row = OrderedDict()
        for m in ("STOP0", "FWD", "ODO", "RM"):
            pt = (pols.get(m) or {}).get("per_task") or {}
            if not pt:
                continue
            row[m] = {k: rate(sum(1 for s in stems if s.startswith(k) and (pt.get(s) or {}).get("success")),
                              sum(1 for s in stems if s.startswith(k))) for k in ("nav", "vis")}
        out[name] = row
    return out


# ------------------------------------------------------------------------------------------------- eval
def eval_name(model: str, task_set: str, constrained: str, odometer: str = "off", blind: bool = False) -> str:
    return (f"wbv__{START}__{model}__{task_set}__640__{constrained}" + ("__odo" if odometer == "line" else "")
            + ("__blind" if blind else ""))


def job_evals(variant: str, base: str | None = None, x: Path = X_ROOT) -> tuple:
    if variant in JOB_EVALS:
        return JOB_EVALS[variant]
    cm, odo = eval_mode(variant, base, x)
    return ((variant, variant, "s2sel", cm, odo, False), (variant, variant, "s2sel", cm, odo, True))


def odometer_agent(cls, R):
    """The run_vpd EvalAgent with the odometer line in every user turn (V1 probes, V1T).  walked: the shown distance
    of the chosen MOVE (observation['available_moves'] at that step) is added once the move executed successfully;
    the line is odometer.insert_line of the stock text, as the V1T demo rewrite."""

    class OdometerAgent(cls):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._odo = O.Odometer()
            self._odo_moves, self._odo_pending = {}, None

        def create_session(self, *a, **k):
            self._odo = O.Odometer()  # one episode per agent in WalkerBench's loop; reset anyway
            self._odo_moves, self._odo_pending = {}, None
            return super().create_session(*a, **k)

        def decide_action(self, observation):
            self._odo_moves = O.moves_in_obs(observation.get("available_moves"))
            self._odo_pending = None
            action = super().decide_action(observation)
            if isinstance(action, dict) and action.get("type") == "move":
                self._odo_pending = self._odo_moves.get(action.get("move_id"))
            return action

        def _build_user_message(self, observation):
            msg = super()._build_user_message(observation)
            part = msg["content"][0] if isinstance(msg.get("content"), list) else None
            if not part or part.get("type") != "text":
                raise RuntimeError("the stock user message does not start with a text part")
            part["text"] = O.insert_line(part["text"], self._odo.walked)
            with R.LOCK:
                R.COUNTS["odometer_lines"] += 1
            return msg

        def execute_action(self, action):
            result = super().execute_action(action)
            if action.get("type") == "move" and result.get("success") and self._odo_pending is not None:
                self._odo.add(self._odo_pending)
                with R.LOCK:
                    R.COUNTS["odometer_moves_added"] += 1
            self._odo_pending = None
            return result

    OdometerAgent.__name__ = f"Odometer{cls.__name__}"
    return OdometerAgent


def install_odometer(R, mode: str):
    """Wrap run_vpd.make_eval_agent (looked up at call time by run_tasks) when mode == 'line'; returns the undo."""
    orig = R.make_eval_agent
    if mode == "off":
        return lambda: None
    if mode != "line":
        raise SystemExit(f"--odometer must be off or line, got {mode!r}")

    def make(base_agent, opts, copy):
        return odometer_agent(orig(base_agent, opts, copy), R)

    R.make_eval_agent = make

    def undo():
        R.make_eval_agent = orig
    return undo


def split_eval_argv(argv: list) -> tuple:
    """(odometer mode, argv without --odometer) ; '--odometer X' and '--odometer=X' forms."""
    mode, out, i = "off", [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--odometer":
            if i + 1 >= len(argv):
                raise SystemExit("--odometer needs a value (off|line)")
            mode, i = argv[i + 1], i + 2
            continue
        if a.startswith("--odometer="):
            mode, i = a.split("=", 1)[1], i + 1
            continue
        out.append(a)
        i += 1
    if mode not in ("off", "line"):
        raise SystemExit(f"--odometer must be off or line, got {mode!r}")
    return mode, out


def cmd_eval(argv: list) -> int:
    mode, rest = split_eval_argv(list(argv))
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--run-name")
    ap.add_argument("--out-root", type=Path, default=X_ROOT)
    ap.add_argument("--task-set")
    ap.add_argument("--rescore", action="store_true")
    a, _ = ap.parse_known_args(rest)
    if a.rescore:
        raise SystemExit("variants eval: --rescore goes through run_vpd directly")
    if a.run_dir is None and not a.run_name:
        raise SystemExit("variants eval: give --run-name (and --out-root) or --run-dir")
    if a.run_dir is None and not any(x == "--out-root" or x.startswith("--out-root=") for x in rest):
        rest = rest + ["--out-root", str(a.out_root)]  # run_vpd's own default is runs/, never used here
    run_dir = (a.run_dir or a.out_root / a.run_name).resolve()
    refuse_pilot(run_dir)
    refuse_s2test(a.task_set, TASK_SETS.get(a.task_set) or (paths.WB_S2VAL_TASKS if a.task_set == "s2val" else None))
    R = register_task_sets()
    undo = install_odometer(R, mode)
    rc = 1
    try:
        R.main(rest)
        rc = 0
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if rc and not isinstance(e.code, int):
            print(e.code, file=sys.stderr)
    finally:
        undo()
        if run_dir.is_dir():
            write_json(run_dir / "explore.json", {"schema": SCHEMA + ":eval", "exploratory": True, "utc": utc(),
                                                  "wrapper": "python -m vpd.walkerbench.variants eval",
                                                  "odometer": mode, "argv": list(argv), "rc": rc,
                                                  "task_sets_added": {k: str(v) for k, v in TASK_SETS.items()},
                                                  "odometer_counts": {k: R.COUNTS.get(k, 0) for k in
                                                                      ("odometer_lines", "odometer_moves_added")},
                                                  "code": code_state()})
    return rc


# ------------------------------------------------------------------------------------------------- analysis
def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact binomial test on the discordant pairs (b image-only wins, c blind-only wins), p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def bar_k(n: int, floor_rate: float) -> int | None:
    """The smallest k whose frozen Wilson lower bound (pilot_gate.wilson_lb) exceeds floor_rate."""
    from vpd.walkerbench.pilot_gate import wilson_lb

    return next((k for k in range(n + 1) if wilson_lb(k, n) > floor_rate), None)


def eval_dirs(x: Path) -> list:
    out = []
    for p in sorted(Path(x).glob("*/evals/*")):
        m = EVAL_NAME_RE.match(p.name)
        if p.is_dir() and m:
            out.append((p.parent.parent.name, p, m.groupdict()))
    return out


def per_task_rows(d: Path) -> dict:
    p = Path(d) / "per_task.jsonl"
    return {r["stem"]: r for r in read_jsonl(p)} if p.is_file() else {}


def kind_table(rows: dict, subsets: dict) -> dict:
    out = OrderedDict()
    for name, stems in subsets.items():
        out[name] = OrderedDict()
        for k in ("nav", "vis"):
            ss = [s for s in stems if s.startswith(k)]
            out[name][k] = rate(sum(int((rows.get(s) or {}).get("success") or 0) for s in ss), len(ss))
            out[name][k + "_oracle"] = rate(sum(1 for s in ss if (rows.get(s) or {}).get("oracle") == 1), len(ss))
        ss = list(stems)
        out[name]["pooled"] = rate(sum(int((rows.get(s) or {}).get("success") or 0) for s in ss), len(ss))
    return out


def thought_stats(d: Path, rows: dict, tasks: dict) -> dict:
    """THOUGHT format rate and odometer error of a short-THOUGHT eval: per answered call (the last one per task and
    step), parse odometer.parse_thought(THOUGHT) and compare it with the true walked before that step (the shown
    distances of the executed MOVEs in the episode log)."""
    d = Path(d)
    logs = d / "WalkerBench" / "logs"
    truth = {}
    for stem, r in rows.items():
        if not r.get("log") or not (logs / r["log"]).is_file():
            continue
        _, acts = L.parse_log(logs / r["log"])
        w = 0.0
        for i, a in enumerate(acts):
            truth[(stem, i + 1)] = w
            act = a.get("action") or {}
            if act.get("type") == "move":
                dist = O.moves_in_obs(a.get("available_moves")).get(act.get("move_id"))
                if dist is not None:
                    w = O.add(w, dist)
    calls = {}
    cp = d / "calls.jsonl"
    for c in (read_jsonl(cp) if cp.is_file() else []):
        if c.get("error") is None and isinstance(c.get("reply"), str):
            calls[(c.get("task"), c.get("step"))] = c["reply"]
    n = fmt_ok = exact = stated_ok = stated_n = 0
    errs = []
    for key, reply in calls.items():
        if key not in truth:
            continue
        n += 1
        try:
            th = json.loads(reply).get("THOUGHT")
        except (ValueError, AttributeError):
            th = None
        p = O.parse_thought(th)
        if p is None:
            continue
        fmt_ok += 1
        e = abs(p["walked"] - truth[key])
        errs.append(e)
        exact += e < 0.05
        stem = key[0]
        if stem.startswith("nav"):
            st = O.stated_total((tasks.get(stem) or {}).get("description", ""))
            stated_n += 1
            stated_ok += p["stated"] == st
    return {"calls": n, "format_ok": rate(fmt_ok, n), "walked_exact_0.05": rate(exact, fmt_ok),
            "walked_abs_err_mean": L.mean(errs), "walked_abs_err_median": L.median(errs),
            "walked_abs_err_max": round(max(errs), 2) if errs else None,
            "nav_stated_matches_task": rate(stated_ok, stated_n),
            "rule": "THOUGHT == odometer.thought format; truth = shown distances of the executed MOVEs before the step"}


def analyze(x: Path = X_ROOT, sel_dir: Path = S2SEL_DIR, db=L.CACHE_DB) -> dict:
    from vpd.walkerbench.pilot_gate import wilson_lb  # the frozen formula (PREREG_WB_PILOT section 9)

    x = Path(x)
    refuse_pilot(x)
    subsets = members(sel_dir)
    tdir = Path(sel_dir) / "tasks"
    tasks = L.load_tasks(tdir) if tdir.is_dir() else {}
    pano = set()
    for t in tasks.values():
        pano |= L.task_panos(t)
    coords = L.load_coords(pano, db)
    fl = read_json(x / "floors" / "floors_s2sel.json") or {}
    rm_logs = L.policy_logs(Path(fl["logs"]) / "RM") if fl.get("logs") else {}
    nodes = L.expert_nodes(tasks, rm_logs)
    fsub = floor_subsets(fl, subsets) if fl else {}
    doc = OrderedDict([("schema", SCHEMA + ":analyze"), ("utc", utc()), ("exploratory", True),
                       ("note", "numbers only (DECISIONS G4); no gate is decided here; s2test is never used"),
                       ("x", str(x)), ("subsets", {k: len(v) for k, v in subsets.items()}),
                       ("floors", {"source": str(x / "floors" / "floors_s2sel.json") if fl else None,
                                   "subsets": fsub}),
                       ("evals", OrderedDict()), ("variants", OrderedDict())])
    sel_floor = {}
    if fsub.get("SEL"):
        sel_floor = {"nav": fsub["SEL"].get("ODO", {}).get("nav"), "vis": fsub["SEL"].get("FWD", {}).get("vis")}
    doc["floors"]["g4_floor"] = {"nav": "ODO", "vis": "FWD", "SEL": sel_floor}
    rows_of = {}
    for owner, d, g in eval_dirs(x):
        key = f"{owner}/{d.name}"
        res = read_json(d / "results.json")
        ex = read_json(d / "explore.json") or {}
        e = OrderedDict([("dir", str(d)), ("owner", owner), ("model", g["model"]), ("task_set", g["set"]),
                         ("constrained", g["mode"]), ("odometer", "line" if g["odo"] else "off"),
                         ("blind", bool(g["blind"]))])
        if res is None:
            e["state"] = "invalid" if (d / "results_invalid.json").exists() else "no results.json"
            doc["evals"][key] = e
            continue
        if ex.get("odometer") not in (None, e["odometer"]):
            e["odometer_mismatch"] = ex.get("odometer")
        rows = per_task_rows(d)
        rows_of[key] = rows
        e.update(state="ok", adapter_sha256=(res.get("adapter") or {}).get("adapter_sha256"),
                 blind_results=res.get("blind"), regex=res.get("regex"), tasks_without_log=res.get("tasks_without_log"),
                 parse_valid_pct=res.get("parse_valid_pct"), wall_seconds=res.get("wall_seconds"),
                 odometer_counts=ex.get("odometer_counts"), table=kind_table(rows, subsets))
        if tasks:
            e["behaviour_SEL"] = L.summarize_eval(L.eval_episodes(d, tasks, coords, nodes))
        if g["mode"] == "short":
            e["thought"] = thought_stats(d, rows, tasks)
        doc["evals"][key] = e
    for v in TRAINED:
        own = {k: e for k, e in doc["evals"].items() if e["owner"] == v and e["model"] == v}
        prim = next((k for k, e in own.items() if not e["blind"]), None)
        blind = next((k for k, e in own.items() if e["blind"]), None)
        if prim is None:
            continue
        pe = doc["evals"][prim]
        vd = OrderedDict([("primary", prim), ("blind", blind), ("state", pe.get("state"))])
        if pe.get("state") != "ok":
            doc["variants"][v] = vd
            continue
        t = pe["table"]["SEL"]
        vd["selection_score_nav_k_plus_vis_k"] = t["nav"]["k"] + t["vis"]["k"]
        checks = OrderedDict()
        for k in ("nav", "vis"):
            c = OrderedDict([("k", t[k]["k"]), ("n", t[k]["n"]), ("rate", t[k]["rate"])])
            c["wilson_lb"] = round(wilson_lb(t[k]["k"], t[k]["n"]), 4) if t[k]["n"] else None
            f = sel_floor.get(k)
            if f and f.get("n"):
                c["floor"] = {"policy": "ODO" if k == "nav" else "FWD", "k": f["k"], "n": f["n"], "rate": f["rate"]}
                c["bar_k"] = bar_k(t[k]["n"], f["k"] / f["n"])
                c["A2_lb_above_floor"] = c["wilson_lb"] is not None and c["wilson_lb"] > f["k"] / f["n"]
            else:
                c["floor"], c["A2_lb_above_floor"] = None, None
            be = doc["evals"].get(blind) if blind else None
            if be and be.get("state") == "ok":
                ra, rb = rows_of[prim], rows_of[blind]
                ss = [s for s in subsets["SEL"] if s.startswith(k)]
                b = sum(1 for s in ss if (ra.get(s) or {}).get("success") and not (rb.get(s) or {}).get("success"))
                cc = sum(1 for s in ss if (rb.get(s) or {}).get("success") and not (ra.get(s) or {}).get("success"))
                p = exact_mcnemar(b, cc)
                c["A3_paired"] = {"image_only": b, "blind_only": cc, "p_two_sided_exact": round(p, 5),
                                  "blind_k": be["table"]["SEL"][k]["k"]}
                c["A3_images_better_p_lt_0.05"] = b > cc and p < ALPHA_A3
            else:
                c["A3_paired"], c["A3_images_better_p_lt_0.05"] = None, None
            c["A4_rate_le_0.80"] = t[k]["rate"] is not None and t[k]["rate"] <= RATE_MAX_A4
            vals = [c["A2_lb_above_floor"], c["A3_images_better_p_lt_0.05"], c["A4_rate_le_0.80"]]
            c["all_three"] = None if any(v is None for v in vals) else all(vals)
            checks[k] = c
        vd["g4g"] = checks
        vd["pooled_rate"] = t["pooled"]["rate"]
        dman = read_json(vdir(x, v) / "data" / "manifest.json") or {}
        vd["data"] = {k: dman.get(k) for k in ("base", "rule", "sha256", "n", "epochs", "loss_weight")}
        meta = read_json(vdir(x, v) / "adapter" / "meta.json") or {}
        vd["adapter"] = {k: meta.get(k) for k in ("epochs", "optimizer_steps", "loss_first", "loss_last",
                                                  "loss_last10_mean", "train_seconds", "code_sha256", "data_sha256",
                                                  "loss_weights")}
        doc["variants"][v] = vd
    navs = [vd["g4g"]["nav"]["k"] for vd in doc["variants"].values() if vd.get("g4g")]
    doc["g4h_inputs"] = {"best_nav_k_SEL": max(navs) if navs else None,
                         "vis_image_vs_blind_discordance": {v: (vd["g4g"]["vis"].get("A3_paired") or {})
                                                            for v, vd in doc["variants"].items() if vd.get("g4g")}}
    out = x / "analysis"
    write_json(out / "ANALYZE.json", doc)
    write_text(out / "REPORT_EXPLORATORY.md", report_md(doc))
    return doc


def _r(b) -> str:
    if not b or not b.get("n"):
        return "-"
    ci = b.get("wilson95") or [0, 0]
    return f"{b['k']}/{b['n']} ({100 * b['rate']:.1f}%, {100 * ci[0]:.1f}-{100 * ci[1]:.1f})"


def _k(b) -> str:
    return "-" if not b or not b.get("n") else f"{b['k']}/{b['n']}"


def report_md(doc: dict) -> str:
    L_ = ["# WalkerBench limited exploration (EXPLORATORY)", "",
          f"Generated {doc['utc']}. Q3B only; SEL = s2val + s2dev74. Numbers only: no gate is decided here and s2test "
          "is never used (DECISIONS G4).", ""]
    fs = (doc.get("floors") or {}).get("subsets") or {}
    if fs:
        L_ += ["## Floors (in-job recompute, rendering stubbed)", "", "| subset | policy | nav | vis |", "|---|---|---|---|"]
        for sub, row in fs.items():
            for m, v in row.items():
                L_.append(f"| {sub} | {m} | {_r(v.get('nav'))} | {_r(v.get('vis'))} |")
        L_.append("")
    L_ += ["## Evals", "", "| eval | state | SEL nav | SEL vis | s2val nav | s2val vis | s2dev74 nav | s2dev74 vis | "
           "oracle nav | oracle vis | ended STOP | cap hits |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, e in doc["evals"].items():
        if e.get("state") != "ok":
            L_.append(f"| {key} | {e.get('state')} | | | | | | | | | | |")
            continue
        t = e["table"]
        b = (e.get("behaviour_SEL") or {}).get("all") or {}
        L_.append(f"| {key} | ok | {_r(t['SEL']['nav'])} | {_r(t['SEL']['vis'])} | {_k(t['s2val']['nav'])} | "
                  f"{_k(t['s2val']['vis'])} | {_k(t['s2dev74']['nav'])} | {_k(t['s2dev74']['vis'])} | "
                  f"{_k(t['SEL']['nav_oracle'])} | {_k(t['SEL']['vis_oracle'])} | {_k(b.get('ended_stop'))} | "
                  f"{b.get('cap_hits', '-')} |")
    L_ += ["", "## G4(g) numbers per trained variant (SEL)", "",
           "| variant | score nav+vis | kind | k/n | Wilson LB | floor | bar k | A2 | image-only : blind-only | p | A3 | "
           "A4 | all three |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, vd in doc["variants"].items():
        for k, c in (vd.get("g4g") or {}).items():
            a3 = c.get("A3_paired") or {}
            f = c.get("floor") or {}
            L_.append(f"| {v} | {vd.get('selection_score_nav_k_plus_vis_k')} | {k} | {c['k']}/{c['n']} | "
                      f"{c.get('wilson_lb')} | {f.get('policy', '-')} {f.get('k', '-')}/{f.get('n', '-')} | "
                      f"{c.get('bar_k')} | {c.get('A2_lb_above_floor')} | {a3.get('image_only', '-')} : "
                      f"{a3.get('blind_only', '-')} | {a3.get('p_two_sided_exact', '-')} | "
                      f"{c.get('A3_images_better_p_lt_0.05')} | {c.get('A4_rate_le_0.80')} | {c.get('all_three')} |")
    th = [(k, e["thought"]) for k, e in doc["evals"].items() if e.get("thought")]
    if th:
        L_ += ["", "## THOUGHT odometer (short-THOUGHT evals)", "",
               "| eval | calls | format ok | walked exact | mean abs err (m) | max abs err (m) | nav stated ok |",
               "|---|---|---|---|---|---|---|"]
        for k, s in th:
            L_.append(f"| {k} | {s['calls']} | {_k(s['format_ok'])} | {_k(s['walked_exact_0.05'])} | "
                      f"{s['walked_abs_err_mean']} | {s['walked_abs_err_max']} | {_k(s['nav_stated_matches_task'])} |")
    L_ += ["", "---", "Numbers only. A2 = Wilson LB above the SEL floor (nav ODO, vis FWD); A3 = paired exact test, same "
           "adapter with vs without images, image-only > blind-only and two-sided p < 0.05; A4 = kind rate <= 80%."]
    return "\n".join(L_) + "\n"


# ------------------------------------------------------------------------------------------------- outputs
def expected_outputs(variant: str, base: str | None = None, x: Path = X_ROOT, b1_out: Path = B1_OUT) -> list:
    x = Path(x)
    out = [data_path(x, variant), vdir(x, variant) / "data" / "manifest.json"]
    out += [vdir(x, variant) / "adapter" / f for f in ("adapter.pt", "meta.json", "log.jsonl")]
    for (owner, model, s, cm, odo, blind) in job_evals(variant, base, x):
        out.append(vdir(x, owner) / "evals" / eval_name(model, s, cm, odo, blind) / "results.json")
    if variant == "V0":
        out += [x / "floors" / "floors_s2sel.json", x / "b1" / "B1_CHECK.json", Path(b1_out) / "demos_all.jsonl",
                Path(b1_out) / "manifest.json"]
    out.append(x / "analysis" / "ANALYZE.json")
    return out


# ------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "eval":
        return cmd_eval(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup", help="s2dev74 + s2sel (created or verified) and the X dirs")
    p.add_argument("--x", type=Path, default=X_ROOT)
    p = sub.add_parser("data")
    p.add_argument("--variant", choices=TRAINED, required=True)
    p.add_argument("--base", default=None)
    p.add_argument("--x", type=Path, default=X_ROOT)
    p = sub.add_parser("floors")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True, help="sandbox parent (removed)")
    p.add_argument("--tasks", default=None, help="comma-separated stems (subset; tests)")
    p = sub.add_parser("init-check")
    p.add_argument("--init", type=Path, default=INIT)
    p.add_argument("--expect-sha", default=INIT_SHA_PREFIX)
    p.add_argument("--need-loss-weight", action="store_true")
    for nm in ("adapter-ok", "eval-mode"):
        p = sub.add_parser(nm)
        p.add_argument("--variant", choices=TRAINED, required=True)
        p.add_argument("--base", default=None)
        p.add_argument("--x", type=Path, default=X_ROOT)
        p.add_argument("--init", type=Path, default=INIT)
    p = sub.add_parser("b1")
    p.add_argument("--stage", choices=("precheck", "before", "after", "done"), required=True)
    p.add_argument("--x", type=Path, default=X_ROOT)
    p.add_argument("--out-dir", type=Path, default=B1_OUT)
    p = sub.add_parser("analyze")
    p.add_argument("--x", type=Path, default=X_ROOT)
    p = sub.add_parser("expect")
    p.add_argument("--variant", choices=TRAINED, required=True)
    p.add_argument("--base", default=None)
    p.add_argument("--x", type=Path, default=X_ROOT)
    p.add_argument("--b1-out", type=Path, default=B1_OUT)
    p.add_argument("--check", action="store_true", help="exit 1 when anything is missing")
    args = ap.parse_args(argv)
    L.thread_env()
    if args.cmd == "setup":
        print(json.dumps(setup(args.x)))
    elif args.cmd == "data":
        print(json.dumps(build_data(args.variant, args.x, args.base)))
    elif args.cmd == "floors":
        refuse_pilot(args.out)
        stems = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
        rep = floors(args.out, args.logs, args.work, stems=stems)
        print(json.dumps(rep, default=str))
    elif args.cmd == "init-check":
        probs, facts = init_check(args.init, args.expect_sha, args.need_loss_weight)
        print(json.dumps({"ok": not probs, "problems": probs, **facts}))
        return 1 if probs else 0
    elif args.cmd == "adapter-ok":
        eval_mode(args.variant, args.base, args.x)  # the base must be consistent
        rc, msg = adapter_check(args.variant, args.x, args.init)
        print(f"[adapter-ok] {msg}")
        return rc
    elif args.cmd == "eval-mode":
        print(" ".join(eval_mode(args.variant, args.base, args.x)))
    elif args.cmd == "b1":
        bdir = Path(args.x) / "b1"
        refuse_pilot(bdir)
        if args.stage == "precheck":
            probs, facts = b1_precheck()
        elif args.stage == "before":
            sha = sha256_file(FROZEN_DEMOS)
            probs = [] if sha == FROZEN_DEMOS_SHA256 else [f"{FROZEN_DEMOS}: sha256 {sha} != {FROZEN_DEMOS_SHA256}"]
            facts = {"frozen": str(FROZEN_DEMOS), "frozen_sha256": sha, "utc": utc()}
            write_json(bdir / "frozen_before.json", facts)
        elif args.stage == "after":
            probs, facts = b1_after(args.out_dir, read_json(bdir / "frozen_before.json"))
            write_json(bdir / "B1_CHECK.json", {"schema": SCHEMA + ":b1", "exploratory": False, "ok": not probs,
                                                "problems": probs, "out_dir": str(args.out_dir), "stem": B1_STEM,
                                                "decision": "DECISIONS G6", "utc": utc(), **facts})
        else:  # done: a verified rebuild whose demos_all.jsonl is still the checked one
            chk = read_json(bdir / "B1_CHECK.json") or {}
            da = Path(args.out_dir) / "demos_all.jsonl"
            ok = bool(chk.get("ok")) and da.is_file() and sha256_file(da) == chk.get("demos_all_sha256")
            probs, facts = ([] if ok else ["no verified B1 rebuild"]), {"check": str(bdir / "B1_CHECK.json")}
        print(json.dumps({"stage": args.stage, "ok": not probs, "problems": probs, **facts}, default=str))
        return 1 if probs else 0
    elif args.cmd == "analyze":
        doc = analyze(args.x)
        print(f"[variants] wrote {Path(args.x) / 'analysis' / 'ANALYZE.json'}")
        for k, e in doc["evals"].items():
            if e.get("state") == "ok":
                t = e["table"]["SEL"]
                print(f"  {k:60s} SEL nav {_k(t['nav'])} vis {_k(t['vis'])}")
    elif args.cmd == "expect":
        missing = 0
        for pth in expected_outputs(args.variant, args.base, args.x, args.b1_out):
            ok = Path(pth).exists()
            missing += not ok
            print(f"{'ok     ' if ok else 'MISSING'} {pth}")
        print(f"[variants] {args.variant}{' on ' + args.base if args.base else ''}: {missing} missing")
        return 1 if (args.check and missing) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
