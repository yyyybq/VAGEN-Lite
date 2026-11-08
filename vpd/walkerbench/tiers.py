"""Nested, stratified demo tiers over WalkerBench train-rest (483 tasks, 336 routes).

    python -m vpd.walkerbench.tiers order --seed 1001          # -> OUT/walkerbench/tiers/order_seed1001.json
    python -m vpd.walkerbench.tiers materialize --seed 1001 --tiers 25,50,100,200,all \
        --demos OUT/walkerbench/demos/s2train_640x400/demos_all.jsonl   # -> tiers/seed1001/demos_<N>.jsonl

Order rule (port of scratchpad vpd_demos/b1_split.py:order_for, frozen in the prereg):
  keys = sorted(route tuples of train-rest); rnd = random.Random(seed); rnd.shuffle(keys)
  alternate nav, vis, nav, ...: each slot takes the first unused route (in shuffled order) that has a variant of that
  kind, and draws the variant with rnd.choice over that route's variants of the kind (sorted by stem); stop at the
  first slot no unused route can fill.  Then the remaining variants, sorted by stem and shuffled with the same rnd.
Tier N = the first N entries, so tiers are nested; N <= 200 uses distinct routes and an exact nav/vis split.
Seeds: 1001 = pilot, 1/2/3 = confirmatory.  Demos are matched by record id == task file stem.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

from vpd import paths
from vpd.walkerbench import split as S

SCHEMA = "vpd.walkerbench.tier_order/1"
TIERS = (25, 50, 100, 200)
EXPECTED_SPLIT = {25: (13, 12), 50: (25, 25), 100: (50, 50), 200: (100, 100)}
N_REST = 483
DISTINCT_PREFIX = (335, 336)  # survey: seeds 1, 2 -> 336; seed 3 -> 335
DEFAULT_DEMOS = paths.WB_DEMOS / "s2train_640x400" / "demos_all.jsonl"


def rest_tasks(split_doc: dict) -> dict[str, tuple[str, tuple]]:
    """stem -> (kind, route tuple) for train-rest."""
    t, routes = split_doc["tasks"], split_doc["routes"]
    out = {s: (t[s]["kind"], S.route_tuple(routes[t[s]["route_id"]])) for s in split_doc["splits"]["train_rest"]}
    if len(out) != N_REST:
        raise RuntimeError(f"train-rest has {len(out)} tasks, expected {N_REST}")
    return out


def make_order(rest: dict[str, tuple[str, tuple]], seed: int) -> tuple[list[str], int]:
    """(full order of all train-rest stems, length of the alternating distinct-route prefix)."""
    names = sorted(rest)
    # the prototype sorted basenames ('<stem>.json'); stems sort the same way, asserted here
    if names != [n[:-5] for n in sorted(n + ".json" for n in names)]:
        raise RuntimeError("stem order differs from basename order; the prototype order would not be reproduced")
    by_route = collections.defaultdict(list)
    for n in names:
        by_route[rest[n][1]].append(n)
    keys = sorted(by_route)
    rnd = random.Random(seed)
    rnd.shuffle(keys)
    used, seq, want, i = set(), [], ("nav", "vis"), 0
    while True:
        k = want[i % 2]
        pick = None
        for r in keys:
            if r in used:
                continue
            c = [n for n in sorted(by_route[r]) if rest[n][0] == k]
            if c:
                pick = (r, rnd.choice(c))
                break
        if pick is None:
            break
        used.add(pick[0])
        seq.append(pick[1])
        i += 1
    distinct = len(seq)
    taken = set(seq)
    left = [n for n in names if n not in taken]
    rnd.shuffle(left)
    return seq + left, distinct


def tier_stats(order: list[str], rest: dict, n: int) -> dict:
    sub = order[:n]
    c = collections.Counter(rest[s][0] for s in sub)
    return {"n": len(sub), "nav": c["nav"], "vis": c["vis"], "routes": len({rest[s][1] for s in sub})}


def check_order(order: list[str], rest: dict, distinct: int) -> list[str]:
    probs = []
    if sorted(order) != sorted(rest) or len(order) != len(set(order)):
        probs.append("order is not a permutation of train-rest")
    for n, (nav, vis) in EXPECTED_SPLIT.items():
        st = tier_stats(order, rest, n)
        if (st["nav"], st["vis"]) != (nav, vis):
            probs.append(f"tier {n}: nav/vis {st['nav']}/{st['vis']} != {nav}/{vis}")
        if st["routes"] != n:
            probs.append(f"tier {n}: {st['routes']} distinct routes, expected {n}")
    if not DISTINCT_PREFIX[0] <= distinct <= DISTINCT_PREFIX[1]:
        probs.append(f"distinct-route prefix {distinct} outside {DISTINCT_PREFIX}")
    return probs


def order_doc(split_doc: dict, split_sha: str, seed: int) -> dict:
    rest = rest_tasks(split_doc)
    order, distinct = make_order(rest, seed)
    probs = check_order(order, rest, distinct)
    if probs:
        raise RuntimeError(f"seed {seed}: " + "; ".join(probs))
    t = split_doc["tasks"]
    return {"schema": SCHEMA, "seed": seed, "role": "pilot" if seed == 1001 else "confirmatory",
            "split_s2_sha256": split_sha, "code_sha256": S.sha256_file(__file__),
            "rule": __doc__.split("Order rule")[1].split("Tier N")[0].strip(),
            "n": len(order), "distinct_route_prefix": distinct,
            "tiers": {str(n): tier_stats(order, rest, n) for n in TIERS} | {"all": tier_stats(order, rest, len(order))},
            "order": order, "kinds": [rest[s][0] for s in order], "route_ids": [t[s]["route_id"] for s in order]}


def _write_if_new(path: Path, text: str, force: bool) -> str:
    """'wrote' | 'same' ; refuses to replace different content unless force (files a running job reads stay put)."""
    if path.exists():
        if path.read_text() == text:
            return "same"
        if not force:
            raise FileExistsError(f"{path} exists with different content (use --force only if no job reads it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return "wrote"


def cmd_order(seed: int, split_dir: Path, out: Path, force: bool = False) -> Path:
    split_path = Path(split_dir) / "split_s2.json"
    doc = order_doc(json.loads(split_path.read_text()), S.sha256_file(split_path), seed)
    p = Path(out) / f"order_seed{seed}.json"
    st = _write_if_new(p, json.dumps(doc, indent=1) + "\n", force)
    tiers = " ".join(f"{k}={v['nav']}/{v['vis']}" for k, v in doc["tiers"].items())
    print(f"seed {seed}: {st} {p}; distinct-route prefix {doc['distinct_route_prefix']}; tiers nav/vis {tiers}")
    return p


def parse_tiers(text: str, n_all: int) -> list[tuple[str, int]]:
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok == "all":
            out.append(("all", n_all))
        else:
            n = int(tok)
            if not 0 < n <= n_all:
                raise ValueError(f"tier {n} outside 1..{n_all}")
            out.append((str(n), n))
    if [n for _, n in out] != sorted(n for _, n in out):
        raise ValueError("tiers must be increasing")
    return out


def read_demos(path: Path) -> dict[str, str]:
    """id -> the record's JSON line (verbatim, so tier files are byte prefixes of each other)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            rid = json.loads(line).get("id")
            if not rid:
                raise ValueError(f"{path}:{i}: record without id")
            if rid in out:
                raise ValueError(f"{path}:{i}: duplicate demo id {rid}")
            out[rid] = line
    return out


def cmd_materialize(seed: int, tiers: str, demos: Path, out: Path, force: bool = False,
                    split_dir: Path | None = paths.WB_SPLIT) -> Path:
    op = Path(out) / f"order_seed{seed}.json"
    if not op.exists():
        raise FileNotFoundError(f"{op} missing: run `tiers order --seed {seed}` first")
    od = json.loads(op.read_text())
    sp = Path(split_dir) / "split_s2.json" if split_dir else None
    if sp is not None and sp.exists() and S.sha256_file(sp) != od["split_s2_sha256"]:
        raise RuntimeError(f"{op} was made from another split_s2.json than {sp}")
    order, kinds = od["order"], od["kinds"]
    recs = read_demos(Path(demos))
    missing = [s for s in order if s not in recs]
    if missing:
        raise RuntimeError(f"{len(missing)} train-rest tasks have no demo in {demos} (first {missing[:5]}); "
                           "a missing demo is a hard error")
    for s, k in zip(order, kinds):
        tt = json.loads(recs[s]).get("task_type")
        if tt is not None and tt != k:
            raise RuntimeError(f"demo {s}: task_type {tt} != {k}")
    sd = Path(out) / f"seed{seed}"
    man = {"schema": "vpd.walkerbench.tiers/1", "seed": seed, "order": str(op), "order_sha256": S.sha256_file(op),
           "demos": str(demos), "demos_sha256": S.sha256_file(demos), "n_demo_records": len(recs), "tiers": {}}
    prev = None
    for name, n in parse_tiers(tiers, len(order)):
        text = "".join(recs[s] + "\n" for s in order[:n])
        if prev is not None and not text.startswith(prev):
            raise RuntimeError(f"tier {name} is not an extension of the previous tier")
        prev = text
        p = sd / f"demos_{name}.jsonl"
        st = _write_if_new(p, text, force)
        c = collections.Counter(kinds[:n])
        man["tiers"][name] = {"file": p.name, "n": n, "nav": c["nav"], "vis": c["vis"],
                              "routes": len(set(od["route_ids"][:n])),
                              "sha256": hashlib.sha256(text.encode()).hexdigest(), "ids_head": order[:3]}
        print(f"seed {seed} tier {name}: {st} {p} ({n} demos, {c['nav']} nav / {c['vis']} vis)")
    _write_if_new(sd / "manifest.json", json.dumps(man, indent=1) + "\n", True)
    return sd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("order")
    o.add_argument("--seed", type=int, required=True, action="append", help="repeatable")
    m = sub.add_parser("materialize")
    m.add_argument("--seed", type=int, required=True)
    m.add_argument("--tiers", default="25,50,100,200,all")
    m.add_argument("--demos", type=Path, default=DEFAULT_DEMOS)
    for p in (o, m):
        p.add_argument("--split-dir", type=Path, default=paths.WB_SPLIT)
        p.add_argument("--out", type=Path, default=paths.WB_TIERS)
        p.add_argument("--force", action="store_true", help="replace files with different content")
    a = ap.parse_args(argv)
    if a.cmd == "order":
        for s in a.seed:
            cmd_order(s, a.split_dir, a.out, a.force)
    else:
        cmd_materialize(a.seed, a.tiers, a.demos, a.out, a.force, a.split_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
