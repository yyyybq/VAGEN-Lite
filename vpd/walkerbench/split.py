"""Freeze the WalkerBench S2 split (city-disjoint): s2test 179, s2val 57, s2train 540 (= s2val + train-rest 483).

    python -m vpd.walkerbench.split            # write OUT/walkerbench/split/ (refuses to overwrite a different split)
    python -m vpd.walkerbench.split --check    # verify an existing output: hashes, counts, membership

Port of the read-only survey prototypes (scratchpad vpd_demos/b1_split.py, dsv/wb/a10_split2.py):
- Input: the frozen split_city.json (copied to vpd/walkerbench/data/, sha256 SPLIT_CITY_SHA256): 23 held-out cities,
  179 test and 548 train basenames, picked from the 700 full nav/vis tasks plus the 27 standard nav/vis files whose
  sha256 is not among the full ones, by taking whole cities in md5(city name) order until >= 160 test tasks.
- Each basename resolves to REPO_WB/tasks/full/<name>, or tasks/standard/<name> when that standard file's sha256 is
  not among the full files (no such name also exists in full; asserted).
- city = the cities.json entry nearest (haversine) to the spawn pano (cache.db locations).
- route = (spawn_point, target_pano_ids, visual_path pano ids); nav and vis variants of one route share it.
- The 8 train tasks whose visual_path steps to a pano outside the geofence (not linked, so the expert cannot follow
  it) are excluded.  vis_1103 (test) has the same kind of break and stays in test unmodified; expert_splice.json
  holds the in-fence BFS splice as a side note only.
- S2-val: continue the md5(city) order over the remaining train cities, whole cities, until >= 40 tasks.
Everything is keyed on the file stem, never on task_config['task_id'] (vis files carry their nav twin's id).
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import math
import shutil
import sqlite3
import sys
from pathlib import Path

from vpd import paths
from vpd.walkerbench import sandbox

DATA = Path(__file__).resolve().parent / "data"
SPLIT_CITY = DATA / "split_city.json"
SPLIT_CITY_SHA256 = "41f9635a80777175bc717712829f932a3532f6bfdec63aae5b4d7ddbda982786"
SCHEMA = "vpd.walkerbench.split_s2/1"
SPLIT_DIRS = ("s2test", "s2val", "s2train")

BROKEN_TRAIN = (
    "nav_1089_target_20260125_0339_3", "nav_1090_target_20260125_0339_4", "vis_0846_target_20260127_2002_02",
    "vis_0916_target_20260127_2003_12", "vis_1019_target_20260127_2006_33", "vis_1089_target_20260127_2009_53",
    "vis_1090_target_20260127_2009_04", "vis_1182_target_20260127_2048_13",
)
BROKEN_REASON = "visual_path break outside geofence"
SPLICE_TASK = "vis_1103_target_20260127_2009_30"
VAL_MIN_TASKS = 40
EXPECTED_VAL_CITIES = ("Phoenix, USA", "Cali, Colombia", "Bogota, Colombia", "Bangkok, Thailand",
                       "Washington D.C., USA", "Caracas, Venezuela")
# (nav, vis) counts asserted against the survey (b1_out.txt)
EXPECTED_COUNTS = {"test": (85, 94), "train_raw": (279, 269), "train": (277, 263), "val": (29, 28),
                   "rest": (248, 235)}
EXPECTED_TEST_ROUTES = 140
EXPECTED_REST_ROUTES = 336
EXPECTED_TEST_STANDARD_ONLY = 10


# ---------------------------------------------------------------- helpers ------------------------------------------

def sha256_file(p: Path | str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stem(name: str) -> str:
    return name[:-5] if name.endswith(".json") else name


def kind_of(name: str) -> str:
    k = Path(name).name.split("_")[0]
    if k not in ("nav", "vis"):
        raise ValueError(f"not a nav/vis task: {name}")
    return k


def haversine(a, b) -> float:
    r0 = 6371000
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r0 * math.atan2(math.sqrt(x), math.sqrt(1 - x))


def route_of(task: dict) -> list:
    """[spawn_point, [target_pano_ids], [visual_path pano ids]] (JSON form of the prototype's route tuple)."""
    return [task["spawn_point"], list(task["target_pano_ids"]), [s["pano_id"] for s in task["visual_path"]]]


def route_tuple(route: list) -> tuple:
    """The prototype's hashable/sortable route key: (spawn, (targets...), (path...))."""
    return route[0], tuple(route[1]), tuple(route[2])


def route_id(route: list) -> str:
    return hashlib.sha256(json.dumps(route, separators=(",", ":")).encode()).hexdigest()[:16]


def nav_vis(names) -> tuple[int, int]:
    c = collections.Counter(kind_of(n) for n in names)
    return c["nav"], c["vis"]


class Geo:
    """Read-only WalkerBench geography: pano locations and links (cache.db), cities, geofences."""

    def __init__(self, repo: Path = paths.REPO_WB):
        repo = Path(repo)
        db = sqlite3.connect(f"file:{repo / 'data' / 'cache.db'}?mode=ro&immutable=1", uri=True)
        try:
            self.loc = {p: (a, b) for p, a, b in db.execute("select pano_id, lat, lng from locations")}
            self.links = {p: [(lk.get("panoId") or lk.get("pano_id")) for lk in (json.loads(x) if x else [])]
                          for p, x in db.execute("select pano_id, links from metadata")}
        finally:
            db.close()
        self.cities = json.loads((repo / "data" / "cities.json").read_text())
        self.fences = {k: set(v) for k, v in json.loads((repo / "config" / "geofence_config.json").read_text()).items()}

    def city_of(self, pano: str) -> str:
        ll = self.loc[pano]
        return min(self.cities, key=lambda c: haversine(ll, (c["lat"], c["lng"])))["name"]

    def breaks(self, task: dict) -> list[dict]:
        """visual_path steps i -> i+1 whose target pano is not linked from pano i (stays on the pano are fine)."""
        vp = [s["pano_id"] for s in task["visual_path"]]
        fence = self.fences.get(task.get("geofence"), set())
        out = []
        for i in range(len(vp) - 1):
            a, c = vp[i], vp[i + 1]
            if c == a or c in set(self.links.get(a, [])):
                continue
            out.append({"index": i, "from": a, "to": c, "to_in_geofence": c in fence,
                        "gap_m": round(haversine(self.loc[a], self.loc[c]), 1) if a in self.loc and c in self.loc else None})
        return out

    def bfs(self, a: str, goals: set, fence: set) -> list | None:
        """Shortest in-fence link path a -> any goal (as in b3_proto.py; a itself need not be in the fence)."""
        seen = {a: None}
        q = collections.deque([a])
        while q:
            u = q.popleft()
            if u in goals:
                path = []
                while u is not None:
                    path.append(u)
                    u = seen[u]
                return path[::-1]
            for v in self.links.get(u, []):
                if v not in seen and v in fence and v in self.links:
                    seen[v] = u
                    q.append(v)
        return None


def resolve_sources(names, repo: Path = paths.REPO_WB) -> dict[str, tuple[str, Path]]:
    """basename -> ('full' | 'standard', path).  standard only when that file's sha256 is not among the full files."""
    repo = Path(repo)
    full = {p.name: p for p in (repo / "tasks" / "full").glob("*.json") if p.name.split("_")[0] in ("nav", "vis")}
    std = {p.name: p for p in (repo / "tasks" / "standard").glob("*.json") if p.name.split("_")[0] in ("nav", "vis")}
    full_h = {sha256_file(p) for p in full.values()}
    std_only = {n: p for n, p in std.items() if sha256_file(p) not in full_h}
    clash = sorted(set(std_only) & set(full))
    if clash:
        raise RuntimeError(f"standard-only files whose basename is also in tasks/full (ambiguous): {clash}")
    out = {}
    for n in names:
        if n in std_only:
            out[n] = ("standard", std_only[n])
        elif n in full:
            out[n] = ("full", full[n])
        else:
            raise FileNotFoundError(f"{n} is in neither tasks/full nor tasks/standard of {repo}")
    return out


def load_split_city(path: Path = SPLIT_CITY, expect_sha: str | None = SPLIT_CITY_SHA256) -> tuple[dict, str]:
    h = sha256_file(path)
    if expect_sha and h != expect_sha:
        raise RuntimeError(f"{path} sha256 {h} != frozen {expect_sha}")
    return json.loads(Path(path).read_text()), h


# ---------------------------------------------------------------- build (no writes) --------------------------------

def build(repo: Path = paths.REPO_WB, split_city: Path = SPLIT_CITY, geo: Geo | None = None,
          strict: bool = True) -> dict:
    """Compute the S2 split.  Returns the split_s2.json document (without created_utc) plus '_src' paths."""
    repo = Path(repo)
    sc, sc_sha = load_split_city(split_city)
    geo = geo or Geo(repo)
    test_n, train_n = list(sc["test"]), list(sc["train"])
    if set(test_n) & set(train_n):
        raise RuntimeError("split_city.json: test and train overlap")
    src = resolve_sources(test_n + train_n, repo)
    tasks_raw = {n: json.loads(p.read_text()) for n, (_, p) in src.items()}

    city = {n: geo.city_of(tasks_raw[n]["spawn_point"]) for n in src}
    held = set(sc["held_out_cities"])
    bad_test = sorted(n for n in test_n if city[n] not in held)
    bad_train = sorted(n for n in train_n if city[n] in held)
    if bad_test or bad_train:
        raise RuntimeError(f"city mismatch with split_city.json: test outside held-out {bad_test}, "
                           f"train inside held-out {bad_train}")

    breaks = {n: geo.breaks(tasks_raw[n]) for n in src}
    broken = {n for n, b in breaks.items() if b}
    broken_train = sorted(stem(n) for n in broken if n in set(train_n))
    broken_test = sorted(stem(n) for n in broken if n in set(test_n))
    if broken_train != sorted(BROKEN_TRAIN) or broken_test != [SPLICE_TASK]:
        raise RuntimeError(f"broken-path tasks changed: train {broken_train} (expected {sorted(BROKEN_TRAIN)}), "
                           f"test {broken_test} (expected {[SPLICE_TASK]})")
    inside = sorted(n for n in broken if any(b["to_in_geofence"] for b in breaks[n]))
    if inside:
        raise RuntimeError(f"breaks that stay inside the geofence (reason text would be wrong): {inside}")

    excluded = sorted(n for n in train_n if stem(n) in set(BROKEN_TRAIN))
    pool = [n for n in train_n if n not in set(excluded)]
    tc = collections.Counter(city[n] for n in pool)
    order = sorted(tc, key=lambda c: hashlib.md5(c.encode()).hexdigest())
    val_c, k = [], 0
    for c in order:
        if k >= VAL_MIN_TASKS:
            break
        val_c.append(c)
        k += tc[c]
    val = [n for n in pool if city[n] in set(val_c)]
    rest = [n for n in pool if city[n] not in set(val_c)]

    counts = {"test": nav_vis(test_n), "train_raw": nav_vis(train_n), "train": nav_vis(pool), "val": nav_vis(val),
              "rest": nav_vis(rest)}
    routes = {n: route_of(tasks_raw[n]) for n in src}
    rid = {n: route_id(r) for n, r in routes.items()}
    n_routes = {"test": len({rid[n] for n in test_n}), "val": len({rid[n] for n in val}),
                "rest": len({rid[n] for n in rest})}
    std_only_test = sorted(stem(n) for n in test_n if src[n][0] == "standard")
    problems = []
    for key, want in EXPECTED_COUNTS.items():
        if counts[key] != want:
            problems.append(f"{key} nav/vis {counts[key]} != survey {want}")
    if tuple(val_c) != EXPECTED_VAL_CITIES:
        problems.append(f"S2-val cities {val_c} != {list(EXPECTED_VAL_CITIES)}")
    if n_routes["test"] != EXPECTED_TEST_ROUTES or n_routes["rest"] != EXPECTED_REST_ROUTES:
        problems.append(f"routes test {n_routes['test']} / rest {n_routes['rest']} != "
                        f"{EXPECTED_TEST_ROUTES} / {EXPECTED_REST_ROUTES}")
    if len(std_only_test) != EXPECTED_TEST_STANDARD_ONLY:
        problems.append(f"standard-only test tasks {len(std_only_test)} != {EXPECTED_TEST_STANDARD_ONLY}")
    if problems and strict:
        raise RuntimeError("S2 split differs from the survey: " + "; ".join(problems))

    members = {"test": test_n, "val": val, "rest": rest}
    overlap = {}
    for a, b in (("test", "val"), ("test", "rest"), ("val", "rest")):
        A, B = members[a], members[b]
        overlap[f"{a}|{b}"] = {
            "routes": len({rid[n] for n in A} & {rid[n] for n in B}),
            "geofences": len({tasks_raw[n].get("geofence") for n in A} & {tasks_raw[n].get("geofence") for n in B}),
            "path_panos": len({p for n in A for p in routes[n][2]} & {p for n in B for p in routes[n][2]}),
            "cities": len({city[n] for n in A} & {city[n] for n in B}),
        }
    if any(v["routes"] or v["cities"] for v in overlap.values()):
        raise RuntimeError(f"splits share routes or cities: {overlap}")

    split_of = {**{n: "test" for n in test_n}, **{n: "val" for n in val}, **{n: "rest" for n in rest},
                **{n: "excluded" for n in excluded}}
    tasks = {}
    for n in sorted(src):
        tasks[stem(n)] = {"file": n, "src": src[n][0], "sha256": sha256_file(src[n][1]), "kind": kind_of(n),
                          "city": city[n], "geofence": tasks_raw[n].get("geofence"), "route_id": rid[n],
                          "split": split_of[n], "n_path": len(routes[n][2])}
    splice = expert_splice(tasks_raw[SPLICE_TASK + ".json"], geo, breaks[SPLICE_TASK + ".json"],
                           sha256_file(src[SPLICE_TASK + ".json"][1]))
    doc = {
        "schema": SCHEMA,
        "source": {"split_city": "vpd/walkerbench/data/split_city.json", "split_city_sha256": sc_sha,
                   "split_city_origin": str(paths.SPLIT_CITY_SRC), "walkerbench_repo": str(repo),
                   "walkerbench_commit": sandbox.repo_commit(repo), "code": "vpd/walkerbench/split.py",
                   "code_sha256": sha256_file(__file__)},
        "rules": {
            "resolve": "tasks/full/<name>, or tasks/standard/<name> when its sha256 is not among the full files",
            "city": "cities.json entry nearest (haversine) to the spawn pano",
            "route": "(spawn_point, target_pano_ids, visual_path pano ids)",
            "excluded": f"{BROKEN_REASON}: a visual_path step to a pano that is not linked and lies outside the "
                        "task's geofence (the expert cannot follow it); train only",
            "s2val": f"continue the md5(city) order over the train cities, whole cities, until >= {VAL_MIN_TASKS} tasks",
            "s2train": "s2val + train-rest (540); tier orders draw only from train-rest (483)",
            "key": "file stem (never task_config['task_id'])",
        },
        "held_out_cities": list(sc["held_out_cities"]),
        "val_cities": val_c,
        "train_city_md5_order": order,
        "counts": {k: {"nav": v[0], "vis": v[1], "all": v[0] + v[1]} for k, v in counts.items()},
        "n_routes": n_routes,
        "standard_only_test": std_only_test,
        "overlap": overlap,
        "survey_mismatches": problems,
        "excluded": {stem(n): {"reason": BROKEN_REASON, "breaks": breaks[n]} for n in excluded},
        "splits": {"s2test": sorted(stem(n) for n in test_n), "s2val": sorted(stem(n) for n in val),
                   "s2train": sorted(stem(n) for n in val + rest), "train_rest": sorted(stem(n) for n in rest)},
        "routes": {rid[n]: routes[n] for n in sorted(src)},
        "tasks": tasks,
    }
    doc["_src"] = {stem(n): str(p) for n, (_, p) in src.items()}
    doc["_splice"] = splice
    return doc


def expert_splice(task: dict, geo: Geo, breaks: list[dict], file_sha: str) -> dict:
    """In-fence BFS splice of vis_1103's visual_path after its break (side note; the test file is not modified)."""
    if len(breaks) != 1:
        raise RuntimeError(f"{SPLICE_TASK}: expected one break, got {breaks}")
    br = breaks[0]["index"]
    vp = [s["pano_id"] for s in task["visual_path"]]
    fence = geo.fences[task["geofence"]]
    tail = geo.bfs(vp[br], set(task["target_pano_ids"]), fence)
    if not tail:
        raise RuntimeError(f"{SPLICE_TASK}: no in-fence BFS path from {vp[br]} to a target")
    spliced = vp[:br + 1] + tail[1:]
    return {"task": SPLICE_TASK, "task_sha256": file_sha, "break": breaks[0], "splice_after_index": br,
            "tail_hops": len(tail) - 1, "original_path": vp, "spliced_path": spliced,
            "note": "side note only: the s2test file is unmodified and the expert ceiling by visual_path is 178/179; "
                    "with this splice the RM expert reaches the target (179/179). Checked by baselines.py (RM_SPLICE)."}


# ---------------------------------------------------------------- write / check ------------------------------------

def _selection(doc: dict, split: str) -> dict:
    stems = doc["splits"][split]
    t = doc["tasks"]
    sel = {"split": split, "schema": SCHEMA, "source_split_city_sha256": doc["source"]["split_city_sha256"],
           "walkerbench_commit": doc["source"]["walkerbench_commit"],
           "counts": dict(zip(("nav", "vis"), nav_vis(stems)), all=len(stems)),
           "n_routes": len({t[s]["route_id"] for s in stems}),
           "files": {t[s]["file"]: {"sha256": t[s]["sha256"], "src": t[s]["src"], "city": t[s]["city"]}
                     for s in stems}}
    if split == "s2train":
        sel["members"] = {"s2val": doc["splits"]["s2val"], "train_rest": doc["splits"]["train_rest"]}
    return sel


def _public(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if not k.startswith("_") and k != "created_utc"}


def write(doc: dict, out: Path = paths.WB_SPLIT) -> Path:
    """Write split_s2.json, expert_splice.json and the three split dirs.  Refuses to touch an existing split."""
    out = Path(out)
    if (out / "split_s2.json").exists() or any((out / d).exists() for d in SPLIT_DIRS):
        raise FileExistsError(f"{out} already holds a split; run --check (the split is frozen, not rewritten)")
    out.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_DIRS:
        td = out / split / "tasks"
        td.mkdir(parents=True)
        for s in doc["splits"][split]:
            dst = td / doc["tasks"][s]["file"]
            shutil.copyfile(doc["_src"][s], dst)
            if sha256_file(dst) != doc["tasks"][s]["sha256"]:
                raise RuntimeError(f"copy of {s} changed its hash")
        (out / split / "selection.json").write_text(json.dumps(_selection(doc, split), indent=1) + "\n")
    pub = dict(_public(doc), created_utc=utc_now())
    (out / "split_s2.json").write_text(json.dumps(pub, indent=1, ensure_ascii=False) + "\n")
    (out / "expert_splice.json").write_text(json.dumps(dict(doc["_splice"], created_utc=pub["created_utc"]),
                                                       indent=1) + "\n")
    return out / "split_s2.json"


def load(out: Path = paths.WB_SPLIT) -> dict:
    return json.loads((Path(out) / "split_s2.json").read_text())


def check(out: Path = paths.WB_SPLIT, repo: Path = paths.REPO_WB, rebuild: bool = True) -> list[str]:
    """Problems found in an existing output (empty = OK): file hashes and sets, selection.json, membership."""
    out = Path(out)
    probs = []
    doc = load(out)
    if doc.get("schema") != SCHEMA:
        probs.append(f"schema {doc.get('schema')} != {SCHEMA}")
    if doc["source"]["split_city_sha256"] != SPLIT_CITY_SHA256 or sha256_file(SPLIT_CITY) != SPLIT_CITY_SHA256:
        probs.append("split_city.json sha256 differs from the frozen value")
    want_n = {"s2test": 179, "s2val": 57, "s2train": 540}
    for split in SPLIT_DIRS:
        stems = doc["splits"][split]
        if len(stems) != want_n[split]:
            probs.append(f"{split}: {len(stems)} tasks, expected {want_n[split]}")
        td = out / split / "tasks"
        files = {p.name: p for p in td.glob("*.json")}
        want = {doc["tasks"][s]["file"]: doc["tasks"][s]["sha256"] for s in stems}
        if set(files) != set(want):
            probs.append(f"{split}: files on disk differ from split_s2.json "
                         f"(missing {sorted(set(want) - set(files))[:5]}, extra {sorted(set(files) - set(want))[:5]})")
        for n in sorted(set(files) & set(want)):
            if sha256_file(files[n]) != want[n]:
                probs.append(f"{split}: {n} sha256 differs")
        sel = json.loads((out / split / "selection.json").read_text())
        if sel != _selection(doc, split):
            probs.append(f"{split}: selection.json differs from split_s2.json")
    if len(doc["splits"]["train_rest"]) != 483:
        probs.append(f"train_rest {len(doc['splits']['train_rest'])} != 483")
    sp = json.loads((out / "expert_splice.json").read_text())
    if sp.get("task") != SPLICE_TASK or sp.get("task_sha256") != doc["tasks"][SPLICE_TASK]["sha256"]:
        probs.append("expert_splice.json does not match the s2test copy of vis_1103")
    if rebuild:
        fresh = build(repo)
        if _public(fresh)["source"]["code_sha256"] != doc["source"]["code_sha256"]:
            print(f"note: split.py changed since the split was written (sha256 {doc['source']['code_sha256'][:12]} -> "
                  f"{fresh['source']['code_sha256'][:12]}); membership is compared below")
        a, b = _public(fresh), {k: v for k, v in doc.items() if k != "created_utc"}
        a["source"] = {k: v for k, v in a["source"].items() if k != "code_sha256"}
        b["source"] = {k: v for k, v in b["source"].items() if k != "code_sha256"}
        if a != b:
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            probs.append(f"rebuilding from the WalkerBench checkout gives a different split (keys {diff})")
        sp_fresh = dict(fresh["_splice"])
        if {k: v for k, v in sp.items() if k != "created_utc"} != sp_fresh:
            probs.append("expert_splice.json differs from a rebuild")
    return probs


def summary(doc: dict) -> str:
    c = doc["counts"]
    lines = [f"s2test {c['test']['all']} ({c['test']['nav']} nav / {c['test']['vis']} vis, {doc['n_routes']['test']} "
             f"routes, {len(doc['held_out_cities'])} cities; standard-only {len(doc['standard_only_test'])})",
             f"train {c['train_raw']['all']} -> {c['train']['all']} usable after excluding {len(doc['excluded'])} "
             f"({BROKEN_REASON})",
             f"s2val {c['val']['all']} ({c['val']['nav']}/{c['val']['vis']}, {doc['n_routes']['val']} routes): "
             f"{', '.join(doc['val_cities'])}",
             f"train-rest {c['rest']['all']} ({c['rest']['nav']}/{c['rest']['vis']}, {doc['n_routes']['rest']} routes); "
             f"s2train dir = s2val + train-rest = {len(doc['splits']['s2train'])}",
             f"overlap {json.dumps(doc['overlap'])}"]
    if doc.get("survey_mismatches"):
        lines.append("WARN survey mismatches: " + "; ".join(doc["survey_mismatches"]))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="verify an existing output instead of writing")
    ap.add_argument("--out", type=Path, default=paths.WB_SPLIT)
    ap.add_argument("--repo", type=Path, default=paths.REPO_WB)
    ap.add_argument("--no-rebuild", action="store_true", help="--check: hashes only, skip the rebuild comparison")
    a = ap.parse_args(argv)
    if a.check:
        probs = check(a.out, a.repo, rebuild=not a.no_rebuild)
        for p in probs:
            print("FAIL", p)
        print(("CHECK OK " if not probs else "CHECK FAILED ") + str(a.out / "split_s2.json"))
        return 1 if probs else 0
    doc = build(a.repo)
    p = write(doc, a.out)
    print(summary(doc))
    print(f"wrote {p} (sha256 {sha256_file(p)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
