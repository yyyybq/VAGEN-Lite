"""WP-WBDATA tests: S2 split freeze, nested tier orders, tier materialisation, baselines on s2test.

Login node, 1 core, no GPU, PY_EVAL (WalkerBench env), about 2 min (the baselines run in a subprocess):
    cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
      nice -n 19 $PY_EVAL -m pytest -q -p no:cacheprovider vpd/tests/test_wbdata.py
Scratch goes under eval_external_assets/vpd/scratch/wbdata/pytest_<rand>/ (env VPD_WBDATA_TEST_DIR overrides), never
/tmp, and is removed at the end.  The frozen outputs under OUT/walkerbench/ are only read.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import types
import uuid
from pathlib import Path

import pytest

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vpd import paths  # noqa: E402
from vpd.walkerbench import baselines as B  # noqa: E402
from vpd.walkerbench import split as S  # noqa: E402
from vpd.walkerbench import tiers as T  # noqa: E402

WORKTREE = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def workdir():
    base = Path(os.environ.get("VPD_WBDATA_TEST_DIR", paths.scratch("wbdata")))
    d = base / f"pytest_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d)  # only what this test created


@pytest.fixture(scope="module")
def doc():
    return S.build()


@pytest.fixture(scope="module")
def split_dir(doc, workdir):
    out = workdir / "split"
    S.write(doc, out)
    return out


# ---------------------------------------------------------------- split --------------------------------------------

def test_split_city_frozen():
    assert S.sha256_file(S.SPLIT_CITY) == S.SPLIT_CITY_SHA256
    if paths.SPLIT_CITY_SRC.exists():
        assert S.sha256_file(paths.SPLIT_CITY_SRC) == S.SPLIT_CITY_SHA256


def test_split_city_not_git_ignored():
    """The root .gitignore ignores every data/ dir; vpd/.gitignore re-includes vpd/walkerbench/data so that
    `git add vpd/` commits the frozen split source (audit 2026-11-08).  Read-only git (check-ignore)."""
    if shutil.which("git") is None or not (WORKTREE / ".git").exists():
        pytest.skip("no git checkout")
    rel = S.SPLIT_CITY.resolve().relative_to(WORKTREE.resolve())
    res = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(WORKTREE), "check-ignore", "-v", "--no-index",
                          str(rel)], capture_output=True, text=True, timeout=60)
    assert res.returncode == 1, f"{rel} is git-ignored: {res.stdout.strip()}"  # 1 = not ignored, 0 = ignored


def test_split_counts(doc):
    sp = doc["splits"]
    assert (len(sp["s2test"]), len(sp["s2val"]), len(sp["s2train"]), len(sp["train_rest"])) == (179, 57, 540, 483)
    c = {k: (v["nav"], v["vis"]) for k, v in doc["counts"].items()}
    assert c == S.EXPECTED_COUNTS
    assert doc["val_cities"] == list(S.EXPECTED_VAL_CITIES)
    assert doc["n_routes"] == {"test": 140, "val": 46, "rest": 336}
    assert len(doc["held_out_cities"]) == 23
    assert set(sp["s2train"]) == set(sp["s2val"]) | set(sp["train_rest"])
    assert not doc["survey_mismatches"]
    assert len(doc["standard_only_test"]) == 10
    assert all(doc["tasks"][s]["src"] == "standard" for s in doc["standard_only_test"])


def test_split_excluded(doc):
    assert sorted(doc["excluded"]) == sorted(S.BROKEN_TRAIN)
    for s, e in doc["excluded"].items():
        assert e["reason"] == "visual_path break outside geofence"
        assert e["breaks"] and all(not b["to_in_geofence"] for b in e["breaks"])
        assert doc["tasks"][s]["split"] == "excluded"
        assert s not in set(doc["splits"]["s2train"]) | set(doc["splits"]["s2test"])
    assert S.SPLICE_TASK in doc["splits"]["s2test"]  # broken too, but kept unmodified


def test_split_disjoint(doc):
    t = doc["tasks"]
    groups = {k: doc["splits"][k] for k in ("s2test", "s2val", "train_rest")}
    for a in groups:
        for b in groups:
            if a < b:
                assert not set(groups[a]) & set(groups[b])
                assert not {t[s]["route_id"] for s in groups[a]} & {t[s]["route_id"] for s in groups[b]}, (a, b)
                assert not {t[s]["city"] for s in groups[a]} & {t[s]["city"] for s in groups[b]}, (a, b)
    assert {t[s]["city"] for s in groups["s2test"]} <= set(doc["held_out_cities"])
    assert all(v["routes"] == 0 and v["cities"] == 0 for v in doc["overlap"].values())


def test_split_keyed_on_stem(doc):
    """vis files carry their nav twin's task_id, so task_config['task_id'] is not a key."""
    src = doc["_src"]
    clash = 0
    for s in doc["splits"]["s2test"]:
        tid = json.loads(Path(src[s]).read_text()).get("task_id")
        clash += tid is not None and str(tid) != s
    assert clash > 0
    assert all(s == Path(v["file"]).stem for s, v in doc["tasks"].items())


def test_split_write_and_check(doc, split_dir):
    for d, n in (("s2test", 179), ("s2val", 57), ("s2train", 540)):
        files = list((split_dir / d / "tasks").glob("*.json"))
        assert len(files) == n
        sel = json.loads((split_dir / d / "selection.json").read_text())
        assert sel["source_split_city_sha256"] == S.SPLIT_CITY_SHA256
        assert sel["walkerbench_commit"].startswith(paths.WB_COMMIT)
        assert sel["counts"]["all"] == n and len(sel["files"]) == n
    assert S.check(split_dir, rebuild=False) == []
    sp = json.loads((split_dir / "expert_splice.json").read_text())
    assert sp["task"] == S.SPLICE_TASK and sp["splice_after_index"] == 15 and sp["tail_hops"] == 2
    assert sp["spliced_path"][:16] == sp["original_path"][:16]
    with pytest.raises(FileExistsError):
        S.write(doc, split_dir)


def test_split_check_detects_tamper(doc, workdir):
    out = workdir / "split_tamper"
    S.write(doc, out)
    victim = next((out / "s2val" / "tasks").glob("*.json"))
    victim.write_text(victim.read_text() + " ")
    (out / "s2test" / "tasks" / "extra_nav.json").write_text("{}")
    probs = S.check(out, rebuild=False)
    assert any("s2val" in p and "sha256" in p for p in probs)
    assert any("s2test" in p and "extra" in p for p in probs)
    shutil.rmtree(out)


def test_split_check_rebuild(split_dir):
    assert S.check(split_dir, rebuild=True) == []


def test_frozen_split_output():
    """The frozen output, if written (it is read only here)."""
    if not (paths.WB_SPLIT / "split_s2.json").exists():
        pytest.skip("OUT/walkerbench/split not written yet")
    assert S.check(paths.WB_SPLIT, rebuild=False) == []


# ---------------------------------------------------------------- tiers --------------------------------------------

@pytest.fixture(scope="module")
def rest(doc):
    return T.rest_tasks(doc)


def test_tier_order_seed1001(rest):
    order, distinct = T.make_order(rest, 1001)
    assert T.check_order(order, rest, distinct) == []
    assert len(order) == 483 and sorted(order) == sorted(rest)
    for n, (nav, vis) in T.EXPECTED_SPLIT.items():
        st = T.tier_stats(order, rest, n)
        assert (st["nav"], st["vis"], st["routes"]) == (nav, vis, n)
    # strict alternation inside the distinct-route prefix: nav, vis, nav, ...
    assert [rest[s][0] for s in order[:distinct]] == [("nav", "vis")[i % 2] for i in range(distinct)]
    assert len({rest[s][1] for s in order[:distinct]}) == distinct == 336
    # nesting: every tier is a prefix of the full order, so of every larger tier
    assert T.make_order(rest, 1001) == (order, distinct)


def test_tier_order_survey_seeds(rest):
    """Survey (b1_out.txt): the alternating distinct-route prefix is 336, 336, 335 for seeds 1, 2, 3."""
    got = {s: T.make_order(rest, s) for s in (1, 2, 3)}
    assert {s: d for s, (_, d) in got.items()} == {1: 336, 2: 336, 3: 335}
    assert len({tuple(o[:25]) for o, _ in got.values()}) == 3
    pilot = T.make_order(rest, 1001)[0]
    assert all(o[:50] != pilot[:50] for o, _ in got.values())


def test_tier_files(doc, split_dir, workdir):
    out = workdir / "tiers"
    p = T.cmd_order(1001, split_dir, out)
    od = json.loads(p.read_text())
    assert od["seed"] == 1001 and od["role"] == "pilot" and od["n"] == 483
    assert od["tiers"]["200"] == {"n": 200, "nav": 100, "vis": 100, "routes": 200}
    assert od["split_s2_sha256"] == S.sha256_file(split_dir / "split_s2.json")
    assert T.cmd_order(1001, split_dir, out) == p  # identical rewrite is a no-op
    od_bad = dict(od, order=od["order"][::-1])
    p.write_text(json.dumps(od_bad))
    with pytest.raises(FileExistsError):
        T.cmd_order(1001, split_dir, out)
    p.write_text(json.dumps(od, indent=1) + "\n")


def _fake_demos(path: Path, ids, drop=(), dup=None):
    ids = [i for i in ids if i not in set(drop)]
    random.Random(0).shuffle(ids)
    with open(path, "w") as f:
        for i in ids:
            f.write(json.dumps({"id": i, "task_type": i.split("_")[0]}) + "\n")
        if dup:
            f.write(json.dumps({"id": dup}) + "\n")


def test_tier_materialize_fake(doc, split_dir, workdir):
    out = workdir / "tiers_mat"
    T.cmd_order(1001, split_dir, out)
    demos = workdir / "demos_all_fake.jsonl"
    _fake_demos(demos, doc["splits"]["s2train"])  # 540 ids incl. s2val, shuffled
    sd = T.cmd_materialize(1001, "25,50,100,200,all", demos, out, split_dir=split_dir)
    order = json.loads((out / "order_seed1001.json").read_text())["order"]
    prev = ""
    for name, n in (("25", 25), ("50", 50), ("100", 100), ("200", 200), ("all", 483)):
        text = (sd / f"demos_{name}.jsonl").read_text()
        ids = [json.loads(x)["id"] for x in text.splitlines()]
        assert ids == order[:n]
        assert text.startswith(prev)
        prev = text
    man = json.loads((sd / "manifest.json").read_text())
    assert man["tiers"]["50"]["nav"] == 25 and man["tiers"]["all"]["n"] == 483
    assert not set(doc["splits"]["s2val"]) & set(order)
    # missing demo: hard error
    demos2 = workdir / "demos_missing.jsonl"
    _fake_demos(demos2, doc["splits"]["s2train"], drop=[order[300]])
    with pytest.raises(RuntimeError, match="missing demo"):
        T.cmd_materialize(1001, "25,all", demos2, workdir / "tiers_mat", split_dir=split_dir)
    demos3 = workdir / "demos_dup.jsonl"
    _fake_demos(demos3, doc["splits"]["s2train"], dup=order[0])
    with pytest.raises(ValueError, match="duplicate"):
        T.cmd_materialize(1001, "25", demos3, out, split_dir=split_dir)
    with pytest.raises(ValueError):
        T.parse_tiers("50,25", 483)


def test_frozen_tier_orders():
    for seed in (1001, 1, 2, 3):
        p = paths.WB_TIERS / f"order_seed{seed}.json"
        if not p.exists():
            pytest.skip(f"{p} not written yet")
        od = json.loads(p.read_text())
        assert od["split_s2_sha256"] == S.sha256_file(paths.WB_SPLIT / "split_s2.json")
        rest = T.rest_tasks(S.load(paths.WB_SPLIT))
        assert od["order"] == T.make_order(rest, seed)[0]


# ---------------------------------------------------------------- baselines: policy units ---------------------------

def _session(path, cur, heading, opt=0.0):
    return types.SimpleNamespace(task_config={"visual_path": [{"pano_id": p} for p in path],
                                              "ground_truth": {"optimal_distance_meters": opt}},
                                 state=types.SimpleNamespace(pano_id=cur, heading=heading), task_id="x")


def test_rm_expert_rules():
    path = ["a", "b", "c"]
    moves = [{"id": 1, "pano_id": "z", "heading": 0.0}, {"id": 2, "pano_id": "b", "heading": 100.04}]
    rot = set()
    s = _session(path, "a", 10.0)
    assert B.rm_expert(s, lambda: moves, rot) == {"type": "rotation", "heading": 100.0, "pitch": 0, "reason": ""}
    assert B.rm_expert(s, lambda: moves, rot) == {"type": "move", "move_id": 2, "reason": ""}  # rotated once
    s2 = _session(path, "a", 80.0)  # 20 deg off: move at once
    assert B.rm_expert(s2, lambda: moves, set())["type"] == "move"
    assert B.rm_expert(_session(path, "c", 0.0), lambda: moves, set()) == B.stop("")
    assert B.rm_expert(_session(path, "q", 0.0), lambda: moves, set()) == B.stop("")  # off path
    assert B.rm_expert(_session(["a", "y"], "a", 0.0), lambda: moves, set()) == B.stop("path break")
    # revisit: the last occurrence of the current pano decides the next target
    s3 = _session(["a", "b", "a", "c"], "a", 0.0)
    m3 = [{"id": 5, "pano_id": "c", "heading": 10.0}]
    assert B.rm_expert(s3, lambda: m3, set()) == {"type": "move", "move_id": 5, "reason": ""}
    # override path (RM_SPLICE)
    assert B.rm_expert(_session(["a", "y"], "a", 100.0), lambda: moves, set(), ["a", "b"])["type"] == "move"


def test_blind_policies():
    moves = [{"id": 3, "pano_id": "f", "heading": 0.0, "distance": 10.0}, {"id": 4, "pano_id": "g", "heading": 180.0}]
    s = _session(["a"], "a", 33.0, opt=15.0)
    assert B.blind_action("STOP0", s, moves, {"walked": 0.0}) == B.stop("x")
    st = {"walked": 0.0}
    assert B.blind_action("FWD", s, moves, st)["move_id"] == 3 and st["walked"] == 10.0
    st = {"walked": 0.0}
    assert B.blind_action("ODO", s, moves, st)["type"] == "move"
    assert B.blind_action("ODO", s, moves, st)["type"] == "move"  # 10 < 15
    assert B.blind_action("ODO", s, moves, st) == B.stop("x")  # 20 >= 15
    assert B.blind_action("FWD", s, [], {"walked": 0.0}) == {"type": "rotation", "heading": 33.0, "pitch": 0,
                                                              "reason": "x"}
    assert B.blind_action("ODO", s, [], {"walked": 0.0}) == B.stop("x")


def test_wilson():
    lo, hi = B.wilson(27, 179)
    assert 0.105 < lo < 0.106 and 0.210 < hi < 0.211
    assert B.wilson(0, 10)[0] == 0.0 and B.wilson(10, 10)[1] == pytest.approx(1.0)
    assert B.wilson(5, 0) == (0.0, 1.0)


# ---------------------------------------------------------------- baselines: WalkerBench loop -----------------------

def _run_baselines(split_dir: Path, out: Path, work: Path, *extra) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(WORKTREE), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    cmd = [sys.executable, "-m", "vpd.walkerbench.baselines", "--split-dir", str(split_dir), "--out", str(out),
           "--work", str(work), *extra]
    return subprocess.run(cmd, cwd=WORKTREE, env=env, capture_output=True, text=True, timeout=900)


@pytest.fixture(scope="module")
def baselines_doc(split_dir, workdir):
    out = workdir / "baselines"
    r = _run_baselines(split_dir, out, workdir / "bwork", "--strict")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert not list((workdir / "bwork").glob("baselines_*")), "sandbox not removed"
    return json.loads((out / "baselines_s2test.json").read_text()), json.loads((out / "hard_subsets.json").read_text())


def test_baselines_match_survey(baselines_doc):
    b, hs = baselines_doc
    assert b["warnings"] == []
    pol = b["policies"]
    for m, want in B.EXPECTED.items():
        sm = pol[m]["summary"]
        for key, k in want.items():
            assert (sm[key]["k"], sm[key]["n"]) == (k, B.EXPECTED_N[key]), (m, key)
        assert sm["tasks_without_log"] == 0
        assert len(pol[m]["per_task"]) == 179
    assert sorted(t for t, r in pol["RM"]["per_task"].items() if not r["success"]) == [S.SPLICE_TASK]
    assert pol["RM_SPLICE"]["per_task"][S.SPLICE_TASK]["success"] == 1
    assert b["fwd_step_cap_hits"] == 179
    f = b["floors"]
    assert (f["nav_ODO"]["k"], f["nav_ODO"]["n"]) == (24, 85)
    assert (f["vis_FWD"]["k"], f["vis_FWD"]["n"]) == (3, 94)
    assert (f["pooled_F"]["k"], f["pooled_F"]["n"]) == (27, 179)
    assert hs["counts"] == {"nav_odo_fail": 61, "vis_fwd_fail": 91}
    assert all(t.startswith("nav") for t in hs["nav_odo_fail"]) and all(t.startswith("vis") for t in hs["vis_fwd_fail"])
    assert not set(hs["nav_odo_fail"]) & {t for t, r in pol["ODO"]["per_task"].items() if r["success"]}
    assert "stubbed" in b["rendering"]


def test_baselines_subset_guard(split_dir, workdir):
    r = _run_baselines(split_dir, paths.WB_BASELINES, workdir / "bwork2", "--tasks", S.SPLICE_TASK)
    assert r.returncode != 0 and "subset" in (r.stderr + r.stdout)


def test_frozen_baselines():
    p = paths.WB_BASELINES / "baselines_s2test.json"
    if not p.exists():
        pytest.skip("OUT/walkerbench/baselines not written yet")
    b = json.loads(p.read_text())
    assert b["warnings"] == [] and b["split_s2_sha256"] == S.sha256_file(paths.WB_SPLIT / "split_s2.json")
    assert (b["floors"]["pooled_F"]["k"], b["floors"]["pooled_F"]["n"]) == (27, 179)
    hs = json.loads((paths.WB_BASELINES / "hard_subsets.json").read_text())
    assert hs["baselines_sha256"] == S.sha256_file(p)
