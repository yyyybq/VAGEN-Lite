"""EXPLORATORY round 5, experiment B (forced opening actions): the inputs and checks of vpd/jobs/tvr_intervene.sh.
Stdlib only; reads finished eval dirs, writes only the files it is told to.

    python -B -m vpd.tvr.intervene sets --a-dir A --b-dir B --out-dir D [--a-name only_a] [--b-name only_b]
    python -B -m vpd.tvr.intervene pick --sets F1[,F2...] --n-each N --out FILE
    python -B -m vpd.tvr.intervene donors --donor-dir DIR --donor LABEL --tasks FILE --out FILE
    python -B -m vpd.tvr.intervene check --set NAME --tasks FILE --sets-manifest FILE --pilot-root DIR --start S
          --forced FILE --donor LABEL --source DIR --out FILE
    python -B -m vpd.tvr.intervene adapter-match --adapter-dir DIR --eval-dir DIR
    python -B -m vpd.tvr.intervene runstate --dir D --k K --tasks FILE --forced FILE [--harness-r5 SHA]
    python -B -m vpd.tvr.intervene expect --out-root D --prefix P --start S --tier T --ks K[,K...] --tasks FILE
          --forced FILE [--harness-r5 SHA] --out FILE
Outputs of sets / pick / donors are deterministic (no time stamps).  An existing output is refused; --overwrite
rewrites it only with the same content; --replace changes it.

sets    A, B = two eval dirs (valid results.json, per_task.jsonl with task_id / success) of the SAME task set (else
        refused): D/<b-name>.json = the sorted task ids B solved and A failed, D/<a-name>.json = A solved and B failed,
        D/sets_manifest.json (input paths + sha256, counts).  The pilot (A = runs/pilot/Q3S/evals/pilot__Q3S__ALL__c,
        B = the Q3B dir): only_b = T101 (101 tasks), only_a = T55 (55).
pick    the first N ids of each set file (file order), duplicates dropped, sorted -> FILE (the smoke subset).
donors  the vpd.tvr.forced/1 file of run_tvr_forced --forced: {"schema", "donor": LABEL, "source": DIR, "prefix":
        {id: DIR/tasks/<id>/trajectory.json "actions"}, "source_sha256": {id: sha256 of that trajectory.json},
        "tasks_file", "tasks_sha256", "source_protocol_sha256"}.  DIR must be an arm-c run of start LABEL; every
        trajectory must carry that run's protocol_sha256.
check   the in-job guard of tvr_intervene.sh (exit 0 ok, 2 problems; FILE records the inputs' sha256): the set file
        is the manifest's entry NAME (sha256; T101 = 101 tasks B solved, T55 = 55 tasks A solved) or, NAME smoke, 8
        tasks of T101 + T55 (both files as in the manifest); the manifest's A / B dirs are the pilot's Q3S / Q3B
        ALL c dirs under --pilot-root (realpath); the forced file was built from this set file, its donor is the
        other pilot start, its source that start's pilot ALL c dir, and for every task its actions are the donor
        trajectory's "actions" and the trajectory is unchanged (source_sha256).
adapter-match  DIR/adapter.pt's sha256 == the eval dir's results.json adapter.adapter_sha256 (and its protocol's).
runstate  exit 0: D has no run yet, or its protocol "forced" / "task_subset" are those of these inputs (k, the forced
        file sha / donor, the harness_r5 sha when given, the task file sha / n); 3: a run of other inputs or code.
expect  the job's expect list: per K the run dir <prefix>__<S>__<T>__c__f<K> has a valid results.json of every task,
        protocol "forced" of these inputs (+ module_sha256 / harness_r5_sha256, the results summary the same) and
        "task_subset" {sha256, n}, official_check.json equal, and every trajectory forced exactly
        forced_prefix(donor actions, K) as its first actions.  Exit 0 ok, 1 not.
Forcing rule (run_tvr_forced.forced_prefix, the same code here): the first K donor actions after dropping every
Stop and InvalidAction; all of them when fewer remain.

Inputs of the job (login node, as the user; OUT = /mnt/umm/users/yinbaiqiao/eval_external_assets/vpd):
    I=OUT/tvr/runs/explore/intervene/inputs; P=OUT/tvr/runs/pilot
    intervene sets --a-dir $P/Q3S/evals/pilot__Q3S__ALL__c --b-dir $P/Q3B/evals/pilot__Q3B__ALL__c --out-dir $I \
        --a-name T55 --b-name T101
    intervene pick --sets $I/T101.json,$I/T55.json --n-each 4 --out $I/smoke.json
    for SET in T101 T55 smoke: intervene donors --donor-dir $P/Q3B/evals/pilot__Q3B__ALL__c --donor Q3B \
        --tasks $I/$SET.json --out $I/forced_Q3S_$SET.json  (and Q3S -> forced_Q3B_$SET.json)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

FORCED_SCHEMA = "vpd.tvr.forced/1"  # == run_tvr_forced.FORCED_SCHEMA
SETS_SCHEMA = "vpd.tvr.intervene_sets/1"
CHECK_SCHEMA = "vpd.tvr.intervene_check/1"
EXPECT_SCHEMA = "vpd.tvr.intervene_expect/1"
ACTIONS = ("MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateRight", "RotateLeft", "LookUp", "LookDown",
           "Stop")  # == pilot_cfg.ACTIONS
FORCED_SKIP = ("Stop", "InvalidAction")  # == run_tvr_forced.FORCED_SKIP
VOCAB = frozenset(ACTIONS + FORCED_SKIP)
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
PILOT_A, PILOT_B = "Q3S", "Q3B"  # sets: A = the Q3S pilot ALL c dir, B = the Q3B one
EXPECTED_SETS = {"T101": ("B", 101), "T55": ("A", 55)}  # set -> (the side that solved it, its size)
SMOKE, SMOKE_N = "smoke", 8  # the smoke subset: 8 tasks of T101 + T55
EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_STALE = 0, 1, 2, 3


class Refused(Exception):
    pass


def sha256_file(p) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def canon_sha256(obj) -> str:
    """run_tvr's protocol_sha256: sha256 of json.dumps(sort_keys, compact separators)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(p) -> Any:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def need_json(p, what: str) -> Any:
    doc = read_json(p)
    if doc is None:
        raise Refused(f"{what} {p}: missing or not JSON")
    return doc


def json_text(obj) -> str:
    return json.dumps(obj, indent=1) + "\n"


def write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".part{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def write_outputs(outs: Dict[Path, str], overwrite: bool, replace: bool) -> None:
    """Write every output, or none: an existing file is refused, kept with --overwrite only if the content is the
    same, changed only with --replace."""
    for p, text in outs.items():
        if not p.exists():
            continue
        same = p.read_text(encoding="utf-8") == text
        if not (replace or (overwrite and same)):
            raise Refused(f"{p} exists" + (" (pass --overwrite to rewrite it)" if same else
                                           " with other content (pass --replace to change it)"))
    for p, text in outs.items():
        write_text(p, text)


def apath(p) -> str:
    """The absolute path as given (no symlink resolution): the donor file's "source" and the job's --source."""
    return os.path.abspath(str(p))


def same_dir(a, b) -> bool:
    return a is not None and b is not None and os.path.realpath(str(a)) == os.path.realpath(str(b))


def pilot_c_dir(root, start: str) -> Path:
    return Path(root) / start / "evals" / f"pilot__{start}__ALL__c"


def forced_prefix(actions: Sequence[str], k: int) -> List[str]:
    """The first K donor actions after dropping every Stop and InvalidAction (all of them when fewer remain)."""
    return [x for x in actions if x not in FORCED_SKIP][:max(0, int(k))]


def task_list(path) -> List[str]:
    """A task file: a non-empty JSON list of distinct task_id strings."""
    ids = need_json(path, "task file")
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
        raise Refused(f"task file {path}: not a non-empty JSON list of task_id strings")
    if len(set(ids)) != len(ids):
        raise Refused(f"task file {path}: duplicated ids")
    return ids


# --- sets / pick --------------------------------------------------------------------------------------------------
def eval_dir_facts(d: Path) -> Dict[str, Any]:
    """{task_id: success} of a finished eval dir (valid results.json, per_task.jsonl) and its provenance."""
    d = Path(d)
    res = need_json(d / "results.json", "results")
    if res.get("valid") is not True:
        raise Refused(f"{d}/results.json is not a valid result")
    p = d / "per_task.jsonl"
    try:
        lines = [x for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    except OSError as e:
        raise Refused(f"{p}: {e!r}")
    succ: Dict[str, int] = {}
    for x in lines:
        try:
            r = json.loads(x)
        except ValueError:
            raise Refused(f"{p}: a line is not JSON ({x[:120]!r})")
        tid, s = r.get("task_id"), r.get("success")
        if not isinstance(tid, str) or s not in (0, 1) or tid in succ:
            raise Refused(f"{p}: bad or duplicated row {x[:120]!r}")
        succ[tid] = int(s)
    if res.get("n_tasks") != len(succ):
        raise Refused(f"{d}: results n_tasks {res.get('n_tasks')} != {len(succ)} per_task rows")
    return {"succ": succ, "dir": apath(d), "per_task_sha256": sha256_file(p),
            "results_sha256": sha256_file(d / "results.json"), "protocol_sha256": sha256_file(d / "protocol.json"),
            "n_tasks": len(succ), "n_success": sum(succ.values())}


def cmd_sets(a) -> int:
    for n in (a.a_name, a.b_name):
        if not NAME_RE.match(n) or n == "sets_manifest":
            raise Refused(f"bad set name {n!r}")
    if a.a_name == a.b_name:
        raise Refused("--a-name and --b-name must differ")
    fa, fb = eval_dir_facts(Path(a.a_dir)), eval_dir_facts(Path(a.b_dir))
    sa, sb = fa.pop("succ"), fb.pop("succ")
    if set(sa) != set(sb):
        raise Refused(f"the two dirs hold different task sets ({len(set(sa) - set(sb))} only in A, "
                      f"{len(set(sb) - set(sa))} only in B)")
    only_a = sorted(t for t in sa if sa[t] == 1 and sb[t] == 0)
    only_b = sorted(t for t in sa if sb[t] == 1 and sa[t] == 0)
    out = Path(a.out_dir)
    outs: Dict[Path, str] = {}
    sets = {}
    for name, ids, rule in ((a.b_name, only_b, "B success 1 and A success 0"),
                            (a.a_name, only_a, "A success 1 and B success 0")):
        text = json_text(ids)
        outs[out / f"{name}.json"] = text
        sets[name] = {"file": f"{name}.json", "rule": rule, "n": len(ids),
                      "sha256": hashlib.sha256(text.encode()).hexdigest()}
    outs[out / "sets_manifest.json"] = json_text({
        "schema": SETS_SCHEMA, "a": fa, "b": fb, "sets": sets, "n_tasks": len(sa),
        "both": sum(1 for t in sa if sa[t] == 1 and sb[t] == 1),
        "neither": sum(1 for t in sa if sa[t] == 0 and sb[t] == 0)})
    write_outputs(outs, a.overwrite, a.replace)
    print(json.dumps({k: v["n"] for k, v in sets.items()}))
    return EXIT_OK


def cmd_pick(a) -> int:
    n = int(a.n_each)
    if n < 1:
        raise Refused("--n-each must be >= 1")
    ids: List[str] = []
    for f in [x for x in a.sets.split(",") if x]:
        for t in task_list(f)[:n]:
            if t not in ids:
                ids.append(t)
    write_outputs({Path(a.out): json_text(sorted(ids))}, a.overwrite, a.replace)
    print(json.dumps({"n": len(ids), "out": str(a.out)}))
    return EXIT_OK


# --- donors -------------------------------------------------------------------------------------------------------
def cmd_donors(a) -> int:
    d = Path(a.donor_dir)
    ids = task_list(a.tasks)
    proto = need_json(d / "protocol.json", "donor protocol")
    if proto.get("arm") != "c" or proto.get("start") != a.donor:
        raise Refused(f"{d}: arm {proto.get('arm')!r} start {proto.get('start')!r}, want arm 'c' start {a.donor!r}")
    psha = canon_sha256(proto)
    prefix, shas = {}, {}
    for tid in ids:
        p = d / "tasks" / tid / "trajectory.json"
        tr = need_json(p, "donor trajectory")
        acts = tr.get("actions")
        if tr.get("task_id") != tid or not isinstance(acts, list) or any(x not in VOCAB for x in acts):
            raise Refused(f"{p}: not the trajectory of {tid} or bad actions")
        if (tr.get("vpd") or {}).get("protocol_sha256") != psha:
            raise Refused(f"{p}: protocol_sha256 is not the donor run's ({psha[:12]})")
        prefix[tid] = list(acts)
        shas[tid] = sha256_file(p)
    doc = {"schema": FORCED_SCHEMA, "donor": a.donor, "source": apath(d), "prefix": prefix, "source_sha256": shas,
           "tasks_file": apath(a.tasks), "tasks_sha256": sha256_file(a.tasks),
           "source_protocol_sha256": sha256_file(d / "protocol.json")}
    write_outputs({Path(a.out): json_text(doc)}, a.overwrite, a.replace)
    print(json.dumps({"n": len(ids), "donor": a.donor, "out": str(a.out)}))
    return EXIT_OK


# --- check (the in-job guard) -------------------------------------------------------------------------------------
def set_problems(name: str, tasks, ids: List[str], man_path, pilot_root) -> List[str]:
    """The set file against the sets manifest (check, module doc)."""
    man = read_json(man_path)
    if not isinstance(man, dict) or man.get("schema") != SETS_SCHEMA:
        return [f"{man_path}: missing or not a {SETS_SCHEMA} manifest"]
    bad = []
    for side, start in (("a", PILOT_A), ("b", PILOT_B)):
        want = pilot_c_dir(pilot_root, start)
        if not same_dir((man.get(side) or {}).get("dir"), want):
            bad.append(f"{man_path}: {side} dir {(man.get(side) or {}).get('dir')!r} is not {want}")
    sets = man.get("sets") or {}
    for sname, (side, n) in EXPECTED_SETS.items():
        e = sets.get(sname) or {}
        f = Path(man_path).parent / str(e.get("file"))
        got = read_json(f)
        if (e.get("n") != n or not str(e.get("rule", "")).startswith(f"{side} success 1") or sha256_file(f) !=
                e.get("sha256") or not isinstance(got, list) or len(got) != n):
            bad.append(f"{man_path}: set {sname} is not the registered one ({n} tasks {side} solved, file {f})")
    if name in EXPECTED_SETS:
        e = sets.get(name) or {}
        if sha256_file(tasks) != e.get("sha256") or len(ids) != EXPECTED_SETS[name][1]:
            bad.append(f"{tasks}: not the manifest's {name} ({len(ids)} tasks, sha256 {str(sha256_file(tasks))[:12]})")
    elif name == SMOKE:
        pool = set()
        for sname in EXPECTED_SETS:
            got = read_json(Path(man_path).parent / str((sets.get(sname) or {}).get("file")))
            pool.update(got if isinstance(got, list) else [])
        if len(ids) != SMOKE_N or not set(ids) <= pool:
            bad.append(f"{tasks}: the smoke set must be {SMOKE_N} tasks of {' + '.join(EXPECTED_SETS)} "
                       f"({len(ids)} tasks, {len(set(ids) - pool)} outside)")
    else:
        bad.append(f"unknown set {name!r} (want {', '.join(list(EXPECTED_SETS) + [SMOKE])})")
    return bad


def check_inputs(a) -> Dict[str, Any]:
    bad: List[str] = []
    files = {apath(p): sha256_file(p) for p in (a.tasks, a.forced, a.sets_manifest)}
    try:
        ids = task_list(a.tasks)
    except Refused as e:
        return {"ok": False, "problems": [str(e)], "files": files}
    bad += set_problems(a.set, a.tasks, ids, a.sets_manifest, a.pilot_root)
    pilots = (PILOT_A, PILOT_B)
    if a.start not in pilots or a.donor not in pilots or a.donor == a.start:
        bad.append(f"start {a.start!r} / donor {a.donor!r}: the donor must be the other pilot start of {pilots}")
    if not same_dir(a.source, pilot_c_dir(a.pilot_root, a.donor)):
        bad.append(f"--source {a.source} is not the donor's pilot dir {pilot_c_dir(a.pilot_root, a.donor)}")
    doc = read_json(a.forced)
    if not isinstance(doc, dict) or doc.get("schema") != FORCED_SCHEMA:
        return {"ok": False, "problems": bad + [f"{a.forced}: missing or not a {FORCED_SCHEMA} file"], "files": files}
    if doc.get("donor") != a.donor:
        bad.append(f"{a.forced}: donor {doc.get('donor')!r} != {a.donor!r}")
    if doc.get("source") != apath(a.source):
        bad.append(f"{a.forced}: source {doc.get('source')!r} != {apath(a.source)!r}")
    if doc.get("tasks_sha256") != files[apath(a.tasks)]:
        bad.append(f"{a.forced}: built from another task file (tasks_sha256 {str(doc.get('tasks_sha256'))[:12]})")
    prefix, shas = doc.get("prefix") or {}, doc.get("source_sha256") or {}
    missing = [t for t in ids if t not in prefix or t not in shas]
    if missing:
        bad.append(f"{a.forced}: {len(missing)} tasks without a donor entry (first {missing[:3]})")
    changed, other = [], []
    for t in ids:
        p = Path(a.source) / "tasks" / t / "trajectory.json"
        if t in shas and sha256_file(p) != shas[t]:
            changed.append(t)
        if t in prefix and (read_json(p) or {}).get("actions") != prefix[t]:
            other.append(t)
    if changed:
        bad.append(f"{a.source}: {len(changed)} donor trajectories changed since the forced file (first {changed[:3]})")
    if other:
        bad.append(f"{a.forced}: {len(other)} prefixes are not the donor trajectory's actions (first {other[:3]})")
    return {"ok": not bad, "problems": bad, "files": files, "set": a.set, "n_tasks": len(ids), "start": a.start,
            "donor": a.donor, "source": apath(a.source)}


def cmd_check(a) -> int:
    rep = check_inputs(a)
    rep["schema"] = CHECK_SCHEMA
    write_text(a.out, json_text(rep))
    for b in rep["problems"]:
        print(f"[intervene] check: {b}", file=sys.stderr)
    print(json.dumps({"ok": rep["ok"], "problems": len(rep["problems"])}))
    return EXIT_OK if rep["ok"] else EXIT_REFUSED


def cmd_adapter_match(a) -> int:
    pt = Path(a.adapter_dir) / "adapter.pt"
    disk = sha256_file(pt)
    res = read_json(Path(a.eval_dir) / "results.json") or {}
    proto = read_json(Path(a.eval_dir) / "protocol.json") or {}
    served = (res.get("adapter") or {}).get("adapter_sha256")
    ok = disk is not None and served == disk and proto.get("adapter_sha256") == disk
    print(f"[intervene] adapter-match: {pt} {str(disk)[:12]}, {a.eval_dir} served {str(served)[:12]} "
          f"(protocol {str(proto.get('adapter_sha256'))[:12]}): {'ok' if ok else 'MISMATCH'}")
    return EXIT_OK if ok else EXIT_REFUSED


# --- runstate (the job's stale-input rule) --------------------------------------------------------------------
def forced_block_problems(fb, k: int, forced_sha, donor, harness_r5: Optional[str]) -> List[str]:
    fb = fb if isinstance(fb, dict) else {}
    want = {"k": k, "file_sha256": forced_sha, "donor": donor}
    bad = [f"forced.{x} {fb.get(x)!r} != {v!r}" for x, v in want.items() if fb.get(x) != v]
    if not fb.get("module_sha256"):
        bad.append("forced.module_sha256 missing")
    if harness_r5 and fb.get("harness_r5_sha256") != harness_r5:
        bad.append(f"forced.harness_r5_sha256 {str(fb.get('harness_r5_sha256'))[:12]} != {harness_r5[:12]}")
    return bad


def cmd_runstate(a) -> int:
    d = Path(a.dir)
    if not d.is_dir() or not any(d.iterdir()):
        print(f"[intervene] runstate: {d}: no run yet")
        return EXIT_OK
    proto = read_json(d / "protocol.json")
    if not isinstance(proto, dict):
        print(f"[intervene] runstate: {d}: a run without a readable protocol.json: stale", file=sys.stderr)
        return EXIT_STALE
    doc = need_json(a.forced, "forced file")
    ids = task_list(a.tasks)
    bad = forced_block_problems(proto.get("forced"), int(a.k), sha256_file(a.forced), doc.get("donor"),
                                a.harness_r5)
    want_s = {"sha256": sha256_file(a.tasks), "n": len(ids)}
    if proto.get("task_subset") != want_s:
        bad.append(f"task_subset {proto.get('task_subset')} != {want_s}")
    for b in bad:
        print(f"[intervene] runstate: {d}: {b}", file=sys.stderr)
    print(f"[intervene] runstate: {d}: " + ("these inputs" if not bad else "STALE (other inputs or code)"))
    return EXIT_OK if not bad else EXIT_STALE


# --- expect -------------------------------------------------------------------------------------------------------
def run_dir_items(d: Path, k: int, ids: List[str], doc: Dict[str, Any], tasks_sha: str, forced_sha: str,
                  harness_r5: Optional[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    def item(name, ok, detail=""):
        items.append({"name": f"f{k}:{name}", "ok": bool(ok), "detail": detail, "path": str(d)})

    res = read_json(d / "results.json") or {}
    item("results_valid", res.get("valid") is True and res.get("n_tasks") == len(ids) and not res.get("tasks_missing"),
         f"valid {res.get('valid')} n_tasks {res.get('n_tasks')}")
    proto = read_json(d / "protocol.json") or {}
    fb = proto.get("forced") or {}
    bad = forced_block_problems(fb, k, forced_sha, doc.get("donor"), harness_r5)
    summ = res.get("forced") or {}
    if any(summ.get(x) != fb.get(x) for x in ("k", "file_sha256", "donor", "module_sha256", "harness_r5_sha256")):
        bad.append("results forced summary != protocol forced")
    item("protocol_forced", not bad, "; ".join(bad))
    want_s = {"sha256": tasks_sha, "n": len(ids)}
    item("protocol_task_subset", proto.get("task_subset") == want_s, f"{proto.get('task_subset')} vs {want_s}")
    off = read_json(d / "official_check.json") or {}
    item("official_equal", off.get("equal") is True, f"equal {off.get('equal')}")
    off_k = []
    acts = doc.get("prefix") or {}
    for tid in ids:
        tr = read_json(d / "tasks" / tid / "trajectory.json") or {}
        f = (tr.get("vpd") or {}).get("forced") or {}
        want = forced_prefix(acts.get(tid) or [], k)
        if (f.get("k") != k or f.get("prefix") != want or f.get("n_forced") != len(want)
                or (tr.get("actions") or [])[:len(want)] != want):
            off_k.append(tid)
    item("trajectories_forced", not off_k, f"{len(off_k)} tasks off the forced prefix (first {off_k[:3]})")
    return items


def cmd_expect(a) -> int:
    ids = task_list(a.tasks)
    doc = need_json(a.forced, "forced file")
    items_k = [x for x in a.ks.split(",") if x != ""]
    if not items_k or not all(x.isascii() and x.isdigit() for x in items_k):
        raise Refused(f"--ks must be K[,K...] (integers >= 0), got {a.ks!r}")
    tasks_sha, forced_sha = sha256_file(a.tasks), sha256_file(a.forced)
    items: List[Dict[str, Any]] = []
    for k in [int(x) for x in items_k]:
        d = Path(a.out_root) / f"{a.prefix}__{a.start}__{a.tier}__c__f{k}"
        items += run_dir_items(d, k, ids, doc, tasks_sha, forced_sha, a.harness_r5)
    failed = [i["name"] for i in items if not i["ok"]]
    write_text(a.out, json_text({"schema": EXPECT_SCHEMA, "ok": not failed, "failed": failed, "items": items}))
    print(json.dumps({"ok": not failed, "failed": failed}))
    return EXIT_OK if not failed else EXIT_FAILED


# --- CLI ----------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def outputs(p):
        p.add_argument("--overwrite", action="store_true", help="rewrite an existing output with the same content")
        p.add_argument("--replace", action="store_true", help="change an existing output")

    s = sub.add_parser("sets")
    s.add_argument("--a-dir", required=True)
    s.add_argument("--b-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--a-name", default="only_a", help="A solved, B failed (default only_a)")
    s.add_argument("--b-name", default="only_b", help="B solved, A failed (default only_b)")
    outputs(s)
    p = sub.add_parser("pick")
    p.add_argument("--sets", required=True)
    p.add_argument("--n-each", type=int, required=True)
    p.add_argument("--out", required=True)
    outputs(p)
    d = sub.add_parser("donors")
    d.add_argument("--donor-dir", required=True)
    d.add_argument("--donor", required=True)
    d.add_argument("--tasks", required=True)
    d.add_argument("--out", required=True)
    outputs(d)
    c = sub.add_parser("check")
    for opt in ("--set", "--tasks", "--sets-manifest", "--pilot-root", "--start", "--forced", "--donor", "--source",
                "--out"):
        c.add_argument(opt, required=True)
    m = sub.add_parser("adapter-match")
    m.add_argument("--adapter-dir", required=True)
    m.add_argument("--eval-dir", required=True)
    r = sub.add_parser("runstate")
    for opt in ("--dir", "--k", "--tasks", "--forced"):
        r.add_argument(opt, required=True)
    r.add_argument("--harness-r5", default=None)
    e = sub.add_parser("expect")
    for opt in ("--out-root", "--prefix", "--start", "--tier", "--ks", "--tasks", "--forced", "--out"):
        e.add_argument(opt, required=True)
    e.add_argument("--harness-r5", default=None)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    fn = {"sets": cmd_sets, "pick": cmd_pick, "donors": cmd_donors, "check": cmd_check,
          "adapter-match": cmd_adapter_match, "runstate": cmd_runstate, "expect": cmd_expect}[a.cmd]
    try:
        return fn(a)
    except Refused as e:
        print(f"[intervene] REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
