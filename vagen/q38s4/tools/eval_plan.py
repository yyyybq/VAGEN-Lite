"""q38s4 stage 5 (PREREG_v1 sections 2-8): the C line's evaluations as eval_only training-chain plans, the main line's
machinery as the 27B base zero-shot used it (vagen/q38/base_zs/make_plan.py: the same sources, sha256 checks, set
composition and options).  Builds the chain folder, copies of the task sets and plan.json; submits nothing.

    python -m vagen.q38s4.tools.eval_plan length --arm c_nfp --step 500  --model DIR --commit HEAD
    python -m vagen.q38s4.tools.eval_plan quick  --arm ARM   --step 3000 --model DIR --commit HEAD
    python -m vagen.q38s4.tools.eval_plan full   --arm ARM   --step N    --model DIR --commit HEAD   (1000, 2000, 3441)

Chain folder q38/s4/evals/<kind>_<arm>_<step> (TR; must not exist), result name <arm>_<step> (T 0.8: <arm>_<step>_t08):
- length (PREREG section 8, the first bf16 copy, C-NFP step 500): EASI-8's 9 jobs with "limit" 50 (lmms-eval
  --limit: the first 50 questions of each lmms-eval task; blink is 14 sub-tasks, so 700 questions), only to measure
  answer tokens and seconds per question (tools.gen_length); results TR/qa, never mixed with a full run's.
- quick (section 2, step 3000; at 1000 and 2000 the full plan holds the same jobs): val 413 at T 0.8 (an "extra"
  set only the T 0.8 variant uses) + the probe at 256 tokens.  The chain's tables cover only eval_sets, so the val
  T 0.8 table is made by hand afterwards (TABLE_CMD; printed by main and in the plan's note).
- full (sections 3-7, steps 1000, 2000 and the final 3441): val 413, test 292 and the screened test's 59 new tasks,
  greedy and T 0.8 (the screened 674 is composed from these as for the base); P0 5,590 at 256 px; the probe at 256
  and 1,024 tokens (probe_variants probe1024); EASI-8 (9 jobs, site_video on 2 cards).
Every EASI job gets --no-stack-dump (qa_eval "no_stack_dump", section 8 / 11.3).  Pool h800, NORMAL priority
(VAGEN_LITE_JOB_PRIORITY=NORMAL in the chain's environment), action jobs on 2 cards (render service + vLLM at 0.9).

Checks before anything is written:
- the commit is the s4 worktree's HEAD;
- the model: config.json model_type qwen3_5, no nfp_head.* weight (vLLM's qwen3_5 loader refuses them), and it is the
  arm's step: C-noNFP the bf16 copy RUNS/c_nonfp/ckpt_bf16/checkpoint-<step> itself, C-NFP a strip_head copy whose
  strip_manifest.json names that bf16 copy as its source and a passed check; the bf16 copy's bf16_manifest.json names
  RUNS/<arm>/ckpt/checkpoint-<step> (it is written after the copy is verified);
- the sets: the sources' sha256 prefix and rows (make_plan.SOURCES), the screened sets = v2 rows + 59 new test tasks;
  then every copy is compared by md5 with the base's copy (q38/runs/q38-base-zs/sets; section 2): a difference stops
  here (report it; the base would be rerun on the C line's copy);
- P0 (5,590) and the probe (838): row counts, and not modified after the base's results were written (mtime).
TR/inputs.json records the sets' sources, sha256, md5 and the base comparison, and the P0 / probe files' sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from vagen.q38.base_zs import make_plan as MP

U = Path("/mnt/umm/users/yinbaiqiao")
WORKTREE = U / "VAGEN-Lite-q38-s4"
S4 = U / "q38/s4"
RUNS = S4 / "runs/formal"
EVALS = S4 / "evals"
BASE = U / "q38/runs/q38-base-zs"  # the base zero-shot: its set copies and P0 / probe results
ARMS = ("c_nfp", "c_nonfp")
FINAL = 3441
STEPS = {"length": (500,), "quick": (3000,), "full": (1000, 2000, FINAL)}
EASI_JOBS = ("vsi_bench", "mmsi_bench", "mindcube_tiny", "viewspatial", "site_image", "site_video", "blink",
             "3dsrbench", "embspatial")  # qa_eval.EASI_8_JOBS (checked in main against the module)
LENGTH_LIMIT = 50
MAX_JOBS = {"length": 9, "quick": 2, "full": 10}
SET_KEYS = dict(MP.SET_KEYS)  # {"val": "q38bz_eval_val_v2", "test": ..., "new59": ...}
HEAD_PREFIX = "nfp_head."
INDEX = "model.safetensors.index.json"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def weight_keys(model: Path) -> List[str]:
    """The tensor names of a copy (its index, else the single file's header)."""
    if (model / INDEX).is_file():
        return list(json.loads((model / INDEX).read_text())["weight_map"])
    single = model / "model.safetensors"
    if not single.is_file():
        raise SystemExit(f"{model}: no {INDEX} and no model.safetensors")
    raw = single.read_bytes()[:8]
    n = int.from_bytes(raw, "little")
    with open(single, "rb") as f:
        f.seek(8)
        header = json.loads(f.read(n))
    return [k for k in header if k != "__metadata__"]


def check_model(arm: str, step: int, model: Path, runs: Path = RUNS) -> Dict[str, Any]:
    """The evaluated folder is the arm's step and loads in vLLM (no head); returns what was checked."""
    copy = runs / arm / "ckpt_bf16" / f"checkpoint-{step}"
    cfg = json.loads((model / "config.json").read_text())
    if cfg.get("model_type") != "qwen3_5":
        raise SystemExit(f"{model}: model_type {cfg.get('model_type')}, expected qwen3_5")
    head = [k for k in weight_keys(model) if k.startswith(HEAD_PREFIX)]
    if head:
        raise SystemExit(f"{model} has {len(head)} {HEAD_PREFIX}* weights: evaluate a strip_head copy")
    bf = copy / "bf16_manifest.json"
    if not bf.is_file():
        raise SystemExit(f"{copy}: no bf16_manifest.json (the bf16 copy is missing or not verified yet)")
    src = Path(json.loads(bf.read_text()).get("source") or "")
    if src.resolve() != (runs / arm / "ckpt" / f"checkpoint-{step}").resolve():
        raise SystemExit(f"{bf}: source {src}, expected {runs / arm / 'ckpt' / f'checkpoint-{step}'}")
    out: Dict[str, Any] = {"model": str(model), "bf16_copy": str(copy), "bf16_manifest_md5": md5(bf)}
    if arm == "c_nonfp":
        if model.resolve() != copy.resolve():
            raise SystemExit(f"C-noNFP is evaluated from its bf16 copy {copy} itself, not {model}")
        return out
    man_path = model / "strip_manifest.json"
    if not man_path.is_file():
        raise SystemExit(f"{model}: no strip_manifest.json (C-NFP is evaluated from a strip_head copy)")
    man = json.loads(man_path.read_text())
    if Path(man.get("source") or "").resolve() != copy.resolve():
        raise SystemExit(f"{man_path}: source {man.get('source')}, expected {copy}")
    if Path(man.get("dst") or "").resolve() != model.resolve():
        raise SystemExit(f"{man_path}: dst {man.get('dst')}, expected {model}")
    if not (man.get("check") or {}).get("ok"):
        raise SystemExit(f"{man_path}: the strip check did not pass")
    if man.get("source_bf16_manifest_md5") not in (None, out["bf16_manifest_md5"]):
        raise SystemExit(f"{man_path}: made from another bf16 copy (bf16_manifest md5 differs)")
    return {**out, "strip_manifest_md5": md5(man_path)}


def check_commit(commit: str, worktree: Path = WORKTREE) -> None:
    head = subprocess.run(["git", "-C", str(worktree), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    if len(commit) < 8 or not head.startswith(commit):
        raise SystemExit(f"--commit {commit!r} is not the s4 worktree's HEAD {head[:8]} (commit the changes first)")


def screened_new59() -> List[Dict[str, Any]]:
    """make_plan's composition: the screened test's tasks not in eval_test_v2, after the same checks."""
    for src, prefix, n in MP.SOURCES.values():
        MP.check_source(src, prefix, n)
    MP.check_source(*MP.SCREENED_TEST)
    MP.check_source(*MP.SCREENED_VAL)
    test_rows = {r["task_id"]: r for r in MP.rows(MP.SOURCES[SET_KEYS["test"]][0])}
    val_rows = {r["task_id"]: r for r in MP.rows(MP.SOURCES[SET_KEYS["val"]][0])}
    screened = MP.rows(MP.SCREENED_TEST[0])
    new = [r for r in screened if r["task_id"] not in test_rows]
    if len(new) != 59 or any(test_rows[r["task_id"]] != r for r in screened if r["task_id"] in test_rows) \
            or any(val_rows.get(r["task_id"]) != r for r in MP.rows(MP.SCREENED_VAL[0])):
        raise SystemExit("the screened sets are not the v2 rows plus 59 new test tasks")
    return new


def set_name(kind: str, arm: str, step: int, key: str) -> str:
    """A new name for every copy (section 8): the base's name with this run's prefix instead of q38bz."""
    return f"c4{kind[0]}_{arm}_{step}_" + SET_KEYS[key][len("q38bz_"):]


def write_sets(tr: Path, kind: str, arm: str, step: int, keys: List[str], base: Path = BASE) -> Dict[str, Any]:
    """Copies of the sets in TR/sets (read-only, source.json as make_plan.write_set), each compared by md5 with the
    base's copy; a difference stops before the plan is written."""
    new = screened_new59() if "new59" in keys else None
    if "new59" not in keys:
        for key in keys:
            MP.check_source(*MP.SOURCES[SET_KEYS[key]])
    done: Dict[str, Any] = {}
    for key in keys:
        name = set_name(kind, arm, step, key)
        if key == "new59":
            MP.write_set(tr, name, new, {"source": str(MP.SCREENED_TEST[0]), "what": "the screened test set's tasks "
                         "not in eval_test_v2 (the scene screen's 59 replacements), in its order (make_plan)",
                                         "task_ids": [r["task_id"] for r in new]})
        else:
            src = MP.SOURCES[SET_KEYS[key]][0]
            MP.write_set(tr, name, src.read_bytes(), {"source": str(src)})
        mine = tr / "sets" / name / "tasks.jsonl"
        theirs = base / "sets" / SET_KEYS[key] / "tasks.jsonl"
        same = theirs.is_file() and md5(mine) == md5(theirs)
        done[key] = {"dir": str(mine.parent), "md5": md5(mine), "base_copy": str(theirs),
                     "base_md5": md5(theirs) if theirs.is_file() else None, "same_as_base": same}
        if not same:
            raise SystemExit(f"{mine} differs from the base's copy {theirs} (md5): report it before any result is "
                             "read (PREREG_v1 section 2: the base is rerun on the C line's copy)")
    return done


def check_qa_file(path: Path, rows: int, base_done: Path) -> Dict[str, Any]:
    if len(MP.rows(path)) != rows:
        raise SystemExit(f"{path}: {len(MP.rows(path))} rows, expected {rows}")
    if not base_done.is_file():
        raise SystemExit(f"{base_done}: the base's result is missing")
    if path.stat().st_mtime > base_done.stat().st_mtime:
        raise SystemExit(f"{path} was modified after the base's result {base_done} was written")
    return {"path": str(path), "sha256": MP.sha(path), "rows": rows, "base_done": str(base_done)}


def base_plan(tr: Path, commit: str, model: Path, kind: str) -> Dict[str, Any]:
    return {"tr": str(tr), "sft_dir": str(tr / "sft_unused"), "commit": commit, "eval_only": True,
            "model_type": "qwen3_5", "model": str(model), "job_prefix": "q38s4-ev", "pool": "h800",
            "eval_cards": 2, "max_eval_jobs": MAX_JOBS[kind], "retries": 2, "suspended_grace_min": 10}


def easi_block(tr: Path, name: str) -> Dict[str, Any]:
    return {"out": str(tr / "qa"), "easi": "easi_8", "cards": {"site_video": 2}, "no_stack_dump": list(EASI_JOBS),
            "models": [name]}


TABLE_CMD = "python -m data_gen.interiorgs_v10.evaluation table --eval {eval_dir}"


def build(kind: str, arm: str, step: int, model: Path, commit: str, tr: Path,
          sets: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The plan (pure: the set folders come from write_sets)."""
    name = f"{arm}_{step}"
    plan = base_plan(tr, commit, model, kind)
    sets = sets or {}
    if kind == "length":
        plan.update({"eval_sets": {}, "eval": {"models": {name: {"model": str(model), "options": []}}},
                     "qa_eval": {**easi_block(tr, name), "limit": LENGTH_LIMIT},
                     "note": f"C line {arm} step {step}: the answer-length measurement of PREREG_v1 section 8 -- "
                             f"EASI-8's 9 jobs on the first {LENGTH_LIMIT} questions of each lmms-eval task (--limit; blink: 14 "
                             f"sub-tasks, {14 * LENGTH_LIMIT} questions), "
                             "--no-stack-dump; only tokens and seconds per question are read (tools.gen_length), "
                             "no score."})
    elif kind == "quick":
        plan.update({"eval_sets": {},
                     "eval": {"models": {name: {"model": str(model), "options": MP.GREEDY}},
                              "extra_sets": {"val": sets["val"]["dir"]},
                              "variants": [{"suffix": MP.T08_SUFFIX, "sets": ["val"], "options": MP.T08,
                                            "models": [name]}]},
                     "qa_eval": {"out": str(tr / "qa"), "easi": [], "probe": str(MP.PROBE), "models": [name]},
                     "note": f"C line {arm} step {step}: the quick evaluation of PREREG_v1 section 2 -- val 413 at "
                             "T 0.8 (top-p 0.92, entropy, format example action_names; as the base's q38base_t08) "
                             "and the probe at 256 tokens.  Its table is not made by the chain (tables() covers "
                             "eval_sets only): afterwards run " + TABLE_CMD.format(eval_dir=sets["val"]["dir"]) + "."})
    elif kind == "full":
        keys = list(SET_KEYS)
        plan.update({"eval_sets": {k: sets[k]["dir"] for k in keys},
                     "eval": {"models": {name: {"model": str(model), "options": MP.GREEDY}},
                              "variants": [{"suffix": MP.T08_SUFFIX, "sets": keys, "options": MP.T08,
                                            "models": [name]}]},
                     "qa_eval": {**easi_block(tr, name), "p0": str(MP.P0_SEL), "p0_res": 256,
                                 "probe": str(MP.PROBE), "probe_variants": {"probe1024": {"max_tokens": 1024}}},
                     "note": f"C line {arm} step {step}: the full evaluation of PREREG_v1 sections 3-7 -- val 413, "
                             "test 292, the screened test's 59 new tasks, greedy and T 0.8 (as the base's q38base / "
                             "q38base_t08); P0 5,590 at 256 px; the probe at 256 and 1,024 tokens; EASI-8 (9 jobs, "
                             "site_video on 2 cards, --no-stack-dump)."})
    else:
        raise SystemExit(f"unknown kind {kind}")
    plan["note"] += " h800, NORMAL priority (VAGEN_LITE_JOB_PRIORITY in the chain's environment)."
    return plan


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=sorted(STEPS))
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--step", required=True, type=int)
    ap.add_argument("--model", required=True, type=Path, help="C-NFP: a strip_head copy; C-noNFP: the bf16 copy")
    ap.add_argument("--commit", required=True, help="the s4 worktree's HEAD (the chain's check_code)")
    a = ap.parse_args(argv)
    from data_gen.interiorgs_v10.qa_eval import EASI_8_JOBS

    if tuple(EASI_8_JOBS) != EASI_JOBS:
        raise SystemExit(f"qa_eval.EASI_8_JOBS changed: {EASI_8_JOBS}")
    if a.step not in STEPS[a.kind]:
        raise SystemExit(f"{a.kind}: step {a.step} not in {STEPS[a.kind]} (PREREG_v1 sections 2 and 8)")
    if a.kind == "length" and a.arm != "c_nfp":
        raise SystemExit("length: the first bf16 copy is C-NFP's (section 8)")
    check_commit(a.commit)
    model = a.model.resolve()
    checked = check_model(a.arm, a.step, model)
    tr = EVALS / f"{a.kind}_{a.arm}_{a.step}"
    if tr.exists():
        raise SystemExit(f"{tr} exists")
    qa_files = {}
    if a.kind in ("quick", "full"):
        qa_files["probe"] = check_qa_file(MP.PROBE, MP.PROBE_ROWS, BASE / "qa/probe/q38base/done.json")
    if a.kind == "full":
        qa_files["p0"] = check_qa_file(MP.P0_SEL, MP.P0_ROWS, BASE / "qa/p0/q38base/done.json")
    tr.mkdir(parents=True)
    keys = {"length": [], "quick": ["val"], "full": list(SET_KEYS)}[a.kind]
    sets = write_sets(tr, a.kind, a.arm, a.step, keys) if keys else {}
    plan = build(a.kind, a.arm, a.step, model, a.commit, tr, sets)
    (tr / "inputs.json").write_text(json.dumps({"model": checked, "sets": sets, "qa_files": qa_files},
                                               indent=1) + "\n")
    (tr / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n")
    print(tr / "plan.json")
    print(json.dumps({k: v["md5"] for k, v in sets.items()}))
    if a.kind == "quick":
        print("after the chain: " + TABLE_CMD.format(eval_dir=sets["val"]["dir"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
