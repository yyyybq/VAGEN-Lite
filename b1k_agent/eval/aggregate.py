"""Result tables from OFFICIAL evaluator outputs plus agent episode summaries (red line 6: never hand-typed).

    python -m eval.aggregate RUN_DIR [RUN_DIR ...] --out OUT_DIR [--instances-declared FILE]
                             [--bootstrap 2000] [--seed 0] [--compare CFG_A CFG_B]

Inputs per run directory (all read-only; each file's sha256 goes into the provenance record):
  <run>/eval_output/json/<task>_<instance_id>_<rollout_id>.json   official per-rollout results, or the
        directory named by status.json ``eval_output_dir`` (OmniGibson/omnigibson/eval/evaluator.py record_fn)
  <run>/status.json         launcher status (optional). Read keys: eval_output_dir, task, instance_ids
        (resolved) or mode + instance_indices, num_rollouts, seed, max_steps, config_name, config_hash,
        infra_error (whole run failed for infrastructure reasons), infra_errors ([{task, instance_id,
        rollout_id, reason}]), evaluator_exit_code / exit_code
  <run>/run_manifest.json   agent manifest (agent/config.py build_manifest): config, config_hash, seed
  <run>/episodes/*/summary.json   agent episode summaries (agent/recorder.py episode_end, NAS copies)

Scoring mirrors OmniGibson/omnigibson/eval/utils/score_utils.py compute_final_q_score (v3.9.3-post2):
per-task mean of ``q_score.final``, overall = mean over tasks, success = q == 1 / ``success``, time score
= 1.5/0.5 - 1/(0.5 * normalized_time). Deliberate differences, all visible in the outputs:
  * the per-task divisor is the number of evaluated (or declared) instances, not the fixed 20 of
    score_utils (docs/PROTOCOL.md: the report set is 10 instances);
  * infrastructure errors (failure_reason in --infra-failure-reasons, or flagged by status.json) are
    counted and listed but excluded from the Q means; ``mean_q_infra_as_zero`` shows the strict value;
  * with ``--instances-declared``, every declared (task, instance) without a result counts as Q = 0 and
    is flagged ``missing``; results outside the declaration are kept in the CSV but not averaged;
  * ``q_budget2026`` applies the 2026 budget rule (Q = 0 when the rollout broke max_steps x 1 s or the
    600 s per-query cap) next to the raw official ``q_final``. budget_ok comes from failure_reason
    (2026/eval JSONs), else the official multi-port ``time.evaluation_time``, else the agent's own
    ``elapsed_s`` (clock starts after the uncharged first query), each compared with the budget; unknown
    stays unknown and is counted, not zeroed;
  * re-runs (6.5): attempts of one (config, task, instance, seed, rollout) are taken in input order; the
    first non-infra attempt counts, earlier infra attempts are superseded, later attempts are flagged
    (policy failures must never be re-run) and never replace the first valid result.

The bootstrap CI is task-stratified: resample episodes (instances x seeds) with replacement within each
task, average per task, then over tasks. Paired comparisons use the (task, instance, seed, rollout)
units evaluated in both configs (missing, infra and superseded rows left out).

Status: written and unit-tested on SYNTHETIC result files (tests/test_eval_analysis_*.py); not yet run
on real outputs from server 105.
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import datetime
import glob
import hashlib
import json
import math
import os
import re
import sys
import zlib
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import yaml

from agent.config import REPO_ROOT, git_info

EVAL_TIMEOUT_MULTIPLIER = 1.5  # OmniGibson/omnigibson/eval/utils/eval_utils.py
TEST_ID_START = {"public_test": 301, "hidden_test": 321}  # eval_utils.TEST_INSTANCE_IDS = range(301, 341)
SPLIT_SIZE = 20  # NUM_PUBLIC_TEST_INSTANCES; hidden = 40 - 20
BUDGET_FAILURE_REASONS = frozenset({"group_time_budget_exceeded", "policy_query_timeout"})
DEFAULT_INFRA_FAILURE_REASONS = ("policy_connection_lost",)
SECONDS_PER_STEP = 1.0  # 2026 rule: rollout budget = max_steps x 1 s
MAX_INFRA_RERUNS = 2

# Keys that do not change agent behaviour; dropped before hashing configs into one comparison group.
# run.seed is its own column so that seeds of one config pool together.
VOLATILE_CONFIG_KEYS = (
    "_meta",
    "run.seed",
    "run.log_root",
    "run.local_log_root",
    "run.frame_every",
    "run.save_frames",
    "server.host",
    "server.port",
    "executor.params.host",
    "executor.params.port",
)

ROW_FIELDS = (
    "config", "config_key", "config_hash", "run_id", "task", "instance_id", "rollout_id", "seed",
    "q_final", "q_budget2026", "success", "steps", "sim_time", "normalized_time", "time_score",
    "norm_dist_base", "norm_dist_left", "norm_dist_right",
    "failure_reason", "budget_ok", "budget_source", "evaluation_time", "elapsed_s", "wallclock_budget_s",
    "provider_calls", "provider_s", "tokens_in", "tokens_out", "cost_usd", "provider_errors",
    "recoveries", "replans", "fallbacks", "n_subtasks",
    "infra_error", "infra_reason", "missing", "missing_reason", "in_declared", "superseded",
    "superseded_reason", "attempt", "included", "synthetic",
    "episode_id", "join", "json_path", "video_path", "summary_path", "run_dir",
)

SECONDARY = ("time_score", "steps", "sim_time", "norm_dist_base", "norm_dist_left", "norm_dist_right",
             "provider_calls", "provider_s", "cost_usd", "recoveries", "replans", "fallbacks")


# ------------------------------------------------------------------------------------ small helpers


def _int(x: Any) -> Optional[int]:
    try:
        return int(str(x).strip())
    except (TypeError, ValueError):
        return None


def _float(x: Any) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _get(d: Any, dotted: str, default: Any = None) -> Any:
    for p in dotted.split("."):
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def _pop_dotted(d: dict, dotted: str) -> None:
    parts = dotted.split(".")
    for p in parts[:-1]:
        d = d.get(p)
        if not isinstance(d, dict):
            return
    d.pop(parts[-1], None)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def time_score(normalized_time: Optional[float]) -> Optional[float]:
    if normalized_time is None or not math.isfinite(normalized_time) or normalized_time <= 0:
        return None
    m = EVAL_TIMEOUT_MULTIPLIER
    return m / (m - 1.0) - 1.0 / ((m - 1.0) * normalized_time)


def resolve_instance_ids(mode: str, indices: Iterable[int]) -> list[int]:
    """Mirror of evaluator.resolve_instance_ids: public_test i -> 301+i, hidden_test i -> 321+i, train raw."""
    indices = [int(i) for i in indices]
    if mode == "train":
        return indices
    if mode not in TEST_ID_START:
        raise ValueError(f"unknown mode {mode!r}")
    bad = [i for i in indices if not 0 <= i < SPLIT_SIZE]
    if bad:
        raise ValueError(f"{mode} indices must be in range({SPLIT_SIZE}), got {bad}")
    return [TEST_ID_START[mode] + i for i in indices]


def parse_indices(spec: Any) -> list[int]:
    """"0-9", "0,3,5", 4 or [0, 1] -> list of ints."""
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def config_key(cfg: dict) -> str:
    clean = json.loads(json.dumps(cfg, default=str))
    for k in VOLATILE_CONFIG_KEYS:
        _pop_dotted(clean, k)
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class InputLog:
    """Reads input files once, read-only, and remembers each file's sha256 for provenance."""

    def __init__(self):
        self.files: dict[str, dict] = {}

    def read_bytes(self, path: str) -> bytes:
        path = os.path.abspath(path)
        with open(path, "rb") as f:
            data = f.read()
        self.files[path] = {"path": path, "sha256": sha256_bytes(data), "bytes": len(data)}
        return data

    def json(self, path: str) -> Any:
        return json.loads(self.read_bytes(path).decode("utf-8"))

    def yaml(self, path: str) -> Any:
        return yaml.safe_load(self.read_bytes(path).decode("utf-8"))

    def listing(self) -> list[dict]:
        return [self.files[p] for p in sorted(self.files)]


# ------------------------------------------------------------------------------------ one run directory


@dataclasses.dataclass
class RunData:
    run_dir: str
    run_id: str
    config_name: str
    config_hash: Optional[str]
    config_key: str
    seed: Any
    status: dict
    manifest: dict
    eval_output_dir: str
    official: list  # [(path, record)]
    summaries: list  # [(path, summary)]
    order: int = 0
    warnings: list = dataclasses.field(default_factory=list)


def _eval_output_dir(run_dir: str, status: dict) -> tuple[str, Optional[str]]:
    default = os.path.join(run_dir, "eval_output")
    cand = status.get("eval_output_dir")
    if not cand:
        return default, None
    path = cand if os.path.isabs(cand) else os.path.join(run_dir, cand)
    if os.path.isdir(path):
        return path, None
    if os.path.isdir(default):
        return default, f"status eval_output_dir {cand!r} not found; using {default}"
    return path, f"eval output dir {path} not found"


def load_run(run_dir: str, inputs: InputLog, order: int = 0) -> RunData:
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run dir {run_dir} does not exist")
    status_path = os.path.join(run_dir, "status.json")
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    status = inputs.json(status_path) if os.path.isfile(status_path) else {}
    manifest = inputs.json(manifest_path) if os.path.isfile(manifest_path) else {}
    warnings = []
    out_dir, warn = _eval_output_dir(run_dir, status)
    if warn:
        warnings.append(warn)

    official = []
    for p in sorted(glob.glob(os.path.join(out_dir, "json", "*.json"))):
        try:
            official.append((os.path.abspath(p), inputs.json(p)))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            warnings.append(f"unreadable official result {p}: {e}")
    summaries = []
    for p in sorted(glob.glob(os.path.join(run_dir, "episodes", "*", "summary.json"))):
        try:
            summaries.append((os.path.abspath(p), inputs.json(p)))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            warnings.append(f"unreadable episode summary {p}: {e}")

    cfg = manifest.get("config") if isinstance(manifest.get("config"), dict) else None
    name = _get(cfg, "run.name") or status.get("config_name") or os.path.basename(run_dir)
    chash = manifest.get("config_hash") or status.get("config_hash")
    key = config_key(cfg) if cfg is not None else (chash or f"name:{name}")
    seed = manifest.get("seed", _get(cfg, "run.seed"))
    if seed is None:
        seed = status.get("seed")
    run_id = manifest.get("run_id") or status.get("run_id") or os.path.basename(run_dir)
    if not official and not status:
        warnings.append(f"run {run_dir}: no official results and no status.json")
    return RunData(run_dir=run_dir, run_id=str(run_id), config_name=str(name), config_hash=chash, config_key=key,
                   seed=seed, status=status, manifest=manifest, eval_output_dir=out_dir, official=official,
                   summaries=summaries, order=order, warnings=warnings)


def _status_instance_ids(status: dict) -> Optional[list[int]]:
    ids = status.get("instance_ids")
    if ids is not None:
        return [int(i) for i in ids]
    if status.get("instance_indices") is not None and status.get("mode"):
        return resolve_instance_ids(status["mode"], parse_indices(status["instance_indices"]))
    return None


def _official_key(path: str, rec: dict) -> tuple[str, int, int]:
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.rsplit("_", 2)
    task = rec.get("task") or parts[0]
    iid = _int(rec.get("instance_id"))
    rid = _int(rec.get("rollout_id"))
    if iid is None and len(parts) == 3:
        iid = _int(parts[1])
    if rid is None:
        rid = _int(parts[2]) if len(parts) == 3 else 0
    return str(task), int(iid if iid is not None else -1), int(rid or 0)


_EP_RE = re.compile(r"-ep(\d+)-env(\d+)$")


def match_summaries(run: RunData, official_keys: Sequence[tuple]) -> tuple[dict, list]:
    """Assign each agent episode summary to an official (task, instance_id, rollout_id).

    Order of evidence: the server's --tag instance=<id> (a comma list is indexed by the env index in
    the episode id), else status.json instance ids indexed by env index, else the only result of that
    task in the run. Rollout = tag, else the rank of the episode counter among that instance's episodes
    (one evaluator run() == one reset == one rollout, evaluator.py run()).
    """
    tasks = {k[0] for k in official_keys}
    status_task = run.status.get("task") or run.status.get("task_name")
    status_ids = _status_instance_ids(run.status)
    cands = []
    for path, s in run.summaries:
        tags = s.get("tags") or {}
        task = tags.get("task") or s.get("task_name") or status_task
        if task not in tasks and status_task:
            task = status_task
        m = _EP_RE.search(str(s.get("episode_id", "")))
        ep, env = (int(m.group(1)), int(m.group(2))) if m else (0, None)
        raw = tags.get("instance_id", tags.get("instance"))
        tag_ids = [i for i in (_int(x) for x in str(raw).split(",")) if i is not None] if raw is not None else []
        iid, how = None, None
        if len(tag_ids) == 1:
            iid, how = tag_ids[0], "tag"
        elif len(tag_ids) > 1 and env is not None and env < len(tag_ids):
            iid, how = tag_ids[env], "tag_env_index"
        elif status_ids and env is not None and env < len(status_ids):
            iid, how = status_ids[env], "status_env_index"
        rollout = _int(tags.get("rollout_id", tags.get("rollout")))
        cands.append({"task": task, "iid": iid, "rollout": rollout, "ep": ep, "how": how, "path": path, "s": s})

    by_inst = collections.defaultdict(list)
    for c in cands:
        if c["iid"] is not None and c["rollout"] is None:
            by_inst[(c["task"], c["iid"])].append(c)
    for group in by_inst.values():
        for r, c in enumerate(sorted(group, key=lambda c: (c["ep"], c["path"]))):
            c["rollout"] = r
    unresolved = collections.Counter(c["task"] for c in cands if c["iid"] is None)
    for c in cands:
        if c["iid"] is None:
            only = [k for k in official_keys if k[0] == c["task"]]
            if len(only) == 1 and unresolved[c["task"]] == 1:
                c["iid"], c["rollout"], c["how"] = only[0][1], only[0][2], "only_episode"

    matched, unmatched = {}, []
    for c in cands:
        key = (c["task"], c["iid"], c["rollout"])
        if c["iid"] is None or key in matched:
            unmatched.append(c["path"])
            continue
        matched[key] = (c["path"], c["s"], c["how"])
    return matched, unmatched


def _infra_flags(status: dict) -> tuple[Optional[str], dict]:
    """(run-level infra reason or None, {(task, instance_id, rollout_id): reason}) from status.json."""
    whole = status.get("infra_error")
    whole = "infra error (status.json)" if whole is True else (str(whole) if whole else None)
    exit_code = status.get("evaluator_exit_code", status.get("exit_code"))
    if whole is None and exit_code not in (None, 0):
        whole = f"evaluator exit code {exit_code}"
    per = {}
    for e in status.get("infra_errors") or []:
        key = (str(e.get("task") or status.get("task") or ""), _int(e.get("instance_id")), _int(e.get("rollout_id")) or 0)
        per[key] = str(e.get("reason") or "infra error (status.json)")
    return whole, per


def _budget(rec: Optional[dict], summary: Optional[dict], status: dict) -> tuple[Optional[bool], str, Optional[float], Optional[float], Optional[float]]:
    budget_s = _float((summary or {}).get("wallclock_budget_s"))
    if budget_s is None and status.get("max_steps") is not None:
        budget_s = float(status["max_steps"]) * SECONDS_PER_STEP
    elapsed = _float((summary or {}).get("elapsed_s"))
    eval_time = _float(_get(rec, "time.evaluation_time"))
    if rec is not None and "failure_reason" in rec:
        return rec["failure_reason"] not in BUDGET_FAILURE_REASONS, "failure_reason", budget_s, elapsed, eval_time
    if eval_time is not None and budget_s is not None:
        return eval_time <= budget_s, "evaluation_time", budget_s, elapsed, eval_time
    if elapsed is not None and budget_s is not None:
        return elapsed <= budget_s, "agent_elapsed", budget_s, elapsed, eval_time
    return None, "unknown", budget_s, elapsed, eval_time


def _base_row(run: RunData, task: str, iid: int, rid: int) -> dict:
    row = {f: None for f in ROW_FIELDS}
    row.update(config=run.config_name, config_key=run.config_key, config_hash=run.config_hash, run_id=run.run_id,
               task=task, instance_id=iid, rollout_id=rid, seed=run.seed, infra_error=False, missing=False,
               in_declared=True, superseded=False, synthetic=False, run_dir=run.run_dir,
               video_path=os.path.join(run.eval_output_dir, "videos", f"{task}_{iid}_{rid}.mp4"))
    return row


def _attach_summary(row: dict, entry: Optional[tuple]) -> None:
    if entry is None:
        return
    path, s, how = entry
    row.update(episode_id=s.get("episode_id"), join=how, summary_path=path,
               provider_calls=s.get("provider_calls"), provider_s=s.get("provider_s", s.get("provider_time_s")),
               tokens_in=s.get("tokens_in"), tokens_out=s.get("tokens_out"), cost_usd=s.get("cost_usd"),
               provider_errors=s.get("provider_errors"), recoveries=s.get("recoveries"), replans=s.get("replans"),
               fallbacks=s.get("fallbacks"),
               n_subtasks=len(s["subtasks"]) if isinstance(s.get("subtasks"), list) else None)
    if s.get("_synthetic"):
        row["synthetic"] = True


def episode_rows(run: RunData, infra_reasons: Iterable[str] = DEFAULT_INFRA_FAILURE_REASONS) -> tuple[list[dict], list[str]]:
    """One row per official result of the run, plus placeholders for results status.json expected."""
    infra_reasons = set(infra_reasons)
    warnings = list(run.warnings)
    keyed = [(_official_key(p, r), p, r) for p, r in run.official]
    matched, unmatched = match_summaries(run, [k for k, _, _ in keyed])
    whole_infra, per_infra = _infra_flags(run.status)
    rows = []
    seen = set()
    for (task, iid, rid), path, rec in keyed:
        row = _base_row(run, task, iid, rid)
        q = _float(_get(rec, "q_score.final"))
        reason = rec.get("failure_reason")
        nt = _float(_get(rec, "time.normalized_time"))
        entry = matched.get((task, iid, rid))
        summary = entry[1] if entry else None
        ok, source, budget_s, elapsed, eval_time = _budget(rec, summary, run.status)
        row.update(q_final=q, success=bool(rec.get("success")), steps=rec.get("steps"),
                   sim_time=_get(rec, "time.simulator_time"), normalized_time=nt, time_score=time_score(nt),
                   norm_dist_base=_get(rec, "normalized_agent_distance.base"),
                   norm_dist_left=_get(rec, "normalized_agent_distance.left"),
                   norm_dist_right=_get(rec, "normalized_agent_distance.right"),
                   failure_reason=reason, budget_ok=ok, budget_source=source, evaluation_time=eval_time,
                   elapsed_s=elapsed, wallclock_budget_s=budget_s, json_path=path,
                   synthetic=bool(rec.get("_synthetic")))
        _attach_summary(row, entry)
        infra = per_infra.get((task, iid, rid)) or (f"failure_reason={reason}" if reason in infra_reasons else None)
        if infra:
            row.update(infra_error=True, infra_reason=infra)
        if q is None and not infra:
            row.update(infra_error=True, infra_reason="official result has no q_score.final")
        rows.append(row)
        seen.add((task, iid, rid))

    status_task = run.status.get("task") or run.status.get("task_name")
    ids = _status_instance_ids(run.status)
    if status_task and ids:
        for iid in ids:
            for rid in range(int(run.status.get("num_rollouts") or 1)):
                if (status_task, iid, rid) in seen:
                    continue
                row = _base_row(run, status_task, iid, rid)
                _attach_summary(row, matched.get((status_task, iid, rid)))
                infra = per_infra.get((status_task, iid, rid)) or whole_infra
                if infra:
                    row.update(infra_error=True, infra_reason=infra)
                else:
                    row.update(missing=True, missing_reason="expected by status.json, no official result",
                               q_final=0.0, success=False, budget_source="unknown")
                rows.append(row)
    for p in unmatched:
        warnings.append(f"episode summary not matched to an official result: {p}")
    return rows, warnings


# ------------------------------------------------------------------------------------ declared instances


def load_declared(path: str, inputs: InputLog) -> dict:
    """Declared (task -> instance ids) list, YAML or JSON. Accepted forms::

        tasks: {turning_on_radio: [301, 302], picking_up_trash: {mode: public_test, indices: "0-9"}}
        mode: public_test, indices: "0-9", tasks: [turning_on_radio, picking_up_trash]
        seeds: [0, 1, 2]        # optional; default: the seeds each config was run with
    """
    raw = inputs.yaml(path)  # JSON is read by the same YAML parser
    if isinstance(raw, list):
        raw = {"tasks": raw}
    tasks = raw.get("tasks") or {}
    top_mode, top_idx = raw.get("mode"), raw.get("indices")
    out: dict[str, list[int]] = {}
    items = tasks.items() if isinstance(tasks, dict) else ((t, None) for t in tasks)
    for task, spec in items:
        if spec is None:
            if top_mode is None or top_idx is None:
                raise ValueError(f"declared task {task!r} has no instances and no top-level mode/indices")
            ids = resolve_instance_ids(top_mode, parse_indices(top_idx))
        elif isinstance(spec, dict):
            ids = resolve_instance_ids(spec.get("mode", top_mode or "train"), parse_indices(spec["indices"]))
        else:
            ids = [int(i) for i in parse_indices(spec)]
        out[str(task)] = sorted(set(ids))
    seeds = raw.get("seeds")
    return {"path": os.path.abspath(path), "tasks": out, "seeds": list(seeds) if seeds is not None else None}


# ------------------------------------------------------------------------------------ row post-processing


def resolve_attempts(rows: list[dict]) -> list[str]:
    """Rule 6.5: first non-infra attempt per (config, task, instance, seed, rollout) counts.

    A result status.json expected but the evaluator never wrote (no infra flag) is an attempt scored 0.
    """
    warnings = []
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["config_key"], r["task"], r["instance_id"], str(r["seed"]), r["rollout_id"])].append(r)
    for key, attempts in groups.items():
        chosen = next((r for r in attempts if not r["infra_error"]), attempts[-1])
        ci = attempts.index(chosen)
        for i, r in enumerate(attempts):
            r["attempt"] = i + 1
            if r is chosen:
                continue
            r["superseded"] = True
            r["superseded_reason"] = "infra attempt, re-run" if i < ci else "later attempt after a counted result"
        if ci > MAX_INFRA_RERUNS:
            warnings.append(f"{key[1]}/{key[2]} seed {key[3]} ({chosen['config']}): {ci} infra re-runs (max {MAX_INFRA_RERUNS})")
        later = len(attempts) - ci - 1
        if later:
            warnings.append(f"{key[1]}/{key[2]} seed {key[3]} ({chosen['config']}): {later} attempt(s) after a counted "
                            f"result ignored; policy failures must not be re-run (6.5)")
    return warnings


def apply_declared(rows: list[dict], declared: Optional[dict], configs: dict) -> list[dict]:
    """Flag rows outside the declaration and zero-fill declared (task, instance) pairs with no result."""
    if not declared:
        return []
    want = {(t, i) for t, ids in declared["tasks"].items() for i in ids}
    for r in rows:
        r["in_declared"] = (r["task"], r["instance_id"]) in want
    added = []
    for key, c in configs.items():
        seeds = declared["seeds"] if declared["seeds"] is not None else sorted(c["seeds"], key=str)
        have = {(r["task"], r["instance_id"], str(r["seed"])) for r in rows
                if r["config_key"] == key and r["rollout_id"] == 0 and not r["superseded"]}
        for seed in seeds:
            for task, iid in sorted(want):
                if (task, iid, str(seed)) in have:
                    continue
                row = {f: None for f in ROW_FIELDS}
                row.update(config=c["label"], config_key=key, config_hash=None, run_id="", task=task, instance_id=iid,
                           rollout_id=0, seed=seed, q_final=0.0, q_budget2026=0.0, success=False,
                           budget_source="unknown", infra_error=False, missing=True,
                           missing_reason="declared, no result (counts 0)", in_declared=True, superseded=False,
                           synthetic=False)
                added.append(row)
    return added


def finalize_rows(rows: list[dict]) -> None:
    for r in rows:
        q = r["q_final"]
        r["q_budget2026"] = None if q is None else (0.0 if r["budget_ok"] is False else q)
        r["included"] = bool(r["in_declared"] and not r["superseded"] and not r["infra_error"])


def build_configs(runs: Sequence[RunData]) -> dict:
    """config_key -> {label, names, hashes, seeds, runs, order}; labels disambiguated by key prefix."""
    configs: dict[str, dict] = {}
    for run in runs:
        c = configs.setdefault(run.config_key, {"names": [], "hashes": set(), "seeds": set(), "runs": [], "order": run.order})
        if run.config_name not in c["names"]:
            c["names"].append(run.config_name)
        if run.config_hash:
            c["hashes"].add(run.config_hash)
        c["seeds"].add(run.seed)
        c["runs"].append(run.run_id)
    name_count = collections.Counter(c["names"][0] for c in configs.values())
    for key, c in configs.items():
        name = c["names"][0]
        c["label"] = name if name_count[name] == 1 else f"{name}@{key[:8]}"
    return configs


# ------------------------------------------------------------------------------------ statistics


def _finite_mean(xs: Iterable[Any]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def task_values(rows: Iterable[dict], metric: str) -> dict[str, list[float]]:
    out = collections.defaultdict(list)
    for r in rows:
        v = r[metric]
        if v is not None:
            out[r["task"]].append(float(v))
    return dict(out)


def task_mean(by_task: dict[str, list[float]]) -> Optional[float]:
    means = [sum(v) / len(v) for v in by_task.values() if v]
    return sum(means) / len(means) if means else None


def stratified_bootstrap(by_task: dict[str, list[float]], n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Replicates of the task-averaged mean, resampling units with replacement within each task."""
    tasks = sorted(t for t, v in by_task.items() if v)
    if not tasks or n_boot <= 0:
        return np.empty(0)
    acc = np.zeros(n_boot)
    for t in tasks:
        v = np.asarray(by_task[t], dtype=np.float64)
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        acc += v[idx].mean(axis=1)
    return acc / len(tasks)


def ci_of(samples: np.ndarray, level: float) -> Optional[list[float]]:
    if samples.size == 0:
        return None
    lo, hi = np.percentile(samples, [50 * (1 - level), 50 * (1 + level)])
    return [float(lo), float(hi)]


def _rng(seed: int, label: str) -> np.random.Generator:
    return np.random.default_rng([int(seed), zlib.crc32(label.encode("utf-8"))])


def per_task_table(rows: list[dict], configs: dict) -> list[dict]:
    out = []
    for key, c in sorted(configs.items(), key=lambda kv: kv[1]["order"]):
        mine = [r for r in rows if r["config_key"] == key and r["in_declared"] and not r["superseded"]]
        for task in sorted({r["task"] for r in mine}):
            t = [r for r in mine if r["task"] == task]
            inc = [r for r in t if r["included"]]
            ev = [r for r in inc if not r["missing"]]
            rec = {"config": c["label"], "task": task, "n_included": len(inc), "n_evaluated": len(ev),
                   "n_missing": sum(r["missing"] for r in t), "n_infra": sum(r["infra_error"] for r in t),
                   "mean_q": _finite_mean(r["q_final"] for r in inc),
                   "mean_q_budget2026": _finite_mean(r["q_budget2026"] for r in inc),
                   "success_rate": _finite_mean(float(bool(r["success"])) for r in inc),
                   "n_budget_violations": sum(r["budget_ok"] is False for r in ev),
                   "n_budget_unknown": sum(r["budget_ok"] is None for r in ev)}
            for m in SECONDARY:
                rec[f"mean_{m}"] = _finite_mean(r[m] for r in ev)
            out.append(rec)
    return out


def per_config_summary(rows: list[dict], configs: dict, declared: Optional[dict], n_boot: int, seed: int,
                       level: float) -> list[dict]:
    out = []
    for key, c in sorted(configs.items(), key=lambda kv: kv[1]["order"]):
        mine = [r for r in rows if r["config_key"] == key]
        scoped = [r for r in mine if r["in_declared"] and not r["superseded"]]
        inc = [r for r in scoped if r["included"]]
        ev = [r for r in inc if not r["missing"]]
        infra = [r for r in scoped if r["infra_error"]]
        q_by_task = task_values(inc, "q_final")
        qb_by_task = task_values(inc, "q_budget2026")
        tasks = sorted(q_by_task)
        all_tasks = sorted({r["task"] for r in scoped})
        no_valid = [t for t in all_tasks if t not in q_by_task]
        strict = task_values([dict(r, q_final=0.0) if r["infra_error"] else r for r in scoped], "q_final")
        if declared:
            scope = f"declared list ({len(declared['tasks'])} tasks)"
            if no_valid:
                scope += f"; {len(no_valid)} task(s) with only infra errors left out"
        else:
            scope = f"SUBSET mean over {len(tasks)} evaluated task(s); no declared instance list"
        rec = {
            "config": c["label"], "config_key": key, "config_names": c["names"], "config_hashes": sorted(c["hashes"]),
            "runs": c["runs"], "seeds": sorted((s for s in c["seeds"]), key=str), "scope": scope, "n_tasks": len(tasks),
            "tasks": tasks, "tasks_without_valid_episodes": no_valid, "n_included": len(inc), "n_evaluated": len(ev),
            "n_missing": sum(r["missing"] for r in scoped), "n_infra": len(infra),
            "n_superseded": sum(r["superseded"] for r in mine),
            "n_outside_declared": sum(not r["in_declared"] for r in mine),
            "mean_q": task_mean(q_by_task),
            "mean_q_ci": ci_of(stratified_bootstrap(q_by_task, n_boot, _rng(seed, c["label"] + "|q")), level),
            "mean_q_budget2026": task_mean(qb_by_task),
            "mean_q_budget2026_ci": ci_of(stratified_bootstrap(qb_by_task, n_boot, _rng(seed, c["label"] + "|qb")), level),
            "mean_q_infra_as_zero": task_mean(strict),
            "success_rate": task_mean(task_values([dict(r, success=float(bool(r["success"]))) for r in inc], "success")),
            "n_budget_violations": sum(r["budget_ok"] is False for r in ev),
            "n_budget_unknown": sum(r["budget_ok"] is None for r in ev),
            "total_cost_usd": sum(float(r["cost_usd"] or 0.0) for r in mine if not r["superseded"]),
            "total_provider_calls": sum(int(r["provider_calls"] or 0) for r in mine if not r["superseded"]),
            "synthetic_inputs": any(r["synthetic"] for r in mine),
        }
        for m in SECONDARY:
            rec[f"mean_{m}"] = task_mean(task_values(ev, m))
        out.append(rec)
    return out


def paired_comparison(rows: list[dict], a: str, b: str, metric: str, n_boot: int, seed: int, level: float) -> dict:
    """B minus A on the (task, instance, seed, rollout) units evaluated (not missing, not infra) for both."""
    def units(label):
        return {(r["task"], r["instance_id"], str(r["seed"]), r["rollout_id"]): r[metric]
                for r in rows if r["config"] == label and r["included"] and not r["missing"] and r[metric] is not None}

    ua, ub = units(a), units(b)
    common = sorted(set(ua) & set(ub), key=str)
    diffs, va, vb = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for k in common:
        diffs[k[0]].append(float(ub[k]) - float(ua[k]))
        va[k[0]].append(float(ua[k]))
        vb[k[0]].append(float(ub[k]))
    samples = stratified_bootstrap(dict(diffs), n_boot, _rng(seed, f"{a}|{b}|{metric}"))
    return {"a": a, "b": b, "metric": metric, "n_pairs": len(common), "n_tasks": len(diffs),
            "tasks": sorted(diffs), "mean_a": task_mean(dict(va)), "mean_b": task_mean(dict(vb)),
            "diff_b_minus_a": task_mean(dict(diffs)), "diff_ci": ci_of(samples, level),
            "only_a": len(set(ua) - set(ub)), "only_b": len(set(ub) - set(ua))}


# ------------------------------------------------------------------------------------ outputs


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, float):
        return f"{x:.{nd}f}" if math.isfinite(x) else str(x)
    if isinstance(x, (list, tuple)):
        if len(x) == 2 and all(isinstance(v, float) for v in x):
            return f"[{x[0]:.{nd}f}, {x[1]:.{nd}f}]"
        return ", ".join(str(v) for v in x)
    return str(x)


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(_fmt(c).replace("|", "/") for c in r) + " |" for r in rows]
    return "\n".join(lines)


def write_csv(path: str, records: list[dict], fields: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for rec in records:
            w.writerow({k: (round(v, 6) if isinstance(v, float) and math.isfinite(v) else
                            ("true" if v is True else "false" if v is False else
                             json.dumps(v) if isinstance(v, (list, dict)) else v)) for k, v in rec.items()})


def render_markdown(result: dict) -> str:
    prov = result["provenance"]
    lvl = int(round(result["bootstrap"]["ci_level"] * 100))
    out = [f"# Aggregated results",
           "",
           f"Generated by `eval/aggregate.py` at {prov['generated_at']} from {len(prov['inputs'])} input files "
           f"(code {(prov['code'] or {}).get('commit') or 'unknown'}{' dirty' if (prov['code'] or {}).get('dirty') else ''}). "
           "Do not edit by hand; re-run the aggregator.",
           ""]
    if result["synthetic_inputs"]:
        out += ["**WARNING: SYNTHETIC inputs present. These numbers are not results.**", ""]
    out += ["## Per-config summary", "",
            "Mean Q = mean over tasks of the per-task mean of official `q_score.final`. Infra errors are excluded "
            "from Q means and listed below. `Q (2026 budget)` sets Q = 0 when budget_ok is false (unknown is not "
            f"zeroed; see the unknown count). CIs: task-stratified bootstrap, {lvl}%, "
            f"B = {result['bootstrap']['n_boot']}, seed = {result['bootstrap']['seed']}.", ""]
    out.append(md_table(
        ["config", "scope", "tasks", "episodes (incl. missing)", "missing (=0)", "infra (excl.)", "mean Q", f"{lvl}% CI",
         "Q (2026 budget)", f"{lvl}% CI", "Q infra=0", "success", "budget viol.", "budget unknown", "time score",
         "steps", "provider calls/ep", "cost/ep USD", "recoveries/ep"],
        [[c["config"], c["scope"], c["n_tasks"], c["n_included"], c["n_missing"], c["n_infra"], c["mean_q"],
          c["mean_q_ci"], c["mean_q_budget2026"], c["mean_q_budget2026_ci"], c["mean_q_infra_as_zero"],
          c["success_rate"], c["n_budget_violations"], c["n_budget_unknown"], c["mean_time_score"], c["mean_steps"],
          c["mean_provider_calls"], c["mean_cost_usd"], c["mean_recoveries"]] for c in result["configs"]]))
    out += ["", "## Per task", ""]
    out.append(md_table(
        ["config", "task", "episodes (incl. missing)", "missing", "infra", "mean Q", "Q (2026 budget)", "success", "budget viol.",
         "steps", "time score", "dist base", "provider calls"],
        [[t["config"], t["task"], t["n_included"], t["n_missing"], t["n_infra"], t["mean_q"], t["mean_q_budget2026"],
          t["success_rate"], t["n_budget_violations"], t["mean_steps"], t["mean_time_score"], t["mean_norm_dist_base"],
          t["mean_provider_calls"]] for t in result["per_task"]]))
    if result["paired"]:
        out += ["", "## Paired comparisons (B - A on task/instance/seed/rollout units evaluated in both)", ""]
        out.append(md_table(
            ["A", "B", "metric", "pairs", "tasks", "mean A", "mean B", "B - A", f"{lvl}% CI", "only A", "only B"],
            [[p["a"], p["b"], p["metric"], p["n_pairs"], p["n_tasks"], p["mean_a"], p["mean_b"], p["diff_b_minus_a"],
              p["diff_ci"], p["only_a"], p["only_b"]] for p in result["paired"]]))
    out += ["", "## Infra errors (excluded from Q means)", ""]
    infra = result["infra_errors"]
    out.append(md_table(["config", "task", "instance", "seed", "rollout", "reason", "superseded", "run"],
                        [[r["config"], r["task"], r["instance_id"], r["seed"], r["rollout_id"], r["infra_reason"],
                          r["superseded"], r["run_id"]] for r in infra]) if infra else "none")
    out += ["", "## Missing results (count as Q = 0)", ""]
    miss = result["missing"]
    out.append(md_table(["config", "task", "instance", "seed", "reason"],
                        [[r["config"], r["task"], r["instance_id"], r["seed"], r["missing_reason"]] for r in miss])
               if miss else "none")
    if result["warnings"]:
        out += ["", "## Warnings", ""] + [f"- {w}" for w in result["warnings"]]
    return "\n".join(out) + "\n"


def _guard_outputs(out_dir: str, runs: Sequence[RunData], inputs: InputLog, names: Iterable[str]) -> None:
    out_real = os.path.realpath(out_dir)
    for run in runs:
        official = os.path.realpath(run.eval_output_dir)
        if out_real == official or out_real.startswith(official + os.sep):
            raise SystemExit(f"refusing to write inside the official output dir {official}")
    taken = {os.path.realpath(p) for p in inputs.files}
    for n in names:
        if os.path.realpath(os.path.join(out_dir, n)) in taken:
            raise SystemExit(f"refusing to overwrite input file {os.path.join(out_dir, n)}")


OUTPUT_FILES = ("episodes.csv", "per_task.csv", "per_config.csv", "paired.csv", "summary.md", "summary.json")


def aggregate(run_dirs: Sequence[str], out_dir: Optional[str] = None, declared_path: Optional[str] = None,
              n_boot: int = 2000, seed: int = 0, ci_level: float = 0.95, compare: Sequence[Sequence[str]] = (),
              infra_reasons: Iterable[str] = DEFAULT_INFRA_FAILURE_REASONS, argv: Optional[list] = None) -> dict:
    inputs = InputLog()
    runs = [load_run(d, inputs, order=i) for i, d in enumerate(run_dirs)]
    warnings: list[str] = []
    rows: list[dict] = []
    for run in runs:
        r, w = episode_rows(run, infra_reasons)
        rows += r
        warnings += w
    declared = load_declared(declared_path, inputs) if declared_path else None
    configs = build_configs(runs)
    for r in rows:
        r["config"] = configs[r["config_key"]]["label"]
    warnings += resolve_attempts(rows)
    rows += apply_declared(rows, declared, configs)
    finalize_rows(rows)

    labels = [c["label"] for c in sorted(configs.values(), key=lambda c: c["order"])]
    pairs = [tuple(p) for p in compare] or list(zip(labels, labels[1:]))
    for a, b in pairs:
        for x in (a, b):
            if x not in labels:
                raise SystemExit(f"--compare: unknown config {x!r} (known: {labels})")
    summary = per_config_summary(rows, configs, declared, n_boot, seed, ci_level)
    task_sets = {tuple(c["tasks"]) for c in summary}
    if len(task_sets) > 1:
        warnings.append("configs cover different task sets: compare their means only through the paired table")
    synthetic = any(r["synthetic"] for r in rows)
    if synthetic:
        warnings.append("SYNTHETIC inputs present: these numbers are not results")
    result = {
        "provenance": {
            "generator": "eval/aggregate.py",
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "code": git_info(REPO_ROOT),
            "argv": list(argv) if argv is not None else None,
            "run_dirs": [r.run_dir for r in runs],
            "declared": declared,
            "inputs": inputs.listing(),
        },
        "bootstrap": {"n_boot": n_boot, "seed": seed, "ci_level": ci_level, "method": "task-stratified percentile"},
        "budget_rule": {"seconds_per_step": SECONDS_PER_STEP, "budget_failure_reasons": sorted(BUDGET_FAILURE_REASONS)},
        "infra_failure_reasons": sorted(set(infra_reasons)),
        "configs": summary,
        "per_task": per_task_table(rows, configs),
        "paired": [paired_comparison(rows, a, b, m, n_boot, seed, ci_level) for a, b in pairs
                   for m in ("q_final", "q_budget2026")],
        "infra_errors": [r for r in rows if r["infra_error"]],
        "missing": [r for r in rows if r["missing"]],
        "synthetic_inputs": synthetic,
        "warnings": warnings,
        "rows": rows,
    }
    if out_dir:
        _guard_outputs(out_dir, runs, inputs, OUTPUT_FILES)
        os.makedirs(out_dir, exist_ok=True)
        write_csv(os.path.join(out_dir, "episodes.csv"), rows, ROW_FIELDS)
        write_csv(os.path.join(out_dir, "per_task.csv"), result["per_task"],
                  list(result["per_task"][0]) if result["per_task"] else ["config", "task"])
        write_csv(os.path.join(out_dir, "per_config.csv"), summary, list(summary[0]) if summary else ["config"])
        write_csv(os.path.join(out_dir, "paired.csv"), result["paired"],
                  list(result["paired"][0]) if result["paired"] else ["a", "b"])
        with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
            f.write(render_markdown(result))
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in result.items() if k != "rows"}, f, indent=2, default=str)
    return result


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="run directories, in chronological order (re-run rule 6.5)")
    ap.add_argument("--out", required=True, help="output directory (never inside an official eval output dir)")
    ap.add_argument("--instances-declared", default=None, help="YAML/JSON declared task -> instance list")
    ap.add_argument("--bootstrap", type=int, default=2000, help="bootstrap replicates (0 = no CI)")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    ap.add_argument("--ci", type=float, default=0.95, help="CI level")
    ap.add_argument("--compare", nargs=2, action="append", default=[], metavar=("A", "B"),
                    help="paired comparison B - A (config labels); default: consecutive configs in input order")
    ap.add_argument("--infra-failure-reasons", default=",".join(DEFAULT_INFRA_FAILURE_REASONS),
                    help="official failure_reason values counted as infra errors (comma list, may be empty)")
    args = ap.parse_args(argv)
    reasons = [x for x in args.infra_failure_reasons.split(",") if x]
    result = aggregate(args.run_dirs, args.out, args.instances_declared, args.bootstrap, args.seed, args.ci,
                       args.compare, reasons, argv=sys.argv if argv is None else argv)
    for c in result["configs"]:
        print(f"{c['config']}: mean Q {_fmt(c['mean_q'])} CI {_fmt(c['mean_q_ci'])} over {c['n_tasks']} task(s), "
              f"{c['n_included']} episode(s), {c['n_missing']} missing, {c['n_infra']} infra  [{c['scope']}]")
    for w in result["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    print(f"wrote {', '.join(OUTPUT_FILES)} to {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
