"""Blind floor F (G10(f), CONTRACT_wave2 5.1.5, 6.8): kin_env only, no images, no model, frozen per-task sets.

  (i)   STOP0  Stop as the first reply; k on the set (0 expected: no task starts at its target).
  (ii)  FIX    for every a in NON_STOP and k in 0..39 the policy "a repeated k times, then Stop"; on a task with
               cap c, a policy with k + 1 > c never stops (failure).  Every (a, k) is scored on sft1600; the max
               sft successes is chosen, ties -> smaller k, then ACTIONS order; the chosen one is scored on the set.
  (iii) RAND   per seed s in (0, 1, 2) and task: rng = random.Random(f"tvr-floor-rand-v1|{s}|{task_id}");
               m = expert_medians[category] (median_low of the public non-Stop lengths); for j in range(m):
               nonstop = [v for v in valid if v != "Stop"]; a = nonstop[rng.randrange(len(nonstop))]; step; then
               Stop (m + 1 > cap never stops).  rate_s = k_s / n; mean_rate = (k_0 + k_1 + k_2) / (3 n).
  F = max(stop0 rate, fixed rate on the set, random mean_rate); ties -> the first of (stop0, fixed, random);
  k_equiv = ceil(F * n) computed exactly (fractions), n = the set's size (500 for eval500).

Floors on dev200 are context for the calibration only.  The output has no timestamp, and an existing file with the
same content is kept byte-for-byte (its sha256 goes into pilot_cfg data.floors_eval500_sha256).

CLI (PY_Q3, 1 core):
  python -B -m vpd.tvr.floors --task-sets eval500,dev200 --reach OUT/tvr/reach/v1 --out OUT/tvr/floors
  -> floors_eval500.json, floors_dev200.json
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from vpd.tvr import kin_env, pilot_cfg, tasks as T

SCHEMA = "vpd.tvr.floors/1"
FL = pilot_cfg.FROZEN["floors"]
K_MAX = FL["fixed_k_max"]
SEEDS = tuple(FL["random_seeds"])
SALT = pilot_cfg.FROZEN["salts"]["rand_floor"]
SOURCES = ("stop0", "fixed", "random")
RULE = ("F = max(stop0 rate, fixed (chosen on sft1600: max successes, ties smaller k then ACTIONS order) rate, "
        "random mean_rate = (k0+k1+k2)/(3n)); ties -> first of stop0, fixed, random; k_equiv = ceil(F*n) exact; "
        "a policy needing more than cap replies never stops")


def _poses(t: Mapping) -> Tuple[kin_env.Pose, kin_env.Pose]:
    return kin_env.pose_of(t["start"]), kin_env.pose_of(t["target"])


def stop0(task_list: Sequence[Mapping]) -> Dict:
    k = sum(1 for t in task_list if kin_env.is_exact(*_poses(t)))
    n = len(task_list)
    return {"k": k, "n": n, "rate": k / n if n else 0.0}


def fixed_hits(task: Mapping, reach, k_max: int = K_MAX) -> Dict[str, List[int]]:
    """{a: [k ...]}: the k in 0..k_max for which "a^k then Stop" succeeds on the task (cap rule applied)."""
    s, g = _poses(task)
    cap = pilot_cfg.cap_of(task)
    out: Dict[str, List[int]] = {}
    for a in pilot_cfg.NON_STOP:
        p, ks = s, []
        for k in range(k_max + 1):
            if k + 1 <= cap and kin_env.is_exact(p, g):
                ks.append(k)
            p = kin_env.step(p, a, reach)[0]
        out[a] = ks
    return out


def fixed_table(task_list: Sequence[Mapping], reach_index, k_max: int = K_MAX) -> Dict[str, int]:
    """{"<a>:<k>": successes} over the tasks for every a in NON_STOP and k in 0..k_max."""
    tab = {f"{a}:{k}": 0 for a in pilot_cfg.NON_STOP for k in range(k_max + 1)}
    for t in task_list:
        for a, ks in fixed_hits(t, reach_index.reach(t["task_id"]), k_max).items():
            for k in ks:
                tab[f"{a}:{k}"] += 1
    return tab


def choose_fixed(table: Mapping[str, int], k_max: int = K_MAX) -> Tuple[str, int, int]:
    """(action, k, successes): max successes; ties -> smaller k, then ACTIONS order."""
    best: Optional[Tuple[str, int, int]] = None
    for k in range(k_max + 1):
        for a in pilot_cfg.NON_STOP:
            v = table[f"{a}:{k}"]
            if best is None or v > best[2]:
                best = (a, k, v)
    assert best is not None
    return best


def fixed_score(task_list: Sequence[Mapping], reach_index, action: str, k: int) -> int:
    n = 0
    for t in task_list:
        if k + 1 > pilot_cfg.cap_of(t):
            continue
        s, g = _poses(t)
        r = reach_index.reach(t["task_id"])
        p = s
        for _ in range(k):
            p = kin_env.step(p, action, r)[0]
        n += kin_env.is_exact(p, g)
    return n


def random_success(task: Mapping, reach, m: int, seed: int) -> bool:
    rng = random.Random(f"{SALT}{seed}|{task['task_id']}")
    s, g = _poses(task)
    if m + 1 > pilot_cfg.cap_of(task):
        return False
    p = s
    for _ in range(m):
        nonstop = [v for v in kin_env.valid_actions(p, reach, include_stop=True) if v != "Stop"]
        a = nonstop[rng.randrange(len(nonstop))]
        p = kin_env.step(p, a, reach)[0]
    return kin_env.is_exact(p, g)


def random_floor(task_list: Sequence[Mapping], reach_index, medians: Mapping[str, int],
                 seeds: Sequence[int] = SEEDS) -> Dict:
    n = len(task_list)
    ks = []
    for s in seeds:
        ks.append(sum(random_success(t, reach_index.reach(t["task_id"]), medians[pilot_cfg.category_of(t)], s)
                      for t in task_list))
    return {"medians": dict(medians), "seeds": list(seeds), "k": ks, "rates": [k / n for k in ks],
            "mean_rate": sum(ks) / (len(seeds) * n), "_frac": Fraction(sum(ks), len(seeds) * n)}


def floors_doc(name: str, task_list: Sequence[Mapping], sft_tasks: Sequence[Mapping], reach_index,
               medians: Mapping[str, int], public_sha256: Optional[str] = None) -> Dict:
    n = len(task_list)
    s0 = stop0(task_list)
    table = fixed_table(sft_tasks, reach_index)
    a, k, k_sft = choose_fixed(table)
    k_ev = fixed_score(task_list, reach_index, a, k)
    rnd = random_floor(task_list, reach_index, medians)
    fr = {"stop0": Fraction(s0["k"], n), "fixed": Fraction(k_ev, n), "random": rnd.pop("_frac")}
    F = max(fr.values())
    source = next(s for s in SOURCES if fr[s] == F)
    warnings = []
    if s0["k"]:
        warnings.append(f"stop0: {s0['k']} tasks start at their target pose (0 expected)")
    not_in = {"start": [], "target": []}
    for t in task_list:
        s, g = _poses(t)
        r = reach_index.reach(t["task_id"])
        if (s.x, s.z) not in r:
            not_in["start"].append(t["task_id"])
        if (g.x, g.z) not in r:
            not_in["target"].append(t["task_id"])
    return {"schema": SCHEMA, "task_set": name, "n": n, "reach_index_sha256": reach_index.sha256,
            "stop0": s0,
            "fixed": {"chosen": {"action": a, "k": k}, "sft": {"k": k_sft, "n": len(sft_tasks)},
                      "eval": {"k": k_ev, "n": n, "rate": k_ev / n}, "table_sft": table},
            "random": rnd,
            "F": {"rate": float(F), "source": source, "k_equiv": int(math.ceil(F * n))},
            "warnings": warnings, "rule": RULE,
            "notes": {"start_not_in_reach": not_in["start"], "target_not_in_reach": not_in["target"]},
            "public_sha256": public_sha256, "code_sha256": T.code_sha256()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vpd.tvr.floors", description="TVR blind floor F (G10(f))")
    ap.add_argument("--task-sets", required=True)
    ap.add_argument("--reach", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--public", default=None, help="sft_train.json (default: pilot_cfg public path)")
    ap.add_argument("--tasks-dir", default=None)
    ap.add_argument("--tvr-out", default=None, help="root holding split/dev200/selection.json (default OUT/tvr)")
    a = ap.parse_args(argv)
    from vpd.tvr import reach as R

    try:
        idx = R.ReachIndex(a.reach)
        sft = T.task_set("sft1600", a.tvr_out, a.tasks_dir)
        pub_path = Path(a.public) if a.public else T.PUBLIC_JSON
        public = T.public_sequences(pub_path, [t["task_id"] for t in sft])
        medians = T.expert_medians(public, T.by_id(sft))
        pub_sha = T.sha256_file(pub_path)
        out = Path(a.out)
        for name in T.parse_sets(a.task_sets):
            doc = floors_doc(name, T.task_set(name, a.tvr_out, a.tasks_dir), sft, idx, medians, pub_sha)
            sha, kept = T.write_json_stable(out / f"floors_{name}.json", doc)
            print(f"[floors] {name}: F {doc['F']['rate']:.4f} ({doc['F']['source']}, k_equiv "
                  f"{doc['F']['k_equiv']}) stop0 {doc['stop0']['k']}/{doc['n']} fixed "
                  f"{doc['fixed']['chosen']} {doc['fixed']['eval']['k']}/{doc['n']} random k {doc['random']['k']} "
                  f"warnings {len(doc['warnings'])} sha256 {sha}{' (kept)' if kept else ''}")
    except (OSError, ValueError, KeyError) as e:
        print(f"[floors] error: {e!r}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
