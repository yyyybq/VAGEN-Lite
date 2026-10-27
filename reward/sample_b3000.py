"""Reward line, step 2a: the 3,000 training tasks of B and 300 backups, drawn from the reserve training pool.

The pool is what the RL configuration trains on: the rows file's reserve training rows minus the depth check's
flagged tasks (training_prep's rule; 10,798 tasks, checked against q3d's train.jsonl).  Strata are (family, d*).
Each stratum gets round(n * N_s / N) tasks by largest remainder (so the counts add up to n exactly and the shares
match the pool's), and likewise round(backup * N_s / N) backups from the stratum's remaining tasks.  Within a stratum
the task ids are sorted and shuffled by random.Random(f"{seed}:{family}:{d_star}"): the first n_s are the tasks, the
next b_s the backups (in that order, so a stratum's replacements are taken first to last).

Writes OUT/b3000.txt (sorted ids), OUT/b3000_backup300.txt (stratum by stratum, in replacement order),
OUT/b3000_sample.json (seed, rule, pool, every stratum's N_s / n_s / b_s, sha256 of both lists) and OUT/calib10.txt
(the ten tasks of the 2a check: every family, d* from the smallest to the largest, preferring tasks that q3s0 / q3d
failed with at least min(12, d* + 4) primitive actions, i.e. likely beyond the old table's depth).

    python reward/sample_b3000.py --rows ROWS --exclude DEPTH_CHECK.json --q3d-train TRAIN.jsonl \
        --records RUN_DIR [RUN_DIR ...] --out reward_runs/tasks [--seed 20261027]
"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CALIB = [("projective_relations", 1), ("fov_inclusion", 2), ("equidistance", 3), ("screen_occupancy", 4),
         ("occlusion_alignment", 5), ("absolute_positioning", 6), ("projective_relations", 8), ("equidistance", 7),
         ("fov_inclusion", 6), ("absolute_positioning", 8)]


def largest_remainder(total, weights):
    """Integer counts proportional to ``weights`` adding up to ``total`` (ties by key, so the result is fixed)."""
    s = sum(weights.values())
    raw = {k: total * w / s for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    left = total - sum(out.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), str(k)))[:left]:
        out[k] += 1
    return out


def far_failures(record_dirs, wanted):
    """{task_id: failed episodes with at least min(12, d* + 4) primitive actions} over the runs' training records."""
    out = Counter()
    for run in record_dirs:
        for f in sorted(Path(run, "rollout_data").glob("*.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                tid = r.get("task_id")
                if tid in wanted and not float(r.get("traj_success") or 0) and \
                        int(r.get("n_primitive_steps") or 0) >= min(12, wanted[tid] + 4):
                    out[tid] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--exclude", required=True)
    ap.add_argument("--q3d-train", required=True)
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--backup", type=int, default=300)
    args = ap.parse_args()

    flagged = set(json.loads(Path(args.exclude).read_text())["flagged_task_ids"])
    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    pool = [r for r in rows if r.get("reserve") and r.get("split") == "train" and r["task_id"] not in flagged]
    q3d = {json.loads(l)["task_id"] for l in Path(args.q3d_train).read_text().splitlines() if l.strip()}
    assert {r["task_id"] for r in pool} == q3d, "the pool differs from q3d's training tasks"
    strata = defaultdict(list)
    for r in pool:
        strata[(r["task_type"], int(r["difficulty"]["d_star"]))].append(r["task_id"])
    sizes = {k: len(v) for k, v in strata.items()}
    n_s = largest_remainder(args.n, sizes)
    b_s = largest_remainder(args.backup, sizes)
    chosen, backup, table = [], [], []
    for key in sorted(strata):
        ids = sorted(strata[key])
        random.Random(f"{args.seed}:{key[0]}:{key[1]}").shuffle(ids)
        assert n_s[key] + b_s[key] <= len(ids), key
        chosen += ids[:n_s[key]]
        backup += ids[n_s[key]:n_s[key] + b_s[key]]
        table.append({"family": key[0], "d_star": key[1], "pool": sizes[key], "tasks": n_s[key], "backups": b_s[key]})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "b3000.txt").write_text("".join(t + "\n" for t in sorted(chosen)))
    (out / "b3000_backup300.txt").write_text("".join(t + "\n" for t in backup))

    d_of = {r["task_id"]: int(r["difficulty"]["d_star"]) for r in pool}
    fam_of = {r["task_id"]: r["task_type"] for r in pool}
    far = far_failures(args.records, {t: d_of[t] for t in chosen})
    calib, used = [], set()
    for fam, d in CALIB:
        cands = [t for t in chosen if fam_of[t] == fam and t not in used]
        cands.sort(key=lambda t: (abs(d_of[t] - d), -far[t], t))
        pick = cands[0]
        calib.append({"task_id": pick, "family": fam, "d_star": d_of[pick], "wanted_d_star": d,
                      "far_failures_in_q3_records": far[pick]})
        used.add(pick)
    (out / "calib10.txt").write_text("".join(c["task_id"] + "\n" for c in calib))

    def sha(p):
        return hashlib.sha256((out / p).read_bytes()).hexdigest()

    fam_pool, fam_chosen = Counter(fam_of.values()), Counter(fam_of[t] for t in chosen)
    d_pool, d_chosen = Counter(d_of.values()), Counter(d_of[t] for t in chosen)
    summary = {"seed": args.seed, "rule": "stratified by (family, d*), largest-remainder allocation, "
               "random.Random(f'{seed}:{family}:{d_star}').shuffle(sorted ids) per stratum",
               "pool": len(pool), "pool_source": {"rows": args.rows, "exclude": args.exclude, "flagged": len(flagged),
                                                  "equals_q3d_train": True},
               "tasks": len(chosen), "backups": len(backup),
               "sha256": {"b3000.txt": sha("b3000.txt"), "b3000_backup300.txt": sha("b3000_backup300.txt"),
                          "calib10.txt": sha("calib10.txt")},
               "by_family": {f: {"pool": fam_pool[f], "tasks": fam_chosen[f],
                                 "pool_share": round(fam_pool[f] / len(pool), 4),
                                 "tasks_share": round(fam_chosen[f] / len(chosen), 4)} for f in sorted(fam_pool)},
               "by_d_star": {str(d): {"pool": d_pool[d], "tasks": d_chosen[d],
                                      "pool_share": round(d_pool[d] / len(pool), 4),
                                      "tasks_share": round(d_chosen[d] / len(chosen), 4)} for d in sorted(d_pool)},
               "strata": table, "calib10": calib}
    (out / "b3000_sample.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("seed", "pool", "tasks", "backups", "sha256", "by_family", "by_d_star",
                                              "calib10")}, indent=1))


if __name__ == "__main__":
    main()
