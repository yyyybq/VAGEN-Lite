"""C0 step 300 on the screened test set (2026-10-29): check the rerun on the 30 control tasks against the 2026-10-19
results, then write each checkpoint's 292 results (233 kept tasks from 10-19, 59 added tasks from today) into
eval_test_v2_screened/<name>/ with the provenance, and print the success rates."""
import collections
import json
import shutil
from pathlib import Path

ROOT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data")
R12 = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r12_20261019/eval_test_v2")
SCREENED = ROOT / "exps/matrix_inputs_20261029/eval_test_v2_screened"
TOPUP = ROOT / "exps/matrix_inputs_20261029/eval_test_v2_screened_c0_topup"
L = lambda p: [json.loads(l) for l in open(p) if l.strip()]  # noqa: E731
sel = json.loads((TOPUP / "selection.json").read_text())
new_ids, control_ids = set(sel["new_task_ids"]), sel["control_task_ids"]
tasks = L(SCREENED / "tasks.jsonl")
fam = {t["task_id"]: t["task_type"] for t in tasks}
SAME = ("success", "actions_to_success", "turns", "actions", "collisions", "invalid_turns")
pooled = collections.Counter()
summary = {}
for old_name, name in (("p1s0_300", "c0_p1s0_300"), ("p1d_300", "c0_p1d_300")):
    old = {r["task_id"]: r for r in L(R12 / old_name / "results.jsonl")}
    new = {r["task_id"]: r for r in L(TOPUP / name / "results.jsonl")}
    assert set(new) == new_ids | set(control_ids), (name, len(new))
    agree = [t for t in control_ids if all(old[t].get(k) == new[t].get(k) for k in SAME)]
    succ_agree = sum(old[t]["success"] == new[t]["success"] for t in control_ids)
    traj_same = sum(json.dumps(old[t].get("trajectory"), sort_keys=True) == json.dumps(new[t].get("trajectory"), sort_keys=True)
                    for t in control_ids if "trajectory" in old[t])
    print(f"\n{name}: control {len(control_ids)}: success agrees {succ_agree}, all of {SAME} agree {len(agree)}, "
          f"trajectory identical {traj_same}")
    for t in control_ids:
        if t not in agree:
            print("   differs:", t, {k: (old[t].get(k), new[t].get(k)) for k in SAME if old[t].get(k) != new[t].get(k)})
    rows = [old[t["task_id"]] if t["task_id"] not in new_ids else new[t["task_id"]] for t in tasks]
    out = SCREENED / name
    out.mkdir(exist_ok=True)
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    shutil.copy(TOPUP / name / "model.json", out / "model.json")
    combined = {t["task_id"]: (new if t["task_id"] in new_ids else old)[t["task_id"]] for t in tasks}  # as written
    rate = lambda ids: 100 * sum(bool(combined[t]["success"]) for t in ids) / len(ids)  # noqa: E731
    kept = [t["task_id"] for t in tasks if t["task_id"] not in new_ids]
    allids = [t["task_id"] for t in tasks]
    (out / "source.json").write_text(json.dumps({
        "made": "2026-10-29", "tasks": len(rows), "from_2026_10_19": {"folder": str(R12 / old_name), "tasks": len(kept)},
        "from_2026_10_29": {"folder": str(TOPUP / name), "tasks": len(new_ids), "model": json.loads((TOPUP / name / "model.json").read_text())["model"]},
        "rerun_check": {"control_tasks": len(control_ids), "success_agrees": succ_agree, "all_fields_agree": len(agree),
                        "trajectory_identical": traj_same}}, indent=1))
    by = collections.defaultdict(list)
    for t in allids:
        by[fam[t]].append(t)
    print(f"  screened test 292: {rate(allids):.1f}%  (kept 233: {rate(kept):.1f}%, added 59: {rate(sorted(new_ids)):.1f}%); "
          + ", ".join(f"{f[:6]} {rate(v):.0f}%/{len(v)}" for f, v in sorted(by.items())))
    pooled["ok"] += sum(bool(combined[t]["success"]) for t in allids)
    pooled["n"] += len(allids)
    summary[name] = {"control": {"tasks": len(control_ids), "success_agrees": succ_agree, "all_fields_agree": len(agree),
                                 "trajectory_identical": traj_same},
                     "screened_292": round(rate(allids), 1), "kept_233": round(rate(kept), 1), "added_59": round(rate(sorted(new_ids)), 1),
                     "full_v2_292": round(100 * sum(bool(r["success"]) for r in old.values()) / len(old), 1),
                     "by_family": {f: [round(rate(v), 1), len(v)] for f, v in sorted(by.items())}}
summary["pooled_292"] = round(100 * pooled["ok"] / pooled["n"], 1)
(SCREENED / "c0_summary.json").write_text(json.dumps(summary, indent=1))
print(f"\npooled S0 + D on the screened test set: {100 * pooled['ok'] / pooled['n']:.1f}% of {pooled['n']}")
