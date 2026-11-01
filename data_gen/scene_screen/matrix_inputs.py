"""Matrix inputs without the 128 excluded scenes (user 2026-10-29: 1a, 2b with the test set topped up to 292, 3a),
in exps/matrix_inputs_20261029 (the main line's files are only read):

* rl_rows/rows.jsonl: the round-11 RL rows (interiorgs_v10_r11_20261016/rl_rows) without those scenes;
* eval_val_v2_screened: eval_val_v2 without those scenes (not topped up);
* eval_test_v2_screened: eval_test_v2 without those scenes, each removed task replaced by an unused, unflagged formal
  test task of a kept scene from the same family and d* band, taken in a seeded order and only if its replay in the
  env succeeds exactly at d* without a collision (the round-12 top-up rule, evaluation.top_up).  The occlusion family
  has no such task left, so its slots go to the other families in turn, same band.
"""
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data")
sys.path.insert(0, str(ROOT))
from data_gen.interiorgs_v10 import evaluation as EV  # noqa: E402
from data_gen.interiorgs_v10 import replay_env  # noqa: E402

V = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial")
OUT = ROOT / "exps/matrix_inputs_20261029"
RUN, MAN = V / "interiorgs_v10_gen_r10_20261015/output", V / "interiorgs_v10_gen_r10_20261015/split_manifest.json"
FLAG = V / "interiorgs_v10_r10_20261015/depth_check/depth_check_summary.json"
SCENES_ROOT = Path("/mnt/umm/users/yinbaiqiao/InteriorGS_meta")
TEMPLATE = ROOT / "data_gen/interiorgs_v10/configs/rl_env_v10_S0.yaml"
EXCL = ROOT / "configs/scene_screen/scene_exclusions_20261029.json"
SEED = 20261029
ex = set(json.loads(EXCL.read_text())["excluded"])
L = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]  # noqa: E731
W = lambda p, rows: p.write_text("".join(json.dumps(r) + "\n" for r in rows))  # noqa: E731
cell = lambda r: (r["task_type"], EV.band_name(int(r["difficulty"]["d_star"])))  # noqa: E731


def summary(rows):
    return {"tasks": len(rows), "scenes": len({r["scene_id"] for r in rows}),
            "by_family": dict(sorted(collections.Counter(r["task_type"] for r in rows).items())),
            "by_cell": {f"{f}|{b}": n for (f, b), n in sorted(collections.Counter(cell(r) for r in rows).items())}}


# ---------------------------------------------------------------- RL rows
src = V / "interiorgs_v10_r11_20261016/rl_rows/rows.jsonl"
rows = L(src)
kept = [r for r in rows if r["scene_id"] not in ex]
(OUT / "rl_rows").mkdir(parents=True, exist_ok=True)
W(OUT / "rl_rows/rows.jsonl", kept)
gone = [r for r in rows if r["scene_id"] in ex]
(OUT / "rl_rows/selection.json").write_text(json.dumps({
    "source": str(src), "exclusions": str(EXCL), "rows": len(rows), "kept": len(kept), "dropped": len(gone),
    "dropped_by_split": dict(collections.Counter(r.get("split", "?") for r in gone)),
    "dropped_reserve": sum(1 for r in gone if r.get("pool") == "reserve" or r.get("reserve")),
    "dropped_scenes": len({r["scene_id"] for r in gone})}, indent=1))
print("rl rows", len(rows), "->", len(kept), "dropped by split", dict(collections.Counter(r.get("split", "?") for r in gone)))

# ---------------------------------------------------------------- val
src = V / "interiorgs_v10_r12_20261019/eval_val_v2"
val = L(src / "tasks.jsonl")
vk = [r for r in val if r["scene_id"] not in ex]
d = OUT / "eval_val_v2_screened"
d.mkdir(parents=True, exist_ok=True)
W(d / "tasks.jsonl", vk)
(d / "selection.json").write_text(json.dumps({
    "version": EV.VERSION, "source": str(src), "exclusions": str(EXCL), "topped_up": False,
    "removed_task_ids": [r["task_id"] for r in val if r["scene_id"] in ex], "before": summary(val), "after": summary(vk),
    "task_ids": [r["task_id"] for r in vk]}, indent=1))
print("val", len(val), "->", len(vk))

# ---------------------------------------------------------------- test
src = V / "interiorgs_v10_r12_20261019/eval_test_v2"
test = L(src / "tasks.jsonl")
have = {r["task_id"] for r in test}
flagged = EV.flagged_ids(str(FLAG))
pool = [r for r in EV.split_rows(RUN, MAN, "test") if r["task_id"] not in have and r["task_id"] not in flagged
        and r["scene_id"] not in ex]
cand = collections.defaultdict(list)
for r in sorted(pool, key=lambda r: r["task_id"]):
    cand[cell(r)].append(r)
rng = random.Random(SEED)
for v in cand.values():
    rng.shuffle(v)
removed = sorted((r for r in test if r["scene_id"] in ex), key=lambda r: r["task_id"])
families = sorted({c[0] for c in cand})
want = collections.Counter()   # per cell: how many replacements
slot_of = {}                    # removed task -> the cell its replacement comes from
turn = 0
for r in removed:
    f, b = cell(r)
    if not cand.get((f, b)):    # no task of this family and band left (the occlusion family): another family, same band
        options = [g for g in families if g != f and cand.get((g, b))]
        f = options[turn % len(options)]
        turn += 1
    slot_of[r["task_id"]] = (f, b)
    want[(f, b)] += 1
rules = replay_env.template_rules(TEMPLATE)
taken, replayed = collections.defaultdict(list), []
for c, n in sorted(want.items()):
    i = 0
    while len(taken[c]) < n:
        batch = cand[c][i:i + 2 * (n - len(taken[c])) + 1]
        if not batch:
            raise SystemExit(f"cell {c}: only {len(taken[c])} of {n} replacements pass the replay")
        i += len(batch)
        res = {x["task_id"]: x for x in replay_env.replay(batch, SCENES_ROOT, rules)}
        for row in batch:
            rec = res.get(row["task_id"]) or {"task_id": row["task_id"], "ok": False, "error": "not replayed"}
            replayed.append({"task_id": row["task_id"], "cell": list(c), "ok": bool(rec.get("ok")),
                             "success_step": rec.get("success_step"), "collisions": rec.get("collisions"),
                             "error": rec.get("error")})
            if rec.get("ok") and len(taken[c]) < n:
                taken[c].append(row)
out_rows, swaps = [], []
for r in test:
    if r["scene_id"] not in ex:
        out_rows.append(r)
        continue
    c = slot_of[r["task_id"]]
    new = taken[c].pop(0)
    out_rows.append(new)
    swaps.append({"out": r["task_id"], "out_scene": r["scene_id"], "out_cell": list(cell(r)), "in": new["task_id"],
                  "in_scene": new["scene_id"], "in_cell": list(c)})
assert len(out_rows) == len(test) and len({r["task_id"] for r in out_rows}) == len(out_rows)
assert not any(r["scene_id"] in ex for r in out_rows)
d = OUT / "eval_test_v2_screened"
d.mkdir(parents=True, exist_ok=True)
W(d / "tasks.jsonl", out_rows)
(d / "selection.json").write_text(json.dumps({
    "version": EV.VERSION, "seed": SEED, "source": str(src), "exclusions": str(EXCL), "flagged": str(FLAG),
    "env_template": str(TEMPLATE), "pool": "formal test tasks of kept scenes, not in eval_test_v2, not flagged",
    "pool_size": len(pool), "removed": len(removed), "replaced": len(swaps),
    "occlusion_slots_to_other_families": sum(1 for s in swaps if s["out_cell"][0] != s["in_cell"][0]),
    "before": summary(test), "after": summary(out_rows), "swaps": swaps, "replayed": len(replayed),
    "replay_failed": [x for x in replayed if not x["ok"]], "task_ids": [r["task_id"] for r in out_rows]}, indent=1))
print("test", len(test), "->", len(out_rows), "; replaced", len(swaps), "; replayed", len(replayed), "failed",
      sum(1 for x in replayed if not x["ok"]))
print("families before", summary(test)["by_family"])
print("families after ", summary(out_rows)["by_family"])
print("scenes", summary(test)["scenes"], "->", summary(out_rows)["scenes"])
