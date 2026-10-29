"""The user's scene-screen choices (2026-10-29) applied to the presets: the final exclusion list, how it departs from
the recommendation, and what it removes from each data set.  Reads the screen's scene_types.json and the page's
decision documents; writes OUT/scene_exclusions.json."""
import collections
import json
import sys
from pathlib import Path

S = Path(sys.argv[1])               # exps/scene_screen_20261028
DEC = Path(sys.argv[2])             # the page's decisions, one JSON per scene
BANK = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data/exps/transition_qa_20261019b")
d = json.loads((S / "scene_types.json").read_text())
sc, N = d["scenes"], d["names"]
PRESET = {"A": "exclude", "B": "open", "C": "keep"}
docs = {}
for p in DEC.glob("*.json"):
    b = json.loads(p.read_text())
    b = b.get("data", b)
    docs[b["scene"]] = b
final = {k: (docs[k]["decision"] if k in docs and docs[k].get("decision") else PRESET[v["tier"]]) for k, v in sc.items()}
print("final", dict(collections.Counter(final.values())))
ex = sorted(k for k, v in final.items() if v == "exclude")
print("excluded by split", dict(collections.Counter(sc[k]["split"] for k in ex)))
print("open (still undecided):", [k for k, v in final.items() if v == "open"])
print("\n== excluded, by type (tier counts)")
by = collections.defaultdict(list)
for k in ex:
    by[sc[k]["type"]].append(k)
for t, ks in sorted(by.items(), key=lambda kv: -len(kv[1])):
    tiers = collections.Counter(sc[k]["tier"] for k in ks)
    total = sum(1 for v in sc.values() if v["type"] == t)
    print(f"{N[t]} {len(ks)} of {total} {dict(tiers)}: " + ", ".join(k[:4] for k in ks))
print("\n== departures from the recommendation")
for label, cond in (("A kept", lambda k: sc[k]["tier"] == "A" and final[k] == "keep"),
                    ("B kept", lambda k: sc[k]["tier"] == "B" and final[k] == "keep"),
                    ("C excluded", lambda k: sc[k]["tier"] == "C" and final[k] == "exclude")):
    ks = sorted(k for k in sc if cond(k))
    print(f"{label} {len(ks)}: " + "; ".join(f"{k} {N[sc[k]['type']]} core {sc[k]['core']} area {sc[k]['area']:.0f} top "
                                            + ",".join(f"{a}:{b}" for a, b in sc[k]['top'][:4]) for k in ks))
print("\n== kept non-home types (what stays)")
kept = collections.Counter(sc[k]["type"] for k, v in final.items() if v == "keep")
print(", ".join(f"{N[t]} {n}" for t, n in kept.most_common()))
USE = ("tasks_registry", "rl_r12a_train", "eval_val_v2", "eval_test_v2", "p1_train", "p1_test", "p1_probe")
tot = {u: sum(v["use"].get(u) or 0 for v in sc.values()) for u in USE}
cut = {u: sum(sc[k]["use"].get(u) or 0 for k in ex) for u in USE}
print("\n== removed / total: " + "; ".join(f"{u} {cut[u]} / {tot[u]} ({100 * cut[u] / tot[u]:.1f}%)" for u in USE))
# the probe and the test set by question type (the probe changes again; the prereg types matter)
exs = set(ex)
for f in ("p1_probe.jsonl", "p1_test.jsonl", "p1_train.jsonl", "p1_train_held.jsonl", "p1_test_held.jsonl"):
    rows = [json.loads(l) for l in open(BANK / f)]
    gone = [r for r in rows if r["scene"] in exs]
    c = collections.Counter(r["qtype"] for r in gone)
    keep = collections.Counter(r["qtype"] for r in rows if r["scene"] not in exs)
    print(f"\n{f}: {len(rows)} -> {len(rows) - len(gone)} (-{len(gone)}); classes removed {dict(collections.Counter(r['qclass'] for r in gone))}")
    print("  removed by qtype", dict(sorted(c.items())))
    if f == "p1_probe.jsonl":
        for q in ("goal_now", "goal_after", "occluded_now", "occluded_after"):
            left = [r for r in rows if r["scene"] not in exs and r["qtype"] == q]
            print(f"  {q}: {keep[q]} left, answers {dict(collections.Counter(r['answer_value'] for r in left))}")
        print("  scenes", len({r["scene"] for r in rows}), "->", len({r["scene"] for r in rows if r["scene"] not in exs}))
# eval sets of the main line
R12 = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r12_20261019")
for name in ("eval_val_v2", "eval_test_v2"):
    tasks = [json.loads(l) for l in open(R12 / name / "tasks.jsonl")]
    gone = [t for t in tasks if t["scene_id"] in exs]
    fam = collections.Counter(t.get("task_type") or t.get("family") or "?" for t in gone)
    allfam = collections.Counter(t.get("task_type") or t.get("family") or "?" for t in tasks)
    print(f"\n{name}: {len(tasks)} -> {len(tasks) - len(gone)}; removed by family " + ", ".join(f"{k} {fam[k]}/{allfam[k]}" for k in sorted(allfam)))
groups = collections.defaultdict(list)
for k, v in sc.items():
    if v.get("group"):
        groups[v["group"]].append(k)
mixed = [g for g, ks in groups.items() if len({final[k] for k in ks}) > 1]
print("\nphysical groups split by the choice:", len(mixed), [groups[g] for g in mixed][:6])
(S / "scene_exclusions.json").write_text(json.dumps({
    "made": "2026-10-29", "source": "the user's choices on https://claude.ai/artifact/A5UDzK4r3o9Ckb7NhkRqKz over the tier presets",
    "excluded": ex, "by_type": {t: ks for t, ks in by.items()},
    "final": final, "removed": cut, "totals": tot}, ensure_ascii=False, indent=1))
print("wrote", S / "scene_exclusions.json", len(ex))
