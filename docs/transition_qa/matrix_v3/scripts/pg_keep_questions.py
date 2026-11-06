"""Pg v2's train file: v1's questions (the same sample ids, kinds and order, so the recipe draws the same 9,075) with
v2's truth (the pre-launch review's fixes, e4e245c0: boxes cut at the near plane, walls by the env's structure sign).
41 candidates that v2 adds in the four re-signed scenes are not drawn; a fresh `pointing_qa build` over v2 would
reshuffle 5,968 of the 32,498 questions (by_kind's seeded shuffle over the grown pool), so it is not used for pg_train.
pg_test and pg_probe come from the v2 build unchanged (the same questions as v1).  Writes pg_v2/<fmt>/pg_train.jsonl
and pg_v2/pg_train_keep.json."""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix")
from data_gen.transition_qa import pointing_qa as PQ  # noqa: E402

G = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/g_prep_20261106")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


cand = {}
for f in sorted((G / "pg_v2" / "raw").glob("*.jsonl")):
    for r in jl(f):
        cand[(r["task_id"], r["state_id"], r["slot"])] = r
report = {}
for fmt in ("targets_pixel", "targets_norm1000"):
    v1 = jl(G / "pg" / fmt / "pg_train.jsonl")
    out, c = [], collections.Counter()
    for r in v1:
        rec = dict(cand[(r["task_id"], r["state_id"], r["slot"])])
        rec["kind"] = PQ.kind_of(rec)
        rec["sample_id"] = PQ.sample_id(rec)
        assert (rec["kind"], rec["sample_id"]) == (r["judge"], r["sample_id"]), r["sample_id"]
        row = PQ.to_lf(rec, fmt)
        assert row["messages"][0] == r["messages"][0] and row["images"] == r["images"], r["sample_id"]
        c["changed_answer"] += row["messages"][1]["content"] != r["messages"][1]["content"]
        c["changed_box"] += row["box"] != r["box"]
        out.append(row)
    assert [x["sample_id"] for x in out] == [x["sample_id"] for x in v1]
    with open(G / "pg_v2" / fmt / "pg_train.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report[fmt] = {"questions": len(out), **dict(c)}
    for f in ("pg_test.jsonl", "pg_probe.jsonl"):  # the v2 build: the same questions as v1
        a, b = jl(G / "pg" / fmt / f), jl(G / "pg_v2" / fmt / f)
        assert [x["sample_id"] for x in a] == [x["sample_id"] for x in b], f
        report[fmt][f] = {"questions": len(b), "changed_answer": sum(x["messages"][1] != y["messages"][1] for x, y in zip(a, b))}
(G / "pg_v2" / "pg_train_keep.json").write_text(json.dumps({"note": __doc__.split("\n")[0], **report}, indent=1) + "\n")
print(json.dumps(report))
