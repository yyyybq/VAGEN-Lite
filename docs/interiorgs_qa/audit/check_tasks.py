"""Check data_gen/interiorgs_qa/tasks.py against COVERAGE.md's judgments and against generated questions.

    python docs/interiorgs_qa/audit/check_tasks.py [questions.jsonl ...]

1. Every task in TASKS is judged doable in coverage_v2_judgments.py (same / differs as recorded), lands in the group
   COVERAGE names, and every doable judgment is a task (except MindCube's two setting rows, which are labels).
2. Every question in the given files passes tasks.check, except questions of tasks the user dropped (counted).
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import coverage_v2_judgments as cov  # noqa: E402
from data_gen.interiorgs_qa import tasks  # noqa: E402

LABEL_ROWS = {"MC-setting-around", "MC-setting-among"}
bad = []
doable = {k: v for k, v in cov.J.items() if v[0] in (cov.SAME, cov.DIFF)}
for k, t in tasks.TASKS.items():
    if k not in doable:
        bad.append(f"{k}: not doable in COVERAGE")
        continue
    level = tasks.SAME if doable[k][0] == cov.SAME else tasks.DIFF
    if level != t.definition:
        bad.append(f"{k}: definition {t.definition} != COVERAGE {level}")
    where = doable[k][1]
    if not any(g in where for g in t.groups):
        bad.append(f"{k}: groups {t.groups} not in COVERAGE '{where}'")
for k in doable:
    if k not in tasks.TASKS and k not in LABEL_ROWS:
        bad.append(f"{k}: doable in COVERAGE but no task")
print(json.dumps({"tasks": len(tasks.TASKS), "doable_in_coverage": len(doable), "table_problems": bad}, ensure_ascii=False))

dropped = {k for k, v in cov.J.items() if v[0] == cov.DROP}
counts, mism, skipped = collections.Counter(), [], collections.Counter()
for f in sys.argv[1:]:
    for line in open(f):
        if not line.strip():
            continue
        q = json.loads(line)
        q = q.get("question", q)
        if q["subtasks"][0] in dropped:
            skipped[q["subtasks"][0]] += 1
            continue
        try:
            counts[tasks.check(q)] += 1
        except tasks.TaskMismatch as e:
            mism.append(str(e))
print(json.dumps({"questions_ok": sum(counts.values()), "tasks_seen": len(counts), "dropped_task_questions": dict(skipped),
                  "mismatches": len(mism), "examples": mism[:10]}, ensure_ascii=False))
sys.exit(1 if bad or mism else 0)
