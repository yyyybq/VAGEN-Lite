"""Reward line, step 2b check: the poses a replay looked up that are not in their task's table (replay_q3's
missing_examples, up to 3 a group).  For each: its contract key's cell (1 cm x 1 cm x 1 deg) and whether a
neighbouring cell (x, y +-1 cm or +-2 cm, yaw +-1 deg) is in the table, i.e. whether the env's exact pose has
rounded into a cell next to the ones the merged graph reached; and how many misses each task has.

    python -m reward.explain_misses --replays OUT/replays.jsonl --tables DIR [--out FILE]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

OFF = 1 << 21


def unpack(k):
    return (k >> 31) - OFF, ((k >> 9) & ((1 << 22) - 1)) - OFF, k & 511


def pack(kx, ky, kyaw):
    return ((kx + OFF) << 31) | ((ky + OFF) << 9) | (kyaw % 360)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from data_gen.interiorgs_v10 import distance_table as DT
    from data_gen.interiorgs_v10.difficulty import _split, contract_keys

    per_task, out, kinds = Counter(), [], Counter()
    for line in Path(args.replays).read_text().splitlines():
        g = json.loads(line)
        cov = g.get("coverage") or {}
        miss = int(cov.get("lookups", 0)) - int(cov.get("in_new", 0))
        if miss <= 0:
            continue
        per_task[g["task_id"]] += miss
        table = DT.load(DT.table_path(Path(args.tables), g["task_id"]))
        keys = set(table["keys"].tolist())
        for pose in g.get("missing_examples") or []:
            k = int(contract_keys(*_split(np.asarray(pose)))[0])
            kx, ky, kyaw = unpack(k)
            near1 = sum(pack(kx + dx, ky + dy, kyaw + dz) in keys
                        for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0))
            near2 = sum(pack(kx + dx, ky + dy, kyaw + dz) in keys
                        for dx in range(-2, 3) for dy in range(-2, 3) for dz in (-1, 0, 1)
                        if max(abs(dx), abs(dy)) == 2)
            kind = "next to a reached cell (1 cm / 1 deg)" if near1 else (
                "2 cm from a reached cell" if near2 else "no reached cell within 2 cm / 1 deg")
            kinds[kind] += 1
            out.append({"task_id": g["task_id"], "update": g["update"], "key_cell": [int(kx), int(ky), int(kyaw)],
                        "neighbours_within_1": int(near1), "neighbours_at_2": int(near2), "kind": kind})
    report = {"tasks_with_misses": len(per_task), "misses": sum(per_task.values()),
              "misses_per_task": dict(per_task.most_common()), "examples_checked": len(out), "kinds": dict(kinds)}
    if args.out:
        Path(args.out).write_text(json.dumps(dict(report, examples=out), indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
