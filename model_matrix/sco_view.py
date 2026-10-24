"""Where a job of the model line may go now (the user's rule of 2026-10-24: GPU jobs only on idle machines; the main
line's pause file is obeyed).  Read only.

    python -m model_matrix.sco_view            # per pool: machines, machines with this workspace's jobs, idle estimate,
                                               # machines with free cards, waiting jobs (main line / downstream / ours)

The idle estimate is the pool's machines minus the machines with a job of this workspace (other workspaces' jobs
are not seen, so it can be too high), as the downstream line's watcher counts.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional

from data_gen.interiorgs_v10 import job_sco

PAUSE = Path("/mnt/umm/users/yinbaiqiao/downstream_runs/PAUSE_SUBMISSIONS")
OURS = "models-"  # this line's job names (models-*; chains started with VAGEN_LITE_WORKTREE use interiorgs-v10-*)


def pool_machines(pool: str) -> Optional[int]:
    res = subprocess.run([job_sco.SCO, "aec2", "clusters", "describe", "--name", pool, "-o", "json"],
                         capture_output=True, text=True)
    try:
        return len(json.loads(res.stdout)["properties"]["acn_uids"])
    except (ValueError, KeyError, TypeError):
        return None


def view(pool: str) -> Dict:
    v = job_sco.pool_view(pool)
    n = pool_machines(pool)
    if v is None:
        return {"pool": pool, "error": "the job list or a machine could not be read"}
    used = {h: sum(int(j["cards"]) for j in js) for h, js in v["hosts"].items()}
    waiting = v["waiting"]
    return {"pool": pool, "machines": n, "busy_machines": len(used),
            "idle_estimate": None if n is None else max(0, n - len(used)),
            "machines_with_free_cards": sum(1 for c in used.values() if c < 8),
            "waiting": len(waiting), "waiting_main_line": sum(1 for w in waiting if w.startswith("interiorgs-v10")),
            "waiting_downstream": sum(1 for w in waiting if w.startswith("downstream")),
            "waiting_ours": sum(1 for w in waiting if w.startswith(OURS))}


def main() -> None:
    print(json.dumps({"pause_file": PAUSE.exists(), "pools": [view(p) for p in job_sco.POOLS]}, indent=1))


if __name__ == "__main__":
    main()
