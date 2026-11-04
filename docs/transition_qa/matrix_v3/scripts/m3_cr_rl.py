"""Cr's RL (the user's 甲, 2026-11-04): one chain per seed from Cr's pick, as the cell watcher starts a cell's RL chains
at its pick (m3_cells.step); Cr's cell stopped at "picked" when it had no seeds (matrix-v3 cb42874a gives it two).

    python exps/matrix_v3/watch/m3_cr_rl.py          # the two plans, nothing started
    python exps/matrix_v3/watch/m3_cr_rl.py --go     # write them, copy the set files, start both chains
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_cells as C  # noqa: E402

st = C.load("Cr")
assert st["stage"] == "picked", st["stage"]
model = st["pick_model"]
seeds = [int(x) for x in C.MP.v3_vars()["rl"]["seeds"]][:C.MP.V3_CELLS["Cr"][1]]
assert seeds == [20261019, 20261021], seeds
commit = C.head()
plans = {s: C.MP.plan_v3_cell_rl("Cr", s, commit, model, dense="registered") for s in seeds}
for s, p in plans.items():
    assert not Path(p["tr"]).exists(), p["tr"]
    print(s, p["tr"], p["rl_name"], list(p["rl_runs"]), p["start_model"], p["pool"], p["commit"][:8],
          p["rl_runs"][next(iter(p["rl_runs"].keys() - {next(iter(p["rl_runs"]))}))]["rl"]["standard"],
          json.dumps(p["rl"]["guards"]))
if "--go" in sys.argv:
    rl = {}
    for s, p in plans.items():
        C.MP.v3_prepare_sets(p)
        rl[s] = {"tr": p["tr"], "pid": C.start_chain(p)}
    st.update(stage="rl_started", rl=rl, rl_commit=commit, rl_by="the user's 甲 of 2026-11-04 (watch/m3_cr_rl.py)")
    C.save("Cr", st)
    C.log(f"Cr: RL chains {[r['tr'] for r in rl.values()]} started from epoch {st['pick']} ({model}; the user, 2026-11-04)")
