"""Dry run of exps/matrix_v3/watch/m3_cells.py for group G in a temporary folder: the SFT chain done -> the acceptance
plan (G's format example and SFT folder); the acceptance done -> both summaries, stop at "accepted" (no pick, no RL);
`pick G --rule action` -> "picked", no chain started.  Nothing is written outside the temporary folder (chains are not
started: start_chain, chain_state, the summaries and epochs are replaced)."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

W = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix"
sys.path.insert(0, W)
sys.path.insert(0, W + "/exps/matrix_v3/watch")
import m3_cells as C  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="yb_sim_g_"))
tr = tmp / "m3_g_sft"
tr.mkdir()
C.sft_tr = lambda group: tr
C.LOG = tmp / "cells.log"
started = []
C.start_chain = lambda plan: started.append(plan) or 4242
state = {"chain": "done"}
C.chain_state = lambda p: state["chain"]
C.epochs = lambda group: {e: f"/m3_g_sft/lf_m3g/model/checkpoint-{e}" for e in range(1, 11)}

plan = C.MP.plan_v3_cell_sft("G", C.head())
(tr / "plan.json").write_text(json.dumps(plan))
C.save("G", {"group": "G", "stage": "sft", "sft_tr": str(tr), "commit": plan["commit"]})

# 1. the SFT chain done -> the acceptance chain's plan
assert C.step("G") is False
st = C.load("G")
assert st["stage"] == "accept", st
acc = started[-1]
opts = next(iter(acc["eval"]["models"].values()))["options"]
assert opts[opts.index("--format-example") + 1] == "targets_norm1000", opts
assert acc["sft_dir"] == plan["sft_dir"] and acc["sft_dir"].endswith("sft_dir_targets_norm1000"), acc["sft_dir"]
assert acc["pool"] == "zoetrope" and acc["eval_only"] and len(acc["eval"]["models"]) == 10
print("acceptance plan:", acc["tr"], opts, acc["sft_dir"])

# 2. the acceptance done -> both summaries, stop at "accepted"
names = [f"sft_m3g_e{e}_s4" for e in range(1, 11)]
(tr / "progress.json").write_text(json.dumps({"steps": {"pick_sft": {"decision": {"scores": {
    f"sft_m3g_e{e}": 0.30 + 0.01 * e for e in range(1, 11)}}}}}))
real_run = subprocess.run


def fake_run(cmd, **kw):
    out = Path(cmd[cmd.index("--out") + 1])
    if "accept_summary" in " ".join(map(str, cmd)) and "m3_accept_g" not in " ".join(map(str, cmd)):
        out.write_text(json.dumps({n: {"pass": False, "samples": 4, "temperature": 0.8, "unreadable_share": 0.0}
                                   for n in names}))
    else:  # m3_accept_g: epochs 6-8 pass on the action segment only
        out.write_text(json.dumps({n: {"pass_all": False, "pass_action": int(n.split("_e")[1].split("_")[0]) in (6, 7, 8)}
                                   for n in names}))
    return subprocess.CompletedProcess(cmd, 0, "", "")


C.subprocess.run = fake_run
n_started = len(started)
assert C.step("G") is True
st = C.load("G")
assert st["stage"] == "accepted" and len(started) == n_started, st
print("after acceptance:", st["stage"], "chains started:", len(started))

# 3. the user's rule -> the pick, no RL
C.pick("G", "action")
st = C.load("G")
assert st["stage"] == "picked" and st["pick"] == 8 and st["passing"] == [6, 7, 8] and len(started) == n_started, st
print("picked:", st["pick"], st["pick_model"], "passing", st["passing"], "rule", st["pass_rule"], "chains started:", len(started))
C.subprocess.run = real_run
print("log:\n" + (tmp / "cells.log").read_text())
print("OK", tmp)
