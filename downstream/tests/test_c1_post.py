"""C1's evaluations after training, the one-card job (downstream/x/run_c1_post.sh; the user's decisions 5 of
2026-11-02 and 5 of 2026-11-03), with stand-ins for every tool it calls: the order (export, v2 val and test, the
answers' loss of C1 from its copy and of DS-B-030-s1 from its checkpoint, X0 for both, the two analyses), what a
relaunch skips, and a missing checkpoint."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STUB = r"""import json, os, sys
from pathlib import Path
script, a = Path(sys.argv[1]).name, sys.argv[2:]
def arg(k):
    return a[a.index(k) + 1] if k in a else None
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps([script] + a) + "\n")
if script == "export_starvla_backbone.py":
    out = Path(arg("--out")); out.mkdir(parents=True); (out / "starvla_export.json").write_text(arg("--ckpt"))
elif script == "x4_ppl.py" and a[0] == "score":
    Path(arg("--out")).write_text("{}\n"); print(json.dumps({"scored": 1}))
elif script == "q0_ask.py":
    q, ans = arg("--ask").split("=", 1); Path(ans).write_text("{}\n{}\n")
elif script == "x0b_qa.py":
    Path(arg("--out")).write_text("{}")
"""

EVAL = """#!/usr/bin/env bash
echo "[\\"eval\\", \\"$1\\", \\"$2\\", \\"$3\\", \\"$4\\", \\"$5\\", \\"$6\\"]" >> "$STUB_LOG"
mkdir -p "$3/$4" && echo '{}' > "$3/$4/done.json"
"""


def setup(tmp_path):
    runs = tmp_path / "runs"
    for e in ("C1-B-030-s1", "DS-B-030-s1"):
        ck = runs / "experiments" / e / "run/final_model/pytorch_model.pt"
        ck.parent.mkdir(parents=True)
        ck.write_text("w")
    (runs / "x/X0/run_20261026/out").mkdir(parents=True)
    main = tmp_path / "main"
    (main / "scripts").mkdir(parents=True)
    (main / "scripts/export_starvla_backbone.py").write_text("")  # the stand-in is called with its path
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    py = tmp_path / "py.sh"
    py.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{stub}" "$@"\n')
    py.chmod(0o755)
    ev = tmp_path / "eval.sh"
    ev.write_text(EVAL)
    env = dict(os.environ, C1P_RUNS=str(runs), C1P_MAIN=str(main), C1P_VLM_PY=str(py), C1P_EVAL=str(ev),
               C1P_B_PATH="/models/B", X4_MAIN=str(main), X4_PY=str(py), X4_VLM_PY=str(py),
               STUB_LOG=str(tmp_path / "log.jsonl"))
    return runs, env


def calls(tmp_path):
    return [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]


def test_steps_run_in_order_and_a_relaunch_skips_what_is_done(tmp_path):
    runs, env = setup(tmp_path)
    res = subprocess.run(["bash", str(ROOT / "x/run_c1_post.sh")], capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
    c = calls(tmp_path)
    ck = str(runs / "experiments/C1-B-030-s1/run/final_model/pytorch_model.pt")
    rk = str(runs / "experiments/DS-B-030-s1/run/final_model/pytorch_model.pt")
    hf = str(runs / "x/X4/hf/C1-B-030-200k")
    names = [x[0] if x[0] != "x4_ppl.py" else f"x4_ppl.py {x[1]}" for x in c]
    assert names == ["export_starvla_backbone.py", "eval", "eval", "x4_ppl.py score", "x4_ppl.py analyze",
                     "x4_ppl.py score", "x4_ppl.py analyze", "q0_ask.py", "q0_ask.py", "x0b_qa.py", "x0b_qa.py"]
    assert c[0][c[0].index("--ckpt") + 1] == ck
    assert [x[3] for x in c[1:3]] == [str(runs / "x/X4/eval_val_v2"), str(runs / "x/X4/eval_test_v2")]
    assert all(x[4:7] == ["C1-B-030-200k", hf, "no_think"] for x in c[1:3])
    assert c[3][c[3].index("--model") + 1] == hf and "--starvla-pt" not in c[3]
    assert c[5][c[5].index("--model") + 1] == "/models/B" and c[5][c[5].index("--starvla-pt") + 1] == rk
    assert c[5][c[5].index("--out") + 1].endswith("ppl/DS-B-030-s1-200k.jsonl")
    assert [x[x.index("--starvla-pt") + 1] for x in c[7:9]] == [ck, rk] and all(x[x.index("--model") + 1] == "B"
                                                                             for x in c[7:9])
    assert [x[x.index("--reference") + 1] for x in c[9:]] == ["B", "DS-B-030-s1"]
    assert c[9][c[9].index("--answers") + 3].startswith("C1-B-030-s1=")
    (tmp_path / "log.jsonl").unlink()
    again = subprocess.run(["bash", str(ROOT / "x/run_c1_post.sh")], capture_output=True, text=True, env=env, timeout=120)
    assert again.returncode == 0
    assert [x[0] for x in calls(tmp_path)][:2] == ["x4_ppl.py", "x4_ppl.py"]  # no export, no evaluation again
    log = (runs / "x/C1/qa/c1_post.log").read_text()
    assert "v2 val: done before" in log and "v2 test: done before" in log


def test_a_missing_checkpoint_stops_before_anything_runs(tmp_path):
    runs, env = setup(tmp_path)
    (runs / "experiments/DS-B-030-s1/run/final_model/pytorch_model.pt").unlink()
    res = subprocess.run(["bash", str(ROOT / "x/run_c1_post.sh")], capture_output=True, text=True, env=env, timeout=60)
    assert res.returncode == 6 and not (tmp_path / "log.jsonl").exists()
