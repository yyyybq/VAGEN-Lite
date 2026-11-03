"""The cells after matrix v3 (the user, 2026-11-03; exploratory, PREREG_D_vs_C.md notes 6 and 7), from the SFT-only
chain to the RL chains, unattended.

    python exps/matrix_v3/watch/m3_cells.py start-sft D_noP0       # the SFT-only plan at HEAD, its chain started
    nohup python exps/matrix_v3/watch/m3_cells.py watch D_noP0 ... &   # every 10 minutes, for each cell:

1. its SFT-only chain done -> one evaluation-only chain on the cell's own copy of the main line's acceptance set: every
   epoch sampled x4 at temperature 0.8, top-p 0.92, with the token entropy (results sft_<group>_e<k>_s4);
2. that chain done -> the main line's accept_summary (VAGEN-Lite-q3dev, read only); the pick: among the epochs that
   pass (entropy <= 0.06, unreadable turns <= 1%) the best on val (the SFT chain's own scores), ties to the earlier;
   none passing -> no RL, the cell stops there for the user;
3. a pick -> one RL chain per seed of the cell (matrix_plan --start the picked checkpoint, the registered V4) at the
   worktree's HEAD, started.
A chain that fails stops its cell (logged; nothing is restarted here).  State: <tr>_sft/cell_state.json; log:
exps/matrix_v3/watch/cells.log.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

W = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix")
M3 = W / "exps" / "matrix_v3"
PY = "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python"
ACCEPT_SET = M3 / "eval_accept_reserve200"         # the copy of the main line's set (watch/m3_accept.py)
SUMMARY = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-q3dev")
SAMPLED = ["--samples", "4", "--temperature", "0.8", "--top-p", "0.92", "--entropy", "--format-example", "action_names"]
LOG = M3 / "watch" / "cells.log"
ENV = {**os.environ, "VAGEN_LITE_WORKTREE": str(W), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
       "PYTHONDONTWRITEBYTECODE": "1"}
POLL_S = 600

sys.path.insert(0, str(W))
from data_gen.interiorgs_v10.train_chain import epoch_checkpoints  # noqa: E402
from data_gen.sft_mixture import matrix_plan as MP  # noqa: E402


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def head() -> str:
    dirty = subprocess.run(["git", "-C", str(W), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"tracked changes in {W}: {dirty[:200]}")
    return subprocess.run(["git", "-C", str(W), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def start_chain(plan: dict) -> int:
    tr = Path(plan["tr"])
    tr.mkdir(parents=True, exist_ok=True)
    if (tr / "plan.json").exists():
        raise RuntimeError(f"{tr}/plan.json exists")
    (tr / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n")
    out = open(tr / "chain.log", "a")
    proc = subprocess.Popen(["nice", "-n", "19", PY, "-m", "data_gen.interiorgs_v10.train_chain", "run", "--plan",
                             str(tr / "plan.json")], cwd=W, env=ENV, stdout=out, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    return proc.pid


def chain_state(tr: Path) -> str:
    try:
        return json.loads((tr / "progress.json").read_text()).get("state") or "?"
    except (OSError, ValueError):
        return "starting"


def sft_tr(group: str) -> Path:
    return Path(MP.v3_names(group, int(MP.v3_vars()["rl"]["seeds"][0]))["sft_tr"])


def load(group: str) -> dict:
    p = sft_tr(group) / "cell_state.json"
    return json.loads(p.read_text()) if p.exists() else {"group": group, "stage": "new"}


def save(group: str, st: dict) -> None:
    st["updated_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    (sft_tr(group) / "cell_state.json").write_text(json.dumps(st, indent=1, ensure_ascii=False) + "\n")


def start_sft(group: str) -> None:
    plan = MP.plan_v3_cell_sft(group, head())
    MP.v3_prepare_sets(plan)
    pid = start_chain(plan)
    save(group, {"group": group, "stage": "sft", "sft_tr": plan["tr"], "sft_pid": pid, "commit": plan["commit"]})
    log(f"{group}: SFT-only chain {plan['tr']} started (pid {pid}, commit {plan['commit'][:8]})")


def epochs(group: str) -> dict:
    """{epoch: checkpoint} of the cell's SFT (the chain's lf dir, checkpoint-<step> in step order)."""
    plan = json.loads((sft_tr(group) / "plan.json").read_text())
    lf = Path(next(iter(plan["sft_lf_dirs"].values())))
    return dict(epoch_checkpoints(lf / "model"))


def step(group: str) -> bool:
    """Advance one cell; True once it needs nothing more from this script."""
    st = load(group)
    tr = sft_tr(group)
    sft_group = next(iter(json.loads((tr / "plan.json").read_text())["sft_groups"]))
    if st["stage"] == "sft":
        s = chain_state(tr)
        if s == "failed":
            st["stage"] = "failed"
            log(f"{group}: the SFT-only chain failed; stopped")
        elif s == "done":
            ck = epochs(group)
            acc_set = tr / "eval_accept_reserve200"
            acc_set.mkdir(parents=True, exist_ok=True)
            for f in ("tasks.jsonl", "selection.json"):
                if not (acc_set / f).exists():
                    shutil.copyfile(ACCEPT_SET / f, acc_set / f)
                assert hashlib.sha256((acc_set / f).read_bytes()).digest() == \
                    hashlib.sha256((ACCEPT_SET / f).read_bytes()).digest(), f
            v = MP.v3_vars()
            commit = head()
            plan = {"tr": str(tr.parent / f"{tr.name}_accept"), "commit": commit, "eval_only": True,
                    "model_type": v["model_type"], "sft_dir": v["format_check_sft_dir"],
                    "job_prefix": v["chain"]["job_prefix"], "pool": v["chain"]["pool"], "max_eval_jobs": 5,
                    "retries": 4, "suspended_grace_min": 10, "gpu_log": True, "eval_sets": {"accept": str(acc_set)},
                    "rl": {"rows": v["rl"]["rows"], "exclude": v["rl"]["exclude"]},
                    "eval": {"models": {f"sft_{sft_group}_e{e}_s4": {"model": str(p), "options": SAMPLED}
                                        for e, p in sorted(ck.items())}},
                    "note": f"matrix v3 cell {group}: the sampled acceptance of every SFT epoch (PREREG note 6)"}
            st.update(stage="accept", accept_tr=plan["tr"], accept_pid=start_chain(plan), epochs={e: str(p) for e, p in ck.items()})
            log(f"{group}: SFT done ({len(ck)} epochs); acceptance chain {plan['tr']} started")
    elif st["stage"] == "accept":
        s = chain_state(Path(st["accept_tr"]))
        if s == "failed":
            st["stage"] = "failed"
            log(f"{group}: the acceptance chain failed; stopped")
        elif s == "done":
            names = [f"sft_{sft_group}_e{e}_s4" for e in sorted(int(k) for k in st["epochs"])]
            out = tr / "accept_summary.json"
            res = subprocess.run([PY, "-m", "data_gen.interiorgs_v10.accept_summary", "--set",
                                  str(tr / "eval_accept_reserve200"), "--out", str(out), *names], cwd=SUMMARY,
                                 env={**ENV, "PYTHONPATH": str(SUMMARY)}, capture_output=True, text=True)
            if res.returncode:
                raise RuntimeError(f"accept_summary: {res.stderr[-800:]}")
            acc = json.loads(out.read_text())
            scores = json.loads((tr / "progress.json").read_text())["steps"]["pick_sft"]["decision"]["scores"]
            passing = [int(k) for k in sorted(st["epochs"], key=int) if acc[f"sft_{sft_group}_e{k}_s4"].get("pass")]
            ranked = sorted(passing, key=lambda e: (-scores[f"sft_{sft_group}_e{e}"], e))
            st.update(passing=passing, val={e: scores[f"sft_{sft_group}_e{e}"] for e in sorted(int(k) for k in st["epochs"])})
            if not ranked:
                st["stage"] = "no_pick"
                log(f"{group}: no epoch passes the sampling check; no RL (PREREG note 6: back to the user)")
            else:
                pick = ranked[0]
                model = st["epochs"][str(pick)] if str(pick) in st["epochs"] else st["epochs"][pick]
                commit = head()
                seeds = [int(x) for x in MP.v3_vars()["rl"]["seeds"]][:MP.V3_CELLS[group][1]]
                rl = {}
                for seed in seeds:
                    plan = MP.plan_v3_cell_rl(group, seed, commit, model, dense="registered")
                    MP.v3_prepare_sets(plan)
                    rl[seed] = {"tr": plan["tr"], "pid": start_chain(plan)}
                st.update(stage="rl_started", pick=pick, pick_model=model, rl=rl, rl_commit=commit)
                log(f"{group}: picked epoch {pick} ({model}; passing {passing}); RL chains {[r['tr'] for r in rl.values()]}")
    save(group, st)
    return st["stage"] in ("failed", "no_pick", "rl_started")


def watch(groups) -> None:
    left = list(groups)
    while left:
        for g in list(left):
            try:
                if step(g):
                    left.remove(g)
            except Exception as exc:  # keep watching the others; this one is retried next round
                log(f"{g}: {type(exc).__name__}: {exc}")
        if left:
            time.sleep(POLL_S)
    log(f"watch over: {groups}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "start-sft":
        start_sft(args[0])
    elif cmd == "watch":
        watch(args)
    else:
        sys.exit(__doc__)
