#!/usr/bin/env bash
# Reward line, phase 3: start a training chain from a given worktree (reward/start_chain_b.sh with the code directory
# and its expected commit as arguments; the earlier start scripts of rw_p2_st1_easi / rw_p2_st1_p1 merged in).
#   bash reward/start_chain_p3.sh WORKTREE COMMIT_PREFIX TR/plan_template.json
# - WORKTREE: VAGEN-Lite-reward-p3run (experiment D, a frozen detached copy of reward-design) or VAGEN-Lite-reward-mix
#   (experiment A's QA chains: the only code with qa_eval / diagnostics); its HEAD must start with COMMIT_PREFIX (at
#   least 8 characters), it must be clean and its verl must sit at the recorded gitlink;
# - the plan (the template with its commit filled in) is checked before it is put in place as TR/plan.json: the inputs
#   it names must exist (eval sets and their tasks, SFT data, rows, start and evaluated models, probes, the downstream
#   line's questions, gs_root, scenes_root) and no result name may be taken already (eval-set results of the RL runs,
#   EASI results, P1 probe and X0 / G0 answers); an existing TR/plan.json must be identical to it (a failed earlier
#   start leaves one behind; a changed template is refused, not silently replaced);
# - a chain already running for TR (progress.json chain_pid alive) is refused;
# - before submitting: the quota table (disk_runs/monitor_state.json, read at most 2 hours ago) must read below 19 TB
#   (decision 4 option 甲 of PLAN_P3_20261106: the table prints whole TB, so a reading of 19 stops), and the disk line's
#   stop flag must be absent; P3_QUOTA_STOP=20 moves the stop line to a reading of 20 TB for one submission, only
#   on the user's word (11-07 18:5x: "你先全都提交，我会接着删"); default 19;
# - the chain is a resident process on the login node: one thread per library and nice -n 19.
set -euo pipefail
C=$1
WANT=$2
PLAN_SRC=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
if [ "${#WANT}" -lt 8 ]; then echo "COMMIT_PREFIX must have at least 8 characters"; exit 1; fi
cd "$C"
export PYTHONPATH=$C PYTHONDONTWRITEBYTECODE=1 VAGEN_LITE_WORKTREE=$C
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
HEAD=$(git rev-parse HEAD)
case "$HEAD" in "$WANT"*) ;; *) echo "HEAD $HEAD does not start with $WANT"; exit 1;; esac
DIRTY=$(git status --porcelain --untracked-files=no) || { echo "git status failed"; exit 1; }
if [ -n "$DIRTY" ]; then echo "worktree not clean:"; echo "$DIRTY" | head; exit 1; fi
SUB=$(git submodule status verl) || { echo "git submodule status failed"; exit 1; }
case "$SUB" in " "*) ;; *) echo "verl is not at the recorded gitlink: $SUB"; exit 1;; esac
TR=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['tr'])" "$PLAN_SRC")
NEW=$(mktemp /tmp/rw_p3_plan.XXXXXX.json)
trap 'rm -f "$NEW"' EXIT
sed "s/COMMIT_PLACEHOLDER/$HEAD/" "$PLAN_SRC" > "$NEW"
if [ -f "$TR/plan.json" ] && ! cmp -s "$NEW" "$TR/plan.json"; then
  echo "$TR/plan.json exists and differs from the template: look at it, remove it by hand if it is stale"; exit 1
fi
$PY - "$NEW" "$HEAD" <<'PY'
import datetime as dt, json, os, re, sys
from pathlib import Path
p = json.load(open(sys.argv[1]))
assert "PLACEHOLDER" not in json.dumps(p), "a placeholder is left in the plan"
assert p["commit"] == sys.argv[2], (p["commit"], sys.argv[2])
prog = Path(p["tr"]) / "progress.json"
if prog.exists():
    pid = json.load(open(prog)).get("chain_pid")
    if pid:
        try:
            os.kill(int(pid), 0)
            raise SystemExit(f"a chain (pid {pid}) is already running for {p['tr']}")
        except ProcessLookupError:
            pass
rl, runs = p.get("rl") or {}, p.get("rl_runs") or {}
need = [*p["eval_sets"].values(), p["sft_dir"]] + [rl[k] for k in ("rows", "exclude") if rl.get(k)]
need += [p[k] for k in ("rl_start_model", "model", "gs_root", "scenes_root") if p.get(k)]
need += [p["sft_dir"] + "/" + t for t in (p.get("sft_groups") or {}).values()]
need += [r["rl"]["distance_tables"] for r in runs.values() if (r.get("rl") or {}).get("distance_tables")]
need += [r["rl"]["und_mix"]["pool"] + "/manifest.json" for r in runs.values() if isinstance((r.get("rl") or {}).get("und_mix"), dict)]
dag = [r["rl"]["dagger"] for r in runs.values() if isinstance((r.get("rl") or {}).get("dagger"), dict)]
for d in dag:  # experiment B: the tables, and a worktree whose verl carries the actor's dagger hooks
    assert any(Path(d["tables"]).glob("*.npz")), f"{d['tables']} holds no tables"
if dag:
    cfg_src = Path("verl/verl/workers/config/actor.py").read_text()
    act_src = Path("verl/verl/workers/actor/dp_actor.py").read_text()
    assert "dagger: Optional[dict]" in cfg_src and "DaggerMixer" in act_src, "this worktree's verl has no dagger hooks"
models = (p.get("eval") or {}).get("models") or {}
need += [m["model"] for m in models.values()] + [m["model"] + "/config.json" for m in models.values()]
diag = p.get("diagnostics") or {}
need += [v for v in (diag.get("probes") or {}).values() if v]
dq = diag.get("downstream_qa") or {}
need += list((dq.get("sets") or {}).values()) + list((dq.get("reference") or {}).values())
need += [dq[k] for k in ("downstream", "python") if dq.get(k)]
need += [s + "/tasks.jsonl" for s in p["eval_sets"].values()]
missing = [n for n in need if not Path(n).exists()]
assert not missing, missing
taken = []
for s in p["eval_sets"].values():  # an RL run's results: SET/<run>_<update>
    taken += [str(d) for d in Path(s).glob("*") if d.is_dir() and any(d.name.startswith(r + "_") for r in runs)]
qa = p.get("qa_eval") or {}
if qa and Path(qa["out"]).is_dir():
    taken += [str(d) for d in Path(qa["out"]).glob("*/*") if d.name in models]
for f in (Path(p["tr"]) / "diag").glob("**/*.json*"):
    if any(f.name.startswith(n + ".") or re.search(r"_%s\.jsonl$" % re.escape(n), f.name) for n in models):
        taken.append(str(f))
assert not taken, f"result names already used: {taken[:5]}"
root = Path("/mnt/umm/users/yinbaiqiao")
mon = json.load(open(root / "disk_runs/monitor_state.json"))
used = float(re.match(r"\s*([0-9.]+)\s*TB", mon["table_used"]).group(1))
last = dt.datetime.fromisoformat(mon["last_pass"].replace("Z", "+00:00"))
last = last if last.tzinfo else last.replace(tzinfo=dt.timezone.utc)
age = dt.datetime.now(dt.timezone.utc) - last
assert age < dt.timedelta(hours=2), f"quota reading too old: {mon['last_pass']}"
stop = float(os.environ.get("P3_QUOTA_STOP", "19"))  # raised only on the user's word for one submission
assert used < stop, f"quota table reads {mon['table_used']}: the stop line is a reading of {stop:g} TB"
assert not (root / "disk_runs/flags/STOP_SUBMIT_DISK_LOW").exists(), "the disk line's stop flag is set"
print("plan ok, commit", p["commit"][:8], "quota", mon["table_used"], "at", mon["last_pass"], "pool", p.get("pool"),
      "runs", {n: (r.get("rl") or {}) for n, r in runs.items()}, "seed", rl.get("seed"),
      "eval_sets", list(p["eval_sets"]), "models", list(models), "qa_eval", qa or None,
      "diagnostics", sorted(diag) or None)
PY
if [ "${CHECK_ONLY:-0}" = "1" ]; then echo "CHECK_ONLY=1: checks passed, nothing written, nothing started"; exit 0; fi
mkdir -p "$TR"
if [ ! -f "$TR/plan.json" ]; then cp "$NEW" "$TR/plan.json"; fi
setsid nohup nice -n 19 $PY -m data_gen.interiorgs_v10.train_chain run --plan "$TR/plan.json" >> "$TR/chain.log" 2>&1 < /dev/null &
echo "chain pid $!"
sleep 5
{ $PY -m data_gen.interiorgs_v10.train_chain status --dir "$TR" || true; } | head -16
date -u
