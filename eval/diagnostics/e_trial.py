"""Group E's trial on Qwen3 (the user's decision D of 2026-11-01): with the free_think prompt and a 6,144-token answer
budget per episode, how often a model's answers cannot be read and how long each turn's answer is -- one card, at
most one card-hour, before E's RL is planned (E stops if more than 20% of the answers cannot be read).

    python -m eval.diagnostics.e_trial prepare --run RUN --tasks-from EVAL --model HF [--per-family 8] [--submit]
    python -m eval.diagnostics.e_trial report --run RUN

The job runs ``data_gen.interiorgs_v10.evaluation run`` on a per-family subset of EVAL's tasks with the free_think
prompt (format example action_names, as the SFT data) and the RL rollouts' sampling (temperature 0.8, top-p 0.92,
history), with one change: a turn may generate what is left of the episode's budget -- the RL concat mode's
data.max_response_length, E's 6,144; vagen/agent_loop/gym_agent_loop.py gives each turn the budget not used yet --
instead of the evaluation's fixed 384 tokens per turn, which would cut thinking text short.  Every model call is
logged in RUN/turns.jsonl (tokens generated, the limit, vLLM's finish reason).  The run stops after ``--minutes``
(default 45) so that loading, rendering and the run stay within one card-hour; the episodes finished by then count.

``report`` gives the share of unreadable answers (per turn, and per episode averaged: the RL stop line's measure),
the tokens per turn (all, readable, unreadable), the share of turns that ran into the budget, the success rate, and
whether the 20% line is crossed.  Files: RUN/eval/tasks.jsonl (the subset), RUN/eval/<name>/results.jsonl,
RUN/turns.jsonl, RUN/report.json and report.md.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import shlex
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from data_gen.interiorgs_v10 import job_sco

CHECKOUT = Path(__file__).resolve().parents[2]
ENV_ROOT = "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite"
QWEN3_PYTHON = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/.venv-qwen3vl-runtime/bin/python"
BUDGET = 6144          # E's rl-config --max-response-length (configs/matrix_v3/E.yaml)
STOP_LINE = 0.20       # E: given up if more than 20% of the answers cannot be read (the user, 2026-10-18 / 11-01)
NAME = "e_trial"

LAUNCH = """#!/usr/bin/env bash
# group E's trial (eval/diagnostics/e_trial.py): the render service and the evaluation on one card, as the job
# script's eval mode (examples/train/active_spatial/sco_interiorgs_v10_train.sh), with the per-episode budget
set -uo pipefail
RUN={run}
ENV_ROOT={env_root}
PY={python}
GS_ROOT=/mnt/umm/users/yinbaiqiao/InteriorGS_ply
VALIDATED_GSPLAT_CACHE=/mnt/umm/users/yinbaiqiao/gsplat_cuda_validated_20261007
OWNER=$(stat -c %u:%g "$RUN")
mkdir -p "$RUN/logs"
trap 'status=$?; [[ -n ${{renderer_pid:-}} ]] && kill "$renderer_pid" 2>/dev/null; printf "{{\\"exit_status\\":%s,\\"ended_utc\\":\\"%s\\"}}\\n" "$status" "$(date -u +%FT%TZ)" > "$RUN/logs/e_trial_exit.json"; chown -R "$OWNER" "$RUN" 2>/dev/null || true; exit $status' EXIT
exec > >(tee -a "$RUN/logs/e_trial-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname).log") 2>&1
cd {checkout}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 NO_PROXY='*' no_proxy='*' WANDB_MODE=offline
export PYTHONPATH="{checkout}/verl:{checkout}/scripts:{checkout}:${{PYTHONPATH:-}}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_CUDA_ARCH_LIST=9.0
export CC=$ENV_ROOT/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV_ROOT/bin/x86_64-conda-linux-gnu-g++
export TORCH_EXTENSIONS_DIR=$RUN/torch_extensions_e_trial
export INTERIORGS_HTTP_TIMEOUT=900 INTERIORGS_HTTP_RETRIES=2 INTERIORGS_HTTP_BACKOFF=1 INTERIORGS_HTTP_MAX_BACKOFF=10
WHEEL_ROOT=$ENV_ROOT/lib/python3.12/site-packages/nvidia/curand
export CUDA_HOME=$ENV_ROOT CURAND_INCLUDE_DIR=$WHEEL_ROOT/include CURAND_LIBRARY_DIR=$WHEEL_ROOT/lib
export PATH="$CUDA_HOME/bin:$ENV_ROOT/bin:$PATH"
export CPATH="$CURAND_INCLUDE_DIR:$ENV_ROOT/targets/x86_64-linux/include:${{CPATH:-}}"
export LIBRARY_PATH="$CURAND_LIBRARY_DIR:$ENV_ROOT/targets/x86_64-linux/lib:${{LIBRARY_PATH:-}}"
export LD_LIBRARY_PATH="$CURAND_LIBRARY_DIR:$ENV_ROOT/lib:$ENV_ROOT/targets/x86_64-linux/lib:${{LD_LIBRARY_PATH:-}}"
export VLLM_USE_FLASHINFER_SAMPLER=1 CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
git -C {checkout} rev-parse HEAD || true
mkdir -p "$TORCH_EXTENSIONS_DIR"
[[ -d $TORCH_EXTENSIONS_DIR/gsplat_cuda ]] || cp -a "$VALIDATED_GSPLAT_CACHE" "$TORCH_EXTENSIONS_DIR/gsplat_cuda"
bash examples/train/active_spatial/start_gs_render_http_service.sh --gs-root "$GS_ROOT" --host 0.0.0.0 --port 8915 \\
  --gpus 0 --max-workers 12 --max-inflight 12 --admit-timeout 900 --conda-env "$ENV_ROOT" \\
  > "$RUN/logs/renderer_service.log" 2>&1 &
renderer_pid=$!
for _ in $(seq 1 180); do
  "$PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8915/health', timeout=2)" 2>/dev/null && break
  kill -0 "$renderer_pid" || exit 1
  sleep 2
done
export R1_RENDER_URL=http://127.0.0.1:8915/render
date -u +%FT%TZ > "$RUN/logs/e_trial_started_utc.txt"
timeout --signal=INT {minutes}m "$PY" -m eval.diagnostics.e_trial run --run "$RUN" --budget {budget} -- \\
  --eval "$RUN/eval" --name {name} --model {model} --prompt-format free_think --format-example action_names \\
  --temperature 0.8 --top-p 0.92
status=$?
echo "run exit $status (124 or 130: stopped at the time limit; the finished episodes count)"
"$PY" -m eval.diagnostics.e_trial report --run "$RUN" || true
[[ $status -eq 0 || $status -eq 124 || $status -eq 130 ]]
"""


def subset(tasks_from: Path, per_family: int, seed: int = 20261101) -> List[Dict[str, Any]]:
    """``per_family`` tasks of every task family of EVAL/tasks.jsonl, drawn with a fixed seed, in the file's order."""
    rows = [json.loads(line) for line in (tasks_from / "tasks.jsonl").read_text().splitlines() if line.strip()]
    by_family: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_family[str(r.get("task_type"))].append(i)
    rng = random.Random(seed)
    keep = sorted(i for fam in sorted(by_family) for i in rng.sample(by_family[fam], min(per_family, len(by_family[fam]))))
    return [rows[i] for i in keep]


def prepare(run: Path, tasks_from: Path, model: str, per_family: int = 8, budget: int = BUDGET, minutes: int = 45,
            python: str = QWEN3_PYTHON, pool: Optional[str] = None) -> Path:
    """RUN/eval/tasks.jsonl (the subset), RUN/job/launch.sh and prepared.json in job_sco's format (one card)."""
    run = run.resolve()
    if (run / "job" / "submission.json").exists():
        raise SystemExit(f"{run} was already submitted")
    rows = subset(Path(tasks_from), per_family)
    (run / "eval").mkdir(parents=True, exist_ok=True)
    (run / "eval" / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    job = run / "job"
    job.mkdir(parents=True, exist_ok=True)
    launch = job / "launch.sh"
    launch.write_text(LAUNCH.format(run=shlex.quote(str(run)), env_root=ENV_ROOT, python=shlex.quote(python),
                                    checkout=shlex.quote(str(CHECKOUT)), minutes=int(minutes), budget=int(budget),
                                    name=NAME, model=shlex.quote(str(model))))
    launch.chmod(0o755)
    (job / "prepared.json").write_text(json.dumps({
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "gpus": 1, "spec": job_sco.SPECS[1],
        "pool": pool or job_sco.pick_pool(), "mode_args": ["e-trial", str(run)], "python": python, "model": str(model),
        "tasks_from": str(tasks_from), "per_family": per_family, "tasks": len(rows), "budget": budget,
        "minutes": minutes}, indent=1))
    return launch


def budget_policy(base: type, budget: int, log: Path) -> type:
    """``base`` (evaluation.VllmPolicy) whose turns may generate what is left of ``budget`` tokens in the episode (the
    earlier answers' tokens subtracted), each call logged in ``log``."""
    from data_gen.interiorgs_v10 import evaluation as EV

    class BudgetPolicy(base):  # type: ignore[misc, valid-type]
        def generate(self, system: str, user_text: str, image: Any, earlier: Sequence = (), as_sft: bool = False,
                     seed: Optional[int] = None) -> str:
            from vllm import SamplingParams

            if getattr(self, "cambrian", None) is not None:
                raise SystemExit("e_trial: Qwen models only")
            tok = self.llm.get_tokenizer()
            used = sum(len(tok.encode(str(e[2]), add_special_tokens=False)) for e in earlier)
            limit = max(1, int(budget) - used)
            params = SamplingParams(temperature=self.temperature, top_p=self.top_p, seed=seed, max_tokens=limit,
                                    stop=["</action>"], include_stop_str_in_output=True)
            out = self.llm.chat(EV.chat_messages(system, earlier, user_text, image, EV._image_url, as_sft),
                                sampling_params=params, use_tqdm=False)
            o = out[0].outputs[0]
            self.last_prompt_tokens = len(out[0].prompt_token_ids)
            with open(log, "a") as fh:
                fh.write(json.dumps({"turn": len(earlier), "seed": seed, "used_before": used, "limit": limit,
                                     "tokens": len(o.token_ids), "finish": o.finish_reason,
                                     "prompt_tokens": self.last_prompt_tokens}) + "\n")
            return o.text

    return BudgetPolicy


def cmd_run(run: Path, budget: int, eval_argv: Sequence[str]) -> int:  # pragma: no cover - needs a GPU
    from data_gen.interiorgs_v10 import evaluation as EV

    EV.VllmPolicy = budget_policy(EV.VllmPolicy, budget, run / "turns.jsonl")
    return EV.main(["run", *eval_argv]) or 0


def _stats(xs: Sequence[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    return {"n": len(s), "mean": round(statistics.fmean(s), 1), "median": s[len(s) // 2],
            "p90": s[min(len(s) - 1, int(0.9 * len(s)))], "max": s[-1]}


def report(run: Path) -> Dict[str, Any]:
    from data_gen.interiorgs_v10 import evaluation as EV

    run = run.resolve()
    res = run / "eval" / NAME / "results.jsonl"
    rows = [json.loads(line) for line in res.read_text().splitlines() if line.strip()] if res.exists() else []
    calls = [json.loads(line) for line in (run / "turns.jsonl").read_text().splitlines() if line.strip()] \
        if (run / "turns.jsonl").exists() else []
    ok = [r for r in rows if "error" not in r]
    m = EV.metrics(rows) if ok else {}
    steps = [t for r in ok for t in (r.get("trajectory") or [])]
    # the calls of finished episodes come first and in order (one episode at a time); a call of the episode cut off
    # by the time limit may follow
    n_turns = sum(r["turns"] for r in ok)
    done_calls = calls[:n_turns]
    tokens = [c["tokens"] for c in done_calls]
    valid = [bool(t.get("valid")) for t in steps]
    paired = len(valid) == len(tokens)
    per_episode, k = [], 0
    for r in ok:
        per_episode.append(sum(c["tokens"] for c in done_calls[k:k + r["turns"]]))
        k += r["turns"]
    started = run / "logs" / "e_trial_started_utc.txt"
    invalid = m.get("invalid_rate")
    out = {
        "run": str(run), "budget": json.loads((run / "job" / "prepared.json").read_text()).get("budget", BUDGET),
        "episodes": len(ok), "errors": len(rows) - len(ok), "tasks_planned": sum(1 for _ in open(run / "eval" / "tasks.jsonl")),
        "turns": n_turns, "success": m.get("success"),
        "unreadable_turn_share": invalid, "unreadable_episode_mean": m.get("invalid_episode_mean"),
        "unreadable_forms": EV.unreadable_forms(rows) if ok else {},
        "think_share": round(sum("<think>" in str(t.get("answer") or "") for t in steps) / len(steps), 4) if steps else None,
        "tokens_per_turn": _stats(tokens),
        "tokens_per_turn_readable": _stats([n for n, v in zip(tokens, valid) if v]) if paired else None,
        "tokens_per_turn_unreadable": _stats([n for n, v in zip(tokens, valid) if not v]) if paired else None,
        "turns_at_limit": sum(c["finish"] == "length" for c in done_calls),
        "tokens_per_episode": _stats(per_episode),
        "episodes_using_budget": sum(n >= 0.99 * BUDGET for n in per_episode),
        "started_utc": started.read_text().strip() if started.exists() else None,
        "stop_line": STOP_LINE,
        "verdict": (None if invalid is None else
                    "over the 20% line: stop E and ask" if max(invalid, m.get("invalid_episode_mean") or 0) > STOP_LINE
                    else "under the 20% line"),
    }
    (run / "report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    t = out["tokens_per_turn"]
    (run / "report.md").write_text("\n".join([
        f"# E 组试跑（free_think，每局 {out['budget']} 个词元）", "",
        f"- 完成 {out['episodes']} 局（计划 {out['tasks_planned']} 道），{out['turns']} 个回合，成功率 {out['success']}",
        f"- 读不出的回答：按回合 {invalid}，按局平均 {out['unreadable_episode_mean']}（线 {STOP_LINE}）：{out['verdict']}",
        f"- 每回合词元：平均 {t.get('mean')}，中位 {t.get('median')}，p90 {t.get('p90')}，最多 {t.get('max')}",
        f"- 撞到上限的回合 {out['turns_at_limit']}；用满预算的局 {out['episodes_using_budget']}",
        f"- 带 <think> 的回答占 {out['think_share']}；读不出的形式 {out['unreadable_forms']}", ""]))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--tasks-from", required=True, type=Path, help="an evaluation folder (tasks.jsonl)")
    p.add_argument("--model", required=True)
    p.add_argument("--per-family", type=int, default=8)
    p.add_argument("--budget", type=int, default=BUDGET)
    p.add_argument("--minutes", type=int, default=45)
    p.add_argument("--python", default=QWEN3_PYTHON)
    p.add_argument("--pool", choices=job_sco.POOLS, default=None)
    p.add_argument("--submit", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--run", required=True, type=Path)
    r.add_argument("--budget", type=int, default=BUDGET)
    w = sub.add_parser("report")
    w.add_argument("--run", required=True, type=Path)
    if argv[:1] == ["run"] and "--" in argv:
        cut = argv.index("--")
        a = ap.parse_args(argv[:cut])
        return cmd_run(a.run.resolve(), a.budget, argv[cut + 1:])
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        print(prepare(a.run, a.tasks_from, a.model, a.per_family, a.budget, a.minutes, a.python, a.pool))
        if a.submit:
            print(job_sco.submit(a.run.resolve()))
    elif a.cmd == "report":
        print(json.dumps(report(a.run), indent=1, ensure_ascii=False))
    else:
        ap.error("run takes the evaluation's options after --")
    return 0


if __name__ == "__main__":
    sys.exit(main())
