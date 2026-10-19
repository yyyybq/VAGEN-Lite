"""WalkerBench Standard-100 with one model code, inside one one-card sco job.

    python -m eval_external.closed_loop.walkerbench.run --model base-7B --run-dir RUN [--blind] [--lenient]

What runs is WalkerBench's own agent and loop (b197f51): examples/vln_agent.py VLNAgent (its system prompts per
task type, its observation text, its JSON action format MOVE / ROTATION / STOP, full dialogue history) driven by
scripts/run_benchmark_parallel.py run_single_task (step budget = round(1.5 x reference path length), its
logs), scored by scripts/evaluate_all.py evaluate_session (nav/vis: within 15 m of the target; height +-15%;
distance +-20%; angle +-30 deg).  The total is the mean success rate of the five task types.

Changes, all recorded in the manifest:
1. The run uses a copy of the checkout (RUN/WalkerBench) whose config/settings.py points at the unpacked
   panoramas and at tasks/standard (the README's way to pick a split).
2. Images: the reference get_image_base64 calls requests.get but `import requests` is commented out at the top
   of vln_agent.py, so the reference agent catches a NameError and sends no image at any step.  We read the
   same file the server would serve.  --blind keeps the reference behaviour (no image), to compare with
   published numbers.
3. The reference agent repairs unparseable replies by calling gpt-4o-mini, an external paid model; that is
   switched off.  An unparseable reply becomes the agent's own fallback (turn in place) and is counted.  (The
   repair call goes to the same API address as the model, so with a local model server it fails and the reference
   code falls back the same way.)  --lenient, a diagnostic only, puts a local rule in its place: a reply whose text
   names a WalkerBench action outside the JSON ("Answer: STOP(24.0", "ACTION: MOVE(1)") becomes {"ACTION": that
   action}, its last one, a missing final ")" added; every repair is logged to RUN/lenient_repairs.jsonl.
4. Temperature: --temperature (default 0, greedy, as in the QA runs); the agent's own default is 0.3.
The model is served by vLLM on the model's view (closed_loop/vllm_server.py); no external API is called.

Context overflow: the agent keeps every earlier turn, image included (max_history_turns 1000), and asks for
max_tokens 16384, so a long episode outgrows the model's 128,000-token context and vLLM rejects the call; the
agent then takes its own fallback (turn in place) for that step and every later one.  Such a step is the model's
failed step, as it would be with any 128k-context model under these settings: it is tagged in the step log,
counted (context_overflow_steps / _tasks) and does not make the run invalid.  Any other unanswered call does.
    python -m eval_external.closed_loop.walkerbench.run --rescore RUN    (runs made before the tag; see rescore)
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import difflib
import importlib.util
import json
import os
import re
import shutil
import sys
import threading
import types
from collections import defaultdict
from pathlib import Path

from eval_external.adapters import models as M
from eval_external.closed_loop.vllm_server import SERVER_DEFAULTS, VLLMServer
from eval_external.common.manifest import git_state, write_manifest
from eval_external.common.stats import wilson

ROOT = Path("/mnt/umm/users/yinbaiqiao/eval_external_assets")
REPO = ROOT / "repos" / "WalkerBench"
PANORAMAS = ROOT / "data" / "walkerbench_pano" / "panoramas"
TYPES = ("angle", "dis", "height", "nav", "vis")  # Spatial-Interactor's column order
UNPARSEABLE_MARKERS = ("Failed to parse response", "Invalid action format", "Unknown command")
API_FAILED_MARKER = "All API retries failed"  # the model server did not answer: the run is not valid ...
OVERFLOW_MARKERS = ("maximum context length", "longer than the maximum model length")  # ... unless vLLM said this
OVERFLOW_TAG = "[eval_external: context overflow]"
COUNTS = defaultdict(int)
LOCK = threading.Lock()
_WB_ACTION = re.compile(r"MOVE\(\s*\d+\s*\)?|ROTATION\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)?|STOP\([^)\n]*\)?")


def lenient_json(text: str) -> str | None:
    """--lenient: the last WalkerBench action named in a reply, as the JSON the agent expects; None if none."""
    found = _WB_ACTION.findall(text or "")
    if not found:
        return None
    action = found[-1].strip()
    return json.dumps({"THOUGHT": "", "ACTION": action if action.endswith(")") else action + ")"})


def make_copy(run_dir: Path) -> tuple[Path, str]:
    copy = run_dir / "WalkerBench"
    if copy.exists():
        shutil.rmtree(copy)
    shutil.copytree(REPO, copy, ignore=shutil.ignore_patterns(".git", "logs", "temp_images"))
    settings = copy / "config" / "settings.py"
    old = settings.read_text()
    new, n1 = re.subn(r"(?m)^PANORAMAS_DIR = .*$", f'PANORAMAS_DIR = Path("{PANORAMAS}")', old)
    new, n2 = re.subn(r"(?m)^TASKS_DIR = .*$", 'TASKS_DIR = BASE_DIR / "tasks" / "standard"', new)
    if (n1, n2) != (1, 1):
        raise RuntimeError("config/settings.py layout changed; PANORAMAS_DIR / TASKS_DIR not found once each")
    settings.write_text(new)
    diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "settings.py", "settings.py"))
    return copy, diff


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_tasks(copy: Path, code: str, base_url: str, temperature: float, blind: bool, workers: int,
              limit: int | None = None, lenient_log: Path | None = None) -> list[str]:
    os.chdir(copy)
    sys.path.insert(0, str(copy))
    # WalkerBench's examples/ has no __init__.py; an installed package named "examples" would win over it
    examples = types.ModuleType("examples")
    examples.__path__ = [str(copy / "examples")]
    sys.modules["examples"] = examples
    os.environ.update({"API_BASE_URL": base_url, "API_KEY": "local-vllm-no-key", "TEMPERATURE": str(temperature),
                       "BENCHMARK_URL": "http://127.0.0.1:1"})  # placeholder: the agent never needs the server
    rbp = load_module("wb_run_benchmark_parallel", copy / "scripts" / "run_benchmark_parallel.py")
    base_agent = rbp.VLNAgent

    class EvalAgent(base_agent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._overflow = False
            completions = self.client.chat.completions
            create = completions.create

            def create_noting_overflow(*a, **k):  # the agent catches the error; keep why the last attempt failed
                self._overflow = False
                try:
                    return create(*a, **k)
                except Exception as e:
                    self._overflow = any(m in str(e) for m in OVERFLOW_MARKERS)
                    raise

            completions.create = create_noting_overflow

        def decide_action(self, observation):
            action = super().decide_action(observation)
            if self._overflow and API_FAILED_MARKER in str(action.get("reason")):
                action["reason"] = f"{action['reason']} {OVERFLOW_TAG}"
            return action

        def get_image_base64(self, image_url):
            if blind or not image_url:
                with LOCK:
                    COUNTS["images_withheld" if blind else "images_missing_url"] += 1
                return None
            path = copy / image_url.lstrip("/")
            if not path.exists():
                with LOCK:
                    COUNTS["images_missing_file"] += 1
                return None
            with LOCK:
                COUNTS["images_sent"] += 1
            return base64.b64encode(path.read_bytes()).decode("utf-8")

        def _repair_json_with_gpt4o(self, malformed_response):
            if lenient_log is None:
                with LOCK:
                    COUNTS["repair_skipped"] += 1
                return None
            fixed = lenient_json(malformed_response)
            with LOCK:
                COUNTS["lenient_repairs" if fixed else "lenient_no_action"] += 1
                with open(lenient_log, "a") as f:
                    f.write(json.dumps({"original": malformed_response, "repaired": fixed}) + "\n")
            return fixed

    rbp.VLNAgent = EvalAgent
    rbp.TASKS_DIR = copy / "tasks" / "standard"
    rbp.LOGS_DIR = copy / "logs"
    tasks = sorted(p.stem for p in rbp.TASKS_DIR.glob("*.json"))
    if limit:  # smoke tests only: the first task of each type, then the second of each type, ...
        by_type = {}
        for t in tasks:
            by_type.setdefault(t.split("_", 1)[0], []).append(t)
        tasks = [v[k] for k in range(limit) for v in by_type.values() if k < len(v)][:limit]
    rbp.total_tasks_count = len(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        ok = list(pool.map(lambda t: rbp.run_single_task(code, t), tasks))
    COUNTS["tasks_without_log"] = sum(not x for x in ok)
    return tasks


def count_steps(logs: list[Path]) -> dict:
    """Steps of the step logs: all, unparseable, unanswered, unanswered for context overflow (tagged), and the
    tasks (log names) with an unanswered step."""
    c = {"steps_logged": 0, "unparseable_steps": 0, "api_failed_steps": 0, "context_overflow_steps": 0}
    failed_tasks = set()
    for log in logs:
        for line in open(log):
            e = json.loads(line)
            if e.get("event") != "action":
                continue
            reason = str(e.get("reason"))
            c["steps_logged"] += 1
            c["unparseable_steps"] += any(m in reason for m in UNPARSEABLE_MARKERS)
            c["api_failed_steps"] += API_FAILED_MARKER in reason
            c["context_overflow_steps"] += OVERFLOW_TAG in reason
            if API_FAILED_MARKER in reason:
                failed_tasks.add(log.stem)
    c["api_failed_tasks"] = len(failed_tasks)
    return c


def overflow_from_run_log(run_log: Path) -> dict:
    """For runs made before the tag: the agent prints every failed attempt with vLLM's message; a step is
    unanswered when its last attempt (k/k) fails.  Counts those last attempts, and the ones that were overflow."""
    last = [line for line in open(run_log, errors="replace")
            if re.search(r"API call failed \(attempt (\d+)/\1\)", line)]
    return {"last_attempt_failures": len(last),
            "overflow": sum(any(m in line for m in OVERFLOW_MARKERS) for line in last)}


def is_valid(res: dict) -> bool:
    return bool(res["server_alive_at_end"]) and res["api_failed_steps"] == res["context_overflow_steps"]


def score(copy: Path, tasks: list[str]) -> dict:
    ev = load_module("wb_evaluate_all", copy / "scripts" / "evaluate_all.py")
    logs = sorted((copy / "logs").glob("*.jsonl"))
    per_task = {}
    for log in logs:
        r = ev.evaluate_session(log, copy / "tasks" / "standard")
        if r:
            per_task[r["task_id"]] = r
    steps = count_steps(logs)
    by_type = defaultdict(list)
    for t in tasks:
        r = per_task.get(t)
        ttype = t.split("_", 1)[0]
        by_type[ttype].append(r["success"] if r else 0)
    out = {"per_type": {}, "n_tasks": len(tasks), "n_scored": len(per_task)}
    for ttype in TYPES:
        s = by_type.get(ttype, [])
        p, lo, hi = wilson(sum(s), len(s))
        out["per_type"][ttype] = {"n": len(s), "success": sum(s), "sr": round(100 * p, 1),
                                  "ci95": [round(100 * lo, 1), round(100 * hi, 1)]}
    srs = [out["per_type"][t]["sr"] for t in TYPES if out["per_type"][t]["n"]]
    allsucc = sum(sum(v) for v in by_type.values())
    p, lo, hi = wilson(allsucc, len(tasks))
    out["total_sr"] = round(sum(srs) / len(srs), 2)
    out["overall_ci95"] = [round(100 * lo, 1), round(100 * hi, 1)]
    out["mean_steps"] = round(sum(r["steps"] for r in per_task.values()) / max(len(per_task), 1), 2)
    out.update(steps)
    out["unparseable_pct"] = round(100 * steps["unparseable_steps"] / max(steps["steps_logged"], 1), 2)
    out["per_task"] = per_task
    return out


def rescore(run_dir: Path) -> bool:
    """A run made before the overflow tag that was judged invalid (results_invalid.json): its unanswered steps
    count as overflow when run.log shows exactly that many last-attempt failures and every one of them is vLLM's
    context-length message.  Writes results.json (with "rescored") when the run is then valid."""
    res = json.loads((run_dir / "results_invalid.json").read_text())
    steps = count_steps(sorted((run_dir / "WalkerBench" / "logs").glob("*.jsonl")))
    log = overflow_from_run_log(run_dir / "job" / "run.log")
    all_overflow = log["last_attempt_failures"] == log["overflow"] == steps["api_failed_steps"]
    res.update(steps)
    res["context_overflow_steps"] = steps["api_failed_steps"] if all_overflow else 0
    res["rescored"] = {"from": "results_invalid.json + job/run.log", "run_log": log, "all_overflow": all_overflow}
    ok = is_valid(res)
    if ok:
        with open(run_dir / "results.json", "w", newline="\n") as f:
            json.dump(res, f, indent=1, default=str)
            f.write("\n")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--blind", action="store_true", help="send no image (the reference agent's behaviour)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rescore", action="store_true", help="rescore an earlier invalid run (no model, login node)")
    ap.add_argument("--lenient", action="store_true",
                    help="diagnostic: a WalkerBench action named outside the JSON is accepted (local rule, logged)")
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    if args.rescore:
        raise SystemExit(0 if rescore(run_dir) else f"{run_dir}: still invalid")
    if not args.model:
        ap.error("--model is required")
    run_dir.mkdir(parents=True, exist_ok=True)
    copy, diff = make_copy(run_dir)
    write_manifest(
        run_dir, models=[M.describe(args.model)],
        benchmark={"name": "WalkerBench Standard-100", "items_expected": 100,
                   "source": {"code": git_state(REPO), "panoramas": "lalayg/pano@415676872c07 panoramas.zip (zoom 2)",
                              "tasks": "tasks/standard (5 types x 20)"},
                   "metric": "success rate per type; total = mean of the 5 types; steps = mean executed actions"},
        backend={"name": "WalkerBench agent + loop + evaluator (in-process)", "settings_diff": diff,
                 "server": {"vllm": SERVER_DEFAULTS}, "workers": args.workers},
        prompt={"impl": "examples/vln_agent.py (system prompts config/system_prompt_*.txt)",
                "images": "withheld (reference behaviour)" if args.blind else "sent (local file)",
                "json_repair": ("lenient local rule (diagnostic): the last MOVE / ROTATION / STOP named in the reply"
                                if args.lenient else "off (reference uses gpt-4o-mini)")},
        frames=None, decoding={"temperature": args.temperature, "max_tokens": 16384, "other": "vLLM defaults"})
    lenient_log = run_dir / "lenient_repairs.jsonl" if args.lenient else None
    if lenient_log is not None and lenient_log.exists():
        lenient_log.unlink()
    with VLLMServer(args.model, log=run_dir / "vllm.log") as srv:
        tasks = run_tasks(copy, args.model, srv.base_url, args.temperature, args.blind, args.workers,
                          lenient_log=lenient_log)
        server_alive = srv.alive()
    res = score(copy, tasks)
    res.update({"model": args.model, "blind": args.blind, "lenient": args.lenient, "temperature": args.temperature,
                "counts": dict(COUNTS), "server_alive_at_end": server_alive})
    # A task whose episode raised inside WalkerBench's own loop has no log and scores 0 (score() counts every task):
    # the loop indexes the observation after an action that failed and returned none ("'NoneType' object is not
    # subscriptable"); res["counts"]["tasks_without_log"] reports how many.  Only a dead model server, or a call
    # left unanswered for any reason but context overflow, makes the run invalid: then no results.json is written,
    # so the queue marks it failed.
    valid = is_valid(res)
    with open(run_dir / ("results.json" if valid else "results_invalid.json"), "w", newline="\n") as f:
        json.dump(res, f, indent=1, default=str)
        f.write("\n")
    print(json.dumps({k: v for k, v in res.items() if k != "per_task"}, indent=1))
    if not valid:
        raise SystemExit(f"invalid run: server alive {server_alive}, {res['api_failed_steps']} steps without an "
                         f"answer from the model, {res['context_overflow_steps']} of them context overflow")


if __name__ == "__main__":
    main()
