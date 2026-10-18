"""WalkerBench Standard-100 with one model code, inside one one-card sco job.

    python -m eval_external.closed_loop.walkerbench.run --model base-7B --run-dir RUN [--blind] [--temperature 0]

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
   switched off.  An unparseable reply becomes the agent's own fallback (turn in place) and is counted.
4. Temperature: --temperature (default 0, greedy, as in the QA runs); the agent's own default is 0.3.
The model is served by vLLM on the model's view (closed_loop/vllm_server.py); no external API is called.
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
FALLBACK_MARKERS = ("Failed to parse response", "Invalid action format", "Unknown command",
                    "All API retries failed")
COUNTS = defaultdict(int)
LOCK = threading.Lock()


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
              limit: int | None = None) -> list[str]:
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
            with LOCK:
                COUNTS["repair_skipped"] += 1
            return None

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


def score(copy: Path, tasks: list[str]) -> dict:
    ev = load_module("wb_evaluate_all", copy / "scripts" / "evaluate_all.py")
    per_task, fallback_steps, total_steps = {}, 0, 0
    for log in sorted((copy / "logs").glob("*.jsonl")):
        r = ev.evaluate_session(log, copy / "tasks" / "standard")
        if r:
            per_task[r["task_id"]] = r
        for line in open(log):
            e = json.loads(line)
            if e.get("event") == "action":
                total_steps += 1
                fallback_steps += any(m in str(e.get("reason")) for m in FALLBACK_MARKERS)
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
    out["steps_logged"] = total_steps
    out["fallback_steps"] = fallback_steps
    out["fallback_pct"] = round(100 * fallback_steps / max(total_steps, 1), 2)
    out["per_task"] = per_task
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--blind", action="store_true", help="send no image (the reference agent's behaviour)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
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
                "json_repair": "off (reference uses gpt-4o-mini)"},
        frames=None, decoding={"temperature": args.temperature, "max_tokens": 16384, "other": "vLLM defaults"})
    with VLLMServer(args.model, log=run_dir / "vllm.log") as srv:
        tasks = run_tasks(copy, args.model, srv.base_url, args.temperature, args.blind, args.workers)
    res = score(copy, tasks)
    res.update({"model": args.model, "blind": args.blind, "temperature": args.temperature, "counts": dict(COUNTS)})
    with open(run_dir / "results.json", "w", newline="\n") as f:
        json.dump(res, f, indent=1, default=str)
        f.write("\n")
    print(json.dumps({k: v for k, v in res.items() if k != "per_task"}, indent=1))


if __name__ == "__main__":
    main()
