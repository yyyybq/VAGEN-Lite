"""WalkerBench closed-loop eval of one start (raw view or merged LoRA weights), inside one one-card sco job.

    python -m vpd.walkerbench.run_vpd --model Q3S --task-set s2test --run-name pilot__Q3S__raw__640__empty \
        --out-root OUT/walkerbench/runs/pilot/Q3S/evals [--model-dir /tmp/vpd_merge/Q3S_200 --adapter-meta A/meta.json]
        [--render-size 640x400] [--constrained empty|short|off] [--history-turns K] [--limit N]
    python -m vpd.walkerbench.run_vpd --rescore --run-dir RUN      (re-score an existing run; login node)

Copied from eval_external/closed_loop/walkerbench/run.py @ bf4b2553 (that file is not edited).  What runs is
still WalkerBench's own agent, loop and scorer (b197f51): examples/vln_agent.py VLNAgent, scripts/
run_benchmark_parallel.py run_single_task (step cap round(1.5 x visual_path length)), scripts/evaluate_all.py
evaluate_session (nav/vis success: the last logged position within 15 m of a target).  The original's changes are
kept (local images instead of the commented-out requests call, no gpt-4o-mini repair, --lenient diagnostic,
temperature 0, the context-overflow tag, is_valid unchanged).  VPD changes (plan section 3.5, DECISIONS B4 B9 B13
B16 B19, PREREG_WB_PILOT sections 7-8):
1. Task sets s2test / s2val = OUT/walkerbench/split/{s2test,s2val}/tasks (179 / 57), next to their selection.json.
2. The per-run copy is vpd.walkerbench.sandbox.make_sandbox(render_size): RENDER_OUTPUT_SIZE (1280, 800) becomes
   --render-size (default 640x400), so the engine renders the agent's view at that size; no resize step.  The copy
   leaves out sandbox.LEAN_EXCLUDE (sample panoramas, tasks/full, tasks/sample) unless --full-copy.
3. Constrained decoding (--constrained, default empty): every call carries
   extra_body={"structured_outputs": {"regex": R}}, R = action_regex(len(available_moves), 0) for "empty".
   "short" allows a THOUGHT of at most --thought-max-chars (default 32) printable ASCII characters; Qwen3's
   tokenizer is byte-level BPE, so that is at most 32 tokens (B19 / PREREG 7.1: short = at most 32 tokens).
   "off" sends no regex (the secondary, unconstrained eval).  Every reply is checked with re.fullmatch against
   its step's R (regex_mismatch).
4. --max-tokens (default 512) replaces the agent's 16384 and is what the manifest records.
5. --history-turns K > 0: the agent's _trim_history is vpd.walkerbench.history.window (clean
   [sys] + (u, a) * K + u_t, DECISIONS C1); K = 0 keeps the stock full history.
6. Invalid-MOVE crash (B13): WalkerBench's loop sets observation = None after an action that failed (invalid
   move id) and the next step raises, leaving the task without a log.  Here such an action ends the episode: its
   step stays in the log as the terminal step, tagged "[vpd: invalid_move_crash]" (any other failed action:
   "[vpd: action_failed]"), is counted, and is a parse-invalid step in the parse-valid denominator.  The episode is
   then scored by evaluate_session as usual (its last logged position).
7. The server is vpd.common.vllm_server_vpd.VLLMServer: --model-dir serves merged weights under the start's
   code; xgrammar structured outputs; max_images 80; the 1,024,000-pixel cap of B16 as --mm-processor-kwargs
   (with size.longest_edge, the field that takes effect in transformers 4.57.6; see vllm_server_vpd docstring 3).
8. Every model call is logged to RUN/calls.jsonl (step, role pattern, images, usage, finish reason, regex check,
   reply); the first --prompt-dump calls (default 3) go to RUN/prompt_dump.jsonl with each image replaced by its
   sha256 and temp_images path, for the template-parity check (B5).
9. score() adds: pooled success with Wilson 95%, per type, type mean, parse-valid % of steps, tasks without a log,
   mean SPL and final error, oracle success (any position on the reconstructed path, i.e. every logged step's
   position plus evaluate_session's final one, within 15 m of a target; oracle >= success is asserted),
   hard-subset success (hard_subsets.json), the regex counts, prompt tokens per added image from usage deltas
   (descriptive only: the delta also holds the previous reply and the new text) and, with --token-audit (default
   on), image tokens per image from the processor's image_grid_thw on the logged JPEGs (the G0 measure, B4).
   Each task's result goes to RUN/per_task.jsonl (stem, kind, route, success, spl, error, steps, oracle).
10. The manifest also records the server cfg and extra_args, render size, K, the constrained mode with an example
   regex, the model dir, the adapter sha256 and meta (start, tier, seed), the vpd git state and code sha256.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys
import threading
import time
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from eval_external.common.manifest import git_state, write_manifest
from eval_external.common.stats import wilson
from vpd import paths
from vpd.walkerbench import history, sandbox
from vpd.walkerbench.action_regex import action_regex

SOURCE = "eval_external/closed_loop/walkerbench/run.py@bf4b2553"
SCHEMA = "vpd.walkerbench.results/1"
ROOT = paths.ASSETS_ROOT
TYPES = ("angle", "dis", "height", "nav", "vis")  # Spatial-Interactor's column order
NAV_TYPES = ("nav", "vis")
UNPARSEABLE_MARKERS = ("Failed to parse response", "Invalid action format", "Unknown command")
API_FAILED_MARKER = "All API retries failed"  # the model server did not answer: the run is not valid ...
OVERFLOW_MARKERS = ("maximum context length", "longer than the maximum model length")  # ... unless vLLM said this
OVERFLOW_TAG = "[eval_external: context overflow]"
CRASH_TAG = "[vpd: invalid_move_crash]"  # B13: a MOVE that failed (invalid id) ended the episode
FAILED_TAG = "[vpd: action_failed]"  # any other action that failed and returned no observation
REPAIRED_PREFIX = "[Repaired]"  # --lenient repairs (reason prefix set by the agent's _parse_repaired_json)
COUNTS = defaultdict(int)
LOCK = threading.Lock()
IMAGE_PATHS: dict[str, str] = {}  # sha256 of a sent JPEG -> its path in the copy (for prompt_dump.jsonl)
_WB_ACTION = re.compile(r"MOVE\(\s*\d+\s*\)?|ROTATION\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)?|STOP\([^)\n]*\)?")

TASK_SETS = {"standard": paths.REPO_WB / "tasks" / "standard",  # Standard-100 (the checkout's own tasks)
             "extra300": ROOT / "data_prep" / "walkerbench_extra300" / "tasks",  # eval_external extra_set.py
             "s2test": paths.WB_S2TEST_TASKS,  # 179 held-out-city tasks (WP-WBDATA split.py)
             "s2val": paths.WB_S2VAL_TASKS}  # 57 val-city tasks
HARD_SUBSETS = paths.WB_BASELINES / "hard_subsets.json"
SPLIT_S2 = paths.WB_SPLIT / "split_s2.json"
CONSTRAINED_MODES = ("off", "empty", "short")
SHORT_THOUGHT_MAX_CHARS = 32  # B19: short THOUGHT = at most 32 tokens; 32 ASCII chars <= 32 byte-level BPE tokens
THOUGHT_CLASS = r'[^"\\\n]'  # action_regex's THOUGHT character class
ASCII_THOUGHT_CLASS = r'[ !#-\[\]-~]'  # printable ASCII without '"' and '\'
EXPECTED_IMAGE_TOKENS = {(640, 400): 242, (1280, 800): 1002}  # PREREG 8.1 (grid [1,24,40] / [1,50,80], +2)
CODE_FILES = ("vpd/walkerbench/run_vpd.py", "vpd/common/vllm_server_vpd.py", "vpd/walkerbench/action_regex.py",
              "vpd/walkerbench/history.py", "vpd/walkerbench/sandbox.py")


def lenient_json(text: str) -> str | None:
    """--lenient: the last WalkerBench action named in a reply, as the JSON the agent expects; None if none."""
    found = _WB_ACTION.findall(text or "")
    if not found:
        return None
    action = found[-1].strip()
    return json.dumps({"THOUGHT": "", "ACTION": action if action.endswith(")") else action + ")"})


def step_regex(n_moves: int, mode: str, thought_max_chars: int = SHORT_THOUGHT_MAX_CHARS) -> str | None:
    """The regex sent with one call: None (off), THOUGHT empty (empty), or THOUGHT <= thought_max_chars ASCII."""
    if mode == "off":
        return None
    if mode == "empty":
        return action_regex(n_moves, 0)
    if mode == "short":
        if thought_max_chars <= 0:
            raise ValueError("--constrained short needs --thought-max-chars > 0")
        r = action_regex(n_moves, thought_max_chars)
        old = THOUGHT_CLASS + "{0,%d}" % thought_max_chars
        if r.count(old) != 1:
            raise RuntimeError("action_regex THOUGHT class changed; cannot build the ASCII short-THOUGHT regex")
        return r.replace(old, ASCII_THOUGHT_CLASS + "{0,%d}" % thought_max_chars)
    raise ValueError(f"constrained mode must be one of {CONSTRAINED_MODES}, got {mode!r}")


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def code_sha256() -> dict:
    return {f: (sha256_file(paths.WORKTREE / f) if (paths.WORKTREE / f).exists() else None) for f in CODE_FILES}


def resolve_model(arg: str) -> dict:
    """--model is a start name of vpd/starts.json (Q3S) or an eval-registry code (c0q3_sft_e8).  The served model
    name is the code (the start name for a start without a code, which then needs --model-dir)."""
    from vpd import starts

    table = starts.load_table()
    if arg in table:
        start, code = arg, table[arg]["code"]
    else:
        start, code = next((n for n, e in table.items() if e["code"] == arg), None), arg
    return {"arg": arg, "start": start, "code": code, "served": code or start}


def make_copy(run_dir: Path, task_set: str = "s2test", render_size: tuple[int, int] = (640, 400),
              lean: bool = True) -> tuple[Path, str, Path]:
    """(copy, settings diff, the task dir inside the copy)."""
    tasks_src = TASK_SETS[task_set]
    copy, diff = sandbox.make_sandbox(run_dir, tasks_src, render_size=render_size,
                                      exclude=sandbox.LEAN_EXCLUDE if lean else ())
    return copy, diff, copy / "tasks" / sandbox.task_label(tasks_src)


def tasks_source(task_set: str):
    if task_set == "standard":
        return "tasks/standard (5 types x 20)"
    sel = TASK_SETS[task_set].parent / "selection.json"
    return json.loads(sel.read_text()) if sel.exists() else str(TASK_SETS[task_set])


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def select_tasks(stems: list[str], limit: int | None) -> list[str]:
    """All stems sorted; with a limit (smoke tests): the first task of each type, then the second of each, ..."""
    tasks = sorted(stems)
    if limit:
        by_type = {}
        for t in tasks:
            by_type.setdefault(t.split("_", 1)[0], []).append(t)
        tasks = [v[k] for k in range(limit) for v in by_type.values() if k < len(v)][:limit]
    return tasks


def max_step_cap(tasks_dir: Path) -> int:
    """The largest step budget of a task dir, as run_single_task computes it: round(1.5 x len(visual_path)), or
    the task's max_steps, or 30.  With the full history a call holds one image per step taken so far."""
    cap = 0
    for p in tasks_dir.glob("*.json"):
        t = json.loads(p.read_text(encoding="utf-8"))
        vp = t.get("visual_path") or []
        cap = max(cap, int(round(len(vp) * 1.5)) if vp else int(t.get("max_steps") or 30))
    return cap


def count_images(messages) -> int:
    return sum(1 for m in messages or [] if isinstance(m.get("content"), list)
               for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url")


def redact_messages(messages) -> list[dict]:
    """The messages with every image replaced by {"type": "image_url", "image_sha256", "path"} (prompt dump)."""
    out = []
    for m in messages:
        c = m.get("content")
        if not isinstance(c, list):
            out.append(dict(m))
            continue
        parts = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "image_url":
                url = (p.get("image_url") or {}).get("url", "")
                data = url.split(",", 1)[1] if "," in url else ""
                sha = hashlib.sha256(base64.b64decode(data)).hexdigest() if data else None
                parts.append({"type": "image_url", "image_sha256": sha, "path": IMAGE_PATHS.get(sha)})
            else:
                parts.append(p)
        out.append({**m, "content": parts})
    return out


@dataclass
class EvalOptions:
    constrained: str = "empty"
    thought_max_chars: int = SHORT_THOUGHT_MAX_CHARS
    max_tokens: int = 512
    history_turns: int = 0
    blind: bool = False
    lenient_log: Path | None = None
    calls_log: Path | None = None
    prompt_dump: Path | None = None
    prompt_dump_n: int = 3

    def __post_init__(self):
        if self.constrained not in CONSTRAINED_MODES:
            raise ValueError(f"constrained must be one of {CONSTRAINED_MODES}")
        if self.history_turns < 0 or self.max_tokens <= 0:
            raise ValueError("history_turns must be >= 0 and max_tokens > 0")
        step_regex(3, self.constrained, self.thought_max_chars)  # fail early on a bad short setting


def _append_jsonl(path: Path | None, rec: dict) -> None:
    if path is None:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def make_eval_agent(base_agent, opts: EvalOptions, copy: Path):
    """WalkerBench's VLNAgent with the VPD hooks (module docstring items 3-6, 8)."""

    class EvalAgent(base_agent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.config.max_tokens = opts.max_tokens
            self._overflow = False
            self._regex = None
            self._n_moves = None
            self._task = None
            completions = self.client.chat.completions
            create = completions.create

            def create_vpd(*a, **k):  # the agent catches errors; keep why the last attempt failed
                self._overflow = False
                k["max_tokens"] = opts.max_tokens
                if self._regex is not None:
                    extra = dict(k.get("extra_body") or {})
                    extra["structured_outputs"] = {"regex": self._regex}
                    k["extra_body"] = extra
                try:
                    resp = create(*a, **k)
                except Exception as e:
                    self._overflow = any(m in str(e) for m in OVERFLOW_MARKERS)
                    self._record_call(k, None, e)
                    raise
                self._record_call(k, resp, None)
                return resp

            completions.create = create_vpd

        def _record_call(self, k: dict, resp, err) -> None:
            msgs = k.get("messages") or []
            rec = {"task": self._task, "step": self.step_count + 1, "pattern": history.role_pattern(msgs),
                   "n_images": count_images(msgs), "n_moves": self._n_moves, "max_tokens": k.get("max_tokens"),
                   "constrained": self._regex is not None}
            if err is not None:
                rec["error"] = str(err)[:500]
            else:
                choice = resp.choices[0]
                content = choice.message.content
                usage = getattr(resp, "usage", None)
                rec.update(prompt_tokens=getattr(usage, "prompt_tokens", None),
                           completion_tokens=getattr(usage, "completion_tokens", None),
                           finish_reason=choice.finish_reason, reply=content)
                if self._regex is not None:
                    rec["regex_ok"] = isinstance(content, str) and re.fullmatch(self._regex, content) is not None
            with LOCK:
                COUNTS["model_calls"] += 1
                if err is not None:
                    COUNTS["model_call_errors"] += 1
                elif self._regex is not None:
                    COUNTS["constrained_calls"] += 1
                    COUNTS["regex_mismatch"] += not rec["regex_ok"]
                if rec.get("finish_reason") == "length":
                    COUNTS["finish_length"] += 1
                _append_jsonl(opts.calls_log, rec)
                if err is None and opts.prompt_dump is not None and COUNTS["prompt_dumped"] < opts.prompt_dump_n:
                    COUNTS["prompt_dumped"] += 1
                    _append_jsonl(opts.prompt_dump, {**rec, "model": k.get("model"),
                                                     "temperature": k.get("temperature"),
                                                     "structured_outputs": (k.get("extra_body") or {}).get(
                                                         "structured_outputs"),
                                                     "messages": redact_messages(msgs)})

        def run(self, task_id, *args, **kwargs):
            self._task = task_id
            return super().run(task_id, *args, **kwargs)

        def decide_action(self, observation):
            self._n_moves = len(observation.get("available_moves") or [])
            self._regex = step_regex(self._n_moves, opts.constrained, opts.thought_max_chars)
            action = super().decide_action(observation)
            if self._overflow and API_FAILED_MARKER in str(action.get("reason")):
                action["reason"] = f"{action['reason']} {OVERFLOW_TAG}"
            return action

        def _trim_history(self):
            if opts.history_turns > 0:
                self.messages = list(history.window(self.messages, opts.history_turns))
            else:
                super()._trim_history()  # stock: max_history_turns 1000, i.e. the full history

        def execute_action(self, action):
            result = super().execute_action(action)
            if not result["success"] and result["observation"] is None and not result["done"]:
                tag = CRASH_TAG if action.get("type") == "move" else FAILED_TAG
                # the trajectory holds this same dict, so the tag reaches the step log
                action["reason"] = f"{action.get('reason') or ''} {tag} {result.get('error')}".strip()
                with LOCK:
                    COUNTS["invalid_move_crash" if tag == CRASH_TAG else "action_failed"] += 1
                return {**result, "done": True, "done_reason": tag.strip("[]").split(": ", 1)[1]}
            return result

        def get_image_base64(self, image_url):
            if opts.blind or not image_url:
                with LOCK:
                    COUNTS["images_withheld" if opts.blind else "images_missing_url"] += 1
                return None
            path = copy / image_url.lstrip("/")
            if not path.exists():
                with LOCK:
                    COUNTS["images_missing_file"] += 1
                return None
            data = path.read_bytes()
            with LOCK:
                COUNTS["images_sent"] += 1
                IMAGE_PATHS[hashlib.sha256(data).hexdigest()] = image_url.lstrip("/")
            return base64.b64encode(data).decode("utf-8")

        def _repair_json_with_gpt4o(self, malformed_response):
            if opts.lenient_log is None:
                with LOCK:
                    COUNTS["repair_skipped"] += 1
                return None
            fixed = lenient_json(malformed_response)
            with LOCK:
                COUNTS["lenient_repairs" if fixed else "lenient_no_action"] += 1
                _append_jsonl(opts.lenient_log, {"original": malformed_response, "repaired": fixed})
            return fixed

    return EvalAgent


def run_tasks(copy: Path, tasks_dir: Path, served: str, base_url: str, opts: EvalOptions, temperature: float = 0.0,
              workers: int = 8, limit: int | None = None) -> tuple[list[str], float]:
    """Run every task of tasks_dir (or a --limit selection) through WalkerBench's loop; (tasks, wall seconds)."""
    cwd = os.getcwd()
    os.chdir(copy)
    sys.path.insert(0, str(copy))
    # WalkerBench's examples/ has no __init__.py; an installed package named "examples" would win over it
    examples = types.ModuleType("examples")
    examples.__path__ = [str(copy / "examples")]
    sys.modules["examples"] = examples
    os.environ.update({"API_BASE_URL": base_url, "API_KEY": "local-vllm-no-key", "TEMPERATURE": str(temperature),
                       "BENCHMARK_URL": "http://127.0.0.1:1"})  # placeholder: the agent never needs the server
    try:
        import cv2  # the renderer's; its own pool would otherwise use every core

        cv2.setNumThreads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    except Exception:
        pass
    try:
        rbp = load_module("wb_run_benchmark_parallel", copy / "scripts" / "run_benchmark_parallel.py")
        rbp.VLNAgent = make_eval_agent(rbp.VLNAgent, opts, copy)
        rbp.TASKS_DIR = tasks_dir
        rbp.LOGS_DIR = copy / "logs"
        tasks = select_tasks([p.stem for p in tasks_dir.glob("*.json")], limit)
        rbp.total_tasks_count = len(tasks)
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            ok = list(pool.map(lambda t: rbp.run_single_task(served, t), tasks))
        wall = time.time() - t0
    finally:
        os.chdir(cwd)
    COUNTS["tasks_without_log"] = sum(not x for x in ok)
    return tasks, wall


# ------------------------------------------------------------------ scoring -----------------------------------

def step_kind(reason) -> str:
    """One logged step: api_failed | invalid_move_crash | action_failed | unparseable | repaired | ok."""
    r = str(reason)
    if API_FAILED_MARKER in r:
        return "api_failed"
    if CRASH_TAG in r:
        return "invalid_move_crash"
    if FAILED_TAG in r:
        return "action_failed"
    if any(m in r for m in UNPARSEABLE_MARKERS):
        return "unparseable"
    if r.startswith(REPAIRED_PREFIX):
        return "repaired"
    return "ok"


def step_stats(events: list[dict]) -> dict:
    c = {"steps_logged": 0, "unparseable_steps": 0, "api_failed_steps": 0, "context_overflow_steps": 0,
         "invalid_move_crash_steps": 0, "action_failed_steps": 0, "repaired_steps": 0, "parse_valid_steps": 0}
    for e in events:
        if e.get("event") != "action":
            continue
        reason = e.get("reason")
        kind = step_kind(reason)
        c["steps_logged"] += 1
        c["context_overflow_steps"] += OVERFLOW_TAG in str(reason)
        key = {"api_failed": "api_failed_steps", "invalid_move_crash": "invalid_move_crash_steps",
               "action_failed": "action_failed_steps", "unparseable": "unparseable_steps",
               "repaired": "repaired_steps", "ok": "parse_valid_steps"}[kind]
        c[key] += 1
    return c


def read_events(log: Path) -> list[dict]:
    events = []
    with open(log, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def oracle_for(events: list[dict], task_config: dict, kind: str, ev) -> tuple[int | None, float | None]:
    """(oracle success, min distance to a target) over the path evaluate_session reconstructs: the spawn point,
    every logged step's position and so also its final position.  None for non-navigation types."""
    if kind not in NAV_TYPES or not task_config:
        return None, None
    start_pano = task_config.get("spawn_point") or task_config.get("start_pano_id")
    targets = task_config.get("target_pano_ids", [])
    local = {}
    for e in events:  # the same coordinate cache as evaluate_session
        state = e.get("state", {}) or e.get("initial_state", {})
        pid = state.get("pano_id")
        lat, lng = state.get("lat"), state.get("lng")
        if pid and lat is not None and lng is not None:
            local[pid] = (lat, lng)
    path = ev.reconstruct_path_from_events(events, start_pano)
    best = float("inf")
    for pano in path:
        for t in targets:
            best = min(best, ev.get_distance(pano, t, local))
    if best == float("inf"):
        return 0, None
    return int(best <= ev.NAV_SUCCESS_THRESHOLD_M), best


def wblock(k: float, n: int) -> dict:
    if n <= 0:
        return {"k": k, "n": 0, "rate": None, "wilson95": None}
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(p, 4), "wilson95": [round(lo, 4), round(hi, 4)]}


def load_json(path: Path | None):
    if path is None or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def read_calls(path: Path | None) -> list[dict]:
    if path is None or not Path(path).exists():
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def prompt_tokens_delta(calls: list[dict]) -> dict:
    """prompt_tokens(step t+1) - prompt_tokens(step t) for consecutive answered calls of one task that add exactly
    one image.  Descriptive only: the delta also holds the previous reply, the role tokens and the new user text,
    so it is an upper bound of the per-image tokens; the G0 measure is image_tokens (B4)."""
    by_task = defaultdict(list)
    for c in calls:
        if c.get("error") is None and c.get("prompt_tokens") is not None:
            by_task[c.get("task")].append(c)
    deltas = []
    for cs in by_task.values():
        cs.sort(key=lambda c: c["step"])
        for a, b in zip(cs, cs[1:]):
            if b["n_images"] - a["n_images"] == 1 and b["step"] == a["step"] + 1:
                deltas.append(b["prompt_tokens"] - a["prompt_tokens"])
    if not deltas:
        return {"n": 0, "median": None, "min": None, "max": None, "note": "no consecutive call pair adds one image"}
    return {"n": len(deltas), "median": statistics.median(deltas), "min": min(deltas), "max": max(deltas),
            "note": "usage delta per added image incl. previous reply and new text; descriptive, not the G0 measure"}


def regex_counts(calls: list[dict]) -> dict:
    answered = [c for c in calls if c.get("error") is None]
    constrained = [c for c in answered if c.get("constrained")]
    bad = sum(1 for c in constrained if not c.get("regex_ok"))
    return {"calls": len(calls), "answered": len(answered), "errors": len(calls) - len(answered),
            "constrained_calls": len(constrained), "regex_mismatch": bad,
            "regex_valid_pct": round(100 * (len(constrained) - bad) / len(constrained), 2) if constrained else None,
            "finish_length": sum(1 for c in answered if c.get("finish_reason") == "length")}


def summarize(tasks: list[str], records: dict[str, dict], *, routes: dict | None = None, hard: dict | None = None,
              calls: list[dict] | None = None, duplicate_logs: int = 0) -> dict:
    """The results of one run from evaluated records {stem: evaluate_session result + oracle, oracle_error,
    step_stats, log}.  A task without a record scores 0 everywhere (as in the original score())."""
    routes = routes or {}
    out = {"schema": SCHEMA, "n_tasks": len(tasks), "n_scored": sum(t in records for t in tasks),
           "duplicate_logs": duplicate_logs}
    kind_of = {t: t.split("_", 1)[0] for t in tasks}
    succ = {t: int(records[t]["success"]) if t in records else 0 for t in tasks}
    # the original's fields (percentages, mean of the types present)
    by_type = defaultdict(list)
    for t in tasks:
        by_type[kind_of[t]].append(succ[t])
    out["per_type"] = {}
    for ttype in TYPES:
        s = by_type.get(ttype, [])
        p, lo, hi = wilson(sum(s), len(s))
        out["per_type"][ttype] = {"n": len(s), "success": sum(s),
                                  "sr": round(100 * p, 1) if s else None,
                                  "ci95": [round(100 * lo, 1), round(100 * hi, 1)] if s else None}
    srs = [out["per_type"][t]["sr"] for t in TYPES if out["per_type"][t]["n"]]
    p, lo, hi = wilson(sum(succ.values()), len(tasks))
    out["total_sr"] = round(sum(srs) / len(srs), 2) if srs else None
    out["overall_ci95"] = [round(100 * lo, 1), round(100 * hi, 1)] if tasks else None
    scored = [records[t] for t in tasks if t in records]
    out["mean_steps"] = round(sum(r["steps"] for r in scored) / max(len(scored), 1), 2)
    # VPD fields (fractions; {k, n, rate, wilson95} as in baselines_s2test.json)
    kinds = [k for k in TYPES if by_type.get(k)]
    out["pooled"] = wblock(sum(succ.values()), len(tasks))
    out["per_kind"] = {k: wblock(sum(by_type[k]), len(by_type[k])) for k in kinds}
    rates = [out["per_kind"][k]["rate"] for k in kinds]
    out["type_mean"] = round(sum(rates) / len(rates), 4) if rates else None
    out["tasks_without_log"] = sum(t not in records for t in tasks)
    out["spl_mean"] = round(sum(float(records[t]["spl"]) if t in records else 0.0 for t in tasks)
                            / max(len(tasks), 1), 4)
    errs = [float(records[t]["error"]) for t in tasks if t in records and kind_of[t] in NAV_TYPES
            and records[t].get("error") is not None and float(records[t]["error"]) >= 0]
    out["final_error_mean"] = round(sum(errs) / len(errs), 2) if errs else None
    out["final_error_n"] = len(errs)
    # oracle (navigation types only); oracle >= success per task
    orc = {}
    for t in tasks:
        if kind_of[t] not in NAV_TYPES:
            continue
        r = records.get(t)
        o = int(r.get("oracle") or 0) if r else 0
        if o < succ[t]:
            raise AssertionError(f"{t}: oracle {o} < success {succ[t]} (the final position is on the path)")
        orc[t] = o
    nav_tasks = [t for t in tasks if kind_of[t] in NAV_TYPES]
    out["oracle"] = {"pooled": wblock(sum(orc.values()), len(nav_tasks)),
                     "per_kind": {k: wblock(sum(orc[t] for t in nav_tasks if kind_of[t] == k),
                                            sum(1 for t in nav_tasks if kind_of[t] == k))
                                  for k in NAV_TYPES if any(kind_of[t] == k for t in nav_tasks)}}
    # hard subsets (OUT/walkerbench/baselines/hard_subsets.json)
    if hard:
        hs = {"source": str(HARD_SUBSETS), "baselines_sha256": hard.get("baselines_sha256")}
        tset = set(tasks)
        for key in ("nav_odo_fail", "vis_fwd_fail"):
            listed = list(hard.get(key) or [])
            stems = [s for s in listed if s in tset]
            hs[key] = {**wblock(sum(succ[s] for s in stems), len(stems)), "n_listed": len(listed),
                       "complete": len(stems) == len(listed)}
        out["hard_subsets"] = hs
    else:
        out["hard_subsets"] = None
    # steps
    agg = defaultdict(int)
    failed_tasks = 0
    for r in scored:
        for k, v in r["step_stats"].items():
            agg[k] += v
        failed_tasks += r["step_stats"]["api_failed_steps"] > 0
    for k in ("steps_logged", "unparseable_steps", "api_failed_steps", "context_overflow_steps",
              "invalid_move_crash_steps", "action_failed_steps", "repaired_steps", "parse_valid_steps"):
        out[k] = agg[k]
    out["api_failed_tasks"] = failed_tasks
    n_steps = max(agg["steps_logged"], 1)
    out["unparseable_pct"] = round(100 * agg["unparseable_steps"] / n_steps, 2)
    out["parse_valid_pct"] = round(100 * agg["parse_valid_steps"] / n_steps, 2) if agg["steps_logged"] else None
    answered = agg["steps_logged"] - agg["api_failed_steps"]
    out["parse_valid_pct_answered"] = round(100 * agg["parse_valid_steps"] / answered, 2) if answered else None
    out["invalid_move_crash"] = agg["invalid_move_crash_steps"]
    out["parse_valid_rule"] = ("parse-valid step = the reply parsed to an action that executed (not unparseable, "
                               "not a failed/invalid MOVE, not repaired, not an unanswered call) / all logged steps")
    # model calls
    calls = calls or []
    out["regex"] = regex_counts(calls)
    out["regex_mismatch"] = out["regex"]["regex_mismatch"]
    out["prompt_tokens_per_image_delta"] = prompt_tokens_delta(calls)
    # per task rows (per_task.jsonl)
    rows = []
    for t in tasks:
        r = records.get(t)
        st = r["step_stats"] if r else {}
        route = routes.get(t, {}).get("route_id") if isinstance(routes.get(t), dict) else None
        rows.append({"stem": t, "kind": kind_of[t], "route": route, "has_log": r is not None, "success": succ[t],
                     "spl": float(r["spl"]) if r else 0.0, "error": r.get("error") if r else None,
                     "steps": r["steps"] if r else 0, "oracle": orc.get(t),
                     "oracle_error": r.get("oracle_error") if r else None,
                     "crash": bool(st.get("invalid_move_crash_steps") or st.get("action_failed_steps")),
                     "parse_valid_steps": st.get("parse_valid_steps", 0), "log": r.get("log") if r else None})
    out["per_task_rows"] = rows
    out["per_task"] = {t: {k: v for k, v in records[t].items() if k != "step_stats"} for t in tasks if t in records}
    return out


def score(copy: Path, tasks: list[str], tasks_dir: Path, calls_log: Path | None = None,
          hard_path: Path | None = HARD_SUBSETS, split_path: Path | None = SPLIT_S2) -> dict:
    """Evaluate every log of the copy with WalkerBench's evaluate_session, add oracle and step stats, summarize."""
    if str(copy) not in sys.path:
        sys.path.insert(0, str(copy))
    ev = load_module("wb_evaluate_all", copy / "scripts" / "evaluate_all.py")
    records, dup = {}, 0
    for log in sorted((copy / "logs").glob("*.jsonl")):
        r = ev.evaluate_session(log, tasks_dir)
        if not r:
            continue
        events = read_events(log)
        tid = r["task_id"]
        cfg = ev.load_task_config(tid, tasks_dir) or {}
        r = dict(r)
        r["oracle"], r["oracle_error"] = oracle_for(events, cfg, r["task_type"], ev)
        r["step_stats"] = step_stats(events)
        r["log"] = log.name
        dup += tid in records
        records[tid] = r
    split = load_json(split_path)
    return summarize(tasks, records, routes=(split or {}).get("tasks"), hard=load_json(hard_path),
                     calls=read_calls(calls_log), duplicate_logs=dup)


def is_valid(res: dict) -> bool:
    return bool(res["server_alive_at_end"]) and res["api_failed_steps"] == res["context_overflow_steps"]


# ------------------------------------------------------------------ image-token audit (B4) --------------------

def image_token_audit(copy: Path, processor_dir: Path | str, max_images: int = 8,
                      max_pixels: int = paths.MAX_PIXELS, prompt_dump: Path | None = None) -> dict:
    """Tokens per image from the processor's image_grid_thw on logged JPEGs (prod(grid) / merge^2 + 2 for
    <|vision_start|>/<|vision_end|>); and, when prompt_dump exists and the full processor loads, HF prompt length
    (chat template + generation prompt) of each dumped call next to vLLM's usage.prompt_tokens (B5 on eval
    contexts).  Loads processor files only (no weights)."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor

    from vpd import starts

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    pdir = Path(processor_dir)
    ip = AutoImageProcessor.from_pretrained(str(pdir))
    starts.set_max_pixels(ip, max_pixels)
    merge = int(getattr(ip, "merge_size", 2))
    jpgs = sorted((copy / "temp_images").glob("*/step_*.jpg"))
    pick = jpgs[:: max(1, len(jpgs) // max_images)][:max_images] if jpgs else []
    samples = []
    for p in pick:
        with Image.open(p) as im:
            im = im.convert("RGB")
            grid = ip(images=[im], return_tensors="pt")["image_grid_thw"][0].tolist()
            n = 1
            for g in grid:
                n *= int(g)
            samples.append({"path": str(p.relative_to(copy)), "size": list(im.size), "grid_thw": grid,
                            "tokens": n // (merge * merge) + 2})
    out = {"method": "processor image_grid_thw on logged JPEGs: prod(grid)/merge^2 + 2 (DECISIONS B4)",
           "processor_dir": str(pdir), "max_pixels": max_pixels, "n": len(samples),
           "tokens_per_image": sorted({s["tokens"] for s in samples}),
           "sizes": sorted({tuple(s["size"]) for s in samples}), "samples": samples}
    if prompt_dump is not None and Path(prompt_dump).exists():
        try:
            out["prompt_parity"] = hf_prompt_parity(copy, pdir, Path(prompt_dump), max_pixels)
        except Exception as e:  # e.g. a tokenizer that only root can read on the login node
            out["prompt_parity"] = {"error": f"{type(e).__name__}: {e}"[:500]}
    return out


def hf_prompt_parity(copy: Path, processor_dir: Path, prompt_dump: Path, max_pixels: int) -> dict:
    """HF token count of each dumped call (OpenAI messages -> HF content parts, images from temp_images, chat
    template with add_generation_prompt) versus the server's usage.prompt_tokens."""
    from PIL import Image

    from vpd import starts

    proc = starts.load_processor(path=processor_dir, max_pixels=max_pixels)
    rows = []
    for line in open(prompt_dump, encoding="utf-8"):
        rec = json.loads(line)
        msgs, images = [], []
        for m in rec["messages"]:
            c = m.get("content")
            if isinstance(c, list):
                parts = []
                for p in c:
                    if p.get("type") == "image_url":
                        with Image.open(copy / p["path"]) as im:
                            images.append(im.convert("RGB"))
                        parts.append({"type": "image"})
                    else:
                        parts.append({"type": "text", "text": p.get("text", "")})
                msgs.append({"role": m["role"], "content": parts})
            else:
                msgs.append({"role": m["role"], "content": c})
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = proc(text=[text], images=images or None, return_tensors="pt")["input_ids"]
        rows.append({"task": rec.get("task"), "step": rec.get("step"), "n_images": len(images),
                     "hf_prompt_tokens": int(ids.shape[-1]), "server_prompt_tokens": rec.get("prompt_tokens")})
    return {"n": len(rows), "equal": all(r["hf_prompt_tokens"] == r["server_prompt_tokens"] for r in rows),
            "rows": rows}


# ------------------------------------------------------------------ manifest, results -------------------------

def adapter_info(meta_path: Path | None, start: str | None = None) -> dict | None:
    """Provenance of the merged adapter: meta.json (start, tier = meta phase, seed) and adapter.pt's sha256.
    Refuses an adapter trained for another start."""
    if meta_path is None:
        return None
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text())
    if start is not None and meta.get("start") not in (None, start):
        raise ValueError(f"{meta_path}: adapter of start {meta.get('start')!r}, but --model is {start!r}")
    pt = meta_path.parent / "adapter.pt"
    return {"meta_path": str(meta_path), "meta_sha256": sha256_file(meta_path),
            "adapter_path": str(pt) if pt.exists() else None, "adapter_sha256": sha256_file(pt) if pt.exists() else None,
            "start": meta.get("start"), "tier": meta.get("phase"), "seed": meta.get("seed"),
            "data": meta.get("data"), "data_sha256": meta.get("data_sha256"), "init_adapter": meta.get("init_adapter"),
            "code_sha256": meta.get("code_sha256"), "optimizer_steps": meta.get("optimizer_steps")}


def describe_model(info: dict, model_dir: Path | None) -> dict:
    """eval_external's model description of the start's code (plus the merged dir), never failing the run."""
    d = {"start": info["start"], "code": info["code"], "served_model_name": info["served"]}
    if info["code"]:
        try:
            from eval_external.adapters import models as M

            d["registry"] = M.describe(info["code"])
        except Exception as e:
            d["registry_error"] = f"{type(e).__name__}: {e}"[:300]
    if model_dir is not None:
        d["model_dir"] = str(model_dir)
        d["weights"] = "merged LoRA (job-local disk)"
    return d


def write_run_manifest(run_dir: Path, *, task_set: str, n_tasks: int, diff: str, render_size: tuple[int, int],
                       opts: EvalOptions, server: dict, models: list[dict], workers: int, temperature: float,
                       model_dir: Path | None, adapter: dict | None, run_name: str, lean: bool, limit: int | None,
                       lenient: bool) -> Path:
    k = opts.history_turns
    vpd_git = git_state(paths.WORKTREE)
    # DECISIONS G1 code snapshot: RUN/code has no .git and the job image no git, so git_state reads commit None;
    # the commit is then the snapshot's (SNAPSHOT.json, written at prepare from a clean tree).  Live: unchanged.
    from vpd import snapshot as SNAP

    try:
        snap = SNAP.snapshot_state(paths.WORKTREE)
    except SNAP.SnapshotError as e:  # an unreadable SNAPSHOT.json: recorded, commit left as git_state read it
        vpd_git["snapshot_error"] = str(e)[:300]
    else:
        if snap["mode"] == "snapshot":
            vpd_git.update(commit=snap["commit"], dirty=False, read=SNAP.MARKER)
    return write_manifest(
        run_dir, models=models,
        benchmark={"name": f"WalkerBench {task_set}", "items_expected": n_tasks,
                   "source": {"code": git_state(paths.REPO_WB), "panoramas": str(paths.PANOS),
                              "tasks": tasks_source(task_set)},
                   "metric": "pooled and per-type success (Wilson 95%), SPL, final error, oracle, hard subsets, "
                             "parse-valid %; nav/vis success = last logged position within 15 m (evaluate_all)"},
        backend={"name": "WalkerBench agent + loop + evaluator (in-process)", "source": SOURCE,
                 "settings_diff": diff, "render_size": list(render_size),
                 "sandbox": {"helper": "vpd.walkerbench.sandbox.make_sandbox",
                             "exclude": list(sandbox.LEAN_EXCLUDE) if lean else []},
                 "server": server, "workers": workers,
                 "history": {"turns": k, "rule": ("vpd.walkerbench.history.window: [sys] + (u, a) * K + u_t" if k
                                                  else "full history (stock _trim_history, max_history_turns 1000)")}},
        prompt={"impl": "examples/vln_agent.py (system prompts config/system_prompt_*.txt)",
                "images": "withheld (reference behaviour)" if opts.blind else "sent (local file)",
                "json_repair": ("lenient local rule (diagnostic): the last MOVE / ROTATION / STOP named in the reply"
                                if lenient else "off (reference uses gpt-4o-mini)"),
                "constrained": {"mode": opts.constrained,
                                "thought_max_chars": opts.thought_max_chars if opts.constrained == "short" else 0,
                                "transport": "extra_body.structured_outputs.regex (vLLM 0.11, xgrammar)",
                                "example_regex_3_moves": step_regex(3, opts.constrained, opts.thought_max_chars)}},
        frames=None,
        decoding={"temperature": temperature, "max_tokens": opts.max_tokens,
                  "structured_outputs": opts.constrained != "off", "other": "vLLM defaults"},
        subset={"kind": "limit" if limit else "full", "limit": limit, "seed": None},
        extra={"vpd": {"git": vpd_git, "code_sha256": code_sha256(), "run_name": run_name,
                       "task_set": task_set, "model_dir": None if model_dir is None else str(model_dir),
                       "adapter": adapter}})


def finalize(run_dir: Path, copy: Path, tasks_dir: Path, tasks: list[str], *, meta: dict, server_alive: bool,
             wall: float | None, audit_processor: Path | str | None = None) -> tuple[dict, bool]:
    """Score, write per_task.jsonl and results.json (results_invalid.json when not is_valid); (results, valid)."""
    res = score(copy, tasks, tasks_dir, calls_log=run_dir / "calls.jsonl")
    rows = res.pop("per_task_rows")
    res.update(meta)
    res["counts"] = dict(COUNTS) if COUNTS else meta.get("counts", {})
    res["server_alive_at_end"] = server_alive
    res["wall_seconds"] = None if wall is None else round(wall, 1)
    res["seconds_per_step"] = (round(wall / res["steps_logged"], 3)
                               if wall is not None and res["steps_logged"] else None)
    res["image_tokens"] = None
    if audit_processor:
        try:
            res["image_tokens"] = image_token_audit(copy, audit_processor, prompt_dump=run_dir / "prompt_dump.jsonl")
        except Exception as e:
            res["image_tokens"] = {"error": f"{type(e).__name__}: {e}"[:500], "processor_dir": str(audit_processor)}
    with open(run_dir / "per_task.jsonl", "w", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    valid = is_valid(res)
    for name in ("results.json", "results_invalid.json"):
        if (run_dir / name).exists():
            (run_dir / name).unlink()  # only this run's own earlier result file
    with open(run_dir / ("results.json" if valid else "results_invalid.json"), "w", newline="\n") as f:
        json.dump(res, f, indent=1, default=str)
        f.write("\n")
    return res, valid


def audit_dir_for(info: dict, model_dir: Path | None) -> Path | None:
    """Where the image-token audit reads the processor: the merged dir, else the start's processor dir, else the
    code's eval view."""
    from vpd import starts

    if model_dir is not None and os.path.lexists(Path(model_dir) / "preprocessor_config.json"):
        return Path(model_dir)
    if info["start"]:
        return starts.processor_dir(info["start"])
    if info["code"]:
        return paths.MODEL_VIEWS / info["code"]
    return None


def rescore(run_dir: Path) -> bool:
    """Re-score an existing run from its copy and logs (scoring code changed); keeps its meta and counts."""
    old_path = next((run_dir / n for n in ("results.json", "results_invalid.json") if (run_dir / n).exists()), None)
    if old_path is None:
        raise SystemExit(f"{run_dir}: no results.json / results_invalid.json")
    old = json.loads(old_path.read_text())
    tasks = json.loads((run_dir / "tasks.json").read_text())
    meta = {k: old.get(k) for k in RESULT_META_KEYS if k in old}
    meta["counts"] = old.get("counts", {})
    meta["rescored"] = {"from": old_path.name, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    copy = run_dir / "WalkerBench"
    tasks_dir = copy / "tasks" / old["tasks_label"]
    audit = old.get("image_tokens", {}) or {}
    _, valid = finalize(run_dir, copy, tasks_dir, tasks, meta=meta, server_alive=old["server_alive_at_end"],
                        wall=old.get("wall_seconds"), audit_processor=audit.get("processor_dir"))
    return valid


RESULT_META_KEYS = ("model", "start", "code", "served_model_name", "model_dir", "adapter", "run_name", "task_set",
                    "tasks_label", "render_size", "constrained", "thought_max_chars", "history_turns", "max_tokens",
                    "max_images", "blind", "lenient", "temperature", "limit", "max_step_cap", "server")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="start name (vpd/starts.json) or eval-registry code; names the served model")
    ap.add_argument("--run-dir", type=Path, help="default: --out-root / --run-name")
    ap.add_argument("--run-name")
    ap.add_argument("--out-root", type=Path, default=paths.WB_RUNS)
    ap.add_argument("--task-set", choices=sorted(TASK_SETS), default="s2test")
    ap.add_argument("--limit", type=int, default=None, help="smoke runs: the first N tasks, alternating types")
    ap.add_argument("--render-size", default="640x400", help="WxH of the agent's view (engine render size)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-images", type=int, default=80)
    ap.add_argument("--history-turns", type=int, default=0, help="K > 0: clean K-turn window; 0: full history")
    ap.add_argument("--constrained", choices=CONSTRAINED_MODES, default="empty")
    ap.add_argument("--thought-max-chars", type=int, default=SHORT_THOUGHT_MAX_CHARS,
                    help="THOUGHT cap of --constrained short, in printable ASCII chars (<= that many tokens)")
    ap.add_argument("--model-dir", type=Path, help="merged weights to serve (job-local disk)")
    ap.add_argument("--adapter-meta", type=Path, help="the merged adapter's meta.json (provenance only)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--blind", action="store_true", help="send no image (the reference agent's behaviour)")
    ap.add_argument("--lenient", action="store_true",
                    help="diagnostic: a WalkerBench action named outside the JSON is accepted (local rule, logged)")
    ap.add_argument("--full-copy", action="store_true", help="copy the whole checkout (default: LEAN_EXCLUDE)")
    ap.add_argument("--no-token-audit", action="store_true", help="skip the image_grid_thw audit (B4)")
    ap.add_argument("--prompt-dump", type=int, default=3, help="dump the first N calls' messages (B5)")
    ap.add_argument("--overwrite", action="store_true", help="replace an earlier run in the same dir")
    ap.add_argument("--rescore", action="store_true", help="re-score --run-dir from its logs (no model)")
    args = ap.parse_args(argv)

    if args.run_dir is None and not args.run_name:
        ap.error("give --run-dir or --run-name")
    run_dir = (args.run_dir or args.out_root / args.run_name).resolve()
    if args.rescore:
        raise SystemExit(0 if rescore(run_dir) else f"{run_dir}: invalid run")
    if not args.model:
        ap.error("--model is required")
    if args.max_images <= 0 or args.workers <= 0:
        ap.error("--max-images and --workers must be > 0")
    info = resolve_model(args.model)
    if info["code"] is None and args.model_dir is None:
        ap.error(f"start {info['start']} has no eval view; give --model-dir")
    if args.adapter_meta is not None and args.model_dir is None:
        ap.error("--adapter-meta describes merged weights; give --model-dir too")
    for name in ("results.json", "results_invalid.json"):
        if (run_dir / name).exists() and not args.overwrite:
            raise SystemExit(f"{run_dir / name} exists; pass --overwrite to replace this run")
    render_size = sandbox.parse_size(args.render_size)
    run_name = args.run_name or run_dir.name

    from vpd.common.vllm_server_vpd import VLLMServer  # after argument checks

    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("calls.jsonl", "prompt_dump.jsonl", "lenient_repairs.jsonl", "per_task.jsonl", "tasks.json"):
        if (run_dir / name).exists():
            (run_dir / name).unlink()  # this run dir's own files from an earlier (overwritten) run
    adapter = adapter_info(args.adapter_meta, info["start"])
    copy, diff, tasks_dir = make_copy(run_dir, args.task_set, render_size, lean=not args.full_copy)
    n_tasks = len(list(tasks_dir.glob("*.json")))
    cap = max_step_cap(tasks_dir)
    if args.history_turns == 0 and cap > args.max_images:
        raise SystemExit(f"a task allows {cap} steps but --max-images is {args.max_images}: with the full history "
                         f"its later calls would be refused and the run would be invalid")
    srv = VLLMServer(info["served"], log=run_dir / "vllm.log", model_dir=args.model_dir, max_images=args.max_images,
                     gpu_memory_utilization=args.gpu_memory_utilization)
    opts = EvalOptions(constrained=args.constrained, thought_max_chars=args.thought_max_chars,
                       max_tokens=args.max_tokens, history_turns=args.history_turns, blind=args.blind,
                       lenient_log=run_dir / "lenient_repairs.jsonl" if args.lenient else None,
                       calls_log=run_dir / "calls.jsonl",
                       prompt_dump=run_dir / "prompt_dump.jsonl" if args.prompt_dump > 0 else None,
                       prompt_dump_n=args.prompt_dump)
    server = srv.manifest()
    write_run_manifest(run_dir, task_set=args.task_set, n_tasks=n_tasks, diff=diff, render_size=render_size,
                       opts=opts, server=server, models=[describe_model(info, args.model_dir)],
                       workers=args.workers, temperature=args.temperature, model_dir=args.model_dir, adapter=adapter,
                       run_name=run_name, lean=not args.full_copy, limit=args.limit, lenient=args.lenient)
    with srv:
        server["ready_seconds"] = getattr(srv, "ready_seconds", None)
        tasks, wall = run_tasks(copy, tasks_dir, info["served"], srv.base_url, opts, args.temperature,
                                args.workers, args.limit)
        server_alive = srv.alive()
    (run_dir / "tasks.json").write_text(json.dumps(tasks) + "\n")
    meta = {"model": args.model, "start": info["start"], "code": info["code"], "served_model_name": info["served"],
            "model_dir": None if args.model_dir is None else str(args.model_dir), "adapter": adapter,
            "run_name": run_name, "task_set": args.task_set, "tasks_label": tasks_dir.name,
            "render_size": list(render_size), "constrained": args.constrained,
            "thought_max_chars": args.thought_max_chars if args.constrained == "short" else 0,
            "history_turns": args.history_turns, "max_tokens": args.max_tokens, "max_images": args.max_images,
            "blind": args.blind, "lenient": args.lenient, "temperature": args.temperature, "limit": args.limit,
            "max_step_cap": cap, "server": server}
    audit = None if args.no_token_audit else audit_dir_for(info, args.model_dir)
    res, valid = finalize(run_dir, copy, tasks_dir, tasks, meta=meta, server_alive=server_alive, wall=wall,
                          audit_processor=audit)
    # A task whose episode raised inside WalkerBench's own loop has no log and scores 0; tasks_without_log counts
    # them.  Only a dead model server, or a call left unanswered for any reason but context overflow, makes the
    # run invalid: then results_invalid.json is written instead of results.json.
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_task", "server")}, indent=1, default=str))
    if not valid:
        raise SystemExit(f"invalid run: server alive {server_alive}, {res['api_failed_steps']} steps without an "
                         f"answer from the model, {res['context_overflow_steps']} of them context overflow")


if __name__ == "__main__":
    main()
