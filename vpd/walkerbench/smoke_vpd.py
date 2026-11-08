"""CPU smoke test of the VPD WalkerBench eval (login node, no GPU, no model).

    python -m vpd.walkerbench.smoke_vpd --task-set s2test --limit 6 --render-size 640x400 --constrained empty
        [--history-turns 2] [--constrained off|short] [--run-dir DIR] [--audit-processor Q3S|''] [--keep]

Copied from eval_external/closed_loop/walkerbench/smoke.py @ bf4b2553 (that file is not edited).  A tiny
OpenAI-compatible server stands in for vLLM; everything else is what run_vpd.py does: the sandbox copy at
--render-size, WalkerBench's own agent and loop with the VPD hooks, the scorer, per_task.jsonl, results.json and
the manifest.  The fake server answers, by the step number in the user text, MOVE(lowest id), ROTATION(244.4, 0),
MOVE(highest id), MOVE(lowest id), then STOP("ARRIVED"); in constrained modes the legal MOVE ids are read from the
request's regex.  It checks, on every request:
- constrained modes: structured_outputs.regex is present and the reply re.fullmatches it; off: no regex;
- max_tokens equals --max-tokens;
- the last user message carries one JPEG whose size is --render-size;
- the role pattern is sys, (u, a) * j, u with j <= K and at most K + 1 images (K = --history-turns > 0), or the
  full history (j = step - 1, one image per step) when K = 0.
After the run: every JPEG under temp_images is --render-size; every task has a log; results.json carries the VPD
score fields; regex_mismatch == 0 (constrained); the usage-delta tokens per image equal the fake's 1000; the
manifest records max_tokens, render size, K, the mode and an example regex; with --audit-processor (default Q3S;
processor files only) the image_grid_thw audit gives 242 tokens at 640x400 and 1002 at 1280x800.
--constrained off also makes the fake answer one reply without JSON (unparseable) and one MOVE(99) (invalid move,
DECISIONS B13): exactly one of each must be counted, the crashed task must still have a log ending in the tagged
step, and both steps must be in the parse-valid denominator.
Writes RUN/smoke_report.json; removes RUN/WalkerBench unless --keep.  Exit 0 = pass.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from vpd import paths
from vpd.walkerbench import history, sandbox
from vpd.walkerbench import run_vpd as R
from vpd.walkerbench.action_regex import STOP_ACTION, reply

FAKE_PER_IMAGE = 1000  # fake usage: prompt_tokens = FAKE_BASE + FAKE_PER_IMAGE * images in the request
FAKE_BASE = 50
STATE = {"mode": "empty", "k": 0, "size": (640, 400), "max_tokens": 512, "calls": [], "problems": [],
         "unparseable_task": None, "crash_task": None}
LOCK = threading.Lock()
MOVE_ALT = re.compile(r"MOVE\\\(\(([0-9|]+)\)\\\)")
MOVES_TEXT = re.compile(r"(?m)^  (\d+): ")
STEP_TEXT = re.compile(r"\*\*Step (\d+)\*\*")
TASK_TEXT = re.compile(r"\*\*Task:\*\* (.*)")
NEW_FIELDS = ("schema", "pooled", "per_kind", "type_mean", "parse_valid_pct", "tasks_without_log", "spl_mean",
              "final_error_mean", "oracle", "hard_subsets", "regex", "regex_mismatch", "invalid_move_crash",
              "prompt_tokens_per_image_delta", "image_tokens", "seconds_per_step", "render_size", "history_turns",
              "constrained", "max_tokens")


def _problem(msg: str) -> None:
    with LOCK:
        if len(STATE["problems"]) < 50:
            STATE["problems"].append(msg)


def _text_of(msg: dict) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    return "\n".join(p.get("text", "") for p in c or [] if isinstance(p, dict) and p.get("type") == "text")


def _jpeg_size(url: str) -> tuple[int, int] | None:
    from PIL import Image

    try:
        data = base64.b64decode(url.split(",", 1)[1])
        with Image.open(io.BytesIO(data)) as im:
            return im.size if im.format == "JPEG" else None
    except Exception:
        return None


def fake_reply(step: int, ids: list[int], thought: str) -> str:
    t = step - 1
    if t in (0, 3) and ids:
        action = f"MOVE({min(ids)})"
    elif t == 2 and ids:
        action = f"MOVE({max(ids)})"
    elif t < 4:
        action = "ROTATION(244.4, 0)"
    else:
        action = STOP_ACTION
    return reply(action, thought)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = req["messages"]
        last = msgs[-1]
        text = _text_of(last)
        m = STEP_TEXT.search(text)
        step = int(m.group(1)) if m else 0
        task = (TASK_TEXT.search(text).group(1) if TASK_TEXT.search(text) else "")[:80]
        pattern = history.role_pattern(msgs)
        n_img = R.count_images(msgs)
        mode, k = STATE["mode"], STATE["k"]
        so = req.get("structured_outputs")
        regex = so.get("regex") if isinstance(so, dict) else None
        where = f"step {step} task {task[:40]!r}"
        # request checks
        if mode == "off" and so is not None:
            _problem(f"{where}: unconstrained request carries structured_outputs")
        if mode != "off" and not regex:
            _problem(f"{where}: constrained request without structured_outputs.regex")
        if req.get("max_tokens") != STATE["max_tokens"]:
            _problem(f"{where}: max_tokens {req.get('max_tokens')} != {STATE['max_tokens']}")
        last_imgs = [p for p in last.get("content", []) if isinstance(p, dict) and p.get("type") == "image_url"]
        if len(last_imgs) != 1:
            _problem(f"{where}: {len(last_imgs)} images in the last user message")
        else:
            size = _jpeg_size(last_imgs[0]["image_url"]["url"])
            if size != tuple(STATE["size"]):
                _problem(f"{where}: sent image {size} != render size {STATE['size']}")
        if k > 0:
            if not history.is_clean_window(msgs, k) or n_img > k + 1:
                _problem(f"{where}: pattern {pattern} with {n_img} images is not sys,(u,a)*<= {k},u")
        elif not history.is_clean_window(msgs, 0) or pattern.count("u") != step or n_img != step:
            _problem(f"{where}: full history expected, got pattern {pattern} with {n_img} images")
        # reply
        if regex:
            alt = MOVE_ALT.search(regex)
            ids = [int(x) for x in alt.group(1).split("|")] if alt else []
        else:
            ids = [int(x) for x in MOVES_TEXT.findall(text)]
        thought = "go on" if mode == "short" else ""
        content = fake_reply(step, ids, thought)
        if regex and re.fullmatch(regex, content) is None:
            _problem(f"{where}: fake reply {content} does not fullmatch the request regex")
        if mode == "off":
            with LOCK:
                if step == 2 and STATE["unparseable_task"] is None:
                    STATE["unparseable_task"] = task
                    content = "I will take MOVE(1) now"  # no JSON: the agent's fallback, counted unparseable
                elif step == 3 and STATE["crash_task"] is None and task != STATE["unparseable_task"]:
                    STATE["crash_task"] = task
                    content = reply("MOVE(99)")  # parses, but no such move: the B13 crash
        with LOCK:
            STATE["calls"].append({"step": step, "task": task, "pattern": pattern, "images": n_img,
                                   "regex": bool(regex), "temperature": req.get("temperature")})
        self._send(200, {"id": "x", "object": "chat.completion", "created": 0, "model": req["model"],
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                      "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": FAKE_BASE + FAKE_PER_IMAGE * n_img,
                                   "completion_tokens": len(content) // 4 + 1,
                                   "total_tokens": FAKE_BASE + FAKE_PER_IMAGE * n_img + len(content) // 4 + 1}})


def check_jpegs(copy: Path, size: tuple[int, int]) -> dict:
    from PIL import Image

    bad, n = [], 0
    for p in sorted((copy / "temp_images").glob("*/step_*.jpg")):
        n += 1
        with Image.open(p) as im:
            if im.size != tuple(size) or im.format != "JPEG":
                bad.append([str(p.relative_to(copy)), list(im.size), im.format])
    return {"n": n, "bad": bad[:10], "n_bad": len(bad)}


def last_action_reason(log: Path) -> str | None:
    acts = [e for e in R.read_events(log) if e.get("event") == "action"]
    return acts[-1].get("reason") if acts else None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="default: eval_external_assets/vpd/scratch/wbeval/smoke_<mode>_<WxH>_k<K>_<utc>")
    ap.add_argument("--task-set", default="s2test", choices=sorted(R.TASK_SETS))
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--render-size", default="640x400")
    ap.add_argument("--constrained", choices=R.CONSTRAINED_MODES, default="empty")
    ap.add_argument("--thought-max-chars", type=int, default=R.SHORT_THOUGHT_MAX_CHARS)
    ap.add_argument("--history-turns", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--audit-processor", default="Q3S",
                    help="start whose processor files the image-token audit reads ('' = skip)")
    ap.add_argument("--keep", action="store_true", help="keep RUN/WalkerBench (the copy, logs, temp_images)")
    args = ap.parse_args(argv)
    size = sandbox.parse_size(args.render_size)
    STATE.update(mode=args.constrained, k=args.history_turns, size=size, max_tokens=args.max_tokens)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = (args.run_dir or paths.scratch("wbeval") /
               f"smoke_{args.constrained}_{size[0]}x{size[1]}_k{args.history_turns}_{stamp}").resolve()
    if (run_dir / "results.json").exists() or (run_dir / "results_invalid.json").exists():
        raise SystemExit(f"{run_dir} holds an earlier smoke run; pick another --run-dir")
    run_dir.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    copy, diff, tasks_dir = R.make_copy(run_dir, args.task_set, size)
    opts = R.EvalOptions(constrained=args.constrained, thought_max_chars=args.thought_max_chars,
                         max_tokens=args.max_tokens, history_turns=args.history_turns,
                         calls_log=run_dir / "calls.jsonl", prompt_dump=run_dir / "prompt_dump.jsonl")
    fake_server = {"fake_openai_server": base_url, "cfg": {"max_images": 80}, "extra_args": [], "cmd": None}
    R.write_run_manifest(run_dir, task_set=args.task_set, n_tasks=len(list(tasks_dir.glob("*.json"))), diff=diff,
                         render_size=size, opts=opts, server=fake_server, models=[{"code": "fake"}],
                         workers=args.workers, temperature=0.0, model_dir=None, adapter=None,
                         run_name=run_dir.name, lean=True, limit=args.limit, lenient=False)
    tasks, wall = R.run_tasks(copy, tasks_dir, "fake", base_url, opts, 0.0, workers=args.workers, limit=args.limit)
    (run_dir / "tasks.json").write_text(json.dumps(tasks) + "\n")
    audit = None
    if args.audit_processor:
        from vpd import starts

        audit = starts.processor_dir(args.audit_processor)
    meta = {"model": "fake", "run_name": run_dir.name, "task_set": args.task_set, "tasks_label": tasks_dir.name,
            "render_size": list(size), "constrained": args.constrained,
            "thought_max_chars": args.thought_max_chars if args.constrained == "short" else 0,
            "history_turns": args.history_turns, "max_tokens": args.max_tokens, "max_images": 80,
            "limit": args.limit, "server": fake_server}
    res, valid = R.finalize(run_dir, copy, tasks_dir, tasks, meta=meta, server_alive=True, wall=wall,
                            audit_processor=audit)
    server.shutdown()

    checks = {}
    calls = STATE["calls"]
    checks["calls"] = len(calls) > 0
    checks["request_problems"] = not STATE["problems"]
    jp = check_jpegs(copy, size)
    checks["jpeg_dims"] = jp["n"] > 0 and jp["n_bad"] == 0
    checks["results_json"] = valid and (run_dir / "results.json").exists()
    saved = json.loads((run_dir / "results.json").read_text()) if checks["results_json"] else {}
    checks["new_fields"] = all(f in saved for f in NEW_FIELDS)
    rows = [json.loads(x) for x in open(run_dir / "per_task.jsonl")]
    checks["per_task_jsonl"] = (len(rows) == len(tasks) and all(
        set(r) >= {"stem", "kind", "route", "success", "spl", "error", "steps", "oracle"} for r in rows))
    checks["every_task_logged"] = res["n_scored"] == len(tasks) and res["tasks_without_log"] == 0
    checks["routes_joined"] = args.task_set not in ("s2test", "s2val") or all(r["route"] for r in rows)
    if args.task_set == "s2test" and args.limit:
        kinds = [t.split("_", 1)[0] for t in tasks]
        checks["limit_alternates_types"] = (len(tasks) == min(args.limit, 179)
                                            and abs(kinds.count("nav") - kinds.count("vis")) <= 1)
    rx = res["regex"]
    if args.constrained == "off":
        checks["no_constrained_calls"] = rx["constrained_calls"] == 0
        checks["one_unparseable"] = res["unparseable_steps"] == 1
        checks["one_invalid_move_crash"] = (res["invalid_move_crash"] == 1
                                            and R.COUNTS.get("invalid_move_crash", 0) == 1)
        crashed = [r for r in rows if r["crash"]]
        checks["crash_task_logged_and_tagged"] = (len(crashed) == 1 and crashed[0]["has_log"] and R.CRASH_TAG in str(
            last_action_reason(copy / "logs" / crashed[0]["log"])))
        n = res["steps_logged"]
        checks["parse_valid_denominator"] = (res["parse_valid_steps"] == n - 2
                                             and res["parse_valid_pct"] == round(100 * (n - 2) / n, 2))
    else:
        checks["regex_all_match"] = rx["constrained_calls"] == rx["answered"] > 0 and res["regex_mismatch"] == 0
        checks["parse_valid_100"] = res["parse_valid_pct"] == 100.0
    delta = res["prompt_tokens_per_image_delta"]
    checks["usage_delta"] = delta["n"] > 0 and delta["min"] == delta["max"] == FAKE_PER_IMAGE
    man = json.loads((run_dir / "manifest.json").read_text())
    con = man["prompt"]["constrained"]
    checks["manifest_fields"] = (man["decoding"]["max_tokens"] == args.max_tokens
                                 and man["backend"]["render_size"] == list(size)
                                 and man["backend"]["history"]["turns"] == args.history_turns
                                 and con["mode"] == args.constrained
                                 and (con["example_regex_3_moves"] is None) == (args.constrained == "off")
                                 and man["extra"]["vpd"]["git"].get("commit") is not None
                                 and all(man["extra"]["vpd"]["code_sha256"].values()))
    it = res.get("image_tokens")
    if audit is not None:
        want = R.EXPECTED_IMAGE_TOKENS.get(tuple(size))
        checks["image_tokens"] = (isinstance(it, dict) and "error" not in it and it["n"] > 0
                                  and (want is None or it["tokens_per_image"] == [want]))
    report = {"run_dir": str(run_dir), "tasks": tasks, "calls": len(calls), "problems": STATE["problems"],
              "jpegs": jp, "checks": checks, "pass": all(checks.values()), "settings_diff": diff,
              "summary": {k: res.get(k) for k in ("pooled", "per_kind", "oracle", "steps_logged", "parse_valid_pct",
                                                  "unparseable_steps", "invalid_move_crash", "regex",
                                                  "prompt_tokens_per_image_delta", "tasks_without_log",
                                                  "seconds_per_step")},
              "image_tokens": None if not isinstance(it, dict) else {
                  k: it.get(k) for k in ("tokens_per_image", "sizes", "n", "error", "processor_dir")},
              "prompt_parity_rows": (it or {}).get("prompt_parity", {}).get("n") if isinstance(it, dict) else None,
              "unparseable_task": STATE["unparseable_task"], "crash_task": STATE["crash_task"]}
    (run_dir / "smoke_report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))
    if not args.keep:
        shutil.rmtree(copy)  # this run's own sandbox copy (marker-checked by make_sandbox on creation)
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
