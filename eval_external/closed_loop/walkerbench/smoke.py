"""CPU smoke test of the WalkerBench runner (login node, no GPU, no model).

    python -m eval_external.closed_loop.walkerbench.smoke --run-dir /tmp/wb_smoke [--limit 5] [--error overflow]

A tiny OpenAI-compatible server answers every call with one fixed action (first a ROTATION, then STOP).  The run
goes through everything run.py does except vLLM: the per-run copy with its settings change, WalkerBench's own
agent and loop, the image hook, the scorer.  It checks that every model call carried exactly one image (the
reference agent sends none) and that each task produced a scored log.

--error overflow / other: from the second turn on, every call gets a 400 (vLLM's context-length message, or
another one).  Checks that the unanswered steps are counted, tagged as overflow only in the first case, and that
only the second case makes the run invalid.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from eval_external.closed_loop.walkerbench import run as R

CALLS = []
ERROR = {"mode": "none"}
MESSAGES = {  # as vLLM 0.11 words them
    "overflow": "'max_tokens' or 'max_completion_tokens' is too large: 16384. This model's maximum context length "
                "is 128000 tokens and your request has 117561 input tokens (16384 > 128000 - 117561). None",
    "other": "At most 64 image(s) may be provided in one prompt.",
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        last = req["messages"][-1]
        images = sum(1 for part in last.get("content", []) if isinstance(part, dict) and part.get("type") == "image_url")
        turn = sum(1 for m in req["messages"] if m["role"] == "assistant")
        CALLS.append({"images": images, "turn": turn, "temperature": req.get("temperature")})
        if ERROR["mode"] != "none" and turn >= 1:
            self._send(400, {"error": {"message": MESSAGES[ERROR["mode"]], "type": "BadRequestError", "param": None,
                                       "code": 400}})
            return
        action = "ROTATION(90, 0)" if turn == 0 else "STOP(10)"
        content = json.dumps({"THOUGHT": "smoke test", "ACTION": action})
        self._send(200, {"id": "x", "object": "chat.completion", "created": 0, "model": req["model"],
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                      "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--error", choices=("none", "overflow", "other"), default="none")
    args = ap.parse_args()
    ERROR["mode"] = args.error
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    copy, diff = R.make_copy(run_dir)
    tasks = R.run_tasks(copy, "fake", base_url, 0.0, blind=False, workers=2, limit=args.limit)
    res = R.score(copy, tasks)
    res["server_alive_at_end"] = True
    server.shutdown()
    report = {"tasks": tasks, "calls": len(CALLS), "calls_with_one_image": sum(c["images"] == 1 for c in CALLS),
              "temperatures": sorted({c["temperature"] for c in CALLS}, key=str), "counts": dict(R.COUNTS),
              "n_scored": res["n_scored"], "total_sr": res["total_sr"], "mean_steps": res["mean_steps"],
              "unparseable_steps": res["unparseable_steps"], "api_failed_steps": res["api_failed_steps"],
              "context_overflow_steps": res["context_overflow_steps"], "api_failed_tasks": res["api_failed_tasks"],
              "valid": R.is_valid(res), "settings_diff": diff}
    print(json.dumps(report, indent=1))
    ok = report["calls"] > 0 and report["calls_with_one_image"] == report["calls"] and res["n_scored"] == len(tasks)
    if args.error == "none":
        ok = ok and res["api_failed_steps"] == 0 and report["valid"]
    elif args.error == "overflow":
        ok = ok and res["api_failed_steps"] > 0 and res["context_overflow_steps"] == res["api_failed_steps"] \
            and report["valid"]
    else:
        ok = ok and res["api_failed_steps"] > 0 and res["context_overflow_steps"] == 0 and not report["valid"]
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
