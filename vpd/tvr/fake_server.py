"""A fake OpenAI-compatible chat server for the TVR harness tests (CONTRACT_wave2 5.2.5).  Stdlib only; no model.

    with FakeServer(policy="stop_at", stop_after=3) as srv:
        srv.base_url        # http://127.0.0.1:<port>/v1
        srv.requests        # every decoded /v1/chat/completions body, in arrival order

Serves GET /v1/models and POST /v1/chat/completions on 127.0.0.1 (ThreadingHTTPServer, keep-alive).  Replies:
  "cycle"     the n-th assistant turn of the request (n = number of assistant messages already in it) gets
              ACTIONS[n % 9] (so the 9th reply is Stop);
  "stop_at"   "Action: Stop" once n >= stop_after, else "Action: <first>" (default RotateRight);
  "junk"      unparseable text on even n, "action: movEahead please" on odd n (ignores any regex: tests of the
              unconstrained arm and of regex-mismatch accounting);
  callable    policy(body: dict, n: int) -> reply text.
Every reply of a request that carries structured_outputs.regex fullmatches that regex (a non-matching policy
reply becomes "Action: Stop"), except policy "junk".  usage.prompt_tokens = prompt_tokens(messages): the
characters of every text content + 222 per image part (a deterministic stand-in of the 222-token frames);
completion_tokens = len(reply) // 4 + 1.  fail_first=k answers the first k chat requests with HTTP 500 (retry
tests); delay_s sleeps before every chat reply; idle_close_s closes a keep-alive connection idle that long (as
vLLM's uvicorn does after VLLM_HTTP_TIMEOUT_KEEP_ALIVE = 5 s); kill() makes the server die (no reply on open
connections, no new connections, alive() False: dead-server tests).
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Union

from vpd.tvr import pilot_cfg

FAKE_IMAGE_TOKENS = 222
POLICIES = ("cycle", "stop_at", "junk")


def prompt_tokens(messages: List[Dict[str, Any]]) -> int:
    """The fake server's prompt length: text characters + FAKE_IMAGE_TOKENS per image part (string contents
    count as text).  Works on live request messages and on prompt-dump (redacted) messages alike."""
    n = 0
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            n += len(c)
        elif isinstance(c, list):
            for p in c:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "image_url":
                    n += FAKE_IMAGE_TOKENS
                elif p.get("type") == "text":
                    n += len(p.get("text") or "")
    return n


def n_assistant(messages) -> int:
    return sum(1 for m in messages or [] if m.get("role") == "assistant")


class FakeServer:
    fake = True  # run_tvr then audits with the fake token rules (image_tokens by arithmetic, prompt_tokens above)

    def __init__(self, policy: Union[str, Callable] = "cycle", port: Optional[int] = None, *,
                 model: Optional[str] = None, stop_after: int = 2, first: str = "RotateRight", fail_first: int = 0,
                 delay_s: float = 0.0, keep_bodies: bool = True, idle_close_s: Optional[float] = None):
        if not callable(policy) and policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES} or a callable")
        self.policy, self.port_req, self.model = policy, port, model
        self.stop_after, self.first = int(stop_after), first
        self.fail_first, self.delay_s, self.keep_bodies = int(fail_first), float(delay_s), keep_bodies
        self.idle_close_s = None if idle_close_s is None else float(idle_close_s)
        self.dead = False
        self.n_connections = 0
        self.requests: List[Dict[str, Any]] = []
        self.n_chat = 0
        self.lock = threading.Lock()
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.proc = None  # no process (run_tvr's fast shutdown checks server.proc)

    # -- replies
    def reply_for(self, body: Dict[str, Any]) -> str:
        msgs = body.get("messages") or []
        n = n_assistant(msgs)
        p = self.policy
        if callable(p):
            text = str(p(body, n))
        elif p == "cycle":
            text = "Action: " + pilot_cfg.ACTIONS[n % len(pilot_cfg.ACTIONS)]
        elif p == "stop_at":
            text = "Action: Stop" if n >= self.stop_after else "Action: " + self.first
        else:  # junk
            return "I am not sure where to go." if n % 2 == 0 else "action: movEahead please"
        rx = ((body.get("structured_outputs") or {}).get("regex")
              or ((body.get("extra_body") or {}).get("structured_outputs") or {}).get("regex"))
        if rx and re.fullmatch(rx, text) is None:
            text = "Action: Stop"
        return text

    def _handler(self):
        srv = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            timeout = srv.idle_close_s  # socket timeout: an idle keep-alive connection is closed

            def setup(self):
                super().setup()
                with srv.lock:
                    srv.n_connections += 1

            def log_message(self, *a):  # quiet
                pass

            def _send(self, code: int, obj: Dict[str, Any]) -> None:
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.rstrip("/").endswith("/v1/models"):
                    self._send(200, {"object": "list", "data": [{"id": srv.model or "fake", "object": "model"}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                if srv.dead:  # a dead server: the connection is dropped without a reply
                    self.close_connection = True
                    return
                if not self.path.rstrip("/").endswith("/v1/chat/completions"):
                    self._send(404, {"error": "not found"})
                    return
                try:
                    body = json.loads(raw)
                except ValueError:
                    self._send(400, {"error": "bad json"})
                    return
                with srv.lock:
                    srv.n_chat += 1
                    k = srv.n_chat
                    if srv.keep_bodies:
                        srv.requests.append(body)
                if k <= srv.fail_first:
                    self._send(500, {"error": "fake failure"})
                    return
                if srv.delay_s:
                    time.sleep(srv.delay_s)
                text = srv.reply_for(body)
                pt = prompt_tokens(body.get("messages"))
                self._send(200, {"id": f"fake-{k}", "object": "chat.completion", "model": body.get("model"),
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                              "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": pt, "completion_tokens": len(text) // 4 + 1,
                                           "total_tokens": pt + len(text) // 4 + 1}})

        return H

    # -- life cycle
    def start(self) -> "FakeServer":
        self.httpd = ThreadingHTTPServer(("127.0.0.1", int(self.port_req or 0)), self._handler())
        self.httpd.daemon_threads = True
        self.httpd.block_on_close = False  # stop() never waits for a handler blocked on an idle keep-alive socket
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        self.thread.start()
        self.started = time.time()
        self.ready_seconds = 0.0
        return self

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None

    def kill(self) -> None:
        """The server dies: open connections get no reply, new ones are refused, alive() is False.  Safe to call
        from a request handler (a policy callable): the shutdown runs in its own thread."""
        self.dead = True
        h = self.httpd
        if h is not None:
            def down():
                h.shutdown()
                h.server_close()

            threading.Thread(target=down, daemon=True).start()

    def alive(self) -> bool:
        return not self.dead and self.httpd is not None and self.thread is not None and self.thread.is_alive()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def manifest(self) -> Dict[str, Any]:
        return {"source": "vpd.tvr.fake_server.FakeServer", "fake": True,
                "policy": self.policy if isinstance(self.policy, str) else "callable",
                "port": getattr(self, "port", None), "cfg": {"max_model_len": None, "max_images": None}}

    def __enter__(self) -> "FakeServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def main(argv=None) -> int:
    """Serve until killed (tests that need the fake model in another process)."""
    import argparse
    import signal

    ap = argparse.ArgumentParser(description="fake OpenAI-compatible chat server for the TVR harness tests")
    ap.add_argument("--policy", default="stop_at", choices=POLICIES)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--stop-after", type=int, default=2)
    a = ap.parse_args(argv)
    srv = FakeServer(policy=a.policy, port=a.port, model=a.model, stop_after=a.stop_after,
                     keep_bodies=False).start()
    print(json.dumps({"base_url": srv.base_url}), flush=True)
    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    try:
        done.wait()
    except KeyboardInterrupt:
        pass
    srv.stop()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
