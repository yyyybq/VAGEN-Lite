"""E3VS closed-loop eval of one start / tier / decoding mode (PREREG_E3VS_PILOT sections 4-8, 12).

    PY_Q3 -m vpd.e3vs.run --start Q3S --tier 100 --mode u[,c] [--model-dir LOCAL --adapter-meta ADAPTER/meta.json]
        [--split test] [--out-root OUT/e3vs/runs/pilot/Q3S/evals] [--concurrency 24] [--gsplat-probe RUN/probe.json]
    PY_Q3 -m vpd.e3vs.run --start Q3S --tier raw --mode c --renderer fake --fake-server --limit 3 --run-dir DIR

One process tree (sco job, PY_Q3, as root): vpd.common.vllm_server_vpd.VLLMServer (gpu_memory_utilization 0.80,
1 image per request, xgrammar) is started FIRST, then the gsplat scenes are loaded in this process (render.py), so
vLLM sees the free memory it asks for.  --base-url uses an already running server; --fake-server starts the
built-in FakeServer (tests and login smoke; no model).

--mode u,c runs both decoding modes of the same weights in ONE server session: the episodes of both modes go
through one pool, scene-major (scene, mode, episode), so each scene's PLY and mesh are loaded once for both; each
mode still gets its own run dir, manifest, audit and results (identical to two single-mode runs but for timing).
Scene objects (renderer, collider) live in a refcounted cache: an episode holds its scene's objects from its
first to its last decision, a scene with an episode in flight is never evicted, and at most SCENE_CACHE idle
scenes stay resident; so a scene-major run loads every scene exactly once (manifest scene_loads).
Preflight, before any server starts: with the frozen collider the mesh validation (OUT/e3vs/mesh/
validation.json) must pass and every scene needs its collision mesh with the validated sha256 (a mesh set other
than the validated one is refused), with gsplat every scene needs a non-empty PLY; otherwise the run is refused
(SystemExit), no GPU time is spent.
SIGTERM (the payload's step timeout) aborts promptly: queued episodes are cancelled, running ones stop at their
next decision, the vLLM server is shut down, the process exits 143, no results.json is written.

Per episode (official evaluator.py semantics, restrict_collision): up to 25 decisions, a STOP counts as one;
decision t renders the current pose (512x512 PNG), sends [system, user(text + image)] (the frozen prompt of
make_sft.Prompts, no history), greedy, max_tokens 128.  The reply is parsed with the official lenient parser
(agents/base_agent.py _parse_response_to_action, non-think: "action: <name>" in the last 5 lines, then anywhere,
then 'stop' / an action name in the last 200 characters, else move_forward with error=True; an empty reply is
stop with error=True as agents/qwen.py).  A move whose segment collides under the frozen collision model, the
official E3VS Collisions on the ScanNet++ mesh (--collides, default vpd.e3vs.mesh_collision:load_collider), is
skipped: the pose stays, the step is consumed and the next prompt says "Collision occurred".  Rotations are never
tested (the official ray of a rotation has length 0).
Modes: u = unconstrained (PRIMARY, B12); c = constrained, every request carries
extra_body {"structured_outputs": {"regex": make_sft.REGEX}} (THOUGHT empty; the E2 mechanism check).  Both modes
record per decision: lenient action and error flag, strict regex fullmatch, collision, finish reason, tokens.

RUN dir (--run-dir, else --out-root/<pilot__START__TIER__MODE>): episodes.jsonl (one line per finished episode:
poses, actions, collisions, parse flags, replies), prompt_dump.jsonl (the first 3 calls of the first episode,
images as paths), images/ (the PNGs of the first --save-images episodes), manifest.json (git, E3VS commit
bb7e603c + episode md5s, PLY sha256 and mesh sha256 per scene, the collision model (official file sha256,
tolerance, the validated mesh_set_sha256), prompt template sha256, decoding, server, code (code.e3vs_files: the
sha256 of vpd/e3vs/*.py and vpd/jobs/e3vs_*.sh, recording only), adapter, gsplat version + the job's B18 probe
(--gsplat-probe, else $VPD_GSPLAT_PROBE), scene load counts and seconds), audit.json (image tokens from the
processor's image_grid_thw; HF vs server prompt tokens, B4/B5), and results.json from vpd.e3vs.summary.score
(results_invalid.json when an episode is missing or failed).  An existing results.json is refused (the job payload
skips the step instead).
"""

from __future__ import annotations

import argparse
import base64
import collections
import contextlib
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from vpd import paths
from vpd.e3vs import ACTIONS, MAX_DECISIONS, TRANSLATIONS, check_repo, code_meta, load_episodes, repo_dir, \
    sha256_file
from vpd.e3vs import geometry as G
from vpd.e3vs import make_sft as M
from vpd.e3vs import mesh_collision as MC

TIERS = ("raw", "0W", "100", "300", "ALL")  # 300 / ALL: round 4 (OUT/plan/ROUND4.md, PREREG_E3VS_PILOT_V2)
MODES = ("u", "c")
MODE_NAMES = {"u": "unconstrained (primary)", "c": "constrained (secondary, E2 mechanism check)"}
DEFAULT_COLLIDES = MC.FACTORY  # vpd.e3vs.mesh_collision:load_collider
MAX_TOKENS = 128  # official max_output_tokens (single_frame.yaml)
CONCURRENCY = 24
SCENE_CACHE = 2  # idle scenes kept resident (scenes with an episode in flight are never evicted); scene-major
#                  order needs no idle slot to load each scene once, so this only bounds GPU memory (the
#                  21-scene test simulation peaks at 5-6 resident scenes with 1 idle slot, 8-9 with 4)
SERVER_OVERRIDES = {"gpu_memory_utilization": 0.80, "max_images": 1, "max_model_len": 16384}
EXPECTED_IMAGE_TOKENS = 258  # 512x512 -> grid [1, 32, 32] -> 256 + vision_start/end (PREREG section 3, E0)
PROMPT_DUMP_CALLS = 3
_ACTION_RE = re.compile(r"action\s*:\s*([a-z_]+)", re.IGNORECASE)


def eval_name(start: str, tier: str, mode: str) -> str:
    return f"pilot__{start}__{tier}__{mode}"


# ------------------------------------------------------------------------------------------------------------
# parsing
# ------------------------------------------------------------------------------------------------------------
def parse_lenient(text: str, available=ACTIONS) -> dict:
    """The official non-think parser (module docstring); {"action", "error", "empty"}."""
    text = (text or "").strip()
    if not text:
        return {"action": "stop", "error": True, "empty": True}
    for line in reversed(text.split("\n")[-5:]):
        m = _ACTION_RE.search(line.strip())
        if m and m.group(1).strip() in available:
            return {"action": m.group(1).strip(), "error": False, "empty": False}
    m = _ACTION_RE.search(text)
    if m and m.group(1).strip() in available:
        return {"action": m.group(1).strip(), "error": False, "empty": False}
    last = text.lower()[-200:]
    if "stop" in last:
        return {"action": "stop", "error": False, "empty": False}
    for a in available:
        if a.replace("_", " ") in last or a in last:
            return {"action": a, "error": False, "empty": False}
    return {"action": "move_forward", "error": True, "empty": False}


def regex_ok(text: str) -> bool:
    return re.fullmatch(M.REGEX, text or "") is not None


# ------------------------------------------------------------------------------------------------------------
# server I/O
# ------------------------------------------------------------------------------------------------------------
def chat(base_url: str, model: str, messages: list, max_tokens: int, regex: str | None = None,
         timeout: float = 600.0, retries: int = 3) -> dict:
    body = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": int(max_tokens)}
    if regex is not None:
        body["structured_outputs"] = {"regex": regex}
    data = json.dumps(body).encode()
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(f"{base_url}/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            ch = out["choices"][0]
            u = out.get("usage") or {}
            return {"content": ch["message"].get("content") or "", "finish_reason": ch.get("finish_reason"),
                    "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens")}
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError, KeyError,
                json.JSONDecodeError) as e:  # HTTPException: IncompleteRead, BadStatusLine, RemoteDisconnected
            last = e
            if isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500:
                break
            time.sleep(2.0 * (2 ** k))
    raise RuntimeError(f"chat request failed after {retries} tries: {type(last).__name__}: {last}")


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


class SceneCache:
    """Thread-safe, refcounted cache of per-scene objects (renderers, colliders).

    ``with cache.hold(scene) as obj:`` pins the scene for the block.  A pinned scene is never evicted; of the idle
    (unpinned) scenes at most ``capacity`` stay, least recently acquired first out.  A scene is loaded by one
    thread, outside the cache lock (other scenes stay available; other threads wanting the same scene wait for
    it).  A failed load is not cached (the next acquire tries again) and raises in the acquiring thread.
    """

    def __init__(self, factory, capacity: int = SCENE_CACHE):
        self.factory, self.capacity = factory, max(0, int(capacity))
        self.d = collections.OrderedDict()
        self.refs = collections.Counter()
        self.loading = {}
        self.lock = threading.Lock()
        self.loaded = []          # scene ids in load order (a repeat = a reload)
        self.load_seconds = {}    # scene -> seconds of its (last) load
        self.max_resident = 0
        self.evicted = 0

    def acquire(self, scene_id: str):
        while True:
            with self.lock:
                if scene_id in self.d:
                    self.d.move_to_end(scene_id)
                    self.refs[scene_id] += 1
                    return self.d[scene_id]
                ev = self.loading.get(scene_id)
                owner = ev is None
                if owner:
                    ev = self.loading[scene_id] = threading.Event()
            if not owner:
                ev.wait()
                continue
            t0 = time.time()
            try:
                obj = self.factory(scene_id)
            except BaseException:
                with self.lock:
                    del self.loading[scene_id]
                ev.set()
                raise
            with self.lock:
                self.d[scene_id] = obj
                self.refs[scene_id] += 1
                self.loaded.append(scene_id)
                self.load_seconds[scene_id] = round(time.time() - t0, 3)
                del self.loading[scene_id]
                self.max_resident = max(self.max_resident, len(self.d))
                self._evict()
            ev.set()
            return obj

    def release(self, scene_id: str) -> None:
        with self.lock:
            self.refs[scene_id] -= 1
            if self.refs[scene_id] <= 0:
                del self.refs[scene_id]
            self._evict()

    @contextlib.contextmanager
    def hold(self, scene_id: str):
        obj = self.acquire(scene_id)
        try:
            yield obj
        finally:
            self.release(scene_id)

    def _evict(self) -> None:  # under self.lock
        idle = [k for k in self.d if self.refs.get(k, 0) <= 0]
        for k in idle[:max(0, len(idle) - self.capacity)]:
            del self.d[k]
            self.evicted += 1

    def stats(self) -> dict:
        with self.lock:
            secs = list(self.load_seconds.values())
            return {"n_loads": len(self.loaded), "n_scenes": len(set(self.loaded)),
                    "reloads": len(self.loaded) - len(set(self.loaded)), "evicted": self.evicted,
                    "max_resident": self.max_resident, "capacity_idle": self.capacity,
                    "load_seconds_total": round(sum(secs), 2), "load_seconds_max": max(secs) if secs else None,
                    "load_seconds": dict(self.load_seconds)}


# ------------------------------------------------------------------------------------------------------------
# episode loop
# ------------------------------------------------------------------------------------------------------------
class Aborted(RuntimeError):
    """The run was stopped (SIGTERM / an exception in the main thread) while this episode was running."""


class Ctx:
    """Per-mode settings of an eval; the caches, render lock and stop event are shared by the modes of a session."""

    def __init__(self, *, prompts, base_url, model, mode, renderers, colliders, run_dir, max_decisions,
                 max_tokens, save_images: int, chat_fn=chat, render_lock=None, stop=None):
        self.prompts, self.system = prompts, prompts.system()
        self.base_url, self.model, self.mode = base_url, model, mode
        self.regex = M.REGEX if mode == "c" else None
        self.renderers, self.colliders = renderers, colliders
        self.run_dir, self.max_decisions, self.max_tokens = Path(run_dir), int(max_decisions), int(max_tokens)
        self.save_images = int(save_images)
        self.chat_fn = chat_fn
        self.render_lock = render_lock or threading.Lock()  # one GPU render at a time
        self.stop = stop or threading.Event()
        self.write_lock = threading.Lock()

    def render_png(self, renderer, c2w) -> bytes:
        from vpd.e3vs import render as RD

        with self.render_lock:
            rgb = renderer.render_rgb(c2w, M.IMAGE_W, M.IMAGE_H)
        return RD.to_png(rgb)

    def append(self, name: str, rec: dict) -> None:
        with self.write_lock:
            with open(self.run_dir / name, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_episode(ep: dict, pos: int, ctx: Ctx) -> dict:
    """One episode; never raises (a failure is recorded as status 'error')."""
    t0 = time.time()
    rec = {"episode_index": ep["episode_index"], "scene_id": ep["scene_id"], "question_type": ep["question_type"],
           "start_viewpoint": ep["start_viewpoint"], "status": "ok", "error": None, "steps": [],
           "poses": [np.asarray(ep["start_c2w"]).tolist()], "stopped": False, "stop_step": None}
    try:
        with contextlib.ExitStack() as hold:  # the scene's renderer / collider, pinned for the whole episode
            if ctx.stop.is_set():
                raise Aborted("run aborted before this episode started")
            renderer = hold.enter_context(ctx.renderers.hold(ep["scene_id"]))
            collide = hold.enter_context(ctx.colliders.hold(ep["scene_id"])) if ctx.colliders is not None else None
            _episode_loop(ep, pos, ctx, rec, renderer, collide)
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"[:1000]
        rec["traceback"] = traceback.format_exc()[-3000:]
    rec["n_decisions"] = len(rec["steps"])
    rec["seconds"] = round(time.time() - t0, 2)
    ctx.append("episodes.jsonl", rec)
    return rec


def _episode_loop(ep: dict, pos: int, ctx: Ctx, rec: dict, renderer, collide) -> None:
    c2w = np.array(ep["start_c2w"], dtype=np.float64)
    last, coll = None, False
    for t in range(ctx.max_decisions):
        if ctx.stop.is_set():
            raise Aborted(f"run aborted at decision {t}")
        png = ctx.render_png(renderer, c2w)
        img_path = None
        if pos < ctx.save_images:
            img_path = ctx.run_dir / "images" / f"{ep['episode_index']:05d}" / f"t{t:02d}.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img_path.write_bytes(png)
        text = ctx.prompts.user(ep["question"], t, c2w[:3, 3], last, coll)
        msgs = [{"role": "system", "content": ctx.system},
                {"role": "user", "content": [{"type": "text", "text": text},
                                             {"type": "image_url", "image_url": {"url": data_url(png)}}]}]
        r = ctx.chat_fn(ctx.base_url, ctx.model, msgs, ctx.max_tokens, ctx.regex)
        p = parse_lenient(r["content"])
        step = {"t": t, "action": p["action"], "error": p["error"], "empty": p["empty"],
                "regex_ok": regex_ok(r["content"]), "collision": False, "finish_reason": r["finish_reason"],
                "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                "reply": r["content"]}
        if pos == 0 and t < PROMPT_DUMP_CALLS:
            ctx.append("prompt_dump.jsonl", {
                "episode_index": ep["episode_index"], "t": t, "prompt_tokens": r["prompt_tokens"],
                "messages": [msgs[0], {"role": "user", "content": [
                    msgs[1]["content"][0], {"type": "image_url", "path": None if img_path is None else
                                            str(img_path.relative_to(ctx.run_dir))}]}]})
        if p["action"] == "stop":
            rec["steps"].append(step)
            rec["poses"].append(c2w.tolist())
            rec["stopped"], rec["stop_step"] = True, t
            break
        nxt = G.apply_action(c2w, p["action"])
        if collide is not None and p["action"] in TRANSLATIONS and collide(c2w[:3, 3].copy(), nxt[:3, 3].copy()):
            nxt = c2w.copy()
            step["collision"] = True
        rec["steps"].append(step)
        rec["poses"].append(nxt.tolist())
        last, coll, c2w = p["action"], step["collision"], nxt


# ------------------------------------------------------------------------------------------------------------
# fake OpenAI server (tests, login smoke)
# ------------------------------------------------------------------------------------------------------------
def default_fake_policy(step: int, user_text: str, regex: str | None) -> str:
    seq = ("move_forward", "turn_left", "look_down", "move_up")
    return M.reply(seq[step] if step < len(seq) else "stop")


class FakeServer:
    """A tiny OpenAI-compatible chat server; policy(step, user_text, regex) -> reply text.  Records requests."""

    def __init__(self, policy=default_fake_policy):
        self.policy = policy
        self.requests, self.problems = [], []
        self.lock = threading.Lock()
        self.httpd = None

    def start(self) -> str:
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, payload):
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
                user = req["messages"][-1]["content"]
                text = "".join(p.get("text", "") for p in user if p.get("type") == "text")
                n_img = sum(p.get("type") == "image_url" for p in user)
                m = re.search(r"Current step: (\d+)", text)
                step = int(m.group(1)) if m else -1
                so = req.get("structured_outputs")
                regex = so.get("regex") if isinstance(so, dict) else None
                content = srv.policy(step, text, regex)
                with srv.lock:
                    srv.requests.append({"step": step, "text": text, "regex": regex, "n_images": n_img,
                                         "max_tokens": req.get("max_tokens"), "temperature": req.get("temperature"),
                                         "reply": content, "system": req["messages"][0]["content"]})
                    if regex is not None and re.fullmatch(regex, content) is None:
                        srv.problems.append(f"step {step}: reply {content!r} does not fullmatch the regex")
                pt = 300 + len(text) // 4
                self._send(200, {"id": "x", "object": "chat.completion", "created": 0, "model": req["model"],
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                              "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": pt, "completion_tokens": 8, "total_tokens": pt + 8}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def stop(self):
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()


# ------------------------------------------------------------------------------------------------------------
# provenance and audits
# ------------------------------------------------------------------------------------------------------------
def git_facts() -> dict:
    """Read-only git facts of paths.WORKTREE.  In a DECISIONS G1 code snapshot (SNAPSHOT.json; no .git, no git in
    the job image): commit = the snapshot's, vpd_status [] (prepare refused a dirty tree), and prereg_commit from
    $RUN/prereg_commit.txt when the job's prereg guard wrote it (else as git read it, None); read says so.  Live
    worktree: unchanged."""
    from vpd import snapshot

    try:
        snap = snapshot.snapshot_state(paths.WORKTREE)
    except snapshot.SnapshotError:  # an unreadable record: the live path below (facts None in a job)
        snap = {"mode": "unreadable"}
    if snap["mode"] == "snapshot":
        pre, src = None, None
        run = os.environ.get("RUN")
        f = Path(run) / "prereg_commit.txt" if run else None
        if f is not None and f.is_file():
            c = f.read_text().strip()
            if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", c):
                pre, src = c, str(f)
        return {"commit": snap["commit"], "vpd_status": [], "prereg_commit": pre, "read": snapshot.MARKER,
                "prereg_commit_from": src}

    def g(*a):
        try:
            r = subprocess.run(["git", "--no-optional-locks", "-c", "safe.directory=*", "-C", str(paths.WORKTREE),
                                *a], capture_output=True, text=True, timeout=30)
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    st = g("status", "--porcelain", "--", "vpd")
    return {"commit": g("rev-parse", "HEAD"), "vpd_status": None if st is None else st.splitlines(),
            "prereg_commit": g("log", "-1", "--format=%H", "--", "vpd/PREREG_E3VS_PILOT.md") or None}


def adapter_info(meta_path: Path | None, start: str) -> dict | None:
    if meta_path is None:
        return None
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text())
    if meta.get("start") not in (None, start):
        raise SystemExit(f"{meta_path}: adapter of start {meta.get('start')!r}, not {start!r}")
    pt = meta_path.parent / "adapter.pt"
    return {"meta": str(meta_path), "start": meta.get("start"), "phase": meta.get("phase"), "seed": meta.get("seed"),
            "epochs": meta.get("epochs"), "code_sha256": meta.get("code_sha256"),
            "init_adapter": meta.get("init_adapter"), "adapter_sha256": sha256_file(pt) if pt.is_file() else None}


def image_token_audit(run_dir: Path, processor_dir: Path, max_images: int = 8) -> dict:
    """Tokens per image from the processor's image_grid_thw on saved PNGs (B4), and HF vs server prompt tokens of
    the dumped calls (B5).  Processor files only."""
    import torch
    from PIL import Image

    from vpd import starts

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    proc = starts.load_processor(path=processor_dir, name=None) if (Path(processor_dir) /
                                                                    "preprocessor_config.json").exists() else None
    if proc is None:
        return {"error": f"no processor files in {processor_dir}"}
    ip = proc.image_processor
    merge = int(getattr(ip, "merge_size", 2))
    pngs = sorted((run_dir / "images").glob("*/t*.png"))[:max_images]
    samples = []
    for p in pngs:
        with Image.open(p) as im:
            grid = ip(images=[im.convert("RGB")], return_tensors="pt")["image_grid_thw"][0].tolist()
        n = int(np.prod(grid))
        samples.append({"path": str(p.relative_to(run_dir)), "grid_thw": grid, "tokens": n // (merge * merge) + 2})
    out = {"processor_dir": str(processor_dir), "n": len(samples),
           "tokens_per_image": sorted({s["tokens"] for s in samples}), "expected": EXPECTED_IMAGE_TOKENS,
           "samples": samples}
    rows = []
    dump = run_dir / "prompt_dump.jsonl"
    if dump.exists():
        for line in open(dump, encoding="utf-8"):
            r = json.loads(line)
            ipath = r["messages"][1]["content"][1].get("path")
            if not ipath:
                continue
            with Image.open(run_dir / ipath) as im:
                img = im.convert("RGB")
            msgs = [{"role": "system", "content": r["messages"][0]["content"]},
                    {"role": "user", "content": [{"type": "text", "text": r["messages"][1]["content"][0]["text"]},
                                                 {"type": "image"}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = proc(text=[text], images=[img], return_tensors="pt")["input_ids"]
            rows.append({"t": r["t"], "hf_prompt_tokens": int(ids.shape[-1]), "server_prompt_tokens": r["prompt_tokens"]})
    out["prompt_parity"] = {"n": len(rows), "equal": bool(rows) and all(
        x["hf_prompt_tokens"] == x["server_prompt_tokens"] for x in rows), "rows": rows}
    return out


# ------------------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------------------
def select_episodes(split: str, repo: Path, limit: int | None, ids: str | None) -> list[dict]:
    eps = load_episodes(split, repo)
    if ids:
        want = {int(x) for x in ids.split(",") if x}
        eps = [e for e in eps if e["episode_index"] in want]
        if len(eps) != len(want):
            raise SystemExit(f"unknown episode ids in --episodes for split {split}")
    if limit:
        eps = eps[:limit]
    return sorted(eps, key=lambda e: (e["scene_id"], e["episode_index"]))  # scene-major: few PLY loads


def scene_provenance(scenes, renderer: str, collides: str | None) -> dict:
    from vpd.e3vs import render as RD

    val = MC.load_validation() if collides == DEFAULT_COLLIDES else None
    vsc = (val or {}).get("scenes") or {}
    out = {}
    for s in sorted(set(scenes)):
        e = {"ply_sha256": None, "ply_present": None, "mesh_path": None, "mesh_sha256": None,
             "mesh_validated_sha256": None}
        if renderer == "gsplat":
            p = RD.ply_path(s)
            e["ply_present"] = p.is_file() and p.stat().st_size > 0
            e["ply_sha256"] = RD.ply_sha256(s) if e["ply_present"] else None
        if collides == DEFAULT_COLLIDES:
            p = MC.mesh_path(s)
            e["mesh_path"] = str(p)
            e["mesh_sha256"] = MC.mesh_file_sha256(p) if p.is_file() else None  # re-hashed; load_collider reuses it
            e["mesh_validated_sha256"] = (vsc.get(s) or {}).get("sha256")
        out[s] = e
    return out


def collision_provenance(collides: str | None) -> dict | None:
    """The frozen collision model of the manifest: official code facts, tolerance, the validation and its key."""
    if collides != DEFAULT_COLLIDES:
        return None
    vp = MC.validation_path()
    val = MC.load_validation()
    return {"factory": MC.FACTORY, "official_file": str(MC.METRICS_REL), "official_sha256": MC.METRICS_SHA256,
            "official_class": MC.OFFICIAL_CLASS, "tolerance_m": MC.TOLERANCE_M, "in_obstacle_m": MC.IN_OBSTACLE_M,
            "validation": str(vp), "validation_pass": None if val is None else val.get("pass"),
            "validation_problems": None if val is None else MC.record_problems(val),
            "mesh_set_sha256": None if val is None else val.get("mesh_set_sha256"),
            "validation_official": None if val is None else val.get("official"),
            "frame": MC.frame_facts(val)}  # descriptive: frame_ok, warnings, suspect scenes, any ack-frame record


def preflight(prov: dict, renderer: str, collides: str | None, coll: dict | None = None) -> list[str]:
    """Problems that would make every episode of a scene fail; checked before any server starts."""
    bad = []
    if collides == DEFAULT_COLLIDES:
        coll = coll or {}
        if coll.get("validation_pass") is not True or coll.get("validation_problems"):
            bad.append(f"no passing mesh validation of this collision model ({coll.get('validation')}: pass "
                       f"{coll.get('validation_pass')!r}, {coll.get('validation_problems')}; e3vs-prep mesh_val)")
        miss = [s for s, e in prov.items() if e["mesh_sha256"] is None]
        if miss:
            bad.append(f"{len(miss)} scenes without a collision mesh (e.g. {miss[:3]}; the user downloads them, "
                       "copied to OUT/e3vs/mesh, mesh_verify)")
        other = [s for s, e in prov.items() if e["mesh_sha256"] and e["mesh_sha256"] != e["mesh_validated_sha256"]]
        if other:
            bad.append(f"{len(other)} scenes whose mesh is not the validated one (e.g. {other[:3]}; the mesh set "
                       "key differs from the validation's)")
    if renderer == "gsplat":
        miss = [s for s, e in prov.items() if not e["ply_present"]]
        if miss:
            bad.append(f"{len(miss)} scenes without a PLY (e.g. {miss[:3]}; download_splats.sh, user OK)")
    return bad


def gsplat_facts(renderer: str, probe: Path | None) -> dict | None:
    """B18 in the manifest: the gsplat version and the job's probe (did gsplat compile on the spot, how long)."""
    if renderer != "gsplat":
        return None
    import importlib.metadata

    try:
        ver = importlib.metadata.version("gsplat")
    except importlib.metadata.PackageNotFoundError:
        ver = None
    out = {"version": ver, "probe_path": None if probe is None else str(probe), "probe": None,
           "TORCH_EXTENSIONS_DIR": os.environ.get("TORCH_EXTENSIONS_DIR")}
    if probe is not None and Path(probe).is_file():
        try:
            out["probe"] = json.loads(Path(probe).read_text())
        except (OSError, json.JSONDecodeError) as e:
            out["probe"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    return out


def resolve_start(start: str) -> dict:
    from vpd import starts

    e = starts.resolve(start)
    return {"start": start, "code": e["code"], "served": e["code"] or start, "path": e["path"]}


def parse_modes(spec: str) -> list[str]:
    ms = [m.strip() for m in str(spec).split(",") if m.strip()]
    if not ms or any(m not in MODES for m in ms) or len(set(ms)) != len(ms):
        raise SystemExit(f"--mode wants u, c or u,c (got {spec!r})")
    return ms


@contextlib.contextmanager
def _sigterm_aborts():
    """In the main thread: SIGTERM raises SystemExit(143) there, so the finally blocks stop vLLM."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _term(signum, frame):
        raise SystemExit(143)

    old = signal.signal(signal.SIGTERM, _term)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, old)


def run_pool(tasks, ctxs: dict, concurrency: int, stop: threading.Event) -> dict:
    """tasks [(mode, pos, ep)] -> {mode: [episode record]}.  On an exception in this (main) thread, e.g. the
    SystemExit of SIGTERM, the queue is cancelled and the running episodes stop at their next decision."""
    pool = ThreadPoolExecutor(max_workers=max(1, int(concurrency)))
    ok = False
    try:
        futs = [(m, pool.submit(run_episode, ep, pos, ctxs[m])) for m, pos, ep in tasks]
        out = {m: [] for m in ctxs}
        for m, f in futs:
            out[m].append(f.result())
        ok = True
        return out
    finally:
        if not ok:
            stop.set()
        pool.shutdown(wait=ok, cancel_futures=not ok)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", required=True, help="a start of vpd/starts.json")
    ap.add_argument("--tier", required=True, choices=TIERS)
    ap.add_argument("--mode", required=True,
                    help="u = unconstrained (primary), c = constrained, u,c = both in one server session")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--episodes", default=None, help="comma list of episode indices")
    ap.add_argument("--model-dir", type=Path, default=None, help="merged weights on job-local disk")
    ap.add_argument("--adapter-meta", type=Path, default=None, help="provenance of the merged adapter")
    ap.add_argument("--out-root", type=Path, default=None, help="default OUT/e3vs/runs/pilot/<START>/evals")
    ap.add_argument("--run-dir", type=Path, default=None, help="single mode only (default OUT-ROOT/<eval name>)")
    ap.add_argument("--renderer", default="gsplat", choices=("gsplat", "fake"))
    ap.add_argument("--collides", default=DEFAULT_COLLIDES, help="MODULE:FUNC factory, or 'none' (fake only)")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--scene-cache", type=int, default=None, help=f"idle scenes kept (default {SCENE_CACHE})")
    ap.add_argument("--max-decisions", type=int, default=MAX_DECISIONS)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--save-images", type=int, default=3, help="keep the PNGs of the first N episodes")
    ap.add_argument("--base-url", default=None, help="use a running OpenAI-compatible server")
    ap.add_argument("--fake-server", action="store_true", help="built-in fake server (no model)")
    ap.add_argument("--audit-processor", default="auto", help="'auto' (model dir or the start's), a dir, or ''")
    ap.add_argument("--floors", type=Path, default=None, help="default OUT/e3vs/floors/floors_mesh.json if present")
    ap.add_argument("--opt-len", type=Path, default=None, help="default OUT/e3vs/bfs/mesh/<split>/opt_len.json")
    ap.add_argument("--gsplat-probe", type=Path, default=None,
                    help="the job's B18 probe JSON (default $VPD_GSPLAT_PROBE), copied into manifest.gsplat")
    ap.add_argument("--gpu-memory-utilization", type=float, default=SERVER_OVERRIDES["gpu_memory_utilization"])
    ap.add_argument("--max-model-len", type=int, default=SERVER_OVERRIDES["max_model_len"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repo", type=Path, default=None)
    a = ap.parse_args(argv)
    modes = parse_modes(a.mode)

    from vpd.e3vs import render as RD
    from vpd.e3vs import summary as S
    from vpd.e3vs.bfs_demos import load_collider_factory

    if a.collides == "none" and a.renderer != "fake":
        raise SystemExit("--collides none is for the fake renderer only (the pilot uses the frozen mesh collision "
                         "model)")
    if a.adapter_meta is not None and a.model_dir is None:
        raise SystemExit("--adapter-meta needs --model-dir (the merged weights it describes)")
    if a.run_dir is not None and len(modes) > 1:
        raise SystemExit("--run-dir is for one mode; with --mode u,c use --out-root")
    info = resolve_start(a.start)
    if info["code"] is None and a.model_dir is None and not (a.fake_server or a.base_url):
        raise SystemExit(f"start {a.start} has no eval view; pass --model-dir")
    names = {m: eval_name(a.start, a.tier, m) for m in modes}
    root = a.out_root or paths.E3VS_RUNS / "pilot" / a.start / "evals"
    run_dirs = {m: Path(a.run_dir or Path(root) / names[m]).resolve() for m in modes}
    for rd in run_dirs.values():
        if (rd / "results.json").exists():
            raise SystemExit(f"{rd}/results.json exists; refusing to overwrite")
        for f in ("episodes.jsonl", "prompt_dump.jsonl"):
            if (rd / f).exists():
                raise SystemExit(f"{rd}/{f} exists from an unfinished run; move it away first")
    repo = Path(a.repo) if a.repo else repo_dir()
    e3vs = check_repo(repo, splits=(a.split,))
    eps = select_episodes(a.split, repo, a.limit, a.episodes)
    collides = None if a.collides == "none" else a.collides
    prov = scene_provenance([e["scene_id"] for e in eps], a.renderer, collides)
    coll = collision_provenance(collides)
    bad = preflight(prov, a.renderer, collides, coll)
    if bad:
        raise SystemExit("preflight refused (no server started): " + "; ".join(bad))
    prompts = M.Prompts(repo)
    factory = None if collides is None else load_collider_factory(collides)
    env_probe = os.environ.get("VPD_GSPLAT_PROBE")
    probe = a.gsplat_probe or (Path(env_probe) if env_probe else None)
    base = {
        "start": a.start, "code": info["code"], "tier": a.tier, "split": a.split, "n_episodes": len(eps),
        "episode_indices": [e["episode_index"] for e in eps] if (a.limit or a.episodes) else "all",
        "prompt": prompts.meta(), "renderer": a.renderer, "collides": collides, "collision": coll, "scenes": prov,
        "gsplat": gsplat_facts(a.renderer, probe),
        "e3vs": {**e3vs, "pinned_commit": paths.E3VS_COMMIT}, "model_dir": None if a.model_dir is None else
        str(a.model_dir), "adapter": adapter_info(a.adapter_meta, a.start), "concurrency": a.concurrency,
        "session": {"modes": modes, "evals": [names[m] for m in modes]},
        "git": git_facts(), "code": code_meta(Path(__file__), Path(M.__file__), Path(RD.__file__),
                                              Path(S.__file__), Path(G.__file__), Path(MC.__file__),
                                              e3vs_files=True),
        "argv": sys.argv[1:] if argv is None else list(argv),
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "uid": os.getuid()}
    manifests = {}
    for m in modes:
        run_dirs[m].mkdir(parents=True, exist_ok=True)
        manifests[m] = {"eval": names[m], **base, "mode": m, "mode_name": MODE_NAMES[m],
                        "decoding": {"temperature": 0.0, "max_tokens": a.max_tokens,
                                     "regex": M.REGEX if m == "c" else None,
                                     "parser": "official lenient (base_agent.py)", "max_decisions": a.max_decisions}}

    def write_manifests():
        for m in modes:
            (run_dirs[m] / "manifest.json").write_text(json.dumps(manifests[m], indent=1) + "\n")

    fake, srv = None, None
    t0 = time.time()
    cap = SCENE_CACHE if a.scene_cache is None else a.scene_cache
    renderers = SceneCache(lambda s: RD.make_renderer(a.renderer, s, device=a.device), cap)
    colliders = None if factory is None else SceneCache(factory, cap)
    with _sigterm_aborts():
        try:
            if a.fake_server:
                fake = FakeServer()
                base_url, served = fake.start(), "fake"
                server = {"fake_server": base_url}
            elif a.base_url:
                base_url, served, server = a.base_url.rstrip("/"), info["served"], {"base_url": a.base_url}
            else:
                from vpd.common.vllm_server_vpd import VLLMServer

                # open3d's dir is for this process only (the colliders): the vLLM server gets the env without it
                os.environ["PYTHONPATH"] = MC.scrub_o3d_pythonpath(os.environ.get("PYTHONPATH"))
                srv = VLLMServer(info["served"], log=run_dirs[modes[0]] / "vllm.log", model_dir=a.model_dir,
                                 gpu_memory_utilization=a.gpu_memory_utilization, max_images=1,
                                 max_model_len=a.max_model_len)
                server = srv.manifest()
                srv.__enter__()  # vLLM first: it must see the free GPU memory before gsplat allocates any
                base_url, served = srv.base_url, info["served"]
                server["ready_seconds"] = getattr(srv, "ready_seconds", None)
            for m in modes:
                manifests[m]["server"] = server
            write_manifests()
            stop, rlock = threading.Event(), threading.Lock()
            ctxs = {m: Ctx(prompts=prompts, base_url=base_url, model=served, mode=m, renderers=renderers,
                           colliders=colliders, run_dir=run_dirs[m], max_decisions=a.max_decisions,
                           max_tokens=a.max_tokens, save_images=a.save_images, render_lock=rlock, stop=stop)
                    for m in modes}
            tasks = sorted(((m, pos, ep) for m in modes for pos, ep in enumerate(eps)),
                           key=lambda x: (x[2]["scene_id"], modes.index(x[0]), x[2]["episode_index"]))
            recs = run_pool(tasks, ctxs, a.concurrency, stop)
            alive = srv.alive() if srv is not None else True
        finally:
            if srv is not None:
                srv.__exit__(None, None, None)
            if fake is not None:
                fake.stop()
    wall = round(time.time() - t0, 1)
    loads = {"renderers": renderers.stats(), "colliders": None if colliders is None else colliders.stats()}
    for m in modes:
        manifests[m].update(server_alive_at_end=alive, wall_seconds=wall,
                            n_failed=sum(r["status"] != "ok" for r in recs[m]), scene_loads=loads,
                            collision_loaded=MC.loaded_official() if collides == DEFAULT_COLLIDES else None)
        if fake is not None:
            manifests[m]["fake_server_problems"] = fake.problems
    write_manifests()

    audit_dir = None
    if a.audit_processor == "auto":
        from vpd import starts

        audit_dir = a.model_dir if a.model_dir and (a.model_dir / "preprocessor_config.json").exists() else \
            starts.processor_dir(a.start)
    elif a.audit_processor:
        audit_dir = Path(a.audit_processor)
    floors = a.floors or (paths.E3VS_OUT / "floors" / f"floors_{MC.TAG}.json")
    opt = a.opt_len or (paths.E3VS_BFS / MC.TAG / a.split / "opt_len.json")
    all_valid = True
    for m in modes:
        rd = run_dirs[m]
        if audit_dir is not None:
            try:
                audit = image_token_audit(rd, Path(audit_dir))
            except Exception as e:  # noqa: BLE001  (an audit failure is recorded; the gate's E0 then fails)
                audit = {"error": f"{type(e).__name__}: {e}"[:500]}
            (rd / "audit.json").write_text(json.dumps(audit, indent=1) + "\n")
        res = S.score(rd, repo=repo, floors=floors if floors.exists() else None,
                      opt_len=opt if opt.exists() else None)
        p = res["pooled"]
        all_valid = all_valid and res["valid"]
        print(f"[e3vs.run] {names[m]}: success {p['k']}/{p['n']} = {p['rate']}, valid {res['valid']}, "
              f"lenient-valid {res['parse']['lenient_valid_rate']}, regex-valid {res['parse']['regex_valid_rate']}, "
              f"{wall} s (session {','.join(modes)}), {loads['renderers']['n_loads']} scene loads -> {rd}",
              flush=True)
    return 0 if all_valid else 3


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
