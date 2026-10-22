"""Run / episode logging (red line 6: results tables are generated from these logs, never typed).

Per-step and per-call logs go to a LOCAL directory (many small writes; NFS is slow for that);
``finalize`` copies the per-episode summary and the event log to the NAS log directory.

Layout under <local_root>/<run_id>/episodes/<episode_id>/:
    events.jsonl     decisions (slot, return value), executor calls, provider calls
    steps.jsonl      one line per control step: step, t, source, action (23 floats)
    frames/          head-camera JPEGs at decisions and every ``frame_every`` steps
    summary.json     counters at episode end (also copied to <nas_root>/<run_id>/episodes/)
"""

from __future__ import annotations

import dataclasses
import enum
import json
import os
import shutil
import threading
import time
from typing import Any, Optional

import numpy as np


def to_jsonable(x: Any, depth: int = 0) -> Any:
    if depth > 8:
        return repr(x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: to_jsonable(getattr(x, f.name), depth + 1) for f in dataclasses.fields(x)}
    if isinstance(x, enum.Enum):
        return x.value
    if isinstance(x, np.ndarray):
        if x.size <= 64:
            return np.round(x.astype(np.float64), 5).tolist()
        return {"ndarray": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict) or hasattr(x, "items"):
        return {str(k): to_jsonable(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v, depth + 1) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


class RunRecorder:
    def __init__(
        self,
        run_id: str,
        local_root: str,
        nas_root: Optional[str] = None,
        frame_every: int = 300,
        save_frames: bool = True,
        tags: Optional[dict] = None,
    ):
        self.run_id = run_id
        self.local_dir = os.path.join(os.path.expanduser(local_root), run_id)
        self.nas_dir = os.path.join(os.path.expanduser(nas_root), run_id) if nas_root else None
        self.frame_every = int(frame_every)
        self.save_frames = save_frames
        self.tags = dict(tags or {})
        self._lock = threading.Lock()
        self._files: dict[str, dict] = {}
        os.makedirs(self.local_dir, exist_ok=True)

    # ------------------------------------------------------------------ files

    def _ep_dir(self, episode_id: str) -> str:
        return os.path.join(self.local_dir, "episodes", episode_id)

    def _write(self, episode_id: str, name: str, rec: dict) -> None:
        with self._lock:
            files = self._files.get(episode_id)
            if files is None:
                return
            f = files[name]
            f.write(json.dumps(rec, default=repr) + "\n")

    def write_run_file(self, name: str, obj: Any) -> str:
        path = os.path.join(self.local_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(obj), f, indent=2, sort_keys=True)
        if self.nas_dir:
            os.makedirs(self.nas_dir, exist_ok=True)
            shutil.copy2(path, os.path.join(self.nas_dir, name))
        return path

    # ------------------------------------------------------------------ loop hooks

    def episode_start(self, ctx, modules_desc: dict) -> None:
        d = self._ep_dir(ctx.episode_id)
        os.makedirs(os.path.join(d, "frames"), exist_ok=True)
        with self._lock:
            self._files[ctx.episode_id] = {
                "events": open(os.path.join(d, "events.jsonl"), "a", encoding="utf-8"),
                "steps": open(os.path.join(d, "steps.jsonl"), "a", encoding="utf-8"),
                "counters": {"provider_calls": 0, "provider_s": 0.0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
                             "provider_errors": 0, "decisions": 0},
                "t0": time.time(),
            }
        self._write(ctx.episode_id, "events", {"kind": "episode_start", "t": time.time(), "ctx": to_jsonable(ctx),
                                               "modules": modules_desc, "tags": self.tags})

    def step(self, state, obs, action: np.ndarray, source: str) -> None:
        eid = state.ctx.episode_id
        self._write(eid, "steps", {"step": obs.step, "t": round(obs.received_at, 4), "src": source,
                                   "a": np.asarray(action, np.float32).tolist()})  # exact float32 values (replayable)
        if self.save_frames and self.frame_every > 0 and obs.step % self.frame_every == 0:
            self.save_frame(eid, obs, tag="periodic")

    def decision(self, state, slot: str, out: Any, seconds: float) -> None:
        eid = state.ctx.episode_id
        with self._lock:
            files = self._files.get(eid)
            if files is not None:
                files["counters"]["decisions"] += 1
        self._write(eid, "events", {"kind": "decision", "slot": slot, "step": state.step, "seconds": round(seconds, 4),
                                    "out": to_jsonable(out)})

    def executor_call(self, state, chunk) -> None:
        self._write(state.ctx.episode_id, "events", {"kind": "executor", "step": state.step,
                                                     "instruction": chunk.instruction, "source": chunk.source,
                                                     "infer_ms": chunk.infer_ms, "n": int(np.asarray(chunk.actions).shape[0]),
                                                     "open_loop": chunk.open_loop_steps, "meta": to_jsonable(chunk.meta)})

    def provider_call(self, episode_id: str, rec: dict) -> None:
        with self._lock:
            files = self._files.get(episode_id)
            if files is not None:
                c = files["counters"]
                c["provider_calls"] += 1
                c["provider_s"] += float(rec.get("latency_s") or 0.0)
                usage = rec.get("usage") or {}
                c["tokens_in"] += int(usage.get("input_tokens") or 0)
                c["tokens_out"] += int(usage.get("output_tokens") or 0)
                c["cost_usd"] += float(rec.get("cost_usd") or 0.0)
                if rec.get("error"):
                    c["provider_errors"] += 1
        self._write(episode_id, "events", {"kind": "provider_call", **rec})

    def save_frame(self, episode_id: str, obs, tag: str) -> Optional[str]:
        img = obs.rgb.get("head") if obs.rgb else None
        if img is None:
            return None
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover
            return None
        path = os.path.join(self._ep_dir(episode_id), "frames", f"{obs.step:06d}_{tag}.jpg")
        Image.fromarray(np.asarray(img, dtype=np.uint8)).save(path, quality=85)
        return path

    def snapshot(self, state) -> None:
        """Write summary.json for a still-open episode (e.g. when the evaluator's connection closes;
        the 2026 evaluator may reconnect, so the episode itself stays open)."""
        eid = state.ctx.episode_id
        with self._lock:
            files = self._files.get(eid)
            if files is None:
                return
            files["events"].flush()
            files["steps"].flush()
            counters = dict(files["counters"])
        summary = self._summary(state, counters)
        summary["open"] = True
        self._write_summary(eid, summary)

    def _write_summary(self, eid: str, summary: dict) -> None:
        d = self._ep_dir(eid)
        with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if self.nas_dir:
            nd = os.path.join(self.nas_dir, "episodes", eid)
            os.makedirs(nd, exist_ok=True)
            shutil.copy2(os.path.join(d, "summary.json"), os.path.join(nd, "summary.json"))
            shutil.copy2(os.path.join(d, "events.jsonl"), os.path.join(nd, "events.jsonl"))

    def episode_end(self, state) -> dict:
        eid = state.ctx.episode_id
        with self._lock:
            files = self._files.pop(eid, None)
        if files is None:
            return {}
        summary = self._summary(state, files["counters"])
        files["events"].write(json.dumps({"kind": "episode_end", "t": time.time(), "summary": summary}, default=repr) + "\n")
        files["events"].close()
        files["steps"].close()
        self._write_summary(eid, summary)
        return summary

    def _summary(self, state, counters: dict) -> dict:
        eid = state.ctx.episode_id
        return {
            "episode_id": eid,
            "run_id": self.run_id,
            "tags": self.tags,
            "task_name": state.ctx.task_name,
            "task_id": state.ctx.task_id,
            "steps": state.step + 1,
            "executed_actions": state.executed_actions,
            "elapsed_s": round(state.elapsed_s(state.t_latest_obs), 3) if state.t_latest_obs else 0.0,
            "wallclock_budget_s": state.ctx.wallclock_budget_s,
            "provider_time_s": round(state.provider_time_s, 3),
            "subtasks": [to_jsonable(r) for r in state.completed]
            + ([{"subtask": to_jsonable(state.current_subtask), "outcome": "open"}] if state.current_subtask else []),
            "replans": state.replans,
            "recoveries": state.recoveries,
            "fallbacks": state.fallbacks,
            "plan_exhausted": state.plan_exhausted,
            "low_risk": state.low_risk,
            "subgoal_estimates": to_jsonable(state.subgoal_estimates),
            **counters,
        }
