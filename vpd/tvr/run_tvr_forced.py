"""EXPLORATORY round 5, experiment B (forced opening actions): run_tvr eval with a task subset and forced opening
turns.  PY_Q3.  Not a run_tvr CODE_FILE: vpd/tvr/run_tvr.py, its code_sha256 (the registered harness 64edd575f36b...)
and every pilot_tvr guard stay as they are; this module imports run_tvr and is recorded on its own (provenance below).

    PY_Q3 -B -m vpd.tvr.run_tvr_forced eval <run_tvr eval options> [--task-subset IDS.json]
          [--forced FORCED.json --forced-k K[,K...] (with --arms c only)]
    PY_Q3 -B -m vpd.tvr.run_tvr_forced provenance [--out FILE]   # prints harness_r5_sha256
(replay / rescore are run_tvr's own.)  Without the three options an eval is run_tvr's: the same files, keys and values.

--task-subset IDS.json: a JSON list of task_ids (non-empty, distinct, all in the --task-set); the run uses these tasks
in task-set order; protocol "task_subset": {"sha256", "n"}.  --forced FORCED.json ({"schema": "vpd.tvr.forced/1",
"donor", "source", "prefix": {task_id: donor actions}}, vpd.tvr.intervene donors; an entry for every task of the run)
with --forced-k K[,K...] (distinct ints >= 0; both options or neither, only with --arms c): one run dir
DIR/<prefix>__<START>__<TIER>__c__f<K> per K (K = 0: no turn forced), all in the one server session / THOR pool;
protocol "forced": {"k", "file_sha256", "donor", "module_sha256", "harness_r5_sha256"} (so the protocol sha and the
resume rules differ per K, per donor file and per code).  Forcing (image arm c): prefix = the first K donor actions
after dropping every Stop and InvalidAction (all of them when fewer remain); turn t < len(prefix) sends no request:
the reply is "Action: " + prefix[t], parsed by the arm's own parser (TaskError "protocol" unless it gives prefix[t]),
appended as the assistant turn and stepped as a model reply (images, user turns, valid lists, kin shadow, success and
dumps as for a model turn); model calls (ar.steps, perf) exclude forced turns (ar.forced_steps).  Forced runs add
"forced": true|false to every call and prompt-dump row, trajectory vpd "forced": {"k", "prefix", "n_forced", "donor"}
and results "forced" (a summary, with module_sha256 / harness_r5_sha256); their prompt-parity audit reads the model
calls' rows only (RUN/prompt_dump_model.jsonl: a forced turn sent no request, so it has no server prompt_tokens).
Provenance: module_sha256 = sha256 of this file (paths.WORKTREE / MODULE_REL, as run_tvr.code_sha256 reads its
files), harness_r5_sha256 = sha256 of "<run_tvr code_sha256> <module_sha256>".
Copies: Harness.episode_image / _audit / finalize, cmd_eval, fake_prompt_parity and hf_prompt_parity are run_tvr's
function bodies verbatim except the lines marked "# r5:forced" (vpd/tests/test_tvr_intervene.py compares them with
run_tvr's and checks that every global they use is run_tvr's object, except ArmRun / Harness / the two audits).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from vpd import paths
from vpd.tvr import kin_env, pilot_cfg, probe_lvp, run_tvr, score_tvr, thor_worker
from vpd.tvr.run_tvr import (DUMP_TURNS, EVAL_SETS, EXIT_INVALID, EXIT_OK, EXIT_REFUSED, EXIT_TERM, IMAGE_ARMS,
                             NEW_TASK_MIN_S, NON_STOP, PROTO, PROTOCOL_SCHEMA, TIER_PHASE, TIERS, Abandon, ChatClient,
                             Convo, Ctl, KinShadow, Refused, TaskError, Terminated, adapter_info, bfs_name,
                             check_reach, code_sha256, concat_jsonl, deadline_partial, exit_code,
                             fake_image_token_audit, image_part, image_token_audit, load_reach, log, make_pool,
                             mark_near_deadline, preflight, processor_dir_for, read_json, render_block,
                             render_defaults, reset_perf_parts, resolve_tasks, run_name, served_name, sha256_bytes,
                             sha256_file, start_server, utc, write_json_atomic, write_text_atomic)

MODULE_REL = "vpd/tvr/run_tvr_forced.py"
FORCED_SCHEMA = "vpd.tvr.forced/1"  # == vpd.tvr.intervene.FORCED_SCHEMA
FORCED_SKIP = ("Stop", "InvalidAction")  # never forced: dropped from the donor actions before the first K
FORCED_ARMS = ("c",)
MARK = "# r5:forced"  # the lines of the copied run_tvr functions that differ from run_tvr's


def provenance(code: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """{"code_sha256" (run_tvr's), "module_sha256" (this file), "harness_r5_sha256"} (module doc)."""
    code = code or code_sha256()
    msha = sha256_file(paths.WORKTREE / MODULE_REL)
    return {"code_sha256": code["code_sha256"], "module_sha256": msha,
            "harness_r5_sha256": sha256_bytes(f"{code['code_sha256']} {msha}".encode())}


# --- --task-subset, --forced / --forced-k -------------------------------------------------------------------------
def _read_json_arg(opt: str, path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"{opt} {path}: {e!r}")


def load_task_subset(path, tasks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """(the tasks of the JSON id list at PATH in task-set order, the protocol block {"sha256", "n"}); Refused when
    the list is empty, not a list of strings, has a duplicate or an id outside the task set."""
    ids = _read_json_arg("--task-subset", path)
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise Refused(f"--task-subset {path}: not a JSON list of task_id strings")
    if not ids:
        raise Refused(f"--task-subset {path}: empty")
    seen: set = set()
    dup = sorted({x for x in ids if x in seen or seen.add(x)})
    if dup:
        raise Refused(f"--task-subset {path}: {len(dup)} duplicated ids (first {dup[:3]})")
    known = {t["task_id"] for t in tasks}
    bad = [x for x in ids if x not in known]
    if bad:
        raise Refused(f"--task-subset {path}: {len(bad)} ids not in the task set (first {bad[:3]})")
    return [t for t in tasks if t["task_id"] in seen], {"sha256": sha256_file(path), "n": len(ids)}


def forced_ks(a, arms: List[str]) -> Optional[List[int]]:
    """--forced-k as a list of distinct ints >= 0, or None without --forced; Refused unless both options are given
    (or neither) and --arms is exactly c."""
    if a.forced is None and a.forced_k is None:
        return None
    if a.forced is None:
        raise Refused("--forced-k needs --forced")
    if a.forced_k is None:
        raise Refused("--forced needs --forced-k")
    if list(arms) != list(FORCED_ARMS):
        raise Refused(f"--forced / --forced-k need --arms c exactly, got {a.arms!r}")
    items = str(a.forced_k).split(",")
    if not all(x.isascii() and x.isdigit() for x in items):
        raise Refused(f"--forced-k must be distinct integers >= 0 (K[,K...]), got {a.forced_k!r}")
    ks = [int(x) for x in items]
    if len(set(ks)) != len(ks):
        raise Refused(f"--forced-k must be distinct integers >= 0, got {a.forced_k!r}")
    return ks


def load_forced(path, task_ids: List[str]) -> Dict[str, Any]:
    """{"donor", "source", "file_sha256", "actions": {task_id: donor actions}} of a vpd.tvr.forced/1 file; Refused
    when the schema / donor / prefix is malformed, a task of the run has no entry or an entry holds a non-action."""
    doc = _read_json_arg("--forced", path)
    if not isinstance(doc, dict) or doc.get("schema") != FORCED_SCHEMA:
        raise Refused(f"--forced {path}: not a {FORCED_SCHEMA} file")
    donor, prefix = doc.get("donor"), doc.get("prefix")
    if not isinstance(donor, str) or not donor or not isinstance(prefix, dict):
        raise Refused(f"--forced {path}: needs a donor label and a prefix object")
    missing = [t for t in task_ids if t not in prefix]
    if missing:
        raise Refused(f"--forced {path}: {len(missing)} tasks of the run have no donor entry (first {missing[:3]})")
    vocab = set(pilot_cfg.ACTIONS) | set(FORCED_SKIP)
    bad = [t for t in task_ids if not isinstance(prefix[t], list) or any(x not in vocab for x in prefix[t])]
    if bad:
        raise Refused(f"--forced {path}: {len(bad)} entries are not lists of actions (first {bad[:3]})")
    return {"donor": donor, "source": doc.get("source"), "file_sha256": sha256_file(path),
            "actions": {t: list(prefix[t]) for t in task_ids}}


def forced_prefix(actions: Sequence[str], k: int) -> List[str]:
    """The first K donor actions after dropping every Stop and InvalidAction (all of them when fewer remain)."""
    return [x for x in actions if x not in FORCED_SKIP][:max(0, int(k))]


def forced_run(forced: Dict[str, Any], k: int, prov: Dict[str, Any]) -> Dict[str, Any]:
    """ArmRun.forced of the run with K forced turns."""
    return {"k": int(k), "donor": forced["donor"], "source": forced["source"], "file_sha256": forced["file_sha256"],
            "module_sha256": prov["module_sha256"], "harness_r5_sha256": prov["harness_r5_sha256"],
            "prefix": {t: forced_prefix(acts, k) for t, acts in forced["actions"].items()}}


# --- audits on the model calls' dump rows -----------------------------------------------------------------------
def model_call_dump(run_dir: Path) -> Path:
    """RUN/prompt_dump_model.jsonl: the prompt-dump rows of model calls only (a forced turn sent no request, so its
    row has no server prompt_tokens to audit)."""
    src = run_dir / "prompt_dump.jsonl"
    lines = [ln for ln in (src.read_text(encoding="utf-8").splitlines() if src.is_file() else [])
             if ln.strip() and not json.loads(ln).get("forced")]
    out = run_dir / "prompt_dump_model.jsonl"
    write_text_atomic(out, "".join(x + "\n" for x in lines))
    return out


def fake_prompt_parity(run_dir: Path, dump: Optional[Path] = None) -> Dict[str, Any]:  # r5:forced
    from vpd.tvr import fake_server

    rows = []
    p = run_dir / "prompt_dump.jsonl" if dump is None else Path(dump)  # r5:forced
    for line in (p.read_text(encoding="utf-8").splitlines() if p.is_file() else []):
        rec = json.loads(line)
        rows.append({"task": rec.get("task"), "step": rec.get("step"), "n_images": rec.get("n_images"),
                     "hf_prompt_tokens": fake_server.prompt_tokens(rec["messages"]),
                     "server_prompt_tokens": rec.get("prompt_tokens")})
    return {"method": "FAKE (fake server): fake_server.prompt_tokens on the dumped messages", "n": len(rows),
            "equal": bool(rows) and all(r["hf_prompt_tokens"] == r["server_prompt_tokens"] for r in rows),
            "rows": rows}


def hf_prompt_parity(run_dir: Path, processor_dir: Path, max_pixels: int = paths.MAX_PIXELS,  # r5:forced
                     dump: Optional[Path] = None) -> Dict[str, Any]:  # r5:forced
    """run_vpd.hf_prompt_parity on RUN/prompt_dump.jsonl (image parts -> the kept PNG files, RUN-relative)."""
    from vpd.walkerbench import run_vpd

    dump = run_dir / "prompt_dump.jsonl" if dump is None else Path(dump)  # r5:forced
    return run_vpd.hf_prompt_parity(run_dir, Path(processor_dir), dump, max_pixels)  # r5:forced


# --- the arm and the harness ----------------------------------------------------------------------------------------
class ArmRun(run_tvr.ArmRun):
    """run_tvr.ArmRun + forced ({"k", "donor", "source", "file_sha256", "module_sha256", "harness_r5_sha256",
    "prefix": {task_id: forced actions}}, None: no forcing) and forced_steps (forced turns; not model calls)."""

    def __init__(self, *, forced: Optional[Dict[str, Any]] = None, **kw):
        super().__init__(**kw)
        self.forced = forced
        self.forced_steps = 0


class Harness(run_tvr.Harness):
    @staticmethod
    def _forced_reply(action: str) -> Dict[str, Any]:
        """A forced turn: the registered c-arm output format; no request, so no usage / latency."""
        return {"reply": f"Action: {action}", "prompt_tokens": None, "completion_tokens": None, "finish_reason": None,
                "latency_vlm_s": None, "retries": 0}

    def episode_image(self, ar: ArmRun, wid: int, task: Dict[str, Any], attempt: int,
                      client: Optional[ChatClient]):
        tid = task["task_id"]
        replay = ar.arm == "replay"
        cap = pilot_cfg.cap_of(task)
        positions, R = self.reach_of(tid)
        if replay:
            img_dir: Optional[Path] = ar.images_dir / tid
        elif tid in ar.keep_ids:
            img_dir = ar.run_dir / "images" / tid
        else:
            img_dir = None

        def ipath(name: str) -> Optional[str]:
            return str(img_dir / name) if img_dir is not None else None

        def rel(name: str) -> Optional[str]:
            if img_dir is None:
                return None
            return str(img_dir / name) if replay else f"images/{tid}/{name}"

        inline = not replay
        t_task = time.time()
        r = self._call(wid, {"op": "reset", "task": task, "reach": positions,
                             "images": {"inline": inline, "cur": ipath("step_00.png"), "tgt": ipath("target.png")}},
                       "reset")
        used = r.get("reach_used") or {}
        if used.get("source") != "frozen" or used.get("digest") != R.digest:
            raise TaskError("protocol", f"frozen reach set not in force: {used}")
        states = [r["state"]]
        valids = [list(r["valid"])]
        texts = [r["obs_str"]]
        images = [{"cur_sha256": r["cur"]["sha256"], "cur_path": rel("step_00.png")}]
        target_img = {"sha256": r["tgt"]["sha256"], "path": rel("target.png")}
        kin = KinShadow(R)
        kin.check_valid(0, states[0], valids[0])
        convo = None
        tgt_pair = None
        if not replay:
            convo = Convo(self.served, ar.max_tokens, ar.regex)
            tgt_pair = (image_part(r["tgt"]["png_b64"]),
                        {"type": "image_url", "sha256": r["tgt"]["sha256"], "path": rel("target.png")})
            convo.add_user_images(r["obs_str"], (image_part(r["cur"]["png_b64"]),
                                                 {"type": "image_url", "sha256": r["cur"]["sha256"],
                                                  "path": rel("step_00.png")}), tgt_pair)
        public = ar.public[tid]["actions"] if replay else None
        n_turns = min(cap, len(public)) if replay else cap
        forced = None if ar.forced is None else list(ar.forced["prefix"][tid])  # r5:forced
        n_forced = 0  # r5:forced
        actions: List[str] = []
        replies: List[str] = []
        succ: List[Optional[bool]] = []
        calls: List[Dict[str, Any]] = []
        dumps: List[Dict[str, Any]] = []
        last_dump: Optional[Dict[str, Any]] = None
        n_blocked = n_unparse = n_regex_bad = 0
        stopped = False
        keep = tid in ar.keep_ids
        for t in range(n_turns):
            if self.ctl.stopped():
                raise Abandon()
            is_forced = forced is not None and t < len(forced)  # r5:forced
            if replay:
                a0 = public[t]
                m: Dict[str, Any] = {"reply": f"Action: {a0}", "prompt_tokens": None, "completion_tokens": None,
                                     "finish_reason": None, "latency_vlm_s": None, "retries": 0}
                with ar.lock:
                    ar.steps += 1
            elif is_forced:  # r5:forced
                m = self._forced_reply(forced[t])  # r5:forced
            else:
                m = self._model(ar, client, convo)
            reply = m["reply"]
            parsed = self._parse(ar.arm, reply)
            if is_forced:  # r5:forced
                if parsed != forced[t]:  # r5:forced
                    raise TaskError("protocol", f"forced turn {t}: {reply!r} parses to {parsed!r}, "  # r5:forced
                                                f"not {forced[t]!r}")  # r5:forced
                n_forced += 1  # r5:forced
                with ar.lock:  # r5:forced
                    ar.forced_steps += 1  # r5:forced
            a = parsed or "InvalidAction"
            if a == "InvalidAction":
                n_unparse += 1
            if ar.arm == "c" and parsed is None:
                n_regex_bad += 1
            if convo is not None and keep:
                row = self._dump_row(ar, tid, t, convo, m)
                if ar.forced is not None:  # r5:forced
                    row["forced"] = is_forced  # r5:forced
                if t in DUMP_TURNS:
                    dumps.append(row)
                else:
                    last_dump = row
            if convo is not None:
                convo.add_assistant(reply)
            last = t == n_turns - 1
            t0 = time.perf_counter()
            rs = self._call(wid, {"op": "step", "action": a,
                                  "images": {"inline": inline, "cur": ipath(f"step_{t + 1:02d}.png"), "skip": last}},
                            "step")
            lat_env = time.perf_counter() - t0
            info = rs.get("info") or {}
            prev = states[-1]
            states.append(rs["state"])
            s = info.get("action_success") if a in NON_STOP else None
            succ.append(s)
            if a in NON_STOP and s is False:
                n_blocked += 1
            kin.check_step(t, prev, a, rs["state"], s)
            actions.append(a)
            replies.append(reply)
            calls.append({"task_id": tid, "t": t, "arm": ar.arm, "n_images": (convo.n_images if convo else None),
                          "prompt_tokens": m.get("prompt_tokens"), "completion_tokens": m.get("completion_tokens"),
                          "finish_reason": m.get("finish_reason"), "reply": reply, "action": a,
                          "regex_ok": (parsed is not None) if ar.arm == "c" else None,
                          "parse_ok": (parsed is not None) if ar.arm == "u" else None,
                          "valid": valids[t], "action_in_valid": a in valids[t], "action_success": s,
                          "pose_before": list(kin_env.pose_of(prev)), "pose_after": list(kin_env.pose_of(rs["state"])),
                          "cur_sha256": images[t]["cur_sha256"], "latency_vlm_s": m.get("latency_vlm_s"),
                          "latency_env_s": round(lat_env, 4), "retries": m.get("retries"), "utc": utc(),
                          "error": None})
            if ar.forced is not None:  # r5:forced
                calls[-1]["forced"] = is_forced  # r5:forced
            if a == "Stop":
                stopped = True
                break
            if rs.get("done") or last:
                break
            valids.append(list(rs["valid"]))
            texts.append(rs["obs_str"])
            kin.check_valid(t + 1, rs["state"], rs["valid"])
            cur = rs.get("cur") or {}
            name = f"step_{t + 1:02d}.png"
            images.append({"cur_sha256": cur.get("sha256"), "cur_path": rel(name)})
            if convo is not None:
                convo.add_user_images(rs["obs_str"], (image_part(cur["png_b64"]),
                                                      {"type": "image_url", "sha256": cur.get("sha256"),
                                                       "path": rel(name)}), tgt_pair)
        if last_dump is not None:
            dumps.append(last_dump)
        forced_rec = None  # r5:forced
        if forced is not None:  # r5:forced
            forced_rec = {"k": ar.forced["k"], "prefix": forced, "n_forced": n_forced,  # r5:forced
                          "donor": ar.forced["donor"]}  # r5:forced
        traj = self._trajectory(ar, task, cap=cap, states=states, actions=actions, replies=replies, succ=succ,
                                valids=valids, texts=texts, images=images, target_img=target_img, stopped=stopped,
                                n_blocked=n_blocked, n_unparse=n_unparse, n_regex_bad=n_regex_bad, R=R,
                                reach_live_equal=r.get("reach_equal"), kin=kin, worker=wid, attempt=attempt,
                                wall=time.time() - t_task, reach_used=used, forced=forced_rec)  # r5:forced
        return traj, calls, dumps

    def _trajectory(self, ar: ArmRun, task, *, forced=None, **kw) -> Dict[str, Any]:
        traj = super()._trajectory(ar, task, **kw)
        if forced is not None:  # forced runs only: vpd "forced" after run_tvr's keys
            traj["vpd"]["forced"] = forced
        return traj

    def _audit(self, ar: ArmRun, partial: bool) -> Dict[str, Any]:
        if partial:
            return {"image_tokens": "skipped_deadline", "prompt_parity": "skipped_deadline"}
        if ar.arm in ("replay", "blind"):
            why = "replay (no model)" if ar.arm == "replay" else "blind arm (no images)"
            return {"image_tokens": {"skipped": why}, "prompt_parity": {"skipped": why}}
        out: Dict[str, Any] = {}
        fake = bool(getattr(self.server, "fake", False)) or bool(getattr(self, "fake_server", False))
        try:
            pdir = None if fake else processor_dir_for(self.start, self.model_dir)
            out["image_tokens"] = (fake_image_token_audit(ar.run_dir) if fake
                                   else image_token_audit(ar.run_dir, pdir))
        except Exception as e:  # noqa: BLE001
            out["image_tokens"] = {"error": f"{type(e).__name__}: {e}"[:500]}
        try:
            dump = None if ar.forced is None else model_call_dump(ar.run_dir)  # r5:forced
            pdir = None if fake else processor_dir_for(self.start, self.model_dir)  # r5:forced
            out["prompt_parity"] = (fake_prompt_parity(ar.run_dir, dump=dump) if fake  # r5:forced
                                    else hf_prompt_parity(ar.run_dir, pdir, dump=dump))  # r5:forced
            if dump is not None:  # r5:forced
                out["prompt_parity"]["dump"] = f"{dump.name} (model calls only)"  # r5:forced
        except Exception as e:  # noqa: BLE001
            out["prompt_parity"] = {"error": f"{type(e).__name__}: {e}"[:500]}
        # an audit over nothing never passes (run_vpd.hf_prompt_parity's all([]) would be True)
        pp, it = out.get("prompt_parity"), out.get("image_tokens")
        if isinstance(pp, dict) and "error" not in pp and not pp.get("n"):
            pp["equal"], pp["empty"] = False, True
        if isinstance(it, dict) and "error" not in it and not it.get("n"):
            it["empty"] = True
        return out

    @staticmethod
    def forced_summary(ar: ArmRun) -> Dict[str, Any]:
        """results "forced" of a forced run: the file, K, the provenance, the tasks whose donor had fewer than K
        forceable actions, the forced turns of the trajectories on disk (every process) and this process's forced
        turns / model calls."""
        f = ar.forced
        short = [tid for tid in ar.task_ids if len(f["prefix"][tid]) < f["k"]]
        n_forced = 0
        for tid in ar.task_ids:
            tr = read_json(ar.task_dir(tid) / "trajectory.json")
            if tr is not None:
                n_forced += int(((tr.get("vpd") or {}).get("forced") or {}).get("n_forced") or 0)
        return {"k": f["k"], "donor": f["donor"], "source": f["source"], "file_sha256": f["file_sha256"],
                "module_sha256": f["module_sha256"], "harness_r5_sha256": f["harness_r5_sha256"],
                "skipped_actions": list(FORCED_SKIP), "short_prefix_tasks": short, "forced_turns": n_forced,
                "this_process": {"forced_turns": ar.forced_steps, "model_calls": ar.steps}}

    def finalize(self, ar: ArmRun, partial: bool) -> Tuple[Dict[str, Any], bool]:
        n_calls = concat_jsonl(ar, "calls.jsonl")
        concat_jsonl(ar, "prompt_dump.jsonl")
        audit = self._audit(ar, partial)
        kin, reach = self._kin_reach(ar)
        trajs_steps = []
        for tid in ar.task_ids:
            tr = read_json(ar.task_dir(tid) / "trajectory.json")
            if tr is not None:
                trajs_steps.append(int(tr["result"]["num_steps"]))
        alive = True if ar.arm == "replay" else self.server_alive()
        perf = self.perf(ar)
        perf.update(vllm_ready_s=self.server_ready_s, max_images_per_request=ar.max_images,
                    max_prompt_tokens=ar.max_prompt,
                    mean_steps_per_task=(round(sum(trajs_steps) / len(trajs_steps), 4) if trajs_steps else None))
        ai2 = sorted({str(r.get("ai2thor")) for r in (self.pool.readys if self.pool is not None else []) if r})
        server = None
        if ar.arm != "replay":
            # VLLMServer.manifest() flat (the WB run_vpd convention: server.cfg / cmd / extra_args /
            # mm_processor_kwargs) and also under "manifest"; our keys win (no collision with the manifest's)
            server = dict(self.server_manifest or {})
            server.update(manifest=self.server_manifest,
                          versions=dict(self.versions, ai2thor=(ai2[0] if len(ai2) == 1 else ai2)),
                          max_model_len=self.max_model_len, ready_s=self.server_ready_s,
                          served_model_name=self.served)
        meta = {"run_name": ar.name, "start": self.start, "code": self.served, "served_model_name": self.served, "tier": self.tier, "task_set": self.task_set,
                "mechanism_only": ar.mech, "model_dir": self.model_dir, "adapter": self.adapter, "audit": audit,
                "kin": kin, "reach": reach, "renderer": self._renderer(ar), "settings": self.settings(ar),
                "server": server, "code_fingerprint": self.code, "code_sha256": self.code["code_sha256"],
                "protocol_sha256": ar.protocol_sha, "wall_seconds": round(ar.wall, 3), "calls": n_calls,
                "api_errors": ar.api_errors, "task_failures": len(ar.failures), "failed_tasks": ar.failures,
                "requeued_tasks": ar.requeued, "server_alive_at_end": alive, "perf": perf, "bfs_path": self.bfs_path,
                "deadline_s": self.deadline_s,
                "deadline": {"hit": self.ctl.deadline_hit, "new_tasks_stopped": self.ctl.new_tasks_stopped},
                "server_dead": self.server_dead, "utc_end": utc()}
        if ar.forced is not None:  # r5:forced
            meta["forced"] = self.forced_summary(ar)  # r5:forced
        prev = self.resumed_from(ar)
        if prev is not None:
            meta["resumed_from"] = prev
        meta["meta_keys"] = sorted(meta) + ["meta_keys"]
        bfs = None if (ar.mech or ar.arm == "replay") else self.bfs_table
        res, valid = score_tvr.score_run(ar.run_dir, task_ids=ar.task_ids, bfs_table=bfs,
                                         reach_index=self.reach_index, arm=ar.arm, mechanism_only=ar.mech,
                                         meta=meta, partial=partial)
        self.write_manifest(ar)
        log(f"{ar.name}: finalized {'PARTIAL' if partial else ('valid' if valid else 'INVALID')}: "
            f"{res.get('n_scored')}/{res.get('n_tasks')} tasks, missing {len(res.get('tasks_missing') or [])}, "
            f"failures {len(ar.failures)}, api errors {ar.api_errors}")
        return res, valid


# --- CLI ------------------------------------------------------------------------------------------------------------
def cmd_eval(a, *, ctl: Optional[Ctl] = None, pool_factory: Optional[Callable] = None,
             server_factory: Optional[Callable] = None) -> int:
    ctl = ctl or Ctl(a.deadline_s)
    arms = [x for x in a.arms.split(",") if x]
    if not arms or len(set(arms)) != len(arms) or any(x not in pilot_cfg.ARMS for x in arms):
        raise Refused(f"--arms must be distinct items of {pilot_cfg.ARMS}, got {a.arms!r}")
    if "blind" in arms and arms[-1] != "blind":
        raise Refused("the blind arm runs after the image arms: put it last")
    ks = forced_ks(a, arms)  # r5:forced
    if a.task_set not in EVAL_SETS:
        raise Refused(f"--task-set must be one of {EVAL_SETS}")
    if a.tier not in TIERS:
        raise Refused(f"--tier must be one of {TIERS}")
    if a.limit is not None and not a.allow_limit:
        raise Refused("--limit needs --allow-limit (never in a pilot eval)")
    if a.fake_server and a.base_url:
        raise Refused("--fake-server and --base-url are exclusive")
    mml = int(PROTO["max_model_len"])
    if a.max_model_len is not None:
        if int(a.max_model_len) != int(PROTO["max_model_len_fallback"]):
            raise Refused(f"--max-model-len only takes the registered fallback {PROTO['max_model_len_fallback']}")
        mml = int(a.max_model_len)
    adapter = None
    if a.adapter_meta is not None:
        if a.model_dir is None:
            raise Refused("--adapter-meta describes merged weights; give --model-dir too")
        adapter = adapter_info(Path(a.adapter_meta))
        if adapter["start"] != a.start:
            raise Refused(f"adapter of start {adapter['start']!r}, but --start is {a.start!r}")
        if a.tier == "raw" or adapter["tier"] != TIER_PHASE[a.tier]:
            raise Refused(f"--tier {a.tier} does not match the adapter phase {adapter['tier']!r}")
    elif a.tier != "raw":
        raise Refused(f"--tier {a.tier} needs --adapter-meta (and --model-dir)")
    if a.vllm_parity and not Path(a.vllm_parity).is_file():
        raise Refused(f"--vllm-parity {a.vllm_parity}: no such dry-run json")
    served = served_name(a.start)
    tvr_out = Path(a.tvr_out)
    tasks = resolve_tasks(a.task_set, tvr_out=tvr_out, tasks_dir=Path(a.tasks_dir))
    subset = None  # r5:forced
    if a.task_subset is not None:  # r5:forced
        tasks, subset = load_task_subset(a.task_subset, tasks)  # r5:forced
    if a.limit is not None:
        tasks = tasks[:int(a.limit)]
    forced = None if ks is None else load_forced(a.forced, [t["task_id"] for t in tasks])  # r5:forced
    ridx = load_reach(a.reach or tvr_out / "reach" / "v1")
    check_reach(tasks, ridx)
    bfs_path = None
    bfs_table = None
    if not a.mechanism_only:
        bfs_path = str(a.bfs or tvr_out / "bfs" / f"{bfs_name(a.task_set)}_opt.json")
        bfs_table = read_json(bfs_path)
        if bfs_table is None:
            raise Refused(f"cannot read the BFS table {bfs_path}")
    image_arms = [x for x in arms if x in IMAGE_ARMS]
    envs = lp = prof = None
    if image_arms:
        envs, lp, prof = render_defaults(a, "eval")
    code = code_sha256()
    prov = None if ks is None else provenance(code)  # r5:forced
    ids = [t["task_id"] for t in tasks]
    out_root = Path(a.out_root)
    ars: List[ArmRun] = []
    runs = [(arm, None) for arm in arms] if ks is None else [(arms[0], k) for k in ks]  # r5:forced
    for arm, k in runs:  # r5:forced
        proto = {"schema": PROTOCOL_SCHEMA, "arm": arm, "start": a.start, "code": served, "tier": a.tier,
                 "adapter_sha256": (adapter or {}).get("adapter_sha256"), "model_dir_is_merged": a.model_dir is not None,
                 "task_set": a.task_set, "task_ids_sha256": sha256_bytes("\n".join(ids).encode()),
                 "protocol": probe_lvp.jsafe(PROTO), "max_model_len_used": mml,
                 "render": render_block(envs, lp, prof) if arm in IMAGE_ARMS else None,
                 "blind_concurrency": int(a.blind_concurrency) if arm == "blind" else None,
                 "reach_index_sha256": ridx.sha256, "code_sha256": code["code_sha256"],
                 "mechanism_only": bool(a.mechanism_only)}
        if subset is not None:  # r5:forced
            proto["task_subset"] = dict(subset)  # r5:forced
        if k is not None:  # r5:forced
            proto["forced"] = {"k": k, "file_sha256": forced["file_sha256"], "donor": forced["donor"],  # r5:forced
                               "module_sha256": prov["module_sha256"],  # r5:forced
                               "harness_r5_sha256": prov["harness_r5_sha256"]}  # r5:forced
        name = run_name(a.name_prefix, a.start, a.tier, arm) + ("" if k is None else f"__f{k}")  # r5:forced
        ar = ArmRun(arm=arm, name=name, run_dir=out_root / name, tasks=tasks, protocol=proto,
                    mechanism_only=a.mechanism_only, keep=a.keep_images_per_category,  # r5:forced
                    forced=None if k is None else forced_run(forced, k, prov))  # r5:forced
        preflight(ar, a.resume)
        ars.append(ar)
    pending = [ar for ar in ars if not ar.complete]
    for ar in ars:
        if ar.complete:
            log(f"{ar.name}: complete (valid results.json, equal protocol): skipped")
    if not pending:
        return EXIT_OK
    for ar in pending:
        ar.run_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(ar.run_dir / "protocol.json", ar.protocol)
        write_json_atomic(ar.run_dir / "task_ids.json", ar.task_ids)
        reset_perf_parts(ar, a.resume)
    base = f"{a.name_prefix}__{a.start}__{a.tier}"
    h = Harness(ctl=ctl, reach_index=ridx, served=served, bfs_table=bfs_table, bfs_path=bfs_path, start=a.start,
                tier=a.tier, task_set=a.task_set, model_dir=a.model_dir, adapter=adapter, envs=envs, lp=lp,
                profile=prof, blind_concurrency=a.blind_concurrency, max_model_len=mml, deadline_s=a.deadline_s,
                argv=sys.argv)
    results: List[Tuple[bool, bool]] = []
    srv = (server_factory or (lambda: start_server(a, served, out_root / f"{base}__vllm.log", mml)))()
    h.server = srv  # before __enter__: a SIGTERM during the start-up still kills the server process
    try:
        t0 = time.time()
        try:
            srv.__enter__()
        except Exception as e:  # noqa: BLE001  (the payload may retry with the O9 fallback)
            write_json_atomic(out_root / f"{base}__server_error.json",
                              {"error": f"{type(e).__name__}: {e}"[:2000], "max_model_len": mml, "utc": utc()})
            log(f"server did not start: {e!r}")
            h.server = None
            return EXIT_INVALID
        h.fake_server = bool(getattr(srv, "fake", False))
        h.server_ready_s = getattr(srv, "ready_seconds", round(time.time() - t0, 1))
        h.server_manifest = srv.manifest() if hasattr(srv, "manifest") else None
        if a.vllm_parity:
            from vpd.tvr import parity_tvr

            out = out_root / f"{base}__vllm_parity.json"
            try:
                rep = parity_tvr.vllm_parity(a.start, a.vllm_parity, srv.base_url, out, served=served)
                log(f"vllm parity: equal {rep['equal']} ({rep['n']} cases)")
            except Exception as e:  # noqa: BLE001  (recorded; the smoke verdict / gate read the file)
                write_json_atomic(out, {"equal": False, "error": f"{type(e).__name__}: {e}"[:2000], "utc": utc()})
        img_pending = [ar for ar in pending if ar.arm in IMAGE_ARMS]
        if img_pending and ctl.left() >= NEW_TASK_MIN_S:  # no pool start-up when no task could start anyway
            houses = Path(a.houses or tvr_out / "houses")
            pool = (pool_factory(envs, lp, prof, out_root / f"{base}__pool") if pool_factory
                    else make_pool(a, envs, lp, prof, out_root / f"{base}__pool", houses))
            h.pool = pool
            try:
                pool.start()
            except thor_worker.PoolError as e:
                log(f"THOR pool did not start: {e}")
        elif img_pending:  # the image arms are not run and, without a measured renderer, cannot be valid
            ctl.new_tasks_stopped = True
            log(f"no THOR pool start-up: {ctl.left():.0f} s left (< {NEW_TASK_MIN_S:.0f} s)")
        # every pending arm is finalized (a result file each), also when it cannot run: the deadline passed / new
        # tasks were stopped (-> results_partial.json) or the server died in an earlier arm (-> results_invalid)
        for ar in pending:
            mark_near_deadline(ar, ctl)
            h.write_manifest(ar)
            if ctl.stopped() or ctl.new_tasks_stopped or h.server_dead:
                why = ("server dead" if h.server_dead else "soft deadline" if ctl.left() <= 0 or ctl.deadline_hit
                       else "stopped" if ctl.stop.is_set() else f"< {NEW_TASK_MIN_S:.0f} s left")
                log(f"{ar.name}: not run ({why}); {len(ar.todo)} of {len(ar.tasks)} tasks left")
            else:
                log(f"{ar.name}: {len(ar.todo)} of {len(ar.tasks)} tasks to run")
                if ar.arm in IMAGE_ARMS:
                    if h.pool is not None and any(h.pool.alive(w) for w in range(h.pool.n)):
                        h.run_image_arm(ar)
                else:
                    if h.pool is not None and not ctl.stopped():
                        h.pool.close()
                    h.run_blind_arm(ar)
            partial = deadline_partial(ar, ctl, h)
            _, valid = h.finalize(ar, partial)
            results.append((valid, partial))
        return exit_code(results, ctl, len(pending))
    finally:
        if ctl.terminated or ctl.deadline_hit:
            h.fast_shutdown()
        else:
            h.normal_shutdown()


def build_parser() -> argparse.ArgumentParser:
    """run_tvr's parser + the three eval options + `provenance`."""
    ap = run_tvr.build_parser()
    ap.description = __doc__.split("\n\n")[0]
    sub = next(x for x in ap._actions if isinstance(x, argparse._SubParsersAction))
    e = sub.choices["eval"]
    e.add_argument("--task-subset", default=None, help="a JSON list of task_ids of the --task-set")
    e.add_argument("--forced", default=None, help="a vpd.tvr.forced/1 file (needs --forced-k and --arms c)")
    e.add_argument("--forced-k", default=None, help="K[,K...] forced opening turns, one run per K")
    p = sub.add_parser("provenance", help="print harness_r5_sha256 (and write the provenance json to --out)")
    p.add_argument("--out", default=None)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    """eval: run_tvr.main's eval branch with this module's cmd_eval (run_tvr's signal handler and Ctl); replay /
    rescore: run_tvr.main."""
    t0 = time.monotonic()
    a = build_parser().parse_args(argv)
    if a.cmd == "provenance":
        prov = provenance()
        if a.out:
            write_json_atomic(a.out, dict(prov, module=MODULE_REL, utc=utc()))
        print(prov["harness_r5_sha256"])
        return EXIT_OK if prov["module_sha256"] else EXIT_INVALID
    if a.cmd != "eval":
        return run_tvr.main(argv)
    run_tvr._CTL = Ctl(a.deadline_s, t0=t0)
    signal.signal(signal.SIGTERM, run_tvr._on_signal)
    signal.signal(signal.SIGINT, run_tvr._on_signal)
    try:
        return cmd_eval(a, ctl=run_tvr._CTL)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except Terminated:
        log("terminated (SIGTERM): fast shutdown done, no results written")
        return EXIT_TERM


if __name__ == "__main__":
    sys.exit(main())
