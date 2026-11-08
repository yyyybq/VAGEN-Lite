"""Real-render RM expert demonstrations on WalkerBench s2train, in n2_train's sharegpt format.

    python -m vpd.walkerbench.demos --shard 3/8          # one shard -> OUT/.../s2train_640x400/shards/shard3of8.*
    python -m vpd.walkerbench.demos --merge              # -> demos_all.jsonl + manifest.json
    python -m vpd.walkerbench.demos --limit 3 --out SCRATCH/x    # quick look (never into the frozen dir)

Port of scratchpad vpd_demos/b3_proto.py (survey part 3):
- a private sandbox of the WalkerBench checkout (vpd.walkerbench.sandbox.make_sandbox) over split/s2train/tasks with
  RENDER_OUTPUT_SIZE = --size (default 640x400); the engine renders each view directly (py360convert e2p, hFOV 90,
  JPEG q90); no resize anywhere;
- WalkerBench's own loop (scripts/run_benchmark_parallel.run_single_task) with rbp.VLNAgent subclassed:
  decide_action = the RM expert, vpd.walkerbench.baselines.rm_expert (DECISIONS B6, b3_proto.py lines 43-60: follow
  visual_path; rotate first when the next pano is > 45 deg off the heading and this node has not been rotated at,
  then MOVE; STOP("ARRIVED") at the end), so the demo expert and the baseline ceiling are one function;
  get_image_base64 moves the engine JPEG to images/<stem>/turn_XX.jpg and returns a dummy;
- each episode becomes {id: stem, task_type: nav|vis, split: train_rest|val, messages, images, n_turns, size, ...};
  user turns are the agent's own text with '<image>' appended at the end, replies json.dumps({"THOUGHT": "",
  "ACTION": ...}).
Per-task checks (a failure drops the task into the manifest's 'dropped' with its reasons): no missing image;
#images == #user turns == #'<image>'; every image is --size (PIL); every reply round-trips through the agent's
_parse_response to the expert's action and matches the constrained-decoding regex for that step's move count; the
last reply is STOP("ARRIVED"); evaluate_session success == 1.  Records are keyed by the file stem.
The 57 s2val demos are rendered too (split='val'); the tier orders draw from train_rest only.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import datetime as _dt
import hashlib
import io
import json
import os
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

from vpd import paths
from vpd.walkerbench import action_regex as AR
from vpd.walkerbench import baselines as B
from vpd.walkerbench import sandbox
from vpd.walkerbench import split as S

SCHEMA = "vpd.walkerbench.demos/1"
SHARD_SCHEMA = "vpd.walkerbench.demos_shard/1"
IMAGE = "<image>"
AGENT = "expert"
DEFAULT_SIZE = (640, 400)
STOP_REPLY = AR.reply(AR.STOP_ACTION)


def default_out(size: tuple[int, int]) -> Path:
    return paths.WB_DEMOS / f"s2train_{size[0]}x{size[1]}"


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- pure helpers -------------------------------------

def action_text(act: dict) -> str:
    """The ACTION string of an engine action dict (b3_proto.reply_text)."""
    t = act["type"]
    if t == "move":
        return "MOVE(%d)" % act["move_id"]
    if t == "rotation":
        return "ROTATION(%s, 0)" % act["heading"]
    if t == "stop":
        return AR.STOP_ACTION
    raise ValueError(f"unknown action type {t!r}")


def reply_text(act: dict) -> str:
    return AR.reply(action_text(act))


def same_action(back: dict, act: dict) -> bool:
    """Parsed reply == expert action on the fields the engine uses."""
    if back.get("type") != act["type"]:
        return False
    keys = {"move": ("move_id",), "rotation": ("heading", "pitch"), "stop": ("answer",)}[act["type"]]
    return all(back.get(k) == act.get(k) for k in keys)


def regex_ok(text: str, n_moves: int) -> bool:
    return re.fullmatch(AR.action_regex(n_moves), text) is not None


def to_messages(raw: list[dict]) -> list[dict]:
    """Agent messages (OpenAI parts) -> sharegpt messages; an image part becomes '<image>' at the end of the text."""
    out = []
    for m in raw:
        c = m["content"]
        if isinstance(c, str):
            out.append({"role": m["role"], "content": c})
            continue
        if not c or c[0].get("type") != "text" or len(c) > 2 or (len(c) == 2 and c[1].get("type") != "image_url"):
            raise ValueError(f"unexpected {m['role']} content parts {[p.get('type') for p in c]}")
        out.append({"role": m["role"], "content": c[0]["text"] + (IMAGE if len(c) == 2 else "")})
    return out


def shard_stems(stems: list[str], i: int, n: int) -> list[str]:
    if not (n >= 1 and 0 <= i < n):
        raise ValueError(f"bad shard {i}/{n}")
    return sorted(stems)[i::n]


def parse_shard(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", text)
    if not m:
        raise ValueError(f"--shard must look like 3/8, got {text!r}")
    i, n = int(m.group(1)), int(m.group(2))
    shard_stems([], i, n)
    return i, n


def image_tokens(size: tuple[int, int], factor: int = 32) -> int:
    """Qwen3-VL image tokens for one view (smart_resize to multiples of 32, 2x2 merge, + vision start/end):
    640x400 -> 640x384 -> 240 + 2; 1280x800 -> 1000 + 2.  An estimate for the manifest; G0 measures it."""
    w, h = size
    hh, ww = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    return (hh // factor) * (ww // factor) + 2


def _fallback_parse(text: str, observation: dict | None = None) -> dict:
    """Strict stand-in for VLNAgent._parse_response on canonical replies (tests; no WalkerBench import)."""
    d = json.loads(text)
    a = d["ACTION"]
    m = re.fullmatch(r"([A-Z]+)\((.*)\)", a.strip())
    cmd, args = m.group(1), m.group(2)
    if cmd == "MOVE":
        return {"type": "move", "move_id": int(args), "reason": d["THOUGHT"]}
    if cmd == "ROTATION":
        v = [float(x) for x in args.split(",")]
        return {"type": "rotation", "heading": v[0], "pitch": v[1] if len(v) > 1 else 0.0, "reason": d["THOUGHT"]}
    if cmd == "STOP":
        return {"type": "stop", "answer": args.strip().strip("'").strip('"'), "reason": d["THOUGHT"]}
    raise ValueError(cmd)


def image_size(p: str | Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(p) as im:
        return im.size


def check_record(rec: dict, size: tuple[int, int], parse=None) -> list[str]:
    """Problems of one demo record (empty = valid).  parse(text, obs) -> action dict; default a strict stand-in.
    Checks the stored record itself: schema, '<image>' only at the end of each user text, #images == #user turns,
    the image files' size, every reply canonical and parse-round-tripping, the last reply STOP("ARRIVED")."""
    parse = parse or _fallback_parse
    probs = []
    for k in ("id", "task_type", "messages", "images", "n_turns", "size"):
        if k not in rec:
            probs.append(f"missing key {k}")
    if probs:
        return probs
    msgs, imgs = rec["messages"], rec["images"]
    if not msgs or msgs[0]["role"] != "system" or not isinstance(msgs[0]["content"], str):
        probs.append("first message is not a system string")
    roles = [m["role"] for m in msgs[1:]]
    if roles != ["user", "assistant"] * (len(roles) // 2) or len(roles) % 2:
        probs.append(f"turns are not user/assistant pairs: {''.join(r[0] for r in roles)[:40]}")
    users = [m["content"] for m in msgs if m["role"] == "user"]
    replies = [m["content"] for m in msgs if m["role"] == "assistant"]
    if rec["n_turns"] != len(users):
        probs.append(f"n_turns {rec['n_turns']} != {len(users)} user turns")
    if len(imgs) != len(users):
        probs.append(f"{len(imgs)} images for {len(users)} user turns")
    for j, u in enumerate(users):
        if u.count(IMAGE) != 1 or not u.endswith(IMAGE):
            probs.append(f"user turn {j}: '<image>' not exactly once at the end")
    if any(IMAGE in m["content"] for m in msgs if m["role"] != "user"):
        probs.append("'<image>' outside a user turn")
    if list(rec["size"]) != list(size):
        probs.append(f"record size {rec['size']} != {list(size)}")
    for p in imgs:
        if not Path(p).is_file():
            probs.append(f"image missing: {p}")
        elif image_size(p) != tuple(size):
            probs.append(f"image {p} is {image_size(p)}, not {tuple(size)}")
    for j, r in enumerate(replies):
        try:
            d = json.loads(r)
            act = d["ACTION"]
            if list(d) != ["THOUGHT", "ACTION"] or d["THOUGHT"] != "" or r != AR.reply(act):
                probs.append(f"reply {j} not canonical: {r[:80]}")
                continue
            want = _fallback_parse(r)
            back = parse(r, {"heading": -1.0, "pitch": -1.0})  # the parser's fallback would return these
            if not same_action(back, want) or (want["type"] == "rotation" and want["pitch"] != 0):
                probs.append(f"reply {j} does not round-trip: {r[:80]} -> {back}")
        except Exception as e:  # noqa: BLE001 - every malformed reply is a finding
            probs.append(f"reply {j} unparsable ({e!r}): {r[:80]}")
    if not replies or replies[-1] != STOP_REPLY:
        probs.append("last reply is not STOP(\"ARRIVED\")")
    return probs


def split_tags(split_doc: dict) -> dict[str, str]:
    """stem -> 'train_rest' | 'val' for every s2train task."""
    sp = split_doc["splits"]
    tags = {s: "train_rest" for s in sp["train_rest"]}
    tags.update({s: "val" for s in sp["s2val"]})
    if set(tags) != set(sp["s2train"]):
        raise RuntimeError("s2train != s2val + train_rest in split_s2.json")
    return tags


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def code_state() -> dict:
    """Code identity for the manifests: sha256 of the files that make a demo, plus the worktree state."""
    here = Path(__file__).resolve()
    st = {"demos.py": S.sha256_file(here), "sandbox.py": S.sha256_file(sandbox.__file__),
          "baselines.py (rm_expert)": S.sha256_file(B.__file__), "action_regex.py": S.sha256_file(AR.__file__)}
    try:
        from vpd.sco import vpd_code_state
        st["worktree"] = vpd_code_state()
    except Exception as e:  # noqa: BLE001 - the manifest records why it is missing
        st["worktree"] = {"error": repr(e)}
    return st


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_if_new(path: Path, text: str, force: bool) -> str:
    """'wrote' | 'same'; refuses to replace different content unless force (no job may be reading it)."""
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return "same"
        if not force:
            raise FileExistsError(f"{path} exists with different content (use --force only if no job reads it)")
    _write_atomic(path, text)
    return "wrote"


# ---------------------------------------------------------------- rendering ----------------------------------------

def load_engine(copy: Path, tasks_label: str, size: tuple[int, int]):
    """The sandbox copy's runner/scorer/engine in this process, with REAL rendering at size."""
    wb = B.load_walkerbench(copy, tasks_label)  # stubs rendering; undone below
    wb.og._observation_generator = None  # get_observation_generator() now builds the real generator lazily
    gen = wb.og.get_observation_generator()
    if tuple(gen.output_size) != tuple(size):
        raise RuntimeError(f"engine renders {gen.output_size}, expected {size}: sandbox patch not in effect")
    return wb


def make_agent_class(wb, state: dict, img_root: Path):
    """rbp.VLNAgent subclass: RM expert + image capture.  state[session_id] collects one episode."""
    sm, ax = wb.sm, wb.ax

    class DemoAgent(wb.rbp.VLNAgent):
        def _st(self):
            s = sm.get_session(self.session_id)
            return state.setdefault(self.session_id, {
                "stem": s.task_id, "images": [], "missing": 0, "rotated": set(), "parse_mismatch": 0,
                "regex_mismatch": 0, "move_mismatch": 0, "actions": [], "msgs": None,
                "parser": self._parse_response})

        def get_image_base64(self, url):
            st = self._st()
            src = wb.copy / str(url).lstrip("/")
            if not src.is_file():
                st["missing"] += 1
                return None
            d = img_root / st["stem"]
            d.mkdir(parents=True, exist_ok=True)
            dst = d / ("turn_%02d.jpg" % len(st["images"]))
            shutil.copyfile(src, dst)
            src.unlink()  # this sandbox's temp file; a failed later render can then never reuse a stale view
            st["images"].append(str(dst))
            return "x"

        def decide_action(self, obs):
            st = self._st()
            self.messages.append(self._build_user_message(obs))
            s = sm.get_session(self.session_id)
            moves = ax._get_available_moves(s)
            act = B.rm_expert(s, lambda: moves, st["rotated"])
            if act["type"] == "move":
                seen = {m["id"]: m.get("heading") for m in obs["available_moves"]}
                mine = next(m for m in moves if m["id"] == act["move_id"])
                if [m["id"] for m in obs["available_moves"]] != [m["id"] for m in moves] or \
                        abs((seen.get(act["move_id"]) or 0) - (mine.get("heading") or 0)) > 1e-6:
                    st["move_mismatch"] += 1
            txt = reply_text(act)
            back = self._parse_response(txt, obs)
            st["parse_mismatch"] += not same_action(back, act)
            st["regex_mismatch"] += not regex_ok(txt, len(obs["available_moves"]))
            self.messages.append({"role": "assistant", "content": txt})
            st["msgs"] = self.messages
            st["actions"].append(act)
            return act

    return DemoAgent


def episode_record(st: dict, ev: dict | None, tag: str, size: tuple[int, int], split_doc: dict) -> tuple[dict, list]:
    """(record, problems) for one finished episode."""
    stem = st["stem"]
    probs = []
    msgs = to_messages(st["msgs"] or [])
    t = split_doc["tasks"][stem]
    rec = {"id": stem, "task_type": S.kind_of(stem), "split": tag, "messages": msgs, "images": list(st["images"]),
           "n_turns": sum(m["role"] == "user" for m in msgs), "size": [int(size[0]), int(size[1])],
           "source": {"file": t["file"], "src": t["src"], "sha256": t["sha256"], "route_id": t["route_id"],
                      "geofence": t["geofence"], "city": t["city"], "n_path": t["n_path"]},
           "expert": {"policy": "RM", "steps": len(st["actions"]),
                      "rotations": sum(a["type"] == "rotation" for a in st["actions"]),
                      "moves": sum(a["type"] == "move" for a in st["actions"]),
                      "success": None if ev is None else int(ev["success"]),
                      "error_m": None if ev is None else ev.get("error"),
                      "spl": None if ev is None else ev.get("spl")}}
    for k in ("missing", "parse_mismatch", "regex_mismatch", "move_mismatch"):
        if st[k]:
            probs.append(f"{k} {st[k]}")
    if any(a.get("reason") == "path break" for a in st["actions"]):
        probs.append("expert stopped at a path break")
    if ev is None:
        probs.append("no evaluate_session result")
    elif int(ev["success"]) != 1:
        probs.append(f"evaluate_session success {ev['success']} (error {ev.get('error')} m)")
    return rec, probs


def run_shard(tasks_dir: Path, out: Path, work: Path, size: tuple[int, int], shard: tuple[int, int],
              limit: int | None = None, stems_arg: list[str] | None = None, split_dir: Path = paths.WB_SPLIT,
              force: bool = False, keep_sandbox: bool = False) -> dict:
    tasks_dir, out, work, split_dir = (Path(x).resolve() for x in (tasks_dir, out, work, split_dir))
    i, n = shard
    tag = f"shard{i}of{n}"
    shards = out / "shards"
    jl, mf = shards / f"{tag}.jsonl", shards / f"{tag}.json"
    if (jl.exists() or mf.exists()) and not force:
        raise FileExistsError(f"{jl} or {mf} exists (use --force to re-render this shard; no job may read it)")
    split_path = split_dir / "split_s2.json"
    split_doc = json.loads(split_path.read_text())
    tags = split_tags(split_doc)
    all_stems = sorted(p.stem for p in tasks_dir.glob("*.json"))
    unknown = sorted(set(all_stems) - set(tags))
    if unknown:
        raise RuntimeError(f"{len(unknown)} tasks in {tasks_dir} are not s2train tasks (first {unknown[:3]})")
    mine = shard_stems(all_stems, i, n)
    if stems_arg:
        bad = sorted(set(stems_arg) - set(all_stems))
        if bad:
            raise ValueError(f"not in {tasks_dir}: {bad[:5]}")
        mine = [s for s in mine if s in set(stems_arg)]
    if limit is not None:
        mine = mine[:limit]
    img_root = out / "images"
    for s in mine:
        d = img_root / s
        if d.exists() and any(d.iterdir()) and not force:
            raise FileExistsError(f"{d} already holds images (use --force to overwrite them in place)")
    stamp = B.utc_stamp()
    wdir = work / f"demos_{stamp}_{tag}"
    logs = out / "logs" / f"{stamp}_{tag}"
    if logs.exists():
        raise FileExistsError(f"{logs} exists")
    logs.mkdir(parents=True)
    t_all = time.time()
    copy, diff = sandbox.make_sandbox(wdir, tasks_dir, render_size=size, exclude=sandbox.LEAN_EXCLUDE)
    t_sbx = time.time() - t_all
    cwd = os.getcwd()
    records, dropped, per_task = [], [], {}
    try:
        wb = load_engine(copy, sandbox.task_label(tasks_dir), size)
        state: dict = {}
        wb.rbp.VLNAgent = make_agent_class(wb, state, img_root)
        wb.rbp.TASKS_DIR = wb.tasks_dir
        wb.rbp.LOGS_DIR = logs
        for k, stem in enumerate(mine):
            t0 = time.time()
            buf = io.StringIO()
            before = set(state)
            with contextlib.redirect_stdout(buf):
                ok = wb.rbp.run_single_task(AGENT, stem)
            sids = [s for s in state if s not in before]
            lg = sorted(logs.glob(f"{AGENT}_{stem}_*.jsonl"))
            ev = wb.ev.evaluate_session(lg[0], wb.tasks_dir) if len(lg) == 1 else None
            if len(sids) != 1:
                dropped.append({"id": stem, "reasons": [f"run_single_task returned {ok}; {len(sids)} sessions"],
                                "stdout_tail": buf.getvalue()[-2000:]})
                continue
            st = state.pop(sids[0])
            rec, probs = episode_record(st, ev, tags[stem], size, split_doc)
            if ok is False:
                probs.append("run_single_task returned False")
            if len(lg) != 1:
                probs.append(f"{len(lg)} logs")
            files = sorted(p.name for p in (img_root / stem).glob("*")) if (img_root / stem).exists() else []
            if files != sorted(Path(p).name for p in rec["images"]):
                probs.append(f"images/{stem} holds {len(files)} files for {len(rec['images'])} turns (stale files?)")
            probs += check_record(rec, size, parse=st["parser"])  # the stored record, re-parsed by the agent
            secs = round(time.time() - t0, 2)
            per_task[stem] = {"turns": rec["n_turns"], "rotations": rec["expert"]["rotations"], "seconds": secs,
                              "success": rec["expert"]["success"]}
            if probs:
                dropped.append({"id": stem, "reasons": probs, "images": str(img_root / stem),
                                "stdout_tail": buf.getvalue()[-2000:]})
            else:
                records.append(rec)
            print(f"[{tag}] {k + 1}/{len(mine)} {stem}: turns {rec['n_turns']} rot {rec['expert']['rotations']} "
                  f"success {rec['expert']['success']} {'DROP ' + '; '.join(probs) if probs else 'ok'} ({secs} s)",
                  flush=True)
        # views rendered but never shown (the engine also renders the view after the final STOP: about 1 per task)
        leftover = len(list((copy / "temp_images").rglob("*.jpg")))
    finally:
        os.chdir(cwd)
        if not keep_sandbox:
            shutil.rmtree(wdir)  # this run's own scratch sandbox
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    _write_atomic(jl, text)
    n_img = sum(len(r["images"]) for r in records)
    man = {"schema": SHARD_SCHEMA, "created_utc": utc_now(), "shard": [i, n], "size": list(size),
           "tasks_dir": str(tasks_dir), "n_tasks_dir": len(all_stems), "assigned": mine,
           "partial": bool(limit is not None or stems_arg), "limit": limit, "stems_arg": stems_arg,
           "n_records": len(records), "dropped": dropped, "n_images": n_img,
           "image_bytes": sum(Path(p).stat().st_size for r in records for p in r["images"]),
           "jsonl": jl.name, "jsonl_sha256": sha256_text(text), "split_s2_sha256": S.sha256_file(split_path),
           "sandbox_settings_diff": diff, "walkerbench_commit": sandbox.repo_commit(), "logs": str(logs),
           "unconsumed_temp_images": leftover, "seconds": round(time.time() - t_all, 1),
           "sandbox_seconds": round(t_sbx, 1), "per_task": per_task, "code": code_state()}
    _write_atomic(mf, json.dumps(man, indent=1) + "\n")
    print(f"[{tag}] {len(records)} records, {len(dropped)} dropped, {n_img} images, {man['seconds']} s -> {jl}")
    return man


# ---------------------------------------------------------------- merge --------------------------------------------

def merge(out: Path, size: tuple[int, int], tasks_dir: Path = paths.WB_S2TRAIN_TASKS,
          split_dir: Path = paths.WB_SPLIT, allow_partial: bool = False, force: bool = False) -> dict:
    out, tasks_dir, split_dir = Path(out), Path(tasks_dir), Path(split_dir)
    mans = [json.loads(p.read_text()) for p in sorted((out / "shards").glob("shard*of*.json"))]
    if not mans:
        raise FileNotFoundError(f"no shard manifests under {out / 'shards'}")
    ns = {m["shard"][1] for m in mans}
    if len(ns) != 1:
        raise RuntimeError(f"shard manifests from different shardings: n in {sorted(ns)}")
    n = ns.pop()
    got = sorted(m["shard"][0] for m in mans)
    if got != list(range(n)):
        raise RuntimeError(f"shards present {got}, expected 0..{n - 1}")
    if any(list(m["size"]) != list(size) for m in mans):
        raise RuntimeError("a shard was rendered at another size")
    if len({m["sandbox_settings_diff"] for m in mans}) != 1:
        raise RuntimeError("shards differ in their sandbox settings diff")
    if len({m["split_s2_sha256"] for m in mans}) != 1:
        raise RuntimeError("shards were made from different split_s2.json files")
    partial = [m["shard"][0] for m in mans if m["partial"]]
    if partial and not allow_partial:
        raise RuntimeError(f"shards {partial} are partial (--limit/--stems); pass --allow-partial")
    split_path = split_dir / "split_s2.json"
    split_doc = json.loads(split_path.read_text())
    if S.sha256_file(split_path) != mans[0]["split_s2_sha256"]:
        raise RuntimeError(f"shards were made from another split_s2.json than {split_path}")
    tags = split_tags(split_doc)
    assigned = [s for m in mans for s in m["assigned"]]
    if len(assigned) != len(set(assigned)):
        raise RuntimeError("a task was assigned to two shards")
    if not allow_partial:
        want = sorted(p.stem for p in tasks_dir.glob("*.json"))
        if sorted(assigned) != want:
            raise RuntimeError(f"shards cover {len(assigned)} tasks, {tasks_dir} has {len(want)}")
    recs = {}
    for m in mans:
        p = out / "shards" / m["jsonl"]
        text = p.read_text(encoding="utf-8")
        if sha256_text(text) != m["jsonl_sha256"]:
            raise RuntimeError(f"{p} changed after its manifest was written")
        for line in text.splitlines():
            r = json.loads(line)
            if r["id"] in recs:
                raise RuntimeError(f"duplicate demo {r['id']}")
            if r["id"] not in m["assigned"]:
                raise RuntimeError(f"{p}: record {r['id']} not assigned to this shard")
            recs[r["id"]] = line
    dropped = [d for m in mans for d in m["dropped"]]
    lost = sorted(set(assigned) - set(recs) - {d["id"] for d in dropped})
    if lost:
        raise RuntimeError(f"{len(lost)} assigned tasks neither recorded nor dropped (first {lost[:3]})")
    order = sorted(recs)
    text = "".join(recs[s] + "\n" for s in order)
    probs, imgs, nbytes, turns, rots, chars = [], 0, 0, [], 0, 0
    by = collections.Counter()
    for s in order:
        r = json.loads(recs[s])
        pr = check_record(r, size)
        if pr:
            probs.append({"id": s, "problems": pr[:5]})
        if r["split"] != tags[s] or r["task_type"] != S.kind_of(s):
            probs.append({"id": s, "problems": [f"split/kind tag {r['split']}/{r['task_type']} wrong"]})
        by[(r["split"], r["task_type"])] += 1
        imgs += len(r["images"])
        nbytes += sum(Path(p).stat().st_size for p in r["images"])
        turns.append(r["n_turns"])
        rots += r["expert"]["rotations"]
        chars += sum(len(m["content"]) for m in r["messages"])
    if probs:
        raise RuntimeError(f"{len(probs)} merged records fail check_record: {probs[:3]}")
    dst = out / "demos_all.jsonl"
    st = _write_if_new(dst, text, force)
    tpi = image_tokens(size)
    man = {"schema": SCHEMA, "created_utc": utc_now(), "size": list(size), "tasks_dir": str(tasks_dir),
           "split_s2_sha256": S.sha256_file(split_path), "demos_all": dst.name, "demos_all_sha256": sha256_text(text),
           "partial": bool(partial), "counts": {
               "assigned": len(assigned), "records": len(order), "dropped": len(dropped),
               "by_split": {k: sum(v for (sp, _), v in by.items() if sp == k) for k in ("train_rest", "val")},
               "by_split_kind": {f"{sp}/{kd}": v for (sp, kd), v in sorted(by.items())}},
           "dropped": dropped, "n_images": imgs, "image_bytes": nbytes, "turns": {
               "total": sum(turns), "min": min(turns) if turns else None, "max": max(turns) if turns else None,
               "median": statistics.median(turns) if turns else None}, "rotations": rots,
           "tokens_estimate": {"image_tokens_per_image": tpi, "image_tokens": imgs * tpi, "text_chars": chars,
                               "note": "image tokens by smart_resize arithmetic; text not tokenized here"},
           "expert": "vpd.walkerbench.baselines.rm_expert (DECISIONS B6 = b3_proto.py lines 43-60)",
           "sandbox_settings_diff": mans[0]["sandbox_settings_diff"], "walkerbench_commit": mans[0]["walkerbench_commit"],
           "shards": [{k: m[k] for k in ("shard", "n_records", "seconds", "sandbox_seconds", "jsonl_sha256",
                                         "unconsumed_temp_images", "logs")} for m in mans],
           "code": code_state()}
    _write_atomic(out / "manifest.json", json.dumps(man, indent=1) + "\n")
    c = man["counts"]
    print(f"merge: {st} {dst}: {c['records']} demos ({c['by_split_kind']}), {c['dropped']} dropped, {imgs} images, "
          f"{nbytes / 1e6:.1f} MB")
    return man


# ---------------------------------------------------------------- main ---------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks-dir", type=Path, default=paths.WB_S2TRAIN_TASKS)
    ap.add_argument("--split-dir", type=Path, default=paths.WB_SPLIT)
    ap.add_argument("--size", default="%dx%d" % DEFAULT_SIZE)
    ap.add_argument("--out", type=Path, default=None, help="default OUT/walkerbench/demos/s2train_<W>x<H>")
    ap.add_argument("--work", type=Path, default=paths.scratch("wbdemo"), help="sandbox parent (removed afterwards)")
    ap.add_argument("--shard", default="0/1", help="i/n: this process renders sorted(stems)[i::n]")
    ap.add_argument("--limit", type=int, default=None, help="first N tasks of the shard (needs --out)")
    ap.add_argument("--stems", default=None, help="comma-separated stems (needs --out)")
    ap.add_argument("--merge", action="store_true", help="concatenate the shards into demos_all.jsonl + manifest.json")
    ap.add_argument("--allow-partial", action="store_true", help="--merge: accept --limit/--stems shards")
    ap.add_argument("--force", action="store_true", help="overwrite outputs in place")
    ap.add_argument("--keep-sandbox", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any task is dropped")
    a = ap.parse_args(argv)
    size = sandbox.parse_size(a.size)
    out = (a.out or default_out(size)).resolve()
    frozen = out == default_out(size).resolve()
    if a.merge:
        if frozen and a.allow_partial:
            ap.error("--allow-partial must not write into the frozen demos dir; pass --out")
        merge(out, size, a.tasks_dir, a.split_dir, a.allow_partial, a.force)
        return 0
    stems = [s.strip() for s in a.stems.split(",") if s.strip()] if a.stems else None
    if frozen and (a.limit is not None or stems):
        ap.error("--limit/--stems runs must not write into the frozen demos dir; pass --out")
    man = run_shard(a.tasks_dir, out, a.work, size, parse_shard(a.shard), a.limit, stems, a.split_dir, a.force,
                    a.keep_sandbox)
    for d in man["dropped"]:
        print(f"WARN dropped {d['id']}: {'; '.join(d['reasons'])}")
    # Drops are recorded, not fatal: merge lists them and tiers.materialize hard-fails on a missing train-rest demo.
    return 1 if (a.strict and man["dropped"]) else 0


if __name__ == "__main__":
    B.thread_env()
    sys.exit(main())
