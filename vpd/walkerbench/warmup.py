"""The 20-dialogue single-turn format warm-up set (PREREG_WB_PILOT section 5).

    python -m vpd.walkerbench.warmup                      # -> OUT/walkerbench/warmup/{warmup20.jsonl, images/, manifest.json}
    python -m vpd.walkerbench.warmup --limit 3 --out SCRATCH/warm

Port of scratchpad vpd_demos/b4_warmup.py (survey part 4), wording and selection unchanged:
- candidate panos: in a train-rest fence; not a spawn, target or visual_path pano of ANY task file in tasks/full or
  tasks/standard (all five types; the split's own task files are added too); its image present; at least 2 linked
  neighbours in the same fence with images;
- in each fence the candidates are sorted by md5(pano id) and the first is kept; the fences are sorted by that md5 and
  the first 20 used: dialogues 0-9 MOVE, 10-14 ROTATION, 15-19 STOP;
- each dialogue is a real create_session at that pano (spawn heading round(int(md5[:6], 16) % 3600 / 10, 1)) in a
  sandbox rendering at --size; the user text is the agent's own step-1 text (_build_user_message) with the **Task:**
  text replaced by one of the three frozen instructions below, '<image>' appended; the system prompt is the agent's
  (config/system_prompt_nav.txt, verbatim); the reply is json.dumps({"THOUGHT": "", "ACTION": ...}).
Asserts: the fences are disjoint from s2test and s2val; the system prompt equals system_prompt_nav.txt; all replies
round-trip through the agent's _parse_response and match the step's constrained-decoding regex; image size == --size.
The survey prototype's output (plan/login_tmp/vpd_demos/warmup_proto/warmup20.jsonl) is compared message by message
(--proto-check strict: any difference fails a full run) and image by image (recorded only).
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path

from vpd import paths
from vpd.walkerbench import action_regex as AR
from vpd.walkerbench import baselines as B
from vpd.walkerbench import demos as D
from vpd.walkerbench import sandbox
from vpd.walkerbench import split as S

SCHEMA = "vpd.walkerbench.warmup/1"
N_DIALOGUES = 20
PLAN = ["move"] * 10 + ["rotation"] * 5 + ["stop"] * 5
ROT_SIDES = [("LEFT", -90), ("RIGHT", 90), ("BACK", 180), ("RIGHT", 90), ("LEFT", -90)]  # by i % 5
# Frozen wording (b4_warmup.py lines 60, 65, 68; PREREG_WB_PILOT section 5).  Do not edit.
T_MOVE = "Format practice, not a navigation task: take available move %d (%s)."
T_ROT = "Format practice, not a navigation task: without moving, look %s (heading=%s, pitch=0)."
T_STOP = "Format practice, not a navigation task: you are already at the destination. Stop now."
SPAWN_KEYS = ("spawn_point", "spawn_pano_id", "start_pano_id")
PROTO = paths.PLAN_DIR / "login_tmp" / "vpd_demos" / "warmup_proto" / "warmup20.jsonl"
PROTO_SHA256 = "8d0c643e2bb2c926192321353ba1f413d323ff8190c07c301dd33d48d39ce391"  # PREREG section 5


def md5(x: str) -> str:
    return hashlib.md5(x.encode()).hexdigest()


def spawn_heading(h: str) -> float:
    return round(int(h[:6], 16) % 3600 / 10, 1)


# ---------------------------------------------------------------- selection (pure) ---------------------------------

def used_panos(task_files) -> set[str]:
    used = set()
    for p in task_files:
        t = json.loads(Path(p).read_text())
        for st in t.get("visual_path") or []:
            used.add(st["pano_id"])
        for k in SPAWN_KEYS:
            if t.get(k):
                used.add(t[k])
        for x in t.get("target_pano_ids") or []:
            used.add(x)
    return used


def select(rest_fences, fences: dict[str, set], links: dict[str, list], imgs: set[str], used: set[str],
           n: int = N_DIALOGUES) -> tuple[list[tuple[str, str, str]], dict]:
    """[(md5, fence, pano)] for the first n fences, and per-fence candidate counts."""
    cands, counts = [], {}
    for g in sorted(rest_fences):
        fence = fences[g]
        c = sorted((x for x in fence if x in imgs and x not in used and x in links
                    and sum(v in fence and v in imgs for v in links[x]) >= 2), key=md5)
        counts[g] = len(c)
        if c:
            cands.append((md5(c[0]), g, c[0]))
    cands.sort()
    return cands[:n], counts


def instruction(i: int, obs: dict) -> tuple[str, str]:
    """(instruction text, ACTION string) of dialogue i on its step-1 observation (b4_warmup.py lines 57-69)."""
    kind = PLAN[i]
    if kind == "move":
        moves = obs["available_moves"]
        k = moves[i % len(moves)]
        return T_MOVE % (k["id"], k["direction"]), "MOVE(%d)" % k["id"]
    if kind == "rotation":
        side, d = ROT_SIDES[i % 5]
        hh = round((obs["heading"] + d) % 360, 1)
        return T_ROT % (side, hh), "ROTATION(%s, 0)" % hh
    return T_STOP, AR.STOP_ACTION


def load_inputs(repo: Path, panos: Path, split_dir: Path):
    db = sqlite3.connect(f"file:{repo}/data/cache.db?mode=ro&immutable=1", uri=True)
    try:
        links = {p: [(l.get("panoId") or l.get("pano_id")) for l in (json.loads(x) if x else [])]
                 for p, x in db.execute("select pano_id, links from metadata")}
    finally:
        db.close()
    imgs = {f[:-7] for f in os.listdir(panos) if f.endswith("_z2.jpg")}
    fences = {k: set(v) for k, v in json.loads((repo / "config" / "geofence_config.json").read_text()).items()}
    split_doc = json.loads((split_dir / "split_s2.json").read_text())
    files = sorted(glob.glob(str(repo / "tasks" / "full" / "*.json")) + glob.glob(str(repo / "tasks" / "standard" / "*.json")))
    files += sorted(str(p) for d in S.SPLIT_DIRS for p in (split_dir / d / "tasks").glob("*.json"))
    return links, imgs, fences, split_doc, files


def fence_sets(split_doc: dict) -> tuple[list[str], set[str]]:
    t, sp = split_doc["tasks"], split_doc["splits"]
    rest = sorted({t[s]["geofence"] for s in sp["train_rest"]})
    other = {t[s]["geofence"] for s in sp["s2test"] + sp["s2val"]}
    if set(rest) & other:
        raise RuntimeError(f"train-rest fences overlap test/val fences: {sorted(set(rest) & other)[:3]}")
    return rest, other


def task_json(tid: str, fence: str, pano: str, heading: float) -> dict:
    return {"task_id": tid, "task_type": "format_warmup", "geofence": fence, "spawn_point": pano,
            "spawn_heading": heading, "description": "", "ground_truth": {}, "target_pano_ids": [pano],
            "visual_path": [{"step_index": 0, "pano_id": pano}]}


def compare_proto(recs: list[dict], out_imgs: list[str], proto: Path) -> dict:
    """Message-by-message and image-byte comparison with the survey prototype output."""
    if not proto.exists():
        return {"path": str(proto), "present": False}
    prev = [json.loads(line) for line in proto.read_text(encoding="utf-8").splitlines() if line.strip()]
    by = {r["id"]: r for r in prev}
    msg_diff = [r["id"] for r in recs if r["id"] not in by or by[r["id"]]["messages"] != r["messages"]
                or by[r["id"]]["source"] != r["source"] or by[r["id"]]["task_type"] != r["task_type"]]
    img_same, img_missing = 0, 0
    for r, mine in zip(recs, out_imgs):
        q = proto.parent / "images" / Path(by.get(r["id"], {}).get("images", ["?"])[0]).name
        if not q.exists():
            img_missing += 1
        elif S.sha256_file(q) == S.sha256_file(mine):
            img_same += 1
    return {"path": str(proto), "present": True, "sha256": S.sha256_file(proto),
            "sha256_expected": PROTO_SHA256, "n_proto": len(prev), "n_compared": len(recs),
            "messages_differ": msg_diff, "images_byte_identical": img_same, "images_missing": img_missing}


# ---------------------------------------------------------------- build --------------------------------------------

def build(out: Path, work: Path, size: tuple[int, int] = D.DEFAULT_SIZE, limit: int | None = None,
          split_dir: Path = paths.WB_SPLIT, repo: Path = paths.REPO_WB, panos: Path = paths.PANOS,
          proto: Path = PROTO, proto_check: str = "strict", force: bool = False, keep_sandbox: bool = False) -> dict:
    out, work, split_dir, repo, panos = (Path(x).resolve() for x in (out, work, split_dir, repo, panos))
    n = N_DIALOGUES if limit is None else min(limit, N_DIALOGUES)
    name = f"warmup{n}.jsonl"
    jl, mf = out / name, out / "manifest.json"
    img_dir = out / "images"
    if (jl.exists() or mf.exists()) and not force:
        raise FileExistsError(f"{jl} or {mf} exists (use --force to overwrite in place; no job may read it)")
    t_all = time.time()
    links, imgs, fences, split_doc, files = load_inputs(repo, panos, split_dir)
    rest, other = fence_sets(split_doc)
    used = used_panos(files)
    picks, counts = select(rest, fences, links, imgs, used)
    if len(picks) < N_DIALOGUES:
        raise RuntimeError(f"only {len(picks)} train-rest fences have a candidate pano, need {N_DIALOGUES}")
    picks = picks[:n]
    stamp = B.utc_stamp()
    wdir = work / f"warmup_{stamp}"
    src = wdir / "src" / "warmup" / "tasks"
    src.mkdir(parents=True)
    tasks = {}
    for i, (h, g, pano) in enumerate(picks):
        tid = "nav_warm_%02d" % i
        tasks[tid] = task_json(tid, g, pano, spawn_heading(h))
        (src / f"{tid}.json").write_text(json.dumps(tasks[tid]))
    copy, diff = sandbox.make_sandbox(wdir / "sbx", src, render_size=size, exclude=sandbox.LEAN_EXCLUDE)
    cwd = os.getcwd()
    recs, saved_all, checks = [], [], []
    try:
        wb = D.load_engine(copy, sandbox.task_label(src), size)
        nav_prompt = (copy / "config" / "system_prompt_nav.txt").read_text(encoding="utf-8").strip()
        repo_prompt = (repo / "config" / "system_prompt_nav.txt").read_text(encoding="utf-8").strip()
        if nav_prompt != repo_prompt:
            raise RuntimeError("sandbox system_prompt_nav.txt differs from the checkout")
        img_dir.mkdir(parents=True, exist_ok=True)
        for i, (h, g, pano) in enumerate(picks):
            tid = "nav_warm_%02d" % i
            with contextlib.redirect_stdout(io.StringIO()):
                ag = wb.rbp.VLNAgent(wb.rbp.get_agent_config("warmup"))
            saved = []

            def grab(url, _s=saved, _t=tid):
                s_ = wb.copy / str(url).lstrip("/")
                if not s_.is_file():
                    return None
                dst = img_dir / f"{_t}.jpg"
                shutil.copyfile(s_, dst)
                s_.unlink()  # this sandbox's temp file
                _s.append(str(dst))
                return "x"

            ag.get_image_base64 = grab
            obs = ag.create_session(tid, "warmup")
            if obs["pano_id"] != pano or abs(obs["heading"] - tasks[tid]["spawn_heading"]) > 1e-9:
                raise RuntimeError(f"{tid}: session at {obs['pano_id']} heading {obs['heading']}, expected {pano}")
            instr, act = instruction(i, obs)
            obs["task_description"] = instr
            msg = ag._build_user_message(obs)
            if len(msg["content"]) != 2 or len(saved) != 1:
                raise RuntimeError(f"{tid}: user message has {len(msg['content'])} parts, {len(saved)} images")
            text = msg["content"][0]["text"]
            if text.count("**Task:** " + instr) != 1:
                raise RuntimeError(f"{tid}: the instruction is not on the **Task:** line")
            if ag.system_prompt != nav_prompt:
                raise RuntimeError(f"{tid}: system prompt != config/system_prompt_nav.txt")
            reply = AR.reply(act)
            rec = {"id": tid, "task_type": "warmup_" + PLAN[i], "messages": [
                {"role": "system", "content": ag.system_prompt},
                {"role": "user", "content": text + D.IMAGE},
                {"role": "assistant", "content": reply}], "images": saved,
                "source": {"pano": pano, "geofence": g}}
            back = ag._parse_response(reply, dict(obs, heading=-1.0, pitch=-1.0))
            want = D._fallback_parse(reply)
            rx = D.regex_ok(reply, len(obs["available_moves"]))
            probs = D.check_record(dict(rec, n_turns=1, size=list(size)), size, parse=ag._parse_response)
            if PLAN[i] == "stop":
                probs = [p for p in probs if "last reply" not in p] + ([] if reply == D.STOP_REPLY else ["not STOP"])
            else:
                probs = [p for p in probs if "last reply" not in p]  # single-turn: only STOP dialogues end in STOP
            if not D.same_action(back, want) or not rx or probs:
                raise RuntimeError(f"{tid}: reply {reply} parse {back} regex {rx} problems {probs}")
            checks.append({"id": tid, "kind": PLAN[i], "action": act, "n_moves": len(obs["available_moves"]),
                           "parsed": {k: back.get(k) for k in ("type", "move_id", "heading", "pitch", "answer")
                                      if k in back}, "regex_ok": rx})
            recs.append(rec)
            saved_all.append(saved[0])
            print(f"{tid} {PLAN[i]:8s} {g} moves {len(obs['available_moves'])} -> {act}", flush=True)
    finally:
        os.chdir(cwd)
        if not keep_sandbox:
            shutil.rmtree(wdir)  # this run's own scratch (task JSONs + sandbox)
    cmp = compare_proto(recs, saved_all, Path(proto)) if proto_check != "off" else {"skipped": True}
    if proto_check == "strict" and limit is None:
        if not cmp.get("present"):
            raise RuntimeError(f"prototype {proto} missing (use --proto-check warn/off)")
        if cmp["messages_differ"] or cmp["sha256"] != PROTO_SHA256:
            raise RuntimeError(f"warm-up differs from the frozen prototype: {cmp}")
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    D._write_atomic(jl, text)
    kinds = {k: PLAN[:n].count(k) for k in ("move", "rotation", "stop")}
    man = {"schema": SCHEMA, "created_utc": D.utc_now(), "n": len(recs), "file": name,
           "sha256": D.sha256_text(text), "kinds": kinds, "size": list(size),
           "templates": {"move": T_MOVE, "rotation": T_ROT, "stop": T_STOP},
           "system_prompt_sha256": D.sha256_text(recs[0]["messages"][0]["content"]) if recs else None,
           "fences": [g for _, g, _ in picks], "panos": [p for _, _, p in picks],
           "spawn_headings": [spawn_heading(h) for h, _, _ in picks],
           "selection": {"train_rest_fences": len(rest), "fences_with_candidate": sum(1 for v in counts.values() if v),
                         "candidate_panos": sum(counts.values()), "used_panos": len(used),
                         "task_files_scanned": len(files), "test_val_fences": len(other),
                         "disjoint_from_test_val": True},
           "images": [{"file": Path(p).name, "sha256": S.sha256_file(p)} for p in saved_all],
           "checks": checks, "prototype": cmp, "split_s2_sha256": S.sha256_file(split_dir / "split_s2.json"),
           "sandbox_settings_diff": diff, "walkerbench_commit": sandbox.repo_commit(repo),
           "seconds": round(time.time() - t_all, 1),
           "code": dict(D.code_state(), **{"warmup.py": S.sha256_file(Path(__file__).resolve())})}
    D._write_atomic(mf, json.dumps(man, indent=1) + "\n")
    print(f"warm-up: {len(recs)} dialogues {kinds} -> {jl}; prototype messages differ: "
          f"{cmp.get('messages_differ')}, images byte-identical {cmp.get('images_byte_identical')}/{len(recs)}")
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=paths.WB_WARMUP)
    ap.add_argument("--work", type=Path, default=paths.scratch("wbdemo"), help="sandbox parent (removed afterwards)")
    ap.add_argument("--size", default="%dx%d" % D.DEFAULT_SIZE)
    ap.add_argument("--limit", type=int, default=None, help="first N dialogues (needs --out)")
    ap.add_argument("--split-dir", type=Path, default=paths.WB_SPLIT)
    ap.add_argument("--proto", type=Path, default=PROTO)
    ap.add_argument("--proto-check", choices=("strict", "warn", "off"), default="strict")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-sandbox", action="store_true")
    a = ap.parse_args(argv)
    if a.out.resolve() == paths.WB_WARMUP.resolve() and a.limit is not None:
        ap.error("--limit runs must not write into the frozen warm-up dir; pass --out")
    if a.limit is not None and a.limit < 1:
        ap.error("--limit must be >= 1")
    build(a.out, a.work, sandbox.parse_size(a.size), a.limit, a.split_dir, proto=a.proto, proto_check=a.proto_check,
          force=a.force, keep_sandbox=a.keep_sandbox)
    return 0


if __name__ == "__main__":
    B.thread_env()
    sys.exit(main())
