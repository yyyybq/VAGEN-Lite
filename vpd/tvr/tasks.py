"""TVR task sets, the dev selection, the public expert sequences and the demo tiers (CONTRACT_wave2 4.1, 5.1.2, 6.1,
6.6; DECISIONS G10 (b) (c) (d)).

Named task sets (task_set(name)): sft1600 (sft.json order), eval500 (eval.json order), rl4800 (rl.json order),
dev200 (selection.json order), smoke8 / smoke4 (selection.json; the first 2 / 1 dev200 ids of each category in dev
order), sft2 (SFT2_IDS, the probe's public sample tasks).

Dev selection (G10(d)): per category, the rl.json tasks sorted by sha256("tvr-dev-v1|" + task_id) are taken in
that order, skipping a task whose (category, scene) already holds 2; stop at 50.  Output order: SR-easy block,
SR-hard, LR-easy, LR-hard (each in hash order).  Scenes are disjoint from sft and eval by construction
(scene_splits.json); asserted.  eval.json is never used for any selection.

0W pool (G10(b)): per category the 5 kept demos with the smallest sha256("tvr-0w-v1|" + id).  Tier pool = kept
minus 0W.  Tiers (G10(c)): per category the pool ordered by sha256("tvr-tier-v1|" + id); tier N = the first N/4
per category; ALL = the whole pool; every file lists category blocks in CATEGORIES order, hash order inside.

CLI (PY_Q3, 1 core, stdlib):
  python -B -m vpd.tvr.tasks dev-select --out OUT/tvr/split/dev200        (exit 0 written / identical kept; 2 refused)
  python -B -m vpd.tvr.tasks tiers --demos OUT/tvr/demos/sft1600_lvp/demos_all.jsonl --out OUT/tvr/tiers/v1
                                                                          (exit 0 written / identical no-op; 2 refused)
      tiers refuses (exit 2) unless the demos dir holds DONE and a manifest.json whose demos_all_sha256 matches.
  The --tasks-dir / --per-category / --zero-w-per-category / --sizes options exist for fake trees in tests; the
  data job passes none of them (the outputs record the values used).

Also holds the small I/O helpers of the tvrdata modules (atomic JSON, sha256, code_sha256).  Python 3.10 / 3.12,
stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from vpd import paths
from vpd.tvr import lvp, pilot_cfg, tvr_text

CATEGORIES = pilot_cfg.CATEGORIES
SALTS = pilot_cfg.FROZEN["salts"]
DEV = pilot_cfg.FROZEN["dev"]
ZERO_W = pilot_cfg.FROZEN["zero_w"]
SFT2_IDS = ("ithor_FloorPlan11_easy_t00_s00", "procthor_procthor_09990_hard_t19_s01")
SPLIT_OF_SET = {"sft1600": "sft", "eval500": "eval", "rl4800": "rl"}
DEV_SETS = ("dev200", "smoke8", "smoke4")
TASK_SETS = ("sft1600", "eval500", "rl4800", "dev200", "smoke8", "smoke4", "sft2")
TIER_SIZES = (100, 400)
DEV_RULE = ("per category: the rl.json tasks sorted by sha256(salt + task_id) ascending, taken in that order, "
            "skipping a task whose (category, scene) already holds max_per_scene_category; stop at per_category. "
            "Output order: SR-easy, SR-hard, LR-easy, LR-hard blocks, hash order inside. smoke8 / smoke4 = the "
            "first 2 / 1 ids of each category in dev order. Scenes disjoint from sft and eval (asserted).")
DATA_CODE_FILES = ("vpd/tvr/pilot_cfg.py", "vpd/tvr/tvr_text.py", "vpd/tvr/kin_env.py", "vpd/tvr/tasks.py",
                   "vpd/tvr/bfs.py", "vpd/tvr/reach.py", "vpd/tvr/floors.py", "vpd/tvr/demos.py",
                   "vpd/tvr/lvp.py")


# --- small helpers (shared by the tvrdata modules) -------------------------------------------------------------------
def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_bytes_atomic(path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def json_bytes(obj) -> bytes:
    """CONTRACT R7: UTF-8, indent=1."""
    return (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path, obj) -> str:
    """Atomic JSON write (tmp + os.replace); returns the sha256 of the bytes written."""
    data = json_bytes(obj)
    write_bytes_atomic(path, data)
    return sha256_bytes(data)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


VOLATILE_KEYS = ("created_utc", "code_sha256")


def same_content(old, new, volatile=VOLATILE_KEYS) -> bool:
    """Parsed JSON equality ignoring the top-level volatile keys (a timestamp, the code fingerprint: a rerun after
    the orchestrator's pilot_cfg fill must not change a file whose data are the same)."""
    if not isinstance(old, dict) or not isinstance(new, dict):
        return old == new
    a = {k: v for k, v in old.items() if k not in volatile}
    b = {k: v for k, v in json.loads(json.dumps(new)).items() if k not in volatile}
    return a == b


def write_json_stable(path, doc) -> tuple:
    """Write doc unless an existing file holds the same content apart from VOLATILE_KEYS (then it is kept
    byte-for-byte, so its sha256 stays the one pilot_cfg may record); returns (sha256, kept)."""
    path = Path(path)
    if path.is_file():
        try:
            if same_content(read_json(path), doc):
                return sha256_file(path), True
        except (OSError, ValueError):
            pass
    return write_json(path, doc), False


def code_sha256(rels: Sequence[str] = DATA_CODE_FILES, root=None) -> str:
    """sha256 of the '<rel> <sha256>' lines of the code files (run_vpd.code_sha256 convention)."""
    root = Path(paths.WORKTREE if root is None else root)
    lines = []
    for rel in rels:
        p = root / rel
        lines.append(f"{rel} {sha256_file(p) if p.is_file() else 'missing'}")
    return sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def ids_sha256(ids: Iterable[str]) -> str:
    return sha256_bytes("\n".join(ids).encode("utf-8"))


def out_path(spec: str) -> Path:
    """A pilot_cfg path ("OUT/...") as an absolute path."""
    return paths.OUT / spec[len("OUT/"):] if spec.startswith("OUT/") else Path(spec)


PUBLIC_JSON = out_path(pilot_cfg.FROZEN["public"]["sft_train_json"])


# --- tasks ----------------------------------------------------------------------------------------------------------
def load_split(name: str, tasks_dir=None) -> List[Dict]:
    """data/tasks/<name>.json of the pinned checkout ("sft" | "eval" | "rl")."""
    if name not in ("sft", "eval", "rl"):
        raise ValueError(f"unknown split {name!r}")
    return lvp.load_tasks(name, lvp.TASKS_DIR if tasks_dir is None else tasks_dir)


def by_id(tasks: Iterable[Mapping]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for t in tasks:
        if t["task_id"] in out:
            raise ValueError(f"duplicate task id {t['task_id']}")
        out[t["task_id"]] = dict(t)
    return out


def hash_key(salt: str, task_id: str) -> str:
    return hashlib.sha256((salt + task_id).encode("utf-8")).hexdigest()


def select_dev(rl_tasks: Sequence[Mapping], per_cat: int = DEV["per_category"],
               max_per_scene_cat: int = DEV["max_per_scene_category"], salt: str = SALTS["dev"],
               forbidden_scenes: Iterable[str] = ()) -> List[str]:
    """G10(d) dev selection (module doc); ValueError if a category cannot be filled or a chosen scene is in
    `forbidden_scenes` (the sft and eval scenes)."""
    forbidden = set(forbidden_scenes)
    out: List[str] = []
    for cat in CATEGORIES:
        pool = sorted((t for t in rl_tasks if pilot_cfg.category_of(t) == cat),
                      key=lambda t: hash_key(salt, t["task_id"]))
        per_scene: Dict[str, int] = {}
        chosen = []
        for t in pool:
            if len(chosen) >= per_cat:
                break
            s = t["scene"]
            if per_scene.get(s, 0) >= max_per_scene_cat:
                continue
            per_scene[s] = per_scene.get(s, 0) + 1
            chosen.append(t)
        if len(chosen) < per_cat:
            raise ValueError(f"{cat}: only {len(chosen)} of {per_cat} dev tasks available")
        bad = sorted({t["scene"] for t in chosen} & forbidden)
        if bad:
            raise ValueError(f"{cat}: dev scenes {bad} are sft / eval scenes")
        out.extend(t["task_id"] for t in chosen)
    return out


def _selection_path(root) -> Path:
    return Path(paths.TVR_OUT if root is None else root) / "split" / "dev200" / "selection.json"


def task_set(name: str, root=None, tasks_dir=None) -> List[Dict]:
    """The tasks of a named set (CONTRACT 4.1), in the set's order."""
    if name in SPLIT_OF_SET:
        return load_split(SPLIT_OF_SET[name], tasks_dir)
    if name == "sft2":
        sft = by_id(load_split("sft", tasks_dir))
        return [sft[i] for i in SFT2_IDS]
    if name in DEV_SETS:
        sel = read_json(_selection_path(root))
        ids = sel["ids"] if name == "dev200" else sel[name]
        rl = by_id(load_split("rl", tasks_dir))
        return [rl[i] for i in ids]
    raise ValueError(f"unknown task set {name!r} (one of {TASK_SETS})")


def task_sets(names: Iterable[str], root=None, tasks_dir=None) -> Dict[str, List[Dict]]:
    return {n: task_set(n, root, tasks_dir) for n in names}


def parse_sets(spec: str) -> List[str]:
    names = [s.strip() for s in spec.split(",") if s.strip()]
    for n in names:
        if n not in TASK_SETS:
            raise ValueError(f"unknown task set {n!r} (one of {TASK_SETS})")
    return names


# --- public expert sequences ----------------------------------------------------------------------------------------
_GPT_RE = re.compile(r"Action: (\w+)")


def public_sequences(path=None, sft_ids: Optional[Iterable[str]] = None, tasks_dir=None) -> Dict[str, Dict]:
    """sft_train.json -> {task_id: {"actions", "valid_lines", "n_turns", "n_images", "system_ok"}} in file order.

    task_id = images[0].split("/")[1].  Asserts (ValueError): ids unique and in sft.json; turns alternate
    human / gpt starting with human; every gpt value fullmatches "Action: (\\w+)" with a TVRBench action; exactly
    one Stop and it is last; 2 images per human turn, [images/<id>/step_kk.png, images/<id>/target.png]; every
    human value is build_concat_user_text(its valid list); system == tvr_text.SYSTEM_PROMPT.  The pinned file
    (PUBLIC_JSON, also when given explicitly) must have the registered sha256 (public.sft_train_json_sha256).
    """
    path = PUBLIC_JSON if path is None else Path(path)
    raw = path.read_bytes()
    if os.path.abspath(str(path)) == os.path.abspath(str(PUBLIC_JSON)):
        want = pilot_cfg.FROZEN["public"]["sft_train_json_sha256"]
        if sha256_bytes(raw) != want:
            raise ValueError(f"{path}: sha256 {sha256_bytes(raw)} != registered public.sft_train_json_sha256 {want}")
    data = json.loads(raw)
    if sft_ids is None:
        sft_ids = [t["task_id"] for t in load_split("sft", tasks_dir)]
    sft_ids = set(sft_ids)
    out: Dict[str, Dict] = {}
    for i, rec in enumerate(data):
        imgs = rec.get("images") or []
        if not imgs:
            raise ValueError(f"record {i}: no images")
        tid = imgs[0].split("/")[1]
        where = f"record {i} ({tid})"
        if tid in out:
            raise ValueError(f"{where}: duplicate task id")
        if tid not in sft_ids:
            raise ValueError(f"{where}: not an sft.json task")
        if rec.get("system") != tvr_text.SYSTEM_PROMPT:
            raise ValueError(f"{where}: system prompt differs from TVRBench's SYSTEM_PROMPT")
        conv = rec.get("conversations") or []
        if len(conv) % 2 or not conv:
            raise ValueError(f"{where}: odd or empty conversation")
        actions, valid_lines = [], []
        for k in range(0, len(conv), 2):
            h, g = conv[k], conv[k + 1]
            if h.get("from") != "human" or g.get("from") != "gpt":
                raise ValueError(f"{where}: turn {k // 2} is not human then gpt")
            m = _GPT_RE.fullmatch(g.get("value", ""))
            if not m or m.group(1) not in pilot_cfg.ACTIONS:
                raise ValueError(f"{where}: reply {g.get('value')!r} is not 'Action: <TVRBench action>'")
            actions.append(m.group(1))
            try:
                valid_lines.append(tvr_text.valid_from_user_text(h.get("value", "")))
            except ValueError as e:
                raise ValueError(f"{where}: turn {k // 2}: {e}") from None
        if actions.count("Stop") != 1 or actions[-1] != "Stop":
            raise ValueError(f"{where}: expected exactly one Stop, last (got {actions})")
        n = len(actions)
        want = []
        for k in range(n):
            want += [f"images/{tid}/step_{k:02d}.png", f"images/{tid}/target.png"]
        if list(imgs) != want:
            raise ValueError(f"{where}: images are not [step_kk, target] per human turn")
        out[tid] = {"actions": actions, "valid_lines": valid_lines, "n_turns": n, "n_images": len(imgs),
                    "system_ok": True}
    return out


def expert_medians(public: Mapping[str, Mapping], tasks_by_id: Mapping[str, Mapping]) -> Dict[str, int]:
    """statistics.median_low of the non-Stop action count per category over the public sequences."""
    counts: Dict[str, List[int]] = {c: [] for c in CATEGORIES}
    for tid, seq in public.items():
        counts[pilot_cfg.category_of(tasks_by_id[tid])].append(sum(1 for a in seq["actions"] if a != "Stop"))
    missing = [c for c, v in counts.items() if not v]
    if missing:
        raise ValueError(f"no public sequences in categories {missing}")
    return {c: int(statistics.median_low(v)) for c, v in counts.items()}


# --- 0W pool and tiers ----------------------------------------------------------------------------------------------
def _by_cat(ids: Iterable[str], tasks_by_id: Mapping[str, Mapping]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {c: [] for c in CATEGORIES}
    for i in ids:
        out[pilot_cfg.category_of(tasks_by_id[i])].append(i)
    return out


def zero_w_pool(kept_ids: Iterable[str], tasks_by_id: Mapping[str, Mapping],
                per_cat: int = ZERO_W["per_category"]) -> List[str]:
    """G10(b): per category the per_cat kept ids with the smallest hash_key("tvr-0w-v1|", id); category blocks in
    CATEGORIES order, hash order inside."""
    out: List[str] = []
    for cat, ids in _by_cat(kept_ids, tasks_by_id).items():
        ids = sorted(ids, key=lambda i: hash_key(SALTS["zero_w"], i))
        if len(ids) < per_cat:
            raise ValueError(f"{cat}: only {len(ids)} kept demos for a 0W pool of {per_cat}")
        out.extend(ids[:per_cat])
    return out


def tier_order(pool_ids: Iterable[str], tasks_by_id: Mapping[str, Mapping]) -> Dict[str, List[str]]:
    """G10(c): per category the pool ids by hash_key("tvr-tier-v1|", id) ascending."""
    return {cat: sorted(ids, key=lambda i: hash_key(SALTS["tier"], i))
            for cat, ids in _by_cat(pool_ids, tasks_by_id).items()}


def tier_ids(order: Mapping[str, Sequence[str]], n) -> List[str]:
    """Tier n (an int divisible by 4: the first n/4 per category) or "ALL" (every pool id); category blocks in
    CATEGORIES order, hash order inside."""
    if n == "ALL":
        return [i for c in CATEGORIES for i in order.get(c, [])]
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0 or n % 4:
        raise ValueError(f"tier size must be a positive int divisible by 4 or 'ALL', got {n!r}")
    k = n // 4
    out: List[str] = []
    for c in CATEGORIES:
        ids = list(order.get(c, []))
        if len(ids) < k:
            raise ValueError(f"tier {n}: {c} has only {len(ids)} pool demos (needs {k})")
        out.extend(ids[:k])
    return out


# --- CLI: dev-select ------------------------------------------------------------------------------------------------
def build_selection(tasks_dir=None, per_cat: int = DEV["per_category"],
                    max_per_scene_cat: int = DEV["max_per_scene_category"],
                    smoke_per_cat: int = DEV["smoke_per_category"]) -> Dict:
    td = Path(lvp.TASKS_DIR if tasks_dir is None else tasks_dir)
    rl = load_split("rl", td)
    forbidden = {t["scene"] for t in load_split("sft", td)} | {t["scene"] for t in load_split("eval", td)}
    ids = select_dev(rl, per_cat, max_per_scene_cat, SALTS["dev"], forbidden)
    rl_by = by_id(rl)
    by_cat = _by_cat(ids, rl_by)
    scenes: Dict[str, int] = {}
    for i in ids:
        scenes[rl_by[i]["scene"]] = scenes.get(rl_by[i]["scene"], 0) + 1
    return {"schema": "vpd.tvr.dev/1", "salt": SALTS["dev"], "rule": DEV_RULE,
            "rl_json_sha256": sha256_file(td / "rl.json"), "per_category": per_cat,
            "max_per_scene_category": max_per_scene_cat, "ids": ids, "by_category": by_cat,
            "scenes": dict(sorted(scenes.items())),
            "smoke8": [i for c in CATEGORIES for i in by_cat[c][:smoke_per_cat]],
            "smoke4": [i for c in CATEGORIES for i in by_cat[c][:1]],
            "created_utc": utc(), "code_sha256": code_sha256()}


def cmd_dev_select(a) -> int:
    out = Path(a.out) / "selection.json"
    sel = build_selection(a.tasks_dir, a.per_category, a.max_per_scene, a.smoke_per_category)
    if out.exists():
        try:
            old = read_json(out)
        except (OSError, ValueError) as e:
            print(f"[tasks] refusing: unreadable existing {out}: {e!r}")
            return 2
        same = (old.get("ids") == sel["ids"] and old.get("rl_json_sha256") == sel["rl_json_sha256"]
                and old.get("smoke8") == sel["smoke8"] and old.get("smoke4") == sel["smoke4"])
        if same:
            print(f"[tasks] {out} exists and is identical (ids, rl.json sha256): kept")
            return 0
        print(f"[tasks] refusing: {out} exists with a different selection")
        return 2
    sha = write_json(out, sel)
    print(f"[tasks] wrote {out} ({len(sel['ids'])} ids, {len(sel['scenes'])} scenes, sha256 {sha})")
    return 0


# --- CLI: tiers -----------------------------------------------------------------------------------------------------
def read_demo_lines(path) -> List[bytes]:
    """The non-empty lines (bytes, without the newline) of a JSONL file."""
    return [ln for ln in Path(path).read_bytes().split(b"\n") if ln.strip()]


def build_tiers(demos_path, tasks_dir=None, zero_w_per_cat: int = ZERO_W["per_category"],
                sizes: Sequence[int] = TIER_SIZES) -> Dict:
    """In-memory tier set: {"files": {name: bytes}, "order": dict, "manifest": dict (without created_utc)}."""
    lines = read_demo_lines(demos_path)
    recs: Dict[str, bytes] = {}
    n_turns: Dict[str, int] = {}
    for ln in lines:
        d = json.loads(ln)
        if d["id"] in recs:
            raise ValueError(f"duplicate demo id {d['id']}")
        recs[d["id"]] = ln
        n_turns[d["id"]] = int(d["n_turns"])
    tasks_by_id = by_id(load_split("sft", tasks_dir))
    for i in recs:
        if i not in tasks_by_id:
            raise ValueError(f"demo {i} is not an sft.json task")
    kept = list(recs)
    zw = zero_w_pool(kept, tasks_by_id, zero_w_per_cat)
    zset = set(zw)
    pool = [i for i in kept if i not in zset]
    order = tier_order(pool, tasks_by_id)
    tiers = {str(n): tier_ids(order, int(n)) for n in sizes}
    tiers["ALL"] = tier_ids(order, "ALL")
    longest = sorted(kept, key=lambda i: (-n_turns[i], i))[0]
    p3 = [longest]
    for cat in ("SR-hard", "LR-easy"):
        cand = [i for i in order[cat] if i not in p3]
        if not cand:
            raise ValueError(f"parity3: no {cat} pool demo")
        p3.append(cand[0])

    def body(ids):
        return b"".join(recs[i] + b"\n" for i in ids)

    files = {"warmup0w20.jsonl": body(zw)}
    for k, ids in tiers.items():
        files[f"demos_{k}.jsonl"] = body(ids)
    files["parity3.jsonl"] = body(p3)
    order_doc = {"salts": {"zero_w": SALTS["zero_w"], "tier": SALTS["tier"]}, "zero_w": zw, "pool": order,
                 "tiers": tiers, "parity3": p3}
    files["order.json"] = json_bytes(order_doc)
    tier_sets = [set(tiers[str(n)]) for n in sorted(sizes)] + [set(tiers["ALL"])]
    nested = all(a <= b for a, b in zip(tier_sets, tier_sets[1:]))
    disjoint = all(not (s & zset) for s in tier_sets)
    counts = {name: {c: 0 for c in CATEGORIES} for name in files if name.endswith(".jsonl")}
    for name in counts:
        for ln in files[name].split(b"\n"):
            if ln.strip():
                counts[name][pilot_cfg.category_of(tasks_by_id[json.loads(ln)["id"]])] += 1
    shas = {name: sha256_bytes(b) for name, b in files.items()}
    manifest = dict(shas)
    manifest.update({"schema": "vpd.tvr.tiers/1", "files": dict(shas), "counts": counts, "nested": nested,
                     "disjoint_from_0w": disjoint, "kept": len(kept), "zero_w_per_category": zero_w_per_cat,
                     "sizes": [int(n) for n in sizes], "demos_all": str(demos_path),
                     "demos_all_sha256": sha256_bytes(Path(demos_path).read_bytes()),
                     "default_sizes": (list(sizes) == list(TIER_SIZES)
                                       and zero_w_per_cat == ZERO_W["per_category"]),
                     "code_sha256": code_sha256()})
    if not nested or not disjoint:
        raise ValueError(f"tier invariants broken: nested {nested}, disjoint_from_0w {disjoint}")
    return {"files": files, "order": order_doc, "manifest": manifest}


def demos_done(demos_path) -> str:
    """The demos build that wrote demos_path completed: its dir holds DONE and a manifest.json whose
    demos_all_sha256 is demos_path's sha256.  Returns the manifest's sha256; ValueError otherwise."""
    p = Path(demos_path)
    if not (p.parent / "DONE").is_file():
        raise ValueError(f"{p.parent}: no DONE (the demos build did not complete)")
    man_p = p.parent / "manifest.json"
    man = read_json(man_p)
    if man.get("demos_all_sha256") != sha256_file(p):
        raise ValueError(f"{p}: sha256 differs from {man_p} demos_all_sha256 {man.get('demos_all_sha256')}")
    return sha256_file(man_p)


def cmd_tiers(a) -> int:
    out = Path(a.out)
    sizes = [int(s) for s in a.sizes.split(",") if s.strip()]
    demos_man_sha = demos_done(a.demos)
    ts = build_tiers(a.demos, a.tasks_dir, a.zero_w_per_category, sizes)
    ts["manifest"]["demos_manifest_sha256"] = demos_man_sha
    man_p = out / "manifest.json"
    if man_p.exists():
        same = all((out / n).is_file() and (out / n).read_bytes() == b for n, b in ts["files"].items())
        if same:
            print(f"[tasks] {out}: identical tier set exists: no-op")
            return 0
        print(f"[tasks] refusing: {out} holds a different tier set")
        return 2
    for n, b in ts["files"].items():
        write_bytes_atomic(out / n, b)
    man = dict(ts["manifest"], created_utc=utc())
    write_json(man_p, man)
    print(f"[tasks] wrote {out}: " + ", ".join(f"{n} {sum(c.values())}" for n, c in man["counts"].items()))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vpd.tvr.tasks", description="TVR dev selection and demo tiers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dev-select", help="write split/dev200/selection.json (G10(d))")
    d.add_argument("--out", required=True)
    d.add_argument("--tasks-dir", default=None, help="data/tasks dir (default: the pinned checkout's)")
    d.add_argument("--per-category", type=int, default=DEV["per_category"])
    d.add_argument("--max-per-scene", type=int, default=DEV["max_per_scene_category"])
    d.add_argument("--smoke-per-category", type=int, default=DEV["smoke_per_category"])
    t = sub.add_parser("tiers", help="write tiers/v1 (G10(b)(c)) from demos_all.jsonl")
    t.add_argument("--demos", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--tasks-dir", default=None)
    t.add_argument("--zero-w-per-category", type=int, default=ZERO_W["per_category"])
    t.add_argument("--sizes", default=",".join(str(n) for n in TIER_SIZES))
    a = ap.parse_args(argv)
    try:
        return cmd_dev_select(a) if a.cmd == "dev-select" else cmd_tiers(a)
    except (OSError, ValueError, KeyError) as e:
        print(f"[tasks] refusing: {e!r}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
