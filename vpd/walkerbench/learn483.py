"""EXPLORATORY WalkerBench learnability check: Q3B fine-tuned on all 483 train demos (user request 2026-11-09:
"只用 Q3B，在全部 483 道训练题上微调；不要做判断").  See vpd/EXPLORATORY_WB_LEARN483.md.

Not part of the pre-registered pilot (PREREG_WB_PILOT.md) and no decision rule: every output is numbers only.  Nothing
here writes under OUT/walkerbench/runs/pilot/ (the pilot's Q3B adapters 0W / 50 / 200 and its evals are read only;
GATE.md / GATE.json are never touched).  Root X = OUT/walkerbench/runs/explore/x483/Q3B.

Login node (PY_EVAL, 1 core, nice -n 19):
    python -m vpd.walkerbench.learn483 setup        # s2in50 task set + the X dirs, as the login user (before prepare)
    python -m vpd.walkerbench.learn483 floors --task-set s2val|s2in50 --out X/floors_<set>.json  (also in-job step 1)
    python -m vpd.walkerbench.learn483 analyze      # -> X/diag.json, X/REPORT_EXPLORATORY.md, X/tf/tf_summary.json
    python -m vpd.walkerbench.learn483 expect --arm A|B [--check]     # the outputs to verify before `vpd.sco delete`
In-job (vpd/jobs/wb_learn483.sh ARM RUN):
    PY_EVAL  -m ... init-check --init P/adapters/0W     # 0W adapter.pt sha256 5cdbfd4366a3, trainer code 63a3a8b92ce7
    PY_Q3    -m ... eval <run_vpd args>                 # run_vpd.main with the extra task set s2in50 (registered here,
                                                         # at run time: run_vpd.py itself is unchanged)
    PY_TRAIN -m ... tf --out X/tf --adapter 483A=DIR --adapter raw= --adapter 0W=DIR ...   # teacher-forced probe
    PY_TRAIN -m ... tf-parity --merged LOCAL --adapter X/adapters/483A --out X/tf/merge_parity.json
    PY_EVAL  -m ... parity-ok --file X/tf/merge_parity.json --adapter X/adapters/483A

Task set s2in50: the 50 task JSONs of tiers/seed1001/demos_50.jsonl copied from split/s2train/tasks to
OUT/walkerbench/split/s2in50/tasks (+ selection.json with ids and sha256).  These 50 tasks are in the training data
of the 50, 200 and 483 adapters (tiers nested: demos_all[:200] == demos_200, demos_50 == demos_200[:50]) and unseen by
raw and 0W.  split.check only walks s2test / s2val / s2train, and the pilot's eval-ok / adapter-ok checks do not read
TASK_SETS, so the extra set changes nothing of the pilot.

Teacher-forced probe (tf; PY_TRAIN = starVLA env, HF transformers, bf16 + float32 LoRA masters under autocast, sdpa,
no grad).  Base Q3B is loaded once with LoRA modules from downstream/train/lora.py (r16, alpha 32, dropout off, eval
mode); each adapter's A/B come from n2_train_vpd.load_init_adapter (raw = A = B = 0, exactly the base model); the
inputs are n2_train_vpd.samples / encode / reply_spans unchanged (full history, the same images, 640x400), so they are
bit-identical to training.  One forward per demo (logits only at the reply positions); demo-outer / adapter-inner, so
each demo is encoded once.  Demo sets: VAL56 (split=val rows of demos/s2train_640x400/demos_all.jsonl; 29 nav / 27
vis, val cities, unseen by every adapter), TR200 (tiers/seed1001/demos_200.jsonl; its first 50 = demos_50), TR283
(tiers/seed1001/demos_all.jsonl rows 201-483).  Per turn (X/tf/tf_<adapter>_<set>.jsonl):
- action type, restricted to the legal set as the eval grammar is: candidates STOP("ARRIVED"), MOVE(1),
  ROTATION(<expert heading, or the current heading>, 0), tokenized as replies (the expert's own reply for its type);
  at the first token where the candidates diverge, their log-probs are renormalised; candidates sharing that token are
  extended by the next token(s) while the shared prefix is the expert's (else the pair is reported as one group);
- MOVE id (expert MOVE turns): argmax over the legal single-digit ids 1..min(n, 9) at the position after the
  expert's "MOVE("; turns with n >= 10 legal moves or an expert id >= 10 are flagged;
- ROTATION (expert ROTATION turns): teacher-forced exact match of the integer heading digits (and a non-digit next);
  on VAL56 only, a greedy continuation of at most 8 tokens from "ROTATION(" gives |delta heading| in degrees;
- losses: reply-token CE (trainer weighting: per demo the summed CE over its reply tokens / its reply tokens) and the
  decision-token CE (type token + the argument's integer digits);
- next-action exact match: type and id; a rotation counts within 15 degrees (greedy, VAL56) or by the teacher-forced
  integer match (TR sets; a stricter rule, recorded per row as next_rule).
Summaries per adapter (X/tf/tf_summary_<adapter>.json; merged into X/tf/tf_summary.json): action-type accuracy and
per-class recall, STOP precision, legal-set next-action exact match, turn-node (expert ROTATION or MOVE(k>1)) vs
straight-node (MOVE(1)) accuracy, STOP AUROC (expert-STOP turn vs all others), mean P(STOP) at final / non-final
turns, greedy STOP recall at the final turn, nav P(STOP) by walked/stated bin, reference rows (type-only and type+id
marginal predictors, the rule "STOP iff walked >= stated" on nav) on the same demos.

Merge parity (tf-parity): TF reply CE on the first 20 VAL56 demos, merged bf16 483A folder vs unmerged LoRA.

Floors (floors): STOP0 / FWD / ODO / RM through vpd.walkerbench.baselines (load_walkerbench, run_policy; rendering
stubbed) on s2val or s2in50; baselines.py itself stays hard-wired to s2test and is not changed.

analyze: per eval (pilot Q3B evals on s2test, read only; X/evals/*) numbers per kind: success with Wilson 95%, oracle,
mean steps, share ending in STOP, cap hits, non-trivial success (spawn > 15 m from the target), P(STOP | <= 15 m) vs
P(STOP | > 15 m), the next action at the first arrival on the target pano, turn-node vs straight-node accuracy along
the expert path, per-step agreement with FWD, the counterfactual "own path + stop at walked >= stated" (nav), walked /
stated at STOP within +-20%; the floors, the training losses with the marginal references, the TF summary, merge parity
and the paper's numbers next to them.  Definitions follow the 11-09 diagnosis scripts (scratch/diag-failures diag.py,
diag3.py, diag4.py; diag-protocol odo_trunc.py) so that the pilot rows reproduce their numbers.  No verdict, no gate.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import uuid
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

from vpd import paths

SCHEMA = "vpd.walkerbench.learn483/1"
START = "Q3B"
X_ROOT = paths.WB_RUNS / "explore" / "x483" / START
PILOT_ROOT = paths.WB_RUNS / "pilot"
PILOT_Q3B = PILOT_ROOT / START
TIERS = paths.WB_TIERS / "seed1001"
DEMOS_ALL = paths.WB_DEMOS / "s2train_640x400" / "demos_all.jsonl"
S2IN50_DIR = paths.WB_SPLIT / "s2in50"
S2IN50_TASKS = S2IN50_DIR / "tasks"
TASK_DIRS = {"s2test": paths.WB_S2TEST_TASKS, "s2val": paths.WB_S2VAL_TASKS, "s2in50": S2IN50_TASKS}
CACHE_DB = paths.REPO_WB / "data" / "cache.db"
BASELINES_S2TEST = paths.WB_BASELINES / "baselines_s2test.json"
INIT_SHA_PREFIX = "5cdbfd4366a3"  # the pilot's Q3B 0W adapter.pt (reused, never retrained here)
CODE_SHA_PREFIX = "63a3a8b92ce7"  # n2_train_vpd code_fingerprint of the pilot adapters (adapter-ok provenance)
RANK, ALPHA, LORA_SEED = 16, 32.0, 1001
THR_M = 15.0
SETS = ("VAL56", "TR200", "TR283")
SET_SIZES = {"VAL56": 56, "TR200": 200, "TR283": 283}
GREEDY_SETS = ("VAL56",)
GREEDY_MAX_NEW = 8
ROT_OK_DEG = 15.0
PARITY_N = 20
TYPES = ("MOVE", "ROTATION", "STOP")
REPLY_PREFIX = '{"THOUGHT": "", "ACTION": "'
STOP_ACTION = 'STOP("ARRIVED")'
# walked/stated bins of the nav P(STOP) table: [0, .5), [.5, .8), [.8, .95), [.95, 1.05], > 1.05
STOP_BINS = ("[0,0.5)", "[0.5,0.8)", "[0.8,0.95)", "[0.95,1.05]", ">1.05")
ADAPTERS = ("raw", "0W", "50", "200", "483A", "483B")
# paper / diagnosis numbers shown next to the table (no rule attached)
REFERENCES = {
    "paper_Qwen3-VL-8B_Original": {"nav": 0.1029, "vis": 0.1000, "source": "WalkerBench paper Table 3 (n=350 each)"},
    "paper_Qwen3-VL-8B_Spatial-IDE": {"nav": 0.3229, "vis": 0.2857, "source": "WalkerBench paper (training-free scaffold)"},
    "ODO_full350_nav": {"k": 103, "n": 350, "rate": 0.2943, "source": "blind ODO on the paper's 350 nav tasks (diag-lit)"},
    "marginal_loss_demos_all": {"type_only": 0.0394, "with_ids": 0.0431,
                                "source": "11-09 diagnosis constants (per reply token; definition not restated); "
                                          "the TF summary's marginal_reference recomputes three stated variants on the "
                                          "same demos with exact token counts (login-node check 11-09 on TR200+TR283: "
                                          "type only 0.0287, uniform legal ids 0.0912)"},
    "marginal_loss_demos_200": {"type_only": 0.0404, "with_ids": 0.0444, "source": "11-09 diagnosis constants"},
}


def eval_name(model: str, task_set: str, res: str, mode: str, variant: str = "") -> str:
    return f"explore__{START}__{model}__{task_set}__{res}__{mode}" + (f"__{variant}" if variant else "")


# (label, model, task set, res, mode, variant) in the payload's order
ARM_EVALS = {
    "A": (("X1", "483A", "s2test", "640", "empty", ""), ("X9", "483A", "s2test", "640", "empty", "blind"),
          ("X2", "483A", "s2val", "640", "empty", ""), ("X5", "483A", "s2in50", "640", "empty", ""),
          ("X12", "483A", "s2test", "640", "empty", "rep2"), ("X3", "200", "s2val", "640", "empty", ""),
          ("X6", "200", "s2in50", "640", "empty", ""), ("X7", "50", "s2in50", "640", "empty", ""),
          ("X4", "raw", "s2val", "640", "empty", ""), ("X8", "raw", "s2in50", "640", "empty", ""),
          ("X10", "raw", "s2test", "640", "off", ""), ("X11", "raw", "s2test", "1280", "off", "")),
    "B": (("Y1", "483B", "s2test", "640", "empty", ""), ("Y2", "483B", "s2val", "640", "empty", ""),
          ("Y3", "483B", "s2in50", "640", "empty", "")),
}
ARM_TF = {"A": ("483A", "raw", "0W", "50", "200"), "B": ("483B",)}
ARM_TRAIN = {"A": ("483A", "auto"), "B": ("483B", "10")}


# ------------------------------------------------------------------------------------------------- utilities
def utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def thread_env(n: int | None = None) -> None:
    """One thread per process unless VPD_THREADS says otherwise (login-node rule)."""
    v = str(n or os.environ.get("VPD_THREADS") or 1)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = v
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def read_jsonl(p) -> list:
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_json(p, obj) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    os.replace(tmp, p)
    return p


def write_text(p, text: str) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)
    return p


def ids_sha(recs: list) -> str:
    return sha256_bytes("\n".join(r["id"] for r in recs).encode())


def is_under(path, root) -> bool:
    p, r = os.path.realpath(str(path)), os.path.realpath(str(root))
    return p == r or p.startswith(r.rstrip("/") + "/")


def refuse_pilot(path) -> None:
    """Every output of this exploratory run stays out of the pilot's tree (GATE.md / GATE.json untouched)."""
    if is_under(path, PILOT_ROOT):
        raise SystemExit(f"[learn483] refused: {path} is under {PILOT_ROOT} (the pilot's outputs are read only)")


def wilson(k: float, n: int, z: float = 1.959964) -> tuple:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / d
    return max(0.0, c - h), min(1.0, c + h)


def rate(k, n) -> dict:
    if not n:
        return {"k": k, "n": 0, "rate": None, "wilson95": None}
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4), "wilson95": [round(lo, 4), round(hi, 4)]}


def frac(k, n):
    return round(k / n, 4) if n else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 6) if xs else None


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return round(xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2, 4)


def hav(a, b) -> float:
    R = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = math.radians(b[0] - a[0]), math.radians(b[1] - a[1])
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(x), math.sqrt(1 - x))


def angdiff(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def auroc(pos: list, neg: list):
    """Mann-Whitney AUROC with average ranks for ties; None if a side is empty."""
    pos = [x for x in pos if x is not None]
    neg = [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg], key=lambda t: t[0])
    r_pos, i, n = 0.0, 0, len(allv)
    while i < n:
        j = i
        while j + 1 < n and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        r_pos += avg * sum(1 for k in range(i, j + 1) if allv[k][1])
        i = j + 1
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return round(u / (len(pos) * len(neg)), 4)


def stop_bin(ratio: float) -> str:
    if ratio < 0.5:
        return STOP_BINS[0]
    if ratio < 0.8:
        return STOP_BINS[1]
    if ratio < 0.95:
        return STOP_BINS[2]
    if ratio <= 1.05:
        return STOP_BINS[3]
    return STOP_BINS[4]


def load_coords(pano_ids, db=CACHE_DB) -> dict:
    """{pano: (lat, lng)} from WalkerBench's cache.db (read only, immutable), locations first, then metadata."""
    import sqlite3

    want = sorted({p for p in pano_ids if p})
    out = {}
    if not want or not Path(db).exists():
        return out
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    try:
        for tab in ("locations", "metadata"):
            todo = [p for p in want if p not in out]
            for i in range(0, len(todo), 500):
                chunk = todo[i:i + 500]
                q = f"SELECT pano_id, lat, lng FROM {tab} WHERE pano_id IN ({','.join('?' * len(chunk))})"
                for pid, lat, lng in con.execute(q, chunk):
                    out.setdefault(pid, (lat, lng))
    finally:
        con.close()
    return out


def spawn_distance(task: dict, coords: dict):
    """Metres from the spawn point to the nearest target (None without coordinates)."""
    sp = task.get("spawn_point")
    tg = [t for t in task.get("target_pano_ids") or [] if t in coords]
    if sp not in coords or not tg:
        return None
    return min(hav(coords[sp], coords[t]) for t in tg)


def _le(x, thr) -> bool:
    return x is not None and x <= thr


def task_panos(task: dict) -> set:
    s = {p.get("pano_id") for p in task.get("visual_path") or []}
    s.update(task.get("target_pano_ids") or [])
    s.add(task.get("spawn_point"))
    return {p for p in s if p}


# ------------------------------------------------------------------------------------------ demo parsing
MOVE_LINE = re.compile(r"^\s+(\d+): (.+?)(?: \(([0-9.]+)m\))?\s*$")
STATED_RE = re.compile(r"Total distance: approximately (\d+(?:\.\d+)?) m")
HEAD_RE = re.compile(r"keep current heading=(-?[0-9]+(?:\.[0-9]+)?(?:[eE]-?[0-9]+)?)")
COMPASS_RE = re.compile(r"- Heading: (-?[0-9]+(?:\.[0-9]+)?)")
ROT_RE = re.compile(r"ROTATION\((-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\)")


def parse_user_turn(text: str) -> dict:
    """Legal moves (id, direction, distance), current heading and the instruction's stated total distance of one
    stock WalkerBench user turn (examples/vln_agent.py _build_user_message)."""
    moves = []
    if "**Available Moves:**" in text:
        block = text.split("**Available Moves:**", 1)[1].split("\n\n", 1)[0]
        for line in block.splitlines():
            m = MOVE_LINE.match(line)
            if m:
                moves.append({"id": int(m.group(1)), "direction": m.group(2),
                              "distance": float(m.group(3)) if m.group(3) else None})
    st = STATED_RE.search(text)
    hd = HEAD_RE.search(text) or COMPASS_RE.search(text)
    return {"moves": moves, "stated_m": float(st.group(1)) if st else None,
            "heading": float(hd.group(1)) if hd else None}


def parse_action(action: str) -> dict:
    """The ACTION string of a canonical reply: MOVE(k) / ROTATION(h, p) / STOP(...)."""
    m = re.fullmatch(r"MOVE\((\d+)\)", action)
    if m:
        return {"type": "MOVE", "id": int(m.group(1))}
    m = ROT_RE.fullmatch(action)
    if m:
        return {"type": "ROTATION", "heading": float(m.group(1)), "heading_text": m.group(1),
                "pitch": float(m.group(2))}
    if action.startswith("STOP("):
        return {"type": "STOP"}
    raise ValueError(f"not a WalkerBench action: {action!r}")


def reply_text(action: str) -> str:
    return json.dumps({"THOUGHT": "", "ACTION": action})


def turn_table(rec: dict, task: dict | None = None, coords: dict | None = None) -> list:
    """Per assistant turn of a demo: expert action, legal moves, walked distance before the decision (sum of the
    shown distances of the moves taken), the stated total distance, and (with the task and coordinates) the pano the
    RM expert stands on and its distance to the nearest target.  The RM expert walks the task's visual_path
    (baselines.rm_expert): after a MOVE it is on vp[last index of the current pano + 1]."""
    vp = [p["pano_id"] for p in (task or {}).get("visual_path") or []]
    last = {p: i for i, p in enumerate(vp)}
    targets = [t for t in (task or {}).get("target_pano_ids") or [] if coords and t in coords]
    cur = (task or {}).get("spawn_point") or (vp[0] if vp else None)
    rows, walked, user, t = [], 0.0, None, 0
    for i, m in enumerate(rec["messages"]):
        if m["role"] == "user":
            user = parse_user_turn(m["content"])
            continue
        if m["role"] != "assistant":
            continue
        u = user or {"moves": [], "stated_m": None, "heading": None}
        act_text = json.loads(m["content"])["ACTION"]
        act = parse_action(act_text)
        dist = None
        if targets and cur in coords:
            dist = round(min(hav(coords[cur], coords[x]) for x in targets), 2)
        node = "stop" if act["type"] == "STOP" else ("straight" if act["type"] == "MOVE" and act["id"] == 1
                                                      else "turn")
        rows.append({"turn": t, "msg_index": i, "reply": m["content"], "expert": act_text, "expert_type": act["type"],
                     "expert_id": act.get("id"), "expert_heading": act.get("heading"),
                     "expert_heading_text": act.get("heading_text"), "heading_now": u["heading"],
                     "n_legal": len(u["moves"]), "walked_m": round(walked, 2), "stated_m": u["stated_m"],
                     "dist_to_target_m": dist, "pano": cur, "is_final": False, "node": node})
        if act["type"] == "MOVE":
            mv = next((x for x in u["moves"] if x["id"] == act["id"]), None)
            walked += (mv or {}).get("distance") or 0.0
            if cur in last and last[cur] + 1 < len(vp):
                cur = vp[last[cur] + 1]
            else:
                cur = None
        t += 1
    if rows:
        rows[-1]["is_final"] = True
    return rows


def demo_sets(demos_all=DEMOS_ALL, tiers=TIERS) -> "OrderedDict[str, list]":
    """VAL56 (split=val rows of demos_all), TR200 (demos_200), TR283 (tiers demos_all rows 201-483)."""
    allrecs = read_jsonl(demos_all)
    tr200 = read_jsonl(Path(tiers) / "demos_200.jsonl")
    tall = read_jsonl(Path(tiers) / "demos_all.jsonl")
    if [r["id"] for r in tall[:200]] != [r["id"] for r in tr200]:
        raise SystemExit(f"{tiers}: demos_all[:200] is not demos_200 (the tiers must be nested)")
    return OrderedDict([("VAL56", [r for r in allrecs if r.get("split") == "val"]), ("TR200", tr200),
                        ("TR283", tall[200:])])


def demo_task(rec: dict, tasks_dir=paths.WB_S2TRAIN_TASKS):
    f = Path(tasks_dir) / ((rec.get("source") or {}).get("file") or f"{rec['id']}.json")
    return read_json(f)


def adapter_ids(adapter_dir) -> set:
    """Ids of the demos an adapter was trained on (its meta.json 'data' file); empty for raw / unreadable."""
    if adapter_dir is None:
        return set()
    meta = read_json(Path(adapter_dir) / "meta.json") or {}
    data = meta.get("data")
    if not data or not Path(data).exists():
        return set()
    return {json.loads(x).get("id") for x in Path(data).read_text().splitlines() if x.strip()}


# ------------------------------------------------------------------------------------- TF: pure helpers
def common_prefix_len(seqs: list) -> int:
    n = min(len(s) for s in seqs)
    k = 0
    while k < n and all(s[k] == seqs[0][k] for s in seqs):
        k += 1
    return k


def cand_logprobs(cands: dict, expert_ids: list, lp) -> dict:
    """Action-type probabilities restricted to the candidates (the legal set, as the eval grammar).

    cands {type: reply token ids}; expert_ids: the expert's reply ids; lp(j, tok) = log p(token tok at reply index j |
    expert_ids[:j]) from the teacher-forced pass.  d = the first index where the candidates diverge (the expert must
    share the prefix).  Each candidate's log-prob is summed from d until its tokens are unique among the candidates;
    that is computable only while its tokens after d are the expert's (they are then on the forced path).  Candidates
    that share the token at d but leave the expert's path there form one group whose mass is p(token at d).  The
    entries are renormalised: {"d", "p": {type: prob | None}, "groups": {"A|B": prob}, "pred", "resolved"}."""
    types = list(cands)
    d = common_prefix_len([cands[t] for t in types])
    if list(expert_ids[:d]) != list(cands[types[0]][:d]):
        raise ValueError("the expert reply does not share the candidates' common prefix")
    logp = {}
    for t in types:
        c = cands[t]
        k = d
        while k < len(c) and any(o != t and cands[o][:k + 1] == c[:k + 1] for o in types):
            k += 1
        if k >= len(c) or list(c[d:k]) != list(expert_ids[d:k]):
            logp[t] = None
            continue
        logp[t] = sum(lp(j, c[j]) for j in range(d, k + 1))
    groups = OrderedDict()
    for t in types:
        if logp[t] is None:
            groups.setdefault(cands[t][d], []).append(t)
    entries = OrderedDict(((t,), v) for t, v in logp.items() if v is not None)
    for tok, members in groups.items():
        entries[tuple(members)] = lp(d, tok)
    mx = max(entries.values())
    z = sum(math.exp(v - mx) for v in entries.values())
    probs = OrderedDict((k, math.exp(v - mx) / z) for k, v in entries.items())
    best = max(probs, key=lambda k: probs[k])
    return {"d": d, "p": {t: (probs[(t,)] if (t,) in probs else None) for t in types},
            "groups": {"|".join(k): v for k, v in probs.items() if len(k) > 1}, "pred": "|".join(best),
            "resolved": not groups}


def candidate_texts(turn: dict) -> dict:
    """The three candidate replies of one turn; the expert's type is its own reply."""
    h = turn["expert_heading"] if turn["expert_type"] == "ROTATION" else turn.get("heading_now")
    h = 0.0 if h is None else float(h) % 360
    out = {"MOVE": reply_text("MOVE(1)"), "ROTATION": reply_text("ROTATION(%s, 0)" % round(h, 1)),
           "STOP": reply_text(STOP_ACTION)}
    out[turn["expert_type"]] = turn.get("reply") or reply_text(turn["expert"])
    return out


def next_ok(row: dict, greedy: bool):
    """Legal-set next-action exact match: type + id; a rotation within ROT_OK_DEG (greedy) or by TF integer match."""
    if row.get("pred_type") is None:
        return None, None
    if row["pred_type"] != row["expert_type"]:
        return False, "type"
    if row["expert_type"] == "STOP":
        return True, "type"
    if row["expert_type"] == "MOVE":
        return (row.get("move_pred_id") == row["expert_id"]), "type+id"
    if greedy and row.get("rot_abs_delta") is not None:
        return row["rot_abs_delta"] <= ROT_OK_DEG, "greedy15"
    if greedy and row.get("rot_greedy_text") is not None:
        return False, "greedy15"  # the continuation did not start with a number
    return bool(row.get("rot_tf_int_match")), "tf_int"


# ------------------------------------------------------------------------------------- TF: summaries
def marginal_costs(rows: list) -> dict:
    """Reference predictors on the same turns, which see no input: (a) the empirical action-type frequencies
    (type_only); (b) (a) plus the empirical MOVE-id distribution (type_emp_id); (c) (a) plus a uniform guess over the
    legal ids (type_uniform_id).  The fixed JSON tokens and the rotation heading cost nothing here.  Costs in nats per
    turn and per reply token with the trainer's weighting (per demo summed cost / its reply tokens, averaged over
    demos).  The 11-09 diagnosis constants (REFERENCES) used a definition that is not reproduced here."""
    n = len(rows)
    if not n:
        return {}
    f = Counter(r["expert_type"] for r in rows)
    p = {t: f[t] / n for t in f}
    ids = Counter(r.get("expert_id") for r in rows if r["expert_type"] == "MOVE")
    nm = sum(ids.values())
    by_demo = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    tot = [0.0, 0.0, 0.0]
    for r in rows:
        a = -math.log(p[r["expert_type"]])
        b = a - (math.log(ids[r.get("expert_id")] / nm) if r["expert_type"] == "MOVE" else 0.0)
        c = a + (math.log(max(1, r.get("n_legal") or 1)) if r["expert_type"] == "MOVE" else 0.0)
        d = by_demo[r["demo"]]
        for i, v in enumerate((a, b, c)):
            d[i] += v
            tot[i] += v
        d[3] += r.get("n_reply_tokens") or 0
    tok_ok = [d for d in by_demo.values() if d[3]]

    def per_tok(i):
        return round(sum(d[i] / d[3] for d in tok_ok) / len(tok_ok), 5) if tok_ok else None

    move1 = sum(1 for r in rows if r["expert_type"] == "MOVE" and r.get("expert_id") == 1)
    return {"type_freq": {t: round(v, 4) for t, v in sorted(p.items())},
            "majority_type": max(sorted(p), key=p.get), "majority_type_acc": round(max(p.values()), 4),
            "always_MOVE1_exact": frac(move1, n),
            "type_only_nats_per_turn": round(tot[0] / n, 5), "type_emp_id_nats_per_turn": round(tot[1] / n, 5),
            "type_only_per_reply_token": per_tok(0), "type_emp_id_per_reply_token": per_tok(1),
            "type_uniform_id_per_reply_token": per_tok(2)}


def _binary(pred: list, gold: list) -> dict:
    tp = sum(1 for a, b in zip(pred, gold) if a and b)
    fp = sum(1 for a, b in zip(pred, gold) if a and not b)
    fn = sum(1 for a, b in zip(pred, gold) if not a and b)
    n = len(gold)
    return {"n": n, "acc": frac(sum(1 for a, b in zip(pred, gold) if a == b), n), "stop_recall": frac(tp, tp + fn),
            "stop_precision": frac(tp, tp + fp), "pred_stop": tp + fp}


def summarize_rows(rows: list) -> dict:
    """The TF aggregates of one adapter on one demo set (see the module docstring)."""
    out = {"n_turns": len(rows), "n_demos": len({r["demo"] for r in rows})}
    if not rows:
        return out
    sc = [r for r in rows if r.get("tok_ok") and r.get("pred_type") is not None]
    out["n_type_scored"] = len(sc)
    out["n_tok_mismatch"] = sum(1 for r in rows if not r.get("tok_ok"))
    out["n_unresolved"] = sum(1 for r in sc if not r.get("type_resolved", True))
    out["type_acc"] = rate(sum(1 for r in sc if r["pred_type"] == r["expert_type"]), len(sc))
    out["recall"] = {t: rate(sum(1 for r in sc if r["expert_type"] == t and r["pred_type"] == t),
                             sum(1 for r in sc if r["expert_type"] == t)) for t in TYPES}
    ps = [r for r in sc if r["pred_type"] == "STOP"]
    out["stop_precision"] = rate(sum(1 for r in ps if r["expert_type"] == "STOP"), len(ps))
    ne = [r for r in sc if r.get("next_ok") is not None]
    out["next_exact"] = rate(sum(1 for r in ne if r["next_ok"]), len(ne))
    out["next_rules"] = dict(Counter(r.get("next_rule") for r in ne))
    for node in ("turn", "straight", "stop"):
        nn = [r for r in ne if r["node"] == node]
        out[f"next_exact_{node}"] = rate(sum(1 for r in nn if r["next_ok"]), len(nn))
    pst = [(r["p_type"].get("STOP"), r["expert_type"] == "STOP", r["is_final"]) for r in sc
           if r.get("p_type") and r["p_type"].get("STOP") is not None]
    out["stop_auroc"] = auroc([p for p, s, _ in pst if s], [p for p, s, _ in pst if not s])
    out["p_stop_final_mean"] = mean([p for p, _, f in pst if f])
    out["p_stop_nonfinal_mean"] = mean([p for p, _, f in pst if not f])
    fin = [r for r in sc if r["is_final"]]
    out["greedy_stop_recall_final"] = rate(sum(1 for r in fin if r["pred_type"] == "STOP"), len(fin))
    bins = OrderedDict((b, {"n": 0, "p_stop": [], "expert_stop": 0, "pred_stop": 0}) for b in STOP_BINS)
    navr = [r for r in sc if r["kind"] == "nav" and r.get("stated_m")]
    for r in navr:
        b = bins[stop_bin(r["walked_m"] / r["stated_m"])]
        b["n"] += 1
        b["p_stop"].append(r["p_type"].get("STOP"))
        b["expert_stop"] += r["expert_type"] == "STOP"
        b["pred_stop"] += r["pred_type"] == "STOP"
    out["nav_p_stop_by_walked_over_stated"] = OrderedDict(
        (k, {"n": v["n"], "mean_p_stop": mean(v["p_stop"]), "expert_stop_rate": frac(v["expert_stop"], v["n"]),
             "pred_stop_rate": frac(v["pred_stop"], v["n"])}) for k, v in bins.items())
    mv = [r for r in rows if r["expert_type"] == "MOVE" and r.get("move_pred_id") is not None]
    mvc = [r for r in mv if not r.get("move_id_flag")]
    out["move_id_tf"] = {"all": rate(sum(1 for r in mv if r["move_pred_id"] == r["expert_id"]), len(mv)),
                         "unflagged": rate(sum(1 for r in mvc if r["move_pred_id"] == r["expert_id"]), len(mvc)),
                         "flagged": len(mv) - len(mvc),
                         "expert_id_gt1": rate(sum(1 for r in mvc if r["expert_id"] > 1 and
                                                   r["move_pred_id"] == r["expert_id"]),
                                               sum(1 for r in mvc if r["expert_id"] > 1))}
    rot = [r for r in rows if r["expert_type"] == "ROTATION" and r.get("rot_tf_int_match") is not None]
    out["rotation_tf_int_match"] = rate(sum(1 for r in rot if r["rot_tf_int_match"]), len(rot))
    gr = [r for r in rows if r["expert_type"] == "ROTATION" and r.get("rot_greedy_text") is not None]
    deltas = [r.get("rot_abs_delta") for r in gr]
    out["rotation_greedy"] = {"n": len(gr), "parsed": sum(1 for x in deltas if x is not None),
                              "median_abs_delta": median(deltas),
                              "within_15": rate(sum(1 for x in deltas if x is not None and x <= ROT_OK_DEG), len(gr))}
    by_demo = defaultdict(lambda: [0.0, 0])
    for r in rows:
        if r.get("ce_reply_sum") is not None:
            by_demo[r["demo"]][0] += r["ce_reply_sum"]
            by_demo[r["demo"]][1] += r["n_reply_tokens"]
    dm = [a / b for a, b in by_demo.values() if b]
    out["reply_ce_trainer_weighting"] = round(sum(dm) / len(dm), 6) if dm else None
    tot = sum(b for _, b in by_demo.values())
    out["reply_ce_token_pooled"] = round(sum(a for a, _ in by_demo.values()) / tot, 6) if tot else None
    dt = sum(r.get("n_decision_tokens") or 0 for r in sc)
    out["decision_ce_per_token"] = round(sum(r.get("ce_decision_sum") or 0.0 for r in sc) / dt, 6) if dt else None
    out["type_token_ce_per_turn"] = mean([r.get("ce_type") for r in sc])
    out["marginal_reference"] = marginal_costs(rows)
    nav = [r for r in sc if r["kind"] == "nav" and r.get("stated_m")]
    gold = [r["expert_type"] == "STOP" for r in nav]
    out["nav_stop_detection"] = {
        "model_argmax": _binary([r["pred_type"] == "STOP" for r in nav], gold),
        "rule_walked_ge_stated": _binary([r["walked_m"] >= r["stated_m"] for r in nav], gold)}
    return out


def summarize_set(rows: list, subsets: dict | None = None) -> dict:
    """{all, nav, vis, and named demo-id subsets} -> summarize_rows."""
    out = OrderedDict([("all", summarize_rows(rows)),
                       ("nav", summarize_rows([r for r in rows if r["kind"] == "nav"])),
                       ("vis", summarize_rows([r for r in rows if r["kind"] == "vis"]))])
    for name, ids in (subsets or {}).items():
        out[name] = summarize_rows([r for r in rows if r["demo"] in ids])
    return out


# ------------------------------------------------------------------------------------- TF: torch part
class TFRunner:
    """Teacher-forced scoring of encoded demos with one model (LoRA-wrapped or plain)."""

    def __init__(self, model, processor, device="cuda"):
        from vpd.train import n2_train_vpd as T

        self.T = T
        self.model, self.processor, self.device = model, processor, str(device)
        tok = processor.tokenizer
        self.tok = tok
        self.start_ids = tok.encode(T.REPLY_START, add_special_tokens=False)
        self.end_id = tok.convert_tokens_to_ids(T.END)
        self.digit_ids = []
        for i in range(10):
            ids = tok.encode(str(i), add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"digit {i} is not one token: {ids}")
            self.digit_ids.append(ids[0])
        self.digit_set = set(self.digit_ids)
        self.p_move = tok.encode(REPLY_PREFIX + "MOVE(", add_special_tokens=False)
        self.p_rot = tok.encode(REPLY_PREFIX + "ROTATION(", add_special_tokens=False)
        self.image_token_id = tok.convert_tokens_to_ids(getattr(processor, "image_token", None) or "<|image_pad|>")
        ip = getattr(processor, "image_processor", None)
        self.merge = int(getattr(ip, "merge_size", 2) or 2)
        self._cache = {}

    def ids(self, text: str) -> list:
        if text not in self._cache:
            self._cache[text] = self.tok.encode(text, add_special_tokens=False)
        return self._cache[text]

    def encode(self, rec: dict) -> dict:
        T = self.T
        s = T.samples(rec, 0, False)[0]
        inputs, labels, spans, _ = T.encode(self.processor, s)
        T.check_spans(rec.get("id"), s["messages"], spans)
        return {"inputs": dict(inputs), "labels": labels, "spans": spans}

    def _autocast(self):
        import torch

        dev = "cuda" if self.device.startswith("cuda") else "cpu"
        return torch.autocast(dev, dtype=torch.bfloat16)

    def forward(self, enc: dict):
        import torch

        inputs = {k: v.to(self.device) for k, v in enc["inputs"].items()}
        labels = enc["labels"]
        idx = (labels[0, 1:] != -100).nonzero().squeeze(1)
        with torch.no_grad(), self._autocast():
            out = self.model(**inputs, logits_to_keep=idx.to(self.device), use_cache=False)
        logp = torch.log_softmax(out.logits[0].float(), dim=-1)
        pos = [int(p) + 1 for p in idx.tolist()]
        return logp, {p: r for r, p in enumerate(pos)}, pos, inputs

    def greedy_heading(self, inputs: dict, cut: int, n_img: int):
        """Greedy continuation (<= GREEDY_MAX_NEW tokens) of input_ids[:cut] (ending with "ROTATION(") with the
        first n_img images; (heading or None, text) or (None, None) when the image rows do not line up."""
        import torch

        grid = inputs["image_grid_thw"][:n_img]
        rows = int(grid.prod(dim=1).sum())
        ids = inputs["input_ids"][:, :cut]
        if int((ids == self.image_token_id).sum()) != rows // (self.merge * self.merge):
            return None, None
        kw = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values": inputs["pixel_values"][:rows],
              "image_grid_thw": grid}
        with torch.no_grad(), self._autocast():
            out = self.model.generate(**kw, max_new_tokens=GREEDY_MAX_NEW, do_sample=False, use_cache=True,
                                      pad_token_id=self.tok.pad_token_id if self.tok.pad_token_id is not None
                                      else self.end_id)
        text = self.tok.decode(out[0, cut:].tolist())
        m = re.match(r"\s*(\d+(?:\.\d+)?)", text)
        return (float(m.group(1)) if m else None), text

    def demo_rows(self, enc: dict, rec: dict, turns: list, set_name: str, greedy: bool = False) -> list:
        import torch

        ids = enc["inputs"]["input_ids"][0].tolist()
        spans = enc["spans"]
        if len(spans) != len(turns):
            raise ValueError(f"{rec['id']}: {len(spans)} reply spans for {len(turns)} turns")
        logp, row_of, pos, inputs = self.forward(enc)
        tgt = torch.tensor([ids[p] for p in pos], device=logp.device)
        ce = (-logp.gather(1, tgt[:, None]).squeeze(1)).tolist()
        ce_at = dict(zip(pos, ce))
        img_before = []  # images in messages before each assistant message (one per user turn)
        n_img = 0
        for m in rec["messages"]:
            if m["role"] == "assistant":
                img_before.append(n_img)
            n_img += m["content"].count("<image>")
        out = []
        for t, ((a, b), turn) in enumerate(zip(spans, turns)):
            expert = ids[a:b - 1]
            row = {"demo": rec["id"], "kind": rec.get("task_type"), "set": set_name,
                   **{k: turn[k] for k in ("turn", "expert", "expert_type", "expert_id", "expert_heading", "n_legal",
                                           "walked_m", "stated_m", "dist_to_target_m", "is_final", "node")},
                   "n_turns": len(turns), "n_reply_tokens": b - a,
                   "ce_reply_sum": round(sum(ce_at[p] for p in range(a, b)), 6),
                   "tok_ok": (expert == self.ids(turn.get("reply") or reply_text(turn["expert"]))
                              and ids[b - 1] == self.end_id)}
            row.update(p_type=None, p_groups={}, pred_type=None, type_resolved=None, ce_type=None,
                       ce_decision_sum=None, n_decision_tokens=0, move_pred_id=None, move_id_flag=None,
                       rot_tf_int_match=None, rot_greedy_heading=None, rot_greedy_text=None, rot_abs_delta=None)
            if row["tok_ok"]:
                def lp(j, tok_id, a=a):
                    return float(logp[row_of[a + j], tok_id])

                cands = {k: self.ids(v) for k, v in candidate_texts(turn).items()}
                cl = cand_logprobs(cands, expert, lp)
                d = cl["d"]
                row.update(p_type={k: (None if v is None else round(v, 6)) for k, v in cl["p"].items()},
                           p_groups={k: round(v, 6) for k, v in cl["groups"].items()}, pred_type=cl["pred"],
                           type_resolved=cl["resolved"], ce_type=round(ce_at[a + d], 6))
                dec = [a + d]
                if turn["expert_type"] == "MOVE":
                    j = len(self.p_move)
                    if expert[:j] == self.p_move:
                        n_legal = max(1, turn["n_legal"] or 1)
                        legal = self.digit_ids[1:min(n_legal, 9) + 1]
                        sc = logp[row_of[a + j], legal]
                        row["move_pred_id"] = int(sc.argmax()) + 1
                        row["move_id_flag"] = n_legal >= 10 or turn["expert_id"] >= 10
                        dec += [a + j + q for q in range(len(str(turn["expert_id"])))]
                elif turn["expert_type"] == "ROTATION":
                    j = len(self.p_rot)
                    int_txt = (turn.get("expert_heading_text") or "").split(".")[0].lstrip("-")
                    nd = len(int_txt)
                    if expert[:j] == self.p_rot and nd and all(expert[j + q] in self.digit_set for q in range(nd)):
                        hit = all(int(logp[row_of[a + j + q]].argmax()) == expert[j + q] for q in range(nd))
                        nxt = int(logp[row_of[a + j + nd]].argmax())
                        row["rot_tf_int_match"] = bool(hit and nxt not in self.digit_set)
                        dec += [a + j + q for q in range(nd)]
                        if greedy:
                            h, text = self.greedy_heading(inputs, a + j, img_before[t])
                            row["rot_greedy_heading"], row["rot_greedy_text"] = h, text
                            if h is not None and turn["expert_heading"] is not None:
                                row["rot_abs_delta"] = round(angdiff(h, float(turn["expert_heading"])), 2)
                row["ce_decision_sum"] = round(sum(ce_at[p] for p in dec), 6)
                row["n_decision_tokens"] = len(dec)
            row["next_ok"], row["next_rule"] = next_ok(row, greedy)
            out.append(row)
        del logp
        return out


class LoRABank:
    """The adapters of one TF run on the device: each LoRALinear's (A, B) per adapter, swapped by copy_."""

    def __init__(self, model):
        import lora  # downstream/train (on sys.path through n2_train_vpd)

        self.mods = OrderedDict((n, m) for n, m in model.named_modules() if isinstance(m, lora.LoRALinear))
        if not self.mods:
            raise RuntimeError("no LoRALinear modules on the model")
        self.states = {}
        self.current = None

    def add(self, name: str, adapter_dir, model) -> dict:
        import torch

        from vpd.train import n2_train_vpd as T

        if adapter_dir is None:
            with torch.no_grad():
                for m in self.mods.values():
                    m.lora_A.zero_()
                    m.lora_B.zero_()
            prov = {"dir": None, "sha256": None, "modules": len(self.mods), "raw": True}
        else:
            prov = T.load_init_adapter(model, adapter_dir, RANK, ALPHA)
        self.states[name] = {n: (m.lora_A.detach().clone(), m.lora_B.detach().clone()) for n, m in self.mods.items()}
        self.current = name
        return prov

    def use(self, name: str) -> None:
        import torch

        if self.current == name:
            return
        with torch.no_grad():
            for n, (a, b) in self.states[name].items():
                self.mods[n].lora_A.copy_(a)
                self.mods[n].lora_B.copy_(b)
        self.current = name


def wrap_for_probe(model) -> dict:
    """LoRA modules as in training (r16, alpha 32) with dropout off, float32 masters, eval mode."""
    from vpd.train import n2_train_vpd as T  # noqa: F401  (puts downstream/train on sys.path)
    import lora

    model.config.use_cache = False
    info = lora.wrap(model, RANK, ALPHA, 0.0, LORA_SEED)
    for m in model.modules():
        if isinstance(m, lora.LoRALinear):
            m.lora_A.data, m.lora_B.data = m.lora_A.data.float(), m.lora_B.data.float()
    model.eval()
    return info


class PartWriter:
    """Rows of one (adapter, set) appended per demo to tf_<a>_<s>.jsonl.part, so a cut job resumes at the next demo;
    finish() writes tf_<a>_<s>.jsonl and its .done.json (provenance) and removes the part files (this run's own)."""

    def __init__(self, out: Path, adapter: str, set_name: str, prov: dict):
        self.out, self.adapter, self.set = Path(out), adapter, set_name
        self.base = self.out / f"tf_{adapter}_{set_name}"
        self.part = self.base.with_suffix(".jsonl.part")
        self.part_meta = self.base.with_suffix(".part.json")
        self.prov = prov
        self.done = OrderedDict()
        old = read_json(self.part_meta)
        if old and all(old.get(k) == prov.get(k) for k in ("adapter_sha256", "ids_sha256", "set")) \
                and self.part.exists():
            rows, keep = [], []
            for line in self.part.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    break  # a torn last line from a cut job
                if "demo_end" in r:
                    self.done[r["demo_end"]] = True
                    keep.extend(rows + [r])
                    rows = []
                else:
                    rows.append(r)
            self.part.write_text("".join(json.dumps(r) + "\n" for r in keep))
        else:
            for p in (self.part, self.part_meta):
                if p.exists():
                    p.unlink()  # this probe's own stale partial output
            write_json(self.part_meta, prov)
            self.part.write_text("")
        self._f = open(self.part, "a")

    def append(self, demo: str, rows: list) -> None:
        for r in rows:
            self._f.write(json.dumps(r) + "\n")
        self._f.write(json.dumps({"demo_end": demo}) + "\n")
        self._f.flush()
        self.done[demo] = True

    def finish(self, want_ids: list) -> Path:
        self._f.close()
        missing = [i for i in want_ids if i not in self.done]
        if missing:
            raise RuntimeError(f"{self.base.name}: {len(missing)} demos missing (e.g. {missing[:3]})")
        rows = [json.loads(x) for x in self.part.read_text().splitlines() if x.strip()]
        rows = [r for r in rows if "demo_end" not in r]
        order = {d: i for i, d in enumerate(want_ids)}
        rows.sort(key=lambda r: (order[r["demo"]], r["turn"]))
        write_text(self.base.with_suffix(".jsonl"), "".join(json.dumps(r) + "\n" for r in rows))
        write_json(self.base.with_suffix(".done.json"), dict(self.prov, n_demos=len(want_ids), n_rows=len(rows),
                                                             finished_utc=utc()))
        for p in (self.part, self.part_meta):
            if p.exists():
                p.unlink()
        return self.base.with_suffix(".jsonl")

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()


def set_done(out: Path, adapter: str, set_name: str, prov: dict) -> bool:
    base = Path(out) / f"tf_{adapter}_{set_name}"
    d = read_json(base.with_suffix(".done.json"))
    return bool(d) and base.with_suffix(".jsonl").exists() and all(
        d.get(k) == prov.get(k) for k in ("adapter_sha256", "ids_sha256", "set"))


def parse_adapters(specs: list) -> "OrderedDict[str, Path | None]":
    out = OrderedDict()
    for s in specs or []:
        name, sep, d = s.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z0-9]+", name):
            raise SystemExit(f"--adapter wants NAME=DIR (raw=), got {s!r}")
        if name in out:
            raise SystemExit(f"--adapter {name} given twice")
        if name == "raw":
            if d:
                raise SystemExit("--adapter raw= takes no dir (raw = A = B = 0)")
            out[name] = None
        else:
            if not d:
                raise SystemExit(f"--adapter {name}= needs a dir (only raw has none)")
            p = Path(d)
            if not (p / "adapter.pt").is_file():
                raise SystemExit(f"--adapter {name}: no {p / 'adapter.pt'}")
            out[name] = p
    if not out:
        raise SystemExit("give at least one --adapter")
    return out


def code_state() -> dict:
    files = {}
    for rel in ("vpd/walkerbench/learn483.py", "vpd/train/n2_train_vpd.py", "downstream/train/lora.py"):
        p = paths.WORKTREE / rel
        files[rel] = sha256_file(p) if p.is_file() else None
    return files


def iterate_encoded(runner: TFRunner, recs: list, order: list, num_workers: int):
    """(index, encoded demo) for recs[i], i in order: inline, or a prefetching DataLoader (workers 1 thread each)."""
    if num_workers <= 0:
        for i in order:
            yield i, runner.encode(recs[i])
        return
    import torch
    from torch.utils.data import DataLoader

    from vpd.train import n2_train_vpd as T

    ds = _EncodeData(runner.processor, recs)
    dl = DataLoader(ds, batch_size=None, sampler=list(order), num_workers=num_workers, prefetch_factor=2,
                    worker_init_fn=T._worker_init, collate_fn=T._identity, persistent_workers=False,
                    generator=torch.Generator().manual_seed(0))
    for item in dl:
        yield item["i"], item["enc"]


class _EncodeData:
    def __init__(self, processor, recs):
        self.processor, self.recs = processor, recs

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        from vpd.train import n2_train_vpd as T

        rec = self.recs[i]
        s = T.samples(rec, 0, False)[0]
        inputs, labels, spans, _ = T.encode(self.processor, s)
        T.check_spans(rec.get("id"), s["messages"], spans)
        return {"i": i, "enc": {"inputs": dict(inputs), "labels": labels, "spans": spans}}


def turns_for(recs: list, tasks_dir=paths.WB_S2TRAIN_TASKS, db=CACHE_DB) -> dict:
    tasks = {r["id"]: demo_task(r, tasks_dir) for r in recs}
    pano = set()
    for t in tasks.values():
        if t:
            pano |= task_panos(t)
    try:
        coords = load_coords(pano, db)
    except Exception as e:  # noqa: BLE001 - distances are descriptive; the probe runs without them
        print(f"[learn483] coordinates unavailable ({type(e).__name__}: {e}); dist_to_target_m = null", flush=True)
        coords = {}
    return {r["id"]: turn_table(r, tasks[r["id"]], coords) for r in recs}


def write_adapter_summary(out: Path, name: str, adapter_dir, sets: dict, prov: dict, tiers: Path = TIERS) -> dict:
    """tf_summary_<name>.json from the finished set files (incomplete sets listed)."""
    res = OrderedDict()
    done, missing = [], []
    subsets = {}
    if "TR200" in sets:
        t50 = Path(tiers) / "demos_50.jsonl"
        ids50 = {r["id"] for r in read_jsonl(t50)} if t50.exists() else {r["id"] for r in sets["TR200"][:50]}
        subsets = {"TR200": {"first50": ids50, "rest150": {r["id"] for r in sets["TR200"]} - ids50}}
    member = adapter_ids(adapter_dir)
    for s, recs in sets.items():
        f = Path(out) / f"tf_{name}_{s}.jsonl"
        if not set_done(out, name, s, dict(prov, set=s, ids_sha256=ids_sha(recs))):
            missing.append(s)
            continue
        done.append(s)
        res[s] = summarize_set(read_jsonl(f), subsets.get(s))
        res[s]["in_sample_demos"] = sum(1 for r in recs if r["id"] in member)
        res[s]["n_demos_set"] = len(recs)
    doc = {"schema": SCHEMA + ":tf_summary", "adapter": name, "adapter_dir": None if adapter_dir is None
           else str(adapter_dir), "adapter_sha256": prov.get("adapter_sha256"), "complete_sets": done,
           "missing_sets": missing, "sets": res, "utc": utc(), "exploratory": True,
           "note": "teacher-forced probe; numbers only, no decision rule"}
    write_json(Path(out) / f"tf_summary_{name}.json", doc)
    return doc


def merge_summaries(out: Path) -> Path:
    docs = OrderedDict()
    for a in ADAPTERS:
        d = read_json(Path(out) / f"tf_summary_{a}.json")
        if d:
            docs[a] = d
    for p in sorted(Path(out).glob("tf_summary_*.json")):
        a = p.stem[len("tf_summary_"):]
        if a not in docs:
            docs[a] = read_json(p)
    return write_json(Path(out) / "tf_summary.json", {"schema": SCHEMA + ":tf_summary_all", "utc": utc(),
                                                      "exploratory": True, "adapters": docs})


def load_probe_model(args):
    import torch

    from vpd import starts

    torch.set_num_threads(max(1, int(args.threads)))
    model, processor = starts.load_model(args.start, device=args.device, attn=args.attn, path=args.model_path)
    return model, processor


def cmd_tf(args) -> int:
    out = Path(args.out)
    refuse_pilot(out)
    adapters = parse_adapters(args.adapter)
    sets = demo_sets(Path(args.demos_all), Path(args.tiers))
    names = [s.strip() for s in args.sets.split(",") if s.strip()]
    for s in names:
        if s not in sets:
            raise SystemExit(f"unknown set {s!r}; one of {list(sets)}")
    sets = OrderedDict((s, sets[s][:args.limit] if args.limit else sets[s]) for s in names)
    if not args.limit:
        for s, recs in sets.items():
            if len(recs) != SET_SIZES[s]:
                raise SystemExit(f"{s}: {len(recs)} demos, expected {SET_SIZES[s]}")
    greedy_sets = {s.strip() for s in (args.greedy_sets or "").split(",") if s.strip()}
    out.mkdir(parents=True, exist_ok=True)
    shas = OrderedDict((n, None if d is None else sha256_file(d / "adapter.pt")) for n, d in adapters.items())
    base_prov = {n: {"schema": SCHEMA + ":tf", "adapter": n, "adapter_dir": None if d is None else str(d),
                     "adapter_sha256": shas[n], "start": args.start, "code": code_state(),
                     "greedy": None} for n, d in adapters.items()}

    def prov(n, s):
        return dict(base_prov[n], set=s, ids_sha256=ids_sha(sets[s]), n_set=len(sets[s]), greedy=s in greedy_sets)

    todo = OrderedDict((s, [n for n in adapters if not set_done(out, n, s, prov(n, s))]) for s in sets)
    print(f"[learn483] tf: adapters {list(adapters)} sets {list(sets)}; to do "
          f"{ {s: v for s, v in todo.items() if v} }", flush=True)
    if any(todo.values()):
        t0 = time.time()
        model, processor = load_probe_model(args)
        info = wrap_for_probe(model)
        bank = LoRABank(model)
        provs = {}
        need = [n for n in adapters if any(n in v for v in todo.values())]
        for n in need:
            provs[n] = bank.add(n, adapters[n], model)
            print(f"[learn483] adapter {n}: {provs[n].get('modules')} modules, sha256 "
                  f"{(provs[n].get('sha256') or 'raw')[:12]}", flush=True)
        runner = TFRunner(model, processor, args.device)
        print(f"[learn483] model ready in {time.time() - t0:.0f} s; lora {info['wrapped']} modules", flush=True)
        for s, todo_n in todo.items():
            if not todo_n:
                continue
            recs = sets[s]
            turns = turns_for(recs)
            writers = {n: PartWriter(out, n, s, prov(n, s)) for n in todo_n}
            order = [i for i, r in enumerate(recs) if any(r["id"] not in w.done for w in writers.values())]
            t1 = time.time()
            try:
                for k, (i, enc) in enumerate(iterate_encoded(runner, recs, order, args.num_workers)):
                    rec = recs[i]
                    for n in todo_n:
                        w = writers[n]
                        if rec["id"] in w.done:
                            continue
                        bank.use(n)
                        w.append(rec["id"], runner.demo_rows(enc, rec, turns[rec["id"]], s, s in greedy_sets))
                    if (k + 1) % 20 == 0 or k + 1 == len(order):
                        print(f"[learn483] {s}: {k + 1}/{len(order)} demos ({time.time() - t1:.0f} s)", flush=True)
            except BaseException:
                for w in writers.values():
                    w.close()  # the part files stay: a resubmitted job resumes at the next demo
                raise
            for n in todo_n:
                writers[n].finish([r["id"] for r in recs])
                print(f"[learn483] wrote {out / f'tf_{n}_{s}.jsonl'}", flush=True)
    for n, d in adapters.items():
        doc = write_adapter_summary(out, n, d, sets, dict(base_prov[n]), Path(args.tiers))
        a = doc["sets"].get("VAL56", {}).get("all", {})
        print(f"[learn483] summary {n}: complete {doc['complete_sets']}; VAL56 type_acc "
              f"{(a.get('type_acc') or {}).get('rate')} stop_auroc {a.get('stop_auroc')} "
              f"reply_ce {a.get('reply_ce_trainer_weighting')}", flush=True)
    merge_summaries(out)
    return 0


def demo_ce(rows: list) -> float | None:
    s = sum(r["ce_reply_sum"] for r in rows)
    n = sum(r["n_reply_tokens"] for r in rows)
    return s / n if n else None


def compare_rows(a: dict, b: dict) -> dict:
    """Merge parity numbers between two {demo: rows} maps (LoRA vs merged)."""
    per = []
    dp, agree, nturn = [], 0, 0
    for demo in a:
        ra, rb = a[demo], b[demo]
        ca, cb = demo_ce(ra), demo_ce(rb)
        per.append({"demo": demo, "ce_lora": round(ca, 6), "ce_merged": round(cb, 6), "abs_diff": round(abs(ca - cb), 6)})
        for x, y in zip(ra, rb):
            nturn += 1
            agree += x.get("pred_type") == y.get("pred_type")
            for t in TYPES:
                px, py = (x.get("p_type") or {}).get(t), (y.get("p_type") or {}).get(t)
                if px is not None and py is not None:
                    dp.append(abs(px - py))
    diffs = [p["abs_diff"] for p in per]
    return {"per_demo": per, "n": len(per), "mean_abs_diff_ce": mean(diffs), "max_abs_diff_ce": max(diffs) if diffs else None,
            "mean_ce_lora": mean([p["ce_lora"] for p in per]), "mean_ce_merged": mean([p["ce_merged"] for p in per]),
            "pred_type_agree": rate(agree, nturn), "max_abs_diff_p_type": round(max(dp), 6) if dp else None}


def cmd_tf_parity(args) -> int:
    import gc

    import torch

    from vpd import starts

    out = Path(args.out)
    refuse_pilot(out)
    ad = Path(args.adapter)
    recs = demo_sets(Path(args.demos_all), Path(args.tiers))["VAL56"][:args.n]
    turns = turns_for(recs)
    torch.set_num_threads(max(1, int(args.threads)))
    t0 = time.time()
    model, processor = starts.load_model(args.start, device=args.device, attn=args.attn, path=args.model_path)
    wrap_for_probe(model)
    from vpd.train import n2_train_vpd as T

    prov = T.load_init_adapter(model, ad, RANK, ALPHA)
    runner = TFRunner(model, processor, args.device)
    lora_rows = OrderedDict()
    for i, enc in iterate_encoded(runner, recs, list(range(len(recs))), args.num_workers):
        lora_rows[recs[i]["id"]] = runner.demo_rows(enc, recs[i], turns[recs[i]["id"]], "VAL56", False)
    del runner, model
    gc.collect()
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    merged, _ = starts.load_model(args.start, device=args.device, attn=args.attn, path=args.merged)
    merged.config.use_cache = False
    merged.eval()
    runner = TFRunner(merged, processor, args.device)  # the base processor: identical inputs
    merged_rows = OrderedDict()
    for i, enc in iterate_encoded(runner, recs, list(range(len(recs))), args.num_workers):
        merged_rows[recs[i]["id"]] = runner.demo_rows(enc, recs[i], turns[recs[i]["id"]], "VAL56", False)
    cmp = compare_rows(lora_rows, merged_rows)
    doc = {"schema": SCHEMA + ":merge_parity", "utc": utc(), "exploratory": True, "adapter": str(ad),
           "adapter_sha256": prov["sha256"], "merged": str(args.merged),
           "merged_vpd_merge_json": read_json(Path(args.merged) / "vpd_merge.json"), "set": "VAL56",
           "demos": [r["id"] for r in recs], "seconds": round(time.time() - t0, 1), "code": code_state(),
           "note": "TF reply CE (trainer weighting per demo), merged bf16 folder vs unmerged LoRA; numbers only", **cmp}
    write_json(out, doc)
    print(f"[learn483] merge parity: mean |dCE| {cmp['mean_abs_diff_ce']} max {cmp['max_abs_diff_ce']} "
          f"pred_type agree {cmp['pred_type_agree']['rate']} -> {out}", flush=True)
    return 0


def cmd_parity_ok(args) -> int:
    doc = read_json(args.file)
    pt = Path(args.adapter) / "adapter.pt"
    if not doc or not pt.is_file():
        print(f"[learn483] parity-ok: {args.file} or {pt} missing")
        return 1
    if doc.get("adapter_sha256") != sha256_file(pt) or doc.get("n") != PARITY_N:
        print(f"[learn483] parity-ok: {args.file} is for another adapter / n; re-run")
        return 1
    print(f"[learn483] parity-ok: {args.file} fresh")
    return 0


# ------------------------------------------------------------------------------------- task set s2in50
def insample_plan(tier50=TIERS / "demos_50.jsonl", split_dir=paths.WB_SPLIT) -> dict:
    split_dir = Path(split_dir)
    doc = json.loads((split_dir / "split_s2.json").read_text())
    ids = [json.loads(x)["id"] for x in Path(tier50).read_text().splitlines() if x.strip()]
    rest = set(doc["splits"]["train_rest"])
    bad = [i for i in ids if i not in rest]
    if bad or len(set(ids)) != len(ids):
        raise SystemExit(f"{tier50}: ids not in train_rest or repeated: {bad[:5]}")
    files = OrderedDict((doc["tasks"][i]["file"], doc["tasks"][i]["sha256"]) for i in ids)
    kinds = Counter(doc["tasks"][i]["kind"] for i in ids)
    return {"split": "s2in50", "schema": "vpd.walkerbench.s2in50/1", "exploratory": True,
            "purpose": "EXPLORATORY in-sample closed-loop set (learn483): the 50 tasks of tiers/seed1001/demos_50.jsonl, "
                       "trained on by the 50 / 200 / 483 adapters, unseen by raw and 0W; not part of the pilot",
            "source_tier": str(tier50), "source_tier_sha256": sha256_file(tier50),
            "split_s2_sha256": sha256_file(split_dir / "split_s2.json"), "ids": ids, "files": files,
            "counts": {"nav": kinds.get("nav", 0), "vis": kinds.get("vis", 0), "total": len(ids)}}


def insample_problems(dest: Path, plan: dict) -> list:
    dest = Path(dest)
    probs = []
    sel = read_json(dest / "selection.json")
    if not sel or sel.get("ids") != plan["ids"] or sel.get("files") != plan["files"]:
        probs.append(f"{dest}/selection.json differs from the demos_50 plan")
    have = {p.name: p for p in (dest / "tasks").glob("*.json")}
    if set(have) != set(plan["files"]):
        probs.append(f"{dest}/tasks: files differ (missing {sorted(set(plan['files']) - set(have))[:3]}, "
                     f"extra {sorted(set(have) - set(plan['files']))[:3]})")
    for f in sorted(set(have) & set(plan["files"])):
        if sha256_file(have[f]) != plan["files"][f]:
            probs.append(f"{dest}/tasks/{f}: sha256 differs")
    return probs


def build_insample(dest=S2IN50_DIR, tier50=TIERS / "demos_50.jsonl", split_dir=paths.WB_SPLIT) -> dict:
    """Create dest/{tasks/, selection.json} (atomic rename of a sibling temp dir), or verify an existing one."""
    dest, split_dir = Path(dest), Path(split_dir)
    plan = insample_plan(tier50, split_dir)
    if dest.exists():
        probs = insample_problems(dest, plan)
        if probs:
            raise SystemExit("[learn483] s2in50 exists but differs (nothing overwritten): " + "; ".join(probs))
        return {"state": "verified", "dir": str(dest), **plan["counts"]}
    tmp = dest.parent / f".{dest.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    (tmp / "tasks").mkdir(parents=True)
    try:
        src = split_dir / "s2train" / "tasks"
        for f, sha in plan["files"].items():
            data = (src / f).read_bytes()
            if sha256_bytes(data) != sha:
                raise SystemExit(f"{src / f}: sha256 differs from split_s2.json")
            (tmp / "tasks" / f).write_bytes(data)
        write_json(tmp / "selection.json", dict(plan, created_utc=utc()))
        try:
            os.rename(tmp, dest)
            state = "created"
        except OSError:
            state = "raced"
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)  # this call's own temp dir
    probs = insample_problems(dest, plan)
    if probs:
        raise SystemExit("[learn483] s2in50 check failed: " + "; ".join(probs))
    return {"state": state, "dir": str(dest), **plan["counts"]}


def register_task_sets(run_vpd=None):
    """Add 's2in50' to run_vpd.TASK_SETS in this process only (run_vpd.py is not edited)."""
    if run_vpd is None:
        from vpd.walkerbench import run_vpd
    cur = run_vpd.TASK_SETS.get("s2in50")
    if cur is None:
        run_vpd.TASK_SETS["s2in50"] = S2IN50_TASKS
    elif Path(cur) != S2IN50_TASKS:
        raise SystemExit(f"run_vpd.TASK_SETS['s2in50'] is {cur}, expected {S2IN50_TASKS}")
    return run_vpd


def cmd_eval(argv: list) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--run-name")
    ap.add_argument("--out-root", type=Path, default=paths.WB_RUNS)
    a, _ = ap.parse_known_args(argv)
    if a.run_dir is None and not a.run_name:
        raise SystemExit("learn483 eval: give --run-name (and --out-root) or --run-dir")
    run_dir = (a.run_dir or a.out_root / a.run_name).resolve()
    refuse_pilot(run_dir)
    R = register_task_sets()
    rc = 1
    try:
        R.main(argv)
        rc = 0
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if rc and not isinstance(e.code, int):
            print(e.code, file=sys.stderr)
    finally:
        if run_dir.is_dir():
            write_json(run_dir / "explore.json", {"schema": SCHEMA + ":eval", "exploratory": True, "utc": utc(),
                                                  "wrapper": "python -m vpd.walkerbench.learn483 eval",
                                                  "argv": argv, "rc": rc,
                                                  "task_sets_added": {"s2in50": str(S2IN50_TASKS)},
                                                  "code": code_state()})
    return rc


# ------------------------------------------------------------------------------------- floors
def run_floors(task_set: str, out: Path, logs_root: Path, work: Path, stems: list | None = None,
               policies=("STOP0", "FWD", "ODO", "RM"), tasks_dir: Path | None = None) -> dict:
    """STOP0 / FWD / ODO / RM on one task set through baselines.load_walkerbench / run_policy (rendering stubbed).
    One sandbox per process (baselines' rule): call once per set."""
    from vpd.walkerbench import baselines as B
    from vpd.walkerbench import sandbox

    out, logs_root, work = Path(out), Path(logs_root), Path(work)
    refuse_pilot(out)
    if is_under(out, paths.WB_BASELINES):
        raise SystemExit("floors of the exploratory run never go to the frozen baselines dir")
    tdir = Path(tasks_dir or TASK_DIRS[task_set])
    all_stems = sorted(p.stem for p in tdir.glob("*.json"))
    stems = list(stems) if stems else all_stems
    if set(stems) - set(all_stems):
        raise SystemExit(f"not tasks of {tdir}: {sorted(set(stems) - set(all_stems))[:5]}")
    stamp = B.utc_stamp()
    wdir = work / f"floors_{task_set}_{stamp}_{os.getpid()}"
    copy, diff = sandbox.make_sandbox(wdir, tdir, render_size=(640, 400), exclude=sandbox.LEAN_EXCLUDE)
    cwd = os.getcwd()
    t0 = time.time()
    res = OrderedDict()
    try:
        wb = B.load_walkerbench(copy, sandbox.task_label(tdir))
        for m in policies:
            t1 = time.time()
            r = B.run_policy(wb, m, stems, logs_root / stamp / m)
            r["summary"] = B.summarise(r["per_task"], stems, set())
            r["seconds"] = round(time.time() - t1, 1)
            res[m] = r
            s = r["summary"]
            print(f"[learn483] floors {task_set} {m:5s} nav {s['nav']['k']}/{s['nav']['n']} vis {s['vis']['k']}/"
                  f"{s['vis']['n']} no log {s['tasks_without_log']}", flush=True)
        if wb.norender.calls == 0 or any((copy / "temp_images").rglob("*.jpg")):
            raise RuntimeError("rendering stub was not in effect")
        render_calls = wb.norender.calls
    finally:
        os.chdir(cwd)
        shutil.rmtree(wdir, ignore_errors=True)  # this run's own sandbox copy
    nav = [t for t in stems if t.startswith("nav")]
    vis = [t for t in stems if t.startswith("vis")]

    def k(m, ts):
        return sum(1 for t in ts if (res.get(m, {}).get("per_task", {}).get(t) or {}).get("success"))

    floors = {}
    if "ODO" in res and "FWD" in res:
        floors = {"nav_ODO": rate(k("ODO", nav), len(nav)), "vis_FWD": rate(k("FWD", vis), len(vis)),
                  "pooled_F": rate(k("ODO", nav) + k("FWD", vis), len(stems))}
    doc = {"schema": SCHEMA + ":floors", "exploratory": True, "created_utc": stamp, "task_set": task_set,
           "tasks_dir": str(tdir), "n_tasks": len(stems), "stems": stems, "subset": stems != all_stems,
           "rendering": f"stubbed ({render_calls} calls)", "logs": str(logs_root / stamp),
           "seconds": round(time.time() - t0, 1), "sandbox_settings_diff": diff, "floors": floors,
           "policies": res, "code": {"learn483": sha256_file(__file__),
                                     "baselines": sha256_file(B.__file__), "sandbox": sha256_file(sandbox.__file__)},
           "note": "EXPLORATORY floors (learn483); s2test floors are baselines/baselines_s2test.json"}
    write_json(out, doc)
    return doc


# ------------------------------------------------------------------------------------- analysis
def parse_log(path: Path):
    ev = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    ev.append(json.loads(line))
                except ValueError:
                    pass
    st = next((e for e in ev if e.get("event") == "session_start"), None)
    return st, [e for e in ev if e.get("event") == "action"]


def load_tasks(tasks_dir: Path) -> dict:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(tasks_dir).glob("*.json"))}


def policy_logs(logs_dir: Path) -> dict:
    out = {}
    if logs_dir and Path(logs_dir).is_dir():
        for f in sorted(Path(logs_dir).glob("*.jsonl")):
            st, acts = parse_log(f)
            if st:
                out[st["task_id"]] = (st, acts)
    return out


def expert_nodes(tasks: dict, rm_logs: dict) -> dict:
    """{stem: {"rot": path indices where RM rotated, "k": indices where RM's MOVE id was > 1}}."""
    out = {}
    for stem, task in tasks.items():
        vp = [p["pano_id"] for p in task.get("visual_path") or []]
        last = {p: i for i, p in enumerate(vp)}
        rot, kk = set(), set()
        for a in rm_logs.get(stem, (None, []))[1]:
            pid = (a.get("state") or {}).get("pano_id")
            if pid not in last:
                continue
            act = a.get("action") or {}
            if act.get("type") == "rotation":
                rot.add(last[pid])
            elif act.get("type") == "move" and (act.get("move_id") or 1) > 1:
                kk.add(last[pid])
        out[stem] = {"rot": rot, "k": kk}
    return out


def episode(stem: str, task: dict, st: dict, acts: list, coords: dict, nodes: dict | None = None,
            row: dict | None = None) -> dict:
    """Behaviour of one logged episode (definitions of diag.py / diag3.py / diag4.py)."""
    vp = [p["pano_id"] for p in task.get("visual_path") or []]
    last = {p: i for i, p in enumerate(vp)}
    targets = [t for t in task.get("target_pano_ids") or [] if t in coords]
    kinds = [(a.get("action") or {}).get("type") for a in acts]
    states = [a.get("state") or {} for a in acts]
    if not states:
        states = [(st or {}).get("initial_state") or {}]
    lls = [(s.get("lat"), s.get("lng")) if s.get("lat") is not None else None for s in states]
    panos = [s.get("pano_id") for s in states]

    def dist(ll):
        return min(hav(ll, coords[t]) for t in targets) if targets and ll is not None else None

    dists = [dist(ll) for ll in lls]
    walked = [0.0]
    for i in range(1, len(lls)):
        step = hav(lls[i - 1], lls[i]) if lls[i - 1] is not None and lls[i] is not None else 0.0
        walked.append(walked[-1] + step)
    n = len(acts)
    cap = int(round(len(vp) * 1.5)) if vp else int(task.get("max_steps") or 30)
    ended = "stop" if kinds and kinds[-1] == "stop" else ("cap" if n >= cap else "other")
    sp = task.get("spawn_point")
    d0 = dist(coords[sp]) if sp in coords else None
    fwd = sum(1 for a in acts if (a.get("action") or {}).get("type") == "move" and (a.get("available_moves") or [])
              and a["available_moves"][0].get("id") == a["action"].get("move_id"))
    nd = nodes or {"rot": set(), "k": set()}
    node_dec = []  # (idx, at an RM rotation node, at an RM rotation-or-MOVE(k>1) node, ok)
    k = 0
    while k < n:
        p = panos[k]
        if p in last and last[p] < len(vp) - 1:
            idx = last[p]
            j = k
            while j < n and kinds[j] == "rotation":
                j += 1
            if j < n and kinds[j] == "move" and j + 1 < n:
                node_dec.append((idx, idx in nd["rot"], idx in nd["rot"] or idx in nd["k"], panos[j + 1] == vp[idx + 1]))
            elif j < n and kinds[j] == "stop":
                node_dec.append((idx, idx in nd["rot"], idx in nd["rot"] or idx in nd["k"], False))
            k = j + 1
        else:
            k += 1
    first = next((i for i, p in enumerate(panos) if vp and p == vp[-1]), None) if acts else None
    at_target = None
    if first is not None:
        at_target = {"move": "M", "rotation": "R", "stop": "S"}.get(kinds[first], "?") if first < n else "END"
    opt = float(((task.get("ground_truth") or {}).get("optimal_distance_meters")) or 0)
    kk = next((i for i, w in enumerate(walked) if w >= opt), len(walked) - 1)
    cf = dists[kk] is not None and dists[kk] <= THR_M
    stop_ratio = walked[-1] / opt if (ended == "stop" and n > 1 and opt) else None
    e = {"stem": stem, "kind": stem.split("_", 1)[0], "n": n, "cap": cap, "ended": ended, "d0": d0,
         "trivial": d0 is not None and d0 <= THR_M, "fwd_agree": fwd, "node_dec": node_dec, "at_target": at_target,
         "near_steps": [(dists[i] is not None and dists[i] <= THR_M, kinds[i] == "stop") for i in range(n)],
         "cf_success": int(cf), "walked": walked[-1], "opt": opt, "stop_ratio": stop_ratio,
         "seq": [(kinds[i], (acts[i].get("action") or {}).get("move_id"), (acts[i].get("action") or {}).get("heading"))
                 for i in range(n)]}
    if row is not None:
        e.update(success=int(row.get("success") or 0), oracle=row.get("oracle"), steps=row.get("steps"))
    return e


def summarize_eps(eps: list) -> dict:
    n = len(eps)
    out = {"n": n}
    if not n:
        return out
    out["success"] = rate(sum(e.get("success", 0) for e in eps), n)
    out["oracle"] = rate(sum(1 for e in eps if e.get("oracle") == 1), n)
    out["mean_steps"] = round(sum(e["n"] for e in eps) / n, 2)
    out["ended_stop"] = rate(sum(1 for e in eps if e["ended"] == "stop"), n)
    out["cap_hits"] = sum(1 for e in eps if e["ended"] == "cap")
    nt = [e for e in eps if not e["trivial"]]
    out["nontrivial_success"] = rate(sum(e.get("success", 0) for e in nt), len(nt))
    near = [s for e in nt for (isn, s) in e["near_steps"] if isn]
    far = [s for e in nt for (isn, s) in e["near_steps"] if not isn]
    out["p_stop_within15"] = rate(sum(near), len(near))
    out["p_stop_beyond15"] = rate(sum(far), len(far))
    out["first_arrival_target_pano_next"] = dict(Counter(e["at_target"] for e in nt if e["at_target"] is not None))
    dec = [d for e in eps for d in e["node_dec"]]
    out["node_turn_rot"] = rate(sum(1 for d in dec if d[1] and d[3]), sum(1 for d in dec if d[1]))
    out["node_straight_rot"] = rate(sum(1 for d in dec if not d[1] and d[3]), sum(1 for d in dec if not d[1]))
    out["node_turn_any"] = rate(sum(1 for d in dec if d[2] and d[3]), sum(1 for d in dec if d[2]))
    out["node_straight_any"] = rate(sum(1 for d in dec if not d[2] and d[3]), sum(1 for d in dec if not d[2]))
    steps = sum(e["n"] for e in eps)
    out["fwd_step_agreement"] = rate(sum(e["fwd_agree"] for e in eps), steps)
    out["cf_own_path_stop_at_walked_ge_stated"] = rate(sum(e["cf_success"] for e in eps), n)
    sr = [e["stop_ratio"] for e in eps if e["stop_ratio"] is not None]
    out["stop_walked_over_stated_within20"] = rate(sum(1 for x in sr if abs(x - 1) <= 0.2), len(sr))
    out["stop_walked_over_stated_median"] = median(sr)
    return out


def summarize_eval(eps: list) -> dict:
    return OrderedDict([("nav", summarize_eps([e for e in eps if e["kind"] == "nav"])),
                        ("vis", summarize_eps([e for e in eps if e["kind"] == "vis"])),
                        ("all", summarize_eps(eps))])


def eval_episodes(run_dir: Path, tasks: dict, coords: dict, nodes: dict) -> list:
    rows = read_jsonl(Path(run_dir) / "per_task.jsonl")
    logs = Path(run_dir) / "WalkerBench" / "logs"
    eps = []
    for r in rows:
        stem = r["stem"]
        if stem not in tasks:
            continue
        st, acts = parse_log(logs / r["log"]) if r.get("log") and (logs / r["log"]).exists() else (None, [])
        eps.append(episode(stem, tasks[stem], st, acts, coords, nodes.get(stem), r))
    return eps


def floor_episodes(per_task: dict, logs: dict, tasks: dict, coords: dict, nodes: dict) -> list:
    eps = []
    for stem, task in tasks.items():
        if stem in logs:
            st, acts = logs[stem]
            r = per_task.get(stem) or {}
            eps.append(episode(stem, task, st, acts, coords, nodes.get(stem),
                               {"success": r.get("success"), "oracle": None, "steps": r.get("steps")}))
    return eps


def pair_runs(a: list, b: list) -> dict:
    ea, eb = {e["stem"]: e for e in a}, {e["stem"]: e for e in b}
    common = sorted(set(ea) & set(eb))
    same = sum(1 for s in common if ea[s]["seq"] == eb[s]["seq"])
    return {"tasks": len(common), "identical_action_seq": same,
            "success_both": sum(1 for s in common if ea[s].get("success") and eb[s].get("success")),
            "success_only_first": sum(1 for s in common if ea[s].get("success") and not eb[s].get("success")),
            "success_only_second": sum(1 for s in common if eb[s].get("success") and not ea[s].get("success"))}


def eval_table(x: Path, pilot: Path) -> "OrderedDict[str, OrderedDict]":
    """{task set: {row label: eval dir}} (pilot rows on s2test, read only)."""
    pe = Path(pilot) / "evals"
    xe = Path(x) / "evals"
    t = OrderedDict()
    t["s2test"] = OrderedDict([
        ("raw", pe / "pilot__Q3B__raw__640__empty"), ("0W", pe / "pilot__Q3B__0W__640__empty"),
        ("50", pe / "pilot__Q3B__50__640__empty"), ("200", pe / "pilot__Q3B__200__640__empty"),
        ("483A", xe / eval_name("483A", "s2test", "640", "empty")),
        ("483B", xe / eval_name("483B", "s2test", "640", "empty")),
        ("483A blind", xe / eval_name("483A", "s2test", "640", "empty", "blind")),
        ("483A rep2 (workers 4)", xe / eval_name("483A", "s2test", "640", "empty", "rep2")),
        ("raw off 640 (max_tokens 1024)", xe / eval_name("raw", "s2test", "640", "off")),
        ("raw off 1280 (max_tokens 2048)", xe / eval_name("raw", "s2test", "1280", "off")),
        ("raw 1280 empty (pilot)", pe / "pilot__Q3B__raw__1280__empty"),
        ("200 off (pilot)", pe / "pilot__Q3B__200__640__off")])
    t["s2val"] = OrderedDict([("raw", xe / eval_name("raw", "s2val", "640", "empty")),
                              ("200", xe / eval_name("200", "s2val", "640", "empty")),
                              ("483A", xe / eval_name("483A", "s2val", "640", "empty")),
                              ("483B", xe / eval_name("483B", "s2val", "640", "empty"))])
    t["s2in50"] = OrderedDict([("raw", xe / eval_name("raw", "s2in50", "640", "empty")),
                               ("50", xe / eval_name("50", "s2in50", "640", "empty")),
                               ("200", xe / eval_name("200", "s2in50", "640", "empty")),
                               ("483A", xe / eval_name("483A", "s2in50", "640", "empty")),
                               ("483B", xe / eval_name("483B", "s2in50", "640", "empty"))])
    return t


def floor_sources(x: Path) -> dict:
    """{set: (per-policy per_task, logs dir) } for s2test (frozen baselines) and the learn483 floors."""
    out = {}
    b = read_json(BASELINES_S2TEST)
    if b:
        out["s2test"] = ({m: v.get("per_task", {}) for m, v in b.get("policies", {}).items()}, Path(b.get("logs", "")),
                         str(BASELINES_S2TEST))
    for s in ("s2val", "s2in50"):
        f = Path(x) / f"floors_{s}.json"
        d = read_json(f)
        if d:
            out[s] = ({m: v.get("per_task", {}) for m, v in d.get("policies", {}).items()}, Path(d.get("logs", "")),
                      str(f))
    return out


def adapter_losses(name: str, d: Path) -> dict | None:
    meta = read_json(Path(d) / "meta.json")
    if not meta:
        return None
    ep = defaultdict(list)
    log = Path(d) / "log.jsonl"
    if log.exists():
        for r in read_jsonl(log):
            ep[r.get("epoch")].append(r.get("loss"))
    return {"adapter": name, "dir": str(d), "data": meta.get("data"), "conversations": meta.get("conversations"),
            "epochs": meta.get("epochs"), "optimizer_steps": meta.get("optimizer_steps"),
            "micro_steps": meta.get("micro_steps"), "loss_first": meta.get("loss_first"),
            "loss_last": meta.get("loss_last"), "loss_last10_mean": meta.get("loss_last10_mean"),
            "epoch_mean_loss": {str(k): mean(v) for k, v in sorted(ep.items(), key=lambda kv: kv[0] or 0)},
            "train_seconds": meta.get("train_seconds"), "code_sha256": meta.get("code_sha256"),
            "init_sha256": (meta.get("init_adapter") or {}).get("sha256"),
            "note": "online loss (dropout on) per reply token, trainer weighting"}


def analyze(x: Path = X_ROOT, pilot: Path = PILOT_Q3B, db=CACHE_DB) -> dict:
    x, pilot = Path(x), Path(pilot)
    refuse_pilot(x)
    table = eval_table(x, pilot)
    fsrc = floor_sources(x)
    doc = OrderedDict([("schema", SCHEMA + ":diag"), ("utc", utc()), ("exploratory", True),
                       ("note", "numbers only; no verdict, no gate (user request 2026-11-09)"),
                       ("references", REFERENCES), ("sets", OrderedDict())])
    eps_keep = {}
    for s, rows in table.items():
        tdir = TASK_DIRS[s]
        if not tdir.is_dir():
            doc["sets"][s] = {"missing": f"no task dir {tdir}"}
            continue
        tasks = load_tasks(tdir)
        pano = set()
        for t in tasks.values():
            pano |= task_panos(t)
        coords = load_coords(pano, db)
        per_policy, logs_dir, fsrc_path = fsrc.get(s, ({}, None, None))
        rm_logs = policy_logs(logs_dir / "RM") if logs_dir else {}
        nodes = expert_nodes(tasks, rm_logs)
        sd = OrderedDict([("n_tasks", len(tasks)), ("kinds", dict(Counter(k.split("_", 1)[0] for k in tasks))),
                          ("trivial_tasks_d0_le15", sorted(k for k, t in tasks.items()
                                                           if _le(spawn_distance(t, coords), THR_M))),
                          ("missing_coords", sum(1 for p in pano if p not in coords)),
                          ("floors_source", fsrc_path), ("floors", OrderedDict()), ("evals", OrderedDict())])
        for m in ("STOP0", "FWD", "ODO", "RM"):
            pt = per_policy.get(m)
            if not pt:
                continue
            kn = {k: [stem for stem in tasks if stem.startswith(k)] for k in ("nav", "vis")}
            sd["floors"][m] = {k: rate(sum(1 for stem in v if (pt.get(stem) or {}).get("success")), len(v))
                               for k, v in kn.items()}
            if m in ("ODO", "RM") and logs_dir:
                eps = floor_episodes(pt, policy_logs(logs_dir / m), tasks, coords, nodes)
                sd["floors"][m]["behaviour"] = summarize_eval(eps)
        for label, d in rows.items():
            if not (Path(d) / "results.json").exists():
                sd["evals"][label] = {"dir": str(d), "state": "not run" if not Path(d).exists() else "no results.json"}
                continue
            res = read_json(Path(d) / "results.json") or {}
            eps = eval_episodes(d, tasks, coords, nodes)
            eps_keep[(s, label)] = eps
            sd["evals"][label] = OrderedDict([("dir", str(d)), ("state", "ok"),
                                              ("adapter_sha256", (res.get("adapter") or {}).get("adapter_sha256")),
                                              ("blind", res.get("blind")), ("constrained", res.get("constrained")),
                                              ("render_size", res.get("render_size")),
                                              ("max_tokens", res.get("max_tokens")),
                                              ("regex_valid_pct", (res.get("regex") or {}).get("regex_valid_pct")),
                                              ("finish_length", (res.get("regex") or {}).get("finish_length")),
                                              ("tasks_without_log", res.get("tasks_without_log")),
                                              ("context_overflow_steps", res.get("context_overflow_steps")),
                                              ("wall_seconds", res.get("wall_seconds")),
                                              ("summary", summarize_eval(eps))])
        doc["sets"][s] = sd
    pairs = OrderedDict()
    for (la, lb) in (("483A", "483A rep2 (workers 4)"), ("483A", "483A blind"), ("raw", "raw off 640 (max_tokens 1024)")):
        if ("s2test", la) in eps_keep and ("s2test", lb) in eps_keep:
            pairs[f"s2test: {la} vs {lb}"] = pair_runs(eps_keep[("s2test", la)], eps_keep[("s2test", lb)])
    doc["pairs"] = pairs
    losses = OrderedDict()
    for name, d in (("0W", pilot / "adapters" / "0W"), ("50", pilot / "adapters" / "50"),
                    ("200", pilot / "adapters" / "200"), ("483A", x / "adapters" / "483A"),
                    ("483B", x / "adapters" / "483B")):
        losses[name] = adapter_losses(name, d)
    doc["training"] = losses
    tf_dir = x / "tf"
    if tf_dir.is_dir() and any(tf_dir.glob("tf_summary_*.json")):
        merge_summaries(tf_dir)
    doc["tf"] = read_json(tf_dir / "tf_summary.json")
    doc["merge_parity"] = {k: v for k, v in (read_json(tf_dir / "merge_parity.json") or {}).items() if k != "per_demo"} \
        or None
    write_json(x / "diag.json", doc)
    write_text(x / "REPORT_EXPLORATORY.md", report_md(doc))
    return doc


def _r(b) -> str:
    if not b or not b.get("n"):
        return "—"
    ci = b.get("wilson95") or [None, None]
    return f"{b['k']}/{b['n']} ({100 * b['rate']:.1f}%, {100 * ci[0]:.1f}–{100 * ci[1]:.1f})"


def _k(b) -> str:
    return "—" if not b or not b.get("n") else f"{b['k']}/{b['n']}"


def _p(b, digits=3) -> str:
    return "—" if not b or b.get("rate") is None else f"{b['rate']:.{digits}f}"


def report_md(doc: dict) -> str:
    L = ["# WalkerBench 483 可学性检查（探索性，EXPLORATORY）", "",
         f"生成时间 {doc['utc']}。用户 2026-11-09 要求：只用 Q3B，在全部 483 道训练题上微调，**不做判断**。",
         "这不是事先登记的试点的一部分；没有任何判定规则，下面只列数字。试点的 G1 结果（runs/pilot/GATE.md）不变。", "",
         "代号：raw = 原始 Q3B；0W / 50 / 200 = 试点的 Q3B 适配器（只读复用）；483A = 483 题、--epochs auto（3 轮）；"
         "483B = 483 题、10 轮。s2in50 = demos_50 的 50 道题（50/200/483 训练过，raw/0W 没见过）。", ""]
    for s, sd in doc["sets"].items():
        if "evals" not in sd:
            L += [f"## {s}", "", sd.get("missing", ""), ""]
            continue
        L += [f"## {s}（{sd['n_tasks']} 题，{sd['kinds']}；起点 15 m 内的题 {len(sd['trivial_tasks_d0_le15'])} 道）", "",
              "### 闭环成功率", "",
              "| 模型 | nav 成功 | vis 成功 | oracle nav | oracle vis | 平均步数 | 以 STOP 结束 | 撞步数上限 | 非平凡 nav | 非平凡 vis |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for label, e in sd["evals"].items():
            if e.get("state") != "ok":
                L.append(f"| {label} | {e.get('state')} | | | | | | | | |")
                continue
            sm = e["summary"]
            L.append(f"| {label} | {_r(sm['nav'].get('success'))} | {_r(sm['vis'].get('success'))} | "
                     f"{_k(sm['nav'].get('oracle'))} | {_k(sm['vis'].get('oracle'))} | {sm['all'].get('mean_steps')} | "
                     f"{_k(sm['all'].get('ended_stop'))} | {sm['all'].get('cap_hits')} | "
                     f"{_k(sm['nav'].get('nontrivial_success'))} | {_k(sm['vis'].get('nontrivial_success'))} |")
        L += ["", "### 行为", "",
              "| 模型 | P(STOP\\|≤15m) | P(STOP\\|>15m) | 首次到达目标全景后的动作 | 转弯节点（RM 旋转处） | 转弯节点（旋转或 MOVE(k>1)） | 直行节点 | 与 FWD 每步一致 | 自己路径+走够就停 (nav) | STOP 时 走过/给定 在 ±20% |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for label, e in list(sd["evals"].items()) + [(f"floor {m}", {"state": "ok", "summary": v["behaviour"]})
                                                     for m, v in sd["floors"].items() if v.get("behaviour")]:
            if e.get("state") != "ok":
                continue
            a, nv = e["summary"]["all"], e["summary"]["nav"]
            L.append(f"| {label} | {_k(a.get('p_stop_within15'))} = {_p(a.get('p_stop_within15'))} | "
                     f"{_k(a.get('p_stop_beyond15'))} = {_p(a.get('p_stop_beyond15'))} | "
                     f"{a.get('first_arrival_target_pano_next')} | {_k(a.get('node_turn_rot'))} | {_k(a.get('node_turn_any'))} | "
                     f"{_k(a.get('node_straight_rot'))} | {_p(a.get('fwd_step_agreement'))} | "
                     f"{_k(nv.get('cf_own_path_stop_at_walked_ge_stated'))} | {_k(a.get('stop_walked_over_stated_within20'))} |")
        L += ["", f"### 下限（{sd.get('floors_source')}）", "", "| 策略 | nav | vis |", "|---|---|---|"]
        for m, v in sd["floors"].items():
            L.append(f"| {m} | {_r(v.get('nav'))} | {_r(v.get('vis'))} |")
        L.append("")
    ref = doc["references"]
    L += ["## 参照数字（只列出，不作比较规则）", "",
          f"- 论文 Qwen3-VL-8B Original：nav {100 * ref['paper_Qwen3-VL-8B_Original']['nav']:.2f}%，vis "
          f"{100 * ref['paper_Qwen3-VL-8B_Original']['vis']:.2f}%；Spatial-IDE：nav "
          f"{100 * ref['paper_Qwen3-VL-8B_Spatial-IDE']['nav']:.2f}%，vis {100 * ref['paper_Qwen3-VL-8B_Spatial-IDE']['vis']:.2f}%。",
          f"- 全部 350 道 nav 的盲 ODO：{ref['ODO_full350_nav']['k']}/{ref['ODO_full350_nav']['n']} = "
          f"{100 * ref['ODO_full350_nav']['rate']:.1f}%。",
          f"- 边际预测器的每回复 token 损失：demos_all 只猜类型 {ref['marginal_loss_demos_all']['type_only']}，"
          f"加均匀猜 MOVE id {ref['marginal_loss_demos_all']['with_ids']}；demos_200 "
          f"{ref['marginal_loss_demos_200']['type_only']} / {ref['marginal_loss_demos_200']['with_ids']}。", ""]
    if doc.get("pairs"):
        L += ["## 成对比较（同一批题）", "", "| 比较 | 题数 | 动作序列完全相同 | 都成功 | 只有前者成功 | 只有后者成功 |", "|---|---|---|---|---|---|"]
        for k, v in doc["pairs"].items():
            L.append(f"| {k} | {v['tasks']} | {v['identical_action_seq']} | {v['success_both']} | "
                     f"{v['success_only_first']} | {v['success_only_second']} |")
        L.append("")
    L += ["## 训练损失（在线损失，dropout 开，每回复 token）", "",
          "| 适配器 | 数据条数 | 轮数 | 优化步 | 各轮平均 | 最后一步 | 最后 10 步平均 |", "|---|---|---|---|---|---|---|"]
    for n, v in doc["training"].items():
        if not v:
            L.append(f"| {n} | 不存在 | | | | | |")
            continue
        L.append(f"| {n} | {v['conversations']} | {v['epochs']} | {v['optimizer_steps']} | "
                 f"{', '.join(f'{x}' for x in v['epoch_mean_loss'].values())} | {v['loss_last']} | "
                 f"{None if v['loss_last10_mean'] is None else round(v['loss_last10_mean'], 5)} |")
    L.append("")
    tf = (doc.get("tf") or {}).get("adapters") or {}
    if tf:
        L += ["## Teacher-forced 探针（只看下一步动作，不闭环）", "",
              "| 适配器 | 示范集 | 训练过的示范 | 类型准确率 | 召回 MOVE / ROT / STOP | STOP 精确率 | 下一步完全一致 | 转弯节点 | 直行节点 | STOP AUROC | P(STOP) 末步 / 非末步 | 末步贪心 STOP | 回复 CE | 决策 token CE | 边际：只类型 / +经验 id / +均匀 id |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a, d in tf.items():
            for s, sv in (d.get("sets") or {}).items():
                v = sv.get("all") or {}
                mr = v.get("marginal_reference") or {}
                rc = v.get("recall") or {}
                L.append(f"| {a} | {s} | {sv.get('in_sample_demos')}/{sv.get('n_demos_set')} | {_p(v.get('type_acc'))} | "
                         f"{_p(rc.get('MOVE'))} / {_p(rc.get('ROTATION'))} / {_p(rc.get('STOP'))} | "
                         f"{_p(v.get('stop_precision'))} | {_p(v.get('next_exact'))} | {_p(v.get('next_exact_turn'))} | "
                         f"{_p(v.get('next_exact_straight'))} | {v.get('stop_auroc')} | {v.get('p_stop_final_mean')} / "
                         f"{v.get('p_stop_nonfinal_mean')} | {_p(v.get('greedy_stop_recall_final'))} | "
                         f"{v.get('reply_ce_trainer_weighting')} | {v.get('decision_ce_per_token')} | "
                         f"{mr.get('type_only_per_reply_token')} / {mr.get('type_emp_id_per_reply_token')} / {mr.get('type_uniform_id_per_reply_token')} |")
        L += ["", "nav 上 P(STOP) 按 走过/给定 分箱（每格：n，平均 P(STOP)，专家 STOP 比例）：", ""]
        L += ["| 适配器 | 示范集 | " + " | ".join(STOP_BINS) + " | 规则“走够就停”的 STOP 召回 / 精确率 | 模型 argmax 的 STOP 召回 / 精确率 |",
              "|---|---|" + "---|" * len(STOP_BINS) + "---|---|"]
        for a, d in tf.items():
            for s, sv in (d.get("sets") or {}).items():
                v = sv.get("all") or {}
                bins = v.get("nav_p_stop_by_walked_over_stated") or {}
                det = v.get("nav_stop_detection") or {}
                rule, mod = det.get("rule_walked_ge_stated") or {}, det.get("model_argmax") or {}
                L.append(f"| {a} | {s} | " + " | ".join(
                    f"{(bins.get(b) or {}).get('n')}, {(bins.get(b) or {}).get('mean_p_stop')}, "
                    f"{(bins.get(b) or {}).get('expert_stop_rate')}" for b in STOP_BINS) +
                    f" | {rule.get('stop_recall')} / {rule.get('stop_precision')} | {mod.get('stop_recall')} / "
                    f"{mod.get('stop_precision')} |")
        L.append("")
    mp = doc.get("merge_parity")
    if mp:
        L += ["## 合并一致性（VAL56 前 20 条，合并后 bf16 vs 未合并 LoRA）", "",
              f"- 平均 |ΔCE| {mp.get('mean_abs_diff_ce')}，最大 {mp.get('max_abs_diff_ce')}；类型预测一致 "
              f"{_k(mp.get('pred_type_agree'))}；最大 |ΔP(type)| {mp.get('max_abs_diff_p_type')}。", ""]
    L += ["---", "本报告只列数字，没有结论，没有闸门。"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------- outputs, checks
def expected_outputs(arm: str, x: Path = X_ROOT) -> list:
    x = Path(x)
    out = []
    name, _ = ARM_TRAIN[arm]
    out += [x / "adapters" / name / f for f in ("adapter.pt", "meta.json", "log.jsonl")]
    if arm == "A":
        out += [x / "floors_s2val.json", x / "floors_s2in50.json", x / "tf" / "merge_parity.json"]
    for (_, model, s, res, mode, variant) in ARM_EVALS[arm]:
        out.append(x / "evals" / eval_name(model, s, res, mode, variant) / "results.json")
    for a in ARM_TF[arm]:
        for s in SETS:
            out += [x / "tf" / f"tf_{a}_{s}.jsonl", x / "tf" / f"tf_{a}_{s}.done.json"]
        out.append(x / "tf" / f"tf_summary_{a}.json")
    return out


def init_check(init_dir: Path, expect_sha: str = INIT_SHA_PREFIX, expect_code: str = CODE_SHA_PREFIX) -> tuple:
    from vpd.train import n2_train_vpd as T

    init_dir = Path(init_dir)
    probs, facts = [], {}
    pt = init_dir / "adapter.pt"
    facts["init_sha256"] = sha256_file(pt) if pt.is_file() else None
    if not facts["init_sha256"] or not facts["init_sha256"].startswith(expect_sha):
        probs.append(f"{pt}: sha256 {facts['init_sha256']} does not start with {expect_sha}")
    facts["trainer_code_sha256"] = T.code_fingerprint()["code_sha256"]
    if not facts["trainer_code_sha256"].startswith(expect_code):
        probs.append(f"trainer code {facts['trainer_code_sha256'][:12]} != pilot {expect_code} (n2_train_vpd changed)")
    meta = read_json(init_dir / "meta.json") or {}
    facts["init_code_sha256"] = meta.get("code_sha256")
    if not str(meta.get("code_sha256") or "").startswith(expect_code):
        probs.append(f"{init_dir}/meta.json code {meta.get('code_sha256')} != {expect_code}")
    return probs, facts


def setup(x: Path = X_ROOT, dest: Path = S2IN50_DIR, tier50: Path = TIERS / "demos_50.jsonl",
          split_dir: Path = paths.WB_SPLIT) -> dict:
    """s2in50 (created or verified) and X/{adapters, tf, evals}; mkdir only, nothing else is written."""
    x = Path(x)
    refuse_pilot(x)
    rep = {"s2in50": build_insample(dest, tier50, split_dir)}
    for d in (x, x / "adapters", x / "tf", x / "evals"):
        d.mkdir(parents=True, exist_ok=True)
    rep["x"] = str(x)
    return rep


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "eval":
        return cmd_eval(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup", help="s2in50 + X dirs (login node, before prepare; in-job with --ensure)")
    p.add_argument("--x", type=Path, default=X_ROOT)
    p.add_argument("--ensure", action="store_true", help="accepted for the payload; setup always ensures s2in50")
    p = sub.add_parser("insample", help="create or verify OUT/walkerbench/split/s2in50")
    p.add_argument("--dest", type=Path, default=S2IN50_DIR)
    p = sub.add_parser("floors")
    p.add_argument("--task-set", choices=("s2val", "s2in50", "s2test"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--logs", type=Path, default=None, help="default: OUT dir of --out / floors_logs / SET")
    p.add_argument("--work", type=Path, default=paths.scratch("x483-floors"), help="sandbox parent (removed)")
    p.add_argument("--tasks", default=None, help="comma-separated stems (subset; tests)")
    p = sub.add_parser("init-check")
    p.add_argument("--init", type=Path, default=PILOT_Q3B / "adapters" / "0W")
    p.add_argument("--expect-sha", default=INIT_SHA_PREFIX)
    p.add_argument("--expect-code", default=CODE_SHA_PREFIX)
    for nm in ("tf", "tf-parity"):
        p = sub.add_parser(nm)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--start", default=START)
        p.add_argument("--model-path", type=Path, default=None)
        p.add_argument("--attn", default="sdpa", choices=("sdpa", "fa2"))
        p.add_argument("--device", default="cuda")
        p.add_argument("--threads", type=int, default=2)
        p.add_argument("--num-workers", type=int, default=3)
        p.add_argument("--demos-all", type=Path, default=DEMOS_ALL)
        p.add_argument("--tiers", type=Path, default=TIERS)
        if nm == "tf":
            p.add_argument("--adapter", action="append", default=[], help="NAME=DIR, or raw= (A = B = 0)")
            p.add_argument("--sets", default=",".join(SETS))
            p.add_argument("--greedy-sets", default=",".join(GREEDY_SETS))
            p.add_argument("--limit", type=int, default=None, help="first N demos per set (tests)")
        else:
            p.add_argument("--merged", type=Path, required=True)
            p.add_argument("--adapter", type=Path, required=True)
            p.add_argument("--n", type=int, default=PARITY_N)
    p = sub.add_parser("parity-ok")
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--x", type=Path, default=X_ROOT)
    p.add_argument("--pilot", type=Path, default=PILOT_Q3B)
    p = sub.add_parser("expect")
    p.add_argument("--arm", choices=("A", "B"), required=True)
    p.add_argument("--x", type=Path, default=X_ROOT)
    p.add_argument("--check", action="store_true", help="exit 1 when anything is missing")
    args = ap.parse_args(argv)
    if args.cmd not in ("tf", "tf-parity"):
        thread_env()
    if args.cmd == "setup":
        print(json.dumps(setup(args.x)))
    elif args.cmd == "insample":
        print(json.dumps(build_insample(args.dest)))
    elif args.cmd == "floors":
        logs = args.logs or Path(args.out).parent / "floors_logs" / args.task_set
        stems = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
        doc = run_floors(args.task_set, args.out, logs, args.work, stems)
        print(json.dumps({"floors": doc["floors"], "out": str(args.out)}))
    elif args.cmd == "init-check":
        probs, facts = init_check(args.init, args.expect_sha, args.expect_code)
        print(json.dumps({"ok": not probs, "problems": probs, **facts}))
        return 1 if probs else 0
    elif args.cmd == "tf":
        return cmd_tf(args)
    elif args.cmd == "tf-parity":
        return cmd_tf_parity(args)
    elif args.cmd == "parity-ok":
        return cmd_parity_ok(args)
    elif args.cmd == "analyze":
        doc = analyze(args.x, args.pilot)
        print(f"[learn483] wrote {Path(args.x) / 'diag.json'} and {Path(args.x) / 'REPORT_EXPLORATORY.md'}")
        for s, sd in doc["sets"].items():
            for label, e in (sd.get("evals") or {}).items():
                if e.get("state") == "ok":
                    sm = e["summary"]
                    print(f"  {s:7s} {label:32s} nav {_k(sm['nav'].get('success'))} vis {_k(sm['vis'].get('success'))}")
    elif args.cmd == "expect":
        missing = 0
        for pth in expected_outputs(args.arm, args.x):
            ok = pth.exists()
            missing += not ok
            print(f"{'ok     ' if ok else 'MISSING'} {pth}")
        print(f"[learn483] arm {args.arm}: {missing} missing")
        return 1 if (args.check and missing) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
