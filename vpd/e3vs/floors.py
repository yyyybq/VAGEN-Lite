"""Blind E3VS floors: the best fixed action sequence (length 0-3) chosen on train, reported on val and test.

Every sequence over the 10 non-stop actions (vpd.e3vs.SEARCH_ORDER) of length 0, 1, 2 or 3 (1,111 sequences)
is executed from each episode's start pose with the official action semantics (geometry.apply_action, camera
mode, 0.25 m / 30 deg, no pitch limit) and, if ``--collides`` is given, the official restrict_collision rule
(a colliding move is skipped and the step consumed).  The final pose (the blind policy then issues STOP) is
scored with geometry.success (0.25 m AND <= 20 deg to any answerable viewpoint).

Choice: the highest train success; ties -> the shorter sequence -> the earlier one in enumeration order
(lengths ascending, itertools.product order over SEARCH_ORDER).  Val and test never influence the choice.

Pre-registered gate floor (DECISIONS B8, PREREG_E3VS_PILOT): "the train-chosen fixed action sequence's test
success under the frozen collision model".  Run without collision this reproduces the survey: (move_up,
look_down) train 16.1%, val 10.8%, test 14.1%.  The frozen floor comes from the e3vs-prep job with
``--collides <occupancy factory>``.

Output (``--out``, default OUT/e3vs/floors/floors_<tag>.json): the choice with k/n/rate/Wilson 95% per split,
the train top 10, reference sequences, the per-episode test and val outcomes of the chosen sequence, inputs and
code sha256.

Usage: python -m vpd.e3vs.floors [--collides MOD:FUNC --tag T] [--out FILE]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from vpd import paths
from vpd.e3vs import (LOGIN_TMP_REPO, SEARCH_ORDER, SPLITS, SUCCESS_ANG_DEG, SUCCESS_POS_M, check_repo, code_meta,
                      load_episodes, repo_dir)
from vpd.e3vs import geometry as G
from vpd.e3vs.bfs_demos import load_collider_factory

MAX_LEN = 3
REFERENCE_SEQS = (("move_up", "look_down"), ("look_down",), ("move_up",), ("move_up", "move_up", "look_down"))
GATE_DEFINITION = ("the train-chosen fixed action sequence's test success under the frozen collision model "
                   "(DECISIONS B8)")


def sequences(max_len: int = MAX_LEN, order=SEARCH_ORDER) -> list[tuple[str, ...]]:
    return [()] + [s for n in range(1, max_len + 1) for s in itertools.product(order, repeat=n)]


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def episode_hits(ep: dict, seqs, collides=None) -> np.ndarray:
    """Boolean success of each sequence (in ``seqs`` order; prefixes must precede extensions) for one episode."""
    end = {}
    poses = np.empty((len(seqs), 4, 4))
    for i, s in enumerate(seqs):
        if not s:
            c = np.array(ep["start_c2w"], dtype=np.float64)
        else:
            prev = end[s[:-1]]
            a = s[-1]
            c = G.apply_action(prev, a)
            if collides is not None and a.startswith("move_") and collides(prev[:3, 3].copy(), c[:3, 3].copy()):
                c = prev
        end[s] = c
        poses[i] = c
    g = np.stack([ep["goals"][v] for v in sorted(ep["goals"])])
    gp = g[:, :3, 3]
    gf = g[:, :3, 2] / np.linalg.norm(g[:, :3, 2], axis=1, keepdims=True)
    p = poses[:, :3, 3]
    f = poses[:, :3, 2] / np.linalg.norm(poses[:, :3, 2], axis=1, keepdims=True)
    d2 = ((p[:, None, :] - gp[None, :, :]) ** 2).sum(-1)
    dot = f @ gf.T
    ok = (d2 <= SUCCESS_POS_M * SUCCESS_POS_M) & (dot >= math.cos(math.radians(SUCCESS_ANG_DEG)))
    return ok.any(axis=1)


def split_hits(eps, seqs, factory=None) -> np.ndarray:
    """(n_episodes, n_seqs) boolean table."""
    out = np.zeros((len(eps), len(seqs)), dtype=bool)
    cache_scene, col = None, None
    for i, ep in enumerate(eps):
        if factory is not None and ep["scene_id"] != cache_scene:
            cache_scene, col = ep["scene_id"], factory(ep["scene_id"])
        out[i] = episode_hits(ep, seqs, col)
    return out


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": round(k / n, 6) if n else None, "wilson95": [round(lo, 6), round(hi, 6)]}


def choose(train_hits: np.ndarray) -> int:
    """Index of the train-best sequence; np.argmax returns the first maximum = shorter, then enumeration order."""
    return int(np.argmax(train_hits.sum(axis=0)))


def run(out: Path | None = None, collides: str | None = None, tag: str | None = None, repo: Path | None = None,
        limit: int | None = None, quiet: bool = False) -> dict:
    repo = Path(repo) if repo else repo_dir()
    tag = tag or ("nocoll" if not collides else collides.replace(":", "_").replace(".", "_"))
    if out is None:
        if repo.resolve() == LOGIN_TMP_REPO.resolve():
            raise SystemExit("the repo is the read-only login_tmp copy (provenance pending the user's OK): "
                             "pass --out under the scratch dir instead of writing OUT/e3vs/floors")
        out = paths.E3VS_OUT / "floors" / f"floors_{tag}.json"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    inputs = check_repo(repo)
    factory = load_collider_factory(collides)
    seqs = sequences()
    t0 = time.time()
    eps = {sp: load_episodes(sp, repo, check=False)[:limit] if limit else load_episodes(sp, repo, check=False)
           for sp in SPLITS}
    hits = {sp: split_hits(eps[sp], seqs, factory) for sp in SPLITS}
    best = choose(hits["train"])
    n = {sp: len(eps[sp]) for sp in SPLITS}
    k = {sp: hits[sp].sum(axis=0) for sp in SPLITS}

    def row(i):
        r = {"seq": list(seqs[i])}
        for sp in SPLITS:
            r[sp] = _rate(k[sp][i], n[sp])
        r["valtest"] = _rate(k["val"][i] + k["test"][i], n["val"] + n["test"])
        return r

    order = sorted(range(len(seqs)), key=lambda i: (-int(k["train"][i]), len(seqs[i]), i))
    idx = {s: i for i, s in enumerate(seqs)}
    res = {
        "tag": tag, "collides": collides, "max_len": MAX_LEN, "n_sequences": len(seqs),
        "search_order": list(SEARCH_ORDER), "tolerance": {"pos_m": SUCCESS_POS_M, "ang_deg": SUCCESS_ANG_DEG},
        "chosen": row(best),
        "gate_floor": {"definition": GATE_DEFINITION, "seq": list(seqs[best]), "split": "test",
                       **_rate(k["test"][best], n["test"])},
        "top_train": [row(i) for i in order[:10]],
        "reference_seqs": [row(idx[s]) for s in REFERENCE_SEQS],
        "per_episode": {sp: {str(ep["episode_index"]): bool(hits[sp][j, best]) for j, ep in enumerate(eps[sp])}
                        for sp in ("val", "test")},
        "start_success": {sp: int(k[sp][0]) for sp in SPLITS},
        "limit": limit, "seconds": round(time.time() - t0, 1), "inputs": inputs,
        "code": code_meta(Path(__file__), Path(G.__file__), Path(sys.modules["vpd.e3vs"].__file__)),
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1) + "\n")
    os.replace(tmp, out)
    if not quiet:
        print("start already success (should be 0):", res["start_success"])
        for r in res["top_train"][:6]:
            print("train-best fixed seq", tuple(r["seq"]), "train %.3f val %.3f test %.3f"
                  % (r["train"]["rate"], r["val"]["rate"], r["test"]["rate"]))
        for r in res["reference_seqs"]:
            print("seq", tuple(r["seq"]), "test %.3f val %.3f val+test %.3f"
                  % (r["test"]["rate"], r["val"]["rate"], r["valtest"]["rate"]))
        c = res["chosen"]
        print(f"CHOSEN {tuple(c['seq'])}: train {c['train']['rate']:.3f} val {c['val']['rate']:.3f} "
              f"test {c['test']['rate']:.3f} ({c['test']['k']}/{c['test']['n']}); {res['seconds']}s; wrote {out}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--collides", default=None, help="MODULE:FUNC factory scene_id -> collides(p0, p1)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N episodes per split (tests only)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    run(out=a.out, collides=a.collides, tag=a.tag, repo=a.repo, limit=a.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
