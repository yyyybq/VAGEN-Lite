"""Group G's sampling check, the parts the main line's accept_summary (VAGEN-Lite-q3dev, read only) does not read
(the pre-launch review of 2026-11-06): the token entropy split at the action segment (evaluation run --entropy records
entropy_action / entropy_targets per answer since matrix-v3 e4e245c0), the targets segment's form (ok / old / bad and
the errors, the pointing score at temperature 0.8) and the answers that copy the format block's example point.  Both
pass lines are written, over the whole answer ("pass_all", note 6 as written) and over the action segment
("pass_action"), each with entropy <= 0.06 and unreadable turns <= 1%; which one decides G's pick is the user's.

    python m3_accept_g.py --set SET_DIR --main MAIN_SUMMARY.json --out OUT.json NAME [NAME ...]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix")
from data_gen.interiorgs_v10 import evaluation as EV  # noqa: E402

ENTROPY_MAX = 0.06
UNREADABLE_MAX = 0.01
EXAMPLE = "(469,344)"  # prompt.targets_example of targets_norm1000


def summarise(folder: Path, main: dict) -> dict:
    if not (folder / "done.json").is_file():
        return {"missing": True}
    rows = [json.loads(l) for l in (folder / "results.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "error" not in r]
    turns = [t for r in rows for t in (r.get("trajectory") or [])]
    ent = EV.token_entropy(rows)
    pts = [t["pointing"] for t in turns if t.get("pointing")]
    errors = collections.Counter(p.get("error") or "ok" for p in pts)
    score = sum(float(p.get("score") or 0.0) for p in pts) / max(1, len(pts))
    copies = sum(EXAMPLE in str(t.get("answer") or "").replace(" ", "") for t in turns)
    share = float(main.get("unreadable_share", 1.0))
    sampled = bool(main.get("samples") and main.get("temperature"))
    out = {"turns": len(turns), "unreadable_share": share, **ent,
           "pointing_turns": len(pts), "targets_errors": dict(errors), "pointing_score": round(score, 4),
           "targets_ok_share": round(errors.get("ok", 0) / max(1, len(pts)), 4),
           "example_copies": copies, "format_example": main.get("format_example")}
    out["pass_all"] = bool(sampled and ent.get("entropy") is not None and ent["entropy"] <= ENTROPY_MAX
                           and share <= UNREADABLE_MAX)
    out["pass_action"] = bool(sampled and ent.get("entropy_action") is not None and ent["entropy_action"] <= ENTROPY_MAX
                              and share <= UNREADABLE_MAX)
    if not pts:
        out["note"] = "no pointing records: the run did not use a targets format"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True)
    ap.add_argument("--main", required=True, help="the main line's accept_summary output for the same names")
    ap.add_argument("--out", required=True)
    ap.add_argument("names", nargs="+")
    a = ap.parse_args()
    base = json.loads(Path(a.main).read_text())
    out = {n: summarise(Path(a.set) / n, base.get(n) or {}) for n in a.names}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for n, s in out.items():
        print(n, json.dumps({k: s.get(k) for k in ("entropy", "entropy_action", "entropy_targets", "unreadable_share",
                                                    "targets_ok_share", "pointing_score", "example_copies",
                                                    "pass_all", "pass_action")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
