"""Reward line (the final report, 2026-11-06): card-hours of every sco job under a runs root, from the job logs.

Each job writes <mode>-<UTC start>-pt-<id>-worker-<k>.log next to its outputs.
- rl / sft / eval / qa / run ... logs: a job's span is the start in the name to the file's last write (mtime).
- renderer logs are written only when the service starts; a render service's span is its first log's start to
  renderer_exit.json "ended_utc" in the same logs folder (else the newest renderer_service*.log there), one span per
  sco job id.  A service that waited for its trainer's cards counts the wait: it held its card.
Cards: rl and sft 8, every other mode 1.  Jobs whose logs follow another pattern are not counted (this line's table
build, tables/jobs/b3000_all: one 8-card CPU job, its own build logs give the hours).

    python -m reward.cardhours --root /mnt/umm/users/yinbaiqiao/reward_runs [--stages STAGES.json] [--lines]

STAGES.json maps a stage name to folder prefixes relative to the root ({"B": ["runs/rw_b_"], ...}); without it the
totals are per top folder (runs/<chain>, jobs/<job>, tables/jobs).  Read-only; standard library only.
"""

import argparse
import datetime as dt
import glob
import json
import os
import re
from collections import defaultdict

PAT = re.compile(r"^([a-z0-9_]+)-(\d{8}T\d{6}Z)-pt-([0-9a-f]+)-worker-(\d+)\.log$")
EIGHT = {"rl", "sft"}
SKIP_DIRS = {"checkpoints", "rollout_data", "frames", "huggingface", "torch_extensions_eval",
             "torch_extensions_renderer"}
UTC = dt.timezone.utc


def _mtime(path):
    return dt.datetime.fromtimestamp(os.path.getmtime(path), tz=UTC)


def _renderer_end(logdir, start):
    ex = os.path.join(logdir, "renderer_exit.json")
    if os.path.exists(ex):
        try:
            with open(ex) as f:
                return dt.datetime.fromisoformat(json.load(f)["ended_utc"].replace("Z", "+00:00"))
        except (ValueError, KeyError, OSError):
            pass
    cands = glob.glob(os.path.join(logdir, "renderer_service*.log"))
    return max(_mtime(c) for c in cands) if cands else start


def spans(root):
    """[(relative log path, mode, start, end, cards)] for every job log under root."""
    out, renderers = [], {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            m = PAT.match(fn)
            if not m:
                continue
            mode, start, pid, _worker = m.groups()
            path = os.path.join(dirpath, fn)
            t0 = dt.datetime.strptime(start, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
            if mode == "renderer":
                key = (dirpath, pid)
                if key not in renderers or t0 < renderers[key][0]:
                    renderers[key] = (t0, os.path.relpath(path, root))
                continue
            out.append((os.path.relpath(path, root), mode, t0, _mtime(path), 8 if mode in EIGHT else 1))
    for (logdir, _pid), (t0, rel) in renderers.items():
        out.append((rel, "renderer", t0, _renderer_end(logdir, t0), 1))
    return sorted(out, key=lambda r: r[2])


def group_of(rel, stages):
    if stages:
        for name, prefixes in stages.items():
            if any(rel.startswith(p) for p in prefixes):
                return name
        return "other"
    return "/".join(rel.split("/")[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stages", default=None)
    ap.add_argument("--lines", action="store_true", help="print every job span")
    args = ap.parse_args()
    stages = None
    if args.stages:
        with open(args.stages) as f:
            stages = json.load(f)
    total, by_mode = defaultdict(float), defaultdict(float)
    for rel, mode, t0, t1, cards in spans(args.root):
        ch = max(0.0, (t1 - t0).total_seconds() / 3600.0) * cards
        g = group_of(rel, stages)
        total[g] += ch
        by_mode[(g, mode)] += ch
        if args.lines:
            print("%-90s %-9s %s %s x%d %7.2f" % (rel[:90], mode, t0.strftime("%m-%d %H:%M"),
                                                  t1.strftime("%m-%d %H:%M"), cards, ch))
    for (g, mode) in sorted(by_mode):
        print("%-28s %-14s %8.2f card-h" % (g, mode, by_mode[(g, mode)]))
    print()
    for g in sorted(total):
        print("%-28s %8.2f card-h" % (g, total[g]))
    print("%-28s %8.2f card-h" % ("TOTAL", sum(total.values())))


if __name__ == "__main__":
    main()
