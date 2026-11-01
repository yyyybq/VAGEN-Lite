#!/usr/bin/env python3
"""Paths another line depends on, which a deletion batch may touch only after the user says yes for that batch
(the user, 2026-11-01: "删除批次里涉及这些路径的先问我"). The list is docs/disk/PROTECTED_PATHS.tsv: a folder per
row, glob wildcards allowed; a file is protected when it lies below one of them.

  python disk/protected.py records   every batch's deleted.tsv / restored_*.tsv and every not-yet-executed plan.tsv,
                                     against the list: what each batch removed (or restored, or plans) below each row

archive_delete.py refuses plan and delete for a batch whose files hit the list unless its spec carries
"protected_ok": the user's words that approved touching them.
"""
import fnmatch
import glob
import os
import sys

U = "/mnt/umm/users/yinbaiqiao"
LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "disk", "PROTECTED_PATHS.tsv")


def patterns():
    with open(LIST) as f:
        f.readline()
        return [line.split("\t")[0].rstrip("/") for line in f if line.strip()]


def match(path, pats=None):
    """The list row a path lies below (or is), else None."""
    for p in pats if pats is not None else patterns():
        if fnmatch.fnmatchcase(path, p) or fnmatch.fnmatchcase(path, p + "/*"):
            return p
    return None


def hits(rows, pats=None):
    """rows: dicts with "path"; returns {row pattern: [rows below it]}."""
    pats = pats if pats is not None else patterns()
    out = {}
    for r in rows:
        p = match(r["path"], pats)
        if p:
            out.setdefault(p, []).append(r)
    return out


def refuse_unless_ok(spec, rows):
    """Called by archive_delete.py plan and delete."""
    h = hits(rows)
    if not h:
        print("protected paths (docs/disk/PROTECTED_PATHS.tsv): none touched")
        return
    for p, rs in h.items():
        print("PROTECTED %s: %d files, %.1f GB" % (p, len(rs), sum(r["bytes"] for r in rs) / 1e9))
    if not spec.get("protected_ok"):
        sys.exit("refused: the batch touches paths other lines depend on; ask the user, then record the answer in the "
                 "spec as \"protected_ok\"")
    print("touched with the user's approval: %s" % spec["protected_ok"])


def read_tsv(path, col):
    rows = []
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        i, j = head.index(col), head.index("bytes")
        for line in f:
            c = line.rstrip("\n").split("\t")
            rows.append({"path": c[i], "bytes": int(c[j])})
    return rows


def group(rows, base):
    by = {}
    for r in rows:
        rel = os.path.relpath(r["path"], base).split("/")
        k = "/".join(rel[:4]) if len(rel) > 4 else "/".join(rel[:-1]) or "."
        by.setdefault(k, [0, 0])
        by[k][0] += 1
        by[k][1] += r["bytes"]
    return by


def cmd_records():
    pats = patterns()
    found = 0
    for run in sorted(glob.glob(os.path.join(U, "disk_runs", "batches", "*")), key=lambda d: int(os.path.basename(d)[1:].split("_")[0])):
        b = os.path.basename(run)
        sources = []
        if os.path.exists(os.path.join(run, "deleted.tsv")):
            sources.append(("deleted", read_tsv(os.path.join(run, "deleted.tsv"), "path")))
        elif os.path.exists(os.path.join(run, "plan.tsv")):
            sources.append(("planned, not executed", read_tsv(os.path.join(run, "plan.tsv"), "path")))
        for rp in sorted(glob.glob(os.path.join(run, "restored_*.tsv"))):
            sources.append(("restored (%s)" % os.path.basename(rp), read_tsv(rp, "path")))
        for what, rows in sources:
            for p, rs in hits(rows, pats).items():
                found += 1
                base = p if "*" not in p else os.path.dirname(p)
                print("%-16s %-22s %s: %d files, %.1f GB" % (b, what, p, len(rs), sum(r["bytes"] for r in rs) / 1e9))
                for k, (n, by) in sorted(group(rs, base).items()):
                    print("      %-90s %5d files %8.1f GB" % (k, n, by / 1e9))
    print("%d batch/record pairs touch the list" % found)


if __name__ == "__main__":
    if sys.argv[1:] != ["records"]:
        sys.exit(__doc__)
    cmd_records()
