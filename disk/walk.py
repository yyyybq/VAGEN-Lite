#!/usr/bin/env python3
"""One full walk of a directory tree (what du does), read only, with the facts du does not give.

For every directory down to --agg-depth below ROOT it writes one row: bytes (apparent size, every
inode counted once as du does), allocated bytes (st_blocks * 512), files, directories, the newest
mtime and the oldest mtime of any file below it, the number of root-owned files, and the bytes of
files whose last modification is older than 7 / 14 / 30 days (counted at --now).
It also lists every file of at least --big bytes (path, size, inode, links, mtime, uid) and every
file whose name looks like a deletion or clean-up record (delet / cleanup / reclaim / removed /
purge / retain / prune), so that past clean-ups can be traced.

Symlinks are listed, never followed. Unreadable directories are counted and reported.

    python walk.py ROOT --out DIR [--agg-depth 6] [--big 268435456]

Writes DIR/dirs.tsv, DIR/big_files.tsv, DIR/symlinks.tsv, DIR/cleanup_records.tsv,
DIR/errors.tsv and DIR/summary.json (timing included). Progress goes to stderr every 60 s.
"""
import argparse
import json
import os
import re
import stat
import sys
import time

RECORD_RE = re.compile(r"delet|cleanup|clean_up|reclaim|removed|purge|retain|prune", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--agg-depth", type=int, default=6)
    ap.add_argument("--big", type=int, default=256 * 1024 * 1024)
    ap.add_argument("--now", type=float, default=None, help="epoch seconds for the age buckets (default: start time)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    now = args.now or t0
    ages = [7, 14, 30]
    cut = [now - d * 86400 for d in ages]

    big = open(os.path.join(args.out, "big_files.tsv"), "w")
    big.write("path\tbytes\tino\tnlink\tmtime\tuid\n")
    links = open(os.path.join(args.out, "symlinks.tsv"), "w")
    links.write("path\ttarget\tmtime\tuid\n")
    recs = open(os.path.join(args.out, "cleanup_records.tsv"), "w")
    recs.write("path\tbytes\tmtime\n")
    errs = open(os.path.join(args.out, "errors.tsv"), "w")
    errs.write("path\terror\n")

    # aggregates for directories at depth <= agg_depth, keyed by path
    agg = {}
    seen_inodes = set()  # inodes with nlink > 1 already counted
    n_files = n_dirs = n_err = 0
    last_report = t0

    # a row is a list (dicts for some 10^5-10^6 directories cost too much memory)
    B, A, F, D, NEW, OLD, RF, O7, O14, O30 = range(10)

    def new_row():
        return [0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0]

    # iterative DFS; for each directory we know its ancestors at depth <= agg_depth
    stack = [(root, 0)]
    agg[root] = new_row()
    while stack:
        path, depth = stack.pop()
        # the aggregate rows this directory's contents add to
        parts = path[len(root):].strip("/").split("/") if path != root else []
        keys = [root] + ["/".join([root] + parts[:i]) for i in range(1, min(len(parts), args.agg_depth) + 1)]
        rows = [agg[k] for k in keys]
        try:
            it = os.scandir(path)
        except OSError as e:
            errs.write("%s\t%r\n" % (path, e))
            n_err += 1
            continue
        with it:
            for e in it:
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError as err:
                    errs.write("%s\t%r\n" % (e.path, err))
                    n_err += 1
                    continue
                mode = st.st_mode
                if stat.S_ISLNK(mode):
                    try:
                        tgt = os.readlink(e.path)
                    except OSError:
                        tgt = "?"
                    links.write("%s\t%s\t%d\t%d\n" % (e.path, tgt, int(st.st_mtime), st.st_uid))
                    continue
                if stat.S_ISDIR(mode):
                    n_dirs += 1
                    for r in rows:
                        r[D] += 1
                    if depth + 1 <= args.agg_depth:
                        agg[e.path] = new_row()
                    stack.append((e.path, depth + 1))
                    continue
                n_files += 1
                if st.st_nlink > 1:
                    key = (st.st_dev, st.st_ino)
                    if key in seen_inodes:
                        continue
                    seen_inodes.add(key)
                size = st.st_size
                mt = st.st_mtime
                alloc = st.st_blocks * 512
                is_root = st.st_uid == 0
                for r in rows:
                    r[B] += size
                    r[A] += alloc
                    r[F] += 1
                    if mt > r[NEW]:
                        r[NEW] = mt
                    if r[OLD] == 0 or mt < r[OLD]:
                        r[OLD] = mt
                    if is_root:
                        r[RF] += 1
                    if mt < cut[0]:
                        r[O7] += size
                        if mt < cut[1]:
                            r[O14] += size
                            if mt < cut[2]:
                                r[O30] += size
                if size >= args.big:
                    big.write("%s\t%d\t%d\t%d\t%d\t%d\n" % (e.path, size, st.st_ino, st.st_nlink, int(mt), st.st_uid))
                if RECORD_RE.search(e.name) and size < 1 << 30:
                    recs.write("%s\t%d\t%d\n" % (e.path, size, int(mt)))
        if time.time() - last_report > 60:
            last_report = time.time()
            print("%.0f s: %d files, %d dirs, %d errors, %.2f TB so far, at %s" % (
                last_report - t0, n_files, n_dirs, n_err, agg[root][B] / 1e12, path), file=sys.stderr, flush=True)

    with open(os.path.join(args.out, "dirs.tsv"), "w") as f:
        f.write("depth\tpath\tbytes\talloc\tfiles\tdirs\tnewest\toldest\troot_files\told7\told14\told30\n")
        for k in sorted(agg):
            r = agg[k]
            d = 0 if k == root else k[len(root):].strip("/").count("/") + 1
            f.write("%d\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n" % (
                d, k, r[B], r[A], r[F], r[D], int(r[NEW]), int(r[OLD]), r[RF], r[O7], r[O14], r[O30]))
    for fh in (big, links, recs, errs):
        fh.close()
    summary = {"root": root, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
               "seconds": round(time.time() - t0, 1), "files": n_files, "dirs": n_dirs, "errors": n_err,
               "bytes": agg[root][B], "alloc": agg[root][A], "hardlinked_inodes": len(seen_inodes),
               "now_for_ages": now}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary), file=sys.stderr)


if __name__ == "__main__":
    main()
