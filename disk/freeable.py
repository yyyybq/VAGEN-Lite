#!/usr/bin/env python3
"""How much deleting each candidate folder would really free, and how old its contents are (read only).

For every PATH (a folder or a file) it walks the tree once and reports:
  bytes        apparent size of every file below it, each inode once
  freeable     bytes that deleting the whole PATH frees: files with one link, plus hard-linked files whose every
               link lies inside the candidate set (a file also linked from outside stays allocated)
  linked_out   bytes of hard-linked files with a link outside the candidate set (not freed)
  files, newest_mtime, oldest_mtime, root_files (owned by root: this user cannot delete them),
  other_uid_files, symlinks
With --skip SUBPATH the subtree is left out (the part of PATH that would be kept).

    python disk/freeable.py --out FILE.tsv [--skip KEEP ...] PATH [PATH ...]
    python disk/freeable.py --out FILE.tsv --list LIST.txt     # one path per line (# comments allowed)
"""
import argparse
import os
import stat
import sys
import time

ME = os.getuid()


def walk(path, skip):
    """Yield (path, lstat) for every non-directory entry below path (path itself if it is a file)."""
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode):
        yield path, st
        return
    stack = [path]
    while stack:
        d = stack.pop()
        if d in skip:
            continue
        try:
            it = os.scandir(d)
        except OSError as e:
            print("scandir failed", d, e, file=sys.stderr)
            continue
        with it:
            for e in it:
                try:
                    s = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISDIR(s.st_mode):
                    stack.append(e.path)
                else:
                    yield e.path, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--list")
    ap.add_argument("--skip", action="append", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = list(args.paths)
    if args.list:
        paths += [l.strip() for l in open(args.list) if l.strip() and not l.startswith("#")]
    skip = {os.path.abspath(s) for s in args.skip}
    rows = {}
    links = {}  # (dev, ino) -> [nlink, links seen inside the candidate set, size]
    for p in paths:
        t0 = time.time()
        r = {"bytes": 0, "single": 0, "files": 0, "newest": 0, "oldest": 0, "root_files": 0, "other_uid": 0,
             "symlinks": 0, "linked": set()}
        seen = set()
        for fp, s in walk(os.path.abspath(p), skip):
            if stat.S_ISLNK(s.st_mode):
                r["symlinks"] += 1
                continue
            key = (s.st_dev, s.st_ino)
            r["files"] += 1
            if s.st_uid == 0:
                r["root_files"] += 1
            elif s.st_uid != ME:
                r["other_uid"] += 1
            r["newest"] = max(r["newest"], s.st_mtime)
            r["oldest"] = s.st_mtime if not r["oldest"] else min(r["oldest"], s.st_mtime)
            if s.st_nlink > 1:
                l = links.setdefault(key, [s.st_nlink, 0, s.st_size])
                l[1] += 1
                r["linked"].add(key)
                if key in seen:
                    continue
            seen.add(key)
            r["bytes"] += s.st_size
            if s.st_nlink == 1:
                r["single"] += s.st_size
        r["seconds"] = time.time() - t0
        rows[p] = r
        print("%-90s %8.1f GB %7.0f files %5.0f s" % (p, r["bytes"] / 1e9, r["files"], r["seconds"]), file=sys.stderr, flush=True)
    with open(args.out, "w") as f:
        f.write("path\tbytes\tfreeable\tlinked_out\tfiles\tnewest_mtime\toldest_mtime\troot_files\tother_uid_files\tsymlinks\tseconds\n")
        for p, r in rows.items():
            inside = sum(links[k][2] for k in r["linked"] if links[k][1] >= links[k][0])
            out = sum(links[k][2] for k in r["linked"] if links[k][1] < links[k][0])
            fmt = lambda t: time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(t)) if t else ""
            f.write("%s\t%d\t%d\t%d\t%d\t%s\t%s\t%d\t%d\t%d\t%.0f\n" % (
                p, r["bytes"], r["single"] + inside, out, r["files"], fmt(r["newest"]), fmt(r["oldest"]),
                r["root_files"], r["other_uid"], r["symlinks"], r["seconds"]))


if __name__ == "__main__":
    main()
