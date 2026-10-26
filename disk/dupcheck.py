#!/usr/bin/env python3
"""Which big files are real copies of each other (read only).

Takes the big-file rows of an xtree.tsv ("f" rows, every file >= --min-file), groups them by size, and inside
each group with more than one inode compares a fingerprint: SHA-256 of the first and the last --sample bytes plus
the size. Files with equal fingerprints are counted as copies (a full hash would read terabytes; two different
checkpoints of the same shape differ in their first megabytes, so the sample is enough to tell them apart, but a
match is "very likely the same", not proven). Hard links (same inode) are one file and cost nothing extra.

    python disk/dupcheck.py XTREE.tsv --out DUPS.tsv [--sample 4194304]

DUPS.tsv: one row per set of copies: extra bytes (size x (copies - 1)), size, copies, paths.
"""
import argparse
import collections
import hashlib
import os
import sys
import time


def fingerprint(path, size, sample):
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(sample))
        if size > 2 * sample:
            f.seek(size - sample)
            h.update(f.read(sample))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xtree")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=4 << 20)
    args = ap.parse_args()
    by_size = collections.defaultdict(dict)  # size -> {ino: path}
    with open(args.xtree) as f:
        f.readline()
        for line in f:
            v = line.rstrip("\n").split("\t")
            if v[0] != "f":
                continue
            by_size[int(v[3])].setdefault(int(v[8]), v[2])
    t0 = time.time()
    n_read = 0
    sets = []
    for size, inodes in by_size.items():
        if len(inodes) < 2:
            continue
        groups = collections.defaultdict(list)
        for ino, path in inodes.items():
            try:
                groups[fingerprint(path, size, args.sample)].append(path)
                n_read += 1
            except OSError as e:
                print("unreadable", path, e, file=sys.stderr)
        for fp, paths in groups.items():
            if len(paths) > 1:
                sets.append((size * (len(paths) - 1), size, sorted(paths)))
    sets.sort(key=lambda s: -s[0])
    with open(args.out, "w") as f:
        f.write("extra_bytes\tsize\tcopies\tpaths\n")
        for extra, size, paths in sets:
            f.write("%d\t%d\t%d\t%s\n" % (extra, size, len(paths), " | ".join(paths)))
    print("files read %d, sets of copies %d, extra %.1f GB, %.0f s" % (n_read, len(sets), sum(s[0] for s in sets) / 1e9,
                                                                       time.time() - t0), file=sys.stderr)


if __name__ == "__main__":
    main()
