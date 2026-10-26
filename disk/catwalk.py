#!/usr/bin/env python3
"""Sizes by folder and by kind of file, for folders whose content has to be judged run by run (read only).

    python disk/catwalk.py --out FILE.tsv --depth N ROOT [ROOT ...]

For every folder N levels below each ROOT (shallower files count for the ROOT itself) it writes one row per kind:
folder, kind, files, bytes, newest mtime, root-owned files, files with other hard links.
Kinds by file name: checkpoint (.ckpt .pt .pth .bin .safetensors .distcp), feature (.npy .npz .sft .h5 .hdf5 .zarr .pkl),
media (.png .jpg .jpeg .mp4 .gif .webm), table (.csv .tsv .json .jsonl .parquet .yaml .yml), log (.log .txt .out .err),
archive (.tar .gz .zst .zip), other. Symlinks are not followed and not counted.
"""
import argparse
import collections
import os
import stat
import sys
import time

KINDS = [("checkpoint", (".ckpt", ".pt", ".pth", ".bin", ".safetensors", ".distcp")),
         ("feature", (".npy", ".npz", ".sft", ".h5", ".hdf5", ".zarr", ".pkl")),
         ("media", (".png", ".jpg", ".jpeg", ".mp4", ".gif", ".webm")),
         ("table", (".csv", ".tsv", ".json", ".jsonl", ".parquet", ".yaml", ".yml")),
         ("log", (".log", ".txt", ".out", ".err")),
         ("archive", (".tar", ".gz", ".zst", ".zip"))]


def kind_of(name):
    n = name.lower()
    for k, exts in KINDS:
        if n.endswith(exts):
            return k
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()
    agg = collections.defaultdict(lambda: [0, 0, 0.0, 0, 0])  # (folder, kind) -> files, bytes, newest, root, linked
    t0 = time.time()
    n = 0
    for root in args.roots:
        root = os.path.abspath(root)
        stack = [(root, 0)]
        while stack:
            d, dep = stack.pop()
            try:
                it = os.scandir(d)
            except OSError as e:
                print("scandir failed", d, e, file=sys.stderr)
                continue
            key_dir = d if dep <= args.depth else None
            with it:
                for e in it:
                    try:
                        s = e.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if stat.S_ISDIR(s.st_mode):
                        stack.append((e.path, dep + 1))
                        continue
                    if not stat.S_ISREG(s.st_mode):
                        continue
                    # the folder this file is counted under: its ancestor at --depth (or its own folder if shallower)
                    rel = os.path.relpath(d, root)
                    parts = [] if rel == "." else rel.split(os.sep)
                    folder = os.path.join(root, *parts[:args.depth]) if parts else root
                    a = agg[(folder, kind_of(e.name))]
                    a[0] += 1
                    a[1] += s.st_size
                    a[2] = max(a[2], s.st_mtime)
                    a[3] += s.st_uid == 0
                    a[4] += s.st_nlink > 1
                    n += 1
                    if n % 200000 == 0:
                        print("%.0f s: %d files" % (time.time() - t0, n), file=sys.stderr, flush=True)
    with open(args.out, "w") as f:
        f.write("folder\tkind\tfiles\tbytes\tnewest\troot_files\tlinked_files\n")
        for (folder, k), a in sorted(agg.items()):
            f.write("%s\t%s\t%d\t%d\t%s\t%d\t%d\n" % (folder, k, a[0], a[1], time.strftime("%Y-%m-%d", time.gmtime(a[2])), a[3], a[4]))
    print("files %d, %.0f s" % (n, time.time() - t0), file=sys.stderr)


if __name__ == "__main__":
    main()
