#!/usr/bin/env python3
"""Directory sizes from quarkfs's recursive xattrs, without walking every file.

quarkfs (the FUSE file system behind /mnt/umm) answers getxattr on a directory
with counts the server keeps for the whole subtree:

    quarkfs.dir.rfbytes   bytes of all files below the directory (recursive)
    quarkfs.dir.rfiles    number of files below it (recursive)
    quarkfs.dir.fbytes    bytes of the files directly in it
    quarkfs.dir.files     number of files directly in it

The recursive ones are computed when asked (about 225k files a second, measured
2026-10-26), the direct ones are instant. This walks the tree from ROOT, reads
rfbytes / rfiles for every directory it visits, descends only into directories
of at least --min-bytes and at most --max-depth levels down, and stats the
files of a directory only when its direct bytes reach --min-file (so every file
of at least --min-file is listed). Symlinks are recorded, never followed.

Output (TSV, one row per directory visited, per big file and per symlink):
    kind depth path bytes files mtime uid nlink ino target
kind is d (directory), f (file >= --min-file), l (symlink).

Read only: it opens nothing for writing except --out.
"""
import argparse
import os
import sys
import time


def xattr_int(path, name):
    try:
        return int(os.getxattr(path, "quarkfs.dir." + name, follow_symlinks=False).decode())
    except OSError:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-bytes", type=float, default=50e9, help="descend into directories at least this big")
    ap.add_argument("--min-file", type=float, default=1e9, help="list files at least this big")
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--skip", action="append", default=[], help="path prefix not to descend into (still sized)")
    ap.add_argument("--files-depth", type=int, default=2,
                    help="read the recursive file count only down to this depth (each recursive read costs the server "
                         "a walk of the subtree; one read per directory below it)")
    args = ap.parse_args()

    t0 = time.time()
    n_dirs = 0
    with open(args.out + ".tmp", "w") as out:
        out.write("kind\tdepth\tpath\tbytes\tfiles\tmtime\tuid\tnlink\tino\ttarget\n")
        stack = [(os.path.abspath(args.root), 0)]
        while stack:
            path, depth = stack.pop()
            try:
                st = os.lstat(path)
            except OSError as e:
                print("lstat failed", path, e, file=sys.stderr)
                continue
            rb = xattr_int(path, "rfbytes")
            rf = xattr_int(path, "rfiles") if depth <= args.files_depth else -1
            out.write("d\t%d\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t\n" % (depth, path, rb, rf, int(st.st_mtime), st.st_uid, st.st_nlink, st.st_ino))
            n_dirs += 1
            if n_dirs % 200 == 0:
                print("%.0f s: %d directories, at %s" % (time.time() - t0, n_dirs, path), file=sys.stderr, flush=True)
            if depth >= args.max_depth or rb < args.min_bytes:
                continue
            if any(path.startswith(s) for s in args.skip):
                continue
            fb = xattr_int(path, "fbytes")
            try:
                entries = list(os.scandir(path))
            except OSError as e:
                print("scandir failed", path, e, file=sys.stderr)
                continue
            for e in entries:
                try:
                    if e.is_symlink():
                        try:
                            tgt = os.readlink(e.path)
                        except OSError:
                            tgt = "?"
                        lst = e.stat(follow_symlinks=False)
                        out.write("l\t%d\t%s\t0\t0\t%d\t%d\t%d\t%d\t%s\n" % (depth + 1, e.path, int(lst.st_mtime), lst.st_uid, lst.st_nlink, lst.st_ino, tgt))
                    elif e.is_dir(follow_symlinks=False):
                        stack.append((e.path, depth + 1))
                    elif fb >= args.min_file:
                        fst = e.stat(follow_symlinks=False)
                        if fst.st_size >= args.min_file:
                            out.write("f\t%d\t%s\t%d\t1\t%d\t%d\t%d\t%d\t\n" % (depth + 1, e.path, fst.st_size, int(fst.st_mtime), fst.st_uid, fst.st_nlink, fst.st_ino))
                except OSError as err:
                    print("entry failed", e.path, err, file=sys.stderr)
            out.flush()
    os.replace(args.out + ".tmp", args.out)
    print("dirs %d, %.1f s" % (n_dirs, time.time() - t0), file=sys.stderr)


if __name__ == "__main__":
    main()
