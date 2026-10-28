#!/usr/bin/env python3
"""Pull files of an executed batch back from AOSS to their original paths (the inverse of archive_delete.py delete).

    python disk/restore_from_aoss.py BATCH --match REGEX [--files 2 --threads 2] [--yes]

Reads disk_runs/batches/BATCH/deleted.tsv (utc, item, path, bytes, aoss_key), keeps the rows whose path matches REGEX,
prints the full list and the total, and with --yes downloads each object to PATH.restore_tmp, checks its size against
the record, then renames it to PATH. A path that already exists with the recorded size is skipped; one with another
size stops the run (nothing is overwritten). Writes disk_runs/batches/BATCH/restored_<UTC>.tsv.
The login-node rule (the user, 2026-10-28): at most 4 cores per task, so files x threads <= 4 here; run with the
probe-spatial Python (boto3), proxies unset, under nice -n 19.
"""
import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import archive_delete as ad  # noqa: E402  (client(), BUCKET, U)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--match", required=True, help="regex on the original path")
    ap.add_argument("--files", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    if args.files * args.threads > 4:
        sys.exit("files x threads must be at most 4 on the login node")
    run = os.path.join(ad.U, "disk_runs", "batches", args.batch)
    rows = [r for r in csv.DictReader(open(os.path.join(run, "deleted.tsv")), delimiter="\t") if re.search(args.match, r["path"])]
    total = sum(int(r["bytes"]) for r in rows)
    for r in rows:
        print("  %14s  %s" % (r["bytes"], r["path"]))
    print("== %d files, %.1f GB (%d bytes)" % (len(rows), total / 1e9, total), flush=True)
    for r in rows:
        if os.path.exists(r["path"]) and os.path.getsize(r["path"]) != int(r["bytes"]):
            sys.exit("exists with another size, stop: %s" % r["path"])
    if not args.yes:
        print("dry run: add --yes to download")
        return
    from boto3.s3.transfer import TransferConfig
    cfg = TransferConfig(max_concurrency=args.threads, multipart_chunksize=64 << 20, multipart_threshold=64 << 20)
    c = ad.client()
    t0, done, bad, out = time.time(), [0, 0], [], []

    def one(r):
        p, n = r["path"], int(r["bytes"])
        if os.path.exists(p) and os.path.getsize(p) == n:
            return r, "skipped (already there)"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".restore_tmp"
        c.download_file(ad.BUCKET, r["aoss_key"], tmp, Config=cfg)
        got = os.path.getsize(tmp)
        if got != n:
            os.remove(tmp)
            return r, "SIZE MISMATCH %d != %d" % (got, n)
        os.rename(tmp, p)
        return r, "ok"

    with ThreadPoolExecutor(max_workers=args.files) as ex:
        for r, st in ex.map(one, rows):
            done[0] += 1
            done[1] += int(r["bytes"])
            out.append((time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), r["path"], r["bytes"], r["aoss_key"], st))
            if not st.startswith(("ok", "skipped")):
                bad.append((r["path"], st))
            if done[0] % 10 == 0 or done[0] == len(rows):
                dt = time.time() - t0
                print("  %d/%d files, %.1f GB, %.0f MB/s, %d not ok" % (done[0], len(rows), done[1] / 1e9, done[1] / dt / 1e6, len(bad)), flush=True)
    rec = os.path.join(run, "restored_%s.tsv" % time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    with open(rec, "w") as f:
        f.write("utc\tpath\tbytes\taoss_key\tstatus\n")
        for row in out:
            f.write("\t".join(row) + "\n")
    print("restore done: %d files, %d not ok; record %s" % (len(out), len(bad), rec))
    for p, st in bad:
        print("  NOT OK %s: %s" % (p, st))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
