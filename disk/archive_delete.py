#!/usr/bin/env python3
"""Archive to AOSS, then delete, one approved batch of the disk line (the user's iron rule of 2026-10-26).

A batch is a JSON spec (docs/disk/batches/<batch>.json):
  {"batch": "b1_20261026", "approved": "who / when / which list items",
   "items": [{"id": "5.2.1", "root": "/mnt/umm/users/yinbaiqiao/...",          # a folder
              "include": ["regex on the path relative to root", ...],       # optional: only these files
              "exclude": ["regex ...", ...],                                # optional: never these files
              "min_bytes": 0,                                               # optional: skip smaller files
              "marker": "what was deleted and why (optional)",
              "marker_name": "DELETED_TO_AOSS.txt",                        # optional
              "marker_dirs": ["folders that get the marker file"]}]}       # optional, default: [root]

Phases (each re-reads the spec; nothing is deleted before `delete`):
  plan     list every file (no symlinks; files of another owner, root's included, are left and listed), write
           RUN/plan.tsv, print the full list and the total size.
  archive  upload every planned file to fj:s3://baiqiao/vagen_lite_archive/<path relative to /mnt/umm/users/yinbaiqiao>
           (skipped when an object of the same size is there), check every object's size, write RUN/manifest.jsonl;
           copy plan and manifest to AOSS vagen_lite_archive/disk_line_manifests/<batch>/.
  delete   only with --yes: refuse unless every planned file is in the manifest with a matching remote size and is
           unchanged on disk (size and mtime as planned); print the full list and the total again; delete; write
           RUN/deleted.tsv (and copy it to AOSS); write the marker files; remove the folders the batch emptied.
RUN = /mnt/umm/users/yinbaiqiao/disk_runs/batches/<batch>/. Run with the probe-spatial Python (boto3), proxies unset.
"""
import argparse
import configparser
import datetime as dt
import json
import os
import re
import stat
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

U = "/mnt/umm/users/yinbaiqiao"
BUCKET = "baiqiao"
PREFIX = "vagen_lite_archive"
CONF = U + "/aoss/petreloss.conf"
ME = os.getuid()


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def client():
    import boto3
    from botocore.config import Config
    cp = configparser.ConfigParser()
    cp.read(CONF)
    s = cp["fj"]
    return boto3.client("s3", endpoint_url=s["host_base"], aws_access_key_id=s["access_key"],
                        aws_secret_access_key=s["secret_key"],
                        config=Config(retries={"max_attempts": 8}, max_pool_connections=96, s3={"addressing_style": "path"}))


def key_of(path):
    assert path.startswith(U + "/"), path
    return "%s/%s" % (PREFIX, path[len(U) + 1:])


def enumerate_item(item):
    root = item["root"].rstrip("/")
    inc = [re.compile(r) for r in item.get("include", [])]
    exc = [re.compile(r) for r in item.get("exclude", [])]
    min_b = item.get("min_bytes", 0)
    files, left = [], []
    for dp, dns, fns in os.walk(root):
        dns.sort()
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root)
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                continue
            if inc and not any(r.search(rel) for r in inc):
                continue
            if any(r.search(rel) for r in exc) or st.st_size < min_b:
                continue
            row = {"item": item["id"], "path": p, "bytes": st.st_size, "mtime": st.st_mtime, "uid": st.st_uid,
                   "nlink": st.st_nlink}
            (files if st.st_uid == ME else left).append(row)
    return files, left


def load(spec_path):
    spec = json.load(open(spec_path))
    run = os.path.join(U, "disk_runs", "batches", spec["batch"])
    os.makedirs(run, exist_ok=True)
    return spec, run


def read_plan(run):
    rows = []
    with open(os.path.join(run, "plan.tsv")) as f:
        f.readline()
        for line in f:
            item, path, b, mt, uid, nl = line.rstrip("\n").split("\t")
            rows.append({"item": item, "path": path, "bytes": int(b), "mtime": float(mt), "uid": int(uid), "nlink": int(nl)})
    return rows


def show(rows, title):
    print("== %s: %d files, %.1f GB" % (title, len(rows), sum(r["bytes"] for r in rows) / 1e9))
    for r in rows:
        print("%14d  %s" % (r["bytes"], r["path"]))
    by = {}
    for r in rows:
        by.setdefault(r["item"], [0, 0])
        by[r["item"]][0] += 1
        by[r["item"]][1] += r["bytes"]
    for k, (n, b) in by.items():
        print("   item %-8s %6d files %10.1f GB" % (k, n, b / 1e9))
    print("== total %d files, %.1f GB (%d bytes)" % (len(rows), sum(r["bytes"] for r in rows) / 1e9, sum(r["bytes"] for r in rows)),
          flush=True)


def cmd_plan(spec, run, args):
    rows, left = [], []
    for item in spec["items"]:
        f, l = enumerate_item(item)
        rows += f
        left += l
    with open(os.path.join(run, "plan.tsv"), "w") as f:
        f.write("item\tpath\tbytes\tmtime\tuid\tnlink\n")
        for r in rows:
            f.write("%s\t%s\t%d\t%.6f\t%d\t%d\n" % (r["item"], r["path"], r["bytes"], r["mtime"], r["uid"], r["nlink"]))
    with open(os.path.join(run, "left_other_owner.tsv"), "w") as f:
        for r in left:
            f.write("%s\t%s\t%d\t%d\n" % (r["item"], r["path"], r["bytes"], r["uid"]))
    show(rows, "plan of batch %s" % spec["batch"])
    linked = [r for r in rows if r["nlink"] > 1]
    print("files with other hard links (deleting this link frees nothing): %d, %.1f GB" % (len(linked), sum(r["bytes"] for r in linked) / 1e9))
    print("files of another owner, left: %d (left_other_owner.tsv)" % len(left))


def put_meta(c, run, spec, names):
    for n in names:
        p = os.path.join(run, n)
        if os.path.exists(p):
            c.upload_file(p, BUCKET, "%s/disk_line_manifests/%s/%s" % (PREFIX, spec["batch"], n))


def cmd_archive(spec, run, args):
    from boto3.s3.transfer import TransferConfig
    c = client()
    cfg = TransferConfig(max_concurrency=args.threads, multipart_chunksize=64 << 20, multipart_threshold=64 << 20)
    rows = read_plan(run)
    man_path = os.path.join(run, "manifest.jsonl")
    done = {}
    if os.path.exists(man_path):
        for line in open(man_path):
            m = json.loads(line)
            if m.get("ok"):
                done[m["path"]] = m
    todo = [r for r in rows if r["path"] not in done or done[r["path"]]["bytes"] != r["bytes"]]
    print("%d planned, %d already archived, %d to upload (%.1f GB)" % (len(rows), len(rows) - len(todo), len(todo),
                                                                     sum(r["bytes"] for r in todo) / 1e9), flush=True)
    lock = threading.Lock()
    t0, moved = time.time(), [0]

    def remote_size(k):
        try:
            return c.head_object(Bucket=BUCKET, Key=k)["ContentLength"]
        except Exception:
            return None

    def one(r):
        k = key_of(r["path"])
        err = None
        if remote_size(k) != r["bytes"]:
            for attempt in range(3):
                try:
                    c.upload_file(r["path"], BUCKET, k, Config=cfg)
                    err = None
                    break
                except Exception as e:  # retried, then reported
                    err = repr(e)[:300]
                    time.sleep(10 * (attempt + 1))
        got = remote_size(k)
        with lock:
            moved[0] += r["bytes"]
        return {"path": r["path"], "key": k, "bytes": r["bytes"], "remote_bytes": got, "mtime": r["mtime"],
                "ok": err is None and got == r["bytes"], "error": err, "utc": now()}

    bad = 0
    with ThreadPoolExecutor(args.files) as pool, open(man_path, "a") as man:
        futs = [pool.submit(one, r) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            m = f.result()
            man.write(json.dumps(m) + "\n")
            man.flush()
            bad += not m["ok"]
            if i % 25 == 0 or i == len(futs):
                el = time.time() - t0
                print("  %d/%d files, %.1f GB, %.0f MB/s, %d not ok" % (i, len(futs), moved[0] / 1e9, moved[0] / max(el, 1) / 1e6, bad),
                      flush=True)
    put_meta(c, run, spec, ["plan.tsv", "manifest.jsonl", "left_other_owner.tsv"])
    print("archive done: %d not ok; manifest %s (copy on AOSS %s/disk_line_manifests/%s/)" % (bad, man_path, PREFIX, spec["batch"]))


def cmd_delete(spec, run, args):
    rows = read_plan(run)
    man = {}
    for line in open(os.path.join(run, "manifest.jsonl")):
        m = json.loads(line)
        if m.get("ok"):
            man[m["path"]] = m
    problems = []
    for r in rows:
        m = man.get(r["path"])
        if not m or m["remote_bytes"] != r["bytes"]:
            problems.append(("not archived", r["path"]))
            continue
        try:
            st = os.lstat(r["path"])
        except FileNotFoundError:
            problems.append(("already gone", r["path"]))
            continue
        if st.st_size != r["bytes"] or abs(st.st_mtime - r["mtime"]) > 1e-3:
            problems.append(("changed since plan", r["path"]))
    gone = [p for why, p in problems if why == "already gone"]
    hard = [(why, p) for why, p in problems if why != "already gone"]
    if hard:
        for why, p in hard[:50]:
            print("REFUSE", why, p)
        sys.exit("refused: %d problems (nothing deleted)" % len(hard))
    rows = [r for r in rows if r["path"] not in set(gone)]
    show(rows, "TO DELETE, batch %s" % spec["batch"])
    if gone:
        print("already gone before this run (someone else deleted them): %d files" % len(gone))
    if not args.yes:
        print("dry run: add --yes to delete")
        return
    c = client()
    log_path = os.path.join(run, "deleted.tsv")
    touched = set()
    with open(log_path, "a") as log:
        if os.path.getsize(log_path) == 0:
            log.write("utc\titem\tpath\tbytes\taoss_key\n")
        for r in rows:
            os.remove(r["path"])
            log.write("%s\t%s\t%s\t%d\t%s\n" % (now(), r["item"], r["path"], r["bytes"], key_of(r["path"])))
            touched.add(os.path.dirname(r["path"]))
    freed = sum(r["bytes"] for r in rows)
    # remove folders the batch emptied (deepest first, climbing up to but never removing an item root)
    roots = {i["root"].rstrip("/") for i in spec["items"]}
    for d in sorted(touched, key=lambda x: -x.count("/")):
        while d not in roots and d.startswith(U + "/"):
            try:
                os.rmdir(d)
            except OSError:
                break
            d = os.path.dirname(d)
    # marker files: what was deleted and where it is now, in the folders the spec names ("marker_dirs", default: root)
    for it in spec["items"]:
        if not it.get("marker") or not any(r["item"] == it["id"] for r in rows):
            continue
        for d in it.get("marker_dirs", [it["root"]]):
            if os.path.isdir(d):
                with open(os.path.join(d, it.get("marker_name", "DELETED_TO_AOSS.txt")), "w") as f:
                    f.write(it["marker"].rstrip("\n") + "\nAOSS: fj:s3://%s/%s/<same path below /mnt/umm/users/yinbaiqiao>; "
                            "records: %s and fj:s3://%s/%s/disk_line_manifests/%s/\n" % (BUCKET, PREFIX, run, BUCKET, PREFIX, spec["batch"]))
    put_meta(c, run, spec, ["deleted.tsv"])
    print("deleted %d files, %.1f GB; record %s (copy on AOSS)" % (len(rows), freed / 1e9, log_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["plan", "archive", "delete"])
    ap.add_argument("spec")
    ap.add_argument("--files", type=int, default=8, help="files uploaded at once")
    ap.add_argument("--threads", type=int, default=8, help="multipart threads per file")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(k, None)
    spec, run = load(args.spec)
    {"plan": cmd_plan, "archive": cmd_archive, "delete": cmd_delete}[args.phase](spec, run, args)


if __name__ == "__main__":
    main()
