#!/usr/bin/env python3
"""AOSS side of the disk line: what the bucket reports about itself, whether the archive manifests are complete,
and how fast uploads and downloads are from the login node.

Run with the probe-spatial Python (boto3) and without proxy variables (AOSS hangs through the login-node proxy):

    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
        /mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python -I disk/aoss_check.py bucket
    ... aoss_check.py prefixes [PREFIX]            # top-level (or PREFIX/) folders: object count and bytes
    ... aoss_check.py manifest FILE.jsonl          # every {"path","uri","bytes","ok"} row: object there, same size?
    ... aoss_check.py ledger FILE.jsonl            # probe_spatial ledgers: rows with "key" and "bytes"/"size"
    ... aoss_check.py download KEY                 # one object to /dev/null (nothing written locally)
    ... aoss_check.py upload LOCAL_FILE KEY        # one local file to KEY (creates that object)

Credentials are read from petreloss.conf section [fj] inside the process and never printed.
Everything except `upload` only reads.
"""
import configparser
import json
import sys
import time

CONF = "/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf"
BUCKET = "baiqiao"


def client():
    import boto3
    from botocore.config import Config
    cp = configparser.ConfigParser()
    cp.read(CONF)
    sec = cp["fj"]
    return boto3.client("s3", endpoint_url=sec["host_base"], aws_access_key_id=sec["access_key"],
                        aws_secret_access_key=sec["secret_key"],
                        config=Config(retries={"max_attempts": 5}, max_pool_connections=32, s3={"addressing_style": "path"}))


def cmd_bucket(c):
    r = c.head_bucket(Bucket=BUCKET)
    hdr = r.get("ResponseMetadata", {}).get("HTTPHeaders", {})
    safe = {k: v for k, v in hdr.items() if not any(s in k.lower() for s in ("auth", "token", "secret", "key"))}
    print(json.dumps({"head_bucket_headers": safe}, indent=1))
    for name in ("get_bucket_location", "get_bucket_versioning", "get_bucket_lifecycle_configuration"):
        try:
            out = getattr(c, name)(Bucket=BUCKET)
            out.pop("ResponseMetadata", None)
            print(name, json.dumps(out, default=str))
        except Exception as e:  # not supported or not set
            print(name, "->", type(e).__name__, str(e)[:200])


def list_all(c, prefix, delimiter=None):
    """Every page under PREFIX. AOSS answers list_objects_v2 with IsTruncated but no NextContinuationToken, so boto3's
    paginator stops after 1,000 keys; the v1 call with Marker / NextMarker pages correctly (checked 2026-10-26)."""
    kw = {"Bucket": BUCKET, "Prefix": prefix}
    if delimiter:
        kw["Delimiter"] = delimiter
    marker = ""
    while True:
        page = c.list_objects(Marker=marker, **kw)
        yield page
        if not page.get("IsTruncated"):
            return
        nxt = page.get("NextMarker") or max([o["Key"] for o in page.get("Contents", [])] +
                                            [p["Prefix"] for p in page.get("CommonPrefixes", [])])
        if nxt <= marker:
            raise RuntimeError("listing does not advance at %r" % nxt)
        marker = nxt


def cmd_prefixes(c, prefix=""):
    t0 = time.time()
    tops = []
    for page in list_all(c, prefix, "/"):
        tops += [p["Prefix"] for p in page.get("CommonPrefixes", [])]
        for o in page.get("Contents", []):
            print("object\t%s\t%d" % (o["Key"], o["Size"]))
    print("%d folders under %r (%.1f s)" % (len(tops), prefix, time.time() - t0), flush=True)
    for p in tops:
        n = b = 0
        t = time.time()
        for page in list_all(c, p):
            for o in page.get("Contents", []):
                n += 1
                b += o["Size"]
        print("folder\t%s\t%d objects\t%d bytes\t%.2f TB\t%.0f s" % (p, n, b, b / 1e12, time.time() - t), flush=True)


def remote_sizes(c, prefix):
    out = {}
    for page in list_all(c, prefix):
        for o in page.get("Contents", []):
            out[o["Key"]] = o["Size"]
    return out


def common_prefix(keys):
    keys = list(keys)
    if not keys:
        return ""
    s1, s2 = min(keys), max(keys)
    i = 0
    while i < min(len(s1), len(s2)) and s1[i] == s2[i]:
        i += 1
    return s1[:i].rsplit("/", 1)[0] + "/" if "/" in s1[:i] else ""


def check_rows(c, rows):
    """rows: list of (key, bytes). Lists the common prefix once and compares."""
    pre = common_prefix(k for k, _ in rows)
    t = time.time()
    remote = remote_sizes(c, pre)
    missing = [k for k, b in rows if k not in remote]
    wrong = [(k, b, remote[k]) for k, b in rows if k in remote and remote[k] != b]
    tot = sum(b for _, b in rows)
    print(json.dumps({"rows": len(rows), "bytes": tot, "listed_prefix": pre, "remote_objects_under_prefix": len(remote),
                      "missing": len(missing), "size_mismatch": len(wrong), "examples_missing": missing[:5],
                      "examples_mismatch": wrong[:5], "list_seconds": round(time.time() - t, 1)}, indent=1))


def cmd_manifest(c, path):
    rows = []
    bad_local = 0
    for line in open(path):
        r = json.loads(line)
        if not r.get("ok"):
            bad_local += 1
            continue
        uri = r["uri"]
        assert uri.startswith("fj:s3://%s/" % BUCKET), uri
        rows.append((uri[len("fj:s3://%s/" % BUCKET):], int(r["bytes"])))
    print("rows marked not ok in the manifest:", bad_local)
    check_rows(c, rows)


def cmd_ledger(c, path):
    rows, actions = [], {}
    for line in open(path):
        r = json.loads(line)
        a = r.get("action", "?")
        actions[a] = actions.get(a, 0) + 1
        key = r.get("key")
        size = r.get("bytes", r.get("size"))
        if key and size is not None and a in ("deleted", "archived", "uploaded", "verified"):
            rows.append((key, int(size)))
    print("actions:", actions)
    if rows:
        check_rows(c, rows)


class Null:
    def __init__(self):
        self.n = 0

    def write(self, b):
        self.n += len(b)
        return len(b)


def cmd_download(c, key):
    from boto3.s3.transfer import TransferConfig
    size = c.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    for label, cfg in (("1 stream", TransferConfig(use_threads=False, multipart_threshold=1 << 40)),
                       ("10 threads x 8 MiB", TransferConfig(max_concurrency=10, multipart_chunksize=8 << 20)),
                       ("32 threads x 64 MiB", TransferConfig(max_concurrency=32, multipart_chunksize=64 << 20))):
        sink = Null()
        t = time.time()
        c.download_fileobj(BUCKET, key, sink, Config=cfg)
        dt = time.time() - t
        print("download %s: %d bytes in %.1f s = %.0f MB/s" % (label, sink.n, dt, sink.n / dt / 1e6), flush=True)
    assert sink.n == size


def cmd_upload(c, local, key):
    import os
    from boto3.s3.transfer import TransferConfig
    size = os.path.getsize(local)
    for label, cfg in (("10 threads x 8 MiB (petreloss.conf)", TransferConfig(max_concurrency=10, multipart_chunksize=8 << 20)),
                       ("32 threads x 64 MiB", TransferConfig(max_concurrency=32, multipart_chunksize=64 << 20))):
        t = time.time()
        c.upload_file(local, BUCKET, key, Config=cfg)
        dt = time.time() - t
        got = c.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
        print("upload %s: %d bytes in %.1f s = %.0f MB/s, remote size %s" % (label, size, dt, size / dt / 1e6,
                                                                           "matches" if got == size else got), flush=True)


def main():
    c = client()
    cmd, args = sys.argv[1], sys.argv[2:]
    {"bucket": cmd_bucket, "prefixes": cmd_prefixes, "manifest": cmd_manifest, "ledger": cmd_ledger,
     "download": cmd_download, "upload": cmd_upload}[cmd](c, *args)


if __name__ == "__main__":
    main()
