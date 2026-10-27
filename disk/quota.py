#!/usr/bin/env python3
"""Quota, used and remaining space of the user folder /mnt/umm/users/yinbaiqiao (read only).

Three numbers and where each comes from:

  quota      `sco afs dir-quota list -i VOLUME`, the row /users/yinbaiqiao, column CAPACITY QUOTA, taken as printed
             ("22 TB" = 22e12 bytes; the user's decision 4 of 2026-10-26). The rule behind it is probably whole TiB
             (every quota of the volume prints as 1.1 / 5.5 / 11 / 22 TB, i.e. 1 / 5 / 10 / 20 TiB, so the exact
             figure may be 21.99e12); quota_bytes_from_display() gives that reading. --quota-bytes overrides.
  used       quarkfs's own count of the bytes below the folder, the extended attribute
             quarkfs.dir.rfbytes of /mnt/umm/users/yinbaiqiao (exact bytes; the server sums the tree when
             asked, about one to two minutes for 13.7 million files). The quota table's USED CAPACITY
             ("16 TB", two significant digits, so +-0.5 TB) is printed beside it as a check.
             --fast skips the attribute and uses the table's rounded figure.
  remaining  quota - used.

    python3 disk/quota.py            # three lines for people
    python3 disk/quota.py --csv      # one CSV row (header with --header) for docs/disk/usage.csv
    python3 disk/quota.py --json

Needs nothing but the standard library and the sco CLI. Never writes anywhere.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
VOLUME = "019ec9f9-6d12-7d49-aad4-864b15c9eb06"
USER_DIR = "/mnt/umm/users/yinbaiqiao"
QUOTA_ROW = "/users/yinbaiqiao"
TIB = 1024 ** 4
UNITS = {"B": 1, "kB": 1e3, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12, "PB": 1e15,
         "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3, "TiB": TIB}
CSV_FIELDS = ["utc", "quota_bytes", "used_bytes", "remaining_bytes", "used_source", "table_quota", "table_used",
              "table_files", "rfiles", "seconds"]


def parse_size(text):
    m = re.fullmatch(r"([0-9.]+)\s*([A-Za-z]+)", text.strip())
    if not m or m.group(2) not in UNITS:
        raise ValueError("size not understood: %r" % text)
    return float(m.group(1)) * UNITS[m.group(2)]


def quota_table_row(timeout=180):
    """(quota_display, used_display, used_files) of our row, from the full table (the -d form returns a wrong record)."""
    res = subprocess.run([SCO, "afs", "dir-quota", "list", "-i", VOLUME], capture_output=True, text=True, timeout=timeout)
    for line in res.stdout.splitlines():
        cols = re.split(r"\s{2,}", line.strip())
        if len(cols) >= 6 and cols[1] == QUOTA_ROW:
            # ID, DIR PATH, FILE QUANTITY QUOTA, USED FILE QUANTITY, CAPACITY QUOTA, USED CAPACITY, RUNNING STATE
            return cols[4], cols[5], int(cols[3])
    raise RuntimeError("row %s not found in the quota table (rc %s): %s" % (QUOTA_ROW, res.returncode, res.stderr[-300:]))


def quota_bytes_from_display(display):
    """'22 TB' -> 20 TiB: the volume's rules are whole TiB printed in SI units with two significant digits."""
    approx = parse_size(display)
    tib = round(approx / TIB)
    if tib >= 1 and abs(tib * TIB - approx) <= 0.05 * approx:
        return tib * TIB
    return int(approx)


def measure(fast=False, quota_override=None):
    t0 = time.time()
    q_disp, u_disp, files = quota_table_row()
    # the user's decision 4 of 2026-10-26: use the quota as the table prints it (22 TB), not the TiB reading
    quota = int(quota_override) if quota_override else int(parse_size(q_disp))
    rfiles = None
    if fast:
        used, source = int(parse_size(u_disp)), "quota_table_rounded"
    else:
        try:
            used = int(os.getxattr(USER_DIR, "quarkfs.dir.rfbytes"))
            rfiles = int(os.getxattr(USER_DIR, "quarkfs.dir.rfiles"))
            source = "xattr_rfbytes"
        except OSError as e:
            print("xattr failed (%r), using the quota table" % (e,), file=sys.stderr)
            used, source = int(parse_size(u_disp)), "quota_table_rounded"
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), "quota_bytes": quota, "used_bytes": used,
            "remaining_bytes": quota - used, "used_source": source, "table_quota": q_disp, "table_used": u_disp,
            "table_files": files, "rfiles": rfiles if rfiles is not None else "", "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--header", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fast", action="store_true", help="use the quota table's rounded figure for used")
    ap.add_argument("--quota-bytes", type=float, default=None)
    args = ap.parse_args()
    r = measure(args.fast, args.quota_bytes)
    if args.csv:
        if args.header:
            print(",".join(CSV_FIELDS))
        print(",".join(str(r[k]) for k in CSV_FIELDS))
    elif args.json:
        print(json.dumps(r))
    else:
        tb = lambda b: "%.2f TB (%.2f TiB)" % (b / 1e12, b / TIB)
        print("配额 quota    : %s   [quota table %r, taken as printed (decision 4)]" % (tb(r["quota_bytes"]), r["table_quota"]))
        print("已用 used     : %s   [%s; quota table says %r, %s files]" % (tb(r["used_bytes"]), r["used_source"], r["table_used"], r["table_files"]))
        print("剩余 remaining: %s" % tb(r["remaining_bytes"]))
        print("(%s, %.0f s)" % (r["utc"], r["seconds"]))


if __name__ == "__main__":
    main()
