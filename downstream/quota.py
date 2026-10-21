#!/usr/bin/env python3
"""Free space under the user's folder quota (the quota is invisible to df).

    python downstream/quota.py            # {"quota_tb": 22.0, "used_tb": 12.0, "free_tb_min": 9.5, ...}

Reads `sco afs dir-quota list -i <volume> -d /users/yinbaiqiao` (the method of data_gen/interiorgs_v10/STATUS.md,
"配额事故").  sco prints sizes with two significant digits ("12 TB", "8.6 TB"), so `free_tb_min` subtracts half a
unit of the last printed digit from the free space: a lower bound, used by the evaluation watcher before every
submission.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys

SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
VOLUME = "019ec9f9-6d12-7d49-aad4-864b15c9eb06"
USER_DIR = "/users/yinbaiqiao"
UNITS = {"B": 1e-12, "KB": 1e-9, "MB": 1e-6, "GB": 1e-3, "TB": 1.0, "PB": 1e3}
SIZE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*([kKMGTP]?B)\b")


def _tb(number: str, unit: str):
    """(value in TB, half of the last printed digit in TB)."""
    scale = UNITS[unit.upper()]
    decimals = len(number.split(".")[1]) if "." in number else 0
    return float(number) * scale, 0.5 * 10 ** (-decimals) * scale


def parse(text: str, user_dir: str = USER_DIR):
    """{quota_tb, used_tb, free_tb_min} from the table sco prints, or None if the row is missing."""
    for line in text.splitlines():
        cols = line.split()
        if user_dir not in cols:
            continue
        sizes = SIZE.findall(line[line.index(user_dir) + len(user_dir):])
        if len(sizes) < 2:
            return None
        (quota, _), (used, used_half) = _tb(*sizes[-2]), _tb(*sizes[-1])
        return {"quota_tb": round(quota, 3), "used_tb": round(used, 3),
                "free_tb_min": round(quota - used - used_half, 3), "row": line.strip()}
    return None


def free_space(run=subprocess.run):
    """parse() of a fresh listing; None when sco fails or the row is not there."""
    try:
        res = run([SCO, "afs", "dir-quota", "list", "-i", VOLUME, "-d", USER_DIR], capture_output=True, text=True,
                  timeout=120)
    except Exception:
        return None
    return parse(res.stdout) if res.returncode == 0 else None


if __name__ == "__main__":
    out = free_space()
    print(json.dumps(out))
    sys.exit(0 if out else 1)
