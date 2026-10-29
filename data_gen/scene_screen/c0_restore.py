"""Restore C0's step-300 HF checkpoints (round 12's r12a pair, archived to AOSS on 2026-10-21) into this branch's exps,
for the user's evaluation of C0 on the 59 new test tasks (2026-10-29).  Only the actor/huggingface files; every file's
size is checked against the archive manifest.  Run with the probe-spatial Python, without proxies."""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(k, None)
from aoss_client.client import Client  # noqa: E402

MANIFEST = Path("/mnt/umm/users/yinbaiqiao/aoss_archive_manifests/manifest_20261021T055911Z.jsonl")
OUT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data/exps/c0_restore_20261029")
CONF = "/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf"
client = Client(CONF)
rows = [json.loads(l) for l in open(MANIFEST)]
todo = []
for run in ("rl_run_p1s0", "rl_run_p1d"):
    pre = f"/interiorgs_v10_train_r12a_20261019/{run}/checkpoints/global_step_300/actor/huggingface/"
    for r in rows:
        if pre in r["path"]:
            assert r["ok"], r
            todo.append((r["uri"], OUT / run / "global_step_300/actor/huggingface" / r["path"].split(pre)[1], r["bytes"], r["path"]))
print(f"{len(todo)} files, {sum(t[2] for t in todo) / 1e9:.1f} GB", flush=True)
t0 = time.time()


def one(item):
    uri, dst, size, src = item
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == size:
        return dst, size, "already there"
    for attempt in range(3):
        try:
            client.download_file(uri, str(dst))
            break
        except Exception as exc:  # retried, then reported
            err = repr(exc)
            time.sleep(5 * (attempt + 1))
    else:
        return dst, size, f"download failed: {err}"
    got = dst.stat().st_size if dst.exists() else None
    return dst, size, None if got == size else f"size mismatch: archive {size}, local {got}"


log = []
with ThreadPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
    for f in as_completed([ex.submit(one, t) for t in todo]):
        dst, size, err = f.result()
        log.append({"path": str(dst), "bytes": size, "error": err})
        print(f"{time.time() - t0:6.0f}s {dst.relative_to(OUT)} {size / 1e9:.2f} GB {err or 'ok'}", flush=True)
bad = [x for x in log if x["error"] not in (None, "already there")]
(OUT / "restore_log.json").write_text(json.dumps({"manifest": str(MANIFEST), "files": len(log), "bad": bad,
                                                  "seconds": round(time.time() - t0), "log": log}, indent=1))
print("done", len(log), "files,", len(bad), "bad,", round(time.time() - t0), "s", flush=True)
