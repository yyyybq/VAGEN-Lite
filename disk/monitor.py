#!/usr/bin/env python3
"""Resident disk monitor of the user folder (read only except for its own outputs).

Every --every seconds (default 1800):
  - reads the quota table row (sco afs dir-quota list) and quarkfs's recursive byte count of the user folder
    (one server-side query, about 2 minutes; --no-xattr skips it);
  - appends a row to docs/disk/usage.csv: utc, quota_TB, table_used_TB, remaining_TB, table_files, rfbytes_TB,
    level, seconds;
  - remaining = 22 TB - the quota table's reading (the user's decision of 2026-10-26: "按22T正常用");
  - level 0: remaining >= 3 TB; level 1: below 3 TB; level 2: below 1.5 TB. On entering level 1 or 2 it writes the
    warning file disk_runs/WARN_DISK_LOW (what, when, numbers) and an event; when the level falls back to 0 it
    removes that file (its own) and writes an event. With --stop-flag (approved by the user on 2026-10-26) it also
    writes disk_runs/flags/STOP_SUBMIT_DISK_LOW below 1.5 TB remaining and removes it at 3 TB or more; the lines'
    submission code checks that file (docs/disk/STOP_FLAG_PROPOSAL.md).
Once a day (the first pass after --daily-hour UTC, and at start-up if there is no snapshot yet) it reads the
recursive byte count of every folder directly under the user folder, one at a time, appends them to
docs/disk/daily_top.csv and rewrites docs/disk/daily_growth.md (growth per folder and per line since the previous
snapshot, largest first).

Events (one JSON object per line) go to disk_runs/monitor_events.jsonl; the heartbeat to disk_runs/monitor_heartbeat.
Start / stop / status: bash disk/monitor.sh start|stop|status.
"""
import argparse
import csv
import datetime as dt
import json
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import classify  # noqa: E402  (line rules)
import quota  # noqa: E402

U = "/mnt/umm/users/yinbaiqiao"
RUNS = U + "/disk_runs"
QUOTA_TB = 22.0
LEVELS = [(1.5, 2), (3.0, 1)]
STOP = False


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def event(kind, **kw):
    rec = {"utc": utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "kind": kind, **kw}
    with open(os.path.join(RUNS, "monitor_events.jsonl"), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def rfbytes(path):
    return int(os.getxattr(path, "quarkfs.dir.rfbytes"))


def level_of(remaining):
    for lim, lv in LEVELS:
        if remaining < lim:
            return lv
    return 0


def one_pass(args, state):
    t0 = time.time()
    try:
        q_disp, u_disp, files = quota.quota_table_row()
    except Exception as e:  # sco unreachable: say so, try again next pass
        event("error", what="quota table", error=repr(e)[:300])
        return
    used_tb = quota.parse_size(u_disp) / 1e12
    remaining = QUOTA_TB - used_tb
    rb = ""
    if not args.no_xattr:
        try:
            rb = "%.3f" % (rfbytes(U) / 1e12)
        except OSError as e:
            event("error", what="rfbytes", error=repr(e)[:300])
    lv = level_of(remaining)
    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "quota_TB", "table_used_TB", "remaining_TB", "table_files", "rfbytes_TB", "level", "seconds"])
        w.writerow([utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), QUOTA_TB, "%.1f" % used_tb, "%.1f" % remaining, files, rb, lv,
                    "%.0f" % (time.time() - t0)])
    prev = state.get("level", 0)
    warn = os.path.join(RUNS, "WARN_DISK_LOW")
    if lv > prev:
        text = ("剩余 %.1f TB（配额 %.0f TB，配额表已用 %s，%s UTC），低于 %s TB\n" %
                (remaining, QUOTA_TB, u_disp, utcnow().strftime("%m-%d %H:%M"), "1.5" if lv == 2 else "3"))
        with open(warn, "w") as f:
            f.write(text)
        event("warn", level=lv, remaining_TB=round(remaining, 1), table_used=u_disp, rfbytes_TB=rb)
    elif lv < prev:
        event("level_down", level=lv, remaining_TB=round(remaining, 1), table_used=u_disp)
        if lv == 0 and os.path.exists(warn):
            os.remove(warn)  # the monitor's own warning file
    if args.stop_flag:  # the user's approval of 2026-10-26 (STOP_FLAG_PROPOSAL.md): set below 1.5 TB, clear at 3 TB
        flag = os.path.join(RUNS, "flags", "STOP_SUBMIT_DISK_LOW")
        if remaining < 1.5 and not os.path.exists(flag):
            os.makedirs(os.path.dirname(flag), exist_ok=True)
            with open(flag, "w") as f:
                f.write(json.dumps({"utc": utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "table_used": u_disp,
                                    "remaining_TB": round(remaining, 1), "set_below_TB": 1.5, "cleared_at_TB": 3.0,
                                    "by": "disk line monitor (disk/monitor.py)"}) + "\n")
            event("flag_set", remaining_TB=round(remaining, 1), table_used=u_disp)
        elif remaining >= 3.0 and os.path.exists(flag):
            os.remove(flag)  # the monitor's own flag
            event("flag_cleared", remaining_TB=round(remaining, 1), table_used=u_disp)
    if u_disp != state.get("table_used"):
        event("table_change", old=state.get("table_used"), new=u_disp, rfbytes_TB=rb)
    state.update(level=lv, table_used=u_disp, last_pass=utcnow().isoformat())


def daily(args, state):
    snap = {}
    t0 = time.time()
    for e in sorted(os.scandir(U), key=lambda e: e.name):
        if STOP:
            return
        if e.is_dir(follow_symlinks=False):
            try:
                snap[e.name] = rfbytes(e.path)
            except OSError:
                snap[e.name] = -1
    stamp = utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    new = not os.path.exists(args.daily_csv)
    with open(args.daily_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "folder", "bytes"])
        for k, v in snap.items():
            w.writerow([stamp, k, v])
    prev, prev_t = state.get("daily_snapshot"), state.get("daily_utc")
    if prev:
        hours = max((utcnow() - dt.datetime.fromisoformat(prev_t)).total_seconds() / 3600, 1e-6)
        rows = []
        for k in set(snap) | set(prev):
            a, b = prev.get(k, 0), snap.get(k, 0)
            rows.append((b - a, k, b))
        rows.sort(key=lambda r: -abs(r[0]))
        per_line = {}
        for d, k, b in rows:
            line = classify.line_of(U + "/" + k)
            per_line[line] = per_line.get(line, 0) + d
        with open(args.daily_md, "w") as f:
            f.write("# 个人目录每日增量\n\n本次 %s，上次 %s，相隔 %.1f 小时。按文件系统统计（rfbytes），只列变化超过 1 GB 的一级目录。\n\n"
                    % (stamp, prev_t[:16].replace("T", " "), hours))
            f.write("| 线 | 变化（TB） | 折合每天（TB） |\n| --- | ---: | ---: |\n")
            for line, d in sorted(per_line.items(), key=lambda x: -x[1]):
                if abs(d) >= 1e9:
                    f.write("| %s | %+.3f | %+.3f |\n" % (line, d / 1e12, d / 1e12 * 24 / hours))
            f.write("\n| 一级目录 | 现在（TB） | 变化（TB） |\n| --- | ---: | ---: |\n")
            for d, k, b in rows:
                if abs(d) >= 1e9:
                    f.write("| `%s` | %.3f | %+.3f |\n" % (k, b / 1e12, d / 1e12))
        top = max(per_line.items(), key=lambda x: x[1])
        event("daily", hours=round(hours, 1), fastest_line=top[0], fastest_TB=round(top[1] / 1e12, 3),
              seconds=round(time.time() - t0))
    else:
        event("daily", note="first snapshot", seconds=round(time.time() - t0))
    state.update(daily_snapshot=snap, daily_utc=utcnow().isoformat())


def main():
    global STOP
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=1800)
    ap.add_argument("--no-xattr", action="store_true")
    ap.add_argument("--daily-hour", type=int, default=6)
    ap.add_argument("--stop-flag", action="store_true",
                    help="write disk_runs/flags/STOP_SUBMIT_DISK_LOW below 1.5 TB remaining, remove it at 3 TB or more")
    ap.add_argument("--csv", default=os.path.join(HERE, "..", "docs", "disk", "usage.csv"))
    ap.add_argument("--daily-csv", default=os.path.join(HERE, "..", "docs", "disk", "daily_top.csv"))
    ap.add_argument("--daily-md", default=os.path.join(HERE, "..", "docs", "disk", "daily_growth.md"))
    args = ap.parse_args()
    for p in ("csv", "daily_csv", "daily_md"):
        setattr(args, p, os.path.abspath(getattr(args, p)))
    os.makedirs(RUNS, exist_ok=True)
    state_path = os.path.join(RUNS, "monitor_state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}

    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    with open(os.path.join(RUNS, "monitor.pid"), "w") as f:  # monitor.sh reads it (setsid may fork, so $! is not this pid)
        f.write("%d\n" % os.getpid())
    event("start", pid=os.getpid(), every=args.every)
    while not STOP:
        one_pass(args, state)
        now = utcnow()
        last = state.get("daily_utc")
        due = last is None or (now - dt.datetime.fromisoformat(last)).total_seconds() >= 23 * 3600 and now.hour >= args.daily_hour
        if due and not STOP:
            daily(args, state)
        with open(state_path + ".tmp", "w") as f:
            json.dump(state, f)
        os.replace(state_path + ".tmp", state_path)
        with open(os.path.join(RUNS, "monitor_heartbeat"), "w") as f:
            f.write(utcnow().strftime("%Y-%m-%dT%H:%M:%SZ\n"))
        for _ in range(args.every):
            if STOP:
                break
            time.sleep(1)
    event("stop", pid=os.getpid())


if __name__ == "__main__":
    main()
