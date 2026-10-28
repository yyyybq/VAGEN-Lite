#!/usr/bin/env bash
# Start / stop / status of the resident disk monitor (disk/monitor.py) on the login node.
#   bash disk/monitor.sh start [monitor.py options]   # detached (setsid nohup nice -n 19, one thread: the login node has 14 cores for the account; the user's rule of 2026-10-28); log disk_runs/logs/monitor.log
#   bash disk/monitor.sh stop                         # SIGTERM; it finishes the current query, then exits
#   bash disk/monitor.sh status                       # pid, heartbeat, last row of usage.csv, warning file
set -u
U=/mnt/umm/users/yinbaiqiao
PY=$U/.conda/envs/vagen-lite/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=$U/disk_runs
PIDF=$RUNS/monitor.pid
mkdir -p "$RUNS/logs"
running() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
case "${1:-status}" in
  start)
    shift
    if running; then echo "already running: pid $(cat "$PIDF")"; exit 0; fi
    (cd "$HERE/.." && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false setsid nohup nice -n 19 "$PY" -u "$HERE/monitor.py" "$@" >> "$RUNS/logs/monitor.log" 2>&1 < /dev/null &)
    old=$(cat "$PIDF" 2>/dev/null); for i in $(seq 1 15); do sleep 1; [ "$(cat "$PIDF" 2>/dev/null)" != "$old" ] && running && break; done
    running && echo "started: pid $(cat "$PIDF")" || { echo "failed to start, see $RUNS/logs/monitor.log"; exit 1; } ;;
  stop)
    if running; then kill "$(cat "$PIDF")" && echo "stop sent to pid $(cat "$PIDF")"; else echo "not running"; fi ;;
  status)
    if running; then echo "running: pid $(cat "$PIDF")"; else echo "NOT running"; fi
    echo "heartbeat: $(cat "$RUNS/monitor_heartbeat" 2>/dev/null)"
    echo "last row:  $(tail -1 "$HERE/../docs/disk/usage.csv" 2>/dev/null)"
    [ -f "$RUNS/WARN_DISK_LOW" ] && echo "WARNING: $(cat "$RUNS/WARN_DISK_LOW")" || echo "no warning file" ;;
  *) echo "usage: $0 start|stop|status"; exit 2 ;;
esac
