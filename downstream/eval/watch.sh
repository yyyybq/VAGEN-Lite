#!/usr/bin/env bash
# Start / stop / check the evaluation watcher (downstream/eval/watch.py) on the login node.  It runs under nohup in its
# own session (setsid), so it survives the ssh session that started it.
#   bash downstream/eval/watch.sh start    # refuses if one is already running
#   bash downstream/eval/watch.sh stop
#   bash downstream/eval/watch.sh status
# Board: downstream_runs/BOARD.md (every 15 min); log: downstream_runs/logs/watch.log; PID: downstream_runs/watch.pid;
# state (finished jobs, parallelism choice, step-3 gate): downstream_runs/WATCH_STATE.json.
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R=/mnt/umm/users/yinbaiqiao/downstream_runs
PIDFILE="$R/watch.pid"
running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
case "${1:-status}" in
  start)
    running && { echo "already running: PID $(cat "$PIDFILE")"; exit 1; }
    cd "$WT"
    setsid nohup env PYTHONDONTWRITEBYTECODE=1 python3 downstream/eval/watch.py --every 900 --board "$R/BOARD.md" \
      >> "$R/logs/watch.log" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    sleep 2
    running && echo "started: PID $(cat "$PIDFILE"), code $(git -C "$WT" rev-parse --short HEAD)" || { echo "failed to start"; tail -5 "$R/logs/watch.log"; exit 1; }
    ;;
  stop)
    running && kill "$(cat "$PIDFILE")" && echo "stopped PID $(cat "$PIDFILE")" || echo "not running"
    rm -f "$PIDFILE"
    ;;
  status)
    running && echo "running: PID $(cat "$PIDFILE")" || echo "not running"
    head -12 "$R/BOARD.md" 2>/dev/null
    ;;
  *) echo "usage: watch.sh start|stop|status"; exit 2 ;;
esac
