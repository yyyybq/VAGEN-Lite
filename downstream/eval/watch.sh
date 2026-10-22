#!/usr/bin/env bash
# Start / stop / check the evaluation watcher (downstream/eval/watch.py) on the login node.  It runs under nohup in its
# own session (setsid), so it survives the ssh session that started it.
#   bash downstream/eval/watch.sh start [WATCH.PY OPTIONS...]   # refuses if one is already running
#   bash downstream/eval/watch.sh stop
#   bash downstream/eval/watch.sh status
# e.g. bash downstream/eval/watch.sh start --hold-until-running pt-a pt-b pt-c
# Board: downstream_runs/BOARD.md (every 15 min, placement checks every 2 min); log: downstream_runs/logs/watch.log;
# PID: downstream_runs/watch.pid; options of the running one: downstream_runs/watch.args;
# state (finished jobs, placement checks, step-3 gate, hold): downstream_runs/WATCH_STATE.json.
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R=/mnt/umm/users/yinbaiqiao/downstream_runs
PIDFILE="$R/watch.pid"
running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
ACTION="${1:-status}"; shift || true
case "$ACTION" in
  start)
    running && { echo "already running: PID $(cat "$PIDFILE")"; exit 1; }
    cd "$WT"
    printf '%s\n' "$@" > "$R/watch.args"
    setsid nohup env PYTHONDONTWRITEBYTECODE=1 python3 downstream/eval/watch.py --every 900 --board "$R/BOARD.md" "$@" \
      >> "$R/logs/watch.log" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    sleep 2
    running && echo "started: PID $(cat "$PIDFILE"), code $(git -C "$WT" rev-parse --short HEAD), options: $*" \
      || { echo "failed to start"; tail -5 "$R/logs/watch.log"; exit 1; }
    ;;
  stop)
    running && kill "$(cat "$PIDFILE")" && echo "stopped PID $(cat "$PIDFILE")" || echo "not running"
    rm -f "$PIDFILE"
    ;;
  status)
    running && echo "running: PID $(cat "$PIDFILE"), options: $(tr '\n' ' ' < "$R/watch.args" 2>/dev/null)" || echo "not running"
    head -14 "$R/BOARD.md" 2>/dev/null
    ;;
  *) echo "usage: watch.sh start [options]|stop|status"; exit 2 ;;
esac
