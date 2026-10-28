#!/usr/bin/env bash
# Start / stop / check the evaluation watcher (downstream/eval/watch.py) on the login node.  It runs under nohup in its
# own session (setsid), so it survives the ssh session that started it.
#   bash downstream/eval/watch.sh start WATCH.PY OPTIONS...      # refuses if one is already running or no option given
#   bash downstream/eval/watch.sh restart                       # stop, then start again with the saved options
#   bash downstream/eval/watch.sh stop
#   bash downstream/eval/watch.sh status
# e.g. bash downstream/eval/watch.sh start --hold-until-running pt-a pt-b pt-c
# Board: downstream_runs/BOARD.md (every 15 min, placement checks every 2 min); log: downstream_runs/logs/watch.log;
# PID: downstream_runs/watch.pid; options of the running one: downstream_runs/watch.args;
# state (finished jobs, placement checks, step-3 gate, hold): downstream_runs/WATCH_STATE.json.
# Login-node rule (the user's, 2026-10-28, ~/.claude/CLAUDE.md): the account has 14 cores there for every window, so
# the watcher and everything it starts (sco, the report build) run with one thread per process and at nice 19.
# `start` without options is refused: on 10-28 it emptied watch.args and the watcher ran two minutes without its
# options; `restart` reuses the saved ones.  (WATCH_RUNS_ROOT only changes for the tests.)
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R="${WATCH_RUNS_ROOT:-/mnt/umm/users/yinbaiqiao/downstream_runs}"
PIDFILE="$R/watch.pid"
running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
start() {
  running && { echo "already running: PID $(cat "$PIDFILE")"; return 1; }
  (( $# > 0 )) || { echo "start needs the watcher's options (restart reuses the saved ones): not started"; return 2; }
  cd "$WT" || return 1
  mkdir -p "$R/logs"
  printf '%s\n' "$@" > "$R/watch.args"
  setsid nohup nice -n 19 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    python3 downstream/eval/watch.py --every 900 --board "$R/BOARD.md" "$@" >> "$R/logs/watch.log" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  sleep 2
  running && echo "started: PID $(cat "$PIDFILE"), code $(git -C "$WT" rev-parse --short HEAD 2>/dev/null), options: $*" \
    || { echo "failed to start"; tail -5 "$R/logs/watch.log"; return 1; }
}
stop() {
  running && kill "$(cat "$PIDFILE")" && echo "stopped PID $(cat "$PIDFILE")" || echo "not running"
  rm -f "$PIDFILE"
}
ACTION="${1:-status}"; shift || true
case "$ACTION" in
  start) start "$@"; exit $? ;;
  restart)
    saved=()
    [[ -s "$R/watch.args" ]] && mapfile -t saved < "$R/watch.args"
    (( ${#saved[@]} > 0 )) || { echo "no saved options in $R/watch.args: use start with the options"; exit 2; }
    stop
    sleep 1
    start "${saved[@]}"; exit $?
    ;;
  stop) stop ;;
  status)
    running && echo "running: PID $(cat "$PIDFILE"), options: $(tr '\n' ' ' < "$R/watch.args" 2>/dev/null)" || echo "not running"
    head -14 "$R/BOARD.md" 2>/dev/null
    ;;
  *) echo "usage: watch.sh start OPTIONS...|restart|stop|status"; exit 2 ;;
esac
