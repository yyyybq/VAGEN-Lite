#!/usr/bin/env bash
# Step-4 GPU sampler: records nvidia-smi every INTERVAL ms while a measured run executes; the caller starts and
# stops it. It only QUERIES nvidia-smi; it never touches GPU settings or other processes.
#
#   bash scripts/gpu_sampler.sh start RUN_DIR [INTERVAL_MS]   # default 2000 ms
#   bash scripts/gpu_sampler.sh stop RUN_DIR                  # prints the peak memory per GPU
#   bash scripts/gpu_sampler.sh status RUN_DIR
#
# Files in RUN_DIR (scripts/measure_throughput.py reads them):
#   gpu_samples.csv   timestamp,index,uuid,memory_used_mib,memory_total_mib,utilization_pct
#   gpu_apps.csv      timestamp,pid,gpu_uuid,used_memory_mib   (compute processes, every user)
#   gpu_sampler.pid   "pid start_ticks" of the two nvidia-smi loops this script started
#   gpu_sampler.err   nvidia-smi stderr
#
# `stop` signals ONLY the processes listed in gpu_sampler.pid, and only after checking that each one still is
# the same process (same /proc/<pid>/stat start time), belongs to the current user, and is nvidia-smi. Anything
# else is reported and left alone (CLAUDE.md "GPU discipline": never signal other users' processes).
#
# Status: written and syntax-checked (bash -n, tests/test_ops_shell.py); not yet run on 105.
set -u

usage() { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

start_ticks() { sed 's/^.*) //' "/proc/$1/stat" 2>/dev/null | awk '{print $20}'; }

is_our_sampler() {  # pid ticks
    local pid=$1 ticks=$2
    [ -r "/proc/$pid/stat" ] || return 1
    [ "$(stat -c %u "/proc/$pid" 2>/dev/null)" = "$(id -u)" ] || return 1
    [ "$(start_ticks "$pid")" = "$ticks" ] || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -- "nvidia-smi" || return 1
}

peaks() {  # per-GPU peak / first memory and peak utilization from gpu_samples.csv
    awk -F', *' 'NR > 1 && $2 ~ /^[0-9]+$/ {
        i = $2; n[i]++; if (!(i in first)) first[i] = $4
        if ($4 + 0 > peak[i] + 0) peak[i] = $4; if ($6 + 0 > util[i] + 0) util[i] = $6; tot[i] = $5
    } END {
        for (i in n) printf "GPU%s samples %d first %s MiB peak %s MiB (total %s) peak util %s%%\n", i, n[i], first[i], peak[i], tot[i], util[i]
    }' "$1"
}

cmd=${1:-}
dir=${2:-}
[ -n "$cmd" ] && [ -n "$dir" ] || usage
pidfile="$dir/gpu_sampler.pid"

case "$cmd" in
start)
    interval=${3:-2000}
    command -v nvidia-smi >/dev/null 2>&1 || { echo "gpu_sampler: nvidia-smi not found (not on server 105?)" >&2; exit 1; }
    mkdir -p "$dir" || exit 1
    if [ -f "$pidfile" ]; then
        while read -r pid ticks; do
            if is_our_sampler "$pid" "$ticks"; then echo "gpu_sampler: already running (pid $pid) for $dir" >&2; exit 1; fi
        done < "$pidfile"
    fi
    echo "timestamp,index,uuid,memory_used_mib,memory_total_mib,utilization_pct" > "$dir/gpu_samples.csv"
    echo "timestamp,pid,gpu_uuid,used_memory_mib" > "$dir/gpu_apps.csv"
    nohup nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits -lms "$interval" >> "$dir/gpu_samples.csv" 2>> "$dir/gpu_sampler.err" < /dev/null &
    p1=$!
    nohup nvidia-smi --query-compute-apps=timestamp,pid,gpu_uuid,used_memory \
        --format=csv,noheader,nounits -lms "$interval" >> "$dir/gpu_apps.csv" 2>> "$dir/gpu_sampler.err" < /dev/null &
    p2=$!
    sleep 1  # long enough for nvidia-smi to reject a query field it does not support
    : > "$pidfile"
    alive=0
    for p in "$p1" "$p2"; do
        t=$(start_ticks "$p")
        if [ -n "$t" ] && is_our_sampler "$p" "$t"; then
            echo "$p $t" >> "$pidfile"
            alive=$((alive + 1))
        else
            echo "gpu_sampler: nvidia-smi loop $p exited at once; see $dir/gpu_sampler.err" >&2
        fi
    done
    [ "$alive" -gt 0 ] || exit 1
    echo "gpu_sampler: started $alive sampler(s) (every ${interval} ms) -> $dir/gpu_samples.csv, $dir/gpu_apps.csv"
    ;;
stop)
    if [ ! -f "$pidfile" ]; then echo "gpu_sampler: no $pidfile" >&2; exit 1; fi
    while read -r pid ticks; do
        if is_our_sampler "$pid" "$ticks"; then
            kill -TERM "$pid" && echo "gpu_sampler: stopped $pid"
        else
            echo "gpu_sampler: pid $pid is gone or no longer our nvidia-smi sampler; left alone"
        fi
    done < "$pidfile"
    rm -f "$pidfile"
    [ -f "$dir/gpu_samples.csv" ] && peaks "$dir/gpu_samples.csv"
    ;;
status)
    if [ -f "$pidfile" ]; then
        while read -r pid ticks; do
            if is_our_sampler "$pid" "$ticks"; then echo "running: $pid"; else echo "not running: $pid"; fi
        done < "$pidfile"
    else
        echo "no sampler started for $dir"
    fi
    [ -f "$dir/gpu_samples.csv" ] && peaks "$dir/gpu_samples.csv"
    ;;
*)
    usage
    ;;
esac
