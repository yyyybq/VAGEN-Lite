#!/usr/bin/env bash
# Step-1 takeover inventory for server 105. READ-ONLY and safe to run at any time.
#
#   bash scripts/takeover_check.sh            # -> ${OUT_DIR:-/nas/baiqiao/b1k_agent/docs/takeover}/takeover_<ts>.md
#   OUT_DIR=$HOME/takeover bash scripts/takeover_check.sh
#
# It never starts, stops, kills, renices or signals any process, never uses sudo, writes only under OUT_DIR, and
# never launches Isaac Sim: versions come from package metadata; the single `import omnigibson` check runs with
# CUDA_VISIBLE_DEVICES="" (no GPU) and is skipped with TAKEOVER_SKIP_IMPORTS=1 (that import itself leaves an empty
# temp dir, omnigibson/__init__.py tempfile.mkdtemp). git runs with GIT_OPTIONAL_LOCKS=0, so `git status` does not
# rewrite the index of the shared checkouts.
#
# The report quotes the user's scripts, logs and command lines, and OUT_DIR defaults to a directory inside this repo,
# so every line goes through `redact` (values of *KEY / *TOKEN / *SECRET / *PASSWORD assignments, --*key / --*token
# flags, Bearer tokens, sk- / hf_ / ghp_ prefixes become <redacted>). Still read the report before committing it.
#
# Sections
#   1 smoke-test watcher: matching processes (ps), their cwd and open log files, tmux/screen/crontab entries,
#     the user's scripts in RUNS_DIR (printed so they can be read and reused, never duplicated), the newest logs
#     under RUNS_DIR, and a state not_started / running / finished / failed / unknown with its evidence lines
#   2 GPUs: nvidia-smi, driver, CUDA, free boards (< 2048 MiB and < 10 %), compute apps with owners,
#     vulkaninfo --summary, EGL vendor and Vulkan ICD files, recent Isaac Sim (Kit) log errors
#   3 versions: BEHAVIOR-1K and openpi git state, omnigibson / isaacsim / torch / warp-lang / pyarrow / numpy in the
#     behavior env, the openpi venv packages
#   4 disk: df, du -shD of datasets/* (symlinked entries followed, each capped by DU_TIMEOUT), inventory_datasets.py
#   5 checkpoint: scripts/inspect_checkpoint.py
#
# Environment (defaults): OUT_DIR, B1K_ROOT (/nas/baiqiao/BEHAVIOR-1K), DATA_ROOT ($B1K_ROOT/datasets),
# OPENPI_ROOT (/nas/baiqiao/openpi), BEHAVIOR_ENV (/nas/baiqiao/envs/behavior), RUNS_DIR (/nas/baiqiao/b1k_runs),
# CKPT_DIR (/nas/baiqiao/checkpoints/pi05_turn_on_the_radio), WATCH_PATTERN ('b1k_runs|smoke|watcher'),
# LOG_MMIN (1440), MAX_LOGS (5), TAIL_LINES (40), DU_TIMEOUT (120 s), PY (python for our stdlib scripts: the
# openpi venv python, else python3), B1K_ENV_SETUP (sourced in a subshell before the behavior-env import check,
# as eval/launch.py does before the evaluator), TAKEOVER_SKIP_IMPORTS (1 = metadata only).
#
# Status: written and syntax-checked (bash -n, tests/test_ops_shell.py); not yet run on 105.
set -u

B1K_ROOT=${B1K_ROOT:-/nas/baiqiao/BEHAVIOR-1K}
DATA_ROOT=${DATA_ROOT:-$B1K_ROOT/datasets}
OPENPI_ROOT=${OPENPI_ROOT:-/nas/baiqiao/openpi}
BEHAVIOR_ENV=${BEHAVIOR_ENV:-/nas/baiqiao/envs/behavior}
RUNS_DIR=${RUNS_DIR:-/nas/baiqiao/b1k_runs}
CKPT_DIR=${CKPT_DIR:-/nas/baiqiao/checkpoints/pi05_turn_on_the_radio}
OUT_DIR=${OUT_DIR:-/nas/baiqiao/b1k_agent/docs/takeover}
WATCH_PATTERN=${WATCH_PATTERN:-b1k_runs|smoke|watcher}
LOG_MMIN=${LOG_MMIN:-1440}
MAX_LOGS=${MAX_LOGS:-5}
TAIL_LINES=${TAIL_LINES:-40}
DU_TIMEOUT=${DU_TIMEOUT:-120}
TAKEOVER_SKIP_IMPORTS=${TAKEOVER_SKIP_IMPORTS:-0}
export GIT_OPTIONAL_LOCKS=0
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TS=$(date +%Y%m%d-%H%M%S)
if [ -z "${PY:-}" ]; then
    if [ -x "$OPENPI_ROOT/.venv/bin/python" ]; then PY="$OPENPI_ROOT/.venv/bin/python"; else PY=python3; fi
fi

FAIL_RE='Traceback|[Ee]rror|FAIL|[Ff]ailed|Segmentation fault|core dumped|Killed|exit code [1-9]|non-zero'
OK_RE='PASS|[Pp]assed|SUCCESS|[Ss]uccess|SMOKE[ _]OK|[Ff]inished|[Cc]ompleted'
WAIT_RE='[Ww]aiting|no free GPU|[Nn]o (free|idle) (GPU|card)|GPUs? busy|[Ss]leep|[Rr]etry|[Pp]olling'
RUN_RE='[Ss]tarting|[Ll]aunch|[Rr]unning|omnigibson|[Ii]saac|reset|step'

WATCH_STATE=unknown
WATCH_REASON=""
FREE_GPUS="unmeasured"
DRIVER_CUDA=""

progress() { echo "[takeover_check] $*" >&3; }
h2() { printf '\n## %s\n\n' "$*"; }
fence() { echo '```'; }
tmo() { local s=$1; shift; if command -v timeout >/dev/null 2>&1; then timeout "$s" "$@"; else "$@"; fi; }
have() { command -v "$1" >/dev/null 2>&1; }
redact() {  # secrets must never reach the report (CLAUDE.md: keys only from env vars, never logged)
    sed -E \
        -e 's/\b([A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD|PASSWD)[A-Z0-9_]*[[:space:]]*=[[:space:]]*["'\'']?)[^"'\''[:space:]]+/\1<redacted>/g' \
        -e 's/(((api|access|secret|auth)[_-]?(key|token)|token|secret|password|passwd)["'\'']?[[:space:]]*[:=][[:space:]]*["'\'']?)[^"'\''[:space:],;]+/\1<redacted>/Ig' \
        -e 's/(--[A-Za-z0-9-]*(key|token|secret|password)[A-Za-z0-9-]*[[:space:]]+)[^-[:space:]][^[:space:]]*/\1<redacted>/Ig' \
        -e 's/(bearer[[:space:]]+)[^[:space:]"'\'']+/\1<redacted>/Ig' \
        -e 's/\b(sk-|hf_|ghp_|gho_|github_pat_|xox[bp]-)[A-Za-z0-9_-]{6,}/\1<redacted>/g'
}
last_kind() {  # last classified line of a log: prints "<kind>\t<line>"
    tail -n 300 "$1" 2>/dev/null | awk -v f="$FAIL_RE" -v o="$OK_RE" -v w="$WAIT_RE" -v r="$RUN_RE" '
        $0 ~ f { k = "fail"; l = $0; next } $0 ~ o { k = "ok"; l = $0; next }
        $0 ~ w { k = "wait"; l = $0; next } $0 ~ r { k = "run"; l = $0 }
        END { if (k != "") printf "%s\t%s\n", k, l }'
}

section_host() {
    h2 "0. Host"
    fence
    echo "date: $(date -Is)   host: $(hostname)   user: $(id -un) ($(id -u))"
    echo "kernel: $(uname -r)   cpus: $(nproc 2>/dev/null)"
    head -n 2 /etc/os-release 2>/dev/null
    free -g 2>/dev/null | head -n 2
    for p in "$B1K_ROOT" "$DATA_ROOT" "$OPENPI_ROOT" "$BEHAVIOR_ENV" "$RUNS_DIR" "$CKPT_DIR"; do
        if [ -e "$p" ]; then echo "present: $p"; else echo "MISSING: $p"; fi
    done
    have nvidia-smi && echo "nvidia-smi: $(command -v nvidia-smi)" || echo "nvidia-smi: MISSING (not server 105?)"
    fence
}

section_watcher() {
    progress "1/5 smoke-test watcher"
    h2 "1. Smoke-test watcher"
    local mypgid matches logs fdlogs newest kind_line kind alive=0
    mypgid=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
    matches=$(ps -eo pid=,pgid=,user=,etime=,args= 2>/dev/null | awk -v pat="$WATCH_PATTERN" -v mine="$mypgid" '
        $2 != mine && $0 ~ pat && $0 !~ /takeover_check/ { print }')
    echo "Processes matching \`$WATCH_PATTERN\` (this script's process group excluded):"
    fence
    if [ -z "$matches" ]; then
        echo "(none)"
    else
        while read -r pid _pgid user etime args; do
            alive=$((alive + 1))
            echo "pid $pid  user $user  etime $etime"
            echo "  cmd: $args"
            echo "  cwd: $(readlink "/proc/$pid/cwd" 2>/dev/null || echo '(not readable)')"
            ls -l "/proc/$pid/fd" 2>/dev/null | grep -E 'log|out' | sed 's/^/  fd: /' | head -n 20
        done <<< "$matches"
    fi
    fence
    echo "tmux / screen sessions and crontab entries (read-only listing):"
    fence
    have tmux && tmux ls 2>&1 | head -n 20
    have screen && screen -ls 2>&1 | head -n 20
    crontab -l 2>/dev/null | grep -E "$WATCH_PATTERN" || echo "(no matching crontab lines)"
    fence

    echo "Contents of $RUNS_DIR (the user's run helpers; reuse them):"
    fence
    ls -la "$RUNS_DIR" 2>&1 | head -n 60
    fence
    local f
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        echo "#### $f"
        fence
        head -n 200 "$f" 2>&1
        fence
    done < <(find "$RUNS_DIR" -maxdepth 2 -type f \( -name '*.sh' -o -name '*.py' -o -name '*.bash' \) 2>/dev/null | sort)

    logs=$(find "$RUNS_DIR" -maxdepth 6 -type f \( -name '*.log' -o -name '*.out' -o -name '*.err' -o -name 'nohup*' \) \
        -mmin "-$LOG_MMIN" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n "$MAX_LOGS" | cut -d' ' -f2-)
    fdlogs=""
    if [ -n "$matches" ]; then
        fdlogs=$(while read -r pid _rest; do
            for fd in /proc/"$pid"/fd/*; do
                t=$(readlink "$fd" 2>/dev/null) || continue
                case "$t" in *log*|*.out) [ -f "$t" ] && echo "$t" ;; esac
            done
        done <<< "$matches" | sort -u)
    fi
    logs=$(printf '%s\n%s\n' "$fdlogs" "$logs" | awk 'NF && !seen[$0]++')
    echo "Newest logs (open by a matching process, or modified in the last $LOG_MMIN min under $RUNS_DIR):"
    if [ -z "$logs" ]; then
        echo "(none)"
    else
        while IFS= read -r f; do
            echo "#### $f"
            fence
            ls -l "$f" 2>&1
            tail -n "$TAIL_LINES" "$f" 2>&1
            fence
        done <<< "$logs"
    fi

    newest=$(printf '%s\n' "$logs" | head -n 1)
    kind_line=""
    [ -n "$newest" ] && kind_line=$(last_kind "$newest")
    kind=${kind_line%%$'\t'*}
    if [ "$alive" -gt 0 ]; then
        if [ "$kind" = "wait" ]; then
            WATCH_STATE=not_started; WATCH_REASON="watcher process alive and its newest log says it is waiting"
        else
            WATCH_STATE=running; WATCH_REASON="watcher/smoke process alive (newest log kind: ${kind:-none})"
        fi
    elif [ -z "$newest" ]; then
        WATCH_STATE=unknown; WATCH_REASON="no matching process and no log modified in the last $LOG_MMIN min under $RUNS_DIR"
    elif [ "$kind" = "fail" ]; then
        WATCH_STATE=failed; WATCH_REASON="no matching process; the last classified line of the newest log is a failure"
    elif [ "$kind" = "ok" ]; then
        WATCH_STATE=finished; WATCH_REASON="no matching process; the last classified line of the newest log is a success"
    else
        WATCH_STATE=unknown; WATCH_REASON="no matching process; newest log inconclusive (kind: ${kind:-none})"
    fi
    echo
    echo "**Watcher state: $WATCH_STATE** ($WATCH_REASON). Heuristic: confirm from the evidence lines."
    echo
    echo "Evidence (newest logs, last matching lines):"
    fence
    [ -n "$kind_line" ] && echo "decisive line in $newest: ${kind_line#*$'\t'}"
    printf '%s\n' "$logs" | head -n 3 | while IFS= read -r f; do
        [ -n "$f" ] || continue
        grep -n -E "$FAIL_RE|$OK_RE|$WAIT_RE" "$f" 2>/dev/null | tail -n 10 | sed "s|^|$(basename "$f"):|"
    done
    fence
}

section_gpus() {
    progress "2/5 GPUs and rendering stack"
    h2 "2. GPUs, driver, rendering stack"
    if ! have nvidia-smi; then
        echo "nvidia-smi not found: not server 105, or no driver."
    else
        fence
        tmo 60 nvidia-smi 2>&1
        echo
        tmo 60 nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu,uuid --format=csv 2>&1
        fence
        DRIVER_CUDA=$(tmo 60 nvidia-smi 2>/dev/null | grep -o -E '(Driver|CUDA) Version: [0-9.]+' | paste -sd ',' -)
        echo "Driver / CUDA: ${DRIVER_CUDA:-unknown}"
        FREE_GPUS=$(tmo 60 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', *' '$2 ~ /^[0-9.]+$/ && $3 ~ /^[0-9.]+$/ && $2 < 2048 && $3 < 10 {printf "%s%s", sep, $1; sep=","}')
        FREE_GPUS=${FREE_GPUS:-none}
        echo "Free GPUs (memory.used < 2048 MiB and utilization < 10 %): $FREE_GPUS"
        echo
        echo "Compute processes and owners (read-only):"
        fence
        tmo 60 nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name --format=csv,noheader,nounits 2>&1 \
            | while IFS= read -r line; do
                pid=$(echo "${line%%,*}" | tr -d ' ')
                case "$pid" in ''|*[!0-9]*) echo "$line"; continue ;; esac
                echo "owner $(ps -o user= -p "$pid" 2>/dev/null || echo '?'): $line"
            done
        fence
    fi
    echo "Vulkan:"
    fence
    if have vulkaninfo; then tmo 60 vulkaninfo --summary 2>&1 | head -n 80; else echo "vulkaninfo not installed"; fi
    for d in /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
        [ -d "$d" ] && { ls -l "$d"; for j in "$d"/*nvidia*.json; do [ -f "$j" ] && { echo "--- $j"; cat "$j"; }; done; }
    done
    fence
    echo "EGL (headless rendering):"
    fence
    for d in /usr/share/glvnd/egl_vendor.d /etc/glvnd/egl_vendor.d; do
        if [ -d "$d" ]; then ls -l "$d"; for j in "$d"/*.json; do [ -f "$j" ] && { echo "--- $j"; cat "$j"; }; done
        else echo "$d: absent"; fi
    done
    have ldconfig && ldconfig -p 2>/dev/null | grep -E 'libEGL_nvidia|libGLX_nvidia|libnvidia-glcore|libvulkan' | head -n 10
    env | grep -E '^(__EGL|__GLX|VK_|DISPLAY|OMNIGIBSON|OMNI_|CUDA_|LD_PRELOAD|LD_LIBRARY_PATH)' \
        | grep -v -i -E 'key|token|pass|secret' | sed 's/^/env: /'
    fence
    echo "Isaac Sim / Kit logs (newest, error-like lines) and OmniGibson appdata:"
    fence
    local k
    for k in $(find "$HOME/.nvidia-omniverse/logs" -type f -name '*.log' -mmin "-$LOG_MMIN" -printf '%T@ %p\n' 2>/dev/null \
            | sort -rn | head -n 2 | cut -d' ' -f2-); do
        echo "--- $k"
        grep -i -E 'error|vulkan|driver|gpu|rtx|egl|crash' "$k" 2>/dev/null | tail -n 30
    done
    ls -la "$HOME/.omnigibson_appdata" 2>&1 | head -n 20
    fence
}

section_versions() {
    progress "3/5 versions"
    h2 "3. Versions"
    fence
    if [ -d "$B1K_ROOT/.git" ] || [ -f "$B1K_ROOT/.git" ]; then
        echo "BEHAVIOR-1K $B1K_ROOT"
        git -C "$B1K_ROOT" describe --tags --always 2>&1
        git -C "$B1K_ROOT" rev-parse HEAD 2>&1
        git -C "$B1K_ROOT" log -1 --format='%ci %s' 2>&1
        git -C "$B1K_ROOT" branch --show-current 2>&1
        echo "status (first 20):"; git -C "$B1K_ROOT" status --short 2>&1 | head -n 20
        git -C "$B1K_ROOT" worktree list 2>&1
    else
        echo "BEHAVIOR-1K: no git checkout at $B1K_ROOT"
    fi
    echo
    if [ -d "$OPENPI_ROOT/.git" ] || [ -f "$OPENPI_ROOT/.git" ]; then
        echo "openpi $OPENPI_ROOT"
        git -C "$OPENPI_ROOT" rev-parse HEAD 2>&1
        git -C "$OPENPI_ROOT" branch --show-current 2>&1
        git -C "$OPENPI_ROOT" log -1 --format='%ci %s' 2>&1
        echo "status (first 20):"; git -C "$OPENPI_ROOT" status --short 2>&1 | head -n 20
    else
        echo "openpi: no git checkout at $OPENPI_ROOT"
    fi
    fence
    local meta_py='
import importlib.metadata as m, sys
print("python", sys.version.split()[0], sys.executable)
for p in sys.argv[1:]:
    try:
        print(p, m.version(p))
    except m.PackageNotFoundError:
        print(p, "not installed")
extra = sorted({(d.metadata["Name"] or "") for d in m.distributions() if (d.metadata["Name"] or "").lower().startswith("isaacsim")})
print("isaacsim distributions:", ", ".join(extra[:15]) or "none")
'
    echo "behavior env $BEHAVIOR_ENV (package metadata, nothing imported):"
    fence
    if [ -x "$BEHAVIOR_ENV/bin/python" ]; then
        tmo 120 "$BEHAVIOR_ENV/bin/python" -c "$meta_py" omnigibson isaacsim bddl torch warp-lang pyarrow numpy 2>&1
        echo "pip list (selected):"
        tmo 120 "$BEHAVIOR_ENV/bin/python" -m pip list --disable-pip-version-check 2>/dev/null \
            | grep -i -E '^(torch|warp-lang|pyarrow|numpy|isaacsim|omnigibson|bddl|bddl3)[[:space:]]' | head -n 20
    else
        echo "no python at $BEHAVIOR_ENV/bin/python"
    fi
    fence
    if [ "$TAKEOVER_SKIP_IMPORTS" != "1" ] && [ -x "$BEHAVIOR_ENV/bin/python" ]; then
        echo "import check (env activation hooks via conda run when available; CUDA hidden; Isaac Sim not launched):"
        fence
        (
            set +u  # conda activate hooks and user setup files are not nounset-safe
            [ -n "${B1K_ENV_SETUP:-}" ] && [ -f "$B1K_ENV_SETUP" ] && . "$B1K_ENV_SETUP"
            export CUDA_VISIBLE_DEVICES="" OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES
            code='import omnigibson; print("omnigibson.__version__", omnigibson.__version__)'
            conda_exe=${CONDA_EXE:-$(type -P conda || true)}  # a binary, not the conda shell function
            if [ -n "$conda_exe" ]; then
                tmo 300 "$conda_exe" run -p "$BEHAVIOR_ENV" --no-capture-output python -c "$code" 2>&1 | tail -n 15
            else
                tmo 300 "$BEHAVIOR_ENV/bin/python" -c "$code" 2>&1 | tail -n 15
            fi
        ) || true
        fence
    fi
    echo "openpi venv $OPENPI_ROOT/.venv:"
    fence
    if [ -x "$OPENPI_ROOT/.venv/bin/python" ]; then
        tmo 120 "$OPENPI_ROOT/.venv/bin/python" -c "$meta_py" openpi jax jaxlib flax torch numpy websockets msgpack \
            pyyaml pillow httpx pytest tyro lerobot 2>&1
    else
        echo "no python at $OPENPI_ROOT/.venv/bin/python"
    fi
    echo "python for our scripts: $PY ($("$PY" --version 2>&1))"
    fence
}

section_disk() {
    progress "4/5 disk and datasets"
    h2 "4. Disk and datasets"
    fence
    df -h /nas/baiqiao "$HOME" /tmp 2>&1
    fence
    echo "du -shD $DATA_ROOT/* (symlinked entries are followed; each capped at ${DU_TIMEOUT}s):"
    fence
    local d
    for d in "$DATA_ROOT"/*; do
        [ -e "$d" ] || { echo "(empty or missing: $DATA_ROOT)"; break; }
        [ -L "$d" ] && echo "$d -> $(readlink "$d")"
        tmo "$DU_TIMEOUT" du -shD "$d" 2>/dev/null || echo "timeout or unreadable: $d"
    done
    fence
    echo "scripts/inventory_datasets.py (JSON: $OUT_DIR/datasets_$TS.json):"
    fence
    tmo 900 "$PY" "$SCRIPT_DIR/inventory_datasets.py" "$DATA_ROOT" --task-data "$B1K_ROOT/docs/challenge/task_data.json" \
        --out "$OUT_DIR/datasets_$TS.json" --time-limit-s 600 2>&1 | head -n 250
    fence
}

section_checkpoint() {
    progress "5/5 checkpoint"
    h2 "5. Checkpoint"
    fence
    tmo 300 "$PY" "$SCRIPT_DIR/inspect_checkpoint.py" "$CKPT_DIR" --out "$OUT_DIR/ckpt_$TS.json" 2>&1 | head -n 250
    fence
}

section_summary() {
    h2 "Summary"
    echo "- smoke-test watcher: **$WATCH_STATE** ($WATCH_REASON)"
    echo "- free GPUs now: $FREE_GPUS; ${DRIVER_CUDA:-driver / CUDA unknown}"
    echo "- files: $REPORT, $OUT_DIR/datasets_$TS.json, $OUT_DIR/ckpt_$TS.json"
    echo "- record the facts in STATUS.md; this report is evidence, not a result"
}

main() {
    echo "# Takeover inventory $TS"
    echo
    echo "Generated by scripts/takeover_check.sh (read-only; nothing was started, stopped or signalled)."
    section_host
    section_watcher
    section_gpus
    section_versions
    section_disk
    section_checkpoint
    section_summary
}

mkdir -p "$OUT_DIR" || { echo "takeover_check: cannot create $OUT_DIR" >&2; exit 1; }
REPORT="$OUT_DIR/takeover_$TS.md"
exec 3>&2
main > >(redact > "$REPORT") 2>&1  # main stays in this shell (WATCH_STATE); only its output is filtered
wait $!                             # the redact process (bash >= 4.4)
echo "[takeover_check] watcher state: $WATCH_STATE ($WATCH_REASON); free GPUs: $FREE_GPUS" >&3
echo "$REPORT"
