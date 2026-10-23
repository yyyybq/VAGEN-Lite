#!/usr/bin/env bash
# Creates a NEW Python 3.11 venv for the agent server and our tools. It never touches an existing env: it refuses
# to run when the target exists, and the behavior conda env and the openpi venv are off limits as targets.
#
#   bash scripts/setup_agent_env.sh [--prefix /nas/baiqiao/envs/b1k_agent] [--with-anthropic] [--dry-run]
#
# Installs exact pins (the same versions as openpi's uv.lock at 0cc8e355, plus pytest):
#   numpy==1.26.4 websockets==15.0.1 msgpack==1.1.0 pyyaml==6.0.2 pillow==11.2.1 httpx==0.28.1 pytest==8.3.4
# --with-anthropic additionally installs the Anthropic SDK (`anthropic`, or $ANTHROPIC_SPEC, e.g. anthropic==X.Y.Z)
# for providers/anthropic_sdk.py, the Claude SUBSTITUTE provider (its results must be labelled claude-substitute,
# never Astra). The exact command is printed and the resulting `pip freeze` is saved in <prefix>/b1k_agent_env.txt.
#
# Alternative without any install: the agent server only needs the pinned packages above, and the openpi venv
# already has exactly these versions, so it can run there with PYTHONPATH:
#   PYTHONPATH=/nas/baiqiao/b1k_agent /nas/baiqiao/openpi/.venv/bin/python -m agent.server --config ...
# (pytest there, if present, is openpi's dev pin 8.3.5.) Use this script only when you want a separate env, pytest,
# or the Claude provider. Record the command you ran in docs/ENVIRONMENT.md.
#
# Tooling: `uv venv -p 3.11` when uv is on PATH, else `python3.11 -m venv`.
# Status: written and syntax-checked (bash -n, tests/test_ops_shell.py); not yet run on 105.
set -euo pipefail

PREFIX=/nas/baiqiao/envs/b1k_agent
WITH_ANTHROPIC=0
DRY_RUN=0
PINS=(numpy==1.26.4 websockets==15.0.1 msgpack==1.1.0 pyyaml==6.0.2 pillow==11.2.1 httpx==0.28.1 pytest==8.3.4)
PROTECTED=(/nas/baiqiao/envs/behavior /nas/baiqiao/openpi/.venv)

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) [ $# -ge 2 ] || { echo "setup_agent_env: --prefix needs a path" >&2; exit 2; }; PREFIX=$2; shift 2 ;;
        --with-anthropic) WITH_ANTHROPIC=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "setup_agent_env: unknown argument $1" >&2; exit 2 ;;
    esac
done

real_prefix=$(realpath -m "$PREFIX")
for p in "${PROTECTED[@]}"; do
    if [ "$real_prefix" = "$(realpath -m "$p")" ]; then
        echo "setup_agent_env: refusing to touch the existing env $p; choose another --prefix" >&2
        exit 1
    fi
done
if [ -e "$PREFIX" ]; then
    echo "setup_agent_env: $PREFIX already exists; this script only creates NEW envs (remove it yourself or use --prefix)" >&2
    exit 1
fi

run() {
    echo "+ $*"
    if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

if command -v uv >/dev/null 2>&1; then
    run uv venv -p 3.11 "$PREFIX"
    PIP=(uv pip install --python "$PREFIX/bin/python")
elif command -v python3.11 >/dev/null 2>&1; then
    run python3.11 -m venv "$PREFIX"
    PIP=("$PREFIX/bin/python" -m pip install)
else
    echo "setup_agent_env: neither uv nor python3.11 found on PATH" >&2
    exit 1
fi

run "${PIP[@]}" "${PINS[@]}"
if [ "$WITH_ANTHROPIC" = 1 ]; then
    echo "# --with-anthropic: Claude substitute provider (providers/anthropic_sdk.py); label its results claude-substitute"
    run "${PIP[@]}" "${ANTHROPIC_SPEC:-anthropic}"
fi

if [ "$DRY_RUN" = 0 ]; then
    "$PREFIX/bin/python" -c 'import numpy, websockets, msgpack, yaml, PIL, httpx, pytest
print("ok: numpy", numpy.__version__, "websockets", websockets.__version__, "msgpack", msgpack.version,
      "pyyaml", yaml.__version__, "pillow", PIL.__version__, "httpx", httpx.__version__, "pytest", pytest.__version__)'
    {
        echo "# created $(date -Is) by scripts/setup_agent_env.sh (with_anthropic=$WITH_ANTHROPIC)"
        if command -v uv >/dev/null 2>&1; then uv pip freeze --python "$PREFIX/bin/python"; else "$PREFIX/bin/python" -m pip freeze; fi
    } > "$PREFIX/b1k_agent_env.txt"
    echo "env ready: $PREFIX (package list in $PREFIX/b1k_agent_env.txt)"
    echo "use: cd /nas/baiqiao/b1k_agent && PYTHONPATH=. $PREFIX/bin/python -m pytest -q"
fi
