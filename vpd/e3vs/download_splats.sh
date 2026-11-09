#!/usr/bin/env bash
# E3VS-Bench inputs: the pinned repo and the SceneSplat++ splats of the 99 E3VS scenes.
#
# STOP-AND-ASK (DECISIONS A3): every subcommand except `plan` / DRY_RUN=1 reads from outside the cluster or
# copies the not-yet-approved repo copy.  Run it only after the user's OK.  The splat subcommands also need the
# user's HF token file (vpd/e3vs/USER_STEPS.md).  huggingface_hub reads the token from HF_TOKEN_PATH, and one
# stdlib python HEAD request reads the same file; the token is never echoed, logged or put on a command line.
#
# Usage (from the vpd worktree, login node; network-bound, one process, nice -n 19):
#   bash vpd/e3vs/download_splats.sh repo              # pinned shallow fetch of E3VS-BENCH (default https proxy)
#   bash vpd/e3vs/download_splats.sh repo-from-copy    # or: copy the verified plan/login_tmp copy (no network)
#   bash vpd/e3vs/download_splats.sh check             # HEAD with the token on one val PLY, no download
#   bash vpd/e3vs/download_splats.sh splats valtest    # the 31 val+test scenes (~11 GB)
#   bash vpd/e3vs/download_splats.sh splats train      # the 68 train scenes (~24 GB)
#   bash vpd/e3vs/download_splats.sh splats all        # val+test first, then train (~35 GB)
#   bash vpd/e3vs/download_splats.sh verify [set]      # sizes + sha256 manifest of what is on disk
#   bash vpd/e3vs/download_splats.sh plan [set]        # same as DRY_RUN=1 splats [set]
# Env:
#   DRY_RUN=1      print the plan only: no network, no token read, no writes.
#   ROUTE=mirror   (default) hf-mirror.com with every *_proxy unset; the mirror operator can see the token.
#   ROUTE=hf       huggingface.co through the default corporate proxy (token stays inside TLS; slow).
#   E3VS_DATA_DIR  override the target (tests only).
# Every splat run checks access first: HTTP 2xx/3xx continues; 401/403 prints the fallback and stops (exit 3).
set -euo pipefail
umask 022

OUT=/mnt/umm/users/yinbaiqiao/eval_external_assets/vpd
E3VS=$OUT/e3vs
DATA=${E3VS_DATA_DIR:-$E3VS/data/scannetpp_v2_mcmc_3dgs}
REPO_DIR=$E3VS/repo/E3VS-BENCH
LOGIN_TMP_REPO=$OUT/plan/login_tmp/vpd_e3vs/repo
REPO_URL=https://github.com/K0uya/E3VS-BENCH
COMMIT=bb7e603ccdcf2dd77fbeedd15660a09310a8a881
PY_EVAL=/mnt/umm/users/yinbaiqiao/.conda/envs/eval-external/bin/python
ENV_BIN=$(dirname "$PY_EVAL")
DATASET=GaussianWorld/scannetpp_v2_mcmc_3dgs
TOKEN_FILE=/mnt/umm/users/yinbaiqiao/.secrets/hf_token_e3vs
PROBE_SCENE=00dd871005   # first val scene
PLY=ckpts/point_cloud_30000.ply
DRY_RUN=${DRY_RUN:-0}
ROUTE=${ROUTE:-mirror}
LOG=$E3VS/download.log

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

# split scene ply_bytes, from the survey's hf-mirror tree listing (plan/login_tmp/vpd_e3vs/scenes_sizes.json,
# 2026-11-08); val first, then test, then train.  vpd/tests/test_e3vs_geometry.py checks it against the episodes.
scene_table() {
    cat <<'TABLE'
val 00dd871005 354001006
val 02c2ddee2a 353983070
val 0caa1ae59a 354001478
val 15c4aa5bbb 354001242
val 2e5c249de3 354001478
val 2e74812d00 354000298
val 30f4a2b44d 354001478
val 37c9538a2b 354001478
val 390eda9157 354001478
val 75d29d69b8 354001242
test 00a231a370 354000062
test 053a94cf68 354001478
test 06bd2cf800 354001478
test 0cf2e9402d 354001478
test 108ec0b806 354001478
test 120acffd90 354001478
test 1a22a99186 354001478
test 251443268c 354001478
test 29b18b8c4f 354001478
test 2ab7bea148 354001478
test 3583c4c70f 354001478
test 3878be46f0 354001478
test 3d838ee1ab 354001006
test 45b0dac5e3 354001478
test 51bdbf173f 354001478
test 53755e535e 354001242
test 5f99900f09 354000298
test 603964346e 354001242
test 610eff6509 354001242
test 612f70fe00 354001478
test 76f0eb2577 354001478
train 0373d65cce 354001478
train 04d0dc245b 354001478
train 04df8734b7 354001478
train 052d72e137 354001478
train 062e5a23a6 354000770
train 0658da5bc0 354001478
train 06b5863f73 354001242
train 06bc6d1b24 354001242
train 079a326597 354001478
train 07ff1c45bb 354001478
train 086f09d6e3 354001242
train 08bd80ce2a 354001478
train 0a184cf634 353999826
train 0a5c013435 354001478
train 0c5385e84b 354001242
train 0e350246d3 354001478
train 16c9bd2e1e 354001478
train 17268bec90 354001478
train 17a5e7d36c 354001242
train 196876abd7 354001478
train 1a130d092a 354001478
train 1a3100752b 354001478
train 20207b4844 354001242
train 216b9e55e8 354001478
train 2284bf5c9d 354001478
train 238b940049 354001478
train 260fa55d50 354001478
train 29c7afafed 354001242
train 2a1b555966 354001478
train 2b71155e0d 354000062
train 2d58610dce 354001478
train 2f6f83ea1f 354001478
train 302a7f6b67 354001478
train 303745abc7 353999826
train 324d07a5b3 354001478
train 33f78338cd 354001478
train 3549335582 354001242
train 37562e7f48 354001478
train 3799bd47b3 354001478
train 394a542a19 354001478
train 3cb9f85891 354001478
train 3ce88cdad2 354001242
train 3e8bba0176 354001478
train 3f6ee92fd7 354001478
train 40aec5fffa 354001478
train 4117a4f02c 354001478
train 413085a827 354001478
train 4318f8bb3c 354001478
train 4380e4646a 354001242
train 4423a61d09 353996522
train 4897e95232 354001478
train 48da625e8c 354001478
train 49789448b8 354001478
train 49fe768553 354001478
train 4aef651da7 354000770
train 504cf57907 354001478
train 50809ea0d8 354001478
train 511061232 354001478
train 54b005d19d 354001006
train 55b2bf8036 354001242
train 57d9fecae9 354001478
train 58960ff105 354001478
train 589f5c7c58 354001478
train 59e3f1ea37 354001478
train 5aeac3800a 354001478
train 5c215ef3b0 354000770
train 709ab5bffe 354001242
train 725b8f0cba 354001242
TABLE
}

die() { echo "ERROR: $1" >&2; exit "${2:-1}"; }
log() {
    local m
    m="$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"
    echo "$m"
    if [ "$DRY_RUN" != 1 ]; then mkdir -p "$E3VS"; echo "$m" >> "$LOG"; fi
}

scenes_for() {  # $1 = valtest | train | all
    case "$1" in
        valtest) scene_table | awk '$1 == "val" || $1 == "test"' ;;
        train) scene_table | awk '$1 == "train"' ;;
        all) scene_table ;;
        *) die "unknown scene set '$1' (valtest | train | all)" 2 ;;
    esac
}

hf_cli() {
    if [ -x "$ENV_BIN/hf" ]; then echo "$ENV_BIN/hf"
    elif [ -x "$ENV_BIN/huggingface-cli" ]; then echo "$ENV_BIN/huggingface-cli"
    else die "neither hf nor huggingface-cli in $ENV_BIN"; fi
}

setup_route() {
    unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
    export HF_HUB_DISABLE_XET=1 HF_TOKEN_PATH="$TOKEN_FILE" HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_TELEMETRY=1
    export HF_HOME="$E3VS/hf_home"   # keep huggingface_hub state off ~/.cache
    case "$ROUTE" in
        mirror)
            local v
            for v in $(env | awk -F= 'tolower($1) ~ /_proxy$/ {print $1}'); do unset "$v"; done
            export HF_ENDPOINT=https://hf-mirror.com ;;
        hf) export HF_ENDPOINT=https://huggingface.co ;;   # keeps the default proxy
        *) die "ROUTE must be mirror or hf" 2 ;;
    esac
}

check_token_file() {
    [ -f "$TOKEN_FILE" ] || die "no token file $TOKEN_FILE; follow vpd/e3vs/USER_STEPS.md (the user places it)"
    [ -s "$TOKEN_FILE" ] || die "token file $TOKEN_FILE is empty"
    [ "$(stat -c %a "$TOKEN_FILE")" = 600 ] || die "token file must be mode 600 (chmod 600 $TOKEN_FILE)"
    [ "$(stat -c %u "$TOKEN_FILE")" = "$(id -u)" ] || die "token file is not owned by $(id -un)"
}

# HEAD with the token on one val PLY; prints "<http status> <x-repo-commit or ->" only.
head_status() {
    VPD_TOKEN_FILE="$TOKEN_FILE" VPD_URL="$HF_ENDPOINT/datasets/$DATASET/resolve/main/$PROBE_SCENE/$PLY" \
    nice -n 19 "$PY_EVAL" -I - <<'PY'
import os, urllib.error, urllib.request
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None  # a 3xx means authorised; never follow it (that would start the download)
with open(os.environ["VPD_TOKEN_FILE"]) as f:
    tok = f.read().strip()
req = urllib.request.Request(os.environ["VPD_URL"], method="HEAD",
                             headers={"Authorization": "Bearer " + tok, "User-Agent": "vpd-e3vs-check"})
del tok
op = urllib.request.build_opener(NoRedirect)  # env proxies: none on the mirror route, the default one on ROUTE=hf
try:
    r = op.open(req, timeout=60)
    code, hdr = r.status, r.headers
except urllib.error.HTTPError as e:
    code, hdr = e.code, e.headers
except Exception as e:
    print("000 -", type(e).__name__)
    raise SystemExit(0)
print(code, (hdr.get("x-repo-commit") or "-") if hdr else "-")
PY
}

check_access() {
    check_token_file
    local out code rev
    out=$(head_status)
    code=${out%% *}
    rev=$(echo "$out" | awk '{print $2}')
    case "$code" in
        2??|3??)
            log "access OK via $ROUTE ($HF_ENDPOINT): HTTP $code, dataset revision $rev"
            if [ "$rev" != - ]; then echo "$rev" > "$E3VS/data_revision.txt"; fi
            return 0 ;;
        401|403)
            echo "access REFUSED via $ROUTE ($HF_ENDPOINT): HTTP $code." >&2
            if [ "$ROUTE" = mirror ]; then
                echo "Fallback: huggingface.co through the default proxy (token stays inside TLS; about 0.1 MB/s" >&2
                echo "per connection in Oct). Ask the user first, then: ROUTE=hf bash vpd/e3vs/download_splats.sh check" >&2
            else
                echo "Check that both gated forms were accepted for the account that owns the token (USER_STEPS.md)." >&2
            fi
            exit 3 ;;
        *) die "unexpected HEAD result '$out' via $ROUTE" 4 ;;
    esac
}

file_ok() {  # $1 path, $2 expected bytes (or - for any non-empty file)
    [ -f "$1" ] || return 1
    if [ "$2" = - ]; then [ -s "$1" ]; else [ "$(stat -c %s "$1")" = "$2" ]; fi
}

download_set() {  # $1 = valtest | train | all
    local set=$1 hf rev
    hf=$(hf_cli)
    if [ "$DRY_RUN" = 1 ]; then
        local ep=https://huggingface.co
        if [ "$ROUTE" = mirror ]; then ep=https://hf-mirror.com; fi
        echo "DRY_RUN: route=$ROUTE endpoint=$ep (mirror route unsets every *_proxy)"
        echo "DRY_RUN: target $DATA; token file $TOKEN_FILE (not read)"
        echo "DRY_RUN: per file: nice -n 19 $hf download $DATASET <scene>/<file> --repo-type dataset --revision <rev> --local-dir $DATA --max-workers 1"
        scenes_for "$set" | awk '{i++; print "PLAN", i, $1, $2, $3}'
        scenes_for "$set" | awk '{s+=$3} END {printf "DRY_RUN: %d scenes, %.2f GB\n", NR, s/1e9}'
        return 0
    fi
    check_token_file   # before anything is created
    mkdir -p "$E3VS"
    exec 9>"$E3VS/.download.lock"
    flock -n 9 || die "another download_splats.sh is running"
    check_access
    rev=$(cat "$E3VS/data_revision.txt" 2>/dev/null || echo main)
    mkdir -p "$DATA"
    local need=0 split scene bytes avail i=0 n f
    while read -r split scene bytes; do
        file_ok "$DATA/$scene/$PLY" "$bytes" || need=$((need + bytes))
    done < <(scenes_for "$set")
    avail=$(df -B1 --output=avail "$DATA" | tail -1)
    [ "$avail" -gt $((need + 2000000000)) ] || die "need $need B + 2 GB free under $DATA, have $avail B"
    log "download set=$set route=$ROUTE rev=$rev need=$need B"
    n=$(scenes_for "$set" | wc -l)
    while read -r split scene bytes; do
        i=$((i + 1))
        if file_ok "$DATA/$scene/$PLY" "$bytes" && file_ok "$DATA/$scene/cfg.yml" - \
           && grep -q " $scene/$PLY\$" "$DATA/SHA256SUMS" 2>/dev/null; then
            log "[$i/$n] $split $scene already complete"
            continue
        fi
        # children get </dev/null: this loop reads the scene list on stdin, and a child that read stdin would
        # swallow the remaining scenes (hardening; verify_all below would also catch a skipped scene)
        for f in "$scene/$PLY" "$scene/cfg.yml"; do   # one file per call: hf_hub_download, no repo listing
            nice -n 19 "$hf" download "$DATASET" "$f" --repo-type dataset --revision "$rev" \
                --local-dir "$DATA" --max-workers 1 --quiet > /dev/null < /dev/null
        done
        file_ok "$DATA/$scene/$PLY" "$bytes" \
            || die "$scene: PLY size $(stat -c %s "$DATA/$scene/$PLY" 2>/dev/null || echo missing) != $bytes" 5
        file_ok "$DATA/$scene/cfg.yml" - || die "$scene: cfg.yml missing" 5
        (cd "$DATA" && nice -n 19 sha256sum "$scene/$PLY" "$scene/cfg.yml" < /dev/null) >> "$DATA/SHA256SUMS"
        log "[$i/$n] $split $scene OK ($bytes B)"
    done < <(scenes_for "$set")
    verify_all "$set"
}

verify_all() {  # sizes of the set + the sha256 manifest of everything listed in it
    local set=${1:-all} bad=0 split scene bytes
    while read -r split scene bytes; do
        if ! file_ok "$DATA/$scene/$PLY" "$bytes"; then echo "MISSING/SIZE $split $scene"; bad=$((bad + 1)); fi
    done < <(scenes_for "$set")
    if [ -f "$DATA/SHA256SUMS" ]; then
        sort -u -k2 "$DATA/SHA256SUMS" -o "$DATA/SHA256SUMS"
        (cd "$DATA" && nice -n 19 sha256sum --quiet -c SHA256SUMS) || bad=$((bad + 1))
    fi
    [ "$bad" = 0 ] || die "verify: $bad problem(s)" 6
    log "verify set=$set OK ($(scenes_for "$set" | wc -l) scenes)"
}

verify_repo() {  # $1 = checkout
    local got
    got=$(git -C "$1" rev-parse HEAD)
    [ "$got" = "$COMMIT" ] || die "$1: HEAD $got != $COMMIT"
    (cd "$1/dataset" && md5sum -c --quiet - <<'MD5'
d21ba21fafb1846699bd8d2779f56a40  episode_info_train.jsonl
6a6df2992f3f9f549738a0b8f0718288  episode_info_val.jsonl
c537eb6100ec31b371091258b3533737  episode_info_test.jsonl
MD5
    ) || die "$1: episode md5 mismatch"
    log "repo $1 OK at $COMMIT"
}

fetch_repo() {
    if [ -e "$REPO_DIR" ]; then verify_repo "$REPO_DIR"; return; fi
    if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN: git fetch --depth 1 $REPO_URL $COMMIT -> $REPO_DIR"; return; fi
    mkdir -p "$(dirname "$REPO_DIR")"
    local tmp="$REPO_DIR.partial.$$"
    git init -q "$tmp"
    git -C "$tmp" remote add origin "$REPO_URL"
    nice -n 19 git -C "$tmp" fetch -q --depth 1 origin "$COMMIT"
    git -C "$tmp" checkout -q --detach FETCH_HEAD
    verify_repo "$tmp"
    mv "$tmp" "$REPO_DIR"
    log "repo fetched to $REPO_DIR"
}

copy_repo() {
    if [ -e "$REPO_DIR" ]; then verify_repo "$REPO_DIR"; return; fi
    if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN: verify + cp -a $LOGIN_TMP_REPO $REPO_DIR"; return; fi
    verify_repo "$LOGIN_TMP_REPO"
    mkdir -p "$(dirname "$REPO_DIR")"
    cp -a "$LOGIN_TMP_REPO" "$REPO_DIR.partial.$$"
    mv "$REPO_DIR.partial.$$" "$REPO_DIR"
    verify_repo "$REPO_DIR"
}

cmd=${1:-}
case "$cmd" in
    repo) fetch_repo ;;
    repo-from-copy) copy_repo ;;
    check)
        if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN: HEAD $DATASET/$PROBE_SCENE/$PLY via $ROUTE"; exit 0; fi
        setup_route
        check_token_file   # before anything is created
        mkdir -p "$E3VS"
        check_access ;;
    splats)
        [ $# -ge 2 ] || die "usage: $0 splats valtest|train|all" 2
        scenes_for "$2" > /dev/null
        if [ "$DRY_RUN" != 1 ]; then setup_route; fi
        download_set "$2" ;;
    verify) verify_all "${2:-all}" ;;
    plan) DRY_RUN=1; download_set "${2:-all}" ;;
    *) die "usage: $0 repo | repo-from-copy | check | splats valtest|train|all | verify [set] | plan [set]" 2 ;;
esac
