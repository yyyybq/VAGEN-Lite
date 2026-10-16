#!/usr/bin/env bash
# Does anything still in use point at a path that a move manifest sends into archive/cluster/?
#   bash scripts/cluster/audit_archived_paths.sh [MANIFEST.tsv]      (default: the 2026-10-16 move manifest)
# Run it BEFORE moving data (on the planned manifest) and after. Search strings = every original path in the
# manifest plus distinctive basenames (>= 14 characters with a digit). Searched with fixed-string grep in:
#   1. files tracked on origin/main, outside archive/ and docs/reports/
#   2. text files of the live experiment folders (LIVE, default: interiorgs_v10_* folders of the last 3 days),
#      without checkpoints/renders/frames
#   3. untracked files of the shared checkout outside exps/, archive/, third_party/, playground/ (other sessions)
# Output in $OUT (default /tmp/psv/cleanup_inv/audit): hits_{main,live,untracked}.txt as file:line:match.
# Then read the hits: a reference only counts if the code reads the path (not an output default, a comment,
# a code snapshot under job/src or code/, or a file that is itself archived on the cleanup branch).
set -uo pipefail
REPO=${REPO:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}
MAN=${1:-$REPO/archive/cluster/MANIFEST_moves_20261016.tsv}
OUT=${OUT:-/tmp/psv/cleanup_inv/audit}
LIVE=${LIVE:-$(find $REPO/exps/vagen_active_spatial -maxdepth 1 -name "interiorgs_v10_*" -mtime -3 -type d | tr "\n" " ")}
mkdir -p "$OUT"
awk -F"\t" 'NR>1 && $1!="" {sub(/\/$/,"",$1); print $1; n=split($1,a,"/"); b=a[n]; if (length(b)>=14 && b ~ /[0-9]/) print b}' \
  "$MAN" | sort -u > "$OUT/needles.txt"
echo "search strings: $(wc -l < "$OUT/needles.txt")"
git -C "$REPO" fetch -q origin main 2>/dev/null
git -C "$REPO" grep -n -o -F -f "$OUT/needles.txt" origin/main -- . ':!archive' ':!docs/reports' > "$OUT/hits_main.txt" 2>/dev/null
echo "1 tracked on origin/main: $(wc -l < "$OUT/hits_main.txt") hits"
for d in $LIVE; do
  timeout 900 grep -r -n -o -F -f "$OUT/needles.txt" -I \
    --include='*.sh' --include='*.py' --include='*.yaml' --include='*.yml' --include='*.json' --include='*.txt' \
    --include='*.md' --include='*.cfg' --include='*.toml' --include='*.ini' \
    --exclude-dir=checkpoints --exclude-dir=torch_extensions --exclude-dir=renders --exclude-dir=frames \
    --exclude-dir=images --exclude-dir=wandb --exclude-dir=__pycache__ --exclude-dir=huggingface "$d"
done > "$OUT/hits_live.txt" 2>/dev/null
echo "2 live experiment folders: $(wc -l < "$OUT/hits_live.txt") hits"
git -C "$REPO" ls-files --others --exclude-standard | grep -vE '^(exps|archive|third_party|playground)/' \
  | grep -E '\.(sh|py|ya?ml|json|txt|md|cfg|toml|ini)$' | sed "s|^|$REPO/|" > "$OUT/untracked.txt"
xargs -a "$OUT/untracked.txt" -d '\n' grep -n -o -H -F -f "$OUT/needles.txt" -I > "$OUT/hits_untracked.txt" 2>/dev/null
echo "3 untracked files of the shared checkout: $(wc -l < "$OUT/hits_untracked.txt") hits"
echo "most frequent matches (very short names such as 'ele' or 'wandb' are noise):"
cat "$OUT"/hits_*.txt | awk -F: '{print $NF}' | sort | uniq -c | sort -rn | head -15
