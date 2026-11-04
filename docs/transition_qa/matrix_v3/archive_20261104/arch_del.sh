# Right before phase 2: no live sco job names the B'/D_noP0 SFT folders, their SFT and acceptance chains are done, no
# login-node process names them; then phase 2 (the tool re-checks every size and deletes) in the background.
set -euo pipefail
SCO=/mnt/umm/users/yinbaiqiao/.sco/bin/sco
M=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3
R=$M/sft/archive_20261104
$SCO acp jobs list --workspace-name=aigc --page-size 500 -o json > $R/jobs_before_delete.json
python3 - "$R/jobs_before_delete.json" <<'EOF'
import json, re, sys
jobs = json.load(open(sys.argv[1]))
live = [j for j in jobs if str(j.get("state")) not in ("SUCCEEDED", "FAILED", "STOPPED", "DELETED", "COMPLETED")]
pat = re.compile(r"matrix_v3/sft/(B|D_noP0)/model")
hits = sorted({j.get("name") for j in live if pat.search(json.dumps(j))})
print("live jobs", len(live), "naming B/D_noP0 SFT:", hits)
sys.exit(1 if hits else 0)
EOF
for t in m3_b_sft m3_dn_sft m3_b_sft_accept m3_dn_sft_accept; do
  s=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('state'))" $M/$t/progress.json)
  echo "$t $s"; [ "$s" = done ]
done
for p in /proc/[0-9]*; do c=$(tr "\0" " " < $p/cmdline 2>/dev/null || true)
  case "$c" in *sft/B/mode[l]*|*sft/D_noP0/mode[l]*) echo "process ${p#/proc/} names the folders: ${c:0:120}"; exit 1;; esac; done
echo "checks passed $(date -u +%FT%TZ)"
cd $R; nohup bash /tmp/yb_msgs/arch_run.sh delete >> $R/delete.log 2>&1 < /dev/null &
echo "phase 2 started, pid $!"
