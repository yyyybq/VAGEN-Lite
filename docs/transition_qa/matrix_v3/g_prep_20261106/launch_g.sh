#!/usr/bin/env bash
# Group G's SFT (the user's 甲 of 2026-11-06): copy recipe G (0-1000 build) to exps/matrix_v3/sft/G with only the two
# paths of train.yaml changed, check it, then start the SFT-only chain and the cells' watcher (m3_cells.py, which
# for G stops at "accepted": no pick, no RL).  Stops at the first failed check.
set -euo pipefail
M=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix
X=$M/exps/matrix_v3
SRC=$X/sft_mixture_g_norm1000_20261106/G
DST=$X/sft/G
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
test ! -e "$DST" || { echo "$DST exists"; exit 1; }
test ! -e "$X/m3_g_sft/plan.json" || { echo "m3_g_sft/plan.json exists"; exit 1; }
test -z "$(git -C "$M" status --porcelain --untracked-files=no)" || { echo "tracked changes in $M"; exit 1; }
"$PY" - "$SRC" <<'PY'
import json, sys
from pathlib import Path
src = Path(sys.argv[1])
rec = json.loads((src / "recipe.json").read_text())
assert rec["config"] == "matrix_v3_qwen3_g_norm1000_20261106", rec["config"]
assert rec["counts"] == {"P0": 10144, "P1": 9075, "P2t": 9075, "Pg": 9075}, rec["counts"]
head = "Respond with the targets (0 to 1000 across the image), then the action:"
for f in ("P2t_train.jsonl", "P2t_val.jsonl"):
    rows = [json.loads(l) for l in open(src / f)]
    assert all(any(m["role"] == "user" and head in m["content"] for m in r["messages"]) for r in rows), f
    assert not any("pixels of the 256 x 256 image" in m["content"] for r in rows for m in r["messages"]), f
print("recipe checked:", rec["config"], rec["counts"])
PY
mkdir "$DST"
for f in "$SRC"/*; do
  b=$(basename "$f")
  [ "$b" = train.yaml ] && continue
  [ -d "$f" ] && { echo "unexpected folder $f"; exit 1; }
  cp -p "$f" "$DST/$b"
done
sed -e "s#^dataset_dir: $SRC\$#dataset_dir: $DST#" -e "s#^output_dir: $SRC/model\$#output_dir: $DST/model#" "$SRC/train.yaml" > "$DST/train.yaml"
diff "$SRC/train.yaml" "$DST/train.yaml" || true
test "$(diff "$SRC/train.yaml" "$DST/train.yaml" | grep -c '^[<>]')" = 4 || { echo "train.yaml: not exactly the two paths changed"; exit 1; }
grep -q "^dataset_dir: $DST\$" "$DST/train.yaml" && grep -q "^output_dir: $DST/model\$" "$DST/train.yaml"
( cd "$SRC" && sha256sum $(ls | grep -v '^train.yaml$') ) > /tmp/yb_msgs/g/sha_src.txt
( cd "$DST" && sha256sum $(ls | grep -v '^train.yaml$') ) > /tmp/yb_msgs/g/sha_dst.txt
cmp /tmp/yb_msgs/g/sha_src.txt /tmp/yb_msgs/g/sha_dst.txt
"$PY" - "$SRC" "$DST" "$X/sft/copied_G_20261106.json" "$(git -C "$M" rev-parse HEAD)" <<'PY'
import hashlib, json, sys, time
from pathlib import Path
src, dst, out, head = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(dst.iterdir()) if f.is_file()}
Path(out).write_text(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "from": str(src), "to": str(dst),
                                 "commit": head, "files_sha256": files,
                                 "note": "recipe G (targets_norm1000, Pg 9,075; the user's 甲 of 2026-11-06), data files byte "
                                         "for byte, train.yaml with dataset_dir and output_dir changed"}, indent=1) + "\n")
print("copy recorded:", out)
PY
cd "$M"
nice -n 19 "$PY" exps/matrix_v3/watch/m3_cells.py start-sft G
nohup nice -n 19 "$PY" exps/matrix_v3/watch/m3_cells.py watch G > "$X/watch/cells_watch_g.out" 2>&1 < /dev/null &
echo "watcher pid $!"
sleep 5
cat "$X/m3_g_sft/cell_state.json"
tail -3 "$X/watch/cells.log"
