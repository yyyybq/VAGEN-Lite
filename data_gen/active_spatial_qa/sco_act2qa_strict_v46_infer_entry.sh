#!/usr/bin/env bash
# Inference-only Act->QA canary for the pinned Base / v46 pair.
# Reuses the frozen 20260919 visual bank. Does not render and does not write
# into that bank's directory. Completed prediction rows are kept and resumed.
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
OLD=${ROOT}/data_gen/active_spatial_qa/artifacts/act2qa_real_canary_20260919T0618
OUT=${ACT2QA_OUT:-${ROOT}/data_gen/active_spatial_qa/artifacts/act2qa_real_canary_strict_v46_20260925}
BANK_SRC=${ACT2QA_BANK:-${OLD}/frozen_visual_manifest.jsonl}
ELIGIBLE_SRC=${ACT2QA_ELIGIBLE:-${OLD}/eligible_ids.json}
BASE=${BASE_MODEL:-${ROOT}/exps/vagen_active_spatial/r1_h1_aoss_repair_20260905/r1_canonical_dev_eval32_20260915/model_restore/qwen25vl7b_pretrained_cc594898}
ACT=${ACT_MODEL:-${ROOT}/exps/vagen_active_spatial/r1_h1_aoss_repair_20260905/r1_canonical_dev_eval32_20260915/model_restore/v46_step250_hf}
TRAIN_CONFIG=${TRAIN_CONFIG:-${ROOT}/exps/vagen_active_spatial/qwen_v46_clean_d0pass_20260822_sco_renderer_r1_full/hydra_run/.hydra/config.yaml}

if [[ "$(readlink -f "${OUT}")" == "$(readlink -f "${OLD}")" ]]; then
  echo "[fatal] refusing to write into the frozen canary directory" >&2
  exit 2
fi

mkdir -p "${OUT}/eval" "${OUT}/logs"
exec > >(tee -a "${OUT}/worker.log") 2>&1
export PATH=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin:${PATH}
export PYTHONPATH=${ROOT}
export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VLLM_TORCH_COMPILE_LEVEL=0 TORCH_COMPILE_DISABLE=1 VLLM_ATTENTION_BACKEND=TORCH_SDPA
echo "[compiler] PATH=${PATH}"
which gcc g++ cc c++ || true
gcc --version || true

cleanup() {
  status=$?
  if [[ "${status}" -ne 0 ]]; then
    printf '{"exit_status":%s,"stopped_utc":"%s"}\n' "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUT}/worker_exit.json"
  fi
}
trap cleanup EXIT INT TERM

test -s "${BANK_SRC}"
test -s "${ELIGIBLE_SRC}"
test -r "${BASE}/config.json"
test -r "${ACT}/config.json"
test -r "${TRAIN_CONFIG}"

BANK="${OUT}/frozen_visual_manifest.jsonl"
ELIGIBLE="${OUT}/eligible_ids.json"
if [[ ! -s "${BANK}" ]]; then
  cp -n "${BANK_SRC}" "${BANK}"
fi
if [[ ! -s "${ELIGIBLE}" ]]; then
  cp -n "${ELIGIBLE_SRC}" "${ELIGIBLE}"
fi
cmp -s "${BANK_SRC}" "${BANK}"
cmp -s "${ELIGIBLE_SRC}" "${ELIGIBLE}"

"${PY}" - "${OUT}" "${BANK}" "${ELIGIBLE}" "${BASE}" "${ACT}" <<'PY'
import json, sys
from pathlib import Path
out, bank, eligible, base, act = sys.argv[1:]
rows = [json.loads(x) for x in open(bank) if x.strip()]
ids = json.loads(open(eligible).read())["eligible_ids"]
got = [r["sample_id"] for r in rows]
if got != ids:
    raise SystemExit("bank sample_ids differ from pinned eligible_ids")
if any(r.get("observability_validity") != "valid" or not r.get("public_observation", {}).get("image_path") for r in rows):
    raise SystemExit("pinned bank contains a non-observable row")
info = Path(out) / "JOB_INFO.json"
if not info.exists():
    info.write_text(json.dumps({
        "job_id": __import__("os").environ.get("SCO_JOB_ID", "unknown"),
        "output_dir": out,
        "bank": bank,
        "eligible_ids": eligible,
        "base": base,
        "act": act,
        "n": len(ids),
        "pair_label": "STRICT_MATCH",
        "resume": True,
        "renders": False,
    }, indent=2) + "\n")
print(json.dumps({"pinned_n": len(ids), "job_info_exists": True}))
PY

if [[ ! -s "${OUT}/model_identities.json" ]]; then
  "${PY}" "${ROOT}/data_gen/active_spatial_qa/finalize_act2qa.py" identities \
    --output-dir "${OUT}" \
    --base "${BASE}" \
    --act "${ACT}" \
    --act-source "${ACT}" \
    --train-config "${TRAIN_CONFIG}"
fi

complete=$("${PY}" - "${OUT}" "${ELIGIBLE}" <<'PY'
import json, sys
from pathlib import Path
out, eligible = sys.argv[1:]
ids = set(json.loads(open(eligible).read())["eligible_ids"])
root = Path(out)

def covered(path):
    if not path.exists():
        return False
    got = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            return False
        if rec.get("sample_id") in ids and "prediction" in rec:
            got.add(rec["sample_id"])
    return got == ids

base_ok = covered(root / "eval" / "base_predictions.jsonl")
act_ok = covered(root / "eval" / "act_predictions.jsonl")
exit_ok = False
exit_path = root / "worker_exit.json"
if exit_path.exists():
    try:
        exit_ok = json.loads(exit_path.read_text()).get("exit_status") == 0
    except Exception:
        exit_ok = False
table_ok = (root / "ACT2QA_TABLE.md").exists() and (root / "act2qa_table.json").exists()
flag = "complete" if base_ok and act_ok and exit_ok and table_ok else "incomplete"
print(flag)
print("base", int(base_ok), "act", int(act_ok), "exit", int(exit_ok), "table", int(table_ok), file=sys.stderr)
PY
)
echo "[stage] coverage ${complete}"
if [[ "${complete}" == complete ]]; then
  echo "[done] pinned Base/v46 inference and evaluation already complete"
  exit 0
fi

infer() {
  local name="$1" ckpt="$2" dest="$3"
  echo "[stage] ${name} inference"
  "${PY}" "${ROOT}/data_gen/active_spatial_qa/model_qa_eval.py" \
    --bank "${BANK}" \
    --eligible-ids "${ELIGIBLE}" \
    --checkpoint "${ckpt}" \
    --output "${dest}" \
    --resume \
    --backend "${ACT2QA_BACKEND:-vllm}" \
    --tp 1 \
    --gpu-memory-utilization 0.60
}

if [[ "${ACT2QA_BACKEND:-vllm}" == "transformers" && ! -s "${OUT}/eval/base_predictions.jsonl" ]]; then
  echo "[stage] Base smoke inference limit=1"
  "${PY}" "${ROOT}/data_gen/active_spatial_qa/model_qa_eval.py" \
    --bank "${BANK}" --eligible-ids "${ELIGIBLE}" --checkpoint "${BASE}" \
    --output "${OUT}/eval/base_predictions.jsonl" --resume --backend transformers --limit 1
fi
infer Base "${BASE}" "${OUT}/eval/base_predictions.jsonl"
infer Act "${ACT}" "${OUT}/eval/act_predictions.jsonl"

echo "[stage] Evaluation"
"${PY}" "${ROOT}/data_gen/active_spatial_qa/qa_eval.py" \
  --bank "${BANK}" \
  --predictions "${OUT}/eval/base_predictions.jsonl" \
  --output "${OUT}/eval/base_qa_eval.json" \
  --require-observable
"${PY}" "${ROOT}/data_gen/active_spatial_qa/qa_eval.py" \
  --bank "${BANK}" \
  --predictions "${OUT}/eval/act_predictions.jsonl" \
  --output "${OUT}/eval/act_qa_eval.json" \
  --require-observable
"${PY}" "${ROOT}/data_gen/active_spatial_qa/finalize_act2qa.py" table \
  --output-dir "${OUT}" \
  --bank "${BANK}" \
  --base-pred "${OUT}/eval/base_predictions.jsonl" \
  --act-pred "${OUT}/eval/act_predictions.jsonl"
printf '{"exit_status":0,"finished_utc":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUT}/worker_exit.json"
echo "[done] artifacts at ${OUT}"
