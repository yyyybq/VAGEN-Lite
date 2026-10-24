# Sourced by backbone_load.sh / backbone_smoke.sh after the StarVLA env (its python):  starvla_entry START_PATH
# sets ENTRY to the Cambrian backbone entry (model_matrix/downstream/cambrian_entry.py) when the start is a Cambrian-S
# model, with the line's cache (SigLIP2) and the Cambrian source; for every other start ENTRY stays empty and the
# commands run as before.  Use as:  python ${ENTRY[@]+"${ENTRY[@]}"} SCRIPT ARGS
starvla_entry() {
  ENTRY=()
  if python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1] + "/config.json")).get("model_type") == "cambrian_qwen" else 1)' "$1" 2>/dev/null; then
    ENTRY=("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-models/model_matrix/downstream/cambrian_entry.py")
    export HF_HUB_CACHE=/mnt/umm/shared_model/vagen-lite-models/hub SIGLIP_CACHE=/mnt/umm/shared_model/vagen-lite-models/hub
    export CAMBRIAN_SRC=/mnt/umm/users/yinbaiqiao/cambrian-s
    echo "[starvla_entry] Cambrian-S start: ${ENTRY[0]}"
  fi
}
