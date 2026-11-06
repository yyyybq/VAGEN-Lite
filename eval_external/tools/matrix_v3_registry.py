"""Write adapters/registry_matrix_v3.yaml: the matrix-v3 Qwen3-VL-8B models for the QA transfer evaluation
(task of 2026-11-05), with every checkpoint path read from the matrix's own handover records, never built here.

    python -m eval_external.tools.matrix_v3_registry            # writes the file, prints the batches

Sources (their sha256 and mtime go into the file):
  HANDOVER_MAIN   C, D, old E and C0: runs[].prereg.model, sft_starts, weights_read_only.sft
  HANDOVER_CELLS  B', D_noP0, Cr and E2: sft{}.model, runs[].prereg.model
The raw model is the shared snapshot every matrix SFT started from (train.yaml model_name_or_path); its weights are
root-only (mode 0600), so its view and hash are built inside an sco job.

Pre-registered reading (the handovers' "prereg"): each RL's update-300 checkpoint, the last checkpoint of a run a
stop line ended, and for a run that ended before its first checkpoint (m3cs1d) its SFT start.  Old E (archived
2026-11-03, exploratory) is listed with archived: true and left out of the batches.

Batches (the user's plan): 1 raw + C0, C, D, E2 SFT; 2 Cr, D_noP0, B' SFT + every seed-20261019 S0/V4 RL;
3 the seed-20261021 RL.  An RL whose pre-registered model is its SFT start adds no model.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "adapters" / "registry_matrix_v3.yaml"
HANDOVER_MAIN = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data/docs/transition_qa/matrix_v3/handover_main_20261103.json")
HANDOVER_CELLS = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/handover_cells_20261104/"
                      "handover_cells_20261104.json")
RAW = {"code": "qwen3vl-8B", "repo": "Qwen/Qwen3-VL-8B-Instruct", "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
       "path": "/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
               "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"}
SEEDS = {1: 20261019, 2: 20261021, 20261019: 20261019, 20261021: 20261021}
ARCHIVED = {"E"}
# the SFT start points the user listed (2026-11-05); the handover paths must equal these
USER_SFT = {
    "C0": "/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_sft_20261026/lf_q3formal/model/checkpoint-1232",
    "C": "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/C/model/checkpoint-1204",
    "D": "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/D/model/checkpoint-3987",
    "Cr": "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/Cr/model/checkpoint-2107",
    "D_noP0": "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/D_noP0/model/checkpoint-2272",
    "B'": "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/B/model/checkpoint-1870",
    "E2": "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/E2/model/checkpoint-3010",
}
CELL_NOTE = {
    "C0": "P2q 9,802 rows (the model line's Qwen3 formal SFT, edge scenes kept), epoch 8",
    "C": "P0 + P2q 9,075, epoch 4",
    "D": "P0 + P1 + P2q 9,075, epoch 9",
    "Cr": "C retrained (same data, recipe and seed), epoch 7",
    "D_noP0": "P1 + P2q 9,075, epoch 8",
    "B'": "P0 + 1,795 mainline format dialogues, epoch 10",
    "E2": "P0 + P1, no action trajectories, epoch 10 (QA upper bound; no RL)",
}


def _stamp(p: Path) -> dict:
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "mtime_utc": datetime.datetime.fromtimestamp(p.stat().st_mtime, datetime.timezone.utc).isoformat()}


def build() -> dict:
    main, cells = json.loads(HANDOVER_MAIN.read_text()), json.loads(HANDOVER_CELLS.read_text())
    models = {RAW["code"]: {"source": "path", "path": RAW["path"], "repo": RAW["repo"], "revision": RAW["revision"],
                            "family": "qwen3_vl", "params": "8B", "cell": "raw", "kind": "raw",
                            "note": "raw Qwen3-VL-8B-Instruct (shared snapshot; weights root-only, read in sco jobs)"}}
    sft = {}  # cell -> code
    for key, path in main["weights_read_only"]["sft"].items():  # "C (epoch 4)", "C0 / E start (epoch 8)"
        cell = key.split()[0]
        sft[cell] = (main["sft_starts"][cell], path)
    for cell, rec in cells["sft"].items():
        sft[cell] = (rec["result"], rec["model"])
    for cell, (code, path) in sft.items():
        if USER_SFT.get(cell) != path:
            raise SystemExit(f"{cell}: handover SFT path {path} differs from the user's {USER_SFT.get(cell)}")
        models[code] = {"source": "path", "path": path, "family": "qwen3_vl", "params": "8B", "cell": cell,
                        "kind": "sft", "note": f"{cell} SFT: {CELL_NOTE[cell]}"}
    runs = []
    for r in main["runs"]:
        runs.append({"cell": r["group"], "seed": SEEDS[r["seed"]], "side": r["side"], "run": r["run"],
                     "prereg": r["prereg"]["result"], "path": r["prereg"]["model"], "sft": main["sft_starts"][r["group"]],
                     "from": HANDOVER_MAIN.name})
    for r in cells["runs"]:
        runs.append({"cell": r["cell"], "seed": SEEDS[r["seed"]], "side": r["side"], "run": r["run"],
                     "prereg": r["prereg"]["result"], "path": r["prereg"]["model"], "sft": sft[r["cell"]][0],
                     "from": HANDOVER_CELLS.name})
    for r in runs:
        r["archived"] = r["cell"] in ARCHIVED
        r["prereg_is_sft"] = r["prereg"] == r["sft"]
        if r["prereg_is_sft"]:
            if models[r["sft"]]["path"] != r["path"]:
                raise SystemExit(f"{r['run']}: prereg is its SFT start but the paths differ")
            continue
        if not Path(r["path"], "config.json").is_file():
            raise SystemExit(f"{r['run']}: no config.json in {r['path']}")
        if r["prereg"] in models:
            raise SystemExit(f"{r['prereg']} listed twice")
        models[r["prereg"]] = {"source": "path", "path": r["path"], "family": "qwen3_vl", "params": "8B",
                               "cell": r["cell"], "kind": "rl", "seed": r["seed"], "side": r["side"],
                               "sft_start": r["sft"], "archived": r["archived"],
                               "note": f"{r['cell']} RL {r['side']} seed {r['seed']} ({r['run']}), pre-registered "
                                       f"checkpoint {r['prereg']}, from {r['sft']}"}
    live = [r for r in runs if not r["archived"]]
    batches = {
        1: [RAW["code"], sft["C0"][0], sft["C"][0], sft["D"][0], sft["E2"][0]],
        2: [sft["Cr"][0], sft["D_noP0"][0], sft["B'"][0]]
           + [r["prereg"] for r in live if r["seed"] == 20261019 and not r["prereg_is_sft"]],
        3: [r["prereg"] for r in live if r["seed"] == 20261021 and not r["prereg_is_sft"]],
    }
    return {"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "sources": [_stamp(HANDOVER_MAIN), _stamp(HANDOVER_CELLS)], "models": models,
            "matrix": {"runs": runs, "batches": batches}}


def main() -> None:
    reg = build()
    head = ("# Written by eval_external/tools/matrix_v3_registry.py from the matrix-v3 handover records (sources below);\n"
            "# do not edit by hand.  Paths are the handovers' own; every checkpoint is read-only.\n")
    tmp = OUT.with_suffix(".yaml.tmp")
    tmp.write_text(head + yaml.safe_dump(reg, sort_keys=False, allow_unicode=True, width=120))
    os.replace(tmp, OUT)
    for b, codes in reg["matrix"]["batches"].items():
        print(f"batch {b}: {len(codes)} models: {' '.join(codes)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
