"""Download and unpack what a QA benchmark needs, on the login node, so that the sco jobs can run offline.

    python -m eval_external.qa_transfer.prepare_data VSI-Bench SAT-Real            # with network
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m eval_external.qa_transfer.prepare_data --check VSI-Bench

VLMEvalKit benchmarks: the question table must already be in $LMUData (tools/fetch_assets.py datasets
easi_tsv); building the dataset then downloads and unpacks the media (VSI-Bench videos into $HF_HOME/hub).
lmms-eval benchmarks: building the task downloads its Hugging Face dataset into $HF_HOME.  Each run writes
ROOT/data_prep/<bench>.json with the item count and the dataset revisions found in the cache.
On this cluster run it without the login-node proxy and with HF_ENDPOINT=https://hf-mirror.com.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

from eval_external.qa_transfer.benchmarks import BENCHMARKS, LMMS_INCLUDE

ROOT = Path("/mnt/umm/users/yinbaiqiao/eval_external_assets")


def hub_revisions() -> dict:
    hub = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    out = {}
    for d in sorted(hub.glob("datasets--*")):
        ref = d / "refs" / "main"
        out[d.name[len("datasets--"):].replace("--", "/")] = ref.read_text().strip() if ref.exists() else None
    return out


def prepare_vlmevalkit(bench: dict) -> dict:
    from vlmeval.dataset import build_dataset
    from vlmeval.dataset.video_dataset_config import supported_video_datasets

    name = bench["data"]
    ds = supported_video_datasets[name]() if name in supported_video_datasets else build_dataset(name)
    return {"items": len(ds.data), "data_file": getattr(ds, "data_file", None),
            "data_root": getattr(ds, "data_root", None)}


def prepare_lmms(bench: dict) -> dict:
    from lmms_eval.tasks import TaskManager, get_task_dict

    tasks = get_task_dict([bench["task"]], TaskManager(include_path=str(LMMS_INCLUDE)))
    task = tasks[bench["task"]]
    docs = task.test_docs() if task.has_test_docs() else task.validation_docs()
    return {"items": len(docs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benches", nargs="+", choices=sorted(BENCHMARKS))
    ap.add_argument("--check", action="store_true", help="only load (run with HF offline to prove jobs can)")
    args = ap.parse_args()
    out_dir = ROOT / "data_prep"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.benches:
        bench = BENCHMARKS[name]
        info = prepare_vlmevalkit(bench) if bench["backend"] == "vlmevalkit" else prepare_lmms(bench)
        info.update({"bench": name, "backend": bench["backend"], "expected_items": bench["items"],
                     "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
                     "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     "hub_dataset_revisions": hub_revisions()})
        path = out_dir / f"{name.replace('@', '_at_')}{'_check' if args.check else ''}.json"
        path.write_text(json.dumps(info, indent=1, default=str) + "\n")
        flag = "OK" if info["items"] == bench["items"] else "ITEM COUNT DIFFERS"
        print(f"{name}: {info['items']} items (expected {bench['items']}) {flag} -> {path}")


if __name__ == "__main__":
    main()
