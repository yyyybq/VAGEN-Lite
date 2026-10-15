"""Export questions to the training format of Act-SFT: LLaMA-Factory "sharegpt" records and a recipe.

    # questions straight from render/bank runs (replaced questions dropped with --exclude, as on the review site)
    python -m data_gen.interiorgs_qa.export_lf --run RUN1 --run RUN2 ... [--exclude RULES] --res 256 --out LF
    # questions chosen by the sampler (sample.py): one file per split
    python -m data_gen.interiorgs_qa.export_lf --selection SEL/train.jsonl --val-selection SEL/val.jsonl --res 256 --out LF

Act-SFT's export (scripts/prepare_active_spatial_sft.py, not changed here) writes ``{"messages": [...], "images":
[...]}`` with one ``<image>`` placeholder per image, LLaMA-Factory 0.9.3, template ``qwen2_vl``, and the recipe in
``exps/vagen_active_spatial/interiorgs_v10_train_20261014/lf/train.yaml``. This exporter writes the same format:

* single-image and multi-image questions: one user turn (the images, then the question text) and one assistant
  turn (the answer);
* video questions: one conversation per video; the 32 frames go in once as a ``<video>`` (a list of frame
  files, which LLaMA-Factory 0.9.3 accepts as a video) in the first user turn, and every following turn asks one
  question about that video;
* training phrasing: each question uses one of the original benchmark format and the three training wordings,
  chosen per question with a fixed seed (the held-out wording never enters training); options were already
  shuffled per wording when the question was made;
* answer: the option letter, or the number for numeric questions (every benchmark's extraction rule in
  scoring.py accepts it);
* images: the rendered PNG at the chosen resolution (512, or the 256 copy).

The recipe copies Act-SFT's recipe (same model, template, cutoff, optimiser settings) and only changes the
datasets, the output folder and the image/video pixel limits (so 512 inputs are not shrunk to LLaMA-Factory's
video default of 256 x 256).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from .qa import answer_target, excluded, parse_exclude, render

ACT_RECIPE = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_train_20261014/lf/train.yaml")
TRAIN_FAMILIES = ("orig", "t1", "t2", "t3")


def _jsonl(p: Path) -> List[Dict[str, Any]]:
    out = []
    for l in Path(p).read_text().splitlines():
        if l.strip():
            try:
                out.append(json.loads(l))
            except ValueError:
                continue
    return out


def load_runs(runs: Sequence[str], exclude: str = "") -> List[Tuple[Path, Dict[str, Any]]]:
    rules = parse_exclude(exclude)
    out = []
    for r in runs:
        rp = Path(r).resolve()
        for q in _jsonl(rp / "questions.jsonl"):
            if not excluded(rp, q, rules):
                out.append((rp, q))
    return out


def load_selection(path: str) -> List[Tuple[Path, Dict[str, Any]]]:
    """Sampler output: one line per question, {"run_dir": ..., "question": {...}}."""
    return [(Path(r["run_dir"]), r["question"]) for r in _jsonl(Path(path))]


def image_path(run: Path, rel: str, res: int) -> str:
    p = run / rel
    if res == 256:
        p = p.with_name(p.stem + "_256.png")
    return str(p)


def pick_family(q: Dict[str, Any], seed: int, families: Sequence[str]) -> str:
    fams = [f for f in families if f in q["variants"]]
    return random.Random(f"{seed}|{q['qid']}").choice(fams)


def answer_text(q: Dict[str, Any], fam: str) -> str:
    """The answer in the form the wording's instruction asks for (qa.answer_target)."""
    return answer_target(q, fam)


def video_key(q: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    return q["scene_id"], tuple(q["views"])


def build_records(items: Sequence[Tuple[Path, Dict[str, Any]]], res: int, seed: int,
                  families: Sequence[str] = TRAIN_FAMILIES) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """LLaMA-Factory records and an audit row per record (question ids, kind, image inputs)."""
    records, audit = [], []
    videos: "OrderedDict[Tuple[str, Tuple[str, ...]], List[Tuple[Path, Dict[str, Any]]]]" = OrderedDict()
    for run, q in items:
        if q["media"] == "video":
            videos.setdefault(video_key(q), []).append((run, q))
            continue
        fam = pick_family(q, seed, families)
        imgs = [image_path(run, p, res) for p in q["extra"]["images"]]
        user = "<image>" * len(imgs) + "\n" + render(q, fam)
        records.append({"messages": [{"role": "user", "content": user},
                                     {"role": "assistant", "content": answer_text(q, fam)}],
                        "images": imgs, "videos": []})
        audit.append({"kind": q["media"], "qids": [q["qid"]], "groups": [q["group"]], "families": [fam],
                      "image_inputs": len(imgs), "scene_id": q["scene_id"]})
    for key, qs in videos.items():
        run0, q0 = qs[0]
        frames = [image_path(run0, p, res) for p in q0["extra"]["images"]]
        msgs, fams = [], []
        for i, (run, q) in enumerate(qs):
            fam = pick_family(q, seed, families)
            fams.append(fam)
            text = render(q, fam)
            msgs.append({"role": "user", "content": ("<video>\n" + text) if i == 0 else text})
            msgs.append({"role": "assistant", "content": answer_text(q, fam)})
        records.append({"messages": msgs, "images": [], "videos": [frames]})
        audit.append({"kind": "video", "qids": [q["qid"] for _, q in qs], "groups": [q["group"] for _, q in qs],
                      "families": fams, "image_inputs": len(frames), "scene_id": q0["scene_id"]})
    return records, audit


DATASET_INFO = {"formatting": "sharegpt", "columns": {"messages": "messages", "images": "images", "videos": "videos"},
                "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                         "assistant_tag": "assistant", "system_tag": "system"}}


def write_lf(out: Path, splits: Dict[str, List[Dict[str, Any]]], res: int, recipe: Path = ACT_RECIPE) -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    info = {}
    for name, recs in splits.items():
        (out / f"qa_{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
        info[f"qa_{name}"] = dict(DATASET_INFO, file_name=f"qa_{name}.jsonl")
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    cfg = yaml.safe_load(recipe.read_text())
    cfg.update({"dataset_dir": str(out), "dataset": "qa_train", "output_dir": str(out / "model"),
                "image_max_pixels": res * res, "video_max_pixels": res * res})
    if "val" in splits and splits["val"]:
        cfg["eval_dataset"] = "qa_val"
    else:
        cfg.pop("eval_dataset", None)
        cfg["do_eval"] = False
        cfg["eval_strategy"] = "no"
    (out / "train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], help="render/bank run folders (questions.jsonl)")
    ap.add_argument("--exclude", default="", help="replaced questions, as in site.py")
    ap.add_argument("--selection", default="", help="sampler output for training (train.jsonl)")
    ap.add_argument("--val-selection", default="", help="sampler output for the validation split")
    ap.add_argument("--val-every", type=int, default=0,
                    help="without --val-selection: every N-th record goes to qa_val (smoke tests only)")
    ap.add_argument("--res", type=int, choices=(256, 512), default=256)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--recipe", default=str(ACT_RECIPE))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = load_selection(a.selection) if a.selection else load_runs(a.run, a.exclude)
    recs, audit = build_records(items, a.res, a.seed)
    splits: Dict[str, List[Dict[str, Any]]] = {"train": recs}
    if a.val_selection:
        vrecs, vaudit = build_records(load_selection(a.val_selection), a.res, a.seed)
        splits["val"] = vrecs
        audits = {"train": audit, "val": vaudit}
    elif a.val_every:
        splits = {"train": [r for i, r in enumerate(recs) if i % a.val_every], "val": recs[::a.val_every]}
        audits = {"train": [x for i, x in enumerate(audit) if i % a.val_every], "val": audit[::a.val_every]}
    else:
        audits = {"train": audit}
    out = Path(a.out)
    write_lf(out, splits, a.res, Path(a.recipe))
    for name, aud in audits.items():   # line i describes record i of qa_<name>.jsonl
        (out / f"audit_{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in aud))
    allaud = [x for aud in audits.values() for x in aud]
    summary = {"res": a.res, "records": {k: len(v) for k, v in splits.items()},
               "questions": sum(len(x["qids"]) for x in allaud),
               "by_kind": dict(Counter(x["kind"] for x in allaud)),
               "image_inputs": {k: sum(x["image_inputs"] for x in aud) for k, aud in audits.items()}}
    (out / "export_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
