"""Token counts of an exported dataset under the real training preprocessing (LLaMA-Factory 0.9.3).

    python -m data_gen.interiorgs_qa.lf_tokens --lf LF [--split train] --out LF/tokens_train.json

Runs LLaMA-Factory's own data pipeline on the recipe in LF/train.yaml (the same calls as its SFT workflow:
load_tokenizer -> get_template_and_fix_tokenizer -> get_dataset(stage="sft")): Qwen2.5-VL tokenizer, the
qwen2_vl template, and the image / video processor at the recipe's pixel limits. Per record it counts

* total tokens (the input_ids the model sees, after the recipe's cutoff),
* loss tokens (labels that are not ignored: the assistant answers and their end-of-turn token),
* visual tokens (image and video placeholder tokens after the processor expanded them),

and compares them with the estimates stored on each question (qa.count_tokens), grouped by record kind.
No GPU and no model weights are loaded.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import yaml


def lf_dataset(lf: Path, split: str):
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    cfg = yaml.safe_load((lf / "train.yaml").read_text())
    cfg.update({"dataset": f"qa_{split}", "output_dir": tempfile.mkdtemp(prefix="lf_tokens_"), "bf16": False,
                "do_eval": False, "eval_strategy": "no", "report_to": "none", "overwrite_cache": True,
                "preprocessing_num_workers": 8})
    cfg.pop("eval_dataset", None)
    model_args, data_args, training_args, _, _ = get_train_args(cfg)
    tok = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tok["tokenizer"], data_args)
    ds = get_dataset(template, model_args, data_args, training_args, stage="sft", **tok)["train_dataset"]
    return ds, tok["tokenizer"], cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lf", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--questions", action="append", default=[],
                    help="questions.jsonl files holding the per-question estimates (optional)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lf = Path(a.lf)
    ds, tokenizer, cfg = lf_dataset(lf, a.split)
    audit = [json.loads(l) for l in (lf / f"audit_{a.split}.jsonl").read_text().splitlines() if l.strip()]
    est: Dict[str, Dict[str, Any]] = {}
    for f in a.questions:
        for l in Path(f).read_text().splitlines():
            try:
                q = json.loads(l)
            except ValueError:
                continue
            est[q["qid"]] = q.get("tokens") or {}
    res = int(round(cfg["image_max_pixels"] ** 0.5))
    pad_ids = {tokenizer.convert_tokens_to_ids(t) for t in ("<|image_pad|>", "<|video_pad|>")}
    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        ids, labels = ex["input_ids"], ex["labels"]
        rec = {"total": len(ids), "loss": sum(1 for x in labels if x != -100),
               "visual": sum(1 for x in ids if x in pad_ids)}
        if i < len(audit):
            au = audit[i]
            rec.update(kind=au["kind"], n_questions=len(au["qids"]), image_inputs=au["image_inputs"])
            e = [est[q][str(res)] for q in au["qids"] if q in est and str(res) in est[q]]
            if e:
                # estimate of the record: visual tokens once (a video conversation shows the frames once), text of
                # every question, plus each question's answer
                rec["est_visual"] = e[0]["visual"]
                rec["est_total_per_question_sum"] = sum(x["total"] for x in e)
                rec["est_total"] = e[0]["total"] + sum(x["total"] - x["visual"] for x in e[1:])
        rows.append(rec)
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r.get("kind", "?")].append(r)

    def mean(xs):
        return round(statistics.mean(xs), 1) if xs else None

    summary = {"split": a.split, "resolution": res, "records": len(rows), "audit_records": len(audit),
               "cutoff_len": cfg.get("cutoff_len"),
               "truncated_records": sum(1 for r in rows if r["total"] >= cfg.get("cutoff_len", 10 ** 9)),
               "by_kind": {}}
    for k, rs in by.items():
        summary["by_kind"][k] = {
            "records": len(rs), "questions": sum(r.get("n_questions", 1) for r in rs),
            "total_mean": mean([r["total"] for r in rs]), "total_max": max(r["total"] for r in rs),
            "loss_mean": mean([r["loss"] for r in rs]), "visual_mean": mean([r["visual"] for r in rs]),
            "est_visual_mean": mean([r["est_visual"] for r in rs if "est_visual" in r]),
            "est_total_mean": mean([r["est_total"] for r in rs if "est_total" in r]),
            "image_inputs_mean": mean([r.get("image_inputs", 0) for r in rs]),
        }
    summary["all"] = {"total": sum(r["total"] for r in rows), "loss": sum(r["loss"] for r in rows),
                      "visual": sum(r["visual"] for r in rows)}
    Path(a.out).write_text(json.dumps({"summary": summary, "records": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
