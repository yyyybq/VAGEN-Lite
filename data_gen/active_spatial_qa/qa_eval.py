#!/usr/bin/env python3
"""Frozen QA evaluator and cross-table writer.

Predictions are supplied separately so test-time evaluation never updates
weights or interacts with the environment.
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path

def parse_answer(text: str) -> str | None:
    # Accept only one unambiguous answer, with optional terminal punctuation.
    m = re.fullmatch(r"\s*(yes|no)[.!]?\s*", str(text or ""), re.I)
    return m.group(1).title() if m else None

def answer_metrics(golds, parsed) -> dict:
    """All gold rows, including missing/invalid predictions, stay in denominators."""
    golds, parsed = list(golds), list(parsed)
    if len(golds) != len(parsed):
        raise ValueError("gold/prediction lengths differ")
    pairs = list(zip(golds, parsed))
    rates = {}
    for label, name in (("Yes", "yes"), ("No", "no")):
        group = [(g, p) for g, p in pairs if g == label]
        rates[name + "_n"] = len(group)
        rates[name + "_accuracy"] = sum(g == p for g, p in group) / len(group) if group else None
    failures = sum(p not in ("Yes", "No") for p in parsed)
    present = [rates[name + "_accuracy"] for name in ("yes", "no") if rates[name + "_n"]]
    return {"n": len(pairs), "accuracy": sum(g == p for g, p in pairs) / len(pairs) if pairs else None,
            "balanced_accuracy": sum(present) / len(present) if present else None,
            "format_failures": failures, "format_failure_rate": failures / len(pairs) if pairs else 0.0,
            **rates}

def evaluate(bank: str, predictions: str | None, output: str, require_observable: bool = False) -> dict:
    rows = [json.loads(x) for x in Path(bank).read_text().splitlines() if x.strip()]
    excluded = 0
    if require_observable:
        kept = []
        for row in rows:
            if row.get("observability_validity") == "valid" and row.get("public_observation", {}).get("image_path"):
                kept.append(row)
            else:
                excluded += 1
        rows = kept
    pred = {}
    if predictions and Path(predictions).exists():
        for x in Path(predictions).read_text().splitlines():
            d = json.loads(x); pred[str(d.get("sample_id"))] = parse_answer(d.get("prediction", d.get("response", "")))
    groups = defaultdict(list)
    for row in rows:
        got = pred.get(row["sample_id"]); gold = row["private_answer"]
        groups[(row["task_type"], row["split"], row["public_observation"].get("observation_config", "single_image"))].append((gold, got))
    yes_total = sum(r.get("private_answer") == "Yes" for r in rows)
    no_total = sum(r.get("private_answer") == "No" for r in rows)
    result = {"model": "unknown" if not predictions else str(predictions), "groups": {}, "prediction_count": len(pred), "rows_total": len(rows) + excluded, "rows_evaluated": len(rows), "excluded_unobservable": excluded, "status": "EMPTY_BLOCKED" if not rows else "OK", "always_yes_accuracy": (yes_total / len(rows) if rows else None), "always_no_accuracy": (no_total / len(rows) if rows else None), "positive_fraction": (yes_total / len(rows) if rows else None)}
    for key, vals in groups.items():
        result["groups"]["|".join(key)] = answer_metrics((g for g, _ in vals), (p for _, p in vals))
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--bank",required=True); ap.add_argument("--predictions"); ap.add_argument("--output",required=True); ap.add_argument("--require-observable", action="store_true"); print(json.dumps(evaluate(**vars(ap.parse_args())),indent=2))
if __name__ == "__main__": main()
