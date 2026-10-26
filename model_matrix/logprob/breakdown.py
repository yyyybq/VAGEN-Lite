"""Where the rollout-vs-trainer log-probability difference sits (the model line, 2026-10-26, the user's decision 1),
from the per-token files of vagen/logprob_tokens.py (RUN's logprob_audit/tokens/step_NNNN.npz).

For every token: d = trainer log-probability - vLLM log-probability (the new/old probability ratio of the first
PPO pass is exp(d)).  Per group of tokens: count, mean |d|, mean d, and the share with |exp(d) - 1| > 5% (the
audit's frac_abs_ratio_minus_1_gt_0p05).  Groups:
    turn            first answer of the episode / later answers
    place           the answer's first token / its 2nd-5th / the rest
    image distance  sequence positions back to the nearest image token: < 50 / 50-99 / 100-199 / >= 200
    kind            format tokens (<action>, </action>, |, <|im_end|>) / the rest (action names, text), by the
                    tokenizer's decoding

    python -m model_matrix.logprob.breakdown --tokenizer TOK --run NAME=TOKENS_DIR [--run NAME=DIR ...] \
        [--steps 2-10] --out OUT

writes OUT.json and OUT.md (one column per run).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

FORMAT_PIECES = ("<action>", "</action>", "|", "<|im_end|>")
DISTANCE_EDGES = (50, 100, 200)


def load(tokens_dir: Path, steps: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
    files = sorted(Path(tokens_dir).glob("step_*.npz"))
    files = [f for f in files if ".tmp" not in f.name]
    if steps is not None:
        files = [f for f in files if int(f.stem.split("_")[1]) in set(steps)]
    if not files:
        raise SystemExit(f"no per-token files in {tokens_dir}")
    parts = [dict(np.load(f)) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def stats(d: np.ndarray) -> Dict[str, float]:
    if d.size == 0:
        return {"tokens": 0, "mean_abs": None, "mean": None, "share_ratio_off_5pct": None}
    return {"tokens": int(d.size), "mean_abs": round(float(np.abs(d).mean()), 5), "mean": round(float(d.mean()), 5),
            "share_ratio_off_5pct": round(float((np.abs(np.exp(d) - 1.0) > 0.05).mean()), 4)}


def format_ids(tokenizer, token_ids: np.ndarray) -> np.ndarray:
    """A mask of the tokens whose decoding is one of the format pieces (spaces stripped)."""
    unique = np.unique(token_ids)
    fmt = {int(t) for t in unique if tokenizer.decode([int(t)]).strip() in FORMAT_PIECES}
    return np.isin(token_ids, list(fmt))


def groups(rec: Dict[str, np.ndarray], is_format: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    dist = rec["image_distance"]
    out = {"all": np.ones(rec["token"].shape[0], dtype=bool),
           "turn: first answer": rec["turn"] == 0, "turn: later answers": rec["turn"] > 0,
           "place: first token": rec["pos_in_turn"] == 0,
           "place: 2nd-5th token": (rec["pos_in_turn"] >= 1) & (rec["pos_in_turn"] <= 4),
           "place: the rest": rec["pos_in_turn"] >= 5}
    lo = 0
    for hi in DISTANCE_EDGES:
        out[f"image distance: {lo}-{hi - 1}"] = (dist >= lo) & (dist < hi)
        lo = hi
    out[f"image distance: >= {lo}"] = dist >= lo
    out["image distance: no image before"] = dist < 0
    if is_format is not None:
        out["kind: format tokens"] = is_format
        out["kind: the rest"] = ~is_format
    return out


def breakdown(rec: Dict[str, np.ndarray], is_format: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    d = rec["trainer"].astype(np.float64) - rec["rollout"].astype(np.float64)
    return {name: stats(d[mask]) for name, mask in groups(rec, is_format).items()}


def parse_steps(text: str) -> Optional[List[int]]:
    if not text:
        return None
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def report(results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    runs = list(results)
    head = ["组"] + [f"{r}：token 数" for r in runs] + [f"{r}：平均 \\|差\\|" for r in runs] + \
           [f"{r}：概率比偏离超过 5%" for r in runs]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    fmt = lambda v, p: "—" if v is None else f"{v:.{p}f}"  # noqa: E731
    for g in results[runs[0]]:
        cells = [g] + [str(results[r][g]["tokens"]) for r in runs]
        cells += [fmt(results[r][g]["mean_abs"], 4) for r in runs]
        cells += [fmt(None if results[r][g]["share_ratio_off_5pct"] is None
                      else 100 * results[r][g]["share_ratio_off_5pct"], 1) + "%" for r in runs]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=TOKENS_DIR")
    ap.add_argument("--tokenizer", default=None, help="a tokenizer folder for the format-token group")
    ap.add_argument("--steps", default="", help="updates to use, e.g. 2-10 (default: all)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    results = {}
    for item in a.run:
        name, _, path = item.partition("=")
        rec = load(Path(path), parse_steps(a.steps))
        results[name] = breakdown(rec, format_ids(tok, rec["token"]) if tok is not None else None)
    a.out.with_suffix(".json").write_text(json.dumps({"steps": a.steps or "all", "runs": results}, indent=1))
    a.out.with_suffix(".md").write_text(report(results), encoding="utf-8")
    print(report(results))


if __name__ == "__main__":
    main()
