#!/usr/bin/env python3
"""Linear probes on X0's questions (the user's decision of 2026-10-30, "按你拟的做"; design
reports/X0B_PD_LP_DESIGN_20261028.md): can the spatial answer be read linearly from a model's hidden state, even
where its reply is wrong?

    python downstream/x/probe_lp.py --questions X0_QUESTIONS.jsonl --hidden M=HIDDEN.npz [M2=...] --out R.json

HIDDEN.npz (q0_ask.py --hidden): per question the last prompt token's hidden state after the layers at 1/4, 1/2, 3/4
and all of the language model's depth.  Two targets per question:
    relation  what the question asks, independent of the option order: left_right "the first-named entity is to
              the left of the second", near_far "the first-named is closer", front_back "the first-named is
              farther forward"
    letter    the right option letter (A or B), what the model has to output
Per model, layer, kind (and all 510 together) and target: L2-regularised logistic regression (C = 1 on features
standardised within the training folds; the same objective as scikit-learn's default, solved by Newton's method in
the n-dimensional dual since there are far more features than questions), 5-fold cross-validation grouped by scene
(task, seed), so a scene is never in both the training and the test folds.  Reported: accuracy over the held-out
predictions with its 95% Wilson interval, and, paired by question, McNemar against the family's base (B for the
Qwen2.5 models, Q3B for the Qwen3 ones) at the same layer.  numpy only.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import x0_qa  # noqa: E402

FOLDS = 5
TARGETS = ("relation", "letter")


def labels(q: dict) -> dict:
    first = q["a"]["name"]
    if q["kind"] == "left_right":
        rel = q["answer_text"] == "left"
    else:
        rel = q["answer_text"] == first
    return {"relation": int(rel), "letter": int(q["answer"] == "A")}


def scene_folds(questions: list, k: int = FOLDS, seed: int = 0) -> list:
    """Fold index per question; all questions of a scene (task, seed) share one fold."""
    scenes = sorted({(q["task"], str(q["seed"])) for q in questions})
    random.Random(seed).shuffle(scenes)
    fold = {s: i % k for i, s in enumerate(scenes)}
    return [fold[(q["task"], str(q["seed"]))] for q in questions]


def logistic_dual(X, y, C: float = 1.0, iters: int = 50):
    """L2 logistic regression 0.5|w|^2 + C sum log(1 + exp(-y' (w.x + b))) with w = X^T alpha (n << d).  Returns
    (alpha, b).  Newton's method with backtracking on (alpha, b)."""
    import numpy as np

    n = X.shape[0]
    K = X @ X.T
    alpha, b = np.zeros(n), 0.0

    def loss(a, bb):
        z = K @ a + bb
        return C * np.sum(np.logaddexp(0.0, -(2 * y - 1) * z)) + 0.5 * a @ K @ a

    cur = loss(alpha, b)
    for _ in range(iters):
        z = K @ alpha + b
        p = 0.5 * (1.0 + np.tanh(0.5 * z))  # the logistic function without overflow
        g = p - y
        w = p * (1 - p)
        grad = np.concatenate([K @ (C * g + alpha), [C * g.sum()]])
        H = np.empty((n + 1, n + 1))
        KW = K * w  # K diag(w)
        H[:n, :n] = C * KW @ K + K
        H[:n, n] = H[n, :n] = C * KW.sum(axis=1)
        H[n, n] = C * w.sum()
        H[:n, :n] += 1e-9 * np.eye(n)
        step = -np.linalg.lstsq(H, grad, rcond=None)[0]
        t, accepted = 1.0, False
        while t > 1e-6:
            na, nb = alpha + t * step[:n], b + t * step[n]
            new = loss(na, nb)
            if new <= cur:
                accepted = True
                break
            t /= 2
        if not accepted:  # no step lowers the loss any more: converged
            break
        small = cur - new < 1e-10 * max(1.0, abs(cur))
        alpha, b, cur = na, nb, new
        if small:
            break
    return alpha, b


def cv_predict(X, y, folds, C: float = 1.0):
    """Held-out predictions (0/1) of the grouped cross-validation."""
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    folds = np.asarray(folds)
    pred = np.zeros(len(y), dtype=int)
    for f in sorted(set(folds.tolist())):
        tr, te = folds != f, folds == f
        if len(set(y[tr].tolist())) < 2:  # one class only in training: predict it
            pred[te] = int(y[tr][0])
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        alpha, b = logistic_dual(Xtr, y[tr], C)
        pred[te] = ((Xte @ Xtr.T) @ alpha + b > 0).astype(int)
    return pred


def probe(questions: list, h, layers, C: float = 1.0) -> dict:
    """{layer: {group: {target: {"correct": [bool per question of the group], ...}}}} for one model."""
    folds = scene_folds(questions)
    lab = [labels(q) for q in questions]
    groups = {"all": list(range(len(questions))),
              **{k: [i for i, q in enumerate(questions) if q["kind"] == k] for k in x0_qa.KINDS}}
    out = {}
    for li, layer in enumerate(layers):
        out[int(layer)] = {}
        for g, idx in groups.items():
            out[int(layer)][g] = {}
            for t in TARGETS:
                y = [lab[i][t] for i in idx]
                pred = cv_predict([h[i][li] for i in idx], y, [folds[i] for i in idx], C)
                correct = [int(p == yy) for p, yy in zip(pred.tolist(), y)]
                k, n = sum(correct), len(correct)
                out[int(layer)][g][t] = {"n": n, "correct_n": k, "accuracy": k / n if n else float("nan"),
                                         "ci": x0_qa.wilson(k, n), "ids": [questions[i]["id"] for i in idx],
                                         "correct": correct}
    return out


def reference_of(model: str):
    return None if model in ("B", "Q3B") else ("Q3B" if model.startswith("Q3") else "B")


def compare(res: dict) -> None:
    """Adds McNemar against the family's base at the same layer depth (the n-th of the four probed layers)."""
    for m, layers in res.items():
        ref = reference_of(m)
        if not ref or ref not in res:
            continue
        ref_layers = list(res[ref])
        for pos, layer in enumerate(layers):
            rl = ref_layers[pos]
            for g, targets in layers[layer].items():
                for t, row in targets.items():
                    other = dict(zip(res[ref][rl][g][t]["ids"], res[ref][rl][g][t]["correct"]))
                    only_m = sum(1 for i, c in zip(row["ids"], row["correct"]) if c and not other[i])
                    only_r = sum(1 for i, c in zip(row["ids"], row["correct"]) if not c and other[i])
                    row["vs_reference"] = {"reference": ref, "reference_layer": rl, "only_model": only_m,
                                           "only_reference": only_r, "p": x0_qa.mcnemar_exact(only_m, only_r)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--hidden", nargs="+", required=True, help="MODEL=HIDDEN.npz")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    import numpy as np

    questions = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
    res = {}
    for spec in args.hidden:
        model, path = spec.split("=", 1)
        z = np.load(path, allow_pickle=False)
        order = {i: n for n, i in enumerate(z["ids"].tolist())}
        h = z["h"][[order[q["id"]] for q in questions]].astype(np.float64)
        res[model] = probe(questions, h, z["layers"].tolist(), args.C)
        print(f"[probe] {model}: " + ", ".join(f"layer {L} {res[model][L]['all']['relation']['accuracy']:.3f}"
                                             for L in res[model]), flush=True)
    compare(res)
    slim = {m: {L: {g: {t: {k: v for k, v in r.items() if k not in ("ids", "correct")} for t, r in ts.items()}
                    for g, ts in gs.items()} for L, gs in ls.items()} for m, ls in res.items()}
    Path(args.out).write_text(json.dumps(slim, indent=1) + "\n")
    Path(args.out).with_suffix(".per_question.json").write_text(json.dumps(res) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
