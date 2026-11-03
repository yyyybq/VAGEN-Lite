"""Reward line, 11-03 (the user: the rw_tc1 drift check, plus how large a turn credit is against the number of actions
in the turn): from the drift replay (turns.jsonl.gz of reward/drift_tc1.py, every 3rd update of rw_tc1 / rw_tc2, the
episodes that entered the update) per run, block of 25 updates and actions in the turn:
* the raw turn value (scale * (gamma phi_next - phi) + r, the "credit" field) and the answer tokens;
* the trained turn term 0.5 z (z: the value centred on the update mean and divided by its std) and its mass over the
  answer tokens (0.5 z x tokens; the loss weighs tokens, so this is how hard the turn pulls);
* rw_tc1 only: the episode term E (verl GRPO of the S0 return in the group, then standardised over the update) and
  its mass E x tokens.
    python -m reward.drift_turn_size --dir reward_runs/analysis/14_drift_tc1   (writes DIR/turn_size.json, .md)"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EPS = 1e-6


def bucket(n):
    return "6+" if n >= 6 else str(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="the drift_tc1 output folder (turns.jsonl.gz)")
    A = Path(ap.parse_args().dir)
    lines = [json.loads(l) for l in gzip.open(A / "turns.jsonl.gz", "rt")]
    by_update = defaultdict(list)
    for l in lines:
        by_update[(l["run"], l["update"])].append(l)
    cells = defaultdict(lambda: defaultdict(list))
    for (run, upd), tasks in sorted(by_update.items()):
        eps, turns = [], []
        for t in tasks:
            kept = [e for e in t["episodes"] if e.get("kept", True)]
            r = np.asarray([e["return_s0"] for e in kept], float)
            raw = (r - r.mean()) / (r.std(ddof=1) + EPS) if len(r) > 1 else np.zeros(len(r))
            for e, g in zip(kept, raw):
                eps.append(g)
                for tu in e["turns"]:
                    turns.append((len(eps) - 1, tu))
        E = np.asarray(eps)
        E = (E - E.mean()) / (E.std() + EPS) if E.std() > 0 else np.zeros_like(E)
        v = np.asarray([tu["credit"] for _, tu in turns], float)
        z = (v - v.mean()) / (v.std() + EPS)
        lo = (upd - 1) // 25 * 25 + 1
        block = "%d-%d" % (lo, lo + 24)
        for (i, tu), zz in zip(turns, z):
            cells[(run, block)][bucket(tu["actions"])].append(
                (tu["credit"], tu["tokens"], 0.5 * zz, 0.5 * zz * tu["tokens"], E[i], E[i] * tu["tokens"]))
    out = {}
    md = ["| 运行 | 更新 | 动作数 | 回合 | 原始逐回合值 | token | 0.5z | 0.5z × token | E | E × token |",
          "|" + " --- |" * 10]
    for (run, block), ks in sorted(cells.items(), key=lambda x: (x[0][0], int(x[0][1].split("-")[0]))):
        for k in sorted(ks, key=lambda s: int(s.rstrip("+"))):
            a = np.asarray(ks[k])
            tc1 = run == "rw_tc1"
            row = {"turns": len(a), "value": a[:, 0].mean(), "tokens": a[:, 1].mean(), "turn_term": a[:, 2].mean(),
                   "turn_mass": a[:, 3].mean(), "abs_turn_mass": np.abs(a[:, 3]).mean(),
                   "episode_term": a[:, 4].mean() if tc1 else None, "episode_mass": a[:, 5].mean() if tc1 else None}
            out.setdefault(run, {}).setdefault(block, {})[k] = {
                x: (round(float(y), 4) if y is not None else None) for x, y in row.items()}
            f = (lambda y: "—" if y is None else "%+.2f" % y)
            md.append("| %s | %s | %s | %d | %.3f | %.1f | %+.2f | %+.2f | %s | %s |" % (
                run, block, k, len(a), row["value"], row["tokens"], row["turn_term"], row["turn_mass"],
                f(row["episode_term"]), f(row["episode_mass"])))
    (A / "turn_size.json").write_text(json.dumps(out, indent=1))
    (A / "turn_size.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
