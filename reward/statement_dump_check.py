"""Reward line, phase 2 direction 3: check the estimator's dump (vagen/custom_advantage/turn_credit.py, settings
dump_dir / dump_updates) of a statement-credit run (turn_term "statement") against what PLAN_DIR3_20261103.md section 4
says.  reward/turn_credit_dump_check.py covers the spans against the response mask and the offline recomputation; with
the statement term an answer has two advantage values (statement / the rest), so its span_adv_not_constant is
expected there and this check covers the statement part.

Per turn, from the dumped records, token ids and advantages:

* the statement span: starts where the answer does and ends inside it; decoded, it is the <observe>...</observe> block
  (at most a 2-character tail of a merged token after "</observe>"), and the rest of the answer holds the <action>
  block and no <observe>; a turn without a readable statement has the whole answer as its span;
* the advantages, recomputed here without vagen.turn_credit: every answer token outside the statement has the
  episode's verl GRPO advantage E = (R - group mean) / (group std (n - 1) + eps), every statement token
  E + turn_weight x z, z = (accuracy - batch mean) / (batch std + eps) over the update's turns;
* per number of actions in the turn (the rw_tc1 drift check): turns, statement tokens, accuracy, turn_weight x z and
  the statement advantage mass (turn_weight x z x statement tokens).

    python -m reward.statement_dump_check --dump DIR --ppo RUN/ppo.yaml --tokenizer PATH --out OUT.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ACTION = re.compile(r"<action>(.*?)</action>", re.S)


def grpo(returns, uids, eps):
    groups = defaultdict(list)
    for i, u in enumerate(uids):
        groups[u].append(i)
    out = [0.0] * len(returns)
    for idx in groups.values():
        if len(idx) < 2:
            continue
        r = np.asarray([returns[i] for i in idx], dtype=np.float64)
        for i, v in zip(idx, r):
            out[i] = float((v - r.mean()) / (r.std(ddof=1) + eps))
    return out


def n_actions(text):
    m = ACTION.search(text)
    return len([a for a in m.group(1).split("|") if a.strip()]) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ppo", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import yaml
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cfg = yaml.safe_load(Path(args.ppo).read_text())["algorithm"]["turn_credit"]
    assert cfg.get("turn_term") == "statement" and cfg.get("episode_norm") == "none", cfg
    wt, eps = float(cfg.get("turn_weight", 0.5)), float(cfg.get("eps", 1e-6))
    n = Counter()
    bad = {}
    by_actions = defaultdict(lambda: defaultdict(list))
    per_update = {}

    def note(kind, ex):
        n[kind] += 1
        if len(bad.setdefault(kind, [])) < 5:
            bad[kind].append(ex)

    for f in sorted(Path(args.dump).glob("update_*.jsonl")):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        ep = grpo([r["return"] for r in rows], [r["uid"] for r in rows], eps)
        acc = [float(t.get("statement_accuracy") or 0.0) for r in rows for t in r["record"]["turns"]]
        mu, sd = float(np.mean(acc)), float(np.std(acc))
        per_update[f.stem] = {"samples": len(rows), "turns": len(acc), "accuracy_mean": round(mu, 4),
                              "accuracy_std": round(sd, 4),
                              "readable": round(float(np.mean([t["statement_span"] != t["span"] for r in rows
                                                               for t in r["record"]["turns"]])), 4)}
        for i, r in enumerate(rows):
            n["samples"] += 1
            a, ids = r["adv"], r["ids"]
            for t, turn in enumerate(r["record"]["turns"]):
                n["turns"] += 1
                if "statement_span" not in turn:
                    note("turn_without_statement", {"file": f.name, "i": i, "turn": t})
                    continue
                s, e = turn["span"][0], min(turn["span"][1], len(a))
                ss, se = turn["statement_span"][0], min(turn["statement_span"][1], len(a))
                if e <= s:
                    n["turns_without_tokens"] += 1
                    continue
                readable = turn["statement_span"] != turn["span"]
                n["readable_turns"] += int(readable)
                z = (float(turn["statement_accuracy"]) - mu) / (sd + eps)
                want_stmt, want_rest = ep[i] + wt * z, ep[i]
                if readable:
                    if ss != s or not s < se < e:
                        note("statement_span_outside_answer", {"file": f.name, "i": i, "turn": t,
                                                               "span": [s, e], "statement_span": [ss, se]})
                        continue
                    stext = tok.decode(ids[ss:se], skip_special_tokens=False)
                    rest = tok.decode(ids[se:e], skip_special_tokens=False)
                    k = stext.find("</observe>")
                    if not stext.lstrip().startswith("<observe>") or k < 0 or len(stext) - k - len("</observe>") > 2:
                        note("statement_text", {"file": f.name, "i": i, "turn": t, "text": stext[:200]})
                    if "action>" not in rest or "<observe>" in rest:
                        note("rest_text", {"file": f.name, "i": i, "turn": t, "text": rest[:200]})
                    n["statement_tokens"] += se - ss
                    n["rest_tokens"] += e - se
                    if max(abs(a[k2] - want_stmt) for k2 in range(ss, se)) > 1e-4:
                        note("statement_adv", {"file": f.name, "i": i, "turn": t, "got": a[ss], "want": want_stmt})
                    if max(abs(a[k2] - want_rest) for k2 in range(se, e)) > 1e-4:
                        note("rest_adv", {"file": f.name, "i": i, "turn": t, "got": a[se], "want": want_rest})
                    answer = stext + rest
                else:
                    if (ss, se) != (s, e):
                        note("unreadable_span_not_answer", {"file": f.name, "i": i, "turn": t})
                    if max(abs(a[k2] - want_stmt) for k2 in range(s, e)) > 1e-4:
                        note("unreadable_adv", {"file": f.name, "i": i, "turn": t, "got": a[s], "want": want_stmt})
                    answer = tok.decode(ids[s:e], skip_special_tokens=False)
                    if len(bad.get("unreadable_examples", [])) < 5:
                        bad.setdefault("unreadable_examples", []).append(answer[:200])
                g = by_actions[str(min(n_actions(answer), 5))]
                g["statement_tokens"].append((se - ss) if readable else 0)
                g["accuracy"].append(float(turn["statement_accuracy"]))
                g["weighted_z"].append(wt * z)
                g["mass"].append(wt * z * ((se - ss) if readable else (e - s)))
    actions = {k: {"turns": len(v["accuracy"]), **{m: round(float(np.mean(x)), 4) for m, x in v.items()}}
               for k, v in sorted(by_actions.items())}
    out = {"dump": args.dump, "settings": cfg, "counts": dict(n),
           "mismatch_kinds": {k: n[k] for k in bad if k != "unreadable_examples"},
           "per_update": per_update, "by_actions_per_turn": actions, "examples": bad}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("counts", "mismatch_kinds", "per_update", "by_actions_per_turn")}, indent=1))


if __name__ == "__main__":
    main()
