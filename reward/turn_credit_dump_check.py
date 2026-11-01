"""Reward line (phase 2, direction 1; step 2 of the approved plan, folded into the smoke): check the estimator's dump
(vagen/custom_advantage/turn_credit.py, settings dump_dir / dump_updates) of a training run's first updates.

Per sample:

* the turns' spans against the response mask: every span token has mask 1, every mask-1 token is in a span, spans do
  not overlap and come in order;
* each span decoded with the run's tokenizer: an answer (an <action> tag; no vision placeholders, no chat-role
  headers), and the text between spans (mask 0) the observations;
* the token advantages: 0 on every mask-0 token, constant within a span, and equal (to 1e-4) to what
  vagen.turn_credit.token_advantages gives offline from the dumped records, returns and mask with the run's settings.

    python -m reward.turn_credit_dump_check --dump DIR --ppo RUN/ppo.yaml --tokenizer PATH --out OUT.json

Prints and writes the counts and up to 5 examples of each kind of mismatch.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ppo", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    import yaml
    from transformers import AutoTokenizer

    from vagen.turn_credit import settings, token_advantages

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ppo = yaml.safe_load(Path(args.ppo).read_text())
    cfg = settings(ppo["algorithm"])
    n = Counter()
    bad = {}

    def note(kind, ex):
        n[kind] += 1
        bad.setdefault(kind, [])
        if len(bad[kind]) < 5:
            bad[kind].append(ex)

    per_update = {}
    for f in sorted(Path(args.dump).glob("update_*.jsonl")):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        length = max(len(r["mask"]) for r in rows)
        mask = torch.zeros((len(rows), length), dtype=torch.int64)
        for i, r in enumerate(rows):
            mask[i, :len(r["mask"])] = torch.tensor(r["mask"])
        recs = [r["record"] for r in rows]
        adv, metrics = token_advantages(recs, [r["uid"] for r in rows], mask, [r["return"] for r in rows], cfg)
        per_update[f.stem] = {k: round(v, 4) for k, v in metrics.items()}
        for i, r in enumerate(rows):
            n["samples"] += 1
            m, a, ids = r["mask"], r["adv"], r["ids"]
            covered = [False] * len(m)
            last_end = 0
            for t, turn in enumerate(r["record"]["turns"]):
                s, e = turn["span"]
                n["turns"] += 1
                if s < last_end:
                    note("span_order", {"file": f.name, "i": i, "turn": t, "span": [s, e]})
                last_end = max(last_end, e)
                e2 = min(e, len(m))
                if e2 <= s:
                    n["turns_without_tokens"] += 1
                    continue
                if any(m[k] != 1 for k in range(s, e2)):
                    note("span_has_mask0", {"file": f.name, "i": i, "turn": t, "span": [s, e]})
                for k in range(s, e2):
                    covered[k] = True
                vals = {round(a[k], 4) for k in range(s, e2)}
                if len(vals) != 1:
                    note("span_adv_not_constant", {"file": f.name, "i": i, "turn": t, "values": sorted(vals)[:4]})
                if ids is not None:
                    text = tok.decode(ids[s:e2], skip_special_tokens=False)
                    n["span_tokens"] += e2 - s
                    if "<action>" not in text:
                        note("span_without_action", {"file": f.name, "i": i, "turn": t, "text": text[:200]})
                    if "<|vision_start|>" in text or "<|image_pad|>" in text or "<|im_start|>user" in text:
                        note("span_with_observation", {"file": f.name, "i": i, "turn": t, "text": text[:200]})
            for k, (mk, cv) in enumerate(zip(m, covered)):
                if mk == 1 and not cv:
                    note("mask1_outside_spans", {"file": f.name, "i": i, "pos": k})
                if mk == 0 and abs(a[k]) > 1e-6:
                    note("mask0_with_advantage", {"file": f.name, "i": i, "pos": k, "adv": a[k]})
            mine = adv[i, :len(a)].tolist()
            diff = max((abs(x - y) for x, y in zip(mine, a)), default=0.0)
            n["max_abs_diff_x1e6"] = max(n["max_abs_diff_x1e6"], int(diff * 1e6))
            if diff > 1e-4:
                note("offline_online_differ", {"file": f.name, "i": i, "max_abs_diff": diff})
            if ids is not None and r["record"]["turns"]:
                s0 = r["record"]["turns"][0]["span"][0]
                n["text_before_first_span"] += int(s0 > 0)
    out = {"dump": args.dump, "settings": cfg, "counts": dict(n), "examples": bad, "offline_metrics": per_update}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({"counts": dict(n), "mismatch_kinds": {k: len(v) for k, v in bad.items()}}, indent=1))


if __name__ == "__main__":
    main()
