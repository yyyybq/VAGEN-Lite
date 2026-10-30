#!/usr/bin/env python3
"""X4 follow-up (the user's decision of 2026-10-27): how much of the main-line answering each model keeps, measured as
the loss on the main line's expert answers, which still separates models whose success rate is 0.

Data: the main line's round-11 SFT validation trajectories (TRAJECTORIES, read by path: 1,293 expert trajectories of
scenes outside its training split; sharegpt messages system / user with <image> / assistant "<action>...</action>",
256 x 256 frames).  Each trajectory is rendered with Qwen2.5-VL's chat template and the original 7B's processor, the
same for every model, so all models see exactly the same tokens, and run with teacher forcing (the expert's earlier
answers in the history, as the SFT trained on them).  On every answer token (the answer and its closing <|im_end|>)
the model's negative log likelihood is recorded, and whether its top-1 token is the expert's.

    python downstream/x/x4_ppl.py score --model PATH --out X4/ppl/orig-S.jsonl [--limit N]     (one GPU)
    python downstream/x/x4_ppl.py analyze --dir X4/ppl

score writes one line per trajectory: {"i", "tokens", "nll_sum", "correct", "turns", "turns_exact", "length"}
(turns_exact: answers whose every token is the model's top-1, i.e. the model would greedily give the expert's answer
given the history); it is resumable (trajectories already in OUT are skipped) and leaves out trajectories longer than
MAX_TOKENS (recorded as {"i", "skipped"}).  analyze writes results.json / results.md: per model the mean loss per
answer token, its perplexity exp(loss), the top-1 token accuracy and the exact-answer rate, with bootstrap 95%
intervals over trajectories; per start the E1 model against its original start, and the starts' damages
(E1 loss - start loss) against B's, trajectory by trajectory (Wilcoxon signed-rank).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

TRAJECTORIES = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_train_r11_20261016/"
                    "lf_formal/trajectory_val.jsonl")
PROCESSOR = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/inputs/base_7b")
MAX_TOKENS = 16384  # the SFT's cutoff_len
STARTS = ("B", "S", "RS", "RD")
IM_START, IM_END, ASSISTANT, NEWLINE = 151644, 151645, 77091, 198  # Qwen2.5 tokens: <|im_start|> <|im_end|> assistant \n


def to_chat(messages) -> list:
    """sharegpt messages -> chat-template messages; '<image>' in a user turn becomes an image part in its place."""
    out = []
    for m in messages:
        if m["role"] == "user":
            parts = []
            for k, text in enumerate(m["content"].split("<image>")):
                if k:
                    parts.append({"type": "image"})
                if text:
                    parts.append({"type": "text", "text": text})
            out.append({"role": "user", "content": parts})
        else:
            out.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
    return out


def answer_spans(ids) -> list:
    """[(first, last)] token positions of every assistant answer: after '<|im_start|>assistant\\n', up to and including
    its '<|im_end|>' (an answer cut off at the end runs to the last token)."""
    ids = list(ids)
    spans = []
    for t in range(len(ids) - 2):
        if ids[t] == IM_START and ids[t + 1] == ASSISTANT and ids[t + 2] == NEWLINE:
            first = t + 3
            last = first
            while last < len(ids) - 1 and ids[last] != IM_END:
                last += 1
            if first < len(ids):
                spans.append((first, last))
    return spans


def score(model_path: Path, out: Path, limit: int = None, trajectories: Path = TRAJECTORIES,
          processor_path: Path = PROCESSOR, starvla_pt: Path = None) -> int:
    """starvla_pt (2026-10-30, C1's comparisons: DS-B-030-s1 without a copy): the architecture from model_path, every
    weight then replaced in memory by the StarVLA checkpoint's VLM (q0_ask.starvla_vlm_state)."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    done = set()
    if out.is_file():
        for line in out.read_text().splitlines():
            try:
                done.add(json.loads(line)["i"])
            except (ValueError, KeyError):
                continue
    processor = AutoProcessor.from_pretrained(str(processor_path))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path), torch_dtype=torch.bfloat16,
                                                               attn_implementation="sdpa")
    if starvla_pt:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import q0_ask

        missing, unexpected = model.load_state_dict(q0_ask.starvla_vlm_state(str(starvla_pt)), strict=False)
        if missing or unexpected:
            raise RuntimeError(f"{starvla_pt}: {len(missing)} missing, {len(unexpected)} unexpected weights")
    model = model.cuda().eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "a") as f, torch.no_grad():
        for i, line in enumerate(Path(trajectories).read_text().splitlines()):
            if limit is not None and i >= limit:
                break
            if i in done:
                continue
            r = json.loads(line)
            text = processor.apply_chat_template(to_chat(r["messages"]), tokenize=False, add_generation_prompt=False)
            images = [Image.open(p).convert("RGB") for p in r["images"]]
            inputs = processor(text=[text], images=images or None, return_tensors="pt")
            ids = inputs["input_ids"][0]
            if len(ids) > MAX_TOKENS:
                f.write(json.dumps({"i": i, "skipped": f"{len(ids)} tokens"}) + "\n")
                continue
            spans = answer_spans(ids.tolist())
            pos = torch.tensor([t - 1 for a, b in spans for t in range(a, b + 1)])
            inputs = {k: v.cuda() for k, v in inputs.items()}
            hidden = model.model(**inputs, use_cache=False).last_hidden_state[0]
            logits = model.lm_head(hidden[pos.cuda()]).float()
            target = inputs["input_ids"][0][pos.cuda() + 1]
            nll = -logits.log_softmax(-1).gather(1, target[:, None]).squeeze(1)
            top1 = (logits.argmax(-1) == target).cpu().tolist()
            k, exact = 0, 0
            for a, b in spans:
                exact += all(top1[k:k + b - a + 1])
                k += b - a + 1
            f.write(json.dumps({"i": i, "tokens": len(top1), "nll_sum": round(float(nll.sum()), 5),
                                "correct": int(sum(top1)), "turns": len(spans), "turns_exact": exact,
                                "length": int(len(ids))}) + "\n")
            f.flush()
            n += 1
    return n


# ---------------------------------------------------------------- analysis (any Python with numpy; scipy for Wilcoxon)

def read(path: Path) -> dict:
    out = {}
    for line in Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "skipped" not in r and r.get("tokens"):
            out[r["i"]] = r
    return out


def summary(rows, keys, resamples: int = 1000, seed: int = 0) -> dict:
    """Token-weighted loss, perplexity, top-1 accuracy; exact-answer rate; bootstrap intervals over trajectories."""
    def stats(sample):
        tok = sum(rows[i]["tokens"] for i in sample)
        loss = sum(rows[i]["nll_sum"] for i in sample) / tok
        return (loss, sum(rows[i]["correct"] for i in sample) / tok,
                sum(rows[i]["turns_exact"] for i in sample) / sum(rows[i]["turns"] for i in sample))
    loss, acc, exact = stats(keys)
    rng = random.Random(seed)
    boot = [stats([rng.choice(keys) for _ in keys]) for _ in range(resamples)]

    def ci(j):
        v = sorted(b[j] for b in boot)
        return [round(v[int(0.025 * resamples)], 5), round(v[int(0.975 * resamples) - 1], 5)]
    return {"trajectories": len(keys), "answer_tokens": sum(rows[i]["tokens"] for i in keys),
            "loss": round(loss, 5), "loss_ci95": ci(0), "perplexity": round(math.exp(loss), 4),
            "perplexity_ci95": [round(math.exp(x), 4) for x in ci(0)],
            "top1": round(acc, 5), "top1_ci95": ci(1), "exact": round(exact, 5), "exact_ci95": ci(2)}


def per_traj(rows, keys) -> list:
    return [rows[i]["nll_sum"] / rows[i]["tokens"] for i in keys]


def paired(a: list, b: list, resamples: int = 1000, seed: int = 0) -> dict:
    """a - b per trajectory: mean with a bootstrap 95% interval and the Wilcoxon signed-rank p."""
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(resamples))
    try:
        from scipy.stats import wilcoxon

        p = float(wilcoxon(d).pvalue) if any(d) else 1.0
    except Exception:
        p = None
    return {"pairs": len(d), "mean_diff": round(sum(d) / len(d), 5),
            "ci95": [round(means[int(0.025 * resamples)], 5), round(means[int(0.975 * resamples) - 1], 5)],
            "higher": sum(x > 0 for x in d), "lower": sum(x < 0 for x in d), "wilcoxon_p": p}


def analyze(folder: Path) -> dict:
    runs = {p.stem: read(p) for p in sorted(Path(folder).glob("*.jsonl"))}
    runs = {k: v for k, v in runs.items() if v}
    keys = sorted(set.intersection(*(set(v) for v in runs.values()))) if runs else []
    res = {"common_trajectories": len(keys), "models": {k: summary(v, keys) for k, v in runs.items()} if keys else {},
           "e1_vs_start": {}, "damage_vs_B": {}}
    damage = {}
    for s in STARTS:
        a, b = runs.get(f"E1-{s}-200k"), runs.get(f"orig-{s}")
        if a and b and keys:
            res["e1_vs_start"][s] = paired(per_traj(a, keys), per_traj(b, keys))
            damage[s] = [x - y for x, y in zip(per_traj(a, keys), per_traj(b, keys))]
    for s in damage:
        if s != "B" and "B" in damage:
            res["damage_vs_B"][s] = paired(damage[s], damage["B"])
    out = Path(folder) / "results"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "results.md").write_text(markdown(res), encoding="utf-8")
    return res


def markdown(res: dict) -> str:
    lines = ["# X4 困惑度（自动生成）", "", f"主线第 11 轮 SFT 验证集的专家轨迹，{res['common_trajectories']} 条（所有模型都有的），"
             "老师强制；括号里是对轨迹自助重抽样的 95% 区间。", "",
             "| 模型 | 每个回答词元的损失 | 困惑度 | 逐词元 top-1 一致 | 整个回答一致 |", "| --- | ---: | ---: | ---: | ---: |"]
    order = [m for s in STARTS for m in (f"orig-{s}", f"E1-{s}-200k")]
    for name in order + sorted(set(res["models"]) - set(order)):
        m = res["models"].get(name)
        if not m:
            continue
        lines.append(f"| {name} | {m['loss']:.3f}（{m['loss_ci95'][0]:.3f}–{m['loss_ci95'][1]:.3f}） | "
                     f"{m['perplexity']:.3g}（{m['perplexity_ci95'][0]:.3g}–{m['perplexity_ci95'][1]:.3g}） | "
                     f"{100 * m['top1']:.1f}% | {100 * m['exact']:.1f}% |")
    lines += ["", "## E1 模型比自己的原始起点多出的损失（每条轨迹平均每词元，E1 − 起点）", "",
              "| 起点 | 轨迹 | 平均差（95% 区间） | 变高 / 变低的轨迹 | Wilcoxon p |", "| --- | ---: | ---: | --- | ---: |"]
    for s, r in res["e1_vs_start"].items():
        lines.append(f"| {s} | {r['pairs']} | {r['mean_diff']:+.3f}（{r['ci95'][0]:+.3f}–{r['ci95'][1]:+.3f}） | "
                     f"{r['higher']} / {r['lower']} | {r['wilcoxon_p']} |")
    lines += ["", "## 各起点的损伤和 B 比（损伤 = E1 − 起点；起点 − B）", "",
              "| 起点 | 轨迹 | 平均差（95% 区间） | 更大 / 更小的轨迹 | Wilcoxon p |", "| --- | ---: | ---: | --- | ---: |"]
    for s, r in res["damage_vs_B"].items():
        lines.append(f"| {s} | {r['pairs']} | {r['mean_diff']:+.3f}（{r['ci95'][0]:+.3f}–{r['ci95'][1]:+.3f}） | "
                     f"{r['higher']} / {r['lower']} | {r['wilcoxon_p']} |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--starvla-pt", type=Path, default=None,
                   help="a StarVLA pytorch_model.pt whose VLM replaces --model's weights in memory")
    a = sub.add_parser("analyze")
    a.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "score":
        print(json.dumps({"scored": score(args.model, args.out, args.limit, starvla_pt=args.starvla_pt)}))
    else:
        res = analyze(args.dir)
        print(json.dumps({k: v["perplexity"] for k, v in res["models"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
