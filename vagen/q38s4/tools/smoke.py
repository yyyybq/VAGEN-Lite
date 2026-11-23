"""C line (q38 s4), stage 3: helpers of the smoke and scaling tests S1-S4 and of the data checks.

  subset   --data DATA --out OUT --qa N --nfp M [--seed S]
           a seeded sample of a data_build folder (N rows of qa.jsonl, M of nfp.jsonl, both in source order) with
           its dataset_info.json and a build_manifest.json (rows_out, so launch prepare can estimate the steps);
  plain    --run RUN       RUN/run_plain.sh: RUN/code's run_train.sh with the training module
           vagen.q38s4.lf_nfp replaced by vagen.q38s4.tools.lf_plain (S4's existing SFT trainer); RUN must be a train
           run folder of vagen.q38s4.launch prepare;
  submit   --run RUN --pool POOL [--plain] [--yes]
           launch's sco command for RUN, with RUN/run_plain.sh as the entry for --plain (dry run without --yes);
  inspect  --data DATA --tokenized TOK --model MODEL --out OUT [--n 50] [--seed 0] [--cutoff 16384] [--durations]
           the tokenized data's length distribution (all rows, rows at the cutoff), and for --n seeded rows (half QA
           video rows, a quarter NFP, a quarter image rows when there are that many): source, kind, total and label
           tokens, the video's time units (runs of <|video_pad|>), tokens per unit, the timestamp texts, the answer
           (label tokens), a think block, the text with pad runs collapsed; --durations: every distinct video's
           duration (container metadata), quantiles per source; OUT/inspect.json and OUT/inspect.md;
  summary  --run RUN [--skip 2]
           per-step losses (trainer_state.json log history, else trainer_log.jsonl), the copy baseline against the
           head's nfp_cos, step times (the slowest rank's step_s per step, median over the steps after the first
           --skip), samples per second (global batch / median step), GPU peaks over every rank, host memory peaks
           (the q38 callback's rank-0 lines and every host_mem_sft file); RUN/smoke_summary.json;
  scaling  --runs RUN1 RUN2 ... [--base-nodes 2]   efficiency of each run = its samples/s / ((nodes / B) x the
           B-node run's samples/s), B = --base-nodes (2 since 2026-11-23: 27B does not fit one node on real data);
  compare  --a RUN_A --b RUN_B [--key loss]   per-step equality of a logged value (bit for bit: the float64s of
           the log history), the first step that differs and the largest difference.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from vagen.q38s4 import launch as L

PLAIN_FROM = '-m vagen.q38s4.lf_nfp "$RUN/train.yaml"\n'
PLAIN_TO = '-m vagen.q38s4.tools.lf_plain "$RUN/train.yaml"\n'
TS_RE = re.compile(r"<(\d+(?:\.\d+)?) seconds>")


def utc() -> str:
    return L.utc()


# ------------------------------------------------------------------------------------------------------ subset
def _pick(path: Path, n: int, rng: random.Random) -> List[str]:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    if n > len(lines):
        raise SystemExit(f"{path}: {n} rows asked, {len(lines)} there")
    keep = sorted(rng.sample(range(len(lines)), n))
    return [lines[i] for i in keep]


def subset(data: Path, out: Path, n_qa: int, n_nfp: int, seed: int = 0) -> Dict[str, Any]:
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists: pick a new folder")
    info = json.loads((data / "dataset_info.json").read_text())
    rng = random.Random(f"subset:{seed}")
    qa = _pick(data / "qa.jsonl", n_qa, rng)
    nfp = _pick(data / "nfp.jsonl", n_nfp, rng) if n_nfp else []
    out.mkdir(parents=True)
    (out / "qa.jsonl").write_text("".join(qa), encoding="utf-8")
    (out / "nfp.jsonl").write_text("".join(nfp), encoding="utf-8")
    (out / "nfp_heldout.jsonl").write_text("", encoding="utf-8")
    keep = {k: v for k, v in info.items() if k in ("s4_qa", "s4_nfp", "s4_nfp_heldout")}
    (out / "dataset_info.json").write_text(json.dumps(keep, indent=2) + "\n")
    man = {"subset_of": str(data), "seed": seed, "utc": utc(),
           "rows_out": {"qa": len(qa), "nfp": len(nfp), "qa_heldout": 0, "nfp_heldout": 0}}
    (out / "build_manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    return man


# ------------------------------------------------------------------------------------------------ plain / submit
def plain_entry(run: Path) -> Path:
    run = Path(run).resolve()
    plan = json.loads((run / "plan.json").read_text())
    if plan["mode"] != "train":
        raise SystemExit(f"{run}: a train run folder is needed (mode {plan['mode']})")
    src = (run / "code" / "vagen" / "q38s4" / "run_train.sh").read_text()
    lines = src.splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines) if ln.endswith(PLAIN_FROM) and "Q38S4_TOKENIZE_ONLY" not in ln]
    if len(hits) != 1:
        raise SystemExit(f"run_train.sh: expected one training line ending in {PLAIN_FROM!r}, found {len(hits)}")
    lines[hits[0]] = lines[hits[0]][: -len(PLAIN_FROM)] + PLAIN_TO
    dest = run / "run_plain.sh"
    if dest.exists():
        raise SystemExit(f"{dest} exists")
    dest.write_text("# generated by vagen.q38s4.tools.smoke plain: run_train.sh with lf_plain (S4)\n" + "".join(lines))
    return dest


def submit(run: Path, pool: str, plain: bool, yes: bool, job_name: Optional[str] = None) -> List[str]:
    from vagen.q38s4.tools import job as J

    run = Path(run).resolve()
    plan = json.loads((run / "plan.json").read_text())
    name = job_name or f"q38s4-smoke-{'plain' if plain else plan['arm']}-{run.name}"
    cmd = L.sco_command(run, plan, pool, name)
    entry = f"bash {run}/code/vagen/q38s4/run_train.sh {run}"
    if not cmd[-1] == f"--command={entry}":
        raise SystemExit(f"unexpected launch command {cmd[-1]!r}")
    if plain:
        if not (run / "run_plain.sh").is_file():
            raise SystemExit(f"{run}/run_plain.sh missing: run smoke plain first")
        entry = f"bash {run}/run_plain.sh {run}"
    cmd[-1] = f"--command={J.UMASK}{entry}"  # tools.job: h800 jobs write /mnt/umm as another uid
    if yes:
        J.open_up(run, run / "logs")
    J.run_sco(cmd, run / "submits.jsonl", name, pool, yes)
    return cmd


# ------------------------------------------------------------------------------------------------------ inspect
def pad_runs(ids: Sequence[int], pad: int) -> List[int]:
    runs, n = [], 0
    for t in ids:
        if t == pad:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    return runs


def collapse(ids: Sequence[int], pads: Dict[int, str], decode) -> str:
    parts, buf, i = [], [], 0
    while i < len(ids):
        t = ids[i]
        if t in pads:
            j = i
            while j < len(ids) and ids[j] == t:
                j += 1
            if buf:
                parts.append(decode(buf))
                buf = []
            parts.append(f"{pads[t]}x{j - i}")
            i = j
        else:
            buf.append(t)
            i += 1
    if buf:
        parts.append(decode(buf))
    return "".join(parts)


def row_kind(row: Dict[str, Any]) -> Tuple[str, str]:
    vids = row.get("videos") or []
    imgs = row.get("images") or []
    if vids:
        v = vids[0] if isinstance(vids[0], str) else str(vids[0])
        kind = "nfp" if "|s4|nfp" in v else "qa_video"
        path = v.split("|s4|")[0]
    elif imgs:
        kind, path = "qa_image", imgs[0] if isinstance(imgs[0], str) else str(imgs[0])
    else:
        kind, path = "text", ""
    return kind, path


def source_of(path: str, data: Path) -> str:
    try:
        rel = Path(path).resolve().relative_to(Path(data).resolve())
        return rel.parts[0]
    except ValueError:
        parts = Path(path).parts
        return parts[-2] if len(parts) >= 2 else "?"


def quantiles(xs: Iterable[float]) -> Dict[str, float]:
    xs = sorted(xs)
    if not xs:
        return {}
    pick = lambda q: round(xs[min(len(xs) - 1, int(q * (len(xs) - 1)))], 2)  # noqa: E731
    return {"n": len(xs), "min": round(xs[0], 2), "p10": pick(0.1), "p50": pick(0.5), "p90": pick(0.9),
            "p99": pick(0.99), "max": round(xs[-1], 2), "mean": round(sum(xs) / len(xs), 2)}


def pick_rows(kinds: List[str], n: int, seed: int) -> List[int]:
    by: Dict[str, List[int]] = defaultdict(list)
    for i, k in enumerate(kinds):
        by[k].append(i)
    rng = random.Random(f"inspect:{seed}")
    want = {"qa_video": n // 2, "nfp": n // 4, "qa_image": n - n // 2 - n // 4}
    out: List[int] = []
    for k, m in want.items():
        pool = by.get(k, [])
        out += rng.sample(pool, min(m, len(pool)))
    taken = set(out)
    rest = [i for i in range(len(kinds)) if i not in taken]
    out += rng.sample(rest, min(n - len(out), len(rest)))
    return sorted(out)


def describe_row(ids: List[int], labels: List[int], tok_ids: Dict[str, int], decode) -> Dict[str, Any]:
    vruns = pad_runs(ids, tok_ids["video_pad"])
    iruns = pad_runs(ids, tok_ids["image_pad"])
    pads = {tok_ids["video_pad"]: "<|video_pad|>", tok_ids["image_pad"]: "<|image_pad|>"}
    text = collapse(ids, pads, decode)
    ans_ids = [t for t in labels if t != -100]
    answer = decode(ans_ids) if ans_ids else ""
    ts = [float(x) for x in TS_RE.findall(text)]
    return {"tokens": len(ids), "label_tokens": len(ans_ids), "video_units": len(vruns),
            "video_tokens": sum(vruns), "tokens_per_unit": dict(Counter(vruns)), "image_tokens": sum(iruns),
            "timestamps": {"n": len(ts), "first": ts[:3], "last": ts[-2:]},
            "think_block": "<think>" in text, "empty_think": bool(re.search(r"<think>\s*</think>", text)),
            "answer": answer, "text": text}


def video_durations(paths: Iterable[str], data: Path) -> Dict[str, Dict[str, float]]:
    import av

    from vagen.q38s4.lf_video import stream_info

    per: Dict[str, List[float]] = defaultdict(list)
    for p in sorted(set(paths)):
        with av.open(p, "r") as c:
            s = next(x for x in c.streams if x.type == "video")
            per[source_of(p, data)].append(stream_info(c, s)["duration"])
    return {k: quantiles(v) for k, v in sorted(per.items())}


def inspect(data: Path, tokenized: Path, model: str, out: Path, n: int = 50, seed: int = 0, cutoff: int = 16384,
            durations: bool = False) -> Dict[str, Any]:
    import datasets
    import pyarrow.compute as pc
    from transformers import AutoTokenizer

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ds = datasets.load_from_disk(str(tokenized))
    ds = ds["train"] if isinstance(ds, datasets.DatasetDict) else ds
    lens = pc.list_value_length(ds.data.column("input_ids")).to_pylist()
    cols = [c for c in ("videos", "images") if c in ds.column_names]
    meta = ds.select_columns(cols).to_list()
    kinds = [row_kind(r)[0] for r in meta]
    by_kind: Dict[str, List[int]] = defaultdict(list)
    for k, ln in zip(kinds, lens):
        by_kind[k].append(ln)
    tok = AutoTokenizer.from_pretrained(model)
    tok_ids = {"video_pad": tok.convert_tokens_to_ids("<|video_pad|>"),
               "image_pad": tok.convert_tokens_to_ids("<|image_pad|>")}
    decode = lambda xs: tok.decode(xs, skip_special_tokens=False)  # noqa: E731
    rows = []
    for i in pick_rows(kinds, n, seed):
        r = ds[i]
        kind, path = row_kind(r)
        d = describe_row(list(r["input_ids"]), list(r["labels"]), tok_ids, decode)
        rows.append({"index": i, "kind": kind, "source": source_of(path, data) if path else "", "media": path,
                     "media_spec": (r.get("videos") or [""])[0].split("|s4|")[1] if kind != "qa_image" and r.get("videos") else "",
                     **d})
    rep: Dict[str, Any] = {
        "utc": utc(), "tokenized": str(tokenized), "rows": len(lens), "cutoff": cutoff,
        "lengths": {"all": quantiles(lens), **{k: quantiles(v) for k, v in sorted(by_kind.items())}},
        "at_or_over_cutoff": sum(1 for x in lens if x >= cutoff),
        "rows_by_kind": dict(Counter(kinds)),
        "rows_by_source": dict(sorted(Counter(f"{k}/{source_of(row_kind(m)[1], data)}" for k, m in zip(kinds, meta)
                                              if k != "text").items())),
        "sample": rows,
    }
    if durations:
        rep["video_duration_s"] = video_durations((row_kind(m)[1] for m, k in zip(meta, kinds) if k != "qa_image"
                                                   and k != "text"), data)
    (out / "inspect.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    (out / "inspect.md").write_text(inspect_md(rep), encoding="utf-8")
    return rep


def inspect_md(rep: Dict[str, Any]) -> str:
    out = [f"# tokenized data check ({rep['utc']})", "", f"- data: `{rep['tokenized']}`, {rep['rows']:,} rows",
           f"- rows at or over the cutoff {rep['cutoff']}: {rep['at_or_over_cutoff']}",
           f"- rows by kind: {json.dumps(rep['rows_by_kind'])}", "", "## lengths (tokens)", ""]
    for k, q in rep["lengths"].items():
        out.append(f"- {k}: {json.dumps(q)}")
    if "video_duration_s" in rep:
        out += ["", "## video durations (s) by source", ""]
        out += [f"- {k}: {json.dumps(q)}" for k, q in rep["video_duration_s"].items()]
    out += ["", "## rows by source", ""] + [f"- {k}: {v:,}" for k, v in rep["rows_by_source"].items()]
    out += ["", f"## {len(rep['sample'])} sampled rows", ""]
    for r in rep["sample"]:
        out += [f"### row {r['index']} ({r['kind']}, {r['source']}, {r['media_spec']})", "",
                f"- tokens {r['tokens']}, label tokens {r['label_tokens']}, video units {r['video_units']}, "
                f"video tokens {r['video_tokens']}, tokens per unit {json.dumps(r['tokens_per_unit'])}, image tokens "
                f"{r['image_tokens']}", f"- timestamps {json.dumps(r['timestamps'])}; think block {r['think_block']} "
                f"(empty {r['empty_think']})", f"- answer: `{r['answer'][:300]!r}`", "", "```",
                r["text"][:3000] + (" ..." if len(r["text"]) > 3000 else ""), "```", ""]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------------ summary
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line cut by a killed job
    return out


def log_history(run: Path) -> List[Dict[str, Any]]:
    st = run / "ckpt" / "trainer_state.json"
    if st.is_file():
        return [h for h in json.loads(st.read_text())["log_history"] if "loss" in h and "step" in h]
    tl = run / "ckpt" / "trainer_log.jsonl"
    if tl.is_file():
        return [{"step": r["current_steps"], "loss": r["loss"]} for r in read_jsonl(tl) if "loss" in r]
    return []


def _mean(xs: List[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 6) if xs else None


def loss_trend(hist: List[Dict[str, Any]], key: str, k: int = 10) -> Optional[Dict[str, Any]]:
    xs = [h[key] for h in hist if key in h and h[key] is not None]
    if len(xs) < 2:
        return None
    k = max(1, min(k, len(xs) // 2))
    first, last = _mean(xs[:k]), _mean(xs[-k:])
    return {"n": len(xs), "first": xs[0], "last": xs[-1], f"mean_first_{k}": first, f"mean_last_{k}": last,
            "down": last < first}


def summary(run: Path, skip: int = 2) -> Dict[str, Any]:
    run = Path(run).resolve()
    plan = json.loads((run / "plan.json").read_text())
    hist = log_history(run)
    rep: Dict[str, Any] = {"run": str(run), "nodes": plan["nodes"], "world": plan["world"],
                           "global_batch": plan["global_batch"], "steps_logged": len(hist),
                           "losses": {k: loss_trend(hist, k) for k in ("loss", "text_loss", "nfp_mse", "nfp_cos",
                                                                         "copy_mse", "copy_cos", "grad_norm")}}
    pairs = [(h["step"], h["nfp_cos"], h["copy_cos"]) for h in hist if "nfp_cos" in h and "copy_cos" in h]
    if pairs:
        half = len(pairs) // 2
        gap = [c - n for _, n, c in pairs]
        rep["nfp_vs_copy"] = {"steps": len(pairs), "steps_below_copy": sum(1 for g in gap if g > 0),
                              "gap_first_half": _mean(gap[:half]), "gap_second_half": _mean(gap[half:]),
                              "note": "gap = copy_cos - nfp_cos (1 - cos); > 0: the head beats copying the frame"}
    mdir = run / "sft_metrics"
    steps: Dict[int, float] = {}
    gpu_alloc = gpu_res = rss = 0.0
    used = cg = 0.0
    for f in sorted(mdir.glob("rank*.jsonl")):
        for r in read_jsonl(f):
            if r.get("event") != "step":
                continue
            if r.get("step_s") is not None:
                steps[r["step"]] = max(steps.get(r["step"], 0.0), r["step_s"])
            gpu_alloc = max(gpu_alloc, r.get("max_allocated_gib") or 0)
            gpu_res = max(gpu_res, r.get("max_reserved_gib") or 0)
            rss = max(rss, r.get("rss_gb") or 0)
            used = max(used, r.get("used_gb") or 0)
            cg = max(cg, r.get("cgroup_gb") or r.get("cgroup_current_gb") or 0)
    for f in sorted(mdir.glob("host_mem_sft-*.jsonl")):
        for r in read_jsonl(f):
            pk = r.get("peak") or {}
            used = max(used, pk.get("used_gb") or 0)
            cg = max(cg, pk.get("cgroup_gb") or 0)
    timed = [steps[s] for s in sorted(steps) if s > skip]
    med = statistics.median(timed) if timed else None
    rep["step_s"] = {"steps": len(steps), "timed_steps": len(timed), "median": med,
                     "min": min(timed) if timed else None, "max": max(timed) if timed else None,
                     "first": [steps[s] for s in sorted(steps)[:skip]]}
    rep["samples_per_s"] = round(plan["global_batch"] / med, 4) if med else None
    rep["gpu_peak_gib"] = {"allocated": gpu_alloc, "reserved": gpu_res}
    rep["host_peak_gb"] = {"machine_used": used, "cgroup": cg, "rank_rss": rss, "threshold": 1600,
                           "under_threshold": used < 1600 if used else None}
    (run / "smoke_summary.json").write_text(json.dumps(rep, indent=1) + "\n")
    return rep


def scaling(reps: List[Dict[str, Any]], base_nodes: int = 1) -> Dict[str, Any]:
    """Efficiency of each run = its samples/s / ((nodes / base_nodes) x the base run's samples/s).

    base_nodes 2 since 2026-11-23: 27B does not fit one node on real data (user: 2 nodes are the smallest unit).
    """
    base = [r for r in reps if r["nodes"] == base_nodes]
    if len(base) != 1 or not base[0]["samples_per_s"]:
        raise SystemExit(f"scaling: exactly one {base_nodes}-node run with a step time is needed")
    one = base[0]["samples_per_s"]
    rows = []
    for r in sorted(reps, key=lambda x: x["nodes"]):
        eff = r["samples_per_s"] / (r["nodes"] / base_nodes * one) if r["samples_per_s"] else None
        rows.append({"run": r["run"], "nodes": r["nodes"], "samples_per_s": r["samples_per_s"],
                     "efficiency": None if eff is None else round(eff, 4)})
    eight = [x for x in rows if x["nodes"] == 8 and x["efficiency"] is not None]
    verdict = None
    if eight:
        e = eight[0]["efficiency"]
        verdict = "8 nodes (>= 0.70)" if e >= 0.70 else ("stop and report (< 0.50)" if e < 0.50 else
                                                         "between 0.50 and 0.70: ask the user")
    return {"base_nodes": base_nodes, "base_samples_per_s": one, "rows": rows, "verdict": verdict}


def compare(a: List[Dict[str, Any]], b: List[Dict[str, Any]], key: str = "loss") -> Dict[str, Any]:
    da = {h["step"]: h[key] for h in a if key in h}
    db = {h["step"]: h[key] for h in b if key in h}
    common = sorted(set(da) & set(db))
    diffs = [(s, abs(float(da[s]) - float(db[s]))) for s in common]
    first = next((s for s, d in diffs if da[s] != db[s]), None)
    return {"key": key, "steps_a": len(da), "steps_b": len(db), "common": len(common),
            "equal_steps": sum(1 for s in common if da[s] == db[s]), "first_different_step": first,
            "max_abs_diff": max((d for _, d in diffs), default=None),
            "bitwise_equal": bool(common) and first is None and len(da) == len(db)}


# ------------------------------------------------------------------------------------------------------ cli
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--qa", type=int, required=True)
    s.add_argument("--nfp", type=int, required=True)
    s.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("plain")
    p.add_argument("--run", required=True)
    u = sub.add_parser("submit")
    u.add_argument("--run", required=True)
    u.add_argument("--pool", required=True, choices=L.POOLS)
    u.add_argument("--plain", action="store_true")
    u.add_argument("--job-name")
    u.add_argument("--yes", action="store_true")
    i = sub.add_parser("inspect")
    i.add_argument("--data", required=True)
    i.add_argument("--tokenized", required=True)
    i.add_argument("--model", default=L.MODEL_27B)
    i.add_argument("--out", required=True)
    i.add_argument("--n", type=int, default=50)
    i.add_argument("--seed", type=int, default=0)
    i.add_argument("--cutoff", type=int, default=16384)
    i.add_argument("--durations", action="store_true")
    m = sub.add_parser("summary")
    m.add_argument("--run", required=True)
    m.add_argument("--skip", type=int, default=2)
    c = sub.add_parser("scaling")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--skip", type=int, default=2)
    c.add_argument("--base-nodes", type=int, default=2)
    q = sub.add_parser("compare")
    q.add_argument("--a", required=True)
    q.add_argument("--b", required=True)
    q.add_argument("--key", default="loss")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    a = parse_args(argv)
    if a.cmd == "subset":
        res: Any = subset(Path(a.data), Path(a.out), a.qa, a.nfp, a.seed)
    elif a.cmd == "plain":
        res = str(plain_entry(Path(a.run)))
    elif a.cmd == "submit":
        submit(Path(a.run), a.pool, a.plain, a.yes, a.job_name)
        return
    elif a.cmd == "inspect":
        rep = inspect(Path(a.data), Path(a.tokenized), a.model, Path(a.out), a.n, a.seed, a.cutoff, a.durations)
        res = {k: rep[k] for k in ("rows", "at_or_over_cutoff", "rows_by_kind", "lengths")}
    elif a.cmd == "summary":
        res = summary(Path(a.run), a.skip)
    elif a.cmd == "scaling":
        res = scaling([summary(Path(r), a.skip) for r in a.runs], a.base_nodes)
    else:
        res = {k: compare(log_history(Path(a.a).resolve()), log_history(Path(a.b).resolve()), k)
               for k in dict.fromkeys([a.key, "grad_norm"])}
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
