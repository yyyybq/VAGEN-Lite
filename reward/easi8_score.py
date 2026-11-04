"""Reward line: EASI-8 scores of a set of weights, computed as the main line's matrix v3 summary computes them
(VAGEN-Lite exps/vagen_active_spatial/interiorgs_v10_q3_20261027/scripts/matrix_summary_q3.py: easi_files, easi and
mindcube_lenient, copied here with the folders as arguments), so the reward line's numbers and the main line's read
the same way:

* official: the mean over the eight benchmarks of lmms-eval's overall score as EASI's LmmsEvalAdapter reads it (SITE
  merged from its image and video jobs);
* re-judged: the same mean with MindCube-Tiny re-judged by the option an answer names (its letter, else the one
  option whose text it gives).

A weight is scored only when all nine of its jobs (ROOT/<job>/<name>/done.json) are in, every one on the full
benchmark (no limit) and naming the same model.  Its result files are linked into <out folder>/merged/<name>
(symlinks; the result folders are only read) and scored there.

    python -B -m reward.easi8_score --easi EASI/scripts/submissions --model NAME=ROOT [--model ...] \
        [--label NAME=LABEL ...] --out FILE.json

FILE.json and FILE.md (a table in Chinese).
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

EASI_8 = ("vsi_bench", "mmsi_bench", "mindcube_tiny", "viewspatial", "site", "blink", "3dsrbench", "embspatial")
JOBS = ("vsi_bench", "mmsi_bench", "mindcube_tiny", "viewspatial", "site_image", "site_video", "blink", "3dsrbench",
        "embspatial")
SHORT = {"vsi_bench": "VSI", "mmsi_bench": "MMSI", "mindcube_tiny": "MindCube", "viewspatial": "ViewSpatial",
         "site": "SITE", "blink": "BLINK", "3dsrbench": "3DSR", "embspatial": "EmbSpatial"}


def link(dst: Path, files) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for name, f in files:
        t = dst / name
        assert not t.exists() or t.resolve() == f.resolve(), t
        if not t.exists():
            t.symlink_to(f)
    return dst


def easi_files(root: Path, name):
    """(the lmms-eval files of every job of ``name``, the model its jobs name); a missing job raises."""
    out, models = [], set()
    for b in JOBS:
        d = root / b / name
        done = json.loads((d / "done.json").read_text())
        assert done.get("limit") is None, (d, done.get("limit"))
        models.add(done.get("model"))
        sub = [s for s in d.iterdir() if s.is_dir() and any(s.glob("*_results.json"))]
        assert len(sub) == 1, (d, sub)
        out += [(f"{b}__{f.name}", f) for f in sorted(sub[0].glob("*_results.json"))
                + sorted(sub[0].glob("*_samples_*.jsonl"))]
    assert len(models) == 1, (name, models)
    return out, models.pop()


def mindcube_lenient(folder: Path) -> float:
    """As easi8_table_q3.py: MindCube-Tiny re-judged by the option an answer names (letter, else unique option text)."""
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()  # noqa: E731
    f = sorted(folder.glob("*_samples_mindcube_tiny*.jsonl"))[-1]
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    right = 0
    for r in rows:
        x = r.get("filtered_resps") or r.get("resps")
        while isinstance(x, list) and x:
            x = x[0]
        s = re.sub(r"</?answer>", "", str(x)).strip()
        q = r["input"].split("[Question]")[-1].strip()
        opts = dict(re.findall(r"(?:^|\s)([A-F])\.\s(.*?)(?=\s[A-F]\.\s|$)", q, flags=re.S))
        target = re.search(r"<answer>\s*([A-F])", str(r["target"])).group(1)
        m = re.match(r"^\(?([A-F])[\.\):]?(\s|$)", s)
        if m:
            pred = m.group(1)
        else:
            hits = [k for k, v in opts.items() if norm(v) and norm(v) == norm(s)] or \
                   [k for k, v in opts.items() if norm(v) and (norm(v) in norm(s) or norm(s) in norm(v))]
            pred = hits[0] if len(hits) == 1 else None
        right += pred == target
    return 100 * right / len(rows)


def score(root: Path, name, merged: Path, adapter, bench):
    """{official, rejudged, per_benchmark, mindcube_lenient} of one weight (means over the eight benchmarks, 0-100)."""
    files, model = easi_files(root, name)
    folder = link(merged / name, files)
    got = adapter.extract_scores(folder, name, bench)
    main = {b: got[b].overall if b in got else None for b in EASI_8}
    assert all(v is not None for v in main.values()), (name, main)
    lenient = mindcube_lenient(folder)
    rejudged = [main[b] for b in EASI_8 if b != "mindcube_tiny"] + [lenient]
    return {"model": model, "results": str(root / "<job>" / name), "official": statistics.mean(main.values()),
            "rejudged": statistics.mean(rejudged), "per_benchmark": main, "mindcube_lenient": lenient}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--easi", required=True, help="EASI's scripts/submissions folder (its LmmsEvalAdapter)")
    ap.add_argument("--model", action="append", required=True, help="NAME=ROOT (the jobs are ROOT/<job>/<NAME>)")
    ap.add_argument("--label", action="append", default=[], help="NAME=LABEL for the table")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.easi)
    from backends.lmmseval import _TASK_MAP, LmmsEvalAdapter

    bench = {k: v for k, v in _TASK_MAP.items() if k in EASI_8 or k in ("site_image", "site_video")}
    labels = dict(x.split("=", 1) for x in args.label)
    out_file = Path(args.out)
    adapter = LmmsEvalAdapter()
    out = {}
    for m in args.model:
        name, root = m.split("=", 1)
        out[name] = {"label": labels.get(name, name),
                     **score(Path(root), name, out_file.parent / "merged", adapter, bench)}
    out_file.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    L = ["| 模型 | 官方 | 重判 | " + " | ".join(SHORT[b] for b in EASI_8) + " | MindCube 重判 |",
         "|" + " --- |" * (len(EASI_8) + 4)]
    for r in out.values():
        L.append(f"| {r['label']} | {r['official']:.2f} | {r['rejudged']:.2f} | "
                 + " | ".join(f"{r['per_benchmark'][b]:.1f}" for b in EASI_8) + f" | {r['mindcube_lenient']:.1f} |")
    out_file.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
