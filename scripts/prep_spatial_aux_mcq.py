#!/usr/bin/env python3
"""Build A1 spatial-aux MCQ jsonl from SITE-Bench image samples (or ViewSpatial).

Default source: existing EASI SITE image sample jsonl + sitebench image cache.
Output schema per line:
  {id, source, question, options, answer, image_paths}
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

LETTER_RE = re.compile(r"^\s*([A-D])\b", re.IGNORECASE)


def _letter_from_target(target: str) -> str | None:
    t = (target or "").strip()
    m = LETTER_RE.match(t)
    if m:
        return m.group(1).upper()
    # <answer>C</answer> / C. xxx
    m = re.search(r"([A-D])", t)
    return m.group(1).upper() if m else None


def _load_site_from_samples(sample_paths: list[Path], image_root: Path, max_n: int, seed: int):
    rng = random.Random(seed)
    rows = []
    for sp in sample_paths:
        with sp.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                item = json.loads(line)
                media = item.get("input_media") or []
                if not media:
                    continue
                abs_paths = []
                ok = True
                for rel in media:
                    p = Path(rel)
                    if not p.is_absolute():
                        p = image_root / rel
                    if not p.is_file():
                        ok = False
                        break
                    abs_paths.append(str(p.resolve()))
                if not ok:
                    continue
                ans = _letter_from_target(str(item.get("target", "")))
                if not ans:
                    continue
                q = str(item.get("input", "")).strip()
                if not q:
                    continue
                rows.append(
                    {
                        "id": f"site_{item.get('doc_id', len(rows))}",
                        "source": "site_bench_image",
                        "question": q,
                        "options": [],
                        "answer": ans,
                        "image_paths": abs_paths,
                    }
                )
    rng.shuffle(rows)
    if max_n > 0:
        rows = rows[:max_n]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--easi-root",
        type=Path,
        default=Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/unified_eval_runs/easi_results"),
    )
    ap.add_argument(
        "--ckpt",
        default="qwen_pretrain_baseline",
        help="Which ckpt's SITE sample shards to harvest questions from",
    )
    ap.add_argument(
        "--site-image-root",
        type=Path,
        default=Path("/mnt/umm/users/yinbaiqiao/.cache/huggingface/sitebench"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/data_gen/active_spatial_pipeline/output_aux/spatial_aux_site_mcq.jsonl"
        ),
    )
    ap.add_argument("--max-n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sample_paths = sorted(
        (args.easi_root / args.ckpt / "shards").glob(
            "site_bench_image_*/**/*samples_site_bench_image.jsonl"
        )
    )
    if not sample_paths:
        # merged location
        sample_paths = sorted(
            (args.easi_root / args.ckpt).rglob("*samples_site_bench_image.jsonl")
        )
    if not sample_paths:
        raise SystemExit(f"No SITE sample jsonl under {args.easi_root / args.ckpt}")

    rows = _load_site_from_samples(sample_paths, args.site_image_root, args.max_n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(rows)} MCQ items -> {args.out}")
    print(f"     sources={sorted({r['source'] for r in rows})} from {len(sample_paths)} sample files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
