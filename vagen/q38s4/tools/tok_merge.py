"""C line (q38 s4): merge tokenized parts (one LF tokenize per dataset) into the tokenized dataset a training reads.

  python -m vagen.q38s4.tools.tok_merge --parts TOK_QA TOK_QA_IMG TOK_NFP --out TOK [--check REF] [--sample 64]

Why (2026-11-23): the one-pass tokenize of the full s4_v2 (dataset s4_qa,s4_qa_img,s4_nfp) failed at 84% in
datasets' map writer: ``Couldn't cast array of type list<item: string> to null``.  LF concatenates the datasets in
order (mix strategy concat) and tokenizes with dataset.map(num_proc=12), which cuts the rows into 12 contiguous
shards; each shard's writer types a new column from its first written batch.  A shard that starts inside s4_qa
(video rows, images all None) types images null and then fails on its first image row; a shard that starts inside
s4_qa_img types videos null and fails on the NFP video rows.  The subsets passed because a shard of a few hundred
rows is one batch that holds both media.  So each dataset is tokenized on its own (launch prepare --mode tokenize
--set dataset=ONE: one medium per run, nothing to cast) and this tool joins the parts.

Each part is LF's tokenize output: a DatasetDict saved with save_to_disk whose only split is train.  The merge
concatenates the train splits in the order of --parts, which must be the training's dataset order (s4_qa,
s4_qa_img, s4_nfp, train.yaml's dataset key), so row i of TOK is row i of what the one-pass tokenize makes: LF
tokenizes row by row (no packing) and does not shuffle before the training's sampler.  Every part must have the same
columns in the same order; a column typed null in a part (images in a video-only part) takes the type of the others
(concatenate_datasets aligns null with any type).  TOK is written as DatasetDict({"train": ...}) to TOK.partial and
renamed at the end, so a TOK that exists is complete (launch refuses a train run folder while TOK is missing).

After writing, TOK is read back and checked against the parts: the row count of each part, the first and last row
of each part and --sample seeded rows equal (every column).  --check REF also compares TOK with a reference
tokenized dataset row by row (every row, every column) and the column types: the subset check against a one-pass
tokenize of the same rows.  TOK/tok_merge.json records the parts (path, rows, columns and types), the order, the
checks and the time.  Exit code 1 when a check fails (TOK.partial is then left for inspection and TOK not made).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PARTIAL = ".partial"
MANIFEST = "tok_merge.json"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_part(path: Path):
    """The train split of LF's tokenized DatasetDict at PATH (refused: anything else)."""
    from datasets import DatasetDict, load_from_disk
    try:
        d = load_from_disk(str(path))
    except IndexError:  # datasets 5.0.1 saves a split with no rows as 0 shards and cannot read it back
        raise SystemExit(f"{path}: a split with no rows") from None
    if not isinstance(d, DatasetDict) or list(d.keys()) != ["train"]:
        kind = type(d).__name__ if not isinstance(d, DatasetDict) else f"splits {list(d.keys())}"
        raise SystemExit(f"{path}: expected LF's tokenized DatasetDict with only a train split, got {kind}")
    return d["train"]


def types(ds) -> Dict[str, str]:
    return {k: str(v) for k, v in zip(ds.data.schema.names, ds.data.schema.types)}


def rows_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return list(a.keys()) == list(b.keys()) and all(a[k] == b[k] for k in a)


def sample_index(sizes: Sequence[int], n: int, seed: int) -> List[int]:
    """Global row indices to check: each part's first and last row and n seeded rows over all rows."""
    idx, start = set(), 0
    for s in sizes:
        if s:
            idx.update((start, start + s - 1))
        start += s
    total = start
    rng = random.Random(seed)
    for _ in range(min(n, total)):
        idx.add(rng.randrange(total))
    return sorted(idx)


def check_against_parts(merged, parts: List, n: int, seed: int) -> Dict[str, Any]:
    sizes = [len(p) for p in parts]
    out: Dict[str, Any] = {"rows": len(merged), "rows_parts": sum(sizes), "rows_ok": len(merged) == sum(sizes)}
    bad, idx = [], sample_index(sizes, n, seed)
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    for g in idx:
        k = max(i for i, s in enumerate(starts) if s <= g and sizes[i])
        if not rows_equal(merged[g], parts[k][g - starts[k]]):
            bad.append(g)
    out.update({"rows_checked": len(idx), "rows_differ": bad[:20], "n_differ": len(bad),
                "ok": out["rows_ok"] and not bad})
    return out


def check_against_ref(merged, ref, batch: int = 256) -> Dict[str, Any]:
    """Every row and column of MERGED against REF, and the column types."""
    out: Dict[str, Any] = {"rows": len(merged), "rows_ref": len(ref), "columns_ok": merged.column_names ==
                           ref.column_names, "types": types(merged), "types_ref": types(ref)}
    out["types_ok"] = out["types"] == out["types_ref"]
    bad: List[int] = []
    if out["columns_ok"] and len(merged) == len(ref):
        for s in range(0, len(ref), batch):
            a, b = merged[s:s + batch], ref[s:s + batch]
            for i in range(len(a[ref.column_names[0]])):
                if any(a[c][i] != b[c][i] for c in ref.column_names):
                    bad.append(s + i)
    out.update({"rows_differ": bad[:20], "n_differ": len(bad),
                "ok": out["columns_ok"] and out["types_ok"] and len(merged) == len(ref) and not bad})
    return out


def merge(parts: Sequence[Path], out: Path, check: Optional[Path] = None, sample: int = 64, seed: int = 0,
          num_proc: Optional[int] = None) -> Dict[str, Any]:
    from datasets import DatasetDict, concatenate_datasets, load_from_disk
    parts, out = [Path(p).resolve() for p in parts], Path(out).resolve()
    tmp = out.with_name(out.name + PARTIAL)
    if len(parts) < 2:
        raise SystemExit("give two or more parts (in the training's dataset order)")
    if len(set(parts)) != len(parts):
        raise SystemExit("a part is given twice")
    for p in (out, tmp):
        if p.exists():
            raise SystemExit(f"{p} exists: the merge writes a new folder")
    t0 = time.time()
    ds = [load_part(p) for p in parts]
    empty = [str(p) for p, d in zip(parts, ds) if len(d) == 0]
    if empty:  # a dataset with no rows is left out of the training's dataset list (datasets cannot join it either)
        raise SystemExit(f"parts with no rows: {empty}")
    cols = [d.column_names for d in ds]
    if any(c != cols[0] for c in cols):
        raise SystemExit("the parts' columns differ: " + "; ".join(f"{p}: {c}" for p, c in zip(parts, cols)))
    merged = concatenate_datasets(ds)
    DatasetDict({"train": merged}).save_to_disk(str(tmp), num_proc=num_proc)
    back = load_part(tmp)
    man: Dict[str, Any] = {
        "utc": utc(), "out": str(out), "order": [str(p) for p in parts],
        "parts": [{"path": str(p), "rows": len(d), "types": types(d)} for p, d in zip(parts, ds)],
        "rows": len(back), "types": types(back), "check_parts": check_against_parts(back, ds, sample, seed)}
    if check is not None:
        man["check_ref"] = dict(check_against_ref(back, load_part(Path(check).resolve())), ref=str(check))
    man["ok"] = man["check_parts"]["ok"] and man.get("check_ref", {"ok": True})["ok"]
    man["seconds"] = round(time.time() - t0, 1)
    (tmp / MANIFEST).write_text(json.dumps(man, indent=1) + "\n")
    if man["ok"]:
        tmp.rename(out)
    return man


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--parts", nargs="+", required=True, help="tokenized parts, in the training's dataset order")
    p.add_argument("--out", required=True)
    p.add_argument("--check", help="a reference tokenized dataset of the same rows (row-by-row check)")
    p.add_argument("--sample", type=int, default=64, help="seeded rows checked against the parts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-proc", type=int, help="save_to_disk processes (default: one)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    a = parse_args(argv)
    man = merge(a.parts, Path(a.out), Path(a.check) if a.check else None, a.sample, a.seed, a.num_proc)
    short = {k: man[k] for k in ("rows", "ok", "seconds")}
    short["check_parts"] = {k: man["check_parts"][k] for k in ("rows_ok", "rows_checked", "n_differ")}
    if "check_ref" in man:
        short["check_ref"] = {k: man["check_ref"][k] for k in ("columns_ok", "types_ok", "n_differ", "rows",
                                                               "rows_ref")}
    print(json.dumps(short, indent=1))
    sys.exit(0 if man["ok"] else 1)


if __name__ == "__main__":
    main()
