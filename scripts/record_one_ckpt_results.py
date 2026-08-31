#!/usr/bin/env python3
"""Summarize one-ckpt nav+EASI run and optionally upsert MASTER_RESULTS.md."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite")
MASTER = ROOT / "exps/unified_eval_runs/MASTER_RESULTS.md"

NAV_SUITES = [
    "id_test_stratified400",
    "ood_category",
    "ood_scene",
    "ood_instance",
    "ood_geometry",
    "ood_template",
    "ood_v2_centering",
]
OOD5 = ["ood_category", "ood_scene", "ood_instance", "ood_geometry", "ood_template"]
EASI8 = [
    "vsi_bench",
    "mmsi_bench",
    "mindcube_tiny",
    "viewspatial",
    "site",
    "blink",
    "3dsrbench",
    "embspatial",
]
EASI_COL = {
    "vsi_bench": "VSI",
    "mmsi_bench": "MMSI",
    "mindcube_tiny": "MindCube",
    "viewspatial": "ViewSpatial",
    "site": "SITE",
    "blink": "BLINK",
    "3dsrbench": "3DSR",
    "embspatial": "EmbSpatial",
}


def pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{100.0 * x:.1f}"


def load_nav(run_dir: Path, exp: str, step: int) -> dict[str, float | None]:
    out: dict[str, float | None] = {s: None for s in NAV_SUITES}
    base = run_dir / "nav" / "sweep" / exp / f"global_step_{step}"
    for suite in NAV_SUITES:
        p = base / suite / "model" / "results_model.json"
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
            sr = (((data.get("metrics") or {}).get("overall") or {}).get("success_rate"))
            if isinstance(sr, (int, float)):
                out[suite] = float(sr)
        except Exception:
            pass
    return out


def load_easi(run_dir: Path, easi_key: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {b: None for b in EASI8}
    final = run_dir / "easi" / easi_key / "easi_results.json"
    payloads: list[dict[str, Any]] = []
    if final.exists():
        try:
            payloads.append(json.loads(final.read_text()))
        except Exception:
            pass
    for bench in EASI8:
        for p in (run_dir / "easi").glob(f"{easi_key}__shard_{bench}/**/easi_results.json"):
            try:
                payloads.append(json.loads(p.read_text()))
            except Exception:
                pass
    for payload in payloads:
        scores = payload.get("scores") or {}
        for bench in EASI8:
            if out[bench] is not None:
                continue
            v = scores.get(bench)
            if isinstance(v, (int, float)):
                # EASI scores may already be percent or fraction.
                out[bench] = float(v) / 100.0 if abs(float(v)) > 1.0 else float(v)
    return out


def mean(vals: list[float | None]) -> float | None:
    xs = [v for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else None


def fmt_nav_row(label: str, nav: dict[str, float | None], notes: str) -> str:
    id400 = pct(nav.get("id_test_stratified400"))
    ood5 = pct(mean([nav.get(s) for s in OOD5]))
    cells = [
        label,
        id400,
        ood5,
        pct(nav.get("ood_category")),
        pct(nav.get("ood_scene")),
        pct(nav.get("ood_instance")),
        pct(nav.get("ood_geometry")),
        pct(nav.get("ood_template")),
        pct(nav.get("ood_v2_centering")),
        notes,
    ]
    return "| " + " | ".join(cells) + " |"


def fmt_easi_row(label: str, easi: dict[str, float | None], notes: str) -> str:
    scores = [easi.get(b) for b in EASI8]
    macro = pct(mean(scores) if all(s is not None for s in scores) else None)
    cells = [label, macro] + [pct(easi.get(b)) for b in EASI8] + [notes]
    return "| " + " | ".join(cells) + " |"


def fmt_combined_row(label: str, nav: dict[str, float | None], easi: dict[str, float | None]) -> str:
    id400 = pct(nav.get("id_test_stratified400"))
    ood5 = pct(mean([nav.get(s) for s in OOD5]))
    scores = [easi.get(b) for b in EASI8]
    macro = pct(mean(scores) if all(s is not None for s in scores) else None)
    return f"| {label} | {id400} | {ood5} | {macro} |"


def upsert_table_row(md: str, section_header: str, label: str, new_row: str) -> str:
    # Find table under section_header; replace row with same first-col label or append before blank line after table.
    sec = md.find(section_header)
    if sec < 0:
        return md
    # Find first markdown table after section
    t0 = md.find("\n| ", sec)
    if t0 < 0:
        return md
    t0 += 1
    # Table ends at first blank line after header+rows
    end = md.find("\n\n", t0)
    if end < 0:
        end = len(md)
    table = md[t0:end].strip("\n")
    lines = table.splitlines()
    if len(lines) < 2:
        return md
    header, sep, *rows = lines  # must unpack lines, not (str,str,list)
    key = f"| {label} |"
    replaced = False
    new_rows = []
    for r in rows:
        if r.startswith(key) or r.startswith(f"| {label} |"):
            new_rows.append(new_row)
            replaced = True
        else:
            new_rows.append(r)
    if not replaced:
        new_rows.append(new_row)
    new_table = "\n".join([header, sep] + new_rows)
    return md[:t0] + new_table + md[end:]


def bump_stamp(md: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return re.sub(
        r"\*\*Last updated:\*\*.*",
        f"**Last updated:** {stamp}",
        md,
        count=1,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--update-master", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    meta = {}
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    easi_key = meta.get("easi_key") or f"{args.exp}_step{args.step}"

    nav = load_nav(run_dir, args.exp, args.step)
    easi = load_easi(run_dir, easi_key)
    notes = f"one_ckpt eval `{run_dir.name}`"

    summary = {
        "label": args.label,
        "exp": args.exp,
        "step": args.step,
        "easi_key": easi_key,
        "checkpoint": meta.get("checkpoint"),
        "model_type": meta.get("model_type"),
        "nav_success": nav,
        "nav_id400": nav.get("id_test_stratified400"),
        "nav_ood5": mean([nav.get(s) for s in OOD5]),
        "easi_scores": easi,
        "easi_macro": mean([easi.get(b) for b in EASI8])
        if all(easi.get(b) is not None for b in EASI8)
        else None,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
    }
    (run_dir / "results_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    md_lines = [
        f"# Results: {args.label}",
        "",
        f"- exp: `{args.exp}` step `{args.step}`",
        f"- ckpt: `{summary.get('checkpoint')}`",
        f"- run: `{run_dir}`",
        "",
        "## Navigation (success %)",
        "",
        fmt_nav_row(args.label, nav, notes),
        "",
        "## EASI-8 (score %)",
        "",
        fmt_easi_row(args.label, easi, notes),
        "",
        "## Combined",
        "",
        fmt_combined_row(args.label, nav, easi),
        "",
    ]
    (run_dir / "results_summary.md").write_text("\n".join(md_lines))
    print(json.dumps(summary, indent=2))
    print(f"[wrote] {run_dir / 'results_summary.json'}")
    print(f"[wrote] {run_dir / 'results_summary.md'}")

    if args.update_master:
        text = MASTER.read_text()
        text = upsert_table_row(text, "## Navigation (ID / OOD)", args.label, fmt_nav_row(args.label, nav, notes))
        text = upsert_table_row(text, "## EASI-8 QA", args.label, fmt_easi_row(args.label, easi, notes))
        text = upsert_table_row(
            text, "## Combined snapshot", args.label, fmt_combined_row(args.label, nav, easi)
        )
        # Ensure Sources mentions this run
        src_line = f"- One-ckpt `{args.label}`: `{run_dir.relative_to(ROOT)}/`"
        if str(run_dir) not in text and src_line not in text:
            text = text.replace("## Sources\n", "## Sources\n\n" + src_line + "\n", 1)
        text = bump_stamp(text)
        MASTER.write_text(text)
        print(f"[updated] {MASTER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
