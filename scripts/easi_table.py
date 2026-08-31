#!/usr/bin/env python3
"""Render checkpoint comparisons from official EASI result payloads."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = SCRIPT_DIR / "easi_registry.yaml"
OUTPUT_DIR = SCRIPT_DIR.parent / "easi_results"
EASI_8 = [
    "vsi_bench", "mmsi_bench", "mindcube_tiny", "viewspatial",
    "site", "blink", "3dsrbench", "embspatial",
]
DISPLAY = {
    "vsi_bench": "VSI-Bench",
    "mmsi_bench": "MMSI-Bench",
    "mindcube_tiny": "MindCube-Tiny",
    "viewspatial": "ViewSpatial",
    "site": "SITE",
    "blink": "BLINK",
    "3dsrbench": "3DSRBench",
    "embspatial": "EmbSpatial",
}


def normalize(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value / 100.0 if abs(value) > 1.0 else value


def load_payload(output_dir: Path, checkpoint: str) -> dict[str, Any] | None:
    path = output_dir / checkpoint / "easi_results.json"
    return json.loads(path.read_text()) if path.exists() else None


def render_table(registry: dict[str, Any], benchmarks: list[str], output_dir: Path) -> str:
    checkpoints = registry.get("checkpoints", {})
    results: dict[str, dict[str, float | None]] = {}
    for name in checkpoints:
        payload = load_payload(output_dir, name)
        if payload:
            raw = payload.get("scores", {})
            results[name] = {bench: normalize(raw.get(bench)) for bench in benchmarks}

    names = list(results)
    base_name = "base" if "base" in results else (names[0] if names else None)
    now = time.strftime("%Y-%m-%d %H:%M")
    lines = [
        "# EASI-8 Results",
        "",
        f"> Official EASI payloads · updated {now} · `—` = missing",
        "",
        "| Checkpoint | EASI-8 macro | " + " | ".join(DISPLAY.get(b, b) for b in benchmarks) + " |",
        "| --- | ---: | " + " | ".join(["---:"] * len(benchmarks)) + " |",
    ]
    for name in checkpoints:
        scores = results.get(name, {})
        values = [scores.get(bench) for bench in benchmarks]
        present = [v for v in values if v is not None]
        macro = sum(present) / len(present) if len(present) == len(benchmarks) else None
        cells = [name, "—" if macro is None else f"{macro * 100:.1f}%"]
        for bench, value in zip(benchmarks, values):
            if value is None:
                cells.append("—")
                continue
            text = f"{value * 100:.1f}%"
            base = results.get(base_name or "", {}).get(bench)
            if base is not None and name != base_name:
                text += f" ({(value - base) * 100:+.1f})"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")

    complete = sum(
        1 for scores in results.values()
        if all(scores.get(bench) is not None for bench in benchmarks)
    )
    lines += ["", f"**Complete EASI-8 checkpoints: {complete}/{len(checkpoints)}**", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render official EASI result table")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--registry", default=str(REGISTRY_PATH))
    parser.add_argument("--benchmarks", default="easi_8")
    parser.add_argument("--out")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()

    registry = yaml.safe_load(Path(args.registry).read_text()) or {}
    groups = registry.get("benchmark_groups", {})
    benchmarks = groups.get(args.benchmarks) or [
        item.strip() for item in args.benchmarks.split(",") if item.strip()
    ]
    output_dir = Path(args.output_dir)

    def run_once() -> None:
        table = render_table(registry, benchmarks, output_dir)
        print(table)
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(table)

    if not args.watch:
        run_once()
        return
    try:
        while True:
            os.system("clear")
            run_once()
            time.sleep(60)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
