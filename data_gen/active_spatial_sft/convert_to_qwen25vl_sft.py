"""
Convert active_spatial SFT JSONL to Qwen2.5-VL SFT training format.

The generator (sft_generator.py) produces records in this internal format:

    {
        "id": "sft_000000",
        "conversations": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "...<image>...", "image_path": "images/foo.jpg"},
            {"role": "assistant", "content": "<think>...</think>\\n<action>...</action>"},
            ...
        ],
        "image_paths": ["images/foo.jpg", ...],
        ...metadata...
    }

Qwen2.5-VL (and LLaMA-Factory / ms-swift / HuggingFace TRL) expects:

    {
        "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "<image>\\n..."},
            {"role": "assistant", "content": "..."},
            ...
        ],
        "images": ["/abs/path/to/image1.jpg", "/abs/path/to/image2.jpg"]
    }

Rules applied by this converter:
  1.  "conversations"  →  "messages"
  2.  Per-turn "image_path" keys are stripped from messages (absorbed into top-level "images")
  3.  "images" list contains **absolute** paths; relative paths are resolved against
      --image_base_dir (defaults to the directory containing the input JSONL).
  4.  Metadata fields are moved into an optional "metadata" sub-dict when
      --keep_metadata is set; otherwise they are dropped to keep the file compact.
  5.  The system prompt is kept only when --no_system is NOT specified.
      (Some SFT frameworks fold the system prompt into the first user turn.)

Usage
-----
    # Basic — think mode (default)
    python convert_to_qwen25vl_sft.py \\
        --input  output_0267_v7/sft_data.jsonl \\
        --output qwen_sft_think.jsonl \\
        --image_base_dir output_0267_v7

    # Action-only mode (strip <think> blocks entirely)
    python convert_to_qwen25vl_sft.py \\
        --input  output_0267_v7/sft_data.jsonl \\
        --output qwen_sft_no_think.jsonl \\
        --strip_think

    # Produce both in one call
    python convert_to_qwen25vl_sft.py \\
        --input  output_0267_v7/sft_data.jsonl \\
        --output qwen_sft_think.jsonl \\
        --output_no_think qwen_sft_no_think.jsonl \\
        --image_base_dir output_0267_v7
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_METADATA_KEYS = {
    "id", "source_item_idx", "scene_id", "task_type", "task_description",
    "trajectory_steps", "total_actions", "initial_score", "final_score", "success",
}


def _strip_think_from_content(content: str) -> str:
    """Remove <think>...</think> blocks and clean up surrounding whitespace."""
    content = _THINK_RE.sub("", content)
    # Collapse multiple blank lines left by the removal
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    return content


def _resolve_image_path(rel_path: str, image_base_dir: Path) -> str:
    """Return an absolute path for an image, resolving relative paths."""
    p = Path(rel_path)
    if p.is_absolute():
        return str(p)
    resolved = (image_base_dir / p).resolve()
    return str(resolved)


def _convert_record(
    record: Dict[str, Any],
    image_base_dir: Path,
    strip_think: bool = False,
    keep_metadata: bool = False,
    include_system: bool = True,
) -> Dict[str, Any]:
    """Convert a single internal record to Qwen2.5-VL SFT format."""
    messages: List[Dict[str, str]] = []
    images: List[str] = []

    for turn in record.get("conversations", []):
        role = turn["role"]
        content = turn.get("content", "")

        if role == "system" and not include_system:
            continue

        # Strip oracle think blocks if requested
        if strip_think and role == "assistant":
            content = _strip_think_from_content(content)

        # Collect image paths from user turns
        if role == "user" and "image_path" in turn:
            abs_path = _resolve_image_path(turn["image_path"], image_base_dir)
            images.append(abs_path)

        messages.append({"role": role, "content": content})

    out: Dict[str, Any] = {"messages": messages, "images": images}

    if keep_metadata:
        out["metadata"] = {k: record[k] for k in _METADATA_KEYS if k in record}

    return out


def convert_jsonl(
    input_path: Path,
    output_path: Path,
    image_base_dir: Path,
    strip_think: bool = False,
    keep_metadata: bool = False,
    include_system: bool = True,
    to_parquet: bool = False,
) -> Tuple[int, int]:
    """Convert an entire JSONL file.

    Outputs either JSONL (default) or Parquet (when ``to_parquet`` is True).
    Returns (total_records, converted_records).
    """
    total = 0
    converted = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records_out: List[Dict[str, Any]] = []

    with open(input_path, "r", encoding="utf-8") as fin:
        fout = None
        if not to_parquet:
            fout = open(output_path, "w", encoding="utf-8")
        try:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    record = json.loads(line)
                    out = _convert_record(
                        record,
                        image_base_dir=image_base_dir,
                        strip_think=strip_think,
                        keep_metadata=keep_metadata,
                        include_system=include_system,
                    )
                    if to_parquet:
                        records_out.append(out)
                    else:
                        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    converted += 1
                except Exception as e:
                    print(f"[convert] Skipping malformed record: {e}", file=sys.stderr)
        finally:
            if fout is not None:
                fout.close()

    if to_parquet:
        try:
            import pandas as pd  # local import — only needed for parquet mode
        except ImportError as e:
            raise SystemExit(
                "[convert] --to_parquet requires pandas + pyarrow. "
                "Install with: pip install pandas pyarrow"
            ) from e
        df = pd.DataFrame.from_records(records_out)
        df.to_parquet(output_path, index=False)

    return total, converted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert active_spatial SFT JSONL to Qwen2.5-VL format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input", required=True,
        help="Path to the source sft_data.jsonl produced by sft_generator.py.",
    )
    p.add_argument(
        "--output", required=True,
        help="Output JSONL path for the with-think version.",
    )
    p.add_argument(
        "--output_no_think", default=None,
        help="If provided, also write an action-only (no <think>) version to this path.",
    )
    p.add_argument(
        "--image_base_dir", default=None,
        help=(
            "Directory that serves as the base for resolving relative image paths. "
            "Defaults to the directory containing --input."
        ),
    )
    p.add_argument(
        "--strip_think", action="store_true",
        help="Strip <think> blocks from the primary --output as well.",
    )
    p.add_argument(
        "--keep_metadata", action="store_true",
        help="Preserve metadata (scene_id, task_type, …) in a 'metadata' sub-dict.",
    )
    p.add_argument(
        "--no_system", action="store_true",
        help="Omit the system prompt turn (fold into user context instead).",
    )
    p.add_argument(
        "--to_parquet", action="store_true",
        help=(
            "Write output as Parquet (DataFrame with `messages` + `images` columns) "
            "instead of JSONL.  Required by the verl SFT trainer."
        ),
    )
    return p.parse_args()


def main():
    args = _parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if args.image_base_dir:
        image_base_dir = Path(args.image_base_dir).resolve()
    else:
        image_base_dir = input_path.parent

    include_system = not args.no_system

    # ── Primary output (with or without think depending on --strip_think) ────
    total, converted = convert_jsonl(
        input_path=input_path,
        output_path=output_path,
        image_base_dir=image_base_dir,
        strip_think=args.strip_think,
        keep_metadata=args.keep_metadata,
        include_system=include_system,
        to_parquet=args.to_parquet,
    )
    think_label = "no-think" if args.strip_think else "with-think"
    fmt_label = "parquet" if args.to_parquet else "jsonl"
    print(f"[convert] {think_label} ({fmt_label}): {converted}/{total} records → {output_path}")

    # ── Optional no-think output ─────────────────────────────────────────────
    if args.output_no_think:
        nt_path = Path(args.output_no_think).resolve()
        _, nt_converted = convert_jsonl(
            input_path=input_path,
            output_path=nt_path,
            image_base_dir=image_base_dir,
            strip_think=True,
            keep_metadata=args.keep_metadata,
            include_system=include_system,
            to_parquet=args.to_parquet,
        )
        print(f"[convert] no-think   ({fmt_label}): {nt_converted}/{total} records → {nt_path}")


if __name__ == "__main__":
    main()
