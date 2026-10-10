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
  6.  Action-only export removes think blocks and their instructions/examples
      from system, user, and assistant turns, so the prompt and label agree.

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
    "source_task_id", "score_contract",
}


def _strip_think_from_content(content: str) -> str:
    """Turn a prompt or response into a consistent action-only variant."""
    contained_think = "<think>" in content
    content = _THINK_RE.sub("", content)
    if contained_think:
        for instruction in (
            "You should first give your thought process, and then your answer.",
            "You should first describe what you observe, then reason about the actions needed, and finally provide your action.",
            "You should first reason about your actions and predict the expected outcome, then provide your action.",
            "You should describe your observation, reason about actions, predict the outcome, then provide your action.",
        ):
            content = content.replace(instruction, "You should provide only your action.")
        content = content.replace(
            "Respond in the following format:", "Respond with only the action:"
        )
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


def _validate_record(record: Dict[str, Any], image_base_dir: Path) -> None:
    """Fail closed on malformed multimodal supervision before fine-tuning."""
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("missing conversations")
    previous_role = None
    user_image_paths: List[str] = []
    for index, turn in enumerate(conversations):
        role = turn.get("role")
        content = turn.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"invalid conversation turn {index}")
        if role == "system" and index != 0:
            raise ValueError("system message must be first")
        if role != "system" and previous_role == role:
            raise ValueError(f"non-alternating role at turn {index}: {role}")
        if role == "user":
            placeholders = content.count("<image>")
            if placeholders != 1 or not turn.get("image_path"):
                raise ValueError(
                    f"user turn {index} needs exactly one <image> and image_path"
                )
            image_path = _resolve_image_path(turn["image_path"], image_base_dir)
            if not Path(image_path).is_file():
                raise ValueError(f"missing image for turn {index}: {image_path}")
            user_image_paths.append(str(turn["image_path"]))
        if role == "assistant" and "<action>" not in content:
            raise ValueError(f"assistant turn {index} has no <action> supervision")
        previous_role = role
    if conversations[-1].get("role") != "assistant":
        raise ValueError("training conversation must end with an assistant label")

    declared_paths = record.get("image_paths") or []
    if not isinstance(declared_paths, list):
        raise ValueError("image_paths must be a list")
    for index, value in enumerate(declared_paths):
        if not isinstance(value, str) or not Path(
            _resolve_image_path(value, image_base_dir)
        ).is_file():
            raise ValueError(f"missing declared trajectory image {index}: {value}")

    primitive_paths = record.get("primitive_image_paths") or []
    if primitive_paths:
        expected = int(record.get("total_actions", -1)) + 1
        if not isinstance(primitive_paths, list) or len(primitive_paths) != expected:
            raise ValueError(
                f"primitive image count mismatch: expected={expected}, "
                f"actual={len(primitive_paths) if isinstance(primitive_paths, list) else 'invalid'}"
            )
        for index, value in enumerate(primitive_paths):
            if not isinstance(value, str) or not Path(
                _resolve_image_path(value, image_base_dir)
            ).is_file():
                raise ValueError(f"missing primitive trajectory image {index}: {value}")

    expected_user_paths = record.get("conversation_image_paths")
    if expected_user_paths is None:
        # Backward compatibility for existing explicit-done records, where
        # every rendered frame was necessarily consumed by a user turn.
        expected_user_paths = declared_paths
    if not isinstance(expected_user_paths, list):
        raise ValueError("conversation_image_paths must be a list")
    if expected_user_paths != declared_paths[:len(expected_user_paths)]:
        raise ValueError("conversation images must be an ordered prefix of trajectory images")
    if user_image_paths != expected_user_paths:
        raise ValueError(
            "conversation image paths do not match user turns: "
            f"turns={user_image_paths}, declared={expected_user_paths}"
        )


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
        if strip_think:
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
    strict: bool = True,
) -> Tuple[int, int]:
    """Convert an entire JSONL file.

    Outputs either JSONL (default) or Parquet (when ``to_parquet`` is True).
    Returns (total_records, converted_records).
    """
    total = 0
    converted = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite converted dataset: {output_path}")
    temporary_path = output_path.parent / f".{output_path.name}.tmp"
    if temporary_path.exists():
        raise FileExistsError(f"stale conversion temporary file exists: {temporary_path}")

    records_out: List[Dict[str, Any]] = []

    try:
        with open(input_path, "r", encoding="utf-8") as fin:
            fout = None
            if not to_parquet:
                fout = open(temporary_path, "x", encoding="utf-8")
            try:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        record = json.loads(line)
                        if strict:
                            _validate_record(record, image_base_dir)
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
                        if strict:
                            raise ValueError(f"record {total} failed validation: {e}") from e
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
            df.to_parquet(temporary_path, index=False)
        temporary_path.replace(output_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

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
    p.add_argument(
        "--no_strict_validation", action="store_true",
        help="Skip malformed records instead of failing the conversion.",
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
        strict=not args.no_strict_validation,
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
            strict=not args.no_strict_validation,
        )
        print(f"[convert] no-think   ({fmt_label}): {nt_converted}/{total} records → {nt_path}")


if __name__ == "__main__":
    main()
