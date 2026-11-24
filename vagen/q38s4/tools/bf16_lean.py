"""C line (q38 s4): vagen.q38.sft.to_bf16 with the Trainer's resume states left out of the bf16 copy.

    python -m vagen.q38s4.tools.bf16_lean convert SRC DST [--max-shard-gb 5] [--fp32-modules A_log]

to_bf16's command line; lf_nfp's bf16 copy after each save (vagen.q38.sft.lf_launcher's callback) runs this module
instead of to_bf16.  With save_only_model false (the user's decision of 2026-11-23) a checkpoint also holds what the
Trainer reads to resume: optimizer.bin (the 27B: 215.8 GB), pytorch_model_fsdp.bin (the FSDP full state dict, the
model a second time: 108.7 GB), scheduler.pt and one rng_state_<rank>.pth per rank.  to_bf16 copies every file of the
checkpoint that is not a weight shard, so S1's bf16 copy of checkpoint-25 (2 nodes) was 380 GB instead of 55 GB and
took 1,489 s to write instead of about 180.  Here to_bf16 runs unchanged except for its side-file copy, which also
leaves out the files is_resume_state names (RESUME_PREFIXES; files only, as to_bf16 copies only files): the
conversion, the shards, the check and the manifest are to_bf16's, and the manifest also lists the files left out
("skipped_resume_state").  The fp32 checkpoint keeps every file (the Trainer resumes from it); a bf16 copy is for
evaluation and is never resumed from.  vagen.q38 itself is not changed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

from vagen.q38.sft import to_bf16 as B

MODULE = "vagen.q38s4.tools.bf16_lean"
RESUME_PREFIXES = ("optimizer", "pytorch_model_fsdp", "rng_state", "scheduler", "scaler")
_COPY_SIDE_FILES, _WRITE_COPY = B.copy_side_files, B.write_copy  # to_bf16's own, before install()


def is_resume_state(name: str) -> bool:
    return name.startswith(RESUME_PREFIXES)


def resume_files(src: Path) -> List[str]:
    return sorted(p.name for p in Path(src).iterdir() if p.is_file() and is_resume_state(p.name))


def copy_side_files(src: Path, dst: Path, weights: List[str]) -> List[str]:
    """to_bf16's side-file copy with the resume states added to the names it leaves out."""
    return _COPY_SIDE_FILES(src, dst, list(weights) + resume_files(src))


def write_copy(src: Path, dst: Path, max_shard_gb: float, fp32_modules=B.FP32_MODULES) -> Dict:
    info = _WRITE_COPY(src, dst, max_shard_gb, fp32_modules)
    info["skipped_resume_state"] = resume_files(src)
    info["wrapper"] = MODULE
    return info


def install() -> None:
    """to_bf16's convert looks both names up at call time."""
    B.copy_side_files = copy_side_files
    B.write_copy = write_copy


def main(argv: Optional[List[str]] = None) -> None:
    install()
    B.main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
