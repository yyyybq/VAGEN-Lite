"""q38: the LLaMA-Factory folder of a Qwen3.8-27B SFT, made from an 8B one (SFT_OPTIONS.md option A, TEMPLATE_CHECK.md).

The 8B folder (e.g. models_runs/chains/m_qwen3vl8b_formal_sft_20261026/lf_q3formal) holds sharegpt jsonl files, their
dataset_info.json and the recipe train.yaml.  The 27B folder gets:

- every dataset file of the source, each message's content stripped of leading and trailing whitespace (Python
  ``str.strip()``, which is what jinja's ``|trim`` does): Qwen3.8's chat template trims every message
  (chat_template.jinja lines 70, 78, 92, 103), so the RL agent loop and the evaluator see the trimmed text, while
  LLaMA-Factory's qwen3_8 template strips only the system turn (TEMPLATE_CHECK conclusion 2).  Nothing else of a record
  changes (images, videos, roles, order);
- dataset_info.json: the source's entries of the datasets written;
- train.yaml: the 8B recipe with only what the model needs changed -- model_name_or_path, template qwen3_8,
  enable_thinking false (the empty thinking block <think>\\n\\n</think>\\n\\n goes to the prompt side of every
  assistant turn and is not trained; LLaMA-Factory's DataArguments.enable_thinking defaults to true, which would train
  it), dataset_dir / output_dir -- plus ddp_timeout (the rank-0 save of a 27B fp32 checkpoint; 30 min is the default);
  a profile (smoke) and --set change more, and every change is listed in the manifest;
- build_manifest.json: source and output files with sha256 and line counts, the strip counts per role, the selected
  line numbers of a subset, the recipe changes.

    python -m vagen.q38.sft.lf_data build --src 8B_LF_DIR --out OUT --model MODEL [--profile formal|smoke]
        [--limit trajectory_train=400 --limit trajectory_val=64] [--seed 42] [--set key=value ...]

Records are checked as they are read: roles system? (user assistant)+, one <image> per image, one <video> per video,
no thinking tags, no message empty after stripping.  A record that fails stops the build (nothing is dropped
silently).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

TEMPLATE = "qwen3_8"
ROLES = ("system", "user", "assistant")
THINK_TAGS = ("<think>", "</think>")
# the recipe of the 8B SFT that must not change with the model (handoff section 4, PREREG_Q38_DRAFT row "SFT");
# build() refuses a source train.yaml that differs from it, so a wrong source folder cannot pass for C0's recipe
RECIPE_8B = {
    "stage": "sft", "do_train": True, "finetuning_type": "full", "freeze_vision_tower": True,
    "freeze_multi_modal_projector": False, "cutoff_len": 16384, "mask_history": False, "seed": 42,
    "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8, "learning_rate": 1.0e-5,
    "num_train_epochs": 10, "lr_scheduler_type": "cosine", "warmup_ratio": 0.03, "bf16": True,
    "gradient_checkpointing": True, "flash_attn": "sdpa", "save_strategy": "epoch", "save_only_model": True,
}
DDP_TIMEOUT_S = 10800
# the launcher's per-epoch bf16 copy (vagen.q38.sft.lf_launcher): sibling keeps the fp32 checkpoint, in_place replaces
# its shards (deletes fp32: only after the user's OK, P1_REPORT decisions 4 / 5), off makes none
BF16_COPY = ("sibling", "in_place", "off")
PROFILES: Dict[str, Dict[str, Any]] = {
    "formal": {},
    # SMOKE_PLAN 2c: one epoch of a subset, a log line every optimizer step (LogCallback's trainer_log.jsonl gives the
    # per-step seconds), the validation subset evaluated once at the end of the epoch
    "smoke": {"num_train_epochs": 1, "logging_steps": 1, "eval_strategy": "epoch"},
}


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strip_record(rec: Dict[str, Any], where: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """The record with every message's content stripped, and {role/leading|trailing: 1} for what changed.
    Raises ValueError on a record LLaMA-Factory or the chat template would read differently from the RL agent loop."""
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError(f"{where}: no messages")
    roles = [m.get("role") for m in msgs]
    body = roles[1:] if roles[0] == "system" else roles
    if not body or any(r != ("user" if i % 2 == 0 else "assistant") for i, r in enumerate(body)) or body[-1] != "assistant":
        raise ValueError(f"{where}: roles {roles} are not system? (user assistant)+")
    changed: Dict[str, int] = {}
    out = []
    n_image = n_video = 0
    for m in msgs:
        content = m.get("content")
        if not isinstance(content, str):
            raise ValueError(f"{where}: a {m.get('role')} content is not a string")
        if any(t in content for t in THINK_TAGS):
            raise ValueError(f"{where}: a {m['role']} message holds a thinking tag")
        if m["role"] != "user" and ("<image>" in content or "<video>" in content):
            raise ValueError(f"{where}: a {m['role']} message holds an <image> / <video> placeholder")
        n_image += content.count("<image>")
        n_video += content.count("<video>")
        new = content.strip()
        if not new:
            raise ValueError(f"{where}: a {m['role']} message is empty after stripping")
        if new != content:
            if content[: len(content) - len(content.lstrip())]:
                changed[f"{m['role']}/leading"] = 1
            if content[len(content.rstrip()):]:
                changed[f"{m['role']}/trailing"] = 1
            changed[f"{m['role']}/any"] = 1
        out.append({**m, "content": new})
    if n_image != len(rec.get("images") or []):
        raise ValueError(f"{where}: {n_image} <image> placeholders, {len(rec.get('images') or [])} images")
    if n_video != len(rec.get("videos") or []):
        raise ValueError(f"{where}: {n_video} <video> placeholders, {len(rec.get('videos') or [])} videos")
    return {**rec, "messages": out}, changed


def select_lines(n: int, limit: Optional[int], seed: int) -> List[int]:
    """All line numbers, or ``limit`` of them drawn with ``seed`` (kept in file order)."""
    if limit is None or limit >= n:
        return list(range(n))
    return sorted(random.Random(seed).sample(range(n), limit))


def convert_file(src: Path, dst: Path, limit: Optional[int], seed: int) -> Dict[str, Any]:
    lines = [l for l in src.read_text().splitlines() if l.strip()]
    keep = select_lines(len(lines), limit, seed)
    counts: Dict[str, int] = {}
    messages_by_role: Dict[str, int] = {}
    with open(dst, "w") as f:
        for i in keep:
            rec, changed = strip_record(json.loads(lines[i]), f"{src.name}:{i + 1}")
            for k, v in changed.items():
                counts[k] = counts.get(k, 0) + v
            for m in rec["messages"]:
                messages_by_role[m["role"]] = messages_by_role.get(m["role"], 0) + 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"source": str(src), "source_sha256": sha256(src), "source_records": len(lines), "records": len(keep),
            "selected_lines": keep if len(keep) < len(lines) else "all", "seed": seed if len(keep) < len(lines) else None,
            "messages_by_role": messages_by_role, "messages_stripped": counts, "output": str(dst),
            "output_sha256": sha256(dst)}


def parse_value(text: str) -> Any:
    return yaml.safe_load(text)


def recipe(src_yaml: Dict[str, Any], model: str, out: Path, profile: str, sets: Dict[str, Any],
           eval_dataset: Optional[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(the 27B train.yaml, {key: [8B value, 27B value]} of every change)."""
    wrong = {k: (src_yaml.get(k), v) for k, v in RECIPE_8B.items() if src_yaml.get(k) != v}
    if wrong:
        raise ValueError(f"the source train.yaml is not the 8B recipe: {wrong}")
    cfg = dict(src_yaml)
    cfg.update(model_name_or_path=model, template=TEMPLATE, enable_thinking=False, dataset_dir=str(out),
               output_dir=str(out / "model"), ddp_timeout=DDP_TIMEOUT_S)
    if eval_dataset is None:
        for k in ("eval_dataset", "eval_strategy", "eval_steps", "per_device_eval_batch_size"):
            cfg.pop(k, None)
        cfg["do_eval"] = False
    cfg.update(PROFILES[profile])
    cfg.update(sets)
    changes = {k: [src_yaml.get(k), cfg.get(k)] for k in sorted(set(src_yaml) | set(cfg)) if src_yaml.get(k) != cfg.get(k)}
    return cfg, changes


def git_head(path: Path) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build(src: Path, out: Path, model: str, profile: str = "formal", limits: Optional[Dict[str, int]] = None,
          seed: int = 42, sets: Optional[Dict[str, Any]] = None, datasets: Optional[Iterable[str]] = None,
          bf16_copy: str = "sibling") -> Dict[str, Any]:
    limits, sets = dict(limits or {}), dict(sets or {})
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}")
    if bf16_copy not in BF16_COPY:
        raise ValueError(f"bf16_copy must be one of {BF16_COPY}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give a new folder")
    info = json.loads((src / "dataset_info.json").read_text())
    src_yaml = yaml.safe_load((src / "train.yaml").read_text())
    names = list(datasets) if datasets else [n for n in info if (src / info[n]["file_name"]).is_file()
                                            and (src / info[n]["file_name"]).stat().st_size > 0]
    train = src_yaml.get("dataset")
    evald = src_yaml.get("eval_dataset")
    for need in (train, evald):
        if need and need not in names:
            raise ValueError(f"the recipe trains / evaluates on {need}, which is not converted (datasets {names})")
    unknown = set(limits) - set(names)
    if unknown:
        raise ValueError(f"--limit names datasets that are not converted: {sorted(unknown)}")
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in names:
        entry = info[name]
        files[name] = convert_file(src / entry["file_name"], out / entry["file_name"], limits.get(name), seed)
    (out / "dataset_info.json").write_text(json.dumps({n: info[n] for n in names}, indent=2) + "\n")
    cfg, changes = recipe(src_yaml, model, out, profile, sets, evald)
    (out / "train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    settings = {"bf16_copy": bf16_copy, "host_mem_s": 10}
    (out / "q38_sft.json").write_text(json.dumps(settings, indent=1) + "\n")  # read by vagen.q38.sft.lf_launcher
    manifest = {"utc": utc(), "tool": "vagen.q38.sft.lf_data", "worktree_head": git_head(Path(__file__).resolve().parents[3]),
                "source_dir": str(src), "source_train_yaml_sha256": sha256(src / "train.yaml"), "model": model,
                "profile": profile, "recipe_changes": changes, "launcher_settings": settings, "datasets": files,
                "strip_rule": "every message content -> str.strip() (= jinja |trim, chat_template.jinja 70/78/92/103)"}
    (out / "build_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return manifest


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    b.add_argument("--model", required=True)
    b.add_argument("--profile", default="formal", choices=sorted(PROFILES))
    b.add_argument("--limit", action="append", default=[], help="DATASET=N: a seeded subset of N records")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--set", action="append", default=[], help="KEY=VALUE (yaml value) for train.yaml")
    b.add_argument("--datasets", default=None, help="comma-separated dataset names (default: every non-empty file)")
    b.add_argument("--bf16-copy", default="sibling", choices=BF16_COPY, help="q38_sft.json for the launcher")
    a = ap.parse_args(argv)
    limits = {k: int(v) for k, v in (s.split("=", 1) for s in a.limit)}
    sets = {k: parse_value(v) for k, v in (s.split("=", 1) for s in a.set)}
    m = build(a.src, a.out, a.model, a.profile, limits, a.seed, sets, a.datasets.split(",") if a.datasets else None,
              a.bf16_copy)
    summary = {n: {k: d[k] for k in ("records", "source_records", "messages_stripped")} for n, d in m["datasets"].items()}
    print(json.dumps({"out": str(a.out), "profile": m["profile"], "datasets": summary,
                      "recipe_changes": m["recipe_changes"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
