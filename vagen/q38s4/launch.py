"""C line (q38 s4): run folders and sco jobs for the stage-4 SFT (multi-node accelerate + FSDP1).

  python -m vagen.q38s4.launch prepare --run RUN --mode train --arm nfp --nodes 8 --data DATA --tokenized TOK
  python -m vagen.q38s4.launch prepare --run RUN --mode tokenize --data DATA --tokenized TOK
  python -m vagen.q38s4.launch submit --run RUN --pool zoetrope          # prints the sco command (dry run)
  python -m vagen.q38s4.launch submit --run RUN --pool zoetrope --yes    # submits it

prepare writes RUN (which must not exist yet):
- code/: ``git archive`` of the worktree's committed vagen/ (the worktree must have no uncommitted change under
  vagen/), so a running job never reads files that later change; code/COMMIT names the commit;
- train.yaml: LLaMA-Factory's arguments, the stage-4 recipe (RECIPE: 1 epoch, lr 1e-5, cosine, warmup 3%, wd 0,
  global batch 256, bf16, frozen encoder, trained merger + LM, D4 video settings, save every 500 steps keeping the
  latest checkpoint) plus the paths and the batch plan; ``--set key=value`` (YAML value) changes more, each change
  listed in plan.json;
- q38s4.json: the arm's NFP weights (nfp: mse 0.1, cos 0.1; nonfp: 0, 0) and ``--logits`` (targets, the default:
  the LM head only where the loss reads it; all: LF's path, for the bit-for-bit check against the plain trainer)
  for vagen.q38s4.lf_nfp;
- q38_sft.json: bf16_copy "sibling" (every saved checkpoint gets a bf16 HF copy in <output_dir>_bf16/checkpoint-N;
  the fp32 checkpoint is LF's and the Trainer keeps only the latest), host memory sampling every 10 s;
- plan.json: mode, arm, nodes, GPUs per node, world size, per-device batch, gradient accumulation, global batch, the
  estimated optimizer steps (from DATA/build_manifest.json), the sco settings, the --set changes, the commit.

The batch plan: global batch = nodes x GPUs per node x per-device batch x gradient accumulation; a node count for
which the accumulation is not a whole number is refused (no rounding of the global batch).  The tokenize mode is one
process on a 1-card job (LF tokenizes the data once, rank 0 saves it to --tokenized, the run stops before the model
loads: lf_nfp's Q38S4_TOKENIZE_ONLY); both arms then load the same tokenized data (a train run folder is refused
while --tokenized does not exist).

submit runs nothing without --yes.  The pool is given every time (zoetrope or h800, no automatic choice); priority
NORMAL; the verl cu129 image (the wc-dev image has no libibverbs: NCCL falls back to sockets, STAGE0_REPORT section
4); the job's entry is RUN/code/vagen/q38s4/run_train.sh RUN.  Each submission appends to RUN/submits.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

WORKTREE = Path(__file__).resolve().parents[2]
MODEL_27B = "/mnt/umm/users/yinbaiqiao/q38/models/Qwen3.8-27B"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
POOLS = ("zoetrope", "h800")
IMAGE = ("registry.cn-fz-01.fjscms.com/ccr_fj2/"
         "verl:cu129_lightllm_sandbox_megatron0.14.0_vllm0.13.0R3_torch2.9.0_fa3_te2.5.0_0709")
STORAGE = "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
GLOBAL_BATCH = 256
ARMS = {"nfp": {"nfp_mse": 0.1, "nfp_cos": 0.1}, "nonfp": {"nfp_mse": 0.0, "nfp_cos": 0.0}}
MODES = ("train", "tokenize")
# the training datasets, in this order, of those DATA/dataset_info.json lists: s4_qa_img (image QA rows, a file of
# their own since data_build v2) is absent from v1 builds and from data without image rows; both arms use the same list
TRAIN_DATASETS = ("s4_qa", "s4_qa_img", "s4_nfp")
REQUIRED_DATASETS = ("s4_qa", "s4_nfp")


def train_datasets(data: Path) -> str:
    info = json.loads((Path(data) / "dataset_info.json").read_text())
    missing = [d for d in REQUIRED_DATASETS if d not in info]
    if missing:
        raise SystemExit(f"{data}/dataset_info.json lacks {missing}")
    return ",".join(d for d in TRAIN_DATASETS if d in info)

# Cambrian-S stage 4 on Qwen3.8-27B (STAGE0_REPORT sections 2-3; the 27B matrix SFT's LF settings where the stage-4
# recipe says nothing: template, thinking off, cutoff, sdpa, sync_each_batch, ddp_timeout, image pixels)
RECIPE: Dict[str, Any] = {
    "stage": "sft", "do_train": True, "finetuning_type": "full",
    "freeze_vision_tower": True, "freeze_multi_modal_projector": False,
    "template": "qwen3_8", "enable_thinking": False, "cutoff_len": 16384, "mask_history": False, "seed": 42,
    "video_fps": 2.0, "video_maxlen": 256, "video_max_pixels": 65536, "image_max_pixels": 65536,
    "per_device_train_batch_size": 1,
    "learning_rate": 1.0e-5, "num_train_epochs": 1, "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
    "weight_decay": 0.0, "max_grad_norm": 1.0,
    "bf16": True, "gradient_checkpointing": True, "flash_attn": "sdpa",
    "save_strategy": "steps", "save_steps": 500, "save_total_limit": 1, "save_only_model": True,
    "logging_steps": 1, "report_to": "none", "ddp_timeout": 10800,
    "data_shared_file_system": True, "preprocessing_num_workers": 12, "dataloader_num_workers": 4,
    "overwrite_output_dir": False,
    "accelerator_config": {"gradient_accumulation_kwargs": {"sync_each_batch": True}},
}


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def batch_plan(nodes: int, gpus_per_node: int = 8, per_device: int = 1, global_batch: int = GLOBAL_BATCH) -> Dict[str, int]:
    if nodes < 1 or gpus_per_node < 1 or per_device < 1:
        raise ValueError(f"nodes={nodes}, gpus_per_node={gpus_per_node}, per_device={per_device}")
    world = nodes * gpus_per_node
    per_step = world * per_device
    if global_batch % per_step:
        raise ValueError(f"global batch {global_batch} is not a multiple of {nodes} nodes x {gpus_per_node} GPUs x "
                         f"{per_device} per device = {per_step}")
    return {"nodes": nodes, "gpus_per_node": gpus_per_node, "world": world, "per_device": per_device,
            "grad_accum": global_batch // per_step, "global_batch": global_batch}


def estimated_steps(rows: int, plan: Dict[str, int]) -> int:
    """Optimizer steps of one epoch: each rank's sampler gets ceil(rows / world) samples, every GA of them a step
    (the last, shorter, one too)."""
    per_rank = -(-rows // plan["world"])
    return -(-per_rank // (plan["per_device"] * plan["grad_accum"]))


def parse_sets(items: List[str]) -> Dict[str, Any]:
    out = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"--set {it!r}: expected key=value")
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def train_yaml(run: Path, data: Path, tokenized: Path, model: str, plan: Dict[str, int],
               sets: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {"model_name_or_path": model, **RECIPE,
           "dataset_dir": str(data), "media_dir": str(data), "dataset": train_datasets(data),
           "tokenized_path": str(tokenized),
           "output_dir": str(run / "ckpt"),
           "per_device_train_batch_size": plan["per_device"], "gradient_accumulation_steps": plan["grad_accum"]}
    for k in ("per_device_train_batch_size", "gradient_accumulation_steps", "output_dir"):
        if k in sets:
            raise ValueError(f"--set {k}: set by the batch plan / run folder")
    cfg.update(sets)
    return cfg


def committed_head(worktree: Optional[Path] = None) -> str:
    """HEAD of the worktree, refused while vagen/ has an uncommitted change."""
    worktree = worktree or WORKTREE
    dirty = subprocess.run(["git", "-C", str(worktree), "status", "--porcelain", "--", "vagen"], check=True,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"uncommitted changes under {worktree}/vagen; commit first:\n{dirty}")
    return subprocess.run(["git", "-C", str(worktree), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def snapshot_code(dest: Path, head: str, worktree: Optional[Path] = None) -> str:
    worktree = worktree or WORKTREE
    dest.mkdir(parents=True)
    arch = subprocess.Popen(["git", "-C", str(worktree), "archive", "--format=tar", head, "vagen"],
                            stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(dest)], stdin=arch.stdout, check=True)
    if arch.wait() != 0:
        raise SystemExit("git archive failed")
    (dest / "COMMIT").write_text(head + "\n")
    return head


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    run, data, tokenized = Path(args.run).resolve(), Path(args.data).resolve(), Path(args.tokenized).resolve()
    if run.exists():
        raise SystemExit(f"{run} exists: a run folder is made once")
    if not (data / "dataset_info.json").is_file():
        raise SystemExit(f"{data}/dataset_info.json missing (vagen.q38s4.data_build output expected)")
    if args.mode == "tokenize":
        if tokenized.exists():
            raise SystemExit(f"{tokenized} exists already: nothing to tokenize")
        plan = batch_plan(1, 1, 1, GLOBAL_BATCH)  # one process; the accumulation value is never used
        arm = "nonfp"  # no head; the model is not loaded
    else:
        if args.arm not in ARMS:
            raise SystemExit(f"--arm must be one of {sorted(ARMS)}")
        if not tokenized.exists():  # else every arm's rank 0 would tokenize (and save) on its own
            raise SystemExit(f"{tokenized} missing: run the tokenize job first")
        plan = batch_plan(args.nodes, args.gpus_per_node, args.per_device, args.global_batch)
        arm = args.arm
    sets = parse_sets(args.set or [])
    cfg = train_yaml(run, data, tokenized, args.model, plan, sets)
    manifest = data / "build_manifest.json"
    rows = None
    if manifest.is_file():
        ro = json.loads(manifest.read_text())["rows_out"]
        rows = ro["qa"] + ro.get("qa_img", 0) + ro["nfp"]
    commit = committed_head()  # every check before the folder is made
    run.mkdir(parents=True)
    snapshot_code(run / "code", commit)
    (run / "logs").mkdir()
    (run / "train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run / "q38s4.json").write_text(json.dumps({**ARMS[arm], "logits": args.logits, "note": f"arm {arm}"}, indent=1)
                                    + "\n")
    (run / "q38_sft.json").write_text(json.dumps({"bf16_copy": "sibling", "host_mem_s": 10}, indent=1) + "\n")
    out = {"utc": utc(), "mode": args.mode, "arm": arm, "logits": args.logits, **plan, "commit": commit,
           "data": str(data),
           "tokenized": str(tokenized), "model": args.model, "rows": rows,
           "estimated_steps": None if rows is None or args.mode == "tokenize" else estimated_steps(rows, plan),
           "sets": sets, "worker_spec": f"N4lS.Iq.I80.{plan['gpus_per_node']}", "image": IMAGE}
    (run / "plan.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def sco_command(run: Path, plan: Dict[str, Any], pool: str, job_name: str) -> List[str]:
    if pool not in POOLS:
        raise ValueError(f"pool {pool!r} not in {POOLS}")
    entry = f"bash {run}/code/vagen/q38s4/run_train.sh {run}"
    return [SCO, "acp", "jobs", "create", "--workspace-name=aigc", f"--aec2-name={pool}", f"--job-name={job_name}",
            "--priority=NORMAL", "--quota-type=reserved", f"--container-image-url={plan['image']}",
            f"--storage-mount={STORAGE}", "--training-framework=pytorch", f"--worker-nodes={plan['nodes']}",
            f"--worker-spec={plan['worker_spec']}", f"--command={entry}"]


def submit(args: argparse.Namespace) -> List[str]:
    run = Path(args.run).resolve()
    plan = json.loads((run / "plan.json").read_text())
    name = args.job_name or f"q38s4-{plan['mode']}-{plan['arm'] if plan['mode'] == 'train' else 'tok'}-{run.name}"
    cmd = sco_command(run, plan, args.pool, name)
    print(" ".join(shlex.quote(c) for c in cmd))
    if not args.yes:
        print("(dry run: add --yes to submit)")
        return cmd
    res = subprocess.run(cmd, capture_output=True, text=True)
    rec = {"utc": utc(), "pool": args.pool, "job_name": name, "rc": res.returncode, "stdout": res.stdout,
           "stderr": res.stderr}
    with open(run / "submits.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(res.stdout + res.stderr)
    if res.returncode:
        raise SystemExit(res.returncode)
    return cmd


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run", required=True)
    p.add_argument("--mode", choices=MODES, default="train")
    p.add_argument("--arm", choices=sorted(ARMS))
    p.add_argument("--nodes", type=int, default=8)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--per-device", type=int, default=1)
    p.add_argument("--global-batch", type=int, default=GLOBAL_BATCH, help="the recipe's 256; smaller only for tests")
    p.add_argument("--data", required=True)
    p.add_argument("--tokenized", required=True)
    p.add_argument("--model", default=MODEL_27B)
    p.add_argument("--set", action="append", help="key=value (YAML) for train.yaml")
    p.add_argument("--logits", choices=("targets", "all"), default="targets",
                   help="q38s4.json logits: the LM head only at the targets (default) or at every position (LF's path)")
    s = sub.add_parser("submit")
    s.add_argument("--run", required=True)
    s.add_argument("--pool", required=True, choices=POOLS)
    s.add_argument("--job-name")
    s.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "prepare" and a.mode == "train" and not a.arm:
        ap.error("--arm is required with --mode train")
    return a


def main(argv: Optional[List[str]] = None) -> None:
    a = parse_args(argv)
    if a.cmd == "prepare":
        print(json.dumps(prepare(a), indent=1))
    else:
        submit(a)


if __name__ == "__main__":
    sys.exit(main())
