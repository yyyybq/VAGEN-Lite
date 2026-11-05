"""Understanding samples mixed into the RL actor's update (the reward line, the user's task of 2026-11-05, after
NEO-VLA, arXiv 2611.06918: a VLA keeps its understanding when every other optimizer step adds 0.1 x the next-token
loss of a batch of understanding samples to the action loss).

On the actor's optimizer step k (k = the steps it took before, read from the optimizer state, so a resumed run counts
on) with k % interval == 0, ``batch_size`` understanding samples (all ranks together, the same number on each) add
``weight`` x their next-token loss on the answer tokens (the token mean over the whole batch) to the policy loss.  The
policy gradient is set aside first, the samples' gradient computed, both measured before clipping (norms, cosine),
then added; the actor clips and steps as it always does.  The other steps are the actor's own.

The samples are question-answer conversations the start model's SFT trained on (``prepare`` copies them from the SFT
folder, minus every row that shares a question id, task, scene or image with the held-out question sets), drawn
uniformly (``batch_size`` distinct rows per batch, from the seed and k alone, so a resumed run draws the same).  They
go through the matrix SFT's own LLaMA-Factory path: the model line's Qwen3-VL registrations
(model_matrix.sft.lf_qwen3vl), the SFT recipe's template, cutoff, history masking and image / video pixel limits,
LLaMA-Factory's dataset converter, SupervisedDatasetProcessor and SFTDataCollatorWith4DAttentionMask.  Only the
forward pass is the actor's (_forward_micro_batch, the one the policy loss uses), with the RL path's four position
rows (text positions, then the collator's M-RoPE rows).

Off unless the PPO config's actor.und_mix.enable is true: verl's dp_actor builds the mixer then
(training_prep rl-config --und-mix, train_chain plan rl "und_mix").

    python -m vagen.und_mix prepare --sft-dir SFT --datasets P0_train,P1_train --probe P1=PROBE.jsonl \
        [--questions X0=QUESTIONS.jsonl ...] --out POOL
    python -m vagen.und_mix check --pool POOL --model MODEL [--rows N] --out FILE.json   # CPU, LLaMA-Factory path
    python -m vagen.und_mix counts --pool POOL --log RUN/und_mix/steps.jsonl --out FILE.json   # draws per row
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import random
import shutil
import statistics
import sys
import time
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULTS = {"interval": 2, "weight": 0.1, "batch_size": 32, "micro_batch_size": 1, "seed": 20261105}
REQUEST_KEYS = set(DEFAULTS) | {"pool"}
POOL_DATASET = "und_pool"
IGNORE_INDEX = -100
# the recipe keys that shape a sample (LLaMA-Factory's data and processor arguments); the rest is the SFT's training
RECIPE_DATA_KEYS = ("template", "cutoff_len", "mask_history", "train_on_prompt", "tool_format", "default_system",
                    "enable_thinking")
RECIPE_MODEL_KEYS = ("image_max_pixels", "image_min_pixels", "video_max_pixels", "video_min_pixels", "video_fps",
                     "video_maxlen")
ROW_KEYS = ("und_index", "und_dataset", "und_line")  # what prepare adds to a row


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------------------------------- the PPO block

def ppo_block(request: Any, model: str, run_root: Any) -> Dict[str, Any]:
    """actor.und_mix of a PPO config from a request (a dict, a JSON text or a JSON file's path): "pool" (required,
    made by ``prepare``) and optionally interval, weight, batch_size, micro_batch_size, seed.  The tokenizer and the
    model config are the RL start model's; the mixer writes into RUN/und_mix.  ValueError on a bad request."""
    if isinstance(request, str):
        text = request.strip()
        request = json.loads(Path(text).read_text() if text.endswith(".json") and Path(text).is_file() else text)
    if not isinstance(request, dict):
        raise ValueError(f"a JSON object, not {type(request).__name__}")
    unknown = sorted(set(request) - REQUEST_KEYS)
    if unknown:
        raise ValueError(f"unknown keys {unknown} (known: {sorted(REQUEST_KEYS)})")
    if not request.get("pool"):
        raise ValueError("no pool (python -m vagen.und_mix prepare)")
    pool = Path(request["pool"]).resolve()
    if not (pool / "manifest.json").is_file():
        raise ValueError(f"{pool} has no manifest.json (python -m vagen.und_mix prepare)")
    manifest = json.loads((pool / "manifest.json").read_text())
    block: Dict[str, Any] = {"enable": True, "pool": str(pool)}
    for key, default in DEFAULTS.items():
        value = request.get(key, default)
        block[key] = float(value) if key == "weight" else int(value)
        if isinstance(value, bool) or (key != "weight" and int(value) != value):
            raise ValueError(f"{key} must be a number, not {value!r}")
    if block["interval"] < 1 or block["batch_size"] < 1 or block["micro_batch_size"] < 1 or not block["weight"] > 0:
        raise ValueError(f"interval, batch_size and micro_batch_size must be at least 1, weight above 0: {block}")
    if block["batch_size"] > manifest["rows"]:
        raise ValueError(f"batch_size {block['batch_size']} is more than the pool's {manifest['rows']} rows")
    block.update(tokenizer=str(model), log_dir=str(Path(run_root).resolve() / "und_mix"), pool_rows=manifest["rows"],
                 pool_sha256=manifest["pool_sha256"])
    return block


# ---------------------------------------------------------------------------------------------------- the pool

def _held_out(name: str, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """The identities of a held-out question set: question ids, scenes, tasks, image paths.  P1 probe rows
    (sample_id, scene, task_id, images) and X0 / G0 rows (id, image_path, task, seed: their scene is RoboCasa's
    task and reset seed) both work."""
    ids, scenes, tasks, images = set(), set(), set(), set()
    n = 0
    for r in rows:
        n += 1
        for key in ("sample_id", "id", "source_id"):
            if r.get(key) is not None:
                ids.add(str(r[key]))
        if r.get("scene") is not None:
            scenes.add(str(r["scene"]))
        elif r.get("task") is not None and r.get("seed") is not None:
            scenes.add(f"robocasa/{r['task']}/{r['seed']}")
        if r.get("task_id") is not None:
            tasks.add(str(r["task_id"]))
        for key in ("images", "image_path", "image"):
            value = r.get(key)
            for p in (value if isinstance(value, list) else [value] if value else []):
                images.add(str(Path(p)))
    return {"name": name, "questions": n, "ids": ids, "scenes": scenes, "tasks": tasks, "images": images}


def _row_identity(r: Dict[str, Any]) -> Dict[str, set]:
    meta = r.get("meta") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    ids = {str(r["source_id"])} if r.get("source_id") is not None else set()
    sid = str(r.get("source_id") or "")
    for q in meta.get("qids") or []:  # P0 rows: the bank's question ids (source_id = "p0_" + the first)
        ids |= {str(q), f"p0_{q}"}
    if "_" in sid:
        ids.add(sid.split("_", 1)[1])
    media = [str(Path(p)) for p in r.get("images") or []]
    for v in r.get("videos") or []:
        media += [str(Path(p)) for p in (v if isinstance(v, list) else [v])]
    return {"ids": ids, "scenes": {str(r["scene"])} if r.get("scene") is not None else set(),
            "tasks": {str(meta["task_id"])} if meta.get("task_id") else set(), "images": set(media)}


def prepare(sft_dir: Path, datasets: Sequence[str], held: Sequence[Tuple[str, Path]], out: Path) -> Dict[str, Any]:
    """Copy the SFT folder's rows of ``datasets`` into OUT/pool.jsonl (each with und_index / und_dataset / und_line),
    minus every row that shares a question id, a task, a scene or an image with a held-out set, and every row without
    an image or a video (LLaMA-Factory's collator would add a blank image the actor cannot place).  Also writes
    OUT/dataset_info.json (one dataset, und_pool, with the SFT datasets' own columns and tags), OUT/lf_recipe.yaml
    (the SFT's train.yaml, read for the template, cutoff and pixel limits) and OUT/manifest.json."""
    sft_dir, out = Path(sft_dir), Path(out)
    info = json.loads((sft_dir / "dataset_info.json").read_text())
    attrs = []
    for d in datasets:
        if d not in info:
            raise ValueError(f"{d} is not in {sft_dir / 'dataset_info.json'}")
        attrs.append({k: v for k, v in info[d].items() if k != "file_name"})
    if any(a != attrs[0] for a in attrs[1:]):
        raise ValueError(f"the datasets {list(datasets)} have different columns or tags: {attrs}")
    sets = [_held_out(name, _jsonl(Path(path))) for name, path in held]
    out.mkdir(parents=True, exist_ok=True)
    kept: List[Dict[str, Any]] = []
    excluded: Dict[str, Counter] = {s["name"]: Counter() for s in sets}
    excluded["no_media"] = Counter()
    per_dataset: Dict[str, Dict[str, int]] = {}
    sources: Dict[str, Dict[str, Any]] = {}
    for d in datasets:
        path = sft_dir / info[d]["file_name"]
        sources[d] = {"file": str(path), "sha256": _sha256(path)}
        n_rows = n_kept = 0
        for line_no, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            n_rows += 1
            r = json.loads(line)
            ident = _row_identity(r)
            hit = False
            for s in sets:
                reasons = [k for k in ("ids", "tasks", "scenes", "images") if ident[k] & s[k]]
                for k in reasons:
                    excluded[s["name"]][k] += 1
                hit |= bool(reasons)
            if not ident["images"]:
                excluded["no_media"]["rows"] += 1
                hit = True
            if hit:
                continue
            kept.append(dict(r, und_index=len(kept), und_dataset=d, und_line=line_no))
            n_kept += 1
        per_dataset[d] = {"rows": n_rows, "kept": n_kept, "excluded": n_rows - n_kept}
        sources[d]["rows"] = n_rows
    pool = out / "pool.jsonl"
    pool.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    (out / "dataset_info.json").write_text(json.dumps({POOL_DATASET: dict(attrs[0], file_name="pool.jsonl")}, indent=1))
    shutil.copyfile(sft_dir / "train.yaml", out / "lf_recipe.yaml")
    manifest = {"created_utc": _utc(), "sft_dir": str(sft_dir), "datasets": list(datasets), "sources": sources,
                "per_dataset": per_dataset, "rows": len(kept), "pool_sha256": _sha256(pool),
                "held_out": {s["name"]: {"file": str(p), "questions": s["questions"], "ids": len(s["ids"]),
                                         "scenes": len(s["scenes"]), "tasks": len(s["tasks"]), "images": len(s["images"])}
                             for s, (_, p) in zip(sets, held)},
                "excluded": {k: dict(v) for k, v in excluded.items()},
                "lf_recipe": {"file": str(sft_dir / "train.yaml"), "sha256": _sha256(sft_dir / "train.yaml")},
                "videos": sum(1 for r in kept if r.get("videos"))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return manifest


def load_pool(pool: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pool = Path(pool)
    manifest = json.loads((pool / "manifest.json").read_text())
    rows = _jsonl(pool / "pool.jsonl")
    if len(rows) != manifest["rows"] or any(r["und_index"] != i for i, r in enumerate(rows)):
        raise ValueError(f"{pool / 'pool.jsonl'} does not match its manifest ({len(rows)} rows, {manifest['rows']} listed)")
    return rows, manifest


def draws(seed: int, step: int, n_rows: int, batch_size: int) -> List[int]:
    """The rows of the understanding batch of optimizer step ``step``: ``batch_size`` distinct rows, uniform."""
    import numpy as np

    return [int(i) for i in np.random.default_rng([int(seed), int(step)]).choice(n_rows, size=batch_size, replace=False)]


def rank_slice(items: Sequence[Any], rank: int, world: int) -> List[Any]:
    per = len(items) // world
    if per * world != len(items):
        raise ValueError(f"{len(items)} samples do not split evenly over {world} ranks")
    return list(items[rank * per:(rank + 1) * per])


# ---------------------------------------------------------------------------------------------------- LLaMA-Factory

def rope_model(config: Any) -> Any:
    """What LLaMA-Factory's collator needs of a model: ``config`` and ``model.get_rope_index`` (the M-RoPE positions
    of a Qwen-VL model, a function of its config alone), without weights."""
    model_type = getattr(config, "model_type", None)
    if model_type == "qwen3_vl":
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel as cls
    elif model_type == "qwen2_5_vl":
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel as cls
    elif model_type == "qwen2_vl":
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel as cls
    else:
        raise ValueError(f"no M-RoPE index for model type {model_type!r}")
    holder = types.SimpleNamespace(config=config)
    return types.SimpleNamespace(config=config, model=types.SimpleNamespace(
        get_rope_index=functools.partial(cls.get_rope_index, holder)))


class LFPath:
    """The matrix SFT's LLaMA-Factory path for the pool's rows: a row -> its features (input_ids, labels, media) ->
    collated batches, exactly as the SFT's training read them."""

    def __init__(self, pool: Path, tokenizer_path: str, model_config: Any = None):
        import os

        import torch
        import yaml

        # as the SFT's launcher runs LLaMA-Factory (sco_interiorgs_v10_train.sh lf_train): llamafactory.hparams checks
        # its pinned versions on import (transformers <= 4.52.4; the Qwen3-VL runtime has 4.57), its own pin, not a
        # known incompatibility, and the RL job does not set the variable
        os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
        from model_matrix.sft.lf_qwen3vl import register

        register()  # the model line's Qwen3-VL template and plugin, before LLaMA-Factory reads anything
        from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_template_and_fix_tokenizer
        from llamafactory.data.converter import get_dataset_converter
        from llamafactory.data.parser import get_dataset_list
        from llamafactory.data.processor import SupervisedDatasetProcessor
        from llamafactory.hparams import DataArguments, ModelArguments
        from llamafactory.model import load_tokenizer

        pool = Path(pool)
        recipe = yaml.safe_load((pool / "lf_recipe.yaml").read_text())
        self.recipe = {k: recipe[k] for k in RECIPE_DATA_KEYS + RECIPE_MODEL_KEYS if k in recipe}
        self.model_args = ModelArguments(model_name_or_path=str(tokenizer_path),
                                         **{k: recipe[k] for k in RECIPE_MODEL_KEYS if k in recipe})
        self.data_args = DataArguments(dataset=POOL_DATASET, dataset_dir=str(pool),
                                       **{k: recipe[k] for k in RECIPE_DATA_KEYS if k in recipe})
        module = load_tokenizer(self.model_args)
        self.tokenizer, self.processor = module["tokenizer"], module["processor"]
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        attr = get_dataset_list([POOL_DATASET], str(pool))[0]
        self.converter = get_dataset_converter(attr.formatting, attr, self.data_args)
        self.dataset_processor = SupervisedDatasetProcessor(template=self.template, tokenizer=self.tokenizer,
                                                            processor=self.processor, data_args=self.data_args)
        if model_config is None:
            from transformers import AutoConfig

            model_config = AutoConfig.from_pretrained(str(tokenizer_path))
        # as LLaMA-Factory's SFT workflow builds it for training (pad to a multiple of 8, -100 under the prompt, the
        # recipe's bf16 compute type); the model is only asked for its M-RoPE index
        self.collator = SFTDataCollatorWith4DAttentionMask(
            template=self.template, model=rope_model(model_config), pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_INDEX, block_diag_attn=False, attn_implementation="eager",
            compute_dtype=torch.bfloat16 if recipe.get("bf16", True) else torch.float32, **module)

    def features(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for r in rows:
            example = self.converter({k: v for k, v in r.items() if k not in ROW_KEYS})
            got = self.dataset_processor.preprocess_dataset({k: [v] for k, v in example.items()})
            if len(got["input_ids"]) != 1:
                raise ValueError(f"LLaMA-Factory dropped pool row {r.get('und_index')} ({r.get('source_id')})")
            out.append({k: v[0] for k, v in got.items()})
        return out

    def collate(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return self.collator([dict(f) for f in features])  # the collator pops the media lists


def rl_micro_batch(batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    """A collated batch in the actor's micro-batch form, and the answer mask over its response positions.

    The response part is everything from the batch's first answer token on: _forward_micro_batch returns the
    log-probabilities of those positions' tokens (from the rolled input ids, so "responses" only gives the length),
    and the mask keeps the answer tokens (the collator's labels).  Position ids are the RL path's four rows: text
    positions 0..n-1 over the real tokens (1 under padding, as the agent loop writes them), then the collator's
    three M-RoPE rows."""
    import torch

    input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    bsz, length = input_ids.shape
    text = torch.ones((bsz, length), dtype=torch.long)
    for i in range(bsz):
        valid = attention_mask[i].bool()
        text[i, valid] = torch.arange(int(valid.sum()))
    mrope = batch["position_ids"]
    if mrope.dim() != 3 or mrope.shape[0] != 3:
        raise ValueError(f"expected the collator's (3, batch, length) M-RoPE positions, got {tuple(mrope.shape)}")
    answer = labels != IGNORE_INDEX
    if not bool(answer.any(dim=1).all()):
        raise ValueError("a sample without answer tokens")
    first = int(min(int(row.nonzero()[0]) for row in answer))
    media = {k: v for k, v in batch.items()
             if k not in ("input_ids", "attention_mask", "labels", "position_ids", "rope_deltas")}
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask.long(),
              "position_ids": torch.cat([text.unsqueeze(1), mrope.permute(1, 0, 2).long()], dim=1),
              "responses": input_ids[:, first:], "multi_modal_inputs": [media]}
    return inputs, answer[:, first:].float()


def expected_answer_text(row: Dict[str, Any]) -> str:
    """What the answer tokens of a row decode to under the qwen3_vl_nothink template: each assistant turn's text and
    the end of its turn."""
    return "".join(f"{m['content']}<|im_end|>\n" for m in row["messages"] if m["role"] == "assistant")


def answer_text(tokenizer: Any, inputs: Dict[str, Any], mask: Any, i: int = 0) -> str:
    ids = inputs["responses"][i][mask[i].bool()].tolist()
    return tokenizer.decode(ids, skip_special_tokens=False)


# ---------------------------------------------------------------------------------------------------- the mixer

def _dot(a: Any, b: Any) -> float:
    """The sum of a * b over this rank's shard (float32 products, float64 sum)."""
    import torch

    return float((a.detach().float() * b.detach().float()).sum(dtype=torch.float64))


def mix_gradients(params: Sequence[Any], und_backward: Callable[[], float], reduce: Callable[[List[Any]], List[float]]
                  ) -> Dict[str, float]:
    """Sets the policy gradient (in ``params``' .grad) aside, runs ``und_backward`` (which adds the understanding
    term's gradient into .grad and returns its loss sum on this rank), measures both terms (summed over the ranks by
    ``reduce``) and leaves their sum in .grad.  Returns the norms before clipping, their shares of the total norm
    and their cosine."""
    stash = {}
    for p in params:
        if p.grad is not None:
            stash[p] = p.grad
            p.grad = None
    loss_sum = und_backward()
    sq_u = sq_p = dot = 0.0
    for p in params:
        g_u, g_p = p.grad, stash.get(p)
        if g_u is not None:
            sq_u += _dot(g_u, g_u)
        if g_p is not None:
            sq_p += _dot(g_p, g_p)
        if g_u is not None and g_p is not None:
            dot += _dot(g_u, g_p)
    sq_u, sq_p, dot, loss_sum = reduce([sq_u, sq_p, dot, float(loss_sum)])
    for p, g in stash.items():
        if p.grad is None:
            p.grad = g
        else:
            p.grad.add_(g)
    n_u, n_p = math.sqrt(max(sq_u, 0.0)), math.sqrt(max(sq_p, 0.0))
    total = math.sqrt(max(sq_u + sq_p + 2 * dot, 0.0))
    return {"grad_norm_policy": n_p, "grad_norm_und": n_u, "grad_norm_total": total,
            "grad_share_policy": n_p / total if total > 0 else 0.0, "grad_share_und": n_u / total if total > 0 else 0.0,
            "grad_cos_policy_und": dot / (n_u * n_p) if n_u > 0 and n_p > 0 else 0.0, "loss_sum": loss_sum}


class UndMixer:
    """verl's DataParallelPPOActor calls ``before_optimizer_step`` right before each optimizer step (after the policy
    loss's backward passes) and ``end_of_update`` at the end of update_policy."""

    def __init__(self, cfg: Dict[str, Any], actor: Any, lf: Optional[LFPath] = None):
        import torch.distributed as dist

        self.cfg = {**DEFAULTS, **dict(cfg)}
        self.actor = actor
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        if int(self.cfg["batch_size"]) % self.world:
            raise ValueError(f"und_mix batch_size {self.cfg['batch_size']} does not split over {self.world} ranks")
        if getattr(actor, "scaler", None) is not None:
            raise ValueError("und_mix: fp16 loss scaling is not supported")
        if getattr(actor, "use_ulysses_sp", False):
            raise ValueError("und_mix: Ulysses sequence parallelism is not supported")
        module = actor.actor_module
        self.group = None
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

            if isinstance(module, FSDP):
                if module.sharding_strategy not in (ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP,
                                                    ShardingStrategy.NO_SHARD):
                    raise ValueError(f"und_mix: sharding strategy {module.sharding_strategy} is not supported")
                self.group = module.process_group
        except ImportError:
            pass
        self.rows, self.manifest = load_pool(Path(self.cfg["pool"]))
        if self.cfg.get("pool_sha256") and self.cfg["pool_sha256"] != self.manifest["pool_sha256"]:
            raise ValueError(f"und_mix: the pool {self.cfg['pool']} changed since the PPO config was written")
        self.lf = lf
        self.log_dir = Path(self.cfg["log_dir"]) if self.cfg.get("log_dir") else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.first_batch_written = False
        self.update_counts = Counter()

    # the pieces a test replaces
    def _lf(self) -> LFPath:
        if self.lf is None:
            self.lf = LFPath(Path(self.cfg["pool"]), self.cfg["tokenizer"])
        return self.lf

    def _reduce(self, values: List[Any]) -> List[float]:
        """The sums of ``values`` (numbers) over the ranks."""
        import torch
        import torch.distributed as dist

        t = torch.tensor([float(v) for v in values], dtype=torch.float64)
        if dist.is_initialized() and self.world > 1:
            from verl.utils.device import get_device_id

            t = t.to(get_device_id())
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=self.group)
        return [float(x) for x in t.cpu().tolist()]

    def _device(self) -> Any:
        """Where the actor's parameters are during the update (FSDP's flat shards; on the GPU then)."""
        import torch

        for p in self.actor.actor_module.parameters():
            return p.device
        return torch.device("cpu")

    def opt_step(self) -> int:
        """Optimizer steps taken so far (the optimizer's own count; it comes back with a resumed checkpoint)."""
        optimizer = self.actor.actor_optimizer
        for state in getattr(optimizer, "state", {}).values():
            if "step" in state:
                return int(float(state["step"]))
        return 0

    def _repair(self) -> None:
        try:
            from verl.utils.fsdp_utils import repair_unreduced_fsdp1_grad_shards_
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        except ImportError:
            return
        if isinstance(self.actor.actor_module, FSDP):  # what _optimizer_step does first: every shard reduced
            repair_unreduced_fsdp1_grad_shards_(self.actor.actor_module)

    def _log(self, record: Dict[str, Any]) -> None:
        if self.rank == 0 and self.log_dir is not None:
            with (self.log_dir / "steps.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")

    def before_optimizer_step(self) -> Dict[str, float]:
        import torch

        t0 = time.time()
        k = self.opt_step()
        params = [p for p in self.actor.actor_module.parameters() if p.requires_grad]
        self._repair()
        self.update_counts["opt_steps"] += 1
        if k % int(self.cfg["interval"]):
            norm = math.sqrt(max(self._reduce([sum(_dot(p.grad, p.grad) for p in params if p.grad is not None)])[0],
                                 0.0))
            self._log({"utc": _utc(), "opt_step": k, "inserted": False, "grad_norm_policy": norm,
                       "seconds": time.time() - t0})
            return {"actor/und_inserted": 0.0, "actor/und_samples": 0.0, "actor/und_tokens": 0.0,
                    "actor/und_opt_step": float(k), "actor/grad_norm_policy": norm, "actor/grad_norm_und": 0.0,
                    "actor/grad_share_policy": 1.0, "actor/grad_share_und": 0.0, "actor/grad_cos_policy_und": 0.0,
                    "actor/und_seconds": time.time() - t0}
        batch_rows = draws(int(self.cfg["seed"]), k, len(self.rows), int(self.cfg["batch_size"]))
        mine = rank_slice(batch_rows, self.rank, self.world)
        lf = self._lf()
        features = lf.features([self.rows[i] for i in mine])
        m = int(self.cfg["micro_batch_size"])
        prepared = [rl_micro_batch(lf.collate(features[i:i + m])) for i in range(0, len(features), m)]
        n_local = sum(float(mask.sum()) for _, mask in prepared)
        n_global = self._reduce([n_local])[0]
        # FSDP averages the ranks' gradients: weight x world / (all answer tokens) makes the sum weight x the
        # gradient of the token mean over the whole batch
        scale = float(self.cfg["weight"]) * self.world / max(n_global, 1.0)
        device = self._device()

        def und_backward() -> float:
            total = 0.0
            for inputs, mask in prepared:
                moved = {k2: (v.to(device) if torch.is_tensor(v) else v) for k2, v in inputs.items()}
                _, log_prob = self.actor._forward_micro_batch(moved, temperature=1.0, calculate_entropy=False)
                nll = -(log_prob * mask.to(log_prob.device)).sum()
                (nll * scale).backward()
                total += float(nll.detach())
            self._repair()  # the understanding gradient's shards reduced too, before they are measured and added
            return total

        stats = mix_gradients(params, und_backward, self._reduce)
        und_loss = stats["loss_sum"] / max(n_global, 1.0)
        seconds = time.time() - t0
        if not self.first_batch_written and self.log_dir is not None:  # the answer-mask check, every rank its own
            samples = []
            for j, (inputs, mask) in enumerate(prepared):
                for i in range(inputs["input_ids"].shape[0]):
                    row = self.rows[mine[j * m + i]]
                    got, want = answer_text(lf.tokenizer, inputs, mask, i), expected_answer_text(row)
                    samples.append({"und_index": row["und_index"], "source_id": row.get("source_id"),
                                    "dataset": row.get("und_dataset"), "answer_tokens": int(mask[i].sum()),
                                    "tokens": int(inputs["attention_mask"][i].sum()), "decoded": got,
                                    "expected": want, "match": got == want})
            (self.log_dir / f"first_batch_rank{self.rank}.json").write_text(json.dumps(
                {"opt_step": k, "samples": samples}, indent=1, ensure_ascii=False))
            self.first_batch_written = True
        self.update_counts["inserted"] += 1
        self._log({"utc": _utc(), "opt_step": k, "inserted": True, "draws": batch_rows,
                   "source_ids": [self.rows[i].get("source_id") for i in batch_rows], "und_loss": und_loss,
                   "und_tokens": n_global, "samples": len(batch_rows), "weight": float(self.cfg["weight"]),
                   **{key: stats[key] for key in ("grad_norm_policy", "grad_norm_und", "grad_norm_total",
                                                  "grad_share_policy", "grad_share_und", "grad_cos_policy_und")},
                   "seconds": seconds})
        return {"actor/und_inserted": 1.0, "actor/und_samples": float(len(batch_rows)), "actor/und_tokens": n_global,
                "actor/und_opt_step": float(k), "actor/und_loss": und_loss,
                **{f"actor/{key}": stats[key] for key in ("grad_norm_policy", "grad_norm_und", "grad_share_policy",
                                                           "grad_share_und", "grad_cos_policy_und")},
                "actor/grad_norm_total_preclip": stats["grad_norm_total"], "actor/und_seconds": seconds}

    def end_of_update(self) -> Dict[str, float]:
        """The update's count of understanding batches and optimizer steps (then reset)."""
        out = {"actor/und_batches": float(self.update_counts["inserted"]),
               "actor/und_update_opt_steps": float(self.update_counts["opt_steps"])}
        self.update_counts = Counter()
        return out


# ---------------------------------------------------------------------------------------------------- commands

def cmd_prepare(args: argparse.Namespace) -> None:
    held = []
    for spec in list(args.probe or []) + list(args.questions or []):
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--probe / --questions take NAME=FILE, not {spec!r}")
        held.append((name, Path(path)))
    manifest = prepare(Path(args.sft_dir), [d for d in args.datasets.split(",") if d], held, Path(args.out))
    print(json.dumps({k: manifest[k] for k in ("rows", "per_dataset", "excluded", "videos", "held_out")}, indent=1))


def cmd_check(args: argparse.Namespace) -> None:
    """The LLaMA-Factory path on pool rows (CPU): every video row and ``--rows`` others at random.  For each: the
    answer tokens decode to the assistant turns (and nothing else), the response part ends the input ids, the text
    positions count the real tokens, the image / video grids are there."""
    rows, manifest = load_pool(Path(args.pool))
    rng = random.Random(args.seed)
    video = [r for r in rows if r.get("videos")]
    other = [r for r in rows if not r.get("videos")]
    picked = video + (other if args.rows <= 0 else rng.sample(other, min(args.rows, len(other))))
    t0 = time.time()
    lf = LFPath(Path(args.pool), args.model)
    bad, tokens, answers = [], [], []
    for r in picked:
        feat = lf.features([r])[0]
        inputs, mask = rl_micro_batch(lf.collate([feat]))
        got, want = answer_text(lf.tokenizer, inputs, mask), expected_answer_text(r)
        n_label = sum(1 for x in feat["labels"] if x != IGNORE_INDEX)
        problems = []
        if got != want:
            problems.append("answer tokens")
        if int(mask.sum()) != n_label:
            problems.append("mask size")
        if not bool((inputs["responses"] == inputs["input_ids"][:, -inputs["responses"].shape[1]:]).all()):
            problems.append("responses")
        n_real = int(inputs["attention_mask"][0].sum())
        if inputs["position_ids"][0, 0, :n_real].tolist() != list(range(n_real)):
            problems.append("text positions")
        media = inputs["multi_modal_inputs"][0]
        if r.get("images") and "image_grid_thw" not in media:
            problems.append("image grid")
        if r.get("videos") and "video_grid_thw" not in media:
            problems.append("video grid")
        if problems:
            bad.append({"und_index": r["und_index"], "source_id": r.get("source_id"), "problems": problems,
                        "decoded": got, "expected": want})
        tokens.append(n_real)
        answers.append(int(mask.sum()))
    stat = lambda v: {"mean": round(statistics.mean(v), 1), "median": statistics.median(v), "max": max(v)}  # noqa: E731
    summary = {"utc": _utc(), "pool": str(args.pool), "pool_rows": manifest["rows"], "checked": len(picked),
               "video_rows_checked": len(video), "bad": len(bad), "bad_rows": bad[:50], "tokens": stat(tokens),
               "answer_tokens": stat(answers), "seconds": round(time.time() - t0, 1), "recipe": lf.recipe,
               "model": str(args.model)}
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "bad_rows"}, indent=1, ensure_ascii=False))
    if bad:
        raise SystemExit(f"{len(bad)} of {len(picked)} rows failed the check")


def counts(pool: Path, log: Path) -> Dict[str, Any]:
    """Draws per pool row from a run's steps.jsonl (a step logged twice, by a resumed run, counts once: the last)."""
    rows, _ = load_pool(pool)
    steps: Dict[int, Dict[str, Any]] = {}
    for rec in _jsonl(log):
        if rec.get("inserted"):
            steps[int(rec["opt_step"])] = rec
    per_row = Counter(i for rec in steps.values() for i in rec["draws"])
    hist = Counter(per_row.get(i, 0) for i in range(len(rows)))
    by_dataset = Counter()
    for i, n in per_row.items():
        by_dataset[rows[i]["und_dataset"]] += n
    return {"batches": len(steps), "draws": sum(per_row.values()), "rows": len(rows),
            "rows_drawn": len(per_row), "draws_per_row_histogram": {str(k): v for k, v in sorted(hist.items())},
            "draws_by_dataset": dict(by_dataset),
            "per_row": {rows[i]["source_id"]: n for i, n in sorted(per_row.items())}}


def cmd_counts(args: argparse.Namespace) -> None:
    out = counts(Path(args.pool), Path(args.log))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_row"}, indent=1))


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="the pool of understanding samples from an SFT folder")
    p.add_argument("--sft-dir", required=True, help="the start model's SFT folder (dataset_info.json, train.yaml)")
    p.add_argument("--datasets", required=True, help="comma-separated dataset names of dataset_info.json")
    p.add_argument("--probe", action="append", default=[], help="NAME=FILE: a held-out probe (sample_id, scene, ...)")
    p.add_argument("--questions", action="append", default=[], help="NAME=FILE: a held-out question set (id, task, ...)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_prepare)
    c = sub.add_parser("check", help="the LLaMA-Factory path on pool rows (CPU)")
    c.add_argument("--pool", required=True)
    c.add_argument("--model", required=True, help="the RL start model (its tokenizer, processor and config)")
    c.add_argument("--rows", type=int, default=1000, help="rows besides every video row (0: all)")
    c.add_argument("--seed", type=int, default=20261105)
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_check)
    n = sub.add_parser("counts", help="draws per pool row from a run's und_mix/steps.jsonl")
    n.add_argument("--pool", required=True)
    n.add_argument("--log", required=True)
    n.add_argument("--out", default=None)
    n.set_defaults(func=cmd_counts)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
