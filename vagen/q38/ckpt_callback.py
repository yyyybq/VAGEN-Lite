"""q38 (the Qwen3.8-27B rerun line): the checkpoint callback of the V1 trainers (DESIGN_ASYNC 2.1, 2.5.5).

verl's V1 trainer calls ``on_save`` on the driver after each checkpoint (``trainer.checkpoint_callback_class``,
verl/verl/trainer/ppo/checkpoint_callback.py:33-82; call sites TB:1075-1090, after latest_checkpointed_iteration.txt
is written).  ``VagenCheckpointCallback`` then

1. writes ``checkpoint_metadata.json`` and the ``COMPLETE`` marker into the new global_step_<s> folder, atomically and
   only when the actor checkpoint (and its HF export, when save_contents has hf_model) is there, as the 8B trainer
   does before it publishes a checkpoint (RT:2402-2438);
2. keeps every ``trainer.ckpt_keep_every``-th COMPLETE checkpoint and the latest ``trainer.ckpt_keep_latest`` of the
   others, and deletes the other COMPLETE folders (``retained_steps``, the 8B rule of 2026-10-14, IP:58-63);
3. with ``trainer.ckpt_prune_to_hf``, cuts every kept checkpoint but the newest down to ``actor/huggingface`` and
   ``COMPLETE`` (``prune_to_hf``, IP:37-55): the optimizer state, data.pt and the TransferQueue snapshot are needed
   only to resume from the newest one.

Folders without COMPLETE (a save that did not finish) are never touched.  The callback needs
``trainer.max_actor_ckpt_to_keep`` / ``max_critic_ckpt_to_keep`` null (the workers would otherwise delete the
milestones' HF exports themselves, TB:1021-1050) and synchronous saves (with ``async_save`` the hook fires before the
shards are durable, checkpoint_callback.py:49-62).  A pruned folder keeps its ``actor/`` directory, so V1's own clean-up
of step folders without actor/ (TB:996-1003) leaves it alone.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Iterable, List, Set

from omegaconf import OmegaConf

from verl.trainer.ppo.checkpoint_callback import CheckpointCallback

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

COMPLETE = "COMPLETE"
METADATA = "checkpoint_metadata.json"


# Copied from vagen/interiorgs_v10_ppo.py:37-55 (IP, commit cee1514a; IP imports the 8B trainer at the top and cannot
# be imported in the q38 environment).  tests/test_trainer.py checks that the two copies stay identical.
def prune_to_hf(folder: Path) -> List[str]:
    """Delete everything in a checkpoint folder except ``actor/huggingface`` and the COMPLETE marker; returns the
    removed paths (relative).  A folder without ``actor/huggingface`` is left alone."""
    folder = Path(folder)
    keep = folder / "actor" / "huggingface"
    if not keep.is_dir():
        return []
    removed = []
    for p in sorted(folder.iterdir()):
        if p.name == "COMPLETE":
            continue
        if p.name == "actor":
            for q in sorted(p.iterdir()):
                if q.name != "huggingface":
                    removed.append(str(q.relative_to(folder)))
                    shutil.rmtree(q, ignore_errors=True) if q.is_dir() else q.unlink(missing_ok=True)
            continue
        removed.append(str(p.relative_to(folder)))
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    return removed


# Copied from vagen/interiorgs_v10_ppo.py:58-63 (IP, commit cee1514a).
def retained_steps(steps: Iterable[int], keep_every: int, keep_latest: int) -> Set[int]:
    """Checkpoints to keep: every ``keep_every``-th step and the latest ``keep_latest`` of the others."""
    steps = sorted(set(int(s) for s in steps))
    milestones = {s for s in steps if keep_every > 0 and s % keep_every == 0}
    others = [s for s in steps if s not in milestones]
    return milestones | set(others[-keep_latest:] if keep_latest > 0 else [])


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def complete_steps(root: Path) -> dict[int, Path]:
    """step -> folder of the global_step_<s> folders under ``root`` that carry the COMPLETE marker."""
    out = {}
    for folder in Path(root).glob("global_step_*"):
        try:
            step = int(folder.name.split("global_step_")[-1])
        except ValueError:
            continue
        if folder.is_dir() and (folder / COMPLETE).exists():
            out[step] = folder
    return out


class VagenCheckpointCallback(CheckpointCallback):
    """COMPLETE marker, milestone retention and pruning to HF after every V1 checkpoint (module docstring)."""

    def __init__(self, config=None):
        super().__init__(config=config)
        trainer = config.trainer
        self.keep_every = int(trainer.get("ckpt_keep_every", 0) or 0)
        keep_latest = trainer.get("ckpt_keep_latest", None)
        self.keep_latest = None if keep_latest is None else int(keep_latest)
        self.prune = bool(trainer.get("ckpt_prune_to_hf", False))
        for key in ("max_actor_ckpt_to_keep", "max_critic_ckpt_to_keep"):
            if trainer.get(key, None) is not None:
                raise ValueError(
                    f"q38 checkpoint callback: trainer.{key} must be null (got {trainer.get(key)!r}): the workers would "
                    "delete the kept checkpoints' HF exports themselves (DESIGN 2.5.5)"
                )
        if trainer.get("remove_previous_ckpt_in_save", False):
            raise ValueError("q38 checkpoint callback: trainer.remove_previous_ckpt_in_save must be false")
        actor_ckpt = OmegaConf.select(config, "actor_rollout_ref.actor.checkpoint", default=None) or {}
        if actor_ckpt.get("async_save", False):
            raise ValueError("q38 checkpoint callback: actor checkpoint.async_save must be false")
        self.save_contents = [str(c) for c in (actor_ckpt.get("save_contents", None) or [])]

    def on_save(self, trainer, global_step: int, checkpoint_dir: str, async_save: bool = False, **kwargs) -> None:
        if async_save:
            raise RuntimeError("q38 checkpoint callback: asynchronous saves are not supported (no COMPLETE marker)")
        folder = Path(checkpoint_dir)
        step = int(global_step)
        self.mark_complete(folder, step, use_critic=bool(getattr(trainer, "use_critic", False)))
        self.rotate(folder, step)

    # ---- 1. the marker ---------------------------------------------------------------------------------------------

    def mark_complete(self, folder: Path, step: int, use_critic: bool = False) -> dict:
        actor = folder / "actor"
        metadata = {
            "global_step": step,
            "experiment_name": str(OmegaConf.select(self.config, "trainer.experiment_name", default="")),
            "actor_path": str(actor),
            "has_actor": actor.is_dir(),
            "has_hf_model": (actor / "huggingface").is_dir(),
            "has_critic": (not use_critic) or (folder / "critic").is_dir(),  # str(Role.Critic), TB:1033
            "has_dataloader_state": (folder / "data.pt").is_file(),
            "has_transfer_queue": (folder / "transfer_queue").exists(),
            "created_unix_time": time.time(),
        }
        needs_hf = "hf_model" in self.save_contents
        if not metadata["has_actor"] or not metadata["has_critic"] or (needs_hf and not metadata["has_hf_model"]):
            raise RuntimeError(f"q38 checkpoint callback: refusing COMPLETE for an incomplete checkpoint: {metadata}")
        _write_atomic(folder / METADATA, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        _write_atomic(folder / COMPLETE, str(step))  # the 8B marker's content (RT:2434-2438)
        return metadata

    # ---- 2.-3. retention and pruning ----------------------------------------------------------------------------------

    def rotate(self, current: Path, step: int) -> dict:
        """Delete the COMPLETE folders retained_steps does not keep, then prune the kept ones but the newest.  Returns
        {"kept": [...], "removed": [...], "pruned": {step: [paths]}} (also logged)."""
        root = current.parent
        complete = complete_steps(root)
        report = {"kept": sorted(complete), "removed": [], "pruned": {}}
        if self.keep_every <= 0 and self.keep_latest is None:
            keep = set(complete)  # no retention configured: keep every checkpoint
        else:
            keep_latest = len(complete) if self.keep_latest is None else self.keep_latest
            keep = retained_steps(complete, self.keep_every, keep_latest)
        keep.add(step)
        cur = current.resolve()
        for s, folder in sorted(complete.items()):
            if s not in keep and folder.resolve() != cur:
                logger.info(f"[ckpt-rotate] removing checkpoint {folder} (keeping {sorted(keep)})")
                shutil.rmtree(folder, ignore_errors=True)
                report["removed"].append(s)
        if self.prune:
            newest = max([s for s in complete if s in keep] + [step])
            for s, folder in sorted(complete.items()):
                if s in keep and s != newest and folder.resolve() != cur:
                    removed = prune_to_hf(folder)
                    if removed:
                        logger.info(f"[ckpt-rotate] {folder}: kept actor/huggingface, removed {removed}")
                        report["pruned"][s] = removed
        report["kept"] = sorted(s for s in complete if s in keep)
        return report
