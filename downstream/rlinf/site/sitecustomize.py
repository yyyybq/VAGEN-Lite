"""Startup hooks of the downstream line's RLinf runs (step-4 preparation, the user's decision 5 of 2026-10-22).

Python imports `sitecustomize` at the start of every process whose path holds this folder; downstream/rlinf/run_rlinf.sh
puts it on PYTHONPATH, so the driver, every Ray worker and every environment process load it (forked ones inherit
it).  With DOWNSTREAM_RL_HOOKS=1 it waits for these modules and patches them right after they are imported; RLinf's
and RoboCasa's files are not changed:

  robocasa.models.fixtures.counter       Counter.get_reset_regions gets the fixed region order (the same patch as
                                         the evaluation, downstream/eval/closed_loop.stable_counter_regions), so a
                                         seed gives the same scene in every RLinf environment process
  rlinf.hybrid_engines.fsdp.fsdp_model_manager
                                         FSDPModelManager.optimizer_step first measures the gradient norm of the
                                         value head's parameters and of all other (actor) parameters separately,
                                         before clipping, summed over the FSDP ranks; rank 0 appends them to
                                         $DOWNSTREAM_GRAD_LOG (jsonl)
  rlinf.models.embodiment.starvla.utils.vlm_preprocess
                                         get_train_image_size falls back to datasets.vla_data.obs_image_size when
                                         the checkpoint config has neither image_size nor default_image_resolution
                                         (the user's decision 2 of 2026-10-31: StarVLA checkpoints keep the training
                                         size in obs_image_size, so RLinf fed E3's policies the 256 x 256 frames
                                         unresized; reports/E3_CHECK_20261030.md).  DOWNSTREAM_RL_IMAGE_SIZE_FIX=0
                                         leaves it out (the control run of the check).

Every patch writes one line to $DOWNSTREAM_RL_HOOK_LOG (pid, module, time), so a run shows that it took effect.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys
import time

_DOWNSTREAM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _note(module: str, **extra) -> None:
    path = os.environ.get("DOWNSTREAM_RL_HOOK_LOG")
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write(json.dumps(dict(pid=os.getpid(), module=module, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                      time.gmtime()), **extra)) + "\n")
    except OSError:
        pass


def patch_counter(module) -> None:
    sys.path.insert(0, os.path.join(_DOWNSTREAM, "eval"))
    from closed_loop import with_ordered_set

    counter = module.Counter
    if not getattr(counter.get_reset_regions, "downstream_ordered_set", False):
        counter.get_reset_regions = with_ordered_set(counter.get_reset_regions)
    _note(module.__name__, patch="Counter.get_reset_regions ordered set")


def group_grad_norms(named_parameters, is_value=lambda name: "value_head" in name, all_reduce=None) -> dict:
    """{"actor": norm, "value_head": norm, "total": norm} of the gradients present on this rank; with all_reduce
    (a function summing a float32 tensor over the ranks in place) the squares are summed over the ranks first, which
    gives the norms of FSDP-sharded gradients."""
    import torch

    parts, device = ([], []), None
    for name, p in named_parameters:
        if p.grad is None or p.grad.numel() == 0:
            continue
        g = p.grad.detach()
        device = device or g.device
        parts[1 if is_value(name) else 0].append(g.float().pow(2).sum())
    if device is None:  # no gradient on this rank: still take part in the reduction, on the rank's own device
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    sq = torch.stack([torch.stack(x).sum().to(device) if x else torch.zeros((), device=device) for x in parts])
    if all_reduce is not None:
        all_reduce(sq)
    actor, value = (float(x) ** 0.5 for x in sq.cpu())
    return {"actor": actor, "value_head": value, "total": (actor ** 2 + value ** 2) ** 0.5}


def patch_fsdp_manager(module) -> None:
    manager = module.FSDPModelManager
    original = manager.optimizer_step
    if getattr(original, "downstream_grad_groups", False):
        return

    def optimizer_step(self, *args, **kwargs):
        try:
            import torch.distributed as dist

            reduce = None
            if dist.is_available() and dist.is_initialized():
                reduce = lambda t: dist.all_reduce(t, op=dist.ReduceOp.SUM)  # noqa: E731
            norms = group_grad_norms(self.model.named_parameters(), all_reduce=reduce)
            rank0 = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
            path = os.environ.get("DOWNSTREAM_GRAD_LOG")
            if rank0 and path:
                with open(path, "a") as f:
                    f.write(json.dumps(dict(norms, optimizer_step=int(getattr(self, "optimizer_steps", 0)) + 1,
                                            time=time.time())) + "\n")
        except Exception as exc:  # the measurement must never stop training
            _note(module.__name__, error=f"{type(exc).__name__}: {exc}"[:300])
        return original(self, *args, **kwargs)

    optimizer_step.downstream_grad_groups = True
    manager.optimizer_step = optimizer_step
    _note(module.__name__, patch="FSDPModelManager.optimizer_step grad norms by group")


def obs_image_size(starvla_model):
    """The (height, width) a StarVLA checkpoint was trained at, from its config's datasets.vla_data.obs_image_size
    (StarVLA's own predict_action reads it there), or None."""
    cfg = getattr(starvla_model, "config", None)
    vla_data = getattr(getattr(cfg, "datasets", None), "vla_data", None) if cfg is not None else None
    size = getattr(vla_data, "obs_image_size", None) if vla_data is not None else None
    if size is None:
        return None
    try:
        values = [int(v) for v in size]
    except (TypeError, ValueError):
        return None
    return (values[0], values[1]) if len(values) == 2 else None


def patch_vlm_preprocess(module) -> None:
    original = module.get_train_image_size
    if getattr(original, "downstream_obs_image_size", False):
        return

    def get_train_image_size(starvla_model):
        size = original(starvla_model)
        return size if size else obs_image_size(starvla_model)

    get_train_image_size.downstream_obs_image_size = True
    module.get_train_image_size = get_train_image_size
    _note(module.__name__, patch="get_train_image_size falls back to datasets.vla_data.obs_image_size")


HOOKS = {"robocasa.models.fixtures.counter": patch_counter,
         "rlinf.hybrid_engines.fsdp.fsdp_model_manager": patch_fsdp_manager}
if os.environ.get("DOWNSTREAM_RL_IMAGE_SIZE_FIX", "1") != "0":
    HOOKS["rlinf.models.embodiment.starvla.utils.vlm_preprocess"] = patch_vlm_preprocess


class PostImportHook(importlib.abc.MetaPathFinder):
    """Runs hooks[name](module) right after module `name` is executed (first import only)."""

    def __init__(self, hooks: dict):
        self.hooks = dict(hooks)

    def find_spec(self, name, path=None, target=None):
        if name not in self.hooks:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        hook, loader = self.hooks[name], spec.loader
        original = loader.exec_module

        def exec_module(module):
            original(module)
            try:
                hook(module)
            except Exception as exc:
                _note(name, error=f"{type(exc).__name__}: {exc}"[:300])

        loader.exec_module = exec_module
        return spec


def install(hooks=None) -> PostImportHook:
    finder = PostImportHook(HOOKS if hooks is None else hooks)
    for name, hook in finder.hooks.items():
        if name in sys.modules:  # imported before this ran
            hook(sys.modules[name])
    sys.meta_path.insert(0, finder)
    return finder


if os.environ.get("DOWNSTREAM_RL_HOOKS") == "1":
    install()
