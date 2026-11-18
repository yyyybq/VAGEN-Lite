"""Shared helpers of the q38 sampler tests (test_sampler.py, test_candidate_metrics.py, test_rollout_dump.py).

- the 8B golden runs (``golden/<run>.jsonl.gz``, written by scripts/q38/extract_golden.py; the format is in its
  docstring) and their episodes as the sampler's selection sees them;
- ``FakeTQ``: an in-memory TransferQueue with the calls the replay buffer and the sampler make (kv_list, kv_put,
  kv_batch_put with merged tags, kv_batch_get, kv_clear), installed over the ``transfer_queue`` module;
- the rollout's writes, as the V1 agent loop worker makes them (verl/verl/trainer/ppo/v1/agent_loop_tq.py
  TQW:110-153, TQW:195-242): prompt keys with the q38 draw tags, ``{uid}_{session}_0`` trajectory keys;
- ``FakeTokenizer`` and ``work_dir`` (scratch directories under Q38_TEST_TMP, default
  /mnt/umm/users/yinbaiqiao/q38/tmp_tests, else the system temp directory).
"""

from __future__ import annotations

import copy
import gzip
import json
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from typing import Any, Sequence

import torch

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
GOLDEN_RUNS = ("qw3v4s3", "qw3s0s3", "qw3dcks3")
DEFAULT_TMP = "/mnt/umm/users/yinbaiqiao/q38/tmp_tests"


# ---- scratch directories ---------------------------------------------------------------------------------------


@contextmanager
def work_dir(name: str):
    base = os.environ.get("Q38_TEST_TMP", DEFAULT_TMP)
    try:
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, f"{name}-{uuid.uuid4().hex[:8]}")
        os.makedirs(path)
    except OSError:
        path = tempfile.mkdtemp(prefix=f"{name}-")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ---- the golden runs -------------------------------------------------------------------------------------------


def golden_path(run: str) -> str:
    return os.path.join(GOLDEN_DIR, f"{run}.jsonl.gz")


def load_golden(run: str) -> tuple[dict, list[dict]]:
    with gzip.open(golden_path(run), "rt", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    header, steps = lines[0], lines[1:]
    assert header["kind"] == "header" and all(s["kind"] == "step" for s in steps)
    return header, steps


def golden_episodes(step: dict):
    """(uids, returns, extra columns, draw order) of one golden update, rows in the 8B dump order.  The group of
    an episode is named after the draw position of its task (``g<position>``)."""
    tasks = step["tasks"]
    uids = [f"g{g}" for g in step["g"]]
    extra = {
        "traj_success": [float(x) for x in step["traj_success"]],
        "task_type": [tasks[g][1] for g in step["g"]],
        "task_id": [tasks[g][0] for g in step["g"]],
        "final_score": step["final_score"],
        "score_improvement": step["score_improvement"],
        "invalid_action": step["invalid_action"],
    }
    draw_order = {f"g{j}": j for j in range(len(tasks))}
    return uids, list(step["score"]), extra, draw_order


def golden_filter_cfg(header: dict, steps: Sequence[dict], directory: str) -> dict:
    """The run's filter section; a ``rows`` task file is replaced by one written from the golden d_star values (the
    filter reads only task_id and difficulty.d_star from it)."""
    cfg = copy.deepcopy(header["filter"])
    kwargs = cfg.get("filter_kwargs") or {}
    if kwargs.get("rows"):
        path = os.path.join(directory, f"{header['run']}_rows.jsonl")
        seen = {}
        for s in steps:
            for task_id, _family, d_star in s["tasks"]:
                if d_star is not None:
                    seen[task_id] = d_star
        with open(path, "w", encoding="utf-8") as fh:
            for task_id, d_star in seen.items():
                fh.write(json.dumps({"task_id": task_id, "difficulty": {"d_star": d_star}}) + "\n")
        kwargs["rows"] = path
        cfg["filter_kwargs"] = kwargs
    return cfg


# ---- an in-memory TransferQueue ----------------------------------------------------------------------------------


class FakeTQ:
    """kv store {partition: {key: {"tag": dict, "fields": dict}}}; tags merge on every put (as TransferQueue's
    metadata update), fields merge too."""

    def __init__(self, as_tensordict: bool = False):
        self.store: dict[str, dict[str, dict]] = {}
        self.calls: list[tuple] = []
        # kv_batch_get returns the columns as the rollout packs them for TransferQueue (TQW:237-242:
        # list_of_dict_to_tensordict: jagged tensors, NonTensorStack), instead of plain lists
        self.as_tensordict = as_tensordict

    def _entry(self, partition_id: str, key: str) -> dict:
        return self.store.setdefault(partition_id, {}).setdefault(key, {"tag": {}, "fields": {}})

    def kv_list(self, partition_id: str | None = None):
        self.calls.append(("kv_list", partition_id))
        parts = [partition_id] if partition_id is not None else list(self.store)
        return {p: {k: copy.deepcopy(v["tag"]) for k, v in self.store.get(p, {}).items()} for p in parts}

    def kv_put(self, key: str, partition_id: str, fields: dict | None = None, tag: dict | None = None):
        e = self._entry(partition_id, key)
        if tag:
            e["tag"].update(copy.deepcopy(tag))
        if fields:
            e["fields"].update(fields)

    def kv_batch_put(self, keys, partition_id, fields=None, tags=None):
        for i, k in enumerate(keys):
            self.kv_put(k, partition_id, fields=None if fields is None else fields[i],
                        tag=None if tags is None else tags[i])

    def kv_batch_get(self, keys, partition_id, select_fields=None):
        self.calls.append(("kv_batch_get", tuple(keys), tuple(select_fields or ())))
        part = self.store.get(partition_id, {})
        missing = [k for k in keys if k not in part]
        if missing:
            raise ValueError(f"keys not found: {missing[:3]}")
        names = list(select_fields) if select_fields else sorted({f for k in keys for f in part[k]["fields"]})
        if self.as_tensordict:
            from verl.utils.tensordict_utils import list_of_dict_to_tensordict

            return list_of_dict_to_tensordict([{n: part[k]["fields"][n] for n in names} for k in keys])
        out = {}
        for name in names:
            col = []
            for k in keys:
                if name not in part[k]["fields"]:
                    raise ValueError(f"field {name} missing for key {k}")
                col.append(part[k]["fields"][name])
            out[name] = col
        return out

    def kv_clear(self, keys, partition_id):
        self.calls.append(("kv_clear", tuple(keys)))
        part = self.store.get(partition_id, {})
        for k in [keys] if isinstance(keys, str) else keys:
            part.pop(k, None)

    def install(self, monkeypatch):
        import transfer_queue

        for name in ("kv_list", "kv_put", "kv_batch_put", "kv_batch_get", "kv_clear"):
            monkeypatch.setattr(transfer_queue, name, getattr(self, name))
        return self

    # views for the tests
    def keys(self, partition_id: str = "train") -> set[str]:
        return set(self.store.get(partition_id, {}))

    def prompt_keys(self, partition_id: str = "train") -> set[str]:
        return {k for k, v in self.store.get(partition_id, {}).items() if v["tag"].get("is_prompt")}


# ---- the rollout's writes ----------------------------------------------------------------------------------------


def new_uid() -> str:
    return uuid.uuid4().hex  # no "_": trajectory keys are {uid}_{session}_{index}


def dispatch(fake: FakeTQ, draw_batch: int, global_steps: int, n_groups: int, order: Sequence[int] | None = None,
             partition_id: str = "train") -> list[str]:
    """Register ``n_groups`` prompts of one draw batch as the trainer does (TB:1566-1586 with the q38 draw tags);
    ``order``: the order (of draw positions) the keys are put in, to show the sampler reads the tags."""
    uids = [new_uid() for _ in range(n_groups)]
    for j in (order if order is not None else range(n_groups)):
        fake.kv_put(uids[j], partition_id, fields={"index": j},
                    tag={"is_prompt": True, "status": "pending", "global_steps": global_steps,
                         "draw_batch": draw_batch, "draw_index": j})
    return uids


def reward_info(i: int, success: bool, task_id: str, task_type: str = "fov_inclusion") -> dict:
    """The 10 per-episode values of the 8B agent loop's reward_extra_info (GL:339-352)."""
    return {
        "traj_success": float(success), "task_type": task_type, "scene_id": "0001_000001", "task_id": task_id,
        "initial_score": 0.25, "final_score": 1.0 if success else 0.25 + 0.01 * i,
        "score_improvement": 0.75 if success else 0.01 * i, "n_primitive_steps": 3 + i, "n_turns": 1 + i % 3,
        "invalid_action": 0.0 if i % 4 else 0.25,
    }


def finish_group(fake: FakeTQ, uid: str, returns: Sequence[float], successes: Sequence[bool], global_steps: int,
                 versions: tuple[Any, Any] = (0, 0), task_id: str | None = None, task_type: str = "fov_inclusion",
                 status: str = "finished", stats: dict | None = None, prompt_ids=None, response_ids=None,
                 partition_id: str = "train") -> list[str]:
    """Write a group's trajectories as TQW:195-242 does and mark the prompt (TQW:148)."""
    task_id = task_id or f"task-{uid[:6]}"
    keys = []
    for i, (ret, ok) in enumerate(zip(returns, successes, strict=True)):
        resp = list(response_ids) if response_ids is not None else [5, 6, 7, 2]
        rm = torch.zeros(len(resp), dtype=torch.float32)
        rm[-1] = float(ret)  # AL:145-150
        lo, hi = versions
        extra = {"min_global_steps": lo, "max_global_steps": hi,
                 "reward_extra_info": reward_info(i, bool(ok), task_id, task_type)}
        if stats is not None:
            extra["vagen_stats"] = copy.deepcopy(stats)
        key = f"{uid}_{i}_0"
        fake.kv_put(key, partition_id,
                    fields={"rm_scores": rm, "extra_fields": extra,
                            "prompts": torch.tensor(list(prompt_ids) if prompt_ids is not None else [1, 3, 4]),
                            "responses": torch.tensor(resp), "response_mask": torch.ones(len(resp), dtype=torch.int64)},
                    tag={"status": "success", "global_steps": global_steps, "min_global_steps": lo,
                         "max_global_steps": hi})
        keys.append(key)
    fake.kv_put(uid, partition_id, tag={"status": status})
    return keys


def set_status(fake: FakeTQ, uid: str, status: str, partition_id: str = "train") -> None:
    fake.kv_put(uid, partition_id, tag={"status": status})


def make_sampler(mode: str = "colocate_async", filter_cfg: dict | None = None, candidates: int = 24, n: int = 8,
                 **kw):
    from vagen.q38.sampler import VagenStepBatchSampler

    sampler_kwargs = {
        "filter": filter_cfg if filter_cfg is not None else {"enable": True, "name": "mixed_outcome",
                                                              "filter_kwargs": {"target_groups": 12}},
        "adv_estimator": "grpo", "rollout_n": n, "candidates_per_step": candidates, "rollout_data_dir": None,
        "model_path": None, "replace_image_tokens_for_logging": True, "failure_policy": "raise",
        "stuck_warn": {"min_s": 300, "factor": 3, "repeat_s": 300}, "strict_version_check": True,
        "poll_interval": 0.0,
    }
    sampler_kwargs.update(kw)
    return VagenStepBatchSampler(
        trainer_mode=mode, trainer_config={}, max_off_policy_threshold=2, max_off_policy_strategy="wait",
        sampler_kwargs=sampler_kwargs, refill_fn=lambda count: count,
    )


# ---- a tokenizer ---------------------------------------------------------------------------------------------------


class FakeTokenizer:
    """ids -> fixed strings; the Qwen2-VL image tokens are 10 (vision_start), 11 (image_pad), 12 (vision_end)."""

    VOCAB = {0: "<pad>", 1: "<|im_start|>", 2: "<|im_end|>", 3: "user\n", 4: "hi", 5: "<action>", 6: "move_forward|",
             7: "</action>", 8: "<|endoftext|>", 10: "<|vision_start|>", 11: "<|image_pad|>", 12: "<|vision_end|>",
             13: "\n"}
    pad_token_id = 0

    def batch_decode(self, seqs, skip_special_tokens=False):
        assert skip_special_tokens is False
        return ["".join(self.VOCAB[int(t)] for t in s) for s in seqs]
