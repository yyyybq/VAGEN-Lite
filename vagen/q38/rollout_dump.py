"""q38 (the Qwen3.8-27B rerun line): the per-update rollout dump in the 8B format, written by the sampler.

Two files per training step ``s`` (DESIGN_ASYNC 2.5.5):

- ``<dir>/<s>.jsonl``: every candidate episode of the draw batch the step trains on (192 rows, no padding rows), one
  JSON object per line with the 8B dump's keys (vagen/ray_trainer.py ``_dump_generations`` / ``_log_rollout_data`` /
  ``rollout_dump_view``, RT:1156-1166, RT:1692-1724, RT:1752-1809 at qwen38 551c241e): input, output, gts, score,
  step, then the episode's reward_extra_info (traj_success, task_type, scene_id, task_id, initial_score, final_score,
  score_improvement, n_primitive_steps, n_turns, invalid_action under the V4 recipe), then kept_for_update; and
  after those the q38 keys uid, draw_batch, draw_index, min_global_steps, max_global_steps.  The training chain
  (data_gen/interiorgs_v10/train_chain.py family_success / block_stats / signal_check) reads it as it reads the 8B
  dumps.
- ``<dir>/filter/<s>.jsonl``: one row per candidate group: the selection decision (uid, task_id, task_type,
  draw_batch, draw_index, dispatch_step, kind, return_std, selected, adv_scale, n_episodes).  The training chain
  globs ``rollout_data/*.jsonl`` without recursion, so it does not see this directory.

Each file is written to ``<name>.tmp`` and moved into place with ``os.replace``: a reader sees a complete file or
none, also when the writer raises half way.

Decoding (``decode_rows``) keeps the special tokens (as RT:1772-1777) and replaces each image's token run with
``<image>`` (vagen/utils/image_token_utils.py, as RT:1782-1784 with trainer.replace_image_tokens_for_logging).  The
q38 TransferQueue rows are unpadded, so nothing is stripped from them; a padded (dense 2-D) column is unpadded as RT
does it (prompts padded on the left, responses on the right).
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# RT:1698-1704: the keys every 8B row starts with
BASE_KEYS = ("input", "output", "gts", "score", "step")
# the q38 keys after the 8B ones (DESIGN_ASYNC 2.5.5)
Q38_KEYS = ("uid", "draw_batch", "draw_index", "min_global_steps", "max_global_steps")
# the 8B dump's 16 keys under the V4 recipe (j2 F16), what T3 checks a dump against
KEYS_8B = BASE_KEYS + (
    "traj_success", "task_type", "scene_id", "task_id", "initial_score", "final_score", "score_improvement",
    "n_primitive_steps", "n_turns", "invalid_action", "kept_for_update",
)
GROUP_KEYS = ("uid", "task_id", "task_type", "draw_batch", "draw_index", "dispatch_step", "kind", "return_std",
              "selected", "adv_scale", "n_episodes", "step")


def _json_default(value: Any) -> Any:
    """RT:1717: numpy scalars as python numbers, anything else as its str."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _dumps(entry: Mapping[str, Any]) -> str:
    return json.dumps(entry, ensure_ascii=False, default=_json_default)


def episode_rows(
    step: int,
    inputs: Sequence[str],
    outputs: Sequence[str],
    scores: Sequence[float],
    extra_infos: Sequence[Mapping[str, Any]],
    kept: Sequence[bool],
    q38: Sequence[Mapping[str, Any]],
    gts: Sequence[Any] | None = None,
) -> list[dict]:
    """The rows of ``<s>.jsonl`` in the 8B key order (RT:1698-1712): input, output, gts, score, step, then the
    episode's reward_extra_info in its own order (RT:1706-1708 keeps every key with one value per row), then
    kept_for_update (RT:1165), then the q38 keys."""
    n = len(inputs)
    for name, col in (("outputs", outputs), ("scores", scores), ("extra_infos", extra_infos), ("kept", kept),
                      ("q38", q38)):
        if len(col) != n:
            raise ValueError(f"rollout dump: {name} has {len(col)} entries for {n} episodes")
    if gts is None:
        # RT:1786 reads reward_model.ground_truth; the gym agent dataset rows carry none, so the 8B dumps have null
        gts = [None] * n
    rows = []
    for i in range(n):
        row: dict[str, Any] = {"input": inputs[i], "output": outputs[i], "gts": gts[i], "score": float(scores[i]),
                               "step": int(step)}
        for k, v in extra_infos[i].items():
            if k not in row:
                row[k] = v
        row["kept_for_update"] = bool(kept[i])
        for k in Q38_KEYS:
            row[k] = q38[i].get(k)
        rows.append(row)
    return rows


def _write_atomic(path: str, lines: Iterable[str]) -> None:
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            text = "\n".join(lines)
            fh.write(text + "\n" if text else "")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_step_dump(rollout_data_dir: str, step: int, rows: Sequence[Mapping[str, Any]],
                    groups: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """Write ``<dir>/<step>.jsonl`` (``rows``, the episodes) and ``<dir>/filter/<step>.jsonl`` (``groups``, the
    decision table), each atomically.  Both lines are serialised before either file is touched, so a row that
    cannot be serialised leaves both files as they were.  Returns the two paths."""
    ep_lines = [_dumps(r) for r in rows]
    group_lines = [_dumps(g) for g in groups]
    filter_dir = os.path.join(rollout_data_dir, "filter")
    os.makedirs(filter_dir, exist_ok=True)
    ep_path = os.path.join(rollout_data_dir, f"{int(step)}.jsonl")
    group_path = os.path.join(filter_dir, f"{int(step)}.jsonl")
    _write_atomic(group_path, group_lines)
    _write_atomic(ep_path, ep_lines)
    return ep_path, group_path


def _ids(row: Any) -> list[int]:
    if hasattr(row, "tolist"):
        row = row.tolist()
    return [int(t) for t in row]


def unpad(ids: Sequence[int], pad_token_id: int | None, side: str) -> list[int]:
    """Strip ``pad_token_id`` from the left (prompts) or the right (responses) of a padded row."""
    ids = list(ids)
    if pad_token_id is None:
        return ids
    if side == "left":
        i = 0
        while i < len(ids) and ids[i] == pad_token_id:
            i += 1
        return ids[i:]
    j = len(ids)
    while j > 0 and ids[j - 1] == pad_token_id:
        j -= 1
    return ids[:j]


def decode_rows(tokenizer, rows: Sequence[Any], padded: bool = False, side: str = "right",
                replace_image_tokens: bool = True) -> list[str]:
    """Token rows as text with the special tokens kept (RT:1772-1777), each image's token run as ``<image>``.

    ``padded``: the rows come from a dense padded tensor, so the pad tokens on ``side`` are stripped first (RT
    strips the pad tokens of its left-padded prompts and right-padded responses); the q38 TransferQueue rows are
    unpadded and are decoded as they are.
    """
    pad = getattr(tokenizer, "pad_token_id", None)
    seqs = [unpad(_ids(r), pad, side) if padded else _ids(r) for r in rows]
    texts = tokenizer.batch_decode(seqs, skip_special_tokens=False)
    if replace_image_tokens:
        from vagen.utils.image_token_utils import replace_image_tokens_for_logging

        # processor=None: every known pattern is tried (Qwen2-VL family first), without the "unknown model" warning
        texts = replace_image_tokens_for_logging(list(texts), processor=None, tokenizer=tokenizer)
    return list(texts)


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
