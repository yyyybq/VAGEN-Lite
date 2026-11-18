"""AgenticDatasetV1: the 8B AgenticDataset (vagen/gym_agent_dataset.py) for verl's V1 trainers (DESIGN_ASYNC 2.1).

Selected by ``data.custom_cls: {path: pkg://vagen.q38.dataset, name: AgenticDatasetV1}``.  The rows are the parent's
(one per env instance: env_name, seed, config, max_turns, response_length_per_turn, data_source, agent_name) with three
changes the V1 code path needs (interface note i2, tables e/f):

- the three dummy tensors ``input_ids`` / ``attention_mask`` / ``position_ids`` (gym_agent_dataset.py:220-222) are
  dropped: AgentLoopWorkerTQ copies every dataset field into each stored trajectory (``field.update(kwargs)``,
  verl/verl/trainer/ppo/v1/agent_loop_tq.py:210), where a one-element dummy ``attention_mask`` would sit next to the
  real trajectory tensors, and the async trainers persist the prompt's tensor fields with the prompt key;
- ``raw_prompt: []``: the V1 trainer counts a fetched chunk's prompts with ``len(batch_dict["raw_prompt"])`` to give
  each one a uid (verl/verl/trainer/ppo/v1/trainer_base.py:1536); the agent loop builds its own prompt from the env,
  so the list is empty (after collate_fn it is an object array of one empty list per row, so its length is the row
  count);
- ``index: idx``: AgentLoopWorkerTQ.generate_sequences reads ``batch["index"]`` (agent_loop_tq.py:86).

Every field stays a per-row Python value, so collate_fn makes it an object array and ``tu.get_tensordict`` a
NonTensorStack.  The async trainers persist exactly the non-NonTensorData fields of a dispatched chunk with its prompt
key (trainer_base.py:1566-1577) and re-issue them after a resume (trainer_base.py:971-987), so no row field may be a
batch-level NonTensorData.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Sequence

from omegaconf import ListConfig

from vagen.gym_agent_dataset import AgenticDataset

# the parent's placeholder tensors (gym_agent_dataset.py:220-222), never needed by the V1 agent loop
DUMMY_TENSOR_KEYS = ("input_ids", "attention_mask", "position_ids")


def single_data_file(data_files: Any) -> str:
    """The one env-spec yaml of ``data.train_files`` / ``data.val_files``: a path, or a list holding one path.

    The parent loads exactly one yaml (``load_envspecs(data_files)``); a list of several would be read as a path and
    fail later with an unrelated OmegaConf error, so it is refused here.
    """
    if isinstance(data_files, str):
        return data_files
    if isinstance(data_files, (list, tuple, ListConfig)):
        items = list(data_files)
        if len(items) == 1 and isinstance(items[0], str):
            return items[0]
        raise ValueError(f"AgenticDatasetV1 reads one env-spec yaml, got {len(items)} data files: {items}")
    raise TypeError(f"AgenticDatasetV1: data_files must be a path or a one-element list, got {type(data_files)}")


class AgenticDatasetV1(AgenticDataset):
    """The 8B AgenticDataset with V1 rows (no dummy tensors, ``raw_prompt: []``, ``index``)."""

    def __init__(
        self,
        data_files: str | Sequence[str],
        config: Dict[str, Any],
        tokenizer=None,
        processor=None,
        max_samples: int = -1,
        **kwargs,
    ):
        # verl's create_rl_dataset passes tokenizer / processor / max_samples (verl/verl/trainer/ppo/utils.py:129-135);
        # the parent takes only data_files and config.  max_samples is refused rather than silently ignored: the 8B
        # dataset never subsampled, and V1 builds the val dataset with val_max_samples (default -1) even when
        # validation is off (trainer_base.py:745-752).
        if max_samples is not None and int(max_samples) > 0:
            raise ValueError(f"AgenticDatasetV1 does not subsample (max_samples={max_samples}); use the env yaml")
        super().__init__(data_files=single_data_file(data_files), config=config, **kwargs)

    def __getitem__(self, idx):
        row = dict(self.items[idx])
        for key in DUMMY_TENSOR_KEYS:
            row.pop(key, None)
        # the parent shares one config dict between all rows of a spec; each row gets its own copy so nothing that
        # edits a row's config in place (in this process) changes the other rows
        row["config"] = copy.deepcopy(row["config"])
        row["raw_prompt"] = []
        row["index"] = int(idx)
        return row
