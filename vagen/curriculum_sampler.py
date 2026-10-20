"""Round 12 batch (InteriorGS v10, the user's "easy first" experiment): a training-task order where the first
updates draw only easy tasks (shortest path d* <= max_dstar), the rest all tasks.

PPO config (training_prep rl-config --curriculum-easy-updates N --curriculum-max-dstar D)::

    data.sampler: {class_path: pkg://vagen.curriculum_sampler, class_name: DifficultyCurriculumSampler,
                   easy_updates: N, max_dstar: D}
    data.dataloader_num_workers: 0          # verl requires it with a custom sampler

Each dataset item is one task row (AgenticDataset: item "seed" = the row index in the env config's jsonl_path), so
the sampler reads every row's d* from that file.  Epoch 0: the first easy_updates x train_batch_size indices are
easy tasks (a random draw without repeats), then every other index in random order; later epochs: all indices in
random order.  The order depends only on data.seed and the epoch, so a resumed run continues the same order.
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Sequence

from verl.experimental.dataset.sampler import AbstractSampler


def curriculum_order(n: int, easy: Sequence[int], easy_draws: int, seed: int, epoch: int) -> List[int]:
    """The index order of one epoch: in epoch 0 ``easy_draws`` indices from ``easy`` first (without repeats), then
    the other indices; every index exactly once per epoch."""
    rng = random.Random(f"{seed}|{epoch}")
    if epoch > 0 or easy_draws <= 0:
        order = list(range(n))
        rng.shuffle(order)
        return order
    easy = sorted(set(easy))
    if easy_draws > len(easy):
        raise ValueError(f"{easy_draws} easy draws wanted, only {len(easy)} easy tasks")
    first = rng.sample(easy, easy_draws)
    taken = set(first)
    rest = [i for i in range(n) if i not in taken]
    rng.shuffle(rest)
    return first + rest


def item_dstars(items: Sequence[Dict]) -> List[int]:
    """d* of every dataset item, from the task rows its env config points to (row = the item's seed)."""
    cache: Dict[str, List[int]] = {}
    out = []
    for it in items:
        path = it["config"]["jsonl_path"]
        if path not in cache:
            with open(path, encoding="utf-8") as f:
                cache[path] = [int(json.loads(l)["difficulty"]["d_star"]) for l in f if l.strip()]
        out.append(cache[path][int(it["seed"])])
    return out


class DifficultyCurriculumSampler(AbstractSampler):
    def __init__(self, data_source, data_config):
        cfg = data_config.sampler
        self.n = len(data_source)
        self.seed = int(data_config.get("seed") or 0)
        self.max_dstar = int(cfg.get("max_dstar", 3))
        self.easy_draws = int(cfg.get("easy_updates", 0)) * int(data_config.train_batch_size)
        dstars = item_dstars(getattr(data_source, "items"))
        self.easy = [i for i, d in enumerate(dstars) if d <= self.max_dstar]
        self.epoch = 0

    def __iter__(self):
        order = curriculum_order(self.n, self.easy, self.easy_draws, self.seed, self.epoch)
        self.epoch += 1
        return iter(order)

    def __len__(self):
        return self.n
