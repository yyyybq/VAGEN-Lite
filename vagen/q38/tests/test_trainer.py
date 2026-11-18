"""Tests of the q38 trainer mixin and checkpoint callback (DESIGN_ASYNC 5.1 T8, T9, T12).

- T8 (the P8-0 formula check, MF1): the same synthetic episodes go through verl V1's own loss
  (verl/verl/workers/utils/losses.py ppo_loss with seq-mean-token-mean, the dp / micro-batch metric aggregation of the
  engine) in this environment, and through the 8B update loop (F061 = /mnt/umm/users/yinbaiqiao/VAGEN-Lite/verl,
  dp_actor.py:567-716 with token-mean and one episode per micro-batch) in the old environment vagen-lite (``-I -B``,
  read-only import).  The 8B actor/kl_loss must equal ``kl_to_8b_unit`` of the V1 value (main line: x 8 / n_real),
  with and without padding rows and for the Q5 PT shape; pg_loss, pg_clipfrac, pg_clipfrac_lower (the synthetic rows
  force both the upper clip and the dual-clip), ppo_kl and the objective agree too.  GRPO: verl V1's own
  _compute_advantage (compute_advantage_for_multi_trajectories, V1/utils.py:147-217: last row per session, argmax of
  the mask, broadcast) on a balanced batch with padding rows against F061's compute_grpo_outcome_advantage called as
  RT:1459-1469 does, token by token (mixed, all-fail with return std >= 0.3318, std-0 and negative-return groups);
- T9: VagenCheckpointCallback on a global_step_* tree: COMPLETE, the kept set = IP.retained_steps, pruning to
  actor/huggingface + COMPLETE, the newest whole; the retention helpers are the same code as IP's;
- T12: the mixin with an in-memory TransferQueue, a fake worker group and a fake replay buffer that uses the
  sampler's own strict lookup: draw tags and warm-up, the T=1 calibration forward, the parent's GRPO + TIS followed by
  the factors (missing / unused factors raise, padding rows untouched, float32 like FI), the entropy guard, the
  diagnostics by staleness, the KL conversion.  (The sampler's own version-tag checks, MF3, are in test_sampler.py.)

Run (login node, 1 thread):
    OMP_NUM_THREADS=1 ... nice -n 19 python -B -m pytest vagen/q38/tests/test_trainer.py -q -x -p no:cacheprovider
"""

from __future__ import annotations

import ast
import copy
import json
import math
import os
import subprocess
import uuid
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from vagen.q38.tests.sampler_fixtures import work_dir

REPO = Path(__file__).resolve().parents[3]
SMOKE_CONFIG = REPO / "vagen/q38/configs/smoke_4b_example.yaml"
IP_SOURCE = REPO / "vagen/interiorgs_v10_ppo.py"
PY_8B = os.environ.get("Q38_PY_8B", "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python")
F061_ROOT = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/verl"


def smoke_config(**overrides):
    cfg = OmegaConf.load(SMOKE_CONFIG)
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    return cfg


# ======================================================================================================================
# fakes
# ======================================================================================================================


class TrainerTQ:
    """In-memory TransferQueue for the trainer: per-key fields (1-D tensors or Python values) and tags.  kv_batch_get
    returns a TensorDict of jagged nested tensors (NonTensorStack for non-tensors) like the rollout writes
    (TQW:237-242); a selected field no key has is left out (the parent asks for "values" without a critic), a field
    only some keys have raises."""

    def __init__(self):
        self.fields: dict[str, dict] = {}
        self.tags: dict[str, dict] = {}
        self.gets: list[tuple] = []

    def put_row(self, key, tag, **fields):
        self.fields[key] = dict(fields)
        self.tags[key] = dict(tag)

    def kv_batch_get(self, keys, partition_id, select_fields=None):
        from verl.utils import tensordict_utils as tu

        self.gets.append(tuple(select_fields or ()))
        names = list(select_fields) if select_fields else sorted({n for k in keys for n in self.fields[k]})
        out = {}
        for n in names:
            have = [n in self.fields[k] for k in keys]
            if not any(have):
                continue
            if not all(have):
                raise ValueError(f"field {n} missing for some keys")
            vals = [self.fields[k][n] for k in keys]
            if isinstance(vals[0], torch.Tensor):
                out[n] = torch.nested.as_nested_tensor([v.clone() for v in vals], layout=torch.jagged)
            else:
                out[n] = list(vals)
        return tu.get_tensordict(out)

    def kv_batch_put(self, keys, partition_id, fields=None, tags=None):
        from transfer_queue import KVBatchMeta

        if fields is not None:
            for n in list(fields.keys()):
                col = fields[n]
                rows = col.unbind() if getattr(col, "is_nested", False) else [col[i] for i in range(len(keys))]
                for k, v in zip(keys, rows, strict=True):
                    self.fields.setdefault(k, {})[n] = v.detach().clone()
        if tags is not None:
            for k, t in zip(keys, tags, strict=True):
                self.tags.setdefault(k, {}).update(copy.deepcopy(t))
        return KVBatchMeta(keys=list(keys), tags=[copy.deepcopy(self.tags[k]) for k in keys], partition_id=partition_id,
                           extra_info={})


@pytest.fixture
def fake_tq(monkeypatch):
    import transfer_queue

    fake = TrainerTQ()
    monkeypatch.setattr(transfer_queue, "kv_batch_get", fake.kv_batch_get)
    monkeypatch.setattr(transfer_queue, "kv_batch_put", fake.kv_batch_put)
    return fake


class FakeReplayBuffer:
    """Holds adv_scale like the sampler and uses the sampler's own strict lookup and clear."""

    def __init__(self, scale):
        from vagen.q38.sampler import VagenStepBatchSampler

        self.adv_scale = dict(scale)
        self._lookup = VagenStepBatchSampler.lookup_adv_scale
        self._clear = VagenStepBatchSampler.clear_adv_scale

    def lookup_adv_scale(self, real_uids):
        return self._lookup(self, real_uids)

    def clear_adv_scale(self):
        return self._clear(self)


class FakeWorkerGroup:
    """compute_log_prob writes log_probs / entropy over (prompt + response) as the actor worker does (the response part
    sits at [P-1, P+R-1), response_from_nested); the values depend on the temperature of the call."""

    def __init__(self, fake, seed=7):
        self.fake = fake
        self.calls: list[float] = []
        self.base: dict[str, tuple] = {}
        self.gen = torch.Generator().manual_seed(seed)
        self.update_metrics = None

    def compute_log_prob(self, batch):
        temp = float(batch.extra_info["temperature"])
        self.calls.append(temp)
        for k in batch.keys:
            f = self.fake.fields[k]
            p, r = len(f["prompts"]), len(f["responses"])
            if k not in self.base:
                self.base[k] = (-torch.rand(r, generator=self.gen) * 2 - 0.01,
                                torch.rand(r, generator=self.gen) * 3 + 0.05)
            logp, ent = self.base[k]
            full_lp = torch.full((p + r,), 99.0)
            full_en = torch.full((p + r,), 99.0)
            full_lp[p - 1: p + r - 1] = logp / temp
            full_en[p - 1: p + r - 1] = ent * temp
            f["log_probs"], f["entropy"] = full_lp, full_en
        return batch

    def update_actor(self, batch):
        return {"metrics": copy.deepcopy(self.update_metrics)}


def make_trainer(cls=None, cfg=None, rb=None, wg=None, global_steps=1):
    from vagen.q38.trainer import VagenColocateAsyncTrainer

    cls = cls or VagenColocateAsyncTrainer
    t = object.__new__(cls)
    t.config = cfg if cfg is not None else smoke_config()
    t.replay_buffer = rb
    t.actor_rollout_wg = wg
    t.global_steps = global_steps
    t.trainer_mode = t.config.trainer.v1.trainer_mode
    t._offset = 0
    t._restored_tq_prompt_count = 0
    return t


def add_batch(fake, groups, global_steps, n_pad=0, seed=3):
    """Rows of ``groups`` = [(uid, returns, (min, max) versions)] as the trainer sees them after _balance_batch,
    plus ``n_pad`` padding rows (padding_utils.py construct_minimal_padding_template).  Returns the KVBatchMeta."""
    from transfer_queue import KVBatchMeta

    g = torch.Generator().manual_seed(seed)
    keys, tags = [], []
    for uid, returns, (lo, hi) in groups:
        for i, ret in enumerate(returns):
            p = int(torch.randint(3, 7, (1,), generator=g))
            r = int(torch.randint(2, 7, (1,), generator=g))
            mask = torch.ones(r, dtype=torch.int64)
            if r > 3:
                mask[1] = 0  # an observation token inside the response
            rm = torch.zeros(r, dtype=torch.float32)
            rm[-1] = float(ret)
            rollout_lp = -torch.rand(r, generator=g) * 2 - 0.01
            key = f"{uid}_{i}_0"
            tag = {"status": "success", "global_steps": global_steps, "min_global_steps": lo, "max_global_steps": hi,
                   "seq_len": p + r}
            fake.put_row(key, tag, uid=uid, prompts=torch.arange(p) + 20, responses=torch.arange(r) + 40,
                         response_mask=mask, rm_scores=rm, rollout_log_probs=rollout_lp,
                         ref_log_prob=rollout_lp + 0.01)
            keys.append(key)
            tags.append(tag)
    if n_pad:
        pad_uid = f"pad{uuid.uuid4().hex}"
        for i in range(n_pad):
            key = f"{pad_uid}_{i}_0"
            tag = dict(tags[0], is_padding=True, prompt_len=127, response_len=1, seq_len=128)
            fake.put_row(key, tag, uid=pad_uid, prompts=torch.full((127,), 2), responses=torch.full((1,), 2),
                         response_mask=torch.zeros(1, dtype=torch.int64), rm_scores=torch.zeros(1),
                         rollout_log_probs=torch.zeros(1), ref_log_prob=torch.zeros(1))
            keys.append(key)
            tags.append(tag)
    return KVBatchMeta(keys=keys, tags=tags, partition_id="train", extra_info={"temperature": 0.8})


def mixed(n=8, k=3):
    return [1.0 if i < k else 0.0 for i in range(n)]


def resp(fake, key, name):
    return fake.fields[key][name]


# ======================================================================================================================
# T12: classes, construction
# ======================================================================================================================


def test_classes_are_the_native_mode_classes():
    from verl.trainer.ppo.v1 import PPOTrainerColocateAsync, PPOTrainerSync

    from vagen.q38 import ppo_v1, trainer

    assert issubclass(trainer.VagenColocateAsyncTrainer, PPOTrainerColocateAsync)
    assert issubclass(trainer.VagenSyncTrainer, PPOTrainerSync)
    assert trainer.VagenColocateAsyncTrainer.__mro__[1] is trainer.VagenTrainerMixin

    def cfg(mode):
        return OmegaConf.create({"trainer": {"v1": {"trainer_mode": mode}}, "vagen": {"trainer_impl": "vagen"}})

    assert ppo_v1.select_trainer_cls(cfg("colocate_async")) is trainer.VagenColocateAsyncTrainer
    assert ppo_v1.select_trainer_cls(cfg("sync")) is trainer.VagenSyncTrainer
    with pytest.raises(ValueError):  # phase 4
        ppo_v1.select_trainer_cls(cfg("separate_async"))


def test_init_checks_the_sampler_and_ga_8b():
    from vagen.q38.trainer import VagenTrainerMixin

    class Base:
        def __init__(self, config):
            self.config = config
            self.replay_buffer = Base.rb

    class T(VagenTrainerMixin, Base):
        pass

    Base.rb = object()  # verl's own replay buffer has no factors
    with pytest.raises(TypeError, match="VagenStepBatchSampler"):
        T(smoke_config())
    Base.rb = FakeReplayBuffer({})
    with pytest.raises(ValueError, match="ga_8b"):
        T(smoke_config(**{"vagen.ga_8b": None}))
    t = T(smoke_config())
    assert t._offset == 0 and t._ga_8b() == 12


# ======================================================================================================================
# T12: draw tags and the warm-up
# ======================================================================================================================


def test_draw_tags_warmup_and_dispatch():
    t = make_trainer(global_steps=1)
    dispatched = []
    t._add_prompts_to_generate = lambda n: dispatched.append(t._prompt_tags(list(range(n)))) or n

    t.on_train_begin()  # PPOTrainerColocateAsync: one warm-up batch (num_warmup_batches 1)
    assert len(dispatched) == 1 and len(dispatched[0]) == 24
    assert [tag["draw_batch"] for tag in dispatched[0]] == [0] * 24
    assert [tag["draw_index"] for tag in dispatched[0]] == list(range(24))
    assert all(tag["is_prompt"] and tag["status"] == "pending" and tag["global_steps"] == 1 for tag in dispatched[0])
    assert t._offset == 0
    step1 = t._prompt_tags(list(range(24)))  # prepare_step of step 1
    assert {tag["draw_batch"] for tag in step1} == {1}
    t.global_steps = 7
    assert {tag["draw_batch"] for tag in t._prompt_tags(list(range(3)))} == {7}


def test_warmup_after_resume():
    t = make_trainer(global_steps=11)
    dispatched = []
    t._add_prompts_to_generate = lambda n: dispatched.append(n) or n
    t._restored_tq_prompt_count = 24  # a colocate checkpoint: B_s restored whole, nothing to prefetch
    t.on_train_begin()
    assert dispatched == [] and t._offset == 0
    t._restored_tq_prompt_count = 5
    with pytest.raises(RuntimeError, match="restored 5 prompt groups"):
        t.on_train_begin()
    assert t._offset == 0

    def boom(n):
        raise OSError("dispatch failed")

    t._restored_tq_prompt_count = 0
    t._add_prompts_to_generate = boom
    with pytest.raises(OSError):
        t.on_train_begin()
    assert t._offset == 0  # restored even when the parent raises


def test_sync_has_no_warmup():
    from vagen.q38.trainer import VagenSyncTrainer

    t = make_trainer(VagenSyncTrainer, smoke_config(**{"trainer.v1.trainer_mode": "sync"}), global_steps=1)
    t._add_prompts_to_generate = lambda n: pytest.fail("sync prefetches nothing")
    t.on_train_begin()
    assert {tag["draw_batch"] for tag in t._prompt_tags(list(range(24)))} == {1}


# ======================================================================================================================
# T12: old log-probs and the entropy calibration
# ======================================================================================================================


def _expected_entropy(fake, wg, keys, temp, real):
    total, count = 0.0, 0
    for k in keys:
        if not real[k]:
            continue
        m = resp(fake, k, "response_mask").bool()
        total += float((wg.base[k][1] * temp).double()[m].sum())
        count += int(m.sum())
    return total / count, count


def test_old_log_prob_calibration_step(fake_tq):
    wg = FakeWorkerGroup(fake_tq)
    t = make_trainer(wg=wg, global_steps=1)
    batch = add_batch(fake_tq, [("ga", mixed(), (0, 0)), ("gb", mixed(k=5), (0, 0))], 1, n_pad=4)
    metrics = {}
    t._compute_old_log_prob(batch, metrics)
    assert wg.calls == [1.0, 0.8]  # T=1 first, the parent's T 0.8 forward last
    real = {k: not tag.get("is_padding", False) for k, tag in zip(batch.keys, batch.tags)}
    for k in batch.keys:  # old_log_probs are the T 0.8 values (TIS and the loss use them)
        r = len(resp(fake_tq, k, "responses"))
        assert torch.equal(resp(fake_tq, k, "old_log_probs"), wg.base[k][0] / 0.8)
        assert torch.equal(resp(fake_tq, k, "entropy"), wg.base[k][1] * 0.8)
        assert len(resp(fake_tq, k, "old_log_probs")) == r
    t1, n1 = _expected_entropy(fake_tq, wg, batch.keys, 1.0, real)
    t08, _ = _expected_entropy(fake_tq, wg, batch.keys, 0.8, real)
    assert metrics["vagen/guard/entropy_t1_calib"] == pytest.approx(t1, rel=1e-12)
    assert metrics["vagen/guard/entropy_t08_calib"] == pytest.approx(t08, rel=1e-12)
    assert metrics["vagen/guard/calib_tokens"] == n1
    assert "actor/entropy" in metrics  # the parent's own value is still produced
    assert all(isinstance(v, (int, float)) for k, v in metrics.items() if k.startswith("vagen/"))


def test_old_log_prob_other_steps(fake_tq):
    wg = FakeWorkerGroup(fake_tq)
    t = make_trainer(wg=wg, global_steps=2)
    batch = add_batch(fake_tq, [("ga", mixed(), (1, 1))], 2)
    metrics = {}
    t._compute_old_log_prob(batch, metrics)
    assert wg.calls == [0.8] and not any(k.startswith("vagen/guard") for k in metrics)
    t.config = smoke_config(**{"vagen.calib_entropy_t1_steps": [2, 5]})
    wg.calls.clear()
    t._compute_old_log_prob(batch, {})
    assert wg.calls == [1.0, 0.8]


# ======================================================================================================================
# T12: advantages x factors, entropy guard, diagnostics
# ======================================================================================================================


def _prepare_adv_batch(fake, wg, global_steps=5, n_pad=8):
    groups = [
        ("ga", [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], (global_steps - 1, global_steps - 1)),  # staleness 0
        ("gb", [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0], (global_steps - 2, global_steps - 2)),  # staleness 1
        ("gc", [0.5, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0], (global_steps - 2, global_steps - 1)),  # staleness 0, span 2
    ]
    batch = add_batch(fake, groups, global_steps, n_pad=n_pad)
    t = make_trainer(wg=wg, global_steps=global_steps)
    t._compute_old_log_prob(batch, {})  # old_log_probs, entropy (T 0.8)
    # the fake worker's old_log_probs have nothing to do with add_batch's rollout_log_probs; put every real row's
    # rollout within ln 2 of old (|old - rollout| <= 0.5: no TIS clip, but a non-zero log-prob difference) ...
    g = torch.Generator().manual_seed(17)
    for k, tag in zip(batch.keys, batch.tags):
        if tag.get("is_padding"):
            continue
        f = fake.fields[k]
        old = f["old_log_probs"]
        f["rollout_log_probs"] = old - (torch.rand(old.shape, generator=g) - 0.5)
    # ... then a token far off the rollout (TIS weight clipped at 2) in every row of group gb only (staleness 1)
    for k in batch.keys:
        if k.startswith("gb_"):
            f = fake.fields[k]
            f["old_log_probs"] = f["old_log_probs"].clone()
            f["old_log_probs"][0] = f["rollout_log_probs"][0] + 1.0
    return t, batch


def _snapshot(fake, keys, name):
    return {k: resp(fake, k, name).clone() for k in keys}


def test_advantage_factors_entropy_and_diagnostics(fake_tq):
    import transfer_queue

    wg = FakeWorkerGroup(fake_tq)
    t, batch = _prepare_adv_batch(fake_tq, wg)
    state = copy.deepcopy((fake_tq.fields, fake_tq.tags))

    # the parent alone (every factor 1): the reference advantages
    t.replay_buffer = FakeReplayBuffer({"ga": 1.0, "gb": 1.0, "gc": 1.0})
    t._compute_advantage(batch, {})
    base = _snapshot(fake_tq, batch.keys, "advantages")
    assert any(float(v.abs().sum()) > 0 for k, v in base.items() if not k.startswith("pad"))

    fake_tq.fields, fake_tq.tags = copy.deepcopy(state)
    f_gb = float(torch.tensor(0.61234567, dtype=torch.float32))  # a keep_by_return_std factor (a float32 value)
    t.replay_buffer = FakeReplayBuffer({"ga": 1.0, "gb": f_gb, "gc": 0.0})
    metrics = {"actor/entropy": 123.0}
    out = t._compute_advantage(batch, metrics)
    assert out.keys == batch.keys
    real = {k: not tag.get("is_padding", False) for k, tag in zip(batch.keys, batch.tags)}
    for k in batch.keys:
        adv = resp(fake_tq, k, "advantages")
        if not real[k]:
            assert torch.equal(adv, base[k])  # padding rows untouched
            continue
        f = {"ga": 1.0, "gb": f_gb, "gc": 0.0}[k.split("_")[0]]
        assert torch.equal(adv, base[k] * torch.tensor([f], dtype=torch.float32))  # bitwise, as FI:276-283
    assert t.replay_buffer.adv_scale == {}  # used once
    assert metrics["vagen/adv/real_rows"] == 24
    assert metrics["vagen/adv/scaled_groups"] == 2 and metrics["vagen/adv/zero_groups"] == 1

    # entropy guard: token mean over the real rows' response tokens; the V1 value is kept under _v1
    total, count = 0.0, 0
    for k in batch.keys:
        if real[k]:
            m = resp(fake_tq, k, "response_mask").bool()
            total += float(resp(fake_tq, k, "entropy").double()[m].sum())
            count += int(m.sum())
    assert metrics["actor/entropy_v1"] == 123.0
    assert metrics["actor/entropy"] == pytest.approx(total / count, rel=1e-12)
    assert metrics["vagen/guard/entropy_tokens"] == count

    # diagnostics by staleness: ga, gc -> 0 (V1's (s-1) - max), gb -> 1
    assert (metrics["vagen/async/rows_s0"], metrics["vagen/async/rows_s1"], metrics["vagen/async/rows_s_other"]) == (16, 8, 0)
    for b, prefixes in ((0, ("ga", "gc")), (1, ("gb",))):
        diff, w, clip, n = 0.0, 0.0, 0, 0
        for k in batch.keys:
            if k.split("_")[0] not in prefixes:
                continue
            m = resp(fake_tq, k, "response_mask").bool()
            d = (resp(fake_tq, k, "old_log_probs").double() - resp(fake_tq, k, "rollout_log_probs").double())[m]
            diff += float(d.abs().sum())
            w += float(resp(fake_tq, k, "rollout_is_weights").double()[m].sum())
            clip += int((d > math.log(2.0)).sum())
            n += int(m.sum())
        assert metrics[f"vagen/async/logp_diff_abs_mean_s{b}"] == pytest.approx(diff / n, rel=1e-12)
        assert metrics[f"vagen/async/tis_w_mean_s{b}"] == pytest.approx(w / n, rel=1e-12)
        assert metrics[f"vagen/async/tis_clip_frac_s{b}"] == pytest.approx(clip / n, rel=1e-12)
    assert metrics["vagen/async/tis_clip_frac_s1"] > 0 and metrics["vagen/async/tis_clip_frac_s0"] == 0
    # the TIS weights are the parent's: min(exp(old - rollout), 2) (token mode, threshold 2.0)
    k = next(k for k in batch.keys if k.startswith("gb_"))
    expect_w = torch.clamp(torch.exp(resp(fake_tq, k, "old_log_probs") - resp(fake_tq, k, "rollout_log_probs")), max=2.0)
    m = resp(fake_tq, k, "response_mask").bool()
    assert torch.allclose(resp(fake_tq, k, "rollout_is_weights")[m], expect_w[m], rtol=1e-6)
    assert all(isinstance(v, (int, float)) for kk, v in metrics.items() if kk.startswith(("vagen/", "actor/")))
    assert transfer_queue.kv_batch_get == fake_tq.kv_batch_get


def test_all_factors_one_writes_nothing(fake_tq):
    wg = FakeWorkerGroup(fake_tq)
    t, batch = _prepare_adv_batch(fake_tq, wg)
    t.replay_buffer = FakeReplayBuffer({"ga": 1.0, "gb": 1.0, "gc": 1.0})
    fake_tq.gets.clear()
    metrics = {}
    t._compute_advantage(batch, metrics)
    assert all("advantages" not in g for g in fake_tq.gets)  # nothing reads or rewrites the advantages
    assert metrics["vagen/adv/scaled_groups"] == 0


def test_fallback_zero_factors(fake_tq):
    wg = FakeWorkerGroup(fake_tq)
    t, batch = _prepare_adv_batch(fake_tq, wg, n_pad=0)
    t.replay_buffer = FakeReplayBuffer({"ga": 0.0, "gb": 0.0, "gc": 0.0})
    metrics = {}
    t._compute_advantage(batch, metrics)
    assert all(float(resp(fake_tq, k, "advantages").abs().sum()) == 0.0 for k in batch.keys)
    assert metrics["vagen/adv/zero_groups"] == 3


@pytest.mark.parametrize("scale, message", [
    ({"ga": 1.0, "gb": 0.5}, "without an advantage factor"),  # gc has no factor: never train it as 1
    ({"ga": 1.0, "gb": 0.5, "gc": 1.0, "gx": 1.0}, "missing from the batch"),  # a selected group not in the batch
])
def test_strict_factor_lookup(fake_tq, scale, message):
    wg = FakeWorkerGroup(fake_tq)
    t, batch = _prepare_adv_batch(fake_tq, wg)
    t.replay_buffer = FakeReplayBuffer(scale)
    with pytest.raises(KeyError, match=message):
        t._compute_advantage(batch, {})
    assert t.replay_buffer.adv_scale == scale  # nothing consumed


def test_missing_version_tag_raises(fake_tq):
    wg = FakeWorkerGroup(fake_tq)
    t, batch = _prepare_adv_batch(fake_tq, wg)
    batch.tags[3]["max_global_steps"] = None
    t.replay_buffer = FakeReplayBuffer({"ga": 1.0, "gb": 1.0, "gc": 1.0})
    with pytest.raises(RuntimeError, match="no min/max_global_steps"):
        t._compute_advantage(batch, {})


def test_other_staleness_is_counted(fake_tq):
    wg = FakeWorkerGroup(fake_tq)
    t, batch = _prepare_adv_batch(fake_tq, wg)
    for tag in batch.tags:
        if not tag.get("is_padding"):
            tag["max_global_steps"] = tag["min_global_steps"] = t.global_steps - 4  # 3 versions behind
    t.replay_buffer = FakeReplayBuffer({"ga": 1.0, "gb": 1.0, "gc": 1.0})
    metrics = {}
    t._compute_advantage(batch, metrics)
    assert (metrics["vagen/async/rows_s0"], metrics["vagen/async/rows_s1"], metrics["vagen/async/rows_s_other"]) == (0, 0, 24)
    assert not any(k.startswith("vagen/async/tis_w_mean") for k in metrics)


# ======================================================================================================================
# T12: KL in the 8B unit
# ======================================================================================================================


def _actor_batch(rows, n_pad):
    from transfer_queue import KVBatchMeta

    keys = [f"u{i // 8}_{i % 8}_0" for i in range(rows - n_pad)] + [f"padx_{i}_0" for i in range(n_pad)]
    tags = [{"is_padding": i >= rows - n_pad} for i in range(rows)]
    return KVBatchMeta(keys=keys, tags=tags, partition_id="train", extra_info={})


@pytest.mark.parametrize("rows, n_pad, factor", [
    (96, 0, 8 / 96),  # the main line: x 8 / n_real
    (96, 24, 8 / 72),  # 9 kept groups padded to one mini-batch
    (192, 0, 192 / (12 * 192)),  # two mini-batches of 96 (filter off)
])
def test_kl_to_the_8b_unit(rows, n_pad, factor):
    wg = FakeWorkerGroup(None)
    wg.update_metrics = {"kl_loss": [0.03, 0.05], "pg_loss": [0.1, 0.2], "mfu": [0.5]}
    t = make_trainer(wg=wg, global_steps=3)
    metrics = {}
    t._update_actor(_actor_batch(rows, n_pad), metrics)
    assert metrics["actor/kl_loss_v1"] == pytest.approx(0.04)
    assert metrics["actor/kl_loss"] == pytest.approx(0.04 * factor, rel=1e-12)
    assert metrics["actor/pg_loss"] == pytest.approx(0.15)  # other actor metrics untouched


def test_kl_conversion_errors():
    from vagen.q38.trainer import kl_to_8b_unit

    assert kl_to_8b_unit(0.5, 96, 12, 96) == pytest.approx(0.5 * 8 / 96)
    assert kl_to_8b_unit(0.5, 192, 24, 192) == pytest.approx(0.5 * 8 / 192)  # Q5 PT: mini 24, ga_8b 24
    with pytest.raises(ValueError):
        kl_to_8b_unit(0.5, 96, 12, 0)
    wg = FakeWorkerGroup(None)
    wg.update_metrics = {"kl_loss": [0.03], "mfu": [0.5]}
    t = make_trainer(wg=wg)
    with pytest.raises(ValueError, match="no real"):
        t._update_actor(_actor_batch(96, 96), {})
    with pytest.raises(RuntimeError, match="whole mini-batches"):
        t._update_actor(_actor_batch(100, 4), {})
    wg.update_metrics = {"pg_loss": [0.1], "mfu": [0.5]}  # use_kl_loss off: nothing to convert
    metrics = {}
    t._update_actor(_actor_batch(96, 0), metrics)
    assert "actor/kl_loss" not in metrics and "actor/kl_loss_v1" not in metrics


# ======================================================================================================================
# T8: the P8-0 formula check against the 8B code (F061 in the vagen-lite environment)
# ======================================================================================================================

# Runs under the vagen-lite python with -I -B.  The update loop is dp_actor.py:567-716 of F061 (commit a0ae85c7)
# reduced to the loss and its metrics (no model: log_prob is given), one rank after the other; the metrics of all ranks
# are averaged like DataProto.concat + reduce_metrics (protocol.py:926-968, fsdp_workers.py update_actor).
SCRIPT_8B = r'''
import json, sys
sys.path.insert(0, "@F061@")
import numpy as np
import torch
import verl
from omegaconf import OmegaConf
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty

assert verl.__file__.startswith("@F061@"), verl.__file__
spec = json.load(open(sys.argv[1]))
cfg = OmegaConf.create(spec["actor"])
rows = spec["rows"]
dp, per_gpu, micro = spec["dp"], spec["mini_per_gpu"], spec["micro_per_gpu"]
assert len(rows) % dp == 0
chunk = len(rows) // dp
metrics = {}
objective = 0.0
for rank in range(dp):
    data = rows[rank * chunk:(rank + 1) * chunk]
    mini_batches = [data[i:i + per_gpu] for i in range(0, len(data), per_gpu)]
    on_policy = len(mini_batches) == 1 and cfg.ppo_epochs == 1
    for mini_batch in mini_batches:
        gradient_accumulation = cfg.ppo_mini_batch_size // cfg.ppo_micro_batch_size_per_gpu
        micro_batches = [mini_batch[i:i + micro] for i in range(0, len(mini_batch), micro)]
        for mb in micro_batches:
            L = max(len(r["mask"]) for r in mb)
            def pad(name, fill=0.0):
                return torch.tensor([r[name] + [fill] * (L - len(r[name])) for r in mb], dtype=torch.float32)
            response_mask = pad("mask")
            log_prob, ref_log_prob, advantages = pad("logp"), pad("ref"), pad("adv")
            old_log_prob = pad("old")
            loss_scale_factor = 1 / gradient_accumulation
            if on_policy and not spec["force_old"]:
                old_log_prob = log_prob.detach()
            policy_loss_fn = get_policy_loss_fn(cfg.policy_loss.get("loss_mode", "vanilla"))
            pg_loss, pg_metrics = policy_loss_fn(old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
                                                 response_mask=response_mask, loss_agg_mode=cfg.loss_agg_mode,
                                                 config=cfg, rollout_is_weights=None)
            mbm = dict(pg_metrics)
            policy_loss = pg_loss
            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=cfg.kl_loss_type)
            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=cfg.loss_agg_mode)
            policy_loss = policy_loss + kl_loss * cfg.kl_loss_coef
            mbm["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
            mbm["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
            loss = policy_loss * loss_scale_factor
            objective += float(loss.detach().item()) / dp  # data-parallel gradients are averaged over ranks
            for k, v in mbm.items():
                metrics.setdefault(k, []).append(float(v.item() if hasattr(v, "item") else v))
grpo = None
if spec.get("grpo"):
    # the 8B GRPO call, RT:1459-1469 (vagen/ray_trainer.py compute_advantage; the split-reward branch after it is off
    # in the main line): right-padded token_level_rewards = rm_scores, response_mask, index = uid
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    g_rows = spec["grpo"]["rows"]
    L = max(len(r["mask"]) for r in g_rows)
    rewards = torch.tensor([r["rm"] + [0.0] * (L - len(r["rm"])) for r in g_rows], dtype=torch.float32)
    mask = torch.tensor([r["mask"] + [0] * (L - len(r["mask"])) for r in g_rows], dtype=torch.int64)
    index = np.array([r["uid"] for r in g_rows], dtype=object)
    adv, _ = compute_grpo_outcome_advantage(token_level_rewards=rewards, response_mask=mask, index=index,
                                            norm_adv_by_std_in_grpo=spec["grpo"]["norm_adv_by_std_in_grpo"])
    grpo = [adv[i, :len(r["mask"])].tolist() for i, r in enumerate(g_rows)]
print(json.dumps({"metrics": {k: float(np.mean(v)) for k, v in metrics.items()}, "objective": objective,
                  "grpo": grpo}))
'''


def synthetic_rows(n_real, n_pad, seed, groups_of=8):
    """Episodes as the 8B loss sees them (float32 values as Python floats, so both sides build the same tensors).
    Padding rows: response length 1, mask 0 (padding_utils.py construct_minimal_padding_template).

    Every fourth episode has a negative advantage and old = logp - 1.5 on every third token (ratio e^1.5 ~ 4.5 >
    clip_ratio_c 3: the dual-clip, core_algos.py compute_policy_loss_vanilla pg_losses3); the next one a positive
    advantage and old = logp - 0.5 there (ratio ~ 1.65 > 1 + clip_ratio_high: the upper clip).  Token 0 is always
    one of them and always in the mask.  The others: old = logp + N(0, 0.3), a random advantage."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n_real):
        r = int(torch.randint(4, 30, (1,), generator=g))
        mask = (torch.rand(r, generator=g) > 0.15).float()
        mask[0] = 1.0
        logp = -torch.rand(r, generator=g) * 3 - 0.01
        ref = logp + torch.randn(r, generator=g) * 0.4
        old = logp + torch.randn(r, generator=g) * 0.3  # ratios outside [0.8, 1.2]: some clipped either way
        adv_value = float(torch.randn(1, generator=g)) * 1.5
        forced = torch.arange(r) % 3 == 0
        if i % 4 == 0:  # dual-clip: adv < 0 and ratio > clip_ratio_c
            adv_value = -(0.5 + abs(adv_value))
            old = torch.where(forced, logp - 1.5, old)
        elif i % 4 == 1:  # upper clip: adv > 0 and ratio > 1 + clip_ratio_high
            adv_value = 0.5 + abs(adv_value)
            old = torch.where(forced, logp - 0.5, old)
        rows.append({"mask": mask.tolist(), "logp": logp.tolist(), "ref": ref.tolist(), "old": old.tolist(),
                     "adv": [adv_value] * r, "prompt_len": int(torch.randint(3, 9, (1,), generator=g)), "real": True})
    for _ in range(n_pad):
        rows.append({"mask": [0.0], "logp": [0.0], "ref": [0.0], "old": [0.0], "adv": [0.0], "prompt_len": 127,
                     "real": False})
    return rows


def v1_reference(rows, actor_cfg, dp, mini_n, micro_per_gpu=1):
    """verl V1's numbers on ``rows``: ppo_loss per micro-batch (losses.py:57-155), the per-rank Metric lists and their
    dp aggregation (Metric.aggregate_dp, utils/metric/utils.py:139-156, as engine_workers.py:323-341 does per
    mini-batch), then the mean over mini-batches (reduce_metrics).  The objective is the per-micro loss summed over a
    rank's micro-batches (backward per micro-batch) and averaged over the dp ranks (FSDP gradient reduction)."""
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric import Metric
    from verl.workers.utils.losses import ppo_loss

    assert len(rows) % mini_n == 0 and mini_n % dp == 0
    per_mini: dict[str, list[float]] = {}
    objective = 0.0
    for m0 in range(0, len(rows), mini_n):
        mini = rows[m0:m0 + mini_n]
        per_rank = mini_n // dp
        batch_num_tokens = int(sum(sum(r["mask"]) for r in mini))
        rank_metrics: list[dict] = []
        for rank in range(dp):
            mine = mini[rank * per_rank:(rank + 1) * per_rank]
            acc: dict[str, Metric] = {}
            for i in range(0, len(mine), micro_per_gpu):
                mb = mine[i:i + micro_per_gpu]

                def nested(name, dtype=torch.float32):
                    return torch.nested.as_nested_tensor([torch.tensor(r[name], dtype=dtype) for r in mb],
                                                         layout=torch.jagged)

                prompts = torch.nested.as_nested_tensor([torch.zeros(r["prompt_len"], dtype=torch.int64) for r in mb],
                                                        layout=torch.jagged)
                responses = torch.nested.as_nested_tensor([torch.zeros(len(r["mask"]), dtype=torch.int64) for r in mb],
                                                          layout=torch.jagged)
                data = tu.get_tensordict(
                    {"prompts": prompts, "responses": responses, "response_mask": nested("mask", torch.int64),
                     "old_log_probs": nested("old"), "advantages": nested("adv"), "ref_log_prob": nested("ref")},
                    non_tensor_dict={"dp_size": dp, "batch_num_tokens": batch_num_tokens, "global_batch_size": mini_n},
                )
                # model output over (prompt + response), the response at [P-1, P+R-1) (no_padding_2_padding)
                full = []
                for r in mb:
                    p, n = r["prompt_len"], len(r["mask"])
                    lp = torch.full((p + n,), 7.0)
                    lp[p - 1: p + n - 1] = torch.tensor(r["logp"], dtype=torch.float32)
                    full.append(lp)
                model_output = {"log_probs": torch.nested.as_nested_tensor(full, layout=torch.jagged)}
                loss, metrics = ppo_loss(actor_cfg, model_output, data)
                objective += float(loss.detach()) / dp
                for k, v in metrics.items():
                    if not isinstance(v, Metric):
                        continue
                    acc.setdefault(k, Metric(aggregation=v.aggregation)).append(v)
            rank_metrics.append(acc)
        for k in rank_metrics[0]:
            per_mini.setdefault(k, []).append(float(Metric.aggregate_dp([rm[k] for rm in rank_metrics])))
    # the trainer's rename_dict(..., "actor/") (TB:1937): "kl_loss" -> "actor/kl_loss"
    return {(k if k.startswith("actor/") else f"actor/{k}"): sum(v) / len(v) for k, v in per_mini.items()}, objective


def run_8b(rows, tmp, *, dp, mini_per_gpu, force_old, actor, grpo=None):
    if not os.path.exists(PY_8B):
        pytest.skip(f"the vagen-lite python is not at {PY_8B}")
    spec = {"rows": [r for r in rows if r["real"]], "dp": dp, "mini_per_gpu": mini_per_gpu, "micro_per_gpu": 1,
            "force_old": force_old, "actor": actor, "grpo": grpo}
    path = os.path.join(tmp, f"spec-{uuid.uuid4().hex[:6]}.json")
    with open(path, "w") as fh:
        json.dump(spec, fh)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                                 "RAYON_NUM_THREADS")})
    env.update(TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([PY_8B, "-I", "-B", "-c", SCRIPT_8B.replace("@F061@", F061_ROOT), path],
                          capture_output=True, text=True, timeout=3000, env=env)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def actor_8b(cfg, mini_8b):
    """The 8B actor section that matters (Y8:42-68): token-mean, one episode per micro-batch, mini 12 groups per GPU
    after the worker's normalisation (x n / dp)."""
    a = cfg.actor_rollout_ref.actor
    return {"clip_ratio": float(a.clip_ratio), "clip_ratio_low": float(a.clip_ratio_low),
            "clip_ratio_high": float(a.clip_ratio_high), "clip_ratio_c": float(a.clip_ratio_c),
            "loss_agg_mode": "token-mean", "kl_loss_type": str(a.kl_loss_type), "kl_loss_coef": float(a.kl_loss_coef),
            "ppo_epochs": 1, "ppo_mini_batch_size": mini_8b, "ppo_micro_batch_size_per_gpu": 1,
            "policy_loss": {"loss_mode": "vanilla"}}


@pytest.fixture(scope="module")
def t8_cases():
    """All T8 cases in one 8B subprocess each (importing torch and verl on /mnt/umm takes minutes)."""
    from verl.utils.config import omega_conf_to_dataclass

    cfg = smoke_config()
    assert cfg.actor_rollout_ref.actor.loss_agg_mode == "seq-mean-token-mean"
    # the GRPO rows ride along with the main case's subprocess (the real rows only, in the batch order: 8B has no
    # padding rows)
    grpo_batch = make_grpo_batch(TrainerTQ())
    grpo_spec = {"norm_adv_by_std_in_grpo": bool(cfg.algorithm.norm_adv_by_std_in_grpo),
                 "rows": [{"uid": f["uid"], "rm": f["rm_scores"].tolist(), "mask": f["response_mask"].tolist()}
                          for f in grpo_batch["real_fields"]]}
    out = {}
    with work_dir("b6_t8") as tmp:
        for name, n_real, n_pad, mini_groups, ga, seed in (("main", 96, 0, 12, 12, 11), ("padded", 72, 24, 12, 12, 12),
                                                           ("q5pt", 192, 0, 24, 24, 13)):
            c = copy.deepcopy(cfg)
            c.actor_rollout_ref.actor.ppo_mini_batch_size = mini_groups
            actor_cfg = omega_conf_to_dataclass(c.actor_rollout_ref.actor)
            rows = synthetic_rows(n_real, n_pad, seed=seed)
            mini_n = mini_groups * 8
            v1, v1_obj = v1_reference(rows, actor_cfg, dp=8, mini_n=mini_n)
            ref = run_8b(rows, tmp, dp=8, mini_per_gpu=mini_n // 8, force_old=True, actor=actor_8b(cfg, mini_n // 8),
                         grpo=grpo_spec if name == "main" else None)
            out[name] = {"n_real": n_real, "rows": n_real + n_pad, "ga": ga, "v1": v1, "v1_obj": v1_obj, "b8": ref}
    out["main"]["grpo_spec"] = grpo_spec
    return out


def test_t8_kl_constant_is_8(t8_cases):
    from vagen.q38.trainer import kl_to_8b_unit

    for name, case in t8_cases.items():
        kl_v1, kl_8b = case["v1"]["actor/kl_loss"], case["b8"]["metrics"]["actor/kl_loss"]
        converted = kl_to_8b_unit(kl_v1, case["rows"], case["ga"], case["n_real"])
        assert converted == pytest.approx(kl_8b, rel=1e-6, abs=1e-9), (name, kl_v1, kl_8b)
        assert kl_8b > 0
    main = t8_cases["main"]  # the locked constant: kl_8b = kl_v1 x 8 / n_real
    assert main["v1"]["actor/kl_loss"] * 8 / 96 == pytest.approx(main["b8"]["metrics"]["actor/kl_loss"], rel=1e-6)
    padded = t8_cases["padded"]
    assert padded["v1"]["actor/kl_loss"] * 8 / 72 == pytest.approx(padded["b8"]["metrics"]["actor/kl_loss"], rel=1e-6)


def _err(a, b):
    """The error of the T8 criterion (DESIGN 5.1 T8, <= 1e-6): absolute below magnitude 1, relative above."""
    return abs(a - b) / max(1.0, abs(b))


def test_t8_policy_loss_and_objective(t8_cases):
    errors = {}
    for name in ("main", "q5pt"):  # no padding rows: the same episodes on both sides
        case = t8_cases[name]
        v1, b8 = case["v1"], case["b8"]["metrics"]
        # clip and dual-clip both hit on both sides (synthetic_rows forces them), so both paths are compared
        for side, m in (("v1", v1), ("8b", b8)):
            assert m["actor/pg_clipfrac"] > 0 and m["actor/pg_clipfrac_lower"] > 0, (name, side, m)
        # pg_loss has the KL metric's shape (SUM / global batch vs mean / ga)
        errors[(name, "pg_loss")] = _err(v1["actor/pg_loss"] * case["rows"] / (case["ga"] * case["n_real"]),
                                         b8["actor/pg_loss"])
        for key in ("actor/pg_clipfrac", "actor/ppo_kl", "actor/pg_clipfrac_lower"):
            errors[(name, key)] = _err(v1[key], b8[key])
        errors[(name, "objective")] = _err(case["v1_obj"], case["b8"]["objective"])
    # with padding rows the objective still divides by mini x n = 96, like 8B's 8 ranks x ga 12 (R4)
    padded = t8_cases["padded"]
    errors[("padded", "objective")] = _err(padded["v1_obj"], padded["b8"]["objective"])
    assert len(errors) == 11
    print("T8 errors", errors)  # shown with -s / -rP
    bad = {k: v for k, v in errors.items() if not v <= 1e-6}
    assert not bad, (bad, errors)
    # the objectives are not trivially small (the comparison above is not an absolute-tolerance pass)
    for name in ("main", "q5pt", "padded"):
        assert abs(t8_cases[name]["b8"]["objective"]) > 1e-3, name


# GRPO groups of 8 (T8 GRPO): returns as rm_scores on the last response token (add_batch)
GRPO_STEP = 4
GRPO_GROUPS = (
    ("gmix", [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # mixed success
    ("gshape", [1.0, 0.35, 0.0, 0.6, 1.0, 0.1, 0.0, 0.85]),  # mixed, shaped returns
    ("gfail", [0.0, 0.9, 0.0, 0.0, 0.8, 0.0, 0.1, 0.0]),  # every episode failed, return std 0.388 >= 0.3318 (kept)
    ("gone", [1.0] * 8),  # std 0
    ("gzero", [0.0] * 8),  # std 0
    ("gneg", [-0.1, 0.0, 1.0, -0.1, 0.5, 0.0, 1.0, -0.2]),  # negative returns
)


def make_grpo_batch(fake, n_pad=8):
    """The GRPO rows (GRPO_GROUPS plus ``n_pad`` padding rows) in a shuffled (balanced) order.  Every third row has
    its first response token outside the mask, so V1's argmax (V1/utils.py:209) does not just take token 0;
    old_log_probs = rollout_log_probs (TIS weight 1).  Deterministic apart from the padding uid."""
    from transfer_queue import KVBatchMeta

    groups = [(uid, ret, (GRPO_STEP - 1, GRPO_STEP - 1)) for uid, ret in GRPO_GROUPS]
    meta = add_batch(fake, groups, GRPO_STEP, n_pad=n_pad, seed=29)
    for j, (k, tag) in enumerate(zip(meta.keys, meta.tags)):
        f = fake.fields[k]
        r = len(f["responses"])
        if not tag.get("is_padding") and j % 3 == 0 and r >= 3:
            f["response_mask"][0] = 0
        f["old_log_probs"] = f["rollout_log_probs"].clone()
        f["entropy"] = torch.ones(r)
    order = torch.randperm(len(meta.keys), generator=torch.Generator().manual_seed(31)).tolist()
    batch = KVBatchMeta(keys=[meta.keys[i] for i in order], tags=[meta.tags[i] for i in order],
                        partition_id=meta.partition_id, extra_info=dict(meta.extra_info))
    real_fields = [dict(fake.fields[k], key=k) for k, tag in zip(batch.keys, batch.tags) if not tag.get("is_padding")]
    return {"batch": batch, "real_fields": real_fields}


def test_t8_grpo(t8_cases, fake_tq):
    """V1's GRPO (the parent _compute_advantage through the mixin, every factor 1) equals F061's
    compute_grpo_outcome_advantage as RT calls it, on every token of every real row; padding rows get 0."""
    spec = t8_cases["main"]["grpo_spec"]
    b8 = t8_cases["main"]["b8"]["grpo"]
    assert b8 is not None and len(b8) == len(spec["rows"]) == 8 * len(GRPO_GROUPS)
    assert spec["norm_adv_by_std_in_grpo"] is True  # the main line (Y8:481)
    assert float(torch.tensor(dict(GRPO_GROUPS)["gfail"]).std()) >= 0.3318

    made = make_grpo_batch(fake_tq)
    batch = made["batch"]
    # the very rows the 8B side got (same order, same masks and returns)
    assert [{"uid": f["uid"], "rm": f["rm_scores"].tolist(), "mask": f["response_mask"].tolist()}
            for f in made["real_fields"]] == spec["rows"]
    assert any(f["response_mask"][0] == 0 for f in made["real_fields"])
    t = make_trainer(global_steps=GRPO_STEP, rb=FakeReplayBuffer({uid: 1.0 for uid, _ in GRPO_GROUPS}))
    t._compute_advantage(batch, {})

    worst, n_real, per_group = 0.0, 0, {}
    for k, tag in zip(batch.keys, batch.tags):
        adv = resp(fake_tq, k, "advantages")
        if tag.get("is_padding"):
            assert float(adv.abs().sum()) == 0.0, k
            continue
        ref = torch.tensor(b8[n_real], dtype=torch.float32)
        n_real += 1
        assert adv.dtype == torch.float32 and adv.shape == ref.shape, k
        worst = max(worst, float((adv.double() - ref.double()).abs().max()))
        m = resp(fake_tq, k, "response_mask").bool()
        assert float(adv[~m].abs().sum()) == 0.0, k  # tokens outside the mask carry no advantage
        per_group.setdefault(k.split("_")[0], []).append(float(adv[m][0]))
    assert n_real == len(b8)
    assert worst <= 1e-6, worst
    # the groups behave as GRPO says: std 0 -> 0; the others are normalised (sample std 1 within the group)
    assert all(v == 0.0 for g in ("gone", "gzero") for v in per_group[g])
    for g in ("gmix", "gshape", "gfail", "gneg"):
        s = torch.tensor(per_group[g], dtype=torch.float64)
        assert abs(float(s.mean())) < 1e-6 and float(s.std()) == pytest.approx(1.0, abs=1e-4), g


# ======================================================================================================================
# T9: the checkpoint callback
# ======================================================================================================================


def ckpt_config(root, **trainer):
    t = {"default_local_dir": str(root), "experiment_name": "q38-test", "max_actor_ckpt_to_keep": None,
         "max_critic_ckpt_to_keep": None, "ckpt_keep_every": 50, "ckpt_keep_latest": 2, "ckpt_prune_to_hf": True}
    t.update(trainer)
    return OmegaConf.create({"trainer": t, "actor_rollout_ref": {"actor": {"checkpoint": {
        "save_contents": ["model", "hf_model", "optimizer", "extra"], "async_save": False}}}})


def write_checkpoint(root: Path, step: int, hf=True):
    """What V1's _save_checkpoint leaves (TB:1005-1090): actor shards (+ HF), data.pt, the TQ snapshot, latest."""
    folder = root / f"global_step_{step}"
    (folder / "actor").mkdir(parents=True)
    (folder / "actor" / "model_world_size_8_rank_0.pt").write_text("w")
    (folder / "actor" / "optim_world_size_8_rank_0.pt").write_text("o")
    if hf:
        (folder / "actor" / "huggingface").mkdir()
        (folder / "actor" / "huggingface" / "model.safetensors").write_text("hf")
    (folder / "data.pt").write_text("d")
    (folder / "transfer_queue").mkdir()
    (folder / "transfer_queue" / "snapshot.bin").write_text("tq")
    (root / "latest_checkpointed_iteration.txt").write_text(str(step))
    return folder


def ip_function(name):
    tree = ast.parse(IP_SOURCE.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_retention_helpers_are_ip_s():
    from vagen.q38 import ckpt_callback

    ours = ast.parse(Path(ckpt_callback.__file__).read_text())
    for name in ("prune_to_hf", "retained_steps"):
        mine = next(n for n in ours.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(mine) == ast.dump(ip_function(name)), name


def test_callback_keeps_milestones_and_prunes():
    from vagen.q38.ckpt_callback import COMPLETE, VagenCheckpointCallback, retained_steps

    with work_dir("b6_t9") as tmp:
        root = Path(tmp) / "checkpoints"
        root.mkdir()
        cb = VagenCheckpointCallback(config=ckpt_config(root))
        saved = []
        (root / "global_step_7").mkdir()  # a save that never finished: no COMPLETE, never touched
        for step in range(10, 310, 10):
            folder = write_checkpoint(root, step)
            cb.on_save(trainer=None, global_step=step, checkpoint_dir=str(folder))
            saved.append(step)
            assert (folder / COMPLETE).read_text() == str(step)
            meta = json.loads((folder / "checkpoint_metadata.json").read_text())
            assert meta["global_step"] == step and meta["has_hf_model"] and meta["has_transfer_queue"]
            present = sorted(int(p.name.split("_")[-1]) for p in root.glob("global_step_*") if (p / COMPLETE).exists())
            assert set(present) == retained_steps(saved, 50, 2) | {step}
            newest = max(present)
            for s in present:
                f = root / f"global_step_{s}"
                if s == newest:  # whole: the resume point
                    assert (f / "data.pt").exists() and (f / "transfer_queue").exists()
                    assert (f / "actor" / "optim_world_size_8_rank_0.pt").exists()
                else:  # pruned to the HF export and the marker
                    assert sorted(p.name for p in f.iterdir()) == ["COMPLETE", "actor"]
                    assert [p.name for p in (f / "actor").iterdir()] == ["huggingface"]
                    assert (f / "actor" / "huggingface" / "model.safetensors").exists()
        assert sorted(int(p.name.split("_")[-1]) for p in root.glob("global_step_*")) == [7, 50, 100, 150, 200, 250, 280, 290, 300]

        # V1's own clean-up of step folders without actor/ (TB:996-1003) leaves the pruned milestones alone
        from verl.trainer.ppo.v1.trainer_base import PPOTrainer

        PPOTrainer._remove_pruned_step_folders(str(root), 310)
        assert sorted(int(p.name.split("_")[-1]) for p in root.glob("global_step_*")) == [50, 100, 150, 200, 250, 280, 290, 300]


def test_callback_refusals():
    from vagen.q38.ckpt_callback import VagenCheckpointCallback

    with work_dir("b6_t9r") as tmp:
        root = Path(tmp)
        with pytest.raises(ValueError, match="max_actor_ckpt_to_keep"):
            VagenCheckpointCallback(config=ckpt_config(root, max_actor_ckpt_to_keep=2))
        with pytest.raises(ValueError, match="max_critic_ckpt_to_keep"):
            VagenCheckpointCallback(config=ckpt_config(root, max_critic_ckpt_to_keep=2))
        cfg = ckpt_config(root)
        cfg.actor_rollout_ref.actor.checkpoint.async_save = True
        with pytest.raises(ValueError, match="async_save"):
            VagenCheckpointCallback(config=cfg)
        cb = VagenCheckpointCallback(config=ckpt_config(root))
        folder = write_checkpoint(root, 10, hf=False)  # save_contents has hf_model but no export was written
        with pytest.raises(RuntimeError, match="refusing COMPLETE"):
            cb.on_save(trainer=None, global_step=10, checkpoint_dir=str(folder))
        assert not (folder / "COMPLETE").exists()
        folder = write_checkpoint(root, 20)
        with pytest.raises(RuntimeError, match="asynchronous"):
            cb.on_save(trainer=None, global_step=20, checkpoint_dir=str(folder), async_save=True)

        class Critic:  # a run with a critic needs the critic folder too
            use_critic = True

        with pytest.raises(RuntimeError, match="refusing COMPLETE"):
            cb.on_save(trainer=Critic(), global_step=20, checkpoint_dir=str(folder))
        (folder / "critic").mkdir()
        cb.on_save(trainer=Critic(), global_step=20, checkpoint_dir=str(folder))
        assert (folder / "COMPLETE").exists()


def test_callback_resume_and_no_retention():
    from vagen.q38.ckpt_callback import VagenCheckpointCallback

    with work_dir("b6_t9s") as tmp:
        root = Path(tmp)
        cb = VagenCheckpointCallback(config=ckpt_config(root, ckpt_keep_every=4, ckpt_keep_latest=1))
        for step in (4, 6, 7):  # smoke S3's shape: save every few steps, killed after 7
            cb.on_save(trainer=None, global_step=step, checkpoint_dir=str(write_checkpoint(root, step)))
        assert sorted(p.name for p in root.glob("global_step_*")) == ["global_step_4", "global_step_7"]
        # resumed from 7: the next saves continue the same rule
        for step in (8, 10):
            cb.on_save(trainer=None, global_step=step, checkpoint_dir=str(write_checkpoint(root, step)))
        assert sorted(int(p.name.split("_")[-1]) for p in root.glob("global_step_*")) == [4, 8, 10]
        assert (root / "global_step_10" / "data.pt").exists() and not (root / "global_step_8" / "data.pt").exists()

    with work_dir("b6_t9n") as tmp:
        root = Path(tmp)
        cb = VagenCheckpointCallback(config=ckpt_config(root, ckpt_keep_every=0, ckpt_keep_latest=None,
                                                        ckpt_prune_to_hf=False))
        for step in (1, 2, 3):
            cb.on_save(trainer=None, global_step=step, checkpoint_dir=str(write_checkpoint(root, step)))
        assert len(list(root.glob("global_step_*"))) == 3
        assert all((p / "data.pt").exists() for p in root.glob("global_step_*"))


def test_callback_is_built_by_verl():
    from verl.trainer.ppo.checkpoint_callback import build_checkpoint_callback

    from vagen.q38 import contract
    from vagen.q38.ckpt_callback import VagenCheckpointCallback

    with work_dir("b6_t9b") as tmp:
        cfg = ckpt_config(Path(tmp))
        cfg.trainer.checkpoint_callback_class = contract.CHECKPOINT_CALLBACK
        assert isinstance(build_checkpoint_callback(cfg), VagenCheckpointCallback)
    smoke = smoke_config()
    assert smoke.trainer.checkpoint_callback_class == contract.CHECKPOINT_CALLBACK
    cb = VagenCheckpointCallback(config=smoke)
    assert (cb.keep_every, cb.keep_latest, cb.prune) == (50, 2, True)
