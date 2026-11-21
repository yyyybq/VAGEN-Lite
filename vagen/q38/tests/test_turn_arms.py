"""Phase 2 (docs/q38/DESIGN_ASYNC.md 2.3, 5.5, 6 W10 / W11): the per-turn estimators on the q38 stack.

- G3: the reward line's smoke dump (reward_runs/runs/rw_p2_smoke2/tc_dump, RW 1cecf88e, run rw_tcs2: turn_credit with
  the phase-2 defaults and nominal 24, three updates of 192 episodes each) recomputed by the q38 trainer's estimator
  call (vagen.q38.trainer.turn_advantage_rows): every token's advantage.  The dump keeps each advantage rounded to 5
  decimals (vagen/custom_advantage/turn_credit.py _dump), so the recomputed value rounded the same way must equal the
  dump (error <= 1e-6 on the rounded values, i.e. exact), and the unrounded one lies within the rounding (5e-6);
- T-adv: the mixin's _compute_advantage with turn_credit / turn_gae on an in-memory TransferQueue (test_trainer.py's
  fakes): the real rows' advantages are the same with and without padding rows (padding rows carry a copy of the first
  row's extra_fields, as padding_utils.py does) and equal the pure estimator on the real rows; padding rows get
  advantages 0 and returns 0 (turn_credit) / -100 (turn_gae); turn_gae's returns are -100 off each turn's first token,
  its advantages whitened over the real rows; the rollout correction is the parent's, bit for bit; factor rules;
- T-loss: three PPO passes over one batch through verl V1's own ppo_loss (losses.py:57-155, decoupled, vanilla with
  the dual clip, token TIS, seq-mean-token-mean): pass 1 at pi_old has ratio 1 (ppo_kl 0, no clip); after an
  optimizer step passes 2 and 3 move and clip; every pass equals a hand-written objective; the mixin hands epochs 3 to
  the worker;
- contract and rl-config rules of the two arms (Q5 PT = 辛, the main line's per-turn PPO = qw3tgd).

Run (login node, 1 thread):
    OMP_NUM_THREADS=1 ... nice -n 19 python -B -m pytest vagen/q38/tests/test_turn_arms.py -q -p no:cacheprovider
"""

from __future__ import annotations

import copy
import json
import math
import uuid
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from vagen.q38.tests.test_trainer import (
    FakeReplayBuffer,
    FakeWorkerGroup,
    TrainerTQ,
    make_trainer,
    smoke_config,
)

RW_SMOKE = Path("/mnt/umm/users/yinbaiqiao/reward_runs/runs/rw_p2_smoke2")
PHI_TABLE_8B = Path("/mnt/umm/users/yinbaiqiao/reward_runs/tables/phi_success/success_table_tcn3_v1.json")
STEP = 5


# ======================================================================================================================
# G3: the reward line's smoke dump
# ======================================================================================================================


def _dump_settings():
    import yaml

    from vagen.turn_credit import settings

    ppo = yaml.safe_load((RW_SMOKE / "rl_rw_tcs2" / "ppo.yaml").read_text())
    assert ppo["algorithm"]["adv_estimator"] == "turn_credit"
    return settings({"turn_credit": ppo["algorithm"]["turn_credit"]})


@pytest.mark.skipif(not (RW_SMOKE / "tc_dump").is_dir(), reason=f"needs {RW_SMOKE}/tc_dump")
@pytest.mark.parametrize("update", [0, 1, 2])
def test_g3_recompute_reward_line_dump(update):
    from vagen.q38.trainer import turn_advantage_rows

    cfg = _dump_settings()
    lines = [json.loads(x) for x in (RW_SMOKE / "tc_dump" / f"update_{update:03d}.jsonl").read_text().splitlines()]
    assert len(lines) == 192 and [r["i"] for r in lines] == list(range(192))
    length = max(len(r["mask"]) for r in lines)
    mask = torch.zeros((len(lines), length), dtype=torch.int64)
    rewards = torch.zeros((len(lines), length), dtype=torch.float32)
    for i, r in enumerate(lines):
        mask[i, :len(r["mask"])] = torch.tensor(r["mask"], dtype=torch.int64)
        if r["mask"]:
            rewards[i, len(r["mask"]) - 1] = r["return"]  # the episode return on its last answer token
    assert rewards.sum(dim=-1).tolist() == [r["return"] for r in lines]
    adv, ret, metrics = turn_advantage_rows("turn_credit", [r["record"] for r in lines], [r["uid"] for r in lines],
                                            mask, rewards, turn_credit=cfg)
    assert torch.equal(adv, ret)
    worst_rounded, worst_raw, tokens = 0.0, 0.0, 0
    for i, r in enumerate(lines):
        n = len(r["adv"])
        assert n == len(r["mask"])
        mine = [float(x) for x in adv[i, :n]]
        worst_rounded = max([worst_rounded] + [abs(round(a, 5) - b) for a, b in zip(mine, r["adv"])])
        worst_raw = max([worst_raw] + [abs(a - b) for a, b in zip(mine, r["adv"])])
        tokens += n
        assert float(adv[i, n:].abs().sum()) == 0.0  # nothing after the last answer token
    print(f"G3 update {update}: {tokens} tokens, max |round5 - dump| {worst_rounded:.3g}, max |raw - dump| "
          f"{worst_raw:.3g}")
    assert worst_rounded <= 1e-6
    assert worst_raw <= 5e-6 + 1e-7
    assert metrics["turn_credit/episodes"] == 192.0 and metrics["turn_credit/groups"] == 24.0


# ======================================================================================================================
# T-adv: the mixin's turn branches on the in-memory TransferQueue
# ======================================================================================================================


def _phi_table(tmp_path: Path) -> str:
    """A success table of the env's budget 12 (the 8B table itself when it is readable)."""
    if PHI_TABLE_8B.is_file():
        return str(PHI_TABLE_8B)
    g = torch.Generator().manual_seed(3)
    phi = torch.rand(13, 13, generator=g)
    phi[0] = 1.0
    p = tmp_path / "success_table.json"
    p.write_text(json.dumps({"version": "success_table_v1", "budget": 12, "phi": phi.tolist()}))
    return str(p)


def pt_settings(tmp_path):
    """辛's settings (YQ5 algorithm.turn_credit) with a readable phi table."""
    return {"episode_weight": 0, "baseline": "none", "phi_version": "success_table", "phi_table": _phi_table(tmp_path),
            "unreadable_as_wrong": True, "unreadable_floor": -0.3}


def episode(g):
    """One synthetic episode as the q38 agent loop records it: answers (mask 1) separated by observations (mask 0),
    the env reward of each turn, the reward line's turn records and turn_gae's turn rewards."""
    n_turns = int(torch.randint(1, 6, (1,), generator=g))
    mask, spans, rewards, turns = [], [], [], []
    d = int(torch.randint(1, 9, (1,), generator=g))
    d0, steps = d, 0
    for t in range(n_turns):
        if t:
            mask += [0] * int(torch.randint(1, 4, (1,), generator=g))
        start = len(mask)
        mask += [1] * int(torch.randint(1, 6, (1,), generator=g))
        valid = bool(torch.rand(1, generator=g) > 0.3)
        success = t == n_turns - 1 and bool(torch.rand(1, generator=g) > 0.5)
        if valid:
            steps = min(12, steps + int(torch.randint(1, 3, (1,), generator=g)))
            d = max(0, d + int(torch.randint(-2, 2, (1,), generator=g)))
        reward = 5.0 if success else (-0.1 if not valid else round(float(torch.rand(1, generator=g)) * 0.2 - 0.1, 3))
        spans.append([start, len(mask)])
        rewards.append(reward)
        turns.append({"span": [start, len(mask)], "reward": reward, "phi_after": float(torch.rand(1, generator=g)),
                      "d_after": 0 if success else d, "success": success, "valid": valid, "steps": steps})
    return {"mask": mask, "spans": spans, "rewards": rewards,
            "turn_records": {"phi0": float(torch.rand(1, generator=g)), "d0": d0, "turns": turns},
            "turn_rewards": {"spans": spans, "rewards": rewards}}


def turn_batch(fake, groups, n_pad=0, seed=5):
    """Rows of ``groups`` = [(uid, n episodes, (min, max) versions)] in TransferQueue, ``n_pad`` padding rows interleaved
    (padding_utils.py: response length 1, mask 0, a deep copy of the first row's fields, extra_fields included).
    Returns (KVBatchMeta, {key: episode})."""
    from transfer_queue import KVBatchMeta

    g = torch.Generator().manual_seed(seed)
    keys, tags, episodes = [], [], {}
    for uid, n, (lo, hi) in groups:
        for i in range(n):
            ep = episode(g)
            r = len(ep["mask"])
            p = int(torch.randint(3, 7, (1,), generator=g))
            rm = torch.zeros(r, dtype=torch.float32)
            rm[-1] = float(sum(ep["rewards"]))
            key = f"{uid}_{i}_0"
            tag = {"status": "success", "global_steps": STEP, "min_global_steps": lo, "max_global_steps": hi,
                   "seq_len": p + r}
            extra = {"turn_records": ep["turn_records"], "turn_rewards": ep["turn_rewards"],
                     "reward_extra_info": {"traj_success": float(ep["turn_records"]["turns"][-1]["success"])}}
            fake.put_row(key, tag, uid=uid, prompts=torch.arange(p) + 20, responses=torch.arange(r) + 40,
                         response_mask=torch.tensor(ep["mask"], dtype=torch.int64), rm_scores=rm,
                         rollout_log_probs=-torch.rand(r, generator=g) * 2 - 0.01,
                         ref_log_prob=-torch.rand(r, generator=g) * 2 - 0.01,
                         values=torch.randn(r, generator=g), extra_fields=extra)
            keys.append(key)
            tags.append(tag)
            episodes[key] = ep
    real_keys, real_tags = list(keys), list(tags)
    if n_pad:
        pad_uid = f"pad{uuid.uuid4().hex}"
        first = fake.fields[real_keys[0]]
        positions = sorted(torch.randperm(len(real_keys) + n_pad, generator=g)[:n_pad].tolist())
        for j, pos in enumerate(positions):
            key = f"{pad_uid}_{j}_0"
            tag = dict(real_tags[0], is_padding=True, prompt_len=127, response_len=1, seq_len=128)
            fake.put_row(key, tag, uid=pad_uid, prompts=torch.full((127,), 2), responses=torch.full((1,), 2),
                         response_mask=torch.zeros(1, dtype=torch.int64), rm_scores=torch.zeros(1),
                         rollout_log_probs=torch.zeros(1), ref_log_prob=torch.zeros(1), values=torch.zeros(1),
                         extra_fields=copy.deepcopy(first["extra_fields"]))
            keys.insert(pos, key)
            tags.insert(pos, tag)
    return KVBatchMeta(keys=keys, tags=tags, partition_id="train", extra_info={"temperature": 0.8}), episodes


GROUPS = [("ga", 8, (STEP - 1, STEP - 1)), ("gb", 8, (STEP - 2, STEP - 2)), ("gc", 8, (STEP - 2, STEP - 1))]
ONES = {"ga": 1.0, "gb": 1.0, "gc": 1.0}


def arm_config(estimator, tmp_path, **over):
    cfg = smoke_config()
    OmegaConf.update(cfg, "algorithm.adv_estimator", estimator, force_add=True)
    if estimator == "turn_credit":
        OmegaConf.update(cfg, "vagen.turn_credit", pt_settings(tmp_path), force_add=True)
    if estimator == "turn_gae":
        OmegaConf.update(cfg, "algorithm.gamma", 1.0, force_add=True)
        OmegaConf.update(cfg, "algorithm.lam", 0.95, force_add=True)
    for k, v in over.items():
        OmegaConf.update(cfg, k, v, force_add=True)
    return cfg


def run_adv(monkeypatch, estimator, tmp_path, n_pad, scale=None, cfg=None, use_critic=True, seed=5):
    """(fake TQ, batch, episodes, metrics) after old log-probs and the mixin's _compute_advantage."""
    import transfer_queue

    fake = TrainerTQ()
    monkeypatch.setattr(transfer_queue, "kv_batch_get", fake.kv_batch_get)
    monkeypatch.setattr(transfer_queue, "kv_batch_put", fake.kv_batch_put)
    batch, episodes = turn_batch(fake, GROUPS, n_pad=n_pad, seed=seed)
    wg = FakeWorkerGroup(fake)
    t = make_trainer(cfg=cfg if cfg is not None else arm_config(estimator, tmp_path), wg=wg, global_steps=STEP)
    t.use_critic = use_critic
    t._compute_old_log_prob(batch, {})
    t.replay_buffer = FakeReplayBuffer(scale or ONES)
    metrics = {}
    t._compute_advantage(batch, metrics)
    return fake, batch, episodes, metrics


def _real(batch):
    return [k for k, tag in zip(batch.keys, batch.tags) if not tag.get("is_padding")]


def _pads(batch):
    return [k for k, tag in zip(batch.keys, batch.tags) if tag.get("is_padding")]


@pytest.mark.parametrize("estimator", ["turn_credit", "turn_gae"])
def test_t_adv_padding_rows_change_nothing(monkeypatch, tmp_path, estimator):
    fake_p, batch_p, eps, metrics = run_adv(monkeypatch, estimator, tmp_path, n_pad=8)
    fake_0, batch_0, _, _ = run_adv(monkeypatch, estimator, tmp_path, n_pad=0)
    real = _real(batch_p)
    assert real == list(batch_0.keys) and len(real) == 24 and len(_pads(batch_p)) == 8
    for k in real:
        for name in ("advantages", "returns"):
            assert torch.equal(fake_p.fields[k][name], fake_0.fields[k][name]), (estimator, k, name)
    for k in _pads(batch_p):
        assert fake_p.fields[k]["advantages"].tolist() == [0.0]
        assert fake_p.fields[k]["returns"].tolist() == ([-100.0] if estimator == "turn_gae" else [0.0])
    assert metrics["vagen/adv/turn_rows"] == 24 and metrics["vagen/adv/real_rows"] == 24
    prefix = "turn_credit/" if estimator == "turn_credit" else "critic/turn_gae/"
    assert any(k.startswith(prefix) for k in metrics)
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in metrics.values())


@pytest.mark.parametrize("estimator", ["turn_credit", "turn_gae"])
def test_t_adv_equals_the_estimator_on_the_real_rows(monkeypatch, tmp_path, estimator):
    from vagen import turn_gae as TG
    from vagen.q38.trainer import turn_advantage_rows, turn_credit_settings

    fake, batch, eps, _ = run_adv(monkeypatch, estimator, tmp_path, n_pad=8)
    real = _real(batch)
    length = max(len(eps[k]["mask"]) for k in real)
    mask = torch.zeros((len(real), length), dtype=torch.int64)
    rewards = torch.zeros((len(real), length))
    values = torch.zeros((len(real), length))
    for i, k in enumerate(real):
        n = len(eps[k]["mask"])
        mask[i, :n] = torch.tensor(eps[k]["mask"])
        rewards[i, :n] = fake.fields[k]["rm_scores"]
        values[i, :n] = fake.fields[k]["values"]
    key = "turn_records" if estimator == "turn_credit" else "turn_rewards"
    cfg = arm_config(estimator, tmp_path)
    adv, ret, _ = turn_advantage_rows(estimator, [eps[k][key] for k in real], [k.split("_")[0] for k in real], mask,
                                      rewards, values, algorithm=cfg.algorithm,
                                      turn_credit=turn_credit_settings(cfg) if estimator == "turn_credit" else None)
    tokens, total, sq = 0, 0.0, 0.0
    for i, k in enumerate(real):
        n = len(eps[k]["mask"])
        assert torch.equal(fake.fields[k]["advantages"], adv[i, :n]), k
        assert torch.equal(fake.fields[k]["returns"], ret[i, :n]), k
        m = mask[i, :n].bool()
        assert float(fake.fields[k]["advantages"][~m].abs().sum()) == 0.0  # observation tokens keep 0
        tokens += int(m.sum())
        total += float(adv[i, :n][m].double().sum())
        sq += float((adv[i, :n][m].double() ** 2).sum())
        if estimator == "turn_gae":  # each turn's return on its first token, the sentinel elsewhere
            anchors, folded = TG.turn_anchors(eps[k]["turn_rewards"], mask[i, :n])
            assert folded == 0 and len(anchors) == len(eps[k]["spans"])
            first = {a for a, _, _ in anchors}
            r = fake.fields[k]["returns"]
            assert all((r[p] == -100.0) == (p not in first) for p in range(n))
    if estimator == "turn_gae":  # whitened over the real rows' response tokens: mean 0, unbiased std 1
        mean = total / tokens
        var = (sq - tokens * mean ** 2) / (tokens - 1)
        assert abs(mean) < 1e-5 and abs(var - 1.0) < 1e-4


def test_t_adv_rollout_correction_is_the_parents(monkeypatch, tmp_path):
    """The TIS weights the turn branch writes are the ones the parent's GRPO branch writes for the same rows."""
    fake_t, batch_t, _, mt = run_adv(monkeypatch, "turn_credit", tmp_path, n_pad=8)
    fake_g, batch_g, _, mg = run_adv(monkeypatch, "grpo", tmp_path, n_pad=8, cfg=smoke_config())
    assert len(batch_t.keys) == len(batch_g.keys)  # the same rows (the padding uids differ): compare by position
    for kt, kg in zip(batch_t.keys, batch_g.keys):
        for name in ("rollout_is_weights", "response_mask"):
            assert torch.equal(fake_t.fields[kt][name], fake_g.fields[kg][name]), (kt, name)
    is_keys = sorted(k for k in mg if k.startswith("rollout_corr/"))
    assert is_keys and [mt[k] for k in is_keys] == [mg[k] for k in is_keys]


def test_t_adv_factors(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="std-floor scale"):  # a keep scale under turn_gae
        run_adv(monkeypatch, "turn_gae", tmp_path, n_pad=0, scale={"ga": 1.0, "gb": 0.61, "gc": 1.0})
    fake, batch, _, metrics = run_adv(monkeypatch, "turn_gae", tmp_path, n_pad=8, scale={"ga": 0.0, "gb": 0.0, "gc": 0.0})
    assert all(float(fake.fields[k]["advantages"].abs().sum()) == 0.0 for k in batch.keys)  # the 8B fallback
    assert any(float((fake.fields[k]["returns"] != -100.0).sum()) > 0 for k in _real(batch))  # returns kept
    base, _, _, _ = run_adv(monkeypatch, "turn_credit", tmp_path, n_pad=0)
    fake, batch, _, _ = run_adv(monkeypatch, "turn_credit", tmp_path, n_pad=0, scale={"ga": 1.0, "gb": 0.0, "gc": 1.0})
    for k in batch.keys:
        f = 0.0 if k.startswith("gb_") else 1.0
        assert torch.equal(fake.fields[k]["advantages"], base.fields[k]["advantages"] * torch.tensor([f]))


def test_t_adv_refusals(monkeypatch, tmp_path):
    import transfer_queue

    with pytest.raises(RuntimeError, match="critic's values"):
        run_adv(monkeypatch, "turn_gae", tmp_path, n_pad=0, use_critic=False)
    cfg = arm_config("turn_credit", tmp_path)
    OmegaConf.update(cfg, "algorithm.turn_credit", {"baseline": "none"}, force_add=True)
    with pytest.raises(ValueError, match="vagen.turn_credit"):
        run_adv(monkeypatch, "turn_credit", tmp_path, n_pad=0, cfg=cfg)
    # a run whose agent loop did not record (another estimator's records only)
    fake = TrainerTQ()
    monkeypatch.setattr(transfer_queue, "kv_batch_get", fake.kv_batch_get)
    monkeypatch.setattr(transfer_queue, "kv_batch_put", fake.kv_batch_put)
    batch, _ = turn_batch(fake, GROUPS)
    for k in batch.keys:
        fake.fields[k]["extra_fields"].pop("turn_records")
    t = make_trainer(cfg=arm_config("turn_credit", tmp_path), wg=FakeWorkerGroup(fake), global_steps=STEP)
    t._compute_old_log_prob(batch, {})
    t.replay_buffer = FakeReplayBuffer(ONES)
    with pytest.raises(KeyError, match="extra_fields.turn_records"):
        t._compute_advantage(batch, {})


def test_t_adv_smoke_dump(monkeypatch, tmp_path):
    """turn_credit's smoke dump (settings dump_dir / dump_updates, as the reward line's estimator writes it)."""
    from vagen.custom_advantage import turn_credit as tc_estimator

    monkeypatch.setattr(tc_estimator, "_CALLS", [0])
    cfg = arm_config("turn_credit", tmp_path)
    OmegaConf.update(cfg, "vagen.turn_credit.dump_dir", str(tmp_path / "tc_dump"), force_add=True)
    OmegaConf.update(cfg, "vagen.turn_credit.dump_updates", 1, force_add=True)
    fake, batch, eps, _ = run_adv(monkeypatch, "turn_credit", tmp_path, n_pad=8, cfg=cfg)
    rows = [json.loads(x) for x in (tmp_path / "tc_dump" / "update_000.jsonl").read_text().splitlines()]
    real = _real(batch)
    assert [r["uid"] for r in rows] == [k.split("_")[0] for k in real]
    for r, k in zip(rows, real):
        assert r["record"] == eps[k]["turn_records"]
        assert r["adv"] == [round(float(x), 5) for x in fake.fields[k]["advantages"][:len(r["mask"])]]
        assert r["ids"] == fake.fields[k]["responses"][:len(r["mask"])].tolist()
    run_adv(monkeypatch, "turn_credit", tmp_path, n_pad=0, cfg=cfg)
    assert tc_estimator._CALLS[0] == 2 and not (tmp_path / "tc_dump" / "update_001.jsonl").exists()


# ======================================================================================================================
# T-loss: three passes of verl V1's ppo_loss
# ======================================================================================================================


def _actor_cfg(epochs=3, mini_groups=24):
    from verl.utils.config import omega_conf_to_dataclass

    cfg = smoke_config()
    cfg.actor_rollout_ref.actor.ppo_mini_batch_size = mini_groups
    cfg.actor_rollout_ref.actor.ppo_epochs = epochs
    return cfg, omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)


def _hand_objective(logp, old, ref, adv, w, mask, cfg, n_seq):
    """compute_policy_loss_vanilla (dual clip) x TIS + KL (low_var_kl) - entropy (none here), seq-mean-token-mean over
    n_seq sequences, written out (core_algos.py:1286-1376, 1194-1202; losses.py:57-155)."""
    lo, hi, c = cfg.clip_ratio_low, cfg.clip_ratio_high, cfg.clip_ratio_c
    ratio = torch.exp(torch.clamp(logp - old, -20, 20))
    l1 = -adv * ratio
    l2 = -adv * torch.clamp(ratio, 1 - lo, 1 + hi)
    clipped = torch.maximum(l1, l2)
    pg = torch.where(adv < 0, torch.minimum(-adv * c, clipped), clipped) * w
    kl_x = torch.clamp(ref - logp, -20, 20)  # kl_penalty_forward low_var_kl (core_algos.py:2239-2245)
    kl = torch.clamp(torch.exp(kl_x) - kl_x - 1, -10, 10)

    def seq_mean_token_mean(x):
        return float(((x * mask).sum(-1) / (mask.sum(-1) + 1e-8)).sum()) / n_seq

    return seq_mean_token_mean(pg) + cfg.kl_loss_coef * seq_mean_token_mean(kl), \
        float(((l2 > l1).float() * mask).sum() / mask.sum())


def test_t_loss_three_passes():
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.losses import ppo_loss

    cfg, actor_cfg = _actor_cfg()
    assert actor_cfg.ppo_epochs == 3 and actor_cfg.policy_loss.loss_mode == "vanilla"
    assert actor_cfg.loss_agg_mode == "seq-mean-token-mean"
    g = torch.Generator().manual_seed(23)
    rows, vocab, P = 192, 7, 4
    lengths = torch.randint(3, 12, (rows,), generator=g).tolist()
    tokens = [torch.randint(0, vocab, (n,), generator=g) for n in lengths]
    masks = [(torch.rand(n, generator=g) > 0.2).long() for n in lengths]
    for m in masks:
        m[0] = 1
    logits = torch.nn.Parameter(torch.randn(sum(lengths), vocab, generator=g))
    adv = [torch.full((n,), float(torch.randn(1, generator=g)) * 1.5) for n in lengths]

    def log_probs():
        lp = torch.log_softmax(logits, dim=-1).gather(-1, torch.cat(tokens).unsqueeze(-1)).squeeze(-1)
        return list(torch.split(lp, lengths))

    old = [x.detach().clone() for x in log_probs()]  # pi_old: recomputed once before the update (decoupled)
    rollout = [o + torch.randn(o.shape, generator=g) * 0.3 for o in old]
    weights = [torch.clamp(torch.exp(o - r), max=2.0) for o, r in zip(old, rollout)]  # token TIS, threshold 2
    ref = [o + torch.randn(o.shape, generator=g) * 0.2 for o in old]

    def nested(xs, dtype=torch.float32):
        return torch.nested.as_nested_tensor([x.to(dtype) for x in xs], layout=torch.jagged)

    def padded(xs):
        out = torch.zeros((rows, max(lengths)))
        for i, x in enumerate(xs):
            out[i, :len(x)] = x.detach().float()
        return out

    data = tu.get_tensordict(
        {"prompts": nested([torch.zeros(P, dtype=torch.int64)] * rows, torch.int64),
         "responses": nested(tokens, torch.int64), "response_mask": nested(masks, torch.int64),
         "old_log_probs": nested(old), "advantages": nested(adv), "ref_log_prob": nested(ref),
         "rollout_is_weights": nested(weights)},
        non_tensor_dict={"dp_size": 1, "batch_num_tokens": int(sum(int(m.sum()) for m in masks)),
                         "global_batch_size": rows},
    )
    opt = torch.optim.SGD([logits], lr=20.0)
    seen = []
    for epoch in range(3):
        lp = log_probs()
        full = [torch.cat([torch.full((P - 1,), 7.0), x, torch.zeros(1)]) for x in lp]  # response at [P-1, P+R-1)
        loss, metrics = ppo_loss(actor_cfg, {"log_probs": torch.nested.as_nested_tensor(full, layout=torch.jagged)},
                                 data)
        want, clipfrac = _hand_objective(padded(lp), padded(old), padded(ref), padded(adv), padded(weights),
                                         padded([m.float() for m in masks]), actor_cfg, rows)
        assert abs(float(loss) - want) <= 1e-6 * max(1.0, abs(want)), (epoch, float(loss), want)
        ppo_kl, frac = (float(metrics[k].aggregate()) if hasattr(metrics[k], "aggregate") else float(metrics[k])
                        for k in ("actor/ppo_kl", "actor/pg_clipfrac"))
        assert abs(frac - clipfrac) <= 1e-6
        seen.append((ppo_kl, frac))
        opt.zero_grad()
        loss.backward()
        assert logits.grad is not None and float(logits.grad.abs().sum()) > 0
        opt.step()
    print("T-loss passes (ppo_kl, pg_clipfrac):", seen)
    assert seen[0] == (0.0, 0.0)  # pass 1 at pi_old: ratio exactly 1
    assert all(abs(k) > 1e-4 and f > 0 for k, f in seen[1:])  # passes 2 and 3 have moved and clip


def test_t_loss_epochs_reach_the_worker(tmp_path):
    """The mixin's _update_actor hands ppo_epochs 3 and the 192-row mini-batch to the worker (TB:1924-1936), and the
    per-pass metrics come out (e1..e3)."""
    from transfer_queue import KVBatchMeta

    cfg, _ = _actor_cfg()
    OmegaConf.update(cfg, "vagen.ga_8b", 24, force_add=True)
    OmegaConf.update(cfg, "vagen.profile", "q5_pt", force_add=True)

    class WG:
        def __init__(self):
            self.extra = None

        def update_actor(self, batch):
            self.extra = dict(batch.extra_info)
            return {"metrics": {"kl_loss": [0.01, 0.02, 0.03], "pg_loss": [0.1, 0.2, 0.3], "ppo_kl": [0.0, 0.01, 0.02],
                                "grad_norm": [1.0, 2.0, 3.0], "mfu": [0.0, 0.0, 0.0]}}

    wg = WG()
    t = make_trainer(cfg=cfg, wg=wg, global_steps=STEP)
    keys = [f"u{i // 8}_{i % 8}_0" for i in range(192)]
    batch = KVBatchMeta(keys=keys, tags=[{} for _ in keys], partition_id="train", extra_info={})
    metrics = {}
    t._update_actor(batch, metrics)
    assert wg.extra["epochs"] == 3 and wg.extra["mini_batch_size"] == 192 and wg.extra["global_batch_size"] == 192
    assert [metrics[f"vagen/epoch/ppo_kl_e{e}"] for e in (1, 2, 3)] == [0.0, 0.01, 0.02]
    assert metrics["vagen/epoch/kl_loss_e1"] == pytest.approx(0.01 * 192 / (24 * 192))


# ======================================================================================================================
# contract and rl-config rules of the two arms
# ======================================================================================================================


def test_shapes_are_the_contracts():
    from data_gen.interiorgs_v10 import training_prep as tp
    from vagen.q38 import contract

    assert tp.Q38_PROFILE_SHAPES == contract.SHAPES
    assert tuple(tp.Q38_TURN_PPO_EPOCHS) == tuple(contract.TURN_PPO_EPOCHS)
    assert (tp.Q38_PT_FILTER, tp.Q38_PT_TARGET_GROUPS) == (contract.PT_FILTER, contract.PT_TARGET_GROUPS)
    assert set(tp.Q38_ESTIMATOR_PROFILES.values()) - {None} <= set(contract.PROFILES)
    p = tp.OPT_PACKAGES["p1tgae"]
    assert (p["gamma"], p["lam"], p["critic_warmup"]) == (contract.TURN_PPO_GAMMA, contract.TURN_PPO_LAM,
                                                         contract.TURN_PPO_CRITIC_WARMUP)


def test_pt_phi_table_is_the_8bs():
    import hashlib

    from vagen.q38 import contract

    if not PHI_TABLE_8B.is_file():
        pytest.skip(f"needs {PHI_TABLE_8B}")
    assert PHI_TABLE_8B.name == contract.PT_PHI_TABLE_NAME
    assert hashlib.sha256(PHI_TABLE_8B.read_bytes()).hexdigest() == contract.PT_PHI_TABLE_SHA256


def _options(**over):
    import argparse

    base = dict(trial=False, curriculum_easy_updates=None, invalid_turn_nonpositive=False, progress_adv_weight=None,
                logprob_tokens=False, greedy_val=False, history_mode="concat", opt_package="p1", test_freq=None,
                no_internal_val=True)
    base.update(over)
    return argparse.Namespace(**base)


PT = dict(turn_credit="{}", group_filter="nominal", target_groups=24)


@pytest.mark.parametrize("over", [
    {"opt_package": "p1tgae_c"}, {"opt_package": "p1tgae_fc"}, {"opt_package": "p1gae"},
    {"group_filter": "nominal"}, {"group_filter": "nominal", "target_groups": 24},  # nominal without --turn-credit
    {"turn_credit": "{}"}, {"turn_credit": "{}", "group_filter": "nominal"},  # PT needs all 24
    {**PT, "opt_package": "p1tgae"}, {**PT, "group_filter": "mixed_outcome"}, {**PT, "target_groups": 12},
    {"target_groups": 24}, {"opt_package": "p1tgae", "target_groups": 24}, {**PT, "candidates": 48},
])
def test_refused_arm_options(over):
    from data_gen.interiorgs_v10 import training_prep as tp

    with pytest.raises(SystemExit):
        tp.q38_check_options(_options(**over))


@pytest.mark.parametrize("over", [{}, {"opt_package": "p1tgae"}, {"opt_package": "p1tgae", "group_filter": "keep_by_return_std"},
                                  PT, {**PT, "candidates": 24, "rollouts": 8}])
def test_accepted_arm_options(over):
    from data_gen.interiorgs_v10 import training_prep as tp

    tp.q38_check_options(_options(**over))


# ---------------------------------------------------------------------------------------------------------------
# the contract's profiles of the two arms (test_contract.py's synthetic base config)


def _contract_base():
    from vagen.q38.tests.test_contract import Base

    class _B(Base):
        def runTest(self):  # noqa: N802 - unittest needs one
            pass

    _B.setUpClass()
    b = _B()
    return b


def _pt(cfg, table):
    from vagen.q38.tests.test_contract import A, put

    for key, value in (("vagen.profile", "q5_pt"), (A + ".ppo_mini_batch_size", 24), (A + ".ppo_epochs", 3),
                       ("vagen.ga_8b", 24), ("algorithm.adv_estimator", "turn_credit"), ("filter.enable", True),
                       ("filter.name", "nominal"), ("filter.filter_kwargs", {"target_groups": 24}),
                       ("vagen.turn_credit", {"episode_weight": 0, "baseline": "none", "phi_version": "success_table",
                                              "phi_table": table, "unreadable_as_wrong": True,
                                              "unreadable_floor": -0.3})):
        put(cfg, key, value)
    return cfg


def turn_ppo_critic(cfg):
    """The value network's keys of the per-turn PPO as rl-config writes them (phase 3: contract.TURN_PPO_CRITIC's
    values, TURN_PPO_CRITIC_SAME as interpolations, the critic's checkpoint contents), into ``cfg`` in place."""
    from vagen.q38 import contract as C

    for key, (value, _) in C.TURN_PPO_CRITIC.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    for key, (other, _) in C.TURN_PPO_CRITIC_SAME.items():
        OmegaConf.update(cfg, key, "${" + other + "}", force_add=True)
    OmegaConf.update(cfg, "critic.checkpoint.save_contents", list(C.TURN_PPO_CRITIC_SAVE), force_add=True)
    return cfg


def _turn_ppo(cfg, epochs=1):
    from vagen.q38.tests.test_contract import A, put

    for key, value in (("vagen.profile", "turn_ppo"), ("algorithm.adv_estimator", "turn_gae"), ("critic.enable", True),
                       ("algorithm.gamma", 1.0), ("algorithm.lam", 0.95), ("trainer.critic_warmup", 11),
                       (A + ".ppo_epochs", epochs)):
        put(cfg, key, value)
    return turn_ppo_critic(cfg)


@pytest.mark.skipif(not PHI_TABLE_8B.is_file(), reason=f"needs {PHI_TABLE_8B}")
def test_contract_q5_pt_arm(tmp_path):
    from vagen.q38 import contract as C
    from vagen.q38.tests.test_contract import A, put

    b = _contract_base()
    try:
        report = b.evaluate(_pt(b.base(), str(PHI_TABLE_8B)))
        assert report["passed"], C.summary(report)
        assert report["profile"]["shape"] == "q5_pt" and report["profile"]["filter"]
        other = tmp_path / "other_table.json"
        other.write_text(json.dumps({"version": "success_table_v1", "budget": 12, "phi": [[1.0] * 13] * 13}))
        for changes, rules in (
                ([("filter.name", "mixed_outcome")], ["filter.name"]),
                ([("filter.filter_kwargs", {"target_groups": 12})], ["filter.filter_kwargs"]),
                ([("filter.enable", False)], ["filter.enable"]),
                ([("vagen.turn_credit.baseline", "batch_mean")], ["vagen.turn_credit"]),
                ([("vagen.turn_credit.unreadable_floor", -0.5)], ["vagen.turn_credit"]),
                ([("vagen.turn_credit.phi_table", str(other))], ["vagen.turn_credit"]),  # not the 8B table
                ([("vagen.turn_credit.stall_penalty", 0.1)], ["vagen.turn_credit"]),  # a key outside 辛's
                ([("algorithm.turn_credit", {"baseline": "none"})], ["turn_credit_settings_place"]),
                ([(A + ".ppo_epochs", 1)], [A + ".ppo_epochs"]),
                ([("vagen.profile", None)], ["algorithm.adv_estimator"]),  # turn_credit on the main shape
                ([("critic.enable", True)], ["critic.enable"]),
        ):
            cfg = _pt(b.base(), str(PHI_TABLE_8B))
            for k, v in changes:
                put(cfg, k, v)
            report = b.evaluate(cfg)
            assert not report["passed"] and set(rules) <= set(report["violations"]), (changes, report["violations"])
        # the smoke's dump keys are allowed
        cfg = _pt(b.base(), str(PHI_TABLE_8B))
        put(cfg, "vagen.turn_credit.dump_dir", str(tmp_path / "d"))
        put(cfg, "vagen.turn_credit.dump_updates", 3)
        assert b.evaluate(cfg)["passed"]
        # smoke S4 (GRPO on the Q5 shape, every group kept) still passes
        cfg = _pt(b.base(), str(PHI_TABLE_8B))
        put(cfg, "algorithm.adv_estimator", "grpo")
        put(cfg, "filter.enable", False)
        from vagen.q38.tests.test_contract import delete

        delete(cfg, "vagen.turn_credit")
        report = b.evaluate(cfg)
        assert report["passed"], C.summary(report)
    finally:
        type(b).tearDownClass()


@pytest.mark.parametrize("epochs", [1, 3])
def test_contract_turn_ppo_arm(epochs):
    """Phase 3: the per-turn PPO's config with the 8B qw3tgd value network passes; the stock value loss (verl's own
    trainer classes) is refused and cannot be overridden; every critic key of qw3tgd is held."""
    from vagen.q38 import contract as C
    from vagen.q38.tests.test_contract import A, put

    b = _contract_base()
    try:
        report = b.evaluate(_turn_ppo(b.base(), epochs))
        assert report["passed"], C.summary(report)
        assert report["profile"]["shape"] == "turn_ppo"
        cfg = _turn_ppo(b.base(), epochs)
        put(cfg, "vagen.trainer_impl", "stock")
        put(cfg, "vagen.contract.allow_override", ["critic_value_loss_stock"])
        report = b.evaluate(cfg)
        assert not report["passed"] and "critic_value_loss_stock" in report["violations"]
        assert any("cannot be overridden" in p for p in report["problems"])
        for changes, rules in (
                ([("critic.enable", None)], ["critic.enable", "turn_gae_without_critic"]),
                ([("trainer.critic_warmup", 0)], ["trainer.critic_warmup"]),
                ([("algorithm.gamma", 0.95)], ["algorithm.gamma"]),
                ([(A + ".ppo_epochs", 2)], [A + ".ppo_epochs"]),
                ([(A + ".ppo_mini_batch_size", 24)], [A + ".ppo_mini_batch_size"]),
                ([("algorithm.rollout_correction.rollout_rs", "token_k1")], ["turn_gae_with_rollout_rs"]),
                ([("filter.name", "nominal")], ["filter.name"]),
                # phase 3: the value network as qw3tgd ran it
                ([("vagen.trainer_impl", "stock")], ["critic_value_loss_stock"]),
                ([("critic.optim.lr", 5e-5)], ["critic.optim.lr"]),  # p1tgae_c's critic lr
                ([("critic.cliprange_value", 1e6)], ["critic.cliprange_value"]),  # p1tgae_c's "no clip"
                ([("critic.optim.lr_scheduler_type", "cosine")], ["critic.optim.lr_scheduler_type"]),
                ([("critic.optim.clip_grad", 0.3)], ["critic.optim.clip_grad"]),
                ([("critic.grad_clip", 0.3)], ["critic.grad_clip"]),
                ([("critic.ppo_micro_batch_size_per_gpu", 2)], ["critic.ppo_micro_batch_size_per_gpu"]),
                ([("critic.use_dynamic_bsz", True)], ["critic.use_dynamic_bsz"]),
                ([("critic.loss_agg_mode", "token-mean")], ["critic.loss_agg_mode"]),
                ([("critic.ppo_epochs", 2)], ["critic.ppo_epochs"]),
                ([("critic.ppo_mini_batch_size", 24)], ["critic.ppo_mini_batch_size"]),
                ([("critic.model.path", "/elsewhere")], ["critic.model.path"]),
                ([("critic.strategy", "fsdp2" if b.base().actor_rollout_ref.actor.strategy != "fsdp2" else "fsdp")],
                 ["critic.strategy"]),
                ([("critic.fsdp.strategy", "megatron")], ["critic.fsdp.strategy"]),
                ([("critic.fsdp.param_offload", False)], ["critic.fsdp.param_offload"]),
                ([("critic.fsdp.model_dtype", "bf16")], ["critic.fsdp.model_dtype"]),
                ([("critic.model.use_remove_padding", False)], ["critic.model.use_remove_padding"]),
                ([("critic.checkpoint.save_contents", ["model"])], ["critic.checkpoint.save_contents"]),
                ([("critic.loss_scale_factor", 4096)], ["critic.loss_scale_factor"]),
        ):
            cfg = _turn_ppo(b.base(), epochs)
            for k, v in changes:
                put(cfg, k, v)
            report = b.evaluate(cfg)
            assert set(rules) <= set(report["violations"]), (changes, report["violations"])
    finally:
        type(b).tearDownClass()


# ---------------------------------------------------------------------------------------------------------------
# the step-batch sampler with the nominal filter (vagen/q38/sampler.py select_groups, FILTER_REGISTRY)


@pytest.mark.parametrize("candidates,target", [(24, 24), (30, 24)])
def test_sampler_nominal_filter(candidates, target):
    from vagen.q38.sampler import select_groups

    g = torch.Generator().manual_seed(candidates)
    uids, returns, success = [], [], []
    order = torch.randperm(candidates, generator=g).tolist()
    draw = {f"u{i}": order[i] for i in range(candidates)}
    for i in range(candidates):
        kind = i % 3  # all fail, mixed, all success
        for j in range(8):
            ok = kind == 2 or (kind == 1 and j < 3)
            uids.append(f"u{i}")
            success.append(float(ok))
            returns.append(5.0 if ok else -0.1 * j)
    sel = select_groups(uids, returns, {"traj_success": success}, draw,
                        {"enable": True, "name": "nominal", "filter_kwargs": {"target_groups": target}}, "turn_credit")
    want = sorted(draw, key=lambda u: draw[u])[:target]
    assert sel.selected == want and sel.adv_scale == {u: 1.0 for u in want} and not sel.fallback
    assert sel.metrics["filter/nominal/kept"] == target and sel.metrics["filter/nominal/candidates"] == candidates
    kinds = [sel.metrics[f"filter/nominal/{k}"] for k in ("mixed", "all_success", "all_fail")]
    assert sum(kinds) == target and all(k > 0 for k in kinds)  # every outcome kept
