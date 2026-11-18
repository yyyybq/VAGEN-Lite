"""The q38 step-batch sampler (vagen/q38/sampler.py): T1, T4, T5 of DESIGN_ASYNC 5.1, and its other checks.

T1, the 8B golden: every update of three 8B runs (keep_by_return_std qw3v4s3 and qw3dcks3, mixed_outcome qw3s0s3)
goes through ``select_groups`` with the 8B filter configuration and the draw order of the 8B dataloader
(scripts/q38/extract_golden.py reconstructs it and checks it against every dump).  Because the draw order is the 8B
one, the order-dependent paths (refill order, the zero-advantage fallback, ties at the truncation) are compared on
the real updates too, so no update is excluded (MF7 asked to exclude them only without the order).  Compared:
① the selected groups = the 8B kept_for_update tasks; ② every filter/* value of the 8B console line (and no other
filter/* name); ③ each selected group's factor = the FI formula (1, (std + eps) / (max(std, floor) + eps), 0 on the
fallback).  The order-dependent paths are also built synthetically through the sampler, with the draw positions
taken from the prompt tags.

T4 (in-memory TransferQueue): step s takes draw batch s-1 under colocate_async (W at step 1) and s under sync; a
resumed sampler takes the restored batch by its preserved tag; batch / tag errors raise.
T5: failed groups under failure_policy raise and skip; the zero-advantage fallback.
Also: the version-tag check (MF3), the strict factor lookup (MF2), the stuck warning (MF5), the filter switched off,
and one sample() through a real TransferQueue (Ray local, 2 CPUs).
"""

from __future__ import annotations

import logging
import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sampler_fixtures as fx  # noqa: E402

from vagen.q38.sampler import VagenStepBatchSampler, group_kind, select_groups  # noqa: E402

RUNS = [r for r in fx.GOLDEN_RUNS if os.path.exists(fx.golden_path(r))]
EPS = 1e-6  # FI:237


# ------------------------------------------------------------------------------------------------------------
# T1: the 8B golden
# ------------------------------------------------------------------------------------------------------------


def _expected_factor(kind: str, std: float, fallback: bool, filter_cfg: dict) -> float:
    if fallback:
        return 0.0
    kwargs = filter_cfg.get("filter_kwargs") or {}
    if filter_cfg["name"] == "keep_by_return_std" and kind == "all_fail":
        floor = float(kwargs.get("std_floor") or 0.0)
        return float(np.float32((std + EPS) / (max(std, floor) + EPS)))
    return 1.0


@pytest.mark.parametrize("run", RUNS)
def test_golden_selection(run):
    header, steps = fx.load_golden(run)
    coverage = {"updates": 0, "fallback": 0, "refill_order": 0, "filled": 0, "truncation_tie": 0}
    with fx.work_dir("q38-golden") as d:
        cfg = fx.golden_filter_cfg(header, steps, d)
        for step in steps:
            s = step["step"]
            uids, returns, extra, draw = fx.golden_episodes(step)
            sel = select_groups(uids, returns, extra, draw, cfg, header["adv_estimator"])
            kept_8b = {f"g{g}" for g, k in zip(step["g"], step["kept"]) if k}
            # (1) the selected groups
            assert set(sel.selected) == kept_8b, (
                f"{run} update {s}: ours {sorted(sel.selected)}, 8B {sorted(kept_8b)}"
            )
            assert sel.selected == sorted(sel.selected, key=lambda u: draw[u])
            # (2) the filter metrics, exactly
            logged = {k: v for k, v in step["log"].items() if k.startswith("filter/")}
            assert set(sel.metrics) == set(logged), (
                f"{run} update {s}: only ours {sorted(set(sel.metrics) - set(logged))}, "
                f"only 8B {sorted(set(logged) - set(sel.metrics))}"
            )
            for k, v in logged.items():
                assert float(sel.metrics[k]) == float(v), f"{run} update {s} {k}: ours {sel.metrics[k]!r}, 8B {v!r}"
            # (3) the factors
            groups = {g["uid"]: g for g in sel.groups}
            fallback = bool(sel.metrics.get("filter/mixed/empty_fallback"))
            assert fallback == sel.fallback
            for u in sel.selected:
                want = _expected_factor(groups[u]["kind"], groups[u]["return_std"], fallback, cfg)
                assert sel.adv_scale[u] == want, f"{run} update {s} group {u}: factor {sel.adv_scale[u]} != {want}"
            # what the update exercised
            coverage["updates"] += 1
            coverage["fallback"] += int(fallback)
            mixed = [g for g in sel.groups if g["kind"] == "mixed"]
            coverage["refill_order"] += int(len(mixed) > int(cfg["filter_kwargs"].get("target_groups") or 0))
            filled = [g for g in sel.groups if g["kind"] == "all_fail" and g["selected"] and not fallback]
            coverage["filled"] += len(filled)
            if filled and cfg["name"] == "keep_by_return_std":
                thr = float(cfg["filter_kwargs"]["std_threshold"])
                last = min(g["return_std"] for g in filled)
                left = [g for g in sel.groups if g["kind"] == "all_fail" and not g["selected"]
                        and g["return_std"] >= thr and g["return_std"] == last]
                coverage["truncation_tie"] += int(bool(left))
    assert coverage["updates"] == header["updates"]
    print(f"\n[T1 {run}] {coverage}")


def test_golden_spot_values():
    """DESIGN_ASYNC 5.1 T1's example: update 100 of qw3v4s3, filled_std_mean = 0.44862210750579834."""
    if "qw3v4s3" not in RUNS:
        pytest.skip("golden run qw3v4s3 not extracted")
    header, steps = fx.load_golden("qw3v4s3")
    step = next(s for s in steps if s["step"] == 100)
    with fx.work_dir("q38-golden") as d:
        cfg = fx.golden_filter_cfg(header, steps, d)
        uids, returns, extra, draw = fx.golden_episodes(step)
        sel = select_groups(uids, returns, extra, draw, cfg, "grpo")
    assert sel.metrics["filter/keep/filled_std_mean"] == 0.44862210750579834
    assert sel.metrics["filter/keep/filled_min_steps_mean"] == 4.5
    assert len(sel.selected) == 8
    scales = sorted(v for v in sel.adv_scale.values() if v != 1.0)
    assert len(scales) == 2 and all(0 < v < 1 for v in scales)


# ------------------------------------------------------------------------------------------------------------
# T1, synthetic: the order-dependent paths through the sampler (draw positions from the prompt tags)
# ------------------------------------------------------------------------------------------------------------


MIXED = ([2.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0, -1.0], [1, 0, 1, 0, 1, 0, 1, 0])
ALL_SUCCESS = ([5.0] * 8, [1] * 8)


def _all_fail(spread: float) -> tuple[list[float], list[int]]:
    """An all-fail group whose returns have a sample std (n - 1) of about ``spread``: +-a around 0 gives
    a * sqrt(8/7)."""
    a = spread / float(np.sqrt(8.0 / 7.0))
    return [a, -a] * 4, [0] * 8


def _run_one_batch(fake, sampler, kinds_by_draw, order, global_steps=1, versions=(0, 0)):
    uids = fx.dispatch(fake, draw_batch=global_steps - 1, global_steps=global_steps,
                       n_groups=len(kinds_by_draw), order=order)
    for j in order:
        rets, oks = kinds_by_draw[j]
        fx.finish_group(fake, uids[j], rets, oks, global_steps=global_steps, versions=versions,
                        task_id=f"task{j}")
    batch, metrics = sampler.sample(global_steps=global_steps, partition_id="train", batch_size=24)
    return uids, batch, metrics


def test_refill_follows_draw_tags_not_put_order(monkeypatch):
    fake = fx.FakeTQ().install(monkeypatch)
    sampler = fx.make_sampler()
    mixed_at = {1, 3, 5, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}  # 3 in the nominal 12, 10 among the extras
    kinds = [MIXED if j in mixed_at else ALL_SUCCESS for j in range(24)]
    order = list(range(24))[::-1]  # put in reverse: the dict order is the opposite of the draw order
    uids, batch, metrics = _run_one_batch(fake, sampler, kinds, order)
    want = [uids[j] for j in (1, 3, 5, 13, 14, 15, 16, 17, 18, 19, 20, 21)]  # 9 refills in draw order
    assert sampler.last_step["selected"] == want
    assert {k.split("_")[0] for k in batch.keys} == set(want)
    assert len(batch.keys) == 12 * 8
    assert metrics["filter/mixed/refilled"] == 9 and metrics["filter/mixed/kept"] == 12
    assert metrics["filter/mixed/dropped_all_success"] == 9
    assert set(sampler.adv_scale) == set(want) and set(sampler.adv_scale.values()) == {1.0}


def test_fallback_keeps_the_nominal_batch_with_zero_factors(monkeypatch):
    """T5: no group carries a signal: the nominal 12 (draw positions 0..11) are kept with factor 0."""
    fake = fx.FakeTQ().install(monkeypatch)
    sampler = fx.make_sampler()
    order = random.Random(3).sample(range(24), 24)
    uids, batch, metrics = _run_one_batch(fake, sampler, [ALL_SUCCESS] * 24, order)
    assert sampler.last_step["selected"] == uids[:12]
    assert sampler.last_step["fallback"] is True
    assert set(sampler.adv_scale.values()) == {0.0}
    assert metrics["filter/mixed/empty_fallback"] == 1.0 and metrics["filter/mixed/kept"] == 0
    # keep_by_return_std falls back the same way
    fake2 = fx.FakeTQ().install(monkeypatch)
    keep = fx.make_sampler(filter_cfg={"enable": True, "name": "keep_by_return_std",
                                       "filter_kwargs": {"target_groups": 12, "std_threshold": 0.3318,
                                                         "std_floor": 1.0}})
    uids2, _batch2, m2 = _run_one_batch(fake2, keep, [_all_fail(0.1)] * 24, order)
    assert keep.last_step["selected"] == uids2[:12]
    assert set(keep.adv_scale.values()) == {0.0}
    assert m2["filter/keep/kept_total"] == 0 and m2["filter/mixed/empty_fallback"] == 1.0
    del fake, fake2


def test_keep_tie_at_truncation_goes_to_the_earlier_draw(monkeypatch):
    fake = fx.FakeTQ().install(monkeypatch)
    cfg = {"enable": True, "name": "keep_by_return_std",
           "filter_kwargs": {"target_groups": 12, "std_threshold": 0.3318, "std_floor": 1.0}}
    sampler = fx.make_sampler(filter_cfg=cfg)
    kinds = [ALL_SUCCESS] * 24
    big = [3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]  # 11 groups with distinct large spreads
    for r, j in enumerate(big):
        kinds[j] = _all_fail(2.0 + 0.1 * r)
    for j in (20, 5, 9):  # three equal spreads below the others: one place left, the earliest draw (5) takes it
        kinds[j] = _all_fail(0.5)
    order = random.Random(7).sample(range(24), 24)
    uids, _batch, metrics = _run_one_batch(fake, sampler, kinds, order)
    chosen = set(sampler.last_step["selected"])
    assert uids[5] in chosen and uids[9] not in chosen and uids[20] not in chosen
    assert len(chosen) == 12 and metrics["filter/keep/filled"] == 12
    std5 = float(torch.std(torch.tensor(kinds[5][0], dtype=torch.float32)))
    assert std5 < 1.0
    assert sampler.adv_scale[uids[5]] == float(np.float32((std5 + EPS) / (1.0 + EPS)))
    big_scale = sampler.adv_scale[uids[3]]
    assert big_scale == 1.0  # spread above the floor: (s + eps) / (s + eps)


def test_group_kind():
    assert group_kind(np.array([True, False])) == "mixed"
    assert group_kind(np.array([True, True])) == "all_success"
    assert group_kind(np.array([False, False])) == "all_fail"


# ------------------------------------------------------------------------------------------------------------
# T4: draw batches, warm-up, resume, sync
# ------------------------------------------------------------------------------------------------------------


def _finish_batch(fake, uids, global_steps, versions, kinds=None):
    for j, u in enumerate(uids):
        rets, oks = (kinds[j] if kinds else (MIXED if j % 2 == 0 else ALL_SUCCESS))
        fx.finish_group(fake, u, rets, oks, global_steps=global_steps, versions=versions, task_id=f"t{j}")


def _train_step(fake, sampler, s):
    batch, metrics = sampler.sample(global_steps=s, partition_id="train", batch_size=24)
    # the trainer clears the trained trajectories at the end of the step (TB:536) and uses the factors
    sampler.lookup_adv_scale([k.split("_")[0] for k in batch.keys])
    sampler.clear_adv_scale()
    fake.kv_clear(batch.keys, "train")
    return batch, metrics


def test_colocate_async_takes_the_previous_draw_batch(monkeypatch):
    fake = fx.FakeTQ().install(monkeypatch)
    sampler = fx.make_sampler()
    w = fx.dispatch(fake, draw_batch=0, global_steps=1, n_groups=24)  # on_train_begin's warm-up batch W
    b1 = fx.dispatch(fake, draw_batch=1, global_steps=1, n_groups=24)  # step 1's prepare_step
    _finish_batch(fake, w, 1, (0, 0))
    _finish_batch(fake, b1[:5], 1, (0, 0))  # B1 partly done
    batch, m = _train_step(fake, sampler, 1)
    assert {k.split("_")[0] for k in batch.keys} <= set(w)
    assert m["filter/step/draw_batch"] == 0 and m["filter/step/candidates_finished"] == 24
    assert m["filter/step/next_batch_finished_groups"] == 5
    assert m["vagen/async/version_violations"] == 0
    assert not {k for k in fake.keys() if k.split("_")[0] in set(w)}  # W fully cleared
    assert fake.prompt_keys() == set(b1)
    # step 2: B2 dispatched, B1 finishes (part of it generated under w_1)
    b2 = fx.dispatch(fake, draw_batch=2, global_steps=2, n_groups=24)
    _finish_batch(fake, b1[5:], 1, (0, 1))
    batch2, m2 = _train_step(fake, sampler, 2)
    assert {k.split("_")[0] for k in batch2.keys} <= set(b1)
    assert m2["filter/step/draw_batch"] == 1 and m2["vagen/async/version_spans_max"] in (1, 2)
    assert fake.prompt_keys() == set(b2)
    assert m2["filter/step/gen_s_2step_mean"] == pytest.approx((m["filter/step/wait_s"] + m2["filter/step/wait_s"]) / 2)


def test_resume_takes_the_restored_batch_by_its_tag(monkeypatch):
    """A resume at step k+1 re-dispatches B_k with its tags (draw_batch k kept, TB:984-987); a fresh sampler (new
    process) takes it, not B_{k+1}."""
    fake = fx.FakeTQ().install(monkeypatch)
    k = 7
    restored = fx.dispatch(fake, draw_batch=k, global_steps=k, n_groups=24)
    for u in restored:  # the re-issue rewrites is_prompt, status and global_steps only
        fake.kv_put(u, "train", tag={"is_prompt": True, "status": "pending", "global_steps": k + 1})
    nxt = fx.dispatch(fake, draw_batch=k + 1, global_steps=k + 1, n_groups=24)
    _finish_batch(fake, restored, k + 1, (k, k))
    sampler = fx.make_sampler()
    batch, m = _train_step(fake, sampler, k + 1)
    assert m["filter/step/draw_batch"] == k
    assert {key.split("_")[0] for key in batch.keys} <= set(restored)
    assert fake.prompt_keys() == set(nxt)


def test_sync_takes_its_own_draw_batch(monkeypatch):
    fake = fx.FakeTQ().install(monkeypatch)
    sampler = fx.make_sampler(mode="sync")
    for s in (1, 2):
        b = fx.dispatch(fake, draw_batch=s, global_steps=s, n_groups=24)
        _finish_batch(fake, b, s, (s - 1, s - 1))
        batch, m = _train_step(fake, sampler, s)
        assert m["filter/step/draw_batch"] == s
        assert {key.split("_")[0] for key in batch.keys} <= set(b)
        assert not fake.prompt_keys()


def test_wrong_draw_batches_raise(monkeypatch):
    fake = fx.FakeTQ().install(monkeypatch)
    sampler = fx.make_sampler()
    old = fx.dispatch(fake, draw_batch=3, global_steps=3, n_groups=24)
    _finish_batch(fake, old, 3, (2, 2))
    with pytest.raises(RuntimeError, match="expects draw batch"):
        sampler.sample(global_steps=6, partition_id="train", batch_size=24)
    with pytest.raises(ValueError, match="train partition"):
        sampler.sample(global_steps=4, partition_id="val", batch_size=24)
    # a short batch
    fake2 = fx.FakeTQ().install(monkeypatch)
    short = fx.dispatch(fake2, draw_batch=0, global_steps=1, n_groups=23)
    _finish_batch(fake2, short, 1, (0, 0))
    with pytest.raises(RuntimeError, match="has 23 prompt groups"):
        fx.make_sampler().sample(global_steps=1, partition_id="train", batch_size=24)
    # prompts without the q38 draw tags (a stock trainer)
    fake3 = fx.FakeTQ().install(monkeypatch)
    u = fx.new_uid()
    fake3.kv_put(u, "train", tag={"is_prompt": True, "status": "finished", "global_steps": 1})
    with pytest.raises(RuntimeError, match="draw_batch/draw_index"):
        fx.make_sampler().sample(global_steps=1, partition_id="train", batch_size=24)
    # a group with a missing trajectory
    fake4 = fx.FakeTQ().install(monkeypatch)
    b = fx.dispatch(fake4, draw_batch=0, global_steps=1, n_groups=24)
    _finish_batch(fake4, b, 1, (0, 0))
    fake4.kv_clear([f"{b[4]}_3_0"], "train")
    with pytest.raises(RuntimeError, match="trajectories"):
        fx.make_sampler().sample(global_steps=1, partition_id="train", batch_size=24)


# ------------------------------------------------------------------------------------------------------------
# T5: failed groups
# ------------------------------------------------------------------------------------------------------------


def test_failure_policy_raise(monkeypatch):
    fake = fx.FakeTQ().install(monkeypatch)
    sampler = fx.make_sampler()
    w = fx.dispatch(fake, draw_batch=0, global_steps=1, n_groups=24)
    _finish_batch(fake, w, 1, (0, 0))
    fx.set_status(fake, w[6], "failure")
    with pytest.raises(RuntimeError, match=f"agent loop raised in 1 prompt group.*{w[6]}"):
        sampler.sample(global_steps=1, partition_id="train", batch_size=24)
    # a failure in the batch still generating raises as soon as it is seen
    fake2 = fx.FakeTQ().install(monkeypatch)
    w2 = fx.dispatch(fake2, draw_batch=0, global_steps=1, n_groups=24)
    b2 = fx.dispatch(fake2, draw_batch=1, global_steps=1, n_groups=24)
    fx.set_status(fake2, b2[0], "failure")
    with pytest.raises(RuntimeError, match=b2[0]):
        fx.make_sampler().sample(global_steps=1, partition_id="train", batch_size=24)
    del w2


def test_failure_policy_skip(monkeypatch):
    fake = fx.FakeTQ().install(monkeypatch)
    sampler = fx.make_sampler(failure_policy="skip")
    w = fx.dispatch(fake, draw_batch=0, global_steps=1, n_groups=24)
    _finish_batch(fake, w, 1, (0, 0), kinds=[MIXED] * 24)
    fx.set_status(fake, w[0], "failure")
    fake.kv_clear([f"{w[0]}_{i}_0" for i in range(3)], "train")  # the failed group has only some trajectories
    batch, m = sampler.sample(global_steps=1, partition_id="train", batch_size=24)
    assert m["filter/step/failed_groups"] == 1 and m["filter/step/candidates_finished"] == 23
    assert w[0] not in sampler.last_step["selected"]
    assert sampler.last_step["selected"] == w[1:13]  # the nominal batch is the first 12 non-failed draws
    assert not {k for k in fake.keys() if k.startswith(w[0])}  # cleared
    assert m["episode/num_episodes"] == 23 * 8
    del batch
    with pytest.raises(ValueError, match="failure_policy"):
        fx.make_sampler(failure_policy="ignore")


# ------------------------------------------------------------------------------------------------------------
# the version tags (MF3), the factors (MF2), the stuck warning (MF5), no filter
# ------------------------------------------------------------------------------------------------------------


def test_version_tags(monkeypatch):
    # within the bounds at step 5 (draw batch 4): (s-1) - min = 1 and spans 2
    fake0 = fx.FakeTQ().install(monkeypatch)
    b0 = fx.dispatch(fake0, draw_batch=4, global_steps=4, n_groups=24)
    _finish_batch(fake0, b0, 5, (3, 4), kinds=[MIXED] * 24)
    _batch, m0 = fx.make_sampler().sample(global_steps=5, partition_id="train", batch_size=24)
    assert m0["vagen/async/version_violations"] == 0
    assert m0["vagen/async/version_lag_worst_max"] == 1 and m0["vagen/async/version_spans_max"] == 2
    # (s-1) - min = 2: strict raises
    fake = fx.FakeTQ().install(monkeypatch)
    b = fx.dispatch(fake, draw_batch=4, global_steps=4, n_groups=24)
    _finish_batch(fake, b, 5, (2, 3), kinds=[MIXED] * 24)
    with pytest.raises(RuntimeError, match="outside the version bounds"):
        fx.make_sampler().sample(global_steps=5, partition_id="train", batch_size=24)
    # spans 3 and (s-1) - min = 2, not strict: counted on the 12 selected groups' episodes
    fake2 = fx.FakeTQ().install(monkeypatch)
    b2 = fx.dispatch(fake2, draw_batch=4, global_steps=4, n_groups=24)
    _finish_batch(fake2, b2, 5, (2, 4), kinds=[MIXED] * 24)
    _batch, m = fx.make_sampler(strict_version_check=False).sample(global_steps=5, partition_id="train",
                                                                   batch_size=24)
    assert m["vagen/async/version_violations"] == 12 * 8
    assert m["vagen/async/version_spans_max"] == 3 and m["vagen/async/version_lag_worst_max"] == 2
    # a missing tag always raises
    fake3 = fx.FakeTQ().install(monkeypatch)
    b3 = fx.dispatch(fake3, draw_batch=0, global_steps=1, n_groups=24)
    _finish_batch(fake3, b3, 1, (None, None), kinds=[MIXED] * 24)
    with pytest.raises(RuntimeError, match="min/max_global_steps"):
        fx.make_sampler(strict_version_check=False).sample(global_steps=1, partition_id="train", batch_size=24)


def test_adv_scale_strict_lookup(monkeypatch):
    fake = fx.FakeTQ().install(monkeypatch)
    sampler = fx.make_sampler()
    w = fx.dispatch(fake, draw_batch=0, global_steps=1, n_groups=24)
    _finish_batch(fake, w, 1, (0, 0))
    batch, _ = sampler.sample(global_steps=1, partition_id="train", batch_size=24)
    real = [k.split("_")[0] for k in batch.keys]
    assert sampler.lookup_adv_scale(real) == [1.0] * len(real)
    with pytest.raises(KeyError, match="without an advantage factor"):
        sampler.lookup_adv_scale(real + [w[1]])  # an all-success group: not selected
    with pytest.raises(KeyError, match="missing from the batch"):
        sampler.lookup_adv_scale(real[8:])
    first = dict(sampler.adv_scale)
    sampler.clear_adv_scale()
    assert sampler.adv_scale == {}
    # the next sample() replaces the dict as a whole
    b1 = fx.dispatch(fake, draw_batch=1, global_steps=2, n_groups=24)
    _finish_batch(fake, b1, 2, (1, 1))
    fake.kv_clear(batch.keys, "train")
    fx.dispatch(fake, draw_batch=2, global_steps=2, n_groups=24)
    sampler.sample(global_steps=2, partition_id="train", batch_size=24)
    assert not (set(sampler.adv_scale) & set(first))


def test_stuck_warning_only_warns(monkeypatch, caplog):
    fake = fx.FakeTQ().install(monkeypatch)
    sampler = fx.make_sampler(stuck_warn={"min_s": 10, "factor": 3, "repeat_s": 20})
    w = fx.dispatch(fake, draw_batch=0, global_steps=1, n_groups=24)
    _finish_batch(fake, w[:20], 1, (0, 0))
    clock = {"t": 0.0, "polls": 0}
    sampler._clock = lambda: clock["t"]

    def poll(partition_id, last):
        clock["t"] += 4.0
        clock["polls"] += 1
        if clock["polls"] == 15:  # 60 s: the last groups finish
            _finish_batch(fake, w[20:], 1, (0, 0))
        return last

    sampler._wait_for_next_poll = poll
    logging.getLogger("vagen.q38.sampler").propagate = True
    with caplog.at_level(logging.WARNING, logger="vagen.q38.sampler"):
        _batch, m = sampler.sample(global_steps=1, partition_id="train", batch_size=24)
    warned = [r for r in caplog.records if "has waited" in r.getMessage()]
    assert len(warned) == 3  # at 12 s, 32 s, 52 s
    assert "4 group(s) not finished" in warned[0].getMessage() and w[20] in warned[0].getMessage()
    assert m["filter/step/stuck_warnings"] == 3 and m["filter/step/wait_s"] == 60.0


def test_filter_disabled_selects_everything(monkeypatch):
    fake = fx.FakeTQ().install(monkeypatch)
    sampler = fx.make_sampler(filter_cfg={"enable": False})
    w = fx.dispatch(fake, draw_batch=0, global_steps=1, n_groups=24)
    _finish_batch(fake, w, 1, (0, 0))
    batch, m = sampler.sample(global_steps=1, partition_id="train", batch_size=24)
    assert sampler.last_step["selected"] == w and len(batch.keys) == 192
    assert set(sampler.adv_scale.values()) == {1.0}
    assert not any(k.startswith(("filter/mixed", "filter/keep")) for k in m)
    assert m["filter/step/kept_rows"] == 192


def test_metrics_are_numbers_and_candidate_metrics_cover_every_episode(monkeypatch):
    fake = fx.FakeTQ().install(monkeypatch)
    sampler = fx.make_sampler()
    w = fx.dispatch(fake, draw_batch=0, global_steps=1, n_groups=24)
    stats = {"step_exceptions": 1, "render_s": [0.5, 0.7], "gen_s": [2.0, 3.0], "endoftext_stops": 0,
             "turns_resumed": 1}
    for j, u in enumerate(w):
        rets, oks = MIXED if j < 4 else ALL_SUCCESS
        fx.finish_group(fake, u, rets, oks, global_steps=1, versions=(0, 0), stats=stats)
    _batch, m = sampler.sample(global_steps=1, partition_id="train", batch_size=24)
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in m.values()), \
        {k: type(v) for k, v in m.items() if not isinstance(v, (int, float)) or isinstance(v, bool)}
    assert m["episode/num_episodes"] == 192 and m["transition/reward_count"] == 192
    assert m["vagen/env/step_exceptions"] == 192 and m["vagen/gen/turns_resumed"] == 192
    assert m["vagen/env/render_s_max"] == 0.7


def test_required_kwargs():
    with pytest.raises(ValueError, match="rollout_n"):
        fx.make_sampler(rollout_n=None)
    with pytest.raises(ValueError, match="model_path"):
        fx.make_sampler(rollout_data_dir="/nonexistent/rollout_data")
    s = fx.make_sampler()
    assert isinstance(s, VagenStepBatchSampler) and s.failure_policy == "raise" and s.strict_version_check


def test_sample_reads_the_rollouts_tensordict_columns(monkeypatch):
    """The columns as the rollout packs them for TransferQueue (list_of_dict_to_tensordict: jagged rm_scores /
    prompts / responses of different lengths, extra_fields as a NonTensorStack of nested dicts)."""
    fake = fx.FakeTQ(as_tensordict=True).install(monkeypatch)
    w = fx.dispatch(fake, draw_batch=0, global_steps=1, n_groups=24)
    for j, u in enumerate(w):
        rets, oks = MIXED if j % 3 == 0 else ALL_SUCCESS
        fx.finish_group(fake, u, rets, oks, global_steps=1, versions=(0, 0),
                        stats={"step_exceptions": 0, "render_s": [0.1, 0.2], "gen_s": [1.0]},
                        prompt_ids=[1, 3] + [4] * (j % 5) + [10, 11, 11, 12],
                        response_ids=[5, 6] + [13] * (j % 4) + [7, 2])
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    probe = list_of_dict_to_tensordict([fake.store["train"][f"{w[j]}_0_0"]["fields"] for j in range(3)])
    assert probe["rm_scores"].is_nested and probe["prompts"].is_nested  # jagged, unpadded rows
    assert not isinstance(probe["extra_fields"], torch.Tensor)  # a tensordict LinkedList / NonTensorStack
    with fx.work_dir("q38-td") as d:
        sampler = fx.make_sampler(rollout_data_dir=d, model_path="unused-fake-tokenizer")
        sampler._tokenizer = fx.FakeTokenizer()
        batch, m = sampler.sample(global_steps=1, partition_id="train", batch_size=24)
        rows = [__import__("json").loads(x) for x in open(os.path.join(d, "1.jsonl"), encoding="utf-8")]
    want = [w[j] for j in range(24) if j % 3 == 0]
    assert sampler.last_step["selected"] == want and len(batch.keys) == 8 * 8
    assert m["episode/num_episodes"] == 192 and m["vagen/env/render_s_max"] == 0.2
    assert [r["score"] for r in rows[:8]] == [float(np.float32(x)) for x in MIXED[0]]
    assert rows[8 * 2]["input"] == "<|im_start|>user\nhihi<image>"  # draw 2: two "hi", no padding stripped
    assert rows[8 * 3]["output"] == "<action>move_forward|\n\n\n</action><|im_end|>"


# ------------------------------------------------------------------------------------------------------------
# one sample() through a real TransferQueue (types of the columns, merged tags, clears)
# ------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_tq():
    import signal

    ray = pytest.importorskip("ray")
    import transfer_queue as tq
    from omegaconf import OmegaConf

    import shutil

    # Ray's session directory, short (AF_UNIX socket paths are at most 107 bytes) and outside /tmp
    tmp = os.environ.get("Q38_RAY_TMP", "/mnt/umm/users/yinbaiqiao/q38/rt_b3")
    try:
        os.makedirs(tmp, exist_ok=True)
    except OSError as e:
        pytest.skip(f"no Ray temp directory {tmp}: {e!r}")

    def _timeout(*_):
        raise TimeoutError("Ray / TransferQueue start-up took more than 300 s")

    old = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(300)
    try:
        ray.init(num_cpus=2, include_dashboard=False, object_store_memory=200 * 1024 * 1024, _temp_dir=tmp,
                 log_to_driver=False)
        tq.init(OmegaConf.create({"backend": {"SimpleStorage": {"num_data_storage_units": 1}}}))
    except Exception as e:  # noqa: BLE001
        signal.alarm(0)
        try:
            ray.shutdown()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            pytest.skip(f"no local Ray / TransferQueue: {e!r}")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    yield tq
    tq.close()
    ray.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)


def test_real_transfer_queue_sample(real_tq):
    tq = real_tq
    from verl.utils import tensordict_utils as tu
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    sampler = fx.make_sampler()
    uids = [fx.new_uid() for _ in range(24)]
    for j in random.Random(1).sample(range(24), 24):  # prompt keys as TB:1566-1577 writes them
        batch = tu.get_tensordict({"uid": [uids[j]], "index": torch.tensor([j])})
        tq.kv_batch_put(keys=[uids[j]], partition_id="train", fields=batch,
                        tags=[{"is_prompt": True, "status": "pending", "global_steps": 1, "draw_batch": 0,
                               "draw_index": j}])
    for j, u in enumerate(uids):
        tq.kv_put(key=u, partition_id="train", tag={"status": "running"})  # TQW:113
        rets, oks = MIXED if j % 3 == 0 else ALL_SUCCESS
        keys, fields, tags = [], [], []
        for i in range(8):
            resp = torch.tensor([5, 6, 7, 2] + [13] * (i % 3), dtype=torch.int64)
            rm = torch.zeros(len(resp), dtype=torch.float32)
            rm[-1] = rets[i]
            fields.append({
                "prompts": torch.tensor([1, 3, 10, 11, 11, 12, 4], dtype=torch.int64), "responses": resp,
                "response_mask": torch.ones_like(resp), "rm_scores": rm,
                "extra_fields": {"min_global_steps": 0, "max_global_steps": 0,
                                 "reward_extra_info": fx.reward_info(i, bool(oks[i]), f"task{j}"),
                                 "vagen_stats": {"step_exceptions": 0, "render_s": [0.1], "gen_s": [1.0]}},
            })
            keys.append(f"{u}_{i}_0")
            tags.append({"status": "success", "global_steps": 1, "min_global_steps": 0, "max_global_steps": 0})
        tq.kv_batch_put(keys=keys, partition_id="train", fields=list_of_dict_to_tensordict(fields), tags=tags)
        tq.kv_put(key=u, partition_id="train", tag={"status": "finished"})  # TQW:148
    with fx.work_dir("q38-realtq") as d:
        sampler.rollout_data_dir = d
        sampler._tokenizer = fx.FakeTokenizer()
        batch, m = sampler.sample(global_steps=1, partition_id="train", batch_size=24)
        rows = [__import__("json").loads(x) for x in open(os.path.join(d, "1.jsonl"), encoding="utf-8")]
    want = [uids[j] for j in range(24) if j % 3 == 0]
    assert sampler.last_step["selected"] == want
    assert len(batch.keys) == len(want) * 8
    assert m["episode/num_episodes"] == 192 and m["filter/mixed/kept"] == len(want)
    assert len(rows) == 192 and rows[0]["input"] == "<|im_start|>user\n<image>hi"
    listed = tq.kv_list(partition_id="train").get("train", {})
    assert not any(k.split("_")[0] in set(uids) - set(want) for k in listed)  # the dropped groups are gone
    assert not any(k in listed for k in want)  # the selected prompt keys are cleared (RB:405)
    assert all(k in listed for k in batch.keys)  # the selected trajectories stay for the update
    tq.kv_clear(keys=list(listed), partition_id="train")
