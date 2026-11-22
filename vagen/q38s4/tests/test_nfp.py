"""CPU tests of the NFP head, targets, alignment, loss, freezing and the weight-0 equality (vagen.q38s4.nfp, lf_nfp).

Run (login node, 4 threads):
  cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-q38-s4 && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD nice -n 19 \
    /mnt/umm/users/yinbaiqiao/q38/envs/verl/bin/python -B -m pytest vagen/q38s4/tests -q -p no:cacheprovider
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vagen.q38s4 import lf_nfp as L
from vagen.q38s4 import nfp as N
from vagen.q38s4.tests.tiny import (build, cpu_gdn, freeze_encoder, needs_models, processor, unit_groups,  # noqa: F401
                                    video_batch)

torch.set_num_threads(4)


# ------------------------------------------------------------------------------------------------ pure functions
def test_head_shapes_and_init():
    torch.manual_seed(0)
    head = N.NFPHead(48, 20)
    assert head.fc1.weight.shape == (48, 48) and head.fc2.weight.shape == (20, 48)
    assert isinstance(head.act, torch.nn.GELU)
    for lin in (head.fc1, head.fc2):
        assert lin.weight.abs().max() <= 0.04 + 1e-7 and torch.all(lin.bias == 0)
        assert 0.012 < lin.weight.std() < 0.02  # a normal cut at 2 std has std 0.88 * 0.02
    assert head(torch.randn(3, 7, 48)).shape == (3, 7, 20)
    assert N.NFPHead(48, 20, hidden_dim=16).fc1.weight.shape == (16, 48)


def test_merge_targets_is_the_window_mean():
    x = torch.arange(8 * 3, dtype=torch.float32).view(8, 3)
    want = torch.stack([x[0:4].mean(0), x[4:8].mean(0)])
    assert torch.equal(N.merge_targets(x), want)
    with pytest.raises(ValueError):
        N.merge_targets(torch.zeros(6, 3))


def test_unit_pairs_on_a_synthetic_batch():
    # video 0: 3 units x 4 tokens (NFP); video 1: 2 units x 2 tokens (QA); video 2: 2 units x 2 tokens (NFP);
    # video 3: 1 unit (NFP, nothing to predict)
    grid = torch.tensor([[3, 4, 4], [2, 2, 4], [2, 4, 2], [1, 4, 4]])
    src, dst = N.unit_pairs(grid, [True, False, True, True])
    assert src.tolist() == list(range(0, 8)) + [16, 17]
    assert dst.tolist() == list(range(4, 12)) + [18, 19]
    assert N.count_pairs(grid, torch.tensor([True, False, True, True])) == 10
    assert N.unit_pairs(grid, [False] * 4)[0].numel() == 0
    with pytest.raises(ValueError):
        N.unit_pairs(grid, [True])


def test_error_sums_match_the_official_formulas():
    torch.manual_seed(1)
    p, y = torch.randn(5, 7), torch.randn(5, 7)
    mse, cos = N.error_sums(p, y)
    assert torch.allclose(mse / 5, F.mse_loss(p, y, reduction="none").mean(-1).mean())
    assert torch.allclose(cos / 5, (1 - F.cosine_similarity(p, y, dim=-1)).mean())


def test_scale_gives_the_global_token_mean():
    # 2 ranks x 3 micro-batches with different NFP token counts; the trainer divides by GA, FSDP averages ranks
    torch.manual_seed(2)
    counts = [[5, 0, 9], [1, 7, 3]]
    sums = [[float(torch.rand(1)) * c for c in r] for r in counts]
    total = sum(map(sum, counts))
    world, ga = 2, 3
    scale = N.scale_for_global_mean(world, ga, total)
    grad_like = sum(sum(scale * s / ga for s in r) for r in sums) / world
    assert abs(grad_like - sum(map(sum, sums)) / total) < 1e-12
    assert N.scale_for_global_mean(2, 3, 0) == 0.0


def test_summarize():
    acc = dict.fromkeys(N.ACC_KEYS, 0.0)
    acc.update(ce_sum=3.0, ce_n=2, pairs=10, videos=2, head_pairs=10, mse_sum=5.0, cos_sum=2.0, copy_mse_sum=8.0,
               copy_cos_sum=1.0)
    s = N.summarize(acc)
    assert s == {"text_loss": 1.5, "nfp_pairs": 10, "nfp_videos": 2, "nfp_mse": 0.5, "nfp_cos": 0.2,
                 "copy_mse": 0.8, "copy_cos": 0.1}
    assert N.summarize(dict.fromkeys(N.ACC_KEYS, 0.0)) == {"nfp_pairs": 0, "nfp_videos": 0}


def test_mark_nfp_masks_only_nfp_samples():
    feats = [{"labels": [-100, 5, 6], "videos": ["/v/a.mp4|s4|nfp:u=0.500000"]},
             {"labels": [-100, 7], "videos": ["/v/a.mp4|s4|qa"]},
             {"labels": [-100, 8], "images": ["/i.jpg"]}]
    assert L.mark_nfp(feats) == [True, False]
    assert feats[0]["labels"] == [-100] * 3 and feats[1]["labels"] == [-100, 7] and feats[2]["labels"] == [-100, 8]
    with pytest.raises(ValueError):
        L.mark_nfp([{"labels": [1], "videos": ["/a|s4|qa", "/b|s4|nfp:u=0.100000"]}])


def test_freeze_check_due():
    args = SimpleNamespace(save_steps=500, save_strategy="steps")
    assert L.freeze_check_due(None, 0, args)
    assert not L.freeze_check_due(0, 0, args) and not L.freeze_check_due(0, 499, args)
    assert L.freeze_check_due(0, 500, args) and not L.freeze_check_due(500, 500, args)
    assert not L.freeze_check_due(0, 500, SimpleNamespace(save_steps=500, save_strategy="no"))


def test_settings_file(tmp_path):
    (tmp_path / "q38s4.json").write_text('{"nfp_mse": 0.1, "nfp_cos": 0.1}')
    (tmp_path / "train.yaml").write_text("")
    s = L.settings_for(["x", str(tmp_path / "train.yaml")])
    assert s.head_on and s.nfp_mse == 0.1
    (tmp_path / "q38s4.json").write_text('{"nfp_mse": 0, "nfp_cos": 0}')
    assert not L.settings_for(["x", str(tmp_path / "train.yaml")]).head_on
    (tmp_path / "q38s4.json").write_text('{"nfp_mse": 0, "nfp_cos": 0, "nfp_kl": 1}')
    with pytest.raises(ValueError):
        L.settings_for(["x", str(tmp_path / "train.yaml")])
    (tmp_path / "q38s4.json").unlink()
    with pytest.raises(SystemExit):
        L.settings_for(["x", str(tmp_path / "train.yaml")])


# ------------------------------------------------------------------------------------------------ tiny model
@pytest.fixture(scope="module")
def proc():
    return processor()


def _base(m, inp):
    out = m(**inp)
    return out.loss, out


def _expected_by_layout(model, batch, head):
    """MSE / cos sums recomputed from the token layout: hidden of unit t's j-th token against the window mean of
    the target block's patches of unit t+1, with the encoder run again on its own."""
    blocks_out = {}
    h = model.model.visual.blocks[-2].register_forward_hook(lambda m, a, o: blocks_out.setdefault("x", o.detach()))
    with torch.no_grad():
        out = model(**{k: v for k, v in batch.items() if k != "labels"}, output_hidden_states=True)
    h.remove()
    hidden = model.model.language_model.norm(out.hidden_states[-1])  # last layer's output through the final norm
    patches = blocks_out["x"].float()
    groups = unit_groups(batch["input_ids"], model.config.video_token_id)
    t, gh, gw = batch["video_grid_thw"][0].tolist()
    n = gh * gw // 4
    assert len(groups) == t and all(len(g) == n for g in groups)
    mse = cos = 0.0
    with torch.no_grad():
        for u in range(t - 1):
            for j in range(n):
                tgt = patches[(u + 1) * n * 4 + 4 * j:(u + 1) * n * 4 + 4 * j + 4].mean(0)
                pred = head(hidden[0, groups[u][j]]).float()
                mse += float((pred - tgt).pow(2).mean())
                cos += float(1 - F.cosine_similarity(pred, tgt, dim=0))
    return mse, cos, (t - 1) * n


@needs_models
def test_tap_alignment_against_the_token_layout(cpu_gdn, proc):
    model = build(0)
    head = N.attach_head(model)
    assert head.fc1.weight.shape == (64, 64) and head.fc2.weight.shape == (32, 64)
    batch = video_batch(proc, (8,))
    assert batch["video_grid_thw"].tolist() == [[4, 4, 4]]
    tap = N.NFPTap(model, head)
    tap.arm(torch.tensor([True]))
    with torch.no_grad():
        model(**batch)
    res = tap.collect()
    mse, cos, n = _expected_by_layout(model, batch, head)
    assert res.n_pairs == n == 12 and res.n_videos == 1
    assert abs(float(res.mse_sum) - mse) < 1e-4 * max(1, mse) and abs(float(res.cos_sum) - cos) < 1e-4 * max(1, cos)
    # QA video: nothing
    tap.arm(torch.tensor([False]))
    with torch.no_grad():
        model(**batch)
    res = tap.collect()
    assert res.n_pairs == 0 and res.mse_sum is None
    tap.remove()


@needs_models
def test_two_videos_only_the_nfp_one_counts(cpu_gdn, proc):
    model = build(0)
    head = N.attach_head(model)
    batch = video_batch(proc, (4, 6))
    assert batch["video_grid_thw"].tolist() == [[2, 4, 4], [3, 4, 4]]
    tap = N.NFPTap(model, head)
    for flags, want in (([True, False], 4), ([False, True], 8), ([True, True], 12)):
        tap.arm(torch.tensor(flags))
        with torch.no_grad():
            model(**batch)
        assert tap.collect().n_pairs == want


@needs_models
def test_copy_baseline(cpu_gdn, proc):
    model = build(0)
    batch = video_batch(proc, (6,))
    got = {}
    h = model.model.visual.blocks[-2].register_forward_hook(lambda m, a, o: got.setdefault("x", o.detach()))
    tap = N.NFPTap(model, None)
    tap.arm(torch.tensor([True]))
    with torch.no_grad():
        model(**batch)
    res = tap.collect()
    h.remove()
    tg = N.merge_targets(got["x"])  # 3 units x 4 tokens
    mse, cos = N.error_sums(tg[:8], tg[4:])
    assert res.n_pairs == 8 and res.mse_sum is None
    assert abs(res.copy_mse_sum - float(mse)) < 1e-6 and abs(res.copy_cos_sum - float(cos)) < 1e-6


@needs_models
def test_freezing_and_gradients(cpu_gdn, proc):
    model = build(0)
    freeze_encoder(model)
    N.attach_head(model)
    tap = N.NFPTap(model, model.nfp_head)
    acc = dict.fromkeys(N.ACC_KEYS, 0.0)
    batch = video_batch(proc, (6,))
    batch["nfp_video"] = torch.tensor([True])
    batch["labels"][:] = -100  # an NFP sample: no text loss
    s = L.S4Settings(0.1, 0.1)
    loss, _ = L.s4_loss(model, batch, tap, s, _base, N.scale_for_global_mean(1, 1, 8), acc)
    loss.backward()
    named = dict(model.named_parameters())
    enc = [n for n in named if any(k in n for k in ("visual.blocks", "visual.patch_embed", "visual.pos_embed"))]
    assert enc and all(named[n].grad is None for n in enc)
    assert named["nfp_head.fc2.weight"].grad.abs().sum() > 0
    assert named["model.visual.merger.linear_fc1.weight"].grad.abs().sum() > 0  # the merger trains
    assert named["model.language_model.layers.0.mlp.down_proj.weight"].grad.abs().sum() > 0
    if model.lm_head.weight is not model.get_input_embeddings().weight:  # tied in the 4B config used here
        assert model.lm_head.weight.grad.abs().sum() == 0  # no text loss on an NFP sample
    rep = L.freeze_report(model)
    assert rep["ok"] and rep["encoder"] == len(enc) and rep["encoder_with_grad"] == 0 and rep["head_trainable"] == 4
    assert rep["merger_trainable"] > 0
    assert acc["ce_n"] == 0 and acc["pairs"] == 8 and acc["head_pairs"] == 8
    assert abs(float(loss.detach()) - 0.1 * (acc["mse_sum"] + acc["cos_sum"]) / 8) < 1e-5
    # a frozen parameter with a gradient is caught
    p0 = next(model.model.visual.blocks[0].parameters())
    p0.grad = torch.zeros_like(p0)
    assert not L.freeze_report(model)["ok"]


@needs_models
def test_weights_zero_is_the_plain_sft_loss_bit_for_bit(cpu_gdn, proc, tmp_path):
    from transformers import Trainer, TrainingArguments

    batch = video_batch(proc, (6,))
    plain = build(3)
    s4 = copy.deepcopy(plain)
    for m in (plain, s4):
        freeze_encoder(m)
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[], per_device_train_batch_size=1)
    tr_plain = Trainer(model=plain, args=args)
    loss_plain = tr_plain.compute_loss(plain, dict(batch))
    loss_plain.backward()

    tr_s4 = Trainer(model=s4, args=args)
    tap = N.NFPTap(s4, None)
    acc = dict.fromkeys(N.ACC_KEYS, 0.0)
    inp = dict(batch, nfp_video=torch.tensor([False]))
    loss_s4, _ = L.s4_loss(s4, inp, tap, L.S4Settings(0.0, 0.0),
                           lambda m, i: tr_s4.compute_loss(m, i, return_outputs=True), 0.0, acc)
    loss_s4.backward()
    assert torch.equal(loss_plain, loss_s4)
    gp, gs = dict(plain.named_parameters()), dict(s4.named_parameters())
    assert gp.keys() == gs.keys()
    for n in gp:
        a, b = gp[n].grad, gs[n].grad
        assert (a is None) == (b is None), n
        if a is not None:
            assert torch.equal(a, b), n
    # the same with the copy baseline on (an NFP-flagged video in a text sample is not a real case, but the tap
    # must not touch the loss)
    s4.zero_grad(set_to_none=True)
    inp = dict(batch, nfp_video=torch.tensor([True]))
    loss_c, _ = L.s4_loss(s4, inp, tap, L.S4Settings(0.0, 0.0),
                          lambda m, i: tr_s4.compute_loss(m, i, return_outputs=True), 0.0, acc)
    assert torch.equal(loss_c, loss_plain) and acc["pairs"] == 8 and acc["copy_mse_sum"] > 0


@needs_models
def test_nfp_only_micro_batch_without_head_is_zero_not_nan(cpu_gdn, proc):
    model = build(0)
    batch = video_batch(proc, (4,))
    batch["labels"][:] = -100
    batch["nfp_video"] = torch.tensor([True])
    acc = dict.fromkeys(N.ACC_KEYS, 0.0)
    loss, _ = L.s4_loss(model, batch, N.NFPTap(model, None), L.S4Settings(), _base, 0.0, acc)
    assert float(loss.detach()) == 0.0 and loss.requires_grad
    loss.backward()
    assert torch.isfinite(model.lm_head.weight.grad).all()
