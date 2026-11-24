"""CPU tests of tools.head_loss_probe (the step replay behind STAGE3_REPORT section 8): the rebuilt data order and its
check, the synthetic cross entropy, both layouts' losses from the same hidden states, the step means, the modes on the
tiny untied Qwen3.5."""
from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn.functional as F

from vagen.q38s4 import lf_nfp as L
from vagen.q38s4.tests.tiny import build, cpu_gdn, needs_models, processor, video_batch  # noqa: F401
from vagen.q38s4.tools import head_loss_probe as H

torch.set_num_threads(4)


# ------------------------------------------------------------------------------------------------ data order
def test_data_order_is_the_seeded_randperm_with_the_last_batch_filled():
    st = H.data_order(10, 4, 42, 2)
    assert len(st) == 6 and all(len(b) == 4 for b in st)
    for e in range(2):
        g = torch.Generator()
        g.manual_seed(42 + e)
        perm = torch.randperm(10, generator=g).tolist()
        assert st[3 * e:3 * e + 3] == [perm[0:4], perm[4:8], perm[8:10] + perm[0:2]]
    assert H.data_order(8, 4, 0, 1) == [b for b in H.data_order(8, 4, 0, 2)[:2]]


def test_check_order_counts_the_logged_steps_it_covers():
    st = H.data_order(10, 4, 0, 1)
    nfp = [i % 3 == 0 for i in range(10)]
    logged = {s: {"nfp_videos": sum(nfp[i] for i in st[s - 1])} for s in (1, 2, 3)}
    assert H.check_order(st, nfp, logged) == {"steps": 3, "match": 3, "mismatch": []}
    logged[2]["nfp_videos"] += 1
    logged[5] = {"nfp_videos": 0}  # past the rebuilt steps
    logged[4] = {"loss": 1.0}  # no nfp_videos logged
    assert H.check_order(st, nfp, logged) == {"steps": 3, "match": 2, "mismatch": [2]}


def test_log_steps_keeps_the_training_step_entries(tmp_path):
    hist = [{"step": 1, "loss": 1.0, "text_loss": 0.5}, {"step": 2, "loss": 0.9, "text_loss": 0.4},
            {"step": 2, "eval_loss": 0.3}, {"step": 2, "train_runtime": 3.0, "loss": 1.0}]
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "trainer_state.json").write_text(json.dumps({"log_history": hist}))
    assert H.log_steps(tmp_path) == {1: hist[0], 2: hist[1]}


# ------------------------------------------------------------------------------------------------ synthetic CE
def test_synthetic_ce_small():
    r = H.synthetic_ce(64, 50, torch.device("cpu"))
    assert r["elements"] == 3200 and not r["over_2_31"] and r["rows_past_2_31"] == []
    assert r["rows"] == [1, 32, 60, 61, 62, 63]
    assert r["full_minus_rows_max_abs"] < 1e-6 and r["rows_minus_ref_max_abs"] < 1e-4
    assert abs(r["full_mean_minus_rows_mean"]) < 1e-6
    assert len(r["loss"]) == 6 and min(r["loss"][0::2]) >= 0.0 and max(r["loss"][0::2]) < 1e-3  # confident rows


# ------------------------------------------------------------------------------------------------ replay
def _case(T=20, V=37, Hd=16, seed=0):
    torch.manual_seed(seed)
    h = torch.randn(1, T, Hd)
    W = torch.randn(V, Hd) * 0.3
    labels = torch.full((1, T), -100)
    labels[0, [0, 5, 6, 12, 19]] = torch.tensor([2, 3, 7, 1, 30])  # labels[0] is never a causal target
    return h, W, labels, V


def test_replay_fp32_both_layouts_are_the_models_loss():
    from transformers.loss.loss_utils import ForCausalLMLoss

    h, W, labels, V = _case()
    r = H.replay(h, W, labels, V, train=False)
    assert r["T"] == 20 and r["K"] == 4 and r["dtype_logits"] == "torch.float32"
    assert r["all"] == float(ForCausalLMLoss(F.linear(h, W), labels, V))
    lay = L.at_targets({"labels": labels})
    assert r["targets"] == float(ForCausalLMLoss(F.linear(h[:, lay["logits_to_keep"]], W), lay["labels"], V))
    for k in ("targets", "all_halves", "all_selected"):
        assert abs(r[k] - r["all"]) < 1e-6
    assert len(r["tokens_all"]) == len(r["tokens_targets"]) == 4
    assert max(abs(a - b) for a, b in zip(r["tokens_all"], r["tokens_targets"])) < 1e-6
    assert 0.0 <= r["logits_unequal_frac"] <= 1.0 and isinstance(r["head_equal"], bool)


def test_replay_train_runs_under_autocast_with_grad():
    h, W, labels, V = _case(seed=1)
    r = H.replay(h, W, labels, V, train=True)
    assert r["dtype_logits"] == "torch.bfloat16"
    ref = H.replay(h, W, labels, V, train=False)
    assert all(math.isfinite(r[k]) for k in ("all", "targets", "all_halves", "all_selected"))
    assert abs(r["targets"] / ref["targets"] - 1) < 5e-2 and abs(r["all"] / ref["all"] - 1) < 5e-2
    assert abs(r["all_halves"] - r["all"]) < 1e-5 and abs(r["all_selected"] - r["all"]) < 1e-5
    assert not h.requires_grad  # the caller's tensor is untouched


def test_step_losses_mean_of_row_means():
    recs = {1: {"all": 1.0, "targets": 2.0}, 2: {"all": 3.0, "targets": 4.0}, 3: {"all": 5.0}}
    out = H.step_losses(recs, {1: [1, 2], 2: [2, 3], 3: [4], 4: []}, keys=("all", "targets"))
    assert out == {1: {"all": 2.0, "targets": 3.0}, 2: {"all": 4.0}}


def test_main_refuses_an_existing_out_and_bad_modes(tmp_path):
    with pytest.raises(SystemExit):
        H.main(["--run", str(tmp_path), "--out", str(tmp_path), "--check-run", str(tmp_path)])
    with pytest.raises(SystemExit):
        H.main(["--run", str(tmp_path), "--out", str(tmp_path / "o"), "--check-run", str(tmp_path),
                "--modes", "train,infer"])
    assert not (tmp_path / "o").exists()


# ------------------------------------------------------------------------------------------------ the tiny model
@pytest.fixture(scope="module")
def proc():
    return processor()


@needs_models
def test_capture_infer_then_train_on_the_tiny_model(cpu_gdn, proc):
    """Infer: the captured head output is F.linear of the captured h, the own loss is the replayed "targets" loss
    bit for bit (fp32 CPU), the caller's batch unchanged.  Train: every parameter and buffer bf16, train mode,
    use_cache off, the own loss close to infer's."""
    model = build(3, untied=True).eval()
    batch = video_batch(proc, (6,), answer="B because the chair is left of the table")
    labels = batch["labels"]
    c = H.capture(model, batch, "infer")
    assert "logits_to_keep" not in batch and batch["labels"] is labels
    assert c["h"].device.type == "cpu" and c["h"].shape[:2] == labels.shape
    c["labels"] = labels
    W = model.lm_head.weight.detach()
    assert H.head_check(c, W, "cpu", train=False)
    r = H.replay(c["h"], W, labels, model.config.text_config.vocab_size, train=False)
    assert r["targets"] == c["own_targets"]
    assert abs(r["all"] - r["targets"]) < 1e-5
    d = H.prepare_train(model)
    assert d["fp32_params"] > 0 and d["fp32_params_after"] == 0 and d["fp32_buffers_after"] == 0
    assert model.training and model.config.text_config.use_cache is False
    t = H.capture(model, batch, "train")
    assert t["h"].dtype == torch.bfloat16 and math.isfinite(t["own_targets"])
    assert abs(t["own_targets"] / c["own_targets"] - 1) < 0.1
    t["labels"] = labels
    assert H.head_check(t, model.lm_head.weight.detach(), "cpu", train=True)
    H.release(model)
    assert all(p.numel() == 0 for p in model.parameters())
