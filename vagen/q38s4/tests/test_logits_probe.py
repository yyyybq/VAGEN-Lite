"""CPU tests of tools.logits_probe (the LM-head probe behind STAGE3_REPORT section 8): the at_targets layout read
like the model's loss, the variants, the per-row checks on the tiny untied Qwen3.5, the subset summary."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vagen.q38s4 import lf_nfp as L
from vagen.q38s4.tests.tiny import build, cpu_gdn, needs_models, processor, video_batch  # noqa: F401
from vagen.q38s4.tools import logits_probe as P

torch.set_num_threads(4)


# ------------------------------------------------------------------------------------------------ pure functions
def test_token_ce_reads_at_targets_like_the_models_loss():
    """token_ce on at_targets' layout gives the targets' cross entropies, whose mean is transformers' causal-LM loss
    on the full sequence (labels shifted inside), one value per target in order."""
    from transformers.loss.loss_utils import ForCausalLMLoss

    torch.manual_seed(0)
    V = 11
    labels = torch.tensor([[-100, -100, 4, 6, -100, 2, -100, 9]])
    logits = torch.randn(1, labels.shape[1], V)
    lay = L.at_targets({"labels": labels, "input_ids": labels.clamp_min(0)})
    keep, new = lay["logits_to_keep"], lay["labels"]
    assert keep.tolist() == [1, 2, 4, 6, 6]
    ce = P.token_ce(logits[:, keep], new)
    want = F.cross_entropy(logits[0, [1, 2, 4, 6]], torch.tensor([4, 6, 2, 9]), reduction="none")
    assert torch.equal(ce, want)
    full = ForCausalLMLoss(logits, labels, V)
    assert abs(float(full) - float(ce.mean())) < 1e-6
    assert P.token_ce(logits[:, keep].double(), new).dtype == torch.float64


def test_target_logits_variants_and_flag_restored():
    torch.manual_seed(1)
    h, W = torch.randn(1, 6, 8), torch.randn(70, 8)
    keep = torch.tensor([1, 3, 3])
    old = getattr(torch.backends.cuda.matmul, P.FLAG)
    lg = P.target_logits(h, W, keep, flags=(True, False), chunk=32)
    assert getattr(torch.backends.cuda.matmul, P.FLAG) == old
    assert set(lg) == {"all", "targets", "all_nored", "targets_nored", "ref"}
    assert torch.equal(lg["all"], F.linear(h, W)[:, keep])
    assert torch.equal(lg["targets"], F.linear(h[:, keep], W))
    assert lg["ref"].dtype == torch.float64 and lg["ref"].shape == (1, 3, 70)
    assert torch.allclose(lg["ref"], (h[:, keep].double() @ W.double().T))
    d = P.logit_diff(lg["all"], lg["ref"])
    assert d["max_abs"] < 1e-4 and 0.0 <= d["unequal_frac"] <= 1.0
    same = P.logit_diff(lg["all"], lg["all"].clone())
    assert same == {"unequal_frac": 0.0, "max_abs": 0.0, "mean_abs": 0.0}


def test_select_rows_skips_nfp_and_rows_without_targets():
    rows = [{"labels": [-100, 5, 6], "videos": ["/v/a.mp4|s4|qa"], "images": None},
            {"labels": [-100, -100, -100], "videos": ["/v/b.mp4|s4|nfp:u=0.250000"], "images": None},
            {"labels": [-100, -100, 7], "videos": None, "images": ["/i/c.jpg"]},
            {"labels": [7, -100, -100], "videos": None, "images": ["/i/d.jpg"]},  # only labels[0]: no causal target
            {"labels": [-100, 3, -100], "videos": ["/v/e.mp4|s4|nfp:u=0.500000"], "images": None}]
    assert P.select_rows(rows) == [0, 2]
    assert P.select_rows(rows, limit=1) == [0]
    assert [P.media_of(r) for r in rows[:3]] == ["video", "video", "image"]


def _row(ce_t, scale=1.0, T=50):
    ce_t = list(map(float, ce_t))
    ce = {"targets": ce_t, "all": [x * scale for x in ce_t], "ref": ce_t}
    names = ("all", "targets", "ref")
    return {"T": T, "K": len(ce_t), "ce": ce, "ce_sum": {k: float(np.sum(v)) for k, v in ce.items()},
            "head_equals_targets": True, "own_targets_minus_ce_mean": 0.0, "own_all_loss": None,
            "own_all_minus_ce_mean": None, "h_equal_second_forward": None,
            "logits": {f"{a}~{b}": {"unequal_frac": 0.0, "max_abs": 0.0, "mean_abs": 0.0}
                       for a, b in P.PAIRS if a in names and b in names}}


def test_summarize_subsets():
    rng = np.random.default_rng(0)
    rows = [_row(rng.uniform(0.1, 2.0, size=k), scale=1.001) for k in rng.integers(1, 6, size=12)]
    s = P.summarize(rows, subset=4, draws=500, observed=5e-4)
    p = s["pairs"]["all~targets"]
    assert s["rows"] == 12 and s["tokens"] == sum(r["K"] for r in rows)
    assert abs(p["rel_total"] - 1e-3) < 1e-12 and p["tokens_unequal"] == s["tokens"]
    assert abs(p["subset"]["mean"] - 1e-3) < 1e-12 and p["subset"]["std"] < 1e-12
    assert p["subset"]["share_abs_ge_observed"] == 1.0
    assert P.summarize(rows, 4, 500, observed=2e-3)["pairs"]["all~targets"]["subset"]["share_abs_ge_observed"] == 0.0
    t = s["pairs"]["targets~ref"]
    assert t["rel_total"] == 0.0 and t["tokens_unequal"] == 0
    assert "all_nored~targets_nored" not in s["pairs"]  # variants not computed are left out
    assert s["checks"]["head_equals_targets"] == 12 and s["checks"]["own_all_max_abs"] is None
    # fewer rows than a subset: no subset statistics
    assert "subset" not in P.summarize(rows[:3], subset=4, draws=10)["pairs"]["all~targets"]


def test_main_refuses_an_existing_out(tmp_path):
    with pytest.raises(SystemExit):
        P.main(["--run", str(tmp_path), "--out", str(tmp_path)])


# ------------------------------------------------------------------------------------------------ the tiny model
@pytest.fixture(scope="module")
def proc():
    return processor()


@needs_models
def test_probe_row_reproduces_the_models_losses(cpu_gdn, proc):
    """On the tiny untied model (fp32, CPU): the LM head's output is "targets" bit for bit, the model's own losses
    (at_targets layout; full labels) are the means of "targets" / "all", the second forward's hidden states equal the
    first's, the variants agree to fp32 rounding, and the caller's batch is unchanged."""
    model = build(3, untied=True).eval()
    batch = video_batch(proc, (6,), answer="B because the chair is left of the table")
    labels = batch["labels"]
    n_targets = int((labels[:, 1:] != -100).sum())
    with torch.no_grad():
        rec = P.probe_row(model, batch, flags=(True,), own_all_max=10 ** 6)
    assert "logits_to_keep" not in batch and batch["labels"] is labels
    assert rec["K"] == n_targets == len(rec["ce"]["targets"]) and rec["T"] == labels.shape[1]
    assert set(rec["ce"]) == {"all", "targets", "ref"}
    assert rec["head_equals_targets"] is True
    assert abs(rec["own_targets_minus_ce_mean"]) < 1e-6 and abs(rec["own_all_minus_ce_mean"]) < 1e-6
    assert rec["h_equal_second_forward"] is True
    assert np.allclose(rec["ce"]["all"], rec["ce"]["targets"], atol=1e-5)
    assert np.allclose(rec["ce"]["ref"], rec["ce"]["targets"], atol=1e-4)
    assert rec["logits"]["all~targets"]["max_abs"] < 1e-4
    # the model's own loss on the at_targets layout is what lf_nfp's "targets" path trains on
    with torch.no_grad():
        own = model(**L.at_targets(batch)).loss
    assert float(own) == rec["own_targets_loss"]
    # a long row skips the full-label forward
    with torch.no_grad():
        short = P.probe_row(model, batch, flags=(True,), own_all_max=8, ref=False)
    assert short["own_all_loss"] is None and set(short["ce"]) == {"all", "targets"}
    with torch.no_grad():
        first = P.first_calls(model, batch, n=2)
    assert first["h_equal_last"] == [True, True] and first["loss"][0] == first["loss"][1]
