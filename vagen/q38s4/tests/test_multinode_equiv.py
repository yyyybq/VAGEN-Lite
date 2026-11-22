"""CPU tests of the step-by-step comparison of vagen.q38s4.tools.multinode_equiv."""
from __future__ import annotations

import copy

from vagen.q38s4.tools.multinode_equiv import compare


def hist(n=4):
    return [{"step": s, "loss": 2.0 - 0.1 * s, "grad_norm": 1.5, "text_loss": 2.1, "nfp_mse": 0.5, "nfp_pairs": 40}
            for s in range(1, n + 1)]


def test_same_runs_pass():
    r = compare(hist(), hist(), 1e-2)
    assert not r["failures"] and r["keys"]["loss"]["max_rel"] == 0.0 and "nfp_cos" not in r["keys"]


def test_small_noise_passes_large_fails():
    b = copy.deepcopy(hist())
    b[2]["grad_norm"] *= 1.001
    assert not compare(hist(), b, 1e-2)["failures"]
    b[3]["loss"] *= 1.05
    assert any(f.startswith("loss") for f in compare(hist(), b, 1e-2)["failures"])


def test_first_step_must_be_exact_and_counts_match():
    b = copy.deepcopy(hist())
    b[0]["loss"] += 1e-4
    assert any(f.startswith("first-step") for f in compare(hist(), b, 1e-2)["failures"])
    assert compare(hist(4), hist(3), 1e-2)["failures"] == ["step counts 4 vs 3"]
    c = copy.deepcopy(hist())
    del c[1]["nfp_mse"]
    assert "nfp_mse: logged in one run only" in compare(hist(), c, 1e-2)["failures"]
