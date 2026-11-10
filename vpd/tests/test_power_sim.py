"""Tests of vpd/common/power_sim.py (CONTRACT_wave2 sections 8.3 and 9, package prereg).

Hermetic and small (seconds): stdlib maths checked against closed forms and tabulated values, the CR1 variance on a
hand case, Holm, the realized-scale shift mapping, the seed-variance estimators, and toy simulations that show the
registered seed-aware t test near alpha while the naive cluster z (no seed component) is inflated.  Login node, 1 core:
    cd TREE && OMP_NUM_THREADS=1 PY_Q3 -B -m pytest -q -p no:cacheprovider vpd/tests/test_power_sim.py
"""
from __future__ import annotations

import copy
import json
import math
import random

import pytest

from vpd.common import power_sim as PS
from vpd.tvr import pilot_cfg

np = pytest.importorskip("numpy")


# --------------------------------------------------------------------------------------------- toy design
def toy_design(seeds=3):
    return {"tiers": ["100", "TOP"], "weights": [0.5, 0.5],
            "seeds": {"100": list(range(1, seeds + 1)), "TOP": list(range(1, seeds + 1))},
            "starts": ["A", "B", "C", "D", "E"],
            "contrasts": [["A", ["B"]], ["C", ["D", "E"]]]}


def toy_p(n=500, n_clusters=100):
    p100 = [0.05 + 0.55 * ((i * 37) % 100) / 100.0 for i in range(n)]
    ptop = [min(1.0, x + 0.10) for x in p100]
    clusters = [f"s{i % n_clusters:03d}" for i in range(n)]
    return {"100": p100, "TOP": ptop}, clusters


# --------------------------------------------------------------------------------------------- Holm
def test_holm():
    assert PS.holm_reject({"a": 0.01, "b": 0.04}) == {"a": True, "b": True}
    assert PS.holm_reject({"a": 0.03, "b": 0.04}) == {"a": False, "b": False}
    assert PS.holm_reject({"a": 0.01, "b": 0.06}) == {"a": True, "b": False}
    assert PS.holm_reject({"a": 0.026, "b": 0.001}) == {"a": True, "b": True}
    assert PS.holm_reject({"a": 0.025, "b": 0.05}) == {"a": True, "b": True}   # boundaries are rejections
    assert PS.holm_reject({"a": 0.2, "b": 0.001, "c": 0.02}) == {"a": False, "b": True, "c": True}
    assert PS.holm_reject({"a": 0.01, "b": 0.02, "c": 0.03}) == {"a": True, "b": True, "c": True}
    assert PS.holm_reject({"a": 0.01, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}


# --------------------------------------------------------------------------------------------- Student t
@pytest.mark.parametrize("t,df,p", [
    (2.571, 5, 0.05), (4.032, 5, 0.01), (2.015, 5, 0.10),
    (2.086, 20, 0.05), (2.845, 20, 0.01), (1.725, 20, 0.10),
    (2.064, 24, 0.05), (2.797, 24, 0.01), (1.711, 24, 0.10),
    (2.101, 18, 0.05), (1.980, 120, 0.05),
])
def test_t_sf2_tabulated(t, df, p):
    assert PS.t_sf2(t, df) == pytest.approx(p, abs=3e-4)
    assert PS.t_sf2(-t, df) == PS.t_sf2(t, df)


def test_t_sf2_closed_forms():
    for t in (0.0, 0.3, 1.0, 2.5, 12.706, 40.0):
        assert PS.t_sf2(t, 1) == pytest.approx(1.0 - 2.0 / math.pi * math.atan(abs(t)), abs=1e-12)
        assert PS.t_sf2(t, 2) == pytest.approx(1.0 - abs(t) / math.sqrt(t * t + 2.0), abs=1e-12)
    assert PS.t_sf2(0.0, 24) == 1.0
    assert PS.t_sf2(math.inf, 24) == 0.0
    # large df -> normal
    assert PS.t_sf2(1.959964, 10 ** 5) == pytest.approx(0.05, abs=1e-4)
    with pytest.raises(ValueError):
        PS.t_sf2(1.0, 0)


def test_t_sf2_against_scipy_if_present():
    st = pytest.importorskip("scipy.stats")
    for df in (1, 3, 5, 18, 20, 24, 79):
        for t in (0.1, 0.9, 1.7, 2.06, 3.3, 6.0):
            assert PS.t_sf2(t, df) == pytest.approx(2 * st.t.sf(t, df), rel=1e-9, abs=1e-14)


def test_t_crit2():
    assert PS.t_crit2(0.05, 20) == pytest.approx(2.086, abs=1e-3)
    assert PS.t_crit2(0.05, 24) == pytest.approx(2.064, abs=1e-3)
    assert PS.t_sf2(PS.t_crit2(0.05, 24), 24) == pytest.approx(0.05, abs=1e-10)


# --------------------------------------------------------------------------------------------- CR1
def test_cr1_known_case():
    D, V = PS.cr1_var([1.0, 2.0, 3.0, 4.0], ["a", "a", "b", "b"])
    assert D == 2.5
    # residual cluster sums -2, +2; G/(G-1) * 8 / 16 = 1.0
    assert V == pytest.approx(1.0)
    # singleton clusters = HC1-type: n/(n-1) * sum e^2 / n^2
    d = [0.3, -0.1, 0.5, 0.0, 0.2]
    D2, V2 = PS.cr1_var(d, list("abcde"))
    m = sum(d) / 5
    assert V2 == pytest.approx(5 / 4 * sum((x - m) ** 2 for x in d) / 25)
    with pytest.raises(ValueError):
        PS.cr1_var([1.0, 2.0], ["a", "a"])
    with pytest.raises(ValueError):
        PS.cr1_var([1.0], ["a", "b"])


def test_seed_aware_test_adds_the_seed_component():
    d = [0.1, 0.0, 0.2, 0.1, -0.1, 0.3]
    cl = ["a", "a", "b", "b", "c", "c"]
    D, vcl = PS.cr1_var(d, cl)
    r = PS.seed_aware_test(d, cl, sigma2_hat=0.0009, df=24, k_contrast=2.0, weights={"100": 0.5, "TOP": 0.5},
                           n_seeds_by_tier={"100": 3, "TOP": 3})
    vseed = 2.0 * (0.25 * 0.0009 / 3 + 0.25 * 0.0009 / 3)
    assert r["v_cluster"] == pytest.approx(vcl) and r["v_seed"] == pytest.approx(vseed)
    assert r["se"] == pytest.approx(math.sqrt(vcl + vseed))
    assert r["t"] == pytest.approx(D / r["se"])
    assert r["p"] == pytest.approx(PS.t_sf2(r["t"], 24))
    q = PS.t_crit2(0.05, 24)
    assert r["ci95"] == pytest.approx([D - q * r["se"], D + q * r["se"]])
    r0 = PS.seed_aware_test(d, cl, 0.0, 24, 2.0, {"100": 0.5, "TOP": 0.5}, {"100": 3, "TOP": 3})
    assert r0["se"] < r["se"] and r0["p"] < r["p"]


def test_k_contrast_matches_registration():
    assert PS.k_contrast(["E2"]) == 2.0
    assert PS.k_contrast(["QRL-s1", "QRL-s2"]) == 1.5
    des = pilot_cfg.FROZEN["power"]["design"]
    ks = [PS.k_contrast(c[1]) for c in des["contrasts"]]
    assert ks == [2.0, 1.5]
    assert "2 for Q3S-E2" in pilot_cfg.FROZEN["power"]["test_def"]
    assert "1.5 for Q3RD-mean(QRL)" in pilot_cfg.FROZEN["power"]["test_def"]
    assert [PS.contrast_name(c[0], c[1]) for c in des["contrasts"]] == ["Q3S - E2", "Q3RD - mean(QRL-s1, QRL-s2)"]


# --------------------------------------------------------------------------------------------- shift mapping
def _realized(p, c):
    return sum(min(1.0, max(0.0, x + c)) for x in p) / len(p) - sum(p) / len(p)


def test_shift_table_and_inverse_with_clipping():
    p = [0.0, 0.0, 0.2, 0.5, 0.9, 1.0, 1.0]
    tab = PS.shift_table(p)
    cs, rs = tab
    assert len(cs) == 2001 and cs[0] == -1.0 and cs[-1] == 1.0 and cs[1000] == 0.0
    assert all(b >= a for a, b in zip(rs, rs[1:]))
    rmin, rmax = rs[0], rs[-1]
    assert rmin == pytest.approx(-sum(p) / len(p)) and rmax == pytest.approx(1.0 - sum(p) / len(p))
    last = -2.0
    for j in range(1, 200):
        r = rmin + (rmax - rmin) * j / 200.0
        c = PS.calibrate_shift(tab, r)
        assert abs(_realized(p, c) - r) < 1e-4
        assert c >= last
        last = c
    assert PS.calibrate_shift(tab, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert PS.calibrate_shift(tab, rmin - 0.1) == -1.0
    assert PS.calibrate_shift(tab, rmax + 0.1) == 1.0
    # vectorized version: identical semantics
    x = np.asarray([rmin - 0.5, rmin, rmin + 1e-3, -0.05, 0.0, 0.07, rmax - 1e-6, rmax, rmax + 2.0])
    got = PS._calibrate_np(np, (np.asarray(cs), np.asarray(rs)), x)
    want = [PS.calibrate_shift(tab, float(v)) for v in x]
    assert got.tolist() == pytest.approx(want, abs=1e-12)


def test_shift_table_python_equals_numpy(monkeypatch):
    p = [0.0, 0.1, 0.33, 0.8, 1.0]
    a = PS.shift_table(p, 101)
    monkeypatch.setattr(PS, "_np", lambda: None)
    b = PS.shift_table(p, 101)
    assert a[0] == b[0]
    assert a[1] == pytest.approx(b[1], abs=1e-12)


# --------------------------------------------------------------------------------------------- seed variance
def _draw_cells(rng, p, sigma, n_cells, n_seeds, tab):
    """numpy draws of n_cells seeds x tasks 0/1 matrices with realized-scale shifts (as simulate)."""
    pa = np.asarray(p)
    cells = []
    for _ in range(n_cells):
        r = rng.normal(0.0, sigma, size=n_seeds)
        c = np.asarray([PS.calibrate_shift(tab, float(v)) for v in r])
        q = np.clip(pa[None, :] + c[:, None], 0.0, 1.0)
        cells.append((rng.random(q.shape) < q).astype(int).tolist())
    return cells


def test_seed_var_pooled_unbiased_on_a_toy():
    rng = np.random.default_rng(7)
    p = [0.15 + 0.5 * ((i * 13) % 40) / 40.0 for i in range(40)]
    tab = PS.shift_table(p)
    sigma = 0.05
    ests, dfs = [], set()
    for _ in range(2000):
        cells = _draw_cells(rng, p, sigma, 12, 3, tab)
        ests.append(sum(PS.seed_var_cells(cells)) / 12)
        dfs.add(PS.seed_var_pooled(cells)[1])
    assert dfs == {24}
    mean = sum(ests) / len(ests)
    assert abs(mean - sigma ** 2) < 0.10 * sigma ** 2, (mean, sigma ** 2)


def test_seed_var_pooled_rules():
    # identical seeds -> var_s(SR) = 0, within-task var 0 -> 0
    cell = [[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]]
    assert PS.seed_var_pooled([cell]) == (0.0, 2)
    # pure binomial noise without a shift can give a negative cell estimate -> truncated at 0 after the mean
    neg = [[1, 0, 0, 1], [0, 1, 1, 0]]
    assert PS.seed_var_cells([neg])[0] < 0
    assert PS.seed_var_pooled([neg]) == (0.0, 1)
    # hand case: SR 1.0 and 0.0 over n=2 tasks, 2 seeds: var_s(SR) = 0.5; per task var = 0.5; mean/n = 0.25
    assert PS.seed_var_cells([[[1, 1], [0, 0]]]) == [pytest.approx(0.25)]
    # df = sum over cells of (n_seeds - 1)
    assert PS.seed_var_pooled([[[1, 1], [0, 0]], [[1, 0], [0, 1], [1, 1]]])[1] == 3
    with pytest.raises(ValueError):
        PS.seed_var_pooled([[[1, 0]]])


def test_seed_var_endpoint():
    # per start: e_si = sum_T w_T y; same algebra as a cell, df = sum over starts of (n_seeds - 1)
    e1 = [[1.0, 1.0], [0.0, 0.0]]                 # E_s 1.0 / 0.0 -> var 0.5; per task var 0.5 -> 0.5 / 2
    e2 = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]     # identical seeds -> 0
    s2, df = PS.seed_var_endpoint([e1, e2])
    assert df == 1 + 2 and s2 == pytest.approx((0.25 + 0.0) / 2)
    assert PS.seed_var_endpoint([[[1.0, 0.0], [0.0, 1.0]]]) == (0.0, 1)   # negative estimate truncated
    with pytest.raises(ValueError):
        PS.seed_var_endpoint([])


def test_seed_sd():
    r = PS.seed_sd([1, 0, 1, 0], [1, 0, 1, 0])
    assert r["sigma_hat"] == 0.0 and r["sigma_used"] == 0.03
    r = PS.seed_sd([1, 1, 0, 0], [0, 0, 0, 0])
    assert r["seed_diff_mean"] == 0.5 and r["S"] == pytest.approx(1 / 12)
    assert r["sigma_hat"] == pytest.approx(math.sqrt((0.25 - 1 / 12) / 2))
    assert r["sigma_used"] == r["sigma_hat"]
    r = PS.seed_sd([1] * 10, [0] * 10)
    assert r["S"] == 0.0 and r["sigma_hat"] == pytest.approx(math.sqrt(0.5))
    assert PS.seed_sd([1, 0, 0, 1], [0, 1, 1, 0], a_priori=0.05)["sigma_used"] == 0.05
    with pytest.raises(ValueError):
        PS.seed_sd([1], [0])


# --------------------------------------------------------------------------------------------- confirmatory rule
def _res(D, p, hi):
    return {"D": D, "p": p, "ci95": [D - 0.1, hi]}


def test_confirmatory_outcome_three_way():
    f = PS.confirmatory_outcome
    o = f({"a": _res(0.08, 0.001, 0.12), "b": _res(0.01, 0.5, 0.06)})
    assert o["per_contrast"] == {"a": "support", "b": "inconclusive"} and o["overall"] == "支持"
    o = f({"a": _res(0.08, 0.001, 0.12), "b": _res(-0.08, 0.002, -0.03)})
    assert o["per_contrast"]["b"] == "no_support" and o["overall"] == "无定论"   # opposite rejection
    o = f({"a": _res(0.0, 0.9, 0.04), "b": _res(-0.01, 0.6, 0.049)})
    assert o["per_contrast"] == {"a": "no_support", "b": "no_support"} and o["overall"] == "不支持"
    o = f({"a": _res(0.0, 0.9, 0.04), "b": _res(0.02, 0.3, 0.07)})
    assert o["overall"] == "无定论"
    # Holm: 0.03 alone is not enough (0.03 > 0.025) when the other p is large
    o = f({"a": _res(0.06, 0.03, 0.11), "b": _res(0.0, 0.8, 0.05)})
    assert o["holm"] == {"a": False, "b": False} and o["per_contrast"]["a"] == "inconclusive"
    assert pilot_cfg.FROZEN["power"]["outcome"]["min_effect"] == 0.05


# --------------------------------------------------------------------------------------------- simulation
def test_stats_numpy_equals_python():
    p, cl = toy_p(60, 12)
    des = toy_design()
    parts = PS._design_parts(des, p, cl)
    rng = np.random.default_rng(3)
    B = 4
    y = {(S, T): rng.random((B, 3, 60)) < 0.4 for S in parts["starts"] for T in parts["tiers"]}
    uniq = sorted(set(cl))
    idx = np.asarray([uniq.index(g) for g in cl])
    for test in PS.TESTS:
        s2, df, res = PS._stats_numpy(np, y, parts, idx, len(uniq), test)
        for b in range(B):
            yb = {k: v[b].astype(int).tolist() for k, v in y.items()}
            s2p, dfp, rp = PS._stats_python(yb, parts, cl, test)
            # cells: 5 starts x 2 tiers x (3 - 1); endpoint variant: 5 starts x (3 - 1)
            assert dfp == df == (10 if test == "seed_aware_endpoint_t" else 20)
            assert float(s2[b]) == pytest.approx(s2p, abs=1e-12)
            for (D, vc, vs), r in zip(res, rp):
                assert float(D[b]) == pytest.approx(r["D"], abs=1e-12)
                pn = PS._p_of(float(D[b]), float(vc[b]), float(vs[b]), df, test)
                assert pn == pytest.approx(r["p"], abs=1e-9)
    # the cluster sums need no BLAS: a shuffled cluster labelling gives the same CR1 variance as cr1_var
    d = rng.random(60)
    perm = list(rng.permutation(12))
    lab = [f"g{perm[i % 12]}" for i in range(60)]
    ul = sorted(set(lab))
    one = {(S, T): np.zeros((1, 3, 60), dtype=bool) for S in parts["starts"] for T in parts["tiers"]}
    one[("A", "100")][0, :, :] = d[None, :] > 0.5
    _, _, res = PS._stats_numpy(np, one, parts, np.asarray([ul.index(g) for g in lab]), 12, "naive_cluster_z")
    rp = PS._stats_python({k: v[0].astype(int).tolist() for k, v in one.items()}, parts, lab, "naive_cluster_z")[2]
    assert float(res[0][1][0]) == pytest.approx(PS.cr1_var([0.5 * float(x > 0.5) for x in d], lab)[1], abs=1e-15)
    assert float(res[0][1][0]) ** 0.5 == pytest.approx(rp[0]["se"], abs=1e-12)


def test_import_sets_single_thread_defaults():
    import os
    import subprocess
    import sys
    from vpd import paths

    code = ("import os, sys; [os.environ.pop(v, None) for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', "
            "'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')]; os.environ['MKL_NUM_THREADS'] = '3'; "
            "import vpd.common.power_sim; print(','.join(os.environ[v] for v in ('OMP_NUM_THREADS', "
            "'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')))")
    env = {k: v for k, v in os.environ.items() if not k.endswith("_NUM_THREADS")}
    env["PYTHONPATH"] = str(paths.WORKTREE)
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=60, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "1,3,1,1"          # defaults to 1, never overrides a caller's value


def test_simulate_reproducible_by_seed():
    p, cl = toy_p(60, 12)
    des = toy_design()
    for be, ns in (("numpy", 60), ("python", 20)):
        a = PS.simulate(p, cl, des, 0.05, 0.03, ns, seed=11, backend=be)
        b = PS.simulate(p, cl, des, 0.05, 0.03, ns, seed=11, backend=be)
        c = PS.simulate(p, cl, des, 0.05, 0.03, ns, seed=12, backend=be)
        assert a == b
        assert a["backend"] == be and a["df"] == 20
        assert (a["power"], a["sigma2_hat_mean"]) != (c["power"], c["sigma2_hat_mean"])
    z = PS.simulate(p, cl, des, 0.0, 0.03, 30, seed=1, backend="numpy")
    assert set(z) >= {"fwer", "type1_by_contrast", "reject_holm", "power", "sigma2_hat_mean"}
    assert z["k_contrast"] == {"A - B": 2.0, "C - mean(D, E)": 1.5}


def test_type1_seed_aware_near_alpha_and_naive_inflated():
    p, cl = toy_p(500, 100)
    des = toy_design()
    for sigma in (0.0, 0.03):
        r = PS.simulate(p, cl, des, 0.0, sigma, 400, seed=5, backend="numpy")
        assert r["fwer"] <= 0.07, (sigma, r["fwer"])
        assert all(v <= 0.07 for v in r["type1_by_contrast"].values()), r
    naive = PS.simulate(p, cl, des, 0.0, 0.03, 400, seed=5, backend="numpy", test="naive_cluster_z")
    assert naive["fwer"] > 0.2, naive["fwer"]
    # the estimated seed variance tracks the truth on the realized scale (sigma 0.03 -> 9e-4)
    r = PS.simulate(p, cl, des, 0.0, 0.03, 400, seed=5, backend="numpy")
    assert r["sigma2_hat_mean"] == pytest.approx(0.0009, rel=0.15)


def test_power_monotone_and_only_treated_better_counted():
    p, cl = toy_p(500, 100)
    des = toy_design()
    pw = [PS.simulate(p, cl, des, d, 0.03, 200, seed=9, backend="numpy")["power"] for d in (0.0, 0.05, 0.10)]
    for k in ("A - B", "C - mean(D, E)"):
        seq = [x[k] for x in pw]
        assert seq == sorted(seq), (k, seq)
        assert seq[-1] > 0.5
    neg = PS.simulate(p, cl, des, -0.10, 0.03, 200, seed=9, backend="numpy")
    for k in ("A - B", "C - mean(D, E)"):
        assert neg["power"][k] <= 0.01
        assert neg["reject_holm"][k] >= 0.5


def test_effect_is_delta_on_the_realized_scale_under_clipping():
    """The registered effect model: the treated start uses calibrate_shift(r + delta), so its expected REALIZED SR
    gain is delta even when many tasks sit at 0 or 1 (a pre-clip shift c + delta would give about 0.6 delta here)."""
    p = [0.0] * 20 + [1.0] * 20 + [0.5] * 10
    tab = PS.shift_table(p)
    rng = random.Random(4)
    for sigma in (0.0, 0.03):
        rs = [rng.gauss(0.0, sigma) for _ in range(2000)]
        gain = sum(_realized(p, PS.calibrate_shift(tab, r + 0.08)) for r in rs) / len(rs)
        base = sum(_realized(p, PS.calibrate_shift(tab, r)) for r in rs) / len(rs)
        assert gain - base == pytest.approx(0.08, abs=1e-3), sigma
    assert _realized(p, 0.08) == pytest.approx(0.08 * 30 / 50)          # the wrong (pre-clip) model
    # one simulate block: the shifted contrast's mean simulated D is about delta, the other about 0
    pp = {"100": p * 10, "TOP": p * 10}
    cl = [f"s{i % 50:02d}" for i in range(500)]
    r = PS.simulate(pp, cl, toy_design(), 0.08, 0.0, 300, seed=2, backend="numpy")
    assert r["sim_mean_D"]["A - B"] == pytest.approx(0.08, abs=0.006)
    assert r["sim_mean_D"]["C - mean(D, E)"] == pytest.approx(0.08, abs=0.006)   # its own block
    z = PS.simulate(pp, cl, toy_design(), 0.0, 0.0, 300, seed=2, backend="numpy")
    assert all(abs(v) < 0.006 for v in z["sim_mean_D"].values())


def test_tier_corr_sensitivity():
    p, cl = toy_p(500, 100)
    des = toy_design()
    base = PS.simulate(p, cl, des, 0.0, 0.03, 60, seed=5, backend="numpy")
    assert PS.simulate(p, cl, des, 0.0, 0.03, 60, seed=5, backend="numpy", tier_corr=0.0) == base
    assert base["tier_corr"] == 0.0
    py = PS.simulate(p, cl, des, 0.0, 0.03, 5, seed=5, backend="python", tier_corr=1.0)
    assert py["tier_corr"] == 1.0 and py["df"] == 20
    with pytest.raises(ValueError):
        PS.simulate(p, cl, des, 0.0, 0.03, 5, seed=5, tier_corr=1.5)
    unpaired = toy_design()
    unpaired["seeds"]["TOP"] = [1, 2]
    with pytest.raises(ValueError):
        PS.simulate(p, cl, unpaired, 0.0, 0.03, 5, seed=5, test="seed_aware_endpoint_t")
    # fully correlated tier shifts at a large sigma: the registered test (independence assumed) is inflated,
    # the endpoint-level variant stays near alpha
    reg = PS.simulate(p, cl, des, 0.0, 0.05, 400, seed=6, backend="numpy", tier_corr=1.0)
    ept = PS.simulate(p, cl, des, 0.0, 0.05, 400, seed=6, backend="numpy", tier_corr=1.0,
                      test="seed_aware_endpoint_t")
    assert ept["df"] == 10 and ept["fwer"] <= 0.08, ept["fwer"]
    assert reg["fwer"] > ept["fwer"] + 0.02, (reg["fwer"], ept["fwer"])
    # the endpoint estimator tracks Var(sum_T w_T r_T) = sigma^2 (w1 + w2)^2 = sigma^2 when tier_corr = 1
    assert ept["sigma2_hat_mean"] == pytest.approx(0.0025, rel=0.2)


def test_simulate_refuses_bad_design():
    p, cl = toy_p(20, 4)
    with pytest.raises(ValueError):
        PS.simulate(p, cl, toy_design(seeds=1), 0.0, 0.03, 5, seed=0)
    bad = toy_design()
    bad["contrasts"] = [["A", ["Z"]]]
    with pytest.raises(ValueError):
        PS.simulate(p, cl, bad, 0.0, 0.03, 5, seed=0)
    with pytest.raises(ValueError):
        PS.simulate({"100": p["100"]}, cl, toy_design(), 0.0, 0.03, 5, seed=0)


# --------------------------------------------------------------------------------------------- run_tvr
def _write_pilot(root, n=40, n_scenes=8, replicate=True, seed=0):
    rng = random.Random(seed)
    starts = ("Q3B", "Q3S", "E2")
    ids = [f"task_{i:03d}" for i in range(n)]
    runs = [(S, t) for S in starts for t in ("100", "ALL")] + ([("Q3B", "100s1002")] if replicate else [])
    for S, tier in runs:
        d = PS.eval_dir(root, S, tier)
        d.mkdir(parents=True)
        with open(d / "per_task.jsonl", "w") as f:
            for i, t in enumerate(ids):
                f.write(json.dumps({"task_id": t, "category": "SR-easy", "scene": f"scene{i % n_scenes}",
                                    "success": int(rng.random() < 0.3 + 0.01 * i)}) + "\n")
    return ids


def _cfg():
    cfg = copy.deepcopy(pilot_cfg.FROZEN)
    cfg["tiers"]["n_top"] = "ALL"
    return cfg


def test_run_tvr_end_to_end(tmp_path):
    _write_pilot(tmp_path / "pilot")
    out = tmp_path / "pilot" / "POWER.json"
    res = PS.run_tvr(tmp_path / "pilot", cfg=_cfg(), out=out, n_sims=30, backend="numpy")
    disk = json.loads(out.read_text(encoding="utf-8"))
    assert disk["schema"] == PS.SCHEMA and disk["pass"] == res["pass"]
    assert set(disk["tables"]) == {"main", "fallback_top_seeds_1_2"}
    for name, rows in disk["tables"].items():
        assert len(rows) == 3 * 4   # sigma rows (used, a priori, sensitivity) x deltas (0 + 3)
        assert {r["sigma_label"] for r in rows} == {"used", "a_priori", "sensitivity"}
        assert {r["delta"] for r in rows} == {0.0, 0.05, 0.08, 0.10}
        assert {r["df"] for r in rows} == ({24} if name == "main" else {18})
    s = disk["sigma"]
    assert s["sigma_used"] == max(0.03, s["sigma_hat"]) and s["replicate"]["seed_b"] == 1002
    assert disk["inputs"]["tiers"] == {"100": "100", "TOP": "ALL"}
    assert len(disk["inputs"]["files_sha256"]) == 7
    assert set(disk["p_summary"]) == {"100", "TOP"}            # pooled only; nothing per start
    assert disk["registered"]["n_sims_used"] == 30 and disk["registered"]["n_sims_registered"] == 2000
    v = disk["verdicts"]["main"]
    assert v["pass"] == (v["type1_ok"] and v["power_ok"])
    assert set(v["power_at_primary"]) == {"Q3S - E2", "Q3RD - mean(QRL-s1, QRL-s2)"}
    assert disk["registered"]["allow_no_replicate"] is False and s["replicate_missing"] is False
    # provenance names: the json sha of the cfg used and the pilot_cfg.py file sha (GATE.json's pilot_cfg_sha256)
    assert disk["inputs"]["pilot_cfg_json_sha256"] == PS.hashlib.sha256(pilot_cfg.as_json(_cfg()).encode()).hexdigest()
    assert disk["inputs"]["pilot_cfg_py_sha256"] == PS.sha256_file(pilot_cfg.__file__)
    assert "pilot_cfg_sha256" not in disk["inputs"]
    # the tier-correlation sensitivity block: reported, never gated
    sens = disk["sensitivity_tier_corr"]
    assert sens["gated"] is False and sens["design"] == "main"
    assert [(x["test"], x["tier_corr"]) for x in sens["summary"]] == [
        ("seed_aware_cluster_t", 1.0), ("seed_aware_endpoint_t", 0.0), ("seed_aware_endpoint_t", 1.0)]
    assert [x["registered"] for x in sens["summary"]] == [True, False, False]
    assert {x["df"] for x in sens["summary"] if x["test"] == "seed_aware_endpoint_t"} == {6 * 2}
    assert all(x["sigma"] == s["sigma_used"] for x in sens["summary"])
    # separation (brief section 3): no key anywhere in POWER.json names a pilot start except the SIMULATED
    # confirmatory contrasts and the input file paths; start-bearing string values only where whitelisted; the
    # p summary is by tier only
    starts = set(pilot_cfg.FROZEN["starts"])
    sim_contrasts = {PS.contrast_name(c[0], c[1]) for c in pilot_cfg.FROZEN["power"]["design"]["contrasts"]}

    def walk(o, path=()):
        if isinstance(o, dict):
            for k, x in o.items():
                if path[-1:] != ("files_sha256",) and k not in sim_contrasts:
                    assert not any(st in str(k) for st in starts), (path, k)
                yield from walk(x, path + (k,))
        elif isinstance(o, list):
            for x in o:
                yield from walk(x, path)
        else:
            yield path, o

    for path, val in walk(disk):
        if isinstance(val, str) and any(st in val for st in starts):
            ok = (path[:2] == ("inputs", "pilot_root") or path[:2] == ("inputs", "starts_pooled")
                  or path[:2] == ("sigma", "replicate") or path[:1] == ("note",)
                  or path[-1:] == ("contrasts",) or path[:3] == ("registered", "test_def")
                  or path[:2] == ("registered", "p_source"))
            assert ok, (path, val)
    assert set(disk["p_summary"]) == {"100", "TOP"} and set(disk["p_summary"]["TOP"]) == {"n", "mean", "n_p0", "n_p1"}


def test_run_tvr_replicate_required_and_refusals(tmp_path):
    _write_pilot(tmp_path / "p", replicate=False)
    # the replicate is required (O4); an explicit opt-out falls back to the a priori sigma and says so
    with pytest.raises(ValueError, match="replicate"):
        PS.run_tvr(tmp_path / "p", cfg=_cfg(), n_sims=5, backend="numpy")
    res = PS.run_tvr(tmp_path / "p", cfg=_cfg(), n_sims=5, backend="numpy", allow_no_replicate=True)
    assert res["sigma"]["replicate_missing"] is True and res["sigma"]["sigma_used"] == 0.03
    assert res["registered"]["allow_no_replicate"] is True
    # n_top not frozen -> refused
    with pytest.raises(ValueError):
        PS.run_tvr(tmp_path / "p", cfg=copy.deepcopy(pilot_cfg.FROZEN) | {"tiers": {**pilot_cfg.FROZEN["tiers"],
                                                                                    "n_top": None}}, n_sims=5,
                   allow_no_replicate=True)
    # a start whose task ids differ -> refused
    f = PS.eval_dir(tmp_path / "p", "E2", "ALL") / "per_task.jsonl"
    lines = f.read_text().splitlines()
    f.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError):
        PS.run_tvr(tmp_path / "p", cfg=_cfg(), n_sims=5, allow_no_replicate=True)
    # a replicate whose task ids differ from the pooled ids -> refused
    _write_pilot(tmp_path / "q")
    f = PS.eval_dir(tmp_path / "q", "Q3B", "100s1002") / "per_task.jsonl"
    f.write_text("\n".join(f.read_text().splitlines()[1:]) + "\n")
    with pytest.raises(ValueError, match="replicate"):
        PS.run_tvr(tmp_path / "q", cfg=_cfg(), n_sims=5)


def test_cli_exit_codes(tmp_path, monkeypatch, capsys):
    """main() reads pilot_cfg.FROZEN at call time; independent of the shipped fill state."""
    _write_pilot(tmp_path / "p")
    _write_pilot(tmp_path / "norep", replicate=False)
    argv = ["--n-sims", "5", "--backend", "numpy"]
    unfrozen = copy.deepcopy(pilot_cfg.FROZEN)
    unfrozen["tiers"]["n_top"] = None
    monkeypatch.setattr(pilot_cfg, "FROZEN", unfrozen)
    assert PS.main(["tvr", "--pilot-root", str(tmp_path / "p"), "--out", str(tmp_path / "a.json"), *argv]) == 2
    assert "n_top" in capsys.readouterr().err and not (tmp_path / "a.json").exists()
    monkeypatch.setattr(pilot_cfg, "FROZEN", _cfg())
    rc = PS.main(["tvr", "--pilot-root", str(tmp_path / "p"), "--out", str(tmp_path / "b.json"), *argv])
    disk = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    assert rc == (0 if disk["pass"] else 1) and disk["registered"]["n_sims_used"] == 5
    assert "power_sim tvr: pass=" in capsys.readouterr().out
    # no replicate: refused unless --allow-no-replicate
    assert PS.main(["tvr", "--pilot-root", str(tmp_path / "norep"), "--out", str(tmp_path / "c.json"), *argv]) == 2
    assert "replicate" in capsys.readouterr().err and not (tmp_path / "c.json").exists()
    rc = PS.main(["tvr", "--pilot-root", str(tmp_path / "norep"), "--out", str(tmp_path / "c.json"), *argv,
                  "--allow-no-replicate"])
    assert rc in (0, 1) and json.loads((tmp_path / "c.json").read_text())["sigma"]["replicate_missing"] is True
