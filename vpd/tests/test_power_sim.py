"""Tests of vpd/common/power_sim.py (CONTRACT_wave2 sections 8.3 and 9, package prereg; DECISIONS G13).

Hermetic and small (seconds): stdlib maths checked against closed forms and tabulated values, the CR1 variance on a
hand case, Holm, the realized-scale shift mapping, the seed-variance estimators, toy simulations that show the
seed-aware t tests near alpha while the naive cluster z (no seed component) is inflated, the registered endpoint test
under correlated tier shifts, the task-persistence draws and estimator (recovers lambda in the model at the pilot's
n; does not move with the start mean SRs, where the wave-2 estimator did), run_tvr's grid and verdict on the
registered cells, and the registered backend.  Wave 2c: the registered lambda margin (clip edges), the POWER.json
persistence whitelist, the statcheck reconstruction of a between-start difference failing on the real run_tvr path,
and the report-only design rows.  Login node, 1 core:
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


def _cov_cell(A, B):
    """cov_s(SR) - mean_i cov_s(y) / n over the paired seeds (ddof 1), the cross term of the endpoint identity."""
    ns, n = len(A), len(A[0])
    sa, sb = [sum(r) / n for r in A], [sum(r) / n for r in B]
    ma, mb = sum(sa) / ns, sum(sb) / ns
    c_sr = sum((x - ma) * (y - mb) for x, y in zip(sa, sb)) / (ns - 1)
    c_i = 0.0
    for i in range(n):
        ca, cb = [A[s][i] for s in range(ns)], [B[s][i] for s in range(ns)]
        xa, xb = sum(ca) / ns, sum(cb) / ns
        c_i += sum((x - xa) * (y - xb) for x, y in zip(ca, cb)) / (ns - 1)
    return c_sr - c_i / n / n


def test_endpoint_identity():
    """The registered endpoint seed variance contains the cross-tier covariance: on random paired cells,
    seed_var_cells(w1 A + w2 B) == w1^2 v(A) + w2^2 v(B) + 2 w1 w2 [cov_s(SR) - mean_i cov_s(y) / n] (before the
    truncation), while the superseded per-tier estimate is the same expression without the cross term."""
    rng = random.Random(3)
    for ns, n, w1 in ((3, 17, 0.5), (2, 9, 0.5), (4, 11, 0.3)):
        w2 = 1.0 - w1
        for _ in range(20):
            A = [[int(rng.random() < 0.4) for _ in range(n)] for _ in range(ns)]
            B = [[int(rng.random() < 0.6) for _ in range(n)] for _ in range(ns)]
            E = [[w1 * a + w2 * b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]
            vA, vB, vE = PS.seed_var_cells([A, B, E])
            assert vE == pytest.approx(w1 ** 2 * vA + w2 ** 2 * vB + 2 * w1 * w2 * _cov_cell(A, B), abs=1e-12)
    # identical tiers (perfectly correlated shifts): the endpoint term is (w1 + w2)^2 v = v, the per-tier one only
    # (w1^2 + w2^2) v = v / 2
    A = [[1, 1, 0, 1], [0, 0, 0, 1], [1, 0, 1, 1]]
    v = PS.seed_var_cells([A])[0]
    E = [[0.5 * a + 0.5 * a for a in r] for r in A]
    assert PS.seed_var_cells([E])[0] == pytest.approx(v) and v > 0


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
    # the registered (endpoint) test stays near alpha at both registered type-I rhos (power.tier_corr.type1)
    for rho in pilot_cfg.FROZEN["power"]["tier_corr"]["type1"]:
        e = PS.simulate(p, cl, des, 0.0, 0.03, 400, seed=6, backend="numpy", tier_corr=rho,
                        test="seed_aware_endpoint_t")
        assert e["fwer"] <= 0.08, (rho, e["fwer"])
        assert e["tier_corr"] == rho and e["df"] == 10


# --------------------------------------------------------------------------------------------- task persistence
def test_persistence_zero_draws_as_before():
    p, cl = toy_p(60, 12)
    des = toy_design()
    for be, ns in (("numpy", 40), ("python", 4)):
        base = PS.simulate(p, cl, des, 0.05, 0.03, ns, seed=5, backend=be, test="seed_aware_endpoint_t")
        z = PS.simulate(p, cl, des, 0.05, 0.03, ns, seed=5, backend=be, test="seed_aware_endpoint_t",
                        persistence=0.0)
        assert z == base and z["persistence"] == 0.0
        lam = PS.simulate(p, cl, des, 0.05, 0.03, ns, seed=5, backend=be, test="seed_aware_endpoint_t",
                          persistence=0.5)
        assert lam["persistence"] == 0.5
        assert (lam["power"], lam["sigma2_hat_mean"]) != (base["power"], base["sigma2_hat_mean"])
    with pytest.raises(ValueError):
        PS.simulate(p, cl, des, 0.0, 0.03, 5, seed=5, persistence=1.5)


def test_persistence_draw_moments():
    """_draw_start_np (the draws of simulate): marginal P(y = 1) = q; two seeds of one task in one start covary by
    lambda q (1 - q); the two tiers share the persistent uniform (comonotone part); different starts are
    independent."""
    p = [0.5] * 200
    tab = PS.shift_table(p)
    tabs = {T: (np.asarray(tab[0]), np.asarray(tab[1])) for T in ("100", "TOP")}
    pa = {T: np.asarray(p) for T in ("100", "TOP")}
    rng = np.random.default_rng(11)
    lam, B = 0.6, 4000
    a = PS._draw_start_np(np, rng, B, ["100", "TOP"], {"100": [0, 1], "TOP": [0, 1]}, 2, tabs, pa, 0.0, 0.0, lam, 0.0)
    b = PS._draw_start_np(np, rng, B, ["100", "TOP"], {"100": [0, 1], "TOP": [0, 1]}, 2, tabs, pa, 0.0, 0.0, lam, 0.0)
    y0, y1 = a["100"][:, 0, :].astype(float), a["100"][:, 1, :].astype(float)
    assert y0.mean() == pytest.approx(0.5, abs=0.005)
    cov = ((y0 - y0.mean()) * (y1 - y1.mean())).mean()
    assert cov == pytest.approx(lam * 0.25, abs=0.01)                      # 0.15
    t0 = a["TOP"][:, 1, :].astype(float)
    assert ((y0 - y0.mean()) * (t0 - t0.mean())).mean() == pytest.approx(lam * 0.25, abs=0.01)   # across tiers
    other = b["100"][:, 0, :].astype(float)
    assert abs(((y0 - y0.mean()) * (other - other.mean())).mean()) < 0.01   # across starts: independent
    # lambda 1, sigma 0: every seed of a start gives the same outcome on a task
    c = PS._draw_start_np(np, rng, 50, ["100"], {"100": [0, 1, 2]}, 3, tabs, pa, 0.0, 0.0, 1.0, 0.0)
    assert (c["100"][:, 0, :] == c["100"][:, 2, :]).all()


def test_persistence_one_backends_agree_on_identical_seeds():
    """lambda 1 and sigma 0: every seed of a start is identical, so the endpoint seed variance estimate is 0 (up to
    float rounding) on both backends (the python path draws persistence too)."""
    p, cl = toy_p(30, 6)
    for be, ns in (("numpy", 20), ("python", 8)):
        r = PS.simulate(p, cl, toy_design(), 0.0, 0.0, ns, seed=2, backend=be, test="seed_aware_endpoint_t",
                        persistence=1.0)
        assert r["sigma2_hat_mean"] == pytest.approx(0.0, abs=1e-15), be
        z = PS.simulate(p, cl, toy_design(), 0.0, 0.0, ns, seed=2, backend=be, test="seed_aware_endpoint_t")
        assert z["sigma2_hat_mean"] > 1e-6, be      # without persistence binomial noise gives positive estimates


def _folded_mean(a, v, k=20001):
    """E|X| for X ~ N(a, v) by a midpoint rule on a +-12 SD grid (independent of shift_gap's closed form)."""
    s = math.sqrt(v)
    lo, hi = a - 12 * s, a + 12 * s
    h = (hi - lo) / k
    tot = 0.0
    for j in range(k):
        x = lo + (j + 0.5) * h
        tot += abs(x) * math.exp(-(x - a) ** 2 / (2 * v))
    return tot * h / math.sqrt(2 * math.pi * v)


def test_shift_gap():
    """kappa = E[|r1 - r2| | dbar, S] under the seed model: the folded-normal posterior mean; |dbar| without noise;
    and marginally unbiased for E|r1 - r2| = 2 sigma / sqrt(pi) when the shifts follow the model (MC)."""
    assert PS.shift_gap(0.07, 0.0, 0.03) == 0.07 and PS.shift_gap(-0.02, 0.0004, 0.0) == 0.02
    for dbar, S, sig in ((0.0, 0.0004, 0.03), (0.05, 0.0004, 0.03), (-0.08, 0.0003, 0.05), (0.01, 0.001, 0.02)):
        tau2 = 2 * sig ** 2
        B = tau2 / (tau2 + S)
        assert PS.shift_gap(dbar, S, sig) == pytest.approx(_folded_mean(B * dbar, B * S), rel=1e-6, abs=1e-9)
    rng = random.Random(5)
    sig, s2 = 0.03, 0.0004
    xs = []
    for _ in range(40000):
        delta = rng.gauss(0.0, math.sqrt(2) * sig)
        xs.append(PS.shift_gap(delta + rng.gauss(0.0, math.sqrt(s2)), s2, sig))
    m = sum(xs) / len(xs)
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))
    assert abs(m - 2 * sig / math.sqrt(math.pi)) <= 3 * se, (m, 2 * sig / math.sqrt(math.pi), se)


def _factor_v_ref(l, ya, yb, groups):
    """Independent reference (statistics.covariance, explicit leave-one-group-out loops) of the jackknifed
    one-factor V of persistence_hat: G V - (G - 1) mean_g V_-g, V = C(l, ya) C(l, yb) / C(ya, yb)."""
    import statistics as st

    def v(idx):
        a, b, c = ([l[i] for i in idx], [ya[i] for i in idx], [yb[i] for i in idx])
        return st.covariance(a, b) * st.covariance(a, c) / st.covariance(b, c)
    allidx = list(range(len(l)))
    V = v(allidx)
    loo = [v([i for i in allidx if i not in g]) for g in groups]
    return V, len(groups) * V - (len(groups) - 1) * sum(loo) / len(groups)


def test_persistence_hat_hand_case():
    """The registered estimator (PREREG 12 item 7) against an independent computation: D, the pair's level L, the
    jackknifed one-factor V (per task and per scene), B = L (1 - L) - V, R = 2 B + m2 / 2 with the posterior moments
    of the shift difference, lambda_hat = (R - D) / (R - kappa); the undefined and clipped cases."""
    y1 = [1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
    y2 = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    ya = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
    yb = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
    l = [(a + b) / 2 for a, b in zip(y1, y2)]
    sd = PS.seed_sd(y1, y2)
    d, S = sd["seed_diff_mean"], sd["S"]
    tau2 = 2 * 0.03 ** 2
    bb = tau2 / (tau2 + S)
    m2 = (bb * d) ** 2 + bb * S
    kappa = PS.shift_gap(d, S, 0.03)
    D = sum((a - b) ** 2 for a, b in zip(y1, y2)) / 10
    L = sum(l) / 10
    for clusters, groups in ((None, [[i] for i in range(10)]),
                             (["s0", "s0", "s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
                              [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])):
        r = PS.persistence_hat(y1, y2, [ya, yb], 0.03, clusters)
        V_plain, V = _factor_v_ref(l, ya, yb, groups)
        B = L * (1 - L) - V
        R = 2 * B + m2 / 2
        assert r["V_plain"] == pytest.approx(V_plain) and r["V"] == pytest.approx(V), clusters
        assert r["D"] == pytest.approx(D) and r["level"] == pytest.approx(L) and r["kappa"] == pytest.approx(kappa)
        assert r["m2"] == pytest.approx(m2) and r["B"] == pytest.approx(B) and r["R"] == pytest.approx(R)
        assert r["lambda_hat"] == pytest.approx((R - D) / (R - kappa))
        assert r["lambda_used"] == min(1.0, max(0.0, r["lambda_hat"] + 0.03)) and r["sigma_used"] == 0.03
        assert r["margin"] == 0.03
        assert r["n"] == 10 and r["n_pooled"] == 3 and r["se"] > 0
    # identical seeds: D = 0, dbar = S = 0 -> kappa = m2 = 0 -> lambda_hat = 1
    assert PS.persistence_hat(y1, y1, [ya, yb], 0.03)["lambda_hat"] == pytest.approx(1.0)
    # no task pattern among the other starts (a constant row: covariance 0): not identified -> the range's upper end
    u = PS.persistence_hat(y1, y2, [ya, [1] * 10], 0.03)
    assert u["lambda_hat"] is None and u["lambda_used"] == 1.0 and u["V"] is None
    # negative lambda_hat is clipped to the range
    neg = PS.persistence_hat([1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [[1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0]], 0.03)
    assert neg["lambda_hat"] < 0 and neg["lambda_used"] == 0.0
    with pytest.raises(ValueError):
        PS.persistence_hat([1, 0, 1], [1, 0], [ya[:3], yb[:3]], 0.03)
    with pytest.raises(ValueError):          # the other two pilot starts are required
        PS.persistence_hat(y1, y2, [ya], 0.03)
    with pytest.raises(ValueError):
        PS.persistence_hat(y1, y2, [ya, yb[:9]], 0.03)
    # m2 is the posterior second moment: a^2 + v; without noise dbar^2
    assert PS.shift_m2(0.04, 0.0, 0.03) == pytest.approx(0.0016) and PS.shift_m2(0.04, 0.0004, 0.0) == pytest.approx(
        0.0016)
    xs = [i / 4000 - 0.5 for i in range(4001)]
    a_, v_ = bb * d, bb * S
    num = sum(x * x * math.exp(-(x - a_) ** 2 / (2 * v_)) for x in xs) / sum(math.exp(-(x - a_) ** 2 / (2 * v_))
                                                                       for x in xs)
    assert PS.shift_m2(d, S, 0.03) == pytest.approx(num, rel=1e-3)


def test_lambda_used_margin_and_clip_edges(tmp_path):
    """wave 2c (independent statistics check): lambda_used = clip(lambda_hat + power.persistence.margin, range), the
    registered margin 0.03; the clip edges at both ends; undefined -> range[1]; persistence_hat applies the
    registered margin by default and an explicit one when given."""
    assert pilot_cfg.FROZEN["power"]["persistence"]["margin"] == 0.03 == PS.registered_margin()
    rule = PS.lambda_used_rule
    assert rule(0.5, 0.03) == pytest.approx(0.53) and rule(0.0, 0.03) == pytest.approx(0.03)
    assert rule(-0.01, 0.03) == pytest.approx(0.02)          # a slightly negative estimate is lifted above 0
    assert rule(-0.03, 0.03) == 0.0 and rule(-0.2, 0.03) == 0.0 and rule(-1.5, 0.03) == 0.0   # lower clip
    assert rule(0.98, 0.03) == 1.0 and rule(1.0, 0.03) == 1.0 and rule(1.4, 0.03) == 1.0       # upper clip
    assert rule(0.96, 0.03) == pytest.approx(0.99)
    assert rule(None, 0.03) == 1.0 and rule(None, 0.03, (0.0, 0.6)) == 0.6                     # undefined
    assert rule(0.55, 0.03, (0.0, 0.5)) == 0.5 and rule(0.2, 0.0) == 0.2
    # persistence_hat on the fake pilot's replicate pair (lambda_hat about 0.44): the margin is applied
    _write_pilot(tmp_path / "p")
    y1, y2, others, cl = _rep_and_others(tmp_path / "p")
    su = PS.seed_sd(y1, y2, 0.03)["sigma_used"]
    r = PS.persistence_hat(y1, y2, others, su, cl)
    assert 0.0 < r["lambda_hat"] < 0.97 and r["margin"] == 0.03
    assert r["lambda_used"] == pytest.approx(r["lambda_hat"] + 0.03)
    for m in (0.0, 0.1, 0.9):
        rm = PS.persistence_hat(y1, y2, others, su, cl, margin=m)
        assert rm["lambda_hat"] == r["lambda_hat"] and rm["margin"] == m
        assert rm["lambda_used"] == pytest.approx(min(1.0, r["lambda_hat"] + m))
    # the clip edges on real estimates: identical seeds (lambda_hat 1) and a strongly negative lambda_hat
    ya, yb = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0], [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
    z = [1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
    assert PS.persistence_hat(z, z, [ya, yb], 0.03)["lambda_used"] == 1.0
    neg = PS.persistence_hat([1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [[1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0]], 0.03)
    assert neg["lambda_hat"] < -0.03 and neg["lambda_used"] == 0.0
    # persistence_check uses the same rule: bias_used includes the margin
    pb, scl = synth_p()
    c0 = PS.persistence_check(pb["100"], scl, 0.0, 0.03, 40, seed=2, margin=0.0)
    c3 = PS.persistence_check(pb["100"], scl, 0.0, 0.03, 40, seed=2)
    assert c3["margin"] == 0.03 and c0["margin"] == 0.0 and c3["mean_hat"] == c0["mean_hat"]
    assert c3["bias_used"] == pytest.approx(c3["mean_used"] - 0.0) and c3["mean_used"] > c0["mean_used"]


def _old_lambda(y1, y2, others, su):
    """The wave-2 estimator (H = m/(m-1) mean_i p_i (1 - p_i), half the mean between-start discordance), kept here
    only to show that the start-offset tests below would catch it."""
    rows = [y1] + list(others)
    n, m = len(y1), len(rows)
    p = [sum(r[i] for r in rows) / m for i in range(n)]
    H = m / (m - 1) * sum(x * (1 - x) for x in p) / n
    D = sum((a - b) ** 2 for a, b in zip(y1, y2)) / n
    sd = PS.seed_sd(y1, y2)
    k = PS.shift_gap(sd["seed_diff_mean"], sd["S"], su)
    return (2 * H - D) / (2 * H - k)


def synth_p():
    """The prereg package's synthetic p (500 tasks, 80 scenes; OUT/scratch/orch/r3/wave2/prereg/diag_tiercorr.py)."""
    rng = random.Random(0)
    cl = [f"scene{i % 80:02d}" for i in range(500)]
    base = [min(1, max(0, rng.betavariate(0.6, 1.4))) for _ in range(500)]
    pb = {"100": [round(x * 3) / 3 for x in base],
          "TOP": [min(1.0, round(x * 3) / 3 + 1 / 3 * (rng.random() < 0.3)) for x in base]}
    return pb, cl


@pytest.mark.parametrize("lam", [0.0, 0.5])
def test_persistence_hat_recovers_lambda_in_the_model(lam):
    """Validation at the pilot's n (500 tasks, 80 scenes, three pooled starts, sigma 0.03): data simulated from the
    registered model (persistence_check -> _draw_start_np -> persistence_hat, the run_tvr path); the mean of the
    unclipped lambda_hat is within 0.03 of the true lambda (the registered estimator's model bias is about -0.015 at
    lambda 0 and -0.005 at 0.6, PREREG 12 item 7)."""
    pb, cl = synth_p()
    r = PS.persistence_check(pb["100"], cl, lam, 0.03, 400, seed=1)
    assert r["n_undefined"] == 0 and r["n_tasks"] == 500 and r["n_pooled"] == 3 and r["offsets"] == [0.0] * 3
    assert abs(r["bias"]) <= 0.03, r
    assert r["sd_hat"] < 0.2


def _check_draws(p_rep, lam, sigma, n_reps, seed, offsets, chunk=250):
    """persistence_check's draws (same generator, same order: the replicate start's two seeds, then the other
    starts), as (y1, y2, others) per replicate."""
    tab = PS.shift_table(p_rep)
    tabs = {"R": (np.asarray(tab[0]), np.asarray(tab[1]))}
    pa = {"R": np.asarray(p_rep, dtype=np.float64)}
    rng = np.random.default_rng(seed)
    out, done = [], 0
    while done < n_reps:
        B = min(chunk, n_reps - done)
        rep = PS._draw_start_np(np, rng, B, ["R"], {"R": [0, 1]}, 2, tabs, pa, sigma, 0.0, lam, offsets[0])["R"]
        oth = [PS._draw_start_np(np, rng, B, ["R"], {"R": [0]}, 1, tabs, pa, sigma, 0.0, lam, offsets[j])["R"][:, 0]
               for j in range(1, len(offsets))]
        out += [(rep[b, 0].astype(int).tolist(), rep[b, 1].astype(int).tolist(),
                 [o[b].astype(int).tolist() for o in oth]) for b in range(B)]
        done += B
    return out


def test_persistence_hat_ignores_start_mean_srs():
    """Separation (G13; review stats wave 2b): start main effects -- realized SR offsets of the three pilot starts
    through calibrate_shift, with the replicate start Q3B the best, the worst, or the others shifted together -- do
    not move lambda_hat (persistence_check, the run_tvr estimator path, common random numbers; 300 replicates at the
    pilot's n).  The wave-2 estimator (H = half the mean between-start discordance) on the same draws moves by about
    0.1, so the test would catch it."""
    pb, cl = synth_p()
    p = pb["100"]
    n_reps, lam = 300, 0.0
    base = PS.persistence_check(p, cl, lam, 0.03, n_reps, seed=7)
    old0 = [_old_lambda(y1, y2, o, PS.seed_sd(y1, y2, 0.03)["sigma_used"])
            for y1, y2, o in _check_draws(p, lam, 0.03, n_reps, 7, [0.0, 0.0, 0.0])]
    old0 = sum(old0) / len(old0)
    for off in ((0.05, 0.0, -0.05), (0.05, 0.0, 0.0), (-0.05, 0.0, 0.05), (0.0, 0.05, 0.05)):
        r = PS.persistence_check(p, cl, lam, 0.03, n_reps, seed=7, offsets=off)
        assert r["offsets"] == list(off) and r["n_undefined"] == 0
        assert abs(r["mean_hat"] - base["mean_hat"]) <= 0.03, (off, r["mean_hat"], base["mean_hat"])
        draws = _check_draws(p, lam, 0.03, n_reps, 7, list(off))
        if off[0] > 0:   # _check_draws are persistence_check's own draws: the same estimator gives its mean
            new = [PS.persistence_hat(y1, y2, o, PS.seed_sd(y1, y2, 0.03)["sigma_used"], cl)["lambda_hat"]
                   for y1, y2, o in _check_draws(p, lam, 0.03, 60, 7, list(off))]
            assert sum(new) / 60 == pytest.approx(
                PS.persistence_check(p, cl, lam, 0.03, 60, seed=7, offsets=off)["mean_hat"])
        old = sum(_old_lambda(y1, y2, o, PS.seed_sd(y1, y2, 0.03)["sigma_used"]) for y1, y2, o in draws) / n_reps
        assert abs(old - old0) >= 0.05, (off, old, old0)
    with pytest.raises(ValueError):
        PS.persistence_check(p, cl, lam, 0.03, 4, seed=7, offsets=(0.0, 0.0))


def test_persistence_lowers_power_and_keeps_type1():
    p, cl = toy_p(500, 100)
    des = toy_design()
    kw = dict(backend="numpy", test="seed_aware_endpoint_t", tier_corr=0.5)
    p0 = PS.simulate(p, cl, des, 0.08, 0.03, 300, seed=4, **kw)["power"]
    p8 = PS.simulate(p, cl, des, 0.08, 0.03, 300, seed=4, persistence=0.8, **kw)["power"]
    for k in p0:
        assert p8[k] < p0[k] - 0.05, (k, p0[k], p8[k])
    z = PS.simulate(p, cl, des, 0.0, 0.03, 400, seed=4, persistence=0.8, **kw)
    assert z["fwer"] <= 0.08, z["fwer"]


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
    pw = pilot_cfg.FROZEN["power"]
    assert disk["schema"] == PS.SCHEMA and disk["pass"] == res["pass"]
    assert set(disk["tables"]) == {"main", "fallback_seeds_1_2"}
    lam = disk["persistence"]["lambda_used"]
    for name, rows in disk["tables"].items():
        # sigma rows (used, a priori, sensitivity) x (2 type-I rhos + 3 power deltas + 2 reported rhos), plus the
        # persistence sensitivity (2 lambdas x (2 type-I rhos + 1 power))
        assert len(rows) == 3 * 7 + 2 * 3
        assert {r["sigma_label"] for r in rows} == {"used", "a_priori", "sensitivity"}
        assert {r["delta"] for r in rows} == {0.0, 0.05, 0.08, 0.10}
        assert {r["df"] for r in rows} == ({12} if name == "main" else {6})       # n_starts (n_seeds - 1)
        assert {r["test"] for r in rows} == {"seed_aware_endpoint_t"}
        assert {r["role"] for r in rows} == {"type1", "power", "reported", "sensitivity_persistence"}
        assert {r["tier_corr"] for r in rows if r["role"] == "type1"} == {0.0, 1.0}
        assert {r["tier_corr"] for r in rows if r["role"] == "power"} == {0.5}
        assert {r["tier_corr"] for r in rows if r["role"] == "reported"} == {0.0, 1.0}
        assert {r["persistence"] for r in rows if r["role"] != "sensitivity_persistence"} == {lam}
        assert {r["persistence"] for r in rows if r["role"] == "sensitivity_persistence"} == {0.0, 0.8}
        assert all(set(r["mc_se_power"]) == set(r["power"]) for r in rows)
        v = disk["verdicts"][name]
        used = [r for r in rows if r["sigma_label"] == "used" and r["persistence"] == lam]
        t1 = [r["fwer"] for r in used if r["role"] == "type1"]
        assert v["type1"] == max(t1) and set(v["type1_by_rho"]) == {"0.0", "1.0"}
        prim = [r for r in used if r["role"] == "power" and r["delta"] == 0.08]
        assert len(prim) == 1 and v["power_at_primary"] == prim[0]["power"] and v["power_tier_corr"] == 0.5
        assert set(v["power_by_rho"]) == {"0.0", "0.5", "1.0"}
        assert v["pass"] == (v["type1_ok"] and v["power_ok"])
        assert v["type1_ok"] == (v["type1"] <= 0.06)
    assert disk["fallback_pass"] == disk["verdicts"]["fallback_seeds_1_2"]["pass"]
    s = disk["sigma"]
    assert s["sigma_used"] == max(0.03, s["sigma_hat"]) and s["replicate"]["seed_b"] == 1002
    assert disk["inputs"]["tiers"] == {"100": "100", "TOP": "ALL"}
    assert len(disk["inputs"]["files_sha256"]) == 7
    assert set(disk["p_summary"]) == {"100", "TOP"}            # pooled only; nothing per start
    reg = disk["registered"]
    assert reg["n_sims_used"] == 30 and reg["n_sims_registered"] == 2000
    assert reg["test"] == "seed_aware_endpoint_t" == pw["test"]
    assert reg["tier_corr"] == pw["tier_corr"] and reg["fallback"] == pw["fallback"]
    assert reg["persistence"] == pw["persistence"] and reg["seed_model"] == pw["seed_model"]
    assert reg["type1_definition"].startswith("family-wise")
    v = disk["verdicts"]["main"]
    assert set(v["power_at_primary"]) == {"Q3S - E2", "Q3RD - mean(QRL-s1, QRL-s2)"}
    assert reg["allow_no_replicate"] is False and s["replicate_missing"] is False
    assert reg["backend"] == "numpy" == pw["backend"] and {r["backend"] for r in disk["tables"]["main"]} == {"numpy"}
    assert disk["schema"] == "vpd.power_sim.tvr/3"
    # persistence: the replicate pair against the other two pilot starts' tier-100 rows (covariances only); the
    # reported simulation check ran; only the whitelisted keys (no level / V / B / R / D / kappa / m2)
    pe = disk["persistence"]
    assert pe["replicate_missing"] is False and pe["n"] == 40 and pe["n_pooled"] == 3
    assert pe["lambda_used"] == (1.0 if pe["lambda_hat"] is None else
                                 min(1.0, max(0.0, pe["lambda_hat"] + pw["persistence"]["margin"])))
    assert pe["check"]["gated"] is False and pe["check"]["result"]["n_reps"] == pw["persistence"]["check_reps"]
    assert set(pe) <= PS.PERSISTENCE_KEYS and pe["margin"] == pw["persistence"]["margin"]
    y1, y2, others, cl = _rep_and_others(tmp_path / "pilot")
    want = PS.persistence_hat(y1, y2, others, s["sigma_used"], cl)
    for k in ("n", "n_pooled", "sigma_used", "range", "lambda_hat", "se", "margin", "lambda_used"):
        assert pe[k] == want[k], k                  # json round-trips floats exactly
    assert disk["reported_designs"]["gated"] is False and len(disk["reported_designs"]["rows"]) == 3
    # provenance names: the json sha of the cfg used and the pilot_cfg.py file sha (GATE.json's pilot_cfg_sha256)
    assert disk["inputs"]["pilot_cfg_json_sha256"] == PS.hashlib.sha256(pilot_cfg.as_json(_cfg()).encode()).hexdigest()
    assert disk["inputs"]["pilot_cfg_py_sha256"] == PS.sha256_file(pilot_cfg.__file__)
    assert "pilot_cfg_sha256" not in disk["inputs"]
    # the superseded test: reported on both designs at rho 0 and 1, never gated
    sens = disk["sensitivity_superseded_test"]
    assert sens["gated"] is False and sens["registered"] is False and sens["test"] == "seed_aware_cluster_t"
    assert [(x["design"], x["tier_corr"]) for x in sens["summary"]] == [
        ("main", 0.0), ("main", 1.0), ("fallback_seeds_1_2", 0.0), ("fallback_seeds_1_2", 1.0)]
    assert [x["df"] for x in sens["summary"]] == [24, 24, 12, 12]
    assert all(x["sigma"] == s["sigma_used"] and x["persistence"] == lam for x in sens["summary"])
    assert "sensitivity_tier_corr" not in disk
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


def _rep_and_others(root):
    """(y1, y2, [Q3S row, E2 row], scenes) at tier 100, read from a written pilot root file by file."""
    rows = {}
    for S, tier in (("Q3B", "100"), ("Q3B", "100s1002"), ("Q3S", "100"), ("E2", "100")):
        rows[(S, tier)] = PS.read_per_task(PS.eval_dir(root, S, tier) / "per_task.jsonl")
    ids = sorted(rows[("Q3B", "100")])
    col = {k: [v[t]["success"] for t in ids] for k, v in rows.items()}
    return (col[("Q3B", "100")], col[("Q3B", "100s1002")], [col[("Q3S", "100")], col[("E2", "100")]],
            [rows[("Q3B", "100")][t]["scene"] for t in ids])


def _write_rows(root, rows: dict, scenes: list):
    """per_task.jsonl files from explicit 0/1 rows {(start, tier label): row}."""
    ids = [f"task_{i:04d}" for i in range(len(scenes))]
    for (S, tier), row in rows.items():
        d = PS.eval_dir(root, S, tier)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "per_task.jsonl", "w") as f:
            for t, sc, y in zip(ids, scenes, row):
                f.write(json.dumps({"task_id": t, "category": "SR-easy", "scene": sc, "success": int(y)}) + "\n")


def _fake_sim(calls, fwer_at, power_at):
    """A stand-in for simulate that records (design seeds, delta, sigma, tier_corr, persistence, test) and returns a
    result whose fwer / power depend on (tier_corr, persistence) through the given maps (default 0.0 / 0.9)."""
    def sim(p_by_tier, clusters, design, delta, sigma, n_sims, seed, **kw):
        rho, lam = kw.get("tier_corr", 0.0), kw.get("persistence", 0.0)
        calls.append({"seeds": copy.deepcopy(design["seeds"]), "delta": delta, "sigma": sigma, "tier_corr": rho,
                      "persistence": lam, "test": kw.get("test")})
        names = [PS.contrast_name(c[0], c[1]) for c in design["contrasts"]]
        pv = power_at.get(rho, 0.9) if delta else 0.0
        return {"delta": delta, "sigma": sigma, "n_sims": n_sims, "test": kw.get("test"), "tier_corr": float(rho),
                "persistence": float(lam), "df": 12, "fwer": fwer_at.get(rho, 0.0) if delta == 0 else None,
                "power": {k: pv for k in names}, "mc_se_power": {k: 0.0 for k in names}}
    return sim


def test_verdict_uses_registered_rhos(tmp_path, monkeypatch):
    """The verdict reads exactly the registered cells: type-I = max FWER over tier_corr.type1 (0 and 1) at sigma_used
    and lambda_used; power from (primary delta, tier_corr.power = 0.5); rho 0 / 1 power only reported."""
    _write_pilot(tmp_path / "p")
    calls = []
    monkeypatch.setattr(PS, "simulate", _fake_sim(calls, fwer_at={0.0: 0.01, 1.0: 0.05}, power_at={0.5: 0.85,
                                                                                                  0.0: 0.95, 1.0: 0.5}))
    res = PS.run_tvr(tmp_path / "p", cfg=_cfg(), n_sims=7)
    v = res["verdicts"]["main"]
    assert v["type1"] == 0.05 and v["type1_by_rho"] == {"0.0": 0.01, "1.0": 0.05} and v["type1_ok"] is True
    assert set(v["power_at_primary"].values()) == {0.85} and v["power_ok"] is True and v["pass"] is True
    names = list(v["power_at_primary"])
    assert v["power_by_rho"] == {"0.0": {k: 0.95 for k in names}, "0.5": {k: 0.85 for k in names},
                                 "1.0": {k: 0.5 for k in names}}
    lam = res["persistence"]["lambda_used"]
    gated = [c for c in calls if c["sigma"] == res["sigma"]["sigma_used"] and c["persistence"] == lam
             and c["test"] == "seed_aware_endpoint_t"]
    assert {(c["delta"], c["tier_corr"]) for c in gated} >= {(0.0, 0.0), (0.0, 1.0), (0.08, 0.5), (0.08, 0.0),
                                                             (0.08, 1.0), (0.05, 0.5), (0.10, 0.5)}
    assert not any(c["delta"] == 0 and c["tier_corr"] == 0.5 for c in gated)   # the null is never judged at 0.5
    # FWER above type1_max at rho 1 only -> type-I fails although rho 0 is fine
    calls.clear()
    monkeypatch.setattr(PS, "simulate", _fake_sim(calls, fwer_at={0.0: 0.01, 1.0: 0.07}, power_at={0.5: 0.85}))
    v = PS.run_tvr(tmp_path / "p", cfg=_cfg(), n_sims=7)["verdicts"]["main"]
    assert v["type1"] == 0.07 and v["type1_ok"] is False and v["pass"] is False
    # power fails at rho 0.5 although it passes at rho 0 -> power fails
    monkeypatch.setattr(PS, "simulate", _fake_sim(calls, fwer_at={}, power_at={0.5: 0.79, 0.0: 0.99, 1.0: 0.99}))
    res = PS.run_tvr(tmp_path / "p", cfg=_cfg(), n_sims=7)
    assert res["verdicts"]["main"]["power_ok"] is False and res["pass"] is False and res["fallback_pass"] is False


def test_run_tvr_fallback_paired(tmp_path, monkeypatch):
    """The registered fallback is seeds 1-2 at BOTH tiers (paired), so the registered endpoint test runs on it
    (the old TOP-only fallback was unpaired and made simulate raise)."""
    _write_pilot(tmp_path / "p")
    seen = []
    real = PS.simulate

    def spy(p_by_tier, clusters, design, *a, **kw):
        seen.append(copy.deepcopy(design["seeds"]))
        return real(p_by_tier, clusters, design, *a, **kw)

    monkeypatch.setattr(PS, "simulate", spy)
    res = PS.run_tvr(tmp_path / "p", cfg=_cfg(), n_sims=10, backend="numpy")
    assert pilot_cfg.FROZEN["power"]["fallback"]["seeds"] == {"100": [1, 2], "TOP": [1, 2]}
    assert {"100": [1, 2], "TOP": [1, 2]} in seen and {"100": [1, 2, 3], "TOP": [1, 2, 3]} in seen
    # plus the report-only designs (4 and 5 seeds per start, paired; reported_designs)
    rep_seeds = [d["seeds"] for d in pilot_cfg.FROZEN["power"]["reported_designs"]["designs"].values()]
    assert {"100": [1, 2, 3, 4], "TOP": [1, 2, 3, 4]} in seen
    assert {"100": [1, 2, 3, 4, 5], "TOP": [1, 2, 3, 4, 5]} in seen
    assert all(s in [{"100": [1, 2], "TOP": [1, 2]}, {"100": [1, 2, 3], "TOP": [1, 2, 3]}] + rep_seeds for s in seen)
    assert {r["df"] for r in res["tables"]["fallback_seeds_1_2"]} == {6}
    assert isinstance(res["fallback_pass"], bool)
    unpaired = toy_design()
    unpaired["seeds"]["TOP"] = [1, 2]
    p, cl = toy_p(20, 4)
    with pytest.raises(ValueError):
        real(p, cl, unpaired, 0.0, 0.03, 5, seed=5, test="seed_aware_endpoint_t")


def test_run_tvr_ignores_differences_between_starts(tmp_path):
    """Separation (brief section 3): swapping outcomes between two pooled starts that are not the replicate start
    (same pooled p_i, same replicate) leaves lambda, sigma and every table and verdict of POWER.json unchanged."""
    _write_pilot(tmp_path / "a")
    a = PS.run_tvr(tmp_path / "a", cfg=_cfg(), n_sims=20, backend="numpy")
    for tier in ("100", "ALL"):
        fq, fe = (PS.eval_dir(tmp_path / "a", s, tier) / "per_task.jsonl" for s in ("Q3S", "E2"))
        q, e = fq.read_text(), fe.read_text()
        fq.write_text(e)
        fe.write_text(q)
    b = PS.run_tvr(tmp_path / "a", cfg=_cfg(), n_sims=20, backend="numpy")
    assert a["inputs"]["files_sha256"] != b["inputs"]["files_sha256"]
    for key in ("sigma", "persistence", "tables", "verdicts", "sensitivity_superseded_test", "p_summary", "pass",
                "fallback_pass", "reported_designs"):
        assert a[key] == b[key], key


def _coupled_root(root, better: float, n=2000, n_scenes=80, seed=3):
    """A pilot root drawn task by task from one uniform per (run, task) (lambda 0): p_i ~ Beta(0.6, 1.4) at tier 100
    (+0.1 at ALL); E2's tier-100 row thresholds the SAME uniforms at clip(p_i + better), so `better` > 0 only turns
    some of E2's failures into successes (an additive start effect); every other file is identical."""
    rng = np.random.default_rng(seed)
    p = rng.beta(0.6, 1.4, size=n)
    scenes = [f"scene{i % n_scenes:02d}" for i in range(n)]
    rows = {}
    for S, tier, pt in (("Q3B", "100", p), ("Q3B", "100s1002", p), ("Q3S", "100", p), ("E2", "100", p),
                        ("Q3B", "ALL", p + 0.1), ("Q3S", "ALL", p + 0.1), ("E2", "ALL", p + 0.1)):
        u = rng.random(n)
        q = np.clip(pt + (better if (S, tier) == ("E2", "100") else 0.0), 0.0, 1.0)
        rows[(S, tier)] = (u < q).astype(int).tolist()
    _write_rows(root, rows, scenes)
    return rows


def test_run_tvr_lambda_ignores_a_better_start(tmp_path):
    """Separation on the real run_tvr path (review stats wave 2b): one NON-replicate pilot start (E2) solves about
    6 % more tier-100 tasks (an additive start effect: the same uniforms thresholded higher), the Q3B 100 and
    100s1002 files identical -> persistence.lambda_hat moves by less than 0.05, while the wave-2 estimator on the
    same files moves up by more than 0.04.  (This construction over 20 seeds: the change of lambda_hat has mean
    -0.005 and SD 0.017, max |change| 0.040; the wave-2 estimator's mean +0.064, min +0.032.)  POWER.json's
    persistence block is persistence_hat on the files (Q3B pair against the other two starts)."""
    cfg = _cfg()
    cfg["power"]["persistence"]["check_reps"] = 2         # the reported check is not under test here
    res, files = {}, {}
    for name, better in (("base", 0.0), ("better", 0.06)):
        rows = _coupled_root(tmp_path / name, better)
        res[name] = PS.run_tvr(tmp_path / name, cfg=cfg, n_sims=5)
        files[name] = rows
    assert files["base"][("Q3B", "100")] == files["better"][("Q3B", "100")]
    assert files["base"][("Q3B", "100s1002")] == files["better"][("Q3B", "100s1002")]
    gain = (sum(files["better"][("E2", "100")]) - sum(files["base"][("E2", "100")])) / 2000
    assert 0.04 <= gain <= 0.07, gain
    assert all(b >= a for a, b in zip(files["base"][("E2", "100")], files["better"][("E2", "100")]))
    for name in res:
        y1, y2, others, cl = _rep_and_others(tmp_path / name)
        want = PS.persistence_hat(y1, y2, others, res[name]["sigma"]["sigma_used"], cl)
        assert res[name]["persistence"]["lambda_hat"] == want["lambda_hat"] is not None
        nopath = {k: v for k, v in res[name]["sigma"].items() if k != "replicate"}   # replicate: the file paths
        assert nopath == {k: v for k, v in res["base"]["sigma"].items() if k != "replicate"}   # the pair unchanged
        assert res[name]["registered"]["backend"] == "numpy"
    new = [res[k]["persistence"]["lambda_hat"] for k in ("base", "better")]
    assert abs(new[1] - new[0]) < 0.05, new
    old = []
    for name in ("base", "better"):
        y1, y2, others, _ = _rep_and_others(tmp_path / name)
        old.append(_old_lambda(y1, y2, others, res[name]["sigma"]["sigma_used"]))
    assert old[1] - old[0] > 0.04, old


def test_run_tvr_backend_is_registered(tmp_path, monkeypatch):
    """power.backend (numpy) is the default; an explicit backend is a recorded test override (registered.backend;
    A4 counts only power.backend); when the registered backend cannot run, run_tvr refuses with ValueError (which
    gate_tvr turns into A4 pending) instead of switching backends."""
    assert pilot_cfg.FROZEN["power"]["backend"] == "numpy"
    _write_pilot(tmp_path / "p")
    a = PS.run_tvr(tmp_path / "p", cfg=_cfg(), n_sims=5)
    assert a["registered"]["backend"] == "numpy" and {r["backend"] for r in a["tables"]["main"]} == {"numpy"}
    assert a["persistence"]["check"]["result"] is not None
    b = PS.run_tvr(tmp_path / "p", cfg=_cfg(), n_sims=5, backend="python")
    assert b["registered"]["backend"] == "python" and {r["backend"] for r in b["tables"]["main"]} == {"python"}
    assert b["persistence"]["check"]["result"] is None and b["persistence"]["lambda_hat"] == \
        a["persistence"]["lambda_hat"]
    assert PS.run_backend({"backend": "numpy"}) == "numpy" and PS.run_backend({}, "python") == "python"
    monkeypatch.setattr(PS, "_np", lambda: None)
    with pytest.raises(ValueError, match="numpy"):
        PS.run_backend({"backend": "numpy"})
    with pytest.raises(ValueError, match="numpy"):
        PS.run_tvr(tmp_path / "p", cfg=_cfg(), n_sims=5)
    assert PS.run_backend({"backend": "numpy"}, "auto") == "python"     # only an explicit override may switch


def test_run_tvr_replicate_required_and_refusals(tmp_path):
    _write_pilot(tmp_path / "p", replicate=False)
    # the replicate is required (O4); an explicit opt-out falls back to the a priori sigma and the conservative
    # lambda (upper end of the registered range) and says so
    with pytest.raises(ValueError, match="replicate"):
        PS.run_tvr(tmp_path / "p", cfg=_cfg(), n_sims=5, backend="numpy")
    res = PS.run_tvr(tmp_path / "p", cfg=_cfg(), n_sims=5, backend="numpy", allow_no_replicate=True)
    assert res["sigma"]["replicate_missing"] is True and res["sigma"]["sigma_used"] == 0.03
    assert res["registered"]["allow_no_replicate"] is True
    assert res["persistence"]["replicate_missing"] is True and res["persistence"]["lambda_hat"] is None
    assert res["persistence"]["lambda_used"] == pilot_cfg.FROZEN["power"]["persistence"]["range"][1]
    assert res["persistence"]["margin"] == 0.03 and set(res["persistence"]) <= PS.PERSISTENCE_KEYS
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


# --------------------------------------------------------------------------------------------- wave 2c
PERSISTENCE_WHITELIST = {"replicate_missing", "lambda_hat", "se", "lambda_used", "margin", "n", "n_pooled", "range",
                         "rule", "sigma_used", "check"}          # + "note" when lambda_hat is undefined / no replicate
CHECK_RESULT_WHITELIST = {"lambda_true", "sigma", "n_reps", "n_pooled", "offsets", "n_tasks", "seed", "margin",
                          "mean_hat", "bias", "sd_hat", "mc_se", "mean_se", "mean_used", "bias_used", "n_undefined"}
LEVEL_KEYS = ("level", "V", "V_plain", "B", "R", "D", "kappa", "m2")


def test_run_tvr_applies_the_registered_margin(tmp_path):
    """R2 (wave 2c): POWER.json's lambda_used = clip(lambda_hat + power.persistence.margin, range), recorded with the
    margin and the rule; every gated and reported row runs at that lambda_used; the reported check of the estimator
    runs at it and its bias_used includes the margin; the cfg's margin is the one used and registered.persistence
    records it (gate_tvr.power_unregistered compares registered.persistence with power.persistence, both from cfg)."""
    _write_pilot(tmp_path / "p")
    cfg = _cfg()
    cfg["power"]["persistence"]["check_reps"] = 3
    res = PS.run_tvr(tmp_path / "p", cfg=cfg, n_sims=5, backend="numpy")
    pe = res["persistence"]
    assert pe["margin"] == 0.03 and 0.0 < pe["lambda_hat"] < 0.97
    assert pe["lambda_used"] == pytest.approx(pe["lambda_hat"] + 0.03)
    assert pe["lambda_used"] == PS.lambda_used_rule(pe["lambda_hat"], 0.03, pe["range"])
    assert pe["rule"] == cfg["power"]["persistence"]["rule"] and "clip(lambda_hat + margin, range)" in pe["rule"]
    lam = pe["lambda_used"]
    assert {r["persistence"] for t in res["tables"].values() for r in t if r["role"] != "sensitivity_persistence"} \
        == {lam}
    assert res["verdicts"]["main"]["persistence"] == lam == res["reported_designs"]["persistence"]
    chk = pe["check"]["result"]
    assert chk["lambda_true"] == lam and chk["margin"] == 0.03
    assert chk["bias_used"] == pytest.approx(chk["mean_used"] - lam)
    assert res["registered"]["persistence"] == cfg["power"]["persistence"]
    # gate_tvr.power_unregistered: registered.persistence (margin included) against cfg power.persistence
    from vpd.tvr import gate_tvr
    want = "registered.persistence != power.persistence"
    assert want not in gate_tvr.power_unregistered(json.loads(json.dumps(res)), cfg)
    bad = json.loads(json.dumps(res))
    bad["registered"]["persistence"]["margin"] = 0.0
    assert want in gate_tvr.power_unregistered(bad, cfg)
    cfg2 = copy.deepcopy(cfg)
    cfg2["power"]["persistence"]["margin"] = 0.1
    res2 = PS.run_tvr(tmp_path / "p", cfg=cfg2, n_sims=5, backend="numpy")
    assert res2["persistence"]["lambda_hat"] == pe["lambda_hat"]
    assert res2["persistence"]["lambda_used"] == pytest.approx(pe["lambda_hat"] + 0.1)
    assert res2["persistence"]["margin"] == 0.1 == res2["registered"]["persistence"]["margin"]
    # the cfg's margin also reaches the reported check (_persistence passes it to persistence_check; without it the
    # check would fall back to the registered 0.03): its result records 0.1 and bias_used is the margin-0.1 one
    chk2 = res2["persistence"]["check"]["result"]
    lam2 = res2["persistence"]["lambda_used"]
    assert chk2["margin"] == 0.1 and chk2["lambda_true"] == lam2
    assert chk2["bias_used"] == pytest.approx(chk2["mean_used"] - lam2)
    p_by_tier, scl, _, _ = PS.pooled_p(tmp_path / "p", list(cfg["starts"]), {"100": "100", "TOP": "ALL"})
    kw = dict(n_pooled=3, a_priori=0.03, lam_range=(0.0, 1.0))
    sig2, reps2, seed2 = res2["sigma"]["sigma_used"], cfg2["power"]["persistence"]["check_reps"], cfg2["power"]["seed"]
    want01 = PS.persistence_check(p_by_tier["100"], scl, lam2, sig2, reps2, seed2, margin=0.1, **kw)
    want003 = PS.persistence_check(p_by_tier["100"], scl, lam2, sig2, reps2, seed2, margin=0.03, **kw)
    assert chk2 == want01
    assert want003["mean_hat"] == want01["mean_hat"] and want003["bias_used"] != want01["bias_used"]
    assert chk2["bias_used"] == want01["bias_used"] != want003["bias_used"]
    # undefined lambda_hat (fake pilot seed 1: a covariance <= 0) -> the range's upper end, no margin arithmetic
    _write_pilot(tmp_path / "u", seed=1)
    u = PS.run_tvr(tmp_path / "u", cfg=cfg, n_sims=5, backend="numpy")["persistence"]
    assert u["lambda_hat"] is None and u["lambda_used"] == 1.0 and u["margin"] == 0.03 and "note" in u


def test_power_json_persistence_keys_are_whitelisted(tmp_path):
    """R1 (wave 2c): POWER.json["persistence"] holds exactly the whitelisted keys -- none of persistence_hat's
    intermediates (level = the replicate start's SR, V / V_plain rebuild it, B / R / D / kappa / m2 follow from
    lambda_hat and the sigma block) -- on the
    estimated, the undefined and the no-replicate paths; its check block holds only simulation results at the
    pooled p; run_tvr refuses to emit anything else."""
    _write_pilot(tmp_path / "p")
    cfg = _cfg()
    cfg["power"]["persistence"]["check_reps"] = 3
    out = tmp_path / "POWER.json"
    PS.run_tvr(tmp_path / "p", cfg=cfg, out=out, n_sims=5, backend="numpy")
    pe = json.loads(out.read_text(encoding="utf-8"))["persistence"]
    assert set(pe) == PERSISTENCE_WHITELIST
    assert not any(k in pe for k in LEVEL_KEYS)
    assert set(pe["check"]) == {"gated", "note", "result"} and set(pe["check"]["result"]) == CHECK_RESULT_WHITELIST
    assert PS.PERSISTENCE_KEYS == PERSISTENCE_WHITELIST | {"note"}
    assert PS.CHECK_RESULT_KEYS == CHECK_RESULT_WHITELIST
    _write_pilot(tmp_path / "u", seed=1)                                   # lambda_hat undefined
    pu = PS.run_tvr(tmp_path / "u", cfg=cfg, n_sims=5, backend="numpy")["persistence"]
    assert set(pu) == PERSISTENCE_WHITELIST | {"note"}
    pp = PS.run_tvr(tmp_path / "p", cfg=cfg, n_sims=5, backend="python")["persistence"]   # no check run
    assert set(pp) == PERSISTENCE_WHITELIST and pp["check"]["result"] is None
    _write_pilot(tmp_path / "norep", replicate=False)
    pn = PS.run_tvr(tmp_path / "norep", cfg=cfg, n_sims=5, backend="numpy", allow_no_replicate=True)["persistence"]
    assert set(pn) <= PERSISTENCE_WHITELIST | {"note"} and pn["lambda_used"] == 1.0
    for k in LEVEL_KEYS:
        with pytest.raises(ValueError, match="whitelist"):
            PS._public_persistence(dict(pe, **{k: 0.3}))
    with pytest.raises(ValueError, match="whitelist"):
        PS._public_persistence(dict(pe, check=dict(pe["check"], result=dict(pe["check"]["result"], level=0.3))))


def _leak_root(root, n=137, n_scenes=12, seed=21):
    """A pilot root with distinct start SRs at both tiers (Q3S about +12 pp and E2 about -6 pp against Q3B, additive
    on one task pattern), Q3B's replicate an independent draw.  n is prime, so no start SR (k / n), pair level or
    difference equals a power fraction (k / 23) or a registered constant."""
    rng = random.Random(seed)
    base = [rng.betavariate(0.6, 1.4) for _ in range(n)]
    scenes = [f"scene{i % n_scenes:02d}" for i in range(n)]
    rows = {}
    for S, a in (("Q3B", 0.0), ("Q3S", 0.12), ("E2", -0.06)):
        for tier, extra in (("100", 0.0), ("ALL", 0.1)):
            rows[(S, tier)] = [int(rng.random() < min(1.0, max(0.0, p + a + extra))) for p in base]
    rows[("Q3B", "100s1002")] = [int(rng.random() < p) for p in base]
    _write_rows(root, rows, scenes)
    return rows


def _statcheck_routes(pj):
    """The wave-2b statcheck reconstruction (OUT/scratch/orch/r3/wave2b/statcheck/scripts/sc_diag.py, cmd_leak): two
    closed-form routes from a POWER.json alone to SR(Q3B, 100, s1001) - mean(SR(Q3S, 100), SR(E2, 100)).  Each
    raises KeyError when POWER.json lacks a value it needs."""
    pers, sig, ps = pj["persistence"], pj["sigma"], pj["p_summary"]

    def via_level():
        sr = pers["level"] + sig["seed_diff_mean"] / 2.0
        return sr - (3.0 * ps["100"]["mean"] - sr) / 2.0

    def via_v():        # without level: L (1 - L) = B + V, and B follows from lambda_hat and the sigma block
        n = sig["n"]
        D = (n - 1) * sig["S"] + sig["seed_diff_mean"] ** 2
        kap = PS.shift_gap(sig["seed_diff_mean"], sig["S"], sig["sigma_used"])
        m2 = PS.shift_m2(sig["seed_diff_mean"], sig["S"], sig["sigma_used"])
        lam = pers["lambda_hat"]
        B = ((D - lam * kap) / (1.0 - lam) - m2 / 2.0) / 2.0
        LL = B + pers["V"]
        sr = (1.0 - math.sqrt(max(0.0, 1.0 - 4.0 * LL))) / 2.0 + sig["seed_diff_mean"] / 2.0
        return sr - (3.0 * ps["100"]["mean"] - sr) / 2.0

    return via_level, via_v


def _numbers(o, path=()):
    if isinstance(o, dict):
        for k, x in o.items():
            yield from _numbers(x, path + (k,))
    elif isinstance(o, list):
        for x in o:
            yield from _numbers(x, path)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield path, float(o)


def test_power_json_gives_no_start_level_and_no_between_start_difference(tmp_path):
    """R1 (wave 2c), on the real run_tvr path (a pilot root with distinct start SRs): the wave-2b statcheck
    reconstruction of SR(Q3B, 100, s1001) - mean(SR(Q3S, 100), SR(E2, 100)) cannot be evaluated from POWER.json
    (both routes need a value it no longer has; fed persistence_hat's in-memory level or V they give the true
    difference, so the routes are live), and no number in POWER.json outside the registered constants and the input
    hashes equals a start's SR (any tier or seed), the replicate pair's level, or a between-start difference."""
    root = tmp_path / "p"
    rows = _leak_root(root)
    n = len(rows[("Q3B", "100")])
    sr = {k: sum(v) / n for k, v in rows.items()}
    target = sr[("Q3B", "100")] - (sr[("Q3S", "100")] + sr[("E2", "100")]) / 2.0
    cfg = _cfg()
    cfg["power"]["persistence"]["check_reps"] = 5
    out = tmp_path / "POWER.json"
    PS.run_tvr(root, cfg=cfg, out=out, n_sims=23, backend="numpy")
    disk = json.loads(out.read_text(encoding="utf-8"))
    assert disk["persistence"]["lambda_hat"] is not None and abs(target) > 0.01
    via_level, via_v = _statcheck_routes(disk)
    for route in (via_level, via_v):
        with pytest.raises(KeyError):
            route()
    y1, y2, others, cl = _rep_and_others(root)
    full = PS.persistence_hat(y1, y2, others, disk["sigma"]["sigma_used"], cl)
    assert full["lambda_hat"] == disk["persistence"]["lambda_hat"] and full["level"] < 0.5
    leaky = {k: dict(disk, persistence=dict(disk["persistence"], **{k: full[k]})) for k in ("level", "V")}
    assert _statcheck_routes(leaky["level"])[0]() == pytest.approx(target, abs=1e-9)
    assert _statcheck_routes(leaky["V"])[1]() == pytest.approx(target, abs=1e-9)
    # no start level and no between-start difference anywhere outside registered.* (constants) and inputs.*
    forbidden = set(sr.values()) | {full["level"]}
    for tier in ("100", "ALL"):
        lv = {("Q3B", "s1001"): sr[("Q3B", tier)], ("Q3S", ""): sr[("Q3S", tier)], ("E2", ""): sr[("E2", tier)]}
        if tier == "100":
            lv[("Q3B", "s1002")] = sr[("Q3B", "100s1002")]
        for a, va in lv.items():
            for b, vb in lv.items():
                if a[0] != b[0]:                         # Q3B seed 1001 vs 1002 is within-start (sigma block)
                    forbidden.add(va - vb)
            rest = [lv[(S, "")] if S != "Q3B" else lv[("Q3B", "s1001")] for S in ("Q3B", "Q3S", "E2") if S != a[0]]
            forbidden |= {va - sum(rest) / 2.0, sum(rest) / 2.0 - va}
    forbidden.discard(0.0)
    assert target in forbidden
    # scanned: every number but the registered constants, the input hashes, the simulation blocks (simulated starts
    # drawn from the pooled p: their rational values can meet a k / n by chance; covered by the invariance below)
    # and sigma.seed_diff_mean (Q3B seed 1001 minus seed 1002: a within-start difference, reported by design)
    skip = {("registered",), ("inputs",), ("tables",), ("verdicts",), ("sensitivity_superseded_test",),
            ("reported_designs",)}
    nums = [(p, x) for p, x in _numbers(disk) if p[:1] not in skip and p != ("sigma", "seed_diff_mean")]
    assert any(p[:1] == ("persistence",) for p, _ in nums) and any(p[:1] == ("p_summary",) for p, _ in nums)
    hits = [(p, x) for p, x in nums if any(abs(x - v) < 1e-12 for v in forbidden)]
    assert not hits, hits[:5]
    # the simulation blocks (and everything else) see the starts only through the pooled p: swapping Q3B's and
    # Q3S's top-tier files changes Q3B - mean(others) at TOP and leaves POWER.json unchanged but for its inputs
    fb, fs = (PS.eval_dir(root, s, "ALL") / "per_task.jsonl" for s in ("Q3B", "Q3S"))
    tb, ts = fb.read_text(), fs.read_text()
    fb.write_text(ts)
    fs.write_text(tb)
    assert sum(rows[("Q3B", "ALL")]) != sum(rows[("Q3S", "ALL")])
    swapped = PS.run_tvr(root, cfg=cfg, n_sims=23, backend="numpy")
    for key in set(disk) - {"utc", "inputs"}:
        assert json.loads(json.dumps(swapped[key])) == disk[key], key


def test_power_sim_docstrings_state_the_exact_separation_property():
    """R1 (wave 2c): no power_sim docstring repeats the false wave-2b claim that POWER.json has nothing per start
    (it reports Q3B's within-start seed-difference statistics and lambda_hat); the module docstring and
    persistence_hat state the exact property: no start's SR level and no between-start difference."""
    docs = {"module": PS.__doc__, "persistence_hat": PS.persistence_hat.__doc__, "run_tvr": PS.run_tvr.__doc__,
            "_persistence": PS._persistence.__doc__, "_public_persistence": PS._public_persistence.__doc__}
    flat = {k: " ".join((v or "").split()) for k, v in docs.items()}
    for k, d in flat.items():
        for bad in ("no per-start value", "nothing per start"):      # wave 2b: "Nothing per start is reported"
            assert bad not in d.lower(), (k, bad)
    assert "it gives no start's SR level" in flat["module"] and "seed_diff_mean" in flat["module"]
    assert "POWER.json gives no start's SR level and no between-start difference" in flat["persistence_hat"]


def test_reported_designs_rows_and_unchanged_verdicts(tmp_path, monkeypatch):
    """R3 (wave 2c): POWER.json reported_designs (role "reported", never gated): Holm power of both contrasts with
    its MC SE for the main design with 4 and 5 seeds per start at both tiers (delta 0.08; df 18, 24) and with 3
    seeds at delta 0.10 (df 12; the main table's own cell), at sigma_used, lambda_used, tier_corr 0.5.  Without them
    every table, verdict and pass is identical, and a reported design with power 0 does not fail A4."""
    _write_pilot(tmp_path / "p")
    cfg = _cfg()
    cfg["power"]["persistence"]["check_reps"] = 3
    reg = pilot_cfg.FROZEN["power"]["reported_designs"]
    assert reg["tier_corr"] == 0.5 and list(reg["designs"]) == ["main_seeds_4", "main_seeds_5", "main_delta_10pp"]
    res = PS.run_tvr(tmp_path / "p", cfg=cfg, n_sims=30, backend="numpy")
    rd = res["reported_designs"]
    assert rd["gated"] is False and rd["role"] == "reported" and res["registered"]["reported_designs"] == reg
    assert rd["tier_corr"] == 0.5 and rd["sigma"] == res["sigma"]["sigma_used"]
    assert rd["persistence"] == res["persistence"]["lambda_used"]
    summ = {s["design"]: s for s in rd["summary"]}
    assert [(summ[k]["seeds_per_start"], summ[k]["delta"], summ[k]["df"]) for k in reg["designs"]] == \
        [(4, 0.08, 18), (5, 0.08, 24), (3, 0.10, 12)]          # df = n_starts (n_seeds - 1), 6 starts
    names = {PS.contrast_name(c[0], c[1]) for c in pilot_cfg.FROZEN["power"]["design"]["contrasts"]}
    for s in rd["summary"]:
        assert s["tier_corr"] == 0.5 and s["sigma"] == res["sigma"]["sigma_used"]
        assert s["persistence"] == res["persistence"]["lambda_used"] and s["test"] == "seed_aware_endpoint_t"
        assert set(s["power"]) == names == set(s["mc_se_power"]) and s["n_sims"] == 30
        assert all(s["mc_se_power"][k] == pytest.approx(math.sqrt(v * (1 - v) / 30)) for k, v in s["power"].items())
    assert [r["design"] for r in rd["rows"]] == list(reg["designs"]) and {r["role"] for r in rd["rows"]} == {"reported"}
    cell = [r for r in res["tables"]["main"] if r["role"] == "power" and r["delta"] == 0.10
            and r["sigma_label"] == "used" and r["tier_corr"] == 0.5]
    assert len(cell) == 1 and summ["main_delta_10pp"]["power"] == cell[0]["power"]
    # never gated: without the reported designs nothing else in POWER.json changes
    cfg0 = copy.deepcopy(cfg)
    cfg0["power"]["reported_designs"]["designs"] = {}
    cfg0["power"]["reported_designs"]["type1"]["designs"] = []
    res0 = PS.run_tvr(tmp_path / "p", cfg=cfg0, n_sims=30, backend="numpy")
    assert res0["reported_designs"]["summary"] == [] and res0["reported_designs"]["rows"] == []
    assert res0["reported_designs"]["type1"]["rows"] == [] and res0["reported_designs"]["type1"]["by_design"] == {}
    for key in ("tables", "verdicts", "pass", "fallback_pass", "sensitivity_superseded_test", "persistence", "sigma",
                "p_summary"):
        assert res0[key] == res[key], key
    # a reported design at power 0 (fake simulate: 0.0 for any seed count but 3 and 2) leaves A4 passing
    calls = []
    fake = _fake_sim(calls, fwer_at={0.0: 0.01, 1.0: 0.02}, power_at={0.5: 0.85, 0.0: 0.9, 1.0: 0.9})

    def sim(p_by_tier, clusters, design, delta, *a, **kw):
        r = fake(p_by_tier, clusters, design, delta, *a, **kw)
        if len(design["seeds"]["100"]) not in (2, 3) and delta:
            r["power"] = {k: 0.0 for k in r["power"]}
        return r

    monkeypatch.setattr(PS, "simulate", sim)
    f = PS.run_tvr(tmp_path / "p", cfg=cfg, n_sims=7)
    assert f["pass"] is True and f["verdicts"]["main"]["power_ok"] is True
    assert [s["power"] for s in f["reported_designs"]["summary"][:2]] == [{k: 0.0 for k in names}] * 2


def test_reported_designs_type1_rows(tmp_path, monkeypatch):
    """Q4 (wave 2c round 2): POWER.json reported_designs.type1 (report only, gated false): for main_seeds_4 and
    main_seeds_5 (power.reported_designs.type1), delta 0 at tier_corr 0 and 1, sigma_used, lambda_used, n_sims and
    the registered seed: FWER over both contrasts (the gated type-I definition, the main design's Holm test) and its
    binomial MC SE, both in [0, 1]; per design the max over the rhos against type1_max.  main_delta_10pp has none
    (at delta 0 it is the main design).  Every verdict equals the one computed without reported_designs, also when a
    reported design's FWER is far above type1_max."""
    _write_pilot(tmp_path / "p")
    cfg = _cfg()
    cfg["power"]["persistence"]["check_reps"] = 3
    reg = pilot_cfg.FROZEN["power"]["reported_designs"]
    assert reg["type1"] == {"designs": ["main_seeds_4", "main_seeds_5"], "tier_corr": [0.0, 1.0]}
    assert set(reg["type1"]["designs"]) < set(reg["designs"]) and "main_delta_10pp" not in reg["type1"]["designs"]
    calls = []
    real = PS.simulate

    def spy(p_by_tier, clusters, design, delta, sigma, n_sims, seed, **kw):
        calls.append({"seeds": copy.deepcopy(design["seeds"]), "delta": delta, "n_sims": n_sims, "seed": seed,
                      "tier_corr": kw.get("tier_corr"), "persistence": kw.get("persistence"), "test": kw.get("test")})
        return real(p_by_tier, clusters, design, delta, sigma, n_sims, seed, **kw)

    monkeypatch.setattr(PS, "simulate", spy)
    out = tmp_path / "POWER.json"
    res = PS.run_tvr(tmp_path / "p", cfg=cfg, out=out, n_sims=40, backend="numpy")
    disk = json.loads(out.read_text(encoding="utf-8"))
    t1 = disk["reported_designs"]["type1"]
    lam, sig = disk["persistence"]["lambda_used"], disk["sigma"]["sigma_used"]
    assert t1["gated"] is False and t1["role"] == "reported" and t1["delta"] == 0.0 and t1["tier_corr"] == [0.0, 1.0]
    assert t1["sigma"] == sig and t1["persistence"] == lam and t1["type1_max"] == 0.06
    assert disk["registered"]["reported_designs"]["type1"] == reg["type1"]
    assert [(s["design"], s["tier_corr"]) for s in t1["summary"]] == [
        ("main_seeds_4", 0.0), ("main_seeds_4", 1.0), ("main_seeds_5", 0.0), ("main_seeds_5", 1.0)]
    assert [s["df"] for s in t1["summary"]] == [18, 18, 24, 24] and [r["design"] for r in t1["rows"]] == \
        [s["design"] for s in t1["summary"]]
    for s, r in zip(t1["summary"], t1["rows"]):
        assert s["delta"] == 0.0 == r["delta"] and s["sigma"] == sig and s["persistence"] == lam
        assert s["test"] == "seed_aware_endpoint_t" and s["n_sims"] == 40 and r["role"] == "reported"
        assert 0.0 <= s["fwer"] <= 1.0 and 0.0 <= s["mc_se_fwer"] <= 1.0 and s["fwer"] == r["fwer"]
        assert s["mc_se_fwer"] == pytest.approx(math.sqrt(s["fwer"] * (1.0 - s["fwer"]) / 40))
        assert set(s["type1_by_contrast"]) == set(disk["verdicts"]["main"]["power_at_primary"])
    for name in reg["type1"]["designs"]:
        rows = [s for s in t1["summary"] if s["design"] == name]
        b = t1["by_design"][name]
        assert b["type1_by_rho"] == {str(s["tier_corr"]): s["fwer"] for s in rows}
        assert b["type1"] == max(s["fwer"] for s in rows) and b["within_type1_max"] == (b["type1"] <= 0.06)
    assert set(t1["by_design"]) == set(reg["type1"]["designs"])
    # the registered n_sims / seed rule of the other rows, and the main design's null cells are not re-drawn
    nulls = [c for c in calls if c["delta"] == 0 and len(c["seeds"]["100"]) in (4, 5)]
    assert sorted((len(c["seeds"]["100"]), c["tier_corr"]) for c in nulls) == [(4, 0.0), (4, 1.0), (5, 0.0), (5, 1.0)]
    assert all(c["n_sims"] == 40 and c["seed"] == cfg["power"]["seed"] and c["persistence"] == lam
               and c["test"] == cfg["power"]["test"] for c in nulls)
    # never gated: without reported_designs every verdict (and every table) is the same
    cfg0 = copy.deepcopy(cfg)
    cfg0["power"]["reported_designs"]["designs"] = {}
    cfg0["power"]["reported_designs"]["type1"]["designs"] = []
    res0 = PS.run_tvr(tmp_path / "p", cfg=cfg0, n_sims=40, backend="numpy")
    assert res0["verdicts"] == res["verdicts"] and res0["tables"] == res["tables"]
    assert (res0["pass"], res0["fallback_pass"]) == (res["pass"], res["fallback_pass"])
    # a reported design with FWER 0.5 (fake simulate) leaves A4 passing; its by_design says it misses type1_max
    fake = _fake_sim([], fwer_at={0.0: 0.01, 1.0: 0.02}, power_at={0.5: 0.85, 0.0: 0.9, 1.0: 0.9})

    def sim(p_by_tier, clusters, design, delta, *a, **kw):
        r = fake(p_by_tier, clusters, design, delta, *a, **kw)
        if len(design["seeds"]["100"]) not in (2, 3) and not delta:
            r["fwer"] = 0.5
        return r

    monkeypatch.setattr(PS, "simulate", sim)
    f = PS.run_tvr(tmp_path / "p", cfg=cfg, n_sims=7)
    f0 = PS.run_tvr(tmp_path / "p", cfg=cfg0, n_sims=7)
    assert f["pass"] is True and f["verdicts"] == f0["verdicts"] and f["verdicts"]["main"]["type1"] == 0.02
    assert {b["type1"] for b in f["reported_designs"]["type1"]["by_design"].values()} == {0.5}
    assert not any(b["within_type1_max"] for b in f["reported_designs"]["type1"]["by_design"].values())
    # a type1 design that is not a reported design is a registration error
    bad = copy.deepcopy(cfg)
    bad["power"]["reported_designs"]["type1"]["designs"] = ["main_seeds_9"]
    with pytest.raises(ValueError, match="reported_designs.type1"):
        PS.run_tvr(tmp_path / "p", cfg=bad, n_sims=7)


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
    # no --backend: the registered power.backend
    PS.main(["tvr", "--pilot-root", str(tmp_path / "p"), "--out", str(tmp_path / "d.json"), "--n-sims", "5"])
    assert json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))["registered"]["backend"] == "numpy"
    capsys.readouterr()
    # no replicate: refused unless --allow-no-replicate
    assert PS.main(["tvr", "--pilot-root", str(tmp_path / "norep"), "--out", str(tmp_path / "c.json"), *argv]) == 2
    assert "replicate" in capsys.readouterr().err and not (tmp_path / "c.json").exists()
    rc = PS.main(["tvr", "--pilot-root", str(tmp_path / "norep"), "--out", str(tmp_path / "c.json"), *argv,
                  "--allow-no-replicate"])
    assert rc in (0, 1) and json.loads((tmp_path / "c.json").read_text())["sigma"]["replicate_missing"] is True
