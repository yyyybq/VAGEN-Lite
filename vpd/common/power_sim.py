"""Seed-aware power simulation and confirmatory test (registered method; CONTRACT_wave2 section 8.3;
vpd/PREREG_TVR_PILOT.md sections 12, 13 (A4) and 15).

PY_EVAL (or PY_Q3), stdlib; numpy is optional (it is only a faster backend of `simulate`).  Login node, 1 core
(importing this module also defaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS to 1, and the numpy path makes no BLAS
call; the prefix below is still the rule):

    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 19 \
        PY_EVAL -m vpd.common.power_sim tvr --pilot-root OUT/tvr/runs/pilot --out OUT/tvr/runs/pilot/POWER.json

The seed replicate (Q3B 100s1002, O4) is required; a root without it is refused (exit 2) unless
--allow-no-replicate is given (then sigma_used = a priori and lambda_used = the upper end of
power.persistence.range, recorded in POWER.json; gate_tvr never reads such a file: A4 stays pending).  The
backend is the registered power.backend (numpy); --backend / --n-sims are test overrides, recorded in
registered.backend / registered.n_sims_used (A4 counts only the registered values).

Registered model (frozen in pilot_cfg.FROZEN["power"]; DECISIONS G13 amended it before the freeze):
- p_i(T) = mean over the three pilot starts of the primary-arm (c) success of task i at pilot tier T in {100, TOP}
  (start-independent pooling: a mean, never a difference between starts).
- In each simulation, for every start of design.starts and every (tier T, seed s), a REALIZED-scale seed shift
  r = sigma (sqrt(rho) z_seed + sqrt(1 - rho) z_tier) (z iid N(0, 1); z_seed shared by the tiers of one (start, seed
  id); rho = tier_corr) is mapped to a pre-clip shift c = calibrate_shift(table_T, r); the treated start of the
  contrast being powered uses calibrate_shift(table_T, r + delta) instead; y_i = 1[W_i < clip(p_i(T) + c, 0, 1)].
- Task persistence (PREREG 12 item 7): W_i = V_{start, i} (one uniform per start and task, shared by all seeds and
  both tiers of the start) with probability sqrt(lambda), else a fresh uniform.  Marginally y_i is still
  Bernoulli(clip(p_i(T) + c)); two outcomes of one task in one start (different seeds or tiers, equal probability q)
  have covariance lambda q (1 - q), i.e. lambda is the intraclass correlation of a task's outcomes inside a start.
  lambda = persistence_hat: the Q3B seed pair's discordance against the no-persistence discordance at the pair's
  own SR level and task spread; the spread comes from covariances over tasks between the pilot starts' tier-100
  rows under a one-factor model (every start its own mean and scale), so start mean SRs cancel; lambda 0 draws
  exactly as before.
- Endpoint per start and task = sum_T w_T * mean_s y_{s,i}.  d_i = treated endpoint - mean of the controls'.
- TEST (A4 and the confirmatory analysis) = seed_aware_endpoint_t (seed_aware_endpoint_test): V = CR1 scene-cluster
  variance of mean(d) + k_contrast * sigma2_E_hat / n_seeds, sigma2_E_hat = seed_var_endpoint of the per-seed
  endpoint rows e_si = sum_T w_T y_{s,T,i} (seeds paired across tiers, so any cross-tier covariance is included),
  t reference with df = n_starts (n_seeds - 1) (12; fallback 6); two-sided; Holm over the contrasts at alpha.
  seed_aware_cluster_t (seed_aware_test, per-(start, tier) cells, df 24) is SUPERSEDED: it assumes independent tier
  shifts (simulated FWER 0.17-0.20 at tier_corr 1); kept for regression tests and a gated:false sensitivity block.
- power of a contrast = fraction of simulations in which it is Holm-rejected WITH THE TREATED START BETTER while the
  other contrast has effect 0; type-I (gated) = fraction of null simulations (delta 0 everywhere) in which Holm
  rejects at least one contrast (family-wise, either direction).  Simulations of one call share their random
  numbers across blocks (common random numbers), so power is comparable between deltas.
- A4 verdict per design (run_tvr): type-I = max FWER over tier_corr in power.tier_corr.type1 ({0, 1}) at sigma_used
  and lambda_used <= type1_max; power at primary_delta, tier_corr power.tier_corr.power (0.5), sigma_used and
  lambda_used >= target_power for both contrasts.  Fallback design = the main design with power.fallback.seeds
  (seeds 1-2 at BOTH tiers), same test, same rules -> fallback_pass.

Nothing here computes or prints a difference between real pilot starts: run_tvr pools the three starts by a mean;
the only real comparison is between two SEEDS of the same start (Q3B 100, seed 1001 vs 1002); persistence_hat
reads the other starts' rows only through covariances over tasks (lambda_hat is invariant to each start's mean SR;
POWER.json has no per-start value).
"""
from __future__ import annotations

import argparse
import bisect
import copy
import datetime
import functools
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

# Login node (contract R5; 14-core quota shared by several windows): numpy is imported lazily by _np(), so these
# defaults reach BLAS / OpenMP before it loads, unless the caller set its own values.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
del _v

from vpd.tvr import pilot_cfg  # noqa: E402

# /2 (wave 2b): tables.fallback_seeds_1_2, sensitivity_superseded_test, the persistence block, registered.tier_corr /
# fallback / persistence / backend and the verdict fields type1_by_rho, power_tier_corr, power_by_rho,
# mc_se_power_at_primary
SCHEMA = "vpd.power_sim.tvr/2"
# seed_aware_endpoint_t: the registered test (DECISIONS G13); seed_aware_cluster_t: superseded (assumes independent
# tier shifts), regression tests and a gated:false sensitivity block only; naive_cluster_z: regression test only
TESTS = ("seed_aware_cluster_t", "naive_cluster_z", "seed_aware_endpoint_t")


# --------------------------------------------------------------------------------------------- small helpers
def _np():
    try:
        import numpy
    except ImportError:  # pragma: no cover - numpy is present in PY_EVAL / PY_Q3
        return None
    return numpy


def _backend(name: str) -> str:
    if name not in ("auto", "numpy", "python"):
        raise ValueError(f"backend {name!r}")
    if name == "auto":
        return "numpy" if _np() is not None else "python"
    if name == "numpy" and _np() is None:
        raise RuntimeError("numpy backend requested but numpy is not importable")
    return name


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------- Holm
def holm_reject(p: dict, alpha: float = 0.05) -> dict:
    """Holm step-down: sort ascending (ties by name); reject while p_(j) <= alpha / (m - j) (j = 0..m-1)."""
    names = sorted(p, key=lambda k: (p[k], k))
    m = len(names)
    out = {k: False for k in p}
    for j, k in enumerate(names):
        if p[k] <= alpha / (m - j):
            out[k] = True
        else:
            break
    return out


# --------------------------------------------------------------------------------------------- shift mapping
def shift_table(p: list, m: int = 2001) -> tuple:
    """Pre-clip shifts c on a uniform grid of [-1, 1] (m points) and the realized shifts
    r(c) = mean(clip(p + c, 0, 1)) - mean(p) (non-decreasing in c)."""
    if m < 3:
        raise ValueError("m >= 3")
    p = [float(x) for x in p]
    if not p or any(not (0.0 <= x <= 1.0) for x in p):
        raise ValueError("p must be a non-empty list of probabilities")
    n = len(p)
    mean_p = sum(p) / n
    cs = [-1.0 + 2.0 * j / (m - 1) for j in range(m)]
    np = _np()
    if np is not None:
        pa = np.asarray(p, dtype=np.float64)
        ca = np.asarray(cs, dtype=np.float64)
        rs = (np.clip(pa[None, :] + ca[:, None], 0.0, 1.0).sum(axis=1) / n - mean_p).tolist()
    else:
        rs = [sum(min(1.0, max(0.0, x + c)) for x in p) / n - mean_p for c in cs]
    # enforce exact monotonicity against float noise (r(c) is mathematically non-decreasing)
    for j in range(1, m):
        if rs[j] < rs[j - 1]:
            rs[j] = rs[j - 1]
    return cs, rs


def calibrate_shift(table, r: float) -> float:
    """Inverse of r(c) by linear interpolation: the first grid index j with rs[j] >= r (bisect_left) and its left
    neighbour; r at or below the reachable minimum -> cs[0] (= -1), above the maximum -> cs[-1] (= +1)."""
    cs, rs = table
    j = bisect.bisect_left(rs, r)
    if j == 0:
        return cs[0]
    if j >= len(rs):
        return cs[-1]
    r0, r1 = rs[j - 1], rs[j]
    return cs[j - 1] + (r - r0) / (r1 - r0) * (cs[j] - cs[j - 1])


def _calibrate_np(np, table_np, x):
    """Vectorized calibrate_shift (identical semantics)."""
    cs, rs = table_np
    j = np.searchsorted(rs, x, side="left")
    out = np.empty(np.shape(x), dtype=np.float64)
    lo = j == 0
    hi = j >= rs.shape[0]
    mid = ~(lo | hi)
    out[lo] = cs[0]
    out[hi] = cs[-1]
    jm = j[mid]
    r0, r1 = rs[jm - 1], rs[jm]
    out[mid] = cs[jm - 1] + (x[mid] - r0) / (r1 - r0) * (cs[jm] - cs[jm - 1])
    return out


# --------------------------------------------------------------------------------------------- variances
def cr1_var(d: list, clusters: list) -> tuple:
    """(mean d, CR1 cluster-robust variance of the mean): G/(G-1) * sum_g (sum_{i in g} (d_i - D))^2 / n^2."""
    n = len(d)
    if n == 0 or n != len(clusters):
        raise ValueError("d and clusters must be non-empty and of equal length")
    D = sum(d) / n
    sums: dict = {}
    for x, g in zip(d, clusters):
        sums[g] = sums.get(g, 0.0) + (x - D)
    G = len(sums)
    if G < 2:
        raise ValueError("CR1 needs >= 2 clusters")
    V = G / (G - 1) * sum(s * s for s in sums.values()) / (n * n)
    return D, V


def seed_var_cells(cells: list) -> list:
    """Per (start, tier) cell (a seeds x tasks 0/1 matrix): var_s(SR_s) (ddof 1) - mean_i var_s(y_si) (ddof 1) / n.
    Unbiased for the variance of the realized SR shift; NOT truncated (seed_var_pooled truncates the mean)."""
    out = []
    for cell in cells:
        ns = len(cell)
        if ns < 2:
            raise ValueError("a cell needs >= 2 seeds")
        n = len(cell[0])
        if n == 0 or any(len(row) != n for row in cell):
            raise ValueError("ragged cell")
        srs = [sum(row) / n for row in cell]
        m = sum(srs) / ns
        v_sr = sum((x - m) ** 2 for x in srs) / (ns - 1)
        w = 0.0
        for col in zip(*cell):
            mc = sum(col) / ns
            w += sum((y - mc) ** 2 for y in col) / (ns - 1)
        out.append(v_sr - (w / n) / n)
    return out


def seed_var_pooled(cells: list) -> tuple:
    """(sigma2_hat, df): sigma2_hat = max(0, mean over cells of seed_var_cells); df = sum over cells (n_seeds - 1)."""
    if not cells:
        raise ValueError("no cells")
    ests = seed_var_cells(cells)
    df = sum(len(c) - 1 for c in cells)
    return max(0.0, sum(ests) / len(ests)), df


def seed_var_endpoint(ecells: list) -> tuple:
    """The registered seed variance (seed_aware_endpoint_t; DECISIONS G13).  ecells: per start, a seeds x tasks
    matrix of e_si = sum_T w_T y_{s,T,i} (seeds paired across tiers).  (sigma2_E_hat, df):
    sigma2_E_hat = max(0, mean over starts of [var_s(E_s) - mean_i var_s(e_si) / n]) (ddof 1; E_s = mean_i e_si),
    the variance of a start's per-seed endpoint shift INCLUDING any cross-tier covariance; df = sum (n_seeds - 1)."""
    if not ecells:
        raise ValueError("no starts")
    ests = seed_var_cells(ecells)   # same algebra on the weighted per-seed rows
    df = sum(len(c) - 1 for c in ecells)
    return max(0.0, sum(ests) / len(ests)), df


# --------------------------------------------------------------------------------------------- Student t
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularized incomplete beta (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, 3000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return h


def betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    bt = math.exp(lbt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf2(t: float, df) -> float:
    """Two-sided Student-t p value P(|T_df| >= |t|) = I_{df/(df+t^2)}(df/2, 1/2)."""
    if df is None or df <= 0:
        raise ValueError(f"df must be > 0, got {df!r}")
    if math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 0.0
    x = df / (df + t * t)
    return min(1.0, max(0.0, betainc_reg(df / 2.0, 0.5, x)))


@functools.lru_cache(maxsize=256)
def t_crit2(alpha: float, df) -> float:
    """t > 0 with t_sf2(t, df) == alpha (two-sided critical value), by bisection."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha in (0, 1)")
    lo, hi = 0.0, 1.0
    while t_sf2(hi, df) > alpha:
        hi *= 2.0
        if hi > 1e12:
            raise ValueError("no critical value")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_sf2(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    return 0.5 * (lo + hi)


def norm_sf2(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


# --------------------------------------------------------------------------------------------- the tests
def seed_aware_test(d, clusters, sigma2_hat, df, k_contrast, weights, n_seeds_by_tier) -> dict:
    """SUPERSEDED (test seed_aware_cluster_t; DECISIONS G13): it assumes independent tier shifts within a (start,
    seed); kept for regression tests and the gated:false sensitivity block.  weights / n_seeds_by_tier: {tier: w} /
    {tier: n_s}.  V = cr1_var + k_contrast * sum_T w_T^2 * sigma2_hat / n_seeds_T; t = D / sqrt(V); p = t_sf2(t, df);
    ci95 = D +/- t_{0.975, df} sqrt(V)."""
    D, v_cl = cr1_var(d, clusters)
    v_seed = k_contrast * sum(float(w) ** 2 * sigma2_hat / n_seeds_by_tier[T] for T, w in weights.items())
    return _t_result(D, v_cl, v_seed, df)


def seed_aware_endpoint_test(d, clusters, sigma2_e_hat, df, k_contrast, n_seeds) -> dict:
    """The registered test (power.test = seed_aware_endpoint_t; A4 and the confirmatory analysis; DECISIONS G13).
    V = cr1_var + k_contrast * sigma2_E_hat / n_seeds with sigma2_E_hat from seed_var_endpoint (valid for any
    correlation of the tier shifts within a (start, seed)); df = n_starts (n_seeds - 1); t = D / sqrt(V);
    p = t_sf2(t, df); ci95 = D +/- t_{0.975, df} sqrt(V)."""
    D, v_cl = cr1_var(d, clusters)
    return _t_result(D, v_cl, k_contrast * sigma2_e_hat / n_seeds, df)


def _t_result(D: float, v_cl: float, v_seed: float, df) -> dict:
    V = v_cl + v_seed
    se = math.sqrt(V) if V > 0 else 0.0
    if se > 0:
        t = D / se
    else:
        t = 0.0 if D == 0 else math.copysign(math.inf, D)
    p = t_sf2(t, df)
    q = t_crit2(0.05, df)
    return {"D": D, "se": se, "t": t, "df": df, "p": p, "ci95": [D - q * se, D + q * se],
            "v_cluster": v_cl, "v_seed": v_seed}


def naive_cluster_z(d, clusters) -> dict:
    """NOT registered: paired scene-cluster z without the seed component (shows the inflation; regression test)."""
    D, v = cr1_var(d, clusters)
    se = math.sqrt(v) if v > 0 else 0.0
    z = D / se if se > 0 else (0.0 if D == 0 else math.copysign(math.inf, D))
    return {"D": D, "se": se, "z": z, "p": norm_sf2(z) if not math.isinf(z) else 0.0}


def k_contrast(controls: list) -> float:
    """Sum of squared contrast coefficients: treated +1, each of the m controls -1/m -> 1 + 1/m."""
    m = len(controls)
    if m < 1:
        raise ValueError("a contrast needs >= 1 control")
    return 1.0 + m * (1.0 / m) ** 2


def contrast_name(treated: str, controls: list) -> str:
    if len(controls) == 1:
        return f"{treated} - {controls[0]}"
    return f"{treated} - mean({', '.join(controls)})"


def confirmatory_outcome(results: dict, alpha: float = 0.05, min_effect: float = 0.05) -> dict:
    """The three-way outcome rule of FROZEN.power.outcome.  results: {contrast name: seed_aware_endpoint_test result}
    (D = treated minus control(s)).  Per contrast: support = Holm-rejected with D > 0; no_support = Holm-rejected
    with D < 0, OR ci95 upper bound < min_effect; else inconclusive.  Overall: 支持 iff >= 1 support and no
    contrast Holm-rejected with D < 0; 不支持 iff every contrast no_support; else 无定论."""
    rej = holm_reject({k: v["p"] for k, v in results.items()}, alpha)
    per = {}
    for k, v in results.items():
        if rej[k] and v["D"] > 0:
            per[k] = "support"
        elif (rej[k] and v["D"] < 0) or v["ci95"][1] < min_effect:
            per[k] = "no_support"
        else:
            per[k] = "inconclusive"
    opposite = any(rej[k] and results[k]["D"] < 0 for k in results)
    if any(x == "support" for x in per.values()) and not opposite:
        overall = "支持"
    elif per and all(x == "no_support" for x in per.values()):
        overall = "不支持"
    else:
        overall = "无定论"
    return {"holm": rej, "per_contrast": per, "overall": overall, "alpha": alpha, "min_effect": min_effect}


def seed_sd(y1: list, y2: list, a_priori: float = 0.03) -> dict:
    """Realized-scale seed SD from two seeds of the SAME start on the same tasks (Q3B 100 c, seed 1001 vs 1002):
    e_i = y1_i - y2_i; d = mean e; S = sum (e_i - d)^2 / (n (n - 1)); sigma_hat = sqrt(max(0, (d^2 - S) / 2));
    sigma_used = max(a_priori, sigma_hat)."""
    n = len(y1)
    if n < 2 or n != len(y2):
        raise ValueError("y1 and y2 must have equal length >= 2")
    e = [float(a) - float(b) for a, b in zip(y1, y2)]
    d = sum(e) / n
    S = sum((x - d) ** 2 for x in e) / (n * (n - 1))
    s2 = max(0.0, (d * d - S) / 2.0)
    hat = math.sqrt(s2)
    return {"n": n, "seed_diff_mean": d, "S": S, "sigma2_hat": s2, "sigma_hat": hat, "a_priori": a_priori,
            "sigma_used": max(a_priori, hat)}


def shift_gap(dbar: float, S: float, sigma_used: float) -> float:
    """kappa of persistence_hat: E[|r1 - r2| | dbar, S] for the realized shift difference of two seeds under the
    registered seed model r1 - r2 ~ N(0, 2 sigma_used^2) with dbar | (r1 - r2) ~ N(r1 - r2, S) (dbar, S as seed_sd):
    B = 2 sigma_used^2 / (2 sigma_used^2 + S), a = B dbar, v = B S (normal posterior),
    kappa = sqrt(v) sqrt(2 / pi) exp(-a^2 / (2 v)) + a (1 - 2 Phi(-a / sqrt(v))) (mean of the folded posterior).
    S <= 0 or sigma_used <= 0 -> |dbar|."""
    tau2 = 2.0 * float(sigma_used) ** 2
    if S <= 0 or tau2 <= 0:
        return abs(dbar)
    B = tau2 / (tau2 + S)
    a, v = B * dbar, B * S
    sv = math.sqrt(v)
    phi = 0.5 * (1.0 + math.erf((-a / sv) / math.sqrt(2.0)))
    return sv * math.sqrt(2.0 / math.pi) * math.exp(-a * a / (2.0 * v)) + a * (1.0 - 2.0 * phi)


def _cov(x: list, z: list) -> float:
    """Covariance over tasks (ddof 1)."""
    n = len(x)
    mx, mz = sum(x) / n, sum(z) / n
    return sum((a - mx) * (b - mz) for a, b in zip(x, z)) / (n - 1)


def shift_m2(dbar: float, S: float, sigma_used: float) -> float:
    """E[(r1 - r2)^2 | dbar, S] under shift_gap's normal posterior: a^2 + v; S <= 0 or sigma_used <= 0 -> dbar^2."""
    tau2 = 2.0 * float(sigma_used) ** 2
    if S <= 0 or tau2 <= 0:
        return float(dbar) ** 2
    B = tau2 / (tau2 + S)
    return (B * dbar) ** 2 + B * S


def _factor_v(lev: list, oth: list, groups: list) -> tuple:
    """(V, V_jk): V = sum_{a<b} C(l, y_a) C(l, y_b) / sum_{a<b} C(y_a, y_b) (C = covariance over tasks, ddof 1) and
    its delete-one-group jackknife G V - (G - 1) mean_g V_-g (groups: lists of task indices).  V None when the
    denominator is <= 0; V_jk None when V or any V_-g is None; V_jk = V when there are < 2 groups."""
    k = len(oth)
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]

    def sums(idx):
        return ([len(idx), sum(lev[i] for i in idx)] + [sum(oth[a][i] for i in idx) for a in range(k)]
                + [sum(lev[i] * oth[a][i] for i in idx) for a in range(k)]
                + [sum(oth[a][i] * oth[b][i] for i in idx) for a, b in pairs])

    def v_of(s):
        m = s[0]
        if m < 3:
            return None
        sl, sy, sly, syy = s[1], s[2:2 + k], s[2 + k:2 + 2 * k], s[2 + 2 * k:]
        c_l = [(sly[a] - sl * sy[a] / m) / (m - 1) for a in range(k)]
        dnm = sum((syy[j] - sy[a] * sy[b] / m) / (m - 1) for j, (a, b) in enumerate(pairs))
        if dnm <= 0:
            return None
        return sum(c_l[a] * c_l[b] for a, b in pairs) / dnm

    tot = sums(range(len(lev)))
    V = v_of(tot)
    if V is None or len(groups) < 2:
        return V, V
    loo = []
    for idx in groups:
        v = v_of([t - x for t, x in zip(tot, sums(idx))])
        if v is None:
            return V, None
        loo.append(v)
    G = len(loo)
    return V, G * V - (G - 1) * sum(loo) / G


def persistence_hat(y1: list, y2: list, others: list, sigma_used: float, clusters=None,
                    lam_range=(0.0, 1.0)) -> dict:
    """Task persistence lambda (registered, PREREG 12 item 7; DECISIONS G13 and its wave-2b amendment) from the seed
    replicate.  y1 / y2 = two seeds of the SAME start on the same tasks at the replicate tier (Q3B 100 c, seed 1001 /
    1002); others = the other pilot starts' outcomes at that tier (>= 2 rows, seed 1001, tasks aligned);
    sigma_used = seed_sd's (from the same pair and the a priori value); clusters = the tasks' scenes (the jackknife
    and SE unit; None = every task its own).
    Model (simulate): lambda = intraclass correlation of a task's outcomes inside a start.  The pair's discordance D
    is compared with R, the discordance two seeds WITHOUT persistence would have at the pair's own SR level L and
    task spread: R = 2 B + (r1 - r2)^2 / 2, B = mean_i q_i (1 - q_i) = L (1 - L) - V, V = var_i(q) of the replicate
    start's success probabilities; with persistence, E[D | r1, r2] = (1 - lambda) R + lambda |r1 - r2| (exact for
    unclipped shifts r1, r2 of the pair).  V comes from covariances over tasks under a one-factor model (every start
    its own mean AND scale of one common task pattern, q_Si = a_S + b_S f_i): C(l, y_a) = b b_a var(f), C(y_a, y_b)
    = b_a b_b var(f), so V = C(l, y_a) C(l, y_b) / C(y_a, y_b), bias-corrected by a delete-one-scene jackknife
    (the plain ratio overstates V: lambda_hat about -0.03 at lambda 0).  So lambda_hat does not move with the start
    mean SRs (exactly in that model; under clipping see persistence_check with offsets).  The wave-2 estimator used
    H = m / (m - 1) mean_i p_i (1 - p_i), half the mean between-start discordance, which moved with the start SRs
    (about +-0.1 at 5 pp).  Nothing per start is reported, no difference between starts is formed.
        D      = mean_i (y1_i - y2_i)^2                                  (discordance of the pair)
        kappa  = shift_gap(dbar, S, sigma_used), m2 = shift_m2(...)      (E[|r1 - r2|], E[(r1 - r2)^2] | the pair)
        l_i    = (y1_i + y2_i) / 2, L = mean_i l_i                       (the pair's SR level)
        V      = _factor_v jackknife of sum_{a<b} C(l, y_a) C(l, y_b) / sum_{a<b} C(y_a, y_b)   (one pair for the
                 three pilot starts; C = covariance over tasks, ddof 1; delete-one-scene)
        B      = L (1 - L) - V;  R = 2 B + m2 / 2
        lambda_hat = (R - D) / (R - kappa); lambda_used = clip(lambda_hat, lam_range);
        a covariance denominator <= 0 (full or delete-one-scene) or R <= kappa -> lambda_hat None and lambda_used =
        lam_range[1] (conservative).
    se: scene-cluster delta-method SE of lambda_hat over the task sampling only (kappa, m2 fixed; B linearized with
    the plain V; reported only; the seed-shift noise makes the true spread larger, see persistence_check's sd_hat)."""
    n = len(y1)
    if n < 3 or n != len(y2):
        raise ValueError("y1 and y2 must have equal length >= 3")
    oth = [[float(x) for x in r] for r in others]
    k = len(oth)
    if k < 2 or any(len(r) != n for r in oth):
        raise ValueError("others: >= 2 rows (the other pilot starts), each aligned with y1")
    cl = list(range(n)) if clusters is None else list(clusters)
    if len(cl) != n:
        raise ValueError("clusters not aligned with y1")
    groups: dict = {}
    for i, g in enumerate(cl):
        groups.setdefault(g, []).append(i)
    lo, hi = float(lam_range[0]), float(lam_range[1])
    sd = seed_sd(y1, y2)
    dsq = [(float(a) - float(b)) ** 2 for a, b in zip(y1, y2)]
    D = sum(dsq) / n
    lev = [(float(a) + float(b)) / 2.0 for a, b in zip(y1, y2)]
    L = sum(lev) / n
    kappa = shift_gap(sd["seed_diff_mean"], sd["S"], sigma_used)
    m2 = shift_m2(sd["seed_diff_mean"], sd["S"], sigma_used)
    V_plain, V = _factor_v(lev, oth, list(groups.values()))
    out = {"n": n, "n_pooled": k + 1, "D": D, "level": L, "V_plain": V_plain, "V": V, "B": None, "R": None,
           "kappa": kappa, "m2": m2, "sigma_used": float(sigma_used), "range": [lo, hi], "lambda_hat": None,
           "se": None, "lambda_used": hi}
    if V is None:
        out["note"] = ("a covariance C(y_a, y_b) <= 0 (full or delete-one-scene): the task pattern is not identified "
                       "-> lambda_used = range upper end")
        return out
    B = L * (1.0 - L) - V
    R = 2.0 * B + m2 / 2.0
    den = R - kappa
    out.update(B=B, R=R)
    if den <= 0:
        out["note"] = "R <= kappa: lambda not identified -> lambda_used = range upper end"
        return out
    lam = (R - D) / den
    # delta method (kappa, m2 fixed): dlam/dD = -1 / den, dlam/dB = 2 (D - kappa) / den^2; B = L (1 - L) - num / dnm
    # (plain V) with the per-task influence of a covariance C(x, z): (x_i - mean x)(z_i - mean z) - C(x, z)
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    c_l = [_cov(lev, r) for r in oth]
    c_ab = [_cov(oth[a], oth[b]) for a, b in pairs]
    num, dnm = sum(c_l[a] * c_l[b] for a, b in pairs), sum(c_ab)
    gd, gb = -1.0 / den, 2.0 * (D - kappa) / (den * den)
    mo = [sum(r) / n for r in oth]
    dl = [x - L for x in lev]
    dev = [[x - mo[a] for x in r] for a, r in enumerate(oth)]
    bb = [0.0] * n
    for i in range(n):
        g = (1.0 - 2.0 * L) * dl[i]
        for (a, b), cab in zip(pairs, c_ab):
            g -= (c_l[b] * (dl[i] * dev[a][i] - c_l[a]) + c_l[a] * (dl[i] * dev[b][i] - c_l[b])) / dnm
            g += num / (dnm * dnm) * (dev[a][i] * dev[b][i] - cab)
        bb[i] = g
    bm = sum(bb) / n
    psi = [gd * (x - D) + gb * (y - bm) for x, y in zip(dsq, bb)]
    sums: dict = {}
    for x, g in zip(psi, cl):
        sums[g] = sums.get(g, 0.0) + x
    G = len(sums)
    if G >= 2:
        out["se"] = math.sqrt(G / (G - 1) * sum(s * s for s in sums.values())) / n
    out["lambda_hat"] = lam
    out["lambda_used"] = min(hi, max(lo, lam))
    return out


# --------------------------------------------------------------------------------------------- simulation
def _design_parts(design: dict, p_by_tier: dict, clusters: list) -> dict:
    tiers = list(design["tiers"])
    weights = dict(zip(tiers, [float(w) for w in design["weights"]]))
    if len(weights) != len(tiers):
        raise ValueError("weights do not match tiers")
    n_seeds = {T: len(design["seeds"][T]) for T in tiers}
    if any(v < 2 for v in n_seeds.values()):
        raise ValueError("every tier needs >= 2 seeds (the seed variance is estimated from them)")
    starts = list(design["starts"])
    contrasts = [(c[0], list(c[1])) for c in design["contrasts"]]
    for tr, ctl in contrasts:
        if tr not in starts or any(c not in starts for c in ctl) or tr in ctl:
            raise ValueError(f"bad contrast {tr} vs {ctl}")
    n = len(clusters)
    for T in tiers:
        if T not in p_by_tier or len(p_by_tier[T]) != n:
            raise ValueError(f"p_by_tier[{T!r}] missing or not aligned with clusters")
    seed_ids = {T: list(design["seeds"][T]) for T in tiers}
    if any(len(set(v)) != len(v) for v in seed_ids.values()):
        raise ValueError("duplicate seed ids in a tier")
    all_ids = sorted({s for v in seed_ids.values() for s in v})
    return {"tiers": tiers, "weights": weights, "n_seeds": n_seeds, "starts": starts, "contrasts": contrasts,
            "names": [contrast_name(t, c) for t, c in contrasts], "k": [k_contrast(c) for _, c in contrasts],
            "n": n, "seed_ids": seed_ids, "all_seed_ids": all_ids,
            "seed_pos": {T: [all_ids.index(s) for s in seed_ids[T]] for T in tiers},
            "paired": all(seed_ids[T] == seed_ids[tiers[0]] for T in tiers)}


def _decide(pvals: list, Ds: list, names: list, alpha: float) -> tuple:
    rej = holm_reject(dict(zip(names, pvals)), alpha)
    return [rej[k] for k in names], [rej[k] and D > 0 for k, D in zip(names, Ds)]


def _stats_python(y: dict, parts: dict, clusters: list, test: str) -> tuple:
    """One simulation's tests from y[(start, tier)] = seeds x tasks 0/1 lists -> (sigma2_hat, df, [result])."""
    tiers, w, ns, n = parts["tiers"], parts["weights"], parts["n_seeds"], parts["n"]
    if test == "seed_aware_endpoint_t":
        ecells = []
        for S in parts["starts"]:
            rows = [[0.0] * n for _ in range(ns[tiers[0]])]
            for T in tiers:
                for s, row in enumerate(y[(S, T)]):
                    for i in range(n):
                        rows[s][i] += w[T] * row[i]
            ecells.append(rows)
        s2, df = seed_var_endpoint(ecells)
    else:
        s2, df = seed_var_pooled([y[(S, T)] for S in parts["starts"] for T in tiers])
    end = {}
    for S in parts["starts"]:
        e = [0.0] * n
        for T in tiers:
            cell = y[(S, T)]
            k = [sum(col) for col in zip(*cell)]
            for i in range(n):
                e[i] += w[T] * k[i] / ns[T]
        end[S] = e
    res = []
    for (tr, ctl), kc in zip(parts["contrasts"], parts["k"]):
        m = len(ctl)
        d = [end[tr][i] - sum(end[c][i] for c in ctl) / m for i in range(n)]
        if test == "seed_aware_cluster_t":
            res.append(seed_aware_test(d, clusters, s2, df, kc, w, ns))
        elif test == "seed_aware_endpoint_t":
            res.append(seed_aware_endpoint_test(d, clusters, s2, df, kc, ns[tiers[0]]))
        else:
            res.append(naive_cluster_z(d, clusters))
    return s2, df, res


def _stats_numpy(np, y: dict, parts: dict, cl_index, G: int, test: str) -> tuple:
    """Vectorized _stats_python over a block: y[(start, tier)] bool/int arrays (B, n_seeds, n).
    Returns (sigma2_hat (B,), df, [(D (B,), v_cl (B,), v_seed (B,))]).  No BLAS call (cluster sums by reduceat),
    so no thread pool is started on the login node."""
    tiers, w, ns, n = parts["tiers"], parts["weights"], parts["n_seeds"], parts["n"]

    def cell_est(a):
        v_sr = a.mean(axis=2).var(axis=1, ddof=1)
        w_i = a.var(axis=1, ddof=1).mean(axis=1)
        return v_sr - w_i / n

    ests = []
    df = 0
    if test == "seed_aware_endpoint_t":
        for S in parts["starts"]:
            e = sum(w[T] * y[(S, T)].astype(np.float64) for T in tiers)   # (B, n_seeds, n), seeds paired
            ests.append(cell_est(e))
            df += ns[tiers[0]] - 1
    else:
        for S in parts["starts"]:
            for T in tiers:
                ests.append(cell_est(y[(S, T)].astype(np.float64)))
                df += ns[T] - 1
    s2 = np.maximum(0.0, np.mean(np.stack(ests, axis=0), axis=0))
    end = {}
    for S in parts["starts"]:
        e = None
        for T in tiers:
            part = w[T] * y[(S, T)].astype(np.float64).sum(axis=1) / ns[T]
            e = part if e is None else e + part
        end[S] = e
    cl_index = np.asarray(cl_index)
    order = np.argsort(cl_index, kind="stable")
    srt = cl_index[order]
    bounds = np.flatnonzero(np.concatenate(([True], srt[1:] != srt[:-1])))
    if bounds.shape[0] != G:
        raise ValueError("cluster index does not cover 0..G-1")
    out = []
    for (tr, ctl), kc in zip(parts["contrasts"], parts["k"]):
        d = end[tr] - sum(end[c] for c in ctl) / len(ctl)
        D = d.mean(axis=1)
        sums = np.add.reduceat((d - D[:, None])[:, order], bounds, axis=1)
        v_cl = G / (G - 1) * (sums ** 2).sum(axis=1) / (n * n)
        if test == "seed_aware_cluster_t":
            v_seed = kc * sum(w[T] ** 2 / ns[T] for T in tiers) * s2
        elif test == "seed_aware_endpoint_t":
            v_seed = kc * s2 / ns[tiers[0]]
        else:
            v_seed = np.zeros_like(v_cl)
        out.append((D, v_cl, v_seed))
    return s2, df, out


def _p_of(D: float, v_cl: float, v_seed: float, df, test: str) -> float:
    if test in ("seed_aware_cluster_t", "seed_aware_endpoint_t"):
        return _t_result(D, v_cl, v_seed, df)["p"]
    se = math.sqrt(v_cl) if v_cl > 0 else 0.0
    if se == 0:
        return 1.0 if D == 0 else 0.0
    return norm_sf2(D / se)


def simulate(p_by_tier: dict, clusters: list, design: dict, delta: float, sigma: float, n_sims: int, seed: int, *,
             alpha: float = 0.05, test: str = "seed_aware_cluster_t", backend: str = "auto",
             chunk: int = 250, m_table: int = 2001, tier_corr: float = 0.0, persistence: float = 0.0) -> dict:
    """Simulated rejection rates of the design at effect `delta` (realized SR scale) and seed SD `sigma`.
    (The default test is the superseded one, for the regression tests; run_tvr always passes power.test.)

    delta == 0: one null block (no start shifted) -> "fwer" (gated type-I: >= 1 Holm rejection, either direction),
    "type1_by_contrast" (unadjusted p <= alpha), "reject_holm", "power" (Holm-rejected with treated better).
    delta != 0: one block per contrast, only that contrast's treated start shifted by delta (the other contrast at
    effect 0) -> "power" (registered: Holm-rejected AND treated better), "reject_holm" (either direction).
    Every block uses random seed `seed` (common random numbers across blocks and deltas).
    tier_corr: correlation of the realized shifts of one (start, seed id) across tiers (power.tier_corr),
    r = sigma (sqrt(rho) z_seed + sqrt(1 - rho) z_tier); 0 draws exactly as before.
    persistence: lambda of the task-persistence term (module docstring; PREREG 12 item 7); 0 draws exactly as before.
    "mc_se_power": binomial MC SE sqrt(p (1 - p) / n_sims) of each power entry.
    "sim_mean_D": mean over simulations of each contrast's D in its block (SIMULATED starts drawn from the pooled p,
    not real ones; checks the registered effect model: about delta for the shifted contrast)."""
    if test not in TESTS:
        raise ValueError(f"test {test!r}")
    if n_sims < 1:
        raise ValueError("n_sims >= 1")
    if not 0.0 <= float(tier_corr) <= 1.0:
        raise ValueError("tier_corr in [0, 1]")
    if not 0.0 <= float(persistence) <= 1.0:
        raise ValueError("persistence in [0, 1]")
    be = _backend(backend)
    parts = _design_parts(design, p_by_tier, clusters)
    if test == "seed_aware_endpoint_t" and not parts["paired"]:
        raise ValueError("seed_aware_endpoint_t needs the same seed ids in every tier")
    tables = {T: shift_table(p_by_tier[T], m_table) for T in parts["tiers"]}
    names = parts["names"]
    blocks = [None] if delta == 0 else list(range(len(parts["contrasts"])))
    out = {"delta": delta, "sigma": sigma, "n_sims": n_sims, "seed": seed, "backend": be, "test": test,
           "tier_corr": float(tier_corr), "persistence": float(persistence), "alpha": alpha, "contrasts": names,
           "k_contrast": dict(zip(names, parts["k"])), "n_tasks": parts["n"], "n_clusters": len(set(clusters)),
           "power": {}, "reject_holm": {}, "sim_mean_D": {}}
    s2_sum, s2_n = 0.0, 0
    for blk in blocks:
        shift = {} if blk is None else {parts["contrasts"][blk][0]: float(delta)}
        cnt = _run_block(p_by_tier, clusters, parts, tables, shift, sigma, n_sims, seed, alpha, test, be, chunk,
                         float(tier_corr), float(persistence))
        s2_sum += cnt["s2_sum"]
        s2_n += n_sims
        out["df"] = cnt["df"]
        if blk is None:
            out["fwer"] = cnt["any"] / n_sims
            out["type1_by_contrast"] = {k: cnt["unadj"][j] / n_sims for j, k in enumerate(names)}
            out["reject_holm"] = {k: cnt["holm"][j] / n_sims for j, k in enumerate(names)}
            out["power"] = {k: cnt["better"][j] / n_sims for j, k in enumerate(names)}
            out["sim_mean_D"] = {k: cnt["D_sum"][j] / n_sims for j, k in enumerate(names)}
        else:
            k = names[blk]
            out["power"][k] = cnt["better"][blk] / n_sims
            out["reject_holm"][k] = cnt["holm"][blk] / n_sims
            out["sim_mean_D"][k] = cnt["D_sum"][blk] / n_sims
    out["sigma2_hat_mean"] = s2_sum / s2_n
    out["mc_se_at_0.05"] = math.sqrt(0.05 * 0.95 / n_sims)
    out["mc_se_power"] = {k: math.sqrt(v * (1.0 - v) / n_sims) for k, v in out["power"].items()}
    return out


def _draw_start_np(np, rng, B, tiers, seed_pos, n_ids, tabs, pa, sigma, rho, lam, shift_r) -> dict:
    """One start's simulated 0/1 outcomes {tier: (B, n_seeds_T, n) bool} (the registered seed model and task
    persistence; module docstring).  Draw order: tier-common z (rho > 0), persistent V (lam > 0), then per tier the
    shifts, the fresh uniforms and (lam > 0) the persistence selectors; rho 0 and lam 0 draw exactly as before."""
    a_c, a_t = math.sqrt(rho), math.sqrt(1.0 - rho)
    zc = rng.normal(0.0, 1.0, size=(B, n_ids)) if rho > 0 else None
    n = pa[tiers[0]].shape[0]
    v = rng.random((B, n)) if lam > 0 else None
    beta = math.sqrt(lam)
    out = {}
    for T in tiers:
        ns = len(seed_pos[T])
        if zc is None:
            r = rng.normal(0.0, sigma, size=(B, ns))
        else:
            zt = rng.normal(0.0, 1.0, size=(B, ns))
            r = sigma * (a_c * zc[:, seed_pos[T]] + a_t * zt)
        c = _calibrate_np(np, tabs[T], r + shift_r)
        q = np.clip(pa[T][None, None, :] + c[:, :, None], 0.0, 1.0)
        u = rng.random(q.shape)
        if v is not None:
            u = np.where(rng.random(q.shape) < beta, v[:, None, :], u)
        out[T] = u < q
    return out


def _run_block(p_by_tier, clusters, parts, tables, shift, sigma, n_sims, seed, alpha, test, be, chunk,
               rho=0.0, lam=0.0) -> dict:
    names = parts["names"]
    nc = len(names)
    cnt = {"any": 0, "unadj": [0] * nc, "holm": [0] * nc, "better": [0] * nc, "D_sum": [0.0] * nc, "s2_sum": 0.0,
           "df": None}
    tiers, starts = parts["tiers"], parts["starts"]
    a_c, a_t = math.sqrt(rho), math.sqrt(1.0 - rho)
    beta = math.sqrt(lam)
    n_ids = len(parts["all_seed_ids"])
    if be == "python":
        rng = random.Random(seed)
        for _ in range(n_sims):
            y = {}
            for S in starts:
                zc = [rng.gauss(0.0, 1.0) for _ in range(n_ids)] if rho > 0 else None
                v = [rng.random() for _ in range(parts["n"])] if lam > 0 else None
                for T in tiers:
                    pT = p_by_tier[T]
                    rows = []
                    for js in parts["seed_pos"][T]:
                        if zc is None:
                            r = rng.gauss(0.0, sigma)
                        else:
                            r = sigma * (a_c * zc[js] + a_t * rng.gauss(0.0, 1.0))
                        c = calibrate_shift(tables[T], r + shift.get(S, 0.0))
                        if v is None:
                            rows.append([1 if rng.random() < min(1.0, max(0.0, pi + c)) else 0 for pi in pT])
                        else:
                            row = []
                            for i, pi in enumerate(pT):
                                u = rng.random()
                                if rng.random() < beta:
                                    u = v[i]
                                row.append(1 if u < min(1.0, max(0.0, pi + c)) else 0)
                            rows.append(row)
                    y[(S, T)] = rows
            s2, df, res = _stats_python(y, parts, clusters, test)
            cnt["s2_sum"] += s2
            cnt["df"] = df
            pv = [r["p"] for r in res]
            _tally(cnt, pv, [r["D"] for r in res], names, alpha)
        return cnt
    np = _np()
    rng = np.random.default_rng(seed)
    tabs = {T: (np.asarray(tables[T][0]), np.asarray(tables[T][1])) for T in tiers}
    pa = {T: np.asarray(p_by_tier[T], dtype=np.float64) for T in tiers}
    uniq = sorted(set(clusters))
    pos = {g: i for i, g in enumerate(uniq)}
    cl_index = np.asarray([pos[g] for g in clusters])
    if len(uniq) < 2:
        raise ValueError("CR1 needs >= 2 clusters")
    done = 0
    while done < n_sims:
        B = min(chunk, n_sims - done)
        y = {}
        for S in starts:
            ys = _draw_start_np(np, rng, B, tiers, parts["seed_pos"], n_ids, tabs, pa, sigma, rho, lam,
                                shift.get(S, 0.0))
            for T in tiers:
                y[(S, T)] = ys[T]
        s2, df, res = _stats_numpy(np, y, parts, cl_index, len(uniq), test)
        cnt["s2_sum"] += float(s2.sum())
        cnt["df"] = df
        for b in range(B):
            pv = [_p_of(float(D[b]), float(vc[b]), float(vs[b]), df, test) for D, vc, vs in res]
            _tally(cnt, pv, [float(D[b]) for D, _, _ in res], names, alpha)
        done += B
    return cnt


def _tally(cnt, pv, Ds, names, alpha) -> None:
    holm, better = _decide(pv, Ds, names, alpha)
    cnt["any"] += int(any(holm))
    for j in range(len(names)):
        cnt["unadj"][j] += int(pv[j] <= alpha)
        cnt["holm"][j] += int(holm[j])
        cnt["better"][j] += int(better[j])
        cnt["D_sum"][j] += Ds[j]


def persistence_check(p_rep: list, clusters, lam: float, sigma: float, n_reps: int, seed: int, *,
                      n_pooled: int = 3, a_priori: float = 0.03, chunk: int = 250, m_table: int = 2001,
                      lam_range=(0.0, 1.0), offsets=None) -> dict:
    """Simulation check of persistence_hat in the registered model (numpy; the draws of simulate, _draw_start_np):
    n_pooled pilot starts at the replicate tier with true p = p_rep, task persistence lam and seed SD sigma; the first
    start also has the replicate seed (the same persistent V, an independent shift).  offsets (default all 0): a
    realized SR offset per pooled start (the first is the replicate start), added to its seed shifts before
    calibrate_shift -- start main effects, which the registered estimator must not see.  Per replicate:
    sigma_used = seed_sd(first seed, replicate seed, a_priori), then persistence_hat(first seed, replicate seed, the
    other starts' seeds, sigma_used), exactly as run_tvr.  Returns the mean / SD / MC SE of lambda_hat (unclipped,
    over the replicates where it is defined), the mean scene-cluster SE, the mean lambda_used and the number of
    replicates where lambda_hat is undefined.  offsets all 0 draw exactly as before."""
    np = _np()
    if np is None:
        raise RuntimeError("persistence_check needs numpy")
    if n_reps < 2:
        raise ValueError("n_reps >= 2")
    off = [0.0] * n_pooled if offsets is None else [float(x) for x in offsets]
    if len(off) != n_pooled:
        raise ValueError("offsets: one per pooled start")
    tab = shift_table(p_rep, m_table)
    tabs = {"R": (np.asarray(tab[0]), np.asarray(tab[1]))}
    pa = {"R": np.asarray(p_rep, dtype=np.float64)}
    rng = np.random.default_rng(seed)
    hats, ses, used, undefined = [], [], [], 0
    done = 0
    while done < n_reps:
        B = min(chunk, n_reps - done)
        rep = _draw_start_np(np, rng, B, ["R"], {"R": [0, 1]}, 2, tabs, pa, sigma, 0.0, lam, off[0])["R"]
        others = [_draw_start_np(np, rng, B, ["R"], {"R": [0]}, 1, tabs, pa, sigma, 0.0, lam, off[j])["R"][:, 0, :]
                  for j in range(1, n_pooled)]
        for b in range(B):
            y1, y2 = rep[b, 0].astype(int).tolist(), rep[b, 1].astype(int).tolist()
            r = persistence_hat(y1, y2, [o[b].astype(int).tolist() for o in others],
                                seed_sd(y1, y2, a_priori)["sigma_used"], clusters, lam_range)
            used.append(r["lambda_used"])
            if r["lambda_hat"] is None:
                undefined += 1
                continue
            hats.append(r["lambda_hat"])
            if r["se"] is not None:
                ses.append(r["se"])
        done += B
    k = len(hats)
    mean = sum(hats) / k if k else None
    sd = math.sqrt(sum((x - mean) ** 2 for x in hats) / (k - 1)) if k > 1 else None
    return {"lambda_true": float(lam), "sigma": float(sigma), "n_reps": n_reps, "n_pooled": n_pooled,
            "offsets": off, "n_tasks": len(p_rep), "seed": seed, "mean_hat": mean,
            "bias": None if mean is None else mean - lam,
            "sd_hat": sd, "mc_se": None if sd is None else sd / math.sqrt(k),
            "mean_se": sum(ses) / len(ses) if ses else None, "mean_used": sum(used) / len(used),
            "n_undefined": undefined}


# --------------------------------------------------------------------------------------------- TVR pilot inputs
def eval_dir(pilot_root, start: str, tier: str, arm: str = "c") -> Path:
    return Path(pilot_root) / start / "evals" / f"pilot__{start}__{tier}__{arm}"


def read_per_task(path) -> dict:
    """{task_id: {"success": 0/1, "scene": str}} from a run's per_task.jsonl (refuses duplicates / bad values)."""
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            tid = r["task_id"]
            if tid in rows:
                raise ValueError(f"{path}: duplicate task {tid}")
            s = r["success"]
            if s not in (0, 1, True, False):
                raise ValueError(f"{path}: success {s!r} for {tid}")
            rows[tid] = {"success": int(s), "scene": str(r["scene"])}
    if not rows:
        raise ValueError(f"{path}: empty")
    return rows


def pooled_p(pilot_root, starts: list, tier_labels: dict, rows_out=None) -> tuple:
    """p_by_tier ({design tier: [p_i]}), clusters, task_ids, input shas.  p_i = mean over the starts (no
    difference).  Every start and tier must cover the same task ids with the same scenes.  rows_out (a dict,
    optional) receives {design tier: [one 0/1 row per start, in `starts` order, tasks in task_ids order]}
    (persistence_hat's input; never written to POWER.json)."""
    ids = None
    scenes = None
    per = {}
    shas = {}
    for T, label in tier_labels.items():
        for S in starts:
            p = eval_dir(pilot_root, S, label) / "per_task.jsonl"
            rows = read_per_task(p)
            shas[str(p)] = sha256_file(p)
            if ids is None:
                ids = sorted(rows)
                scenes = {t: rows[t]["scene"] for t in ids}
            if sorted(rows) != ids:
                raise ValueError(f"{p}: task ids differ from the first run's")
            if any(rows[t]["scene"] != scenes[t] for t in ids):
                raise ValueError(f"{p}: scenes differ")
            per[(S, T)] = rows
    p_by_tier = {T: [sum(per[(S, T)][t]["success"] for S in starts) / len(starts) for t in ids]
                 for T in tier_labels}
    if rows_out is not None:
        rows_out.update({T: [[per[(S, T)][t]["success"] for t in ids] for S in starts] for T in tier_labels})
    return p_by_tier, [scenes[t] for t in ids], ids, shas


def _p_summary(p: list) -> dict:
    n = len(p)
    return {"n": n, "mean": sum(p) / n, "n_p0": sum(1 for x in p if x == 0), "n_p1": sum(1 for x in p if x == 1)}


def _tc(pw) -> tuple:
    """(type-I rhos, power rho, reported rhos) of power.tier_corr."""
    tc = pw["tier_corr"]
    return [float(x) for x in tc["type1"]], float(tc["power"]), [float(x) for x in tc["reported"]]


def _sensitivity_superseded(p_by_tier, clusters, designs, sigma, lam, pw, ns, backend) -> dict:
    """Reported, NOT gated: the superseded seed_aware_cluster_t (independent tier shifts assumed) on both designs at
    sigma_used and lambda_used, null and primary delta, at every rho of power.tier_corr.type1 (0 and 1); shows why it
    was replaced (DECISIONS G13)."""
    test = "seed_aware_cluster_t"
    rho_t1, _, _ = _tc(pw)
    rows, summ = [], []
    for dname, des in designs.items():
        for rho in rho_t1:
            null, prim = (simulate(p_by_tier, clusters, des, dl, sigma, ns, int(pw["seed"]), alpha=float(pw["alpha"]),
                                   test=test, backend=backend, tier_corr=rho, persistence=lam)
                          for dl in (0.0, float(pw["primary_delta"])))
            rows += [dict(null, design=dname), dict(prim, design=dname)]
            summ.append({"design": dname, "test": test, "tier_corr": rho, "persistence": lam, "sigma": sigma,
                         "type1": null["fwer"], "power_at_primary": prim["power"], "df": null["df"]})
    return {"gated": False, "test": test, "registered": test == pw["test"], "sigma_label": "used",
            "note": "superseded test (assumes independent tier shifts within a (start, seed)); sensitivity only",
            "summary": summ, "rows": rows}


def _persistence(pw, sd, y1, y2, others, p_rep, clusters, backend) -> dict:
    """POWER.json["persistence"]: lambda from the seed replicate (persistence_hat, registered rule; others = the
    other pilot starts' rows at the replicate tier) and a reported simulation check of the estimator at the pooled
    p of the replicate tier (lambda_used, sigma_used)."""
    reg = pw["persistence"]
    lam_range = (float(reg["range"][0]), float(reg["range"][1]))
    if y1 is None:
        return {"replicate_missing": True, "lambda_hat": None, "se": None, "lambda_used": lam_range[1],
                "range": list(lam_range),
                "note": "replicate absent (allow_no_replicate): lambda_used = range upper end (conservative)"}
    est = persistence_hat(y1, y2, others, sd["sigma_used"], clusters, lam_range)
    est["replicate_missing"] = False
    est["rule"] = reg["rule"]
    chk = None
    if backend == "numpy":
        chk = persistence_check(p_rep, clusters, est["lambda_used"], sd["sigma_used"], int(reg["check_reps"]),
                                int(pw["seed"]), n_pooled=len(others) + 1, a_priori=float(pw["seed_sd"]["a_priori"]),
                                lam_range=lam_range)
    est["check"] = {"gated": False, "note": "persistence_hat on data simulated from the registered model at the pooled "
                                            "p of the replicate tier (lambda_used, sigma_used): bias = mean_hat - "
                                            "lambda_true", "result": chk}
    return est


def run_backend(pw: dict, backend=None) -> str:
    """The backend run_tvr uses: backend None = the registered power.backend (numpy; the python and numpy backends
    draw different random streams, so the registered seed pins the result only together with the backend); an
    explicit backend is a test override, recorded in POWER.json registered.backend (A4 counts only a POWER.json
    whose registered.backend == power.backend).  ValueError when it cannot run (e.g. numpy not importable)."""
    name = pw.get("backend", "auto") if backend is None else backend
    try:
        return _backend(name)
    except RuntimeError as e:
        raise ValueError(f"backend {name!r}: {e}") from None


def run_tvr(pilot_root, cfg=None, out=None, *, n_sims=None, backend=None,
            allow_no_replicate: bool = False) -> dict:
    """POWER.json for the TVR pilot (A4).  Run only after G0-G3 passed (gate_tvr).  cfg None = pilot_cfg.FROZEN
    (read at call time).  The seed replicate (O4) is required unless allow_no_replicate (then sigma_used = a priori
    and lambda_used = the upper end of power.persistence.range; recorded; gate_tvr never reads such a POWER.json).
    backend None = power.backend (run_backend); n_sims / backend overrides are for tests and recorded.

    Grid per design (main; fallback_seeds_1_2 = main with power.fallback.seeds) and sigma row (used, a_priori,
    sensitivity), all at lambda_used: role "type1" = delta 0 at every rho of tier_corr.type1; role "power" = every
    delta of power.deltas at tier_corr.power; role "reported" = primary delta at every rho of tier_corr.reported.
    Role "sensitivity_persistence" (sigma_used): delta 0 at the type-I rhos and the primary delta at the power rho,
    for every lambda of power.persistence.sensitivity.  Only sigma_used / lambda_used rows of roles type1 and power
    enter the verdict."""
    if cfg is None:
        cfg = pilot_cfg.FROZEN
    pw = cfg["power"]
    backend = run_backend(pw, backend)
    ntop = pilot_cfg.n_top(cfg)
    rep = cfg["tiers"]["replicate"]
    starts = list(cfg["starts"])
    labels = {"100": "100", "TOP": ntop}
    rep_T = next((T for T, lab in labels.items() if lab == rep["tier"]), None)
    if rep_T is None:
        raise ValueError(f"replicate tier {rep['tier']!r} is not a pooled tier {labels}")
    if rep["start"] not in starts:
        raise ValueError(f"replicate start {rep['start']!r} is not a pooled start {starts}")
    rows = {}
    p_by_tier, clusters, ids, shas = pooled_p(pilot_root, starts, labels, rows_out=rows)
    # seed replicate (same start, two seeds): Q3B 100 seed 1001 vs 100s1002
    rep_label = f"{rep['tier']}s{rep['seed']}"
    p1 = eval_dir(pilot_root, rep["start"], rep["tier"]) / "per_task.jsonl"
    p2 = eval_dir(pilot_root, rep["start"], rep_label) / "per_task.jsonl"
    a_priori = float(pw["seed_sd"]["a_priori"])
    y1 = y2 = None
    if p2.is_file():
        r1, r2 = read_per_task(p1), read_per_task(p2)
        if sorted(r1) != ids or sorted(r2) != ids:
            raise ValueError("seed replicate task ids differ from the pooled task ids")
        if any(r2[t]["scene"] != r1[t]["scene"] for t in ids):
            raise ValueError("seed replicate scenes differ")
        y1, y2 = [r1[t]["success"] for t in ids], [r2[t]["success"] for t in ids]
        sd = seed_sd(y1, y2, a_priori)
        sd["replicate"] = {"seed_a": cfg["seed"], "run_a": str(p1), "seed_b": rep["seed"], "run_b": str(p2)}
        sd["replicate_missing"] = False
        shas[str(p2)] = sha256_file(p2)
        shas[str(p1)] = sha256_file(p1)
    elif allow_no_replicate:
        sd = {"replicate_missing": True, "sigma_hat": None, "a_priori": a_priori, "sigma_used": a_priori,
              "note": "replicate absent and allow_no_replicate given: sigma_used = a priori (NOT the O4 design)"}
    else:
        raise ValueError(f"seed replicate {p2} missing (O4 requires it); pass allow_no_replicate / "
                         "--allow-no-replicate to use the a priori sigma")
    others = [row for S, row in zip(starts, rows[rep_T]) if S != rep["start"]]
    pers = _persistence(pw, sd, y1, y2, others, p_by_tier[rep_T], clusters, backend)
    lam = float(pers["lambda_used"])
    sig_rows = [("used", sd["sigma_used"]), ("a_priori", a_priori)]
    sig_rows += [("sensitivity", float(s)) for s in pw["seed_sd"]["sensitivity"]]
    rho_t1, rho_pw, rho_rep = _tc(pw)
    prim_dl = float(pw["primary_delta"])
    cells = []   # (sigma_label, sigma, delta, tier_corr, persistence, role)
    for label, sig in sig_rows:
        cells += [(label, sig, 0.0, rho, lam, "type1") for rho in rho_t1]
        cells += [(label, sig, float(dl), rho_pw, lam, "power") for dl in pw["deltas"]]
        cells += [(label, sig, prim_dl, rho, lam, "reported") for rho in rho_rep]
    for lv in pw["persistence"]["sensitivity"]:
        cells += [("used", sd["sigma_used"], 0.0, rho, float(lv), "sensitivity_persistence") for rho in rho_t1]
        cells.append(("used", sd["sigma_used"], prim_dl, rho_pw, float(lv), "sensitivity_persistence"))
    ns = int(pw["n_sims"] if n_sims is None else n_sims)
    main = copy.deepcopy(pw["design"])
    fallback = copy.deepcopy(main)
    fallback["seeds"] = copy.deepcopy(pw["fallback"]["seeds"])
    designs = {"main": main, "fallback_seeds_1_2": fallback}
    tables = {}
    for dname, des in designs.items():
        rows = []
        done = {}
        for label, sig, dl, rho, lv, role in cells:
            key = (round(sig, 12), dl, rho, lv)
            if key not in done:
                done[key] = simulate(p_by_tier, clusters, des, dl, sig, ns, int(pw["seed"]), alpha=float(pw["alpha"]),
                                     test=pw["test"], backend=backend, tier_corr=rho, persistence=lv)
            rows.append(dict(done[key], sigma_label=label, role=role))
        tables[dname] = rows

    def verdict(rows):
        gated = [r for r in rows if r["sigma_label"] == "used" and r["persistence"] == lam]
        t1 = {r["tier_corr"]: r["fwer"] for r in gated if r["role"] == "type1"}
        prim = [r for r in gated if r["role"] == "power" and r["delta"] == prim_dl and r["tier_corr"] == rho_pw]
        pwr = prim[0]["power"] if prim else {}
        type1 = max(t1.values()) if t1 else None
        ok_t1 = bool(t1) and all(rho in t1 for rho in rho_t1) and type1 <= float(pw["type1_max"])
        ok_pw = bool(pwr) and all(v >= float(pw["target_power"]) for v in pwr.values())
        by_rho = {str(r["tier_corr"]): r["power"] for r in gated if r["role"] in ("power", "reported")
                  and r["delta"] == prim_dl}
        return {"type1": type1, "type1_ok": ok_t1, "type1_by_rho": {str(k): v for k, v in t1.items()},
                "power_at_primary": pwr, "power_tier_corr": rho_pw,
                "mc_se_power_at_primary": prim[0]["mc_se_power"] if prim else {}, "power_ok": ok_pw,
                "power_by_rho": by_rho, "persistence": lam, "sigma": sd["sigma_used"], "pass": ok_t1 and ok_pw}

    res = {"schema": SCHEMA, "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "registered": {"test": pw["test"], "test_def": pw["test_def"], "power_counts": pw["power_counts"],
                          "seed_model": pw["seed_model"], "p_source": pw["p_source"], "alpha": pw["alpha"],
                          "primary_delta": pw["primary_delta"], "target_power": pw["target_power"],
                          "type1_max": pw["type1_max"], "type1_definition":
                              "family-wise: fraction of null simulations with >= 1 Holm rejection (either direction); "
                              "gated value = max over power.tier_corr.type1",
                          "tier_corr": copy.deepcopy(pw["tier_corr"]), "fallback": copy.deepcopy(pw["fallback"]),
                          "persistence": copy.deepcopy(pw["persistence"]),
                          "n_sims_registered": pw["n_sims"], "n_sims_used": ns, "seed": pw["seed"],
                          # the backend THIS run used (the random stream depends on it); A4 needs power.backend
                          "backend": backend, "allow_no_replicate": bool(allow_no_replicate)},
           "inputs": {"pilot_root": str(pilot_root), "tiers": labels, "starts_pooled": starts, "n_tasks": len(ids),
                      "task_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                      "files_sha256": shas,
                      # sha256 of pilot_cfg.as_json(cfg) (the cfg actually used); NOT the file sha that GATE.json's
                      # pilot_cfg_sha256 holds -- that one is pilot_cfg_py_sha256 here
                      "pilot_cfg_json_sha256": hashlib.sha256(pilot_cfg.as_json(cfg).encode()).hexdigest(),
                      "pilot_cfg_py_sha256": sha256_file(pilot_cfg.__file__),
                      "power_sim_sha256": sha256_file(__file__)},
           "p_summary": {T: _p_summary(v) for T, v in p_by_tier.items()},
           "n_clusters": len(set(clusters)), "sigma": sd, "persistence": pers, "tables": tables,
           "verdicts": {k: verdict(v) for k, v in tables.items()},
           "sensitivity_superseded_test": _sensitivity_superseded(p_by_tier, clusters, designs, sd["sigma_used"], lam,
                                                                  pw, ns, backend),
           "note": "p_i pools the three pilot starts by a mean; lambda reads the other starts only through "
                   "covariances over tasks (invariant to start mean SRs); no between-start difference is computed"}
    res["pass"] = res["verdicts"]["main"]["pass"]
    res["fallback_pass"] = res["verdicts"]["fallback_seeds_1_2"]["pass"]
    if out is not None:
        write_json_atomic(out, res)
    return res


# --------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vpd.common.power_sim", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tvr", help="POWER.json for the TVR pilot (A4)")
    t.add_argument("--pilot-root", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--n-sims", type=int, default=None, help="override (tests only; recorded as n_sims_used)")
    t.add_argument("--backend", default=None, choices=("auto", "numpy", "python"),
                   help="override of the registered power.backend (tests only; recorded as registered.backend)")
    t.add_argument("--allow-no-replicate", action="store_true",
                   help="accept a root without the Q3B 100s1002 replicate (sigma_used = a priori, lambda_used = the "
                        "upper end of power.persistence.range; recorded)")
    a = ap.parse_args(argv)
    try:
        res = run_tvr(a.pilot_root, out=a.out, n_sims=a.n_sims, backend=a.backend,
                      allow_no_replicate=a.allow_no_replicate)
    except (OSError, ValueError, KeyError) as e:
        print(f"power_sim: refused: {e!r}", file=sys.stderr)
        return 2
    v = res["verdicts"]["main"]
    print(f"power_sim tvr: pass={res['pass']} type1={v['type1']:.4f} power@{res['registered']['primary_delta']}"
          f"(tier_corr {v['power_tier_corr']})={json.dumps(v['power_at_primary'])} "
          f"sigma_used={res['sigma']['sigma_used']:.4f} lambda_used={res['persistence']['lambda_used']:.4f} "
          f"fallback_pass={res['fallback_pass']} -> {a.out}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
