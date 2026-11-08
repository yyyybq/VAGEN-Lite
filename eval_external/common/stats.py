"""Confidence intervals used in every eval_external table."""

from __future__ import annotations

import math


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion k/n and its Wilson score interval (95% for z=1.96), all as fractions in [0, 1].

    k may be fractional (e.g. an MRA-scored item counts its score); the interval then treats the mean score as a
    proportion, which is what the tables state next to MRA numbers.
    """
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def fmt_pct_ci(k: float, n: int) -> str:
    """'54.7 [46.7, 62.4] (n=150)' style cell."""
    p, lo, hi = wilson(k, n)
    return f"{100 * p:.1f} [{100 * lo:.1f}, {100 * hi:.1f}] (n={n})"
