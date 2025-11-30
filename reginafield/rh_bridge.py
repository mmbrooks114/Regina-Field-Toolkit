from __future__ import annotations

import numpy as np
import pandas as pd
from mpmath import li as mpmath_li


def compute_pi_x(primes: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Compute π(x) for a sorted list of primes and an x-grid."""
    primes = np.asarray(primes, dtype=int)
    x_grid = np.asarray(x_grid, dtype=float)
    primes_sorted = np.sort(primes)

    counts = np.searchsorted(primes_sorted, x_grid, side="right")
    return counts.astype(float)


def compute_li_x(x_grid: np.ndarray) -> np.ndarray:
    """Compute the logarithmic integral Li(x) using mpmath."""
    x_grid = np.asarray(x_grid, dtype=float)
    return np.array([float(mpmath_li(x)) for x in x_grid])


def compute_regina_cumulatives(
    master: pd.DataFrame,
    x_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute simple cumulative Regina-side observables over x.

    Uses:
        - cumulative curvature
        - cumulative |RoyalIndex| (if present)
        - cumulative entropy
    """
    x_grid = np.asarray(x_grid, dtype=float)
    candidates = master["Candidate"].to_numpy(dtype=float)

    order = np.argsort(candidates)
    cand_sorted = candidates[order]
    curv_sorted = master["Curvature"].to_numpy(dtype=float)[order]
    ent_sorted = master["Entropy"].to_numpy(dtype=float)[order]
    royal_abs = None
    if "RoyalIndex" in master.columns:
        royal_abs = np.abs(master["RoyalIndex"].to_numpy(dtype=float))[order]

    curv_cum = np.cumsum(curv_sorted)
    ent_cum = np.cumsum(ent_sorted)
    royal_cum = np.cumsum(royal_abs) if royal_abs is not None else None

    idxs = np.searchsorted(cand_sorted, x_grid, side="right") - 1
    idxs[idxs < 0] = 0

    out = {
        "K_cum": curv_cum[idxs],
        "H_cum": ent_cum[idxs],
    }
    if royal_cum is not None:
        out["E_R"] = royal_cum[idxs]
    return out