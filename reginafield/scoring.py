from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def composite_structural_score(
    df: pd.DataFrame,
    motif_col: str = "MotifSum",
    entropy_col: str = "Entropy",
    curvature_col: str = "Curvature",
    hilbert_col: str | None = "HilbertMag",
    weights: Mapping[str, float] | None = None,
    out_col: str = "CompositeScore",
) -> pd.DataFrame:
    """Compute a normalized composite structural score.

    The score is a weighted combination of normalized:
        - |MotifSum|
        - |Entropy|
        - |Curvature|
        - |HilbertMag| (if present and hilbert_col is not None)
    """
    out = df.copy()

    if weights is None:
        weights = {
            "motif": 1.0,
            "entropy": 1.0,
            "curvature": 1.0,
            "hilbert": 1.0 if hilbert_col is not None and hilbert_col in out.columns else 0.0,
        }

    def norm(series: pd.Series) -> np.ndarray:
        arr = series.to_numpy(dtype=float)
        m = np.nanmax(np.abs(arr)) or 1.0
        return arr / m

    motif_n = norm(out[motif_col])
    ent_n = norm(out[entropy_col])
    curv_n = norm(out[curvature_col])

    score = (
        weights.get("motif", 0.0) * np.abs(motif_n)
        + weights.get("entropy", 0.0) * np.abs(ent_n)
        + weights.get("curvature", 0.0) * np.abs(curv_n)
    )

    if hilbert_col is not None and hilbert_col in out.columns:
        hilb_n = norm(out[hilbert_col])
        score += weights.get("hilbert", 0.0) * np.abs(hilb_n)

    # rescale to [0, 1]
    s_min = float(np.nanmin(score))
    s_max = float(np.nanmax(score))
    if s_max > s_min:
        score = (score - s_min) / (s_max - s_min)
    else:
        score = np.zeros_like(score)

    out[out_col] = score
    return out


def rank_by_structural_score(
    df: pd.DataFrame,
    score_col: str = "CompositeScore",
    ascending: bool = False,
) -> pd.DataFrame:
    """Return a copy sorted by the composite score."""
    if score_col not in df.columns:
        raise ValueError(f"DataFrame is missing '{score_col}'")
    return df.sort_values(score_col, ascending=ascending).reset_index(drop=True)