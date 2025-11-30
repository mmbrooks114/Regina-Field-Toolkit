from __future__ import annotations

import numpy as np
import pandas as pd


def classify_attractor_zones(
    df: pd.DataFrame,
    entropy_col: str = "Entropy",
    curvature_col: str = "Curvature",
    motif_col: str = "MotifSum",
    x_col: str = "PCA_X",
    y_col: str = "PCA_Y",
    out_col: str = "Zone",
) -> pd.DataFrame:
    """Assign attractor zone labels based on entropy, curvature, and radius.

    Zones:
        - Primary
        - Secondary
        - Resonance
        - CurvatureBasin
        - General

    Uses quantile-based thresholds, mirroring the public attractor_zones.csv
    generation logic but keeping everything deterministic and fast.
    """
    out = df.copy()

    ent = out[entropy_col].to_numpy(dtype=float)
    curv = out[curvature_col].to_numpy(dtype=float)
    motif = out[motif_col].to_numpy(dtype=float)
    x = out[x_col].to_numpy(dtype=float)
    y = out[y_col].to_numpy(dtype=float)
    radius = np.sqrt(x * x + y * y)

    q_curv_low = np.quantile(curv, 0.2)
    q_curv_mid = np.quantile(curv, 0.5)
    q_ent_low = np.quantile(ent, 0.25)
    q_ent_high = np.quantile(ent, 0.75)
    q_motif_high = np.quantile(motif, 0.9)
    q_rad_mid = np.quantile(radius, 0.5)
    q_rad_high = np.quantile(radius, 0.8)
    q_abs_curv_small = np.quantile(np.abs(curv), 0.2)

    zone = np.full(out.shape[0], "General", dtype=object)

    # Curvature basins: very small |curv|
    mask_basin = np.abs(curv) <= q_abs_curv_small
    zone[mask_basin] = "CurvatureBasin"

    # Primary: low curvature, mid entropy, high radius
    mask_primary = (
        (curv <= q_curv_low)
        & (ent >= q_ent_low)
        & (ent <= q_ent_high)
        & (radius >= q_rad_high)
    )
    zone[mask_primary] = "Primary"

    # Resonance: high motif and mid entropy (where not already primary/basin)
    mask_res = (
        (zone == "General")
        & (motif >= q_motif_high)
        & (ent >= q_ent_low)
        & (ent <= q_ent_high)
    )
    zone[mask_res] = "Resonance"

    # Secondary: mid-to-low curvature, mid-to-high radius
    mask_secondary = (
        (zone == "General")
        & (curv <= q_curv_mid)
        & (radius >= q_rad_mid)
    )
    zone[mask_secondary] = "Secondary"

    out[out_col] = zone
    out["Radius"] = radius
    return out


def label_anomaly_bands(
    df: pd.DataFrame,
    score_col: str = "AnomalyScore",
    out_col: str = "AnomalyBand",
    n_bands: int = 5,
) -> pd.DataFrame:
    """Assign coarse anomaly bands based on quantiles of an anomaly score column."""
    out = df.copy()
    score = out[score_col].astype(float)
    quantiles = np.linspace(0.0, 1.0, n_bands + 1)
    bins = score.quantile(quantiles).to_numpy()
    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1e-9
    labels = [str(i) for i in range(n_bands)]
    out[out_col] = pd.cut(score, bins=bins, labels=labels, include_lowest=True)
    return out