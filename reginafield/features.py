from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def add_radius(df: pd.DataFrame, x_col: str = "PCA_X", y_col: str = "PCA_Y", out_col: str = "Radius") -> pd.DataFrame:
    """Add Euclidean radius in PCA (x, y) space.

    Does not modify df in-place; returns a copy with a new column.
    """
    out = df.copy()
    x = out[x_col].to_numpy()
    y = out[y_col].to_numpy()
    out[out_col] = np.sqrt(x * x + y * y)
    return out


def add_entropy_tiers(
    df: pd.DataFrame,
    col: str = "Entropy",
    n_bins: int = 4,
    out_col: str = "EntropyTier",
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Add a categorical tier label based on quantiles of the given entropy column."""
    out = df.copy()
    series = out[col].astype(float)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bins = series.quantile(quantiles).to_numpy()
    # ensure strictly increasing
    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1e-9
    if labels is None:
        labels = [f"T{i}" for i in range(n_bins)]
    out[out_col] = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
    return out


def add_curvature_tiers(
    df: pd.DataFrame,
    col: str = "Curvature",
    n_bins: int = 4,
    out_col: str = "CurvatureTier",
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Add a categorical tier label based on quantiles of |Curvature|."""
    out = df.copy()
    series = out[col].astype(float).abs()
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bins = series.quantile(quantiles).to_numpy()
    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1e-9
    if labels is None:
        labels = [f"C{i}" for i in range(n_bins)]
    out[out_col] = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
    return out