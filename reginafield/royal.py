from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def attach_royal_status(
    master: pd.DataFrame,
    royal_df: pd.DataFrame,
    candidate_col: str = "Candidate",
    royal_index_col: str = "RoyalIndex",
    out_flag_col: str = "IsRoyal",
) -> pd.DataFrame:
    """Attach RoyalIndex and a boolean IsRoyal flag to the master dataframe."""
    if candidate_col not in master.columns:
        raise ValueError(f"master is missing '{candidate_col}'")
    if candidate_col not in royal_df.columns:
        raise ValueError(f"royal_df is missing '{candidate_col}'")
    if royal_index_col not in royal_df.columns:
        raise ValueError(f"royal_df is missing '{royal_index_col}'")

    merged = master.merge(
        royal_df[[candidate_col, royal_index_col]],
        on=candidate_col,
        how="left",
        suffixes=("", "_royal"),
    )
    merged[out_flag_col] = merged[royal_index_col].notna()
    return merged


def filter_royal(df: pd.DataFrame, is_royal_col: str = "IsRoyal") -> pd.DataFrame:
    """Return only rows labeled as Royal."""
    if is_royal_col not in df.columns:
        raise ValueError(f"DataFrame is missing '{is_royal_col}'")
    return df[df[is_royal_col]].copy()


def estimate_extremal_ray_direction(
    df: pd.DataFrame,
    x_col: str = "PCA_X",
    y_col: str = "PCA_Y",
    is_royal_col: str = "IsRoyal",
) -> Tuple[float, float]:
    """Estimate a direction vector for the extremal ray from Royal primes.

    Returns a unit vector (dx, dy).
    """
    if is_royal_col not in df.columns:
        raise ValueError(f"DataFrame is missing '{is_royal_col}'")

    royals = df[df[is_royal_col]]
    if royals.empty:
        raise ValueError("No Royal points found to estimate extremal ray.")

    x = royals[x_col].to_numpy(dtype=float)
    y = royals[y_col].to_numpy(dtype=float)

    # direction from origin to mean Royal position
    mean_x = float(x.mean())
    mean_y = float(y.mean())
    norm = (mean_x**2 + mean_y**2) ** 0.5 or 1.0
    return mean_x / norm, mean_y / norm


def project_onto_extremal_ray(
    df: pd.DataFrame,
    direction: tuple[float, float],
    x_col: str = "PCA_X",
    y_col: str = "PCA_Y",
    out_col: str = "RayProjection",
) -> pd.DataFrame:
    """Project each point onto a given extremal ray direction."""
    out = df.copy()
    dx, dy = direction
    x = out[x_col].to_numpy(dtype=float)
    y = out[y_col].to_numpy(dtype=float)
    out[out_col] = x * dx + y * dy
    return out