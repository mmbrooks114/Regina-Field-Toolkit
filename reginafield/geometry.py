from __future__ import annotations

import numpy as np
import pandas as pd


def add_shell_index(
    df: pd.DataFrame,
    x_col: str = "PCA_X",
    y_col: str = "PCA_Y",
    n_shells: int = 5,
    out_col: str = "ShellIndex",
) -> pd.DataFrame:
    """Assign shell indices based on radius quantiles in PCA space."""
    out = df.copy()
    x = out[x_col].to_numpy(dtype=float)
    y = out[y_col].to_numpy(dtype=float)
    radius = np.sqrt(x * x + y * y)
    quantiles = np.linspace(0.0, 1.0, n_shells + 1)
    bins = np.quantile(radius, quantiles)
    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1e-9
    out[out_col] = pd.cut(radius, bins=bins, labels=list(range(n_shells)), include_lowest=True)
    out["Radius"] = radius
    return out