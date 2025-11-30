from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def _get_color_values(df: pd.DataFrame, color_by: Optional[str]):
    if color_by is None:
        return None
    if color_by not in df.columns:
        raise ValueError(f"Column '{color_by}' not in DataFrame.")
    return df[color_by].to_numpy()


def plot_pca_field(
    df: pd.DataFrame,
    x_col: str = "PCA_X",
    y_col: str = "PCA_Y",
    color_by: Optional[str] = None,
    alpha: float = 0.7,
    s: float = 5.0,
) -> None:
    """Scatter plot of the field in PCA space."""
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    c = _get_color_values(df, color_by)

    plt.figure()
    plt.scatter(x, y, c=c, alpha=alpha, s=s)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if color_by is not None:
        plt.title(f"PCA field colored by {color_by}")
    else:
        plt.title("PCA field")
    plt.tight_layout()
    plt.show()



def plot_entropy_vs_curvature(
    df: pd.DataFrame,
    entropy_col: str = "Entropy",
    curvature_col: str = "Curvature",
    color_by: Optional[str] = None,
    alpha: float = 0.7,
    s: float = 5.0,
) -> None:
    """Scatter plot of Entropy vs Curvature."""
    x = df[entropy_col].to_numpy()
    y = df[curvature_col].to_numpy()
    c = _get_color_values(df, color_by)

    plt.figure()
    plt.scatter(x, y, c=c, alpha=alpha, s=s)
    plt.xlabel(entropy_col)
    plt.ylabel(curvature_col)
    if color_by is not None:
        plt.title(f"Entropy vs Curvature (colored by {color_by})")
    else:
        plt.title("Entropy vs Curvature")
    plt.tight_layout()
    plt.show()



def plot_royal_vs_field(
    df: pd.DataFrame,
    x_col: str = "PCA_X",
    y_col: str = "PCA_Y",
    is_royal_col: str = "IsRoyal",
    alpha_field: float = 0.3,
    alpha_royal: float = 0.9,
    s_field: float = 4.0,
    s_royal: float = 12.0,
) -> None:
    """Overlay Royal primes on the full field in PCA space."""
    if is_royal_col not in df.columns:
        raise ValueError(f"DataFrame is missing '{is_royal_col}'")

    field = df.copy()
    royals = field[field[is_royal_col]]
    others = field[~field[is_royal_col]]

    plt.figure()
    # others
    plt.scatter(
        others[x_col].to_numpy(),
        others[y_col].to_numpy(),
        alpha=alpha_field,
        s=s_field,
    )
    # royals
    plt.scatter(
        royals[x_col].to_numpy(),
        royals[y_col].to_numpy(),
        alpha=alpha_royal,
        s=s_royal,
    )
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Royal primes vs field (PCA)")
    plt.tight_layout()
    plt.show()
