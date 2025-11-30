from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class DatasetPaths:
    enriched_master: str
    royal_primes: Optional[str] = None
    curvature_field: Optional[str] = None
    motif_entropy_map: Optional[str] = None
    attractor_zones: Optional[str] = None
    hilbert_peaks: Optional[str] = None
    anomaly_clusters: Optional[str] = None


REQUIRED_MASTER_COLUMNS = [
    "Candidate",
    "MotifSum",
    "Entropy",
    "Curvature",
    "PCA_X",
    "PCA_Y",
]


def _check_required_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_enriched_master(path: str) -> pd.DataFrame:
    """Load ReginaField_EnrichedMaster.csv with basic validation."""
    df = pd.read_csv(path)
    _check_required_columns(df, REQUIRED_MASTER_COLUMNS, "Enriched master")
    return df


def load_royal_primes(path: str) -> pd.DataFrame:
    """Load Royal_Primes.csv.

    Expects at minimum: Candidate, RoyalIndex.
    """
    df = pd.read_csv(path)
    if "Candidate" not in df.columns:
        raise ValueError("Royal_Primes file must contain a 'Candidate' column.")
    if "RoyalIndex" not in df.columns:
        raise ValueError("Royal_Primes file must contain a 'RoyalIndex' column.")
    return df


def load_curvature_field(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _check_required_columns(df, ["Candidate", "Curvature"], "Curvature field")
    return df


def load_motif_entropy_map(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _check_required_columns(df, ["Candidate", "Entropy", "MotifSum"], "Motif–entropy map")
    return df


def load_attractor_zones(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _check_required_columns(df, ["Candidate", "Zone"], "Attractor zones")
    return df


def load_hilbert_peaks(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _check_required_columns(df, ["Candidate", "HilbertMag"], "Hilbert peaks")
    return df


def load_anomaly_clusters(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _check_required_columns(df, ["Candidate", "AnomalyScore", "ClusterID"], "Anomaly clusters")
    return df


def merge_master_with_royal(master: pd.DataFrame, royal: pd.DataFrame) -> pd.DataFrame:
    """Merge enriched master with Royal_Primes on Candidate.

    Adds:
        - RoyalIndex
        - IsRoyal (bool)
    """
    merged = master.merge(
        royal[["Candidate", "RoyalIndex"]],
        on="Candidate",
        how="left",
        suffixes=("", "_royal"),
    )
    merged["IsRoyal"] = merged["RoyalIndex"].notna()
    return merged