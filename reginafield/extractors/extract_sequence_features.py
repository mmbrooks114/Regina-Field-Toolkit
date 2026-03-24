from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def decompose_gap_sequence(gaps: List[int]) -> List[int]:
    """
    Convert prime gaps into a step sequence.

    Rule:
    - keep most gaps as-is
    - decompose gap 6 into [2,4] or [4,2] based on local continuity:
        * if previous emitted step was 2 -> emit [4,2]
        * if previous emitted step was 4 -> emit [2,4]
        * otherwise default to [2,4]
    """
    steps: List[int] = []
    prev = None

    for g in gaps:
        if g == 6:
            if prev == 2:
                seq = [4, 2]
            elif prev == 4:
                seq = [2, 4]
            else:
                seq = [2, 4]
            steps.extend(seq)
            prev = seq[-1]
        else:
            steps.append(int(g))
            prev = int(g)

    return steps


def shannon_entropy(values: List[int]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=int)
    uniq, counts = np.unique(arr, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def local_curvature(values: List[int]) -> float:
    """
    Simple sequence curvature proxy from second differences.
    Returns 0 for too-short windows.
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) < 3:
        return 0.0
    second = np.diff(arr, n=2)
    return float(second.mean())


def compute_sequence_features(
    candidates: np.ndarray,
    gap_window: int = 6,
) -> pd.DataFrame:
    """
    Build true sequence-based features for each prime.

    For each prime p_i:
    - look at recent prime gaps ending at i
    - decompose the local gap window into motif steps
    - compute:
        MotifSum   = sum of emitted motif steps
        Entropy    = Shannon entropy of emitted motif steps
        Curvature  = mean second difference of emitted motif steps

    Notes:
    - first rows have smaller history windows
    - no future leakage: each row only uses past/current sequence context
    """
    n = len(candidates)
    if n == 0:
        return pd.DataFrame(columns=["Candidate", "MotifSum", "Entropy", "Curvature"])

    gaps = np.diff(candidates)
    motif_sum = np.zeros(n, dtype=float)
    entropy = np.zeros(n, dtype=float)
    curvature = np.zeros(n, dtype=float)

    # first prime has no previous gap context
    motif_sum[0] = 0.0
    entropy[0] = 0.0
    curvature[0] = 0.0

    for i in range(1, n):
        # gaps available up to index i-1
        start = max(0, i - gap_window)
        local_gaps = gaps[start:i].tolist()
        steps = decompose_gap_sequence(local_gaps)

        motif_sum[i] = motif_sum[i] = float(sum(steps)) / max(1, len(steps))
        entropy[i] = shannon_entropy(steps)
        curvature[i] = local_curvature(steps)

    return pd.DataFrame(
        {
            "Candidate": candidates,
            "MotifSum": motif_sum,
            "Entropy": entropy,
            "Curvature": curvature,
        }
    )


def extract_features(df: pd.DataFrame, gap_window: int = 6) -> pd.DataFrame:
    if "Candidate" not in df.columns:
        raise ValueError("Input CSV must contain a 'Candidate' column")

    out = df.copy()
    candidates = pd.to_numeric(out["Candidate"], errors="raise").to_numpy(dtype=np.int64)
    order = np.argsort(candidates)
    sorted_candidates = candidates[order]

    feat = compute_sequence_features(sorted_candidates, gap_window=gap_window)

    # map back to original row order if needed
    feat = feat.set_index("Candidate")
    out["MotifSum"] = pd.to_numeric(out["Candidate"], errors="raise").map(feat["MotifSum"])
    out["Entropy"] = pd.to_numeric(out["Candidate"], errors="raise").map(feat["Entropy"])
    out["Curvature"] = pd.to_numeric(out["Candidate"], errors="raise").map(feat["Curvature"])

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV with 'Candidate' column")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--gap-window", type=int, default=6, help="Number of recent prime gaps to use per row")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df = extract_features(df, gap_window=args.gap_window)
    df.to_csv(args.output, index=False)
    print(f"Saved enriched file to: {args.output}")


if __name__ == "__main__":
    main()