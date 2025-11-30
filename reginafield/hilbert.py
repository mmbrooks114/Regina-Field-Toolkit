from __future__ import annotations

import numpy as np
import pandas as pd


def hilbert_envelope(signal: np.ndarray) -> np.ndarray:
    """Compute Hilbert envelope (|analytic signal|) of a 1D real signal."""
    signal = np.asarray(signal, dtype=float)
    n = signal.size
    if n == 0:
        return np.array([], dtype=float)
    fft_vals = np.fft.fft(signal)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    analytic = np.fft.ifft(fft_vals * h)
    return np.abs(analytic)


def detect_local_peaks(envelope: np.ndarray) -> np.ndarray:
    """Return a boolean mask of local maxima for a Hilbert envelope."""
    env = np.asarray(envelope, dtype=float)
    n = env.size
    if n < 3:
        return np.zeros(n, dtype=bool)
    is_peak = np.zeros(n, dtype=bool)
    left = env[:-2]
    mid = env[1:-1]
    right = env[2:]
    peak_inner = (mid > left) & (mid > right)
    is_peak[1:-1] = peak_inner
    return is_peak