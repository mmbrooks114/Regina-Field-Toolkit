"""Regina Field Toolkit

Lightweight utilities for loading, analyzing, and visualizing
the Regina Field datasets (motifs, entropy, curvature, geometry,
Royal primes, attractor zones, anomalies, and Hilbert structure).
"""

from . import datasets, features, zones, royal, geometry, hilbert, scoring, visualize, rh_bridge

__all__ = [
    "datasets",
    "features",
    "zones",
    "royal",
    "geometry",
    "hilbert",
    "scoring",
    "visualize",
    "rh_bridge",
]