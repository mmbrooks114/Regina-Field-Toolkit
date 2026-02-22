# Regina Field Toolkit

A modular, lightweight Python library for analyzing, scoring, and visualizing the structured geometry of prime numbers discovered in the **Regina Field** — a multi-layered system derived from motif decomposition, entropy, curvature, Royal primes, geometric embeddings, Hilbert envelopes, and attractor zone dynamics.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![OSF Registration](https://img.shields.io/badge/OSF-8HQ9B-blue)](https://osf.io/8hq9b/overview?view_only=8b619f64c42b4194b44bea927013dbcc)

This toolkit supports:

- loading the official Regina Field datasets  
- assigning field geometry features (radius, tiers, shells, zones)  
- computing attractor zones  
- overlaying Royal primes  
- Hilbert envelope & structural peak detection  
- composite structural scoring  
- CLI utilities  
- Riemann-error bridges for further research  

The toolkit is designed for reproducibility and for direct use alongside the Regina Field Whitepaper and OSF datasets.

---

## 🚀 Features

- **Dataset loaders** with column validation  
- **Attractor zone classification**  
- **Entropy and curvature tiering**  
- **PCA geometry utilities**  
- **Royal prime analysis**  
- **Hilbert envelope computation + peak detection**  
- **Composite structural scoring**  
- **Visualization utilities**  
- **Minimal, clean CLI tool: `reginafield-cli`**  
- **Supports large datasets** (10M primes) efficiently  

---

## 📦 Installation

Clone the repository and install in editable mode:

```bash
pip install -e .
