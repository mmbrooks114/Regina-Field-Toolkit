# Regina Field Toolkit (Sequence-Based Release)

This toolkit provides feature extraction and analysis utilities for studying structural properties of prime numbers using the Regina Field framework.

## Legacy Version

For the previous digit-based feature extraction approach, see:

`README_legacy.md`

---

## Overview

The toolkit computes structural features derived from **prime step sequences**, including:

- **MotifSum** — motif-based structural signal derived from step decomposition  
- **Entropy** — distributional complexity of motif patterns  
- **Curvature** — second-order structural variation across motifs  

These features define an embedding of primes into a **curvature–entropy feature space**, which is used for structural analysis of prime distributions.

---

## Feature Extraction

### Script: `extract_sequence_features.py`

This script processes a CSV of candidate numbers and computes:

- MotifSum (sequence-based motif structure)
- Entropy (motif distribution)
- Curvature (structural variation)

---

## Input Format

The input CSV must contain a column named:

`Candidate`

Each row should contain an integer value.

---

## Usage

```bash
python extract_sequence_features.py --input known_primes.csv --output enriched_output.csv
```

---

## Output

The output CSV will include:

`Candidate, MotifSum, Entropy, Curvature, ...`

This output is compatible with downstream Regina Field analysis pipelines.

---

## Methodology Summary

The sequence-based approach differs from earlier versions by operating on **prime step sequences** rather than digit patterns.

Key aspects include:

- Prime gaps are converted into step sequences  
- Gaps of 6 are decomposed into `[2,4]` and `[4,2]` motifs  
- Motif frequencies are extracted over local windows  
- Entropy is computed from motif distributions  
- Curvature is derived from second-order variation in motif structure  

This produces a structured representation of primes in feature space.

---

## Important Note on Versions

This release introduces a **sequence-based feature extraction method**.

Previous versions of the toolkit used a digit-based approach.  
Those versions are preserved for transparency but are not used in current analyses.

---

## Relation to Regina Field Paper

The results presented in:

> *The Regina Field: Discrete Entropy-State Structure in Prime Distributions and Its Relationship to Prime Counting Error*

were generated using this version of the toolkit.

---

## Changelog

### v2.0-sequence-regime

- Replaced digit-based feature extraction with sequence-based motif analysis  
- Introduced step-sequence motif decomposition (including `6 → [2,4]` and `[4,2]`)  
- Updated definitions of:
  - MotifSum  
  - Entropy  
  - Curvature  
- Aligns with results presented in the Regina Field paper  

### v1.x (legacy)

- Digit-based feature extraction  
- Retained for historical reference only  

---

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  

---

## Contact

middlebrookstech@gmail.com