# Regina Field Toolkit (Legacy: Digit-Based Feature Extraction)

> ⚠️ **Legacy Documentation**
>
> This document describes the **original digit-based feature extraction system** used in early exploratory phases of the Regina Field project.
>
> It is preserved for transparency and historical reference.
>
> **It is NOT used in the current paper or validated results.**

---

## Overview

This version of the toolkit computes structural features based on **digit patterns** within candidate numbers.

Features include:

- **MotifSum** — derived from digit pattern structures  
- **Entropy** — digit frequency randomness  
- **Curvature** — proxy for deviation from baseline patterns  

These features were used in early-stage exploration of structural patterns in primes.

---

## Feature Extraction (Legacy)

### Script: `extract_features.py`

This script processes a CSV of candidate numbers and computes:

- MotifSum (digit-based)
- Entropy (digit distribution)
- Curvature (digit-based proxy)

---

## Input Format

The input CSV must contain:

`Candidate`

Each row should contain an integer.

---

## Usage

```bash
python extract_features.py --input known_primes.csv --output enriched_output.csv
```

---

## Output

The output CSV includes:

`Candidate, MotifSum, Entropy, Curvature, ...`

---

## Methodology Summary

The digit-based approach operates on the **internal representation of numbers** rather than relationships between primes.

Key characteristics:

- Features derived from digit composition  
- No use of prime gap or sequence structure  
- Local (number-level) rather than relational  

---

## Limitations

This approach was found to be insufficient for capturing the structural properties observed in later analysis.

Specifically:

- Does not reflect prime-to-prime relationships  
- Lacks sensitivity to sequence-based structure  
- Not used in final Regina Field model  

---

## Relation to Current Work

The current Regina Field framework uses a **sequence-based motif analysis**, which replaces this digit-based method.

See:

- `README.md` (current sequence-based toolkit)

---

## Historical Context

This version represents the **initial exploratory phase** of the project.

It is retained to:

- document development progression  
- provide transparency  
- allow comparison between methodologies  

---

## Status

- Maintained for reference only  
- Not used in current results  
- Not recommended for new analysis  

---

## License

Same as main project.

---

## Contact

Refer to main repository.