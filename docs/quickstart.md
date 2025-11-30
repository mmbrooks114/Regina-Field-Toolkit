# Quickstart

```python
import reginafield as rf

# Load data
master = rf.datasets.load_enriched_master("ReginaField_EnrichedMaster.csv")
royals = rf.datasets.load_royal_primes("Royal_Primes.csv")

# Add features
df = rf.royal.attach_royal_status(master, royals)
df = rf.features.add_radius(df)
df = rf.features.add_entropy_tiers(df)
df = rf.zones.classify_attractor_zones(df)

# Visualize
rf.visualize.plot_pca_field(df, color_by="Zone")
```

## CLI Example

```bash
reginafield-cli plot-royal ReginaField_EnrichedMaster.csv Royal_Primes.csv
```
