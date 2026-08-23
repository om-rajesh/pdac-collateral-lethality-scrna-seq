[README.md](https://github.com/user-attachments/files/31343006/README.md)
# Single-Cell Analysis of State-Restricted and Uniform Collateral-Lethality Dependency Signatures in Pancreatic Ductal Adenocarcinoma

Single-author analysis code accompanying the manuscript of the same name. The
notebook regenerates every intermediate, statistic, and figure from the
processed input matrices — one restart-and-run-all reproduces the paper.

## What this does

Scores KRAS-adjacent collateral-lethality **dependency signatures** (_MAT2A,
PRMT5, VPS4A_) against the classical–basal axis in two independent PDAC
scRNA-seq cohorts (Peng, Steele), using within-patient partial Spearman
correlations residualized on QC covariates, with random and expression-matched
null signatures (n = 1000), positive/negative controls, a six-part robustness
battery (V1–V6), independent cross-cohort replication, and Pharos/TCRD
druggability annotation.

## Repository contents

- `analysis.ipynb` — the complete analysis notebook.
- `requirements.txt` — Python dependencies.
- `LICENSE` — see "License" below (not yet added).

Raw data and intermediate `.h5ad` files are **not** committed (too large); see
"Data" below for provenance.

## Environment

Python 3.x. Install dependencies:

```bash
pip install -r requirements.txt
```

For an exact reproduction, pin to the versions you actually ran with
(`pip freeze > requirements.txt` in your working environment) — the shipped
`requirements.txt` is unpinned.

## Data

The notebook's Preflight cell requires **four raw input files** and stops
loudly if any are missing. By default it looks for them in `~/Downloads`
(the `DOWNLOADS` constant in the Configuration cell — change it to point
wherever you keep the files, e.g. a repo-relative `./data`).

| Constant        | Source |
|-----------------|--------|
| `PENG_PROC`     | Peng et al. PDAC scRNA-seq, log-normalized as distributed by **TISCH** (tisch.comp-genomics.org). |
| `STEELE_QC`     | Steele et al. PDAC scRNA-seq, **GEO GSE155698**. |
| `DEPMAP_CRISPR` | **DepMap** (Broad Institute) CRISPR gene-effect (Chronos) matrix — depmap.org/portal. Record the release version you used. |
| `DEPMAP_EXPR`   | **DepMap** expression matrix, same release. |

Two resources are queried **live over the internet** at run time (no local
file): **Pharos/TCRD** (`https://pharos-api.ncats.io/graphql`) for druggability
and **Ensembl** via `pybiomart` for gene mapping. An internet connection is
therefore required for a full run.

Full dataset citations are in the manuscript's *Data Sources* section.

## Reproducing

The Configuration cell sets `SEED = 0`, `N_NULL = 1000`, and
`FORCE_REBUILD = True`. With `FORCE_REBUILD = True`, every intermediate is
rebuilt from the raw inputs even if a cached `.h5ad` exists — a clean, fully
regenerated run:

1. Place the four raw inputs where `DOWNLOADS` points.
2. `Restart & Run All`.

Expect roughly ~90 minutes end to end (Harmony integration, inferCNV, cNMF, the
null distributions, and the live Pharos query dominate the time). Intermediates
(`PENG_MAL`, `STEELE_MAL`, `PENG_CNMF`, `PENG_CNV`) are written back to
`DOWNLOADS`; set `FORCE_REBUILD = False` to reuse them on subsequent runs.

## License

No license file is included yet. Research code is commonly released under the
**MIT License**; add a `LICENSE` file before making the repo public so others
know the terms of reuse.
