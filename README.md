# Basal-like cells in heterogeneous pancreatic tumors preferentially express genes linked to *VPS4A* dependency

Analysis code for the manuscript of the same title. `analysis_notebook.ipynb` computes every number, table and figure in the manuscript and its Supplementary Information, and is saved with the outputs of the reported run.

## What it does

Transcriptional signatures associated with dependency on *VPS4A*, *PRMT5* and *MAT2A* are derived from DepMap cell lines and validated against measured dependency in held-out lines. Within each patient, each signature is then correlated with the classical and basal-like state scores in the malignant cells of three public single-cell RNA-seq cohorts of pancreatic cancer: Peng et al. 2019 and Steele et al. 2020 (discovery) and Werba et al. 2023 (replication). Each association is compared with random and matched gene-set nulls, positive control dependencies and signatures built identically for dependencies without a subtype association.

## Contents

| file | |
|---|---|
| `analysis_notebook.ipynb` | the complete analysis, with outputs |
| `requirements.txt` | Python dependencies |
| `package_versions.txt` | every package version of the reported run |
| `raghavan2021_signatures.csv` | Raghavan et al. 2021 gene lists (input) |
| `LICENSE` | MIT |

## Setup

Python 3.11, then `pip install -r requirements.txt`.

## Data

The notebook reads its inputs from `~/Downloads` (set in section 1) and stops if a required file is missing.

| file | source |
|---|---|
| `peng_processed.h5ad` | Peng et al. 2019, from TISCH (log-normalized, annotated) |
| `steele_qc_lognorm.h5ad` | Steele et al. 2020, GEO GSE155698, after quality control (on Zenodo) |
| `GSE205013_RAW.tar`, `GSE205013_series_matrix.txt.gz` | Werba et al. 2023, GEO GSE205013 |
| `CRISPRGeneEffect.csv`, `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`, `Model.csv` | DepMap Public 26Q1 |
| `OmicsCNGeneWGS.csv` | DepMap Public 26Q3 |
| `D2_combined_gene_dep_scores.csv` | DEMETER2 Data v6 |
| `raghavan2021_signatures.csv` | this repository |

The processed single-cell objects that every reported number came from are deposited at [Zenodo DOI]. Place them in `~/Downloads` too; without them, sections 6–10 rebuild them, and cluster-based compartment calls can shift with package versions.

## Running

*Restart & Run All.* A first run takes several hours; with `RECOMPUTE = False` (section 21) a re-run reads the cached steps and takes minutes. `FORCE_REBUILD = True` (section 1) rebuilds the single-cell objects from the raw inputs (about 90 min more). Results, tables and figures are written to `~/Downloads/pdac_results`. The seed is 0, so a re-run reproduces the printed numbers.

## License

MIT. See `LICENSE`.
