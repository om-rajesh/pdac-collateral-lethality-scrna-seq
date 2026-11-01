# Single-Cell Analysis of Collateral-Lethality Dependency-Associated Transcriptional Signatures Across Classical and Basal-like States in Pancreatic Ductal Adenocarcinoma

Single-author analysis code for the manuscript of the same name. `analysis_notebook.ipynb`
computes every statistic, table and figure in the manuscript and its Supplementary Information,
from the input files listed below; no result is typed into the notebook by hand.

## What the analysis does

- **Derives** transcriptional signatures associated with dependency on *VPS4A*, *PRMT5* and
  *MAT2A* from DepMap cell lines (Chronos gene effects), inside the passenger-loss context of
  each target (*VPS4B*-low or *MTAP*-low lines).
- **Validates** the signatures against measured dependency in held-out DepMap lines (repeated,
  lineage-stratified five-fold cross-validation; DEMETER2 as an independent dataset; held-out
  PDAC lines; derivations from copy-number calls, with lineage held constant, at other cut-offs
  and in passenger-intact lines).
- **Tests** the signatures in the malignant cells of two public PDAC single-cell RNA-seq
  cohorts (Peng et al. 2019; Steele et al. 2020): within each patient, their association with
  the classical and basal-like (Moffitt et al.) state scores, using partial Spearman correlations
  with both sides residualized on sequencing depth, detected genes and mitochondrial fraction.
- **Controls** for alternative explanations: random, expression-matched,
  variability/detection-matched and covariance-matched null gene sets (1,000 each); a commonly
  essential gene set; independently defined epithelial, secretory, methionine-metabolism and
  published single-cell subtype programs; gene-overlap and dependency-based positive controls;
  equivalence tests; Benjamini–Hochberg correction.
- **Checks** the inputs: doublet detection, per-patient copy-number inference and
  high-confidence malignant cells, inferred 9p21 and 18q passenger loss, and a consensus-clustered
  NMF of malignant programs benchmarked against standard cNMF.

The notebook opens with a cell stating its scope, claims and design rationale, and every section
has a table explaining its choices. Sections 1–15 are the original analysis; sections R0–R11 were
added in revision.

## Repository contents

- `analysis_notebook.ipynb` — the complete analysis, saved with the outputs of the reported run.
- `requirements.txt` — Python dependencies.
- `LICENSE` — MIT.

Raw data and intermediate `.h5ad` files are not committed (too large); see **Data**.

## Environment

The reported run used Python 3.11.15, numpy 2.4.6, pandas 2.2.3, scanpy 1.11.5 and statsmodels
0.14.6 (printed by the Configuration cell). Other packages: scipy, scikit-learn, matplotlib,
anndata, h5py, harmonypy, infercnvpy, pybiomart, gseapy and cnmf (psutil optional, for memory
reporting).

```bash
pip install -r requirements.txt
```

## Data

By default the notebook looks for its inputs in `~/Downloads` (the `DOWNLOADS` constant in the
Configuration cell). The input-search cell finds them anywhere on the computer and copies them
there; the Preflight cell stops if a required file is missing.

**Required for sections 1–15**

| constant | file | source |
|---|---|---|
| `PENG_PROC` | `peng_processed.h5ad` | Peng et al. 2019, log-normalized and cell-type annotated, from TISCH |
| `STEELE_QC` | `steele_qc_lognorm.h5ad` | Steele et al. 2020, GEO GSE155698; QC-filtered and log-normalized, with the original counts stored for doublet detection |
| `DEPMAP_CRISPR` | `CRISPRGeneEffect.csv` | DepMap 26Q1, Chronos gene effects |
| `DEPMAP_EXPR` | `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` | DepMap 26Q1 expression |

**Required for the revision sections (R0–R11)**

| file | source |
|---|---|
| `OmicsCNGeneWGS.csv` | DepMap gene-level relative copy number (used only for deletion calls; lines matched by ModelID) |
| `Model.csv` | DepMap model annotation (lineage) |
| `D2_combined_gene_dep_scores.csv` | DEMETER2 Data v6 (RNA interference) |
| `raghavan2021_signatures.csv` | Raghavan et al. 2021 single-cell signatures, two columns: `signature,gene` |

A missing revision input skips only the analyses that need it, and the notebook says so.

**Internet access.** Ensembl gene positions (via `pybiomart`) and the Enrichr gene-set libraries
(via `gseapy`) are downloaded on the first run and cached. Section 16 queries Pharos/TCRD live;
it is kept for reference only and is not part of the manuscript.

Full dataset citations are in the manuscript's Data Sources section.

## Reproducing

1. Place the input files where `DOWNLOADS` points.
2. `Restart & Run All`.

Sections 3–6 build the preprocessed intermediates (`peng_malignant_aim2.h5ad`,
`steele_malignant_scored.h5ad`, `peng_cnmf_K10_varnorm.npz`, `peng_infercnv_v2.h5ad`) only if
they are missing, or always when `FORCE_REBUILD = True` (about 90 minutes). The saved notebook ran
with `FORCE_REBUILD = False`, loading the cached intermediates that every reported number came
from. The revision sections cache their own outputs in `~/Downloads/revision`; `REV_FORCE = True`
recomputes them (about 2–4 hours on a laptop on the first run).

Randomness is fixed (`SEED = 0`), and each stochastic step draws from its own seeded generator,
so a re-run reproduces the printed numbers.

## License

Released under the MIT License — © 2026 om-rajesh. See `LICENSE` for the full text.
