# Cross-species transfer learning for HIV-2 protease-inhibitor resistance prediction

Code, curated data, and verified result manifests for the manuscript:

> **Cross-species transfer learning from HIV-1 to HIV-2 for protease inhibitor resistance
> prediction: biophysical features and external clinical validation**
> Nunna Jaswanth Sriram — Department of Artificial Intelligence & Engineering,
> Amrita Vishwa Vidyapeetham, Bengaluru, India.

---

## Manuscript

- **PDF (read this first):** [`submission/HIV2_cross_species_transfer_manuscript.pdf`](submission/HIV2_cross_species_transfer_manuscript.pdf)
- **Source (HTML):** [`RESCUE/manuscript_corrected.html`](RESCUE/manuscript_corrected.html) — open in a browser and print to PDF to regenerate

## Overview

HIV-2 is intrinsically resistant to several protease inhibitors, yet resistance-interpretation
tools (including the Stanford HIVDB and the rule-based HIV-2EU) are calibrated on HIV-1, and no
machine-learning predictor of HIV-2 protease-inhibitor resistance has been published.

This repository contains the first machine-learning framework for HIV-2 protease-inhibitor
resistance prediction. The model is pre-trained on 5,683 curated HIV-1 phenotypic assays, applied
zero-shot to HIV-2, and then fine-tuned on a 160-row HIV-2 ChEMBL panel. It is evaluated against
**measured** HIV-2 binding affinities and against 2,616 never-seen HIV-2 clinical
genotype–phenotype rows. No clinical label ever enters training.

## Headline results

| Result | Value |
|---|---|
| Within-HIV-1 grouped CV (95% identity) | R = 0.7387 ± 0.0472 |
| Best zero-shot HIV-1 → HIV-2 transfer (ridge meta-ensemble) | R = 0.5818 |
| Transfer fine-tune, out-of-fold benchmark (n=160) | **R = 0.7018** (95% CI 0.6246–0.7887) |
| vs. Stanford-derived rule, **measured phenotypes** | **0.7018 vs 0.4724**, ΔR = +0.229, P(ΔR≤0) = 0.001 |
| Leave-one-sequence-out jackknife (19 sequences) | model ahead in **100% of folds** (ΔR range +0.160 to +0.264) |
| Model on rows the rule cannot rank (n=116) | **R = 0.585**, 95% CI 0.296–0.788, P < 0.001 |
| Clinical benchmark label that is zero (no rule matched) | **93.1%** of 5,232 rows |
| Mutations in measured panel covered by Stanford-derived rules | **3 of 37** (8.1%) |
| Mutations covered by the **official HIV-2EU v4** rules | **5 of 37** (13.5%) |
| Measured panel unscored by official HIV-2EU v4 | **83.1%** (50% because no rule exists for that drug) |
| Clinical drugs with no HIV-2EU protease rules | **6 of 8** |
| External clinical validation (2,616 never-seen rows) | pooled R = 0.4454, ROC-AUC = 0.927 |
| Known drug-resistance-mutation concordance | up to r = 0.79 (LPV/r) |

The head-to-head on measured binding affinities is the primary evidence that the transferred
representation carries information beyond what the existing rule-based interpretation encodes.
That matters because rule-based interpretation is **sparse** — and this holds for the official
published HIV-2EU v4 rules, not only our Stanford-derived baseline: HIV-2EU defines no protease
rules for 6 of 8 clinical drugs, leaves 83.1% of the measured panel unscored, and recognises 5 of
37 observed mutations, while the clinical benchmark itself scores zero for 93.1% of isolates.
On the measured rows the rule leaves unranked, the model retains a significant correlation.

## Repository structure

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── data/                       Curated input datasets (see data/README.md)
├── opencode_shared/            Feature-extraction and shared utilities
└── RESCUE/
    ├── BENCHMARK/              Benchmark, transfer-learning and validation pipeline
    │   ├── benchmark_config.py            Central configuration (paths, drugs, splits)
    │   ├── transfer_learning*.py          Zero-shot / transfer fine-tuning models
    │   ├── clinical_validation_final.py   Locked external clinical validation
    │   ├── capture_per_row_predictions.py Per-row predictions (integrity-gated)
    │   ├── head_to_head_paired.py         Model vs rule on measured phenotypes
    │   ├── loso_head_to_head.py           Leave-one-sequence-out robustness
    │   ├── analysis_rule_coverage.py      Rule coverage / blindness (Stanford-derived)
    │   ├── analysis_hiv2eu_coverage.py    Coverage of the official HIV-2EU v4 rules
    │   └── results/                       Result manifests (JSON) + per-row predictions
    ├── journal_render/
    │   ├── render_*.py                    Biological + result figure renderers
    │   └── figures/                       Manuscript figures (PNG, 600 dpi)
    ├── manuscript_corrected.html          Full manuscript (open in a browser; print to PDF)
    ├── metrics_manifest.json              Reproducible evaluation manifest
    └── legacy_rerun_results.json          Honest re-run of previously reported numbers
```

## Installation

Python 3.12 is required. The exact environment used for every reported number:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` pins the versions that reproduce the results exactly
(`numpy==1.26.4`, `pandas==2.2.1`, `scipy==1.12.0`, `scikit-learn==1.4.1.post1`,
`xgboost==2.0.3`, `biopython==1.83`, `torch==2.2.1`).

> **Note.** `opencode_shared/utils.py` imports `torch` at module load (for RNG seeding).
> If `torch` is unavailable the feature-extraction import silently fails and the pipeline
> degrades to all-zero features. Install `torch` (CPU build is sufficient) before running.

## Reproducing the results

All paths resolve relative to the repository root, so the checkout can live anywhere.

```bash
# 1. Locked external clinical validation (reproduces the 0.7018 benchmark anchor)
python RESCUE/BENCHMARK/clinical_validation_final.py

# 2. Per-row predictions, integrity-gated against the locked manifest
python RESCUE/BENCHMARK/capture_per_row_predictions.py

# 3. Head-to-head against rule-based interpretation on measured phenotypes (Table 8, Fig 7)
python RESCUE/BENCHMARK/head_to_head_paired.py

# 4. Leave-one-sequence-out robustness of that comparison (19 unique sequences)
python RESCUE/BENCHMARK/loso_head_to_head.py

# 5. Rule coverage / blindness analysis (Table 9, Fig 8)
python RESCUE/BENCHMARK/analysis_rule_coverage.py

# 6. Coverage of the official published HIV-2EU v4 rules (Table 9)
python RESCUE/BENCHMARK/analysis_hiv2eu_coverage.py

# 7. Regenerate every manuscript figure
python RESCUE/journal_render/render_biological_figures.py
python RESCUE/journal_render/render_result_figures.py
python RESCUE/journal_render/render_head_to_head_figure.py
python RESCUE/journal_render/render_study_design.py
python RESCUE/journal_render/render_rule_coverage_figure.py

# 8. Re-run all previously reported numbers on the curated corpus
python RESCUE/rerun_legacy_numbers.py
```

Each evaluation writes a JSON manifest to `RESCUE/BENCHMARK/results/`. The scripts embed
integrity gates: if a re-derived metric disagrees with the locked manifest beyond tolerance,
the run refuses to write output.

## Data

| File | Rows | Description | Source |
|---|---|---|---|
| `data/STANFORD_HIV1_GENUINE.csv` | 5,683 | Curated HIV-1 phenotypic assays (IC50 → ΔG) | Stanford HIVDB |
| `data/HIV2_PROTEASE_ML_READY_DATASET.csv` | 320 | HIV-2 protease-phenotype assays | ChEMBL (CHEMBL380 / CHEMBL5074) |
| `data/HIV2_CLINICAL_ML_READY_FLAT.csv` | 5,232 | HIV-2 clinical genotype–phenotype panel | Stanford HIVDB / mined literature |

All input files are hashed (SHA-256) inside every result manifest, so any modification of the
inputs is detectable. See `data/README.md` for full provenance.

## Figures

| Figure | Content |
|---|---|
| 1 | Study design |
| 2 | HIV-1 (HXB2) vs HIV-2 (ROD) protease sequence alignment |
| 3 | Chemical structures of the four protease inhibitors |
| 4 | HIV-2 pocket-19 architecture (3S45) |
| 5 | HIV-1 / HIV-2 homodimer Cα superimposition (RMSD 1.09 Å) |
| 6 | Modelled inhibitor poses in the HIV-2 pocket |
| 7 | Model vs rule-based interpretation on measured phenotypes |
| 8 | Coverage of rule-based interpretation (Stanford-derived and official HIV-2EU v4) |
| 9 | Cross-species transfer model performance |

Structural figures are rendered from experimental RCSB PDB entries only (3S45, 4LL3, 2AQU,
3OXC, 1MUI). No predicted or AI-generated structures are used.

## Citation

If you use this work, please cite the manuscript (see `CITATION.cff`).

## License

Released under the MIT License — see `LICENSE`.

## Contact

Nunna Jaswanth Sriram — `bl.sc.u4aie24029@bl.students.amrita.edu`
