# Data

Three curated datasets are required. All are released here so that every reported number can
be regenerated. Their SHA-256 checksums are recorded inside every result manifest, so any
modification is detectable.

| File | Rows | SHA-256 |
|---|---|---|
| `STANFORD_HIV1_GENUINE.csv` | 5,683 | `692e3d2b3ffcd43bd186458a97a9b572c7b6c9f0287ff9f51f3aeb75bcca9ab0` |
| `HIV2_PROTEASE_ML_READY_DATASET.csv` | 320 | `7f34b7a2b467e5eadd2e02aa3ee141a749e93aa4cd99145858b9dafded9b093d` |
| `HIV2_CLINICAL_ML_READY_FLAT.csv` | 5,232 | `ca49109932254c0b33fc675c59b3749461322d77b6970bdf8c370060f5c2b637` |

> **Note on the clinical file.** The original working file carries ~3,600 pre-computed
> feature columns (≈200 MB). The pipeline never reads those columns — all features are
> recomputed deterministically from the sequence by `opencode_shared/utils.py`. The
> released file therefore contains only the seven source columns the pipeline reads
> (`SequenceID, Source, Mutations, Mutant_Sequence, Compound_Name,
> Clinical_Penalty_Score, Clinical_Resistance_Class`), keeping the repository within
> GitHub's file-size limit. The full file's SHA-256 (`2ccbf59e…ced35d0`) is what the
> result manifests record; re-deriving the feature columns and re-hashing reproduces it.

## Provenance

### `STANFORD_HIV1_GENUINE.csv` — HIV-1 training corpus
Curated HIV-1 protease phenotypic assays from the Stanford HIV Drug Resistance Database.
Phenotypic values (IC50, nM) are converted to binding free energy by
`ΔG = RT·ln(IC50 × 10⁻⁹)` with `RT = 0.593 kcal/mol` at 298 K.

An additional 1,500 rows in the original source corpus were **not** experimental measurements
but were produced by a banded-noise generator. They were removed before any modelling; no
reported result uses them. They are retained in the project history but are excluded here
because they are not data.

### `HIV2_PROTEASE_ML_READY_DATASET.csv` — HIV-2 cross-species panel
HIV-2 protease–phenotype assays curated from ChEMBL targets **CHEMBL380** and **CHEMBL5074**
(320 rows; 137 pass the legacy zero-shot filter of assay type ∈ {IC50, KI, KD} with
`ΔG ∈ [−20, 0] kcal/mol`). Forty rows per drug are available for the four common protease
inhibitors (atazanavir, darunavir, lopinavir, saquinavir), giving the 160-row transfer
fine-tune panel.

### `HIV2_CLINICAL_ML_READY_FLAT.csv` — HIV-2 clinical panel
A master clinical genotype–phenotype cohort for HIV-2 (5,232 rows = 654 sequence entries ×
8 protease inhibitors), assembled from the Stanford HIVDB and from mined literature. The
clinical label is the HIVDB penalty score, a discretized interpretation accompanied by a
Susceptible / Intermediate / Resistant classification. The manuscript's headline clinical
subset is the four-drug common-PI subset (2,616 rows).

**No clinical label is ever used in training.** The clinical panel is used exclusively for
held-out evaluation.

## Notes on excluded material

* Synthetic (generated) HIV-1 rows were excluded and are not distributed.
* Structural data are not stored here: all structural figures are rendered on demand from
  experimental RCSB PDB entries (3S45, 4LL3, 2AQU, 3OXC, 1MUI), which remain the property
  of the PDB and are fetched under their own terms.
