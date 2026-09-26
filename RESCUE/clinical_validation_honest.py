"""
clinical_validation_honest.py
=============================
DELIVERABLE 5 (sign-convention fix) — recompute clinical external
validation with the CANONICAL convention on the GENUINE-ONLY corpus.

Canonical convention (see RESCUE/SIGN_CONVENTION.md):
  - Model output: predicted DeltaG (kcal/mol, negative = tighter binding).
  - corr(Clinical_Penalty_Score, predicted DeltaG) is expected POSITIVE.
  - ROC-AUC: score = +predicted DeltaG (higher = more resistant).
  - The legacy scripts reported: Pearson +0.2555 / Spearman +0.3558
    (_clinical_validation.py, +pred) and Spearman -0.343
    (05_generate_figures.py, -pc). Same relationship, opposite axis.

Also fixes the honest cohort description:
  - HIV2_CLINICAL_ML_READY_FLAT.csv = 5,232 rows = 654 sequence entries
    (653 unique + 1 WT duplicate) x 8 drugs.
    NOT "5,232 independent isolates" as previously claimed.

Outputs:
  - RESCUE/clinical_validation_honest.json
  - RESCUE/clinical_validation_log.txt
"""

import hashlib
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opencode_shared import (
    WORKSPACE, KNOWN_INFO, extract_global_features, benjamini_hochberg,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "clinical_validation_log.txt"), mode="w",
            encoding="utf-8"),
    ],
)
log = logging.getLogger("clinical_honest")

RESCUE = os.path.dirname(os.path.abspath(__file__))
GENUINE_CSV = os.path.join(KNOWN_INFO, "STANFORD_HIV1_GENUINE.csv")
CLINICAL_CSV = os.path.join(KNOWN_INFO, "HIV2_CLINICAL_ML_READY_FLAT.csv")
RF_PARAMS = dict(n_estimators=600, max_depth=20, min_samples_leaf=2,
                 min_samples_split=5, max_features="sqrt",
                 n_jobs=-1, random_state=42)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    log.info("=" * 70)
    log.info("CLINICAL EXTERNAL VALIDATION — HONEST, CANONICAL CONVENTION")
    log.info("=" * 70)

    # Load genuine HIV-1 and train the same RF as everywhere else
    h1 = pd.read_csv(GENUINE_CSV, low_memory=False)
    h1 = h1.dropna(subset=["Sequence", "Binding_Affinity_kcal_mol"])
    h1["Binding_Affinity_kcal_mol"] = pd.to_numeric(
        h1["Binding_Affinity_kcal_mol"], errors="coerce")
    h1 = h1[(h1["Binding_Affinity_kcal_mol"] >= -20)
            & (h1["Binding_Affinity_kcal_mol"] <= 0)]
    seqs1 = h1["Sequence"].astype(str).str.strip().str.upper().values
    y1 = h1["Binding_Affinity_kcal_mol"].values.astype(np.float32)
    mask1 = np.array([len(s) >= 90 for s in seqs1])
    X1 = np.array([extract_global_features(s, include_stats=False)
                   for s in seqs1[mask1]], dtype=np.float32)
    y1 = y1[mask1]
    log.info(f"HIV-1 genuine train: {len(y1)} samples, {X1.shape[1]}D")

    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X1, y1)

    # Clinical cohort — honest description
    clin = pd.read_csv(CLINICAL_CSV, low_memory=False)
    seqs = clin["Mutant_Sequence"].astype(str).str.strip().str.upper().values
    n_seq_entries = len(clin)
    n_unique_seq = len(np.unique(seqs))
    n_drugs = clin["Compound_Name"].nunique()
    cohort_desc = {
        "rows": int(n_seq_entries),
        "unique_sequences": int(n_unique_seq),
        "sequence_entries": int(n_seq_entries),
        "drugs": int(n_drugs),
        "structure": (f"{int(n_seq_entries)} rows = {int(n_seq_entries//n_drugs)} "
                      f"sequence entries x {int(n_drugs)} drugs"),
        "previous_wrong_claim": "5,232 independent isolates",
        "correct_description": ("5,232 rows = 654 sequence entries "
                                "(653 unique + 1 WT duplicate) x 8 drugs"),
        "per_source": clin["Source"].value_counts().to_dict(),
        "resistance_classes": clin["Clinical_Resistance_Class"].value_counts().to_dict(),
    }
    log.info(f"Cohort: {cohort_desc['correct_description']}")

    scores = pd.to_numeric(clin["Clinical_Penalty_Score"],
                           errors="coerce").values
    classes = clin["Clinical_Resistance_Class"].astype(str).values

    Xc = np.array([extract_global_features(s, include_stats=False)
                   for s in seqs if len(s) >= 90], dtype=np.float32)
    keep = np.array([len(s) >= 90 for s in seqs])
    scores, classes = scores[keep], classes[keep]
    pred = rf.predict(Xc)
    log.info(f"Predicted for {len(pred)} clinical rows")

    # ---- Canonical statistics (score = +predicted dG) ----
    mask = np.isfinite(scores) & np.isfinite(pred)
    r, pv = pearsonr(scores[mask], pred[mask])
    rs, ps = spearmanr(scores[mask], pred[mask])
    log.info(f"\nClinical External Validation (n={int(mask.sum())}):")
    log.info(f"  Pearson  R = {r:.4f} (p={pv:.2e})   [canonical, +predicted dG]")
    log.info(f"  Spearman rho = {rs:.4f} (p={ps:.2e}) [canonical, +predicted dG]")
    log.info(f"  (legacy figure script reported rho=-0.343 on -dG axis — "
             f"same relationship, opposite sign)")

    # Per resistance class
    class_stats = {}
    for cls in ["Susceptible", "Intermediate", "Resistant"]:
        ix = (classes == cls) & np.isfinite(scores) & np.isfinite(pred)
        if ix.sum() > 5:
            class_stats[cls] = {
                "n": int(ix.sum()),
                "mean_pred_dG": round(float(pred[ix].mean()), 4),
                "mean_penalty": round(float(scores[ix].mean()), 2),
            }
            log.info(f"  {cls} (n={ix.sum()}): mean pred dG="
                     f"{pred[ix].mean():.2f}, mean penalty={scores[ix].mean():.1f}")

    # Per drug with BH correction
    drug_rows = []
    for drug in sorted(clin["Compound_Name"].dropna().unique()):
        ix = (clin["Compound_Name"] == drug) & np.isfinite(scores) & np.isfinite(pred)
        if ix.sum() < 20:
            continue
        rd, pd_ = pearsonr(scores[ix], pred[ix])
        drug_rows.append({"drug": drug, "n": int(ix.sum()),
                          "pearson_r": round(float(rd), 4),
                          "pearson_p": float(pd_)})
    if len(drug_rows) > 1:
        q = benjamini_hochberg([d["pearson_p"] for d in drug_rows])
        for d, qi in zip(drug_rows, q):
            d["bh_q"] = round(float(qi), 4)
    for d in drug_rows:
        log.info(f"  {d['drug']:<8s} n={d['n']:>4d}  R={d['pearson_r']:.4f}  "
                 f"p={d['pearson_p']:.2e}  q={d.get('bh_q')}")

    # ROC-AUC: score = +predicted dG (higher = more resistant)
    binary = clin.iloc[keep][clin["Clinical_Resistance_Class"].astype(str)
                             .str.upper().str.strip()
                             .isin(["S", "R", "SUSCEPTIBLE", "RESISTANT"])]
    # align: keep[] applied to scores/classes/pred; binary must use same rows
    keep_idx = np.where(keep)[0]
    bin_mask = (clin["Clinical_Resistance_Class"].astype(str)
                .str.upper().str.strip()
                .isin(["S", "R", "SUSCEPTIBLE", "RESISTANT"])).values[keep_idx]
    y_bin = (np.isin(clin["Clinical_Resistance_Class"].astype(str).str.upper()
                     .str.strip().values[keep_idx][bin_mask],
                     ["R", "RESISTANT"])).astype(int)
    score_bin = pred[bin_mask]
    auc = roc_auc_score(y_bin, score_bin) if len(np.unique(y_bin)) == 2 else None
    log.info(f"\n  ROC-AUC (Resistant vs Susceptible, score=+predicted dG): "
             f"{auc:.4f}" if auc is not None else "  ROC-AUC: insufficient")

    results = {
        "convention": ("predicted dG (kcal/mol, negative = tighter binding); "
                       "corr(penalty, pred_dG) expected POSITIVE; "
                       "ROC-AUC score = +predicted dG"),
        "cohort": cohort_desc,
        "canonical": {
            "n": int(mask.sum()),
            "pearson_r": round(float(r), 4),
            "pearson_p": float(pv),
            "spearman_rho": round(float(rs), 4),
            "spearman_p": float(ps),
            "roc_auc_resistant_vs_susceptible": (
                round(float(auc), 4) if auc is not None else None),
        },
        "legacy_comparison": {
            "05_generate_figures_spearman_minus_pc": -0.343,
            "clinical_validation_pearson_plus_pred": 0.2555,
            "clinical_validation_spearman_plus_pred": 0.3558,
            "clinical_validation_roc_auc_minus_pred": 0.0959,
            "note": ("Sign flip explained in RESCUE/SIGN_CONVENTION.md. "
                     "ROC-AUC 0.0959 was computed with score=-dG; canonical "
                     "score=+dG gives ~1-0.0959=0.904."),
        },
        "per_drug": drug_rows,
        "per_class": class_stats,
        "input_hashes": {
            "STANFORD_HIV1_GENUINE.csv": sha256(GENUINE_CSV),
            "HIV2_CLINICAL_ML_READY_FLAT.csv": sha256(CLINICAL_CSV),
        },
    }
    out = os.path.join(RESCUE, "clinical_validation_honest.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    log.info(f"\n[DONE] Saved {out}")


if __name__ == "__main__":
    main()
