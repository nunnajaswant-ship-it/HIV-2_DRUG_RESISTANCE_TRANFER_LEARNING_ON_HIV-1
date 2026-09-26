"""
rerun_legacy_numbers.py
========================
DELIVERABLE — HONEST DISPOSITION OF LEGACY MANUSCRIPT NUMBERS (§9 CATALOG).

Dispositions the legacy numbers computed on the POISONED corpus by
re-running the same model families on the GENUINE-ONLY corpus
(STANFORD_HIV1_GENUINE.csv, 5,683 rows; 1,500 synthetic rows quarantined).

Protocol (identical to retrain_honest.py — the honest covenant):
  - Train ONLY on genuine rows. Grouped CV at 95% greedy identity
    (cluster IDs loaded from RESCUE/cluster_splits_genuine/, seed 42).
  - Feature space: 2,079-D biochem (Z3+K10+VHSE8 per residue x 99,
    include_stats=False) — the documented manuscript space.
  - HIV-2 test: the honest n=137 assay rows (IC50/KI/KD, dG in [-20,0],
    len>=90), same filter as retrain_honest.py.
  - Wildtype references from opencode_shared/config.py (HIV2_WT = ROD,
    A92 verified vs NCBI M15390.1).

Items covered:
  1. XGBoost zero-shot cross-species        (legacy xgb 0.449 / stacked 0.4893)
  2. Simple + Ridge meta-ensembles          (legacy simple 0.5017 / ridge 0.4853)
  3. Per-drug zero-shot (RF/XGB/ensembles)  (legacy per-drug n=40, e.g. AMP)
  4. Pocket-restricted RF/XGB zero-shot     (legacy Tier2 pocket R=0.3909)
  5. Exp3 wildtype->HIV-1 substitution      (legacy deltadG table, e.g. K7Q 1.4871)
  6. Within-HIV-1 grouped CV (RF+XGB)       (legacy ceiling R=0.759, ungrouped)

NOT re-run here (documented separately):
  - ProtBERT (BERT_1..1024 cols are all-NULL in the genuine corpus;
    HIV2_PROTEASE_ML_READY_DATASET.csv has NO BERT cols; on-disk BERT CSVs
    are PDB-ID-keyed, not corpus-keyed; re-extraction needs GPU/Colab +
    model download -> D1-class infeasibility, documented, not rerun).
  - GNN / DBPT+ (torch CPU-only, days of GPU compute -> D1, already
    documented in metrics_reruns.json).
  - Epistasis matrix -> separate honest rerun (rerun_epistasis_honest.py).

Outputs:
  - RESCUE/legacy_rerun_results.json  (honest values + legacy comparison)
  - RESCUE/legacy_rerun_log.txt
"""

import hashlib
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opencode_shared import (
    WORKSPACE, KNOWN_INFO, extract_global_features, compute_ic50_to_delta_g,
)
from opencode_shared.config import HIV1_WT, HIV2_WT, POCKET_1INDEX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "legacy_rerun_log.txt"), mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("rerun_legacy_numbers")

RESCUE = os.path.dirname(os.path.abspath(__file__))
SPLIT_DIR = os.path.join(RESCUE, "cluster_splits_genuine")

GENUINE_CSV = os.path.join(KNOWN_INFO, "STANFORD_HIV1_GENUINE.csv")
SYNTH_CSV = os.path.join(KNOWN_INFO, "STANFORD_HIV1_SYNTHETIC_QUARANTINED.csv")
H2_ASSAY_CSV = os.path.join(KNOWN_INFO, "HIV2_PROTEASE_ML_READY_DATASET.csv")

RF_PARAMS = dict(n_estimators=600, max_depth=20, min_samples_leaf=2,
                 min_samples_split=5, max_features="sqrt",
                 n_jobs=-1, random_state=42)
# XGB config mirroring legacy run_stacked_ensemble.py / Tier5 stacked runs
XGB_PARAMS = dict(n_estimators=800, max_depth=6, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.1, reg_lambda=2.0,
                  n_jobs=-1, random_state=42)
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
SEED = 42
CLUSTER_THRESHOLD = 0.95
N_FOLDS = 5

# Legacy values from Tier5_comprehensive_results.json / per_drug / upgrade
LEGACY = {
    "xgb_zero_shot": 0.449,
    "ensemble_zero_shot": 0.458,
    "stacked_xgb": 0.4893,
    "stacked_simple": 0.5017,
    "stacked_ridge": 0.4853,
    "tier1_xgb_zero_shot": 0.4006,
    "pocket_rf_zero_shot": 0.3909,
    "within_species_ungrouped": 0.759,
    "exp3": {"K7Q": 1.4871},  # top legacy mutagenesis value
    "per_drug_n": 40,
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 3:
        return None
    r, pv = pearsonr(y, p)
    rho, _ = spearmanr(y, p)
    return {
        "n": int(len(y)),
        "pearson_r": round(float(r), 4),
        "pearson_p": float(pv),
        "spearman_rho": round(float(rho), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 4),
        "mae": round(float(mean_absolute_error(y, p)), 4),
        "r2": round(float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)), 4),
    }


def load_genuine_hiv1():
    df = pd.read_csv(GENUINE_CSV, low_memory=False)
    df = df.dropna(subset=["Sequence", "Binding_Affinity_kcal_mol"])
    df["Binding_Affinity_kcal_mol"] = pd.to_numeric(
        df["Binding_Affinity_kcal_mol"], errors="coerce")
    df = df[(df["Binding_Affinity_kcal_mol"] >= -20)
            & (df["Binding_Affinity_kcal_mol"] <= 0)]
    seqs = df["Sequence"].astype(str).str.strip().str.upper().values
    y = df["Binding_Affinity_kcal_mol"].values.astype(np.float32)
    keep = np.array([len(s) >= 90 for s in seqs])
    X = np.array([extract_global_features(s, include_stats=False)
                  for s in seqs[keep]], dtype=np.float32)
    return X, y[keep], df.loc[keep].copy()


def load_hiv2_assays():
    df = pd.read_csv(H2_ASSAY_CSV, low_memory=False)
    valid = df["Assay_Type"].astype(str).str.upper().str.strip().isin(
        ["IC50", "KI", "KD"])
    df = df[valid].copy()
    df["_dg"] = df["Standard_Value"].apply(
        lambda x: compute_ic50_to_delta_g(float(x)) if float(x) > 0
        else float("nan"))
    df = df.dropna(subset=["_dg"])
    df = df[(df["_dg"] >= -20) & (df["_dg"] <= 0)]
    seqs = df["Mutant_Sequence"].astype(str).str.strip().str.upper().values
    y = df["_dg"].values.astype(np.float32)
    drugs = df["Compound_Name"].astype(str).values
    keep = np.array([len(s) >= 90 for s in seqs])
    X = np.array([extract_global_features(s, include_stats=False)
                  for s in seqs[keep]], dtype=np.float32)
    return X, y[keep], drugs[keep], df.loc[keep].copy()


def pocket_features(X):
    """Subset 2079-D global features to the 16 pocket residues (336-D)."""
    cols = []
    for p in POCKET_1INDEX:
        cols += list(range((p - 1) * 21, (p - 1) * 21 + 21))
    return X[:, cols]


def make_rf(seed=SEED):
    return RandomForestRegressor(**{**RF_PARAMS, "random_state": seed})


def make_xgb(seed=SEED):
    from xgboost import XGBRegressor
    return XGBRegressor(**{**XGB_PARAMS, "random_state": seed})


def meta_features_cv(X_full, y_full, n_folds=N_FOLDS, random_state=SEED):
    """5-fold CV meta-features [RF_pred, XGB_pred, RF_var, XGB_var] —
    mirrors legacy run_stacked_ensemble.compute_meta_features_cv."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    B = X_full.shape[0]
    meta = np.zeros((B, 4), dtype=np.float32)
    for fold, (tr, va) in enumerate(kf.split(X_full)):
        rf = make_rf(seed=random_state + fold)
        rf.fit(X_full[tr], y_full[tr])
        p_rf = rf.predict(X_full[va])
        meta[va, 0] = p_rf
        tree_preds = np.array([t.predict(X_full[va]) for t in rf.estimators_])
        meta[va, 2] = tree_preds.var(axis=0)
        xgb = make_xgb(seed=random_state + fold)
        xgb.fit(X_full[tr], y_full[tr])
        p_xgb = xgb.predict(X_full[va])
        meta[va, 1] = p_xgb
        meta[va, 3] = np.abs(p_rf - p_xgb) + 0.01
    return meta


def meta_features_test(X_test, y_full, X_full, random_state=SEED):
    """Full-data models -> test meta-features (legacy compute_meta_features_test)."""
    B = X_test.shape[0]
    meta = np.zeros((B, 4), dtype=np.float32)
    rf = make_rf(seed=random_state)
    rf.fit(X_full, y_full)
    p_rf = rf.predict(X_test)
    meta[:, 0] = p_rf
    tree_preds = np.array([t.predict(X_test) for t in rf.estimators_])
    meta[:, 2] = tree_preds.var(axis=0)
    xgb = make_xgb(seed=random_state)
    xgb.fit(X_full, y_full)
    p_xgb = xgb.predict(X_test)
    meta[:, 1] = p_xgb
    meta[:, 3] = np.abs(p_rf - p_xgb) + 0.01
    return meta


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("HONEST DISPOSITION OF LEGACY MANUSCRIPT NUMBERS (§9)")
    log.info("Genuine-only corpus, grouped CV, seed 42, 2079-D biochem")
    log.info("=" * 70)

    results = {
        "pipeline": "RESCUE/rerun_legacy_numbers.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_hashes": {
            "STANFORD_HIV1_GENUINE.csv": sha256(GENUINE_CSV),
            "STANFORD_HIV1_SYNTHETIC_QUARANTINED.csv": sha256(SYNTH_CSV),
            "HIV2_PROTEASE_ML_READY_DATASET.csv": sha256(H2_ASSAY_CSV),
        },
        "seed": SEED,
        "feature_space": "2079-D biochem (Z3+K10+VHSE8 x99, include_stats=False)",
        "rf_params": RF_PARAMS,
        "xgb_params": XGB_PARAMS,
        "wildtype_references": "opencode_shared/config.py (HIV2_WT ROD A92, "
                               "HIV1_WT HXB2)",
    }

    log.info("\n[1] Load genuine HIV-1 corpus + honest HIV-2 test set")
    X1, y1, df1 = load_genuine_hiv1()
    X2, y2, drugs2, df2 = load_hiv2_assays()
    log.info(f"  HIV-1 train: {len(y1)} rows  |  HIV-2 test: {len(y2)} rows")
    n_synth = sum(1 for _ in open(SYNTH_CSV, encoding="utf-8")) - 1
    results["corpus"] = {
        "genuine_rows_trained": int(len(df1)),
        "synthetic_rows_excluded": int(n_synth),
        "hiv2_test_rows": int(len(y2)),
    }

    # Cluster IDs (same grouping as retrain_honest.py)
    cluster_ids = np.load(os.path.join(SPLIT_DIR, "cluster_ids_genuine.npy"))
    log.info(f"  Cluster IDs loaded: {len(cluster_ids)} (95% identity grouping)")

    # ---------------------------------------------------------------
    log.info("\n[2] Within-HIV-1 grouped CV (RF and XGB) — honest ceilings")
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_rows_rf, fold_rows_xgb = [], []
    for fold, (tr, te) in enumerate(gkf.split(X1, y1, groups=cluster_ids)):
        rf = make_rf()
        rf.fit(X1[tr], y1[tr])
        m_rf = metrics(y1[te], rf.predict(X1[te]))
        fold_rows_rf.append(m_rf["pearson_r"])
        xgb = make_xgb()
        xgb.fit(X1[tr], y1[tr])
        m_xgb = metrics(y1[te], xgb.predict(X1[te]))
        fold_rows_xgb.append(m_xgb["pearson_r"])
        log.info(f"  Fold {fold+1}: RF R={m_rf['pearson_r']:.4f}  "
                 f"XGB R={m_xgb['pearson_r']:.4f}")
    results["within_species_grouped_cv"] = {
        "rf": {"mean_pearson_r": round(float(np.mean(fold_rows_rf)), 4),
               "std_pearson_r": round(float(np.std(fold_rows_rf)), 4),
               "per_fold": [round(float(r), 4) for r in fold_rows_rf]},
        "xgb": {"mean_pearson_r": round(float(np.mean(fold_rows_xgb)), 4),
                "std_pearson_r": round(float(np.std(fold_rows_xgb)), 4),
                "per_fold": [round(float(r), 4) for r in fold_rows_xgb]},
        "note": ("Grouped CV at 95% identity, seed 42. Legacy manuscript "
                 "ceiling 0.759+-0.017 was UNGROUPED KFold(5) on the poisoned "
                 "7,183-row corpus."),
    }

    # ---------------------------------------------------------------
    log.info("\n[3] Zero-shot cross-species: RF, XGB, simple ensemble, "
             "Ridge meta-ensemble (genuine-only train)")
    rf = make_rf()
    rf.fit(X1, y1)
    p_rf = rf.predict(X2)
    m_rf = metrics(y2, p_rf)

    xgb = make_xgb()
    xgb.fit(X1, y1)
    p_xgb = xgb.predict(X2)
    m_xgb = metrics(y2, p_xgb)

    p_simple = (p_rf + p_xgb) / 2.0
    m_simple = metrics(y2, p_simple)

    meta_tr = meta_features_cv(X1, y1)
    meta_te = meta_features_test(X2, y1, X1)
    scaler = StandardScaler()
    meta_tr_sc = scaler.fit_transform(meta_tr)
    ridge = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True,
                    scoring="neg_mean_squared_error")
    ridge.fit(meta_tr_sc, y1)
    p_ridge = ridge.predict(scaler.transform(meta_te))
    m_ridge = metrics(y2, p_ridge)

    log.info(f"  RF    zero-shot: R={m_rf['pearson_r']:.4f}  "
             f"RMSE={m_rf['rmse']:.4f}  n={m_rf['n']}")
    log.info(f"  XGB   zero-shot: R={m_xgb['pearson_r']:.4f}  "
             f"RMSE={m_xgb['rmse']:.4f}  n={m_xgb['n']}")
    log.info(f"  Simple ensemble : R={m_simple['pearson_r']:.4f}  "
             f"RMSE={m_simple['rmse']:.4f}")
    log.info(f"  Ridge meta-ens  : R={m_ridge['pearson_r']:.4f}  "
             f"RMSE={m_ridge['rmse']:.4f}  alpha={ridge.alpha_:.4f}")

    within_rf = results["within_species_grouped_cv"]["rf"]["mean_pearson_r"]
    within_xgb = results["within_species_grouped_cv"]["xgb"]["mean_pearson_r"]
    results["zero_shot_cross_species"] = {
        "rf": {**m_rf, "transfer_ratio_vs_within": round(m_rf["pearson_r"] / within_rf, 3)},
        "xgb": {**m_xgb, "transfer_ratio_vs_within": round(m_xgb["pearson_r"] / within_xgb, 3)},
        "simple_ensemble": m_simple,
        "ridge_meta_ensemble": {**m_ridge, "ridge_alpha": float(ridge.alpha_)},
        "legacy_values": {
            "xgb_cross_species": LEGACY["xgb_zero_shot"],
            "ensemble_cross_species": LEGACY["ensemble_zero_shot"],
            "stacked_xgb": LEGACY["stacked_xgb"],
            "stacked_simple": LEGACY["stacked_simple"],
            "stacked_ridge": LEGACY["stacked_ridge"],
            "note": ("Legacy ensemble numbers were computed on the poisoned "
                     "7,183-row corpus; legacy stacked used biochem+BERT "
                     "(3103-D). Honest runs use genuine-only 2,079-D biochem "
                     "(BERT unavailable: GENUINE BERT cols are all-null)."),
        },
    }

    # ---------------------------------------------------------------
    log.info("\n[4] Per-drug zero-shot (RF/XGB/simple/ridge) on honest "
             "n=137 test set")
    per_drug = {}
    for drug in sorted(set(drugs2)):
        mask = drugs2 == drug
        if mask.sum() < 5:
            per_drug[drug] = {"n": int(mask.sum()), "note": "n<5, R not reported"}
            continue
        y_d = y2[mask]
        per_drug[drug] = {
            "n": int(mask.sum()),
            "rf": metrics(y_d, p_rf[mask]),
            "xgb": metrics(y_d, p_xgb[mask]),
            "simple_ensemble": metrics(y_d, p_simple[mask]),
            "ridge_meta_ensemble": metrics(y_d, p_ridge[mask]),
        }
        log.info(f"  {drug:<12s} n={mask.sum():>3d}  RF={per_drug[drug]['rf']['pearson_r']:.4f}"
                 f"  XGB={per_drug[drug]['xgb']['pearson_r']:.4f}"
                 f"  simple={per_drug[drug]['simple_ensemble']['pearson_r']:.4f}"
                 f"  ridge={per_drug[drug]['ridge_meta_ensemble']['pearson_r']:.4f}")
    results["per_drug_zero_shot"] = {
        "per_drug": per_drug,
        "legacy_values": {
            "per_drug_n": LEGACY["per_drug_n"],
            "source": "Tier5_per_drug_results.json / per_drug_results.csv "
                      "(poisoned 7,183-row train, n=40 per drug on 321-row test)",
            "example_AMP_legacy": {"rf": 0.4621, "xgb": 0.4462,
                                   "simple": 0.4868, "ridge": 0.4996},
        },
    }

    # ---------------------------------------------------------------
    log.info("\n[5] Pocket-restricted zero-shot (RF/XGB) — 336-D pocket features")
    X1p, X2p = pocket_features(X1), pocket_features(X2)
    rfp = make_rf()
    rfp.fit(X1p, y1)
    m_pocket_rf = metrics(y2, rfp.predict(X2p))
    xgbp = make_xgb()
    xgbp.fit(X1p, y1)
    m_pocket_xgb = metrics(y2, xgbp.predict(X2p))
    log.info(f"  Pocket RF : R={m_pocket_rf['pearson_r']:.4f}  "
             f"RMSE={m_pocket_rf['rmse']:.4f}")
    log.info(f"  Pocket XGB: R={m_pocket_xgb['pearson_r']:.4f}  "
             f"RMSE={m_pocket_xgb['rmse']:.4f}")
    results["pocket_zero_shot"] = {
        "feature_dim": int(X1p.shape[1]),
        "pocket_residues": POCKET_1INDEX,
        "rf": m_pocket_rf,
        "xgb": m_pocket_xgb,
        "legacy": {"tier2_pocket_rf_zero_shot": LEGACY["pocket_rf_zero_shot"],
                   "source": ("Tier2_pocket_rf_xgboost_results.json, poisoned "
                              "corpus, 321-row test")},
    }

    # ---------------------------------------------------------------
    log.info("\n[6] Exp3 wildtype->HIV-1 in-silico mutagenesis (deltadG)")
    wt = np.array(list(HIV2_WT))
    h1 = np.array(list(HIV1_WT))
    if len(wt) != 99 or len(h1) != 99:
        raise RuntimeError(f"WT length mismatch: {len(wt)} / {len(h1)}")
    rf_mut = make_rf()
    rf_mut.fit(X1, y1)
    wt_feat = extract_global_features(HIV2_WT, include_stats=False).reshape(1, -1)
    wt_pred = float(rf_mut.predict(wt_feat)[0])
    divergent = [(i + 1, wt[i], h1[i]) for i in range(99) if wt[i] != h1[i]]
    rows = []
    for pos, aa_wt, aa_h1 in divergent:
        mut = list(HIV2_WT)
        mut[pos - 1] = aa_h1
        mut_feat = extract_global_features("".join(mut), include_stats=False).reshape(1, -1)
        mut_pred = float(rf_mut.predict(mut_feat)[0])
        rows.append({"position": pos, "hiv2_aa": aa_wt, "to_hiv1_aa": aa_h1,
                     "delta_dg": round(mut_pred - wt_pred, 4)})
    rows.sort(key=lambda r: -abs(r["delta_dg"]))
    top20 = rows[:20]
    log.info(f"  Divergent positions: {len(divergent)}/99")
    log.info(f"  {'Pos':<5s} {'H2':<4s} {'->':<4s} {'H1':<4s} {'deltadG':<10s}")
    for r in top20:
        log.info(f"  {r['position']:<5d} {r['hiv2_aa']:<4s} {'->':<4s} "
                 f"{r['to_hiv1_aa']:<4s} {r['delta_dg']:<10.4f}")
    results["exp3_mutagenesis"] = {
        "n_divergent_positions": len(divergent),
        "wildtype_dG_pred": round(wt_pred, 4),
        "top20_by_abs_deltadg": top20,
        "legacy": {"K7Q_legacy": LEGACY["exp3"]["K7Q"],
                   "source": ("Tier5_comprehensive_results.json "
                              "exp3_wildtype_substitution (poisoned corpus)")},
    }

    # ---------------------------------------------------------------
    results["protbert_disposition"] = {
        "decision": "NOT_RERUN",
        "reason": ("ProtBERT embeddings are unavailable on the honest corpora: "
                   "(1) GENUINE.csv BERT_1..1024 columns are all-NULL "
                   "(verified); (2) HIV2_PROTEASE_ML_READY_DATASET.csv has NO "
                   "BERT columns (verified, 0 of 3639); (3) on-disk "
                   "BERT_FEATURES CSVs are keyed by PDB IDs (e.g. 6B36_1|Chains) "
                   "with no verified mapping to the corpus; (4) honest "
                   "re-extraction requires Rostlab/prot_bert download + GPU/"
                   "Colab inference (torch 2.10.0+cpu, cuda=False) -> "
                   "D1-class infeasibility, documented, not rerun."),
        "legacy_values": {"cross_species_bert_only": 0.1632,
                          "within_species_bert_only": 0.617,
                          "combined_3103D": 0.4753},
    }
    results["gnn_dbpt_disposition"] = {
        "decision": "NOT_RERUN (already D1 in metrics_reruns.json)",
        "reason": ("torch is CPU-only; GNN/DBPT+ are days of GPU compute. "
                   "Legacy zero-shot values were already near-zero or negative "
                   "(GNN R=-0.1709, DBPT+ R=0.2122)."),
    }

    results["run_seconds"] = round(time.time() - t0, 1)
    out = os.path.join(RESCUE, "legacy_rerun_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    log.info(f"\n[DONE] Honest values written to {out}")
    log.info(f"       Elapsed {results['run_seconds']}s")
    return results


if __name__ == "__main__":
    main()
