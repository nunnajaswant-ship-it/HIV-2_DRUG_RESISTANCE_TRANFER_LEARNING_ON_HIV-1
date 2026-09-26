"""
Transfer Learning Pipeline v2 — HIV-1 -> HIV-2 (TRUE transfer + ONION-Net refinement)
======================================================================================
v2 upgrades over transfer_learning.py:

* FIX: "fine-tuned" stage now performs TRUE transfer. HIV-1 pre-trained models'
  zero-shot predictions are injected as meta-features into the HIV-2 fine-tune
  model. Previously fine-tuned == supervised because the HIV-1 model was never
  used in stage 2.
* FIX: pooled zero-shot is computed two ways:
    (a) legacy-comparable: biochem 2079-D (include_stats=False), HIV-2 rows
        filtered to IC50/KI/KD, legacy RF params (600 trees, leaf=2, split=5,
        max_features=sqrt). This reproduces the manuscript's R=0.5163 claim.
    (b) combined_pocket full HIV-2 set (the v2 deployment feature).
* NEW PART 6: ONION-Net structural refinement. Residual correction on the
  10,478-D docking features, evaluated with GroupKFold BY STRUCTURE HASH
  (fixes the legacy plain-KFold sequence leak that inflated residual R).
* NEW PART 7: pocket position importance. Ranks the POCKET19 positions by RF
  feature importance using per-residue ESM-2 blocks limited to the pocket.

Primary feature set for deployment: combined_pocket (biochem 2121-D +
ESM-2 pocket-19 1280-D). Primary model: RandomForest (600, depth 20, sqrt).
"""
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark_config import (
    DATA_DIR,
    ESM2_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    RANDOM_SEED,
)

from transfer_learning import (
    R_KCAL,
    T_310,
    RT_310,
    N_SPLITS,
    N_BOOTSTRAP,
    LEARNING_FRACTIONS,
    LC_N_SEEDS,
    CONFORMAL_CAL_FRAC,
    HIV1_DRUG_MAP,
    HIV2_DRUG_MAP,
    COMMON_DRUGS,
    FEATURE_SETS,
    FEATURE_LABELS,
    POCKET19_1INDEX,
    MODEL_KEYS,
    MODEL_LABELS,
    set_seed,
    ic50_to_dg,
    evaluate_predictions,
    _load_esm2_mapping,
    FeatureStore,
    _feature_store,
    Bundle,
    build_bundles,
    build_fast_model,
    group_kfold_oof,
    kfold_oof,
    mean_r,
    pretrain_hiv1,
    zero_shot_eval,
    zero_shot_pooled_eval,
    finetune_oof,
    supervised_oof,
    run_feature_ablation,
    run_model_comparison,
    run_learning_curve,
    run_conformal,
    plot_transfer_comparison,
    save_results,
    load_hiv1_data,
    load_hiv2_chembl,
    load_hiv2_clinical,
)

warnings.filterwarnings("ignore")

WORKSPACE = Path(__file__).resolve().parents[2]
SEQ_TO_PDB = WORKSPACE / "advanced_feature_extraction" / "esmfold_structures" / "sequence_to_pdb.csv"
ONION_NPZ = WORKSPACE / "advanced_feature_extraction" / "docked_complexes" / "docked_onionnet_features.npz"

LEGACY_RF_PARAMS = dict(
    n_estimators=600, max_depth=20, min_samples_leaf=2,
    min_samples_split=5, max_features="sqrt", n_jobs=-1, random_state=42,
)


# ============================================================================
# TRUE TRANSFER FINE-TUNE — HIV-1 predictions as meta-features into HIV-2 model
# ============================================================================
def transfer_finetune_oof(bundle, model_name, drugs=COMMON_DRUGS,
                          seed=RANDOM_SEED):
    """HIV-1 pre-train per drug -> zero-shot prediction column -> HIV-2 model
    trained on [HIV-2 features | zs_pred] with 5-fold OOF.  This is genuine
    knowledge transfer: the HIV-1 genotype->phenotype map enters the HIV-2
    model as a feature, so fine-tuned != supervised."""
    res = dict.fromkeys(drugs, None)
    pooled_t, pooled_p = [], []
    hiv1_models, _ = pretrain_hiv1(bundle, model_name)
    for drug in drugs:
        if drug not in hiv1_models:
            continue
        mask = bundle.drug2 == drug
        if mask.sum() < 8:
            continue
        X2d = bundle.X2[mask]
        y2d = bundle.y2[mask]
        zs = np.ravel(hiv1_models[drug].predict(X2d)).reshape(-1, 1)
        Xtr = np.hstack([X2d, zs]).astype(np.float32)
        oof_t, oof_p = kfold_oof(Xtr, y2d,
                                 lambda: build_fast_model(model_name),
                                 seed=seed)
        res[drug] = evaluate_predictions(oof_t, oof_p)
        pooled_t.extend(oof_t.tolist())
        pooled_p.extend(oof_p.tolist())
    res["MEAN"] = evaluate_predictions(pooled_t, pooled_p)
    return res


# ============================================================================
# POOLED ZERO-SHOT — legacy-comparable (reproduces manuscript R=0.5163)
# ============================================================================
def zero_shot_pooled_legacy():
    """Legacy-comparable pooled zero-shot: biochem 2079-D (include_stats=False),
    HIV-2 filtered to IC50/KI/KD, legacy RF hyper-parameters -> one model on all
    HIV-1 (5683) predicting all filtered HIV-2 (n~137)."""
    from opencode_shared.utils import extract_global_features

    df1 = pd.read_csv(DATA_DIR / "STANFORD_HIV1_GENUINE.csv", low_memory=False)
    df1 = df1.dropna(subset=["Sequence", "Binding_Affinity_kcal_mol"])
    df1["Binding_Affinity_kcal_mol"] = pd.to_numeric(
        df1["Binding_Affinity_kcal_mol"], errors="coerce")
    df1 = df1[(df1["Binding_Affinity_kcal_mol"] >= -20)
              & (df1["Binding_Affinity_kcal_mol"] <= 0)]
    seqs1 = df1["Sequence"].astype(str).str.strip().str.upper().values
    y1 = df1["Binding_Affinity_kcal_mol"].values.astype(np.float32)
    keep1 = np.array([len(s) >= 90 for s in seqs1])
    X1 = np.array([extract_global_features(s, include_stats=False)
                   for s in seqs1[keep1]], dtype=np.float32)
    n1 = int(keep1.sum())

    df2 = pd.read_csv(DATA_DIR / "HIV2_PROTEASE_ML_READY_DATASET.csv",
                      low_memory=False)
    valid = df2["Assay_Type"].astype(str).str.upper().str.strip().isin(
        ["IC50", "KI", "KD"])
    df2 = df2[valid].copy()
    df2["_dg"] = df2["Standard_Value"].apply(
        lambda x: ic50_to_dg(float(x)) if float(x) > 0 else float("nan"))
    df2 = df2.dropna(subset=["_dg"])
    df2 = df2[(df2["_dg"] >= -20) & (df2["_dg"] <= 0)]
    seqs2 = df2["Mutant_Sequence"].astype(str).str.strip().str.upper().values
    y2 = df2["_dg"].values.astype(np.float32)
    keep2 = np.array([len(s) >= 90 for s in seqs2])
    X2 = np.array([extract_global_features(s, include_stats=False)
                   for s in seqs2[keep2]], dtype=np.float32)
    n2 = int(keep2.sum())

    rf = RandomForestRegressor(**LEGACY_RF_PARAMS)
    rf.fit(X1, y1[keep1])
    p2 = rf.predict(X2)
    m = evaluate_predictions(y2[keep2], p2)
    print(f"\n  --- Pooled zero-shot LEGACY-COMPARABLE (biochem 2079-D, "
          f"IC50/KI/KD-only, legacy RF) ---")
    print(f"    HIV-1 n={n1} | HIV-2 n={n2} | "
          f"Pearson R = {m['pearson_r']:.4f} "
          f"(CI=[{m['ci_pearson_lo']:.4f}, {m['ci_pearson_hi']:.4f}])")
    m["n_hiv1"] = n1
    m["n_hiv2"] = n2
    m["feature_dim"] = int(X1.shape[1])
    return m


# ============================================================================
# PART 6 — ONION-NET RESIDUAL REFINEMENT (leakage-free GroupKFold)
# ============================================================================
def onet_refinement(bundle, base_model, drugs=COMMON_DRUGS, seed=RANDOM_SEED):
    """Base model (fine-tuned on HIV-2) predicts dG; an RF on the 10,478-D
    ONION-Net docking features predicts the residual.  Residual RF is evaluated
    with GroupKFold grouping by structure hash so the same sequence never
    appears in both train and validation (the legacy plain-KFold leak is
    removed).  Runs on the subset of HIV-2 ChEMBL rows with docking features."""
    print("\n" + "=" * 78)
    print("PART 6 — ONION-NET STRUCTURAL REFINEMENT "
          "(GroupKFold by structure, no leak)".ljust(78))
    print("=" * 78)
    if not SEQ_TO_PDB.exists() or not ONION_NPZ.exists():
        print("  [warn] ONION-Net files missing — skipping Part 6")
        return None

    stp = pd.read_csv(SEQ_TO_PDB)
    seq_hash = dict(zip(stp["Cleaned_Sequence"].astype(str).str.upper(),
                        stp["Hash"].astype(str)))
    npz = np.load(ONION_NPZ, allow_pickle=True)
    onion_keys = [str(k) for k in np.asarray(npz["keys"])]
    onion_feats = np.asarray(npz["features"])
    onion = {k: i for i, k in enumerate(onion_keys)}
    rev_map = {v: k for k, v in HIV2_DRUG_MAP.items()}

    feat_idx = []
    for seq, drug in zip(bundle.seq2, bundle.drug2):
        h = seq_hash.get(str(seq).strip().upper(), "MISSING")
        d = str(rev_map.get(drug, "MISSING")).strip().upper()
        feat_idx.append(onion.get(f"{d}_{h}", -1))
    feat_idx = np.array(feat_idx)
    valid = (feat_idx >= 0) & np.isin(bundle.drug2, drugs)
    n_valid = int(valid.sum())
    if n_valid < 20:
        print(f"  [warn] only {n_valid} ONION-mapped rows — skipping Part 6")
        return None

    Xo = onion_feats[feat_idx[valid]].astype(np.float32)
    Xb = bundle.X2[valid]
    y_o = bundle.y2[valid]
    drugs_o = bundle.drug2[valid]
    print(f"  ONION-mapped rows: {n_valid} | feature dim: {Xo.shape[1]} "
          f"| drugs: {dict(pd.Series(drugs_o).value_counts())}")

    # Base model OOF predictions on this subset
    base_oof = np.full(n_valid, np.nan)
    n_split = min(N_SPLITS, max(2, n_valid // 8))
    kf = KFold(n_splits=n_split, shuffle=True, random_state=seed)
    for tr, te in kf.split(Xb):
        m = build_fast_model(base_model)
        m.fit(Xb[tr], y_o[tr])
        base_oof[te] = np.ravel(m.predict(Xb[te]))

    residuals = y_o - base_oof
    grp = np.array([seq_hash.get(str(s).strip().upper(), "0")
                    for s in bundle.seq2[valid]])
    uniq_g = len(np.unique(grp))
    print(f"  Unique structures: {uniq_g}")

    # Residual RF evaluated by GroupKFold on structure hash (no leak)
    resid_preds = np.full(n_valid, np.nan)
    gkf_n = min(N_SPLITS, max(2, uniq_g // 2))
    gkf = GroupKFold(n_splits=gkf_n).split(Xo, groups=grp)
    for tr, te in gkf:
        rf_r = RandomForestRegressor(
            n_estimators=400, max_depth=12, max_features=0.3,
            min_samples_leaf=2, min_samples_split=5, n_jobs=-1,
            random_state=seed)
        rf_r.fit(Xo[tr], residuals[tr])
        resid_preds[te] = rf_r.predict(Xo[te])

    ok = np.isfinite(base_oof) & np.isfinite(resid_preds)
    corr = base_oof[ok] + resid_preds[ok]
    m_base = evaluate_predictions(y_o[ok], base_oof[ok])
    m_resid = evaluate_predictions(residuals[ok], resid_preds[ok])
    m_corr = evaluate_predictions(y_o[ok], corr)

    # Per-drug corrected metrics
    per_drug = {}
    for d in drugs:
        dm = (np.asarray(drugs_o[ok]) == d)
        if dm.sum() >= 5:
            per_drug[d] = evaluate_predictions(y_o[ok][dm], corr[dm])

    print(f"  BASE (fine-tuned)         R={m_base['pearson_r']:.4f}")
    print(f"  RESIDUAL (ONION RF)       R={m_resid['pearson_r']:.4f} "
          f"(RMSE={m_resid['rmse']:.4f})")
    print(f"  CORRECTED (base+residual) R={m_corr['pearson_r']:.4f} "
          f"(R2={1 - np.mean((y_o[ok] - corr) ** 2) / np.var(y_o[ok]):.4f})")
    for d in drugs:
        if d in per_drug:
            print(f"    {d}: corrected R={per_drug[d]['pearson_r']:.4f}")
    print(f"  -> variance explained by ONION residual: "
          f"{max(0.0, m_resid['pearson_r']) ** 2 * 100:.1f}%")

    return {
        "n_rows": n_valid,
        "n_structures": int(uniq_g),
        "feature_dim": int(Xo.shape[1]),
        "base": m_base,
        "residual": m_resid,
        "corrected": m_corr,
        "per_drug_corrected": {str(k): v for k, v in per_drug.items()},
    }


# ============================================================================
# PART 7 — POCKET POSITION IMPORTANCE (per-residue ESM-2, 19 positions)
# ============================================================================
def pocket_position_importance(hiv1_df, max_rows=2000):
    """USING ONLY THE 19 CONSERVED POCKET POSITIONS, rank them by RF feature
    importance from per-residue ESM-2 embeddings.  Shows which pocket residues
    carry the HIV-1 -> HIV-2 transfer signal."""
    print("\n" + "=" * 78)
    print("PART 7 — POCKET POSITION IMPORTANCE "
          "(per-residue ESM-2 blocks, 19 positions)".ljust(78))
    print("=" * 78)
    _, resid_map = _load_esm2_mapping()

    lab = {}
    for _, row in hiv1_df.iterrows():
        s = str(row["sequence"]).strip().upper()
        lab.setdefault(s, []).append(float(row["dg"]))
    seqs = [s for s in resid_map.keys() if s in lab]
    rng = np.random.RandomState(RANDOM_SEED)
    if len(seqs) > max_rows:
        seqs = list(rng.choice(seqs, size=max_rows, replace=False))

    X_list = []
    y_list = []
    for s in seqs:
        r = resid_map[s]
        if r.shape[0] < 99:
            continue
        idx = [p - 1 for p in POCKET19_1INDEX if 1 <= p <= r.shape[0]]
        X_list.append(r[idx].reshape(-1))
        y_list.append(float(np.median(lab[s])))
    if len(X_list) < 100:
        print("  [warn] too few sequences with labels — skipping Part 7")
        return None
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"  Sequences: {len(y)} | block dim: {X.shape[1]} "
          f"({len(POCKET19_1INDEX)} pos x 1280)")

    rf = RandomForestRegressor(n_estimators=300, max_depth=12,
                               max_features=0.2, n_jobs=-1,
                               random_state=RANDOM_SEED)
    rf.fit(X, y)
    imp = rf.feature_importances_.reshape(len(POCKET19_1INDEX), 1280)
    pos_imp = imp.sum(axis=1)
    rank = np.argsort(pos_imp)[::-1]

    print(f"  Pocket position importance ranking:")
    rows = []
    for r_i, p_i in enumerate(rank, 1):
        pos = int(POCKET19_1INDEX[p_i])
        print(f"    {r_i:2d}. position {pos:3d}"
              f"   importance {pos_imp[p_i]:.5f}")
        rows.append({"rank": r_i, "position": pos,
                     "importance": float(pos_imp[p_i])})

    total = pos_imp.sum()
    print(f"  Sum importance: {total:.5f} | "
          f"top-5 positions hold {100 * pos_imp[rank[:5]].sum() / total:.1f}% "
          f"of pocket signal")
    return {
        "n_sequences": int(len(y)),
        "ranked_positions": rows,
        "importance_sum": float(total),
    }


# ============================================================================
# MAIN  (v2)
# ============================================================================
def main():
    t_start = time.time()
    set_seed()
    print("=" * 78)
    print("TRANSFER LEARNING PIPELINE v2 — HIV-1 -> HIV-2 "
          "(TRUE transfer + ONION-Net)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 78)

    # ------------------------------------------------------------------ load
    hiv1_df = load_hiv1_data()
    hiv2_df = load_hiv2_chembl()
    hiv2_clinical_df, clinical_info = load_hiv2_clinical()

    bundles = build_bundles(hiv1_df, hiv2_df)

    output = {
        "data_stats": {
            "hiv1_n": int(len(hiv1_df)),
            "hiv1_drugs": {str(k): int(v)
                           for k, v in hiv1_df["drug"].value_counts().items()},
            "hiv1_dg_range": [float(hiv1_df["dg"].min()),
                              float(hiv1_df["dg"].max())],
            "hiv2_chembl_n": int(len(hiv2_df)),
            "hiv2_chembl_drugs": {str(k): int(v)
                                  for k, v in hiv2_df["drug"].value_counts().items()},
            "hiv2_common4_n": int(np.sum(
                np.isin(hiv2_df["drug"].values, COMMON_DRUGS))),
            "clinical": clinical_info,
            "common_drugs": COMMON_DRUGS,
        },
        "constants": {
            "rt_310": RT_310,
            "n_bootstrap_ci": N_BOOTSTRAP,
            "n_splits": N_SPLITS,
            "pocket19_1index": POCKET19_1INDEX,
        },
    }

    # --------------------------------------------------------------- Part 1
    part1 = run_feature_ablation(bundles, model_name="T4_RF")
    output["part1_feature_ablation"] = part1

    # Pooled zero-shots (two flavors)
    try:
        zs_legacy = zero_shot_pooled_legacy()
        output["part1_pooled_zero_shot_legacy_biochem_2079"] = {
            str(k): v for k, v in zs_legacy.items()}
    except Exception as e:
        print(f"  [warn] legacy pooled zero-shot failed: {e}")
        output["part1_pooled_zero_shot_legacy_biochem_2079"] = {"error": str(e)}

    zs_cp = zero_shot_pooled_eval(bundles["combined_pocket"], "T4_RF")
    output["part1_pooled_zero_shot_combined_pocket"] = {
        str(k): v for k, v in zs_cp.items()}

    # --------------------------------------------------------------- Part 2
    part2, model_best = run_model_comparison(bundles["combined_pocket"])
    output["part2_model_comparison"] = {
        mk: {
            "model": r["model_label"],
            "hiv1_cv_mean_r": r["hiv1_cv_mean_r"],
            "finetuned_mean_r": r["finetuned_mean_r"],
            "hiv1_cv": {str(k): v for k, v in r["hiv1_cv"].items()},
            "finetuned": {str(k): v for k, v in r["finetuned"].items()},
        }
        for mk, r in part2.items()
    }

    rf_ft = part2.get("T4_RF", {}).get("finetuned_mean_r", float("nan"))
    best_ft = part2.get(model_best, {}).get("finetuned_mean_r", float("nan"))
    if np.isfinite(rf_ft) and (not np.isfinite(best_ft)
                               or rf_ft >= best_ft * 0.95):
        model_best = "T4_RF"
        print(f"\n  >> Using T4_RF as primary model "
              f"(R={rf_ft:.4f} >= 95% of best)")
    output["best_model"] = model_best
    output["model_labels"] = MODEL_LABELS

    # --------------------------------------------------------------- Part 3
    n_hiv2_rows = len(hiv2_df)
    curve = run_learning_curve(bundles["combined_pocket"], model_best,
                               n_hiv2_rows)
    output["part3_learning_curve"] = {
        "best_model": model_best,
        "n_hiv2_samples": int(n_hiv2_rows),
        "zero_shot_r": curve["zero_shot_r"],
        "fractions": {str(k): v for k, v in curve["fractions"].items()},
    }

    # --------------------------------------------------------------- Part 4
    conformal = run_conformal(bundles["combined_pocket"], model_best)
    output["part4_conformal_prediction"] = {
        str(k): v for k, v in conformal.items()}

    # --------------------------------------------------------------- Part 5
    print("\n" + "=" * 78)
    print(f"PART 5 — APPROACH COMPARISON (combined_pocket, "
          f"{MODEL_LABELS[model_best]})".ljust(78))
    print("=" * 78)
    bundle = bundles["combined_pocket"]

    zs_models, _ = pretrain_hiv1(bundle, model_best)
    zero_shot = zero_shot_eval(bundle, zs_models)
    transfer_ft = transfer_finetune_oof(bundle, model_best)
    sup = supervised_oof(bundle, model_best)

    approaches = {
        "Zero-shot (HIV-1 only)": zero_shot,
        "Transfer fine-tune (HIV-1 meta)": transfer_ft,
        "Supervised (HIV-2 only)": sup,
    }
    print("\nAPPROACH COMPARISON (Pearson R)")
    header = (f"{'Approach':<32}"
              + "".join(f"{d:>8}" for d in COMMON_DRUGS)
              + f"{'MEAN':>8}")
    print(header)
    print("-" * len(header))
    for name, res in approaches.items():
        row = f"{name:<32}"
        for d in COMMON_DRUGS:
            r = res.get(d, {}).get("pearson_r", np.nan)
            row += (f"{r:>8.3f}" if np.isfinite(r) else f"{'N/A':>8}")
        m = res.get("MEAN", {}).get("pearson_r", np.nan)
        row += (f"{m:>8.3f}" if np.isfinite(m) else f"{'N/A':>8}")
        print(row)
    print("-" * len(header))
    output["part5_approach_comparison"] = {
        name: {str(k): v for k, v in res.items()}
        for name, res in approaches.items()
    }
    output["approach_means"] = {name: res.get("MEAN", {})
                                for name, res in approaches.items()}

    try:
        plot_transfer_comparison(approaches,
                                 save_path=FIGURES_DIR / "fig_v2_approaches.png")
    except Exception as e:
        print(f"  [warn] figure failed: {e}")

    # --------------------------------------------------------------- Part 6
    onion = onet_refinement(bundle, model_best)
    output["part6_onion_refinement"] = onion

    # --------------------------------------------------------------- Part 7
    pocket_imp = pocket_position_importance(hiv1_df)
    output["part7_pocket_importance"] = pocket_imp

    # ------------------------------------------------------------ save json
    out_path = RESULTS_DIR / "transfer_learning_v2_results.json"
    save_results(output, out_path)

    elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print("PIPELINE v2 COMPLETE")
    print(f"Duration: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Results:  {out_path}")
    print(f"Figures:  {FIGURES_DIR / 'fig_v2_approaches.png'}")
    print("=" * 78)
    return output


if __name__ == "__main__":
    main()