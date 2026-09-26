"""
retrain_honest.py
=================
DELIVERABLE 5 — HONEST MODEL RETRAINING ON THE QUARANTINED CORPUS.

What this script does (and why):
  - Trains ONLY on STANFORD_HIV1_GENUINE.csv (5,683 rows; the 1,500
    seed-42 synthetic rows are quarantined in
    STANFORD_HIV1_SYNTHETIC_QUARANTINED.csv and MUST NOT touch training).
  - Uses grouped CV: sequences clustered at 95% greedy identity
    (same algorithm as run_clustering_splits.py, threshold 0.95) and
    GroupKFold(5) so no cluster leaks across folds.
  - Recomputes, from genuine data ONLY:
      (a) within-HIV-1 grouped CV  (manuscript claimed 0.759 +- 0.017)
      (b) zero-shot cross-species  (manuscript claimed R=0.504)
      (c) ONION-Net residual stage (manuscript claimed R=0.667, 44.5%)
      (d) hybrid corrected R       (manuscript claimed R=0.734)
      (e) honest HIVDB baseline audit (manuscript claimed R=0.722)
  - Every headline number is written to RESCUE/metrics_manifest.json
    together with the SHA-256 of every input file used. Numbers covenant:
    nothing on disk that is not produced by this script.
  - Feature space: 2,079-D biochem (Z3+K10+VHSE8 per residue, 99
    positions, include_stats=False) matching the manuscript's documented
    "2079-D biochem" description. NOTE: the legacy Tier5 script silently
    used extract_global_features default (2,121-D). Documented in the
    manifest as a discrepancy.

Sign convention (fixed and documented here; see RESCUE/SIGN_CONVENTION.md):
  - Model output is predicted DeltaG (kcal/mol, negative = tighter binding).
  - HIVDB Clinical_Penalty_Score: higher = more resistant.
  - Biology: resistant -> weaker binding -> LESS negative DeltaG.
  - Therefore the biologically-correct correlation between penalty score
    and predicted DeltaG is POSITIVE. The manuscript figure used -pc
    (i.e. -predicted DeltaG on the y-axis) which flips the sign; both
    describe the same relationship. This script reports +pred and states
    the convention explicitly.

Outputs:
  - RESCUE/cluster_splits_genuine/  (95% cluster assignments + GroupKFold ids)
  - RESCUE/metrics_manifest.json    (numbers covenant)
  - RESCUE/retrain_log.txt          (full run log)
  - RESCUE/within_species_cv_folds.csv
  - RESCUE/zero_shot_predictions.csv
  - RESCUE/hybrid_predictions.csv
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opencode_shared import (
    WORKSPACE, KNOWN_INFO, extract_global_features, compute_ic50_to_delta_g,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "retrain_log.txt"), mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("retrain_honest")

RESCUE = os.path.dirname(os.path.abspath(__file__))
SPLIT_DIR = os.path.join(RESCUE, "cluster_splits_genuine")
os.makedirs(SPLIT_DIR, exist_ok=True)

GENUINE_CSV = os.path.join(KNOWN_INFO, "STANFORD_HIV1_GENUINE.csv")
SYNTH_CSV = os.path.join(KNOWN_INFO, "STANFORD_HIV1_SYNTHETIC_QUARANTINED.csv")
H2_ASSAY_CSV = os.path.join(KNOWN_INFO, "HIV2_PROTEASE_ML_READY_DATASET.csv")
H2_CLINICAL_CSV = os.path.join(KNOWN_INFO, "HIV2_CLINICAL_ML_READY_FLAT.csv")
SEQ_TO_PDB = os.path.join(WORKSPACE, "advanced_feature_extraction",
                          "esmfold_structures", "sequence_to_pdb.csv")
ONION_NPZ = os.path.join(WORKSPACE, "advanced_feature_extraction",
                         "docked_complexes", "docked_onionnet_features.npz")

RF_PARAMS = dict(n_estimators=600, max_depth=20, min_samples_leaf=2,
                 min_samples_split=5, max_features="sqrt",
                 n_jobs=-1, random_state=42)
SEED = 42
CLUSTER_THRESHOLD = 0.95
N_FOLDS = 5


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pearson(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3:
        return None
    return pearsonr(y[m], p[m])


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


def bootstrap_ci(y, p, n=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    rs = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2 or len(np.unique(p[idx])) < 2:
            continue
        try:
            rs.append(pearsonr(y[idx], p[idx])[0])
        except Exception:
            pass
    rs = np.array(rs)
    if len(rs) < 100:
        return None
    return {"ci_lower": round(float(np.percentile(rs, 2.5)), 4),
            "ci_upper": round(float(np.percentile(rs, 97.5)), 4),
            "mean": round(float(np.mean(rs)), 4)}


def hamming_identity(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(x == y for x, y in zip(a[:n], b[:n])) / max(len(a), len(b))


def greedy_cluster(sequences, threshold=CLUSTER_THRESHOLD):
    """Greedy clustering: sort by length desc, assign to first cluster whose
    representative shares identity >= threshold. Mirrors run_clustering_splits.py."""
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]),
                   reverse=True)
    clusters = []
    reps = []
    for i in order:
        seq = sequences[i]
        assigned = False
        for cid, rep in enumerate(reps):
            if hamming_identity(seq[:len(rep)], rep) >= threshold:
                clusters[cid].append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])
            reps.append(seq)
    return clusters, order


def load_genuine_hiv1():
    df = pd.read_csv(GENUINE_CSV, low_memory=False)
    df = df.dropna(subset=["Sequence", "Binding_Affinity_kcal_mol"])
    df["Binding_Affinity_kcal_mol"] = pd.to_numeric(
        df["Binding_Affinity_kcal_mol"], errors="coerce")
    df = df[(df["Binding_Affinity_kcal_mol"] >= -20)
            & (df["Binding_Affinity_kcal_mol"] <= 0)]
    seqs = df["Sequence"].astype(str).str.strip().str.upper().values
    y = df["Binding_Affinity_kcal_mol"].values.astype(np.float32)
    X = np.array([extract_global_features(s, include_stats=False)
                  for s in seqs if len(s) >= 90], dtype=np.float32)
    keep = np.array([len(s) >= 90 for s in seqs])
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
    X = np.array([extract_global_features(s, include_stats=False)
                  for s in seqs if len(s) >= 90], dtype=np.float32)
    keep = np.array([len(s) >= 90 for s in seqs])
    return X, y[keep], drugs[keep], df.loc[keep].copy()


def load_onion_lookup():
    stp = pd.read_csv(SEQ_TO_PDB)
    seq_hash = dict(zip(stp["Cleaned_Sequence"].astype(str).str.upper(),
                        stp["Hash"].astype(str)))
    npz = np.load(ONION_NPZ, allow_pickle=True)
    keys = [str(k) for k in npz["keys"]]
    feats = npz["features"]
    onion = {k: i for i, k in enumerate(keys)}
    return seq_hash, onion, feats


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("HONEST RETRAINING — GENUINE-ONLY CORPUS, GROUPED CV, SEED 42")
    log.info("=" * 70)

    manifest = {
        "pipeline": "RESCUE/retrain_honest.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_hashes": {
            "STANFORD_HIV1_GENUINE.csv": sha256(GENUINE_CSV),
            "STANFORD_HIV1_SYNTHETIC_QUARANTINED.csv": sha256(SYNTH_CSV),
            "HIV2_PROTEASE_ML_READY_DATASET.csv": sha256(H2_ASSAY_CSV),
            "HIV2_CLINICAL_ML_READY_FLAT.csv": sha256(H2_CLINICAL_CSV),
            "sequence_to_pdb.csv": sha256(SEQ_TO_PDB),
            "docked_onionnet_features.npz": sha256(ONION_NPZ),
        },
        "model_config": RF_PARAMS,
        "seed": SEED,
        "feature_space": "2079-D biochem (Z3+K10+VHSE8 per residue x99, include_stats=False)",
        "legacy_discrepancy_note": (
            "Legacy Tier5 script used extract_global_features default "
            "(2121-D, include_stats=True) while manuscript documents 2079-D. "
            "This run uses the documented 2079-D space."),
        "cluster_threshold": CLUSTER_THRESHOLD,
        "cv": f"GroupKFold({N_FOLDS}) on {int(CLUSTER_THRESHOLD*100)}% identity clusters",
    }

    # ---------------------------------------------------------------
    log.info("\n[1] Load genuine-only HIV-1 corpus")
    X1, y1, df1 = load_genuine_hiv1()
    n_synth = sum(1 for _ in open(SYNTH_CSV, encoding="utf-8")) - 1
    manifest["corpus"] = {
        "genuine_rows_trained": int(len(df1)),
        "genuine_unique_sequences": int(df1["Sequence"].nunique()),
        "synthetic_rows_excluded": int(n_synth),
        "synthetic_quarantine_file": "STANFORD_HIV1_SYNTHETIC_QUARANTINED.csv",
        "feature_matrix": [int(X1.shape[0]), int(X1.shape[1])],
        "drugs_present": sorted(df1["Inhibitor"].dropna().unique().tolist()),
    }
    log.info(f"  genuine rows: {len(df1)}  unique seqs: {df1['Sequence'].nunique()}"
             f"  synthetic excluded: {n_synth}")

    # ---------------------------------------------------------------
    log.info(f"\n[2] Greedy clustering at {CLUSTER_THRESHOLD:.0%} identity")
    uniq_seqs = list(dict.fromkeys(df1["Sequence"].astype(str).str.upper()))
    clusters, order = greedy_cluster(uniq_seqs)
    seq_to_cluster = {}
    for cid, members in enumerate(clusters):
        for mi in members:
            seq_to_cluster[uniq_seqs[mi]] = cid
    cluster_ids = np.array([seq_to_cluster[s] for s in
                            df1["Sequence"].astype(str).str.upper()])
    manifest["clustering"] = {
        "unique_sequences": len(uniq_seqs),
        "n_clusters": len(clusters),
        "singletons": int(sum(1 for c in clusters if len(c) == 1)),
        "max_cluster_size": int(max(len(c) for c in clusters)),
        "median_cluster_size": float(np.median([len(c) for c in clusters])),
    }
    log.info(f"  {len(uniq_seqs)} unique seqs -> {len(clusters)} clusters")
    np.save(os.path.join(SPLIT_DIR, "cluster_ids_genuine.npy"), cluster_ids)
    np.save(os.path.join(SPLIT_DIR, "cluster_sizes.npy"),
            np.array([len(c) for c in clusters]))
    pd.DataFrame({"cluster_id": cluster_ids,
                  "Sequence": df1["Sequence"].astype(str).str.upper().values}
                 ).to_csv(os.path.join(SPLIT_DIR, "cluster_assignments.csv"),
                          index=False)

    # ---------------------------------------------------------------
    log.info(f"\n[3] Within-HIV-1 grouped CV (GroupKFold({N_FOLDS}))")
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_rows = []
    fold_R = []
    for fold, (tr, te) in enumerate(gkf.split(X1, y1, groups=cluster_ids)):
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X1[tr], y1[tr])
        p_te = rf.predict(X1[te])
        m = metrics(y1[te], p_te)
        m["fold"] = fold + 1
        m["n_train"], m["n_test"] = int(len(tr)), int(len(te))
        fold_rows.append(m)
        fold_R.append(m["pearson_r"])
        log.info(f"  Fold {fold+1}: R={m['pearson_r']:.4f} "
                 f"RMSE={m['rmse']:.4f} n_tr={len(tr)} n_te={len(te)}")
    mean_r = float(np.mean(fold_R))
    std_r = float(np.std(fold_R))
    manifest["within_species_grouped_cv"] = {
        "mean_pearson_r": round(mean_r, 4),
        "std_pearson_r": round(std_r, 4),
        "per_fold": fold_rows,
        "note": ("Grouped CV at 95% sequence-identity clusters. "
                 "Legacy ungrouped KFold(5) on poisoned 7,183-row corpus "
                 "produced 0.759 +- 0.017; that corpus contained 1,500 "
                 "synthetic rows and its split was not leakage-free."),
    }
    pd.DataFrame(fold_rows).to_csv(
        os.path.join(RESCUE, "within_species_cv_folds.csv"), index=False)
    log.info(f"  WITHIN-SPECIES GROUPED CV: R = {mean_r:.4f} +- {std_r:.4f}")

    # ---------------------------------------------------------------
    log.info("\n[4] Zero-shot cross-species (train genuine HIV-1, "
             "predict HIV-2 assays)")
    X2, y2, drugs2, df2 = load_hiv2_assays()
    rf_zs = RandomForestRegressor(**RF_PARAMS)
    rf_zs.fit(X1, y1)
    p2 = rf_zs.predict(X2)
    m_zs = metrics(y2, p2)
    ci_zs = bootstrap_ci(y2, p2)
    transfer = m_zs["pearson_r"] / mean_r if mean_r != 0 else None
    manifest["zero_shot_cross_species"] = {
        "n_hiv2_assay_rows": int(len(y2)),
        "metrics": m_zs,
        "bootstrap_ci_95": ci_zs,
        "transfer_ratio_vs_within_species": (
            round(transfer, 3) if transfer is not None else None),
        "assay_filter": "Assay_Type in {IC50,KI,KD}; Standard_Value>0; dG in [-20,0]",
        "note": ("Legacy value R=0.5044 (p=1.5e-19) came from the same "
                 "configuration on the POISONED 7,183-row corpus; this run "
                 "uses genuine-only 5,683 rows."),
    }
    log.info(f"  ZERO-SHOT: R={m_zs['pearson_r']:.4f} (p={m_zs['pearson_p']:.2e}) "
             f"CI={ci_zs['ci_lower']}..{ci_zs['ci_upper']} n={len(y2)}")
    pd.DataFrame({"Mutant_Sequence": df2["Mutant_Sequence"].values,
                  "Drug": drugs2, "dG_true": y2,
                  "dG_pred": p2}).to_csv(
        os.path.join(RESCUE, "zero_shot_predictions.csv"), index=False)

    # ---------------------------------------------------------------
    log.info("\n[5] Hybrid: ONION-Net residual stage on docked HIV-2 rows")
    seq_hash, onion_lookup, onion_feats = load_onion_lookup()
    drug_map = {d.upper(): d for d in [
        "AMPRENAVIR", "ATAZANAVIR", "DARUNAVIR", "INDINAVIR", "LOPINAVIR",
        "NELFINAVIR", "RITONAVIR", "SAQUINAVIR", "TIPRANAVIR"]}
    feat_idx = []
    for seq, drug in zip(df2["Mutant_Sequence"].astype(str).str.upper(),
                         df2["Compound_Name"].values):
        h = seq_hash.get(seq, "MISSING")
        d = drug_map.get(str(drug).strip().upper(), "MISSING")
        feat_idx.append(onion_lookup.get(f"{d}_{h}", -1))
    feat_idx = np.array(feat_idx)
    valid = feat_idx >= 0
    log.info(f"  HIV-2 assay rows with ONION-Net features: {int(valid.sum())}"
             f"/{len(feat_idx)}")
    if int(valid.sum()) < 50:
        raise RuntimeError("Too few ONION-Net mapped rows — aborting hybrid stage")

    Xo = onion_feats[feat_idx[valid]]
    y_o = y2[valid]
    base_o = p2[valid]
    residuals = y_o - base_o
    rmse_base = float(np.sqrt((residuals ** 2).mean()))

    from sklearn.model_selection import KFold
    kf_o = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    resid_preds = np.zeros_like(residuals)
    for tr, te in kf_o.split(Xo):
        rf_r = RandomForestRegressor(**RF_PARAMS)
        rf_r.fit(Xo[tr], residuals[tr])
        resid_preds[te] = rf_r.predict(Xo[te])

    m_resid = metrics(residuals, resid_preds)
    pred_corr = base_o + resid_preds
    m_corr = metrics(y_o, pred_corr)
    var_expl = max(0.0, m_resid["pearson_r"]) ** 2 * 100.0
    manifest["hybrid_onionnet_stage"] = {
        "n_hiv2_docked_rows": int(valid.sum()),
        "onion_feature_dim": int(Xo.shape[1]),
        "residual_metrics": m_resid,
        "residual_variance_explained_pct": round(var_expl, 1),
        "baseline_rmse": round(rmse_base, 4),
        "corrected_metrics": m_corr,
        "note": ("Manuscript claimed residual R=0.667 / 44.5% variance and "
                 "hybrid R=0.734. On-disk honest values from the legacy "
                 "poisoned-corpus run: residual R=0.5153, variance 26.6%, "
                 "corrected R=0.6696, R2=0.448. This run: see numbers "
                 "above, genuine-only corpus."),
    }
    log.info(f"  RESIDUAL: R={m_resid['pearson_r']:.4f} "
             f"variance={var_expl:.1f}%")
    log.info(f"  CORRECTED: R={m_corr['pearson_r']:.4f} "
             f"R2={m_corr['r2']:.4f} RMSE={m_corr['rmse']:.4f}")
    pd.DataFrame({"Mutant_Sequence": df2["Mutant_Sequence"].values[valid],
                  "Drug": drugs2[valid], "dG_true": y_o,
                  "base_pred": base_o, "residual_pred": resid_preds,
                  "corrected_pred": pred_corr}).to_csv(
        os.path.join(RESCUE, "hybrid_predictions.csv"), index=False)

    # ---------------------------------------------------------------
    log.info("\n[6] HIVDB baseline audit (manuscript claimed R=0.722)")
    ss = pd.to_numeric(df1["Stanford_Score"], errors="coerce")
    ss_synth = pd.read_csv(SYNTH_CSV, low_memory=False)
    ss_synth = pd.to_numeric(ss_synth["Stanford_Score"], errors="coerce")
    # Forensic reconstruction: compute the correlation the manuscript used
    af_synth = pd.to_numeric(
        pd.read_csv(SYNTH_CSV, low_memory=False)["Binding_Affinity_kcal_mol"],
        errors="coerce")
    fm = ss_synth.notna() & af_synth.notna()
    r_syn, p_syn = pearsonr(ss_synth[fm], af_synth[fm])
    # Forensic: is p=1.5e-19 plausible for n=137, r=0.5044?
    from scipy.stats import t as tdist
    import math
    def p_for(n, r):
        t = r * math.sqrt((n - 2) / (1 - r * r))
        return float(2 * tdist.sf(abs(t), n - 2))
    baseline_audit = {
        "manuscript_claim": "R=0.722 (n=1500) — hardcoded in "
                            "_final_summary.py:15, n=1500 == synthetic rows",
        "genuine_corpus_stanford_score_nonnull": int(ss.notna().sum()),
        "forensic_reconstruction_on_synthetic_rows": {
            "n": int(fm.sum()),
            "pearson_r": round(float(r_syn), 4),
            "pearson_p": float(p_syn),
            "origin": ("Synthetic rows were generated by "
                       "expand_training_database.py with BANDED uniform noise: "
                       "score>=30 -> uniform(-6.5,-4.5), else uniform(-12.0,-9.5). "
                       "R=0.7216 is a self-fulfilling artifact of that banded "
                       "generator, NOT a measured relationship."),
        },
        "legacy_pvalue_forensic_check": {
            "claimed": "p=1.5e-19 for n=137, r=0.5044",
            "actual_p_for_n137_r05044": p_for(137, 0.5044),
            "n_required_for_p15e19": next(
                (n for n in range(138, 2000)
                 if p_for(n, 0.5044) < 1.5e-19), None),
            "verdict": ("p=1.5e-19 is IMPOSSIBLE for n=137, r=0.5044 "
                        "(requires n>=550). The legacy p-value was inflated."),
        },
        "synthetic_stanford_score_unique_values": sorted(
            ss_synth.dropna().unique().tolist()),
        "correlation_with_constant_undefined": True,
        "verdict": ("R=0.722 is UNREPRODUCIBLE as an honest baseline. "
                    "Stanford_Score is NULL in every genuine row. The n=1500 "
                    "value is a construction artifact of the synthetic "
                    "generator's banded noise, not real data."),
    }
    manifest["hivdb_baseline_audit"] = baseline_audit
    log.info(f"  genuine Stanford_Score non-null: {int(ss.notna().sum())}/"
             f"{len(df1)}")
    log.info(f"  FORENSIC: synthetic rows alone give R={r_syn:.4f} "
             f"(n={int(fm.sum())}) — banded-noise artifact")
    log.info(f"  FORENSIC: legacy p=1.5e-19 impossible for n=137,r=0.5044 "
             f"(real p={p_for(137, 0.5044):.2e})")
    log.info("  VERDICT: R=0.722 is UNREPRODUCIBLE (constant / NULL data)")

    # ---------------------------------------------------------------
    manifest["run_seconds"] = round(time.time() - t0, 1)
    out = os.path.join(RESCUE, "metrics_manifest.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    log.info(f"\n[DONE] Manifest written to {out}")
    log.info(f"       Elapsed {manifest['run_seconds']}s")
    return manifest


if __name__ == "__main__":
    main()
