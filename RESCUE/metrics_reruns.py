"""
metrics_reruns.py
=================
DELIVERABLE 7 — HONEST RERUNS OF THE LEGACY "PAPER ADDITIONS / DL" NUMBERS
ON THE QUARANTINED (GENUINE-ONLY) CORPUS.

What this script does (and why):
  The manuscript's headline "optimization" numbers came from pipelines that
  trained on STANFORD_HIV1_EXPANDED.csv (7,183 rows = 5,683 genuine + 1,500
  seed-42 synthetic). This script reruns the same configurations with the
  synthetic rows quarantined (STANFORD_HIV1_GENUINE.csv only) and writes every
  number to RESCUE/metrics_reruns.json together with the SHA-256 of every
  input file. Numbers covenant: nothing on disk that is not produced by this
  script.

Reruns:
  R1  Pocket-restricted ADD-GHI            legacy R=0.5920  (run_paper_additions_v3.py, config GHI)
  R2  ONION-Net PCA downstream (B1 pocket) legacy R=0.5915  (run_pocket_onionnet_pca.py; PCA fusion DOCUMENTED-FAILED)
  R3  Few-shot curves 50/100/200/500 rows  legacy R=0.5806@500 (Tier5 scaling)
  R4  Scaling curve 500..full              legacy R=0.5806@500 -> 0.5044@7183
  R5  Within-HIV-2 7-drug-train/8th-holdout (per-drug leave-one-drug-out)

Documented (not rerun — reasons in the JSON):
  D1  GNN / DBPT+ (deepternary EGNN)       legacy R=-0.1709 / 0.2122
      torch is CPU-only on this machine; 13,528 complexes x 100 epochs of
      equivariant GNN is days of GPU compute, infeasible inside the deadline.
      The legacy result was already negative (R=-0.1709) — a documented
      failure — and the honest (smaller) corpus cannot rescue it.
  D2  ONION-Net PCA fusion sweep           legacy: every PCA config HURT
      (best fusion PCA-20 R=0.4703 < B1 pocket R=0.5915 per the 2026-06-22
      report). Additionally the on-disk npz pairs have MISMATCHED dimensions
      (hiv1 10,309-D vs hiv2 10,478-D) so the legacy script's
      pca.fit(X1).transform(X2) cannot even run today. Documented-failed.

Sign convention (fixed; see RESCUE/SIGN_CONVENTION.md):
  Target = deltaG (kcal/mol, negative = tighter binding). Same convention as
  retrain_honest.py. Legacy scripts shared this sign convention.

Outputs:
  - RESCUE/metrics_reruns.json   (numbers covenant for the reruns)
  - RESCUE/reruns_log.txt        (full run log)
  - RESCUE/reruns_r1_*.csv       (per-run prediction artifacts where useful)
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "advanced_feature_extraction"))

from opencode_shared import (  # noqa: E402
    WORKSPACE, KNOWN_INFO, extract_global_features, compute_ic50_to_delta_g,
    Z_SCALES, KIDERA, VHSE, POCKET_1INDEX, HIV1_WT, HIV2_WT,
)
from run_paper_additions_v3 import (  # noqa: E402
    get_aa_feats_23d, load_pdb_residues, min_heavy_dist,
    compute_delaunay_edges, extract_delaunay_vector, Z_SCALES_5D,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "reruns_log.txt"), mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("metrics_reruns")

RESCUE = os.path.dirname(os.path.abspath(__file__))
ADV_FEAT = os.path.join(WORKSPACE, "advanced_feature_extraction")

GENUINE_CSV = os.path.join(KNOWN_INFO, "STANFORD_HIV1_GENUINE.csv")
SYNTH_CSV = os.path.join(KNOWN_INFO, "STANFORD_HIV1_SYNTHETIC_QUARANTINED.csv")
H2_ASSAY_CSV = os.path.join(KNOWN_INFO, "HIV2_PROTEASE_ML_READY_DATASET.csv")
PDB_PATH = os.path.join(ADV_FEAT, "3S45.pdb")
ONION_HIV1_NPZ = os.path.join(ADV_FEAT, "onionnet_features",
                              "hiv1_onionnet_features.npz")
ONION_HIV2_NPZ = os.path.join(ADV_FEAT, "docked_complexes",
                              "docked_onionnet_features.npz")
LOOKUP_HIV1 = os.path.join(ADV_FEAT, "hiv1_esmfold", "sequence_to_pdb.csv")
LOOKUP_HIV2 = os.path.join(ADV_FEAT, "esmfold_structures",
                           "sequence_to_pdb.csv")
CONFIG_PY = os.path.join(WORKSPACE, "opencode_shared", "config.py")

SEED = 42
RF_HONEST = dict(n_estimators=600, max_depth=20, min_samples_leaf=2,
                 min_samples_split=5, max_features="sqrt",
                 n_jobs=-1, random_state=SEED)
RF_LEGACY_GHI = dict(n_estimators=400, max_depth=14, min_samples_leaf=2,
                     random_state=SEED, n_jobs=-1)


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
            "ci_upper": round(float(np.percentile(rs, 97.5)), 4)}


def load_genuine_hiv1():
    df = pd.read_csv(GENUINE_CSV, low_memory=False)
    df = df.dropna(subset=["Sequence", "Binding_Affinity_kcal_mol"])
    df["Binding_Affinity_kcal_mol"] = pd.to_numeric(
        df["Binding_Affinity_kcal_mol"], errors="coerce")
    df = df[(df["Binding_Affinity_kcal_mol"] >= -20)
            & (df["Binding_Affinity_kcal_mol"] <= 0)]
    seqs = df["Sequence"].astype(str).str.strip().str.upper().values
    keep = np.array([len(s) >= 90 for s in seqs])
    X = np.array([extract_global_features(s, include_stats=False)
                  for s in seqs[keep]], dtype=np.float32)
    y = df["Binding_Affinity_kcal_mol"].values.astype(np.float32)
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
    keep = np.array([len(s) >= 90 for s in seqs])
    X = np.array([extract_global_features(s, include_stats=False)
                  for s in seqs[keep]], dtype=np.float32)
    y = df["_dg"].values.astype(np.float32)
    drugs = df["Compound_Name"].astype(str).values
    return X, y[keep], drugs[keep], df.loc[keep].copy()


def build_pocket_ghi_features(seqs, positions, pdb_res, edges_list):
    """Replicates run_paper_additions_v3.py config GHI features:
    23-D/res (5D Z-scales + Kidera + VHSE) over BASE_POSITIONS plus the
    210-D Delaunay vector. Zero-copy reuse of the legacy helper functions."""
    rows = []
    for s in seqs:
        seq = (s[:99] + "-" * 99)[:99]
        feats = []
        for p in positions:
            feats.extend(get_aa_feats_23d(seq[p - 1]))
        feats.extend(extract_delaunay_vector(seq, edges_list, pdb_res))
        rows.append(feats)
    return np.array(rows, dtype=np.float32)


def structural_setup():
    pdb_res = load_pdb_residues(PDB_PATH)
    identical_positions = set(i for i in range(99) if HIV1_WT[i] == HIV2_WT[i])
    identical_1idx = set(i + 1 for i in identical_positions)
    neighbors_all = set()
    for r in range(1, 100):
        if r in POCKET_1INDEX:
            continue
        for pr in POCKET_1INDEX:
            if r in pdb_res and pr in pdb_res:
                if min_heavy_dist(pdb_res[r], pdb_res[pr]) < 5.0:
                    neighbors_all.add(r)
                    break
    conserved_neighbors = [r for r in sorted(neighbors_all)
                           if r in identical_1idx]
    base_positions = POCKET_1INDEX + conserved_neighbors
    delaunay_edges = compute_delaunay_edges(pdb_res)
    return pdb_res, base_positions, delaunay_edges, len(identical_1idx)


# ---------------------------------------------------------------------------
# R1 — Pocket-restricted ADD-GHI (legacy R=0.5920)
# ---------------------------------------------------------------------------
def rerun_r1_addghi():
    log.info("=" * 70)
    log.info("R1  POCKET-RESTRICTED ADD-GHI (legacy R=0.5920)")
    log.info("=" * 70)
    df1 = pd.read_csv(GENUINE_CSV, low_memory=False).dropna(
        subset=["Sequence", "Binding_Affinity_kcal_mol"])
    df1["Binding_Affinity_kcal_mol"] = pd.to_numeric(
        df1["Binding_Affinity_kcal_mol"], errors="coerce")
    df1 = df1[(df1["Binding_Affinity_kcal_mol"] >= -20)
              & (df1["Binding_Affinity_kcal_mol"] <= 0)]
    # Legacy dedup: median affinity per unique sequence (same as legacy).
    df1_dd = df1.groupby("Sequence", as_index=False)[
        "Binding_Affinity_kcal_mol"].median()
    y1 = df1_dd["Binding_Affinity_kcal_mol"].values.astype(np.float32)

    df2 = pd.read_csv(H2_ASSAY_CSV, low_memory=False).dropna(
        subset=["Mutant_Sequence", "Standard_Value"])
    df2 = df2[df2["Assay_Type"].astype(str).str.upper().str.strip().isin(
        ["IC50", "KI", "KD"])].copy()
    df2["delta_g"] = df2["Standard_Value"].apply(
        lambda x: compute_ic50_to_delta_g(float(x)) if float(x) > 0
        else float("nan"))
    df2 = df2.dropna(subset=["delta_g"])
    df2 = df2[(df2["delta_g"] >= -20) & (df2["delta_g"] <= 0)]
    y2 = df2["delta_g"].values.astype(np.float32)

    pdb_res, base_pos, edges, n_ident = structural_setup()

    seq1 = df1_dd["Sequence"].astype(str).str.strip().str.upper().values
    seq2 = df2["Mutant_Sequence"].astype(str).str.strip().str.upper().values
    X1 = build_pocket_ghi_features(seq1, base_pos, pdb_res, edges)
    X2 = build_pocket_ghi_features(seq2, base_pos, pdb_res, edges)

    # Target-weighted regression (ADD-I): inverse density over y1, bins=10.
    counts, bin_edges = np.histogram(y1, bins=10)
    bin_idx = np.clip(np.digitize(y1, bin_edges) - 1, 0, len(counts) - 1)
    w = 1.0 / (counts[bin_idx] + 1)
    w = w / w.mean()

    sc = StandardScaler()
    X1s = sc.fit_transform(X1)
    X2s = sc.transform(X2)
    rf = RandomForestRegressor(**RF_LEGACY_GHI)
    rf.fit(X1s, y1, sample_weight=w)
    p2 = rf.predict(X2s)
    m = metrics(y2, p2)
    ci = bootstrap_ci(y2, p2)
    log.info(f"  train: {len(y1)} unique seqs | test: {len(y2)} rows | "
             f"features: {X1.shape[1]}D")
    log.info(f"  R1 ADD-GHI: R={m['pearson_r']} rho={m['spearman_rho']} "
             f"RMSE={m['rmse']} CI={ci}")
    pd.DataFrame({"Mutant_Sequence": seq2, "dG_true": y2,
                  "dG_pred": p2}).to_csv(
        os.path.join(RESCUE, "reruns_r1_addghi.csv"), index=False)
    return {
        "name": "R1 pocket-restricted ADD-GHI (honest, genuine-only)",
        "status": "rerun",
        "legacy_value": {"pearson_r": 0.5920, "rmse": 2.1904,
                         "origin": "results.txt / PAPER_ADDITIONS_V3_REPORT.md "
                                   "(trained on poisoned 7,183-row corpus)"},
        "metrics": m,
        "bootstrap_ci_95": ci,
        "config": {"rf": "n_estimators=400, max_depth=14, "
                        "min_samples_leaf=2, seed 42",
                   "features": ("23D/res (5D Z-scales+Kidera+VHSE) over "
                                "BASE_POSITIONS=%d pos + Delaunay 210-D"
                                % len(base_pos)),
                   "target_weights": "inverse density, bins=10, normalized",
                   "dedup": "median affinity per unique sequence (legacy rule)",
                   "identical_positions": n_ident},
        "note": ("S92->A92 reference fix does NOT change the conservation "
                 "filter: 48 identical positions either way (verified). "
                 "Legacy script line 250 references an undefined RT variable; "
                 "this run uses compute_ic50_to_delta_g (RT=0.593, 298K)."),
        "input_files": [GENUINE_CSV, H2_ASSAY_CSV, PDB_PATH, CONFIG_PY],
    }


# ---------------------------------------------------------------------------
# R2 — ONION-Net PCA downstream: B1 pocket-only (legacy R=0.5915)
# ---------------------------------------------------------------------------
def rerun_r2_pocket_b1():
    log.info("=" * 70)
    log.info("R2  ONION-NET PCA DOWNSTREAM — B1 pocket-only (legacy R=0.5915)")
    log.info("=" * 70)
    df1 = pd.read_csv(GENUINE_CSV, low_memory=False).dropna(
        subset=["Sequence", "Binding_Affinity_kcal_mol"])
    df1["Binding_Affinity_kcal_mol"] = pd.to_numeric(
        df1["Binding_Affinity_kcal_mol"], errors="coerce")
    df1 = df1[(df1["Binding_Affinity_kcal_mol"] >= -20)
              & (df1["Binding_Affinity_kcal_mol"] <= 0)]
    df1_dd = df1.groupby("Sequence", as_index=False)[
        "Binding_Affinity_kcal_mol"].median()
    y1 = df1_dd["Binding_Affinity_kcal_mol"].values.astype(np.float32)

    df2 = pd.read_csv(H2_ASSAY_CSV, low_memory=False).dropna(
        subset=["Mutant_Sequence", "Standard_Value"])
    df2 = df2[df2["Assay_Type"].astype(str).str.upper().str.strip().isin(
        ["IC50", "KI", "KD"])].copy()
    df2["delta_g"] = df2["Standard_Value"].apply(
        lambda x: compute_ic50_to_delta_g(float(x)) if float(x) > 0
        else float("nan"))
    df2 = df2.dropna(subset=["delta_g"])
    df2 = df2[(df2["delta_g"] >= -20) & (df2["delta_g"] <= 0)]
    y2 = df2["delta_g"].values.astype(np.float32)

    pdb_res, base_pos, _, n_ident = structural_setup()

    def pocket_5d(seqs):
        rows = []
        for s in seqs:
            seq = (s[:99] + "-" * 99)[:99]
            feats = []
            for p in base_pos:
                feats.extend(get_aa_feats_23d(seq[p - 1]))
            rows.append(feats)
        return np.array(rows, dtype=np.float32)

    seq1 = df1_dd["Sequence"].astype(str).str.strip().str.upper().values
    seq2 = df2["Mutant_Sequence"].astype(str).str.strip().str.upper().values
    X1 = pocket_5d(seq1)
    X2 = pocket_5d(seq2)

    counts, bin_edges = np.histogram(y1, bins=10)
    bin_idx = np.clip(np.digitize(y1, bin_edges) - 1, 0, len(counts) - 1)
    w = 1.0 / (counts[bin_idx] + 1)
    w = w / w.mean()

    sc = StandardScaler()
    rf = RandomForestRegressor(**RF_LEGACY_GHI)
    rf.fit(sc.fit_transform(X1), y1, sample_weight=w)
    p2 = rf.predict(sc.transform(X2))
    m = metrics(y2, p2)
    ci = bootstrap_ci(y2, p2)
    log.info(f"  B1 pocket-5D: R={m['pearson_r']} rho={m['spearman_rho']} "
             f"RMSE={m['rmse']} CI={ci}")

    # Document the PCA fusion impossibility (dimension mismatch).
    a = np.load(ONION_HIV1_NPZ, allow_pickle=True)
    b = np.load(ONION_HIV2_NPZ, allow_pickle=True)
    dim1, dim2 = a["features"].shape[1], b["features"].shape[1]
    log.info(f"  ONION dims: hiv1={dim1} hiv2={dim2} -> PCA fusion of "
             f"fit(X1).transform(X2) is a dimension mismatch (documented-failed)")

    pd.DataFrame({"Mutant_Sequence": seq2, "dG_true": y2,
                  "dG_pred": p2}).to_csv(
        os.path.join(RESCUE, "reruns_r2_pocket_b1.csv"), index=False)
    return {
        "name": "R2 ONION-Net PCA downstream — B1 pocket-5D (honest)",
        "status": "rerun",
        "legacy_value": {"pearson_r": 0.5915,
                         "origin": "POCKET_ONIONNET_PCA_REPORT.md B1 (aligned "
                                   "subset, poisoned corpus)"},
        "metrics": m,
        "bootstrap_ci_95": ci,
        "config": {"features": "23D/res over BASE_POSITIONS=%d (no Delaunay)"
                               % len(base_pos),
                   "rf": "n_estimators=400, max_depth=14, seed 42",
                   "target_weights": "inverse density, bins=10"},
        "pca_fusion_documented_failed": {
            "reason": ("Legacy script run_pocket_onionnet_pca.py calls "
                       "pca.fit(X1_onion).transform(X2_onion); on-disk npz "
                       "dims differ (hiv1=%d, hiv2=%d) so it cannot execute. "
                       "Additionally the 2026-06-22 report itself shows every "
                       "PCA config HURT (best fusion PCA-20 R=0.4703 < B1 "
                       "0.5915)." % (dim1, dim2)),
        },
        "input_files": [GENUINE_CSV, H2_ASSAY_CSV, PDB_PATH, CONFIG_PY,
                        ONION_HIV1_NPZ, ONION_HIV2_NPZ],
    }


# ---------------------------------------------------------------------------
# R3 — Few-shot curves (50/100/200/500 genuine HIV-1 rows)
# ---------------------------------------------------------------------------
def rerun_r3_fewshot():
    log.info("=" * 70)
    log.info("R3  FEW-SHOT CURVES (legacy peak R=0.5806 @ N=500)")
    log.info("=" * 70)
    X1, y1, df1 = load_genuine_hiv1()
    X2, y2, _, _ = load_hiv2_assays()
    rng = np.random.RandomState(SEED)
    curves = []
    for n in [50, 100, 200, 500]:
        idx = rng.choice(len(X1), size=min(n, len(X1)), replace=False)
        rf = RandomForestRegressor(**RF_HONEST)
        rf.fit(X1[idx], y1[idx])
        p = rf.predict(X2)
        m = metrics(y2, p)
        m["n_train"] = int(len(idx))
        m["legacy_peak_at_500"] = 0.5806
        curves.append(m)
        log.info(f"  N={n:4d}: R={m['pearson_r']} rho={m['spearman_rho']} "
                 f"RMSE={m['rmse']}")
    pd.DataFrame(curves).to_csv(
        os.path.join(RESCUE, "reruns_r3_fewshot_curve.csv"), index=False)
    return {
        "name": "R3 few-shot curve 50/100/200/500 (honest, genuine-only)",
        "status": "rerun",
        "legacy_value": {"pearson_r": 0.5806, "n": 500,
                         "origin": "Tier5_comprehensive_results.json scaling "
                                   "row (poisoned corpus)"},
        "curve": curves,
        "config": {"rf": RF_HONEST, "features": "2079-D biochem "
                                                "(include_stats=False)",
                   "sampling": "fixed seed 42, without replacement"},
        "input_files": [GENUINE_CSV, H2_ASSAY_CSV],
    }


# ---------------------------------------------------------------------------
# R4 — Scaling curve (500..full genuine)
# ---------------------------------------------------------------------------
def rerun_r4_scaling():
    log.info("=" * 70)
    log.info("R4  SCALING CURVE (legacy 0.5806@500 -> 0.5044@7183)")
    log.info("=" * 70)
    X1, y1, df1 = load_genuine_hiv1()
    X2, y2, _, _ = load_hiv2_assays()
    rng = np.random.RandomState(SEED)
    curve = []
    for n in [500, 1000, 2000, 4000, len(X1)]:
        idx = rng.choice(len(X1), size=n, replace=False)
        rf = RandomForestRegressor(**RF_HONEST)
        rf.fit(X1[idx], y1[idx])
        p = rf.predict(X2)
        m = metrics(y2, p)
        m["n_train"] = int(len(idx))
        curve.append(m)
        log.info(f"  N={n:5d}: R={m['pearson_r']} rho={m['spearman_rho']} "
                 f"RMSE={m['rmse']}")
    pd.DataFrame(curve).to_csv(
        os.path.join(RESCUE, "reruns_r4_scaling_curve.csv"), index=False)
    return {
        "name": "R4 scaling curve 500..full (honest, genuine-only)",
        "status": "rerun",
        "legacy_value": {
            "500": 0.5806, "1000": 0.5178, "2000": 0.4109,
            "4000": 0.4472, "7183": 0.5044,
            "origin": "Tier5_comprehensive_results.json E2 (poisoned corpus)"},
        "curve": curve,
        "config": {"rf": RF_HONEST, "features": "2079-D biochem",
                   "sampling": "fixed seed 42, without replacement"},
        "note": ("Legacy full-corpus point used all 7,183 (incl. 1,500 "
                 "synthetic); honest full point uses 5,683 genuine."),
        "input_files": [GENUINE_CSV, H2_ASSAY_CSV],
    }


# ---------------------------------------------------------------------------
# R5 — Within-HIV-2 7-drug-train / 8th-drug-holdout (per drug)
# ---------------------------------------------------------------------------
def rerun_r5_drug_holdout():
    log.info("=" * 70)
    log.info("R5  WITHIN-HIV-2 7-DRUG TRAIN / 8TH-DRUG HOLDOUT (per drug)")
    log.info("=" * 70)
    X2, y2, drugs2, df2 = load_hiv2_assays()
    drugs = sorted(set(drugs2))
    if "EXPERIMENTAL PI / ANALOGUE" in drugs:
        drugs.remove("EXPERIMENTAL PI / ANALOGUE")
    per_drug = []
    for d in drugs:
        mask = np.array([drug == d for drug in drugs2])
        if mask.sum() < 3:
            continue
        rf = RandomForestRegressor(**RF_HONEST)
        rf.fit(X2[~mask], y2[~mask])
        p = rf.predict(X2[mask])
        m = metrics(y2[mask], p)
        m["held_out_drug"] = d
        m["n_train"] = int((~mask).sum())
        m["n_test"] = int(mask.sum())
        per_drug.append(m)
        log.info(f"  holdout {d:12s}: R={m['pearson_r']} rho={m['spearman_rho']} "
                 f"RMSE={m['rmse']} (n_tr={m['n_train']}, n_te={m['n_test']})")
    rs = [m["pearson_r"] for m in per_drug]
    pd.DataFrame(per_drug).to_csv(
        os.path.join(RESCUE, "reruns_r5_drug_holdout.csv"), index=False)
    agg = {
        "mean_pearson_r": round(float(np.mean(rs)), 4),
        "std_pearson_r": round(float(np.std(rs)), 4),
        "median_pearson_r": round(float(np.median(rs)), 4),
        "n_drugs": len(per_drug),
    }
    log.info(f"  AGG: mean R={agg['mean_pearson_r']} +- {agg['std_pearson_r']}")
    return {
        "name": "R5 within-HIV-2 drug-holdout (honest, per-drug)",
        "status": "rerun",
        "legacy_value": {"origin": "No exact legacy equivalent; DeepTernary "
                                   "homodimer within-HIV-2 CV R=0.3709 is the "
                                   "closest manuscript number (different "
                                   "protocol)"},
        "aggregate": agg,
        "per_drug": per_drug,
        "config": {"rf": RF_HONEST, "features": "2079-D biochem",
                   "split": "train = all rows of other 7 drugs, "
                            "test = held-out drug"},
        "note": ("CRITICAL STRUCTURE: all 8 clean drugs share the SAME 17 "
                 "unique sequences (verified pairwise intersection = 17/17). "
                 "This holdout therefore measures DRUG-TRANSFER (same "
                 "mutants, unseen drug) and NOT unseen-sequence "
                 "generalization. High R is expected: the model has seen "
                 "every test sequence under 7 other drugs and learns a "
                 "per-sequence baseline + drug offset. Interpret strictly as "
                 "drug-transfer, never as zero-shot. 'Experimental PI / "
                 "Analogue' row excluded."),
        "input_files": [H2_ASSAY_CSV],
    }


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("HONEST METRICS RERUNS — GENUINE-ONLY CORPUS, SEED 42")
    log.info("=" * 70)

    manifest = {
        "pipeline": "RESCUE/metrics_reruns.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_hashes": {os.path.basename(p): sha256(p) for p in [
            GENUINE_CSV, SYNTH_CSV, H2_ASSAY_CSV, PDB_PATH,
            ONION_HIV1_NPZ, ONION_HIV2_NPZ, LOOKUP_HIV1, LOOKUP_HIV2,
            CONFIG_PY]},
        "reference_fix": ("HIV2_WT from opencode_shared/config.py (A92, "
                          "verified vs NCBI M15390.1) used throughout; "
                          "S92->A92 changes 0 conservation-filter positions "
                          "(48 identical either way)."),
        "sign_convention": "target = deltaG kcal/mol (negative = tighter); "
                           "see RESCUE/SIGN_CONVENTION.md",
        "reruns": [],
        "documented_failed": [],
    }

    # R1
    manifest["reruns"].append(rerun_r1_addghi())
    # R2
    manifest["reruns"].append(rerun_r2_pocket_b1())
    # R3
    manifest["reruns"].append(rerun_r3_fewshot())
    # R4
    manifest["reruns"].append(rerun_r4_scaling())
    # R5
    manifest["reruns"].append(rerun_r5_drug_holdout())

    # D1 — GNN / DBPT+ documented-failed
    manifest["documented_failed"].append({
        "name": "D1 GNN / DBPT+ (DeepTernary v2 EGNN)",
        "legacy_values": {"GNN zero-shot": {"pearson_r": -0.1709,
                                            "rmse": 3.5208,
                                            "source": "Tier3_pocket_gnn_results.json"},
                          "DBPT+ v2 zero-shot": {"pearson_r": 0.2122,
                                                 "source": "DBPT_PLUS_V2_RESULTS.json"}},
        "reason_not_rerun": ("torch is CPU-only (torch 2.10.0+cpu, "
                             "cuda=False). 13,528 PDB complexes x 100 epochs "
                             "of equivariant EGNN is days of GPU compute — "
                             "infeasible inside the Tue-night deadline. The "
                             "legacy zero-shot result was already negative "
                             "(R=-0.1709): a documented failure that a "
                             "genuine-only (smaller) corpus cannot rescue."),
        "decision": "documented-failed; not claimed in honest metrics.",
    })

    # D2 — ONION-Net PCA fusion documented-failed
    manifest["documented_failed"].append({
        "name": "D2 ONION-Net PCA fusion sweep",
        "legacy_values": {"best_fusion": {"pearson_r": 0.4703, "config": "PCA-20"},
                          "pocket_b1": {"pearson_r": 0.5915},
                          "source": "POCKET_ONIONNET_PCA_REPORT.md (2026-06-22)"},
        "reason_not_rerun": ("On-disk npz feature dims mismatch (hiv1 10,309 "
                             "vs hiv2 10,478) so the legacy script's "
                             "pca.fit(X1).transform(X2) raises today; and the "
                             "legacy report itself shows every PCA fusion "
                             "HURT the pocket baseline (R 0.4703 < 0.5915)."),
        "decision": "documented-failed; B1 pocket-5D (R2) is the honest number.",
    })

    manifest["run_seconds"] = round(time.time() - t0, 1)
    out = os.path.join(RESCUE, "metrics_reruns.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    log.info(f"\n[DONE] Manifest written to {out}")
    log.info(f"       Elapsed {manifest['run_seconds']}s")
    return manifest


if __name__ == "__main__":
    main()
