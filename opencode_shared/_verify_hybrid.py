"""
Verify hybrid cross-species numbers matching manuscript parameters.
Stage 1: RF(600, max_depth=20) - biochem baseline
Stage 2: RF(600, max_depth=20) - ONION-Net residual correction
5-fold CV, seed=42
"""
import os, json, warnings, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from opencode_shared import WORKSPACE
from opencode_shared import extract_global_features as fb, compute_ic50_to_delta_g

warnings.filterwarnings('ignore')
np.random.seed(42)

W = WORKSPACE

# Load data
h1 = pd.read_csv(os.path.join(W, "known_info_master", "STANFORD_HIV1_EXPANDED.csv"), low_memory=False)
h2 = pd.read_csv(os.path.join(W, "known_info_master", "HIV2_PROTEASE_ML_READY_DATASET.csv"))

onion_data = np.load(os.path.join(W, "advanced_feature_extraction", "docked_complexes", "docked_onionnet_features.npz"),
                     allow_pickle=True)
onion_keys = list(onion_data['keys'])
onion_feats = onion_data['features']
log.info(f"ONION-Net features: {len(onion_keys)} vectors, dim={onion_feats.shape[1]}")

# Filter HIV-2 to valid assay types
VALID_ASSAY_TYPES = ['IC50', 'Ki', 'Kd']
h2 = h2[h2['Assay_Type'].str.upper().str.strip().isin(VALID_ASSAY_TYPES)].copy()
h2['_dg'] = h2['Standard_Value'].apply(
    lambda x: compute_ic50_to_delta_g(float(x)) if float(x) > 0 else float('nan'))
h2 = h2.dropna(subset=['_dg'])
h2 = h2[(h2['_dg'] >= -20) & (h2['_dg'] <= 0)].copy()

# Map to ONION-Net features
lookup = pd.read_csv(os.path.join(W, "advanced_feature_extraction", "esmfold_structures", "sequence_to_pdb.csv"))
seq_hash = dict(zip(lookup['Sequence'].str.strip().str.upper(), lookup['Hash']))
drug_name_map = {d.upper(): d for d in sorted([
    'AMPRENAVIR','ATAZANAVIR','DARUNAVIR','INDINAVIR','LOPINAVIR',
    'NELFINAVIR','RITONAVIR','SAQUINAVIR','TIPRANAVIR'])}
onion_lookup = {k: i for i, k in enumerate(onion_keys)}

h2_feat_idx = []
for seq, drug in zip(h2['Mutant_Sequence'].astype(str).str.strip().str.upper(),
                     h2['Compound_Name'].values):
    h = seq_hash.get(seq, 'MISSING')
    d = drug_name_map.get(str(drug).strip().upper(), 'MISSING')
    key = f"{d}_{h}"
    idx = onion_lookup.get(key, -1)
    h2_feat_idx.append(idx)

h2_feat_idx = np.array(h2_feat_idx)
valid_mask = h2_feat_idx >= 0
h2 = h2.iloc[valid_mask].copy()
h2_feat_idx = h2_feat_idx[valid_mask]
n = len(h2)
log.info(f"HIV-2 entries with ONION-Net features: {n}")

y_h2 = h2['_dg'].values.astype(np.float32)
X_h2_onion = onion_feats[h2_feat_idx]
X_h2_bio = np.array([fb(s) for s in h2['Mutant_Sequence'].values], dtype=np.float32)

# HIV-1
X_h1_bio = np.array([fb(s) for s in h1['Sequence'].values], dtype=np.float32)
y_h1 = h1['Binding_Affinity_kcal_mol'].values.astype(np.float32)
log.info(f"HIV-1 train: {len(y_h1)} | HIV-2 test: {n}")

# ── Experiment A: Baseline ──
log.info("\n" + "="*60)
log.info("EXPERIMENT A: Cross-species RF on biochem only")
rf_base = RandomForestRegressor(600, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=42)
rf_base.fit(X_h1_bio, y_h1)
pred_base = rf_base.predict(X_h2_bio)
r_base, _ = pearsonr(y_h2, pred_base)
residuals = y_h2 - pred_base
rmse_base = np.sqrt((residuals**2).mean())
mae_base = np.abs(residuals).mean()
log.info(f"  Baseline: R={r_base:.4f}, RMSE={rmse_base:.4f}, MAE={mae_base:.4f}")

# ── Experiment B: ONION-Net predicts residuals (5-fold CV) ──
log.info("\n" + "="*60)
log.info("EXPERIMENT B: HIV-2 ONION-Net -> residuals (5-fold CV)")

kf = KFold(5, shuffle=True, random_state=42)

# RF(600, max_depth=20) on residuals - MATCHING MANUSCRIPT
rf_resid = RandomForestRegressor(600, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=42)
resid_preds_rf = np.zeros_like(residuals)
for tr, te in kf.split(X_h2_onion):
    rf_resid.fit(X_h2_onion[tr], residuals[tr])
    resid_preds_rf[te] = rf_resid.predict(X_h2_onion[te])
r_resid_rf, _ = pearsonr(residuals, resid_preds_rf)
log.info(f"  ONION-Net RF -> residuals R: {r_resid_rf:.4f}")
log.info(f"  Residual variance explained: {r_resid_rf**2*100:.1f}%")

# Ridge for comparison
from sklearn.linear_model import Ridge
import logging
log = logging.getLogger(__name__)
ridge_resid = Ridge(alpha=1.0)
resid_preds_ridge = np.zeros_like(residuals)
for tr, te in kf.split(X_h2_onion):
    ridge_resid.fit(X_h2_onion[tr], residuals[tr])
    resid_preds_ridge[te] = ridge_resid.predict(X_h2_onion[te])
r_resid_ridge, _ = pearsonr(residuals, resid_preds_ridge)
log.info(f"  ONION-Net Ridge -> residuals R: {r_resid_ridge:.4f}")

# ── Experiment C: Corrected prediction ──
log.info("\n" + "="*60)
log.info("EXPERIMENT C: Hybrid corrected prediction")

pred_corrected_rf = pred_base + resid_preds_rf
r_corr_rf, _ = pearsonr(y_h2, pred_corrected_rf)
resid_corr_rf = y_h2 - pred_corrected_rf
rmse_corr_rf = np.sqrt((resid_corr_rf**2).mean())
dR_rf = r_corr_rf - r_base

pred_corrected_ridge = pred_base + resid_preds_ridge
r_corr_ridge, _ = pearsonr(y_h2, pred_corrected_ridge)
resid_corr_ridge = y_h2 - pred_corrected_ridge
rmse_corr_ridge = np.sqrt((resid_corr_ridge**2).mean())
dR_ridge = r_corr_ridge - r_base

log.info(f"  RF-corrected:  R={r_corr_rf:.4f}, RMSE={rmse_corr_rf:.4f}  (dR={dR_rf:+.4f})")
log.info(f"  Ridge-corrected: R={r_corr_ridge:.4f}, RMSE={rmse_corr_ridge:.4f}  (dR={dR_ridge:+.4f})")
log.info(f"  Baseline:      R={r_base:.4f}, RMSE={rmse_base:.4f}")

# ── Per-drug ──
log.info("\n" + "="*60)
log.info("PER-DRUG BREAKDOWN")
drugs = h2['Compound_Name'].str.strip().str.upper().unique()
for drug in sorted(drugs):
    mask = h2['Compound_Name'].str.strip().str.upper() == drug
    if mask.sum() < 4:
        continue
    r_d, _ = pearsonr(y_h2[mask], pred_base[mask])
    r_dc, _ = pearsonr(y_h2[mask], pred_corrected_rf[mask])
    log.info(f"  {drug:<12s} n={mask.sum():>3d}  baseline R={r_d:.4f}  corrected R={r_dc:.4f}  d={r_dc-r_d:+.4f}")

# ── Summary comparison ──
log.info("\n" + "="*60)
log.info("COMPARISON WITH SAVED JSON (old run) vs MANUSCRIPT")
log.info(f"  Metric                   | This Run   | Saved JSON | Manuscript")
log.info(f"  -------------------------|------------|------------|-----------")
log.info(f"  Baseline R               | {r_base:.4f}    | 0.5420     | 0.540")
log.info(f"  ONION-Net residual R     | {r_resid_rf:.4f}    | 0.5206     | 0.667")
log.info(f"  Hybrid RF-corrected R    | {r_corr_rf:.4f}    | 0.6708     | 0.734")
log.info(f"  Hybrid RMSE              | {rmse_corr_rf:.4f}   | 1.1759     | 1.392")
log.info()

# Note: seed sensitivity analysis
log.info("NOTE: RF has stochastic elements. Run 3x with seeds 42, 123, 999")
log.info("to estimate variance. Manuscript numbers likely from tuned version.\n")

# Save
results = {
    'baseline': {'r': round(float(r_base),4), 'rmse': round(float(rmse_base),4), 'mae': round(float(mae_base),4)},
    'onionnet_residual_rf_r': round(float(r_resid_rf),4),
    'onionnet_residual_ridge_r': round(float(r_resid_ridge),4),
    'residual_variance_explained_pct': round(max(0, r_resid_rf**2)*100, 1),
    'corrected_rf': {'r': round(float(r_corr_rf),4), 'rmse': round(float(rmse_corr_rf),4)},
    'corrected_ridge': {'r': round(float(r_corr_ridge),4), 'rmse': round(float(rmse_corr_ridge),4)},
}
out_dir = os.path.join(W, "final_paper", "03_results", "integration_results")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'hybrid_cross_species_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
log.info(f"Results saved to {os.path.join(out_dir, 'hybrid_cross_species_results.json')}")
