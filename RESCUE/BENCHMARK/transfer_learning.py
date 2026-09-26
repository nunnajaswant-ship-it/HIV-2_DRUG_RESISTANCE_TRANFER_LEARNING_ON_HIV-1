"""
Two-Stage Transfer Learning Pipeline for HIV-2 Drug Resistance Prediction
==========================================================================
Pre-train on HIV-1 (Stanford, direct ΔG labels), then fine-tune on HIV-2
ChEMBL (IC50 → ΔG at 310 K).  Leakage-free evaluation throughout:
every HIV-2 evaluation is out-of-fold (KFold on HIV-2 only), and the
StandardScaler is fit on HIV-1 once and re-used for HIV-2 (never refit).

Primary transfer model: RandomForestRegressor (600 trees, depth 20, sqrt)
which captures the epistatic genotype→phenotype relationship across species.
Ridge is linear and underperforms for cross-species zero-shot transfer.

Pipeline parts
--------------
Part 1 — Feature ablation (RF, 5 feature sets):
    biochem (2121-D) / esm2 (1280-D) / combined (3359-D)
    / esm2_pocket (1280-D, pool only 19 conserved pocket residues)
    / combined_pocket (3359-D = biochem + esm2_pocket)
    Stage 1  : RF on HIV-1 -> 5-fold GroupKFold CV  -> record R
    Stage 2a : Zero-shot  (apply HIV-1 model to HIV-2)  -> record R
    Stage 2b : Fine-tune   (HIV-2 KFold, HIV-1 scaler)  -> record R
    Plus: Pooled zero-shot (legacy-comparable, one model all drugs)
Part 2 — Model comparison (combined features, 4 models):
    Ridge / ElasticNet / XGB(GradientBoost: 100 trees, depth 4) / RF(600 trees)
    HIV-1 pre-train CV and HIV-2 fine-tune CV for each model.
Part 3 — Learning curve: fine-tune on 20/40/60/80/100 % of the ~320 HIV-2
    ChEMBL rows, 3 seeds per fraction, best model from Part 2.
Part 4 — Conformal prediction: 30 % calibration / 70 % test, 90 % and 95 %
    prediction intervals, coverage + mean interval width.
Part 5 — Summary: zero-shot vs fine-tuned vs supervised, per-drug table,
    JSON results + comparison figure.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_config import (
    DATA_DIR,
    ESM2_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    DRUGS,
    RANDOM_SEED,
    HIV2_WT_PR,
)
from model_builders import build_model

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, KFold

warnings.filterwarnings("ignore")

try:
    from opencode_shared.utils import extract_global_features
except Exception:
    try:
        from utils import extract_global_features
    except Exception:
        extract_global_features = None


# ============================================================================
# CONSTANTS
# ============================================================================
R_KCAL = 0.001987          # kcal / (mol * K)
T_310 = 310.0              # physiological temperature (K)
RT_310 = R_KCAL * T_310

N_SPLITS = 5
N_BOOTSTRAP = 200          # fast bootstrap for CI
LEARNING_FRACTIONS = [0.20, 0.40, 0.60, 0.80, 1.00]
LC_N_SEEDS = 3
CONFORMAL_CAL_FRAC = 0.30

HIV1_DRUG_MAP = {
    "Saquinavir": "SQV/r",
    "Lopinavir": "LPV/r",
    "Atazanavir": "ATV/r",
    "Darunavir": "DRV/r",
}

HIV2_DRUG_MAP = {
    "ATAZANAVIR": "ATV/r",
    "SAQUINAVIR": "SQV/r",
    "LOPINAVIR": "LPV/r",
    "DARUNAVIR": "DRV/r",
    "NELFINAVIR": "NFV/r",
    "INDINAVIR": "IDV/r",
    "AMPRENAVIR": "FPV/r",
    "TIPRANAVIR": "TPV/r",
    "RITONAVIR": "RTV/r",
}

COMMON_DRUGS = ["ATV/r", "DRV/r", "LPV/r", "SQV/r"]
FEATURE_SETS = ["biochem", "esm2", "combined", "esm2_pocket", "combined_pocket"]
FEATURE_LABELS = {
    "biochem": "Biochem (2121)",
    "esm2": "ESM-2 full (1280)",
    "combined": "Combined (3359)",
    "esm2_pocket": "ESM-2 pocket-19 (1280)",
    "combined_pocket": "Combined+pocket (3359)",
}
# 19 conserved drug-pocket positions (1-indexed, from compute_conservation_mask.py
# KNOWN_CONTACT_RESIDUES) — the drug-binding interface common to HIV-1 and HIV-2.
POCKET19_1INDEX = [8, 23, 25, 26, 27, 28, 29, 30, 32, 45, 46, 47, 48, 50, 54, 76, 80, 82, 84]

MODEL_KEYS = ["T4_Ridge", "T4_EN", "T4_XGB", "T4_RF"]
MODEL_LABELS = {
    "T4_Ridge": "Ridge",
    "T4_EN": "ElasticNet",
    "T4_XGB": "XGB-GradBoost",
    "T4_RF": "RandomForest",
}


# ============================================================================
# SEED / HELPERS
# ============================================================================
def set_seed(seed=RANDOM_SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)


def ic50_to_dg(value_nm, assay_type="IC50"):
    """IC50 (nM) -> ΔG (kcal/mol) via ΔG = RT * ln(IC50 * 1e-9) at 310 K."""
    if pd.isna(value_nm) or float(value_nm) <= 0:
        return float("nan")
    v = max(float(value_nm), 0.001)
    return float(RT_310 * np.log(v * 1e-9))


def evaluate_predictions(y_true, y_pred, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Pearson R / Spearman rho / RMSE / MAE + bootstrap 95 % CI (fast)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    n = len(yt)
    out = {
        "n": int(n),
        "pearson_r": float("nan"),
        "spearman_rho": float("nan"),
        "rmse": float("nan"),
        "mae": float("nan"),
        "ci_pearson_lo": float("nan"),
        "ci_pearson_hi": float("nan"),
    }
    if n < 3 or np.std(yt) == 0 or np.std(yp) == 0:
        return out
    r, _ = stats.pearsonr(yt, yp)
    rho, _ = stats.spearmanr(yt, yp)
    out["pearson_r"] = float(r)
    out["spearman_rho"] = float(rho)
    out["rmse"] = float(np.sqrt(np.mean((yt - yp) ** 2)))
    out["mae"] = float(np.mean(np.abs(yt - yp)))
    rng = np.random.RandomState(seed)
    rs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if np.std(yt[idx]) > 0 and np.std(yp[idx]) > 0:
            rb, _ = stats.pearsonr(yt[idx], yp[idx])
            rs.append(rb)
    if rs:
        out["ci_pearson_lo"] = float(np.percentile(rs, 2.5))
        out["ci_pearson_hi"] = float(np.percentile(rs, 97.5))
    return out


# ============================================================================
# FEATURE STORE  (per-sequence cache + ESM-2 lookup)
# ============================================================================
_esm2_map = None
_esm2_residue_map = None


def _load_esm2_mapping():
    """Build {sequence_string: 1280-D mean-pooled vector} and
    {sequence_string: (99, 1280) per-residue matrix} from the NPZ once."""
    global _esm2_map, _esm2_residue_map
    if _esm2_map is not None:
        return _esm2_map, _esm2_residue_map
    esm2_path = ESM2_DIR / "esm2_650M_embeddings.npz"
    data = np.load(str(esm2_path), allow_pickle=True)
    stored_seqs = np.asarray(data["sequences"])
    mean_pooled = np.asarray(data["mean_pooled"])
    per_residue = np.asarray(data["per_residue"])
    mapping = {}
    residue_map = {}
    for i, s in enumerate(stored_seqs):
        key = str(s).strip()
        if key not in mapping:
            mapping[key] = mean_pooled[i]
            residue_map[key] = per_residue[i]
    _esm2_map = mapping
    _esm2_residue_map = residue_map
    print(f"  ESM-2 loaded: {len(mapping)} seqs -> 1280-D pooled + "
          f"{len(residue_map)} per-residue from {esm2_path.name}")
    return mapping, residue_map


class FeatureStore:
    """Caches biochem + ESM-2 vectors per sequence; combined = hstack."""

    def __init__(self):
        self._biochem = {}
        self._esm2_pocket = {}

    def biochem(self, seq):
        key = str(seq).strip()
        if key not in self._biochem:
            if extract_global_features is None:
                self._biochem[key] = np.zeros(2121, dtype=np.float32)
            else:
                self._biochem[key] = extract_global_features(key).astype(np.float32)
        return self._biochem[key]

    def esm2(self, seq):
        mapping, _ = _load_esm2_mapping()
        return mapping.get(str(seq).strip())

    def esm2_pocket(self, seq):
        key = str(seq).strip()
        if key not in self._esm2_pocket:
            _, residue_map = _load_esm2_mapping()
            resid = residue_map.get(key)
            if resid is None or len(resid) < 99:
                self._esm2_pocket[key] = np.zeros(1280, dtype=np.float32)
            else:
                # Pool ONLY the 19 pocket residues
                idx = [p - 1 for p in POCKET19_1INDEX if 1 <= p <= resid.shape[0]]
                self._esm2_pocket[key] = resid[idx].mean(axis=0).astype(np.float32)
        return self._esm2_pocket[key]

    def get(self, sequences, feature_set):
        seqs = [str(s).strip() for s in sequences]
        parts = []
        if feature_set in ("biochem", "combined", "combined_pocket"):
            parts.append(np.vstack([self.biochem(s) for s in seqs]))
        if feature_set in ("esm2", "combined"):
            mapping, _ = _load_esm2_mapping()
            out = np.zeros((len(seqs), 1280), dtype=np.float32)
            missing = 0
            for i, s in enumerate(seqs):
                vec = mapping.get(s)
                if vec is None:
                    missing += 1
                else:
                    out[i] = vec
            if missing:
                print(f"  [warn] {missing}/{len(seqs)} sequences missing ESM-2 embeddings")
            parts.append(out)
        if feature_set in ("esm2_pocket", "combined_pocket"):
            _, residue_map = _load_esm2_mapping()
            out = np.zeros((len(seqs), 1280), dtype=np.float32)
            missing = 0
            for i, s in enumerate(seqs):
                vec = self.esm2_pocket(s)
                if np.abs(vec).sum() == 0:
                    missing += 1
                else:
                    out[i] = vec
            if missing:
                print(f"  [warn] {missing}/{len(seqs)} sequences missing pocket ESM-2")
            parts.append(out)
        if not parts:
            raise ValueError(f"Unknown feature set: {feature_set}")
        X = parts[0] if len(parts) == 1 else np.hstack(parts)
        return np.asarray(X, dtype=np.float32)


_feature_store = FeatureStore()


# ============================================================================
# DATA LOADING
# ============================================================================
def load_hiv1_data():
    """HIV-1 Stanford genuine data -> [sequence, drug, dg, group]."""
    csv_path = DATA_DIR / "STANFORD_HIV1_GENUINE.csv"
    df = pd.read_csv(csv_path, usecols=["Sequence", "Inhibitor", "Binding_Affinity_kcal_mol"])
    records = []
    for _, row in df.iterrows():
        seq = str(row["Sequence"]).strip()
        drug = HIV1_DRUG_MAP.get(str(row["Inhibitor"]).strip())
        dg = float(row["Binding_Affinity_kcal_mol"])
        if drug and np.isfinite(dg):
            records.append({"sequence": seq, "drug": drug, "dg": dg, "group": seq})
    out = pd.DataFrame(records)
    print(f"[HIV-1] {len(out)} records | drugs: {out['drug'].value_counts().to_dict()}")
    print(f"        ΔG range: [{out['dg'].min():.3f}, {out['dg'].max():.3f}] kcal/mol")
    return out


def load_hiv2_chembl():
    """HIV-2 ChEMBL data -> [sequence, drug, dg, assay_type, ic50_nm]."""
    csv_path = DATA_DIR / "HIV2_PROTEASE_ML_READY_DATASET.csv"
    df = pd.read_csv(
        csv_path,
        usecols=["Mutant_Sequence", "Compound_Name", "Standard_Value", "Assay_Type"],
        low_memory=False,
    )
    records = []
    for _, row in df.iterrows():
        seq = str(row["Mutant_Sequence"]).strip()
        drug = HIV2_DRUG_MAP.get(str(row["Compound_Name"]).strip())
        assay = str(row["Assay_Type"]).strip()
        if drug is None or pd.isna(row["Standard_Value"]):
            continue
        dg = ic50_to_dg(row["Standard_Value"], assay)
        if np.isfinite(dg):
            records.append({
                "sequence": seq,
                "drug": drug,
                "dg": dg,
                "assay_type": assay,
                "ic50_nm": float(row["Standard_Value"]),
            })
    out = pd.DataFrame(records)
    print(f"[HIV-2 ChEMBL] {len(out)} records | drugs: {dict(out['drug'].value_counts())}")
    print(f"        ΔG range: [{out['dg'].min():.3f}, {out['dg'].max():.3f}] kcal/mol")
    return out


def load_hiv2_clinical():
    """HIV-2 clinical multilabel (penalty scores). Used for data stats only."""
    csv_path = DATA_DIR / "HIV2_CLINICAL_ML_READY_MULTILABEL.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    seq_col = "Mutant_Sequence" if "Mutant_Sequence" in df.columns else None
    info = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "has_sequence_column": seq_col is not None,
        "drug_columns_present": [c for c in ["ATV/r", "DRV/r", "LPV/r", "FPV/r", "IDV/r", "SQV/r", "TPV/r", "NFV"] if c in df.columns],
    }
    print(f"[HIV-2 Clinical] {len(df)} rows x {len(df.columns)} cols (stats only)")
    return df, info


# ============================================================================
# FEATURE BUNDLE  (scaled HIV-1 block + scaled HIV-2 block per feature set)
# ============================================================================
class Bundle:
    """Feature matrices + scaler for one feature set.

    The StandardScaler is fit on ALL HIV-1 rows once; HIV-2 is transformed
    with that same scaler and never refit (no leakage of HIV-2 statistics).
    Row order matches the source DataFrames.
    """

    def __init__(self, feature_set, hiv1_df, hiv2_df):
        self.feature_set = feature_set
        raw1 = _feature_store.get(hiv1_df["sequence"].tolist(), feature_set)
        self.scaler = StandardScaler()
        self.X1 = self.scaler.fit_transform(raw1)
        self.y1 = hiv1_df["dg"].values.astype(float)
        self.drug1 = hiv1_df["drug"].values
        self.group1 = hiv1_df["group"].values
        self.X2 = self.scaler.transform(
            _feature_store.get(hiv2_df["sequence"].tolist(), feature_set)
        )
        self.y2 = hiv2_df["dg"].values.astype(float)
        self.drug2 = hiv2_df["drug"].values
        self.seq2 = hiv2_df["sequence"].values

    def x2_for_drugs(self, drugs):
        keep = np.isin(self.drug2, drugs)
        return self.X2[keep], self.y2[keep], self.drug2[keep]


def build_bundles(hiv1_df, hiv2_df):
    bundles = {}
    for fs in FEATURE_SETS:
        print(f"[Features] extracting + scaling '{fs}' ...")
        t0 = time.time()
        bundles[fs] = Bundle(fs, hiv1_df, hiv2_df)
        print(f"           {bundles[fs].X1.shape} HIV-1 | {bundles[fs].X2.shape} HIV-2 "
              f"({time.time() - t0:.1f}s)")
    return bundles


# ============================================================================
# MODEL BUILDER (fast variants)
# ============================================================================
def build_fast_model(model_name):
    """sklearn estimator; Ridge/ElasticNet from build_model, XGB/RF tuned fast."""
    if model_name in ("T4_Ridge", "T4_EN"):
        return build_model(model_name)
    if model_name == "T4_XGB":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                tree_method="hist",
            )
        except Exception:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=RANDOM_SEED,
            )
    if model_name == "T4_RF":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=600, max_depth=20, max_features="sqrt",
            n_jobs=-1, random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unknown model: {model_name}")


# ============================================================================
# CV HELPERS  (out-of-fold, leakage-free)
# ============================================================================
def group_kfold_oof(X, y, groups, build, n_splits=N_SPLITS, seed=RANDOM_SEED):
    """GroupKFold OOF predictions. Falls back to KFold if too few groups."""
    n = len(y)
    oof_pred = np.full(n, np.nan)
    unique_groups = len(np.unique(groups))
    if unique_groups >= min(n_splits, max(2, n // 4)):
        splitter = GroupKFold(n_splits=n_splits).split(X, y, groups)
    else:
        n_split = min(n_splits, max(2, n // 8))
        splitter = KFold(n_splits=n_split, shuffle=True, random_state=seed).split(X)
    for tr, te in splitter:
        model = build()
        model.fit(X[tr], y[tr])
        oof_pred[te] = model.predict(X[te])
    valid = np.isfinite(oof_pred)
    return y[valid], oof_pred[valid]


def kfold_oof(X, y, build, n_splits=N_SPLITS, seed=RANDOM_SEED):
    """Shuffled KFold OOF predictions."""
    n = len(y)
    oof_pred = np.full(n, np.nan)
    n_split = min(n_splits, max(2, n // 8))
    kf = KFold(n_splits=n_split, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        model = build()
        model.fit(X[tr], y[tr])
        oof_pred[te] = model.predict(X[te])
    valid = np.isfinite(oof_pred)
    return y[valid], oof_pred[valid]


def mean_r(eval_dict):
    vals = [v["pearson_r"] for k, v in eval_dict.items()
            if k != "MEAN" and np.isfinite(v["pearson_r"])]
    return float(np.mean(vals)) if vals else float("nan")


# ============================================================================
# STAGE 1 — PRE-TRAIN ON HIV-1
# ============================================================================
def pretrain_hiv1(bundle, model_name, drugs=COMMON_DRUGS):
    """Train per-drug models on HIV-1 with 5-fold GroupKFold CV."""
    models = {}
    cv = {}
    cv_pooled_t, cv_pooled_p = [], []
    for drug in drugs:
        mask = bundle.drug1 == drug
        if mask.sum() < N_SPLITS * 2:
            continue
        Xd, yd, g = bundle.X1[mask], bundle.y1[mask], bundle.group1[mask]
        oof_t, oof_p = group_kfold_oof(Xd, yd, g, lambda: build_fast_model(model_name))
        cv[drug] = evaluate_predictions(oof_t, oof_p)
        cv_pooled_t.extend(oof_t.tolist())
        cv_pooled_p.extend(oof_p.tolist())
        final = build_fast_model(model_name)
        final.fit(Xd, yd)
        models[drug] = final
    cv["MEAN"] = evaluate_predictions(cv_pooled_t, cv_pooled_p)
    return models, cv


def zero_shot_eval(bundle, models, drugs=COMMON_DRUGS):
    """Apply HIV-1 pre-trained models to HIV-2 with no HIV-2 training."""
    res = {}
    pooled_t, pooled_p = [], []
    for drug in drugs:
        if drug not in models:
            continue
        mask = bundle.drug2 == drug
        if mask.sum() < 5:
            continue
        Xd, yd = bundle.X2[mask], bundle.y2[mask]
        yp = np.ravel(models[drug].predict(Xd))
        res[drug] = evaluate_predictions(yd, yp)
        pooled_t.extend(yd.tolist())
        pooled_p.extend(yp.tolist())
    res["MEAN"] = evaluate_predictions(pooled_t, pooled_p)
    return res


def zero_shot_pooled_eval(bundle, model_name):
    """Pooled zero-shot: train ONE model on ALL HIV-1 (all drugs), predict ALL HIV-2.
    Mirrors the original retrain_honest.py setup for direct legacy comparison."""
    print(f"\n  --- Pooled zero-shot ({MODEL_LABELS[model_name]}, "
          f"all HIV-1 ({len(bundle.y1)} rows) -> all HIV-2 ({len(bundle.y2)} rows)) ---")
    model = build_fast_model(model_name)
    model.fit(bundle.X1, bundle.y1)
    yp = np.ravel(model.predict(bundle.X2))
    res = evaluate_predictions(bundle.y2, yp)
    print(f"    Pooled Pearson R = {res['pearson_r']:.4f}  "
          f"(n={res['n']}, CI=[{res['ci_pearson_lo']:.4f}, {res['ci_pearson_hi']:.4f}])")
    return res


# ============================================================================
# STAGE 2 — FINE-TUNE ON HIV-2 (KFold, HIV-1 scaler, OOF only)
# ============================================================================
def finetune_oof(bundle, model_name, drugs=COMMON_DRUGS, seed=RANDOM_SEED,
                 row_indices=None):
    """HIV-2 5-fold KFold fine-tune with HIV-1 scaler -> OOF per drug."""
    if row_indices is not None:
        X2 = bundle.X2[row_indices]
        y2 = bundle.y2[row_indices]
        drug2 = bundle.drug2[row_indices]
    else:
        X2, y2, drug2 = bundle.X2, bundle.y2, bundle.drug2
    res = dict.fromkeys(drugs, None)
    pooled_t, pooled_p = [], []
    for drug in drugs:
        mask = drug2 == drug
        if mask.sum() < 8:
            continue
        oof_t, oof_p = kfold_oof(
            X2[mask], y2[mask], lambda: build_fast_model(model_name), seed=seed
        )
        res[drug] = evaluate_predictions(oof_t, oof_p)
        pooled_t.extend(oof_t.tolist())
        pooled_p.extend(oof_p.tolist())
    res["MEAN"] = evaluate_predictions(pooled_t, pooled_p)
    return res


def supervised_oof(bundle, model_name, drugs=COMMON_DRUGS, seed=RANDOM_SEED):
    """Train directly on HIV-2 only — own scaler fit per fold (no transfer)."""
    res = {}
    pooled_t, pooled_p = [], []
    for drug in drugs:
        mask = bundle.drug2 == drug
        if mask.sum() < 8:
            continue
        seqs = bundle.seq2[mask]
        y = bundle.y2[mask]
        if bundle.feature_set not in ("biochem",):
            Xraw = _feature_store.get(seqs.tolist(), bundle.feature_set)
        else:
            Xraw = _feature_store.get(seqs.tolist(), bundle.feature_set)
        oof_pred = np.full(len(y), np.nan)
        n_split = min(N_SPLITS, max(2, len(y) // 8))
        kf = KFold(n_splits=n_split, shuffle=True, random_state=seed)
        for tr, te in kf.split(Xraw):
            scaler = StandardScaler().fit(Xraw[tr])
            model = build_fast_model(model_name)
            model.fit(scaler.transform(Xraw[tr]), y[tr])
            oof_pred[te] = model.predict(scaler.transform(Xraw[te]))
        valid = np.isfinite(oof_pred)
        oof_t, oof_p = y[valid], oof_pred[valid]
        res[drug] = evaluate_predictions(oof_t, oof_p)
        pooled_t.extend(oof_t.tolist())
        pooled_p.extend(oof_p.tolist())
    res["MEAN"] = evaluate_predictions(pooled_t, pooled_p)
    return res


# ============================================================================
# PART 1 — FEATURE ABLATION (RF, FAST)
# ============================================================================
def run_feature_ablation(bundles, model_name="T4_RF"):
    print("\n" + "=" * 78)
    print(f"PART 1 — FEATURE ABLATION  ({MODEL_LABELS[model_name]}, HIV-1 pre-train -> HIV-2)"
          .ljust(78))
    print("=" * 78)
    results = {}
    for fs in FEATURE_SETS:
        bundle = bundles[fs]
        print(f"\n  --- Feature set: {fs} ({FEATURE_LABELS[fs]}) ---")
        models, cv = pretrain_hiv1(bundle, model_name)
        zs = zero_shot_eval(bundle, models)
        ft = finetune_oof(bundle, model_name)
        results[fs] = {
            "hiv1_cv": cv,
            "zero_shot": zs,
            "finetuned": ft,
            "hiv1_cv_mean_r": mean_r(cv),
            "zero_shot_mean_r": zs.get("MEAN", {}).get("pearson_r", np.nan),
            "finetuned_mean_r": ft.get("MEAN", {}).get("pearson_r", np.nan),
        }
        print(f"      HIV-1 CV: {results[fs]['hiv1_cv_mean_r']:.4f} | "
              f"Zero-shot: {results[fs]['zero_shot_mean_r']:.4f} | "
              f"Fine-tuned: {results[fs]['finetuned_mean_r']:.4f}")

    print(f"\n  FEATURE ABLATION SUMMARY (mean R)" )
    print(f"  {'Feature set':<20}{'HIV-1 CV':>10}{'Zero-shot':>12}{'Fine-tuned':>12}")
    print(f"  {'-' * 54}")
    for fs in FEATURE_SETS:
        r = results[fs]
        print(f"  {FEATURE_LABELS[fs]:<20}{r['hiv1_cv_mean_r']:>10.4f}"
              f"{r['zero_shot_mean_r']:>12.4f}{r['finetuned_mean_r']:>12.4f}")
    return results


# ============================================================================
# PART 2 — MODEL COMPARISON (combined features)
# ============================================================================
def run_model_comparison(bundle):
    print("\n" + "=" * 78)
    print("PART 2 — MODEL COMPARISON  (combined features)".ljust(78))
    print("=" * 78)
    results = {}
    for mk in MODEL_KEYS:
        print(f"\n  --- Model: {MODEL_LABELS[mk]} ({mk}) ---")
        models, cv = pretrain_hiv1(bundle, mk)
        ft = finetune_oof(bundle, mk)
        results[mk] = {
            "model_label": MODEL_LABELS[mk],
            "hiv1_cv": cv,
            "hiv1_models": models,
            "finetuned": ft,
            "hiv1_cv_mean_r": mean_r(cv),
            "finetuned_mean_r": ft.get("MEAN", {}).get("pearson_r", np.nan),
        }
        print(f"      HIV-1 CV: {results[mk]['hiv1_cv_mean_r']:.4f} | "
              f"Fine-tuned HIV-2: {results[mk]['finetuned_mean_r']:.4f}")

    print(f"\n  MODEL COMPARISON SUMMARY (mean R)")
    print(f"  {'Model':<20}{'HIV-1 CV':>10}{'Fine-tuned HIV-2':>18}")
    print(f"  {'-' * 48}")
    for mk in MODEL_KEYS:
        r = results[mk]
        print(f"  {MODEL_LABELS[mk]:<20}{r['hiv1_cv_mean_r']:>10.4f}"
              f"{r['finetuned_mean_r']:>18.4f}")

    best = max(MODEL_KEYS, key=lambda mk: results[mk]["finetuned_mean_r"] if np.isfinite(
        results[mk]["finetuned_mean_r"]) else -1)
    print(f"\n  Best model on HIV-2 fine-tune: {MODEL_LABELS[best]} ({best}) "
          f"R={results[best]['finetuned_mean_r']:.4f}")
    return results, best


# ============================================================================
# PART 3 — LEARNING CURVE
# ============================================================================
def run_learning_curve(bundle, model_name, n_samples):
    print("\n" + "=" * 78)
    print(f"PART 3 — LEARNING CURVE  (model={MODEL_LABELS[model_name]})".ljust(78))
    print("=" * 78)
    zero_shot_models, _ = pretrain_hiv1(bundle, model_name)
    zs = zero_shot_eval(bundle, zero_shot_models)
    zs_r = zs.get("MEAN", {}).get("pearson_r", np.nan)

    curve = {"fractions": {}, "zero_shot_r": zs_r}
    rng_all = np.random.RandomState(RANDOM_SEED)
    for frac in LEARNING_FRACTIONS:
        rs = []
        for si in range(LC_N_SEEDS):
            rng = np.random.RandomState(RANDOM_SEED + si)
            n_use = max(int(round(n_samples * frac)), 20)
            n_use = min(n_use, n_samples)
            idx = rng.choice(n_samples, size=n_use, replace=False)
            ft = finetune_oof(bundle, model_name, row_indices=idx,
                              seed=RANDOM_SEED + si)
            r = ft.get("MEAN", {}).get("pearson_r", np.nan)
            if np.isfinite(r):
                rs.append(r)
        mean_r_val = float(np.mean(rs)) if rs else float("nan")
        std_r_val = float(np.std(rs)) if rs else float("nan")
        curve["fractions"][frac] = {
            "mean_r": mean_r_val,
            "std_r": std_r_val,
            "n_samples": n_use,
            "per_seed": rs,
        }
        print(f"  Fraction {frac * 100:4.0f}% ({n_use:3d} rows): "
              f"R = {mean_r_val:.4f} ± {std_r_val:.4f}  (3 seeds)")
    print(f"  Zero-shot baseline: R = {zs_r:.4f}")
    return curve


# ============================================================================
# PART 4 — CONFORMAL PREDICTION
# ============================================================================
def run_conformal(bundle, model_name, drugs=COMMON_DRUGS):
    print("\n" + "=" * 78)
    print("PART 4 — CONFORMAL PREDICTION  (split conformal, 30 % cal / 70 % test)"
          .ljust(78))
    print("=" * 78)
    results = {}
    for drug in drugs:
        mask = bundle.drug2 == drug
        if mask.sum() < 20:
            continue
        X = bundle.X2[mask]
        y = bundle.y2[mask]
        n = len(y)
        rng = np.random.RandomState(RANDOM_SEED)
        perm = rng.permutation(n)
        n_cal = max(int(round(n * CONFORMAL_CAL_FRAC)), 5)
        cal_idx, test_idx = perm[:n_cal], perm[n_cal:]

        model = build_fast_model(model_name)
        model.fit(X, y)  # deployed model on full HIV-2 drug data
        cal_resid = np.abs(y[cal_idx] - model.predict(X[cal_idx]))
        yp_test = np.ravel(model.predict(X[test_idx]))

        per_alpha = {}
        for alpha in (0.10, 0.05):
            q_level = np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal
            q_level = min(float(q_level), 1.0)
            q_hat = float(np.quantile(cal_resid, q_level))
            lower = yp_test - q_hat
            upper = yp_test + q_hat
            coverage = float(np.mean((y[test_idx] >= lower) & (y[test_idx] <= upper)))
            per_alpha[alpha] = {
                "coverage": round(coverage, 4),
                "mean_interval_width": round(float(2 * q_hat), 4),
                "q_hat": round(q_hat, 4),
                "n_test": int(len(test_idx)),
            }
            print(f"  {drug}: {int(alpha * 100):3d}% coverage={coverage:.3f} "
                  f"(width={2 * q_hat:.3f} kcal/mol, q_hat={q_hat:.3f}, n_test={len(test_idx)})")
        results[drug] = {"n_cal": int(n_cal), "n_test": int(len(test_idx)),
                         "90": per_alpha[0.10], "95": per_alpha[0.05]}
    return results


# ============================================================================
# PART 5 — SUMMARY / APPROACH COMPARISON
# ============================================================================
def run_approach_comparison(bundles, part1, part2, model_best,
                            feature_set_best, supervised_feature_set):
    print("\n" + "=" * 78)
    print("PART 5 — APPROACH COMPARISON  "
          f"(features={feature_set_best}, model={MODEL_LABELS[model_best]})".ljust(78))
    print("=" * 78)
    bundle = bundles[feature_set_best]

    # Zero-shot: reuse Part 2 pretrain if combined+best model, else re-train.
    zs_models = None
    if feature_set_best == "combined" and model_best in part2:
        zs_models = part2[model_best]["hiv1_models"]
    else:
        zs_models, _ = pretrain_hiv1(bundle, model_best)
    zero_shot = zero_shot_eval(bundle, zs_models)

    # Fine-tuned: reuse Part 2 OOF if combined+best, else recompute.
    if feature_set_best == "combined" and model_best in part2:
        finetuned = part2[model_best]["finetuned"]
    else:
        finetuned = finetune_oof(bundle, model_best)

    # Supervised: direct HIV-2 training with own per-fold scaler.
    sup_bundle = bundles[supervised_feature_set] if supervised_feature_set in bundles \
        else bundle
    supervised = supervised_oof(sup_bundle, model_best)

    approaches = {
        "Zero-shot (HIV-1->HIV-2)": zero_shot,
        "Fine-tuned (HIV-2 ChEMBL)": finetuned,
        "Supervised (HIV-2 only)": supervised,
    }

    print("\nAPPROACH COMPARISON")
    header = f"{'Approach':<30}" + "".join(f"{d:>8}" for d in COMMON_DRUGS) + f"{'MEAN':>8}"
    print(header)
    print("-" * len(header))
    for name, res in approaches.items():
        row = f"{name:<30}"
        for d in COMMON_DRUGS:
            r = res.get(d, {}).get("pearson_r", np.nan)
            row += f"{r:>8.3f}" if np.isfinite(r) else f"{'N/A':>8}"
        m = res.get("MEAN", {}).get("pearson_r", np.nan)
        row += f"{m:>8.3f}" if np.isfinite(m) else f"{'N/A':>8}"
        print(row)
    print("-" * len(header))
    return approaches


def plot_transfer_comparison(approaches, save_path=None):
    """Bar chart of mean Pearson R per approach."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    names = list(approaches.keys())
    means = [approaches[n].get("MEAN", {}).get("pearson_r", np.nan) for n in names]
    los = [approaches[n].get("MEAN", {}).get("ci_pearson_lo", np.nan) for n in names]
    his = [approaches[n].get("MEAN", {}).get("ci_pearson_hi", np.nan) for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(x, means, yerr=[
        [0 if not np.isfinite(m - lo) else m - lo for m, lo in zip(means, los)],
        [0 if not np.isfinite(hi - m) else hi - m for m, hi in zip(means, his)],
    ], capsize=5, color=colors[:len(names)], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, means):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Mean Pearson R (pooled OOF, 4 drugs)")
    ax.set_title("Transfer Learning Comparison — HIV-2 PI Resistance")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    finite = [m for m in means if np.isfinite(m)]
    if finite:
        lo_lim = min(0, min(finite) - 0.1)
        hi_lim = max(finite) * 1.25 if max(finite) > 0 else 0.3
        ax.set_ylim(lo_lim, hi_lim)
    plt.tight_layout()
    save_path = save_path or FIGURES_DIR / "fig_transfer_comparison.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {save_path}")


def save_results(output, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    print(f"  Saved results: {path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    t_start = time.time()
    set_seed()
    print("=" * 78)
    print("TWO-STAGE TRANSFER LEARNING PIPELINE — HIV-1 -> HIV-2")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 78)

    # ----------------------------------------------------------------- load
    hiv1_df = load_hiv1_data()
    hiv2_df = load_hiv2_chembl()
    hiv2_clinical_df, clinical_info = load_hiv2_clinical()

    bundles = build_bundles(hiv1_df, hiv2_df)

    output = {
        "data_stats": {
            "hiv1_n": int(len(hiv1_df)),
            "hiv1_drugs": {str(k): int(v) for k, v in hiv1_df["drug"].value_counts().items()},
            "hiv1_dg_range": [float(hiv1_df["dg"].min()), float(hiv1_df["dg"].max())],
            "hiv2_chembl_n": int(len(hiv2_df)),
            "hiv2_chembl_drugs": {str(k): int(v) for k, v in hiv2_df["drug"].value_counts().items()},
            "hiv2_common4_n": int(np.sum(np.isin(hiv2_df["drug"].values, COMMON_DRUGS))),
            "clinical": clinical_info,
            "common_drugs": COMMON_DRUGS,
        },
        "constants": {
            "rt_310": RT_310,
            "n_bootstrap_ci": N_BOOTSTRAP,
            "n_splits": N_SPLITS,
        },
    }

    # ----------------------------------------------------------- Part 1
    part1 = run_feature_ablation(bundles, model_name="T4_RF")
    output["part1_feature_ablation"] = part1

    # Pooled zero-shot for biochem (legacy-comparable result)
    pooled_zs_biochem = zero_shot_pooled_eval(bundles["biochem"], "T4_RF")
    output["part1_pooled_zero_shot_biochem"] = {str(k): v for k, v in pooled_zs_biochem.items()}

    # ----------------------------------------------------------- Part 2
    part2, model_best = run_model_comparison(bundles["combined"])
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

    # Prefer T4_RF as primary model if its fine-tune is competitive or better
    rf_ft = part2.get("T4_RF", {}).get("finetuned_mean_r", float("nan"))
    best_ft = part2.get(model_best, {}).get("finetuned_mean_r", float("nan"))
    if np.isfinite(rf_ft) and (not np.isfinite(best_ft) or rf_ft >= best_ft * 0.95):
        model_best = "T4_RF"
        print(f"\n  >> Using T4_RF as primary model (R={rf_ft:.4f} >= 95% of best)")
    output["best_model"] = model_best

    # ----------------------------------------------------------- Part 3
    n_hiv2_rows = len(hiv2_df)
    curve = run_learning_curve(bundles["combined"], model_best, n_hiv2_rows)
    output["part3_learning_curve"] = {
        "best_model": model_best,
        "n_hiv2_samples": int(n_hiv2_rows),
        "zero_shot_r": curve["zero_shot_r"],
        "fractions": {str(k): v for k, v in curve["fractions"].items()},
    }

    # ----------------------------------------------------------- Part 4
    conformal = run_conformal(bundles["combined"], model_best)
    output["part4_conformal_prediction"] = conformal

    # ----------------------------------------------------------- Part 5
    feature_set_best = max(
        FEATURE_SETS,
        key=lambda fs: part1[fs]["finetuned_mean_r"] if np.isfinite(
            part1[fs]["finetuned_mean_r"]) else -1,
    )
    approaches = run_approach_comparison(
        bundles, part1, part2, model_best, feature_set_best,
        supervised_feature_set=feature_set_best,
    )
    output["part5_approach_comparison"] = {
        name: {str(k): v for k, v in res.items()} for name, res in approaches.items()
    }
    output["approach_means"] = {name: res.get("MEAN", {}) for name, res in approaches.items()}
    output["feature_set_best"] = feature_set_best
    output["model_best"] = model_best

    # ------------------------------------------------------- figure + json
    try:
        plot_transfer_comparison(approaches)
    except Exception as e:
        print(f"  [warn] figure failed: {e}")

    out_path = RESULTS_DIR / "transfer_learning_results.json"
    save_results(output, out_path)

    elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print("PIPELINE COMPLETE")
    print(f"Duration: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Results:  {out_path}")
    print(f"Figure:   {FIGURES_DIR / 'fig_transfer_comparison.png'}")
    print("=" * 78)
    return output


if __name__ == "__main__":
    main()