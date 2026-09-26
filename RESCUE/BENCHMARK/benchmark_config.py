"""
HIV-2 Drug Resistance Benchmark — Configuration
================================================
Single source of truth for all drugs, methods, features, splits, metrics.
Every other benchmark file imports from here.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
# Resolved relative to this file so the repository is portable:
#   <repo root>/RESCUE/BENCHMARK/benchmark_config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESCUE_DIR = PROJECT_ROOT / "RESCUE"
BENCHMARK_DIR = RESCUE_DIR / "BENCHMARK"
RESULTS_DIR = BENCHMARK_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
OOF_DIR = RESULTS_DIR / "oof_predictions"
# Packaged layout uses <repo root>/data; fall back to the original
# development folder name if data/ is absent.
DATA_DIR = PROJECT_ROOT / "data"
if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / "known_info_master"
ESM2_DIR = PROJECT_ROOT / "advanced_feature_extraction"
SHARED_DIR = PROJECT_ROOT / "opencode_shared"

# Data files
CLINICAL_MULTILABEL = DATA_DIR / "HIV2_CLINICAL_ML_READY_MULTILABEL.csv"
CLINICAL_FLAT = DATA_DIR / "HIV2_CLINICAL_ML_READY_FLAT.csv"
ESM2_650M_NPZ = ESM2_DIR / "esm2_650M_embeddings.npz"
HIV1_TRAINING = DATA_DIR / "STANFORD_HIV1_GENUINE.csv"

# Create output dirs
for d in [RESULTS_DIR, FIGURES_DIR, OOF_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Drug Definitions ──────────────────────────────────────────────────
DRUGS = ["ATV/r", "DRV/r", "FPV/r", "IDV/r", "LPV/r", "SQV/r", "TPV/r", "NFV/r"]

DRUG_NAMES = {
    "ATV/r": "Atazanavir/r",
    "DRV/r": "Darunavir/r",
    "FPV/r": "Fosamprenavir/r",
    "IDV/r": "Indinavir/r",
    "LPV/r": "Lopinavir/r",
    "SQV/r": "Saquinavir/r",
    "TPV/r": "Tipranavir/r",
    "NFV/r": "Nelfinavir/r",
}

# ── Feature Registry ─────────────────────────────────────────────────
FEATURES = {
    "F1": {"name": "Binary mutation matrix", "dim": "dynamic", "source": "computed"},
    "F2": {"name": "One-hot sequence", "dim": "dynamic", "source": "computed"},
    "F3": {"name": "ESM-2 mean-pool", "dim": 1280, "source": "esm2_650M_embeddings.npz"},
    "F4": {"name": "ESM-2 attention-pool", "dim": 1280, "source": "esm2_650M_embeddings.npz"},
    "F5": {"name": "ESM-2 max-pool", "dim": 1280, "source": "esm2_650M_embeddings.npz"},
    "F6": {"name": "Biochemical (Z+KF+VHSE)", "dim": 2121, "source": "computed"},
    "F7": {"name": "Pocket features", "dim": 782, "source": "computed"},
}

# ── Model Registry ────────────────────────────────────────────────────
MODELS = {
    "T0_mean": {"name": "Constant-mean predictor", "type": "trivial"},
    "T1_hiv2eu": {"name": "HIV-2EU rules", "type": "rule-based"},
    "T2_LR": {"name": "Logistic Regression", "type": "classification"},
    "T2_RF": {"name": "Random Forest", "type": "classification"},
    "T2_XGB": {"name": "XGBoost", "type": "classification"},
    "T2_SVM": {"name": "SVM-RBF", "type": "classification"},
    "T3_CNN": {"name": "1D-CNN", "type": "deep-learning"},
    "T3_LSTM": {"name": "BiLSTM", "type": "deep-learning"},
    "T4_Ridge": {"name": "Ridge Regression", "type": "regression"},
    "T4_EN": {"name": "ElasticNet", "type": "regression"},
    "T4_XGB": {"name": "XGBoost Regressor", "type": "regression"},
    "T4_RF": {"name": "Random Forest Regressor", "type": "regression"},
    "T5_ensemble": {"name": "Ensemble (Ridge+RF+XGB avg)", "type": "meta"},
}

# ── CV Schemes ────────────────────────────────────────────────────────
CV_SCHEMES = {
    "S1_GroupKFold5": {"name": "GroupKFold(5)", "n_splits": 5, "description": "Group by sequence identity (95% ID)"},
    "S2_StratifiedKFold5": {"name": "StratifiedKFold(5)", "n_splits": 5, "description": "Stratify by binary resistance class"},
}

# ── Metrics ───────────────────────────────────────────────────────────
METRICS = [
    "pearson_r", "spearman_rho", "rmse", "mae", "r2",
    "p1_accuracy", "sensitivity", "specificity", "auroc"
]

# ── Evaluation Constants ──────────────────────────────────────────────
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42
N_SEEDS = 5

# ── HIV-2 Wild-Type Sequence ──────────────────────────────────────────
HIV2_WT_PR = (
    "PQFSLWKRPVVTAYIEGQPVEVLLDTGADDSIVAGIELGNNYSPKIVGGIGGF"
    "INTKEYKNVEIEVLNKKVRATIMTGDTPINIFGRNILTALGMSLNL"
)

SEQ_LENGTH = len(HIV2_WT_PR)

# ── HIV-2EU Rule Table (approximate, from published literature) ───────
HIV2EU_RULES = {
    "I50V":   {"ATV/r": 2, "DRV/r": 1, "FPV/r": 1, "IDV/r": 1, "LPV/r": 1, "SQV/r": 1, "NFV/r": 1},
    "I54M":   {"DRV/r": 1, "LPV/r": 1},
    "I54V":   {"DRV/r": 1, "LPV/r": 2, "NFV/r": 1},
    "I82F":   {"LPV/r": 1, "IDV/r": 2, "NFV/r": 2, "ATV/r": 1, "FPV/r": 2},
    "I84V":   {"ATV/r": 2, "DRV/r": 1, "FPV/r": 2, "IDV/r": 2, "LPV/r": 2, "SQV/r": 2, "NFV/r": 2},
    "L90M":   {"SQV/r": 3, "LPV/r": 1, "ATV/r": 1, "NFV/r": 1},
    "V47A":   {"LPV/r": 2},
    "G48V":   {"SQV/r": 2},
    "V82A":   {"IDV/r": 2, "NFV/r": 2, "LPV/r": 1, "FPV/r": 2},
    "V82I":   {"IDV/r": 1, "NFV/r": 1},
    "V82L":   {"IDV/r": 1, "NFV/r": 1},
    "V82T":   {"IDV/r": 1, "NFV/r": 1},
    "T56V":   {"LPV/r": 2},
    "N83D":   {"DRV/r": 1},
    "L10F":   {"DRV/r": 1, "FPV/r": 1, "IDV/r": 1, "LPV/r": 1, "NFV/r": 1, "SQV/r": 1},
    "L10I":   {"ATV/r": 1, "DRV/r": 1, "FPV/r": 1, "IDV/r": 1, "LPV/r": 1, "NFV/r": 1, "SQV/r": 1},
    "K20R":   {"ATV/r": 1, "IDV/r": 1, "NFV/r": 1},
    "L23I":   {"ATV/r": 1},
    "L24I":   {"ATV/r": 1, "FPV/r": 1, "IDV/r": 1},
    "D30N":   {"NFV/r": 3},
    "V32I":   {"DRV/r": 1, "FPV/r": 1, "IDV/r": 1, "LPV/r": 1, "NFV/r": 1},
    "I47V":   {"DRV/r": 1, "LPV/r": 2},
    "I50L":   {"ATV/r": 3},
    "I53V":   {"DRV/r": 1, "LPV/r": 1, "NFV/r": 1},
    "I62V":   {"IDV/r": 1, "LPV/r": 1},
    "A71V":   {"IDV/r": 1, "LPV/r": 1, "NFV/r": 1},
    "G73S":   {"IDV/r": 1, "NFV/r": 1},
    "L76V":   {"DRV/r": 2, "LPV/r": 2},
    "V82F":   {"IDV/r": 2, "NFV/r": 2, "FPV/r": 2, "LPV/r": 1},
    "N88D":   {"NFV/r": 2},
    "N88S":   {"NFV/r": 3},
    "L89V":   {"ATV/r": 1, "IDV/r": 1, "NFV/r": 1},
    "V62A":   {"IDV/r": 1, "NFV/r": 1},
    "L99F":   {"IDV/r": 1, "LPV/r": 1},
}

# ── Amino Acid Encoding ──────────────────────────────────────────────
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

# ── Reproducibility ──────────────────────────────────────────────────
import numpy as np
import random

def set_seed(seed=RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
