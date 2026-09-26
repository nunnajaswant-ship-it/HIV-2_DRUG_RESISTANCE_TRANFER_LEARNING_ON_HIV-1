import logging
import os
import subprocess
import sys
import random
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

from opencode_shared.biochem_scales import Z_SCALES, KIDERA, VHSE, DRUG_SMILES
from opencode_shared.config import SEQ_LENGTH, HIV1_WT, HIV2_WT, POCKET_1INDEX, POCKET_0INDEX

import torch

import random

def set_all_seeds(seed: int = 42):
    """Set random seeds for all frameworks to ensure reproducibility.
    
    Must be called at the START of every training script.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def bh_correct(p_values: List[float], alpha: float = 0.05) -> List[float]:
    """Benjamini-Hochberg correction for multiple testing.
    
    Returns adjusted p-values (q-values). Controls FDR at alpha.
    """
    p = np.array(p_values)
    n = len(p)
    ranked = np.argsort(p)
    q = np.zeros(n)
    q[ranked] = np.minimum(1, p[ranked] * n / (np.arange(1, n + 1)))
    for i in range(n - 2, -1, -1):
        q[ranked[i]] = min(q[ranked[i]], q[ranked[i + 1]])
    return q.tolist()


GLOBAL_FEATURE_DIM = 2121
PER_RESIDUE_FEATURE_DIM = 2079

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


def set_up_logging(verbose: bool = True) -> None:
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: float = 1.0
) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]

    if len(yt) < 2:
        return {'n': len(yt), 'rmse': float('nan'), 'mae': float('nan'),
                'pearson_r': float('nan'), 'spearman_r': float('nan'),
                'pearson_pval': float('nan'), 'p1.0_accuracy': float('nan')}

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r, p_val = pearsonr(yt, yp)
    rho, _ = spearmanr(yt, yp)
    p1 = float(np.mean(np.abs(yt - yp) <= tolerance))
    p2 = float(np.mean(np.abs(yt - yp) <= 2.0))

    return {
        'n': int(len(yt)),
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'pearson_r': round(float(r), 4),
        'spearman_r': round(float(rho), 4),
        'pearson_pval': round(float(p_val), 6),
        'p1.0_accuracy': round(p1, 4),
        'p2.0_accuracy': round(p2, 4),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Any = pearsonr,
    n_bootstrap: int = 1000,
    ci: float = 0.95
) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    n = len(yt)
    if n < 5:
        return {'ci_lower': float('nan'), 'ci_upper': float('nan'), 'mean': float('nan')}

    stats = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(yt[idx])) > 1 and len(np.unique(yp[idx])) > 1:
            r, _ = metric_fn(yt[idx], yp[idx])
            stats.append(r)
        else:
            stats.append(0.0)

    stats = np.sort(stats)
    lower_pct = (1.0 - ci) / 2.0
    upper_pct = 1.0 - lower_pct
    return {
        'ci_lower': round(float(np.percentile(stats, lower_pct * 100)), 4),
        'ci_upper': round(float(np.percentile(stats, upper_pct * 100)), 4),
        'mean': round(float(np.mean(stats)), 4),
    }


VALID_AA = set('ACDEFGHIKLMNPQRSTVWY-')

def validate_sequence(sequence: str) -> str:
    seq = (sequence[:SEQ_LENGTH] + '-' * SEQ_LENGTH)[:SEQ_LENGTH]
    invalid = set(seq.upper()) - VALID_AA
    if invalid:
        log.warning(f"Sequence contains invalid amino acids: {invalid}. Replacing with '-'.")
        for aa in invalid:
            seq = seq.replace(aa, '-')
    return seq.upper()


def extract_per_residue_features(sequence: str) -> np.ndarray:
    seq = validate_sequence(sequence)
    mat = np.zeros((SEQ_LENGTH, 21), dtype=np.float32)
    for i, aa in enumerate(seq):
        z = Z_SCALES.get(aa, [0.0, 0.0, 0.0])
        k = KIDERA.get(aa, [0.0] * 10)
        v = VHSE.get(aa, [0.0] * 8)
        mat[i] = z + k + v
    return mat


def extract_global_features(sequence: str, include_stats: bool = True) -> np.ndarray:
    mat = extract_per_residue_features(sequence)
    flat = mat.flatten()
    if not include_stats:
        return flat.astype(np.float32)
    # include_stats=True: 2,079 per-residue + 21 mean + 21 std = 2,121-D
    # include_stats=False: 2,079-D (matches manuscript description)
    gmean = mat.mean(axis=0)
    gstd = mat.std(axis=0)
    return np.concatenate([flat, gmean, gstd]).astype(np.float32)


def extract_pocket_features(
    sequence: str,
    pocket_indices: Optional[List[int]] = None
) -> np.ndarray:
    if pocket_indices is None:
        pocket_indices = POCKET_0INDEX
    seq = (sequence[:SEQ_LENGTH] + '-' * SEQ_LENGTH)[:SEQ_LENGTH]
    n_pocket = len(pocket_indices)

    pocket_mat = np.zeros((n_pocket, 21), dtype=np.float32)
    pocket_aas = []
    for j, idx in enumerate(pocket_indices):
        aa = seq[idx] if idx < len(seq) else '-'
        pocket_aas.append(aa)
        z = Z_SCALES.get(aa, [0.0, 0.0, 0.0])
        k = KIDERA.get(aa, [0.0] * 10)
        v = VHSE.get(aa, [0.0] * 8)
        pocket_mat[j] = z + k + v

    per_residue_flat = pocket_mat.flatten()
    pocket_mean = pocket_mat.mean(axis=0)
    pocket_std = pocket_mat.std(axis=0)
    pocket_max = pocket_mat.max(axis=0)
    pocket_min = pocket_mat.min(axis=0)

    from opencode_shared.config import AROMATIC, HYDROPHOBIC, POLAR_UNCHARGED, POSITIVE, NEGATIVE, SMALL
    n_aromatic = sum(1 for aa in pocket_aas if aa in AROMATIC) / n_pocket
    n_hydrophobic = sum(1 for aa in pocket_aas if aa in HYDROPHOBIC) / n_pocket
    n_polar = sum(1 for aa in pocket_aas if aa in POLAR_UNCHARGED) / n_pocket
    n_positive = sum(1 for aa in pocket_aas if aa in POSITIVE) / n_pocket
    n_negative = sum(1 for aa in pocket_aas if aa in NEGATIVE) / n_pocket
    n_small = sum(1 for aa in pocket_aas if aa in SMALL) / n_pocket
    counts_feat = np.array([n_aromatic, n_hydrophobic, n_polar, n_positive, n_negative, n_small])

    total_electrostatic = float(pocket_mat[:, 2].sum())
    total_hydrophobic = float(pocket_mat[:, 0].sum())
    mean_steric = float(pocket_mat[:, 1].mean())
    pocket_diversity = float(pocket_mat[:, 0].std())
    scalar_feats = np.array([total_electrostatic, total_hydrophobic, mean_steric, pocket_diversity])

    features = np.concatenate([
        per_residue_flat,
        pocket_mean, pocket_std, pocket_max, pocket_min,
        counts_feat, scalar_feats
    ])
    return features.astype(np.float32)


def compute_ic50_to_delta_g(
    ic50_nm: float,
    temperature_k: float = 298.0,
    assay_type: str = 'IC50'
) -> float:
    """Convert IC50 (nM) to binding free energy ΔG (kcal/mol).
    
    Uses ΔG = RT·ln(IC50·1e-9) where IC50 is in nM.
    RT = 0.593 kcal/mol at 298K (standard assay temp, default).
    RT = 0.616 kcal/mol at 310K (physiological).
    
    For Ki values: this is the true thermodynamic ΔG.
    For IC50 values: this is an approximation (Cheng-Prusoff correction
    IC50 = Ki·(1+[S]/Km) cannot be applied without enzyme kinetic params).
    
    Parameters
    ----------
    ic50_nm : float
        Binding affinity value in nM (IC50, Ki, or Kd).
    temperature_k : float
        Temperature in Kelvin. Default 298K (standard lab temp).
    assay_type : str
        'KI' = exact ΔG, 'IC50' = approximate ΔG, 'KD' = exact ΔG.
    """
    R = 0.001987  # kcal/(mol·K)
    RT = R * temperature_k
    if ic50_nm <= 0:
        return float('nan')
    if ic50_nm < 0.001:
        ic50_nm = 0.001
    dg = float(RT * np.log(ic50_nm * 1e-9))
    return dg


VALID_ASSAY_TYPES = {'IC50', 'KI', 'KD'}

def filter_assay_types(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to only IC50/Ki/Kd assay types.
    
    Removes EC50, Ratio, Fold, and other unitless/non-binding measurements
    that conflate binding affinity with cellular permeability.
    
    Handles suffixed entries (e.g. "IC50 (nM)", "Ki (apparent)").
    """
    assay_col = find_column(df, ['Assay_Type', 'assay_type', 'STANDARD_TYPE', 'Standard_Type'])
    if assay_col is not None:
        before = len(df)
        df[assay_col] = df[assay_col].astype(str).str.strip().str.upper()
        df = df[df[assay_col].str.startswith(('IC50', 'KI', 'KD'), na=False)].copy()
        after = len(df)
        if after < before:
            log.info(f"  Assay filter: removed {before - after} non-binding entries (IC50/Ki/Kd only)")
    return df


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_featurized_data(
    csv_path: str,
    pocket_indices: Optional[List[int]] = None,
    use_pocket: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    if not os.path.exists(csv_path):
        return None, None, None

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except (OSError, ValueError, RuntimeError) as e:
        log.error(f"Failed to read {csv_path}: {e}")
        return None, None, None

    seq_col = find_column(df, ['Mutant_Sequence', 'Sequence', 'sequence'])
    if seq_col is None:
        return None, None, None

    global_feats_list = []
    pocket_feats_list = []
    valid_indices = []

    for idx, row in df.iterrows():
        seq = str(row[seq_col]).strip().upper()
        if len(seq) < 90:
            continue
        global_feats_list.append(extract_global_features(seq))
        if use_pocket:
            pocket_feats_list.append(extract_pocket_features(seq, pocket_indices))
        valid_indices.append(idx)

    global_feats = np.array(global_feats_list)
    pocket_feats = np.array(pocket_feats_list) if use_pocket else None
    return global_feats, pocket_feats, valid_indices


def clean_drug_name(name):
    """Standardize drug name to uppercase canonical form."""
    mapping = {
        'AMP': 'AMPRENAVIR', 'APV': 'AMPRENAVIR',
        'ATV': 'ATAZANAVIR', 'AZV': 'ATAZANAVIR',
        'DRV': 'DARUNAVIR',
        'IDV': 'INDINAVIR',
        'LPV': 'LOPINAVIR',
        'NFV': 'NELFINAVIR',
        'RTV': 'RITONAVIR',
        'SQV': 'SAQUINAVIR',
        'TPV': 'TIPRANAVIR',
        'AMPRENAVIR': 'AMPRENAVIR', 'ATAZANAVIR': 'ATAZANAVIR',
        'DARUNAVIR': 'DARUNAVIR', 'INDINAVIR': 'INDINAVIR',
        'LOPINAVIR': 'LOPINAVIR', 'NELFINAVIR': 'NELFINAVIR',
        'RITONAVIR': 'RITONAVIR', 'SAQUINAVIR': 'SAQUINAVIR',
        'TIPRANAVIR': 'TIPRANAVIR',
    }
    return mapping.get(str(name).strip().upper(), str(name).strip())


def detect_feature_columns(df):
    """Detect feature column groups in a dataframe."""
    z_cols = [c for c in df.columns if c.startswith('Z_pos_')]
    kf_cols = [c for c in df.columns if c.startswith('KF_pos_')]
    vhse_cols = [c for c in df.columns if c.startswith('VHSE_pos_')]
    bert_cols = [c for c in df.columns if c.startswith('ProtBERT_dim_') or c.startswith('BERT_')]
    struct_cols = [c for c in df.columns if c.startswith('Struct_')]
    drug_cols = [c for c in df.columns if any(d in c for d in ['Drug_Mol', 'Drug_LogP', 'Drug_Morgan', 'Drug_HDonor', 'Drug_HAcceptor'])]
    return {
        'z': z_cols, 'kidera': kf_cols, 'vhse': vhse_cols,
        'bert': bert_cols, 'struct': struct_cols, 'drug': drug_cols,
        'all_biochem': z_cols + kf_cols + vhse_cols,
        'all': z_cols + kf_cols + vhse_cols + bert_cols + struct_cols + drug_cols,
    }


def benjamini_hochberg(p_values):
    """Apply Benjamini-Hochberg FDR correction to p-values."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    ranks = np.arange(1, n + 1)
    thresholds = ranks / n * 0.05
    significant = sorted_p <= thresholds
    if significant.any():
        max_sig = np.where(significant)[0].max()
        adjusted = np.where(ranks <= max_sig + 1, sorted_p * n / ranks, 1.0)
    else:
        adjusted = np.ones(n)
    result = np.ones(n)
    for i, adj in zip(sorted_idx, adjusted):
        result[i] = min(adj, 1.0)
    return result


def get_provenance() -> Dict[str, Any]:
    """Capture git hash and key dependency versions for provenance tracking."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
    }
    # Git hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        info["git_hash"] = result.stdout.strip() if result.returncode == 0 else "not_a_git_repo"
    except (OSError, ValueError, RuntimeError):
        info["git_hash"] = "unknown"

    # Key dependency versions
    deps = {}
    for mod_name in ["numpy", "scipy", "pandas", "sklearn", "torch", "Bio"]:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", None)
            if ver:
                deps[mod_name] = ver
        except ImportError:
            deps[mod_name] = "not_installed"
    info["dependencies"] = deps
    return info


