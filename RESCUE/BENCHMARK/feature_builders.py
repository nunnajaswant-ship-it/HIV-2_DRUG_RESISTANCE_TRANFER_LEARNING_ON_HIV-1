"""
Feature builders for the HIV-2 Drug Resistance Benchmark.
7 feature representations, each returning a standardized np.ndarray.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "opencode_shared"))

from config import HIV2_WT, SEQ_LENGTH, AA_LIST, AA_TO_IDX
from benchmark_config import HIV2_WT_PR


def build_binary_mutation(sequences, wt_seq=HIV2_WT_PR):
    """
    F1: Binary mutation matrix.
    For each position: 20 channels, 1 if AA != WT, 0 if AA == WT.
    Shape: (N, SEQ_LENGTH × 20) = (N, 1980)
    """
    N = len(sequences)
    mat = np.zeros((N, SEQ_LENGTH, 20), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq_str = str(seq).strip()
        for pos in range(min(len(seq_str), SEQ_LENGTH)):
            aa = seq_str[pos].upper()
            if aa in AA_TO_IDX and aa != wt_seq[pos].upper():
                mat[i, pos, AA_TO_IDX[aa]] = 1.0
    
    return mat.reshape(N, -1)


def build_onehot(sequences):
    """
    F2: One-hot sequence encoding.
    Shape: (N, SEQ_LENGTH × 20) = (N, 1980)
    """
    N = len(sequences)
    mat = np.zeros((N, SEQ_LENGTH, 20), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq_str = str(seq).strip()
        for pos in range(min(len(seq_str), SEQ_LENGTH)):
            aa = seq_str[pos].upper()
            if aa in AA_TO_IDX:
                mat[i, pos, AA_TO_IDX[aa]] = 1.0
    
    return mat.reshape(N, -1)


def build_esm2_mean(per_residue_data):
    """
    F3: ESM-2 mean-pooling.
    Input: per_residue (N, L, 1280)
    Output: (N, 1280)
    """
    if per_residue_data is None:
        raise ValueError("Per-residue data required for mean pooling")
    return per_residue_data.mean(axis=1)


def build_esm2_attention(per_residue_data):
    """
    F4: ESM-2 variance-weighted attention pooling.
    Positions with higher variance across channels are more informative.
    Input: per_residue (N, L, 1280)
    Output: (N, 1280)
    """
    if per_residue_data is None:
        raise ValueError("Per-residue data required for attention pooling")
    
    # Variance across feature dimension for each position
    variances = per_residue_data.var(axis=2)  # (N, L)
    
    # Softmax-normalize across positions
    variances = variances - variances.max(axis=1, keepdims=True)  # numerical stability
    weights = np.exp(variances)
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)  # (N, L)
    
    # Weighted sum
    pooled = np.einsum("nlk,nl->nk", per_residue_data, weights)  # (N, 1280)
    return pooled


def build_esm2_maxpool(per_residue_data):
    """
    F5: ESM-2 max-pooling.
    Input: per_residue (N, L, 1280)
    Output: (N, 1280)
    """
    if per_residue_data is None:
        raise ValueError("Per-residue data required for max pooling")
    return per_residue_data.max(axis=1)


# ── Biochemical features (imported from existing codebase) ──────────
_biochem_cache = {}

def build_biochem(sequences):
    """
    F6: Biochemical features (Z-scale + Kidera-Factor + VHSE).
    Uses existing extract_global_features from opencode_shared.
    Shape: (N, 2121)
    """
    cache_key = "biochem"
    if cache_key in _biochem_cache:
        return _biochem_cache[cache_key]
    
    try:
        from utils import extract_global_features
        features = []
        for seq in sequences:
            feat = extract_global_features(str(seq))
            features.append(feat)
        features = np.array(features, dtype=np.float32)
        _biochem_cache[cache_key] = features
        print(f"Computed biochemical features: {features.shape}")
        return features
    except ImportError as e:
        print(f"Warning: Could not import extract_global_features: {e}")
        print("  Returning zeros as placeholder")
        return np.zeros((len(sequences), 2121), dtype=np.float32)


def build_pocket(sequences):
    """
    F7: Pocket features.
    Uses existing extract_pocket_features from opencode_shared.
    Shape: (N, 782)
    """
    cache_key = "pocket"
    if cache_key in _biochem_cache:
        return _biochem_cache[cache_key]
    
    try:
        from utils import extract_pocket_features
        features = []
        for seq in sequences:
            feat = extract_pocket_features(str(seq))
            features.append(feat)
        features = np.array(features, dtype=np.float32)
        _biochem_cache[cache_key] = features
        print(f"Computed pocket features: {features.shape}")
        return features
    except ImportError as e:
        print(f"Warning: Could not import extract_pocket_features: {e}")
        print("  Returning zeros as placeholder")
        return np.zeros((len(sequences), 782), dtype=np.float32)


# ── Master builder ──────────────────────────────────────────────────
def build_features(sequences, esm2_data, feature_ids=None):
    """
    Build all requested features.
    
    Args:
        sequences: list of amino acid sequences
        esm2_data: dict with 'mean_pooled' and optionally 'per_residue'
        feature_ids: list of feature IDs to build (default: all F1-F7)
    
    Returns: dict of {feature_id: np.ndarray}
    """
    if feature_ids is None:
        feature_ids = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]
    
    features = {}
    
    print(f"Building features: {feature_ids}")
    
    for fid in feature_ids:
        print(f"  {fid}: ", end="", flush=True)
        
        if fid == "F1":
            features[fid] = build_binary_mutation(sequences)
            print(f"shape={features[fid].shape}")
            
        elif fid == "F2":
            features[fid] = build_onehot(sequences)
            print(f"shape={features[fid].shape}")
            
        elif fid == "F3":
            features[fid] = esm2_data["mean_pooled"]
            print(f"shape={features[fid].shape}")
            
        elif fid == "F4":
            if esm2_data.get("per_residue") is not None:
                features[fid] = build_esm2_attention(esm2_data["per_residue"])
                print(f"shape={features[fid].shape}")
            else:
                print("SKIPPED (no per-residue data)")
                
        elif fid == "F5":
            if esm2_data.get("per_residue") is not None:
                features[fid] = build_esm2_maxpool(esm2_data["per_residue"])
                print(f"shape={features[fid].shape}")
            else:
                print("SKIPPED (no per-residue data)")
                
        elif fid == "F6":
            features[fid] = build_biochem(sequences)
            print(f"shape={features[fid].shape}")
            
        elif fid == "F7":
            features[fid] = build_pocket(sequences)
            print(f"shape={features[fid].shape}")
            
        else:
            print(f"UNKNOWN feature ID: {fid}")
    
    return features


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Feature builders loaded successfully.")
    print(f"  F1: Binary mutation matrix (N, {SEQ_LENGTH * 20})")
    print(f"  F2: One-hot sequence (N, {SEQ_LENGTH * 20})")
    print(f"  F3: ESM-2 mean-pool (N, 1280)")
    print(f"  F4: ESM-2 attention-pool (N, 1280)")
    print(f"  F5: ESM-2 max-pool (N, 1280)")
    print(f"  F6: Biochemical (N, 2121)")
    print(f"  F7: Pocket (N, 782)")
