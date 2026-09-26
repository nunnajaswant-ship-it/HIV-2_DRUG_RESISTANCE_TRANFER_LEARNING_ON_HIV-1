"""
Unified data loading for the HIV-2 Drug Resistance Benchmark.
Loads clinical targets, ESM-2 embeddings, and sequences.
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / "opencode_shared"))

import numpy as np
import pandas as pd
from pathlib import Path
from config import HIV2_WT, SEQ_LENGTH, AA_LIST, AA_TO_IDX

# Import benchmark config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_config import (
    CLINICAL_MULTILABEL, ESM2_650M_NPZ, DRUGS, HIV2_WT_PR
)


def load_clinical_targets():
    """
    Load clinical per-drug integer penalty scores.
    Returns: DataFrame with columns ['seq_id', 'sequence'] + DRUGS
    """
    df = pd.read_csv(CLINICAL_MULTILABEL)
    
    # Identify the sequence column (might be named differently)
    seq_col = None
    for col in ["sequence", "seq", "amino_acid", "aa_seq", "PR_sequence", "prot_seq"]:
        if col in df.columns:
            seq_col = col
            break
    if seq_col is None:
        # Try first column that looks like sequences
        for col in df.columns:
            if df[col].dtype == object and df[col].str.len().median() > 50:
                seq_col = col
                break
    
    # Identify the ID column
    id_col = None
    for col in ["seq_id", "id", "sample_id", "patient_id", "sequence_id"]:
        if col in df.columns:
            id_col = col
            break
    if id_col is None:
        df["seq_id"] = range(len(df))
        id_col = "seq_id"
    
    # Ensure all drug columns exist
    drug_cols_found = [d for d in DRUGS if d in df.columns]
    if len(drug_cols_found) < len(DRUGS):
        print(f"Warning: Only found {len(drug_cols_found)} of {len(DRUGS)} drug columns")
        print(f"  Found: {drug_cols_found}")
        print(f"  Available columns: {list(df.columns)}")
    
    result = df[[id_col, seq_col] + drug_cols_found].copy()
    result.columns = ["seq_id", "sequence"] + drug_cols_found
    
    # Ensure targets are numeric
    for drug in drug_cols_found:
        result[drug] = pd.to_numeric(result[drug], errors="coerce").fillna(0).astype(int)
    
    print(f"Loaded {len(result)} clinical sequences, {len(drug_cols_found)} drugs")
    print(f"  Unique sequences: {result['seq_id'].nunique()}")
    print(f"  Target range per drug:")
    for drug in drug_cols_found:
        nz = (result[drug] > 0).sum()
        print(f"    {drug}: 0-{result[drug].max()} ({nz} resistant, {len(result)-nz} susceptible)")
    
    return result


def load_esm2():
    """
    Load ESM-2 650M embeddings.
    Returns: dict of {sequence_string: 1280-D mean-pooled vector}
    """
    data = np.load(ESM2_650M_NPZ, allow_pickle=True)
    
    # Discover structure
    keys = list(data.keys())
    print(f"ESM-2 NPZ keys: {keys}")
    
    # Find mean-pooled and per-residue
    mean_pooled = None
    per_residue = None
    sequences = None
    
    for key in keys:
        arr = data[key]
        if arr.ndim == 2 and arr.shape[1] == 1280:
            mean_pooled = arr
            print(f"  Found mean-pooled: {key} shape={arr.shape}")
        elif arr.ndim == 3 and arr.shape[2] == 1280:
            per_residue = arr
            print(f"  Found per-residue: {key} shape={arr.shape}")
        elif arr.ndim == 1 and arr.dtype.kind in ('U', 'S', 'O'):
            sequences = arr
    
    if mean_pooled is None:
        raise ValueError(f"No mean-pooled (N×1280) array found in {ESM2_650M_NPZ}")
    
    # If sequences not stored, we can't map; return arrays directly
    if sequences is None:
        print("  Warning: No sequence mapping found in NPZ. Returning arrays directly.")
        return mean_pooled, per_residue
    
    # Build sequence → embedding mapping
    seq_to_vec = {}
    for i, seq in enumerate(sequences):
        seq_str = str(seq)
        seq_to_vec[seq_str] = mean_pooled[i]
    
    print(f"  Mapped {len(seq_to_vec)} sequences to ESM-2 embeddings")
    return seq_to_vec


def load_per_residue():
    """Load per-residue ESM-2 embeddings for attention/max pooling."""
    data = np.load(ESM2_650M_NPZ, allow_pickle=True)
    keys = list(data.keys())
    
    for key in keys:
        arr = data[key]
        if arr.ndim == 3 and arr.shape[2] == 1280:
            print(f"Loaded per-residue: shape={arr.shape}")
            return arr
    
    print("Warning: No per-residue data found")
    return None


def build_mutation_matrix(sequences, wt_seq=HIV2_WT_PR):
    """
    Build binary mutation matrix for each sequence.
    For each position: 20 channels, 1 if AA != WT, 0 if AA == WT.
    
    Returns: np.ndarray of shape (N, SEQ_LENGTH, 20)
    """
    N = len(sequences)
    mat = np.zeros((N, SEQ_LENGTH, 20), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq = str(seq).strip()
        for pos in range(min(len(seq), SEQ_LENGTH)):
            aa = seq[pos].upper()
            if aa in AA_TO_IDX and aa != wt_seq[pos].upper():
                mat[i, pos, AA_TO_IDX[aa]] = 1.0
    
    return mat.reshape(N, -1)  # Flatten to (N, 1980)


def build_onehot_matrix(sequences):
    """
    Build one-hot encoding of sequences.
    
    Returns: np.ndarray of shape (N, SEQ_LENGTH * 20)
    """
    N = len(sequences)
    mat = np.zeros((N, SEQ_LENGTH, 20), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq = str(seq).strip()
        for pos in range(min(len(seq), SEQ_LENGTH)):
            aa = seq[pos].upper()
            if aa in AA_TO_IDX:
                mat[i, pos, AA_TO_IDX[aa]] = 1.0
    
    return mat.reshape(N, -1)  # Flatten to (N, 1980)


def match_esm2_to_clinical(clinical_df, esm2_data):
    """
    Match ESM-2 embeddings to clinical sequences by sequence identity.
    Returns: np.ndarray of shape (N, 1280)
    """
    if isinstance(esm2_data, dict):
        # Dict mode: sequence → vector
        embeddings = []
        matched = 0
        for seq in clinical_df["sequence"]:
            seq_str = str(seq).strip()
            if seq_str in esm2_data:
                embeddings.append(esm2_data[seq_str])
                matched += 1
            else:
                # Try without whitespace
                found = False
                for key in esm2_data:
                    if key.replace(" ", "").replace("\n", "") == seq_str.replace(" ", "").replace("\n", ""):
                        embeddings.append(esm2_data[key])
                        matched += 1
                        found = True
                        break
                if not found:
                    embeddings.append(np.zeros(1280))
        
        embeddings = np.array(embeddings)
    else:
        # Array mode: assume same ordering as clinical_df
        embeddings = esm2_data[:len(clinical_df)]
    
    print(f"Matched {matched if isinstance(esm2_data, dict) else len(clinical_df)}/{len(clinical_df)} sequences to ESM-2 embeddings")
    return embeddings


def load_all(drugs=None):
    """
    Master loading function. Returns everything needed for the benchmark.
    
    Returns dict:
        clinical_df: DataFrame with seq_id, sequence, drug targets
        esm2_mean: np.ndarray (N, 1280) mean-pooled
        per_residue: np.ndarray or None (N, 99, 1280)
        mutation_features: np.ndarray (N, 1980)
        onehot_features: np.ndarray (N, 1980)
        targets: dict of {drug: np.ndarray (N,)}
        drug_names: list of drug column names
    """
    if drugs is None:
        drugs = DRUGS
    
    print("=" * 60)
    print("LOADING DATA FOR BENCHMARK")
    print("=" * 60)
    
    # Load clinical
    clinical_df = load_clinical_targets()
    
    # Load ESM-2
    esm2_mean, per_residue = None, None
    esm2_raw = load_esm2()
    if isinstance(esm2_raw, tuple):
        esm2_mean, per_residue = esm2_raw
    else:
        esm2_mean = esm2_raw
    
    # Match ESM-2 to clinical
    esm2_matched = match_esm2_to_clinical(clinical_df, esm2_mean)
    
    # Build mutation and one-hot features
    sequences = clinical_df["sequence"].tolist()
    mutation_features = build_mutation_matrix(sequences)
    onehot_features = build_onehot_matrix(sequences)
    
    # Extract targets
    available_drugs = [d for d in drugs if d in clinical_df.columns]
    targets = {}
    for drug in available_drugs:
        targets[drug] = clinical_df[drug].values.astype(float)
    
    print(f"\nFinal data shapes:")
    print(f"  ESM-2 mean-pooled: {esm2_matched.shape}")
    if per_residue is not None:
        print(f"  Per-residue: {per_residue.shape}")
    print(f"  Mutation features: {mutation_features.shape}")
    print(f"  One-hot features: {onehot_features.shape}")
    print(f"  Target drugs: {available_drugs}")
    print("=" * 60)
    
    return {
        "clinical_df": clinical_df,
        "esm2_mean": esm2_matched,
        "per_residue": per_residue,
        "mutation_features": mutation_features,
        "onehot_features": onehot_features,
        "targets": targets,
        "drug_names": available_drugs,
    }


if __name__ == "__main__":
    data = load_all()
    print(f"\nSample target distribution for ATV/r:")
    vals = data["targets"]["ATV/r"]
    print(f"  Mean={vals.mean():.2f}, Median={np.median(vals):.1f}, "
          f"Range=[{vals.min():.0f}, {vals.max():.0f}], "
          f"Nonzero={(vals>0).sum()}/{len(vals)}")
