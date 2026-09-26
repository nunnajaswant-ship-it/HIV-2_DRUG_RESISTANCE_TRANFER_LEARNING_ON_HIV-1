"""
Cross-validation schemes for the HIV-2 Drug Resistance Benchmark.
GroupKFold, StratifiedKFold, and Leave-One-Drug-Out.
"""
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold


def get_groups(clinical_df):
    """
    Create groups for GroupKFold based on sequence identity.
    Sequences that are >95% identical get the same group.
    For simplicity, group by exact sequence (since we confirmed 653 unique in 654 rows).
    """
    # Group by sequence string
    sequences = clinical_df["sequence"].astype(str).values
    unique_seqs = list(set(sequences))
    seq_to_group = {s: i for i, s in enumerate(unique_seqs)}
    groups = np.array([seq_to_group[s] for s in sequences])
    
    print(f"GroupKFold: {len(groups)} samples, {len(unique_seqs)} unique groups")
    return groups


def get_stratification_bins(targets, drug_name, n_bins=5):
    """
    Create stratification bins for a drug's target variable.
    Bins: 0 (susceptible), 1-2, 3-5, 6-10, 11+ (resistant tiers).
    """
    y = targets[drug_name]
    
    bins = np.zeros(len(y), dtype=int)
    bins[y > 0] = 1
    bins[y > 2] = 2
    bins[y > 5] = 3
    bins[y > 10] = 4
    
    return bins


def group_kfold_cv(n_samples, groups, n_splits=5, random_state=42):
    """
    GroupKFold CV ensuring no group leakage.
    
    Yields: (train_idx, test_idx) pairs
    """
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(np.zeros(n_samples), groups=groups))
    
    print(f"GroupKFold({n_splits}): {len(splits)} folds")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Fold {i}: train={len(train_idx)}, test={len(test_idx)}")
    
    return splits


def stratified_kfold_cv(y, n_splits=5, random_state=42):
    """
    StratifiedKFold using binary resistance class (0 vs >0).
    Falls back to KFold if stratification fails (e.g., too few minority samples).
    """
    y_binary = (y > 0).astype(int)
    
    # Check if stratification is possible
    unique, counts = np.unique(y_binary, return_counts=True)
    if len(unique) < 2 or min(counts) < n_splits:
        print(f"Warning: Stratification impossible (class counts={dict(zip(unique, counts))})")
        print("  Falling back to KFold")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(kf.split(np.zeros(len(y))))
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(skf.split(np.zeros(len(y)), y_binary))
    
    print(f"StratifiedKFold({n_splits}): {len(splits)} folds")
    return splits


def leave_one_drug_out_cv(drug_names, target_drug):
    """
    Leave-One-Drug-Out CV.
    Uses a dummy split since drug selection is done per-drug.
    
    Returns: placeholder that signals "use all drugs except target_drug"
    """
    train_drugs = [d for d in drug_names if d != target_drug]
    print(f"Leave-One-Drug-Out: train on {train_drugs}, test on {target_drug}")
    return train_drugs


def get_cv_splits(scheme_name, n_samples, clinical_df, targets, drug_names,
                  drug_for_stratification="ATV/r"):
    """
    Master function to get CV splits by scheme name.
    
    Returns: list of (train_idx, test_idx) pairs
    """
    if scheme_name == "S1_GroupKFold5":
        groups = get_groups(clinical_df)
        return group_kfold_cv(n_samples, groups, n_splits=5)
    
    elif scheme_name == "S2_StratifiedKFold5":
        # Use first drug with enough resistance for stratification
        y = targets[drug_for_stratification]
        return stratified_kfold_cv(y, n_splits=5)
    
    elif scheme_name == "S3_LeaveDrugOut":
        # Not a CV scheme per se; handled differently in the runner
        return None
    
    else:
        raise ValueError(f"Unknown CV scheme: {scheme_name}")


if __name__ == "__main__":
    print("CV schemes loaded:")
    print("  S1_GroupKFold5 — GroupKFold(5) by sequence identity")
    print("  S2_StratifiedKFold5 — StratifiedKFold(5) by resistance class")
    print("  S3_LeaveDrugOut — Leave-One-Drug-Out")
