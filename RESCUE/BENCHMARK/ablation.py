"""
Ablation studies for the HIV-2 Drug Resistance Benchmark.
Feature ablation, model ablation, data ablation (learning curves), drug ablation.
"""
import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "opencode_shared"))

from benchmark_config import DRUGS, RESULTS_DIR, RANDOM_SEED
from data_loader import load_all
from feature_builders import build_features
from model_builders import build_model
from cross_validation import get_cv_splits, get_groups
from evaluator import evaluate
from sklearn.model_selection import GroupKFold


def feature_ablation(data, feature_matrices, drug="ATV/r", model_name="T4_Ridge"):
    """
    Test each feature representation independently.
    Returns: dict of {feature_id: pearson_r}
    """
    print(f"\n{'='*60}")
    print(f"FEATURE ABLATION — {drug} — {model_name}")
    print(f"{'='*60}")
    
    targets = data["targets"][drug]
    groups = get_groups(data["clinical_df"])
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.zeros(len(targets)), groups=groups))
    
    results = {}
    for fid, X in feature_matrices.items():
        oof_true, oof_pred = [], []
        
        for train_idx, test_idx in splits:
            model = build_model(model_name, n_features=X.shape[1])
            model.fit(X[train_idx], targets[train_idx])
            pred = np.clip(model.predict(X[test_idx]), 0, None)
            
            oof_true.extend(targets[test_idx].tolist())
            oof_pred.extend(pred.tolist())
        
        oof_true = np.array(oof_true)
        oof_pred = np.array(oof_pred)
        
        r = evaluate(oof_true, oof_pred, method_name=f"{fid}+{model_name}", drug_name=drug)
        results[fid] = r.get("pearson_r", np.nan)
        print(f"  {fid}: R = {results[fid]:.4f}")
    
    # Also test combinations
    combo_tests = [
        ("F3+F6", ["F3", "F6"]),  # ESM-2 + Biochem
        ("F3+F7", ["F3", "F7"]),  # ESM-2 + Pocket
        ("F3+F6+F7", ["F3", "F6", "F7"]),  # ESM-2 + Biochem + Pocket
    ]
    
    for combo_name, fids in combo_tests:
        available = [f for f in fids if f in feature_matrices]
        if len(available) < 2:
            continue
        
        X_combo = np.hstack([feature_matrices[f] for f in available])
        oof_true, oof_pred = [], []
        
        for train_idx, test_idx in splits:
            model = build_model(model_name, n_features=X_combo.shape[1])
            model.fit(X_combo[train_idx], targets[train_idx])
            pred = np.clip(model.predict(X_combo[test_idx]), 0, None)
            
            oof_true.extend(targets[test_idx].tolist())
            oof_pred.extend(pred.tolist())
        
        r = evaluate(np.array(oof_true), np.array(oof_pred),
                     method_name=combo_name, drug_name=drug)
        results[combo_name] = r.get("pearson_r", np.nan)
        print(f"  {combo_name}: R = {results[combo_name]:.4f}")
    
    return results


def model_ablation(data, esm2_features, drug="ATV/r"):
    """
    Test different models on the same ESM-2 features.
    Returns: dict of {model_name: pearson_r}
    """
    print(f"\n{'='*60}")
    print(f"MODEL ABLATION — {drug} — ESM-2 Features")
    print(f"{'='*60}")
    
    targets = data["targets"][drug]
    groups = get_groups(data["clinical_df"])
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.zeros(len(targets)), groups=groups))
    
    model_names = ["T4_Ridge", "T4_EN", "T4_XGB", "T4_RF", "T5_ensemble"]
    
    results = {}
    for mname in model_names:
        oof_true, oof_pred = [], []
        
        for train_idx, test_idx in splits:
            try:
                model = build_model(mname, n_features=esm2_features.shape[1])
                model.fit(esm2_features[train_idx], targets[train_idx])
                pred = np.clip(model.predict(esm2_features[test_idx]), 0, None)
                
                oof_true.extend(targets[test_idx].tolist())
                oof_pred.extend(pred.tolist())
            except Exception as e:
                print(f"  {mname}: ERROR — {e}")
                continue
        
        if len(oof_true) > 0:
            r = evaluate(np.array(oof_true), np.array(oof_pred),
                        method_name=mname, drug_name=drug)
            results[mname] = r.get("pearson_r", np.nan)
            print(f"  {mname}: R = {results[mname]:.4f}")
    
    return results


def learning_curve(data, feature_matrix, drug="ATV/r", model_name="T4_Ridge",
                   fractions=None, n_seeds=3):
    """
    Data-size ablation: train on N% of data.
    Returns: dict of {fraction: (mean_r, std_r)}
    """
    if fractions is None:
        fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    
    print(f"\n{'='*60}")
    print(f"LEARNING CURVE — {drug} — {model_name}")
    print(f"{'='*60}")
    
    targets = data["targets"][drug]
    groups = get_groups(data["clinical_df"])
    n_total = len(targets)
    
    results = {}
    for frac in fractions:
        n_train = max(int(n_total * frac), 30)  # minimum 30 samples
        
        all_r = []
        for seed in range(n_seeds):
            rng = np.random.RandomState(RANDOM_SEED + seed)
            
            # Sample fraction of groups
            unique_groups = np.unique(groups)
            n_groups = max(int(len(unique_groups) * frac), 5)
            selected_groups = rng.choice(unique_groups, size=n_groups, replace=False)
            mask = np.isin(groups, selected_groups)
            
            X_sub = feature_matrix[mask]
            y_sub = targets[mask]
            groups_sub = groups[mask]
            
            if len(np.unique(groups_sub)) < 2:
                continue
            
            gkf = GroupKFold(n_splits=min(5, len(np.unique(groups_sub))))
            splits = list(gkf.split(np.zeros(len(y_sub)), groups=groups_sub))
            
            for train_idx, test_idx in splits:
                model = build_model(model_name, n_features=X_sub.shape[1])
                model.fit(X_sub[train_idx], y_sub[train_idx])
                pred = np.clip(model.predict(X_sub[test_idx]), 0, None)
                
                r, _ = __import__('scipy').stats.pearsonr(y_sub[test_idx], pred)
                if np.isfinite(r):
                    all_r.append(r)
        
        if all_r:
            mean_r = np.mean(all_r)
            std_r = np.std(all_r)
            results[frac] = (mean_r, std_r)
            print(f"  {frac*100:5.1f}% data ({n_train:4d} samples): R = {mean_r:.4f} ± {std_r:.4f}")
    
    return results


def drug_ablation(data, feature_matrices, model_name="T4_Ridge", feature_id="F3"):
    """
    Per-drug analysis: difficulty, performance, class balance.
    Returns: dict of {drug: analysis_dict}
    """
    print(f"\n{'='*60}")
    print(f"DRUG ABLATION — {model_name}+{feature_id}")
    print(f"{'='*60}")
    
    X = feature_matrices[feature_id]
    groups = get_groups(data["clinical_df"])
    gkf = GroupKFold(n_splits=5)
    
    results = {}
    for drug in DRUGS:
        if drug not in data["targets"]:
            continue
        
        targets = data["targets"][drug]
        splits = list(gkf.split(np.zeros(len(targets)), groups=groups))
        
        oof_true, oof_pred = [], []
        for train_idx, test_idx in splits:
            model = build_model(model_name, n_features=X.shape[1])
            model.fit(X[train_idx], targets[train_idx])
            pred = np.clip(model.predict(X[test_idx]), 0, None)
            oof_true.extend(targets[test_idx].tolist())
            oof_pred.extend(pred.tolist())
        
        oof_true = np.array(oof_true)
        oof_pred = np.array(oof_pred)
        
        r = evaluate(oof_true, oof_pred, method_name=f"{model_name}+{feature_id}", drug_name=drug)
        
        n_resistant = (targets > 0).sum()
        n_susceptible = (targets == 0).sum()
        
        results[drug] = {
            **r,
            "n_resistant": int(n_resistant),
            "n_susceptible": int(n_susceptible),
            "target_mean": float(targets.mean()),
            "target_max": float(targets.max()),
            "target_pct_resistant": round(n_resistant / len(targets) * 100, 1),
        }
        
        print(f"  {drug}: R={r.get('pearson_r', np.nan):.4f}, "
              f"n_resist={n_resistant}/{len(targets)} ({results[drug]['target_pct_resistant']}%), "
              f"target_range=[0, {targets.max():.0f}]")
    
    return results


def run_all_ablations(data, feature_matrices, output_dir=None):
    """Run all ablation studies and save results."""
    if output_dir is None:
        output_dir = RESULTS_DIR / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # 1. Feature ablation (using first drug with resistance)
    drug_for_feat = "ATV/r"
    if drug_for_feat in data["targets"]:
        all_results["feature_ablation"] = feature_ablation(
            data, feature_matrices, drug=drug_for_feat, model_name="T4_Ridge"
        )
    
    # 2. Model ablation (ESM-2 features)
    if "F3" in feature_matrices:
        all_results["model_ablation"] = model_ablation(
            data, feature_matrices["F3"], drug=drug_for_feat
        )
    
    # 3. Learning curve (ESM-2 + Ridge)
    if "F3" in feature_matrices:
        all_results["learning_curve"] = learning_curve(
            data, feature_matrices["F3"], drug=drug_for_feat, model_name="T4_Ridge"
        )
    
    # 4. Drug ablation
    all_results["drug_ablation"] = drug_ablation(
        data, feature_matrices, model_name="T4_Ridge", feature_id="F3"
    )
    
    # Save
    output_path = output_dir / "ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nAblation results saved to {output_path}")
    return all_results


if __name__ == "__main__":
    data = load_all()
    
    sequences = data["clinical_df"]["sequence"].tolist()
    esm2_data = {"mean_pooled": data["esm2_mean"], "per_residue": data["per_residue"]}
    feature_matrices = build_features(sequences, esm2_data, feature_ids=["F1", "F2", "F3", "F4", "F5"])
    
    run_all_ablations(data, feature_matrices)
