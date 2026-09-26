"""
Main benchmark runner for HIV-2 Drug Resistance Prediction.
Runs ALL combinations: methods × features × splits × drugs.

Usage:
    python RESCUE/BENCHMARK/run_benchmark.py
    python RESCUE/BENCHMARK/run_benchmark.py --methods T4_Ridge T4_XGB --features F3
    python RESCUE/BENCHMARK/run_benchmark.py --quick  # Fast subset for testing
"""
import sys
import json
import time
import argparse
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "opencode_shared"))

from benchmark_config import (
    DRUGS, FEATURES, MODELS, CV_SCHEMES, RANDOM_SEED, RESULTS_DIR,
    OOF_DIR
)
from data_loader import load_all
from feature_builders import build_features
from model_builders import build_model
from cross_validation import get_cv_splits
from evaluator import evaluate, evaluate_all_drugs, save_results, format_results_table


def train_and_predict(model_name, feature_matrix, targets, train_idx, test_idx):
    """
    Train a model on train_idx, predict on test_idx.
    
    Returns: (y_true, y_pred) for test set
    """
    X_train = feature_matrix[train_idx]
    y_train = targets[train_idx]
    X_test = feature_matrix[test_idx]
    y_true = targets[test_idx]
    
    try:
        model = build_model(model_name, n_features=X_train.shape[1])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Clip predictions to non-negative
        y_pred = np.clip(y_pred, 0, None)
        
        return y_true, y_pred
    
    except Exception as e:
        print(f"  ERROR in {model_name}: {e}")
        return None, None


def run_single_combination(model_name, feature_id, feature_matrix, 
                           targets_dict, cv_splits, drug_names):
    """
    Run one (model × feature × CV) combination across all drugs.
    """
    oof_preds = {drug: {"y_true": [], "y_pred": []} for drug in drug_names}
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        for drug in drug_names:
            y_true, y_pred = train_and_predict(
                model_name, feature_matrix, targets_dict[drug],
                train_idx, test_idx
            )
            if y_true is not None:
                oof_preds[drug]["y_true"].extend(y_true.tolist())
                oof_preds[drug]["y_pred"].extend(y_pred.tolist())
    
    # Evaluate
    results = {}
    for drug in drug_names:
        yt = np.array(oof_preds[drug]["y_true"])
        yp = np.array(oof_preds[drug]["y_pred"])
        if len(yt) > 0:
            results[drug] = evaluate(yt, yp, method_name=f"{model_name}+{feature_id}", drug_name=drug)
    
    # Compute mean
    pearson_vals = [results[d]["pearson_r"] for d in drug_names if d in results and "pearson_r" in results[d]]
    if pearson_vals:
        results["MEAN"] = {
            "pearson_r": round(float(np.mean(pearson_vals)), 4),
            "pearson_std": round(float(np.std(pearson_vals)), 4),
            "method": f"{model_name}+{feature_id}",
            "n_drugs": len(pearson_vals),
        }
    
    return results, oof_preds


def main():
    parser = argparse.ArgumentParser(description="HIV-2 Drug Resistance Benchmark")
    parser.add_argument("--methods", nargs="+", default=None,
                       help="Model names to run (default: all)")
    parser.add_argument("--features", nargs="+", default=None,
                       help="Feature IDs to use (default: F1-F5)")
    parser.add_argument("--splits", nargs="+", default=["S1_GroupKFold5"],
                       help="CV schemes (default: S1)")
    parser.add_argument("--drugs", nargs="+", default=None,
                       help="Drugs to evaluate (default: all)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick subset: Ridge + XGB on ESM-2 features only")
    parser.add_argument("--jobs", type=int, default=1,
                       help="Number of parallel jobs")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON path")
    
    args = parser.parse_args()
    
    # ── Configuration ──────────────────────────────────────────────────
    print("=" * 70)
    print("HIV-2 DRUG RESISTANCE BENCHMARK")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    
    # Determine methods
    if args.quick:
        methods = ["T4_Ridge", "T4_XGB", "T4_RF", "T5_ensemble"]
        features = ["F3"]  # ESM-2 mean only
        print("QUICK MODE: Ridge, XGB, RF, Ensemble on ESM-2")
    else:
        methods = args.methods or [
            "T0_mean", "T1_hiv2eu",
            "T4_Ridge", "T4_EN", "T4_XGB", "T4_RF",
            "T5_ensemble"
        ]
        features = args.features or ["F1", "F2", "F3", "F4", "F5"]
    
    cv_schemes = args.splits
    drugs = args.drugs or DRUGS
    
    print(f"\nMethods: {methods}")
    print(f"Features: {features}")
    print(f"CV Schemes: {cv_schemes}")
    print(f"Drugs: {drugs}")
    
    # ── Load Data ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    
    data = load_all(drugs=drugs)
    
    # ── Build Features ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: BUILDING FEATURES")
    print("=" * 70)
    
    sequences = data["clinical_df"]["sequence"].tolist()
    esm2_data = {
        "mean_pooled": data["esm2_mean"],
        "per_residue": data["per_residue"],
    }
    
    feature_matrices = build_features(sequences, esm2_data, feature_ids=features)
    
    # ── Run Benchmark ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: RUNNING BENCHMARK")
    print("=" * 70)
    
    all_results = {}
    all_oof = {}
    
    for scheme_name in cv_schemes:
        print(f"\n--- CV Scheme: {scheme_name} ---")
        
        n_samples = len(data["clinical_df"])
        cv_splits = get_cv_splits(
            scheme_name, n_samples, data["clinical_df"],
            data["targets"], drugs
        )
        
        if cv_splits is None:
            print(f"  Skipping {scheme_name} (not a fold-based CV)")
            continue
        
        scheme_results = {}
        
        for method_name in methods:
            for feature_id in features:
                combo_name = f"{method_name}+{feature_id}"
                print(f"\n  Running: {combo_name}", end="", flush=True)
                t0 = time.time()
                
                if feature_id not in feature_matrices:
                    print(f" SKIPPED (feature {feature_id} not available)")
                    continue
                
                try:
                    results, oof_preds = run_single_combination(
                        method_name, feature_id, feature_matrices[feature_id],
                        data["targets"], cv_splits, drugs
                    )
                    
                    scheme_results[combo_name] = results
                    all_oof[f"{scheme_name}_{combo_name}"] = oof_preds
                    
                    mean_r = results.get("MEAN", {}).get("pearson_r", np.nan)
                    elapsed = time.time() - t0
                    print(f" → R={mean_r:.3f} ({elapsed:.1f}s)")
                
                except Exception as e:
                    print(f" → ERROR: {e}")
                    traceback.print_exc()
        
        all_results[scheme_name] = scheme_results
        
        # Print table
        print(f"\n{'='*70}")
        print(f"RESULTS TABLE — {scheme_name}")
        print(f"{'='*70}")
        print(format_results_table(scheme_results, drugs))
    
    # ── Save Results ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: SAVING RESULTS")
    print("=" * 70)
    
    output_path = args.output or str(RESULTS_DIR / "benchmark_manifest.json")
    save_results(all_results, output_path)
    
    # Save OOF predictions
    for combo_key, oof in all_oof.items():
        oof_path = OOF_DIR / f"{combo_key}.npz"
        # Save as compressed npz
        save_dict = {}
        for drug in drugs:
            if drug in oof:
                save_dict[f"{drug}_true"] = np.array(oof[drug]["y_true"])
                save_dict[f"{drug}_pred"] = np.array(oof[drug]["y_pred"])
        if save_dict:
            np.savez_compressed(oof_path, **save_dict)
    
    print(f"\nOOF predictions saved to {OOF_DIR}")
    
    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    
    # Find best method
    best_method = None
    best_r = -1
    for scheme, results in all_results.items():
        for method, drug_results in results.items():
            r = drug_results.get("MEAN", {}).get("pearson_r", -1)
            if r > best_r:
                best_r = r
                best_method = f"{method} ({scheme})"
    
    print(f"Best method: {best_method} → Mean Pearson R = {best_r:.3f}")
    print(f"Completed: {datetime.now().isoformat()}")
    print(f"Results: {output_path}")
    
    return all_results


if __name__ == "__main__":
    results = main()
