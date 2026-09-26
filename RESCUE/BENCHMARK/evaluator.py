"""
Unified evaluation for the HIV-2 Drug Resistance Benchmark.
8 metrics, bootstrap CIs, permutation tests.
"""
import sys
import json
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent / "opencode_shared"))

from utils import compute_metrics, bootstrap_ci


def evaluate(y_true, y_pred, method_name="", drug_name="", threshold=0.5):
    """
    Evaluate a single prediction set with all metrics.
    
    Args:
        y_true: ground truth (N,)
        y_pred: predictions (N,)
        method_name: for logging
        drug_name: for logging
        threshold: binary classification threshold (for resistance)
    
    Returns: dict with all metrics + CIs
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Remove NaN/Inf
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    n = len(y_true)
    
    if n < 10:
        return {"error": f"Too few samples: {n}", "n": n}
    
    # Regression metrics
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_rho, spearman_p = stats.spearmanr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R² (coefficient of determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    # p±1 accuracy (within 1 unit of integer target)
    p1_acc = np.mean(np.abs(y_true - y_pred) <= threshold)
    
    # Bootstrap CIs for Pearson R
    try:
        ci_low, ci_high = bootstrap_ci(y_true, y_pred, metric="pearson",
                                       n_bootstrap=1000, confidence=0.95)
    except Exception:
        ci_low, ci_high = np.nan, np.nan
    
    # Classification metrics (if applicable)
    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)
    
    # Sensitivity and Specificity
    tp = ((y_pred_binary == 1) & (y_true_binary == 1)).sum()
    fn = ((y_pred_binary == 0) & (y_true_binary == 1)).sum()
    tn = ((y_pred_binary == 0) & (y_true_binary == 0)).sum()
    fp = ((y_pred_binary == 1) & (y_true_binary == 0)).sum()
    
    sensitivity = tp / (tp + fn + 1e-10)  # Recall for resistant class
    specificity = tn / (tn + fp + 1e-10)  # Recall for susceptible class
    
    # AUROC
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(y_true_binary, y_pred)
    except Exception:
        auroc = np.nan
    
    result = {
        "n": n,
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": float(pearson_p),
        "spearman_rho": round(float(spearman_rho), 4),
        "spearman_p": float(spearman_p),
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 4),
        "p1_accuracy": round(float(p1_acc), 4),
        "ci_95_low": round(float(ci_low), 4),
        "ci_95_high": round(float(ci_high), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "auroc": round(float(auroc), 4) if not np.isnan(auroc) else None,
        "method": method_name,
        "drug": drug_name,
    }
    
    return result


def permutation_test(y_true, y_pred_a, y_pred_b, metric="pearson",
                     n_permutations=1000, seed=42):
    """
    Paired permutation test between two methods.
    Tests whether method A is significantly better than method B.
    
    Returns: (delta_metric, p_value)
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred_a, dtype=float)
    y_pred_b = np.asarray(y_pred_b, dtype=float)
    
    # Remove NaN
    valid = np.isfinite(y_true) & np.isfinite(y_pred_a) & np.isfinite(y_pred_b)
    y_true = y_true[valid]
    y_pred_a = y_pred_a[valid]
    y_pred_b = y_pred_b[valid]
    
    def _metric(y_true, y_pred):
        if metric == "pearson":
            r, _ = stats.pearsonr(y_true, y_pred)
            return r
        elif metric == "spearman":
            r, _ = stats.spearmanr(y_true, y_pred)
            return r
        elif metric == "rmse":
            return -np.sqrt(np.mean((y_true - y_pred) ** 2))  # negative for "higher is better"
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    observed_delta = _metric(y_true, y_pred_a) - _metric(y_true, y_pred_b)
    
    count = 0
    for _ in range(n_permutations):
        # Randomly flip signs of residuals
        signs = rng.choice([-1, 1], size=len(y_true))
        y_perm_b = y_pred_b + signs * (y_pred_b - y_true)
        y_perm_b = np.clip(y_perm_b, 0, None)  # keep non-negative
        
        perm_delta = _metric(y_true, y_pred_a) - _metric(y_true, y_perm_b)
        if perm_delta >= observed_delta:
            count += 1
    
    p_value = count / n_permutations
    return observed_delta, p_value


def evaluate_all_drugs(y_true_dict, y_pred_dict, method_name, drug_names):
    """
    Evaluate across all drugs and compute mean.
    
    Args:
        y_true_dict: {drug: y_true array}
        y_pred_dict: {drug: y_pred array}
        method_name: string identifier
        drug_names: list of drug names
    
    Returns: dict with per-drug results + MEAN
    """
    results = {}
    all_pearson = []
    all_spearman = []
    
    for drug in drug_names:
        if drug in y_true_dict and drug in y_pred_dict:
            result = evaluate(
                y_true_dict[drug], y_pred_dict[drug],
                method_name=method_name, drug_name=drug
            )
            results[drug] = result
            if "pearson_r" in result:
                all_pearson.append(result["pearson_r"])
                all_spearman.append(result["spearman_rho"])
    
    # Compute means
    if all_pearson:
        results["MEAN"] = {
            "pearson_r": round(float(np.mean(all_pearson)), 4),
            "pearson_std": round(float(np.std(all_pearson)), 4),
            "spearman_rho": round(float(np.mean(all_spearman)), 4),
            "spearman_std": round(float(np.std(all_spearman)), 4),
            "n_drugs": len(all_pearson),
            "method": method_name,
        }
    
    return results


def format_results_table(results_dict, drug_names):
    """
    Format results as a pretty-printed table for console output.
    
    Args:
        results_dict: {method_name: {drug: metrics_dict, "MEAN": ...}}
        drug_names: list of drugs
    
    Returns: string table
    """
    header = f"{'Method':<25} " + " ".join(f"{d:>8}" for d in drug_names) + f" {'MEAN':>8}"
    sep = "-" * len(header)
    
    lines = [header, sep]
    
    for method, drug_results in sorted(results_dict.items()):
        if method.startswith("_"):
            continue
        
        row = f"{method:<25} "
        for drug in drug_names:
            r = drug_results.get(drug, {}).get("pearson_r", np.nan)
            row += f"{r:>8.3f} "
        
        mean_r = drug_results.get("MEAN", {}).get("pearson_r", np.nan)
        row += f"{mean_r:>8.3f}"
        lines.append(row)
    
    lines.append(sep)
    return "\n".join(lines)


def save_results(results_dict, output_path):
    """Save full results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        "benchmark_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "results": results_dict,
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    # Quick test
    y_true = np.random.uniform(0, 5, 100)
    y_pred = y_true + np.random.normal(0, 0.5, 100)
    
    result = evaluate(y_true, y_pred, "Test", "ATV/r")
    print("Test evaluation:")
    for k, v in result.items():
        print(f"  {k}: {v}")
