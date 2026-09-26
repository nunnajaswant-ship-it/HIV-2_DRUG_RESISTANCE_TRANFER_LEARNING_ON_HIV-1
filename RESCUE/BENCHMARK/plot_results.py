"""
Publication figures for the HIV-2 Drug Resistance Benchmark.
Generates all figures for the paper.
"""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_config import DRUGS, DRUG_NAMES, RESULTS_DIR, FIGURES_DIR


def set_style():
    """Set publication-quality plot style."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (8, 5),
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    sns.set_palette("Set2")


def fig1_method_comparison(results_dict, save_path=None):
    """
    Fig 1: Bar chart of mean Pearson R across all drugs for each method.
    Error bars = 95% CI.
    """
    set_style()
    
    methods = []
    means = []
    ci_lows = []
    ci_highs = []
    
    for method, drug_results in sorted(results_dict.items()):
        if method.startswith("_") or "MEAN" not in drug_results:
            continue
        methods.append(method)
        means.append(drug_results["MEAN"]["pearson_r"])
        
        # Average CI across drugs
        ci_low_vals = [drug_results[d].get("ci_95_low", np.nan) for d in DRUGS if d in drug_results]
        ci_high_vals = [drug_results[d].get("ci_95_high", np.nan) for d in DRUGS if d in drug_results]
        ci_lows.append(np.nanmean(ci_low_vals))
        ci_highs.append(np.nanmean(ci_highs_vals if ci_highs_vals else ci_high_vals))
    
    if not methods:
        print("No results to plot for Fig 1")
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, means, xerr=[np.array(means) - np.array(ci_lows), 
                                         np.array(ci_highs) - np.array(means)],
                   capsize=3, edgecolor="black", linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel("Mean Pearson R (across 8 PI drugs)")
    ax.set_title("Method Comparison — HIV-2 PI Resistance Prediction")
    ax.set_xlim(0, max(means) * 1.2 if means else 1)
    ax.axvline(x=0, color="gray", linewidth=0.5)
    
    # Add value labels
    for i, (mean, method) in enumerate(zip(means, methods)):
        ax.text(mean + 0.01, i, f"{mean:.3f}", va="center", fontsize=9)
    
    plt.tight_layout()
    
    save_path = save_path or FIGURES_DIR / "fig1_method_comparison.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig2_drug_heatmap(results_dict, save_path=None):
    """
    Fig 2: Heatmap of Pearson R for each method × each drug.
    """
    set_style()
    
    methods = [m for m in sorted(results_dict.keys()) if not m.startswith("_") and "MEAN" in results_dict[m]]
    drugs = [d for d in DRUGS if any(d in results_dict[m] for m in methods)]
    
    if not methods or not drugs:
        print("No results to plot for Fig 2")
        return
    
    matrix = np.zeros((len(methods), len(drugs)))
    for i, method in enumerate(methods):
        for j, drug in enumerate(drugs):
            matrix[i, j] = results_dict[method].get(drug, {}).get("pearson_r", np.nan)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(methods) * 0.4)))
    
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=drugs, yticklabels=methods,
                vmin=0, vmax=1, center=0.5, ax=ax,
                linewidths=0.5, linecolor="gray")
    
    ax.set_title("Per-Drug Pearson R — Method × Drug Heatmap")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    save_path = save_path or FIGURES_DIR / "fig2_drug_heatmap.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig3_feature_ablation(ablation_results, save_path=None):
    """
    Fig 3: Bar chart of feature ablation results.
    """
    set_style()
    
    if "feature_ablation" not in ablation_results:
        print("No feature ablation results for Fig 3")
        return
    
    data = ablation_results["feature_ablation"]
    features = list(data.keys())
    r_values = [data[f] for f in features]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    colors = sns.color_palette("Set2", len(features))
    bars = ax.bar(features, r_values, color=colors, edgecolor="black", linewidth=0.5)
    
    for bar, val in zip(bars, r_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    
    ax.set_ylabel("Pearson R")
    ax.set_title("Feature Ablation — Which Features Matter Most?")
    ax.set_ylim(0, max(r_values) * 1.15 if r_values else 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    save_path = save_path or FIGURES_DIR / "fig3_feature_ablation.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig4_learning_curve(ablation_results, save_path=None):
    """
    Fig 4: Learning curve — performance vs training data size.
    """
    set_style()
    
    if "learning_curve" not in ablation_results:
        print("No learning curve data for Fig 4")
        return
    
    data = ablation_results["learning_curve"]
    fractions = sorted([float(k) for k in data.keys()])
    means = [data[str(f)][0] if isinstance(data[str(f)], (list, tuple)) else data[str(f)] for f in fractions]
    stds = [data[str(f)][1] if isinstance(data[str(f)], (list, tuple)) else 0 for f in fractions]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    ax.fill_between(fractions, np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds), alpha=0.2, color="steelblue")
    ax.plot(fractions, means, "o-", color="steelblue", linewidth=2, markersize=6)
    
    ax.set_xlabel("Fraction of Training Data")
    ax.set_ylabel("Pearson R")
    ax.set_title("Learning Curve — ESM-2 + Ridge on HIV-2 PI Resistance")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, max(means) * 1.15 if means else 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = save_path or FIGURES_DIR / "fig4_learning_curves.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig5_drug_difficulty(ablation_results, save_path=None):
    """
    Fig 5: Per-drug difficulty breakdown.
    """
    set_style()
    
    if "drug_ablation" not in ablation_results:
        print("No drug ablation results for Fig 5")
        return
    
    data = ablation_results["drug_ablation"]
    drugs = [d for d in DRUGS if d in data]
    
    if not drugs:
        return
    
    r_vals = [data[d].get("pearson_r", 0) for d in drugs]
    pct_resistant = [data[d].get("target_pct_resistant", 0) for d in drugs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left: Pearson R per drug
    colors = sns.color_palette("Set2", len(drugs))
    ax1.bar(drugs, r_vals, color=colors, edgecolor="black", linewidth=0.5)
    for i, (d, r) in enumerate(zip(drugs, r_vals)):
        ax1.text(i, r + 0.01, f"{r:.3f}", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("Pearson R")
    ax1.set_title("Performance by Drug")
    ax1.set_ylim(0, max(r_vals) * 1.15)
    plt.sca(ax1)
    plt.xticks(rotation=45, ha="right")
    
    # Right: % resistant per drug
    ax2.bar(drugs, pct_resistant, color="coral", edgecolor="black", linewidth=0.5)
    for i, (d, p) in enumerate(zip(drugs, pct_resistant)):
        ax2.text(i, p + 0.5, f"{p:.1f}%", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("% Resistant Sequences")
    ax2.set_title("Class Balance by Drug")
    plt.sca(ax2)
    plt.xticks(rotation=45, ha="right")
    
    plt.suptitle("Drug Difficulty Analysis", fontsize=13, y=1.02)
    plt.tight_layout()
    
    save_path = save_path or FIGURES_DIR / "fig5_drug_difficulty.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig6_parity_plots(oof_dir, methods=None, drugs=None, save_path=None):
    """
    Fig 6: Parity plots (predicted vs actual) for top methods.
    """
    set_style()
    
    if drugs is None:
        drugs = DRUGS[:4]  # First 4 drugs
    
    if methods is None:
        # Find available OOF files
        oof_files = list(oof_dir.glob("*.npz"))
        methods = [f.stem for f in oof_files[:4]]
    
    n_methods = min(len(methods), 3)
    n_drugs = min(len(drugs), 4)
    
    if n_methods == 0 or n_drugs == 0:
        print("No OOF data for parity plots")
        return
    
    fig, axes = plt.subplots(n_methods, n_drugs, figsize=(3*n_drugs, 3*n_methods))
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    if n_drugs == 1:
        axes = axes.reshape(-1, 1)
    
    for i, method in enumerate(methods[:n_methods]):
        for j, drug in enumerate(drugs[:n_drugs]):
            ax = axes[i, j]
            
            oof_path = oof_dir / f"{method}.npz"
            if oof_path.exists():
                data = np.load(oof_path)
                true_key = f"{drug}_true"
                pred_key = f"{drug}_pred"
                
                if true_key in data and pred_key in data:
                    y_true = data[true_key]
                    y_pred = data[pred_key]
                    
                    ax.scatter(y_true, y_pred, alpha=0.3, s=10, c="steelblue")
                    
                    # Perfect prediction line
                    lim = max(y_true.max(), y_pred.max()) * 1.1
                    ax.plot([0, lim], [0, lim], "k--", linewidth=0.5, alpha=0.5)
                    
                    # Pearson R label
                    from scipy.stats import pearsonr
                    r, _ = pearsonr(y_true, y_pred)
                    ax.text(0.05, 0.95, f"R={r:.3f}", transform=ax.transAxes,
                           fontsize=9, va="top", fontweight="bold")
            
            if i == 0:
                ax.set_title(drug, fontsize=9)
            if j == 0:
                ax.set_ylabel(method, fontsize=8)
            if i == n_methods - 1:
                ax.set_xlabel("True", fontsize=8)
            ax.tick_params(labelsize=7)
    
    plt.suptitle("Parity Plots — Predicted vs Actual", fontsize=12, y=1.02)
    plt.tight_layout()
    
    save_path = save_path or FIGURES_DIR / "fig6_parity_plots.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def generate_all_figures(results_path=None, oof_dir=None, ablation_path=None):
    """Generate all publication figures."""
    if results_path is None:
        results_path = RESULTS_DIR / "benchmark_manifest.json"
    if oof_dir is None:
        oof_dir = RESULTS_DIR / "oof_predictions"
    if ablation_path is None:
        ablation_path = RESULTS_DIR / "ablations" / "ablation_results.json"
    
    # Load benchmark results
    if results_path.exists():
        with open(results_path) as f:
            results_data = json.load(f)
        # Use first CV scheme's results
        for scheme, scheme_results in results_data.get("results", {}).items():
            fig1_method_comparison(scheme_results)
            fig2_drug_heatmap(scheme_results)
            break
    else:
        print(f"Warning: {results_path} not found — skipping Figs 1-2")
    
    # Load ablation results
    if ablation_path.exists():
        with open(ablation_path) as f:
            ablation_data = json.load(f)
        fig3_feature_ablation(ablation_data)
        fig4_learning_curve(ablation_data)
        fig5_drug_difficulty(ablation_data)
    else:
        print(f"Warning: {ablation_path} not found — skipping Figs 3-5")
    
    # Parity plots
    fig6_parity_plots(oof_dir)
    
    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
