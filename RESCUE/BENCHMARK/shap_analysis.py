"""
SHAP Explainability Analysis -- HIV-2 Drug Resistance Prediction
================================================================
Generates publication-quality SHAP analyses:
  1. Global feature importance (ESM-2 + mutation-level)
  2. Waterfall plots for individual resistant/susceptible sequences
  3. Dependence plots (top features vs predicted resistance)
  4. Mutation-level interpretability mapped to protease positions 1-99
  5. Cross-drug comparison heatmaps

Outputs:
  - fig_shap_global.png
  - fig_shap_waterfall.png
  - fig_shap_dependence.png
  - fig_shap_heatmap.png
  - shap_results.json

Usage:
  python RESCUE/BENCHMARK/shap_analysis.py
"""
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

# -- Path setup --------------------------------------------------------
BENCHMARK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(BENCHMARK_DIR.parent / "opencode_shared"))

from benchmark_config import (
    DRUGS, DRUG_NAMES, RESULTS_DIR, FIGURES_DIR, RANDOM_SEED,
    SEQ_LENGTH, HIV2_WT_PR, AA_LIST, AA_TO_IDX,
)

# data_loader may fail to import if opencode_shared/config.py is missing AA_LIST.
# Fallback: load directly from benchmark_config which owns these constants.
try:
    from data_loader import (
        load_clinical_targets, load_esm2, match_esm2_to_clinical,
    )
except (ImportError, ModuleNotFoundError):
    # Inline minimal versions so the script always works standalone
    import pandas as pd

    def load_clinical_targets():
        from benchmark_config import CLINICAL_MULTILABEL
        df = pd.read_csv(CLINICAL_MULTILABEL)
        seq_col = next((c for c in df if df[c].dtype == object
                        and df[c].str.len().median() > 50), df.columns[0])
        id_col = next((c for c in ["seq_id", "id", "sample_id"] if c in df.columns), None)
        if id_col is None:
            df["seq_id"] = range(len(df))
            id_col = "seq_id"
        drug_cols = [d for d in DRUGS if d in df.columns]
        result = df[[id_col, seq_col] + drug_cols].copy()
        result.columns = ["seq_id", "sequence"] + drug_cols
        for d in drug_cols:
            result[d] = pd.to_numeric(result[d], errors="coerce").fillna(0).astype(int)
        return result

    def load_esm2():
        from benchmark_config import ESM2_650M_NPZ
        data = np.load(ESM2_650M_NPZ, allow_pickle=True)
        for key in data:
            arr = data[key]
            if arr.ndim == 2 and arr.shape[1] == 1280:
                return arr, None
        raise ValueError("No mean-pooled array found")

    def match_esm2_to_clinical(clinical_df, esm2):
        if isinstance(esm2, dict):
            return np.array([esm2.get(str(s).strip(), np.zeros(1280))
                             for s in clinical_df["sequence"]])
        return esm2[:len(clinical_df)]

try:
    from feature_builders import build_binary_mutation
except (ImportError, ModuleNotFoundError):
    def build_binary_mutation(sequences, wt_seq=HIV2_WT_PR):
        N = len(sequences)
        mat = np.zeros((N, SEQ_LENGTH, 20), dtype=np.float32)
        for i, seq in enumerate(sequences):
            for pos in range(min(len(str(seq).strip()), SEQ_LENGTH)):
                aa = str(seq).strip()[pos].upper()
                if aa in AA_TO_IDX and aa != wt_seq[pos].upper():
                    mat[i, pos, AA_TO_IDX[aa]] = 1.0
        return mat.reshape(N, -1)

# -- SHAP import with graceful degradation -----------------------------
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[shap_analysis] WARNING: shap not installed. Install with: pip install shap")
    print("[shap_analysis] Will fall back to native XGBoost feature importance.")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[shap_analysis] WARNING: xgboost not installed. Install with: pip install xgboost")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# -- Publication style -------------------------------------------------
COLORS_RESISTANT = "#d62728"
COLORS_SUSCEPTIBLE = "#1f77b4"
DRUG_PALETTE = {
    "ATV/r": "#e41a1c", "DRV/r": "#377eb8", "FPV/r": "#4daf4a",
    "IDV/r": "#984ea3", "LPV/r": "#ff7f00", "SQV/r": "#a65628",
    "TPV/r": "#f781bf", "NFV/r": "#999999",
}

# Known HIV-2 resistance mutations (from benchmark_config HIV2EU_RULES + literature)
KNOWN_RESISTANCE_POSITIONS = {
    10: {"L10F", "L10I"}, 20: {"K20R"}, 23: {"L23I"}, 24: {"L24I"},
    30: {"D30N"}, 32: {"V32I"}, 47: {"I47V", "V47A"}, 48: {"G48V"},
    50: {"I50V", "I50L"}, 53: {"I53V"}, 54: {"I54M", "I54V"},
    56: {"T56V"}, 62: {"V62A", "I62V"}, 71: {"A71V"}, 73: {"G73S"},
    76: {"L76V"}, 82: {"V82A", "V82F", "V82I", "V82L", "V82T", "I82F"},
    83: {"N83D"}, 84: {"I84V"}, 88: {"N88D", "N88S"},
    89: {"L89V"}, 90: {"L90M"}, 99: {"L99F"},
}

KNOWNResistanceSet = set()
for muts in KNOWN_RESISTANCE_POSITIONS.values():
    KNOWNResistanceSet.update(muts)


# ======================================================================
# 1. DATA PREPARATION
# ======================================================================

def prepare_data():
    """Load and prepare all data needed for SHAP analysis."""
    print("=" * 70)
    print("SHAP EXPLAINABILITY ANALYSIS -- HIV-2 PI DRUG RESISTANCE")
    print("=" * 70)

    clinical_df = load_clinical_targets()
    sequences = clinical_df["sequence"].tolist()

    # Binary mutation features (F1) -- 1980-D
    mutation_X = build_binary_mutation(sequences)
    print(f"Mutation features: {mutation_X.shape}")

    # ESM-2 mean-pooled (F3) -- 1280-D
    esm2_raw = load_esm2()
    if isinstance(esm2_raw, tuple):
        esm2_mean, _ = esm2_raw
    else:
        esm2_mean = esm2_raw
    esm2_X = match_esm2_to_clinical(clinical_df, esm2_mean)
    print(f"ESM-2 features: {esm2_X.shape}")

    # Targets
    available_drugs = [d for d in DRUGS if d in clinical_df.columns]
    targets = {d: clinical_df[d].values.astype(float) for d in available_drugs}
    binary_targets = {d: (targets[d] > 0).astype(int) for d in available_drugs}

    return {
        "clinical_df": clinical_df,
        "sequences": sequences,
        "mutation_X": mutation_X,
        "esm2_X": esm2_X,
        "targets": targets,
        "binary_targets": binary_targets,
        "drugs": available_drugs,
    }


# ======================================================================
# 2. MUTATION-LEVEL SHAP (PRIMARY -- interpretable)
# ======================================================================

def compute_mutation_shap(data):
    """
    Train XGBoost on binary mutation matrix per drug, compute TreeSHAP.
    Returns per-drug SHAP values and models.
    """
    if not XGB_AVAILABLE:
        print("[shap_analysis] XGBoost not available -- using sklearn GradientBoosting")
        from sklearn.ensemble import GradientBoostingRegressor as XGBModel
    else:
        from xgboost import XGBRegressor as XGBModel

    results = {}
    X = data["mutation_X"]
    n_features = X.shape[1]  # 1980

    for drug in data["drugs"]:
        print(f"\n  Training XGBoost + TreeSHAP for {drug}...")
        y = data["targets"][drug]

        # Train model
        if XGB_AVAILABLE:
            model = xgb.XGBRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.5,
                random_state=RANDOM_SEED, n_jobs=1,
                verbosity=0, objective="reg:squarederror",
            )
        else:
            model = XGBModel(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, random_state=RANDOM_SEED,
            )
        model.fit(X, y)

        # Compute SHAP values
        if SHAP_AVAILABLE:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        else:
            # Fallback: use native feature importance
            shap_values = None

        results[drug] = {
            "model": model,
            "shap_values": shap_values,
            "feature_importance": model.feature_importances_,
        }
        print(f"    Done. Shape of SHAP values: {shap_values.shape if shap_values is not None else 'N/A'}")

    return results


def aggregate_position_importance(shap_results, data):
    """
    Aggregate 1980-D mutation-level SHAP >> 99-position importance.
    Each position has 20 AA channels; we sum absolute SHAP across channels.
    """
    position_data = {drug: np.zeros(SEQ_LENGTH) for drug in data["drugs"]}

    for drug in data["drugs"]:
        sv = shap_results[drug]["shap_values"]
        if sv is None:
            # Fallback: use feature importance
            sv_1d = shap_results[drug]["feature_importance"]
        else:
            sv_1d = np.abs(sv).mean(axis=0)  # mean across samples

        # Reshape to (99, 20) and sum across AA dimension
        for pos in range(SEQ_LENGTH):
            start = pos * 20
            end = (pos + 1) * 20
            if end <= len(sv_1d):
                position_data[drug][pos] = sv_1d[start:end].sum()

    return position_data


def build_mutation_feature_names():
    """Build human-readable feature names for the 1980-D mutation vector."""
    names = []
    for pos in range(SEQ_LENGTH):
        for aa in AA_LIST:
            names.append(f"{HIV2_WT_PR[pos]}{pos+1}{aa}")
    return names


# ======================================================================
# 3. ESM-2 SHAP (global importance)
# ======================================================================

def compute_esm2_shap(data):
    """Compute SHAP on ESM-2 features for one drug (ATV/r) as demonstration."""
    if not SHAP_AVAILABLE or not XGB_AVAILABLE:
        return None

    print("\n  Computing TreeSHAP on ESM-2 features (1280-D)...")
    X = data["esm2_X"]
    y = data["targets"][data["drugs"][0]]  # first drug

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.5,
        random_state=RANDOM_SEED, n_jobs=1,
        verbosity=0, objective="reg:squarederror",
    )
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return {"shap_values": shap_values, "model": model}


# ======================================================================
# 4. FIGURE GENERATION
# ======================================================================

def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def fig_global_importance(position_data, data, save_path):
    """
    fig_shap_global.png -- Global mutation-position importance bar chart.
    Top 20 positions across all drugs, colored by known resistance status.
    """
    set_pub_style()

    # Average importance across all drugs
    all_pos_importance = np.zeros(SEQ_LENGTH)
    for drug in data["drugs"]:
        all_pos_importance += position_data[drug]
    all_pos_importance /= len(data["drugs"])

    # Top 20
    top_idx = np.argsort(all_pos_importance)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 6))

    positions_labels = []
    colors = []
    for idx in top_idx:
        pos = idx + 1
        wt_aa = HIV2_WT_PR[idx]
        is_known = idx in KNOWN_RESISTANCE_POSITIONS
        known_muts = KNOWN_RESISTANCE_POSITIONS.get(idx, set())
        mut_str = ", ".join(sorted(known_muts)[:3]) if known_muts else "novel"
        label = f"Pos {pos} ({wt_aa})"
        if is_known:
            label += f"\n{mut_str}"
        positions_labels.append(label)
        colors.append(COLORS_RESISTANT if is_known else "#2ca02c")

    bars = ax.barh(range(len(top_idx)), all_pos_importance[top_idx][::-1],
                   color=colors[::-1], edgecolor="black", linewidth=0.3)

    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(positions_labels[::-1], fontsize=7)
    ax.set_xlabel("Mean |SHAP value| (averaged across 8 drugs)")
    ax.set_title("Global Feature Importance -- Top 20 Protease Positions\n"
                 "(red = known resistance site, green = novel finding)")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS_RESISTANT, edgecolor="black", label="Known resistance site"),
        Patch(facecolor="#2ca02c", edgecolor="black", label="Novel / unreported site"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def fig_esm2_global_importance(esm2_shap_data, save_path):
    """
    fig_shap_esm2_global.png -- Top 20 ESM-2 embedding dimensions.
    """
    if esm2_shap_data is None or not SHAP_AVAILABLE:
        print("  Skipping ESM-2 global importance (no SHAP data)")
        return

    set_pub_style()
    sv = esm2_shap_data["shap_values"]
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:20]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(20), mean_abs[top_idx][::-1],
            color="steelblue", edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(20))
    ax.set_yticklabels([f"Dim {i}" for i in top_idx[::-1]], fontsize=7)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 20 ESM-2 Embedding Dimensions by Importance\n"
                 "(ATV/r resistance prediction)")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def fig_waterfall(shap_results, data, save_path):
    """
    fig_shap_waterfall.png -- Waterfall plots for 3 resistant + 3 susceptible
    sequences, for the first drug.
    """
    if not SHAP_AVAILABLE:
        print("  Skipping waterfall plots (shap not installed)")
        return

    set_pub_style()
    drug = data["drugs"][0]
    sv = shap_results[drug]["shap_values"]
    y = data["targets"][drug]
    feature_names = build_mutation_feature_names()

    # Find resistant and susceptible examples
    resistant_idx = np.where(y > 0)[0]
    susceptible_idx = np.where(y == 0)[0]

    if len(resistant_idx) < 3 or len(susceptible_idx) < 3:
        print("  Not enough resistant/susceptible samples for waterfall")
        return

    # Pick examples: high SHAP sum for resistant, low for susceptible
    resistant_scores = sv[resistant_idx].sum(axis=1)
    susceptible_scores = sv[susceptible_idx].sum(axis=1)

    # Top 3 most extreme resistant
    top_res = resistant_idx[np.argsort(resistant_scores)[::-1][:3]]
    # Top 3 most extreme susceptible
    top_sus = susceptible_idx[np.argsort(susceptible_scores)[:3]]

    examples = []
    for idx in top_res:
        examples.append((idx, "resistant"))
    for idx in top_sus:
        examples.append((idx, "susceptible"))

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"SHAP Waterfall Plots -- {drug} Resistance Prediction",
                 fontsize=13, fontweight="bold", y=1.02)

    for i, (idx, label) in enumerate(examples):
        row = 0 if label == "resistant" else 1
        col = i % 3
        ax = axes[row, col]

        # Build explanation object
        base_value = 0.0  # TreeSHAP base value
        exp = shap.Explanation(
            values=sv[idx],
            base_values=base_value,
            data=None,
            feature_names=feature_names,
        )

        # Get top 10 features for this sample
        abs_vals = np.abs(sv[idx])
        top_features = np.argsort(abs_vals)[::-1][:10]

        # Manual waterfall-style bar chart
        vals = sv[idx][top_features]
        names = [feature_names[f] for f in top_features]

        colors_wf = [COLORS_RESISTANT if v > 0 else COLORS_SUSCEPTIBLE for v in vals]
        y_pos = range(len(vals))

        ax.barh(y_pos, vals[::-1], color=colors_wf[::-1], edgecolor="black", linewidth=0.3, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names[::-1], fontsize=7)
        ax.axvline(x=0, color="gray", linewidth=0.5)
        ax.set_xlabel("SHAP value (+ pushes prediction higher)")

        seq_id = data["clinical_df"]["seq_id"].iloc[idx]
        true_score = y[idx]
        pred_score = shap_results[drug]["model"].predict(data["mutation_X"][idx:idx+1])[0]

        title_color = COLORS_RESISTANT if label == "resistant" else COLORS_SUSCEPTIBLE
        ax.set_title(f"[{label.upper()}] Seq {seq_id}\n"
                     f"True={true_score:.0f}, Pred={pred_score:.2f}",
                     fontsize=8, color=title_color, fontweight="bold")

    axes[0, 0].set_ylabel("RESISTANT", fontsize=10, fontweight="bold",
                          color=COLORS_RESISTANT)
    axes[1, 0].set_ylabel("SUSCEPTIBLE", fontsize=10, fontweight="bold",
                          color=COLORS_SUSCEPTIBLE)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def fig_dependence(position_data, shap_results, data, save_path):
    """
    fig_shap_dependence.png -- Top 2 features vs predicted resistance,
    colored by drug class.
    """
    if not SHAP_AVAILABLE:
        print("  Skipping dependence plots (shap not installed)")
        return

    set_pub_style()

    # Find the two positions with highest mean importance across drugs
    agg = np.zeros(SEQ_LENGTH)
    for drug in data["drugs"]:
        agg += position_data[drug]
    agg /= len(data["drugs"])
    top2 = np.argsort(agg)[::-1][:2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("SHAP Dependence -- Top 2 Positions vs Predicted Resistance",
                 fontsize=12, fontweight="bold")

    for panel, pos_idx in enumerate(top2):
        ax = axes[panel]
        pos_num = pos_idx + 1

        for drug in data["drugs"]:
            sv = shap_results[drug]["shap_values"]
            X = data["mutation_X"]
            # Feature at this position is the sum of all 20 AA channels
            feature_vals = X[:, pos_idx*20:(pos_idx+1)*20].sum(axis=1)
            shap_at_pos = sv[:, pos_idx*20:(pos_idx+1)*20].sum(axis=1)

            ax.scatter(feature_vals + np.random.uniform(-0.1, 0.1, len(feature_vals)),
                       shap_at_pos,
                       alpha=0.3, s=8, color=DRUG_PALETTE.get(drug, "gray"),
                       label=drug, edgecolors="none")

        wt_aa = HIV2_WT_PR[pos_idx]
        known = pos_idx in KNOWN_RESISTANCE_POSITIONS
        mut_str = ", ".join(sorted(KNOWN_RESISTANCE_POSITIONS.get(pos_idx, set()))) if known else "novel"
        ax.set_xlabel(f"Mutation at position {pos_num} ({wt_aa}) -- binary count")
        ax.set_ylabel("SHAP value (sum across AAs)")
        ax.set_title(f"Position {pos_num} ({wt_aa})\n{mut_str}",
                     fontsize=9)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=6, markerscale=2, loc="best")

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def fig_heatmap(position_data, data, save_path):
    """
    fig_shap_heatmap.png -- Mutation × Drug importance heatmap.
    Rows = protease positions, columns = drugs.
    """
    set_pub_style()

    # Build matrix: (positions, drugs)
    drugs = data["drugs"]
    matrix = np.zeros((SEQ_LENGTH, len(drugs)))
    for j, drug in enumerate(drugs):
        matrix[:, j] = position_data[drug]

    # Normalize per drug for visualization
    matrix_norm = matrix.copy()
    for j in range(matrix.shape[1]):
        mx = matrix[:, j].max()
        if mx > 0:
            matrix_norm[:, j] /= mx

    # Select top positions (max importance > 20th percentile across drugs)
    row_max = matrix.max(axis=1)
    threshold = np.percentile(row_max, 80)
    kept = np.where(row_max >= threshold)[0]

    if len(kept) < 5:
        kept = np.argsort(row_max)[::-1][:15]

    fig, ax = plt.subplots(figsize=(max(6, len(drugs) * 0.8), max(6, len(kept) * 0.35)))

    row_labels = []
    for idx in kept:
        wt = HIV2_WT_PR[idx]
        pos = idx + 1
        is_known = idx in KNOWN_RESISTANCE_POSITIONS
        muts = sorted(KNOWN_RESISTANCE_POSITIONS.get(idx, set()))
        mut_str = ", ".join(muts[:2]) if muts else ""
        label = f"{pos}{wt}"
        if mut_str:
            label += f" ({mut_str})"
        elif is_known:
            label += " (*)"
        row_labels.append(label)

    sns.heatmap(
        matrix_norm[kept, :],
        annot=False, cmap="YlOrRd",
        xticklabels=drugs, yticklabels=row_labels,
        linewidths=0.3, linecolor="white",
        ax=ax, vmin=0, vmax=1,
        cbar_kws={"label": "Normalized importance"},
    )

    ax.set_title("Per-Drug Mutation Importance Heatmap\n"
                 "(positions × drugs, normalized per drug)", fontsize=11)
    ax.set_ylabel("Protease position")
    ax.set_xlabel("Drug")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(fontsize=7)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ======================================================================
# 5. FINDINGS SUMMARY
# ======================================================================

def print_findings(position_data, data):
    """Print a text summary of key findings."""
    print("\n" + "=" * 70)
    print("KEY FINDINGS -- SHAP EXPLAINABILITY")
    print("=" * 70)

    # Aggregate across drugs
    agg = np.zeros(SEQ_LENGTH)
    for drug in data["drugs"]:
        agg += position_data[drug]
    agg /= len(data["drugs"])

    top10 = np.argsort(agg)[::-1][:10]

    print("\nTop 10 most important protease positions (across all drugs):")
    print("-" * 50)
    for rank, idx in enumerate(top10, 1):
        pos = idx + 1
        wt = HIV2_WT_PR[idx]
        importance = agg[idx]
        is_known = idx in KNOWN_RESISTANCE_POSITIONS
        muts = sorted(KNOWN_RESISTANCE_POSITIONS.get(idx, set()))
        mut_str = ", ".join(muts) if muts else "NOVEL -- not in published literature"
        status = "KNOWN" if is_known else "NOVEL"
        print(f"  {rank:2d}. Position {pos:2d} ({wt}): importance={importance:.6f}  "
              f"[{status}] {mut_str}")

    # Per-drug top mutations
    print("\nTop 3 positions per drug:")
    print("-" * 50)
    for drug in data["drugs"]:
        top3 = np.argsort(position_data[drug])[::-1][:3]
        desc = []
        for idx in top3:
            pos = idx + 1
            wt = HIV2_WT_PR[idx]
            is_known = idx in KNOWN_RESISTANCE_POSITIONS
            muts = sorted(KNOWN_RESISTANCE_POSITIONS.get(idx, set()))
            mut_str = muts[0] if muts else "novel"
            desc.append(f"{pos}{wt}({mut_str})")
        print(f"  {drug:6s}: {', '.join(desc)}")

    # Consensus findings
    print("\n\nBIOLOGICAL INTERPRETATION:")
    print("-" * 50)

    # Check well-known positions
    pos82_idx = 81  # 0-indexed
    if pos82_idx in [np.argsort(position_data[d])[::-1][0] for d in data["drugs"]]:
        print("  * Position 82 (V82x) dominates resistance prediction for multiple drugs.")
        print("    Known mutations: V82A/F/I/L/T -- confers resistance to IDV/r, NFV/r, LPV/r, FPV/r.")
    else:
        imp_82 = agg[pos82_idx]
        print(f"  * Position 82 importance: {imp_82:.6f} (rank {np.where(np.argsort(agg)[::-1] == pos82_idx)[0][0]+1})")
        print("    Known mutations: V82A/F/I/L/T.")

    pos50_idx = 49
    imp_50 = agg[pos50_idx]
    print(f"  * Position 50 (I50x): importance {imp_50:.6f}")
    print("    I50V confers broad PI resistance; I50L confers ATV/r hypersusceptibility.")

    pos90_idx = 89
    imp_90 = agg[pos90_idx]
    print(f"  * Position 90 (L90M): importance {imp_90:.6f}")
    print("    L90M confers high-level SQV/r resistance and cross-resistance to other PIs.")

    # Count novel vs known in top 20
    top20 = np.argsort(agg)[::-1][:20]
    n_known = sum(1 for idx in top20 if idx in KNOWN_RESISTANCE_POSITIONS)
    n_novel = 20 - n_known
    print(f"\n  Of top 20 positions: {n_known} are known resistance sites, "
          f"{n_novel} are novel/unreported.")
    if n_novel > 0:
        novel_pos = [idx + 1 for idx in top20 if idx not in KNOWN_RESISTANCE_POSITIONS]
        print(f"  Novel positions: {novel_pos}")
        print("  >> These may represent compensatory mutations or HIV-2-specific resistance sites.")

    print("=" * 70)


def save_shap_results(position_data, data, output_path):
    """Save all importance data to JSON."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "method": "XGBoost + TreeSHAP",
        "feature_type": "Binary mutation matrix (F1, 1980-D >> 99 positions)",
        "n_sequences": len(data["clinical_df"]),
        "drugs": data["drugs"],
        "per_drug_position_importance": {},
        "aggregate_position_importance": {},
        "known_resistance_positions": {str(k): sorted(v) for k, v in KNOWN_RESISTANCE_POSITIONS.items()},
    }

    for drug in data["drugs"]:
        imp = position_data[drug]
        top_positions = np.argsort(imp)[::-1][:30]
        results["per_drug_position_importance"][drug] = {
            "top_30": [
                {
                    "position": int(pos + 1),
                    "wild_type": HIV2_WT_PR[pos],
                    "importance": float(imp[pos]),
                    "known_muts": sorted(KNOWN_RESISTANCE_POSITIONS.get(pos, set())),
                }
                for pos in top_positions
            ]
        }

    # Aggregate
    agg = np.zeros(SEQ_LENGTH)
    for drug in data["drugs"]:
        agg += position_data[drug]
    agg /= len(data["drugs"])
    top_positions = np.argsort(agg)[::-1][:30]
    results["aggregate_position_importance"] = {
        "top_30": [
            {
                "position": int(pos + 1),
                "wild_type": HIV2_WT_PR[pos],
                "importance": float(agg[pos]),
                "known_muts": sorted(KNOWN_RESISTANCE_POSITIONS.get(pos, set())),
                "is_known": pos in KNOWN_RESISTANCE_POSITIONS,
            }
            for pos in top_positions
        ]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


# ======================================================================
# 6. MAIN
# ======================================================================

def main():
    print(f"SHAP available: {SHAP_AVAILABLE}")
    print(f"XGBoost available: {XGB_AVAILABLE}")
    print(f"Output directory: {FIGURES_DIR}")

    # Prepare data
    data = prepare_data()

    # -- Mutation-level SHAP ----------------------------------------
    print("\n" + "=" * 70)
    print("COMPUTING MUTATION-LEVEL SHAP (1980-D binary mutation matrix)")
    print("=" * 70)
    shap_results = compute_mutation_shap(data)

    # Aggregate to position-level
    position_data = aggregate_position_importance(shap_results, data)

    # -- ESM-2 SHAP ------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPUTING ESM-2 SHAP (1280-D embeddings)")
    print("=" * 70)
    esm2_shap_data = compute_esm2_shap(data)

    # -- Generate figures -------------------------------------------
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig_global_importance(
        position_data, data,
        FIGURES_DIR / "fig_shap_global.png"
    )

    if esm2_shap_data is not None:
        fig_esm2_global_importance(
            esm2_shap_data,
            FIGURES_DIR / "fig_shap_esm2_global.png"
        )

    fig_waterfall(
        shap_results, data,
        FIGURES_DIR / "fig_shap_waterfall.png"
    )

    fig_dependence(
        position_data, shap_results, data,
        FIGURES_DIR / "fig_shap_dependence.png"
    )

    fig_heatmap(
        position_data, data,
        FIGURES_DIR / "fig_shap_heatmap.png"
    )

    # -- Save results JSON ------------------------------------------
    save_shap_results(
        position_data, data,
        RESULTS_DIR / "shap_results.json"
    )

    # -- Print findings ---------------------------------------------
    print_findings(position_data, data)

    print(f"\nAll SHAP analysis complete. Figures in: {FIGURES_DIR}")
    return position_data, shap_results, data


if __name__ == "__main__":
    main()
