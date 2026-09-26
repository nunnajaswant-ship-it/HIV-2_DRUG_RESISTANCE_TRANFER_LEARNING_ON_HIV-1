"""
Evolutionary Conservation and Mutation Analysis of HIV-2 Protease Sequences
============================================================================
Comprehensive analysis pipeline for HIV-2 PR evolutionary conservation,
mutation frequency, drug binding site analysis, cross-resistance correlations,
and intrinsic resistance documentation.

Outputs:
  - fig_conservation.png        - conservation profile across positions 1-99
  - fig_mutation_frequency.png  - bar chart of mutation frequency per position
  - fig_cross_resistance.png    - heatmap of cross-drug correlation
  - fig_hiv2_vs_hiv1.png        - natural polymorphism comparison
  - conservation_results.json   - full statistical summary
"""
import sys
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "opencode_shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import HIV2_WT, HIV1_WT, SEQ_LENGTH
from benchmark_config import (
    DRUGS, DRUG_NAMES, RESULTS_DIR, FIGURES_DIR,
    CLINICAL_MULTILABEL, HIV2_WT_PR, HIV2EU_RULES,
    AA_LIST, AA_TO_IDX,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Wild-type reference ───────────────────────────────────────────────
# HIV2_WT_PR (benchmark_config) is an HIV-1-derived reference that does NOT
# match the clinical HIV-2 sequences (0/654 share its N-terminus). The true
# HIV-2 EU wild-type is HIV2_WT (shared config); we validate at load time.
WT_PR = HIV2_WT_PR  # default; resolved to HIV2_WT if mismatched

# ── Structural regions (1-indexed) ─────────────────────────────────────
ACTIVE_SITE       = [25, 27]
FLAP_REGION       = list(range(45, 56))
DIMER_INTERFACE   = list(range(95, 99)) + [1]
DRUG_BINDING      = [8, 23, 25, 27, 28, 29, 30, 31, 32, 47, 48, 49, 50, 81, 82, 84]
CRITICAL_HIV2     = [32, 47, 76, 82]
POCKET_POSITIONS  = [8, 23, 25, 27, 28, 29, 30, 31, 32, 47, 48, 49, 50, 81, 82, 84]

# Natural polymorphisms in HIV-2 that resemble HIV-1 resistance mutations
NATURAL_POLYMORPHISMS_HIV2 = {
    "L10V": "HIV-2 naturally has V10 in many strains; HIV-1 L10V is a minor PI mutation",
    "V32I": "HIV-2 wild-type has I32 at this position; HIV-1 V32I confers DRV/IDV resistance",
    "M36I": "HIV-2 naturally carries I36; HIV-1 M36I is a minor PI mutation",
    "I47V": "HIV-2 EU uses I47V as resistance; HIV-2 WT has V47 in some clades",
    "A71V": "HIV-2 naturally has V71 in many strains; HIV-1 A71V is accessory",
    "G73A": "HIV-2 has A73 commonly; HIV-1 G73S/T are minor PI mutations",
    "M46I": "HIV-1 M46I is major PI mutation; HIV-2 has I46 in many lineages",
}

# Known HIV-2 protease resistance positions (from literature)
KNOWN_RESISTANCE_POSITIONS = {
    10, 20, 23, 24, 30, 32, 47, 48, 50, 53, 54, 56, 62, 71, 73, 76, 82, 83, 84, 88, 89, 90, 99
}


# ══════════════════════════════════════════════════════════════════════
# 1. SEQUENCE LOADING & ALIGNMENT
# ══════════════════════════════════════════════════════════════════════

def load_sequences():
    """Load all HIV-2 protease sequences from clinical CSV."""
    df = pd.read_csv(CLINICAL_MULTILABEL, low_memory=False)

    seq_col = None
    for col in ["Mutant_Sequence", "sequence", "seq", "amino_acid", "aa_seq",
                "PR_sequence", "prot_seq"]:
        if col in df.columns:
            seq_col = col
            break
    if seq_col is None:
        for col in df.columns:
            if df[col].dtype == object and df[col].str.len().median() > 50:
                seq_col = col
                break

    id_col = None
    for col in ["SequenceID", "seq_id", "id", "sample_id", "patient_id", "sequence_id"]:
        if col in df.columns:
            id_col = col
            break
    if id_col is None:
        df["seq_id"] = range(len(df))
        id_col = "seq_id"

    drug_cols = [d for d in DRUGS if d in df.columns]

    sequences = []
    seq_ids = []
    for _, row in df.iterrows():
        s = str(row[seq_col]).strip().upper()
        s = "".join(c for c in s if c in AA_LIST)
        sequences.append(s)
        seq_ids.append(row[id_col])

    print(f"Loaded {len(sequences)} sequences from {CLINICAL_MULTILABEL.name}")
    print(f"  Drug columns found: {drug_cols}")
    print(f"  Sequence lengths: min={min(len(s) for s in sequences)}, "
          f"max={max(len(s) for s in sequences)}, target={SEQ_LENGTH}")

    # ── Validate wild-type reference against clinical data ──────────
    global WT_PR
    n_match_ref = sum(s[:9] == WT_PR[:9] for s in sequences)
    n_match_hiv2 = sum(s[:9] == HIV2_WT[:9] for s in sequences)
    if n_match_ref < 0.5 * len(sequences) and n_match_hiv2 > n_match_ref:
        print(f"  NOTE: benchmark_config HIV2_WT_PR matches only {n_match_ref}/{len(sequences)} "
              f"sequences (it is HIV-1-derived).")
        print(f"        Falling back to validated HIV-2 wild-type (matches {n_match_hiv2}/{len(sequences)}).")
        WT_PR = HIV2_WT
    else:
        print(f"  Reference HIV2_WT_PR validated: prefix matches {n_match_ref}/{len(sequences)} sequences.")

    return sequences, seq_ids, drug_cols, df


def align_column_wise(sequences, wt_seq=None, target_len=SEQ_LENGTH):
    """
    Column-wise alignment against wild-type reference.
    Pads/truncates each sequence to target_len.
    Returns: list of aligned strings of length target_len.
    """
    if wt_seq is None:
        wt_seq = WT_PR
    aligned = []
    for seq in sequences:
        s = seq[:target_len].ljust(target_len, "-")
        aligned.append(s)
    return aligned


def build_alignment_matrix(aligned_seqs):
    """Convert aligned sequences to a (N, L) numpy char array."""
    return np.array([list(s) for s in aligned_seqs])


# ══════════════════════════════════════════════════════════════════════
# 2. CONSERVATION SCORING
# ══════════════════════════════════════════════════════════════════════

def shannon_entropy(column):
    """Calculate Shannon entropy for a single alignment column."""
    valid = [c for c in column if c in AA_LIST]
    if not valid:
        return 0.0
    counts = Counter(valid)
    total = len(valid)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy


def compute_conservation_scores(aligned_matrix):
    """
    Compute per-position conservation scores.
    Returns dict with keys: entropy, conservation, max_entropy, per_pos.
    """
    n_seq, n_pos = aligned_matrix.shape
    max_entropy = np.log2(20)  # maximum entropy for 20 amino acids

    entropy = np.zeros(n_pos)
    for j in range(n_pos):
        entropy[j] = shannon_entropy(aligned_matrix[:, j])

    conservation = 1.0 - entropy / max_entropy
    conservation = np.clip(conservation, 0, 1)

    return {
        "entropy": entropy,
        "conservation": conservation,
        "max_entropy": max_entropy,
        "n_sequences": n_seq,
        "n_positions": n_pos,
    }


# ══════════════════════════════════════════════════════════════════════
# 3. MUTATION FREQUENCY ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def mutation_frequency_analysis(aligned_matrix, wt_seq=None):
    """
    Count mutations at each position and classify them.
    Returns: list of dicts sorted by frequency.
    """
    if wt_seq is None:
        wt_seq = WT_PR
    n_seq = aligned_matrix.shape[0]
    position_data = []

    for j in range(aligned_matrix.shape[1]):
        col = aligned_matrix[:, j]
        valid = [c for c in col if c in AA_LIST]
        wt_aa = wt_seq[j] if j < len(wt_seq) else "X"

        mutations = Counter()
        for aa in valid:
            if aa != wt_aa:
                mutations[f"{wt_aa}{j+1}{aa}"] += 1

        total_valid = len(valid)
        n_mutated = sum(mutations.values())
        pct_mutated = (n_mutated / total_valid * 100) if total_valid > 0 else 0

        # Classify each mutation
        classified = {}
        for mut, count in mutations.items():
            if mut in HIV2EU_RULES:
                classified[mut] = "known_resistance"
            elif j + 1 in KNOWN_RESISTANCE_POSITIONS:
                classified[mut] = "at_resistance_position"
            else:
                classified[mut] = "natural_polymorphism"

        most_common = mutations.most_common(1)[0] if mutations else ("N/A", 0)

        position_data.append({
            "position": j + 1,
            "wt": wt_aa,
            "n_sequences_valid": total_valid,
            "n_mutated": n_mutated,
            "pct_mutated": round(pct_mutated, 2),
            "n_unique_mutations": len(mutations),
            "most_common_mutation": most_common[0],
            "most_common_count": most_common[1],
            "mutations": dict(mutations),
            "classification": classified,
        })

    position_data.sort(key=lambda x: x["pct_mutated"], reverse=True)
    return position_data


def classify_all_mutations(position_data):
    """
    Classify mutations into: known_resistance, natural_polymorphism, novel.
    """
    known_resistance = []
    natural_poly = []
    novel = []

    for pos_info in position_data:
        for mut, cls in pos_info["classification"].items():
            entry = {
                "mutation": mut,
                "position": pos_info["position"],
                "frequency": pos_info["mutations"].get(mut, 0),
                "pct": round(pos_info["mutations"].get(mut, 0) / max(pos_info["n_sequences_valid"], 1) * 100, 2),
            }
            if cls == "known_resistance":
                known_resistance.append(entry)
            elif cls == "at_resistance_position":
                natural_poly.append(entry)
            else:
                novel.append(entry)

    return {
        "known_resistance": sorted(known_resistance, key=lambda x: -x["frequency"]),
        "natural_polymorphism": sorted(natural_poly, key=lambda x: -x["frequency"]),
        "novel": sorted(novel, key=lambda x: -x["frequency"]),
    }


# ══════════════════════════════════════════════════════════════════════
# 4. CONSERVATION AT DRUG BINDING SITES
# ══════════════════════════════════════════════════════════════════════

def binding_site_conservation(conservation_scores):
    """Compare conservation across structural regions."""
    cons = conservation_scores["conservation"]

    def mean_cons(positions_1idx):
        idx = [p - 1 for p in positions_1idx if 0 <= p - 1 < len(cons)]
        return float(np.mean(cons[idx])) if idx else 0.0

    regions = {
        "Active site (D25, D27)": ACTIVE_SITE,
        "Flap region (45-55)": FLAP_REGION,
        "Dimer interface": DIMER_INTERFACE,
        "Drug binding pocket": DRUG_BINDING,
        "Critical HIV-2 positions (32,47,76,82)": CRITICAL_HIV2,
        "Known resistance positions": sorted(KNOWN_RESISTANCE_POSITIONS),
        "Full protease (1-99)": list(range(1, 100)),
    }

    result = {}
    for name, positions in regions.items():
        result[name] = {
            "mean_conservation": round(mean_cons(positions), 4),
            "positions": positions,
            "n_positions": len(positions),
        }
    return result


# ══════════════════════════════════════════════════════════════════════
# 5. CROSS-RESISTANCE CORRELATION
# ══════════════════════════════════════════════════════════════════════

def cross_resistance_correlation(df, drug_cols):
    """
    Compute pairwise Pearson correlation between drug penalty scores.
    """
    available = [d for d in drug_cols if d in df.columns]
    if len(available) < 2:
        print("  Warning: fewer than 2 drug columns, skipping correlation")
        return None, None

    drug_data = df[available].astype(float)
    corr_matrix = drug_data.corr(method="pearson")

    # Also compute pairwise mutation co-occurrence correlation
    print(f"  Correlation matrix computed for {len(available)} drugs")

    return corr_matrix, available


# ══════════════════════════════════════════════════════════════════════
# 6. HIV-2 INTRINSIC RESISTANCE vs HIV-1
# ══════════════════════════════════════════════════════════════════════

def hiv2_vs_hiv1_polymorphisms(aligned_matrix, wt_seq_hiv2=None):
    """
    Document natural polymorphisms where HIV-2 differs from HIV-1
    at positions known as resistance mutations in HIV-1.
    """
    if wt_seq_hiv2 is None:
        wt_seq_hiv2 = WT_PR
    wt_hiv1 = HIV1_WT

    comparison = []
    for j in range(min(len(wt_seq_hiv2), len(wt_hiv1), aligned_matrix.shape[1])):
        hiv2_wt = wt_seq_hiv2[j]
        hiv1_wt = wt_hiv1[j]

        col = aligned_matrix[:, j]
        valid = [c for c in col if c in AA_LIST]
        if not valid:
            continue

        counts = Counter(valid)
        total = len(valid)

        # Check if the dominant HIV-2 residue differs from HIV-1 WT
        dominant_aa = counts.most_common(1)[0][0]
        dominant_pct = counts.most_common(1)[0][1] / total * 100

        is_polymorphic = dominant_aa != hiv1_wt and dominant_pct > 50

        # Check if position is a known resistance position in HIV-1
        hiv1_resistance_pos = j + 1 in KNOWN_RESISTANCE_POSITIONS

        comparison.append({
            "position": j + 1,
            "hiv2_wt": hiv2_wt,
            "hiv1_wt": hiv1_wt,
            "hiv2_dominant": dominant_aa,
            "hiv2_dominant_pct": round(dominant_pct, 1),
            "hiv1_differs": dominant_aa != hiv1_wt,
            "is_hiv1_resistance_position": hiv1_resistance_pos,
            "is_natural_polymorphism": is_polymorphic and hiv1_resistance_pos,
            "all_residues": {aa: round(count / total * 100, 1) for aa, count in counts.most_common()},
        })

    return comparison


def document_hiv2_intrinsic_resistance(comparison):
    """
    Generate the documented list of HIV-2 natural polymorphisms.
    These are positions where HIV-2 naturally carries what HIV-1 considers
    a resistance mutation.
    """
    documented = []
    for item in comparison:
        if item["is_natural_polymorphism"]:
            pos = item["position"]
            mut_key = f"{item['hiv1_wt']}{pos}{item['hiv2_dominant']}"
            note = NATURAL_POLYMORPHISMS_HIV2.get(mut_key, "")

            # Check if this is in known HIV-2EU resistance table
            hiv2eu_known = False
            for k in HIV2EU_RULES:
                if k.endswith(str(pos)) or k.startswith(str(item["hiv2_dominant"])):
                    pass  # more precise check below
            for k in HIV2EU_RULES:
                k_pos = int("".join(c for c in k if c.isdigit()))
                if k_pos == pos:
                    hiv2eu_known = True
                    break

            documented.append({
                "mutation": mut_key,
                "position": pos,
                "hiv2_dominant_freq": item["hiv2_dominant_pct"],
                "description": note if note else f"HIV-2 has {item['hiv2_dominant']} at position {pos} naturally",
                "in_hiv2eu_table": hiv2eu_known,
            })

    return documented


# ══════════════════════════════════════════════════════════════════════
# 7. SUBTYPE / CLUSTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def simple_subtype_analysis(aligned_matrix, sequences):
    """
    Identify sequence clusters by simple hamming distance grouping.
    Uses single-linkage clustering with a threshold.
    """
    n_seq = aligned_matrix.shape[0]
    wt_arr = np.array(list(WT_PR[:aligned_matrix.shape[1]]))

    # Build mutation profile per sequence
    profiles = []
    for i in range(n_seq):
        seq_arr = aligned_matrix[i]
        muts = set()
        for j in range(len(seq_arr)):
            if seq_arr[j] in AA_LIST and j < len(wt_arr) and seq_arr[j] != wt_arr[j]:
                muts.add(f"{wt_arr[j]}{j+1}{seq_arr[j]}")
        profiles.append(muts)

    # Simple frequency-based clustering: group by dominant mutation pattern
    cluster_map = {}
    cluster_id = 0
    assigned = [False] * n_seq

    for i in range(n_seq):
        if assigned[i]:
            continue
        # Start new cluster
        cluster_map[i] = cluster_id
        assigned[i] = True
        for j in range(i + 1, n_seq):
            if assigned[j]:
                continue
            # Jaccard similarity of mutation sets
            if len(profiles[i] | profiles[j]) == 0:
                sim = 1.0
            else:
                sim = len(profiles[i] & profiles[j]) / len(profiles[i] | profiles[j])
            if sim > 0.8:
                cluster_map[j] = cluster_id
                assigned[j] = True
        cluster_id += 1

    # Summarize clusters
    clusters = defaultdict(list)
    for idx, cid in cluster_map.items():
        clusters[cid].append(idx)

    # Sort by size
    sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))

    result = {
        "n_clusters": len(sorted_clusters),
        "clusters": [],
    }
    for cid, members in sorted_clusters[:10]:  # top 10
        result["clusters"].append({
            "cluster_id": cid,
            "size": len(members),
            "pct_of_total": round(len(members) / n_seq * 100, 1),
            "sample_indices": members[:5],
        })

    return result


# ══════════════════════════════════════════════════════════════════════
# 8. FIGURE GENERATION
# ══════════════════════════════════════════════════════════════════════

def set_style():
    """Publication-quality plot style."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.figsize": (12, 5),
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    sns.set_palette("Set2")


def fig_conservation_profile(conservation_scores, save_path=None):
    """
    Fig: Line plot of conservation across positions 1-99.
    Highlight resistance positions, active site, and flaps.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    positions = np.arange(1, conservation_scores["n_positions"] + 1)
    conservation = conservation_scores["conservation"]
    entropy = conservation_scores["entropy"]

    # Plot conservation as line
    ax.plot(positions, conservation, color="#2171b5", linewidth=1.5,
            label="Conservation (1 - H/Hmax)", zorder=3)
    ax.fill_between(positions, conservation, alpha=0.15, color="#2171b5")

    # Highlight known resistance positions
    for pos in KNOWN_RESISTANCE_POSITIONS:
        if pos <= len(conservation):
            ax.axvline(x=pos, color="#cb181d", alpha=0.3, linewidth=0.8, linestyle="--")

    # Highlight active site
    for pos in ACTIVE_SITE:
        if pos <= len(conservation):
            ax.axvline(x=pos, color="#238b45", alpha=0.6, linewidth=2, linestyle="-",
                       label="Active site" if pos == ACTIVE_SITE[0] else "")

    # Highlight critical HIV-2 positions
    for pos in CRITICAL_HIV2:
        if pos <= len(conservation):
            ax.plot(pos, conservation[pos - 1], "o", color="#e6550d",
                    markersize=8, zorder=5,
                    label="Critical HIV-2" if pos == CRITICAL_HIV2[0] else "")

    # Mark highly variable positions
    top_variable = np.argsort(conservation)[:10]
    for idx in top_variable:
        ax.annotate(f"{idx+1}", (idx+1, conservation[idx]),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=7, ha="center", color="#756bb1", fontweight="bold")

    ax.set_xlabel("Protease Position")
    ax.set_ylabel("Conservation Score")
    ax.set_title("HIV-2 Protease Evolutionary Conservation Profile")
    ax.set_xlim(1, 99)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5, label="Purifying selection (0.9)")

    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower left", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def fig_mutation_frequency(position_data, save_path=None):
    """Bar chart of mutation frequency per position."""
    set_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    # Take top 30 most variable
    top = position_data[:30]
    positions = [d["position"] for d in top]
    frequencies = [d["pct_mutated"] for d in top]

    # Color by classification
    colors = []
    for d in top:
        n_known = sum(1 for c in d["classification"].values() if c == "known_resistance")
        if n_known > 0:
            colors.append("#cb181d")  # red for known resistance
        elif d["position"] in KNOWN_RESISTANCE_POSITIONS:
            colors.append("#e6550d")  # orange for resistance positions
        else:
            colors.append("#2171b5")  # blue for natural variation

    bars = ax.bar(range(len(positions)), frequencies, color=colors, edgecolor="white",
                  linewidth=0.5, zorder=3)

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"{d['position']}\n({d['wt']})" for d in top],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("% Sequences Mutated")
    ax.set_title("HIV-2 Protease - Mutation Frequency at Most Variable Positions")

    # Annotate counts
    for i, (bar, freq) in enumerate(zip(bars, frequencies)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{freq:.1f}%", ha="center", va="bottom", fontsize=6)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#cb181d", label="Known resistance mutation"),
        Patch(facecolor="#e6550d", label="At resistance position"),
        Patch(facecolor="#2171b5", label="Natural variation"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def fig_cross_resistance_heatmap(corr_matrix, save_path=None):
    """Heatmap of cross-drug correlation."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    # Shorten drug names for display
    short_names = {d: d.replace("/r", "").replace("r", "") for d in corr_matrix.index}
    display_names = [short_names.get(d, d) for d in corr_matrix.index]

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5,
        xticklabels=display_names,
        yticklabels=display_names,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Cross-Resistance Correlation Matrix\n(HIV-2 PI Penalty Scores)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def fig_hiv2_vs_hiv1(comparison, save_path=None):
    """Comparison of natural polymorphisms between HIV-2 and HIV-1."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: positions where HIV-2 differs from HIV-1
    diff_positions = [c for c in comparison if c["hiv1_differs"]]
    positions = [c["position"] for c in diff_positions[:30]]
    hiv2_pcts = [c["hiv2_dominant_pct"] for c in diff_positions[:30]]

    colors = ["#cb181d" if c["is_natural_polymorphism"] else "#6baed6"
              for c in diff_positions[:30]]

    ax1.barh(range(len(positions)), hiv2_pcts, color=colors, edgecolor="white")
    ax1.set_yticks(range(len(positions)))
    ax1.set_yticklabels(
        [f"Pos {p} ({c['hiv1_wt']}->{c['hiv2_dominant']})"
         for p, c in zip(positions, diff_positions[:30])],
        fontsize=7)
    ax1.set_xlabel("% HIV-2 Sequences with Dominant Residue")
    ax1.set_title("A. Positions Where HIV-2 Differs from HIV-1 WT")
    ax1.axvline(x=50, color="gray", linestyle="--", alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#cb181d", label="HIV-1 resistance position"),
        Patch(facecolor="#6baed6", label="Non-resistance position"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=7)

    # Panel B: Conservation at key differing positions
    polymorphisms = [c for c in comparison if c["is_natural_polymorphism"]]
    labels = [f"{c['hiv1_wt']}{c['position']}{c['hiv2_dominant']}" for c in polymorphisms]
    freqs = [c["hiv2_dominant_pct"] for c in polymorphisms]

    if labels:
        y_pos = range(len(labels))
        ax2.barh(y_pos, freqs, color="#e6550d", edgecolor="white")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_xlabel("% Frequency in HIV-2 Population")
        ax2.set_title("B. Documented HIV-2 Natural Polymorphisms\n(Resemble HIV-1 Resistance)")
        ax2.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="50% threshold")

        for i, freq in enumerate(freqs):
            ax2.text(freq + 1, i, f"{freq:.1f}%", va="center", fontsize=7)
    else:
        ax2.text(0.5, 0.5, "No natural polymorphisms found\nat documented positions",
                 transform=ax2.transAxes, ha="center", va="center", fontsize=10)
        ax2.set_title("B. Documented HIV-2 Natural Polymorphisms")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 9. STATISTICAL SUMMARY
# ══════════════════════════════════════════════════════════════════════

def statistical_summary(conservation_scores, position_data, binding_site_info,
                        classified_mutations, subtype_info, intrinsic_docs):
    """Compile comprehensive statistical summary."""
    cons = conservation_scores["conservation"]
    entropy = conservation_scores["entropy"]
    wt_seq = WT_PR

    # Most conserved (top 10)
    most_conserved_idx = np.argsort(-cons)[:10]
    most_conserved = [
        {"position": int(i + 1), "conservation": round(float(cons[i]), 4),
         "wt": wt_seq[i] if i < len(wt_seq) else "?"}
        for i in most_conserved_idx
    ]

    # Most variable (top 10)
    most_variable_idx = np.argsort(cons)[:10]
    most_variable = [
        {"position": int(i + 1), "conservation": round(float(cons[i]), 4),
         "wt": wt_seq[i] if i < len(wt_seq) else "?"}
        for i in most_variable_idx
    ]

    # Purifying selection
    purifying = int(np.sum(cons > 0.9))
    pct_purifying = round(purifying / len(cons) * 100, 1)

    summary = {
        "n_sequences_analyzed": conservation_scores["n_sequences"],
        "protease_length": conservation_scores["n_positions"],
        "mean_conservation": round(float(np.mean(cons)), 4),
        "median_conservation": round(float(np.median(cons)), 4),
        "std_conservation": round(float(np.std(cons)), 4),
        "mean_entropy": round(float(np.mean(entropy)), 4),
        "most_conserved_top10": most_conserved,
        "most_variable_top10": most_variable,
        "positions_under_purifying_selection": {
            "count": purifying,
            "percentage": pct_purifying,
            "threshold": ">0.9 conservation",
        },
        "binding_site_conservation": binding_site_info,
        "mutation_classification": {
            "known_resistance_count": len(classified_mutations["known_resistance"]),
            "natural_polymorphism_count": len(classified_mutations["natural_polymorphism"]),
            "novel_count": len(classified_mutations["novel"]),
            "known_resistance_top10": classified_mutations["known_resistance"][:10],
        },
        "hiv2_intrinsic_resistance": {
            "documented_count": len(intrinsic_docs),
            "polymorphisms": intrinsic_docs,
        },
        "subtype_analysis": subtype_info,
    }
    return summary


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  HIV-2 PROTEASE - EVOLUTIONARY CONSERVATION & MUTATION ANALYSIS")
    print("=" * 70)

    # ── Load sequences ──────────────────────────────────────────────
    print("\n[1/9] Loading sequences...")
    sequences, seq_ids, drug_cols, df = load_sequences()

    # ── Align ───────────────────────────────────────────────────────
    print("\n[2/9] Aligning sequences (column-wise against HIV-2 WT)...")
    aligned = align_column_wise(sequences)
    matrix = build_alignment_matrix(aligned)
    print(f"  Alignment matrix: {matrix.shape}")

    # ── Conservation scoring ────────────────────────────────────────
    print("\n[3/9] Computing conservation scores (Shannon entropy)...")
    conservation = compute_conservation_scores(matrix)
    print(f"  Mean conservation: {np.mean(conservation['conservation']):.4f}")
    print(f"  Mean entropy: {np.mean(conservation['entropy']):.4f} bits")

    # ── Mutation frequency ──────────────────────────────────────────
    print("\n[4/9] Analyzing mutation frequencies...")
    mut_freq = mutation_frequency_analysis(matrix)
    print(f"  Top 5 most variable positions:")
    for d in mut_freq[:5]:
        print(f"    Pos {d['position']} ({d['wt']}): {d['pct_mutated']:.1f}% mutated, "
              f"top={d['most_common_mutation']} ({d['most_common_count']})")

    # ── Classify mutations ──────────────────────────────────────────
    print("\n[5/9] Classifying mutations...")
    classified = classify_all_mutations(mut_freq)
    print(f"  Known resistance mutations: {len(classified['known_resistance'])}")
    print(f"  At resistance positions: {len(classified['natural_polymorphism'])}")
    print(f"  Novel/other: {len(classified['novel'])}")

    # ── Binding site conservation ───────────────────────────────────
    print("\n[6/9] Analyzing conservation at drug binding sites...")
    binding = binding_site_conservation(conservation)
    for region, info in binding.items():
        print(f"  {region}: conservation = {info['mean_conservation']:.4f}")

    # ── Cross-resistance correlation ────────────────────────────────
    print("\n[7/9] Computing cross-resistance correlations...")
    corr_matrix, available_drugs = cross_resistance_correlation(df, drug_cols)
    if corr_matrix is not None:
        print(f"  Correlation matrix: {corr_matrix.shape}")

    # ── HIV-2 vs HIV-1 ─────────────────────────────────────────────
    print("\n[8/9] Documenting HIV-2 intrinsic resistance...")
    comparison = hiv2_vs_hiv1_polymorphisms(matrix)
    intrinsic_docs = document_hiv2_intrinsic_resistance(comparison)
    print(f"  Natural polymorphisms documented: {len(intrinsic_docs)}")
    for doc in intrinsic_docs[:7]:
        print(f"    {doc['mutation']}: {doc['description']}")

    # ── Subtype analysis ────────────────────────────────────────────
    print("\n[9/9] Performing subtype/cluster analysis...")
    subtypes = simple_subtype_analysis(matrix, sequences)
    print(f"  Identified {subtypes['n_clusters']} clusters")
    for cl in subtypes["clusters"][:5]:
        print(f"    Cluster {cl['cluster_id']}: {cl['size']} sequences ({cl['pct_of_total']}%)")

    # ── Generate figures ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    fig_dir = FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig_conservation_profile(conservation, save_path=fig_dir / "fig_conservation.png")
    fig_mutation_frequency(mut_freq, save_path=fig_dir / "fig_mutation_frequency.png")
    if corr_matrix is not None:
        fig_cross_resistance_heatmap(corr_matrix, save_path=fig_dir / "fig_cross_resistance.png")
    fig_hiv2_vs_hiv1(comparison, save_path=fig_dir / "fig_hiv2_vs_hiv1.png")

    # ── Statistical summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPILING STATISTICAL SUMMARY")
    print("=" * 70)

    summary = statistical_summary(
        conservation, mut_freq, binding, classified, subtypes, intrinsic_docs
    )

    # Add correlation data to summary
    if corr_matrix is not None:
        summary["cross_resistance_correlation"] = {
            "drugs": available_drugs,
            "matrix": corr_matrix.round(4).to_dict(),
        }

    # Save JSON
    results_path = RESULTS_DIR / "conservation_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    # ── Print key findings ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"  Sequences analyzed: {summary['n_sequences_analyzed']}")
    print(f"  Mean conservation: {summary['mean_conservation']:.4f}")
    print(f"  Positions under purifying selection (>0.9): "
          f"{summary['positions_under_purifying_selection']['count']}/99 "
          f"({summary['positions_under_purifying_selection']['percentage']}%)")
    print(f"\n  Most conserved positions: "
          f"{', '.join(str(p['position']) for p in summary['most_conserved_top10'])}")
    print(f"  Most variable positions:  "
          f"{', '.join(str(p['position']) for p in summary['most_variable_top10'])}")

    print(f"\n  Binding site conservation:")
    for region, info in summary["binding_site_conservation"].items():
        print(f"    {region}: {info['mean_conservation']:.4f}")

    print(f"\n  HIV-2 natural polymorphisms (resistance-like):")
    for doc in summary["hiv2_intrinsic_resistance"]["polymorphisms"]:
        print(f"    {doc['mutation']}: {doc['description']}")

    print(f"\n  Mutation classification:")
    mc = summary["mutation_classification"]
    print(f"    Known resistance: {mc['known_resistance_count']}")
    print(f"    At resistance positions: {mc['natural_polymorphism_count']}")
    print(f"    Novel: {mc['novel_count']}")

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    summary = main()
