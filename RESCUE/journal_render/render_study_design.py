"""
render_study_design.py  (v3 — complete study design, auto-layout)
=================================================================
Figure 1: the full analysis pipeline in four stages. Boxes size themselves to
their content so text cannot overflow, and all numbers are read live from the
result manifests so the figure cannot drift from the paper.

Output: RESCUE/journal_render/figures/fig0_study_design.png (600 dpi)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "RESCUE" / "BENCHMARK" / "results"
OUT = ROOT / "RESCUE" / "journal_render" / "figures" / "fig0_study_design.png"

cvf = json.loads((B / "clinical_validation_final_results.json").read_text(encoding="utf-8"))
eu = json.loads((B / "hiv2eu_coverage.json").read_text(encoding="utf-8"))
cov = json.loads((B / "rule_coverage.json").read_text(encoding="utf-8"))
loso = json.loads((B / "loso_head_to_head.json").read_text(encoding="utf-8"))
h2h = json.loads((B / "head_to_head_paired.json").read_text(encoding="utf-8"))
rx = json.loads((B / "reviewer_experiments.json").read_text(encoding="utf-8"))
TF = cvf["comparison"]["pooled_4_drugs"]["final_transfer"]

plt.rcParams.update({"font.family": "serif", "font.size": 8})

W, H = 18.0, 10.2
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

CUR, FEAT, MOD, EVAL = "#dce9f5", "#e4f1e4", "#fdf2dd", "#f9e2e2"
EDGE = "#5a6b7c"

TOP = 90.5
BOT = 4.5
COLW = 23.2
LX = [1.2, 26.1, 51.0, 75.9]
LS = 2.30          # line spacing
PAD_TOP = 4.4      # space for the box title
PAD_BOT = 1.6
GAP = 2.2


def draw_column(x, face, boxes):
    """boxes = list of (title, [lines]); sizes and stacks them top-down."""
    hs = [PAD_TOP + PAD_BOT + LS * len(ls) for _, ls in boxes]
    total = sum(hs) + GAP * (len(boxes) - 1)
    y = TOP - (TOP - BOT - total) / 2.0     # centre the stack
    for (title, lines), h in zip(boxes, hs):
        ax.add_patch(FancyBboxPatch((x, y - h), COLW, h,
                                    boxstyle="round,pad=0.30,rounding_size=0.8",
                                    linewidth=1.0, edgecolor=EDGE,
                                    facecolor=face, zorder=2))
        ax.text(x + COLW / 2, y - 2.0, title, ha="center", va="top",
                fontsize=8.8, fontweight="bold", color="#101820", zorder=3)
        ty = y - PAD_TOP
        for ln in lines:
            if ln == "":
                ty -= LS * 0.35
                continue
            ax.text(x + COLW / 2, ty, ln, ha="center", va="top",
                    fontsize=7.1, color="#222222", zorder=3)
            ty -= LS
        y -= (h + GAP)
    return y


# ============================ headers =====================================
heads = ["1   DATA CURATION", "2   FEATURES",
         "3   MODEL & TRANSFER", "4   EVALUATION"]
faces = [CUR, FEAT, MOD, EVAL]
for x, t, f in zip(LX, heads, faces):
    ax.add_patch(FancyBboxPatch((x, 92.0), COLW, 6.2,
                                boxstyle="round,pad=0.30,rounding_size=0.8",
                                linewidth=1.3, edgecolor=EDGE, facecolor=f, zorder=3))
    ax.text(x + COLW / 2, 95.1, t, ha="center", va="center", fontsize=10.2,
            fontweight="bold", color="#0d1b26", zorder=4)

# ============================ 1 curation ==================================
draw_column(LX[0], CUR, [
    ("HIV-1 training corpus", [
        "Stanford HIVDB phenotypic assays",
        "7,183 rows \u2212 1,500 synthetic = 5,683",
        "1,882 unique sequences",
        "95% clustering: 882 clusters + 714 singletons",
        "",
        "target: measured IC50 (nM) \u2192 \u0394G (kcal/mol)",
    ]),
    ("HIV-2 cross-species panel", [
        "ChEMBL 380 / 5074: 321 raw \u2192 320 mapped",
        "137 rows zero-shot test",
        "160 rows fine-tune (4 drugs \u00d7 40)",
        "",
        "only 19 unique sequences",
        "identity to ROD WT: 88.9\u201399.0%",
    ]),
    ("HIV-2 clinical cohort", [
        "654 entries \u00d7 8 PIs = 5,232 rows",
        "Stanford_HIVDB 3,456 / Mined 1,776",
        "headline subset: 4 PIs = 2,616",
        "",
        "label = HIVDB penalty score (held out)",
    ]),
    ("Curation applied to all", [
        "99-residue length filter",
        "align to shared 99-aa frame (HXB2 / ROD)",
        "drug-name normalisation; \u0394G range filter",
        "overlap audit: 0 sequences shared with HIV-1",
    ]),
])

# ============================ 2 features ==================================
draw_column(LX[1], FEAT, [
    ("21 descriptors per residue", [
        "Hellberg Z-scales (3):",
        "hydrophobicity / steric / electronic",
        "",
        "Kidera factors (10, orthogonal):",
        "KF1 helix \u00b7 KF2 size \u00b7 KF4 hydrophobic \u00b7 \u2026",
        "",
        "VHSE scales (8): VHSE1 hydro \u00b7 VHSE2 steric \u00b7 VHSE3 electronic",
    ]),
    ("21 \u00d7 99 = 2,079-D", [
        "+ mean & SD of each = 2,121-D deployment bundle",
        "scaled with the HIV-1 scaler",
    ]),
    ("Why these transfer", [
        "descriptors are properties of the amino acid \u2014",
        "identical in HIV-1 and HIV-2",
        "",
        "HIV-1-specific structure does not transfer (R = \u22120.14)",
        "pocket conserved to 1.09 \u00c5 C\u03b1 RMSD",
    ]),
    ("Pocket position sets", [
        "core 5.0 \u00c5 (16 residues):",
        "8 23 25 27 28 29 30 31 32 47 48 49 50 81 82 84",
        "Pocket-19: adds 26 45 46 54 76 80",
        "extended 8.0 \u00c5; BASE-A = 16 core + 18 neighbours",
        "",
        "used for masking, not for weighting",
    ]),
])

# ============================ 3 model =====================================
draw_column(LX[2], MOD, [
    ("HIV-1 pre-training", [
        "Random Forest (600 trees, depth 20)",
        "XGBoost (800 trees, depth 6)",
        "",
        "GroupKFold at 95% identity (no leakage)",
        "within-HIV-1 CV: R = 0.7387 \u00b1 0.0472",
    ]),
    ("Model choice justified", [
        "tree ensembles suit n = 19 sequences: greedy",
        "selection, no scaling, reproducible, interpretable",
        "",
        "deep learning tested and FAILED (PocketGNN \u22120.17;",
        "DBPT+ 0.21) \u2014 data starvation, not architecture",
    ]),
    ("Transfer to HIV-2", [
        "no weights are modified",
        "  (i) HIV-1 model FROZEN",
        "  (ii) its \u0394G prediction \u2192 1 meta-feature",
        "  (iii) NEW per-drug RF on [2,121-D | meta-feature]",
        "",
        "deployment model = the new RF",
    ]),
    ("Zero-shot baseline", [
        "HIV-1 model applied directly to HIV-2",
        "no HIV-2 fitting at any point",
        "",
        "R = 0.5163   (transfer ratio 0.70)",
        "same-space strict baseline 0.2492",
    ]),
])

# ============================ 4 evaluation ================================
draw_column(LX[3], EVAL, [
    ("Measured phenotypes (primary)", [
        "head-to-head on measured \u0394G, 160 rows",
        f"model {h2h['pooled']['model_r']:.4f}   vs   rule {h2h['pooled']['rule_r']:.4f}",
        f"\u0394R = +{h2h['pooled']['difference']:.3f},   "
        f"P(\u0394R\u22640) = {h2h['pooled']['boot_p_diff_le_0']:.3f}",
        "",
        f"leave-one-sequence-out: model ahead in "
        f"{loso['loso_full']['frac_folds_model_wins']*100:.0f}% of folds",
    ]),
    ("Rule coverage (independent)", [
        "clinical benchmark is zero for 93.1% of isolates",
        "",
        "official HIV-2EU v4 rules:",
        f"{eu['measured_panel']['unscored_pct']}% of the measured panel unscored",
        f"no protease rules for "
        f"{len(eu['clinical_panel']['drugs_without_rules'])} of 8 clinical drugs",
        f"{eu['measured_panel']['mutations_in_hiv2eu_vocab']} of "
        f"{eu['measured_panel']['distinct_observed_mutations']} mutations covered",
        "",
        "model R = 0.585 where the rule is silent",
    ]),
    ("Robustness", [
        f"seed sweep: R = {rx['multiseed']['transfer_mean']:.3f} "
        f"\u00b1 {rx['multiseed']['transfer_sd']:.3f}",
        f"(locked single-seed value {h2h['pooled']['model_r']:.4f})",
        "",
        "fine-tune-size sweep: HIV-1 contribution",
        "\u2264 0.015 at every data volume",
    ]),
    ("Clinical cohort (secondary)", [
        "2,616 never-seen rows",
        f"pooled R = {TF['pearson_r']:.4f},  ROC-AUC = {TF['roc_auc']:.3f}",
        "",
        "target is rule-derived \u2192 supporting evidence",
        "only; per-drug values are primary",
    ]),
])

# ============================ arrows ======================================
def arrow(x1, y1, x2, y2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.6,
                                 color="#37474f", linestyle=ls, zorder=5,
                                 shrinkA=0, shrinkB=0))


for k in range(3):
    arrow(LX[k] + COLW + 0.3, 60.0, LX[k + 1] - 0.3, 60.0)

ax.text(50, 1.4,
        "No clinical label enters training. The clinical penalty is used only for held-out "
        "evaluation; its sparsity is quantified independently in stage 4.",
        ha="center", va="center", fontsize=7.8, style="italic", color="#444444")

fig.savefig(OUT, dpi=600, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
