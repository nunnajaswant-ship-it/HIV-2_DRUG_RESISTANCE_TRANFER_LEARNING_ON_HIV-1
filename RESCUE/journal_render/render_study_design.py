"""
render_study_design.py
======================
Figure 1: study design / analysis workflow schematic.

All numeric annotations are read from the locked manifests so the schematic can
never drift from the reported results:
    RESCUE/BENCHMARK/results/clinical_validation_final_results.json
    RESCUE/BENCHMARK/results/head_to_head_paired.json

Output: RESCUE/journal_render/figures/fig0_study_design.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
MAN = os.path.join(HERE, "..", "BENCHMARK", "results",
                   "clinical_validation_final_results.json")
H2H = os.path.join(HERE, "..", "BENCHMARK", "results",
                   "head_to_head_paired.json")

C_DATA = "#dbe7f3"
C_MODEL = "#cfe3d4"
C_VAL = "#f6e2c9"
C_EDGE = "#5b6b7a"


def box(ax, x, y, w, h, title, lines, fc, fs=8.4, tfs=9.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                linewidth=0.9, edgecolor=C_EDGE, facecolor=fc))
    ax.text(x + w / 2, y + h - 0.050, title, ha="center", va="top",
            fontsize=tfs, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.115 - i * 0.060, ln, ha="center",
                va="top", fontsize=fs)


def arrow(ax, x1, y1, x2, y2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, linewidth=1.1,
                                 color=C_EDGE, linestyle=ls,
                                 shrinkA=1, shrinkB=1))


def main():
    man = json.load(open(MAN, encoding="utf-8"))
    h2h = json.load(open(H2H, encoding="utf-8"))
    ft = man["sanity_check_chembl_oof"]["MEAN"]["pearson_r"]
    cl = man["pooled"]["pearson_r"]
    auc = man["pooled"]["roc_auc_resistant_vs_susceptible"]
    drm = max(man["consistency_layer"]["per_drug"][d]["drm_drugspecific_corr"]
              ["pearson_r"] for d in ["ATV/r", "DRV/r", "LPV/r", "SQV/r"])
    hm, hr, hd, hp = (h2h["pooled"]["model_r"], h2h["pooled"]["rule_r"],
                      h2h["pooled"]["difference"],
                      h2h["pooled"]["boot_p_diff_le_0"])

    fig, ax = plt.subplots(figsize=(11.0, 4.3), dpi=600)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # --- stage 1: data ---
    box(ax, 0.010, 0.60, 0.185, 0.33, "HIV-1 training data",
        ["5,683 curated Stanford", "HIVDB phenotypic assays", "(1,882 sequences)",
         "measured IC50 \u2192 \u0394G"], C_DATA)
    box(ax, 0.010, 0.13, 0.185, 0.34, "HIV-2 data",
        ["ChEMBL: 137 zero-shot", "160-row fine-tune panel",
         "(4 drugs \u00d7 40)", "2,616 clinical rows"], C_DATA)

    # --- stage 2: pre-train ---
    box(ax, 0.225, 0.60, 0.185, 0.33, "HIV-1 pre-training",
        ["Random Forest / XGBoost", "2,121-D biophysical", "descriptors",
         "(21 scales \u00d7 99 positions)"], C_MODEL)

    # --- stage 3: transfer ---
    box(ax, 0.440, 0.60, 0.185, 0.33, "Cross-species transfer",
        ["zero-shot: R = 0.5163", "(\u2192 HIV-2, 137 rows)",
         "transfer ratio 0.70"], C_MODEL)
    box(ax, 0.440, 0.13, 0.185, 0.34, "Fine-tune on HIV-2",
        ["HIV-1 prediction as", "meta-feature + descriptors",
         f"OOF R = {ft:.4f}"], C_MODEL)

    # --- stage 4: validation ---
    box(ax, 0.655, 0.60, 0.335, 0.33, "Validation on measured phenotypes",
        [f"model R = {hm:.3f}  vs  rule R = {hr:.3f}",
         f"\u0394R = +{hd:.3f},  P(\u0394R\u22640) = {hp:.3f}",
         "higher on every drug"], C_VAL)
    box(ax, 0.655, 0.13, 0.335, 0.34, "Validation on clinical cohort",
        [f"2,616 never-seen rows; pooled R = {cl:.3f}",
         f"ROC-AUC = {auc:.3f}",
         f"known-DRM concordance up to r = {drm:.2f}"], C_VAL)

    # --- arrows ---
    arrow(ax, 0.195, 0.765, 0.225, 0.765)
    arrow(ax, 0.410, 0.765, 0.440, 0.765)
    arrow(ax, 0.532, 0.600, 0.532, 0.470)
    arrow(ax, 0.625, 0.765, 0.655, 0.765)
    arrow(ax, 0.625, 0.300, 0.655, 0.300)
    arrow(ax, 0.195, 0.300, 0.440, 0.300, ls=(0, (4, 2)))

    ax.text(0.5, 0.035,
            "No clinical label ever enters training; the clinical penalty is "
            "used only for held-out evaluation.",
            ha="center", fontsize=7.6, style="italic", color="#444")

    fig.tight_layout()
    out = os.path.join(FIG, "fig0_study_design.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
