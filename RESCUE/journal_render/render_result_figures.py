"""
render_result_figures.py
========================
Result figures for the HIV-1 -> HIV-2 cross-species transfer manuscript.

Every number is read from the LOCKED result manifest
    RESCUE/BENCHMARK/results/clinical_validation_final_results.json
(produced by RESCUE/BENCHMARK/clinical_validation_final.py). No value is
typed by hand; if a key is missing the script raises rather than guess.

Output: RESCUE/journal_render/figures/fig6_results.png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
MANIFEST = os.path.join(HERE, "..", "BENCHMARK", "results",
                        "clinical_validation_final_results.json")

DRUGS = ["ATV/r", "DRV/r", "LPV/r", "SQV/r"]
C_FINAL = "#1f4e79"
C_ZS = "#c00000"
C_OOF = "#2e7d32"


def load():
    with open(MANIFEST, encoding="utf-8") as fh:
        d = json.load(fh)
    return d


def panel_clinical(ax, d):
    fin = [d["per_drug"][k]["pearson_r"] for k in DRUGS]
    zs = [d["per_drug"][k]["zeroshot_recomputed"]["pearson_r"] for k in DRUGS]
    x = np.arange(len(DRUGS))
    w = 0.38
    b1 = ax.bar(x - w / 2, fin, w, color=C_FINAL, label="final transfer model")
    b2 = ax.bar(x + w / 2, zs, w, color=C_ZS, label="same-space zero-shot")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    ax.set_xticks(x, DRUGS)
    ax.set_ylabel("Pearson $r$ vs HIVDB penalty")
    ax.set_ylim(0, 1.0)
    ax.set_title("(a) Clinical validation, per drug\n"
                 "$\\it{zero}$-$\\it{shot}$ is higher within every drug",
                 fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.25)


def panel_oof(ax, d):
    san = d["sanity_check_chembl_oof"]
    r = [san[k]["pearson_r"] for k in DRUGS]
    lo = [san[k]["ci_pearson_lo"] for k in DRUGS]
    hi = [san[k]["ci_pearson_hi"] for k in DRUGS]
    yerr = [np.array(r) - np.array(lo), np.array(hi) - np.array(r)]
    x = np.arange(len(DRUGS))
    ax.bar(x, r, 0.55, color=C_OOF, yerr=yerr, capsize=3,
           error_kw=dict(lw=0.8))
    for xi, ri in zip(x, r):
        ax.text(xi, ri + 0.03, f"{ri:.3f}", ha="center", fontsize=7)
    ax.axhline(san["MEAN"]["pearson_r"], color="0.2", ls="--", lw=1)
    ax.set_xticks(x, DRUGS)
    ax.set_ylabel("Pearson $R$ (out-of-fold)")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"(b) HIV-2 ChEMBL benchmark, transfer fine-tune\n"
                 f"(n={san['MEAN']['n']}; dashed line = mean "
                 f"$R$={san['MEAN']['pearson_r']:.4f})", fontsize=9, loc="left")
    ax.grid(axis="y", alpha=0.25)


def panel_drm(ax, d):
    cl = d["consistency_layer"]["per_drug"]
    r = [cl[k]["drm_drugspecific_corr"]["pearson_r"] for k in DRUGS]
    x = np.arange(len(DRUGS))
    ax.bar(x, r, 0.55, color="#e36c09")
    for xi, ri in zip(x, r):
        ax.text(xi, ri + 0.02, f"{ri:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, DRUGS)
    ax.set_ylabel("Pearson $r$")
    ax.set_ylim(0, 1.0)
    ax.set_title("(c) Known-DRM concordance\n(predicted $\\Delta G$ vs drug-specific "
                 "DRM count)", fontsize=9, loc="left")
    ax.grid(axis="y", alpha=0.25)


def main():
    d = load()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), dpi=600)
    panel_clinical(axes[0], d)
    panel_oof(axes[1], d)
    panel_drm(axes[2], d)
    fig.suptitle("Cross-species transfer model performance", fontsize=11, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG, "fig6_results.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
