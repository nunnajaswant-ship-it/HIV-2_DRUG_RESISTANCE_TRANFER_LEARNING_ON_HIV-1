"""
render_head_to_head_figure.py
=============================
Figure 7: cross-species transfer model vs Stanford-derived rule-based HIV-2
interpretation, on MEASURED binding affinities (HIV-2 ChEMBL panel).

All numbers are read from RESCUE/BENCHMARK/results/head_to_head_paired.json
(produced by head_to_head_paired.py). Nothing is typed by hand.

Output: RESCUE/journal_render/figures/fig7_head_to_head.png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "..", "BENCHMARK", "results", "head_to_head_paired.json")

C_MODEL = "#1f4e79"
C_RULE = "#c00000"


def main():
    d = json.load(open(RES, encoding="utf-8"))
    drugs = ["ATV/r", "DRV/r", "LPV/r", "SQV/r"]
    model = [d["per_drug"][k]["model_r"] for k in drugs]
    rule = [d["per_drug"][k]["rule_r"] for k in drugs]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=600,
                             gridspec_kw={"width_ratios": [2.1, 1.0]})

    # --- panel (a): per-drug ---
    ax = axes[0]
    x = np.arange(len(drugs))
    w = 0.38
    b1 = ax.bar(x - w / 2, model, w, color=C_MODEL,
                label="cross-species transfer model")
    b2 = ax.bar(x + w / 2, rule, w, color=C_RULE,
                label="Stanford-derived rule")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    ax.set_xticks(x, drugs)
    ax.set_ylabel("Pearson $R$ vs measured $\\Delta G$")
    ax.set_ylim(0, 1.0)
    ax.set_title("(a) Per-drug, measured HIV-2 binding affinity (n=40 each)",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.25)

    # --- panel (b): pooled with clustered bootstrap CI ---
    ax = axes[1]
    p = d["pooled"]
    vals = [p["model_r"], p["rule_r"]]
    ci = [p["boot_model_ci"], p["boot_rule_ci"]]
    err = [[vals[i] - ci[i][0] for i in range(2)],
           [ci[i][1] - vals[i] for i in range(2)]]
    bb = ax.bar([0, 1], vals, 0.5, color=[C_MODEL, C_RULE], yerr=err,
                capsize=4, error_kw=dict(lw=0.9))
    ax.bar_label(bb, fmt="%.3f", padding=9, fontsize=7.5)
    ax.set_xticks([0, 1], ["model", "rule"])
    ax.set_ylim(0, 1.0)
    ax.set_title("(b) Pooled (n=160)", fontsize=9.5, loc="left")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.5, 0.03,
            f"$\\Delta R$ = +{p['difference']:.3f}\n"
            f"95% CI [{p['boot_diff_ci'][0]:.2f}, {p['boot_diff_ci'][1]:.2f}]\n"
            f"$P(\\Delta R\\leq0)$ = {p['boot_p_diff_le_0']:.3f}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="0.7", boxstyle="round,pad=0.35"))

    fig.suptitle("Cross-species transfer model outperforms rule-based HIV-2 "
                 "interpretation on measured phenotypes", fontsize=10.5, y=1.03)
    fig.tight_layout()
    out = os.path.join(FIG, "fig7_head_to_head.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
