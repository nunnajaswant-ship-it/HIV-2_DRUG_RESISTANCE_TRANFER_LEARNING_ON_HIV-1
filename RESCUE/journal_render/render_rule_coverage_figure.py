"""
render_rule_coverage_figure.py
==============================
Figure 8: how much of HIV-2 protease sequence space the rule-based
interpretation covers, and where the transferred model retains signal.

Reads live values from RESCUE/BENCHMARK/results/rule_coverage.json.
Outputs RESCUE/journal_render/figures/fig8_rule_coverage.png (600 dpi).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "RESCUE" / "BENCHMARK" / "results" / "rule_coverage.json"
OUT = ROOT / "RESCUE" / "journal_render" / "figures" / "fig8_rule_coverage.png"

R = json.load(open(RES, encoding="utf-8"))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))

# ---------------- (A) clinical benchmark label coverage -------------------
ax = axes[0]
q1 = R["Q1_clinical_label_coverage"]
drugs = list(q1["per_drug"].keys())
zero = [q1["per_drug"][d]["zero_pct"] for d in drugs]
order = np.argsort(zero)
drugs = [drugs[i] for i in order]
zero = [zero[i] for i in order]
y = np.arange(len(drugs))
ax.barh(y, zero, color="#8c8c8c", height=0.62, zorder=3)
ax.axvline(q1["overall_zero_pct"], color="#b2182b", lw=1.4, ls="--", zorder=4)
ax.text(q1["overall_zero_pct"] - 1.5, -0.85,
        f"overall {q1['overall_zero_pct']:.1f}%",
        color="#b2182b", fontsize=7.5, ha="right", va="center")
ax.set_yticks(y)
ax.set_yticklabels(drugs, fontsize=8)
ax.set_xlabel("clinical label equal to zero (%)")
ax.set_xlim(0, 100)
ax.set_title("(A) The benchmark label is\nzero for 93% of HIV-2 isolates",
             fontsize=9.5, loc="left")
ax.grid(axis="x", alpha=0.25, lw=0.5, zorder=0)

# ---------------- (B) mutation coverage, both rule sources ----------------
ax = axes[1]
q4 = R["Q4_mutation_coverage"]
n_tot = q4["distinct_observed_mutations"]
n_stan = q4["mutations_with_explicit_rule"]
EU = ROOT / "RESCUE" / "BENCHMARK" / "results" / "hiv2eu_coverage.json"
eu = json.load(open(EU, encoding="utf-8")) if EU.exists() else None
n_eu = eu["measured_panel"]["mutations_in_hiv2eu_vocab"] if eu else None

xs = [0, 1]
vals = [n_stan] + ([n_eu] if n_eu is not None else [])
labs = ["Stanford-\nderived", "official\nHIV-2EU v4"][:len(vals)]
cols = ["#4393c3", "#b2182b"][:len(vals)]
ax.bar(xs[:len(vals)], vals, color=cols, width=0.55, zorder=3)
ax.axhline(n_tot, color="#888888", lw=0.8, ls=":", zorder=2)
ax.text(1.52, n_tot, f"{n_tot} observed", fontsize=7.5, color="#666666",
        va="bottom", ha="right")
for x, v in zip(xs, vals):
    ax.text(x, v + 0.9, str(v), ha="center", fontsize=9.5, fontweight="bold",
            color="#333333")
ax.set_xticks(xs[:len(vals)])
ax.set_xticklabels(labs, fontsize=8)
ax.set_xlim(-0.6, 1.6)
ax.set_ylabel("mutations with an explicit rule")
ax.set_ylim(0, n_tot * 1.12)
ax.set_title(f"(B) Both rule sets recognise only\na handful of {n_tot} mutations",
             fontsize=9.5, loc="left")
ax.grid(axis="y", alpha=0.25, lw=0.5, zorder=0)

# ---------------- (C) stratified performance ------------------------------
ax = axes[2]
q3 = R["Q3_rule_blind_vs_covered"]
labels = ["model,\nrule-blind rows", "model,\nrule-covered rows",
          "rule,\nrule-covered rows"]
vals = [q3["blind_model_r"], q3["covered_model_r"], q3["covered_rule_r"]]
err = None
if "blind_model_r_ci" in q3:
    lo, hi = q3["blind_model_r_ci"]
    err = [[max(0, q3["blind_model_r"] - lo)],
           [hi - q3["blind_model_r"]]]
colors = ["#b2182b", "#4393c3", "#d9d9d9"]
x = np.arange(3)
ax.bar(x[:2], vals[:2], color=colors[:2], width=0.55, zorder=3)
ax.bar(x[2:], vals[2:], color=colors[2], width=0.55, zorder=3)
if err:
    ax.errorbar([0], [vals[0]], yerr=err, fmt="none", ecolor="#4d4d4d",
                capsize=3, lw=0.9, zorder=4)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Pearson R vs measured $\\Delta$G")
ax.set_ylim(0, 0.95)
ax.set_title("(C) On the rows the rule cannot\nrank, the model still predicts",
             fontsize=9.5, loc="left")
ax.grid(axis="y", alpha=0.25, lw=0.5, zorder=0)
ax.text(0, vals[0] + 0.045, f"{vals[0]:.3f}", ha="center", fontsize=8.5,
        color="#b2182b", fontweight="bold")
ax.text(1, vals[1] + 0.045, f"{vals[1]:.3f}", ha="center", fontsize=8.5)
ax.text(2, vals[2] + 0.045, f"{vals[2]:.3f}", ha="center", fontsize=8.5)
ax.text(2, vals[2] / 2, "rule has zero\nvariance here",
        ha="center", va="center", fontsize=6.6, color="#666666")

fig.tight_layout(w_pad=2.2)
fig.savefig(OUT, dpi=600, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
print(f"  (A) overall zero = {q1['overall_zero_pct']}%")
print(f"  (B) Stanford {n_stan}/{n_tot}, HIV-2EU {n_eu}/{n_tot}")
print(f"  (C) blind R = {q3['blind_model_r']} CI {q3.get('blind_model_r_ci')}")
