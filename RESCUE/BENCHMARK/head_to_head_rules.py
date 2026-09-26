"""
head_to_head_rules.py
=====================
Three-way comparison on identical HIV-2 clinical sequences:

  (i)   the cross-species transfer model (trained only on HIV-1 measured dG)
  (ii)  the Stanford-derived HIV-2 expert rule engine (mutation penalty lookup)
  (iii) the Stanford HIVDB clinical penalty score (the evaluation target)

Inputs (all committed / produced by committed code):
  RESCUE/BENCHMARK/results/per_row_clinical_predictions.csv
      -> from capture_per_row_predictions.py (integrity-gated to the locked manifest)
  stanford db/HIV2_EXPERT_INTERPRETATION_RESULTS.xlsx
      -> from chembl/hiv2_expert_rules_engine.py (Stanford HIV-2 single+combo penalties)

Output: RESCUE/BENCHMARK/results/head_to_head_rules.json  (+ console report)

Nothing is typed by hand; every number is computed from the two files.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score

BENCH = Path(__file__).resolve().parent
RES = BENCH.parent
PRED = BENCH / "results" / "per_row_clinical_predictions.csv"
EXPERT = RES.parent / "stanford db" / "HIV2_EXPERT_INTERPRETATION_RESULTS.xlsx"
OUT = BENCH / "results" / "head_to_head_rules.json"

DRUGS = ["ATV/r", "DRV/r", "LPV/r", "SQV/r"]
# expert thresholds (from Resistance_Summary sheet of the same workbook)
SUS_CUT, RES_CUT = 15, 30


def main():
    pred = pd.read_csv(PRED)
    pred["SequenceID"] = pred["SequenceID"].astype(str).str.strip()

    exp = pd.read_excel(EXPERT, sheet_name="HIV2_Scored_Sequences")
    exp["SequenceID"] = exp["SequenceID"].astype(str).str.strip()
    long = exp.melt(id_vars=["SequenceID"], value_vars=DRUGS,
                    var_name="Drug", value_name="rule_penalty")
    long["rule_penalty"] = pd.to_numeric(long["rule_penalty"], errors="coerce")

    df = pred.merge(long, on=["SequenceID", "Drug"], how="inner")
    print(f"joined rows: {len(df)}  sequences: {df.SequenceID.nunique()}")

    out = {"n_joined_rows": int(len(df)),
           "n_joined_sequences": int(df.SequenceID.nunique()),
           "drugs": {}}
    pooled = {k: [] for k in ("stan", "rule", "ft", "zs")}

    print("\nPer-drug (n, corr vs Stanford penalty):")
    print(f"{'drug':7} {'n':>5} {'model R':>9} {'rule R':>9} "
          f"{'model-vs-rule':>14} {'rule==stan %':>13}")
    for d in DRUGS:
        g = df[df.Drug == d].dropna(subset=["penalty", "pred_ft",
                                            "pred_zs", "rule_penalty"])
        if len(g) < 10:
            continue
        r_model = pearsonr(g.penalty, g.pred_ft)[0]
        r_rule = pearsonr(g.penalty, g.rule_penalty)[0]
        r_mr = pearsonr(g.rule_penalty, g.pred_ft)[0]
        # exact agreement of rule with Stanford penalty value
        agree = float((g.rule_penalty == g.penalty).mean() * 100)
        out["drugs"][d] = {
            "n": int(len(g)),
            "model_pearson_vs_stanford": round(float(r_model), 4),
            "rule_pearson_vs_stanford": round(float(r_rule), 4),
            "model_vs_rule_pearson": round(float(r_mr), 4),
            "rule_exact_agreement_pct": round(agree, 2),
            "rule_nonzero_n": int((g.rule_penalty != 0).sum()),
        }
        pooled["stan"].append(g.penalty.values)
        pooled["rule"].append(g.rule_penalty.values)
        pooled["ft"].append(g.pred_ft.values)
        pooled["zs"].append(g.pred_zs.values)
        print(f"{d:7} {len(g):>5} {r_model:>9.4f} {r_rule:>9.4f} "
              f"{r_mr:>14.4f} {agree:>13.2f}")

    P = {k: np.concatenate(v) for k, v in pooled.items()}
    r_model_p = pearsonr(P["stan"], P["ft"])[0]
    r_rule_p = pearsonr(P["stan"], P["rule"])[0]
    r_mr_p = pearsonr(P["rule"], P["ft"])[0]
    r_zs_p = pearsonr(P["stan"], P["zs"])[0]
    out["pooled"] = {
        "n": int(len(P["stan"])),
        "model_pearson_vs_stanford": round(float(r_model_p), 4),
        "zeroshot_pearson_vs_stanford": round(float(r_zs_p), 4),
        "rule_pearson_vs_stanford": round(float(r_rule_p), 4),
        "model_vs_rule_pearson": round(float(r_mr_p), 4),
    }
    print(f"\nPooled (n={len(P['stan'])}):")
    print(f"  model     vs Stanford : R={r_model_p:.4f}")
    print(f"  zero-shot vs Stanford : R={r_zs_p:.4f}")
    print(f"  rule      vs Stanford : R={r_rule_p:.4f}")
    print(f"  model     vs rule     : R={r_mr_p:.4f}")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
