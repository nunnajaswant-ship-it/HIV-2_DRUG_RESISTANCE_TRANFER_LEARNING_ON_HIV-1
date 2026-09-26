"""
Compute the pooled zero-shot ROC-AUC for Table 7 (honest disclosure).

The locked manifest (clinical_validation_final_results.json) stores per-drug
zero-shot AUCs but NOT the pooled zero-shot AUC (the pipeline computes it
internally but never persists it). This script re-runs the exact locked
pipeline with save_json=False and reports the pooled zero-shot AUC so the
manuscript cell can be filled with a real, reproducible value.

No files are modified. The locked manifest is untouched.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clinical_validation_final import run

out = run(save_json=False)

pooled = out["pooled"]
print("\n" + "=" * 60)
print("POOLED ZERO-SHOT AUC (recomputed, same 2121-D scaled space)")
print("=" * 60)
print(f"  n (rows)            : {pooled['n']}")
print(f"  n binary-classified : {pooled['n_binary_class']}")
print(f"  n Resistant         : {pooled['n_resistant']}")
print(f"  zero-shot Pearson R : {pooled['zeroshot_recomputed']['pearson_r']}")
print(f"  zero-shot Spearman  : {pooled['zeroshot_recomputed']['spearman_rho']}")
print(f"  zero-shot RMSE      : {pooled['zeroshot_recomputed']['rmse']}")
print(f"  zero-shot ROC-AUC   : {pooled['zeroshot_recomputed'].get('roc_auc')}")
print("=" * 60)