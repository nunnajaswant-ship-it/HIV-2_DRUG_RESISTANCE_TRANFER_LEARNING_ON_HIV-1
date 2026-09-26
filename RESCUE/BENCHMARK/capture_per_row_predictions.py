"""
capture_per_row_predictions.py
==============================
Runs the locked clinical-validation pipeline and captures PER-ROW model
predictions (SequenceID, drug, penalty, class, pred_ft, pred_zs) so that the
model can be compared head-to-head against rule-based HIV-2 interpretations
on identical sequences.

The committed script persists summary statistics only; this driver intercepts
the module-internal `_per_drug_clinical` (no model code modified) exactly as
the supplementary S4 driver does, and additionally records the sequence IDs
needed for the join.

Integrity gate: re-derived per-drug Pearson R must match the committed
clinical_validation_final_results.json within 1e-3, otherwise nothing is written.

Run:
    python RESCUE/BENCHMARK/capture_per_row_predictions.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BENCH = Path(__file__).resolve().parent
_RES = _BENCH.parent
sys.path.insert(0, str(_BENCH))

import clinical_validation_final as cvf  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

OUT = _BENCH / "results" / "per_row_clinical_predictions.csv"
MANIFEST = _BENCH / "results" / "clinical_validation_final_results.json"

CAP = {}

_orig_p4 = cvf.phase4_clinical


def _p4(ctx):
    CAP["clin"] = ctx["clin"]
    return _orig_p4(ctx)


cvf.phase4_clinical = _p4

_orig = cvf._per_drug_clinical


def _patched(bundle, hiv1_models, final_models, Xc, row_idx, ix, drug,
             scores, classes):
    res, pred_ft, pred_zs = _orig(bundle, hiv1_models, final_models, Xc,
                                  row_idx, ix, drug, scores, classes)
    CAP.setdefault("rows", {})[drug] = (
        np.asarray(ix), np.asarray(scores, dtype=float),
        np.asarray(pred_ft, dtype=float), np.asarray(pred_zs, dtype=float),
        np.asarray(classes, dtype=object))
    return res, pred_ft, pred_zs


cvf._per_drug_clinical = _patched

print("Running committed clinical_validation_final.py (4-drug run) ...")
cvf.run(drugs=list(cvf.COMMON_DRUGS), save_json=False)

clin = CAP["clin"]
seq_ids = clin["SequenceID"].values
seqs = clin["_seq"].values

recs = []
for drug, (ix, scores, pred_ft, pred_zs, classes) in CAP["rows"].items():
    for k in range(len(ix)):
        row = int(ix[k])
        recs.append({
            "SequenceID": str(seq_ids[row]),
            "Drug": drug,
            "penalty": float(scores[k]),
            "cls": str(classes[k]),
            "pred_ft": float(pred_ft[k]),
            "pred_zs": float(pred_zs[k]),
            "sequence": str(seqs[row]),
        })
df = pd.DataFrame(recs)

committed = json.load(open(MANIFEST, encoding="utf-8"))["per_drug"]
ok_all = True
print("\nIntegrity gate (re-derived R vs committed manifest):")
for drug, (ix, scores, pred_ft, pred_zs, classes) in CAP["rows"].items():
    m = np.isfinite(scores) & np.isfinite(pred_ft)
    r, _ = pearsonr(scores[m], pred_ft[m])
    ref = committed[drug]["pearson_r"]
    flag = "OK " if abs(r - ref) <= 1e-3 else "FAIL"
    if abs(r - ref) > 1e-3:
        ok_all = False
    print(f"  {flag} {drug}: re-derived R={r:.4f}  committed={ref:.4f}  "
          f"|diff|={abs(r-ref):.1e}  n={int(m.sum())}")

if not ok_all:
    raise SystemExit("Integrity gate failed - refusing to write per-row file.")

df.to_csv(OUT, index=False)
print(f"\nwrote {OUT}  ({len(df)} rows, {df['SequenceID'].nunique()} sequences)")
