"""
head_to_head_paired.py
======================
Definitive, NON-CIRCULAR head-to-head on MEASURED HIV-2 binding affinities.

Captures the model's per-row out-of-fold predictions on the HIV-2 ChEMBL panel
by intercepting `kfold_oof` inside the committed pipeline, then applies the
Stanford-derived HIV-2 expert rule to the SAME sequences and performs a
sequence-clustered paired bootstrap of the difference in Pearson R.

Target = measured dG (from IC50). Neither the rule nor the model was fitted to
this target's rule form, so the comparison is not circular.

Output: RESCUE/BENCHMARK/results/head_to_head_paired.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parents[1]
sys.path.insert(0, str(BENCH))

import clinical_validation_final as cvf  # noqa: E402

SINGLES = ROOT / "stanford db" / "known data from stanford.xlsx"
COMBOS = ROOT / "stanford db" / "known data from stanford combinations.xlsx"
OUT = BENCH / "results" / "head_to_head_paired.json"
DRUGS = ["ATV/r", "DRV/r", "LPV/r", "SQV/r"]
HIV2_WT = ("PQFSLWKRPVVTAYIEGQPVEVLLDTGADDSIVAGIELGNNYSPKIVGGIGGFINTKEYKNV"
           "EIEVLNKKVRATIMTGDTPINIFGRNILTALGMSLNL")


# ---------- rule engine (same tables that reproduced Stanford exactly) -----
def parse_req(s):
    m = re.search(r"([A-Z]?)(\d+)([A-Z]+)", str(s).strip())
    return None if not m else {"pos": int(m.group(2)),
                               "alleles": list(m.group(3))}


def build_rules():
    singles = {str(r["Rule"]).strip(): r
               for _, r in pd.read_excel(SINGLES).iterrows()}
    combos = []
    for _, r in pd.read_excel(COMBOS).iterrows():
        reqs = [parse_req(p) for p in str(r["Combination Rule"]).split("+")]
        if all(reqs):
            combos.append({"reqs": reqs, "pen": r})
    return singles, combos


def muts_from_seq(seq, wt):
    return {f"{a}{i+1}{b}" for i, (a, b) in enumerate(zip(wt, seq))
            if b in "ACDEFGHIKLMNPQRSTVWY" and a != b}


def interpret(muts, singles, combos, drugs):
    clean = set(muts)
    sc = {d: 0.0 for d in drugs}
    for c in combos:
        ok, matched = True, set()
        for req in c["reqs"]:
            hit = False
            for m in clean:
                mm = re.search(r"(\d+)([A-Z]+)", m)
                if mm and int(mm.group(1)) == req["pos"] and \
                        any(a in req["alleles"] for a in mm.group(2)):
                    hit, _ = True, matched.add(m)
                    break
            if not hit:
                ok = False
                break
        if ok:
            for d in drugs:
                v = pd.to_numeric(c["pen"].get(d, 0), errors="coerce")
                sc[d] += 0.0 if pd.isna(v) else float(v)
            clean -= matched
    for m in list(clean):
        if m in singles:
            for d in drugs:
                v = pd.to_numeric(singles[m].get(d, 0), errors="coerce")
                sc[d] += 0.0 if pd.isna(v) else float(v)
            clean.discard(m)
    return sc


# ---------- capture model OOF ---------------------------------------------
CAP = {}
_orig_oof = cvf.kfold_oof


def _patched_oof(X, y, factory, seed):
    t, p = _orig_oof(X, y, factory, seed)
    CAP.setdefault("oof", []).append((np.asarray(t, float),
                                      np.asarray(p, float)))
    return t, p


cvf.kfold_oof = _patched_oof

print("Running committed pipeline (capturing ChEMBL OOF) ...")
ctx = {"drugs": list(cvf.COMMON_DRUGS)}
ctx = cvf.phase1_load(ctx)
ctx = cvf.phase2_bundle(ctx)
ctx = cvf.phase3_pretrain_sanity(ctx)

bundle = ctx["bundle"]
oof_by_drug = {}
for (t, p) in CAP["oof"]:
    oof_by_drug.setdefault(len(oof_by_drug), (t, p))

# rebuild per-drug pairing deterministically (same drug order as phase3)
rows = []
idx = 0
for drug in cvf.COMMON_DRUGS:
    mask = bundle.drug2 == drug
    if mask.sum() < 8:
        continue
    t, p = CAP["oof"][idx]
    idx += 1
    seqs = bundle.seq2[mask]
    for s, yt, yp in zip(seqs, t, p):
        rows.append({"seq": str(s), "drug": drug, "dG": float(yt),
                     "model": float(yp)})
df = pd.DataFrame(rows)
print("paired rows:", len(df), "| unique seqs:", df.seq.nunique())

singles, combos = build_rules()
cache = {}
df["rule"] = [cache.setdefault(s, interpret(muts_from_seq(s, HIV2_WT),
                                             singles, combos, DRUGS))[d]
              for s, d in zip(df.seq, df.drug)]


def pooled_r(g):
    return pearsonr(g.dG, g.model)[0], pearsonr(g.dG, g.rule)[0]


out = {"n_rows": int(len(df)), "n_sequences": int(df.seq.nunique()),
       "per_drug": {}}
print(f"\n{'drug':7} {'n':>4} {'model R':>9} {'rule R':>9} {'model-rule':>11}")
for drug in DRUGS:
    g = df[df.drug == drug]
    if len(g) < 8:
        continue
    rm = pearsonr(g.dG, g.model)[0]
    rr = pearsonr(g.dG, g.rule)[0]
    out["per_drug"][drug] = {"n": int(len(g)),
                             "model_r": round(float(rm), 4),
                             "rule_r": round(float(rr), 4)}
    print(f"{drug:7} {len(g):>4} {rm:>9.4f} {rr:>9.4f} {rm-rr:>11.4f}")

rm_p, rr_p = pooled_r(df)
print(f"\nPOOLED (n={len(df)}): model R={rm_p:.4f}  rule R={rr_p:.4f}  "
      f"diff={rm_p-rr_p:.4f}")

# ---- sequence-clustered paired bootstrap of the R difference -------------
rng = np.random.default_rng(42)
seqs = df.seq.unique()
diffs, rms, rrs = [], [], []
for _ in range(2000):
    pick = rng.choice(seqs, size=len(seqs), replace=True)
    b = pd.concat([df[df.seq == s] for s in pick], ignore_index=True)
    if b.model.std() == 0 or b.rule.std() == 0:
        continue
    a = pearsonr(b.dG, b.model)[0]
    c = pearsonr(b.dG, b.rule)[0]
    rms.append(a)
    rrs.append(c)
    diffs.append(a - c)
diffs = np.array(diffs)
lo, hi = np.percentile(diffs, [2.5, 97.5])
out["pooled"] = {
    "n": int(len(df)),
    "model_r": round(float(rm_p), 4),
    "rule_r": round(float(rr_p), 4),
    "difference": round(float(rm_p - rr_p), 4),
    "boot_model_ci": [round(float(np.percentile(rms, 2.5)), 4),
                      round(float(np.percentile(rms, 97.5)), 4)],
    "boot_rule_ci": [round(float(np.percentile(rrs, 2.5)), 4),
                     round(float(np.percentile(rrs, 97.5)), 4)],
    "boot_diff_ci": [round(float(lo), 4), round(float(hi), 4)],
    "boot_p_diff_le_0": round(float((diffs <= 0).mean()), 4),
    "n_boot": int(len(diffs)),
}
print(f"bootstrap (2000, clustered by sequence):")
print(f"  model R 95% CI: {out['pooled']['boot_model_ci']}")
print(f"  rule  R 95% CI: {out['pooled']['boot_rule_ci']}")
print(f"  diff 95% CI   : {out['pooled']['boot_diff_ci']}")
print(f"  P(diff<=0)    : {out['pooled']['boot_p_diff_le_0']}")

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print(f"\nwrote {OUT}")
