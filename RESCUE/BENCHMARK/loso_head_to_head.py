"""
loso_head_to_head.py
====================
Leave-one-sequence-out (LOSO) robustness of the measured-phenotype head-to-head.

The 160-row panel contains only 19 unique sequences, so the effective sample
size is 19, not 160. This script quantifies how much any single sequence drives
the headline result by recomputing the pooled Pearson R with each sequence
removed in turn.

Also reports unique-sequence counts (and the rule-blind subset) so that every
claim can be stated in terms of the effective n.

Output: RESCUE/BENCHMARK/results/loso_head_to_head.json
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
OUT = BENCH / "results" / "loso_head_to_head.json"
HIV2_WT = ("PQFSLWKRPVVTAYIEGQPVEVLLDTGADDSIVAGIELGNNYSPKIVGGIGGFINTKEYKNV"
           "EIEVLNKKVRATIMTGDTPINIFGRNILTALGMSLNL")
DRUGS = ["ATV/r", "DRV/r", "LPV/r", "SQV/r"]


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


def safe_r(a, b):
    if len(a) < 4 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b)[0])


def main():
    singles, combos = build_rules()

    CAP = {}
    _orig = cvf.kfold_oof

    def _patched(X, y, factory, seed):
        t, p = _orig(X, y, factory, seed)
        CAP.setdefault("oof", []).append((np.asarray(t, float),
                                          np.asarray(p, float)))
        return t, p

    cvf.kfold_oof = _patched
    ctx = {"drugs": list(cvf.COMMON_DRUGS)}
    ctx = cvf.phase1_load(ctx)
    ctx = cvf.phase2_bundle(ctx)
    ctx = cvf.phase3_pretrain_sanity(ctx)
    bundle = ctx["bundle"]

    rows, idx = [], 0
    for drug in cvf.COMMON_DRUGS:
        mask = bundle.drug2 == drug
        if mask.sum() < 8:
            continue
        t, p = CAP["oof"][idx]
        idx += 1
        for s, yt, yp in zip(bundle.seq2[mask], t, p):
            rows.append({"seq": str(s), "drug": drug,
                         "dG": float(yt), "model": float(yp)})
    df = pd.DataFrame(rows)
    cache = {}
    df["rule"] = [cache.setdefault(s, interpret(muts_from_seq(s, HIV2_WT),
                                                singles, combos, DRUGS))[d]
                  for s, d in zip(df.seq, df.drug)]

    out = {"n_rows": int(len(df)), "n_sequences": int(df.seq.nunique())}
    print(f"panel: {len(df)} rows, {df.seq.nunique()} unique sequences "
          f"(effective n = {df.seq.nunique()})")

    # ---------------- full-sample values ----------------------------------
    r_model = safe_r(df.dG.values, df.model.values)
    r_rule = safe_r(df.dG.values, df.rule.values)
    out["full"] = {"model_r": round(r_model, 4), "rule_r": round(r_rule, 4),
                   "diff": round(r_model - r_rule, 4)}
    print(f"\nfull sample: model R={r_model:.4f}  rule R={r_rule:.4f}  "
          f"diff={r_model-r_rule:.4f}")

    # ---------------- LOSO on the full panel ------------------------------
    seqs = sorted(df.seq.unique())
    models, rules, diffs = [], [], []
    for s in seqs:
        g = df[df.seq != s]
        a = safe_r(g.dG.values, g.model.values)
        b = safe_r(g.dG.values, g.rule.values)
        if np.isnan(a) or np.isnan(b):
            continue
        models.append(a)
        rules.append(b)
        diffs.append(a - b)
    models, rules, diffs = map(np.array, (models, rules, diffs))
    out["loso_full"] = {
        "n_folds": int(len(diffs)),
        "model_r_mean": round(float(models.mean()), 4),
        "model_r_min": round(float(models.min()), 4),
        "model_r_max": round(float(models.max()), 4),
        "rule_r_mean": round(float(rules.mean()), 4),
        "rule_r_min": round(float(rules.min()), 4),
        "rule_r_max": round(float(rules.max()), 4),
        "diff_mean": round(float(diffs.mean()), 4),
        "diff_min": round(float(diffs.min()), 4),
        "diff_max": round(float(diffs.max()), 4),
        "frac_folds_model_wins": round(float((diffs > 0).mean()), 4),
    }
    print(f"\nLOSO ({len(diffs)} folds):")
    print(f"  model R  mean {models.mean():.4f}  "
          f"range [{models.min():.4f}, {models.max():.4f}]")
    print(f"  rule  R  mean {rules.mean():.4f}  "
          f"range [{rules.min():.4f}, {rules.max():.4f}]")
    print(f"  diff     mean {diffs.mean():.4f}  "
          f"range [{diffs.min():.4f}, {diffs.max():.4f}]")
    print(f"  model wins in {100*(diffs>0).mean():.1f}% of folds")

    # ---------------- rule-blind subset -----------------------------------
    blind = df[df.rule == 0]
    out["rule_blind"] = {"n_rows": int(len(blind)),
                         "n_sequences": int(blind.seq.nunique())}
    rb = safe_r(blind.dG.values, blind.model.values)
    out["rule_blind"]["model_r"] = round(rb, 4)
    seqs_b = sorted(blind.seq.unique())
    bmodels = []
    for s in seqs_b:
        g = blind[blind.seq != s]
        a = safe_r(g.dG.values, g.model.values)
        if not np.isnan(a):
            bmodels.append(a)
    bmodels = np.array(bmodels)
    out["rule_blind"]["loso_model_r_mean"] = round(float(bmodels.mean()), 4)
    out["rule_blind"]["loso_model_r_min"] = round(float(bmodels.min()), 4)
    out["rule_blind"]["loso_model_r_max"] = round(float(bmodels.max()), 4)
    out["rule_blind"]["loso_frac_positive"] = round(
        float((bmodels > 0).mean()), 4)
    print(f"\nrule-blind subset: {len(blind)} rows, "
          f"{blind.seq.nunique()} unique sequences, model R={rb:.4f}")
    print(f"  LOSO model R mean {bmodels.mean():.4f}  "
          f"range [{bmodels.min():.4f}, {bmodels.max():.4f}]  "
          f"positive in {100*(bmodels>0).mean():.0f}% of folds")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
