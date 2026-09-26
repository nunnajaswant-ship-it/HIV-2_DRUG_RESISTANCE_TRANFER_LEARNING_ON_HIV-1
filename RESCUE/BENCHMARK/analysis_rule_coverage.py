"""
analysis_rule_coverage.py
=========================
Quantify how much of HIV-2 protease sequence space the rule-based
interpretation actually covers, and test whether the transferred model
retains signal precisely where the rules are silent.

Three questions:
  Q1  What fraction of the clinical benchmark label is non-zero?
      (i.e. how often does the rule-based interpretation say anything at all)
  Q2  What fraction of the measured ChEMBL panel does the rule score?
  Q3  On the measured panel, restricted to rule-blind rows (rule == 0),
      does the model still predict measured dG?

Output: RESCUE/BENCHMARK/results/rule_coverage.json
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parents[1]
sys.path.insert(0, str(BENCH))

import clinical_validation_final as cvf  # noqa: E402

SINGLES = ROOT / "stanford db" / "known data from stanford.xlsx"
COMBOS = ROOT / "stanford db" / "known data from stanford combinations.xlsx"
CLINICAL = ROOT / "known_info_master" / "HIV2_CLINICAL_ML_READY_FLAT.csv"
OUT = BENCH / "results" / "rule_coverage.json"

HIV2_WT = ("PQFSLWKRPVVTAYIEGQPVEVLLDTGADDSIVAGIELGNNYSPKIVGGIGGFINTKEYKNV"
           "EIEVLNKKVRATIMTGDTPINIFGRNILTALGMSLNL")
DRUGS = ["ATV/r", "DRV/r", "LPV/r", "SQV/r"]


# ---------------- rule engine (identical tables/logic to head_to_head) -----
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


# --------------------------------------------------------------------------
def main():
    report = {}
    singles, combos = build_rules()

    # rule vocabulary
    rule_vocab = set()
    for k in singles:
        mm = re.search(r"([A-Z]?)(\d+)([A-Z]+)", k)
        if mm:
            for a in mm.group(3):
                rule_vocab.add(f"{mm.group(2)}{a}")
    for c in combos:
        for req in c["reqs"]:
            for a in req["alleles"]:
                rule_vocab.add(f"{req['pos']}{a}")
    report["rule_vocabulary_positions"] = len(
        {re.sub(r"[A-Z]+$", "", v) for v in rule_vocab})
    report["rule_single_rules"] = len(singles)
    report["rule_combination_rules"] = len(combos)

    # ---------------- Q1: clinical label coverage --------------------------
    clin = pd.read_csv(CLINICAL)
    pen = pd.to_numeric(clin["Clinical_Penalty_Score"], errors="coerce").fillna(0)
    clin = clin.assign(_pen=pen)
    q1 = {"n_rows": int(len(clin)), "overall_zero_pct": round(
        float((clin._pen == 0).mean() * 100), 2)}
    per_drug = {}
    for drug, g in clin.groupby("Compound_Name"):
        per_drug[str(drug)] = {
            "n": int(len(g)),
            "zero_pct": round(float((g._pen == 0).mean() * 100), 1),
            "nonzero_n": int((g._pen != 0).sum()),
        }
    q1["per_drug"] = per_drug
    report["Q1_clinical_label_coverage"] = q1

    # ---------------- Q2/Q3: measured panel --------------------------------
    CAP = {}
    _orig_oof = cvf.kfold_oof

    def _patched_oof(X, y, factory, seed):
        t, p = _orig_oof(X, y, factory, seed)
        CAP.setdefault("oof", []).append((np.asarray(t, float),
                                          np.asarray(p, float)))
        return t, p

    cvf.kfold_oof = _patched_oof
    ctx = {"drugs": list(cvf.COMMON_DRUGS)}
    ctx = cvf.phase1_load(ctx)
    ctx = cvf.phase2_bundle(ctx)
    ctx = cvf.phase3_pretrain_sanity(ctx)
    bundle = ctx["bundle"]

    rows = []
    idx = 0
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

    q2 = {"n_rows": int(len(df)), "n_sequences": int(df.seq.nunique()),
          "zero_rule_pct": round(float((df.rule == 0).mean() * 100), 2),
          "per_drug": {}}
    for drug in DRUGS:
        g = df[df.drug == drug]
        if len(g) == 0:
            continue
        q2["per_drug"][drug] = {
            "n": int(len(g)),
            "zero_rule_pct": round(float((g.rule == 0).mean() * 100), 1),
            "nonzero_n": int((g.rule != 0).sum()),
        }
    report["Q2_measured_rule_coverage"] = q2

    # Q3: stratify
    blind = df[df.rule == 0]
    covered = df[df.rule != 0]
    q3 = {"n_rule_blind": int(len(blind)), "n_rule_covered": int(len(covered))}
    if len(blind) >= 8 and blind.model.std() > 0:
        q3["blind_model_r"] = round(float(pearsonr(blind.dG, blind.model)[0]), 4)
        q3["blind_model_rho"] = round(
            float(spearmanr(blind.dG, blind.model)[0]), 4)
        q3["blind_dG_span"] = round(
            float(blind.dG.max() - blind.dG.min()), 3)
        q3["blind_rule_r"] = None  # zero variance by construction
        # sequence-clustered bootstrap CI for the rule-blind model R
        rng = np.random.default_rng(7)
        seqs_b = blind.seq.unique()
        rs = []
        for _ in range(2000):
            pick = rng.choice(seqs_b, size=len(seqs_b), replace=True)
            b = pd.concat([blind[blind.seq == s] for s in pick],
                          ignore_index=True)
            if b.model.std() == 0 or b.dG.std() == 0:
                continue
            rs.append(pearsonr(b.dG, b.model)[0])
        rs = np.array(rs)
        q3["blind_model_r_ci"] = [round(float(np.percentile(rs, 2.5)), 4),
                                  round(float(np.percentile(rs, 97.5)), 4)]
        q3["blind_model_r_p_le_0"] = round(float((rs <= 0).mean()), 4)
        q3["n_boot"] = int(len(rs))
    if len(covered) >= 8 and covered.model.std() > 0 and \
            covered.rule.std() > 0:
        q3["covered_model_r"] = round(
            float(pearsonr(covered.dG, covered.model)[0]), 4)
        q3["covered_rule_r"] = round(
            float(pearsonr(covered.dG, covered.rule)[0]), 4)
    report["Q3_rule_blind_vs_covered"] = q3

    # ---------------- Q4: mutation-level coverage --------------------------
    observed = Counter()
    for s in df.seq.unique():
        for m in muts_from_seq(s, HIV2_WT):
            observed[m] += 1
    known = sum(1 for m in observed
                if (re.search(r"(\d+)([A-Z]+)", m) and
                    m in singles))
    q4 = {"distinct_observed_mutations": len(observed),
          "mutations_with_explicit_rule": int(known),
          "coverage_pct": round(100.0 * known / max(1, len(observed)), 1),
          "observed_mutations": [
              {"mutation": m, "n_seqs": int(c),
               "has_explicit_rule": bool(m in singles)}
              for m, c in observed.most_common()]}
    report["Q4_mutation_coverage"] = q4

    # ---------------- print ------------------------------------------------
    print("=" * 68)
    print("RULE COVERAGE ANALYSIS")
    print("=" * 68)
    print(f"rule tables: {len(singles)} single + {len(combos)} combination")
    print(f"\nQ1  clinical benchmark label (n={q1['n_rows']}): "
          f"{q1['overall_zero_pct']}% of rows have penalty == 0")
    for d, v in q1["per_drug"].items():
        print(f"      {d:22} n={v['n']:>4}  zero={v['zero_pct']:>5}%  "
              f"non-zero={v['nonzero_n']}")
    print(f"\nQ2  measured ChEMBL panel (n={q2['n_rows']}, "
          f"{q2['n_sequences']} seqs): {q2['zero_rule_pct']}% rule == 0")
    for d, v in q2["per_drug"].items():
        print(f"      {d:7} n={v['n']:>3}  zero={v['zero_rule_pct']:>5}%  "
              f"non-zero={v['nonzero_n']}")
    print(f"\nQ3  on the measured panel:")
    print(f"      rule-blind rows   : {q3['n_rule_blind']}"
          f"   model R = {q3.get('blind_model_r')}"
          f"  CI {q3.get('blind_model_r_ci')}"
          f"  P(R<=0)={q3.get('blind_model_r_p_le_0')}")
    print(f"      rule-covered rows : {q3['n_rule_covered']}"
          f"   model R = {q3.get('covered_model_r')}"
          f"   rule R = {q3.get('covered_rule_r')}")
    print(f"\nQ4  mutation-level: {q4['mutations_with_explicit_rule']} / "
          f"{q4['distinct_observed_mutations']} observed mutations have an "
          f"explicit rule ({q4['coverage_pct']}%)")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
