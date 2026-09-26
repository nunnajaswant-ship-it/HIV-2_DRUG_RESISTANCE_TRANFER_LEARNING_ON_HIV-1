"""
analysis_hiv2eu_coverage.py
===========================
Coverage of the OFFICIAL published HIV-2EU v4 protease-inhibitor rules
(Charpentier et al., Clin Infect Dis 2025; doi:10.1093/cid/ciaf110).

This addresses the strongest objection to the paper: that the "rule-based
interpretation is sparse" claim might only hold for our Stanford-derived
re-implementation. Here we score the SAME measured panel with the actual
published HIV-2EU v4 rule set.

HIV-2EU v4 protease rules (verbatim structure from the publication):
  Darunavir  resistance : I50V | I54M | I84V + L90M
             possible   : I84V | L90M
  Lopinavir  resistance : V47A | I54M | >=2 of {I82F, I84V, L90M}
             possible   : V62A + L99F | >=1 of {I82F, I84V, L90M}
  Atazanavir, saquinavir, and all other PIs: NO rules
  (saquinavir was removed in v4 as no longer manufactured; atazanavir is
   classified as naturally resistant and therefore not scored)

Output: RESCUE/BENCHMARK/results/hiv2eu_coverage.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent
CHEMBL = ROOT / "chembl" / "HIV2_PROTEASE_ML_READY_DATASET.csv"
CLINICAL = ROOT / "known_info_master" / "HIV2_CLINICAL_ML_READY_FLAT.csv"
OUT = BENCH / "results" / "hiv2eu_coverage.json"

HIV2_WT = ("PQFSLWKRPVVTAYIEGQPVEVLLDTGADDSIVAGIELGNNYSPKIVGGIGGFINTKEYKNV"
           "EIEVLNKKVRATIMTGDTPINIFGRNILTALGMSLNL")
DRUG_MAP = {"ATAZANAVIR": "ATV/r", "DARUNAVIR": "DRV/r",
            "LOPINAVIR": "LPV/r", "SAQUINAVIR": "SQV/r"}

# ---- the official HIV-2EU v4 protease vocabulary -------------------------
HIV2EU_VOCAB = {
    "I50V", "I54M", "I84V", "L90M",          # darunavir
    "V47A", "I82F",                            # lopinavir (resistance)
    "V62A", "L99F",                            # lopinavir (possible)
}
CORE3 = {"I82F", "I84V", "L90M"}


def muts_from_seq(seq, wt):
    return {f"{a}{i+1}{b}" for i, (a, b) in enumerate(zip(wt, seq))
            if b in "ACDEFGHIKLMNPQRSTVWY" and a != b}


def hiv2eu_interpret(muts, drug):
    """Return (level, triggered) where level in {none, possible, resistant,
    norule}. `triggered` lists the mutations responsible."""
    if drug not in ("DRV/r", "LPV/r"):
        return "norule", []
    m = set(muts)
    got = sorted(m & HIV2EU_VOCAB)
    if drug == "DRV/r":
        res = ("I50V" in m) or ("I54M" in m) or ({"I84V", "L90M"} <= m)
        poss = ("I84V" in m) or ("L90M" in m)
    else:  # LPV/r
        res = ("V47A" in m) or ("I54M" in m) or (len(m & CORE3) >= 2)
        poss = ({"V62A", "L99F"} <= m) or (len(m & CORE3) >= 1)
    if res:
        return "resistant", got
    if poss:
        return "possible", got
    return "none", got


def main():
    rep = {}

    # ---------------- measured ChEMBL panel --------------------------------
    df = pd.read_csv(CHEMBL)
    df["drug"] = df["Compound_Name"].map(DRUG_MAP)
    df = df[df["drug"].notna()].copy()
    df["seq"] = df["Mutant_Sequence"].astype(str).str.strip().str.upper()
    df = df[df["seq"].str.len() == len(HIV2_WT)]

    cache = {}
    lv, got = [], []
    for s, d in zip(df["seq"], df["drug"]):
        if s not in cache:
            cache[s] = muts_from_seq(s, HIV2_WT)
        a, b = hiv2eu_interpret(cache[s], d)
        lv.append(a)
        got.append(b)
    df["hiv2eu_level"] = lv
    df["hiv2eu_hits"] = got
    df["n_hits"] = [len(x) for x in got]

    per_drug = {}
    for drug, g in df.groupby("drug"):
        per_drug[str(drug)] = {
            "n": int(len(g)),
            "no_rule_pct": round(float((g.hiv2eu_level == "none").mean() * 100), 1),
            "norule_scope_pct": round(
                float((g.hiv2eu_level == "norule").mean() * 100), 1),
            "resistant_n": int((g.hiv2eu_level == "resistant").sum()),
            "possible_n": int((g.hiv2eu_level == "possible").sum()),
        }
    all_muts = set()
    for s in cache:
        all_muts |= cache[s]
    covered = sorted(all_muts & HIV2EU_VOCAB)
    rep["measured_panel"] = {
        "n_rows": int(len(df)),
        "n_sequences": int(df["seq"].nunique()),
        "unscored_pct": round(
            float((df.hiv2eu_level.isin(["none", "norule"])).mean() * 100), 1),
        "norule_drug_pct": round(
            float((df.hiv2eu_level == "norule").mean() * 100), 1),
        "per_drug": per_drug,
        "distinct_observed_mutations": len(all_muts),
        "mutations_in_hiv2eu_vocab": len(covered),
        "covered_list": covered,
        "coverage_pct": round(100.0 * len(covered) / max(1, len(all_muts)), 1),
        "uncovered_list": sorted(all_muts - HIV2EU_VOCAB),
    }

    # ---------------- clinical panel scope ---------------------------------
    cl = pd.read_csv(CLINICAL)
    drugs = sorted(cl["Compound_Name"].astype(str).unique())
    scored = [d for d in drugs if d in ("DRV/r", "LPV/r")]
    rep["clinical_panel"] = {
        "n_drugs": len(drugs),
        "drugs": drugs,
        "drugs_with_hiv2eu_pi_rules": scored,
        "drugs_without_rules": [d for d in drugs if d not in scored],
        "pct_drugs_without_rules": round(
            100.0 * (len(drugs) - len(scored)) / max(1, len(drugs)), 1),
    }

    # ---------------- print ------------------------------------------------
    print("=" * 70)
    print("HIV-2EU v4 (official published rules) COVERAGE")
    print("=" * 70)
    mp = rep["measured_panel"]
    print(f"measured panel: {mp['n_rows']} rows, {mp['n_sequences']} sequences")
    print(f"  unscored by HIV-2EU : {mp['unscored_pct']}%")
    print(f"    of which no rules exist for the drug : {mp['norule_drug_pct']}%")
    for d, v in mp["per_drug"].items():
        print(f"  {d:7} n={v['n']:>3}  unscored={v['no_rule_pct']:>5}%  "
              f"(no-rules-for-drug={v['norule_scope_pct']:>5}%)  "
              f"resistant={v['resistant_n']}  possible={v['possible_n']}")
    print(f"\nmutation coverage: {mp['mutations_in_hiv2eu_vocab']} of "
          f"{mp['distinct_observed_mutations']} observed mutations "
          f"({mp['coverage_pct']}%)")
    print(f"  covered   : {mp['covered_list']}")
    print(f"\nclinical panel: {rep['clinical_panel']['n_drugs']} drugs, "
          f"{len(rep['clinical_panel']['drugs_without_rules'])} have NO "
          f"HIV-2EU PI rules -> {rep['clinical_panel']['drugs_without_rules']}")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
