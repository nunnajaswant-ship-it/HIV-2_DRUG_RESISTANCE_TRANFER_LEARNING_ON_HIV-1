"""
head_to_head_measured.py
========================
NON-CIRCULAR head-to-head on MEASURED phenotypes.

Compares, on the same HIV-2 ChEMBL rows (measured IC50 -> dG):

  (A) the cross-species transfer model      (locked benchmark R, from manifest)
  (B) the Stanford-derived HIV-2 expert rule (applied here to the same sequences)

The rule engine is the same one that reproduced the Stanford clinical penalty
exactly (head_to_head_rules.py). Here the target is a MEASURED binding free
energy, not a rule, so the comparison is not circular.

Rule tables:
  stanford db/known data from stanford.xlsx             (single-mutation penalties)
  stanford db/known data from stanford combinations.xlsx(combination penalties)

Output: RESCUE/BENCHMARK/results/head_to_head_measured.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent
CHEMBL = ROOT / "chembl" / "HIV2_PROTEASE_ML_READY_DATASET.csv"
SINGLES = ROOT / "stanford db" / "known data from stanford.xlsx"
COMBOS = ROOT / "stanford db" / "known data from stanford combinations.xlsx"
OUT = BENCH / "results" / "head_to_head_measured.json"

HIV2_WT = ("PQFSLWKRPVVTAYIEGQPVEVLLDTGADDSIVAGIELGNNYSPKIVGGIGGFINTKEYKNV"
           "EIEVLNKKVRATIMTGDTPINIFGRNILTALGMSLNL")
DRUGS = ["ATV/r", "DRV/r", "LPV/r", "SQV/r"]
DRUG_MAP = {"ATAZANAVIR": "ATV/r", "DARUNAVIR": "DRV/r",
            "LOPINAVIR": "LPV/r", "SAQUINAVIR": "SQV/r"}
RT = 0.593  # kcal/mol at 298 K (same as the pipeline)


def parse_req(s):
    m = re.search(r"([A-Z]?)(\d+)([A-Z]+)", str(s).strip())
    if not m:
        return None
    return {"pos": int(m.group(2)), "alleles": list(m.group(3))}


def build_rules():
    singles = {}
    for _, r in pd.read_excel(SINGLES).iterrows():
        singles[str(r["Rule"]).strip()] = r
    combos = []
    for _, r in pd.read_excel(COMBOS).iterrows():
        parts = str(r["Combination Rule"]).split("+")
        reqs = [parse_req(p) for p in parts]
        if all(reqs):
            combos.append({"reqs": reqs, "pen": r})
    return singles, combos


def mutations_from_seq(seq, wt):
    muts = set()
    for i, (a, b) in enumerate(zip(wt, seq)):
        if b not in "ACDEFGHIKLMNPQRSTVWY":
            continue
        if a != b:
            muts.add(f"{a}{i + 1}{b}")
    return muts


def interpret(muts, singles, combos, drugs):
    clean = set(muts)
    scores = {d: 0.0 for d in drugs}
    for c in combos:
        ok = True
        matched = set()
        for req in c["reqs"]:
            found = False
            for m in clean:
                mm = re.search(r"(\d+)([A-Z]+)", m)
                if mm and int(mm.group(1)) == req["pos"] and \
                        any(a in req["alleles"] for a in mm.group(2)):
                    found = True
                    matched.add(m)
                    break
            if not found:
                ok = False
                break
        if ok:
            for d in drugs:
                v = pd.to_numeric(c["pen"].get(d, 0), errors="coerce")
                scores[d] += 0.0 if pd.isna(v) else float(v)
            clean -= matched
    for m in list(clean):
        if m in singles:
            row = singles[m]
            for d in drugs:
                v = pd.to_numeric(row.get(d, 0), errors="coerce")
                scores[d] += 0.0 if pd.isna(v) else float(v)
            clean.discard(m)
    return scores


def main():
    singles, combos = build_rules()
    df = pd.read_csv(CHEMBL)
    df["drug"] = df["Compound_Name"].map(DRUG_MAP)
    df = df[df["drug"].notna()].copy()
    df["seq"] = df["Mutant_Sequence"].astype(str).str.strip().str.upper()
    df = df[df["seq"].str.len() == len(HIV2_WT)]
    # measured dG from IC50 (nM)
    df["dG"] = RT * np.log(df["Standard_Value"].astype(float) * 1e-9)

    cache = {}
    rows = []
    for _, r in df.iterrows():
        s = r["seq"]
        if s not in cache:
            cache[s] = interpret(mutations_from_seq(s, HIV2_WT), singles, combos,
                                 DRUGS)
        rows.append({"drug": r["drug"], "dG": r["dG"],
                     "rule": cache[s][r["drug"]],
                     "nmuts": len(mutations_from_seq(s, HIV2_WT))})
    d = pd.DataFrame(rows)
    print(f"rows: {len(d)}  sequences: {df['seq'].nunique()}")
    print(d.groupby("drug").size().to_string())

    out = {"n_rows": int(len(d)), "per_drug": {}}
    print(f"\n{'drug':7} {'n':>4} {'rule R vs dG':>13} {'rule rho':>9} "
          f"{'rule nonzero':>13}")
    for drug in DRUGS:
        g = d[d.drug == drug]
        if len(g) < 8:
            continue
        r = pearsonr(g.dG, g.rule)[0] if g.rule.std() > 0 else float("nan")
        rho = spearmanr(g.dG, g.rule)[0] if g.rule.std() > 0 else float("nan")
        out["per_drug"][drug] = {
            "n": int(len(g)),
            "rule_pearson_vs_measured_dG": round(float(r), 4),
            "rule_spearman": round(float(rho), 4),
            "rule_nonzero": int((g.rule != 0).sum()),
        }
        print(f"{drug:7} {len(g):>4} {r:>13.4f} {rho:>9.4f} "
              f"{int((g.rule != 0).sum()):>13}")

    if d.rule.std() > 0:
        rp = pearsonr(d.dG, d.rule)[0]
        sp = spearmanr(d.dG, d.rule)[0]
    else:
        rp = sp = float("nan")
    out["pooled"] = {"n": int(len(d)),
                     "rule_pearson_vs_measured_dG": round(float(rp), 4),
                     "rule_spearman": round(float(sp), 4)}
    print(f"\nPOOLED (n={len(d)}): rule R vs measured dG = {rp:.4f}  rho={sp:.4f}")

    man = json.load(open(BENCH / "results" /
                         "clinical_validation_final_results.json",
                         encoding="utf-8"))
    out["model_benchmark"] = {
        "note": "locked transfer fine-tune OOF on the 160-row 4-drug panel",
        "mean_pearson_r": man["sanity_check_chembl_oof"]["MEAN"]["pearson_r"],
    }
    print(f"model (locked benchmark, n=160): "
          f"R={man['sanity_check_chembl_oof']['MEAN']['pearson_r']}")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
