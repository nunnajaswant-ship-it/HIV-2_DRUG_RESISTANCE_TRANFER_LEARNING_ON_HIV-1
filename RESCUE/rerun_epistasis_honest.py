"""
rerun_epistasis_honest.py
==========================
HONEST RERUN of the spatial pocket co-mutation / epistasis analysis
(spatial_epistasis_analyzer.py) on the CURRENT clinical cohort.

Why rerun: the legacy report (known_info_master/SPATIAL_EPISTASIS_REPORT.md)
claimed "771 unique clinical genotypes" and reported 19 epistatic pairs with
synergy values (e.g. L90M+A92T +17.3/+27.7). The honest cohort manifest
(RESCUE/clinical_validation_honest.json) establishes the current clinical
file contains 654 sequence entries (653 unique + 1 WT duplicate) x 8 drugs =
5,232 rows. The clinical data itself was never poisoned (the poison was
restricted to the HIV-1 Stanford training corpus), but the legacy pair counts
were computed on a different cohort version / dedup and MUST be recomputed.

Protocol: byte-for-byte mirrors spatial_epistasis_analyzer.py (same PDB 3S45,
same POCKET_1INDEX, same heavy-atom distance matrix, same mutation parsing
and per-drug synergy definition: synergy = max(0, mean_both - max(mean_i,
mean_j)) on Clinical_Penalty_Score). The only differences:
  - Input: current HIV2_CLINICAL_ML_READY_FLAT.csv (honest cohort)
  - Outputs: written to RESCUE/epistasis_rerun/ — legacy artifacts in
    known_info_master/ are NEVER overwritten.

Outputs:
  - RESCUE/epistasis_rerun/spatial_epistasis_matrix.npy
  - RESCUE/epistasis_rerun/SPATIAL_EPISTASIS_REPORT.md
  - RESCUE/epistasis_rerun/EPISTASIS_RERUN_SUMMARY.json  (honest numbers)
"""

import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opencode_shared.config import (
    PDB_PATH, HIV2_CLINICAL_CSV, POCKET_1INDEX,
)

RESCUE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(RESCUE, "epistasis_rerun")
os.makedirs(OUT_DIR, exist_ok=True)

NPY_OUTPUT = os.path.join(OUT_DIR, "spatial_epistasis_matrix.npy")
REPORT_PATH = os.path.join(OUT_DIR, "SPATIAL_EPISTASIS_REPORT.md")
SUMMARY_PATH = os.path.join(OUT_DIR, "EPISTASIS_RERUN_SUMMARY.json")

N_POCKET = len(POCKET_1INDEX)
t0 = time.time()

# -----------------------------------------------------------------------
# STEP 1: PDB heavy-atom coordinates (identical to legacy analyzer)
# -----------------------------------------------------------------------
if not os.path.exists(PDB_PATH):
    raise FileNotFoundError(f"PDB file not found: {PDB_PATH}")

parser = PDBParser(QUIET=True)
structure = parser.get_structure("3S45", PDB_PATH)
chain_a = structure[0]["A"]
chain_b = structure[0]["B"]

coords_a = {res.id[1]: res for res in chain_a
            if res.id[0] == " " and 1 <= res.id[1] <= 99}
coords_b = {res.id[1]: res for res in chain_b
            if res.id[0] == " " and 1 <= res.id[1] <= 99}


def min_heavy_atom_dist(res1, res2):
    min_d = 999.0
    for a1 in res1:
        if a1.element == "H":
            continue
        for a2 in res2:
            if a2.element == "H":
                continue
            d = np.linalg.norm(a1.get_coord() - a2.get_coord())
            if d < min_d:
                min_d = d
    return min_d


pocket_dist = np.full((N_POCKET, N_POCKET), 999.0)
pocket_coupling = [[""] * N_POCKET for _ in range(N_POCKET)]
for i, ri in enumerate(POCKET_1INDEX):
    for j, rj in enumerate(POCKET_1INDEX):
        if i == j:
            pocket_dist[i][j] = 0.0
            pocket_coupling[i][j] = "Self"
            continue
        d_intra = (min_heavy_atom_dist(coords_a[ri], coords_a[rj])
                   if (ri in coords_a and rj in coords_a) else 999.0)
        d_inter1 = (min_heavy_atom_dist(coords_a[ri], coords_b[rj])
                    if (ri in coords_a and rj in coords_b) else 999.0)
        d_inter2 = (min_heavy_atom_dist(coords_b[ri], coords_a[rj])
                    if (ri in coords_b and rj in coords_a) else 999.0)
        d_inter = min(d_inter1, d_inter2)
        d_min = min(d_intra, d_inter)
        pocket_dist[i][j] = d_min
        if d_min < 5.0:
            pocket_coupling[i][j] = ("Inter-chain" if d_min == d_inter
                                     else "Intra-chain")

n_close = int(np.sum((pocket_dist > 0) & (pocket_dist < 5.0)))

# -----------------------------------------------------------------------
# STEP 2: Clinical cohort (current honest file)
# -----------------------------------------------------------------------
df_clinical = pd.read_csv(HIV2_CLINICAL_CSV, low_memory=False)


def parse_mutations(mut_str):
    if pd.isna(mut_str) or not isinstance(mut_str, str):
        return []
    return [m.strip().upper() for m in re.split(r"[,\s]+", mut_str)
            if m.strip()]


def has_mutation_at(muts_list, pos):
    for m in muts_list:
        m_obj = re.search(r"\d+", m)
        if m_obj and int(m_obj.group()) == pos:
            return m
    return None


unique_muts = df_clinical[["SequenceID", "Mutations"]].drop_duplicates(
    subset="Mutations")
all_mutation_sets = [parse_mutations(m) for m in unique_muts["Mutations"]]
n_unique_genotypes = len(all_mutation_sets)

# -----------------------------------------------------------------------
# STEP 3: Co-mutation frequency matrix
# -----------------------------------------------------------------------
co_mut_matrix = np.zeros((N_POCKET, N_POCKET), dtype=np.float32)
for i, ri in enumerate(POCKET_1INDEX):
    for j, rj in enumerate(POCKET_1INDEX):
        count_both = 0
        for muts in all_mutation_sets:
            mi = has_mutation_at(muts, ri)
            mj = has_mutation_at(muts, rj)
            if mi and mj:
                count_both += 1
        co_mut_matrix[i][j] = count_both

max_count = co_mut_matrix.max()
normalized = co_mut_matrix / max_count if max_count > 0 else co_mut_matrix.copy()
np.save(NPY_OUTPUT, normalized)

# -----------------------------------------------------------------------
# STEP 4: Per-drug epistasis (synergy on Clinical_Penalty_Score)
# -----------------------------------------------------------------------
df_geno = df_clinical.groupby("Mutations").agg({
    "Mutant_Sequence": "first",
    "SequenceID": "first",
}).reset_index()

drugs = sorted(df_clinical["Compound_Name"].unique())
drugs = [d for d in drugs if str(d) != "nan"]

results = []
for i, ri in enumerate(POCKET_1INDEX):
    for j, rj in enumerate(POCKET_1INDEX):
        if i >= j:
            continue
        if pocket_dist[i][j] >= 5.0:
            continue
        mutation_combos = set()
        for muts in all_mutation_sets:
            mi = has_mutation_at(muts, ri)
            mj = has_mutation_at(muts, rj)
            if mi and mj:
                mutation_combos.add((mi, mj))
        for mut_a, mut_b in sorted(mutation_combos):
            scores = {d: {"only_a": [], "only_b": [], "both": []}
                      for d in drugs}
            for muts_str in df_geno["Mutations"]:
                muts = parse_mutations(muts_str)
                mi = has_mutation_at(muts, ri)
                mj = has_mutation_at(muts, rj)
                subset = df_clinical[df_clinical["Mutations"] == muts_str]
                if mi == mut_a and mj == mut_b:
                    for d in drugs:
                        vals = subset[subset["Compound_Name"] == d][
                            "Clinical_Penalty_Score"].values
                        if len(vals):
                            scores[d]["both"].append(float(vals[0]))
                elif mi == mut_a and not mj:
                    for d in drugs:
                        vals = subset[subset["Compound_Name"] == d][
                            "Clinical_Penalty_Score"].values
                        if len(vals):
                            scores[d]["only_a"].append(float(vals[0]))
                elif mj == mut_b and not mi:
                    for d in drugs:
                        vals = subset[subset["Compound_Name"] == d][
                            "Clinical_Penalty_Score"].values
                        if len(vals):
                            scores[d]["only_b"].append(float(vals[0]))
            both_count = len(scores[drugs[0]]["both"]) if drugs else 0
            if both_count == 0:
                continue
            row = {
                "res_i": ri, "res_j": rj,
                "mut_a": mut_a, "mut_b": mut_b,
                "distance": round(pocket_dist[i][j], 2),
                "coupling_type": pocket_coupling[i][j],
                "count_only_a": len(scores[drugs[0]]["only_a"]),
                "count_only_b": len(scores[drugs[0]]["only_b"]),
                "count_both": both_count,
            }
            for d in drugs:
                dkey = d.replace("/", "_").replace(" ", "_")
                m_i = np.mean(scores[d]["only_a"]) if scores[d]["only_a"] else 0.0
                m_j = np.mean(scores[d]["only_b"]) if scores[d]["only_b"] else 0.0
                m_both = np.mean(scores[d]["both"]) if scores[d]["both"] else 0.0
                synergy = max(0.0, m_both - max(m_i, m_j))
                row[f"{dkey}_i"] = round(m_i, 1)
                row[f"{dkey}_j"] = round(m_j, 1)
                row[f"{dkey}_both"] = round(m_both, 1)
                row[f"{dkey}_synergy"] = round(synergy, 1)
            results.append(row)

df_results = pd.DataFrame(results)
if len(df_results):
    df_results = df_results.sort_values(by="count_both", ascending=False)

# -----------------------------------------------------------------------
# STEP 5: Report (mirror of legacy report, written to RESCUE only)
# -----------------------------------------------------------------------
report_md = f"""# HIV-2 Protease 3D Spatial Pocket Epistasis Report — HONEST RERUN
## Benchmarked against PDB 3S45 and Current Clinical Cohort (genuine data)

**Pocket residues (16):** {', '.join(str(r) for r in POCKET_1INDEX)}

**Unique clinical genotypes:** {n_unique_genotypes}
**Epistatic pairs identified:** {len(df_results)}

## Distance Matrix (Heavy-Atom, A)

| Residue | {' | '.join(str(r) for r in POCKET_1INDEX)} |
|---------|{'-|' * N_POCKET}
"""
for i, ri in enumerate(POCKET_1INDEX):
    row_vals = " | ".join(
        f"{pocket_dist[i][j]:.1f}" if pocket_dist[i][j] < 999 else "---"
        for j in range(N_POCKET))
    report_md += f"| {ri} | {row_vals} |\n"

report_md += f"""

## Co-Mutation Frequency Matrix (Normalized)

| Residue | {' | '.join(str(r) for r in POCKET_1INDEX)} |
|---------|{'-|' * N_POCKET}
"""
for i, ri in enumerate(POCKET_1INDEX):
    row_vals = " | ".join(f"{normalized[i][j]:.3f}" for j in range(N_POCKET))
    report_md += f"| {ri} | {row_vals} |\n"

if len(df_results):
    report_md += """

## Top Epistatic Mutation Pairs (honest cohort)
| Mutation Pair | Distance (A) | Coupling | Count Both |
|:---|---:|:---|---:|
"""
    for _, r in df_results.head(20).iterrows():
        report_md += (f"| {r['mut_a']} + {r['mut_b']} | {r['distance']} | "
                      f"{r['coupling_type']} | {r['count_both']} |\n")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_md)

# -----------------------------------------------------------------------
# Summary JSON with the honest numbers + legacy comparison
# -----------------------------------------------------------------------
l90m_a92t = None
for _, r in df_results.iterrows():
    if r["mut_a"] == "L90M" and r["mut_b"] == "A92T":
        l90m_a92t = r.to_dict()
        break
    if r["mut_b"] == "L90M" and r["mut_a"] == "A92T":
        l90m_a92t = r.to_dict()
        break

summary = {
    "pipeline": "RESCUE/rerun_epistasis_honest.py",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "input": HIV2_CLINICAL_CSV,
    "cohort": {
        "clinical_rows": int(len(df_clinical)),
        "sequence_entries_expected": 654,
        "unique_genotypes_by_mutations": n_unique_genotypes,
        "note": ("Honest cohort = 654 sequence entries (653 unique + 1 WT "
                 "dup) x 8 drugs = 5,232 rows. Legacy report claimed 771 "
                 "unique clinical genomes."),
    },
    "pocket_pairs_within_5A": n_close,
    "epistatic_pairs_found": int(len(df_results)),
    "legacy_values": {
        "claimed_unique_genotypes": 771,
        "claimed_epistatic_pairs": 19,
        "claimed_L90M_A92T_synergy": {"first_drug": 17.3, "second_drug": 27.7},
        "source": "known_info_master/SPATIAL_EPISTASIS_REPORT.md (legacy cohort)",
    },
    "L90M_A92T": l90m_a92t,
    "top_pairs_by_cooccurrence": (
        df_results.head(10).to_dict(orient="records")
        if len(df_results) else []),
    "run_seconds": round(time.time() - t0, 1),
}
with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)

print("unique genotypes:", n_unique_genotypes)
print("pairs within 5A:", n_close)
print("epistatic pairs:", len(df_results))
print("L90M+A92T:", json.dumps(l90m_a92t, indent=1, default=str) if l90m_a92t else None)
print("summary written:", SUMMARY_PATH)
