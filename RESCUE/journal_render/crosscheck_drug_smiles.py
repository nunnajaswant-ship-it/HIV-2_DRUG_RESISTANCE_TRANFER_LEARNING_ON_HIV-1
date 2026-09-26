"""
crosscheck_drug_smiles.py
=========================
Rigorous stereo-aware cross-check of the manuscript drug SMILES
(make_fig_structures.py) against two authoritative sources:

  1. PubChem (Absolute SMILES + InChI from the PUG record endpoint)
  2. RCSB chemcomp descriptors (SMILES_stereo from data.rcsb.org)

Comparison is done on canonical RDKit InChI (stereo-aware), NOT on the
SMILES strings, so aromatic-ring / tautomer spelling differences do not
produce false mismatches.
"""
from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import inchi

HERE = Path(__file__).resolve().parent
LIG = HERE / "structures" / "ligands"

# Exactly as defined in final_paper/03_figures/make_fig_structures.py
MANUSCRIPT = {
    "Darunavir": "CC(C)CN(C[C@H]([C@H](CC1=CC=CC=C1)NC(=O)O[C@H]2CO[C@@H]3[C@H]2CCO3)O)S(=O)(=O)C4=CC=C(C=C4)N",
    "Atazanavir": "CC(C)(C)[C@@H](C(=O)N[C@@H](CC1=CC=CC=C1)[C@H](CN(CC2=CC=C(C=C2)C3=CC=CC=N3)NC(=O)[C@H](C(C)(C)C)NC(=O)OC)O)NC(=O)OC",
    "Saquinavir": "CC(C)(C)NC(=O)[C@@H]1C[C@@H]2CCCC[C@@H]2CN1C[C@H]([C@H](CC3=CC=CC=C3)NC(=O)[C@H](CC(=O)N)NC(=O)C4=NC5=CC=CC=C5C=C4)O",
    "Lopinavir": "CC1=C(C(=CC=C1)C)OCC(=O)N[C@@H](CC2=CC=CC=C2)[C@H](C[C@H](CC3=CC=CC=C3)NC(=O)[C@H](C(C)C)N4CCCNC4=O)O",
}
# Stereo-aware Absolute SMILES from PubChem PUG record endpoint (captured)
PUBCHEM = {
    "Atazanavir": "CC(C)(C)[C@@H](C(=O)N[C@@H](CC1=CC=CC=C1)[C@H](CN(CC2=CC=C(C=C2)C3=CC=CC=N3)NC(=O)[C@H](C(C)(C)C)NC(=O)OC)O)NC(=O)OC",
    "Darunavir": "CC(C)CN(C[C@H]([C@H](CC1=CC=CC=C1)NC(=O)O[C@H]2CO[C@@H]3[C@H]2CCO3)O)S(=O)(=O)C4=CC=C(C=C4)N",
    "Lopinavir": "CC1=C(C(=CC=C1)C)OCC(=O)N[C@@H](CC2=CC=CC=C2)[C@H](C[C@H](CC3=CC=CC=C3)NC(=O)[C@H](C(C)C)N4CCCNC4=O)O",
    "Saquinavir": "CC(C)(C)NC(=O)[C@@H]1C[C@@H]2CCCC[C@@H]2CN1C[C@H]([C@H](CC3=CC=CC=C3)NC(=O)[C@H](CC(=O)N)NC(=O)C4=NC5=CC=CC=C5C=C4)O",
}
RCSB_FILES = {
    "Darunavir": LIG / "chemcomp_017_darunavir.json",
    "Atazanavir": LIG / "chemcomp_DR7_atazanavir.json",
    "Saquinavir": LIG / "chemcomp_ROC_saquinavir.json",
    "Lopinavir": LIG / "chemcomp_AB1_lopinavir.json",
}


def canon_inchi(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, 0
    return inchi.MolToInchi(mol), inchi.MolToInchiKey(mol), \
        sum(1 for a in mol.GetAtoms()
            if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)


report = []
for drug in ["Darunavir", "Atazanavir", "Saquinavir", "Lopinavir"]:
    m_i, m_k, m_centers = canon_inchi(MANUSCRIPT[drug])
    p_i, p_k, p_centers = canon_inchi(PUBCHEM[drug])
    rcsb_json = json.loads(RCSB_FILES[drug].read_text(encoding="utf-8"))
    rcsb_smiles = rcsb_json.get("rcsb_chem_comp_descriptor", {}).get("SMILES_stereo")
    r_i, r_k, r_centers = canon_inchi(rcsb_smiles) if rcsb_smiles else (None, None, 0)

    vs_pubchem = (m_i == p_i)
    vs_rcsb = (m_i == r_i)
    report.append({
        "drug": drug,
        "manuscript_inchi": m_i,
        "pubchem_inchi": p_i,
        "rcsb_inchi": r_i,
        "manu_vs_pubchem_MATCH": vs_pubchem,
        "manu_vs_rcsb_MATCH": vs_rcsb,
        "stereocenters": m_centers,
        "rcsb_atom_order_differs": (p_i == m_i or None),
    })
    print(f"{drug:12s} manuscript-InChI == PubChem : {vs_pubchem}  == RCSB-stereo : {vs_rcsb}  | stereo centers: {m_centers}")
    print(f"    manuscript: {m_i}")
    print(f"    pubchem  : {p_i}")
    if report[-1]["manu_vs_pubchem_MATCH"]:
        pass

out = HERE / "figures" / "drug_smiles_crosscheck.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\nWrote {out}")