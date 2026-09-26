"""
Biological figure set for the HIV-1 -> HIV-2 cross-species transfer manuscript.

All figures use ONLY real experimental data:
  - Sequences: HIV-1 HXB2 (K03455) and HIV-2 ROD (M15390) wild-type references
    (opencode_shared_OLD/config.py, verified by tests/test_integration.py).
  - Structures: experimental X-ray PDB entries fetched from RCSB
    (RESCUE/journal_render/structures/):
      3S45  HIV-2 PR (ROD) + amprenavir, 1.51 A   (anchor, pocket source)
      4LL3  HIV-1 PR (subtype B LAI, HXB2-lineage) + darunavir, 1.40 A
      2AQU  HIV-1 PR + atazanavir
      3OXC  HIV-1 PR + saquinavir, 1.16 A
      1MUI  HIV-1 PR + lopinavir, 2.80 A
  - Pocket definitions (single source of truth):
      POCKET19_1INDEX (transfer pipeline, RESCUE/BENCHMARK/transfer_learning.py:115)
      POCKET_1INDEX   (16-residue 5A definition, opencode_shared_OLD/config.py:71)
  - Ligand codes: darunavir=017, atazanavir=DR7, saquinavir=ROC, lopinavir=AB1,
    amprenavir=478 (3S45).

No AI-generated structures. No AlphaFold/ESM-predicted coordinates.
Outputs: RESCUE/journal_render/figures/fig1_seq_alignment.png
         RESCUE/journal_render/figures/fig2_structure_overlay.png
         RESCUE/journal_render/figures/fig3_pocket19.png
         RESCUE/journal_render/figures/fig4_drug_poses.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from Bio.PDB import PDBParser, Superimposer, Selection

HERE = os.path.dirname(os.path.abspath(__file__))
STRUCT = os.path.join(HERE, "structures")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------------------
# Reference sequences (verified: tests/test_integration.py)
# ---------------------------------------------------------------------------
HIV1_WT = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
HIV2_WT = "PQFSLWKRPVVTAYIEGQPVEVLLDTGADDSIVAGIELGNNYSPKIVGGIGGFINTKEYKNVEIEVLNKKVRATIMTGDTPINIFGRNILTALGMSLNL"

# 1-indexed pocket definitions
POCKET19_1INDEX = [8, 23, 25, 26, 27, 28, 29, 30, 32, 45, 46, 47, 48, 50, 54, 76, 80, 82, 84]
POCKET_1INDEX = [8, 23, 25, 27, 28, 29, 30, 31, 32, 47, 48, 49, 50, 81, 82, 84]
CATALYTIC_DYAD = [25, 27]
FLAP_REGION = list(range(45, 56))

# BLOSUM62 (subset needed)
BLOSUM62 = {
    ('A','A'):4,('A','S'):1,('A','T'):0,('A','P'):-1,('A','G'):0,('A','V'):0,('A','C'):0,('A','D'):-2,('A','E'):-1,('A','N'):-2,('A','Q'):-1,('A','H'):-2,('A','R'):-1,('A','K'):-1,('A','M'):-1,('A','I'):-1,('A','L'):-1,('A','F'):-2,('A','W'):-3,('A','Y'):-2,
    ('S','S'):4,('S','T'):1,('S','P'):-1,('S','G'):0,('S','V'):-2,('S','C'):-1,('S','D'):0,('S','E'):0,('S','N'):1,('S','Q'):0,('S','H'):-1,('S','R'):-1,('S','K'):0,('S','M'):-1,('S','I'):-2,('S','L'):-2,('S','F'):-2,('S','W'):-3,('S','Y'):-2,
    ('T','T'):5,('T','P'):-1,('T','G'):-2,('T','V'):0,('T','C'):-1,('T','D'):0,('T','E'):0,('T','N'):0,('T','Q'):0,('T','H'):-2,('T','R'):-1,('T','K'):0,('T','M'):-1,('T','I'):-1,('T','L'):-1,('T','F'):-2,('T','W'):-2,('T','Y'):-2,
    ('P','P'):7,('P','G'):-2,('P','V'):-2,('P','C'):-3,('P','D'):-1,('P','E'):-2,('P','N'):-2,('P','Q'):-2,('P','H'):-2,('P','R'):-2,('P','K'):-1,('P','M'):-2,('P','I'):-3,('P','L'):-3,('P','F'):-4,('P','W'):-4,('P','Y'):-3,
    ('G','G'):6,('G','V'):-3,('G','C'):-3,('G','D'):-1,('G','E'):-2,('G','N'):0,('G','Q'):-2,('G','H'):-2,('G','R'):-2,('G','K'):-2,('G','M'):-3,('G','I'):-4,('G','L'):-4,('G','F'):-3,('G','W'):-2,('G','Y'):-3,
    ('V','V'):4,('V','C'):-1,('V','D'):-3,('V','E'):-2,('V','N'):-3,('V','Q'):-3,('V','H'):-3,('V','R'):-3,('V','K'):-2,('V','M'):1,('V','I'):3,('V','L'):1,('V','F'):-1,('V','W'):-3,('V','Y'):-1,
    ('C','C'):9,('C','D'):-3,('C','E'):-4,('C','N'):-3,('C','Q'):-3,('C','H'):-3,('C','R'):-3,('C','K'):-3,('C','M'):-1,('C','I'):-1,('C','L'):-1,('C','F'):-2,('C','W'):-2,('C','Y'):-2,
    ('D','D'):6,('D','E'):2,('D','N'):1,('D','Q'):0,('D','H'):-1,('D','R'):-2,('D','K'):-1,('D','M'):-3,('D','I'):-3,('D','L'):-4,('D','F'):-3,('D','W'):-4,('D','Y'):-3,
    ('E','E'):5,('E','N'):0,('E','Q'):2,('E','H'):0,('E','R'):0,('E','K'):1,('E','M'):-2,('E','I'):-3,('E','L'):-3,('E','F'):-3,('E','W'):-3,('E','Y'):-2,
    ('N','N'):6,('N','Q'):0,('N','H'):1,('N','R'):0,('N','K'):0,('N','M'):-2,('N','I'):-3,('N','L'):-3,('N','F'):-3,('N','W'):-4,('N','Y'):-2,
    ('Q','Q'):5,('Q','H'):0,('Q','R'):1,('Q','K'):1,('Q','M'):0,('Q','I'):-3,('Q','L'):-2,('Q','F'):-3,('Q','W'):-2,('Q','Y'):-1,
    ('H','H'):8,('H','R'):0,('H','K'):-1,('H','M'):-2,('H','I'):-3,('H','L'):-3,('H','F'):-1,('H','W'):-2,('H','Y'):2,
    ('R','R'):5,('R','K'):2,('R','M'):-1,('R','I'):-3,('R','L'):-2,('R','F'):-3,('R','W'):-3,('R','Y'):-2,
    ('K','K'):5,('K','M'):-1,('K','I'):-3,('K','L'):-2,('K','F'):-3,('K','W'):-3,('K','Y'):-2,
    ('M','M'):5,('M','I'):1,('M','L'):2,('M','F'):0,('M','W'):-1,('M','Y'):-1,
    ('I','I'):4,('I','L'):2,('I','F'):0,('I','W'):-3,('I','Y'):-1,
    ('L','L'):4,('L','F'):0,('L','W'):-2,('L','Y'):-1,
    ('F','F'):6,('F','W'):1,('F','Y'):3,
    ('W','W'):11,('W','Y'):2,
    ('Y','Y'):7,
}
def blosum(a, b):
    return BLOSUM62.get((a, b), BLOSUM62.get((b, a), -4))

# ---------------------------------------------------------------------------
# Figure 1: sequence alignment HXB2 vs ROD with pocket-19 + regions
# ---------------------------------------------------------------------------
def fig1_alignment():
    n = len(HIV1_WT)
    identical = [i for i in range(n) if HIV1_WT[i] == HIV2_WT[i]]
    similar = [i for i in range(n) if HIV1_WT[i] != HIV2_WT[i] and blosum(HIV1_WT[i], HIV2_WT[i]) > 0]
    fig, ax = plt.subplots(figsize=(13.5, 3.6), dpi=600)
    ax.set_xlim(0, n); ax.set_ylim(-0.6, 2.6)
    # region shading
    region_colors = {
        "active site (23-32)": (0.90, 0.95, 1.0),
        "flap (45-55)": (1.0, 0.95, 0.85),
        "wall / dimer interface (76-84)": (0.93, 0.97, 0.90),
    }
    ax.axvspan(22.5, 32.5, color=region_colors["active site (23-32)"], zorder=0)
    ax.axvspan(44.5, 55.5, color=region_colors["flap (45-55)"], zorder=0)
    ax.axvspan(75.5, 84.5, color=region_colors["wall / dimer interface (76-84)"], zorder=0)
    # residue cells
    for i in range(n):
        c = "0.82" if i + 1 in POCKET19_1INDEX else "white"
        if HIV1_WT[i] == HIV2_WT[i]:
            c = "#c6e0b4"  # identical
        elif blosum(HIV1_WT[i], HIV2_WT[i]) > 0:
            c = "#ffe699"  # similar
        ax.add_patch(Rectangle((i, 0.0), 1, 1, facecolor=c, edgecolor="0.85", lw=0.3))
        ax.add_patch(Rectangle((i, 1.0), 1, 1, facecolor=c, edgecolor="0.85", lw=0.3))
        ax.text(i + 0.5, 0.5, HIV2_WT[i], ha="center", va="center", fontsize=5.2, family="monospace")
        ax.text(i + 0.5, 1.5, HIV1_WT[i], ha="center", va="center", fontsize=5.2, family="monospace")
    # catalytic triad markers
    for pos in CATALYTIC_DYAD:
        ax.add_patch(Rectangle((pos - 1, -0.05), 1, 2.1, fill=False, edgecolor="#c00000", lw=1.4))
    # pocket-19 bracket markers (top)
    for pos in POCKET19_1INDEX:
        ax.plot([pos - 0.5, pos - 0.5], [2.02, 2.12], color="#1f4e79", lw=0.8)
        ax.plot([pos + 0.5, pos + 0.5], [2.02, 2.12], color="#1f4e79", lw=0.8)
    ax.plot([min(POCKET19_1INDEX) - 0.5, max(POCKET19_1INDEX) + 0.5], [2.12, 2.12], color="#1f4e79", lw=1.0)
    ax.text((min(POCKET19_1INDEX) + max(POCKET19_1INDEX)) / 2, 2.22, "pocket-19 (5 A of bound ligand, 3S45)",
            ha="center", va="bottom", fontsize=7.5, color="#1f4e79")
    # position numbers every 10
    for i in range(0, n, 10):
        ax.text(i + 0.5, -0.35, str(i + 1), ha="center", va="top", fontsize=4.6, color="0.35")
    ax.text(-0.6, 0.5, "HIV-2 ROD", ha="right", va="center", fontsize=9, style="italic")
    ax.text(-0.6, 1.5, "HIV-1 HXB2", ha="right", va="center", fontsize=9, style="italic")
    ax.set_yticks([]); ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#c6e0b4", label=f"identical ({len(identical)}/99 = 48.5%)"),
        Patch(facecolor="#ffe699", label=f"similar, BLOSUM62>0 ({len(similar)})"),
        Patch(facecolor="white", edgecolor="0.85", label="divergent"),
        Patch(facecolor="none", edgecolor="#c00000", label="catalytic triad D25-T26-G27"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.62),
              ncol=4, fontsize=7, frameon=False)
    fig.suptitle("HIV-1 (HXB2) vs HIV-2 (ROD) protease: 48.5% sequence identity, conserved catalytic triad",
                 fontsize=10, y=0.98)
    fig.tight_layout()
    out = os.path.join(FIG, "fig1_seq_alignment.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("fig1:", out, "identical", len(identical), "similar", len(similar))

# ---------------------------------------------------------------------------
# PDB helpers
# ---------------------------------------------------------------------------
def get_ca_atoms(pdb_path, chain_id="A"):
    """Return list of (residue_number, CA atom) for standard residues 1-99."""
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("s", pdb_path)
    model = s[0]
    chain = model[chain_id]
    atoms = []
    for res in chain:
        if res.id[0] != " ":
            continue
        if "CA" in res:
            atoms.append((res.id[1], res["CA"]))
    return atoms

def get_ligand_atoms(pdb_path, chain_id=None):
    """Return list of (resname, atom) for hetero residues (ligands).

    If chain_id is None, searches ALL chains (ligands may sit in chain B).
    """
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("s", pdb_path)
    model = s[0]
    chains = [model[chain_id]] if chain_id else list(model)
    atoms = []
    for chain in chains:
        for res in chain:
            if res.id[0] != " ":
                for atom in res:
                    atoms.append((res.resname, atom))
    return atoms

def get_ca_atoms_all_chains(pdb_path):
    """Return list of (chain_id, residue_number, CA atom) for standard residues."""
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("s", pdb_path)
    model = s[0]
    atoms = []
    for chain in model:
        for res in chain:
            if res.id[0] != " ":
                continue
            if "CA" in res:
                atoms.append((chain.id, res.id[1], res["CA"]))
    return atoms

def superimpose(mobile_atoms, target_atoms):
    """Kabsch superimpose mobile onto target (lists of Atom). Returns (rmsd, rot, tran)."""
    sup = Superimposer()
    sup.set_atoms(target_atoms, mobile_atoms)
    sup.apply(mobile_atoms)
    return sup.rms, sup.rotran

# ---------------------------------------------------------------------------
# Figure 2: structure overlay 3S45 (HIV-2 ROD) vs 4LL3 (HIV-1 LAI)
# ---------------------------------------------------------------------------
def fig2_overlay():
    pdb2 = os.path.join(STRUCT, "3S45.pdb")   # HIV-2 ROD + amprenavir
    pdb1 = os.path.join(STRUCT, "4LL3.pdb")   # HIV-1 subtype B LAI + darunavir
    ca2 = get_ca_atoms_all_chains(pdb2)   # (chain, resnum, atom)
    ca1 = get_ca_atoms_all_chains(pdb1)
    # match by (chain, resnum) — homodimer chains A/B
    d2 = {(c, r): a for c, r, a in ca2}
    d1 = {(c, r): a for c, r, a in ca1}
    common = sorted(set(d2) & set(d1))
    a2 = [d2[k] for k in common]
    a1 = [d1[k] for k in common]
    rmsd, (rot, tran) = superimpose(a1, a2)
    coords2 = np.array([a.coord for a in a2])
    coords1 = np.array([a.coord for a in a1])
    fig = plt.figure(figsize=(7.2, 6.2), dpi=600)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(coords2[:, 0], coords2[:, 1], coords2[:, 2], color="#1f4e79", lw=1.6, label="HIV-2 ROD (3S45, 1.51 A)")
    ax.plot(coords1[:, 0], coords1[:, 1], coords1[:, 2], color="#c00000", lw=1.6, label="HIV-1 LAI (4LL3, 1.40 A)")
    # catalytic dyad D25 (both chains) markers
    for (c, r), a in d2.items():
        if r in (25, 27):
            ax.scatter(*a.coord, color="#1f4e79", s=40, depthshade=False)
    for (c, r), a in d1.items():
        if r in (25, 27):
            ax.scatter(*a.coord, color="#c00000", s=40, depthshade=False)
    ax.set_title(f"C-alpha backbone superimposition (homodimer)\nRMSD = {rmsd:.2f} A over {len(common)} C-alpha atoms (Kabsch)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_axis_off()
    fig.tight_layout()
    out = os.path.join(FIG, "fig2_structure_overlay.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("fig2:", out, "RMSD", round(rmsd, 3), "n", len(common))

# ---------------------------------------------------------------------------
# Figure 3: pocket-19 highlighted on 3S45 with regions
# ---------------------------------------------------------------------------
def fig3_pocket19():
    pdb = os.path.join(STRUCT, "3S45.pdb")
    ca = dict(get_ca_atoms(pdb))
    lig = get_ligand_atoms(pdb)  # all chains (amprenavir 478 sits in chain B)
    # keep only the actual drug ligand (amprenavir, residue 478); drop waters/ions
    lig = [(rn, a) for rn, a in lig if rn == "478"]
    # region assignment for pocket-19
    def region(pos):
        if pos in CATALYTIC_DYAD:
            return "catalytic (D25/T26/G27)", "#c00000"
        if pos in FLAP_REGION:
            return "flap (45-55)", "#e36c09"
        if 76 <= pos <= 84:
            return "wall / dimer interface (76-84)", "#2e7d32"
        if 23 <= pos <= 32:
            return "active-site cleft (23-32)", "#1f4e79"
        return "other pocket", "#7f7f7f"
    fig = plt.figure(figsize=(7.2, 6.2), dpi=600)
    ax = fig.add_subplot(111, projection="3d")
    # backbone trace
    coords = np.array([a.coord for _, a in sorted(ca.items())])
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color="0.75", lw=1.0, alpha=0.7)
    # pocket-19 residues as spheres
    for pos in POCKET19_1INDEX:
        if pos in ca:
            name, col = region(pos)
            ax.scatter(*ca[pos].coord, color=col, s=90, depthshade=False, edgecolors="k", linewidths=0.4)
    # ligand (amprenavir) sticks
    lig_coords = np.array([a.coord for _, a in lig])
    if len(lig_coords):
        ax.scatter(lig_coords[:, 0], lig_coords[:, 1], lig_coords[:, 2],
                   color="#ffd966", s=8, depthshade=False, label="amprenavir (bound, 3S45)")
    ax.set_title("HIV-2 protease (3S45, ROD): 19 pocket residues within 5 A of bound ligand",
                 fontsize=10)
    ax.set_axis_off()
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c00000", markersize=8, label="catalytic D25/T26/G27"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f4e79", markersize=8, label="active-site cleft (23-32)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e36c09", markersize=8, label="flap (45-55)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2e7d32", markersize=8, label="wall / dimer interface (76-84)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7f7f7f", markersize=8, label="other pocket (residue 8)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ffd966", markersize=8, label="amprenavir (bound)"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIG, "fig3_pocket19.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("fig3:", out, "pocket19 residues", len(POCKET19_1INDEX), "ligand atoms", len(lig_coords))

# ---------------------------------------------------------------------------
# Figure 4: drug poses - HIV-1 complex ligands superposed onto 3S45 pocket
# ---------------------------------------------------------------------------
def fig4_drug_poses():
    drugs = [
        ("Atazanavir", "2AQU", "DR7"),
        ("Darunavir", "4LL3", "017"),
        ("Saquinavir", "3OXC", "ROC"),
        ("Lopinavir", "1MUI", "AB1"),
    ]
    pdb2 = os.path.join(STRUCT, "3S45.pdb")
    ca2 = dict(get_ca_atoms(pdb2))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), dpi=600, subplot_kw={"projection": "3d"})
    for ax, (name, pdbid, ligcode) in zip(axes.ravel(), drugs):
        pdb1 = os.path.join(STRUCT, f"{pdbid}.pdb")
        ca1 = dict(get_ca_atoms(pdb1))
        common = sorted(set(ca2) & set(ca1))
        a2 = [ca2[r] for r in common]
        a1 = [ca1[r] for r in common]
        rmsd, (rot, tran) = superimpose(a1, a2)
        # backbone traces
        c2 = np.array([a.coord for a in a2])
        c1 = np.array([a.coord for a in a1])
        ax.plot(c2[:, 0], c2[:, 1], c2[:, 2], color="#1f4e79", lw=1.2, alpha=0.85, label="HIV-2 ROD (3S45)")
        ax.plot(c1[:, 0], c1[:, 1], c1[:, 2], color="0.6", lw=0.8, alpha=0.5)
        # ligand from HIV-1 complex, transformed into the 3S45 frame by the
        # SAME Kabsch rotation/translation that mapped the C-alpha atoms.
        # (Superimposer.apply only moves the atoms passed to it, so the ligand
        #  coordinates must be transformed explicitly with sup.rotran.)
        lig = get_ligand_atoms(pdb1)  # all chains
        lig_atoms = [(rn, a) for rn, a in lig if rn == ligcode]
        if not lig_atoms:
            lig_atoms = lig  # fallback: any hetero residue
        lc_raw = np.array([a.coord for _, a in lig_atoms])
        lc = (np.dot(lc_raw, rot) + tran) if len(lc_raw) else lc_raw
        if len(lc):
            ax.scatter(lc[:, 0], lc[:, 1], lc[:, 2], color="#ffd966", s=10, depthshade=False,
                       edgecolors="k", linewidths=0.3, label=f"{name} (from {pdbid})")
        # pocket-19 markers
        for pos in POCKET19_1INDEX:
            if pos in ca2:
                ax.scatter(*ca2[pos].coord, color="#c00000", s=22, depthshade=False, alpha=0.7)
        ax.set_title(f"{name} (HIV-1 {pdbid}) in HIV-2 3S45 pocket\nmodeled pose; C-alpha RMSD {rmsd:.2f} A",
                     fontsize=8.5)
        ax.set_axis_off()
        ax.legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Protease-inhibitor poses modeled into the HIV-2 (3S45) pocket\n"
                 "(experimental HIV-1 complex ligands superposed by C-alpha Kabsch alignment; no HIV-2 crystal with these drugs exists)",
                 fontsize=10, y=0.99)
    fig.tight_layout()
    out = os.path.join(FIG, "fig4_drug_poses.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("fig4:", out)

if __name__ == "__main__":
    fig1_alignment()
    fig2_overlay()
    fig3_pocket19()
    fig4_drug_poses()
    print("ALL FIGURES DONE")