"""
render_drug_panel_2d.py
=======================
ACS-style 2-D drug structure panel at 600 dpi (venue-independent asset).
Depicts the four ritonavir-boosted protease inhibitors using the verified
SMILES from make_fig_structures.py (cross-checked vs PubChem + RCSB).

ACS depiction conventions applied:
  * black bonds / white background (no color fills)
  * wedge bonds for tetrahedral stereocenters
  * no decorative transparency or gradients
  * panel labels (a-d) outside the structure area
  * 2x2 grid, tight padding, single/double-column compatible

Render path uses RDKit's native MolDraw2DCairo (no external Cairo DLL).

Output: RESCUE/journal_render/figures/fig_drug_structures_2d.png  (600 dpi)
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures" / "fig_drug_structures_2d.png"

# Verified stereo SMILES (canonical-InChI identical to PubChem & RCSB chemcomp)
DRUGS = [
    ("Darunavir", "DRV",
     "CC(C)CN(C[C@H]([C@H](CC1=CC=CC=C1)NC(=O)O[C@H]2CO[C@@H]3[C@H]2CCO3)O)S(=O)(=O)C4=CC=C(C=C4)N"),
    ("Atazanavir", "ATV",
     "CC(C)(C)[C@@H](C(=O)N[C@@H](CC1=CC=CC=C1)[C@H](CN(CC2=CC=C(C=C2)C3=CC=CC=N3)NC(=O)[C@H](C(C)(C)C)NC(=O)OC)O)NC(=O)OC"),
    ("Saquinavir", "SQV",
     "CC(C)(C)NC(=O)[C@@H]1C[C@@H]2CCCC[C@@H]2CN1C[C@H]([C@H](CC3=CC=CC=C3)NC(=O)[C@H](CC(=O)N)NC(=O)C4=NC5=CC=CC=C5C=C4)O"),
    ("Lopinavir", "LPV",
     "CC1=C(C(=CC=C1)C)OCC(=O)N[C@@H](CC2=CC=CC=C2)[C@H](C[C@H](CC3=CC=CC=C3)NC(=O)[C@H](C(C)C)N4CCCNC4=O)O"),
]

DPI = 600
W_IN, H_IN = 4.0, 3.4  # panel size in inches


def render_one(smiles: str) -> Image.Image:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"invalid SMILES: {smiles}"
    d = rdMolDraw2D.MolDraw2DCairo(int(W_IN * DPI), int(H_IN * DPI))
    o = d.drawOptions()
    o.bondLineWidth = 3
    o.padding = 0.08
    o.fixedBondLength = 32
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText())).convert("RGB")


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.4), dpi=DPI)
    for ax, (name, abbr, smi), lbl in zip(axes.flat, DRUGS, "abcd"):
        img = render_one(smi)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#000000")
            s.set_linewidth(0.8)
        ax.text(-0.055, 0.99, lbl, transform=ax.transAxes, fontsize=16,
                fontweight="bold", va="top", ha="left")
        ax.set_title(f"{name} ({abbr})", fontsize=13, pad=5)
    fig.suptitle("Protease inhibitors in ritonavir-boosted regimens",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB at {DPI} dpi)")


if __name__ == "__main__":
    main()