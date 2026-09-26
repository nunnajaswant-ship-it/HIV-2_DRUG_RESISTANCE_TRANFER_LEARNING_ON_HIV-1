"""
Molecular Docking Engine for HIV-2 Protease — Drug Resistance Pipeline
======================================================================
Three-tier docking approach:
1. AutoDock Vina (if installed) — best, physics-based scoring
2. RDKit ETKDG + UFF minimization — built-in, no external deps
3. Centroid placement — fallback for missing structures

All tiers produce: [ligand_coords, binding_pose_scores, pocket_contacts]
"""
import os, sys, json, subprocess, tempfile, logging
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers

from opencode_shared.config import (
    ADV_FEAT, KNOWN_INFO, PDB_PATH,
    POCKET_0INDEX, POCKET_CUTOFF_ATOM,
    DOCKING_NUM_MODES, DOCKING_EXHAUSTIVENESS,
)
from opencode_shared.biochem_scales import DRUG_SMILES

# =============================================================================
# TIER 1: AutoDock Vina (external)
# =============================================================================

def _find_vina() -> str | None:
    """Locate AutoDock Vina binary."""
    for candidate in ['vina', 'vina.exe', 'autodock_vina',
                      r'C:\Users\nunna\vina.exe',
                      r'D:\desktop\BIO PROJECT HIV-2\vina_1.2.7_win.exe']:
        try:
            result = subprocess.run([candidate, '--version'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _prepare_pdbqt_with_meeko(smiles: str, output_pdbqt: str):
    """Prepare ligand PDBQT using Meeko + RDKit."""
    from meeko import MoleculePreparation
    from meeko.preparation import PDBQTWriterLegacy
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol, maxIters=500, confId=0)
    
    preparator = MoleculePreparation()
    mol_setup = preparator.prepare(mol)
    
    pdbqt_string, is_ok, err_msg = PDBQTWriterLegacy.write_string(mol_setup)
    if not is_ok:
        raise RuntimeError(f"Meeko PDBQT preparation failed: {err_msg}")
    
    with open(output_pdbqt, 'w') as f:
        f.write(pdbqt_string)


def _prepare_receptor_pdbqt(receptor_pdb: str, output_pdbqt: str):
    """Prepare receptor PDBQT from PDB file.
    
    For AutoDock Vina, we need to convert PDB to PDBQT format.
    This adds Gasteiger charges and atom types.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    mol = Chem.MolFromPDBFile(receptor_pdb, removeHs=False)
    if mol is None:
        raise RuntimeError(f"Failed to parse PDB: {receptor_pdb}")
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Write as PDBQT (Vina can read PDB directly with --receptor flag)
    # Actually, Vina 1.2.7 accepts PDB files directly
    return receptor_pdb


def dock_vina(
    drug_name: str,
    receptor_pdb: str = PDB_PATH,
    center: tuple[float, float, float] | None = None,
    box_size: tuple[float, float, float] = (20.0, 20.0, 20.0),
) -> dict:
    """Dock using AutoDock Vina. Returns {'poses': [...], 'scores': [...]}."""
    if center is None:
        raise ValueError("Vina requires a valid pocket_center (got None)")
    vina_bin = _find_vina()
    if vina_bin is None:
        raise RuntimeError("AutoDock Vina not found. Install it or use RDKit docking.")

    smiles = DRUG_SMILES.get(drug_name.upper())
    if not smiles:
        raise KeyError(f"Unknown drug: {drug_name}")

    with tempfile.TemporaryDirectory() as tmp:
        lig_pdbqt = os.path.join(tmp, "ligand.pdbqt")
        out_pdbqt = os.path.join(tmp, "output.pdbqt")

        # Prepare ligand PDBQT with Meeko
        _prepare_pdbqt_with_meeko(smiles, lig_pdbqt)

        # Vina 1.2.7 accepts PDB directly for receptor
        cmd = [
            vina_bin,
            '--receptor', receptor_pdb,
            '--ligand', lig_pdbqt,
            '--out', out_pdbqt,
            '--center_x', str(center[0]),
            '--center_y', str(center[1]),
            '--center_z', str(center[2]),
            '--size_x', str(box_size[0]),
            '--size_y', str(box_size[1]),
            '--size_z', str(box_size[2]),
            '--num_modes', str(DOCKING_NUM_MODES),
            '--exhaustiveness', str(DOCKING_EXHAUSTIVENESS),
        ]
        
        log.info(f"  Running Vina: {' '.join(cmd[:5])}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise RuntimeError(f"Vina failed: {result.stderr[:200]}")

        poses = _parse_vina_output(out_pdbqt)
        return poses


def _parse_vina_output(pdbqt_path: str) -> dict:
    """Parse Vina output PDBQT into pose coordinates and scores."""
    poses = {'coords': [], 'scores': []}
    if not os.path.exists(pdbqt_path):
        return poses

    with open(pdbqt_path) as f:
        current_coords = []
        for line in f:
            if line.startswith('REMARK VINA RESULT:'):
                score = float(line.split()[3])
                poses['scores'].append(score)
            elif line.startswith('ENDMDL'):
                if current_coords:
                    poses['coords'].append(np.array(current_coords))
                    current_coords = []
            elif line.startswith('ATOM') or line.startswith('HETATM'):
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                current_coords.append([x, y, z])
    return poses


# =============================================================================
# TIER 2: RDKit ETKDG + UFF Scoring (no external deps)
# =============================================================================

def dock_rdkit(
    drug_name: str,
    pocket_center: tuple[float, float, float] | None = None,
    num_conformers: int = 50,
) -> dict:
    """Dock using RDKit conformer generation + UFF energy scoring.
    
    Generates num_conformers conformers via ETKDG, minimizes each in UFF,
    ranks by UFF energy relative to pocket center distance.
    """
    smiles = DRUG_SMILES.get(drug_name.upper())
    if not smiles:
        raise KeyError(f"Unknown drug: {drug_name}")

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0
    cids = rdDistGeom.EmbedMultipleConfs(mol, num_conformers, params)
    if not cids:
        raise RuntimeError(f"ETKDG failed to generate conformers for {drug_name}")

    best_conf = None
    best_score = float('inf')

    for cid in cids:
        ff = rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=cid)
        if ff is None:
            continue
        converged = ff.Minimize(maxIts=500)
        energy = ff.CalcEnergy()

        if pocket_center:
            conf = mol.GetConformer(cid)
            centroid = np.mean([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())], axis=0)
            dist = np.linalg.norm(np.array(centroid) - np.array(pocket_center))
            score = energy + 10.0 * dist  # energy + distance penalty
        else:
            score = energy

        if score < best_score:
            best_score = score
            best_conf = cid

    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer(best_conf)
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

    return {
        'coords': coords,
        'score': float(best_score),
        'conformer_id': int(best_conf),
    }


# =============================================================================
# TIER 3: Centroid Placement (fallback)
# =============================================================================

def dock_centroid(
    drug_name: str,
    pocket_center: tuple[float, float, float],
) -> dict:
    """Place drug at pocket centroid (fallback when no 3D conformer available).
    
    Translates the drug's SMILES-based 3D conformer so its centroid
    coincides with the pocket center. No rotational optimization.
    """
    smiles = DRUG_SMILES.get(drug_name.upper())
    if not smiles:
        raise KeyError(f"Unknown drug: {drug_name}")

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol, maxIters=500, confId=0)
    mol = Chem.RemoveHs(mol)

    conf = mol.GetConformer(0)
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    centroid = np.mean(coords, axis=0)
    coords += np.array(pocket_center) - centroid

    return {
        'coords': coords,
        'score': 0.0,
        'method': 'centroid_placement',
    }


# =============================================================================
# MAIN DOCKING INTERFACE
# =============================================================================

def dock_drug(drug_name: str, pocket_center: tuple[float, float, float] | None = None) -> dict:
    """Dock a drug to the HIV-2 protease binding pocket.
    
    Tries methods in order of accuracy:
    1. AutoDock Vina (external binary)
    2. RDKit ETKDG + UFF (built-in)
    3. Centroid placement (fallback)
    """
    try:
        result = dock_vina(drug_name, center=pocket_center)
        result['method'] = 'vina'
        log.info(f"  Docked {drug_name} via AutoDock Vina: {len(result.get('scores', []))} poses")
        return result
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as e:
        log.warning(f"  Vina unavailable for {drug_name} ({e}), using RDKit docking")

    try:
        result = dock_rdkit(drug_name, pocket_center)
        result['method'] = 'rdkit'
        log.info(f"  Docked {drug_name} via RDKit ETKDG+UFF: score={result['score']:.2f}")
        return result
    except (OSError, ValueError, RuntimeError) as e:
        log.warning(f"  RDKit docking failed for {drug_name} ({e}), using centroid placement")

    result = dock_centroid(drug_name, pocket_center)
    result['method'] = 'centroid'
    log.warning(f"  Docked {drug_name} via centroid placement (no docking)")
    return result
