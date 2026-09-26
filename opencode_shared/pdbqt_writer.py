"""
PDBQT writer using RDKit — no Meeko dependency needed.
Converts SMILES → 3D conformer → PDBQT for AutoDock Vina.

PDBQT column format (1-indexed, PDB-compatible):
  1-6:   HETATM
  7-11:  serial (5d)
  12:    space
  13-16: atom name (4 chars)
  17:    space (altLoc)
  18-20: residue name (3s)
  21:    space
  22:    chain ID
  23-26: resSeq (4d)
  27:    space (iCode)
  28-30: spaces
  31-38: X (8.3f)
  39-46: Y (8.3f)
  47-54: Z (8.3f)
  55-61: charge (7.4f)
  62:    space
  63-64: AutoDock atom type (2s)
"""
import os, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# AutoDock atom types and charges for organic molecules
AD_ATOM_PROPS = {
    'C':  ('C',   0.0),
    'N':  ('NA', -0.35),
    'O':  ('OA', -0.40),
    'S':  ('SA', -0.10),
    'P':  ('P',   0.10),
    'F':  ('F',  -0.20),
    'CL': ('Cl', -0.15),
    'BR': ('Br', -0.10),
    'I':  ('I',  -0.05),
    'H':  ('HD',  0.0),
}


def smiles_to_pdbqt(smiles: str, output_path: str, name: str = 'LIG',
                     translate_to: tuple = None) -> int:
    """Convert SMILES to PDBQT file.
    
    Args:
        smiles: SMILES string
        output_path: Path to write PDBQT
        name: 3-letter residue name
        translate_to: Optional (x,y,z) center to translate ligand to
    
    Returns:
        Number of heavy atoms written
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        raise RuntimeError("ETKDG failed to generate 3D conformer")
    
    AllChem.UFFOptimizeMolecule(mol, maxIters=500, confId=0)
    conf = mol.GetConformer()
    
    # Compute translation if needed
    if translate_to is not None:
        heavy_positions = []
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol().upper() != 'H':
                pos = conf.GetAtomPosition(i)
                heavy_positions.append([pos.x, pos.y, pos.z])
        centroid = np.mean(heavy_positions, axis=0)
        offset = np.array(translate_to) - centroid
    else:
        offset = np.zeros(3)
    
    lines = [f'REMARK  {name}\n']
    serial = 0
    
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        elem = atom.GetSymbol().upper()
        
        if elem == 'H':
            continue
        
        serial += 1
        pos = conf.GetAtomPosition(i)
        x = pos.x + offset[0]
        y = pos.y + offset[1]
        z = pos.z + offset[2]
        
        ad_type, charge = AD_ATOM_PROPS.get(elem, ('C', 0.0))
        
        # 4-char atom name: element symbol + serial number
        atom_name = f'{elem}{serial}'
        if len(atom_name) > 4:
            atom_name = atom_name[:4]
        elif len(atom_name) < 4:
            atom_name = atom_name.ljust(4)  # Left-justified for HETATM
        
        # Build PDBQT line EXACTLY following column spec
        out_line = (
            f'HETATM'
            f'{serial:5d}'
            f' '
            f'{atom_name}'  # 4 chars
            f' '
            f'{name:3s}'    # residue name
            f' '
            f'A'            # chain
            f'{1:4d}'       # resSeq
            f' '            # iCode
            f'   '          # blanks
            f'{x:8.3f}'
            f'{y:8.3f}'
            f'{z:8.3f}'
            f'{charge:7.4f}'
            f' '
            f'{ad_type:>2s}'
            f'\n'
        )
        lines.append(out_line)
    
    lines.append('END\n')
    
    with open(output_path, 'w') as f:
        f.writelines(lines)
    
    return serial


if __name__ == '__main__':
    from opencode_shared.biochem_scales import DRUG_SMILES
    
    for drug, smi in DRUG_SMILES.items():
        if '/' in drug:
            continue
        out = os.path.join(r'D:\desktop\BIO PROJECT HIV-2', f'{drug.lower()}.pdbqt')
        n = smiles_to_pdbqt(smi, out, drug[:3])
        print(f"  {drug}: {n} heavy atoms")
