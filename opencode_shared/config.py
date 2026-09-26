import os
from pathlib import Path
from Bio.PDB import PDBParser

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_workspace():
    """Return the project root Path."""
    return PROJECT_ROOT


def adv_feat_path(*parts):
    return str(PROJECT_ROOT / "advanced_feature_extraction" / Path(*parts))


def known_info_path(*parts):
    return str(PROJECT_ROOT / "known_info_master" / Path(*parts))


def final_paper_path(*parts):
    return str(PROJECT_ROOT / "final_paper" / Path(*parts))


# Shared PDBParser — import this in scripts instead of creating new instances
PARSER = PDBParser(QUIET=True)

WORKSPACE = str(PROJECT_ROOT)
KNOWN_INFO = os.path.join(WORKSPACE, "known_info_master")
ADV_FEAT = os.path.join(WORKSPACE, "advanced_feature_extraction")
CHEMBL_DIR = os.path.join(WORKSPACE, "chembl")
STANFORD_DIR = os.path.join(WORKSPACE, "stanford db")
DATASET_FINAL = os.path.join(WORKSPACE, "DATASET FINAL")
PREPROC_DIR = os.path.join(DATASET_FINAL, "PRE-PROCESSING OF DATA")
MERGE_DIR = os.path.join(PREPROC_DIR, "Merge_Geno&Pheno")
ESMFOLD_DIR = os.path.join(ADV_FEAT, "esmfold_structures")

PDB_PATH = os.path.join(ADV_FEAT, "3S45.pdb")
POCKET_JSON = os.path.join(KNOWN_INFO, "pocket_residues.json")
POCKET_MASK = os.path.join(KNOWN_INFO, "pocket_mask.npy")
CA_COORDS = os.path.join(KNOWN_INFO, "pocket_ca_coords.npy")
DRUG_CENTROID = os.path.join(KNOWN_INFO, "drug_centroid.npy")
CONSERVE_MASK = os.path.join(KNOWN_INFO, "conservation_mask.npy")
CONSERVE_MASKS_NPZ = os.path.join(KNOWN_INFO, "conservation_masks_all.npz")

HIV1_TRAIN_CSV = os.path.join(KNOWN_INFO, "STANFORD_HIV1_EXPANDED.csv")
HIV2_TEST_CSV = os.path.join(KNOWN_INFO, "HIV2_PROTEASE_ML_READY_DATASET.csv")
HIV2_TEST2_CSV = os.path.join(MERGE_DIR, "FINAL_MASTER_HIV2.csv")
HIV2_CLINICAL_CSV = os.path.join(KNOWN_INFO, "HIV2_CLINICAL_ML_READY_FLAT.csv")
HIV2_CLINICAL_MULTILABEL_CSV = os.path.join(KNOWN_INFO, "HIV2_CLINICAL_ML_READY_MULTILABEL.csv")

# Short aliases (used by many scripts)
HIV1_TRAIN = HIV1_TRAIN_CSV
HIV2_TEST = HIV2_TEST_CSV
HIV2_TEST2 = HIV2_TEST2_CSV

SEQ_LENGTH = 99

HIV1_WT = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
HIV2_WT = "PQFSLWKRPVVTAYIEGQPVEVLLDTGADDSIVAGIELGNNYSPKIVGGIGGFINTKEYKNVEIEVLNKKVRATIMTGDTPINIFGRNILTALGMSLNL"

# =============================================================================
# POCKET DEFINITION — SINGLE SOURCE OF TRUTH
# =============================================================================
# All pocket-related code MUST use these constants. Do NOT hardcode pockets.
#
# Source: 3S45 crystal structure (HIV-2 protease + Amprenavir, 1.51Å).
# Residues within 5.0Å of any drug heavy atom, confirmed by visual inspection.

# 1-indexed (biochemistry convention, positions 1-99)
POCKET_1INDEX = [8, 23, 25, 27, 28, 29, 30, 31, 32, 47, 48, 49, 50, 81, 82, 84]

# 0-indexed (Python convention, positions 0-98)
POCKET_0INDEX = [i - 1 for i in POCKET_1INDEX]

# Extended pocket (8.0Å) for structural feature extraction
POCKET_1INDEX_EXTENDED = [8, 23, 25, 26, 27, 28, 29, 30, 31, 32, 45, 46, 47, 48, 49, 50, 53, 54, 76, 80, 81, 82, 83, 84]
POCKET_0INDEX_EXTENDED = [i - 1 for i in POCKET_1INDEX_EXTENDED]

# Catalytic residues
CATALYTIC_DYAD = [25, 27]

# Flap region (involved in drug binding kinetics)
FLAP_REGION = list(range(45, 56))

# Pocket cutoff distances (Angstroms)
POCKET_CUTOFF_CALPHA = 8.0   # for Cα-based pocket definition
POCKET_CUTOFF_ATOM = 5.0     # for heavy-atom-based pocket definition

# Docking parameters
DOCKING_NUM_MODES = 9        # number of docking poses to generate
DOCKING_EXHAUSTIVENESS = 8   # AutoDock Vina exhaustiveness

CRITICAL_DRUGS = ["Darunavir", "Lopinavir", "Atazanavir", "Saquinavir"]

AROMATIC = set('FYWH')
HYDROPHOBIC = set('ACFGILMVWY')
POLAR_UNCHARGED = set('NQST')
POSITIVE = set('RKH')
NEGATIVE = set('DE')
SMALL = set('AGSTPD')

HIV1_COLS = {
    'seq': ['Sequence', 'Mutant_Sequence', 'sequence'],
    'target': ['Binding_Affinity_kcal_mol'],
    'fold': ['Fold_Resistance', 'Log_Fold_Resistance'],
}
HIV2_COLS = {
    'seq': ['Mutant_Sequence', 'Sequence', 'sequence'],
    'target': ['Binding_Affinity_kcal_mol', 'Standard_Value'],
    'drug': ['Compound_Name', 'Inhibitor', 'drug'],
}
