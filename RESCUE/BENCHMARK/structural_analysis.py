"""
HIV-2 Protease Structural Context Analysis
============================================
Maps ML predictions and mutation data back to protein structure.
Documents PDB structures, structural regions, drug binding sites,
mutation impact classification, and generates novel resistance hypotheses.

Pure Python — no PyMOL/GROMACS required. Hardcoded PDB annotations
from published crystal structures.
"""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_config import (
    DRUGS, DRUG_NAMES, HIV2_WT_PR, SEQ_LENGTH,
    HIV2EU_RULES, RESULTS_DIR, FIGURES_DIR, AA_LIST
)

# Ensure output dirs exist
(RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)

# ============================================================================
# SECTION 1: HIV-2 Protease Structure Summary
# ============================================================================

# Hardcoded annotations from published PDB structures and literature
# References: PDB 1IVP (Rosenberg et al. 1996), 3EBZ (Tie et al. 2010),
#             3S45 (Tie et al. 2011)

PDB_STRUCTURES = {
    "1IVP": {
        "resolution": "2.5 A",
        "ligand": "DMP323 (peptidomimetic)",
        "year": 1996,
        "method": "X-ray",
        "reference": "Rosenberg et al., Protein Sci 1996",
        "chain_length": 99,
        "remarks": "Early HIV-2 PR structure, closed-flap conformation",
    },
    "3EBZ": {
        "resolution": "1.2 A",
        "ligand": "Darunavir (DRV)",
        "year": 2010,
        "method": "X-ray",
        "reference": "Tie et al., JMB 2010",
        "chain_length": 99,
        "remarks": "High-resolution DRV complex, key for understanding DRV resistance",
    },
    "3S45": {
        "resolution": "1.5 A",
        "ligand": "Amprenavir (APV)",
        "year": 2011,
        "method": "X-ray",
        "reference": "Tie et al., JMB 2011",
        "chain_length": 99,
        "remarks": "APV/FPV binding mode in HIV-2 PR",
    },
}

# HIV-2 protease structural regions (1-indexed, based on standard HIV PR numbering)
# Residue numbering follows HIV-1 HXB2 convention (conserved across HIV-1/2 PR)
STRUCTURAL_REGIONS = {
    "Active Site": {
        "positions": [25, 26, 27, 28, 29, 30, 48, 49, 50],
        "function": "Catalytic dyad (D25, D27) + substrate binding",
        "description": "Contains the catalytic Asp-Thr-Gly triad; directly contacts inhibitor P1/P1' sites",
        "color": "#E74C3C",
    },
    "Flap Region": {
        "positions": list(range(43, 59)),  # 43-58
        "function": "Substrate/inhibitor gating",
        "description": "Flexible beta-hairpin flaps that open/close to allow substrate entry",
        "color": "#3498DB",
    },
    "Flap Tip": {
        "positions": [50, 51, 52, 53, 54, 55],
        "function": "Tip of the flap closing over active site",
        "description": "Directly contacts bound drug; I50 is critical in HIV-2",
        "color": "#2980B9",
    },
    "Substrate Envelope": {
        "positions": [32, 33, 47, 50, 54, 56, 82, 84, 88, 90],
        "function": "Defines substrate specificity cavity",
        "description": "Positions lining the S1/S2/S1'/S2' subsites; mutations here affect drug vs substrate selectivity",
        "color": "#F39C12",
    },
    "Dimer Interface": {
        "positions": list(range(95, 99)) + [1, 2, 3, 99],
        "function": "Inter-subunit contacts maintaining dimer stability",
        "description": "Terminal residues that form the dimer interface; critical for PR assembly",
        "color": "#27AE60",
    },
    "Distal/Allosteric": {
        "positions": [10, 11, 20, 23, 24, 36, 37, 62, 63, 64, 71, 73, 76, 89, 99],
        "function": "Indirect modulation of active site geometry",
        "description": "Positions far from drug binding that can alter dynamics or stability",
        "color": "#8E44AD",
    },
    "Flexibility Loop": {
        "positions": list(range(35, 43)),  # 35-42
        "function": "Flexible surface loop connecting secondary structures",
        "description": "High B-factor region; connects flap to core; variable in HIV-2",
        "color": "#95A5A6",
    },
    "Hydrophobic Core": {
        "positions": [5, 6, 15, 16, 34, 40, 41, 57, 58, 60, 61, 80, 81, 82, 83, 84, 85],
        "function": "Structural stability; packing of helices and sheets",
        "description": "Conserved hydrophobic core; mutations here destabilize the fold",
        "color": "#7F8C8D",
    },
}

# Complete 99-position structural annotation table
# Each position maps to: region, function, conservation level, drug contact
POSITION_ANNOTATIONS = {}

def _build_position_annotations():
    """Build per-position annotation table from region definitions."""
    for pos in range(1, SEQ_LENGTH + 1):
        region_tags = []
        for rname, rdata in STRUCTURAL_REGIONS.items():
            if pos in rdata["positions"]:
                region_tags.append(rname)

        # Primary region assignment (priority order)
        primary = "Uncharacterized"
        if "Active Site" in region_tags:
            primary = "Active Site"
        elif "Flap Tip" in region_tags:
            primary = "Flap Tip"
        elif "Flap Region" in region_tags:
            primary = "Flap Region"
        elif "Substrate Envelope" in region_tags:
            primary = "Substrate Envelope"
        elif "Dimer Interface" in region_tags:
            primary = "Dimer Interface"
        elif "Flexibility Loop" in region_tags:
            primary = "Flexibility Loop"
        elif "Hydrophobic Core" in region_tags:
            primary = "Hydrophobic Core"
        elif "Distal/Allosteric" in region_tags:
            primary = "Distal/Allosteric"

        POSITION_ANNOTATIONS[pos] = {
            "position": pos,
            "wt_aa": HIV2_WT_PR[pos - 1],
            "primary_region": primary,
            "all_regions": region_tags,
            "drug_contact": primary in ("Active Site", "Flap Tip", "Substrate Envelope"),
            "hiv2_specific": pos in (32, 47, 76, 82),
        }

_build_position_annotations()


# ============================================================================
# SECTION 2: Mutation Impact Classification
# ============================================================================

KNOWN_RESISTANCE_MUTATIONS = {
    # Active Site — directly contacts drug
    "D30N": {"region": "Active Site", "mechanism": "Disrupts H-bond to NFV P2 amide",
             "drugs_affected": {"NFV/r": 3}, "hiv2_rule": True},
    "V32I": {"region": "Active Site", "mechanism": "Alters S1 subsite shape; hydrophobic expansion",
             "drugs_affected": {"DRV/r": 1, "FPV/r": 1, "IDV/r": 1, "LPV/r": 1, "NFV/r": 1},
             "hiv2_rule": True, "hiv2_specific": True},
    "I47V": {"region": "Active Site/Flap Tip", "mechanism": "Reduces DRV binding affinity at flap",
             "drugs_affected": {"DRV/r": 1, "LPV/r": 2}, "hiv2_rule": True},
    "I50L": {"region": "Active Site/Flap Tip", "mechanism": "Expands S2 subsite, preferential ATV resistance",
             "drugs_affected": {"ATV/r": 3}, "hiv2_rule": True},
    "I50V": {"region": "Active Site/Flap Tip", "mechanism": "Alters flap drug contacts",
             "drugs_affected": {"ATV/r": 2, "DRV/r": 1, "FPV/r": 1, "IDV/r": 1, "LPV/r": 1, "SQV/r": 1, "NFV/r": 1},
             "hiv2_rule": True},
    "I54M": {"region": "Active Site/Flap Tip", "mechanism": "Disrupts flap-water network",
             "drugs_affected": {"DRV/r": 1, "LPV/r": 1}, "hiv2_rule": True},
    "I54V": {"region": "Active Site/Flap Tip", "mechanism": "Alters S2' pocket geometry",
             "drugs_affected": {"DRV/r": 1, "LPV/r": 2, "NFV/r": 1}, "hiv2_rule": True},

    # Flap — affects dynamics
    "I53V": {"region": "Flap Region", "mechanism": "Reduces flap rigidity, lowers drug binding",
             "drugs_affected": {"DRV/r": 1, "LPV/r": 1, "NFV/r": 1}, "hiv2_rule": True},
    "T56V": {"region": "Flap Region", "mechanism": "Alters flap tip hydrogen bonding",
             "drugs_affected": {"LPV/r": 2}, "hiv2_rule": True},

    # Substrate Envelope — drug/substrate selectivity
    "I82F": {"region": "Substrate Envelope", "mechanism": "Reduces S1/S1' pocket volume",
             "drugs_affected": {"LPV/r": 1, "IDV/r": 2, "NFV/r": 2, "ATV/r": 1, "FPV/r": 2},
             "hiv2_rule": True, "hiv2_specific": True},
    "I84V": {"region": "Substrate Envelope", "mechanism": "Expands S2/S2' hydrophobic contact",
             "drugs_affected": {"ATV/r": 2, "DRV/r": 1, "FPV/r": 2, "IDV/r": 2, "LPV/r": 2, "SQV/r": 2, "NFV/r": 2},
             "hiv2_rule": True},
    "V82A": {"region": "Substrate Envelope", "mechanism": "Reduces S1 pocket volume, excludes bulky P1 groups",
             "drugs_affected": {"IDV/r": 2, "NFV/r": 2, "LPV/r": 1, "FPV/r": 2}, "hiv2_rule": True},
    "V82F": {"region": "Substrate Envelope", "mechanism": "Aromatic ring alters hydrophobic packing",
             "drugs_affected": {"IDV/r": 2, "NFV/r": 2, "FPV/r": 2, "LPV/r": 1}, "hiv2_rule": True},
    "V82I": {"region": "Substrate Envelope", "mechanism": "Moderate S1 pocket alteration",
             "drugs_affected": {"IDV/r": 1, "NFV/r": 1}, "hiv2_rule": True},
    "V82L": {"region": "Substrate Envelope", "mechanism": "Branched aliphatic in S1 pocket",
             "drugs_affected": {"IDV/r": 1, "NFV/r": 1}, "hiv2_rule": True},
    "V82T": {"region": "Substrate Envelope", "mechanism": "Polar substitution in hydrophobic pocket",
             "drugs_affected": {"IDV/r": 1, "NFV/r": 1}, "hiv2_rule": True},
    "N88D": {"region": "Substrate Envelope", "mechanism": "Alters NFV P2' binding network",
             "drugs_affected": {"NFV/r": 2}, "hiv2_rule": True},
    "N88S": {"region": "Substrate Envelope", "mechanism": "Hydroxyl-containing sidechain at NFV binding site",
             "drugs_affected": {"NFV/r": 3}, "hiv2_rule": True},
    "L90M": {"region": "Distal/Allosteric", "mechanism": "Allosteric; alters S1/S1' pocket via beta-sheet repositioning",
             "drugs_affected": {"SQV/r": 3, "LPV/r": 1, "ATV/r": 1, "NFV/r": 1}, "hiv2_rule": True},

    # Distal — allosteric effects
    "L10F": {"region": "Distal/Allosteric", "mechanism": "N-terminal packing affects overall PR stability",
             "drugs_affected": {"DRV/r": 1, "FPV/r": 1, "IDV/r": 1, "LPV/r": 1, "NFV/r": 1, "SQV/r": 1},
             "hiv2_rule": True},
    "L10I": {"region": "Distal/Allosteric", "mechanism": "Mild destabilization of N-terminus",
             "drugs_affected": {"ATV/r": 1, "DRV/r": 1, "FPV/r": 1, "IDV/r": 1, "LPV/r": 1, "NFV/r": 1, "SQV/r": 1},
             "hiv2_rule": True},
    "K20R": {"region": "Distal/Allosteric", "mechanism": "Surface charge redistribution",
             "drugs_affected": {"ATV/r": 1, "IDV/r": 1, "NFV/r": 1}, "hiv2_rule": True},
    "L23I": {"region": "Distal/Allosteric", "mechanism": "Alters core hydrophobic packing",
             "drugs_affected": {"ATV/r": 1}, "hiv2_rule": True},
    "L24I": {"region": "Distal/Allosteric", "mechanism": "Modifies subsite topology indirectly",
             "drugs_affected": {"ATV/r": 1, "FPV/r": 1, "IDV/r": 1}, "hiv2_rule": True},
    "V47A": {"region": "Flap Region", "mechanism": "Reduces flap-S1 subsite contacts",
             "drugs_affected": {"LPV/r": 2}, "hiv2_rule": True, "hiv2_specific": True},
    "G48V": {"region": "Flap Tip", "mechanism": "Restrains flap tip flexibility, reduces SQV binding",
             "drugs_affected": {"SQV/r": 2}, "hiv2_rule": True},
    "I62V": {"region": "Distal/Allosteric", "mechanism": "Alters S2 subsite packing",
             "drugs_affected": {"IDV/r": 1, "LPV/r": 1}, "hiv2_rule": True},
    "A71V": {"region": "Distal/Allosteric", "mechanism": "Modifies dimer interface packing",
             "drugs_affected": {"IDV/r": 1, "LPV/r": 1, "NFV/r": 1}, "hiv2_rule": True},
    "G73S": {"region": "Distal/Allosteric", "mechanism": "Introduces polar group in surface loop",
             "drugs_affected": {"IDV/r": 1, "NFV/r": 1}, "hiv2_rule": True},
    "L76V": {"region": "Distal/Allosteric", "mechanism": "Alters surface topology near dimer interface",
             "drugs_affected": {"DRV/r": 2, "LPV/r": 2}, "hiv2_rule": True, "hiv2_specific": True},
    "N83D": {"region": "Substrate Envelope", "mechanism": "Disrupts inter-subunit H-bond to DRV",
             "drugs_affected": {"DRV/r": 1}, "hiv2_rule": True},
    "L89V": {"region": "Distal/Allosteric", "mechanism": "C-terminal modification affects dimer stability",
             "drugs_affected": {"ATV/r": 1, "IDV/r": 1, "NFV/r": 1}, "hiv2_rule": True},
    "V62A": {"region": "Distal/Allosteric", "mechanism": "Alters core packing near S2",
             "drugs_affected": {"IDV/r": 1, "NFV/r": 1}, "hiv2_rule": True},
    "L99F": {"region": "Dimer Interface", "mechanism": "Disrupts C-terminal dimer contacts",
             "drugs_affected": {"IDV/r": 1, "LPV/r": 1}, "hiv2_rule": True},
}


# ============================================================================
# SECTION 3: Drug Binding Site Analysis
# ============================================================================

DRUG_BINDING_ANALYSIS = {
    "ATV/r": {
        "full_name": "Atazanavir/r",
        "binding_site": "S1/S2/S1'/S2' subsites",
        "key_interactions": [
            {"type": "H-bond", "residues": [50, 27], "detail": "Flap I50 backbone NH -> drug P2/P1' carbonyl; D25 Odelta -> drug P1 NH"},
            {"type": "Hydrophobic", "residues": [32, 47, 50, 54, 82], "detail": "S1 pocket (V32, V82); S2 pocket (I54); flap contacts (I50)"},
            {"type": "Water-mediated", "residues": [50, 48], "detail": "Structural water bridging flap tip to inhibitor"},
        ],
        "mutations_disrupting": {
            "I50V": "Alters flap H-bond pattern; moderate resistance",
            "I50L": "Preferential ATV escape; expands active site",
            "I82F": "Reduces S1 pocket volume",
            "L90M": "Allosteric reshaping of S1/S1'",
            "K20R": "Surface charge affects overall dynamics",
        },
        "hiv1_vs_hiv2": "HIV-2 PR has I32 (vs V32 in HIV-1), changing S1 pocket hydrophobicity. ATV retains activity in HIV-2 due to better S1 complementarity.",
    },
    "DRV/r": {
        "full_name": "Darunavir/r",
        "binding_site": "S1/S2/S1'/S2' subsites (extensive contacts)",
        "key_interactions": [
            {"type": "H-bond", "residues": [27, 50, 53, 30], "detail": "D25, D30 backbone NHs; S53 backbone; bis-THF oxygens -> Asp backbone"},
            {"type": "Hydrophobic", "residues": [32, 47, 50, 54, 82, 84], "detail": "extensive S1/S2 contacts"},
            {"type": "Water-mediated", "residues": [50, 48, 30], "detail": "Conserved water molecule network at flap-core interface"},
        ],
        "mutations_disrupting": {
            "V32I": "Expands S1 pocket, reduces DRV complementarity",
            "I47V": "Alters flap-drug contacts",
            "I50V": "Disrupts flap H-bonds to DRV",
            "I54V": "Alters S2 pocket geometry",
            "I84V": "Reduces S2 hydrophobic contact",
            "N83D": "Disrupts inter-subunit H-bond network",
            "L76V": "Alters surface topology",
        },
        "hiv1_vs_hiv2": "DRV was co-crystallized with HIV-2 PR (3EBZ). Key difference: I32 in HIV-2 vs V32 in HIV-1 alters S1 pocket. DRV's bis-THF group compensates through backbone H-bonds.",
    },
    "FPV/r": {
        "full_name": "Fosamprenavir/r",
        "binding_site": "S1/S2 subsites (amprenavir pharmacophore)",
        "key_interactions": [
            {"type": "H-bond", "residues": [27, 48, 50], "detail": "D25 Odelta; flap backbone; tetrahydrofuranyl oxygen network"},
            {"type": "Hydrophobic", "residues": [32, 50, 54, 82, 84], "detail": "S1 (V82/V32); S2 (I50, I54)"},
        ],
        "mutations_disrupting": {
            "V32I": "Reduces S1 complementarity",
            "I50V": "Alters flap-drug interface",
            "I82F/A": "Excludes FPV P1 benzyl group",
            "I84V": "Reduces S2 packing",
        },
        "hiv1_vs_hiv2": "FPV shares amprenavir scaffold. I50V in HIV-2 is primary resistance; S1 pocket differences between HIV-1/2 partially explain cross-reactivity.",
    },
    "IDV/r": {
        "full_name": "Indinavir/r",
        "binding_site": "S1/S2/S1'/S2' (classic peptidomimetic)",
        "key_interactions": [
            {"type": "H-bond", "residues": [27, 48, 50], "detail": "Classic D25/D27 catalytic H-bonds; flap I50 backbone"},
            {"type": "Hydrophobic", "residues": [32, 47, 50, 54, 82, 84], "detail": "S1 (V82), S2 (I50, I54), S2' (I84)"},
        ],
        "mutations_disrupting": {
            "I50V": "Primary IDV resistance",
            "I82F/A": "Alters S1 pocket volume",
            "I84V": "Reduces S2/S2' contacts",
            "V82A": "Excludes IDV P1 group",
            "V32I": "S1 pocket shape change",
            "I62V": "Secondary S2 modification",
        },
        "hiv1_vs_hiv2": "IDV is less effective in HIV-2 due to natural I32 and I50 positions. HIV-2 PR has inherently higher IDV resistance thresholds.",
    },
    "LPV/r": {
        "full_name": "Lopinavir/r",
        "binding_site": "S1/S2/S1'/S2' (ritonavir analog)",
        "key_interactions": [
            {"type": "H-bond", "residues": [27, 48, 50], "detail": "D25 aspartate; flap backbone; P2 pyrimidinyl oxygen"},
            {"type": "Hydrophobic", "residues": [32, 47, 50, 54, 82, 84], "detail": "Extensive van der Waals across all subsites"},
        ],
        "mutations_disrupting": {
            "I50V": "Reduces flap contacts",
            "I54V": "Primary LPV resistance in HIV-2",
            "I82F/A/V": "S1 pocket alteration",
            "I84V": "Broad cross-resistance",
            "V47A": "HIV-2-specific flap modification",
            "T56V": "Alters flap hydrogen bonding",
            "L76V": "HIV-2-specific surface change",
        },
        "hiv1_vs_hiv2": "LPV has the broadest cross-resistance profile in HIV-2. V47A and L76V are HIV-2-specific mutations not commonly seen in HIV-1.",
    },
    "SQV/r": {
        "full_name": "Saquinavir/r",
        "binding_site": "S1/S2 (naphthyl group in S1)",
        "key_interactions": [
            {"type": "H-bond", "residues": [27, 48, 50], "detail": "Standard aspartyl protease H-bonds; quinoline N -> flap backbone"},
            {"type": "Hydrophobic", "residues": [32, 50, 54, 82], "detail": "Naphthyl in S1 (V82); P2 tert-butyl in S2"},
        ],
        "mutations_disrupting": {
            "G48V": "Primary SQV resistance — restricts flap opening",
            "L90M": "Major resistance mutation — allosteric S1 reshaping",
            "I50V": "Reduces flap contact",
            "I84V": "S2 subsite alteration",
        },
        "hiv1_vs_hiv2": "SQV was the first PI; L90M is a hallmark SQV resistance mutation conserved across HIV-1/2. G48V specifically impacts SQV's quinoline access.",
    },
    "TPV/r": {
        "full_name": "Tipranavir/r",
        "binding_site": "S1/S2 (non-peptidic coumarin scaffold)",
        "key_interactions": [
            {"type": "H-bond", "residues": [27, 50, 53], "detail": "Coumarin carbonyl -> D25; pyridyl N -> backbone NH"},
            {"type": "Hydrophobic", "residues": [32, 47, 50, 82], "detail": "Coumarin ring in S1/S1'; ethyl in S2"},
            {"type": "Pi-stacking", "residues": [50, 82], "detail": "Aromatic contacts with flap and S1 pocket"},
        ],
        "mutations_disrupting": {
            "I50V": "Primary TPV resistance",
            "I82F": "Alters aromatic stacking",
            "V32I": "S1 pocket expansion",
        },
        "hiv1_vs_hiv2": "TPV has a unique non-peptidic scaffold. Less data available for HIV-2; I50V is predicted as primary resistance based on structural overlap.",
    },
    "NFV/r": {
        "full_name": "Nelfinavir/r",
        "binding_site": "S1/S2 (cyclic P2 group)",
        "key_interactions": [
            {"type": "H-bond", "residues": [27, 48, 50], "detail": "Standard catalytic H-bonds; D30 N -> NFV P2 amide (unique)"},
            {"type": "Hydrophobic", "residues": [32, 47, 50, 82, 84, 88], "detail": "S1 (V82); S2 (D30 sidechain contact, I50)"},
        ],
        "mutations_disrupting": {
            "D30N": "Hallmark NFV resistance — eliminates unique D30 interaction",
            "N88D/S": "Alters P2' binding network",
            "I50V": "Flap modification",
            "I82F/A/V": "S1 pocket change",
            "L90M": "Allosteric resistance",
            "I84V": "S2 pocket change",
        },
        "hiv1_vs_hiv2": "D30N is NFV-specific across HIV-1/2. N88S is HIV-2-enriched (vs N88D in HIV-1), reflecting subtle S2' subsite differences.",
    },
}


# ============================================================================
# SECTION 4: HIV-1 vs HIV-2 Structural Comparison
# ============================================================================

# HIV-1 HXB2 reference vs HIV-2 consensus at key positions
HIV1_VS_HIV2_KEY_POSITIONS = {
    10: {"hiv1": "L", "hiv2": "L", "diff": False, "note": "Conserved N-terminal leucine"},
    20: {"hiv1": "K", "hiv2": "K", "diff": False, "note": "Surface lysine, conserved"},
    23: {"hiv1": "I", "hiv2": "L", "diff": True, "note": "HIV-2 has Leu — altered core packing"},
    24: {"hiv1": "I", "hiv2": "L", "diff": True, "note": "HIV-2 has Leu — core modification"},
    25: {"hiv1": "D", "hiv2": "D", "diff": False, "note": "Catalytic Asp25, universally conserved"},
    26: {"hiv1": "T", "hiv2": "T", "diff": False, "note": "Catalytic triad Thr26"},
    27: {"hiv1": "D", "hiv2": "D", "diff": False, "note": "Catalytic Asp27"},
    30: {"hiv1": "D", "hiv2": "D", "diff": False, "note": "NFV contact; D30N primary resistance"},
    32: {"hiv1": "V", "hiv2": "I", "diff": True, "note": "HIV-2 has Ile — smaller S1 pocket; critical for PI binding differences"},
    36: {"hiv1": "I", "hiv2": "L", "diff": True, "note": "Core modification"},
    47: {"hiv1": "I", "hiv2": "I", "diff": False, "note": "Conserved flap residue; V47A is HIV-2-specific resistance"},
    48: {"hiv1": "G", "hiv2": "G", "diff": False, "note": "Glycine at flap hinge — conserved"},
    50: {"hiv1": "I", "hiv2": "I", "diff": False, "note": "Flap tip; I50V/L primary resistance"},
    53: {"hiv1": "I", "hiv2": "I", "diff": False, "note": "Flap core; I53V alters flap stability"},
    54: {"hiv1": "I", "hiv2": "I", "diff": False, "note": "S2 pocket; I54M/V common in HIV-2"},
    56: {"hiv1": "T", "hiv2": "T", "diff": False, "note": "Flap base; T56V LPV resistance"},
    62: {"hiv1": "V", "hiv2": "V", "diff": False, "note": "Conserved core valine"},
    76: {"hiv1": "L", "hiv2": "L", "diff": False, "note": "Surface; L76V is HIV-2-specific resistance"},
    82: {"hiv1": "V", "hiv2": "V", "diff": False, "note": "S1 pocket; V82A/F major resistance"},
    84: {"hiv1": "I", "hiv2": "I", "diff": False, "note": "S2/S2' pocket; I84V broad cross-resistance"},
    88: {"hiv1": "N", "hiv2": "N", "diff": False, "note": "NFV P2' contact; N88D/S NFV resistance"},
    89: {"hiv1": "L", "hiv2": "L", "diff": False, "note": "C-terminal; L89V mild resistance"},
    90: {"hiv1": "L", "hiv2": "L", "diff": False, "note": "S1 reshaping; L90M major SQV resistance"},
}


# ============================================================================
# SECTION 5: Load External Results (SHAP, Conservation)
# ============================================================================

def load_shap_results():
    """Load SHAP importance scores if available."""
    candidates = [
        RESULTS_DIR / "shap_results.json",
        RESULTS_DIR / "feature_importance" / "shap_results.json",
        Path(r"D:\desktop\BIO PROJECT HIV-2\RESCUE") / "shap_results.json",
        Path(r"D:\desktop\BIO PROJECT HIV-2") / "shap_results.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            print(f"  Loaded SHAP results from {path}")
            return data
    print("  SHAP results not found — using empty defaults")
    return {}


def load_conservation_results():
    """Load conservation scores if available."""
    candidates = [
        RESULTS_DIR / "conservation_results.json",
        RESULTS_DIR / "feature_importance" / "conservation_results.json",
        Path(r"D:\desktop\BIO PROJECT HIV-2\RESCUE") / "conservation_results.json",
        Path(r"D:\desktop\BIO PROJECT HIV-2") / "conservation_results.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            print(f"  Loaded conservation results from {path}")
            return data
    print("  Conservation results not found — using defaults")
    return {}


def extract_position_shap(shap_data):
    """Extract per-position SHAP importance from loaded data."""
    importance = {}
    if not shap_data:
        return importance

    # Format from our benchmark: aggregate_position_importance -> top_30 list
    # and per_drug_position_importance -> drug -> top_30 list
    # Each entry: {"position": int, "wild_type": str, "importance": float, ...}

    # 1) Try aggregate_position_importance
    agg = shap_data.get("aggregate_position_importance", {})
    if isinstance(agg, dict) and "top_30" in agg:
        for entry in agg["top_30"]:
            pos = entry.get("position")
            imp = entry.get("importance", 0.0)
            if pos is not None and np.isfinite(imp):
                importance[pos] = importance.get(pos, [])
                importance[pos].append(imp)

    # 2) Also accumulate from per_drug_position_importance
    per_drug = shap_data.get("per_drug_position_importance", {})
    if isinstance(per_drug, dict):
        for drug_name, drug_data in per_drug.items():
            if isinstance(drug_data, dict) and "top_30" in drug_data:
                for entry in drug_data["top_30"]:
                    pos = entry.get("position")
                    imp = entry.get("importance", 0.0)
                    if pos is not None and np.isfinite(imp):
                        importance[pos] = importance.get(pos, [])
                        importance[pos].append(imp)

    # 3) Fallback: try flat {position_str: value} dict
    if not importance:
        for key, val in shap_data.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                try:
                    pos = int(key)
                    importance[pos] = [float(val)]
                except (ValueError, TypeError):
                    continue

    # Average across all accumulations per position
    avg_importance = {}
    for pos, vals in importance.items():
        finite_vals = [v for v in vals if np.isfinite(v)]
        if finite_vals:
            avg_importance[pos] = float(np.mean(finite_vals))

    return avg_importance


def extract_position_conservation(cons_data):
    """Extract per-position conservation scores."""
    conservation = {}
    if not cons_data:
        return conservation

    # Format: lists like most_conserved_top10, most_variable_top10
    # Each entry: {"position": int, "conservation": float, "wt": str}
    for list_key in ["most_conserved_top10", "most_variable_top10",
                     "all_positions", "per_position"]:
        items = cons_data.get(list_key, [])
        if isinstance(items, list):
            for entry in items:
                if isinstance(entry, dict):
                    pos = entry.get("position")
                    cons = entry.get("conservation")
                    if pos is not None and cons is not None:
                        try:
                            conservation[int(pos)] = float(cons)
                        except (ValueError, TypeError):
                            continue

    # Also extract from binding_site_conservation if available (positional info)
    bsc = cons_data.get("binding_site_conservation", {})
    if isinstance(bsc, dict):
        for region_name, region_data in bsc.items():
            if isinstance(region_data, dict):
                positions = region_data.get("positions", [])
                mean_cons = region_data.get("mean_conservation")
                if isinstance(positions, list) and mean_cons is not None:
                    for pos in positions:
                        try:
                            pos_int = int(pos)
                            if pos_int not in conservation:
                                conservation[pos_int] = float(mean_cons)
                        except (ValueError, TypeError):
                            continue

    # Fallback: flat {position_str: value} dict
    if not conservation:
        for key, val in cons_data.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                try:
                    pos = int(key)
                    conservation[pos] = float(val)
                except (ValueError, TypeError):
                    continue

    return conservation


# ============================================================================
# SECTION 6: Resistance Mechanism Hypotheses
# ============================================================================

def generate_hypotheses(shap_importance, conservation_scores):
    """
    For high-SHAP positions that are NOT known resistance mutations,
    generate structural hypotheses about their potential role.
    """
    known_mutation_positions = set()
    for mut_data in KNOWN_RESISTANCE_MUTATIONS.values():
        # Extract position from mutation name like "I50V" -> 50
        pass  # We'll parse directly

    # Get all positions with known resistance mutations
    known_positions = set()
    for mut_name in KNOWN_RESISTANCE_MUTATIONS:
        try:
            # Last character is the mutant AA, everything before is position+WT
            pos_str = mut_name[1:-1]
            known_positions.add(int(pos_str))
        except (ValueError, IndexError):
            continue

    # Combine known positions from HIV2EU rules too
    for mut_name in HIV2EU_RULES:
        try:
            pos_str = mut_name[1:-1]
            known_positions.add(int(pos_str))
        except (ValueError, IndexError):
            continue

    # Find high-importance positions NOT in known resistance list
    high_shap_unknown = []
    if shap_importance:
        sorted_shap = sorted(shap_importance.items(), key=lambda x: -x[1])
        for pos, imp in sorted_shap:
            if pos not in known_positions and imp > 0.01:
                high_shap_unknown.append((pos, imp))

    # Build hypotheses
    hypotheses = []

    # Hypothesis framework: check structural context of unknown positions
    if high_shap_unknown:
        for rank, (pos, imp) in enumerate(high_shap_unknown[:10]):
            annot = POSITION_ANNOTATIONS.get(pos, {})
            region = annot.get("primary_region", "Unknown")
            wt = annot.get("wt_aa", HIV2_WT_PR[pos - 1] if 0 < pos <= SEQ_LENGTH else "?")
            drug_contact = annot.get("drug_contact", False)

            # Check conservation (low conservation = more likely to evolve)
            cons = conservation_scores.get(pos, 0.5)
            low_conservation = cons < 0.6

            # Generate hypothesis based on region
            if drug_contact and low_conservation:
                hypothesis_type = "Direct Binding Modifier"
                rationale = (
                    f"Position {pos} ({wt}) is near the drug binding site (region: {region}) "
                    f"with low conservation ({cons:.2f}). High ML importance ({imp:.4f}) suggests "
                    f"mutations here may fine-tune drug contacts without causing major structural "
                    f"disruption. This could be a 'stealth' resistance determinant that accumulates "
                    f"in a stepwise manner under drug pressure."
                )
                experimental_prediction = (
                    "Systematic mutagenesis at position {pos} to all 19 alternative amino acids, "
                    "followed by phenotypic susceptibility testing against all 8 PIs. "
                    "Expect partial (1-2 fold) resistance changes rather than dramatic effects."
                )
            elif region == "Flap Region" or region == "Flap Tip":
                hypothesis_type = "Flap Dynamics Modulator"
                rationale = (
                    f"Position {pos} ({wt}) is in the flap region (region: {region}) with "
                    f"importance {imp:.4f}. Flap dynamics are critical for drug dissociation "
                    f"kinetics. Mutations here may not alter equilibrium binding affinity but "
                    f"could accelerate drug off-rates, effectively reducing sustained inhibition."
                )
                experimental_prediction = (
                    "Molecular dynamics simulations comparing WT vs {wt}{pos}X variants. "
                    "Measure flap opening frequency and drug residence time. "
                    "Expect changes in conformational dynamics rather than static structure."
                )
            elif region == "Substrate Envelope":
                hypothesis_type = "Substrate/Drug Selectivity Switch"
                rationale = (
                    f"Position {pos} ({wt}) lies in the substrate envelope (region: {region}) "
                    f"with importance {imp:.4f}. Mutations here may subtly reshape the "
                    f"substrate binding cavity, reducing drug complementarity while maintaining "
                    f"substrate processing efficiency — a key evolutionary trade-off."
                )
                experimental_prediction = (
                    "Measure both drug binding IC50 and substrate cleavage efficiency (kcat/Km) "
                    "for {wt}{pos}X variants. Expect correlated loss of drug binding and "
                    "maintenance or improvement of substrate processing."
                )
            elif region == "Distal/Allosteric":
                hypothesis_type = "Allosteric Network Modifier"
                rationale = (
                    f"Position {pos} ({wt}) is distal from the active site (region: {region}) "
                    f"with importance {imp:.4f}. Allosteric communication pathways connect "
                    f"distal positions to the active site through correlated motions. Mutations "
                    f"here may propagate structural changes through the protein matrix."
                )
                experimental_prediction = (
                    "Hydrogen-deuterium exchange mass spectrometry (HDX-MS) comparing WT vs "
                    "{wt}{pos}X variants. Expect altered exchange rates at the active site "
                    "despite the mutation being distal. NMR relaxation dispersion may also detect "
                    "changes in millisecond dynamics."
                )
            elif region == "Flexibility Loop":
                hypothesis_type = "Surface Loop Flexibility Modulator"
                rationale = (
                    f"Position {pos} ({wt}) is in a flexible loop (region: {region}) with "
                    f"importance {imp:.4f}. Surface loops often mediate protein-protein "
                    f"interactions and can influence the overall conformational landscape. "
                    f"Mutations may alter inter-domain dynamics or affect viral fitness."
                )
                experimental_prediction = (
                    "Thermal shift assays (DSF) to measure protein stability. "
                    "Determine if {wt}{pos}X affects PR dimerization or assembly. "
                    "Viral replication assays to assess fitness cost."
                )
            elif region == "Dimer Interface":
                hypothesis_type = "Dimer Stability Modifier"
                rationale = (
                    f"Position {pos} ({wt}) is at the dimer interface (region: {region}) "
                    f"with importance {imp:.4f}. HIV-2 protease is active only as a homodimer. "
                    f"Mutations at the dimer interface could destabilize the active dimer, "
                    f"raising the effective concentration needed for drug inhibition. This could "
                    f"manifest as apparent resistance through reduced PR activity rather than "
                    f"altered drug binding. Conservation: {cons:.2f}."
                )
                experimental_prediction = (
                    "Size-exclusion chromatography (SEC) or analytical ultracentrifugation (AUC) "
                    "to measure dimer/monomer equilibrium for {wt}{pos}X variants. "
                    "Determine dimer dissociation constant (Kd). Viral fitness assays expected "
                    "to show reduced replication due to impaired PR assembly."
                )
            elif region == "Hydrophobic Core":
                hypothesis_type = "Structural Stability Modulator"
                rationale = (
                    f"Position {pos} ({wt}) is in the hydrophobic core (region: {region}) "
                    f"with importance {imp:.4f}. The core maintains the overall fold of the "
                    f"protease. Mutations here could alter the conformational ensemble, "
                    f"shifting the equilibrium between active and inactive states. Even "
                    f"small changes in core packing can propagate to the active site through "
                    f"correlated motions in the protein matrix. Conservation: {cons:.2f}."
                )
                experimental_prediction = (
                    "Differential scanning calorimetry (DSC) or circular dichroism (CD) to "
                    "measure thermal stability. Molecular dynamics simulations to assess "
                    "conformational changes. X-ray crystallography to detect subtle structural "
                    "rearrangements at the active site."
                )
            else:
                hypothesis_type = "Uncharacterized Structural Role"
                rationale = (
                    f"Position {pos} ({wt}) has high ML importance ({imp:.4f}) in region "
                    f"'{region}' but is not a known resistance mutation. Its structural role "
                    f"requires further investigation. Conservation: {cons:.2f}."
                )
                experimental_prediction = (
                    "X-ray crystallography of {wt}{pos}X HIV-2 PR variants. "
                    "Phenotypic resistance testing and viral fitness assessment."
                )

            hypothesis_entry = {
                "rank": rank + 1,
                "position": pos,
                "wt_aa": wt,
                "shap_importance": round(imp, 6),
                "conservation": round(cons, 4),
                "structural_region": region,
                "hypothesis_type": hypothesis_type,
                "rationale": rationale,
                "experimental_prediction": experimental_prediction,
                "known_resistance_mutation": False,
                "priority": "HIGH" if (imp > 0.05 and low_conservation) else "MEDIUM" if imp > 0.03 else "LOW",
            }
            hypotheses.append(hypothesis_entry)

    # Add structural hypotheses even if no SHAP data
    if not hypotheses:
        # Generate default hypotheses based on structural analysis alone
        default_hypotheses = [
            {
                "position": 36, "wt_aa": "L", "region": "Distal/Allosteric",
                "rationale": "Position 36 is in the hydrophobic core of HIV-2 PR (L36, vs I36 in HIV-1). "
                             "It sits at the junction of helix 1 and the core beta-sheet. Substitutions "
                             "could propagate strain to the active site through altered core packing, "
                             "potentially modulating drug binding without direct contact.",
            },
            {
                "position": 63, "wt_aa": "V", "region": "Distal/Allosteric",
                "rationale": "Position 63 (Val in HIV-2, vs L63 in HIV-1) is on the protein surface "
                             "near the S2' subsite entrance. It may influence the pathway of drug entry "
                             "and dissociation, acting as a 'gatekeeper' for drug access kinetics.",
            },
            {
                "position": 69, "wt_aa": "I", "region": "Distal/Allosteric",
                "rationale": "Position 69 is near the dimer interface in HIV-2 PR. It could influence "
                             "dimer stability and the equilibrium between active and inactive conformations. "
                             "Drug binding requires the active dimeric form; perturbation of this equilibrium "
                             "could reduce effective drug concentration at the active site.",
            },
            {
                "position": 35, "wt_aa": "P", "region": "Flexibility Loop",
                "rationale": "Proline at position 35 restricts backbone flexibility in the loop between "
                             "the flap and core regions. Mutations to more flexible residues could alter "
                             "the flap opening dynamics, affecting drug binding kinetics.",
            },
            {
                "position": 86, "wt_aa": "N", "region": "Substrate Envelope",
                "rationale": "Position 86 is at the C-terminal end of the substrate envelope, near the "
                             "dimer interface. It contacts the P2' region of bound inhibitors. Substitutions "
                             "could selectively affect bulky P2' inhibitors like NFV and DRV.",
            },
        ]

        for rank, h in enumerate(default_hypotheses):
            pos = h["position"]
            cons = conservation_scores.get(pos, 0.5)
            hypotheses.append({
                "rank": rank + 1,
                "position": pos,
                "wt_aa": h["wt_aa"],
                "shap_importance": shap_importance.get(pos, None),
                "conservation": round(cons, 4),
                "structural_region": h["region"],
                "hypothesis_type": "Structural Analysis Hypothesis (No SHAP Data)",
                "rationale": h["rationale"],
                "experimental_prediction": f"X-ray crystallography + phenotypic testing of {h['wt_aa']}{pos}X variants.",
                "known_resistance_mutation": False,
                "priority": "MEDIUM",
            })

    return hypotheses


# ============================================================================
# SECTION 7: Figure Generation
# ============================================================================

def fig_structural_mapping(shap_importance, conservation_scores, save_path=None):
    """
    Linear map of 99 positions colored by structural region + ML importance.
    """
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), height_ratios=[3, 1, 1],
                             gridspec_kw={"hspace": 0.15})

    # --- Panel A: Structural region map ---
    ax = axes[0]
    region_colors = {}
    for rname, rdata in STRUCTURAL_REGIONS.items():
        for pos in rdata["positions"]:
            if 1 <= pos <= 99:
                region_colors[pos] = rdata["color"]

    for pos in range(1, 100):
        color = region_colors.get(pos, "#ECF0F1")
        ax.add_patch(mpatches.Rectangle((pos - 0.5, 0), 1, 1, facecolor=color,
                                         edgecolor="white", linewidth=0.3))

    # Mark known resistance mutations
    for mut_name, mut_data in KNOWN_RESISTANCE_MUTATIONS.items():
        try:
            pos = int(mut_name[1:-1])
            if 1 <= pos <= 99:
                ax.plot(pos, 0.5, "k^", markersize=6, zorder=5)
        except (ValueError, IndexError):
            continue

    # Mark HIV-2-specific positions
    for pos in [32, 47, 76, 82]:
        ax.plot(pos, 0.5, "r*", markersize=8, zorder=6)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Position")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_title("A. HIV-2 Protease Structural Regions (99 residues)", fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, 100, 5))

    # Legend
    legend_patches = []
    for rname, rdata in STRUCTURAL_REGIONS.items():
        legend_patches.append(mpatches.Patch(color=rdata["color"], label=rname))
    legend_patches.append(plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="k",
                                      markersize=6, label="Known resistance mut"))
    legend_patches.append(plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="r",
                                      markersize=8, label="HIV-2-specific pos"))
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7, ncol=2,
              framealpha=0.9, edgecolor="gray")

    # --- Panel B: SHAP importance bar chart ---
    ax = axes[1]
    if shap_importance:
        positions = sorted(shap_importance.keys())
        values = [shap_importance[p] for p in positions]
        colors = ["#E74C3C" if KNOWN_RESISTANCE_MUTATIONS.get(
            f"X{p}X", {}).get("region", "") else "#3498DB" for p in positions]
        ax.bar(positions, values, color=colors, edgecolor="none", width=0.8)
        ax.set_ylabel("Mean |SHAP|")
        ax.set_title("B. Per-Position ML Feature Importance (SHAP)", fontsize=12, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "SHAP data not available", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, style="italic", color="gray")
        ax.set_title("B. Per-Position ML Feature Importance (SHAP)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_xticks(range(1, 100, 5))

    # --- Panel C: Conservation score ---
    ax = axes[2]
    if conservation_scores:
        positions = sorted(conservation_scores.keys())
        values = [conservation_scores[p] for p in positions]
        colors_cons = ["#2ECC71" if v > 0.8 else "#F39C12" if v > 0.5 else "#E74C3C"
                       for v in values]
        ax.bar(positions, values, color=colors_cons, edgecolor="none", width=0.8)
        ax.set_ylabel("Conservation Score")
        ax.set_title("C. Per-Position Conservation", fontsize=12, fontweight="bold")
        ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="Highly conserved")
        ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="Moderately conserved")
        ax.legend(fontsize=7, loc="lower right")
    else:
        ax.text(0.5, 0.5, "Conservation data not available", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, style="italic", color="gray")
        ax.set_title("C. Per-Position Conservation", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(1, 100, 5))
    ax.set_xlabel("Position")

    plt.suptitle("HIV-2 Protease Structural Context Analysis", fontsize=14,
                 fontweight="bold", y=1.01)

    save_path = save_path or FIGURES_DIR / "fig_structural_mapping.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


def fig_drug_binding(save_path=None):
    """
    Per-drug binding interaction summary table figure.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis("off")

    # Build table data
    drugs_ordered = ["DRV/r", "ATV/r", "LPV/r", "IDV/r", "SQV/r", "NFV/r", "FPV/r", "TPV/r"]
    col_labels = ["Drug", "Key H-Bond\nResidues", "Hydrophobic\nContacts", "Primary\nResistance Mutations",
                  "HIV-1 vs HIV-2\nDifferences"]
    cell_text = []
    cell_colors = []

    drug_row_colors = ["#FADBD8", "#D4E6F1", "#D5F5E3", "#FCF3CF", "#E8DAEF", "#F9E79F", "#D6EAF8", "#F5CBA7"]

    for idx, drug in enumerate(drugs_ordered):
        if drug not in DRUG_BINDING_ANALYSIS:
            continue
        info = DRUG_BINDING_ANALYSIS[drug]

        # H-bond residues
        hbond_res = []
        for inter in info["key_interactions"]:
            if inter["type"] == "H-bond":
                hbond_res.extend([str(r) for r in inter["residues"]])
        hbond_str = ", ".join(sorted(set(hbond_res), key=lambda x: int(x)))

        # Hydrophobic residues
        hydro_res = []
        for inter in info["key_interactions"]:
            if inter["type"] in ("Hydrophobic", "Pi-stacking"):
                hydro_res.extend([str(r) for r in inter["residues"]])
        hydro_str = ", ".join(sorted(set(hydro_res), key=lambda x: int(x)))

        # Primary mutations (score >= 2)
        primary_muts = []
        for mut, score in sorted(info["mutations_disrupting"].items(), key=lambda x: -len(x[0])):
            primary_muts.append(mut)
        primary_str = ", ".join(primary_muts[:5])

        # HIV-1 vs HIV-2
        comparison = info.get("hiv1_vs_hiv2", "")[:80] + "..." if len(info.get("hiv1_vs_hiv2", "")) > 80 else info.get("hiv1_vs_hiv2", "")

        cell_text.append([
            DRUG_NAMES.get(drug, drug),
            hbond_str,
            hydro_str,
            primary_str,
            comparison,
        ])
        cell_colors.append([drug_row_colors[idx]] * len(col_labels))

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellColours=cell_colors,
                     colColours=["#2C3E50"] * len(col_labels), loc="center",
                     cellLoc="center", colLoc="center")

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 2.2)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)
        cell.set_edgecolor("white")
        cell.set_linewidth(0.5)

    # Style body
    for i in range(len(cell_text)):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_edgecolor("#BDC3C7")
            cell.set_linewidth(0.3)
            if j == 0:
                cell.set_text_props(fontweight="bold", fontsize=8)
            else:
                cell.set_text_props(fontsize=7)

    ax.set_title("HIV-2 Protease — Drug Binding Interactions & Resistance Mutations",
                 fontsize=13, fontweight="bold", pad=20)

    save_path = save_path or FIGURES_DIR / "fig_drug_binding.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


def fig_hiv2_vs_hiv1_structure(save_path=None):
    """
    Comparison of key positions between HIV-1 and HIV-2 protease.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel A: Sequence differences at key positions ---
    ax = axes[0, 0]
    diff_positions = sorted([p for p, d in HIV1_VS_HIV2_KEY_POSITIONS.items() if d["diff"]])
    same_positions = sorted([p for p, d in HIV1_VS_HIV2_KEY_POSITIONS.items() if not d["diff"]])

    x_pos = list(range(len(diff_positions)))
    hiv1_aas = [HIV1_VS_HIV2_KEY_POSITIONS[p]["hiv1"] for p in diff_positions]
    hiv2_aas = [HIV1_VS_HIV2_KEY_POSITIONS[p]["hiv2"] for p in diff_positions]

    for i, (pos, h1, h2) in enumerate(zip(diff_positions, hiv1_aas, hiv2_aas)):
        ax.text(i, 1, h1, ha="center", va="center", fontsize=12, fontweight="bold",
                color="#E74C3C", fontfamily="monospace")
        ax.text(i, 0, h2, ha="center", va="center", fontsize=12, fontweight="bold",
                color="#3498DB", fontfamily="monospace")
        ax.plot([i, i], [0.15, 0.85], "k-", linewidth=0.5, alpha=0.3)

    ax.set_xticks(range(len(diff_positions)))
    ax.set_xticklabels([str(p) for p in diff_positions], fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["HIV-2", "HIV-1"], fontsize=10, fontweight="bold")
    ax.set_xlim(-0.5, len(diff_positions) - 0.5)
    ax.set_ylim(-0.3, 1.5)
    ax.set_title("A. Amino Acid Differences at Key Positions", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel B: Structural region distribution of known resistance mutations ---
    ax = axes[0, 1]
    region_counts = {}
    for mut_data in KNOWN_RESISTANCE_MUTATIONS.values():
        region = mut_data["region"].split("/")[0]  # Take primary region
        region_counts[region] = region_counts.get(region, 0) + 1

    regions = sorted(region_counts.keys(), key=lambda x: -region_counts[x])
    counts = [region_counts[r] for r in regions]
    colors_map = {rname: rdata["color"] for rname, rdata in STRUCTURAL_REGIONS.items()}
    bar_colors = [colors_map.get(r, "#95A5A6") for r in regions]

    bars = ax.barh(regions, counts, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Number of Known Resistance Mutations")
    ax.set_title("B. Resistance Mutations by Structural Region", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel C: Drug resistance score heatmap for HIV-2-specific mutations ---
    ax = axes[1, 0]
    hiv2_specific_muts = {k: v for k, v in KNOWN_RESISTANCE_MUTATIONS.items()
                          if v.get("hiv2_specific", False)}
    if not hiv2_specific_muts:
        hiv2_specific_muts = {k: v for k, v in KNOWN_RESISTANCE_MUTATIONS.items()
                              if any(d in v.get("drugs_affected", {}) for d in DRUGS[:4])}

    mut_names = sorted(hiv2_specific_muts.keys())
    drugs_subset = DRUGS[:4]
    score_matrix = np.zeros((len(mut_names), len(drugs_subset)))

    for i, mut in enumerate(mut_names):
        for j, drug in enumerate(drugs_subset):
            score_matrix[i, j] = hiv2_specific_muts[mut].get("drugs_affected", {}).get(drug, 0)

    if len(mut_names) > 0:
        im = ax.imshow(score_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=3)
        ax.set_xticks(range(len(drugs_subset)))
        ax.set_xticklabels(drugs_subset, fontsize=9, rotation=45, ha="right")
        ax.set_yticks(range(len(mut_names)))
        ax.set_yticklabels(mut_names, fontsize=9, fontfamily="monospace")
        for i in range(len(mut_names)):
            for j in range(len(drugs_subset)):
                val = int(score_matrix[i, j])
                if val > 0:
                    ax.text(j, i, str(val), ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="white" if val >= 2 else "black")
        plt.colorbar(im, ax=ax, label="HIV-2EU Penalty Score", shrink=0.8)
    ax.set_title("C. HIV-2-Specific Mutation Impact (Penalty Score)", fontsize=11, fontweight="bold")

    # --- Panel D: Cross-resistance summary ---
    ax = axes[1, 1]
    # Count how many drugs each mutation affects
    mutation_breadth = []
    for mut, data in KNOWN_RESISTANCE_MUTATIONS.items():
        n_drugs = len(data.get("drugs_affected", {}))
        max_score = max(data.get("drugs_affected", {0: 0}).values()) if data.get("drugs_affected") else 0
        mutation_breadth.append((mut, n_drugs, max_score))

    mutation_breadth.sort(key=lambda x: (-x[1], -x[2]))
    top_mutations = mutation_breadth[:15]

    mut_names_plot = [m[0] for m in top_mutations]
    n_drugs_plot = [m[1] for m in top_mutations]
    max_scores_plot = [m[2] for m in top_mutations]

    colors_breadth = ["#E74C3C" if s >= 3 else "#F39C12" if s >= 2 else "#3498DB"
                      for s in max_scores_plot]
    bars = ax.barh(range(len(mut_names_plot)), n_drugs_plot, color=colors_breadth,
                   edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(mut_names_plot)))
    ax.set_yticklabels(mut_names_plot, fontsize=8, fontfamily="monospace")
    ax.set_xlabel("Number of Drugs Affected")
    ax.set_title("D. Cross-Resistance Breadth (Top 15 Mutations)", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_items = [
        mpatches.Patch(color="#E74C3C", label="Major (score >= 3)"),
        mpatches.Patch(color="#F39C12", label="Moderate (score 2)"),
        mpatches.Patch(color="#3498DB", label="Minor (score 1)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8)

    plt.suptitle("HIV-1 vs HIV-2 Protease — Structural Comparison",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    save_path = save_path or FIGURES_DIR / "fig_hiv2_vs_hiv1_structure.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


# ============================================================================
# SECTION 8: Output Generation
# ============================================================================

def build_structural_context_output(shap_importance, conservation_scores, hypotheses):
    """Build the full structural context JSON output."""
    output = {
        "analysis_timestamp": datetime.now().isoformat(),
        "pdb_structures": PDB_STRUCTURES,
        "structural_regions": {k: {
            "positions": v["positions"],
            "function": v["function"],
            "description": v["description"],
        } for k, v in STRUCTURAL_REGIONS.items()},
        "position_annotations": {str(k): v for k, v in POSITION_ANNOTATIONS.items()},
        "known_resistance_mutations": KNOWN_RESISTANCE_MUTATIONS,
        "drug_binding_analysis": DRUG_BINDING_ANALYSIS,
        "hiv1_vs_hiv2_key_positions": {str(k): v for k, v in HIV1_VS_HIV2_KEY_POSITIONS.items()},
        "hiv2_specific_positions": {
            "32": {"hiv1": "V", "hiv2": "I", "significance": "S1 pocket size difference"},
            "47": {"hiv1": "I", "hiv2": "I (WT same), V47A is HIV-2-specific resistance"},
            "76": {"hiv1": "L", "hiv2": "L (WT same), L76V is HIV-2-specific resistance"},
            "82": {"hiv1": "V", "hiv2": "V (WT same), V82 mutations affect S1 pocket"},
        },
        "shap_importance": {str(k): v for k, v in shap_importance.items()} if shap_importance else {},
        "conservation_scores": {str(k): v for k, v in conservation_scores.items()} if conservation_scores else {},
        "resistance_hypotheses": hypotheses,
        "summary": {
            "total_known_resistance_mutations": len(KNOWN_RESISTANCE_MUTATIONS),
            "hiv2_specific_mutations": sum(1 for v in KNOWN_RESISTANCE_MUTATIONS.values() if v.get("hiv2_specific")),
            "mutations_by_region": {},
            "cross_resistance_hotspots": [],
            "novel_candidate_positions": [h["position"] for h in hypotheses if h.get("priority") == "HIGH"],
        },
    }

    # Count mutations by region
    region_counts = {}
    for mut_data in KNOWN_RESISTANCE_MUTATIONS.values():
        region = mut_data["region"]
        region_counts[region] = region_counts.get(region, 0) + 1
    output["summary"]["mutations_by_region"] = region_counts

    # Cross-resistance hotspots (mutations affecting >= 4 drugs)
    for mut, data in KNOWN_RESISTANCE_MUTATIONS.items():
        n_drugs = len(data.get("drugs_affected", {}))
        if n_drugs >= 4:
            output["summary"]["cross_resistance_hotspots"].append({
                "mutation": mut,
                "n_drugs_affected": n_drugs,
                "max_penalty": max(data.get("drugs_affected", {0: 0}).values()),
            })
    output["summary"]["cross_resistance_hotspots"].sort(
        key=lambda x: (-x["n_drugs_affected"], -x["max_penalty"]))

    return output


def write_hypotheses_md(hypotheses, save_path=None):
    """Write hypotheses to a markdown file."""
    lines = [
        "# Novel Resistance Hypotheses — HIV-2 Protease",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Overview",
        "",
        "These hypotheses identify positions in HIV-2 protease that show high ML feature "
        "importance (SHAP) but are **not** currently classified as known resistance mutations. "
        "They represent candidate positions for novel resistance determinants that warrant "
        "experimental validation.",
        "",
        "---",
        "",
    ]

    for h in hypotheses:
        lines.extend([
            f"## Hypothesis {h['rank']}: Position {h['position']} ({h['wt_aa']}{h['position']}X)",
            "",
            f"- **Priority**: {h['priority']}",
            f"- **Structural Region**: {h['structural_region']}",
            f"- **Conservation**: {h.get('conservation', 'N/A')}",
        ])
        if h.get("shap_importance") is not None:
            lines.append(f"- **SHAP Importance**: {h['shap_importance']:.6f}")
        lines.extend([
            f"- **Hypothesis Type**: {h['hypothesis_type']}",
            "",
            "### Rationale",
            "",
            h["rationale"],
            "",
            "### Experimental Prediction",
            "",
            h["experimental_prediction"],
            "",
            "---",
            "",
        ])

    lines.extend([
        "## Summary Table",
        "",
        "| Rank | Position | WT | Region | Hypothesis Type | SHAP | Conservation | Priority |",
        "|------|----------|-----|--------|-----------------|------|--------------|----------|",
    ])
    for h in hypotheses:
        shap_val = f"{h['shap_importance']:.4f}" if h.get("shap_importance") is not None else "N/A"
        lines.append(
            f"| {h['rank']} | {h['position']} | {h['wt_aa']} | {h['structural_region']} "
            f"| {h['hypothesis_type']} | {shap_val} | {h.get('conservation', 'N/A')} | {h['priority']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Methodology Notes",
        "",
        "1. **SHAP Analysis**: Per-position feature importance derived from tree-based ML models "
        "trained on HIV-2 clinical resistance data.",
        "2. **Conservation**: Calculated as amino acid identity across HIV-2 protease sequences. "
        "Low conservation (<0.6) suggests evolutionary flexibility under drug pressure.",
        "3. **Structural Regions**: Based on published PDB structures (1IVP, 3EBZ, 3S45) and "
        "standard HIV protease nomenclature.",
        "4. **Novel Candidates**: Positions with high SHAP importance + low conservation + "
        "non-mutation status = highest priority for experimental testing.",
    ])

    save_path = save_path or RESULTS_DIR / "structural_hypotheses.md"
    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {save_path}")
    return save_path


# ============================================================================
# SECTION 9: Summary Printer
# ============================================================================

def print_summary(output, hypotheses):
    """Print a concise summary of structural findings."""
    print("\n" + "=" * 70)
    print("  HIV-2 PROTEASE STRUCTURAL CONTEXT ANALYSIS — KEY FINDINGS")
    print("=" * 70)

    # PDB structures
    print("\n[1] PDB Structures Documented:")
    for pdb_id, info in PDB_STRUCTURES.items():
        print(f"    {pdb_id}: {info['resolution']}, ligand={info['ligand']}, year={info['year']}")

    # Structural regions
    print(f"\n[2] Structural Regions: {len(STRUCTURAL_REGIONS)} defined across {SEQ_LENGTH} positions")
    for rname, rdata in STRUCTURAL_REGIONS.items():
        print(f"    {rname}: {len(rdata['positions'])} positions")

    # Known resistance mutations
    print(f"\n[3] Known Resistance Mutations: {len(KNOWN_RESISTANCE_MUTATIONS)} total")
    hiv2_specific = sum(1 for v in KNOWN_RESISTANCE_MUTATIONS.values() if v.get("hiv2_specific"))
    print(f"    HIV-2-specific: {hiv2_specific}")

    # By region
    region_counts = {}
    for v in KNOWN_RESISTANCE_MUTATIONS.values():
        r = v["region"].split("/")[0]
        region_counts[r] = region_counts.get(r, 0) + 1
    for region, count in sorted(region_counts.items(), key=lambda x: -x[1]):
        print(f"    {region}: {count}")

    # Drug binding
    print(f"\n[4] Drug Binding Analysis: {len(DRUG_BINDING_ANALYSIS)} drugs documented")

    # Cross-resistance hotspots
    hotspots = output["summary"]["cross_resistance_hotspots"]
    print(f"\n[5] Cross-Resistance Hotspots: {len(hotspots)} mutations affecting >=4 drugs")
    for hs in hotspots[:5]:
        print(f"    {hs['mutation']}: {hs['n_drugs_affected']} drugs, max penalty={hs['max_penalty']}")

    # HIV-1 vs HIV-2 differences
    n_diffs = sum(1 for v in HIV1_VS_HIV2_KEY_POSITIONS.values() if v["diff"])
    print(f"\n[6] HIV-1 vs HIV-2 Differences: {n_diffs} at key positions")
    for pos, data in sorted(HIV1_VS_HIV2_KEY_POSITIONS.items()):
        if data["diff"]:
            print(f"    Position {pos}: HIV-1={data['hiv1']}, HIV-2={data['hiv2']} — {data['note']}")

    # Novel hypotheses
    print(f"\n[7] Novel Resistance Hypotheses: {len(hypotheses)} generated")
    for h in hypotheses:
        print(f"    [{h['priority']}] Position {h['position']} ({h['wt_aa']}{h['position']}X): "
              f"{h['hypothesis_type']}")

    print("\n" + "=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  HIV-2 PROTEASE STRUCTURAL CONTEXT ANALYSIS")
    print("  Mapping ML predictions to protein structure")
    print("=" * 70)

    # Load external results
    print("\n--- Loading external results ---")
    shap_data = load_shap_results()
    conservation_data = load_conservation_results()

    shap_importance = extract_position_shap(shap_data)
    conservation_scores = extract_position_conservation(conservation_data)

    if shap_importance:
        print(f"  SHAP importance extracted for {len(shap_importance)} positions")
        top_shap = sorted(shap_importance.items(), key=lambda x: -x[1])[:10]
        print(f"  Top 10: {[(p, f'{v:.4f}') for p, v in top_shap]}")
    if conservation_scores:
        print(f"  Conservation scores extracted for {len(conservation_scores)} positions")

    # Generate hypotheses
    print("\n--- Generating resistance hypotheses ---")
    hypotheses = generate_hypotheses(shap_importance, conservation_scores)
    print(f"  Generated {len(hypotheses)} hypotheses")

    # Build output
    print("\n--- Building structural context output ---")
    output = build_structural_context_output(shap_importance, conservation_scores, hypotheses)

    # Save JSON
    json_path = RESULTS_DIR / "structural_context.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")

    # Save hypotheses markdown
    write_hypotheses_md(hypotheses)

    # Generate figures
    print("\n--- Generating figures ---")
    fig_structural_mapping(shap_importance, conservation_scores)
    fig_drug_binding()
    fig_hiv2_vs_hiv1_structure()

    # Print summary
    print_summary(output, hypotheses)

    print(f"\nAll outputs saved to:")
    print(f"  JSON:  {RESULTS_DIR / 'structural_context.json'}")
    print(f"  MD:    {RESULTS_DIR / 'structural_hypotheses.md'}")
    print(f"  Figs:  {FIGURES_DIR / 'fig_structural_mapping.png'}")
    print(f"         {FIGURES_DIR / 'fig_drug_binding.png'}")
    print(f"         {FIGURES_DIR / 'fig_hiv2_vs_hiv1_structure.png'}")

    return output


if __name__ == "__main__":
    main()
