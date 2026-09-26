"""
Rule-based baselines for the HIV-2 Drug Resistance Benchmark.
Implements HIV-2EU-style mutation penalty lookup algorithm.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "opencode_shared"))

from config import HIV2_WT, SEQ_LENGTH


# ── HIV-2EU-style mutation → drug penalty lookup table ─────────────────
# Source: Márquez et al. 2013 (CID), van der Ende et al. 2013,
#         HIV-GRADE/HIV2EU v3.0 update (2025)
# Penalties are integer scores where higher = more resistance
# This is a simplified programmatic approximation

MUTATION_PENALTIES = {
    # Position: {mutation: {drug: penalty}}
    # PI resistance mutations (Protease positions 1-99)
    "L10F":  {"DRV/r": 1, "FPV/r": 1, "IDV/r": 1, "LPV/r": 1, "NFV/r": 1, "SQV/r": 1},
    "L10I":  {"ATV/r": 1, "DRV/r": 1, "FPV/r": 1, "IDV/r": 1, "LPV/r": 1, "NFV/r": 1, "SQV/r": 1},
    "K20R":  {"ATV/r": 1, "IDV/r": 1, "NFV/r": 1},
    "L23I":  {"ATV/r": 1},
    "L24I":  {"ATV/r": 1, "FPV/r": 1, "IDV/r": 1},
    "D30N":  {"NFV/r": 3},
    "V32I":  {"DRV/r": 1, "FPV/r": 1, "IDV/r": 1, "LPV/r": 1, "NFV/r": 1},
    "V47A":  {"LPV/r": 2},
    "I47V":  {"DRV/r": 1, "LPV/r": 2},
    "I50L":  {"ATV/r": 3},
    "I50V":  {"ATV/r": 2, "DRV/r": 1, "FPV/r": 1, "IDV/r": 1, "LPV/r": 1, "SQV/r": 1, "NFV/r": 1},
    "I53V":  {"DRV/r": 1, "LPV/r": 1, "NFV/r": 1},
    "I54M":  {"DRV/r": 1, "LPV/r": 1},
    "I54V":  {"DRV/r": 1, "LPV/r": 2, "NFV/r": 1},
    "T56V":  {"LPV/r": 2},
    "I62V":  {"IDV/r": 1, "LPV/r": 1},
    "A71V":  {"IDV/r": 1, "LPV/r": 1, "NFV/r": 1},
    "G73S":  {"IDV/r": 1, "NFV/r": 1},
    "L76V":  {"DRV/r": 2, "LPV/r": 2},
    "N83D":  {"DRV/r": 1},
    "I82F":  {"LPV/r": 1, "IDV/r": 2, "NFV/r": 2, "ATV/r": 1, "FPV/r": 2},
    "I82L":  {"IDV/r": 1},
    "I82T":  {"IDV/r": 1},
    "I84V":  {"ATV/r": 2, "DRV/r": 1, "FPV/r": 2, "IDV/r": 2, "LPV/r": 2, "SQV/r": 2, "NFV/r": 2},
    "N88D":  {"NFV/r": 2},
    "N88S":  {"NFV/r": 3},
    "L89V":  {"ATV/r": 1, "IDV/r": 1, "NFV/r": 1},
    "L90M":  {"SQV/r": 3, "LPV/r": 1, "ATV/r": 1, "NFV/r": 1},
    "G48V":  {"SQV/r": 2},
    "V62A":  {"IDV/r": 1, "NFV/r": 1},
    "L99F":  {"IDV/r": 1, "LPV/r": 1},
    "V82A":  {"IDV/r": 2, "NFV/r": 2, "LPV/r": 1, "FPV/r": 2},
    "V82F":  {"IDV/r": 2, "NFV/r": 2, "FPV/r": 2, "LPV/r": 1},
    "V82I":  {"IDV/r": 1, "NFV/r": 1},
    "V82L":  {"IDV/r": 1, "NFV/r": 1},
    "V82T":  {"IDV/r": 1, "NFV/r": 1},
}


def parse_mutations(sequence, wt_seq=HIV2_WT, prefix=""):
    """
    Parse a sequence and identify mutations relative to wild-type.
    
    Returns: list of mutation strings (e.g., "L10F", "I50V")
    """
    mutations = []
    seq = str(sequence).strip()
    
    for pos in range(min(len(seq), len(wt_seq))):
        aa = seq[pos].upper()
        wt_aa = wt_seq[pos].upper()
        
        if aa != wt_aa and aa in "ACDEFGHIKLMNPQRSTVWY":
            mut_str = f"{wt_aa}{pos+1}{aa}"
            mutations.append(mut_str)
    
    return mutations


def hiv2eu_predict(sequence, drug, wt_seq=HIV2_WT):
    """
    Predict HIV-2EU-style penalty score for a sequence against a drug.
    
    Returns: integer penalty score (0 = susceptible)
    """
    mutations = parse_mutations(sequence, wt_seq)
    total_penalty = 0
    
    for mut in mutations:
        if mut in MUTATION_PENALTIES:
            drug_penalties = MUTATION_PENALTIES[mut]
            if drug in drug_penalties:
                total_penalty += drug_penalties[drug]
    
    return total_penalty


def hiv2eu_predict_all(sequences, drugs, wt_seq=HIV2_WT):
    """
    Predict penalty scores for all sequences × all drugs.
    
    Returns: np.ndarray of shape (N, len(drugs))
    """
    N = len(sequences)
    scores = np.zeros((N, len(drugs)), dtype=float)
    
    for i, seq in enumerate(sequences):
        for j, drug in enumerate(drugs):
            scores[i, j] = hiv2eu_predict(seq, drug, wt_seq)
    
    return scores


def analyze_rule_based_performance(y_true, y_pred, drug_name=""):
    """
    Analyze rule-based prediction quality.
    """
    from scipy import stats
    
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Only compute correlation if there's variance
    if np.std(y_pred) < 1e-8:
        return {
            "drug": drug_name,
            "pearson_r": 0.0,
            "spearman_rho": 0.0,
            "rmse": np.sqrt(np.mean((y_true - y_pred) ** 2)),
            "mae": np.mean(np.abs(y_true - y_pred)),
            "n_predicted_resistant": int((y_pred > 0).sum()),
            "n_true_resistant": int((y_true > 0).sum()),
            "note": "No variance in predictions (no known mutations detected)"
        }
    
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_rho, spearman_p = stats.spearmanr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {
        "drug": drug_name,
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": float(pearson_p),
        "spearman_rho": round(float(spearman_rho), 4),
        "spearman_p": float(spearman_p),
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "n_predicted_resistant": int((y_pred > 0).sum()),
        "n_true_resistant": int((y_true > 0).sum()),
    }


def run_rule_based_benchmark(sequences, targets_dict, drugs, wt_seq=HIV2_WT):
    """
    Run complete rule-based benchmark.
    
    Args:
        sequences: list of amino acid sequences
        targets_dict: {drug: np.ndarray of true penalty scores}
        drugs: list of drug names
        wt_seq: wild-type reference
    
    Returns: dict of results per drug
    """
    print("=" * 60)
    print("RULE-BASED BENCHMARK (HIV-2EU approximation)")
    print("=" * 60)
    
    # Predict
    pred_matrix = hiv2eu_predict_all(sequences, drugs, wt_seq)
    
    # Analyze per drug
    results = {}
    for j, drug in enumerate(drugs):
        if drug in targets_dict:
            y_true = targets_dict[drug]
            y_pred = pred_matrix[:, j]
            
            result = analyze_rule_based_performance(y_true, y_pred, drug)
            results[drug] = result
            
            print(f"  {drug}: R={result.get('pearson_r', 'N/A')}, "
                  f"predicted_resistant={result['n_predicted_resistant']}/{len(y_true)}, "
                  f"true_resistant={result['n_true_resistant']}/{len(y_true)}")
    
    # Compute mean
    r_vals = [r.get("pearson_r", np.nan) for r in results.values() if "pearson_r" in r]
    if r_vals:
        results["MEAN"] = {
            "pearson_r": round(float(np.mean(r_vals)), 4),
            "n_drugs": len(r_vals),
            "method": "HIV-2EU_rule_based"
        }
        print(f"\n  MEAN Pearson R: {results['MEAN']['pearson_r']:.4f}")
    
    return results


def mutation_frequency_analysis(sequences, wt_seq=HIV2_WT):
    """
    Analyze mutation frequencies across the HIV-2 protease.
    Useful for understanding which positions are most variable.
    """
    from collections import Counter
    
    n_seq = len(sequences)
    position_mutations = [Counter() for _ in range(SEQ_LENGTH)]
    
    for seq in sequences:
        seq_str = str(seq).strip()
        for pos in range(min(len(seq_str), len(wt_seq))):
            aa = seq_str[pos].upper()
            wt_aa = wt_seq[pos].upper()
            if aa != wt_aa and aa in "ACDEFGHIKLMNPQRSTVWY":
                mut = f"{wt_aa}{pos+1}{aa}"
                position_mutations[pos][mut] += 1
    
    # Summary
    freq_data = []
    for pos in range(SEQ_LENGTH):
        if position_mutations[pos]:
            total_mut = sum(position_mutations[pos].values())
            pct_mut = total_mut / n_seq * 100
            top_mut = position_mutations[pos].most_common(1)[0]
            freq_data.append({
                "position": pos + 1,
                "wt": wt_seq[pos],
                "n_mutation_types": len(position_mutations[pos]),
                "pct_sequences_mutated": round(pct_mut, 1),
                "most_common": top_mut[0],
                "most_common_freq": top_mut[1],
                "all_mutations": dict(position_mutations[pos]),
            })
    
    # Sort by frequency
    freq_data.sort(key=lambda x: x["pct_sequences_mutated"], reverse=True)
    
    print(f"\nTop 20 most variable positions:")
    print(f"  {'Pos':>4}  {'WT':>2}  {'%Mut':>6}  {'Top Mutation':>12}  {'Count':>5}")
    for d in freq_data[:20]:
        print(f"  {d['position']:>4}  {d['wt']:>2}  {d['pct_sequences_mutated']:>6.1f}%  "
              f"{d['most_common']:>12}  {d['most_common_freq']:>5}")
    
    return freq_data


if __name__ == "__main__":
    # Quick test with synthetic sequence
    test_seq = "PQITLWQRPLVTIRIGGQLKEALLDTGADDTVLEDINLPGKWKPKMIGGIGGFIKVRQYDQIPIEICGHKVIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
    
    print("Rule-based predictor test:")
    mutations = parse_mutations(test_seq)
    print(f"  Mutations vs HIV-2 WT: {mutations}")
    
    for drug in ["ATV/r", "DRV/r", "LPV/r", "NFV/r"]:
        score = hiv2eu_predict(test_seq, drug)
        print(f"  {drug} penalty: {score}")
