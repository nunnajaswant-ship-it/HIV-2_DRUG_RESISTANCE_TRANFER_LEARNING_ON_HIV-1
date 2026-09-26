"""
Transfer Learning Pipeline v4 — Pocket-Restricted Biochem (client hypothesis)
==============================================================================
INSPIRATION: the client's insight — "if an ESM-2 feature over all 99 positions
was noise and masking to the 19 conserved drug-pocket positions (POCKET19,
5-Angstrom contact residues) improved transfer, then the SAME masking must be
applied to the biochem features."

BIOCHEM LAYOUT (verified in opencode_shared/utils.py):
  extract_per_residue_features(seq)  -> (99, 21)  per-residue Z3+K10+VHSE8
  extract_global_features(seq, True) -> 2079-D flat + 21 mean + 21 std = 2121-D

NEW FEATURE SETS:
  biochem_pocket : rows [p-1 for p in POCKET19_1INDEX] flattened -> 399-D
                   (the 19 x 21 pocket physchem descriptors, NO global stats)
  combined_bp    : biochem_pocket (399) + esm2_pocket (1280) -> 1679-D

The other 80 non-pocket positions are species-variable and add cross-species
noise; restricting to the conserved drug-binding interface should preserve
HIV-1 accuracy while RAISING HIV-2 transfer accuracy — mirroring the proven
esm2_pocket result (+43% zero-shot).
"""
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark_config import RESULTS_DIR, FIGURES_DIR, RANDOM_SEED

import transfer_learning as tl
from transfer_learning import (
    FeatureStore,
    Bundle,
    _load_esm2_mapping,
    pretrain_hiv1,
    zero_shot_eval,
    finetune_oof,
    build_fast_model,
    kfold_oof,
    evaluate_predictions,
    N_SPLITS,
)
from transfer_learning_v2 import (
    COMMON_DRUGS,
    POCKET19_1INDEX,
    MODEL_LABELS,
    set_seed,
    transfer_finetune_oof,
    onet_refinement,
    load_hiv1_data,
    load_hiv2_chembl,
    load_hiv2_clinical,
    save_results,
)
from transfer_learning_v3 import selection_finetune_oof

warnings.filterwarnings("ignore")

BIOCHEM_PER_POS = 21


class V4FeatureStore(FeatureStore):
    """Extends the v1 FeatureStore with pocket-restricted biochem features."""

    def __init__(self):
        super().__init__()
        self._biochem_pocket = {}

    def biochem_pocket(self, seq):
        from opencode_shared.utils import extract_per_residue_features
        key = str(seq).strip()
        if key not in self._biochem_pocket:
            mat = extract_per_residue_features(key)          # (99, 21)
            idx = [p - 1 for p in POCKET19_1INDEX]
            rows = mat[idx]                                   # (19, 21)
            self._biochem_pocket[key] = rows.reshape(-1).astype(np.float32)
        return self._biochem_pocket[key]

    def get(self, sequences, feature_set):
        seqs = [str(s).strip() for s in sequences]
        if feature_set == "biochem_pocket":
            out = np.vstack([self.biochem_pocket(s) for s in seqs])
            return np.asarray(out, dtype=np.float32)
        if feature_set == "combined_bp":
            bio = np.vstack([self.biochem_pocket(s) for s in seqs])
            esm = np.zeros((len(seqs), 1280), dtype=np.float32)
            for i, s in enumerate(seqs):
                vec = self.esm2_pocket(s)                    # inherited method
                if np.abs(vec).sum() > 0:
                    esm[i] = vec
            return np.hstack([bio, esm]).astype(np.float32)
        return super().get(sequences, feature_set)


# ============================================================================
# PART C — PARAMETERISED WEIGHTED BLEND (same logic as v3 P8)
# ============================================================================
def blend_oof(bundles, key_a, key_b, base_model="T4_RF",
              drugs=COMMON_DRUGS, seed=RANDOM_SEED):
    print("\n" + "=" * 78)
    print(f"PART C — WEIGHTED PREDICTION BLEND  ({key_a} vs {key_b})".ljust(78))
    print("=" * 78)
    A = bundles[key_a]
    B = bundles[key_b]
    res = {}  # also allow 'MEAN' key
    pooled_t, pooled_p = [], []
    weights = {}
    for drug in drugs:
        mask_a = A.drug2 == drug
        mask_b = B.drug2 == drug
        if mask_a.sum() < 8:
            continue
        Xa, Xb_, y = A.X2[mask_a], B.X2[mask_b], A.y2[mask_a]
        oof_pred = np.full(len(y), np.nan)
        n_split = min(N_SPLITS, max(2, len(y) // 8))
        kf = KFold(n_splits=n_split, shuffle=True, random_state=seed)
        fold_ws = []
        for tr, te in kf.split(Xa):
            m_a = build_fast_model(base_model)
            m_a.fit(Xa[tr], y[tr])
            m_b = build_fast_model(base_model)
            m_b.fit(Xb_[tr], y[tr])
            p_at = np.ravel(m_a.predict(Xa[tr]))
            p_bt = np.ravel(m_b.predict(Xb_[tr]))
            best_w, best_s = 0.5, -np.inf
            for w in np.linspace(0.0, 1.0, 11):
                p = w * p_at + (1.0 - w) * p_bt
                if np.std(p) > 0:
                    s = float(np.corrcoef(y[tr], p)[0, 1])
                else:
                    s = -np.inf
                if s > best_s:
                    best_s, best_w = s, w
            oof_pred[te] = (best_w * np.ravel(m_a.predict(Xa[te]))
                            + (1.0 - best_w) * np.ravel(m_b.predict(Xb_[te])))
            fold_ws.append(best_w)
        valid = np.isfinite(oof_pred)
        res[drug] = evaluate_predictions(y[valid], oof_pred[valid])
        res[drug]["weight_a_mean"] = float(np.mean(fold_ws))
        weights[drug] = float(np.mean(fold_ws))
        pooled_t.extend(y[valid].tolist())
        pooled_p.extend(oof_pred[valid].tolist())
    res["MEAN"] = evaluate_predictions(pooled_t, pooled_p)
    for d in drugs:
        if d in weights:
            print(f"    {d}: w_a={weights[d]:.2f}  R={res[d].get('pearson_r', float('nan')):.4f}")
    print(f"  MEAN blended R = {res['MEAN']['pearson_r']:.4f}")
    return res


from sklearn.model_selection import KFold


# ============================================================================
# PART A — FEATURE ABLATION on the pocket-restricted sets
# ============================================================================
def run_ablation(bundles, feature_sets):
    print("\n" + "=" * 78)
    print("PART A — FEATURE ABLATION (RF, pocket-restricted + baselines)".ljust(78))
    print("=" * 78)
    out = {}
    print(f"  {'Feature set':<22}{'Dim':>6}{'HIV-1 CV':>10}{'Zero-shot':>12}{'Fine-tuned':>12}")
    print("  " + "-" * 62)
    for fs in feature_sets:
        b = bundles[fs]
        models, cv = pretrain_hiv1(b, "T4_RF")
        zs = zero_shot_eval(b, models)
        ft = finetune_oof(b, "T4_RF")
        mean_cv = np.nanmean([v["pearson_r"] for k, v in cv.items()
                              if k != "MEAN" and np.isfinite(v["pearson_r"])])
        mean_zs = zs.get("MEAN", {}).get("pearson_r", np.nan)
        mean_ft = ft.get("MEAN", {}).get("pearson_r", np.nan)
        out[fs] = {
            "dim": int(b.X1.shape[1]),
            "hiv1_cv_mean_r": float(mean_cv),
            "zero_shot_mean_r": float(mean_zs),
            "finetuned_mean_r": float(mean_ft),
            "zero_shot": {str(k): v for k, v in zs.items()},
            "finetuned": {str(k): v for k, v in ft.items()},
        }
        print(f"  {fs:<22}{b.X1.shape[1]:>6}{mean_cv:>10.4f}{mean_zs:>12.4f}{mean_ft:>12.4f}")
    return out


# ============================================================================
# MAIN (v4)
# ============================================================================
def main():
    t_start = time.time()
    set_seed()
    print("=" * 78)
    print("TRANSFER LEARNING PIPELINE v4 — pocket-restricted biochem")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 78)

    # Patch the shared feature store so Bundle uses the v4 feature sets
    tl._feature_store = V4FeatureStore()

    hiv1_df = load_hiv1_data()
    hiv2_df = load_hiv2_chembl()
    hiv2_clinical_df, clinical_info = load_hiv2_clinical()

    FEATURE_SETS_V4 = ["biochem", "biochem_pocket", "esm2_pocket", "combined_bp"]
    bundles = {}
    for fs in FEATURE_SETS_V4:
        print(f"[Features] extracting + scaling '{fs}' ...")
        t0 = time.time()
        bundles[fs] = Bundle(fs, hiv1_df, hiv2_df)
        print(f"           {bundles[fs].X1.shape} HIV-1 | "
              f"{bundles[fs].X2.shape} HIV-2 ({time.time() - t0:.1f}s)")

    output = {
        "version": "v4",
        "pocket19_1index": POCKET19_1INDEX,
        "data_stats": {
            "hiv1_n": int(len(hiv1_df)),
            "hiv2_chembl_n": int(len(hiv2_df)),
            "common_drugs": COMMON_DRUGS,
        },
        "feature_sets": {
            fs: {"dim": int(bundles[fs].X1.shape[1])} for fs in FEATURE_SETS_V4
        },
    }

    # ---------------------------------------------------------------- PART A
    ablation = run_ablation(bundles, FEATURE_SETS_V4)
    output["partA_ablation"] = ablation

    # ---------------------------------------------------------------- PART B
    print("\n" + "=" * 78)
    print("PART B — TRUE TRANSFER FINE-TUNE (pocket-restricted sets)".ljust(78))
    print("=" * 78)
    for fs in ("biochem_pocket", "combined_bp"):
        tf = transfer_finetune_oof(bundles[fs], "T4_RF")
        print(f"  Transfer fine-tune [{fs}] MEAN R = {tf['MEAN']['pearson_r']:.4f}")
        for d in COMMON_DRUGS:
            if tf[d]:
                print(f"    {d}: R={tf[d]['pearson_r']:.4f}")
        output[f"partB_transfer_finetune_{fs}"] = {
            str(k): {kk: vv for kk, vv in v.items()} for k, v in tf.items()
        }

    # ---------------------------------------------------------------- PART C
    blend1 = blend_oof(bundles, "biochem_pocket", "esm2_pocket",
                       seed=RANDOM_SEED)
    output["partC_blend_biochem_pocket_esm2_pocket"] = {
        str(k): {kk: vv for kk, vv in v.items()} for k, v in blend1.items()
    }
    blend2 = blend_oof(bundles, "biochem", "biochem_pocket", seed=RANDOM_SEED)
    output["partC_blend_biochem_full_pocket"] = {
        str(k): {kk: vv for kk, vv in v.items()} for k, v in blend2.items()
    }

    # ---------------------------------------------------------------- PART D
    sel1, meta1 = selection_finetune_oof(bundles["combined_bp"],
                                         keep_frac=0.25, seed=RANDOM_SEED)
    sel2, meta2 = selection_finetune_oof(bundles["combined_bp"],
                                         keep_frac=0.50, seed=RANDOM_SEED)
    output["partD_selection_combined_bp"] = {
        "keep_0.25": meta1,
        "keep_0.25_results": {str(k): {kk: vv for kk, vv in v.items()}
                              for k, v in sel1.items()},
        "keep_0.50": meta2,
        "keep_0.50_results": {str(k): {kk: vv for kk, vv in v.items()}
                              for k, v in sel2.items()},
    }

    # ---------------------------------------------------------------- PART E
    onion = onet_refinement(bundles["biochem_pocket"], "T4_RF")
    output["partE_onion_biochem_pocket_base"] = onion

    # ---------------------------------------------------------------- PART F
    print("\n" + "=" * 78)
    print("PART F — FINAL HONEST SUMMARY (HIV-2, MEAN Pearson R)".ljust(78))
    print("=" * 78)
    # Load prior baselines
    baselines = {}
    for name, p in (("v2", RESULTS_DIR / "transfer_learning_v2_results.json"),
                    ("v3", RESULTS_DIR / "transfer_learning_v3_results.json")):
        if p.exists():
            try:
                with open(p) as f:
                    d = json.load(f)
                if name == "v2":
                    pa = d.get("part5_approach_comparison", {})
                    p1 = d.get("part1_feature_ablation", {})
                    baselines.update({
                        "zero_shot_combined_pocket": pa.get("Zero-shot (HIV-1 only)", {}).get("MEAN", {}).get("pearson_r"),
                        "transfer_finetune_combined_pocket": pa.get("Transfer fine-tune (HIV-1 meta)", {}).get("MEAN", {}).get("pearson_r"),
                        "supervised_combined_pocket": pa.get("Supervised (HIV-2 only)", {}).get("MEAN", {}).get("pearson_r"),
                        "fine_biochem": p1.get("biochem", {}).get("finetuned_mean_r"),
                        "fine_esm2_pocket": p1.get("esm2_pocket", {}).get("finetuned_mean_r"),
                        "pooled_zero_legacy": d.get("part1_pooled_zero_shot_legacy_biochem_2079", {}).get("pearson_r"),
                    })
                if name == "v3":
                    s = d.get("part12_summary", [])
                    for row in s:
                        baselines[row["method"]] = row["mean_pearson_r"]
            except Exception as e:
                print(f"  [warn] baseline load fail {name}: {e}")

    rows = []
    def add_row(method, r, src):
        if r is not None and np.isfinite(float(r)):
            rows.append({"method": method, "mean_pearson_r": float(r), "source": src})

    # v4 numbers
    add_row("Fine-tuned biochem (RF)", ablation["biochem"]["finetuned_mean_r"], "v4-A")
    add_row("Fine-tuned biochem_pocket (RF)", ablation["biochem_pocket"]["finetuned_mean_r"], "v4-A")
    add_row("Fine-tuned esm2_pocket (RF)", ablation["esm2_pocket"]["finetuned_mean_r"], "v4-A")
    add_row("Fine-tuned combined_bp (RF)", ablation["combined_bp"]["finetuned_mean_r"], "v4-A")
    add_row("Zero-shot biochem_pocket", ablation["biochem_pocket"]["zero_shot_mean_r"], "v4-A")
    add_row("Zero-shot combined_bp", ablation["combined_bp"]["zero_shot_mean_r"], "v4-A")
    add_row("Transfer fine-tune biochem_pocket (P-B)", output.get("partB_transfer_finetune_biochem_pocket", {}).get("MEAN", {}).get("pearson_r"), "v4-B")
    add_row("Transfer fine-tune combined_bp (P-B)", output.get("partB_transfer_finetune_combined_bp", {}).get("MEAN", {}).get("pearson_r"), "v4-B")
    add_row("Weighted blend biochem_pocket+esm2_pocket (P-C)", blend1["MEAN"].get("pearson_r"), "v4-C")
    add_row("Weighted blend biochem_full+pocket (P-C)", blend2["MEAN"].get("pearson_r"), "v4-C")
    add_row("Selection top-25% combined_bp (P-D)", sel1["MEAN"].get("pearson_r"), "v4-D")
    add_row("Selection top-50% combined_bp (P-D)", sel2["MEAN"].get("pearson_r"), "v4-D")
    if onion and onion.get("corrected"):
        add_row("ONION corrected (biochem_pocket base)", onion["corrected"].get("pearson_r"), "v4-E")
    # prior baselines
    add_row("Fine-tuned biochem (RF, v2)", baselines.get("fine_biochem"), "v2")
    add_row("Transfer fine-tune biochem+HIV1meta (v3)", baselines.get("Transfer fine-tune (biochem + HIV-1 meta)"), "v3")
    add_row("Weighted blend bio+esm2_pocket (v3)", baselines.get("Weighted blend bio+esm2_pocket (P8)"), "v3")
    add_row("ONION corrected biochem base (v3)", baselines.get("ONION corrected (biochem base)"), "v3")
    add_row("Pooled zero-shot legacy-comparable (manuscript)", baselines.get("pooled_zero_legacy"), "v2")

    rows_sorted = sorted(rows, key=lambda r: r["mean_pearson_r"], reverse=True)
    print(f"  {'Method':<55}{'R':>8}")
    print("  " + "-" * 63)
    for r in rows_sorted:
        print(f"  {r['method']:<55}{r['mean_pearson_r']:>8.4f}")
    output["partF_summary"] = rows_sorted
    output["baselines"] = {str(k): v for k, v in baselines.items()}

    out_path = RESULTS_DIR / "transfer_learning_v4_results.json"
    save_results(output, out_path)

    elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print("PIPELINE v4 COMPLETE")
    print(f"Duration: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Results:  {out_path}")
    print("=" * 78)
    return output


if __name__ == "__main__":
    main()