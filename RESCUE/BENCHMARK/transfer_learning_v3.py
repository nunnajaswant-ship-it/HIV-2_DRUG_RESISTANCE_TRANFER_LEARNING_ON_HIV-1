"""
Transfer Learning Pipeline v3 — Weighted fusion, selection & stacking blend
============================================================================
Built on v2 (TRUE transfer + ONION-Net + legacy-comparable pooled zero-shot).

v2 finding: biochem is the strongest single feature (fine-tuned R=0.7015);
adding ESM-2 pocket dims via naive concatenation DILUTES RandomForest
(combined_pocket R=0.6781) because sqrt() feature sampling drowns informative
dims in 1280 extra ones.  v3 implements the client hypothesis — "features that
carry more weight matter more" — with three mechanisms:

PART 8 — Weighted feature blend: per-drug, per-fold grid-optimised weight w
          between biochem-RF and esm2_pocket-RF predictions (w*bio + (1-w)*esm).
          Learnt inside each fold (no leakage); weight on biochem reported.
PART 9 — Transfer fine-tune on the STRONGEST feature set (biochem):
          HIV-1 zero-shot predictions as meta-features + biochem features.
PART 10 — Importance-guided feature selection: pooled HIV-1 RF importances,
          keep top-keep_frac of combined_pocket (3401-D), fine-tune on the
          surviving features.  Reports how many biochem vs esm2_pocket dims
          survive — direct evidence of "which features carry the weight".
PART 11 — ONION-Net structural refinement on the biochem base (was combined_pocket).
PART 12 — Final summary table: all honest HIV-2 numbers in one place.
"""
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark_config import RESULTS_DIR, FIGURES_DIR, RANDOM_SEED

from transfer_learning_v2 import (
    COMMON_DRUGS,
    POCKET19_1INDEX,
    MODEL_LABELS,
    set_seed,
    evaluate_predictions,
    build_bundles,
    build_fast_model,
    kfold_oof,
    pretrain_hiv1,
    transfer_finetune_oof,
    onet_refinement,
    load_hiv1_data,
    load_hiv2_chembl,
    load_hiv2_clinical,
    save_results,
    N_SPLITS,
    HIV2_DRUG_MAP,
)

warnings.filterwarnings("ignore")

BIOCHEM_DIM = 2121   # biochem block size in combined/combined_pocket
ESM2_DIM = 1280


# ============================================================================
# PART 8 — WEIGHTED PREDICTION BLEND (biochem RF vs esm2_pocket RF)
# ============================================================================
def prediction_blend_oof(bundles, base_model="T4_RF",
                         drugs=COMMON_DRUGS, seed=RANDOM_SEED):
    """Per-drug, per-fold grid-optimised blend w between biochem-RF and
    esm2_pocket-RF OOF predictions.  w is chosen on the train fold only
    (11-point grid on rank correlation), then applied to the held-out fold.
    This is the 'features carry more weight' hypothesis implemented as a
    learnt, honest per-fold fusion weight."""
    print("\n" + "=" * 78)
    print("PART 8 — WEIGHTED PREDICTION BLEND (biochem vs esm2_pocket)".ljust(78))
    print("=" * 78)
    bio = bundles["biochem"]
    esm = bundles["esm2_pocket"]
    res = dict.fromkeys(drugs, None)
    pooled_t, pooled_p = [], []
    weights = {}
    for drug in drugs:
        mb = bio.drug2 == drug
        me = esm.drug2 == drug
        if mb.sum() < 8:
            continue
        Xb, Xe = bio.X2[mb], esm.X2[me]
        y = bio.y2[mb]
        oof_pred = np.full(len(y), np.nan)
        n_split = min(N_SPLITS, max(2, len(y) // 8))
        kf = KFold(n_splits=n_split, shuffle=True, random_state=seed)
        fold_ws = []
        for tr, te in kf.split(Xb):
            m_b = build_fast_model(base_model)
            m_b.fit(Xb[tr], y[tr])
            m_e = build_fast_model(base_model)
            m_e.fit(Xe[tr], y[tr])
            p_bt = np.ravel(m_b.predict(Xb[tr]))
            p_et = np.ravel(m_e.predict(Xe[tr]))
            best_w, best_s = 0.5, -np.inf
            for w in np.linspace(0.0, 1.0, 11):
                p = w * p_bt + (1.0 - w) * p_et
                if np.std(p) > 0:
                    s = float(np.corrcoef(y[tr], p)[0, 1])
                else:
                    s = -np.inf
                if s > best_s:
                    best_s, best_w = s, w
            folds_pred_te = (best_w * np.ravel(m_b.predict(Xb[te]))
                             + (1.0 - best_w) * np.ravel(m_e.predict(Xe[te])))
            oof_pred[te] = folds_pred_te
            fold_ws.append(best_w)
        valid = np.isfinite(oof_pred)
        res[drug] = evaluate_predictions(y[valid], oof_pred[valid])
        res[drug]["blend_weight_biochem_mean"] = float(np.mean(fold_ws))
        weights[drug] = float(np.mean(fold_ws))
        pooled_t.extend(y[valid].tolist())
        pooled_p.extend(oof_pred[valid].tolist())
    res["MEAN"] = evaluate_predictions(pooled_t, pooled_p)
    print(f"\n  Per-drug mean biochem weight w (0=only ESM-2 pocket, 1=only biochem):")
    for d in drugs:
        if d in weights:
            print(f"    {d}: w_bio={weights[d]:.2f}  "
                  f"R={res[d].get('pearson_r', float('nan')):.4f}")
    print(f"  MEAN blended R = {res['MEAN']['pearson_r']:.4f}")
    return res


# ============================================================================
# PART 10 — IMPORTANCE-GUIDED FEATURE SELECTION
# ============================================================================
def selection_finetune_oof(bundle, keep_frac=0.25, base_model="T4_RF",
                           drugs=COMMON_DRUGS, seed=RANDOM_SEED):
    """Pooled HIV-1 RF feature importances -> keep top-keep_frac dims of the
    combined_pocket bundle -> HIV-2 fine-tune on survivors.  Reports how many
    biochem vs esm2_pocket dimensions survive — evidence of which block carries
    the cross-species weight."""
    print("\n" + "=" * 78)
    print(f"PART 10 — IMPORTANCE-GUIDED FEATURE SELECTION "
          f"(top {keep_frac * 100:.0f}% of {bundle.X1.shape[1]} dims)".ljust(78))
    print("=" * 78)
    rf = build_fast_model(base_model)
    rf.fit(bundle.X1, bundle.y1)
    imp = rf.feature_importances_
    k = max(100, int(round(len(imp) * keep_frac)))
    top = np.argsort(imp)[::-1][:k]
    n_bio = int(np.sum(top < BIOCHEM_DIM))
    n_esm = int(np.sum(top >= BIOCHEM_DIM))
    print(f"  Selected {k} dims: biochem {n_bio} | esm2_pocket {n_esm}")
    print(f"  (biochem share of selected: {100 * n_bio / k:.1f}%)")

    X1s = bundle.X1[:, top]
    X2s = bundle.X2[:, top]
    res = dict.fromkeys(drugs, None)
    pooled_t, pooled_p = [], []
    for drug in drugs:
        mask = bundle.drug2 == drug
        if mask.sum() < 8:
            continue
        oof_t, oof_p = kfold_oof(
            X2s[mask], bundle.y2[mask],
            lambda: build_fast_model(base_model), seed=seed)
        res[drug] = evaluate_predictions(oof_t, oof_p)
        pooled_t.extend(oof_t.tolist())
        pooled_p.extend(oof_p.tolist())
    res["MEAN"] = evaluate_predictions(pooled_t, pooled_p)
    print(f"  Fine-tuned MEAN R on selected dims = {res['MEAN']['pearson_r']:.4f}")
    for d in drugs:
        if res[d]:
            print(f"    {d}: R={res[d]['pearson_r']:.4f}")
    meta = {
        "keep_frac": keep_frac,
        "k_selected": int(k),
        "biochem_dims_selected": int(n_bio),
        "esm2_pocket_dims_selected": int(n_esm),
        "full_dim": int(bundle.X1.shape[1]),
    }
    return res, meta


# ============================================================================
# MAIN (v3)
# ============================================================================
def main():
    t_start = time.time()
    set_seed()
    print("=" * 78)
    print("TRANSFER LEARNING PIPELINE v3 — weighted fusion & selection")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 78)

    # ------------------------------------------------------------------ load
    hiv1_df = load_hiv1_data()
    hiv2_df = load_hiv2_chembl()
    hiv2_clinical_df, clinical_info = load_hiv2_clinical()
    bundles = build_bundles(hiv1_df, hiv2_df)

    output = {
        "version": "v3",
        "data_stats": {
            "hiv1_n": int(len(hiv1_df)),
            "hiv2_chembl_n": int(len(hiv2_df)),
            "hiv2_common4_n": int(np.sum(
                np.isin(hiv2_df["drug"].values, COMMON_DRUGS))),
            "common_drugs": COMMON_DRUGS,
        },
        "biochem_dim": BIOCHEM_DIM,
        "esm2_pocket_dim": ESM2_DIM,
    }

    # ------------------------------------------------------------- PART 8
    blend = prediction_blend_oof(bundles, seed=RANDOM_SEED)
    output["part8_weighted_blend"] = {
        str(k): {kk: vv for kk, vv in v.items()} for k, v in blend.items()
    }

    # ------------------------------------------------------------- PART 9
    print("\n" + "=" * 78)
    print("PART 9 — TRUE TRANSFER FINE-TUNE ON BIOCHEM (strongest feature)"
          .ljust(78))
    print("=" * 78)
    tf_bio = transfer_finetune_oof(bundles["biochem"], "T4_RF")
    print(f"  Transfer fine-tune (biochem + HIV-1 meta) MEAN R = "
          f"{tf_bio['MEAN']['pearson_r']:.4f}")
    for d in COMMON_DRUGS:
        if tf_bio[d]:
            print(f"    {d}: R={tf_bio[d]['pearson_r']:.4f}")
    output["part9_transfer_finetune_biochem"] = {
        str(k): {kk: vv for kk, vv in v.items()} for k, v in tf_bio.items()
    }

    # ------------------------------------------------------------ PART 10
    sel_025, meta_025 = selection_finetune_oof(bundles["combined_pocket"],
                                               keep_frac=0.25, seed=RANDOM_SEED)
    sel_050, meta_050 = selection_finetune_oof(bundles["combined_pocket"],
                                               keep_frac=0.50, seed=RANDOM_SEED)
    output["part10_selection"] = {
        "keep_0.25": meta_025,
        "keep_0.25_results": {str(k): {kk: vv for kk, vv in v.items()}
                              for k, v in sel_025.items()},
        "keep_0.50": meta_050,
        "keep_0.50_results": {str(k): {kk: vv for kk, vv in v.items()}
                              for k, v in sel_050.items()},
    }

    # ------------------------------------------------------------ PART 11
    onion_bio = onet_refinement(bundles["biochem"], "T4_RF")
    output["part11_onion_biochem_base"] = onion_bio

    # ------------------------------------------------------------ PART 12
    print("\n" + "=" * 78)
    print("PART 12 — FINAL HONEST SUMMARY (HIV-2, MEAN Pearson R)".ljust(78))
    print("=" * 78)
    # Baselines from v2 JSON (load if present)
    v2_path = RESULTS_DIR / "transfer_learning_v2_results.json"
    baselines = {}
    if v2_path.exists():
        try:
            with open(v2_path) as f:
                v2 = json.load(f)
            pa = v2.get("part5_approach_comparison", {})
            p1 = v2.get("part1_feature_ablation", {})
            p6 = v2.get("part6_onion_refinement", {}) or {}
            baselines = {
                "zero_shot_combined_pocket": pa.get("Zero-shot (HIV-1 only)", {}).get("MEAN", {}).get("pearson_r"),
                "transfer_finetune_combined_pocket": pa.get("Transfer fine-tune (HIV-1 meta)", {}).get("MEAN", {}).get("pearson_r"),
                "supervised_combined_pocket": pa.get("Supervised (HIV-2 only)", {}).get("MEAN", {}).get("pearson_r"),
                "biochem_finetuned": p1.get("biochem", {}).get("finetuned_mean_r"),
                "esm2_pocket_finetuned": p1.get("esm2_pocket", {}).get("finetuned_mean_r"),
                "combined_pocket_finetuned": p1.get("combined_pocket", {}).get("finetuned_mean_r"),
                "onion_corrected_combined_pocket_base": (p6.get("corrected", {}) or {}).get("pearson_r"),
                "pooled_zero_shot_legacy": v2.get("part1_pooled_zero_shot_legacy_biochem_2079", {}).get("pearson_r"),
                "pooled_zero_shot_combined_pocket": v2.get("part1_pooled_zero_shot_combined_pocket", {}).get("pearson_r"),
            }
            p7 = v2.get("part7_pocket_importance", {})
            if p7:
                baselines["pocket_top5_positions"] = [r["position"]
                                                      for r in p7.get("ranked_positions", [])[:5]]
        except Exception as e:
            print(f"  [warn] could not load v2 baselines: {e}")

    rows = []
    def add_row(name, r_tuple, src):
        r = r_tuple
        if r is not None and np.isfinite(float(r)):
            rows.append({"method": name, "mean_pearson_r": float(r), "source": src})

    add_row("Zero-shot (combined_pocket)", baselines.get("zero_shot_combined_pocket"), "v2")
    add_row("Transfer fine-tune (biochem + HIV-1 meta)", tf_bio["MEAN"].get("pearson_r"), "v3-P9")
    add_row("Supervised (combined_pocket)", baselines.get("supervised_combined_pocket"), "v2")
    add_row("Fine-tuned biochem (RF)", baselines.get("biochem_finetuned"), "v2")
    add_row("Fine-tuned esm2_pocket (RF)", baselines.get("esm2_pocket_finetuned"), "v2")
    add_row("Fine-tuned combined_pocket (RF)", baselines.get("combined_pocket_finetuned"), "v2")
    add_row("Weighted blend bio+esm2_pocket (P8)", blend["MEAN"].get("pearson_r"), "v3-P8")
    add_row("Selection top-25% combined_pocket (P10)", sel_025["MEAN"].get("pearson_r"), "v3-P10")
    add_row("Selection top-50% combined_pocket (P10)", sel_050["MEAN"].get("pearson_r"), "v3-P10")
    add_row("ONION corrected (combined_pocket base)", baselines.get("onion_corrected_combined_pocket_base"), "v2")
    if onion_bio and onion_bio.get("corrected"):
        add_row("ONION corrected (biochem base)", onion_bio["corrected"].get("pearson_r"), "v3-P11")
    add_row("Pooled zero-shot legacy-comparable (R=0.5163 manuscript)", baselines.get("pooled_zero_shot_legacy"), "v2")
    add_row("Pooled zero-shot combined_pocket", baselines.get("pooled_zero_shot_combined_pocket"), "v2")

    rows_sorted = sorted(rows, key=lambda r: r["mean_pearson_r"], reverse=True)
    print(f"  {'Method':<55}{'R':>8}")
    print("  " + "-" * 63)
    for r in rows_sorted:
        print(f"  {r['method']:<55}{r['mean_pearson_r']:>8.4f}")
    output["part12_summary"] = rows_sorted
    output["baselines_v2"] = {str(k): v for k, v in baselines.items()}

    out_path = RESULTS_DIR / "transfer_learning_v3_results.json"
    save_results(output, out_path)

    elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print("PIPELINE v3 COMPLETE")
    print(f"Duration: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Results:  {out_path}")
    print("=" * 78)
    return output


if __name__ == "__main__":
    main()