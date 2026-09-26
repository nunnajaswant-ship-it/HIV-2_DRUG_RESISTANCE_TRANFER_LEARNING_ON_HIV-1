"""
Transfer Learning Pipeline v5 — THE FULL STACK (client's exact question)
================================================================================
"what is the final acc with biochem(train on hiv-1, tested on hiv-2)
 + esm2(train on hiv-1, tested on hiv-2) + onionnet(only hiv-2)?"

Layer 1  biochem    : HIV-1 pretrain -> HIV-2 transfer fine-tune
                      (HIV-1 zero-shot prediction injected as a meta-feature)
Layer 2  esm2_pocket: HIV-1 pretrain -> HIV-2 transfer fine-tune (same)
Layer 3  blend      : per-drug weighted fusion of the two transfer models
                      (weight grid-searched INSIDE folds, no leakage)
Layer 4  ONION-Net  : HIV-2-only 10,478-D docking features -> residual
                      correction ON TOP of the blended transfer base
                      (GroupKFold by structure hash, no sequence leak)

v3's 0.7018 was Layer 1 alone. v2's 0.6910 was ONION on a NON-transfer base.
v5 measures the complete stack honestly, end to end.
"""
import sys
import io
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold

from benchmark_config import (
    DATA_DIR, ESM2_DIR, RESULTS_DIR, FIGURES_DIR, RANDOM_SEED,
)

from transfer_learning import (
    N_SPLITS,
    COMMON_DRUGS,
    HIV2_DRUG_MAP,
    set_seed,
    evaluate_predictions,
    FeatureStore,
    Bundle,
    build_fast_model,
    kfold_oof,
    pretrain_hiv1,
    save_results,
    load_hiv1_data,
    load_hiv2_chembl,
    load_hiv2_clinical,
)

warnings.filterwarnings("ignore")

WORKSPACE = Path(__file__).resolve().parents[2]
SEQ_TO_PDB = WORKSPACE / "advanced_feature_extraction" / "esmfold_structures" / "sequence_to_pdb.csv"
ONION_NPZ = WORKSPACE / "advanced_feature_extraction" / "docked_complexes" / "docked_onionnet_features.npz"


# ============================================================================
# PART 1 — TRANSFER BLEND (biochem + esm2_pocket, per-drug weight, no leak)
# ============================================================================
def transfer_blend_oof(bundle_a, bundle_b, model_name,
                       drugs=COMMON_DRUGS, seed=RANDOM_SEED):
    """Both modalities enter in true-transfer mode: HIV-1 pretrained models'
    zero-shot predictions are meta-features for the HIV-2 fine-tune.  The two
    transfer models are then fused per drug with a weight grid-searched inside
    the training folds only."""
    print("\n" + "=" * 78)
    print("PART 1 - TRANSFER-LEVEL BLEND  biochem + esm2_pocket".ljust(78))
    print("=" * 78)
    n = len(bundle_a.drug2)
    preds_full = np.full((3, n), np.nan)
    y_full = np.full(n, np.nan)

    hiv1_a, _ = pretrain_hiv1(bundle_a, model_name)
    hiv1_b, _ = pretrain_hiv1(bundle_b, model_name)

    stand_a = {}
    stand_b = {}
    blend = {}
    weights = {}
    for drug in drugs:
        if drug not in hiv1_a or drug not in hiv1_b:
            continue
        mask = bundle_a.drug2 == drug
        if mask.sum() < 8:
            continue
        Xa = bundle_a.X2[mask]
        Xb = bundle_b.X2[mask]
        y = bundle_a.y2[mask]
        zsa = np.ravel(hiv1_a[drug].predict(Xa)).reshape(-1, 1)
        zsb = np.ravel(hiv1_b[drug].predict(Xb)).reshape(-1, 1)
        Xa_t = np.hstack([Xa, zsa]).astype(np.float32)
        Xb_t = np.hstack([Xb, zsb]).astype(np.float32)

        # standalone (same transfer mechanics, one modality)
        oa_t, oa_p = kfold_oof(Xa_t, y, lambda: build_fast_model(model_name),
                               seed=seed)
        ob_t, ob_p = kfold_oof(Xb_t, y, lambda: build_fast_model(model_name),
                               seed=seed)
        stand_a[drug] = evaluate_predictions(oa_t, oa_p)
        stand_b[drug] = evaluate_predictions(ob_t, ob_p)

        # blend with in-fold weight selection
        oof = np.full(len(y), np.nan)
        n_split = min(N_SPLITS, max(2, len(y) // 8))
        kf = KFold(n_splits=n_split, shuffle=True, random_state=seed)
        fold_ws = []
        for tr, te in kf.split(Xa_t):
            ma = build_fast_model(model_name)
            ma.fit(Xa_t[tr], y[tr])
            mb = build_fast_model(model_name)
            mb.fit(Xb_t[tr], y[tr])
            pa_tr = np.ravel(ma.predict(Xa_t[tr]))
            pb_tr = np.ravel(mb.predict(Xb_t[tr]))
            best_w, best_s = 0.5, -np.inf
            for w in np.linspace(0.0, 1.0, 11):
                p = w * pa_tr + (1.0 - w) * pb_tr
                if np.std(p) > 0:
                    s = float(np.corrcoef(y[tr], p)[0, 1])
                else:
                    s = -np.inf
                if s > best_s:
                    best_s, best_w = s, w
            pred_te = (best_w * np.ravel(ma.predict(Xa_t[te]))
                       + (1.0 - best_w) * np.ravel(mb.predict(Xb_t[te])))
            oof[te] = pred_te
            fold_ws.append(best_w)
        valid = np.isfinite(oof)
        blend[drug] = evaluate_predictions(y[valid], oof[valid])
        weights[drug] = float(np.mean(fold_ws))
        idx = np.where(mask)[0]
        preds_full[0, idx] = oa_p      # biochem standalone OOF
        preds_full[1, idx] = ob_p      # esm2_pocket standalone OOF
        preds_full[2, idx] = oof       # blend OOF
        y_full[idx] = y

    print("  Standalone transfer (one modality each):")
    for d in drugs:
        if d in stand_a:
            print(f"    {d}: biochem={stand_a[d]['pearson_r']:.4f}  "
                  f"esm2_pocket={stand_b[d]['pearson_r']:.4f}")
    print("  Per-drug blend weights (w=0 pure esm2_pocket, 1 pure biochem):")
    for d in drugs:
        if d in weights:
            print(f"    {d}: w={weights[d]:.2f}  R={blend[d]['pearson_r']:.4f}")
    finite0 = np.isfinite(preds_full[0])
    finite1 = np.isfinite(preds_full[1])
    finite2 = np.isfinite(preds_full[2])
    m_stand_a = evaluate_predictions(y_full[finite0], preds_full[0, finite0])
    m_stand_b = evaluate_predictions(y_full[finite1], preds_full[1, finite1])
    m_blend = evaluate_predictions(y_full[finite2], preds_full[2, finite2])
    print(f"  MEAN R  biochem-only={m_stand_a['pearson_r']:.4f}  "
          f"esm2-only={m_stand_b['pearson_r']:.4f}  "
          f"BLEND={m_blend['pearson_r']:.4f}")
    return {
        "standalone_biochem": {str(k): v for k, v in stand_a.items()},
        "standalone_esm2": {str(k): v for k, v in stand_b.items()},
        "blend": {str(k): v for k, v in blend.items()},
        "standalone_biochem_mean": m_stand_a,
        "standalone_esm2_mean": m_stand_b,
        "blend_mean": m_blend,
        "weights": {str(k): v for k, v in weights.items()},
        "preds_biochem_full": preds_full[0],
        "preds_esm2_full": preds_full[1],
        "preds_blend_full": preds_full[2],
        "y_full": y_full,
    }


# ============================================================================
# PART 2 — ONION-NET RESIDUAL ON TOP OF AN EXTERNAL BASE (GroupKFold, no leak)
# ============================================================================
def onet_apply_to_preds(bundle, base_preds, model_name,
                        drugs=COMMON_DRUGS, seed=RANDOM_SEED):
    """Residual correction on top of externally computed base predictions
    (e.g. the v5 transfer blend).  Residual RF trained/evaluated GroupKFold by
    structure hash so the same sequence never appears train+valid."""
    print("\n" + "=" * 78)
    print("PART 2 - ONION-NET RESIDUAL ON EXTERNAL BASE".ljust(78))
    print("=" * 78)
    if not SEQ_TO_PDB.exists() or not ONION_NPZ.exists():
        print("  [warn] ONION-Net files missing - skipping")
        return None
    stp = pd.read_csv(SEQ_TO_PDB)
    seq_hash = dict(zip(stp["Cleaned_Sequence"].astype(str).str.upper(),
                        stp["Hash"].astype(str)))
    npz = np.load(ONION_NPZ, allow_pickle=True)
    onion_keys = [str(k) for k in np.asarray(npz["keys"])]
    onion_feats = np.asarray(npz["features"])
    onion = {k: i for i, k in enumerate(onion_keys)}
    rev_map = {v: k for k, v in HIV2_DRUG_MAP.items()}

    feat_idx = []
    for seq, drug in zip(bundle.seq2, bundle.drug2):
        h = seq_hash.get(str(seq).strip().upper(), "MISSING")
        d = str(rev_map.get(drug, "MISSING")).strip().upper()
        feat_idx.append(onion.get(f"{d}_{h}", -1))
    feat_idx = np.array(feat_idx)
    valid = (feat_idx >= 0) & np.isin(bundle.drug2, drugs)
    n_valid = int(valid.sum())
    if n_valid < 20:
        print(f"  [warn] only {n_valid} ONION-mapped rows - skipping")
        return None

    print(f"  ONION-mapped rows: {n_valid} | "
          f"drugs: {dict(pd.Series(bundle.drug2[valid]).value_counts())}")
    base_oof = np.asarray(base_preds, dtype=float)[valid]
    ok_base = np.isfinite(base_oof)
    if ok_base.sum() < 20:
        print("  [warn] too few finite base predictions on ONION rows")
        return None
    y_o = bundle.y2[valid][ok_base]
    Xo = onion_feats[feat_idx[valid]][ok_base].astype(np.float32)
    base_oof = base_oof[ok_base]
    residuals = y_o - base_oof
    grp = np.array([seq_hash.get(str(s).strip().upper(), "0")
                    for s in bundle.seq2[valid]])[ok_base]
    uniq_g = len(np.unique(grp))
    print(f"  Unique structures: {uniq_g}")
    print(f"  BASE     R={evaluate_predictions(y_o, base_oof)['pearson_r']:.4f}")

    resid_preds = np.full(len(y_o), np.nan)
    gkf_n = min(N_SPLITS, max(2, uniq_g // 2))
    gkf = GroupKFold(n_splits=gkf_n).split(Xo, groups=grp)
    for tr, te in gkf:
        rf_r = RandomForestRegressor(
            n_estimators=400, max_depth=12, max_features=0.3,
            min_samples_leaf=2, min_samples_split=5, n_jobs=-1,
            random_state=seed)
        rf_r.fit(Xo[tr], residuals[tr])
        resid_preds[te] = rf_r.predict(Xo[te])

    ok = np.isfinite(resid_preds)
    corr = base_oof[ok] + resid_preds[ok]
    m_base = evaluate_predictions(y_o[ok], base_oof[ok])
    m_resid = evaluate_predictions(residuals[ok], resid_preds[ok])
    m_corr = evaluate_predictions(y_o[ok], corr)
    print(f"  RESIDUAL R={m_resid['pearson_r']:.4f} (RMSE={m_resid['rmse']:.4f})")
    print(f"  CORRECTED R={m_corr['pearson_r']:.4f} "
          f"(delta vs base={m_corr['pearson_r'] - m_base['pearson_r']:+.4f})")
    per_drug = {}
    for d in drugs:
        dm = (np.asarray(bundle.drug2[valid][ok_base])[ok] == d)
        if dm.sum() >= 5:
            per_drug[d] = evaluate_predictions(y_o[ok][dm], corr[dm])
            print(f"    {d}: corrected R={per_drug[d]['pearson_r']:.4f}")
    print(f"  -> variance explained by ONION residual: "
          f"{max(0.0, m_resid['pearson_r']) ** 2 * 100:.1f}%")
    return {
        "n_rows": n_valid,
        "n_structures": int(uniq_g),
        "feature_dim": int(Xo.shape[1]),
        "base": m_base,
        "residual": m_resid,
        "corrected": m_corr,
        "corrected_minus_base": float(m_corr["pearson_r"] - m_base["pearson_r"]),
        "per_drug_corrected": {str(k): v for k, v in per_drug.items()},
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    t_start = time.time()
    set_seed()
    print("=" * 78)
    print("TRANSFER LEARNING PIPELINE v5 - THE FULL STACK")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 78)

    hiv1_df = load_hiv1_data()
    hiv2_df = load_hiv2_chembl()
    try:
        load_hiv2_clinical()
    except Exception:
        pass

    bundles = {}
    for fs in ("biochem", "esm2_pocket"):
        print(f"[Features] extracting + scaling '{fs}' ...")
        t0 = time.time()
        bundles[fs] = Bundle(fs, hiv1_df, hiv2_df)
        print(f"           {bundles[fs].X1.shape} HIV-1 | "
              f"{bundles[fs].X2.shape} HIV-2 ({time.time() - t0:.1f}s)")

    # ----------------------------- PART 1 --------------------------------
    res1 = transfer_blend_oof(bundles["biochem"], bundles["esm2_pocket"],
                              "T4_RF")
    r_bio = res1["standalone_biochem_mean"]["pearson_r"]
    r_esm = res1["standalone_esm2_mean"]["pearson_r"]
    r_mix = res1["blend_mean"]["pearson_r"]

    # ----------------------------- PART 2 --------------------------------
    on_blend = onet_apply_to_preds(bundles["biochem"],
                                   res1["preds_blend_full"], "T4_RF")
    # ----------------------------- PART 3 --------------------------------
    on_bio = onet_apply_to_preds(bundles["biochem"],
                                 res1["preds_biochem_full"], "T4_RF")

    # ----------------------------- PART 4 --------------------------------
    print("\n" + "=" * 78)
    print("PART 4 - FINAL SUMMARY (HIV-2, Pearson R)".ljust(78))
    print("=" * 78)
    rows = []
    rows.append((f"Layer1 transfer biochem-only (320 rows)", r_bio, "v5-P1"))
    rows.append((f"Layer2 transfer esm2_pocket-only (320 rows)", r_esm, "v5-P1"))
    rows.append((f"Layer3 transfer BLEND bio+esm2 (320 rows)", r_mix, "v5-P1"))
    if on_blend:
        rows.append(("Layer4 ONION corrected on BLEND base (ONION rows)",
                     on_blend["corrected"]["pearson_r"], "v5-P2"))
        # honest subset base comparison
        rows.append(("     (blend base R on those rows)",
                     on_blend["base"]["pearson_r"], "v5-P2"))
    if on_bio:
        rows.append(("Layer4 ONION corrected on biochem base (ONION rows)",
                     on_bio["corrected"]["pearson_r"], "v5-P3"))
        rows.append(("     (biochem base R on those rows)",
                     on_bio["base"]["pearson_r"], "v5-P3"))

    # v3 locked baselines (defensive)
    p3 = RESULTS_DIR / "transfer_learning_v3_results.json"
    if p3.exists():
        try:
            with open(p3, encoding="utf-8") as f:
                d3 = json.load(f)
            for row in d3.get("part12_summary", []):
                if row["method"] in (
                    "Transfer fine-tune (biochem + HIV-1 meta)",
                    "Fine-tuned biochem (RF)",
                    "ONION corrected (biochem base)",
                    "Weighted blend bio+esm2_pocket (P8)",
                ):
                    rows.append((row["method"] + " [v3]", row["mean_pearson_r"],
                                 "v3"))
        except Exception as e:
            print(f"  [warn] v3 baseline load failed: {e}")

    print(f"  {'Method':<55}{'R':>8}")
    print("  " + "-" * 63)
    for label, r, src in sorted(rows, key=lambda x: -x[1]):
        print(f"  {label:<55}{r:>8.4f}")
    if on_blend:
        full_r = on_blend["corrected"]["pearson_r"]
        print("\n  >>> FULL STACK (Layer1+2+3 blend + Layer4 ONION): "
              f"R = {full_r:.4f} (over blend base "
              f"R={on_blend['base']['pearson_r']:.4f})")
    else:
        print("\n  >>> FULL STACK could not be measured (ONION mapping failed)")

    output = {
        "version": "v5",
        "part1": {
            "standalone_biochem_mean": res1["standalone_biochem_mean"],
            "standalone_esm2_mean": res1["standalone_esm2_mean"],
            "blend_mean": res1["blend_mean"],
            "per_drug_blend": {
                str(k): v for k, v in res1["blend"].items()},
            "weights": res1["weights"],
        },
        "part2_onion_on_blend": on_blend,
        "part3_onion_on_biochem": on_bio,
    }
    out_path = RESULTS_DIR / "transfer_learning_v5_results.json"
    save_results(output, out_path)
    elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print("PIPELINE v5 COMPLETE")
    print(f"Duration: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Results:  {out_path}")
    print("=" * 78)
    return output


if __name__ == "__main__":
    main()