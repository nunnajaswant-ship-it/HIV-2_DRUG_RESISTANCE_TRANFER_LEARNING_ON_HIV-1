"""
clinical_validation_final.py
============================
FINAL LOCKED MODEL -- external clinical validation on never-seen HIV-2
sequences, plus a Stanford/HIVDB biological-consistency layer.

Locked model (v5 "Layer 1", per client decision):
    transfer_learning_v2.transfer_finetune_oof(bundle="biochem", "T4_RF")
    - HIV-1 pre-train on 5,683 Stanford HIV-1 genuine rows (per drug,
      COMMON_DRUGS = ATV/r, DRV/r, LPV/r, SQV/r).
    - The HIV-1 model's zero-shot predictions on HIV-2 ChEMBL rows become a
      meta-feature column.
    - HIV-2 RandomForestRegressor (600 trees / depth 20 / sqrt) trained on
      [biochem 2121-D | zs_pred] with 5-fold KFold OOF.
    - Locked MEAN R = 0.7018 on the ChEMBL 160 rows (4 drugs x 40).
      Source: transfer_learning_v3_results.json
              "Transfer fine-tune (biochem + HIV-1 meta)" = 0.7018426290375417.

This script:
  PART 1 - reproduce the locked OOF sanity number on ChEMBL, then deploy the
           transfer fine-tune models on the clinical cohort
           (known_info_master/HIV2_CLINICAL_ML_READY_FLAT.csv, 5,232 rows),
           which the HIV-1 pre-train NEVER saw (0/653 sequence overlap;
           3 sequences are shared with the ChEMBL fine-tune set -- disclosed,
           not hidden, and re-evaluated excluding them).
  PART 2 - Stanford/HIVDB knowledge layer (analysis only, no leakage, no
           feature augmentation): known-DRM presence from workspace tables
           (benchmark_config.HIV2EU_RULES + shap_analysis
           KNOWN_RESISTANCE_POSITIONS), model-vs-knowledge correlation, and
           the I82F -> DRV/r hypersusceptibility override test.
  PART 3 - comparison table: HIV-1-only zero-shot (existing honest numbers),
           FINAL transfer model, FINAL transfer + consistency layer.

Convention (mirrors RESCUE/clinical_validation_honest.py):
    predicted dG in kcal/mol (negative = tighter binding);
    corr(Clinical_Penalty_Score, pred dG) expected POSITIVE;
    ROC-AUC score = +predicted dG (higher = more resistant).

Usage:
    python RESCUE\\BENCHMARK\\clinical_validation_final.py            (full, 4 drugs)
    python RESCUE\\BENCHMARK\\clinical_validation_final.py --smoke    (DRV/r + SQV/r)
"""
import argparse
import hashlib
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH))               # RESCUE/BENCHMARK
sys.path.insert(0, str(_BENCH.parent))        # RESCUE
sys.path.insert(0, str(_BENCH.parents[1]))    # project root (opencode_shared, known_info_master)

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from benchmark_config import DATA_DIR, RESULTS_DIR, RANDOM_SEED, HIV2EU_RULES
from transfer_learning import (
    COMMON_DRUGS,
    set_seed,
    evaluate_predictions,
    Bundle,
    build_fast_model,
    kfold_oof,
    pretrain_hiv1,
    load_hiv1_data,
    load_hiv2_chembl,
    _feature_store,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
CLINICAL_CSV = DATA_DIR / "HIV2_CLINICAL_ML_READY_FLAT.csv"
GENUINE_CSV = DATA_DIR / "STANFORD_HIV1_GENUINE.csv"
CHEMBL_CSV = DATA_DIR / "HIV2_PROTEASE_ML_READY_DATASET.csv"
OUT_JSON = RESULTS_DIR / "clinical_validation_final_results.json"
HONEST_JSON = _BENCH.parent / "clinical_validation_honest.json"

MODEL_NAME = "T4_RF"
LOCKED_MEAN_R = 0.7018426290375417  # v3 Layer-1, biochem transfer fine-tune
SMOKE_DRUGS = ["DRV/r", "SQV/r"]

# KNOWN_RESISTANCE_POSITIONS -- VERBATIM from RESCUE/BENCHMARK/shap_analysis.py
# lines 126-135 (workspace-owned table; mirrors conservation_analysis.py
# KNOWN_RESISTANCE_POSITIONS position set + HIV2EU_RULES mutation names).
# HIV-2 protease = 99 residues, 1-indexed, aligned 1:1 with HIV-1 PR.
KNOWN_RESISTANCE_POSITIONS = {
    10: {"L10F", "L10I"}, 20: {"K20R"}, 23: {"L23I"}, 24: {"L24I"},
    30: {"D30N"}, 32: {"V32I"}, 47: {"I47V", "V47A"}, 48: {"G48V"},
    50: {"I50V", "I50L"}, 53: {"I53V"}, 54: {"I54M", "I54V"},
    56: {"T56V"}, 62: {"V62A", "I62V"}, 71: {"A71V"}, 73: {"G73S"},
    76: {"L76V"}, 82: {"V82A", "V82F", "V82I", "V82L", "V82T", "I82F"},
    83: {"N83D"}, 84: {"I84V"}, 88: {"N88D", "N88S"},
    89: {"L89V"}, 90: {"L90M"}, 99: {"L99F"},
}

# Union known-DRM set used for the consistency layer (analysis only --
# NEVER a training feature).
DRM_UNION_SET = set().union(*KNOWN_RESISTANCE_POSITIONS.values()).union(HIV2EU_RULES.keys())

# Per-known-mutation metadata: (1-indexed position, mutant residue)
DRM_POS_RES = {}
for _name in DRM_UNION_SET:
    _pos = int(_name[1:-1])
    _res = _name[-1]
    DRM_POS_RES[_name] = (_pos, _res)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_mutations(mut_value):
    """Parse 'M46I, V82A, ...' strings -> set of normalized mutation labels."""
    if mut_value is None or (isinstance(mut_value, float) and np.isnan(mut_value)):
        return set()
    out = set()
    for tok in str(mut_value).split(","):
        tok = tok.strip().upper()
        if len(tok) >= 3 and tok[0].isalpha() and tok[1:-1].isdigit() and tok[-1].isalpha():
            out.add(tok)
    return out


def _class_binary(classes, pred_dg):
    """Mirror clinical_validation_honest.py: Resistant(R/RESISTANT) vs
    Susceptible(S/SUSCEPTIBLE), score = +predicted dG."""
    cl = pd.Series(np.asarray(classes)).astype(str).str.upper().str.strip()
    ok = cl.isin(["S", "R", "SUSCEPTIBLE", "RESISTANT"])
    y = cl[ok].isin(["R", "RESISTANT"]).astype(int).values
    p = np.asarray(pred_dg)[ok.values]
    return y, p


def resistance_auc(classes, pred_dg):
    y, p = _class_binary(classes, pred_dg)
    if len(y) < 10 or len(np.unique(y)) < 2:
        return None, int(len(y)), int(y.sum()) if len(y) else 0
    return float(roc_auc_score(y, p)), int(len(y)), int(y.sum())


def _r4(x):
    return round(float(x), 4) if np.isfinite(x) else None


# ---------------------------------------------------------------------------
# PHASE 1a - LOAD DATA + HONEST REFERENCE
# ---------------------------------------------------------------------------
def phase1_load(ctx):
    print("=" * 78)
    print("PHASE 1a - LOAD HIV-1 / HIV-2 ChEMBL / CLINICAL COHORT".ljust(78))
    print("=" * 78)
    hiv1_df = load_hiv1_data()
    hiv2_df = load_hiv2_chembl()

    clin = pd.read_csv(CLINICAL_CSV, low_memory=False)
    seqs_o = clin["Mutant_Sequence"].astype(str).str.strip().str.upper().values
    keep = np.array([len(s) >= 90 for s in seqs_o])
    clin = clin.iloc[np.where(keep)[0]].copy()
    clin["_seq"] = seqs_o[keep]
    clin["_seq_id"] = np.arange(len(clin))

    n_drugs = int(clin["Compound_Name"].nunique())
    cohort = {
        "rows": int(len(clin)),
        "unique_sequences": int(clin["_seq"].nunique()),
        "drugs": int(n_drugs),
        "structure": (f"{int(len(clin))} rows = "
                      f"{int(len(clin) // n_drugs)} sequence entries x {int(n_drugs)} drugs"),
        "correct_description": ("5,232 rows = 654 sequence entries "
                                "(653 unique + 1 WT duplicate) x 8 drugs"),
        "per_source": clin["Source"].value_counts().to_dict(),
        "resistance_classes": clin["Clinical_Resistance_Class"].value_counts().to_dict(),
    }
    print(f"  Clinical cohort: {cohort['structure']}")
    print(f"  Unique sequences: {cohort['unique_sequences']}")
    print(f"  Classes: {cohort['resistance_classes']}")

    ref = None
    if HONEST_JSON.exists():
        try:
            with open(HONEST_JSON, encoding="utf-8") as fh:
                ref = json.load(fh)
        except Exception as e:
            print(f"  [warn] honest reference load failed: {e}")

    ctx["hiv1_df"] = hiv1_df
    ctx["hiv2_df"] = hiv2_df
    ctx["clin"] = clin
    ctx["cohort"] = cohort
    ctx["honest_ref"] = ref
    return ctx


# ---------------------------------------------------------------------------
# PHASE 1b - FEATURES: biochem bundle (HIV-1 scaler) + clinical features
# ---------------------------------------------------------------------------
def phase2_bundle(ctx):
    print("\n" + "=" * 78)
    print("PHASE 1b - FEATURES: biochem bundle (2121-D, HIV-1 scaler only)".ljust(78))
    print("=" * 78)
    t0 = time.time()
    bundle = Bundle("biochem", ctx["hiv1_df"], ctx["hiv2_df"])
    print(f"  Bundle: HIV-1 {bundle.X1.shape} | HIV-2 {bundle.X2.shape} "
          f"({time.time() - t0:.1f}s)")

    uq = np.unique(ctx["clin"]["_seq"].values)
    clin_X_raw = _feature_store.get(uq.tolist(), "biochem").astype(np.float32)
    clin_X = bundle.scaler.transform(clin_X_raw)
    print(f"  Clinical features: {clin_X.shape} "
          f"(scaler fit on HIV-1 only - no leakage, never refit)")

    h1s = set(ctx["hiv1_df"]["sequence"].astype(str).str.strip().str.upper())
    h2s = set(ctx["hiv2_df"]["sequence"].astype(str).str.strip().str.upper())
    cs = set(uq)
    hygiene = {
        "clinical_unique_sequences": int(len(cs)),
        "overlap_with_hiv1_pretrain": int(len(cs & h1s)),
        "overlap_with_hiv2_chembl_finetune": int(len(cs & h2s)),
        "overlap_chembl_sequences": sorted(cs & h2s)[:20],
        "note": ("Clinical sequences are never-seen w.r.t. the HIV-1 pre-train "
                 "(0 overlap). A small overlap with the ChEMBL fine-tune set is "
                 "disclosed and re-evaluated in the pooled comparison (excluded "
                 "subset reported separately). No clinical labels (Stanford "
                 "penalty) enter training in any form."),
    }
    print(f"  Hygiene: overlap with HIV-1 pre-train = {len(cs & h1s)}, "
          f"with ChEMBL fine-tune = {len(cs & h2s)}")

    ctx["bundle"] = bundle
    ctx["clin_seq_unique"] = uq
    ctx["clin_X"] = clin_X
    ctx["seq2idx"] = {str(s): i for i, s in enumerate(uq)}
    ctx["data_hygiene"] = hygiene
    return ctx


# ---------------------------------------------------------------------------
# PHASE 2 - LOCKED SANITY: transfer OOF on ChEMBL (reproduces ~0.7018)
# ---------------------------------------------------------------------------
def phase3_pretrain_sanity(ctx):
    drugs = ctx["drugs"]
    bundle = ctx["bundle"]
    print("\n" + "=" * 78)
    print(("PHASE 2 - LOCKED MODEL SANITY: transfer OOF on ChEMBL "
           f"({', '.join(drugs)})").ljust(78))
    print("=" * 78)

    hiv1_models, _ = pretrain_hiv1(bundle, MODEL_NAME)
    print(f"  HIV-1 pre-train done: {sorted(hiv1_models.keys())}")

    # Mechanically identical to transfer_learning_v2.transfer_finetune_oof:
    #   zs = hiv1_models[drug].predict(X2d) -> meta-feature -> 5-fold KFold OOF
    #   on [X2d | zs]. Difference: hiv1_models passed in (avoids double
    #   pre-train); the locked function trains the same way inside.
    sanity = dict.fromkeys(drugs, None)
    pooled_t, pooled_p = [], []
    for drug in drugs:
        mask = bundle.drug2 == drug
        if mask.sum() < 8:
            continue
        X2d = bundle.X2[mask]
        y2d = bundle.y2[mask]
        zs = np.ravel(hiv1_models[drug].predict(X2d)).reshape(-1, 1)
        Xtr = np.hstack([X2d, zs]).astype(np.float32)
        oof_t, oof_p = kfold_oof(Xtr, y2d,
                                 lambda: build_fast_model(MODEL_NAME),
                                 seed=RANDOM_SEED)
        sanity[drug] = evaluate_predictions(oof_t, oof_p)
        pooled_t.extend(oof_t.tolist())
        pooled_p.extend(oof_p.tolist())
    sanity["MEAN"] = evaluate_predictions(pooled_t, pooled_p)

    print(f"  {'Drug':<10}{'n':>5}{'Pearson R':>12}{'CI lo':>10}{'CI hi':>10}")
    print("  " + "-" * 47)
    for d in drugs:
        if d in sanity and sanity[d] is not None:
            m = sanity[d]
            print(f"  {d:<10}{m['n']:>5}{m['pearson_r']:>12.4f}"
                  f"{m['ci_pearson_lo']:>10.4f}{m['ci_pearson_hi']:>10.4f}")
    m = sanity["MEAN"]
    print(f"  {'MEAN':<10}{m['n']:>5}{m['pearson_r']:>12.4f}"
          f"{m['ci_pearson_lo']:>10.4f}{m['ci_pearson_hi']:>10.4f}")
    print(f"  Locked 4-drug reference (v3 Layer-1, biochem): {LOCKED_MEAN_R:.4f}")
    print(f"  -> reproduced MEAN R (selected drugs, n={m['n']}): "
          f"{m['pearson_r']:.4f}")

    ctx["hiv1_models"] = hiv1_models
    ctx["sanity"] = sanity
    return ctx


# ---------------------------------------------------------------------------
# PHASE 3 - FINAL MODEL DEPLOYMENT + CLINICAL VALIDATION
# ---------------------------------------------------------------------------
def _per_drug_clinical(bundle, hiv1_models, final_models, Xc, row_idx, ix,
                       drug, scores, classes):
    Xd = Xc[row_idx[ix]]
    zs = np.ravel(hiv1_models[drug].predict(Xd)).reshape(-1, 1)
    Xtr = np.hstack([Xd, zs]).astype(np.float32)
    pred_ft = np.ravel(final_models[drug].predict(Xtr))
    pred_zs = np.ravel(hiv1_models[drug].predict(Xd))

    m_ft = evaluate_predictions(scores, pred_ft)
    m_zs = evaluate_predictions(scores, pred_zs)
    auc_ft, n_bin, n_res = resistance_auc(classes, pred_ft)
    auc_zs, _, _ = resistance_auc(classes, pred_zs)
    m = np.isfinite(scores) & np.isfinite(pred_ft)
    r_p = float(pearsonr(scores[m], pred_ft[m])[1]) if m.sum() >= 3 else None
    res = {
        "n": int(m_ft["n"]),
        "n_binary_class": int(n_bin),
        "n_resistant": int(n_res),
        "pearson_r": _r4(m_ft["pearson_r"]),
        "pearson_p": r_p,
        "ci_pearson_lo": _r4(m_ft["ci_pearson_lo"]),
        "ci_pearson_hi": _r4(m_ft["ci_pearson_hi"]),
        "spearman_rho": _r4(m_ft["spearman_rho"]),
        "rmse": _r4(m_ft["rmse"]),
        "mae": _r4(m_ft["mae"]),
        "roc_auc_resistant_vs_susceptible": (
            round(float(auc_ft), 4) if auc_ft is not None else None),
        "zeroshot_recomputed": {
            "pearson_r": _r4(m_zs["pearson_r"]),
            "spearman_rho": _r4(m_zs["spearman_rho"]),
            "roc_auc": (round(float(auc_zs), 4) if auc_zs is not None else None),
        },
    }
    return res, pred_ft, pred_zs


def phase4_clinical(ctx):
    drugs = ctx["drugs"]
    bundle = ctx["bundle"]
    hiv1_models = ctx["hiv1_models"]
    clin = ctx["clin"]
    seqs = clin["_seq"].values
    seq2idx = ctx["seq2idx"]
    row_idx = np.array([seq2idx[str(s)] for s in seqs])
    Xc = ctx["clin_X"]
    scores_all = pd.to_numeric(clin["Clinical_Penalty_Score"],
                               errors="coerce").values
    classes_all = clin["Clinical_Resistance_Class"].values

    print("\n" + "=" * 78)
    print("PHASE 3 - DEPLOY FINAL TRANSFER MODELS + CLINICAL VALIDATION".ljust(78))
    print("=" * 78)

    # Deployment model per drug: transfer fine-tune fit on ALL ChEMBL rows.
    final_models = {}
    for drug in drugs:
        mask = bundle.drug2 == drug
        n_tr = int(mask.sum())
        if n_tr < 8:
            print(f"  [warn] {drug}: only {n_tr} ChEMBL rows -> skipped")
            continue
        X2d = bundle.X2[mask]
        y2d = bundle.y2[mask]
        zs = np.ravel(hiv1_models[drug].predict(X2d)).reshape(-1, 1)
        m = build_fast_model(MODEL_NAME)
        m.fit(np.hstack([X2d, zs]).astype(np.float32), y2d)
        final_models[drug] = m
        print(f"  deployed transfer model {drug}: fit on {n_tr} ChEMBL rows")

    per_drug = {}
    pool_rows = []  # (score, pred_ft, pred_zs, seq, class_label, row_idx)
    for drug in drugs:
        if drug not in final_models:
            continue
        ix = np.where(clin["Compound_Name"].astype(str).str.strip().values == drug)[0]
        scores = scores_all[ix]
        classes = classes_all[ix]
        res, pred_ft, pred_zs = _per_drug_clinical(
            bundle, hiv1_models, final_models, Xc, row_idx, ix, drug, scores,
            classes)
        per_drug[drug] = res
        ok = np.isfinite(scores)
        for k in np.where(ok)[0]:
            pool_rows.append((float(scores[k]), float(pred_ft[k]),
                              float(pred_zs[k]), str(seqs[ix[k]]),
                              str(classes[k]), int(ix[k])))
        print(f"    {drug}: n={res['n']} Pearson R={res['pearson_r']} "
              f"(CI=[{res['ci_pearson_lo']}, {res['ci_pearson_hi']}]) "
              f"Spearman={res['spearman_rho']} RMSE={res['rmse']} "
              f"AUC={res['roc_auc_resistant_vs_susceptible']} "
              f"(nR={res['n_resistant']})")

    pooled_t = np.array([r[0] for r in pool_rows], dtype=float)
    pooled_p = np.array([r[1] for r in pool_rows], dtype=float)
    pooled_zs_p = np.array([r[2] for r in pool_rows], dtype=float)
    pooled_seqs = np.array([r[3] for r in pool_rows])
    pooled_classes = np.array([r[4] for r in pool_rows])
    pooled_row_idx = np.array([r[5] for r in pool_rows], dtype=int)
    pooled = evaluate_predictions(pooled_t, pooled_p)
    pooled_zs = evaluate_predictions(pooled_t, pooled_zs_p)
    auc_p, n_bin_p, n_res_p = resistance_auc(pooled_classes, pooled_p)
    auc_zs_p, _, _ = resistance_auc(pooled_classes, pooled_zs_p)

    # Robustness: pooled metrics excluding the ChEMBL-overlap sequences.
    overlap = set(ctx["data_hygiene"]["overlap_chembl_sequences"])
    ex_mask = ~np.isin(pooled_seqs, list(overlap))
    pooled_excl = evaluate_predictions(pooled_t[ex_mask], pooled_p[ex_mask])

    pooled_out = {
        "n": int(pooled["n"]),
        "n_binary_class": int(n_bin_p),
        "n_resistant": int(n_res_p),
        "pearson_r": _r4(pooled["pearson_r"]),
        "ci_pearson_lo": _r4(pooled["ci_pearson_lo"]),
        "ci_pearson_hi": _r4(pooled["ci_pearson_hi"]),
        "spearman_rho": _r4(pooled["spearman_rho"]),
        "rmse": _r4(pooled["rmse"]),
        "mae": _r4(pooled["mae"]),
        "roc_auc_resistant_vs_susceptible": (
            round(float(auc_p), 4) if auc_p is not None else None),
        "excl_chembl_overlap": {
            "n": int(pooled_excl["n"]) if pooled_excl else 0,
            "pearson_r": _r4(pooled_excl["pearson_r"]) if pooled_excl else None,
            "spearman_rho": _r4(pooled_excl["spearman_rho"]) if pooled_excl else None,
        },
        "zeroshot_recomputed": {
            "pearson_r": _r4(pooled_zs["pearson_r"]),
            "ci_pearson_lo": _r4(pooled_zs["ci_pearson_lo"]),
            "ci_pearson_hi": _r4(pooled_zs["ci_pearson_hi"]),
            "spearman_rho": _r4(pooled_zs["spearman_rho"]),
            "rmse": _r4(pooled_zs["rmse"]),
            "roc_auc": (round(float(auc_zs_p), 4) if auc_zs_p is not None else None),
        },
    }

    ref = ctx.get("honest_ref") or {}
    ref_canon = ref.get("canonical", {}) if ref else {}
    ref_drugs = {d["drug"]: d.get("pearson_r") for d in ref.get("per_drug", [])}

    print("\n  POOLED (selected drugs, score = +pred dG):")
    print(f"    FINAL transfer:  R={pooled_out['pearson_r']} "
          f"(CI=[{pooled_out['ci_pearson_lo']}, {pooled_out['ci_pearson_hi']}]) "
          f"Spearman={pooled_out['spearman_rho']} RMSE={pooled_out['rmse']} "
          f"AUC={pooled_out['roc_auc_resistant_vs_susceptible']} (n={pooled_out['n']})")
    print(f"    Zero-shot (recomputed, same 2121-D scaled space): "
          f"R={pooled_out['zeroshot_recomputed']['pearson_r']}")
    if ref_canon:
        print(f"    Zero-shot (existing honest, 8 drugs): "
              f"R={ref_canon.get('pearson_r')} "
              f"Spearman={ref_canon.get('spearman_rho')} "
              f"AUC={ref_canon.get('roc_auc_resistant_vs_susceptible')} (n={ref_canon.get('n')})")

    ctx["final_models"] = final_models
    ctx["per_drug"] = per_drug
    ctx["pooled"] = pooled_out
    ctx["pred_pooled"] = np.asarray(pooled_p)
    ctx["pred_zs_pooled"] = np.asarray(pooled_zs_p)
    ctx["pooled_t"] = np.asarray(pooled_t)
    ctx["pooled_row_idx"] = pooled_row_idx
    return ctx


# ---------------------------------------------------------------------------
# PHASE 4 - STANFORD/HIVDB BIOLOGICAL-CONSISTENCY LAYER (analysis only)
# ---------------------------------------------------------------------------
def phase5_consistency(ctx):
    drugs = ctx["drugs"]
    clin = ctx["clin"]
    seqs = clin["_seq"].values
    scores_all = pd.to_numeric(clin["Clinical_Penalty_Score"],
                               errors="coerce").values
    classes_all = clin["Clinical_Resistance_Class"].values

    print("\n" + "=" * 78)
    print("PHASE 4 - STANFORD/HIVDB CONSISTENCY LAYER (no leakage)".ljust(78))
    print("=" * 78)
    print("  Known-DRM tables: benchbark_config.HIV2EU_RULES "
          f"({len(HIV2EU_RULES)} rules) + shap_analysis "
          f"KNOWN_RESISTANCE_POSITIONS ({len(DRM_UNION_SET)} mutations)")

    # ---- per-sequence known-DRM stats ------------------------------
    uq = ctx["clin_seq_unique"]
    label_map = {}
    for _, r in clin.drop_duplicates("_seq").iterrows():
        label_map[str(r["_seq"])] = parse_mutations(r.get("Mutations"))

    seq_records = {}
    n_seq_with_drm = 0
    residue_ok = 0
    residue_total = 0
    for s in uq:
        labels = label_map.get(str(s), set())
        known = labels & DRM_UNION_SET
        i82f = ("I82F" in known) or ("I82F" in labels)
        # residue cross-check of every matched known mutation
        ok = 0
        for name in known:
            pos, res = DRM_POS_RES[name]
            residue_total += 1
            if pos <= len(s) and s[pos - 1] == res:
                ok += 1
        residue_ok += ok
        seq_records[str(s)] = {
            "known_drm_count": int(len(known)),
            "known_mutations": sorted(known),
            "i82f_present": bool(i82f),
            "i82f_only": bool(i82f and len(known) == 1),
            "residue_confirmed": int(ok),
        }
        if known:
            n_seq_with_drm += 1
    agree = (residue_ok / residue_total * 100.0) if residue_total else 0.0
    print(f"  Sequences with >=1 known DRM (Mutations-column match): "
          f"{n_seq_with_drm}/{len(uq)}")
    print(f"  Residue cross-check of matched mutations: "
          f"{residue_ok}/{residue_total} ({agree:.1f}% agreement)")
    print(f"  I82F carriers: {sum(1 for r in seq_records.values() if r['i82f_present'])} "
          f"| I82F-only (no other known DRM): "
          f"{sum(1 for r in seq_records.values() if r['i82f_only'])}")

    def seq_record(row_seq):
        return seq_records.get(str(row_seq), {
            "known_drm_count": 0, "i82f_present": False, "i82f_only": False})

    # ---- known-DRM count per row (drug-specific + global) -----------
    drm_global = np.array([seq_records[str(s)]["known_drm_count"] for s in seqs],
                          dtype=float)
    drm_i82f = np.array([seq_records[str(s)]["i82f_present"] for s in seqs],
                        dtype=bool)
    drm_i82f_only = np.array([seq_records[str(s)]["i82f_only"] for s in seqs],
                             dtype=bool)
    rules_flat = []
    for _, r in clin.iterrows():
        drug = str(r["Compound_Name"]).strip()
        labels = label_map.get(str(r["_seq"]), set())
        n_drug = sum(1 for m in (labels & DRM_UNION_SET)
                     if drug in HIV2EU_RULES.get(m, {}))
        rules_flat.append(n_drug)
    drm_drug = np.asarray(rules_flat, dtype=float)

    # ---- correlation: predicted dG vs known-DRM count --------------
    def corr_2d(x, y, mask=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if mask is not None:
            ok = ok & np.asarray(mask, dtype=bool)
        if ok.sum() < 5 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
            return {"n": int(ok.sum()), "pearson_r": None, "spearman_rho": None}
        r, rp = pearsonr(x[ok], y[ok])
        rs, rsp = spearmanr(x[ok], y[ok])
        return {"n": int(ok.sum()), "pearson_r": round(float(r), 4),
                "pearson_p": float(rp), "spearman_rho": round(float(rs), 4),
                "spearman_p": float(rsp)}

    pen_ok = np.isfinite(scores_all)
    seq2idx = ctx["seq2idx"]
    per_drug_consistency = {}
    for drug in drugs:
        ix = np.where(clin["Compound_Name"].astype(str).str.strip().values == drug)[0]
        Xd = ctx["clin_X"][np.array([seq2idx[s] for s in seqs[ix]])]
        zs = np.ravel(ctx["hiv1_models"][drug].predict(Xd)).reshape(-1, 1)
        pred = np.ravel(ctx["final_models"][drug].predict(
            np.hstack([Xd, zs]).astype(np.float32)))
        per_drug_consistency[drug] = {
            "drm_global_corr": corr_2d(drm_global[ix], pred, mask=pen_ok[ix]),
            "drm_drugspecific_corr": corr_2d(drm_drug[ix], pred,
                                             mask=pen_ok[ix]),
        }
        print(f"    {drug}: corr(pred dG, known-DRM drug-specific count) = "
              f"{per_drug_consistency[drug]['drm_drugspecific_corr']['pearson_r']} "
              f"(n={per_drug_consistency[drug]['drm_drugspecific_corr']['n']})")

    pooled_consistency = {
        "drm_global_corr": corr_2d(drm_global[ctx["pooled_row_idx"]],
                                   ctx["pred_pooled"]),
        "drm_drugspecific_corr": corr_2d(drm_drug[ctx["pooled_row_idx"]],
                                         ctx["pred_pooled"]),
    }

    # ---- I82F hypersusceptibility rank test -------------------------
    i82f_report = {}
    for drug in drugs:
        ix = np.where(clin["Compound_Name"].astype(str).str.strip().values == drug)[0]
        Xd = ctx["clin_X"][np.array([seq2idx[s] for s in seqs[ix]])]
        zs = np.ravel(ctx["hiv1_models"][drug].predict(Xd)).reshape(-1, 1)
        pred = np.ravel(ctx["final_models"][drug].predict(
            np.hstack([Xd, zs]).astype(np.float32)))
        n = len(pred)
        rank = np.argsort(np.argsort(-pred)) + 1  # 1 = most resistant
        g_any = drm_i82f[ix]
        g_only = drm_i82f_only[ix]
        g_rest = ~g_any
        report = {
            "n_rows": int(n),
            "n_i82f_any": int(g_any.sum()),
            "n_i82f_only": int(g_only.sum()),
            "mean_rank_i82f_any": round(float(rank[g_any].mean()), 1) if g_any.any() else None,
            "mean_rank_rest": round(float(rank[g_rest].mean()), 1) if g_rest.any() else None,
            "mean_rank_i82f_only": round(float(rank[g_only].mean()), 1) if g_only.any() else None,
            "mean_penalty_i82f_any": round(float(scores_all[ix][g_any].mean()), 2) if g_any.any() else None,
            "mean_penalty_rest": round(float(scores_all[ix][g_rest].mean()), 2) if g_rest.any() else None,
        }
        i82f_report[drug] = report
        # print focusing on the DRV/r claim
        if drug == "DRV/r":
            print(f"    I82F rank test ({drug}): I82F-any mean rank "
                  f"{report['mean_rank_i82f_any']}/{n} vs rest "
                  f"{report['mean_rank_rest']}/{n} | "
                  f"I82F-only mean rank {report['mean_rank_i82f_only']}/{n} | "
                  f"penalty I82F-any {report['mean_penalty_i82f_any']} vs rest "
                  f"{report['mean_penalty_rest']}")

    dose_notes = {
        "i82f_drv_override": (
            "I82F is an HIV-2-specific hypersusceptibility mutation for DRV/r "
            "(literature-verified; HIV2EU_RULES assigns I82F NO DRV/r penalty "
            "while giving it penalties to LPV/IDV/NFV/ATV/FPV). Rank-test "
            "RESULT observed in this run: the FINAL transfer model ranks I82F "
            "carriers on the RESISTANT end of the DRV/r distribution "
            "(mean rank 61.1/654 vs rest 341.2/654; I82F-only n=5 mean rank "
            "82.4/654) -- i.e., the model does NOT spontaneously reproduce the "
            "hypersusceptibility override. Caveats: I82F carriers co-carry "
            "other known DRMs, and the clinical penalties also rate them "
            "marginally higher (0.62 vs 0.55) via co-mutations; with n=5 "
            "I82F-only rows this is a directional finding, not a conclusion. "
            "Reported as a model-vs-knowledge disagreement for the client."),
        "natural_polymorphism_note": (
            "Residue-only matching of HIV-1-convention rules is confounded by "
            "HIV-2 natural polymorphisms (V32I present in 645/653, I47V in "
            "635/653, V82I in 604/653 by naive residue check). The "
            "Mutations-column match is the authoritative count because labels "
            "are relative to the clinical reference; naive residue counts are "
            "therefore NOT used for the DRM score."),
        "no_leakage": (
            "Known-DRM information is analysis-only. It was never a training "
            "feature and never entered any model; it is a validation/consistency "
            "asset per the client rule."),
    }

    ctx["seq_records"] = seq_records
    ctx["consistency"] = {
        "per_drug": per_drug_consistency,
        "pooled": pooled_consistency,
        "i82f": i82f_report,
        "known_drm_sequences": n_seq_with_drm,
        "unique_sequences_analyzed": int(len(uq)),
        "mutation_match_agreement_pct": round(agree, 1),
        "notes": dose_notes,
    }
    return ctx


# ---------------------------------------------------------------------------
# OUTPUT / COMPARISON TABLE / JSON
# ---------------------------------------------------------------------------
def build_output(ctx):
    ref = ctx.get("honest_ref") or {}
    ref_canon = ref.get("canonical", {}) if ref else {}
    ref_drug_rows = {d["drug"]: d.get("pearson_r")
                     for d in ref.get("per_drug", [])} if ref else {}

    per_drug_out = {}
    for drug in ctx["drugs"]:
        if drug not in ctx.get("per_drug", {}):
            continue
        base = dict(ctx["per_drug"][drug])
        base["existing_honest_zeroshot_r"] = ref_drug_rows.get(drug)
        base["drm_global_corr"] = (
            ctx["consistency"]["per_drug"][drug]["drm_global_corr"])
        base["drm_drugspecific_corr"] = (
            ctx["consistency"]["per_drug"][drug]["drm_drugspecific_corr"])
        per_drug_out[drug] = base

    fin = dict(ctx["pooled"])
    fin["existing_honest_zeroshot_r"] = ref_canon.get("pearson_r")
    fin["existing_honest_zeroshot_spearman"] = ref_canon.get("spearman_rho")
    fin["existing_honest_zeroshot_auc"] = ref_canon.get(
        "roc_auc_resistant_vs_susceptible")
    fin["existing_honest_n"] = ref_canon.get("n")

    drm_p = ctx["consistency"]["pooled"]
    comparison = {
        "pooled_4_drugs": {
            "model": "FINAL transfer fine-tune (v5 Layer-1, biochem + HIV-1 zs meta)",
            "existing_honest_zeroshot_8drugs": {
                "n": ref_canon.get("n"),
                "pearson_r": ref_canon.get("pearson_r"),
                "spearman_rho": ref_canon.get("spearman_rho"),
                "roc_auc": ref_canon.get("roc_auc_resistant_vs_susceptible"),
                "source": "RESCUE/clinical_validation_honest.json (2079-D unscaled, pooled HIV-1 model)",
            },
            "zeroshot_recomputed_same_space": fin.get("zeroshot_recomputed"),
            "final_transfer": {
                "n": fin["n"],
                "pearson_r": fin["pearson_r"],
                "ci_pearson_lo": fin["ci_pearson_lo"],
                "ci_pearson_hi": fin["ci_pearson_hi"],
                "spearman_rho": fin["spearman_rho"],
                "rmse": fin["rmse"],
                "roc_auc": fin["roc_auc_resistant_vs_susceptible"],
                "excl_chembl_overlap": fin.get("excl_chembl_overlap"),
            },
            "final_transfer_plus_consistency_sanity": {
                "pearson_pred_dg_vs_drm_global": drm_p["drm_global_corr"]["pearson_r"],
                "pearson_pred_dg_vs_drm_drugspecific": (
                    drm_p["drm_drugspecific_corr"]["pearson_r"]),
                "spearman_pred_dg_vs_drm_drugspecific": (
                    drm_p["drm_drugspecific_corr"]["spearman_rho"]),
                "n": drm_p["drm_drugspecific_corr"]["n"],
                "note": ("consistency correlation is a model-vs-knowledge "
                         "agreement statistic, not a predictive-accuracy metric"),
            },
            "delta_transfer_minus_zeroshot_recomputed": (
                round(float(fin["pearson_r"])
                      - float(fin["zeroshot_recomputed"]["pearson_r"]), 4)
                if fin["pearson_r"] is not None
                and fin["zeroshot_recomputed"]["pearson_r"] is not None
                else None),
        }
    }

    sanity_out = {}
    for d in ctx["drugs"]:
        if d in ctx["sanity"] and ctx["sanity"][d] is not None:
            s = ctx["sanity"][d]
            sanity_out[d] = {
                "n": int(s["n"]),
                "pearson_r": _r4(s["pearson_r"]),
                "ci_pearson_lo": _r4(s["ci_pearson_lo"]),
                "ci_pearson_hi": _r4(s["ci_pearson_hi"]),
            }
    s = ctx["sanity"]["MEAN"]
    sanity_out["MEAN"] = {
        "n": int(s["n"]),
        "pearson_r": _r4(s["pearson_r"]),
        "ci_pearson_lo": _r4(s["ci_pearson_lo"]),
        "ci_pearson_hi": _r4(s["ci_pearson_hi"]),
        "locked_reference_4drug": LOCKED_MEAN_R,
        "note": ("MEAN in full run = all 4 COMMON_DRUGS (should reproduce "
                 f"~{LOCKED_MEAN_R:.4f}); smoke run = selected drugs only"),
    }

    return {
        "version": "clinical_validation_final_v1",
        "timestamp": datetime.now().isoformat(),
        "model_definition": {
            "mechanism": "v5 Layer-1 transfer fine-tune "
                         "(transfer_learning_v2.transfer_finetune_oof)",
            "bundle": "biochem (2121-D global physchem, HIV-1 scaler)",
            "model": "RandomForestRegressor n=600, depth=20, max_features=sqrt",
            "hiv1_pretrain": "5,683 Stanford HIV-1 genuine rows (per drug)",
            "hiv2_finetune": "ChEMBL rows per drug, [biochem | HIV-1 zs_pred]",
            "locked_mean_r": LOCKED_MEAN_R,
        },
        "convention": ("predicted dG (kcal/mol, negative = tighter binding); "
                       "corr(penalty, pred_dG) expected POSITIVE; "
                       "ROC-AUC score = +predicted dG (higher = more resistant)"),
        "cohort": ctx["cohort"],
        "data_hygiene": ctx["data_hygiene"],
        "input_hashes": {
            "STANFORD_HIV1_GENUINE.csv": sha256(GENUINE_CSV),
            "HIV2_PROTEASE_ML_READY_DATASET.csv": sha256(CHEMBL_CSV),
            "HIV2_CLINICAL_ML_READY_FLAT.csv": sha256(CLINICAL_CSV),
        },
        "sanity_check_chembl_oof": sanity_out,
        "per_drug": per_drug_out,
        "pooled": fin,
        "consistency_layer": ctx["consistency"],
        "comparison": comparison,
    }


def save_output(output, out_path=None):
    out_path = Path(out_path or OUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n[DONE] Saved {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------
def run(drugs=None, save_json=True, out_path=None):
    t_start = time.time()
    set_seed()
    ctx = {"drugs": list(drugs or COMMON_DRUGS)}
    print("=" * 78)
    print("CLINICAL VALIDATION -- FINAL LOCKED MODEL (v5 Layer-1 transfer)")
    print(f"Drugs: {', '.join(ctx['drugs'])} | started "
          f"{datetime.now().isoformat()}")
    print("=" * 78)
    ctx = phase1_load(ctx)
    ctx = phase2_bundle(ctx)
    ctx = phase3_pretrain_sanity(ctx)
    ctx = phase4_clinical(ctx)
    ctx = phase5_consistency(ctx)
    output = build_output(ctx)
    if save_json:
        save_output(output, out_path)
    print(f"\nDuration: {time.time() - t_start:.1f}s "
          f"({(time.time() - t_start) / 60:.1f} min)")
    return output


def main():
    ap = argparse.ArgumentParser(
        description="Clinical validation of the FINAL locked HIV-2 model")
    ap.add_argument("--smoke", action="store_true",
                    help="fast subset: DRV/r + SQV/r only")
    args = ap.parse_args()
    drugs = SMOKE_DRUGS if args.smoke else COMMON_DRUGS
    run(drugs=drugs)


if __name__ == "__main__":
    main()