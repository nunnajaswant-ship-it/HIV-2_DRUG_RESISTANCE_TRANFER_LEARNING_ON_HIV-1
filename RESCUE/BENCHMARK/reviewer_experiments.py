"""
reviewer_experiments.py
=======================
Answers the three reviewer critiques with real runs on the locked pipeline.

#3  Multi-seed: how much does the headline R=0.7018 move under different
    random seeds?  (report mean +- SD over seeds)

#1  Fine-tune-size sweep: train on k HIV-2 rows per drug (k = 1,2,5,10,20,30)
    and test on the held-out remainder, for two models:
        transfer  = [biochem | HIV-1 zero-shot prediction]   (uses HIV-1)
        no-HIV-1  = [biochem] only                            (no HIV-1 knowledge)
    Question: does the HIV-1 contribution grow as HIV-2 data shrinks?

#2  Pooled-math test: show that pooling the no-HIV-1 model across drugs gives a
    pooled R below every per-drug R, then test the explanation by standardising
    each drug's predictions before pooling.

Output: RESCUE/BENCHMARK/results/reviewer_experiments.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, train_test_split

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parents[1]
sys.path.insert(0, str(BENCH))

import clinical_validation_final as cvf  # noqa: E402

OUT = BENCH / "results" / "reviewer_experiments.json"
R4 = lambda x: round(float(x), 4)


def r_of(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(pearsonr(a[m], b[m])[0])


def main():
    print("loading locked pipeline (features + HIV-1 pre-train)...")
    ctx = {"drugs": list(cvf.COMMON_DRUGS)}
    ctx = cvf.phase1_load(ctx)
    ctx = cvf.phase2_bundle(ctx)
    bundle = ctx["bundle"]

    hiv1_models, _ = cvf.pretrain_hiv1(bundle, cvf.MODEL_NAME)
    print("HIV-1 pre-train done.")

    # per-drug HIV-2 features, targets and zero-shot meta-feature
    D = {}
    for drug in cvf.COMMON_DRUGS:
        mask = bundle.drug2 == drug
        X = bundle.X2[mask]
        y = bundle.y2[mask]
        zs = np.ravel(hiv1_models[drug].predict(X))
        D[drug] = (X, y, zs)
        print(f"  {drug}: n={len(y)}")

    rep = {}

    # ------------------------------------------------------------------ #3
    print("\n" + "=" * 70)
    print("#3  MULTI-SEED RERUNS of the headline OOF benchmark")
    print("=" * 70)
    seeds = [42, 1, 7, 13, 21, 99, 123, 2024, 5, 17]
    multi = []
    for s in seeds:
        pt, pp, nt, np_ = [], [], [], []
        for drug in cvf.COMMON_DRUGS:
            X, y, zs = D[drug]
            for tr, te in KFold(5, shuffle=True,
                                random_state=s).split(X):
                # transfer: biochem + zero-shot meta-feature
                m_t = cvf.build_fast_model(cvf.MODEL_NAME)
                cvf.set_seed(s)
                m_t.fit(np.hstack([X[tr], zs[tr].reshape(-1, 1)]).astype(np.float32), y[tr])
                pt.extend(y[te].tolist())
                pp.extend(np.ravel(m_t.predict(np.hstack([X[te], zs[te].reshape(-1, 1)]).astype(np.float32))).tolist())
                # no-HIV-1: biochem only
                m_n = cvf.build_fast_model(cvf.MODEL_NAME)
                cvf.set_seed(s)
                m_n.fit(X[tr].astype(np.float32), y[tr])
                nt.extend(y[te].tolist())
                np_.extend(np.ravel(m_n.predict(X[te].astype(np.float32))).tolist())
        multi.append({"seed": s, "transfer_r": R4(r_of(pt, pp)),
                      "no_hiv1_r": R4(r_of(nt, np_))})
        print(f"  seed {s:>5}: transfer R={multi[-1]['transfer_r']:.4f}   "
              f"no-HIV-1 R={multi[-1]['no_hiv1_r']:.4f}")
    tr = np.array([m["transfer_r"] for m in multi])
    nt = np.array([m["no_hiv1_r"] for m in multi])
    rep["multiseed"] = {
        "seeds": seeds, "per_seed": multi,
        "transfer_mean": R4(tr.mean()), "transfer_sd": R4(tr.std(ddof=1)),
        "transfer_min": R4(tr.min()), "transfer_max": R4(tr.max()),
        "no_hiv1_mean": R4(nt.mean()), "no_hiv1_sd": R4(nt.std(ddof=1)),
    }
    print(f"\n  MEAN over {len(seeds)} seeds:  transfer R = {tr.mean():.4f} "
          f"+- {tr.std(ddof=1):.4f}  (range {tr.min():.4f}-{tr.max():.4f})")
    print(f"                        no-HIV-1 R = {nt.mean():.4f} "
          f"+- {nt.std(ddof=1):.4f}")

    # ------------------------------------------------------------------ #1
    print("\n" + "=" * 70)
    print("#1  FINE-TUNE-SIZE SWEEP  (k HIV-2 rows per drug, held-out test)")
    print("=" * 70)
    ks = [1, 2, 5, 10, 20, 30]
    n_rep = 20
    sweep = []
    for k in ks:
        tri, noi = [], []
        for r in range(n_rep):
            pt, pp, nt2, np2 = [], [], [], []
            for drug in cvf.COMMON_DRUGS:
                X, y, zs = D[drug]
                if k >= len(y):
                    continue
                tr_idx, te_idx = train_test_split(
                    np.arange(len(y)), train_size=k, random_state=1000 * r + k)
                m_t = cvf.build_fast_model(cvf.MODEL_NAME)
                cvf.set_seed(r)
                m_t.fit(np.hstack([X[tr_idx], zs[tr_idx].reshape(-1, 1)]).astype(np.float32), y[tr_idx])
                pt.extend(y[te_idx].tolist())
                pp.extend(np.ravel(m_t.predict(np.hstack([X[te_idx], zs[te_idx].reshape(-1, 1)]).astype(np.float32))).tolist())
                m_n = cvf.build_fast_model(cvf.MODEL_NAME)
                cvf.set_seed(r)
                m_n.fit(X[tr_idx].astype(np.float32), y[tr_idx])
                nt2.extend(y[te_idx].tolist())
                np2.extend(np.ravel(m_n.predict(X[te_idx].astype(np.float32))).tolist())
            tri.append(r_of(pt, pp))
            noi.append(r_of(nt2, np2))
        gap = np.array(tri) - np.array(noi)
        sweep.append({"k_per_drug": k, "n_train_total": 4 * k,
                      "transfer_r_mean": R4(np.nanmean(tri)),
                      "transfer_r_sd": R4(np.nanstd(tri, ddof=1)),
                      "no_hiv1_r_mean": R4(np.nanmean(noi)),
                      "no_hiv1_r_sd": R4(np.nanstd(noi, ddof=1)),
                      "gap_mean": R4(np.nanmean(gap)),
                      "gap_sd": R4(np.nanstd(gap, ddof=1))})
        s = sweep[-1]
        print(f"  k={k:>2}/drug (n={4*k:>3}): transfer {s['transfer_r_mean']:.4f}"
              f"+-{s['transfer_r_sd']:.3f}   no-HIV-1 {s['no_hiv1_r_mean']:.4f}"
              f"+-{s['no_hiv1_r_sd']:.3f}   gap {s['gap_mean']:+.4f}")
    rep["fine_tune_size_sweep"] = {"n_repeats": n_rep, "rows": sweep}

    # ------------------------------------------------------------------ #2
    print("\n" + "=" * 70)
    print("#2  POOLED-MATH TEST  (why pooled no-HIV-1 R < every per-drug R)")
    print("=" * 70)
    per = {}
    all_t, all_p = [], []
    per_z_t, per_z_p = [], []
    for drug in cvf.COMMON_DRUGS:
        X, y, zs = D[drug]
        pt, pp = [], []
        for tr, te in KFold(5, shuffle=True, random_state=42).split(X):
            m = cvf.build_fast_model(cvf.MODEL_NAME)
            cvf.set_seed(42)
            m.fit(X[tr].astype(np.float32), y[tr])
            pt.extend(y[te].tolist())
            pp.extend(np.ravel(m.predict(X[te].astype(np.float32))).tolist())
        pt, pp = np.array(pt), np.array(pp)
        per[drug] = R4(r_of(pt, pp))
        all_t.extend(pt.tolist())
        all_p.extend(pp.tolist())
        mu, sd = pp.mean(), pp.std()
        per_z_t.extend(pt.tolist())
        per_z_p.extend(((pp - mu) / sd).tolist())
        print(f"  {drug}: per-drug no-HIV-1 R = {per[drug]:.4f}")
    pooled_raw = R4(r_of(all_t, all_p))
    pooled_std = R4(r_of(per_z_t, per_z_p))
    print(f"\n  pooled (raw predictions)          R = {pooled_raw:.4f}")
    print(f"  pooled (per-drug z-scored)        R = {pooled_std:.4f}")
    print(f"  lowest per-drug R                 = {min(per.values()):.4f}")
    rep["pooled_math_test"] = {
        "per_drug_no_hiv1_r": per,
        "lowest_per_drug_r": R4(min(per.values())),
        "pooled_raw_r": pooled_raw,
        "pooled_after_per_drug_zscore_r": pooled_std,
        "explanation_supported": bool(pooled_std > pooled_raw),
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
