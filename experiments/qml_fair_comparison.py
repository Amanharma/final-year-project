"""Quantum-kernel vs classical classifiers under equal tuning effort.

`quantum_vs_classical.py` is the first, untuned comparison (one seed, 400 balanced training
segments, fixed C). This script answers what that one leaves open:

  1. Training-set size. Every model is trained twice: on balanced 400-segment subsamples
     (5 seeds) and on the full official training split (1,594 segments).
  2. Tuning. QSVC (kernel bandwidth alpha, C) and SVM-RBF (gamma, C) are both grid-searched by
     5-fold out-of-fold PR-AUC on the training data. The untuned settings are kept as
     "default" rows so the effect of tuning is visible.
  3. Decision threshold. F1 is reported at the default threshold and at the threshold that
     maximises out-of-fold F1 on the training data. PR-AUC and ROC-AUC are threshold-free.
  4. Uncertainty. Mean and standard deviation over subsample seeds, and a 95% bootstrap
     interval over test segments for every run.
  5. PCA against subsample. The SVM-RBF on all features is run on both training sets, so the
     cost of PCA and the cost of the smaller training set can be read separately.

The test split is used only for the final scores.

    python experiments/qml_fair_comparison.py            # about 10-20 min on a laptop
    python experiments/qml_fair_comparison.py --vqc      # also retrain the VQC (slow)
    python experiments/qml_fair_comparison.py --quick    # 1 seed, small grids (smoke test)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dt.evaluation import best_f1_threshold, oof_svc_scores, report, select_svc  # noqa: E402
from dt.ingest import build_features  # noqa: E402
from dt.quantum_ml import SEED, balanced_subsample, make_reducer, make_vqc, statevector_kernel  # noqa: E402

DIMS = (2, 4, 6)
N_BALANCED = 400
BALANCED_SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (0.1, 0.25, 0.5, 1.0)            # features are scaled to [0, alpha * pi]
C_GRID = (0.1, 1.0, 10.0, 100.0)
GAMMA_GRID = ("scale", 0.01, 0.1, 1.0, 10.0)
DEFAULT_C = 10.0                          # the untuned setting of quantum_vs_classical.py


def row(model, tuned, train_set, seed, dims, n_train, y_te, score, thr_default, thr_oof, n_boot, **extra):
    """One result row: metrics at the default threshold, plus F1 at the OOF-chosen threshold."""
    base = report(y_te, score, thr_default, n_boot)
    at_oof = report(y_te, score, thr_oof, n_boot)
    return {"model": model, "tuned": tuned, "train_set": train_set, "seed": seed, "dims": dims,
            "n_train": n_train, **base,
            "f1_oofthr": at_oof["f1"], "f1_oofthr_ci_lo": at_oof["f1_ci_lo"],
            "f1_oofthr_ci_hi": at_oof["f1_ci_hi"], "precision_oofthr": at_oof["precision"],
            "recall_oofthr": at_oof["recall"], **extra}


def svc_rows(name, train_set, seed, dims, cands_tr, cands_te, y_tr, y_te, grid, default_key,
             default_params, precomputed, n_boot):
    """Default and tuned SVC on the same candidates. cands_*: {key: X or K}."""
    out = []
    oof = oof_svc_scores(cands_tr[default_key], y_tr, precomputed, **default_params)
    chosen = select_svc(cands_tr, y_tr, grid, precomputed)
    for tuned, key, params, thr in ((False, default_key, default_params, best_f1_threshold(y_tr, oof)),
                                    (True, chosen["key"], chosen["params"], chosen["threshold"])):
        m = SVC(kernel="precomputed" if precomputed else "rbf", class_weight="balanced", **params)
        m.fit(cands_tr[key], y_tr)
        score = m.decision_function(cands_te[key])
        out.append(row(name, tuned, train_set, seed, dims, len(y_tr), y_te, score, 0.0, thr, n_boot,
                       setting=json.dumps({"key": key, **params})))
    return out


def vqc_row(d, A_tr, y_tr, A_te, y_te, seed, n_boot, maxiter, restarts):
    """Best of several restarts by final training loss; threshold fixed at 0.5."""
    best = None
    for r in range(restarts):
        hist = []
        m = make_vqc(d, maxiter=maxiter, history=hist, seed=SEED + r).fit(A_tr, y_tr)
        if best is None or hist[-1] < best[0]:
            best = (hist[-1], m, r)
    loss, m, r = best
    score = np.asarray(m.predict_proba(A_te))[:, 1]
    return row("VQC", True, "balanced400", seed, d, len(y_tr), y_te, score, 0.5, 0.5, n_boot,
               setting=json.dumps({"maxiter": maxiter, "restart": r, "train_loss_bits": round(loss, 4)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "results"), help="output directory")
    ap.add_argument("--quick", action="store_true", help="1 seed, small grids, few bootstrap draws")
    ap.add_argument("--vqc", action="store_true", help="also retrain the VQC (300 iterations, 3 restarts)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = BALANCED_SEEDS[:1] if args.quick else BALANCED_SEEDS
    alphas = (0.25, 1.0) if args.quick else ALPHAS
    c_grid = (1.0, 10.0) if args.quick else C_GRID
    gammas = ("scale", 0.1) if args.quick else GAMMA_GRID
    n_boot = 200 if args.quick else 2000

    fs = build_features()
    X_tr_df, X_te_df, y_tr_s, y_te_s = fs.split
    X_tr, X_te = X_tr_df.to_numpy(float), X_te_df.to_numpy(float)
    y_tr, y_te = y_tr_s.to_numpy(int), y_te_s.to_numpy(int)
    n_features = X_tr.shape[1]

    configs = [("balanced400", s, *balanced_subsample(X_tr, y_tr, N_BALANCED, seed=s)) for s in seeds]
    configs.append(("full", 0, X_tr, y_tr))

    rows, t_start = [], time.perf_counter()
    for train_set, seed, Xc, yc in configs:
        t0 = time.perf_counter()
        # reference: SVM-RBF on every feature, same training segments
        sc = StandardScaler().fit(Xc)
        rows += svc_rows("SVM-RBF", train_set, seed, "all", {"std": sc.transform(Xc)},
                         {"std": sc.transform(X_te)}, yc, y_te, {"C": c_grid, "gamma": gammas},
                         "std", {"C": DEFAULT_C, "gamma": "scale"}, False, n_boot)
        for d in DIMS:
            red = make_reducer(d).fit(Xc)
            A_tr, A_te = red.transform(Xc), red.transform(X_te)
            # quantum kernel: one training kernel per bandwidth, tuned without touching the test set
            K_tr = {a: statevector_kernel(d, a * A_tr) for a in alphas}
            K_te = {a: statevector_kernel(d, a * A_te, a * A_tr) for a in alphas}
            rows += svc_rows("QSVC", train_set, seed, d, K_tr, K_te, yc, y_te, {"C": c_grid},
                             1.0, {"C": DEFAULT_C}, True, n_boot)
            rows += svc_rows("SVM-RBF", train_set, seed, d, {"pca": A_tr}, {"pca": A_te}, yc, y_te,
                             {"C": c_grid, "gamma": gammas}, "pca",
                             {"C": DEFAULT_C, "gamma": "scale"}, False, n_boot)
            rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED,
                                        n_jobs=-1)
            oof = cross_val_predict(rf, A_tr, yc, method="predict_proba",
                                    cv=StratifiedKFold(5, shuffle=True, random_state=SEED))[:, 1]
            score = rf.fit(A_tr, yc).predict_proba(A_te)[:, 1]
            rows.append(row("RandomForest", False, train_set, seed, d, len(yc), y_te, score, 0.5,
                            best_f1_threshold(yc, oof), n_boot, setting="{}"))
            if args.vqc and train_set == "balanced400" and seed == seeds[0]:
                rows.append(vqc_row(d, A_tr[:200], yc[:200], A_te, y_te, seed, n_boot,
                                    maxiter=20 if args.quick else 300, restarts=1 if args.quick else 3))
        print(f"{train_set:11s} seed={seed} n={len(yc)} done in {time.perf_counter() - t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df["dims"] = df["dims"].astype(str)
    df.round(4).to_csv(out_dir / "qml_fair_runs.csv", index=False)

    metrics = ["f1", "f1_oofthr", "precision_oofthr", "recall_oofthr", "roc_auc", "pr_auc"]
    agg = df.groupby(["train_set", "dims", "model", "tuned"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index().merge(
        df.groupby(["train_set", "dims", "model", "tuned"]).agg(
            runs=("seed", "count"), n_train=("n_train", "first"),
            f1_oofthr_ci_lo=("f1_oofthr_ci_lo", "mean"), f1_oofthr_ci_hi=("f1_oofthr_ci_hi", "mean")
        ).reset_index())
    agg.round(4).to_csv(out_dir / "qml_fair_summary.csv", index=False)

    red_full = make_reducer(max(DIMS)).fit(X_tr)
    evr = np.cumsum(red_full.named_steps["pca"].explained_variance_ratio_)

    def pick(train_set, dims, model, tuned, col="f1_oofthr_mean"):
        m = agg[(agg.train_set == train_set) & (agg.dims == str(dims)) & (agg.model == model)
                & (agg.tuned == tuned)]
        return round(float(m[col].iloc[0]), 4) if len(m) else None

    full_tuned = agg[(agg.train_set == "full") & (agg.dims != "all")
                     & ((agg.tuned) | (agg.model == "RandomForest"))]
    best_q = full_tuned[full_tuned.model == "QSVC"].sort_values("pr_auc_mean").iloc[-1]
    best_c = full_tuned[full_tuned.model != "QSVC"].sort_values("pr_auc_mean").iloc[-1]
    summary = {
        "n_train_full": int(len(y_tr)), "n_test": int(len(y_te)), "test_anomalies": int(y_te.sum()),
        "n_features": int(n_features), "balanced_seeds": list(seeds), "bootstrap_draws": n_boot,
        "grids": {"alpha": list(alphas), "C": list(c_grid), "gamma": list(gammas)},
        "pca_cumulative_explained_variance": {int(d): round(float(evr[d - 1]), 4) for d in DIMS},
        "best_quantum_full_tuned": best_q.to_dict(),
        "best_classical_same_features_full_tuned": best_c.to_dict(),
        # reading these four numbers together separates the two effects
        "svm_f1_oofthr_decomposition": {
            "all_features_full": pick("full", "all", "SVM-RBF", True),
            "all_features_balanced400": pick("balanced400", "all", "SVM-RBF", True),
            f"pca{max(DIMS)}_full": pick("full", max(DIMS), "SVM-RBF", True),
            f"pca{max(DIMS)}_balanced400": pick("balanced400", max(DIMS), "SVM-RBF", True)},
        "wall_s": round(time.perf_counter() - t_start, 1), "seed": SEED,
    }
    (out_dir / "qml_fair_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    show = agg[(agg.dims != "all") & ((agg.tuned) | (agg.model == "RandomForest")) & (agg.model != "VQC")]
    for (train_set, model), g in show.groupby(["train_set", "model"]):
        g = g.assign(d=g.dims.astype(int)).sort_values("d")
        style = "o-" if train_set == "full" else "s--"
        if train_set == "full":
            err = [g.f1_oofthr_mean - g.f1_oofthr_ci_lo, g.f1_oofthr_ci_hi - g.f1_oofthr_mean]
        else:
            err = g.f1_oofthr_std.fillna(0)
        ax[0].errorbar(g.d, g.f1_oofthr_mean, yerr=err, fmt=style, capsize=3, label=f"{model} ({train_set})")
        ax[1].plot(g.d, g.pr_auc_mean, style, label=f"{model} ({train_set})")
    ax[0].set_ylabel("F1 at training-chosen threshold")
    ax[1].set_ylabel("PR-AUC")
    for a in ax:
        a.set_xlabel("PCA components = qubits"); a.set_xticks(DIMS); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Tuned quantum kernel vs tuned classical models (test n={len(y_te)}); "
                 "bars: 95% bootstrap (full) or std over seeds (balanced400)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "qml_fair.png", dpi=200)

    print(agg[["train_set", "dims", "model", "tuned", "runs", "f1_mean", "f1_oofthr_mean",
               "f1_oofthr_std", "pr_auc_mean", "roc_auc_mean"]].round(3).to_string(index=False))
    print(json.dumps(summary["svm_f1_oofthr_decomposition"], indent=2))


if __name__ == "__main__":
    main()
