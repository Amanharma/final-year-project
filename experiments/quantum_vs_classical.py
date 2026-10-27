"""Quantum vs classical anomaly classification on identical PCA features.

For d in {2, 4, 6} qubits/components:
  - QSVC (ZZ fidelity kernel, exact statevector) trained on a balanced 400-segment subsample
  - VQC (ZZ + RealAmplitudes, COBYLA, 1024 shots) trained on a balanced 200-segment subsample
  - SVM-RBF and RandomForest trained on the same 400-segment subsample and PCA features
All are evaluated on the full official OPSSAT-AD test split (529 segments).
Also measures quantum-kernel matrix time vs training-set size (O(N^2) entries).
"""
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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dt.ingest import build_features  # noqa: E402
from dt.quantum_ml import (SEED, balanced_subsample, circuit_stats, kernel_matrix_time,  # noqa: E402
                           make_qsvc, make_reducer, make_vqc)

RES = ROOT / "results"
DIMS = (2, 4, 6)
N_KERNEL, N_VQC = 400, 200


def metrics(name, d, y_te, y_hat, score, fit_s, inf_s, n_train):
    return {"model": name, "qubits_or_dims": d, "n_train": n_train,
            "precision": precision_score(y_te, y_hat, zero_division=0),
            "recall": recall_score(y_te, y_hat, zero_division=0),
            "f1": f1_score(y_te, y_hat, zero_division=0),
            "roc_auc": roc_auc_score(y_te, score) if score is not None else np.nan,
            "fit_s": fit_s, "infer_ms_per_segment": 1000 * inf_s / len(y_te)}


def run(model, name, d, Xtr, ytr, Xte, yte):
    t0 = time.perf_counter(); model.fit(Xtr, ytr); fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_hat = np.asarray(model.predict(Xte)).astype(int).ravel()
    if hasattr(model, "decision_function"):
        score = model.decision_function(Xte)
    elif hasattr(model, "predict_proba"):
        score = np.asarray(model.predict_proba(Xte))[:, 1]
    else:
        score = None
    inf_s = time.perf_counter() - t0
    m = metrics(name, d, yte, y_hat, score, fit_s, inf_s, len(ytr))
    print(f"d={d} {name:13s} F1={m['f1']:.3f} AUC={m['roc_auc']:.3f} fit={fit_s:.1f}s")
    return m


def main():
    fs = build_features()
    X_tr_df, X_te_df, y_tr_s, y_te_s = fs.split
    X_tr, X_te = X_tr_df.to_numpy(float), X_te_df.to_numpy(float)
    y_tr, y_te = y_tr_s.to_numpy(int), y_te_s.to_numpy(int)

    Xk, yk = balanced_subsample(X_tr, y_tr, N_KERNEL)
    Xv, yv = Xk[:N_VQC], yk[:N_VQC]
    print(f"subsample {len(yk)} ({yk.mean():.2f} anomalous), VQC {len(yv)}, test {len(y_te)}")

    rows, curves, circuits = [], {}, []
    for d in DIMS:
        red = make_reducer(d).fit(Xk)
        A_k, A_v, A_te = red.transform(Xk), red.transform(Xv), red.transform(X_te)
        rows.append(run(make_qsvc(d), "QSVC", d, A_k, yk, A_te, y_te))
        hist = []
        rows.append(run(make_vqc(d, history=hist), "VQC", d, A_v, yv, A_te, y_te))
        curves[d] = hist
        rows.append(run(SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
                            random_state=SEED), "SVM-RBF", d, A_k, yk, A_te, y_te))
        rows.append(run(RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               random_state=SEED, n_jobs=-1),
                        "RandomForest", d, A_k, yk, A_te, y_te))
        circuits.append(circuit_stats(d))

    df = pd.DataFrame(rows)
    df.round(4).to_csv(RES / "qml_metrics.csv", index=False)
    pd.DataFrame(circuits).to_csv(RES / "qml_circuits.csv", index=False)

    # kernel cost vs training size (d = 4)
    red4 = make_reducer(4).fit(X_tr)
    A_all = red4.transform(X_tr)
    scaling = [{"n": n, "kernel_entries": n * n, "seconds": round(kernel_matrix_time(4, A_all[:n]), 4)}
               for n in (50, 100, 200, 400, 800, 1594)]
    pd.DataFrame(scaling).to_csv(RES / "qml_kernel_scaling.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, g in df.groupby("model"):
        ax[0].plot(g.qubits_or_dims, g.f1, "o-", label=name)
    ax[0].axhline(0.919, color="gray", ls=":", label="SVM-RBF, all 43 features, full train")
    ax[0].set_xlabel("PCA components = qubits"); ax[0].set_ylabel("F1 (test, n=529)")
    ax[0].set_xticks(DIMS); ax[0].legend(fontsize=8); ax[0].set_title("Quantum vs classical")
    for d, h in curves.items():
        ax[1].plot(h, label=f"d={d}")
    ax[1].set_xlabel("COBYLA iteration"); ax[1].set_ylabel("cross-entropy loss")
    ax[1].legend(); ax[1].set_title("VQC training")
    s = pd.DataFrame(scaling)
    ax[2].loglog(s.n, s.seconds, "o-")
    ax[2].set_xlabel("training segments N"); ax[2].set_ylabel("kernel matrix time [s]")
    ax[2].set_title("Quantum kernel cost (d=4, statevector)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "qml_comparison.png", dpi=150)

    best_q = df[df.model.isin(["QSVC", "VQC"])].sort_values("f1").iloc[-1]
    best_c = df[~df.model.isin(["QSVC", "VQC"])].sort_values("f1").iloc[-1]
    summary = {"n_kernel_train": int(len(yk)), "n_vqc_train": int(len(yv)), "n_test": int(len(y_te)),
               "best_quantum": best_q.to_dict(), "best_classical_same_features": best_c.to_dict(),
               "vqc_final_loss": {d: round(h[-1], 4) for d, h in curves.items()},
               "kernel_scaling": scaling, "circuits": circuits, "seed": SEED}
    (RES / "qml_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("best_quantum", "best_classical_same_features")},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
