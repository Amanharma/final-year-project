"""Classical anomaly-detection baselines on the official OPSSAT-AD train/test split.

Outputs:
  results/classical_metrics.csv        all models x feature sets
  results/classical_per_channel.csv    per-channel scores of the best model
  results/classical_confusion.png      confusion matrix of the best model
  results/classical_importance.png     top-15 RandomForest feature importances
  results/models/best_classical.joblib fitted best model (used by the twin)
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dt.classical import evaluate, make_models, predict  # noqa: E402
from dt.ingest import PUBLISHED, build_features  # noqa: E402

RES = ROOT / "results"
(RES / "models").mkdir(parents=True, exist_ok=True)


def main():
    fs = build_features()
    X_tr, X_te, y_tr, y_te = fs.split
    print(f"train {X_tr.shape}, test {X_te.shape}, test anomalies {int(y_te.sum())}")

    ch_cols = [c for c in fs.X.columns if c.startswith("ch_")]
    own = [c for c in fs.X.columns if c not in PUBLISHED and c not in ch_cols]
    feature_sets = {
        "published": PUBLISHED + ch_cols,
        "own": own + ch_cols,
        "combined": list(fs.X.columns),
    }

    rows, fitted = [], {}
    contamination = float(y_tr.mean())
    for fs_name, cols in feature_sets.items():
        for name, model in make_models(contamination).items():
            m, mdl = evaluate(name, model, X_tr[cols], y_tr, X_te[cols], y_te)
            m["features"] = fs_name
            m["n_features"] = len(cols)
            rows.append(m)
            fitted[(fs_name, name)] = mdl
            print(f"{fs_name:9s} {name:15s} F1={m['f1']:.3f} AUC={m['roc_auc']:.3f} "
                  f"P={m['precision']:.3f} R={m['recall']:.3f}")

    df = pd.DataFrame(rows)[["features", "n_features", "model", "accuracy", "precision", "recall", "f1",
                             "roc_auc", "tp", "fp", "fn", "tn", "fit_s", "infer_ms_per_segment"]]
    df.round(4).to_csv(RES / "classical_metrics.csv", index=False)

    best = df.sort_values("f1", ascending=False).iloc[0]
    key = (best["features"], best["model"])
    model, cols = fitted[key], feature_sets[best["features"]]
    joblib.dump({"model": model, "columns": cols, "name": best["model"], "features": best["features"]},
                RES / "models" / "best_classical.joblib")

    # the live twin can only compute our own features from the stream (the published ones come
    # from the dataset authors' generator), so it uses the best supervised model on "own" features
    live = df[(df.features == "own") & (df.model != "IsolationForest")].sort_values("f1").iloc[-1]
    from dt.ingest import channel_stats, load_segments
    joblib.dump({"model": fitted[("own", live["model"])], "columns": feature_sets["own"],
                 "name": live["model"], "features": "own",
                 "channel_stats": channel_stats(load_segments())},
                RES / "models" / "live_classical.joblib")

    y_hat = predict(model, X_te[cols], best["model"] == "IsolationForest")
    per_ch = []
    for ch in sorted(fs.channel.unique()):
        mask = (fs.channel.loc[X_te.index] == ch).to_numpy()
        yt, yp = y_te.to_numpy()[mask], y_hat[mask]
        per_ch.append({"channel": ch, "test_segments": int(mask.sum()), "test_anomalies": int(yt.sum()),
                       "f1": round(f1_score(yt, yp, zero_division=0), 4) if yt.sum() else None,
                       "recall": round(recall_score(yt, yp, zero_division=0), 4) if yt.sum() else None,
                       "false_alarms": int(((yp == 1) & (yt == 0)).sum())})
    pd.DataFrame(per_ch).to_csv(RES / "classical_per_channel.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_te, y_hat, display_labels=["nominal", "anomaly"], ax=ax,
                                            colorbar=False)
    ax.set_title(f"{best['model']} ({best['features']} features)\nOPSSAT-AD test set, n={len(y_te)}")
    fig.tight_layout()
    fig.savefig(RES / "classical_confusion.png", dpi=150)

    rf = fitted[("combined", "RandomForest")]
    imp = pd.Series(rf.feature_importances_, index=feature_sets["combined"]).nlargest(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.plot.barh(ax=ax)
    ax.set_title("RandomForest feature importance (top 15, combined features)")
    fig.tight_layout()
    fig.savefig(RES / "classical_importance.png", dpi=150)

    summary = {"best": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()},
               "n_train": len(y_tr), "n_test": len(y_te), "test_anomalies": int(y_te.sum()),
               "seed": 42}
    (RES / "classical_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["best"], indent=2, default=str))


if __name__ == "__main__":
    np.random.seed(42)
    main()
