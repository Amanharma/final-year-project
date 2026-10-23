"""Classical anomaly detectors used as baselines for the quantum models."""
import time

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42


def make_models(contamination: float = 0.2) -> dict:
    return {
        # unsupervised: fitted on nominal training segments only, never sees anomaly labels
        "IsolationForest": make_pipeline(StandardScaler(), IsolationForest(
            n_estimators=300, contamination=contamination, random_state=SEED)),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(
            kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", random_state=SEED)),
    }


def anomaly_score(model, X, unsupervised: bool = False) -> np.ndarray:
    """Higher = more anomalous, for every model type."""
    if unsupervised:
        return -model.score_samples(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def predict(model, X, unsupervised: bool = False) -> np.ndarray:
    p = model.predict(X)
    # IsolationForest returns -1 for outliers, +1 for inliers
    return (p == -1).astype(int) if unsupervised else p.astype(int)


def evaluate(name: str, model, X_tr, y_tr, X_te, y_te) -> tuple[dict, object]:
    unsupervised = name == "IsolationForest"
    t0 = time.perf_counter()
    model.fit(X_tr[y_tr == 0] if unsupervised else X_tr, None if unsupervised else y_tr)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_hat = predict(model, X_te, unsupervised)
    score = anomaly_score(model, X_te, unsupervised)
    t_inf = time.perf_counter() - t0

    tn, fp, fn, tp = confusion_matrix(y_te, y_hat, labels=[0, 1]).ravel()
    return {
        "model": name,
        "accuracy": accuracy_score(y_te, y_hat),
        "precision": precision_score(y_te, y_hat, zero_division=0),
        "recall": recall_score(y_te, y_hat, zero_division=0),
        "f1": f1_score(y_te, y_hat, zero_division=0),
        "roc_auc": roc_auc_score(y_te, score),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "fit_s": t_fit,
        "infer_ms_per_segment": 1000 * t_inf / len(X_te),
    }, model
