"""Evaluation helpers shared by the experiments: model selection, thresholds, confidence intervals.

Nothing here touches the test split except `report` and `bootstrap_ci`, which only score
predictions that were already fixed. Hyper-parameters and decision thresholds are chosen from
out-of-fold (OOF) scores on the training data.
"""
import itertools

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

SEED = 42


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Score threshold (predict 1 when score >= threshold) that maximises F1 on (y, score)."""
    prec, rec, thr = precision_recall_curve(y, score)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def bootstrap_ci(y: np.ndarray, y_hat: np.ndarray, metric=f1_score, n_boot: int = 2000,
                 level: float = 0.95, seed: int = SEED) -> tuple[float, float]:
    """Percentile bootstrap interval of `metric` over resampled test segments."""
    y, y_hat = np.asarray(y), np.asarray(y_hat)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    vals = np.array([metric(y[i], y_hat[i], zero_division=0) for i in idx])
    lo, hi = np.quantile(vals, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def report(y: np.ndarray, score: np.ndarray, threshold: float = 0.0, n_boot: int = 2000) -> dict:
    """Test metrics of fixed scores at a fixed threshold, with a bootstrap interval on F1."""
    y_hat = (np.asarray(score) >= threshold).astype(int)
    lo, hi = bootstrap_ci(y, y_hat, n_boot=n_boot)
    return {"precision": precision_score(y, y_hat, zero_division=0),
            "recall": recall_score(y, y_hat, zero_division=0),
            "f1": f1_score(y, y_hat, zero_division=0),
            "f1_ci_lo": lo, "f1_ci_hi": hi,
            "roc_auc": roc_auc_score(y, score),
            "pr_auc": average_precision_score(y, score)}


def oof_svc_scores(X_or_K: np.ndarray, y: np.ndarray, precomputed: bool, folds: int = 5,
                   seed: int = SEED, **svc_params) -> np.ndarray:
    """Out-of-fold SVC decision values. With `precomputed`, X_or_K is the (n, n) training kernel."""
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        m = SVC(kernel="precomputed" if precomputed else "rbf", class_weight="balanced", **svc_params)
        if precomputed:
            m.fit(X_or_K[np.ix_(tr, tr)], y[tr])
            oof[va] = m.decision_function(X_or_K[np.ix_(va, tr)])
        else:
            m.fit(X_or_K[tr], y[tr])
            oof[va] = m.decision_function(X_or_K[va])
    return oof


def select_svc(candidates: dict, y: np.ndarray, grid: dict, precomputed: bool, folds: int = 5,
               seed: int = SEED) -> dict:
    """Grid search by OOF PR-AUC.

    candidates: {key: X or K}; one entry per data-side setting (e.g. kernel bandwidth).
    grid:       SVC parameters, e.g. {"C": [...], "gamma": [...]}.
    Returns the best key and parameters, its OOF PR-AUC and the OOF F1-optimal threshold.
    """
    best = None
    names = list(grid)
    for key, data in candidates.items():
        for combo in itertools.product(*(grid[n] for n in names)):
            params = dict(zip(names, combo))
            oof = oof_svc_scores(data, y, precomputed, folds, seed, **params)
            ap = average_precision_score(y, oof)
            if best is None or ap > best["oof_pr_auc"] + 1e-12:
                best = {"key": key, "params": params, "oof_pr_auc": float(ap),
                        "threshold": best_f1_threshold(y, oof)}
    return best
