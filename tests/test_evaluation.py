"""Tests for the evaluation helpers and the additions used by the tuned comparison."""
import numpy as np
import pandas as pd
import pytest
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from sklearn.metrics import f1_score

from dt.evaluation import best_f1_threshold, bootstrap_ci, oof_svc_scores, report, select_svc
from dt.physics import Faults, SatellitePhysics
from dt.quantum_ml import feature_map, statevector_kernel
from dt.quantum_opt import brute_force, build_problem, conflict_penalty, feasible, random_search
from dt.twin import SatelliteTwin


@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (120, 3)), rng.normal(2.5, 1, (40, 3))])
    y = np.array([0] * 120 + [1] * 40)
    return X, y


def test_best_f1_threshold_beats_arbitrary_threshold():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 0])
    s = np.array([0.1, 0.2, 0.3, 0.35, 0.6, 0.7, 0.8, 0.9])
    thr = best_f1_threshold(y, s)
    assert f1_score(y, s >= thr) >= max(f1_score(y, s >= t) for t in (0.0, 0.5, 0.85))


def test_bootstrap_ci_brackets_point_estimate_and_is_seeded(blobs):
    X, y = blobs
    y_hat = (X.sum(axis=1) > 3.5).astype(int)
    lo, hi = bootstrap_ci(y, y_hat, n_boot=300)
    assert 0.0 <= lo <= f1_score(y, y_hat) <= hi <= 1.0
    assert (lo, hi) == bootstrap_ci(y, y_hat, n_boot=300)


def test_report_keys_and_ranges(blobs):
    X, y = blobs
    r = report(y, X.sum(axis=1), threshold=3.5, n_boot=100)
    assert {"precision", "recall", "f1", "f1_ci_lo", "f1_ci_hi", "roc_auc", "pr_auc"} <= set(r)
    assert all(0.0 <= v <= 1.0 for v in r.values())


def test_oof_scores_cover_every_sample_and_never_see_their_own_label(blobs):
    X, y = blobs
    oof = oof_svc_scores(X, y, precomputed=False, C=1.0, gamma="scale")
    assert oof.shape == y.shape and np.all(oof != 0)
    # a kernel that memorises (identity) gives no out-of-fold signal: every score is the bias only
    K = np.eye(len(y))
    flat = oof_svc_scores(K, y, precomputed=True, C=1.0)
    assert np.ptp(flat) < 1e-6


def test_select_svc_prefers_the_informative_candidate(blobs):
    X, y = blobs
    noise = np.random.default_rng(1).normal(size=X.shape)
    best = select_svc({"noise": noise, "signal": X}, y, {"C": [1.0, 10.0]}, precomputed=False)
    assert best["key"] == "signal" and best["oof_pr_auc"] > 0.8 and best["params"]["C"] in (1.0, 10.0)


def test_statevector_kernel_matches_qiskit_kernel():
    rng = np.random.default_rng(0)
    A, B = rng.uniform(0, np.pi, (10, 3)), rng.uniform(0, np.pi, (4, 3))
    ref = FidelityStatevectorKernel(feature_map=feature_map(3))
    np.testing.assert_allclose(statevector_kernel(3, A), ref.evaluate(A), atol=1e-10)
    np.testing.assert_allclose(statevector_kernel(3, A, B), ref.evaluate(A, B), atol=1e-10)


def test_random_search_is_feasible_and_penalty_dominates():
    mk = lambda s, e, r: {"station": "X", "start_s": s, "end_s": e, "duration_s": e - s, "rate_mbps": r}
    prob = build_problem([mk(0, 300, 2.0), mk(200, 500, 1.0), mk(1000, 1240, 1.0), mk(2000, 2120, 1.0)], 8)
    x, f = random_search(prob, shots=256)
    assert feasible(prob, x) and f == pytest.approx(brute_force(prob)[1])   # 16 schedules, 256 draws
    assert conflict_penalty(prob) > prob["value"].sum()


def test_physics_path_runs_without_an_anomaly_model():
    sp = SatellitePhysics()
    twin = SatelliteTwin({"model": None, "columns": [], "channel_stats": pd.DataFrame(columns=["mean", "std"])},
                         physics=sp)
    onset = sp.orbit.period_s
    sim = sp.simulate(n_orbits=2, dt_s=60, faults=Faults(panel_degradation=(onset, 0.5)))
    alerts = np.array([twin.update_physics(t, {"solar_w": sim["solar_w"][i], "soc": sim["soc"][i],
                                               "temp_k": sim["temp_k"][i]})["alert"]
                       for i, t in enumerate(sim["t_s"])])
    assert not alerts[sim["t_s"] < onset].any() and alerts[sim["t_s"] >= onset].any()
