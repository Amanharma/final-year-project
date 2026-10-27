import numpy as np
import pytest
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

from dt.physics import SatellitePhysics
from dt.quantum_ml import balanced_subsample, feature_map, make_qsvc, make_reducer
from dt.quantum_opt import (brute_force, build_problem, energy_budget_units, feasible, greedy,
                            objective, qubo_stats, solve_exact_qubo, solve_qaoa)


def toy_passes():
    # p0 overlaps p1; p2, p3 independent
    mk = lambda s, e, r: {"station": "X", "start_s": s, "end_s": e, "duration_s": e - s, "rate_mbps": r}
    return [mk(0, 300, 2.0), mk(200, 500, 1.0), mk(1000, 1240, 1.0), mk(2000, 2120, 1.0)]


@pytest.fixture
def prob():
    return build_problem(toy_passes(), budget=8)


def test_conflicts_and_costs(prob):
    assert prob["conflicts"] == [(0, 1)]
    assert list(prob["cost"]) == [5, 5, 4, 2]


def test_feasibility_rules(prob):
    assert not feasible(prob, [1, 1, 0, 0])      # overlap
    assert not feasible(prob, [1, 0, 1, 0])      # 9 tx-minutes > budget 8
    assert feasible(prob, [1, 0, 0, 1])


def test_exact_qubo_equals_brute_force(prob):
    x_bf, f_bf = brute_force(prob)
    ex = solve_exact_qubo(prob)
    assert ex["objective"] == pytest.approx(f_bf)
    assert feasible(prob, ex["x"])


def test_greedy_is_feasible_and_bounded(prob):
    x, f = greedy(prob)
    assert feasible(prob, x) and f <= brute_force(prob)[1] + 1e-9


def test_anomaly_priority_boosts_later_passes():
    p = toy_passes()
    base = build_problem(p, 10)["value"]
    boosted = build_problem(p, 10, t_anomaly=900)["value"]
    assert np.all(boosted[2:] > base[2:]) and np.allclose(boosted[:2], base[:2])


def test_energy_budget_from_soc():
    assert energy_budget_units(0.7) == 0
    assert energy_budget_units(0.9) > energy_budget_units(0.8) > 0


def test_qaoa_returns_feasible_solution_on_toy(prob):
    r = solve_qaoa(prob, reps=1, maxiter=60)
    assert r["feasible"] and r["objective"] <= brute_force(prob)[1] + 1e-9
    assert qubo_stats(prob, 1)["qubits"] >= 4


def test_geometric_passes_are_plausible():
    ps = SatellitePhysics().ground_passes(hours=24)
    assert 10 < len(ps) < 60
    assert all(0 < p["duration_s"] <= 15 * 60 for p in ps)
    assert all(p["max_elev_deg"] >= 10 for p in ps)


def test_quantum_kernel_is_valid():
    X = np.random.default_rng(0).uniform(0, np.pi, (12, 3))
    K = FidelityStatevectorKernel(feature_map=feature_map(3)).evaluate(X)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-9)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-9


def test_qsvc_learns_separable_toy():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(-2, 0.3, (20, 4)), rng.normal(2, 0.3, (20, 4))])
    y = np.array([0] * 20 + [1] * 20)
    red = make_reducer(2).fit(X)
    model = make_qsvc(2).fit(red.transform(X), y)
    assert (model.predict(red.transform(X)) == y).mean() >= 0.95


def test_balanced_subsample():
    X = np.arange(100).reshape(-1, 1)
    y = np.array([1] * 10 + [0] * 90)
    Xs, ys = balanced_subsample(X, y, 40)
    assert ys.sum() == 10 and (ys == 0).sum() == 20
