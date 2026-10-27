"""Ground-station downlink scheduling as a QUBO, solved with QAOA (Qiskit) and classical baselines.

Decision x_p = 1 -> downlink on pass p.
  maximise   sum_p w_p * v_p * x_p
  subject to x_p * x_q = 0                for overlapping passes (single transmitter)
             sum_p c_p * x_p <= B          (battery energy budget, integer tx-minutes)
v_p : data volume [MB] = rate * duration / 8
w_p : priority, 1 + boost * exp(-(start_p - t_anomaly) / tau) for passes after an anomaly
c_p : transmit minutes (ceil), B from the twin's battery state of charge
"""
import itertools
import math
import time

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import StatevectorSampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.optimizers import COBYLA

TX_POWER_W = 10.0         # S-band transmitter draw (assumed CubeSat value)
SEED = 42


def overlaps(a: dict, b: dict) -> bool:
    return a["start_s"] <= b["end_s"] and b["start_s"] <= a["end_s"]


def energy_budget_units(soc: float, capacity_wh: float = 40.0, soc_floor: float = 0.7) -> int:
    """Transmit-minutes affordable without taking the battery below soc_floor (eclipse reserve)."""
    wh = max(0.0, soc - soc_floor) * capacity_wh
    return int(wh / (TX_POWER_W / 60.0))


def build_problem(passes: list[dict], budget: int, t_anomaly: float | None = None,
                  boost: float = 2.0, tau_s: float = 6 * 3600) -> dict:
    v = np.array([p["rate_mbps"] * p["duration_s"] / 8 for p in passes])
    w = np.ones(len(passes))
    if t_anomaly is not None:
        dt = np.array([p["start_s"] - t_anomaly for p in passes])
        w = np.where(dt >= 0, 1 + boost * np.exp(-dt / tau_s), 1.0)
    value = np.round(w * v, 2)
    cost = np.array([math.ceil(p["duration_s"] / 60) for p in passes], dtype=int)
    conflicts = [(i, j) for i, j in itertools.combinations(range(len(passes)), 2)
                 if overlaps(passes[i], passes[j])]
    return {"passes": passes, "value": value, "cost": cost, "budget": int(budget), "conflicts": conflicts}


def to_quadratic_program(prob: dict) -> QuadraticProgram:
    n = len(prob["value"])
    qp = QuadraticProgram("downlink_schedule")
    for i in range(n):
        qp.binary_var(f"x{i}")
    penalty = float(prob["value"].sum()) + 1.0
    qp.maximize(linear={f"x{i}": float(prob["value"][i]) for i in range(n)},
                quadratic={(f"x{i}", f"x{j}"): -penalty for i, j in prob["conflicts"]})
    qp.linear_constraint({f"x{i}": int(prob["cost"][i]) for i in range(n)}, "<=", prob["budget"], "energy")
    return qp


def objective(prob: dict, x) -> float:
    return float(np.dot(prob["value"], x))


def feasible(prob: dict, x) -> bool:
    x = np.asarray(x).round().astype(int)
    return (np.dot(prob["cost"], x) <= prob["budget"]
            and all(not (x[i] and x[j]) for i, j in prob["conflicts"]))


def brute_force(prob: dict) -> tuple[np.ndarray, float]:
    best_x, best = np.zeros(len(prob["value"]), int), 0.0
    for bits in itertools.product((0, 1), repeat=len(prob["value"])):
        x = np.array(bits)
        if feasible(prob, x) and objective(prob, x) > best:
            best_x, best = x, objective(prob, x)
    return best_x, best


def greedy(prob: dict) -> tuple[np.ndarray, float]:
    order = np.argsort(-prob["value"] / prob["cost"])
    x = np.zeros(len(order), int)
    for i in order:
        x[i] = 1
        if not feasible(prob, x):
            x[i] = 0
    return x, objective(prob, x)


def qubo_stats(prob: dict, reps: int) -> dict:
    qubo = QuadraticProgramToQubo().convert(to_quadratic_program(prob))
    op, _ = qubo.to_ising()
    circ = transpile(qaoa_ansatz(op, reps=reps).decompose(),
                     basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=SEED)
    return {"qubits": qubo.get_num_vars(), "slack_qubits": qubo.get_num_vars() - len(prob["value"]),
            "depth": circ.depth(), "cx": circ.count_ops().get("cx", 0)}


def solve_qaoa(prob: dict, reps: int = 1, maxiter: int = 200, shots: int = 4096, seed: int = SEED) -> dict:
    qp = to_quadratic_program(prob)
    rng = np.random.default_rng(seed)
    qaoa = QAOA(StatevectorSampler(default_shots=shots, seed=seed), COBYLA(maxiter=maxiter), reps=reps,
                initial_point=rng.uniform(0, np.pi, 2 * reps))
    t0 = time.perf_counter()
    res = MinimumEigenOptimizer(qaoa).solve(qp)
    elapsed = time.perf_counter() - t0
    x = np.asarray(res.x).round().astype(int)
    return {"x": x, "objective": objective(prob, x) if feasible(prob, x) else 0.0,
            "feasible": feasible(prob, x), "samples": res.samples, "seconds": elapsed}


def solve_exact_qubo(prob: dict) -> dict:
    t0 = time.perf_counter()
    res = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(to_quadratic_program(prob))
    x = np.asarray(res.x).round().astype(int)
    return {"x": x, "objective": objective(prob, x), "seconds": time.perf_counter() - t0}


def prob_of_optimum(samples, x_opt: np.ndarray) -> float:
    return float(sum(s.probability for s in samples
                     if np.array_equal(np.asarray(s.x).round().astype(int), x_opt)))
