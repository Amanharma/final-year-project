"""Quantum anomaly classifiers (Qiskit): quantum-kernel SVM (QSVC) and variational classifier (VQC).

Pipeline (fitted on the training subsample only):
  StandardScaler -> PCA(d) -> MinMaxScaler to [0, pi]   (one PCA component per qubit)
QSVC: fidelity kernel K(x, x') = |<phi(x)|phi(x')>|^2 with a ZZ feature map, exact statevector.
VQC : ZZ feature map + RealAmplitudes ansatz, COBYLA, StatevectorSampler (finite shots).
"""
import time

import numpy as np
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import QSVC, VQC
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_machine_learning.optimizers import COBYLA
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SEED = 42


def balanced_subsample(X, y, n: int, seed: int = SEED):
    """n samples with equal class counts (or all minority samples if fewer)."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in (0, 1):
        pool = np.where(y == c)[0]
        idx += list(rng.choice(pool, size=min(n // 2, len(pool)), replace=False))
    idx = rng.permutation(idx)
    return X[idx], y[idx]


def make_reducer(d: int):
    return make_pipeline(StandardScaler(), PCA(n_components=d, random_state=SEED),
                         MinMaxScaler(feature_range=(0, np.pi), clip=True))


def feature_map(d: int, reps: int = 2):
    return zz_feature_map(feature_dimension=d, reps=reps, entanglement="linear")


def make_qsvc(d: int, C: float = 10.0) -> QSVC:
    kernel = FidelityStatevectorKernel(feature_map=feature_map(d))
    return QSVC(quantum_kernel=kernel, C=C, class_weight="balanced")


def make_vqc(d: int, maxiter: int = 100, shots: int = 1024, history: list | None = None) -> VQC:
    def cb(_weights, loss):
        if history is not None:
            history.append(float(loss))

    return VQC(feature_map=feature_map(d), ansatz=real_amplitudes(d, reps=3),
               optimizer=COBYLA(maxiter=maxiter),
               sampler=StatevectorSampler(default_shots=shots, seed=SEED), callback=cb)


def circuit_stats(d: int) -> dict:
    fm = feature_map(d)
    ans = real_amplitudes(d, reps=3)
    full = fm.compose(ans)
    return {"qubits": d,
            "feature_map_depth": fm.decompose().depth(),
            "vqc_circuit_depth": full.decompose().depth(),
            "vqc_trainable_params": ans.num_parameters,
            "cx_count_vqc": full.decompose().count_ops().get("cx", 0)}


def kernel_matrix_time(d: int, X: np.ndarray) -> float:
    k = FidelityStatevectorKernel(feature_map=feature_map(d))
    t0 = time.perf_counter()
    k.evaluate(X)
    return time.perf_counter() - t0
