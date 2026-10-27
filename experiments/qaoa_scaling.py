"""QAOA ground-station scheduling: solution quality and cost vs problem size and depth.

Instances: windows of n consecutive passes from a 24 h geometric pass list (4 stations,
515 km SSO), anomaly at the window start, energy budget = 50% of total transmit-minutes.
Solvers: brute force (true optimum of the constrained problem), NumPyMinimumEigensolver on
the QUBO, greedy (value/cost), QAOA p = 1, 2, 3 (COBYLA, 4096 shots, statevector sampler).
"""
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dt.physics import SatellitePhysics  # noqa: E402
from dt.quantum_opt import (brute_force, build_problem, greedy, prob_of_optimum, qubo_stats,  # noqa: E402
                            solve_exact_qubo, solve_qaoa)

RES = ROOT / "results"
SIZES = (4, 6, 8, 10)
STARTS = (5, 9, 14)          # three different windows of the pass list per size
REPS = (1, 2, 3)
SHOTS = 4096               # solve_qaoa default


def main():
    passes = SatellitePhysics().ground_passes(hours=24)
    pd.DataFrame(passes).to_csv(RES / "qaoa_pass_list.csv", index=False)
    rows = []
    for n in SIZES:
        for s in STARTS:
            window = passes[s:s + n]
            budget = int(0.5 * sum(math.ceil(p["duration_s"] / 60) for p in window))
            prob = build_problem(window, budget, t_anomaly=window[0]["start_s"])
            t0 = time.perf_counter(); x_opt, f_opt = brute_force(prob); t_bf = time.perf_counter() - t0
            ex = solve_exact_qubo(prob)
            t0 = time.perf_counter(); _, f_gr = greedy(prob); t_gr = time.perf_counter() - t0
            base = {"n_passes": n, "instance": s, "conflicts": len(prob["conflicts"]), "budget": budget,
                    "optimum": f_opt, "bruteforce_s": t_bf, "exact_qubo_matches": bool(
                        abs(ex["objective"] - f_opt) < 1e-6), "exact_qubo_s": ex["seconds"],
                    "greedy_ratio": f_gr / f_opt, "greedy_s": t_gr}
            for p in REPS:
                st = qubo_stats(prob, p)
                q = solve_qaoa(prob, reps=p)
                row = {**base, "reps": p, **st, "qaoa_feasible": q["feasible"],
                       "qaoa_ratio": q["objective"] / f_opt, "p_optimal": prob_of_optimum(q["samples"], x_opt),
                       "qaoa_s": q["seconds"]}
                rows.append(row)
                print(f"n={n} inst={s} p={p} qubits={st['qubits']} ratio={row['qaoa_ratio']:.3f} "
                      f"P(opt)={row['p_optimal']:.4f} greedy={base['greedy_ratio']:.3f} t={q['seconds']:.1f}s",
                      flush=True)
    df = pd.DataFrame(rows)
    df.round(5).to_csv(RES / "qaoa_scaling.csv", index=False)
    summarise(df)


def summarise(df: pd.DataFrame):
    # The schedule depends only on the n pass variables (slack qubits are marginalised out of the
    # samples), so the uniform-random baseline is 1/2^n, not 1/2^qubits.
    df = df.assign(p_random=1 / 2.0 ** df.n_passes,
                   random_hit_prob=1 - (1 - 1 / 2.0 ** df.n_passes) ** SHOTS)
    df = df.assign(p_opt_vs_random=df.p_optimal / df.p_random)
    agg = df.groupby(["n_passes", "reps"]).agg(
        qubits=("qubits", "mean"), depth=("depth", "mean"), cx=("cx", "mean"),
        qaoa_ratio=("qaoa_ratio", "mean"), qaoa_ratio_min=("qaoa_ratio", "min"),
        p_optimal=("p_optimal", "mean"), p_random=("p_random", "mean"),
        p_opt_vs_random=("p_opt_vs_random", "mean"), random_hit_prob=("random_hit_prob", "mean"),
        greedy_ratio=("greedy_ratio", "mean"), qaoa_s=("qaoa_s", "mean"),
        exact_qubo_matches=("exact_qubo_matches", "all")).reset_index()
    agg.round(5).to_csv(RES / "qaoa_summary.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for p, g in agg.groupby("reps"):
        ax[0].plot(g.n_passes, g.qaoa_ratio, "o-", label=f"QAOA p={p}")
        ax[1].semilogy(g.n_passes, g.p_optimal, "o-", label=f"QAOA p={p}")
        ax[2].semilogy(g.n_passes, g.qaoa_s, "o-", label=f"QAOA p={p}")
    g1 = agg[agg.reps == 1]
    ax[0].plot(g1.n_passes, g1.greedy_ratio, "k--", label="greedy")
    ax[1].semilogy(g1.n_passes, g1.p_random, "k:", label="uniform random (1/2^n)")
    ax[0].set_ylabel("approximation ratio of best sample (mean of 3)")
    ax[1].set_ylabel("P(optimal schedule) per shot")
    ax[2].set_ylabel("QAOA wall time [s]")
    for a in ax:
        a.set_xlabel("passes n (decision variables)"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("QAOA downlink scheduling vs exact and greedy (statevector sampler, 4096 shots)")
    fig.tight_layout()
    fig.savefig(RES / "qaoa_scaling.png", dpi=150)
    first = df.drop_duplicates(["n_passes", "instance"])
    (RES / "qaoa_summary.json").write_text(json.dumps(
        {"instances": int(len(first)), "runs": len(df),
         "qaoa_found_optimum_runs": int((df.qaoa_ratio > 0.9999).sum()),
         "qaoa_infeasible_runs": int((~df.qaoa_feasible).sum()),
         "greedy_optimal_instances": int((first.greedy_ratio > 0.9999).sum()),
         "greedy_min_ratio": round(float(first.greedy_ratio.min()), 5),
         "exact_qubo_matches_bruteforce": bool(df.exact_qubo_matches.all()),
         "max_qubits": int(df.qubits.max()),
         "mean_p_opt_vs_random": round(float(df.p_opt_vs_random.mean()), 3),
         "random_sampling_hit_prob_at_max_n": round(float(df[df.n_passes == df.n_passes.max()].random_hit_prob.iloc[0]), 4),
         "shots": SHOTS}, indent=2))
    print(agg.to_string())


if __name__ == "__main__":
    if "--from-csv" in sys.argv:          # re-summarise saved runs without re-solving
        summarise(pd.read_csv(RES / "qaoa_scaling.csv"))
    else:
        main()
