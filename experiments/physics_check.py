"""Validate the physics twin against closed-form orbital mechanics and record the environment."""
import json
import os
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dt.physics import Faults, OrbitModel, SatellitePhysics  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
RES.mkdir(exist_ok=True)


def env_info():
    return {
        "os": platform.platform(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "qiskit", "qiskit-aer",
                                              "qiskit-machine-learning", "qiskit-optimization",
                                              "fastapi", "streamlit"]},
    }


def main():
    (RES / "env.json").write_text(json.dumps(env_info(), indent=2))

    rows = []
    for beta in (0, 20, 40, 60, 70, 75):
        o = OrbitModel(beta_deg=beta)
        sim = SatellitePhysics(orbit=o).simulate(n_orbits=5, dt_s=5)
        f_sim = 1 - sim["sunlit"].mean()
        f_ana = o.eclipse_fraction_analytic()
        rows.append({"beta_deg": beta, "eclipse_analytic": round(f_ana, 4), "eclipse_sim": round(f_sim, 4),
                     "abs_err": round(abs(f_sim - f_ana), 4)})

    sp = SatellitePhysics()
    t0 = time.perf_counter()
    nominal = sp.simulate(n_orbits=10, dt_s=10)
    elapsed = time.perf_counter() - t0
    T = sp.orbit.period_s
    faulty = sp.simulate(n_orbits=10, dt_s=10, faults=Faults(panel_degradation=(4 * T, 0.5),
                                                              heater_stuck_on=(6 * T, 4.0)))

    report = {
        "altitude_km": sp.orbit.altitude_m / 1e3,
        "period_min": round(T / 60, 3),
        "eclipse_validation": rows,
        "max_eclipse_abs_err": max(r["abs_err"] for r in rows),
        "sim_10_orbits_s": round(elapsed, 4),
        "sim_orbits_per_s": round(10 / elapsed, 1),
        "nominal_soc_min": round(float(nominal["soc"].min()), 4),
        "faulty_soc_min": round(float(faulty["soc"].min()), 4),
    }
    # thermal metrics over the last 3 orbits, after the start-up transient has settled
    late = nominal["t_s"] >= 7 * T
    for name, run in (("nominal", nominal), ("faulty", faulty)):
        tc = run["temp_k"][late] - 273.15
        report[f"{name}_temp_c_last3orbits"] = {"min": round(float(tc.min()), 2), "max": round(float(tc.max()), 2),
                                                "mean": round(float(tc.mean()), 2)}
    (RES / "physics_validation.json").write_text(json.dumps(report, indent=2))

    t_h = nominal["t_s"] / 3600
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t_h, nominal["solar_w"], label="nominal")
    ax[0].plot(t_h, faulty["solar_w"], "--", label="panel degradation @ orbit 4")
    ax[0].set_ylabel("solar power [W]")
    ax[1].plot(t_h, nominal["soc"])
    ax[1].plot(t_h, faulty["soc"], "--")
    ax[1].set_ylabel("battery SoC")
    ax[2].plot(t_h, nominal["temp_k"] - 273.15)
    ax[2].plot(t_h, faulty["temp_k"] - 273.15, "--", label="+ heater stuck on @ orbit 6")
    ax[2].set_ylabel("temperature [°C]")
    ax[2].set_xlabel("time [h]")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(); ax[2].legend()
    fig.suptitle("Physics twin: 515 km LEO CubeSat, 10 orbits, nominal vs injected faults")
    fig.tight_layout()
    fig.savefig(RES / "physics_twin.png", dpi=150)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
