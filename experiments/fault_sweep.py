"""Physics-path detection limits: which fault sizes the residual thresholds catch, and how fast.

`twin_replay.py` injects one fault of each type. This sweep varies the fault magnitude and
the sensor-noise seed, so the paper can state a detection floor instead of "caught every
fault". Needs no telemetry data: only the physics path of the twin is exercised.

For each fault type and magnitude (fault starts at orbit 4 of 10, 10 s step, 5 noise seeds):
  detection rate, detection delay, and the false-alarm rate before the fault starts.
A run counts as detected when any residual crosses its threshold after the fault starts.

Outputs: results/fault_sweep.csv, results/fault_sweep.json, results/fault_sweep.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dt.physics import Faults, SatellitePhysics  # noqa: E402
from dt.twin import PHYSICS_THRESHOLDS, SatelliteTwin  # noqa: E402

RES = ROOT / "results"
NOISE = {"solar_w": 0.5, "soc": 0.005, "temp_k": 0.5}   # same assumed 1-sigma noise as twin_replay.py
SEEDS = (0, 1, 2, 3, 4)
ONSET_ORBIT, N_ORBITS, DT_S = 4, 10, 10.0
SWEEP = {
    "panel_degradation": ("fraction of panel output lost", (0.05, 0.10, 0.20, 0.30, 0.50)),
    "battery_fade": ("fraction of battery capacity lost", (0.05, 0.10, 0.20, 0.40, 0.60)),
    "heater_stuck_on": ("extra heater power [W]", (0.5, 1.0, 2.0, 4.0, 8.0)),
}
# the physics path uses no anomaly model, so the twin gets an empty model bundle
NO_MODEL = {"model": None, "columns": [], "channel_stats": pd.DataFrame(columns=["mean", "std"])}


def run(sp: SatellitePhysics, faults: Faults, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Alert flag per time step for one noisy run through the twin's physics path."""
    rng = np.random.default_rng(seed)
    sim = sp.simulate(n_orbits=N_ORBITS, dt_s=DT_S, faults=faults)
    twin = SatelliteTwin(NO_MODEL, physics=sp)
    alerts = np.array([twin.update_physics(t, {k: sim[k][i] + rng.normal(0, NOISE[k]) for k in NOISE})["alert"]
                       for i, t in enumerate(sim["t_s"])])
    return sim["t_s"], alerts


def main():
    sp = SatellitePhysics()
    onset = ONSET_ORBIT * sp.orbit.period_s
    rows = []
    nominal_fa = [float(run(sp, Faults(), s)[1].mean()) for s in SEEDS]
    for fault, (_, magnitudes) in SWEEP.items():
        for mag in magnitudes:
            for seed in SEEDS:
                t, alerts = run(sp, Faults(**{fault: (onset, mag)}), seed)
                post = np.flatnonzero(alerts & (t >= onset))
                rows.append({"fault": fault, "magnitude": mag, "seed": seed,
                             "detected": bool(len(post)),
                             "delay_min": float((t[post[0]] - onset) / 60) if len(post) else np.nan,
                             "false_alarm_rate_before_onset": float(alerts[t < onset].mean()),
                             "alert_rate_after_onset": float(alerts[t >= onset].mean())})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(RES / "fault_sweep.csv", index=False)

    agg = df.groupby(["fault", "magnitude"]).agg(
        detection_rate=("detected", "mean"), delay_min_median=("delay_min", "median"),
        delay_min_max=("delay_min", "max"),
        false_alarm_rate=("false_alarm_rate_before_onset", "mean")).reset_index()
    floor = {f: (float(g[g.detection_rate == 1.0].magnitude.min()) if (g.detection_rate == 1.0).any() else None)
             for f, g in agg.groupby("fault")}
    summary = {"thresholds": PHYSICS_THRESHOLDS, "sensor_noise_sigma": NOISE, "noise_seeds": len(SEEDS),
               "orbits": N_ORBITS, "fault_onset_orbit": ONSET_ORBIT, "dt_s": DT_S,
               "runs": int(len(df)), "nominal_false_alarm_rate": round(float(np.mean(nominal_fa)), 5),
               "smallest_magnitude_always_detected": floor,
               "by_fault": {f: g.drop(columns="fault").round(3).to_dict(orient="records")
                            for f, g in agg.groupby("fault")}}
    (RES / "fault_sweep.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for a, (fault, (label, _)) in zip(ax, SWEEP.items()):
        g = agg[agg.fault == fault]
        a.plot(g.magnitude, g.detection_rate, "o-", color="C0")
        a.set_ylim(-0.05, 1.05); a.set_xlabel(label); a.set_title(fault.replace("_", " "), fontsize=10)
        a.set_ylabel("detection rate", color="C0"); a.grid(alpha=0.3)
        b = a.twinx()
        b.plot(g.magnitude, g.delay_min_median, "s--", color="C1")
        b.set_ylabel("median delay [min]", color="C1")
    fig.tight_layout()
    fig.savefig(RES / "fault_sweep.png", dpi=200)
    print(agg.round(3).to_string(index=False))
    print(json.dumps({k: summary[k] for k in ("nominal_false_alarm_rate", "smallest_magnitude_always_detected")}))


if __name__ == "__main__":
    main()
