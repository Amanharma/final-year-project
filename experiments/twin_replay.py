"""End-to-end twin experiments.

A. Telemetry path: replay the OPSSAT-AD *test* split sample-by-sample through SatelliteTwin.
   Checks that streaming predictions equal offline batch predictions, and measures
   throughput and per-segment scoring latency.
B. Physics path: feed noisy simulated telemetry (nominal, and with injected faults) through
   the twin and measure false-alarm rate and fault-detection delay.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dt.ingest import build_features, load_segments, stream  # noqa: E402
from dt.physics import Faults, SatellitePhysics  # noqa: E402
from dt.twin import SatelliteTwin, TwinConfig  # noqa: E402

RES = ROOT / "results"
NOISE = {"solar_w": 0.5, "soc": 0.005, "temp_k": 0.5}   # assumed sensor noise (1 sigma)


def telemetry_path(bundle):
    seg = load_segments()
    twin = SatelliteTwin(bundle, config=TwinConfig(db_path=str(RES / "twin_replay.db")))
    twin.db.execute("DELETE FROM segment_events")

    t0 = time.perf_counter()
    events = [e for s in stream(seg, split=0) if (e := twin.ingest(s)) is not None]
    wall = time.perf_counter() - t0
    twin.db.commit()

    # offline batch predictions on the same segments
    fs = build_features(seg)
    _, X_te, _, y_te = fs.split
    batch = dict(zip(X_te.index, bundle["model"].predict(X_te[bundle["columns"]])))
    live = {e["segment"]: e["predicted"] for e in events}
    agree = np.mean([live[s] == batch[s] for s in live])

    y_true = np.array([y_te[s] for s in live])
    y_pred = np.array([live[s] for s in live])
    lat = np.array([e["latency_ms"] for e in events])
    return {
        "test_segments_replayed": len(events),
        "samples_replayed": twin.state.samples_seen,
        "wall_s": round(wall, 3),
        "throughput_samples_per_s": round(twin.state.samples_seen / wall, 1),
        "segment_scoring_latency_ms": {"p50": round(float(np.percentile(lat, 50)), 3),
                                       "p95": round(float(np.percentile(lat, 95)), 3),
                                       "max": round(float(lat.max()), 3)},
        "stream_vs_batch_agreement": round(float(agree), 4),
        "stream_f1": round(f1_score(y_true, y_pred), 4),
        "stream_precision": round(precision_score(y_true, y_pred), 4),
        "stream_recall": round(recall_score(y_true, y_pred), 4),
        "alerts_raised": len(twin.state.alerts),
        "final_health": twin.state.health,
        "final_channel_risk": {k: round(v, 3) for k, v in twin.state.channel_risk.items()},
    }


def physics_path(bundle, rng):
    sp = SatellitePhysics()
    T = sp.orbit.period_s
    runs = {
        "nominal": Faults(),
        "panel_degradation_30pct": Faults(panel_degradation=(4 * T, 0.3)),
        "battery_fade_40pct": Faults(battery_fade=(4 * T, 0.4)),
        "heater_stuck_on_4W": Faults(heater_stuck_on=(4 * T, 4.0)),
    }
    out, traces = {}, {}
    for name, faults in runs.items():
        sim = sp.simulate(n_orbits=10, dt_s=10, faults=faults)
        twin = SatelliteTwin(bundle, physics=sp)
        alerts = np.zeros(len(sim["t_s"]), dtype=bool)
        for i, t in enumerate(sim["t_s"]):
            meas = {k: sim[k][i] + rng.normal(0, NOISE[k]) for k in NOISE}
            alerts[i] = twin.update_physics(t, meas)["alert"]
        onset = 4 * T if name != "nominal" else None
        pre = alerts if onset is None else alerts[sim["t_s"] < onset]
        res = {"false_alarm_rate": round(float(pre.mean()), 4)}
        if onset is not None:
            post = np.where(alerts & (sim["t_s"] >= onset))[0]
            res["detected"] = bool(len(post))
            res["detection_delay_min"] = round(float((sim["t_s"][post[0]] - onset) / 60), 2) if len(post) else None
            res["alert_rate_after_onset"] = round(float(alerts[sim["t_s"] >= onset].mean()), 4)
        out[name] = res
        traces[name] = (sim["t_s"], twin.history("physics_events"))

    # plot residuals of the battery-fade case
    t_s, h = traces["battery_fade_40pct"]
    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for a, var in zip(ax, ["solar_w", "soc", "temp_k"]):
        d = h[h.variable == var]
        a.plot(d.t_s / 3600, d.residual, lw=0.7)
        thr = {"solar_w": 3.0, "soc": 0.05, "temp_k": 3.0}[var]
        a.axhline(thr, color="r", ls="--", lw=0.8); a.axhline(-thr, color="r", ls="--", lw=0.8)
        a.axvline(4 * T / 3600, color="k", ls=":", lw=0.8)
        a.set_ylabel(f"residual {var}")
    ax[0].set_title("Twin physics residuals, battery capacity fade 40% at orbit 4 (dotted), thresholds dashed")
    ax[-1].set_xlabel("time [h]")
    fig.tight_layout()
    fig.savefig(RES / "twin_physics_residuals.png", dpi=150)
    return out


def main():
    rng = np.random.default_rng(42)
    bundle = joblib.load(RES / "models" / "live_classical.joblib")
    report = {"live_model": f"{bundle['name']} on '{bundle['features']}' features",
              "telemetry_path": telemetry_path(bundle),
              "physics_path": {"sensor_noise_sigma": NOISE, **physics_path(bundle, rng)}}
    (RES / "twin_replay.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
