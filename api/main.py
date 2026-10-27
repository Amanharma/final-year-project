"""REST API of the satellite digital twin.

Run:  uvicorn api.main:app --port 8000      (docs at http://127.0.0.1:8000/docs)
Requires results/models/live_classical.joblib (python experiments/classical_baselines.py).
"""
import threading
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dt.ingest import load_segments, stream
from dt.physics import Faults, SatellitePhysics
from dt.quantum_opt import (brute_force, build_problem, energy_budget_units, feasible, greedy,
                            objective, solve_qaoa)
from dt.twin import SatelliteTwin

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "results" / "models" / "live_classical.joblib"
QAOA_MAX_PASSES = 8           # keeps an interactive QAOA call to a few seconds

app = FastAPI(title="Satellite Digital Twin API", version="1.0")
physics = SatellitePhysics()
_lock = threading.Lock()
_twin: SatelliteTwin | None = None
_replay = {"iter": None}


def twin() -> SatelliteTwin:
    global _twin
    if _twin is None:
        if not MODEL.exists():
            raise HTTPException(503, "model missing: run experiments/classical_baselines.py")
        _twin = SatelliteTwin(joblib.load(MODEL), physics=physics)
    return _twin


class Sample(BaseModel):
    timestamp: str
    channel: str
    value: float
    segment: int
    end_of_segment: bool = False


class Segment(BaseModel):
    channel: str
    timestamps: list[str] = Field(min_length=3)
    values: list[float] = Field(min_length=3)


class PhysicsReading(BaseModel):
    t_s: float = Field(ge=0)
    solar_w: float | None = None
    soc: float | None = None
    temp_k: float | None = None


class ScheduleRequest(BaseModel):
    soc: float = Field(0.9, ge=0, le=1)
    anomaly_time_s: float | None = None
    start_index: int = Field(5, ge=0)
    n_passes: int = Field(6, ge=1, le=20)
    solver: Literal["greedy", "exact", "qaoa"] = "greedy"
    qaoa_reps: int = Field(1, ge=1, le=3)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _twin is not None or MODEL.exists()}


@app.get("/twin/state")
def twin_state():
    return twin().state.to_dict()


@app.post("/telemetry")
def telemetry(s: Sample):
    with _lock:
        return {"event": twin().ingest(s.model_dump())}


@app.post("/anomaly")
def anomaly(seg: Segment):
    if len(seg.timestamps) != len(seg.values):
        raise HTTPException(422, "timestamps and values differ in length")
    with _lock:
        try:
            return twin().score_segment(seg.channel, seg.timestamps, seg.values)
        except KeyError as e:
            raise HTTPException(404, str(e))


@app.post("/replay")
def replay(n_segments: int = 20):
    """Server-side replay of the next n test segments of OPS-SAT telemetry through the twin."""
    with _lock:
        if _replay["iter"] is None:
            _replay["iter"] = stream(load_segments(), split=0)
        events = []
        for s in _replay["iter"]:
            e = twin().ingest(s)
            if e:
                events.append(e)
                if len(events) >= n_segments:
                    break
        return {"events": events, "state": twin().state.to_dict()}


@app.post("/physics")
def physics_update(r: PhysicsReading):
    meas = {k: v for k, v in r.model_dump().items() if k != "t_s" and v is not None}
    with _lock:
        return twin().update_physics(r.t_s, meas)


@app.get("/simulate")
def simulate(n_orbits: float = 3, dt_s: float = 30, panel_degradation: float = 0.0,
             battery_fade: float = 0.0, heater_w: float = 0.0, fault_orbit: float = 1.0):
    if not (0 < n_orbits <= 30 and 1 <= dt_s <= 600):
        raise HTTPException(422, "n_orbits in (0, 30], dt_s in [1, 600]")
    T = physics.orbit.period_s
    onset = fault_orbit * T
    f = Faults(panel_degradation=(onset, panel_degradation) if panel_degradation else None,
               battery_fade=(onset, battery_fade) if battery_fade else None,
               heater_stuck_on=(onset, heater_w) if heater_w else None)
    sim = physics.simulate(n_orbits=n_orbits, dt_s=dt_s, faults=f)
    return {"period_min": T / 60, "eclipse_fraction": physics.orbit.eclipse_fraction_analytic(),
            **{k: np.round(v, 4).tolist() for k, v in sim.items()}}


@app.post("/schedule")
def schedule(req: ScheduleRequest):
    passes = physics.ground_passes(hours=24)
    window = passes[req.start_index:req.start_index + req.n_passes]
    if not window:
        raise HTTPException(422, f"start_index beyond {len(passes)} passes")
    if req.solver == "qaoa" and len(window) > QAOA_MAX_PASSES:
        raise HTTPException(422, f"QAOA limited to {QAOA_MAX_PASSES} passes in the API")
    budget = energy_budget_units(req.soc)
    prob = build_problem(window, budget, t_anomaly=req.anomaly_time_s)
    if req.solver == "greedy":
        x, _ = greedy(prob)
    elif req.solver == "exact":
        x, _ = brute_force(prob)
    else:
        x = solve_qaoa(prob, reps=req.qaoa_reps)["x"]
    return {"solver": req.solver, "budget_tx_minutes": budget, "feasible": bool(feasible(prob, x)),
            "objective_mb_weighted": round(objective(prob, x), 2),
            "tx_minutes_used": int(np.dot(prob["cost"], x)),
            "passes": [{**p, "selected": bool(x[i]), "value": float(prob["value"][i]),
                        "tx_minutes": int(prob["cost"][i])} for i, p in enumerate(window)],
            "conflicts": prob["conflicts"]}
