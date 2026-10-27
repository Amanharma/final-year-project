"""SatelliteTwin: the digital-twin core.

Two synchronisation paths:
  1. Telemetry path: streamed OPS-SAT samples are buffered per segment; when a segment
     closes, features are computed and scored by the anomaly model, and the channel's
     risk (EWMA of anomaly flags) and the overall health score are updated.
  2. Physics path: measured power/thermal values are compared with the physics model's
     nominal expectation at the same orbit time; residuals beyond a threshold raise alerts.
All events are persisted to SQLite.
"""
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from dt.ingest import segment_features
from dt.physics import SatellitePhysics

# residual thresholds for the physics path (absolute units of each variable)
PHYSICS_THRESHOLDS = {"solar_w": 3.0, "soc": 0.05, "temp_k": 3.0}


@dataclass
class TwinConfig:
    ewma_alpha: float = 0.3          # weight of the newest segment in channel risk
    alert_risk: float = 0.5          # channel risk that raises an alert
    db_path: str = ":memory:"


@dataclass
class TwinState:
    samples_seen: int = 0
    segments_scored: int = 0
    channel_risk: dict = field(default_factory=dict)
    alerts: list = field(default_factory=list)
    physics_residual: dict = field(default_factory=dict)
    last_update: str | None = None

    @property
    def health(self) -> float:
        """1 = fully healthy, 0 = at least one channel certainly anomalous."""
        worst = max(self.channel_risk.values(), default=0.0)
        phys = 1.0 if any(abs(r) > PHYSICS_THRESHOLDS[k]
                          for k, r in self.physics_residual.items()) else 0.0
        return round(1.0 - max(worst, phys), 4)

    def to_dict(self) -> dict:
        return {"samples_seen": self.samples_seen, "segments_scored": self.segments_scored,
                "health": self.health, "channel_risk": {k: round(v, 4) for k, v in self.channel_risk.items()},
                "active_alerts": self.alerts[-10:], "physics_residual": self.physics_residual,
                "last_update": self.last_update}


class SatelliteTwin:
    def __init__(self, model_bundle: dict, physics: SatellitePhysics | None = None,
                 config: TwinConfig | None = None):
        self.model = model_bundle["model"]
        self.columns = model_bundle["columns"]
        self.stats = model_bundle["channel_stats"]
        self.cfg = config or TwinConfig()
        self.physics = physics or SatellitePhysics()
        self.state = TwinState()
        self._buf = defaultdict(list)
        self._nominal = None
        self.db = sqlite3.connect(self.cfg.db_path, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS segment_events(
                ts TEXT, channel TEXT, segment INTEGER, n_samples INTEGER,
                score REAL, predicted INTEGER, channel_risk REAL, health REAL, latency_ms REAL);
            CREATE TABLE IF NOT EXISTS physics_events(
                t_s REAL, variable TEXT, measured REAL, expected REAL, residual REAL, alert INTEGER);
        """)

    # ---- telemetry path -------------------------------------------------
    def ingest(self, sample: dict) -> dict | None:
        """Feed one telemetry sample. Returns the segment event when a segment closes."""
        self.state.samples_seen += 1
        self._buf[sample["segment"]].append((sample["timestamp"], sample["value"]))
        self.state.last_update = str(sample["timestamp"])
        if sample["end_of_segment"]:
            return self._score_segment(sample["segment"], sample["channel"])
        return None

    def _features(self, channel: str, ts: list, values: np.ndarray) -> pd.DataFrame:
        z = (values - self.stats.loc[channel, "mean"]) / self.stats.loc[channel, "std"]
        dt_s = pd.Series(pd.to_datetime(ts)).diff().dt.total_seconds().dropna().to_numpy()
        f = segment_features(z, dt_s)
        for c in self.columns:
            if c.startswith("ch_"):
                f[c] = 1.0 if c == f"ch_{channel}" else 0.0
        return pd.DataFrame([f])[self.columns]

    def _score_segment(self, seg_id: int, channel: str) -> dict:
        ts, vals = zip(*self._buf.pop(seg_id))
        return self.score_segment(channel, list(ts), list(vals), seg_id)

    def score_segment(self, channel: str, ts: list, vals: list, seg_id: int = -1) -> dict:
        """Score one complete segment and update the twin state."""
        if channel not in self.stats.index:
            raise KeyError(f"unknown channel {channel}")
        t0 = time.perf_counter()
        X = self._features(channel, list(ts), np.asarray(vals, dtype=float))
        pred = int(self.model.predict(X)[0])
        score = float(self.model.decision_function(X)[0]) if hasattr(self.model, "decision_function") \
            else float(self.model.predict_proba(X)[0, 1])

        a = self.cfg.ewma_alpha
        risk = a * pred + (1 - a) * self.state.channel_risk.get(channel, 0.0)
        self.state.channel_risk[channel] = risk
        self.state.segments_scored += 1
        latency_ms = 1000 * (time.perf_counter() - t0)

        event = {"ts": str(ts[-1]), "channel": channel, "segment": int(seg_id), "n_samples": len(vals),
                 "score": round(score, 4), "predicted": pred, "channel_risk": round(risk, 4),
                 "health": self.state.health, "latency_ms": round(latency_ms, 3)}
        if pred and risk >= self.cfg.alert_risk:
            self.state.alerts.append({"ts": event["ts"], "channel": channel, "segment": int(seg_id),
                                      "risk": round(risk, 4)})
        self.db.execute("INSERT INTO segment_events VALUES (?,?,?,?,?,?,?,?,?)", tuple(event.values()))
        return event

    # ---- physics path ---------------------------------------------------
    def expected(self, t_s: float, dt_s: float = 10.0) -> dict:
        """Nominal physics expectation at orbit time t_s (cached nominal run)."""
        horizon = t_s + dt_s
        if self._nominal is None or self._nominal["t_s"][-1] < horizon:
            n_orbits = max(10, np.ceil(horizon / self.physics.orbit.period_s) + 1)
            self._nominal = self.physics.simulate(n_orbits=n_orbits, dt_s=dt_s)
        i = int(np.searchsorted(self._nominal["t_s"], t_s))
        return {k: float(self._nominal[k][i]) for k in PHYSICS_THRESHOLDS}

    def update_physics(self, t_s: float, measured: dict) -> dict:
        exp = self.expected(t_s)
        res, alerts = {}, []
        for k, v in measured.items():
            if k not in PHYSICS_THRESHOLDS:
                continue
            r = float(v) - exp[k]
            res[k] = round(r, 4)
            alert = abs(r) > PHYSICS_THRESHOLDS[k]
            alerts.append(alert)
            self.db.execute("INSERT INTO physics_events VALUES (?,?,?,?,?,?)",
                            (t_s, k, float(v), exp[k], r, int(alert)))
        self.state.physics_residual = res
        return {"t_s": t_s, "expected": exp, "residual": res, "alert": any(alerts)}

    # ---- persistence ------------------------------------------------------
    def history(self, table: str = "segment_events") -> pd.DataFrame:
        if table not in ("segment_events", "physics_events"):
            raise ValueError(table)
        return pd.read_sql(f"SELECT * FROM {table}", self.db)
