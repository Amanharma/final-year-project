"""Latency benchmark against a running API (uvicorn api.main:app --port 8000).
Sequential requests from the same machine (loopback), so this measures service time,
not network latency. Writes results/api_latency.csv and results/api_integration_log.json."""
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

API = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
RES = Path(__file__).resolve().parents[1] / "results"
TS = [f"2022-01-01T00:00:{i:02d}Z" for i in range(60)]
VALUES = list(np.sin(np.linspace(0, 6, 60)) * 2e-5)


def wait_ready(c, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if c.get("/health").status_code == 200:
                return
        except httpx.TransportError:
            time.sleep(0.5)
    raise SystemExit("API not reachable")


def bench(c, name, n, fn):
    lat, errors = [], 0
    for _ in range(n):
        t0 = time.perf_counter()
        r = fn(c)
        lat.append(1000 * (time.perf_counter() - t0))
        errors += r.status_code != 200
    lat = np.array(lat)
    row = {"endpoint": name, "requests": n, "errors": errors, "p50_ms": np.percentile(lat, 50),
           "p95_ms": np.percentile(lat, 95), "p99_ms": np.percentile(lat, 99), "mean_ms": lat.mean(),
           "throughput_rps": n / (lat.sum() / 1000)}
    print({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()})
    return row


def main():
    with httpx.Client(base_url=API, timeout=120) as c:
        wait_ready(c)
        log = {"replay": c.post("/replay", params={"n_segments": 50}).json()["state"]}
        rows = [
            bench(c, "GET /twin/state", 1000, lambda c: c.get("/twin/state")),
            bench(c, "POST /anomaly (60 samples)", 1000,
                  lambda c: c.post("/anomaly", json={"channel": "CADC0872", "timestamps": TS, "values": VALUES})),
            bench(c, "POST /physics", 1000,
                  lambda c: c.post("/physics", json={"t_s": 1200, "solar_w": 30, "soc": 0.95, "temp_k": 270})),
            bench(c, "POST /schedule greedy (8 passes)", 200,
                  lambda c: c.post("/schedule", json={"n_passes": 8, "solver": "greedy"})),
            bench(c, "POST /schedule exact (8 passes)", 200,
                  lambda c: c.post("/schedule", json={"n_passes": 8, "solver": "exact"})),
            bench(c, "POST /schedule qaoa p=1 (6 passes)", 5,
                  lambda c: c.post("/schedule", json={"n_passes": 6, "solver": "qaoa"})),
        ]
        log["schedule_example"] = {s: c.post("/schedule", json={"n_passes": 6, "solver": s, "soc": 0.9,
                                                                 "anomaly_time_s": 30000}).json()
                                   for s in ("greedy", "exact", "qaoa")}
        log["final_state"] = c.get("/twin/state").json()
    pd.DataFrame(rows).round(3).to_csv(RES / "api_latency.csv", index=False)
    (RES / "api_integration_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
