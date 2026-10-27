"""API contract tests. Need results/models/live_classical.joblib and the OPS-SAT data."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "results/models/live_classical.joblib").exists()
    or not (ROOT / "data/raw/opssat/segments.csv").exists(),
    reason="run data/download.py and experiments/classical_baselines.py first")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_replay_updates_state(client):
    r = client.post("/replay", params={"n_segments": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body["events"]) == 5 and body["state"]["segments_scored"] >= 5
    s = client.get("/twin/state").json()
    assert 0 <= s["health"] <= 1


def test_anomaly_endpoint_and_validation(client):
    ts = [f"2022-01-01T00:00:{i:02d}Z" for i in range(30)]
    ok = client.post("/anomaly", json={"channel": "CADC0872", "timestamps": ts, "values": [1e-5] * 30})
    assert ok.status_code == 200 and ok.json()["predicted"] in (0, 1)
    bad = client.post("/anomaly", json={"channel": "NOPE", "timestamps": ts, "values": [0.0] * 30})
    assert bad.status_code == 404
    short = client.post("/anomaly", json={"channel": "CADC0872", "timestamps": ts[:2], "values": [0, 0]})
    assert short.status_code == 422


def test_physics_endpoint_flags_fault(client):
    sim = client.get("/simulate", params={"n_orbits": 1, "dt_s": 60}).json()
    i = 20
    nominal = {"t_s": sim["t_s"][i], "solar_w": sim["solar_w"][i], "soc": sim["soc"][i]}
    assert client.post("/physics", json=nominal).json()["alert"] is False
    hot = dict(nominal, temp_k=400.0)
    assert client.post("/physics", json=hot).json()["alert"] is True


def test_simulate_validation(client):
    assert client.get("/simulate", params={"n_orbits": 100}).status_code == 422


@pytest.mark.parametrize("solver", ["greedy", "exact", "qaoa"])
def test_schedule_solvers_feasible(client, solver):
    r = client.post("/schedule", json={"soc": 0.9, "n_passes": 5, "solver": solver})
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] and body["tx_minutes_used"] <= body["budget_tx_minutes"]


def test_schedule_low_battery_selects_nothing(client):
    body = client.post("/schedule", json={"soc": 0.65, "n_passes": 5}).json()
    assert body["budget_tx_minutes"] == 0 and not any(p["selected"] for p in body["passes"])


def test_schedule_qaoa_size_limit(client):
    assert client.post("/schedule", json={"n_passes": 12, "solver": "qaoa"}).status_code == 422
