"""Ingestion and twin tests on a small synthetic dataset with the OPSSAT-AD schema
(so they run without the raw data download)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from dt.ingest import PUBLISHED, build_features, channel_stats, segment_features, stream
from dt.twin import SatelliteTwin


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    rows, pub = [], []
    sid = 0
    for ch, scale in (("CADC0001", 1e-5), ("CADC0002", 1.0)):
        for k in range(12):
            sid += 1
            anomaly, train = int(k % 4 == 0), int(k < 8)
            n = 50
            v = scale * (rng.normal(0, 1, n) + (np.linspace(0, 8, n) if anomaly else 0))
            ts = pd.date_range("2022-01-01", periods=n, freq="5s") + pd.Timedelta(hours=sid)
            rows += [{"channel": ch, "timestamp": t, "value": x, "anomaly": anomaly,
                      "segment": sid, "train": train} for t, x in zip(ts, v)]
            pub.append({"segment": sid, "anomaly": anomaly, "train": train, "channel": ch,
                        **{c: float(rng.random() + 1e-3) for c in PUBLISHED}})
    return pd.DataFrame(rows), pd.DataFrame(pub).set_index("segment")


def test_channel_stats_use_training_only(synthetic):
    seg, _ = synthetic
    st = channel_stats(seg)
    tr = seg[(seg.train == 1) & (seg.channel == "CADC0002")]["value"]
    assert st.loc["CADC0002", "mean"] == pytest.approx(tr.mean())


def test_segment_features_deterministic_and_finite():
    rng = np.random.default_rng(1)
    z = rng.normal(size=100)
    dt = np.full(99, 5.0)
    a, b = segment_features(z, dt), segment_features(z * 1.0, dt)
    assert a == b
    assert all(np.isfinite(v) for v in a.values())


def test_build_features_shapes(synthetic):
    seg, pub = synthetic
    fs = build_features(seg, pub)
    assert len(fs.X) == 24 and not fs.X.isna().any().any()
    assert {"ch_CADC0001", "ch_CADC0002"} <= set(fs.X.columns)
    X_tr, X_te, y_tr, y_te = fs.split
    assert len(X_tr) == 16 and len(X_te) == 8


def test_stream_marks_segment_ends(synthetic):
    seg, _ = synthetic
    samples = list(stream(seg, split=0))
    assert len(samples) == (seg.train == 0).sum()
    assert sum(s["end_of_segment"] for s in samples) == 8
    ts = [s["timestamp"] for s in samples]
    assert ts == sorted(ts)


@pytest.fixture(scope="module")
def twin(synthetic):
    seg, pub = synthetic
    fs = build_features(seg, pub)
    cols = [c for c in fs.X.columns if c not in PUBLISHED]
    X_tr, _, y_tr, _ = fs.split
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_tr[cols], y_tr)
    return SatelliteTwin({"model": model, "columns": cols, "channel_stats": channel_stats(seg)})


def test_twin_stream_matches_batch(synthetic, twin):
    seg, pub = synthetic
    events = [e for s in stream(seg, split=0) if (e := twin.ingest(s))]
    fs = build_features(seg, pub)
    _, X_te, _, _ = fs.split
    batch = dict(zip(X_te.index, twin.model.predict(X_te[twin.columns])))
    assert len(events) == 8
    assert all(e["predicted"] == batch[e["segment"]] for e in events)
    assert len(twin.history()) == 8
    assert 0.0 <= twin.state.health <= 1.0


def test_physics_residual_zero_for_nominal_and_flags_fault(twin):
    exp = twin.expected(1200.0)
    assert twin.update_physics(1200.0, exp)["alert"] is False
    bad = dict(exp, temp_k=exp["temp_k"] + 10)
    out = twin.update_physics(1200.0, bad)
    assert out["alert"] is True and out["residual"]["temp_k"] == pytest.approx(10, abs=1e-6)
    assert twin.state.health == 0.0
