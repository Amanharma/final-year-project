"""Streamlit dashboard of the satellite digital twin. Talks to the FastAPI service over HTTP.

Run:  uvicorn api.main:app --port 8000
      streamlit run dashboard/app.py
"""
import os

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("TWIN_API", "http://127.0.0.1:8000")

st.set_page_config(page_title="Satellite Digital Twin", layout="wide")
st.title("Hybrid Quantum-Classical Digital Twin: LEO CubeSat")


def call(method, path, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=120, **kw)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API error on {path}: {e}")
        st.stop()


health = call("GET", "/health")
st.caption(f"API {API}: {health['status']}")

tab_twin, tab_phys, tab_sched = st.tabs(["Telemetry health", "Physics twin", "Downlink schedule"])

with tab_twin:
    st.subheader("Live OPS-SAT telemetry replay")
    n = st.slider("segments to replay", 5, 100, 20, step=5)
    if st.button("Replay next segments"):
        out = call("POST", "/replay", params={"n_segments": n})
        st.session_state["events"] = st.session_state.get("events", []) + out["events"]
    state = call("GET", "/twin/state")
    c1, c2, c3 = st.columns(3)
    c1.metric("Health score", f"{state['health']:.2f}")
    c2.metric("Segments scored", state["segments_scored"])
    c3.metric("Samples ingested", state["samples_seen"])
    if state["channel_risk"]:
        st.bar_chart(pd.Series(state["channel_risk"], name="risk").sort_index())
    ev = pd.DataFrame(st.session_state.get("events", []))
    if not ev.empty:
        st.write("Recent segment verdicts")
        st.dataframe(ev.tail(25)[["ts", "channel", "segment", "n_samples", "predicted", "score",
                                  "channel_risk", "latency_ms"]], use_container_width=True)
    if state["active_alerts"]:
        st.warning(f"{len(state['active_alerts'])} recent alerts")
        st.table(pd.DataFrame(state["active_alerts"]))

with tab_phys:
    st.subheader("Orbit / power / thermal model with fault injection")
    c = st.columns(5)
    n_orb = c[0].number_input("orbits", 1, 20, 6)
    panel = c[1].slider("panel loss", 0.0, 0.9, 0.0, 0.1)
    fade = c[2].slider("battery fade", 0.0, 0.9, 0.0, 0.1)
    heater = c[3].slider("heater stuck [W]", 0.0, 8.0, 0.0, 1.0)
    onset = c[4].number_input("fault at orbit", 0.0, 20.0, 2.0)
    sim = call("GET", "/simulate", params={"n_orbits": n_orb, "dt_s": 30, "panel_degradation": panel,
                                           "battery_fade": fade, "heater_w": heater, "fault_orbit": onset})
    nominal = call("GET", "/simulate", params={"n_orbits": n_orb, "dt_s": 30})
    st.caption(f"Orbital period {sim['period_min']:.2f} min, eclipse fraction {sim['eclipse_fraction']:.3f}")
    t_h = pd.Index([t / 3600 for t in sim["t_s"]], name="hours")
    for key, label in (("solar_w", "Solar power [W]"), ("soc", "Battery SoC"), ("temp_k", "Temperature [K]")):
        st.write(label)
        st.line_chart(pd.DataFrame({"twin (expected)": nominal[key], "measured (with fault)": sim[key]},
                                   index=t_h))

with tab_sched:
    st.subheader("Ground-station downlink scheduling (QUBO)")
    c = st.columns(5)
    soc = c[0].slider("battery SoC", 0.5, 1.0, 0.8, 0.05)
    n_p = c[1].slider("passes in window", 3, 12, 6)
    start = c[2].number_input("first pass index", 0, 20, 5)
    solver = c[3].selectbox("solver", ["greedy", "exact", "qaoa"])
    reps = c[4].selectbox("QAOA depth p", [1, 2, 3])
    anomaly = st.checkbox("anomaly detected at window start (raises priority of early passes)", True)
    if st.button("Plan downlinks"):
        body = {"soc": soc, "n_passes": n_p, "start_index": start, "solver": solver, "qaoa_reps": reps}
        if anomaly:
            passes = call("POST", "/schedule", json={**body, "solver": "greedy"})["passes"]
            body["anomaly_time_s"] = passes[0]["start_s"]
        res = call("POST", "/schedule", json=body)
        m = st.columns(4)
        m[0].metric("Weighted MB", res["objective_mb_weighted"])
        m[1].metric("Tx minutes", f"{res['tx_minutes_used']} / {res['budget_tx_minutes']}")
        m[2].metric("Feasible", "yes" if res["feasible"] else "no")
        m[3].metric("Conflicting pairs", len(res["conflicts"]))
        df = pd.DataFrame(res["passes"])
        df["start_h"] = (df.start_s / 3600).round(2)
        st.dataframe(df[["station", "start_h", "duration_s", "max_elev_deg", "rate_mbps", "value",
                         "tx_minutes", "selected"]], use_container_width=True)
