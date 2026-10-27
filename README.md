# Hybrid Quantum-Classical Digital Twin for Satellite Health Monitoring

B.Tech final-year capstone project. A digital twin of a Low-Earth-Orbit CubeSat that combines:

- a **physics model** (orbit/eclipse, power, thermal) with fault injection,
- **real ESA OPS-SAT telemetry** with labelled anomalies,
- **classical and quantum (Qiskit) anomaly classifiers**,
- **QAOA** for ground-station pass scheduling.

Project documentation is shared separately with the project guide.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
python data/download.py         # OPSSAT-AD from Zenodo (~19 MB)
```

## Reproduce all results

```bash
python experiments/run_all.py         # every experiment with fixed seeds, then pytest
```

| Script | Output |
|---|---|
| `experiments/eda.py` | dataset statistics, `results/eda_*` |
| `experiments/physics_check.py` | physics twin validation, `results/physics_*`, `results/env.json` |
| `experiments/classical_baselines.py` | IsolationForest / RandomForest / SVM, `results/classical_*` |
| `experiments/twin_replay.py` | streaming replay + fault injection, `results/twin_*` |
| `experiments/quantum_vs_classical.py` | QSVC / VQC vs SVM / RF on identical PCA features, `results/qml_*` |
| `experiments/qaoa_scaling.py` | QAOA vs exact vs greedy downlink scheduling, `results/qaoa_*` |

## Run the twin

```bash
uvicorn api.main:app --port 8000      # REST API, docs at http://127.0.0.1:8000/docs
streamlit run dashboard/app.py        # dashboard at http://localhost:8501
python experiments/api_benchmark.py   # latency benchmark against the running API
```

## Layout

```
dt/physics.py      orbit/eclipse, power, thermal models, fault injection, ground-station passes
dt/ingest.py       OPS-SAT loading, normalisation, features, stream replay
dt/twin.py         SatelliteTwin: health state, physics residuals, SQLite history
dt/classical.py    classical anomaly detectors
dt/quantum_ml.py   QSVC and VQC (Qiskit)
dt/quantum_opt.py  downlink scheduling QUBO, QAOA, exact and greedy solvers
api/main.py        FastAPI service
dashboard/app.py   Streamlit UI
tests/             pytest suite
```

## Dataset credit

OPSSAT-AD, KP Labs / ESA, MIT licence. Zenodo record 15108715.
Ruszczak et al., "The OPS-SAT benchmark for detecting anomalies in satellite telemetry", arXiv:2407.04730.
