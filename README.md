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

## Reproduce current results

```bash
python experiments/eda.py             # dataset statistics -> results/eda_*
python experiments/physics_check.py   # physics twin validation -> results/physics_*
```

## Dataset credit

OPSSAT-AD, KP Labs / ESA, MIT licence. Zenodo record 15108715.
Ruszczak et al., "The OPS-SAT benchmark for detecting anomalies in satellite telemetry", arXiv:2407.04730.
