# Hybrid Quantum-Classical Digital Twin for Satellite Health Monitoring and Ground-Station Scheduling

[![tests](https://github.com/Amanharma/final-year-project/actions/workflows/tests.yml/badge.svg)](https://github.com/Amanharma/final-year-project/actions/workflows/tests.yml)

A digital-twin testbed for a CubeSat in low Earth orbit, used to check whether quantum methods
help with two operations tasks: detecting anomalies in telemetry and choosing which
ground-station passes to use for downlink. Every quantum method is run against a classical
method on the same inputs.

**Author:** Aman Sharma, final-year B.Tech CSE (AI/ML), IILM University, Greater Noida.
Final-year project, 2026-27.

**AI assistance:** the code and experiments were developed with Claude Code (Anthropic) as a
coding assistant. The author ran, reviewed and verified all code and results.

## What is in the twin

| Part | What it does | Code |
|---|---|---|
| Physics model | circular orbit, cylindrical-shadow eclipse, battery energy balance, single-node thermal model, step faults, ground-station pass windows | `dt/physics.py` |
| Telemetry path | replays ESA OPS-SAT segments in time order, computes features when a segment closes, scores it, keeps a per-channel risk | `dt/ingest.py`, `dt/twin.py` |
| Physics path | compares measured solar power, state of charge and temperature with the model and alerts on large residuals | `dt/twin.py` |
| Classical detectors | IsolationForest, RandomForest, RBF-kernel SVM | `dt/classical.py` |
| Quantum classifiers | quantum-kernel SVM (ZZ feature map, exact fidelity kernel) and a variational classifier | `dt/quantum_ml.py` |
| Scheduler | downlink selection as a QUBO; QAOA, exact eigensolver, brute force, greedy, random search | `dt/quantum_opt.py` |
| Service | FastAPI REST service and Streamlit dashboard | `api/`, `dashboard/` |

All quantum circuits are simulated with Qiskit statevector primitives. No quantum hardware
was used.

## Results

Numbers below are read from the files in `results/`, produced on the machine described in
`results/env.json` (seed 42).

**Anomaly detection, official OPSSAT-AD split (529 test segments, 113 anomalous)** â
`results/classical_metrics.csv`

| Features | Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| published (18 + channel) | SVM-RBF | 0.936 | 0.903 | 0.919 | 0.986 |
| own (16 + channel) | SVM-RBF | 0.904 | 0.912 | 0.907 | 0.991 |
| combined (43) | SVM-RBF | 0.936 | 0.903 | 0.919 | 0.988 |
| combined (43) | RandomForest | 0.943 | 0.885 | 0.913 | 0.994 |
| combined (43) | IsolationForest (unsupervised) | 0.536 | 0.850 | 0.658 | 0.897 |

The live twin computes only the "own" features from the stream, so it runs the 0.907 model.
Its streamed predictions match offline scoring on all 529 segments (`results/twin_replay.json`).

**Quantum vs classical classifiers, untuned first comparison** â `results/qml_metrics.csv`
(400 balanced training segments, one seed, fixed C = 10)

| PCA components = qubits | QSVC F1 | VQC F1 | SVM-RBF F1 | RandomForest F1 |
|---|---|---|---|---|
| 2 | 0.466 | 0.219 | 0.621 | 0.581 |
| 4 | 0.646 | 0.438 | 0.779 | 0.759 |
| 6 | 0.624 | 0.432 | 0.829 | 0.813 |

This comparison is untuned and uses a quarter of the training data, so it should not be read
as the final word. `experiments/qml_fair_comparison.py` reruns it with the full training set,
equal tuning for both sides, thresholds chosen on training data, five subsample seeds and
bootstrap intervals. **Its results are not committed yet**: run it (10-20 minutes) and commit
`results/qml_fair_*`.

**Downlink scheduling, 12 instances of 4-10 passes, QAOA depth 1-3** â
`results/qaoa_summary.json`

| Solver | Found the optimum |
|---|---|
| Brute force / exact QUBO eigensolver | 12 of 12 instances |
| Uniform random search, 4,096 samples | 12 of 12 instances |
| Greedy (value per transmit-minute) | 11 of 12 instances (0.945 of optimum on the other) |
| QAOA best of 4,096 shots | 35 of 36 runs |

QAOA's probability of drawing the optimal schedule in one shot, relative to uniform random
sampling: median 0.89, mean 1.30, above 1 in 16 of 36 runs. The mean is carried by a single
run at 14.3; without it the mean is 0.93. At these sizes QAOA did not sample the optimum more
often than chance.

**Physics path detection limits, 5 noise seeds per setting** â `results/fault_sweep.json`

| Fault | Smallest size detected in every run | Median delay at that size |
|---|---|---|
| Panel output loss | 5% | 27 min (10% and above: within one 10 s step) |
| Battery capacity fade | 20% (5% and 10% are never detected) | 64 min |
| Heater stuck on | 1 W (0.5 W detected in 1 of 5 runs) | 173 min |

No false alarms on nominal runs with the assumed sensor noise (0.5 W, 0.005 SoC, 0.5 K).
The eclipse fraction matches the closed form within 0.0002 (`results/physics_validation.json`).

## Limitations

- Circuits are simulated without hardware noise, at 2-6 qubits for classification and up to
  16 for scheduling. Nothing here says how the methods behave at larger sizes.
- The physics path is tested on simulated telemetry. OPS-SAT channel names are anonymised,
  so the real telemetry cannot be mapped to solar power, state of charge or temperature.
  The telemetry path and the physics path therefore do not share data.
- The orbit model is two-body and circular: no J2, drag or TLE propagation. The thermal model
  has one node and no albedo term.
- Segments are scored when they close, using the segment boundaries of the benchmark.
- Downlink rates, transmitter power and the battery reserve are assumed CubeSat values.

## Setup

Python 3.12.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
python data/download.py         # OPSSAT-AD from Zenodo (~19 MB)
```

## Reproduce

```bash
python experiments/run_all.py               # every experiment with fixed seeds, then pytest
python experiments/run_all.py --skip-qaoa   # skip the slow QAOA study
```

| Script | Needs dataset | Output |
|---|---|---|
| `experiments/eda.py` | yes | dataset statistics, `results/eda_*` |
| `experiments/physics_check.py` | no | physics validation, `results/physics_*`, `results/env.json` |
| `experiments/classical_baselines.py` | yes | classical detectors, `results/classical_*`, fitted models |
| `experiments/twin_replay.py` | yes | streaming replay and fault injection, `results/twin_*` |
| `experiments/fault_sweep.py` | no | detection limits of the physics path, `results/fault_sweep*` |
| `experiments/quantum_vs_classical.py` | yes | untuned QSVC / VQC vs SVM / RF, `results/qml_*` |
| `experiments/qml_fair_comparison.py` | yes | tuned, full-training-set comparison with intervals, `results/qml_fair*` |
| `experiments/qaoa_scaling.py` | no | QAOA vs exact, greedy and random search, `results/qaoa_*` |

Options: `qml_fair_comparison.py --vqc` also retrains the variational classifier (slow);
`qaoa_scaling.py --extended` runs 24 instances with 3 seeds; `qaoa_scaling.py --from-csv`
rebuilds the summary from saved runs.

## Run the twin

```bash
uvicorn api.main:app --port 8000      # REST API, docs at http://127.0.0.1:8000/docs
streamlit run dashboard/app.py        # dashboard at http://localhost:8501
python experiments/api_benchmark.py   # latency benchmark against the running API
```

The API needs the model saved by `experiments/classical_baselines.py`.

## Tests

```bash
pytest
```

Tests that need the dataset or a fitted model are skipped when those are absent, so the suite
runs on a fresh clone.

## Layout

```
dt/physics.py      orbit/eclipse, power, thermal models, fault injection, ground-station passes
dt/ingest.py       OPS-SAT loading, normalisation, features, stream replay
dt/twin.py         SatelliteTwin: health state, physics residuals, SQLite history
dt/classical.py    classical anomaly detectors
dt/quantum_ml.py   QSVC, VQC and the exact fidelity kernel (Qiskit)
dt/quantum_opt.py  downlink scheduling QUBO, QAOA, exact, greedy and random-search solvers
dt/evaluation.py   model selection, thresholds, bootstrap intervals
api/main.py        FastAPI service
dashboard/app.py   Streamlit UI
experiments/       one script per result
results/           saved outputs of the experiments
tests/             pytest suite
```

## Citation

If you use this code, please cite it using `CITATION.cff` (GitHub shows a "Cite this
repository" button).

## Dataset credit

OPSSAT-AD, KP Labs / ESA, MIT licence, Zenodo record 15108715.
B. Ruszczak et al., "The OPS-SAT benchmark for detecting anomalies in satellite telemetry",
arXiv:2407.04730. The dataset is downloaded by `data/download.py` and is not redistributed here.

## Licence

MIT, see `LICENSE`.
