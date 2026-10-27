"""Reproduce every reported result from scratch (fixed seeds), then run the test suite.

    python experiments/run_all.py            # all experiments + tests
    python experiments/run_all.py --skip-qaoa  # skip the slow QAOA scaling study

The API benchmark and dashboard screenshots need running servers; see README.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("data/download.py", "dataset download (skips existing files)"),
    ("experiments/eda.py", "dataset statistics"),
    ("experiments/physics_check.py", "physics twin validation + environment"),
    ("experiments/classical_baselines.py", "classical detectors, saves live model"),
    ("experiments/twin_replay.py", "end-to-end twin replay + fault injection"),
    ("experiments/quantum_vs_classical.py", "QSVC / VQC vs classical"),
    ("experiments/qaoa_scaling.py", "QAOA scheduling study"),
]


def main():
    skip_qaoa = "--skip-qaoa" in sys.argv
    timings = []
    for script, what in STEPS:
        if skip_qaoa and "qaoa" in script:
            continue
        print(f"\n=== {script}: {what}", flush=True)
        t0 = time.perf_counter()
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
        timings.append((script, time.perf_counter() - t0))
    print("\n=== pytest", flush=True)
    with open(ROOT / "results" / "test_report_latest.txt", "w") as f:
        rc = subprocess.run([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider"],
                            cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    print("\n".join(f"{s:42s} {t:8.1f} s" for s, t in timings))
    print("tests:", "PASSED" if rc == 0 else "FAILED (see results/test_report_latest.txt)")
    sys.exit(rc)


if __name__ == "__main__":
    main()
