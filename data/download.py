"""Download the OPSSAT-AD satellite telemetry anomaly dataset (ESA OPS-SAT, KP Labs, 2024).

Source: Zenodo record 15108715 (https://zenodo.org/records/15108715), MIT licence.
Paper:  Ruszczak et al., "The OPS-SAT benchmark for detecting anomalies in satellite
        telemetry", arXiv:2407.04730.

Files land in data/raw/opssat/:
  segments.csv  raw telemetry samples (channel, timestamp, value, anomaly, segment, train)
  dataset.csv   per-segment engineered features published with the benchmark

Note: the NASA SMAP/MSL archive (s3://telemanom/data.zip) returned HTTP 403 on
22-Oct-2026, which is why OPS-SAT is the primary dataset of this project.
"""
from pathlib import Path

import requests

RECORD = "https://zenodo.org/api/records/15108715/files/{}/content"
FILES = ["segments.csv", "dataset.csv", "LICENSE"]
OUT = Path(__file__).resolve().parent / "raw" / "opssat"


def download():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = OUT / name
        if path.exists():
            print(f"{name}: already present ({path.stat().st_size} bytes)")
            continue
        r = requests.get(RECORD.format(name), timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
        print(f"{name}: {len(r.content)} bytes")


if __name__ == "__main__":
    download()
