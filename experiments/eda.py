"""Exploratory data analysis of OPSSAT-AD. Writes results/eda_*.csv|png|json."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "opssat"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)


def main():
    seg = pd.read_csv(RAW / "segments.csv", parse_dates=["timestamp"])
    feat = pd.read_csv(RAW / "dataset.csv")

    summary = {
        "raw_samples": int(len(seg)),
        "segments": int(feat["segment"].nunique()),
        "channels": sorted(feat["channel"].unique().tolist()),
        "anomalous_segments": int(feat["anomaly"].sum()),
        "nominal_segments": int((feat["anomaly"] == 0).sum()),
        "anomaly_ratio": round(float(feat["anomaly"].mean()), 4),
        "train_segments": int(feat["train"].sum()),
        "test_segments": int((feat["train"] == 0).sum()),
        "time_span": [str(seg["timestamp"].min()), str(seg["timestamp"].max())],
        "segment_len": feat["len"].describe().round(1).to_dict(),
        "sampling_rates": feat["sampling"].value_counts().to_dict(),
        "feature_columns": [c for c in feat.columns if c not in ("segment", "anomaly", "train", "channel")],
    }
    (RES / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    per_ch = (
        feat.groupby("channel")
        .agg(segments=("segment", "count"), anomalies=("anomaly", "sum"),
             train=("train", "sum"), mean_len=("len", "mean"))
        .assign(anomaly_ratio=lambda d: (d.anomalies / d.segments).round(3))
    )
    per_ch.to_csv(RES / "eda_per_channel.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    per_ch[["segments", "anomalies"]].plot.bar(ax=ax)
    ax.set_title("OPSSAT-AD: segments and anomalies per telemetry channel")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(RES / "eda_per_channel.png", dpi=150)

    # one nominal and one anomalous segment from the same channel
    ch = per_ch.anomalies.idxmax()
    fig, axes = plt.subplots(2, 1, figsize=(9, 5))
    for ax, a in zip(axes, (0, 1)):
        sid = feat[(feat.channel == ch) & (feat.anomaly == a)]["segment"].iloc[0]
        s = seg[seg.segment == sid]
        ax.plot(s["timestamp"], s["value"], lw=0.8, color="tab:red" if a else "tab:blue")
        ax.set_title(f"{ch} segment {sid} ({'anomaly' if a else 'nominal'})")
    fig.tight_layout()
    fig.savefig(RES / "eda_example_segments.png", dpi=150)

    print(json.dumps({k: v for k, v in summary.items() if k != "feature_columns"}, indent=2, default=str))
    print(per_ch)


if __name__ == "__main__":
    main()
