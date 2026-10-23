"""OPSSAT-AD ingestion: load segments, normalise per channel, extract features, replay as a stream.

Every OPSSAT-AD segment belongs to exactly one telemetry channel and carries one label
(anomaly 0/1) and one split flag (train 0/1), so classification is done per segment.
Channel magnitudes differ by ~5 orders (1e-5 vs 1e0), hence per-channel z-scoring using
statistics of *training* segments only (no test leakage).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "opssat"

# features published with the benchmark (dataset.csv); computed on raw values
PUBLISHED = ["duration", "len", "mean", "var", "std", "kurtosis", "skew", "n_peaks",
             "smooth10_n_peaks", "smooth20_n_peaks", "diff_peaks", "diff2_peaks", "diff_var",
             "diff2_var", "gaps_squared", "len_weighted", "var_div_duration", "var_div_len"]


def load_segments(raw_dir: Path = RAW) -> pd.DataFrame:
    seg = pd.read_csv(raw_dir / "segments.csv", parse_dates=["timestamp"])
    return seg.sort_values(["segment", "timestamp"]).reset_index(drop=True)


def load_published_features(raw_dir: Path = RAW) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "dataset.csv").set_index("segment").sort_index()


def channel_stats(seg: pd.DataFrame) -> pd.DataFrame:
    """Mean/std per channel from training segments only."""
    tr = seg[seg["train"] == 1]
    st = tr.groupby("channel")["value"].agg(["mean", "std"])
    st["std"] = st["std"].replace(0, 1.0)
    return st


def normalise(seg: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    seg = seg.copy()
    m = seg["channel"].map(stats["mean"])
    s = seg["channel"].map(stats["std"])
    seg["z"] = (seg["value"] - m) / s
    return seg


def segment_features(z: np.ndarray, dt_s: np.ndarray) -> dict:
    """Scale-free features computed on a normalised segment (our additions to PUBLISHED)."""
    n = len(z)
    d = np.diff(z) if n > 1 else np.zeros(1)
    zc = z - z.mean()
    spec = np.abs(np.fft.rfft(zc)) ** 2
    spec = spec[1:] if len(spec) > 1 else np.ones(1)
    total = spec.sum() + 1e-12
    k = max(1, len(spec) // 4)
    freqs = np.arange(1, len(spec) + 1) / max(n, 1)
    ac1 = float(np.corrcoef(z[:-1], z[1:])[0, 1]) if n > 2 and z.std() > 0 else 0.0
    return {
        "z_mean": float(z.mean()),
        "z_std": float(z.std()),
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "z_range": float(z.max() - z.min()),
        "z_absmax": float(np.abs(z).max()),
        "diff_absmax": float(np.abs(d).max()),
        "diff_std": float(d.std()),
        "zero_cross_rate": float(np.mean(np.diff(np.sign(zc)) != 0)) if n > 1 else 0.0,
        "autocorr1": 0.0 if np.isnan(ac1) else ac1,
        "spec_low_ratio": float(spec[:k].sum() / total),
        "spec_high_ratio": float(spec[-k:].sum() / total),
        "spec_centroid": float((freqs * spec).sum() / total),
        "spec_entropy": float(-(spec / total * np.log(spec / total + 1e-12)).sum() / np.log(len(spec) + 1)),
        "gap_max_s": float(dt_s.max()) if len(dt_s) else 0.0,
        "gap_irregularity": float(dt_s.std() / (dt_s.mean() + 1e-9)) if len(dt_s) else 0.0,
    }


@dataclass
class FeatureSet:
    X: pd.DataFrame          # features, index = segment id
    y: pd.Series             # 1 = anomaly
    train: pd.Series         # 1 = official training split
    channel: pd.Series

    @property
    def split(self):
        tr, te = self.train == 1, self.train == 0
        return self.X[tr], self.X[te], self.y[tr], self.y[te]


def build_features(seg: pd.DataFrame | None = None, published: pd.DataFrame | None = None,
                   include_channel: bool = True) -> FeatureSet:
    seg = load_segments() if seg is None else seg
    published = load_published_features() if published is None else published
    seg = normalise(seg, channel_stats(seg))

    rows = {}
    for sid, g in seg.groupby("segment", sort=True):
        dt_s = g["timestamp"].diff().dt.total_seconds().dropna().to_numpy()
        rows[sid] = segment_features(g["z"].to_numpy(), dt_s)
    own = pd.DataFrame.from_dict(rows, orient="index")

    X = published[PUBLISHED].join(own, how="inner")
    if include_channel:
        X = X.join(pd.get_dummies(published["channel"], prefix="ch", dtype=float))
    # heavy-tailed raw-scale published features -> signed log
    for c in ["mean", "var", "std", "diff_var", "diff2_var", "var_div_duration", "var_div_len"]:
        X[c] = np.sign(X[c]) * np.log10(np.abs(X[c]) + 1e-15)
    return FeatureSet(X=X, y=published.loc[X.index, "anomaly"], train=published.loc[X.index, "train"],
                      channel=published.loc[X.index, "channel"])


def stream(seg: pd.DataFrame, split: int | None = 0) -> Iterator[dict]:
    """Replay telemetry in timestamp order, emulating a downlink. Emits one dict per sample,
    with `end_of_segment=True` on the last sample of each segment."""
    s = seg if split is None else seg[seg["train"] == split]
    s = s.sort_values("timestamp")
    last_idx = s.groupby("segment")["timestamp"].idxmax()
    last = set(last_idx.to_numpy())
    for idx, r in zip(s.index, s.itertuples(index=False)):
        yield {"timestamp": r.timestamp, "channel": r.channel, "value": r.value,
               "segment": r.segment, "end_of_segment": idx in last}
