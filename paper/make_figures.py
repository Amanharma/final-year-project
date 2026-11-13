"""Column-width figures for the paper, drawn from the saved results (no experiment is re-run,
except one noisy physics run for the residual plot).

    python paper/make_figures.py        # writes paper/figures/*.pdf
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dt.physics import Faults, SatellitePhysics  # noqa: E402
from dt.twin import PHYSICS_THRESHOLDS, SatelliteTwin  # noqa: E402

RES, OUT = ROOT / "results", ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "lines.linewidth": 1.0, "lines.markersize": 3.5,
                     "font.family": "serif", "pdf.fonttype": 42})
COL = 3.45   # IEEE column width [in]


def residuals():
    sp = SatellitePhysics()
    onset = 4 * sp.orbit.period_s
    noise = {"solar_w": 0.5, "soc": 0.005, "temp_k": 0.5}
    rng = np.random.default_rng(0)
    sim = sp.simulate(n_orbits=10, dt_s=10, faults=Faults(battery_fade=(onset, 0.4)))
    twin = SatelliteTwin({"model": None, "columns": [], "channel_stats": pd.DataFrame(columns=["mean", "std"])},
                         physics=sp)
    res = {k: [] for k in noise}
    for i, t in enumerate(sim["t_s"]):
        out = twin.update_physics(t, {k: sim[k][i] + rng.normal(0, noise[k]) for k in noise})
        for k in noise:
            res[k].append(out["residual"][k])
    labels = {"solar_w": "solar [W]", "soc": "SoC", "temp_k": "temp. [K]"}
    fig, ax = plt.subplots(3, 1, figsize=(COL, 2.6), sharex=True)
    for a, k in zip(ax, noise):
        a.plot(sim["t_s"] / 3600, res[k], lw=0.4, color="C0")
        for s in (1, -1):
            a.axhline(s * PHYSICS_THRESHOLDS[k], color="C3", ls="--", lw=0.7)
        a.axvline(onset / 3600, color="k", ls=":", lw=0.7)
        a.set_ylabel(labels[k]); a.grid(alpha=0.3)
    ax[-1].set_xlabel("time [h]")
    fig.align_ylabels(ax); fig.tight_layout(pad=0.3, h_pad=0.2)
    fig.savefig(OUT / "residuals.pdf")


def qml():
    """Left: tuned models on the full training set with 95% bootstrap intervals.
    Right: exact quantum-kernel matrix time against training-set size."""
    df = pd.read_csv(RES / "qml_fair_summary.csv")
    df = df[(df.train_set == "full")]
    sc = pd.read_csv(RES / "qml_kernel_scaling.csv")
    fig, ax = plt.subplots(1, 2, figsize=(COL, 2.05))
    rows = {"QSVC": ("o-", True), "SVM-RBF": ("^-", True), "RandomForest": ("v--", False)}
    for i, (name, (style, tuned)) in enumerate(rows.items()):
        g = df[(df.model == name) & (df.tuned == tuned) & (df.dims != "all")].copy()
        g["d"] = g.dims.astype(int); g = g.sort_values("d")
        err = [g.f1_oofthr_mean - g.f1_oofthr_ci_lo, g.f1_oofthr_ci_hi - g.f1_oofthr_mean]
        ax[0].errorbar(g.d + (i - 1) * 0.15, g.f1_oofthr_mean, yerr=err, fmt=style, capsize=2,
                       elinewidth=0.7, label=name + (" (tuned)" if tuned else ""))
    ceil = df[(df.dims == "all") & (df.model == "SVM-RBF") & (df.tuned)].f1_oofthr_mean.iloc[0]
    ax[0].axhline(ceil, color="gray", ls=":", lw=0.8, label="SVM-RBF, 43 feat.")
    ax[0].set_xticks([2, 4, 6]); ax[0].set_xlabel("PCA components = qubits"); ax[0].set_ylabel("test F1")
    ax[0].set_ylim(0.35, 1.0)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=2, handlelength=1.6,
               columnspacing=1.0, frameon=False)
    ax[1].loglog(sc.n, sc.seconds, "o-")
    ax[1].set_xlabel("training segments $N$"); ax[1].set_ylabel("kernel matrix time [s]")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0, 1, 0.83))
    fig.savefig(OUT / "qml.pdf")


def qaoa():
    df = pd.read_csv(RES / "qaoa_scaling.csv")
    df["ratio"] = df.p_optimal * 2.0 ** df.n_passes
    floor = 0.05                      # runs with no optimal sample are drawn on the floor of the log axis
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 2, figsize=(COL, 1.9))
    for p, m in zip((1, 2, 3), ("o", "s", "^")):
        g = df[df.reps == p]
        ax[0].scatter(g.n_passes + (p - 2) * 0.35 + rng.uniform(-0.08, 0.08, len(g)), g.ratio.clip(lower=floor),
                      marker=m, s=9, label=f"$p={p}$", alpha=0.85)
        t = g.groupby("n_passes").qaoa_s.mean()
        ax[1].semilogy(t.index, t.values, m + "-", label=f"$p={p}$")
    ax[0].axhline(1.0, color="k", ls=":", lw=0.8)
    ax[0].set_yscale("log"); ax[0].set_xticks([4, 6, 8, 10])
    ax[0].set_xlabel("passes $n$"); ax[0].set_ylabel(r"$P_{\mathrm{opt}}$ / random")
    ax[1].set_xticks([4, 6, 8, 10]); ax[1].set_xlabel("passes $n$"); ax[1].set_ylabel("QAOA wall time [s]")
    fig.legend(*ax[1].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / "qaoa.pdf")


if __name__ == "__main__":
    residuals(); qml(); qaoa()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
