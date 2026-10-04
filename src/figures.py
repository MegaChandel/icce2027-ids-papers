"""Project 2 figures (vector PDF, IEEE 3.5 in column). Same palette/style as Project 1.
python -m src.figures
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
R, F = ROOT / "results", ROOT / "figures"
COL = 3.5
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
MARK = ["o", "s", "^", "D", "v", "P"]
SETTINGS = ["Within-OLD", "Within-NEW", "OLD->NEW", "NEW->OLD", "Within-SHADOW", "HCRL->SHADOW",
            "SHADOW->OLD", "SHADOW->NEW"]
SET_LBL = ["W-OLD", "W-NEW", "OLD→\nNEW", "NEW→\nOLD", "W-SHAD", "HCRL→\nSHAD", "SHAD→\nOLD", "SHAD→\nNEW"]
MODELS = ["M1_LSTM", "M2_CNN", "M3_AE", "M4_NGRAM", "R1_RATE", "R2_HIST"]
MOD_LBL = {"M1_LSTM": "LSTM", "M2_CNN": "1D-CNN", "M3_AE": "LSTM-AE", "M4_NGRAM": "n-gram LR",
           "R1_RATE": "Rate (trivial)", "R2_HIST": "Histogram (trivial)"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02})


def _axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def _save(fig, name):
    fig.savefig(F / f"{name}.pdf")
    fig.savefig(F / f"{name}.png", dpi=200)
    plt.close(fig)


def fig_matrix():
    df = pd.read_csv(R / "matrix.csv")
    v = df[(df.attack == "all") & (df.vocab == "train")]
    m = v.groupby(["model", "setting"])["auc_roc"].mean().unstack("setting").loc[MODELS, SETTINGS]
    # diverging around chance: below 0.5 = worse than random, so it gets its own pole
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(COL, 2.05))
    ax.imshow(m.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            x = m.values[i, j]
            ax.text(j, i, f"{x:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(x - 0.5) > 0.33 else INK)
    ax.set_xticks(range(len(SETTINGS)), SET_LBL, fontsize=6)
    ax.set_yticks(range(len(MODELS)), [MOD_LBL[x] for x in MODELS])
    ax.axvline(1.5, color="white", lw=2); ax.axvline(3.5, color="white", lw=2); ax.axvline(4.5, color="white", lw=2)
    ax.axhline(3.5, color="white", lw=2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.tick_top()
    _save(fig, "fig_matrix")


def fig_footprint():
    e = pd.read_csv(R / "errors.csv")
    order = [("OLD", "normal"), ("OLD", "heartbeat"), ("OLD", "ping"), ("OLD", "request"),
             ("NEW", "normal"), ("NEW", "heartbeat"), ("NEW", "ping"), ("NEW", "request"),
             ("SHADOW", "normal"), ("SHADOW", "relay")]
    e = e.set_index(["campaign", "attack"]).loc[order].reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.9), gridspec_kw={"width_ratios": [1.15, 1]})
    camp_col = {"OLD": C[0], "NEW": C[1], "SHADOW": C[2]}
    y = np.arange(len(e))[::-1]
    for ax, col, xl, logx in [(axs[0], "rate_fps", "UAV output (frames/s)", True),
                              (axs[1], "jsd_own", "JSD to own normal", False)]:
        if col == "jsd_own":
            vals = [r[f"jsd_to_{r['campaign']}_normal"] for _, r in e.iterrows()]
        else:
            vals = e[col].to_numpy()
        for yi, v, r, mk in zip(y, vals, e.itertuples(), MARK * 3):
            ax.hlines(yi, 1e-9 if not logx else 10, v, color=GRID, lw=1.0)
            ax.plot(v, yi, "o" if r.attack == "normal" else "D", color=camp_col[r.campaign], ms=4,
                    mec="white", mew=0.4)
        if logx:
            ax.set_xscale("log"); ax.set_xlim(15, 5000)
        else:
            ax.set_xlim(0, 0.95)
        ax.set_xlabel(xl)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    axs[0].set_yticks(y, [f"{r.campaign} {r.attack}" for r in e.itertuples()], fontsize=6.5)
    axs[1].set_yticks(y, [""] * len(y))
    for s in ("left",):
        axs[1].spines[s].set_visible(False)
    axs[1].tick_params(axis="y", length=0)
    _save(fig, "fig_footprint")


def fig_sensitivity():
    files = {32: R / "matrix.csv", **{L: R / f"sens_L{L}.csv" for L in (16, 64, 128)}}
    if not all(p.exists() for p in files.values()):
        return
    rows = []
    for L, p in files.items():
        d = pd.read_csv(p)
        d = d[(d.attack == "all") & (d.vocab == "train")]
        d = d.groupby(["model", "setting"])["auc_roc"].mean().reset_index()
        d["L"] = L
        rows.append(d)
    d = pd.concat(rows)
    show = [("Within-NEW", "OLD->NEW"), ("Within-SHADOW", "HCRL->SHADOW")]
    models = ["M1_LSTM", "M4_NGRAM", "R1_RATE", "R2_HIST"]
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.8), sharey=True)
    for ax, (w, x) in zip(axs, show):
        for i, mname in enumerate(models):
            for st, ls in ((w, "-"), (x, (0, (2, 1.5)))):
                q = d[(d.model == mname) & (d.setting == st)].sort_values("L")
                ax.plot(q.L, q.auc_roc, color=C[i], ls=ls, marker=MARK[i], ms=3, lw=1.1,
                        mec="white", mew=0.3, label=MOD_LBL[mname] if st == w else None)
        ax.axhline(0.5, color=INK2, lw=0.6, ls=(0, (3, 2)))
        ax.set_xscale("log", base=2); ax.set_xticks([16, 32, 64, 128], ["16", "32", "64", "128"])
        ax.set_xlabel("Window length (messages)")
        ax.set_title("Target: " + w.split("-")[1], fontsize=7)
        _axes(ax)
    axs[0].set_ylabel("AUC-ROC"); axs[0].set_ylim(0.2, 1.03)
    axs[0].legend(loc="lower center", bbox_to_anchor=(1.05, 1.12), ncol=4, frameon=False,
                  handlelength=1.5, columnspacing=0.8)
    _save(fig, "fig_sensitivity")


def main():
    F.mkdir(exist_ok=True)
    fig_matrix()
    fig_footprint()
    fig_sensitivity()
    print("figures written to", F)


if __name__ == "__main__":
    main()
