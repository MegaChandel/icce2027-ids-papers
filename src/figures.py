"""Step I1: paper figures from results/*.csv (vector PDF, IEEE 3.5 in column width).

Colours: reference categorical slots 1-3 (validated all-pairs, light surface),
always paired with hatching / markers so the figures survive grayscale print.
python -m src.figures
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .detectors import ByteRange  # noqa: E402
from .io_road import ROOT, load_cached, load_captures_cfg  # noqa: E402
from .pipeline import capture_windows, load_ambient, load_experiment_cfg, split_ambient  # noqa: E402

R, F = ROOT / "results", ROOT / "figures"
COL = 3.5
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
MARK = ["o", "s", "^", "D", "v"]
ORDER = ["max_speedometer_attack", "max_engine_coolant_temp_attack", "correlated_signal_attack",
         "reverse_light_off_attack", "reverse_light_on_attack"]
TICK = {"max_speedometer_attack": "Max\nspeedo", "max_engine_coolant_temp_attack": "Max\ncoolant",
        "correlated_signal_attack": "Corre-\nlated", "reverse_light_off_attack": "Rev. light\noff",
        "reverse_light_on_attack": "Rev. light\non"}
LABEL = {"max_speedometer_attack": "Max speedo", "max_engine_coolant_temp_attack": "Max coolant",
         "correlated_signal_attack": "Correlated", "reverse_light_off_attack": "Rev. light off",
         "reverse_light_on_attack": "Rev. light on"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "hatch.linewidth": 0.6,
})


def _axes(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)


def fig2_floor_vs_published():
    floor = pd.read_csv(R / "floor.csv")
    v = floor[(floor["agg"] == "mean") & (floor.family == "masquerade") & (floor.detector == "T2_range")]
    tri = v.groupby("attack_type")["auc_roc"].mean()
    q = pd.read_csv(ROOT / "data/published/moriano2024_table2.csv", comment="#")
    pub_max = q.groupby("attack_type")["max"].max()
    pub_mu = q.groupby("attack_type")["mu"].max()
    ctl = pd.read_csv(R / "controls.csv")
    ifm = ctl[ctl.family == "masquerade"].groupby(["attack_type", "seed"])["auc_roc"].mean() \
        .groupby("attack_type").agg(["mean", "std"])
    series = [("Trivial range rule (T2)", tri, None, C[0], ""),
              ("Best published, tuned on test", pub_max, None, C[1], "////"),
              ("Isolation forest (3 seeds)", ifm["mean"], ifm["std"], C[2], "\\\\\\\\")]
    x = np.arange(len(ORDER))
    w = 0.26
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    for i, (name, s, err, col, hatch) in enumerate(series):
        vals = [s[a] for a in ORDER]
        ax.bar(x + (i - 1) * w, vals, w - 0.03, color=col, edgecolor="white", linewidth=0.6,
               hatch=hatch, label=name, zorder=2,
               yerr=None if err is None else [err[a] for a in ORDER],
               error_kw=dict(lw=0.6, capsize=1.5, ecolor=INK2))
    ax.scatter(x, [pub_mu[a] for a in ORDER], marker="_", s=60, color=INK, zorder=3, lw=1.2,
               label="Best published, mean over grid")
    ax.axhline(0.5, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(-0.62, 0.51, "chance", color=INK2, fontsize=6, ha="left", va="bottom")
    ax.set_xticks(x, [TICK[a] for a in ORDER])
    ax.set_xlim(-0.65, len(ORDER) - 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Window AUC-ROC")
    _axes(ax)
    h, l = ax.get_legend_handles_labels()
    order = [1, 2, 0, 3]  # T2, tuned max, mean-over-grid, IF
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.6, columnspacing=1.0)
    fig.savefig(F / "fig2_floor_vs_published.pdf")
    fig.savefig(F / "fig2_floor_vs_published.png", dpi=200)
    plt.close(fig)


def fig3_sensitivity():
    s = pd.read_csv(R / "sensitivity.csv")
    s = s[s.detector == "T2_range"]
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.7), sharey=True)
    for ax, sweep, xl, logx in [(axs[0], "window_length_s", "Window length (s)", True),
                                (axs[1], "ambient_fraction", "Ambient data used", False)]:
        g = s[s.sweep == sweep].groupby(["attack_type", "value"])["auc_roc"].mean()
        for i, a in enumerate(ORDER):
            d = g[a]
            ax.plot(d.index, d.values, color=C[i], marker=MARK[i], ms=3.5, lw=1.2, label=LABEL[a],
                    mec="white", mew=0.4)
        ax.axhline(0.5, color=INK2, lw=0.6, ls=(0, (3, 2)))
        ax.text(0.5, 0.80, "speedo, coolant and correlated\nall overlap at 1.00", transform=ax.transAxes,
                ha="center", va="center", fontsize=6, color=INK2)
        ax.set_xlabel(xl)
        if logx:
            ax.set_xscale("log", base=2)
            ax.set_xticks([0.25, 0.5, 1, 2], ["0.25", "0.5", "1", "2"])
        else:
            ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0], ["10%", "", "50%", "", "100%"])
            ax.tick_params(axis="x", labelsize=6)
        _axes(ax)
    axs[0].set_ylabel("T2 window AUC-ROC")
    axs[0].set_ylim(0.25, 1.06)
    axs[0].legend(loc="lower center", bbox_to_anchor=(1.05, 1.0), ncol=3, frameon=False,
                  handlelength=1.6, columnspacing=0.8)
    fig.savefig(F / "fig3_sensitivity.pdf")
    fig.savefig(F / "fig3_sensitivity.png", dpi=200)
    plt.close(fig)


def fig4_byte_heatmap():
    h = pd.read_csv(R / "byte_flag_rates.csv")
    h = h[h.family == "masquerade"].copy()
    h["type"] = [next(a for a in ORDER if c.startswith(a.replace("_attack", ""))) for c in h.capture]
    h["k"] = h["type"].map({a: i for i, a in enumerate(ORDER)})
    h = h.sort_values(["k", "capture"])
    caps = load_captures_cfg()
    pat = {c["name"]: c["injection_data"] for c in caps["attacks"]}
    m = h[[f"b{j}" for j in range(8)]].to_numpy()
    fig, ax = plt.subplots(figsize=(COL, 2.25))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seqblue", ["#f4f3f0", "#86b6ef", "#2a78d6", "#0d366b"])
    ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, cap in enumerate(h.capture):
        p = pat[cap]
        for j in range(8):
            injected = p[2 * j: 2 * j + 2] != "XX"
            if injected:  # outline the bytes the attack actually wrote
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=C[1], lw=1.1))
            if m[i, j] > 0:
                ax.text(j, i, f"{m[i, j]:.0%}", ha="center", va="center", fontsize=5.5,
                        color="white" if m[i, j] > 0.5 else INK)
    names = [LABEL[t] + " " + (c.split("_")[-2] if c.split("_")[-2].isdigit() else "")
             for t, c in zip(h["type"], h.capture)]
    ax.set_yticks(range(len(h)), [n.strip() for n in names], fontsize=6)
    ax.set_xticks(range(8), [f"B{j}" for j in range(8)])
    ax.set_xlabel("Payload byte of the attacked identifier")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Share of injected frames outside the ambient byte range\n"
                 "(orange outline = byte written by the attack)", fontsize=7, color=INK)
    fig.savefig(F / "fig4_byte_flags.pdf")
    fig.savefig(F / "fig4_byte_flags.png", dpi=200)
    plt.close(fig)


def fig5_timeseries():
    exp, caps = load_experiment_cfg(), load_captures_cfg()
    W = exp["window"]
    fit, _ = split_ambient(caps, exp)
    t2 = ByteRange().fit(load_ambient(fit))
    by = {c["name"]: c for c in caps["attacks"]}
    show = ["max_speedometer_attack_1_masquerade", "correlated_signal_attack_1_masquerade",
            "reverse_light_on_attack_1_masquerade"]
    fig, axs = plt.subplots(len(show), 1, figsize=(COL, 3.0), sharey=True,
                            gridspec_kw={"hspace": 0.55})
    for ax, name in zip(axs, show):
        cap = by[name]
        df = load_cached(name)
        w, _ = capture_windows(df, W["length_s"], W["stride_s"])
        from .evaluate import window_scores
        flag = (t2.byte_excess(df) > 0).any(1).astype(float)
        ws = window_scores(df["t"].to_numpy(), flag, w, "mean")
        ax.axvspan(cap["interval_start_s"], cap["interval_end_s"], color=C[1], alpha=0.15, lw=0)
        ax.plot(w.mean(1), ws * 100, color=C[0], lw=1.0)
        ax.set_ylabel("% flagged", fontsize=6.5)
        ax.set_title(LABEL[cap["attack_type"]] + " (masquerade)", loc="left", fontsize=6.5,
                     color=INK, pad=2)
        ax.set_xlim(0, w[-1, 1])
        ax.set_ylim(0, 6)
        _axes(ax)
    axs[-1].set_xlabel("Time (s); shaded = labelled injection interval")
    fig.align_ylabels(axs)
    fig.savefig(F / "fig5_timeseries.pdf")
    fig.savefig(F / "fig5_timeseries.png", dpi=200)
    plt.close(fig)


def main():
    F.mkdir(exist_ok=True)
    fig2_floor_vs_published()
    if (R / "sensitivity.csv").exists():
        fig3_sensitivity()
    fig4_byte_heatmap()
    fig5_timeseries()
    print("figures written to", F)


if __name__ == "__main__":
    main()
