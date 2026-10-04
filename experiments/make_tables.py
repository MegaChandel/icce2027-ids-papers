"""Step H1: results/*.csv -> paper tables (CSV + LaTeX). No number is typed by hand.

Outputs (results/tables/):
  table2_floor.{csv,tex}   per masquerade capture: AUC [95% CI] for T1-T4, det. rate @1% FPR for T2
  table3_delta.{csv,tex}   per attack type: trivial floor (T2) vs quoted Moriano et al. detectors,
                           IsolationForest control, and delta-over-trivial
  table4_ablation.{csv,tex}
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
R, OUT = ROOT / "results", ROOT / "results" / "tables"
FLOOR_DET = "T2_range"  # the pre-specified floor detector (simplest rule)

PRETTY = {"correlated_signal_attack": "Correlated signal", "max_engine_coolant_temp_attack": "Max coolant temp",
          "max_speedometer_attack": "Max speedometer", "reverse_light_off_attack": "Reverse light off",
          "reverse_light_on_attack": "Reverse light on"}
DET = {"T1_valueset": "T1", "T2_range": "T2", "T3_envelope": "T3", "T3_envelope_p0.1": "T3$_{0.1}$",
       "T4_novelty": "T4"}


def short(capture):
    c = capture.replace("_masquerade", "").replace("_attack", "")
    return c.replace("max_engine_coolant_temp", "coolant").replace("max_speedometer", "speedo") \
            .replace("correlated_signal", "corr").replace("reverse_light", "rl").replace("_", "-")


def table2(floor):
    v = floor[(floor["agg"] == "mean") & (floor["family"] == "masquerade")].copy()
    v["cell"] = [f"{a:.2f} [{lo:.2f}, {hi:.2f}]" for a, lo, hi in zip(v.auc_roc, v.ci_low, v.ci_high)]
    t = v.pivot_table(index=["attack_type", "capture"], columns="detector", values="cell", aggfunc="first")
    t = t[[c for c in DET if c in t.columns]].rename(columns=DET)
    det = v[v.detector == FLOOR_DET].set_index(["attack_type", "capture"])["det_rate_at_1pct_fpr"]
    t["T2 TPR@1%FPR"] = (det * 100).round(0).astype(int).astype(str) + "%"
    t.index = [short(c) for _, c in t.index]
    return t


def table3(floor, controls, quoted):
    v = floor[(floor["agg"] == "mean") & (floor["family"] == "masquerade") & (floor.detector == FLOOR_DET)]
    tri = v.groupby("attack_type")["auc_roc"].mean().rename("Trivial T2")
    tpr = (v.groupby("attack_type")["det_rate_at_1pct_fpr"].mean() * 100).rename("T2 TPR@1%")
    c = controls[controls.family == "masquerade"].groupby(["attack_type", "seed"])["auc_roc"].mean()
    ifm = c.groupby("attack_type").agg(["mean", "std"])
    ifs = pd.Series([f"{m:.2f}±{s:.2f}" for m, s in ifm.values], index=ifm.index, name="IF control")
    best_mu = quoted.groupby("attack_type")["mu"].max().rename("Best published (mean)")
    best_max = quoted.groupby("attack_type")["max"].max().rename("Best published (tuned max)")
    who = quoted.loc[quoted.groupby("attack_type")["max"].idxmax()].set_index("attack_type")["method"]
    t = pd.concat([tri, tpr, ifs, best_mu, best_max, who.rename("method")], axis=1)
    t["Delta (mean)"] = t["Best published (mean)"] - t["Trivial T2"]
    t["Delta (tuned max)"] = t["Best published (tuned max)"] - t["Trivial T2"]
    t.index = [PRETTY[i] for i in t.index]
    return t


def table4(abl):
    v = abl[abl.family == "masquerade"]
    cols = ["T2_range", "T2_norm", "T2_target_only", "T2_excl_target"] + [f"T2_byte{j}" for j in range(8)]
    t = v.pivot_table(index="attack_type", columns="variant", values="auc_roc", aggfunc="mean")[cols]
    t.index = [PRETTY[i] for i in t.index]
    return t


def write(t, name, fmt="%.2f"):
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / f"{name}.csv", float_format=fmt)
    (OUT / f"{name}.tex").write_text(t.to_latex(float_format=fmt.__mod__, escape=False))
    print(f"\n== {name}\n{t.to_string(float_format=fmt.__mod__)}")


def main():
    floor = pd.read_csv(R / "floor.csv")
    quoted = pd.read_csv(ROOT / "data/published/moriano2024_table2.csv", comment="#")
    write(table2(floor), "table2_floor")
    if (R / "controls.csv").exists():
        write(table3(floor, pd.read_csv(R / "controls.csv"), quoted), "table3_delta")
    if (R / "ablation.csv").exists():
        write(table4(pd.read_csv(R / "ablation.csv")), "table4_ablation")


if __name__ == "__main__":
    sys.exit(main())
