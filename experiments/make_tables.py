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


TYPE_ORDER = ["max_speedometer_attack", "max_engine_coolant_temp_attack", "correlated_signal_attack",
              "reverse_light_off_attack", "reverse_light_on_attack"]
METHOD_TEX = {"MatrixCorrelationDistribution": "MCD", "MatrixCorrelationCorrelation": "MCC",
              "Ganesan17": "G17", "Moriano22": "M22"}


def paper_table_floor(floor, controls):
    """Booktabs body: one row per masquerade capture, window AUC for T1-T4 + IF, T2 TPR@1%FPR."""
    v = floor[(floor["agg"] == "mean") & (floor.family == "masquerade")]
    auc = v.pivot_table(index="capture", columns="detector", values="auc_roc")
    tpr = v[v.detector == FLOOR_DET].set_index("capture")["det_rate_at_1pct_fpr"]
    ifm = controls[controls.family == "masquerade"].groupby("capture")["auc_roc"].mean()
    typ = v.drop_duplicates("capture").set_index("capture")["attack_type"]
    lines = []
    for k, a in enumerate(TYPE_ORDER):
        if k:
            lines.append(r"\midrule")
        for c in sorted(typ[typ == a].index):
            cells = [f"{auc.loc[c, d]:.2f}" for d in ("T1_valueset", "T2_range", "T3_envelope", "T4_novelty")]
            best = max(float(x) for x in cells[:3])
            cells = [rf"\textbf{{{x}}}" if float(x) >= 0.95 and i < 3 else x for i, x in enumerate(cells)]
            lines.append(" & ".join([short(c)] + cells + [f"{ifm[c]:.2f}", f"{tpr[c] * 100:.0f}"]) + r" \\")
    (OUT / "paper_table_floor.tex").write_text("\n".join(lines) + "\n")


def paper_table_delta(floor, controls, quoted):
    t3 = table3(floor, controls, quoted)
    lines = []
    for a in TYPE_ORDER:
        r = t3.loc[PRETTY[a]]
        q = quoted[quoted.attack_type == a]
        mu_m = METHOD_TEX[q.loc[q["mu"].idxmax(), "method"]]
        mx_m = METHOD_TEX[q.loc[q["max"].idxmax(), "method"]]
        d = r["Delta (tuned max)"]
        lines.append(f"{PRETTY[a]} & {r['Trivial T2']:.2f} & {r['Best published (mean)']:.2f} ({mu_m}) & "
                     f"{r['Best published (tuned max)']:.2f} ({mx_m}) & {sgn(d)} \\\\")
    (OUT / "paper_table_delta.tex").write_text("\n".join(lines) + "\n")


def sgn(x):
    """Signed number for prose: true minus sign, and 0.00 without a sign."""
    if round(x, 2) == 0:
        return "0.00"
    return (r"\ensuremath{-}" if x < 0 else "+") + f"{abs(x):.2f}"


def numbers(floor, controls, quoted, abl, sens):
    """LaTeX macros for every number quoted in the paper prose."""
    m = {}
    v = floor[(floor["agg"] == "mean") & (floor.family == "masquerade")]
    t2 = v[v.detector == FLOOR_DET]
    ext = t2[t2.attack_type.isin(TYPE_ORDER[:3])]
    rl = t2[t2.attack_type.isin(TYPE_ORDER[3:])]
    m["NExtCaps"] = len(ext)
    m["NRlCaps"] = len(rl)
    m["ExtAUCmin"] = f"{ext.auc_roc.min():.2f}"
    m["RlAUCmin"] = f"{rl.auc_roc.min():.2f}"
    m["RlAUCmax"] = f"{rl.auc_roc.max():.2f}"
    sc = ext[ext.attack_type != "max_engine_coolant_temp_attack"]
    m["SCTPRmin"] = f"{sc.det_rate_at_1pct_fpr.min() * 100:.0f}"
    m["SCTPRmax"] = f"{sc.det_rate_at_1pct_fpr.max() * 100:.0f}"
    fab = floor[(floor["agg"] == "mean") & (floor.family == "fabrication") & (floor.detector == FLOOR_DET)]
    pair = fab.set_index(fab.capture + "_masquerade")["auc_roc"]
    diff = (pair - t2.set_index("capture")["auc_roc"]).abs().dropna()
    m["FabMasqMaxDiffExt"] = f"{diff[ext.capture].max():.2f}"  # extreme-value twins only
    t3 = table3(floor, controls, quoted)
    d = t3["Delta (tuned max)"]
    m["DeltaExtMin"] = sgn(d[[PRETTY[a] for a in TYPE_ORDER[:3]]].min())
    m["DeltaExtMax"] = sgn(d[[PRETTY[a] for a in TYPE_ORDER[:3]]].max())
    m["DeltaRlMin"] = sgn(d[[PRETTY[a] for a in TYPE_ORDER[3:]]].min())
    m["DeltaRlMax"] = sgn(d[[PRETTY[a] for a in TYPE_ORDER[3:]]].max())
    ifc = controls[controls.family == "masquerade"].groupby("attack_type")["auc_roc"].mean()
    m["IFrlOn"] = f"{ifc['reverse_light_on_attack']:.2f}"
    m["IFrlOff"] = f"{ifc['reverse_light_off_attack']:.2f}"
    m["IFspeedo"] = f"{ifc['max_speedometer_attack']:.2f}"
    a = abl[abl.family == "masquerade"]
    ex = a[(a.variant == "T2_excl_target") & a.attack_type.isin(TYPE_ORDER[:3])]
    m["ExclTargetMin"], m["ExclTargetMax"] = f"{ex.auc_roc.min():.2f}", f"{ex.auc_roc.max():.2f}"
    s = sens[(sens.detector == FLOOR_DET) & sens.attack_type.isin(TYPE_ORDER[:3])]
    m["SensExtMin"] = f"{s.auc_roc.min():.2f}"
    caps = __import__("yaml").safe_load(open(ROOT / "configs/captures.yaml"))
    coolant = next(c for c in caps["attacks"] if c["name"] == "max_engine_coolant_temp_attack_masquerade")
    lab = pd.read_parquet(ROOT / f"data/interim/{coolant['name']}.raw.parquet", columns=["label"])
    m["CoolantNinj"] = int(lab.label.sum())
    if (R / "crysys.csv").exists():
        cr = pd.read_csv(R / "crysys.csv")
        cm = cr[(cr.family == "masquerade") & (cr.detector == FLOOR_DET)]
        m["CryNrec"] = cr.recording.nunique()
        m["CryNmasq"] = len(cm)
        m["CryMasqAUC"] = f"{cm.auc_roc.mean():.2f}"
        per = cm.groupby("strategy")["auc_roc"].mean()
        m["CryStratMin"], m["CryStratMax"] = f"{per.min():.2f}", f"{per.max():.2f}"
        m["CryFlagged"] = f"{cm.flagged_share.mean() * 100:.1f}"
        m["CryFrameAUC"] = f"{cm.frame_auc_roc.mean():.2f}"
    tex = "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in m.items()) + "\n"
    (OUT / "numbers.tex").write_text(tex)
    print("\n== numbers.tex\n" + tex)


def paper_table_construction():
    """Per attack type: target AID, bytes written, injected value(s), ambient range of each written
    byte (fit-set ambient captures only) and whether the injected value lies outside it."""
    sys.path.insert(0, str(ROOT))
    from src.io_road import load_captures_cfg
    from src.pipeline import load_ambient, load_experiment_cfg, split_ambient
    caps, exp = load_captures_cfg(), load_experiment_cfg()
    fit, _ = split_ambient(caps, exp)
    amb = load_ambient(fit)
    lines, seen = [], set()
    for a in TYPE_ORDER:
        c = next(x for x in caps["attacks"] if x["attack_type"] == a and x["family"] == "masquerade")
        aid, pat = c["target_aid"], c["injection_data"]
        g = amb[amb.aid == aid]
        cells = []
        for j in range(8):
            hx = pat[2 * j: 2 * j + 2]
            if hx == "XX":
                continue
            val, col = int(hx, 16), g[f"b{j}"]
            lo, hi, nd = int(col.min()), int(col.max()), col.nunique()
            out = val < lo or val > hi
            rng = "{" + ",".join(f"{x}" for x in sorted(col.unique())) + "}" if nd <= 3 else f"[{lo},{hi}]"
            cells.append(f"B{j}={val}" + (r"$^\ast$" if out else "") + f" {rng}")
        if a == "correlated_signal_attack":  # full 8-byte write; show only the high bytes that matter
            cells = [x for x in cells if x[:2] in ("B0", "B2", "B4", "B6")]
        lines.append(f"{PRETTY[a]} & 0x{aid:X} & " + "; ".join(cells).replace("{", r"\{").replace("}", r"\}")
                     + r" \\")
    (OUT / "paper_table_construction.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def deep_section():
    """Per-ID table (ROAD / ROAD-IR / benign TWIN) and macros for the deep-model section."""
    import glob
    m = {}
    t2 = pd.read_csv(R / "t2_perid.csv").groupby(["attack_type", "variant"])["auc_roc"].mean()
    sup = pd.read_csv(R / "deep_sup.csv")
    sup = sup.groupby(["attack_type", "variant", "seed"])["auc_roc"].mean().groupby(["attack_type", "variant"]).mean()
    cs = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "ctx_scoring_*.csv")))], ignore_index=True)
    per = cs[cs.rule == "perid_tail"]
    ctx = per[per["mode"] == "ctx"].groupby(["attack_type", "variant", "seed"])["auc_roc"].mean()
    ctx_sd = ctx.groupby(["attack_type", "variant"]).std()
    ctx = ctx.groupby(["attack_type", "variant"]).mean()
    pae = per[per["mode"] == "pae"].groupby(["attack_type", "variant"])["auc_roc"].mean() if (per["mode"] == "pae").any() else None
    lines = []
    for a in TYPE_ORDER:
        cells = []
        for v in ("ROAD", "ROAD-IR", "TWIN"):
            vals = [t2[(a, v)], sup[(a, v)], pae[(a, v)] if pae is not None else np.nan, ctx[(a, v)]]
            cells += ["--" if np.isnan(x) else (rf"\textbf{{{x:.2f}}}" if (v != "TWIN" and x >= 0.95) else f"{x:.2f}") for x in vals]
        lines.append(f"{PRETTY[a]} & " + " & ".join(cells) + r" \\")
    (OUT / "paper_table_deep.tex").write_text("\n".join(lines) + "\n")
    n_seeds = per[per["mode"] == "ctx"].seed.nunique()
    m["CtxSeeds"] = n_seeds
    m["CtxRoadMin"] = f"{min(ctx[(a, 'ROAD')] for a in TYPE_ORDER):.2f}"
    m["CtxIRMin"] = f"{min(ctx[(a, 'ROAD-IR')] for a in TYPE_ORDER):.2f}"
    four = [a for a in TYPE_ORDER if a != "max_engine_coolant_temp_attack"]
    m["CtxFourMin"] = f"{min(min(ctx[(a, 'ROAD')], ctx[(a, 'ROAD-IR')]) for a in four):.2f}"
    cs_cool = per[(per["mode"] == "ctx") & (per.attack_type == "max_engine_coolant_temp_attack") & (per.variant == "ROAD")]
    m["CtxCoolMean"] = f"{cs_cool.auc_roc.mean():.2f}"
    m["CtxCoolMin"] = f"{cs_cool.auc_roc.min():.2f}"
    m["CtxCoolMax"] = f"{cs_cool.auc_roc.max():.2f}"
    m["CtxRlMin"] = f"{min(ctx[(a, 'ROAD')] for a in TYPE_ORDER[3:]):.2f}"
    m["CtxTwinMin"] = f"{min(ctx[(a, 'TWIN')] for a in TYPE_ORDER):.2f}"
    m["CtxTwinMax"] = f"{max(ctx[(a, 'TWIN')] for a in TYPE_ORDER):.2f}"
    m["CtxSdMax"] = f"{np.nanmax(ctx_sd.values):.2f}" if n_seeds > 1 else "--"
    m["TtwoIRMax"] = f"{max(t2[(a, 'ROAD-IR')] for a in TYPE_ORDER[:3]):.2f}"
    m["SupRoadMin"] = f"{min(sup[(a, 'ROAD')] for a in TYPE_ORDER):.2f}"
    m["SupRoadMax"] = f"{max(sup[(a, 'ROAD')] for a in TYPE_ORDER):.2f}"
    tw = sorted(sup[(a, "TWIN")] for a in TYPE_ORDER)
    m["SupTwinHiN"] = sum(x >= 0.7 for x in tw)
    m["SupTwinHiMin"] = f"{min(x for x in tw if x >= 0.7):.2f}"
    m["SupTwinMax"] = f"{max(tw):.2f}"
    m["SupRlOffTwin"] = f"{sup[('reverse_light_off_attack', 'TWIN')]:.2f}"
    m["SupRlOffRoad"] = f"{sup[('reverse_light_off_attack', 'ROAD')]:.2f}"
    if pae is not None:
        m["PaeRlMax"] = f"{max(pae[(a, 'ROAD')] for a in TYPE_ORDER[3:]):.2f}"
        m["PaeIRMin"] = f"{min(pae[(a, 'ROAD-IR')] for a in TYPE_ORDER[:3]):.2f}"
        m["PaeIRMax"] = f"{max(pae[(a, 'ROAD-IR')] for a in TYPE_ORDER[:3]):.2f}"
    if (R / "deep_sup_trainIR.csv").exists():
        si = pd.read_csv(R / "deep_sup_trainIR.csv").groupby(["attack_type", "variant"])["auc_roc"].mean()
        hi = [a for a in TYPE_ORDER if sup[(a, "TWIN")] >= 0.7]  # same types as the ROAD-trained model
        m["SupIRTwinMin"] = f"{min(si[(a, 'TWIN')] for a in hi):.2f}"
        m["SupIRTwinMax"] = f"{max(si[(a, 'TWIN')] for a in hi):.2f}"
    tpr = per[(per["mode"] == "ctx") & (per.variant == "ROAD")].groupby("attack_type")["det_rate_at_1pct_fpr"].mean()
    m["CtxTprSC"] = f"{min(tpr['max_speedometer_attack'], tpr['correlated_signal_attack']) * 100:.0f}"
    bus = cs[(cs.rule == "bus_calib") & (cs["mode"] == "ctx") & (cs.variant == "ROAD")].groupby("attack_type")["auc_roc"].mean()
    m["BusMax"] = f"{bus.max():.2f}"
    tex = "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in m.items()) + "\n"
    (OUT / "numbers_deep.tex").write_text(tex)
    print(tex)


def main():
    paper_table_construction()
    if (R / "deep_sup.csv").exists() and (R / "t2_perid.csv").exists():
        deep_section()
    floor = pd.read_csv(R / "floor.csv")
    quoted = pd.read_csv(ROOT / "data/published/moriano2024_table2.csv", comment="#")
    write(table2(floor), "table2_floor")
    if (R / "controls.csv").exists():
        controls = pd.read_csv(R / "controls.csv")
        write(table3(floor, controls, quoted), "table3_delta")
        paper_table_floor(floor, controls)
        paper_table_delta(floor, controls, quoted)
    if (R / "ablation.csv").exists():
        abl = pd.read_csv(R / "ablation.csv")
        write(table4(abl), "table4_ablation")
        numbers(floor, controls, quoted, abl, pd.read_csv(R / "sensitivity.csv"))


if __name__ == "__main__":
    sys.exit(main())
