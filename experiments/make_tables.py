"""Project 2: results/*.csv -> paper tables (LaTeX bodies) and number macros. Nothing typed by hand.

python experiments/make_tables.py
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
R, OUT = ROOT / "results", ROOT / "results" / "tables"
SETTINGS = ["Within-OLD", "Within-NEW", "OLD->NEW", "NEW->OLD", "Within-SHADOW", "HCRL->SHADOW",
            "SHADOW->OLD", "SHADOW->NEW"]
SET_TEX = {"Within-OLD": "W-OLD", "Within-NEW": "W-NEW", "OLD->NEW": r"O$\to$N", "NEW->OLD": r"N$\to$O",
           "Within-SHADOW": "W-SH", "HCRL->SHADOW": r"H$\to$S", "SHADOW->OLD": r"S$\to$O",
           "SHADOW->NEW": r"S$\to$N"}
MODELS = ["M1_LSTM", "M2_CNN", "M3_AE", "M4_NGRAM", "R1_RATE", "R2_HIST"]
MOD_TEX = {"GALAXY": "GALAXY$^\\dagger$", "GALAXY128": "GALAXY$^\\dagger$ ($L$=128)", "TF": "Transformer",
           "DANN_LSTM": "DANN-LSTM", "M1_LSTM": "LSTM", "M2_CNN": "1D-CNN", "M3_AE": "LSTM-AE", "M4_NGRAM": "n-gram LR",
           "R1_RATE": "Rate (trivial)", "R2_HIST": "Histogram (trivial)",
           "M1_LSTM+dt": "LSTM + $\\Delta t$", "M1_LSTM+dtz": "LSTM + $\\Delta t$ (z)"}


def sgn(x):
    if round(x, 2) == 0:
        return "0.00"
    return (r"\ensuremath{-}" if x < 0 else "+") + f"{abs(x):.2f}"


def cell(m, s, stoch):
    return f"{m:.2f}" + (rf"\,{{\scriptsize$\pm${s:.2f}}}" if stoch else "")


def matrix_table(df):
    v = df[(df.attack == "all") & (df.vocab == "train")]
    g = v.groupby(["model", "setting"])["auc_roc"].agg(["mean", "std"])
    lines = []
    for m in [x for x in ["GALAXY", "GALAXY128", "TF"] if x in g.index.get_level_values(0)] + MODELS:
        stoch = m in ("M1_LSTM", "M2_CNN", "M3_AE", "GALAXY", "GALAXY128", "TF")
        cells = [cell(g.loc[(m, s), "mean"], g.loc[(m, s), "std"], stoch) for s in SETTINGS]
        if m in ("M1_LSTM", "R1_RATE"):
            lines.append(r"\midrule")
        lines.append(" & ".join([MOD_TEX[m]] + cells) + r" \\")
    (OUT / "paper_table_matrix.tex").write_text("\n".join(lines) + "\n")
    return g["mean"].unstack("setting")[SETTINGS]


def attack_table(df):
    """Per-attack AUC, within-target vs cross-to-target, HCRL campaigns."""
    v = df[(df.vocab == "train") & df.attack.isin(["heartbeat", "ping", "request"])]
    g = v.groupby(["model", "setting", "attack"])["auc_roc"].mean()
    lines = []
    for m in MODELS:
        cells = []
        for tgt, src in (("Within-NEW", "OLD->NEW"), ("Within-OLD", "NEW->OLD")):
            for a in ("heartbeat", "ping", "request"):
                cells.append(f"{g[(m, tgt, a)]:.2f}/{g[(m, src, a)]:.2f}")
        if m == "R1_RATE":
            lines.append(r"\midrule")
        lines.append(" & ".join([MOD_TEX[m]] + cells) + r" \\")
    (OUT / "paper_table_attack.tex").write_text("\n".join(lines) + "\n")


def ablation_table():
    rows = []
    if (R / "ablation_timing.csv").exists():
        t = pd.read_csv(R / "ablation_timing.csv")
        t = t[t.attack == "all"].groupby(["model", "setting"])["auc_roc"].mean().unstack("setting")
        for m in ["M1_LSTM", "M1_LSTM+dt", "M1_LSTM+dtz"]:
            if m in t.index:
                rows.append(" & ".join([MOD_TEX[m]] + [f"{t.loc[m, s]:.2f}" for s in SETTINGS]) + r" \\")
    (OUT / "paper_table_ablation.tex").write_text("\n".join(rows) + "\n")


def numbers(df, mean):
    m = {}
    sup = ["M1_LSTM", "M2_CNN", "M4_NGRAM"]
    m["WithinSupMin"] = f"{mean.loc[sup, ['Within-OLD', 'Within-NEW']].min().min():.2f}"
    m["CrossHCRLSupMin"] = f"{mean.loc[sup, ['OLD->NEW', 'NEW->OLD']].min().min():.2f}"
    m["CrossHCRLSupMax"] = f"{mean.loc[sup, ['OLD->NEW', 'NEW->OLD']].max().max():.2f}"
    m["WithinShadowSupMin"] = f"{mean.loc[sup, 'Within-SHADOW'].min():.2f}"
    m["WithinShadowSupMax"] = f"{mean.loc[sup, 'Within-SHADOW'].max():.2f}"
    sh = ["HCRL->SHADOW", "SHADOW->OLD", "SHADOW->NEW"]
    m["CrossShSupMin"] = f"{mean.loc[sup, sh].min().min():.2f}"
    m["CrossShSupMax"] = f"{mean.loc[sup, sh].max().max():.2f}"
    m["CrossShTrivMin"] = f"{mean.loc[['R1_RATE', 'R2_HIST'], sh].min().min():.2f}"
    m["CrossShTrivMax"] = f"{mean.loc[['R1_RATE', 'R2_HIST'], sh].max().max():.2f}"
    m["RateNtoO"] = f"{mean.loc['R1_RATE', 'NEW->OLD']:.2f}"
    m["LSTMNtoO"] = f"{mean.loc['M1_LSTM', 'NEW->OLD']:.2f}"
    gap = mean[["Within-NEW"]].values - mean[["OLD->NEW"]].values
    m["GapOtoNLSTM"] = f"{(mean.loc['M1_LSTM', 'Within-NEW'] - mean.loc['M1_LSTM', 'OLD->NEW']):.2f}"
    m["GapNtoOLSTM"] = f"{(mean.loc['M1_LSTM', 'Within-OLD'] - mean.loc['M1_LSTM', 'NEW->OLD']):.2f}"
    v = df[(df.vocab == "train")]
    a = v.groupby(["model", "setting", "attack"])["auc_roc"].mean()
    m["HbWithinNewSup"] = f"{min(a[(x, 'Within-NEW', 'heartbeat')] for x in sup):.2f}"
    m["HbWithinNewTrivMax"] = f"{max(a[(x, 'Within-NEW', 'heartbeat')] for x in ['R1_RATE', 'R2_HIST', 'M3_AE']):.2f}"
    m["HbOtoNLSTM"] = f"{a[('M1_LSTM', 'OLD->NEW', 'heartbeat')]:.2f}"
    m["HbOtoNCNN"] = f"{a[('M2_CNN', 'OLD->NEW', 'heartbeat')]:.2f}"
    m["HbOtoNNgram"] = f"{a[('M4_NGRAM', 'OLD->NEW', 'heartbeat')]:.2f}"
    oov = v[v.attack == "all"].groupby("setting").oov_rate.mean()
    m["OOVShMin"] = f"{oov[sh].min() * 100:.0f}"
    m["OOVShMax"] = f"{oov[sh].max() * 100:.0f}"
    m["OOVOtoN"] = f"{oov['OLD->NEW'] * 100:.0f}"
    e = pd.read_csv(R / "errors.csv").set_index(["campaign", "attack"])
    m["JsdOldNewNormal"] = f"{e.loc[('NEW', 'normal'), 'jsd_to_OLD_normal']:.3f}"
    m["JsdNewHb"] = f"{e.loc[('NEW', 'heartbeat'), 'jsd_to_NEW_normal']:.3f}"
    m["JsdOldHb"] = f"{e.loc[('OLD', 'heartbeat'), 'jsd_to_OLD_normal']:.3f}"
    m["RateNewHb"] = f"{e.loc[('NEW', 'heartbeat'), 'rate_fps']:.0f}"
    m["RateNewNormal"] = f"{e.loc[('NEW', 'normal'), 'rate_fps']:.0f}"
    m["RateOldHb"] = f"{e.loc[('OLD', 'heartbeat'), 'rate_fps']:.0f}"
    m["RateOldNormal"] = f"{e.loc[('OLD', 'normal'), 'rate_fps']:.0f}"
    m["RateShNormal"] = f"{e.loc[('SHADOW', 'normal'), 'rate_fps']:.0f}"
    m["RateShRelay"] = f"{e.loc[('SHADOW', 'relay'), 'rate_fps']:.0f}"
    m["JsdShHcrl"] = f"{e.loc[('SHADOW', 'normal'), 'jsd_to_OLD_normal']:.2f}"
    if (R / "recording_control.csv").exists():
        rc = pd.read_csv(R / "recording_control.csv").groupby(["task", "model"])["auc_roc"].mean()
        bb = "benign-vs-benign (OLD, temporal split)"
        m["BenignBenignMin"] = f"{min(rc[(bb, x)] for x in sup):.2f}"
        m["BenignBenignMax"] = f"{max(rc[(bb, x)] for x in sup):.2f}"
    if (R / "ablation_timing.csv").exists():
        t = pd.read_csv(R / "ablation_timing.csv")
        t = t[t.attack == "all"].groupby(["model", "setting"])["auc_roc"].mean()
        for k, nm in (("M1_LSTM+dt", "Dt"), ("M1_LSTM+dtz", "Dtz")):
            if (k, "HCRL->SHADOW") in t.index:
                m[f"LSTM{nm}HtoS"] = f"{t[(k, 'HCRL->SHADOW')]:.2f}"
                m[f"LSTM{nm}NtoO"] = f"{t[(k, 'NEW->OLD')]:.2f}"
                m[f"LSTM{nm}OtoN"] = f"{t[(k, 'OLD->NEW')]:.2f}"
    import sys
    sys.path.insert(0, str(ROOT))
    from src.inventory import load, load_cfg
    caps = load_cfg()["captures"]
    for camp, key in (("OLD", "Old"), ("NEW", "New"), ("SHADOW", "Sh")):
        cs = [c for c in caps if c["campaign"] == camp]
        ids = set()
        for c in cs:
            ids |= set(load(c["name"]).msgid.unique().tolist())
        m[f"Vocab{key}"] = len(ids)
        m[f"Files{key}"] = len(cs)
    if (R / "matrix_gpu.csv").exists():
        gp = pd.read_csv(R / "matrix_gpu.csv")
        gp = gp[gp.attack == "all"].groupby(["model", "setting"])["auc_roc"].mean()
        cross = ["OLD->NEW", "NEW->OLD", "HCRL->SHADOW", "SHADOW->OLD", "SHADOW->NEW"]
        m["GalWSh"] = f"{gp[('GALAXY', 'Within-SHADOW')]:.3f}"
        m["GalCrossMin"] = f"{min(gp[('GALAXY', c)] for c in cross):.2f}"
        m["GalCrossMax"] = f"{max(gp[('GALAXY', c)] for c in cross):.2f}"
        m["TFCrossMax"] = f"{max(gp[('TF', c)] for c in cross):.2f}"
        for c, k in (("NEW->OLD", "NtoO"), ("OLD->NEW", "OtoN"), ("SHADOW->OLD", "StoO"), ("HCRL->SHADOW", "HtoS")):
            m[f"Dann{k}"] = f"{gp[('DANN_LSTM', c)]:.2f}"
            m[f"Lstm{k}"] = f"{mean.loc['M1_LSTM', c]:.2f}"
    if (R / "matrix_galaxy128.csv").exists():
        g1 = pd.read_csv(R / "matrix_galaxy128.csv")
        g1 = g1[g1.attack == "all"].groupby("setting")["auc_roc"].mean()
        m["GalLWSh"] = f"{g1['Within-SHADOW']:.2f}"
        m["GalLCrossMax"] = f"{max(g1[c] for c in ['OLD->NEW', 'NEW->OLD', 'HCRL->SHADOW', 'SHADOW->OLD', 'SHADOW->NEW']):.2f}"
    if (R / "adapt.csv").exists():
        ad = pd.read_csv(R / "adapt.csv").groupby(["setting", "model", "mode", "N_s"])["auc_roc"].mean()
        m["AdHistHtoSsrc"] = f"{ad[('HCRL->SHADOW', 'R2_HIST', 'source', 0)]:.2f}"
        m["AdHistHtoSfifteen"] = f"{ad[('HCRL->SHADOW', 'R2_HIST', 'target', 15)]:.2f}"
        m["AdRateStoOsrc"] = f"{ad[('SHADOW->OLD', 'R1_RATE', 'source', 0)]:.2f}"
        m["AdRateStoOfifteen"] = f"{ad[('SHADOW->OLD', 'R1_RATE', 'target', 15)]:.2f}"
        m["AdAEStoOsrc"] = f"{ad[('SHADOW->OLD', 'M3_AE', 'source', 0)]:.2f}"
        m["AdAEStoOfifteen"] = f"{ad[('SHADOW->OLD', 'M3_AE', 'target', 15)]:.2f}"
        m["AdAEHtoStarget"] = f"{max(ad[('HCRL->SHADOW', 'M3_AE', 'target', n)] for n in (15, 30, 60, 120)):.2f}"
    voc = {}
    for camp in ("OLD", "NEW", "SHADOW"):
        voc[camp] = set()
        for c in [c for c in caps if c["campaign"] == camp]:
            voc[camp] |= set(load(c["name"]).msgid.unique().tolist())
    m["SharedOldSh"] = len(voc["OLD"] & voc["SHADOW"])
    m["SharedNewSh"] = len(voc["NEW"] & voc["SHADOW"])
    u = df[df.attack == "all"].groupby(["vocab", "model", "setting"])["auc_roc"].mean().unstack("vocab")
    du = (u["union"] - u["train"]).loc["M1_LSTM"]
    m["UnionLSTMMaxAbs"] = f"{du.abs().max():.2f}"
    m["LSTMWSh"] = f"{mean.loc['M1_LSTM', 'Within-SHADOW']:.2f}"
    if (R / "ablation_timing.csv").exists():
        m["LSTMDtWSh"] = f"{t[('M1_LSTM+dt', 'Within-SHADOW')]:.2f}"
    tex = "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in m.items()) + "\n"
    (OUT / "numbers.tex").write_text(tex)
    print(tex)


def adapt_table():
    """Proposed fix vs DANN: AUC on each cross setting."""
    a = pd.read_csv(R / "adapt.csv").groupby(["setting", "model", "mode", "N_s"])["auc_roc"].mean()
    mx = pd.read_csv(R / "matrix.csv")
    within = mx[(mx.attack == "all") & (mx.vocab == "train")].groupby(["model", "setting"])["auc_roc"].mean()
    gpu = pd.read_csv(R / "matrix_gpu.csv")
    gpu = gpu[gpu.attack == "all"].groupby(["model", "setting"])["auc_roc"].mean()
    wmap = {"OLD->NEW": "Within-NEW", "NEW->OLD": "Within-OLD", "HCRL->SHADOW": "Within-SHADOW",
            "SHADOW->OLD": "Within-OLD", "SHADOW->NEW": "Within-NEW"}
    lines = []
    for st in ["OLD->NEW", "NEW->OLD", "HCRL->SHADOW", "SHADOW->OLD", "SHADOW->NEW"]:
        # one pre-specified one-class detector (Histogram); no selection on test results
        best_src = a[(st, "R2_HIST", "source", 0)]
        best15 = a[(st, "R2_HIST", "target", 15)]
        best60 = a[(st, "R2_HIST", "target", 60)]
        w = within[("R2_HIST", wmap[st])]
        lstm, dann = within[("M1_LSTM", st)], gpu[("DANN_LSTM", st)]
        lines.append(f"{SET_TEX[st]} & {lstm:.2f} & {dann:.2f} & {best_src:.2f} & \\textbf{{{best15:.2f}}} & {best60:.2f} & {w:.2f} \\\\")
    (OUT / "paper_table_adapt.tex").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(R / "matrix.csv")
    extra = [pd.read_csv(R / f) for f in ("matrix_gpu.csv",) if (R / f).exists()]
    if (R / "matrix_galaxy128.csv").exists():
        g = pd.read_csv(R / "matrix_galaxy128.csv"); g["model"] = "GALAXY128"; g["vocab"] = "train"; extra.append(g)
    df_all = pd.concat([df] + [e.assign(vocab=e.get("vocab", "train")) for e in extra], ignore_index=True)
    mean_all = matrix_table(df_all)
    mean = mean_all.loc[MODELS]
    print(mean_all.round(2).to_string())
    adapt_table()
    attack_table(df)
    ablation_table()
    numbers(df, mean)


if __name__ == "__main__":
    main()
