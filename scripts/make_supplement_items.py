"""Write paper/supplement_items.md from its template and the result tables (no numbers are typed by hand).

    python scripts/make_supplement_items.py

The template paper/supplement_items_template.md holds the captions, notes and hand-written blocks. Each line of the
form <!-- TABLE:name --> is replaced by a Markdown table built here from outputs/tables/*.csv.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
TEMPLATE = ROOT / "paper" / "supplement_items_template.md"
OUT = ROOT / "paper" / "supplement_items.md"

MODEL = {"rule": "Rule (RMSSD/RR)", "logreg": "Logistic regression", "lgbm": "Gradient boosting", "cnn": "1-D CNN"}
MODELS = list(MODEL)
NOISE = {"bw": "Baseline wander", "ma": "Muscle artifact", "em": "Electrode motion", "none": "None (clean)"}
DET = {"neurokit": "NeuroKit2 (study)", "pantompkins1985": "Pan–Tompkins", "hamilton2002": "Hamilton",
       "elgendi2010": "Elgendi", "engzeemod2012": "Engelse–Zeelenberg", "kalidas2017": "Kalidas–Tamil",
       "rodrigues2021": "Rodrigues"}
MEAS = {"avgqrs": "Average QRS (NeuroKit2)", "zhao": "Zhao fuzzy index", "ksqi": "Kurtosis (kSQI)",
        "psqi": "Relative QRS power (pSQI)", "bassqi": "Baseline power (basSQI)", "bsqi": "Two-detector agreement",
        "learned": "Learned measure"}


def f(v, nd=3):
    return "–" if pd.isna(v) else f"{v:.{nd}f}".replace("-", "−")


def ci(v, lo, hi, nd=3):
    return f"{f(v, nd)} [{f(lo, nd)}–{f(hi, nd)}]".replace("[−", "[−")


def snr(s):
    return "clean" if pd.isna(s) or s == float("inf") else (f"{s:+g} dB".replace("+0 dB", "0 dB").replace("-", "−"))


def cond_label(c: str) -> str:
    if c.endswith("clean"):
        return "Clean"
    m = re.search(r"__(bw|ma|em)__([+-]?\d+)dB", c)
    return f"{NOISE[m.group(1)]}, {snr(float(m.group(2)))}" if m else c


def lc(t: str) -> str:
    """Lower-case only the first letter (keeps units such as dB intact)."""
    return t[:1].lower() + t[1:]


def md(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def snr_order(c: str) -> tuple:
    if c.endswith("clean"):
        return (-1, 0)
    m = re.search(r"__(bw|ma|em)__([+-]?\d+)dB", c)
    return (["bw", "ma", "em"].index(m.group(1)), -int(m.group(2)))


# ----------------------------------------------------------------------------------------------- tables
def external_grid():
    d = pd.read_csv(T / "external_noise.csv")
    d = d[d.variant == "standard"]
    rows = []
    for c in sorted(d.condition.unique(), key=snr_order):
        for m in MODELS:
            r = d[(d.condition == c) & (d.model == m)].iloc[0]
            rows.append([cond_label(c) if m == "rule" else "", MODEL[m], ci(r.auroc, r.auroc_lo, r.auroc_hi),
                         f(r.sensitivity), f(r.specificity), f(r.ppv), f(r.npv)])
    return md(rows, ["Condition", "Model", "AUROC [95% CI]", "Sensitivity", "Specificity", "PPV", "NPV"])


def rhythm():
    d = pd.read_csv(T / "external_by_rhythm.csv")
    d = d[d.variant == "standard"]
    order = ["AFIB", "SR", "AFLT", "STACH", "SBRAD", "SARRH", "SVARR", "SVTAC", "PSVT", "BIGU", "TRIGU", "PVC", "PAC",
             "PACE"]
    rows = []
    for cat in order:
        g = d[d.category == cat]
        n = int(g.n.iloc[0])
        vals = [f(g[(g.condition == c) & (g.model == m)].share_called_af.iloc[0]) for c in
                ["external__clean", "external__em__-6dB__r0"] for m in MODELS]
        rows.append([cat, f"{n:,}"] + vals)
    hdr = ["Statement", "n"] + [f"{MODEL[m]}, {k}" for k in ["clean", "EM −6 dB"] for m in MODELS]
    return md(rows, hdr)


def variance():
    v = pd.read_csv(T / "rev_variance.csv")
    v = v[(v.variant == "standard") & (v.noise.isin(["ma", "em"])) & (v.snr_db.isin([0.0, -6.0]))]
    rows = []
    v = v.assign(_n=v.noise.map({"ma": 0, "em": 1}), _m=v.model.map({m: i for i, m in enumerate(MODELS)}))
    for _, r in v.sort_values(["_n", "snr_db", "_m"], ascending=[True, False, True]).iterrows():
        rows.append([NOISE[r.noise], snr(r.snr_db), MODEL[r.model], int(r.n_realizations),
                     f(r.sensitivity), f(r.sensitivity_sd_realization), f(r.sensitivity_sd_seed),
                     f(r.sensitivity_se_sample), f(r.specificity), f(r.specificity_sd_realization),
                     f(r.specificity_sd_seed), f(r.specificity_se_sample)])
    return md(rows, ["Noise", "SNR", "Model", "Draws", "Sens.", "SD draws", "SD seeds", "SE sample", "Spec.",
                     "SD draws", "SD seeds", "SE sample"])


def altnoise():
    a = pd.read_csv(T / "rev_altnoise.csv")
    a = a[a.variant == "standard"]
    strat = ["power-based (main)", "qrsamp", "burst"]
    rows = []
    for n in ["ma", "em"]:
        for s in [0.0, -6.0]:
            for m in MODELS:
                g = a[(a.noise == n) & (a.snr_db == s) & (a.model == m)].set_index("strategy")
                rows.append([NOISE[n], snr(s), MODEL[m]] + [f"{f(g.loc[k].sensitivity)} / {f(g.loc[k].specificity)}"
                                                            for k in strat])
    return md(rows, ["Noise", "SNR", "Model", "Power-based (main)", "QRS-amplitude scaling", "Bursts"])


def rpeak():
    d = pd.read_csv(T / "rpeak_validation.csv")
    rows = []
    for (n, s) in [("none", float("inf")), ("em", -6.0), ("ma", -6.0)]:
        for det in DET:
            r = d[(d.rhythm == "all") & (d.noise == n) & (d.snr_db == s) & (d.detector == det)].iloc[0]
            rows.append(["All" if (n == "none" and det == "neurokit") else "", NOISE[n] if det == "neurokit" else "", "" if n == "none" else (snr(s) if det == "neurokit" else ""),
                         DET[det], f(r.sensitivity), f(r.ppv), ci(r.f1, r.f1_lo, r.f1_hi),
                         f(r.missed_per_min, 1), f(r.spurious_per_min, 1), f(r.mean_abs_offset_ms, 0)])
    for rh, lab in [("N", "Sinus rhythm"), ("AF", "AF"), ("other", "Other rhythm")]:
        for (n, s) in [("none", float("inf")), ("em", -6.0), ("ma", -6.0)]:
            r = d[(d.rhythm == rh) & (d.noise == n) & (d.snr_db == s) & (d.detector == "neurokit")].iloc[0]
            rows.append([lab if n == "none" else "", NOISE[n], "" if n == "none" else snr(s), DET["neurokit"], f(r.sensitivity), f(r.ppv),
                         ci(r.f1, r.f1_lo, r.f1_hi), f(r.missed_per_min, 1), f(r.spurious_per_min, 1),
                         f(r.mean_abs_offset_ms, 0)])
    return md(rows, ["Rhythm", "Noise", "SNR", "Detector", "Sensitivity", "PPV", "F1 [95% CI]", "Missed/min",
                     "Spurious/min", "Timing error, ms"])


def pseudo():
    d = pd.read_csv(T / "rpeak_pseudo_reference.csv")
    rows = [[NOISE[r.noise], "" if r.noise == "none" else snr(r.snr_db), f(r.f1_expert), f(r.f1_pseudo),
             f(r.f1_pseudo_minus_expert), f(r.window_f1_corr, 2), f(r.frac_windows_f1_below_0_9_expert if
                                                                 "frac_windows_f1_below_0_9_expert" in r else
                                                                 r["frac_windows_f1_below_0.9_expert"])]
            for _, r in d.iterrows()]
    return md(rows, ["Noise", "SNR", "F1 vs expert", "F1 vs clean-signal detections", "Difference",
                     "Window-level correlation", "Windows with F1 < 0.9 (expert)"])


def oracle():
    o = pd.read_csv(T / "rev_oracle.csv")
    o = o[o.variant == "standard"]
    rows = []
    for n in ["em", "ma"]:
        for s in [0.0, -6.0]:
            for m in ["rule", "logreg", "lgbm"]:
                r = o[(o.noise == n) & (o.snr_db == s) & (o.model == m)].iloc[0]
                rows.append([NOISE[n] if m == "rule" else "", snr(s) if m == "rule" else "", MODEL[m],
                             f"{f(r.sens_detected_peaks)} / {f(r.spec_detected_peaks)}",
                             f"{f(r.sens_clean_peaks)} / {f(r.spec_clean_peaks)}",
                             ci(r.spec_gain, r.spec_gain_lo, r.spec_gain_hi), ci(r.sens_gain, r.sens_gain_lo, r.sens_gain_hi),
                             f"{f(r.auroc_detected_peaks)} → {f(r.auroc_clean_peaks)}"])
    return md(rows, ["Noise", "SNR", "Model", "Detected peaks: sens. / spec.", "Clean-signal peaks: sens. / spec.",
                     "Specificity gain [95% CI]", "Sensitivity change [95% CI]", "AUROC"])


def detector():
    d = pd.read_csv(T / "detector_ablation.csv")
    rows = []
    for det in ["hamilton2002", "elgendi2010", "pantompkins1985"]:
        for m in ["rule", "logreg", "lgbm"]:
            g = d[(d.detector == det) & (d.model == m)].set_index("condition")
            cells = []
            for c in ["internal__clean", "internal__em__+0dB__r0", "internal__em__-6dB__r0", "internal__ma__-6dB__r0"]:
                r = g.loc[c]
                cells.append(f"{f(r.sensitivity)} / {f(r.specificity)}")
            rows.append([DET[det] if m == "rule" else "", MODEL[m]] + cells + [f(g.loc["internal__em__-6dB__r0"].auroc)])
    return md(rows, ["Detector", "Model", "Clean", "EM 0 dB", "EM −6 dB", "MA −6 dB", "AUROC, EM −6 dB"])


def regional():
    r = pd.read_csv(T / "rev_regional.csv")
    r = r[r.variant == "standard"]
    lab = {"full": "Full", "qrs_only": "Within QRS only", "nonqrs_only": "Outside QRS only"}
    rows = []
    for s in [-6.0]:
        for m in MODELS:
            for reg in ["full", "qrs_only", "nonqrs_only"]:
                x = r[(r.snr_db == s) & (r.model == m) & (r.region == reg)].iloc[0]
                rows.append([snr(s) if (m == "rule" and reg == "full") else "", MODEL[m] if reg == "full" else "",
                             lab[reg], f(x.auroc), ci(x.sensitivity, x.sensitivity_lo, x.sensitivity_hi),
                             ci(x.specificity, x.specificity_lo, x.specificity_hi)])
    return md(rows, ["EM SNR", "Model", "Noise region", "AUROC", "Sensitivity", "Specificity"])


def cnn():
    c = pd.read_csv(T / "rev_cnn.csv")
    c = c[(c.variant == "standard") & (c.condition.str.contains("clean") | c.condition.str.contains(r"\+6dB|\+0dB|-6dB"))]
    rows = []
    for _, r in c.sort_values(["variant", "condition"], key=lambda s: s.map(snr_order) if s.name == "condition" else
                              s.map({"standard": 0, "augmented": 1})).iterrows():
        rows.append([cond_label(r.condition),
                     f(r.mean_prob_af), f(r.mean_prob_non_af), f(r.auroc), f"{f(r.sens_fixed)} / {f(r.spec_fixed)}",
                     "–" if pd.isna(r.sens_noise_threshold) else f"{f(r.sens_noise_threshold)} / {f(r.spec_noise_threshold)}",
                     f(r.sens_at_clean_specificity_diagnostic)])
    return md(rows, ["Condition", "Mean prob., AF", "Mean prob., non-AF", "AUROC",
                     "Fixed threshold: sens. / spec.", "Noise-specific threshold: sens. / spec.",
                     "Sens. at clean specificity (diagnostic)"])


def afdb():
    a = pd.read_csv(T / "afdb_by_quality.csv").drop_duplicates(["variant", "model", "condition", "quality_fifth"])
    rows = []
    for v in ["standard"]:
        nat = a[(a.variant == v) & (a.condition == "natural")]
        for q in ["Q1 (noisiest)", "Q2", "Q3", "Q4", "Q5 (cleanest)"]:
            for m in MODELS:
                r = nat[(nat.quality_fifth == q) & (nat.model == m)].iloc[0]
                rows.append([f"Natural, {q}" if m == "rule" else "", MODEL[m],
                             f"{int(r.n_windows):,} ({int(r.n_af):,})", f(r.auroc),
                             ci(r.sensitivity, r.sensitivity_lo, r.sensitivity_hi),
                             ci(r.specificity, r.specificity_lo, r.specificity_hi)])
        for c in ["additive_em_+0dB", "additive_em_-6dB", "additive_ma_+0dB", "additive_ma_-6dB"]:
            for m in MODELS:
                r = a[(a.variant == v) & (a.condition == c) & (a.model == m)].iloc[0]
                n, s = c.split("_")[1], c.split("_")[2].replace("dB", "")
                rows.append([f"Q5 + {NOISE[n].lower()}, {snr(float(s))}" if m == "rule" else "", MODEL[m], f"{int(r.n_windows):,} ({int(r.n_af):,})", f(r.auroc),
                             ci(r.sensitivity, r.sensitivity_lo, r.sensitivity_hi),
                             ci(r.specificity, r.specificity_lo, r.specificity_hi)])
    return md(rows, ["Windows", "Model", "n (AF)", "AUROC", "Sensitivity", "Specificity"])


def augablation():
    a = pd.read_csv(T / "rev_augablation.csv")
    cfgs = ["none (standard)", "full", "no_em", "no_ma", "no_bw", "mild", "severe"]
    clab = {"none (standard)": "None (standard)", "full": "Full (refit)", "no_em": "Without EM", "no_ma": "Without MA",
            "no_bw": "Without BW", "mild": "Mild only (6 to 24 dB)", "severe": "Severe only (−6 to 6 dB)"}
    conds = ["internal__em__-6dB__r0", "internal__ma__-6dB__r0", "external__em__-6dB__r0", "external__ma__-6dB__r0"]
    rows = []
    for cf in cfgs:
        for m in ["logreg", "lgbm", "cnn"]:
            g = a[(a.config == cf) & (a.model == m)].set_index("condition")
            if g.empty:
                continue
            rows.append([clab[cf], MODEL[m], f(g.loc["internal__clean"].auroc)] +
                        [f"{f(g.loc[c].sensitivity)} / {f(g.loc[c].specificity)}" for c in conds])
    return md(rows, ["Augmentation", "Model", "Clean AUROC", "Internal EM −6 dB", "Internal MA −6 dB",
                     "PTB-XL EM −6 dB", "PTB-XL MA −6 dB"])


def selective_thresholds():
    s = pd.read_csv(T / "selective_thresholds.csv")
    s = s[s.model == "lgbm"]

    def get(meas, cond, pol):
        g = s[(s.measure == meas) & (s.condition == cond) & s.policy.str.startswith(pol)]
        return g.iloc[0] if not g.empty else None

    rows = []
    for meas in MEAS:
        i = get(meas, "internal__em__-6dB__r0", "source")
        c_src, c_tgt = get(meas, "external__clean", "source"), get(meas, "external__clean", "target")
        rows.append([MEAS[meas], f"{100 * i.withheld_af:.1f} / {100 * i.withheld_non_af:.1f}",
                     f"{f(i.sensitivity)} / {f(i.specificity)}", f"{100 * c_src.withheld_all:.1f}",
                     f"{100 * c_tgt.withheld_all:.1f}"])
    return md(rows, ["Measure", "Internal EM −6 dB, CinC threshold: AF / non-AF withheld, %",
                     "Internal EM −6 dB: GB sens. / spec. on retained", "Clean PTB-XL withheld, CinC threshold, %",
                     "Clean PTB-XL withheld, PTB-XL threshold, %"])


def selective_curves():
    cr = pd.read_csv(T / "selective_coverage_risk.csv")
    rows = []
    for meas in MEAS:
        g = cr[(cr.condition == "internal__em__-6dB__r0") & (cr.measure == meas)]
        cells = []
        for cov in [1.0, 0.8, 0.5]:
            fp = g[(g.model == "lgbm") & (g.coverage == cov)].fpr.iloc[0]
            fn = g[(g.model == "cnn") & (g.coverage == cov)].fnr.iloc[0]
            cells.append(f"{f(fp)} / {f(fn)}")
        rr = g[(g.model == "lgbm") & (g.coverage == 0.8)].af_rejection_ratio.iloc[0]
        rows.append([MEAS[meas]] + cells + [f(rr, 2)])
    return md(rows, ["Measure", "All kept: GB FPR / CNN FNR", "80% kept", "50% kept", "AF rejection ratio, 80% kept"])


def operating():
    o = pd.read_csv(T / "rev_operating.csv")
    s90 = pd.read_csv(T / "rev_sens90.csv")
    o = o[o.variant == "standard"]
    rows = []
    for c in ["internal__clean", "internal__em__-6dB__r0", "internal__ma__-6dB__r0", "external__clean",
              "external__em__-6dB__r0", "external__ma__-6dB__r0"]:
        for m in MODELS:
            r = o[(o.condition == c) & (o.model == m)].iloc[0]
            q = s90[(s90.variant == "standard") & (s90.condition == c) & (s90.model == m)].iloc[0]
            rows.append([("Internal, " if c.startswith("internal") else "PTB-XL, ") + lc(cond_label(c)) if m == "rule" else "",
                         MODEL[m], f"{r.TP:,.1f}", f"{r.FN:,.1f}", f"{r.FP:,.1f}", f"{r.TN:,.1f}", f(r.ppv), f(r.npv),
                         f"{f(q.sensitivity)} / {f(q.specificity)}"])
    return md(rows, ["Test set, condition", "Model", "TP", "FN", "FP", "TN", "PPV", "NPV",
                     "≥90%-sensitivity threshold: sens. / spec."])


def prevalence():
    p = pd.read_csv(T / "rev_prevalence.csv")
    p = p[(p.variant == "standard") & p.condition.str.startswith("external")]
    prevs = sorted(p.prevalence.unique())
    rows = []
    for c in ["external__clean", "external__em__-6dB__r0", "external__ma__-6dB__r0"]:
        for m in MODELS:
            g = p[(p.condition == c) & (p.model == m)].set_index("prevalence")
            rows.append([cond_label(c) if m == "rule" else "", MODEL[m]] + [f(g.loc[x].ppv_hypothetical) for x in prevs] +
                        [f(g.loc[0.01].npv_hypothetical), f(g.loc[0.01].false_alarms_per_true_af, 1)])
    return md(rows, ["PTB-XL condition", "Model"] + [f"PPV at {100 * x:g}%" for x in prevs] +
              ["NPV at 1%", "False alarms per detected AF at 1%"])


TABLES = {"external_grid": external_grid, "rhythm": rhythm, "variance": variance, "altnoise": altnoise,
          "rpeak": rpeak, "pseudo": pseudo, "oracle": oracle, "detector": detector, "regional": regional, "cnn": cnn,
          "afdb": afdb, "augablation": augablation, "selective_thresholds": selective_thresholds,
          "selective_curves": selective_curves, "operating": operating, "prevalence": prevalence}


def main():
    text = TEMPLATE.read_text(encoding="utf-8")

    def sub(mt):
        name = mt.group(1)
        if name not in TABLES:
            sys.exit(f"unknown table {name}")
        return TABLES[name]()

    out = re.sub(r"<!-- TABLE:(\w+) -->", sub, text)
    OUT.write_text(out, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
