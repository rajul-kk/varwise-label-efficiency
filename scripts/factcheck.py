"""Independently recompute every number quoted in README.md / RESULTS.md.

Reads only the stored result CSVs and the raw catalog, recomputes each
claim from scratch, and prints PASS/FAIL against the asserted value. Any
FAIL means the writeup drifted from the data and must be corrected.

Run: python scripts/factcheck.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RDIR = ROOT / "results"

FAILURES = []
CHECKS = 0


def check(label, actual, expected, tol=0.005, fmt="{:.3f}"):
    global CHECKS
    CHECKS += 1
    if actual is None or (isinstance(actual, float) and not np.isfinite(actual)):
        ok = expected is None
    elif isinstance(expected, str):
        ok = str(actual) == expected
    else:
        ok = abs(float(actual) - float(expected)) <= tol
    a = fmt.format(actual) if isinstance(actual, float) else str(actual)
    e = fmt.format(expected) if isinstance(expected, float) else str(expected)
    status = "PASS" if ok else "**FAIL**"
    print(f"  [{status}] {label}: computed {a}, claimed {e}")
    if not ok:
        FAILURES.append(f"{label}: computed {a}, claimed {e}")
    return ok


def final_scores(tag):
    c = pd.read_csv(RDIR / f"curves_track_{tag}.csv")
    last = c[c.n_labels == c.n_labels.max()]
    return last.groupby("strategy").macro_f1.mean(), int(c.n_labels.max()), last


def main():
    print("=" * 72)
    print("SECTION 1 - headline label efficiency")
    print("=" * 72)

    # --- XGBoost Track B ---
    m, budget, last_x = final_scores("b_xgb")
    refx = pd.read_csv(RDIR / "reference_track_b_xgb.csv")
    print("\nTrack B / XGBoost:")
    check("budget", budget, 914, tol=0, fmt="{}")
    check("full-supervised macro F1", refx.macro_f1.mean(), 0.935)
    check("margin macro F1", m["margin"], 0.913)
    check("random macro F1", m["random"], 0.808)
    check("AL as % of full supervision",
          100 * m["margin"] / refx.macro_f1.mean(), 97.6, tol=0.3, fmt="{:.1f}")
    check("random as % of full supervision",
          100 * m["random"] / refx.macro_f1.mean(), 86.5, tol=0.3, fmt="{:.1f}")
    check("reference label count", int(refx.n_labels.mean()), 120000, tol=0, fmt="{}")
    check("labels as fraction of full", 100 * budget / refx.n_labels.mean(),
          0.76, tol=0.02, fmt="{:.2f}")

    px = pd.read_csv(RDIR / "primary_savings_track_b_xgb.csv")
    pxm = px[(px.metric == "macro_f1") & (px.strategy == "margin")]
    check("margin labels-to-match", pxm.labels_to_match.iloc[0], 127, tol=1, fmt="{:.0f}")
    check("margin label saving (XGB)", pxm.label_saving.iloc[0], 0.861, tol=0.003)

    # --- LightGBM Track B ---
    m_l, budget_l, last_l = final_scores("b")
    refl = pd.read_csv(RDIR / "reference_track_b.csv")
    print("\nTrack B / LightGBM:")
    check("full-supervised macro F1", refl.macro_f1.mean(), 0.769)
    check("margin macro F1", m_l["margin"], 0.910)
    check("random macro F1", m_l["random"], 0.781)
    pl = pd.read_csv(RDIR / "primary_savings_track_b.csv")
    plm = pl[(pl.metric == "macro_f1") & (pl.strategy == "margin")]
    check("margin label saving (LGB)", plm.label_saving.iloc[0], 0.857, tol=0.003)

    # --- Track A ---
    m_a, budget_a, last_a = final_scores("a")
    refa = pd.read_csv(RDIR / "reference_track_a.csv")
    print("\nTrack A / LightGBM (distillation):")
    check("budget", budget_a, 918, tol=0, fmt="{}")
    check("full-supervised macro F1", refa.macro_f1.mean(), 0.839)
    check("margin macro F1", m_a["margin"], 0.858)
    check("random macro F1", m_a["random"], 0.774)
    pa = pd.read_csv(RDIR / "primary_savings_track_a.csv")
    pam = pa[(pa.metric == "macro_f1") & (pa.strategy == "margin")]
    check("margin label saving (Track A)", pam.label_saving.iloc[0], 0.724, tol=0.003)

    print("\n" + "=" * 72)
    print("SECTION 2 - rare-class concentration (XGBoost)")
    print("=" * 72)
    lab_b = pd.read_parquet(ROOT / "data" / "track_b_simbad.parquet")["_label"].astype(str)
    lab_b = lab_b[lab_b != "sn"]
    claims_xgb = {  # class: (prevalence %, random F1, best active F1, gap)
        "cv":  (0.14, 0.609, 0.881, 0.271),
        "cep": (0.92, 0.438, 0.817, 0.379),
        "yso": (5.44, 0.862, 0.914, 0.052),
        "rr":  (5.49, 0.838, 0.907, 0.069),
        "agn": (8.45, 0.962, 0.981, 0.019),
        "lpv": (29.10, 0.972, 0.983, 0.011),
        "ecl": (50.47, 0.973, 0.984, 0.011),
    }
    for cls, (prev, rnd, best, gap) in claims_xgb.items():
        col = f"f1_{cls}"
        r = last_x[last_x.strategy == "random"][col].mean()
        act = last_x[~last_x.strategy.isin(["random", "quota"])].groupby("strategy")[col].mean()
        print(f"\n  class {cls}:")
        check(f"  {cls} prevalence %", 100 * (lab_b == cls).mean(), prev, tol=0.02, fmt="{:.2f}")
        check(f"  {cls} random F1", r, rnd)
        check(f"  {cls} best active F1", act.max(), best)
        check(f"  {cls} gap", act.max() - r, gap, tol=0.006)

    print("\n" + "=" * 72)
    print("SECTION 3 - composition-matched control (LightGBM, 5 seeds)")
    print("=" * 72)
    fs = sorted(glob.glob(str(RDIR / "diagnose_gap_track_b_seed*.csv")))
    check("number of seeds", len(fs), 5, tol=0, fmt="{}")
    d = pd.concat([pd.read_csv(f).assign(seed=int(Path(f).stem[-1])) for f in fs])
    d["delta"] = d.al_f1 - d.ctrl_f1
    d["prev"] = d.pool_n / d.groupby("seed").pool_n.transform("sum")
    g = d.groupby("class").agg(prev=("prev", "mean"), al=("al_f1", "mean"),
                               ctrl=("ctrl_f1", "mean"), gain=("delta", "mean"),
                               sd=("delta", "std"))
    for cls, gain, sd in [("cep", 0.294, 0.012), ("cv", 0.160, 0.036),
                          ("yso", 0.049, 0.005), ("rr", 0.045, 0.022),
                          ("agn", 0.017, 0.004), ("lpv", 0.010, 0.001),
                          ("ecl", 0.017, 0.003)]:
        check(f"{cls} gain over matched-random", g.loc[cls, "gain"], gain, tol=0.006)
        check(f"{cls} gain sd", g.loc[cls, "sd"], sd, tol=0.004)
    rho = g["prev"].corr(g["gain"], method="spearman")
    check("Spearman rho(prevalence, gain)", rho, -0.857, tol=0.01)

    print("\n" + "=" * 72)
    print("SECTION 4 - distillation vs real labels")
    print("=" * 72)
    lab_a = pd.read_parquet(ROOT / "data" / "track_a_vartype.parquet")["_label"].astype(str)
    lab_b_all = pd.read_parquet(ROOT / "data" / "track_b_simbad.parquet")["_label"].astype(str)
    check("VarWISE predicts cv at %", 100 * (lab_a == "cv").mean(), 7.51, tol=0.02, fmt="{:.2f}")
    check("SIMBAD truth cv at %", 100 * (lab_b_all == "cv").mean(), 0.14, tol=0.01, fmt="{:.2f}")
    check("VarWISE predicts sn at %", 100 * (lab_a == "sn").mean(), 2.10, tol=0.02, fmt="{:.2f}")
    check("SIMBAD truth sn at %", 100 * (lab_b_all == "sn").mean(), 0.016, tol=0.003, fmt="{:.3f}")
    check("VarWISE predicts cep at %", 100 * (lab_a == "cep").mean(), 0.70, tol=0.02, fmt="{:.2f}")
    check("SIMBAD truth cep at %", 100 * (lab_b_all == "cep").mean(), 0.92, tol=0.02, fmt="{:.2f}")

    print("\n" + "=" * 72)
    print("SECTION 5 - quota negative result")
    print("=" * 72)
    check("quota macro F1 (LGB)", m_l["quota"], 0.690)
    check("quota < random (LGB)", float(m_l["quota"] < m_l["random"]), 1.0, tol=0, fmt="{:.0f}")
    check("quota macro F1 (XGB)", m["quota"], 0.766)
    check("quota < random (XGB)", float(m["quota"] < m["random"]), 1.0, tol=0, fmt="{:.0f}")

    print("\n" + "=" * 72)
    print("SECTION 6 - VarWISE vs SIMBAD validation")
    print("=" * 72)
    sys.path.insert(0, str(ROOT))
    from sklearn.metrics import classification_report, f1_score
    from scripts.build_dataset import SIMBAD_MAP

    XGB_CLASSES = ["agn", "cep", "ecl", "lpv", "rr", "yso"]

    df = pd.read_parquet(ROOT / "data" / "raw" / "varwise_pure.parquet")
    df["vartype"] = df["vartype"].astype("string").str.strip()
    df["simbad_type"] = df["simbad_type"].astype("string").str.strip()
    truth = df["simbad_type"].map(SIMBAD_MAP)
    msk = truth.notna() & df["vartype"].notna() & (df["vartype"] != "unclear")
    sub = df[msk].copy()
    sub["truth"] = truth[msk]
    sub["pred"] = sub["vartype"].replace({"ea": "ecl", "ew": "ecl"})
    check("n scored", int(msk.sum()), 220419, tol=0, fmt="{}")

    # The two mechanisms: XGBoost classes carry a confidence, cv/sn do not.
    print("\n  mechanism separation (cv/sn are rule-assigned, not XGBoost):")
    check("cv confidence-null rate",
          sub[sub.pred == "cv"].confidence.isna().mean(), 0.952, tol=0.002)
    check("sn confidence-null rate",
          sub[sub.pred == "sn"].confidence.isna().mean(), 1.000, tol=0.001)
    for c in XGB_CLASSES:
        check(f"{c} confidence-null rate",
              sub[sub.pred == c].confidence.isna().mean(), 0.0, tol=0.001)

    # (a) XGBoost classes - the like-for-like comparison
    print("\n  (a) XGBoost classes:")
    x = sub[sub.truth.isin(XGB_CLASSES) & sub.pred.isin(XGB_CLASSES)]
    check("n (both sides XGBoost classes)", len(x), 205374, tol=0, fmt="{}")
    rep = classification_report(x.truth, x.pred, labels=XGB_CLASSES,
                                zero_division=0, output_dict=True)
    check("macro F1 (six XGBoost classes)", rep["macro avg"]["f1-score"], 0.879)
    check("ecl F1", rep["ecl"]["f1-score"], 0.990)
    check("lpv F1", rep["lpv"]["f1-score"], 0.965)
    check("rr F1", rep["rr"]["f1-score"], 0.948)
    check("agn F1", rep["agn"]["f1-score"], 0.913)
    check("cep F1", rep["cep"]["f1-score"], 0.864)
    check("yso F1", rep["yso"]["f1-score"], 0.595)
    check("yso recall (the classifier's one weak spot)",
          rep["yso"]["recall"], 0.436, tol=0.003)
    x2 = sub[sub.truth.isin(XGB_CLASSES)]
    check("macro F1 charging rule-stolen objects",
          f1_score(x2.truth, x2.pred, labels=XGB_CLASSES, average="macro",
                   zero_division=0), 0.837)

    # (b) rule-assigned transient classes
    print("\n  (b) rule-assigned cv/sn:")
    for c, n_t, n_p, over, prec, rec in [
            ("cv", 301, 11576, 38.5, 0.0192, 0.7375),
            ("sn", 35, 3379, 96.5, 0.0024, 0.2286)]:
        nt = int((sub.truth == c).sum())
        npd = int((sub.pred == c).sum())
        tp = int(((sub.truth == c) & (sub.pred == c)).sum())
        check(f"{c} SIMBAD count", nt, n_t, tol=0, fmt="{}")
        check(f"{c} VarWISE predicted count", npd, n_p, tol=0, fmt="{}")
        check(f"{c} over-prediction factor", npd / nt, over, tol=0.1, fmt="{:.1f}")
        check(f"{c} precision", tp / npd, prec, tol=0.0005, fmt="{:.4f}")
        check(f"{c} recall", tp / nt, rec, tol=0.001, fmt="{:.4f}")

    check("LPV assigned cv by the rule",
          int(((sub.truth == "lpv") & (sub.pred == "cv")).sum()), 8291, tol=0, fmt="{}")
    check("AGN assigned sn by the rule",
          int(((sub.truth == "agn") & (sub.pred == "sn")).sum()), 3275, tol=0, fmt="{}")

    # photometric evidence the cv false positives are ordinary LPVs
    lpvcv = sub[(sub.truth == "lpv") & (sub.pred == "cv")]
    realcv = sub[(sub.truth == "cv") & (sub.pred == "cv")]
    check("real cv median W1", realcv.w1mag.median(), 14.14, tol=0.02)
    check("LPV->cv median W1", lpvcv.w1mag.median(), 8.27, tol=0.02)
    check("real cv median W1 amplitude", realcv.w1_amp.median(), 0.439, tol=0.002)
    check("LPV->cv median W1 amplitude", lpvcv.w1_amp.median(), 0.106, tol=0.002)

    # selection bias
    print("\n  selection bias on the cv claim:")
    has_s = df["simbad_type"].notna() & (df["simbad_type"] != "")
    pred_all = df["vartype"].replace({"ea": "ecl", "ew": "ecl"})
    mcv = pred_all == "cv"
    check("cv predictions in catalog", int(mcv.sum()), 34316, tol=0, fmt="{}")
    check("cv SIMBAD coverage", (mcv & has_s).sum() / mcv.sum(), 0.373, tol=0.003)
    check("cv scored median W1", df.loc[mcv & has_s, "w1mag"].median(), 8.80, tol=0.02)
    check("cv unscored median W1", df.loc[mcv & ~has_s, "w1mag"].median(), 12.59, tol=0.02)

    print("\n" + "=" * 72)
    print("SECTION 7 - dataset facts")
    print("=" * 72)
    check("Pure Catalog rows downloaded", len(df), 457080, tol=0, fmt="{}")
    check("rows with a SIMBAD type", int((df.simbad_type.notna() &
                                          (df.simbad_type != "")).sum()),
          229365, tol=0, fmt="{}")
    check("Track B rows", len(lab_b_all), 220471, tol=0, fmt="{}")
    check("Track A rows", len(lab_a), 456763, tol=0, fmt="{}")
    check("sn count in Track B", int((lab_b_all == "sn").sum()), 35, tol=0, fmt="{}")
    check("features used", pd.read_parquet(ROOT / "data" /
          "track_b_simbad.parquet").shape[1] - 1 - 1, 28, tol=0, fmt="{}")
    check("full-sup weighted F1 XGB", refx.weighted_f1.mean(), 0.979)
    check("full-sup weighted F1 LGB", refl.weighted_f1.mean(), 0.947)

    print("\n" + "=" * 72)
    print(f"{CHECKS} checks run, {len(FAILURES)} failure(s)")
    print("=" * 72)
    for f in FAILURES:
        print(f"  FAIL  {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
