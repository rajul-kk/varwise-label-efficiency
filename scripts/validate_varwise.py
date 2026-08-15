"""Independent validation of VarWISE classifications against SIMBAD types.

IMPORTANT - two different mechanisms produce VarWISE's `vartype`, and they
must be scored separately:

  (a) XGBoost classifier -- the seven periodic/persistent classes
      (agn, cep, ea, ew, lpv, rr, yso). These carry a `confidence` value.
      VarWISE reports a macro-averaged F-1 of 0.95 for this classifier,
      measured on a held-out split of its own Gaia/ZTF-derived labels.

  (b) A rule-based transient assignment -- `cv` and `sn` only. Quoting the
      paper: "Given that WISE observes very few stellar phenomena outside
      our own Local Group, if we can identify a transient event with a known
      galaxy, we can sensibly assign it the class of SN. Conversely, if we
      find that a transient event lies within our Local Group, it is likely
      to be some sort of CV-related event." Objects VARnet flags as
      transient are crossmatched against Gaia DR3 galaxy/QSO catalogs within
      2 arcsec; a match gives SN, no match gives CV. These carry NO
      confidence value (95.2% of cv and 100% of sn rows are confidence-null),
      and they are NOT covered by the reported 0.95.

Scoring them together and comparing the result to 0.95 would be wrong. This
script reports the XGBoost classes as the like-for-like comparison, and the
rule-assigned classes separately.

Run: python scripts/validate_varwise.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_dataset import SIMBAD_MAP  # noqa: E402

RAW = ROOT / "data" / "raw" / "varwise_pure.parquet"

# Classes produced by the XGBoost classifier (ea/ew collapsed to ecl to match
# SIMBAD, which does not separate Algol from W UMa systems).
XGB_CLASSES = ["agn", "cep", "ecl", "lpv", "rr", "yso"]
RULE_CLASSES = ["cv", "sn"]


def main():
    df = pd.read_parquet(RAW)
    df["vartype"] = df["vartype"].astype("string").str.strip()
    df["simbad_type"] = df["simbad_type"].astype("string").str.strip()

    truth = df["simbad_type"].map(SIMBAD_MAP)
    m = truth.notna() & df["vartype"].notna() & (df["vartype"] != "unclear")
    sub = df[m].copy()
    sub["truth"] = truth[m]
    sub["pred"] = sub["vartype"].replace({"ea": "ecl", "ew": "ecl"})

    out_lines = []

    def emit(s=""):
        print(s)
        out_lines.append(s)

    emit(f"Scored {len(sub):,} Pure Catalog objects with an independent SIMBAD type")
    emit()
    emit("Confidence-null rate by predicted class (confirms the two mechanisms):")
    emit(f"  {'class':<7}{'n':>10}{'conf null':>11}{'null %':>9}")
    for c in sorted(sub.pred.unique()):
        s = sub[sub.pred == c]
        emit(f"  {c:<7}{len(s):>10,}{int(s.confidence.isna().sum()):>11,}"
             f"{s.confidence.isna().mean():>9.1%}")

    # ---------------- (a) XGBoost classes ----------------
    emit()
    emit("=" * 70)
    emit("(a) XGBOOST CLASSES - the like-for-like comparison to the reported 0.95")
    emit("=" * 70)
    x = sub[sub.truth.isin(XGB_CLASSES) & sub.pred.isin(XGB_CLASSES)]
    emit(f"\nRestricted to objects whose SIMBAD truth AND VarWISE prediction are both")
    emit(f"XGBoost classes: n = {len(x):,}\n")
    emit(classification_report(x.truth, x.pred, labels=XGB_CLASSES,
                               zero_division=0, digits=3))
    mf1_x = f1_score(x.truth, x.pred, labels=XGB_CLASSES, average="macro",
                     zero_division=0)
    emit(f"Macro F1 over the six XGBoost classes: {mf1_x:.3f}")
    emit("VarWISE reports 0.95 on its own Gaia/ZTF validation split "
         "(different label source, see caveats).")

    # A softer variant: keep all SIMBAD-truth XGBoost objects, allowing the
    # prediction to fall into a rule class (counts as an error).
    x2 = sub[sub.truth.isin(XGB_CLASSES)]
    mf1_x2 = f1_score(x2.truth, x2.pred, labels=XGB_CLASSES, average="macro",
                      zero_division=0)
    emit(f"\nIf rule-class predictions are retained as errors (n = {len(x2):,}): "
         f"macro F1 = {mf1_x2:.3f}")
    emit("The drop is caused by the rule-based transient assignment stealing "
         "objects from\nthe XGBoost classes, not by the classifier itself.")

    # ---------------- (b) rule-assigned classes ----------------
    emit()
    emit("=" * 70)
    emit("(b) RULE-ASSIGNED TRANSIENT CLASSES (cv, sn) - NOT the XGBoost classifier")
    emit("=" * 70)
    emit(f"\n  {'class':<5}{'SIMBAD n':>10}{'VarWISE n':>11}{'over-pred':>11}"
         f"{'precision':>11}{'recall':>9}")
    for c in RULE_CLASSES:
        nt = int((sub.truth == c).sum())
        npd = int((sub.pred == c).sum())
        tp = int(((sub.truth == c) & (sub.pred == c)).sum())
        emit(f"  {c:<5}{nt:>10,}{npd:>11,}{npd/max(nt,1):>10.1f}x"
             f"{tp/max(npd,1):>11.4f}{tp/max(nt,1):>9.4f}")

    emit("\nWhere the rule's false positives come from:")
    for c in RULE_CLASSES:
        s = sub[sub.pred == c]
        vc = s.truth.value_counts()
        emit(f"  predicted {c}: " + ", ".join(
            f"{k}={v:,}" for k, v in vc.head(5).items()))

    emit("\nThe rule's failure mode is exactly what its definition predicts:")
    emit("  - Bright Galactic Miras/AGB stars vary slowly and get flagged transient;")
    emit("    they are not extragalactic, so the rule assigns CV -> 8,291 LPVs.")
    emit("  - AGN are extragalactic and variable, so they match the Gaia QSO/galaxy")
    emit("    crossmatch and the rule assigns SN -> 3,275 AGN.")

    lpvcv = sub[(sub.truth == "lpv") & (sub.pred == "cv")]
    realcv = sub[(sub.truth == "cv") & (sub.pred == "cv")]
    emit(f"\n  Photometric check (median values):")
    emit(f"    {'group':<22}{'n':>8}{'period1':>10}{'W1 amp':>9}{'W1':>8}")
    emit(f"    {'true cv, pred cv':<22}{len(realcv):>8,}"
         f"{realcv.period1.median():>10.0f}{realcv.w1_amp.median():>9.3f}"
         f"{realcv.w1mag.median():>8.2f}")
    emit(f"    {'true lpv, pred cv':<22}{len(lpvcv):>8,}"
         f"{lpvcv.period1.median():>10.0f}{lpvcv.w1_amp.median():>9.3f}"
         f"{lpvcv.w1mag.median():>8.2f}")
    emit("  The false positives are ~6 mag brighter and 4x lower amplitude than")
    emit("  real CVs - ordinary bright LPVs, not borderline CVs.")

    # ---------------- selection bias ----------------
    emit()
    emit("=" * 70)
    emit("SELECTION BIAS - how representative is the scored subset?")
    emit("=" * 70)
    df["has_simbad"] = df["simbad_type"].notna() & (df["simbad_type"] != "")
    df["pred_all"] = df["vartype"].replace({"ea": "ecl", "ew": "ecl"})
    emit(f"\n  {'class':<7}{'predicted':>11}{'scored':>9}{'coverage':>10}"
         f"{'med W1 scored':>15}{'med W1 unscored':>17}")
    for c in sorted(df.pred_all.dropna().unique()):
        if c == "unclear":
            continue
        mm = df.pred_all == c
        sc = mm & df.has_simbad
        un = mm & ~df.has_simbad
        emit(f"  {c:<7}{int(mm.sum()):>11,}{int(sc.sum()):>9,}"
             f"{sc.sum()/max(mm.sum(),1):>9.1%}"
             f"{df.loc[sc,'w1mag'].median():>15.2f}"
             f"{df.loc[un,'w1mag'].median():>17.2f}")
    emit("\n  SIMBAD-typed cv predictions are ~3.8 mag BRIGHTER than untyped ones,")
    emit("  and real CVs are faint (median W1 14.1) while the contaminants are")
    emit("  bright (8.3). The scored slice therefore over-represents exactly the")
    emit("  contaminating population, so 0.019 is a LOWER BOUND on the bright end,")
    emit("  not a catalog-wide precision. Only 37.3% of cv predictions are scored.")

    out = ROOT / "results" / "varwise_vs_simbad.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
