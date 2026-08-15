"""Adversarial stress-test of the VarWISE-vs-SIMBAD audit.

The audit's headline (cv precision 0.019, sn precision 0.002) is the strongest
claim in this repo, so it gets the hardest scrutiny. Threats to validity,
tested in order:

  T1 SELECTION BIAS - SIMBAD covers only ~48% of the Pure Catalog and is
     skewed toward bright, well-studied objects. If VarWISE's cv predictions
     land preferentially on faint objects that SIMBAD never typed, the
     measured precision is computed on an unrepresentative slice.

  T2 CANDIDATE LABELS - a large share of the SIMBAD "truth" carries
     _Candidate status (e.g. LongPeriodV*_Candidate). If the confusions are
     driven by those, the truth itself is soft.

  T3 CONFIDENCE - VarWISE publishes a per-object `confidence`. If precision
     rises sharply with confidence, the failure is concentrated in low-
     confidence predictions and the catalog is usable when filtered.

  T4 TAXONOMY - maybe VarWISE's `cv` is simply a broader class than SIMBAD's
     CataclyV*. Checked by inspecting what the LPV->cv confusions look like
     photometrically (period, amplitude) versus true cv.

  T5 DIRECTION - precision vs recall. Low precision with high recall means
     over-prediction, which is a different (and milder) failure than getting
     the class wrong in both directions.

Run: python scripts/audit_robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_dataset import SIMBAD_MAP  # noqa: E402

RAW = ROOT / "data" / "raw" / "varwise_pure.parquet"
CANDIDATE_TYPES = {k for k in SIMBAD_MAP if "Candidate" in k}


def main():
    df = pd.read_parquet(RAW)
    df["vartype"] = df["vartype"].astype("string").str.strip()
    df["simbad_type"] = df["simbad_type"].astype("string").str.strip()
    df["has_simbad"] = df["simbad_type"].notna() & (df["simbad_type"] != "")
    df["truth"] = df["simbad_type"].map(SIMBAD_MAP)
    df["pred"] = df["vartype"].replace({"ea": "ecl", "ew": "ecl"})

    print("=" * 74)
    print("T1 - SELECTION BIAS: is the scored subset representative?")
    print("=" * 74)
    print(f"\nPure Catalog: {len(df):,} objects; "
          f"{df.has_simbad.sum():,} ({df.has_simbad.mean():.1%}) have a SIMBAD type")
    print(f"Mapped to the 8-class taxonomy: {df.truth.notna().sum():,} "
          f"({df.truth.notna().mean():.1%} of catalog)\n")

    print(f"  {'VarWISE class':<14}{'predicted':>11}{'w/ SIMBAD':>11}"
          f"{'coverage':>10}{'mapped':>9}{'map cov':>9}")
    for c in sorted(df["pred"].dropna().unique()):
        m = df["pred"] == c
        n, ns, nm = int(m.sum()), int((m & df.has_simbad).sum()), int((m & df.truth.notna()).sum())
        print(f"  {c:<14}{n:>11,}{ns:>11,}{ns/max(n,1):>9.1%}"
              f"{nm:>9,}{nm/max(n,1):>9.1%}")

    print("\n  Brightness check (W1 magnitude, fainter = larger):")
    print(f"  {'group':<28}{'n':>10}{'median W1':>12}{'median n_obs':>14}")
    for label, m in [("all Pure Catalog", pd.Series(True, index=df.index)),
                     ("has SIMBAD type", df.has_simbad),
                     ("no SIMBAD type", ~df.has_simbad),
                     ("VarWISE cv, has SIMBAD", (df.pred == "cv") & df.has_simbad),
                     ("VarWISE cv, no SIMBAD", (df.pred == "cv") & ~df.has_simbad)]:
        print(f"  {label:<28}{int(m.sum()):>10,}"
              f"{df.loc[m, 'w1mag'].median():>12.2f}"
              f"{df.loc[m, 'n_obs'].median():>14.0f}")

    print("\n  => Interpretation: if 'has SIMBAD' is much brighter than 'no SIMBAD',")
    print("     the audit measures precision on the bright, well-studied slice only.")

    print("\n" + "=" * 74)
    print("T2 - CANDIDATE LABELS: is the truth itself soft?")
    print("=" * 74)
    scored = df[df.truth.notna() & df.pred.notna() & (df.vartype != "unclear")].copy()
    scored["is_cand"] = scored.simbad_type.isin(CANDIDATE_TYPES)
    print(f"\n  scored objects: {len(scored):,}; "
          f"{scored.is_cand.sum():,} ({scored.is_cand.mean():.1%}) are _Candidate types")

    for label, sub in [("all SIMBAD types", scored),
                       ("confirmed types only", scored[~scored.is_cand])]:
        prec_cv = ((sub.pred == "cv") & (sub.truth == "cv")).sum() / max((sub.pred == "cv").sum(), 1)
        prec_sn = ((sub.pred == "sn") & (sub.truth == "sn")).sum() / max((sub.pred == "sn").sum(), 1)
        lpv_cv = int(((sub.truth == "lpv") & (sub.pred == "cv")).sum())
        agn_sn = int(((sub.truth == "agn") & (sub.pred == "sn")).sum())
        print(f"\n  {label} (n={len(sub):,}):")
        print(f"    cv precision = {prec_cv:.4f}   sn precision = {prec_sn:.4f}")
        print(f"    LPV->cv = {lpv_cv:,}   AGN->sn = {agn_sn:,}")

    print("\n" + "=" * 74)
    print("T3 - CONFIDENCE: does filtering on VarWISE confidence rescue precision?")
    print("=" * 74)
    print(f"\n  {'conf >=':>9}{'n scored':>11}{'cv prec':>10}{'sn prec':>10}"
          f"{'yso rec':>10}{'macro F1':>10}")
    from sklearn.metrics import f1_score
    for thr in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99):
        sub = scored[scored.confidence >= thr]
        if len(sub) < 100:
            continue
        pcv = ((sub.pred == "cv") & (sub.truth == "cv")).sum() / max((sub.pred == "cv").sum(), 1)
        psn = ((sub.pred == "sn") & (sub.truth == "sn")).sum() / max((sub.pred == "sn").sum(), 1)
        ryso = ((sub.pred == "yso") & (sub.truth == "yso")).sum() / max((sub.truth == "yso").sum(), 1)
        labs = sorted(set(sub.truth))
        mf1 = f1_score(sub.truth, sub.pred, labels=labs, average="macro", zero_division=0)
        print(f"  {thr:>9.2f}{len(sub):>11,}{pcv:>10.4f}{psn:>10.4f}{ryso:>10.4f}{mf1:>10.4f}")

    print("\n" + "=" * 74)
    print("T4 - TAXONOMY: are the LPV->cv objects photometrically like real cv?")
    print("=" * 74)
    grp = {
        "true cv, pred cv": (scored.truth == "cv") & (scored.pred == "cv"),
        "true lpv, pred cv": (scored.truth == "lpv") & (scored.pred == "cv"),
        "true lpv, pred lpv": (scored.truth == "lpv") & (scored.pred == "lpv"),
    }
    print(f"\n  {'group':<22}{'n':>8}{'med period1':>14}{'med W1amp':>12}"
          f"{'med W1-W2':>12}{'med W1':>9}")
    for label, m in grp.items():
        s = scored[m]
        print(f"  {label:<22}{len(s):>8,}{s.period1.median():>14.2f}"
              f"{s.w1_amp.median():>12.3f}"
              f"{(s.w1mag - s.w2mag).median():>12.3f}{s.w1mag.median():>9.2f}")
    print("\n  => If 'true lpv, pred cv' looks like 'true lpv, pred lpv' rather than")
    print("     like real cv, the predictions are wrong rather than the taxonomy.")

    print("\n" + "=" * 74)
    print("T5 - DIRECTION: over-prediction vs two-way confusion")
    print("=" * 74)
    print(f"\n  {'class':<6}{'SIMBAD n':>10}{'VarWISE pred n':>16}"
          f"{'over-pred x':>13}{'precision':>11}{'recall':>9}")
    for c in sorted(set(scored.truth)):
        nt = int((scored.truth == c).sum())
        npd = int((scored.pred == c).sum())
        tp = int(((scored.truth == c) & (scored.pred == c)).sum())
        print(f"  {c:<6}{nt:>10,}{npd:>16,}{npd/max(nt,1):>13.1f}"
              f"{tp/max(npd,1):>11.4f}{tp/max(nt,1):>9.4f}")

    out = ROOT / "results" / "audit_robustness.txt"
    print(f"\n(Full numbers above; see {out.name} in results/ for the saved copy.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
