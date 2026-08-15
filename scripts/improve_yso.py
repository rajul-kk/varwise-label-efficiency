"""Address the YSO recall deficit found in the Pure Catalog audit.

VarWISE's XGBoost classifier recovers only 43.6% of SIMBAD-typed young
stellar objects (precision 0.940, F1 0.595) -- when it says YSO it is
usually right, it simply misses most of them. The missed objects scatter
into `lpv` and, via the transient rule, into `cv`.

Recall matters more than precision for this class: YSO variability is a
headline science case for mid-IR time-domain surveys, and a sample that
misses ~56% of the population biases any statistical study of it. A
follow-up programme can filter false positives; it cannot recover objects
that were never flagged.

This trains a YSO-focused classifier on the same catalog features and reports
the precision/recall trade-off, so a user can pick an operating point rather
than accept a fixed one.

Note on prior work: Refined classification of YSOs vs AGB stars using IR
magnitudes, colours and time-domain analysis was published in Jan 2026
(ApJ, doi:10.3847/1538-4357/ae25f2). This script is not proposing a new YSO
method -- it quantifies and closes a specific gap in one catalog.

Run: python scripts/improve_yso.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "yso_recall.txt"
VARWISE_YSO_RECALL = 0.436   # measured in scripts/validate_varwise.py
VARWISE_YSO_PREC = 0.940
VARWISE_YSO_F1 = 0.595
lines = []


def emit(s=""):
    print(s)
    lines.append(s)


def main():
    df = pd.read_parquet(ROOT / "data" / "track_b_simbad.parquet")
    y_multi = df["_label"].astype(str).values
    X = df.drop(columns=["_label"])
    X = X.drop(columns=[c for c in ["known_extragalactic"] if c in X.columns])
    feat = list(X.columns)
    Xv = X.values.astype(np.float64)
    y = (y_multi == "yso").astype(int)

    emit("=" * 80)
    emit("CLOSING THE YSO RECALL GAP")
    emit("=" * 80)
    emit(f"\n  objects: {len(y):,}   YSOs: {y.sum():,} ({y.mean():.2%})")
    emit(f"  features: {len(feat)} (known_extragalactic excluded as leakage)")
    emit(f"\n  VarWISE baseline (from the Pure Catalog audit):")
    emit(f"    precision {VARWISE_YSO_PREC:.3f}   recall {VARWISE_YSO_RECALL:.3f}   "
         f"F1 {VARWISE_YSO_F1:.3f}")

    # ---- cross-validated probabilities ----
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(Xv, y):
        clf = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                             min_child_samples=20, class_weight="balanced",
                             random_state=0, n_jobs=-1, verbose=-1)
        clf.fit(Xv[tr], y[tr])
        oof[te] = clf.predict_proba(Xv[te])[:, 1]

    ap = average_precision_score(y, oof)
    emit(f"\n  binary YSO classifier, 5-fold CV")
    emit(f"    average precision (area under PR curve): {ap:.4f}")

    # ---- operating points ----
    emit("\n" + "=" * 80)
    emit("OPERATING POINTS - pick recall vs precision explicitly")
    emit("=" * 80)
    emit(f"\n  {'threshold':>10}{'precision':>12}{'recall':>10}{'F1':>9}"
         f"{'n flagged':>12}")
    for thr in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
        pred = (oof >= thr).astype(int)
        p = precision_score(y, pred, zero_division=0)
        r = recall_score(y, pred, zero_division=0)
        f = 0 if p + r == 0 else 2 * p * r / (p + r)
        emit(f"  {thr:>10.2f}{p:>12.3f}{r:>10.3f}{f:>9.3f}{int(pred.sum()):>12,}")

    # ---- matched comparisons against VarWISE ----
    prec, rec, thrs = precision_recall_curve(y, oof)
    emit("\n" + "=" * 80)
    emit("HEAD TO HEAD WITH VarWISE")
    emit("=" * 80)

    # (a) at VarWISE's precision, what recall do we get?
    ok = prec[:-1] >= VARWISE_YSO_PREC
    if ok.any():
        i = np.argmax(rec[:-1] * ok)
        emit(f"\n  (a) Matching VarWISE's precision ({VARWISE_YSO_PREC:.3f}):")
        emit(f"      recall {rec[i]:.3f} vs VarWISE {VARWISE_YSO_RECALL:.3f}  "
             f"({rec[i] / VARWISE_YSO_RECALL:.2f}x)")
        emit(f"      threshold {thrs[i]:.3f}, precision {prec[i]:.3f}")

    # (b) at VarWISE's recall, what precision do we get?
    ok = rec[:-1] >= VARWISE_YSO_RECALL
    if ok.any():
        j = np.argmax(prec[:-1] * ok)
        emit(f"\n  (b) Matching VarWISE's recall ({VARWISE_YSO_RECALL:.3f}):")
        emit(f"      precision {prec[j]:.3f} vs VarWISE {VARWISE_YSO_PREC:.3f}")

    # (c) best F1
    f1s = np.where(prec[:-1] + rec[:-1] > 0,
                   2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12), 0)
    k = int(np.argmax(f1s))
    emit(f"\n  (c) Best F1 operating point:")
    emit(f"      F1 {f1s[k]:.3f} vs VarWISE {VARWISE_YSO_F1:.3f}  "
         f"(precision {prec[k]:.3f}, recall {rec[k]:.3f}, threshold {thrs[k]:.3f})")

    # ---- where do the missed YSOs go? ----
    emit("\n" + "=" * 80)
    emit("WHAT THE HARD YSOs LOOK LIKE")
    emit("=" * 80)
    missed = (y == 1) & (oof < 0.5)
    found = (y == 1) & (oof >= 0.5)
    emit(f"\n  {'group':<26}{'n':>8}{'W1-W2':>9}{'W1':>8}{'W1 amp':>9}{'varSNR':>9}")
    for label, mm in [("YSOs recovered", found), ("YSOs still missed", missed),
                      ("LPVs (contaminant)", y_multi == "lpv")]:
        s = X[mm]
        emit(f"  {label:<26}{int(mm.sum()):>8,}{s['w1_w2'].median():>9.3f}"
             f"{s['w1mag'].median():>8.2f}{s['w1_amp'].median():>9.3f}"
             f"{s['variability_snr'].median():>9.2f}")
    emit("\n  If the missed YSOs sit closer to the LPV locus than the recovered")
    emit("  ones, the residual failure is genuine physical overlap rather than")
    emit("  a modelling shortfall.")

    # ---- feature importance ----
    clf = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                         min_child_samples=20, class_weight="balanced",
                         random_state=0, n_jobs=-1, verbose=-1)
    clf.fit(Xv, y)
    imp = pd.Series(clf.feature_importances_, index=feat).sort_values(ascending=False)
    emit("\n  Most informative features for YSO identification:")
    for k_, v in imp.head(10).items():
        emit(f"    {k_:<22}{int(v):>7}")

    emit("\n" + "=" * 80)
    emit("CAVEATS")
    emit("=" * 80)
    emit("""
  - This is not a like-for-like contest. VarWISE's classifier was trained on
    Gaia/ZTF labels and is evaluated here against SIMBAD; this classifier is
    both trained and evaluated on SIMBAD, so it has the easier task by
    construction. The comparison shows that the information needed for
    higher recall is present in the catalog columns -- not that this model
    would beat theirs on their own training distribution.
  - SIMBAD YSO labels include ~58% YSO_Candidate, which are less certain.
  - Restricted to the 220,471 objects carrying a mapped SIMBAD type, which
    skew bright.
""")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
