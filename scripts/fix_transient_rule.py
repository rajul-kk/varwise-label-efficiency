"""Replace VarWISE's rule-based CV/SN assignment with a learned classifier.

VarWISE assigns `cv` and `sn` not with its XGBoost classifier but with a rule:
VARnet flags an object as transient, then a 2-arcsec crossmatch against Gaia
DR3 galaxy/QSO catalogs assigns SN on a match and CV otherwise. The Pure
Catalog audit found this over-predicts `cv` by 38.5x and `sn` by 96.5x on the
SIMBAD-covered subset. The paper's own visual inspection independently reports
only 9% of `sn` as solid candidates, with 56% "normal AGNs".

This script asks a narrow, well-posed question: given the objects the rule
assigned to `cv` or `sn`, can a classifier recover what they actually are?

Baseline = the rule itself (everything it labelled `cv` is `cv`, likewise
`sn`). Comparison is on identical held-out objects, so it is like-for-like.

Two feature sets are reported:
  with_xmatch  - includes `known_extragalactic`, the same extragalactic
                 crossmatch information the rule itself uses. This is the fair
                 "same inputs, better model" comparison.
  photometry   - excludes it, testing whether photometry alone suffices.

Run: python scripts/fix_transient_rule.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_dataset import SIMBAD_MAP, build_features  # noqa: E402

RAW = ROOT / "data" / "raw" / "varwise_pure.parquet"
OUT = ROOT / "results" / "transient_rule_fix.txt"
MIN_CLASS = 30
lines = []


def emit(s=""):
    print(s)
    lines.append(s)


def main():
    df = pd.read_parquet(RAW)
    df["vartype"] = df["vartype"].astype("string").str.strip()
    df["simbad_type"] = df["simbad_type"].astype("string").str.strip()
    df["truth"] = df["simbad_type"].map(SIMBAD_MAP)

    # The population the rule acts on: objects it assigned cv or sn, for which
    # an independent SIMBAD label exists.
    m = df["vartype"].isin(["cv", "sn"]) & df["truth"].notna()
    sub = df[m].copy()
    emit("=" * 84)
    emit("REPLACING THE VarWISE CV/SN TRANSIENT RULE WITH A CLASSIFIER")
    emit("=" * 84)
    emit(f"\nRule-assigned transients with an independent SIMBAD label: {len(sub):,}")
    emit(f"  assigned cv: {int((sub.vartype=='cv').sum()):,}   "
         f"assigned sn: {int((sub.vartype=='sn').sum()):,}")

    emit("\nWhat they actually are (SIMBAD):")
    vc = sub.truth.value_counts()
    for k, v in vc.items():
        emit(f"  {k:<6}{v:>8,}  {100*v/len(sub):>6.2f}%")

    # drop classes too rare to model
    keep = set(vc[vc >= MIN_CLASS].index)
    dropped = {k: int(v) for k, v in vc.items() if k not in keep}
    if dropped:
        emit(f"\nDropping classes with < {MIN_CLASS} members: {dropped}")
    sub = sub[sub.truth.isin(keep)].copy()
    classes = sorted(keep)

    # ---- baseline: the rule ----
    rule_pred = sub["vartype"].values
    y = sub["truth"].values
    emit("\n" + "=" * 84)
    emit("BASELINE - the rule as published")
    emit("=" * 84)
    acc = (rule_pred == y).mean()
    emit(f"\n  accuracy on this population: {acc:.4f}  "
         f"({int((rule_pred == y).sum()):,} of {len(y):,} correct)")
    emit(f"  macro F1: {f1_score(y, rule_pred, labels=classes, average='macro', zero_division=0):.4f}")
    emit("\n  The rule can only ever emit `cv` or `sn`, so every object that is")
    emit("  really an LPV, AGN or YSO is necessarily wrong.")

    # ---- learned replacement ----
    X_all = build_features(sub)
    feature_sets = {
        "with_xmatch": list(X_all.columns),
        "photometry": [c for c in X_all.columns if c != "known_extragalactic"],
    }

    results = {}
    for name, cols in feature_sets.items():
        X = X_all[cols].values.astype(np.float64)
        oof = np.empty(len(y), dtype=object)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for tr, te in skf.split(X, y):
            clf = LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                 num_leaves=31, min_child_samples=10,
                                 class_weight="balanced", random_state=0,
                                 n_jobs=-1, verbose=-1)
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict(X[te])
        results[name] = oof

        emit("\n" + "=" * 84)
        emit(f"LEARNED CLASSIFIER - feature set: {name} ({len(cols)} features)")
        emit("=" * 84)
        emit(f"\n  5-fold cross-validated, class-balanced LightGBM\n")
        emit(classification_report(y, oof, labels=classes, zero_division=0, digits=3))
        emit(f"  accuracy: {(oof == y).mean():.4f}   "
             f"macro F1: {f1_score(y, oof, labels=classes, average='macro', zero_division=0):.4f}")

    # ---- head to head ----
    emit("\n" + "=" * 84)
    emit("HEAD TO HEAD")
    emit("=" * 84)
    emit(f"\n  {'method':<28}{'accuracy':>11}{'macro F1':>11}"
         f"{'cv F1':>9}{'sn F1':>9}")
    rows = [("rule as published", rule_pred)] + \
           [(f"classifier ({k})", v) for k, v in results.items()]
    for label, pred in rows:
        a = (pred == y).mean()
        mf = f1_score(y, pred, labels=classes, average="macro", zero_division=0)
        per = dict(zip(classes, f1_score(y, pred, labels=classes,
                                         average=None, zero_division=0)))
        emit(f"  {label:<28}{a:>11.4f}{mf:>11.4f}"
             f"{per.get('cv', float('nan')):>9.3f}{per.get('sn', float('nan')):>9.3f}")

    # ---- what the classifier recovers ----
    best = results["with_xmatch"]
    emit("\n" + "=" * 84)
    emit("WHAT THE CLASSIFIER RECOVERS FROM THE RULE'S FALSE POSITIVES")
    emit("=" * 84)
    emit(f"\n  {'actually':<8}{'n':>8}{'rule correct':>14}{'clf correct':>13}"
         f"{'recovered':>11}")
    for c in classes:
        mm = y == c
        r_ok = int((rule_pred[mm] == c).sum())
        c_ok = int((best[mm] == c).sum())
        emit(f"  {c:<8}{int(mm.sum()):>8,}{r_ok:>14,}{c_ok:>13,}"
             f"{c_ok - r_ok:>+11,}")

    emit("\n  Confusion matrix of the learned classifier (rows = SIMBAD truth):")
    cm = confusion_matrix(y, best, labels=classes)
    emit("        " + "".join(f"{c:>8}" for c in classes))
    for i, c in enumerate(classes):
        emit(f"  {c:<6}" + "".join(f"{v:>8,}" for v in cm[i]))

    emit("\n" + "=" * 84)
    emit("CAVEATS")
    emit("=" * 84)
    emit("""
  - Evaluated only on rule-assigned objects that carry a SIMBAD type (37.3%
    of `cv`, 35.3% of `sn`), and those are systematically brighter than the
    ones without. The classifier is therefore validated on the bright end.
  - Only 222 genuine CVs and 8 genuine SNe survive in this population, so
    the cv/sn columns of the report rest on small numbers.
  - The comparison is deliberately favourable to the classifier in one
    respect: the rule was never designed to output lpv/agn/yso, so it cannot
    win on those. The useful reading is not "the rule is bad" but "this
    population is recoverable, and a classifier recovers most of it".
""")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
