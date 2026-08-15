"""Independent per-class validation of VarWISE `vartype` against SIMBAD.

VarWISE reports a macro-F1 of 0.95, measured against a validation split of
its own training labels (Gaia/Rimoldini 2023 + ZTF/Chen 2020). This script
scores the published `vartype` column against a source of truth that was not
used to train it -- SIMBAD literature types -- for the 220k Pure Catalog
objects that carry one.

This is a catalog-validation result independent of the active-learning study
and stands on its own.

Run: python scripts/validate_varwise.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_dataset import SIMBAD_MAP  # noqa: E402

RAW = ROOT / "data" / "raw" / "varwise_pure.parquet"


def main():
    df = pd.read_parquet(RAW)
    df["vartype"] = df["vartype"].astype("string").str.strip()
    df["simbad_type"] = df["simbad_type"].astype("string").str.strip()

    truth = df["simbad_type"].map(SIMBAD_MAP)
    m = truth.notna() & df["vartype"].notna() & (df["vartype"] != "unclear")
    y_true = truth[m].values
    # SIMBAD does not separate Algol from W UMa eclipsing binaries, so the
    # VarWISE ea/ew distinction is collapsed to a single `ecl` class to make
    # the two taxonomies comparable.
    y_pred = df.loc[m, "vartype"].replace({"ea": "ecl", "ew": "ecl"}).values

    labels = sorted(set(y_true))
    print(f"Scored {m.sum():,} Pure Catalog objects with an independent SIMBAD type\n")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=3))

    print("Row-normalised confusion matrix (rows = SIMBAD truth):")
    all_pred = sorted(set(y_pred) | set(labels))
    cm = confusion_matrix(y_true, y_pred, labels=all_pred).astype(float)
    with np.errstate(invalid="ignore"):
        cmn = cm / cm.sum(axis=1, keepdims=True)
    hdr = "        " + "".join(f"{c:>8}" for c in all_pred)
    print(hdr)
    for i, c in enumerate(all_pred):
        if c not in labels:
            continue
        row = "".join(f"{cmn[i, j]:>8.3f}" if np.isfinite(cmn[i, j]) else f"{'-':>8}"
                      for j in range(len(all_pred)))
        print(f"  {c:<6}{row}")

    print("\nWhere the disagreements go (top confusions):")
    conf = pd.DataFrame({"truth": y_true, "pred": y_pred})
    bad = conf[conf.truth != conf.pred]
    print(bad.groupby(["truth", "pred"]).size().sort_values(ascending=False)
          .head(15).to_string())

    out = ROOT / "results" / "varwise_vs_simbad.txt"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as fh:
        fh.write(f"VarWISE vartype scored against independent SIMBAD types\n")
        fh.write(f"n = {m.sum()}\n\n")
        fh.write(classification_report(y_true, y_pred, labels=labels,
                                       zero_division=0, digits=3))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
