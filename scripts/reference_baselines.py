"""Diagnostic: is active learning genuinely beating full supervision, or is
the full-supervised reference just badly configured for imbalanced data?

The main run produced AL macro-F1 ~0.91 at 914 labels against a
full-supervised reference of ~0.78 at 120,000 labels. The obvious
explanation is that the reference is trained on the natural class
distribution (ecl 50.5%, cv 0.14%), so it under-predicts rare classes and
scores poorly on *macro*-F1, while active acquisition implicitly builds a
class-balanced training set.

If a properly balanced full-supervised model closes the gap, then the
headline is "AL rebalances for free", not "AL beats supervision", and the
label-savings normalisation must use the balanced reference.

References compared, all on the same pool/test splits as the main run:
  natural          - LightGBM, natural distribution      (what the main run used)
  class_weight     - LightGBM, class_weight='balanced'
  undersampled     - LightGBM on a balanced subsample (n = rarest class)
  matched_budget   - LightGBM on a *balanced random* draw of 914 labels,
                     i.e. what random sampling would give if it could sample
                     class-stratified. Isolates "balance" from "informativeness".

Run: python scripts/reference_baselines.py --track b
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_experiment import (  # noqa: E402
    MAX_POOL, TEST_FRAC, load_track, make_estimator,
)


def evaluate(model, X_test, y_test, classes):
    pred = model.predict(X_test)
    out = {
        "macro_f1": float(f1_score(y_test, pred, average="macro",
                                   labels=classes, zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted",
                                      labels=classes, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_test, pred)),
    }
    per = f1_score(y_test, pred, average=None, labels=classes, zero_division=0)
    for c, v in zip(classes, per):
        out[f"f1_{c}"] = float(v)
    return out


def balanced_indices(y, n_per_class, rng):
    idx = []
    for c in np.unique(y):
        ci = np.flatnonzero(y == c)
        idx.extend(rng.choice(ci, size=min(n_per_class, len(ci)), replace=False))
    return np.array(idx, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["a", "b"], default="b")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--budget", type=int, default=914,
                    help="label budget for the matched-budget balanced draw")
    args = ap.parse_args()

    X, y, _ = load_track(args.track)
    classes = sorted(np.unique(y).tolist())

    rows = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        X_pool_all, X_test, y_pool_all, y_test = train_test_split(
            X, y, test_size=TEST_FRAC, stratify=y, random_state=seed)
        if len(y_pool_all) > MAX_POOL:
            X_pool, _, y_pool, _ = train_test_split(
                X_pool_all, y_pool_all, train_size=MAX_POOL,
                stratify=y_pool_all, random_state=seed)
        else:
            X_pool, y_pool = X_pool_all, y_pool_all

        variants = {}

        m = make_estimator(len(classes), seed)
        m.fit(X_pool, y_pool)
        variants["natural"] = (m, len(y_pool))

        m = make_estimator(len(classes), seed)
        m.set_params(class_weight="balanced")
        m.fit(X_pool, y_pool)
        variants["class_weight"] = (m, len(y_pool))

        n_rare = int(pd.Series(y_pool).value_counts().min())
        bidx = balanced_indices(y_pool, n_rare, rng)
        m = make_estimator(len(classes), seed)
        m.fit(X_pool[bidx], y_pool[bidx])
        variants["undersampled"] = (m, len(bidx))

        per_cls = max(1, args.budget // len(classes))
        midx = balanced_indices(y_pool, per_cls, rng)
        m = make_estimator(len(classes), seed)
        m.fit(X_pool[midx], y_pool[midx])
        variants["matched_budget"] = (m, len(midx))

        for name, (model, n_lab) in variants.items():
            rec = {"seed": seed, "variant": name, "n_labels": n_lab}
            rec.update(evaluate(model, X_test, y_test, classes))
            rows.append(rec)
            print(f"[seed {seed}] {name:<15} n={n_lab:>7,} "
                  f"macro_f1={rec['macro_f1']:.4f} bal_acc={rec['balanced_acc']:.4f}")

    df = pd.DataFrame(rows)
    out = ROOT / "results" / f"reference_variants_track_{args.track}.csv"
    df.to_csv(out, index=False)

    print("\n=== mean across seeds ===")
    agg = df.groupby("variant").agg(
        n_labels=("n_labels", "mean"),
        macro_f1=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        balanced_acc=("balanced_acc", "mean"),
    ).sort_values("macro_f1")
    print(agg.to_string())

    print("\n=== per-class F1 by variant ===")
    fcols = [c for c in df.columns if c.startswith("f1_")]
    print(df.groupby("variant")[fcols].mean().to_string())
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
