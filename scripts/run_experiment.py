"""Label-efficiency study: active learning vs random sampling on VarWISE.

Core deliverable: label-efficiency curves, overall (macro-F1) and per-class,
with explicit attention to whether AL helps the rare classes more than the
common ones.

Run:  python scripts/run_experiment.py --track b
      python scripts/run_experiment.py --track a
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from functools import partial
from pathlib import Path

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.active_learning import (  # noqa: E402
    ActiveLearner,
    class_balanced_uncertainty_score,
    margin_score,
    prototype_distance_score,
    quota_score,
    random_score,
    uncertainty_score,
)

# `sn` has only 35 SIMBAD-confirmed members and scores F1 = 0.000 even with
# the full 60k-label reference, so it contributes a constant zero to macro-F1
# and only adds variance. Excluded from the curves and reported separately as
# not learnable from catalog-level features at this sample size. cv (n=302)
# is retained -- it is rare but does learn (reference F1 ~ 0.40).
MIN_CLASS_COUNT = 100

# Rare-class recovery is pool-limited: at 0.14% prevalence, a 60k pool holds
# only ~84 cv examples. Keep more of the pool so the rare-class ceiling is set
# by the features rather than by subsampling.
MAX_POOL = 120_000
TEST_FRAC = 0.30


def make_estimator(n_classes: int, seed: int):
    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=31,
        min_child_samples=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


# `known_extragalactic` is an external cross-match flag, built from the same
# kind of literature catalogs that supply the SIMBAD labels. It nearly
# determines agn/sn membership on its own, which would flatter precisely the
# rare classes this study is about. Dropped by default; --keep-leaky restores
# it for the sensitivity check.
LEAKY_FEATURES = ["known_extragalactic"]


def load_track(track: str, keep_leaky: bool = False):
    path = ROOT / "data" / (f"track_{track}_vartype.parquet" if track == "a"
                            else f"track_{track}_simbad.parquet")
    df = pd.read_parquet(path)
    y = df["_label"].astype(str).values
    X = df.drop(columns=["_label"])
    if not keep_leaky:
        drop = [c for c in LEAKY_FEATURES if c in X.columns]
        if drop:
            print(f"Dropping leakage-prone features: {drop}")
            X = X.drop(columns=drop)
    feat_names = list(X.columns)

    counts = pd.Series(y).value_counts()
    keep = set(counts[counts >= MIN_CLASS_COUNT].index)
    dropped = {k: int(v) for k, v in counts.items() if k not in keep}
    if dropped:
        print(f"Dropping classes below {MIN_CLASS_COUNT}: {dropped}")
    m = np.isin(y, list(keep))
    return X.values[m].astype(np.float64), y[m], feat_names


def stratified_seed(y_pool: np.ndarray, per_class: int, rng: np.random.Generator):
    """Seed set with `per_class` examples of every class present in the pool."""
    idx = []
    for c in np.unique(y_pool):
        c_idx = np.flatnonzero(y_pool == c)
        take = min(per_class, len(c_idx))
        idx.extend(rng.choice(c_idx, size=take, replace=False))
    return np.array(sorted(idx), dtype=int)


def build_eval_fn(X_test, y_test, classes):
    def eval_fn(model):
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
    return eval_fn


def run_track(track: str, seeds, n_rounds: int, batch_size: int, seed_per_class: int,
              keep_leaky: bool = False, tag: str = ""):
    X, y, feat_names = load_track(track, keep_leaky=keep_leaky)
    classes = sorted(np.unique(y).tolist())
    print(f"\n=== Track {track.upper()} ===")
    print(f"{len(y):,} rows, {X.shape[1]} features, {len(classes)} classes: {classes}")
    for c in classes:
        n = int((y == c).sum())
        print(f"  {c:<6} {n:>8,}  {100*n/len(y):6.2f}%")

    strategies = {
        "random": random_score,
        "uncertainty": uncertainty_score,
        "margin": margin_score,
        "class_balanced": class_balanced_uncertainty_score,
        "quota": partial(quota_score, base_score_fn=uncertainty_score),
        "prototype": partial(prototype_distance_score, base_score_fn=uncertainty_score),
    }

    records = []
    full_ref = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        X_pool_all, X_test, y_pool_all, y_test = train_test_split(
            X, y, test_size=TEST_FRAC, stratify=y, random_state=seed
        )
        # cap pool size for tractable repeated refits, keeping stratification
        if len(y_pool_all) > MAX_POOL:
            X_pool, _, y_pool, _ = train_test_split(
                X_pool_all, y_pool_all, train_size=MAX_POOL,
                stratify=y_pool_all, random_state=seed
            )
        else:
            X_pool, y_pool = X_pool_all, y_pool_all

        eval_fn = build_eval_fn(X_test, y_test, classes)

        # full-supervised reference on the entire pool
        t0 = time.time()
        ref_model = make_estimator(len(classes), seed)
        ref_model.fit(X_pool, y_pool)
        ref = eval_fn(ref_model)
        ref["n_labels"] = len(y_pool)
        ref["seed"] = seed
        full_ref.append(ref)
        print(f"\n[seed {seed}] full-supervised on {len(y_pool):,} labels: "
              f"macro_f1={ref['macro_f1']:.4f} bal_acc={ref['balanced_acc']:.4f} "
              f"({time.time()-t0:.0f}s)")

        init_idx = stratified_seed(y_pool, seed_per_class, rng)
        label_fn = lambda idx: y_pool[idx]  # noqa: E731

        for name, fn in strategies.items():
            t0 = time.time()
            learner = ActiveLearner(
                estimator=make_estimator(len(classes), seed),
                X=X_pool,
                label_fn=label_fn,
                score_fn=fn,
                init_indices=init_idx,
                batch_size=batch_size,
                eval_fn=eval_fn,
                strategy_name=name,
                random_state=seed,
            )
            hist = learner.run(n_rounds)
            for n_lab, met in zip(hist.n_labels, hist.metrics):
                rec = {"track": track, "seed": seed, "strategy": name,
                       "n_labels": n_lab, **met}
                records.append(rec)
            final = hist.metrics[-1]
            print(f"  [seed {seed}] {name:<15} {hist.n_labels[-1]:>5} labels -> "
                  f"macro_f1={final['macro_f1']:.4f} "
                  f"({100*final['macro_f1']/ref['macro_f1']:.1f}% of full) "
                  f"[{time.time()-t0:.0f}s]")

    curves = pd.DataFrame(records)
    refs = pd.DataFrame(full_ref)
    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    suffix = f"{track}{tag}"
    curves.to_csv(outdir / f"curves_track_{suffix}.csv", index=False)
    refs.to_csv(outdir / f"reference_track_{suffix}.csv", index=False)
    with open(outdir / f"meta_track_{suffix}.json", "w") as fh:
        json.dump({"track": track, "classes": classes, "features": feat_names,
                   "n_rows": int(len(y)), "seeds": list(seeds),
                   "n_rounds": n_rounds, "batch_size": batch_size,
                   "seed_per_class": seed_per_class, "keep_leaky": keep_leaky,
                   "max_pool": MAX_POOL, "test_frac": TEST_FRAC}, fh, indent=2)
    print(f"\nWrote {outdir / f'curves_track_{suffix}.csv'}")
    return curves, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["a", "b"], default="b")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    # The smoke run showed random sampling matching the 60k-label reference by
    # ~240 labels, so the informative region is 10-900 labels, not 40-2040.
    # Small seed + small batches resolve the part of the curve where the
    # strategies actually differ.
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--seed-per-class", type=int, default=2)
    ap.add_argument("--keep-leaky", action="store_true",
                    help="retain known_extragalactic (sensitivity check)")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    args = ap.parse_args()

    run_track(args.track, args.seeds, args.rounds, args.batch_size,
              args.seed_per_class, keep_leaky=args.keep_leaky, tag=args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
