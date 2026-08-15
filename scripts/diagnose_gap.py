"""Diagnose why active learning at ~900 labels beats 120,000-label supervision.

Three candidate explanations, tested directly:

  H1 CONTAMINATION - pool and test share (near-)identical rows, so an AL set
     that happens to cover them memorises the test. Checked by hashing feature
     rows across the split.

  H2 COMPOSITION - the gain is purely that AL's labeled set is class-balanced,
     not that its examples are individually informative. Checked by training a
     control on a RANDOM draw with the *identical per-class counts* as AL's
     final labeled set. If the control matches AL, the acquisition function is
     doing nothing beyond rebalancing.

  H3 INFORMATIVENESS - AL additionally selects boundary-adjacent majority-class
     examples that sharpen the rare-class decision boundary. This is what is
     left if H1 and H2 are ruled out.

Run: python scripts/diagnose_gap.py --track b --seed 0
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.active_learning import ActiveLearner, uncertainty_score  # noqa: E402
from scripts.run_experiment import (  # noqa: E402
    MAX_POOL, TEST_FRAC, load_track, make_estimator, stratified_seed,
)


def row_hashes(X):
    return np.array([hashlib.md5(np.ascontiguousarray(r).tobytes()).hexdigest()
                     for r in X])


def per_class_f1(model, X_test, y_test, classes):
    pred = model.predict(X_test)
    per = f1_score(y_test, pred, average=None, labels=classes, zero_division=0)
    macro = f1_score(y_test, pred, average="macro", labels=classes, zero_division=0)
    return macro, dict(zip(classes, per))


def draw_matching_counts(y_pool, counts, rng):
    """Random draw with exactly `counts` examples of each class."""
    idx = []
    for c, n in counts.items():
        ci = np.flatnonzero(y_pool == c)
        idx.extend(rng.choice(ci, size=min(int(n), len(ci)), replace=False))
    return np.array(idx, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["a", "b"], default="b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=15)
    args = ap.parse_args()

    X, y, _ = load_track(args.track)
    classes = sorted(np.unique(y).tolist())
    seed = args.seed
    rng = np.random.default_rng(seed)

    X_pool_all, X_test, y_pool_all, y_test = train_test_split(
        X, y, test_size=TEST_FRAC, stratify=y, random_state=seed)
    if len(y_pool_all) > MAX_POOL:
        X_pool, _, y_pool, _ = train_test_split(
            X_pool_all, y_pool_all, train_size=MAX_POOL,
            stratify=y_pool_all, random_state=seed)
    else:
        X_pool, y_pool = X_pool_all, y_pool_all

    print(f"pool={len(y_pool):,}  test={len(y_test):,}")
    print("pool class counts:", pd.Series(y_pool).value_counts().to_dict())
    print("test class counts:", pd.Series(y_test).value_counts().to_dict())

    # ---------------- H1: contamination ----------------
    print("\n=== H1: pool/test row overlap ===")
    hp, ht = row_hashes(X_pool), row_hashes(X_test)
    shared = set(hp) & set(ht)
    n_test_dup = int(np.isin(ht, list(shared)).sum()) if shared else 0
    print(f"  identical feature rows shared across split: {len(shared):,} distinct")
    print(f"  test rows matching some pool row: {n_test_dup:,} "
          f"({100*n_test_dup/len(ht):.3f}% of test)")

    # ---------------- run AL, capture the labeled set ----------------
    init_idx = stratified_seed(y_pool, 2, rng)
    learner = ActiveLearner(
        estimator=make_estimator(len(classes), seed),
        X=X_pool, label_fn=lambda i: y_pool[i],
        score_fn=uncertainty_score, init_indices=init_idx,
        batch_size=args.batch_size, eval_fn=None,
        strategy_name="uncertainty", random_state=seed)
    learner.run(args.rounds)
    al_idx = learner.labeled_idx
    al_counts = pd.Series(y_pool[al_idx]).value_counts()
    print(f"\n=== AL labeled set: {len(al_idx)} labels ===")
    pool_counts = pd.Series(y_pool).value_counts()
    print(f"  {'class':<7}{'AL n':>7}{'pool n':>9}{'AL %':>8}{'pool %':>9}{'coverage':>10}")
    for c in classes:
        a = int(al_counts.get(c, 0))
        p = int(pool_counts.get(c, 0))
        print(f"  {c:<7}{a:>7}{p:>9,}{100*a/len(al_idx):>7.1f}%"
              f"{100*p/len(y_pool):>8.2f}%{100*a/max(p,1):>9.1f}%")

    al_model = make_estimator(len(classes), seed)
    al_model.fit(X_pool[al_idx], y_pool[al_idx])
    al_macro, al_per = per_class_f1(al_model, X_test, y_test, classes)

    # ---------------- H2: composition-matched random control ----------------
    print("\n=== H2: random draw with IDENTICAL per-class counts ===")
    ctrl_macro, ctrl_per = [], []
    for rep in range(5):
        r = np.random.default_rng(1000 + rep)
        cidx = draw_matching_counts(y_pool, al_counts.to_dict(), r)
        m = make_estimator(len(classes), seed)
        m.fit(X_pool[cidx], y_pool[cidx])
        mac, per = per_class_f1(m, X_test, y_test, classes)
        ctrl_macro.append(mac)
        ctrl_per.append(per)
    ctrl_mean = {c: float(np.mean([p[c] for p in ctrl_per])) for c in classes}
    ctrl_sd = {c: float(np.std([p[c] for p in ctrl_per])) for c in classes}

    print(f"  {'class':<7}{'AL F1':>9}{'ctrl F1':>10}{'sd':>7}{'delta':>9}")
    for c in classes:
        print(f"  {c:<7}{al_per[c]:>9.3f}{ctrl_mean[c]:>10.3f}"
              f"{ctrl_sd[c]:>7.3f}{al_per[c]-ctrl_mean[c]:>+9.3f}")
    print(f"  {'MACRO':<7}{al_macro:>9.3f}{np.mean(ctrl_macro):>10.3f}"
          f"{np.std(ctrl_macro):>7.3f}{al_macro-np.mean(ctrl_macro):>+9.3f}")

    print("\n=== verdict ===")
    gap = al_macro - float(np.mean(ctrl_macro))
    if n_test_dup > 0.01 * len(ht):
        print("  H1 SUPPORTED: material pool/test row overlap -- results contaminated.")
    elif abs(gap) < 3 * (np.std(ctrl_macro) + 1e-9):
        print("  H2 SUPPORTED: composition-matched random matches AL. The gain is")
        print("  rebalancing, not informativeness -- AL is not selecting better")
        print("  examples, only a better class mix.")
    else:
        print(f"  H3 SUPPORTED: AL beats a composition-matched random draw by "
              f"{gap:+.3f} macro-F1,")
        print("  so the acquisition function adds value beyond rebalancing.")

    out = ROOT / "results" / f"diagnose_gap_track_{args.track}_seed{seed}.csv"
    pd.DataFrame([
        {"class": c, "al_f1": al_per[c], "ctrl_f1": ctrl_mean[c],
         "ctrl_sd": ctrl_sd[c], "al_n": int(al_counts.get(c, 0)),
         "pool_n": int(pool_counts.get(c, 0))} for c in classes
    ]).to_csv(out, index=False)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
