"""Second-pass synthetic majorization test, redesigned around two diagnosed
flaws in the first pass (see THEORY_NOTE.md):

  FLAW 1: only 6 scenario (t) points. A Spearman correlation over 6 points
  has enormous sampling variance no matter how many seeds back each point --
  confirmed directly (10-seed rerun sharpened each point but destabilized
  the correlation). FIX: 16 t-values instead of 6, at a reduced 5 seeds/t so
  total compute stays comparable, since scenario count is the actual lever.

  FLAW 2: the "savings" metric (labels for margin to reach random's own
  endpoint) requires threshold-crossing interpolation on a noisy curve, and
  sometimes never converges at all within the label budget (t=0.6 in the
  first pass). That fragility adds noise unrelated to the real question.
  FIX: use F1 GAP AT A FIXED BUDGET (margin_F1 - random_F1, rarest class) as
  the primary metric instead -- no threshold-crossing, always defined, and
  it's exactly the quantity that was already rock-solid in the "raw
  mechanism" table (random's own F1 vs pi_min).

  FLAW 3 (statistical, not just design): the first pass averaged over seeds
  BEFORE correlating, throwing away 5-10x the usable data points. FIX:
  correlate at the (t, seed) level -- e.g. 16 t x 5 seeds = 80 points, not
  16 scenario means -- which is the statistically correct way to spend a
  fixed compute budget on correlation power rather than per-point precision.

Run: python scripts/synthetic_majorization_finegrid.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.active_learning import ActiveLearner, margin_score, random_score  # noqa: E402

OUT_CSV = ROOT / "results" / "synthetic_finegrid_curves.csv"
OUT_SUMMARY = ROOT / "results" / "synthetic_finegrid_summary.csv"
OUT_TXT = ROOT / "results" / "synthetic_finegrid_report.txt"
lines = []


def emit(s=""):
    print(s)
    lines.append(s)


PI_0 = np.array([0.50, 0.25, 0.10, 0.08, 0.05, 0.02])
K = len(PI_0)
T_VALUES = np.linspace(0.0, 1.0, 16)
SEEDS = [0, 1, 2, 3, 4]
N_POOL = 15000
TEST_FRAC = 0.30
BATCH_SIZE = 30
N_ROUNDS = 25
SEED_PER_CLASS = 3
BUDGET_CHECKPOINTS = [150, 300, 450]  # test robustness across budgets


def pi_of_t(t):
    u = np.full(K, 1.0 / K)
    p = (1 - t) * PI_0 + t * u
    return p / p.sum()


def eval_fn_factory(X_test, y_test, classes):
    def eval_fn(model):
        pred = model.predict(X_test)
        out = {"macro_f1": f1_score(y_test, pred, labels=classes,
                                    average="macro", zero_division=0)}
        per = f1_score(y_test, pred, labels=classes, average=None,
                       zero_division=0)
        for c, v in zip(classes, per):
            out[f"f1_{c}"] = float(v)
        return out
    return eval_fn


def stratified_seed(y_pool, per_class, rng):
    idx = []
    for c in np.unique(y_pool):
        ci = np.flatnonzero(y_pool == c)
        idx.extend(rng.choice(ci, size=min(per_class, len(ci)), replace=False))
    return np.array(sorted(idx), dtype=int)


def value_at_budget(curve_df, strategy, budget, col):
    s = curve_df[curve_df.strategy == strategy].sort_values("n_labels")
    idx = (s.n_labels - budget).abs().idxmin()
    return float(s.loc[idx, col])


def main():
    emit("=" * 88)
    emit("SYNTHETIC MAJORIZATION TEST, FINE-GRID REDESIGN")
    emit("=" * 88)
    emit(f"\n{len(T_VALUES)} t-values x {len(SEEDS)} seeds = "
         f"{len(T_VALUES)*len(SEEDS)} (t, seed) pairs for the correlation")
    emit(f"(first pass: 6 t-values x 10 seeds, correlated only 6 scenario means)")
    emit(f"\nt-grid: {np.round(T_VALUES, 3).tolist()}")

    records = []
    for t in T_VALUES:
        p = pi_of_t(t)
        rare_class = int(np.argmin(p))
        for seed in SEEDS:
            X, y = make_classification(
                n_samples=N_POOL, n_features=15, n_informative=10,
                n_redundant=3, n_classes=K, weights=p.tolist(),
                n_clusters_per_class=1, class_sep=1.2, flip_y=0.02,
                random_state=seed,
            )
            X_pool, X_test, y_pool, y_test = train_test_split(
                X, y, test_size=TEST_FRAC, stratify=y, random_state=seed)
            classes = sorted(np.unique(y_pool).tolist())
            eval_fn = eval_fn_factory(X_test, y_test, classes)
            rng = np.random.default_rng(seed)
            init_idx = stratified_seed(y_pool, SEED_PER_CLASS, rng)

            seed_curves = {}
            for strat_name, score_fn in (("random", random_score),
                                         ("margin", margin_score)):
                learner = ActiveLearner(
                    estimator=LogisticRegression(max_iter=2000, C=1.0),
                    X=X_pool, label_fn=lambda idx: y_pool[idx],
                    score_fn=score_fn, init_indices=init_idx,
                    batch_size=BATCH_SIZE, eval_fn=eval_fn,
                    strategy_name=strat_name, random_state=seed,
                )
                hist = learner.run(N_ROUNDS)
                rows = [{"strategy": strat_name, "n_labels": n, **m}
                       for n, m in zip(hist.n_labels, hist.metrics)]
                seed_curves[strat_name] = pd.DataFrame(rows)

            rare_col = f"f1_{rare_class}"
            row = {"t": t, "pi_min": p.min(), "seed": seed,
                  "rare_class": rare_class}
            for budget in BUDGET_CHECKPOINTS:
                r = value_at_budget(seed_curves["random"], "random", budget, rare_col)
                a = value_at_budget(seed_curves["margin"], "margin", budget, rare_col)
                row[f"rand_f1_b{budget}"] = r
                row[f"margin_f1_b{budget}"] = a
                row[f"gap_b{budget}"] = a - r
            records.append(row)
        print(f"  t={t:.3f} done")

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)

    # ---------------- primary test: gap vs pi_min, per (t,seed) pair -------
    emit("\n" + "=" * 88)
    emit("PRIMARY TEST: F1 gap (margin - random) at fixed budget vs pi_min")
    emit("Correlated at the (t, seed) level -- 80 points, not 16 scenario means")
    emit("=" * 88)
    for budget in BUDGET_CHECKPOINTS:
        col = f"gap_b{budget}"
        rho = df["pi_min"].corr(df[col], method="spearman")
        emit(f"\n  budget={budget}: Spearman rho(pi_min, gap) = {rho:+.3f}  "
             f"(n={len(df)} (t,seed) pairs)")
        emit(f"    mean gap at low pi_min (<0.05): {df[df.pi_min<0.05][col].mean():+.4f}")
        emit(f"    mean gap at high pi_min (>0.12): {df[df.pi_min>0.12][col].mean():+.4f}")

    # also report the scenario-mean version for direct comparison to pass 1
    emit("\n  For comparison, scenario-MEAN version (like pass 1's approach):")
    scen = df.groupby("t").agg(pi_min=("pi_min", "first"),
                               **{f"gap_b{b}": (f"gap_b{b}", "mean")
                                  for b in BUDGET_CHECKPOINTS}).reset_index()
    for budget in BUDGET_CHECKPOINTS:
        rho = scen["pi_min"].corr(scen[f"gap_b{budget}"], method="spearman")
        emit(f"    budget={budget}: rho over {len(scen)} scenario means = {rho:+.3f}")

    # ---------------- raw mechanism sanity (should still be clean) --------
    emit("\n" + "=" * 88)
    emit("SANITY: raw mechanism (random's own rarest-class F1 vs pi_min)")
    emit("=" * 88)
    for budget in BUDGET_CHECKPOINTS:
        rho = df["pi_min"].corr(df[f"rand_f1_b{budget}"], method="spearman")
        emit(f"  budget={budget}: rho(pi_min, random's own rare-F1) = {rho:+.3f}  "
             f"(should be strongly positive, as in both prior runs)")

    # ---------------- scatter table for eyeballing ----------------
    emit("\n" + "=" * 88)
    emit("SCENARIO MEANS (for inspection)")
    emit("=" * 88)
    emit(f"\n  {'t':>6}{'pi_min':>9}{'rand F1@300':>13}{'margin F1@300':>15}"
         f"{'gap@300':>10}")
    for _, row in scen.sort_values("pi_min").iterrows():
        rf = df[df.t == row.t]["rand_f1_b300"].mean()
        af = df[df.t == row.t]["margin_f1_b300"].mean()
        emit(f"  {row.t:>6.3f}{row.pi_min:>9.4f}{rf:>13.4f}{af:>15.4f}"
             f"{row.gap_b300:>10.4f}")

    scen.to_csv(OUT_SUMMARY, index=False)
    emit(f"\nWrote {OUT_CSV}")
    emit(f"Wrote {OUT_SUMMARY}")
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_TXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
