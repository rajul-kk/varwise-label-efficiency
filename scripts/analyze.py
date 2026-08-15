"""Turn raw label-efficiency curves into the study's headline numbers.

Defines label savings as: the reduction in labels an active strategy needs,
relative to random sampling, to first reach a target fraction of the
full-supervised reference score. Computed on the seed-averaged curve, both
for macro-F1 overall and for each class's own F1 -- the per-class version is
what answers "does AL help the rare classes more than the common ones?".

Run: python scripts/analyze.py --track b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (0.90, 0.95, 0.99)


def labels_to_reach(n_labels: np.ndarray, scores: np.ndarray, target: float):
    """First label count whose score >= target, with linear interpolation
    between the bracketing rounds. None if the curve never gets there.
    """
    hit = np.flatnonzero(scores >= target)
    if len(hit) == 0:
        return None
    i = hit[0]
    if i == 0:
        return float(n_labels[0])
    x0, x1 = n_labels[i - 1], n_labels[i]
    y0, y1 = scores[i - 1], scores[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def savings_vs_random_endpoint(curves: pd.DataFrame, metric: str):
    """Primary label-savings metric.

    Normalising against the full-supervised reference breaks on this dataset:
    active learning *exceeds* the 120k-label reference on macro-F1 (because
    the reference is trained on the natural distribution and suppresses the
    rare classes), so "labels to reach 95% of reference" is satisfied in the
    first round and the ratio is meaningless.

    Instead, take the score random sampling achieves at the full label budget
    and ask how many labels each active strategy needs to match it. That is a
    like-for-like statement -- same task, same budget ceiling, same estimator
    -- and it is the quantity a practitioner actually cares about: how much
    labeling effort is avoided to get what random would have given you.
    """
    mean_curve = (curves.groupby(["strategy", "n_labels"])[metric]
                  .mean().reset_index())
    rnd = mean_curve[mean_curve.strategy == "random"].sort_values("n_labels")
    if rnd.empty:
        return pd.DataFrame()
    budget = int(rnd.n_labels.max())
    target = float(rnd[rnd.n_labels == budget][metric].iloc[0])

    rows = []
    for strat in sorted(mean_curve.strategy.unique()):
        s = mean_curve[mean_curve.strategy == strat].sort_values("n_labels")
        n_s = labels_to_reach(s.n_labels.values, s[metric].values, target)
        rows.append({
            "metric": metric, "random_budget": budget, "random_score": target,
            "strategy": strat, "labels_to_match": n_s,
            "label_saving": (budget - n_s) / budget if n_s is not None else None,
            "final_score": float(s[metric].iloc[-1]),
        })
    return pd.DataFrame(rows)


def savings_table(curves: pd.DataFrame, refs: pd.DataFrame, metric: str):
    """Secondary metric: label savings referenced to full supervision.

    Retained for comparability with studies that report it, but see
    savings_vs_random_endpoint -- on this dataset the full-supervised
    reference is *below* what active learning achieves, so these numbers
    saturate and should not be read as the headline.
    """
    ref_mean = refs[metric].mean()
    mean_curve = (curves.groupby(["strategy", "n_labels"])[metric]
                  .mean().reset_index())

    rows = []
    rnd = mean_curve[mean_curve.strategy == "random"].sort_values("n_labels")
    for target in TARGETS:
        thresh = target * ref_mean
        n_rand = labels_to_reach(rnd.n_labels.values, rnd[metric].values, thresh)
        for strat in sorted(mean_curve.strategy.unique()):
            s = mean_curve[mean_curve.strategy == strat].sort_values("n_labels")
            n_s = labels_to_reach(s.n_labels.values, s[metric].values, thresh)
            saving = None
            if n_rand is not None and n_s is not None and n_rand > 0:
                saving = (n_rand - n_s) / n_rand
            rows.append({
                "metric": metric, "target_frac": target,
                "target_score": thresh, "reference": ref_mean,
                "strategy": strat, "labels_needed": n_s,
                "labels_needed_random": n_rand,
                "label_saving": saving,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["a", "b"], default="b")
    args = ap.parse_args()
    track = args.track

    rdir = ROOT / "results"
    curves = pd.read_csv(rdir / f"curves_track_{track}.csv")
    refs = pd.read_csv(rdir / f"reference_track_{track}.csv")
    meta = json.loads((rdir / f"meta_track_{track}.json").read_text())
    classes = meta["classes"]

    # class prevalence, to correlate savings against rarity
    prev = {}
    src = ROOT / "data" / (f"track_{track}_vartype.parquet" if track == "a"
                           else f"track_{track}_simbad.parquet")
    lab = pd.read_parquet(src)["_label"].astype(str)
    lab = lab[lab.isin(classes)]
    for c in classes:
        prev[c] = float((lab == c).mean())

    print(f"=== Track {track.upper()} ===")
    print(f"full-supervised reference ({int(refs.n_labels.mean()):,} labels): "
          f"macro_f1={refs.macro_f1.mean():.4f} +/- {refs.macro_f1.std():.4f}")

    all_tables = [savings_table(curves, refs, "macro_f1"),
                  savings_table(curves, refs, "balanced_acc")]
    for c in classes:
        col = f"f1_{c}"
        if col in curves.columns:
            all_tables.append(savings_table(curves, refs, col))
    tab = pd.concat(all_tables, ignore_index=True)
    tab.to_csv(rdir / f"savings_track_{track}.csv", index=False)

    # ---- headline: labels needed to match random's endpoint ----
    print("\n--- PRIMARY: labels needed to match random sampling's final score ---")
    prim_all = []
    for metric in ["macro_f1", "balanced_acc"] + [f"f1_{c}" for c in classes]:
        if metric not in curves.columns:
            continue
        t = savings_vs_random_endpoint(curves, metric)
        if not t.empty:
            prim_all.append(t)
    prim = pd.concat(prim_all, ignore_index=True)
    prim.to_csv(rdir / f"primary_savings_track_{track}.csv", index=False)

    for metric in ("macro_f1", "balanced_acc"):
        sub = prim[prim.metric == metric]
        if sub.empty:
            continue
        b = int(sub.random_budget.iloc[0])
        tgt = sub.random_score.iloc[0]
        print(f"\n  {metric}: random reaches {tgt:.4f} at {b} labels")
        for _, r in sub.iterrows():
            if r.strategy == "random":
                continue
            if pd.isna(r.labels_to_match):
                print(f"    {r.strategy:<15} never matched "
                      f"(final {r.final_score:.4f})")
            else:
                print(f"    {r.strategy:<15} matched at {r.labels_to_match:>6.0f} "
                      f"labels  saving={r.label_saving:>6.1%}  "
                      f"(final {r.final_score:.4f})")

    print("\n  per-class (labels to match random's final class F1):")
    print(f"    {'class':<8}{'rand F1':>9}  " +
          "".join(f"{s:>15}" for s in
                  [x for x in sorted(curves.strategy.unique()) if x != "random"]))
    for c in sorted(classes, key=lambda z: prev[z]):
        sub = prim[prim.metric == f"f1_{c}"]
        if sub.empty:
            continue
        line = f"    {c:<8}{sub.random_score.iloc[0]:>9.3f}  "
        for s in [x for x in sorted(curves.strategy.unique()) if x != "random"]:
            r = sub[sub.strategy == s]
            if r.empty or pd.isna(r.labels_to_match.iloc[0]):
                line += f"{'never':>15}"
            else:
                line += f"{r.label_saving.iloc[0]:>14.1%} "
        print(line)

    # ---- secondary: overall macro-F1 savings vs full supervision ----
    print("\n--- SECONDARY: savings referenced to full supervision "
          "(saturates; see note) ---")
    m = tab[(tab.metric == "macro_f1")]
    for target in TARGETS:
        sub = m[m.target_frac == target]
        n_rand = sub.labels_needed_random.iloc[0]
        print(f"\n  target = {target:.0%} of full-supervised "
              f"({sub.target_score.iloc[0]:.4f}); random needs "
              f"{('%.0f' % n_rand) if n_rand is not None and pd.notna(n_rand) else 'never reached'}")
        for _, r in sub.iterrows():
            if r.strategy == "random":
                continue
            if pd.isna(r.labels_needed):
                print(f"    {r.strategy:<15} never reached")
            else:
                sv = r.label_saving
                sv_s = f"{sv:+7.1%}" if pd.notna(sv) else "    n/a"
                print(f"    {r.strategy:<15} {r.labels_needed:>7.0f} labels  saving={sv_s}")

    # ---- per-class savings at the 95% target ----
    print("\n--- Per-class label savings at 95% of full-supervised class F1 ---")
    strategies = [s for s in sorted(curves.strategy.unique()) if s != "random"]
    hdr = f"  {'class':<8}{'prev':>7}  " + "".join(f"{s:>16}" for s in strategies)
    print(hdr)
    per_class_rows = []
    for c in sorted(classes, key=lambda z: prev[z]):
        col = f"f1_{c}"
        sub = tab[(tab.metric == col) & (tab.target_frac == 0.95)]
        if sub.empty:
            continue
        line = f"  {c:<8}{prev[c]:>6.2%}  "
        row = {"class": c, "prevalence": prev[c]}
        for s in strategies:
            r = sub[sub.strategy == s]
            if r.empty or pd.isna(r.label_saving.iloc[0]):
                line += f"{'n/a':>16}"
                row[s] = np.nan
            else:
                v = float(r.label_saving.iloc[0])
                line += f"{v:>15.1%} "
                row[s] = v
        per_class_rows.append(row)
        print(line)

    pc = pd.DataFrame(per_class_rows)
    pc.to_csv(rdir / f"per_class_savings_track_{track}.csv", index=False)

    # ---- does savings correlate with rarity? ----
    print("\n--- Savings vs rarity (Spearman rho, negative = rarer classes save more) ---")
    for s in strategies:
        d = pc[["prevalence", s]].dropna()
        if len(d) >= 4:
            rho = d["prevalence"].corr(d[s], method="spearman")
            print(f"  {s:<15} rho={rho:+.3f}  (n={len(d)} classes)")
        else:
            print(f"  {s:<15} too few classes reached target (n={len(d)})")

    print(f"\nWrote {rdir / f'savings_track_{track}.csv'}")
    print(f"Wrote {rdir / f'per_class_savings_track_{track}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
