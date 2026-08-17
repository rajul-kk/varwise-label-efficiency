"""Emit the study's headline tables in markdown, straight from the result CSVs.

Everything quoted in RESULTS.md is generated here so the writeup cannot drift
from the numbers.

Run: python scripts/summarize.py > RESULTS_tables.md
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RDIR = ROOT / "results"


PCT_COLS = {"prevalence", "label saving"}
INT_COLS = {"labels", "n_labels"}


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and np.isfinite(v):
                if c in PCT_COLS:
                    cells.append(f"{v:.2%}")
                elif c in INT_COLS or str(c).startswith("labels to match"):
                    cells.append(f"{v:,.0f}")
                else:
                    cells.append(floatfmt.format(v))
            elif v is None or (isinstance(v, float) and not np.isfinite(v)):
                cells.append("n/a")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def section_curves(track, tag="", label=None):
    suffix = f"{track}{tag}"
    p = RDIR / f"curves_track_{suffix}.csv"
    if not p.exists():
        return None
    curves = pd.read_csv(p)
    refs = pd.read_csv(RDIR / f"reference_track_{suffix}.csv")
    meta = json.loads((RDIR / f"meta_track_{suffix}.json").read_text())
    classes = meta["classes"]
    budget = int(curves.n_labels.max())
    label = label or f"Track {track.upper()}"

    print(f"\n### {label} — final scores at {budget} labels\n")
    last = curves[curves.n_labels == budget]
    agg = last.groupby("strategy").agg(
        macro_f1=("macro_f1", "mean"), sd=("macro_f1", "std"),
        balanced_acc=("balanced_acc", "mean")).reset_index()
    agg = agg.sort_values("macro_f1", ascending=False)
    agg.columns = ["strategy", "macro F1", "sd", "balanced acc"]
    print(md_table(agg))
    print(f"\nFull-supervised reference ({int(refs.n_labels.mean()):,} labels): "
          f"macro F1 = {refs.macro_f1.mean():.3f} "
          f"(sd {refs.macro_f1.std():.3f}), "
          f"weighted F1 = {refs.weighted_f1.mean():.3f}")

    # per-class final F1: best AL strategy vs random
    print(f"\n### {label} — per-class F1 at {budget} labels, "
          "best active strategy vs random\n")
    rows = []
    src = ROOT / "data" / (f"track_{track}_vartype.parquet" if track == "a"
                           else f"track_{track}_simbad.parquet")
    lab = pd.read_parquet(src)["_label"].astype(str)
    lab = lab[lab.isin(classes)]
    for c in classes:
        col = f"f1_{c}"
        if col not in last.columns:
            continue
        rnd = last[last.strategy == "random"][col].mean()
        act = last[last.strategy != "random"].groupby("strategy")[col].mean()
        act = act.drop(index=["quota"], errors="ignore")
        best_s, best_v = act.idxmax(), act.max()
        rows.append({"class": c, "prevalence": float((lab == c).mean()),
                     "random F1": rnd, "best active F1": best_v,
                     "best strategy": best_s, "gap": best_v - rnd})
    df = pd.DataFrame(rows).sort_values("prevalence")
    print(md_table(df))
    return curves


def section_primary(track, tag="", label=None):
    p = RDIR / f"primary_savings_track_{track}{tag}.csv"
    if not p.exists():
        return
    prim = pd.read_csv(p)
    label = label or f"Track {track.upper()}"
    print(f"\n### {label} — labels to match random sampling's "
          "final score\n")
    sub = prim[prim.metric == "macro_f1"].copy()
    sub = sub[sub.strategy != "random"]
    b = int(sub.random_budget.iloc[0])
    sub = sub[["strategy", "labels_to_match", "label_saving", "final_score"]]
    sub.columns = ["strategy", f"labels to match ({b} budget)",
                   "label saving", "final macro F1"]
    print(md_table(sub.sort_values("label saving", ascending=False)))


def section_diagnostic():
    fs = sorted(glob.glob(str(RDIR / "diagnose_gap_track_b_seed*.csv")))
    if not fs:
        return
    d = pd.concat([pd.read_csv(f).assign(seed=Path(f).stem[-1]) for f in fs])
    d["delta"] = d.al_f1 - d.ctrl_f1
    d["prev"] = d.pool_n / d.groupby("seed").pool_n.transform("sum")
    g = d.groupby("class").agg(
        prevalence=("prev", "mean"),
        al_f1=("al_f1", "mean"),
        matched_random_f1=("ctrl_f1", "mean"),
        gain=("delta", "mean"),
        gain_sd=("delta", "std"),
    ).reset_index().sort_values("prevalence")
    g.columns = ["class", "prevalence", "AL F1", "matched-random F1",
                 "gain", "gain sd"]
    print(f"\n### Informativeness beyond rebalancing "
          f"({len(fs)} seeds)\n")
    print("Random draw with *identical per-class counts* to the AL labeled "
          "set, so class composition is held fixed and only example choice "
          "differs.\n")
    print(md_table(g))

    sub = g.dropna(subset=["gain"])
    if len(sub) >= 4:
        rho = sub["prevalence"].corr(sub["gain"], method="spearman")
        print(f"\nSpearman rho(prevalence, gain) = **{rho:+.3f}** "
              f"over {len(sub)} classes — the rarer the class, the more the "
              "acquisition function adds.")


def section_reference_variants():
    p = RDIR / "reference_variants_track_b.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    g = d.groupby("variant").agg(
        n_labels=("n_labels", "mean"), macro_f1=("macro_f1", "mean"),
        sd=("macro_f1", "std"), balanced_acc=("balanced_acc", "mean"),
        f1_cv=("f1_cv", "mean")).reset_index().sort_values("macro_f1")
    g.columns = ["reference variant", "labels", "macro F1", "sd",
                 "balanced acc", "cv F1"]
    print("\n### Full-supervised reference variants (Track B)\n")
    print(md_table(g, floatfmt="{:.3f}"))


def main():
    print("<!-- generated by scripts/summarize.py; do not edit by hand -->")
    runs = [
        ("b", "", "Track B (SIMBAD labels, LightGBM)"),
        ("b", "_xgb", "Track B (SIMBAD labels, XGBoost)"),
        ("a", "", "Track A (VarWISE vartype, LightGBM)"),
        ("a", "_xgb", "Track A (VarWISE vartype, XGBoost)"),
    ]
    for track, tag, label in runs:
        section_curves(track, tag, label)
        section_primary(track, tag, label)
    section_reference_variants()
    section_diagnostic()
    return 0


if __name__ == "__main__":
    sys.exit(main())
