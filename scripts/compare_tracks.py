"""Contrast label efficiency when the target is a classifier's own output
(Track A, VarWISE `vartype`) versus independent literature labels
(Track B, SIMBAD types).

Motivation: a large share of astronomical ML papers train on catalog columns
that are themselves the output of an upstream classifier. If measured label
savings are systematically larger against such distillation targets than
against real labels, then label-efficiency numbers reported on catalog
columns are optimistic, and that gap is worth quantifying rather than
assuming away.

Run: python scripts/compare_tracks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RDIR = ROOT / "results"


def load(track):
    return (pd.read_csv(RDIR / f"savings_track_{track}.csv"),
            pd.read_csv(RDIR / f"reference_track_{track}.csv"))


def main():
    missing = [t for t in ("a", "b") if not (RDIR / f"savings_track_{t}.csv").exists()]
    if missing:
        print(f"Missing savings for track(s) {missing}; run analyze.py for both first.")
        return 1

    sa, ra = load("a")
    sb, rb = load("b")

    print("=== Full-supervised reference ===")
    print(f"  Track A (vartype, distillation): macro_f1={ra.macro_f1.mean():.4f}  "
          f"weighted_f1={ra.weighted_f1.mean():.4f}")
    print(f"  Track B (simbad, real labels)  : macro_f1={rb.macro_f1.mean():.4f}  "
          f"weighted_f1={rb.weighted_f1.mean():.4f}")

    print("\n=== Aggregate macro-F1 label savings vs random ===")
    print(f"  {'strategy':<16}{'target':>8}{'Track A':>12}{'Track B':>12}{'A - B':>10}")
    for target in (0.90, 0.95, 0.99):
        a = sa[(sa.metric == "macro_f1") & (sa.target_frac == target)]
        b = sb[(sb.metric == "macro_f1") & (sb.target_frac == target)]
        for strat in sorted(set(a.strategy) & set(b.strategy)):
            if strat == "random":
                continue
            va = a[a.strategy == strat].label_saving
            vb = b[b.strategy == strat].label_saving
            va = va.iloc[0] if len(va) else float("nan")
            vb = vb.iloc[0] if len(vb) else float("nan")
            fa = f"{va:.1%}" if pd.notna(va) else "n/a"
            fb = f"{vb:.1%}" if pd.notna(vb) else "n/a"
            gap = f"{va - vb:+.1%}" if pd.notna(va) and pd.notna(vb) else "n/a"
            print(f"  {strat:<16}{target:>7.0%}{fa:>12}{fb:>12}{gap:>10}")

    # per-class comparison on the classes both tracks share
    print("\n=== Per-class savings at 95% target, shared classes ===")
    pa = pd.read_csv(RDIR / "per_class_savings_track_a.csv")
    pb = pd.read_csv(RDIR / "per_class_savings_track_b.csv")
    shared = sorted(set(pa["class"]) & set(pb["class"]))
    strategies = [c for c in pa.columns if c not in ("class", "prevalence")]
    for s in strategies:
        print(f"\n  -- {s} --")
        print(f"    {'class':<8}{'A prev':>9}{'A save':>10}{'B prev':>9}{'B save':>10}")
        for c in shared:
            ra_ = pa[pa["class"] == c]
            rb_ = pb[pb["class"] == c]
            if ra_.empty or rb_.empty or s not in ra_ or s not in rb_:
                continue
            av, bv = ra_[s].iloc[0], rb_[s].iloc[0]
            print(f"    {c:<8}{ra_.prevalence.iloc[0]:>8.2%}"
                  f"{(f'{av:.1%}' if pd.notna(av) else 'n/a'):>10}"
                  f"{rb_.prevalence.iloc[0]:>8.2%}"
                  f"{(f'{bv:.1%}' if pd.notna(bv) else 'n/a'):>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
