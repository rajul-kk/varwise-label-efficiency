"""Plot label-efficiency curves: overall and per-class.

Run: python scripts/plot_curves.py --track b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "random": "#888888",
    "uncertainty": "#1f77b4",
    "margin": "#d62728",
    "class_balanced": "#2ca02c",
    "quota": "#ff7f0e",
    "prototype": "#9467bd",
}


def band(ax, curves, metric, ref_val=None):
    for strat in sorted(curves.strategy.unique()):
        s = curves[curves.strategy == strat]
        g = s.groupby("n_labels")[metric]
        mean, std = g.mean(), g.std()
        c = COLORS.get(strat, None)
        ax.plot(mean.index, mean.values, label=strat, color=c,
                lw=2.0 if strat == "random" else 1.6,
                ls="--" if strat == "random" else "-")
        ax.fill_between(mean.index, mean - std, mean + std, color=c, alpha=0.13, lw=0)
    if ref_val is not None:
        ax.axhline(ref_val, color="k", ls=":", lw=1.2)
        ax.axhline(0.95 * ref_val, color="k", ls="-.", lw=0.8, alpha=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["a", "b"], default="b")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    suffix = f"{args.track}{args.tag}"

    rdir = ROOT / "results"
    curves = pd.read_csv(rdir / f"curves_track_{suffix}.csv")
    refs = pd.read_csv(rdir / f"reference_track_{suffix}.csv")
    meta = json.loads((rdir / f"meta_track_{suffix}.json").read_text())
    classes = meta["classes"]

    # ---- overall ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    band(axes[0], curves, "macro_f1", refs.macro_f1.mean())
    axes[0].set_xlabel("labels acquired")
    axes[0].set_ylabel("macro F1")
    axes[0].set_title(f"Track {args.track.upper()}: overall label efficiency")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    band(axes[1], curves, "balanced_acc", refs.balanced_acc.mean())
    axes[1].set_xlabel("labels acquired")
    axes[1].set_ylabel("balanced accuracy")
    axes[1].set_title("Balanced accuracy")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    p = rdir / f"curve_overall_track_{suffix}.png"
    fig.savefig(p, dpi=140)
    print(f"Wrote {p}")

    # ---- per class ----
    cols = [c for c in classes if f"f1_{c}" in curves.columns]
    n = len(cols)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    for i, c in enumerate(cols):
        ax = axes[i // ncol][i % ncol]
        band(ax, curves, f"f1_{c}", refs[f"f1_{c}"].mean())
        ax.set_title(f"{c}")
        ax.set_xlabel("labels")
        ax.set_ylabel("F1")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Track {args.track.upper()}: per-class label efficiency", y=1.00)
    fig.tight_layout()
    p = rdir / f"curve_per_class_track_{suffix}.png"
    fig.savefig(p, dpi=140)
    print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
