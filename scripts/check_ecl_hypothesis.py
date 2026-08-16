"""Follow up on the unresolved `ecl` correction hypothesis.

scripts/apply_transient_fix.py flagged 21,289 objects as predicted `ecl`
but low-reliability (the ecl sub-model was trained on only 344 SIMBAD-
confirmed examples, CV F1 0.598). CATALOG_ASSESSMENT.md named this "the
clearest target for follow-up" and left it there.

A cheap, independent check: eclipsing binaries are, by definition, strongly
periodic -- that's the entire physical basis for classifying something as
`ea`/`ew` in the first place. If these 21,289 objects genuinely are
eclipsing binaries the rule swept into `cv`, they should show significant
periodicity (period_significance comparable to VarWISE's own confirmed ecl
population). If they don't, the classifier is very likely defaulting to
`ecl` as some kind of majority-class/catch-all behavior rather than
detecting real eclipsing signal, and the hypothesis should be downgraded
further, not just flagged low-reliability.

Run: python scripts/check_ecl_hypothesis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORR = ROOT / "results" / "varwise_transient_corrections.csv"
PURE = ROOT / "data" / "raw" / "varwise_pure.parquet"
EXT = ROOT / "data" / "raw" / "varwise_ext_transients.parquet"
OUT = ROOT / "results" / "ecl_hypothesis_check.txt"
lines = []


def emit(s=""):
    print(s)
    lines.append(s)


def main():
    if not CORR.exists():
        print(f"Missing {CORR}. Run scripts/apply_transient_fix.py first.")
        return 1

    corr = pd.read_csv(CORR)
    pure = pd.read_parquet(PURE)[["cluster_id", "period1", "period_significance",
                                   "suspect_period", "vartype"]]
    ext = pd.read_parquet(EXT)[["cluster_id", "period1", "period_significance",
                                 "suspect_period"]]

    # join period info back onto the corrections table by tier
    p_pure = corr[corr.tier == "pure"].merge(pure, on="cluster_id", how="left")
    p_ext = corr[corr.tier == "extended"].merge(ext, on="cluster_id", how="left")
    joined = pd.concat([p_pure, p_ext], ignore_index=True)

    emit("=" * 86)
    emit("FOLLOW-UP: is the low-reliability `ecl` correction real periodicity")
    emit("or a classifier default?")
    emit("=" * 86)

    ecl_low = joined[(joined.corrected_class == "ecl") & (joined.reliability == "low")]
    ecl_hi = joined[(joined.corrected_class == "ecl") &
                    (joined.reliability.isin(["validated", "high"]))]
    emit(f"\n  predicted ecl, low reliability:  {len(ecl_low):,}")
    emit(f"  predicted ecl, validated/high:   {len(ecl_hi):,}")

    # reference: VarWISE's own confirmed ea/ew population (the audit already
    # validated this class at F1 0.99 against SIMBAD)
    ref = pure[pure.vartype.isin(["ea", "ew"])]
    ref = ref[ref.period_significance.notna()]

    emit("\n" + "=" * 86)
    emit("PERIOD SIGNIFICANCE: hypothesis population vs a genuine ecl reference")
    emit("=" * 86)
    emit(f"\n  {'group':<38}{'n':>9}{'median psig':>13}{'% psig>20':>11}"
         f"{'% no period':>13}")
    for label, sub in [
            ("Reference: VarWISE ea/ew (validated F1~0.99)", ref),
            ("Predicted ecl, low reliability", ecl_low),
            ("Predicted ecl, validated/high reliability", ecl_hi)]:
        n = len(sub)
        if n == 0:
            continue
        no_period = sub.period1.isna().mean() if "period1" in sub else np.nan
        psig = sub.period_significance.dropna()
        med = psig.median() if len(psig) else np.nan
        frac20 = (psig > 20).mean() if len(psig) else np.nan
        emit(f"  {label:<38}{n:>9,}{med:>13.2f}{100*frac20:>10.1f}%"
             f"{100*no_period:>12.1f}%")

    emit("\n" + "=" * 86)
    emit("INTERPRETATION")
    emit("=" * 86)
    ref_med = ref.period_significance.median()
    low_med = ecl_low.period_significance.median() if len(ecl_low) else np.nan
    hi_med = ecl_hi.period_significance.median() if len(ecl_hi) else np.nan
    no_period_low = ecl_low.period1.isna().mean() if len(ecl_low) else np.nan

    if pd.notna(low_med) and low_med < 0.3 * ref_med:
        emit(f"""
  The low-reliability `ecl` predictions have median period_significance
  {low_med:.2f}, far below the genuine ea/ew reference ({ref_med:.2f}) --
  less than a third. {100*no_period_low:.0f}% have no usable period at all,
  which is disqualifying for a class defined by periodicity.

  VERDICT: the hypothesis is NOT supported. These are not eclipsing binaries
  the classifier correctly identified from weak training data; the model is
  defaulting to `ecl` as something closer to a residual/catch-all class when
  it isn't confident about lpv/agn/yso, rather than detecting genuine
  eclipsing signal. The `reliability=low` flag in
  varwise_transient_corrections.csv is doing its job -- users should treat
  these 21,289 rows as unclassified, not as tentative eclipsing binaries.
""")
    elif pd.notna(low_med) and low_med >= 0.7 * ref_med:
        emit(f"""
  The low-reliability `ecl` predictions have median period_significance
  {low_med:.2f}, comparable to the genuine reference ({ref_med:.2f}).

  VERDICT: the hypothesis is PARTIALLY supported -- these objects do show
  real periodic signal consistent with eclipsing binaries, despite the
  model's own cross-validated F1 being weak (which reflects limited training
  data, not necessarily wrong predictions on this specific subset). Worth a
  closer follow-up (e.g. light-curve inspection of a sample) rather than a
  blanket "do not use."
""")
    else:
        emit(f"""
  The low-reliability `ecl` predictions have median period_significance
  {low_med:.2f} vs the genuine reference's {ref_med:.2f} -- meaningfully
  lower but not negligible.

  VERDICT: mixed. Some genuine periodic signal is present, but weaker than
  a confirmed eclipsing-binary population. Consistent with a mix of some
  real eclipsing binaries and some model over-assignment. The low-
  reliability flag remains the right call; this does not clear the
  population for use, but it also isn't pure noise.
""")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
