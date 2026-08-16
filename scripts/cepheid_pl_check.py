"""Extend the period-luminosity validation to Cepheids.

The RR Lyrae PL check (scripts/pl_relation_check.py) was decisive: Extended
`rr` sat 3 magnitudes off the expected locus, and the recommended confidence
cut did nothing about it while period_significance > 20 fixed it.

Cepheid PL relations are the tightest standard-candle relation in astronomy
(intrinsic scatter ~0.1-0.2 mag in the mid-IR, versus RR Lyrae's near-
degenerate ~0.5 mag scatter at fixed magnitude). If VarWISE's Cepheid periods
are unreliable in the same way as its RR Lyrae periods, a PL check should
show it even more clearly than the RR Lyrae case did, because the relation
being tested against has much less intrinsic slack to hide behind.

Uses the same M_W1 = W1 + 5*log10(plx_mas) - 10 definition and the same
robust-fit machinery as scripts/pl_relation_check.py, applied to the `cep`
class the earlier script already downloaded (data/raw/pl_sample.parquet).

Run: python scripts/cepheid_pl_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "pl_sample.parquet"
OUT = ROOT / "results" / "cepheid_pl_check.txt"
lines = []


def emit(s=""):
    print(s)
    lines.append(s)


def abs_mag(m, plx_mas):
    return m + 5 * np.log10(plx_mas) - 10


def robust_fit(logp, absm, n_iter=5, clip=3.0):
    keep = np.isfinite(logp) & np.isfinite(absm)
    if keep.sum() < 15:
        return np.nan, np.nan, np.nan, keep
    for _ in range(n_iter):
        if keep.sum() < 15:
            break
        b, a = np.polyfit(logp[keep], absm[keep], 1)
        resid = absm - (a + b * logp)
        s = 1.4826 * np.median(np.abs(resid[keep] - np.median(resid[keep])))
        if not np.isfinite(s) or s == 0:
            break
        keep = np.isfinite(logp) & np.isfinite(absm) & (np.abs(resid) < clip * s)
    return b, a, s, keep


def report_fit(sub, label, min_n=15):
    if len(sub) < min_n:
        emit(f"  {label:<48}{len(sub):>8,}      (too few)")
        return None
    logp = np.log10(sub.period1.values)
    absm = abs_mag(sub.w1mag.values, sub.plx.values)
    b, a, s, keep = robust_fit(logp, absm)
    if not np.isfinite(s):
        emit(f"  {label:<48}{len(sub):>8,}      (fit failed)")
        return None
    emit(f"  {label:<48}{len(sub):>8,}{s:>10.3f}{100*keep.sum()/len(sub):>9.1f}%"
         f"{b:>9.2f}{a:>9.2f}")
    return s


def main():
    if not CACHE.exists():
        print(f"Missing {CACHE}. Run scripts/pl_relation_check.py first "
              f"(it builds this cache for lpv/rr/cep across both tiers).")
        return 1

    df = pd.read_parquet(CACHE)
    df = df[np.isfinite(df.w1mag) & (df.plx > 0) & (df.period1 > 0)].copy()
    cep = df[df.vartype == "cep"]

    emit("=" * 88)
    emit("CEPHEID PERIOD-LUMINOSITY VALIDATION")
    emit("=" * 88)
    emit(f"\nSample: cep with period + parallax S/N > 5")
    emit(f"  Pure:     {int((cep.tier=='pure').sum()):>7,}")
    emit(f"  Extended: {int((cep.tier=='extended').sum()):>7,}")
    emit("\nMid-IR Cepheid PL relations have intrinsic scatter ~0.1-0.2 mag --")
    emit("far tighter than RR Lyrae's near-degenerate relation. If periods are")
    emit("bad, this relation has much less slack to absorb it, so contamination")
    emit("should show up as scatter more clearly here than for `rr`.")

    emit("\n" + "=" * 88)
    emit("SCATTER UNDER SUCCESSIVE CUTS")
    emit("=" * 88)
    emit(f"\n  {'sample':<48}{'n':>8}{'scatter':>10}{'kept':>9}{'slope':>9}"
         f"{'intercept':>9}")
    for tier in ("pure", "extended"):
        sub = cep[cep.tier == tier]
        report_fit(sub, f"{tier}, no cuts")
        report_fit(sub[sub.confidence >= 0.9], f"{tier}, confidence >= 0.9")
        report_fit(sub[sub.period_significance > 20],
                   f"{tier}, period_significance > 20")
        report_fit(sub[sub.suspect_period == 0], f"{tier}, suspect_period = 0")
        emit("  " + "-" * 82)

    # ---------------- absolute magnitude comparison ----------------
    emit("\n" + "=" * 88)
    emit("ABSOLUTE MAGNITUDE BY TIER (the decisive test used for RR Lyrae)")
    emit("=" * 88)
    emit("\n  Unlike RR Lyrae (fixed M_W1), Cepheid M_W1 depends on period via")
    emit("  the PL relation itself, so there is no single reference magnitude.")
    emit("  Instead: does the Pure-tier PL FIT predict the Extended-tier")
    emit("  points well, or do they scatter off it systematically?\n")

    pure_cep = cep[(cep.tier == "pure") & (cep.confidence >= 0.9)]
    if len(pure_cep) >= 15:
        logp = np.log10(pure_cep.period1.values)
        absm = abs_mag(pure_cep.w1mag.values, pure_cep.plx.values)
        b, a, s, keep = robust_fit(logp, absm)
        emit(f"  Pure-tier reference fit (confidence>=0.9): "
             f"M_W1 = {a:.2f} + {b:.2f}*log10(P), scatter={s:.3f}")

        for tier, extra, label in [
                ("extended", "", "Extended, no cuts"),
                ("extended", "conf", "Extended, confidence >= 0.9"),
                ("extended", "psig", "Extended, period_significance > 20")]:
            sub = cep[cep.tier == tier]
            if extra == "conf":
                sub = sub[sub.confidence >= 0.9]
            elif extra == "psig":
                sub = sub[sub.period_significance > 20]
            if len(sub) < 15:
                continue
            logp_e = np.log10(sub.period1.values)
            absm_e = abs_mag(sub.w1mag.values, sub.plx.values)
            pred = a + b * logp_e
            resid = absm_e - pred
            good = np.isfinite(resid)
            rms = np.sqrt(np.mean(resid[good] ** 2))
            median_offset = np.median(resid[good])
            emit(f"  {label:<40} n={good.sum():>7,}  "
                 f"RMS off Pure fit={rms:.3f}  median offset={median_offset:+.3f}")

    emit("\n" + "=" * 88)
    emit("CAVEATS")
    emit("=" * 88)
    emit("""
  - Same Lutz-Kelker / no-extinction-correction caveats as the RR Lyrae
    check. Cepheids are typically more distant and more affected by
    extinction than RR Lyrae, which could inflate scatter independent of
    period quality -- read the comparison as relative between cuts, not as
    an absolute scatter budget.
  - VarWISE's own inspection already flags Cepheid periods near 6 months/1
    year as suspect due to the survey's cadence (see RESULTS.md); this test
    is complementary to, not a replacement for, that check.
  - Sample sizes are much smaller than the RR Lyrae case (thousands, not
    hundreds of thousands), so results here carry more sampling noise.
""")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
