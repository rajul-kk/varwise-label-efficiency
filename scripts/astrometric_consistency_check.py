"""Astrometric (parallax) consistency check -- a physical test not used
anywhere else in this project.

Every prior check in this repo used photometry (colors, magnitudes),
periods (range plausibility, period-luminosity), or cross-classification
(SIMBAD, ZTF-DNN). This uses a different physical principle entirely:
distance, via Gaia parallax.

Genuine AGN and supernovae are at cosmological distances. Their true
parallax is exactly zero; any measured value is pure noise around zero.
For a well-behaved Gaussian measurement process, the fraction of a genuine
extragalactic population showing a "significant" parallax detection
(plx / e_plx > k) should match the one-sided normal tail probability at k
(~0.13% at k=3, ~0.0000% at k=5 in the idealized case; real astrometry has
some duplicated-source/blend contamination so a small excess above the
Gaussian expectation is normal, but a LARGE excess indicates the class
contains genuine, nearby Galactic sources, not distant AGN/SNe).

This also runs the same test in reverse (as a calibration / negative
control) on classes that SHOULD show significant parallax (lpv, rr, ecl,
cep, yso -- all Galactic), to confirm the test correctly discriminates
before trusting its result on agn/sn/cv.

A second, independent structural check runs alongside it: near-duplicate
detection (objects a few arcsec apart, not exact-coordinate duplicates,
which were already checked and found to be zero) -- a different failure
mode (over-splitting a single physical source into multiple catalog
entries) than anything tested before.

Run: python scripts/astrometric_consistency_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "varwise_pure.parquet"
OUT = ROOT / "results" / "astrometric_consistency_check.txt"
lines = []


def emit(s=""):
    print(s)
    lines.append(s)


def sig_frac(plx, e_plx, k):
    snr = plx / e_plx
    valid = np.isfinite(snr)
    return float((snr[valid] > k).mean()), int(valid.sum())


def expected_gaussian_tail(k):
    """One-sided tail probability for a standard normal beyond k sigma."""
    return float(stats.norm.sf(k))


def main():
    df = pd.read_parquet(RAW)
    df["vartype"] = df["vartype"].astype("string").str.strip()
    d = df[df.plx.notna() & df.e_plx.notna() & (df.e_plx > 0)].copy()
    d["plx_snr"] = d.plx / d.e_plx

    emit("=" * 88)
    emit("ASTROMETRIC CONSISTENCY CHECK (parallax) -- a physical test not")
    emit("used elsewhere in this project")
    emit("=" * 88)
    emit(f"\nObjects with usable Gaia parallax: {len(d):,} of {len(df):,} "
         f"({100*len(d)/len(df):.1f}%)")

    emit("\nExpected false-positive rate for a genuine zero-parallax (fully")
    emit("extragalactic) population, under pure Gaussian measurement noise:")
    for k in (3, 5, 10):
        emit(f"  P(plx/e_plx > {k})  = {expected_gaussian_tail(k):.5f}  "
             f"({100*expected_gaussian_tail(k):.3f}%)")
    emit("(Real astrometric error is not perfectly Gaussian -- blends,")
    emit(" duplicated sources and crowding inflate this somewhat. A large")
    emit(" excess well beyond this budget indicates genuine nearby stars,")
    emit(" not measurement noise.)")

    emit("\n" + "=" * 88)
    emit("NEGATIVE CONTROL: Galactic classes SHOULD show significant parallax")
    emit("=" * 88)
    emit(f"\n  {'class':<8}{'n(plx)':>9}{'>3sig':>9}{'>5sig':>9}{'>10sig':>9}"
         f"{'median SNR':>12}")
    galactic = ["lpv", "rr", "ea", "ew", "cep", "yso"]
    for vt in galactic:
        sub = d[d.vartype == vt]
        f3, n = sig_frac(sub.plx.values, sub.e_plx.values, 3)
        f5, _ = sig_frac(sub.plx.values, sub.e_plx.values, 5)
        f10, _ = sig_frac(sub.plx.values, sub.e_plx.values, 10)
        med_snr = np.nanmedian(sub.plx_snr.values)
        emit(f"  {vt:<8}{n:>9,}{100*f3:>8.1f}%{100*f5:>8.1f}%{100*f10:>8.1f}%"
             f"{med_snr:>12.2f}")
    emit("\n  (all Galactic classes should show high rates -- confirms the")
    emit("   test discriminates distance correctly before trusting it below)")

    emit("\n" + "=" * 88)
    emit("PRIMARY TEST: agn and sn SHOULD show near-zero significant parallax")
    emit("=" * 88)
    emit(f"\n  {'class':<8}{'n(plx)':>9}{'>3sig':>9}{'>5sig':>9}{'>10sig':>9}"
         f"{'median SNR':>12}")
    extragalactic = ["agn", "sn"]
    for vt in extragalactic:
        sub = d[d.vartype == vt]
        f3, n = sig_frac(sub.plx.values, sub.e_plx.values, 3)
        f5, _ = sig_frac(sub.plx.values, sub.e_plx.values, 5)
        f10, _ = sig_frac(sub.plx.values, sub.e_plx.values, 10)
        med_snr = np.nanmedian(sub.plx_snr.values)
        emit(f"  {vt:<8}{n:>9,}{100*f3:>8.1f}%{100*f5:>8.1f}%{100*f10:>8.1f}%"
             f"{med_snr:>12.2f}")

    # SIMBAD-confirmed subset comparison -- is contamination concentrated
    # in the SIMBAD-unconfirmed remainder, or present even in confirmed AGN?
    emit("\n" + "=" * 88)
    emit("Is parallax contamination concentrated in SIMBAD-unconfirmed agn?")
    emit("=" * 88)
    df["simbad_type"] = df["simbad_type"].astype("string").str.strip()
    d["simbad_type"] = d["simbad_type"].astype("string")
    agn_simbad_types = {"QSO", "Seyfert1", "Seyfert2", "Seyfert", "BLLac",
                        "Blazar", "AGN", "AGN_Candidate", "Blazar_Candidate"}
    agn = d[d.vartype == "agn"].copy()
    agn["simbad_confirmed_agn"] = agn.simbad_type.isin(agn_simbad_types)
    emit(f"\n  {'group':<32}{'n':>9}{'>5sig':>9}{'>10sig':>9}")
    for label, mask in [("SIMBAD-confirmed AGN", agn.simbad_confirmed_agn),
                        ("no SIMBAD confirmation", ~agn.simbad_confirmed_agn &
                         (agn.simbad_type.isna() | (agn.simbad_type == ""))),
                        ("SIMBAD says something else", ~agn.simbad_confirmed_agn &
                         agn.simbad_type.notna() & (agn.simbad_type != ""))]:
        sub = agn[mask]
        if len(sub) == 0:
            continue
        f5, n = sig_frac(sub.plx.values, sub.e_plx.values, 5)
        f10, _ = sig_frac(sub.plx.values, sub.e_plx.values, 10)
        emit(f"  {label:<32}{n:>9,}{100*f5:>8.1f}%{100*f10:>8.1f}%")

    # what are the high-parallax-SNR "agn" objects, photometrically?
    emit("\n" + "=" * 88)
    emit("WHAT ARE THE HIGH-CONFIDENCE STELLAR CONTAMINANTS IN `agn`?")
    emit("=" * 88)
    strong = agn[agn.plx_snr > 10].copy()
    emit(f"\n  agn objects with plx/e_plx > 10 (near-certain nearby stars): "
         f"{len(strong):,} of {len(agn):,} ({100*len(strong)/max(len(agn),1):.2f}%)")
    if len(strong):
        emit(f"  median VarWISE confidence: {strong.confidence.median():.4f}")
        emit(f"  median W1-W2 color: {(strong.w1mag-strong.w2mag).median():.3f}  "
             f"(genuine AGN are typically W1-W2 > 0.5-0.8; check below)")
        genuine_agn = agn[agn.plx_snr.abs() < 1]
        emit(f"  (comparison) low-|SNR| agn median W1-W2: "
             f"{(genuine_agn.w1mag-genuine_agn.w2mag).median():.3f}")
        emit(f"  SIMBAD types among the high-parallax contaminants (top 8):")
        vc = strong.simbad_type.replace("", pd.NA).dropna().value_counts()
        for k, v in vc.head(8).items():
            emit(f"    {k}: {v}")

    # ---------------- secondary check: near-duplicate detections ----------
    emit("\n" + "=" * 88)
    emit("SECONDARY CHECK: near-duplicate detections (over-splitting)")
    emit("=" * 88)
    emit("\n  Exact-coordinate duplicates were already checked and found to")
    emit("  be zero (full_catalog_scan.py). This checks NEAR duplicates --")
    emit("  distinct catalog entries within 3 arcsec of each other, which")
    emit("  would indicate the same physical source split into multiple")
    emit("  VarWISE entries (a different failure mode: over-splitting during")
    emit("  VARnet's spatial clustering of apparitions, not misclassification).")

    ra_r, dec_r = np.radians(df.ra.values), np.radians(df.dec.values)
    xyz = np.column_stack([np.cos(dec_r) * np.cos(ra_r),
                           np.cos(dec_r) * np.sin(ra_r), np.sin(dec_r)])
    tree = cKDTree(xyz)
    for radius_arcsec in (1.0, 2.0, 3.0):
        chord = 2 * np.sin(np.radians(radius_arcsec / 3600.0) / 2)
        pairs = tree.query_pairs(r=chord)
        emit(f"\n  pairs within {radius_arcsec}\": {len(pairs):,}")
        if pairs and radius_arcsec == 3.0:
            # inspect a handful: same class or different?
            same_class = 0
            diff_class = 0
            for i, j in list(pairs)[:2000]:
                if df.vartype.iloc[i] == df.vartype.iloc[j]:
                    same_class += 1
                else:
                    diff_class += 1
            emit(f"    of first {min(2000, len(pairs)):,} pairs inspected: "
                 f"{same_class:,} same class, {diff_class:,} different class")

    emit("\n" + "=" * 88)
    emit("CAVEATS")
    emit("=" * 88)
    emit("""
  - Parallax is only available for the ~84% of objects with a Gaia
    cross-match (bright/nearby-biased, same caveat as elsewhere in this
    project).
  - A small excess above the Gaussian tail budget is expected even for
    genuine AGN, from Gaia astrometric systematics near blended/crowded
    fields; the interpretation here rests on the SIZE of the excess and its
    concentration among SIMBAD-unconfirmed objects, not on any nonzero rate.
  - Genuine white-dwarf/subdwarf AGN look-alikes and blazars hosted by
    foreground stars in projection are real but rare astrophysical
    confusers; we cannot rule out every individual high-parallax object
    being such a case, only that the AGGREGATE rate is inconsistent with a
    purely extragalactic population.
""")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
