"""Verify the downloaded VarWISE data actually matches what IRSA serves.

Everything so far checked internal consistency: does result X follow
correctly from stored parquet Y. This checks something different and more
basic: does stored parquet Y actually match the live source it claims to be
a copy of. Specifically:

  1. Fresh row counts, right now, against what was downloaded.
  2. Row-for-row, column-for-column comparison of a random sample of already-
     downloaded objects against a fresh independent re-fetch by cluster_id.
  3. RA-slice boundary integrity - the download used 12 RA slices with
     `>= lo AND < hi`; check objects near every seam for duplication or gaps.
  4. String-column integrity (designation encoding, byte corruption).
  5. Cross-check the Extended-tier cv/sn cache and the PL-relation cache the
     same way.

Run: python scripts/verify_download_integrity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyvo

ROOT = Path(__file__).resolve().parents[1]
TAP_URL = "https://irsa.ipac.caltech.edu/TAP"
tap = pyvo.dal.TAPService(TAP_URL)

RAW = ROOT / "data" / "raw" / "varwise_pure.parquet"
OUT = ROOT / "results" / "download_integrity_check.txt"
lines = []
ISSUES = []


def emit(s=""):
    print(s)
    lines.append(s)


def issue(msg):
    ISSUES.append(msg)
    emit(f"    [ISSUE] {msg}")


def decode_votable(df):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
    return df


def main():
    df = pd.read_parquet(RAW)
    emit("=" * 86)
    emit("DOWNLOAD INTEGRITY VERIFICATION")
    emit("=" * 86)
    emit(f"\nStored parquet: {len(df):,} rows x {len(df.columns)} columns")

    # ---------------- 1. fresh row counts ----------------
    emit("\n" + "=" * 86)
    emit("1. FRESH ROW COUNTS (queried live, right now)")
    emit("=" * 86)
    live_pure = int(tap.search("SELECT COUNT(*) AS n FROM varwisepure"
                               ).to_table()[0]["n"])
    live_ext = int(tap.search("SELECT COUNT(*) AS n FROM varwiseext"
                              ).to_table()[0]["n"])
    emit(f"\n  varwisepure live count: {live_pure:,}  (stored: {len(df):,})")
    if live_pure != len(df):
        issue(f"live Pure count {live_pure:,} != stored {len(df):,}")
    else:
        emit("    MATCH")
    emit(f"  varwiseext  live count: {live_ext:,}  (paper reports 1,918,082)")
    if live_ext != 1918082:
        issue(f"live Extended count {live_ext:,} != paper's published 1,918,082")
    else:
        emit("    MATCH")

    # ---------------- 2. row-level spot check ----------------
    emit("\n" + "=" * 86)
    emit("2. ROW-LEVEL SPOT CHECK - fresh re-fetch vs stored, by cluster_id")
    emit("=" * 86)
    rng = np.random.default_rng(12345)
    sample_ids = rng.choice(df.cluster_id.values, size=60, replace=False)
    id_list = ",".join(str(int(i)) for i in sample_ids)
    cols = [c for c in df.columns if c not in ("x", "y", "z", "htm20")]
    sql = f"SELECT {', '.join(cols)} FROM varwisepure WHERE cluster_id IN ({id_list})"
    fresh = tap.search(sql).to_table().to_pandas()
    fresh = decode_votable(fresh)

    emit(f"\n  re-fetched {len(fresh):,} of {len(sample_ids)} sampled cluster_ids")
    if len(fresh) != len(sample_ids):
        issue(f"re-fetch returned {len(fresh):,} rows for {len(sample_ids)} "
              f"requested cluster_ids")

    merged = df[df.cluster_id.isin(sample_ids)][cols].merge(
        fresh, on="cluster_id", suffixes=("_stored", "_fresh"))
    mismatches = 0
    num_cols = [c for c in cols if c not in ("cluster_id", "designation",
                                             "vartype", "simbad_type")]
    for c in num_cols:
        a, b = merged[f"{c}_stored"], merged[f"{c}_fresh"]
        both_nan = a.isna() & b.isna()
        diff = (~both_nan) & (~np.isclose(a.fillna(-999999), b.fillna(-999999),
                                          rtol=1e-6, atol=1e-9))
        if diff.any():
            mismatches += int(diff.sum())
            issue(f"{int(diff.sum())} value mismatches in column '{c}'")
    for c in ("designation", "vartype", "simbad_type"):
        a, b = merged[f"{c}_stored"].astype(str), merged[f"{c}_fresh"].astype(str)
        diff = a != b
        if diff.any():
            mismatches += int(diff.sum())
            issue(f"{int(diff.sum())} string mismatches in column '{c}'")
    emit(f"\n  compared {len(merged):,} rows x {len(cols)-1} columns "
         f"({len(merged)*(len(cols)-1):,} cell comparisons)")
    emit(f"  mismatches found: {mismatches}")
    if mismatches == 0:
        emit("  ALL VALUES MATCH the live source exactly")

    # ---------------- 3. RA slice boundary check ----------------
    emit("\n" + "=" * 86)
    emit("3. RA-SLICE BOUNDARY INTEGRITY (download used 12 slices of 30 deg)")
    emit("=" * 86)
    edges = [i * 30.0 for i in range(13)]
    emit(f"\n  {'boundary':>10}{'stored n (+/-0.01deg)':>24}{'live n':>10}{'status':>10}")
    boundary_bad = 0
    for edge in edges[1:-1]:  # skip 0 and 360 (not real seams)
        lo, hi = edge - 0.01, edge + 0.01
        stored_n = int(((df.ra >= lo) & (df.ra < hi)).sum())
        live_n = int(tap.search(
            f"SELECT COUNT(*) AS n FROM varwisepure WHERE ra >= {lo} AND ra < {hi}"
        ).to_table()[0]["n"])
        status = "OK" if stored_n == live_n else "MISMATCH"
        if stored_n != live_n:
            boundary_bad += 1
        emit(f"  {edge:>10.0f}{stored_n:>24,}{live_n:>10,}{status:>10}")
    if boundary_bad:
        issue(f"{boundary_bad} RA slice boundaries show a stored/live count mismatch")
    else:
        emit("\n  no duplication or gaps at any of the 11 internal RA seams")

    # duplicate check across the whole downloaded set (would catch seam double-counts)
    dup = int(df.cluster_id.duplicated().sum())
    emit(f"\n  duplicate cluster_id in the full downloaded set: {dup}")
    if dup:
        issue(f"{dup} duplicate cluster_id rows in the stored download")

    # ---------------- 4. string integrity ----------------
    emit("\n" + "=" * 86)
    emit("4. STRING-COLUMN INTEGRITY")
    emit("=" * 86)
    des = df.designation.astype(str)
    bad_null_byte = int(des.str.contains("\x00", regex=False).sum())
    bad_replacement = int(des.str.contains("�", regex=False).sum())
    emit(f"\n  designations containing a null byte: {bad_null_byte}")
    emit(f"  designations containing a Unicode replacement char (decode failure): "
         f"{bad_replacement}")
    if bad_null_byte or bad_replacement:
        issue(f"{bad_null_byte + bad_replacement} designations show byte-decode "
              f"corruption")
    # VarWISE designations follow a fixed sexagesimal pattern (catalog-
    # specific prefix "VarWISE J", confirmed against the actual data rather
    # than assumed from generic WISE naming conventions)
    pat = r"^VarWISE J\d{6}\.\d{2}[+-]\d{6}\.\d$"
    pattern_ok = des.str.match(pat, na=False).mean()
    emit(f"  designations matching 'VarWISE Jhhmmss.ss+ddmmss.s': "
         f"{100*pattern_ok:.1f}%")
    if pattern_ok < 0.99:
        sample_bad = des[~des.str.match(pat, na=False)].head(3).tolist()
        issue(f"only {100*pattern_ok:.1f}% of designations match the expected "
              f"pattern; examples: {sample_bad}")

    # ---------------- 5. cross-check derived caches ----------------
    emit("\n" + "=" * 86)
    emit("5. DERIVED-CACHE CROSS-CHECKS")
    emit("=" * 86)

    ext_cache = ROOT / "data" / "raw" / "varwise_ext_transients.parquet"
    if ext_cache.exists():
        ec = pd.read_parquet(ext_cache)
        emit(f"\n  varwise_ext_transients.parquet: {len(ec):,} rows")
        for vt, expected in (("cv", 69418), ("sn", 9875)):
            live_n = int(tap.search(
                f"SELECT COUNT(*) AS n FROM varwiseext WHERE vartype='{vt}'"
            ).to_table()[0]["n"])
            stored_n = int((ec.vartype == vt).sum())
            emit(f"    {vt}: cached={stored_n:,}  live={live_n:,}  "
                 f"reported-in-paper-analysis={expected:,}")
            if stored_n != live_n:
                issue(f"Extended {vt} cache ({stored_n:,}) != live count "
                      f"({live_n:,})")
    else:
        emit("\n  varwise_ext_transients.parquet not present (regenerate with "
             "apply_transient_fix.py)")

    pl_cache = ROOT / "data" / "raw" / "pl_sample.parquet"
    if pl_cache.exists():
        pl = pd.read_parquet(pl_cache)
        emit(f"\n  pl_sample.parquet: {len(pl):,} rows")
        live_n = int(tap.search(
            "SELECT COUNT(*) AS n FROM varwisepure WHERE vartype='rr' "
            "AND period1 > 0 AND plx > 0 AND plx/e_plx > 5"
        ).to_table()[0]["n"])
        stored_n = int(((pl.vartype == "rr") & (pl.tier == "pure")).sum())
        emit(f"    Pure rr subset: cached={stored_n:,}  live={live_n:,}")
        if stored_n != live_n:
            issue(f"PL-sample Pure rr cache ({stored_n:,}) != live ({live_n:,})")
    else:
        emit("\n  pl_sample.parquet not present")

    # ---------------- 6. schema / dtype sanity ----------------
    emit("\n" + "=" * 86)
    emit("6. SCHEMA AND DTYPE SANITY")
    emit("=" * 86)
    emit(f"\n  {'column':<20}{'dtype':<12}{'sample value'}")
    for c in ["designation", "ra", "vartype", "confidence", "w1mag", "plx"]:
        emit(f"  {c:<20}{str(df[c].dtype):<12}{df[c].iloc[0]}")
    # float64 precision check on ra/dec (parquet could silently downcast)
    if df.ra.dtype != np.float64:
        issue(f"ra stored as {df.ra.dtype}, expected float64 - precision risk")
    if df.dec.dtype != np.float64:
        issue(f"dec stored as {df.dec.dtype}, expected float64 - precision risk")

    # ---------------- summary ----------------
    emit("\n" + "=" * 86)
    emit("SUMMARY")
    emit("=" * 86)
    if not ISSUES:
        emit("\n  No integrity issues found. The stored data matches the live")
        emit("  IRSA source exactly on every check performed: fresh row counts,")
        emit("  60-object value-for-value spot check, all 11 RA-slice seams,")
        emit("  string encoding, and derived-cache consistency.")
    else:
        emit(f"\n  {len(ISSUES)} issue(s) found:")
        for m in ISSUES:
            emit(f"    - {m}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 1 if ISSUES else 0


if __name__ == "__main__":
    sys.exit(main())
