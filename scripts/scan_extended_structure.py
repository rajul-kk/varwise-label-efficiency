"""Structural scan of the Extended Catalog, via TAP aggregates (1.9M rows is
too large to profitably download in full for a duplicate/range check).

Closes a gap: scripts/full_catalog_scan.py only ever covered the Pure tier.
"""
import sys

import pyvo

TAP = "https://irsa.ipac.caltech.edu/TAP"
tap = pyvo.dal.TAPService(TAP)
OUT = "d:/Work/NEOWISE-analysis/results/extended_structure_scan.txt"
lines = []


def emit(s=""):
    print(s)
    lines.append(s)


def n(sql):
    return int(tap.search(sql).to_table()[0]["n"])


def main():
    emit("=" * 84)
    emit("STRUCTURAL SCAN - VarWISE Extended Catalog (via TAP aggregates)")
    emit("=" * 84)

    total = n("SELECT COUNT(*) AS n FROM varwiseext")
    uniq_id = n("SELECT COUNT(DISTINCT cluster_id) AS n FROM varwiseext")
    uniq_des = n("SELECT COUNT(DISTINCT designation) AS n FROM varwiseext")
    emit(f"\n  total rows        : {total:,}")
    emit(f"  unique cluster_id : {uniq_id:,}  "
         f"({'OK' if uniq_id == total else f'{total-uniq_id:,} DUPLICATES'})")
    emit(f"  unique designation: {uniq_des:,}  "
         f"({'OK' if uniq_des == total else f'{total-uniq_des:,} DUPLICATES'})")

    emit("\n  coordinate range checks:")
    bad_ra = n("SELECT COUNT(*) AS n FROM varwiseext WHERE ra < 0 OR ra > 360")
    bad_dec = n("SELECT COUNT(*) AS n FROM varwiseext WHERE dec < -90 OR dec > 90")
    emit(f"    ra outside [0,360] : {bad_ra:,}")
    emit(f"    dec outside [-90,90]: {bad_dec:,}")

    emit("\n  amplitude / SNR / period range checks:")
    for cond, label in [
            ("w1_amp < 0", "negative w1_amp"),
            ("w2_amp < 0", "negative w2_amp"),
            ("variability_snr < 0", "negative variability_snr"),
            ("n_obs < 0", "negative n_obs"),
            ("period1 < 0", "negative period1"),
            ("n_obs < 20", "n_obs < 20 (marginal for periods)"),
            ("confidence < 0 OR confidence > 1", "confidence outside [0,1]")]:
        c = n(f"SELECT COUNT(*) AS n FROM varwiseext WHERE {cond}")
        emit(f"    {label:<38}{c:>10,}")

    emit("\n  blended_source / latent_artifact bounds (Pure caps both at 0.5):")
    r = tap.search("SELECT MIN(blended_source) AS lo, MAX(blended_source) AS hi, "
                   "MIN(latent_artifact) AS llo, MAX(latent_artifact) AS lhi "
                   "FROM varwiseext").to_table()[0]
    emit(f"    blended_source  [{r['lo']:.3f}, {r['hi']:.3f}]")
    emit(f"    latent_artifact [{r['llo']:.3f}, {r['lhi']:.3f}]")

    emit("\n  vartype completeness:")
    null_vt = n("SELECT COUNT(*) AS n FROM varwiseext WHERE vartype IS NULL")
    emit(f"    rows with null vartype: {null_vt:,}")

    emit("\n  cv/sn mechanism split (mirrors the Pure-tier finding):")
    for vt in ("cv", "sn"):
        tot = n(f"SELECT COUNT(*) AS n FROM varwiseext WHERE vartype='{vt}'")
        rule = n(f"SELECT COUNT(*) AS n FROM varwiseext WHERE vartype='{vt}' "
                 f"AND confidence IS NULL AND period1 IS NULL")
        clf = n(f"SELECT COUNT(*) AS n FROM varwiseext WHERE vartype='{vt}' "
                f"AND confidence IS NOT NULL AND period1 IS NOT NULL")
        mismatched = tot - rule - clf
        emit(f"    {vt:<4} total={tot:>8,}  rule={rule:>8,} ({100*rule/tot:.1f}%)  "
             f"classifier={clf:>7,} ({100*clf/tot:.1f}%)  mismatched={mismatched:,}")

    OUT_PATH = OUT
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
