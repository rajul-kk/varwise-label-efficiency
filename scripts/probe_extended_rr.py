"""Spot-check the Extended Catalog's RR Lyrae class for plausibility.

The Extended Catalog assigns `rr` to 443,991 objects. For scale, Gaia DR3's
dedicated RR Lyrae catalogue contains ~271,000 across the whole sky, and RR
Lyrae are among the best-inventoried variable classes in astronomy. An
infrared survey claiming ~1.6x the total known population is a red flag
worth checking before building anything on the Extended tier.

Diagnostic: real RR Lyrae have periods tightly confined to ~0.2-1.0 d
(RRab ~0.4-0.9 d, RRc ~0.2-0.45 d). Contamination from period aliasing
piles up at 1 d, 0.5 d, and the survey cadence. Compares Extended `rr`
against Pure `rr`, which the audit found to be clean (F1 0.948).
"""
import numpy as np
import pyvo

TAP = "https://irsa.ipac.caltech.edu/TAP"
tap = pyvo.dal.TAPService(TAP)


def stats(table, where, label):
    sql = (f"SELECT COUNT(*) AS n, AVG(confidence) AS conf, "
           f"AVG(period1) AS p_mean, AVG(w1_amp) AS amp, AVG(w1mag) AS w1, "
           f"AVG(variability_snr) AS vsnr, AVG(period_significance) AS psig "
           f"FROM {table} WHERE {where}")
    r = tap.search(sql).to_table()[0]
    print(f"  {label:<28} n={int(r['n']):>8,}  conf={r['conf']:.3f}  "
          f"P={r['p_mean']:.3f}d  amp={r['amp']:.3f}  W1={r['w1']:.2f}  "
          f"varSNR={r['vsnr']:.1f}  Psig={r['psig']:.1f}")


print("=== Class-level comparison: Pure vs Extended `rr` ===")
stats("varwisepure", "vartype='rr'", "Pure rr (audited F1 0.948)")
stats("varwiseext", "vartype='rr'", "Extended rr")
print()
stats("varwisepure", "vartype='lpv'", "Pure lpv (reference)")
stats("varwiseext", "vartype='lpv'", "Extended lpv")

print("\n=== Period distribution of `rr` (real RR Lyrae: 0.2-1.0 d) ===")
bins = [(0.0, 0.2), (0.2, 0.45), (0.45, 1.0), (1.0, 1.05),
        (1.05, 2.0), (2.0, 10.0), (10.0, 100.0), (100.0, 1e6)]
for table in ("varwisepure", "varwiseext"):
    tot = int(tap.search(f"SELECT COUNT(*) AS n FROM {table} WHERE vartype='rr' "
                         f"AND period1 IS NOT NULL").to_table()[0]["n"])
    print(f"\n  {table} (n with period = {tot:,})")
    in_range = 0
    for lo, hi in bins:
        n = int(tap.search(f"SELECT COUNT(*) AS n FROM {table} WHERE vartype='rr' "
                           f"AND period1 >= {lo} AND period1 < {hi}"
                           ).to_table()[0]["n"])
        flag = ""
        if 0.2 <= lo and hi <= 1.0:
            in_range += n
            flag = "  <- physical RR Lyrae range"
        if lo == 1.0:
            flag = "  <- 1-day alias"
        print(f"    {lo:>7.2f} - {hi:<8.2f} {n:>9,}  {100*n/max(tot,1):>6.2f}%{flag}")
    print(f"    => inside the physical RR Lyrae period range: "
          f"{100*in_range/max(tot,1):.1f}%")

print("\n=== Confidence distribution of Extended `rr` ===")
for lo in (0.0, 0.4, 0.6, 0.8, 0.9, 0.95):
    n = int(tap.search(f"SELECT COUNT(*) AS n FROM varwiseext WHERE vartype='rr' "
                       f"AND confidence >= {lo}").to_table()[0]["n"])
    print(f"  confidence >= {lo:.2f}: {n:>9,}")

print("\n=== suspect_period rate ===")
for table in ("varwisepure", "varwiseext"):
    for vt in ("rr", "lpv", "ew"):
        tot = int(tap.search(f"SELECT COUNT(*) AS n FROM {table} WHERE vartype='{vt}'"
                             ).to_table()[0]["n"])
        sus = int(tap.search(f"SELECT COUNT(*) AS n FROM {table} WHERE vartype='{vt}' "
                             f"AND suspect_period=1").to_table()[0]["n"])
        print(f"  {table:<14} {vt:<4} suspect_period=1: {sus:>8,} / {tot:>8,} "
              f"({100*sus/max(tot,1):>5.1f}%)")
