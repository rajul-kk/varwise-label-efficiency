"""Does the Extended Catalog's rr anomaly survive the authors' own quality cuts?

The VarWISE paper recommends `confidence >= 0.9`, and defines the Pure tier by
blended_source=0, latent_artifact=0, variability_snr>5, and
(confidence>0.8 OR variability_snr>10).

If the rr period anomaly disappears once those documented cuts are applied,
then it is not a catalog defect -- it is what happens when a user ignores the
published guidance, which is a much weaker claim. This script settles that
before anything is built on top of it.
"""
import pyvo

TAP = "https://irsa.ipac.caltech.edu/TAP"
tap = pyvo.dal.TAPService(TAP)

PHYS_LO, PHYS_HI = 0.2, 1.0  # physical RR Lyrae period range


def n(sql):
    return int(tap.search(sql).to_table()[0]["n"])


def profile(table, vartype, extra, label):
    base = f"vartype='{vartype}'" + (f" AND {extra}" if extra else "")
    tot = n(f"SELECT COUNT(*) AS n FROM {table} WHERE {base}")
    if tot == 0:
        print(f"  {label:<44} n=0")
        return
    withp = n(f"SELECT COUNT(*) AS n FROM {table} WHERE {base} AND period1 IS NOT NULL")
    phys = n(f"SELECT COUNT(*) AS n FROM {table} WHERE {base} "
             f"AND period1 >= {PHYS_LO} AND period1 < {PHYS_HI}")
    impossible = n(f"SELECT COUNT(*) AS n FROM {table} WHERE {base} AND period1 >= 2.0")
    print(f"  {label:<44} n={tot:>9,}  physical={100*phys/max(withp,1):>5.1f}%  "
          f"P>=2d={100*impossible/max(withp,1):>5.1f}%")


CUTS = [
    ("", "no cuts"),
    ("confidence >= 0.8", "confidence >= 0.8"),
    ("confidence >= 0.9", "confidence >= 0.9  (authors' recommendation)"),
    ("confidence >= 0.95", "confidence >= 0.95"),
    ("confidence >= 0.99", "confidence >= 0.99"),
    ("blended_source = 0 AND latent_artifact = 0 AND variability_snr > 5 "
     "AND (confidence > 0.8 OR variability_snr > 10)", "full Pure-tier criteria"),
    ("blended_source = 0 AND latent_artifact = 0 AND variability_snr > 5 "
     "AND confidence >= 0.9", "Pure-tier criteria + confidence >= 0.9"),
    ("suspect_period = 0 AND confidence >= 0.9", "confidence >= 0.9 AND not suspect_period"),
    ("period_significance > 20 AND confidence >= 0.9",
     "confidence >= 0.9 AND period_significance > 20"),
]

print("=" * 92)
print("EXTENDED CATALOG - `rr` class under successive quality cuts")
print(f"(physical RR Lyrae period range = {PHYS_LO}-{PHYS_HI} d; P >= 2 d is unphysical)")
print("=" * 92)
for extra, label in CUTS:
    profile("varwiseext", "rr", extra, label)

print("\n" + "=" * 92)
print("PURE CATALOG - `rr` for comparison (audited F1 0.948)")
print("=" * 92)
for extra, label in CUTS[:5]:
    profile("varwisepure", "rr", extra, label)

print("\n" + "=" * 92)
print("EXTENDED CATALOG - all classes at the authors' recommended cut")
print("=" * 92)
print("\n  Class counts before vs after confidence >= 0.9:")
print(f"  {'class':<9}{'no cuts':>12}{'conf>=0.9':>12}{'retained':>11}")
for vt in ("lpv", "rr", "ew", "agn", "ea", "cv", "yso", "cep", "sn", "unclear"):
    a = n(f"SELECT COUNT(*) AS n FROM varwiseext WHERE vartype='{vt}'")
    b = n(f"SELECT COUNT(*) AS n FROM varwiseext WHERE vartype='{vt}' "
          f"AND confidence >= 0.9")
    print(f"  {vt:<9}{a:>12,}{b:>12,}{100*b/max(a,1):>10.1f}%")

print("\n  Note: cv and sn carry NO confidence (rule-assigned), so a confidence")
print("  cut removes them entirely rather than filtering them.")

print("\n" + "=" * 92)
print("How many RR Lyrae does Extended claim after cuts, vs Gaia DR3's 270,905?")
print("=" * 92)
for extra, label in [("", "no cuts"),
                     ("confidence >= 0.9", "confidence >= 0.9"),
                     ("confidence >= 0.9 AND period1 >= 0.2 AND period1 < 1.0",
                      "confidence >= 0.9 AND physical period")]:
    c = n(f"SELECT COUNT(*) AS n FROM varwiseext WHERE vartype='rr'"
          + (f" AND {extra}" if extra else ""))
    print(f"  {label:<44} {c:>9,}   ({c/270905:.2f}x Gaia DR3)")
