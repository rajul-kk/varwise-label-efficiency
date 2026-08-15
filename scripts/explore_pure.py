"""Characterize the VarWISE Pure Catalog before designing the AL study.

Answers: class balance, confidence distribution, how much independent
(SIMBAD) ground truth exists, and feature completeness.
"""
import pyvo

TAP_URL = "https://irsa.ipac.caltech.edu/TAP"
tap = pyvo.dal.TAPService(TAP_URL)


def q(sql, label):
    print(f"\n=== {label} ===")
    try:
        return tap.search(sql).to_table()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR {type(e).__name__}: {e}")
        return None


t = q("SELECT COUNT(*) AS n FROM varwisepure", "total rows (Pure)")
if t is not None:
    print(f"n = {t['n'][0]}")

t = q(
    "SELECT vartype, COUNT(*) AS n FROM varwisepure GROUP BY vartype ORDER BY n DESC",
    "class balance (vartype)",
)
if t is not None:
    total = sum(int(r["n"]) for r in t)
    for r in t:
        print(f"  {str(r['vartype']):<12} {int(r['n']):>8}  {100*int(r['n'])/total:6.2f}%")

t = q(
    "SELECT COUNT(*) AS n_simbad FROM varwisepure WHERE simbad_type IS NOT NULL",
    "rows with SIMBAD type (independent labels)",
)
if t is not None:
    print(f"  n_simbad = {int(t['n_simbad'][0])}")

t = q(
    "SELECT simbad_type, COUNT(*) AS n FROM varwisepure "
    "WHERE simbad_type IS NOT NULL GROUP BY simbad_type ORDER BY n DESC",
    "SIMBAD type distribution (top)",
)
if t is not None:
    for r in list(t)[:40]:
        print(f"  {str(r['simbad_type']):<16} {int(r['n']):>8}")

# Feature completeness: how many rows have each key column non-null
cols = [
    "period1", "period2", "period_significance", "w1_amp", "w2_amp",
    "variability_snr", "n_obs", "w1mag", "w2mag", "w3mag", "w4mag",
    "jmag", "hmag", "kmag", "gmag", "bpmag", "rpmag", "plx",
]
print("\n=== feature completeness (non-null counts) ===")
for c in cols:
    t = tap.search(f"SELECT COUNT(*) AS n FROM varwisepure WHERE {c} IS NOT NULL").to_table()
    print(f"  {c:<22} {int(t['n'][0]):>8}")

t = q(
    "SELECT MIN(confidence) AS lo, MAX(confidence) AS hi, AVG(confidence) AS mean "
    "FROM varwisepure",
    "confidence range",
)
if t is not None:
    print(f"  min={t['lo'][0]:.4f} max={t['hi'][0]:.4f} mean={t['mean'][0]:.4f}")
