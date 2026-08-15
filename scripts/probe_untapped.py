"""Survey what remains unexploited in the VarWISE / NEOWISE archive.

Checks: the Associations table (per-object NEOWISE detections = actual light
curves, which the published catalogs do not contain), the Extended Catalog's
composition versus the Pure Catalog, and how much independent label coverage
exists in the Extended tier.
"""
import sys

import pyvo

TAP = "https://irsa.ipac.caltech.edu/TAP"
tap = pyvo.dal.TAPService(TAP)


def q(sql, label, limit=40):
    print(f"\n=== {label} ===")
    try:
        t = tap.search(sql).to_table()
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR {type(e).__name__}: {e}")
        return None
    for r in list(t)[:limit]:
        print("  " + "  ".join(f"{c}={r[c]}" for c in t.colnames))
    return t


# ---- what VarWISE tables exist at all? ----
q("SELECT table_name FROM TAP_SCHEMA.tables WHERE table_name LIKE '%varwise%' "
  "OR table_name LIKE '%vw%assoc%' ORDER BY table_name",
  "VarWISE tables on IRSA TAP")

# ---- the Associations table: schema and size ----
for cand in ("varwiseassoc", "varwiseassociations", "varwise_assoc"):
    t = q(f"SELECT column_name, datatype FROM TAP_SCHEMA.columns "
          f"WHERE table_name = '{cand}'", f"columns of {cand}", limit=60)
    if t is not None and len(t):
        q(f"SELECT COUNT(*) AS n FROM {cand}", f"row count of {cand}")
        break

# ---- Extended Catalog composition ----
q("SELECT vartype, COUNT(*) AS n FROM varwiseext GROUP BY vartype ORDER BY n DESC",
  "Extended Catalog class balance")

q("SELECT COUNT(*) AS n_simbad FROM varwiseext WHERE simbad_type IS NOT NULL "
  "AND simbad_type != ''",
  "Extended Catalog rows with an independent SIMBAD type")

q("SELECT COUNT(*) AS n FROM varwiseext WHERE confidence IS NULL",
  "Extended Catalog rows with null confidence (rule-assigned cv/sn)")

# ---- periodicity coverage: how many have usable periods ----
q("SELECT COUNT(*) AS n FROM varwisepure WHERE period_significance > 10",
  "Pure Catalog: strongly significant periods")

q("SELECT suspect_period, COUNT(*) AS n FROM varwisepure GROUP BY suspect_period",
  "Pure Catalog: suspect_period flag distribution")

# ---- parallax availability, for absolute-magnitude work ----
q("SELECT COUNT(*) AS n FROM varwisepure WHERE plx > 0 AND plx/e_plx > 5",
  "Pure Catalog: parallax S/N > 5 (usable distances)")

# ---- how many LPVs have both period and parallax (period-luminosity work) ----
q("SELECT COUNT(*) AS n FROM varwisepure WHERE vartype = 'lpv' "
  "AND period1 > 0 AND plx > 0 AND plx/e_plx > 5",
  "LPVs with both a period and a good parallax")
