"""Check feasibility of the two Tier 2 directions before committing time to
either: is unTimely queryable via IRSA TAP, and how large/queryable is the
raw NEOWISE single-exposure table needed to rebuild VarWISE's light-curve
features.
"""
import sys

import pyvo

TAP = "https://irsa.ipac.caltech.edu/TAP"
tap = pyvo.dal.TAPService(TAP)


def q(sql, label, limit=60):
    print(f"\n=== {label} ===")
    try:
        t = tap.search(sql).to_table()
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR {type(e).__name__}: {e}")
        return None
    for r in list(t)[:limit]:
        print("  " + "  ".join(f"{c}={r[c]}" for c in t.colnames))
    return t


print("=" * 84)
print("R4 FEASIBILITY: is unTimely on IRSA TAP?")
print("=" * 84)
q("SELECT table_name FROM TAP_SCHEMA.tables WHERE table_name LIKE '%untimely%' "
  "OR table_name LIKE '%unwise%' OR table_name LIKE '%timely%' ORDER BY table_name",
  "unTimely-related tables")

print("\n" + "=" * 84)
print("R5 FEASIBILITY: raw NEOWISE single-exposure table scale")
print("=" * 84)
q("SELECT table_name FROM TAP_SCHEMA.tables WHERE table_name LIKE '%neowiser%' "
  "OR table_name LIKE '%p1bs%' ORDER BY table_name", "raw NEOWISE tables")
