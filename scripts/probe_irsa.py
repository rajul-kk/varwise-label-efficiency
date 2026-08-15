"""Probe IRSA for VarWISE catalog availability and schema.

Run: python scripts/probe_irsa.py
"""
import sys

import pyvo

TAP_URL = "https://irsa.ipac.caltech.edu/TAP"


def main():
    tap = pyvo.dal.TAPService(TAP_URL)

    print("=== searching TAP_SCHEMA for candidate tables ===")
    patterns = ["%varwise%", "%var_wise%", "%neowise_var%", "%irsa656%"]
    found = []
    for pat in patterns:
        q = f"SELECT table_name, description FROM TAP_SCHEMA.tables WHERE table_name LIKE '{pat}'"
        try:
            rows = tap.search(q).to_table()
        except Exception as e:  # noqa: BLE001
            print(f"  {pat}: ERROR {type(e).__name__}: {e}")
            continue
        print(f"  {pat}: {len(rows)} hit(s)")
        for r in rows:
            print(f"    - {r['table_name']}")
            found.append(str(r["table_name"]))

    if not found:
        print("\nNo VarWISE table exposed via TAP. Listing all wise/neowise tables:")
        q = (
            "SELECT table_name FROM TAP_SCHEMA.tables "
            "WHERE table_name LIKE '%wise%' ORDER BY table_name"
        )
        try:
            for r in tap.search(q).to_table():
                print(f"    - {r['table_name']}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {type(e).__name__}: {e}")
        return 1

    for t in found:
        print(f"\n=== columns of {t} ===")
        q = (
            "SELECT column_name, datatype, description FROM TAP_SCHEMA.columns "
            f"WHERE table_name = '{t}'"
        )
        try:
            for r in tap.search(q).to_table():
                print(f"    {r['column_name']:<28} {r['datatype']}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
