"""Download the VarWISE Pure Catalog from IRSA TAP to local parquet.

The Pure Catalog (457,080 objects) is the highest-confidence VarWISE tier.
Downloaded via async TAP job, in RA slices to stay under per-query limits.

Run: python scripts/download_varwise.py
"""
import sys
import time
from pathlib import Path

import pandas as pd
import pyvo

TAP_URL = "https://irsa.ipac.caltech.edu/TAP"
TABLE = "varwisepure"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "varwise_pure.parquet"

COLUMNS = [
    "cluster_id", "designation", "ra", "dec",
    "vartype", "confidence", "variability_snr",
    "period1", "period2", "period_significance", "suspect_period",
    "w1_amp", "w2_amp", "n_obs",
    "w1mag", "w1emag", "w2mag", "w2emag",
    "w3mag", "w3emag", "w4mag", "w4emag",
    "jmag", "jemag", "hmag", "hemag", "kmag", "kemag",
    "gmag", "bpmag", "rpmag", "plx", "e_plx",
    "simbad_type", "known_extragalactic", "blended_source", "latent_artifact",
]

N_SLICES = 12  # RA slices of 30 deg each


def fetch_slice(tap, ra_lo, ra_hi):
    cols = ", ".join(COLUMNS)
    where = f"ra >= {ra_lo} AND ra < {ra_hi}"
    sql = f"SELECT {cols} FROM {TABLE} WHERE {where}"
    job = tap.submit_job(sql)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=1800)
    if job.phase != "COMPLETED":
        raise RuntimeError(f"job phase={job.phase} for RA [{ra_lo},{ra_hi})")
    df = job.fetch_result().to_table().to_pandas()
    job.delete()
    return df


def main():
    if OUT.exists():
        df = pd.read_parquet(OUT)
        print(f"Already downloaded: {OUT} ({len(df):,} rows)")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tap = pyvo.dal.TAPService(TAP_URL)

    edges = [i * (360.0 / N_SLICES) for i in range(N_SLICES + 1)]
    parts = []
    for i in range(N_SLICES):
        lo, hi = edges[i], edges[i + 1]
        t0 = time.time()
        for attempt in (1, 2, 3):
            try:
                d = fetch_slice(tap, lo, hi)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  slice {i} attempt {attempt} failed: {type(e).__name__}: {e}")
                if attempt == 3:
                    raise
                time.sleep(10 * attempt)
        parts.append(d)
        print(f"slice {i+1}/{N_SLICES} RA [{lo:.0f},{hi:.0f}): "
              f"{len(d):,} rows in {time.time()-t0:.0f}s")

    df = pd.concat(parts, ignore_index=True)
    # decode any bytes columns from VOTable
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)

    df.to_parquet(OUT, index=False)
    print(f"\nWrote {len(df):,} rows x {len(df.columns)} cols -> {OUT}")
    print(f"Expected 457,080 -> {'OK' if len(df) == 457080 else 'MISMATCH'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
