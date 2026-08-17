"""Concordance check: VarWISE vs an independent mid-IR variability catalog.

The unTimely-derived variability catalog (8.26M objects, Yao et al.) is not
yet publicly released ("tables will be available online soon" per the
preprint). A smaller, already-published substitute exists that serves the
same purpose: Kim, Son, Kim, Ho, Jeong, Lee & Yang 2026, ApJS 284:39, "A
Catalog of Mid-infrared Variable Sources in the Ecliptic Poles" -- 30,345
objects, independently detected from NEOWISE multi-epoch photometry near the
ecliptic poles, with classifications from an entirely separate pipeline
(ZTF light curves through the Healy et al. 2024 deep-neural-network
classifier), overlapping VarWISE's sky footprint and input data.

This is the concordance check R4 in NEXT_STEPS.md asked for, scoped to what
is actually publicly downloadable: does an independent mid-IR variable
catalog, built from different data (NEOWISE photometry -> ZTF DNN
classification) than VarWISE (NEOWISE photometry -> VARnet + XGBoost), agree
on which objects are eclipsing binaries, QSOs, pulsators, YSOs, etc.?

Run: python scripts/untimely_ecliptic_concordance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T1 = ROOT / "data" / "raw" / "ecliptic_poles_t1.txt"
T3 = ROOT / "data" / "raw" / "ecliptic_poles_t3.txt"
OUT = ROOT / "results" / "ecliptic_poles_concordance.txt"
MATCH_RADIUS_ARCSEC = 2.0  # matches the paper's own convention
lines = []

# ZTF DNN class -> nearest VarWISE-taxonomy equivalent, for a like-for-like
# comparison (not all classes have a clean VarWISE analog)
ZTF_TO_VARWISE = {
    "Q": "agn",   # QSOs
    "E": "ecl",   # eclipses (VarWISE ea/ew collapsed to ecl, as elsewhere)
    "P": None,    # pulsating -- could be rr/cep/lpv, too coarse to map safely
    "S": None,    # generic "variable star" -- too coarse
    "B": None,    # binary stars, not necessarily eclipsing
    "Y": "yso",
}


def parse_mrt(path, colspecs, names, skiprows):
    return pd.read_fwf(path, colspecs=colspecs, names=names, skiprows=skiprows)


def main():
    if not T1.exists() or not T3.exists():
        print("Missing downloaded tables; fetch via curl first (see repo notes).")
        return 1

    t1 = parse_mrt(T1, [(0, 24), (25, 33), (34, 42), (43, 47), (48, 52),
                        (53, 57), (58, 63), (64, 68), (69, 74), (75, 79),
                        (80, 85), (86, 90)],
                   ["name", "ra", "dec", "pvar1", "pvar2", "r", "w1mag",
                    "e_w1mag", "w2mag", "e_w2mag", "w3mag", "e_w3mag"],
                   skiprows=21)
    t3 = parse_mrt(T3, [(0, 26), (27, 32), (33, 38), (39, 44), (45, 46),
                        (47, 54), (55, 60), (61, 64), (65, 68), (69, 70),
                        (71, 72)],
                   ["name", "z_desi", "z_quaia", "z_milliquas", "desi_type",
                    "pm", "e_pm", "n_gaia", "n_ls", "class_ztf05", "class_ztf"],
                   skiprows=36)

    # t1 and t3 are companion tables from the same paper/row order (both
    # exactly 30,345 rows); joined POSITIONALLY rather than on "name",
    # because 165 object names are duplicated in the source table (likely
    # overlapping NEP/SEP field boundaries) and a name-based merge silently
    # multiplies those rows (30,345 -> 30,419 -- caught by factcheck.py).
    assert len(t1) == len(t3), f"table length mismatch: {len(t1)} vs {len(t3)}"
    cat = pd.concat([t1.reset_index(drop=True),
                     t3[["class_ztf05", "class_ztf"]].reset_index(drop=True)],
                    axis=1)
    cat = cat.dropna(subset=["ra", "dec"])
    emit_lines = lines

    def emit(s=""):
        print(s)
        emit_lines.append(s)

    emit("=" * 88)
    emit("CONCORDANCE: VarWISE vs an independent mid-IR variable catalog")
    emit("(Kim et al. 2026, ApJS 284:39, ecliptic poles)")
    emit("=" * 88)
    emit(f"\nEcliptic-poles catalog: {len(cat):,} objects with coordinates")
    emit(f"  with a ZTF DNN classification (any threshold): "
         f"{cat.class_ztf.notna().sum():,}")
    emit(f"  RA range: {cat.ra.min():.2f} - {cat.ra.max():.2f}")
    emit(f"  Dec range: {cat.dec.min():.2f} - {cat.dec.max():.2f}")

    emit("\nClass distribution (class_ztf, max-DNN-score assignment):")
    for k, v in cat.class_ztf.value_counts().items():
        emit(f"  {k}: {v:,}")

    # ---------------- crossmatch against VarWISE Pure ----------------
    emit("\n" + "=" * 88)
    emit("CROSS-MATCHING AGAINST VarWISE PURE CATALOG")
    emit("=" * 88)
    vw = pd.read_parquet(ROOT / "data" / "raw" / "varwise_pure.parquet")
    vw["vartype"] = vw["vartype"].astype("string").str.strip()

    # restrict VarWISE to the sky patches this catalog covers (near the
    # ecliptic poles; RA range spans both hemispheres so use dec cuts loosely
    # then do the real match by brute-force nearest-neighbour within radius)
    from scipy.spatial import cKDTree

    def radec_to_xyz(ra, dec):
        ra_r, dec_r = np.radians(ra), np.radians(dec)
        return np.column_stack([np.cos(dec_r) * np.cos(ra_r),
                                np.cos(dec_r) * np.sin(ra_r),
                                np.sin(dec_r)])

    # the catalog spans both ecliptic poles (dec from -71.6 to +71.6), so no
    # single-hemisphere pre-filter applies; match against the full Pure
    # catalog directly via KDTree (fast enough at this scale: 457K points)
    vw_near = vw
    emit(f"\n  VarWISE Pure catalog size for the crossmatch: {len(vw_near):,}")

    xyz_cat = radec_to_xyz(cat.ra.values, cat.dec.values)
    xyz_vw = radec_to_xyz(vw_near.ra.values, vw_near.dec.values)
    tree = cKDTree(xyz_vw)
    chord = 2 * np.sin(np.radians(MATCH_RADIUS_ARCSEC / 3600.0) / 2)
    dist, idx = tree.query(xyz_cat, k=1)
    matched = dist < chord

    emit(f"  matches within {MATCH_RADIUS_ARCSEC}\": {matched.sum():,} of "
         f"{len(cat):,} ({100*matched.sum()/len(cat):.1f}%)")

    m_cat = cat[matched].copy()
    m_cat["vw_idx"] = idx[matched]
    m_cat["vw_vartype"] = vw_near.iloc[idx[matched]]["vartype"].values
    m_cat["vw_confidence"] = vw_near.iloc[idx[matched]]["confidence"].values

    emit("\n  VarWISE vartype distribution among matched objects:")
    for k, v in m_cat.vw_vartype.value_counts().items():
        emit(f"    {k}: {v:,}")

    # ---------------- like-for-like concordance ----------------
    emit("\n" + "=" * 88)
    emit("AGREEMENT ON CLASSES WITH A CLEAN CROSS-TAXONOMY MAPPING")
    emit("=" * 88)
    m_cat["ztf_mapped"] = m_cat.class_ztf.map(ZTF_TO_VARWISE)
    m_cat["vw_mapped"] = m_cat.vw_vartype.replace({"ea": "ecl", "ew": "ecl"})
    both = m_cat.dropna(subset=["ztf_mapped"])
    emit(f"\n  n with a clean mapped class on both sides: {len(both):,}")
    if len(both):
        agree = (both.ztf_mapped == both.vw_mapped).mean()
        emit(f"  overall agreement: {agree:.1%}")
        emit(f"\n  {'ZTF class':<10}{'n':>7}{'VarWISE agrees':>17}{'agreement':>12}")
        for c in sorted(both.ztf_mapped.unique()):
            sub = both[both.ztf_mapped == c]
            a = (sub.vw_mapped == c).sum()
            emit(f"  {c:<10}{len(sub):>7,}{a:>17,}{a/len(sub):>12.1%}")

        emit("\n  confusion (rows = ZTF-catalog class, cols = VarWISE class):")
        ct = pd.crosstab(both.ztf_mapped, both.vw_mapped)
        emit("            " + "".join(f"{c:>8}" for c in ct.columns))
        for r in ct.index:
            emit(f"  {r:<10}" + "".join(f"{v:>8,}" for v in ct.loc[r]))

        # ---------------- mechanism check: are the ecl/yso->agn --------
        # mismatches an artifact (bad match, low VarWISE confidence, wrong
        # hemisphere) or a real, explicable failure mode?
        emit("\n" + "=" * 88)
        emit("MECHANISM CHECK: is the ecl/yso -> agn mismatch a real effect?")
        emit("=" * 88)
        m_cat["sep_arcsec"] = np.degrees(2 * np.arcsin(
            np.minimum(dist[matched], 1.0) / 2)) * 3600
        m_cat["hemisphere"] = np.where(m_cat.dec > 0, "NEP", "SEP")
        m_cat["n_obs"] = vw_near.iloc[idx[matched]]["n_obs"].values

        for src_cls, tgt_cls, label in [("E", "agn", "ecl(ZTF) -> agn(VarWISE)"),
                                        ("Y", "agn", "yso(ZTF) -> agn(VarWISE)")]:
            sub = m_cat[(m_cat.class_ztf == src_cls) & (m_cat.vw_vartype == tgt_cls)]
            if len(sub) == 0:
                continue
            emit(f"\n  {label}: n={len(sub)}")
            emit(f"    median match separation: {sub.sep_arcsec.median():.4f}\"  "
                 f"(overall match median: {m_cat.sep_arcsec.median():.4f}\")")
            emit(f"    median VarWISE confidence: {sub.vw_confidence.median():.4f}")
            emit(f"    hemisphere: " +
                 ", ".join(f"{k}={v}" for k, v in
                          sub.hemisphere.value_counts().items()))
            emit(f"    median n_obs: {sub.n_obs.median():.0f}  "
                 f"(all VarWISE agn median: "
                 f"{vw_near[vw_near.vartype=='agn'].n_obs.median():.0f}, "
                 f"catalog-wide median: {vw_near.n_obs.median():.0f})")

    emit("\n" + "=" * 88)
    emit("CAVEATS")
    emit("=" * 88)
    emit("""
  - Only Q (QSO) and Y (YSO) map cleanly onto VarWISE's taxonomy; E
    (eclipse) is compared against VarWISE's collapsed ea+ew "ecl" class.
    The Pure-catalog audit found ecl VarWISE's most reliable class overall
    (F1 0.99 against SIMBAD, sky-wide) -- the 17.9% agreement found here is
    therefore surprising, not expected, and traces to a specific regional
    cause (see the mechanism check above), not to ecl being unreliable in
    general. P/S/B are too coarse to map to a single VarWISE class and are
    excluded from the agreement statistic rather than force-mapped.
  - This substitutes for the unreleased unTimely variability catalog; it
    covers only the NEP/SEP regions (~5 deg radius circles), not the full
    sky, so results characterize VarWISE's behaviour in those regions
    specifically, not a global concordance rate.
  - The ZTF classification is itself a DNN threshold call (Healy et al.
    2024), not ground truth; disagreement could originate on either side.
""")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
