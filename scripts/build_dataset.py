"""Build model-ready feature matrices and labels from the VarWISE Pure Catalog.

Two label tracks:

  Track A ("vartype")  - VarWISE's own XGBoost prediction as the target.
      Reproduces the shape of their published task (9 classes) and is the
      closest available comparison point to their reported macro-F1 0.95.
      Caveat: the target is model output, so this measures label efficiency
      for *distilling* the VarWISE classifier, not for the science task.

  Track B ("simbad")   - independent SIMBAD literature types as the target.
      Real ground truth, curated externally. This is the scientifically
      meaningful track and the one that mirrors the real cost structure
      active learning exists to address (labels cost telescope/literature
      time). Taxonomy differs slightly: SIMBAD's "EclBin" does not separate
      Algol (ea) from W UMa (ew), so those merge into a single `ecl` class.

Features deliberately EXCLUDE vartype, confidence, and simbad_type - all
three are targets or near-targets and would leak.

Run: python scripts/build_dataset.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "varwise_pure.parquet"
OUTDIR = ROOT / "data"

# SIMBAD type -> VarWISE-style class. Types not listed are dropped as
# ambiguous (e.g. C* carbon stars straddle LPV and non-variable;
# "Variable*", "PulsV*", "SB*" are too generic to assign).
SIMBAD_MAP = {
    # Cepheids
    "ClassicalCep": "cep", "Type2Cep": "cep", "Cepheid": "cep",
    # RR Lyrae
    "RRLyrae": "rr", "RRLyrae_Candidate": "rr",
    # Long-period variables
    "LongPeriodV*": "lpv", "LongPeriodV*_Candidate": "lpv",
    "Mira": "lpv", "RVTauV*": "lpv",
    # Eclipsing binaries (ea + ew merged; SIMBAD does not separate them)
    "EclBin": "ecl", "EclBin_Candidate": "ecl",
    # Cataclysmic variables
    "CataclyV*": "cv", "Nova": "cv",
    # Young stellar objects
    "YSO": "yso", "YSO_Candidate": "yso", "TTauri*": "yso",
    "TTauri*_Candidate": "yso", "OrionV*": "yso", "Ae*": "yso",
    # Active galactic nuclei
    "QSO": "agn", "Seyfert1": "agn", "Seyfert2": "agn", "Seyfert": "agn",
    "BLLac": "agn", "Blazar": "agn", "Blazar_Candidate": "agn",
    "AGN": "agn", "AGN_Candidate": "agn",
    # Supernovae
    "Supernova": "sn", "Supernova_Candidate": "sn",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate VarWISE's feature philosophy using published catalog columns.

    VarWISE's own 31 features are derived from raw NEOWISE light curves
    (Fourier coefficients, Stetson indices, chi^2 statistics) which live in
    the separate Associations table, not in the published catalog. What is
    reproducible here: colors, amplitudes, periodicity, and variability
    significance -- the same six categories minus the light-curve morphology
    and flux-statistic blocks.
    """
    f = pd.DataFrame(index=df.index)

    # --- Colors (VarWISE "colors/physical" block) ---
    f["w1_w2"] = df["w1mag"] - df["w2mag"]
    f["w2_w3"] = df["w2mag"] - df["w3mag"]
    f["w3_w4"] = df["w3mag"] - df["w4mag"]
    f["j_h"] = df["jmag"] - df["hmag"]
    f["h_k"] = df["hmag"] - df["kmag"]
    f["k_w1"] = df["kmag"] - df["w1mag"]
    f["bp_rp"] = df["bpmag"] - df["rpmag"]
    f["g_rp"] = df["gmag"] - df["rpmag"]
    f["g_w1"] = df["gmag"] - df["w1mag"]

    # --- Apparent brightness ---
    f["w1mag"] = df["w1mag"]
    f["w2mag"] = df["w2mag"]

    # --- Variability amplitude ---
    f["w1_amp"] = df["w1_amp"]
    f["w2_amp"] = df["w2_amp"]
    with np.errstate(divide="ignore", invalid="ignore"):
        f["amp_ratio"] = df["w2_amp"] / df["w1_amp"].replace(0, np.nan)
    f["variability_snr"] = df["variability_snr"]
    f["log_var_snr"] = np.log10(df["variability_snr"].clip(lower=1e-3))

    # --- Periodicity (VarWISE "periodicity" block) ---
    f["period1"] = df["period1"]
    f["period2"] = df["period2"]
    f["log_period1"] = np.log10(df["period1"].where(df["period1"] > 0))
    f["log_period2"] = np.log10(df["period2"].where(df["period2"] > 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        f["period_ratio"] = df["period2"] / df["period1"].replace(0, np.nan)
    f["period_significance"] = df["period_significance"]
    f["suspect_period"] = df["suspect_period"]

    # --- Sampling ---
    f["n_obs"] = df["n_obs"]

    # --- Distance-dependent physical quantity ---
    # absolute W1 magnitude from Gaia parallax (mas); only where plx is a
    # meaningful positive detection
    plx = df["plx"].where(df["plx"] > 0)
    snr_plx = df["plx"] / df["e_plx"].replace(0, np.nan)
    good = plx.notna() & (snr_plx > 3)
    f["abs_w1"] = np.where(good, df["w1mag"] + 5 * np.log10(plx / 100.0), np.nan)
    f["log_plx"] = np.log10(plx.where(good))

    # --- Data-quality flags (VarWISE publishes these; not targets) ---
    f["known_extragalactic"] = df["known_extragalactic"]
    f["blended_source"] = df["blended_source"]
    f["latent_artifact"] = df["latent_artifact"]

    return f.replace([np.inf, -np.inf], np.nan)


def main():
    if not RAW.exists():
        print(f"Missing {RAW}. Run scripts/download_varwise.py first.")
        return 1

    df = pd.read_parquet(RAW)
    print(f"Loaded {len(df):,} rows")

    # normalize string columns
    for c in ("vartype", "simbad_type"):
        df[c] = df[c].astype("string").str.strip()

    X = build_features(df)
    print(f"Built {X.shape[1]} features")
    print("\nFeature null fraction:")
    for c in X.columns:
        print(f"  {c:<22} {X[c].isna().mean():6.2%}")

    # ---------------- Track A: vartype ----------------
    a_mask = df["vartype"].notna() & (df["vartype"] != "unclear")
    Xa, ya = X[a_mask], df.loc[a_mask, "vartype"]
    print(f"\nTrack A (vartype): {len(ya):,} rows, {ya.nunique()} classes")
    print(ya.value_counts().to_string())
    Xa.assign(_label=ya.values).to_parquet(OUTDIR / "track_a_vartype.parquet", index=False)

    # ---------------- Track B: SIMBAD ----------------
    mapped = df["simbad_type"].map(SIMBAD_MAP)
    b_mask = mapped.notna()
    Xb, yb = X[b_mask], mapped[b_mask]
    print(f"\nTrack B (simbad): {len(yb):,} rows, {yb.nunique()} classes")
    print(yb.value_counts().to_string())
    Xb.assign(_label=yb.values).to_parquet(OUTDIR / "track_b_simbad.parquet", index=False)

    # How much SIMBAD coverage was dropped as unmappable?
    unmapped = df.loc[df["simbad_type"].notna() & mapped.isna(), "simbad_type"]
    print(f"\nDropped {len(unmapped):,} SIMBAD-typed rows as ambiguous:")
    print(unmapped.value_counts().head(15).to_string())

    # Agreement between VarWISE prediction and SIMBAD truth, for context
    both = df[a_mask & b_mask].copy()
    both["_simbad"] = mapped[a_mask & b_mask]
    coll = both["vartype"].replace({"ea": "ecl", "ew": "ecl"})
    agree = (coll == both["_simbad"]).mean()
    print(f"\nVarWISE-vs-SIMBAD agreement (ea/ew collapsed): {agree:.3%} on {len(both):,} rows")
    print("\nPer-SIMBAD-class agreement:")
    for cls in sorted(both["_simbad"].unique()):
        m = both["_simbad"] == cls
        print(f"  {cls:<6} n={m.sum():>7,}  agree={(coll[m] == cls).mean():7.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
