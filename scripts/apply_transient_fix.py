"""Produce corrected labels for every rule-assigned VarWISE transient.

scripts/fix_transient_rule.py showed a classifier recovers 96.6% of the
population VarWISE's CV/SN rule mislabels, but only validated it on the
14,955 objects that happen to carry a SIMBAD type. Those are systematically
bright. This script:

  1. Tests whether the result survives to faint magnitudes, by measuring
     accuracy in W1 magnitude bins across the labelled set. This is the
     decisive check: every finding in this repo rests on the bright,
     SIMBAD-covered subset, and if accuracy collapses at the faint end then
     the corrections cannot be extrapolated.
  2. Trains on all labelled rule-assigned objects and applies the model to
     every rule-assigned object in BOTH catalog tiers.
  3. Writes a value-added table keyed on cluster_id, flagging which rows had
     independent validation and which are extrapolated.

Run: python scripts/apply_transient_fix.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyvo
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_dataset import SIMBAD_MAP, build_features  # noqa: E402

RAW = ROOT / "data" / "raw" / "varwise_pure.parquet"
EXT_CACHE = ROOT / "data" / "raw" / "varwise_ext_transients.parquet"
OUT_TABLE = ROOT / "results" / "varwise_transient_corrections.csv"
OUT_REPORT = ROOT / "results" / "transient_corrections_report.txt"

TAP = "https://irsa.ipac.caltech.edu/TAP"
COLUMNS = [
    "cluster_id", "designation", "ra", "dec", "vartype", "confidence",
    "variability_snr", "period1", "period2", "period_significance",
    "suspect_period", "w1_amp", "w2_amp", "n_obs",
    "w1mag", "w1emag", "w2mag", "w2emag", "w3mag", "w3emag", "w4mag", "w4emag",
    "jmag", "jemag", "hmag", "hemag", "kmag", "kemag",
    "gmag", "bpmag", "rpmag", "plx", "e_plx",
    "simbad_type", "known_extragalactic", "blended_source", "latent_artifact",
]
MIN_CLASS = 30
lines = []


def emit(s=""):
    print(s)
    lines.append(s)


def fetch_extended_transients():
    """Rule-assigned objects (cv/sn) from the Extended tier."""
    if EXT_CACHE.exists():
        return pd.read_parquet(EXT_CACHE)
    tap = pyvo.dal.TAPService(TAP)
    cols = ", ".join(COLUMNS)
    parts = []
    for vt in ("cv", "sn"):
        job = tap.submit_job(
            f"SELECT {cols} FROM varwiseext WHERE vartype = '{vt}'")
        job.run()
        job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=3600)
        if job.phase != "COMPLETED":
            raise RuntimeError(f"Extended {vt} query phase={job.phase}")
        d = job.fetch_result().to_table().to_pandas()
        job.delete()
        print(f"  fetched Extended {vt}: {len(d):,}")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
    EXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(EXT_CACHE, index=False)
    return df


def make_model():
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          min_child_samples=10, class_weight="balanced",
                          random_state=0, n_jobs=-1, verbose=-1)


def main():
    pure = pd.read_parquet(RAW)
    pure["tier"] = "pure"
    print("Fetching Extended-tier rule-assigned transients...")
    ext = fetch_extended_transients()
    ext["tier"] = "extended"
    # Extended contains Pure; keep Extended rows not already in Pure
    ext = ext[~ext.cluster_id.isin(set(pure.cluster_id))].copy()

    for d in (pure, ext):
        d["vartype"] = d["vartype"].astype("string").str.strip()
        d["simbad_type"] = d["simbad_type"].astype("string").str.strip()
        d["truth"] = d["simbad_type"].map(SIMBAD_MAP)

    allrows = pd.concat(
        [pure[pure.vartype.isin(["cv", "sn"])], ext[ext.vartype.isin(["cv", "sn"])]],
        ignore_index=True)

    emit("=" * 84)
    emit("CORRECTED LABELS FOR VarWISE RULE-ASSIGNED TRANSIENTS")
    emit("=" * 84)
    emit(f"\nRule-assigned objects to correct:")
    for tier in ("pure", "extended"):
        s = allrows[allrows.tier == tier]
        emit(f"  {tier:<10} cv={int((s.vartype=='cv').sum()):>7,}  "
             f"sn={int((s.vartype=='sn').sum()):>7,}  total={len(s):>7,}")
    emit(f"  {'TOTAL':<10} {len(allrows):>28,}")

    lab = allrows[allrows.truth.notna()].copy()
    vc = lab.truth.value_counts()
    keep = set(vc[vc >= MIN_CLASS].index)
    lab = lab[lab.truth.isin(keep)].copy()
    classes = sorted(keep)
    emit(f"\nWith an independent SIMBAD label: {len(lab):,} "
         f"({len(lab)/len(allrows):.1%})   classes: {classes}")

    X_lab = build_features(lab).values.astype(np.float64)
    y_lab = lab.truth.values

    # ---------- 1. does it survive to faint magnitudes? ----------
    emit("\n" + "=" * 84)
    emit("GENERALIZATION CHECK - accuracy vs brightness")
    emit("=" * 84)
    oof = np.empty(len(y_lab), dtype=object)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(X_lab, y_lab):
        m = make_model()
        m.fit(X_lab[tr], y_lab[tr])
        oof[te] = m.predict(X_lab[te])

    w1 = lab.w1mag.values
    emit(f"\n  labelled objects span W1 = {np.nanmin(w1):.1f} to {np.nanmax(w1):.1f}")
    emit(f"\n  {'W1 range':<16}{'n':>8}{'accuracy':>11}{'macro F1':>11}"
         f"{'dominant truth':>17}")
    edges = [0, 8, 10, 11, 12, 13, 14, 30]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (w1 >= lo) & (w1 < hi)
        if m.sum() < 30:
            continue
        acc = accuracy_score(y_lab[m], oof[m])
        mf1 = f1_score(y_lab[m], oof[m], labels=classes, average="macro",
                       zero_division=0)
        dom = pd.Series(y_lab[m]).value_counts().index[0]
        emit(f"  {f'{lo}-{hi}':<16}{int(m.sum()):>8,}{acc:>11.4f}{mf1:>11.4f}"
             f"{dom:>17}")

    # unlabelled magnitude distribution, for extrapolation distance
    unlab = allrows[allrows.truth.isna()]
    emit(f"\n  labelled   median W1 = {np.nanmedian(lab.w1mag):.2f}")
    emit(f"  unlabelled median W1 = {np.nanmedian(unlab.w1mag):.2f}  "
         f"(n = {len(unlab):,})")
    frac_beyond = float((unlab.w1mag > np.nanpercentile(lab.w1mag, 95)).mean())
    emit(f"  unlabelled objects fainter than the labelled 95th percentile: "
         f"{frac_beyond:.1%}")

    # ---------- 2. train on all labels, apply everywhere ----------
    emit("\n" + "=" * 84)
    emit("APPLYING THE MODEL")
    emit("=" * 84)
    model = make_model()
    model.fit(X_lab, y_lab)

    X_all = build_features(allrows).values.astype(np.float64)
    proba = model.predict_proba(X_all)
    pred = model.classes_[proba.argmax(axis=1)]
    pmax = proba.max(axis=1)

    # Per-class cross-validated F1 governs how far each prediction can be
    # trusted. Accuracy is carried by lpv/agn/yso; the rarer classes in this
    # population are modelled from very few examples and must be marked down.
    per_f1 = dict(zip(classes, f1_score(y_lab, oof, labels=classes,
                                        average=None, zero_division=0)))
    n_train = pd.Series(y_lab).value_counts().to_dict()

    def reliability(row_class, prob, validated):
        if validated:
            return "validated"
        f1c = per_f1.get(row_class, 0.0)
        if f1c >= 0.85 and prob >= 0.9:
            return "high"
        if f1c >= 0.60 and prob >= 0.8:
            return "medium"
        return "low"

    out = pd.DataFrame({
        "cluster_id": allrows.cluster_id.values,
        "designation": allrows.designation.values,
        "ra": allrows.ra.values,
        "dec": allrows.dec.values,
        "tier": allrows.tier.values,
        "varwise_vartype": allrows.vartype.values,
        "corrected_class": pred,
        "corrected_prob": np.round(pmax, 4),
        "class_cv_f1": [round(per_f1.get(c, 0.0), 3) for c in pred],
        "simbad_type": allrows.simbad_type.values,
        "independent_label": allrows.truth.values,
        "validated": allrows.truth.notna().values,
        "w1mag": allrows.w1mag.values,
    })
    out["reliability"] = [
        reliability(c, p, v) for c, p, v
        in zip(out.corrected_class, out.corrected_prob, out.validated)
    ]
    for i, c in enumerate(model.classes_):
        out[f"p_{c}"] = np.round(proba[:, i], 4)

    emit("\n  Per-class cross-validated reliability of the model itself:")
    emit(f"  {'class':<8}{'train n':>10}{'CV F1':>9}{'verdict':>12}")
    for c in classes:
        v = ("reliable" if per_f1[c] >= 0.85
             else "marginal" if per_f1[c] >= 0.60 else "UNRELIABLE")
        emit(f"  {c:<8}{n_train.get(c, 0):>10,}{per_f1[c]:>9.3f}{v:>12}")

    OUT_TABLE.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_TABLE, index=False)

    emit(f"\n  Corrected-class distribution ({len(out):,} objects):")
    emit(f"  {'corrected':<10}{'n':>9}{'share':>9}{'median prob':>13}"
         f"{'median W1':>11}")
    for c, n_ in out.corrected_class.value_counts().items():
        s = out[out.corrected_class == c]
        emit(f"  {c:<10}{n_:>9,}{n_/len(out):>8.1%}"
             f"{s.corrected_prob.median():>13.3f}{s.w1mag.median():>11.2f}")

    emit(f"\n  Original VarWISE assignment vs corrected class:")
    ct = pd.crosstab(out.varwise_vartype, out.corrected_class)
    emit("            " + "".join(f"{c:>10}" for c in ct.columns))
    for idx in ct.index:
        emit(f"  {idx:<10}" + "".join(f"{v:>10,}" for v in ct.loc[idx]))

    # ---------- 3. sanity of the extrapolated population ----------
    emit("\n" + "=" * 84)
    emit("SANITY OF THE EXTRAPOLATED (UNVALIDATED) PREDICTIONS")
    emit("=" * 84)
    emit(f"\n  {'group':<26}{'n':>9}{'med prob':>10}{'lpv %':>8}{'agn %':>8}"
         f"{'yso %':>8}{'cv %':>7}")
    for label, m in [("validated (has SIMBAD)", out.validated),
                     ("extrapolated", ~out.validated)]:
        s = out[m]
        d = s.corrected_class.value_counts(normalize=True)
        emit(f"  {label:<26}{len(s):>9,}{s.corrected_prob.median():>10.3f}"
             f"{100*d.get('lpv',0):>7.1f}%{100*d.get('agn',0):>7.1f}%"
             f"{100*d.get('yso',0):>7.1f}%{100*d.get('cv',0):>6.1f}%")

    emit("\n  Photometric coherence: do predicted classes occupy the right colours?")
    feats = build_features(allrows)
    emit(f"  {'predicted':<10}{'n':>9}{'med W1-W2':>12}{'med W1':>9}"
         f"{'med W1amp':>11}")
    for c in sorted(out.corrected_class.unique()):
        m = (out.corrected_class == c).values
        emit(f"  {c:<10}{int(m.sum()):>9,}{feats.loc[m,'w1_w2'].median():>12.3f}"
             f"{feats.loc[m,'w1mag'].median():>9.2f}"
             f"{feats.loc[m,'w1_amp'].median():>11.3f}")
    emit("\n  Reference loci from the labelled set (SIMBAD truth):")
    lf = build_features(lab)
    for c in classes:
        m = (lab.truth == c).values
        emit(f"  {c:<10}{int(m.sum()):>9,}{lf.loc[m,'w1_w2'].median():>12.3f}"
             f"{lf.loc[m,'w1mag'].median():>9.2f}"
             f"{lf.loc[m,'w1_amp'].median():>11.3f}")

    emit("\n" + "=" * 84)
    emit("RELIABILITY BREAKDOWN")
    emit("=" * 84)
    emit(f"\n  {'tier':<12}{'n':>9}{'share':>9}  meaning")
    meaning = {
        "validated": "independent SIMBAD label agrees or exists",
        "high": "class CV F1 >= 0.85 and prob >= 0.9",
        "medium": "class CV F1 >= 0.60 and prob >= 0.8",
        "low": "weak class model or low probability - do not use",
    }
    for t in ("validated", "high", "medium", "low"):
        s = out[out.reliability == t]
        emit(f"  {t:<12}{len(s):>9,}{len(s)/len(out):>8.1%}  {meaning[t]}")

    emit(f"\n  Reliability by corrected class:")
    emit(f"  {'class':<8}" + "".join(f"{t:>12}" for t in
                                     ("validated", "high", "medium", "low")))
    for c in sorted(out.corrected_class.unique()):
        s = out[out.corrected_class == c]
        emit(f"  {c:<8}" + "".join(
            f"{int((s.reliability == t).sum()):>12,}"
            for t in ("validated", "high", "medium", "low")))

    emit("\n" + "=" * 84)
    emit("HOW TO USE THIS TABLE - AND WHAT NOT TO USE IT FOR")
    emit("=" * 84)
    emit(f"""
  {OUT_TABLE.name} is keyed on cluster_id and carries:
    corrected_class     predicted true class
    corrected_prob      model probability for that class
    class_cv_f1         cross-validated F1 for that class (model quality)
    validated           True if an independent SIMBAD label existed
    reliability         validated / high / medium / low
    p_<class>           full probability vector

  USE the validated and high tiers. They cover the lpv / agn / yso
  corrections, which are the bulk of the rule's failures and are supported
  by thousands of labelled examples each.

  DO NOT use the low tier for population statistics. Two specific warnings:

    - `ecl` is predicted for a large share of the extrapolated population but
      was trained on only a few hundred examples and scores poorly in
      cross-validation. Those predictions are hypotheses, not corrections.
    - `sn` and `cep` have too few labelled examples in this population to
      support any per-object claim.

  Aggregate accuracy (0.966) is carried by the majority classes. Macro F1 is
  far lower (0.40-0.63 depending on magnitude bin), and that gap is the
  honest measure of how unevenly this model performs across classes.
""")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_TABLE}")
    print(f"Wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
