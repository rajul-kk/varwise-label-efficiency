# Label-efficiency of active learning on the VarWISE infrared variable catalog

An active-learning label-efficiency study on NEOWISE infrared variability
classification, using the VarWISE catalog (Paz et al. 2026, ApJS 284:41,
[arXiv:2605.19059](https://arxiv.org/abs/2605.19059)) as the labeled archive.

Two questions:

1. **Label efficiency.** How many labels does uncertainty/margin/quota-based
   active learning need to match a randomly-sampled baseline on VarWISE's
   variability-class taxonomy — overall, and per class?
2. **Rare classes.** VarWISE's classes span 50.5% (eclipsing binaries) down to
   0.14% (cataclysmic variables). Does active learning help the rare,
   undersampled classes disproportionately? That is a more specific and more
   useful finding than an aggregate accuracy number.

A third result fell out of building the ground truth and is reported in its
own right: an **independent validation of VarWISE's published classifications
against SIMBAD** (see below).

## Relationship to prior work

- **Richards et al. 2011** ([arXiv:1106.2832](https://arxiv.org/abs/1106.2832))
  is the direct methodological ancestor: active learning for photometric
  variable-star classification on Hipparcos/OGLE. Active learning for variable
  stars is *not* new. What is unclaimed here is the archive (NEOWISE/VarWISE)
  and the per-class rare-class label-efficiency question.
- **El-Kholy & Hayman 2026** ([arXiv:2602.23666](https://arxiv.org/abs/2602.23666))
  supplies the experimental template: margin sampling vs random querying on a
  public astronomical catalog under extreme class imbalance, gradient-boosted
  tree baseline.
- **Liu et al. 2025** ([arXiv:2412.02409](https://arxiv.org/abs/2412.02409),
  RB-C1000) is the shape of the target result: a label-efficiency curve with a
  quantified label saving.

## Data

VarWISE **Pure Catalog** (457,080 objects, the highest-confidence tier),
pulled from IRSA TAP (`varwisepure`) in RA slices:

```
python scripts/download_varwise.py     # -> data/raw/varwise_pure.parquet
python scripts/build_dataset.py        # -> data/track_a_vartype.parquet, track_b_simbad.parquet
```

### Two label tracks

The catalog's `vartype` column is **VarWISE's own XGBoost prediction**, not
ground truth. Training on it measures label efficiency for *distilling their
classifier*. 229,365 Pure Catalog objects also carry a `simbad_type` — an
independent, literature-curated label. Both are run:

| | Track A | Track B |
|---|---|---|
| Target | `vartype` (VarWISE prediction) | `simbad_type` (SIMBAD literature) |
| Rows | 456,763 | 220,471 |
| Classes | 9 | 8 (`ea`+`ew` merged to `ecl`) |
| Interpretation | label efficiency for distillation | label efficiency for the science task |

SIMBAD does not separate Algol-type from W UMa-type eclipsing binaries, so
those two VarWISE classes collapse into a single `ecl` class in Track B.
Ambiguous SIMBAD types (`C*`, `SB*`, `Variable*`, …) are dropped rather than
force-mapped.

### Features

VarWISE's own 31 classifier features are derived from raw NEOWISE light curves
(Fourier coefficients, Stetson indices, χ² statistics) which live in the
separate Associations table, **not** in the published catalog. This study
reproduces the reproducible subset — colors, amplitudes, periodicity,
variability significance — 28 features in total. This is a close approximation,
not an exact reproduction, and the gap is the light-curve morphology and
flux-statistic blocks.

`known_extragalactic` is dropped by default: it is an external cross-match
flag built from the same kind of literature catalogs that supply the SIMBAD
labels, and it nearly determines `agn`/`sn` membership on its own, which would
flatter precisely the rare classes the study is about. `--keep-leaky` restores
it for a sensitivity check.

`sn` (35 SIMBAD-confirmed objects, 0.016%) scores F1 = 0.000 even against the
full 60k-label reference and is excluded from the curves as not learnable from
catalog-level features at this sample size.

## Method

`common/active_learning.py` is reused from the Chandra/eROSITA project
(`chandra-toolkit`) — a pool-based loop that is agnostic to estimator and
domain. Strategies compared:

| Strategy | Idea |
|---|---|
| `random` | control |
| `uncertainty` | least confidence, `1 - max P` |
| `margin` | smallest gap between top-two class probabilities |
| `class_balanced` | uncertainty ranked *within* each predicted class |
| `quota` | hard-reserved batch slots per class |
| `prototype` | feature-space proximity to the rarest labeled class, bypassing `predict_proba` |

One adaptation was required for this archive: `prototype_distance_score` uses
`cdist`, which has no missing-value handling, and astronomical photometry is
sparsely missing (~13% of VarWISE rows lack 2MASS *JHK*, ~63% lack a usable
parallax). Left unpatched it returns an all-NaN score vector and silently
degrades to an arbitrary ordering. It now centres on nan-statistics and treats
a missing feature as population-mean.

Estimator is LightGBM (handles NaN natively). 5 seeds, 30% stratified test
split, pool capped at 120,000, seeded with 2 examples per class, 60 rounds of
15 queries (14 → 914 labels).

```
python scripts/run_experiment.py --track b
python scripts/analyze.py --track b
python scripts/plot_curves.py --track b
```

## Independent validation of VarWISE against SIMBAD

Scoring the published `vartype` against the 220,419 objects carrying an
independent SIMBAD type (`python scripts/validate_varwise.py`):

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| ecl | 0.995 | 0.984 | **0.989** | 111,241 |
| rr | 0.934 | 0.963 | **0.948** | 12,106 |
| lpv | 0.942 | 0.861 | **0.899** | 64,132 |
| cep | 0.801 | 0.924 | **0.858** | 2,020 |
| agn | 0.842 | 0.802 | **0.822** | 18,617 |
| yso | 0.939 | 0.342 | **0.501** | 11,967 |
| cv | 0.019 | 0.738 | **0.037** | 301 |
| sn | 0.002 | 0.229 | **0.005** | 35 |
| **macro avg** | 0.684 | 0.730 | **0.632** | 220,419 |
| **weighted avg** | 0.957 | 0.896 | **0.918** | 220,419 |

The common classes hold up well. The rare classes have severe **precision**
failures, dominated by two systematic confusions:

- **8,291 SIMBAD long-period variables are classified `cv`** — the `cv` class
  is ~98% contaminated on this subset.
- **3,275 SIMBAD AGN are classified `sn`.**
- **YSOs scatter** into `agn` (19.7%), `cv` (21.2%), and `lpv` (23.8%), with
  only 34.2% recovered.

**Caveats, which matter.** This is not a like-for-like comparison with the
paper's reported macro-F1 of 0.95, which is measured against a held-out split
of its own Gaia/ZTF-derived training labels. SIMBAD coverage is 48% of the
Pure Catalog and is biased toward bright, well-studied objects. Precision here
is computed only over SIMBAD-typed objects, so it is not the catalog-wide
precision. Some LPV/CV confusion may be astrophysically genuine (symbiotic and
interacting systems are labeled inconsistently across surveys). The size and
systematic direction of the LPV→`cv` and AGN→`sn` confusions are nonetheless
large enough to warrant caution when using the rare-class VarWISE labels.

## Results

See `RESULTS.md` (generated after the experiment runs).

## Kill condition

Pre-registered: if active learning yields less than ~15–20% label savings over
random sampling, that is reported as a mixed/negative result rather than
oversold. The secondary question stands on its own — even with modest
aggregate savings, whether active learning meaningfully helps the *rare*
classes is the finding that would matter.
