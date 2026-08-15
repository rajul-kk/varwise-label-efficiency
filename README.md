# Label-efficient classification of NEOWISE infrared variables

How many labels does it actually take to classify infrared variable objects?

This repository runs an active-learning label-efficiency study on the
**VarWISE** catalog (Paz et al. 2026, ApJS 284:41,
[arXiv:2605.19059](https://arxiv.org/abs/2605.19059)) — a catalog of
infrared-variable objects built from the full NEOWISE single-exposure archive.

**Headline: 86% of the labels are unnecessary.** Choosing which objects to
label, rather than sampling them at random, matches random sampling's
914-label performance using 127 labels — and the benefit is almost entirely
confined to the *rare* variability classes.

A second result fell out of building the ground truth and is reported in its
own right: an **independent validation of VarWISE's published classifications**
against SIMBAD. It finds the XGBoost classifier holds up well (macro-F1 0.879
on labels it never saw), while the separate rule-based CV/SN transient
assignment over-predicts by 38× and 96×.

📊 **[Full results and figures → RESULTS.md](RESULTS.md)**
🔢 **[Generated tables → RESULTS_tables.md](RESULTS_tables.md)**

---

## The question

Labeling astronomical objects is expensive — it costs literature
cross-matching, and often spectroscopic follow-up. Active learning asks the
classifier which objects it would most benefit from having labeled, instead of
labeling a random sample.

VarWISE trained its XGBoost classifier on a fixed, pre-curated set of 910,697
labeled objects drawn from Gaia (Rimoldini et al. 2023) and ZTF (Chen et al.
2020). Nothing in the paper uses active learning. Two questions follow:

1. **Label efficiency** — how many labels are actually needed?
2. **Rare classes** — VarWISE's classes span 50.5% (eclipsing binaries) down
   to 0.14% (cataclysmic variables). Does active learning help the rare
   classes disproportionately? That is a more specific and more useful finding
   than an aggregate accuracy number.

---

## Results at a glance

Track B (independent SIMBAD labels), XGBoost — VarWISE's own booster family:

| | macro F1 | labels |
|---|---|---|
| Full supervision | 0.935 | 120,000 |
| **Margin sampling** | **0.913** | **914** |
| Random sampling | 0.808 | 914 |

- **86% label saving.** Margin sampling matches random's 914-label score with
  127 labels; 85.7% on LightGBM, 86.1% on XGBoost. Reaches 97.7% of
  full-supervised macro-F1 using 0.76% of the labels.
- **The saving is concentrated in the rare classes.** At equal budget, active
  learning gains **+0.38 F1 on `cep`** (0.92% of objects) and **+0.27 on `cv`**
  (0.14%), versus **+0.01 on `ecl`** (50%) and **`lpv`** (29%).
- **It is not merely rebalancing.** Against a random draw with *identical
  per-class counts* — so class mix is held fixed and only the choice of
  examples differs — the gain survives at +0.294 (`cep`) and +0.160 (`cv`).
  **Spearman ρ(prevalence, gain) = −0.857**: the rarer the class, the more the
  acquisition function contributes.
- **Hard per-class quotas underperform random sampling** on both estimators —
  a negative result reproducing the same observation from the Chandra project.
- **Distillation targets understate the benefit**: 72.4% saving against
  VarWISE's own predictions versus 85.7% against real labels, because VarWISE
  over-predicts rare classes and so looks more balanced than reality.

### Independent validation of VarWISE

Scoring the published `vartype` against 220,419 objects carrying an
independent SIMBAD type — labels the catalog was not trained on.

**Two mechanisms produce `vartype` and must be scored separately.** The six
periodic/persistent classes come from the XGBoost classifier and carry a
`confidence` value. `cv` and `sn` come from a rule — VARnet flags a transient,
then a 2″ crossmatch against Gaia DR3 galaxy/QSO catalogs assigns SN on a
match and CV otherwise — and carry **no confidence** (95.2% of `cv` and 100%
of `sn` rows are confidence-null). The paper's reported macro-F1 of 0.95
covers the classifier only.

**(a) The classifier validates well** (n = 205,374):

| class | precision | recall | F1 |
|---|---|---|---|
| `ecl` | 0.996 | 0.985 | **0.990** |
| `lpv` | 0.942 | 0.989 | **0.965** |
| `rr` | 0.934 | 0.963 | **0.948** |
| `agn` | 0.844 | 0.995 | **0.913** |
| `cep` | 0.803 | 0.935 | **0.864** |
| `yso` | 0.940 | **0.436** | **0.595** |
| **macro avg** | 0.910 | 0.884 | **0.879** |

Macro-F1 **0.879** against labels it never saw, versus 0.95 on its own
validation split — a reasonable degradation. **The one real weakness is YSO
recall (0.436)**: when it says YSO it is usually right, but it misses most.

**(b) The rule-based transient assignment fails badly:**

| class | SIMBAD n | VarWISE n | over-prediction | precision |
|---|---|---|---|---|
| `cv` | 301 | 11,576 | **38.5×** | 0.019 |
| `sn` | 35 | 3,379 | **96.5×** | 0.002 |

Driven by **8,291 long-period variables assigned `cv`** and **3,275 AGN
assigned `sn`** — exactly what the rule's definition predicts, since bright
Miras look transient but aren't extragalactic, while AGN are. The `cv` false
positives are ~6 mag brighter and 4× lower amplitude than real CVs, so they
are ordinary LPVs rather than borderline cases.

⚠️ **Selection bias materially qualifies the `cv` number.** Only 37.3% of `cv`
predictions carry a SIMBAD type, and those that do are ~3.8 mag brighter than
those that don't. Real CVs are faint; contaminants are bright. So 0.019 is
**not** a catalog-wide precision. See [RESULTS.md §6](RESULTS.md).

---

## Reproducing

Requires Python 3.10+, `pyvo astropy pandas numpy scikit-learn lightgbm
xgboost matplotlib`.

```bash
python scripts/download_varwise.py      # VarWISE Pure Catalog from IRSA TAP -> parquet
python scripts/build_dataset.py         # feature matrices + both label tracks

python scripts/run_experiment.py --track b                      # main study
python scripts/run_experiment.py --track b --estimator xgboost --tag _xgb
python scripts/run_experiment.py --track a                      # distillation track

python scripts/analyze.py --track b --tag _xgb                  # label-savings tables
python scripts/plot_curves.py --track b --tag _xgb              # figures
python scripts/reference_baselines.py --track b                 # reference variants
python scripts/diagnose_gap.py --track b --seed 0               # rebalancing vs informativeness
python scripts/validate_varwise.py                              # VarWISE vs SIMBAD
python scripts/summarize.py > RESULTS_tables.md                 # all headline tables
python scripts/factcheck.py                                     # verify every quoted number
```

`scripts/factcheck.py` recomputes every quantitative claim in the writeup from
source and fails loudly on any mismatch — **93 checks, 0 failures**.

The download takes ~5 minutes (457,080 rows, RA-sliced async TAP queries). The
main experiment takes ~25 minutes per track.

---

## Data and design

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
| Interpretation | efficiency for distillation | efficiency for the science task |

SIMBAD does not separate Algol-type from W UMa-type eclipsing binaries, so
those two VarWISE classes collapse into one in Track B. Ambiguous SIMBAD types
(`C*`, `SB*`, `Variable*`, …) are dropped rather than force-mapped.

### Features

VarWISE's own 31 classifier features derive from raw NEOWISE light curves
(Fourier coefficients, Stetson indices, χ² statistics) which live in the
separate Associations table, **not** in the published catalog. This study uses
the 28 reproducible from catalog columns — colors, amplitudes, periodicity,
variability significance. **A close approximation, not a reproduction.**

That the full-supervised weighted F1 (0.979 XGBoost) sits close to VarWISE's
reported 0.95 suggests the approximation is adequate.

Two exclusions, both deliberate:

- **`known_extragalactic`** is dropped: an external cross-match flag built
  from the same kind of literature catalogs that supply the SIMBAD labels, it
  nearly determines `agn`/`sn` membership on its own and would flatter exactly
  the rare classes the study is about. `--keep-leaky` restores it.
- **`sn`** (35 SIMBAD-confirmed objects) scores F1 = 0.000 even against the
  full 120k-label reference. Not learnable from catalog features at this
  sample size.

### Method

`common/active_learning.py` is reused from the Chandra/eROSITA project
(`chandra-toolkit`) — a pool-based loop agnostic to estimator and domain.

| Strategy | Idea |
|---|---|
| `random` | control |
| `uncertainty` | least confidence, `1 - max P` |
| `margin` | smallest gap between top-two class probabilities |
| `class_balanced` | uncertainty ranked *within* each predicted class |
| `quota` | hard-reserved batch slots per class |
| `prototype` | feature-space proximity to the rarest labeled class |

One adaptation was required: `prototype_distance_score` uses `cdist`, which
has no missing-value handling, and astronomical photometry is sparsely missing
(~13% of rows lack 2MASS *JHK*, ~63% lack a usable parallax). Left unpatched
it returns an all-NaN score vector and silently degrades to arbitrary
ordering. It now centres on nan-statistics.

5 seeds (3 for XGBoost), 30% stratified test split, pool capped at 120,000,
seeded with 2 examples/class, 60 rounds × 15 queries (14 → ~914 labels).

---

## Honest accounting

**Two claims were retracted during the study.**

1. **"Active learning beats full supervision"** (macro-F1 0.91 vs 0.77) was a
   **LightGBM artifact** — at these hyperparameters LightGBM collapses on the
   rare classes under the natural distribution (`cv` F1 = 0.230 with all
   120,000 labels), while XGBoost reaches 0.935 on the same data. Caught by an
   estimator robustness check; corrected in [RESULTS.md §3](RESULTS.md). The
   label-efficiency and rare-class results are within-estimator comparisons
   and reproduce on both boosters, so they are unaffected.

2. **"VarWISE scores macro-F1 0.632 against SIMBAD, versus its reported
   0.95"** conflated two different mechanisms — an XGBoost classifier and a
   separate rule-based transient assignment — and **overstated the problem**.
   Caught by checking why 38,015 catalog rows have a null `confidence`. The
   corrected, mechanism-split analysis is in
   [RESULTS.md §6](RESULTS.md): the classifier scores 0.879, and the failure
   is localised to the CV/SN rule.

**Prior art.** Richards et al. 2011
([arXiv:1106.2832](https://arxiv.org/abs/1106.2832)) already applied active
learning to photometric variable-star classification on Hipparcos/OGLE.
Active learning for variable stars is **not** new. See
[RESULTS.md §7](RESULTS.md) for a finding-by-finding novelty classification.

**Other limitations**: single archive; pool capped at 120,000 of ~154,000
available Track B rows; XGBoost run used 3 seeds rather than 5; SIMBAD
coverage is 48% of the Pure Catalog and biased toward bright, well-studied
objects.

---

## Related work

- **Paz et al. 2026**, VarWISE — the catalog under study
  ([arXiv:2605.19059](https://arxiv.org/abs/2605.19059))
- **Richards et al. 2011** — active learning for photometric variable stars;
  direct methodological ancestor
  ([arXiv:1106.2832](https://arxiv.org/abs/1106.2832))
- **El-Kholy & Hayman 2026**, PASP 138(5) — margin sampling vs random querying
  on a public astronomical catalog under extreme class imbalance; experimental
  template ([arXiv:2602.23666](https://arxiv.org/abs/2602.23666))
- **Liu et al. 2025**, RB-C1000, A&A 693 A105 — active + semi-supervised
  learning for ZTF real/bogus; the shape of the target result
  ([arXiv:2412.02409](https://arxiv.org/abs/2412.02409))

## Kill condition

Pre-registered: if active learning yields less than ~15–20% label savings over
random sampling, report it as a mixed/negative result rather than overselling
it. **Not triggered** — the saving is ~86%, and the rare-class effect it was
meant to protect is the strongest part of the result.
